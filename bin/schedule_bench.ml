(* Matmul schedule benchmark (docs/proposals/schedule-ir-optops.md, phases S1-S3; gh-ocannl-412
   acceptance): times the naive 1x1-launch kernel against three hand-written schedules on the
   configured backend --

   - parallel: one thread per output element (Split i / Split j into Grid x Workgroup; the S1 shape,
   Boehm kernel 1 equivalent); - smem: + Split k, operands staged through workgroup-shared tiles,
   the output accumulator privatized to a per-thread scalar (S2 + Privatize, Boehm kernel 3); -
   regtile: + second-level splits with materialized-unroll TM x TN register tiles accumulating into
   a privatized per-thread tile (S3 + Privatize, Boehm kernel 4/5 shape).

   On the C backends the shared schedules are rejected and the CPU variants run instead: cpupack (S4
   cache tiling + operand packing, all-Serial), tensorize (whole-triple register-tiled Tile_mma,
   gh-ocannl-469), packmma (Tile_mma composed with cache tiling + packing, all-Serial GEBP),
   packmma_par (the same composition with pool-parallel Grid row blocks and per-chunk privatized A~
   tiles; the B~ pack at k_o re-enters the parallel construct once per k-block), and the three
   grid-outermost flavors — one dispatch spanning the whole GEBP triple (gh-ocannl-473 /
   gh-ocannl-475): pm_hoist (hoisted constant-pool B~ panel, A read in place; autotune's sk_grid &&
   sk_hoist seed), pm_mixed (hoisted B~ plus an in-kernel per-chunk A~ pack; the sk_pack_rest seed),
   and pm_bpk (both operands packed in-kernel, each chunk re-packing its own B~ panel — needs the
   panel under the per-chunk privatization cap, config cc_grid_private_bytes_cap, or it silently
   declines to serial: check with --ocannl_schedule_log_declines=true).

   Usage: dune exec bin/schedule_bench.exe -- [n] [repeats] [m] [k] (defaults 256, 20, n, n; all
   dims must be multiples of 64; the output is m x n, the reduction depth k — deep-K gradient-GEMM
   geometries are m,n << k). Run with OCANNL_BACKEND=metal (or cuda); the C backends reject the
   shared schedules. Timing includes kernel executions and one device-to-host transfer per variant
   (runs queue on the stream; get_values synchronizes). *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module LL = Ir.Low_level
module Sched = Ir.Schedule
module Asgns = Ir.Assignments
module Numerics = Ir.Numerics

(* CUDA's uniform-f32 MMA arm is gated on tf32 (the Numerics policy): without it the [Tile_mma]s
   of the mma_pd* variants render the lane-0 scalar fallback and the labels would time a kernel
   that never tensorizes ("timed is not tensorized", docs/agent-notes.md). Metal's f32
   [simdgroup_matrix] path has no such gate and is unaffected. This bench asserts no parity, so
   the tf32 rounding is free; cross-check a new backend's emission with
   --ocannl_schedule_log_declines=true before trusting the labels. *)
let () = Numerics.set_policy { (Numerics.get ()) with tf32_matmuls = true }
let p = Stdio.printf

let nest_paths (llc : LL.t) : Ir.Indexing.symbol list list =
  let strip stmts = List.filter stmts ~f:(function LL.Noop | LL.Comment _ -> false | _ -> true) in
  let rec path (llc : LL.t) : Ir.Indexing.symbol list =
    match llc with
    | LL.For_loop { index; body; _ } ->
        index :: (match strip (LL.flat_lines [ body ]) with [ single ] -> path single | _ -> [])
    | LL.If { body; _ } -> path body
    | _ -> []
  in
  List.filter_map (LL.flat_lines [ llc ]) ~f:(fun stmt ->
      match path stmt with [] -> None | p -> Some p)

let named name (comp : Asgns.comp) : Asgns.comp =
  { comp with asgns = Asgns.Block_comment (name, comp.asgns) }

let () =
  (* Positional integer args only — --ocannl_* config flags share the argv. *)
  let pos_args =
    Array.to_list (Sys.get_argv ())
    |> List.tl_exn
    |> List.filter ~f:(fun s -> not (String.is_prefix s ~prefix:"-"))
  in
  let arg i default =
    match List.nth pos_args i with Some s -> Int.of_string s | None -> default
  in
  let n = arg 0 256 in
  let repeats = arg 1 20 in
  let m = arg 2 n in
  let k = arg 3 n in
  (* The naive 1x1-launch kernel is minutes per run at large sizes (a single GPU thread); cap its
     repeats separately so the scheduled variants can be timed at scale (arg 5, default = repeats;
     0 skips the naive leg entirely, including its warmup run — speedups then print as nan). *)
  let naive_repeats = arg 4 repeats in
  assert (n % 64 = 0 && m % 64 = 0 && k % 64 = 0);
  let mav = Array.init (m * k) ~f:(fun i -> Float.of_int (i % 13) *. 0.25) in
  let mbv = Array.init (k * n) ~f:(fun i -> Float.of_int (i % 17) -. 8.) in
  let ma = TDSL.ndarray mav ~label:[ "ma" ] ~input_dims:[ k ] ~output_dims:[ m ] () in
  let mb = TDSL.ndarray mbv ~label:[ "mb" ] ~input_dims:[ n ] ~output_dims:[ k ] () in
  let flops = 2.0 *. Float.of_int m *. Float.of_int n *. Float.of_int k in

  let accum_syms opt =
    let paths = nest_paths opt.LL.llc in
    match List.find_exn paths ~f:(fun p -> List.length p = 3) with
    | [ i; j; k ] -> (i, j, k)
    | _ -> assert false
  in

  (* One thread per output element. *)
  let parallel_schedule ~mc opt =
    let i, j, k = accum_syms opt in
    ignore k;
    let ez, zsyms = Sched.expand_zero ~tn:mc in
    let zi, zj = match zsyms with [ zi; zj ] -> (zi, zj) | _ -> assert false in
    let sp_zi, _, _ = Sched.split ~axis:zi ~factor:16 ~outer:LL.Grid ~inner:LL.Workgroup in
    let sp_zj, _, _ = Sched.split ~axis:zj ~factor:16 ~outer:LL.Grid ~inner:LL.Workgroup in
    let sp_i, _, _ = Sched.split ~axis:i ~factor:16 ~outer:LL.Grid ~inner:LL.Workgroup in
    let sp_j, _, _ = Sched.split ~axis:j ~factor:16 ~outer:LL.Grid ~inner:LL.Workgroup in
    [ ez; sp_zi; sp_zj; sp_i; sp_j ]
  in

  (* + shared-memory operand tiles (32x32x8, 32x32 threads is over most limits: use 16x16x8). *)
  let smem_schedule ~mc opt =
    let bm, bn, bk = (16, 16, 8) in
    let i, j, k = accum_syms opt in
    let ez, zsyms = Sched.expand_zero ~tn:mc in
    let zi, zj = match zsyms with [ zi; zj ] -> (zi, zj) | _ -> assert false in
    let sp_zi, _, _ = Sched.split ~axis:zi ~factor:bm ~outer:LL.Grid ~inner:LL.Workgroup in
    let sp_zj, _, _ = Sched.split ~axis:zj ~factor:bn ~outer:LL.Grid ~inner:LL.Workgroup in
    let sp_i, _, i_i = Sched.split ~axis:i ~factor:bm ~outer:LL.Grid ~inner:LL.Workgroup in
    let sp_j, _, j_i = Sched.split ~axis:j ~factor:bn ~outer:LL.Grid ~inner:LL.Workgroup in
    let sp_k, k_o, k_i = Sched.split ~axis:k ~factor:bk ~outer:LL.Serial ~inner:LL.Serial in
    [
      ez;
      sp_zi;
      sp_zj;
      sp_i;
      sp_j;
      sp_k;
      Sched.Stage
        {
          source = ma.Tensor.value;
          tile_loops = [ i_i; k_i ];
          shared = true;
          cooperative = None;
          hoisted = false;
          swizzle = None;
          pad_stride = None;
          pipeline_depth = 1;
        };
      Sched.Stage
        {
          source = mb.Tensor.value;
          tile_loops = [ k_i; j_i ];
          shared = true;
          cooperative = None;
          hoisted = false;
          swizzle = None;
          pad_stride = None;
          pipeline_depth = 1;
        };
      Sched.Privatize { target = mc; over = k_o };
    ]
  in

  (* CPU cache tiling + operand packing (all-Serial; the S4 shape, Boehm's packed CPU kernel). *)
  let cpupack_schedule ~mc opt =
    let bm, bn, bk = (64, 64, 16) in
    let i, j, k = accum_syms opt in
    let sp_i, _, i_i = Sched.split ~axis:i ~factor:bm ~outer:LL.Serial ~inner:LL.Serial in
    let sp_j, j_o, j_i = Sched.split ~axis:j ~factor:bn ~outer:LL.Serial ~inner:LL.Serial in
    let sp_k, k_o, k_i = Sched.split ~axis:k ~factor:bk ~outer:LL.Serial ~inner:LL.Serial in
    let sink sym below = List.map below ~f:(fun inner -> Sched.Swap { outer = sym; inner }) in
    [ sp_i; sp_j; sp_k ]
    @ sink i_i [ j_o; j_i; k_o; k_i ]
    @ sink j_i [ k_o; k_i; i_i ]
    @ [
        Sched.Stage
          {
            source = ma.Tensor.value;
            tile_loops = [ i_i; k_i ];
            shared = false;
            cooperative = None;
            hoisted = false;
            swizzle = None;
            pad_stride = None;
            pipeline_depth = 1;
          };
        Sched.Stage
          {
            source = mb.Tensor.value;
            tile_loops = [ k_i; j_i ];
            shared = false;
            cooperative = None;
            hoisted = false;
            swizzle = None;
            pad_stride = None;
            pipeline_depth = 1;
          };
        Sched.Privatize { target = mc; over = k_o };
      ]
  in

  (* + TM x TN register tiles via materialized unroll (64x64 block, 8x8 per thread). *)
  let regtile_schedule ~mc opt =
    let bm, bn, bk, tm, tn = (64, 64, 8, 8, 8) in
    let i, j, k = accum_syms opt in
    let ez, zsyms = Sched.expand_zero ~tn:mc in
    let zi, zj = match zsyms with [ zi; zj ] -> (zi, zj) | _ -> assert false in
    let sp_zi, _, zi_i = Sched.split ~axis:zi ~factor:bm ~outer:LL.Grid ~inner:LL.Serial in
    let sp_zi2, _, _ = Sched.split ~axis:zi_i ~factor:tm ~outer:LL.Workgroup ~inner:LL.Serial in
    let sp_zj, _, zj_i = Sched.split ~axis:zj ~factor:bn ~outer:LL.Grid ~inner:LL.Serial in
    let sp_zj2, _, _ = Sched.split ~axis:zj_i ~factor:tn ~outer:LL.Workgroup ~inner:LL.Serial in
    let sp_i, _, i_i = Sched.split ~axis:i ~factor:bm ~outer:LL.Grid ~inner:LL.Serial in
    let sp_i2, i_w, i_t = Sched.split ~axis:i_i ~factor:tm ~outer:LL.Workgroup ~inner:LL.Serial in
    let sp_j, j_o, j_i = Sched.split ~axis:j ~factor:bn ~outer:LL.Grid ~inner:LL.Serial in
    let sp_j2, j_w, j_t = Sched.split ~axis:j_i ~factor:tn ~outer:LL.Workgroup ~inner:LL.Serial in
    let sp_k, k_o, k_i = Sched.split ~axis:k ~factor:bk ~outer:LL.Serial ~inner:LL.Serial in
    let sink sym below = List.map below ~f:(fun inner -> Sched.Swap { outer = sym; inner }) in
    [ ez; sp_zi; sp_zi2; sp_zj; sp_zj2; sp_i; sp_i2; sp_j; sp_j2; sp_k ]
    @ sink i_t [ j_o; j_w; j_t; k_o; k_i ]
    @ sink j_t [ k_o; k_i ]
    @ [
        Sched.Stage
          {
            source = ma.Tensor.value;
            tile_loops = [ i_w; i_t; k_i ];
            shared = true;
            cooperative = None;
            hoisted = false;
            swizzle = None;
            pad_stride = None;
            pipeline_depth = 1;
          };
        Sched.Stage
          {
            source = mb.Tensor.value;
            tile_loops = [ k_i; j_w; j_t ];
            shared = true;
            cooperative = None;
            hoisted = false;
            swizzle = None;
            pad_stride = None;
            pipeline_depth = 1;
          };
        Sched.Privatize { target = mc; over = k_o };
        Sched.Unroll { axis = i_t; materialize = true };
        Sched.Unroll { axis = j_t; materialize = true };
      ]
  in

  (* Staged + tensorized simdgroup mma (the autotune staged-seed shape, gh-ocannl-487): both
     operand tiles cooperatively staged at the serial [k_o] anchor and the micro-kernel tensorized,
     with the software-pipelining depth as the knob — [pipeline_depth = 1] is the strictly phased
     form, [2] the double-buffered one, bitwise identical by the transform's invariant, so the
     paired timing difference is the prefetch overlap's alone. *)
  let mma_staged_schedule ~pipeline_depth ~mc opt =
    let bm, bn, bk, w = (32, 32, 32, 32) in
    let i, j, k = accum_syms opt in
    let ez, zsyms = Sched.expand_zero ~tn:mc in
    let zi, zj = match zsyms with [ zi; zj ] -> (zi, zj) | _ -> assert false in
    let sp_zi, _, _ = Sched.split ~axis:zi ~factor:bm ~outer:LL.Grid ~inner:LL.Serial in
    let sp_zj, _, _ = Sched.split ~axis:zj ~factor:w ~outer:LL.Grid ~inner:LL.Workgroup in
    let sp_i, _, i_i = Sched.split ~axis:i ~factor:bm ~outer:LL.Grid ~inner:LL.Serial in
    let sp_j, j_o, j_i = Sched.split ~axis:j ~factor:bn ~outer:LL.Grid ~inner:LL.Serial in
    let sp_k, k_o, k_i = Sched.split ~axis:k ~factor:bk ~outer:LL.Serial ~inner:LL.Serial in
    let sink sym below = List.map below ~f:(fun inner -> Sched.Swap { outer = sym; inner }) in
    let tz, _lane = Sched.tensorize ~i:i_i ~j:j_i ~k:k_i ~simd_width:w in
    let stage source tile_loops =
      Sched.Stage
        {
          source;
          tile_loops;
          shared = true;
          cooperative = Some w;
          hoisted = false;
          swizzle = None;
          pad_stride = None;
          pipeline_depth;
        }
    in
    [ ez; sp_zi; sp_zj; sp_i; sp_j; sp_k ]
    @ sink i_i [ j_o ] @ sink j_i [ k_o ] @ sink i_i [ k_o ]
    @ [ stage ma.Tensor.value [ i_i; k_i ]; stage mb.Tensor.value [ k_i; j_i ]; tz ]
  in

  (* Register-tiled Tile_mma micro-kernel (gh-ocannl-469): the whole i x j x k triple becomes one
     Tile_mma statement, which the C backends render tinyBLAS-style — the C-tile held in an RM x RN
     grid of vector registers across the entire k-loop, edges peeled. The zeroing must cover the
     lane slot (validate_parallel's coverage rule), so its column loop becomes the Workgroup axis
     and the lane width matches its extent (the lane loop renders serially on the C backends,
     executing the guarded statement once). *)
  let tensorize_schedule ~mc opt =
    let i, j, k = accum_syms opt in
    let ez, zsyms = Sched.expand_zero ~tn:mc in
    let zj = match zsyms with [ _; zj ] -> zj | _ -> assert false in
    let rz = Sched.Retype { axis = zj; ty = LL.Workgroup } in
    let tz, _lane = Sched.tensorize ~i ~j ~k ~simd_width:n in
    [ ez; rz; tz ]
  in

  (* Tile_mma composed with cache tiling + operand packing (the GEBP shape; the closing piece of
     gh-ocannl-469, autotune's [cpu_mma_pack_sketch_schedule]): pack the B panel [bk x n] at k_o
     (reused across all row blocks) and the A tile [bm x bk] at i_o, then tensorize the inner triple
     — the register-tiled micro-kernel streams the contiguous packed tiles. All-Serial with a unit
     lane, so the whole-node zeroing stays legal. *)
  let packmma_schedule ~mc:_ opt =
    let bm, bk = (64, 64) in
    let i, j, k = accum_syms opt in
    let sp_i, i_o, i_i = Sched.split ~axis:i ~factor:bm ~outer:LL.Serial ~inner:LL.Serial in
    let sp_k, k_o, k_i = Sched.split ~axis:k ~factor:bk ~outer:LL.Serial ~inner:LL.Serial in
    let sink sym below = List.map below ~f:(fun inner -> Sched.Swap { outer = sym; inner }) in
    let stage source tile_loops =
      Sched.Stage
        { source; tile_loops; shared = false; cooperative = None; hoisted = false; swizzle = None; pad_stride = None; pipeline_depth = 1 }
    in
    let tz, _lane = Sched.tensorize ~i:i_i ~j ~k:k_i ~simd_width:1 in
    [ sp_i; sp_k ] @ sink j [ k_o ] @ sink i_i [ k_o ] @ sink i_o [ k_o ]
    @ [ stage mb.Tensor.value [ k_i; j ]; stage ma.Tensor.value [ i_i; k_i ]; tz ]
  in

  (* The fully parallel packed GEMM (gh-ocannl-469 follow-up): the row-block loop is Grid-typed and
     pool-parallelizes — the per-row-block A~ tile is privatized to per-chunk block-scope storage by
     the renderer, the B~ panel packed at k_o is read-only inside the Grid body (behind a pointer
     alias under the blocks extension). The whole-node zeroing is no longer legal beside a
     hardware-annotated loop, so it expands with the same Grid row geometry. *)
  let packmma_par_schedule ~mc opt =
    let bm, bk = (64, 64) in
    let i, j, k = accum_syms opt in
    let ez, zsyms = Sched.expand_zero ~tn:mc in
    let zi = match zsyms with [ zi; _ ] -> zi | _ -> assert false in
    let sp_zi, _, _ = Sched.split ~axis:zi ~factor:bm ~outer:LL.Grid ~inner:LL.Serial in
    let sp_i, i_o, i_i = Sched.split ~axis:i ~factor:bm ~outer:LL.Grid ~inner:LL.Serial in
    let sp_k, k_o, k_i = Sched.split ~axis:k ~factor:bk ~outer:LL.Serial ~inner:LL.Serial in
    let sink sym below = List.map below ~f:(fun inner -> Sched.Swap { outer = sym; inner }) in
    let stage source tile_loops =
      Sched.Stage
        { source; tile_loops; shared = false; cooperative = None; hoisted = false; swizzle = None; pad_stride = None; pipeline_depth = 1 }
    in
    let tz, _lane = Sched.tensorize ~i:i_i ~j ~k:k_i ~simd_width:1 in
    [ ez; sp_zi; sp_i; sp_k ] @ sink j [ k_o ] @ sink i_i [ k_o ] @ sink i_o [ k_o ]
    @ [ stage mb.Tensor.value [ k_i; j ]; stage ma.Tensor.value [ i_i; k_i ]; tz ]
  in

  (* Grid-outermost flavors (one dispatch spanning the whole GEBP triple; gh-ocannl-473 /
     gh-ocannl-475): the row-block Grid loop stays outermost — no [sink i_o [k_o]] — so anything
     packed at [k_o] lands inside the Grid body. [pack_b = `Hoist] packs the (constant) B~ panel at
     link time into the constant pool; [`Chunk] packs it in-kernel, each chunk re-packing its own
     panel (redundant work, but no dispatch-per-k-block; the panel must fit the per-chunk
     privatization cap). [pack_a] toggles the in-kernel per-chunk A~ pack vs. reading A in place. *)
  let packmma_outer_schedule ~pack_a ~pack_b ~mc opt =
    let bm, bk = (64, 64) in
    let i, j, k = accum_syms opt in
    let ez, zsyms = Sched.expand_zero ~tn:mc in
    let zi = match zsyms with [ zi; _ ] -> zi | _ -> assert false in
    let sp_zi, _, _ = Sched.split ~axis:zi ~factor:bm ~outer:LL.Grid ~inner:LL.Serial in
    let sp_i, _, i_i = Sched.split ~axis:i ~factor:bm ~outer:LL.Grid ~inner:LL.Serial in
    let sp_k, k_o, k_i = Sched.split ~axis:k ~factor:bk ~outer:LL.Serial ~inner:LL.Serial in
    let sink sym below = List.map below ~f:(fun inner -> Sched.Swap { outer = sym; inner }) in
    let stage ~hoisted source tile_loops =
      Sched.Stage
        { source; tile_loops; shared = false; cooperative = None; hoisted; swizzle = None; pad_stride = None; pipeline_depth = 1 }
    in
    let stage_b =
      match pack_b with
      | `Hoist -> [ stage ~hoisted:true mb.Tensor.value [ k_i; j ] ]
      | `Chunk -> [ stage ~hoisted:false mb.Tensor.value [ k_i; j ] ]
    in
    let stage_a = if pack_a then [ stage ~hoisted:false ma.Tensor.value [ i_i; k_i ] ] else [] in
    let tz, _lane = Sched.tensorize ~i:i_i ~j ~k:k_i ~simd_width:1 in
    [ ez; sp_zi; sp_i; sp_k ] @ sink j [ k_o ] @ sink i_i [ k_o ] @ stage_b @ stage_a @ [ tz ]
  in

  let bench ?(repeats = repeats) ~variant ~schedule () =
    let%op mc = ma * mb in
    let comp = named ("mm_" ^ variant) (Train.forward mc) in
    let transform opt =
      match schedule with None -> opt | Some s -> Sched.apply (s ~mc:mc.Tensor.value opt) opt
    in
    let ctx = Context.auto () in
    let ctx, routine = Context.compile ~lowered_transform:transform ctx comp Ir.Indexing.Empty in
    (* Warmup (includes any lazy initialization and host transfers). *)
    let ctx = Context.run ctx routine in
    let _ = Context.get_values ctx mc.Tensor.value in
    let start = Time_now.nanoseconds_since_unix_epoch () in
    let ctx =
      Stdlib.Array.fold_left
        (fun ctx () -> Context.run ctx routine)
        ctx (Stdlib.Array.make repeats ())
    in
    let values = Context.get_values ctx mc.Tensor.value in
    let stop = Time_now.nanoseconds_since_unix_epoch () in
    let secs = Float.of_int63 Int63.(stop - start) /. 1e9 /. Float.of_int repeats in
    p "%-10s %8.3f ms  %8.2f GFLOP/s  (spot check %.1f)\n" variant (secs *. 1e3)
      (flops /. secs /. 1e9)
      values.(n + 1);
    secs
  in
  p "matmul m=%d n=%d k=%d, %d repeats, backend from config/OCANNL_BACKEND\n" m n k repeats;
  let backend = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc") in
  (* HIP belongs here too: the backend renders workgroup-shared placement exactly as CUDA and Metal
     do, so the shared/staged/tensorized variants — including the gh-ocannl-487 [mma_pd1]/[mma_pd2]
     pipelining pair — are expressible on it. Omitting it silently routed every HIP run into the
     CPU branch below, which is why the pipelining pair had never been timed on an AMD GPU. *)
  let has_shared =
    String.is_substring backend ~substring:"metal"
    || String.is_substring backend ~substring:"cuda"
    || String.is_substring backend ~substring:"hip"
  in
  let t_naive =
    if naive_repeats > 0 then bench ~repeats:naive_repeats ~variant:"naive" ~schedule:None ()
    else Float.nan
  in
  if has_shared then
    let t_par = bench ~variant:"parallel" ~schedule:(Some parallel_schedule) () in
    let t_smem = bench ~variant:"smem" ~schedule:(Some smem_schedule) () in
    let t_reg = bench ~variant:"regtile" ~schedule:(Some regtile_schedule) () in
    let t_mma1 =
      bench ~variant:"mma_pd1" ~schedule:(Some (mma_staged_schedule ~pipeline_depth:1)) () in
    let t_mma2 =
      bench ~variant:"mma_pd2" ~schedule:(Some (mma_staged_schedule ~pipeline_depth:2)) () in
    p "speedups vs naive: parallel %.1fx, smem %.1fx, regtile %.1fx, mma_pd1 %.1fx, mma_pd2 %.1fx\n"
      (t_naive /. t_par) (t_naive /. t_smem) (t_naive /. t_reg) (t_naive /. t_mma1)
      (t_naive /. t_mma2)
  else
    let t_pack = bench ~variant:"cpupack" ~schedule:(Some cpupack_schedule) () in
    let t_tmma = bench ~variant:"tensorize" ~schedule:(Some tensorize_schedule) () in
    let t_pmma = bench ~variant:"packmma" ~schedule:(Some packmma_schedule) () in
    let t_pmmap = bench ~variant:"packmma_par" ~schedule:(Some packmma_par_schedule) () in
    let t_hoist =
      bench ~variant:"pm_hoist" ~schedule:(Some (packmma_outer_schedule ~pack_a:false ~pack_b:`Hoist)) () in
    let t_mixed =
      bench ~variant:"pm_mixed" ~schedule:(Some (packmma_outer_schedule ~pack_a:true ~pack_b:`Hoist)) () in
    let t_bpk =
      bench ~variant:"pm_bpk" ~schedule:(Some (packmma_outer_schedule ~pack_a:true ~pack_b:`Chunk)) () in
    p
      "speedups vs naive: cpupack %.1fx, tensorize %.1fx, packmma %.1fx, packmma_par %.1fx, \
       pm_hoist %.1fx, pm_mixed %.1fx, pm_bpk %.1fx\n"
      (t_naive /. t_pack) (t_naive /. t_tmma) (t_naive /. t_pmma) (t_naive /. t_pmmap)
      (t_naive /. t_hoist) (t_naive /. t_mixed) (t_naive /. t_bpk)
