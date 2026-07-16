(* Matmul schedule benchmark (docs/proposals/schedule-ir-optops.md, phases S1-S3; gh-ocannl-412
   acceptance): times the naive 1x1-launch kernel against three hand-written schedules on the
   configured backend --

   - parallel: one thread per output element (Split i / Split j into Grid x Workgroup; the S1
     shape, Boehm kernel 1 equivalent);
   - smem: + Split k, operands staged through workgroup-shared tiles, the output accumulator
     privatized to a per-thread scalar (S2 + Privatize, Boehm kernel 3);
   - regtile: + second-level splits with materialized-unroll TM x TN register tiles accumulating
     into a privatized per-thread tile (S3 + Privatize, Boehm kernel 4/5 shape).

   On the C backends the shared schedules are rejected and the CPU variants run instead:
   cpupack (S4 cache tiling + operand packing, all-Serial), tensorize (whole-triple register-tiled
   Tile_mma, gh-ocannl-469), packmma (Tile_mma composed with cache tiling + packing, all-Serial
   GEBP), and packmma_par (the same composition with pool-parallel Grid row blocks and per-chunk
   privatized A~ tiles).

   Usage: dune exec bin/schedule_bench.exe -- [n] [repeats] (defaults 256 and 20; n must be a
   multiple of 64). Run with OCANNL_BACKEND=metal (or cuda); the C backends reject the shared
   schedules. Timing includes kernel executions and one device-to-host transfer per variant
   (runs queue on the stream; get_values synchronizes). *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module LL = Ir.Low_level
module Sched = Ir.Schedule
module Asgns = Ir.Assignments

let p = Stdio.printf

let nest_paths (llc : LL.t) : Ir.Indexing.symbol list list =
  let strip stmts =
    List.filter stmts ~f:(function LL.Noop | LL.Comment _ -> false | _ -> true)
  in
  let rec path (llc : LL.t) : Ir.Indexing.symbol list =
    match llc with
    | LL.For_loop { index; body; _ } -> (
        index :: (match strip (LL.flat_lines [ body ]) with [ single ] -> path single | _ -> []))
    | LL.If { body; _ } -> path body
    | _ -> []
  in
  List.filter_map (LL.flat_lines [ llc ]) ~f:(fun stmt ->
      match path stmt with [] -> None | p -> Some p)

let named name (comp : Asgns.comp) : Asgns.comp =
  { comp with asgns = Asgns.Block_comment (name, comp.asgns) }

let () =
  let n = if Array.length (Sys.get_argv ()) > 1 then Int.of_string (Sys.get_argv ()).(1) else 256 in
  let repeats =
    if Array.length (Sys.get_argv ()) > 2 then Int.of_string (Sys.get_argv ()).(2) else 20
  in
  assert (n % 64 = 0);
  let mav = Array.init (n * n) ~f:(fun i -> Float.of_int (i % 13) *. 0.25) in
  let mbv = Array.init (n * n) ~f:(fun i -> Float.of_int (i % 17) -. 8.) in
  let ma = TDSL.ndarray mav ~label:[ "ma" ] ~input_dims:[ n ] ~output_dims:[ n ] () in
  let mb = TDSL.ndarray mbv ~label:[ "mb" ] ~input_dims:[ n ] ~output_dims:[ n ] () in
  let flops = 2.0 *. Float.of_int n *. Float.of_int n *. Float.of_int n in

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
      ez; sp_zi; sp_zj; sp_i; sp_j; sp_k;
      Sched.Stage { source = ma.Tensor.value; tile_loops = [ i_i; k_i ]; shared = true; cooperative = None; hoisted = false };
      Sched.Stage { source = mb.Tensor.value; tile_loops = [ k_i; j_i ]; shared = true; cooperative = None; hoisted = false };
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
        Sched.Stage { source = ma.Tensor.value; tile_loops = [ i_i; k_i ]; shared = false; cooperative = None; hoisted = false };
        Sched.Stage { source = mb.Tensor.value; tile_loops = [ k_i; j_i ]; shared = false; cooperative = None; hoisted = false };
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
        Sched.Stage { source = ma.Tensor.value; tile_loops = [ i_w; i_t; k_i ]; shared = true; cooperative = None; hoisted = false };
        Sched.Stage { source = mb.Tensor.value; tile_loops = [ k_i; j_w; j_t ]; shared = true; cooperative = None; hoisted = false };
        Sched.Privatize { target = mc; over = k_o };
        Sched.Unroll { axis = i_t; materialize = true };
        Sched.Unroll { axis = j_t; materialize = true };
      ]
  in

  (* Register-tiled Tile_mma micro-kernel (gh-ocannl-469): the whole i x j x k triple becomes one
     Tile_mma statement, which the C backends render tinyBLAS-style — the C-tile held in an RM x
     RN grid of vector registers across the entire k-loop, edges peeled. The zeroing must cover
     the lane slot (validate_parallel's coverage rule), so its column loop becomes the Workgroup
     axis and the lane width matches its extent (the lane loop renders serially on the C
     backends, executing the guarded statement once). *)
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
     (reused across all row blocks) and the A tile [bm x bk] at i_o, then tensorize the inner
     triple — the register-tiled micro-kernel streams the contiguous packed tiles. All-Serial
     with a unit lane, so the whole-node zeroing stays legal. *)
  let packmma_schedule ~mc:_ opt =
    let bm, bk = (64, 64) in
    let i, j, k = accum_syms opt in
    let sp_i, i_o, i_i = Sched.split ~axis:i ~factor:bm ~outer:LL.Serial ~inner:LL.Serial in
    let sp_k, k_o, k_i = Sched.split ~axis:k ~factor:bk ~outer:LL.Serial ~inner:LL.Serial in
    let sink sym below = List.map below ~f:(fun inner -> Sched.Swap { outer = sym; inner }) in
    let stage source tile_loops =
      Sched.Stage { source; tile_loops; shared = false; cooperative = None; hoisted = false }
    in
    let tz, _lane = Sched.tensorize ~i:i_i ~j ~k:k_i ~simd_width:1 in
    [ sp_i; sp_k ]
    @ sink j [ k_o ]
    @ sink i_i [ k_o ]
    @ sink i_o [ k_o ]
    @ [ stage mb.Tensor.value [ k_i; j ]; stage ma.Tensor.value [ i_i; k_i ]; tz ]
  in

  (* The fully parallel packed GEMM (gh-ocannl-469 follow-up): the row-block loop is Grid-typed
     and pool-parallelizes — the per-row-block A~ tile is privatized to per-chunk block-scope
     storage by the renderer, the B~ panel packed at k_o is read-only inside the Grid body
     (behind a pointer alias under the blocks extension). The whole-node zeroing is no longer
     legal beside a hardware-annotated loop, so it expands with the same Grid row geometry. *)
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
      Sched.Stage { source; tile_loops; shared = false; cooperative = None; hoisted = false }
    in
    let tz, _lane = Sched.tensorize ~i:i_i ~j ~k:k_i ~simd_width:1 in
    [ ez; sp_zi; sp_i; sp_k ]
    @ sink j [ k_o ]
    @ sink i_i [ k_o ]
    @ sink i_o [ k_o ]
    @ [ stage mb.Tensor.value [ k_i; j ]; stage ma.Tensor.value [ i_i; k_i ]; tz ]
  in

  let bench ~variant ~schedule =
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
    let ctx = Stdlib.Array.fold_left (fun ctx () -> Context.run ctx routine) ctx
        (Stdlib.Array.make repeats ()) in
    let values = Context.get_values ctx mc.Tensor.value in
    let stop = Time_now.nanoseconds_since_unix_epoch () in
    let secs = Float.of_int63 Int63.(stop - start) /. 1e9 /. Float.of_int repeats in
    p "%-10s %8.3f ms  %8.2f GFLOP/s  (spot check %.1f)\n" variant (secs *. 1e3)
      (flops /. secs /. 1e9)
      values.(n + 1);
    secs
  in
  p "matmul %dx%dx%d, %d repeats, backend from config/OCANNL_BACKEND\n" n n n repeats;
  let backend = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc") in
  let has_shared =
    String.is_substring backend ~substring:"metal" || String.is_substring backend ~substring:"cuda"
  in
  let t_naive = bench ~variant:"naive" ~schedule:None in
  if has_shared then (
    let t_par = bench ~variant:"parallel" ~schedule:(Some parallel_schedule) in
    let t_smem = bench ~variant:"smem" ~schedule:(Some smem_schedule) in
    let t_reg = bench ~variant:"regtile" ~schedule:(Some regtile_schedule) in
    p "speedups vs naive: parallel %.1fx, smem %.1fx, regtile %.1fx\n" (t_naive /. t_par)
      (t_naive /. t_smem) (t_naive /. t_reg))
  else
    let t_pack = bench ~variant:"cpupack" ~schedule:(Some cpupack_schedule) in
    let t_tmma = bench ~variant:"tensorize" ~schedule:(Some tensorize_schedule) in
    let t_pmma = bench ~variant:"packmma" ~schedule:(Some packmma_schedule) in
    let t_pmmap = bench ~variant:"packmma_par" ~schedule:(Some packmma_par_schedule) in
    p "speedups vs naive: cpupack %.1fx, tensorize %.1fx, packmma %.1fx, packmma_par %.1fx\n"
      (t_naive /. t_pack) (t_naive /. t_tmma) (t_naive /. t_pmma) (t_naive /. t_pmmap)
