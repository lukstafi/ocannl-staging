(* Tensorize / Tile_mma, T1+T2 of docs/proposals/tensorize-mma.md: the tensorized matmul schedule
   executed against the serial twin on every backend.

   [mc = ma * mb] (32x32 times 32x32) lowers to a zeroing nest plus the naive triple loop. The
   schedule: expand the zeroing, split its row loop into Grid(2) x Serial(16) and retype its column
   loop to Workgroup(32) (partition-aligned with the accumulation's grid blocks, and covering the
   lane slot -- barrier-strength uniformity requires every workgroup extent to equal 32 once a
   [Tile_mma] is present); split the accumulation's i by 16 into Grid x Serial; then [Tensorize { i
   = i_i; j; k }], which replaces the serial 16x32x32 micro-kernel with a [Tile_mma] block statement
   wrapped in a fresh extent-32 Workgroup lane loop.

   On Metal the statement renders as [simdgroup_matrix] fragments (simdgroup_load /
   simdgroup_multiply_accumulate / simdgroup_store, barrier-bracketed) and must match the serial
   twin within f32 tolerance (the tile reduction reassociates). On the C backends the f32 statement
   renders as the register-tiled vector micro-kernel (gh-ocannl-469, tinyBLAS's mnpack: the C-tile
   in an RM×RN grid of vector registers held across the k-loop, edges peeled) under the same [if
   (lane == 0)] guard as the scalar fallback; each output element's k-chain stays in serial order
   with the same fused rounding, so the values must match the serial twin BITWISE. Non-FMA-form and
   non-f32/f64 statements (the half and fp8 cases below) keep the scalar fallback on the C backends,
   also bitwise; so does uniform f32 on CUDA, whose tensor-core emissions cover the half/bf16 (wmma)
   and fp8 (inline-PTX mma.sync) combinations. The negative check pins Tensorize's pattern
   discipline. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module Tn = Ir.Tnode
module LL = Ir.Low_level
module Sched = Ir.Schedule
module Asgns = Ir.Assignments
module Numerics = Ir.Numerics

let () = Utils.settings.output_debug_files_in_build_directory <- true
let () = Numerics.set_policy { (Numerics.get ()) with tf32_matmuls = false }
let p name b = Stdio.printf "%s: %b\n" name b
let approx a b = Float.(abs (a - b) < 1e-2)
let approx_rel a b = Float.(abs (a - b) <= 1e-2 * max 1. (abs b))
let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")
let on_metal = String.is_substring backend_name ~substring:"metal"

let on_gpu =
  on_metal
  || String.is_substring backend_name ~substring:"cuda"
  || String.is_substring backend_name ~substring:"hip"

let read_generated base_name =
  let ext =
    if on_metal then ".metal"
    else if String.is_substring backend_name ~substring:"hip" then ".hip"
    else if on_gpu then ".cu"
    else ".c"
  in
  let path = Utils.build_file (base_name ^ ext) in
  if Stdlib.Sys.file_exists path then Some (Stdio.In_channel.read_all path) else None

(* The maximal single-child chains of statement-level loops: one symbol list per top-level nest. *)
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

(* The cross-[k_o] accumulator-residency structural pin (gh-ocannl-480), shared by every staged half
   leg and backend: the accumulator fragment is loaded once before the serial [k_o] body (bracketed
   by [body_begin]/[body_end] marker comments) and stored once after it, and the body in between
   carries update-only MMA steps (with the trailing [barrier] that releases the staged tiles) and no
   fragment-array load/store of its own. Parity alone cannot see the difference — both the resident
   and the per-[k_o] forms are correct — so the pin is on the emitted source. *)
let residency_holds src ~frag_load ~body_begin ~body_end ~frag_store ~barrier =
  match
    ( String.substr_index src ~pattern:frag_load,
      String.substr_index src ~pattern:body_begin,
      String.substr_index src ~pattern:body_end,
      String.substr_index src ~pattern:frag_store )
  with
  | Some load, Some beg, Some fin, Some store when load < beg && beg < fin && fin < store ->
      let reduction_body = String.sub src ~pos:beg ~len:(fin - beg) in
      let update_has_trailing_barrier =
        match String.substr_index reduction_body ~pattern:"/* tile_mma fragment update" with
        | Some update_pos ->
            String.is_substring (String.drop_prefix reduction_body update_pos) ~substring:barrier
        | None -> false
      in
      (not (String.is_substring reduction_body ~substring:frag_load))
      && (not (String.is_substring reduction_body ~substring:frag_store))
      && update_has_trailing_barrier
  | _ -> false

(* The per-backend marker sets for [residency_holds] on a staged half leg. The fragment scope emits
   the same anchor comments regardless of accumulator element type, so both the uniform-f16 and the
   f16->f32 legs share these. Metal keeps the fragment first in its store; the wmma backends put the
   destination pointer first. *)
let staged_half_resident src =
  if on_metal then
    residency_holds src ~frag_load:"simdgroup_load(__mma_fragment_"
      ~body_begin:"/* simdgroup fragment reduction body begins */"
      ~body_end:"/* simdgroup fragment reduction body ends */"
      ~frag_store:"simdgroup_store(__mma_fragment_"
      ~barrier:"threadgroup_barrier(mem_flags::mem_threadgroup);"
  else if String.is_substring backend_name ~substring:"hip" then
    residency_holds src ~frag_load:"rocwmma::load_matrix_sync(__mma_fragment_"
      ~body_begin:"/* rocwmma fragment reduction body begins */"
      ~body_end:"/* rocwmma fragment reduction body ends */"
      ~frag_store:"rocwmma::store_matrix_sync(__mma_dp" ~barrier:"__syncthreads();"
  else
    residency_holds src ~frag_load:"nvcuda::wmma::load_matrix_sync(__mma_fragment_"
      ~body_begin:"/* wmma fragment reduction body begins */"
      ~body_end:"/* wmma fragment reduction body ends */"
      ~frag_store:"nvcuda::wmma::store_matrix_sync(__mma_dp" ~barrier:"__syncthreads();"

let n = 32

(* The row-block factor must keep the [Tile_mma] block extents multiples of every backend's
   intrinsic tile: 8x8x8 on Metal, but 16x16x16 for CUDA wmma and m16-n8-k32 for the fp8 [mma.sync]
   path — with [bm = 8] the CUDA emissions decline (m = 8) and silently take the scalar fallback. 16
   is divisible by 8, so one factor exercises the intrinsics everywhere. *)
let bm = 16

(* The cooperating width of one tile-MMA instruction: the Metal simdgroup and the CUDA warp are both
   32 wide ({!Ir.Backend_intf.mma_simd_width}); on the C backends the lane loop renders serially and
   any extent is correct. *)
let simd_width = 32

let () =
  let mav = Array.init (n * n) ~f:(fun i -> Float.of_int (i % 13) *. 0.25) in
  let mbv = Array.init (n * n) ~f:(fun i -> Float.of_int (i % 17) -. 8.) in
  let ma = TDSL.ndarray mav ~label:[ "ma" ] ~input_dims:[ n ] ~output_dims:[ n ] () in
  let mb = TDSL.ndarray mbv ~label:[ "mb" ] ~input_dims:[ n ] ~output_dims:[ n ] () in

  (* --- Serial twin --- *)
  let%op mc0 = ma * mb in
  let serial_comp = named "mm_serial" (Train.forward mc0) in
  let ctx_s = Context.auto () in
  let ctx_s, routine_s =
    Context.compile ~lowered_transform:(fun opt -> opt) ctx_s serial_comp Ir.Indexing.Empty
  in
  let ctx_s = Context.run ctx_s routine_s in
  let got_serial = Context.get_values ctx_s mc0.Tensor.value in

  (* --- The tensorized schedule --- *)
  let mma_schedule ?(bm = bm) ~out (opt : LL.optimized) : Sched.schedule =
    let paths = nest_paths opt.LL.llc in
    let i, j, k =
      match List.find_exn paths ~f:(fun p -> List.length p = 3) with
      | [ i; j; k ] -> (i, j, k)
      | _ -> assert false
    in
    let ez, zsyms = Sched.expand_zero ~tn:out in
    let zi, zj = match zsyms with [ zi; zj ] -> (zi, zj) | _ -> assert false in
    (* Zeroing: Grid(2) x Serial(16) rows aligned with the accumulation's grid blocks, and the
       column loop as the Workgroup(32) axis -- each lane zeroes its own column, and the extent
       matches the lane loop (barrier-strength uniformity). *)
    let sp_zi, _, _ = Sched.split ~axis:zi ~factor:bm ~outer:LL.Grid ~inner:LL.Serial in
    let rz = Sched.Retype { axis = zj; ty = LL.Workgroup } in
    let sp_i, _, i_i = Sched.split ~axis:i ~factor:bm ~outer:LL.Grid ~inner:LL.Serial in
    let tz, _lane = Sched.tensorize ~i:i_i ~j ~k ~simd_width in
    [ ez; sp_zi; rz; sp_i; tz ]
  in

  (* --- CUDA tf32 policy (gh-ocannl-478): positive and negative execution pins. The policy-off leg
     is compiled first and must retain the lane-0 scalar fallback with bitwise parity. The policy-on
     legs must both execute within tf32 tolerance and structurally render wmma tf32; [mma_census] is
     checked as well so a loosened numeric comparison cannot hide a decline. The k=24 leg is
     deliberately divisible by tf32's k=8 tile but not the f16/bf16 k=16 tile. The transposed-A leg
     covers the col-major fragment form retained by [wmma_combo]. --- *)
  let compile_mma_with_census ?(inspect = fun (_ : LL.optimized) -> ()) ~name tensor =
    Ir.C_syntax.mma_census := [];
    Ir.C_syntax.mma_census_enabled := true;
    let ctx, routine =
      Exn.protect
        ~f:(fun () ->
          let transform opt =
            inspect opt;
            Sched.apply (mma_schedule ~out:tensor.Tensor.value opt) opt
          in
          Context.compile ~lowered_transform:transform (Context.auto ())
            (named name (Train.forward tensor))
            Ir.Indexing.Empty)
        ~finally:(fun () -> Ir.C_syntax.mma_census_enabled := false)
    in
    let census = List.rev_map !Ir.C_syntax.mma_census ~f:snd in
    let ctx = Context.run ctx routine in
    (Context.get_values ctx tensor.Tensor.value, census)
  in
  let compile_serial ~name tensor =
    let ctx, routine =
      Context.compile
        ~lowered_transform:(fun opt -> opt)
        (Context.auto ())
        (named name (Train.forward tensor))
        Ir.Indexing.Empty
    in
    let ctx = Context.run ctx routine in
    Context.get_values ctx tensor.Tensor.value
  in
  let tf32_inputs ~tag ~k =
    let av =
      Array.init (n * k) ~f:(fun x ->
          ((Float.of_int (x % 23) -. 11.) *. 0.03125) +. (Float.of_int (x % 3) *. 0.0007))
    in
    let bv =
      Array.init (k * n) ~f:(fun x ->
          ((Float.of_int (x % 19) -. 9.) *. 0.046875) -. (Float.of_int (x % 5) *. 0.0003))
    in
    ( TDSL.ndarray av ~label:[ tag ^ "a" ] ~input_dims:[ k ] ~output_dims:[ n ] (),
      TDSL.ndarray bv ~label:[ tag ^ "b" ] ~input_dims:[ n ] ~output_dims:[ k ] () )
  in
  let tf32_limits =
    {
      Ir.Backend_intf.no_hardware_limits with
      mma =
        Some
          {
            Ir.Backend_intf.mma_simd_width = 32;
            mma_tile = (16, 16, 16);
            mma_format_tiles =
              [
                ( (Ir.Backend_intf.Mma_tf32, Ir.Backend_intf.Mma_tf32, Ir.Backend_intf.Mma_f32),
                  (16, 16, 8) );
              ];
            mma_staged_layouts = [];
            mma_pipeline_depths = [];
          };
    }
  in
  let tf32_mma_seeded opt =
    List.exists (Autotune.sketch_seed_params ~is_gpu:true ~is_cpu:false ~limits:tf32_limits opt)
      ~f:(fun p -> p.Autotune.sk_mma)
  in
  if String.is_substring backend_name ~substring:"cuda" then (
    let a_off, b_off = tf32_inputs ~tag:"tf32_off_" ~k:n in
    let%op c_off_serial = a_off * b_off in
    let%op c_off_mma = a_off * b_off in
    Numerics.set_policy { (Numerics.get ()) with tf32_matmuls = false };
    let want_off = compile_serial ~name:"mm_tf32_off_serial" c_off_serial in
    let off_seeded = ref false in
    let got_off, census_off =
      compile_mma_with_census
        ~inspect:(fun opt -> off_seeded := tf32_mma_seeded opt)
        ~name:"mm_tf32_off_mma" c_off_mma
    in
    let src_off = read_generated "mm_tf32_off_mma" in
    p "tf32 policy-off matmul matches the serial twin bitwise"
      (Array.for_all2_exn got_off want_off ~f:Float.equal);
    p "tf32 policy-off renders and records the scalar fallback"
      (List.for_all census_off ~f:(Ir.C_syntax.equal_mma_rendering Ir.C_syntax.Mma_scalar_fallback)
      && (not (List.is_empty census_off))
      && Option.value_map src_off ~default:false ~f:(fun src ->
          String.is_substring src ~substring:"== 0)"
          && (not (String.is_substring src ~substring:"(wmma-tf32)"))
          && not (String.is_substring src ~substring:"precision::tf32")));
    p "tf32 policy-off autotune omits tensorized candidates" (not !off_seeded);

    Numerics.set_policy { (Numerics.get ()) with tf32_matmuls = true };
    let a_on, b_on = tf32_inputs ~tag:"tf32_on_" ~k:n in
    let%op c_on_serial = a_on * b_on in
    let%op c_on_mma = a_on * b_on in
    let want_on = compile_serial ~name:"mm_tf32_on_serial" c_on_serial in
    let got_on, census_on = compile_mma_with_census ~name:"mm_tf32_on_mma" c_on_mma in
    let src_on = read_generated "mm_tf32_on_mma" in
    p "tf32 policy-on matmul matches the serial twin within tolerance"
      (Array.for_all2_exn got_on want_on ~f:approx_rel);
    p "tf32 policy-on renders and records wmma tf32"
      (List.for_all census_on ~f:(Ir.C_syntax.equal_mma_rendering Ir.C_syntax.Mma_intrinsics)
      && (not (List.is_empty census_on))
      && Option.value_map src_on ~default:false ~f:(fun src ->
          String.is_substring src ~substring:"(wmma-tf32)"
          && String.is_substring src ~substring:"precision::tf32"
          && String.is_substring src ~substring:"__float_to_tf32"));

    let a_k24, b_k24 = tf32_inputs ~tag:"tf32_k24_" ~k:24 in
    let%op c_k24_serial = a_k24 * b_k24 in
    let%op c_k24_mma = a_k24 * b_k24 in
    let want_k24 = compile_serial ~name:"mm_tf32_k24_serial" c_k24_serial in
    let k24_seeded = ref false in
    let got_k24, census_k24 =
      compile_mma_with_census
        ~inspect:(fun opt -> k24_seeded := tf32_mma_seeded opt)
        ~name:"mm_tf32_k24_mma" c_k24_mma
    in
    p "tf32 k=24 divergent tile matches and emits"
      (Array.for_all2_exn got_k24 want_k24 ~f:approx_rel
      && List.exists census_k24 ~f:(Ir.C_syntax.equal_mma_rendering Ir.C_syntax.Mma_intrinsics)
      && Option.value_map (read_generated "mm_tf32_k24_mma") ~default:false ~f:(fun src ->
          String.is_substring src ~substring:"tile_mma 16x32x24 (wmma-tf32)"));
    p "tf32 k=24 autotune seeds tensorized candidates" !k24_seeded;

    let atv =
      Array.init (n * n) ~f:(fun x ->
          ((Float.of_int (x % 17) -. 8.) *. 0.0625) +. (Float.of_int (x % 7) *. 0.0002))
    in
    let btv =
      Array.init (n * n) ~f:(fun x ->
          ((Float.of_int (x % 13) -. 6.) *. 0.078125) -. (Float.of_int (x % 3) *. 0.0004))
    in
    let at = TDSL.ndarray atv ~label:[ "tf32_at" ] ~output_dims:[ n; n ] () in
    let bt = TDSL.ndarray btv ~label:[ "tf32_bt" ] ~output_dims:[ n; n ] () in
    let%op ct_serial = at +* "ki;kj=>ij" bt in
    let%op ct_mma = at +* "ki;kj=>ij" bt in
    let want_t = compile_serial ~name:"mm_tf32_ta_serial" ct_serial in
    let got_t, census_t = compile_mma_with_census ~name:"mm_tf32_ta_mma" ct_mma in
    p "tf32 transposed-A matmul matches and emits"
      (Array.for_all2_exn got_t want_t ~f:approx_rel
      && List.exists census_t ~f:(Ir.C_syntax.equal_mma_rendering Ir.C_syntax.Mma_intrinsics)
      && Option.value_map (read_generated "mm_tf32_ta_mma") ~default:false ~f:(fun src ->
          String.is_substring src ~substring:"matrix_a, 16, 16, 8"
          && String.is_substring src ~substring:"col_major"
          && String.is_substring src ~substring:"(wmma-tf32)"));
    Numerics.set_policy { (Numerics.get ()) with tf32_matmuls = false })
  else (
    p "tf32 policy-off matmul matches the serial twin bitwise" true;
    p "tf32 policy-off renders and records the scalar fallback" true;
    p "tf32 policy-off autotune omits tensorized candidates" true;
    p "tf32 policy-on matmul matches the serial twin within tolerance" true;
    p "tf32 policy-on renders and records wmma tf32" true;
    p "tf32 k=24 divergent tile matches and emits" true;
    p "tf32 k=24 autotune seeds tensorized candidates" true;
    p "tf32 transposed-A matmul matches and emits" true);

  let%op mc1 = ma * mb in
  let mma_comp = named "mm_mma" (Train.forward mc1) in
  let transform opt = Sched.apply (mma_schedule ~out:mc1.Tensor.value opt) opt in
  let ctx_a = Context.auto () in
  let ctx_a, routine_a =
    Context.compile ~lowered_transform:transform ctx_a mma_comp Ir.Indexing.Empty
  in
  let ctx_a = Context.run ctx_a routine_a in
  let got_mma = Context.get_values ctx_a mc1.Tensor.value in
  p "tensorized matmul values match the serial twin"
    (Array.for_all2_exn got_mma got_serial ~f:approx);
  p "C-backend fallback matches bitwise"
    (on_gpu || Array.for_all2_exn got_mma got_serial ~f:Float.equal);
  (match read_generated "mm_mma" with
  | None -> p "tensorized structure as expected" false
  | Some src ->
      let has s = String.is_substring src ~substring:s in
      let ok =
        if on_metal then
          (* The intrinsic path: fragment loads and stores, the mma step, and the bracketing
             barriers; no lane-0 fallback guard. *)
          has "simdgroup_load"
          && has "simdgroup_multiply_accumulate"
          && has "simdgroup_store" && has "threadgroup_barrier"
          && not (has "== 0)")
        else if on_gpu then
          (* CUDA: the wmma draft supports the half/bf16 combinations only, so uniform f32 declines
             to the scalar fallback under the lane-0 guard (the register tiling is CPU-only — the
             packed vector style never takes it). *)
          has "== 0)" && has "fma" && not (has "simdgroup")
        else
          (* The register-tiled path (gh-ocannl-469): the vector C-tile under the lane-0 guard (a
             serial loop of extent 32 binds the lane on the C backends); fused per-element chains
             are what makes the bitwise parity above hold. *)
          has "== 0)" && has "Tile_mma register tiling" && has "fma" && not (has "simdgroup")
      in
      p "tensorized structure as expected" ok);

  (* --- Half precision: [simdgroup_half8x8] on Metal, the wmma f16 path on CUDA (T3 draft), rocWMMA
     on HIP, the scalar fallback on the C backends. The inputs are multiples of 1/8 and 1/4 with
     32-term sums bounded by 12, so every product and partial sum is exactly representable in f16:
     the result is EXACT regardless of accumulation order, and parity is bitwise on every backend
     and either rendering path. --- *)
  let mah =
    NTDSL.init ~l:"mah" ~prec:Ir.Ops.half ~i:[ n ] ~o:[ n ]
      ~f:(fun idcs -> Float.of_int (((idcs.(0) * n) + idcs.(1)) % 5) *. 0.125)
      ()
  in
  let mbh =
    NTDSL.init ~l:"mbh" ~prec:Ir.Ops.half ~i:[ n ] ~o:[ n ]
      ~f:(fun idcs -> (Float.of_int (((idcs.(0) * n) + idcs.(1)) % 7) -. 3.) *. 0.25)
      ()
  in
  let%op mch0 = mah * mbh in
  Tn.update_prec mch0.Tensor.value Ir.Ops.half;
  let ctx_hs = Context.auto () in
  let ctx_hs, routine_hs =
    Context.compile
      ~lowered_transform:(fun opt -> opt)
      ctx_hs
      (named "mm_h_serial" (Train.forward mch0))
      Ir.Indexing.Empty
  in
  let ctx_hs = Context.run ctx_hs routine_hs in
  let got_h_serial = Context.get_values ctx_hs mch0.Tensor.value in
  let%op mch1 = mah * mbh in
  Tn.update_prec mch1.Tensor.value Ir.Ops.half;
  let transform_h opt = Sched.apply (mma_schedule ~out:mch1.Tensor.value opt) opt in
  let ctx_h = Context.auto () in
  let ctx_h, routine_h =
    Context.compile ~lowered_transform:transform_h ctx_h
      (named "mm_h_mma" (Train.forward mch1))
      Ir.Indexing.Empty
  in
  let ctx_h = Context.run ctx_h routine_h in
  let got_h = Context.get_values ctx_h mch1.Tensor.value in
  p "half tensorized matmul matches the serial twin bitwise"
    (Array.for_all2_exn got_h got_h_serial ~f:Float.equal);
  (match read_generated "mm_h_mma" with
  | None -> p "half tensorized structure as expected" false
  | Some src ->
      let has s = String.is_substring src ~substring:s in
      let ok =
        if on_metal then has "simdgroup_half8x8" && not (has "== 0)")
        else if String.is_substring backend_name ~substring:"hip" then
          (* HIP: the rocWMMA f16 intrinsic (verified on gfx1151), no lane-0 fallback guard. *)
          has "rocwmma::mma_sync" && not (has "== 0)")
        else if on_gpu then
          (* CUDA: the wmma f16 intrinsic, or the lane-0 fallback on older devices. *)
          has "nvcuda::wmma" || has "== 0)"
        else
          (* Half precision declines the register tiling (single/double only): the scalar
             fallback. *)
          has "== 0)" && (not (has "simdgroup")) && not (has "Tile_mma register tiling")
      in
      p "half tensorized structure as expected" ok);

  (* --- Bfloat16 (gh-ocannl-545): the two accumulator shapes, which are NOT interchangeable on
     CUDA. [nvcuda::wmma] has no bf16 accumulator fragment (mma.hpp declares [__nv_bfloat16] operands
     against a [float] accumulator only), so bf16 x bf16 -> f32 takes the wmma template path while
     the uniform bf16 x bf16 -> bf16 combination — the one a uniformly-bf16 network actually
     produces — takes the inline-PTX [mma.sync.aligned.m16n8k16...f32.bf16.bf16.f32] path, whose
     per-lane accumulator registers convert at the [d] boundary. Both legs are pinned so a
     regression on either shows up as a rendering change, not just a timing one. HIP has both
     rocWMMA combinations and Metal has the uniform one ([simdgroup_bfloat8x8]).

     The inputs are multiples of 1/4 and 1/2 with 32-term sums bounded by 16, so every product is a
     multiple of 1/8 and every partial sum is exactly representable in bf16 (8 mantissa bits): the
     result is EXACT regardless of accumulation order or accumulator width, and parity is bitwise on
     every backend and either rendering path. --- *)
  let mab =
    NTDSL.init ~l:"mab" ~prec:Ir.Ops.bfloat16 ~i:[ n ] ~o:[ n ]
      ~f:(fun idcs -> Float.of_int (((idcs.(0) * n) + idcs.(1)) % 3) *. 0.25)
      ()
  in
  let mbb =
    NTDSL.init ~l:"mbb" ~prec:Ir.Ops.bfloat16 ~i:[ n ] ~o:[ n ]
      ~f:(fun idcs -> (Float.of_int (((idcs.(0) * n) + idcs.(1)) % 5) -. 2.) *. 0.5)
      ()
  in
  let mtab =
    NTDSL.init ~l:"mtab" ~prec:Ir.Ops.bfloat16 ~o:[ n; n ]
      ~f:(fun idcs -> Float.of_int (((idcs.(0) * n) + idcs.(1)) % 3) *. 0.25)
      ()
  in
  let mtbb =
    NTDSL.init ~l:"mtbb" ~prec:Ir.Ops.bfloat16 ~o:[ n; n ]
      ~f:(fun idcs -> (Float.of_int (((idcs.(0) * n) + idcs.(1)) % 5) -. 2.) *. 0.5)
      ()
  in
  let bf16_leg ?bm ~tag ~acc_prec ~build ~check () =
    let mcb0 = build () in
    Tn.update_prec mcb0.Tensor.value acc_prec;
    let ctx_bs = Context.auto () in
    let ctx_bs, routine_bs =
      Context.compile
        ~lowered_transform:(fun opt -> opt)
        ctx_bs
        (named ("mm_" ^ tag ^ "_serial") (Train.forward mcb0))
        Ir.Indexing.Empty
    in
    let ctx_bs = Context.run ctx_bs routine_bs in
    let want = Context.get_values ctx_bs mcb0.Tensor.value in
    let mcb1 = build () in
    Tn.update_prec mcb1.Tensor.value acc_prec;
    let transform_b opt = Sched.apply (mma_schedule ?bm ~out:mcb1.Tensor.value opt) opt in
    let ctx_b = Context.auto () in
    let ctx_b, routine_b =
      Context.compile ~lowered_transform:transform_b ctx_b
        (named ("mm_" ^ tag ^ "_mma") (Train.forward mcb1))
        Ir.Indexing.Empty
    in
    let ctx_b = Context.run ctx_b routine_b in
    let got = Context.get_values ctx_b mcb1.Tensor.value in
    (* Bitwise on CUDA and Metal, where the tensor core returns the exactly-rounded dot product for
       these exactly-representable inputs. NOT bitwise on HIP: gfx1151's WMMA does not, in any of
       the four rocWMMA format combinations, and the deviation is a hardware property rather than a
       rendering defect — a standalone rocWMMA program (no OCANNL in the loop) reproduces it cell
       for cell from the same data. Measured on a single exact 16x16x16 [mma_sync] with a
       [fill_fragment]-zeroed accumulator, against an exact double reference:

         bf16 x bf16 -> f32   227/256 cells differ, worst 1.19e-07  (one f32 ulp at ~1.0)
         bf16 x bf16 -> bf16  235/256 cells differ, worst 5.86e-03
         f16  x f16  -> f32   227/256 cells differ, worst 1.19e-07
         f16  x f16  -> f16    18/256 cells differ, worst 5.96e-08

       So the tolerance is per accumulator format, and the narrow-accumulator one has to be wide:
       the uniform-bf16 combination — HIP's only tensor-core route for a uniformly-bf16 network —
       loses close to a bf16 ulp at the partial-sum scale. The [staged+tensorized half] leg below
       already carried this exception for the f16->f32 combination; these legs arrived later
       (gh-ocannl-545) without AMD hardware in the loop, so they asserted the CUDA/Metal premise on
       a backend where it does not hold. Keeping the comparison numeric rather than dropping it
       preserves what it is for: a wrong stride or a mis-mapped fragment moves values by O(1), not
       by 1e-2. *)
    let eq =
      if not (String.is_substring backend_name ~substring:"hip") then Float.equal
      else if Ir.Ops.equal_prec acc_prec Ir.Ops.bfloat16 then fun a b ->
        Float.(abs (a - b) <= 0.05 * max 1. (abs b))
      else fun a b -> Float.(abs (a - b) <= 1e-5 * max 1. (abs b))
    in
    p
      (Printf.sprintf "%s tensorized matmul matches the serial twin" tag)
      (Array.for_all2_exn got want ~f:eq);
    match read_generated ("mm_" ^ tag ^ "_mma") with
    | None -> p (Printf.sprintf "%s tensorized structure as expected" tag) false
    | Some src -> p (Printf.sprintf "%s tensorized structure as expected" tag) (check src)
  in
  let mul () =
    let%op t = mab * mbb in
    t
  in
  bf16_leg ~tag:"bf32" ~acc_prec:Ir.Ops.single ~build:mul
    ~check:(fun src ->
      let has s = String.is_substring src ~substring:s in
      if on_metal then
        (* [simdgroup_matrix] is uniform-precision only: this mixed combination declines. *)
        has "== 0)"
      else if String.is_substring backend_name ~substring:"hip" then
        has "rocwmma::mma_sync" && not (has "== 0)")
      else if on_gpu then
        (* CUDA: the wmma bf16 fragment path. Both bf16 renderings need sm_80, the same floor the
           tf32 legs above already pin strictly. *)
        has "(wmma-bf16)"
      else has "== 0)" && (not (has "simdgroup")) && not (has "Tile_mma register tiling"))
    ();
  (* The uniform combination in all three operand orientations. CUDA's inline-PTX arm gathers every
     fragment element through an explicit (row, col) address, so a transposed operand only swaps
     that address — unlike the fp8 arm, which declines [ta]/[tb]. A mis-mapped gather would compute
     a different product, so the bitwise parity is what pins the per-lane layout. *)
  let uniform_check src =
    let has s = String.is_substring src ~substring:s in
    if on_metal then has "simdgroup_bfloat8x8" && not (has "== 0)")
    else if String.is_substring backend_name ~substring:"hip" then
      has "rocwmma::mma_sync" && not (has "== 0)")
    else if on_gpu then
      (* CUDA: the inline-PTX bf16 path (sm_80+). *)
      has "mma.sync.aligned.m16n8k16"
    else has "== 0)" && (not (has "simdgroup")) && not (has "Tile_mma register tiling")
  in
  bf16_leg ~tag:"bfu" ~acc_prec:Ir.Ops.bfloat16 ~build:mul ~check:uniform_check ();
  bf16_leg ~tag:"bfu_ta" ~acc_prec:Ir.Ops.bfloat16 ~check:uniform_check
    ~build:(fun () ->
      let%op t = mtab +* "ki;kj=>ij" mtbb in
      t)
    ();
  bf16_leg ~tag:"bfu_tb" ~acc_prec:Ir.Ops.bfloat16 ~check:uniform_check
    ~build:(fun () ->
      let%op t = mtab +* "ik;jk=>ij" mtbb in
      t)
    ();
  (* [bm = 32] makes the block extents 32x32x32, so the row-tile loop [__mi] runs twice: the
     m-direction fragment addressing is only exercised above one intrinsic row tile. *)
  bf16_leg ~bm:(2 * bm) ~tag:"bfu_m2" ~acc_prec:Ir.Ops.bfloat16 ~build:mul ~check:uniform_check ();

  (* --- Fp8 (e5m2) inputs accumulated in f32: the inline-PTX [mma.sync] path on CUDA sm_89+
     (tensorize-mma T3+; wmma cannot express fp8), the scalar fallback elsewhere. e5m2 has 2
     mantissa bits: inputs from {-1,-0.5,0,0.5,1} and {-1.5..1.5 step 0.5} are exact, every product
     is a multiple of 0.25 bounded by 1.5, and a 32-term f32 sum of such products is exact
     regardless of accumulation order — so parity is bitwise on every backend and either rendering
     path. Skipped on Metal, which has no fp8 storage precision at all ([typ_of_prec] rejects
     [Fp8_prec]) — even the serial twin cannot compile there.

     All three operand orientations are covered (gh-ocannl-481 item 1): the arm's per-lane byte
     gathers now address every fragment element at its transposed (col, row) offset under
     [ta]/[tb], so the gradient GEMMs' layouts ([dA = g.B^T], [dB = A^T.g]) tensorize instead of
     silently scalar-falling-back. A mis-mapped gather computes a different product, so the bitwise
     parity is what pins the per-lane layout. --- *)
  (if on_metal then
     List.iter [ "f8"; "f8_ta"; "f8_tb" ] ~f:(fun tag ->
         p (Printf.sprintf "%s tensorized matmul matches the serial twin bitwise" tag) true;
         p (Printf.sprintf "%s tensorized structure as expected" tag) true)
   else
     let fp8_a ~l ~o ~i =
       NTDSL.init ~l ~prec:Ir.Ops.fp8 ?i ~o
         ~f:(fun idcs -> (Float.of_int (((idcs.(0) * n) + idcs.(1)) % 5) *. 0.5) -. 1.)
         ()
     in
     let fp8_b ~l ~o ~i =
       NTDSL.init ~l ~prec:Ir.Ops.fp8 ?i ~o
         ~f:(fun idcs -> (Float.of_int (((idcs.(0) * n) + idcs.(1)) % 7) -. 3.) *. 0.5)
         ()
     in
     let maf = fp8_a ~l:"maf" ~i:(Some [ n ]) ~o:[ n ] in
     let mbf = fp8_b ~l:"mbf" ~i:(Some [ n ]) ~o:[ n ] in
     let mtaf = fp8_a ~l:"mtaf" ~i:None ~o:[ n; n ] in
     let mtbf = fp8_b ~l:"mtbf" ~i:None ~o:[ n; n ] in
     let fp8_leg ~tag ~build =
       let mcf0 = build () in
       Tn.update_prec mcf0.Tensor.value Ir.Ops.single;
       let ctx_fs = Context.auto () in
       let ctx_fs, routine_fs =
         Context.compile
           ~lowered_transform:(fun opt -> opt)
           ctx_fs
           (named ("mm_" ^ tag ^ "_serial") (Train.forward mcf0))
           Ir.Indexing.Empty
       in
       let ctx_fs = Context.run ctx_fs routine_fs in
       let want = Context.get_values ctx_fs mcf0.Tensor.value in
       let mcf1 = build () in
       Tn.update_prec mcf1.Tensor.value Ir.Ops.single;
       let transform_f8 opt = Sched.apply (mma_schedule ~out:mcf1.Tensor.value opt) opt in
       let ctx_f8 = Context.auto () in
       let ctx_f8, routine_f8 =
         Context.compile ~lowered_transform:transform_f8 ctx_f8
           (named ("mm_" ^ tag ^ "_mma") (Train.forward mcf1))
           Ir.Indexing.Empty
       in
       let ctx_f8 = Context.run ctx_f8 routine_f8 in
       let got = Context.get_values ctx_f8 mcf1.Tensor.value in
       p
         (Printf.sprintf "%s tensorized matmul matches the serial twin bitwise" tag)
         (Array.for_all2_exn got want ~f:Float.equal);
       match read_generated ("mm_" ^ tag ^ "_mma") with
       | None -> p (Printf.sprintf "%s tensorized structure as expected" tag) false
       | Some src ->
           let has s = String.is_substring src ~substring:s in
           let ok =
             if on_gpu then
               (* CUDA sm_89+: the inline-PTX path; the lane-0 fallback on older devices. *)
               has "mma.sync.aligned.m16n8k32" || has "== 0)"
             else
               (* Fp8 declines the register tiling (single/double only): the scalar fallback. *)
               has "== 0)" && (not (has "simdgroup")) && not (has "Tile_mma register tiling")
           in
           p (Printf.sprintf "%s tensorized structure as expected" tag) ok
     in
     fp8_leg ~tag:"f8" ~build:(fun () ->
         let%op t = maf * mbf in
         t);
     fp8_leg ~tag:"f8_ta" ~build:(fun () ->
         let%op t = mtaf +* "ki;kj=>ij" mtbf in
         t);
     fp8_leg ~tag:"f8_tb" ~build:(fun () ->
         let%op t = mtaf +* "ik;jk=>ij" mtbf in
         t));

  (* --- Edge extents (gh-ocannl-469): a 7x19 output of a 7x13 by 13x19 matmul, tensorized over the
     whole triple. The register tiling covers the full 4x(RN*lanes) blocks and peels the partial row
     block and column strip into scalar loops; per-element chains stay serial-ordered and fused, so
     cc parity with the serial twin is BITWISE. Pinned on the C backends only: the GPU intrinsic
     paths decline non-multiple-of-tile extents by contract (already covered by the half case's
     fallback pin). --- *)
  if not on_gpu then (
    let mi = 7 and mk = 13 and mj = 19 in
    let eav = Array.init (mi * mk) ~f:(fun x -> Float.of_int (x % 11) *. 0.375) in
    let ebv = Array.init (mk * mj) ~f:(fun x -> Float.of_int (x % 9) -. 4.) in
    let ea = TDSL.ndarray eav ~label:[ "ea" ] ~input_dims:[ mk ] ~output_dims:[ mi ] () in
    let eb = TDSL.ndarray ebv ~label:[ "eb" ] ~input_dims:[ mj ] ~output_dims:[ mk ] () in
    let%op ec0 = ea * eb in
    let ctx_e0 = Context.auto () in
    let ctx_e0, routine_e0 =
      Context.compile
        ~lowered_transform:(fun opt -> opt)
        ctx_e0
        (named "mm_edge_serial" (Train.forward ec0))
        Ir.Indexing.Empty
    in
    let ctx_e0 = Context.run ctx_e0 routine_e0 in
    let got_edge_serial = Context.get_values ctx_e0 ec0.Tensor.value in
    let%op ec1 = ea * eb in
    let edge_schedule (opt : LL.optimized) : Sched.schedule =
      let paths = nest_paths opt.LL.llc in
      let i, j, k =
        match List.find_exn paths ~f:(fun p -> List.length p = 3) with
        | [ i; j; k ] -> (i, j, k)
        | _ -> assert false
      in
      (* The zeroing must cover the lane slot (validate_parallel's coverage rule), so its column
         loop becomes the Workgroup axis and the lane width matches its extent — arbitrary on the C
         backends, where the lane loop renders serially. *)
      let ez, zsyms = Sched.expand_zero ~tn:ec1.Tensor.value in
      let zj = match zsyms with [ _; zj ] -> zj | _ -> assert false in
      let rz = Sched.Retype { axis = zj; ty = LL.Workgroup } in
      let tz, _lane = Sched.tensorize ~i ~j ~k ~simd_width:mj in
      [ ez; rz; tz ]
    in
    let transform_e opt = Sched.apply (edge_schedule opt) opt in
    let ctx_e = Context.auto () in
    let ctx_e, routine_e =
      Context.compile ~lowered_transform:transform_e ctx_e
        (named "mm_edge_mma" (Train.forward ec1))
        Ir.Indexing.Empty
    in
    let ctx_e = Context.run ctx_e routine_e in
    let got_edge = Context.get_values ctx_e ec1.Tensor.value in
    p "edge-extent tensorized matmul matches the serial twin bitwise"
      (Array.for_all2_exn got_edge got_edge_serial ~f:Float.equal);
    match read_generated "mm_edge_mma" with
    | None -> p "edge-extent register tiling with peeled edges" false
    | Some src ->
        let has s = String.is_substring src ~substring:s in
        p "edge-extent register tiling with peeled edges"
          (has "Tile_mma register tiling" && has "full blocks 4x"))
  else (
    p "edge-extent tensorized matmul matches the serial twin bitwise" true;
    p "edge-extent register tiling with peeled edges" true);

  (* --- The staged + tensorized composition (lane-aware Stage): shared tiles for ma and mb,
     cooperatively loaded under fresh extent-32 Workgroup lane loops, then the micro-kernel
     tensorized. Loop order after the swaps is i_o(Grid) { k_o { i_i { j { k_i } } } }, so both
     stages anchor at k_o (loads + barriers per k-block) and Tensorize replaces the perfectly nested
     i_i x j x k_i micro-kernel reading the tiles. The ma tile's minor extent (8) is below the
     width: 8 lanes load under a folded-or-surviving [lane < 8] guard; the mb tile's minor extent
     (32) equals the width: the lane replaces the loop outright. GPU backends execute and must match
     the serial twin; the C backends cannot express shared placement and must reject cleanly (same
     pinning as the SMEM matmul test). --- *)
  let%op mc3 = ma * mb in
  let staged_schedule ~out ~src_a ~src_b (opt : LL.optimized) : Sched.schedule =
    let paths = nest_paths opt.LL.llc in
    let i, j, k =
      match List.find_exn paths ~f:(fun p -> List.length p = 3) with
      | [ i; j; k ] -> (i, j, k)
      | _ -> assert false
    in
    let ez, zsyms = Sched.expand_zero ~tn:out in
    let zi, zj = match zsyms with [ zi; zj ] -> (zi, zj) | _ -> assert false in
    let sp_zi, _, _ = Sched.split ~axis:zi ~factor:bm ~outer:LL.Grid ~inner:LL.Serial in
    let rz = Sched.Retype { axis = zj; ty = LL.Workgroup } in
    let sp_i, _, i_i = Sched.split ~axis:i ~factor:bm ~outer:LL.Grid ~inner:LL.Serial in
    let sp_k, k_o, k_i = Sched.split ~axis:k ~factor:bm ~outer:LL.Serial ~inner:LL.Serial in
    let tz, _lane = Sched.tensorize ~i:i_i ~j ~k:k_i ~simd_width in
    [
      ez;
      sp_zi;
      rz;
      sp_i;
      sp_k;
      Sched.Swap { outer = j; inner = k_o };
      Sched.Swap { outer = i_i; inner = k_o };
      Sched.Stage
        {
          source = src_a;
          tile_loops = [ i_i; k_i ];
          shared = true;
          cooperative = Some simd_width;
          hoisted = false;
          swizzle = None;
          pad_stride = None;
          pipeline_depth = 1;
        };
      Sched.Stage
        {
          source = src_b;
          tile_loops = [ k_i; j ];
          shared = true;
          cooperative = Some simd_width;
          hoisted = false;
          swizzle = None;
          pad_stride = None;
          pipeline_depth = 1;
        };
      tz;
    ]
  in
  let staged_comp = named "mm_staged_mma" (Train.forward mc3) in
  let staged_transform opt =
    Sched.apply
      (staged_schedule ~out:mc3.Tensor.value ~src_a:ma.Tensor.value ~src_b:mb.Tensor.value opt)
      opt
  in
  let ctx_c = Context.auto () in
  if on_gpu then (
    let ctx_c, routine_c =
      Context.compile ~lowered_transform:staged_transform ctx_c staged_comp Ir.Indexing.Empty
    in
    let ctx_c = Context.run ctx_c routine_c in
    let got_staged = Context.get_values ctx_c mc3.Tensor.value in
    p "staged+tensorized matmul parity (GPU) or clean rejection (CPU)"
      (Array.for_all2_exn got_staged got_serial ~f:approx);
    match read_generated "mm_staged_mma" with
    | None -> p "staged+tensorized structure as expected" false
    | Some src ->
        let has s = String.is_substring src ~substring:s in
        let count_sub sub =
          String.substr_index_all src ~may_overlap:false ~pattern:sub |> List.length
        in
        let ok =
          if on_metal then
            (* Two shared tiles cooperatively loaded, the mma reading them from threadgroup memory;
               the ma tile's partial-width load keeps its [lane < 8] guard. The marked accumulator
               fragment is loaded before the serial [k_o] body and stored after it; the body itself
               contains update-only MMA calls (gh-ocannl-480). *)
            let body_begin = "/* simdgroup fragment reduction body begins */" in
            let body_end = "/* simdgroup fragment reduction body ends */" in
            let fragment_load = "simdgroup_load(__mma_fragment_" in
            let fragment_store = "simdgroup_store(__mma_fragment_" in
            let resident =
              match
                ( String.substr_index src ~pattern:fragment_load,
                  String.substr_index src ~pattern:body_begin,
                  String.substr_index src ~pattern:body_end,
                  String.substr_index src ~pattern:fragment_store )
              with
              | Some load, Some beg, Some fin, Some store
                when load < beg && beg < fin && fin < store ->
                  let reduction_body = String.sub src ~pos:beg ~len:(fin - beg) in
                  let update = "/* tile_mma fragment update" in
                  let barrier = "threadgroup_barrier(mem_flags::mem_threadgroup);" in
                  let update_has_trailing_barrier =
                    match String.substr_index reduction_body ~pattern:update with
                    | Some update_pos ->
                        String.is_substring
                          (String.drop_prefix reduction_body update_pos)
                          ~substring:barrier
                    | None -> false
                  in
                  (not (String.is_substring reduction_body ~substring:fragment_load))
                  && (not (String.is_substring reduction_body ~substring:fragment_store))
                  && update_has_trailing_barrier
              | _ -> false
            in
            count_sub "threadgroup float tile_" = 2
            && has "simdgroup_multiply_accumulate"
            && has "threadgroup_barrier" && has "tile_mma fragment update" && resident
            && not (has "float fragment_")
          else
            (* CUDA: shared tiles; the wmma draft declines f32, so the lane-0 fallback guard. *)
            count_sub "__shared__ float tile_" = 2 && has "__syncthreads()"
        in
        p "staged+tensorized structure as expected" ok)
  else (
    (match
       try
         ignore
           (Context.compile ~lowered_transform:staged_transform ctx_c staged_comp Ir.Indexing.Empty
             : Context.t * Context.routine);
         None
       with Invalid_argument msg -> Some msg
     with
    | Some msg ->
        p "staged+tensorized matmul parity (GPU) or clean rejection (CPU)"
          (String.is_substring msg ~substring:"not supported")
    | None -> p "staged+tensorized matmul parity (GPU) or clean rejection (CPU)" false);
    p "staged+tensorized structure as expected" true);

  (* --- The staged + tensorized composition at half precision, accumulated in f32: the leg that
     pins the cross-[k_o] accumulator residency (gh-ocannl-480) on the tensor-core backends. The f32
     staged leg above cannot see it on CUDA/HIP (wmma has no uniform-f32 combination, so it declines
     to the scalar fallback); with f16 operands the marked accumulator renders as a fragment array
     loaded from [d] once before the serial [k_o] body and stored once after it, the body containing
     update-only MMA steps — Metal through [simdgroup_half8x8], CUDA through wmma accumulator
     fragments. The inputs are the exact-in-f16 values of the mm_h case, and every partial sum is
     also exact in f32, so parity with the half serial twin is bitwise on every backend and either
     rendering path. --- *)
  let%op mchs = mah * mbh in
  Tn.update_prec mchs.Tensor.value Ir.Ops.single;
  if on_gpu then (
    let transform_hs opt =
      Sched.apply
        (staged_schedule ~out:mchs.Tensor.value ~src_a:mah.Tensor.value ~src_b:mbh.Tensor.value opt)
        opt
    in
    let ctx_d = Context.auto () in
    let ctx_d, routine_d =
      Context.compile ~lowered_transform:transform_hs ctx_d
        (named "mm_h_staged_mma" (Train.forward mchs))
        Ir.Indexing.Empty
    in
    let ctx_d = Context.run ctx_d routine_d in
    let got_h_staged = Context.get_values ctx_d mchs.Tensor.value in
    (* Parity is bitwise on CUDA (wmma computes the exactly-rounded dot product for these f16-exact
       inputs) and on Metal (this mixed f16->f32 combination declines to the exact scalar fallback).
       On HIP it is only within f32 tolerance: RDNA3's [v_wmma_f32_16x16x16_f16] does not produce
       the exactly-rounded result (observed max abs diff ~1.3e-7), so the f32 accumulator differs
       from the f16 serial twin by rounding. The uniform-f16 leg below stays bitwise on every
       backend. *)
    let h32_eq =
      if String.is_substring backend_name ~substring:"hip" then approx else Float.equal
    in
    p "staged+tensorized half matmul matches the serial twin"
      (Array.for_all2_exn got_h_staged got_h_serial ~f:h32_eq);
    match read_generated "mm_h_staged_mma" with
    | None -> p "staged+tensorized half fragment residency" false
    | Some src ->
        let has s = String.is_substring src ~substring:s in
        (* f16 operands with an f32 accumulator: the wmma backends (HIP rocWMMA, CUDA wmma) render
           the marked accumulator as an f32 fragment array resident across [k_o]. Metal's
           [simdgroup_matrix] is uniform-precision only, so this mixed combination declines there to
           the scalar fallback — the uniform-f16 leg below is the one that exercises Metal's
           fragment path. HIP is verified on gfx1151, so its pin is strict; CUDA also accepts the
           pre-sm_70 lane-0 fallback. *)
        let ok =
          if on_metal then has "== 0)"
          else if String.is_substring backend_name ~substring:"hip" then staged_half_resident src
          else staged_half_resident src || has "== 0)"
        in
        p "staged+tensorized half fragment residency" ok)
  else (
    p "staged+tensorized half matmul matches the serial twin" true;
    p "staged+tensorized half fragment residency" true);

  (* --- The same staged half composition with a uniform-f16 accumulator (gh-ocannl-480): the leg
     that exercises the same-type accumulator fragment element (half, not f32) on every tensor-core
     backend — including Metal's [simdgroup_half8x8], which the mixed f16->f32 leg above cannot
     reach ([simdgroup_matrix] is uniform-precision only). Same f16-exact inputs, so parity with the
     half serial twin stays bitwise. --- *)
  let%op mchu = mah * mbh in
  Tn.update_prec mchu.Tensor.value Ir.Ops.half;
  if on_gpu then (
    let transform_hu opt =
      Sched.apply
        (staged_schedule ~out:mchu.Tensor.value ~src_a:mah.Tensor.value ~src_b:mbh.Tensor.value opt)
        opt
    in
    let ctx_u = Context.auto () in
    let ctx_u, routine_u =
      Context.compile ~lowered_transform:transform_hu ctx_u
        (named "mm_hu_staged_mma" (Train.forward mchu))
        Ir.Indexing.Empty
    in
    let ctx_u = Context.run ctx_u routine_u in
    let got_hu = Context.get_values ctx_u mchu.Tensor.value in
    p "staged+tensorized uniform-f16 matmul matches the serial twin bitwise"
      (Array.for_all2_exn got_hu got_h_serial ~f:Float.equal);
    match read_generated "mm_hu_staged_mma" with
    | None -> p "staged+tensorized uniform-f16 fragment residency" false
    | Some src ->
        let has s = String.is_substring src ~substring:s in
        (* Uniform f16->f16 is a valid combination on all three tensor-core backends, so the
           accumulator fragment stays resident across [k_o] on each. HIP strict (verified on
           gfx1151); Metal/CUDA also accept the pre-Apple7 / pre-sm_70 lane-0 fallback. *)
        let ok =
          if String.is_substring backend_name ~substring:"hip" then staged_half_resident src
          else staged_half_resident src || has "== 0)"
        in
        p "staged+tensorized uniform-f16 fragment residency" ok)
  else (
    p "staged+tensorized uniform-f16 matmul matches the serial twin bitwise" true;
    p "staged+tensorized uniform-f16 fragment residency" true);

  (* --- Transposed operand layouts (the gradient-GEMM access patterns): [d[i,j] += at[k,i] *
     b[k,j]] (a stored transposed) and [d[i,j] += a[i,k] * bt[j,k]] (b stored transposed). Tensorize
     infers the orientation from the index discipline and sets [Tile_mma.ta]/[tb]; on Metal the
     tiles load via [simdgroup_load]'s [transpose_matrix] flag with swapped offset arithmetic (no
     operand copy, tolerance parity — the tile reduction reassociates). On the C backends a
     transposed A costs nothing (the A feeds are scalar element splats either way), so the register
     tiling fires with swapped index arithmetic and parity stays BITWISE; a transposed B would turn
     the per-k row vector loads into strided gathers, so the register tiling declines and the scalar
     fallback keeps parity bitwise (a packing [Stage] with [tile_loops] in micro-kernel order
     normalizes the layout instead — see schedule_pack_mma_matmul.ml). CUDA's wmma draft declines
     uniform f32 regardless. --- *)
  let mtav = Array.init (n * n) ~f:(fun x -> Float.of_int (x % 7) *. 0.5) in
  let mtbv = Array.init (n * n) ~f:(fun x -> Float.of_int (x % 11) -. 5.) in
  let mta = TDSL.ndarray mtav ~label:[ "mta" ] ~output_dims:[ n; n ] () in
  let mtb = TDSL.ndarray mtbv ~label:[ "mtb" ] ~output_dims:[ n; n ] () in
  let check_transposed ~tag ~c_tiled ~serial ~tensorized =
    let serial_comp = named ("mm_" ^ tag ^ "_serial") (Train.forward serial) in
    let ctx0 = Context.auto () in
    let ctx0, routine0 =
      Context.compile ~lowered_transform:(fun opt -> opt) ctx0 serial_comp Ir.Indexing.Empty
    in
    let ctx0 = Context.run ctx0 routine0 in
    let want = Context.get_values ctx0 serial.Tensor.value in
    let mma_comp = named ("mm_" ^ tag ^ "_mma") (Train.forward tensorized) in
    let transform opt = Sched.apply (mma_schedule ~out:tensorized.Tensor.value opt) opt in
    let ctx1 = Context.auto () in
    let ctx1, routine1 =
      Context.compile ~lowered_transform:transform ctx1 mma_comp Ir.Indexing.Empty
    in
    let ctx1 = Context.run ctx1 routine1 in
    let got = Context.get_values ctx1 tensorized.Tensor.value in
    p
      (Printf.sprintf "%s tensorized matmul matches the serial twin" tag)
      (Array.for_all2_exn got want ~f:approx);
    p
      (Printf.sprintf "%s C-backend fallback matches bitwise" tag)
      (on_gpu || Array.for_all2_exn got want ~f:Float.equal);
    match read_generated ("mm_" ^ tag ^ "_mma") with
    | None -> p (Printf.sprintf "%s tensorized structure as expected" tag) false
    | Some src ->
        let has s = String.is_substring src ~substring:s in
        let ok =
          if on_metal then
            (* The intrinsic path with the transposing load; no lane-0 fallback guard. *)
            has "ulong2(0), true)" && has "simdgroup_multiply_accumulate" && not (has "== 0)")
          else if on_gpu then
            (* CUDA/HIP decline uniform f32 to the scalar fallback. *)
            has "== 0)" && not (has "Tile_mma register tiling")
          else if c_tiled then
            (* C backends, transposed A: the register tiling fires (lane-0 guarded) with the A index
               arithmetic swapped. *)
            has "== 0)" && has "Tile_mma register tiling"
          else
            (* C backends, transposed B: the register tiling declines to the scalar fallback. *)
            has "== 0)" && not (has "Tile_mma register tiling")
        in
        p (Printf.sprintf "%s tensorized structure as expected" tag) ok
  in
  let%op tc0 = mta +* "ki;kj=>ij" mtb in
  let%op tc1 = mta +* "ki;kj=>ij" mtb in
  check_transposed ~tag:"ta" ~c_tiled:true ~serial:tc0 ~tensorized:tc1;
  let%op td0 = mta +* "ik;jk=>ij" mtb in
  let%op td1 = mta +* "ik;jk=>ij" mtb in
  check_transposed ~tag:"tb" ~c_tiled:false ~serial:td0 ~tensorized:td1;

  (* --- Pattern discipline: Tensorize on a non-micro-kernel nest is a targeted error --- *)
  let%op mc2 = ma * mb in
  let bad_transform (opt : LL.optimized) : LL.optimized =
    let paths = nest_paths opt.LL.llc in
    let i, j, k =
      match List.find_exn paths ~f:(fun p -> List.length p = 3) with
      | [ i; j; k ] -> (i, j, k)
      | _ -> assert false
    in
    (* Roles misassigned: j as the row symbol of the accumulator fails the index discipline. *)
    let tz, _ = Sched.tensorize ~i:j ~j:i ~k ~simd_width in
    Sched.apply [ tz ] opt
  in
  let bad_comp = named "mm_bad" (Train.forward mc2) in
  match
    try
      ignore
        (Context.compile ~lowered_transform:bad_transform (Context.auto ()) bad_comp
           Ir.Indexing.Empty
          : Context.t * Context.routine);
      None
    with Invalid_argument msg -> Some msg
  with
  | Some msg ->
      p "misassigned roles are rejected with a targeted error"
        (String.is_substring msg ~substring:"Schedule.Tensorize")
  | None -> p "misassigned roles are rejected with a targeted error" false
