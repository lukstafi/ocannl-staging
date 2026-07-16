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

let () = Utils.settings.output_debug_files_in_build_directory <- true
let p name b = Stdio.printf "%s: %b\n" name b
let approx a b = Float.(abs (a - b) < 1e-2)
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
  let mma_schedule ~out (opt : LL.optimized) : Sched.schedule =
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

  (* --- Fp8 (e5m2) inputs accumulated in f32: the inline-PTX [mma.sync] path on CUDA sm_89+
     (tensorize-mma T3+; wmma cannot express fp8), the scalar fallback elsewhere. e5m2 has 2
     mantissa bits: inputs from {-1,-0.5,0,0.5,1} and {-1.5..1.5 step 0.5} are exact, every product
     is a multiple of 0.25 bounded by 1.5, and a 32-term f32 sum of such products is exact
     regardless of accumulation order — so parity is bitwise on every backend and either rendering
     path. Skipped on Metal, which has no fp8 storage precision at all ([typ_of_prec] rejects
     [Fp8_prec]) — even the serial twin cannot compile there. --- *)
  (if on_metal then (
     p "fp8 tensorized matmul matches the serial twin bitwise" true;
     p "fp8 tensorized structure as expected" true)
   else
     let maf =
       NTDSL.init ~l:"maf" ~prec:Ir.Ops.fp8 ~i:[ n ] ~o:[ n ]
         ~f:(fun idcs -> (Float.of_int (((idcs.(0) * n) + idcs.(1)) % 5) *. 0.5) -. 1.)
         ()
     in
     let mbf =
       NTDSL.init ~l:"mbf" ~prec:Ir.Ops.fp8 ~i:[ n ] ~o:[ n ]
         ~f:(fun idcs -> (Float.of_int (((idcs.(0) * n) + idcs.(1)) % 7) -. 3.) *. 0.5)
         ()
     in
     let%op mcf0 = maf * mbf in
     Tn.update_prec mcf0.Tensor.value Ir.Ops.single;
     let ctx_fs = Context.auto () in
     let ctx_fs, routine_fs =
       Context.compile
         ~lowered_transform:(fun opt -> opt)
         ctx_fs
         (named "mm_f8_serial" (Train.forward mcf0))
         Ir.Indexing.Empty
     in
     let ctx_fs = Context.run ctx_fs routine_fs in
     let got_f8_serial = Context.get_values ctx_fs mcf0.Tensor.value in
     let%op mcf1 = maf * mbf in
     Tn.update_prec mcf1.Tensor.value Ir.Ops.single;
     let transform_f8 opt = Sched.apply (mma_schedule ~out:mcf1.Tensor.value opt) opt in
     let ctx_f8 = Context.auto () in
     let ctx_f8, routine_f8 =
       Context.compile ~lowered_transform:transform_f8 ctx_f8
         (named "mm_f8_mma" (Train.forward mcf1))
         Ir.Indexing.Empty
     in
     let ctx_f8 = Context.run ctx_f8 routine_f8 in
     let got_f8 = Context.get_values ctx_f8 mcf1.Tensor.value in
     p "fp8 tensorized matmul matches the serial twin bitwise"
       (Array.for_all2_exn got_f8 got_f8_serial ~f:Float.equal);
     match read_generated "mm_f8_mma" with
     | None -> p "fp8 tensorized structure as expected" false
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
         p "fp8 tensorized structure as expected" ok);

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
  let staged_schedule (opt : LL.optimized) : Sched.schedule =
    let paths = nest_paths opt.LL.llc in
    let i, j, k =
      match List.find_exn paths ~f:(fun p -> List.length p = 3) with
      | [ i; j; k ] -> (i, j, k)
      | _ -> assert false
    in
    let ez, zsyms = Sched.expand_zero ~tn:mc3.Tensor.value in
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
          source = ma.Tensor.value;
          tile_loops = [ i_i; k_i ];
          shared = true;
          cooperative = Some simd_width;
          hoisted = false;
        };
      Sched.Stage
        {
          source = mb.Tensor.value;
          tile_loops = [ k_i; j ];
          shared = true;
          cooperative = Some simd_width;
          hoisted = false;
        };
      tz;
    ]
  in
  let staged_comp = named "mm_staged_mma" (Train.forward mc3) in
  let staged_transform opt = Sched.apply (staged_schedule opt) opt in
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
               the ma tile's partial-width load keeps its [lane < 8] guard. *)
            count_sub "threadgroup float tile_" = 2
            && has "simdgroup_multiply_accumulate"
            && has "threadgroup_barrier"
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
