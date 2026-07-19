(* Swizzled shared-memory staging ([Stage ~swizzle], the bank-conflict follow-up of
   docs/proposals/tensorize-mma.md): the S2 shared-memory matmul schedule with both operand tiles
   stored XOR-swizzled, executed against the serial twin.

   [mc = ma * mb] (32x32 times 32x32) with the same schedule as schedule_smem_matmul.ml — Grid x
   Workgroup splits, [Stage ~shared ~swizzle] of both operands at 8x8 tiles, Privatize — except the
   tiles are marked in [optimized.swizzled]: codegen remaps each tile access [P*8 + col] to
   [P*8 + (col ^ (P & 7))], a per-row bijection of the minor axis, so the values are unchanged
   while same-column reads (the classic strided access of the staged A tile) spread across
   shared-memory banks. GPU backends execute and must match the serial twin; the C backends cannot
   express shared placement and must reject cleanly (same pinning as the SMEM matmul test).

   The staged+tensorized composition (lane-aware Stage feeding Tensorize) with swizzled tiles pins
   the decline path: the tile-MMA intrinsic and fragment renderings assume row-major pointer+stride
   operands, so swizzled operands must fall back to the lane-0 scalar micro-kernel — which reads
   elementwise through the swizzle-aware offsets and stays correct.

   The error legs pin [Schedule.Stage]'s swizzle validation on every backend: swizzle requires
   shared staging, a tile with at least two axes, and a power-of-two minor tile dim. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module LL = Ir.Low_level
module Sched = Ir.Schedule
module Asgns = Ir.Assignments

let () = Utils.settings.output_debug_files_in_build_directory <- true
let p name b = Stdio.printf "%s: %b\n" name b
let approx a b = Float.(abs (a - b) < 1e-3)
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

let accum_syms opt =
  let paths = nest_paths opt.LL.llc in
  match List.find_exn paths ~f:(fun p -> List.length p = 3) with
  | [ i; j; k ] -> (i, j, k)
  | _ -> assert false

let n = 32
let bm, bn, bk = (8, 8, 8)
let simd_width = 32

let () =
  let mav = Array.init (n * n) ~f:(fun i -> Float.of_int (i % 13) *. 0.25) in
  let mbv = Array.init (n * n) ~f:(fun i -> Float.of_int (i % 17) -. 8.) in
  let ma = TDSL.ndarray mav ~label:[ "ma" ] ~input_dims:[ n ] ~output_dims:[ n ] () in
  let mb = TDSL.ndarray mbv ~label:[ "mb" ] ~input_dims:[ n ] ~output_dims:[ n ] () in

  (* --- Serial twin --- *)
  let%op mc0 = ma * mb in
  let serial_comp = named "swz_mm_serial" (Train.forward mc0) in
  let ctx_s = Context.auto () in
  let ctx_s, routine_s =
    Context.compile ~lowered_transform:(fun opt -> opt) ctx_s serial_comp Ir.Indexing.Empty
  in
  let ctx_s = Context.run ctx_s routine_s in
  let got_serial = Context.get_values ctx_s mc0.Tensor.value in

  (* --- The swizzled SMEM schedule --- *)
  let%op mc1 = ma * mb in
  let smem_schedule (opt : LL.optimized) : Sched.schedule =
    let i, j, k = accum_syms opt in
    let ez, zsyms = Sched.expand_zero ~tn:mc1.Tensor.value in
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
          swizzle = true;
        };
      Sched.Stage
        {
          source = mb.Tensor.value;
          tile_loops = [ k_i; j_i ];
          shared = true;
          cooperative = None;
          hoisted = false;
          swizzle = true;
        };
      Sched.Privatize { target = mc1.Tensor.value; over = k_o };
    ]
  in
  let smem_comp = named "mm_swizzled_smem" (Train.forward mc1) in
  let transform opt = Sched.apply (smem_schedule opt) opt in
  let ctx_a = Context.auto () in
  if on_gpu then (
    let ctx_a, routine_a =
      Context.compile ~lowered_transform:transform ctx_a smem_comp Ir.Indexing.Empty
    in
    let ctx_a = Context.run ctx_a routine_a in
    let got_smem = Context.get_values ctx_a mc1.Tensor.value in
    p "swizzled SMEM matmul parity (GPU) or clean rejection (CPU)"
      (Array.for_all2_exn got_smem got_serial ~f:approx);
    match read_generated "mm_swizzled_smem" with
    | None -> p "tile accesses XOR-swizzled (GPU) or rejected (CPU)" false
    | Some src ->
        let has sub = String.is_substring src ~substring:sub in
        let count_sub sub =
          String.substr_index_all src ~may_overlap:false ~pattern:sub |> List.length
        in
        let shared_ok =
          if on_metal then count_sub "threadgroup float tile_" = 2 && has "threadgroup_barrier"
          else count_sub "__shared__ float tile_" = 2 && has "__syncthreads()"
        in
        (* Each tile is written by its cooperative load and read by the micro-kernel, all through
           the swizzled offset [P*8 + (col ^ (P & 7))]: at least 4 masked-XOR sites. *)
        p "tile accesses XOR-swizzled (GPU) or rejected (CPU)"
          (shared_ok && count_sub " & 7)" >= 4 && count_sub " ^ " >= 4))
  else (
    (match
       try
         ignore
           (Context.compile ~lowered_transform:transform ctx_a smem_comp Ir.Indexing.Empty
             : Context.t * Context.routine);
         None
       with Invalid_argument msg -> Some msg
     with
    | Some msg ->
        p "swizzled SMEM matmul parity (GPU) or clean rejection (CPU)"
          (String.is_substring msg ~substring:"not supported")
    | None -> p "swizzled SMEM matmul parity (GPU) or clean rejection (CPU)" false);
    p "tile accesses XOR-swizzled (GPU) or rejected (CPU)" true);

  (* --- Swizzled tiles feeding Tensorize: the intrinsic/fragment renderings assume row-major
     pointer+stride operands, so they must decline and the lane-0 scalar fallback must run — and
     stay correct, reading elementwise through the swizzled offsets. Same pipeline as the staged
     leg of schedule_mma_matmul.ml, with [swizzle = true] on both stages ([bm = 16] keeps the
     block extents intrinsic-sized, so a surviving intrinsic would fire — its absence below is the
     decline, not a shape accident). --- *)
  let%op mc2 = ma * mb in
  let bt = 16 in
  let staged_schedule (opt : LL.optimized) : Sched.schedule =
    let i, j, k = accum_syms opt in
    let ez, zsyms = Sched.expand_zero ~tn:mc2.Tensor.value in
    let zi, zj = match zsyms with [ zi; zj ] -> (zi, zj) | _ -> assert false in
    let sp_zi, _, _ = Sched.split ~axis:zi ~factor:bt ~outer:LL.Grid ~inner:LL.Serial in
    let rz = Sched.Retype { axis = zj; ty = LL.Workgroup } in
    let sp_i, _, i_i = Sched.split ~axis:i ~factor:bt ~outer:LL.Grid ~inner:LL.Serial in
    let sp_k, k_o, k_i = Sched.split ~axis:k ~factor:bt ~outer:LL.Serial ~inner:LL.Serial in
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
          swizzle = true;
        };
      Sched.Stage
        {
          source = mb.Tensor.value;
          tile_loops = [ k_i; j ];
          shared = true;
          cooperative = Some simd_width;
          hoisted = false;
          swizzle = true;
        };
      tz;
    ]
  in
  let staged_comp = named "mm_swizzled_mma" (Train.forward mc2) in
  let staged_transform opt = Sched.apply (staged_schedule opt) opt in
  let ctx_c = Context.auto () in
  if on_gpu then (
    let ctx_c, routine_c =
      Context.compile ~lowered_transform:staged_transform ctx_c staged_comp Ir.Indexing.Empty
    in
    let ctx_c = Context.run ctx_c routine_c in
    let got_staged = Context.get_values ctx_c mc2.Tensor.value in
    p "swizzled staged+tensorized parity (GPU) or clean rejection (CPU)"
      (Array.for_all2_exn got_staged got_serial ~f:approx);
    match read_generated "mm_swizzled_mma" with
    | None -> p "swizzled operands decline the MMA intrinsics to the lane-0 fallback" false
    | Some src ->
        let has sub = String.is_substring src ~substring:sub in
        let count_sub sub =
          String.substr_index_all src ~may_overlap:false ~pattern:sub |> List.length
        in
        let shared_ok =
          if on_metal then count_sub "threadgroup float tile_" = 2
          else count_sub "__shared__ float tile_" = 2
        in
        let no_intrinsics =
          if on_metal then
            (not (has "simdgroup_multiply_accumulate")) && not (has "simdgroup_load")
          else not (has "wmma")
        in
        (* The ma tile is [16 x 16] (mask 15), the mb tile [16 x 32] (mask 31); each is written by
           its cooperative load and read by the fallback micro-kernel: >= 4 XOR sites total. *)
        p "swizzled operands decline the MMA intrinsics to the lane-0 fallback"
          (shared_ok && no_intrinsics && has "== 0)" && count_sub " ^ " >= 4 && has " & 15)"
          && has " & 31)"))
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
        p "swizzled staged+tensorized parity (GPU) or clean rejection (CPU)"
          (String.is_substring msg ~substring:"not supported")
    | None -> p "swizzled staged+tensorized parity (GPU) or clean rejection (CPU)" false);
    p "swizzled operands decline the MMA intrinsics to the lane-0 fallback" true);

  (* --- Validation errors, uniform on every backend (raised by [Schedule.apply] before any
     backend-specific compilation) --- *)
  let expect_error name ~substring (mc : Tensor.t) schedule_of =
    let comp = named name (Train.forward mc) in
    let transform opt = Sched.apply (schedule_of opt) opt in
    let ctx = Context.auto () in
    match
      try
        ignore
          (Context.compile ~lowered_transform:transform ctx comp Ir.Indexing.Empty
            : Context.t * Context.routine);
        None
      with Invalid_argument msg -> Some msg
    with
    | Some msg -> p name (String.is_substring msg ~substring)
    | None -> p name false
  in
  let%op mc3 = ma * mb in
  expect_error "swizzle requires shared staging" ~substring:"requires shared" mc3 (fun opt ->
      let i, _j, k = accum_syms opt in
      let sp_i, _, i_i = Sched.split ~axis:i ~factor:bm ~outer:LL.Serial ~inner:LL.Serial in
      let sp_k, _, k_i = Sched.split ~axis:k ~factor:bk ~outer:LL.Serial ~inner:LL.Serial in
      [
        sp_i;
        sp_k;
        Sched.Stage
          {
            source = ma.Tensor.value;
            tile_loops = [ i_i; k_i ];
            shared = false;
            cooperative = None;
            hoisted = false;
            swizzle = true;
          };
      ]);
  let%op mc4 = ma * mb in
  expect_error "swizzle requires at least two tile axes" ~substring:"at least two axes" mc4
    (fun opt ->
      let _i, _j, k = accum_syms opt in
      let sp_k, _, k_i = Sched.split ~axis:k ~factor:bk ~outer:LL.Serial ~inner:LL.Serial in
      [
        sp_k;
        Sched.Stage
          {
            source = ma.Tensor.value;
            tile_loops = [ k_i ];
            shared = true;
            cooperative = None;
            hoisted = false;
            swizzle = true;
          };
      ]);
  (* A 24-wide matmul: splitting k by 12 gives a [8 x 12] tile whose minor dim is not a power of
     two (every divisor of 32 is, so the main tensors cannot produce this case). *)
  let n2 = 24 in
  let ma2v = Array.init (n2 * n2) ~f:(fun i -> Float.of_int (i % 7) *. 0.5) in
  let mb2v = Array.init (n2 * n2) ~f:(fun i -> Float.of_int (i % 11) -. 5.) in
  let ma2 = TDSL.ndarray ma2v ~label:[ "ma2" ] ~input_dims:[ n2 ] ~output_dims:[ n2 ] () in
  let mb2 = TDSL.ndarray mb2v ~label:[ "mb2" ] ~input_dims:[ n2 ] ~output_dims:[ n2 ] () in
  let%op mc5 = ma2 * mb2 in
  expect_error "swizzle requires a power-of-two minor tile dim" ~substring:"power-of-two" mc5
    (fun opt ->
      let i, _j, k = accum_syms opt in
      let sp_i, _, i_i = Sched.split ~axis:i ~factor:8 ~outer:LL.Serial ~inner:LL.Serial in
      let sp_k, _, k_i = Sched.split ~axis:k ~factor:12 ~outer:LL.Serial ~inner:LL.Serial in
      [
        sp_i;
        sp_k;
        Sched.Stage
          {
            source = ma2.Tensor.value;
            tile_loops = [ i_i; k_i ];
            shared = true;
            cooperative = None;
            hoisted = false;
            swizzle = true;
          };
      ])
