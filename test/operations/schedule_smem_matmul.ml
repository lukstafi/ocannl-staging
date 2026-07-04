(* Schedule IR, Phase S2 (docs/proposals/schedule-ir-optops.md §5): the hand-written
   shared-memory matmul schedule, built from Split/Retype/Stage optops and executed against the
   serial twin.

   [mc = ma * mb] (32x32 times 32x32) lowers to a zeroing nest (2 loops) plus the naive triple
   loop. The schedule: split the zeroing nest's loops into Grid x Workgroup pairs (whole-node
   Zero_out is not distributed across threads, so the zeroing writes must cover the same hardware
   dimensions as the accumulation); split i and j of the accumulation by 8 into Grid x Workgroup,
   split k by 8 into Serial x Serial; then [Stage ma [i_i; k_i] ~shared] and
   [Stage mb [k_i; j_i] ~shared]. Stage inserts cooperative loads (reusing the Workgroup axis of
   each tile, iterating the serial k_i under a fresh symbol, restricting the redundant workgroup
   axis to thread 0) and barriers inside k_o; the 8x8 tiles become [threadgroup] / [__shared__]
   declarations via [optimized.workgroup_shared]. Tile sizes divide the extents, so Split's
   remainder guards and Stage's edge guards all fold away -- required: surviving whole-body
   guards would place the barriers under divergent control flow ([validate_parallel] rejects).

   On GPU backends (Metal locally, CUDA in CI) the kernel executes with a 4x4 grid of 8x8
   workgroups and must match the serial twin. The C backends cannot express shared placement;
   they must reject cleanly -- that rejection is what this test pins there. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module Tn = Ir.Tnode
module LL = Ir.Low_level
module Sched = Ir.Schedule
module Asgns = Ir.Assignments

let () = Utils.settings.output_debug_files_in_build_directory <- true
let p name b = Stdio.printf "%s: %b\n" name b
let approx a b = Float.(abs (a - b) < 1e-3)

let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"sync_cc")

let has_shared =
  String.is_substring backend_name ~substring:"metal"
  || String.is_substring backend_name ~substring:"cuda"

let read_generated base_name =
  let ext = if String.is_substring backend_name ~substring:"metal" then ".metal" else ".cu" in
  let path = Stdlib.Filename.concat "build_files" (base_name ^ ext) in
  if Stdlib.Sys.file_exists path then Some (Stdio.In_channel.read_all path) else None

(* The maximal single-child chains of statement-level loops: one symbol list per top-level nest. *)
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

let n = 32
let bm, bn, bk = (8, 8, 8)

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

  (* --- The SMEM schedule --- *)
  let%op mc1 = ma * mb in
  let smem_schedule (opt : LL.optimized) : Sched.schedule =
    let paths = nest_paths opt.LL.llc in
    (* The lowered code is [Zero_out mc1] (an opaque statement until Expand_zero) plus the naive
       triple loop -- the only loop nest. *)
    let accum = List.find_exn paths ~f:(fun p -> List.length p = 3) in
    let i, j, k = match accum with [ i; j; k ] -> (i, j, k) | _ -> assert false in
    let ez, zsyms = Sched.expand_zero ~tn:mc1.Tensor.value in
    let zi, zj = match zsyms with [ zi; zj ] -> (zi, zj) | _ -> assert false in
    let sp_zi, _, _ = Sched.split ~axis:zi ~factor:bm ~outer:LL.Grid ~inner:LL.Workgroup in
    let sp_zj, _, _ = Sched.split ~axis:zj ~factor:bn ~outer:LL.Grid ~inner:LL.Workgroup in
    let sp_i, _, i_i = Sched.split ~axis:i ~factor:bm ~outer:LL.Grid ~inner:LL.Workgroup in
    let sp_j, _, j_i = Sched.split ~axis:j ~factor:bn ~outer:LL.Grid ~inner:LL.Workgroup in
    let sp_k, _, k_i = Sched.split ~axis:k ~factor:bk ~outer:LL.Serial ~inner:LL.Serial in
    [
      ez;
      sp_zi;
      sp_zj;
      sp_i;
      sp_j;
      sp_k;
      Sched.Stage { source = ma.Tensor.value; tile_loops = [ i_i; k_i ]; shared = true };
      Sched.Stage { source = mb.Tensor.value; tile_loops = [ k_i; j_i ]; shared = true };
    ]
  in
  let smem_comp = named "mm_smem" (Train.forward mc1) in
  let transform opt = Sched.apply (smem_schedule opt) opt in
  let ctx_a = Context.auto () in
  if has_shared then (
    let ctx_a, routine_a =
      Context.compile ~lowered_transform:transform ctx_a smem_comp Ir.Indexing.Empty
    in
    let ctx_a = Context.run ctx_a routine_a in
    let got_smem = Context.get_values ctx_a mc1.Tensor.value in
    p "SMEM matmul parity (GPU) or clean rejection (CPU)"
      (Array.for_all2_exn got_smem got_serial ~f:approx);
    match read_generated "mm_smem" with
    | None -> p "shared tiles, barriers, no edge guards (GPU) or rejected (CPU)" false
    | Some src ->
        let has sub = String.is_substring src ~substring:sub in
        let count_sub sub = String.substr_index_all src ~may_overlap:false ~pattern:sub |> List.length in
        let shared_ok =
          if String.is_substring backend_name ~substring:"metal" then
            count_sub "threadgroup float tile_" = 2 && has "threadgroup_barrier"
          else count_sub "__shared__ float tile_" = 2 && has "__syncthreads()"
        in
        (* All tile sizes divide the extents: Split remainder guards and Stage edge guards fold;
           the only Ifs left are the two thread-0 load restrictions. *)
        p "shared tiles, barriers, no edge guards (GPU) or rejected (CPU)"
          (shared_ok && count_sub "if (" = 2))
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
        p "SMEM matmul parity (GPU) or clean rejection (CPU)"
          (String.is_substring msg ~substring:"not supported")
    | None -> p "SMEM matmul parity (GPU) or clean rejection (CPU)" false);
    p "shared tiles, barriers, no edge guards (GPU) or rejected (CPU)" true)
