open Base
open Ocannl
open Ocannl.Nn_blocks.DSL_modules

(* gh-509 task 4: virtual packed [uniform] results are inlined per cell via the lane-extract form
   [vec_convert(counter[flat / lanes]).v[flat mod lanes]], and must match materialized runs
   bitwise. For each precision and shape we run the same program twice -- once with the uniform
   tensor forced materialized, once left to virtualize -- and compare the consumer's values
   bit-for-bit (NaN-safe: fp8 random bit patterns can be NaN). The generated source is also
   checked structurally: the materialized run stores via the vectorized builtin, the virtual run
   reads via the lane builtin and emits no vectorized store. (Test disabled on Metal only because
   the fp8 section needs FP8 support; the lane path itself is backend-generic.) *)

let () = Utils.settings.output_debug_files_in_build_directory <- true
let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")

let read_generated base_name =
  let ext = if String.is_substring backend_name ~substring:"metal" then ".metal" else ".c" in
  let ext = if String.is_substring backend_name ~substring:"cuda" then ".cu" else ext in
  let ext = if String.is_substring backend_name ~substring:"hip" then ".hip" else ext in
  let path = Utils.build_file (base_name ^ ext) in
  if Stdlib.Sys.file_exists path then Some (Stdio.In_channel.read_all path) else None

(* The consumer multiplies by 1: exact in every precision (including the fp8 float bridge), and
   reads each cell of [u] exactly once so [u] stays a virtualization candidate. Both runs create
   tensors in the same order after [unsafe_reinitialize], so the threefry keys (self ids) line
   up. *)
let run ~virtual_ ~prec ?input_dims output_dims =
  Tensor.unsafe_reinitialize ();
  let ctx = Context.auto () in
  let u = TDSL.uniform () ?input_dims ~output_dims () in
  Ir.Tnode.update_prec u.value prec;
  if not virtual_ then Train.set_materialized u.value;
  let%op uvl = u *. 1. in
  Ir.Tnode.update_prec uvl.value prec;
  let ctx = Train.forward_once ctx uvl in
  let values = Context.get_values ctx uvl.value in
  (values, read_generated "uvl_fwd")

let bits_equal a b = Int64.equal (Int64.bits_of_float a) (Int64.bits_of_float b)

(* Structural checks scan only the routine body: the prepended builtin definitions mention both
   the vectorized and the lane builtins (the lane builtins are implemented via the _vec ones). *)
let has sub = function
  | None -> false
  | Some s ->
      let body =
        match String.substr_index s ~pattern:"Main logic" with
        | Some i -> String.subo s ~pos:i
        | None -> s
      in
      String.is_substring body ~substring:sub

let check ~name ~prec ?input_dims output_dims =
  let ref_vals, ref_src = run ~virtual_:false ~prec ?input_dims output_dims in
  let vir_vals, vir_src = run ~virtual_:true ~prec ?input_dims output_dims in
  let parity =
    Array.length ref_vals = Array.length vir_vals
    && Array.for_alli vir_vals ~f:(fun i v -> bits_equal v ref_vals.(i))
  in
  Stdio.printf "%s: %d values, bitwise parity %b | materialized: vec store %b, lane %b | virtual: lane %b, vec store %b\n"
    name (Array.length ref_vals) parity
    (has "_uniform_vec(" ref_src)
    (has "_uniform_lane(" ref_src)
    (has "_uniform_lane(" vir_src)
    (has "_uniform_vec(" vir_src)

let () =
  check ~name:"single n=1 (lone partial block)" ~prec:Ir.Ops.single [ 1 ];
  check ~name:"single n=5 (tail peel)" ~prec:Ir.Ops.single [ 5 ];
  check ~name:"single n=8 (divisible)" ~prec:Ir.Ops.single [ 8 ];
  check ~name:"single 5->3 (multi-axis, 15 elements)" ~prec:Ir.Ops.single ~input_dims:[ 5 ] [ 3 ];
  check ~name:"half n=9" ~prec:Ir.Ops.half [ 9 ];
  check ~name:"double n=3" ~prec:Ir.Ops.double [ 3 ];
  (* fp8 is the stress case: 16 lanes per block, and random bit patterns include NaNs (compared by
     bits, not value). *)
  check ~name:"fp8 n=17" ~prec:Ir.Ops.fp8 [ 17 ]
