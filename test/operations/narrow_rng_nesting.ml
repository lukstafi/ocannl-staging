(* The storage/compute precision split (gh-ocannl-517) must not reach an RNG conversion, even when
   one is buried inside an assignment rather than being the whole of it.

   [uint4x32_to_bfloat16_uniform_lane] is a different generator from
   [uint4x32_to_single_uniform_lane], not a rounding of it -- they consume different numbers of the
   128 random bits -- and which one is emitted follows the precision the *conversion* renders at,
   which [pp_scalar] inherits from its enclosing operator. A virtualized narrow uniform consumed by
   further arithmetic is the ordinary case, not an exotic one: the default centered-scaled parameter
   initializer has exactly this shape. Rendering its consumer wide would silently change every
   random draw, and would also break parity with the GPU backends and with the materialized
   [Set_from_vec] path.

   The assertion is that the values do not depend on the compute-precision policy at all. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module Tn = Ir.Tnode

let () = Utils.settings.output_debug_files_in_build_directory <- true
let p name b = Stdio.printf "%s: %b\n" name b

(* A narrow uniform consumed by further arithmetic: the uniform node is virtual, so its conversion
   ends up nested inside the consumer's expression rather than being the consumer's whole value. *)
let run ~prec =
  (* Fresh generation per leg: the threefry keys are self ids, so both legs must create their
     tensors in the same order for the draws to be comparable at all. *)
  Tensor.unsafe_reinitialize ();
  let u = TDSL.uniform () ~output_dims:[ 8 ] () in
  let%op y = u *. 2.0 in
  Tn.update_prec y.Tensor.value prec;
  Train.set_materialized y.Tensor.value;
  let ctx = Train.forward_once (Context.auto ()) y in
  Context.get_values ctx y.Tensor.value

let () =
  Tensor.unsafe_reinitialize ();
  let base = Ir.Numerics.get () in
  let leg policy prec =
    Ir.Numerics.set_policy policy;
    let v = run ~prec in
    Ir.Numerics.set_policy base;
    v
  in
  List.iter
    [ ("bf16", Ir.Ops.bfloat16); ("half", Ir.Ops.half) ]
    ~f:(fun (label, prec) ->
      let wide = leg { base with narrow_compute_f32 = true } prec in
      let per_op = leg { base with narrow_compute_f32 = false } prec in
      p (label ^ " nested uniform is independent of the compute-precision policy")
        (Array.length wide = 8 && Array.for_all2_exn wide per_op ~f:Float.equal);
      (* All eight draws distinct rules out the degenerate way the check above could pass. *)
      p (label ^ " draws are distinct")
        (Set.length (Set.of_array (module Float) wide) = 8))
