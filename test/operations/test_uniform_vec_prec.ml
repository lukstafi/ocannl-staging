open Base
open Ocannl.Nn_blocks.DSL_modules

(* Regression test for the vectorized uint4x32_to_prec_uniform builtins: the fp8 variant used to
   return uint8x16_t while codegen declared the result as int8x16_t (per vec_typ_of_prec), making
   any vectorized uniform generation at fp8 precision fail to compile on the cc backend with
   "invalid initializer". Exercises the vectorized path at all the small-integer-backed
   precisions. *)

let test_prec ~name prec =
  let ctx = Context.auto () in
  let%op random_bits = threefry4x32 (embed_self_id ()) 42L in
  let%op uniform_vals = uint4x32_to_prec_uniform random_bits in
  Ir.Tnode.update_prec uniform_vals.value prec;
  Ocannl.Train.set_materialized uniform_vals.value;
  let ctx = Ocannl.Train.forward_once ctx uniform_vals in
  let result = Context.get_values ctx uniform_vals.value in
  Stdio.printf "%s: generated %d values:\n" name (Array.length result);
  Array.iteri result ~f:(fun i x ->
      Stdio.printf "  [%d]: %s\n" i (Ir.Ndarray.concise_float ~prec:4 x))

let () =
  test_prec ~name:"fp8" Ir.Ops.fp8;
  test_prec ~name:"byte" Ir.Ops.byte;
  test_prec ~name:"int64" Ir.Ops.int64;
  test_prec ~name:"uint16" Ir.Ops.uint16
