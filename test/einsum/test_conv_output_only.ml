open Ocannl
open Nn_blocks.DSL_modules

(* Regression test for gh-496: shape inference used to hang (non-terminating deferral loop in
   [Row.unify_row]) on a 1-D valid-convolution einsum over output-only axes. Expected behavior:
   output extent = input - kernel + 1 = 4, and result[o] = a[o] - a[o+1] = -1 for a = range 5,
   kernel = [1; -1]. *)
let () =
  Tensor.unsafe_reinitialize ();
  let ctx = Context.auto () in
  let a = TDSL.range 5 in
  let kernel = Tensor.term_init [| 1.0; -1.0 |] () in
  let%op result = a +* "o<+k; k => o" kernel in
  let ctx = Train.forward_once ctx result in
  Train.printf ~here:[%here] ~with_code:false ~with_grad:false ctx result
