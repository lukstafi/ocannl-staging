(* Numeric check for Nn_blocks.gelu (tanh-approximate GeLU, the GPT-2 activation) against
   reference values computed with float64: gelu(x) = 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3))). *)

open Base
open Ocannl
open Stdio
open Nn_blocks.DSL_modules

let reference x =
  0.5 *. x *. (1.0 +. Float.tanh (0.7978845608028654 *. (x +. (0.044715 *. (x **. 3.)))))

let () =
  let inputs = [| -3.0; -2.0; -1.0; -0.5; 0.0; 0.5; 1.0; 2.0; 3.0 |] in
  let x = TDSL.ndarray inputs ~label:[ "x" ] ~batch_dims:[ Array.length inputs ] () in
  let y = Nn_blocks.gelu x in
  let ctx = Train.forward_once (Context.auto ()) y in
  let got = Context.get_values ctx y.Tensor.value in
  Array.iteri inputs ~f:(fun i v ->
      let expect = reference v in
      (* Single precision on device: compare to ~1e-6 absolute. *)
      let ok = Float.(abs (got.(i) - expect) < 1e-6) in
      printf "gelu(%+.1f) = %+.6f, reference %+.6f, ok: %b\n" v got.(i) expect ok)
