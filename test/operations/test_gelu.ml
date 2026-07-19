(* Numeric check for Nn_blocks.gelu (tanh-approximate GeLU, the GPT-2 activation) against reference
   values computed with float64: gelu(x) = 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3))).

   The [Tanh_approx] primitive may lower to a fast hardware approximation (CUDA's [__tanhf] with max
   relative error 2^-11, Metal's fast-math [tanh]), so the tolerance must cover that contract, not
   libm accuracy: through gelu the worst case for these inputs is ~0.5*|x|*2^-11 < 1e-3. The
   computed value is only printed on failure, keeping the expected output backend-independent. *)

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
      if Float.(abs (got.(i) - expect) < 1e-3) then
        printf "gelu(%+.1f) ~ reference %+.6f, ok: true\n" v expect
      else printf "gelu(%+.1f) = %+.6f, reference %+.6f, ok: false\n" v got.(i) expect)
