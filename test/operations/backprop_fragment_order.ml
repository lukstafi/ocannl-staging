(* Regression test for backprop-code fragment ordering (gh-461, gradient side): the fragment that
   embeds a shared node's backprop code must run after all fragments that accumulate into that
   node's gradient. Backprop ownership belongs to the first-constructed *differentiable* consumer,
   so when the shared node's *forward* is first consumed by a non-differentiable operation, the
   forward-fragment ordering gives no constraint and the id-order fallback can run the owner's
   fragment first (siblings combined right-to-left get smaller subtree ids).

   Here [l]'s forward is owned by the no-grad [s = l *. l], its backprop by [q]'s subtree; in [(qsum
   + ksum) *. ssum] the reduce of [k] is constructed before the reduce of [q], so without the
   backprop ordering pass [l]'s backprop (inside [qsum]'s fragment) ran before [ksum]'s contribution
   to [l]'s gradient, losing the [k] path: w.grad came out as wq*c*S instead of (wq + wk)*c*S. *)

open Base
open Ocannl
open Stdio
open Nn_blocks.DSL_modules

let () =
  let dm = 4 in
  let c = TDSL.ndarray [| 1.; 2.; 3.; 4. |] ~label:[ "c" ] ~output_dims:[ dm ] () in
  let w = TDSL.param ~values:[| 0.5; -1.; 2.; 3. |] "w" ~output_dims:[ dm ] () in
  let wq = TDSL.ndarray [| 1.; 2.; 1.; 2. |] ~label:[ "wq" ] ~output_dims:[ dm ] () in
  let wk = TDSL.ndarray [| 3.; 1.; 4.; 1. |] ~label:[ "wk" ] ~output_dims:[ dm ] () in
  let%op l = w *. c in
  (* No-grad consumer built first: owns l's *forward* code but not its backprop. *)
  let s = NTDSL.O.( *. ) l l in
  let%op q = wq *. l in
  let%op k = wk *. l in
  let%op loss = (q ++ "... => 0" + (k ++ "... => 0")) *. (s ++ "... => 0") in
  let grad = (Option.value_exn w.Tensor.diff).Tensor.grad in
  Train.set_materialized grad;
  let ctx = Train.update_once (Context.auto ()) loss in
  let got = Context.get_values ctx grad in
  (* loss = (sum (wq *. l) + sum (wk *. l)) * S with S = sum (l *. l) treated as a constant
     (prohibited gradient), l_i = w_i * c_i: dloss/dw_i = (wq_i + wk_i) * c_i * S. *)
  let lv = Array.init dm ~f:(fun i -> [| 0.5; -1.; 2.; 3. |].(i) *. [| 1.; 2.; 3.; 4. |].(i)) in
  let s_val = Array.fold lv ~init:0. ~f:(fun acc v -> acc +. (v *. v)) in
  let expected =
    Array.init dm ~f:(fun i ->
        ([| 1.; 2.; 1.; 2. |].(i) +. [| 3.; 1.; 4.; 1. |].(i)) *. [| 1.; 2.; 3.; 4. |].(i) *. s_val)
  in
  Array.iteri got ~f:(fun i g ->
      printf "w.grad[%d] = %.2f, expected %.2f, ok: %b\n" i g expected.(i)
        Float.(abs (g -. expected.(i)) < 1e-4 *. abs expected.(i)))
