(* gh-555: smoke test for the greedy inlining-flip refinement of [Train.tune_placements].

   A matmul-plus-relu routine has a policy-virtual intermediate (the matmul result inlines into the
   pointwise consumer), so the winning arm's default compile reports at least one [`Materialize]
   flip candidate. With [~inline_flips:2] the driver runs the placement A/B, captures the decision
   surface, and searches the top flips — each search is observed through [?report], so the probe
   "at least three reports" (arm A, arm B, one-plus flips) pins that the refinement actually
   searched. The shipped routine must compute the same values as a plain compile.

   Printed facts are booleans so the expected output stays backend-stable. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules

let p name b = Stdio.printf "%s: %b\n" name b
let approx a b = Float.(abs (a -. b) < 1e-4)
let n = 8

let () =
  let mav = Array.init (n * n) ~f:(fun i -> Float.of_int (i % 7) *. 0.5) in
  let mbv = Array.init (n * n) ~f:(fun i -> Float.of_int (i % 5) -. 2.) in
  let ma = TDSL.ndarray mav ~label:[ "ma" ] ~input_dims:[ n ] ~output_dims:[ n ] () in
  let mb = TDSL.ndarray mbv ~label:[ "mb" ] ~input_dims:[ n ] ~output_dims:[ n ] () in
  let%op mc = ma * mb in
  let%op t2 = relu mc in
  ignore mc;
  let comp = Train.forward t2 in
  (* Reference values from a plain compile. *)
  let ctx_ref, routine_ref = Context.compile (Context.auto ()) comp Ir.Indexing.Empty in
  let ctx_ref = Context.run ctx_ref routine_ref in
  let expected = Context.get_values ctx_ref t2.Tensor.value in
  (* Tuned with the flip refinement enabled; cache disabled so the probes are run-independent. *)
  let reports = ref [] in
  let ctx_t, routine_t =
    Train.tune_placements ~beam_width:2 ~rounds:0 ~repeats:1 ~cache_dir:""
      ~report:(fun r -> reports := r :: !reports)
      ~inline_flips:2 (Context.auto ()) t2 comp Ir.Indexing.Empty
  in
  let ctx_t = Context.run ctx_t routine_t in
  let got = Context.get_values ctx_t t2.Tensor.value in
  p "tuned routine values match the plain compile" (Array.for_all2_exn got expected ~f:approx);
  p "flip refinement searched at least one flip beyond the A/B arms"
    (List.length !reports >= 3)
