(* Minimal repro attempt for the fused-segment cross-entropy corruption seen in bench_conv_diag on
   metal: replicate the CE cluster (logits matmul + bias, running max, log-sum-exp, scalar sum,
   normalization) without the conv tower. Scratch diagnostic. *)

open Base
open Ocannl
open Nn_blocks.DSL_modules

let () =
  let b = 8 and d = 84 and v = 10 in
  let xv = Array.init (b * d) ~f:(fun i -> Float.sin (Float.of_int i)) in
  let wv = Array.init (v * d) ~f:(fun i -> 0.1 *. Float.cos (Float.of_int i)) in
  let x = TDSL.ndarray xv ~label:[ "x" ] ~batch_dims:[ b ] ~output_dims:[ d ] () in
  let w = TDSL.ndarray wv ~label:[ "w" ] ~input_dims:[ d ] ~output_dims:[ v ] () in
  let labels = List.init b ~f:(fun i -> i % v) in
  let targets = Nn_blocks.dense_one_hot_of_int_list ~num_classes:v labels in
  let%op logits = (w * x) + 0. in
  let%op batch_loss =
    Nn_blocks.cross_entropy_loss ~spec:"...|v" ~normalize_by:!..b () ~logits ~targets
  in
  let comp = Train.forward batch_loss in
  let ctx = Context.auto () in
  let ctx, routine = Context.compile ctx comp Ir.Indexing.Empty in
  let ctx = Context.run ctx routine in
  (* Device-eye view of the one-hot: summed by a second routine from the device buffer. *)
  let%op oh_sum = targets ++ "b|c => 0" in
  let ctx, oh_routine = Context.compile ctx (Train.forward oh_sum) Ir.Indexing.Empty in
  let ctx = Context.run ctx oh_routine in
  let got = (Context.get_values ctx batch_loss.Tensor.value).(0) in
  (* Host oracle. *)
  let logit s c =
    let acc = ref 0. in
    for k = 0 to d - 1 do
      acc := !acc +. (wv.((c * d) + k) *. xv.((s * d) + k))
    done;
    !acc
  in
  let expected =
    let total = ref 0. in
    for s = 0 to b - 1 do
      let mx = ref Float.neg_infinity in
      for c = 0 to v - 1 do
        mx := Float.max !mx (logit s c)
      done;
      let sum = ref 0. in
      for c = 0 to v - 1 do
        sum := !sum +. Float.exp (logit s c -. !mx)
      done;
      total := !total +. (!mx +. Float.log !sum -. logit s (s % v))
    done;
    !total /. Float.of_int b
  in
  Stdio.printf "backend %s: got %.6f expected %.6f\n" (Context.backend_name ctx) got expected;
  Stdio.printf "device one_hot sum: %.1f (expected %d)\n"
    (Context.get_values ctx oh_sum.Tensor.value).(0)
    b;
  let ctx = Context.run ctx routine in
  Stdio.printf "second run: got %.6f\n" (Context.get_values ctx batch_loss.Tensor.value).(0)
