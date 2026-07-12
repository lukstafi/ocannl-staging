(* Regression for the Metal shader-compiler miscompilation of constant-indexed scalar
   read-modify-write accumulations (the [volatile_scalar_rmw] workaround in c_syntax.ml; the raw
   compiler bug is reproduced standalone in benchmarks/runners/ocannl/bench_metal_bug.ml): a
   cross-entropy-style loss — per-sample log-sum-exp reduced into a scalar — must match the host
   oracle. Before the workaround, under the default GPU schedule the fissioned Metal kernel's
   [ce[0] = ce[0] + f(s)] loop kept only the last sample's contribution. *)

open Base
open Ocannl
open Nn_blocks.DSL_modules

let () =
  let b = 64 and d = 84 and v = 10 in
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
  let ctx = Context.auto () in
  let ctx, routine = Context.compile ctx (Train.forward batch_loss) Ir.Indexing.Empty in
  let ctx = Context.run ctx routine in
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
  Stdio.printf "scalar rmw accumulation matches oracle: %b\n"
    Float.(abs (got - expected) < 1e-4);
  (* Rerun: the re-zeroed accumulator must not drift. *)
  let ctx = Context.run ctx routine in
  let got2 = (Context.get_values ctx batch_loss.Tensor.value).(0) in
  Stdio.printf "rerun stable: %b\n" Float.(abs (got2 - got) < 1e-6)
