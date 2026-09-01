(* A dispatch launches on the static index the caller had bound when it called [Context.run], not on
   whichever index the caller has moved on to by the time the device gets there.

   [Indexing.apply] dereferences a launch's static indices inside the kernel task, and the
   [multidev_cc] scheduler hands that task to a worker domain and returns; the caller's loop --
   [Train.sequential_loop], and every training loop -- rebinds for the next step immediately. If the
   worker read the caller's own ref it would launch on whatever the host had raced ahead to, which
   is a silent wrong answer rather than a slowdown: batches skipped and repeated, and any host
   schedule keyed on a static index (a learning rate on the step counter) read at the wrong point.

   The probe makes that observable without a device sync. It sweeps the index over its whole static
   range with NOTHING between the dispatch and the next rebind, accumulating the bound index and its
   square on device, then syncs once and compares both moments against their closed forms. Two
   moments rather than one because a single sum is blind to a swap; either alone already fails when
   the worker lags and re-reads late indices, which is how a training loop's batch sweep degenerates
   into re-training one batch. Both sums are integers below 2^24, so single precision holds them
   exactly and the claims can be equalities.

   It selects [multidev_cc] itself, which needs no hardware, so the claim is evaluated on every
   substrate rather than only where the configured backend happens to schedule asynchronously. *)

open Base
open Stdio
open Ocannl
module IDX = Train.IDX
open Nn_blocks.DSL_modules

let () =
  let n = 64 in
  let i, bindings = IDX.get_static_symbol ~static_range:n IDX.empty in
  let sum = Train.loss_accumulator ~label:"sum_of_index" () in
  let sum_sq = Train.loss_accumulator ~label:"sum_of_index_squared" () in
  let%cd probe =
    ~~("async launch bindings";
       sum =+ !@i;
       sum_sq =+ !@i *. !@i)
  in
  let ctx = Context.cpu ~threads:2 () in
  let ctx, routine = Context.compile ctx probe bindings in
  let idx = IDX.find_exn routine.Context.bindings i in
  let ctx = Context.set_values ctx sum.Tensor.value [| 0. |] in
  let ctx = Context.set_values ctx sum_sq.Tensor.value [| 0. |] in
  for k = 0 to n - 1 do
    idx := k;
    ignore (Context.run ctx routine : Context.t)
  done;
  Context.sync ctx;
  let got_sum = (Context.get_values ctx sum.Tensor.value).(0) in
  let got_sum_sq = (Context.get_values ctx sum_sq.Tensor.value).(0) in
  let want_sum = Float.of_int (n * (n - 1) / 2) in
  let want_sum_sq = Float.of_int (n * (n - 1) * ((2 * n) - 1) / 6) in
  eprintf
    "backend=%s, sum=%.1f (want %.1f), sum of squares=%.1f (want %.1f) (not part of the golden)\n%!"
    (Context.backend_name ctx) got_sum want_sum got_sum_sq want_sum_sq;
  Verdict.p "each dispatch launched on the index bound for it (sum over the sweep)"
    Float.(got_sum = want_sum);
  Verdict.p "each dispatch launched on the index bound for it (sum of squares over the sweep)"
    Float.(got_sum_sq = want_sum_sq)
