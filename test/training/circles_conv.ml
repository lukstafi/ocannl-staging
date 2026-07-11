(** Circle counting training test using synthetic dataset.

    This test trains a model to classify images by the number of circles they contain. Uses
    cross-entropy loss for classification.

    {2 Known Issues with conv2d in Training}

    When attempting to use [Nn_blocks.lenet] with SGD training, a shape inference issue was
    encountered:

    1. {b max_pool2d row variable mismatch}: [max_pool2d] uses [..c..] for channel row variable,
    while [conv2d] uses [..oc..] for output channels. When composing [max_pool2d (conv2d x)], the
    shape inference fails with "incompatible stride" errors because the row variables don't unify.

    {b Workaround}: Use an MLP instead - OCANNL's matrix multiplication handles multi-dimensional
    inputs automatically without explicit flattening. *)

open Base
open Ocannl
open Stdio
module Tn = Ir.Tnode
module IDX = Train.IDX
open Nn_blocks.DSL_modules
module Asgns = Ir.Assignments

let lenet = Nn_blocks.lenet
let softmax = Nn_blocks.softmax

let () =
  let seed = 42 in
  Utils.settings.fixed_state_for_init <- Some seed;
  Tensor.unsafe_reinitialize ();

  (* Use scaled initialization to prevent activation explosion. Default uniform1() in [0,1] causes
     logits to grow to millions. *)
  TDSL.default_param_init := NTDSL.xavier ~scale_sq:0.06 TDSL.O.uniform1;

  (* Configuration for circle dataset *)
  let image_size = 16 in
  let max_circles = 3 in
  let num_classes = max_circles in
  (* Classes: 1, 2, 3 circles -> indices 0, 1, 2 *)
  let config =
    Dataprep.Circles.Config.
      { image_size; max_radius = 4; min_radius = 2; max_circles; seed = Some seed }
  in

  (* Generate training data. Batch 32 over the same 160 images (5 batches instead of 20) keeps
     GPU wall time bounded: the fissioned sgd step has a near-constant per-step dispatch cost, so
     fewer, larger steps at the same samples-per-epoch are strictly Metal-friendlier (20k steps
     at batch 8 -> 5k steps at batch 32). *)
  let batch_size = 32 in
  let total_samples = batch_size * 5 in
  let n_batches = total_samples / batch_size in

  printf "Generating %d circle counting images (%d classes)...\n%!" total_samples num_classes;
  let images_data, labels_data =
    Dataprep.Circles.generate_single_prec ~config ~len:total_samples ()
  in
  printf "Dataset generated: images shape [%d; %d; %d; 1], labels shape [%d; 1]\n%!" total_samples
    image_size image_size total_samples;

  (* Convert labels to 0-based indices for one-hot encoding *)
  let labels_array = Bigarray.array2_of_genarray labels_data in
  let labels_list =
    List.init total_samples ~f:(fun i ->
        (* Labels are 1 to max_circles, convert to 0-based *)
        Int.of_float (Bigarray.Array2.get labels_array i 0) - 1)
  in

  (* Convert to tensors *)
  let images_ndarray = Ir.Ndarray.as_array Ir.Ops.Single images_data in
  let labels_one_hot = Nn_blocks.dense_one_hot_of_int_list ~num_classes labels_list in

  let batch_n, bindings = IDX.get_static_symbol ~static_range:n_batches IDX.empty in
  let step_n, bindings = IDX.get_static_symbol bindings in

  let images = TDSL.rebatch ~l:"images" images_ndarray () in

  (* Batch input/output *)
  let%op batch_images = images @| batch_n in
  let%op batch_labels = labels_one_hot @| batch_n in

  let%op logits = lenet () ~train_step:None batch_images in

  (* Softmax and cross-entropy loss *)
  let%op probs = softmax ~spec:"...|v" () logits in
  (* Sum the probability mass for the correct class *)
  let%op correct_prob = (probs *. batch_labels) ++ "...|... => ...|0" in
  (* Cross-entropy: -log(p) for each sample, then average *)
  let%op sample_loss = neg (log correct_prob) in
  let%op batch_loss = (sample_loss ++ "...|... => |->0") /. !..batch_size in

  (* Training setup *)
  let epochs = 1000 in
  let total_steps = epochs * n_batches in
  Train.every_non_literal_materialized batch_loss;
  (* Accumulate the loss on device so the training loop syncs on the host only once per epoch:
     a per-step [Context.get_values] awaits the whole device, serializing the stream. *)
  let loss_accum = Train.loss_accumulator () in
  let update = Train.grad_update ~accum_loss:loss_accum batch_loss in
  (* Mild lr scaling for the larger batch (0.01 at batch 8 -> 0.015 at batch 32), partially
     compensating the 4x fewer updates; this lenet destabilizes at 0.02+ (loss collapses to the
     uniform-prediction plateau). *)
  let%op learning_rate = 0.015 *. ((1.2 *. !..total_steps) - !@step_n) /. !..total_steps in
  Train.set_materialized learning_rate.value;
  let sgd = Train.sgd_update ~learning_rate batch_loss in

  (* Ensure we can read loss on host *)
  Train.set_materialized batch_loss.value;

  let ctx = Context.auto () in
  let ctx = Train.init_params ctx bindings batch_loss in
  (* Tune the step schedule empirically. [rounds:0] keeps only the preset seed candidates,
     which all preserve reduction order — the trained values are schedule-invariant, so this
     file's expected output stays deterministic no matter which seed wins. [timing_ctx] gives the
     tuner a scratch lineage (with its own freshly initialized parameter buffers) to time
     candidates against, so the timing runs cannot perturb the real training state (a step timed
     on all-zero data inputs poisons parameters with inf/NaN through log 0). *)
  let scratch = Train.init_params (Context.auto ()) bindings batch_loss in
  let ctx, sgd_routine = Autotune.tune ~rounds:0 ~timing_ctx:scratch ctx (Asgns.sequence [ update; sgd ]) bindings in
  let step_ref = IDX.find_exn (Context.bindings sgd_routine) step_n in
  step_ref := 0;

  printf "\nStarting training for %d epochs (%d steps)...\n%!" epochs total_steps;

  for epoch = 1 to epochs do
    Train.sequential_loop (Context.bindings sgd_routine) ~f:(fun () ->
        Train.run ctx sgd_routine;
        Int.incr step_ref);
    (* The only device sync of the epoch: read the accumulated loss sum, then reset it. *)
    let epoch_loss = ref (Context.get_values ctx loss_accum.Tensor.value).(0) in
    ignore (Context.set_values ctx loss_accum.Tensor.value [| 0. |] : Context.t);
    (* One decimal: cross-backend float drift over thousands of steps can flip the second
       decimal at a rounding boundary (cc vs metal differed at 0.215+-drift), and the expected
       output must stay byte-identical across backends. *)
    if epoch % 10 = 0 && (epoch <= 100 || epochs - epoch <= 100) then
      printf "Epoch %d: avg loss = %.1f\n%!" epoch (!epoch_loss /. Float.of_int n_batches);
    if epoch = epochs then
      printf "Final avg loss below threshold=%b\n%!"
        Float.(!epoch_loss /. Float.of_int n_batches < 0.3)
  done;

  printf "\nTraining complete!\n%!"
