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

  (* Generate training data. Batch 32 over the same 160 images (5 batches instead of 20) keeps GPU
     wall time bounded: the fissioned sgd step has a near-constant per-step dispatch cost, so fewer,
     larger steps at the same samples-per-epoch are strictly Metal-friendlier (20k steps at batch 8
     -> 3.75k steps at batch 32 with the shortened schedule below). *)
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
  (* The original 1,000-epoch schedule continued well past the point needed by this regression.
     Seven hundred and fifty epochs still take the same CNN from near-random loss (~1.06, and
     ln 3 = 1.10 is chance for the three classes) to a converged one, while removing 1,250
     expensive fissioned GPU steps. *)
  let epochs = 750 in
  let total_steps = epochs * n_batches in
  Train.every_non_literal_materialized batch_loss;
  (* Accumulate the loss on device so the training loop syncs on the host only once per epoch: a
     per-step [Context.get_values] awaits the whole device, serializing the stream. *)
  let loss_accum = Train.loss_accumulator () in
  let update = Train.grad_update ~accum_loss:loss_accum batch_loss in
  (* Mild lr scaling for the larger batch and shorter schedule partially compensates for the 4x
     fewer updates than the original batch-8 recipe. This lenet destabilizes at 0.02+ (loss
     collapses to the uniform-prediction plateau). *)
  let%op learning_rate = 0.018 *. ((1.2 *. !..total_steps) - !@step_n) /. !..total_steps in
  Train.set_materialized learning_rate.value;
  let sgd = Train.sgd_update ~learning_rate batch_loss in

  (* Ensure we can read loss on host *)
  Train.set_materialized batch_loss.value;

  let ctx = Context.auto () in
  let ctx = Train.init_params ctx bindings batch_loss in
  (* Tune the step schedule empirically. [rounds:0] keeps only the preset seed candidates, which all
     preserve reduction order — the trained values are schedule-invariant, so this file's expected
     output stays deterministic no matter which seed wins. The fixed, launch-bound graph gains
     nothing from also compiling the generic 64/128/256/512 GPU block-size sweep; the
     backend-default whole/fission presets remain in the search when [seed_block_sizes] is empty.
     [timing_ctx] gives the tuner a scratch lineage (with its own freshly initialized parameter
     buffers) to time candidates against, so the timing runs cannot perturb the real training state
     (a step timed on all-zero data inputs poisons parameters with inf/NaN through log 0). *)
  let scratch = Train.init_params (Context.auto ()) bindings batch_loss in
  let ctx, sgd_routine =
    Autotune.tune ~rounds:0 ~seed_block_sizes:[] ~timing_ctx:scratch ctx
      (Asgns.sequence [ update; sgd ])
      bindings
  in
  let step_ref = IDX.find_exn sgd_routine.Context.bindings step_n in
  step_ref := 0;

  printf "\nStarting training for %d epochs (%d steps)...\n%!" epochs total_steps;

  let logged_losses = ref [] in
  let final_avg = ref Float.infinity in
  (* Two-sided, because an upper bound alone is one-sided: if `neg (log correct_prob)` lost its
     negation, or a backend emitted the wrong sign, the trajectory would still fall and still land
     under the threshold, and only the digits this commit removed would have caught it (Codex
     round 1, P2).
     Checked on EVERY epoch, not just the logged ones -- the golden used to show a tenth of them. *)
  let all_valid = ref true in
  for epoch = 1 to epochs do
    Train.sequential_loop sgd_routine.Context.bindings ~f:(fun () ->
        Train.run ctx sgd_routine;
        Int.incr step_ref);
    (* The only device sync of the epoch: read the accumulated loss sum, then reset it. *)
    let epoch_loss = ref (Context.get_values ctx loss_accum.Tensor.value).(0) in
    ignore (Context.set_values ctx loss_accum.Tensor.value [| 0. |] : Context.t);
    (* Exact trajectory digits to stderr (gh-ocannl-725): cross-backend float drift over thousands
       of steps reaches whatever decimal a fixed precision prints, and lowering the precision only
       relocates the tie -- cifar_conv's %.1f epoch-30 mean landed on one (1.04 on cc, 1.05 on cuda)
       and no promotion could serve both backends. The portable stdout record is the pair of claims
       below: the loss fell across the logged epochs, and the final mean is under threshold. *)
    let avg = !epoch_loss /. Float.of_int n_batches in
    if not Float.(is_finite avg && avg >= -1e-3) then all_valid := false;
    if epoch % 10 = 0 && (epoch <= 100 || epochs - epoch <= 100) then (
      logged_losses := (epoch, avg) :: !logged_losses;
      eprintf "Epoch %d: avg loss = %.2f (not part of the golden)\n%!" epoch avg);
    if epoch = epochs then final_avg := avg
  done;
  (match List.rev !logged_losses with
  | [] -> ()
  | (first_epoch, first_loss) :: _ as logged ->
      let last_epoch, last_loss = List.last_exn logged in
      Verdict.pf "avg loss fell from epoch %d to epoch %d" first_epoch last_epoch
        Float.(last_loss < first_loss));
  Verdict.p "every epoch's avg loss is a finite, nonnegative cross-entropy" !all_valid;
  (* The threshold is 0.5, not the 0.3 it started at: the epoch-750 mean is ONE noisy sample, and
     0.3 left it no headroom on either kind of backend. The deterministic backends (cc, metal, cuda,
     hip) descend smoothly and land on exactly 0.29 -- three hundredths under the old bound, which
     their own epoch-650 value of 0.32 was still above, so they crossed 0.3 only in the last fifty
     epochs. multidev_cc converges lower but jitters epoch to epoch: five daily sweeps landed 0.18 / 0.13 / 0.13 / 0.11 / 0.32, the last of them a
     single-epoch excursion off a 0.14-0.19 tail (an equally large one at epoch 680 of an earlier
     run went unnoticed only because it was not the epoch sampled). 0.5 clears the worst observed
     sample by half again and still discriminates by a wide margin: an untrained or broken run sits
     near chance, and even epoch 100 is at 0.95 (cc) / 0.81 (multidev_cc). *)
  Verdict.p "Final avg loss below threshold" Float.(!final_avg < 0.5);

  printf "\nTraining complete!\n%!"
