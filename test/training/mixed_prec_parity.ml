(* Executed coverage for Ocannl.Mixed_prec (gh-ocannl-492 tasks 2 and 3): a two-layer MLP trained
   for a few SGD steps at the default f32 is the oracle for the same model (same initial weights,
   copied into the f32 masters via set_values) trained under master-weights cast twins, in bf16 (no
   loss scaling) and in f16 with dynamic loss scaling. Structure-only checks are not sufficient for
   optimizer-path changes (see the repo's testing notes), so the load-bearing assertions are
   executed loss trajectories; the precision printout pins the master-weights contract: masters and
   their gradient accumulators stay single, the graph-facing twins carry the reduced precision.

   The dynamic-scaling leg engineers a deterministic overflow: the gradient of the last-layer bias
   twin is exactly the loss scale (the loss is a plain sum), so with grad_prec = f16 an initial
   scale of 65536 (> 65504, the max finite f16) must produce a non-finite checksum on the first step
   — the step is skipped and the scale backs off to 32768, which is exactly representable. With
   growth_interval = 3, the scale then regrows after three good steps and overflows again, pinning
   the whole backoff/growth cycle. *)

open! Base
open Ocannl.Nn_blocks.DSL_modules
module Train = Ocannl.Train
module IDX = Train.IDX
module PP = Ocannl.Precision_policy
module MP = Ocannl.Mixed_prec
module Tn = Ir.Tnode
module Asgns = Ir.Assignments

let hid = 6
let out = 2
let x_vals = [| 0.5; -0.3; 0.9; 0.1 |]
let steps = 7
let lr = 0.05

let make_model () =
  let%op f x =
    ({ w2 } * relu (({ w1 } * x) + { b1 = 0.; o = [ hid ] })) + { b2 = 0.; o = [ out ] }
  in
  f

let find_param root name =
  List.find_exn (Set.to_list root.Tensor.params) ~f:(fun t ->
      String.equal (Tn.debug_name t.Tensor.value) name)

(* The cast twin is the (unique, in this model) direct consumer of a master parameter. *)
let find_consumer root master =
  let visited = Hashtbl.create (module Int) in
  let rec walk t =
    if not (Hashtbl.mem visited t.Tensor.value.Tn.id) then (
      Hashtbl.set visited ~key:t.Tensor.value.Tn.id ~data:();
      if
        List.exists t.Tensor.children ~f:(fun c ->
            c.Tensor.subtensor.Tensor.value.Tn.id = master.Tensor.value.Tn.id)
      then Some t
      else List.find_map t.Tensor.children ~f:(fun c -> walk c.Tensor.subtensor))
    else None
  in
  Option.value_exn ~here:[%here] (walk root)

let prec_str tn = Ir.Ops.prec_string (Lazy.force tn.Tn.storage_prec)

let build_leg ~input_l ~build ~prepare =
  let f = build () in
  let x =
    NTDSL.init ~l:input_l ~prec:Ir.Ops.single
      ~o:[ Array.length x_vals ]
      ~f:(function [| i |] -> x_vals.(i) | _ -> assert false)
      ()
  in
  let y = f x in
  let%op loss = y ++ "i=>0" in
  prepare loss;
  Train.set_materialized y.Tensor.value;
  (loss, y)

let learning_rate () = TDSL.number ~label:[ "lr" ] lr

(* Copy the oracle's initial weights into this leg's (master) parameters. *)
let copy_weights ctx loss = function
  | None -> ctx
  | Some (v1, v2) ->
      let w1 = find_param loss "w1" and w2 = find_param loss "w2" in
      let ctx = Context.set_values ctx w1.Tensor.value v1 in
      Context.set_values ctx w2.Tensor.value v2

(* Unscaled leg: forward+backprop+sgd as a single routine, [steps] iterations. *)
let run_plain base_ctx ~input_l ~build ~prepare ~w_vals =
  let loss, y = build_leg ~input_l ~build ~prepare in
  let update = Train.grad_update loss in
  let sgd = Train.sgd_update ~learning_rate:(learning_rate ()) loss in
  let ctx = Train.init_params base_ctx IDX.empty loss in
  let routine = Train.to_routine ctx IDX.empty (Asgns.sequence [ update; sgd ]) in
  let ctx = Context.context routine in
  let ctx = copy_weights ctx loss w_vals in
  let losses = ref [] in
  let ctx =
    Fn.apply_n_times ~n:steps
      (fun ctx ->
        let ctx = Context.run ctx routine in
        losses := (Context.get_values ctx loss.Tensor.value).(0) :: !losses;
        ctx)
      ctx
  in
  (ctx, loss, y, Array.of_list_rev !losses)

(* Dynamically-scaled leg: gradient routine and optimizer routine compiled separately, with the host
   reading the gradient checksum in between (Mixed_prec.scaled_step). *)
let run_scaled base_ctx ~input_l ~build ~prepare ~w_vals ~scaler =
  let loss, y = build_leg ~input_l ~build ~prepare in
  let checksum, grad_comp = MP.scaled_grad_update scaler loss in
  let sgd_comp = MP.scaled_sgd_update scaler ~learning_rate:(learning_rate ()) loss in
  let ctx = Train.init_params base_ctx IDX.empty loss in
  let grad_routine = Train.to_routine ctx IDX.empty grad_comp in
  let ctx = Context.context grad_routine in
  let sgd_routine = Train.to_routine ctx IDX.empty sgd_comp in
  let ctx = Context.context sgd_routine in
  let ctx = copy_weights ctx loss w_vals in
  let losses = ref [] and stepped = ref [] and scales = ref [] in
  let ctx =
    Fn.apply_n_times ~n:steps
      (fun ctx ->
        let ctx, ran = MP.scaled_step ~scaler ~grad_routine ~sgd_routine ~checksum ctx in
        losses := (Context.get_values ctx loss.Tensor.value).(0) :: !losses;
        stepped := ran :: !stepped;
        scales := MP.Loss_scaler.scale_value scaler :: !scales;
        ctx)
      ctx
  in
  (ctx, loss, y, Array.of_list_rev !losses, List.rev !stepped, List.rev !scales)

let max_abs_diff a b =
  Array.foldi a ~init:0. ~f:(fun i acc va -> Float.max acc (Float.abs (va -. b.(i))))

let () =
  Tensor.unsafe_reinitialize ();
  let base_ctx = Context.auto () in

  (* Init-only leg: recover deterministic initial weights that seed every trained leg. *)
  let loss_i, _y_i = build_leg ~input_l:"xi" ~build:make_model ~prepare:(fun _ -> ()) in
  let ctx_i = Train.init_params base_ctx IDX.empty loss_i in
  let w1_i = find_param loss_i "w1" and w2_i = find_param loss_i "w2" in
  let w_vals =
    Some (Context.get_values ctx_i w1_i.Tensor.value, Context.get_values ctx_i w2_i.Tensor.value)
  in

  (* Leg A, the oracle: default f32 everywhere. *)
  let _ctx_a, _loss_a, _y_a, losses_a =
    run_plain base_ctx ~input_l:"xa" ~build:make_model ~prepare:(fun _ -> ()) ~w_vals
  in
  Stdio.printf "oracle losses finite: %b\n" (Array.for_all losses_a ~f:Float.is_finite);

  (* Leg B: bf16 master weights, no loss scaling (bf16 has f32's exponent range). Masters and their
     gradient accumulators must settle at single; the graph-facing twins at bfloat16. *)
  let build_bf16 () = MP.with_master_weights ~prec:Ir.Ops.bfloat16 (fun () -> make_model ()) in
  let prepare_bf16 loss =
    PP.apply { PP.param_prec = None; activation_prec = Some Ir.Ops.bfloat16; grad_prec = None } loss
  in
  let _ctx_b, loss_b, y_b, losses_b =
    run_plain base_ctx ~input_l:"xb" ~build:build_bf16 ~prepare:prepare_bf16 ~w_vals
  in
  let w1_b = find_param loss_b "w1" in
  let twin_b = find_consumer loss_b w1_b in
  Stdio.printf "bf16 leg master w1: %s\n" (prec_str w1_b.Tensor.value);
  Stdio.printf "bf16 leg master w1 grad: %s\n"
    (prec_str (Option.value_exn w1_b.Tensor.diff).Tensor.grad);
  Stdio.printf "bf16 leg twin of w1: %s\n" (prec_str twin_b.Tensor.value);
  Stdio.printf "bf16 leg twin of w1 grad: %s\n"
    (prec_str (Option.value_exn twin_b.Tensor.diff).Tensor.grad);
  Stdio.printf "bf16 leg y activation: %s\n" (prec_str y_b.Tensor.value);
  Stdio.printf "bf16 leg loss trajectory parity within 0.1: %b\n"
    Float.(max_abs_diff losses_a losses_b < 0.1);

  (* Leg C: f16 master weights with dynamic loss scaling at a benign initial scale — every step
     runs, and the trajectory tracks the f32 oracle within f16 tolerance. Gradients are stored in
     f16 (grad_prec) except the pinned-single master accumulators, exercising the f16 -> f32
     widening at the twin-to-master gradient hand-off. *)
  let build_f16 () = MP.with_master_weights ~prec:Ir.Ops.half (fun () -> make_model ()) in
  let prepare_f16 loss =
    PP.apply
      { PP.param_prec = None; activation_prec = Some Ir.Ops.half; grad_prec = Some Ir.Ops.half }
      loss
  in
  let scaler_c = MP.Loss_scaler.create ~init_scale:8. ~growth_interval:100 () in
  let _ctx_c, loss_c, _y_c, losses_c, stepped_c, _scales_c =
    run_scaled base_ctx ~input_l:"xc" ~build:build_f16 ~prepare:prepare_f16 ~w_vals ~scaler:scaler_c
  in
  let w1_c = find_param loss_c "w1" in
  let twin_c = find_consumer loss_c w1_c in
  Stdio.printf "f16 leg master w1: %s\n" (prec_str w1_c.Tensor.value);
  Stdio.printf "f16 leg master w1 grad: %s\n"
    (prec_str (Option.value_exn w1_c.Tensor.diff).Tensor.grad);
  Stdio.printf "f16 leg twin of w1: %s\n" (prec_str twin_c.Tensor.value);
  Stdio.printf "f16 leg twin of w1 grad: %s\n"
    (prec_str (Option.value_exn twin_c.Tensor.diff).Tensor.grad);
  Stdio.printf "f16 leg all steps ran: %b\n" (List.for_all stepped_c ~f:Fn.id);
  Stdio.printf "f16 leg loss trajectory parity within 0.1: %b\n"
    Float.(max_abs_diff losses_a losses_c < 0.1);

  (* Leg D: the dynamic-scale backoff/growth cycle. The b2 twin gradient is exactly the scale, so
     init_scale 65536 > 65504 (max finite f16) overflows deterministically: step 1 skips and backs
     off to 32768; three good steps (growth_interval) regrow to 65536; step 5 overflows again. *)
  let scaler_d =
    MP.Loss_scaler.create ~init_scale:65536. ~growth_interval:3 ~growth_factor:2.
      ~backoff_factor:0.5 ()
  in
  let _ctx_d, _loss_d, _y_d, losses_d, stepped_d, scales_d =
    run_scaled base_ctx ~input_l:"xd" ~build:build_f16 ~prepare:prepare_f16 ~w_vals ~scaler:scaler_d
  in
  Stdio.printf "dynamic leg stepped flags: %s\n"
    (String.concat ~sep:" " (List.map stepped_d ~f:(fun b -> if b then "T" else "F")));
  Stdio.printf "dynamic leg scale after each step: %s\n"
    (String.concat ~sep:" " (List.map scales_d ~f:(fun s -> Printf.sprintf "%.0f" s)));
  Stdio.printf "dynamic leg losses on good steps finite: %b\n"
    (List.for_alli stepped_d ~f:(fun i ran -> (not ran) || Float.is_finite losses_d.(i)));
  Stdio.printf "%!"
