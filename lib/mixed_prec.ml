open Base
module Tn = Ir.Tnode
module Ops = Ir.Ops
module Asgns = Ir.Assignments
module Tensor = Ocannl_tensor.Tensor
module Operation = Ocannl_tensor.Operation
open Ocannl_tensor.Operation.DSL_modules

(** Mixed-precision training recipe (gh-ocannl-492): master weights via cast twins, and dynamic loss
    scaling. This is the torch-AMP / JAX-policy recipe translated to OCANNL's structures; it
    composes with {!Precision_policy} (storage-precision assignment over the rest of the graph) and
    with [Ir.Numerics] (compute-precision policy, gh-ocannl-478).

    The three pieces and how they fit together:

    - {b Master weights} ({!with_master_weights}): the parameter tensor node stays at the session
      default precision (typically f32) and remains the optimizer's target — the canonical value
      that SGD updates and {!Persistence} saves. The graph instead reads a reduced-precision "cast
      twin" ({!Operation.cast}) inserted by a {!Tensor.param_postprocess} hook at parameter
      construction. The twin is recomputed from the master each forward pass; whether it is a
      virtual node (recompute-on-read) or a materialized per-step copy is the [placement] choice.
      The twin's gradient flows back into the master's f32 gradient through an accumulating
      assignment, which is where the f16 -> f32 widening happens.
    - {b Loss scaling} ({!Loss_scaler}): f16's exponent range underflows on small gradients, so the
      backprop is seeded with a scale factor ([Train.grad_update ~loss_scale]) and gradients are
      unscaled in the optimizer step ([Train.sgd_update ~grad_unscale]). The dynamic schedule backs
      the scale off when non-finite gradients are detected ([Train.grad_checksum] read on the host
      between the two routines — see {!scaled_step}) and grows it after a run of good steps. bf16
      has f32's exponent range and skips loss scaling entirely.
    - {b Everything else} stays {!Precision_policy}'s job: apply a policy with [param_prec = None]
      (masters and twins already carry [Specified] precisions, so any [param_prec] would be a no-op
      on them anyway) to assign activation and gradient storage precisions, with the [except]
      predicate pinning precision-sensitive ops (softmax, norms, losses) at the default. *)

type twin_placement =
  | Twin_auto  (** Leave the twin's memory mode to the compiler (virtualization heuristics). *)
  | Twin_virtual  (** Recompute the cast at every read site. *)
  | Twin_materialized  (** A device-resident copy, refreshed once per step by the forward code. *)

(** [cast_param ~prec p] wraps a differentiable parameter [p] with a reduced-precision cast twin and
    returns the twin; non-differentiable parameters are returned unchanged. The master [p] and its
    gradient are pinned at [master_prec] (default [Ops.single]) — pinned, not left alone, because
    parameters are [top_down_prec]: the cast op registers an [Inferred] top-down update of the
    twin's precision into the master, and only a [Specified] precision overrides it (same finding as
    {!Precision_policy}'s [except] semantics). *)
let cast_param ?(placement = Twin_auto) ?(master_prec = Ops.single) ~prec (p : Tensor.t) =
  match p.Tensor.diff with
  | None -> p
  | Some diff ->
      let twin = Operation.cast ~grad_spec:Tensor.If_needed p () in
      Tn.update_prec twin.Tensor.value prec;
      Tn.update_prec p.Tensor.value master_prec;
      Tn.update_prec diff.grad master_prec;
      (match placement with
      | Twin_auto -> ()
      | Twin_virtual -> Train.set_virtual twin.Tensor.value
      | Twin_materialized -> Train.set_materialized twin.Tensor.value);
      twin

(** [with_master_weights ~prec f] runs the model-building thunk [f] with a
    {!Tensor.param_postprocess} hook installed that gives every differentiable parameter a
    [prec]-precision cast twin (see {!cast_param}). Wrap the model {e construction} — the [()]
    application that creates the inline parameters — not just the application to inputs. The
    optimizer path is unchanged: [loss.params] still contains the f32 masters, so
    [Train.sgd_update], [Train.init_params] and {!Persistence} all keep operating on the canonical
    full-precision values. *)
let with_master_weights ?placement ?master_prec ~prec f =
  let saved = !Tensor.param_postprocess in
  (Tensor.param_postprocess := fun p -> cast_param ?placement ?master_prec ~prec (saved p));
  Exn.protect ~f ~finally:(fun () -> Tensor.param_postprocess := saved)

(** Dynamic loss-scale state, host-managed: the scale (and its reciprocal) live in tiny
    device-resident tensors embedded in the compiled routines, and are overwritten via
    [Context.set_values] when the schedule changes the scale — no recompilation. The schedule is
    torch-AMP's: multiply the scale by [backoff_factor] whenever a step produced non-finite
    gradients (the step is skipped), and by [growth_factor] after [growth_interval] consecutive good
    steps. *)
module Loss_scaler = struct
  type t = {
    scale : Tensor.t;  (** Pass as [Train.grad_update ~loss_scale]. *)
    unscale : Tensor.t;  (** The reciprocal; pass as [Train.sgd_update ~grad_unscale]. *)
    mutable scale_val : float;
    mutable good_steps : int;
    growth_factor : float;
    backoff_factor : float;
    growth_interval : int;
  }

  (* A device-resident, broadcastable scalar the host can overwrite. Data-backed on purpose: a
     1-element [term_init] would come out as a [Constant] fetch, re-fetched by every step's forward
     code, silently undoing [Context.set_values]. The [bcast_if_1] axis basis (as in
     [Tensor.number]) lets the reciprocal broadcast into gradients of any shape. *)
  let host_scalar ~l v =
    let ndarray =
      Ir.Ndarray.init_array ~debug:l Ops.single ~dims:[| 1 |] ~padding:None ~f:(fun _ -> v)
    in
    let t =
      (* Reshape rather than Keep_shape_no_padding: the latter pins the data axis at the default
         basis, which conflicts with the [bcast_if_1] tag (bases are incompatible atoms). *)
      Tensor.term ~grad_spec:Tensor.Prohibit_grad ~init_data:(Asgns.Reshape ndarray) ~label:[ l ]
        ~batch_dims:[] ~input_dims:[]
        ~output_axes:[ (Row.bcast_if_1, 1) ]
        ()
    in
    Train.set_materialized t.Tensor.value;
    Tn.set_observable t.Tensor.value;
    t

  let create ?(init_scale = 65536.) ?(growth_factor = 2.) ?(backoff_factor = 0.5)
      ?(growth_interval = 200) () =
    let scale = host_scalar ~l:"loss_scale" init_scale in
    let unscale = host_scalar ~l:"grad_unscale" (1. /. init_scale) in
    {
      scale;
      unscale;
      scale_val = init_scale;
      good_steps = 0;
      growth_factor;
      backoff_factor;
      growth_interval;
    }

  let scale_value t = t.scale_val

  let set_scale t ctx v =
    t.scale_val <- v;
    let ctx = Context.set_values ctx t.scale.Tensor.value [| v |] in
    Context.set_values ctx t.unscale.Tensor.value [| 1. /. v |]

  let update t ctx ~grads_finite =
    if grads_finite then (
      t.good_steps <- t.good_steps + 1;
      if t.good_steps >= t.growth_interval then (
        t.good_steps <- 0;
        set_scale t ctx (t.scale_val *. t.growth_factor))
      else ctx)
    else (
      t.good_steps <- 0;
      set_scale t ctx (t.scale_val *. t.backoff_factor))
end

(** [scaled_grad_update scaler loss] is {!Train.grad_update} seeded with the scaler's scale,
    followed by the {!Train.grad_checksum} reduction, as one computation; returns the checksum flag
    tensor alongside. Parameter gradients are materialized here because the optimizer half
    ({!scaled_sgd_update}) is compiled as a separate routine — the host must read the checksum in
    between — and reads them across the routine boundary. *)
let scaled_grad_update ?setup_for_parallel ?accum_loss (scaler : Loss_scaler.t) loss =
  Set.iter loss.Tensor.params ~f:(fun p ->
      Train.set_materialized (Option.value_exn ~here:[%here] p.Tensor.diff).grad);
  let update =
    Train.grad_update ?setup_for_parallel ?accum_loss ~loss_scale:scaler.Loss_scaler.scale loss
  in
  let flag, check = Train.grad_checksum loss in
  (flag, Asgns.sequence [ update; check ])

(** {!Train.sgd_update} with the scaler's reciprocal as [grad_unscale]: gradients are unscaled in
    place before the optimizer math, so the momentum buffer and any later gradient reader see true
    gradient magnitudes. *)
let scaled_sgd_update (scaler : Loss_scaler.t) ~learning_rate ?momentum ?weight_decay ?nesterov loss
    =
  Train.sgd_update ~learning_rate ?momentum ?weight_decay ?nesterov
    ~grad_unscale:scaler.Loss_scaler.unscale loss

(** One dynamically-scaled training step: run the gradient routine (compiled from
    {!scaled_grad_update}'s computation), read the gradient checksum on the host, and either run the
    optimizer routine (compiled from {!scaled_sgd_update}'s) or — on non-finite gradients — skip it;
    then let the scaler adjust the scale. Returns whether the optimizer step ran. The per-step
    [Context.get_values] awaits the device, so this recipe trades stream overlap for the inf/nan
    gate — the same trade torch-AMP's unscale-and-check makes. *)
let scaled_step ~(scaler : Loss_scaler.t) ~grad_routine ~sgd_routine ~checksum ctx =
  let ctx = Context.run ctx grad_routine in
  let sum = (Context.get_values ctx checksum.Tensor.value).(0) in
  let grads_finite = Float.is_finite sum in
  let ctx = if grads_finite then Context.run ctx sgd_routine else ctx in
  let ctx = Loss_scaler.update scaler ctx ~grads_finite in
  (ctx, grads_finite)
