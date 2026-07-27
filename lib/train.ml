open Base
module Ops = Ir.Ops
module Tn = Ir.Tnode
module Nd = Ir.Ndarray
module Asgns = Ir.Assignments
module Idx = Ir.Indexing
module Task = Ir.Task
open Ocannl_tensor.Operation.DSL_modules

let _get_local_debug_runtime = Utils.get_local_debug_runtime

[%%global_debug_log_level 0]

(* export OCANNL_LOG_LEVEL_TRAIN=9 to enable debugging into the log_files/ directory. *)
[%%global_debug_log_level_from_env_var "OCANNL_LOG_LEVEL_TRAIN"]

module CDSL = struct
  let half = Ir.Ops.half
  let single = Ir.Ops.single
  let double = Ir.Ops.double
  let virtualize_settings = Ir.Low_level.virtualize_settings

  let enable_all_debugs ?(debug_logs = false) ?(hosted_only = true) () =
    Utils.set_log_level @@ max 2 @@ Utils.settings.log_level;
    Utils.settings.output_debug_files_in_build_directory <- true;
    if hosted_only then virtualize_settings.enable_device_only <- false;
    if debug_logs then Utils.settings.debug_log_from_routines <- true

  let disable_all_debugs ?(restore_defaults = false) () =
    Utils.settings.debug_log_from_routines <- false;
    Utils.set_log_level 0;
    Utils.settings.output_debug_files_in_build_directory <- false;
    if restore_defaults then virtualize_settings.enable_device_only <- true
end

module IDX = struct
  let empty = Idx.Empty
  let get_static_symbol = Idx.get_static_symbol
  let find_exn = Idx.find_exn
end

let run ctx routine = ignore (Context.run ctx routine)

(* Parameter persistence now lives in {!Persistence} (gh-ocannl-373) and is context-mediated; the
   old hosted-array-based save/restore helpers were removed with the hosted memory mode
   (gh-ocannl-333). *)

let set_materialized (a : Tn.t) = Tn.update_memory_mode a On_device 28

(** Sets the tensor's value as materialized (device-resident, inspectable on demand via the
    context), and returns the tensor's forward code with a label-derived comment. *)
let forward t =
  let fwd = Tensor.consume_forward_code t in
  set_materialized t.Tensor.value;
  let label = Tn.debug_name t.value in
  { fwd with asgns = Asgns.Block_comment (label ^ " fwd", fwd.asgns) }

(** A scalar non-differentiable accumulator for {!grad_update}'s [?accum_loss]: zero-initialized at
    allocation and materialized. Read it with [Context.get_values] (which awaits the device) and
    reset it with [Context.set_values ctx t.value [| 0. |]] — e.g. once per epoch. *)
let loss_accumulator ?(label = "loss_accum") () =
  let t = NTDSL.init ~l:label ~prec:Ir.Ops.single ~o:[ 1 ] ~f:(fun _ -> 0.) () in
  set_materialized t.Tensor.value;
  t

(** Returns the tensor's forward, zeroing gradients, and backprop code wrapped with label-derived
    comments. Sets the tensor's value as materialized. If [setup_for_parallel] is true (false by
    default), sets the parameters and their gradients as "non-local" (on-device). When [accum_loss]
    is given (see {!loss_accumulator}), the update also accumulates the loss value into it
    ([accum_loss =+ loss]): training loops can then read the loss sum once per epoch instead of once
    per step — on GPU backends a per-step [Context.get_values] awaits the whole device, serializing
    the stream, while steps that only accumulate on device queue up and overlap with host-side
    scheduling. When [loss_scale] is given (see {!Mixed_prec.Loss_scaler}), the backprop is seeded
    with the scale's value instead of 1 ([loss.grad =: loss_scale]), so all gradients come out
    multiplied by the scale — unscale them before the optimizer update (the [grad_unscale] argument
    of {!sgd_update}). *)
let grad_update ?(setup_for_parallel = false) ?accum_loss ?loss_scale loss =
  set_materialized loss.Tensor.value;
  (* Training loops read the loss from the host; declare the intent so the liveness memory planner
     (config [buffer_aliasing], gh-ocannl-489) never aliases the loss buffer -- like param
     gradients' observation intent declared in [Tensor.param]. *)
  Tn.set_observable loss.Tensor.value;
  if setup_for_parallel then
    Set.iter loss.Tensor.params ~f:(fun p ->
        set_materialized (Option.value_exn ~here:[%here] p.diff).grad);
  (* Note: the %cd syntax for [loss.grad] does not modify roots. *)
  [%cd
    ~~(loss "forward and gradient update";
       (* In the accumulating branch, referencing [loss] embeds its forward code (the single
          consumption of it), so the one statement computes the loss and accumulates it. *)
       (match accum_loss with
       | Some acc -> acc =+ loss
       | None -> loss.forward);
       ~~(loss "zero grads and backprop";
          loss.zero_grads;
          (match loss_scale with
          | Some scale -> loss.grad =: scale
          | None -> loss.grad =: 1);
          loss.backprop))]

(** A scalar checksum over all parameter gradients of [loss]: returns the flag tensor and the code
    that resets it to 0 and accumulates the sum of every gradient cell into it. The sum is
    non-finite if and only if some gradient cell is non-finite (a finite sum cannot arise from
    non-finite cells: same-sign infinities stay infinite, opposite-sign infinities and NaNs produce
    NaN; a spurious overflow of large finite gradients only triggers a benign extra backoff).
    Sequence it after {!grad_update} in the same routine, read the flag with [Context.get_values]
    and gate the optimizer step on [Float.is_finite] — the dynamic loss scaling recipe
    ({!Mixed_prec.step}) does exactly this. *)
let grad_checksum loss =
  let flag = NTDSL.init ~l:"grad_checksum" ~prec:Ir.Ops.single ~o:[ 1 ] ~f:(fun _ -> 0.) () in
  set_materialized flag.Tensor.value;
  Tn.set_observable flag.Tensor.value;
  (* Settle shape inference for the parameters first: the total-reduce einsum below unifies each
     parameter's rows with the spec's row variables, and a parameter row still unsolved at
     settlement would then be refused the close-to-empty guess (row.ml's "You forgot to specify
     the hidden dimension(s)" — a row variable used in an einsum spec is no longer safe to guess).
     Forcing dims here closes e.g. a bias's inferred-empty input row before the spec touches it —
     which is also why [grad_checksum] must be called only after the model and the loss are fully
     constructed. *)
  Set.iter loss.Tensor.params ~f:(fun p ->
      ignore (Lazy.force p.Tensor.value.Tn.dims : int array));
  let one_param p =
    if Option.is_none p.Tensor.diff then
      raise @@ Tensor.Session_error ("Train.grad_checksum: not differentiable", Some p);
    [%cd flag =+ id p.grad ~logic:"...|...->... => |->0"]
  in
  let comps = Set.to_list loss.Tensor.params |> List.map ~f:one_param in
  let reset = [%cd flag =: 0] in
  let comp = Asgns.sequence (reset :: comps) in
  (flag, { comp with asgns = Asgns.Block_comment ("grad_checksum", comp.asgns) })

(** See: https://github.com/tinygrad/tinygrad/blob/master/tinygrad/nn/optim.py

    When [grad_unscale] is given (the reciprocal of {!grad_update}'s [loss_scale]), the gradient is
    first multiplied in place by it, so the optimizer math below — including the momentum buffer —
    sees unscaled gradients, and so does any later reader of [p.grad] (e.g. gradient clipping). *)
let sgd_one ~learning_rate ?(momentum = 0.0) ?(weight_decay = 0.0) ?(nesterov = false) ?grad_unscale
    p =
  if Option.is_none p.Tensor.diff then
    raise @@ Tensor.Session_error ("Train.sgd_one: not differentiable", Some p);
  [%cd
    ~~(p "param sgd step";
       (match grad_unscale with
       (* The binary form: a unary [p.grad =* unscale] would be a Pointwise_un, which does not
          broadcast the scalar's closed rows against parameters with input axes. *)
       | Some unscale -> p.grad =: p.grad * unscale ~logic:"."
       | None -> Asgns.empty_comp);
       { sgd_delta } =: p.grad + (!.weight_decay *. p);
       if Float.(momentum > 0.0) then (
         { sgd_momentum } =: (!.momentum *. sgd_momentum) + sgd_delta;
         if nesterov then sgd_delta =+ !.momentum *. sgd_momentum else sgd_delta =: sgd_momentum);
       p =- learning_rate * sgd_delta ~logic:".")]

let sgd_update ~learning_rate ?momentum ?weight_decay ?nesterov ?grad_unscale loss =
  let f = sgd_one ~learning_rate ?momentum ?weight_decay ?nesterov ?grad_unscale in
  let comp = Set.to_list loss.Tensor.params |> List.map ~f |> Asgns.sequence in
  { comp with asgns = Asgns.Block_comment ("sgd_update", comp.asgns) }

(** All and only bindings with associated ranges are iterated, with the binding's initial value
    lost. Bindings without ranges remain at their initial values, as do symbolic extents (gh-490):
    an extent is a size set once by the user, not an index to iterate. *)
let%track3_sexp sequential_loop ~f lowered_bindings =
  let rec loop = function
    | [] -> f ()
    | ({ Idx.static_range = None; static_symbol = _; _ }, _) :: more -> loop more
    | ({ Idx.used_as_extent = true; _ }, _) :: more -> loop more
    | ({ Idx.static_range = Some range; static_symbol = _; _ }, idx) :: more ->
        let old_idx = !idx in
        for i = 0 to range - 1 do
          idx := i;
          loop more
        done;
        idx := old_idx
  in
  loop lowered_bindings

let set_virtual (a : Tn.t) = Tn.update_memory_mode a Virtual 29

(** Materializes every non-literal embedded tensor node of [t] (so its value is inspectable on
    demand via the context). Replaces the old [every_non_literal_on_host] now that there is no
    hosted memory mode (gh-ocannl-333). *)
let every_non_literal_materialized =
  Tensor.iter_embedded ~f:(fun a ->
      if Tn.mode_is_unspecified a && not (Tn.known_constant a) then set_materialized a)

(** Placement A/B autotuning: {!Autotune.tune} on [comp] under the graph's current (default)
    placements — virtual intermediates plus the compiler's promotions — and again with every
    embedded node of [loss] materialized, keeping the measured winner (the arms' [best_ms] are
    min-of-N timings on the same device, so directly comparable). By construction the result is at
    least as fast as the better of the default and materialize-all placements, whichever the search
    would find; this generalizes the old "materialize everything before tuning" recipe instead of
    replacing one fixed placement policy with another. Respecting the two-level memory-mode split
    (docs/proposals/context-scoped-memory-modes.md) — tnode-level [memory_mode] is declared,
    semantics-bearing intent, while placement {e decisions} are context-level and functional — the B
    arm does not touch intent: it tunes from {!Context.decide_materialized} siblings of [ctx] (and
    of [timing_ctx]), so the arms are hermetic and [tune_placements] leaves no trace on the graph or
    on the caller's contexts beyond the returned winner. See
    test/operations/materialize_after_compile.ml. [report], when given, observes both arms' reports
    in order. Other arguments are forwarded to {!Autotune.tune}; the same caveats apply (notably
    [timing_ctx] and non-idempotent routines — both arms share [timing_ctx]'s device for their
    searches). *)
let tune_placements ?beam_width ?rounds ?repeats ?timing_ctx ?report ctx loss comp bindings =
  (* Arm attribution on the same stderr trace as Autotune's config [autotune_log] — winner-arm
     ambiguity misdirected the CUDA benchmark debugging on PR #140. *)
  let log_arms =
    match
      String.lowercase
        (String.strip (Utils.get_global_arg ~arg_name:"autotune_log" ~default:"false"))
    with
    | "true" | "1" -> true
    | _ -> false
  in
  let logf fmt =
    Stdlib.Printf.ksprintf (fun s -> if log_arms then Stdio.eprintf "tune_placements: %s\n%!" s) fmt
  in
  let best_ms = ref Float.infinity in
  let capture r =
    best_ms := r.Autotune.best_ms;
    Option.iter report ~f:(fun f -> f r)
  in
  let tune arm ctx timing_ctx =
    logf "arm %s search:" arm;
    best_ms := Float.infinity;
    let result =
      Autotune.tune ?beam_width ?rounds ?repeats ?timing_ctx ~report:capture ctx comp bindings
    in
    logf "arm %s best: %.4f ms" arm !best_ms;
    (result, !best_ms)
  in
  let a, a_ms = tune "A (default placements)" ctx timing_ctx in
  let embedded = ref [] in
  Tensor.iter_embedded ~f:(fun tn -> embedded := tn :: !embedded) loss;
  (* [decide_materialized] skips the nodes constrained away from materialization (constants,
     declared-virtual), mirroring [every_non_literal_materialized]'s guards at the decision
     level. *)
  let materialize c = Context.decide_materialized c !embedded in
  let b, b_ms =
    tune "B (materialize-all)" (materialize ctx) (Option.map timing_ctx ~f:materialize)
  in
  logf "winner: arm %s (A %.4f ms vs B %.4f ms)"
    (if Float.( <= ) a_ms b_ms then "A" else "B")
    a_ms b_ms;
  if Float.( <= ) a_ms b_ms then a else b

module Lazy = Utils.Lazy

(* The untuned compile of the recipes, behind the gh-ocannl-491 config gate: with
   [model_default_schedule=true], the default schedule is picked by the analytic cost model
   ({!Autotune.model_default} — zero timing runs, advisory, falls back to the ordinary default
   pipeline); otherwise plain [Context.compile]. *)
let compile_with_model_gate ctx comp bindings =
  if Lazy.force Autotune.model_default_enabled then Autotune.model_default ctx comp bindings
  else Context.compile ctx comp bindings

let%track7_sexp to_routine (ctx : Context.t) ?(output_cd_file = false) bindings comp =
  if output_cd_file then (
    let name = Asgns.get_name_exn comp.Asgns.asgns in
    if not Utils.settings.output_debug_files_in_build_directory then
      raise
      @@ Utils.User_error
           "Train.to_routine: output_cd_file is true, but output_debug_files_in_build_directory is \
            false";
    let cd_source = Utils.output_to_build_file ~fname:(name ^ "-debug.cd") in
    let static_indices = Idx.bound_symbols bindings in
    match cd_source with
    | None -> ()
    | Some callback -> callback (Asgns.to_doc ~name ~static_indices () comp.Asgns.asgns));
  (* Materialize the guessed output nodes so they persist across calls and are inspectable on demand
     via the context (gh-ocannl-333). *)
  Set.iter (snd @@ Asgns.collect_nodes_guess_output comp.Asgns.asgns) ~f:set_materialized;
  let _ctx, routine = compile_with_model_gate ctx comp bindings in
  (* Return just the routine for backward compatibility - ctx is discarded here *)
  routine

(** [init_params] initializes the parameters of [t], via running their forward code or copying from
    the host as appropriate. If [reinit_all] is true, all parameters are reinitialized, otherwise
    only the parameters that are not in [ctx.ctx_buffers] are initialized. *)
let init_params ?(reinit_all = false) ctx bindings t =
  let comp =
    if reinit_all then Tensor.init_params t
    else
      (* Check which params are already initialized *)
      let skip = Map.empty (module Tn) in
      Set.fold t.Tensor.params ~init:skip ~f:(fun skip p ->
          if Context.is_initialized ctx p.Tensor.value then
            Map.set skip ~key:p.Tensor.value ~data:()
          else skip)
      |> fun skip -> Tensor.init_params ~skip t
  in
  (* Materialize the parameters being initialized so they persist and are inspectable on demand. *)
  Set.iter (snd @@ Asgns.collect_nodes_guess_output comp.Asgns.asgns) ~f:set_materialized;
  (* Compile and run the initialization. Literal/ndarray-backed embedded nodes are uploaded into the
     context automatically at link time from [Host_inits] (gh-ocannl-333); there is no longer a
     separate host-array copy step here. *)
  let ctx, routine = Context.compile ctx comp bindings in
  Context.run ctx routine

type example_train_result = {
  inputs : Tensor.t;
  outputs : Tensor.t;
  model_result : Tensor.t;  (** Do not use [model_result] for deriving gradients. *)
  infer_callback : float array -> float array;
      (** Computes the output for the given input via the [model_result] tensor. Note:
          [infer_callback] is inefficient as it is not batched. *)
  rev_batch_losses : float list;
  rev_epoch_losses : float list;
  learning_rates : float list;
  used_memory : int;
}

(** [run_once] is a wrapper around {!init_params} that additionally runs code of [f t] and returns
    the context. If [skip_init] is true (false by default), no initialization is performmed. If
    [reinit_all] is true (false by default), all parameters are reinitialized, otherwise only the
    parameters that are not in [ctx.ctx_buffers] are initialized.

    If [output_cd_file] is true, the global setting [output_debug_files_in_build_directory] must be
    true, and the update code is output to a file before shape inference potentially crashes at
    [init_params]. *)
let%track3_sexp run_once ?(output_cd_file = false) ?(skip_init = false) ?reinit_all
    ?(bindings = IDX.empty) ~f ctx (t : Tensor.t) : Context.t =
  set_materialized t.Tensor.value;
  (* Compute the update early, to ensure the shape inference is done. *)
  let update = f t in
  if output_cd_file then (
    let name = Asgns.get_name_exn update.Asgns.asgns in
    if not Utils.settings.output_debug_files_in_build_directory then
      raise
      @@ Utils.User_error
           "Train.run_once: output_cd_file is true, but output_debug_files_in_build_directory is \
            false";
    let cd_source = Utils.output_to_build_file ~fname:(name ^ "-debug.cd") in
    let static_indices = Idx.bound_symbols bindings in
    match cd_source with
    | None -> ()
    | Some callback -> callback (Asgns.to_doc ~name ~static_indices () update.Asgns.asgns));
  let ctx =
    if skip_init || Set.is_empty t.params then ctx else init_params ?reinit_all ctx bindings t
  in
  let ctx, routine = compile_with_model_gate ctx update bindings in
  Context.run ctx routine

(** Context-based versions of training functions for the new simplified API *)

(** [forward_once] is a wrapper around {!run_once} that runs the forward code of [t]. *)
let forward_once ?output_cd_file ?(skip_init = false) ?reinit_all ?(bindings = IDX.empty) ctx t =
  let ctx = run_once ?output_cd_file ~skip_init ?reinit_all ~bindings ~f:forward ctx t in
  (* FIXME: this is going away soon. *)
  Tensor.remove_bprop_root t;
  ctx

(** [update_once] is a wrapper around {!run_once} that runs the gradient update code of [t]: both
    forward and backprop. *)
let update_once ?output_cd_file ?(skip_init = false) ?reinit_all ?(bindings = IDX.empty) ctx t =
  run_once ?output_cd_file ~skip_init ?reinit_all ~bindings ~f:grad_update ctx t

(* For-print materialization (gh-ocannl-333 AC 5): the [%cd "for_print" =: t] trick. When a tensor's
   value is not already materialized in the printing context, recompile a copy of it ([for_print = t
   + 0]) into a fresh device-resident node and register that node as a for-print proxy, so the
   printer reads the tensor's value through it.

   This is best-effort: it works for recomputable (e.g. virtual / fetch-defined) tensors. For a
   tensor that is materialized elsewhere but simply absent from this context, the copy cannot be
   linked (its operand has no value here) — in that case we fall back to the metadata placeholder
   rather than crash. A fresh copy is built each call because [forward_once] consumes the copy's
   forward root; the for-print node is registered as the source's proxy for subsequent reads. *)
let ensure_printable (ctx : Context.t) (t : Tensor.t) : Context.t =
  if Context.mem ctx t.Tensor.value then ctx
  else
    try
      let for_print =
        let%op for_print = t + 0 in
        for_print
      in
      let ctx = forward_once ctx for_print in
      Context.register_for_print ~src:t.Tensor.value ~proxy:for_print.Tensor.value;
      ctx
    with _ -> ctx

(** [printf] is a wrapper around {!Tensor.print} that assumes [~force:true], and by default sets
    [~with_code:false], [~with_grad:true], and [~style:`Default]. It takes an explicit context and
    retrieves values on demand (gh-ocannl-333). If the tensor's value is not already materialized in
    [ctx], it is recomputed via the [for_print] copy trick so real values are still shown. *)
let%debug7_sexp printf ?here ?(with_grad = true) ?(with_code = false) ?(with_low_level = false)
    ?(style = `Default) (ctx : Context.t) (t : Tensor.t) : unit =
  let ctx = ensure_printable ctx t in
  Tensor.print ?here ~force:true ~ctx ~with_grad ~with_code ~with_low_level style t

(** [printf_tree] is a wrapper around {!Tensor.print_tree} that assumes [~force:true], and by
    default sets [~with_value:true], [~with_grad:true], and [~depth:9]. It takes an explicit context
    and retrieves values on demand (recomputing via [for_print] if not already materialized). *)
let printf_tree ?here ?with_value ?(with_grad = true) ?(depth = 9) (ctx : Context.t) t =
  let ctx = ensure_printable ctx t in
  Tensor.print_tree ?here ~force:true ~ctx ?with_value ~with_grad ~depth t
