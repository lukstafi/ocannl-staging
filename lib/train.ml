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
    sees unscaled gradients, and so does any later reader of [p.grad] (e.g. gradient clipping).

    When [update_gate] is given (a broadcastable scalar holding 1 to apply the step and 0 to skip
    it, computed on device — see [Mixed_prec.gated_scaled_update], gh-ocannl-492 task 5), every
    optimizer-state mutation is gated by [Where] {e selection}: on a skipped step the parameter and
    the momentum buffer keep their previous values exactly. Selection, not multiplication — the
    skipped steps are the ones whose gradients hold [inf]/[nan], and [0 * inf] is [nan]. *)
let sgd_one ~learning_rate ?(momentum = 0.0) ?(weight_decay = 0.0) ?(nesterov = false) ?grad_unscale
    ?update_gate p =
  if Option.is_none p.Tensor.diff then
    raise @@ Tensor.Session_error ("Train.sgd_one: not differentiable", Some p);
  match update_gate with
  | None ->
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
             if nesterov then sgd_delta =+ !.momentum *. sgd_momentum
             else sgd_delta =: sgd_momentum);
           p =- learning_rate * sgd_delta ~logic:".")]
  | Some gate ->
      [%cd
        ~~(p "param sgd step";
           (match grad_unscale with
           | Some unscale -> p.grad =: p.grad * unscale ~logic:"."
           | None -> Asgns.empty_comp);
           { sgd_delta } =: p.grad + (!.weight_decay *. p);
           if Float.(momentum > 0.0) then (
             { sgd_momentum } =: where gate ((!.momentum *. sgd_momentum) + sgd_delta) sgd_momentum;
             if nesterov then sgd_delta =+ !.momentum *. sgd_momentum
             else sgd_delta =: sgd_momentum);
           (* The final selection covers every path: without momentum it discards the possibly
              non-finite delta; with momentum the buffer kept its old (finite) value above, and
              this still zeroes the step so [p] is untouched. *)
           sgd_delta =: where gate sgd_delta 0;
           p =- learning_rate * sgd_delta ~logic:".")]

let sgd_update ~learning_rate ?momentum ?weight_decay ?nesterov ?grad_unscale ?update_gate loss =
  let f = sgd_one ~learning_rate ?momentum ?weight_decay ?nesterov ?grad_unscale ?update_gate in
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
    in order — arm A first, then arm B — and the arm with the smaller [best_ms] is the one that
    ships, so a consumer holding both reports can attribute every per-arm fact to a shipping or a
    discarded artifact without reading the log. That is how "a [Schedule.Tensorize] was crowned in
    an arm that did not ship" becomes reportable (gh-ocannl-546): [best_tensorized] on the losing
    arm's report, with [mma_best_ms] against [best_ms] for the margin. The same conclusion is
    logged here under config [autotune_log]. Other arguments are forwarded to {!Autotune.tune}; the
    same caveats apply (notably [timing_ctx] and non-idempotent routines — both arms share
    [timing_ctx]'s device for their searches).

    An arm that fails is a {e losing} arm, not a failed run (gh-ocannl-550): a search that
    terminates on a fatal failure ranks at [infinity], the other arm's completed winner ships and
    stays cached, and the failed arm's own (partial) report — carrying its [terminal_failure] —
    still reaches [report] in position, so the failure is recorded rather than downgraded to "that
    arm merely lost". A partial arm's [best_ms] is deliberately {e not} shippable and not compared:
    {!Autotune.tune} raised, so no routine was compiled from [ctx] for it. Only when every arm fails
    does [tune_placements] propagate, with the first failure's original backtrace. Consumers
    attributing arms by arrival order should read [terminal_failure] (equivalently [partial]) before
    [best_ms], as benchmarks/runners/ocannl/bench_harness.ml does.

    The arms differ in which candidates {e exist}, not only in how they rank: a tensorized candidate
    is seeded only when the matmul site's operand and destination storage precisions resolve to a
    tile the backend advertises ({!Autotune.mma_tile_for_precisions}), and placement decides which
    nodes the site reads. Under the mixed-precision recipe on a uniform-format backend (Metal's
    simdgroup matrices) that makes arm A tensorization-free: the reduced-precision cast twins are
    virtual there, so the site reads f32 masters into a reduced-precision destination — a mixed
    triple no tile matches — while materialize-all turns the twins into real reduced-precision nodes
    and the seeds fire. Materializing just the twins ([Mixed_prec.Twin_materialized]) reaches the
    same seeds at arm A's cost; see benchmarks/report-gh546-metal.md.

    gh-555: the A/B is the coarse level of the hierarchical inlining search — inlining decided
    first, tiling/scheduling within each arm by the nested {!Autotune.tune}. [inline_flips] (config
    [tune_inline_flips], default 0) adds a greedy per-node refinement level: the default-policy
    arm's compile reports its searchable decision dimensions
    ({!Ir.Low_level.field-flip_candidates}, ranked by the recompute-cost bound), and the top
    candidates are tried one at a time from arm A's context — [Materialize] via
    {!Context.decide_materialized} (walking toward arm B one node at a time), [Inline] via
    {!Context.decide_inline} — each accepted flip becoming the base for the next, and the refined
    result shipping only if it beats the A/B winner. Every tried flip costs a full search like an
    arm, so the budget is explicit and defaults to zero. Flip searches report through
    [flip_report], not [report]: the positional arm-A-then-arm-B contract of [report] is preserved
    regardless of the budget. *)
let tune_placements ?beam_width ?rounds ?repeats ?cache_dir ?timing_ctx ?report ?flip_report
    ?inline_flips ctx loss comp bindings =
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
  let last = ref None in
  (* The public [?report] contract is positional — arm A's report then arm B's, which consumers
     (e.g. the benchmark harness) attribute by arrival order — so flip-refinement searches report
     through the separate [?flip_report] instead ([~to_report] selects the callback).

     gh-ocannl-550: the arms are independent experiments, so one arm's terminal failure is no
     evidence about another arm's completed result — and must not destroy it. Per-candidate
     containment inside a search is {!Autotune.tune}'s job (gh-ocannl-533/536) and it holds: on the
     reproduction that motivated this (benchmarks/report-gh528-gpt2-cuda.md §3, five of five tf32
     gpt2_mini runs) the OOMing candidates were absorbed as ordinary [Backend_link] declines and the
     search ran to its end. What escaped is the aftermath: with the device exhausted, the arm's
     winner replay could not compile and neither could its untuned-default fallback, so
     [Autotune.tune] raised — and with no handler here, arm B's failure took arm A's already
     finished, already cached winner (106.389 ms, the best arm-A result of that whole leg) out of
     the process with it. Catching per arm is the whole fix: a failed search returns [Error] and
     ranks at [infinity], which is not the same as a completed search that timed nothing (which
     returns the untuned default compile and also ranks at [infinity] — hence [Result.t] rather than
     a sentinel time). *)
  let tune ?to_report arm ctx timing_ctx =
    let to_report = Option.value to_report ~default:report in
    let capture r =
      last := Some r;
      Option.iter to_report ~f:(fun f -> f r)
    in
    logf "arm %s search:" arm;
    last := None;
    let result =
      match
        Autotune.tune ?beam_width ?rounds ?repeats ?cache_dir ?timing_ctx ~report:capture ctx comp
          bindings
      with
      | compiled -> Ok compiled
      | exception exn -> Error (exn, Stdlib.Printexc.get_raw_backtrace ())
    in
    let r = !last in
    let best_ms =
      match result with
      (* Not [r.best_ms]: the partial report's best is a measurement of the search context, and no
         routine was compiled from [ctx] for it. There is nothing to ship at that time. *)
      | Error _ -> Float.infinity
      | Ok _ -> Option.value_map r ~default:Float.infinity ~f:(fun r -> r.Autotune.best_ms)
    in
    (match result with
    | Error (exn, _) ->
        logf "arm %s FAILED, it loses the comparison (%s): %s" arm
          (Option.value_map r ~default:"it reported nothing" ~f:(fun r ->
               if Float.is_inf r.Autotune.best_ms then "it had timed nothing"
               else
                 Printf.sprintf "its pre-failure best of %.4f ms is not shippable"
                   r.Autotune.best_ms))
          (Exn.to_string exn)
    | Ok _ ->
        logf "arm %s best: %.4f ms (%s)" arm best_ms
          (Option.value_map r ~default:"no report" ~f:(fun r ->
               Printf.sprintf "%s%s, best tensorized %s"
                 (if String.is_empty r.Autotune.best_label then "nothing timed"
                  else r.Autotune.best_label)
                 (if r.Autotune.best_tensorized then " [tensorized]" else "")
                 (if Float.is_inf r.Autotune.mma_best_ms then "none"
                  else Printf.sprintf "%.4f ms" r.Autotune.mma_best_ms))));
    (result, best_ms, r)
  in
  let a, a_ms, a_report = tune "A (default placements)" ctx timing_ctx in
  let embedded = ref [] in
  Tensor.iter_embedded ~f:(fun tn -> embedded := tn :: !embedded) loss;
  (* [decide_materialized] skips the nodes constrained away from materialization (constants,
     declared-virtual), mirroring [every_non_literal_materialized]'s guards at the decision
     level. *)
  let materialize c = Context.decide_materialized c !embedded in
  let b, b_ms, b_report =
    tune "B (materialize-all)" (materialize ctx) (Option.map timing_ctx ~f:materialize)
  in
  (* Both arms gone: there is no winner to ship, and the first failure is the one that has not been
     cascaded from — a device the other arm's failure exhausted or a lineage it poisoned would
     otherwise be reported as the cause. *)
  (match (a, b) with
  | Error (a_exn, a_backtrace), Error (b_exn, _) ->
      logf "both arms failed, nothing to ship (A: %s; B: %s)" (Exn.to_string a_exn)
        (Exn.to_string b_exn);
      Stdlib.Printexc.raise_with_backtrace a_exn a_backtrace
  | _ -> ());
  let a_wins =
    match (a, b) with
    (* A failed arm never wins, whatever the other arm's time is — including [infinity], which a
       completed search that timed nothing legitimately reports. *)
    | Ok _, Error _ -> true
    | Error _, Ok _ -> false
    | Ok _, Ok _ | Error _, Error _ -> Float.( <= ) a_ms b_ms
  in
  let arm_ms r ms = match r with Error _ -> "FAILED" | Ok _ -> Printf.sprintf "%.4f ms" ms in
  logf "winner: arm %s (A %s vs B %s)" (if a_wins then "A" else "B") (arm_ms a a_ms)
    (arm_ms b b_ms);
  (* gh-ocannl-546: a tensorized winner of the arm that is then discarded reaches no artifact and no
     end-to-end number, so the placement A/B is where it has to be said. Stated as the margin it
     lost by, not as a bare flag: on a small routine the arms can be separated by less than the
     candidate-level timing spread. *)
  let shipped, dropped = if a_wins then (a_report, b_report) else (b_report, a_report) in
  Option.iter dropped ~f:(fun d ->
      if d.Autotune.best_tensorized then
        logf
          "NOTE arm %s crowned a tensorized candidate (%s at %.4f ms%s) and did NOT ship: arm %s \
           wins the placement A/B at %.4f ms%s"
          (if a_wins then "B" else "A")
          d.Autotune.best_label d.Autotune.best_ms
          (* A partial arm's crown is mid-search: it lost the A/B by failing, not by its time. *)
          (if Option.is_some d.Autotune.terminal_failure then ", before that arm failed" else "")
          (if a_wins then "A" else "B")
          (if a_wins then a_ms else b_ms)
          (Option.value_map shipped ~default:"" ~f:(fun s ->
               if s.Autotune.best_tensorized then " (which is tensorized too)"
               else if Float.is_inf s.Autotune.mma_best_ms then
                 " (no tensorized candidate was timed in the shipping arm)"
               else
                 Printf.sprintf " (its own best tensorized candidate: %.4f ms)"
                   s.Autotune.mma_best_ms)));
  let winner, winner_ms = if a_wins then (a, a_ms) else (b, b_ms) in
  let inline_flips =
    match inline_flips with
    | Some n -> n
    | None -> Int.of_string (Utils.get_global_arg ~arg_name:"tune_inline_flips" ~default:"0")
  in
  (* The unwrap is the type-level statement of an invariant already established above: both arms
     [Error] propagated, [a_wins] never picks a failed arm, and the flip chain only ever replaces
     its incumbent with a strictly faster {e completed} search ([infinity] ranks a failed one). *)
  let ship = function
    | Ok compiled -> compiled
    | Error (exn, backtrace) -> Stdlib.Printexc.raise_with_backtrace exn backtrace
  in
  if inline_flips <= 0 then ship winner
  else (
    (* gh-555: greedy per-node refinement over the inlining decision vector. The vector lives on
       the default-policy arm (arm B's placements are caller-seeded wholesale, so its compile
       reports no policy decisions to flip), so the chain refines from arm A's context — a
       Materialize chain walks from A toward B one node at a time — and the refined result ships
       only if it beats the A/B winner. A capture compile (no timing) reads the decision surface
       off the default compile. *)
    let module LL = Ir.Low_level in
    let captured = ref [] in
    (* The capture compile runs outside the tuner's failure containment; a backend that rejects
       the unscheduled base lowering (the A/B searches above can still have crowned a scheduled
       winner) must skip the refinement, not fail the tune. *)
    (match
       Context.compile
         ~lowered_transform:(fun o ->
           captured := o.LL.flip_candidates @ !captured;
           o)
         ctx comp bindings
     with
    | (_ : Context.t), (_ : Context.routine) -> ()
    | exception exn ->
        captured := [];
        logf "flip refinement skipped: the capture compile failed: %s" (Exn.to_string exn));
    let candidates =
      List.fold !captured ~init:[] ~f:(fun acc fc ->
          (* Identity is [Tn.uid] ([Tn.equal]), not the session [id], which can repeat across
             namespaces and reinitializations. *)
          if List.exists acc ~f:(fun c -> Tn.equal c.LL.fc_tn fc.LL.fc_tn) then acc
          else fc :: acc)
      |> List.sort ~compare:(fun a b ->
             match Int.compare b.LL.fc_recompute_cost a.LL.fc_recompute_cost with
             | 0 -> Tn.compare a.LL.fc_tn b.LL.fc_tn
             | c -> c)
      |> fun l -> List.take l inline_flips
    in
    logf "flip refinement: trying %d candidate(s) within budget %d" (List.length candidates)
      inline_flips;
    let chain = ref (a, a_ms, ctx, timing_ctx) in
    List.iter candidates ~f:(fun fc ->
        let _, chain_ms, base_ctx, base_timing = !chain in
        let apply c =
          match fc.LL.fc_flip with
          | `Materialize -> Context.decide_materialized c [ fc.LL.fc_tn ]
          | `Inline -> Context.decide_inline c [ fc.LL.fc_tn ]
        in
        let arm =
          Printf.sprintf "flip %s %s (cost %d)"
            (match fc.LL.fc_flip with `Inline -> "inline" | `Materialize -> "materialize")
            (Tn.debug_name fc.LL.fc_tn) fc.LL.fc_recompute_cost
        in
        let ctx' = apply base_ctx in
        let timing' = Option.map base_timing ~f:apply in
        let r, ms, _rep = tune ~to_report:flip_report arm ctx' timing' in
        if Float.(ms < chain_ms) then chain := (r, ms, ctx', timing'));
    let chain_result, chain_ms, _, _ = !chain in
    if Float.(chain_ms < winner_ms) then (
      logf "flip refinement ships: %.4f ms (the placement A/B winner was %.4f ms)" chain_ms
        winner_ms;
      ship chain_result)
    else (
      logf "flip refinement did not improve on the A/B winner (%.4f ms vs %.4f ms)" chain_ms
        winner_ms;
      ship winner))

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
