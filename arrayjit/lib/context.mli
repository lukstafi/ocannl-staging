(** Simplified context-based interface for backend operations *)

open Base
module Backends_deprecated = Backends

type t [@@deriving sexp_of]
(** Execution context managing device, compilation, and buffers *)

type routine
(** A compiled computational routine ready for execution *)

val bindings : routine -> Ir.Indexing.lowered_bindings
val context : routine -> t

(** {2 Context creation} *)

val cuda : ?device_id:int -> unit -> t
(** Create a CUDA context. *)

val hip : ?device_id:int -> unit -> t
(** Create an AMD HIP (ROCm) context. *)

val metal : ?device_id:int -> unit -> t
(** Create a Metal context. *)

val cpu : ?threads:int -> unit -> t
(** Create a CPU context. [threads] > 1 selects the [multidev_cc] backend (multiple worker-domain
    CPU devices, for debugging parallel workflows); otherwise the [cc] backend. Kernel-level CPU
    parallelism is automatic either way. *)

val auto : unit -> t
(** Automatically select the best available backend: the [backend] setting if configured, otherwise
    the first of metal, cuda, hip, cc whose device discovery succeeds. Only
    {!Ir.Backend_intf.Backend_unavailable} moves on to the next backend — a driver that is present
    but fails to initialize, an interrupt, or an assertion failure propagates rather than silently
    downgrading the run (gh-ocannl-536). *)

val advances_to_next_backend : exn -> bool
(** The selection policy of {!auto}, exposed so it can be pinned without a device: [true] exactly
    for the failures that mean "this backend is not available on this machine". *)

(** {2 Core operations} *)

val compile :
  ?name:string ->
  ?lowered_transform:(Ir.Low_level.optimized -> Ir.Low_level.optimized) ->
  ?lowered_transforms:(Ir.Low_level.optimized -> Ir.Low_level.optimized list) ->
  t ->
  Ir.Assignments.comp ->
  Ir.Indexing.unit_bindings ->
  t * routine
(** Compile assignments into an executable routine. Returns updated context and the compiled
    routine. The returned context carries the updated compilation frontier for dependency tracking;
    the input context is unchanged (see {!section:execution_deps}). [name] names the routine and its
    compilation artifacts (generated source files, kernel functions); if omitted, the name is
    derived from the comp's {!Ir.Assignments.Block_comment} labels via
    {!Ir.Assignments.get_name_exn}, which raises if the comp contains no block comment.
    [lowered_transform] rewrites the
    optimized lowered code before backend compilation — the seam for schedule transforms and for
    hand-annotating hardware axis types in tests (docs/proposals/axis-types-for-loops.md).
    [lowered_transforms] is the plural seam for transforms that split the routine into several
    kernels (fission): the returned segments run back-to-back on the routine's stream with
    device-side events at the boundaries, like {!Ir.Schedule.maybe_default_schedules}' segments.
    Pass at most one of the two. *)

val compile_outcome :
  ?name:string ->
  ?lowered_transform:(Ir.Low_level.optimized -> Ir.Low_level.optimized) ->
  ?lowered_transforms:(Ir.Low_level.optimized -> Ir.Low_level.optimized list) ->
  provenance:Ir.Schedule_outcome.provenance ->
  ?candidate:string ->
  t ->
  Ir.Assignments.comp ->
  Ir.Indexing.unit_bindings ->
  (t * routine) Ir.Schedule_outcome.outcome
(** Internal containment-aware form of {!compile}. The caller supplies the schedule provenance;
    public user code should continue to use {!compile}. *)

val decision_surface :
  ?name:string ->
  t ->
  Ir.Assignments.comp ->
  Ir.Indexing.unit_bindings ->
  Ir.Low_level.flip_candidate list
(** gh-560: the analyze-only entry point — the routine's searchable inlining decision dimensions
    ({!Ir.Low_level.field-flip_candidates}, most expensive first) as {!compile} would report them
    from this context, computed by lowering and optimization alone: no backend codegen, no linking,
    and no effect on the context (the lineage state is forked like a compile's, and the compilation
    frontier and ledger are untouched). Sibling compiles of the same routine share the underlying
    analysis via the analysis cache, so after a compile this costs only the specialization replay.
    Used by [Train.tune_placements]' flip refinement to read the decision surface. *)

val run : t -> routine -> t
(** Execute a compiled routine. Mutates buffers in-place. Returns updated context with newly
    initialized nodes tracked. Raises [Failure] if execution dependencies are not satisfied. *)

val check_runnable : t -> routine -> unit
(** {!run}'s pre-dispatch validation on its own — poisoned lineage, uninitialized inputs,
    unsatisfied execution dependencies, out-of-range bindings — raising exactly what [run] would.
    All of it precedes the dispatch, so a failure here proves nothing was executed and no device
    buffer was written. That is the point (gh-ocannl-550): a caller running a routine inside a
    launch-tagged failure boundary validates through this in its own
    {!Ir.Schedule_outcome.Preflight} region, so an unattributable failure at [Launch] means dispatch
    was attempted and the lineage must be condemned, while an unsatisfied dependency — fixable, and
    retryable — is the contained decline it is (gh-ocannl-564). *)

val sync : t -> unit
(** Blocks until the context's device is idle. Host reads ({!to_host}, {!get_values}) synchronize on
    their own; explicit [sync] is for timing runs (e.g. the autotuner) and for fencing against
    out-of-band observation. *)

val failure_classifier :
  t -> Ir.Schedule_outcome.phase -> exn -> Ir.Schedule_outcome.classified_cause option
(** The backend's own failure classifier, for callers that wrap {!run} / {!sync} in
    {!Ir.Schedule_outcome.protect}. {!compile_outcome} passes it in itself; launch and sync are
    raising APIs, so the autotuner obtains it here. Passing a classifier that always answers [None]
    (as the timing loop did before gh-ocannl-536) makes every launch failure fatal by phase default,
    with no way for a backend to declare one its candidate's fault. *)

val rollback_execution : t -> int -> unit
(** Undoes {!run}'s execution marking for the given routine id. {!run} marks a routine executed
    before the later {!sync} can report an asynchronous failure, so a contained launch/sync
    rejection has to withdraw that claim — otherwise the next routine compiled in this lineage
    waits on a dependency that never completed. Only sound when the failure is known not to have
    written device buffers; otherwise use {!poison_lineage}. *)

val poison_lineage : t -> routine_name:string -> exn -> unit
(** Marks this execution lineage unusable because a failure may have left device buffers partially
    written. Every subsequent {!run}, {!sync}, {!to_host} and {!from_host} on any context sharing
    the lineage raises, naming the routine and the original failure. There is deliberately no
    restore: recovering would mean rebuilding inputs and parameters, which the current
    [timing_ctx]-shaped API cannot express (gh-ocannl-536). *)

val poisoned_failure : t -> exn option
(** The failure that poisoned this execution lineage, if any — the exception every entrypoint on it
    now raises. Lets a caller that would otherwise start fresh work on the lineage see that it
    cannot run: [Train.tune_placements] checks it before searching the second placement arm, since
    the arms share a lineage and a poisoned one refuses every timing run (gh-ocannl-550). *)

val hardware_limits : t -> Ir.Backend_intf.hardware_limits
(** The backend's conservative per-workgroup device limits (all-[None] on backends that do not bind
    hardware axes). Chiefly for schedule transforms and the autotuner. *)

(** {2 Execution dependency tracking}

    Execution dependencies mirror compilation dependencies: they record which routines must execute
    before which others based on tensor-node read/write hazards (RAW, WAR, WAW).

    Dependencies are scoped to compilation lineage: two routines compiled from the {i same}
    [Context.t] are independent siblings, even if they access the same nodes. Only routines compiled
    from the {i returned} (child) context of a prior [compile] call can depend on that prior
    routine. This matches how [compile] advances backend state only in the returned context. *)

val routine_id : routine -> int
(** A unique integer identifying the routine within its root context's lifetime. *)

val routine_name : routine -> string
(** The name of the routine, derived from the backend compilation. *)

val execution_deps : routine -> int list
(** The routine IDs that must execute before this routine, derived from RAW, WAR, and WAW hazards on
    tensor nodes at compile time. An empty list means the routine is independent of all previously
    compiled routines in its lineage. *)

val can_run : t -> routine -> bool
(** Whether all execution dependencies of the routine have been satisfied (i.e., all prerequisite
    routine IDs have been executed). *)

(** {2 Data operations} *)

(** Note: These operations work with backend-specific buffer types hidden behind the context
    abstraction. *)

val copy : ?into_merge_buffer:Ir.Backend_intf.merge_buffer_use -> src:t -> dst:t -> Ir.Tnode.t -> t
(** Copies the node's device buffer from [src] into [dst] (default [~into_merge_buffer:No]), or into
    [dst]'s stream's merge buffer for [~into_merge_buffer:Copy], returning the updated destination
    context. When both contexts come from the same backend the copy stays on-device via the
    backend's [device_to_device] transfer machinery (for [Copy], the returned context carries the
    merge-buffer node against which the next [compile] of merge-consuming code is statically
    verified); a cross-backend copy falls back to a host round-trip ([Copy] raises). Nodes absent
    from [src]'s device buffers fall back to the host round-trip as well, serving host-init literals
    and for-print proxies. *)

(** {2 On-demand host access}

    After [gh-ocannl-333] no tensor data is stored on the host side of a tensor node. All CPU-side
    value access is an {b on-demand, context-mediated} device-to-host (or host-to-device) transfer
    through a temporary host buffer. There is no cache: every call performs a fresh transfer, which
    is {b expensive on non-unified-memory backends} — prefer batching over polling.

    Which nodes are observable is determined by the compilation lineage's placement resolution
    ({!placements}; the tnode's {!Ir.Tnode.field-memory_mode} only records declared intent):
    - [On_device] (materialized) nodes have a context buffer; {!to_host}/{!get_values} read it
      directly.
    - [Virtual] nodes have no buffer anywhere, but they remain observable: their defining
      computation is tracked, so their value can be recomputed on demand — [Train.printf] does this
      via the for-print proxy mechanism ({!register_for_print}); raw {!to_host} on them raises
      unless a proxy or host-init data exists. Observability is inductive: it holds only when every
      node the tracked computation reads is itself observable — a [Virtual] node depending (even
      transitively through other [Virtual] nodes) on a [Local] node inherits its unobservability.
    - [Local] nodes are routine-scoped scratch and {b unobservable}: their computation is not
      tracked, and they are stored (to whatever degree the optimizer decides on) only within a
      single routine invocation. This is a deliberate opt-out from the observability guarantee that
      licenses backend optimizations (e.g. stack allocation). The mode is only ever assigned by the
      compiler, to nodes never read outside their defining routine; to prevent it, request
      materialization (e.g. [Train.set_materialized]) before the first routine using the node is
      compiled. *)

val mem : t -> Ir.Tnode.t -> bool
(** Whether the node has a device buffer allocated in this context. *)

val register_for_print : src:Ir.Tnode.t -> proxy:Ir.Tnode.t -> unit
(** Registers [proxy] as a for-print copy of [src] (gh-ocannl-333 AC 5): when [src] is not present
    in a context, {!to_host}/{!get_values} on [src] read through [proxy] instead. Used by
    [Train.printf] to render the value of a tensor that is not directly materialized in the context.
*)

val to_host : t -> Ir.Tnode.t -> Ir.Ndarray.t
(** Transfers the node's device buffer into a fresh host [Ndarray] and returns it. Raises if the
    node is not present in the context (and has no host-init data or for-print proxy). *)

val from_host : t -> Ir.Tnode.t -> Ir.Ndarray.t -> t
(** Uploads the host buffer into the node's device buffer (allocating it if needed) and returns a
    context in which the node is marked initialized. *)

val get_values : t -> Ir.Tnode.t -> float array
(** Retrieves all (unpadded) values of the node via an on-demand device-to-host transfer. *)

val set_values : t -> Ir.Tnode.t -> float array -> t
(** Sets all (unpadded) values of the node via an on-demand host-to-device transfer, returning a
    context in which the node is marked initialized. *)

val get_value : t -> Ir.Tnode.t -> int array -> float
(** Retrieves a single value at the given index via an on-demand device-to-host transfer. *)

val set_value : t -> Ir.Tnode.t -> int array -> float -> t
(** Sets a single value at the given index, preserving the other elements. Returns a context in
    which the node is marked initialized. *)

val points_1d : ?from_axis:int -> xdim:int -> t -> Ir.Tnode.t -> float array
(** Like {!get_values} but extracts a 1d slice of points for plotting. *)

val points_2d : ?from_axis:int -> xdim:int -> ydim:int -> t -> Ir.Tnode.t -> (float * float) array
(** Like {!get_values} but extracts a 2d slice of points for plotting. *)

(** {2 Node tracking operations} *)

val is_initialized : t -> Ir.Tnode.t -> bool
(** Check if a node is initialized. *)

(** {2 Debug operations} *)

val backend_name : t -> string
(** Get the name of the backend. *)

val device_id : t -> int

val get_used_memory : t -> int
(** (An upper bound of) the memory used for arrays on the context's device, in bytes. Device-wide:
    covers all contexts sharing the device. Useful for asserting the footprint effect of the
    liveness memory planner (config [buffer_aliasing], gh-ocannl-489). *)
(** Get the device ID. *)

val placements : t -> Ir.Tnode.Placements.t
(** The context's compilation lineage's memory-mode resolution
    (docs/proposals/context-scoped-memory-modes.md): which nodes this lineage decided to inline
    ([Virtual]), keep as routine-scoped scratch ([Local]), or give a device buffer ([On_device]).
    Reads are side-effect free; chiefly for tests and diagnostics. *)

val decide_materialized : t -> Ir.Tnode.t list -> t
(** A child context whose compilation lineage additionally decides [On_device] placement for the
    given nodes: subsequent compiles from the returned context materialize them. This is the
    functional, context-scoped counterpart of strengthening tnode-level intent
    ([Train.set_materialized]) — the nodes' declared intent is untouched and the argument context
    (with its other descendants) is unaffected, so a default-placement sibling and a materialize-all
    sibling can coexist (e.g. the placement-A/B arms of [Train.tune_placements]). Nodes the lineage
    or intent already constrains away from plain materialization ([Virtual], [Local], or constant)
    are skipped. *)

val decide_inline : t -> Ir.Tnode.t list -> t
(** A child context whose compilation lineage additionally prefers the given nodes inline
    (gh-555): subsequent compiles exempt them from the heuristic virtualization caps
    ([virtualize_max_visits], [virtualize_max_inline_reduction]) — the caps are priors of the
    default placement policy, not legality. Legality still applies: a preferred node the
    virtualizer rejects (escaping symbols, non-injective producer indices, opaque effects, ...)
    materializes as before, and the observability pessimizations (read-only, read-before-write)
    are unaffected. The preference steers placements the lineage has {e not yet decided}: a node
    already materialized by an earlier compile in this lineage keeps that decision (decisions are
    final within a lineage — compiled routines depend on them), so apply the preference to a
    pre-compile sibling of the routines that set the node, as [Train.tune_placements] does.
    Together with {!decide_materialized} this spans the per-node inlining decision vector:
    [Inline] here, [Materialize] there, the default heuristics elsewhere. Hermetic like
    {!decide_materialized}: the argument context and its other descendants are unaffected. *)

(** {2 Memory-budget planning (gh-ocannl-498)} *)

type memory_budget =
  | Bytes of int  (** Fit the routine's scored footprint under this many bytes. *)
  | Minimize  (** Take every flip that relieves footprint, whatever the recompute cost. *)
[@@deriving sexp_of]

type budget_plan = {
  bp_baseline : Backends_deprecated.footprint;  (** The default-policy placement vector's score. *)
  bp_final : Backends_deprecated.footprint;  (** The score after the accepted flips. *)
  bp_flips : (Ir.Tnode.t * int * int) list;
      (** The accepted flips in acceptance order: the node demoted to recompute-at-use, the
          {e marginal} bytes it relieved on top of the flips accepted before it, and its
          recompute-cost bound. Flips committed as one joint group (see {!plan_memory_budget})
          carry [0] each except the one that closed the group, which carries the group's whole
          relief — so the reliefs sum to exactly [bp_baseline - bp_final] either way. *)
  bp_considered : int;  (** Inline candidates individually scored. *)
  bp_dropped : int;  (** Inline candidates the [max_candidates] cut left unscored. *)
  bp_within_budget : bool;
      (** Whether [bp_final] meets the budget. Always [true] for {!Minimize}, which has no target
          to miss. A [false] here is a planning outcome, not an error: the selector reports that
          the decision vector cannot reach the budget rather than forcing illegal flips. *)
}
[@@deriving sexp_of]

val compare_relief_ratio : int -> int -> int -> int -> int
(** gh-ocannl-498: [compare_relief_ratio ra ca rb cb] compares the rationals [ra/ca] and [rb/cb]
    exactly, as {!plan_memory_budget} ranks candidates by footprint relief per unit of recompute
    cost. [ca] and [cb] must be positive; the numerators are byte counts and may be negative, since
    inlining a node can cost footprint rather than free it. Never cross-multiplies (the products of
    a byte count and a recompute cost can overflow) and never uses floats (the order must be
    bit-reproducible). Exposed for unit testing the ordering, including the sign cases where a
    continued-fraction descent over truncating division would otherwise invert it. *)

val footprint :
  ?name:string ->
  t ->
  Ir.Assignments.comp ->
  Ir.Indexing.unit_bindings ->
  Backends_deprecated.footprint
(** The byte footprint {!compile} of this routine from this context would imply under the default
    placement policy ({!Backends_deprecated.score_footprint}). Analyze-only and hermetic, exactly
    like {!decision_surface}: lowering and optimization, no backend codegen, no linking, no effect
    on the context. *)

val plan_memory_budget :
  ?name:string ->
  ?max_candidates:int ->
  budget:memory_budget ->
  t ->
  Ir.Assignments.comp ->
  Ir.Indexing.unit_bindings ->
  t * budget_plan
(** gh-ocannl-498 rematerialization: choose which materialized intermediates to demote to
    recompute-at-use so that the routine's scored footprint ({!footprint}) fits [budget], and
    return a child context that decides them inline ({!decide_inline}) together with the plan.

    A deterministic planning pass, not a timed search: recompute-vs-store under a budget is
    decidable from the two cost sides — the recompute-cost bound each [`Inline] flip candidate
    carries ({!Ir.Low_level.field-flip_candidates}) and the footprint relief scored against the
    actual arena layout. Nothing is compiled, linked or executed; given the same code, config and
    placements the pass always chooses the same flips.

    The selection is greedy in two rounds. First every candidate is scored on its own against the
    baseline layout — footprint relief is not a function of the node's own size, since a node
    whose live span was already shared with another's frees no bytes by leaving. That solo relief
    only {e ranks}: candidates are ordered by relief per unit of recompute cost (an exact rational
    comparison, never a cross-multiplication that could overflow nor a float, so the order is
    bit-reproducible), zero-relief ones last.

    Round two accepts a prefix, re-scoring the {e cumulative} vector at each step, since inlining
    one node moves the others' spans. A candidate that adds nothing on top of the accepted set is
    not dropped but held {e speculatively}: relief is not additive in either direction, and two
    nodes pinning the same arena peak each free nothing alone yet free the whole range together.
    Each later candidate is then scored both with and without the held group, because a held flip
    can be actively harmful rather than merely unpaid, and judging every later candidate only in its
    company would let one bad hold mask a candidate that pays on its own. A group that beats the
    candidate alone is load-bearing and commits with it; a group that loses is harmful and is
    discarded, never reconsidered (this is a bounded planner, not a search over subsets); a group
    that ties is merely neutral and keeps being held — committing it would pay recompute for zero
    bytes, and discarding it would throw away a flip that may still be half of a later pair.
    Speculatives never joined by a paying flip are discarded at the end, so recompute is never paid
    for zero bytes. Acceptance stops as soon as the budget is met; {!Minimize} takes every flip that
    helps.

    [max_candidates] bounds the individually-scored candidates, keeping the cheapest-to-recompute
    ones; the count left unscored is reported as [bp_dropped] and logged under config
    [log_memory_budget], never silently dropped. It defaults to 32 for a {!Bytes} budget — which
    stops as soon as it is met, so the cut is a cost guard — and to {e unbounded} for {!Minimize},
    whose contract is every flip that still relieves footprint and whose config-only users
    ([memory_budget=minimize]) cannot raise a cap. Passing it explicitly bounds either kind, at two
    lowerings per candidate scored.

    Only the [`Inline] direction is considered — the opposite of the [`Materialize] chain
    {!Ir.Low_level.field-flip_candidates} feeds in [Train.tune_placements]. Legality and
    observability are not this pass's to enforce and it does not try: {!decide_inline} records a
    preference, the virtualizer's [check_and_store_virtual] settles legality, and a rejected
    preference simply reproduces the materialized placement — which is why relief is scored from a
    real lowering rather than assumed.

    Raises {!Ir.Utils.User_error} when config [buffer_aliasing] is off: without the liveness
    planner every node is always-live and the score has nothing to do with what the allocator
    would do. *)
