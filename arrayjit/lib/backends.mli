(** {1 A collection of the execution backends} *)

open Base
module Schedulers = Schedulers

val plan_pool_segments :
  cap:int ->
  what:string ->
  debug_name:(int -> string) ->
  (int * int) list ->
  (int * int) list * int list
(** gh-ocannl-344 pool-allocator planner. Lays out [(size, alignment)] allocations (in order) into
    pools so no pool's bumped extent exceeds [cap] bytes (the uint32 4 GB per-pool ceiling when
    [large_models = false]). Returns each item's [(segment_index, byte_offset)] and the byte size of
    each segment. Raises {!Ir.Utils.User_error} (naming [what] and [debug_name i]) when a single
    item exceeds [cap]. Exposed for unit testing the segmenting/cap behavior with synthetic sizes.
*)

val plan_arena_offsets :
  cap:int -> (int * int * string * (int * int) option) list -> (int list * int) option
(** gh-ocannl-489 liveness-aware pool layout: lays out
    [(size, alignment, precision_class, live_span)] allocations into ONE pool where two allocations
    may overlap iff both carry a live span, the spans are disjoint (closed intervals) and the
    precision classes are equal. A [None] span means always-live (conflicts with everything). Greedy
    by decreasing size, deterministic. Returns per-item byte offsets (in input order) and the pool's
    total size, or [None] when the layout exceeds [cap] (callers fall back to
    {!plan_pool_segments}). Exposed for unit testing the coloring with synthetic sizes. *)

type footprint = {
  fp_total : int;  (** [fp_working + fp_constants]: the number a memory budget is compared to. *)
  fp_working : int;
      (** Bytes of the working (non-constant) pool as the arena planner would lay it out. Equals
          [fp_dedicated] when there is no liveness plan (config [buffer_aliasing] off, code opaque
          to the liveness fold, or a layout over the per-pool cap). *)
  fp_constants : int;  (** Bytes of the constant / read-only pool, always bump-packed. *)
  fp_dedicated : int;  (** What [fp_working] would be with every node on its own bytes. *)
  fp_planned : int;  (** How many working nodes carried a live span, i.e. were arena-eligible. *)
  fp_nodes : int;  (** In-context nodes scored (working + constants). *)
}
[@@deriving sexp_of, equal]
(** gh-ocannl-498: the byte footprint implied by a routine's placement vector. *)

val score_footprint :
  backend_name:string ->
  limits:Ir.Backend_intf.hardware_limits ->
  static_indices:Ir.Indexing.static_symbol list ->
  Ir.Low_level.optimized ->
  footprint
(** gh-ocannl-498: score the peak footprint of a lowered routine under its own placements, with the
    same machinery the allocator uses — the default schedule and fission, the gh-ocannl-489 live
    spans, and {!plan_arena_offsets} over the working group. This is the cost side
    {!Ir.Low_level.field-flip_candidates} does not carry: the recompute-cost bound says what
    inlining a node costs, this says what it saves. Meaningful only with config [buffer_aliasing]
    on; without it every node is always-live and the score degenerates to bump packing.

    Scored over the routine's whole in-context node set rather than a context's allocation delta, so
    the number depends only on the code and the placements — the precondition for a deterministic
    selector ({!Context.plan_memory_budget}). It is therefore a {e model} of the peak, not a
    prediction of {!Context.get_used_memory}: the real allocator skips nodes a prior context already
    holds, and the driver page-rounds pool bases. Enumeration is canonical (by {!Ir.Tnode.uid}) so
    the greedy coloring is reproducible across processes. *)

val finalize :
  'dev 'runner 'event.
  (module Ir.Backend_intf.Backend
     with type dev = 'dev
      and type event = 'event
      and type runner = 'runner) ->
  ('dev, 'runner, 'event) Ir.Backend_intf.context ->
  unit
(** Frees the pools that are specific to the context -- not contained in the parent context. Note:
    use [finalize] to optimize memory, it is not obligatory because all pools are freed when their
    backend buffers are garbage-collected. *)

val lower_assignments :
  Ir.Low_level.optimize_ctx ->
  ?name:string ->
  'a Ir.Indexing.bindings ->
  Ir.Assignments.t ->
  string * Ir.Low_level.optimized
(** The shared lowering front half of backend [compile]: forks the lineage state
    ([Low_level.copy_optimize_ctx], so the compile's decisions stay hermetic), derives the routine
    name, wires the debug-file callbacks, and runs [Assignments.lower]. Exposed for analyze-only
    consumers (gh-560: {!Context.decision_surface}) that read the optimized code's decision surface
    without backend codegen. *)

(** {2 The implemented backends}

    Each backend is instantiated once per process, so its context type is nameable and two
    independently-created contexts on the same backend unify -- the precondition for [Context.copy]
    dispatching to the backend's [device_to_device] via {!wrapped_context}. Instantiation touches no
    driver or hardware: device discovery stays lazy inside the backends (forced at first
    [get_device], where [Context.auto]'s fallback can catch an unusable driver/device per call), and
    on platforms without the corresponding library the dune-[select]ed missing stub is what gets
    instantiated -- harmless at init, raising on use. Backend caches consequently persist across
    [Tensor.unsafe_reinitialize]; that is safe because tnode identity ([Tnode.uid]) is never reused.
*)

module Cc_b : Ir.Backend_intf.Backend
module Multidev_cc_b : Ir.Backend_intf.Backend
module Cuda_b : Ir.Backend_intf.Backend
module Hip_b : Ir.Backend_intf.Backend
module Metal_b : Ir.Backend_intf.Backend

(** The implemented backends. Constructors statically imply the corresponding singleton module ([Cc]
    -> {!Cc_b}, ...). *)
type backend = Cc | Multidev_cc | Cuda | Hip | Metal [@@deriving sexp, equal]

val get_backend : ?backend_name:string -> unit -> backend
(** The backend corresponding to [backend_name], or if omitted, selected via the global [backend]
    setting. Non-generative: the same backend value (hence the same singleton state) each call. *)

val backend_name : backend -> string
(** Inverse of {!get_backend}'s name parsing. *)

val backend_module : backend -> (module Ir.Backend_intf.Backend)
(** The singleton module as an existentially-packed first-class module, for generic consumers that
    thread a single backend through (the raw-API tests, [Parallel]). Code that must re-correlate two
    contexts later (e.g. [Context.copy]) should use {!wrapped_context} instead: this projection
    erases the type components. *)

(** {2 Contexts wrapped with their backend}

    A closed disjunction over the implemented backends' context types: matching two values on the
    same constructor recovers type equality directly, which is what lets [Context.copy] fall onto
    the backend-specific [device_to_device] when both contexts come from the same backend. *)

type wrapped_context =
  | Cc_ctx of Cc_b.context
  | Multidev_cc_ctx of Multidev_cc_b.context
  | Cuda_ctx of Cuda_b.context
  | Hip_ctx of Hip_b.context
  | Metal_ctx of Metal_b.context

val wrapped_backend : wrapped_context -> backend

val make_context : ?device_id:int -> backend -> wrapped_context
(** A fresh root context (empty [optimize_ctx]) on the backend's device [device_id] (default 0).
    Raises when the backend's hardware or library is unavailable. *)

type 'a ctx_op = {
  f :
    'dev 'runner 'event.
    (module Ir.Backend_intf.Backend
       with type dev = 'dev
        and type runner = 'runner
        and type event = 'event) ->
    ('dev, 'runner, 'event) Ir.Backend_intf.context ->
    ('dev, 'runner, 'event) Ir.Backend_intf.context * 'a;
}
(** A context-transforming backend operation, polymorphic over the backend's type components so
    {!with_backend} can rebuild the same {!wrapped_context} constructor around the result. *)

val with_backend : wrapped_context -> 'a ctx_op -> wrapped_context * 'a

type 'a ctx_query = {
  q :
    'dev 'runner 'event.
    (module Ir.Backend_intf.Backend
       with type dev = 'dev
        and type runner = 'runner
        and type event = 'event) ->
    ('dev, 'runner, 'event) Ir.Backend_intf.context ->
    'a;
}
(** A read-only backend operation; like {!ctx_op} but leaves the context untouched. *)

val query : wrapped_context -> 'a ctx_query -> 'a
