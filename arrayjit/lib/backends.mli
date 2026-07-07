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

(** {2 The implemented backends}

    Each backend is instantiated once per process, so its context type is nameable and two
    independently-created contexts on the same backend unify -- the precondition for
    [Context.copy] dispatching to the backend's [device_to_device] via {!wrapped_context}.
    Instantiation touches no driver or hardware: device discovery stays lazy inside the backends
    (forced at first [get_device], where [Context.auto]'s fallback can catch an unusable
    driver/device per call), and on platforms without the corresponding library the
    dune-[select]ed missing stub is what gets instantiated -- harmless at init, raising on use.
    Backend caches consequently persist across [Tensor.unsafe_reinitialize]; that is safe because
    tnode identity ([Tnode.uid]) is never reused. *)

module Cc_b : Ir.Backend_intf.Backend
module Multicore_cc_b : Ir.Backend_intf.Backend
module Cuda_b : Ir.Backend_intf.Backend
module Metal_b : Ir.Backend_intf.Backend

type backend = Cc | Multicore_cc | Cuda | Metal [@@deriving sexp, equal]
(** The implemented backends. Constructors statically imply the corresponding singleton module
    ([Cc] -> {!Cc_b}, ...). *)

val get_backend : ?backend_name:string -> unit -> backend
(** The backend corresponding to [backend_name], or if omitted, selected via the global [backend]
    setting. Non-generative: the same backend value (hence the same singleton state) each call. *)

val backend_name : backend -> string
(** Inverse of {!get_backend}'s name parsing. *)

val backend_module : backend -> (module Ir.Backend_intf.Backend)
(** The singleton module as an existentially-packed first-class module, for generic consumers
    that thread a single backend through (the raw-API tests, [Parallel]). Code that must
    re-correlate two contexts later (e.g. [Context.copy]) should use {!wrapped_context} instead:
    this projection erases the type components. *)

(** {2 Contexts wrapped with their backend}

    A closed disjunction over the implemented backends' context types: matching two values on the
    same constructor recovers type equality directly, which is what lets [Context.copy] fall onto
    the backend-specific [device_to_device] when both contexts come from the same backend. *)

type wrapped_context =
  | Cc_ctx of Cc_b.context
  | Multicore_cc_ctx of Multicore_cc_b.context
  | Cuda_ctx of Cuda_b.context
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
