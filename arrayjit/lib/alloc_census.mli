(** A process-global census of the allocation classes whose accumulation across a schedule search
    exhausted a 12 GB card (gh-ocannl-550).

    The question the census exists to answer is {e which class} grows with candidates processed, so
    the classes are separated rather than summed: the device buffers behind the backends' pool tables
    (working, i.e. context-owned, vs. constant, i.e. per-device and deduped), the backend contexts,
    and the loaded code modules. Bytes are tracked only for pools; a module's device footprint is not
    a number any backend API reports.

    Counting sites, and therefore the exact coverage:

    - Pools: the {e shared} allocator seam ([Backends.allocate_delta], and the context [finalize] that
      frees it). A device's reserved merge-buffer pool is NOT counted: it is one entry per device that
      grows in place, so it cannot be a per-candidate growth class, and its allocation site is inside
      each backend rather than at the shared seam. Read the device's [merge_buffer_capacity] for that
      one.
    - Contexts: [Backend_impl.Device.make_context] / [make_child], shared by every backend, against
      the context [finalize].
    - Modules: per-backend link sites. Instrumented on [cc] and [cuda] (the backends the
      gh-ocannl-550 measurements and its regression test use); [hip] and [metal] leave the counters
      at zero rather than reporting a wrong number.

    Consequently [live_*] is exact for what it covers and is not a device-memory total: use
    [Context.get_used_memory] for the backend's own view. Note also which counters are {e live} and
    which are not: pools and modules are (a pool is in the table or it is not; a module's unload is
    counted by the finalizer that performs it), whereas contexts are only ever decremented by an
    explicit release — see {!unreleased_contexts}. *)

val record_pool : device_id:int -> pool_id:int -> constant:bool -> size_in_bytes:int -> unit
(** Records a pool as live. Replacing an existing [(device_id, pool_id)] entry replaces its size
    (a pool grown in place is still one live pool). *)

val forget_pool : device_id:int -> pool_id:int -> unit
(** Drops a pool from the live set, counting a free. Idempotent: a second call for a pool already
    forgotten does nothing, so a backend whose [free_pool] runs twice cannot double-count. *)

val count_context_created : unit -> unit
val count_context_released : unit -> unit
val count_module_loaded : unit -> unit
val count_module_unloaded : unit -> unit

type t = {
  live_working_pools : int;
  live_working_bytes : int;
  live_constant_pools : int;
  live_constant_bytes : int;
  working_pools_allocated : int;
  constant_pools_allocated : int;
  pools_freed : int;
  contexts_created : int;
  contexts_released : int;
  modules_loaded : int;
  modules_unloaded : int;
}
[@@deriving sexp_of, equal]
(** A point-in-time reading. The [live_*] fields come from the live table, the rest are cumulative
    since process start. *)

val snapshot : unit -> t
val live_pools : t -> int
val live_pool_bytes : t -> int
val unreleased_contexts : t -> int
(** [contexts_created - contexts_released]. NOT a live count, and the difference matters: unlike a
    pool, a context is not rooted by any table, so an unreleased one may well have been collected
    already — nothing on the device depends on the record's liveness. What the number tracks is
    contexts whose pools were never explicitly freed, which is the useful signal precisely because
    those pools ARE rooted. Compare {!live_pools}, {!live_modules}: those two are genuinely live,
    the former because the table is the authority and the latter because unloads are counted from
    the GC finalizer that performs them. *)
val live_modules : t -> int
val to_string : t -> string
