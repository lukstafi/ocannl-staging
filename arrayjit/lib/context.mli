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

val metal : ?device_id:int -> unit -> t
(** Create a Metal context. *)

val cpu : ?threads:int -> unit -> t
(** Create a CPU context. [threads] > 1 selects the [multidev_cc] backend (multiple worker-domain
    CPU devices, for debugging parallel workflows); otherwise the [cc] backend. Kernel-level CPU
    parallelism is automatic either way. *)

val auto : unit -> t
(** Automatically select the best available backend. *)

(** {2 Core operations} *)

val compile :
  ?lowered_transform:(Ir.Low_level.optimized -> Ir.Low_level.optimized) ->
  ?lowered_transforms:(Ir.Low_level.optimized -> Ir.Low_level.optimized list) ->
  t ->
  Ir.Assignments.comp ->
  Ir.Indexing.unit_bindings ->
  t * routine
(** Compile assignments into an executable routine. Returns updated context and the compiled
    routine. The returned context carries the updated compilation frontier for dependency tracking;
    the input context is unchanged (see {!section:execution_deps}). [lowered_transform] rewrites
    the optimized lowered code before backend compilation — the seam for schedule transforms and
    for hand-annotating hardware axis types in tests
    (docs/proposals/axis-types-for-loops.md). [lowered_transforms] is the plural seam for
    transforms that split the routine into several kernels (fission): the returned segments run
    back-to-back on the routine's stream with device-side events at the boundaries, like
    {!Ir.Schedule.maybe_default_schedules}' segments. Pass at most one of the two. *)

val run : t -> routine -> t
(** Execute a compiled routine. Mutates buffers in-place. Returns updated context with newly
    initialized nodes tracked. Raises [Failure] if execution dependencies are not satisfied. *)

val sync : t -> unit
(** Blocks until the context's device is idle. Host reads ({!to_host}, {!get_values}) synchronize
    on their own; explicit [sync] is for timing runs (e.g. the autotuner) and for fencing against
    out-of-band observation. *)

val hardware_limits : t -> Ir.Backend_intf.hardware_limits
(** The backend's conservative per-workgroup device limits (all-[None] on backends that do not
    bind hardware axes). Chiefly for schedule transforms and the autotuner. *)

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

val copy :
  ?into_merge_buffer:Ir.Backend_intf.merge_buffer_use -> src:t -> dst:t -> Ir.Tnode.t -> t
(** Copies the node's device buffer from [src] into [dst] (default [~into_merge_buffer:No]), or
    into [dst]'s stream's merge buffer for [~into_merge_buffer:Copy], returning the updated
    destination context. When both contexts come from the same backend the copy stays on-device
    via the backend's [device_to_device] transfer machinery (for [Copy], the returned context
    carries the merge-buffer node against which the next [compile] of merge-consuming code is
    statically verified); a cross-backend copy falls back to a host round-trip ([Copy] raises).
    Nodes absent from [src]'s device buffers fall back to the host round-trip as well, serving
    host-init literals and for-print proxies. *)

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
      computation is tracked, so their value can be recomputed on demand — [Train.printf] does
      this via the for-print proxy mechanism ({!register_for_print}); raw {!to_host} on them
      raises unless a proxy or host-init data exists. Observability is inductive: it holds only
      when every node the tracked computation reads is itself observable — a [Virtual] node
      depending (even transitively through other [Virtual] nodes) on a [Local] node inherits its
      unobservability.
    - [Local] nodes are routine-scoped scratch and {b unobservable}: their computation is not
      tracked, and they are stored (to whatever degree the optimizer decides on) only within a
      single routine invocation. This is a deliberate opt-out from the observability guarantee
      that licenses backend optimizations (e.g. stack allocation). The mode is only ever assigned
      by the compiler, to nodes never read outside their defining routine; to prevent it, request
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
(** Get the device ID. *)

val placements : t -> Ir.Tnode.Placements.t
(** The context's compilation lineage's memory-mode resolution
    (docs/proposals/context-scoped-memory-modes.md): which nodes this lineage decided to inline
    ([Virtual]), keep as routine-scoped scratch ([Local]), or give a device buffer ([On_device]).
    Reads are side-effect free; chiefly for tests and diagnostics. *)
