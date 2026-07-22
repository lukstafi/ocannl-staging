(** {1 The interface types for backends}

    The shared backend-interface types: the user-facing API ({!Backend}, {!routine}, {!buffer_loc})
    together with the interface pieces the implementation layers assemble from (marked
    implementation-facing where applicable). Implementation-only components live in {!Backend_impl}.
*)

open Base

type buffer_loc = { pool_id : int; offset : int } [@@deriving sexp, compare, equal]
(** A backend-agnostic, deterministic per-device buffer location: a [pool_id] into the device's
    backend-private [pool_id -> 'base] pool table, plus a byte [offset] within that pool. The
    concrete backend pointer ([Metal.Buffer.t] / [CUdeviceptr] / [void*]) lives only in that private
    table -- it never appears in any type of this shared interface -- so [buffer_loc] (pure
    integers) is stable across runs, diffable, and meaningful in logs and [.expected] files. Phase-1
    policy is one pool per tnode at [offset = 0], byte-for-byte equivalent to per-tnode allocation.
    An alias (future work) is the parent's [{ pool_id; offset = offset + delta }]. *)

type ctx_buffers = buffer_loc Map.M(Tnode).t [@@deriving sexp_of]

type mma_input_format =
  | Mma_f32  (** Genuine f32 multiply-accumulate (Metal [simdgroup_float8x8]). *)
  | Mma_tf32
      (** f32 storage computed with a 10-bit mantissa (CUDA wmma [precision::tf32], sm_80+). Not a
          storage precision — data lives in memory as ordinary f32; only tensor-core loads
          truncate. Gated by {!Numerics.t.tf32_matmuls}. *)
  | Mma_f16
  | Mma_bf16
  | Mma_fp8_e5m2
      (** OCANNL's single fp8 today ([Ops.Fp8_prec], e5m2). An e4m3 constructor slots in here when
          the precision exists (gh-ocannl-481 item 2); descriptor entries are keyed per operand
          pair, so mixed e5m2×e4m3 combinations need no interface change. *)
[@@deriving sexp, compare, equal]
(** Element formats tensor-core instructions accept for their multiplicand operands. This is
    deliberately NOT [Ops.prec]: formats like tf32 have no byte layout of their own, so they must
    never appear as a tensor node's storage precision. *)

type mma_capability = {
  mma_simd_width : int;
      (** Threads cooperating in one tile-MMA instruction (CUDA warp / Metal simdgroup width). *)
  mma_tile : int * int * int;
      (** The canonical intrinsic tile shape [(m, n, k)] (8×8×8 for Metal [simdgroup_matrix],
          16×16×16 for CUDA wmma), autotune's divisibility filter for seeding; a
          {!Low_level.t.Tile_mma}'s block extents must be multiples of the tile of the format
          actually emitted. *)
  mma_format_tiles : ((mma_input_format * mma_input_format) * (int * int * int)) list;
      (** Per (a-operand, b-operand) input-format intrinsic tile shapes, for formats whose tile
          diverges from [mma_tile] as well as the ones matching it (e.g. CUDA fp8 16×8×32, tf32
          16×16×8). Advertises capability only — whether a given call emits is decided by the
          backend's [mma_syntax] hook plus the {!Numerics} policy. *)
}
[@@deriving sexp, compare, equal]
(** Tensor-core capability descriptor (docs/proposals/tensorize-mma.md §6). Which operand precisions
    are supported is decided per call by the backend's [mma_syntax] hook (the emission is the source
    of truth); this record carries what schedule construction needs. *)

type hardware_limits = {
  max_threads_per_workgroup : int option;
      (** Upper bound on the number of threads in one workgroup (CUDA thread block / Metal
          threadgroup); [None] when the backend imposes no limit (the C backends render annotated
          loops serially). *)
  max_workgroup_memory_bytes : int option;
      (** Capacity in bytes of the workgroup-shared memory (CUDA [__shared__] / Metal
          [threadgroup]); [None] when the backend imposes no limit. *)
  mma : mma_capability option;
      (** Tile-MMA units ([simdgroup_matrix] / tensor cores); [None] when the backend has none wired
          — [Tile_mma] statements then render their scalar fallback. *)
  simd_vector_bytes : int;
      (** Vector register width in bytes used by the C backends' explicit vector-extension
          renderings ([Vectorized] loops, the register-tiled [Tile_mma] micro-kernel); [0] when the
          backend does no such rendering (GPU backends bind hardware axes instead). Carried here so
          schedule construction (autotune's seeding pre-filter, gh-ocannl-479) can statically rule
          out candidates the renderer must decline, e.g. a micro-kernel column extent below one
          vector's lane count. *)
  peak_flops : float option;
      (** Advisory peak arithmetic throughput in FLOP/s (single-precision, FMA counted as two), the
          hardware envelope of the analytic cost model (gh-ocannl-491): rough documented constants
          or cheap device queries — the model ranks candidate schedules, it does not predict
          runtimes. Never gates compilation and never overrides a measured timing; [None] when the
          backend offers no estimate. *)
  peak_memory_bandwidth : float option;
      (** Advisory peak main-memory bandwidth in bytes/s, the other leg of the roofline envelope
          (gh-ocannl-491). Same contract as [peak_flops]: advisory, rough, never load-bearing for
          correctness; [None] when the backend offers no estimate. *)
}
[@@deriving sexp, compare, equal]

let no_hardware_limits =
  {
    max_threads_per_workgroup = None;
    max_workgroup_memory_bytes = None;
    mma = None;
    simd_vector_bytes = 0;
    peak_flops = None;
    peak_memory_bandwidth = None;
  }

(** The backend slab allocator, replacing the per-tnode [Alloc_buffer] interface. The shared
    allocator seam (see {!Backends}) mints deterministic per-device [pool_id]s and calls these
    int-in / int-out primitives; the backend keeps the [pool_id -> 'base] table private. The
    [pool_id -> 'base] resolution (then [base + offset]) stays inside the backend. *)
module type Slab_alloc = sig
  type device

  val alloc_pool :
    ?mode:Tnode.memory_mode -> device -> pool_id:int -> size_in_bytes:int -> alignment:int -> unit
  (** Allocates the slab for [pool_id] on [device]. The optional [?mode] carries the tnode's memory
      mode so backends can pick a storage mode (Metal private vs. shared); backends that do not care
      ignore it. *)

  val free_pool : (device -> pool_id:int -> unit) option
  (** Frees the slab for [pool_id] and drops its table entry. [None] for backends that rely on GC.
  *)

  val memset_zero : device -> pool_id:int -> offset:int -> size_in_bytes:int -> unit
  (** Zero-initializes [size_in_bytes] at [base_of pool_id + offset]. *)
end

type merge_buffer_use = No | Copy [@@deriving sexp_of]

(** Kernel-parameter sources: the codegen <-> backend contract for a compiled routine's parameters.
    Implementation-facing (consumed by {!C_syntax} and the backends' link steps); it lives in this
    file because the shared {!Backend_impl.Lowered_no_device_backend} signature mentions it. *)
type kparam_source =
  | Log_file_name
  | Merge_buffer
  | Kparam_ptr of Tnode.t
  | Kparam_pool_slab of int
      (** gh-ocannl-344: the [i]-th pool base-pointer parameter of a pooled kernel (Metal). A fixed
          number of these is emitted; at link the backend binds slab [i] to the pool assigned index
          [i] (or a duplicate of an in-use pool for the unused tail). Lets a kernel reach hundreds
          of tensor nodes through a handful of bound pools, staying under Metal's ~31 binding limit.
      *)
  | Kparam_pool_slots of Tnode.t list
      (** gh-ocannl-344: the per-routine slot table accompanying {!Kparam_pool_slab}. For the [k]-th
          tnode in this list the backend writes (pool_index, byte_offset); the shader reads it to
          form the typed pointer by casting (pools at pool_index) + byte_offset. Emitted only by
          pooled (Metal) codegen; per-tnode pointer backends (C, CUDA) never produce it. *)
  | Static_idx of Indexing.static_symbol
[@@deriving sexp_of]

type 'context routine = {
  context : 'context;
  schedule : Task.t;
  bindings : Indexing.lowered_bindings;
  name : string;
  inputs : Set.M(Tnode).t;
      (** The materialized read-only and read-before-write (within the routine) non-constant nodes.
          They are inputs in a broad sense, as they could be recurrent nodes or parameters. *)
  merge_buffer_input : Tnode.t option;  (** Similar to {!field-inputs}, for the merge buffer. *)
  outputs : Set.M(Tnode).t;  (** All the materialized nodes written-to by the routine. *)
}
[@@deriving sexp_of]

module type Device_config_common = sig
  type dev [@@deriving sexp_of]
  (** Interface to a device driver. *)

  type runner [@@deriving sexp_of]
  (** Interface to a stream driver. *)

  type event [@@deriving sexp_of]
  (** An event tracks if a device's runner finished computing past a particular point in its
      schedule. These values are used internally for scheduling across devices/queues of the
      backend, and can be used for explicit scheduling. *)

  val name : string
end

type ('dev, 'runner, 'event) device = {
  dev : 'dev;
  ordinal : int;
      (** The number of the represented backend's device, in the range from 0 to the number of the
          backend's devices - 1. *)
  device_id : int;
      (** A unique identifier among all device instances of all backends. Note that multiple
          [device_id] (distinct device instances) might refer to the same physical device. *)
  runner : 'runner;
  merge_buffer : buffer_loc option ref;
      (** The merge buffer's reserved single-tenant pool location, or [None] if not yet allocated.
          The slab can be reused (grown in place) for nodes that fit. *)
  mutable merge_buffer_capacity : int;
      (** Byte capacity of the reserved merge-buffer pool; drives the grow decision. *)
  updating_for : 'event Hashtbl.M(Tnode).t;
      (** The completion event for the most recent updating (writing to) a node via this device. *)
  mutable updating_for_merge_buffer : (Tnode.t * 'event option) option;
      (** The tensor node that was most recently scheduled to be in the device's merge buffer. See
          also {!field-updating_for}. *)
  constant_buffer_cache : buffer_loc Hashtbl.M(Tnode).t;
      (** Per-device cache for read-only/constant buffer allocations. *)
  mutable next_pool_id : int;
      (** Deterministic per-device pool-id counter, advanced by the shared allocator seam in tnode
          iteration order. Pool id 0 is reserved for the merge buffer; tnode pools start at 1. *)
}
(** A device bundles its single compute [runner] with the associated buffer and event tracking: the
    [merge_buffer], the [updating_for] writer events (used for cross-device coherence by
    {!Backend.device_to_device}), and the deterministic pool-id counter. The design is
    forward-compatible with a future fixed-role prefetch/transfer runner. *)

let sexp_of_device _ _ _ device = [%sexp_of: string * int] ("device_id", device.device_id)
let equal_device d1 d2 = d1.device_id = d2.device_id

(** Pool id 0 on every device is reserved for the (single-tenant) merge buffer. *)
let merge_buffer_pool_id = 0

type ('dev, 'runner, 'event) context = {
  device : ('dev, 'runner, 'event) device;
  parent : ('dev, 'runner, 'event) context option;
  ctx_buffers : ctx_buffers;
      (** This map contains the deterministic buffer locations used in this context or an ancestor
          context. *)
  finalized : Utils.atomic_bool;
  optimize_ctx : Low_level.optimize_ctx;
      (** The optimization context threaded through compilation: all OCANNL backends compile through
          the {!Low_level} IR, so this is concretely {!Low_level.optimize_ctx} (the abstraction for
          hypothetical assignments-level backends was retired; the [Assignments.comp -> code] seam
          can be reintroduced if such a backend ever materializes). *)
  merge_buffer_node : Tnode.t option;
      (** The tensor node that a {!Backend.device_to_device} transfer with [into_merge_buffer:Copy]
          placed (or will place) into this context's device's merge buffer. It is a static,
          immutably-chained fact carried producer -> consumer: linking a consumer whose code expects
          a merge-buffer node verifies it against this field at link time. A transfer with
          [into_merge_buffer:No] does not touch the merge buffer and inherits the parent's value. *)
}
[@@deriving sexp_of]

module type Device_types = sig
  include Device_config_common

  type nonrec device = (dev, runner, event) device [@@deriving sexp_of]
  type nonrec context = (dev, runner, event) context [@@deriving sexp_of]
end

module type Device = sig
  include Device_types
  include Slab_alloc with type device := device

  (* [pool_id -> base] resolution is intentionally NOT part of this shared signature: the concrete
     backend pointer never appears in a shared type. Resolution lives backend-side (see
     {!Backend_impl.Make_slab} / each backend's private [Slab]). *)

  val make_device : dev -> runner -> ordinal:int -> device

  val make_context :
    ?ctx_buffers:ctx_buffers -> ?optimize_ctx:Low_level.optimize_ctx -> device -> context
  (** Returns a context without a parent. *)

  val make_child :
    ?ctx_buffers:ctx_buffers ->
    ?optimize_ctx:Low_level.optimize_ctx ->
    ?merge_buffer_node:Tnode.t option ->
    context ->
    context
  (** Returns a context with the same {!field:Backend_intf.context.device},
      {!field:Backend_intf.context.ctx_buffers}, {!field:Backend_intf.context.optimize_ctx},
      {!field:Backend_intf.context.merge_buffer_node} if omitted, as the given context's, which is
      also the {!field:Backend_intf.context.parent}. *)

  val get_name : device -> string
end

(** The device, event and synchronization part of the backend interface, shared by the user-facing
    {!Backend} and the implementation-facing {!Backend_impl.Lowered_backend}. Does not include:
    compilation and linking (they differ between the user-facing and lowered interfaces); copying
    and tensor-node-level synchronization (copying is different for user-facing and
    implementation-facing APIs, synchronization is provided by a component outside of backend
    implementations). *)
module type Backend_device_common = sig
  include Device

  val sync : event -> unit
  (** Blocks till the event completes, if it's not done already.

      It is rarely needed to call [sync] explicitly, because it should always be called internally
      when necessary, in particular before extracting values from host. *)

  val is_done : event -> bool
  (** Whether the event completed. *)

  val will_wait_for : context -> event -> unit
  (** Schedules waiting for the given event on the context's device.

      NOTE: it should rarely be needed to call [will_wait_for] explicitly, because it should always
      be called internally when necessary. *)

  val static_properties : unit -> Sexp.t
  (** Returns a sexp description of the properties of all devices. A function so that computing it
      (device enumeration) does not run at backend-module initialization: singleton backends
      instantiate eagerly at program startup, where touching a driver could fail runs that never use
      the backend. *)

  val hardware_limits : unit -> hardware_limits
  (** Conservative per-workgroup device limits: on multi-device backends the minimum across the
      devices, so code compiled once (compilation is not per-device) is valid wherever it links.
      All-[None] for backends that do not bind hardware axes. A function for the same reason as
      {!static_properties}: computing it (device enumeration) must not run at backend-module
      initialization. *)

  val get_used_memory : device -> int
  (** Returns (an upper bound of) the memory used for arrays, in bytes. *)

  val get_global_debug_info : unit -> Sexp.t
  (** Global debug information; backend-specific and might evolve independently on the backends. *)

  val get_debug_info : device -> Sexp.t
  (** Per-device debug information; backend-specific and might evolve independently on the backends
  *)

  val await : device -> unit
  (** Blocks till the device becomes idle, i.e. synchronizes the device's runner. *)

  val all_work : device -> event
  (** Returns the event indicating if any currently running or scheduled computations on the device
      have completed. *)

  val is_idle : device -> bool
  (** Whether the device's runner is currently waiting for work. *)

  val get_device : ordinal:int -> device
  val num_devices : unit -> int
end

module type With_buffer_retrieval_and_syncing = sig
  type device
  type context
  type event

  val from_host : context -> Tnode.t -> Ndarray.t -> bool
  (** [from_host ctx tn src] schedules a copy of the explicit host buffer [src] into [tn]'s
      in-context device buffer and returns true, or returns false if the node is not in context.
      After [gh-ocannl-333] the host buffer is supplied by the caller (e.g. {!Context.set_values});
      it is no longer read from the tensor node. *)

  val init_from_host : context -> Tnode.t -> Ndarray.t -> context
  (** Schedules a copy from the explicit host buffer to context: a variant of {!from_host} that
      requires the input context to not contain the tensor node, and outputs the context with the
      tensor node. *)

  val to_host : context -> Tnode.t -> Ndarray.t -> bool
  (** [to_host ctx tn dst] schedules a copy of [tn]'s in-context device buffer into the explicit
      host buffer [dst] and returns true, or returns false if the node is not in context. After
      [gh-ocannl-333] the destination buffer is supplied by the caller (e.g. {!Context.to_host}); it
      is no longer the tensor node's own array. *)

  val device_to_device :
    Tnode.t ->
    into_merge_buffer:merge_buffer_use ->
    dst:context ->
    src:context ->
    context routine option
  (** [device_to_device tn ~into_merge_buffer ~dst ~src] builds a transfer {e routine} instead of
      scheduling the copy directly. The caller schedules it (e.g. via [Task.run r.schedule]) or
      links a consumer against [r.context]. It returns:
      - [None] if there is nothing to transfer: the node is absent from [src]; or, for
        [into_merge_buffer=No], the node is absent from [dst] or the source and destination buffers
        are physically the same.
      - [Some r] otherwise. Running [r.schedule] waits for writing into the tensor node on [src] to
        finish, then performs the copy and updates the writer event.
      - For [into_merge_buffer=No], the copy goes from [src] to [dst]; [r.context] is a child of
        [dst] inheriting its {!field:Backend_intf.context.merge_buffer_node}.
      - For [into_merge_buffer=Copy], the copy goes from [src] to the merge buffer of [dst]'s
        stream; [r.context] is a child of [dst] with [merge_buffer_node = Some tn], so that linking
        a consumer of the merge buffer against [r.context] statically verifies the node. *)

  val init_from_device : Tnode.t -> dst:context -> src:context -> context
  (** Schedules a copy from [src] to [dst]: a variant of {!device_to_device} with
      [into_merge_buffer=No] that requires the input [src] context to not contain the tensor node,
      and outputs the [dst] context with the tensor node. *)

  val sync_device : device -> unit
  (** Synchronizes all the streams on a device, and cleans up (removes) all associated events. *)
end

module type Backend = sig
  type code [@@deriving sexp_of]
  type code_batch [@@deriving sexp_of]

  val empty_optimize_ctx : unit -> Low_level.optimize_ctx
  val get_optimize_ctx : code -> Low_level.optimize_ctx
  val get_optimize_ctx_batch : code_batch -> Low_level.optimize_ctx

  val compile :
    Low_level.optimize_ctx ->
    ?name:string ->
    ?lowered_transform:(Low_level.optimized -> Low_level.optimized) ->
    ?lowered_transforms:(Low_level.optimized -> Low_level.optimized list) ->
    Indexing.unit_bindings ->
    Assignments.comp ->
    code
  (** [name] is used to derive names for compilation artifacts. If omitted, it's derived via
      {!Assignments.get_name_exn}. [lowered_transform] is applied to the optimized lowered code
      before backend compilation — the seam where schedule transforms (and hand-annotating tests)
      rewrite loops with hardware axis types, barriers and shared placements
      (docs/proposals/axis-types-for-loops.md). [lowered_transforms] is the plural variant for
      transforms that split the routine into several kernels (fission,
      {!Schedule.fission_scheduled}): the returned segments compile as one fissioned routine and run
      back-to-back on the routine's stream with a device-side event chained at each boundary,
      exactly as {!Schedule.maybe_default_schedules}' segments do. It must return a non-empty list;
      passing both transforms raises [Invalid_argument]. *)

  val compile_batch :
    Low_level.optimize_ctx ->
    ?names:string array ->
    ?occupancy:(name:string -> src_n:int -> bool) ->
    Indexing.unit_bindings ->
    Assignments.comp array ->
    code_batch
  (** [compile_batch] vs. [compile] is mostly about improving the compile time and debugging
      convenience by generating fewer files -- ideally does not affect execution, but there can be
      backend-specific differences. Only array entries for which [occupancy] returns true are
      included. [names] are used to derive names for compilation artifacts. If omitted, they're
      derived via {!Assignments.get_name_exn}. *)

  include Backend_device_common

  val link : context -> code -> context routine
  (** Returns the routine for the code's procedure, in a new context derived from the given context.
  *)

  val link_batch : context -> code_batch -> context * context routine option array
  (** Returns the routines for the procedures included in the code batch. The returned context is
      downstream of all the returned routines. *)

  include
    With_buffer_retrieval_and_syncing
      with type device := device
       and type context := context
       and type event := event
end
