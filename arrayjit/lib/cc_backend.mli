include Ir.Backend_impl.Lowered_no_device_backend

val vector_bytes_setting : unit -> int
(** The vector register width in bytes for the explicit SIMD renderings (config [cc_vector_bytes];
    auto-probed when unset). Exposed for [Schedulers.cpu_mma_limits]'s [simd_vector_bytes]. *)

val effective_pool_width : unit -> int
(** The worker-pool width after the pool policy: the restricted class's logical CPU count when
    the restriction fired, the process's affinity-respecting CPU count otherwise. Sizes the auto
    [cc_parallel_chunks]. *)

val pool_tag : unit -> string
(** Compact pool signature ([w8P], [w24], ...) identifying the pool the process executes on.
    Exposed for [Schedulers.cpu_mma_limits]'s [worker_pool_tag]: schedules crowned on one pool do
    not transfer to another (gh-ocannl-530), so the tag enters the autotune disk-cache key. *)

val has_native_fp16_arithmetic : unit -> bool
(** Whether the configured C compiler and target execute [_Float16] arithmetic natively, at twice
    f32's lane count (ARMv8.2-FP16, AVX512-FP16) -- as opposed to lacking the type, or having it
    with every operation promoted to float (correct, but no throughput win). Probed once per
    process by test-compiling; overridable with [cc_fp16_arithmetic]. Exposed for
    [Schedulers.cpu_mma_limits]'s [native_fp16_arithmetic] and for the compute-precision decision
    in [CC_syntax_config]. *)
