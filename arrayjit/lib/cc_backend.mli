include Ir.Backend_impl.Lowered_no_device_backend

val vector_bytes_setting : unit -> int
(** The vector register width in bytes for the explicit SIMD renderings (config [cc_vector_bytes];
    auto-probed when unset). Exposed for [Schedulers.cpu_mma_limits]'s [simd_vector_bytes]. *)

val has_native_fp16_arithmetic : unit -> bool
(** Whether the configured C compiler and target execute [_Float16] arithmetic natively, at twice
    f32's lane count (ARMv8.2-FP16, AVX512-FP16) -- as opposed to lacking the type, or having it
    with every operation promoted to float (correct, but no throughput win). Probed once per
    process by test-compiling; overridable with [cc_fp16_arithmetic]. Exposed for
    [Schedulers.cpu_mma_limits]'s [native_fp16_arithmetic] and for the compute-precision decision
    in [CC_syntax_config]. *)
