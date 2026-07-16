include Ir.Backend_impl.Lowered_no_device_backend

val vector_bytes_setting : unit -> int
(** The vector register width in bytes for the explicit SIMD renderings (config [cc_vector_bytes];
    auto-probed when unset). Exposed for [Schedulers.cpu_mma_limits]'s [simd_vector_bytes]. *)
