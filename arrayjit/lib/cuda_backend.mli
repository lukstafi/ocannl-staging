module Impl : Ir.Backend_impl.Lowered_backend

val gpu_arch_options : device_cc:int -> string -> string list
(** The [--gpu-architecture] options one nvrtc compile of the given CUDA source gets, given the
    minimum compute capability across the attached devices (as [major * 10 + minor]). A pure
    function of the source's arch markers, separated out of the compile path so the policy can be
    tested without a device — see arrayjit/test/test_cuda_arch_flags.ml and the implementation's
    comment for the floor-vs-family distinction (gh-ocannl-481 item 3, D4). *)
