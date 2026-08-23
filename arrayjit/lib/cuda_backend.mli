module Impl : Ir.Backend_impl.Lowered_backend

val gpu_arch_options : device_cc:int -> string -> string list
(** The [--gpu-architecture] options one nvrtc compile of the given CUDA source gets, given the
    minimum compute capability across the attached devices (as [major * 10 + minor]). A pure
    function of the source's arch markers, separated out of the compile path so the policy can be
    tested without a device — see arrayjit/test/test_cuda_arch_flags.ml and the implementation's
    comment for the floor-vs-family distinction (gh-ocannl-481 item 3, D4). *)

val cuda_include_options : unit -> string list
(** The [-I] options an nvrtc compile needs in order to find the CUDA toolkit headers
    ([<cuda_fp8.h>], [<cuda_fp16.h>], [<mma.h>]): CUDA_PATH where it is set, the no-spaces junction
    ocaml-cudajit creates on Windows, and /usr/local/cuda as the Linux fallback. Separated out of
    the compile path so that other nvrtc callers agree with it rather than reimplementing a subset —
    tools/fp8_soak.ml is one, and a subset that knew only /usr/local/cuda would fail to compile its
    kernel wherever the toolkit lives elsewhere. *)
