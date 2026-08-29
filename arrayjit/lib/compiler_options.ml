(* Pure compiler-option construction shared with the optional GPU backends.

   Keeping this below [arrayjit.ir] lets tests exercise the exact lists handed to an RTC compiler
   without linking its optional library or requiring a device. *)

(* Clang processes floating-point switches left-to-right, so every override here has to stay AFTER
   the umbrella flag or [-ffast-math] silently re-enables what it disables.

   [-fhonor-infinities] re-enables the one piece of IEEE behavior this codebase relies on as a VALUE
   rather than as a test: [C_syntax] deliberately emits [(-INFINITY)] for the neutral element of a
   [Max] accumulation and for [Nn_blocks.default_mask_fill], so a masked softmax's sentinel has to
   survive the subtraction and the [exp]. Under bare [-ffast-math] ([-ffinite-math-only]) that only
   ever held by accident of which optimization the compiler happened to pick -- adding
   [-fno-associative-math] for gh-ocannl-735 changed the pick, and HIP's causally masked half
   softmax started returning [exp(0)] and [exp(1)] where the mask demands exact zeros. NaN stays
   unhonored: it is only ever TESTED for here, and that has its own documented shape (a range
   compare of a runtime value). *)
let clang_fast_math_options ~reassociate =
  [ "-ffast-math" ]
  @ (if reassociate then [] else [ "-fno-associative-math" ])
  @ [ "-fhonor-infinities" ]

let hiprtc ~hip_include_options ~rocwmma_include_options ~uses_rocwmma ~with_debug =
  hip_include_options
  @ (if uses_rocwmma then rocwmma_include_options @ [ "-std=c++17" ] else [])
  (* These are compiler options rather than kernel-body pragmas so they also govern the bf16 and f16
     operators while the HIP headers are parsed. *)
  @ clang_fast_math_options ~reassociate:false
  @ if with_debug then [ "-g" ] else []

(* nvrtc's opt-IN for floating-point reassociation. nvrtc accepts it (13.3 answers a
   [--fassociative-math=false] with "does not accept any argument", so its parser knows the name and
   it is a bare flag), but NVIDIA's nvrtc option reference does not list it and there is no negative
   spelling to pair with it. It exists here only so [nvrtc] below can be checked never to emit it:
   the CUDA half of the reduction-order policy is a MEMBERSHIP claim, not an ordering one. *)
let nvrtc_reassociation_opt_in = "--fassociative-math"

(* nvrtc (CUDA). The left-to-right override discipline [clang_fast_math_options] documents does NOT
   apply here, because nvrtc has no umbrella-plus-overrides shape to defend against:
   [--use_fast_math] is a fixed expansion of four independent switches ([--ftz=true --prec-div=false
   --prec-sqrt=false --fmad=true]) and none of them concerns reassociation. The order is still
   pinned -- it is the state a compile is reproducible against -- but what this function actually
   guarantees is the absence of [nvrtc_reassociation_opt_in].

   Measured on the project's CUDA box for gh-ocannl-784 (nvrtc 13.3, driver 13030, GeForce RTX 5070
   Ti, sm_120) by compiling a 128-term float reduction through nvrtc and EXECUTING it: under
   [--use_fast_math], and also under [--use_fast_math] plus [--extra-device-vectorization], plus
   [--dopt=on --ptxas-options=-O3], and plus [--fmad=false], all three of the spellings that broke
   HIP -- a counted loop, a runtime-bound loop, and 128 repeated statements -- returned the
   bit-exact strictly-sequential value, as did an [(a+b)-a] cancellation probe. The same reduction
   compiled by host gcc at [-O3 -ffast-math] does reassociate and [-fno-associative-math] restores
   it, so the probe can see reassociation when it happens. Every clang-shaped guard
   ([-fno-associative-math], [-ffp-contract=off], [-fno-unsafe-math-optimizations], ...) is rejected
   outright by nvrtc as an unrecognized option.

   So CUDA's reduction-order safety is a DOCUMENTED BOUNDARY rather than a flag: nvrtc's fast math
   does not reassociate, nothing in its documented option list could re-guard it if a future nvrtc
   started to, and the one lever that names reassociation is an undocumented opt-in this list must
   never contain. A [reduction_forms] red on CUDA after a toolkit upgrade should re-run that probe
   before anything else: it is tools/nvrtc_reassoc_probe.c, which links nvrtc and the CUDA driver
   directly and carries its own build line. *)
let nvrtc ~cuda_include_options ~arch_options ~with_device_debug =
  cuda_include_options @ arch_options @ [ "--use_fast_math" ]
  @ if with_device_debug then [ "--device-debug" ] else []

(* One line, for diagnostics that have to travel through a log or an exception message: a compile's
   effective option vector is the state a numeric mismatch is reproducible against, and reading it
   off a failure is what turns "schedule-dependent numeric mismatch" into "optimizer flag". *)
let render options = String.concat " " options
