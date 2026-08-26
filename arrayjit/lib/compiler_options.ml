(* Pure compiler-option construction shared with the optional GPU backends.

   Keeping this below [arrayjit.ir] lets tests exercise the exact lists handed to an RTC compiler
   without linking its optional library or requiring a device. *)

(* Clang processes floating-point switches left-to-right, so every override here has to stay AFTER
   the umbrella flag or [-ffast-math] silently re-enables what it disables.

   [-fhonor-infinities] re-enables the one piece of IEEE behavior this codebase relies on as a
   VALUE rather than as a test: [C_syntax] deliberately emits [(-INFINITY)] for the neutral element
   of a [Max] accumulation and for [Nn_blocks.default_mask_fill], so a masked softmax's sentinel has
   to survive the subtraction and the [exp]. Under bare [-ffast-math] ([-ffinite-math-only]) that
   only ever held by accident of which optimization the compiler happened to pick -- adding
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
