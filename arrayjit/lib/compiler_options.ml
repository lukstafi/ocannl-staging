(* Pure compiler-option construction shared with the optional GPU backends.

   Keeping this below [arrayjit.ir] lets tests exercise the exact lists handed to an RTC compiler
   without linking its optional library or requiring a device. *)

let clang_fast_math_options ~reassociate =
  [ "-ffast-math" ] @ if reassociate then [] else [ "-fno-associative-math" ]

let hiprtc ~hip_include_options ~rocwmma_include_options ~uses_rocwmma ~with_debug =
  hip_include_options
  @ (if uses_rocwmma then rocwmma_include_options @ [ "-std=c++17" ] else [])
  (* Clang processes floating-point switches left-to-right. The disabling override must stay after
     the umbrella flag or [-ffast-math] silently re-enables reassociation (gh-ocannl-735). This is a
     compiler option rather than a kernel-body pragma so it also governs bf16 operators while the
     HIP headers are parsed. *)
  @ clang_fast_math_options ~reassociate:false
  @ if with_debug then [ "-g" ] else []
