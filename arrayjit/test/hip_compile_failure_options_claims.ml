(* gh-ocannl-849's claim sequence, spelled once for both alternatives of the hipjit [select]
   (gh-ocannl-1703): [test_hip_compile_failure_options.real.ml] runs it in a leg open on the hip
   backend, [test_hip_compile_failure_options.missing.ml] in a leg closed for an environment reason.
   Only an open leg calls the evidence functions. *)

open Verdict.Claims

let run ~leg ~failure_has_options ~valid_compiles =
  in_leg ~leg "a hiprtc compile failure carries the effective option vector" (fun label ->
      p label (failure_has_options ()));
  in_leg ~leg "a valid HIP kernel still compiles through the instrumented path" (fun label ->
      p label (valid_compiles ()))
