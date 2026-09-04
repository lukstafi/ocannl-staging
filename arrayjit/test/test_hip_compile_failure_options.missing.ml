(* Selected where hipjit is unavailable. The test's subject is then absent for an environment
   reason, not a verdict about whichever runtime backend was requested. *)

open Base

let claims =
  [
    "a hiprtc compile failure carries the effective option vector";
    "a valid HIP kernel still compiles through the instrumented path";
  ]

let () =
  List.iter claims ~f:(Verdict.skipped ~aggregation:`Environment ~backend:"hipjit unavailable")
