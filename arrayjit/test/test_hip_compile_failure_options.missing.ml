(* Selected where hipjit is unavailable. The test's subject is then absent for an environment
   reason, not a verdict about whichever runtime backend was requested. *)

let unreachable () = failwith "hip_compile_failure_options: the hipjit-unavailable leg is closed"

let () =
  Hip_compile_failure_options_claims.run
    ~leg:(Verdict.leg ~aggregation:`Environment ~when_:false ~on:"hipjit unavailable" ())
    ~failure_has_options:unreachable ~valid_compiles:unreachable
