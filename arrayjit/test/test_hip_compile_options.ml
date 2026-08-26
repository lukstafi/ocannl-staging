(* GPU-free coverage for HIP's compiler-level reduction-order policy (gh-ocannl-735).

   This invokes the complete production option builder used by Hip_backend.Impl, with sentinels for
   the SDK-discovery results, so it can pin the exact ordering without hipjit or a device.
   [reduction_forms] remains the hardware-backed proof that hiprtc honors it numerically, and
   [half_softmax] the proof that [-fhonor-infinities] keeps the [(-INFINITY)] mask sentinel a usable
   value under the same fast-math umbrella. *)

open Base

let build ~uses_rocwmma ~with_debug =
  Ir.Compiler_options.hiprtc ~hip_include_options:[ "-Ihip" ]
    ~rocwmma_include_options:[ "-Irocwmma" ] ~uses_rocwmma ~with_debug

let () =
  let cases =
    [
      ( false,
        false,
        [ "-Ihip"; "-ffast-math"; "-fno-associative-math"; "-fhonor-infinities" ] );
      ( false,
        true,
        [ "-Ihip"; "-ffast-math"; "-fno-associative-math"; "-fhonor-infinities"; "-g" ] );
      ( true,
        false,
        [
          "-Ihip";
          "-Irocwmma";
          "-std=c++17";
          "-ffast-math";
          "-fno-associative-math";
          "-fhonor-infinities";
        ] );
      ( true,
        true,
        [
          "-Ihip";
          "-Irocwmma";
          "-std=c++17";
          "-ffast-math";
          "-fno-associative-math";
          "-fhonor-infinities";
          "-g";
        ] );
    ]
  in
  Verdict.p_all "every HIPRTC variant keeps both fast-math overrides, in order, after the umbrella"
    cases
    ~f:(fun (uses_rocwmma, with_debug, want) ->
      List.equal String.equal (build ~uses_rocwmma ~with_debug) want)
