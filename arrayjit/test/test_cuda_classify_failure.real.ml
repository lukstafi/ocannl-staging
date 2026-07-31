(* The CUDA arm of gh-ocannl-536's typed-outcome containment.

   [Cuda_backend.Impl.classify_failure] is the per-backend plug-in point {!Context.failure_classifier}
   hands to the autotuner: without it, a [Cu.Cuda_error] or [Nvrtc.Nvrtc_error] is opaque to the
   common policy, so it is [Unclassified] and — at [Launch]/[Sync] regardless of
   [strict_failure_classification], since only compile-side phases are softened — fatal. One
   candidate the toolchain refuses would end the whole search and take the measurements already
   collected with it.

   Both probes raise REAL exceptions through the real bindings. Cudajit's [result] is abstract, so a
   [Cuda_error] cannot be constructed by hand; a test that hand-rolled the exception would be
   testing its own fixture. The nvrtc probe needs no GPU (nvrtc is a compiler library, and it is the
   arm this file can guarantee runs everywhere the library is selected); the driver probe is skipped
   where no CUDA device is attached, because this file is selected on cudajit being *installed*, not
   on hardware being present. Neither probe damages device state: an oversized [mem_alloc] is
   refused outright, and a syntax error never reaches a device at all. *)

open Base
module Cu = Cuda
module SO = Ir.Schedule_outcome

let classify = Cuda_backend.Impl.classify_failure

let fail_with_cause label (cause : SO.classified_cause option) =
  let rendered =
    match cause with
    | None -> "None"
    | Some c -> Sexp.to_string_hum (SO.sexp_of_classified_cause c)
  in
  failwith (Printf.sprintf "%s: unexpected classification %s" label rendered)

(* A classified backend rejection with the expected key components and damage verdict. The key
   ([backend], [stage], [severity]) is what {!Autotune.report}'s decline census groups by, so it is
   asserted rather than the detail text. *)
let expect_rejection label ~phase ~stage ~severity ~execution_effect exn =
  match classify phase exn with
  | Some
      ({ SO.cause = SO.Backend_rejected { backend; stage = got_stage; severity = got_severity; _ }; _ }
      as classified)
    when String.equal backend "cuda"
         && String.equal got_stage stage
         && SO.equal_severity got_severity severity
         && SO.equal_execution_effect classified.SO.execution_effect execution_effect
         && SO.equal_phase classified.SO.phase phase ->
      ()
  | other -> fail_with_cause label other

(* nvrtc rejects the source: our codegen's fault, nothing allocated or launched, so an ordinary
   contained decline that the census still counts as a [Compiler_bug]. *)
let check_nvrtc () =
  match
    Nvrtc.compile_to_ptx ~cu_src:"this is not CUDA C++" ~name:"test_cuda_classify_failure.cu"
      ~options:[] ~with_debug:false
  with
  | _ -> failwith "check_nvrtc: expected the bad source to be rejected"
  | exception (Nvrtc.Nvrtc_error _ as exn) ->
      expect_rejection "nvrtc compilation" ~phase:SO.Backend_compile ~stage:"compiler"
        ~severity:SO.Compiler_bug ~execution_effect:SO.No_device_writes exn;
      (* cudajit puts nvrtc's compilation log in the exception message; the detail must carry it
         through, or the decline names no line of the offending kernel. *)
      let detail =
        match classify SO.Backend_compile exn with
        | Some c -> SO.detail_of_cause c.SO.cause
        | None -> ""
      in
      if not (String.is_substring detail ~substring:"nvrtc output:") then
        failwith "check_nvrtc: the compilation log did not reach the cause detail"

(* An allocation the driver refuses outright: returned before any kernel runs and leaving the
   context usable, hence [No_device_writes] — the case the tuner can actually continue past. *)
let check_driver_out_of_memory () =
  let devices = try Cu.init (); Cu.Device.get_count () with _ -> 0 in
  if devices > 0 then (
    Cu.Context.set_current (Cu.Context.get_primary (Cu.Device.get ~ordinal:0));
    (* 32 TiB: larger than any device, comfortably inside [size_t], and refused outright rather
       than by putting the device under allocation pressure. *)
    match Cu.Deviceptr.mem_alloc ~size_in_bytes:(1 lsl 45) with
    | _ -> failwith "check_driver_out_of_memory: expected the oversized allocation to be refused"
    | exception (Cu.Cuda_error _ as exn) ->
        expect_rejection "driver out-of-memory" ~phase:SO.Backend_link
          ~stage:"CUDA_ERROR_OUT_OF_MEMORY" ~severity:SO.Expected
          ~execution_effect:SO.No_device_writes exn)

(* The classifier must claim only what it can read. An OCaml exception carries no driver verdict, so
   answering [None] is what keeps a genuine bug fatal instead of silently absorbed as a decline. *)
let check_declines_to_classify () =
  List.iter [ SO.Backend_compile; SO.Backend_link; SO.Launch; SO.Sync ] ~f:(fun phase ->
      match classify phase (Failure "not a driver failure") with
      | None -> ()
      | other -> fail_with_cause "unrelated exception" other)

let () =
  check_nvrtc ();
  check_driver_out_of_memory ();
  check_declines_to_classify ();
  Stdio.printf "cuda classify_failure: ok\n"
