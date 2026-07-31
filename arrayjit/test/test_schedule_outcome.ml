open Base
open Ir.Schedule_outcome

let no_backend_classification _phase _exn = None

let expect_classified = function
  | Error (Classified classified) -> classified
  | Ok _ | Error (Fatal _) -> failwith "expected a classified failure"

let expect_fatal = function
  | Error (Fatal fatal) -> fatal
  | Ok _ | Error (Classified _) -> failwith "expected a fatal failure"

let () =
  let illegal_1 = Illegal_schedule { check = "Schedule.apply"; detail = "missing loop i" } in
  let illegal_2 = Illegal_schedule { check = "Schedule.apply"; detail = "missing loop j" } in
  assert (equal_rejection_key (key_of_cause illegal_1) (key_of_cause illegal_2));
  let resource_1 =
    Resource_exceeded
      {
        resource = Workgroup_threads;
        requested = 1_024;
        limit = Some 512;
        detail = "too many threads";
      }
  in
  let resource_2 =
    Resource_exceeded
      {
        resource = Workgroup_threads;
        requested = 2_048;
        limit = Some 1_024;
        detail = "still too many threads";
      }
  in
  assert (equal_rejection_key (key_of_cause resource_1) (key_of_cause resource_2));
  let unsupported_1 = Unsupported { feature = "mma"; detail = "precision f64" } in
  let unsupported_2 = Unsupported { feature = "mma"; detail = "extent 7" } in
  assert (equal_rejection_key (key_of_cause unsupported_1) (key_of_cause unsupported_2));
  let backend_rejection ?(backend = "cc") ?(stage = "compiler") ?(severity = Compiler_bug)
      detail =
    Backend_rejected { backend; stage; severity; detail }
  in
  let backend_key = key_of_cause (backend_rejection "diagnostic one") in
  assert (equal_rejection_key backend_key (key_of_cause (backend_rejection "diagnostic two")));
  assert (
    not
      (equal_rejection_key backend_key
         (key_of_cause (backend_rejection ~backend:"cuda" "diagnostic one"))));
  assert (
    not
      (equal_rejection_key backend_key
         (key_of_cause (backend_rejection ~stage:"linker" "diagnostic one"))));
  assert (
    not
      (equal_rejection_key backend_key
         (key_of_cause (backend_rejection ~severity:Expected "diagnostic one"))));
  let unclassified_1 =
    Unclassified { phase = Transform; exn_constructor = "Failure"; detail = "one" }
  in
  let unclassified_2 =
    Unclassified { phase = Transform; exn_constructor = "Failure"; detail = "two" }
  in
  assert (equal_rejection_key (key_of_cause unclassified_1) (key_of_cause unclassified_2));
  assert (
    not
      (equal_rejection_key (key_of_cause unclassified_1)
         (key_of_cause
            (Unclassified
               { phase = Backend_compile; exn_constructor = "Failure"; detail = "one" }))));
  let typed =
    protect ~strict:true ~classify_backend:no_backend_classification ~provenance:Candidate
      ~phase:Transform (fun () -> raise (Cause_at (Transform, illegal_1)))
    |> expect_classified
  in
  assert (equal_cause typed.cause illegal_1);
  let strict_unknown =
    protect ~strict:true ~classify_backend:no_backend_classification ~provenance:Candidate
      ~phase:Backend_compile (fun () -> failwith "compiler vanished")
    |> expect_fatal
  in
  assert (equal_phase strict_unknown.phase Backend_compile);
  let permissive_unknown =
    protect ~strict:false ~classify_backend:no_backend_classification ~provenance:Candidate
      ~phase:Backend_compile (fun () -> failwith "compiler vanished")
    |> expect_classified
  in
  (match permissive_unknown.cause with
  | Unclassified { phase = Backend_compile; exn_constructor; _ } ->
      assert (String.is_suffix exn_constructor ~suffix:"Failure")
  | _ -> failwith "expected an unclassified compile failure");
  ignore
    (protect ~strict:false ~classify_backend:no_backend_classification ~provenance:Candidate
       ~phase:Launch (fun () -> failwith "launch failed")
     |> expect_fatal);
  ignore
    (protect ~strict:false ~classify_backend:no_backend_classification ~provenance:User_schedule
       ~phase:Transform (fun () -> failwith "user transform failed")
     |> expect_fatal);
  ignore
    (protect ~strict:false ~classify_backend:no_backend_classification ~provenance:Candidate
       ~phase:Transform (fun () -> assert false)
     |> expect_fatal);
  let cached_assert =
    protect ~strict:true ~classify_backend:no_backend_classification ~provenance:Cache_replay
      ~phase:Transform (fun () -> assert false)
    |> expect_classified
  in
  (match cached_assert.cause with
  | Unclassified { phase = Transform; _ } -> ()
  | _ -> failwith "expected an unclassified cache-replay assertion");
  let compiler_rejection =
    {
      phase = Backend_compile;
      cause =
        Backend_rejected
          {
            backend = "test";
            stage = "compiler";
            severity = Expected;
            detail = "rejected";
          };
      execution_effect = No_device_writes;
    }
  in
  let classified =
    protect ~strict:true
      ~classify_backend:(fun phase _exn ->
        Option.some_if (equal_phase phase Backend_compile) compiler_rejection)
      ~provenance:Candidate ~phase:Backend_compile (fun () -> failwith "backend exception")
    |> expect_classified
  in
  assert (equal_classified_cause classified compiler_rejection);
  (* A post-link resource rejection (Metal's per-pipeline maxTotalThreadsPerThreadgroup and static
     threadgroup-memory checks) must be BOTH: an ordinary decline for a tuner candidate, and the
     unchanged public [Utils.User_error] for a hand-written schedule. Untyped it would be neither —
     strict classification makes an unrecognized link failure fatal, so one register-heavy candidate
     would end the search. *)
  let link_resource =
    Resource_exceeded
      {
        resource = Workgroup_threads;
        requested = 1_024;
        limit = Some 640;
        detail = "Metal: threadgroup size 1024 for k exceeds maxTotalThreadsPerThreadgroup 640";
      }
  in
  let link_declined =
    protect ~strict:true ~classify_backend:no_backend_classification ~provenance:Candidate
      ~phase:Transform (fun () ->
        tag Backend_link (fun () -> raise (Cause_at (Backend_link, link_resource))))
    |> expect_classified
  in
  assert (equal_cause link_declined.cause link_resource);
  assert (equal_execution_effect link_declined.execution_effect No_device_writes);
  (* The reported phase is where it was raised, not where [protect] was installed. *)
  assert (equal_phase link_declined.phase Backend_link);
  assert (
    equal_rejection_key (key_of_cause link_declined.cause) (Resource_exceeded_key Workgroup_threads));
  (match raise_failure (Classified link_declined) with
  | _ -> failwith "expected the classified cause to be rendered"
  | exception Utils.User_error msg ->
      assert (String.is_substring msg ~substring:"maxTotalThreadsPerThreadgroup"));
  Stdlib.Printexc.record_backtrace true;
  let tagged =
    protect ~strict:true ~classify_backend:no_backend_classification ~provenance:Candidate
      ~phase:Transform (fun () -> tag Backend_link (fun () -> failwith "link failed"))
    |> expect_fatal
  in
  assert (equal_phase tagged.phase Backend_link);
  assert (Stdlib.Printexc.raw_backtrace_length tagged.backtrace > 0)
