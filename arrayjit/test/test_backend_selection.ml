(* gh-ocannl-536 landing step 5: [Context.auto]'s backend selection must fall through only for a
   backend that is genuinely not available on this machine. Everything else — a driver that fails to
   initialize, host exhaustion, an interrupt, a confused compiler — has to propagate, because
   silently continuing on a different backend turns a diagnosable failure into a run whose numbers
   nobody can attribute.

   The policy is pure, so it is pinned here without a device. The one device-touching assertion is
   that selection still produces a usable context on whatever machine runs the suite. *)

open Base

let () =
  let unavailable =
    Ir.Backend_intf.Backend_unavailable { backend = "cuda"; detail = "no CUDA devices" }
  in
  assert (Context.advances_to_next_backend unavailable);
  (* The message names the backend and keeps the discovery detail. *)
  let rendered = Exn.to_string unavailable in
  assert (String.is_substring rendered ~substring:"cuda");
  assert (String.is_substring rendered ~substring:"no CUDA devices");
  (* Everything below stops the search. Assertion failures and interrupts are never a backend's
     fault; a driver-initialization failure and an unrelated internal error mean the installation is
     broken, not absent. *)
  List.iter
    ~f:(fun exn -> assert (not (Context.advances_to_next_backend exn)))
    [
      Out_of_memory;
      Stdlib.Sys.Break;
      Stack_overflow;
      Assert_failure ("cuda_backend.ml", 1, 2);
      Failure "cuInit failed: CUDA_ERROR_NO_DEVICE";
      Invalid_argument "Exec_as_cuda.get_device 3: not enough devices";
      Utils.User_error "set large_models=true";
    ];
  (* An unknown configured name is a spelling mistake and says so, naming the offender. It is not
     [Backend_unavailable]: falling through to another backend would run the wrong thing. *)
  (match Context.Backends_deprecated.get_backend ~backend_name:"no_such_backend" () with
  | _ -> failwith "expected an unknown-backend rejection"
  | exception exn ->
      assert (not (Context.advances_to_next_backend exn));
      assert (String.is_substring (Exn.to_string exn) ~substring:"no_such_backend"));
  (* Selection still works end to end. *)
  let ctx = Context.auto () in
  let name = Context.backend_name ctx in
  assert (
    List.mem [ "cc"; "multidev_cc"; "cuda"; "hip"; "metal" ] name ~equal:String.equal)
