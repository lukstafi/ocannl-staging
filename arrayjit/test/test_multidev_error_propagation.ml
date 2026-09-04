(* Regression test for the Multidev scheduler's error propagation (PR #107 review): a stopped worker
   must not be mistaken for completion. When a task raises, the worker records the error and stops
   spinning, so an event whose "tick" task was queued behind the failure never completes; [sync] on
   that event (and [await] on the device) must re-raise the device error -- otherwise a consumer
   waiting on a producer device would proceed with stale or uninitialized source data.

   Uses a mock raw backend (no real memory), mirroring [test_slab_free_on_grow]. Depending on worker
   timing, the error surfaces either from [sync] or already from [all_work]'s [schedule_task]; both
   re-raise the original exception, so the check is timing-robust. *)

open Base

module Mock_raw = struct
  let name = "mockdev"
  let codegen_capabilities () = Ir.Backend_intf.no_codegen_capabilities

  type buffer_ptr = int

  let sexp_of_buffer_ptr = Int.sexp_of_t
  let get_used_memory () = 0
  let next = ref 0

  let alloc_pool_raw ~size_in_bytes:_ =
    Int.incr next;
    !next

  let free_pool_raw = None
  let memset_zero_raw _ptr ~offset:_ ~size_in_bytes:_ = ()
  let offset_buffer base ~bytes = base + bytes
  let buffer_to_buffer ~dst:_ ~src:_ ~size_in_bytes:_ = ()
  let host_to_buffer _nd ~dst:_ = ()
  let buffer_to_host _nd ~src:_ = ()
end

module Sched = Context.Backends.Schedulers.Multidev (Mock_raw)

let raised_boom f =
  try
    f ();
    false
  with e -> String.is_substring (Exn.to_string e) ~substring:"boom"

let () =
  let dev = Sched.get_device ~ordinal:0 in
  (* Sanity: a healthy device completes work and its event syncs. *)
  let ran = ref false in
  Sched.schedule_task dev
    (Ir.Task.Task { context_lifetime = (); description = "ok"; work = (fun () -> ran := true) });
  let e_ok = Sched.all_work dev in
  Sched.sync e_ok;
  Verdict.p "healthy device: task ran" !ran;
  Verdict.p "healthy device: event done" (Sched.is_done e_ok);
  Sched.schedule_task dev
    (Ir.Task.Task
       { context_lifetime = (); description = "boom"; work = (fun () -> failwith "boom") });
  Verdict.p "sync surfaces worker failure" (raised_boom (fun () -> Sched.sync (Sched.all_work dev)));
  Verdict.p "await re-raises worker failure" (raised_boom (fun () -> Sched.await dev))
