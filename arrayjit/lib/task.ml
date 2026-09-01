open Base
module Lazy = Utils.Lazy

let _get_local_debug_runtime = Utils.get_local_debug_runtime

[%%global_debug_log_level 0]

(* export OCANNL_LOG_LEVEL_TASK=9 to enable debugging into the log_files/ directory. *)
[%%global_debug_log_level_from_env_var "OCANNL_LOG_LEVEL_TASK"]

type t =
  | Task : { context_lifetime : ('a[@sexp.opaque]); description : string; work : unit -> unit } -> t
[@@deriving sexp_of]

let run (Task task) : unit =
  (* [%log_result "run", task.description]; *)
  task.work ()

let prepend ~work (Task task) =
  Task
    {
      task with
      work =
        (fun () ->
          work ();
          task.work ());
    }

let append ~work (Task task) =
  Task
    {
      task with
      work =
        (fun () ->
          task.work ();
          work ());
    }

(* [?snapshot] carries a dispatch's launch parameters across the hand-off to an asynchronous
   scheduler. The scheduled task reads them when the WORKER gets to it, so a caller's loop --
   [Train.sequential_loop], or any hand-written bind/run/rebind -- has by then rebound them for the
   next dispatch. Called on the scheduling thread, [snapshot ()] captures the current values and
   returns the closure that restores them; that closure is prepended to the task, so the restore
   happens on the worker, in queue order, immediately before the launch reads them. Absent, the task
   is scheduled as-is: a synchronous scheduler has nothing to carry, and paying an allocation per
   dispatch for it would be waste. *)
let enschedule ?snapshot ~schedule_task ~get_stream_name stream (Task { description; _ } as task) =
  (* [%log_result "enschedule", description, "on", get_stream_name stream]; *)
  let work () =
    match snapshot with
    | None -> schedule_task stream task
    | Some snapshot -> schedule_task stream (prepend ~work:(snapshot ()) task)
  in
  Task
    {
      context_lifetime = ();
      description = "schedules {" ^ description ^ "} on " ^ get_stream_name stream;
      work;
    }
