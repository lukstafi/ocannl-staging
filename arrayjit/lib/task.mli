(** A unit of scheduled work: what a backend stream executes for one routine launch or one transfer.
    The [context_lifetime] payload exists only to keep the referenced contexts (and with them their
    buffers) reachable for as long as the task can still run. *)

type t = Task : { context_lifetime : 'a; description : string; work : unit -> unit } -> t

val sexp_of_t : t -> Sexplib0.Sexp.t
(** The description only; [context_lifetime] and [work] are opaque. *)

val run : t -> unit

val prepend : work:(unit -> unit) -> t -> t
(** Runs [work] immediately before the task's own work, on whichever thread runs the task. *)

val append : work:(unit -> unit) -> t -> t
(** Runs [work] immediately after the task's own work, on whichever thread runs the task. *)

val enschedule :
  ?snapshot:(unit -> unit -> unit) ->
  schedule_task:('stream -> t -> unit) ->
  get_stream_name:('stream -> string) ->
  'stream ->
  t ->
  t
(** Wraps a task into one that, when run, hands the original to [schedule_task] on [stream].
    [?snapshot] carries a dispatch's launch parameters across the hand-off to an asynchronous
    scheduler: called on the scheduling thread, [snapshot ()] captures the current values and
    returns the closure restoring them, which is prepended to the scheduled task so the restore runs
    on the worker in queue order, immediately before the launch reads them. A synchronous scheduler
    has nothing to carry and passes no snapshot. *)
