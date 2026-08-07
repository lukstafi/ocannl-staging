type phase =
  | Transform
  | Hardware_limits
  | Backend_codegen
  | Backend_compile
  | Backend_link
  | Preflight
      (** {!Context.check_runnable}: the pre-dispatch validation of a launch — poisoned lineage,
          uninitialized inputs, unsatisfied execution dependencies, out-of-range static bindings.
          All of it precedes [Ir.Task.run], so a failure here proves nothing was dispatched, which
          is why {!classify_raw} never makes one fatal (gh-ocannl-564): it is the caller's to fix
          and retry, and the lineage stays usable. Distinguishing it from [Launch] is the whole
          point — inside a [Launch]-tagged boundary such a failure was unattributable, hence fatal,
          hence a condemned lineage for a one-line user mistake. *)
  | Launch
  | Sync
[@@deriving sexp_of, compare, equal]

type resource = Workgroup_threads | Workgroup_memory | Thread_scratch
[@@deriving sexp_of, compare, equal]

type severity = Expected | Compiler_bug [@@deriving sexp_of, compare, equal]
type execution_effect = No_device_writes | Writes_may_have_occurred
[@@deriving sexp_of, compare, equal]

type cause =
  | Illegal_schedule of { check : string; detail : string }
  | Unsupported of { feature : string; detail : string }
  | Resource_exceeded of {
      resource : resource;
      requested : int;
      limit : int option;
      detail : string;
    }
  | Backend_rejected of {
      backend : string;
      stage : string;
      severity : severity;
      detail : string;
    }
  | Unclassified of { phase : phase; exn_constructor : string; detail : string }
  | Seed_evicted of { family : string; detail : string }
      (** A detected seed site the search declined to propose because a candidate-volume cap bound
          (gh-ocannl-541): the site was reachable and ranked, and lost only to the cap. Recorded in
          the decline census so a previously-proposed site that stops being proposed leaves a
          signal instead of vanishing. [family] names the seed family (e.g. ["split_reduce"]). *)
  | Not_dispatched of { origin : string; detail : string }
      (** A candidate the search refused to run on a GPU backend because it binds no hardware
          dimension (gh-ocannl-532): the whole routine would execute in one work-item. Nothing is
          wrong with the candidate — the backend's execution model is what rejects it — but it is a
          decline like any other, and leaving it out of the census made a GPU search that timed one
          candidate indistinguishable from one whose candidates all failed (gh-ocannl-543).
          [origin] names where the refusal happened: ["baseline"] (the serial baseline),
          ["candidate"] (a compiled candidate that degenerated to a serial form), or ["beam_move"]
          (a menu move pruned before compile because it provably cannot parallelize an already
          unparallelized incumbent). *)
[@@deriving sexp_of, equal]

type rejection_key =
  | Illegal_schedule_key of string
  | Unsupported_key of string
  | Resource_exceeded_key of resource
  | Backend_rejected_key of string * string * severity
  | Unclassified_key of phase * string
  | Seed_evicted_key of string
  | Not_dispatched_key of string
[@@deriving sexp_of, compare, equal]

val key_of_cause : cause -> rejection_key
val detail_of_cause : cause -> string
val exception_of_cause : cause -> exn

type fatal = {
  exn : exn;
  backtrace : Stdlib.Printexc.raw_backtrace;
  phase : phase;
  candidate : string option;
}

type classified_cause = { phase : phase; cause : cause; execution_effect : execution_effect }
[@@deriving sexp_of, equal]
(** [phase] is where the failure was raised, not where the enclosing {!protect} was installed: a
    narrow tag or a preserved cause reports its own phase, and a backend classifier's answer is
    pinned to the phase it was handed. This is what lets a report say whether a candidate died at
    link, at launch, or at sync. *)

type failure = Classified of classified_cause | Fatal of fatal
type 'a outcome = ('a, failure) Result.t

val fatal_of_classified : ?candidate:string -> classified_cause -> fatal
(** Escalates a classified rejection that cannot be contained after all — a launch failure whose
    execution effect is [Writes_may_have_occurred], with no API to restore the damaged lineage. The
    cause is rendered into its public exception; the backtrace is the escalation site, since the
    original raise was already contained. *)

exception Cause_at of phase * cause
exception Raised_at of phase * exn * Stdlib.Printexc.raw_backtrace

type provenance = Candidate | Cache_replay | Advisory | User_schedule
[@@deriving sexp_of, compare, equal]

val protect :
  ?strict:bool ->
  classify_backend:(phase -> exn -> classified_cause option) ->
  provenance:provenance ->
  phase:phase ->
  ?candidate:string ->
  (unit -> 'a) ->
  'a outcome
(** Runs a phase boundary while preserving typed causes and raw backtraces. [strict] is exposed for
    policy tests; production callers should omit it and use [strict_failure_classification]. *)

val tag : phase -> (unit -> 'a) -> 'a
(** Tags an exception with the narrow phase where it was raised, preserving its raw backtrace.
    Existing typed transport exceptions pass through unchanged. *)

val raise_cause : cause -> _
(** Renders a typed internal cause using the exception contract of the existing public APIs. *)

val raise_failure : failure -> _
(** Re-raises a fatal failure with its original backtrace, or renders a classified cause. *)
