(** {1 Empirical schedule search (autotuning)}

    tinygrad-style beam search over {!Ir.Schedule} transforms, timed on the real device
    (docs/proposals/schedule-ir-optops.md; the search-harness half of the OptOps port). {!tune} is
    a drop-in replacement for {!Context.compile}: it compiles candidate schedules through the
    [?lowered_transform] seam, times each on the context's device, and returns the routine of the
    fastest one. Candidates are carried in the canonical form of {!Ir.Schedule_cache} — every
    candidate compile re-lowers with fresh symbols, so schedules are rebound structurally inside
    each compile's transform closure, guarded by digest equality. Winning schedules are persisted
    to a disk cache keyed by the code's canonical digest and the backend, so a re-run of the same
    program skips the search.

    Caveats (v1):

    - Timing runs execute the routine several times, mutating its outputs (and accumulators — a
      non-idempotent routine, e.g. gradient accumulation, will accumulate the timing runs).
      Initialize inputs before tuning, and tune before meaningful state exists, or re-initialize
      afterwards.
    - Passing an explicit transform disables the default annotator's kernel fission
      ({!Ir.Schedule.maybe_default_schedules}); the tuner searches whole-routine schedules. When
      the default annotator's conservative analysis rejects a routine wholesale, the seeds reduce
      to the serial baseline (menu actions can still improve serial code).
    - Timing uses wall clock around a device sync, so it includes queue overhead; times are
      min-of-N. *)

type report = {
  cache_hit : bool;  (** The schedule came from the disk cache; no search ran. *)
  candidates_timed : int;  (** Including the serial baseline. *)
  candidates_failed : int;
      (** Candidates rejected by op preconditions, hardware limits, or backend compilation. *)
  rounds_run : int;  (** Beam-expansion rounds actually executed (0 = seeds only). *)
  baseline_ms : float;
  best_ms : float;
  best_schedule : Ir.Schedule_cache.saved_schedule;
}

val tune :
  ?beam_width:int ->
  (* Default from config [autotune_beam_width] (2). *)
  ?rounds:int ->
  (* Maximum beam-expansion rounds beyond the seeds; default from config [autotune_rounds] (2).
     The search also stops when a round improves the incumbent by less than 1%. *)
  ?repeats:int ->
  (* Timed runs per candidate (after one warmup), min taken; default from config
     [autotune_repeats] (3). *)
  ?seed_block_sizes:int list ->
  (* Workgroup sizes swept through {!Ir.Schedule.default_gpu} as seed candidates on GPU backends
     (default [[64; 128; 256; 512]]), in addition to the config-default preset and the serial
     baseline. *)
  ?cache_dir:string ->
  (* Directory of the schedule disk cache; [""] disables caching. Default from config
     [autotune_cache_dir] ([autotune_cache]). *)
  ?report:(report -> unit) ->
  Context.t ->
  Ir.Assignments.comp ->
  Ir.Indexing.unit_bindings ->
  Context.t * Context.routine
(** Like {!Context.compile}, but returns the empirically fastest of the searched schedule
    candidates. The returned context/routine come from an ordinary sibling compile of [ctx], so
    execution-dependency tracking behaves as if the winning compile were the only one. Static
    indices are bound to the midpoint of their declared ranges during timing runs. Raises like
    {!Context.run} would (e.g. uninitialized inputs) — tune in the same state you would run in. *)
