(** {1 Empirical schedule search (autotuning)}

    tinygrad-style beam search over {!Ir.Schedule} transforms, timed on the real device
    (docs/proposals/schedule-ir-optops.md; the search-harness half of the OptOps port). {!tune} is
    a drop-in replacement for {!Context.compile}: it compiles candidate schedules through the
    [?lowered_transform] / [?lowered_transforms] seams, times each on the context's device, and
    returns the routine of the fastest one. Candidates are carried in the canonical form of
    {!Ir.Schedule_cache} — every candidate compile re-lowers with fresh symbols, so schedules are
    rebound structurally inside each compile's transform closure, guarded by digest equality.
    Winning schedules are persisted to a disk cache keyed by the code's canonical digest and the
    backend, so a re-run of the same program skips the search.

    The candidate space:

    - {b Whole-routine presets}: the serial baseline, the default annotator, and a block-size
      sweep through {!Ir.Schedule.default_gpu}.
    - {b Fissioned candidates}: the kernel-fission pipeline ({!Ir.Schedule.fission_scheduled})
      with per-segment schedules — the same preset sweep per segment, and beam rounds that extend
      {e one segment at a time}. Per-segment schedules are cached keyed by the pre-schedule
      segment's canonical digest. [`Zeros] segments keep the default zero-expansion; [`Solo]
      segments stay unscheduled.
    - {b Matmul sketches}: when a matmul micro-kernel is detected, parameterized instantiations
      of the composed pipelines pinned by the schedule tests — register blocktiling
      (Split + Swap + shared Stage + Privatize + materializing Unroll) on GPU backends, operand
      packing (non-shared Stage + Privatize) on CPU backends — with dividing tile sizes.
    - {b Beam-round menu actions} on the incumbents: dividing serial Splits, Swaps of perfect
      serial pairs, Unrolls, Retype-Vectorized on non-accumulating innermost loops (CPU), and
      Tensorize role permutations when the backend reports an mma capability.

    Caveats (v1):

    - Timing runs execute the routine several times, mutating its outputs (and accumulators — a
      non-idempotent routine, e.g. gradient accumulation, will accumulate the timing runs).
      Initialize inputs before tuning, and tune before meaningful state exists, or re-initialize
      afterwards.
    - Timing uses wall clock around a device sync, so it includes queue overhead; times are
      min-of-N. Static indices are bound to the midpoint of their declared ranges during timing
      and restored afterwards. *)

type report = {
  cache_hit : bool;  (** The schedule came from the disk cache; no search ran. *)
  candidates_timed : int;  (** Including the serial baseline. *)
  candidates_failed : int;
      (** Candidates rejected by op preconditions, hardware limits, or backend compilation. *)
  rounds_run : int;  (** Beam-expansion rounds actually executed (0 = seeds only). *)
  sketch_candidates : int;
      (** Matmul-sketch instantiations seeded (0 when no matmul micro-kernel was detected or no
          tile sizes divide the extents). Deterministic given the computation and backend. *)
  fissioned : bool;  (** The winning candidate compiles as multiple fissioned kernels. *)
  baseline_ms : float;
  best_ms : float;
  best_schedule : Ir.Schedule_cache.saved_schedule;
      (** The winner's schedule; for a fissioned winner, the concatenation of the per-segment
          schedules (informational). *)
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
     (default [[64; 128; 256; 512]]), both whole-routine and per-fission-segment, in addition to
     the config-default preset and the serial baseline. *)
  ?cache_dir:string ->
  (* Directory of the schedule disk cache; [""] disables caching. Default from config
     [autotune_cache_dir] ([autotune_cache]). *)
  ?timing_ctx:Context.t ->
  (* A scratch context lineage against which candidates are compiled and timed, so the timing
     runs never mutate [ctx]'s live buffers (parameters, accumulators — running a training step
     on scratch/zero data can even poison them with inf/NaN). It must contain the nodes the
     computation requires from a prior context, e.g. by repeating parameter initialization on a
     fresh root context, and must live on the same backend and device as the target context
     (raises [Invalid_argument] otherwise — candidates timed elsewhere do not predict this
     device). Only the winning schedule is then compiled from [ctx], exactly like a cache hit.
     Without it, the search shares [ctx]'s buffers and the caller should re-initialize mutated
     state afterwards. *)
  ?report:(report -> unit) ->
  Context.t ->
  Ir.Assignments.comp ->
  Ir.Indexing.unit_bindings ->
  Context.t * Context.routine
(** Like {!Context.compile}, but returns the empirically fastest of the searched schedule
    candidates. The returned context/routine come from an ordinary sibling compile of [ctx], so
    execution-dependency tracking behaves as if the winning compile were the only one. Raises
    like {!Context.run} would (e.g. uninitialized inputs) — tune in the same state you would run
    in. *)
