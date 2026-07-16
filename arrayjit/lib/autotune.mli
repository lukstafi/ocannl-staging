(** {1 Empirical schedule search (autotuning)}

    tinygrad-style beam search over {!Ir.Schedule} transforms, timed on the real device
    (docs/proposals/schedule-ir-optops.md; the search-harness half of the OptOps port). {!tune} is a
    drop-in replacement for {!Context.compile}: it compiles candidate schedules through the
    [?lowered_transform] / [?lowered_transforms] seams, times each on the context's device, and
    returns the routine of the fastest one. Every candidate (and the winner replay) derives from a
    hermetic copy of the {e one} base lowering captured at the start — each candidate compile's own
    fresh lowering is ignored, because timing runs settle tensor-node value bounds and later
    lowerings can fold guards or re-segment fission differently, silently corrupting digest
    comparisons and replays. Winning schedules are persisted, in the structurally-rebindable saved
    form of {!Ir.Schedule_cache}, to a disk cache keyed by the code's canonical digest and the
    backend, so a re-run of the same program skips the search (cross-process replay is guarded by
    digest equality against that process's own base lowering).

    The candidate space:

    - {b Whole-routine presets}: the serial baseline, the default annotator, and a block-size sweep
      through {!Ir.Schedule.default_gpu}.
    - {b Fissioned candidates}: the kernel-fission pipeline ({!Ir.Schedule.fission_scheduled}) with
      per-segment schedules — the same preset sweep per segment, and beam rounds that extend
      {e one segment at a time}. Per-segment schedules are cached keyed by the pre-schedule
      segment's canonical digest. [`Zeros] segments keep the default zero-expansion; [`Solo]
      segments stay unscheduled. One seed uses the config-default thresholds, reproducing the
      untuned default pipeline exactly — so the winner is never worse than not tuning, even on
      launch-overhead-bound workloads where every aggressive preset loses to it. Each preset is
      additionally seeded in a {e privatized} variant ({!extend_with_privatize}): per segment, every
      materialized read-modify-write accumulator is contracted into a per-thread register tile
      ({!Ir.Schedule.optop.Privatize}) over its serial reduction loop where the op's preconditions
      permit — a routine-local accumulator beats a device-memory RMW, and on Metal it sidesteps the
      volatile scalar-RMW workaround.
    - {b Matmul sketches}: when a matmul micro-kernel is detected, parameterized instantiations of
      the composed pipelines pinned by the schedule tests — register blocktiling (Split + Swap +
      shared Stage + Privatize + materializing Unroll) on GPU backends, operand packing (non-shared
      Stage + Privatize) on CPU backends — with dividing tile sizes. When the backend reports an mma
      capability, additionally the {e tensorized} pipelines (docs/proposals/tensorize-mma.md): Split
      into Grid blocks + [Tensorize] targeting [simdgroup_matrix]/tensor cores, both unstaged (one
      full-reduction [Tile_mma] block) and cooperatively staged through shared tiles (lane-aware
      Stage) — Stage-only by design, [Privatize] would move the accumulator into thread-space the
      MMA loads cannot address. On the C backends the tensorized whole-triple and Grid-split-row
      forms are seeded regardless of [limits.mma] — their [Tile_mma] renders as the register-tiled
      vector micro-kernel. Seeding matters because the beam cannot reach these compositions
      incrementally: a bare [Tensorize] from the serial baseline loses its round and is discarded
      before Grid retypes could join it. The sketches are seeded whole-routine {e and} per fission
      segment: on a fissionable computation, the fission segmentation is enumerated once and the
      sketch pipelines are instantiated for each segment where a matmul site is detected (keyed by
      the segment's pre-schedule digest), the remaining segments keeping the default preset. A
      segment's site has its [Zero_out] in a separate [`Zeros] segment, so the pipelines skip the
      zero-expansion geometry there — sound because [Privatize] init-loads the accumulator tile from
      the (pre-zeroed) target and [Tile_mma] loads the accumulator fragment before the reduction.
    - {b Beam-round menu actions} on the incumbents: dividing serial Splits, Swaps of perfect serial
      pairs, Unrolls, Retype-Vectorized on innermost loops (explicit SIMD on CPU including the
      reduction-chains rendering of accumulations — gh-ocannl-468 — while GPU accumulations stay
      excluded; 128-bit packed loads/stores on GPU — gh-ocannl-463), and Tensorize role permutations
      when the backend reports an mma capability — including the CPU backends, whose [Tile_mma]
      renders as the register-tiled vector micro-kernel (gh-ocannl-469).

    Caveats (v1):

    - Timing runs execute the routine several times, mutating its outputs (and accumulators — a
      non-idempotent routine, e.g. gradient accumulation, will accumulate the timing runs).
      Initialize inputs before tuning, and tune before meaningful state exists, or re-initialize
      afterwards.
    - Timing uses wall clock around a device sync, so it includes queue overhead; times are
      min-of-N, where fast routines get extra runs beyond [repeats] until ~25 ms of total measured
      time — on sub-millisecond kernels a min-of-3 is launch-jitter roulette and can crown the wrong
      candidate. Static indices are bound to the midpoint of their declared ranges during timing and
      restored afterwards. *)

type sketch_params = {
  sk_gpu : bool;
  sk_mma : bool;
  sk_simd : int;
  sk_bm : int;
  sk_bn : int;
  sk_bk : int;
  sk_tm : int;
  sk_tn : int;
  sk_hoist : bool;
  sk_grid : bool;
  sk_pack_rest : bool;
}
(** Parameters of one matmul-sketch seed candidate; see the implementation's field docs. Exposed for
    tests (the seeding pre-filter of gh-ocannl-479 and the mixed grid-outermost shape of
    gh-ocannl-473 are asserted on directly). *)

val sketch_seed_params :
  is_gpu:bool ->
  is_cpu:bool ->
  limits:Ir.Backend_intf.hardware_limits ->
  Ir.Low_level.optimized ->
  sketch_params list
(** The matmul-sketch seeds proposed for the given lowering: parameterized instantiations of the
    composed pipelines with dividing tile sizes, pre-filtered against rules that statically imply a
    declined rendering (gh-ocannl-479) — on the C backends: operand-precision uniformity (f32/f64),
    the fused accumulation form, micro-kernel column extent at least one vector of lanes
    ([limits.simd_vector_bytes]), and transposed-B storage for shapes that read B in place. Exposed
    for tests. *)

val extend_with_privatize :
  static_indices:Ir.Indexing.static_symbol list ->
  Ir.Schedule.schedule ->
  Ir.Low_level.optimized ->
  Ir.Schedule.schedule
(** The privatized preset extension used by the fissioned candidates: appends a
    [Schedule.Privatize { target; over }] for every materialized read-modify-write accumulator
    detected in the schedule's application to the segment — [over] being the outermost enclosing
    [Serial] loop whose symbol the access vector does not mention and whose subtree contains no
    hardware-typed loop. Each proposal is validated by try-applying the grown schedule against a
    hermetic copy of the segment (proposals violating the op's preconditions are dropped), so the
    result always applies cleanly where the input schedule does. Exposed for tests. *)

type report = {
  cache_hit : bool;  (** The schedule came from the disk cache; no search ran. *)
  candidates_timed : int;  (** Including the serial baseline. *)
  candidates_failed : int;
      (** Candidates rejected by op preconditions, hardware limits, or backend compilation. *)
  rounds_run : int;  (** Beam-expansion rounds actually executed (0 = seeds only). *)
  sketch_candidates : int;
      (** Whole-routine matmul-sketch instantiations seeded (0 when no matmul micro-kernel was
          detected or no tile sizes divide the extents). Deterministic given the computation and
          backend. *)
  fiss_sketch_candidates : int;
      (** Per-fission-segment sketch candidates seeded (0 when the computation does not fission, or
          no segment contains a compatible matmul site). Deterministic given the computation and
          backend. *)
  fiss_sketch_timed : int;
      (** Of the seeded per-fission-segment sketch candidates, those that compiled and were actually
          timed (not rejected by op preconditions or hardware limits, not deduplicated by digest).
      *)
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
  (* Maximum beam-expansion rounds beyond the seeds; default from config [autotune_rounds] (2). The
     search also stops when a round improves the incumbent by less than 1%. *)
  ?repeats:int ->
  (* Timed runs per candidate (after one warmup), min taken; default from config [autotune_repeats]
     (3). *)
  ?seed_block_sizes:int list ->
  (* Workgroup sizes swept through {!Ir.Schedule.default_gpu} as seed candidates on GPU backends
     (default [[64; 128; 256; 512]]), both whole-routine and per-fission-segment, in addition to the
     config-default preset and the serial baseline. *)
  ?cache_dir:string ->
  (* Directory of the schedule disk cache; [""] disables caching. Default from config
     [autotune_cache_dir] ([autotune_cache]). *)
  ?timing_ctx:Context.t ->
  (* A scratch context lineage against which candidates are compiled and timed, so the timing runs
     never mutate [ctx]'s live buffers (parameters, accumulators — running a training step on
     scratch/zero data can even poison them with inf/NaN). It must contain the nodes the computation
     requires from a prior context, e.g. by repeating parameter initialization on a fresh root
     context, and must live on the same backend and device as the target context (raises
     [Invalid_argument] otherwise — candidates timed elsewhere do not predict this device). Only the
     winning schedule is then compiled from [ctx], exactly like a cache hit. Without it, the search
     shares [ctx]'s buffers and the caller should re-initialize mutated state afterwards. *)
  ?report:(report -> unit) ->
  Context.t ->
  Ir.Assignments.comp ->
  Ir.Indexing.unit_bindings ->
  Context.t * Context.routine
(** Like {!Context.compile}, but returns the empirically fastest of the searched schedule
    candidates. The returned context/routine come from an ordinary sibling compile of [ctx], so
    execution-dependency tracking behaves as if the winning compile were the only one. Raises like
    {!Context.run} would (e.g. uninitialized inputs) — tune in the same state you would run in. *)
