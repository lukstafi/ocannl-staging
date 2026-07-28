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
    - {b Convolution sketches} (gh-ocannl-493): when a convolution accumulation site is detected
      ({!detect_conv}), the implicit-GEMM pipeline — the packing [Stage] serving as im2col, the
      micro-kernel the ordinary [Tile_mma]. On the C backends: serial and Grid-parallel flavors, the
      latter adopting the default preset's aligned whole-segment Grid geometry on merged segments
      (lenet's conv+bias/relu+pooling). On GPU backends with an mma capability: the staged flavor —
      outer output loops Grid-typed, both slices staged through cooperative shared tiles at the
      kernel-window anchor, the accumulator fragment resident across the window (gh-ocannl-480).
      Strided rows (stride-2 stems and downsample blocks) are seeded on both legs since the
      compacting [Stage] (gh-ocannl-502) packs the strided window densely.
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
  sk_conv : bool;
  sk_epilogue : bool;
}
(** Parameters of one matmul-sketch seed candidate; see the implementation's field docs. Exposed for
    tests (the seeding pre-filter of gh-ocannl-479 and the mixed grid-outermost shape of
    gh-ocannl-473 are asserted on directly). *)

type conv_axis = {
  cx_o : Ir.Indexing.symbol;  (** Output spatial symbol (a plain iterator of the output). *)
  cx_no : int;
  cx_k : Ir.Indexing.symbol;  (** Kernel-window symbol (read by the kernel, not the output). *)
  cx_nk : int;
  cx_stride : int;
  cx_dilation : int;
  cx_offset : int;  (** Padding offset on the input access ([<= 0] for padded convs). *)
}

type conv_site = {
  c_loops : Ir.Indexing.symbol list;
  c_outer : (Ir.Indexing.symbol * int) list;
  c_kernel : Ir.Indexing.symbol list;
  c_axes : conv_axis list;
  c_row : Ir.Indexing.symbol;
  c_nrow : int;
  c_oc : Ir.Indexing.symbol;
  c_noc : int;
  c_red : Ir.Indexing.symbol;
  c_nred : int;
  c_d : Ir.Tnode.t;
  c_a : Ir.Tnode.t;
  c_b : Ir.Tnode.t;
  c_zeroed : bool;
  c_fma : bool;
}
(** A recognized convolution accumulation site (gh-ocannl-493); see the implementation's field docs.
    Exposed for tests. *)

val detect_conv : Ir.Low_level.t -> conv_site option
(** Recognize a convolution accumulation nest: the output written at plain distinct iterators, one
    operand carrying affine components that mix an output symbol with a kernel-window symbol (the
    projections carry the strides, dilations, and padding offsets), the other operand reading the
    kernel window, exactly one out-channel and one reduction-channel symbol, with the out-channel at
    the output's last axis and a conv axis at its second-to-last (the implicit-GEMM row). Reads off
    the extracted access relations ([Ir.Low_level.affine_accesses] — the gh-494 artifact the
    op-legality oracle also consumes); under config [legality_crosscheck] the retained procedural
    matcher runs alongside and any divergence raises. Exposed for tests. *)

val sketch_seed_params :
  is_gpu:bool ->
  is_cpu:bool ->
  limits:Ir.Backend_intf.hardware_limits ->
  Ir.Low_level.optimized ->
  sketch_params list
(** The matmul-sketch seeds proposed for the given lowering: parameterized instantiations of the
    composed pipelines with dividing tile sizes, pre-filtered against rules that statically imply a
    declined rendering (gh-ocannl-479) — on GPU backends: the operand-format tile advertised by
    [limits.mma.mma_format_tiles], including policy-enabled TF32; on the C backends:
    operand-precision uniformity (f32/f64), the fused accumulation form, micro-kernel column extent
    at least one vector of lanes ([limits.simd_vector_bytes]), and transposed-B storage for shapes
    that read B in place. Exposed for tests. *)

val sketch_schedule : p:sketch_params -> Ir.Low_level.optimized -> Ir.Schedule.schedule
(** The composed pipeline a seed parameterizes, built against the given lowering (the site is
    re-detected). Raises [Invalid_argument] when no site is detected or the parameters do not fit
    the segment. Exposed for tests (the pad-composition seeding of gh-ocannl-485 is executed
    directly). *)

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
          detected or no tile sizes divide the extents), after the model pre-filter when one is
          active ([keep_fraction < 1]). Deterministic given the computation, backend, and
          configuration. *)
  epilogue_sketch_candidates : int;
      (** Of [sketch_candidates], the fused-epilogue twins (gh-ocannl-486): seeded when the site's
          output feeds an eligible elementwise tail ([Schedule.can_fuse_epilogue]) — each sketch is
          then proposed both unfused and with [Schedule.Fuse_epilogue] appended, so the tuner
          measures the one-kernel fused form against the fissioned two-kernel form. *)
  fiss_sketch_candidates : int;
      (** Per-fission-segment sketch candidates seeded (0 when the computation does not fission, or
          no segment contains a compatible matmul site). Deterministic given the computation and
          backend. *)
  fiss_sketch_timed : int;
      (** Of the seeded per-fission-segment sketch candidates, those that compiled and were actually
          timed (not rejected by op preconditions or hardware limits, not deduplicated by digest).
      *)
  model_scored : int;
      (** Sketch candidates the analytic cost model scored during the seed pre-filter
          (gh-ocannl-491); [0] when the pre-filter is off ([keep_fraction >= 1]) or nothing was
          scoreable (e.g. no envelope constants). *)
  model_pruned : int;
      (** Of [model_scored], the candidates dropped before compilation and timing. Candidates
          without model coverage are never counted here — they are always kept. *)
  fissioned : bool;  (** The winning candidate compiles as multiple fissioned kernels. *)
  baseline_ms : float;
  best_ms : float;
  best_schedule : Ir.Schedule_cache.saved_schedule;
      (** The winner's schedule; for a fissioned winner, the concatenation of the per-segment
          schedules (informational). *)
}

val model_score :
  static_indices:Ir.Indexing.static_symbol list ->
  limits:Ir.Backend_intf.hardware_limits ->
  Ir.Low_level.optimized ->
  Ir.Schedule.schedule ->
  float option
(** The analytic cost model's ranking score of a candidate schedule (gh-ocannl-491, the selection
    half): {!Ir.Schedule.apply} on a hermetic copy, {!Ir.Cost_model.analyze}, then the roofline
    lower-bound seconds under the envelope constants — [limits]' advisory [peak_flops] /
    [peak_memory_bandwidth], each overridable by config [model_peak_flops] /
    [model_peak_memory_bandwidth] (calibrated per-machine values beat the class constants). [None] —
    no model coverage — when the schedule fails to apply, the code is opaque to the extraction (its
    counts may under-estimate, so ranking on them could prune the true winner), or no envelope
    constant is present. A ranking score, not a runtime prediction. Exposed for tests. *)

val model_prefilter : keep_fraction:float -> ('a * float option) list -> ('a * float option) list
(** The order-preserving pre-filter over model-scored candidates: keeps every unscored ([None])
    candidate — the no-coverage exemption: never dropped, only measured — plus the best
    [ceil (keep_fraction * n)] of the [n] scored ones (at least one; ties at the cutoff are all
    kept, so the outcome is independent of enumeration order). The identity when
    [keep_fraction >= 1]. Exposed for tests. *)

type model_choice = {
  mc_label : string;
      (** ["default"], or the winning candidate's spec label (matching the [autotune_log] labels).
      *)
  mc_model_ms : float option;
      (** The winner's roofline lower bound in ms — a ranking score, not a runtime prediction;
          [None] when selection did not run. *)
  mc_scored : int;
      (** Model evaluations that produced a score (the default pipeline included; the fissioned flow
          also scores per segment). *)
  mc_skipped : int;  (** Model evaluations without coverage, excluded from the ranking. *)
}

val model_default_enabled : bool Lazy.t
(** Config [model_default_schedule]: recipe-level untuned compiles ({!Train.to_routine},
    {!Train.run_once}, the benchmark runners) route through {!model_default} instead of
    {!Context.compile}. *)

val validate_segments : Ir.Low_level.optimized list -> Ir.Low_level.optimized list
(** {!Ir.Low_level.validate_parallel} over each segment (against its own placements), returning them
    unchanged; raises [Invalid_argument] on the first rejection. The check codegen runs anyway,
    pulled forward to the transform seam so that an advisory transform's rejected output surfaces
    where a fallback can catch it instead of aborting the compile (gh-ocannl-519). *)

val compile_advisory :
  ?on_fallback:(exn -> unit) ->
  ?fallback_if:(unit -> bool) ->
  (Ir.Low_level.optimized -> Ir.Low_level.optimized list) ->
  Context.t ->
  Ir.Assignments.comp ->
  Ir.Indexing.unit_bindings ->
  Context.t * Context.routine
(** {!Context.compile} with the given [lowered_transforms], falling back to a plain
    {!Context.compile} — the ordinary default pipeline — if the transformed compile raises anywhere,
    including inside backend codegen ({!Ir.Low_level.validate_parallel} and the backends' own
    preconditions run there, past the transform seam). [on_fallback] is called with the exception
    when the fallback fires. [fallback_if] (default: always) is consulted first, for transforms that
    may themselves have degraded to the default pipeline — [false] re-raises the original exception,
    backtrace included, instead of duplicating a compile that has nothing to fall back to. For
    advisory transforms only: a failure of the default pipeline itself propagates. See
    {!model_default} (gh-ocannl-519). *)

val model_default :
  ?report:(model_choice -> unit) ->
  Context.t ->
  Ir.Assignments.comp ->
  Ir.Indexing.unit_bindings ->
  Context.t * Context.routine
(** A drop-in for {!Context.compile} that raises the untuned floor (gh-ocannl-491 task 3): inside
    the compile's own transform seam, the untuned default pipeline and the sketch families
    (whole-routine, and per-fission-segment substitutions when the default fissions) are scored with
    the roofline model, and the model-argmin schedule is applied — zero measurement, one backend
    compile. Advisory by construction: a candidate without model coverage is never picked over the
    default, ties go to the default, and missing envelope constants, a disabled default annotator
    ({!Ir.Schedule.automatic_schedule_active}), or any scoring, application, validation
    ({!validate_segments}) or compilation ({!compile_advisory}) failure fall back to the ordinary
    default pipeline — the reported {!model_choice} then says ["default"]. Once the compile is on
    that pipeline there is nothing left to fall back to, so its failures propagate as they would
    from {!Context.compile}, without a duplicate attempt. Unlike {!tune}, nothing
    is executed and no cache is involved — results depend only on the computation, backend, and
    envelope constants. *)

val set_test_bindings : Context.routine -> unit
(** Binds representative values for timing runs: ranged static indices at [range / 2], and gh-490
    symbolic extents at their upper bound [range] (the schedule-cache identity is
    extent-value-independent, so the single tuned entry is measured at the maximum). Unranged
    bindings are left at their current values. Exposed for tests and custom timing harnesses. *)

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
  ?keep_fraction:float ->
  (* The model pre-filter of the sketch seeding (gh-ocannl-491): per candidate family (the
     whole-routine sketches; each fission segment's sketches), rank with {!model_score} and keep the
     best [keep_fraction] of the scored candidates before compiling or timing anything. Default from
     config [autotune_keep_fraction] (1 = pre-filter off). Candidates without model coverage are
     always kept — never dropped, only measured — so the pre-filter never overrides (or precludes) a
     measured result; presets, saved schedules and the baseline are never pruned. *)
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
