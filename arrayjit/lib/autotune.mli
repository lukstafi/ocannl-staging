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
      through {!Ir.Schedule.default_gpu}. On GPU backends the serial baseline (and any candidate
      that degenerates to it) is enumerated but never dispatched: with no hardware dimension bound
      the whole routine runs in one work-item, which cannot win and whose cost is unbounded — hours
      of uninterruptible dispatch on a device shared with the display (gh-ocannl-532). It keeps its
      role as the search's starting point for menu moves and as the code every candidate derives
      from; it just carries no measurement. On CPU backends it runs at full single-core speed and is
      timed as before. Every such refusal enters the report's decline census under
      [Not_dispatched_key] (gh-ocannl-543): [candidates_timed] alone cannot distinguish a GPU search
      whose serial candidates were all refused from one whose candidate space was empty. Beam rounds
      expanding an incumbent that was never dispatched propose only the moves that can bind a
      hardware dimension ([Tensorize], placement retypes); the rest are pruned before compile and
      counted in the same census. Separately, the baseline's compile is protected like any
      candidate's (gh-ocannl-533): a typed rejection — a large softmax/cross-entropy head whose
      unscheduled form exceeds the HIP scratch budget is the case in hand — declines the baseline
      ([baseline_declined] in the report, and its own key in the census rather than a
      [Not_dispatched] refusal, which would assert a reason that is not the one) and the search
      proceeds on the scheduled candidates, which fission and placement promotion routinely bring
      back within budget. Only the base lowering itself is indispensable: a failure before it is
      captured has no search to run and propagates.
    - {b Fissioned candidates}: the kernel-fission pipeline ({!Ir.Schedule.fission_scheduled}) with
      per-segment schedules — the same preset sweep per segment, and beam rounds that extend
      {e one segment at a time}. Per-segment schedules are cached keyed by the pre-schedule
      segment's canonical digest. [`Zeros] segments keep the default zero-expansion; [`Solo]
      segments stay unscheduled. One seed uses the config-default thresholds, reproducing the
      untuned default pipeline exactly — so the winner is never worse than not tuning, even on
      launch-overhead-bound workloads where every aggressive preset loses to it; its measured time
      is surfaced as the report's [default_ms] reference (gh-ocannl-552). Each preset is
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
    - {b Split-reduce seeds} (gh-ocannl-484 task 3): when a reduction-dominated site is detected
      ({!split_reduce_sites} — an rmw accumulation, or the gh-466 [Set_dynamic] scatter, whose
      target has little output parallelism while a long serial reduction loop feeds it: bias and
      weight gradients of convolutions, softmax denominators, skinny split-K GEMMs), the
      deterministic two-pass split reduction ({!Ir.Schedule.constructor-Split_reduce}) with a few
      [num_blocks] values as the tunable. The prelude applies {e whole-routine before fission}: the
      per-block partials edge it mints is exactly the materialized cross-nest edge kernel fission
      cuts at, so the two passes compile as separate kernels with the event chain supplying the
      grid-wide synchronization the combine needs, and each segment then gets the default preset
      (the block loop parallelizes pass 1). Sites are seeded as singles plus one composite
      recombining each site's best-timed [num_blocks]. A split winner persists as prelude +
      post-prelude per-segment schedules; note the numerics pin — the combine tree is a function of
      the schedule, so retuning may change low bits (see the schedule-cache docs).
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

type sr_site = {
  sr_axis : Ir.Indexing.symbol;  (** The reduction loop to split. *)
  sr_target : Ir.Tnode.t;  (** The accumulated node. *)
  sr_red : int;  (** The reduction loop's extent. *)
  sr_out : int;  (** The target's cell count — the site's whole output parallelism. *)
  sr_cost : int;
      (** Estimated segment cost: the accumulation statement's trip count (the product of every
          enclosing loop extent) — the serial work a split could recover. The ranking key
          (gh-ocannl-541). *)
  sr_dynamic : bool;  (** The gh-466 scatter form ([Set_dynamic]). *)
  sr_swaps : (Ir.Indexing.symbol * Ir.Indexing.symbol) list;
      (** The gh-ocannl-537 enabling interchange: [(outer, inner)] [Swap]s applied {e in order}
          before the [Split_reduce], each hoisting an accumulation-cell loop outside [sr_axis].
          Empty when the site is splittable as lowered. *)
}
(** A reduction-dominated accumulation site eligible for {!Ir.Schedule.constructor-Split_reduce}
    seeding (gh-ocannl-484 task 3); see the implementation's field docs. Exposed for tests. *)

val split_reduce_sites :
  ?static_indices:Ir.Indexing.static_symbol list -> Ir.Low_level.optimized -> sr_site list
(** The split-reduce seeding sites of the given lowering: rmw accumulations (and gh-466
    [Set_dynamic] scatters) whose target has at most a few thousand cells while a serial reduction
    loop of substantial extent feeds it — little output parallelism, lots of splittable reduction
    work. Per site the largest-extent enclosing serial loop that passes the hermetic
    {!Ir.Schedule.op_legality} probe of the corresponding [Split_reduce] is chosen (the op's own
    recognizer decides the pinning discipline), so every returned site is seedable as proposed.

    Every detected site is returned, ranked by descending estimated segment cost ([sr_cost], the
    accumulation's trip count) — gh-ocannl-541: the earlier [sr_red / sr_out] integer-division
    ratio zeroed every large-output site, and the in-detection cap then silently excluded (and,
    once gh-537 made more sites reachable, evicted) exactly the sites carrying the most serial
    work. The candidate-volume cap now lives in {!tune} ([max_split_reduce_sites] /
    config [autotune_split_reduce_max_sites]), which records evicted sites in the decline census.

    A candidate rejected {e only} because the accumulation cell's loops sit inside the reduction
    loop — how OCANNL lowers conv bias/weight gradients, where nothing else in the schedule space
    reaches the dominant segment — is re-probed after the enabling loop interchange
    ({!Ir.Schedule.split_reduce_hoist} names the loops, each [Swap] confirmed [Op_legal] on the code
    it acts on); the chain is recorded in [sr_swaps] and replayed by the candidate's prelude.
    [static_indices] only reaches the interchange probe's [Sched.apply]. Exposed for tests. *)

type decline_summary = {
  key : Ir.Schedule_outcome.rejection_key;
  count : int;
  sample_details : string list;
}
(** Aggregate of candidate declines sharing one stable key. Details retain at most the first three
    distinct diagnostics and are never part of the key. *)

type terminal_failure = {
  phase : Ir.Schedule_outcome.phase;
  candidate : string option;
  detail : string;
}

type report = {
  cache_hit : bool;
      (** The schedule came from the disk cache; no search ran. The census is then empty except for
          a declined baseline: the base compile precedes the lookup, so its rejection is real
          information about this process on this device even though nothing was searched. *)
  candidates_timed : int;
      (** Including the serial baseline where it was dispatched — on GPU backends it is not
          (gh-ocannl-532), and neither is any other candidate that binds no hardware dimension. So
          this count is not comparable across a CPU and a GPU backend: every serial-form candidate
          the CPU backends legitimately time is refused on GPU, and the refusals are counted in
          [declines] under [Not_dispatched_key] instead (gh-ocannl-543). *)
  candidates_failed : int;
      (** Candidates rejected by op preconditions, hardware limits, or backend compilation — the
          serial baseline included (gh-ocannl-533) — plus detected seed sites declined before
          proposal (split-reduce sites evicted by [max_split_reduce_sites], gh-ocannl-541) and
          candidates refused as unparallelized on a GPU backend (gh-ocannl-532), which the decline
          census records so a previously-proposed site — or a candidate space the backend's
          execution model empties — never stops being proposed silently. *)
  partial : bool;
      (** [true] when candidate work terminated on a fatal failure after the baseline was handled.
          A base compile that fails before the base lowering is captured, and a baseline timing
          failure, occur before reporting begins. *)
  baseline_declined : bool;
      (** The serial baseline's own compile was rejected with a typed cause and the search ran on
          the scheduled candidates alone (gh-ocannl-533): [baseline_ms] is then [infinity] and the
          rejection is in [declines] like any candidate's. The HIP scratch validator declining the
          unscheduled serial form of a large softmax/cross-entropy head at [Backend_link] is the
          case this exists for — before it was contained, that one rejection ended the search. *)
  declines : decline_summary list;
      (** Candidate rejections aggregated by stable cause key. Their counts sum to
          [candidates_failed]. Cache-entry replay failures are excluded. *)
  terminal_failure : terminal_failure option;
      (** The fatal failure that stopped a partial search; [None] on completed reports. *)
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
  split_reduce_candidates : int;
      (** Split-reduce seeds (gh-ocannl-484 task 3): one candidate per {!split_reduce_sites} site
          within the [max_split_reduce_sites] cap and eligible [num_blocks] value — the two-pass
          deterministic split reduction applied whole-routine before fission, each resulting
          segment getting the default preset. Deterministic given the computation and backend; [0]
          when no reduction-dominated site is detected. Sites the cap evicts appear in [declines]
          under [Seed_evicted_key "split_reduce"]. *)
  split_reduce_timed : int;
      (** Of the split-reduce candidates (the per-site singles and the recombined multi-site
          composite), those that compiled and were actually timed. *)
  mma_candidates : int;
      (** Candidates whose label promises a tensorized ([Schedule.Tensorize]) pipeline that the
          search put through candidate compile: whole-routine and per-fission-segment sketch seeds,
          the cross-segment recombination composite, and beam-expansion candidates. Counted at the
          same point as [mma_timed], so the two always describe the same population. *)
  mma_timed : int;
      (** Of [mma_candidates], those that compiled and were actually timed (dedup'd duplicates
          excluded — an identical candidate was already timed). [mma_candidates > 0] with
          [mma_timed = 0] means the search never measured a tensorized pipeline at all, the state
          gh-ocannl-521 recorded for every GPU backend: candidates are cheap to enumerate and were
          being rejected in bulk at candidate compile. *)
  model_scored : int;
      (** Sketch candidates the analytic cost model scored during the seed pre-filter
          (gh-ocannl-491); [0] when the pre-filter is off ([keep_fraction >= 1]) or nothing was
          scoreable (e.g. no envelope constants). *)
  model_pruned : int;
      (** Of [model_scored], the candidates dropped before compilation and timing. Candidates
          without model coverage are never counted here — they are always kept. *)
  fissioned : bool;
      (** The winning candidate compiles as multiple fissioned kernels; [false] when nothing was
          timed. *)
  baseline_ms : float;
      (** The unscheduled serial baseline's measured time, or [infinity] when it was not dispatched:
          on a GPU backend an unparallelized candidate is never run (gh-ocannl-532 — the whole
          routine in one work-item, unbounded in cost and uninterruptible), so it has no
          measurement and cannot win. Also [infinity] when [baseline_declined]. *)
  default_ms : float option;
      (** The untuned default pipeline's measured time (gh-ocannl-552): the [config_thresholds]
          fissioned-preset seed reproduces {!Ir.Schedule.maybe_default_schedules} exactly, so this
          is the schedule the user gets without tuning — the reference [baseline_ms] cannot provide
          on GPU backends, where it is [infinity]. Attributed by digest, so it is present even when
          the seed deduplicated against an identical earlier candidate (the timed serial baseline
          included, on CPU backends whose config thresholds leave the code unparallelized). On a
          completed search with [default_ms = Some d], [best_ms <= d] by construction — the seed is
          in the pool — and the margin between them is the value tuning added (the question
          gh-ocannl-491 asks). [None] when the seed was not proposed (a backend that is neither GPU
          nor CPU), failed to compile, was refused as unparallelized on GPU (gh-ocannl-532), or on
          a cache hit against an entry written before this field existed. *)
  best_ms : float;
      (** The winner's measured time, or [infinity] when nothing was timed at all — every candidate
          failed and the baseline was not dispatched (or was declined). In that case no cache entry
          is stored and the returned routine is the untuned default compile, not the serial
          baseline. *)
  best_schedule : Ir.Schedule_cache.saved_schedule;
      (** The winner's schedule; for a fissioned winner, the concatenation of the per-segment
          schedules (informational). Empty when nothing was timed. *)
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
  mc_rejected : int;
      (** Candidates excluded from the ranking because their scheduled form fails
          {!Ir.Low_level.validate_parallel}: the model cannot see that a schedule will not compile,
          and rates the tensorized families best, so on a backend where those are unbuildable it
          used to crown one and then degrade to the default (gh-ocannl-522). *)
}

val model_default_enabled : bool Lazy.t
(** Config [model_default_schedule]: recipe-level untuned compiles ({!Train.to_routine},
    {!Train.run_once}, the benchmark runners) route through {!model_default} instead of
    {!Context.compile}. *)

val compile_advisory :
  ?on_fallback:(exn -> unit) ->
  ?fallback_if:(unit -> bool) ->
  (Ir.Low_level.optimized -> Ir.Low_level.optimized list) ->
  Context.t ->
  Ir.Assignments.comp ->
  Ir.Indexing.unit_bindings ->
  Context.t * Context.routine
(** {!Context.compile_outcome} with advisory provenance and the given [lowered_transforms], falling
    back to a plain {!Context.compile} — the ordinary default pipeline — for a classified compiler
    rejection, including validation in backend codegen. Fatal failures propagate without retrying.
    [on_fallback] is called with the public rendering of the cause when fallback fires.
    [fallback_if] (default: always) is consulted first, for transforms that
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
    ({!Ir.Schedule.automatic_schedule_active}), or any classified scoring, application, backend
    validation, or compilation failure fall back to the ordinary default pipeline — the reported
    {!model_choice} then says ["default"]. Fatal failures propagate without retrying. Once the
    compile is on that pipeline there is nothing left to fall back to, so its failures propagate as
    they would from {!Context.compile}, without a duplicate attempt. Unlike {!tune}, nothing
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
  ?max_split_reduce_sites:int ->
  (* Candidate-volume cap on the split-reduce seed family: the top so-many {!split_reduce_sites}
     (ranked by estimated segment cost) are seeded; evicted sites are recorded in the decline
     census under [Seed_evicted_key "split_reduce"] (gh-ocannl-541). Default from config
     [autotune_split_reduce_max_sites] (8); [0] disables the family. *)
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
