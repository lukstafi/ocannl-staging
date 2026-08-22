# Scheduling and autotune

Schedule legality and coverage, sketch families, and how to read what a search actually did.

Part of the agent notes; the [index](../agent-notes.md) carries the scope discipline and the other
files.

- A GPU schedule must cover EVERY materialized-writing nest of the routine, not only the one the
  pipeline builds. Launch dimensions are kernel-global, so `Low_level.validate_parallel` rejects any
  companion write (a bias/relu tail; the elementwise statements an aligned-merged fission segment
  carries) not nested under loops covering every active `(kind, slot)` pair — and on GPU there is no
  all-serial fallback, so the whole candidate fails to compile. Do not relax the rule: an uncovered
  dimension means every hardware index executes that write. Annotate instead, from
  `Schedule.aligned_chains` — the default annotators' cross-nest analysis exposed as data (which
  loops may carry geometry, already trimmed so that chain position *k* denotes the same thread
  coordinate in every linked nest), which lets a pipeline supply its own per-position geometry while
  alignment stays `schedule.ml`'s rule. A pipeline that instead leaves companions bare and counts on
  `Fuse_epilogue` absorbing them has NO surviving form when the fusion declines; that cascade
  (gh-ocannl-521) had every GPU backend seeding tensorized candidates in bulk and timing none of
  them. Residual, shared with the shipped zeroing geometry: a tensorized nest's workgroup slot is the
  opaque `Tensorize` lane, so a per-lane companion reads cells other lanes of the same simdgroup
  produced — safe only because the threadgroup is exactly one simd width; a cross-nest simdgroup
  barrier is the formal fix. Query the analysis at the SITE'S arity (`aligned_chains ?max_chain`,
  default 2 = the presets' Grid+Workgroup shape): a batched matmul's chain is batch loops + row +
  column, and under the default cap a rank-3+ site can never match its full chain, so every seed for
  such a site declines on companion coverage — that single decline held gpt2_mini's five FFN-class
  kernels at a 1024-thread launch, 70% of the CUDA step at 1.3% of fp32 peak (gh-ocannl-569). A
  companion that reduces OVER the site's minor axis (the lm_head's max-logits row) trims the common
  prefix below the site's arity and correctly still declines — that one needs fission, not coverage:
  `fission_scheduled ~arity_cuts:true` (gh-ocannl-574) cuts it apart — measured on HIP/gfx1151 at
  1.30x on gpt2_mini (`report-gh612-hip.md`). **Size a fission's payoff over the WHOLE CHAIN, never
  over the fragment that keeps the site's name**: post-fission the mask, row-max and softmax work runs
  in separate downstream kernels, so dividing a standalone QKᵀ by a fused QKᵀ+mask+row-max reports a
  meaningless 5.2x. Like-for-like, summing every fragment on both sides: the lm_head/CE chain goes 4
  kernels / 8.136 ms → 5 / 0.357 ms (22.8x) and the four QKᵀ chains 8 kernels / 3.666 ms → 16 /
  2.038 ms (1.80x), so the QKᵀ sites are ~17% of what the two freed line items give against the
  lm_head's ~83%. Anchor the CE chain on `logits`, NOT on `wte`: the input token-embedding gather
  reads `wte` too (the embedding is `wte * onehot_x`) and is not part of the head. **The finer fission also COSTS +2.57 ms in the FFN bucket** -- it splits
  residual adds into more separately launched kernels, each re-deriving the running sum -- which the gh-573 fanin guard is what
  recovers, so the two must be measured together or each is mis-attributed. Why such a
  pair merges in the first place: the fission pass's no-parallelism-loss guard compares chains under the presets'
  `max_chain=2` cap, so trimming a rank-3 GEMM's minor axis reads as lossless; and a max-reduce is
  the shape that hits it because its `-inf` init is a `Set` nest, not a `Zero_out` — a sum-reduce's
  `Zero_out` already separates the statements. The arity_cuts mode analyzes uncapped AND requires
  merged nests to share one extent list (the init nest is conflict-free with the GEMM, so a pure
  no-loss rule would still merge it and companion coverage would still decline on it). It is a
  candidate-generation mode — the autotuner seeds fine-flagged per-segment sketches when the finer
  segmentation mints new digests, and a fine winner records `finer_fission` in its cache entry so
  replay re-segments identically — never the default pipeline, which would pay the extra launches
  unconditionally for parallelism its 2-loop presets cannot use. Since gh-ocannl-577 the
  coverage verdict is also a construction-time refutation in the matmul family tree
  (`matmul_coverage_witness`): this is sound because `companion_geometry`'s Ok/Error never depends
  on the geometry its `annotate` callback emits — only on the lowering, the site chain, the fused
  flavor's `skip` and the zeroing expansion. If you ever make the verdict consult the emitted
  geometry, the static witness goes stale and the tree will refute families whose candidates
  would build (or vice versa) — keep the invariant, or re-derive the witness. The fused
  (`Fuse_epilogue`) flavor is judged separately: skipping the epilogue tail can empty the
  coverage demand before the alignment analysis is consulted, so twins can survive a routine the
  unfused family is refuted on. Since gh-ocannl-613 the flavor is the family tree's ROOT level
  (`fusion = unfused | fused`, `matmul_family_tree`): the fused child is refuted with the
  recognizer's own reason (`Schedule.fuse_epilogue_witness`) on a site with no fusable tail —
  so a tree's `refutations` always carries one entry more than its pipelines produce, and a
  test asserting "every refutation is X" must scope to the `("fusion", "unfused")` path — and
  the unfused coverage verdict is shared with the fused branch because it implies it (the fused
  demand is a subset, and `aligned_chains` ignores `skip`): `aligned_chains` runs twice only
  where the unfused flavor is refuted. `sketch_seed_params` is the tree's `leaves`, twins
  included, and `model_default` has no separate twins step. **Consequence for diagnostics: since gh-577 a companion-coverage
  DECLINE CENSUS is empty, and that is not evidence the rule stopped firing** — a refuted family is
  never seeded, so it never reaches the decline log. gh-569's Part 3 read 25 coverage declines out of
  `schedule_log_declines`; the same workload on current master logs zero
  (`report-gh612-hip.md`). Ask the emitted source and the launch geometry instead.
- Batch loops of a GPU matmul sketch are no longer unconditionally `Serial` (gh-ocannl-643, the
  rank-4 q/k/v residue of gh-569: `36.7%` of the gpt2_mini step at 10% of sgemm peak because a
  `(batch, head, seq, head_dim)` site launched 4 blocks with batch and head serial inside). Two
  mechanisms, layered: (1) `Grid` slots >= 2 are legal and FOLD onto the hardware `.z` dimension —
  `Low_level.launch_dims` multiplies their per-slot maxima into `grid.(2)` and each folded loop
  binds `(z / stride) % cap` (`Low_level.grid_fold`; rendered in `C_syntax.hardware_binding`,
  degenerating to the bare `.z` register for a lone slot-2 loop, so pre-existing kernels emit
  byte-identical source); the 3-slot cap now applies to `Workgroup` only, and cc's serial fallback
  is untouched. (2) The GPU matmul sketch families seed each geometry in two batch flavors
  (`sk_batch_grid` twins, a "batch" tree level above "geometry"): batch positions `Retype`d to
  `Grid` vs. the historical `Serial` — TWINS, not a replacement, because the device block-count
  curve is non-monotone (gh-569's probe), so the tuner measures both. Three traps encoded in the
  implementation: the zero nest and every companion nest must carry the SAME per-position batch
  annotation with interior (`m_bi`) batch loops hoisted identically (`companion_role_ops`,
  `zero_geometry ~batch_grid`) or positional slot order diverges between nests and a thread zeroes
  cells another thread accumulates; `companion_geometry`'s `annotate` callback therefore takes the
  companion's whole chain (the hoist `Swap`s name the companion's own symbols) — its Ok/Error
  verdict still never depends on what `annotate` emits, so the gh-577 static witness stays sound;
  and batch products beyond 65535 are never seeded (`max_grid_fold_extent` — the CUDA/HIP
  `gridDim.z` cap; such a candidate could only fail at launch, GPU-only, after compiling).
  The pre-driver gate that backs that seeding filter (`Schedule.check_hardware_limits_classified`)
  covers BOTH 16-bit grid dimensions against the single `hardware_limits.max_grid_yz`: `grid.(2)`
  (the fold) and `grid.(1)` (the row-block count, which overflows on m-extent alone — no batch axis
  involved). One limit field, two typed resources (`Grid_y_extent` / `Grid_z_extent`), since the
  rejection key is what an autotune search groups declines by and the fixes differ.
  Pinned end-to-end by `test/operations/schedule_batch_grid.ml` (structure everywhere, execution
  and emitted-source fold on GPU backends).
  The HIP backend's `static_properties` dump lists the queried `max_grid_size` and
  `max_threads_dim` next to `max_threads_per_block`, so a run on hardware can read back what those
  gates compare against — without them the only evidence the query is not degenerate is that no
  kernel got rejected. On gfx1151/ROCm/WSL2 they read `(2147483647 65535 65535)` and
  `(1024 1024 1024)`, i.e. `max_grid_yz = 65535`.
- "`Tile_mma` is a barrier" is only half true, and the half that fails is the one barrier elision
  wants. Every rendering form ENDS the intrinsic block with a workgroup barrier, so a staging
  barrier that follows one is always redundant (`Schedule.elide_staged_barriers` drops it, and the
  after-loop one, at any pipeline depth). The LEADING bracket is form-dependent: the fragment-scope
  form (`render_mma_fragment_scope`, the crowned Metal/CUDA shape) emits it ONCE, on the scope
  wrapping the whole anchor loop, not per iteration — so a barrier may never be elided against a
  *following* `Tile_mma`. That is why a depth-1 staged k-block keeps exactly one explicit barrier
  (between its loads and its compute) while the pipelined form keeps none: the pipelined prefetch
  writes the *next* iteration's buffer copy, so the previous iteration's trailing bracket is what
  separates it from its reads. Elision is a synchronization-only transform, so the test that pins it
  is an executed BITWISE comparison against the same schedule with barriers re-inserted
  (`schedule_pipelined_matmul`) — adding barriers is always conservative, which makes that reference
  sound no matter what the elision does.
- The CUDA `cp.async` arm (gh-ocannl-487 phase 2, `C_syntax_config.async_copy`) keeps that
  discipline by never touching the intrinsic's brackets: an async copy is only complete for the
  issuing thread after a wait, and only visible to the workgroup after a barrier that FOLLOWS the
  wait — so the rotor loop's body is uniformly prefixed with `ocannl_cp_async_wait_all();
  __syncthreads();`, re-inserting for the async arm exactly the phase opener
  `elide_staged_barriers` drops for synchronous stores (those are published by the previous
  iteration's trailing bracket; an async copy waited AFTER a barrier is published to no one).
  Wait-all (PTX `cp.async.wait_all` = commit_group + wait_group 0) instead of commit/wait-group
  bookkeeping is what makes the emission per-`Set` opportunistic and safe: any staging statement
  the arm declines (precision conversion, surviving fringe ternary, non-global source, elements
  outside 4/8 bytes — sub-4-byte has no cp.async size, and 16-byte needs a destination alignment
  plain shared declarations don't guarantee) falls back to a plain store published by the same
  barrier,
  and correctness never depends on which statements were accepted. It is also why depth stays 2:
  deeper lookahead needs per-group waits. Eligibility is per tile in `compile_proc`
  (`current_async_tiles`; kernel logging disables it — a logged `Set` reads back what an
  in-flight copy cannot provide). Measured on the RTX 5070 Ti (paired in-process pd1/pd2, 9
  replicates, tf32 fragment-scope form): 512³ f32 pd2/pd1 = 0.97 median within a ~14% spread;
  deep-K 256×256×2048 = 0.92 with all 9 replicates in 0.906–0.946 against ≤4.4% arm spread — the
  overlap genuinely pays where the k_o loop dominates, reversing the portable form's Metal
  ~1.4–1.5× cost and HIP's null.
- A schedule can pass `Schedule.apply`'s validation and still be one the RENDERER cannot express:
  the pipelined-tile checks in `c_syntax.ml` (a read reached outside its rotor loop; a rotor loop no
  longer `Serial`) are positional facts about the final IR, which schedule application does not
  re-derive. Such a check must raise `Schedule_outcome.Cause_at (Backend_codegen, Unsupported …)`,
  not `invalid_arg` — an untyped exception at a compile-side phase is `Fatal` under
  `strict_failure_classification`, so one composed candidate ends the whole search (seen on Metal
  searches over `tile_acr_ma`). `raise_cause` re-renders the same `Invalid_argument` at the public
  `Context.compile` boundary, so typing one costs nothing for hand-written schedules. To probe a
  renderer precondition in a test, do surgery on the applied `optimized` (re-point `pt_rotor`)
  rather than retyping loops — a retyped anchor trips `Low_level.validate_parallel` first and never
  reaches the check under test.
- Autotune fault injection keyed by ATTEMPT INDEX (`Autotune.on_candidate_attempt`) is
  backend-dependent and silently vacuous: how many attempts precede an arm's first *timed* candidate
  varies. On Metal a small matmul's materialize-all arm has a baseline binding no hardware dimension
  (gh-532), and its whole `W_preset` block then dedups against that same digest — six attempts, none
  timed. Count timing runs with `Autotune.on_candidate_preflight` (one per timing run) and inject
  relative to that. Relatedly, hoisted (link-time packed) `Stage` candidates are a CPU family only —
  `matmul_seed_params` proposes `sk_hoist` from its `is_cpu` branch — so any test precondition about
  packed-constant pools is false on GPU backends and has to be stated as an equivalence.
- "Seeded" is not "timed". An autotune family can be enumerated in bulk and rejected in bulk at
  candidate compile, and a count of proposals then reads as coverage it does not have — assert on
  the *timed* counter (`report.mma_timed`, `fiss_sketch_timed`, `split_reduce_timed`), and follow it
  with an executed value check, since a candidate that compiles is not yet one that computes.
- The `N segs` in an autotune label such as `F_saved[fine 77 segs]` counts the SAVED PER-SEGMENT
  PLACEMENT ENTRIES, not kernels: the arm that reports `fine 77 segs` emitted 136 `__global__`s
  (gpt2_mini on HIP, `report-gh612-hip.md`), and `[58 segs]` emitted 117. Take kernel counts from
  the launch log (`schedule_log_launches`, whose `seg i/N` names the real fission width) or from the
  emitted source; a report that quotes the label as a kernel count is wrong by ~1.8x. The launch
  log's FIRST fissioned `seg 0/N` (skipping the `N=1` whole-routine probe) is arm A, the next is
  arm B — which is also how to pick the right file out of a content-polling snapshot of
  `<routine>__seg.hip`. The watcher can catch a partially written file, and the kernel count alone
  does NOT identify a usable capture: a torn file can already carry every `__global__` line while its
  last body is incomplete, and glob order is hash order. Require balanced braces AND a clean `hipcc`
  compile before accepting a snapshot (`benchmarks/gh612_cells.sh pick_armA`).
- A per-kernel profile's sum may only be validated against the step time of **the compile it came
  from**. Each search rep crowns a different artifact with different tile sizes, so holding one rep's
  profile against another rep's step p50 measures the search lottery, not the reconstruction — on
  gpt2_mini/HIP that turns a genuine 0.9% agreement into an apparent 2.3% disagreement, and the error
  is invisible because both numbers are real. Quote the paired p50 from the same cell's **pass-2**
  `replay2.out` and nothing else: `search.out`'s p50 is a pass-1 timing carrying the search process's
  own overhead, which `benchmarks/README.md`'s two-pass protocol excludes, and `snap`'s `replay.out`
  is a debug-file run rather than a clean timing pass.
- "Timed" is not "tensorized" either, and that failure is worse: a declined `Tile_mma` renders its
  scalar fallback, which compiles and runs, so the candidate is timed, ranked and possibly crowned
  under an `mma-*` label (gh-ocannl-545: 20 of 20 timed bf16 candidates on CUDA were scalar). The
  emission is the source of truth — grep the emitted kernel for the intrinsic
  (`wmma::`/`mma.sync`/`simdgroup_`), or read `C_syntax.mma_census`; `schedule_log_declines=true`
  names the rule that fired. When seeding and emission can disagree, fix the seeding side too, or the
  measurement budget keeps going to schedules that never tensorize: `mma_format_tiles` is keyed on
  the whole `(a, b, accumulator)` format triple, with per-entry arch floors, precisely so that a
  combination a backend supports at one accumulator width but not the other cannot be seeded.
  `mma_staged_layouts` (gh-ocannl-481) is keyed the same way for the same reason: the swizzled
  staged twin is seeded only where the emission can actually read that layout, which on CUDA is
  the uniform-bf16 combination and not fp8 (whose B side has no 16-bit `ldmatrix` form at the
  orientation the staged sketches mint). The census distinguishes `Mma_intrinsics_ldmatrix` from
  `Mma_intrinsics`, so "tensorized" and "fed at rate" are separable in a sweep.
- "Crowned" is not "shipped", and neither is reproducible on a small routine. `Train.tune_placements`
  runs two searches and keeps one artifact, so a family can win the arm that is then discarded whole
  — read `report.best_label` / `best_tensorized` / `mma_best_ms` per arm (the A/B calls `?report`
  for arm A first and ships the smaller `best_ms`), never the fact that some search crowned it.
  Since gh-ocannl-638, do not re-derive WHICH arm shipped from the reports' times either: config
  `tune_ship_arm=a|b` overrides the comparison, and `?on_ship` (`"A"` / `"B"` / `"flip"`) is the
  callback that says what actually shipped. That knob is what a measurement needs: the discarded
  arm is never executed against anything (`?report` carries timing metadata, `winner replay ok` is
  a dispatchability check, and a per-kernel harness times kernels on synthetic buffers without
  checking results), so profiling arm A while arm B ships leaves the profiled routine
  output-unverified — the limitation `benchmarks/report-gh612-hip.md` states in its verdict and
  `report-gh612-hip-verified.md` closes by forcing the arm. Forcing does not skip the other arm's
  search, so the A-vs-B numbers stay quotable; it does suppress the flip refinement.
  Below GEMM-dominated sizes the crown is a lottery: on `mlp_small`/metal five identical cold-cache
  searches crowned four different families in one arm with a 4.5% spread of best times, while the
  arm gap stayed at 57–95% (gh-ocannl-546, benchmarks/report-gh546-metal.md). Conclusions of the
  form "family X wins/never wins here" need repeats; the arm-level verdict does not.
  The arms are independent experiments and are contained as such since gh-ocannl-550: an arm whose
  search raises is a LOSING arm (ranked `infinity`), the other arm's winner ships and stays cached,
  and the failed arm's partial report still arrives in position carrying `terminal_failure` — read
  that (or `partial`) before `best_ms`, because a partial arm's best is a time whose routine was
  never compiled. Before that fix, one arm's late failure destroyed the other arm's finished work
  in-process (the cache entry survived, since `SC.store` precedes the winner replay — so a warm
  cache could still replay it; five of five tf32 `gpt2_mini` runs lost arm A this way). Note where
  it escaped: NOT at the failing candidate — candidate-grade protection absorbed those OOMs as
  `Backend_link` declines — but after the search concluded, when the exhausted device defeated both
  the winner replay and the untuned-default fallback compile behind it. Containment tests do not
  need a device that can fail: `Autotune.on_candidate_attempt` injects one
  (`test/operations/autotune_arm_containment`).
- A timing failure's *phase* decides whether the lineage is condemned, so pre-dispatch validation
  needs its own. `Context.run` validates (poisoned lineage, uninitialized inputs, unsatisfied
  execution dependencies, out-of-range static bindings) before dispatching; inside a `Launch`-tagged
  boundary those failures were unattributable — `classify_failure` returns `None` on every C
  backend — so `classify_raw` made them `Fatal` and the handler poisoned the lineage. A one-line
  user mistake (a `timing_ctx` scratch context missing one of the caller's initializations, which
  its own docs warn about) thus condemned the search *and* the context, with no restore
  (gh-ocannl-564; #536 for why there is no restore). `Schedule_outcome.Preflight` now tags that
  region and classifies as a contained `No_device_writes` decline **without consulting the
  backend** — host-side validation, so a classifier guessing `Writes_may_have_occurred` would
  escalate a failure that provably wrote nothing. Rule for any new boundary around `Context.run`:
  tag what precedes `Ir.Task.run` as `Preflight`, or a fixable mistake reads as device damage —
  **but only the per-candidate half of it may be contained** (gh-ocannl-569, found on HIP). The
  split is `Context.check_lineage_runnable` (poisoned lineage, uninitialized inputs, unexecuted
  dependencies) against `Context.check_launch_bindings` (out-of-range static bindings), and it is
  the difference between a condition that belongs to the *lineage* and one that belongs to *this
  candidate's* bindings. Only the second can fail one candidate while its siblings time cleanly;
  the first fails every candidate of every arm identically, so containing it is silent —
  a search whose serial baseline is not dispatched (**every GPU search**, gh-ocannl-532) then
  declines every candidate for the one reason, times nothing, and `tune_placements` *returns
  normally* shipping the untuned default out of an unusable lineage, with no exception and no
  `terminal_failure`. On the C backends the dispatched serial baseline hits the condition first and
  takes the arm down with the caller's message, which is why every golden encodes the CPU shape and
  CI never saw it. So: **contain the per-candidate half, raise the lineage-wide half outside the
  boundary** — at the candidate site before `Outcome.protect`, and inside the baseline's match
  scrutinee so `condemn` still reads it as pre-dispatch and leaves the lineage usable. Tag the
  hoisted raise `Preflight` anyway: it carries no boundary there, but `search`'s fallback handler
  would otherwise report a validation error under its `Transform` default.
  These causes also resist injection: they belong to the lineage and the bindings, not to a
  candidate, so a genuine one fails *every* candidate at once, which is why
  `Autotune.on_candidate_preflight` exists — though since the lineage-wide half now escapes the
  region, injecting one of *its* exceptions through that hook exercises the containment machinery
  with a realistic payload rather than mirroring where a real one is raised.
- Placement decides which tensorized candidates *exist*, not just how they rank, because
  `mma_tile_for_precisions` keys on the storage precisions of the nodes the site actually reads.
  Under the mixed-precision recipe on a uniform-format backend (Metal's `simdgroup_matrix`: no mixed
  multiply-accumulate) the default-placement arm seeds **zero** mma candidates — the reduced-precision
  cast twins are virtual, so the site reads f32 masters into an f16 destination and no advertised
  tile matches. `Mixed_prec.Twin_materialized` (three small weight casts) restores the whole family
  at the default arm's cost, whereas materialize-all buys it by doubling the kernel count. If a
  reduced-precision cell reports `mma_candidates = 0`, look at the twins before the seeding rules.
- Supplying a `?lowered_transform` bypasses the default annotator entirely (`backends.ml` `compile`
  only calls `Schedule.maybe_default_schedules` in the `None, None` arm), so **any** code that goes
  through that seam is the unscheduled serial form unless it schedules itself. The autotuner's base
  compile is exactly that, which is why its baseline candidate binds no hardware dimension on GPU:
  the whole routine in one work-item. Such a dispatch is unbounded in cost and uninterruptible on a
  device shared with the display — measured 6.9 s/run on Metal for LeNet (winner 35.7 ms) and hours
  on gfx1151, with driver timeouts and a lost display (gh-ocannl-532). `Autotune.tune` therefore
  does not dispatch an unparallelized candidate on a GPU backend at all: `dispatchable` gates both
  the baseline and every candidate, `baseline_ms` is `infinity` there, and if nothing at all gets
  timed the search stores no cache entry and returns the untuned default compile rather than the
  serial incumbent. On CPU backends the serial form runs at full single-core speed and is still
  timed. When timing anything else through this seam, price the serial form before dispatching it.
- The base compile staying unscheduled is a settled decision, not an oversight (gh-ocannl-552): the
  default pipeline is `maybe_default_schedules` — fission then per-segment annotation, so several
  kernels in general, not one `optimized` to rebase candidates on; every candidate family assumes
  the serial zero point; and annotation reads `hardware_limits`, which would bake per-device
  decisions into the cache's `source_digest`. The "did tuning beat the shipped default?" reference
  is `report.default_ms` instead — the config-thresholds fissioned seed reproduces the untuned
  pipeline exactly and its time is attributed by digest (so a seed that dedups against a timed twin,
  the CPU serial baseline included, still reports).
