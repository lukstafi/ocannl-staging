## [Unreleased]

### Added

- **Config profiles: `reproducible` and `performance`, with picker-inherited precedence**
  (gh-ocannl-559). A new `profile` setting picks a goal-oriented preset bundle through the ordinary
  sources (`--ocannl_profile=…`, `OCANNL_PROFILE=…`, `profile=…` in `ocannl_config`).
  `reproducible` is deterministic and, wherever reasonable, identical across machines — the
  autotuner's *search* off (replaying a committed cache stays allowed: a pinned schedule is
  deterministic), the gh-555 inlining refinement off (a flip is accepted on a measured
  improvement, and inlining is not numerics-neutral where storage is narrower than compute), no
  `-mcpu=native`, no probed SIMD flags, FP contraction pinned off, explicit SIMD rendering off (it
  reassociates strict-FP reductions), and the numerics gates at their exact defaults.
  `performance` is the fastest configuration *at unchanged semantics*: search on with a wider beam,
  the cost model picking untuned compiles, host-targeted arch flags, `fp16_arithmetic`.
  Result-changing gates like `tf32_matmuls` stay on the orthogonal numerics axis, so an A/B across
  profiles keeps measuring scheduling rather than different math.

  Each source level splits into two sublevels — its explicit keys, then the payload of a profile
  *picked at that level* — so a specific setting can never be defeated by an aggregate one of equal
  immediacy (`--ocannl_profile=reproducible --ocannl_autotune_rounds=3` gets both), while a profile
  named on the commandline does override an exhaustive config file. The payloads are partial
  `ocannl_config` files embedded as string constants (no share-directory machinery, nothing for the
  walk-up config search to shadow), quoted verbatim in `ocannl_config.reference`, and each
  payload-derived value is reported with its provenance by `log_config_sourcing`. Companion
  convention: the reference file now ships with every setting commented out, so copying it wholesale
  no longer manufactures a hundred explicit settings the user never chose. New keys:
  `profile`, `autotune_search`, `cc_backend_fp_contract`, and the `none` spelling of
  `cc_backend_arch_flags`.

- **A search report that says which candidate won, and the placement A/B saying when a tensorized
  winner is discarded** (gh-ocannl-546). `Autotune.report` gains `best_label`, `best_tensorized`
  (read off the winner's schedule, not off its label's promise), the winner's `Tile_mma` statement
  and scalar-fallback counts, and `mma_best_ms` — the best *timed* tensorized candidate, whose
  margin against `best_ms` separates "tensorization is uncompetitive here" from "it lost inside the
  noise". `Train.tune_placements` states the cross-arm conclusion, and a tuned benchmark cell emits
  both arms in its result line, so a per-arm win that reaches no artifact survives in
  `results.jsonl` instead of only in a discarded stderr stream. The measurement leg behind it
  ([benchmarks/report-gh546-metal.md](benchmarks/report-gh546-metal.md)) found the Metal placement
  A/B sound — the arms are 57–95% apart, 4–13x the per-arm spread — and relocated the actual gap:
  under the mixed-precision recipe the default-placement arm seeds *zero* tensorized candidates,
  because the virtual cast twins leave the matmul site reading f32 masters into an f16 destination,
  a triple no uniform-format backend advertises. Materializing just the twins
  (`Mixed_prec.Twin_materialized`, exposed as `BENCH_TWIN_PLACEMENT`) restores the whole tensorized
  candidate family at that arm's cost.

- **`gpt2_mini_train`: a transformer training-step cell, and gate-cost legs that cannot go
  missing** (gh-ocannl-551). `bench_gpt` now dispatches its step shape on the fixture's `mode`
  like the Python runners do, and the new `gpt2_mini_train` workload trains the same
  architecture with plain SGD in all three frameworks (every weight a parameter, `wte` tied to
  the lm_head, the positional table added by an einsum since an OCANNL parameter has no batch
  axes). That closes the gh-492 task-5 gap: the four precision variants `f32 / bf16 /
  f16-static / f16-gatedN` were unsatisfiable for the matmul-dominated workload, because
  `BENCH_STATIC_SCALE` / `BENCH_GATE_INTERVAL` existed in `bench_mlp` alone *and* a forward-only
  workload has no loss scale to gate. The flag parsing and the training-step shapes moved into
  `Bench_harness` (shared by every training runner, so the next flag added cannot apply to a
  subset), the legs became orchestrated cells (`--precision f16-static f16-gated16`), and a
  requested cell a workload cannot express is now reported as `NOT APPLICABLE` with its reason —
  in the run log and in a report section — instead of being silently absent, which is
  indistinguishable from an unrun cell. The workload's f16 cells rest on gh-ocannl-548 and
  gh-ocannl-547 below, which landed alongside it: before the `-inf` mask fill and the guard's
  narrowed scope, no gpt cell could compile at f16.
- **The tuner's honest reference point** (gh-ocannl-552, the shared cause behind gh-ocannl-532
  and gh-ocannl-533): `Autotune.tune` reports the untuned default pipeline's measured time as
  `report.default_ms`, next to `baseline_ms` (which is `infinity` on GPU backends, where the
  serial baseline is never dispatched) — the in-search answer to "did tuning beat the schedule
  the user gets without tuning?", the question gh-ocannl-491 could previously only ask in the
  benchmark harness. The measurement is the existing config-thresholds fissioned seed's,
  attributed by digest so a seed that dedups against a timed twin still reports; it persists on
  schedule-cache entries as an optional field, so pre-existing entries stay readable. The
  attribution honors the scheduling gates (with automatic scheduling inactive the untuned default
  is the serial form, reported via the baseline; with `schedule_fission=false` no candidate
  reproduces the default and the field is absent), and cached values are validated against a
  config fingerprint of the gates and preset thresholds, so a config change that redefines the
  default drops the stale diagnostic instead of reporting it. The issue's
  other half is settled in place: the base compile stays the unscheduled serial form (the default
  pipeline is several kernels in general, every candidate family assumes the serial zero point,
  and annotation would bake per-device decisions into the cache digest), documented at the
  capture site.

### Changed

- **Native fp16 arithmetic on the CPU backends** (gh-ocannl-516): fp16 is the one 16-bit format a
  CPU can execute natively, and where it does, computing in it doubles the lane count against f32.
  `cc` probes the configured compiler once per process for `_Float16` and for whether its arithmetic
  is genuinely 16-bit (ARMv8.2-FP16, AVX512-FP16) rather than promoted to float, and reports the
  latter as `hardware_limits.native_fp16_arithmetic`. `Ir.Numerics.fp16_arithmetic` (config
  `fp16_arithmetic`, default **false** — unlike gh-ocannl-517's widening, this trades mantissa for
  throughput) then makes fp16 compute in fp16: `Vectorized` loops mint a `HALF_T` vector and run at
  twice f32's lanes with no conversions at all. The fused multiply-add of both the scalar and the
  vector rendering goes through one shared builtin macro, because `fmaf` on `_Float16` rounds twice
  where the elementwise builtin rounds once — the two paths must not disagree. `Tile_mma` register
  tiling still declines fp16; its tile geometry and packed-`Stage` seeds assume f32 lane counts.

  Measured on an M-series at n = 2^22: the compute-bound control goes from 0.76x to **1.99x** of
  f32, and the streaming kernels from ~1.5x to ~2x.

- **`cc_backend_arch_flags` defaults to `auto`** (found while implementing gh-ocannl-516): the
  previous default `-march=native` is accepted by Apple clang on arm64 and *downgrades* the target
  — 22 `__ARM_FEATURE_*` macros against 26 with no flag and 33 with `-mcpu=native`, losing
  `__ARM_FEATURE_FP16_VECTOR_ARITHMETIC` among them, so a machine with native 16-bit arithmetic
  looked like one without. `auto` asks the target which architecture family it is in and probes that
  family's spelling (`-mcpu=native` on ARM, `-march=native` on x86, where `-mcpu=` is only an alias
  for `-mtune=` and would not select the ISA). Explicit values are passed through verbatim.

- **16-bit storage, f32 compute on the CPU backends** (gh-ocannl-517): a tensor node's precision is
  now its *storage* precision only; the precision its arithmetic runs at is a separate decision at
  the codegen seam (`C_syntax_config.compute_prec`). The GPU backends are unchanged — they have
  native 16-bit types and arithmetic — but `cc`, which had no 16-bit arithmetic and wrapped every
  narrow-float operator in a widen/op/narrow round-trip, now computes narrow floats in f32: a load
  widens once, a store narrows once, and an assignment's intermediates keep f32 mantissa instead of
  being rounded per operator. This is both strictly more accurate and the precondition for the
  `Vectorized` renderings, previously gated to f32/f64, to accept 16-bit nodes at all: narrow loads
  widen into f32 vector registers and stores narrow on the way out, whole vectors at a time
  (shift-based for bf16, `__builtin_convertvector` for fp16 on `_Float16` targets, a per-lane
  fallback otherwise — every arm bitwise-identical to the scalar path, so a vectorized loop still
  matches its serial remainder exactly). Governed by `Ir.Numerics.narrow_compute_f32` (config
  `narrow_compute_f32`, default true); setting it false restores per-operator rounding. Two things
  deliberately stay at storage precision: the RNG lane conversions, whose generator is selected by
  the precision they render at, and operator-free assignments, so a copy loop stays a copy. New
  `bin/narrow_storage_bench.exe` and `test/operations/narrow_storage_compute.ml`.

  The measured verdict is the reverse of the issue's expectation: on an M-series, a bandwidth-bound
  elementwise add reaches 1.97x at fp16 storage but 0.91x at bf16, whose round-to-nearest-even
  narrowing costs more than the halved traffic saves.

- **Operation results close down; `stretch` requests use-site resolution** (gh-ocannl-544): an
  open row of an operation's result no longer widens to what a use site demands — it closes to
  the arguments' shapes and the use site broadcasts it in. Use-site resolution (closing to the
  GLB of the use sites) remains the rule for leaf tensors, parameters, and — via the new
  resolve-at-use row marking — parameter-initialization expressions, whose intermediates must
  fill the parameter's shape rather than broadcast (repeating random values). The new identity
  operation `stretch` requests use-site resolution by name; the `0.5 + 0.5` shape-inferred
  constant idiom is replaced by `stretch 1.0`, and pooling kernels use `stretch 0.0`. The
  gh-ocannl-540 cast-twin pin in `Mixed_prec.cast_param` is reverted: the default now guarantees
  a twin keeps its master's shape (guarded by `test/operations/mixed_prec_twin_shape`). The same
  policy applies to dimension variables: an unmarked, unconstrained operation-result axis is
  guessed minimal instead of widening to its use sites (a learning-rate expression meeting a
  single parameter shape stays scalar); `At_least_dim` axes keep meeting their use sites, since
  direct indexing is a dim-carrying use.

## [0.9] -- 2026-08-03

> Release note: theme — program search and optimization. Since 0.8, the schedule system has
> gained hardware tensor-core paths, native affine legality and cost analyses, symbolic runtime
> extents, convolution sketch families, a liveness-based memory planner, deterministic split
> reductions, and a mixed-precision training recipe. Multi-round search over schedule
> compositions is implemented; cost-model-guided default and beam selection are in (config-gated,
> advisory). End-to-end benchmark validation is complete for this milestone: the cross-machine
> sweep (gh-ocannl-476, re-measured under gh-ocannl-538) was run from a wiped autotune cache on
> Metal, CUDA and HIP, and the checked-in reports under `benchmarks/` are the record. The
> practical theme of the second half of the milestone was making the search *survivable*: a
> candidate the toolchain or the device refuses is now a typed, censused decline rather than a
> fatal, a hung dispatch, or a silent omission.
>
> Reduced-precision status: bf16 and f16 training are correct and measured on every backend, but
> generally slower than f32 (gh-ocannl-535). The sweep located the f16 cost — the loss-scaling
> gate's `grad_checksum`, which lowers to one serial reduction that only split-reduce
> parallelizes; with it scheduled, HIP `mlp_wide` tuned f16-static is the first reduced-precision
> cell in the suite to beat tuned f32. Tensor cores are reached and crowned on CUDA (tf32) and HIP
> (bf16); on Metal a tensorized candidate wins an arm but the placement A/B does not ship it
> (gh-ocannl-546, v1.0). CPU reduced-precision arithmetic (gh-ocannl-516, gh-ocannl-517) is v1.0
> scope.

### Added

- **Typed candidate-failure containment** (gh-ocannl-536): a failing autotune candidate now
  carries a classified cause instead of three call sites guessing. `Schedule_outcome` causes are
  raised at the seams that actually refuse work — Metal's post-link pipeline check
  (`Resource_exceeded Workgroup_threads`, plus the compiled kernel's own static threadgroup
  allocation), CUDA's driver and nvrtc statuses (split on the damage axis: launch/link/PTX
  rejections that leave the context usable are contained declines; the asynchronous memory-fault
  family is counted and then escalated, since the context is left sticky), and sketch
  applicability (no matmul/conv site detected, an uncoverable companion nest) which is a decline
  like a `Schedule.apply` precondition, not a fatal. `Autotune` consults the backend's
  classifier at launch and sync too (via `Context.failure_classifier`), and the reported cause
  names the phase it was raised at. Damage is contained rather than assumed away: a rejection
  the backend proved wrote nothing rolls back the routine's executed marking, while one that may
  have written buffers — or any unattributed fatal — poisons the execution lineage, which then
  refuses further use naming the routine and the original failure. Toolchain and environment
  faults stay deliberately unclassified (`NVRTC_ERROR_BUILTIN_OPERATION_FAILURE` and friends fail
  every candidate identically, and absorbing them would turn a broken installation into a silent
  "nothing worked" report). Backend selection was narrowed to match: device discovery signals
  `Backend_intf.Backend_unavailable` only when the library is not linked in or the driver reports
  no devices, and `Context.auto` advances on that alone — a driver that is present but fails to
  initialize propagates instead of being relabelled "unknown backend".
- **HIP scratch-budget pre-validation** (gh-ocannl-533): a kernel whose private (scratch) segment
  exceeds what the device can back does not fail cleanly on ROCm — it aborts the HSA queue, and
  what reaches OCaml is an undifferentiated `hipErrorInvalidValue` on a stream that is already
  dead. So containment comes from prediction: the linked kernel's private segment size is checked
  at `Backend_link` against an occupancy-derived budget (calibrated on gfx1151/ROCm 7.14; the
  4 GiB total-scratch cap it encodes is not exposed by any HIP or HSA query) and an over-budget
  kernel is declined as `Resource_exceeded Thread_scratch` before it is ever launched. To a tuner
  candidate that is an ordinary censused decline; to a hand-written schedule it is the usual
  `User_error`. New config key `hip_scratch_validation` (default true) disables it on a device
  where the model is wrong, and a device reporting no usable occupancy figures is never rejected.
  Device properties are memoized per ordinal rather than re-queried per link.
- **A decline census that accounts for every refusal**: two new typed causes give the search a
  vocabulary for candidates that were never measured. `Not_dispatched` (gh-ocannl-532,
  gh-ocannl-543) records a candidate refused because it binds no hardware dimension, with an
  `origin` distinguishing the serial baseline, a candidate that degenerated to one, and a beam
  move pruned before compile. `Seed_evicted` (gh-ocannl-541) records a detected seed site that
  lost only to a candidate-volume cap — previously such a site was not proposed, not declined,
  and absent from the report entirely. `BENCH_TUNE_REPORT=1` renders both, and the census is now
  created before the schedule-cache lookup so a cache hit reports the declines it saw.
- **Split-reduce site ranking and capping** (gh-ocannl-541): `Autotune.split_reduce_sites` returns
  every detected site ranked by `sr_cost` — the accumulation statement's trip count, i.e. the
  estimated cost of the segment the split would parallelize — and `tune` applies the
  candidate-volume cap (new config key `autotune_split_reduce_max_sites`, default raised 4 → 8),
  recording each evicted site in the census. The previous `sr_red / sr_out` integer-division
  ratio sent every wide-output site to 0, which after gh-ocannl-537 grew lenet's site count past
  the cap meant excluding exactly the conv weight gradients with the most serial work to recover.
- **Reserved identifiers derived from each backend's own syntax** (gh-ocannl-553): tensor-node
  debug names become identifiers verbatim in the emitted kernel, so `op_syntax_idents` now renders
  every (precision, operator) pair through the backend's *own* syntax functions and harvests the
  names, instead of deriving the blacklist from how C spells each operator. MSL spells
  `Tanh_approx` as `tanh` — which `Tensor.unop`'s `~op_label` makes the label of every node
  `Operation.tanh` mints, so a GPT-2 gelu produces one — and the Metal kernel declared
  `device float *__restrict tanh` next to a call that then resolved to the pointer. Each backend's
  builtins-table keys are reserved as well (a node taking a builtin's name both shadows the
  definition and drags it into kernels that never call it, since builtins are selected by
  searching the rendered kernel for their key). Plain-C names stay a floor, so the change is
  purely additive and no existing node name moves.
- **`?mask_fill` on the attention and transformer entry points**: every `~mask`-taking entry point
  in `Nn_blocks` can now select a finite fill value, for padding masks that can cover a whole
  query row (where the `-inf` default would make the softmax's max-subtraction produce NaN).
- **Benchmark matrix: tuning and precision are independent cell axes** (gh-ocannl-539). An OCANNL
  cell was identified by one `variant` string conflating the scheduling choice
  (default/materialized/tuned) with the storage precision, so a tuned reduced-precision cell could
  not be expressed at all — the one cell where tensor cores can show on a backend whose only mma
  route is a reduced input format (RDNA3/3.5 WMMA has no f32-input shape). Cells are now the
  product of the two, `SKIP_CELLS` keys gained a precision component with `None` as a wildcard,
  results are stamped with the identity that was dispatched rather than the runner's self-report,
  and the report renders precision as its own column, ordered precision-major. `bench_mlp`
  composes `BENCH_TUNE` with `BENCH_PRECISION` (gh-ocannl-529) and reports mma seeded/timed counts
  under `BENCH_TUNE_REPORT`; `BENCH_CELL_LOG_DIR` opts into keeping each cell's raw output, where
  the per-candidate search lines live.
- **Cross-machine sweep reports** (gh-ocannl-476, gh-ocannl-538): full re-measurements at one
  commit with a wiped autotune cache on Metal, CUDA and HIP, plus paired same-session A/B legs for
  gh-ocannl-537 on CUDA and Metal, all under a reporting contract that qualifies every segment
  share by placement, never presents per-segment sums as step times, and reports tensor-core
  reachability as seeded/timed counts rather than a yes/no. Headlines: a rocWMMA candidate is
  seeded, timed and crowned for the first time (HIP `mlp_wide` bf16, −12.8% against the best
  scalar candidate in the same search); CUDA reaches `mma.sync…m16n16k8.f32.tf32.tf32.f32` under
  `tf32_matmuls=true`, so the policy is no longer a no-op; split reduction is worth 46–82% on the
  default-placement arm and −17.7% on CUDA lenet's shipping artifact.
- **`Swap` ∘ `Split_reduce` seeding: conv-gradient accumulations become reachable**
  (gh-ocannl-537). The gh-484 split-reduce family timed cleanly but was inert on the reduction it
  was filed for: OCANNL lowers parameter gradients with the accumulated channel loop *innermost*
  and the reduction loops outside it, so `Split_reduce`'s pinning discipline (every
  accumulation-cell symbol bound by a loop *enclosing* the reduction loop) rejected every axis of
  every parameter gradient — while that one fission segment was 89% of HIP lenet's step after
  gh-527, and 69–95% across three backends. `Schedule.split_reduce_hoist` now reports the
  offending symbols structurally from the recognizer's own hermetic probe, and
  `Autotune.split_reduce_sites` hoists them outside the reduction loop with a chain of adjacent
  `Swap`s (relative order preserved, each `Swap` confirmed `Op_legal` against the code it acts on)
  before re-probing the split; the chain rides on the site (`sr_swaps`) and is replayed by the
  `F_split` candidate's prelude. `BENCH_SR_SITES=1` names the interchange a site was reached
  through, and tags a rejection `[hoistable: …]` when the interchange would remove it.
- **Forward-only reduced precision by load-time conversion** (gh-ocannl-492, the `gpt2_mini`
  leg): data-backed tensors are precision-`Specified` at creation and inference has no
  optimizer, so there is no master copy for a cast twin to preserve — re-precision happens at
  ingestion instead (torch's `model.half()`): `Ir.Ndarray.convert` plus `TDSL.wrap ~prec`
  convert fixture data to the target precision, `bench_gpt` gains `BENCH_PRECISION=bf16|f16`
  (attention params via the storage policy, injected values converted by `set_values`; layer
  norms and the softmax-CE head pinned f32), and `orchestrate.py --precision` covers the gpt
  workloads. Cast twins remain the training-side construct.
- **Fused on-device loss-scaling gate** (gh-ocannl-492 task 5): the f16 recipe's per-step
  host-read inf/nan gate (a full device await plus a routine split, on top of the checksum
  reduction) is now optional. `Train.sgd_update ?update_gate` gates every optimizer-state
  mutation by `Where` selection (skipped steps leave parameters and momentum buffers untouched
  exactly — selection, not multiplication, since `0 * inf = nan`);
  `Mixed_prec.gated_scaled_update` builds the whole dynamically-scaled step as one routine with
  the gate computed on device (a fast-math-robust range test of the checksum), and
  `Mixed_prec.gated_step` samples a sticky window checksum every `check_interval` steps to
  drive backoff/growth — overflowing steps inside a window skip themselves on device, so
  delayed sampling delays only scale adjustment, never poisons state. Bench legs
  `BENCH_STATIC_SCALE=1` (fixed scale, no gate — the discriminating experiment for the gate's
  share of f16's step cost, gh-ocannl-535) and `BENCH_GATE_INTERVAL=N` in `bench_mlp`. The
  checksum reduction itself is the shape the gh-484 split-reduce seeding parallelizes under
  tuning.
- **Virtual packed uniform via lane extraction** (gh-ocannl-509 task 4): packed `uniform`
  results can now be virtual (inlined). A read cell inlines as
  `vec_convert(counter[flat / lanes]).v[flat mod lanes]` through the new IR-internal
  `Uint4x32_to_prec_uniform_lane` binop, whose per-backend builtins index the same converted
  block as the vectorized stores — virtual and materialized runs agree bitwise on every backend
  (pinned by `test_uniform_virtual_lane` across odd/divisible sizes, multi-axis shapes, and
  single/half/double/fp8). Trade-off: the counter tensor of a virtualized uniform is read at a
  runtime block index, so it stays materialized (typically as routine-local scratch, 16 bytes
  per 128-bit block) — dropout-mask-style virtual uses keep working after `uniform1` retires,
  with the conversion recomputed per cell but the threefry chain evaluated once per block.
- **Numerics policy with tf32 matmuls** (gh-ocannl-478): `Ir.Numerics` is a record of
  compute-precision decisions — user-chosen, never optimizer-chosen, identical across sibling
  autotune candidates. Its first field, the opt-in `tf32_matmuls` config key (default false,
  PyTorch-style), routes CUDA uniform-f32 GEMMs through wmma `precision::tf32` (m16n16k8,
  sm_80+) instead of the scalar fallback; off keeps full f32 numerics. Compute precisions like
  tf32 have no byte layout and never appear as a tensor node's storage precision.
- **Precision-assignment policy over a model** (gh-ocannl-492 task 1):
  `Ocannl.Precision_policy.apply` assigns storage precisions across a tensor graph by structural
  class (params / activations / gradients) instead of hand-annotating every tensor.
  User-specified precisions win; integer and uint4x32 (RNG) chains are protected; the `except`
  predicate pins matched nodes at the session default (a skip would be undone by top-down
  precision inference). Verified by an executed f32-vs-bf16 parity test, not only settled-
  precision pins.
- **Mixed-precision training recipe** (gh-ocannl-492 tasks 2-4): `Ocannl.Mixed_prec` — the
  torch-AMP recipe over OCANNL structures. Master weights: `with_master_weights` installs a
  `Tensor.param_postprocess` hook giving every parameter a reduced-precision cast twin (the new
  identity `Operation.cast`, whose gradient accumulates back through the widening assignment);
  the f32 master stays the optimizer/persistence target, pinned `Specified` because the cast
  registers a top-down `Inferred` of the twin's precision into the param. Dynamic loss scaling
  (f16): `Train.grad_update ~loss_scale` seeds backprop with the scale, `Train.sgd_update
  ~grad_unscale` unscales gradients in place before the optimizer math, and
  `Train.grad_checksum` reduces every parameter gradient to one host-readable scalar (non-finite
  iff any gradient cell is) — `Mixed_prec.scaled_step` gates the optimizer routine on it and
  `Loss_scaler` runs the backoff/growth schedule by overwriting tiny device-resident scale
  tensors (no recompilation). Executed-parity coverage in `test/training/mixed_prec_parity.ml`
  (bf16 and f16 trajectories vs the f32 oracle, plus a deterministically engineered f16 overflow
  pinning the backoff/growth cycle); benchmark legs via `BENCH_PRECISION=bf16|f16` in the mlp
  runner and `orchestrate.py --precision`.
- **Tensor-core input-format vocabulary** (gh-ocannl-481 groundwork):
  `Backend_intf.mma_input_format` (f32, tf32, f16, bf16, fp8-e5m2) with per-(a, b)-operand-pair
  intrinsic tile shapes in `mma_capability.mma_format_tiles`; a future e4m3 (and mixed
  e5m2×e4m3) needs only a constructor and descriptor entries.
- **Pad-to-tile scheduling, PADTO** (gh-ocannl-485): `Schedule.Pad { axis; to_multiple_of }`
  extends a Serial loop to the next tile multiple with `If (axis < N)` guards on the effectful
  leaf statements (so barriers stay uniform and downstream `Split`s divide cleanly). `Stage`'s
  cooperative/packing edge guards became identity-filling (`Where`-form, storing 0 — the
  add-reduce identity — to out-of-range tile slots; tiles are tracked in
  `Low_level.optimized.zero_fringe`), and `Tensorize` recognizes pad/remainder guards around the
  micro-kernel: row/column masks move to the accumulator contraction's transfers (0-filled
  init-load, `If`-guarded store-back; on GPU pipelines the masked fragment is placed in
  workgroup-shared memory so the intrinsics still fire), and reduction-axis masks are discharged
  against zero-fringe staged operands. Tensorized paths thus cover arbitrary extents — a
  33x65x70 matmul register-tiles on cc bitwise-exactly and runs Metal simdgroup intrinsics
  within f32 tolerance. Autotune drops the extent-divisibility pre-filters for fully staged
  pipelines (GPU MMA staged seeds, CPU packed compositions, conv whole/blocked GPU flavors and
  CPU row panels), seeding `(pad, tensorize)` compositions the tuner measures against scalar
  alternatives; in-place pipelines keep their gates.
- **Partition / index-set-splitting transform** (gh-ocannl-508): `Schedule.Partition` splits a
  Serial loop's range at static affine breakpoints into separate, individually specialized (and
  individually schedulable) segment nests; per-segment interval folding then erases the guards
  each segment decides. Replaces in-loop guards for `Split` remainders (partition-then-split =
  guard-free main nest + epilogue) and for inlined concatenations' per-component `Where` range
  guards, with `Schedule.partition_breakpoints` deriving the breakpoints from the guards
  themselves.
- **Hardware matrix units and packed microkernels** (gh-ocannl-412): CUDA now renders
  `Tile_mma` through WMMA for f16/bf16 and inline PTX for fp8/e5m2; HIP renders it through
  rocWMMA on RDNA wave32 devices. CPU schedules compose register-tiled `Tile_mma` with
  cache-blocked operand packing and pool-parallel `Grid` panels. Autotune seeds these forms both
  for whole routines and individual fission segments.
- **Persistent staged tensor-core accumulators** (gh-ocannl-480): Metal, CUDA, and HIP keep MMA
  accumulator fragments resident across the serial outer reduction, loading once before `k_o`
  and storing once afterward. Structural regressions pin the fragment scope, not only numeric
  parity.
- **Convolution schedule families** (gh-ocannl-493, gh-ocannl-500): `detect_conv` recognizes
  lowered affine convolution sites and seeds implicit-GEMM schedules using `Stage` as virtual
  im2col, `Tile_mma`, aligned-segment `Grid` geometry, and cache-blocked row panels. A
  cross-framework `cifar_conv` workload exercises realistic channel counts and loss parity.
- **Epilogue fusion**: `Schedule.Fuse_epilogue` folds eligible elementwise consumers into a
  register/tensor-core tile's store-back, with exactly-once and dependency validation.
- **Conv epilogue twins** (gh-ocannl-501): the tensorized accumulator contraction spans the
  whole kernel-window chain (one fragment init/store per output tile instead of per outer-window
  iteration), so the fused-epilogue twins of the conv sketches apply to 2-D convs; on
  aligned-merged segments the twin omits the preset `Grid` retype on the tail nest it consumes
  (fuse-before-annotate), extending the twins to conv+multi-companion segments.
- **Launch-time symbolic extents** (gh-ocannl-490): bounded symbolic shape axes lower to runtime
  extent parameters, so one compiled routine and schedule-cache entry can serve multiple batch
  or sequence lengths. Allocation and analysis use the declared maximum; autotuning measures at
  that upper bound; serial extent guards fuse into loop headers.
- **Liveness-based buffer aliasing** (gh-ocannl-489): the pool planner derives access spans and
  reuses non-overlapping working-buffer slots. `Zero_out` sinking exposes further training-step
  reuse while preserving observable, merge-buffer, and cross-stream exclusions.
- **Native affine program analysis** (gh-ocannl-494): `Ir.Affine` and
  `Low_level.affine_accesses` expose loop boxes and tensor access relations. Conflict, coverage,
  fiber-cardinality, and read-before-write queries now drive shared-memory safety, fission,
  scratch validation, and a `Schedule.op_legality` oracle that prunes proven-illegal autotune
  proposals, including invalid `Stage` and `Tensorize` roles.
- **Analytic cost-model foundation** (gh-ocannl-491): reusable footprint/FLOP extraction,
  arithmetic-intensity and roofline lower bounds, plus advisory per-backend compute/bandwidth
  envelopes.
- **Cost-model-guided selection** (gh-ocannl-491): `Autotune.model_default` picks the untuned
  default schedule by scoring the default pipeline and the sketch families with the roofline
  model — zero timing runs — behind config `model_default_schedule` (routed through
  `Train.to_routine`/`run_once` and the benchmark runners); `Autotune.tune` gains a model
  pre-filter over sketch seeds with configurable keep-fraction (`autotune_keep_fraction`).
  Candidates without model coverage are always kept — never dropped, only measured — so the
  model never overrides or precludes a measured result. Calibration logging pairs model scores
  with measured times (`autotune_calibration_file`, plus `autotune_log` lines), and
  `model_peak_flops`/`model_peak_memory_bandwidth` accept per-machine calibrated envelope
  constants that also enable scoring on backends without advisory values.
- **Swizzled shared staging**: `Stage ~swizzle:true` marks XOR-swizzled shared-memory tiles, with
  validation and explicit intrinsic-decline behavior until swizzle-aware fragment loads land.

### Changed

- **An unparallelized candidate is refused, not dispatched, on GPU backends** (gh-ocannl-532). A
  kernel binding no hardware dimension runs the whole routine in one work-item: on Metal LeNet's
  serial baseline measures 6.9 s per run against a 35.7 ms winner, and on gfx1151 the same
  dispatches ran for hours, uninterruptible, on the device driving the display. This is not
  specific to the baseline — supplying any `?lowered_transform` bypasses the default annotator, so
  the tuner's base compile is the unscheduled serial form on every backend — so `dispatchable`
  gates the baseline and every candidate alike, and `baseline_ms` is infinity where the baseline
  is not run. Three consequences: a search that timed nothing stores no cache entry and returns
  the untuned default compile (rather than caching a never-measured schedule); a cached entry that
  replays to an unparallelized GPU routine is rejected like a stale one, so the fresh search
  overwrites it; and beam moves off an undispatched incumbent are pruned before compile, since
  none of them can introduce a hardware dimension. CPU backends are unaffected by construction —
  the serial form runs at full single-core speed and stays a legitimate competitor.
- **Attention masks fill with `-infinity`** (gh-ocannl-548). The causal mask filled masked-out
  scores with `-1e9`, four orders of magnitude past fp16's largest finite value.
  `Nn_blocks.default_mask_fill` is precision-independent and needs no re-deciding per storage
  format; this is numerics-preserving for models that already worked (`exp` underflows to exactly
  zero for both values at f32, and no existing golden moved).
- **The fp16 magnitude guard is a headroom policy, so infinities are out of its scope**
  (gh-ocannl-547). `Ops.exceeds_fp16_cutoff` tested `abs c >= cutoff`, trivially true for any
  infinity — so it refused `-inf`, which is exactly representable in binary16 and which
  `c_syntax.ml` already emits deliberately as `(-INFINITY)`. No arithmetic can push an infinity
  past a finite bound, so there is no headroom to preserve; finiteness is tested first, covering
  `Min`'s `+inf` identity as well as `Max`'s `-inf`. Refusing them made every fp16 max-reduction,
  hence every fp16 softmax, hence every attention model at f16 fail during lowering on every
  backend.
- The HIP backend reports OCANNL's own slab allocation from `get_used_memory` (gh-ocannl-542),
  matching what gh-ocannl-289 did for CUDA, instead of the driver's device-global `total - free` —
  a granule-quantized number that counts every other process's VRAM and is not even monotonic in
  what OCANNL allocated.
- `Schedule`'s `Tensorize` declines name what broke the discipline: the offending body statements
  for a non-perfect nest, and the operands' *index expressions* against the micro-kernel symbols
  for the unit-coefficient role check. Both were previously unactionable.
- The benchmark runners report the scheduling variant alone rather than folding the storage
  precision into it, now that `orchestrate.py` composes the two axes itself.
- **`Tnode.t` field renames**: `memory_mode` is now `memory_mode_intent` (it has been intent-only
  since the context-scoped `Placements` migration) and `prec` is now `storage_prec` (the settled
  bytes-in-buffers precision, as opposed to the compute precisions the numerics policy governs).
- **Packed `uniform` is total over shapes** (gh-ocannl-509, tasks 1-3): the result's element
  count no longer needs to be a multiple of the 128-bit block width (`16 / bytes-per-element`).
  Shape inference gives the counter `ceil(total / lanes)` blocks via a round-up mode of the
  `Strided_var` total-elements constraint (with a `Range_elems` slack window once the counter is
  solved), and lowering peels the final counter iteration into a shorter `Set_from_vec` store.
  The value stream depends only on the element index, so growing a tensor preserves its prefix
  bitwise, and divisible shapes keep their previous streams unchanged. `Set_from_vec` into a
  padded (halo) target is now rejected explicitly at lowering with a materialize-through-copy
  remedy (it previously assumed dense flat offsets without checking).
- **Default parameter initialization uses the packed `uniform`** (gh-ocannl-509 task 5):
  `default_param_init` now defaults to `default_uniform_param_init` (a centered, scaled packed
  `uniform` over `[-0.25, 0.25)`). Since the packed conversion became total over shapes, the
  pointwise `uniform1` fallback is no longer needed: `uniform1`,
  `centered_uniform1_param_init` and `default_uniform1_param_init` are deprecated (kept for
  reproducing pre-0.9 random streams; `uniform_at`/`uniform_at1` — pointwise mapping of a user
  counter — remain first-class). This changes every default-initialized random stream:
  expectation files were re-promoted suite-wide. Composite initializers over inferred-shape
  parameters exposed premature round-up eliminations in the row solver: the counter was guessed
  at one block while the result rows' broadcast bounds were still arriving. Round-up
  `Total_elems` eliminations are now bounds-aware — a row variable with registered-but-pending
  bounds defers through the stored-constraint path, sibling rows are only closed empty before
  the final stage when they carry no bounds at all, and the resolved-bound consumption arm also
  fires when closed axes folded into the constraint's denominator (previously such a constraint
  sat stored forever, leaving the counter variable unsolved).
- Padding layout and neutral values are committed as tensor-node identity. Padded convolutions
  consequently lower offset-free and can be staged safely; incompatible later padding demands
  fail during shape inference instead of silently reinterpreting an existing buffer.
- Per-fission `F_sketch` candidates are enumerated one segment at a time instead of pairing each
  segment's nth seed, so an invalid seed on one segment no longer masks viable schedules on
  another.
- CUDA and HIP slab allocators report exact allocated bytes rather than inferred pool capacity.
- Development pins now follow their upstream branch heads; release packaging still requires
  released dependency versions because opam-repository packages do not preserve `pin-depends`.

### Fixed

- **bfloat16 math builtins on every GPU backend** (gh-ocannl-549). MSL, CUDA and HIP have no
  bfloat overload of the math library, so a builtin called on bfloat16 operands promotes them and
  returns `float` — the op tables' own `expf`/`logf`/`sqrtf` arms return one, nothing downstream
  introduces it. One root site, three symptoms, because the dialects disagree about where the
  float is illegal: MSL rejects the narrowing assignment, so *every* placement fails (~50
  instances in `gpt2_mini` at bf16, from `sqrt` in `Nn_blocks.layer_norm` and `fmax` in the
  softmax max-reduction), while CUDA and HIP accept it and fail only where inlining makes the
  float an *operand* of a bf16 binop (nvrtc on a mixed-operand `__hadd`, hiprtc on an ambiguous
  `operator '+'`) — which makes a placement-dependent bf16 compile error a clue about inlining,
  not about a fission-introduced mixed type. All three now bridge the result back to bfloat16 on
  the family rather than operator by operator. Verified against the compilers rather than the
  specs: f16 has no such gap, and bfloat arithmetic, comparisons, the ternary and the `0.0bf`
  literal suffix are all native.
- **A tensor node named after a builtin broke codegen** (gh-ocannl-553) — see the reserved-
  identifier entry above.
- **A cast twin inherited its use site's batch axes** (gh-ocannl-540). `Operation.cast` is a
  shape-inferred pointwise op, so a master-weights twin's batch row starts as an open row
  variable; read as the weight operand of a batched matmul, that row is resolved by the use site
  and the twin materializes as `[batch, out, in]` — for `mlp_small`, a 64x64x64 per-batch-row copy
  of a 64x64 weight. Every slice holds the same value, so the loss-trajectory oracle could not see
  it (its input carries no batch axis); the costs are 64x the twin's memory and cast work per step,
  and the row symbol appearing in the matmul's weight operand, which is what made every tensorized
  candidate decline on HIP. A parameter has no batch axes, so neither may its twin. Measured on
  gfx1151, `mlp_small` bf16 tuned: 0 → 17 timed mma candidates, all genuine tile-MMA.
- **A declined baseline no longer ends the search** (gh-ocannl-533). The tuner's base compile — the
  identity-transform capture at the top of `tune` — was the one compile there going through
  raising `Context.compile` rather than `compile_outcome`, so a rejection bypassed `try_spec`, the
  census and the partial report, and took the whole search with it before a single candidate had
  been tried. It is now protected like any other candidate; the capture happens inside the
  transform, so the base lowering survives a rejection and every candidate still derives from it.
  The baseline record became optional, which lets the candidate pool be empty — the state every
  consumer already reads as "nothing was timed". The benchmark diag runners got the same treatment
  and now report a declined segment rather than ending the run.
- **gh-ocannl-521's companion-coverage rejections are typed declines**, not `invalid_arg`. Under
  the now-default `strict_failure_classification` an unattributed compile-side failure is fatal,
  so at stock config an expected decline aborted the entire search on every GPU backend (this is
  shared middle-end code). Two further containment holes found in review: Metal device discovery
  asserted at least one device inside a lazy, so a Metal-linked machine with no Metal device died
  where fall-through to `cc` was intended; and `Context.copy` bypassed the lineage-poisoning guard
  on the same-backend path.
- `autotune_fission_sketch`'s "timed multiple candidates" assertion was cc-shaped and read false on
  HIP (gh-ocannl-543) — not a regression, but 16 candidates refused under gh-ocannl-532 and only
  ever visible in the stderr log. `candidates_timed` is not comparable across a CPU and a GPU
  backend; the assertions are now over the population the census covers, and both goldens are
  byte-identical on `cc` and `hip`.
- Reduced-precision test goldens cannot pin transcendental results: backend math libraries disagree
  in the last mantissa bit (HIP's `exp` differs from `cc`/`metal`/`cuda` by one ulp at 2^-13), and
  no choice of inputs makes an `exp`/`log`/`tanh` output exactly representable. `half_softmax`
  prints coarsely enough to sit above that divergence and moves its numeric content to a tolerance
  comparison against the same softmax evaluated in double.
- **GPU mma sketch candidates unreachable by the tuner** (gh-ocannl-521, route 1):
  `Schedule.Fuse_epilogue` fuses at two previously rejected sites — the whole-K `Tile_mma`
  writing the accumulator directly (the unstaged tensorized pipelines; the tail becomes a
  sibling lane-0 nest over the completed m x n tile), and the pad-masked fragment store-back
  (gh-485 range guards on non-dividing sites; the guards are collected and re-imposed on the
  relocated tail, with guard-aware coverage). The GPU mma seeds' fused twins — their only
  survival path past `validate_parallel`, since companion nests are otherwise uncovered — now
  compile and reach timing instead of failing 100% of the time on Metal/CUDA/HIP;
  `test/operations/epilogue_fusion_mma_seeds.ml` pins seeded → applies/validates → runs with
  correct values against the real seed enumeration. Covering companions without fusion is the
  complementary route 2, tracked on the issue.
- **Non-overlapping pooling gradient cost** (gh-ocannl-527): the gh-512 product-space gradient
  gate made overlapping max windows exact but cost 1.8-2.6x on the conv benchmarks, whose
  pooling is non-overlapping and cannot exercise the exactness. `Operation.einmax1`/`tropical`
  take `?nonoverlapping` restoring the input-space gate on the domain where the two
  formulations agree exactly (ties included) — each RHS1 position feeding at most one result
  position — and `max_pool2d`/`max_pool2d_copy` dispatch on `stride >= window_size`. Bitwise
  gradient parity between the gates on that domain is pinned by
  `test/operations/nonoverlap_pool_grads.ml`; overlapping pooling keeps the exact
  product-space gate.
- Scalar-embedded multi-term affine indices are parenthesized in generated C-family code (an
  unparenthesized `5*i+j / 4` divided only `j`), and integer-precision `Mod` renders the `%`
  operator on Metal/CUDA/HIP instead of the float-only (and on Metal ambiguous) `fmod`.
- `Total_elems` application against a closed row includes the row's `beg_dims` in the known
  product (block/stack rows carry leading dims there); previously a total constraint meeting a
  block row divided by the trailing dims only.
- CUDA random generation no longer bit-casts random bits directly to `double`, and vector helper
  names no longer exceed CUDA identifier limits. fp8 conversion, arithmetic, and uniform
  generation were corrected across C, CUDA, HIP, and Metal paths.
- The process-global slab-pool table is synchronized across `multidev` worker domains, fixing a
  race between pool growth and lookup.
- Shape inference no longer hangs on valid-convolution specs with output-only axes; `Embed_dim`
  lowering forces the solved dimension before reading it.
- Padded max-pooling commits `-infinity` margins on its private copy, and mixed-anchoring row
  unification no longer loses compatible bounds.
- The affine read-before-write proof cancels false `Recurrent` classifications caused by bounded
  tracing on padded convolution/pooling chains, avoiding unnecessary materialization.
- CIFAR dataset downloads are atomic, fail on HTTP errors, tolerate Windows Schannel behavior,
  and self-heal incomplete archives; Windows documentation now calls out PowerShell's unquoted
  Dune-alias splatting trap.
- The model-picked untuned default (`model_default_schedule`) now honors its advisory contract
  against validation failures (gh-ocannl-519): its guard only covered *producing* the picked
  segments, while `Low_level.validate_parallel` — the check that rejects most bad picks — runs
  inside backend codegen, past the transform seam, so a rejected pick escaped the guard and
  killed the process (reproducibly, on Metal `mlp` workloads). Picks are now validated at the
  seam (`Autotune.validate_segments`), and the compile itself is wrapped
  (`Autotune.compile_advisory`) so that a downstream failure of a picked schedule recompiles
  with the ordinary default pipeline, the reported choice degrading to `"default"`. Once the
  compile is on that pipeline there is nothing left to fall back to, so its failures propagate
  without a duplicate attempt.
- The cost model's untuned-default selection no longer ranks candidates it cannot build
  (gh-ocannl-522): the roofline has no notion of compilability and rates the tensorized families
  best, so on backends where those are unbuildable (gh-ocannl-521) it crowned one and — with the
  fallback above — degraded straight back to the default, making the gate a no-op exactly where
  it looked most promising. Each candidate's scheduled form is now checked with
  `Low_level.validate_parallel` during scoring, on the hermetic copy it is already scored on, and
  the dead ones drop out of the argmin; the count is reported as `model_choice.mc_rejected` and
  logged under `autotune_log`. On a 128x128 matmul+relu on `cc` this changes the outcome from
  "default" (22 of 64 candidates unbuildable) to a packed mma sketch that compiles and runs.

## [0.8] -- 2026-07-13

> Release note: theme — parallel schedules and autotuning; AMD HIP backend. Scope changes
> vs. the original plan: **tensor cores are pushed out to v0.9**; conversely **autotuning**
> was not on the original roadmap and lands here as multi-round, execution-measured beam search
> over schedule compositions. See [ROADMAP.md](ROADMAP.md), whose older single-step/full-beam
> distinction is superseded by the implementation described below.

### Added

- **Schedule IR with automatic GPU schedules** (docs/proposals/axis-types-for-loops.md,
  docs/proposals/schedule-ir-optops.md): `Low_level` loops carry an axis type
  (`Serial | Grid | Workgroup | Workgroup_reduce | Unrolled | Vectorized`); `Grid`/`Workgroup`
  loops render as hardware index bindings with launch dimensions, barriers, workgroup-shared
  declarations, and `Low_level.If` extent guards (a new guarded statement that is a barrier for
  CSE/hoisting/virtualization). Schedule transforms are values (`Split`, `Swap`, `Retype`,
  `Unroll`, `Stage` workgroup-shared/packed tiles, `Expand_zero`, `Privatize` accumulator
  privatization, `Tensorize`) applied at the new `Context.compile ?lowered_transform` seam.
  On cuda/hip/metal, kernels without an explicit transform get the default GPU schedule when a
  conservative analysis proves them race-free (`automatic_gpu_schedule`, default true;
  `gpu_schedule_block_size`, `gpu_schedule_min_parallel`); the annotator clamps block sizes to
  the backend's new `hardware_limits` query, and compile validates every kernel's thread count
  and shared-memory bytes (`Schedule.check_hardware_limits`), turning driver launch failures
  into early named errors.
- **Kernel fission** (`schedule_fission`, default true): routines are split into segment kernels
  at the cross-workgroup dependency edges the race analysis would otherwise reject (materialized
  producer/consumer and WAW/WAR pairs, bare materialized writes, whole-node zeroing); segments
  launch in order with a device-side event chained at each boundary and each gets its own launch
  geometry, while adjacent serial segments coalesce back to a single kernel. Aligned cross-nest
  parallelism keeps race-free chains fused: nests linked by producer/consumer pairs stay in one
  kernel when every member trims to a common equal-extent parallel prefix with per-axis-aligned
  accesses, and fission merges across a materialized edge only when lossless (no nest's
  parallelism shrinks). On Metal, a fissioned routine's segments encode into one command buffer
  with a serial-dispatch compute pass (`sequence_segments` backend hook), replacing per-boundary
  event command buffers (circles_conv sgd step 19.7 → 7.1 ms).
- **Within-kernel CPU parallelism** (gh-ocannl-164): the default CPU schedule
  (`automatic_cpu_schedule`, `cpu_schedule_min_parallel`) retypes each nest's outermost
  parallelizable loop to `Grid`, which the C backend renders as contiguous chunks on a
  process-global native pool — `dispatch_apply` on macOS, chunked OpenMP elsewhere
  (`cc_parallel_grid=auto` probes the compiler; `cc_parallel_chunks`); worker threads run pure C
  and never touch the OCaml runtime, and results are bitwise identical to serial execution.
  The rest of the CPU-improvements bundle: `restrict`-qualified kernel pointers, 32-byte-aligned
  pool bases and node offsets, pragma hints on `Vectorized` loops, and probed SIMD compiler
  flags (`cc_backend_simd_flags=auto`).
- **Explicit SIMD codegen on the C backends** (`cc_vector_bytes`, default automatic AVX2/NEON
  width): eligible `Vectorized` loop bodies emit GCC/Clang vector-extension loads, arithmetic
  (including fused vector FMA), stores, and a serial remainder instead of relying on the
  auto-vectorizer; recognized accumulation bodies render as independent vector accumulator
  chains with a horizontal reduce at loop exit, ggml's `vec_dot` pattern (gh-ocannl-468); and a
  `Tile_mma` statement renders tinyBLAS-style register tiling — a grid of vector-register
  accumulators held across the whole k-loop with splat/FMA updates and peeled edges
  (gh-ocannl-469).
- **Tensor-core / GPU vector rendering**: the `Tensorize` optop recognizes the matmul
  micro-kernel and replaces it with a cooperative `Tile_mma` block statement (Metal emits
  `simdgroup_matrix` MMA, CUDA has a draft wmma path with a lane-0 scalar fallback);
  `Workgroup_reduce` accumulations render as llm.c's two-phase block reduction with warp /
  simdgroup shuffle butterflies (gh-ocannl-462); `Vectorized` loops on CUDA and Metal emit
  128-bit packed loads/stores through reinterpret casts with per-lane scalar arithmetic
  preserving serial rounding (gh-ocannl-463).
- **Autotuning**: `Autotune.tune` is a drop-in replacement for `Context.compile` that beam-searches
  schedules (seeds: serial baseline, default annotator, block-size sweep; menu: splits, swaps,
  unrolls, `Retype`-`Vectorized`, `Tensorize` permutations), timing each candidate on the real
  device and returning the fastest routine (`autotune_beam_width`, `autotune_rounds`,
  `autotune_repeats`, `autotune_log`). `Schedule_cache` gives schedules a structural identity —
  canonical symbol/tnode naming with a digest that guards replay — plus a disk cache
  (`autotune_cache_dir`) so re-running the same program skips the search. Follow-ups: fissioned
  candidates with per-segment schedules keyed by pre-schedule segment digests, a matmul sketch
  generator seeding register-blocktiled/SMEM/packed pipelines, `?timing_ctx` to search on a
  scratch context lineage without touching live training state, and `Train.tune_placements`
  which A/B-tunes default vs materialize-all placements (`Context.decide_materialized`) and
  keeps the measured winner — tuned is now the fastest OCANNL variant in every benchmark cell.
- **AMD HIP backend** via the hipjit bindings (gh-ocannl-411): `hip_backend.ml` mirrors the CUDA
  backend with hiprtc compilation to a code object, device-keyed peer copies,
  `__hip_bfloat16`/fp8 type support (conversions routed through float), ported builtins with
  wave32/wave64-correct shuffles, and the lane-0 fallback for tile-MMA. Registers everywhere
  the other backends do (`Backends.Hip`, `Context.hip`, the `Context.auto` fallback list,
  per-backend test goldens, config key `hip_printf_fifo_size`), including on Windows.
- **Cross-framework benchmark suite** under `benchmarks/`: OCANNL vs PyTorch (eager and
  `torch.compile`) vs tinygrad (default and BEAM) training/inference steps on shared
  safetensors fixtures — mlp_small/mlp_wide, LeNet-5, and a GPT-2-style gpt2_mini inference
  workload — with an orchestrator that gates every cell on loss-trajectory parity against the
  PyTorch CPU reference before timings count, and reports synced and queued per-step
  percentiles with compile time separated out. Example reports checked in for Metal, CUDA, and
  Windows/HIP. The parity gate doubles as a correctness oracle: its first run caught two real
  backward-pass bugs (see the gradient-correctness entry under Fixed).
- **Fused cross-entropy classifier** (gh-ocannl-464): `Nn_blocks.cross_entropy_loss` computes
  log-sum-exp cross-entropy from raw logits (optional `?mask` / `?normalize_by`), detaching the
  row max via `stop_gradient` (now exposed in the DSL `O` modules) so the backward accumulates
  `probs - targets` directly into the logits gradient with no `[batch, vocab]` intermediate
  materialized in either pass; `transformer_with_loss` uses it instead of the unstable
  `log (softmax logits)`.
- **GPT-2 with pretrained weights**: `Nn_blocks.gelu` (tanh-approximate), a HuggingFace
  safetensors checkpoint reader (`Safetensors`, lazy payloads with exact buffer tiling), a
  GPT-2 model module assembling HF `GPT2LMHeadModel` semantics from OCANNL operations (verified
  exact against a NumPy reference), a full-scale 124M dry run, and a `gpt2_generate` tutorial
  that downloads the pretrained tokenizer + weights and greedy-decodes. Plus the dataprep BPE
  bridge `Nn_blocks.token_ids_of_array` / `token_ids_of_batch` (uint32 IDs, padding/truncation,
  composing with the one-hot embedding gather) and a hermetic tokenizer-roundtrip test.
- On-device epoch-loss accumulation: `Train.grad_update ?accum_loss` with
  `Train.loss_accumulator` sequences `acc =+ loss` after the update, so training loops read the
  loss once per epoch instead of forcing a device-serializing `Context.get_values` every step.
- Interval analysis over index expressions and scalars (gh-ocannl-134 lineage): an interval
  lattice with a total symbol environment lets `simplify_llc` fold interval-decided comparisons
  and `Where` branches (construct-then-fold for schedule guards), erases provably-true
  dynamic-gather guard conjuncts, and backs `Tnode` bounds propose/settle plus int32
  launch-parameter width checks.
- Recompute-cost guard for virtualization: nodes whose computation contains reduction loops
  with trip-count product exceeding `virtualize_max_inline_reduction` (default 16) are decided
  `Never_virtual`, since inlining would replay the reduction at every read site through chains
  of virtual consumers (gpt2_mini on cc: ~13,000 → 2,361 ms/step at defaults).
- Namespaces for tensor node IDs (gh-ocannl-372): `Tnode.t` carries a namespace derived from
  the defining file (`<prefix>__<file_ns>` debug names), `Tensor.t.id` was dropped in favor of
  the tnode's uid, per-routine node tables are keyed by uid rather than session-dependent ids,
  schedule-minted tile nodes live in a reserved `tile` namespace, and checkpoints are
  namespaced (`Persistence.load ?prefix_namespace`). (Listed under 0.7 in earlier drafts of
  these notes, but merged after the 0.7 tag.)
  `rewrite_one_hot_reductions` also recognizes the transposed one-hot pattern — the
  embedding backward `d_C[o,v] += Σ_pos (v == ids[pos]) * g[pos,o]` — and replaces the
  vocabulary loop with a guarded scatter-accumulate at the dynamic row (`Set_dynamic`,
  the write counterpart of gh-343's `Get_dynamic` gather), dropping the O(vocab)
  per-position work. Positions accumulate in their original serial order and the
  schedule analyses treat dynamically-indexed writes as never parallelizable, so results
  stay deterministic with no atomics (llm.c's encoder-backward invariant; host-side
  bucketing for GPU position-parallelism is a possible phase 2).
- Packed constants (gh-ocannl-470): `Schedule.Stage` gained `hoisted = true`, packing a
  compile-time-constant operand once, out of the routine — the packed layout covers the
  whole source, is computed on the host at link time from the operand's host-init data
  (the compiler-native analog of ggml's `CPU_REPACK` `set_tensor` hook), and is uploaded
  once per device into the constant pool; the kernel reads it directly, with no
  per-invocation load nest. The CPU autotune sketch proposes hoisted and in-kernel
  packing side by side for constant operands, so the choice stays measured. Constancy
  rides a new `Tnode.host_constant` marker set by `Tensor.ndarray` (ndarray-backed
  literals are minted `On_device`, so `Effectively_constant` intent could not stick).

### Changed

- **Breaking notation change**: in einsum and labels specs, a row whose kind separator
  is omitted now reads as the context ellipsis instead of an empty row: `x` is
  equivalent to `...|...->x`, so batch and input axes broadcast through terse specs by
  default (e.g. `counts ++ "... => 0"` is a per-row softmax denominator). The empty-row
  reading is preserved when the separator is written with an empty row spec: `| ->x`
  means no batch and no input axes, `|x` no batch axes, `->x` no input axes. Because an
  omitted row shares the context row variable with rows written `...`, reduce-over-
  everything results must now close their rows — the sum-to-scalar idiom
  `++ "...|... => 0"` becomes `++ "...|... => |->0"` — and multi-operand specs where
  one slot passes batch through must close the other slots' batch rows (e.g. the
  conv kernel slot `; |kh, kw, ..ic.. -> ..oc..` in `Nn_blocks.conv2d`).
- **The default backend is now `cc`**; `sync_cc` was renamed to `cc` and `multicore_cc` to
  `multidev_cc` (old names accepted as deprecated aliases). The `Multidev` scheduler exposes
  multiple worker-domain CPU devices, each with its own FIFO queue, for debugging multi-device
  parallel workflows (`multidev_num_devices`, 0 = recommended domain count).
- **Breaking**: backends are process-wide singletons — `fresh_backend` became `get_backend`
  returning a closed enum, `Backends.wrapped_context` is a closed disjunction over the
  singleton context types (matching two values on the same constructor recovers type equality),
  and `Context.copy` dispatches through it to backend-specific transfers. Backend module tower
  cleanup retired `new_stream` and vestigial signatures. GPU driver init and device discovery
  stay lazy.
- Memory placement decisions are context-scoped: `Tnode.memory_mode` (streamlined to five
  constructors, `Materialized`/`Device_only` folded away) is declared intent only — monotone and
  never written by the compilation pipeline — while decisions land in per-lineage
  `Tnode.Placements` tables riding `Low_level.optimize_ctx`, forked per compile so sibling
  compiles from one context are hermetic (the autotuning forcing function); gradients'
  `Never_virtual` was replaced by an `is_observable` intent.
- Index arithmetic is now signed: `Ops.index_prec` is int32, or int64 under `large_models`
  (unsigned precisions remain for data domains); loop counters and index kernel arguments are
  signed across cc/CUDA/Metal (Metal pool-slot offsets deliberately stay unsigned), and each
  tensor node's padded element count must fit int32 unless `large_models` (checked with a named
  error when dims are forced).
- Generated code and runtime logs go to per-executable subdirectories
  `build_files/<exe>/` and `log_files/<exe>/` (override with `build_files_prefix`; `"."`
  restores the flat layout), so concurrently running tests sharing a working directory no
  longer race on same-named kernels or wholesale startup cleanup.
- The four gated training tests (bigram, fsm_transformer, transformer_names, circles_conv)
  moved from the `slow` alias back to `runtest`: they materialize all intermediates and compile
  through `Autotune.tune ~rounds:0` with a scratch `timing_ctx`, cutting e.g. bigram on Metal
  from 345 s to 25-29 s while keeping deterministic expected outputs.
- Autotune and benchmark step timing uses the monotonic mtime clock
  (QueryPerformanceCounter-backed on Windows): `Unix.gettimeofday` ticks at ~1 ms there, which
  floored sub-millisecond step times to 0/1 ms and made the tuner pick winners by noise.
- Workshop article: benchmarks section and appendix with Metal, CUDA, and partial AMD HIP
  tables (including torch.compile and tinygrad BEAM rows), and automated Markdown-to-LaTeX
  conversion.

### Fixed

- Two executed-backward gradient-correctness bugs found by cross-framework loss parity
  testing (regression test `test/training/virtual_grads_parity.ml`):
  - Per-statement CSE (`Low_level.eliminate_common_subexpressions`, gh-ocannl-351) and
    cross-statement hoisting treated *free* iterator symbols and scope ids as renameable
    during alpha-equivalence, so a nested recomputation of a virtual node at inner loop
    indices was deduplicated into a `Get_local` of the enclosing iteration's stale local
    (e.g. `cross_entropy_loss` backward through a Virtual pre-activation used one class's
    logit for every class in its exp-sum). Renamings are now registered only at binder
    sites (`For_loop` indices, `Local_scope`/`Declare_local` scope ids); free names must
    match exactly.
  - The scalar simplifier rewrote `(a / b) / c` to `(a * c) / b` (a regression from the
    precision-handling refactor; originally `a / (b * c)`), corrupting e.g. the division
    backward of the standalone `Nn_blocks.softmax`.
- 1-element `Reshape`/`rebatch` data no longer collapses to a scalar: when the rows
  receive no other shape information, the leftover row variable keeps a single dim-1
  axis (identity-reshape preference, `keep_axis` on the `Total_elems` constraint); also
  fixed `Tnode.create_with_reshape` crashing on rank-0 targets (gh-ocannl-460).
- `Nn_blocks.layer_norm` now computes the actual mean and variance: it used the `++`
  add-reduction result as if it were a mean — `(x - sum(x))/d` instead of `x - sum(x)/d`, and
  never divided the squared-deviation sum by `d` — an error not absorbable by the learned
  gamma/beta (fixed by scaling the reduction operands, `sum(x /. d) = mean(x)`); added a
  numeric forward+backward oracle test.
- Shape-inference fixes surfaced by the layer_norm work: `compute_row_product` no longer
  latches a product while the row has an open variable tail (xavier/kaiming fan_in was silently
  1, producing wrong init scales — training tests retuned for real fan values, and
  `TDSL.default_param_init` fans fixed likewise); fixed-index axes are kept out of the
  singleton-bounds dimension equality; and GLB-closing a row variable re-emits its `Shape_row`
  constraint so deferred guess-to-1 variables still resolve.
- Repeated random values in inferred-shape parameter initialization: the PRNG counter's shape
  is now pinned to the result in `uniform`/`uniform1` (and `box_muller` keeps its draws pinned)
  via an optional `?spec` on the threefry ops — previously shape inference could close the
  counter's rows smaller than the result and broadcast it, repeating values along the broadcast
  axes (e.g. 50 unique of 5000 for a `[100 -> 50]` weight); regression test on realized
  parameter std.
- `Nn_blocks.conv2d` (and `depthwise_separable_conv2d`) inline bias is per-channel: least-
  commitment inference broadcast the plain `+ { bias = 0. }` to the full feature map (4704
  params for LeNet-5 conv1 where 6 are expected); the bias slot is now pinned to the channel
  row via a spec'd `+++`.
- Forward and backward fragment ordering (gh-ocannl-461): sibling forward fragments are ordered
  topologically by embedded-nodes/read-nodes instead of tensor-id order (GPT-2's attention v
  projection silently read zeros), backprop fragments are ordered by gradient-accumulation
  dependencies with the forward edge reversed, and fragment cycles error instead of silently
  reordering.
- Worked around an Apple Metal shader-compiler miscompile of serial loops accumulating into a
  loop-invariant address (only the last iteration contributed — cross-entropy sums collapsed to
  `correct/batch_size`): affected read-modify-write statements go through a volatile shadow
  pointer; standalone repro checked in. Also on Metal: every kernel launch is now ordered after
  all previously enqueued work (back-to-back runs of the same routine raced, producing NaN),
  `get_values`/`set_values` do full device awaits, and `from_host` uploads await pending device
  work.
- Metal default-schedule pathology on gpt2_mini (81 s → 0.3 s steps): `gpu_schedule_min_parallel`
  default lowered to 64 — any real parallelism beats the serial 1×1 fallback — and `Local`
  scratch crossing a statement boundary is promoted at fission instead of stranding.
- Windows: the CUDA backend was restored (NVRTC targets compute_52 by default, but half
  arithmetic intrinsics need compute_53+ and their bf16 overloads compute_80+ — the arch
  heuristic now floors accordingly), and the test suite is green on the cc and hip backends
  (binary-mode stdout for byte-identical goldens under autocrlf, `NUL` vs `/dev/null`,
  reduction-order-tolerant loss printing, fixed-point formatting for mingw's `%g`).

## [0.7] -- 2026-07-03

> Release note: **0.6.4 is skipped** as a tagged release (last release before 0.7 was 0.6.3). The
> work originally planned for 0.6.4/0.6.5/0.7.0 (frontend finalization, concatenation,
> position embeddings, transformer toy) and for 0.7.2 (compiler optimizations, pool
> allocator) is **consolidated into 0.7**. See [ROADMAP.md](ROADMAP.md).

### Added

- **Removed the hosted tensor mode** (gh-ocannl-333): dropped the `array` field of
  `Tnode.t` and the "hosted" memory mode. Tensor value access and printing are now
  context-mediated; host-init nodes self-initialize at link time. Removed the dead
  `automatic_host_transfers` setting and the `use_host_memory` hook.
- Tensor saving, loading, and restoring (gh-ocannl-373).
- Ternary einsum notation: `Einsum_tern` in shape inference, PPX dispatch, a `Mul3`
  ternary scalar op across all backends, and `einsum3` / where-with-spec (gh-ocannl-305).
- Loop-invariant code motion (loop hoisting) prior to visit counting (gh-ocannl-350).
- Common subexpression elimination after inlining (gh-ocannl-351).
- Virtual-node inlining extended to non-scalar constants and ranges (gh-ocannl-142).
- `Uint32`/`Uint64` precisions, with index-embedding operations selecting precision per
  the `large_models` setting (formerly `big_models`, still accepted as a deprecated alias)
  to avoid unnecessary conversions and to select Metal pool-slot width
  (gh-ocannl-349, gh-ocannl-177, gh-ocannl-344).
- `-march=native` C-compiler flag (gh-ocannl-311); restored CUDA pre-loaded builtins
  referenced by pointer via a cudajit helper (gh-ocannl-353).
- Sasha Rush Tensor Puzzles expressed in the extended einsum notation (gh-ocannl-308).
- Universal pool allocator across backends (gh-ocannl-344): tensors are addressed by
  `{ pool_id; offset }`; working tensors are bump-packed into per-context-delta pools,
  constants into per-device constant pools, and merge buffers stay in their reserved pool.
  Metal now binds pool slabs plus a slot table instead of one buffer per tensor node, staying
  below the backend's binding limit for large routines. CUDA and C keep per-tnode pointer
  kernel parameters resolved from the same pooled locations.
- Data-parallel training: `shard_along` / `gather` sharding primitives and a driver
  with merge-buffer gradient all-reduce (part of gh-ocannl-293).
- Zero-copy slice views (`@|` / `Fetch.Slice`, gh-ocannl-293 subtask 293a): an
  alias-eligible leading-axis slice no longer materializes a copy — it is lowered as a
  view that redirects reads/writes to the parent buffer at the (runtime) batch index.
  **Semantics change**: writing through such a slice now mutates the parent (and vice
  versa). Slices fall back to the materializing copy loop when ineligible (padded,
  precision-converting, virtual/constant parent, or non-leading-axis). Host-side value
  access (`Context.get_values` / `set_values`) of an alias view is rejected with a clear
  error — read or write the parent tensor instead.
- Benchmark tables partitioned by `result_label` into per-group sub-records (gh-ocannl-140).

- `Nn_blocks.batch_norm1d` — MLP batch normalization that normalizes over the
  batch axis only. Mirrors `batch_norm2d`; inherits its running-statistics
  FIXME (inference uses the learned `gamma`/`beta` on batch statistics rather
  than population estimates).
- `test/training/mlp_names.ml` — Bengio-style MLP (makemore Part 2): learned
  character embeddings, `block_size = 3` context, einsum-contracted hidden
  layer, deterministic 80/10/10 train/dev/test split. Prints numeric final
  train/dev/test NLL plus threshold booleans, then three generated names.
- `test/training/mlp_bn_names.ml` — MLP + `batch_norm1d` (makemore Part 3).
  Same data pipeline as Part 2 with BatchNorm between the hidden linear and
  `tanh`. Documents the single-example-inference collapse from the
  running-stats FIXME.
- `docs/makemore_tutorial.md` — walk-through of the makemore progression
  (Parts 1–4 + cross-link to the transformer variant) mirroring Andrej
  Karpathy's *Neural Networks: Zero to Hero* lectures, with a README entry
  and Part 4 instructions for inspecting the generated backward code.
- Axis concatenation/block tensor support in einsum notation (`a^b` syntax)
  - Tensor concatenation (`a; b => a^b`)
  - Axis slicing to extract prefix/suffix (`a^b => a`, `a^b => b`)
  - Block tensor construction with n-ary einsum specs
  - `invalid_vars` tracking for determining which dimension variables can be 0 in Block specs
  - New `++^` operator for concatenation in DSL
  - Concat projection unification in shape inference (`solve_proj_equations`)
- Pointwise operations now optionally accept einsum/permute specs via `?spec` and `?capture_dims` parameters
  - Binary ops: `add`, `sub`, `pointmul`, `pointpow`, `pointdiv`, `lt`, `eq`, `ne`
  - Unary ops: `relu`, `sat01`, `exp`, `log`, `exp2`, `log2`, `sin`, `cos`, `sqrt`, `recip`, `recip_sqrt`, `tanh`, `neg`, `not`, `stop_gradient`
- Common gotchas and idioms section in CLAUDE.md documentation
- `Rev_sides` support in lowering for reverse-direction Block operations
- Completed the workshop article in Markdown and LaTeX, with a rendered PDF published
  under `docs/html/pdfs/ocannl_workshop_article_human.pdf`.
- Added the standalone formal core technical report
  (`docs/ocannl-formal-core-technical-report.md` / `.latex`) covering the core
  shape/projection inference proof effort: dimensions, rows, broadcasting,
  flat row equality, solving, closing, and projection inference.
- Added `docs/shape-constraint-generation.md`, documenting how `tensor/shape.ml`
  generates the core constraints and projection metadata used by inference.
- Added regression coverage for shape-inference counterexamples, closing order,
  and row-rank-cycle behavior used while validating the formalization.

### Changed

- `test/training/bigram_mlp.ml` renamed to `test/training/mlp_names.ml` and
  rewritten as a true multi-character-context Bengio MLP. The old file's name
  misrepresented its architecture (bigram-width input but "MLP" label); the
  new file is the makemore Part 2 example (see `docs/makemore_tutorial.md`).
- Default `%op` parameter initialization now uses centered, scaled `uniform1` over
  `[-0.25, 0.25)` while preserving `uniform1`'s flexible shape behavior.
- Parser updated to allow n-ary einsum specs (e.g., `a;b;c;d=>result`)
- Concat symbols are now grouped into connected components for iteration using union-find
- Product space and product iterators now use list arrays to handle concatenated dimensions
- Moved `datasets/` to separate `dataprep` package
- Relaxed the required `ocannl_` prefix on commandline arguments; config keys are now
  validated, and `ocannl_config.example` was renamed to `ocannl_config.reference`
  (gh-ocannl-409).
- Renamed routine/kernel parameters from `param`/`params` to `kparam`/`kparams`
  (gh-ocannl-356).
- Extended the identifier blacklist with C keywords, primitive-operator names, and
  backend-specific reserved words (gh-ocannl-383); `debug_name` now collapses
  consecutive identical label components, e.g. `ident` ×3 → `ident3` (gh-ocannl-281).
- Removed remaining unnecessary buffer zeroing-out in backend code (gh-ocannl-382).
- Upgraded slipshow presentation rendering to v0.11.0, with Mermaid diagrams
  (gh-ocannl-425).
- Heavy training integration tests are now gated behind Dune's `slow` alias, while
  backend-divergent goldens and CUDA/Metal generated-source expectations were normalized
  for release testing.
- Documentation for the formal core now uses the direct row-subtyping/refinement
  presentation consistently, including the closed-row equality and dimension-closing
  policy clarifications made while preparing the workshop article.
- **Breaking:** `Backend.device_to_device` now returns `context routine option` instead of `bool`.
  Instead of scheduling the copy as a side effect, it builds a transfer *routine*: callers run
  `r.schedule` (or link a consumer against `r.context`). `None` replaces the old `false` ("nothing
  to transfer": node absent from `src`; or, for `into_merge_buffer:No`, node absent from `dst` or
  identical source/destination buffers). The transfer routine's context records the produced
  merge-buffer node in the new `Backend_intf.context.merge_buffer_node` field, so that linking a
  consumer of the merge buffer against it statically verifies the node *at link time* (raising
  `Utils.User_error` from `link`/`link_batch` on a mismatch), "in the right direction" — transfer
  -> consumer (gh-ocannl-288). The runtime `check_merge_buffer` check is kept as a defensive
  backstop.

### Fixed

- Detect rank cycles among row variables during shape inference (gh-ocannl-247).
- CUDA `Where` expression codegen now parenthesizes ternaries correctly, and
  `Uint32`/`Uint64` to `uint4x32` PRNG-counter conversions spread bits rather than
  collapsing entropy.
- Prohibit `~logic:"@"` (`Compose`) with `/` and `**` in the `%cd` extension, and fixed
  ternary `~logic` being mapped to `compose_type` instead of `ternary_type`
  (gh-ocannl-192).
- C-syntax tracing `printf` statements no longer produce bad line breaks / indentation
  (gh-ocannl-179).
- Removed a duplicate `fsm_transformer` test stanza in `test/training/dune`
  that tripped `dune build @check` with `Executable "fsm_transformer" appears
  for the second time in this directory`.
- Missing on-device margin initialization for `Fetch` cases
- `invalid_vars` computation now uses correct four-quantifier logic
- Zero-dimension components filtered out in `s_dim_one` substitution
- Single remaining Concat component properly closed in `close_dim_terminal`
- Inequality constraints preserved for Concat with `invalid_vars`
- Dimension 0 (instead of 1) guessed for `invalid_vars` in all guessing locations
- Concat lowering for unit dims
- Concat index probing guarded against unknown projections
- Concat lowering resolves indices with cumulative offsets correctly
- `d=1` handling in Concat projection components

## [0.6.3] -- 2025-12-19

### Added

- Neutral element tracking during shape inference for proper padding reset
- `use_padding` syntax in einsum notation (replacing global flag)
- Circle counting dataset and MLP training test
- Cross-entropy loss and `one_hot_of_int_list` helper for classification tasks
- `out_channels` parameter to `conv2d` for explicit channel specification
- Projection slot detection by naming convention in `%cd` syntax extension
- Configurable scaling to `kaiming` and `xavier` initialization functions
- New documentation: `tensors_and_contexts.md`, affine indexing for convolutions
- Documentation for `op_fun` and `param_op_fun` types, roots, embedded nodes, and params concepts

### Changed

- Padding is now reset by tracking neutral elements through shape inference
- Changed default random initialization to `uniform1`, which doesn't impose shape constraints
- Refactored `vbs` from Map to list for order-preserving let bindings in syntax extensions
- Infer the shape of inline definitions assigned a slot for `%cd` expressions with `projections` in scope

### Fixed

- Gracefully disable inlining for convolution patterns
- Don't propagate padding across operations, even if the same tensor participates in them
- Padding margin initialization for tensors with multiple operations
- Padding initialization bug for max-pool operations
- `uniform1` periodicity by spreading bits in `*_to_uint4x32` conversions
- `tropical` (max-reduce) backprop to use input-shaped condition tensors
- `tropical` g2 gradient by using correct projection for kernel gradients
- Kernel extent calculation to depend on kernel size parity
- Shape inference for `Total_elems` constraints with `Strided_var` numerators
- `compute_row_product` to return `None` for unresolved variables
- Deferred dim variable guessing to Stage 5 for `Total_elems` propagation
- Padding offset application during lowering for correct buffer indexing
- Intermediate grads from `kaiming`, `xavier` appearing in `zero_grads`
- Random seed initialization missing in transformer test

## [0.6.2] -- 2025-11-27

### Added

- Normal distribution random number generation
- `%%extend_dsls` syntax extension for extending DSL modules
- `interleave` operation in DSL modules
- `Defined_by_cd_logic` shape inference specification for explicit shape logic in forward code
- Menhir-based einsum parser replacing Angstrom for better maintainability
- Name clash detection for inline definitions and variable captures in syntax extensions
- `is_param` flag in shape inference for improved parameter-related error messages
- Teacher forcing support in transformer implementation
- Heuristics for "missing hidden dimensions" error messages with row variables
- `Tree_map` persistent map utility with exposed tree structure in sexp serialization

### Changed

- Migrated shape environment to use `Utils.Tree_map` for ppx_minidebug v3 full-scale debugging
- Replaced explicit non-iteration tracking with improved projection constraints derivation
- Support for offset-only affine expressions in shape inference
- Renamed optional dimension variable parameter from `label` to `name`
- Row IDs replaced with provenance tracking (`Row.id` → `Row.prov`) supporting deduplication
- Tensor labels interface improved: per-operation `op_label` string with `label` list as trailing parameter
- Adapted to ppx_minidebug renaming (`entry_id` → `scope_id`)
- Prefixed block names in `lib/nn_blocks.ml` for better namespace management
- Tests reorganized: more einsum-related tests moved to `test/einsum/`

### Fixed

- Normal distribution test determinism across different machines
- Convolution/affine indexing shape inference offset adjustment by strides
- Parameter gradients not embedded after params moved earlier in processing
- Einsum parser handling of missing convolution and single-character cases
- Shape inference for `Conv_input` additional cases
- Incremental construction of tensors in `Tensor.op`
- Attention masks now have empty output dimensions for proper broadcasting to multihead attentions
- LUB (Least Upper Bound) computation in `dim_ineq`
- Axis labels distinguished from dimension units (labels) in `shape_spec_to_dims_bio`
- Shape inference for dim-1 with labels treated same as dim>1 (only dim-1 without label is different)
- Shape specification requiring LUB incorporation for non-terminal shapes
- Missing CUDA backend cases and NVRTC compatibility
- Premature guessing of dim variables as dim-1 when participating in `Total_elems` constraints
- Generic constraints ignored for unused tensors
- Missing propagation when `set_dim` happened before parsing the spec
- Guard `axis_keys_to_idcs` from un-inferred shapes
- More informative error messages for parameter shape errors
- Crash on repeated variable capture in syntax extensions
- Additional syntax support for binary einsum operators

## [0.6.1] -- 2025-09-12

### Added

- Record-based syntax for inline tensor definitions in `%op` and `%cd` expressions
- `uniform1` variants for non-vectorized random number generation (`Uint4x32_to_prec_uniform1`)
- Support for uint32/uint64 precisions and `big_models` flag for indexing arithmetic
- Created docs landing page with automatic publishing action
- Group Relative Policy Optimization (GRPO) documentation in RL slides
- Counter-based randomness with lightweight (2-round) Threefry variant as default
- Claude GitHub Actions for automated code review and PR assistance
- Heterogeneous precision support for primitive operations
- Both zero-initialized and undefined-initialization buffer creation options
- More output options for `ocannl_read_config` utility
- Added comprehensive RL/REINFORCE tutorial slides with concrete examples
- Added clear explanations of slipshow navigation semantics to CLAUDE.md
- Transformer architecture support with multi-head attention, layer normalization, and positional encodings
- CNN building blocks: conv2d, pooling operations (max/avg), and comprehensive migration guide
- Context API as simplified backend interface replacing stream-based parallelism
- Shape constraint provenance tracking for dramatically improved error messages with origins
- Dimension capture and equality constraints in einsum specifications via `set_dim` and `set_equal`
- New einsum operations: `einmax1` (unary max-reduce) and `tropical` (max-reduce with add)
- `%oc` anti-quotation syntax for improved OCaml integration in ppx extensions
- Tensor initialization operation with configurable strategies
- `offsets` convenience operation for index generation
- Comprehensive migration guide for PyTorch/TensorFlow users
- Shapes and einsum tutorial slides with slipshow presentation format
- Configurable limit on shape constraint provenance tracking
- Origin tracking in shape error messages

### Changed

- Major fix to tensor initialization handling with uniform generation across TDSL, NTDSL, PDSL
- `%op` scope delimiting changed from `~config` to unit parameters for cleaner syntax
- Renamed `zero_initialized` to `zero_initialized_by_code` for clarity
- Split Threefry4x32 into crypto (20-round) and light (2-round) variants
- Changed `Operation.range` semantics to match Python's `range` function
- Improved precision handling in low-level operations with bidirectional inference
- Enhanced record syntax with field shortcuts (`o` → `output_dims`, `i` → `input_dims`, `b` → `batch_dims`)
- More precise and thus more lenient rootness checks in `Tensor.consume_` functions
- Generalized `guess_output_nodes` to `collect_nodes_guess_output` for more reuse
- Converted documentation slides to use slipshow for better updatability and navigation
- Updated CLAUDE.md with record syntax documentation and testing guidelines
- Major reorganization: moved tensor-related modules to dedicated `tensor/` directory
- Renamed einsum operator `*+` to `+*` for better consistency
- Refactored DSL modules into `Operation.DSL_modules` for cleaner API
- Removed stream-based parallelism in favor of simpler Context API
- Improved `%op` and `%cd` syntax extensions with better function application handling
- Enhanced shape inference with proper dimension staging (no closing at stage 2)
- Migrated documentation to `docs/` directory with pandoc rendering support
- Improved `.cd` file generation with clearer rendering of special operations
- Updated ppx_minidebug integration with log pruning for better performance

### Fixed

- Unnecessary dune rules triggering issue
- Precision handling for `Uint4x32_to_prec_uniform1` in scalar computations
- CUDA, Metal, and C backend fixes for various precision and initialization issues
- Zero-dimensional Bigarray indexing
- Test dependency on `OCANNL_BACKEND` environment variable
- Build setup for `ocannl_read_config` utility needed for tests
- Missing package dependencies and assignments in dune configuration
- Critical transformer bugs: mask handling, attention dimension specifications, position encodings
- C backend INFINITY macro usage (was using invalid inf literals)
- Shape inference bugs with dimension closing and constraint generation
- Dropout pseudo-random number splitting
- Layer normalization implementation in `nn_blocks.ml`
- Dimension inference for attention layers with hidden dimensions
- Pooling operations projection inference
- Division simplification for integer precision
- Various syntax extension edge cases and error handling

## [0.6.0] -- 2025-08-19

### Added

- Support for Brain float aka. bfloat16 aka. BF16, and for FP8.
- Support for convolution via affine indexing expressions in: projections, einsum notation, shape inference.
- MNIST and CIFAR10 datasets (borrowed from Raven).
- Names dataset with bigram use-case helper.
- Half-moons synthetic dataset.
- New precision `Uint4x32` that piggybacks on the `Complex.t` type for the `Bigarray` backing.
- New precision `Int64` for integer operations.
- New operation `Threefry4x32`, which is unusually and hopefully uniquely coarse-grained (requiring nontrivial implementation code for each backend that should conform to a common algorithm).
  - This way we avoid introducing multiple operations on bits.
- Support of counter-based randomness via the `Threefry4x32` operation and random seed tracking.
  - The cascade of splits uses the Tnode id, the train step and the tensor cell position.
- Added a new operation `Uint4x32_to_prec_uniform` that converts the 128-bit random values to floating point uniform distributions efficiently.
- Vector operations support with `Set_from_vec` in low-level IR for efficient vectorized assignments.
- Added a field `params` to `Tensor.t` since we need to track parameters to properly initialize computations (see below).
- `Embed_self_id` operation for positional embeddings.
- Bidirectional precision inference (both top-down and bottom-up).
- Enhanced `%cd` syntax with support for `.forward`, `.backprop`, `.zero_grads` and automatic comment generation.
- Inline tensor declarations in `%cd` syntax for standalone expressions.
- `Train.init_params` for streamlined parameter initialization.
- Better configurability with `inline_complex_computations` setting.

### Changed

- Removed the ndarray initialization logic. Some of its functionality is now incorporated into `fetch_op`.
- Refactored `init_op` and the badly named `global_identifier` from `ops.ml` into `dedicated_access` in `low_level.ml` and a bigger `fetch_op` in `assignments.ml` (more meaningful file locations).
  - Also renamed the badly named `Get_global` to `Access`.
- Initialization now needs to be handled via running the corresponding code explicitly. In particular `Tensor.init_params` will run the forward code of tensors from the `params` field.
- Virtual nodes and inlining now also work across routines. This required changing the API to pass the `optimize_ctx` optimization context.
- Made ppx_minidebug logging per-file opt-in at compile time for better control.
- Refactored Tensor API to reduce boilerplate and share parameter signatures.
- Renamed `float_t` to `scalar_t` throughout the codebase for consistency.
- Migrated from heap-local allocation to on-stack allocation by default.
- Improved shape inference with better Total_elems constraint handling and LUB (Least Upper Bound) support.
- Enhanced projections inference with better slot selection heuristics.
- More defensive handling of empty dimensions and zero-dimension scalars.

### Fixed

- Memory leak in builtins.c.
- Context handling for constants initialized on devices.
- Zero-initialization that wasn't being performed on Linux (MacOS zero-initializes by default).
- Surjectivity and bijectivity checking in indexing operations.
- CUDA backend regressions and missing constructs.
- Duplicate Shape_rows constraints elimination.
- Precision inference issues with premature forcing.
- Bus error on large datasets.
- Session-level bugs that appeared only in specific backends.
- Identifier generation to not start with digits.
- Host-device synchronization issues with `devices_not_lagging_host` semantics.
- Shape inference corner cases with Total_elems and row constraints.
- Various issues with convolution and strided iteration support.
- Moved away from using statically loaded builtins.c from routines (kernels), all backends now prepend their builtins textually.
- Emulating _Float16 aka. half on systems with C compilers that don't support it.

## [0.5.3] -- 2025-05-24

### Added

- The Metal framework backend (Apple Silicon).
- Setting `debug_log_to_stream_files` to neatly keep logs from routine execution in their separate files.
- Settings `clean_up_artifacts_on_startup`, `prefer_backend_uniformity`.
- Tools directory and the `minised` tool: regexp replacement file rewrite.
- Directory arrayjit/bin and executable `read_config` for extracting OCANNL configuration into txt files.

### Changed

- Removed `initialize` and `is_initialized` from the backend API; instead, backends should be initialized on functor application. The functors now take `config` as argument.
- More descriptive identifier names in C-syntax code in case of name conflicts.
- Changed the backend config name `cc` to `multicore_cc` for consistency.
- Migrated out of `Stdlib.Format` to `PPrint` for all structured formatting.
- Migrated stdout capture to thread-based (domain-based actually); for Windows compatibility but also much more robust for large logs.

### Fixed

- Avoid conflicts with C math function names like `fma`.
- Satur01_gate had wrong semantics.

## [0.5.2] -- 2025-04-07

### Added

- Lots of new primitive ops:
  - Unary: Satur01 | Exp | Log | Exp2 | Log2 | Sin | Cos | Sqrt | Recip | Recip_sqrt | Neg | Tanh_approx | Not
  - Binary: Satur01_gate | Max | Min | Mod | Cmplt | Cmpeq | Cmpne
  - Ternary: Where | FMA (non-accumulating)
- Ternary tensor operations.
  - A differentiable `where` operation.
- More flexible gradient construction via the `%cd` syntax (better projections inference).
- CC backend piggy-backing on OCaml's C compiler (consistent across OSes).

### Changed

- Updated to printbox 0.12, with upstreamed graphing.
- `-pthread` -> `-lpthread` in `c_library_flags` in `dune` files.
- Removed Numpy support for easier compatibility on native Windows.
- Unary (primitive) ops and relu are now named, not operator syntax.
- Refactored `%cd` parsing of primitive ops.
- `%cd` and `%op` support both curried and uncurried operator application syntax.
- Updated to ppx_minidebug 2.2.0 with support for cross-run diffing.

### Fixed

- Numbers text rendering (consistent across OSes).
- Moved closing row variables to stage 3, because stage 2 may need to process inequalities generating more LUBs.
- Don't unnecessarily prevent bytecode-only build targets.

## [0.5.1] -- 2025-01-01

## Added

- Automatic transfers to host from the context that most recently updated a node.
- Automatic transfers of routine's inputs from host to routine's context if the host array modification was not yet transfered.

## Fixed

- Added `#` as alternative to `~~` for comment lines in `ocannl_config` files, and fixed a bug in their parsing.

## [0.5.0] -- 2024-12-18

### Added

- Interface files for `Backends` and `Low_level`.
- Fixed #245: tracking of used memory. But there's room for improvement.
- Stream-to-stream synchronization functionality, with lazy per-tensor-node synchronization.

### Changed

- Migrated to cudajit 0.6.1.
- Verifying that code is linked with the right contexts, by tracking `embedded_nodes` with assignments.
- Renaming: (virtual) `device` -> `stream`, `physical_device` -> `device`.
- New files: split out `backend_intf.ml`, `backend_impl.ml`, `schedulers.ml` from `backends.ml`; moved `Tnode.task` to `task.ml`; renamed `backend_utils.ml` to `c_syntax.ml`.
- Removed half-static verification of merge buffer nodes inside `device_to_device`.
- Fixed #286: cross-stream-sharing incorporated into `Tnode.memory_mode`.
- Moved the multicore backend from a `device = stream` model to a single device model.
- Got rid of `unsafe_cleanup`.
- Rename `subordinal` to `stream_id`.
- Removed dependency on `core`, broke up dependency on `ppx_jane`.
- Huge refactoring of backend internal interfaces and API (not repeating same code).
- Built per-tensor-node stream-to-stream synchronization into copying functions.
- Re-introduced whole-device blocking synchronization, which now is just a slight optimization as it also cleans up event book-keeping.
- Simplifications: no more explicit compilation postponing; no more hard-coded pointers (all non-local arrays are passed by parameter).
- Fresh backends are now fresh modules to structurally prevent any potential cache leaking.

### Fixed

- Validating merge nodes for the CUDA backend.
- Checking `is_released` on weak array retrieval.

## [0.4.1] -- 2024-09-17

### Added

- Implemented the previously-mocked support for half precision (FP16).
  - We work around the missing Ctypes coverage by not using `Ctypes.bigarray_start`.
  - We check FP16 constants for overflow.
  - We output half precision specific code from the CUDA backend.
- Finally proper support for mixed precision! Lazy precision defaults and delayed precision setting via `Tnode.update_prec`.
- A placeholder `nn_blocks.ml` hinting at an intended design pattern for model components.
- A memory model for the multiple virtual devices per physical device setup, implemented in the CUDA backend. It fixes the CUDA backend behavior in the data parallelism benchmark.
- Slides for the Fun OCaml meetup: [docs/Fun OCaml](docs/OCANNL-slides-basics_backprop_training_loop_codegen.pdf).
- New syntax: inline tensor declarations with a literal float as initial value.

### Changed

- Removed the `pipes_cc, pipes_gccjit` backends (`Pipes_multicore_backend`) -- I had fixed `Pipes_multicore_backend` by using the `poll` library instead of `Unix.select`, but it turns out to be very very slow.
- Changed the `%cd` block comment syntax `~~` to allow detailed structuring. Rewrote `Train.grad_update` to use the `%cd` syntax.
- Made `Train.sgd_one` slightly more thrifty: `p =- learning_rate *. sgd_delta` --> `p =- learning_rate * sgd_delta ~logic:"."` without the inline tensor expression.

### Fixed

- Log levels related de-confusion:
  - Critical bug: logging of computation traces was not properly converted to ppx_minidebug 2.0.
  - Properly restore `log_level` and inform about its setting.
  - By default do not log from tests.
  - `debug_log_from_routines` should only happen when `log_level > 1`.
- Bugs in `Multicore_backend`: `await` was not checking queue emptiness, `worker`'s `Condition.broadcast` was non-atomically guarded (doesn't need to be), possible deadloop due to the lockfree queue -- now replaced with `saturn_lockfree`.
- Reduced busy-waiting inside `c_compile_and_load`, propagating compilation errors now instead of infinite loop on error.
- Fixed loss of significant digits for small numbers when outputting files.
- Added missing mixed-precision conversions in the `C_syntax` backend builder.
- Restored the functionality of debug logging from the cuda backend.
- Always reinitialize global state at the beginning of `let%expect_test`, to make them more deterministic.

## [0.4.0] -- 2024-09-04

### Added

- A new backend "cc": C based on a configurable C compiler command, defaulting to `cc`.
- Merge buffers representational abstraction (one per virtual device):
  - backends just need to support device-to-device transfers,
  - merging gets implemented in "user space".
- CUDA streaming multiprocessor parallelism via streams <-> virtual devices.
- Support for `cuda-gdb` and `compute-sanitizer` (pass the right arguments to cudajit).
- Inline declarations for (non-differentiable) tensors in the `%cd` syntax.
- A minimal wrapper `Sync_backend` creating CPU backends with a single device only, where all calls are synchronous. (It's a baseline and helps debugging.)
- In progress: proper (condition variables based) scheduler. The legacy scheduler (pipes based) kept for now as baseline and to help debugging.
- Documentation for the syntax extensions.
- `%op` syntax: when under a `~config` parameter, refine the inline declared params' labels with `config.label`.
- `%op` syntax: incorporate the input tensor's (if any) label in the resulting tensor's label.
- Comments in config files using the line prefix `~~`.

### Changed

- Terminology in the API: Renamed almost all uses of "jit" into uses of "compile" and / or "link".
- Split the compile-to-ptx phase from the build-module and build-kernel-launcher phase.
- Migrated the CUDA backend to ppx_minidebug-based execution tracing.
- Fixes for mixed precision computations.
- Further terminology refactoring: Renamed `Low_level.compile` to `Low_level.lower`;
  - and `Low_level.compiled` to `Low_level.optimized`, making it a record.
- Further refactoring of the `Backends` API:
  - split the `device` type into virtual `device` and `physical_device`,
  - removed the direct support for `merge`, instead relying on merge buffers.
- Updated to cudajit 0.4.
- A template for C-syntax backends, refactoring CC and CUDA backends.
- Improvements to handling of tensor node labels, and to the `Tnode.debug_name` function.
- Output files generated by backends, and files generated by logging, in separate subdirectories.
- C-syntax logging: also output the pre-assignment value when logging an assignment.
- Migrated to ppx_minidebug 2.0 with the benefits it brings: no runtime passing, `Utils.settings.log_level` unified with ppx_minidebug's log levels.

### Fixed

- Allow verifying that non-embedded tensor nodes of the tensor(s) associated with a linked code are already in the context passed to `link` (resp. `link_batch`), since they won't get introduced into the context. It is the responsibility of helper functions (such as those in `Train`) to ensure the check.
- Fixed both known and newly discovered shortcomings of the syntax extensions.
- In particular, `%op` syntax: lift `~config` applications out of (tensor) functions.
- Multiple other tiny fixes.

## [0.3.3] -- 2024-04-24

### Added

- GitHub workflow for continuous integration and API docs.
- Randomness plug-ins via global config `randomness_lib`: currently only `stdlib` and `for_tests`.

### Fixed

- A bit of code rot in the Cuda backend mock `cuda_backend.missing.ml`.
- NPY: Compatibility with OCaml 5.2.0.
- Renamed the main package name from `ocannl` to `neural_nets_lib`, to prevent the opam linter from complaining about a confusing name.

## [0.3.2] -- 2024-04-22

### Added

- `let%cd _ =` (and `let%op _ =`?) do not affect root tracking (intended for adding shape constraints).
- More expressive shape constraints: allowing row variables to be sandwiched between leftmost axes `beg_dims` and rightmost axes `dims`.
- Einsum notation support for leftmost axes.

### Changed

- Cleaned up "user-facing" API by moving `IDX` and `CDSL` to `Train`, and `Tensor.O` to more precise `Operation.At`.
- Added interface `Tensor.mli` to reduce "the user learning surface".
- Improved documentation and layout of `Shape.mli`.
- A more reasonable syntax for labels specifications and einsum notation. In particular, whitespace insensitive (except whitespace not allowed inside identifiers).
- Vendored the `npy` package while we wait for a PR.

### Fixed

- Moved `cudajit` to `depopts`.
- Slice shape inference is now complete, by using leftmost axes `beg_dims` in constraints.

## [0.3.1] -- 2024-04-15

### Added

- Tensor parameters saving and restoring, Ndarray saving and restoring.
- An operation `outer_sum`: like `einsum` but simpler, addition everywhere.

### Changed

- Tweaks to make the project usable as a package (external library).
- Sanitizing code inclusion via code roots management: `Tensor.consume_forward_code` and `consume_backprop_code`, (optionally but by default) used from `Train`.

### Fixed

- Shape inference in presence of non-0 fixed indexing inside einsums was broken (because actually not implemented).
- Incompleteness of shape inference for slicing was leading to inferring shapes with no axes: constraint generation was intended to raise a shape error instead. Proper fix coming in 0.3.2 will make slice shape inference complete.

## [0.3.0] -- 2024-03-31

Major rewrite. Abandoning the design choices of 0.1 and 0.2.

### Added

- Optionally, inferring or checking tensor (batch) sizes from data (e.g. file) sizes.
- Static indexing. A "slice" operator to select individual batches.
- Established the backends API with first-class modules.
- The `Train` module as an optimization "frontend".
- Parallel optimization across devices.
- Global settings configurable via config files, environment variables, and commandline flags.
- Integration of backend logging with `ppx_minidebug` (the `debug_log_from_routines` setting).

### Changed

- The Cuda backend is not supported for now. It is (optionally) buildable to reduce code rot.
- Dynamic indexing is not supported anymore (to reduce complexity). It might be reintroduced if needed.
- Factored out the `arrayjit` library / package containing compilation (former Ndarray, Node, Code).
- Renamed `Formula` -> `Tensor`
- No more "form vs. non-form" formulas / tensors.
  - Formula/tensor roots are split into forward roots and backprop roots.
- No more `%nn_rs`, `%nn_dt` syntaxes and `Synthetic` fetch primitive.
- Renamed `%nn_op` to `%op` and `%nn_cd` to `%cd`.
- Migrated `gccjit` into a separate repository.
- Migrated `cudajit` into a separate repository.
- Massive rewrite of shape inference in a declarative style.
- Generalize `zero_out` to `initialize_neutral` to prepare arbitrary accumulation operation.
- Renamed `Node` -> `Lazy_array` -> `Tnode` (tensor node).

## [0.2.1] -- 2023-07-19

### Added

- The Cuda backend.
  - The Cudajit interface based on Nvrtc and the Cuda driver API.
  - A naive `Exec_as_cuda` backend where the dedicated `Task_id` axis parallelizes over blocks, and a new dedicated `Sample_num` axis parallelizes over threads in a block.
  - When outputting debug files, stores the source `.cu` code and the assembly `.ptx` code.
  - Supports thread-only tensors, tensors with thread-local "replicated" working copies, constant tensors, and globally updated tensors.
  - The backend uses atomic adds for shared updates, and within-block synchronization to minimize update races and parameter staleness.
  - Debugging: full trace (for thread 0) by logging assignments with the assigned value and indices for the LHS tensor and the RHS tensors, the expression used to compute the assigned value, of values of subexpressions.
- Cuda FFI for retrieving GPU specs and for getting and setting limits.
- `Zero_out` low-level-code primitive using `memset`.
- `Staged_compilation` low-level-code primitive: a (stateful) callback for use by backends.
- When outputting debug files, also stores the high-level code.
- Saving and restoring tensor content to `.npz` (`.npy` archive) files (untested).
- Low-level code based optimizations:
  - unrolls `ToPowOf` with integer exponent,
  - simplifies local computations that are just expressions,
  - some arithmetic simplifications.

### Changed

- Monomorphic `axis_index`, simplified the axes-related types.
- Splits `'a low_level` into monomorphic `unit_low_level` and `float_low_level`.
- Removes integer bigarray types.
- Refactors `Node` + `NodeUI` into `Ndarray` + `Node`.
- Tensor printouts include whether a tensor contains `NaN` or `infinity`.
- Simplifies the `Task_id` functionality: removes `If_task_id_is` and `Global Task_id`; emoves parallelism from `interpret_code`; removes `task_id_func` vs `unit_func` duplication.

### Fixed

- "Non-diff" code inclusion.
- Ensures unique indices/symbols also for the `task_id` and `sample_num` bindings.
- Removes endlines from `PrintBox_utils` benchmark tables cells.

## [0.2.0] -- 2023-06-03

### Added

- The Gccjit backend operates using "on device" copies of tensors, where the "device memory" is the stack of the C function. This is intended to improve cache locality and reduce cache contention.
  - Three / four synchronization heuristics:
    - "parallel": a slice of the tensor is copied host-to-device at the beginning and device-to-host at the end, without interference because each task has a different slice.
    - "update on host": the tensor is copied host-to-device at the beginning; each write is an update, it reads the old value from host to update it on the host. Thus each write is a synchronization point.
    - "replicated": the tensor is copied host-to-device at the beginning; only task 0 copies device-to-host.
    - "device-only": no copying to/from host.
- On-device-only tensors that are not materialized on the OCaml side.
- A new category of axis dimensions is introduced: `Frozen`. It is analogous to the `Parallel` axis category in that a single task execution / "device call" only processes a 1D slice of the axis.
  - Currently, for tensors processed in parallel, we only support processing of a contiguous tensor slice (copied "to device" using `memcpy`).
- A new syntax `%nn_rs` ("postprocess results" variant of `%nn_dt`) for computations that should happen at the end of task execution / refresh step. It's meant to prepare the data to be copied back to the host.

### Changed

- Got rid of backend-agnostic synchronization. It was not worth the complexity / implementation effort at this point.
  - Keeping the `Rebalance` constructor around, but it is not playing any role.
- Got rid of `debug_virtual_nodes`, was tricky to maintain.
- Dynamic indexing now skips over parallel axes: when there is a `Parallel` axis on the left, it is preserved in the resulting tensor (slice), and the next-right axis is indexed into instead.
  - Removed the "indexing axes from-right" functionality for now (fails as not implemented).
- Dynamic indexing now can produce virtual nodes.

### Fixed

- Dynamic indexing fixes.

## [0.1.2] -- 2023-05-12

### Added

- Thread-local parameter `task_id` for automated iteration over a dimension `Parallel`.
  - This implements multicore SGD.
  - Rebalancing of computations that don't use `Parallel`, and synchronization in the `Gccjit` backend, are left as future work.
  - Already provides significant speedups in the interpreter (6-7x for me), but that's a moot point.
  - Giving up further work this approach for now, because the bottleneck is the memory access with `Gccjit`.
  - Keeping the new representation capability around, maybe it will be a stepping stone to other things.
- Monolithic step update with "macrobatch" (multiple steps within one backend call).

### Changed

- Streamlined the source code, e.g. removed the `OCaml` backend.
- Better syntax for `%nn_dt` and `%nn_op` shape specification, allows identifiers.
- Improved virtual node and scalar constant inlining.
- Better debugging, e.g. an option to "trace" `Gccjit` execution by printing the comments.

## [0.1.1] -- 2023-05-06

### Added

- An _inline constants_ optimization that compile-time computes scalar constant subexpressions and inlines the values.

### Changed

- Improved debuggability.

### Fixed

- A last-minute breaking bug (would be nice to have a pre-release or a pre-publish hook to run tests!).
- The virtual nodes optimization is more robust, correct even with aggressive inlining settings (e.g. escaping variables check).

## [0.1.0] -- 2023-05-04

### Added

- The first changes-tracking release. Earlier development history is still somewhat documented via closed issues.
- Supports single and double precision floats, more precisions in the future.
- Generates a monolithic step update routine executed by `refresh_session ()`, but can generate arbitrary additional routines at arbitrary times to be executed at arbitrary other times within a session.
- An `Interpreter` backend that can for example log all individual tensor modifications.
- A `Gccjit` backend that can sometimes be 400x faster than the `Interpreter` backend (without any debug work/output).
- A _virtual nodes (tensors)_ optimization that inlines computation of a cell in lieu of tensor accesses, can sometimes reduce memory consumption by 1/3.
