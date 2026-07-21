## [0.9] -- Unreleased

> Release note: theme — program search and optimization. Since 0.8, the schedule system has
> gained hardware tensor-core paths, native affine legality and cost analyses, symbolic runtime
> extents, convolution sketch families, and a liveness-based memory planner. The remaining 0.9
> intended scope is summarized in [ROADMAP.md](ROADMAP.md), although its statement that full
> beam search remains future work is stale: multi-round search over schedule compositions is
> already implemented. End-to-end benchmark validation and cost-model-guided default/beam
> selection are not yet complete.

### Added

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
  envelopes. Selection of untuned defaults and beam pre-filtering remain follow-up work.
- **Swizzled shared staging**: `Stage ~swizzle:true` marks XOR-swizzled shared-memory tiles, with
  validation and explicit intrinsic-decline behavior until swizzle-aware fragment loads land.

### Changed

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
