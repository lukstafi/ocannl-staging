# Cross-framework benchmark suite: OCANNL vs PyTorch vs tinygrad

Apples-to-apples training-step benchmarks built on two pillars:

1. **Identical math, verified.** Every runner loads the same initial weights and data from a
   shared safetensors fixture and trains the same model (n-layer relu MLP, softmax
   cross-entropy, plain SGD). A **parity gate** compares the loss trajectory of the first
   `parity_steps` steps against the PyTorch CPU reference; timing numbers are only comparable
   when the gate passes (fp32 tolerance, `PARITY_TOL` in `orchestrate.py`).
2. **Identical measurement.** Each runner syncs the device around timed regions, does
   `warmup_steps` untimed steps, then reports per-step wall times two ways: `step_ms`
   percentiles (sync after every step) and `queued_step_ms` (enqueue `timed_steps` steps, one
   final sync). One-time cost (graph build / codegen / JIT capture) is reported separately as
   `compile_s`, never amortized into step time.

The parity gate doubles as a cross-framework correctness oracle for OCANNL: on its first run
it caught two real backward-pass optimizer bugs (wrong gradients with a correct forward), both
since fixed — CSE alpha-equivalence renaming free loop symbols, and the simplifier's
nested-division rewrite; regression test `test/training/virtual_grads_parity.ml`.

## Workloads

- **mlp_small / mlp_wide** (`model: mlp`, training): n-layer relu MLPs; overhead- vs
  GEMM-dominated.
- **lenet** (`model: conv`, training): LeNet-5 with *valid* convolutions on random 32×32×1
  images, built from the idiomatic nn_blocks pieces (`conv2d ~use_padding:false`,
  `max_pool2d`, `mlp_layer`). Fixture weights are injected into the block-created inline
  params by debug-name token matching (`Bench_harness.inject`). Conv biases are the
  conventional per-channel `[oc]` (the `conv2d` block pins its inline bias to the channel
  row); the Python runners pass them as the conv's own bias argument.
- **cifar_conv** (`model: conv`, training): a cifar-scale two-conv classifier — 3-channel
  44×44 images, *valid* 5×5 convs (the classic LeNet kernels) into 32 then 64 channels
  (gh-ocannl-500). Both conv GEMM rows are 8-row-block eligible: `conv1`'s output row is 40
  and `conv2`'s is 16; `conv2` additionally has out-channels 64 and reduction 32, so its
  row/oc/red are all multiples of the GPU intrinsic 8×8×8 tile and the GPU row-block staged
  leg is proposable alongside the CPU cache-panel leg. The 5×5/44 geometry is deliberate:
  with valid 3×3 convs and 2×2 pooling the deep conv's row is always ≡ 2 (mod 4), never
  divisible by 8. Same builder as `lenet` (channel count, kernel, padding, and input
  channels are all spec-driven).
- **cifar_stride** (`model: conv`, training): the stride-2-stem sibling of `cifar_conv`
  (gh-ocannl-502): 3-channel 51×51 images, `conv1` a *valid* 5×5 conv at **stride 2** (the
  strided downsampling site the compacting Stage targets), `conv2` a valid 5×5 at stride 1,
  into 32 then 64 channels. The geometry keeps both conv GEMM rows multiples of 8 (24 and
  8), so once the seeding wave admits compacting-eligible strided sites the same
  blocked/staged legs are proposable here as on `cifar_conv`. Until then the strided conv
  exercises only the reorder-serial and default-fissioned paths — recorded as the baseline
  the compacting-Stage seeding is measured against. Strides are spec-driven
  (`stride1`/`stride2`, default 1, valid-only) through the same builder as `lenet`.
- **gpt2_mini** (`model: gpt`, `mode: infer`): pre-LN GPT-2-style decoder (4 layers, d=256,
  8 heads, seq 128, vocab 1024, tanh-gelu, learned positional embeddings, tied lm_head,
  causal mask filled with `-inf` since gh-ocannl-548; the Python runners keep `-1e9`, which
  is the same number after `exp` at every precision they run), forward-only. The parity
  metric is softmax-CE of the
  logits against fixture target ids, recorded per batch with no updates; the report shows
  tokens/s. Token embedding uses the logical one-hot gather (gh-343); LayerNorm is the
  idiomatic `Nn_blocks.layer_norm` (gammas/betas injected by name like the attention
  weights).
- **gpt2_mini_train** (`model: gpt`, `mode: train`): the same architecture and fixture
  layouts trained with plain SGD (gh-ocannl-551) — every weight (including `wte`, tied to
  the lm_head, and the positional table `wpe`) is a parameter, and the step is backprop plus
  `p -= lr * grad` in all three frameworks. This is the matmul-dominated *training* cell:
  the mixed-precision recipe (master weights, storage policy, f16 dynamic loss scaling) and
  the task-5 gate-cost legs need an optimizer, so on `gpt2_mini` they do not exist at all —
  only here. Fewer steps than the inference workload, since a step is ~3x the work. In
  OCANNL a parameter has no batch axes, so the trained `wpe` is a `[seq]x[d_model]`
  output-axis table added by an einsum that places its seq axis onto the sequence batch axis
  (inference keeps the plain broadcast add over a `[seq]`-batched constant).

## Layout

- `workloads/*.json` — workload specs; `gen_fixtures.py` generates
  `fixtures/<name>.safetensors` with initial weights in OCANNL's axis conventions (output
  axes then input axes; channels-last images — layouts documented per model in the
  generator), dataset, and all hyperparameters in the safetensors `__metadata__` map, so
  fixtures are self-describing and runners need only the fixture path.
- `runners/ocannl/bench_{mlp,conv,gpt}.ml` + `bench_harness.ml` — OCANNL runners
  (`dune build benchmarks/runners/ocannl/bench_mlp.exe` etc.). Env: `BENCH_FIXTURE` (path),
  `BENCH_TUNE=1` (`Train.tune_placements`: autotunes both the default placements graph and
  the materialize-all graph, keeping the measured winner), `BENCH_MATERIALIZE=1` (materialize
  intermediates without tuning), `BENCH_PRECISION=bf16|f16` (mlp and gpt, gh-ocannl-492);
  backend via the usual `--ocannl_backend=cc|metal|cuda|hip`. `bench_gpt` dispatches its step
  shape on the fixture's `mode`, like the Python runners: forward-only for `gpt2_mini`,
  backprop plus SGD for `gpt2_mini_train`.

  **Which precision legs a workload has is a property of the workload, not of the flag**
  (gh-ocannl-551), and the shared flag parsing lives in `Bench_harness` so a new flag cannot
  quietly apply to one runner only:

  | leg | conv | gpt (`mode: infer`) | mlp / gpt (`mode: train`) |
  |---|---|---|---|
  | `BENCH_PRECISION=bf16\|f16` | — | load-time weight conversion | master weights + cast twins |
  | `BENCH_STATIC_SCALE=1`, `BENCH_GATE_INTERVAL=N` | — | — (refused: no optimizer) | yes |

  Under a training fixture `BENCH_PRECISION` is the mixed-precision recipe: f32 master weights
  with reduced-precision cast twins, storage policy over the model body with the loss head (and,
  for gpt, the layer norms) kept f32, and for f16 dynamic loss scaling, whose per-step host-read
  inf/nan gate is included in the reported step times. Under the forward-only gpt fixture it
  re-precisions by load-time conversion instead — data-backed weights convert at wrap
  (`TDSL.wrap ~prec`), attention params through the storage policy (no optimizer, so no master
  copies and no loss scaling). Parity is gated at the looser `PARITY_TOL_PRECISION` envelopes.
  In the gpt graph the attention softmax computes in the reduced precision with the rest of the
  body, f16 included: gh-ocannl-548 made the causal mask's fill `-inf` (representable, unlike the
  `-1e9` that the fp16 constant guard rejected) and gh-ocannl-547 took reduction identities out of
  that guard's scope. Before those two the f16 gpt cells could not compile at all.

  The gh-ocannl-492 task-5 gate-cost legs (f16 and a training workload only):
  `BENCH_STATIC_SCALE=1` fixes the loss scale with no gate and no host read — the
  discriminating experiment for how much of f16's step cost is the dynamic gate;
  `BENCH_GATE_INTERVAL=N` uses the fused on-device gate with the host sampling a sticky window
  checksum every N steps (reported precision `f16-static` / `f16-gatedN`; orchestrate flag
  `--precision f16-static f16-gated16`). On a forward-only fixture the runner *refuses* them
  with a message naming why, rather than ignoring them.

  The f16 legs start from the fixture's `loss_scale` metadata (a workload spec field, defaulting
  to torch's 65536), overridable with `BENCH_LOSS_SCALE=<float>`. It matters most to the static
  leg, which never adapts: a scale whose first step already overflows diverges it outright, while
  the dynamic legs would merely spend backoff steps inside the parity window. Both current
  workloads run at the default.

  Debug helpers: `BENCH_DEBUG=1` prints param names/dims (conv/gpt) or bias gradients (mlp) and
  exits; `BENCH_NO_SGD=1` compiles the gradient update without the SGD step (mlp);
  `BENCH_NO_SLICE=1` skips `@|` batch slicing (mlp, single-batch fixture);
  `BENCH_TWIN_PLACEMENT=materialized|virtual` (mlp, reduced precision) pins the master weights'
  cast twins instead of leaving them to the virtualization heuristics
  (`Mixed_prec.Twin_materialized`). The twins' placement decides whether a tensorized candidate is
  *seeded at all* on a uniform-format backend (gh-ocannl-546): with virtual twins the matmul site
  reads f32 masters into a reduced-precision destination, a mixed triple no advertised tile
  matches, so the default-placement arm proposes zero tensorized candidates — materializing three
  small weight casts is enough to make them reachable at that arm's cost.
- `runners/pytorch/run.py` — flags: `--device cpu|mps|cuda`, `--compile` (torch.compile
  variant).
- `runners/tinygrad/run.py` — flags: `--device CPU|METAL|CUDA|AMD|CL|HIP`, `--jit 0|1`, `--beam N`
  (BEAM=N kernel search, implies jit; the search cost lands in `compile_s`).
- `orchestrate.py` — runs the matrix (dispatching the OCANNL executable on the fixture's
  `model`), enforces the parity gate, writes `results/results.jsonl` and
  `results/report.md`. Flags: `--workloads mlp_small ...`, `--tuned`, `--materialized`,
  `--precision bf16 f16 f16-static f16-gatedN`, `--nojit` (tinygrad nojit), `--torch-compile`
  (pytorch compiled variant), `--beam N`
  (tinygrad BEAM=N variant; wipe tinygrad's kernel cache for from-scratch search costs),
  `--only ocannl pytorch tinygrad`, `--skip-build`, `--no-skip-cells` (run the `SKIP_CELLS`
  entries too — each was observed pathological on a single machine/backend/OS, so use this to
  retest whether an entry still applies in your environment),
  `--gpu metal|cuda|hip|none` (the GPU column of the matrix — OCANNL backend, PyTorch device,
  tinygrad device together; defaults to metal on macOS and cuda elsewhere, `none` runs a
  CPU-only matrix). Env: `BENCH_CELL_LOG_DIR=<dir>` keeps every cell's raw combined output, one
  file per cell label — a successful cell's output is otherwise discarded, which throws away the
  candidate-level evidence a measurement sweep has to report. Combined with
  `OCANNL_AUTOTUNE_LOG=true` it makes the seeded-vs-timed mma and split-reduce counts, the
  `FAILED` blocker breakdown and the split-reduce evictions fall out of the sweep's own search
  passes instead of costing a second round of searches (it does inflate a tuned cell's reported
  `compile_s` a little; step times come from the pass-2 replay and are unaffected).

  **An OCANNL cell is a (scheduling variant, storage precision) pair** (gh-ocannl-539). The two
  are independent axes and the matrix is their product: `--tuned --precision bf16` measures
  *tuned bf16*, which on RDNA3/3.5 is the only route to a tensor-core candidate at all (WMMA has
  no f32-input shape), and which the earlier single-variant-string model could not express. In
  the report, `variant` and `precision` are separate columns and rows are ordered
  precision-major (f32 first), p50-ascending within each precision group; a cell's label
  elsewhere is `variant/precision`, abbreviated to just `variant` at f32. Budget accordingly —
  each requested precision multiplies an mlp/gpt workload's OCANNL cell count by the number of
  variants, and each tuned cell is a two-pass search. `SKIP_CELLS` entries are
  `(workload, backend, variant, precision)` with `None` meaning "at every precision", which is
  what both current scheduling-pathology entries use.

  A requested cell the workload **cannot express** — a reduced precision on the conv runner
  (no `BENCH_PRECISION` support), an f16 gate-cost leg on the forward-only `gpt2_mini` (no
  optimizer, hence no loss scale to gate) — is printed as `NOT APPLICABLE` with its reason and
  listed in a *Cells not applicable* section of the report, so a missing row is distinguishable
  from an unrun one (gh-ocannl-551). Use `gpt2_mini_train` for the gpt gate-cost row.

  With `--gpu hip`, the PyTorch/tinygrad GPU cells run only on Linux (ROCm
  PyTorch presents HIP as its `cuda` device, tinygrad as `AMD`); on Windows neither framework
  reaches an AMD GPU, so OCANNL alone populates the GPU column while the CPU parity
  reference still runs. **Under WSL** both frameworks do reach an AMD GPU, with two caveats:
  there is no `/dev/kfd`, so tinygrad's `AMD` device cannot open and orchestrate falls back to
  its `HIP` device automatically; and torch's bundled `libhsa-runtime64.so` (the KFD build) must
  be replaced with `/opt/rocm/lib/libhsa-runtime64.so.1.21.0`, with
  `/opt/rocm/lib/rocm_sysdeps/lib` on `LD_LIBRARY_PATH`. See
  [example-report.md](example-report.md) (macOS/Metal — the gh-ocannl-538 sweep: tensorized
  candidates now compile and time on Metal, and win a search arm once, but none reaches a shipping
  artifact; split reduction is worth 46-82% on the default-placement arm; f16's cost is the
  loss-scaling gate, not f16 arithmetic),
  [example-report-cuda.md](example-report-cuda.md) (Linux/CUDA),
  [report-hip.md](report-hip.md) (WSL2/HIP on gfx1151, all three frameworks — the gh-ocannl-538
  re-measurement leg, and the first report in which a rocWMMA candidate is seeded, timed and
  crowned),
  [report-gh528-hip.md](report-gh528-hip.md) (WSL2/HIP on gfx1151, the hardware validation of
  gh-ocannl-528's interior-batch `Tensorize` and gh-ocannl-481's HIP declines, plus the `gpt2_mini`
  tensor-core probe: three tensorized sites reached and verified in the emitted source, landing
  within noise of an f32 schedule with none — and the reference for two facts that outlive it,
  that gfx1151's WMMA is not exactly-rounded in any format combination, and that `taskset -c 0-15`
  on that box is 8 SMT-shared cores rather than 16 private ones),
  [report-cifar-cuda.md](report-cifar-cuda.md) (Linux/CUDA, the cifar-scale conv baseline
  for gh-ocannl-500/502 with a per-layer breakdown) and
  [report-gh537-metal.md](report-gh537-metal.md) (macOS/Metal, the paired before/after A/B of
  gh-ocannl-537's `Swap` ∘ `Split_reduce` seeding — the Metal leg of the CUDA measurement in
  `report-gh537-cuda.md`, replicating it),
  [report-gh484-cuda.md](report-gh484-cuda.md) (Linux/CUDA, the paired before/after A/B of
  gh-ocannl-484 task 3's split-reduce seeding — also the reference for why only same-session
  paired runs are trustworthy on that machine),
  [report-gh481-cuda.md](report-gh481-cuda.md) (Linux/CUDA, gh-ocannl-481's `ldmatrix` over
  swizzled staged tiles: a neutral result, and the reference for two measurement traps a tuned-cell
  A/B has to control for — the schedule disk cache silently replaying the other arm's winner, and a
  ~20-40% cell-level spread from the beam not always crowning the same family, which its own
  identical-code negative control exposes) and
  [report-gh537-cuda.md](report-gh537-cuda.md) (Linux/CUDA, its successor: the paired A/B of
  gh-ocannl-537's `Swap` ∘ `Split_reduce` seeding, which removes ~90% of the segment gh-484 was
  filed against — and the reference for why a segment share must name the placement it is a share
  of) for checked-in example output (`results/` itself is gitignored).
  [RESULTS-484-532.md](RESULTS-484-532.md) is the raw extracted dataset behind the gh-ocannl-527
  and gh-ocannl-532 subsections of `report-hip.md`'s **pre-gh-538 revision** (per-segment
  attribution, the split-reduce `op_legality` verdicts, and the nine variant cells) — kept because
  that session's logs were lost to GPU-driver crashes and reboots, so it is the only surviving
  primary source for a state the current report no longer describes.
- `runners/ocannl/bench_{gpt,conv}_diag.ml` — schedule diagnostics: print the default
  fission-pipeline segment census (launch geometry, per-nest loop extents, written nodes with
  materialization markers) for the gpt2_mini / lenet graphs, then optionally time steps
  (`BENCH_STEPS=1`) or dump tensor values (`BENCH_PROBE=1`, `BENCH_DUMP=1`; `BENCH_FWD=1`
  compiles forward-only, `BENCH_PROMOTE=0` disables fission's Local promotion in the census).
  `BENCH_SEG_TIMES=1` (**both runners**) adds per-segment (≈ per-layer) wall times: each fission
  segment is compiled as its own routine (hermetic substitution through the `lowered_transform`
  seam) and timed min-of-N with a sync per run, labeled by the nodes it writes — the per-layer
  breakdown that identifies which conv sites dominate a step, diffable across schedule
  changes (the acceptance instrument for the gh-500 blocking decision). On `bench_gpt_diag` the
  same instrument attributes the transformer blocks: it is what showed gpt2_mini's step to be
  concentrated in the per-layer FFN projections and the lm_head rather than spread evenly, despite
  every segment sharing identical launch geometry (gh-ocannl-531).
  `BENCH_SR_SITES=1` (`bench_conv_diag`) prints what `Autotune.split_reduce_sites` proposes on the
  same graph — the gh-ocannl-484 task-3 seeding can only reach the accumulations listed there, so
  it is the companion to the census above when asking why a seeded split-reduce family did or did
  not move a workload (it is what showed the detector finding only the classifier head on all three
  conv benchmarks; see [report-gh484-cuda.md](report-gh484-cuda.md)). It then prints, for every
  low-output write, the enclosing loop nest and the `Ir.Schedule.op_legality` verdict of splitting
  each enclosing serial loop — the site listing says a segment was not proposed, these verdicts say
  which rule rejected it (the extent floor and the `Split_reduce` recognizer's own preconditions are
  indistinguishable from the listing alone). That is what attributed lenet's `bias_conv1.grad`
  segment; see [report-hip.md](report-hip.md). Since gh-ocannl-537 a listed site also names the
  enabling loop interchange it was reached through (`via N swaps: inner^outer`, absent when the site
  was splittable as lowered), and a rejection verdict is tagged `[hoistable: …]` when the
  interchange would remove it — so the listing distinguishes "not reachable" from "reached by
  composition". Compile-time only, off the timing path, so it composes with `BENCH_SEG_TIMES=1` and
  with `BENCH_MATERIALIZE=1`.
- `runners/ocannl/bench_metal_bug.ml` — standalone (no OCANNL) repro of an Apple Metal
  shader-compiler miscompilation: a serial `acc[0] = acc[0] + f(i)` loop over
  slot-table-derived pool pointers keeps only the last iteration's contribution. OCANNL works
  around it via `volatile_scalar_rmw` in `arrayjit/lib/c_syntax.ml`; this repro documents the
  raw bug (prints the wrong value as long as the toolchain is affected) and
  `test/operations/scalar_rmw_accumulation.ml` guards the workaround end to end.

## Setup

```bash
python3 -m venv benchmarks/.venv
benchmarks/.venv/bin/pip install numpy safetensors torch tinygrad
benchmarks/.venv/bin/python benchmarks/gen_fixtures.py
benchmarks/.venv/bin/python benchmarks/orchestrate.py
```

tinygrad's CPU device JIT-compiles kernels with `clang`; on a machine without clang, point
`CC` at a substitute (a `zig cc` wrapper script from `pip install ziglang` works — translate
`--target=x86_64-none-unknown-elf` to `--target=x86_64-freestanding-none` and add `-g0`).
tinygrad's CUDA device compiles through the system nvrtc; when the CUDA toolkit is newer
than the driver (`CUDA_ERROR_UNSUPPORTED_PTX_VERSION` at module load), run it with
`LD_LIBRARY_PATH` pointing at torch's bundled nvrtc
(`.venv/lib/python*/site-packages/nvidia/cu*/lib`) and wipe `~/.cache/tinygrad` once
(compiled PTX is cached by source hash, so a stale-toolkit artifact survives the switch).

## Methodology notes / fairness pitfalls

- Losses are recorded per step *before* that step's SGD update (forward runs first in every
  framework's step). The first step doubles as the compile probe in the Python runners; for
  OCANNL, `compile_s` wraps `Context.compile` (or `Autotune.tune`).
- SGD is plain `p -= lr * grad` in all three (OCANNL's `Train.sgd_one` and tinygrad's
  `nn.optim.SGD` share the same reference semantics). Don't switch parity workloads to Adam:
  epsilon-placement differs across frameworks and fails the gate for uninteresting reasons.
- The OCANNL runner keeps intermediates Virtual (recomputed in backward) in the untimed
  parity configuration; the tuned variant tunes both that graph and the materialize-all
  graph (placement A/B) and keeps the faster one.
- A tuned OCANNL cell's result line carries a `tune` object (gh-ocannl-546): `shipped` names the
  arm whose artifact was kept, and each arm reports its crowned candidate's label, whether that
  schedule tensorizes, how many of its `Tile_mma` statements rendered as the lane-0 scalar
  fallback, the seeded/timed tensorized counts, and the best *timed* tensorized candidate's time.
  A tensorized win in the arm that loses the A/B reaches no artifact and shows up in no step time,
  so without this the sweep can only find it by grepping `OCANNL_AUTOTUNE_LOG` output that a
  successful cell discards. Read `mma_best_ms` against `best_ms` for the margin: tensorization
  losing by 1% and by 40% are different findings.
- tinygrad's loss must be realized before `opt.step()` (in-place assigns; a later realize
  would recompute the loss from updated weights). tinygrad JIT capture happens during the
  first parity steps; loss values are unaffected.
- Tuned-vs-untuned must be paired within a comparison: OCANNL `BENCH_TUNE=1` corresponds to
  tinygrad `BEAM=...` search and `torch.compile` — one-time search cost for better kernels.
- OCANNL tuned cells run a two-pass protocol: pass 1 runs the search and populates
  `autotune_cache/` (its `compile_s` — the search cost — is what gets reported), then a fresh
  pass-2 process replays the cached winner and provides the step timings. Rationale: the search
  leaves its own process measurably slower (extra per-launch overhead from accumulated
  modules/buffers; 2.5–3.5x on small CUDA kernels), which would penalize the tuned artifact for
  the one-time search it already paid for in `compile_s`. Wipe `autotune_cache/` before a run
  whose `compile_s` should reflect a from-scratch search.
- Timing on a laptop: prefer the p50 of the per-step synced times; rerun and compare rounds
  if thermals are suspect. Keep timing out of CI; the parity gate is the CI-worthy part.
- Untuned-default before/after (gh-ocannl-491): the model-picked default is config-gated, so
  the comparison is the same untuned benchmark run twice — once as-is (the ordinary default
  pipeline) and once with `--ocannl_model_default_schedule=true` (the analytic cost model
  scores the default pipeline and the sketch families inside the compile and applies the
  argmin; zero timing runs, so `compile_s` stays a compile cost). Both runs are untuned in the
  tuned-vs-untuned sense above — do not mix them into a `BENCH_TUNE=1` comparison cell. On
  backends without advisory envelope constants (the C backends), set `--ocannl_model_peak_flops`
  / `--ocannl_model_peak_memory_bandwidth` or the gate falls back to the ordinary default.

## Debugging a parity failure

Write a numpy oracle against the fixture (contiguous batches, fp64) and bisect: forward-only
losses with `lr=0` isolate batch contents; bias gradients after one step isolate backprop
(biases start at zero, so `b == -lr * db`); `BENCH_DEBUG=1`/`BENCH_NO_SGD=1`/`BENCH_NO_SLICE=1`
plus `--ocannl_output_debug_files_in_build_directory=true` (then read `build_files/*.cd`/`.c`)
localize OCANNL-side miscompiles.
