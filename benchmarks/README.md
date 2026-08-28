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
- `fixtures/DIGESTS.txt` + `fixture_digest.py` — **which bytes a published number is on**
  (gh-ocannl-645). The fixtures are gitignored regenerable artifacts, so this file is the only
  checked-in statement of what one contains: `gen_fixtures.py` records `<sha256>  <bytes>
  <name>` as it generates (announcing a *changed* digest loudly, and leaving a reviewable git
  diff), `orchestrate.py` refuses to measure a fixture whose bytes do not match it
  (`--no-fixture-digest-check` opts out, for a deliberate regeneration you are about to
  re-record), and every result row and report section states the digest it ran on. Fixture
  bytes depend on the spec, on the generator, *and* on the numpy version that drew the random
  streams — numpy promises no `Generator` stream stability across releases — so a mismatch is
  real information even when `workloads/` is untouched. A fixture regenerated at a different
  spec revision is otherwise invisible: it is consumed **uniformly** by every cell, and the
  cross-cell parity gate compares cells with each other, not with the workload the report
  names, so it certifies exactly as it certifies the intended one.
- `runners/ocannl/bench_{mlp,conv,gpt}.ml` + `bench_harness.ml` — OCANNL runners
  (`dune build benchmarks/runners/ocannl/bench_mlp.exe` etc.). Env: `BENCH_FIXTURE` (path),
  `BENCH_TUNE=1` (`Train.tune_placements`: autotunes both the default placements graph and
  the materialize-all graph, keeping the measured winner), `BENCH_MATERIALIZE=1` (materialize
  intermediates without tuning), `BENCH_PRECISION=bf16|f16` (mlp and gpt, gh-ocannl-492);
  backend via the usual `--ocannl_backend=cc|metal|cuda|hip`. `bench_gpt` dispatches its step
  shape on the fixture's `mode`, like the Python runners: forward-only for `gpt2_mini`,
  backprop plus SGD for `gpt2_mini_train`.

  `bench_mlp --self-test` (gh-ocannl-702) runs the measurement path with **no fixture and no
  Python**: `Bench_harness.run_self_test` fabricates a tiny model in memory, drives the whole
  protocol -- parity window, warmup, per-step-synced percentiles, queued mean -- and prints a real
  result line, on whatever backend the usual `--ocannl_backend=` selects. It is deliberately *not*
  a comparable cell: its bytes are not the byte-identical fixture the parity gate is built on, and
  the emitted record says so (`"workload":"selftest-tiny"`, `"variant":"self-test"`). Use it to
  smoke a fresh checkout or a new backend before provisioning `benchmarks/.venv`, and to see what a
  result line looks like. `test/operations/bench_self_test` runs the same mode in CI and asserts
  that the emitted record is well-formed, so the emitter every benchmark number flows through has
  an executable guard that needs no fixtures. The mode lives in `Bench_harness`, so a second runner
  can offer it in one line.

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
  `BENCH_PRESEED_TWINS=1` (mlp, reduced precision) reaches the same placement through
  `Context.decide_materialized` instead: a context-level *decision* rather than a tnode-level
  intent, so unlike `BENCH_TWIN_PLACEMENT` it is visible to the placement A/B and composes with the
  gh-555 flip chain. The two are refused together: declaring the intent as well either nullifies
  the pre-seed (`virtual` — `decide_materialized` skips declared-virtual nodes by contract) or
  applies it to both arms (`materialized`), and neither shows up in the result line. `BENCH_TUNE_REPORT=1` (mlp) prints each search's report fields on stderr —
  both placement arms under `tune arm:` and each `tune_inline_flips` refinement search under
  `tune flip:`; `BENCH_FLIP_DUMP=1` (mlp) prints the default-placement compile's whole
  `flip_candidates` list, which is how "this node is not a searchable decision" is told apart from
  "it ranked below `tune_inline_flips`" (gh-ocannl-558,
  [report-gh558-hip-flips.md](report-gh558-hip-flips.md)).
- `runners/pytorch/run.py` — flags: `--device cpu|mps|cuda`, `--compile` (torch.compile
  variant, which reports whether inductor's codegen ran here or came from its cache),
  `--compile-mode MODE` (a `torch.compile` mode such as `max-autotune` — the honest analogue of a
  tuned cell, since it benchmarks kernels; measured for gh-ocannl-675 but not a matrix cell) and
  `--retime` (time a second block of `timed_steps` in the same process, which is what separates
  "a process that searched is slower per launch" from first-block warmup).
- `runners/tinygrad/run.py` — flags: `--device CPU|METAL|CUDA|AMD|CL|HIP`, `--jit 0|1`, `--beam N`
  (BEAM=N kernel search, implies jit; the search cost lands in `compile_s`, and the result line
  reports whether the beam actually searched or replayed `~/.cache/tinygrad`), `--retime` (as
  above). Redirect the kernel cache per run with `CACHEDB=<path>` when its warmth is the
  experiment. On every device whose beam search parallelises, the search runs its candidate
  compiles in a `spawn` pool whose workers re-execute this module top-level, so the runner drops
  `runners/` from `sys.path` again right after importing `bench_common`: left there, the worker's
  own `import tinygrad` resolves `runners/tinygrad/` as a namespace package, every worker dies,
  the pool respawns them forever, and the search hangs rather than failing (gh-ocannl-675).
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
  CPU-only matrix), `--no-fixture-digest-check` (measure fixtures that do not match
  `fixtures/DIGESTS.txt`),
  `--cell-timeout SECONDS` / `--beam-parallel N` / `--no-cache-quarantine` (the wedged-cell
  mitigations below). Env: `BENCH_CELL_LOG_DIR=<dir>` keeps every cell's raw combined
  output, one file per cell label — a successful cell's output is otherwise discarded, which throws away the
  candidate-level evidence a measurement sweep has to report. Combined with
  `OCANNL_AUTOTUNE_LOG=true` it makes the seeded-vs-timed mma and split-reduce counts, the
  `FAILED` blocker breakdown and the split-reduce evictions fall out of the sweep's own search
  passes instead of costing a second round of searches (it does inflate a tuned cell's reported
  `compile_s` a little; step times come from the pass-2 replay and are unaffected).

  **A wedged cell costs the cell, not the sweep** (gh-ocannl-760). tinygrad's parallel beam
  search deadlocks intermittently — the same search that takes 53–115 s in its other repeats
  sits at ~1% CPU with the GPU idle indefinitely, seen on both the CUDA and the HIP box — so
  every cell runs in a process group of its own under a wall-clock cap, `--cell-timeout SECONDS`
  (default 1800; `0` disables). Over the cap, the whole group is killed — the runner *and*
  whatever it spawned, tinygrad's candidate-compile pool included, which is also why the kill is
  a group kill: those workers hold the cell's stdout pipe, so killing the direct child alone
  would move the hang into the sweep's own read — and the cell is recorded as a runner failure,
  in the run log and in the report's **Runner failures** section. Raise the cap for a box whose
  legitimate cells run longer; the failure names it either way.

  A killed beam search leaves a **partial `cache.db`**: the next run over it neither replays a
  complete result nor searches from scratch, while `searched` reports one of the two, so a retry
  over that cache is not the pass it claims to be. tinygrad's cache is a single sqlite file, so
  the kill path renames it (with its `-wal`/`-shm` siblings) to `cache.db.wedged-<timestamp>` —
  the retry starts cold and the torn cache is still there to inspect. `--no-cache-quarantine`
  leaves it in place and still names the risk in the failure. OCANNL's `autotune_cache/` needs no
  equivalent: entries are committed by rename (`Utils.Atomic_file`), so a killed search leaves
  complete entries and the only consequence — a retry replaying the arms that finished — is what
  the `search pass verdict` line already reports.

  `--beam-parallel N` passes tinygrad's own `PARALLEL` knob through to the beam cells. Its
  default is one candidate-compile worker per logical core on a GPU device (24 on the box the
  wedges were seen on, and measured as exactly that many `--multiprocessing-fork` children
  during a search); `--beam-parallel 0` is the value that disables the pool outright and
  compiles the candidates in-process, which is the configuration a pool deadlock cannot occur in
  — `1` still means a one-worker pool, not no pool. It costs search wall time (`mlp_wide` beam
  on the RTX 5070 Ti: 60.5 s of search at the 24-worker default against 149.5 s at
  `--beam-parallel 2`, two runs that overlapped on the GPU, so read the ratio and not the
  seconds), so nothing makes it the default: the root cause is upstream and unchased, and the
  cap already bounds the damage.

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
  [report-gh558-hip-flips.md](report-gh558-hip-flips.md) (WSL2/HIP on gfx1151 — gh-ocannl-558's
  reduced scope: the reduced-precision cast twins *are* on gh-555's inlining decision surface, so
  the flip chain reaches the tensorized candidate family from the default-placement arm unaided,
  worth −37.0% inside that arm on `mlp_wide`/bf16 — −33.7% for the `decide_materialized` control
  that materializes the twins and nothing else; it still loses to materialize-all, which reaches
  two tensorized sites where site-targeted materialization reaches one),
  [report-gh528-hip.md](report-gh528-hip.md) (WSL2/HIP on gfx1151, the hardware validation of
  gh-ocannl-528's interior-batch `Tensorize` and gh-ocannl-481's HIP declines, plus the `gpt2_mini`
  tensor-core probe: three tensorized sites reached and verified in the emitted source, landing
  within noise of an f32 schedule with none — and the reference for two facts that outlive it,
  that gfx1151's WMMA is not exactly-rounded in any format combination, and that `taskset -c 0-15`
  on that box is 8 SMT-shared cores rather than 16 private ones),
  [report-gh675-cuda.md](report-gh675-cuda.md) (WSL2/CUDA on an RTX 5070 Ti — gh-ocannl-675's
  NVIDIA leg: what a searching process actually costs per launch, per cell, against non-searching
  controls and a sign test over paired repeats; the reference for why every non-OCANNL searching
  cell stays single-pass, for the fact that the 2.5–3.5x this file used to quote does not
  reproduce, and for two tinygrad traps — the `sys.path` leak that wedges a parallel beam search
  and the driver JIT cache that makes a second "from-scratch" CUDA search several times cheaper),
  [report-gh569-hip.md](report-gh569-hip.md) (WSL2/HIP on gfx1151 — the cross-backend test of
  gh-ocannl-569's companion-coverage blocker: the same rule declines the same sites at the same
  `8x128x1024` geometry, and the same five kernels dominate in the same naive scalar form at the
  same 1024-thread launch, but they are 47.2% of the step rather than CUDA's 70.2% and run at
  5.6%/2.5% of a measured local peak rather than 1.3%; also the reference for the fact that
  `rocprofv3` collects nothing under WSL2 for want of `/dev/kfd`, and for the profiler-free
  per-kernel reconstruction that replaces it),
  [report-gh612-hip.md](report-gh612-hip.md) (WSL2/HIP on gfx1151 — its successor, and the
  measurement of gh-ocannl-573's fanin guard and gh-ocannl-574's `arity_cuts` finer fission in one
  session against a re-established denominator: both predicted line items reproduce to ~1% in
  absolute terms while neither share survives, 1.30x and 1.31x on the default-placement arm, kernel
  time 32.33 → 18.88 ms — also the reference for why a share is a share of a placement and a tree,
  for the fact that equal kernel counts can absorb a changed materialization decision (on the
  `gpt2_mini` graph `virtualize_max_inline_fanin` 16 shows one node's worth of placement difference
  behind an unchanged 135-kernel arm A, and cap 32 is the only measured cap matching cap −1's
  placement — equality there does not extrapolate, since a zero placement diff cannot show whether
  the guard fired), and for
  the three-instrument discipline (per-kernel profile 0.08–1.6%, untuned-default 0.2–0.7%, shipped
  tuned p50 2.6–19.1%) that a tuned-cell A/B on that box needs),
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
  every segment sharing identical launch geometry (gh-ocannl-531). Each segment line carries two
  censuses of what produced its kernel: `mma:` (did it tensorize, or render the scalar fallback) and
  `vol:` (how many of its serial accumulations the Metal compiler-bug workaround pinned to memory,
  gh-ocannl-782 — on the shapes where the accumulator is the critical path that is worth up to 4x,
  so a surprising segment time is read together with this).
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
  around it via `volatile_serial_accumulation` in `arrayjit/lib/c_syntax.ml`; this repro documents the
  raw bug (prints the wrong value as long as the toolchain is affected) and
  `test/operations/scalar_rmw_accumulation.ml` guards the workaround end to end.
- `runners/ocannl/bench_metal_bug_local.ml` — the same defect in the spelling the serial-reduction
  localizer produces (a scope-local accumulator stored once), which is what OCANNL emits today.
  Also standalone, and it answers two questions rather than one. Its matrix renders the emitted
  kernel with one factor changed at a time — the qualifier, `__restrict`, how the pointers are
  formed, where the preceding device store lands, what the loop reads — each checked against a host
  oracle, so it says what the defect keys on rather than only that it exists; every row is its own
  single-kernel library. Its tax table then times three localized-reduction shapes with and without
  the qualifier on the GPU's own clock, interleaved with a rotating arm order, best of 30, which
  is the measurement behind
  keeping the predicate wide (gh-ocannl-782). Run it whenever the toolchain moves: a matrix that
  comes up all-`ok` means the defect is gone and the workaround can be retired.

## Setup

```bash
python3 -m venv benchmarks/.venv
benchmarks/.venv/bin/pip install numpy safetensors torch tinygrad
benchmarks/.venv/bin/python benchmarks/gen_fixtures.py
benchmarks/.venv/bin/python benchmarks/orchestrate.py
```

`gen_fixtures.py` rewrites `fixtures/DIGESTS.txt` for whatever it regenerates. Review that diff
before publishing numbers: a changed digest means the workload changed, and reports measured on
either side of it are not comparable.

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
  schedule tensorizes, how many of its `Tile_mma` statements were emitted and how many of those
  rendered as the lane-0 scalar fallback, the seeded/timed tensorized counts, and the best *timed*
  tensorized candidate's time.
  A tensorized win in the arm that loses the A/B reaches no artifact and shows up in no step time,
  so without this the sweep can only find it by grepping `OCANNL_AUTOTUNE_LOG` output that a
  successful cell discards. Each arm also names its outcome `state` — `searched`,
  `search-died`, `cache-replay`, `search-disabled` or `pre-search-failure`, the `Autotune.outcome`
  the call reported (gh-ocannl-677) — which is what makes a *mixed* cell readable: one arm cached,
  the other searched because its half of the A/B never was. Read `mma_best_ms` against `best_ms` for the margin: tensorization
  losing by 1% and by 40% are different findings.
- **`tensorized` is what the schedule asked for; `tensorization` is what the emission did**
  (gh-ocannl-626). Each arm carries both: `tensorization` is `"tensorized"`, `"scalar-fallback"`
  (every emitted `Tile_mma` declined to the lane-0 scalar loop) or `"not-requested"` (codegen
  emitted no `Tile_mma` at all), read off the compiled routine's census rather than re-derived by
  a harness, and `null` when there was no crowned candidate to consult — so an arm that consulted
  no census can never read as tensorized. The `tune` object additionally carries `shipped_mma` —
  the census of the routine whose steps were TIMED — and that is what `orchestrate.py` reads, not
  the arm named as shipped: a gh-555 flip refinement that wins ships under `shipped: "flip"` and is
  not an arm at all, and on the `timing_ctx` path the tuner recompiles the winner in the production
  context and falls back to the untuned default when that replay is rejected, so in both cases the
  arm describes a schedule that was discarded. A `Tile_mma` declines for a column extent below the
  compute vector width, a narrow `vector_bytes`, mixed operand precisions, transposed-B storage or
  `debug_log_from_routines`, and the resulting scalar kernel compiles, runs and times perfectly
  well under an `mma-*` label — so `tensorized: true` with any other `tensorization` means the
  row's number is a *scalar* timing. `orchestrate.py` reads that pair into the
  report's `mma` column (**`SCALAR FALLBACK`** / **`NO MMA EMITTED`** are shouted; `tensorized`
  and `—` are not) and prints a `TENSORIZATION NOTICE` naming the mismatched cells. It is a notice
  rather than a gate: declining is sometimes the correct codegen decision, and the defect is
  quoting the row as a tensor-core measurement, not the decline itself. The per-segment table
  `bench_*_diag` prints carries the same label per kernel.
- A non-finite number never reaches a result line: a diverged loss, a time that was never
  measured, an arm that timed nothing are emitted as JSON `null` (gh-ocannl-676). OCaml's `%g`
  spells those `nan` / `inf` / `-inf` and Python's `json.dumps` writes `NaN`, none of which is
  JSON — so a diverged cell, the exact thing the parity gate exists to catch, used to be reported
  as a *broken runner* with its loss trajectory discarded after the whole measurement had been
  paid for. `test/operations/bench_result_line` pins the OCANNL line by re-parsing it with
  fabricated values, and `test_orchestrate.py` re-parses that golden with the reader that has to
  accept it. A `null` in a loss vector is read as **DIVERGED**: a parity-gate failure naming the
  step the trajectory left the finite numbers, with the trajectory itself kept in
  `results/results.jsonl` — not a runner failure, and not a stationary loss.
- tinygrad's loss must be realized before `opt.step()` (in-place assigns; a later realize
  would recompute the loss from updated weights). tinygrad JIT capture happens during the
  first parity steps; loss values are unaffected.
- Tuned-vs-untuned must be paired within a comparison: OCANNL `BENCH_TUNE=1` corresponds to
  tinygrad `BEAM=...` search and `torch.compile` — one-time search cost for better kernels.
- OCANNL tuned cells run a two-pass protocol: pass 1 runs the search and populates
  `autotune_cache/` (its `compile_s` — the search cost — is what gets reported), then a fresh
  pass-2 process replays the cached winner and provides the step timings. Rationale: the search
  leaves its own process measurably slower (extra per-launch overhead from accumulated
  modules/buffers), which would penalize the tuned artifact for the one-time search it already
  paid for in `compile_s`. Wipe `autotune_cache/` before a run whose `compile_s` should reflect a
  from-scratch search — and note that on CUDA the driver's own JIT cache (`~/.nv/ComputeCache`)
  survives that wipe, so a second from-scratch search on the same box is several times cheaper
  than the first.

  **How large that penalty is, per cell and per box, is measured** (gh-ocannl-675;
  [report-gh675-cuda.md](report-gh675-cuda.md) and the ROCm table in the issue). Each searching
  cell was timed twice — once in the process that searched, once in a fresh process replaying its
  cache — with the non-searching cells run as the same pair for a spread control. Reading X as
  (pass-1 step p50)/(pass-2 step p50) − 1, median over paired repeats, on `mlp_small` /
  `gpt2_mini`:

  | cell | RTX 5070 Ti (CUDA 13.3) | Radeon 8060S / gfx1151 (ROCm 7.14) |
  |---|---|---|
  | OCANNL `tuned` | +10.3% behind a 16 s search, ≈0% behind a 4 s one; +0.5% on gpt2_mini | −0.2% / −0.1% |
  | tinygrad `--beam 2` | +6.4% / +2.3% (positive in 9/9 and 4/4 paired repeats) | +14.9% / +9.0% |
  | `torch.compile` (default) | **−12.0% / −4.0%** — the searching process is the *faster* one | +7.2% / +21.2% |
  | `torch.compile mode="max-autotune"` | −2.0% / −1.1% | +12.3% / +33.9% |
  | *controls* (eager, warm jit, cold-compile jit) | \|X\| ≤ 2.5%, sign a coin flip | \|X\| ≤ 1.7% |

  Three things follow, and the matrix is built on them. **(a)** The 2.5–3.5x this file used to
  quote for small CUDA kernels does not reproduce on either box: the largest OCANNL residue
  measured — which is what that figure was about — is +10.3%, so the protocol is kept for a
  measured ≤10.3% rather than for a factor of three. (Two *other* cells do exceed it on ROCm,
  +21.2% and +33.9%; that is the asymmetry this table exists to state, and it is handled in (c),
  not by the OCANNL bound.) **(b)** No mechanism is
  established, and the table should not be read as offering one — both obvious readings die on
  measured rows. "An expensive search leaves a residue" fails on OCANNL/`gpt2_mini`, which searched
  206–613 s (the costliest search in either leg) for +0.5%; "…and only where steps are short enough
  for a per-launch cost to show" fits every CUDA cell and fails on ROCm, where the two torch cells
  are *larger* on `gpt2_mini` (+21.2%, +33.9%) than on `mlp_small`. Explaining why a given
  framework's process carries a cost would need per-framework instrumentation neither leg ran; the
  rule below rests on the measured X per cell per box, which is what the issue asked for. **(c)** No searching
  cell clears a ~10% line on both boxes, and `torch.compile`'s residue does not even keep its
  sign across them, so **the other cells stay single pass** — at ≤6.4% (beam) and a *negative*
  −12.0% (`torch.compile`) on the box whose hardware this rationale names. "On both boxes" is a
  decision the measurement brief did not specify and the two legs forced: the protocol is a
  property of this MATRIX rather than of a machine, and a cell that is two-pass on one box and
  single-pass on another yields two numbers that cannot be compared with each other — the exact
  harm it exists to prevent. Applied per box instead, the beam cell and `max-autotune` would split
  on ROCm (+14.9%, +12.3%) and neither on CUDA; the numbers are the same either way. Splitting them would
  double their wall clock, break comparability with every published report, and — for the torch
  cells here — move the number the wrong way.

  **The result line says which pass produced it** (gh-ocannl-644): `"searched": true|false` is
  whether *this process* ran a search, and the `tune` object breaks it down per arm (each arm's
  `state`) and in total (`searches` / `replays` / `no_searches`, counting the gh-555 flip
  refinements too, which are searches this process ran even though they are not arms). A tuned cell
  under `autotune_search=false` — the `reproducible` profile — lands in `no_searches`: it shipped
  the untuned default, having neither searched nor replayed, and the report says `no search` rather
  than crediting the row with a tuned artifact it does not have. That third case is *stated* by the
  runner rather than derived from two counters that are both zero (gh-ocannl-677); the legacy
  per-arm `searched` / `cache_hit` booleans stay in the wire format for older readers. `orchestrate.py` gates on
  it — a tuned cell whose step times came from a searching process fails the **PROVENANCE
  GATE** and is marked `SEARCH PASS` in the report's `pass` column — and stamps the search
  pass's own verdict as `search_pass`, so a `compile_s` carried over from a process that
  replayed the cache reads as `(cached)`, and one from a process that searched nothing at all as
  `(no search)`, rather than either passing as a from-scratch search cost. One classifier
  (`search_provenance`) reads `searched` for all three consumers: the fact is three-valued, and
  every boolean spelling of it has been wrong somewhere. Before this, both passes emitted an
  identical `framework`/`backend`/`variant`/`precision`, and the only trace of the difference was
  a `cache hit:` line on the stderr of a run whose output a successful cell discards:
  `report-gh612-hip.md` quoted pass-1 step times for fifteen revisions, and it took a reviewer
  noticing that the driver never started a second process to find it.
  A checked-in driver should assert `searched == false` rather than infer a replay from
  `compile_s` being small.

  **Every runner reports it, but only the OCANNL tuned cell is gated on it.** tinygrad's
  `--beam N` cell and `torch.compile` search or codegen *in the timing process* by protocol, so
  their rows read `same-process` — or `cached`, when the beam result came from
  `~/.cache/tinygrad` or the graph from inductor's cache, in which case their `compile_s` is a
  replay cost too. Whether those frameworks pay for searching in the timing process the way
  OCANNL does is **measured, and the answer is per-box** (gh-ocannl-675 — the table above);
  reporting it is the point, since silence there reads as though the question applied to OCANNL
  alone. The two probes read
  framework internals (tinygrad's beam disk cache, torch's FX-graph cache counters) and answer
  `null` — reported as `?` — when they cannot tell, rather than guessing a `false` that would be
  exactly the silent claim this field exists to prevent.
- **A report states the fixture digest its numbers are on.** `orchestrate.py` puts it in each
  workload section of `results/report.md` and in every `results.jsonl` row (`fixture`,
  `fixture_sha256`); a hand-written report quotes the same `fixtures/DIGESTS.txt` line, and a
  hand-written driver pins it (`gh612_cells.sh` refuses to run a cell whose fixture does not
  match a pinned digest, with an env opt-out for deliberate re-generation). Cross-session
  comparisons depend on it entirely: `report-gh569-hip.md`'s 46.65 ms denominator and
  `report-gh612-hip.md`'s 32.33 ms are comparable only if both ran the same bytes, and until
  gh-ocannl-645 that was an assumption no artifact recorded.
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
