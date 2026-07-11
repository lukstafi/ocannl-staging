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

## Layout

- `workloads/*.json` — workload specs (dims, batch size, lr, step counts, data kind, seed).
- `gen_fixtures.py` — generates `fixtures/<name>.safetensors` from a spec: initial weights
  (`w<i>` as `[fan_out, fan_in]` — the shared row-major convention — and `b<i>`), dataset
  (`x`, one-hot `y`), and all hyperparameters embedded in the safetensors `__metadata__` map,
  so fixtures are self-describing and runners need only the fixture path.
- `runners/ocannl/bench_mlp.ml` — OCANNL runner (`dune build benchmarks/runners/ocannl/bench_mlp.exe`).
  Env: `BENCH_FIXTURE` (path), `BENCH_TUNE=1` (materialize-all + `Autotune.tune` variant);
  backend via the usual `--ocannl_backend=cc|metal`. Debug helpers: `BENCH_DEBUG=1` prints
  bias gradients and values after one step and exits; `BENCH_NO_SGD=1` compiles the gradient
  update without the SGD step; `BENCH_NO_SLICE=1` skips `@|` batch slicing (requires a
  single-batch fixture).
- `runners/pytorch/run.py` — flags: `--device cpu|mps`, `--compile` (torch.compile variant).
- `runners/tinygrad/run.py` — flags: `--device CPU|METAL`, `--jit 0|1`.
- `orchestrate.py` — runs the matrix, enforces the parity gate, writes
  `results/results.jsonl` and `results/report.md`. Flags: `--workloads mlp_small ...`,
  `--tuned` (adds the OCANNL autotuned variant), `--nojit` (adds tinygrad nojit),
  `--only ocannl pytorch tinygrad`, `--skip-build`.

## Setup

```bash
python3 -m venv benchmarks/.venv
benchmarks/.venv/bin/pip install numpy safetensors torch tinygrad
benchmarks/.venv/bin/python benchmarks/gen_fixtures.py
benchmarks/.venv/bin/python benchmarks/orchestrate.py
```

## Methodology notes / fairness pitfalls

- Losses are recorded per step *before* that step's SGD update (forward runs first in every
  framework's step). The first step doubles as the compile probe in the Python runners; for
  OCANNL, `compile_s` wraps `Context.compile` (or `Autotune.tune`).
- SGD is plain `p -= lr * grad` in all three (OCANNL's `Train.sgd_one` and tinygrad's
  `nn.optim.SGD` share the same reference semantics). Don't switch parity workloads to Adam:
  epsilon-placement differs across frameworks and fails the gate for uninteresting reasons.
- The OCANNL runner keeps intermediates Virtual (recomputed in backward) in the untimed
  parity configuration; the tuned variant materializes everything before autotuning.
- tinygrad's loss must be realized before `opt.step()` (in-place assigns; a later realize
  would recompute the loss from updated weights). tinygrad JIT capture happens during the
  first parity steps; loss values are unaffected.
- Tuned-vs-untuned must be paired within a comparison: OCANNL `BENCH_TUNE=1` corresponds to
  tinygrad `BEAM=...` search and `torch.compile` — one-time search cost for better kernels.
- Timing on a laptop: prefer the p50 of the per-step synced times; rerun and compare rounds
  if thermals are suspect. Keep timing out of CI; the parity gate is the CI-worthy part.

## Debugging a parity failure

Write a numpy oracle against the fixture (contiguous batches, fp64) and bisect: forward-only
losses with `lr=0` isolate batch contents; bias gradients after one step isolate backprop
(biases start at zero, so `b == -lr * db`); `BENCH_DEBUG=1`/`BENCH_NO_SGD=1`/`BENCH_NO_SLICE=1`
plus `--ocannl_output_debug_files_in_build_directory=true` (then read `build_files/*.cd`/`.c`)
localize OCANNL-side miscompiles.
