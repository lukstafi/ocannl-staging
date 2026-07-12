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
- **gpt2_mini** (`model: gpt`, `mode: infer`): pre-LN GPT-2-style decoder (4 layers, d=256,
  8 heads, seq 128, vocab 1024, tanh-gelu, learned positional embeddings, tied lm_head,
  causal mask filled with -1e9), forward-only. The parity metric is softmax-CE of the
  logits against fixture target ids, recorded per batch with no updates; the report shows
  tokens/s. Token embedding uses the logical one-hot gather (gh-343); LayerNorm is the
  idiomatic `Nn_blocks.layer_norm` (gammas/betas injected by name like the attention
  weights).

## Layout

- `workloads/*.json` — workload specs; `gen_fixtures.py` generates
  `fixtures/<name>.safetensors` with initial weights in OCANNL's axis conventions (output
  axes then input axes; channels-last images — layouts documented per model in the
  generator), dataset, and all hyperparameters in the safetensors `__metadata__` map, so
  fixtures are self-describing and runners need only the fixture path.
- `runners/ocannl/bench_{mlp,conv,gpt}.ml` + `bench_harness.ml` — OCANNL runners
  (`dune build benchmarks/runners/ocannl/bench_mlp.exe` etc.). Env: `BENCH_FIXTURE` (path),
  `BENCH_TUNE=1` (materialize-all + `Autotune.tune`), `BENCH_MATERIALIZE=1` (materialize
  intermediates without tuning); backend via the usual `--ocannl_backend=cc|metal`. Debug
  helpers: `BENCH_DEBUG=1` prints param names/dims (conv/gpt) or bias gradients (mlp) and
  exits; `BENCH_NO_SGD=1` compiles the gradient update without the SGD step (mlp);
  `BENCH_NO_SLICE=1` skips `@|` batch slicing (mlp, single-batch fixture).
- `runners/pytorch/run.py` — flags: `--device cpu|mps`, `--compile` (torch.compile variant).
- `runners/tinygrad/run.py` — flags: `--device CPU|METAL`, `--jit 0|1`.
- `orchestrate.py` — runs the matrix (dispatching the OCANNL executable on the fixture's
  `model`), enforces the parity gate, writes `results/results.jsonl` and
  `results/report.md`. Flags: `--workloads mlp_small ...`, `--tuned`, `--materialized`,
  `--nojit` (tinygrad nojit), `--only ocannl pytorch tinygrad`, `--skip-build`.

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
