# Benchmark results

platform: macOS-26.5.2-arm64-arm-64bit-Mach-O arm64 | ocannl commit: f34d8a1c | parity tol: 0.002 (max rel diff over first parity steps vs pytorch/cpu/eager)
> Checked-in example output (`results/` itself is generated and gitignored): the macOS leg of the
> gh-ocannl-476 measurement sweep, run from scratch (wiped `autotune_cache`) after the gh-ocannl-502
> seeding wave and the gh-ocannl-492 mixed-precision legs landed. 79 cells, no runner failures.
> Tuned cells use the two-pass protocol (pass 1 = the search, reported as `compile_s`; a fresh
> pass-2 process replays the cached winner for step timings). Regenerate with
> `benchmarks/.venv/bin/python benchmarks/orchestrate.py --tuned --materialized --precision bf16 f16`.
>
> Two cells — `mlp_small` and `mlp_wide` `ocannl/metal/bf16` — were re-measured after the Metal
> bf16 `Relu` rendering fix in the same series; every other cell is as measured at `f34d8a1c`. The
> re-measurement also re-ran the unchanged neighbouring cells, which agreed with the main sweep to
> within 0.0–4.3% (mostly under 1%), which is the run-to-run variance this splice sits inside.
> `cifar_conv metal/tuned` is skipped by `SKIP_CELLS`; a standalone from-scratch search of that cell
> completed here in 2069 s without the post-tune reinit hang that motivated the skip, so the entry
> may be stale — confirm before removing it.
>
> The `f16` step times include the dynamic-loss-scaling inf/nan gate's per-step host sync, so they
> are not methodologically comparable to the `f32` legs. That caveat does not explain their size:
> a per-step sync is roughly constant (and `mlp_small/metal` fits that, 2.275 vs 1.465 ms), whereas
> `mlp_wide/metal` is 87.068 vs 6.291 ms, i.e. slow f16 *compute*. `bf16` carries no gate at all and
> is still 5.3x slower than f32 on `cc` (1171.910 vs 219.169 ms on `mlp_wide`), consistent with the
> C backend having no native bf16 arithmetic; on Metal it is roughly a wash (6.192 vs 6.291 ms).

## cifar_conv

| framework | backend | variant | step p50 ms | p10 | p90 | queued ms | compile s | parity |
|---|---|---|---|---|---|---|---|---|
| pytorch | mps | eager | 2.638 | 2.263 | 2.780 | 1.883 | 0.18 | PASS (1.2e-04) |
| tinygrad | METAL | jit | 4.859 | 4.361 | 4.971 | 4.137 | 1.37 | PASS (1.2e-04) |
| pytorch | cpu | eager | 35.402 | 34.894 | 35.833 | 35.408 | 0.06 | REF |
| tinygrad | CPU | jit | 66.984 | 62.064 | 69.932 | 66.266 | 1.44 | PASS (1.2e-04) |
| ocannl | cc | tuned | 275.883 | 267.888 | 282.506 | 275.938 | 324.38 | PASS (3.1e-07) |
| ocannl | metal | materialized | 335.781 | 332.164 | 340.798 | 336.582 | 1.64 | PASS (1.2e-04) |
| ocannl | cc | materialized | 728.701 | 724.973 | 732.124 | 725.835 | 3.51 | PASS (1.2e-04) |
| ocannl | cc | default | 844.678 | 840.500 | 849.924 | 850.104 | 2.58 | PASS (1.2e-04) |
| ocannl | metal | default | 2102.550 | 2099.520 | 2104.760 | 2114.830 | 1.52 | PASS (1.2e-04) |

## cifar_stride

| framework | backend | variant | step p50 ms | p10 | p90 | queued ms | compile s | parity |
|---|---|---|---|---|---|---|---|---|
| pytorch | mps | eager | 1.503 | 1.116 | 1.572 | 0.779 | 0.29 | PASS (2.5e-06) |
| tinygrad | METAL | jit | 1.989 | 1.601 | 2.109 | 1.244 | 1.68 | PASS (3.9e-05) |
| pytorch | cpu | eager | 20.119 | 19.810 | 20.393 | 20.089 | 0.03 | REF |
| tinygrad | CPU | jit | 25.014 | 24.783 | 25.581 | 24.992 | 1.93 | PASS (3.9e-05) |
| ocannl | cc | tuned | 90.523 | 87.482 | 95.042 | 90.471 | 184.09 | PASS (2.4e-06) |
| ocannl | metal | tuned | 93.749 | 93.461 | 94.053 | 92.096 | 547.60 | PASS (3.9e-05) |
| ocannl | metal | materialized | 122.719 | 121.330 | 123.346 | 120.824 | 1.51 | PASS (3.9e-05) |
| ocannl | cc | materialized | 189.167 | 188.766 | 189.694 | 189.101 | 3.11 | PASS (3.9e-05) |
| ocannl | cc | default | 211.720 | 211.164 | 212.471 | 211.781 | 2.63 | PASS (3.9e-05) |
| ocannl | metal | default | 395.206 | 393.979 | 396.222 | 393.452 | 1.40 | PASS (3.9e-05) |

## gpt2_mini

| framework | backend | variant | step p50 ms | p10 | p90 | queued ms | compile s | parity | tok/s |
|---|---|---|---|---|---|---|---|---|---|
| tinygrad | METAL | jit | 3.257 | 3.145 | 3.749 | 2.985 | 0.71 | PASS (1.3e-07) | 314,396 |
| pytorch | mps | eager | 5.497 | 5.184 | 6.571 | 5.354 | 0.06 | PASS (1.3e-07) | 186,272 |
| pytorch | cpu | eager | 13.410 | 13.289 | 13.530 | 13.546 | 0.02 | REF | 76,361 |
| tinygrad | CPU | jit | 45.228 | 44.734 | 46.210 | 45.436 | 0.81 | PASS (8.0e-07) | 22,641 |
| ocannl | metal | tuned | 94.372 | 94.135 | 94.723 | 89.945 | 1946.66 | PASS (2.0e-07) | 10,851 |
| ocannl | cc | tuned | 347.317 | 343.817 | 349.810 | 346.682 | 144.07 | PASS (8.1e-07) | 2,948 |
| ocannl | metal | default | 364.787 | 363.607 | 365.072 | 359.140 | 1.18 | PASS (2.0e-07) | 2,807 |
| ocannl | metal | materialized | 395.698 | 394.348 | 396.446 | 390.722 | 1.51 | PASS (8.1e-07) | 2,588 |
| ocannl | cc | default | 2081.800 | 2077.510 | 2084.200 | 2080.640 | 3.79 | PASS (8.1e-07) | 492 |
| ocannl | cc | materialized | 2351.260 | 2347.900 | 2357.670 | 2361.930 | 15.39 | PASS (8.1e-07) | 436 |

## lenet

| framework | backend | variant | step p50 ms | p10 | p90 | queued ms | compile s | parity |
|---|---|---|---|---|---|---|---|---|
| tinygrad | METAL | jit | 1.236 | 0.770 | 1.325 | 0.572 | 1.32 | PASS (2.1e-07) |
| pytorch | mps | eager | 5.661 | 5.474 | 5.800 | 4.613 | 0.11 | PASS (1.0e-07) |
| tinygrad | CPU | jit | 5.687 | 5.569 | 5.789 | 5.686 | 1.33 | PASS (3.1e-07) |
| pytorch | cpu | eager | 12.355 | 12.033 | 12.584 | 12.336 | 0.02 | REF |
| ocannl | cc | materialized | 14.976 | 14.878 | 15.093 | 14.982 | 2.14 | PASS (2.1e-07) |
| ocannl | cc | tuned | 14.995 | 14.882 | 15.086 | 15.003 | 112.79 | PASS (2.1e-07) |
| ocannl | cc | default | 23.424 | 23.224 | 23.557 | 23.407 | 1.83 | PASS (2.1e-07) |
| ocannl | metal | tuned | 36.238 | 36.042 | 36.546 | 34.451 | 143.51 | PASS (2.1e-07) |
| ocannl | metal | materialized | 37.780 | 37.507 | 38.105 | 35.996 | 1.42 | PASS (2.1e-07) |
| ocannl | metal | default | 502.750 | 501.445 | 503.682 | 503.809 | 1.31 | PASS (2.1e-07) |

## mlp_small

| framework | backend | variant | step p50 ms | p10 | p90 | queued ms | compile s | parity |
|---|---|---|---|---|---|---|---|---|
| ocannl | cc | tuned | 0.117 | 0.105 | 0.145 | 0.095 | 15.34 | PASS (2.2e-07) |
| ocannl | cc | materialized | 0.177 | 0.154 | 0.223 | 0.140 | 0.43 | PASS (2.2e-07) |
| ocannl | cc | default | 0.182 | 0.160 | 0.248 | 0.142 | 0.40 | PASS (2.2e-07) |
| pytorch | cpu | eager | 0.183 | 0.168 | 0.221 | 0.188 | 0.01 | REF |
| tinygrad | CPU | jit | 0.299 | 0.294 | 0.306 | 0.307 | 0.33 | PASS (2.8e-07) |
| ocannl | cc | f16 | 0.339 | 0.302 | 0.400 | 0.279 | 0.80 | PASS (1.5e-04) |
| ocannl | cc | bf16 | 0.765 | 0.757 | 0.847 | 0.762 | 0.43 | PASS (4.2e-04) |
| tinygrad | METAL | jit | 0.812 | 0.314 | 0.953 | 0.625 | 0.36 | PASS (2.4e-07) |
| pytorch | mps | eager | 1.136 | 0.990 | 1.229 | 0.278 | 0.08 | PASS (1.2e-07) |
| ocannl | metal | bf16 | 1.271 | 1.042 | 1.359 | 0.475 | 0.02 | PASS (4.2e-04) |
| ocannl | metal | materialized | 1.287 | 1.210 | 1.391 | 0.486 | 0.06 | PASS (3.2e-07) |
| ocannl | metal | tuned | 1.306 | 1.209 | 1.403 | 0.527 | 5.06 | PASS (3.2e-07) |
| ocannl | metal | default | 1.465 | 1.285 | 1.797 | 0.559 | 0.06 | PASS (3.2e-07) |
| ocannl | metal | f16 | 2.275 | 1.786 | 2.437 | 1.341 | 0.18 | PASS (1.4e-04) |

## mlp_wide

| framework | backend | variant | step p50 ms | p10 | p90 | queued ms | compile s | parity |
|---|---|---|---|---|---|---|---|---|
| pytorch | mps | eager | 1.095 | 0.627 | 1.130 | 0.347 | 0.08 | PASS (2.1e-07) |
| tinygrad | METAL | jit | 1.174 | 0.725 | 1.263 | 1.032 | 0.52 | PASS (2.1e-07) |
| pytorch | cpu | eager | 1.604 | 1.572 | 1.636 | 1.603 | 0.01 | REF |
| ocannl | metal | tuned | 4.137 | 4.071 | 4.470 | 3.175 | 498.52 | PASS (5.2e-07) |
| ocannl | metal | materialized | 5.959 | 5.844 | 6.227 | 4.967 | 0.06 | PASS (5.2e-07) |
| ocannl | metal | bf16 | 6.192 | 6.092 | 6.413 | 5.145 | 0.02 | PASS (2.8e-04) |
| ocannl | metal | default | 6.291 | 6.190 | 6.558 | 5.249 | 0.07 | PASS (5.2e-07) |
| tinygrad | CPU | jit | 13.393 | 13.209 | 13.696 | 13.438 | 0.50 | PASS (7.3e-07) |
| ocannl | cc | tuned | 33.716 | 32.983 | 34.464 | 33.767 | 40.19 | PASS (5.2e-07) |
| ocannl | metal | f16 | 87.068 | 85.632 | 88.789 | 85.843 | 0.22 | PASS (2.0e-05) |
| ocannl | cc | materialized | 219.126 | 218.247 | 220.017 | 219.231 | 0.53 | PASS (6.2e-07) |
| ocannl | cc | default | 219.169 | 218.465 | 220.090 | 219.231 | 0.50 | PASS (6.2e-07) |
| ocannl | cc | f16 | 593.229 | 592.510 | 594.747 | 593.559 | 0.84 | PASS (2.1e-05) |
| ocannl | cc | bf16 | 1171.910 | 1170.890 | 1173.630 | 1172.030 | 0.48 | PASS (2.8e-04) |

## Findings

Instrument notes: the tensor-core check greps the emitted MSL for `simdgroup_multiply_accumulate`,
validated against `schedule_mma_matmul` run with `OCANNL_BACKEND=metal`, which does emit it
(`mm_mma.metal`, `mm_ta_mma.metal`, `mm_staged_mma.metal`) — so the zero counts below are real, not
a broken probe. Candidate-level data comes from fresh searches with `autotune_log=true` and the
cache redirected to a scratch dir, because a warm cache replays the winner and logs nothing about
the losers.

### Tensor cores are unreachable on Metal — blocked before measurement, not declined

Tensorized candidates *are* seeded (30–74 mma-labelled candidates per search), so the gh-ocannl-152
seeding is live. None is ever timed: every one fails at candidate compile, so no tuned Metal winner
in any workload contains a `Tensorize`, and none renders the intrinsic. This is a different failure
mode from the "seeded but renders its scalar fallback" case the sweep was asked to rule out — there
were zero declines and zero dedups. Distinct blockers, by count:

| blocker | mlp_wide/metal | lenet/metal |
|---|---|---|
| `Low_level.validate_parallel`: write to materialized node not nested under loops covering all active hardware dims | 56 | 12 |
| `Schedule.Stage`: source is written in the routine | 24 | 9 |
| `Schedule.Fuse_epilogue`: guarded writes of the reduction output unsupported | 3 | 9 |
| `Schedule.Fuse_epilogue`: accumulator is a whole-K `Tile_mma`; stage the reduction (split K) first | 10 | — |

The missing-slot sets are `Workgroup slot 0, Grid slot 0, Grid slot 1` (30x) and variants. On `cc`
the mma candidates *do* get timed (21 on lenet), so this is Metal/GPU-specific.

### The Metal untuned default is dominated by one backward reduction

`BENCH_SEG_TIMES=1` on `bench_conv_diag`, default pipeline. A single segment writing
`bias_conv1.grad` / `n65.grad` — the first conv layer's *backward* reduction, not the forward GEMM —
accounts for most of the step:

| workload | seg22 | total | share | seg22 geometry |
|---|---|---|---|---|
| lenet/metal | 453.804 ms | 498.486 ms | 91.0% | threads=1792 grid=[64;1;1] block=[28;1;1] ops=3 |
| cifar_conv/metal | 1735.295 ms | 2114.528 ms | 82.1% | threads=2560 grid=[64;1;1] block=[40;1;1] ops=3 |
| cifar_stride/metal | 273.384 ms | 397.203 ms | 68.8% | threads=2560 grid=[64;1;1] block=[40;1;1] ops=3 |

For scale, lenet's seg23 does 17x more statements with 8x more threads in 17.0 ms. On `cc` the graph
is not fissioned at all (1 segment), so this instrument yields no attribution there.

### Conv sketches win at cifar scale on `cc`, and are absent on Metal

Fresh searches, best timed candidate:

| workload/backend | best candidate | time | untuned default |
|---|---|---|---|
| cifar_conv/cc | `F_sketch[conv-mma-cpu 0x0x0 grid, conv-mma-cpu 0x0x0 grid ep, mma-cpu 64x128x256, mma-cpu 16x0x16]` | 264.629 ms | 844.678 ms |
| cifar_stride/cc | `F_sketch[conv-mma-cpu 0x0x0 grid, conv-mma-cpu 0x0x0 grid ep, mma-cpu 64x0x64, mma-cpu 64x128x256]` | 86.043 ms | 211.720 ms |
| lenet/cc | `F_sketch[conv-mma-cpu 0x0x0 ep]` (36.442 ms) loses to the plain baseline | 14.531 ms | 23.424 ms |
| cifar_conv/metal | 0 timed conv candidates (30 failed); best is `F_preset[bs=cfg priv cfg-thresh]` | 290.334 ms | 2102.550 ms |

The `ep` (fused-epilogue twin) components appear inside the crowned composite on both cifar `cc`
cells, so the gh-ocannl-501 twins are proposed, timed and selected there.

### The analytic cost model's default-schedule gate should stay off

`--ocannl_model_default_schedule=true` vs the ordinary default, untuned, same binary
(`model_peak_flops=1.0e12`, `model_peak_memory_bandwidth=2.0e11` supplied for `cc`, which ships no
advisory constants). No cell improved: 10 of 12 are unchanged within +-0.4%, and 2 crash.

| workload | backend | gate OFF | gate ON | change |
|---|---|---|---|---|
| cifar_conv | cc | 829.418 | 829.832 | +0.0% |
| cifar_conv | metal | 2122.220 | 2122.120 | -0.0% |
| cifar_stride | cc | 211.337 | 211.278 | -0.0% |
| cifar_stride | metal | 395.229 | 395.370 | +0.0% |
| gpt2_mini | cc | 2085.460 | 2084.870 | -0.0% |
| gpt2_mini | metal | 367.169 | 367.294 | +0.0% |
| lenet | cc | 23.305 | 23.399 | +0.4% |
| lenet | metal | 502.814 | 502.961 | +0.0% |
| mlp_small | cc | 0.182 | 0.182 | +0.1% |
| mlp_small | metal | 1.478 | **crash** | n/a |
| mlp_wide | cc | 219.349 | 219.271 | -0.0% |
| mlp_wide | metal | 6.277 | **crash** | n/a |

Three separate causes, all visible in `model_default:` log lines:

1. **On the C backends the gate is structurally inert.** It engages but scores exactly one
   candidate — the default (`chose default (model 0.001788 ms; 1 scored, 0 without coverage)` on
   mlp_small/cc, `3.719763 ms; 1 scored` on lenet/cc). The sketch families never enter the scoring
   set, so "chose default" is vacuous.
2. **On Metal it scores the families properly and still prefers the default** (24 scored on
   lenet/metal, 25 on cifar_stride/metal), which is a real result rather than a no-op.
3. **Where it does pick a sketch, it picks one that cannot compile.** On mlp_small/metal it scores
   27 candidates and chooses `W_sketch[mma-gpu 16x32x0]` at `model 0.001375 ms` — the most
   optimistic score in the set, and a member of the family that fails `validate_parallel` 56 times
   in the autotune log. The roofline model rates tensorized code best and has no notion of whether
   it will compile.

The crash is a contract violation, not just a bad pick: `ocannl_config.reference` promises "any
scoring or application failure falls back to it", and `Autotune.model_default`'s guard
(`autotune.ml:2764`) wraps only `apply_action ()`. The chosen schedule is compiled at
`autotune.ml:2770`, *outside* the guard, and `validate_parallel` runs during backend compile — so an
apply-clean/compile-invalid schedule escapes. Autotune's search wraps each candidate compile, which
is why the same failures appear there as logged `FAILED` lines instead of crashes.

Calibration (`autotune_calibration_file`, Metal, non-baseline candidates): measured/model ranges
7.3x–594.7x. The cross-*family* ranking is right in both workloads, but within a family every
`bs=*` variant gets an identical model score (flops and bytes are identical; only launch geometry
differs) while measured times span 4.117–10.044 ms. The model picks the right family and is blind to
block size, which is worth 2.4x.

### Reduced precision: numerically sound, no performance win on macOS

Observed trajectory drift vs the pytorch/cpu reference, post-fix:

| workload | backend | bf16 | f16 |
|---|---|---|---|
| mlp_small | cc | 4.20e-04 | 1.53e-04 |
| mlp_small | metal | 4.20e-04 | 1.35e-04 |
| mlp_wide | cc | 2.79e-04 | 2.08e-05 |
| mlp_wide | metal | 2.79e-04 | 1.99e-05 |

Max observed is 4.20e-04 (bf16) against `PARITY_TOL_PRECISION` of 8e-2 — 190x headroom — and
1.53e-04 (f16) against 2e-2, 130x. Both constants are loose by more than two orders of magnitude;
4e-3 / 2e-3 would keep ~10x headroom. These are global constants and this is macOS-only data with
no `gpt2_mini` precision leg, so the CUDA and HIP legs should confirm before tightening.

Tightening alone is not sufficient. Before the Metal `Relu`/bf16 fix, `mlp_wide metal/bf16` scored
5.17e-03 and **passed** the 8e-2 gate while not training at all — its loss sat at ln(10) = 2.3026
with no batch-to-batch variation. The gate compares a trajectory to a reference that itself barely
moves over 24 steps, so a completely flat run clears it. A "did the loss actually move" check
catches that where no tolerance value does.
