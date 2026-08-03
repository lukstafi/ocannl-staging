# Benchmark results

platform: Linux-6.18.33.2-microsoft-standard-WSL2-x86_64-with-glibc2.43 x86_64 | ocannl commit: e687da82 | parity tol: 0.002 (max rel diff over first parity steps vs pytorch/cpu/eager; reduced precisions get their own envelope: bf16 0.004, f16 0.002)
> Checked-in example output (`results/` itself is generated and gitignored): the **CUDA leg** of the
> gh-ocannl-538 re-measurement sweep, superseding this file's gh-ocannl-476 contents. One commit,
> one session: everything below was measured at `e687da82`, which is `origin/master` and contains
> gh-ocannl-521, -527, -532, -537, -539, -540 and -541. `benchmarks/autotune_cache` did not exist
> when the sweep started, so every reported search is from scratch and nothing was replayed from a
> cache predating those changes.
> The matrix dispatched 96 cells: **87 produced a result and 9 failed**, all nine of them `gpt2_mini`
> reduced-precision or tuned cells — see "Three defects in the gpt2_mini legs". The workload tables
> below carry **88 rows**: those 87 plus one standalone re-measurement of `gpt2_mini ocannl/cuda/tuned`
> (marked †), which failed inside the matrix and was re-run alone at the same commit in the same
> session. The parity gate is green on every cell that produced a result, including all
> reduced-precision legs.
> Hardware: NVIDIA GeForce RTX 5070 Ti Laptop GPU (sm_120, driver 610.62 / CUDA API 13.3, toolkit
> 13.3) under WSL2, Intel Core Ultra 9 275HX (24C/24T). torch 2.13.0+cu130, tinygrad master
> `62273d50f` (CPU cells via the README's zig-cc clang stand-in; CUDA cells compile through the
> system nvrtc, which the 610-branch driver matches). Same machine as
> [report-gh484-cuda.md](report-gh484-cuda.md) and [report-gh537-cuda.md](report-gh537-cuda.md).
> Tuned cells use the two-pass protocol (pass 1 = the search, reported as `compile_s`; a fresh
> pass-2 process replays the cached winner for step timings). Regenerate with
> `benchmarks/.venv/bin/python benchmarks/orchestrate.py --gpu cuda --tuned --materialized --precision bf16 f16`.
>
> The `f16` step times include the dynamic-loss-scaling inf/nan gate's per-step host sync, so they
> are not methodologically comparable to the `f32` legs. The `f16-static` and `f16-gated16` columns
> in "Reduced precision" exist precisely to price that gate; read them before drawing any conclusion
> from an `f16` row.
>
> Per gh-ocannl-538's reporting contract, no number here is compared against the checked-in
> pre-gh-521 tables this file used to contain: different candidate pools, different serial-baseline
> behaviour, and a changed pooling gradient make such a comparison meaningless. The one before/after
> claim in this report is a same-session interleaved A/B with both binaries built and both commits
> named.

## cifar_conv

| framework | backend | variant | precision | step p50 ms | p10 | p90 | queued ms | compile s | parity |
|---|---|---|---|---|---|---|---|---|---|
| pytorch | cuda | eager | f32 | 1.393 | 1.340 | 1.939 | 1.267 | 0.52 | PASS (6.8e-04) |
| tinygrad | CUDA | jit | f32 | 4.624 | 4.547 | 4.798 | 4.440 | 1.95 | PASS (1.2e-04) |
| pytorch | cpu | eager | f32 | 10.406 | 9.666 | 14.595 | 12.217 | 0.20 | REF |
| ocannl | cuda | tuned | f32 | 82.277 | 82.138 | 82.986 | 82.615 | 121.63 | PASS (1.2e-04) |
| ocannl | cuda | materialized | f32 | 95.222 | 86.407 | 95.765 | 92.953 | 1.77 | PASS (1.2e-04) |
| ocannl | cc | tuned | f32 | 181.351 | 167.562 | 219.576 | 207.575 | 658.48 | PASS (1.2e-04) |
| tinygrad | CPU | jit | f32 | 448.600 | 438.752 | 456.842 | 444.625 | 2.27 | PASS (1.2e-04) |
| ocannl | cuda | default | f32 | 535.806 | 526.229 | 546.146 | 535.463 | 1.62 | PASS (1.2e-04) |
| ocannl | cc | materialized | f32 | 1168.280 | 1156.520 | 1180.050 | 1169.380 | 3.18 | PASS (1.2e-04) |
| ocannl | cc | default | f32 | 1349.550 | 1335.870 | 1362.140 | 1345.490 | 2.13 | PASS (1.2e-04) |

## cifar_stride

| framework | backend | variant | precision | step p50 ms | p10 | p90 | queued ms | compile s | parity |
|---|---|---|---|---|---|---|---|---|---|
| pytorch | cuda | eager | f32 | 1.114 | 1.059 | 1.323 | 1.146 | 0.35 | PASS (4.2e-05) |
| tinygrad | CUDA | jit | f32 | 1.584 | 1.576 | 1.628 | 1.482 | 2.04 | PASS (7.1e-05) |
| pytorch | cpu | eager | f32 | 3.993 | 3.759 | 4.794 | 4.616 | 0.14 | REF |
| ocannl | cuda | tuned | f32 | 31.246 | 31.155 | 31.437 | 31.132 | 93.09 | PASS (7.1e-05) |
| ocannl | cuda | materialized | f32 | 31.602 | 31.454 | 31.752 | 31.428 | 1.72 | PASS (7.1e-05) |
| ocannl | cuda | default | f32 | 52.099 | 51.985 | 52.348 | 51.943 | 1.55 | PASS (7.1e-05) |
| ocannl | cc | tuned | f32 | 53.552 | 50.132 | 80.250 | 56.787 | 380.84 | PASS (7.1e-05) |
| tinygrad | CPU | jit | f32 | 121.459 | 118.592 | 128.692 | 123.823 | 2.08 | PASS (7.1e-05) |
| ocannl | cc | materialized | f32 | 311.322 | 307.854 | 321.843 | 308.756 | 2.44 | PASS (7.1e-05) |
| ocannl | cc | default | f32 | 349.314 | 345.026 | 354.918 | 348.671 | 2.06 | PASS (7.1e-05) |

## gpt2_mini

Rows are grouped by precision (f32 first), p50-ascending within each group.

| framework | backend | variant | precision | step p50 ms | p10 | p90 | queued ms | compile s | parity | tok/s |
|---|---|---|---|---|---|---|---|---|---|---|
| pytorch | cuda | eager | f32 | 2.594 | 2.470 | 2.716 | 2.619 | 0.31 | PASS (1.3e-07) | 394,709 |
| tinygrad | CUDA | jit | f32 | 5.303 | 5.244 | 5.526 | 5.151 | 1.08 | PASS (1.3e-07) | 193,092 |
| pytorch | cpu | eager | f32 | 30.820 | 27.243 | 33.393 | 38.744 | 0.12 | REF | 33,225 |
| ocannl | cuda | default | f32 | 186.013 | 185.740 | 186.494 | 185.811 | 1.12 | PASS (8.7e-07) | 5,505 |
| ocannl | cuda | tuned † | f32 | 185.910 | 185.690 | 186.483 | 185.658 | 379.38 | PASS (8.7e-07) | 5,508 |
| ocannl | cuda | materialized | f32 | 221.877 | 221.498 | 222.250 | 221.607 | 1.98 | PASS (8.7e-07) | 4,615 |
| tinygrad | CPU | jit | f32 | 346.827 | 343.672 | 352.588 | 347.618 | 1.57 | PASS (8.7e-07) | 2,952 |
| ocannl | cc | tuned | f32 | 401.816 | 362.361 | 439.352 | 389.537 | 468.11 | PASS (8.0e-07) | 2,548 |
| ocannl | cc | default | f32 | 2110.940 | 2101.110 | 2121.800 | 2104.520 | 1.82 | PASS (8.7e-07) | 485 |
| ocannl | cc | materialized | f32 | 2301.610 | 2297.170 | 2329.320 | 2303.750 | 4.94 | PASS (8.0e-07) | 445 |
| ocannl | cuda | materialized | bf16 | 239.554 | 239.293 | 239.946 | 239.330 | 3.16 | PASS (1.0e-03) | 4,275 |
| ocannl | cc | tuned | bf16 | 2980.930 | 2937.950 | 3027.210 | 2984.080 | 1856.32 | PASS (1.0e-03) | 344 |
| ocannl | cc | default | bf16 | 20312.100 | 20255.800 | 20343.300 | 20276.700 | 1.94 | PASS (9.6e-04) | 50 |
| ocannl | cc | materialized | bf16 | 21202.900 | 21164.700 | 21254.200 | 21220.800 | 3.75 | PASS (1.0e-03) | 48 |

† `gpt2_mini ocannl/cuda/tuned` failed inside the matrix run with `CUDA_ERROR_OUT_OF_MEMORY` and was
re-measured standalone on an idle GPU in the same session, at the same commit, with the same
two-pass protocol. It is the one cell in this report not produced by the single `orchestrate.py`
invocation; see "Three defects in the gpt2_mini legs" for why that distinction matters.
`cuda/{default,tuned}/bf16` and every `f16` cell of this workload failed on both backends and are
absent by measurement, not by omission.

## lenet

| framework | backend | variant | precision | step p50 ms | p10 | p90 | queued ms | compile s | parity |
|---|---|---|---|---|---|---|---|---|---|
| tinygrad | CUDA | jit | f32 | 0.565 | 0.554 | 0.578 | 0.464 | 2.04 | PASS (2.1e-07) |
| pytorch | cuda | eager | f32 | 1.054 | 0.980 | 1.710 | 1.251 | 0.42 | PASS (6.6e-05) |
| pytorch | cpu | eager | f32 | 1.607 | 1.458 | 2.049 | 1.741 | 0.11 | REF |
| ocannl | cc | tuned | f32 | 4.834 | 4.620 | 7.200 | 20.802 | 325.01 | PASS (2.1e-07) |
| ocannl | cuda | tuned | f32 | 9.536 | 9.502 | 9.632 | 9.537 | 78.45 | PASS (2.1e-07) |
| ocannl | cuda | materialized | f32 | 11.698 | 11.663 | 11.879 | 11.634 | 1.47 | PASS (2.1e-07) |
| tinygrad | CPU | jit | f32 | 17.344 | 16.082 | 18.683 | 16.797 | 2.05 | PASS (3.1e-07) |
| ocannl | cc | materialized | f32 | 19.303 | 18.473 | 21.249 | 19.284 | 2.30 | PASS (3.1e-07) |
| ocannl | cc | default | f32 | 32.117 | 30.936 | 34.545 | 32.190 | 2.01 | PASS (3.1e-07) |
| ocannl | cuda | default | f32 | 114.861 | 114.709 | 115.668 | 114.756 | 1.37 | PASS (2.1e-07) |

## mlp_small

Rows are grouped by precision (f32 first), p50-ascending within each group.

| framework | backend | variant | precision | step p50 ms | p10 | p90 | queued ms | compile s | parity |
|---|---|---|---|---|---|---|---|---|---|
| ocannl | cuda | tuned | f32 | 0.098 | 0.085 | 0.133 | 0.098 | 12.12 | PASS (2.2e-07) |
| ocannl | cuda | materialized | f32 | 0.106 | 0.093 | 0.142 | 0.092 | 0.10 | PASS (2.2e-07) |
| ocannl | cc | tuned | f32 | 0.111 | 0.107 | 0.115 | 0.120 | 42.87 | PASS (2.8e-07) |
| ocannl | cuda | default | f32 | 0.111 | 0.091 | 0.137 | 0.089 | 0.10 | PASS (2.2e-07) |
| ocannl | cc | materialized | f32 | 0.132 | 0.129 | 0.137 | 0.153 | 0.32 | PASS (2.8e-07) |
| tinygrad | CUDA | jit | f32 | 0.134 | 0.129 | 0.144 | 0.064 | 0.51 | PASS (2.8e-07) |
| pytorch | cpu | eager | f32 | 0.135 | 0.127 | 0.166 | 0.156 | 0.09 | REF |
| ocannl | cc | default | f32 | 0.137 | 0.134 | 0.145 | 0.144 | 0.31 | PASS (2.8e-07) |
| tinygrad | CPU | jit | f32 | 0.304 | 0.291 | 0.381 | 0.250 | 0.49 | PASS (2.4e-07) |
| pytorch | cuda | eager | f32 | 0.606 | 0.532 | 0.986 | 0.692 | 0.24 | PASS (1.2e-07) |
| ocannl | cuda | materialized | bf16 | 0.106 | 0.094 | 0.145 | 0.098 | 0.28 | PASS (5.2e-04) |
| ocannl | cuda | tuned | bf16 | 0.114 | 0.100 | 0.143 | 0.101 | 19.57 | PASS (5.8e-04) |
| ocannl | cuda | default | bf16 | 0.119 | 0.104 | 0.147 | 0.111 | 0.31 | PASS (5.4e-04) |
| ocannl | cc | tuned | bf16 | 0.719 | 0.632 | 0.905 | 0.829 | 29.61 | PASS (5.6e-04) |
| ocannl | cc | materialized | bf16 | 2.787 | 2.729 | 3.033 | 2.898 | 0.31 | PASS (5.2e-04) |
| ocannl | cc | default | bf16 | 3.062 | 3.001 | 3.675 | 3.162 | 0.31 | PASS (5.4e-04) |
| ocannl | cuda | tuned | f16 | 0.207 | 0.186 | 0.250 | 0.195 | 24.98 | PASS (1.4e-04) |
| ocannl | cuda | default | f16 | 0.214 | 0.189 | 0.297 | 0.193 | 0.42 | PASS (1.4e-04) |
| ocannl | cuda | materialized | f16 | 0.221 | 0.189 | 0.271 | 0.190 | 0.43 | PASS (1.6e-04) |
| ocannl | cc | tuned | f16 | 0.533 | 0.498 | 0.637 | 0.569 | 29.68 | PASS (8.1e-05) |
| ocannl | cc | default | f16 | 1.446 | 1.389 | 1.600 | 1.492 | 0.57 | PASS (1.4e-04) |
| ocannl | cc | materialized | f16 | 1.453 | 1.429 | 1.511 | 1.486 | 0.57 | PASS (1.5e-04) |

## mlp_wide

Rows are grouped by precision (f32 first), p50-ascending within each group.

| framework | backend | variant | precision | step p50 ms | p10 | p90 | queued ms | compile s | parity |
|---|---|---|---|---|---|---|---|---|---|
| pytorch | cuda | eager | f32 | 0.655 | 0.563 | 0.875 | 0.652 | 0.23 | PASS (1.0e-07) |
| tinygrad | CUDA | jit | f32 | 0.909 | 0.901 | 0.929 | 0.800 | 0.77 | PASS (2.1e-07) |
| ocannl | cuda | tuned | f32 | 1.368 | 1.354 | 1.395 | 1.339 | 17.10 | PASS (5.2e-07) |
| pytorch | cpu | eager | f32 | 3.224 | 2.942 | 4.205 | 3.650 | 0.12 | REF |
| ocannl | cuda | default | f32 | 4.121 | 4.109 | 4.182 | 4.086 | 0.13 | PASS (5.2e-07) |
| ocannl | cuda | materialized | f32 | 4.130 | 4.114 | 4.186 | 4.090 | 0.14 | PASS (5.2e-07) |
| ocannl | cc | tuned | f32 | 20.880 | 20.097 | 25.881 | 23.473 | 92.38 | PASS (1.1e-07) |
| tinygrad | CPU | jit | f32 | 107.914 | 105.248 | 110.852 | 106.782 | 0.88 | PASS (6.2e-07) |
| ocannl | cc | materialized | f32 | 225.244 | 216.512 | 244.937 | 237.052 | 0.47 | PASS (6.2e-07) |
| ocannl | cc | default | f32 | 225.297 | 216.373 | 248.921 | 234.924 | 0.45 | PASS (6.2e-07) |
| ocannl | cuda | tuned | bf16 | 1.129 | 1.118 | 1.162 | 1.096 | 30.34 | PASS (2.5e-04) |
| ocannl | cuda | materialized | bf16 | 3.966 | 3.950 | 4.011 | 3.927 | 0.37 | PASS (2.5e-04) |
| ocannl | cuda | default | bf16 | 3.980 | 3.964 | 4.035 | 3.932 | 0.35 | PASS (2.7e-04) |
| ocannl | cc | tuned | bf16 | 218.085 | 211.953 | 257.453 | 231.451 | 259.78 | PASS (2.6e-04) |
| ocannl | cc | default | bf16 | 1622.820 | 1605.410 | 1662.050 | 1666.160 | 0.43 | PASS (2.7e-04) |
| ocannl | cc | materialized | bf16 | 1623.910 | 1601.240 | 1670.030 | 1667.140 | 0.44 | PASS (2.6e-04) |
| ocannl | cuda | tuned | f16 | 4.407 | 4.366 | 4.483 | 4.396 | 42.59 | PASS (1.7e-05) |
| ocannl | cuda | materialized | f16 | 12.998 | 12.923 | 13.236 | 13.087 | 0.39 | PASS (2.1e-05) |
| ocannl | cuda | default | f16 | 13.107 | 12.931 | 13.367 | 13.060 | 0.43 | PASS (2.0e-05) |
| ocannl | cc | tuned | f16 | 93.646 | 92.031 | 113.247 | 105.567 | 205.06 | PASS (1.7e-05) |
| ocannl | cc | materialized | f16 | 925.311 | 912.752 | 952.799 | 943.382 | 0.78 | PASS (2.0e-05) |
| ocannl | cc | default | f16 | 928.266 | 915.024 | 957.127 | 950.333 | 0.76 | PASS (2.0e-05) |

# Findings

## Tensor cores are reached on CUDA — gh-ocannl-521 is fixed here

Counts are **seeded and timed** separately, per gh-ocannl-538 contract items 4 and 8, from fresh
searches with `autotune_log=true` into a scratch cache directory (a warm cache replays the winner
and logs nothing about the losers). `A` = the default-placements arm, `B` = the materialize-all
arm; `tune_placements` searches both and keeps the measured winner, so a cell's totals are A+B.

| cell | A seeded/timed | B seeded/timed | **total** | crowned label | scalar-fallback notes |
|---|---|---|---|---|---|
| mlp_wide cuda f32, `tf32_matmuls=false` (default) | 0 / 0 | 0 / 0 | **0 / 0** | `F_sketch[gpu 32x32x16/2x2, gpu 64x64x8/4x4]` | — |
| mlp_wide cuda f32, `tf32_matmuls=true` | 29 / 11 | 37 / 21 | **66 / 32** | **`F_sketch[mma-gpu 32x32x0, mma-gpu 32x32x0]`** | 0 |
| mlp_small cuda f32, `tf32_matmuls=true` | 19 / 14 | 29 / 17 | **48 / 31** | `F_split[w1.grad …]` (A wins, 0.083 vs 0.156) | 0 |
| mlp_wide cuda bf16 | 0 / 0 | 36 / 20 | **36 / 20** | `F_sketch[gpu 32x32x16/2x2, gpu 64x64x8/4x4]` | **20** |
| mlp_small cuda bf16 | 0 / 0 | 29 / 17 | **29 / 17** | `F_split[n22_cast.grad …]` (A wins) | **17** |
| mlp_wide cuda f16 | 0 / 0 | 37 / 21 | **37 / 21** | `F_split[…]` (A wins) | 0 |
| mlp_wide cc f32 | 27 / 8 | 85 / 13 | **112 / 21** | **`F_sketch[mma-cpu 16x0x16]`** (B) | 0 |
| lenet / cifar_conv / cifar_stride / gpt2_mini, cuda f32 (default config) | 0 / 0 | 0 / 0 | **0 / 0** | see split-reduce section | — |

`mlp_wide/cuda` at f32 with tf32 enabled seeds **66** tensorized candidates and **times 32** of
them, and an `mma-gpu` candidate is crowned in *both* arms. The emitted code carries the real
instruction, not a label: `build_files/bench_mlp/*.cu` for that winner contains
`mma.sync.aligned.row.col.m16n16k8.f32.tf32.tf32.f32` (8 occurrences), 6 `wmma::fragment` /
`wmma::load_matrix_sync` pairs, 2 `wmma::store_matrix_sync`, and `wmma::__float_to_tf32`. Replaying
the `tf32_matmuls=false` and `bf16` winners through the same probe yields **zero** matches, so the
probe discriminates.

**Contract item 4's two failure modes both occur here, and they are not interchangeable.** At f32
with tf32 off, nothing is *seeded*: `Autotune.mma_input_formats_of_prec` yields `[Mma_f32]`,
`cuda_backend.ml`'s `mma_format_tiles` advertises no `Mma_f32` entry, so `mma_tile_for_precisions`
returns `None`. At bf16, by contrast, 36 candidates are seeded and 20 are **timed** — and every one
of those 20 carries `NOTE 1/1 Tile_mma statement(s) rendered as the lane-0 scalar fallback`. Same
for all 17 on `mlp_small`. A "did tensor cores run" column reading yes/no would score bf16 as a
success; it measured scalar code. This is the single most important reason the contract forbids
that column.

Consequences worth acting on:

1. **`tf32_matmuls` is no longer a no-op.** The previous state of this file recorded it as inert by
   construction, because no tensorized candidate survived compile. It now changes both the crowned
   schedule and the time: `mlp_wide/cuda` arm B best goes 1.3074 ms (off) → 1.1526 ms (on), −11.8%,
   in the same process pair of searches. The shipping matrix above was run at the **default**
   (tf32 off), i.e. it does *not* include this win — a user gets 1.368 ms today and would get
   roughly 1.15 ms with the flag.
2. **bf16's mma route on CUDA is nominally reachable and actually dead.** It is the one route that
   does not need a precision policy opt-in, and it is silently degrading to scalar. Whatever rule
   declines at emission is the thing to fix; `schedule_log_declines=true` names it.
3. **`mlp_small` seeds at f32/tf32 but does not crown mma** (48 seeded, 31 timed; a split-reduce
   candidate in the default-placements arm wins at 0.083 ms against the materialize-all arm's best
   sketch at 0.156 ms) — at that size the tensorized tile is measured and loses, which is the
   mechanism behaving correctly.
4. The conv workloads and `gpt2_mini` seed **nothing** at the default config, so their tuned cells
   in the matrix above are entirely non-tensorized. Their tuning story is split reductions.

## Split reduction: gh-ocannl-541's cost ranking reaches the weight gradients

`BENCH_SR_SITES=1` on `bench_conv_diag`, and the seeded/timed/evicted census from the same fresh
searches. Sites are now ranked by estimated segment cost rather than `sr_red / sr_out`, and the cap
is `autotune_split_reduce_max_sites=8`.

| workload | sites detected | evicted by the cap | split-reduce candidates timed (per arm) |
|---|---|---|---|
| lenet | **10** | **2** (`b_logits.grad` cost 640, `cross_entropy` cost 64) | 9 |
| cifar_conv | 9 | 1 (`cross_entropy`) | 9 |
| cifar_stride | 9 | 1 (`cross_entropy`) | 9 |
| gpt2_mini | 21 | 13 | 20 |

lenet's 10 sites, cap 8, two evictions logged in the decline census is exactly the expected value in
gh-ocannl-538. The ranking now puts the *weight* gradients on top —

```
kernel_conv2.grad: reduction extent 64, target cells 2400, est. segment cost 15360000 (via 12 swaps)
kernel_conv1.grad: reduction extent 64, target cells  150, est. segment cost  7526400 (via 9 swaps)
bias_conv1.grad:   reduction extent 64, target cells    6, est. segment cost   301056 (via 3 swaps)
```

— where gh-537's leg recorded them at ratio 0 and excluded. The crowned lenet schedule in the
shipping arm is the 8-site composite `F_split[kernel_conv2.grad …, kernel_conv1.grad …,
bias_conv1.grad …, …]`. That is the mechanism behind the A/B below.

**Seeded is not timed here either**: 9 split-reduce candidates are timed per arm on the conv
workloads against 8-9 sites proposed (8 single-site plus the composite), with the evictions
recorded rather than silent — the gh-ocannl-541 blind spot this census was added to close.

`cifar_stride` now ships the **default-placements** arm (A 30.75 ms vs B 31.10 ms), as does
`gpt2_mini` (A 185.56 vs B 221.81). lenet and `cifar_conv` still ship materialize-all (9.60 vs 11.17;
82.48 vs 120.55). Which arm ships is workload-dependent and is not a property of the backend.

## Before/after: lenet's shipping artifact is −17.7%

Per contract item 3 this is a **same-session interleaved A/B with both binaries built**, not a
comparison against any checked-in table.

- **OLD** = `c1c1ea5d` (merge of PR #256), the commit immediately before gh-ocannl-537's PR #257.
- **NEW** = `e687da82`, this sweep's commit. The library delta is the whole OLD→NEW range, which
  contains gh-ocannl-537 and -541 (the two the claim is about) and also -532, -539, -540.
- Both trees built from source and kept side by side; searches interleaved OLD/NEW, then two
  replicate replays per side alternating OLD/NEW. Separate scratch cache directories per side.
- Two replicates per side is the minimum on this box (gh-537's leg saw a 6% spurious swing on a
  single cifar_stride replay).

| workload | OLD replays | NEW replays | delta (mean) | within-side spread |
|---|---|---|---|---|
| lenet | 11.581 / 11.579 ms | **9.533 / 9.537 ms** | **−17.7%** | OLD 0.01%, NEW 0.05% |
| cifar_stride | 31.171 / 31.199 ms | 31.168 / 31.206 ms | +0.0% | OLD 0.09%, NEW 0.12% |
| cifar_conv | 79.768 / 79.743 ms | 79.694 / 79.696 ms | −0.1% | OLD 0.03%, NEW 0.00% |

−17.7% against gh-ocannl-538's expected ~−17.4%, and the delta is ~150x the within-side spread. The
cifar workloads are neutral, exactly as gh-537's leg predicted from output width: their
`bias_conv1.grad` has 32 cells where lenet's has 6, and the cifar splits are declined after being
timed.

Numerics: over all 24 parity steps the two binaries are **bit-identical** on `cifar_stride` and
`cifar_conv`, and differ by at most **1.04e-07** on lenet. Only lenet crowns a different schedule,
so only lenet's combine order changes — the same signature the predecessor legs recorded.

Search cost fell sharply over the same range, which is a separate effect worth recording:

| workload | OLD search | NEW search |
|---|---|---|
| lenet | 68.7 s | 64.9 s |
| cifar_stride | 182.8 s | 72.8 s |
| cifar_conv | 552.2 s | 100.5 s |

gh-ocannl-532 (the tuner no longer dispatching an unparallelized candidate on a GPU backend) is
inside the OLD→NEW window and is the plausible cause; this A/B cannot attribute it to a single
commit, and does not try to.

## Per-segment attribution, both placements

`BENCH_SEG_TIMES=1` (and `BENCH_MATERIALIZE=1 BENCH_SEG_TIMES=1`) on `bench_conv_diag` /
`bench_gpt_diag`, min-of-20 per segment. **Every share below names the placement it is a share of**,
per contract item 1. Note which arm each workload actually ships (previous section) — for
`cifar_stride` and `gpt2_mini` the *default-placement* rows are the shipping-relevant ones, and for
lenet and `cifar_conv` the materialized rows are.

### Default placements

| workload | top segment | | 2nd | | total |
|---|---|---|---|---|---|
| lenet | seg22 `bias_conv1.grad n65.grad` | 102.480 ms (**89.0%**) | seg23 (SGD + `kernel_conv1.grad`) | 6.541 ms (5.7%) | 115.190 ms |
| cifar_conv | seg22 `bias_conv1.grad n65.grad` | 423.208 ms (**82.5%**) | seg23 `kernel_conv1.grad` | 45.465 ms (8.9%) | 512.736 ms |
| cifar_stride *(ships)* | seg22 `bias_conv1.grad n65.grad` | 23.205 ms (**44.2%**) | seg23 `kernel_conv1.grad` | 17.063 ms (32.5%) | 52.520 ms |
| gpt2_mini *(ships)* | seg111 `logits max_logits` | 15.816 ms (8.6%) | seg25 `n311 n339_gelu` | 13.290 ms (7.2%) | 183.556 ms |

### Materialize-all placements

| workload | top segment | | 2nd | | total |
|---|---|---|---|---|---|
| lenet *(ships)* | seg33 `kernel_conv1.grad bias_conv1.grad` + SGD | 7.401 ms (**59.6%**) | seg31 `kernel_conv2.grad bias_conv2.grad …` | 3.468 ms (27.9%) | 12.424 ms |
| cifar_conv *(ships)* | seg33 `kernel_conv1.grad bias_conv1.grad` | 52.425 ms (**50.5%**) | seg31 `kernel_conv2.grad bias_conv2.grad …` | 17.668 ms (17.0%) | 103.821 ms |
| cifar_stride | seg33 `kernel_conv1.grad bias_conv1.grad` | 17.832 ms (**58.0%**) | seg4 `n79 … max_pool2d` (forward) | 4.647 ms (15.1%) | 30.725 ms |
| gpt2_mini | seg116 (block FFN, `… n756_gelu`) | 17.609 ms (8.2%) | seg64 (block FFN) | 15.773 ms (7.3%) | 215.072 ms |

lenet's 59.6% and cifar_conv's 50.5% reproduce gh-537's leg (59.5% / 49.3%) closely. The
default-placement 89.0% / 82.5% figures are of a pipeline that **neither of those two workloads
ships**, which is the misreading contract item 1 exists to prevent.

`gpt2_mini` is the shape-exception: no segment exceeds 8.6% in either placement, and the cost is
spread across per-layer FFN projections plus the lm_head — consistent with gh-ocannl-531 and
unchanged by placement.

### Instrument error, per regime (contract item 2)

A seg-time total is **not** a step time. Sum of per-segment minima against the matrix's measured
untuned cell at the same placement and commit:

| regime | seg-sum | matrix cell | error |
|---|---|---|---|
| lenet, default | 115.190 ms | 114.861 ms | +0.3% |
| cifar_stride, default | 52.520 ms | 52.099 ms | +0.8% |
| gpt2_mini, default | 183.556 ms | 186.013 ms | −1.3% |
| cifar_conv, default | 512.736 ms | 535.806 ms | −4.3% |
| cifar_stride, materialized | 30.725 ms | 31.602 ms | −2.8% |
| gpt2_mini, materialized | 215.072 ms | 221.877 ms | −3.1% |
| lenet, materialized | 12.424 ms | 11.698 ms | **+6.2%** |
| cifar_conv, materialized | 103.821 ms | 95.222 ms | **+9.0%** |

The macOS leg's finding that the error is regime-dependent and worst on the materialized placement
**reproduces in sign and ordering on CUDA, at a smaller magnitude** (+6.2%/+9.0% here against
Metal's +21.9%). Shares *within* a placement remain sound, and every attribution argument above
rests only on those.

## Reduced precision: four variants, and the loss-scaling gate priced

Contract item 5's four variants. `f16` is the dynamic host-read gate (the matrix cells);
`f16-static` is `BENCH_STATIC_SCALE=1` (fixed scale, no gate, no host read); `f16-gated16` is
`BENCH_GATE_INTERVAL=16` (fused on-device gate, host samples the sticky checksum every 16 steps).
All cells two-pass where tuned. p50 ms:

| workload | be | variant | f32 | bf16 | f16 | f16-static | f16-gated16 |
|---|---|---|---|---|---|---|---|
| mlp_small | cuda | default | 0.111 | 0.119 | 0.214 | **0.131** | 0.152 |
| mlp_small | cuda | tuned | 0.098 | 0.114 | 0.207 | **0.134** | 0.165 |
| mlp_small | cc | default | 0.137 | 3.062 | 1.446 | 1.438 | 1.455 |
| mlp_small | cc | tuned | 0.111 | 0.719 | 0.533 | 0.560 | 0.586 |
| mlp_wide | cuda | default | 4.121 | 3.980 | 13.107 | **3.966** | 12.889 |
| mlp_wide | cuda | tuned | 1.368 | 1.129 | 4.407 | **0.966** | 4.325 |
| mlp_wide | cc | default | 225.297 | 1622.820 | 928.266 | 926.747 | 929.863 |
| mlp_wide | cc | tuned | 20.880 | 218.085 | 93.646 | 95.326 | 94.335 |

**The gate's cost is its on-device arithmetic, not its host round-trip — and the two workloads
separate the mechanisms.** On `mlp_wide/cuda/default`, batching the host reads 16:1 recovers 1.7%
(13.107 → 12.889 ms) while removing the gate entirely recovers **70%** (→ 3.966 ms). On
`mlp_small/cuda/default` the split reverses: batching host reads recovers 29% of a ~0.10 ms fixed
cost (0.214 → 0.152) and removing the gate only a little more (→ 0.131). So the host sync is a
per-step constant that matters only when the step is tiny; at scale, what f16 is paying for is the
gate's own work. Without the static leg those two are indistinguishable — this is gh-ocannl-535's
open question, and it now has an answer on this platform.

Follow-on: **`mlp_wide/cuda/tuned/f16-static` at 0.966 ms is the fastest OCANNL cell for that
workload in this entire report**, ahead of tuned f32 (1.368) and tuned bf16 (1.129). f16 is not
inherently a loss on CUDA; the dynamic gate as currently implemented is. Note the fused on-device
gate (gh-ocannl-250) does not close this — `f16-gated16` is within 2% of the dynamic gate on
`mlp_wide`.

On `cc`, bf16 remains a large loss (22.4x on `mlp_small/cc/default`, 7.2x on `mlp_wide/cc/default`),
consistent with the C backend having no native bf16 arithmetic. f16 on `cc` is *cheaper* than bf16
there, and the static/gated legs are flat within noise — the gate is invisible against a 900 ms step.

### Drift, for gh-ocannl-523's constants

Maximum relative drift over the parity window, worst scheduling variant per (workload, backend):

| workload | backend | bf16 | f16 | f16-static | f16-gated16 |
|---|---|---|---|---|---|
| mlp_small | cc | 5.592e-04 | 1.545e-04 | 1.35e-04 | 1.35e-04 |
| mlp_small | cuda | 5.804e-04 | 1.560e-04 | 1.47e-04 | 1.64e-04 |
| mlp_wide | cc | 2.674e-04 | 2.029e-05 | 2.03e-05 | 2.03e-05 |
| mlp_wide | cuda | 2.674e-04 | 2.053e-05 | 2.03e-05 | 1.98e-05 |
| gpt2_mini | cc | 1.027e-03 | — (does not compile) | n/a | n/a |
| gpt2_mini | cuda | 1.027e-03 | — (does not compile) | n/a | n/a |

Two things this leg adds to the two platforms that already agreed:

1. **`gpt2_mini` at bf16 drifts 1.027e-03**, an order of magnitude above anything the MLPs show and
   a workload no previous leg measured. The proposed tightening `bf16 → 4e-3` still holds, but with
   **3.9x** headroom rather than the ~9.5x the MLP-only evidence suggested. `f16 → 2e-3` keeps 12.8x.
   Anyone proposing to tighten further should use the gpt2_mini figure, not the MLP ones.
2. **No backend-specific divergence.** bf16 drift is identical between `cc` and `cuda` to four
   digits at matching scheduling variants; f16 differs only in the third significant figure. Per the
   handoff's rule that a backend divergence would itself be a bug signal: there is none. Drift does
   vary slightly *by scheduling variant* within a backend (mlp_small/cuda bf16: 5.40 / 5.22 /
   5.80e-04 for default / materialized / tuned), which is the expected combine-order effect.

`f16-static`/`f16-gated16` are `bench_mlp`-only legs: `bench_gpt` is forward-only with no optimizer,
so there is no loss scaling to gate, and the conv runner has no `BENCH_PRECISION` support at all.

## Three defects in the gpt2_mini legs

`gpt2_mini`'s reduced-precision leg had never run on a fixture before this sweep. All nine runner
failures in the matrix are here, and they are three independent defects.

**1. f16 does not compile, on any backend.** Both `cc` and `cuda`, all three scheduling variants:

```
Utils.User_error("Constant -inf is too big for FP16 aka. half precision, risk of
  overflow; increase precision of tensor node max_vals")
Utils.User_error("Constant -1000000000. is too big for FP16 ... tensor node where")
```

raised from `Low_level.simplify_llc.check_constant` (`arrayjit/lib/low_level.ml:2413`), i.e. during
lowering and before any backend sees the code — which is why it is backend-independent. The two
constants are structural to the workload: the softmax's `max_vals` `-inf` initializer and the causal
mask's `-1e9` fill. `bench_gpt.ml`'s storage policy pins the layer norms and the CE head at f32 but
not the attention softmax or the mask, so the reduced precision reaches both. This is a real gap in
the forward-only precision recipe, not a tolerance question.

**2. bf16 on CUDA hits an nvrtc codegen bug, and it is placement-dependent.**
`cross_entropy_loss_fwd__seg.cu`, 8 errors:

```
error: more than one instance of overloaded function "__hadd" matches the argument list:
   function "__hadd(int, int)" (declared implicitly)
   function "__hadd(__nv_bfloat16, __nv_bfloat16)" (cuda_bf16.h:2858)
 argument types are: (__nv_bfloat16, float)
```

The emitted call has **mixed** operand types — `__hadd((__nv_bfloat16)(1), <float expr>)` and
`__hadd(n542[...], <float expr>)`. Same family as the known `0.0h` half-precision nvrtc rejection.
The informative part: `cuda/materialized/bf16` **compiles and runs** (239.554 ms); only
`default/bf16` and `tuned/bf16` fail. So the bad call comes out of the default (virtual +
promotion) placement's fissioned cross-entropy segment, not from bf16 lowering in general.

**Fixed** (gh-ocannl-549), and the "comes out of the placement" reading turned out to be half
right: the float operand is not introduced by fission, it is the *return type of the libm call the
op table already emitted* (`expf`/`logf` in the log-sum-exp, `sqrtf` in layer_norm). Storing it
into a bf16 node is accepted — `__nv_bfloat16`'s converting constructor is implicit — so only the
placement that inlines the call into a bf16 binop trips the ambiguity, which is exactly the
materialized/default split. `gpt2_mini cuda/default/bf16` now runs at 213.5 ms/step.

**3. `gpt2_mini cuda/tuned` at f32 exhausted the 12 GB card during the search** —
`CUDA_ERROR_OUT_OF_MEMORY` from `cu_mem_alloc`, raised in `Autotune.tune.search`
(`autotune.ml:4139`). **It did not reproduce standalone.** Re-run alone on an idle GPU at the same
commit it completes: arm A 185.56 ms, arm B 221.81 ms, winner arm A, 379 s search, and the pass-2
replay gives the 185.910 ms cell marked † above. So this is a marginal-memory failure that appears
under the matrix's back-to-back cell pressure, not a deterministic regression — recorded as a
robustness problem rather than a correctness one. It is worth noting that the candidate pool this
search now walks is larger than it was: gh-ocannl-541 finds **21** split-reduce sites on this graph,
of which 8 survive the cap and 13 are evicted, and 20 split-reduce candidates are timed per arm — so
headroom on a 12 GB card is thinner than it was. gh-ocannl-521 is **not** implicated here: this cell
runs at the default `tf32_matmuls=false` (`ocannl_config.reference:497`), at which `gpt2_mini` seeds
0 and times 0 tensorized candidates (see the table in "Tensor cores are reached on CUDA"), so no
tensorized candidate reaches compile in this search at all. The split-reduction expansion is the only
supported contributor.

## Cross-framework, with per-cell caveats

Best OCANNL cell against the best competitor in each column, this matrix:

| workload | OCANNL best (GPU) | torch cuda | tinygrad CUDA | gap | OCANNL best (`cc`) | torch cpu | tinygrad CPU | gap |
|---|---|---|---|---|---|---|---|---|
| mlp_small | 0.098 (tuned) | 0.606 | 0.134 | **OCANNL wins** | 0.111 (tuned) | 0.135 | 0.304 | **OCANNL wins** |
| mlp_wide | 0.966 (tuned/f16-static) | 0.655 | 0.909 | 1.5x | 20.880 (tuned) | 3.224 | 107.914 | 6.5x |
| lenet | 9.536 (tuned) | 1.054 | 0.565 | 16.9x | 4.834 (tuned) | 1.607 | 17.344 | 3.0x |
| cifar_stride | 31.246 (tuned) | 1.114 | 1.584 | 28.1x | 53.552 (tuned) | 3.993 | 121.459 | 13.4x |
| cifar_conv | 82.277 (tuned) | 1.393 | 4.624 | 59.1x | 181.351 (tuned) | 10.406 | 448.600 | 17.4x |
| gpt2_mini | 185.910 (tuned †) | 2.594 | 5.303 | 71.7x | 401.816 (tuned) | 30.820 | 346.827 | 13.0x |

**Caveats, per contract item 6.** On this machine both GPU references are valid: torch 2.13.0+cu130
and tinygrad's CUDA device both reach the RTX 5070 Ti natively, and the parity gate passes on every
one of their cells. The caveats recorded elsewhere in this cluster do **not** apply here — PyTorch
ROCm's gfx1151 reduction race is a HIP-leg concern, and minix's torch-CPU pathology is a different
box. The one caveat that does apply is the `mlp_wide` GPU cell: OCANNL's best there is an
`f16-static` cell while torch and tinygrad run f32, so that 1.5x compares different arithmetic; at
matched f32 the gap is 2.1x (1.368 vs 0.655).

Reading:

1. **The MLP column is now competitive rather than aspirational.** OCANNL wins `mlp_small` outright
   on both columns, and `mlp_wide/cuda` is within 1.5x of torch (2.1x at matched precision).
2. **The conv gap remains, and it remains reduction shape.** The materialized-placement attribution
   puts 67.5% (cifar_conv) and 87.5% (lenet) of the shipping step in two *weight-gradient* reduction
   segments, against 1.4-15.5% in the forward convs. gh-ocannl-541 has now seeded those segments — that is what buys lenet its −17.7% —
   but `cifar_conv` at 59.1x shows the seeding alone does not close it.
3. **`gpt2_mini` is the largest unexplained gap at 71.7x**, and neither tuning (186.0 → 185.9 ms for
   379 s of search) nor materialization (221.9 ms, worse) moves it. Its cost is spread evenly across
   FFN projections with no dominant segment, so there is no single lever visible in this instrument.
   It is also the workload with all three precision defects above, so it runs f32 in every usable
   cell of this sweep.

## Housekeeping

- `SKIP_CELLS` contains only `("cifar_conv", "metal", "tuned", None)`; there is no CUDA-analogous
  entry and no CUDA cell needed skipping. The staleness retest of that Metal entry belongs to the
  Metal leg.
- Raw logs for every cell, plus the scripts that produced them, are under
  `benchmarks/results/sweep538-cuda-raw/` (gitignored): `matrix.log`, `results.jsonl`,
  `segtimes-*.log`, `srsites-*.log`, `mma-*.log`, `census-*.log`, `gate-*.log`, `gate2-*.log`,
  `ab537-*.log`.

## Reproducing

```bash
# The matrix (one commit, one session; wipe autotune_cache first for from-scratch search costs)
.venv/bin/python orchestrate.py --gpu cuda --tuned --materialized --precision bf16 f16

# Tensor-core seeded/timed counts, with the losers logged (fresh cache, or a warm one replays
# the winner and logs nothing). BENCH_TUNE_REPORT=1 prints mma_candidates / mma_timed per arm.
BENCH_FIXTURE=fixtures/mlp_wide.safetensors BENCH_TUNE=1 BENCH_TUNE_REPORT=1 \
  ../_build/default/benchmarks/runners/ocannl/bench_mlp.exe --ocannl_backend=cuda \
  --ocannl_autotune_log=true --ocannl_autotune_cache_dir=/tmp/fresh --ocannl_tf32_matmuls=true

# Does a crowned winner actually render an mma intrinsic?
... --ocannl_output_debug_files_in_build_directory=true    # then grep build_files/bench_mlp/*.cu
grep -ohE 'wmma::[a-z_]+|mma\.sync[a-z0-9.]*' build_files/bench_mlp/*.cu

# Split-reduce sites, the interchange each is reached through, and the cap's evictions
BENCH_FIXTURE=fixtures/lenet.safetensors BENCH_SR_SITES=1 \
  ../_build/default/benchmarks/runners/ocannl/bench_conv_diag.exe --ocannl_backend=cuda

# Per-segment attribution WITH the in-process step control, in each placement.
# Prefix BENCH_MATERIALIZE=1 for the materialized placement.
BENCH_FIXTURE=fixtures/lenet.safetensors BENCH_SEG_TIMES=1 BENCH_STEPS=1 \
  ../_build/default/benchmarks/runners/ocannl/bench_conv_diag.exe --ocannl_backend=cuda

# The gate-cost legs (f16 only)
BENCH_FIXTURE=fixtures/mlp_wide.safetensors BENCH_PRECISION=f16 BENCH_STATIC_SCALE=1 \
  ../_build/default/benchmarks/runners/ocannl/bench_mlp.exe --ocannl_backend=cuda
BENCH_FIXTURE=fixtures/mlp_wide.safetensors BENCH_PRECISION=f16 BENCH_GATE_INTERVAL=16 \
  ../_build/default/benchmarks/runners/ocannl/bench_mlp.exe --ocannl_backend=cuda
```

All run from `benchmarks/`.
