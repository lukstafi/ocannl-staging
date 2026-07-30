# Benchmark results

platform: Linux-6.18.33.2-microsoft-standard-WSL2-x86_64-with-glibc2.43 x86_64 | ocannl commit: bd075cd5 | parity tol: 0.002 (max rel diff over first parity steps vs pytorch/cpu/eager)
> Checked-in example output (`results/` itself is generated and gitignored): the **CUDA leg** of the
> gh-ocannl-476 measurement sweep, run from scratch (wiped `autotune_cache`) on the RTX 5070 Ti that
> replaced the RTX 3050 Ti this file used to record. 80 cells, no runner failures, parity gate green
> on every cell including all four reduced-precision legs. Hardware: NVIDIA GeForce RTX 5070 Ti
> Laptop GPU (sm_120, driver 610.62 / CUDA API 13.3, toolkit 13.3) under WSL2, Intel Core Ultra 9
> 275HX (24C/24T). torch 2.13.0+cu130, tinygrad master `62273d50f` (CPU cells via the README's
> zig-cc clang stand-in; CUDA cells compile through the system nvrtc, which the 610-branch driver
> matches).
> Tuned cells use the two-pass protocol (pass 1 = the search, reported as `compile_s`; a fresh
> pass-2 process replays the cached winner for step timings). Regenerate with
> `benchmarks/.venv/bin/python benchmarks/orchestrate.py --gpu cuda --tuned --materialized --precision bf16 f16`.
>
> The `f16` step times include the dynamic-loss-scaling inf/nan gate's per-step host sync, so they
> are not methodologically comparable to the `f32` legs.
>
> Two cells are *worse* than the pre-sweep baseline in `report-cifar-cuda.md` and are not measurement
> noise — see "A CUDA default-schedule regression" below. `ocannl/cuda/default` is the affected
> variant; the materialized and tuned cells are unchanged.



## cifar_conv

| framework | backend | variant | step p50 ms | p10 | p90 | queued ms | compile s | parity |
|---|---|---|---|---|---|---|---|---|
| pytorch | cuda | eager | 1.525 | 1.356 | 1.818 | 1.298 | 0.54 | PASS (6.6e-04) |
| tinygrad | CUDA | jit | 4.612 | 4.555 | 4.821 | 4.484 | 2.97 | PASS (1.2e-04) |
| pytorch | cpu | eager | 12.088 | 10.298 | 14.666 | 13.643 | 0.19 | REF |
| ocannl | cuda | materialized | 88.991 | 86.825 | 91.031 | 88.806 | 2.13 | PASS (1.2e-04) |
| ocannl | cuda | tuned | 89.236 | 86.696 | 91.478 | 88.872 | 586.33 | PASS (1.2e-04) |
| tinygrad | CPU | jit | 445.274 | 433.741 | 454.942 | 445.009 | 3.72 | PASS (1.2e-04) |
| ocannl | cc | tuned | 892.680 | 790.938 | 983.127 | 871.982 | 392.08 | PASS (1.2e-04) |
| ocannl | cc | materialized | 1177.790 | 1169.750 | 1185.120 | 1177.330 | 3.22 | PASS (1.2e-04) |
| ocannl | cuda | default | 1282.380 | 1280.930 | 1284.980 | 1283.310 | 1.90 | PASS (1.2e-04) |
| ocannl | cc | default | 1371.450 | 1364.250 | 1380.130 | 1374.060 | 2.19 | PASS (1.2e-04) |

## cifar_stride

| framework | backend | variant | step p50 ms | p10 | p90 | queued ms | compile s | parity |
|---|---|---|---|---|---|---|---|---|
| pytorch | cuda | eager | 1.187 | 1.096 | 1.711 | 1.105 | 0.36 | PASS (6.3e-05) |
| tinygrad | CUDA | jit | 1.596 | 1.584 | 1.738 | 1.504 | 2.05 | PASS (7.1e-05) |
| pytorch | cpu | eager | 4.435 | 3.909 | 6.629 | 5.183 | 0.11 | REF |
| ocannl | cuda | tuned | 31.113 | 30.957 | 31.263 | 30.981 | 190.09 | PASS (7.1e-05) |
| ocannl | cuda | materialized | 31.186 | 30.975 | 31.298 | 31.049 | 2.08 | PASS (7.1e-05) |
| ocannl | cuda | default | 70.660 | 70.455 | 70.866 | 70.469 | 1.79 | PASS (7.1e-05) |
| tinygrad | CPU | jit | 126.803 | 120.444 | 130.339 | 125.074 | 2.86 | PASS (7.1e-05) |
| ocannl | cc | materialized | 315.025 | 311.833 | 318.076 | 314.368 | 2.63 | PASS (7.1e-05) |
| ocannl | cc | tuned | 321.279 | 280.500 | 360.142 | 319.489 | 236.92 | PASS (7.1e-05) |
| ocannl | cc | default | 355.019 | 351.188 | 358.985 | 355.526 | 2.06 | PASS (7.1e-05) |

## gpt2_mini

| framework | backend | variant | step p50 ms | p10 | p90 | queued ms | compile s | parity | tok/s |
|---|---|---|---|---|---|---|---|---|---|
| pytorch | cuda | eager | 2.604 | 2.477 | 2.864 | 2.581 | 0.26 | PASS (1.3e-07) | 393,279 |
| tinygrad | CUDA | jit | 5.382 | 5.302 | 5.649 | 5.259 | 1.85 | PASS (1.3e-07) | 190,262 |
| pytorch | cpu | eager | 29.156 | 25.504 | 32.511 | 31.062 | 0.10 | REF | 35,121 |
| ocannl | cuda | tuned | 188.905 | 188.483 | 189.188 | 188.647 | 450.32 | PASS (8.7e-07) | 5,421 |
| ocannl | cuda | default | 189.124 | 188.848 | 189.473 | 188.837 | 1.57 | PASS (8.7e-07) | 5,414 |
| ocannl | cuda | materialized | 225.194 | 224.975 | 225.661 | 224.814 | 2.69 | PASS (8.7e-07) | 4,547 |
| tinygrad | CPU | jit | 358.690 | 345.546 | 365.627 | 357.813 | 2.42 | PASS (8.7e-07) | 2,855 |
| ocannl | cc | tuned | 391.091 | 368.558 | 409.551 | 400.000 | 100.37 | PASS (8.0e-07) | 2,618 |
| ocannl | cc | default | 2126.800 | 2120.120 | 2130.120 | 2125.320 | 1.79 | PASS (8.7e-07) | 481 |
| ocannl | cc | materialized | 2330.570 | 2322.070 | 2339.230 | 2326.780 | 5.29 | PASS (8.0e-07) | 439 |

## lenet

| framework | backend | variant | step p50 ms | p10 | p90 | queued ms | compile s | parity |
|---|---|---|---|---|---|---|---|---|
| tinygrad | CUDA | jit | 0.572 | 0.563 | 0.606 | 0.470 | 3.46 | PASS (2.1e-07) |
| pytorch | cuda | eager | 1.058 | 0.984 | 1.615 | 1.110 | 0.43 | PASS (6.6e-05) |
| pytorch | cpu | eager | 1.449 | 1.353 | 2.533 | 1.806 | 0.09 | REF |
| ocannl | cuda | tuned | 11.560 | 11.392 | 11.682 | 11.446 | 71.05 | PASS (2.1e-07) |
| ocannl | cuda | materialized | 11.717 | 11.520 | 11.834 | 11.588 | 1.51 | PASS (2.1e-07) |
| tinygrad | CPU | jit | 17.393 | 16.366 | 18.521 | 16.882 | 3.75 | PASS (3.1e-07) |
| ocannl | cc | materialized | 19.497 | 19.169 | 20.951 | 19.938 | 2.56 | PASS (3.1e-07) |
| ocannl | cc | tuned | 19.654 | 19.274 | 21.621 | 20.058 | 173.61 | PASS (3.1e-07) |
| ocannl | cc | default | 34.486 | 33.824 | 36.437 | 35.146 | 1.99 | PASS (3.1e-07) |
| ocannl | cuda | default | 242.612 | 242.202 | 242.981 | 242.736 | 1.39 | PASS (2.1e-07) |

## mlp_small

| framework | backend | variant | step p50 ms | p10 | p90 | queued ms | compile s | parity |
|---|---|---|---|---|---|---|---|---|
| ocannl | cuda | bf16 | 0.103 | 0.096 | 0.140 | 0.102 | 0.29 | PASS (4.2e-04) |
| ocannl | cuda | materialized | 0.111 | 0.094 | 0.153 | 0.097 | 0.28 | PASS (2.2e-07) |
| ocannl | cuda | default | 0.112 | 0.097 | 0.144 | 0.099 | 0.27 | PASS (2.2e-07) |
| ocannl | cc | tuned | 0.112 | 0.111 | 0.125 | 0.153 | 18.44 | PASS (2.8e-07) |
| ocannl | cuda | tuned | 0.117 | 0.100 | 0.138 | 0.101 | 7.91 | PASS (2.2e-07) |
| tinygrad | CUDA | jit | 0.121 | 0.118 | 0.140 | 0.062 | 0.49 | PASS (2.8e-07) |
| pytorch | cpu | eager | 0.130 | 0.124 | 0.169 | 0.146 | 0.13 | REF |
| ocannl | cc | materialized | 0.132 | 0.130 | 0.139 | 0.133 | 0.36 | PASS (2.8e-07) |
| ocannl | cc | default | 0.136 | 0.133 | 0.143 | 0.146 | 0.35 | PASS (2.8e-07) |
| ocannl | cuda | f16 | 0.205 | 0.182 | 0.254 | 0.202 | 0.07 | PASS (1.5e-04) |
| tinygrad | CPU | jit | 0.304 | 0.294 | 0.399 | 0.284 | 0.50 | PASS (2.4e-07) |
| pytorch | cuda | eager | 0.620 | 0.549 | 0.975 | 0.678 | 0.24 | PASS (1.2e-07) |
| ocannl | cc | f16 | 1.248 | 1.233 | 1.328 | 1.300 | 0.63 | PASS (1.5e-04) |
| ocannl | cc | bf16 | 2.752 | 2.728 | 2.877 | 2.802 | 0.36 | PASS (4.2e-04) |

## mlp_wide

| framework | backend | variant | step p50 ms | p10 | p90 | queued ms | compile s | parity |
|---|---|---|---|---|---|---|---|---|
| pytorch | cuda | eager | 0.591 | 0.539 | 0.735 | 0.610 | 0.29 | PASS (1.0e-07) |
| tinygrad | CUDA | jit | 0.912 | 0.902 | 1.034 | 0.814 | 1.65 | PASS (2.1e-07) |
| pytorch | cpu | eager | 3.018 | 2.734 | 4.983 | 3.600 | 0.10 | REF |
| ocannl | cuda | bf16 | 4.109 | 4.080 | 4.346 | 4.091 | 0.34 | PASS (2.8e-04) |
| ocannl | cuda | tuned | 4.139 | 4.098 | 4.347 | 4.128 | 208.50 | PASS (5.2e-07) |
| ocannl | cuda | default | 4.167 | 4.104 | 4.343 | 4.154 | 0.14 | PASS (5.2e-07) |
| ocannl | cuda | materialized | 4.175 | 4.139 | 4.369 | 4.159 | 0.30 | PASS (5.2e-07) |
| ocannl | cuda | f16 | 13.210 | 13.042 | 13.473 | 13.175 | 0.41 | PASS (2.0e-05) |
| ocannl | cc | tuned | 22.326 | 19.897 | 25.673 | 24.529 | 47.90 | PASS (5.2e-07) |
| tinygrad | CPU | jit | 108.355 | 104.141 | 113.069 | 108.242 | 1.92 | PASS (6.2e-07) |
| ocannl | cc | materialized | 223.769 | 216.317 | 238.276 | 227.230 | 0.50 | PASS (6.2e-07) |
| ocannl | cc | default | 224.913 | 216.916 | 237.042 | 226.889 | 0.49 | PASS (6.2e-07) |
| ocannl | cc | f16 | 924.450 | 914.335 | 934.379 | 919.211 | 0.78 | PASS (2.1e-05) |
| ocannl | cc | bf16 | 1594.760 | 1583.240 | 1613.160 | 1669.500 | 0.48 | PASS (2.8e-04) |

## Findings

Instrument notes: the tensor-core check reads the crowned schedules straight out of
`autotune_cache/*.sexp` (counting `Tensorize` nodes) and greps the emitted `.cu`/`.ptx` for
`wmma::`/`mma.sync`. Validated against `test/operations/schedule_mma_matmul` run with
`OCANNL_BACKEND=cuda`, which does emit them — `mma.sync.aligned.row.row.m16n16k16.f16.f16` (6),
the inline-PTX `m16n8k32...e5m2.e5m2.f32` fp8 form (5), `...m16n16k8.f32.tf32.tf32.f32` (22) and 18
`wmma::fragment`/`load_matrix_sync` pairs — so the zero counts below are real, not a broken probe.
Candidate-level data comes from fresh searches with `autotune_log=true` and the cache redirected to
a scratch dir, because a warm cache replays the winner and logs nothing about the losers.

### Tensor cores are unreachable on CUDA too — gh-ocannl-521 is not a Metal bug

**Not one of the 12 crowned CUDA schedules in this sweep contains a `Tensorize`** (nor a `Stage`).
Five of the 12 `cc` winners do, so the CPU packed `Tile_mma` path is selected and the GPU one is
not. This is the same result the macOS leg reported, reached independently. Per workload (each has
two cached placement arms, A = default placements and B = materialize-all):

| workload | `cc` arms with a `Tensorize` | `cuda` arms with a `Tensorize` |
|---|---|---|
| mlp_wide | 2 of 2 | 0 of 2 |
| cifar_conv | 2 of 2 | 0 of 2 |
| cifar_stride | 1 of 2 | 0 of 2 |
| mlp_small | 0 of 2 | 0 of 2 |
| lenet | 0 of 2 | 0 of 2 |
| gpt2_mini | 0 of 2 | 0 of 2 |

(Digest-to-workload attribution by observing which `autotune_cache/*.sexp` a warm replay reads.)
The `cc` pattern is coherent with the macOS leg's: the packed CPU `Tile_mma` is selected exactly
where the workload is GEMM- or conv-GEMM-dominated at a size worth panel-packing, and lenet's
smaller convs decline it there as they did on Apple silicon.

The mechanism is also the same. Tensorized candidates *are* seeded, and none is ever timed —
every one fails at candidate compile, with zero declines and zero dedups:

| blocker (mma-labelled candidates only) | mlp_wide | lenet | cifar_conv | gpt2_mini | Metal lenet |
|---|---|---|---|---|---|
| `Low_level.validate_parallel`: write to materialized node not nested under loops covering all active hardware dims | 36 | 12 | 12 | — | 12 |
| `Schedule.Stage`: source is written in the routine / reads not using identical index vectors | 9 | 9 | 9 | — | 9 |
| `Schedule.Fuse_epilogue`: accumulator is a whole-K `Tile_mma`; stage the reduction (split K) first | 10 | — | — | — | 10 |
| `Schedule.Fuse_epilogue`: guarded writes of the reduction output unsupported | 3 | 9 | 9 | — | 9 |
| `Schedule`: workgroup-shared tile budget exceeded (~1.05 MB staged) | 6 | — | — | — | — |
| `Schedule.Tensorize`: loop must be exactly the body of loop (a perfectly nested i x j x k micro-kernel) | — | — | — | 3 | — |
| `Autotune sketch`: only rank-2 outputs in v1 | — | — | — | 3 | — |
| **total proposed / ever timed** | **64 / 0** | **30 / 0** | **30 / 0** | **6 / 0** | **—/0** |

**130 mma candidates proposed across four workloads, none ever timed.** The lenet column is
numerically identical to Metal's, and the dominant missing-hardware-dimension set is the same string
in the same proportion (`Workgroup slot 0, Grid slot 0, Grid slot 1`, 30x on mlp_wide). Two GPU
backends failing identically points at one schedule-construction defect upstream of both, so
gh-ocannl-521 should be re-scoped from "Metal" to "all GPU backends".

`gpt2_mini` is worth separating out: it is the matmul-dominated workload this whole arc was meant to
serve, and it is the *only* one whose candidates die before the hardware-dimension check, on two
earlier limits — attention's rank-3 outputs (`only rank-2 outputs in v1`) and a micro-kernel nesting
requirement. Only 6 candidates are proposed there at all, against 30-64 elsewhere. Even a complete
fix for gh-ocannl-521 would leave `gpt2_mini` unable to reach tensor cores.

The non-tensorized GPU sketch family fares little better: on mlp_wide/cuda, 20 timed, 55 failed,
18 dedup, with `Fuse_epilogue: output loop must be Serial or Grid` (20) and hardware-dimension
coverage (20) leading. Hardware-dimension coverage is blocking considerably more than the
tensorized flavors alone.

### At plain f32, CUDA seeds no tensorized candidate at all

With `tf32_matmuls` off (the default), `Autotune.mma_input_formats_of_prec` yields `[Mma_f32]`, and
`cuda_backend.ml`'s `mma_format_tiles` advertises only f16/bf16/fp8/tf32 pairs — no `Mma_f32` entry.
`mma_tile_for_precisions` returns `None`, so **zero mma candidates are seeded** and the labels are
the plain `F_sketch[gpu 32x32x16/2x2]` family. CUDA at uniform f32 is in the same position the plan
recorded for HIP (RDNA3.5 WMMA has no f32-input shape); Metal's genuine `Mma_f32` simdgroup tile is
the outlier. The 64/30 counts above therefore required `--ocannl_tf32_matmuls=true`: on CUDA that
flag is what makes the tensor-core question askable at all, not merely a performance leg.

Because `tf32_matmuls` is consulted in exactly two places — the seeding predicate above and the
`mma_syntax` wmma-config arm — it cannot change generated code unless a `Tile_mma` survives to
rendering. While every seeded candidate fails to compile, **the tf32 leg is a no-op by
construction**, which is what the measurements show (mlp_wide/cuda tuned 4.096 ms with tf32 vs
4.106 ms without, inside run-to-run variance). Re-run this leg once gh-ocannl-521 is fixed; until
then it measures nothing.

The bf16 and f16 legs are not an alternative route to the question: `bench_mlp` rejects
`BENCH_TUNE` combined with `BENCH_PRECISION`, so the reduced-precision cells cannot be tuned. bf16
is the one configuration whose native 16x16x16 tile *is* advertised independently of the tf32
policy, and the harness cannot exercise it. This matters more for the HIP leg, where bf16 is the
only tensor-core route at all.

### The CUDA untuned default is dominated by one backward reduction — more so than on Metal

`BENCH_SEG_TIMES=1` on `bench_conv_diag`, default pipeline, min-of-N per segment. A single segment
writing `bias_conv1.grad` / `n65.grad` — the first conv layer's *backward* reduction, not the
forward GEMM — accounts for nearly the whole step:

| workload | seg22 | total | share | (macOS/Metal share) |
|---|---|---|---|---|
| lenet/cuda | 240.684 ms | 254.503 ms | **94.6%** | 91.0% |
| cifar_conv/cuda | 1183.165 ms | 1294.583 ms | **91.4%** | 82.1% |
| cifar_stride/cuda | 48.680 ms | 78.719 ms | **61.8%** | 68.8% |

The per-segment sums track the measured `ocannl/cuda/default` cells closely (254.5 vs 242.6,
1294.6 vs 1282.4, 78.7 vs 70.7 ms), so the attribution is trustworthy.

lenet's seg22 geometry is `threads=1792 grid=[64;1;1] block=[28;1;1] ops=3 stmts=2` — *byte-identical
to the geometry the macOS leg recorded*, since it comes from fission rather than from the backend.
Its loop nest is `loops[64s,28s,28s,6s]`, i.e. **entirely Serial**: 301,056 writes to `n65.grad`
plus a 6-element `bias_conv1.grad` reduction, driven by 1792 threads. Its neighbour seg23 does 17
statements across 14,400 threads in 6.5 ms.

This confirms gh-ocannl-484's priority on a second backend and removes the "maybe it's a Metal
scheduling quirk" reading. It is a statement about the **default schedule only**, though:
materializing the intermediates sidesteps seg22 entirely (lenet/cuda 242.6 ms default → 11.7 ms
materialized, 20.7x), which is why `cuda/materialized` and `cuda/tuned` are within 1% of each other
on every conv workload here. The tuned conv win on CUDA is a placement decision, not a
schedule-search result.

Since the materialized cells are the ones the cross-framework table below compares, they need their
own attribution — `BENCH_MATERIALIZE=1 BENCH_SEG_TIMES=1`:

| workload | top segment | | 2nd | | forward convs | total |
|---|---|---|---|---|---|---|
| lenet/cuda | seg34 `kernel_conv1.grad` | 7.481 ms (59.7%) | seg31 `kernel_conv2.grad bias_conv2.grad` | 3.461 ms (27.6%) | 0.27 ms (2.1%) | 12.53 ms |
| cifar_conv/cuda | seg34 `kernel_conv1.grad bias_conv1.grad` | 60.487 ms (56.2%) | seg31 `kernel_conv2.grad bias_conv2.grad` | 18.496 ms (17.2%) | 16.48 ms (15.3%) | 107.54 ms |
| cifar_stride/cuda | seg34 `kernel_conv1.grad bias_conv1.grad` | 16.564 ms (57.9%) | seg31 `kernel_conv2.grad bias_conv2.grad` | 3.954 ms (13.8%) | 5.42 ms (18.9%) | 28.62 ms |

seg22 is gone, and a **different** conv-backward segment takes over: the *weight* gradient
(`kernel_conv1.grad`) rather than the input gradient plus bias reduction. The top two segments are
both backward reductions and together are 71–87% of the materialized step, while the forward convs
are 2–19%. So the shape of the problem survives materialization even though the specific segment
does not — which is what makes gh-ocannl-484's subject matter (split reductions, two-pass tree
combines) relevant to the *best* cells and not only to the default ones.

### A CUDA default-schedule regression since the checked-in cifar baseline

`report-cifar-cuda.md` recorded the same GPU at commit `1f77a985`. The `ocannl/cuda/default` cells
have gotten substantially worse, while materialized and tuned are unchanged:

| cell | baseline (`1f77a985`) | now (`bd075cd5`) | |
|---|---|---|---|
| cifar_conv cuda/default | 534.207 ms | 1282.380 ms | **2.40x worse** |
| cifar_stride cuda/default | 52.680 ms | 70.660 ms | **1.34x worse** |
| cifar_conv cuda/materialized | 93.190 ms | 88.991 ms | 1.05x better |
| cifar_stride cuda/materialized | 31.741 ms | 31.186 ms | 1.02x better |

The baseline was recorded on an older driver, so the tree and the driver both changed. Rebuilding
`1f77a985` in a scratch worktree and re-running the same instrument on *this* driver separates them.
The regression is present in all three conv workloads and is confined to seg22:

| workload | seg22 baseline tree | seg22 current tree | factor | total baseline | total current |
|---|---|---|---|---|---|
| lenet/cuda | 103.958 ms | 240.684 ms | **2.32x** | 116.82 ms | 254.50 ms |
| cifar_conv/cuda | 453.617 ms | 1183.165 ms | **2.61x** | 544.00 ms | 1294.58 ms |
| cifar_stride/cuda | 26.721 ms | 48.680 ms | **1.82x** | 55.41 ms | 78.72 ms |

`cifar_stride` in full, showing that nothing else moved:

| segment | baseline tree, old driver | baseline tree, this driver | current tree, this driver |
|---|---|---|---|
| seg22 `bias_conv1.grad n65.grad` | 23.383 ms | 26.721 ms | **48.680 ms** |
| seg20 `bias_conv2.grad n79/n80.grad` | 1.788 ms | 1.785 ms | 2.360 ms |
| seg23 `kernel_conv1.grad` | 16.977 ms | 16.354 ms | 17.170 ms |
| seg21 `kernel_conv2.grad ...` | 3.315 ms | 3.309 ms | 3.303 ms |
| seg4 `n79/n80 relu max_pool2d` | 4.731 ms | 4.731 ms | 4.690 ms |
| seg19 `w_fc1.grad b_fc1.grad ...` | 1.095 ms | 1.112 ms | 1.093 ms |
| **total** | 52.79 ms | 55.41 ms | 78.72 ms |

Every segment except seg22 and seg20 is unchanged to within 1%. seg22 is **1.82x slower on the
current tree at identical launch geometry** (`threads=2048 grid=[64;1;1] block=[32;1;1] ops=3
stmts=2` in both), so the difference is in the emitted body of that segment, not in scheduling
shape, hardware, or driver. The driver upgrade accounts for a separate ~14% on that segment.

Filed as gh-ocannl-527. Bisected over the range (on `lenet`, matching the segment by its written `bias_conv1.grad` rather
than by index, since numbering shifts): the first bad commit is **`ee313a09` "tropical/einmax1:
exact gradients via the product-space gate (gh-ocannl-512)"** — 103.569 ms at its parent
`953fc917`, 227.207 ms at `ee313a09`.

This is not an accidental regression: gh-ocannl-512 deliberately moved the gradient gate into the
(output x window) product space, "one bit per (result, contracted position) pair instead of a
last-write-wins bit per input position", which buys exact gradients for *overlapping* pooling
windows (`stride < window_size`) and removes the documented g2 limitation. The extra work is the
price of that exactness, and `max_pool2d`'s backward is what seg22 is computing.

The point worth acting on is who pays it. **All three conv benchmarks pool with
`~stride:2 ~window_size:2` — non-overlapping — where the previous last-write-wins gate was already
exact.** They pay the full product-space cost for a correctness property their geometry cannot
exercise. A `stride >= window_size` specialization back to the cheaper gate would recover ~2.3x on
the segment that is already 91-95% of these steps; the one semantic difference to preserve is tie
handling (the product-space gate credits every achieving pair, the old one credited the last write).

That interacts directly with gh-ocannl-484: this segment is both the largest single cost in the
matrix and now 2.3x more expensive than when the conv baseline was recorded.

### The analytic cost model's default-schedule gate should stay off — and no longer crashes

`--ocannl_model_default_schedule=true` vs the ordinary default, untuned, same binary, cuda column
(the CUDA backend ships advisory envelope constants, so unlike `cc` the gate engages without
supplied numbers):

| workload | gate off | gate on | delta |
|---|---|---|---|
| mlp_small | 0.110 ms | 0.106 ms | -3.6% |
| mlp_wide | 4.201 ms | 4.201 ms | 0.0% |
| lenet | 256.693 ms | 255.705 ms | -0.4% |
| cifar_conv | 1306.160 ms | 1306.480 ms | +0.02% |
| cifar_stride | 71.117 ms | 71.038 ms | -0.1% |
| gpt2_mini | 190.858 ms | 190.160 ms | -0.4% |

Six of six unchanged (mlp_small's -3.6% is 4 µs at a 0.1 ms scale). Same verdict as macOS —
**the gate should stay off** — but with one difference worth recording: the macOS leg saw 2 of 12
cells *crash* under the gate, which is what gh-ocannl-519 and gh-ocannl-522 were filed for. Both
are closed, and no cell crashed here, so those fixes hold on the backend where the underlying
`validate_parallel` failure is most common.

### Reduced precision: numerically sound, and on CUDA not a performance loss

Observed maximum relative drift over the parity window, for gh-ocannl-523's constants:

| workload | backend | bf16 | f16 |
|---|---|---|---|
| mlp_small | cc | 4.203e-04 | 1.532e-04 |
| mlp_small | cuda | 4.203e-04 | 1.509e-04 |
| mlp_wide | cc | 2.791e-04 | 2.078e-05 |
| mlp_wide | cuda | 2.791e-04 | 2.031e-05 |

The proposed tightening (`bf16 -> 4e-3`, `f16 -> 2e-3`) keeps ~9.5x and ~13x headroom over the
largest drift seen here, so this leg **confirms it from a second platform**. The bf16 figures are
*identical* between `cc` and `cuda` to all four digits; f16 differs only in the third significant
figure (the loss-scaling gate's host round-trip is the plausible source). Per the handoff's rule
that a backend-specific divergence is itself a bug signal: there is none.

Performance, unlike macOS, is not a loss on the GPU:

| cell | f32 default | bf16 | f16 |
|---|---|---|---|
| mlp_small/cuda | 0.112 ms | **0.103 ms** | 0.205 ms |
| mlp_wide/cuda | 4.167 ms | **4.109 ms** | 13.210 ms |
| mlp_small/cc | 0.136 ms | 2.752 ms | 1.248 ms |
| mlp_wide/cc | 224.913 ms | 1594.760 ms | 924.450 ms |

bf16 on CUDA is a slight *win* on both MLPs — `mlp_small/cuda/bf16` at 0.103 ms is the fastest cell
in that workload across all three frameworks. That is a storage/bandwidth effect, not tensor cores:
no bf16 cell renders an mma intrinsic (they cannot even be tuned). On `cc` bf16 costs 7.1x and 20x,
consistent with macOS's 5.3x and with the C backend having no native bf16 arithmetic. f16 is
3.2x/1.8x slower on CUDA; its step times include the per-step inf/nan sync, but as on macOS that
constant cannot explain a penalty that scales with problem size.

### Cross-framework: the conv gap is backward reductions, and it dwarfs the tensor-core question

Best OCANNL cell vs the best competitor, this matrix:

| workload | OCANNL best (GPU) | torch cuda | tinygrad CUDA | gap | OCANNL best (`cc`) | torch cpu | tinygrad CPU | gap |
|---|---|---|---|---|---|---|---|---|
| mlp_small | 0.103 (bf16) | 0.620 | 0.121 | **OCANNL wins** | 0.112 | 0.130 | 0.304 | **OCANNL wins** |
| mlp_wide | 4.109 (bf16) | 0.591 | 0.912 | 7.0x | 22.326 | 3.018 | 108.355 | 7.4x (beats tinygrad 4.9x) |
| gpt2_mini | 188.905 | 2.604 | 5.382 | **72.5x** | 391.091 | 29.156 | 358.690 | 13.4x |
| lenet | 11.560 | 1.058 | 0.572 | 20.2x | 19.497 | 1.449 | 17.393 | 13.5x |
| cifar_conv | 88.991 | 1.525 | 4.612 | 58.4x | 892.680 | 12.088 | 445.274 | 73.8x |
| cifar_stride | 31.113 | 1.187 | 1.596 | 26.2x | 321.279 | 4.435 | 126.803 | 72.4x |

Three things this changes:

1. **`gpt2_mini` is 72.5x off torch cuda**, wider than the ~54x the macOS leg measured against torch
   mps, and neither tuning (189.1 -> 188.9 ms, 450 s of search for 0.1%) nor materialization
   (225.2 ms, worse) moves it. It is the workload the tensor-core arc was supposed to serve and it
   has no reduced-precision leg, so it runs f32 in every column of this sweep. It remains the
   largest unexplained gap in the matrix.
2. **The conv gap lives in the backward reductions, not in the forward GEMMs.** OCANNL is 20-74x
   off both frameworks on convs, on both columns. Note the "OCANNL best" cells above are the
   materialized/tuned ones, which *already* sidestep the default schedule's seg22 — so this gap is
   emphatically **not** explained by seg22, and the default-schedule pathology (94.6% of lenet,
   91.4% of cifar_conv, but only 61.8% of cifar_stride) is a separate problem from it. Attributing
   the materialized cells instead (table above) puts 71-87% of the remaining step in two
   *weight-gradient* reduction segments, against 2-19% in the forward convs. Even zeroing the
   single largest of them would leave lenet ~8.8x off tinygrad, so no one segment closes this gap —
   but its character is consistently reduction shape rather than GEMM quality, which is why
   gh-ocannl-484 looks like the lever here and tensor cores do not.
3. **The MLP cells are already competitive**, and OCANNL wins `mlp_small` outright on both columns.
   The CPU column beats tinygrad CPU on both MLPs (4.9x on `mlp_wide`) while losing to it on every
   conv — the same segment showing up on `cc`, where the graph is not fissioned and this instrument
   gives no attribution.

### gh-ocannl-502's strided seeding does not reproduce its `cc` win on this CPU

macOS measured `cifar_stride/cc` tuned at 86.0 ms against a 211.7 ms default (2.5x) and
`cifar_conv/cc` at 264.6 vs 844.7 (3.1x). On this 24-thread Intel box the same searches give:

| cell | default | materialized | tuned | tuned speedup |
|---|---|---|---|---|
| cifar_stride/cc | 355.019 ms | 315.025 ms | 321.279 ms | **1.10x** |
| cifar_conv/cc | 1371.450 ms | 1177.790 ms | 892.680 ms | **1.54x** |

`cifar_stride/cc` tuned is fractionally *slower* than materialized (321.3 vs 315.0 ms, inside its
own p10-p90 of 280.5-360.1), i.e. the search found nothing better than placement. The absolute
numbers are also far worse than Apple silicon's (892.7 vs 264.6 ms tuned on `cifar_conv`), so this
is a machine difference and not only a tuning one, but the conclusion for gh-ocannl-502 is that its
win is not portable across CPUs as measured.

### Housekeeping

`SKIP_CELLS` contains only `("cifar_conv", "metal", "tuned")`; there is no CUDA-analogous entry, and
every CUDA cell in this matrix completed, so nothing needed skipping here.
