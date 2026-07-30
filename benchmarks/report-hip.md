# Benchmark results

platform: Linux-6.18.33.2-microsoft-standard-WSL2-x86_64-with-glibc2.43 x86_64 | ocannl commit: bd075cd5 | parity tol: 0.002 (max rel diff over first parity steps vs pytorch/cpu/eager)
> The HIP/ROCm leg of the gh-ocannl-476 measurement sweep, run from scratch (wiped
> `autotune_cache`) on minix-pc: AMD Ryzen AI Max+ 395 (Strix Halo, Zen 5, 16C/32T) with the
> Radeon 8060S iGPU (gfx1151, RDNA3.5), ROCm 7.14, **under WSL2** — the previous revision of this
> report was recorded on native Windows at `8436e362`, so individual cells are not comparable to it
> cross-revision. (The *pattern* across cells still is, and is informative: see the gh-ocannl-527
> note under per-segment attribution, where the workloads that did not regress bound the
> environment effect.) 65 cells, 2 runner failures. torch 2.13.0+rocm7.1 (HIP 7.1.52802), tinygrad
> 0.13.0, HIP runtime 7.14.60850. Tuned cells use the two-pass protocol (pass 1 = the search,
> reported as `compile_s`; a fresh pass-2 process replays the cached winner for step timings).
> Regenerate with
> `taskset -c 0-15 benchmarks/.venv/bin/python benchmarks/orchestrate.py --tuned --materialized --precision bf16 f16 --gpu hip`
> — the affinity wrapper is not optional here, see the next paragraph.
>
> **All cells ran under `taskset -c 0-15`** (16 physical cores, SMT halves excluded): this machine
> hard-froze during an uncapped all-core `cc` autotune search earlier in the session (Kernel-Power
> 41, no bugcheck; the box also has four `0x7F` double-fault dumps from 2026-07-12), and the cap is
> the agreed mitigation. The `cc` column therefore reflects 16 threads, not the machine's 16C/32T.
>
> **WSL changes what the GPU column can contain.** Both PyTorch and tinygrad reach the 8060S here,
> which native Windows could not do — so this is the first HIP matrix with cross-framework GPU
> cells. Two environment notes: torch's bundled `libhsa-runtime64.so` must be replaced with
> `/opt/rocm/lib/libhsa-runtime64.so.1.21.0` (the wheel's is the KFD build; WSL has no `/dev/kfd`)
> with `/opt/rocm/lib/rocm_sysdeps/lib` on `LD_LIBRARY_PATH`; and tinygrad's `AMD` device is
> likewise unusable, so the GPU column uses its `HIP` device (orchestrate.py falls back
> automatically when `/dev/kfd` is absent).
>
> **Most `pytorch/cuda(hip)` cells are numerically invalid on this hardware — check each cell's
> parity column before reading it as a measurement.** Torch ROCm's column-sum reduction returns
> garbage non-deterministically on gfx1151 (details in Findings), which NaNs the bias gradient and
> stops training. The parity gate caught 5 of the 6: four as `loss stationary`, `gpt2_mini` as a
> 3.6e+34 drift. Their step times are left in the tables for completeness but describe a computation
> that is not the benchmark. The one exception, **`mlp_small` (PASS, 1.7e-07)**, is a valid
> measurement — its reduction shape falls in the clean size range. **tinygrad/HIP is the GPU
> cross-framework reference that is valid across all six workloads.**
>
> Three HIP tuned cells are missing, each for a different reason: `mlp_wide` is in `SKIP_CELLS`
> (search wedges in the middle-end), `cifar_conv` was killed by a 50-minute watchdog (same wedge),
> and `gpt2_mini` died on an HSA `scratch_size overflow` abort. See Findings.
>
> `gpt2_mini` has no reduced-precision leg (inference-only, ndarray-backed `Specified` weights need
> forward-only cast insertion), so the matmul-dominated workload runs f32 in every column — the
> gap most relevant to this issue's stated purpose. The `f16` step times include the dynamic
> loss-scaling gate's per-step host sync and are not methodologically comparable to the `f32` legs.

## cifar_conv

| framework | backend | variant | step p50 ms | p10 | p90 | queued ms | compile s | parity |
|---|---|---|---|---|---|---|---|---|
| pytorch | cuda(hip) | eager | 3.557 | 3.461 | 3.766 | 3.144 | 6.34 | FAIL (inf) (loss stationary) |
| tinygrad | HIP | jit | 7.288 | 7.089 | 7.497 | 6.612 | 11.46 | PASS (2.1e-07) |
| pytorch | cpu | eager | 17.034 | 14.684 | 21.126 | 18.748 | 0.39 | REF |
| ocannl | hip | materialized | 68.424 | 67.183 | 69.822 | 68.379 | 2.79 | PASS (3.1e-07) |
| tinygrad | CPU | jit | 75.791 | 73.471 | 80.847 | 75.984 | 3.78 | PASS (3.1e-07) |
| ocannl | cc | tuned | 533.234 | 498.511 | 565.366 | 538.515 | 371.28 | PASS (3.1e-07) |
| ocannl | cc | materialized | 1168.120 | 1163.330 | 1174.830 | 1174.190 | 3.44 | PASS (3.1e-07) |
| ocannl | cc | default | 1322.210 | 1315.470 | 1331.190 | 1323.640 | 2.48 | PASS (3.1e-07) |
| ocannl | hip | default | 1594.360 | 1592.070 | 1596.880 | 1593.860 | 2.72 | PASS (4.1e-07) |

## cifar_stride

| framework | backend | variant | step p50 ms | p10 | p90 | queued ms | compile s | parity |
|---|---|---|---|---|---|---|---|---|
| tinygrad | HIP | jit | 1.788 | 1.737 | 1.883 | 1.560 | 6.18 | PASS (3.8e-05) |
| pytorch | cuda(hip) | eager | 2.646 | 2.558 | 2.920 | 2.108 | 2.14 | FAIL (inf) (loss stationary) |
| pytorch | cpu | eager | 6.164 | 4.867 | 7.814 | 5.470 | 0.18 | REF |
| ocannl | hip | materialized | 21.305 | 21.155 | 21.458 | 21.189 | 2.55 | PASS (3.8e-05) |
| ocannl | hip | tuned | 21.384 | 21.191 | 21.578 | 21.283 | 211.61 | PASS (3.8e-05) |
| tinygrad | CPU | jit | 39.644 | 37.490 | 41.392 | 38.777 | 2.64 | PASS (3.7e-05) |
| ocannl | cc | tuned | 186.753 | 162.243 | 220.347 | 186.863 | 226.33 | PASS (3.8e-05) |
| ocannl | hip | default | 214.487 | 213.960 | 214.875 | 214.278 | 2.26 | PASS (3.8e-05) |
| ocannl | cc | materialized | 312.064 | 311.680 | 313.510 | 311.823 | 2.97 | PASS (3.8e-05) |
| ocannl | cc | default | 349.788 | 347.371 | 351.930 | 347.969 | 2.38 | PASS (3.8e-05) |

## gpt2_mini

| framework | backend | variant | step p50 ms | p10 | p90 | queued ms | compile s | parity | tok/s |
|---|---|---|---|---|---|---|---|---|---|
| pytorch | cuda(hip) | eager | 5.964 | 5.863 | 6.203 | 5.842 | 0.35 | FAIL (3.6e+34) | 171,688 |
| tinygrad | HIP | jit | 6.241 | 6.080 | 6.307 | 5.893 | 4.07 | PASS (1.3e-07) | 164,066 |
| pytorch | cpu | eager | 27.122 | 25.430 | 28.187 | 29.686 | 0.05 | REF | 37,756 |
| ocannl | hip | default | 71.923 | 69.999 | 73.520 | 71.556 | 2.09 | PASS (2.0e-07) | 14,237 |
| ocannl | hip | materialized | 75.165 | 72.370 | 76.569 | 73.194 | 2.79 | PASS (8.7e-07) | 13,623 |
| tinygrad | CPU | jit | 100.026 | 98.763 | 101.217 | 101.953 | 1.65 | PASS (8.7e-07) | 10,237 |
| ocannl | cc | tuned | 525.591 | 518.764 | 532.349 | 530.038 | 110.00 | PASS (8.0e-07) | 1,948 |
| ocannl | cc | default | 2301.210 | 2286.880 | 2312.040 | 2293.530 | 2.03 | PASS (8.7e-07) | 445 |
| ocannl | cc | materialized | 2477.230 | 2470.960 | 2482.910 | 2474.930 | 6.29 | PASS (8.0e-07) | 413 |

## lenet

| framework | backend | variant | step p50 ms | p10 | p90 | queued ms | compile s | parity |
|---|---|---|---|---|---|---|---|---|
| tinygrad | HIP | jit | 0.872 | 0.834 | 0.976 | 0.581 | 8.46 | PASS (2.1e-07) |
| pytorch | cuda(hip) | eager | 1.721 | 1.626 | 1.904 | 1.223 | 1.19 | FAIL (1.0e-07) (loss stationary) |
| ocannl | hip | tuned | 6.943 | 6.795 | 7.134 | 6.808 | 125.96 | PASS (3.1e-07) |
| ocannl | hip | materialized | 7.165 | 6.872 | 7.516 | 7.113 | 2.29 | PASS (3.1e-07) |
| ocannl | cc | materialized | 22.152 | 22.068 | 22.321 | 22.177 | 2.84 | PASS (3.1e-07) |
| ocannl | cc | tuned | 22.340 | 22.231 | 22.506 | 22.365 | 162.63 | PASS (3.1e-07) |
| tinygrad | CPU | jit | 24.277 | 23.872 | 25.038 | 24.575 | 3.04 | PASS (3.1e-07) |
| ocannl | cc | default | 35.295 | 34.927 | 35.946 | 35.230 | 2.13 | PASS (3.1e-07) |
| pytorch | cpu | eager | 339.967 | 335.939 | 344.011 | 332.080 | 0.29 | REF |
| ocannl | hip | default | 359.769 | 359.432 | 360.211 | 359.809 | 2.21 | PASS (2.1e-07) |

## mlp_small

| framework | backend | variant | step p50 ms | p10 | p90 | queued ms | compile s | parity |
|---|---|---|---|---|---|---|---|---|
| ocannl | cc | tuned | 0.107 | 0.105 | 0.120 | 0.112 | 21.21 | PASS (2.8e-07) |
| ocannl | cc | materialized | 0.133 | 0.126 | 0.146 | 0.134 | 0.39 | PASS (2.8e-07) |
| ocannl | cc | default | 0.145 | 0.140 | 0.166 | 0.150 | 0.38 | PASS (2.8e-07) |
| ocannl | hip | tuned | 0.273 | 0.254 | 0.339 | 0.233 | 13.34 | PASS (3.0e-07) |
| ocannl | hip | materialized | 0.281 | 0.266 | 0.354 | 0.213 | 0.42 | PASS (3.8e-07) |
| ocannl | hip | default | 0.306 | 0.295 | 0.346 | 0.236 | 0.18 | PASS (3.0e-07) |
| ocannl | hip | bf16 | 0.309 | 0.271 | 0.375 | 0.276 | 0.46 | PASS (6.0e-04) |
| tinygrad | HIP | jit | 0.441 | 0.424 | 0.523 | 0.383 | 4.70 | PASS (2.0e-07) |
| ocannl | hip | f16 | 0.581 | 0.537 | 0.682 | 0.524 | 0.78 | PASS (1.5e-04) |
| pytorch | cuda(hip) | eager | 0.740 | 0.603 | 1.120 | 0.715 | 0.19 | PASS (1.7e-07) |
| tinygrad | CPU | jit | 0.797 | 0.746 | 1.179 | 0.848 | 1.21 | PASS (2.4e-07) |
| ocannl | cc | f16 | 1.121 | 1.110 | 1.159 | 1.129 | 0.71 | PASS (1.5e-04) |
| ocannl | cc | bf16 | 3.934 | 3.913 | 3.986 | 3.968 | 0.40 | PASS (4.2e-04) |
| pytorch | cpu | eager | 112.003 | 111.849 | 116.011 | 113.000 | 0.12 | REF |

## mlp_wide

| framework | backend | variant | step p50 ms | p10 | p90 | queued ms | compile s | parity |
|---|---|---|---|---|---|---|---|---|
| tinygrad | HIP | jit | 1.012 | 0.967 | 1.127 | 0.781 | 4.97 | PASS (2.1e-07) |
| pytorch | cuda(hip) | eager | 1.066 | 1.041 | 1.152 | 0.871 | 0.21 | FAIL (1.0e-07) (loss stationary) |
| ocannl | hip | materialized | 2.213 | 2.151 | 2.295 | 2.170 | 0.53 | PASS (4.2e-07) |
| ocannl | hip | default | 2.275 | 2.213 | 2.377 | 2.200 | 0.20 | PASS (4.2e-07) |
| ocannl | hip | bf16 | 3.983 | 3.781 | 4.158 | 4.035 | 0.20 | PASS (2.8e-04) |
| pytorch | cpu | eager | 6.338 | 5.147 | 13.981 | 7.236 | 0.22 | REF |
| ocannl | hip | f16 | 12.347 | 11.981 | 12.697 | 12.296 | 0.82 | PASS (2.0e-05) |
| ocannl | cc | tuned | 19.436 | 17.873 | 22.053 | 20.508 | 52.03 | PASS (5.2e-07) |
| tinygrad | CPU | jit | 36.517 | 34.314 | 41.204 | 36.629 | 1.48 | PASS (6.2e-07) |
| ocannl | cc | materialized | 217.000 | 215.334 | 223.015 | 219.494 | 0.59 | PASS (5.2e-07) |
| ocannl | cc | default | 217.200 | 215.778 | 238.186 | 289.481 | 0.58 | PASS (5.2e-07) |
| ocannl | cc | f16 | 815.527 | 811.772 | 823.776 | 814.919 | 0.89 | PASS (2.1e-05) |
| ocannl | cc | bf16 | 1733.700 | 1724.690 | 1760.580 | 1733.950 | 0.54 | PASS (2.8e-04) |

## Findings

### Tensor cores on HIP: still zero, and structurally unreachable

No tuned HIP winner in any workload contains a rocWMMA intrinsic — same headline as Metal, but the
reasons stack differently, and two of them are prior to gh-ocannl-521:

1. **At f32 there are no mma candidates to seed.** RDNA3/3.5 WMMA has only f16×f16 and bf16×bf16 at
   16×16×16, no f32-input shape, so uniform f32 stays scalar (`hip_backend.ml:531`). Every
   benchmark workload except the reduced-precision mlp legs is f32.
2. **Neither reduced-precision route can be tuned at all.** gfx1151 has two tensor-core routes, not
   one: `hip_backend.ml` supports f16×f16 → f32 (the flagship, and the combination verified on this
   chip via `schedule_mma_matmul`), f16×f16 → f16, bf16×bf16 → f32 and bf16×bf16 → bf16. Both are
   reachable only through `BENCH_PRECISION`, and `bench_mlp` fails outright on `BENCH_TUNE=1`
   together with `BENCH_PRECISION` ("not supported"). Since f32 seeds nothing (item 1), **no tuned
   HIP benchmark cell can contain a rocWMMA intrinsic regardless of the scheduling state** — the
   sweep structurally cannot deliver the gh-152/154/155 payoff measurement on this machine.
3. **gh-ocannl-521 generalizes to HIP.** In the f32 searches that do complete, the GPU sketch
   candidates are seeded and every one **fails candidate compile before being timed** — zero
   declines-to-scalar, zero timed. `mlp_small` hip/tuned (39.6 s, `autotune_log=true`) censuses as:
   10× `Schedule.Fuse_epilogue: output loop … must be Serial or Grid`, 7× `Schedule.Stage: source
   n20_relu is written in the routine`, 8× `Low_level.validate_parallel: … not nested under
   annotated loops covering all active hardware dimensions`, plus 20 `F_sketch[gpu …]`/`ep`
   variants under the same `Invalid_argument`s. The only timed candidates are the serial baseline,
   two `F_preset[bs=cfg …]` and the in-process untuned control. That is Metal's #521 profile, same
   three blocker families — **#521 is not a Metal bug; it blocks the GPU sketch arc on at least two
   of three GPU backends.**

Positive control: `schedule_mma_matmul` under `OCANNL_BACKEND=hip` passes, so intrinsic emission and
numerics are correct when hand-scheduled. The zero is a scheduling/seeding result, not a codegen one.

On `cc` the tensorized candidates *are* reachable and timed normally (`F_sketch[mma-cpu 128x128x128]`
et al.), as on macOS — the CPU half of the arc is delivering here too.

### One serial segment is 93–97% of the HIP step, worse than Metal

`BENCH_SEG_TIMES=1`, min of 20 runs per segment:

| workload | seg22 (`bias_conv1.grad`/`n65.grad`) | total | share |
|---|---|---|---|
| lenet/hip | 350.712 ms | 359.743 ms | **97.5%** |
| cifar_conv/hip | 1503.715 ms | 1575.987 ms | **95.4%** |
| cifar_stride/hip | 201.416 ms | 215.778 ms | **93.3%** |

Same segment as Metal (the first conv layer's *backward* reduction, not the forward GEMM) and an
even larger share (Metal: 91.0 / 82.1 / 68.8%). The census line names the mechanism outright —
`loops[64s,28s,28s,6s] w:bias_conv1.grad!(6)`: a **fully serial** 64×28×28×6 nest, 301,056
iterations on a single GPU thread, producing 6 output elements. Its neighbours are 2–4 orders of
magnitude faster. This is now measured on three independent GPU backends (Metal, CUDA, HIP) and is
the single highest-value target in the GPU pipeline. Note the default-schedule `hip` cells are
exactly where this shows: `lenet` hip/default is 359.8 ms vs 7.2 ms materialized (50×), `cifar_conv`
1594 vs 68 ms (23×).

**Read this section alongside gh-ocannl-527, which is probably a large part of what it measures.**
The CUDA leg traced a 2.4× regression in this same segment to gh-ocannl-512's product-space gate on
`max_pool2d` backward. This report's own history is a control for it: the previous revision of this
file was recorded at `8436e362` (pre-gh-512), and comparing `ocannl/hip/default` then vs now,
**`lenet` is 4.39× worse (82.016 → 359.769 ms) while every workload without `max_pool2d` moved only
1.07–1.61×** (`gpt2_mini` 67.418 → 71.923, `mlp_wide` 1.809 → 2.275, `mlp_small` 0.190 → 0.306). The
two runs also differ in OS, which is exactly what makes the non-pooling rows useful — they bound the
environment effect, and `lenet` carries ~3–4× on top of it. So the 93–97% shares above sit on top of
a regression, and how much of the residue is an inherent serial-reduction problem (gh-ocannl-484's
subject) will only be clear once gh-527 is fixed and this instrument is re-run. Re-running
`BENCH_SEG_TIMES=1` on `lenet/hip` is a cheap acceptance check for that fix.

### PyTorch ROCm is numerically broken on gfx1151 (not an OCANNL bug, not the WSL runtime swap)

Four of the five `pytorch/cuda(hip)` training cells fail parity as `loss stationary` (loss NaNs after
step 1) — `cifar_conv`, `cifar_stride`, `lenet`, `mlp_wide` — and `gpt2_mini` (inference, no updates)
fails at 3.6e+34 drift. **`mlp_small` passes** (1.7e-07, loss moving), and that exception is itself
confirming rather than contradicting: its hidden width is 64, so its bias-gradient reduction is
`[64,64]`, inside the clean size range of the sweep below. Reduced outside the harness to a **pure
reduction**, no benchmark code involved:

```
torch.randn(256,1024).cuda().sum(0)   # 10 identical runs, max|result|:
6.425e+37, 6.425e+37, 1.283e+38, 6.425e+37, 53.82, 1.283e+38, 53.82, 1.283e+38, 53.82, 6.425e+37
```

Non-deterministic — sometimes the correct 53.82, sometimes garbage — i.e. a race in torch's
column-reduction kernel. It is size-dependent: `[256,1024]`, `[128,1024]`, `[64,1024]`, `[1024,1024]`
are affected; `[32,1024]`, `[256,512]`, `[256,256]`, `[256,10]` are clean. Forward GEMMs and `dx`/`dW`
gradients are all correct to ~2e-4; only the reduction is wrong, which is why it surfaces as the bias
gradient and kills training.

Two controls establish it is torch's kernel and not this environment: **tinygrad on the same swapped
HSA runtime is correct and bit-repeatable** on the identical shapes (54.821 across 8 runs), and every
OCANNL `hip` cell passes the parity gate — including the conv workloads, whose bias gradients are the
same reduction shape. Until this is fixed upstream, torch/ROCm numbers from gfx1151 should be treated
as suspect, and this report's torch GPU column is not a measurement.

It is not an isolated report: gfx1151 has a documented family of torch numerical failures —
[ROCm/ROCm#6034](https://github.com/ROCm/ROCm/issues/6034) (5 bf16 bugs found across 93 training
experiments on this exact SoC), [ROCm/TheRock#5259](https://github.com/ROCm/TheRock/issues/5259)
(kernels aborting with `HSA_STATUS_ERROR_EXCEPTION`, with inf/nan appearing upstream of them), and
[unslothai/unsloth#3385](https://github.com/unslothai/unsloth/issues/3385) (NaN losses from step 1).
Stable ROCm does not ship gfx1151 kernels at all, so consumer Strix Halo runs on nightly builds. The
case here is narrower and easier to act on than those: **f32, not bf16, and a two-line repro against
`Tensor.sum(0)` with no model involved** — worth filing upstream on its own.

This is also a second independent vindication of gh-ocannl-523's did-the-loss-move check: without it,
four cells with NaN losses would have scored a *passing* max-rel-diff and been published as if they
had trained.

### Two new HIP tuning bugs

- **Middle-end wedge in the tuned search** (`mlp_wide`, `cifar_conv`) — gh-ocannl-532. The search hangs before its
  first candidate: constant ~3-thread CPU spin, **zero** `autotune_log` lines after `arm A search:`,
  and with `output_debug_files_in_build_directory=true` **zero** debug artifacts written — so it never
  reaches rendering or hiprtc, even for the serial baseline. Reproduced 3× (>65 min, >8 min bounded,
  and the 50-min watchdog kill). `mlp_small` hip/tuned completes in ~40 s; `mlp_wide` hip/default is
  fine (2.3 ms/step); `mlp_wide` cc/tuned searches normally. The native-Windows report has
  `mlp_wide hip/tuned` completing at `8436e362` (compile_s 111.78 s), so this is either a regression
  since the gh-502 seeding wave or WSL-specific. Not bisected.
- **HSA scratch abort escapes as a fatal error** (`gpt2_mini` hip/tuned) — gh-ocannl-533. A candidate requests
  `private_seg_size=163856` per work-item; the runtime aborts the queue with
  `[UpdateScratch] scratch_size overflow!` / `HSA_STATUS_ERROR_INVALID_ARGUMENT` on kernel
  `cross_entropy_loss_fwd`, and the failure propagates out of `hip_stream_synchronize` and kills the
  process instead of being caught as a declined candidate. An unsatisfiable-scratch candidate should
  be declined like any other compile/launch failure, not take the run down.

### Reduced precision: drift confirms macOS, performance does not improve

| workload | backend | bf16 drift | f16 drift |
|---|---|---|---|
| mlp_small | cc | 4.20e-04 | 1.53e-04 |
| mlp_small | hip | 5.96e-04 | 1.50e-04 |
| mlp_wide | cc | 2.79e-04 | 2.08e-05 |
| mlp_wide | hip | 2.79e-04 | 1.97e-05 |

**No backend-specific divergence**: hip matches cc to three significant figures on `mlp_wide`
(2.79e-04 both; f16 1.97e-05 vs 2.08e-05) — the same agreement macOS reported post-fix, which is the
signal that no backend is quietly computing something else. Max observed anywhere is bf16 5.96e-04
and f16 1.53e-04, against the current gates of 4e-3 and 2e-3 — 6.7× and 13× headroom, so the
constants could tighten further (bf16 → 1e-3, f16 → 5e-4 would still hold 1.7× and 3.3×); worth
deciding once the CUDA leg supplies its numbers, since the constants are global.

Performance repeats the macOS story: reduced precision is not a win. On `cc`, bf16 costs 8.0× f32 on
`mlp_wide` (1733.7 vs 217.2 ms) and f16 3.8× (815.5 ms) — consistent with the C backend having no
native reduced-precision arithmetic. On `hip`, bf16 is 1.8× f32 (3.98 vs 2.28 ms) and f16 5.4×
(12.35 ms); the f16 figure includes the loss-scaling sync, but that is a roughly constant per-step
cost and does not explain a 10 ms gap that scales with problem size.

### Cross-framework conclusions

Against the valid GPU reference (tinygrad/HIP), OCANNL's best `hip` cell is 8.0–11.9× off on the conv
and GPT workloads (`lenet` 6.94 vs 0.87 ms, `cifar_conv` 68.4 vs 7.29, `cifar_stride` 21.3 vs 1.79,
`gpt2_mini` 71.9 vs 6.24), and **ahead** on `mlp_small` (0.273 vs 0.441 ms) where launch overhead
dominates; `mlp_wide` is 2.2× off. The `gpt2_mini` gap against a *working* GPU framework is 11.5×
here versus the ~54× off torch/mps quoted for macOS — a materially better standing, though the
comparison is against a different framework on different hardware. Given that 93–97% of the conv
step is one serial backward-reduction segment, the conv gaps are substantially that one segment.

On CPU, OCANNL `cc` tuned wins `mlp_small` (0.107 ms vs tinygrad 0.797) and `lenet` (22.3 vs 24.3),
and loses `mlp_wide` to torch while beating tinygrad (19.4 ms vs torch 6.34 / tinygrad 36.5).

**The torch CPU reference is not a usable performance baseline in this environment**, though it
remains valid as the *parity* oracle (loss values, not times). Its timings are internally
inconsistent: `mlp_small` costs 112.0 ms/step while the far larger `mlp_wide` costs 6.34 ms — the
small workload is 18× *slower* than the wide one on the same device. That inversion (tiny tensors,
16 threads, spin-wait thread thrash in the ROCm build's CPU path) also makes `lenet`'s 340 ms and
`cifar_conv`'s 17.0 ms mutually incomparable. Any OCANNL-vs-torch CPU ratio quoted from this report
should be treated as an artifact of the reference, not a result; tinygrad/CPU is the sound CPU
comparison here.

### Still open

- The tensor-core measurement this issue exists for **has not been made on HIP** and cannot be until
  the mixed-precision graph is tunable (item 2 above) — gh-ocannl-529, which the CUDA leg filed and
  which is the *sole* blocker for this column since HIP has no tf32 escape hatch.
- `SKIP_CELLS` audit: the stale `cifar_conv metal/tuned` entry was not re-tested here (no Metal
  hardware); the new `mlp_wide hip/tuned` entry is freshly justified above.
- The middle-end wedge is unbisected; a `git bisect` against `8436e362` on native Windows vs WSL
  would separate "regression" from "WSL-specific".
