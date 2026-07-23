# Benchmark results — cifar-scale conv baseline (Linux/CUDA)

> Checked-in baseline record for the gh-ocannl-500 blocking decision and the gh-ocannl-502
> seeding wave (`results/` itself is generated and gitignored). Untuned matrix
> (`--materialized`, no `--tuned`): the strided `cifar_stride` stem exercises only the
> reorder-serial and default-fissioned paths — the compacting-Stage seeding is measured
> against exactly these numbers. Hardware: WSL2 on an RTX 5070 Ti Laptop GPU (driver CUDA
> 13.0), 24-thread CPU. tinygrad is master `62273d50f` (CPU cells via the README's zig-cc
> clang stand-in; CUDA cells with torch's bundled nvrtc 13.0, see Setup).

platform: Linux-6.18.33.2-microsoft-standard-WSL2-x86_64-with-glibc2.43 x86_64 | ocannl commit: 1f77a985 (pre-rebase; = the cifar_stride + compacting-Stage series) | parity tol: 0.002 (max rel diff over first parity steps vs pytorch/cpu/eager)

## cifar_conv

| framework | backend | variant | step p50 ms | p10 | p90 | queued ms | compile s | parity |
|---|---|---|---|---|---|---|---|---|
| pytorch | cuda | eager | 1.379 | 1.334 | 1.617 | 1.312 | 0.36 | PASS (6.1e-04) |
| tinygrad | CUDA | jit | 4.678 | 4.637 | 4.850 | 4.561 | 2.85 | PASS (1.2e-04) |
| pytorch | cpu | eager | 12.652 | 10.931 | 17.218 | 14.070 | 0.13 | REF |
| ocannl | cuda | materialized | 93.190 | 92.999 | 93.662 | 92.705 | 2.08 | PASS (1.2e-04) |
| tinygrad | CPU | jit | 430.368 | 425.834 | 438.472 | 430.524 | 4.66 | PASS (1.2e-04) |
| ocannl | cuda | default | 534.207 | 533.540 | 535.519 | 534.522 | 1.79 | PASS (1.2e-04) |
| ocannl | cc | materialized | 1159.460 | 1151.380 | 1168.210 | 1160.660 | 2.85 | PASS (1.2e-04) |
| ocannl | cc | default | 1340.610 | 1329.540 | 1352.770 | 1333.390 | 2.14 | PASS (1.2e-04) |

## cifar_stride

| framework | backend | variant | step p50 ms | p10 | p90 | queued ms | compile s | parity |
|---|---|---|---|---|---|---|---|---|
| pytorch | cuda | eager | 1.206 | 1.097 | 1.652 | 1.276 | 0.36 | PASS (4.7e-05) |
| tinygrad | CUDA | jit | 1.524 | 1.516 | 1.678 | 1.426 | 2.05 | PASS (7.1e-05) |
| pytorch | cpu | eager | 4.167 | 3.659 | 6.013 | 5.025 | 0.13 | REF |
| ocannl | cuda | materialized | 31.741 | 31.604 | 31.880 | 31.564 | 2.04 | PASS (7.1e-05) |
| ocannl | cuda | default | 52.680 | 52.541 | 52.788 | 52.480 | 1.77 | PASS (7.1e-05) |
| tinygrad | CPU | jit | 120.598 | 118.346 | 123.560 | 120.061 | 3.18 | PASS (7.1e-05) |
| ocannl | cc | materialized | 308.707 | 303.336 | 313.926 | 309.268 | 2.47 | PASS (7.1e-05) |
| ocannl | cc | default | 347.204 | 342.881 | 353.140 | 344.883 | 2.10 | PASS (7.1e-05) |

## Per-layer breakdown, cifar_stride on cuda (default schedule)

`BENCH_FIXTURE=fixtures/cifar_stride.safetensors BENCH_SEG_TIMES=1 bench_conv_diag.exe
--ocannl_backend=cuda` — min of 20 runs per segment, one routine per fission segment. The
stride-2 stem's backward dominates the step; the forward convs are minor. This is the
breakdown the compacting-Stage seeding and the gh-500 blocking flavors should move.

```
seg1   N   0.7210 ms  conv1 fwd (+relu, pool)        <- the strided site, forward
seg4   N   4.7313 ms  conv2 fwd (+relu, pool)
seg19  N   1.0949 ms  fc1 backward
seg20  N   1.7878 ms  conv2 bias/input grads
seg21  N   3.3152 ms  conv2 kernel grad
seg22  N  23.3826 ms  conv1 input grad (+bias grad)  <- the strided site, backward
seg23  N  16.9767 ms  conv1 kernel grad              <- the strided site, backward
seg24  N   0.0624 ms  SGD update
(+ 17 segments below 0.15 ms each)
total (sum of per-segment minima): 52.79 ms
```
