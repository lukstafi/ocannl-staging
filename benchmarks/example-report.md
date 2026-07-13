# Benchmark results

platform: macOS-26.5.2-arm64-arm-64bit-Mach-O arm64 | ocannl commit: fc1e45af | parity tol: 0.002 (max rel diff over first parity steps vs pytorch/cpu/eager)
> Checked-in example output (`results/` itself is generated and gitignored): full matrix with
> fresh autotune caches after the placement-A/B + privatized-candidates tuning rework (PR #140);
> tuned cells use the two-pass protocol (pass 1 = the search, reported as `compile_s`; a fresh
> pass-2 process replays the cached winner for step timings). Tuned is the fastest OCANNL variant
> in every cell. Regenerate with
> `benchmarks/.venv/bin/python benchmarks/orchestrate.py --tuned --materialized`.


## gpt2_mini

| framework | backend | variant | step p50 ms | p10 | p90 | queued ms | compile s | parity | tok/s |
|---|---|---|---|---|---|---|---|---|---|
| tinygrad | METAL | jit | 3.662 | 3.165 | 3.764 | 2.954 | 0.69 | PASS (1.3e-07) | 279,600 |
| pytorch | mps | eager | 5.905 | 5.002 | 6.997 | 5.256 | 0.05 | PASS (1.3e-07) | 173,410 |
| pytorch | cpu | eager | 13.457 | 13.323 | 13.671 | 13.543 | 0.02 | REF | 76,095 |
| tinygrad | CPU | jit | 45.157 | 44.748 | 45.377 | 45.221 | 0.79 | PASS (8.0e-07) | 22,676 |
| ocannl | metal | tuned | 92.707 | 92.545 | 92.907 | 88.181 | 1947.18 | PASS (2.0e-07) | 11,046 |
| ocannl | metal | default | 367.015 | 365.519 | 367.335 | 361.560 | 0.23 | PASS (2.0e-07) | 2,790 |
| ocannl | cc | tuned | 381.585 | 379.912 | 382.755 | 381.118 | 150.71 | PASS (8.1e-07) | 2,684 |
| ocannl | metal | materialized | 399.998 | 398.475 | 400.709 | 393.773 | 0.31 | PASS (8.1e-07) | 2,560 |
| ocannl | cc | default | 2085.600 | 2084.080 | 2090.320 | 2085.820 | 3.65 | PASS (8.1e-07) | 491 |
| ocannl | cc | materialized | 2261.310 | 2258.720 | 2263.830 | 2262.010 | 14.97 | PASS (8.1e-07) | 453 |

## lenet

| framework | backend | variant | step p50 ms | p10 | p90 | queued ms | compile s | parity |
|---|---|---|---|---|---|---|---|---|
| tinygrad | METAL | jit | 1.176 | 0.764 | 1.313 | 0.572 | 1.30 | PASS (2.1e-07) |
| pytorch | mps | eager | 5.639 | 5.495 | 5.830 | 4.635 | 0.10 | PASS (1.0e-07) |
| tinygrad | CPU | jit | 5.726 | 5.602 | 5.812 | 5.674 | 1.31 | PASS (3.1e-07) |
| pytorch | cpu | eager | 12.527 | 12.245 | 12.775 | 12.516 | 0.02 | REF |
| ocannl | cc | tuned | 14.269 | 14.196 | 14.365 | 14.313 | 21.86 | PASS (2.1e-07) |
| ocannl | cc | materialized | 14.301 | 14.207 | 14.373 | 14.314 | 1.65 | PASS (2.1e-07) |
| ocannl | cc | default | 18.988 | 18.911 | 19.062 | 18.958 | 1.38 | PASS (2.1e-07) |
| ocannl | metal | tuned | 36.612 | 36.316 | 37.129 | 34.950 | 72.69 | PASS (2.1e-07) |
| ocannl | metal | materialized | 37.819 | 37.443 | 38.327 | 36.191 | 0.55 | PASS (2.1e-07) |
| ocannl | metal | default | 216.387 | 215.266 | 217.781 | 214.629 | 0.55 | PASS (2.1e-07) |

## mlp_small

| framework | backend | variant | step p50 ms | p10 | p90 | queued ms | compile s | parity |
|---|---|---|---|---|---|---|---|---|
| ocannl | cc | tuned | 0.129 | 0.107 | 0.148 | 0.097 | 7.70 | PASS (2.2e-07) |
| ocannl | cc | materialized | 0.177 | 0.155 | 0.216 | 0.140 | 0.46 | PASS (2.2e-07) |
| pytorch | cpu | eager | 0.178 | 0.161 | 0.221 | 0.184 | 0.01 | REF |
| ocannl | cc | default | 0.182 | 0.158 | 0.247 | 0.143 | 0.45 | PASS (2.2e-07) |
| tinygrad | CPU | jit | 0.302 | 0.295 | 0.314 | 0.307 | 0.32 | PASS (2.8e-07) |
| tinygrad | METAL | jit | 0.823 | 0.311 | 0.942 | 0.610 | 0.35 | PASS (2.4e-07) |
| pytorch | mps | eager | 1.134 | 0.990 | 1.235 | 0.279 | 0.07 | PASS (1.2e-07) |
| ocannl | metal | tuned | 1.273 | 1.162 | 1.368 | 0.490 | 1.17 | PASS (3.2e-07) |
| ocannl | metal | materialized | 1.300 | 1.213 | 1.383 | 0.484 | 0.01 | PASS (3.2e-07) |
| ocannl | metal | default | 1.508 | 1.326 | 1.893 | 0.584 | 0.01 | PASS (3.2e-07) |

## mlp_wide

| framework | backend | variant | step p50 ms | p10 | p90 | queued ms | compile s | parity |
|---|---|---|---|---|---|---|---|---|
| pytorch | mps | eager | 1.079 | 0.941 | 1.131 | 0.346 | 0.08 | PASS (2.1e-07) |
| tinygrad | METAL | jit | 1.168 | 0.744 | 1.290 | 1.056 | 0.52 | PASS (2.1e-07) |
| pytorch | cpu | eager | 1.608 | 1.582 | 1.634 | 1.608 | 0.01 | REF |
| ocannl | metal | tuned | 4.144 | 4.091 | 4.313 | 3.131 | 510.25 | PASS (5.2e-07) |
| ocannl | metal | materialized | 5.946 | 5.850 | 6.287 | 4.954 | 0.02 | PASS (5.2e-07) |
| ocannl | metal | default | 6.281 | 6.168 | 6.591 | 5.274 | 0.02 | PASS (5.2e-07) |
| tinygrad | CPU | jit | 13.441 | 13.220 | 13.842 | 13.485 | 0.49 | PASS (7.3e-07) |
| ocannl | cc | tuned | 49.034 | 48.413 | 49.787 | 49.091 | 14.55 | PASS (5.2e-07) |
| ocannl | cc | materialized | 218.940 | 218.124 | 219.787 | 218.973 | 0.57 | PASS (6.2e-07) |
| ocannl | cc | default | 219.023 | 218.179 | 220.031 | 218.995 | 0.54 | PASS (6.2e-07) |
