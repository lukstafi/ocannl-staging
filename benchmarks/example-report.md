# Benchmark results

platform: macOS-26.5.2-arm64-arm-64bit-Mach-O arm64 | ocannl commit: 2f555984 | parity tol: 0.002 (max rel diff over first parity steps vs pytorch/cpu/eager)
> Checked-in example output (`results/` itself is generated and gitignored): the first full-matrix
> run with no skip cells, after the metal default-schedule fixes (PR #138). Regenerate with
> `benchmarks/.venv/bin/python benchmarks/orchestrate.py --tuned --materialized`.


## gpt2_mini

| framework | backend | variant | step p50 ms | p10 | p90 | queued ms | compile s | parity | tok/s |
|---|---|---|---|---|---|---|---|---|---|
| tinygrad | METAL | jit | 3.231 | 3.119 | 3.792 | 2.899 | 0.69 | PASS (1.3e-07) | 316,922 |
| pytorch | mps | eager | 5.582 | 5.098 | 6.063 | 5.391 | 0.05 | PASS (1.3e-07) | 183,441 |
| pytorch | cpu | eager | 13.442 | 13.218 | 13.652 | 13.303 | 0.02 | REF | 76,177 |
| tinygrad | CPU | jit | 44.584 | 44.036 | 44.725 | 44.640 | 0.79 | PASS (8.0e-07) | 22,968 |
| ocannl | metal | default | 361.214 | 360.061 | 361.905 | 356.548 | 0.38 | PASS (2.0e-07) | 2,835 |
| ocannl | cc | tuned | 388.606 | 386.774 | 391.235 | 389.481 | 53.64 | PASS (8.1e-07) | 2,635 |
| ocannl | metal | materialized | 394.008 | 392.761 | 394.571 | 388.767 | 1.43 | PASS (8.1e-07) | 2,599 |
| ocannl | metal | tuned | 401.607 | 400.330 | 401.723 | 394.314 | 1137.69 | PASS (8.1e-07) | 2,550 |
| ocannl | cc | default | 2166.820 | 2164.130 | 2169.770 | 2166.420 | 3.83 | PASS (8.1e-07) | 473 |
| ocannl | cc | materialized | 2353.230 | 2348.480 | 2357.350 | 2352.290 | 15.99 | PASS (8.1e-07) | 435 |

## lenet

| framework | backend | variant | step p50 ms | p10 | p90 | queued ms | compile s | parity |
|---|---|---|---|---|---|---|---|---|
| tinygrad | METAL | jit | 1.270 | 0.779 | 1.337 | 0.567 | 1.30 | PASS (2.1e-07) |
| pytorch | mps | eager | 5.641 | 5.102 | 5.730 | 4.590 | 0.10 | PASS (1.0e-07) |
| tinygrad | CPU | jit | 5.697 | 5.591 | 5.801 | 5.695 | 1.31 | PASS (3.1e-07) |
| pytorch | cpu | eager | 12.523 | 12.287 | 12.798 | 12.515 | 0.03 | REF |
| ocannl | cc | materialized | 14.294 | 14.202 | 14.388 | 14.280 | 1.62 | PASS (2.1e-07) |
| ocannl | cc | tuned | 14.296 | 14.196 | 14.375 | 14.338 | 6.89 | PASS (2.1e-07) |
| ocannl | cc | default | 18.959 | 18.888 | 19.046 | 18.943 | 1.34 | PASS (2.1e-07) |
| ocannl | metal | materialized | 37.189 | 37.112 | 37.329 | 35.552 | 0.82 | PASS (2.1e-07) |
| ocannl | metal | tuned | 37.213 | 37.138 | 37.337 | 35.570 | 28.66 | PASS (2.1e-07) |
| ocannl | metal | default | 204.446 | 203.870 | 204.933 | 203.194 | 0.61 | PASS (2.1e-07) |

## mlp_small

| framework | backend | variant | step p50 ms | p10 | p90 | queued ms | compile s | parity |
|---|---|---|---|---|---|---|---|---|
| ocannl | cc | tuned | 0.118 | 0.106 | 0.146 | 0.096 | 2.71 | PASS (2.2e-07) |
| pytorch | cpu | eager | 0.177 | 0.162 | 0.230 | 0.183 | 0.01 | REF |
| ocannl | cc | materialized | 0.179 | 0.156 | 0.242 | 0.140 | 0.45 | PASS (2.2e-07) |
| ocannl | cc | default | 0.182 | 0.158 | 0.246 | 0.144 | 0.43 | PASS (2.2e-07) |
| tinygrad | CPU | jit | 0.297 | 0.292 | 0.308 | 0.306 | 0.32 | PASS (2.8e-07) |
| tinygrad | METAL | jit | 0.857 | 0.330 | 0.986 | 0.646 | 0.35 | PASS (2.4e-07) |
| pytorch | mps | eager | 1.156 | 1.065 | 1.234 | 0.275 | 0.07 | PASS (1.2e-07) |
| ocannl | metal | materialized | 1.298 | 1.229 | 1.371 | 0.502 | 0.10 | PASS (3.2e-07) |
| ocannl | metal | default | 1.325 | 1.256 | 1.393 | 0.514 | 0.14 | PASS (3.2e-07) |
| ocannl | metal | tuned | 1.876 | 1.800 | 1.962 | 0.961 | 2.08 | PASS (3.2e-07) |

## mlp_wide

| framework | backend | variant | step p50 ms | p10 | p90 | queued ms | compile s | parity |
|---|---|---|---|---|---|---|---|---|
| pytorch | mps | eager | 1.055 | 0.671 | 1.417 | 0.353 | 0.10 | PASS (2.1e-07) |
| tinygrad | METAL | jit | 1.134 | 0.702 | 1.419 | 0.898 | 0.55 | PASS (2.1e-07) |
| pytorch | cpu | eager | 1.671 | 1.611 | 1.729 | 1.664 | 0.01 | REF |
| ocannl | metal | materialized | 5.926 | 5.858 | 6.041 | 4.891 | 0.17 | PASS (5.2e-07) |
| ocannl | metal | tuned | 6.226 | 6.036 | 6.418 | 4.747 | 246.86 | PASS (5.2e-07) |
| ocannl | metal | default | 6.245 | 6.195 | 6.295 | 5.204 | 0.14 | PASS (5.2e-07) |
| tinygrad | CPU | jit | 15.598 | 15.347 | 15.837 | 15.819 | 0.52 | PASS (7.3e-07) |
| ocannl | cc | tuned | 48.701 | 47.970 | 49.473 | 48.708 | 3.92 | PASS (5.2e-07) |
| ocannl | cc | default | 217.825 | 217.159 | 218.930 | 218.355 | 0.53 | PASS (6.2e-07) |
| ocannl | cc | materialized | 217.837 | 217.188 | 218.974 | 217.942 | 0.56 | PASS (6.2e-07) |
