# Benchmark results

platform: Linux-6.18.33.2-microsoft-standard-WSL2-x86_64-with-glibc2.39 x86_64 | ocannl commit: 1120a122 | parity tol: 0.002 (max rel diff over first parity steps vs pytorch/cpu/eager)
> Checked-in example output (`results/` itself is generated and gitignored): the first full-matrix
> CUDA run, on an NVIDIA GeForce RTX 3050 Ti Laptop GPU (4 GB) under WSL2; torch 2.13.0+cu130,
> tinygrad 0.13.0 (CUDA 12.8). Regenerate with
> `benchmarks/.venv/bin/python benchmarks/orchestrate.py --tuned --materialized` (`--gpu cuda` is
> the default off macOS). tinygrad's CPU device JIT-compiles via `clang`; on a machine without
> clang, point `CC` at one (this run used a `zig cc` shim from `pip install ziglang`).
> Known OCANNL anomalies in this run, kept for reference: the lenet cuda default schedule is ~9x
> slower than the materialized variant, and `Autotune.tune` regresses the cuda backend on
> gpt2_mini (2.3 s/step vs 0.25 s untuned) and mlp_small -- the cuda autotuner has not been
> looked at yet.


## gpt2_mini

| framework | backend | variant | step p50 ms | p10 | p90 | queued ms | compile s | parity | tok/s |
|---|---|---|---|---|---|---|---|---|---|
| pytorch | cuda | eager | 7.660 | 6.996 | 8.089 | 7.223 | 0.15 | PASS (6.7e-08) | 133,681 |
| tinygrad | CUDA | jit | 10.735 | 10.640 | 10.873 | 10.584 | 1.69 | PASS (1.3e-07) | 95,385 |
| pytorch | cpu | eager | 70.204 | 66.274 | 77.469 | 67.377 | 0.06 | REF | 14,586 |
| tinygrad | CPU | jit | 99.431 | 97.380 | 102.356 | 98.818 | 1.81 | PASS (8.7e-07) | 10,299 |
| ocannl | cuda | default | 250.926 | 250.773 | 251.082 | 250.669 | 1.36 | PASS (8.7e-07) | 4,081 |
| ocannl | cuda | materialized | 303.530 | 303.301 | 303.833 | 303.385 | 3.19 | PASS (8.7e-07) | 3,374 |
| ocannl | cc | tuned | 1140.300 | 1087.790 | 1272.280 | 1158.620 | 46.95 | PASS (9.4e-07) | 898 |
| ocannl | cuda | tuned | 2271.830 | 2265.300 | 2278.600 | 2417.350 | 288.12 | PASS (8.7e-07) | 451 |
| ocannl | cc | default | 2348.110 | 2265.410 | 2407.430 | 2503.920 | 2.16 | PASS (8.7e-07) | 436 |
| ocannl | cc | materialized | 2549.680 | 2478.030 | 2807.280 | 2555.250 | 6.84 | PASS (9.4e-07) | 402 |

## lenet

| framework | backend | variant | step p50 ms | p10 | p90 | queued ms | compile s | parity |
|---|---|---|---|---|---|---|---|---|
| tinygrad | CUDA | jit | 1.200 | 1.191 | 1.216 | 0.966 | 3.81 | PASS (2.1e-07) |
| pytorch | cuda | eager | 1.689 | 1.559 | 2.619 | 1.728 | 0.31 | PASS (8.8e-05) |
| pytorch | cpu | eager | 3.336 | 2.955 | 3.618 | 3.580 | 0.12 | REF |
| tinygrad | CPU | jit | 16.295 | 15.769 | 17.134 | 16.431 | 3.62 | PASS (3.1e-07) |
| ocannl | cuda | tuned | 16.977 | 16.934 | 17.041 | 16.855 | 36.77 | PASS (2.1e-07) |
| ocannl | cuda | materialized | 17.031 | 16.971 | 17.103 | 16.904 | 1.69 | PASS (2.1e-07) |
| ocannl | cc | tuned | 21.190 | 20.656 | 22.040 | 21.306 | 14.09 | PASS (3.1e-07) |
| ocannl | cc | materialized | 21.298 | 20.872 | 22.561 | 51.032 | 2.92 | PASS (3.1e-07) |
| ocannl | cc | default | 35.955 | 35.191 | 37.027 | 49.149 | 2.14 | PASS (3.1e-07) |
| ocannl | cuda | default | 160.145 | 160.083 | 160.211 | 159.926 | 1.36 | PASS (2.1e-07) |

## mlp_small

| framework | backend | variant | step p50 ms | p10 | p90 | queued ms | compile s | parity |
|---|---|---|---|---|---|---|---|---|
| ocannl | cc | tuned | 0.132 | 0.129 | 0.141 | 0.133 | 3.24 | PASS (2.8e-07) |
| ocannl | cc | materialized | 0.155 | 0.149 | 0.167 | 0.153 | 0.53 | PASS (2.8e-07) |
| ocannl | cuda | default | 0.160 | 0.147 | 0.273 | 0.153 | 0.12 | PASS (2.2e-07) |
| ocannl | cc | default | 0.173 | 0.167 | 0.184 | 0.195 | 0.55 | PASS (2.8e-07) |
| ocannl | cuda | materialized | 0.174 | 0.145 | 0.303 | 0.150 | 0.26 | PASS (2.2e-07) |
| tinygrad | CUDA | jit | 0.198 | 0.191 | 0.209 | 0.124 | 1.24 | PASS (2.8e-07) |
| pytorch | cpu | eager | 0.221 | 0.204 | 0.264 | 0.266 | 0.10 | REF |
| ocannl | cuda | tuned | 0.470 | 0.453 | 0.664 | 0.396 | 22.16 | PASS (2.2e-07) |
| tinygrad | CPU | jit | 0.631 | 0.571 | 0.737 | 0.607 | 1.28 | PASS (2.4e-07) |
| pytorch | cuda | eager | 1.368 | 1.264 | 1.499 | 1.334 | 0.17 | PASS (1.2e-07) |

## mlp_wide

| framework | backend | variant | step p50 ms | p10 | p90 | queued ms | compile s | parity |
|---|---|---|---|---|---|---|---|---|
| pytorch | cuda | eager | 1.091 | 1.029 | 1.569 | 0.938 | 0.17 | PASS (1.0e-07) |
| tinygrad | CUDA | jit | 2.481 | 2.466 | 2.991 | 2.404 | 1.55 | PASS (1.0e-07) |
| pytorch | cpu | eager | 10.909 | 9.732 | 11.919 | 10.575 | 0.12 | REF |
| ocannl | cuda | tuned | 12.675 | 12.389 | 12.893 | 12.582 | 210.84 | PASS (5.1e-07) |
| ocannl | cuda | default | 12.840 | 12.598 | 13.075 | 12.735 | 0.26 | PASS (5.1e-07) |
| ocannl | cuda | materialized | 12.906 | 12.664 | 13.155 | 12.800 | 0.29 | PASS (5.1e-07) |
| tinygrad | CPU | jit | 27.383 | 26.760 | 28.745 | 41.866 | 1.63 | PASS (6.2e-07) |
| ocannl | cc | tuned | 69.585 | 67.829 | 89.838 | 77.824 | 5.28 | PASS (5.2e-07) |
| ocannl | cc | materialized | 258.001 | 244.207 | 311.940 | 298.293 | 0.65 | PASS (5.2e-07) |
| ocannl | cc | default | 264.370 | 249.714 | 335.385 | 287.238 | 0.75 | PASS (5.2e-07) |
