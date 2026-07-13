# Benchmark results

platform: Windows-11-10.0.26200-SP0 AMD64 | ocannl commit: c1a985ef | parity tol: 0.002 (max rel diff over first parity steps vs pytorch/cpu/eager)

Hardware: AMD Strix Halo (Radeon 8060S iGPU, gfx1151), ROCm/HIP SDK 7.1. Run as
`orchestrate.py --tuned --materialized --gpu hip --only ocannl pytorch` (neither PyTorch nor
tinygrad reaches an AMD GPU on Windows). Measured at the PR tip of ocannl-staging#145 — the
commit above is the merge base; the PR's monotonic-clock timing fixes are included, without
which sub-millisecond step times quantize to 0/1 ms on Windows.


## gpt2_mini

| framework | backend | variant | step p50 ms | p10 | p90 | queued ms | compile s | parity | tok/s |
|---|---|---|---|---|---|---|---|---|---|
| pytorch | cpu | eager | 23.356 | 22.367 | 24.945 | 25.696 | 0.03 | REF | 43,844 |
| ocannl | hip | tuned | 67.331 | 66.451 | 70.356 | 67.730 | 542.34 | PASS (1.3e-07) | 15,208 |
| ocannl | hip | default | 68.278 | 66.471 | 71.358 | 67.869 | 1.96 | PASS (1.3e-07) | 14,997 |
| ocannl | hip | materialized | 72.636 | 71.460 | 73.412 | 71.359 | 2.73 | PASS (8.7e-07) | 14,098 |
| ocannl | cc | tuned | 500.138 | 486.113 | 509.318 | 493.680 | 151.19 | PASS (8.0e-07) | 2,047 |
| ocannl | cc | default | 2212.050 | 2209.280 | 2214.540 | 2209.020 | 3.14 | PASS (8.7e-07) | 463 |
| ocannl | cc | materialized | 2387.700 | 2385.920 | 2389.150 | 2386.550 | 10.01 | PASS (8.0e-07) | 429 |

## lenet

| framework | backend | variant | step p50 ms | p10 | p90 | queued ms | compile s | parity |
|---|---|---|---|---|---|---|---|---|
| pytorch | cpu | eager | 1.371 | 1.246 | 1.755 | 1.439 | 0.01 | REF |
| ocannl | hip | materialized | 22.246 | 21.962 | 22.622 | 22.193 | 1.75 | PASS (3.1e-07) |
| ocannl | hip | tuned | 22.318 | 21.994 | 22.616 | 22.795 | 89.72 | PASS (3.1e-07) |
| ocannl | cc | materialized | 22.509 | 22.374 | 22.752 | 22.774 | 3.35 | PASS (3.1e-07) |
| ocannl | cc | tuned | 22.644 | 22.477 | 23.071 | 22.628 | 196.90 | PASS (3.1e-07) |
| ocannl | cc | default | 32.370 | 32.121 | 32.684 | 32.642 | 3.24 | PASS (3.1e-07) |
| ocannl | hip | default | 82.096 | 81.653 | 82.477 | 81.976 | 1.50 | PASS (2.1e-07) |

## mlp_small

| framework | backend | variant | step p50 ms | p10 | p90 | queued ms | compile s | parity |
|---|---|---|---|---|---|---|---|---|
| ocannl | cc | tuned | 0.107 | 0.106 | 0.115 | 0.109 | 16.66 | PASS (2.8e-07) |
| ocannl | cc | materialized | 0.139 | 0.137 | 0.142 | 0.141 | 0.92 | PASS (2.8e-07) |
| ocannl | cc | default | 0.140 | 0.135 | 0.148 | 0.143 | 0.82 | PASS (2.8e-07) |
| ocannl | hip | default | 0.190 | 0.182 | 0.213 | 0.184 | 0.09 | PASS (3.0e-07) |
| ocannl | hip | materialized | 0.208 | 0.199 | 0.241 | 0.205 | 0.09 | PASS (2.8e-07) |
| ocannl | hip | tuned | 0.210 | 0.199 | 0.257 | 0.210 | 5.65 | PASS (3.0e-07) |
| pytorch | cpu | eager | 0.268 | 0.239 | 0.310 | 0.265 | 0.00 | REF |

## mlp_wide

| framework | backend | variant | step p50 ms | p10 | p90 | queued ms | compile s | parity |
|---|---|---|---|---|---|---|---|---|
| ocannl | hip | tuned | 1.630 | 1.584 | 1.730 | 1.605 | 114.01 | PASS (4.2e-07) |
| ocannl | hip | materialized | 1.750 | 1.703 | 1.823 | 1.706 | 0.40 | PASS (4.2e-07) |
| ocannl | hip | default | 1.832 | 1.766 | 2.038 | 1.767 | 0.40 | PASS (4.2e-07) |
| pytorch | cpu | eager | 3.984 | 3.587 | 4.720 | 4.106 | 0.01 | REF |
| ocannl | cc | tuned | 130.773 | 129.064 | 134.170 | 130.999 | 27.34 | PASS (4.2e-07) |
| ocannl | cc | materialized | 331.161 | 329.459 | 333.288 | 331.142 | 1.21 | PASS (5.2e-07) |
| ocannl | cc | default | 331.693 | 329.856 | 333.346 | 331.674 | 1.18 | PASS (5.2e-07) |
