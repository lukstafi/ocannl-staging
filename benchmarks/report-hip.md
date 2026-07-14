# Benchmark results

platform: Windows-11-10.0.26200-SP0 AMD64 | ocannl commit: 8436e362 | parity tol: 0.002 (max rel diff over first parity steps vs pytorch/cpu/eager)

Hardware: AMD Strix Halo (Radeon 8060S iGPU, gfx1151), ROCm/HIP SDK 7.1 (OCANNL hip backend).
Run as `orchestrate.py --tuned --materialized --gpu hip`. All three frameworks reach the GPU
on Windows: PyTorch via AMD's official ROCm 7.2.1 wheels (torch 2.9.1+rocm7.2.1, HIP exposed
as the "cuda" device), tinygrad 0.13 via its OpenCL device (CL). Note the pytorch/cpu/eager
reference is that same ROCm build — noticeably slower on CPU than the stock CPU wheel
(torch 2.13) used in earlier revisions of this report, so cross-revision CPU comparisons
are apples-to-oranges.


## gpt2_mini

| framework | backend | variant | step p50 ms | p10 | p90 | queued ms | compile s | parity | tok/s |
|---|---|---|---|---|---|---|---|---|---|
| pytorch | cuda | eager | 5.996 | 5.897 | 6.093 | 5.971 | 0.46 | PASS (6.7e-08) | 170,786 |
| tinygrad | CL | jit | 10.889 | 9.644 | 13.609 | 8.092 | 0.75 | PASS (1.3e-07) | 94,039 |
| tinygrad | CPU | jit | 56.859 | 55.974 | 58.078 | 56.657 | 0.96 | PASS (8.7e-07) | 18,010 |
| ocannl | hip | default | 67.418 | 66.295 | 69.356 | 67.924 | 1.96 | PASS (6.8e-08) | 15,189 |
| ocannl | hip | tuned | 67.601 | 66.024 | 70.729 | 67.648 | 532.97 | PASS (6.8e-08) | 15,148 |
| ocannl | hip | materialized | 71.350 | 71.023 | 71.748 | 70.909 | 2.72 | PASS (8.7e-07) | 14,352 |
| ocannl | cc | tuned | 493.617 | 478.082 | 504.336 | 487.602 | 151.14 | PASS (8.0e-07) | 2,074 |
| pytorch | cpu | eager | 510.756 | 372.063 | 574.738 | 521.554 | 0.65 | REF | 2,005 |
| ocannl | cc | default | 2207.050 | 2206.010 | 2209.150 | 2206.490 | 2.86 | PASS (8.7e-07) | 464 |
| ocannl | cc | materialized | 2388.750 | 2384.740 | 2389.760 | 2387.440 | 9.94 | PASS (8.0e-07) | 429 |

## lenet

| framework | backend | variant | step p50 ms | p10 | p90 | queued ms | compile s | parity |
|---|---|---|---|---|---|---|---|---|
| tinygrad | CL | jit | 1.321 | 1.305 | 1.411 | 0.775 | 9.07 | PASS (2.1e-07) |
| pytorch | cuda | eager | 1.818 | 1.718 | 1.992 | 1.129 | 0.57 | PASS (2.1e-07) |
| pytorch | cpu | eager | 3.052 | 2.742 | 42.369 | 17.674 | 0.12 | REF |
| tinygrad | CPU | jit | 6.030 | 5.848 | 6.590 | 6.302 | 1.67 | PASS (3.1e-07) |
| ocannl | hip | materialized | 22.259 | 21.870 | 22.570 | 22.249 | 1.72 | PASS (3.1e-07) |
| ocannl | cc | tuned | 22.430 | 22.345 | 22.563 | 22.460 | 194.37 | PASS (3.1e-07) |
| ocannl | cc | materialized | 22.513 | 22.387 | 22.849 | 22.588 | 3.37 | PASS (3.1e-07) |
| ocannl | hip | tuned | 22.644 | 22.206 | 22.964 | 22.183 | 88.94 | PASS (3.1e-07) |
| ocannl | cc | default | 32.493 | 32.254 | 32.822 | 32.363 | 3.09 | PASS (3.1e-07) |
| ocannl | hip | default | 82.016 | 81.662 | 82.485 | 81.811 | 1.50 | PASS (2.1e-07) |

## mlp_small

| framework | backend | variant | step p50 ms | p10 | p90 | queued ms | compile s | parity |
|---|---|---|---|---|---|---|---|---|
| ocannl | cc | tuned | 0.115 | 0.113 | 0.124 | 0.115 | 17.97 | PASS (2.8e-07) |
| ocannl | cc | materialized | 0.140 | 0.137 | 0.142 | 0.140 | 1.06 | PASS (2.8e-07) |
| ocannl | cc | default | 0.163 | 0.159 | 0.167 | 0.165 | 1.41 | PASS (2.8e-07) |
| ocannl | hip | default | 0.190 | 0.183 | 0.217 | 0.180 | 0.33 | PASS (2.9e-07) |
| ocannl | hip | materialized | 0.191 | 0.185 | 0.207 | 0.186 | 0.34 | PASS (2.8e-07) |
| ocannl | hip | tuned | 0.212 | 0.203 | 0.243 | 0.209 | 24.95 | PASS (2.9e-07) |
| tinygrad | CPU | jit | 0.436 | 0.365 | 0.557 | 0.395 | 1.29 | PASS (2.4e-07) |
| pytorch | cuda | eager | 0.438 | 0.425 | 0.547 | 0.336 | 0.44 | PASS (2.0e-07) |
| tinygrad | CL | jit | 0.549 | 0.541 | 0.621 | 0.458 | 0.39 | PASS (2.8e-07) |
| pytorch | cpu | eager | 0.671 | 0.665 | 0.688 | 0.674 | 0.04 | REF |

## mlp_wide

| framework | backend | variant | step p50 ms | p10 | p90 | queued ms | compile s | parity |
|---|---|---|---|---|---|---|---|---|
| pytorch | cuda | eager | 1.180 | 1.124 | 1.325 | 0.825 | 0.44 | PASS (2.1e-07) |
| ocannl | hip | tuned | 1.584 | 1.546 | 1.667 | 1.558 | 111.78 | PASS (4.2e-07) |
| tinygrad | CL | jit | 1.726 | 1.670 | 1.827 | 1.241 | 5.62 | PASS (2.1e-07) |
| ocannl | hip | materialized | 1.739 | 1.698 | 1.833 | 1.705 | 0.40 | PASS (4.2e-07) |
| ocannl | hip | default | 1.809 | 1.754 | 1.919 | 1.764 | 0.39 | PASS (4.2e-07) |
| tinygrad | CPU | jit | 21.918 | 21.264 | 23.260 | 22.321 | 1.58 | PASS (6.2e-07) |
| pytorch | cpu | eager | 77.278 | 53.169 | 214.255 | 104.522 | 0.49 | REF |
| ocannl | cc | tuned | 131.273 | 129.315 | 133.784 | 131.495 | 26.54 | PASS (4.2e-07) |
| ocannl | cc | materialized | 331.131 | 329.700 | 332.870 | 331.952 | 1.41 | PASS (5.2e-07) |
| ocannl | cc | default | 331.485 | 330.086 | 333.139 | 331.993 | 1.69 | PASS (5.2e-07) |
