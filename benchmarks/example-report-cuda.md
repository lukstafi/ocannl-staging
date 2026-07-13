# Benchmark results

platform: Linux-6.18.33.2-microsoft-standard-WSL2-x86_64-with-glibc2.39 x86_64 | ocannl commit: fc1e45af | parity tol: 0.002 (max rel diff over first parity steps vs pytorch/cpu/eager)
> Checked-in example output (`results/` itself is generated and gitignored): full matrix at the
> PR #140 tip (placement A/B tuning with structural replay keys), NVIDIA GeForce RTX 3050 Ti
> Laptop GPU (4 GB) under WSL2; torch 2.13.0+cu130, tinygrad 0.13.0 (CUDA 12.8). Regenerate with
> `benchmarks/.venv/bin/python benchmarks/orchestrate.py --tuned --materialized` (`--gpu cuda` is
> the default off macOS; tinygrad's CPU cells need `CC` pointed at a clang, see README; wipe
> `autotune_cache/` first for from-scratch search costs).
> The tuned cells use the two-pass protocol (see README): `compile s` is the cold search cost
> from pass 1; step timings are a fresh process replaying the cached winner. The tuned >=
> max(default, materialized) invariant holds in all 8 OCANNL cells of this run.


## gpt2_mini

| framework | backend | variant | step p50 ms | p10 | p90 | queued ms | compile s | parity | tok/s |
|---|---|---|---|---|---|---|---|---|---|
| pytorch | cuda | eager | 8.172 | 7.615 | 8.689 | 8.152 | 0.23 | PASS (6.7e-08) | 125,305 |
| tinygrad | CUDA | jit | 12.830 | 11.786 | 12.922 | 11.493 | 1.03 | PASS (1.3e-07) | 79,810 |
| pytorch | cpu | eager | 82.427 | 80.647 | 86.397 | 75.875 | 0.10 | REF | 12,423 |
| tinygrad | CPU | jit | 157.738 | 153.594 | 159.969 | 158.511 | 1.41 | PASS (8.7e-07) | 6,492 |
| ocannl | cuda | default | 274.629 | 274.563 | 274.788 | 274.433 | 1.40 | PASS (8.7e-07) | 3,729 |
| ocannl | cuda | tuned | 275.133 | 275.002 | 275.299 | 274.773 | 633.80 | PASS (8.7e-07) | 3,722 |
| ocannl | cuda | materialized | 330.590 | 330.364 | 331.165 | 331.615 | 2.67 | PASS (8.7e-07) | 3,097 |
| ocannl | cc | tuned | 1212.400 | 1113.860 | 1256.160 | 1216.830 | 148.66 | PASS (9.4e-07) | 845 |
| ocannl | cc | default | 2330.670 | 2275.380 | 2355.680 | 2362.260 | 2.28 | PASS (8.7e-07) | 439 |
| ocannl | cc | materialized | 2590.840 | 2568.950 | 2605.810 | 2598.370 | 7.27 | PASS (9.4e-07) | 395 |

## lenet

| framework | backend | variant | step p50 ms | p10 | p90 | queued ms | compile s | parity |
|---|---|---|---|---|---|---|---|---|
| tinygrad | CUDA | jit | 1.296 | 1.085 | 1.317 | 0.867 | 2.18 | PASS (2.1e-07) |
| pytorch | cuda | eager | 1.782 | 1.669 | 2.701 | 2.186 | 0.56 | PASS (8.8e-05) |
| pytorch | cpu | eager | 3.652 | 3.247 | 4.005 | 3.864 | 0.18 | REF |
| ocannl | cuda | tuned | 18.514 | 18.456 | 18.567 | 18.337 | 89.13 | PASS (2.1e-07) |
| ocannl | cuda | materialized | 18.580 | 18.539 | 18.635 | 18.437 | 1.44 | PASS (2.1e-07) |
| tinygrad | CPU | jit | 22.030 | 21.481 | 22.639 | 22.159 | 2.17 | PASS (3.1e-07) |
| ocannl | cc | tuned | 23.509 | 22.934 | 24.820 | 22.899 | 48.58 | PASS (3.1e-07) |
| ocannl | cc | materialized | 23.876 | 23.324 | 25.042 | 23.956 | 3.29 | PASS (3.1e-07) |
| ocannl | cc | default | 40.553 | 39.790 | 42.219 | 40.804 | 2.49 | PASS (3.1e-07) |
| ocannl | cuda | default | 174.170 | 174.095 | 174.252 | 173.914 | 1.26 | PASS (2.1e-07) |

## mlp_small

| framework | backend | variant | step p50 ms | p10 | p90 | queued ms | compile s | parity |
|---|---|---|---|---|---|---|---|---|
| ocannl | cc | tuned | 0.151 | 0.147 | 0.164 | 0.153 | 10.36 | PASS (2.8e-07) |
| ocannl | cc | materialized | 0.166 | 0.161 | 0.175 | 0.165 | 0.58 | PASS (2.8e-07) |
| ocannl | cuda | tuned | 0.169 | 0.122 | 0.323 | 0.157 | 34.52 | PASS (2.2e-07) |
| tinygrad | CUDA | jit | 0.179 | 0.175 | 0.217 | 0.106 | 0.50 | PASS (2.8e-07) |
| ocannl | cuda | default | 0.180 | 0.126 | 0.303 | 0.135 | 0.13 | PASS (2.2e-07) |
| ocannl | cc | default | 0.181 | 0.176 | 0.193 | 0.180 | 0.56 | PASS (2.8e-07) |
| ocannl | cuda | materialized | 0.203 | 0.152 | 0.305 | 0.156 | 0.14 | PASS (2.2e-07) |
| pytorch | cpu | eager | 0.253 | 0.231 | 0.636 | 0.403 | 0.15 | REF |
| tinygrad | CPU | jit | 0.680 | 0.638 | 0.749 | 0.680 | 0.56 | PASS (2.4e-07) |
| pytorch | cuda | eager | 1.296 | 0.981 | 1.573 | 1.347 | 0.17 | PASS (1.2e-07) |

## mlp_wide

| framework | backend | variant | step p50 ms | p10 | p90 | queued ms | compile s | parity |
|---|---|---|---|---|---|---|---|---|
| pytorch | cuda | eager | 1.110 | 1.067 | 1.415 | 0.965 | 0.17 | PASS (1.0e-07) |
| tinygrad | CUDA | jit | 2.749 | 2.732 | 3.314 | 2.605 | 0.67 | PASS (1.0e-07) |
| pytorch | cpu | eager | 11.554 | 10.863 | 12.394 | 10.797 | 0.13 | REF |
| ocannl | cuda | tuned | 13.912 | 13.604 | 14.212 | 13.821 | 453.11 | PASS (5.1e-07) |
| ocannl | cuda | default | 14.120 | 13.855 | 14.373 | 13.973 | 0.21 | PASS (5.1e-07) |
| ocannl | cuda | materialized | 14.215 | 13.982 | 14.477 | 14.063 | 0.18 | PASS (5.1e-07) |
| tinygrad | CPU | jit | 43.799 | 42.511 | 45.605 | 43.646 | 0.86 | PASS (6.2e-07) |
| ocannl | cc | tuned | 89.214 | 85.214 | 116.715 | 108.097 | 21.66 | PASS (5.2e-07) |
| ocannl | cc | default | 286.176 | 275.921 | 322.762 | 298.615 | 0.73 | PASS (5.2e-07) |
| ocannl | cc | materialized | 289.759 | 278.446 | 322.250 | 299.688 | 0.73 | PASS (5.2e-07) |
