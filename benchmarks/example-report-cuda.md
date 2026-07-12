# Benchmark results

platform: Linux-6.18.33.2-microsoft-standard-WSL2-x86_64-with-glibc2.39 x86_64 | ocannl commit: 5c1ec99a | parity tol: 0.002 (max rel diff over first parity steps vs pytorch/cpu/eager)
> Checked-in example output (`results/` itself is generated and gitignored): full matrix at
> 5c1ec99a (placement A/B tuning), NVIDIA GeForce RTX 3050 Ti Laptop GPU (4 GB) under WSL2;
> torch 2.13.0+cu130, tinygrad 0.13.0 (CUDA 12.8). Regenerate with
> `benchmarks/.venv/bin/python benchmarks/orchestrate.py --tuned --materialized` (`--gpu cuda` is
> the default off macOS; tinygrad's CPU cells need `CC` pointed at a clang, see README; wipe
> `autotune_cache/` first for from-scratch search costs).
> The tuned cells use the two-pass protocol (see README): `compile s` is the cold search cost
> from pass 1; step timings are a fresh process replaying the cached winner. The tuned >=
> max(default, materialized) invariant holds in all 8 OCANNL cells of this run.


## gpt2_mini

| framework | backend | variant | step p50 ms | p10 | p90 | queued ms | compile s | parity | tok/s |
|---|---|---|---|---|---|---|---|---|---|
| pytorch | cuda | eager | 6.875 | 6.816 | 7.785 | 7.079 | 0.21 | PASS (6.7e-08) | 148,955 |
| tinygrad | CUDA | jit | 10.491 | 10.404 | 10.646 | 10.290 | 0.94 | PASS (1.3e-07) | 97,606 |
| pytorch | cpu | eager | 71.127 | 65.290 | 74.619 | 66.431 | 0.09 | REF | 14,397 |
| tinygrad | CPU | jit | 141.292 | 138.941 | 146.366 | 141.806 | 1.26 | PASS (8.7e-07) | 7,247 |
| ocannl | cuda | tuned | 247.177 | 247.071 | 247.297 | 246.934 | 637.63 | PASS (8.7e-07) | 4,143 |
| ocannl | cuda | default | 247.255 | 247.061 | 247.327 | 246.986 | 1.36 | PASS (8.7e-07) | 4,141 |
| ocannl | cuda | materialized | 297.550 | 297.307 | 297.865 | 297.810 | 2.51 | PASS (8.7e-07) | 3,441 |
| ocannl | cc | tuned | 1133.980 | 1058.870 | 1187.990 | 1132.500 | 154.47 | PASS (9.4e-07) | 903 |
| ocannl | cc | default | 2199.210 | 2139.460 | 2363.730 | 2376.410 | 2.12 | PASS (8.7e-07) | 466 |
| ocannl | cc | materialized | 2417.450 | 2400.380 | 2427.470 | 2608.260 | 6.58 | PASS (9.4e-07) | 424 |

## lenet

| framework | backend | variant | step p50 ms | p10 | p90 | queued ms | compile s | parity |
|---|---|---|---|---|---|---|---|---|
| tinygrad | CUDA | jit | 1.108 | 1.098 | 1.154 | 0.930 | 1.91 | PASS (2.1e-07) |
| pytorch | cuda | eager | 2.520 | 1.933 | 2.670 | 2.221 | 0.46 | PASS (8.8e-05) |
| pytorch | cpu | eager | 3.522 | 2.992 | 4.130 | 3.528 | 0.14 | REF |
| ocannl | cuda | tuned | 16.608 | 16.567 | 16.673 | 16.486 | 92.31 | PASS (2.1e-07) |
| ocannl | cuda | materialized | 16.718 | 16.670 | 16.783 | 16.570 | 1.35 | PASS (2.1e-07) |
| tinygrad | CPU | jit | 19.720 | 19.317 | 20.453 | 20.895 | 1.94 | PASS (3.1e-07) |
| ocannl | cc | tuned | 21.334 | 20.946 | 22.804 | 21.496 | 52.89 | PASS (3.1e-07) |
| ocannl | cc | materialized | 21.447 | 21.009 | 22.291 | 21.639 | 2.94 | PASS (3.1e-07) |
| ocannl | cc | default | 36.121 | 35.337 | 37.231 | 36.268 | 2.21 | PASS (3.1e-07) |
| ocannl | cuda | default | 156.766 | 156.709 | 156.830 | 156.837 | 1.21 | PASS (2.1e-07) |

## mlp_small

| framework | backend | variant | step p50 ms | p10 | p90 | queued ms | compile s | parity |
|---|---|---|---|---|---|---|---|---|
| ocannl | cc | tuned | 0.130 | 0.127 | 0.143 | 0.136 | 9.44 | PASS (2.8e-07) |
| ocannl | cc | materialized | 0.149 | 0.145 | 0.160 | 0.148 | 0.56 | PASS (2.8e-07) |
| tinygrad | CUDA | jit | 0.157 | 0.155 | 0.192 | 0.092 | 0.46 | PASS (2.8e-07) |
| ocannl | cuda | tuned | 0.169 | 0.151 | 0.209 | 0.127 | 34.97 | PASS (2.2e-07) |
| ocannl | cc | default | 0.174 | 0.166 | 0.189 | 0.176 | 0.53 | PASS (2.8e-07) |
| pytorch | cpu | eager | 0.220 | 0.203 | 0.262 | 0.248 | 0.12 | REF |
| ocannl | cuda | materialized | 0.266 | 0.244 | 0.293 | 0.245 | 0.13 | PASS (2.2e-07) |
| ocannl | cuda | default | 0.268 | 0.246 | 0.293 | 0.231 | 0.12 | PASS (2.2e-07) |
| tinygrad | CPU | jit | 0.629 | 0.588 | 0.710 | 0.621 | 0.51 | PASS (2.4e-07) |
| pytorch | cuda | eager | 1.072 | 0.887 | 1.473 | 0.965 | 0.16 | PASS (1.2e-07) |

## mlp_wide

| framework | backend | variant | step p50 ms | p10 | p90 | queued ms | compile s | parity |
|---|---|---|---|---|---|---|---|---|
| pytorch | cuda | eager | 1.121 | 1.065 | 1.447 | 1.153 | 0.16 | PASS (1.0e-07) |
| tinygrad | CUDA | jit | 2.504 | 2.487 | 2.560 | 2.351 | 0.62 | PASS (1.0e-07) |
| pytorch | cpu | eager | 10.141 | 8.898 | 11.092 | 10.381 | 0.12 | REF |
| ocannl | cuda | tuned | 12.524 | 12.233 | 12.745 | 12.400 | 451.32 | PASS (5.1e-07) |
| ocannl | cuda | default | 12.675 | 12.428 | 12.906 | 12.587 | 0.18 | PASS (5.1e-07) |
| ocannl | cuda | materialized | 12.753 | 12.541 | 13.017 | 12.638 | 0.17 | PASS (5.1e-07) |
| tinygrad | CPU | jit | 40.170 | 39.085 | 41.719 | 40.508 | 0.79 | PASS (6.2e-07) |
| ocannl | cc | tuned | 75.694 | 71.555 | 107.585 | 93.326 | 19.51 | PASS (5.2e-07) |
| ocannl | cc | default | 263.209 | 249.802 | 306.096 | 313.432 | 0.75 | PASS (5.2e-07) |
| ocannl | cc | materialized | 264.052 | 251.111 | 323.533 | 338.278 | 0.66 | PASS (5.2e-07) |
