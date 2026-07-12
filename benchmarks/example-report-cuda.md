# Benchmark results

platform: Linux-6.18.33.2-microsoft-standard-WSL2-x86_64-with-glibc2.39 x86_64 | ocannl commit: ff4705b3 | parity tol: 0.002 (max rel diff over first parity steps vs pytorch/cpu/eager)
> Checked-in example output (`results/` itself is generated and gitignored): full matrix at
> ff4705b3 (placement A/B tuning, post digest fix), NVIDIA GeForce RTX 3050 Ti Laptop GPU (4 GB)
> under WSL2; torch 2.13.0+cu130, tinygrad 0.13.0 (CUDA 12.8). Regenerate with
> `benchmarks/.venv/bin/python benchmarks/orchestrate.py --tuned --materialized` (`--gpu cuda` is
> the default off macOS; tinygrad's CPU cells need `CC` pointed at a clang, see README).
> KNOWN BUG captured by this run: tuned should be at least as fast as default and materialized in
> every cell, but two cuda cells violate it — gpt2_mini tuned 2623 ms vs 251 ms default, and
> mlp_small tuned 0.47 ms vs 0.16 ms. Diagnosis (via `BENCH_TUNE_REPORT=1`): the default-placements
> arm of `Train.tune_placements` rejects ALL its generated candidates with "fresh lowering does not
> match the tuned code (digest mismatch)" (mlp_small: 20/20 on cuda, 5/5 on cc), degenerating to
> its serial baseline; the materialize-all arm then always wins the A/B, and on cuda its winner can
> be far slower than the untuned default schedule, which the search never times.


## gpt2_mini

| framework | backend | variant | step p50 ms | p10 | p90 | queued ms | compile s | parity | tok/s |
|---|---|---|---|---|---|---|---|---|---|
| pytorch | cuda | eager | 7.620 | 7.478 | 8.063 | 6.963 | 0.21 | PASS (6.7e-08) | 134,379 |
| tinygrad | CUDA | jit | 10.699 | 10.562 | 13.231 | 10.514 | 0.89 | PASS (1.3e-07) | 95,711 |
| pytorch | cpu | eager | 68.385 | 62.185 | 70.495 | 65.618 | 0.09 | REF | 14,974 |
| tinygrad | CPU | jit | 144.319 | 139.199 | 148.062 | 141.356 | 1.22 | PASS (8.7e-07) | 7,095 |
| ocannl | cuda | default | 250.576 | 250.464 | 250.661 | 250.299 | 1.35 | PASS (8.7e-07) | 4,087 |
| ocannl | cuda | materialized | 302.921 | 302.728 | 303.312 | 302.720 | 2.38 | PASS (8.7e-07) | 3,380 |
| ocannl | cc | tuned | 1043.430 | 963.505 | 1258.520 | 1178.690 | 79.73 | PASS (9.4e-07) | 981 |
| ocannl | cc | default | 2122.770 | 2088.460 | 2186.320 | 2484.600 | 1.95 | PASS (8.7e-07) | 482 |
| ocannl | cc | materialized | 2312.990 | 2294.910 | 2401.720 | 2594.930 | 6.43 | PASS (9.4e-07) | 443 |
| ocannl | cuda | tuned | 2623.280 | 2617.170 | 2632.560 | 2916.590 | 537.65 | PASS (8.7e-07) | 390 |

## lenet

| framework | backend | variant | step p50 ms | p10 | p90 | queued ms | compile s | parity |
|---|---|---|---|---|---|---|---|---|
| tinygrad | CUDA | jit | 1.152 | 1.131 | 1.236 | 0.793 | 1.94 | PASS (2.1e-07) |
| pytorch | cuda | eager | 2.510 | 2.384 | 2.638 | 2.572 | 0.50 | PASS (8.8e-05) |
| pytorch | cpu | eager | 3.608 | 3.227 | 3.944 | 3.605 | 0.15 | REF |
| ocannl | cuda | tuned | 16.970 | 16.908 | 17.029 | 16.813 | 68.38 | PASS (2.1e-07) |
| ocannl | cuda | materialized | 17.001 | 16.948 | 17.063 | 16.859 | 1.28 | PASS (2.1e-07) |
| tinygrad | CPU | jit | 19.895 | 18.976 | 21.006 | 20.187 | 1.99 | PASS (3.1e-07) |
| ocannl | cc | materialized | 20.514 | 20.057 | 20.911 | 20.628 | 2.83 | PASS (3.1e-07) |
| ocannl | cc | tuned | 21.160 | 20.029 | 24.657 | 21.037 | 28.24 | PASS (3.1e-07) |
| ocannl | cc | default | 36.299 | 35.907 | 37.858 | 35.825 | 2.17 | PASS (3.1e-07) |
| ocannl | cuda | default | 160.149 | 159.570 | 160.266 | 159.981 | 1.12 | PASS (2.1e-07) |

## mlp_small

| framework | backend | variant | step p50 ms | p10 | p90 | queued ms | compile s | parity |
|---|---|---|---|---|---|---|---|---|
| ocannl | cc | tuned | 0.136 | 0.132 | 0.146 | 0.135 | 5.26 | PASS (2.8e-07) |
| ocannl | cc | materialized | 0.156 | 0.152 | 0.167 | 0.157 | 0.53 | PASS (2.8e-07) |
| tinygrad | CUDA | jit | 0.161 | 0.159 | 0.177 | 0.093 | 0.46 | PASS (2.8e-07) |
| ocannl | cuda | default | 0.163 | 0.134 | 0.283 | 0.279 | 0.12 | PASS (2.2e-07) |
| ocannl | cc | default | 0.166 | 0.162 | 0.178 | 0.167 | 0.52 | PASS (2.8e-07) |
| ocannl | cuda | materialized | 0.172 | 0.148 | 0.278 | 0.149 | 0.13 | PASS (2.2e-07) |
| pytorch | cpu | eager | 0.286 | 0.208 | 0.346 | 0.342 | 0.07 | REF |
| ocannl | cuda | tuned | 0.471 | 0.454 | 0.667 | 0.375 | 24.55 | PASS (2.2e-07) |
| tinygrad | CPU | jit | 0.631 | 0.590 | 0.716 | 0.632 | 0.52 | PASS (2.4e-07) |
| pytorch | cuda | eager | 0.874 | 0.806 | 1.283 | 0.905 | 0.14 | PASS (1.2e-07) |

## mlp_wide

| framework | backend | variant | step p50 ms | p10 | p90 | queued ms | compile s | parity |
|---|---|---|---|---|---|---|---|---|
| pytorch | cuda | eager | 1.432 | 1.200 | 1.524 | 1.416 | 0.16 | PASS (1.0e-07) |
| tinygrad | CUDA | jit | 2.467 | 2.453 | 2.973 | 2.341 | 0.58 | PASS (1.0e-07) |
| pytorch | cpu | eager | 10.106 | 9.371 | 11.035 | 9.370 | 0.12 | REF |
| ocannl | cuda | tuned | 12.413 | 12.212 | 12.665 | 12.355 | 447.61 | PASS (5.1e-07) |
| ocannl | cuda | default | 12.610 | 12.332 | 12.814 | 12.475 | 0.15 | PASS (5.1e-07) |
| ocannl | cuda | materialized | 12.659 | 12.446 | 12.817 | 12.567 | 0.15 | PASS (5.1e-07) |
| tinygrad | CPU | jit | 39.109 | 38.002 | 40.302 | 37.679 | 0.78 | PASS (6.2e-07) |
| ocannl | cc | tuned | 67.937 | 66.153 | 89.003 | 72.183 | 9.96 | PASS (5.2e-07) |
| ocannl | cc | materialized | 251.962 | 239.842 | 295.614 | 300.137 | 0.64 | PASS (5.2e-07) |
| ocannl | cc | default | 257.688 | 245.892 | 287.255 | 296.597 | 0.69 | PASS (5.2e-07) |
