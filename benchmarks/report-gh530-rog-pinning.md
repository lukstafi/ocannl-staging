# gh-530: decomposing rog's conv-sketch tuning-ratio outlier (native Windows, core pinning)

Native-Windows measurement session decomposing rog's tuning-ratio outlier into scheduler,
core-heterogeneity, and seed-portability effects. Measurement only — no seed or schedule code was
changed.

**Verdict: the outlier is caused by P/E core heterogeneity. The operative variable is *uniformity*,
not core strength — and the tuner is behaving correctly.** On the mixed 24-core machine the crowned
schedule is indistinguishable from plain materialize-all on both cells. On either homogeneous
subset — eight big cores *or* sixteen small ones — the same search is worth 26–39%. A schedule
crowned on P-cores, replayed on the full mixed machine, runs *worse than materialize-all*: the
tuner on the mixed machine is not failing to find good schedules, it is correctly reporting that
none of them pay there.

The WSL rows in gh-530 remain the cross-machine ledger; the numbers here diagnose, they do not
replace them.

## Provenance

| field | value |
|---|---|
| machine | rog — Intel Core Ultra 9 275HX (Arrow Lake-HX), 24 cores / 24 LPs, no SMT |
| OS | Windows 11, 10.0.26200 build 26200 — **native**, not WSL |
| compiler | mingw `x86_64-w64-mingw32-gcc` 14.4.0 (cygwin-hosted, opam sandbox) |
| OCaml / dune | 5.5.0 / 3.24.0 |
| branch / commit | `claude/gh-530-pinning-windows-741611` @ `a7672848` |
| backend | `cc` (OpenMP `parallel for` rendering) |
| RAM | 31.3 GB |
| box state | otherwise idle, no CUDA work. An idle WSL VM (`vmmemWSL`, ~2.4 GB RSS) would not stay shut down, but measured **0 CPU-seconds per 10 s wall** — parked, not competing. |
| pinning | `start "" /affinity <mask> /wait /b` (children inherit; verified live via `Process.ProcessorAffinity` on running cells of every arm) |
| protocol | from-scratch search per tuned cell against a **fresh `autotune_cache_dir`**; standard two-pass protocol (pass 1 searches, its `compile_s` is the search cost; a fresh pass 2 replays the cached winner for step times) |
| replication | `cifar_conv` n=2 on all three arms (independent from-scratch searches); `cifar_stride` n=1 |

Absolute times are comparable *within* this report only — the WSL ledger rows used a different
toolchain. Ratios are the claim-bearing quantity.

## Verified core-type map

Established empirically in a prior session by pinned spin-loop timing (clean 1.5x bimodal split),
used as given here:

| class | logical processors | affinity mask |
|---|---|---|
| P-cores | 0, 1, 10–13, 22, 23 | `0xC03C03` |
| E-cores | 2–9, 14–21 | `0x3FC3FC` |
| all | 0–23 | `0xFFFFFF` |

The naive `0–7` range is **two P-cores and six E-cores** — not a P-core proxy.

## Arrow Lake cache geometry

From `Win32_CacheMemory` (WMI `Level` is offset by two: Level 3 = L1, 4 = L2, 5 = L3):

| level | P-side | E-side | shared |
|---|---|---|---|
| L1 | 384 KB, 9-way | 512 KB / 1024 KB, 7-way | — |
| L2 | **24 MB** = 8 P-cores x 3 MB, private per core | **16 MB** = 4 E-clusters x 4 MB, shared per cluster | — |
| L3 | — | — | **36 MB**, single ring, shared by all 24 |

Recorded as the ground-truth row for the parameterization. Note the results argue **against** cache
capacity being the operative variable — see finding 3.

## The pool-width confound — measured

If the pool spawned 24 threads onto 8 pinned cores, the P-only arm would measure oversubscription
rather than heterogeneity. CPU parallelism here is **OpenMP/libgomp** (`cc_parallel_grid=auto`
probes `dispatch` first — macOS-only — then `-fopenmp`,
[cc_backend.ml:278](../arrayjit/lib/cc_backend.ml:278)). Two widths matter, and they behave
differently.

**1. OpenMP thread count — affinity-aware; no knob needed.** Measured with a purpose-built probe
(`GetProcessAffinityMask` + `omp_get_num_procs` / `omp_get_max_threads` / threads observed inside a
parallel region), same compiler, under each mask:

| arm | mask | LPs in mask | `omp_get_num_procs` | `omp_get_max_threads` | threads in region |
|---|---|---|---|---|---|
| unpinned | `0xFFFFFF` | 24 | 24 | 24 | **24** |
| P-only | `0xC03C03` | 8 | 8 | 8 | **8** |
| E-only | `0x3FC3FC` | 16 | 16 | 16 | **16** |

mingw libgomp reads the process affinity mask, so pool width self-matches each arm. Forcing
`OMP_NUM_THREADS` to 8 / 16 reproduced these exactly, confirming the default is genuinely correct
rather than coincidentally equal. **The oversubscription trap does not fire on this platform** —
the opposite of the Linux-intuition failure mode, and a per-platform property worth re-checking
before the assumption is reused elsewhere.

**2. Grid chunk count — affinity-blind; this one needs pinning.** `cc_parallel_chunks` defaults to
`4 x Domain.recommended_domain_count ()` ([cc_backend.ml:427](../arrayjit/lib/cc_backend.ml:427),
[ocannl_config.reference:252](../ocannl_config.reference:252)):

| arm | mask | `Domain.recommended_domain_count ()` | auto chunks | chunks/thread |
|---|---|---|---|---|
| unpinned | `0xFFFFFF` | 24 | 96 | 4 |
| P-only | `0xC03C03` | **24** | **96** | **12** |
| E-only | `0x3FC3FC` | **24** | **96** | **6** |

OCaml's `recommended_domain_count` ignores the affinity mask on Windows. Not oversubscription
(chunks are work units), but an uncontrolled asymmetry across exactly the arms being compared.
**Every arm therefore pins `--ocannl_cc_parallel_chunks` to `4 x threads-in-mask`** — 96 / 32 / 64 —
holding chunks-per-thread constant at 4.

A finding in its own right: on Windows, benchmarking a pinned subset of cores silently yields a
mismatched grid decomposition unless this is set explicitly.

## Decomposition

`tune ratio` = default p50 / tuned p50; tuned p50 from the replay pass. All times ms.

### cifar_conv / cc

| arm | mask | threads | chunks | default | materialized | tuned | **tune ratio** | search `compile_s` |
|---|---|---|---|---|---|---|---|---|
| rog WSL (gh-530 ledger) | — | 24 | — | 1371.450 | — | 892.680 | **1.54x** | — |
| native unpinned, rep 1 | `0xFFFFFF` | 24 | 96 | 1377.18 | 1164.80 | 1171.32 | **1.176x** | 4798.2 |
| native unpinned, rep 2 | `0xFFFFFF` | 24 | 96 | 1342.17 | — | 1162.57 | **1.154x** | 5081.0 |
| native P-only, rep 1 | `0xC03C03` | 8 | 32 | 1346.96 | 1160.38 | **768.71** | **1.752x** | 2268.8 |
| native P-only, rep 2 | `0xC03C03` | 8 | 32 | 1345.07 | — | **768.46** | **1.750x** | 2454.8 |
| native E-only, rep 1 | `0x3FC3FC` | 16 | 64 | 1521.98 | 1306.16 | **793.12** | **1.919x** | 2634.5 |
| native E-only, rep 2 | `0x3FC3FC` | 16 | 64 | 1517.02 | — | **787.27** | **1.927x** | 2826.4 |

Replicate spread: 2.0% (unpinned), 0.1% (P-only), 0.4% (E-only) — far tighter than the 20–40%
cell-level spread documented in [report-gh481-cuda.md](report-gh481-cuda.md), and much smaller than
the 10–65% separation between arms.

### cifar_stride / cc

| arm | mask | threads | chunks | default | materialized | tuned | **tune ratio** | search `compile_s` |
|---|---|---|---|---|---|---|---|---|
| rog WSL (gh-530 ledger) | — | 24 | — | 355.019 | 315.025 | 321.279 | **1.10x** | — |
| native unpinned | `0xFFFFFF` | 24 | 96 | 353.132 | 309.442 | 310.927 | **1.136x** | 1658.6 |
| native P-only | `0xC03C03` | 8 | 32 | 348.50 | 309.831 | **229.56** | **1.518x** | 830.8 |
| native E-only | `0x3FC3FC` | 16 | 64 | 396.126 | 346.075 | **233.24** | **1.698x** | 973.2 |

Ledger reference points: minix (Zen 5) 2.48x / 1.87x; M4 Max 3.19x / 2.46x.

### Crown vs materialize-all (same-toolchain control)

| cell | arm | crown vs materialized |
|---|---|---|
| cifar_conv | unpinned (mixed) | **crown LOSES 0.6%** |
| cifar_conv | P-only | crown wins 33.8% |
| cifar_conv | E-only | crown wins 39.3% |
| cifar_stride | unpinned (mixed) | **crown LOSES 0.5%** |
| cifar_stride | P-only | crown wins 25.9% |
| cifar_stride | E-only | crown wins 32.6% |

### Cross-arm schedule replay (the mechanism experiment)

The P-only-crowned `cifar_conv` schedule, replayed from its `autotune_cache_dir` on the full mixed
machine (`compile_s` ~9 s confirms replay, not re-search):

| schedule | executed on | p50 |
|---|---|---|
| P-only crown | 8 P-cores (uniform) | **768.71** |
| P-only crown | 24 mixed, chunks=32 | 1335.45 |
| P-only crown | 24 mixed, chunks=96 | 1332.39 |
| unpinned crown | 24 mixed | 1171.32 |
| materialize-all | 24 mixed | 1164.80 |

Identical at both chunk counts, so this is not a decomposition artifact.

## Findings

**1. The virtualization/scheduling tax is essentially zero.** Both *default* baselines port
WSL→native within 0.5% (1377.18 vs 1371.450; 353.132 vs 355.019), across different operating
systems *and* compilers. WSL2 was not distorting rog's numbers; the outlier is a property of the
machine.

**2. Removing either core type recovers the tuning ratio, on both cells.** `cifar_conv`
1.15–1.18 → 1.75x (P-only) → 1.92–1.93x (E-only); `cifar_stride` 1.14 → 1.52x → 1.70x. P-only's
default baseline barely moves (−2.2%, −1.3%) while its tuned side improves 34% and 26%: the gain is
entirely on the tuned side, which is the signature heterogeneity predicts — the tuner's better
schedules are chunked parallel loops ending at a barrier, so a straggler worker costs the whole
step.

**3. The operative variable is uniformity, not core strength.** Sixteen *weak* uniform E-cores
produce a *better* tuning ratio than eight *strong* uniform P-cores, on both cells — and do so from
the **worst** default baseline of the three arms (1521.98 and 396.126). If core quality or cache
capacity were the cause, E-only should be the worst arm: its cores are ~1.5x slower and four share
4 MB of L2 where a P-core owns 3 MB privately. It is instead the best.

**4. On the mixed machine the search yields nothing over materialize-all.** The crown loses to
plain `BENCH_MATERIALIZE=1` by 0.6% / 0.5% — the whole apparent 1.15–1.18x / 1.14x is the placement
A/B selecting materialize-all, with the schedule search contributing nothing measurable. Two
independent from-scratch searches agreed to 0.06% with the materialized control. gh-530 recorded
this crown-loses-to-materialized behaviour as specific to `cifar_stride` on rog; natively it
reproduces on `cifar_conv` too, so it is not a strided-seed quirk.

**5. The tuner is not broken — the schedules genuinely do not pay on mixed cores.** The
cross-arm replay is the discriminating experiment: the P-only crown, which runs at 768.71 ms on
uniform cores, runs at **1332 ms on the mixed machine — worse than materialize-all's 1164.80**. So
on the mixed machine the search was not failing to *find* the good schedule; finding it would have
made things worse, and materialize-all really is the best available option there. The search
correctly reported that. **Richer or core-type-aware seeds would therefore not help.**

**6. Searching is much cheaper without E-cores** — `compile_s` 4798 → 2269 s (`cifar_conv`) and
1659 → 831 s (`cifar_stride`), roughly halved, since each candidate's timing run is faster and less
noisy. Separately, absolute search cost on Windows is ~8x the Linux/WSL figure for the same cell
(minix recorded 620.93 s for `cifar_conv`): the search spawns a compiler process per candidate and
Windows `CreateProcess` is far more expensive than `fork`/`exec`. A host property; it does not
affect step-time ratios.

**7. Within-native replication is tight.** 0.1–2.0% on the ratio across independent from-scratch
searches. The 31% gap between the native and WSL `cifar_conv` tuned cells is therefore *not*
run-to-run search noise on this box; it is a cross-toolchain/cross-environment difference, and the
arm ordering here is solid.

## Verdict

**The heterogeneity hypothesis is supported, sharpened, and its practical implication is inverted
relative to the issue's framing.**

The interpretation guide's threshold was "P-only recovers toward ≥~2x on `cifar_conv`,
`cifar_stride` tuned clearly beating materialized". Both hold: `cifar_stride`'s crown goes from
losing to materialize-all to beating it by 26–33%, and `cifar_conv` reaches 1.75–1.93x against the
mixed machine's 1.15–1.18x.

But the guide anticipated the conclusion "the seed parameterization needs core-type awareness".
Findings 3 and 5 say otherwise:

- It is not about *which* core type — any uniform pool works, and the weaker one works better.
- It is not a seeding or search deficiency — the search correctly rejects schedules that genuinely
  lose on mixed cores.

**The actionable lever is pool uniformity, not seed richness**: on a hybrid part, restricting the
worker pool to one core class makes the existing seeds worth 26–39%, where today they are worth
nothing. A natural follow-up is whether `cc` should default to a single core class on hybrid
topologies (and, since `Domain.recommended_domain_count` is affinity-blind on Windows, whether the
chunk count should follow the mask).

## Reproduction

```
OMP threads follow the affinity mask automatically; chunks must be pinned:
  cd benchmarks
  BENCH_FIXTURE=fixtures/cifar_conv.safetensors BENCH_TUNE=1 \
  AFFMASK=C03C03 pin.bat bench_conv.exe \
      --ocannl_backend=cc --ocannl_cc_parallel_chunks=32 \
      --ocannl_autotune_cache_dir=<fresh dir>
where pin.bat is:  start "" /affinity %AFFMASK% /wait /b %*
```

The empty title argument is required — `START` otherwise consumes a quoted command path as a window
title and silently launches a shell instead.

## Environment note (Windows)

Smart App Control was enforced on this machine and blocked the whole toolchain — `dune.exe`,
`ocamlopt.exe`, `flexlink.exe`, `x86_64-w64-mingw32-gcc.exe` and its sub-tools (cygwin-hosted gcc
reports the blocked subprocess as the misleading `spawn: Bad file descriptor`). Deleting and
re-copying each binary clears it for anything with cloud reputation, because SAC caches a stale
*block* verdict per file identity; it does **not** help a locally linked executable, which has no
reputation to find. Running OCANNL's own build products required SAC to be off. `Unblock-File` is
irrelevant (no Mark-of-the-Web on any of them) and SAC has no exclusion list.
