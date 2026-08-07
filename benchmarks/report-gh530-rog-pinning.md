# gh-530: decomposing rog's conv-sketch tuning-ratio outlier (native Windows, core pinning)

Native-Windows measurement session decomposing rog's tuning-ratio outlier. Measurement only -- no
seed or schedule code was changed.

**Verdict: two factors, and they interact. Core heterogeneity is causal, but only at pool widths
above 8, and pool width matters independently.** At a matched width of 16 workers, a uniform pool
tunes 31% better than a mixed one; at width 8 the same contrast is worth under 5%. The full 24-wide
mixed machine is the worst case of both and is where the crowned schedule collapses to
materialize-all.

The WSL rows in gh-530 remain the cross-machine ledger; the numbers here diagnose, they do not
replace them.

## Provenance

| field | value |
|---|---|
| machine | rog -- Intel Core Ultra 9 275HX (Arrow Lake-HX), 24 cores / 24 LPs, no SMT |
| OS | Windows 11, 10.0.26200 build 26200 -- **native**, not WSL |
| compiler | mingw `x86_64-w64-mingw32-gcc` 14.4.0 (cygwin-hosted, opam sandbox) |
| OCaml / dune | 5.5.0 / 3.24.0 |
| backend | `cc` (OpenMP `parallel for` rendering) |
| RAM | 31.3 GB |
| box state | otherwise idle, no CUDA work. An idle WSL VM (`vmmemWSL`, ~2.4 GB RSS) would not stay shut down, but measured **0 CPU-seconds per 10 s wall** -- parked, not competing. |
| pinning | `start "" /affinity <mask> /wait /b` (children inherit; verified live via `Process.ProcessorAffinity` on running cells of every arm) |
| protocol | from-scratch search per tuned cell against a **fresh `autotune_cache_dir`**; two-pass protocol (pass 1 searches, its `compile_s` is the search cost; a fresh pass 2 replays the cached winner for step times). All tuned p50 values below are replay-pass. |
| replication | `cifar_conv` n=2 on the three original arms; the width-controlled arms are n=1 |

Absolute times are comparable *within* this report only. Ratios are the claim-bearing quantity.

## Verified core-type map

Established empirically in a prior session by pinned spin-loop timing (clean 1.5x bimodal split):

| class | logical processors | affinity mask |
|---|---|---|
| P-cores | 0, 1, 10-13, 22, 23 | `0xC03C03` |
| E-cores | 2-9, 14-21 | `0x3FC3FC` |

The naive `0-7` range is **two P-cores and six E-cores** -- not a P-core proxy.

## Arrow Lake cache geometry

From `Win32_CacheMemory`. Two decodings are needed and both are easy to get wrong: WMI `Level` is
offset by two (Level 3 = L1, 4 = L2, 5 = L3), and **`Associativity` is a CIM enumeration, not a way
count** (7 = 8-way, 8 = 16-way, 9 = 12-way). Sizes are per-class totals as WMI reports them.

| level | P-side (8 cores) | E-side (16 cores, 4 clusters) | shared |
|---|---|---|---|
| L1 | 384 KB total, 12-way; 512 KB total, 16-way | 512 KB total, 8-way; 1024 KB total, 8-way | -- |
| L2 | **24 MB** total, 12-way = 3 MB private per core | **16 MB** total, 16-way = 4 MB per 4-core cluster | -- |
| L3 | -- | -- | **36 MB**, 12-way, shared by all 24 |

Recorded as a ground-truth row. The results do not isolate cache capacity as a driver, so this is
reference data rather than evidence for the verdict.

## The pool-width confound

CPU parallelism here is **OpenMP/libgomp** (`cc_parallel_grid=auto` probes `dispatch` first --
macOS-only -- then `-fopenmp`, [cc_backend.ml:278](../arrayjit/lib/cc_backend.ml:278)). Two widths
matter and they behave differently.

**1. OpenMP thread count -- affinity-aware; no knob needed.** Measured with a purpose-built probe
(`GetProcessAffinityMask` plus `omp_get_num_procs` / `omp_get_max_threads` / threads observed inside
a parallel region), same compiler, under each mask:

| mask | LPs in mask | `omp_get_num_procs` | `omp_get_max_threads` | threads in region |
|---|---|---|---|---|
| `0xFFFFFF` | 24 | 24 | 24 | **24** |
| `0xC03C03` | 8 | 8 | 8 | **8** |
| `0x3FC3FC` | 16 | 16 | 16 | **16** |
| `0x3FC` | 8 | 8 | 8 | **8** |
| `0xC3F` | 8 | 8 | 8 | **8** |
| `0xC03FFF` | 16 | 16 | 16 | **16** |

mingw libgomp reads the process affinity mask, so pool width self-matches each arm. Forcing
`OMP_NUM_THREADS` reproduced these exactly. **The oversubscription trap does not fire on this
platform** -- the opposite of the Linux-intuition failure mode, and a per-platform property worth
re-checking before the assumption is reused.

**2. Grid chunk count -- affinity-blind; this one needs pinning.** `cc_parallel_chunks` defaults to
`4 x Domain.recommended_domain_count ()` ([cc_backend.ml:427](../arrayjit/lib/cc_backend.ml:427),
[ocannl_config.reference:252](../ocannl_config.reference:252)). `Domain.recommended_domain_count`
ignores the affinity mask on Windows -- it returns 24 under every mask above, so the auto chunk
count is 96 regardless. **Every arm therefore pins `--ocannl_cc_parallel_chunks` to
`4 x threads-in-mask`**, holding chunks-per-thread constant at 4.

A finding in its own right: on Windows, benchmarking a pinned subset of cores silently yields a
mismatched grid decomposition unless this is set explicitly.

## Composition x width (cifar_conv)

The first three arms varied core composition and pool width together, which cannot separate them.
The width-controlled arms below hold width fixed and vary only composition. All arms use
chunks = 4 x threads; tuned p50 is replay-pass.

| arm | mask | width | composition | default | tuned | **tune ratio** |
|---|---|---|---|---|---|---|
| uniform-P | `0xC03C03` | 8 | 8P | 1346.96 | 768.71 | **1.752x** |
| uniform-E | `0x3FC` | 8 | 8E | 1532.54 | 730.98 | **2.097x** |
| MIXED | `0xC3F` | 8 | 4P + 4E | 1353.26 | 811.74 | **1.667x** |
| uniform-E | `0x3FC3FC` | 16 | 16E | 1521.98 | 793.12 | **1.919x** |
| MIXED | `0xC03FFF` | 16 | 8P + 8E | 1342.76 | 1014.56 | **1.323x** |
| MIXED (full machine) | `0xFFFFFF` | 24 | 8P + 16E | 1377.18 | 1171.32 | **1.176x** |

Composition effect at matched width:

| width | uniform | mixed | effect |
|---|---|---|---|
| 8 | 1.752x (P) | 1.667x | **-4.9%** |
| 16 | 1.919x (E) | 1.323x | **-31.0%** |

A width-24 uniform arm is not constructible on this part: only 8 P-cores and 16 E-cores exist.

## Original three-arm decomposition

Retained because it is what the ledger rows compare against, and because `cifar_stride` was only
measured this way. Note these arms vary composition and width together.

### cifar_conv / cc

| arm | width | default | materialized | tuned | **tune ratio** | search `compile_s` |
|---|---|---|---|---|---|---|
| rog WSL (gh-530 ledger) | 24 | 1371.450 | -- | 892.680 | **1.54x** | -- |
| native unpinned, rep 1 | 24 | 1377.18 | 1164.80 | 1171.32 | **1.176x** | 4798.2 |
| native unpinned, rep 2 | 24 | 1342.17 | -- | 1162.57 | **1.154x** | 5081.0 |
| native P-only, rep 1 | 8 | 1346.96 | 1160.38 | 768.71 | **1.752x** | 2268.8 |
| native P-only, rep 2 | 8 | 1345.07 | -- | 768.46 | **1.750x** | 2454.8 |
| native E-only, rep 1 | 16 | 1521.98 | 1306.16 | 793.12 | **1.919x** | 2634.5 |
| native E-only, rep 2 | 16 | 1517.02 | -- | 787.27 | **1.927x** | 2826.4 |

Replicate spread on the ratio: 2.0% (unpinned), 0.1% (P-only), 0.4% (E-only) -- far tighter than
the 20-40% cell-level spread documented in [report-gh481-cuda.md](report-gh481-cuda.md).

### cifar_stride / cc

| arm | width | default | materialized | tuned | **tune ratio** | search `compile_s` |
|---|---|---|---|---|---|---|
| rog WSL (gh-530 ledger) | 24 | 355.019 | 315.025 | 321.279 | **1.10x** | -- |
| native unpinned | 24 | 353.132 | 309.442 | 310.927 | **1.136x** | 1658.6 |
| native P-only | 8 | 348.50 | 309.831 | 229.56 | **1.518x** | 830.8 |
| native E-only | 16 | 396.126 | 346.075 | 233.24 | **1.698x** | 973.2 |

Ledger reference points: minix (Zen 5) 2.48x / 1.87x; M4 Max 3.19x / 2.46x.

### Crown vs materialize-all (same-toolchain control)

| cell | arm | crown vs materialized |
|---|---|---|
| cifar_conv | unpinned rep 1 | crown loses 0.56% |
| cifar_conv | unpinned rep 2 | crown wins 0.19% |
| cifar_conv | P-only | crown wins 33.8% |
| cifar_conv | E-only | crown wins 39.3% |
| cifar_stride | unpinned | crown loses 0.48% |
| cifar_stride | P-only | crown wins 25.9% |
| cifar_stride | E-only | crown wins 32.6% |

On the full machine the crown is within +/-0.6% of plain materialize-all in both replicates -- i.e.
indistinguishable from it. On every subset arm it wins by 26-39%.

### Cross-arm schedule replay

The P-only-crowned `cifar_conv` schedule replayed from its `autotune_cache_dir` on the full machine
(`compile_s` ~9 s confirms replay, not re-search):

| schedule | executed on | p50 |
|---|---|---|
| P-only crown | 8 P-cores | **768.71** |
| P-only crown | 24 mixed, chunks=32 | 1335.45 |
| P-only crown | 24 mixed, chunks=96 | 1332.39 |
| unpinned crown | 24 mixed | 1171.32 |
| materialize-all | 24 mixed | 1164.80 |

Identical at both chunk counts, so not a decomposition artifact. This tests one schedule moved from
an 8-worker to a 24-worker environment, so it does not isolate composition from width either.

## Findings

**1. The WSL and native baselines agree closely, which bounds the combined effect rather than
isolating virtualization.** Both default baselines port within 0.5% (1377.18 vs 1371.450; 353.132
vs 355.019). Because the OS *and* the compiler both changed between those rows, this shows the net
effect of all differences is small; it does not establish that WSL virtualization overhead
specifically is zero, since offsetting effects could cancel. What it does support is that the
outlier is reproducible outside WSL and is not an artifact of running under it.

**2. Core heterogeneity is causal at width 16, marginal at width 8.** With width and
chunks-per-thread held constant, a uniform pool tunes 31.0% better than a mixed one at width 16
(1.919x vs 1.323x), but only 4.9% better at width 8 (1.752x vs 1.667x). The single-width design in
the first version of this report could not have distinguished this from a pure width effect.

**3. Pool width matters independently, and the two interact.** The mixed arms degrade monotonically
with width -- 1.667x (8) -> 1.323x (16) -> 1.176x (24) -- while uniform arms barely move (2.097x at
8, 1.919x at 16). Notably the width-8 and width-16 mixed arms have the **same 50% E-core
proportion** yet differ by 20%, so the driver is not the proportion of slow workers alone; mixing
and pool size compound.

**4. On the full machine the search yields nothing over materialize-all.** The crown is within
+/-0.6% of the materialized control in both replicates, so the apparent 1.15-1.18x is the placement
A/B selecting materialize-all rather than any schedule the search found. gh-530 recorded this
crown-loses-to-materialized behaviour as specific to `cifar_stride` on rog; natively it reproduces
on `cifar_conv` too, so it is not a strided-seed quirk.

**5. The crowned P-only schedule does not transfer to the full machine.** It runs 768.71 ms on 8
P-cores and 1332 ms on the full machine -- worse than materialize-all's 1164.80. So for *this*
schedule, the search on the full machine was right to reject it. This does **not** show that no
good schedule exists there: the experiment moved one winner across both a composition and a width
change, and it did not evaluate candidates outside the current search space -- in particular, no
core-type-aware schedule (one weighting work by core class, or restricting execution to a uniform
subset) was constructed or measured. Whether such a candidate would pay on a hybrid pool is open.

**6. Searching is cheaper on narrower pools** -- `compile_s` 4798 s (width 24) vs 2269 s (width 8)
on `cifar_conv`, since each candidate's timing run is faster. Absolute search cost on Windows is
~8x the Linux/WSL figure for the same cell (minix recorded 620.93 s): the search spawns a compiler
process per candidate and Windows `CreateProcess` is far more expensive than `fork`/`exec`. A host
property; it does not affect step-time ratios.

**7. Within-native replication is tight** -- 0.1-2.0% on the ratio across independent from-scratch
searches, so the arm ordering is solid. The 31% gap between the native and WSL `cifar_conv` tuned
cells is therefore not run-to-run search noise on this box.

## Verdict

rog's outlier is produced by **pool width and core heterogeneity together**, not by heterogeneity
alone as the first version of this report claimed.

- Heterogeneity is causal: at matched width 16 a uniform pool tunes 31% better.
- But it is width-dependent: the same contrast is worth under 5% at width 8, so heterogeneity by
  itself does not explain the full-machine collapse.
- The two compound, and the full machine (24 wide, 67% E-cores) is the worst case of both.

For the seed parameterization this means **effective width is at least as important an input as
core composition**, and a fix targeting only one of them is unlikely to port. Restricting the `cc`
pool to a uniform subset is worth 26-39% on this machine, but the measurements do not establish
that it is the *best* available response, because no core-type-aware schedule was ever constructed
and measured -- that remains the open question, and it is the natural next experiment.

`cifar_stride` was measured only in the original three-arm form, so its numbers carry the same
composition/width confound; the width-controlled design was run on `cifar_conv` only.

## Reproduction

From a clean checkout, build the runner and generate the fixture first (`fixtures/` is gitignored):

```
dune build benchmarks/runners/ocannl/bench_conv.exe
python3 -m venv benchmarks/.venv
benchmarks/.venv/bin/pip install numpy safetensors
benchmarks/.venv/bin/python benchmarks/gen_fixtures.py benchmarks/workloads/cifar_conv.json
```

A tuned cell is two passes against the same cache: pass 1 searches, pass 2 replays it from a fresh
process. Reporting pass 1's step times measures a different protocol from the tables above, because
the search leaves its own process slower.

```
cd benchmarks
set BENCH_FIXTURE=fixtures/cifar_conv.safetensors
set BENCH_TUNE=1
set AFFMASK=C03C03
rem pass 1 -- from-scratch search; its compile_s is the search cost
pin.bat ../_build/default/benchmarks/runners/ocannl/bench_conv.exe ^
    --ocannl_backend=cc --ocannl_cc_parallel_chunks=32 ^
    --ocannl_autotune_cache_dir=%CD%\cache-p-only
rem pass 2 -- fresh process replays the cached winner; THIS is the reported p50
pin.bat ../_build/default/benchmarks/runners/ocannl/bench_conv.exe ^
    --ocannl_backend=cc --ocannl_cc_parallel_chunks=32 ^
    --ocannl_autotune_cache_dir=%CD%\cache-p-only
```

where `pin.bat` is:

```
@echo off
start "" /affinity %AFFMASK% /wait /b %*
```

The empty title argument is required -- `START` otherwise consumes a quoted command path as a
window title and silently launches a shell instead. Masks used: `C03C03` (8P), `3FC` (8E), `C3F`
(4P+4E), `3FC3FC` (16E), `C03FFF` (8P+8E), `FFFFFF` (all 24); set `cc_parallel_chunks` to 4x the
mask's population count.

## Environment note (Windows)

Smart App Control was enforced on this machine and blocked the whole toolchain -- `dune.exe`,
`ocamlopt.exe`, `flexlink.exe`, `x86_64-w64-mingw32-gcc.exe` and its sub-tools (cygwin-hosted gcc
reports the blocked subprocess as the misleading `spawn: Bad file descriptor`). Deleting and
re-copying each binary clears it for anything with cloud reputation, because SAC caches a stale
*block* verdict per file identity; it does **not** help a locally linked executable, which has no
reputation to find. Running OCANNL's own build products required SAC to be off. `Unblock-File` is
irrelevant (no Mark-of-the-Web on any of them) and SAC has no exclusion list.
