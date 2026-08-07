# gh-530: decomposing rog's conv-sketch tuning-ratio outlier (native Windows, core pinning)

Native-Windows measurement session decomposing rog's tuning-ratio outlier. Measurement only -- no
seed or schedule code was changed.

**Verdict: two factors, both large, and they compound.** Against a like-for-like uniform control a
mixed pool tunes 20.5% worse at width 8 and 31.1% worse at width 16; independently, with grid
decomposition *and* composition held fixed, a mixed pool falls from 1.678x to 1.337x going from
width 8 to width 16. The full 24-wide machine, which is both the widest pool and the most
E-core-heavy (67%), is the worst case of both and is where the crowned schedule collapses to
materialize-all -- though that last step cannot be attributed to either factor alone, since no
composition-matched 24-wide pool exists on this part.

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
| replication | `cifar_conv` n=2 on the three original arms; the composition- and chunk-controlled arms are n=1 |

Absolute times are comparable *within* this report only. Ratios are the claim-bearing quantity.

## Correctness gating

A faster crown is only meaningful if it computes the same thing, so every timing below is gated on
the loss trajectory each run reports. AGENTS.md asks that executed output be asserted "against a
materialized or otherwise independent reference run"; the `BENCH_MATERIALIZE=1` arms provide
exactly that, since materialize-all is a different execution path from both the default
(recompute-intermediates) placement and from any tuned schedule.

Every arm's 24-step parity trajectory was compared against its cell's unpinned **default** run,
using orchestrate.py's own metric (`max |a-b| / max(|b|, 1e-6)`) and tolerance
(`PARITY_TOL = 2e-3`):

| cell | arms compared | worst relative deviation | tolerance | verdict |
|---|---|---|---|---|
| cifar_conv | 38 | **2.035e-7** | 2e-3 | PASS, ~4 orders of magnitude inside |
| cifar_stride | 12 | **2.083e-7** | 2e-3 | PASS, ~4 orders of magnitude inside |

Every *default* arm is bit-identical to the reference (deviation exactly 0), across all six pool
configurations. The uniform ~2e-7 on the tuned and materialized arms is fp32 ordering difference
between recomputing an intermediate and materializing it, not a semantic difference. No arm --
tuned, materialized, replayed, or cross-arm -- deviates measurably.

The trajectories also pass orchestrate.py's non-stationarity guard (`loss_moved`), which exists so
a tolerance cannot rubber-stamp an input-independent forward: `cifar_conv` moves 3.341e-2 over the
window against a 2.316e-6 threshold, `cifar_stride` 1.356e-2.

**Limitation.** This is a materialized/independent-placement reference, not orchestrate.py's
*cross-framework* gate against the PyTorch CPU reference. That gate could not be run here: torch
would not install into the worktree venv (Windows `MAX_PATH` -- torch's nested license tree exceeds
260 characters under this path), the Nsight-bundled CPython is stdlib-stripped and cannot import
it, and the WSL image carries Python 3.14 with no pip and no torch wheels. So these results are
gated against an independent *placement*, which would catch a miscompiling schedule, but not
against an independent *framework*, which would additionally catch an error shared by every OCANNL
placement. The gh-530 ledger rows for both cells did pass the cross-framework gate.

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

Composition effect at matched width. Only an E-cored uniform control exists at width 16 (there are
just 8 P-cores), so the E-vs-E column is the one comparable across widths; the width-8 P control is
shown too rather than picking whichever is more favourable.

| width | mixed | vs uniform-E | vs uniform-P |
|---|---|---|---|
| 8 | 1.667x | 2.097x -> **-20.5%** | 1.752x -> -4.9% |
| 16 | 1.323x | 1.919x -> **-31.1%** | n/a (no 16-P pool exists) |

Read against the consistent E control, composition costs 20.5% at width 8 and 31.1% at width 16.
Each of those is internally chunk-matched (both arms of each row use the same chunk count), so both
are clean *within-width* effects. The width-8 figure against the P control (-4.9%) is the same
mixed arm measured against a *weaker* uniform baseline -- 8 P-cores tune to 1.752x where 8 E-cores
reach 2.097x -- and quoting only that number understates the effect.

**On whether the penalty grows with width, the evidence is strong but one control is missing.**
Comparing 20.5% against 31.1% spans a chunk-count change (32 at width 8, 64 at width 16), so it is
worth checking that the penalty itself is chunk-invariant. Re-running the *width-8* pair with
everything at 96 chunks:

| width-8 arm | @ 32 chunks | @ 96 chunks | chunk sensitivity |
|---|---|---|---|
| uniform-E | 2.097x | 2.097x | **+0.02%** |
| mixed | 1.667x | 1.678x | +0.6% |
| **composition penalty** | **-20.5%** | **-20.0%** | -- |

The penalty is chunk-invariant at width 8, and the width-16 *mixed* arm is chunk-invariant too
(1.323x at 64 vs 1.337x at 96, +1.1%). The one arm not measured at a second chunk count is the
**width-16 uniform-E control**; it was started and stopped as not worth the search time, since
overturning the growth would require it to move ~13% between 64 and 96 chunks when every other arm
measured moved by at most 1.1%. So the growth from ~20% to ~31% is well supported but not
fully chunk-controlled at width 16, and that single arm would close it.

A width-24 uniform arm is not constructible on this part: only 8 P-cores and 16 E-cores exist.

### Width at constant chunk count

The arms above set chunks = 4 x threads, so pool width and grid decomposition co-vary and a width
claim cannot be read off them directly. These arms re-run the mixed pools at a **fixed absolute**
`cc_parallel_chunks=96`, the value the full machine already used:

| arm | mask | width | chunks | default | tuned | **tune ratio** |
|---|---|---|---|---|---|---|
| MIXED | `0xC3F` | 8 | 96 | 1364.79 | 813.54 | **1.678x** |
| MIXED | `0xC03FFF` | 16 | 96 | 1347.98 | 1007.88 | **1.337x** |
| MIXED (full machine) | `0xFFFFFF` | 24 | 96 | 1377.18 | 1171.32 | **1.176x** |

Chunk count turns out to barely matter for these pools: the same arm moves 0.6% between 32 and 96
chunks at width 8 (1.667x -> 1.678x) and 1.1% between 64 and 96 at width 16 (1.323x -> 1.337x),
despite a 3x and 1.5x change in decomposition. These are independent from-scratch searches, not one
replayed schedule.

**The controlled width interval is 8 -> 16 only.** Those two arms are both 50% E-cores, so with
chunks fixed at 96 the 1.678x -> 1.337x decline isolates width. The 16 -> 24 step is *not*
controlled: the full machine is 8P+16E, i.e. 67% E-cores, so that segment changes composition and
width together and cannot be attributed to either alone. A width-24 arm at 50% E-cores would
require 12 P-cores and is not constructible on this part, so this interval cannot be closed on
this machine.

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

**2. Core heterogeneity is causal at both widths measured.** Against the uniform-E control -- the
only control type available at both widths -- **the mixed pool tunes 20.5% worse at width 8 and
31.1% worse at width 16** (equivalently, uniform-E is 25.8% and 45.0% better; the percentages
differ because the denominators do, so the direction has to be stated). Both are within-width,
chunk-matched comparisons. Against the width-8 uniform-P control the same mixed arm is only 4.9%
worse, but that is a weaker baseline: 8 P-cores tune to 1.752x where 8 E-cores reach 2.097x. The
single-width design in the first version of this report could not have distinguished any of this
from a pure width effect.

The penalty appears to *grow* with width (20.5% -> 31.1%), and the width-8 pair re-run entirely at
96 chunks gives -20.0%, showing the penalty is chunk-invariant there. But the cross-width
comparison spans a chunk change, and the width-16 uniform-E control was not measured at a second
chunk count -- so the growth is well supported (it would need that arm to move ~13% where every
measured arm moved at most 1.1%) but not fully controlled. Treat "grows with width" as indicated,
not established.

**3. Pool width is causal over the interval where composition is controlled.** With
`cc_parallel_chunks` fixed at 96 and composition fixed at 50% E-cores, the mixed pools decline
1.678x (width 8) -> 1.337x (width 16). Chunk count is not the driver -- the same pool moves 0.6%
between 32 and 96 chunks at width 8 and 1.1% between 64 and 96 at width 16, across independent
from-scratch searches. The further fall to 1.176x at width 24 is **not** a controlled width result:
the full machine is 67% E-cores, so that segment varies composition and width together, and a
50%-E pool at width 24 would need 12 P-cores and cannot be built on this part. Uniform arms
meanwhile barely move with width (2.097x at 8, 1.919x at 16). Note also that the width-8 and
width-16 mixed arms share the **same 50% E-core proportion** yet differ by 20%, so the driver is
not the proportion of slow workers alone; mixing and pool size compound.

**4. On the full machine the search fails to crown the split-reduce family that wins on every
narrower pool.** The crown is within +/-0.6% of the materialized control in both replicates, and
the runner's emitted `tune` object says why rather than leaving it to be inferred from near-equal
timings. Every arm ships **B**, the materialize-all placement (arm A is default placements, arm B
materialize-all -- [bench_harness.ml:216](runners/ocannl/bench_harness.ml:216)), but the crowned
*schedule label* differs by pool:

| pool | arm A label | arm B label (shipped) |
|---|---|---|
| full machine, width 24 | `F_saved[15 segs]` | `F_saved[19 segs]` |
| every width-8 and width-16 pool | `F_split_saved[31 prelude ops, 15 segs]` | `F_split_saved[31 prelude ops, 30 segs]` |

On every subset the crown is a **split-reduce** schedule; on the full machine it is plain fission.
That is the concrete difference behind the collapse -- not "the tuner found nothing", but "the
tuner did not crown the split-reduce family here", and the shipped placement is materialize-all in
all cases. (`cifar_stride` unpinned crowns `W_saved[0 ops]` on arm B, a third label.) gh-530
recorded this crown-loses-to-materialized behaviour as specific to `cifar_stride` on rog; natively
it reproduces on `cifar_conv` too, so it is not a strided-seed quirk.

**5. The crowned P-only schedule does not transfer to the full machine.** It runs 768.71 ms on 8
P-cores and 1332 ms on the full machine -- worse than materialize-all's 1164.80. So for *this*
schedule, the search on the full machine was right to reject it. This does **not** show that no
good schedule exists there: the experiment moved one winner across both a composition and a width
change, and it did not evaluate candidates outside the current search space -- in particular, no
core-type-aware schedule (one weighting work by core class, or restricting execution to a uniform
subset) was constructed or measured. Whether such a candidate would pay on a hybrid pool is open.

**6. Searching is cheaper on narrower pools** -- `compile_s` 4798 s (width 24) vs 2269 s (width 8)
on `cifar_conv`, consistent with each candidate's timing run being faster on a pool that runs the
workload faster. Separately, and only as an observation: the same cell's search took ~8x longer
here than the 620.93 s minix recorded under WSL2/Linux. That comparison spans machine, OS, compiler
and possibly a different set of candidates, and no per-candidate count or compiler-launch cost was
measured, so it is reported as a cross-host difference in search time and is deliberately **not**
attributed to any particular cause. It does not affect step-time ratios either way.

**7. Within-native replication is tight, but only the original three arms are strictly
replicated.** Those three were run twice at identical settings:

| arm (n=2, identical config) | rep 1 | rep 2 | spread |
|---|---|---|---|
| unpinned, width 24 | 1.176x | 1.154x | 1.81% |
| P-only, width 8 | 1.752x | 1.750x | 0.11% |
| E-only, width 16 | 1.919x | 1.927x | 0.41% |

The composition- and chunk-controlled arms are **n=1 at any single setting**, so this spread does
not directly cover them and the causal conclusions in findings 2 and 3 do not inherit it. What
those arms do have is a second independent from-scratch search at a different chunk count, which is
a near-replicate rather than a replicate -- one factor differs, but that factor is measurably
inert here:

| controlled arm | search 1 | search 2 | difference |
|---|---|---|---|
| mixed, width 8 | 1.667x @ 32 | 1.678x @ 96 | 0.63% |
| mixed, width 16 | 1.323x @ 64 | 1.337x @ 96 | 1.05% |
| uniform-E, width 8 | 2.097x @ 32 | 2.097x @ 96 | 0.04% |

So every controlled arm has been searched twice from scratch and landed within 1.1% both times,
which is evidence that the beam is not selecting wildly different winners on these pools. It is
not a substitute for a true replicate, and the width-16 uniform-E control has only ever been
searched once. Read the controlled-arm ordering as well-supported but n=1; the strict replicate
spread above belongs to the original three arms only.

Separately, the 31% gap between the native and WSL `cifar_conv` tuned cells is much larger than any
of these figures, so it is not run-to-run search noise on this box.

## Verdict

rog's outlier is produced by **pool width and core heterogeneity together**, not by heterogeneity
alone as the first version of this report claimed. Both effects are large.

- Heterogeneity is causal at every width measured: against the uniform-E control a mixed pool loses
  20.5% at width 8 and 31.1% at width 16.
- Pool width is causal over the interval where composition is held fixed: with
  `cc_parallel_chunks` fixed at 96 and composition fixed at 50% E-cores, mixed pools decline
  1.678x -> 1.337x from width 8 to 16.
- The two compound. The full machine is both the widest pool and the most E-core-heavy (67%), and
  is the worst case; its 1.176x cannot be apportioned between the two factors, because no
  composition-matched 24-wide pool exists on this part.

For the seed parameterization this means **effective width and core composition are both
first-class inputs**, and a fix targeting only one is unlikely to port.

Restricting the `cc` pool to a uniform subset is worth **32-34% on `cifar_conv` and 25-26% on
`cifar_stride`** -- that is the full-machine tuned time against the subset tuned time (1171.32 ms
-> 768.71 / 793.12; 310.93 ms -> 229.56 / 233.24), which is what a restriction policy would
actually buy. It is not the same quantity as the 26-39% by which a subset's crown beats
materialize-all *within that subset*.

The measurements do not establish that restriction is the *best* available response, because no
core-type-aware schedule was ever constructed and measured -- that remains the open question and
the natural next experiment.

`cifar_stride` was measured only in the original three-arm form, so its numbers carry the same
composition/width confound; the width-controlled design was run on `cifar_conv` only.

## Reproduction

From a clean checkout, build the runner and generate the fixture first (`fixtures/` is gitignored):

Everything below is native Windows, so the venv's executables live under `Scripts\`, not `bin/`:

```
dune build benchmarks/runners/ocannl/bench_conv.exe
python -m venv benchmarks\.venv
benchmarks\.venv\Scripts\python.exe -m pip install numpy safetensors
benchmarks\.venv\Scripts\python.exe benchmarks\gen_fixtures.py benchmarks\workloads\cifar_conv.json
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
window title and silently launches a shell instead.

Masks: `C03C03` (8P), `3FC` (8E), `C3F` (4P+4E), `3FC3FC` (16E), `C03FFF` (8P+8E), `FFFFFF` (all
24). **The two experiments use different chunk rules and are not interchangeable:**

| experiment | cells | `cc_parallel_chunks` |
|---|---|---|
| composition at matched width | `C03C03`, `3FC`, `C3F` (width 8); `3FC3FC`, `C03FFF` (width 16) | **4x the mask's population count** (32, 32, 32, 64, 64) |
| width at fixed decomposition | `C3F`, `C03FFF`, `FFFFFF` | **96 for every cell** |

Applying the 4x rule to the width series would give 32/64/96 and reproduce the confounded table
instead of the claim-bearing one. Every tuned cell needs its own fresh `autotune_cache_dir`; reusing
one silently replays another cell's winner.

## Environment note (Windows)

Smart App Control was enforced on this machine and blocked the whole toolchain -- `dune.exe`,
`ocamlopt.exe`, `flexlink.exe`, `x86_64-w64-mingw32-gcc.exe` and its sub-tools (cygwin-hosted gcc
reports the blocked subprocess as the misleading `spawn: Bad file descriptor`). Deleting and
re-copying each binary clears it for anything with cloud reputation, because SAC caches a stale
*block* verdict per file identity; it does **not** help a locally linked executable, which has no
reputation to find. Running OCANNL's own build products required SAC to be off. `Unblock-File` is
irrelevant (no Mark-of-the-Web on any of them) and SAC has no exclusion list.
