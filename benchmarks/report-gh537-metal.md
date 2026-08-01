# gh-ocannl-537 on Metal: the interchange reaches the dominant segment, and the artifact moves on lenet

Measurement-only report. The subject is the `Swap` ∘ `Split_reduce` seeding merged as staging PR
#257 (ahrefs/ocannl#537), whose test proves the *mechanism* — a bias-gradient site is detected with
a non-empty interchange and reaches timing — but not that it makes anything faster.

This is the Metal leg of the same question the CUDA leg answers in
[report-gh537-cuda.md](report-gh537-cuda.md). **It replicates the CUDA result closely**, including
the finding that matters: the split is decisive in the default-placement arm, and in the arm that
actually ships it pays only where the site is narrow.

Hardware: Apple silicon, Metal backend, macOS. All cells run serially on an otherwise idle machine,
no foreign GPU process during any cell.

**A/B pair.** `NEW` = staging master `ad46ab21`. `OLD` = the same tree with one branch of
`Autotune.split_reduce_sites` disabled — the `` `Hoist `` re-probe gh-537 added
(`arrayjit/lib/autotune.ml`, in `splittable`) — so it returns exactly its pre-gh-537 verdicts. The
library delta is that one branch and nothing else; the `NEW` binary was rebuilt after restoring the
source and is **byte-identical** (`592f79ec…`) to the one measured. Both binaries kept side by side
and run **interleaved in a single session**, two replicates per side per workload, fresh autotune
cache per search.

`ad46ab21` includes ahrefs/ocannl#532's fix (staging #262). Every search below logs
`baseline: NOT DISPATCHED, binds no hardware dimension on metal`, so none of these numbers include a
serial-baseline dispatch. Measuring before that landed would have compared different candidate
pools.

## Headline

| | OLD | NEW |
|---|---|---|
| sites detected (all three workloads) | 2, both in the classifier head | **4, including `bias_conv1.grad`** |
| `bias_conv1.grad` reached via | not reached | **3 swaps** (lenet `i530^i529 i530^i528 i530^i527`) |
| crowned schedule | `F_saved` | **`F_split_saved`, both arms** |

| workload | arm A (default placements) | | arm B (materialize-all — **ships**) | | shipped artifact p50 | |
|---|---|---|---|---|---|---|
| | OLD | NEW | OLD | NEW | OLD | NEW |
| lenet | 249.12 / 249.13 ms | **39.86 / 39.85 ms** (6.3x) | 35.43 / 35.38 ms | **33.18 / 33.18 ms** (−6.2%) | 35.565 / 35.621 ms | **33.227 / 33.219 ms** (−6.6%) |
| cifar_stride | 241.51 / 239.86 ms | **122.54 / 122.10 ms** (2.0x) | 95.76 / 95.73 ms | 95.46 / 95.11 ms (−0.5%) | 95.86 / 95.75 ms | 96.20 / 96.01 ms (+0.3%) |
| cifar_conv | 1270.28 / 1269.89 ms | **369.44 / 368.96 ms** (3.4x) | 276.12 / 276.09 ms | 275.47 / 275.36 ms (−0.25%) | 276.94 / 276.64 ms | 277.44 / 277.29 ms (+0.2%) |

Within-side spread on the shipped artifact is 0.16% (lenet OLD) and 0.02% (lenet NEW), so lenet's
−6.6% is roughly 40x its noise. The cifar deltas are inside their spread: neutral, not harmful.

## The per-site attribution, in one process

The cleanest evidence, needing no cross-run comparison. One `NEW` lenet search, arm A, same process,
same code — only the schedule differs between lines:

```
F_preset[bs=cfg cfg-thresh]:                      250.86 ms
F_preset[bs=64 priv]:                             249.10 ms
F_split[cross_entropy red64 out1 b32]:            250.59 ms
F_split[bias_conv1.grad red64 out6 b32 swap3]:     44.40 ms   <---
F_split[b_logits.grad red64 out10 b32 swap1]:     250.45 ms
F_split[bias_conv2.grad red64 out16 b32 swap3]:   249.52 ms
F_split[all four sites]:                           39.86 ms
untuned-default in-process control:               250.42 ms
```

Every preset and every *other* split site lands at 249–251 ms. Splitting **one** site —
`bias_conv1.grad`, the write of the nest #537 names, `loops[64s,28s,28s,6s]` producing 6 cells —
takes the step to 44.40 ms (5.6x); the four-site composite to 39.86 ms (6.3x). The classifier-head
sites gh-484 already reached move nothing, exactly as both predecessor legs found.

## What the split had to remove

`BENCH_SEG_TIMES=1` on `bench_conv_diag`, `NEW`, min of 20 per segment. Default placement:

| workload, default placements | seg22 `bias_conv1.grad n65.grad` | total | share |
|---|---|---|---|
| lenet | **214.49 ms** | 256.41 ms | **83.6%** |
| cifar_conv | **956.25 ms** | 1305.00 ms | **73.3%** |
| cifar_stride | **139.61 ms** | 258.30 ms | **54.1%** |

Against the arm-A results above, gh-537 removes ~98% of that segment on lenet (249.1 → 39.9 ms),
~94% on cifar_conv (1270 → 369 ms) and ~84% on cifar_stride (240 → 122 ms). **The claim #537 was
filed on is confirmed on a second backend, and the fix works on the segment it targets.**

### Instrument caveat

Per-segment sums over-count the fused step, because each segment is compiled and dispatched as its
own routine. Default placement is close (lenet 256.41 vs a 250.42 ms in-process control, +2.4%), but
the materialized placement has more, smaller segments and the fixed per-dispatch cost shows: lenet
44.71 vs 36.69 ms, **+21.9%**. Use the shares within a placement; do not read a seg-time total as a
step time. (The CUDA leg reported 0.25% agreement on the default placement only.)

## Why the shipping artifact moves only on lenet

`Train.tune_placements` searches two arms and keeps the measured winner. On all three conv workloads
the materialize-all arm (B) wins, because materializing intermediates already routes around the
serial nest by a different mechanism. gh-537 nearly closes that gap in arm A; it changes which arm
*could* ship, not which one does.

Inside arm B the picture tracks the site's output width. lenet's `bias_conv1.grad` has 6 cells and
the split wins outright:

| lenet arm B, one NEW search | ms |
|---|---|
| best `F_preset` (bs=cfg priv cfg-thresh) | 35.35 |
| `F_split[cross_entropy]` | 36.81 |
| `F_split[b_logits.grad swap1]` | 36.79 |
| `F_split[bias_conv2.grad swap3]` | 36.13 |
| **`F_split[bias_conv1.grad swap3]`** | **33.62** |
| `F_split[all four]` | **33.18** |
| untuned-default in-process control | 36.69 |

On the cifar workloads the same site has 32 cells and the split is actively worse, so the tuner
measures it and declines:

| cifar_conv arm B, one NEW search | ms |
|---|---|
| **best `F_preset` (bs=cfg priv cfg-thresh)** | **275.47** |
| `F_split[b_logits.grad swap1]` | 327.39 |
| `F_split[bias_conv1.grad out32 swap3]` | 330.07 |
| `F_split[bias_conv2.grad out64 swap3]` | 327.46 |
| untuned-default in-process control | 324.92 |

That is the mechanism behaving correctly — a candidate that does not help is measured and dropped —
so the neutral cifar cells are "no gain", not "harm". Cost: two extra candidates per arm.

### The dominant segment of the arm that ships is a *different* one

`BENCH_MATERIALIZE=1 BENCH_SEG_TIMES=1`, the placement that actually ships:

| workload | top segment | ms | share of total |
|---|---|---|---|
| lenet | seg33 `kernel_conv1.grad bias_conv1.grad` + SGD update | 21.93 | 49.1% of 44.71 |
| cifar_conv | seg33 `kernel_conv1.grad bias_conv1.grad` | 181.86 | 53.5% of 339.79 |
| cifar_stride | seg33 `kernel_conv1.grad bias_conv1.grad` | 66.50 | 53.3% of 124.72 |

All three fuse the **kernel** gradient with the bias gradient, and gh-537 splits only the bias half.
Identical to the CUDA leg's conclusion, on a third workload as well.

## `sr_max_sites = 4` binds here too — and it evicts a site that was already seeded

Same finding as the CUDA leg, with one Metal-specific detail worth recording. `OLD` detects two
sites, `cross_entropy` (64/1) and **`n105`** (84/640). `NEW` detects four, and `n105` is **gone**:
the ranking is `sr_red / sr_out` by integer division, so `n105` scores 0 and the three
newly-reachable bias sites evict it at the cap.

So gh-537 does not only add sites — on this graph it also drops one that was previously seeded. That
costs nothing here: `OLD` timed `F_split[n105 red84 out640 b32]` at 250.51 ms in arm A against a
250.51 ms control, and 36.65 ms in arm B against a 36.69 ms control. Inert on both arms, exactly as
gh-484's leg found. But the eviction is a real consequence of the cap binding, and a workload whose
ratio-0 site *did* pay would silently lose it.

## Numerics and search cost

Search cost (wall time per tuned run, both passes) is within run-to-run variation and shows no
systematic cost for the extra candidates:

| workload | OLD | NEW |
|---|---|---|
| lenet | 111 / 99 s | 107 / 102 s |
| cifar_stride | 142 / 129 s | 149 / 137 s |
| cifar_conv | 254 / 250 s | 266 / 254 s |

## An incidental result: `cifar_conv metal/tuned` did not hang

That cell is in `orchestrate.py`'s `SKIP_CELLS` because "the search completes but the post-tune
re-init hangs the process" (Metal reinit-after-tune race, PR #109/#174). All four `cifar_conv`
searches here completed normally in ~250 s each and emitted their result line. This is **not**
sufficient to drop the entry — `bench_conv` under `BENCH_TUNE=1` is not byte-for-byte the cell
orchestrate runs — but it is evidence the entry may be stale, and it is cheap to retest with
`--no-skip-cells`, which is exactly what that flag exists for.

## Reproducing

```bash
# Which sites, and through which interchange
BENCH_FIXTURE=fixtures/lenet.safetensors BENCH_SR_SITES=1 \
  ../_build/default/benchmarks/runners/ocannl/bench_conv_diag.exe --ocannl_backend=metal

# The per-site attribution
BENCH_FIXTURE=fixtures/lenet.safetensors BENCH_TUNE=1 \
  ../_build/default/benchmarks/runners/ocannl/bench_conv.exe --ocannl_backend=metal \
  --ocannl_autotune_cache_dir=/tmp/fresh --ocannl_autotune_log=true 2>&1 \
  | grep -E 'F_split\[|F_preset\[|control'

# What the split has to remove, in each placement
BENCH_FIXTURE=fixtures/lenet.safetensors BENCH_SEG_TIMES=1 [BENCH_MATERIALIZE=1] \
  ../_build/default/benchmarks/runners/ocannl/bench_conv_diag.exe --ocannl_backend=metal
```

All run from `benchmarks/`. Note `timeout(1)` is absent on macOS; `perl -e 'alarm N; exec @ARGV'`
caps a run portably.

## What this leg suggests next

1. **Seed the weight gradients** — the CUDA leg's item 1, independently reinforced. On Metal the
   shipping arm's dominant segment is `kernel_conv1.grad` + `bias_conv1.grad` on *all three*
   workloads, at 49–54% of the step, and gh-537 touches only the bias half. `kernel_conv1.grad` is
   tagged `[hoistable]` and is excluded solely by `sr_max_sites = 4` under a ranking that sends
   every weight gradient to ratio 0.
2. **The cap can evict as well as exclude.** Ranking by `sr_red / sr_out` integer division is
   degenerate — every site with more cells than reduction extent ties at 0. Ranking by estimated
   segment cost would both admit the weight gradients and stop the eviction described above.
3. **Profitability tracks output width**, confirmed on a second backend: 6 cells −6.6%, 32 cells
   declined. A cheap predictor would let seeding skip candidates it can predict will lose.
4. **Retest the `cifar_conv metal/tuned` SKIP_CELLS entry** with `--no-skip-cells`.
