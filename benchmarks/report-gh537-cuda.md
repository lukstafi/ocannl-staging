# gh-ocannl-537 on CUDA: the interchange reaches the dominant segment and removes ~90% of it

Measurement-only report. The subject is the `Swap` ∘ `Split_reduce` seeding merged as staging PR
#257 (ahrefs/ocannl#537). Its test proves the *mechanism* — a bias-gradient site is detected with a
non-empty interchange and reaches timing. This leg answers the question the gh-484 → gh-537 arc
exists to answer: **does it make anything faster.**

It does, decisively, for the segment #537 names. Whether the *shipping* artifact gets faster is
workload-dependent, and the reason is interesting enough to be the main finding here. The
predecessor leg ([report-gh484-cuda.md](report-gh484-cuda.md)) found the split-reduce family
"seeded, timed cleanly, and inert" because the detector only ever found the classifier head. That
part is fully fixed.

Hardware: NVIDIA GeForce RTX 5070 Ti Laptop GPU (sm_120), driver 610.62 / CUDA API 13.3, toolkit
13.3, under WSL2; Intel Core Ultra 9 275HX (24C/24T). Same machine as
[report-gh484-cuda.md](report-gh484-cuda.md) and [example-report-cuda.md](example-report-cuda.md).

**A/B pair.** `NEW` = staging master `081e70e2`. `OLD` = the same tree with one branch of
`Autotune.split_reduce_sites` disabled — the `` `Hoist `` re-probe gh-537 added — so `splittable`
returns exactly its pre-gh-537 verdicts. The library delta is that one branch and nothing else; the
`NEW` binary was rebuilt after restoring the source and is byte-identical to the one measured. Both
binaries were kept side by side and run **interleaved in a single session**, per
[report-gh484-cuda.md](report-gh484-cuda.md)'s "Measurement hygiene": on this box only paired
same-window A/B is meaningful. Two replicates per side per workload; no foreign GPU process during
any cell.

## Headline

| | OLD | NEW |
|---|---|---|
| sites detected (all three workloads) | 2, both in the classifier head | **4, including `bias_conv1.grad`** |
| `bias_conv1.grad` reached via | not reached | **3 swaps** (`i530^i529 i530^i528 i530^i527`) |
| `split_reduce_timed`, lenet search | 6 (3/arm) | **10 (5/arm)** — 0 FAILED, 0 dedup |
| crowned schedule contains `F_split` | yes, classifier head | **yes, `bias_conv1.grad`** |

| workload | default-placement arm (A) | | shipping arm (B), replayed artifact | |
|---|---|---|---|---|
| | OLD | NEW | OLD | NEW |
| lenet | 114.7 / 114.3 ms | **12.2 / 15.8 ms** | 11.589 / 11.583 ms | **10.794 / 10.790 ms** (−6.9%) |
| cifar_stride | 50.78 / 50.82 ms | **31.30 / 39.70 ms** | 33.06 / 31.19 ms | 31.17 / 31.20 ms (neutral) |
| cifar_conv | 522.5 / 524.9 ms | **122.3 / 124.8 ms** | 81.54 / 81.64 ms | 81.73 / 81.61 ms (neutral) |

## The per-site attribution, in one process

This is the cleanest evidence and needs no cross-run comparison. One `NEW` lenet search, arm A,
same process, same code — only the schedule differs between lines:

```
F_preset[bs=cfg cfg-thresh]:                     114.7615 ms
F_preset[bs=64 priv]:                            114.4498 ms
F_split[cross_entropy red64 out1 b32]:           114.7957 ms
F_split[bias_conv1.grad red64 out6 b32 swap3]:    13.8552 ms   <---
F_split[b_logits.grad red64 out10 b32 swap1]:    114.3252 ms
F_split[bias_conv2.grad red64 out16 b32 swap3]:  113.2314 ms
F_split[all four sites]:                          12.2449 ms
untuned-default in-process control:              114.5708 ms
```

Every preset, and every *other* split site, lands at 113–115 ms. Splitting **one** site —
`bias_conv1.grad`, the write of the nest #537 names, `loops[64s,28s,28s,6s]` producing 6 cells —
takes the step to **13.86 ms, an 8.3x speedup**; the four-site composite to 12.24 ms (9.4x). The
classifier-head sites gh-484 did reach move nothing, exactly as the predecessor leg found.

`BENCH_SEG_TIMES=1` on the same (default) pipeline says what was removed:

| lenet, default placements | ms | share |
|---|---|---|
| seg22 `bias_conv1.grad n65.grad` | **102.47** | **87.5%** |
| seg23 (SGD update + `kernel_conv1.grad`) | 8.35 | 7.1% |
| everything else (22 segments) | 6.24 | 5.3% |
| total | 117.06 | |

| cifar_conv, default placements | ms | share |
|---|---|---|
| seg22 `bias_conv1.grad n65.grad` | **426.36** | **82.1%** |
| seg23 `kernel_conv1.grad` | 44.80 | 8.6% |
| everything else | 48.37 | 9.3% |
| total | 519.53 | |

117.06 / 519.53 ms match the in-process untuned controls (114.6 / 522.5 ms) to within noise, and the
tuned arm-A results (12.2 / 122.3 ms) are what remains once that segment is split. So gh-537 removes
roughly 90% of seg22 on lenet and 94% of it on cifar_conv. **The claim #537 was filed on is
confirmed and the fix works on the segment it targets.**

## Why the shipping artifact moves only on lenet

`Train.tune_placements` searches two arms and keeps the measured winner. On every conv workload the
materialize-all arm (B) wins, because materializing the intermediates already routes around the
serial nest by a different mechanism — arm B's untuned default is 11.7 ms against arm A's 114.6 ms
on lenet. gh-537 nearly closes that gap in arm A; it changes which arm *could* ship, not which one
does.

In arm B the picture depends on the site's output width. lenet's `bias_conv1.grad` has 6 cells and
the split still wins:

| lenet arm B, one NEW search | ms |
|---|---|
| best `F_preset` (bs=64) | 11.6587 |
| `F_split[cross_entropy]` | 11.6977 |
| `F_split[b_logits.grad swap1]` | 11.7424 |
| `F_split[bias_conv2.grad swap3]` | 11.6702 |
| **`F_split[bias_conv1.grad swap3]`** | **10.8443** |
| untuned-default in-process control | 11.7521 |

−7.0% against the best preset, −7.7% against the untuned default, same process. On the cifar
workloads the same site has 32 cells and the split is actively worse, so the tuner declines it:

| cifar_conv arm B, one NEW search | ms |
|---|---|
| **best `F_preset` (bs=cfg priv cfg-thresh)** | **81.8007** |
| `F_split[b_logits.grad swap1]` | 86.1172 |
| `F_split[bias_conv2.grad out64 swap3]` | 95.5211 |
| `F_split[bias_conv1.grad out32 swap3]` | 131.7049 |
| untuned-default in-process control | 84.6932 |

That is the mechanism behaving correctly — a candidate that does not help is measured and dropped —
so the neutral cifar cells are "no gain", not "harm". Cost: two extra candidates per arm.

### The dominant segment of the arm that ships is a *different* one

`BENCH_MATERIALIZE=1 BENCH_SEG_TIMES=1`, the placement that actually ships:

| workload | top segment | ms | share of total |
|---|---|---|---|
| lenet | seg33 `kernel_conv1.grad bias_conv1.grad` + SGD update | 7.39 | 59.5% of 12.41 |
| cifar_conv | seg33 `kernel_conv1.grad bias_conv1.grad` | 49.24 | 49.3% of 99.96 |

Both fuse the **kernel** gradient with the bias gradient. gh-537 splits only the bias half — worth
7% on lenet, where 6 output cells make the split profitable, and nothing on cifar_conv, where the
kernel half dominates and the 32-cell bias split is a loss.

## `sr_max_sites = 4` is now binding, and it is what excludes the weight gradients

The probe's `[hoistable: …]` tags show the interchange would also unblock every weight gradient —
`kernel_conv1.grad` (150 cells), `kernel_conv2.grad` (2400), `w_logits.grad` (840), `b_fc1.grad`,
`b_fc2.grad`. None is proposed, because `split_reduce_sites` ranks by `sr_red / sr_out` and
truncates at `sr_max_sites = 4`. On lenet the ranking is 64/1, 64/6, 64/10, 64/16 — then everything
else at ratio **0** (integer division; every weight gradient has more cells than its reduction
extent). The four survivors are exactly the three biases plus the loss.

Before gh-537 the cap never bound (only 2 sites were found), so this is a new question rather than a
regression — and it is the obvious next experiment, because `kernel_conv1.grad` is precisely the
half of the shipping arm's dominant segment that gh-537 does not touch. This leg cannot say whether
splitting it would help: it was never seeded, so it was never timed.

## Numerics and search cost

Over all 24 parity steps, `cifar_stride` and `cifar_conv` losses are **bit-identical** between the
two binaries and `lenet` differs by at most **2.4e-07** — the same picture the predecessor leg
recorded, and consistent with gh-484 task 4's claim that a fixed schedule's combine tree is
deterministic (only lenet crowns a different schedule, so only lenet's combine order changes).
Search cost (pass-1 `compile_s`) is within run-to-run variation:

| workload | OLD | NEW |
|---|---|---|
| lenet | 71.6 / 66.6 s | 70.6 / 70.2 s |
| cifar_stride | 192.4 / 176.0 s | 187.6 / 181.0 s |
| cifar_conv | 555.6 / 547.6 s | 560.4 / 551.7 s |

Replay stability, for reference on how much of a delta is real: within-side spread was 0.06% on
lenet (both sides) and 0.13% on cifar_conv, but one OLD `cifar_stride` replay came in at 33.06 ms
against its sibling's 31.19 ms — the ~7% tuned-artifact instability the predecessor leg documented,
still present. The lenet delta is ~100x its noise; the cifar deltas are inside it.

## Which sites the detector now proposes

`BENCH_SR_SITES=1` on `bench_conv_diag`, `NEW`:

```
lenet:
split-reduce sites detected: 4
  cross_entropy:    reduction extent 64, target cells 1
  bias_conv1.grad:  reduction extent 64, target cells 6  (via 3 swaps: i530^i529 i530^i528 i530^i527)
  b_logits.grad:    reduction extent 64, target cells 10 (via 1 swap: i422^i421)
  bias_conv2.grad:  reduction extent 64, target cells 16 (via 3 swaps: i486^i485 i486^i484 i486^i483)

cifar_conv, cifar_stride (identical):
  cross_entropy:    reduction extent 64, target cells 1
  b_logits.grad:    reduction extent 64, target cells 10 (via 1 swap: i437^i436)
  bias_conv1.grad:  reduction extent 64, target cells 32 (via 3 swaps: i545^i544 i545^i543 i545^i542)
  bias_conv2.grad:  reduction extent 64, target cells 64 (via 3 swaps: i501^i500 i501^i499 i501^i498)
```

Against `OLD`'s two (`cross_entropy`, `n105`/`b_logits.grad`), both classifier head.

## A note on #537's stated verification instrument

#537 asks for verification "via `BENCH_SEG_TIMES=1` on `bench_conv_diag`". That instrument measures
the **default pipeline**, which an autotune-only change cannot affect — the predecessor leg made the
same observation about gh-484 task 3. It remains the right instrument for *locating* the segment and
for sizing what a split has to remove (both tables above), but the instrument that answers "did it
get faster" is the tuned candidate timing, where the split is actually applied. This report leads
with the latter and uses the former to size it.

## What this leg suggests next

1. **Seed the weight gradients.** `kernel_conv1.grad` is `[hoistable]` and is half of the shipping
   arm's dominant segment on both lenet and cifar_conv. It is excluded only by `sr_max_sites = 4`
   under a ranking that sends every weight gradient to ratio 0. Raising the cap, or ranking by
   estimated segment cost rather than `sr_red / sr_out`, is the experiment.
2. **The site's profitability tracks output width.** 6 cells: −7%. 32 cells: +60%. If a cheap
   predictor of that exists, the seeding could stop paying for candidates it can predict will lose.
3. **The 94.6%/91.4% framing needs a placement qualifier.** Those shares are of the *default*
   pipeline; the tuned artifact ships the materialized placement, whose step is 10x smaller and
   whose dominant segment is a different one. Reports that quote a segment share should say which
   placement it is a share of — see ahrefs/ocannl#538.
4. `cifar_stride`'s tuned artifact is still unstable at ~7% (predecessor leg item 3, unchanged).

## Reproducing

```bash
# Which sites, and through which interchange
BENCH_FIXTURE=fixtures/lenet.safetensors BENCH_SR_SITES=1 \
  ../_build/default/benchmarks/runners/ocannl/bench_conv_diag.exe --ocannl_backend=cuda

# The per-site attribution
BENCH_FIXTURE=fixtures/lenet.safetensors BENCH_TUNE=1 \
  ../_build/default/benchmarks/runners/ocannl/bench_conv.exe --ocannl_backend=cuda \
  --ocannl_autotune_cache_dir=/tmp/fresh --ocannl_autotune_log=true 2>&1 \
  | grep -E 'F_split\[|F_preset\[|control'

# What the split has to remove, in each placement
BENCH_FIXTURE=fixtures/lenet.safetensors BENCH_SEG_TIMES=1 [BENCH_MATERIALIZE=1] \
  ../_build/default/benchmarks/runners/ocannl/bench_conv_diag.exe --ocannl_backend=cuda
```

All run from `benchmarks/`. Raw logs for every cell are under `benchmarks/results/gh537-cuda-raw/`
(gitignored).
