# gh-ocannl-484 task 3 on CUDA: the split-reduce seeding is inert on the conv benchmarks

Measurement-only report. The subject is the autotune change merged as staging PR #246
(ahrefs/ocannl#484 task 3: detect reduction-dominated sites, seed `Split_reduce` as a candidate
family). The question this leg answers is whether that family reaches, and improves, the segment
that ahrefs/ocannl#476's CUDA leg ([example-report-cuda.md](example-report-cuda.md)) identified as
94.6% of the `lenet` step and 91.4% of the `cifar_conv` step.

Hardware: NVIDIA GeForce RTX 5070 Ti Laptop GPU (sm_120), driver 610.62 / CUDA API 13.3, toolkit
13.3, under WSL2; Intel Core Ultra 9 275HX (24C/24T). Same machine as
[example-report-cuda.md](example-report-cuda.md).

**A/B pair.** `OLD` = `6f0aa7a3`, the last commit before the task-3 series; `NEW` = `ff54fc3f`,
staging master with #246 merged. The library delta between them is exactly
`arrayjit/lib/autotune.{ml,mli}` + `arrayjit/lib/schedule_cache.mli` — the two seeding commits and
nothing else. Both binaries were kept built side by side and run **interleaved in a single
session**, for reasons in "Measurement hygiene" below.

## Headline

The family is **not** blocked the way ahrefs/ocannl#521 blocks the mma family, and it is **not**
worth its search cost on these workloads — because it never reaches the code that dominates them.

| | result |
|---|---|
| `split_reduce_timed` | **42 across 7 searches** (3 per `tune_placements` arm), **0 failed, 0 dedup'd** |
| `split_reduce_candidates` | 2 per arm (2 sites × 1 legal `num_blocks`) |
| step time, tuned artifact | **+0.10% to +0.35%** — neutral, inside run-to-run noise |
| search cost | **+4.9% to +12.9%** |
| numerics | losses bit-identical, or ≤4.8e-07 where not |
| sites detected | always the same two, both in the classifier head, in every workload and both placement arms |

## `split_reduce_timed` is not zero — ahrefs/ocannl#521 does not generalize

This was the specific risk worth checking: #521's lesson is that a seeded family can be seeded in
quantity and timed zero times, every candidate dying at candidate-compile. That does **not** happen
here. Across every logged search — `lenet`, `cifar_stride`, `cifar_conv`, both arms of
`Train.tune_placements`, seven searches in total:

```
F_split[cross_entropy red64 out1 b32]:                          <timed> ms
F_split[n107 red84 out640 b32]:                                 <timed> ms
F_split[cross_entropy red64 out1 b32,n107 red84 out640 b32]:    <timed> ms
```

42 timed, **0 `FAILED`, 0 `dedup`**. No `Low_level.validate_parallel` message appears against any
`F_split` candidate. This is the expected outcome and confirms the reasoning that motivated the
check: split reductions are ordinary data-parallel kernels, not mma candidates, so the
hardware-dimension coverage rule that kills the tensorized family has no purchase on them.

Positive control: `test/operations/autotune_split_reduce` passes all 15 assertions under
`OCANNL_BACKEND=cuda`, including `split-reduce candidates timed`.

Counting note: `split_reduce_candidates` counts the seeded singles (2), while `split_reduce_timed`
counts 3 — the two singles plus the multi-site composite that recombines each site's best-timed
`num_blocks`. Only `b=32` survives the `2*b <= extent` filter at these sites, so the GPU sweep
`[32; 128; 512]` contributes one value, not three.

## Why it is inert: the detector only ever finds the classifier head

`BENCH_SR_SITES=1` on `bench_conv_diag` (added by this leg) prints what
`Autotune.split_reduce_sites` proposes on the real graph. All three workloads, both the default and
the materialize-all placements, give the identical answer:

```
split-reduce sites detected: 2
  cross_entropy: reduction extent 64, target cells 1
  n107:          reduction extent 84, target cells 640
```

`cross_entropy` reduces over the batch (64) to the scalar loss; `n107` is the logits
(64 × 10 = 640) reducing over `fc2` = 84. They are identical across fixtures because all three
share `fc2=84` / `classes=10`. Both are the classifier head, which is a rounding error in these
steps.

The reductions that actually dominate are never proposed. From the same runs' census, lenet's conv1
gradient nests are:

```
loops[64s,28s,28s,6s]        w:bias_conv1.grad!(6)
loops[64s,28s,28s,6s,5s,5s]  w:kernel_conv1.grad!(150)
```

Both pass the two documented thresholds — 6 and 150 target cells are far below `sr_out_max = 4096`,
and the enclosing batch loop is `Serial` with extent 64, meeting `sr_red_min = 64`. So the published
gates do not explain the miss; the rejection is happening in the rmw self-read test, in
`Sched.op_legality`, or in the dedup-by-axis-symbol in `split_reduce_sites`. **Pinning down which is
the natural follow-up** — this leg measured the outcome, not the mechanism. (`sr_max_sites = 4` is
not binding: only 2 sites are found.)

This is what makes the step-time result uninteresting rather than disappointing: the transform is
not underperforming on the conv gradients, it is never applied to them.

## Step times: neutral

Tuned cells use the report's two-pass protocol (pass 1 = search, pass 2 = a fresh process replaying
the cached winner, which is what is timed). p50 of 100 synced steps.

| workload | OLD `6f0aa7a3` | NEW `ff54fc3f` | delta | samples |
|---|---|---|---|---|
| lenet | 11.732 ms | 11.772 ms | +0.34% | 1 per side |
| cifar_stride | 31.184 ms | 31.294 ms | +0.35% | 1 per side |
| cifar_conv | 101.384 ms (median) | 101.486 ms (median) | **+0.10%** | 4 / 5 |

Replay-noise band for reference: three replays of one cached lenet winner spanned 11.713–11.836 ms
(1.1%). All three deltas are inside it.

Search cost is the real change:

| workload | OLD | NEW | |
|---|---|---|---|
| lenet | 63.2 s | 71.4 s | +12.9% |
| cifar_stride | 175.6 s | 184.3 s | +4.9% |
| cifar_conv | 592.1 s (median) | 621.4 s (median) | +4.9% |

Numerics are unchanged: `cifar_conv` and `cifar_stride` losses are bit-identical between the two
binaries over all 24 parity steps, `lenet` differs by at most 4.8e-07. Consistent with #484 task 4's
claim that a fixed schedule's combine tree is deterministic.

## The family is crowned rarely, and it does not matter when it is

Worth recording because a single run is misleading here. `cifar_conv`, arm B (materialize-all —
the arm that ships on every conv workload), across five NEW searches:

| search | arm-B winner | arm-B best | pass-2 p50 |
|---|---|---|---|
| 1 | **`F_split_saved`** | 93.779 ms | 101.474 ms |
| 2 | `F_saved` | 96.766 ms | 101.694 ms |
| 3 | `F_saved` | 96.678 ms | 101.904 ms |
| 4 | `F_saved` | 94.107 ms | 101.486 ms |
| 5 | `F_saved` | 94.044 ms | 101.296 ms |

The split-reduce schedule is crowned in **1 of 5** searches, and when it is, the artifact replays at
101.474 ms — squarely inside the spread of the preset winners. On `lenet` it is never crowned; on
`cifar_stride` it is crowned in arm A only, which is not the arm that ships. So the one workload
that "picks split reductions" does not go faster for it.

## Measurement hygiene (read before comparing anything here to another report)

Two effects on this machine are large enough to manufacture findings that are not there, and both
bit during this leg:

**1. The tuned artifact is not reproducible run to run.** The *same* OLD binary, searching from a
wiped cache, produced pass-2 replays spanning 95.008–101.878 ms on `cifar_conv` — **7.2%**, with no
code change whatsoever. The cause is visible in the logs: arm B's candidates are separated by 1–3%,
which is within the search's own timing noise, so which preset gets crowned is close to a coin flip,
and different presets replay at materially different speeds. A single OLD-vs-NEW pair on this
workload is worthless; the +0.10% above is a median over 4 and 5 searches. (An early single pair
read +7.26% "regression" and an earlier one read −0.18%; both were the same coin.)

This is arguably a finding in its own right for #484 task 4's "schedule identity pins numerics"
note: on `cifar_conv` schedule identity is *also* not stable across searches, so a re-tune can move
the artifact by 7% for no reason a user can see.

**2. Absolute times drift across sessions, badly.** Against
[example-report-cuda.md](example-report-cuda.md)'s numbers from two days earlier, `ocannl/cuda/default`:

| workload | #243 (07/29) | 07/30 22:0x | 07/30 23:2x | 07/31 08:2x |
|---|---|---|---|---|
| lenet | 242.6 ms | 246.5 | 270.3 | 245.0 / 246.5 |
| cifar_stride | 70.7 ms | 81.6 | 100.0 | 83.4 / 83.3 |
| cifar_conv | 1282.4 ms | 1301.1 | 1314.2 | 1306.4 / 1310.5 |

The 23:2x column is a transient degradation under an hour of sustained load that recovered
overnight. The morning column pairs the OLD and NEW binaries, which agree to 0.09–0.58% — so none of
this is code. `lenet` and `cifar_conv` sit within 1–2% of #243; `cifar_stride` retains a genuine
+18% gap whose cause is not identified here.

**Consequence: on this box, only paired same-window A/B is meaningful.** Do not difference a number
here against a number in another report. Every cell above was bracketed by a foreign-process check;
cells that overlapped another agent's GPU job were discarded and re-run, and the retained set logs
zero interference.

## Segment attribution is unchanged

`BENCH_SEG_TIMES=1` on `bench_conv_diag`, min-of-20 per segment, at `ff54fc3f`. This measures the
*default* pipeline, which the task-3 change cannot affect (it is autotune-only), so this is a
reproduction check on #243's instrument and a statement that the target is still there:

| workload | top segment | now | #243 | | materialized top | now | #243 |
|---|---|---|---|---|---|---|---|
| lenet | seg22 `bias_conv1.grad n65.grad` | **94.5%** | 94.6% | | seg34 `kernel_conv1.grad` | 62.6% | 59.7% |
| cifar_conv | seg22 `bias_conv1.grad n65.grad` | **91.3%** | 91.4% | | seg34 `kernel_conv1.grad bias_conv1.grad` | 56.4% | 56.2% |
| cifar_stride | seg22 `bias_conv1.grad n65.grad` | 57.8% | 61.8% | | seg34 `kernel_conv1.grad bias_conv1.grad` | 67.4% | 57.9% |

`lenet` and `cifar_conv` reproduce to within 0.1 percentage point. `cifar_stride` moves more, which
tracks its being the workload with the unexplained environmental gap above.

So the segment #484 exists to attack is exactly as dominant as it was, and task 3's seeding does not
touch it.

## What this leg suggests next

1. **Find out why the conv-gradient accumulations are rejected by `split_reduce_sites`.** They pass
   both published thresholds. Until they are proposed, the family cannot pay for its search cost on
   any conv workload. This is the whole ballgame for task 3 on these benchmarks.
2. **The +5–13% search cost is currently pure loss on conv workloads.** Not urgent, but if (1)
   proves hard, consider gating the seeding on a site actually being a meaningful fraction of the
   step rather than on the structural predicate alone.
3. **`cifar_conv`'s tuned artifact is unstable at ~7%.** Independent of #484. Worth its own issue —
   it makes every A/B on that workload expensive, and it silently varies what users ship.
4. #521 remains the blocker for the *mma* family; nothing here changes that, and nothing here is
   blocked by it.

## Reproducing

```bash
BENCH_FIXTURE=fixtures/cifar_conv.safetensors BENCH_TUNE=1 \
  ../_build/default/benchmarks/runners/ocannl/bench_conv.exe --ocannl_backend=cuda \
  --ocannl_autotune_cache_dir=/tmp/fresh --ocannl_autotune_log=true 2>&1 | grep 'F_split\['
```

```bash
BENCH_FIXTURE=fixtures/lenet.safetensors BENCH_SR_SITES=1 \
  ../_build/default/benchmarks/runners/ocannl/bench_conv_diag.exe --ocannl_backend=cuda
```

Both run from `benchmarks/`. Raw data for every cell in this report — search logs, pass-2 JSON,
segment times, and the per-cell contention log — is under `benchmarks/results/gh484-cuda-raw/`
(gitignored).
