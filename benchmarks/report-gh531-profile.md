# gh-ocannl-531: where the gpt2_mini step's ~104 ms actually goes

Measurement-only report. No code changes. This is the first-order deliverable behind
[ahrefs/ocannl#531](https://github.com/ahrefs/ocannl/issues/531): a time-attribution profile of the
tuned `gpt2_mini` inference step, kernel by kernel, so that
[#483](https://github.com/ahrefs/ocannl/issues/483) (online-softmax / fused attention) is aimed — or
re-aimed — by measurement rather than by elimination.

**Headline: elimination pointed at the wrong place.** Attention-adjacent `seq²` traffic — #483's
entire prize — is **5.6 ms of the 103.8 ms step (5.4%)**. Five kernels that contain no attention at
all are **72.3 ms (70.2%)**, and they are byte-for-byte identical in the tuned and untuned
artifacts: tuning cannot touch them, because `gh-ocannl-521`'s companion-coverage rule declines
every tensorized *and* scalar seed at their output geometry. Nothing in the step is at roofline;
the closest kernel to any roofline leg reaches 11.9% of memory bandwidth.

## Provenance

- Box: NVIDIA GeForce RTX 5070 Ti **Laptop** GPU (sm_120, **46 SMs**, 12,226 MiB), driver 610.62,
  CUDA toolkit 13.3, **WSL2** (kernel 6.18.33.2-microsoft-standard-WSL2); Intel Core Ultra 9 275HX.
  Otherwise idle throughout.
- Tree: worktree of staging master `70001062`, which contains the gh-550 arm-containment fix
  `a7672848`.
- Workload: `benchmarks/fixtures/gpt2_mini.safetensors` — 4 layers, d=256, 8 heads, seq 128,
  vocab 1024, batch 8, forward-only (`mode: infer`), 1024 tokens/step.
- Profiler: Nsight Systems 2026.1.3, `--trace=cuda --sample=none --cpuctxsw=none
  --cuda-graph-trace=node`. Profiling overhead is not measurable here: the step p50 is 103.33 ms
  under nsys against 103.83 ms unprofiled.
- **Cache origin: cold, this session.** This worktree had no `autotune_cache` at all, so nothing
  from the gh-550 acceptance runs was reused. One from-scratch search, **864 s**, populated
  `benchmarks/autotune_cache`; every profiled run below is a **pass-2 warm replay** of that
  artifact, per the two-pass protocol in `benchmarks/README.md`.
- Primary configuration: **tuned tf32** (`--ocannl_tf32_matmuls=true`), `--ocannl_backend=cuda`
  pinned explicitly. Exit codes were checked unpiped throughout. No goldens promoted.

The search reproduces the ledger's cold-search band (arm A 103.1–105.9 ms across its five
replicates). Arm A ships:

```
tune_placements: arm A best: 106.0098 ms
  (F_sketch[mma-gpu 16x32x0, mma-gpu 32x32x0, mma-gpu 16x32x0] [tensorized])
tune_placements: arm B (materialize-all) FAILED, it loses the comparison
  (its pre-failure best of 224.9654 ms is not shippable)
tune_placements: winner: arm A
```

Arm B still exhausts the card and dies at its winner replay; with `a7672848` that is now a losing
arm rather than a dead run, which is why this profile could be taken at all. Replay step p50 across
three profiled repetitions: **103.33 / 103.74 / 103.84 ms**.

## Reading the right artifact

Both placement arms compile a routine named `cross_entropy_loss_fwd`, so they write the same debug
filenames and **the `.cu` left on disk after a tuned run is arm B — the discarded arm**. Arm B has
130 kernels and zero `mma.sync`; arm A has **117 kernels and 20 `mma.sync`**. Every kernel-level
claim below is read off a mid-run snapshot of arm A, cross-checked against the calibration TSV
(which reports 117 kernels for arm A candidates and 130 for arm B) and against nsys, which observes
exactly `seg0..seg116`, 53 instances each — the 8 parity + 5 warmup + 20 synced + 20 queued steps.

## Part 1 — per-kernel timeline

**117 kernels per step, 102.95 ms of GPU kernel time against a 103.33 ms wall step.**

| bucket | kernels | ms/step | share (r1 / r2 / r3) |
|---|---:|---:|---|
| **FFN GEMMs** | 16 | 59.5 | 57.6% / 57.7% / 57.7% |
| **attention** (QKᵀ, softmax chain, scores·V, out-proj, q/k/v) | 56 | 26.3 | 25.5% / 25.5% / 25.4% |
| **embedding / logits** | 9 | 15.1 | 14.6% / 14.6% / 14.6% |
| **layernorm / elementwise** | 35 | 2.3 | 2.2% / 2.2% / 2.2% |
| other | 1 | ~0 | 0.0% |

Shares are stable to ±0.1 pp across three repetitions; shares, not absolute ns, are the
claim-bearing quantity.

### The step is nine kernels

| seg | ms/step | share | what it computes | tensorized? |
|---|---:|---:|---|---|
| 25, 51, 77, 103 | 14.31 each, **57.5 total** | 55.9% | FFN GEMM 1 (`d→d_ff`) + gelu epilogue, one per layer | no |
| 111 | **14.76** | 14.3% | tied lm_head GEMM (`d→vocab`) + logits epilogue | no |
| 19, 45, 71, 97 | 3.14 each, **12.58 total** | 12.2% | attention out-projection (`w_o`) | no |

Those nine kernels are **84.8 ms = 82.4% of the step**. The remaining 108 kernels share 18.1 ms.

### Inter-kernel gaps: #488's coverage is complete

| | ms/step | of kernel time |
|---|---:|---:|
| within-step inter-kernel idle (6,171 gaps, median **160 ns**, p90 288 ns) | 0.046 | **0.044%** |
| step-boundary idle (29 gaps >50 µs, median 396 µs — the parity steps' host loss reads) | 0.513 | 0.50% |

Graph capture launches the step as one replay and the timeline confirms it: the GPU is **99.6%
busy** inside a step. There is no launch-overhead residue left to collect, and no finding against
#488's coverage. This also retires the "58 segments each paying launch overhead" hypothesis in a
second, independent way from the census that first retired it.

## Part 2 — structure comparison vs torch

`pytorch cuda eager`, same fixture, same runner, profiled identically.

| | OCANNL tuned tf32 | torch cuda eager | ratio |
|---|---:|---:|---:|
| kernel launches / step | **117** | **209** | 0.56× |
| GPU kernel time / step | 102.95 ms | 1.545 ms | **66.6×** |
| wall step p50 | 103.83 ms | 2.90 ms | 35.8× |
| GPU busy fraction of the step | 99.6% | 53% | — |

Two things follow, and both cut against the structural reading:

1. **OCANNL launches fewer kernels than torch, not more.** torch eager spends 47% of its step idle
   on launch overhead; OCANNL spends 0.4%. Measured on GPU work alone the gap is **66.6×**, *larger*
   than the 35.8× wall-clock gap the ledger quotes. The gap is per-kernel quality, not kernel count.
2. **The pass structure is not 39× apart either.** Kernel-level passes over the large intermediates,
   counted from the emitted source:

| buffer class | size | OCANNL passes/step | traffic (OCANNL) |
|---|---:|---:|---:|
| `seq²` scores / exp / softmax | 4 MiB × 4 layers | 36 (16 r, 20 w) | 144 MiB |
| `seq×d` activations | 1 MiB | 208 (136 r, 72 w) | 208 MiB |
| `seq×d_ff` FFN hidden | 4 MiB | 20 (8 r, 12 w) | 80 MiB |

For `seq²` the torch side is countable exactly, because its attention is four explicit ops
(`einsum` → `where` → `softmax` → `einsum`): three writes and three reads, **6 passes per layer,
24 per step**, and nsys confirms the softmax is a single fused kernel per layer
(`cunn_SpatialSoftMaxForward`, 4 launches/step, 0.203 ms). torch's fused step is the existence proof
of the minimal `seq²` structure, and **OCANNL is 1.5× off it** — nine passes per layer against six.
A 1.5× structural excess on a bucket worth 5.4% of the step cannot be the 39×.

I did not count torch's `seq×d` passes: its eager layernorm and gelu decompose into many separate
elementwise tensors, and nsys measures 150 elementwise + 20 reduction launches per step, so torch is
not pass-minimal on that axis either and the comparison would not be the clean one the `seq²` row
is. The OCANNL column stands on its own.

The `seq×d` row is the one genuine structural outlier, and it has a specific cause: the residual
stream is Virtual, so every consumer re-derives it by re-summing all prior contributions. Read
counts fall off linearly with depth — layer 0's attention output is read by **17** kernels, then
15, 13, 11, 9, 7, 5, 3 — the triangular signature of a quadratic-in-depth recomputation. It is real,
but at 575 GB/s those 119 MiB cost ~0.2 ms, consistent with the 2.3 ms measured for the whole
layernorm/elementwise bucket. It is a correctness-of-shape observation, not the time sink.

## Part 3 — model-vs-measured roofline

Envelope measured on this card rather than assumed: **17.2 TFLOP/s** fp32 and **29.2 TFLOP/s** tf32
(4096³ sgemm), **574.7 GB/s** (256 MiB device-to-device copy). For reference `cuda_backend.ml`'s
advisory constants — the ones the gh-491 model actually scores with — are 15 TFLOP/s and 450 GB/s,
i.e. conservative by 15–25%, so using the measured numbers is the harder test.

FLOPs and compulsory bytes are analytic from the model geometry (exact for the GEMMs); times are
from nsys.

| kernel | n | ms/step | share | GFLOP/s | % peak | GB/s | % BW | mma | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|:--:|---|
| FFN GEMM1 + gelu | 4 | 57.51 | 55.9% | 38.1 | 0.22% | 1.0 | 0.18% | no | **well under** |
| lm_head GEMM | 1 | 14.76 | 14.3% | 36.4 | 0.21% | 0.4 | 0.07% | no | **well under** |
| attn out-projection | 4 | 12.58 | 12.2% | 42.7 | 0.25% | 0.8 | 0.13% | no | **well under** |
| q / k / v projections | 12 | 7.62 | 7.4% | 211.5 | 0.72% | 3.7 | 0.65% | yes | well under |
| QKᵀ scores + row max | 4 | 2.15 | 2.1% | 124.6 | 0.72% | 19.6 | 3.41% | no | well under |
| scores·V | 4 | 2.08 | 2.0% | 129.1 | 0.44% | 12.1 | 2.11% | yes | well under |
| FFN GEMM2 | 4 | 1.33 | 1.3% | 1609.4 | 5.51% | 18.9 | 3.28% | yes | well under |
| softmax exp + normalize | 4 | 0.99 | 1.0% | 34.1 | 0.20% | 68.2 | **11.86%** | no | well under |
| final LN + logits epilogue | 1 | 0.28 | 0.3% | 11.1 | 0.06% | 11.1 | 1.93% | no | well under |

**At-roofline: 0.000 ms. Well under roofline: 99.30 ms — 96.5% of the step, which is everything the
table covers.** The single closest kernel to any roofline leg is the softmax normalize at 11.9% of
bandwidth; every GEMM is between 0.2% and 5.5% of its compute leg.

The whole-step version of the same statement, from the gh-491 calibration TSV emitted by the search
itself: **model 1.110 ms, measured 106.01 ms — 95× above the envelope**, at 117 kernels, 7.72 GFLOP
and 441 MB of compulsory traffic. (The model's flop count is independently correct: hand-counting
the architecture gives 7.5 GFLOP.)

So the traffic does not have to shrink. There is no kernel whose bytes are the binding constraint.

### Why the dominant kernels are slow, and why tuning cannot help them

Tuning is a clean natural experiment: it moved the step 188.3 → 102.9 ms, and per-kernel it did
exactly one thing.

| seg | what | untuned | tuned tf32 | speedup |
|---|---|---:|---:|---:|
| 25 | FFN GEMM1 + gelu | 14.31 | 14.31 | **1.00×** |
| 111 | lm_head GEMM | 14.82 | 14.76 | **1.00×** |
| 19 | attn out-projection | 3.16 | 3.14 | **1.00×** |
| 11 | QKᵀ + row max | 0.54 | 0.53 | 1.01× |
| 7 | q projection | 3.24 | 0.63 | **5.11×** |
| 27 | FFN GEMM2 | 12.68 | 0.63 | **20.0×** |

**87.3% of the tuned step sits in kernels tuning did not improve at all** (<1.15×). The 20
`mma.sync` sites in arm A are exactly the kernels that did move: q/k/v projections, scores·V, and
FFN GEMM2. Nothing else tensorized.

The controlled comparison is **FFN GEMM1 against FFN GEMM2**, in the same step, on the same box:
both are 536.9 MFLOP (`2·1024·256·1024`), differing only in which axis is the output minor axis.
GEMM2 (output `8×128×256`) takes **0.63 ms**; GEMM1 (output `8×128×1024`) takes **14.31 ms** —
**23× more for identical arithmetic.**

And the difference is *not* tensorization. In the all-scalar default-flags cell, where nothing
tensorizes anywhere, GEMM2 still runs at 0.83 ms against GEMM1's 14.43 — **17×**. What separates
them is that a better schedule is *reachable* for the `8×128×256` output geometry and blocked for
`8×128×1024`, whether or not tensor cores are on the menu.

Read off arm A's emitted source, GEMM1 in the *shipping tuned artifact* is byte-for-byte the naive
scalar form:

```c
for (int i1705 = 0; i1705 <= 1023; ++i1705)          // output, serial
  for (int i1706 = 0; i1706 <= 255; ++i1706)         // reduction, serial
    n311[(...)*1024 + i1705] =
      fmaf(l0_ffn_w1[...], n309_layer_norm[...], n311[(...)*1024 + i1705]);
```

— a **global-memory read-modify-write on every one of 262,144 inner iterations**, with no register
accumulation, launched at `grid=(8,1,1) × block=(128,1,1)` = **1024 threads on 46 SMs**: 8 SMs hold
any work at all, at roughly 1.4% occupancy. That is the whole explanation of 0.22% of peak.

And the reason it stays that way is in the search log. Arm A's decline census:

| decline | seeds |
|---|---:|
| gh-521 companion coverage, **`8x128x1024` geometry** | 10 tensorized + **10 scalar** |
| gh-521 companion coverage, `8x128x8x128` geometry (rank-4 sites) | 5 tensorized + 5 scalar |
| gh-521 companion coverage, cross-nest race bail (100 companion nests) | 5 + 4 |
| `Hardware_limits`: `seg25` stages 4,194,304–4,202,496 B of workgroup-shared tiles vs 49,152 | 5 |
| `Fuse_epilogue`: write-path loop not Serial/Grid | 2 |

`8x128x1024` is exactly the output geometry of FFN GEMM1 and of the lm_head; `8x128x8x128` is the
out-projection and QKᵀ. **The rule declines the scalar seeds in the same proportion as the
tensorized ones** (10 and 10), so this is a site-geometry limitation, not a tensorization one —
reading only its mma-labelled half would misdiagnose it as an mma problem, as the ledger already
warned for the bf16 leg.

## Deliverable bins

### Bin 1 — attention-adjacent `seq²` traffic: **5.6 ms of 103.8 (5.4%)**

This is **#483's prize, stated explicitly**. The 20 kernels per step that touch a `seq²` buffer
(scores, exp, softmax output) total 5.599 / 5.589 / 5.617 ms across the three repetitions. Perfect
fusion — an online-softmax attention costing *zero* — would take the step from 103.8 ms to ~98.2 ms,
a **1.06× end-to-end win**, and would leave the gap to torch at 33.9× instead of 35.8×.

### Bin 2 — unfused elementwise / normalization passes: **~3.3 ms (3.2%)**

Cheaper remedies than #483, and correspondingly small. The named chains:

- **LayerNorm splits across 2–3 kernels per site** (mean → variance/`sqrt_std_dev` → normalize; e.g.
  `seg3`/`seg5`, `seg21`/`seg23`): 35 kernels, 2.3 ms total. Nine sites × 4 layers + `lnf`.
- **The softmax chain is 4 kernels per layer** where torch uses 1 (`seg10`/`11` scores+max,
  `seg12`/`13` exp+normalize): 0.99 ms, and it is the *most* roofline-efficient thing in the step at
  11.9% of bandwidth — i.e. fusing it further buys little.
- **The Virtual residual stream is re-summed at every consumer** (119 MiB/step, quadratic in depth).
  Structurally the worst offender in the profile and worth a small issue on its own account, but
  ~0.2 ms of time.

### Bin 3 — under-roofline kernels: **99.3 ms (96.5%) — this is the step**

Ordinary tuning residue, except that it is not residue: it is the entire workload, and the existing
tuning machinery is *structurally blocked* from reaching it, not merely failing to find a win. The
concentration is extreme:

- 4 × FFN GEMM1 + gelu + 1 × lm_head = **72.3 ms (70.2%)**, blocked at `8x128x1024`.
- 4 × attention out-projection = **12.6 ms (12.2%)**, blocked at `8x128x8x128`.

**Size of the prize.** FFN GEMM2 is the same-shape, same-step, same-box control: it does 536.9 MFLOP
in 0.63 ms. If GEMM1 and the lm_head reached the same per-FLOP rate, those five kernels would fall
from 72.3 ms to ~3.1 ms and the step would land near **34 ms** — a **3.0× end-to-end win**, against
bin 1's 1.06×. That estimate is a measured within-step control, not a model extrapolation.

### Bin 4 — surprises

**(a) Material, flagged prominently: the schedule cache key does not include the `tf32_matmuls`
numerics policy.** Running the same workload with default flags against a cache populated by a tf32
search produces `cache hit: F_saved[58 segs] ... [tensorized]` and replays a tensorized schedule
under a policy that cannot support it. Measured cost:

| configuration | step p50 |
|---|---:|
| tuned tf32 (cache built with tf32 on) | 103.83 ms |
| untuned default | 188.13 ms |
| **default flags replaying the tf32-built cache** | **1111.93 ms** |

**10.7× slower than the tuned step it was derived from, and 5.9× slower than not tuning at all.**
`docs/schedules_and_autotuning.md` states the invariant as "schedule identity pins numerics"; this
is the converse hazard — the numerics policy changing under a pinned schedule. It is silent (a
normal-looking cache hit), and any tf32-vs-default A/B sharing a cache directory will read it as a
catastrophic regression of the default arm. Reported, not fixed; the secondary measurement below
therefore uses an isolated `autotune_cache_dir`.

**(b) The shared-tile overshoot is not on the CE head.** The gh-528 ledger update attributed the
multi-MB shared-tile overshoot to "`cross_entropy_loss_fwd__seg25` … the CE head". In *this* compile
`seg25` is the **layer-0 FFN up-projection** (`n311`, `n339_gelu`), and the overshoot reproduces
exactly there (4,194,304–4,202,496 B against a 49,152 B limit). The routine is named after its root
tensor — `Train.forward batch_loss` on a `cross_entropy_loss` root — so `cross_entropy_loss_fwd` is
the *routine* name and carries no information about what any `__segN` computes. Segment numbering is
also compile-specific, so I cannot say the bf16 replicate the ledger quoted numbered its segments
the same way; what I can say is that the inference "seg25, therefore the CE head" does not follow,
and that here the overshoot lands on the single most expensive kernel class in the step. That makes
it more interesting than as a CE-head curiosity, not less.

**(c) The declines are not an mma story.** 10 scalar seeds and 10 tensorized seeds die on the same
`8x128x1024` rule. A fix there would help the untuned default too.

**(d) The ledger's "tensorization adds ~7%" is confirmed per-kernel, and localized.** Step totals:
untuned 188.26 → tuned all-scalar 110.53 → tuned tf32 102.95, so tensorization is worth **7.4%** on
top of scalar tuning — the ledger's figure, now with a per-kernel account of *where*. Both kinds of
tuning move the same three groups and only those, tf32 simply moving them further:

| group | untuned → tuned-scalar | → tuned-tf32 |
|---|---:|---:|
| q/k/v projections ×12 | 3.1× | 5.1× |
| FFN GEMM2 ×4 | 15.2× | 38.0× |
| scores·V ×4 | 2.9× | 3.2× |

Every other group is flat in both. So the search's reach — not its choice of instruction — is what
is binding.

## Secondary — does the profile's shape depend on tensorization?

**No.** The default-flags tuned cell (tf32 policy off, so `mma_format_tiles` has no f32 entry and
*zero* tensorized candidates are proposed) was searched cold into an **isolated
`autotune_cache_dir`** — required, see bin 4(a) — crowning `F_sketch[gpu 16x16x8/2x2, gpu
32x32x16/2x2, gpu 16x16x8/4x4]`, all scalar, at 113.61 ms; replay p50 **110.96 ms**.

| bucket | tuned tf32 | tuned default (all-scalar) |
|---|---:|---:|
| FFN GEMMs | 57.6% | 55.7% |
| attention | 25.5% | 28.6% |
| embedding / logits | 14.6% | 13.7% |
| layernorm / elementwise | 2.2% | 2.1% |
| **`seq²`-touching kernels** | **5.4%** | **5.3%** |

The bucket shape is the same to ~3 pp, and bin 1's number is the same to 0.1 pp. Per group, against
the untuned baseline, the invariance is sharper still:

| group | untuned | tuned default | tuned tf32 |
|---|---:|---:|---:|
| **FFN GEMM1 + gelu ×4** | 57.66 | 57.73 | 57.51 |
| **lm_head** | 14.82 | 14.82 | 14.76 |
| **attn out-projection ×4** | 12.64 | 12.62 | 12.58 |
| QKᵀ ×4 | 2.16 | 2.14 | 2.15 |
| softmax ×4 | 1.00 | 1.00 | 0.99 |
| q/k/v projections ×12 | 38.69 | 12.65 | 7.62 |
| FFN GEMM2 ×4 | 50.68 | 3.34 | 1.33 |
| scores·V ×4 | 6.69 | 2.30 | 2.08 |
| **step total** | **188.26** | **110.53** | **102.95** |

**The five dominant kernels cost 72.48 / 72.55 / 72.27 ms — invariant to 0.4% across untuned,
tuned-scalar and tuned-tensorized.** Everything either kind of tuning achieves happens in the other
four rows. Their share of the step therefore *rises* as tuning succeeds elsewhere: 38.5% → 65.6% →
70.2%.

This is the strongest form of the report's claim. The 72 ms core is not a tuning failure that a
better search would find; it is unreachable by the search space as currently gated, at every
precision policy tested.

## Verdict

**Bin 3 dominates, and #483 should be re-scoped or deprioritized: its prize is 5.6 ms of the
103.8 ms step (5.4%), against 72.3 ms (70.2%) sitting in five non-attention kernels that no
configuration of the current tuner can reach.**

The supporting facts, in the order that matters:

1. `seq²`-touching kernels are 5.4% of the step, stable across three repetitions and unchanged
   (5.3%) with tensorization off. A perfect fused attention buys **1.06×**.
2. Five kernels — four FFN up-projections and the tied lm_head — are 70.2%, and cost
   72.48 / 72.55 / 72.27 ms in the untuned, tuned-scalar and tuned-tensorized artifacts. **Invariant
   to 0.4%.**
3. Nothing is at roofline: 0.000 ms at-roofline, 96.5% of the step classified well under, the best
   kernel reaching 11.9% of a bandwidth leg. So "the traffic must shrink" — the premise fusion work
   rests on — is false here.
4. The blocker is named and in the log: gh-521's companion-coverage rule declines 10 tensorized
   **and 10 scalar** seeds at the `8x128x1024` output geometry, which is exactly FFN GEMM1 and the
   lm_head; and 5+5 at `8x128x8x128`, which is the out-projection and QKᵀ.
5. The prize on the other side is measured, not modelled: FFN GEMM2 is the same 536.9 MFLOP GEMM in
   the same step and runs 17–23× faster. At that rate the step lands near **34 ms — 3.0×**.

#483 is not wrong about attention being unfused; it is wrong about attention being where the time
is. The recommendation is to re-aim at the `8x128x1024` / `8x128x8x128` companion-coverage declines
(gh-521), and to revisit #483 afterwards, when `seq²` traffic would be a materially larger share of
a ~34 ms step.

Two caveats on the 3.0× estimate: it assumes the declined sites can reach GEMM2's *per-FLOP* rate,
which the geometry difference may not permit in full, and GEMM2's rate is itself only 5.5% of the
compute roofline — so 34 ms is a milestone, not a floor. Neither caveat affects the ranking of
bin 3 over bin 1, which rests on the 5.4%/70.2% split alone.

## Appendix — memory telemetry (ride-along for #550)

Not in scope for this report; recorded because the profiling runs produced it. The cold search
peaked at **11,911 MiB of 12,227 (97.4%)** and arm B's winner replay died of
`CUDA_ERROR_OUT_OF_MEMORY` exactly as in the gh-528 addendum — the accumulation half of #550 is
unchanged. The pass-2 replay processes that produced every number above need well under 1 GB.

## Reproduction

```bash
cd benchmarks
# pass 1: cold search, populates autotune_cache and emits the calibration TSV
BENCH_FIXTURE=fixtures/gpt2_mini.safetensors BENCH_TUNE=1 OCANNL_AUTOTUNE_LOG=true \
  ../_build/default/benchmarks/runners/ocannl/bench_gpt.exe \
  --ocannl_backend=cuda --ocannl_tf32_matmuls=true \
  --ocannl_autotune_calibration_file=/tmp/calib.tsv
```

```bash
cd benchmarks
# pass 2: profile the warm replay
nsys profile --trace=cuda --sample=none --cpuctxsw=none --cuda-graph-trace=node -o /tmp/tf32 \
  env BENCH_FIXTURE=fixtures/gpt2_mini.safetensors BENCH_TUNE=1 \
  ../_build/default/benchmarks/runners/ocannl/bench_gpt.exe \
  --ocannl_backend=cuda --ocannl_tf32_matmuls=true
nsys stats --report cuda_gpu_kern_sum --format csv /tmp/tf32.nsys-rep
```

To read the *shipping* arm's source, snapshot `build_files/bench_gpt/cross_entropy_loss_fwd__seg.cu`
while the run compiles — arm A is written first and arm B overwrites it. Arm A is the 117-kernel,
20-`mma.sync` version.
