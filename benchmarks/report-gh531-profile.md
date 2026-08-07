# gh-ocannl-531: where the gpt2_mini step's ~104 ms actually goes

Measurement-only report. No code changes. This is the first-order deliverable behind
[ahrefs/ocannl#531](https://github.com/ahrefs/ocannl/issues/531): a time-attribution profile of the
tuned `gpt2_mini` inference step, kernel by kernel, so that
[#483](https://github.com/ahrefs/ocannl/issues/483) (online-softmax / fused attention) is aimed — or
re-aimed — by measurement rather than by elimination.

This commit carries the measurement: the per-kernel timeline, the structure comparison against
torch, and the roofline classification. The four deliverable bins and the verdict they support
follow in the next commit.

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
