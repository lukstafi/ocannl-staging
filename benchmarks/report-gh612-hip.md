# gh-ocannl-612 on HIP: the gh-573 and gh-574 measurement residuals, in one session

Measurement report. [ahrefs/ocannl#612](https://github.com/ahrefs/ocannl/issues/612) asks for the
unmeasured HIP legs of two merged fixes — [#573](https://github.com/ahrefs/ocannl/issues/573)'s
transitive inline-fanin guard (`virtualize_max_inline_fanin`, staging PR #353, `b049505b`) and
[#574](https://github.com/ahrefs/ocannl/issues/574)'s `arity_cuts` finer fission (staging PR #352,
`76f50dcd`) — on the box that motivated them. They are one session because they come out of one
document, [`report-gh569-hip.md`](report-gh569-hip.md), against one 46.65 ms denominator, and
because both change kernel structure, so each overstates its share when measured against the
pre-both baseline.

**Verdict: both fixes pay off on this device, and both of the issue's predicted line items are
confirmed to ~1% in absolute terms — but neither of their *shares* survives, because the
denominator moved 1.44x underneath them. Against the re-established denominator gh-574 is worth
1.30x and gh-573 a further 1.32x on the default-placement arm, 1.71x composed: kernel time goes
32.33 → 18.88 ms, and the five kernels that were 47.2% of the gh-569 step are now 6.8%.**

The two predictions in detail, because the distinction between "the absolute cost was right" and
"the share was right" is the whole reason this needed one session:

- gh-574's tied lm_head was predicted at **7.945 ms / 17.0%**. On the re-established tree it
  measures **8.036 ms — but 24.9%**, having grown as a share while standing still in absolute
  terms. Cutting it apart, together with the four QKᵀ sites, is worth **7.54 ms net (23.3%)**.
- gh-573's layernorm/elementwise bucket was predicted at **4.33 ms / 9.3%**. It measures **4.319 ms
  — but 13.4%**, and the guard is worth **5.93 ms (23.9%)** — *more* than the bucket it was filed
  against, because after gh-574's finer fission the re-summation it removes also lives in
  FFN-classified kernels.

So both payoffs are larger than predicted, and for gh-573 larger in absolute terms too.

One prediction needs a caveat and one needs a correction. The caveat: at its shipped default cap of
8, gh-573's payoff on the *end-to-end* step is inside this box's search-noise floor, for an
identifiable reason given in Part 3 — without the guard the search ships materialize-all instead,
which is the crude form of the same transform. The correction: that is a fact about the default, not
the mechanism. **At cap 4 the same guard is worth 17.48 ms against cap −1's 20.86 with
non-overlapping ranges (1.19x), and any cap ≥ 16 never fires at all on this model** — so 8 is not a
well-centred default here (Part 4).

The four QKᵀ sites are freed here as they were on CUDA, but they are the **smaller** half of
gh-574's win on this device (2.39 ms against the lm_head's 8.04) — the reverse of CUDA.

## Provenance

- Box: AMD Ryzen AI Max+ 395 ("Strix Halo", Zen 5, 16C/32T) with a **Radeon 8060S iGPU — gfx1151,
  RDNA3.5, 40 CUs reported as 20 workgroup processors, 2900 MHz**. **WSL2**, kernel
  6.18.33.2-microsoft-standard-WSL2. ROCm / HIP **7.14.60850**. Otherwise idle. This is the same
  box at the same ROCm version as [`report-gh569-hip.md`](report-gh569-hip.md), so its numbers are
  directly comparable rather than merely analogous.
- The local roofline was **re-measured** rather than carried over
  ([`roofline_hip.cpp`](roofline_hip.cpp)), and it reproduces gh-569's to within run-to-run
  variation: rocBLAS `sgemm` 4096³ fp32 **2527.2 GFLOP/s** (gh-569: 2757), dependency-free FMA issue
  peak **24172.6 GFLOP/s** (24210), device-to-device copy **210.6 GB/s** (210.1). The freshly
  measured `sgemm` figure is the denominator used below. Note that rocBLAS itself reaches only
  10.5% of the FMA issue peak on this part, so "% of `sgemm` peak" compares two implementations and
  is not a distance from a hardware ceiling — both denominators are quoted where it matters.
- Workload: `benchmarks/fixtures/gpt2_mini.safetensors` — 4 layers, d=256, 8 heads, seq 128,
  vocab 1024, batch 8, forward-only (`mode: infer`), 1024 tokens/step. Precision **f32**
  (`default_prec=single`); `mma_seeded = 0` in every arm of every cell, the RDNA3.5 f32 null that
  gh-569 and [`report-gh528-hip.md`](report-gh528-hip.md) establish. The *same* fixture file is
  symlinked into all three trees, so the input is byte-identical across arms.
- Trees, all built and run in this one session:

  | tree | commit | gh-574 `arity_cuts` | gh-573 fanin guard |
  |---|---|---|---|
  | BASE | `6d14f401` (staging PR #351) | absent | absent |
  | FEAT | `76f50dcd` (staging PR #352) | present | absent |
  | master | `5d0c86d8` (staging `origin/master`) | present | present (default 8) |

- Cache discipline: one **cold** search into a fresh `--ocannl_autotune_cache_dir` per cell *per
  rep*, never shared ([#568](https://github.com/ahrefs/ocannl/issues/568): the cache key omits the
  Numerics policy; [`report-gh481-cuda.md`](report-gh481-cuda.md): a warm cache makes an A/B
  vacuous by replaying the other arm's winner). All work under `taskset -c 0-15`.
- Arm order balanced across reps, per gh-481: gh-574 ran BASE→FEAT, FEAT→BASE, BASE→FEAT; gh-573
  ran cap8→cap−1, cap−1→cap8, cap8→cap−1.
- **Correctness gate: all 12 tuned runs emit bit-identical loss sequences**
  (`7.09794, 7.08114, 7.12363, 7.10122, 7.11247, 7.07110, 7.10847, 7.09132`). Every arm computes
  the same thing; nothing below is a comparison across differing numerics.

### Three instruments, three noise floors

The single most important methodological fact here is that the honest instrument is *not* the
shipped step time. Reported side by side throughout:

| instrument | what it is | reproducibility measured here |
|---|---|---|
| **arm A per-kernel profile** | [`gpt2_kernel_harness.py`](gpt2_kernel_harness.py) over the emitted default-placement arm; deterministic given a source | **0.08–1.6%** over 3 harness runs |
| **untuned-default pipeline** | the deterministic non-searched lowering, printed by every search (`autotune: untuned-default pipeline`) | **0.2–0.7%** over 3 reps |
| shipped tuned step p50 | end-to-end, but carries gh-481's family lottery | **2.6–18.7%** over 3 reps |

Arm A (default placements) is profiled in every cell — including the cells where the search
shipped arm B — because it is the like-for-like structural comparand and because it is the arm
gh-569's tables are of. Which arm actually *shipped* is reported separately and never silently
mixed in.

`rocprofv3` still collects nothing on this box (no `/dev/kfd` under WSL2); the per-kernel
reconstruction and its caveats are gh-569's and are not re-argued. Its validation is re-run and is
tighter than before: **the sum of the 136 per-kernel medians is 18.880 / 18.876 / 18.891 ms against
a measured step p50 of 18.50–18.98 ms — agreement to 0.5%**, against gh-569's 1.6–2.0%. It got
tighter for a reportable reason: the lm_head, gh-569's least reproducible kernel (7.945 / 5.888 /
7.989 ms, a 36% spread), is no longer a 7.9 ms outlier.

## Part 0 — the old denominator is gone, and the two line items survived it

BASE `6d14f401` still fissions the default-placement arm into **117** kernels and materialize-all
into **130** — *exactly* gh-569's 117/130 split. The segmentation is the same artifact. The time
is not:

| arm A, 117 kernels | gh-569 (`6c94d7ca`) | BASE (`6d14f401`) | |
|---|---:|---:|---|
| FFN GEMM1 + gelu ×4 | 14.08 ms (30.2%) | **0.949 ms (2.9%)** | **14.8x faster** |
| tied lm_head (+ row-max, fused) | 7.945 ms (17.0%) | **8.036 ms (24.9%)** | +1.1% |
| layernorm / elementwise bucket, 18 kernels | 4.33 ms (9.3%) | **4.319 ms (13.4%)** | −0.3% |
| **step (kernel time)** | **46.65 ms** | **32.33 ms** | 1.44x faster |

The FFN up-projections moved from 8 resident blocks to 128 and got 14.8x faster — that is
gh-569's *own* fix, which landed in between, and it accounts for essentially all of the step's
improvement (13.13 ms of the 14.32 ms). Crediting any of it to the two fixes under test would have
been the error the issue was filed to prevent.

What did **not** move is the pair under test: the lm_head to within 1.1% and the
layernorm/elementwise bucket to within 0.3%, both on unchanged kernel counts (1 and 18). So the two
predicted *absolute* line items are reproduced almost exactly on a tree whose denominator shrank by
1.44x — and their *shares* both grew, the lm_head's from 17.0% to 24.9%. A share is a share of a
placement and a tree; these two absolute costs turned out to be the invariant.

## Part 1 — the re-established denominator

Current master, default cap, arm A: **136 kernels, 18.88 ms**. Three harness runs, three
repetitions of the search:

| bucket | kernels | ms/step | share (r1 / r2 / r3) | BASE `6d14f401` | gh-569 |
|---|---:|---:|---|---:|---:|
| **attention** | 57 | 12.392 | **65.6% / 65.6% / 65.6%** | 43.9% | 32.6% |
| **FFN GEMMs** | 53 | 5.055 | **26.8% / 26.9% / 27.1%** | 17.2% | 40.6% |
| **layernorm / elementwise** | 9 | 0.817 | **4.3% / 4.2% / 4.0%** | 13.4% | 9.3% |
| **embedding / logits** | 17 | 0.614 | **3.3% / 3.3% / 3.3%** | 25.5% | 17.5% |
| total | 136 | 18.880 | 100% | 100% | 100% |

Classification is [`gpt2_bucket.py`](gpt2_bucket.py)'s, unchanged; 83.1% of kernel time is
directly seeded from a named model weight, the rest by propagation. Shares are stable to 0.3 pp —
the embedding/logits instability gh-569 had to report rather than smooth is gone with the lm_head
outlier.

**The five-kernel table, re-rendered.** gh-569's step was five kernels at 47.2%:

| | gh-569 | BASE `6d14f401` | master |
|---|---:|---:|---:|
| FFN GEMM1 + gelu ×4 | 14.08 (30.2%) | 0.949 (2.9%) | 0.961 (5.1%) |
| tied lm_head | 7.945 (17.0%) | 8.036 (24.9%) | 0.320 (1.7%) |
| **the five** | **22.02 (47.2%)** | **8.985 (27.8%)** | **1.281 (6.8%)** |

The concentration is gone. The largest single kernel in the current step is **0.707 ms, 3.7%**,
and no kernel exceeds 4%.

**Launch geometry.** gh-569 found 96 of 117 kernels at `(8,1,1)×(128,1,1)` — 1024 threads, 8 of
20 workgroup processors busy. That signature is *still* the bulk of the kernel count, but no
longer the bulk of the time:

| resident blocks | kernels | ms | share |
|---:|---:|---:|---:|
| 4 | 12 | 6.93 | 36.7% |
| 8 | 106 | 6.67 | 35.3% |
| 16 | 8 | 2.27 | 12.0% |
| 128 | 9 | 3.01 | 15.9% |
| 1 | 1 | 0.02 | 0.1% |

Note the 12 kernels at **4** blocks taking 36.7% of the step. That is the new headline, and Part 5
is about it.

**One instrument retired.** gh-569's Part 3 was a decline census: 25 gh-521 companion-coverage
declines at `8x128x1024`, read from `schedule_log_declines`. That census is now **empty — zero
companion-coverage decline lines in any cell** — and the reason is not that the rule stopped
firing. Since [gh-577](https://github.com/ahrefs/ocannl/issues/577) the coverage verdict is a
construction-time refutation in the matmul family tree (`matmul_coverage_witness`), so a refuted
family is never seeded and therefore never declines. The census is no longer the instrument for
this question; the emitted source and the launch geometry are.

## Part 2 — the gh-574 arm: BASE `6d14f401` vs FEAT `76f50dcd`

Not config-gated (`arity_cuts` is passed unconditionally in the GPU seeding path), so this arm is
two built trees.

| | BASE | FEAT | |
|---|---:|---:|---|
| **arm A per-kernel profile** | **32.33 ms** / 117 kernels | **24.79 ms** / 135 kernels | **1.30x** |
| shipped step p50, 3 reps | 25.04 / 25.72 / 26.24 | 18.87 / 19.81 / 19.85 | **1.30x** (medians) |
| untuned-default pipeline, 3 reps | 65.47 / 65.67 / 65.81 | 65.66 / 65.63 / 65.77 | 1.00x |

The shipped-step ranges do **not** overlap (BASE min 25.04 > FEAT max 19.85), so this one survives
the gh-481 objection without needing the per-kernel instrument — and the two instruments agree on
1.30x to two digits.

The untuned row is an internal control that came out exactly right: `arity_cuts` is a
candidate-*generation* mode, never the default pipeline, so it must not move the untuned number,
and it does not (0.1%).

### Acceptance fingerprints

- **A `fine`-flagged candidate is crowned**, in both arms, in every rep of FEAT and master; and in
  **no** rep of BASE. Arm A ships `F_saved[fine 76 segs]` (FEAT) / `[fine 77 segs]` (master)
  against BASE's `F_saved[58 segs]`. (The `N segs` in an `F_saved` label counts the saved
  per-segment placement entries, **not** kernels — the emitted arm A holds 76/135 and 58/117
  respectively. Kernel counts here always come from the launch log and the emitted source, never
  from that label; mistaking one for the other is easy and would misreport every count below.)
- **`finer_fission true`** in both schedule-cache entries, so replay re-segments identically.
- **The cross-entropy segment is cut apart.** The fused kernel taking `logits, max_logits,
  n794_layer_norm, wte` — the lm_head GEMM *and* its row-max in one kernel, at 8 resident blocks —
  costs **8.036 ms, 24.9% of BASE's step, the largest single kernel on this device**, and is
  **absent from FEAT**. In its place: the GEMM alone (`logits, n794_layer_norm, wte`) at **0.230
  ms**, and the row-max (`logits, max_logits`) as its own downstream kernel. On master the GEMM
  lands at `grid=(32,4,1)×block=(16,16,1)` — **128 blocks, 256 threads**, LDS-staged tiles
  (`tile_wte[512]`, `tile__n84_tile_layer_norm[512]`) and a 2×2 register accumulator — against
  gh-569's `(8,1,1)×(128,1,1)`, 8 blocks and no staging. CUDA's counterpart went from 8 to 512
  blocks. Here the three cells' draws land it at 32, 128 and 512 blocks (0.230 / 0.320 / 0.350 ms),
  and it is worth resisting the temptation to read that as gh-569's non-monotone block-count curve
  reproducing: those are three *different tile shapes*, not one kernel rechunked, so their reuse
  differs too. gh-569's curve was measured by holding the kernel fixed and varying only the chunk
  count; nothing here is that experiment, and the ordering above is not evidence about it either way.
- **The four QKᵀ sites are freed**, as on CUDA. In BASE they are fused with mask and row-max
  (`mask, n240_q, n242_k, n248, n257_max_vals`) at 0.588–0.602 ms each, **2.391 ms / 7.4%**; in
  FEAT the QKᵀ stands alone at **0.457 ms across four kernels, 5.2x cheaper**. But on this device
  they are the *smaller* half of the win — 2.39 ms against the lm_head's 8.04 — where on CUDA they
  were the larger half. Worth naming as a genuine cross-vendor difference rather than glossed.
- **Signature-set diff**: 14 kernel signatures exist only in BASE, totalling **14.663 ms**; 32
  exist only in FEAT, totalling **7.428 ms**.

### And a cost, which is why the two fixes had to be measured together

The finer fission makes the **FFN bucket worse**: 35 kernels / 5.543 ms → 49 kernels / 8.151 ms,
**+2.61 ms**. Cutting the segments apart splits the residual-stream adds into more separately
launched kernels, each of which then re-derives the running sum. That cost is what gh-573 removes
— so measuring gh-574 alone books a real +2.61 ms against it that master does not pay, and
measuring gh-573 against a pre-574 tree would miss the part of its payoff that only exists once
the fission is finer. The interaction is not hypothetical; it is 2.6 ms in both directions.

## Part 3 — the gh-573 arm: `virtualize_max_inline_fanin` −1 vs 8

A config flip on one tree. Code-borne, so each value is a fresh compile.

| | cap −1 (before) | cap 8 (after) | |
|---|---:|---:|---|
| **arm A per-kernel profile** | **24.82 ms** / 135 kernels | **18.89 ms** / 136 kernels | **1.32x** (−23.9%) |
| **untuned-default pipeline**, 3 reps | 65.63 / 65.48 / 65.59 | **61.31 / 60.94 / 60.86** | **1.076x** (−4.65 ms) |
| shipped step p50, 3 reps | 18.94 / 22.48 / 20.86 | 18.98 / 18.82 / 18.50 | 1.11x — **not claimed** |
| arm the search shipped | B, B, B | A, A, A | — |

The first two rows are solid and their ranges do not overlap: the per-kernel instrument reproduces
to 1.4% here against a 23.9% effect, and the untuned rows are 65.48–65.63 against 60.86–61.31 on a
deterministic code path.

The third row is **inside this box's noise floor and is not claimed as a result.** cap −1's own
spread is 18.7% and the ranges overlap. gh-481 measured 20–40% as the resolution floor of a
tuned-cell A/B here, and 1.11x sits inside it. (Part 4 shows this is a property of the *default*
rather than of the guard: at cap 4 the same end-to-end comparison is non-overlapping at 1.19x.)

**Why the end-to-end row understates the fix, mechanically.** Without the guard the search ships
**arm B (materialize-all) in 3 of 3 reps**; with it, **arm A in 3 of 3**. Materialize-all is the
crude form of exactly what the guard does selectively — it materializes the running sum by
materializing everything — so it recovers most of the same benefit and the end-to-end gap closes.
The guard's end-to-end value is therefore not "the step got faster" but "the default-placement arm
now wins", i.e. the benefit arrives without materialize-all's footprint. The analytic traffic the
autotuner's own calibration reports, at constant FLOPs (7.81–7.86 GFLOP across all four cells),
agrees: arm A's crowned candidate goes **528.2 MB → 472.1 MB**, −10.6%.

### Acceptance fingerprints

**The triangle, and its truncation.** gh-569 showed each LayerNorm site re-deriving the residual
by re-summing every prior contribution. At cap −1 that is still exactly what happens, and the cost
grows monotonically with depth:

| LN site | cap −1: params / ffn_b2 prefix / ms | cap 8: params / ffn_b2 prefix / ms |
|---|---|---|
| ln1 l0 | 8 / 0 / 0.032 | 8 / 0 / 0.031 |
| ln2 l0 | 9 / 0 / 0.036 | 9 / 0 / 0.037 |
| ln1 l1 | 11 / 1 / 0.129 | 11 / 1 / 0.129 |
| ln2 l1 | 12 / 1 / 0.287 | 13 / 1 / 0.299 |
| ln1 l2 | 14 / **2** / 0.458 | 6 / **0** / 0.031 |
| ln2 l2 | 15 / **2** / 0.539 | 7 / **0** / 0.033 |
| ln1 l3 | 17 / **3** / 0.620 | 9 / **1** / 0.033 |
| ln2 l3 | 18 / **3** / 0.695 | 10 / **1** / 0.046 |
| lnf | 20 / **4** / 0.758 | 13 / **2** / 0.180 |
| **total** | **3.552 ms** | **0.817 ms** (**4.34x**) |

The prefix runs `0,0,1,1,2,2,3,3,4` at cap −1 and `0,0,1,1,`**`0,0,1,1,2`** at cap 8 — a saw-tooth
that **resets at layer 2**, where `ln1 l2` reads a materialized `centered` node instead of
re-summing from `wpe`. Parameter count peaks at 13 instead of 20, per-site time stops growing with
depth (max 0.18 ms against 0.76), and the site with the accumulated prefix is bounded at the cap's
worth. That is the fingerprint the issue asked for, read off the kernel signatures.

**Signature-set diff, which pins the mechanism rather than inferring it.** 16 kernel signatures
exist only at cap −1, totalling **8.495 ms**, and every large one carries the full triangle:

```
l0_ffn_b2, l1_ffn_b2, l2_ffn_b2, l3_ffn_b2,
n275_multi_head_attention, n341, n414_multi_head_attention, n480, n553_multi_head_attention,
..., n9, wpe
```

17 exist only at cap 8, totalling **1.702 ms**, and read the materialized running sums
(`centered`, `x1`, `n619`, `n792`) in place of the re-summed prefix. **8.495 ms of re-summation
becomes 1.702 ms of bounded-fanin work — −6.79 ms, and every kernel involved is named.** The 119
signatures common to both cells move the other way by **+0.87 ms** (16.32 → 17.19, +5.3%), which is
crowned-tile drift rather than anything the guard does; net −5.92 ms. Reported rather than netted
silently, because it is 15% of the effect and it is the one part of this arm the signature evidence
does *not* explain.

**Where the payoff lands, and the control.** It is not confined to the layernorm bucket:

| bucket | cap −1 | cap 8 | |
|---|---:|---:|---|
| layernorm / elementwise | 3.552 | 0.817 | **−2.74** |
| FFN GEMMs | 8.200 | 5.055 | **−3.15** |
| embedding / logits | 0.661 | 0.614 | −0.05 |
| **attention** | **12.444** | **12.392** | **−0.05 (control)** |

Half the win is booked to the FFN bucket because, after gh-574's finer fission, the residual
re-summation lives in FFN-classified kernels too — the same interaction Part 2 charged +2.61 ms
for. The attention bucket is unchanged to 0.4%, which is the control: the guard has no business
there and does nothing there.

### The negative control, and it is unusually clean

Master with the guard disabled should reproduce FEAT `76f50dcd`, or the config flip is not an
isolation of gh-573 but of gh-573 plus whatever else landed in between. It does:

| | FEAT `76f50dcd` | master, cap −1 |
|---|---:|---:|
| arm A kernels | 135 | 135 |
| **kernel signatures differing** | — | **0** |
| arm A per-kernel profile | 24.794 ms | 24.817 ms (**0.09%**) |
| untuned-default pipeline (median) | 65.66 ms | 65.59 ms (0.1%) |
| analytic bytes, arm A winner | 528 215 820 | 528 209 196 (0.001%) |

**Zero signature differences over 135 kernels.** Everything between `76f50dcd` and `5d0c86d8`
other than the fanin guard is worth 0.1% on this workload, so the flip isolates gh-573 cleanly.
The largest single per-kernel difference between the two is the lm_head GEMM (0.230 vs 0.350 ms) —
same signature, different crowned tile size, i.e. the search lottery, which is the same reason the
shipped-step row above is untrustworthy and the signature-level rows are not.

## Part 4 — the cap sweep: is 8 the right trade on gfx1151?

**No. On this workload 4 is measurably better than 8, and any cap ≥ 16 is indistinguishable from
disabling the guard.** Three reps at caps 2, 4, 8 and −1; one at 16 and 32, which need no more
because they are provably inert (see below).

| cap | tuned step p50, reps | median | untuned-default, reps | median | arm shipped | arm A segs |
|---:|---|---:|---|---:|---|---:|
| **2** | 17.30 / 17.33 / 19.14 | 17.33 | 60.32 / 60.17 / 60.22 | **60.22** | A A B | 85 |
| **4** | **17.48 / 17.68 / 17.43** | **17.48** | 60.53 / 60.60 / 60.53 | **60.53** | A A A | 78 |
| 8 (default) | 18.98 / 18.82 / 18.50 | 18.82 | 61.31 / 60.94 / 60.86 | 60.94 | A A A | 77 |
| 16 | 19.29 | 19.29 | 65.80 | 65.80 | B | 76 |
| 32 | 21.51 | 21.51 | 65.62 | 65.62 | B | 76 |
| −1 (off) | 18.94 / 22.48 / 20.86 | 20.86 | 65.63 / 65.48 / 65.59 | 65.59 | B B B | 76 |

**The cliff between 8 and 16 is the sharpest thing in the table, and it is not a performance
result — it is the guard going silent.** At caps 16, 32 and −1 arm A fissions to **76 segments in
all three**, and the untuned-default times agree to 0.3% (65.48–65.80 ms). The cap is not being
traded off at those settings; it is never reached. On this 4-layer model the residual stream's
maximum transitive inline fan-in therefore lies **between 9 and 16**, and the shipped default of 8
sits one step inside the range where the guard does anything at all.

**Cap 4 beats cap 8 outside the noise floor.** 17.43–17.68 ms against 18.50–18.98 — **the ranges do
not overlap**, over three reps each with balanced arm order, and all six reps ship arm A so no
arm-lottery is in play. That is −7.1% on the median. Cap 2 has the best median (17.33) but its range
runs to 19.14 because one rep shipped arm B, so it overlaps cap 8 and is *not* better than 4 on this
evidence; 4 is the reliable setting, and it is the one that fires exactly once more than 8 (78
segments against 77).

**This also settles the question Part 3 had to leave open.** At the shipped default the end-to-end
comparison against cap −1 is inside the noise floor. At cap 4 it is not: **17.48 ms against 20.86,
ranges 17.43–17.68 against 18.94–22.48 — non-overlapping, 1.19x.** So the fanin guard's payoff
*is* real end to end on this device; what is marginal is the particular default, not the mechanism.

**What this does not license.** One workload, one depth. The cap's bite is a function of how many
contributors a residual stream accumulates, i.e. of model depth, so a 4-layer fixture is the least
favourable case for a large cap and the most favourable for a small one — a deeper model would move
the cliff up and could easily make 8 fire where 4 over-materializes. Changing the global default on
this evidence would be the gh-479 mistake in a new costume. What the table does support is that **8
is not a well-centred default on gfx1151 for shallow models**, and that a cap-vs-depth sweep on a
deeper fixture is the measurement that would justify moving it.

## Part 5 — what the re-established profile says to do next

The step is now attention-bound (65.6%), and inside attention one line item dominates: **the 12
q/k/v projections**. They are *invariant* across every arm of this session — untouched by both
fixes — and become the largest thing in the step purely by everything else shrinking:

| | BASE | FEAT | master cap −1 | master cap 8 |
|---|---:|---:|---:|---:|
| q/k/v, 12 kernels | 6.401 ms | 6.390 | 6.400 | 6.932 |
| share of step | 19.8% | 25.8% | 25.8% | **36.7%** |
| resident blocks | 16 | 16 | 16 | 4 |

Each is 134.22 MFLOP (`2·8·8·128·32·256`), so at 0.533 ms that is **252 GFLOP/s — 10.0% of this
device's measured 2527 GFLOP/s rocBLAS `sgemm` peak, 1.0% of its FMA issue peak** — against the FFN
up-projection's 2235 GFLOP/s in the same step (**88.4%** of `sgemm`, 9.2% of issue peak), which is
to say the up-projection now essentially matches rocBLAS while these do not. The gap between the two
is the finding; neither number is a distance from a hardware ceiling. The mechanism is visible in
the emitted source, and it is
gh-569's rank-3+ observation one axis deeper: the site's output is rank 4 —
(batch, head, seq, head_dim) — and under the presets' `max_chain=2` only `seq` and `head_dim` get
geometry, so **batch and head stay serial loops inside the kernel**:

```c
for (int i1610 = 0; i1610 <= 7; ++i1610) {        // batch  -- serial
  for (int i1612 = 0; i1612 <= 7; ++i1612) {      // head   -- serial
    const int i3205 = (int)blockIdx.y;            // seq
    ...  const int i3209 = (int)blockIdx.x;       // head_dim
```

`grid=(1,4,1)×(16,16,1)` in master's draw is **4 blocks** — 4 of 20 workgroup processors, 12
times over. Three of the four cells' draws put them at 16 blocks and 6.40 ms instead of 6.93, so the
tile choice is inside the lottery; the 6.4 ms floor is not. If these reached the FFN
up-projection's utilization the step would fall to roughly 12.7 ms (~1.49x) — an **analytic
extrapolation, not a measurement**, offered only to size the next target. Nothing here measures a
replacement, and the gh-569 lesson that HIP's block-count curve is non-monotone (peak at 128,
regressing by 1024) applies to any attempt.

## Reproduction

```bash
# three trees, one fixture. The fixture must be the SAME file in all three (byte-identical input).
cd ocannl-staging
for w in master:5d0c86d8 base:6d14f401 feat:76f50dcd; do
  git worktree add --detach ../wt-gh612-${w%%:*} ${w##*:}
  mkdir -p ../wt-gh612-${w%%:*}/benchmarks/fixtures
  ln -sf "$PWD/benchmarks/fixtures/gpt2_mini.safetensors" \
         ../wt-gh612-${w%%:*}/benchmarks/fixtures/gpt2_mini.safetensors
  (cd ../wt-gh612-${w%%:*} && dune build @check bin/ benchmarks/)
done
```

```bash
# one cell = one COLD search. The fresh cache dir is load-bearing and must never be shared
# across reps or arms (ahrefs/ocannl#568, and gh-481's vacuous-A/B trap).
# $EXTRA is empty for the default cap, --ocannl_virtualize_max_inline_fanin=N otherwise.
cd <tree>/benchmarks && rm -rf /tmp/cell/cache
BENCH_FIXTURE=fixtures/gpt2_mini.safetensors BENCH_TUNE=1 taskset -c 0-15 \
  ../_build/default/benchmarks/runners/ocannl/bench_gpt.exe --ocannl_backend=hip \
  --ocannl_autotune_cache_dir=/tmp/cell/cache \
  --ocannl_autotune_log=true --ocannl_schedule_log_declines=true $EXTRA \
  > /tmp/cell/search.out 2> /tmp/cell/search.err
grep -E 'untuned-default pipeline|winner replay ok|tune_placements: winner' /tmp/cell/search.err
grep -o 'finer_fission [a-z]*' /tmp/cell/cache/*.sexp        # gh-574: must be true
```

```bash
# the emitted source + the launch geometry, from a warm replay of exactly that artifact.
# Arm A compiles first and arm B overwrites it, so snapshot by polling on content; the watcher
# can catch a partially written file, so pick the snapshot by KERNEL COUNT.
rm -rf build_files /tmp/cell/armsnap && mkdir -p /tmp/cell/armsnap
F=build_files/bench_gpt/cross_entropy_loss_fwd__seg.hip
( while :; do if [ -f "$F" ]; then h=$(md5sum "$F" | cut -d' ' -f1)
    [ -f "/tmp/cell/armsnap/$h.hip" ] || cp "$F" "/tmp/cell/armsnap/$h.hip"; fi
    sleep 0.02; done ) & W=$!
BENCH_FIXTURE=fixtures/gpt2_mini.safetensors BENCH_TUNE=1 taskset -c 0-15 \
  ../_build/default/benchmarks/runners/ocannl/bench_gpt.exe --ocannl_backend=hip \
  --ocannl_autotune_cache_dir=/tmp/cell/cache \
  --ocannl_output_debug_files_in_build_directory=true \
  --ocannl_schedule_log_launches=true $EXTRA 2> /tmp/cell/launches.err
kill $W
# arm A is the FIRST fissioned compile in the launch log (the total=1 whole-routine probe aside)
N=$(grep -o 'cross_entropy_loss_fwd seg 0/[0-9]*' /tmp/cell/launches.err \
      | awk -F/ '$2>1{print $2; exit}')
for f in /tmp/cell/armsnap/*.hip; do
  [ "$(grep -c '__global__' "$f")" = "$N" ] && A=$f; done; echo "arm A = $A ($N kernels)"
```

```bash
# per-kernel times and the bucket table. Three harness runs; the stderr sum-vs-step line is the
# validation the report quotes.
python3 gpt2_kernel_harness.py --source "$A" --launches /tmp/cell/launches.err \
        --out /tmp/cell/harness.hip
hipcc --offload-arch=gfx1151 -O2 -o /tmp/cell/harness /tmp/cell/harness.hip
for i in 1 2 3; do taskset -c 0-15 /tmp/cell/harness > /tmp/cell/kernels-$i.csv \
                     2> /tmp/cell/kernels-$i.err; tail -1 /tmp/cell/kernels-$i.err; done
python3 gpt2_bucket.py --source "$A" --stats /tmp/cell/kernels-1.csv --steps 1 --dump
```

```bash
# the fingerprints, read off the emitted source rather than inferred.
# gh-573: the LayerNorm triangle -- prefix length must be bounded and must RESET, not ramp.
grep -oE '__global__ void \w+__seg[0-9]+\([^)]*\)' "$A" \
  | grep -E 'gamma_|beta_' | grep -oE 'l[0-9]_ffn_b2' | sort | uniq -c
# gh-574: the CE segment cut apart -- the GEMM alone, the row-max its own kernel.
grep -oE '__global__ void \w+__seg[0-9]+\([^)]*\)' "$A" | grep -E 'wte|max_logits'
# and the geometry the lm_head GEMM actually launched at
grep -E "cross_entropy_loss_fwd seg [0-9]+/$N" /tmp/cell/launches.err \
  | sed -E 's/.*(seg [0-9]+).*(grid=\[[^]]*\]).*(block=\[[^]]*\]).*/\1 \2 \3/'
```

```bash
# the cap sweep. Three reps per cap, and read the UNTUNED column first: it is deterministic, and a
# cap whose untuned time and arm A segment count both match cap -1's is not being traded off, it is
# never firing (that is what happens at 16 and above here).
for cap in 2 4 8 16 32 -1; do for rep in 1 2 3; do
  # ... the cold-search cell above, with --ocannl_virtualize_max_inline_fanin=$cap
  grep -m1 'untuned-default pipeline' /tmp/cell/search.err
  grep -m1 'winner replay ok' /tmp/cell/search.err     # arm A segment count
done; done
```

```bash
# the measured local roofline (CPU quiet: the bandwidth leg shares the LPDDR5X controller with it)
hipcc --offload-arch=gfx1151 -O3 -o /tmp/roofline benchmarks/roofline_hip.cpp \
      -I/opt/rocm/include -L/opt/rocm/lib -lrocblas && taskset -c 0-15 /tmp/roofline
```
