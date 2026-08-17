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
1.30x and gh-573 a further 1.31x on the default-placement arm, 1.71x composed: kernel time goes
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
the mechanism. **Cap 4 beats the default 8 by 5.7% with non-overlapping ranges, measured in one
order-balanced block (and 1.18x against cap −1 when the two balanced blocks are chained); on the
`gpt2_mini` graph placement is identical to cap −1 only at cap 32 and above, while cap 16 already
shows one node's worth of placement difference (at the final layer norm) behind an unchanged kernel
count** — so
8 is not well-centred for this fixture (Part 4, which scopes that narrowly and records the
kernel-count inference that got this wrong first).

The four QKᵀ sites are freed here as they were on CUDA, but measured over the whole post-fission
chain rather than the fragment that keeps the name they are worth 1.80x and contribute **17%** of what
the two freed line items give, against the lm_head chain's 21.2x and 83% — about one sixth of it,
where on CUDA they were the larger half.

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
  ran cap8→cap−1, cap−1→cap8, cap8→cap−1; the cap sweep's claim-bearing pair ran cap8→cap4,
  cap4→cap8, cap8→cap4 inside one block (Part 4).
- Driver checked in as [`gh612_cells.sh`](gh612_cells.sh); every number here came out of it, and
  the Reproduction section quotes its invocations rather than restating commands.
- **Correctness gate, and what it does *not* cover.** All 26 tuned runs of this session emit
  bit-identical loss sequences (`7.09794, 7.08114, 7.12363, 7.10122, 7.11247, 7.07110, 7.10847,
  7.09132`) — one distinct sequence across all 26, spanning both placement arms, all six caps and
  all three trees. But that gate reaches **only the arm the search shipped**: `bench_gpt` keeps the
  routine `Train.tune_placements` returns and reads losses only from it, the discarded arm's
  `?report` carries timing metadata alone, and `autotune: winner replay ok` is a compile-and-
  dispatchability check (digest-guarded), not a value check — the search times candidates without
  comparing their outputs to anything. The per-kernel harness cannot fill the gap either: it runs
  kernels in isolation on synthetic buffers and never checks results.

  This matters here because arm A is profiled in **every** cell while only `master-cap8` shipped it:

  | artifact | shipped? | covered by the loss gate |
  |---|---|---|
  | `master-cap8` arm A — the denominator, both fingerprints, all Part 1 tables | yes (6/6 reps) | **yes** |
  | `base574` / `feat574` / `master-capoff` arm B | yes (3/3 reps each) | **yes** |
  | `base574` / `feat574` / `master-capoff` **arm A** — the Part 2 and Part 3 ratios | no | **no** |

  So the denominator artifact and both acceptance fingerprints rest on an output-verified routine,
  and the three comparison ratios (gh-574's 1.30x, gh-573's 1.31x, and the negative control) rest on
  artifacts that were compiled, dispatched and timed on the real lineage but never checked against a
  reference. `AGENTS.md` is explicit that for passes which change cell values, emitted-IR structure
  is not sufficient — so this is a stated limitation, not a defended position. What partially
  substitutes, and is weaker than an executed check: the autotuner's own accounting puts all four
  cells' arm A within 0.6% on FLOPs (7.81–7.86 GFLOP) at identical losses wherever the arm shipped.

  What is **not** offered as a substitute, having been considered and rejected: the zero
  signature-level difference between `feat574`'s and `master-capoff`'s arm A does not make those two
  "stand or fall together". Five of their canonicalized kernel *bodies* differ, because their crowned
  schedules differ (Part 4), and either schedule could expose a value-changing codegen bug
  independently of the other. Signature parity is evidence about placement, not about values, and
  using it in a correctness argument would be the category error this report spends Part 4 warning
  about. Closing the gap properly needs a way to ship the default-placement arm on demand; there is
  no such config today (both arms always run and the faster one ships), so it is filed rather than
  bodged.

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
reconstruction and its caveats are gh-569's and are not re-argued. Its validation is re-run:
**the sum of the 136 per-kernel medians is 18.880 / 18.876 / 18.891 ms against the step p50 of the
cell those kernels were emitted by — 18.981 ms in its search and 19.019 ms in its replay — so
agreement is 0.5–0.8%**, against gh-569's 1.6–2.0%.

The pairing is the load-bearing part of that sentence and it is worth being explicit about, because
the looser reading is available and wrong. Across the three search reps the step p50 spans
18.50–18.98 ms, and holding the r1 profile against r3's 18.50 would read a 2.1% discrepancy — but
that is not a weaker validation of the same thing, it is not a validation at all: each rep crowns a
different artifact with different tile sizes, and this profile is a profile of r1's. A per-kernel
sum may only be checked against the step time of the compile it came from. Validation got tighter
than gh-569's for a reportable reason: the lm_head, gh-569's least reproducible kernel (7.945 /
5.888 / 7.989 ms, a 36% spread), is no longer a 7.9 ms outlier.

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

| bucket | kernels | ms/step (run 1) | share (run 1 / 2 / 3) | BASE `6d14f401` | gh-569 |
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
- **The four QKᵀ sites are freed**, as on CUDA — but the size of that has to be read off the whole
  chain, not off the fragment that keeps the name. In BASE the QKᵀ is fused with its mask and row-max
  (`mask, n240_q, n242_k, n248, n257_max_vals`, 0.588–0.602 ms each) and followed by a softmax kernel;
  in FEAT the QKᵀ stands alone at 0.114 ms each, but the mask, row-max and softmax work it shed still
  runs, in three downstream kernels per site. Comparing the standalone fragment against the fused
  kernel would report 5.2x and would be meaningless. **Summing every fragment of the chain on both
  sides: 8 kernels / 3.666 ms → 16 kernels / 2.038 ms, a real 1.80x and −1.628 ms.**

  Under the same treatment the lm_head chain (`wte`/`logits`/`max_logits`/`log_probs` fragments) goes
  5 kernels / 8.165 ms → 6 kernels / 0.384 ms, **21.2x**, −7.780 ms. So the QKᵀ sites contribute
  **17% of what those two freed line items give, against the lm_head's 83%** — not merely the smaller
  half but roughly one sixth, where on CUDA they were the larger half. The two together are −9.408 ms
  against gh-574's −7.536 ms net, the difference being the FFN bucket's +2.60 ms below.
- **Signature-set diff**: 14 kernel signatures exist only in BASE, totalling **14.663 ms**; 32
  exist only in FEAT, totalling **7.428 ms**.

### And a cost, which is why the two fixes had to be measured together

The finer fission makes the **FFN bucket worse**: 35 kernels / 5.544 ms → 49 kernels / 8.139 ms,
**+2.60 ms** (medians of three harness runs). Cutting the segments apart splits the residual-stream adds into more separately
launched kernels, each of which then re-derives the running sum. That cost is what gh-573 removes
— so measuring gh-574 alone books a real +2.60 ms against it that master does not pay, and
measuring gh-573 against a pre-574 tree would miss the part of its payoff that only exists once
the fission is finer. The interaction is not hypothetical; it is 2.6 ms in both directions.

## Part 3 — the gh-573 arm: `virtualize_max_inline_fanin` −1 vs 8

A config flip on one tree. Code-borne, so each value is a fresh compile. The rows below are this
arm's **three order-balanced reps**; Part 4 pools all six reps of cap 8 and so quotes slightly
different medians (60.93 untuned, 18.67 step) from the same runs plus three more.

| | cap −1 (before) | cap 8 (after) | |
|---|---:|---:|---|
| **arm A per-kernel profile** | **24.82 ms** / 135 kernels | **18.89 ms** / 136 kernels | **1.31x** (−23.9%) |
| **untuned-default pipeline**, 3 reps | 65.63 / 65.48 / 65.59 | **61.31 / 60.94 / 60.86** | **1.076x** (−4.65 ms) |
| shipped step p50, 3 reps | 18.94 / 22.48 / 20.86 | 18.98 / 18.82 / 18.50 | 1.11x — **not claimed** |
| arm the search shipped | B, B, B | A, A, A | — |

The first two rows are solid and their ranges do not overlap: the per-kernel instrument reproduces
to 1.4% here against a 23.9% effect, and the untuned rows are 65.48–65.63 against 60.86–61.31 on a
deterministic code path.

The third row is **inside this box's noise floor and is not claimed as a result.** cap −1's own
spread is 18.7% and the ranges overlap. gh-481 measured 20–40% as the resolution floor of a
tuned-cell A/B here, and 1.11x sits inside it. (Part 4 shows this is a property of the *default*
rather than of the guard: at cap 4 the same end-to-end comparison is non-overlapping, at a chained
1.18x — see Part 4, which distinguishes that chain from the unpaired endpoint ratio.)

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

Per-kernel medians over the three harness runs, i.e. exactly what `gh612_cells.sh finger` prints:

| LN site | cap −1: params / ffn_b2 prefix / ms | cap 8: params / ffn_b2 prefix / ms |
|---|---|---|
| ln1 l0 | 8 / 0 / 0.032 | 8 / 0 / 0.031 |
| ln2 l0 | 9 / 0 / 0.037 | 9 / 0 / 0.037 |
| ln1 l1 | 11 / 1 / 0.129 | 11 / 1 / 0.124 |
| ln2 l1 | 12 / 1 / 0.287 | 13 / 1 / 0.285 |
| ln1 l2 | 14 / **2** / 0.458 | 6 / **0** / 0.029 |
| ln2 l2 | 15 / **2** / 0.542 | 7 / **0** / 0.030 |
| ln1 l3 | 17 / **3** / 0.620 | 9 / **1** / 0.033 |
| ln2 l3 | 18 / **3** / 0.701 | 10 / **1** / 0.041 |
| lnf | 20 / **4** / 0.763 | 13 / **2** / 0.178 |
| **total** | **3.566 ms** | **0.788 ms** (**4.52x**) |

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

17 exist only at cap 8, totalling **1.702 ms**, and read materialized running sums in place of the
re-summed prefix. The placement difference between the two cells is **four nodes** — `centered`,
`n446`, `n792`, `x1` — so those 16-vs-17 signatures are four materialization decisions' worth of
change plus their downstream churn. "Four" is the node-level difference, which is the closest
available proxy for guard firings and not a count of them: nothing logs provenance-41 decisions, so a
final placement difference cannot say which of the four the guard forced directly and which followed
from a reset fan-in. (An earlier revision listed `n619` among them; `n619` is materialized at cap
−1 too, being an FFN output, so it is not one of the guard's decisions.) **8.495 ms of re-summation
becomes 1.702 ms of bounded-fanin work — −6.79 ms, and every kernel involved is named.** The 119
signatures common to both cells move the other way by **+0.87 ms** (16.32 → 17.19, +5.3%), which is
crowned-tile drift rather than anything the guard does; net −5.92 ms. Reported rather than netted
silently, because it is 15% of the effect and it is the one part of this arm the signature evidence
does *not* explain.

**Where the payoff lands, and the control.** It is not confined to the layernorm bucket:

| bucket (medians of 3 harness runs) | cap −1 | cap 8 | |
|---|---:|---:|---|
| layernorm / elementwise | 3.562 | 0.786 | **−2.78** |
| FFN GEMMs | 8.120 | 5.075 | **−3.05** |
| embedding / logits | 0.661 | 0.619 | −0.04 |
| **attention** | **12.447** | **12.392** | **−0.06 (control)** |

Half the win is booked to the FFN bucket because, after gh-574's finer fission, the residual
re-summation lives in FFN-classified kernels too — the same interaction Part 2 charged +2.60 ms
for. The attention bucket is unchanged to 0.4%, which is the control: the guard has no business
there and does nothing there.

### The negative control, and it is unusually clean

Master with the guard disabled should reproduce FEAT `76f50dcd`, or the config flip is not an
isolation of gh-573 but of gh-573 plus whatever else landed in between. It does:

| | FEAT `76f50dcd` | master, cap −1 |
|---|---:|---:|
| arm A kernels | 135 | 135 |
| **kernel signatures differing** | — | **0** |
| **shared signatures with differing multiplicity** | — | **0** (the multisets agree, not just the sets) |
| arm A per-kernel profile | 24.794 ms | 24.817 ms (**0.09%**) |
| untuned-default pipeline (median) | 65.66 ms | 65.59 ms (0.1%) |
| analytic bytes, arm A winner | 528 215 820 | 528 209 196 (0.001%) |

**Zero signature differences over 135 kernels, and the kernel *multisets* agree too** — set equality
alone would permit the same signature appearing a different number of times on each side, so the
multiplicity check is part of the claim rather than a refinement of it. Everything between `76f50dcd`
and `5d0c86d8`
other than the fanin guard is worth 0.1% on this workload, so the flip isolates gh-573 cleanly.
The largest single per-kernel difference between the two is the lm_head GEMM (0.230 vs 0.350 ms) —
same signature, different crowned tile size, i.e. the search lottery, which is the same reason the
shipped-step row above is untrustworthy and the signature-level rows are not.

## Part 4 — the cap sweep: is 8 the right trade on gfx1151?

**No. Cap 4 beats the default 8 outside the noise floor, and placement is identical to cap −1 only
from cap 32 up — at cap 16 one node's worth of placement difference already appears, behind an
unchanged kernel count.** Three timed reps at caps
2, 4, 8 and −1; one each at 16 and 32, which is enough because what matters about those two is
structural (the emitted kernel multiset) rather than a timing.

This part has been wrong twice, in the same place, and both corrections are recorded rather than
quietly folded in — the sequence is the useful artifact. First revision: the "identical segmentation"
inference rested on an `F_saved` label that counts placement entries, not kernels. Second: with real
kernel counts in hand, equal counts were read as identical segmentation — but **equal fission width
can absorb a changed materialization decision**, which is exactly what cap 16 does. Only the kernel
multiset settles it. The cap-4-vs-cap-8 comparison was separately confounded with session position
and was re-run in one order-balanced block.

| cap | arm A kernels | untuned-default (median, range) | tuned step p50 (median, range) | n | ships |
|---:|---:|---|---|---:|---|
| **2** | **144** | 60.22 (60.17–60.32) | 17.33 (17.30–19.14) | 3 | A A B |
| **4** | **137** | 60.53 (60.36–60.70) | **17.47 (17.35–17.68)** | 6 | A ×6 |
| 8 (default) | **136** | 60.93 (60.86–61.31) | 18.67 (18.45–19.05) | 6 | A ×6 |
| 16 | **135** | 65.80 | 19.29 | 1 | B |
| 32 | **135** | 65.62 | 21.51 | 1 | B |
| −1 (off) | **135** | 65.59 (65.48–65.63) | 20.86 (18.94–22.48) | 3 | B B B |

**Where the guard actually goes silent — and a kernel count is not enough to tell.** Caps 16, 32 and
−1 all emit a 135-kernel arm A, and an earlier revision of this report read that as "identical
segmentation, so the guard never fires at 16 or above". That inference was wrong, and the way it was
wrong is worth keeping: **equal fission width can absorb a changed materialization decision.**
Comparing the emitted kernel *parameter-signature multisets* instead of their sizes
(`gh612_cells.sh diff`, which needs only `snap`). That is the right invariant for this question and
the reason is worth stating: a kernel's pointer parameters are exactly the materialized nodes it
touches, so a signature multiset is sensitive to placement and deliberately **insensitive to the
crowned tile**. Kernel *bodies* are not — they move with the tile as well — so a body diff cannot be
read as a placement change. `diff` prints all three levels (signatures, bodies, materialized nodes)
and the numbers below are the signature and node levels:

| comparison | exclusive signatures | **newly materialized nodes** | verdict |
|---|---|---:|---|
| cap −1 vs **cap 32** | 0 / 0, no multiplicity differences | **0** | **placement identical — the guard is silent** |
| cap −1 vs **cap 16** | 1 / 1 | **1** (`n792`) | the guard fires, once |
| cap −1 vs cap 8 | 16 / 17 | **4** (`centered`, `n446`, `n792`, `x1`) | fires |
| cap −1 vs cap 4 | 26 / 28 | **9** | fires |
| cap −1 vs cap 2 | 40 / 49 | **23** | fires |

**Read the node column, not the signature column, as the count of guard firings.** They are not the
same quantity and an earlier revision of this section conflated them: one node forced materialized
resets fan-in downstream and can change several consumers' parameter lists, so cap 8's 16/17 exclusive
signatures are the churn from **4** nodes' worth of placement change, not sixteen. The node column is
itself a
proxy — a distinct newly materialized node is what provenance 41 produces, but nothing logs those
decisions today, so exact firing counts would need a placement log that does not exist. Filed as
such rather than asserted.

At the body level cap −1 and cap 32 differ in 5 kernels each, and so do `feat574` and
`master-capoff` — in both pairs the crowned arm A sketch differs at one site (cap −1 takes
`16x16x8/2x2` where cap 32 takes `64x64x8/4x4`), which changes those kernels' schedules without
changing which nodes are materialized. So "identical" throughout this report means **identical
placement**, evidenced at the signature and node levels; it does not mean byte-identical kernels, and
no claim here needs it to.

At cap 16 the one site that changes is the **final layer norm**, which gains a materialized `n792` —
its 20-parameter signature becomes 21 while the kernel count stays at 135. Read as a proxy rather than
a firing log, that confines the cap-16 placement difference to `lnf`, the deepest site and the one
with the largest accumulated prefix, and is *consistent with* `lnf` being the only chain on this graph
whose transitive fan-in exceeds 16 — consistent with, not established by, since nothing logs the
decisions themselves. Corrected conclusions: the maximum transitive inline fan-in on this graph is
bounded into **(16, 32]** rather than between 9 and 16; the first cap at which placement changes at
all is 16, not 8; and 16-versus-8 is one node's worth of difference against **four**, not silence
against firing.

The count is still monotone in the cap (144 / 137 / 136 / 135 / 135 / 135) and still the first thing
to look at, but the rule has to be stated correctly: **a cap whose kernel count matches cap −1's may
still have fired — compare the kernel multisets before concluding silence.**

**Cap 4 beats cap 8 by 5.7%, measured in one balanced block.** The first revision reported 7.1% from
reps that had cap 8 running roughly two hours earlier in the session than cap 4 — arm confounded
with session position, exactly the trap gh-481's order rule exists for, and it inflated the effect by
about 1.4 pp. Re-run as three pairs inside one block with the order alternated (cap8→cap4,
cap4→cap8, cap8→cap4):

| | rep 4 | rep 5 | rep 6 | median | range |
|---|---:|---:|---:|---:|---|
| cap 8 | 18.45 | 19.05 | 18.51 | 18.51 | 18.45–19.05 |
| **cap 4** | 17.63 | 17.45 | 17.35 | **17.45** | **17.35–17.63** |

**Non-overlapping** (cap 4's slowest 17.63 against cap 8's fastest 18.45), all six reps shipping arm
A so no arm lottery is in play, and the deterministic untuned column agrees in the same block (cap 8
60.91–60.93 against cap 4 60.36–60.70). Pooled over all six reps per cap the ranges stay disjoint at
−6.4%. The corrected claim is **−5.7%**, from the balanced block.

Cap 2 has the best median (17.33) but its range runs to 19.14 because one rep shipped arm B, so it
overlaps cap 8 and is **not** established as better than 4. Its 144-kernel arm A is also the most
materialized in the table, which is the direction where the guard starts costing launches for
recomputation it no longer had to avoid.

**What the balanced evidence does and does not chain.** Two comparisons were run as balanced blocks:
cap 8 vs cap −1 (overlapping, 1.11x — Part 3's unclaimed row) and cap 4 vs cap 8 (disjoint,
**1.061x** = 18.51/17.45, i.e. the 5.7% time reduction; the two are different numbers and only the
ratio is a speedup).
"Cap 4 against cap −1" is a **chain of those two blocks, 1.108 × 1.061 = 1.18x**, not a balanced
measurement of its own. An earlier revision put it at 1.19x, which is the ratio of the *unpaired*
endpoint medians (20.86 / 17.45) and silently includes cap 8's shift between the two blocks (18.82 ms
in the cap −1 block against 18.51 ms in the cap 4 block) — so 1.18x is the chained value and 1.19x is
the unpaired endpoint ratio, which is not the same statement. What supports the direction
independently is the deterministic untuned instrument, where cap 4's 60.36–60.70 and cap −1's
65.48–65.63 are 8.4% apart and nowhere near touching.

**What this does not license, stated narrowly.** This is **one fixture** — `gpt2_mini` at 4 layers,
d=256, 8 heads — on one device, so what the table establishes is the cap's behaviour *for that
fixture*, and nothing broader. Depth is not even the only thing that would move it: the guard counts
distinct transitive materialized inputs per setter, which varies with architecture and graph shape at
constant depth, so "a 4-layer model" is already an over-generalization of a single measured graph and
"shallow models" more so. Both are avoided above and in the README index entry.

What that leaves is narrow and still useful: on this graph the guard is silent only from cap 32 up,
fires on one node at 16, four at 8, nine at 4 and twenty-three at 2 — and cap 4 is 5.7% faster than
the shipped default. Changing the global
default on that would be the gh-479 mistake in a new costume — the measurement that would justify
moving it is a cap sweep across several fixtures of differing depth *and* differing residual
fan-in structure, which this session did not run.

## Part 5 — what the re-established profile says to do next

The step is now attention-bound (65.6%), and inside attention one line item dominates: **the 12
q/k/v projections**. Neither fix touches them *structurally* — the same 12 kernel signatures appear in
all four cells — and they become the largest thing in the step largely by everything else shrinking.
"Invariant" would be too strong, though, and the table says so: three cells measure 6.39–6.40 ms at 16
resident blocks, while `master-cap8`'s draw crowned a different tile and measures **6.932 ms at 4
blocks**. So roughly **8% of its 36.7% share is the crowned-tile lottery, not the fixes** — the
operation is untouched, the selected schedule is not:

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
times over. Three of the four cells' draws put them at 16 blocks and 6.40 ms instead of 6.93, so that
0.53 ms and the 16→4 block change are the lottery; the ~6.4 ms floor, present in every cell, is not.

At the FFN up-projection's utilization the twelve would take about **0.72 ms**, so the replacement
step is **≈12.67 ms**; what the lottery changes is the baseline that is compared against, giving
**1.49x** against cap 8's measured 18.88 ms or **1.45x** against a lottery-adjusted 18.35 ms (18.88
less the 0.53 ms tile penalty). **1.45x** is the conservative figure and the one to carry. An earlier
revision quoted 13.2 ms / 1.43x, which subtracted the 6.40 ms floor from a total containing the
6.93 ms draw and so left the penalty in the remainder. This is an **analytic extrapolation, not a
measurement**, offered only to size the next target. Nothing here measures a
replacement, and the gh-569 lesson that HIP's block-count curve is non-monotone (peak at 128,
regressing by 1024) applies to any attempt.

## Reproduction

The driver is checked in as [`gh612_cells.sh`](gh612_cells.sh) and every number above came out of
it. That is deliberate rather than tidy: the first revision of this section restated the commands by
hand, and four of the review findings against it were transcription bugs in commands that had never
run in that form — a missing `mkdir`, a path resolved from the wrong directory, a snapshot selected
without its completeness check, and a line-oriented `grep` over multi-line kernel signatures that
matches nothing. A reproduction section that quotes invocations of the real driver cannot drift from
what produced the numbers; one that restates commands always can.

```bash
# three trees, ONE fixture file. The fixture must be the same file in every tree -- symlink it
# rather than regenerating, so the input cannot differ between arms.
cd ocannl-staging
for w in master:5d0c86d8 base:6d14f401 feat:76f50dcd; do
  git worktree add --detach ../wt-gh612-${w%%:*} ${w##*:}
  mkdir -p ../wt-gh612-${w%%:*}/benchmarks/fixtures
  ln -sf "$PWD/benchmarks/fixtures/gpt2_mini.safetensors" \
         ../wt-gh612-${w%%:*}/benchmarks/fixtures/gpt2_mini.safetensors
  (cd ../wt-gh612-${w%%:*} && dune build @check bin/ benchmarks/)
done
D=benchmarks/gh612_cells.sh   # from the tree root; results land under $OUT_ROOT (default /tmp/gh612)
```

```bash
# Part 2, the gh-574 arm: two trees, three reps, arm order ALTERNATED across reps (gh-481) --
# BASE->FEAT, FEAT->BASE, BASE->FEAT. Each `search` is a cold cell with its own fresh cache.
$D search ../wt-gh612-base base574 1;  $D search ../wt-gh612-feat feat574 1
$D search ../wt-gh612-feat feat574 2;  $D search ../wt-gh612-base base574 2
$D search ../wt-gh612-base base574 3;  $D search ../wt-gh612-feat feat574 3
```

```bash
# Part 3, the gh-573 arm: one tree, a config flip, order alternated the same way.
M=../wt-gh612-master
$D search $M master-cap8 1;                                            $D search $M master-capoff 1 --ocannl_virtualize_max_inline_fanin=-1
$D search $M master-capoff 2 --ocannl_virtualize_max_inline_fanin=-1;   $D search $M master-cap8 2
$D search $M master-cap8 3;                                            $D search $M master-capoff 3 --ocannl_virtualize_max_inline_fanin=-1
```

```bash
# the per-kernel profile of the DEFAULT-PLACEMENT arm, per cell. `snap` replays the cached winner
# with debug sources on, then selects arm A by launch-log fission width and validates the snapshot
# (balanced braces + a clean hipcc compile) before accepting it -- a content-polling watcher can
# copy a file whose `__global__` lines are all present but whose last body is torn, and the kernel
# count alone does not catch that.
for c in "../wt-gh612-base base574" "../wt-gh612-feat feat574" "$M master-cap8"; do
  set -- $c; $D snap "$1" "$2" 1; $D profile "$1" "$2" 1 3; $D finger "$2" 1
done
$D snap $M master-capoff 1 --ocannl_virtualize_max_inline_fanin=-1
$D profile $M master-capoff 1 3; $D finger master-capoff 1
```

`finger` prints the two acceptance fingerprints directly: for gh-573 the LayerNorm sites with their
`ffn_b2` prefix lengths — which must be bounded and must **reset** rather than ramp with depth — and
for gh-574 the lm_head/CE tail, which must show the GEMM alone and the row-max as a separate kernel.
It parses signatures with `re.S` because the emitted parameter lists span multiple lines.

```bash
# the cross-cell signature-set diffs, which are what pin each mechanism to named kernels rather
# than to a bucket total. These produce Part 2's 14-vs-32 counts, Part 3's 16-vs-17 and 8.495 ->
# 1.702 ms, and the negative control's zero differing signatures.
$D diff base574 1 feat574 1              # gh-574: the fused lm_head+row-max kernel disappears
$D diff master-capoff 1 master-cap8 1    # gh-573: the ffn_b2 triangle disappears
$D diff feat574 1 master-capoff 1        # the negative control: must print 0 kernels on both sides
```

```bash
# Part 4, the cap sweep. `sweep` reverses the cap order on alternate reps, so no cap sits
# permanently earlier in the session than another -- without that, cap 4 always precedes cap 8 and
# session drift is indistinguishable from the cap's effect.
$D sweep $M 3 8 4                  # the claim-bearing pair, balanced inside one block
$D sweep $M 1 2 16 32 -1           # the shape of the curve
# then snapshot each cap and compare KERNEL MULTISETS, not kernel counts. Equal counts can absorb a
# changed materialization decision -- cap 16 emits 135 kernels exactly like cap -1 yet differs by one
# signature -- so a count cannot establish that a cap did nothing. `diff` needs only `snap` here; the
# ms columns are omitted without `profile`.
for cap in 2 4 8 16 32 -1; do
  $D snap $M sweep-cap$cap 1 --ocannl_virtualize_max_inline_fanin=$cap
done
for cap in 32 16 8 4 2; do                       # 0/0 exclusive signatures => the guard was silent
  echo "== cap -1 vs cap $cap =="; $D diff sweep-cap-1 1 sweep-cap$cap 1
done
```

```bash
# the measured local roofline (compiled from the tree root; CPU quiet, since the bandwidth leg
# shares the LPDDR5X controller with it on this APU)
$D roofline $M
```
