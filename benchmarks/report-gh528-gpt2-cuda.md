# gh-ocannl-528 on CUDA: `ldmatrix` reaches a shipping artifact, and the gpt2_mini ledger moves 1.8x

Measurement-only report. Two questions, both re-opened by gh-ocannl-528's landing (PR #280, batched /
rank-3 mma seeding):

1. **gh-ocannl-481 closed "neutral, because the staged family doesn't ship there."** That rationale is
   now false on `gpt2_mini`: staged tensorized segments ship, and at bf16 the crowned artifact's
   staged tile is b128-swizzled and fed by `ldmatrix`.
2. **gh-ocannl-531's ledger** ("tuning does nothing: 450 s of search moves 189.124 -> 188.905 ms")
   needs a refresh. Tuning now moves it to ~105 ms.

Hardware: NVIDIA GeForce RTX 5070 Ti Laptop GPU (sm_120), driver 610.43.02 / CUDA UMD 13.3, toolkit
13.3, **under WSL2**; Intel Core Ultra 9 275HX. Same box as
[report-gh481-cuda.md](report-gh481-cuda.md), [report-gh537-cuda.md](report-gh537-cuda.md) and
[report-gh484-cuda.md](report-gh484-cuda.md). Tree: staging master `03512535` (contains PR #280,
PR #285, PR #287).

Every cell is one process, `--ocannl_backend=cuda` pinned explicitly (never config lookup), a
**fresh `autotune_cache_dir` per replicate**, on an otherwise idle box.

## 1. Does `ldmatrix` stop being inert on gpt2_mini?

**Yes, it renders in a shipping artifact — and no, the tuner does not reliably choose it.** Both
halves matter, and gh-481's closing sentence is wrong in one direction while its measurement
conclusion survives in the other.

### (a) The staged family now exists here, and reaches the crown

gh-481 closed with "neutral on mlp shapes, because the staged family doesn't ship there". On
`gpt2_mini` after gh-528 the staged family is seeded, timed, and crowned. The crowned arm-A label
across every bf16 search of this leg:

| run | arm A best | seg 1 | seg 2 | seg 3 |
|---|---|---|---|---|
| recon (debug emission on) | 108.186 | `mma-gpu 32x32x0` | `mma-gpu 16x32x0` | **`mma-gpu 16x32x32 swz-b128`** |
| r1 | 108.360 | `mma-gpu 32x32x0` | `mma-gpu 16x32x0` | `gpu 32x32x16/2x2` |
| r2 | 107.470 | `mma-gpu 32x32x0` | `mma-gpu 16x32x0` | `mma-gpu 16x32x0` |
| r3 | 109.001 | `mma-gpu 32x32x0` | `mma-gpu 16x32x0` | `gpu 16x16x8/4x4` |
| r4 | 108.795 | `mma-gpu 32x32x0` | `mma-gpu 16x32x0` | `gpu 16x16x8/2x2` |
| r5 | 107.483 | `mma-gpu 32x32x0` | `mma-gpu 16x32x0` | `mma-gpu 16x32x0` |

Two structural facts, and they pull opposite ways:

- **Segments 1 and 2 are tensorized in 6 of 6 searches, identically.** That is gh-528's win, and it
  is not a lottery: two of the three sketch-scheduled sites take an unstaged `Tile_mma` every time.
- **Segment 3 is a coin flip among four families** spanning staged-swizzled, unstaged-tensorized and
  two plain scalar sketches — while the whole-arm time moves only 1.4% (107.47–109.00). The lottery
  gh-546 documented is real here, but it is *localized to one segment*, not diffuse.

So the staged+swizzled form wins segment 3 in **1 of 6** bf16 searches, and the one that hit is also
the only run with debug emission enabled (its arm-A time sits mid-range, so nothing looks distorted,
but the configuration was not identical — recorded, not leaned on).

### (c) The emitted `.cu` of the shipping artifact does contain `ldmatrix`

Read off the emission, not the label (gh-538 contract item 4). Arm A's winner replay from the run
that crowned the twin:

```
ldmatrix = 12   __shared__ = 8   mma.sync = 12
/* tile_mma 16x32x128 (mma-bf16) */
/* tile_mma 16x32x32  (mma-bf16) ldmatrix a,b */    <- both operands via ldmatrix
/* tile_mma 32x32x256 (mma-bf16) */
```

`mma_scalar_fallbacks = 0`. The ` ldmatrix a,b` tag is the backend's own marker for which operands
came in through the warp-cooperative load, so this is the `ldmatrix` path rendering in an artifact
that ships — the thing gh-481 could not demonstrate on `mlp_wide`.

### (b) The paired twin-vs-plain comparison: still neutral, now at n=45

The tuner times each swizzled twin adjacent to the plain sibling it was minted from, in the same
process and search, with identical tile sizes — the only instrument on this box that resolves below
the ~20-40% tuned-cell spread. Pairing each twin with the **nearest preceding** timing of its
sibling (the same label is timed several times per search, at materially different values, so
"nearest preceding" is required for the pairing to mean anything):

| staged seed | n | median | range |
|---|---|---|---|
| `mma-gpu 16x32x32` | 15 | +0.367% | −0.34% … +1.38% |
| `mma-gpu 32x32x16` | 15 | −1.359% | −8.92% … +1.47% |
| `mma-gpu 32x32x32` | 15 | +0.152% | −0.27% … +2.12% |
| **all, arm A (ships)** | **45** | **+0.096%** | −8.92% … +2.12% |

Mean −0.390%, stdev 1.83%, **twin faster in 20 of 45** — a coin flip centred on zero. This
reproduces gh-481's +0.20% (n=48) on a workload ~100x larger. The `32x32x16` median is the only
cell that looks non-null, and its 10-point range says noise rather than effect.

Arm B's pairs (n=30, median −0.104%, stdev 14.1%) are reported separately and not pooled: its
candidates run ~50x slower and carry ±30-40% outliers.

**Verdict for gh-481: the emission is correct, reaches a shipping artifact on this workload, and is
still worth 0% on the clock.**

## 2. The gh-531 ledger, refreshed

`gpt2_mini` inference, batch 32 x seq 32 = 1024 tokens/step, p50 of 20 synced steps.

| cell | this leg (per replicate) | median | gh-531 (bd075cd5) |
|---|---|---|---|
| ocannl cuda **tuned**, tf32 †  | 104.28, 103.39, 103.15, 104.91, 102.49 | **103.39** | 188.905 |
| ocannl cuda **tuned**, bf16 | 110.53, 110.01, 113.85, 110.64, 109.25 | 110.53 | n/a (no bf16 leg then) |
| ocannl cuda **tuned**, plain f32 (tf32 policy OFF) | 110.86, 111.10 | 110.98 | n/a |
| ocannl cuda default, f32 | 188.01, 188.91 | 188.46 | 189.124 |
| ocannl cuda default, bf16 | 212.39, 212.91 | 212.65 | n/a |
| ocannl cuda materialized, f32 | 223.19, 222.77 | 222.98 | 225.194 |
| ocannl cuda materialized, bf16 | 241.93, 242.20 | 242.06 | n/a |
| pytorch cuda eager | 2.604, 2.650, 2.857 | 2.650 | 2.604 |
| tinygrad CUDA jit | 5.355, 5.367, 5.388 | 5.367 | 5.382 |

† **The tuned-tf32 row is measured from a warm-cache replay of the crowned schedule, not at the end
of a cold search** — every one of the five cold tf32 searches OOMed before reaching the measurement
phase (§3). A replay compiles only the crowned schedule, so it never runs the arm-B search that
exhausts the card. The substitution is calibrated on bf16, where both are available: replay medians
108.46 vs cold 110.53, i.e. **the replay reads ~1.9% faster** because it is not measuring under
11.9 GB of accumulated allocation. Applied to tf32 that would put a cold-equivalent near 105 ms; the
replay number is quoted as measured and the correction is not applied.

The untuned, materialized and both reference cells reproduce gh-531 to within 1.7%, so the
comparison is like-for-like. **What changed is tuning**: gh-531 recorded 450 s of search moving
189.124 -> 188.905 ms (0.1%); it now reaches 103.39 ms, a **1.82x** win, and the gap to torch cuda
eager narrows from gh-531's **72.5x to 39.0x**.

Those untuned cells are also this leg's **negative control**. They run no search, and their
replicate spread is 0.1–0.5% — that is the clock-and-box floor. Every tuned-cell spread beyond it is
search lottery, not measurement noise, which is what licenses the 1.82x claim while forbidding any
small whole-cell claim.

### Most of the win is not tensor cores

The default-flags variant is the control that separates them, and it is decisive. With the tf32
policy off, `mma_input_formats_of_prec` resolves f32 storage to `Mma_f32`, and CUDA's
`mma_format_tiles` has **no genuine (f32, f32, f32) entry** — only the f16, bf16, fp8 and tf32 rows.
So:

| tuned variant | mma seeded | mma timed | crowned | vs untuned 188.46 |
|---|---|---|---|---|
| plain f32 (tf32 policy off) | **0** | **0** | `F_sketch[gpu 32x32x16/2x2, x3]` — all scalar | **1.70x** |
| tf32 | 65 | 25 | staged tensorized in 4 of 5 | 1.82x |

**A tuned schedule that proposes zero tensorized candidates already delivers 1.70x.** Tensorization
adds ~7% on top of that. The headline improvement over gh-531 is a *scheduling and fission* result;
the tensor-core work is a real but secondary increment. `--ocannl_tf32_matmuls=true` is also load-
bearing: without it, tensorization on f32 storage is unreachable by construction, not merely unlucky.

Two further ledger facts:

- **bf16 is slower than f32 here, untuned and tuned alike** (212.65 vs 188.46 default; 110.53 vs
  103.39 tuned). The reduced-precision leg buys nothing on this workload at this size.
- **Materialization is still a loss** (222.98 vs 188.46), as gh-531 found. Arm B loses the placement
  A/B in all 7 searches that reached a winner line (A 107.47–112.98 vs B 200.91–221.89); the five
  tf32 searches OOMed before that line, but their arm A best of 102–108 ms against arm B's ~220 ms
  untuned-default reference leaves no doubt about the direction.

### Seeded / timed / declined, per family

One bf16 replicate, arm A. The counters close exactly: **65 mma candidates seeded, 40 declined,
25 timed** (`mma_seeded`/`mma_timed` from `Autotune.report`), and the 40 decompose as:

| cause | mma seeds | of which swz twins | scalar seeds |
|---|---|---|---|
| companion coverage (gh-521), `8x128x1024` — the vocab-1024 lm_head site | 16 | 6 | 10 |
| companion coverage, `8x128x8x128` — the two-axis / rank-4 site | 8 | 3 | 5 |
| companion coverage, cross-nest race bail | 8 | 3 | 4 |
| `Hardware_limits`: shared-tile overflow, CE head `seg25` | 8 | 3 | 0 |
| `Fuse_epilogue`: write-path loop not Serial/Grid | 0 | 0 | 5 |

All are **clean classified declines**, never fatal. The companion-coverage precondition accounts for
51 of 64 total arm-A declines and hits **scalar and tensorized seeds in proportion** — it is a
site-geometry limitation, not a tensorization-specific one, which is worth stating because it is
easy to read the mma-labelled half alone as an mma problem.

The one genuinely mma-exclusive decline is the shared-tile overflow, and only on epilogue-fused
variants: `cross_entropy_loss_fwd__seg25` asks for **2,097,152–4,202,496 bytes** of workgroup-shared
tiles against the device's **49,152** — a 43–86x overshoot, on the CE head that gh-531 already
singled out.

## 2b. The precision asymmetry, and why it decides gh-487

The staged family's fate differs sharply by precision, on the same workload with the same seeding.

| tf32 replicate | arm A best | crowned segments | staged segs |
|---|---|---|---|
| r1 | 106.389 | `16x32x32`, `16x32x32`, `32x32x0` | 2 |
| r2 | 103.189 | `16x32x32`, `16x32x32`, `16x32x0` | 2 |
| r3 | 102.360 | `16x32x0`, `16x32x32`, `32x32x0` | 1 |
| r4 | 108.298 | `16x32x0`, `gpu 32x32x8/4x4`, `16x32x0` | 0 |
| r5 | 105.246 | `16x32x0`, `16x32x32`, `16x32x0` | 1 |

**4 of 5 at tf32 against 0 of 5 at bf16.** And the emission census confirms the labels rather than
merely restating them — arm A's winner replay, `.cu` grepped per replicate:

| replicate | `ldmatrix` | `__shared__` | `Tile_mma` markers |
|---|---|---|---|
| tf32 r1 | 0 | 32 | 4x `32x32x1024 (wmma-tf32)`, **16x `fragment update 16x32x32`** |
| tf32 r2 | 0 | 32 | 4x `16x32x1024`, **16x `fragment update 16x32x32`** |
| tf32 r3 | 0 | 8 | 12x `16x32x256`, 4x `32x32x1024`, **4x `fragment update 16x32x32`** |
| tf32 r4 | 0 | 8 | 4x `16x32x1024`, 12x `16x32x256` — no staged form, matching its crown |
| tf32 r5 | 0 | 8 | 4x `16x32x1024`, 12x `16x32x256`, **4x `fragment update 16x32x32`** |
| bf16 r1–r4 | **0** | 0–8 | `16x32x128` + `32x32x256`, all unstaged |

`fragment update` is gh-480's accumulator-residency rendering: the accumulator fragment stays
resident across the k-block loop while cooperative shared tiles feed it. That is a genuine staged
GEMM pipeline, in a shipping artifact, in 4 of 5 tf32 searches. (bf16 r5's snapshot was missed by the
watcher — its crown label carries no staged segment, so nothing is claimed from it.)

`ldmatrix = 0` everywhere in this table is expected and is *not* a negative result: none of these ten
crowns contained a `swz-b128` twin, and at tf32 one cannot exist — CUDA advertises exactly one
`mma_staged_layouts` triple, uniform bf16. The single search that did crown a twin is the one in §1(c),
where the same grep returns `ldmatrix = 12`.

**So the two features come apart cleanly.** The b128 swizzle is bf16-only and worth 0%; the staged
pipeline it decorates is chosen routinely at tf32, where the swizzle is unavailable. gh-487
(`cp.async` double-buffering) is precision-agnostic and targets the *staging*, not the layout — so
its addressable surface is the 4-of-5 tf32 case, and the swizzle's measured null says nothing
against it.

## 3. Ride-along for gh-550: the tune loop exhausts the device, and it is not benign

Device memory sampled at 1 Hz for the whole of every cell.

| cell | peak MiB | % of 12,227 MiB | reclaimed at exit? |
|---|---|---|---|
| default (untuned), either precision, n=4 | 912–978 | 7.5–8.0% | n/a |
| materialized, n=4 | 1,039–1,267 | 8.5–10.4% | n/a |
| **tuned, n=7** | **11,897–11,921** | **97.3–97.5%** | **no — 11,880–11,917 still held at exit** |

The program needs ~1 GB. A tuned run reaches **90% of the card 144 s into a 1,188 s search (12% of
the way through)** and then sits flat at ~11.9 GB for the remaining 88%, including through the
measurement phase after the search has ended. So ~11 GB is claimed and never released.

### It OOMed — 5 times out of 5, deterministically — and the failure mode destroys completed work

**Every one of the five tf32 replicates died, each at exactly the same point: arm B
(materialize-all), after 47 timed candidates**, with arm A having already completed its ~53
candidates and crowned a winner. Not stochastic: same index, five times.

The precision dependence has a mechanism. tf32 is a *compute* format over **f32 storage** (4 bytes),
where bf16 stores 2 — so the accumulated candidate footprint is roughly double, and tf32 crosses the
12 GB limit where bf16 (5 of 5 completed) does not. That gives gh-550 a concrete threshold rather
than only a growth curve.

```
F_sketch[gpu 16x16x8/4x4,mma-gpu 32x32x32]: FAILED at Backend_link CUDA_ERROR_OUT_OF_MEMORY   <- absorbed
F_split[max_logits ... 8 segs]:             FAILED at Backend_link CUDA_ERROR_OUT_OF_MEMORY   <- absorbed
untuned-default control compile failed:                           CUDA_ERROR_OUT_OF_MEMORY    <- absorbed
winner replay FAILED (F_saved[62 segs]), falling back to the default compile: OUT_OF_MEMORY
Fatal error: exception Invalid_argument("CUDA driver: CUDA_ERROR_OUT_OF_MEMORY")               <- kills the cell
```

The classifier absorbs the first three as ordinary `Backend_link` declines and the search continues
— that part works. But with the device exhausted, **the winner replay cannot compile and its
fallback has nothing to fall back to**, so the exception escapes `Autotune.tune` and takes the run
down. Arm A's completed result — `F_sketch[mma-gpu 16x32x32, mma-gpu 16x32x32, mma-gpu 32x32x0]` at
**106.389 ms**, the best arm-A time in this whole leg, and notable for carrying *two* staged
segments — was already in hand and was lost.

That is the sharp form of gh-550's accumulation question: accumulation does not merely waste memory,
it **lets a later, independent arm destroy work an earlier arm already finished**.


## 4. Verdict for gh-487 (cp.async software pipelining): **go**

gh-487's stated precondition is *staged tensorized segments actually shipping in crowned artifacts*.
That is met, and verified at the emission level rather than by label:

- **4 of 5** tf32 replicates crown at least one staged segment, and their shipping `.cu` carries
  8–32 `__shared__` cooperative tiles with 4–16 `Tile_mma fragment update` steps — the gh-480
  accumulator-residency form, i.e. exactly the phased load/barrier/compute/barrier loop that
  double-buffering overlaps.
- The staged form is not a curiosity that squeaks in: the two fastest arm-A results of the entire
  leg (102.360 and 103.189 ms) both carry staged segments, and the one tf32 replicate that crowned
  none is the slowest of the five (108.298 ms).

Three qualifications, so the go is not read as more than it is:

1. **At bf16 it is a no-go on current evidence** — 0 of 5 crowns staged, and segment 3's four
   competing families sit inside 1.4%. Any bf16-side gain would first have to make staged win a
   coin flip it currently loses.
2. **Do not expect gh-487 to compound with gh-481.** The swizzle is bf16-only on CUDA and measures
   0% (§1b); the staged pipeline gh-487 targets is a tf32 phenomenon here. They address disjoint
   cells on this workload.
3. **gh-550 should land first, or gh-487 cannot be measured at f32 storage at all.** Every cold tf32
   search on this box OOMs before it can report an end-to-end number (§3). The tf32 row of §2 only
   exists because a warm-cache replay sidesteps the arm-B search. A cp.async tunable would *add* a
   shared-memory-hungry variant to that same search.

The honest one-line summary: the ladder rung is reachable, on tf32, and the thing blocking its
measurement is memory accumulation rather than anything about pipelining.

## Provenance and hygiene

- **WSL2**, not native Linux. Whole-cell numbers are not comparable across machines or sessions.
- Fresh `--ocannl_autotune_cache_dir` per replicate: the disk cache is keyed by the base code's
  canonical digest and otherwise replays a previous run's winner (gh-481's hygiene rule; a warm
  cache is used *deliberately* in the emission census below, where replaying the winner is the point).
- Untuned/materialized cells were measured first, tuned cells after, references last; no two GPU
  processes ever ran concurrently.
- All claims of size < 20% come from **in-process paired comparisons** or from structural fields of
  `Autotune.report`, never from a whole-cell before/after (this box's tuned-cell replicate spread is
  20-40%, established in report-gh481-cuda.md and re-confirmed here).

### A trap this leg hit, and the instrument built to get past it

`Train.tune_placements` searches arm A, then arm B, then returns the **already-compiled** arm
(`if a_wins then a else b`, lib/train.ml). It does not recompile the winner. Debug artifacts are
keyed by routine name, so **arm B's winner replay overwrites arm A's `.cu`** — the file left on disk
when a `BENCH_TUNE=1` run exits belongs to the *discarded* arm whenever arm A ships.

Grepping that file for `ldmatrix` reads the wrong code, and it reads as a clean negative: arm B's
crowned `F_sketch[mma-gpu 32x32x0]` is unstaged, so it has zero `ldmatrix`, zero `__shared__` and
only full-K `tile_mma 32x32x256` markers. This report's first pass concluded "no `ldmatrix` ships"
off exactly that file.

The fix is a snapshot watcher: copy the `.cu` on every mtime change during the run and record the
run log's line count at capture time, which pins each snapshot to a position in the autotune trace.
The snapshot captured at or before the `tune_placements: arm A ... best:` line is arm A's winner
replay — the artifact that ships. With a **warm** cache both arms replay instead of searching, so
the whole confirmation costs minutes.

---

## Addendum (post-gh-550 robustness fix): the tf32 cells report end-to-end now

Measured after the arm-containment fix (a failed arm is a *losing* arm; the surviving arm's winner
ships), same box, staging master `3e7db701` + the fix. **5 of 5 cold tf32 searches now report a
winner and exit 0**, where 5 of 5 previously died. Protocol as in §3: one process per replicate,
`--ocannl_backend=cuda --ocannl_tf32_matmuls=true --ocannl_autotune_log=true`, a fresh
`--ocannl_autotune_cache_dir` per replicate, box otherwise idle.

| run | arm A best (ms) | arm A crowned label | arm B | shipped | step p50 (ms) | search (s) | peak MiB |
|---|---|---|---|---|---|---|---|
| r1 | **105.856** | `F_sketch[mma-gpu 16x32x0, 16x32x32, 32x32x32]` | FAILED (pre-OOM best 224.971) | **A** | 103.954 | 895 | 11,912 |
| r2 | **105.561** | `F_sketch[mma-gpu 16x32x0, 16x32x32, 16x32x0]` | FAILED (222.622) | **A** | 105.030 | 724 | 11,870 |
| r3 | **103.508** | `F_sketch[mma-gpu 16x32x0, 16x32x32, 32x32x0]` | FAILED (225.333) | **A** | 104.036 | 678 | 11,877 |
| r4 | **104.174** | `F_sketch[mma-gpu 16x32x0, 32x32x0, 32x32x0]` | FAILED (223.811) | **A** | 104.628 | 717 | 11,868 |
| r5 | **103.122** | `F_sketch[mma-gpu 16x32x0, 16x32x32, 16x32x0]` | FAILED (224.293) | **A** | 103.670 | 668 | 11,864 |

Arm A best 103.1–105.9 ms (median 104.174); end-to-end step p50 103.7–105.0 ms (median 104.036).
That median lands within 0.6% of the 103.39 ms §2 row, which had to be taken from a **warm-cache
replay** because no cold search survived — so the row's provenance caveat can now be dropped, and
gh-487's precondition "measurable at f32 storage" (§4 qualification 3) is met.

**The OOM is unchanged — only what it destroys is.** Every replicate still exhausts the card and
still fails arm B, at the same place as before: several candidates decline at `Backend_link` with
`CUDA_ERROR_OUT_OF_MEMORY` (absorbed, as they always were), then arm B's winner replay cannot
compile and its untuned-default fallback cannot either. The difference is that this now ends
arm B rather than the run: `tune_placements` ranks the failed arm at `infinity`, ships arm A's
winner, and records arm B's terminal failure in the report and in `results.jsonl`
(`"terminal_failure": "…CUDA_ERROR_OUT_OF_MEMORY raised by cu_mem_alloc"`). Arm B's pre-failure
best (222–225 ms) is deliberately not shippable — no routine was compiled from the caller's
context for it.

### The accumulation curve, sampled per candidate (ride-along for gh-550)

`nvidia-smi` memory sampled every 2 s alongside the autotune trace, so each sample is pinned to a
position in the candidate stream. Medians across the 5 replicates:

| candidates attempted | median MiB used | median t (s) |
|---|---|---|
| 0 | 396 | 0 |
| 20 | 2,663 | 9 |
| 40 | 3,500 | 15 |
| 60 | 4,430 | 23 |
| 80 | 5,037 | 27 |
| 100 | 6,089 | 36 |
| 120 | 10,522 | 54 |
| 140 | 11,802 | 154 |
| 160 | 11,835 | 244 |
| 180 | 11,859 | 287 |
| 240 | 11,869 | 428 |
| 260 | 11,831 | 697 |

Three things this says, none of which the peak-only table in §3 could:

1. **Growth is monotone in candidates attempted, not in time** — ~50 MiB per candidate through the
   first hundred, with no plateau until the card is full. Nothing is being released between
   candidates.
2. **The ceiling is hit at ~candidate 130, one fifth of the way through**, and the remaining ~130
   candidates all run against a full card — which is why the failures cluster at the end (the
   winner replay) rather than at a particular candidate's size.
3. The program itself needs ~1 GB (the untuned cells), and the at-exit reading is still
   11.7–11.9 GB, so nothing is returned after the search either.
