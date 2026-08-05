# HIP validation of staging#280 / #285, and the gpt2_mini bf16 tensor-core probe

platform: Linux-6.18.33.2-microsoft-standard-WSL2-x86_64-with-glibc2.43 x86_64 | **WSL2** |
ocannl commit: `03512535` | AMD Ryzen AI Max+ 395 (Strix Halo, Zen 5, 16C/32T) | Radeon 8060S iGPU
(gfx1151, RDNA3.5) | ROCm 7.14 / HIP 7.14.60850 | torch 2.13.0+rocm7.1, tinygrad 0.13.0

> **Everything in this report was produced under WSL2**, not bare-metal Linux. That matters twice
> here: there is no `/dev/kfd` (so tinygrad falls back from `AMD` to `HIP`, and torch needs the
> wheel's `libhsa-runtime64.so` replaced — see the README's WSL notes), and the guest's logical-CPU
> numbering is sibling-interleaved, which is what Part 0 below is about.
>
> This is a **validation** report, not a re-measurement sweep. Two merged PRs changed HIP-shared
> code with no AMD hardware in the loop — staging#280 (gh-ocannl-528, interior-batch
> `Tensorize`/`Tile_mma` with recorded `ldd`/`lda`/`ldb`) and staging#285 (gh-ocannl-481,
> `mma_fragment_syntax` and the Metal/HIP hook signatures) — so Part 1 puts them on the device.
> Part 2 asks the one question the wave was supposed to unlock: **do the new batched/rank-3 matmul
> sites reach rocWMMA on `gpt2_mini`?**

## Cell accounting

So an absent row is never ambiguous between "failed" and "not attempted".

| | full product | attempted | recorded | failed | not attempted |
|---|---|---|---|---|---|
| Part 1 — `test/operations` under `OCANNL_BACKEND=hip` | (whole dir) | (whole dir) | (whole dir) | 0 after this PR, 2 before | 0 |
| Part 1 — standalone rocWMMA characterization (4 format combos) | 4 | 4 | 4 | 0 | 0 |
| Part 2 — `gpt2_mini` hip (bf16, f32) × (default, tuned) | 4 | 4 | 4 | 0 | 0 |
| Part 2 — tuned search replicates (2 precisions × 3, each wiped-cache + pass 2) | 6 | 6 | 6 | 0 | 0 |
| Part 2 — reference frameworks (pytorch, tinygrad) × (GPU, CPU) | 4 | 4 | 4 | 0 | 0 |
| **total** | **18** | **18** | **18** | **0** | **0** |

Deliberately **not attempted**, and why: the f16 leg (`report-hip.md` diagnosed two distinct
`check_constant` rejections — the causal mask's `-1e9`, a genuine overflow, and `-inf` on
`max_vals`, a guard defect; neither is fixed, and neither is a gh-528/gh-481 question); the `cc`
column (this report is a HIP validation, and co-running `cc` timing with HIP timed cells is
forbidden on this box); and the `materialized` variant (arm B of every tuned run already reports the
materialize-all placement, which is the same information).

## Part 0 — the `taskset -c 0-15` cap does NOT give one thread per physical core

`report-hip.md` states, in the paragraph that governs every `cc` number it contains:

> **All cells ran under `taskset -c 0-15`** (16 physical cores, SMT halves excluded)

**On this machine, under WSL2, that parenthetical is wrong.** The sibling map is interleaved, not
blocked:

```
$ cat /sys/devices/system/cpu/cpu{0,1,2,3}/topology/thread_siblings_list
0-1
0-1
2-3
2-3
```

`cpu0` and `cpu1` are the two SMT threads of the *same* physical core. So `taskset -c 0-15` selects
**physical cores 0–7 with both siblings each** — half the machine, fully SMT-loaded — not 16
distinct cores. (On many bare-metal Linux boxes the numbering is blocked, `0..15` = cores and
`16..31` = their siblings, which is presumably where the parenthetical came from.)

Measured, with a pinned spin loop (16 independent FMA chains, enough ILP to keep a core's FP pipes
busy), 1.5e9 iterations, three solo reps:

| placement | per-spinner seconds | vs solo | aggregate throughput |
|---|---|---|---|
| solo, pinned `cpu0` | 1.863 / 1.868 / 1.907 | 1.00× | 0.54 |
| 8 spinners, one per distinct physical core (`0,2,…,14`) | 2.126 – 2.210 | 1.17× | 3.68 |
| 2 spinners on one core's two siblings (`cpu0`+`cpu1`) | 2.096, 2.102 | 1.12× | 0.95 |
| **16 spinners under `taskset -c 0-15`** | **2.465 – 2.526** | **1.34×** | **6.40** |

Reading: the cap runs **2 threads per physical core on 8 cores**. SMT is doing real work — 6.40
against 3.68 is a 1.74× aggregate gain over the same 8 cores — but each thread is 15% slower than it
would be with a core to itself, and the pool is 8 cores wide, not 16.

**A methodological footnote on the probe.** A naive spin loop (4 dependent FMA chains) reports *no*
contention at all: all 16 capped spinners finish in ~1.81 s against a 2.09 s solo baseline, i.e.
apparently *faster* under load. That probe is latency-bound — one thread leaves the FP pipes half
idle, so its SMT sibling costs it nothing, and the comparison measures frequency behaviour rather
than sharing. Only a throughput-saturating kernel separates the cases. Anyone re-running this check
should confirm their probe saturates a core first.

**What this does and does not change.** It does not invalidate any `cc` timing in `report-hip.md`:
those numbers were produced under this cap and remain what this machine does under this cap. It
changes their *interpretation* — the `cc` column reflects 16 threads on 8 SMT-shared cores, not 16
private cores, so it understates what the CPU backend would do with the whole machine and is not
comparable to a 16-physical-core figure from another box. The cap itself stays: it is the standing
mitigation for the all-core hard freeze, and nothing here argues for lifting it.

*(Not measured: 16 spinners on 16 distinct physical cores (`0,2,…,30`). That is the all-core load
the freeze mitigation exists for, and the placement finding above does not need it.)*

## Part 1 — HIP hardware validation of staging#280 and #285

`dune build @check` passes at `03512535`. `dune build @test/operations/runtest` under
`OCANNL_BACKEND=hip` (`-j 3`, per the iGPU concurrency cap) surfaced exactly **two** failing tests,
both analysed below; everything else — including all three `*.hip.expected` codegen snapshots
(`top_down_prec`, `zero_out_local_decl`, `test_where_precision`) — passed unchanged. **No HIP
codegen snapshot was stale**, which is worth recording since the brief expected some.

### staging#280 (gh-ocannl-528): interior-batch `Tensorize` is correct on hardware

**Verdict: no defect.** The recorded leading-dimension strides flow into the rocWMMA arm correctly,
the interior-batch site renders the intrinsic on a real gfx1151, and the values are right.

The existing `schedule_batched_mma` test could not have shown this: on GPU backends its two
execution assertions were literal `true` (`test/operations/schedule_batched_mma.ml:111-116`), so the
whole interior-batch leg was structural-only on exactly the backends the change was risky for. It
also seeds against *synthetic* limits, which is right for machine-independence but means a real
device is never asked what it advertises.

This PR adds an executed leg that closes both gaps. It seeds against
`Context.hardware_limits`, applies the real `Autotune.sketch_schedule` pipeline, executes, and
requires the emitted source to carry the backend's intrinsic. It runs at **bf16**, because f32 is a
correct null on this device — RDNA3.5 WMMA has no f32 operand shape, so `mma_format_tiles`
advertises only the f16/bf16 combinations and an f32 site seeds nothing (visible in the suite as
`no GPU mma seed for this site on hip` from `autotune_mma_companion` and
`epilogue_fusion_mma_seeds`). The interior-batch leg is the load-bearing one: with `h` between the
tile roles its `lda`/`ldb`/`ldd` are all 64 against minor dims of 32, which is precisely the case a
fragment load addresses differently from a scalar loop.

Result on gfx1151:

```
batched_lb bf16: the backend's advertised tile is seeded: true
batched_lb bf16: some candidate compiles and runs: true
batched_lb bf16: every running candidate matches the serial twin: true
batched_lb bf16: some running candidate renders the tensor-core intrinsic: true
batched_ib bf16: the backend's advertised tile is seeded: true
batched_ib bf16: some candidate compiles and runs: true
batched_ib bf16: every running candidate matches the serial twin: true
batched_ib bf16: some running candidate renders the tensor-core intrinsic: true
```

Both sites seed 5 bf16 mma candidates; 4 per site are executed (the bound is printed on stderr, not
silent — each is a full hiprtc compile against the rocWMMA headers).

### staging#285 (gh-ocannl-481): the CUDA-only features decline cleanly on HIP

**Verdict: clean.** `mma_staged_layouts = []` on HIP means swizzled staged twins are never *seeded*,
and where a swizzled operand does reach the hook, both `mma_syntax` and `mma_fragment_syntax` gate
on the identical `plain d_layout && plain a_layout && plain b_layout` predicate, so they accept and
decline together. On hardware:

- `schedule_swizzle_matmul`: `swizzled operands decline the MMA intrinsics to the lane-0 fallback:
  true`, and all 16 assertions pass.
- `schedule_ldmatrix_matmul`: all 16 assertions pass. On non-CUDA the predicate is not "it didn't
  crash" — it requires the lane-0 scalar rendering to be present **and** the census to be non-empty
  **and** every census entry to be `Mma_scalar_fallback`. A silent mislabel would fail it.
- `tile_mma_declines`: census categories intact; `staged layout not advertised: mma seeds=5
  swizzled=0 unstaged swizzled=0` is the HIP row.

No compile errors, no mis-categorised census entries, no candidates carrying an `mma-*` label that
render the fallback.

### The one real finding: gfx1151's WMMA is not exactly-rounded, and the test assumed it was

`schedule_mma_matmul`'s five bf16 legs (`bf32`, `bfu`, `bfu_ta`, `bfu_tb`, `bfu_m2`) failed on HIP —
**bitwise parity against the serial twin, with `tensorized structure as expected: true`**. That is
the "green structure, wrong values" shape, so it was worth chasing to the bottom.

It is **not an OCANNL defect.** A standalone rocWMMA program with no OCANNL in the loop — same tile
shape, same `ld`, same data — reproduces the deviation cell for cell (749/1024 cells, worst
2.08616e-07, identical values). Reduced to a single 16×16×16 `mma_sync` with a
`fill_fragment`-zeroed accumulator, against an exact `double` reference, with every input verified
to round-trip through the narrow format exactly:

| rocWMMA combination (gfx1151) | cells differing / 256 | worst abs err |
|---|---|---|
| bf16 × bf16 → f32 | 227 | 1.19e-07 (one f32 ulp at ~1.0) |
| **bf16 × bf16 → bf16** | 235 | **5.86e-03** |
| f16 × f16 → f32 | 227 | 1.19e-07 |
| f16 × f16 → f16 | 18 | 5.96e-08 |

Every product here is a multiple of 1/8 and every partial sum is bounded by 16, so the exact result
is representable in *both* accumulator formats and an exactly-rounded tensor core would return it.
gfx1151's does not, in any combination.

The test's premise — "the result is EXACT regardless of accumulation order or accumulator width" —
is therefore true on CUDA and Metal and false on HIP. The legs arrived with gh-ocannl-545 on
2026-08-04, after the last HIP-validated commit (`e687da82`, 2026-08-01), so they had never run on
AMD hardware. Note that the same file *already* carried this exception for the f16→f32 combination
(`staged+tensorized half`, "observed max abs diff ~1.3e-7"); these legs simply did not inherit it.

Fixed here by giving HIP a per-accumulator-format tolerance and dropping the now-inaccurate word
"bitwise" from those five labels. The comparison stays numeric rather than being skipped: a wrong
stride or a mis-mapped fragment moves values by O(1), not by 1e-2.

**This retires an open question from `report-hip.md`.** That report recorded
`mlp_wide`/hip/tuned/bf16 drifting `2.64e-03` against `2.60–2.67e-04` for every other bf16 cell —
"where a tensor core *is* involved, drift jumps 10×" — and left the bf16 parity constant at 4e-3
until the jump was "understood and reproduced on a second backend". It is now understood: the
uniform-bf16 combination, which is what a uniformly-bf16 network crowns, carries ~5.9e-03 of
hardware error at these magnitudes. It is a property of gfx1151, reproducible outside OCANNL, and
it will *not* reproduce on a second backend, because CUDA and Metal do not have it. **The bf16
constant should stay at 4e-3 for HIP, and the reason is now a hardware fact rather than an open
question.**

### Second finding: `narrow_storage_compute` was not backend-portable

Four accuracy assertions plus a four-line structural block are `narrow_compute_f32` /
`fp16_arithmetic` policy checks — C-backend knobs that the GPU backends ignore by design (they have
native 16-bit arithmetic). The file's header already said the structural checks are CPU-only, but
the code gated only *some* of them, and the structural block was inside `if on_cpu then begin … end`
with no else, so a GPU run emitted 12 lines against a 15-line golden and could never match it
however it behaved. Pre-existing, unrelated to this wave. Gated consistently here, with the vacuity
note on stderr per the repo convention.

## Part 2 — `gpt2_mini` on HIP: does the batched site reach rocWMMA?

**Yes — seeded, timed, crowned, and verified in the emitted source.** This is the first time
`gpt2_mini` has produced a *timed* tensor-core candidate on any backend. `report-hip.md` recorded
the opposite: the workload's only mma candidates were the ones its variance-style layer-norm site
was mis-detected into, "all failing at candidate compile". gh-ocannl-528's role exclusions now
reject that site (pinned in `schedule_batched_mma`) and its batched detection finds the real ones.

Two things had to be true first, and both now are:

- **bf16 compiles.** `gpt2_mini`/hip/default/bf16 died on the HIPRTC `__hip_bfloat16`/`float`
  overload ambiguity in `report-hip.md`; the gh-ocannl-549 fix holds on hardware — **82.69 ms/step**
  (p10 82.20, p90 83.54), compile 3.95 s.
- **The gh-533 scratch pre-validator now declines instead of raising.** It was fatal to the whole
  cell before ("the pre-validator raises out of `Autotune.tune` instead of recording a decline and
  moving to the next candidate" — `report-hip.md`, Still open). It is now a per-candidate decline
  and the search continues past it. That open item is closed.

### The census, per arm, from `BENCH_TUNE_REPORT=1` against a wiped cache

Replicate 1 (search 866.6 s):

| arm | mma **seeded** | mma **timed** | scalar-fallback impostors | best candidate | tensorized? |
|---|---|---|---|---|---|
| A (default placements) — **shipped** | 41 | **16** | **0** | 46.1153 ms `F_sketch[mma-gpu 16x32x0,mma-gpu 32x32x0,mma-gpu 32x32x0]` | **yes** |
| B (materialize-all) | 46 | **11** | **0** | 73.4295 ms `F_sketch[mma-gpu 32x32x0,mma-gpu 16x32x32]` | **yes** |

Both arms crown a tensorized composite, and `mma_scalar_fallbacks = 0` in both: no candidate
carrying an `mma-*` label rendered the scalar fallback. `NOTE tensorized candidate emitted no
Tile_mma statement` appears zero times.

### Verified in the emitted source, not from the label

With `output_debug_files_in_build_directory=true`, replaying the crowned schedule:

| check | result |
|---|---|
| `rocwmma::mma_sync` calls | **16** |
| `/* tile_mma 32x32x256 (rocwmma) */` blocks | 12 |
| `/* tile_mma fragment update 16x32x32 (rocwmma) */` (gh-480 resident form) | 4 |
| lane-0 scalar fallback guards (`== 0)`) | **0** |
| fragment element type | `rocwmma::bfloat16_t` throughout (accumulator, matrix_a, matrix_b) |

*Caveat on this artifact:* the fissioned segments share the routine name `cross_entropy_loss_fwd__seg`,
so each overwrites the previous one's debug files (the collision CLAUDE.md warns about) and the
inspected file is the last segment written, not all 58. It is sufficient to establish that real
rocWMMA is emitted and no fallback is; it is not a census of every segment.

### What tensorization is actually worth here — paired, in-process

The search sweeps sites in rounds: within a round it times 5 scalar `gpu` sketch variants and 5
`mma-gpu` variants against the same site, then composes the per-site winners. Pairing is therefore
**within a round**, and comparing a candidate from one round against one from another would compare
different sites. Arm A, replicate 1, best-of-family per round:

| round | best non-mma | best mma | delta |
|---|---|---|---|
| 1 | 75.2546 ms `F_sketch[gpu 32x32x16/2x2]` | 68.3107 ms `F_sketch[mma-gpu 16x32x0]` | **−9.2%** |
| 2 | 79.8544 ms `F_sketch[gpu 16x16x8/2x2]` | 77.7514 ms `F_sketch[mma-gpu 32x32x0]` | **−2.6%** |
| 3 | 64.1921 ms `F_sketch[gpu 32x32x16/2x2]` | 62.7379 ms `F_sketch[mma-gpu 32x32x0]` | **−2.3%** |

The mma family wins every round in which both were timed. That is what tensorization is worth at a
single site on this workload: **2–9%**, not the cell-level number.

**The cell-level number is 1.80×, and most of it is not tensorization.** The composite of the three
per-site winners is 46.1153 ms against 62.7379 ms for the best single-site candidate, and the
end-to-end tuned artifact is 45.98 ms against 82.69 ms untuned. But **the composition step only ever
composes the winners**, and every winner here was an mma candidate, so no scalar composite was
timed. The share of the composite's gain attributable to tensorization rather than to composing
three fission sites at once **is not separable from this data**. Stated plainly because the
temptation to read 1.80× as a tensor-core result is exactly what the reporting contract is for.

### The two-pass timings

| pass | compile s | step p50 | p10 | p90 |
|---|---|---|---|---|
| 1 (the search process) | 866.609 | 46.109 | 45.570 | 46.659 |
| 2 (fresh process, cached winner) | 46.589 | **45.983** | 45.723 | 46.139 |

Loss trajectories are identical across the two passes to all printed digits.

### Decline families in this search, all clean

Every one of these recorded a decline and moved on; none aborted the search or the queue.

| family | where | what |
|---|---|---|
| gh-533 scratch budget | `baseline`, all 5 `W_preset` | `cross_entropy_loss_fwd` needs 221200 B/work-item against 104832 B backable |
| gh-521 companion coverage | all 9 `W_sketch`, and the majority of `F_sketch` rounds | "the accumulation nest's aligned chain was trimmed below its 8x128x1024 geometry" |
| `Hardware_limits` shared memory | all 5 `mma-gpu … ep` | stages ~2.1 MB of workgroup-shared tiles against a 65536 B device limit |
| `Fuse_epilogue` | all 5 scalar `gpu … ep` | "output loop … on the write path must be Serial or Grid" |

The gh-521 companion-coverage family is still the dominant blocker, exactly as `report-hip.md`
identified for the mlp workloads — and it is now visibly the reason the *whole-routine* (`W_sketch`)
arm contributes nothing on this workload: all nine decline there.

### Replicates: the census is exactly stable, the crowned label almost

Three replicates, each against a **wiped cache dir** (protocol item 7), each followed by a fresh
pass-2 process replaying the cached winner:

| rep | search s | arm A best | crowned label | seeded/timed A | arm B best | seeded/timed B | pass-2 p50 |
|---|---|---|---|---|---|---|---|
| 1 | 866.6 | 46.1153 | `[mma-gpu 16x32x0, 32x32x0, 32x32x0]` | 41 / 16 | 73.4295 | 46 / 11 | 45.983 |
| 2 | 179.7 | 45.7118 | `[mma-gpu 16x32x0, 32x32x0, 32x32x0]` | 41 / 16 | 73.3443 | 46 / 11 | 46.039 |
| 3 | 588.5 | 45.1554 | `[mma-gpu 16x32x0, **16x32x32**, 32x32x0]` | 41 / 16 | 72.1428 | 46 / 11 | 46.623 |

- **Seeded/timed counts and `mma_scalar_fallbacks = 0` are identical in all three replicates, in both
  arms.** The reachability claim is not a one-run artifact.
- The crowned composite is a three-site tensorized `F_sketch` every time; only the middle site's
  geometry differs in replicate 3, and the resulting artifacts land within 1.4% of each other.
- **Search wall-time is not stable** — 866.6 / 179.7 / 588.5 s for identical work against a wiped
  autotune cache. The autotune cache was wiped; whatever is warm across replicates (hiprtc/hsaco
  reuse, page cache) is not. Search *cost* numbers from single runs on this machine should not be
  quoted as if repeatable; the artifact they produce is.

### The f32 leg: the correct null, three times

| rep | search s | arm A best | crowned label | tensorized | mma seeded | pass-2 p50 |
|---|---|---|---|---|---|---|
| 1 | 496.1 | 46.2619 | `F_sketch[gpu 16x16x8/2x2, 16x16x8/2x2, 32x32x8/4x4]` | **false** | **0** | 46.132 |
| 2 | 119.7 | 45.5334 | `F_sketch[gpu 16x16x8/2x2, 16x16x8/2x2, 32x32x16/2x2]` | **false** | **0** | 45.746 |
| 3 | 117.3 | 46.7795 | `F_sketch[gpu 16x16x8/2x2, 16x16x8/2x2, 32x32x8/4x4]` | **false** | **0** | 46.300 |

`mma_seeded = 0` in every f32 replicate. That is the right answer, not a failure: RDNA3.5 WMMA has
no f32 operand shape, so nothing is proposed and there are no declines to explain. It is also the
control that makes the bf16 census mean something — the same graph, the same search, the same
machine, and the only thing that changed is the operand format.

### The result that matters most: **the tensor-core route buys nothing end to end**

| cell | step p50 (pass 2) | tok/s | vs its own default |
|---|---|---|---|
| ocannl hip **default** f32 | 65.372 | 15,665 | — |
| ocannl hip **tuned** f32 | **46.13** (45.75 / 46.13 / 46.30) | 22,197 | **1.42×** |
| ocannl hip **default** bf16 | 82.692 | 12,384 | — |
| ocannl hip **tuned** bf16 | **46.04** (45.98 / 46.04 / 46.62) | 22,241 | **1.80×** |

**Tuned bf16 (46.04 ms) and tuned f32 (46.13 ms) are indistinguishable** — the ranges overlap almost
completely (45.98–46.62 against 45.75–46.30). So on `gpt2_mini`, a schedule that reaches rocWMMA at
three sites arrives at the same place as one that never touches a tensor core. The bf16 leg's larger
*ratio* (1.80× vs 1.42×) is mostly explained by its worse starting point: **bf16 costs 26% before
tuning** (82.69 vs 65.37 ms untuned), and tuning recovers that deficit rather than beating f32.

This is the same shape `report-hip.md` found on `mlp_wide` — "it does not yet make bf16 worth using
… the format change costs more than the tensor cores return" — reproduced on a second, much larger
workload, and now with the tensor cores demonstrably reached at three sites rather than one. Whether
that means the tensor cores are not being fed well or that this workload is not matmul-bound is not
answerable from these numbers; the per-round deltas (2–9%) suggest the sites that tensorize are not
where the step time is.

### Cross-framework, same session, same box

| framework | backend | variant | precision | step p50 ms | tok/s | parity |
|---|---|---|---|---|---|---|
| pytorch | cuda(hip) | eager | f32 | 6.268 | 163,374 | PASS (6.7e-08) |
| tinygrad | HIP | jit | f32 | 6.880 | 148,830 | PASS (1.3e-07) |
| pytorch | cpu | eager | f32 | 28.066 | 36,486 | REF |
| **ocannl** | **hip** | **tuned** | **bf16** | **46.04** | **22,241** | (loss trajectory only) |
| **ocannl** | **hip** | **tuned** | **f32** | **46.13** | **22,197** | (loss trajectory only) |
| ocannl | hip | default | f32 | 65.372 | 15,665 | (loss trajectory only) |
| ocannl | hip | default | bf16 | 82.692 | 12,384 | (loss trajectory only) |
| tinygrad | CPU | jit | f32 | 99.101 | 10,333 | PASS (8.7e-07) |

The four reference cells ran through `orchestrate.py` and passed its parity gate. The OCANNL cells
were driven directly through `bench_gpt.exe` (to control the cache dir and capture the census), so
they carry no cross-framework parity verdict here — their loss trajectories are identical across all
three replicates and both passes, and match between search and replay to every printed digit, but
that is a self-consistency check, not the gate.

**The ledger line for gh-ocannl-531: `gpt2_mini` on HIP is now 7.4× off torch, not 11.7×, and
tuning moves it for the first time.** `report-hip.md` had no tuned `hip` cell at any precision
(the gh-533 validator killed the f32 search, HIPRTC killed bf16) and recorded
default 70.342 ms against torch's 6.013 — with "tuning and materialization both move nothing" as
the issue's framing. Both blockers are gone, and tuning is worth 1.42×/1.80×.

## Still open

- **The gh-521 companion-coverage decline is now the whole story on this backend.** It takes all 9
  `W_sketch` candidates and the majority of `F_sketch` rounds on `gpt2_mini`, exactly as it took 12
  of 29 and 16 of 37 on the mlp workloads. Nothing else is close to it as a blocker family.
- **Reaching the tensor cores is no longer the question; feeding them is.** Three tensorized sites,
  verified in the emitted source, land within noise of a schedule with none. The per-round deltas
  (2–9%) say the sites that tensorize are not where the step time is.
- **Search wall-time varies 4.8× across identical wiped-cache replicates** (866.6 / 179.7 / 588.5 s).
  Worth understanding before any search-cost number is quoted as a property of a workload.
- **The bf16 parity constant should stay at 4e-3, and the reason is now settled** — gfx1151's
  uniform-bf16 WMMA carries ~5.9e-03 of hardware error at these magnitudes (Part 1). It will not
  reproduce on CUDA or Metal, so "reproduce it on a second backend before tightening" cannot be the
  gate; it is a per-backend constant.
- **The f16 leg still does not run** — both `check_constant` rejections from `report-hip.md` stand.
  The `-inf` one remains a guard defect independent of the genuine `-1e9` overflow beside it.
- The `mma-gpu … ep` variants all decline on shared-memory limits (~2.1 MB staged against 65536 B).
  If epilogue fusion is wanted on this workload, the staged tile geometry has to shrink.
- Debug artifacts for fissioned segments collide on the routine name (`cross_entropy_loss_fwd__seg`),
  so only the last segment's source survives. That made whole-schedule intrinsic verification
  impossible; a per-segment suffix would fix it.

## Reproducing

All from `benchmarks/`. OCANNL runs need no `LD_LIBRARY_PATH`; the torch reference cells do
(`LD_LIBRARY_PATH=/opt/rocm/lib/rocm_sysdeps/lib`, plus the wheel's `libhsa-runtime64.so` replaced —
see the README's WSL notes).

```bash
# Part 0: the SMT placement check (spin.c / spin2.c are in the report's scratch dir; the
# structural half needs no build at all)
for c in $(seq 0 31); do echo "cpu$c -> $(cat /sys/devices/system/cpu/cpu$c/topology/thread_siblings_list)"; done
```

```bash
# Part 1: the HIP suite
OCANNL_BACKEND=hip taskset -c 0-15 dune build @test/operations/runtest -j 3
```

```bash
# Part 1: the standalone rocWMMA characterization (no OCANNL)
/opt/rocm/bin/hipcc --offload-arch=gfx1151 -O2 -o micro2 micro2.cpp && ./micro2
```

```bash
# Part 2, pass 1: the gpt2_mini bf16 tuned search, wiped cache, with the seeded/timed census.
# Drop BENCH_PRECISION for the f32 leg (which must report mma_seeded=0 — the correct null).
rm -rf /tmp/probe
BENCH_FIXTURE=fixtures/gpt2_mini.safetensors BENCH_TUNE=1 BENCH_TUNE_REPORT=1 BENCH_PRECISION=bf16 \
  taskset -c 0-15 ../_build/default/benchmarks/runners/ocannl/bench_gpt.exe --ocannl_backend=hip \
  --ocannl_autotune_cache_dir=/tmp/probe --ocannl_autotune_log=true

# pass 2: a fresh process replays the cached winner — these are the step timings quoted above
BENCH_FIXTURE=fixtures/gpt2_mini.safetensors BENCH_TUNE=1 BENCH_PRECISION=bf16 \
  taskset -c 0-15 ../_build/default/benchmarks/runners/ocannl/bench_gpt.exe --ocannl_backend=hip \
  --ocannl_autotune_cache_dir=/tmp/probe

# the intrinsic check — the label is not evidence, the emitted source is
BENCH_FIXTURE=fixtures/gpt2_mini.safetensors BENCH_TUNE=1 BENCH_PRECISION=bf16 \
  taskset -c 0-15 ../_build/default/benchmarks/runners/ocannl/bench_gpt.exe --ocannl_backend=hip \
  --ocannl_autotune_cache_dir=/tmp/probe --ocannl_output_debug_files_in_build_directory=true
grep -c "rocwmma::mma_sync" build_files/bench_gpt/*.hip
grep -c "== 0)" build_files/bench_gpt/*.hip   # must be 0: no lane-0 scalar fallback

# the per-round paired comparison (scalar vs mma family within one site's round)
grep -E "^autotune: F_sketch\[" /tmp/search.log
```

```bash
# Part 2: the reference frameworks, through the parity gate
LD_LIBRARY_PATH=/opt/rocm/lib/rocm_sysdeps/lib taskset -c 0-15 \
  .venv/bin/python orchestrate.py --workloads gpt2_mini --gpu hip --only pytorch tinygrad
```
