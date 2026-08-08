# gh-ocannl-569 on HIP: is the companion-coverage blocker cross-backend?

Measurement report. [ahrefs/ocannl#569](https://github.com/ahrefs/ocannl/issues/569) was filed off a
CUDA profile ([`report-gh531-profile.md`](report-gh531-profile.md)): five non-attention FFN-class
kernels at **70.2%** of the `gpt2_mini` step, running at **1.3%** of that card's measured fp32 peak,
invariant across untuned / tuned-scalar / tuned-tensorized, blocked by gh-ocannl-521's
companion-coverage rule at the `8x128x1024` output geometry. This asks whether HIP shares the shape.

**Verdict: the blocker is cross-backend, and it is the same rule at the same geometry on the same
five kernels — but its size is materially smaller here. On HIP those five kernels are 47.2% of the
step (not 70.2%), running at 5.6% and 2.5% of this device's measured achievable fp32 GEMM
throughput (not 1.3%).** The measured prize is correspondingly smaller: spreading the blocked output
axis is worth **4.47x** on the production geometry, bitwise verified, against CUDA's 5.91x — and
because the bucket is 47% rather than 70%, that is a **~1.31x** whole-step floor here against CUDA's
~2.4x.

## Provenance

- Box: AMD Ryzen AI Max+ 395 ("Strix Halo", Zen 5, 16C/32T) with a **Radeon 8060S iGPU — gfx1151,
  RDNA3.5, 40 CUs reported as 20 workgroup processors by `hipGetDeviceProperties`, 2900 MHz**.
  **WSL2**, kernel 6.18.33.2-microsoft-standard-WSL2. ROCm / HIP **7.14.60850**. Otherwise idle.
- Tree: `6c94d7ca`, a worktree of staging master `137b9042` plus this session's gh-569 fix
  (see [Part 0](#part-0--the-tree-this-was-measured-on)).
- Workload: `benchmarks/fixtures/gpt2_mini.safetensors` — 4 layers, d=256, 8 heads, seq 128,
  vocab 1024, batch 8, forward-only (`mode: infer`), 1024 tokens/step.
- Precision: **f32** (`default_prec=single`). This is the right comparand for CUDA's tf32 leg — both
  are f32 storage — and it is also the only honest one here: RDNA3.5 WMMA has no f32 operand shape,
  so `mma_seeded = 0` and no tensorized candidate is ever proposed. That is the correct null, not a
  failure, and it reproduces the f32 leg of [`report-gh528-hip.md`](report-gh528-hip.md).
- Cache discipline: one **cold** search into a fresh `--ocannl_autotune_cache_dir=/tmp/gh569-hip-f32`,
  never shared across precision policies
  ([ahrefs/ocannl#568](https://github.com/ahrefs/ocannl/issues/568): the cache key omits the
  Numerics policy). All CPU-side work under `taskset -c 0-15`; GPU-timed cells run with the CPU
  otherwise quiet, since CPU and iGPU share the LPDDR5X controller.
- Bucket **shares** are the claim-bearing quantity, over 3 repetitions; absolute nanoseconds are not.

### There is no profiler on this box, and that changed the method

`rocprofv3 --kernel-trace` collects **nothing** here. The cause is structural, not a
misinvocation: WSL2 exposes the GPU through `/dev/dxg` and there is **no `/dev/kfd`**, which ROCm's
profiler requires. Verified against a ten-line HIP program, not just against OCANNL — a trivial
`hipLaunchKernelGGL` loop also produces an empty output directory; `--runtime-trace` hangs outright.
So the nsys timeline that Part 1 of the CUDA report rests on has no counterpart here.

The substitute is [`gpt2_kernel_harness.py`](gpt2_kernel_harness.py): it takes the emitted batch
source and the launch geometry the compile actually used (config `schedule_log_launches`), and
generates a HIP program that compiles those kernels as-is and times **each one individually** with
HIP events at its real grid/block. What that buys and what it costs:

- It is a **reconstruction, not a timeline**. Kernels run in isolation on synthetic buffers, so it
  cannot see inter-kernel gaps, and each kernel meets a cache the real step would have left
  differently. Nothing in this report claims a gap or launch-overhead number; that half of the CUDA
  profile is simply not reproduced here.
- It is sound for *these* kernels because every loop bound in the emitted source is a literal and
  the only data-dependent construct is a `select`-shaped `Where`, so the work done does not depend
  on buffer contents. Empirically confirmed: the four FFN GEMMs measure within ±1% across three
  runs whose accumulator contents differ by construction.
- **The validation is arithmetic, and it is the reason these numbers are usable at all: the sum of
  the 117 per-kernel medians is 46.653 / 44.340 / 46.636 ms against a measured step p50 of
  47.39–47.60 ms — agreement to 1.6–2.0%.** Independently, the clean-room probe below reproduces
  the shipped FFN kernel to **0.3%** (3.48 ms against the in-step 3.489 ms).

## Part 0 — the tree this was measured on

The four merges that landed since this box last ran (#498 `memory_budget`, #563 canonical-render
extraction, #564 pre-dispatch declines, #487 phase 1) were validated on HIP first, because a profile
over broken codegen is worse than no profile. Result: **`dune build @check` green, and the whole HIP
suite clean except one test — with zero stale codegen snapshots.** The offset-shifting churn #498
was expected to cause simply did not materialize on HIP; nothing was promoted.

The one failure was real and is fixed in this branch (`6c94d7ca`): gh-564 put *all* of
`Context.check_runnable` inside the timing run's `Preflight` region, which contains every failure as
a per-candidate decline. Two of its three checks — an uninitialized input, an unexecuted dependency
— are **lineage-wide**: they fail every candidate of every arm identically. Contained, they are
silent on every GPU backend, because the serial baseline is refused there (gh-ocannl-532) and so
nothing validates the lineage before the search: every candidate declines for the one reason,
nothing is timed, and `Train.tune_placements` returns normally shipping the untuned default out of a
lineage in which that routine cannot run. On the C backends the dispatched serial baseline hits the
error first and takes the arm down with the caller's message, which is why CI never saw it. The fix
splits `check_lineage_runnable` from `check_launch_bindings` and raises only the former outside the
failure boundary — the same reasoning the poisoned-lineage check was already raised outside it for.

Guard applied to every search below, since that failure mode would silently invalidate a profile:
`nothing was timed` fallbacks **0**, candidates timed **77**, terminal failures **0**, baseline
dispatched.

One HIP failure remains and is **pre-existing**, reproduced at the pre-wave base `a7672848`:
`autotune_arm_containment`'s `arm B had timed candidates before failing`. It is a cc-shaped
assertion — on `cc` arm B's serial baseline is dispatched and timed, on HIP it is refused and the
three candidates before the injection point all dedup — of the same class as the
`autotune_fission_sketch` divergence gh-543 fixed. Not this wave's doing, and orthogonal to codegen.

## Part 1 — the search, and the bucket breakdown

Cold search, 292.1 s. Arm A ships:

```
tune_placements: arm A (default placements) best: 47.1433 ms
  (F_sketch[gpu 16x16x8/2x2, gpu 16x16x8/4x4, gpu 32x32x8/4x4], best tensorized none)
tune_placements: arm B (materialize-all) best: 67.2550 ms (F_sketch[gpu 16x16x8/4x4])
tune_placements: winner: arm A (A 47.1433 ms vs B 67.2550 ms)
```

`mma_seeded = 0` in both arms — the f32 null. Untuned default 65.5 ms, so tuning is worth 1.39x.
Replay step p50 **47.39 / 47.48 / 47.60 ms**.

**Reading the right artifact.** As on CUDA, both arms compile a routine named
`cross_entropy_loss_fwd`, so the file left on disk is **arm B, the discarded arm**. Arm A has
**117** kernels and arm B **130** — the same 117/130 split the CUDA report found, confirmed
independently by the launch log (`seg 0/117` then `seg 0/130`). Arm A was captured with a
content-polling watcher and verified two ways: 117 `__global__`, balanced braces, and it compiles
cleanly under `hipcc --offload-arch=gfx1151` (arm B's 130-kernel file also compiles, so the kernel
count is what separates them; on the f32 leg there is no `mma_sync` in either arm, so the CUDA
report's third discriminator is unavailable).

### The four buckets

117 kernels, three repetitions:

| bucket | kernels | ms/step | share (r1 / r2 / r3) | CUDA (tf32) |
|---|---:|---:|---|---:|
| **FFN GEMMs** | 35 | 18.94 | **40.6% / 42.5% / 40.5%** | 57.7% |
| **attention** | 52 | 15.20 | **32.6% / 34.3% / 32.8%** | 25.5% |
| **embedding / logits** | 12 | 8.18 | **17.5% / 13.8% / 17.6%** | 14.6% |
| **layernorm / elementwise** | 18 | 4.33 | **9.3% / 9.4% / 9.2%** | 2.2% |
| total | 117 | 46.65 | 100% | 100% |

Classification is by [`gpt2_bucket.py`](gpt2_bucket.py), which seeds each kernel from the named
model weights in its signature and propagates only for kernels naming none. **94.1% of kernel time
is directly seeded**; the propagation heuristic carries 5.9%, so the table is essentially read off
the emitted source rather than inferred.

Shares are stable to ~2 pp except embedding/logits, and that one has a single cause: the lm_head
kernel is the least reproducible thing in the step (7.945 / 5.888 / 7.989 ms, a 36% spread) while
the four FFN GEMMs vary by ±1%. Reported rather than smoothed.

**Two differences from CUDA are real and worth naming.** The FFN bucket is 17 pp smaller and the
layernorm/elementwise bucket is 4x larger. The second is the more interesting: it is the *Virtual
residual stream*, which the CUDA report identified as a structural outlier worth ~0.2 ms there. On
HIP it costs 4.3 ms — 9.3% — and it is visible directly in the kernel signatures, because each
LayerNorm site re-derives the residual by re-summing every prior contribution:

```
seg5   (ln1 l0): gamma/beta + wpe
seg31  (ln1 l1): gamma/beta + wpe + l0_ffn_b2
seg57  (ln1 l2): gamma/beta + wpe + l0_ffn_b2 + l1_ffn_b2
seg83  (ln1 l3): gamma/beta + wpe + l0_ffn_b2 + l1_ffn_b2 + l2_ffn_b2
seg109 (lnf)   : gamma/beta + wpe + all four ffn_b2 + all four attention outputs
```

That triangle is the quadratic-in-depth recomputation, and on this device it is a 9.3% line item
rather than a rounding error. (It is also a classification trap: seeding on the FFN biases first
files all nine LayerNorm sites under FFN and empties the elementwise bucket, which is why
`gpt2_bucket.py` seeds gamma/beta first — a gain/bias pair is definitional, and no GEMM names one.)

### The step is five kernels — the same five

| seg | ms/step | share | what it computes |
|---|---:|---:|---|
| 25, 51, 77, 103 | 3.489 / 3.570 / 3.516 / 3.505, **14.08 total** | 30.2% | FFN GEMM 1 (`d->d_ff`) + gelu epilogue, one per layer |
| 111 | **7.945** | 17.0% | tied lm_head (`d->vocab`) + logits epilogue |

**22.02 ms = 47.2% of the step** (44.9% / 47.2% across the other two reps). Confirmed from the
signatures, not from the classifier: `seg25` takes `l0_ffn_w1`, `l0_ffn_b1`, `n309_layer_norm` and
writes `n311`, `n339_gelu`; `seg111` takes `wte`, `n794_layer_norm` and writes `logits`,
`max_logits`.

And the shipped source is **the same naive scalar form CUDA ships**, verbatim:

```c
for (int i1705 = 0; i1705 <= 1023; ++i1705)          // output, serial
  for (int i1706 = 0; i1706 <= 255; ++i1706)         // reduction, serial
    n311[((i1703)*128 + i1704)*1024 + i1705] =
        fmaf(l0_ffn_w1[(i1705)*256 + i1706],
             n309_layer_norm[((i1703)*128 + i1704)*256 + i1706],
             n311[((i1703)*128 + i1704)*1024 + i1705]);
```

launched at `grid=(8,1,1) x block=(128,1,1)` — **1024 threads**, so 8 of the device's 20 workgroup
processors hold any work. The occupancy signature is not confined to the dominant kernels either:
**96 of arm A's 117 kernels launch at exactly that geometry.**

| grid x block | kernels | threads |
|---|---:|---:|
| `(8,1,1) x (128,1,1)` | **96** | 1024 |
| `(2,8,1) x (8,8,1)` | 12 | 1024 |
| `(8,4,1) x (8,8,1)` | 4 | 2048 |
| `(2,8,1) x (4,4,1)` | 4 | 256 |
| `(1,1,1) x (1,1,1)` | 1 | 1 |

## Part 2 — utilization against a measured local roofline

The roofline is measured on this device, not taken from a spec sheet — and on a bandwidth-shared APU
that matters more than usual, because the "device memory" is the same LPDDR5X the CPU uses, so the
bandwidth leg is a shared-controller number and is only meaningful with the CPU quiet.
[`roofline_hip.cpp`](roofline_hip.cpp):

| leg | measured |
|---|---:|
| rocBLAS `sgemm` 4096^3, fp32 | **2.757 TFLOP/s** |
| dependency-free FMA issue peak | 24.21 TFLOP/s |
| device-to-device copy, 256 MiB (read+write) | **210.1 GB/s** |

The sgemm leg is the primary denominator, because it is the same choice the CUDA report made
(cuBLAS 4096^3 -> 17.2 TFLOP/s) and because it is what a *GEMM* can actually reach here. The
8.8x gap between it and the FMA issue peak is itself notable — rocBLAS reaches only ~19% of this
part's theoretical non-dual-issue fp32 rate — but using the higher number would flatter the kernels
below, so the harder-to-beat sgemm figure is used throughout.

FLOPs and compulsory bytes are analytic from the model geometry (exact for the GEMMs); times are the
harness medians.

| kernel | n | ms/step | share | GFLOP/s | % sgemm peak | GB/s | % BW | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| FFN GEMM1 + gelu | 4 | 14.08 | 30.2% | 153.9 | **5.6%** | 3.0 | 1.4% | **well under** |
| tied lm_head | 1 | 7.95 | 17.0% | 67.6 | **2.5%** | 0.8 | 0.4% | **well under** |

**At-roofline: 0.00 ms. Well under roofline: everything.** Same classification as CUDA. Both
kernels are 536.9 MFLOP (`2*1024*256*1024`), so the comparison between them is exact: the lm_head
costs 2.3x what one FFN GEMM costs for identical FLOPs.

The caveats the CUDA report attaches apply here unchanged and are not re-argued: the bytes column is
analytic *compulsory* traffic, a lower bound, and being far under a device-wide envelope does not by
itself prove a kernel is not compute-bound. What settles the mechanism is the control in Part 4.

## Part 3 — the decline census: same rule, same geometry

Read from the cold search's `schedule_log_declines` output, both arms:

| decline | count | geometry |
|---|---:|---|
| **gh-521 companion coverage** | **25** | **`8x128x1024`** |
| gh-521 companion coverage | 15 | `8x128x8x128` |
| gh-521 companion coverage | 15 | `8x128x256` |
| gh-521 cross-nest race bail | 10 | whole-routine (317–318 companion nests) |
| gh-521 cross-nest race bail | 4 | fissioned (100 companion nests) |
| gh-533 HIP scratch budget | 6 | `cross_entropy_loss_fwd`, 163856 B/work-item vs 104832 B backable |
| `Fuse_epilogue` write-path not Serial/Grid | 10 | — |

The message is identical to CUDA's, down to the wording: *"the accumulation nest's aligned chain was
trimmed below its `8x128x1024` geometry, so its companions cannot share it"*. `8x128x1024` is
exactly the output geometry of FFN GEMM1 and of the lm_head; `8x128x8x128` is the out-projection and
QK^T. **This is the same rule declining the same sites at the same shapes on a different vendor's
GPU.**

Two HIP-specific notes. First, counts are not comparable to CUDA's table (that one is arm A only,
and this workload proposes no tensorized candidates at all here, so every declined seed is scalar) —
the *geometries* are the comparable part. Second, HIP declines a geometry CUDA does not:
**`8x128x256`, 15 seeds** — the output shape of FFN GEMM2. Where CUDA's smaller `8x128x256` site was
reachable and got 15-20x faster from tuning, on HIP it is declined by the same coverage rule. That
is a genuine extra reach of the blocker on this backend, and it is consistent with the FFN bucket
here being harder to move than the raw share suggests.

Where CUDA saw `Hardware_limits` shared-memory declines, HIP instead reports the gh-533 scratch
budget refusal on the unscheduled CE head. Both are "the untuned form is too big"; neither touches
the five dominant kernels.

## Part 4 — the prize, measured

[`ffn1_geometry_probe_hip.cpp`](ffn1_geometry_probe_hip.cpp), the HIP analog of the CUDA
`ffn1_geometry_probe.cu`. Clean-room reimplementation of the shipped FFN up-projection at the real
geometry, then the same control: each thread keeps its inner `k` loop and its token mapping, and
only the `j` (output) range is split across `blockIdx.y` in **contiguous** chunks — so every thread
walks the same addresses in the same order, and the only thing that varies is how many blocks are
resident. Inputs are non-periodic by construction, so a variant computing the wrong chunk cannot
accidentally match.

| variant | blocks | ms | vs shipped | verified |
|---|---:|---:|---:|---|
| **as shipped**, `grid=(8,1)` | 8 | **3.48** | 1.00x | reference |
| `j` chunked, `grid=(8,2)` | 16 | 2.33 | 1.49x | bitwise identical |
| `j` chunked, `grid=(8,4)` | 32 | 1.41 | 2.47x | bitwise identical |
| **`j` chunked, `grid=(8,16)`** | 128 | **0.78** | **4.47x** | bitwise identical |
| `j` chunked, `grid=(8,128)` | 1024 | 2.06 | 1.69x | bitwise identical |

Bitwise comparison is legitimate here and does not violate the gfx1151 WMMA caveat: **no variant
uses WMMA**, and each thread performs the identical `fmaf` chain in the same order, so equality is
exact. Every variant is checked cell-by-cell against the shipped kernel's output before its time is
reported.

**The shipped row reproduces production to 0.3%** — 3.48 ms here against the 3.489 ms the harness
measures for `seg25` in the step — so this is a measured replacement, not a cross-toolchain
estimate, with the gelu epilogue inside both numbers.

**Two conclusions, and one divergence from CUDA.**

1. The binding resource is parallelism, not traffic. Time falls with resident block count while the
   bytes touched and the order they are touched in are identical in every row. A byte-bound kernel
   does not do that.
2. **HIP's curve is not monotone.** CUDA's saturates at 1024 blocks (5.91x); HIP peaks at **128
   blocks (4.47x)** and then *regresses* to 1.69x at 1024. At that point the per-block chunk is 8
   output columns, and each block re-reads the same 256-element activation row — the reuse collapses
   faster than the added parallelism pays. So the tuning target on this device is a chunk count
   around 128, not "as many blocks as possible", which is a concrete difference a fix should know
   about.

Post-replacement the kernel runs at 688 GFLOP/s — **25.0% of the measured sgemm peak**, against
CUDA's post-replacement 1.3%. The reachable ceiling is much closer here.

### What that is worth end to end

| | now | measured replacement |
|---|---:|---:|
| FFN GEMM1 + gelu x4 (4.47x) | 14.08 | ~3.15 |
| tied lm_head | 7.95 | not measured (needs fission first) |
| **step (kernel time)** | **46.65** | **~35.7** |

**~1.31x from the FFN up-projections alone**, against CUDA's ~2.4x for the full five. The lm_head is
deliberately left unmeasured: as the CUDA report established, its output axis carries the
`max_logits` reduction, so it cannot simply be chunked — it has to be fissioned into a GEMM half and
a reduce half first, and that fission was not built for HIP here. If it behaved like its CUDA
counterpart the step would reach ~30 ms (~1.6x), but that is an extrapolation and is not claimed.

## Appendix — gh-ocannl-487 phase 1: is the portable pipelined rendering correct on HIP?

Scope: correctness of `Stage ~pipeline_depth` at depth > 1 on HIP, plus whether the occupancy cost
phase 1 measured on Metal (~1.4–1.5x) reproduces off Apple silicon. **Not** a verdict on pipelining;
a loss was the stated prediction.

Answers up front: **the rendering is correct on HIP — bitwise depth-2/depth-1 parity holds with
real rocWMMA in the loop — but the occupancy cost does not reproduce: at a precision where HIP's MMA
is real, depth 2 is a measured null (1.01x).** Both answers changed once the emitted source was
checked instead of the label; see the precision note under each.

**Forcing depth 2 needed no scratch change.** `test/operations/schedule_pipelined_matmul` already
builds the depth-2 cooperative Stage through an explicit hand-built schedule, bypassing the seeding
path entirely, so HIP's empty `mma_pipeline_depths` advertisement is irrelevant to it — and it is
backend-generic in the right way (`on_gpu` names `hip`, `read_generated` resolves `.hip`). All 17
golden lines pass on HIP:

- **executed bitwise parity**: both depths compiled, run, and read back — `depth 2 matches depth 1
  BITWISE: true`. Tolerance is used only where it belongs, against the serial twin (`approx`, 1e-2),
  because Tensorize reassociates the tile reduction.

  **At the suite's f32 this does not exercise rocWMMA**, and the distinction is worth stating rather
  than glossing: RDNA3.5 has no f32 operand shape, so the micro-kernel under the staged tiles is the
  lane-0 scalar fallback (`rocwmma::mma_sync = 0`, lane guard present, scalar `fmaf`). The parity
  result is still exactly what it should be — both legs issue the same scalar chain in the same
  order — but what it validates on HIP/f32 is the **rotation, barrier and staged-tile machinery**,
  which is what phase 1 changed, not the tensor-core path.

  Re-run at bf16 the same test renders **real rocWMMA** — `rocwmma::mma_sync = 1`, no lane guard —
  and **`depth 2 matches depth 1 BITWISE` still holds**, which is the strong form of the claim: the
  rotation is a pure prefetch-timing transform with tensor cores actually in the loop. Two of the
  17 lines go false at that precision, and neither is about pipelining: the serial-twin `approx`
  (1e-2) is tighter than bf16's mantissa affords, and the seeding pin probes a synthetic
  **f32**-only capability that bf16 operands cannot match. Both are artifacts of running the test
  off the precision its goldens were written for, so this leg is reported here rather than promoted
  into the suite.
- **rendering**, read out of the emitted `.hip` rather than from the test's substring counts:

  | | depth 1 | depth 2 |
  |---|---|---|
  | shared tiles | `tile_ma[256]`, `tile_mb[512]` | `tile_ma[512]`, `tile_mb[1024]` (doubled) |
  | rotation terms | 0 | 3 — `((i48+1) % 2)` on the prefetch write, `(i48 % 2)` on the compute read |
  | `__syncthreads` | 8 | 5 |

  Prologue loads sit outside the `k_o` loop, in-loop prefetches are `if (i48 < 1)`-guarded, and the
  barrier count drops — the #567 reduction, visible.
- **cc declines stay clean**: the C backends reject the shared staging composition at compile with
  `Invalid_argument … "not supported"` at either depth, and the full `cc` suite is green (exit 0,
  zero diffs).

**The occupancy cost does NOT reproduce here — on HIP it is a null.** `bin/schedule_bench.ml`
already carries an `mma_pd1`/`mma_pd2` pair, but its `has_shared` gate listed only metal and cuda,
so every HIP run fell into the CPU branch and the pair had never been timed on an AMD GPU. Adding
`hip` to that disjunction (a one-line bug fix — HIP renders workgroup-shared placement exactly as
CUDA and Metal do) makes it run.

**The precision matters, and checking it changed the answer.** RDNA3.5 WMMA has no f32 operand
shape, so at the benchmark's default f32 the `Tensorize` renders as the **lane-0 scalar fallback** —
verified in the emitted source, `rocwmma::mma_sync = 0` — and those numbers are not MMA
measurements at all. Re-run at bf16 (`--ocannl_default_prec=bfloat16`), one of HIP's four advertised
formats, the same kernels render `rocwmma::mma_sync = 1` with no lane-0 guard. Matmul 512^3, bf16,
20 repeats, **9 replicates**, identical spot checks (33.0) throughout:

| | median | mean | range | spread |
|---|---:|---:|---:|---:|
| `mma_pd1` | 0.593 ms | 0.592 ms | 0.559–0.653 | 15.9% |
| `mma_pd2` | 0.601 ms | 0.597 ms | 0.547–0.626 | 13.1% |

**pd2 / pd1 = 1.01x on medians, 1.01x on means.** Per-replicate ratios run 0.88–1.11 and the two
ranges overlap almost completely, so the difference is far inside each arm's own ~15% run-to-run
spread: **depth 2 costs nothing measurable on gfx1151.**

That is neither the predicted loss nor a win, and it is stated as the null it is. **The null is
informative rather than merely inconclusive**: an effect the size of Metal's would be ~1.4–1.5x
against a ~15% spread, i.e. three to four times the noise floor, so this cell had ample power to see
one. What it rules out is a device-independent cost, not a cost on Metal.

So the appendix's second question gets a negative answer: the register/LDS staging penalty phase 1
measured on Apple silicon **does not reproduce on gfx1151**, and "the portable form costs ~1.4x" is
not a property of the portable mechanism as such. An earlier revision of this report claimed 1.31x
here and said it corroborated Metal; that number was measured on the **f32 scalar-fallback** kernel
described above — not an MMA measurement at all — and is withdrawn. Nothing here bears on
`cp.async`.

This is a 512^3 matmul cell rather than the suggested `mlp_wide` one — same staged-mma composition
with the pipelining knob, already in the tree, and now at a precision where the MMA is real.

## Reproduction

```bash
cd benchmarks
# pass 1: COLD search. The empty cache dir is load-bearing; it must never be shared with a run at
# another precision policy (ahrefs/ocannl#568).
rm -rf /tmp/gh569-hip-f32
BENCH_FIXTURE=fixtures/gpt2_mini.safetensors BENCH_TUNE=1 taskset -c 0-15 \
  ../_build/default/benchmarks/runners/ocannl/bench_gpt.exe --ocannl_backend=hip \
  --ocannl_autotune_cache_dir=/tmp/gh569-hip-f32 \
  --ocannl_autotune_log=true --ocannl_schedule_log_declines=true
```

```bash
cd benchmarks
# the emitted source + the launch geometry, from a warm replay of exactly that artifact.
# Arm A is written first and arm B overwrites it, so snapshot by polling on content.
rm -rf build_files /tmp/armsnap && mkdir -p /tmp/armsnap
F=build_files/bench_gpt/cross_entropy_loss_fwd__seg.hip
( while :; do if [ -f "$F" ]; then h=$(md5sum "$F" | cut -d' ' -f1)
    [ -f "/tmp/armsnap/$h.hip" ] || cp "$F" "/tmp/armsnap/$h.hip"; fi; done ) & W=$!
BENCH_FIXTURE=fixtures/gpt2_mini.safetensors BENCH_TUNE=1 taskset -c 0-15 \
  ../_build/default/benchmarks/runners/ocannl/bench_gpt.exe --ocannl_backend=hip \
  --ocannl_autotune_cache_dir=/tmp/gh569-hip-f32 \
  --ocannl_output_debug_files_in_build_directory=true \
  --ocannl_schedule_log_launches=true 2> /tmp/launches.err
kill $W
# arm A is the 117-kernel snapshot; require a clean compile too, since the watcher can copy a
# partially written file (hiprtc sources need the runtime header hipcc does not imply).
for f in /tmp/armsnap/*.hip; do
  printf '%s: %s globals -> ' "$f" "$(grep -c '__global__' "$f")"
  hipcc --offload-arch=gfx1151 -O2 -include hip/hip_runtime.h -c -o /dev/null "$f" \
    2>/dev/null && echo COMPILES || echo INCOMPLETE
done
```

```bash
# per-kernel times, and the bucket table
python3 benchmarks/gpt2_kernel_harness.py --source /tmp/armsnap/<armA>.hip \
        --launches /tmp/launches.err --out /tmp/harness.hip
hipcc --offload-arch=gfx1151 -O2 -o /tmp/harness /tmp/harness.hip
taskset -c 0-15 /tmp/harness > /tmp/kernels.csv     # stderr carries the sum-vs-step validation
python3 benchmarks/gpt2_bucket.py --source /tmp/armsnap/<armA>.hip \
        --stats /tmp/kernels.csv --steps 1 --dump
```

```bash
# the prize
hipcc --offload-arch=gfx1151 -O3 -o /tmp/probe benchmarks/ffn1_geometry_probe_hip.cpp && /tmp/probe
# the measured roofline (CPU quiet: the bandwidth leg shares the controller with it)
hipcc --offload-arch=gfx1151 -O3 -o /tmp/roofline benchmarks/roofline_hip.cpp \
      -I/opt/rocm/include -L/opt/rocm/lib -lrocblas && taskset -c 0-15 /tmp/roofline
```

```bash
# the gh-487 appendix
cd _build/default/test/operations && OCANNL_BACKEND=hip ./schedule_pipelined_matmul.exe
grep -c '% 2' build_files/schedule_pipelined_matmul/pipe_mm_d{1,2}.hip
# f32 renders the lane-0 scalar fallback here; bf16 puts real rocWMMA in the loop and the bitwise
# parity line still holds (2 of 17 lines go false for unrelated precision reasons -- see above).
grep -c 'rocwmma::mma_sync' build_files/schedule_pipelined_matmul/pipe_mm_d{1,2}.hip   # f32: 0
OCANNL_BACKEND=hip ./schedule_pipelined_matmul.exe --ocannl_default_prec=bfloat16
grep -c 'rocwmma::mma_sync' build_files/schedule_pipelined_matmul/pipe_mm_d{1,2}.hip   # bf16: 1
# bf16 is load-bearing: at f32 RDNA3.5 has no WMMA operand shape and the Tensorize silently
# renders as the lane-0 scalar fallback, so f32 numbers here are not MMA numbers. Check first.
cd ../../bin && OCANNL_BACKEND=hip ./schedule_bench.exe 512 3 512 512 0 --ocannl_backend=hip \
  --ocannl_default_prec=bfloat16 --ocannl_output_debug_files_in_build_directory=true
grep -c 'rocwmma::mma_sync' build_files/schedule_bench/mm_mma_pd{1,2}.hip   # must be >= 1
grep -c '== 0)' build_files/schedule_bench/mm_mma_pd{1,2}.hip               # must be 0
for r in $(seq 9); do
  OCANNL_BACKEND=hip ./schedule_bench.exe 512 20 512 512 0 --ocannl_backend=hip \
    --ocannl_default_prec=bfloat16 | grep mma_pd
done
```
