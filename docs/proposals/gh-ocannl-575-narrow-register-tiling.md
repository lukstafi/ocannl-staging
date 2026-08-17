# gh-ocannl-575: Tile_mma register tiling for narrow (16-bit) operands

Re-scoped out of gh-ocannl-530 (originally gh-ocannl-516 task 3, gh-ocannl-517's packing remark,
and gh-ocannl-516 task 4). The landed state of the two parents: narrow storage computes in f32 on
the CPU backends through the `C_syntax_config.compute_prec` seam (`Ir.Numerics.narrow_compute_f32`,
default on), with vectorized convert-on-load/store making the `Vectorized` renderings reachable —
but `try_register_tile`, the CPU `Tile_mma` rendering, still gated on f32/f64 *storage* and
declined every 16-bit GEMM to the scalar fallback.

## Design: finish the seam, don't special-case it

`try_register_tile` was the one explicit-SIMD rendering that never went through the gh-517
storage/compute split — `try_vectorize` and `try_vectorize_reduce` both key their lane geometry
off `comp_prec` and bridge at the memory boundary. The change makes the register tiling do exactly
the same, uniformly:

1. **Compute-precision-keyed gates.** The precision gate becomes
   `B.vector_prec_ok (comp_prec d_storage)` — f32/f64 everywhere, fp16 where arithmetic is native
   — and the uniformity gate compares the three operands' *compute* precisions (their storages may
   differ: a narrow `d` over f32-packed panels is uniform-f32 compute). Lane count, `vec_ext_typ`,
   and the accumulator registers all follow the compute precision.
2. **Memory-boundary bridges.** The B-row vector loads and the C-tile moves go through the
   existing `vec_bridge` (identity memcpy when storage = compute, so the f32 emission is unchanged
   modulo one whitespace split); the A splats and the scalar peel convert through the same
   `convert_precision` spellings the scalar fallback renders with. fp8 storage rides the bridge's
   per-lane fallback arm for free.
3. **Accumulator residency.** The C-tile accumulates at compute precision across the statement's
   whole k extent and narrows once at the store — the same once-per-cell narrowing as
   `try_vectorize_reduce`'s epilogue and the CUDA bf16 `mma.sync` arm (gh-ocannl-545): strictly
   better rounding than the fallback's per-k-step narrowing, never bitwise on inexact values, so
   parity tests pick narrow-exact inputs. Where storage = compute (pure-fp16 included) the
   rendering stays BITWISE equal to the fallback, per-element fused chains in serial k order.
4. **Pure-fp16 tiling** (f16 accumulators, doubled lanes) is not a separate mode: it is what the
   seam resolves to when `Ir.Numerics.fp16_arithmetic` is on AND the target's arithmetic is
   genuinely native (`hardware_limits.native_fp16_arithmetic`; on merely-promoted targets the
   policy is deliberately ignored). The scalar peel's FMA gained the `OCANNL_HALF_FMA` arm so peel
   and vector body round identically (the gh-516 single-vs-double-rounding trap).

### The packing seam: `Stage { tile_prec }`

gh-517's "free lunch": fold the widening into the GEBP packing copy so the conversion happens once
per element at pack time instead of once per read inside the micro-kernel. `Schedule.Stage` gained
`tile_prec : Ops.prec option` — mint the staged tile at this storage precision instead of the
source's. Only exact widenings are accepted (equal, or narrow-float source to f32/f64): a widening
is value-preserving and its round-trip is the identity, so the transform is unconditionally
numerics-preserving under either policy setting, and the packing copy's cross-precision `Set`
renders the conversion through the ordinary `Get` boundary. The hoisted (link-time, constant-pool)
packing converts on the host in `pack_constant_tile` via the exact `get_as_float`/`set_from_float`
round-trip; the same-precision path keeps its raw-copy fast path byte-for-byte.

### Seeding moves with emission (the gh-545 lesson)

The compute-precision resolution now has a single source of truth, `Numerics.cpu_compute_prec
~native_fp16_arithmetic`, shared verbatim by `Cc_backend.compute_prec` (emission) and the autotune
sketch pre-filters (seeding) — so a candidate is never timed under a tensorized label its
rendering must decline. Concretely, in `sketch_families.ml`:

- the matmul and conv CPU branches replace `uniform_f32_64` with compute-precision uniformity +
  vector-capability (fp16 requires `limits.native_fp16_arithmetic`), and key lane counts and the
  packed-tile footprint caps off the compute element size (an f32 panel over fp16 storage is twice
  the storage bytes; a pure-fp16 panel is half the f32 one);
- `sketch_params` gained `sk_pack_prec : Ops.prec option` — the site's compute precision, recorded
  at seed time (schedule construction has no `hardware_limits`) and threaded into the packing
  `Stage`s' `tile_prec`, normalized to `None` per operand already storing at it. Candidate labels
  carry it as ` pack<prec>`.

Pinned by `test/operations/sketch_family_tree.ml`'s half-precision scenarios: default policy seeds
11 candidates with `packprec:single`; `narrow_compute_f32=false` refutes the tensorized branch;
native-fp16 limits + the `fp16_arithmetic` policy seed pure-f16 (no pack precision, doubled
lanes); the policy on a merely-promoted target stays f32-compute.

### Tile width follows the extent, not the register cap

The C-tile is `rm` rows of `rn` vectors, and `rn` is chosen against the *actual* column extent rather
than pinned at the register-pressure cap. The columns `bw = rn * lanes` does not cover are peeled to
the scalar fallback, and a peeled column costs roughly a whole vector slot — so a cap that leaves a
fat remainder loses far more than the extra A-reuse it buys. This is invisible at f32 lane counts and
brutal at doubled ones: at n = 512 a pure-fp16 `bw = 48` peels 32 columns, 6.25% of the work at
scalar speed, and that alone turned the NEON measurement below upside down (37 vs 133 GFLOP/s).
Power-of-two extents — what deep learning actually runs — are never multiples of 48.

The ranking model: per unit of `m*k`, a tile pass issues one vector FMA per lane-column plus the B
row loads (1/`rm` per FMA) and the A splats (1/`rn`), while each peeled column costs a flat 10
lane-slots. The peel weight deliberately does *not* scale with the lane count — the peel loop is the
same scalar code at either width — and the fits agree: ~8 from the 8-lane sweep below, ~10 from the
4-lane one, ~20 from the n = 2048 pair. It only has to *rank* candidates, not predict times; it
reproduces the measured order at n = 512 within a few percent across `rn = 2..6`, and where several
widths divide the extent evenly it lands on the largest affordable one. Measured on the M4 Max (10
repeats, GFLOP/s, `packmma`):

| rn (bw at 4/8 lanes) | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|
| pure-f16, n=512 | 92.0 | 78.7 | **132.6** | 36.9 | 36.9 |
| pure-f16, n=1024 | 136.0 | 96.7 | **203.4** | 89.0 | 111.2 |
| f16→f32, n=512 | 55.4 | 47.5 | **75.3** | 48.3 | 55.7 |
| f16→f32, n=1024 | 66.4 | 65.9 | **100.9** | 80.6 | 62.0 |

The cap itself was never the problem: isolated at extents both widths divide (n = 576, 768),
`rn = 4` and `rn = 6` land within ±10% of each other. The peel is the whole story. The fixed cap had
been the geometry since gh-ocannl-469, so this lifts the f32 register tiling too (n = 512: 55.9 →
74.3 GFLOP/s standalone) — 16-bit operands are what made it visible, not what made it true.

Erring *low* on the peel weight is what costs choices, and it costs them quietly: at n = 2048 a
`lanes`-scaled weight scored the peeling `rn = 6` just under the peel-free `rn = 4`, which measures
1.15x faster (98.4 vs 85.5 GFLOP/s at f32) — the same cliff, one notch smaller. Extents whose only
peel-free widths are narrow are where the model earns its keep, so it must not be tuned to the
extents that were easy to measure.

### Cost model

Nothing to change structurally: per-node footprint widths already come off each node's own
`storage_prec` (so narrow nodes and f32-minted packed tiles are both exact), and FLOP counts are
precision-blind. The one implicit width assumption — `hardware_limits.peak_flops` being a
single-precision ceiling that a native-fp16 kernel doubles — is documented on the field; it cannot
reorder a site's candidates because they all share one policy-resolved compute precision.

## Measurement, x86 (ROG: Core Ultra 9 275HX, WSL2, gcc 15.2, cc backend at -O3, n = 512)

`bin/narrow_gebp_bench.exe` (naive serial vs packed GEBP serial vs grid-outermost per-chunk
variant; exact inputs; readbacks outside the timed region — re-measured after the Codex round-1
fix moved the spot-check readback out of the timed interval). Recorded at the fixed-cap tile
geometry, i.e. before the extent-adapted width above, so every row is a floor rather than a ceiling:
the three f32-compute rows share a 24-wide tile peeling 8 of 512 columns, and the forced pure-f16
row a 48-wide one peeling 32 — see the caveat under the negative control:

| storage → compute | naive | packmma | packmma_par |
|---|---|---|---|
| f32 → f32 | 21.5 GFLOP/s | 74.1 | 55.7 |
| bf16 → f32 (widened panels) | 0.38 | **51.3 (135x)** | 34.3 |
| f16 → f32 (widened panels) | 0.65 | 3.4 (see below) | **37.1 (57x)** |
| f16 → f16 forced on promoted HW | 0.65 | 2.1 | 2.0 |

- The headline: **f32-GEBP-over-narrow-storage is a ~57-135x win over the scalar narrow
  rendering** on x86, and lands within ~30% of the f32 GEBP — the storage seam pays for itself.
- The f16 `packmma` anomaly is a **gcc-15 `-O3` pessimization**, not a rendering defect: on the
  k_o-outermost serial shape with the f16 d-bridge, `-O3` unrolls the k-loop 4x and spills the
  accumulator C-tile (hot loop: 29 insns / 2 stack refs at `-O2` vs 375 insns / 147 stack refs at
  `-O3`); the identical statement compiled at `-O2`, by clang (whose
  `__builtin_elementwise_fma` arm avoids per-lane subscripts), or in the fragment-contracted
  i_o-outermost shape (`packmma_par`'s) runs at full speed. f32 and bf16 barely move between
  levels (63.9 vs ~67 GFLOP/s standalone). The tuner routes around it by measurement; a
  `cc_backend_optimization_level=2` run is the manual workaround on gcc.
- The forced pure-f16 row is the negative control for the gating: computing in f16 where the
  compiler promotes is a ~18x loss against f32-compute — exactly why `fp16_arithmetic` is ignored
  off-native and why the pure-f16 seeds fire only where the probe reports native. **The 18x is
  not all promotion**: forcing native doubles the lane count, so this row alone ran a 48-wide tile
  peeling 32 columns, and the NEON sweep above puts a comparable geometry at ~3.6x of its own loss.
  A clean re-measure at the adapted width is owed on that box. The direction is not in doubt — the
  promoted arm does f32 arithmetic with extra per-value conversions, so it cannot come out ahead of
  computing in f32 directly — but the magnitude is inflated.

## The NEON decision: pure-f16 wins (M4 Max, Apple clang, `-O3 -mcpu=native`, cc)

The issue's reserved decision point — pure-f16 GEBP vs f32-GEBP-over-narrow-storage on a target
whose fp16 arithmetic is genuinely native (the probe resolves ARMv8.2-FP16 to `Native` here).
Sustained `packmma` throughput, medians of seven back-to-back runs at 20 repeats (single-threaded,
so a cold machine's first run turbos ~40% above the sustained figure — the numbers below are the
warm ones):

| storage → compute | naive | packmma, n=512 | packmma, n=1024 |
|---|---|---|---|
| f32 → f32 | 3.02 | 83.2 | 99.8 |
| bf16 → f32 (widened panels) | 0.48 | 82.9 | 102.5 |
| f16 → f32 (widened panels) | 1.03 | 82.9 | 101.6 |
| f16 → f16, native arithmetic | 3.06 | **132.5 (1.60x)** | **181.6 (1.79x)** |

**Pure-f16 GEBP is worth having: 1.6-1.8x over f32-GEBP-over-narrow-storage**, close to the 2x the
doubled lane count allows, and the gap grows with n. The emission is what the design intends — both
arms resolve to a 4x4 C-tile here, and the k-loop body of each is 16 vector FMAs in the by-element
form (`fmla.8h` against `fmla.4s`) plus its B loads and A splats, with no spills: the same
instruction count over twice the lanes. So the seeds this issue gates on
`native_fp16_arithmetic` earn their place, and the gating is exactly right in both directions: the
same policy forced on promoted x86 hardware loses ~18x (the negative control above).

The measurement is only true at the adapted tile width. At the fixed cap, the same comparison at
n = 512 read 37.0 vs 62.1 GFLOP/s — pure-f16 *losing* 1.7x — entirely because a 48-wide tile peels
32 of 512 columns to scalar code. The honest first measurement therefore answered a second question
it was not asked, and the answer changed the first one; see the tile-width section above.

The seam's x86 verdict — narrow storage pays for itself, landing ~30% off the f32 GEBP — holds on
ARM with the gap closed entirely: the widened-panel arms match plain f32 at n = 512 and edge past it
at n = 1024, because the packing copy reads half the bytes for the same micro-kernel.

## Executed coverage

- `test/operations/tile_mma_narrow.ml`: bf16 in-kernel widened packing (f32 panels,
  `narrow storage bridged: d:bfloat16` only), half hoisted host-converting pack, and a pure-f16
  leg (probe forced native via `OCANNL_CC_FP16_ARITHMETIC`; per-op rounding is unchanged under
  promotion, so parity stays bitwise) — all bitwise against serial twins, with emission pins.
- `test/operations/schedule_mma_matmul.ml`: the half/bf16/fp8 whole-triple legs now register-tile
  on the C backends (bitwise, narrow-exact inputs); transposed-B legs still pin the decline; an
  8x40 leg pins the extent-adapted width (`full blocks 8x40 of 8x40` — peel-free at NEON's 4 f32
  lanes and at AVX2's 8, where the old cap took 24 on both), bitwise against its serial twin.
- `test/operations/sketch_family_tree.ml`: the seeding scenarios above.

## Relations

- [tensorize-mma](tensorize-mma.md): the CPU register tiling this extends (gh-ocannl-469), and the
  gh-545 accumulator-precision precedent the residency rule follows.
- gh-ocannl-516/517: the parent issues; their landed seams are the substrate.
- [gh-ocannl-530-pool-uniformity](gh-ocannl-530-pool-uniformity.md): supplies
  `native_fp16_arithmetic` and the uniform pool the seeds tune for.
