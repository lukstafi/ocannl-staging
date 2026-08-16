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

### Cost model

Nothing to change structurally: per-node footprint widths already come off each node's own
`storage_prec` (so narrow nodes and f32-minted packed tiles are both exact), and FLOP counts are
precision-blind. The one implicit width assumption — `hardware_limits.peak_flops` being a
single-precision ceiling that a native-fp16 kernel doubles — is documented on the field; it cannot
reorder a site's candidates because they all share one policy-resolved compute precision.

## Measurement (ROG: Core Ultra 9 275HX, WSL2, gcc 15.2, cc backend at -O3, n = 512)

`bin/narrow_gebp_bench.exe` (naive serial vs packed GEBP serial vs grid-outermost per-chunk
variant; exact inputs; readbacks outside the timed region — re-measured after the Codex round-1
fix moved the spot-check readback out of the timed interval):

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
  off-native and why the pure-f16 seeds fire only where the probe reports native.

**Still pending: the honest NEON measurement.** The issue's decision point — pure-f16 GEBP vs
f32-GEBP-over-narrow-storage on a native-arithmetic target (Apple Silicon / ARMv8.2-FP16) — needs
hardware this machine doesn't have. Run `bin/narrow_gebp_bench.exe f16 512 20
--ocannl_fp16_arithmetic=true` there (the probe resolves native automatically; compare against the
same command without the policy). If pure-f16 loses there too, the recorded outcome should be
"not worth seeding beyond the seam" and the `fp16_arithmetic` gating stays as the only extra
surface.

## Executed coverage

- `test/operations/tile_mma_narrow.ml`: bf16 in-kernel widened packing (f32 panels,
  `narrow storage bridged: d:bfloat16` only), half hoisted host-converting pack, and a pure-f16
  leg (probe forced native via `OCANNL_CC_FP16_ARITHMETIC`; per-op rounding is unchanged under
  promotion, so parity stays bitwise) — all bitwise against serial twins, with emission pins.
- `test/operations/schedule_mma_matmul.ml`: the half/bf16/fp8 whole-triple legs now register-tile
  on the C backends (bitwise, narrow-exact inputs); transposed-B legs still pin the decline.
- `test/operations/sketch_family_tree.ml`: the seeding scenarios above.

## Relations

- [tensorize-mma](tensorize-mma.md): the CPU register tiling this extends (gh-ocannl-469), and the
  gh-545 accumulator-precision precedent the residency rule follows.
- gh-ocannl-516/517: the parent issues; their landed seams are the substrate.
- [gh-ocannl-530-pool-uniformity](gh-ocannl-530-pool-uniformity.md): supplies
  `native_fp16_arithmetic` and the uniform pool the seeds tune for.
