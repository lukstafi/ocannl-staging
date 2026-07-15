# Tensorize: emitting tensor-core (MMA) instructions

Follow-up to [schedule-ir-optops](schedule-ir-optops.md) (§5 `Stage`, §6 presets) and
[gh-ocannl-412](gh-ocannl-412.md): after shared-memory staging and register tiling, the
remaining hardware tier for matmul-class kernels is the matrix-multiply-accumulate
units — CUDA tensor cores (`wmma` / `mma.sync`, 16×16×16 and smaller shapes) and Apple
`simdgroup_matrix` (8×8). This proposal pins down what emitting them requires.

## Goal

A schedule-level `Tensorize` transform that replaces the innermost matmul micro-kernel
of an already-tiled loop nest with a cooperative tile-MMA instruction, on backends that
have one — Metal first (`simdgroup_float8x8` etc.), CUDA second (wmma via nvrtc) — with
every other backend rendering a semantically-equivalent serial fallback. Composition
target, extending the §6 matmul preset:

```
Split i (BM) → Split j (BN) → Split k (BK) → Retype Grid/Workgroup
→ Stage A ~shared → Stage B ~shared
→ Tensorize { m; n; k }        (* replaces: Split TM/TN + Privatize + materializing Unroll *)
```

Each prefix stays independently runnable and parity-testable, preserving the property
the eventual BEAM search (§8) needs.

## The core mismatch

`Low_level.t` has strictly per-thread scalar semantics: thread identity is the tuple of
annotated-loop index values, and `validate_parallel` reasons about which elements each
thread writes. An MMA instruction is **cooperative**: a warp/simdgroup of 32 threads
jointly holds tile fragments and computes `D = A·B + C`; the per-lane ownership of
fragment elements is architecture-defined and deliberately opaque. No composition of
scalar `Set`s can express this, so tensor cores need the IR's second non-scalar
statement form (after `Set_from_vec`, which is the single-thread precedent for a
statement writing multiple elements).

## Design

### 1. The `Tile_mma` statement

```ocaml
| Tile_mma of {
    d : Tnode.t * Indexing.axis_index array;   (* accumulator tile base *)
    a : Tnode.t * Indexing.axis_index array;
    b : Tnode.t * Indexing.axis_index array;
    m : int; n : int; k : int;                 (* intrinsic shape, e.g. 8/8/8 MSL *)
    lane : Indexing.symbol;                    (* the cooperating Workgroup axis *)
    fallback : t;                              (* equivalent scalar micro-kernel *)
  }
```

Semantics: the `m*n*k` fused multiply-adds of one tile step, `d[i,j] += Σ_k a[i,k] *
b[k,j]` relative to the base index vectors, executed cooperatively by the threads of
the `lane` axis. Operand base indices must not mention `lane` (the tile is jointly
owned); strides come from the operand tnodes' dims. `a`/`b` may live in
workgroup-shared or device memory (both are loadable by `simdgroup_load` and
`wmma::load_matrix_sync`); `d` is a per-simdgroup accumulator — in v1 a fragment
held across the serial `k_o` loop, stored back once (see §3).

The `fallback` field follows the `Vectorized` precedent (pragmas where supported, a
plain serial loop elsewhere): it carries the scalar triple-loop over fresh serial
symbols. Backends without an MMA hook render it; capable backends ignore it and emit
the intrinsic. This keeps cc correct with zero backend work, gives parity tests their
twin, and — unlike an opaque `Staged_compilation` — keeps the statement transparent to
`validate_parallel` and the access analyses (the fallback *is* the access-set
description, and tests can check the descriptor against it).

**Lane bookkeeping.** After tensorization the per-lane work no longer exists as loops,
but the launch still needs those 32 threads. Keep a real `Workgroup`-typed loop of
extent `simd_width` (32) enclosing the `Tile_mma` statements, binding `lane`; its index
is not mentioned inside (uniform execution). Hardware backends bind it as usual, so
`launch_dims` keeps the ×32 factor and all lanes reach the intrinsic together. Serial
renderers guard the fallback with `if (lane == 0)` so the tile computes once per
simdgroup, not 32× — the renderer's obligation, keyed off the `lane` field, mirroring
how pool-rendered `Grid` loops already special-case kernel-scope locals.

### 2. Validation: barrier-strength uniformity

`Tile_mma` validates exactly like `Workgroup_barrier` — the checks already exist:

- must not sit under divergent control flow (no lexically-enclosing `If`; same-slot
  workgroup extents uniform). This composes with an existing constraint instead of
  adding one: `guard_annotated_extents` wraps non-dividing extents in `If` guards, and
  §5's shared `Stage` already requires dividing tile sizes for its barriers — so
  tensorized schedules inherit "tile sizes divide extents" wholesale.
- `Split` remainder guards inside the micro-kernel are likewise rejected; v1 requires
  the MMA shape to divide the staged-tile sizes (`BM % m = 0` etc.), checked by the
  transform at construction.
- write coverage: the tile store covers the `lane` slot *by decree* — the fragment
  layout is architecture-opaque, which is precisely why the statement exists. The other
  active slots must be covered by enclosing annotated loops as usual (the `d` base
  indices mention them; same rule as scalar `Set`s).
- `Workgroup_reduce` is the precedent for a semantically-annotated hardware axis; if a
  distinguished flavor for the lane axis proves useful for transforms (e.g. `Stage`
  reusing it as a cooperative-load index), add `Workgroup_lane` the same way. v1 tries
  plain `Workgroup` first.

### 3. The `Tensorize` optop

```ocaml
| Tensorize of { m : int; n : int; k : int; lane : Indexing.symbol; (* fresh *) }
```

Applied after `Split`s and `Stage`s in the matmul preset, `apply_tensorize`:

1. Locates the micro-kernel: the innermost serial `i_i × j_i × k_i` nest whose body is
   a single accumulation `Set { tn = d; llsc = ... Add/Fma (Get a) (Get b) ... }` — the
   same v1 pattern discipline as `Stage` ("all reads use one index vector", fail loudly
   otherwise). The preset built this nest, so recognition is a structural check, not
   inference.
2. Checks divisibility (`extent(i_i) % m = 0` etc.), operand-precision support against
   the backend capability (see §6), and that `d`'s accumulation is a plain add (no
   exotic accumulators in v1; `@^+`-style tropical kernels stay scalar).
3. Replaces the nest with `m/n/k`-strided serial loops over tile steps, each body a
   `Tile_mma` (constructing the `fallback` from the original body by index
   substitution), wrapped in the fresh extent-32 `lane` loop.
4. Accumulator residency: v1 keeps `d` in a fragment across the serial `k_o` loop —
   which requires the `Tile_mma`s of successive `k_o` iterations to target the same
   accumulator. Express this as `Privatize`-style contraction: `Tensorize` runs where
   `Privatize { target = d; over = k_o }` would, producing a per-simdgroup accumulator
   tile (a `Local` node of dims `[|m; n|]` marked with a new `simdgroup_fragment` set on
   `optimized`, sibling to `workgroup_shared`) initialized before the loop, accumulated
   in place, stored back after. Backends map that node to a fragment variable instead
   of an array; the fallback maps it to a stack array — the existing `Local` rendering.

### 4. Backend hooks — the established functor-config pattern

`C_syntax.B` already grew `vectorize_pragma`, `parallel_grid_syntax`,
`shared_decl_prefix`, `hardware_index`; add:

```ocaml
val mma_syntax :
  (shape:int * int * int -> a_prec:Ops.prec -> b_prec:Ops.prec -> c_prec:Ops.prec ->
   ... (* fragment decl / load / mma / store emission *)) option
```

`None` (cc, and any backend until wired) renders the fallback under the lane guard.

- **Metal (first, per the landing order and local testability)**: `simdgroup_float8x8`,
  `simdgroup_half8x8`, `simdgroup_bfloat8x8` (MSL 3.1 is already selected for bfloat);
  `simdgroup_load`/`simdgroup_store` take a source pointer + elements-per-row stride
  and read directly from `threadgroup` memory — i.e. straight out of `Stage`'s shared
  tiles — or from `device` memory; `simdgroup_multiply_accumulate(d, a, b, d)` is the
  step. Only 8×8×8, which just means the tile-step loops of §3.3 have more iterations.
- **CUDA**: the wmma C++ API (`#include <mma.h>`, `nvcuda::wmma::fragment`,
  `load_matrix_sync` / `mma_sync` / `store_matrix_sync`) compiles under nvrtc given an
  `sm_70+` arch flag; 16×16×16 for f16/bf16→f32, plus tf32 shapes on sm_80+. Raw
  `mma.sync` + `ldmatrix` PTX with swizzled shared-memory layouts is the later
  perf-ceiling chase (T4), not the entry point.
- **cc**: fallback only. AMX/SME is out of scope; the explicit-SIMD `Vectorized`
  rendering (gh-ocannl-164, PR #99) is the CPU story and its open matmul micro-kernel
  item is where a CPU tile step would land.

### 5. Precision and layout plumbing

- Mixed-precision accumulation (f16×f16→f32, bf16→f32) is already representable:
  `scalar_arg` carries per-operand precision, and the `Tile_mma` descriptor reads the
  operand tnodes' precs. The transform checks the triple against the backend's
  supported combinations and declines otherwise (fallback or plain register tiling).
- Fragment loads take a leading-dimension stride; `Stage` mints the shared tiles, so a
  `~pad_stride` option there (pad the minor dimension to avoid bank conflicts / satisfy
  alignment) is the natural follow-up knob. v1 uses exact dims — correct, possibly
  bank-conflicted.
- `check_hardware_limits` already budgets `workgroup_shared` bytes; fragments live in
  the register file and need no new budget in v1.

### 6. Capability gating

Extend `Backend_intf.hardware_limits` (the seam added for block-size clamping) with an
MMA capability descriptor — supported `(shape, a_prec, b_prec, c_prec)` combinations
and `simd_width` — populated per backend (`None`s for cc), so `Tensorize` and the
preset can gate without touching drivers at module init (same laziness discipline as
`hardware_limits` itself).

## Rejected alternative

Calling MPS / cuBLAS for matmuls would ship faster but forfeits fusion, non-standard
einsums, mixed layouts, and the schedule-IR direction generally; the wmma/simdgroup
APIs are small targets and the staging/tiling groundwork is already landed. Library
calls remain available as a separate escape hatch if ever needed; they are not this
proposal.

## Phasing

- **T1 — IR + Metal**: `Tile_mma` (with `fallback` and `simdgroup_fragment`),
  barrier-style validation, `mma_syntax` hook, Metal `simdgroup` emission. Exercised by
  a hand-written schedule (no preset changes) on an f32 matmul against the unscheduled
  parity twin — the `schedule_ops` harness pattern; cc runs the fallback and must match
  bitwise.
- **T2 — `Tensorize` optop + preset**: pattern matching over the canonical
  Split/Stage form, accumulator contraction, the tensorized matmul preset variant
  behind a config key (e.g. `matmul_use_mma`); benchmark vs. the S2 SMEM matmul on
  M-series (f32 and f16).
- **T3 — CUDA wmma**: 16×16×16 f16→f32 and bf16→f32 via nvrtc; capability gating by SM
  arch; CI parity (local development stays Metal + cc fallback, per the backend
  ordering).
- **T4 (optional) — the ceiling**: raw `mma.sync`/`ldmatrix`, swizzled staging layouts
  (`Stage ~pad_stride` and beyond), double-buffered tiles; driven by benchmarks, not
  speculation.

## Acceptance criteria

- [ ] `Tile_mma` round-trips through `validate_parallel` with the barrier-uniformity
      rules; a `Tile_mma` under an `If` guard or with non-dividing extents is rejected
      with a targeted error.
- [ ] Metal `simdgroup` matmul matches the unscheduled twin (f32; f16 within
      tolerance), and the cc fallback matches bitwise.
- [ ] The tensorized preset beats the S2 shared-memory matmul on M-series by a
      meaningful multiple for f16 (measure before promising a number).
- [ ] CUDA wmma parity in CI on sm_70+.
- [ ] Every schedule prefix without `Tensorize` behaves exactly as today (the op is
      purely additive).

## As landed (2026-07-06, T1+T2 core)

The implementation adjusted the design in a few places; the differences are recorded here rather
than rewritten into the sections above.

- **Block semantics for `Tile_mma`.** `m`/`n`/`k` are the *covered block extents* (multiples of
  the backend's intrinsic tile), not the intrinsic shape: one statement is the whole
  `d[0..m)×[0..n) += a·b` block over `k` reduction steps. Fragment residency across the reduction
  thereby becomes an intra-statement codegen concern — the Metal emission declares the
  `(m/8)×(n/8)` accumulator fragment array, loads `d` once, loops `k/8` mma steps, stores once —
  so the `simdgroup_fragment` node marking and the `Privatize`-style accumulator contraction of
  §3.4 were not needed. The proposal's per-intrinsic-step statement plus fragment-marked `Local`
  node remains available as a refinement if cross-statement residency (e.g. across a staged `k_o`
  loop) is ever required.
- **`Tensorize` names loops, not shapes**: `Tensorize { i; j; k; lane; simd_width }` (the
  `Schedule.tensorize` helper mints `lane`). It requires the perfectly nested serial `i×j×k`
  micro-kernel with the single accumulation body (`d[...,i,j] += a[...,i,k] * b[...,k,j]`,
  plain-add or FMA form as `optimize`'s simplify leaves it; unit coefficients on the last two
  axes; since 2026-07-15 transposed layouts are accepted — the recognition infers per-operand
  orientation ([a] as [..., i, k] or [..., k, i], [b] as [..., k, j] or [..., j, k]) and records
  it as `Tile_mma.ta`/`tb`, which Metal renders via [simdgroup_load]'s [transpose_matrix] flag
  and the CUDA draft as wmma [col_major] fragments, both with swapped tile-offset arithmetic;
  the cc register tiling declines transposed layouts to the scalar fallback. This covers the
  gradient GEMMs ([dA = g·Bᵀ], [dB = Aᵀ·g]) without operand copies. Divisibility by the
  intrinsic tile is *not* checked by the
  transform — the schedule layer is backend-agnostic; `mma_syntax` declines per call and the
  fallback runs, so the op is always semantics-preserving.
- **The `mma_syntax` hook** receives the three operand precisions (`d_prec`/`a_prec`/`b_prec` —
  the backend decides which combinations its units support: Metal declines mixed, CUDA wmma's
  flagship is f16×f16→f32) and, per operand: a pointer expression to the tile base (already
  offset), the leading-dimension stride in elements, and the address space
  (`` `Device | `Shared | `Thread ``); it returns `PPrint.document option`, `None` declining that
  call (unsupported combination, non-multiple extents, stride constraints, thread-space
  operand).
- **Barrier bracketing.** Sibling statements (the zeroing of `d`, staged loads) execute
  lane-partitioned, so both the emitted intrinsic block and the `if (lane == 0)` fallback are
  bracketed in `threadgroup_barrier`s on backends that bind the lane in hardware. Relatedly,
  `Tile_mma` *is* a barrier for validation and code motion (`contains_barrier` returns true), so
  the workgroup-extent-uniformity and no-`If`-guard rules apply wholesale, as §2 intended.
- **Capability gating**: `Backend_intf.hardware_limits` grew
  `mma : { mma_simd_width; mma_tile } option` (Metal: 32, 8×8×8, gated on the Apple7 GPU family);
  supported precisions live in the emission, not the descriptor.
- **Composition with shared `Stage`** initially deferred (Stage runs before Tensorize and could
  not see the future lane loop, so its cooperative loads would have raced along the lane axis);
  since 2026-07-07 the lane-aware `cooperative` mode implements the composition — see the
  dedicated section below. Plain (unstaged) tensorized schedules remain valid and simpler: one
  statement spans the full reduction extent, amortizing `d` traffic entirely.
- **Parity nuance**: the C-backend fallback matches the serial twin *bitwise* (same operation
  order); the Metal simdgroup path matches within f32 tolerance — the tile reduction
  reassociates, so "cc matches bitwise" is the exact criterion and GPU parity is tolerance-based.
- Exercised by `test/operations/schedule_mma_matmul.ml` (serial twin vs. tensorized schedule; an
  f32 variant and an f16 variant whose inputs make half arithmetic exact — so parity is bitwise on
  every backend and rendering path, `simdgroup_half8x8` included; structural checks per backend;
  pattern-discipline error).
- **T3 status (2026-07-07): drafted, unverified.** `cuda_backend.ml` carries a wmma `mma_syntax`
  (16×16×16 fragment blocks mirroring the Metal emission; f16×f16→f32, f16×f16→f16, and
  bf16×bf16→f32 on sm_80+, the rest declined — uniform f32 could target tf32 shapes but truncates
  the mantissa, left on the scalar path until the numerics policy is decided). `cuda_to_ptx`
  injects `#include <mma.h>` and a `--gpu-architecture` option only into kernels whose source uses
  wmma, so everything else compiles exactly as before; `hardware_limits.mma` is populated from the
  minimum device compute capability (≥ 7.0). There is no CUDA device in local development (the
  Metal-first ordering), so this is compile-checked only: the f16 leg of `schedule_mma_matmul`
  accepts either the wmma or the fallback rendering structurally, while its bitwise parity check
  is the verification signal for CI. Known draft risks: nvrtc's toolkit-header resolution for
  `mma.h`, wmma's 256-bit pointer-alignment requirement for `__shared__` tiles (device pools are
  alignment-safe), and the exact stride-multiple rules per element type.

## T3 connection fixes + the fp8 inline-PTX path (2026-07-15)

Prompted by the RTX 5070 (sm_120 Blackwell GeForce, driver CUDA 13.0) machine. Two classes of
change, both in `cuda_backend.ml`:

- **The arch-flag disconnect.** CUDA 13 removed offline compilation below `compute_75` (Maxwell
  through Volta), so nvrtc 13 rejects the draft's `--gpu-architecture=compute_70` (f16 wmma) and
  `compute_53` (scalar half arithmetic) outright — on a CUDA-13 toolchain every half-precision
  kernel failed to compile, tensor cores or not. Fix: a triggered arch floor is raised to
  `max(floor, min(75, min-device-CC))`, where `min_compute_capability` (formerly `..._major`) now
  returns `major*10 + minor` minimized across devices. A device below sm_75 keeps the literal
  floor (it must be paired with an nvrtc 12.x that still accepts it); everything newer compiles at
  a CUDA-13-acceptable arch. We deliberately do *not* raise to the device arch: PTX targeted at a
  floor arch is forward-JIT-compiled by the driver on every later GPU, whereas targeting e.g.
  `compute_120` invalidates the plain sm_89 fp8 `mma.sync` encoding in favor of Blackwell's
  family-specific `kind::f8f6f4` forms.
- **fp8 tensor cores via inline PTX.** wmma has no fp8 element type, so `mma_syntax` gained a
  second arm: fp8(e5m2) × fp8 → f32 emits `mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32`
  as inline asm with the architecture-defined per-lane fragment layouts (PTX ISA "Matrix
  Fragments for mma.m16n8k32"; fp8 shares the `.s8`/`.u8` layouts) — per-lane byte gathers into
  `.b32` registers, so there are no stride/alignment constraints and both `__shared__` tiles and
  device pointers load through generic addresses. Gated on min device CC ≥ 89 (Ada) and extents
  `m%16 = n%8 = k%32 = 0`; the `(mma-fp8)` source marker makes `cuda_to_ptx` target `compute_89`.
  The `hardware_limits.mma` advisory tile stays at the wmma 16×16×16 (it is only the autotuner's
  divisibility filter; a k-multiple-of-16-but-not-32 fp8 proposal declines at emission into the
  fallback, which is always correct). Uniform-f32-via-tf32 remains parked on the numerics policy.
- Exercised by a new fp8 leg of `schedule_mma_matmul` (e5m2-exact inputs, f32 accumulation —
  parity bitwise on every backend and path), alongside the existing half leg.
- **Verified on hardware (2026-07-15, RTX 5070 / sm_120, driver CUDA 13.0, toolkit 13.3)**: the
  f16 leg emits `nvcuda::wmma` (PTX `.target sm_75`) and the fp8 leg the inline
  `mma.sync...e5m2` (`.target sm_89`); both match their serial twins BITWISE, confirming the
  per-lane m16n8k32 fragment layouts. Two traps uncovered by the run:
  - The test schedule originally split rows by `bm = 8`, sized for Metal's 8×8×8 tile — every
    CUDA emission declined on `m = 8` and silently took the (correct) scalar fallback, so
    "parity passed" said nothing about tensor cores. `bm = 16` exercises the intrinsics on both
    backends; when a GPU leg unexpectedly shows the `== 0)` lane guard, suspect a declined
    emission, not a codegen bug.
  - Toolkit-newer-than-driver skew (nvrtc 13.3 emits PTX ISA 9.3; a CUDA 13.0 driver rejects it
    with `CUDA_ERROR_UNSUPPORTED_PTX_VERSION` for *every* kernel): handled in cudajit's
    `Module.load_data_ex`, which retries with the PTX `.version` header downgraded in place.
  The same session fixed cudajit for CUDA 13 on Windows/mingw (cuCtxCreate_v4 arity shim,
  nvrtcCompileProgram constness shim, cuda.lib's MSVC-only /DEFAULTLIB directives satisfied by
  empty archives + `_fltused`, and the `cmd /C .\` bat-invocation fix) — see the local
  ocaml-cudajit checkout; the OCANNL `pin-depends` for cudajit should be bumped once those land
  upstream.

Possible follow-ups, in profile-completeness order: e4m3 if/when OCANNL grows the second fp8
format; tf32 for uniform f32 behind a numerics-policy config; `ldmatrix`-based fragment loads
from swizzled shared tiles and Blackwell's block-scaled `kind::mxf8f6f4` (the T4 ceiling chase).

## Lane-aware Stage (implemented 2026-07-07)

Composes shared-memory staging with `Tensorize`, dissolving the deferral above:
`Stage { …; cooperative = Some simd_width }` (shared, all-`Serial` tile loops). Exercised by the
staged+tensorized variant of `schedule_mma_matmul` — on Metal the micro-kernel reads both tiles
from `threadgroup` memory via `simdgroup_load`, the mb tile's lane load replaces the minor-axis
loop outright (extent = width, guard folded), and the ma tile keeps its surviving `lane < 8`
guard. Design as sketched:

1. **Stage mints its own lane loop.** A `~cooperative:simd_width` mode on shared `Stage` wraps the
   cooperative load nest in a *fresh* extent-`simd_width` `Workgroup`-typed loop. No coordination
   with `Tensorize`'s lane loop is needed beyond passing the same `simd_width`: positional slot
   assignment puts both innermost `Workgroup` loops in slot 0, and `validate_parallel`'s
   barrier-strength uniformity (a `Tile_mma` *is* a barrier) rejects any extent mismatch. This
   dissolves the ordering dilemma — Stage still runs before Tensorize and never needs to know
   about the statement it feeds.
2. **Partition the loads along the lane.** Fold the lane symbol linearly into the copy nest's
   innermost fresh loop: when that loop's extent `E` is a multiple of `simd_width`, iterate
   `E / simd_width` steps at index `e·W + w` — an `Affine` term, expressible today. Division and
   modulo are *not* in the affine index algebra, so non-multiple extents fall back to the existing
   representative-thread discipline (`w == 0`, or `w < E` when `E < W`) rather than a flattened
   partition. Edge guards stay construct-then-fold.
3. **Barriers unchanged.** The loads sit inside the lane loop, the `Workgroup_barrier` stays its
   sibling at the staging point — hardware `Workgroup` loops bind rather than iterate, so the
   barrier remains uniformly reached.
4. **Tensorize composes as-is.** Its micro-kernel recognition already works over Stage-remapped
   tile reads, and `Tile_mma` takes strides from the operand tnode's dims, so the staged tile's
   leading dimension is automatic. The cost: with staging at a serial `k_o`, `Tile_mma.k` becomes
   `BK` per iteration, so `d` fragments are loaded/stored once per `k_o` — the residency the
   full-`K` block statement otherwise keeps. Recovering cross-`k_o` residency is the original
   proposal's fragment-marked accumulator node (§3.4), or an emission-time hoist of the `d`
   load/store when the enclosing serial loop provably doesn't otherwise touch `d`.
5. **`Workgroup_lane` only if needed.** A distinguished axis flavor becomes worthwhile only when
   some transform must *recognize* lane loops structurally; extent-matching suffices for v1.

Landed scope matched the estimate: ~100 lines in `apply_stage` plus the `cooperative` field; no
IR changes; no `validate_parallel` changes. One addition the sketch missed: the staging-point
workgroup-slot coverage check counts slot 0 as covered by construction in cooperative mode (the
fresh lane loop binds it; extent agreement is enforced downstream by the uniformity rule).

## Autotune sketch seeding (2026-07-15)

The benchmark caches showed no tuned schedule ever contained a `Tensorize`: the greedy beam
(default width 2, rounds 2) cannot reach the composition — a bare `Tensorize` from the serial
baseline leaves every outer loop serial (one simdgroup total), loses round 1, and is discarded
before Grid retypes could join it; candidates seeded from the default schedule have no serial
triples left, so `Tensorize` never enters their menu. The fix seeds the composed pipelines
directly as whole-routine sketch candidates (`Autotune.sketch_seed_params`, gated on
`hardware_limits.mma` for GPU):

- **GPU unstaged** (`bk = 0`): Split `i`/`j` into Grid blocks (`bn` pinned to the lane width so
  the zeroing's column grid blocks align), sink `i_i`, `Tensorize` the inner triple over the
  full reduction — one `Tile_mma` block statement, `d` traffic fully amortized.
- **GPU staged**: additionally Split `k`, sink to `i_o { j_o { k_o { i_i { j_i { k_i }}}}}`,
  cooperative shared `Stage` of both operands at `k_o`, `Tensorize` — the pinned
  staged+tensorized pipeline of `schedule_mma_matmul.ml`.
- **CPU** (seeded regardless of `limits.mma`): the whole-triple `Tensorize`
  (`bin/schedule_bench.ml`'s tensorize variant, rendered register-tiled per gh-ocannl-469) plus
  Grid-split row-block variants for pool parallelism.

Stage-only composition, deliberately no `Privatize`: it would relocate the accumulator into
thread-local scratch, which `simdgroup_load` cannot address (`mma_syntax` declines thread-space
operands and the whole statement silently falls back to scalar); `Tile_mma`'s block semantics
already keep accumulator fragments register-resident across the reduction. The zeroing nest
mirrors the accumulation's grid geometry with an inner Workgroup loop of extent `simd_width`
covering the lane slot (barrier-strength uniformity). Verified on Metal: all five mma sketch
candidates of the 32×32 site compile, validate, and time (`autotune_fission_sketch.ml`).

### Per-fission-segment seeding (`F_sketch`, 2026-07-15)

Whole-routine seeds alone would never reach heavily fissioned graphs (gpt2, multi-layer MLPs),
which tune per segment — so the sketches are additionally seeded per fission segment. The
`F_sketch` fissioned flavor carries `(pre-schedule segment digest, sketch params)` pairs (keyed
like `F_saved`): at seed time the fission segmentation is enumerated once on a hermetic copy of
the base lowering (same pipeline settings as the fissioned preset candidates), `detect_matmul`
runs per `` `Normal `` segment, and each keyed segment gets its sketch pipeline while the rest
keep the default preset — so the segmentation converges with the enumeration. Two consequences
of fission shaped the implementation:

- A segment's site is **unzeroed** — the whole-node `Zero_out` fissions into its own `` `Zeros ``
  segment — so all sketch pipelines now make the zero-expansion geometry conditional on
  `m_zeroed`. Sound without it: `Privatize` init-loads the accumulator tile from the
  (pre-zeroed) target, and `Tile_mma` loads the accumulator fragment before the reduction.
- Hoisted (constant-pool) `Stage` packing used to fail at link time under fission: the packed
  tile registered only in the segment's *filtered* traced store, while context allocation
  enumerates the routine-level (pre-fission) store. `Backends.compile` now folds
  segment-created traced-store entries back into the routine-level store.

Verified on cc (6 per-segment candidates: 2 packing + 2 hoisted + 2 tensorized) and Metal (9:
4 blocktiling + 5 simdgroup) — all compile and time (`autotune_fission_sketch.ml`).

## Relations

- [schedule-ir-optops](schedule-ir-optops.md): §5 `Stage` supplies the shared tiles and
  barriers; §6 the preset this extends; §7 fission is orthogonal (matmuls are single
  nests) but shares the divergence-validation machinery.
- [gh-ocannl-412](gh-ocannl-412.md): the matmul optimization arc this completes.
- [gh-ocannl-164](gh-ocannl-164.md): CPU vectorization; its open matmul micro-kernel
  item is the CPU-side analogue of a tile step.
- [axis-types-for-loops](axis-types-for-loops.md): the axis-type vocabulary
  (`Workgroup_reduce` as the precedent for semantically-annotated hardware axes).
