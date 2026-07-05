# Schedule IR: OptOps-style loop transforms as values

**Date**: 2026-06-12 (stub); elaborated 2026-07-04; implemented 2026-07-05
**Status**: Phases S1–S4 implemented (`arrayjit/lib/schedule.ml`). Deviations and findings:

- The vocabulary gained `Expand_zero` (not in the original §1): lowering keeps `Zero_out` as an
  opaque statement, and whole-node zeroing is rejected in multi-threaded kernels, so annotated
  matmul schedules must first expand it into a splittable loop nest.
- `Split` carries caller-minted fresh symbols (`Schedule.split` smart constructor) instead of
  minting inside `apply` — programmatic schedules need the symbols for later ops.
- `Stage`'s tile shape generalizes §5: dims derive per source axis from the range of the
  tile-loop terms, so multi-term register-tile decompositions (`TM*i_w + i_t`) stage cleanly.
  V1 shared staging requires tile sizes dividing the extents — `Split`'s whole-body remainder
  guards would put the barriers under divergent control flow.
- The §6 default annotator ships and is on by default for cuda/metal (config
  `automatic_gpu_schedule`, `gpu_schedule_block_size`, `gpu_schedule_min_parallel`); it adds a
  conservative race analysis beyond the §6 sketch (cross-nest producer/consumer bail, per-node
  index-agreement on parallel components).
- Benchmarks (`bin/schedule_bench.exe`, Apple-silicon Metal, 256³/512³ f32 matmul): parallel
  (S1 shape) ≈ 100–130 GFLOP/s, 1800–3000× over the naive 1×1 launch — the #412 >10× criterion
  is met by parallelization alone. SMEM (S2) and register tiles (S3) initially landed at parity
  with S1: every fma round-tripped the output element through global memory.
- **`Privatize` (2026-07-05, also not in the original §1)** closes that gap: it contracts the
  read-modify-write of a materialized accumulator across a reduction loop into a per-thread
  `Local` tile (init-load before, store-back after) — recovering for materialized nodes the
  scope-local form virtualization gives virtual accumulators, and sidestepping the aliasing
  obstacle since a routine-local tile cannot alias kernel pointers (no `restrict` needed for
  the accumulator; [gh-ocannl-164](gh-ocannl-164.md) still helps the loads). Composition order:
  Splits → Stages → Privatize → materializing Unrolls (which turn the tile accesses into
  constant-indexed, register-allocatable form). With it, register tiling pulls ahead of plain
  parallel: 200 vs 184 GFLOP/s at 512³, 145 vs 125 at 1024³ (~16%); cc cache tiles + packing
  improve from 3.3× to 3.8× single-threaded (9.0 GFLOP/s at 256³). The remaining headroom is
  tile-size tuning, vectorized loads, and eventually the BEAM search. Both prerequisites landed on 2026-07-04:
[axis-types-for-loops](axis-types-for-loops.md) (Phases A–C: `axis_type` on `For_loop`,
`Workgroup_barrier`, `workgroup_shared`, the `If` guard statement, hardware rendering,
launch dims, `validate_parallel`) and
[interval-analysis-scalar-t](interval-analysis-scalar-t.md) (guard folding). The
application seam also landed: `?lowered_transform` on `Context.compile`
(`backend_intf.ml:198`, applied at `backends.ml:492`). Seeded by the tinygrad deep dive
([a-range-is-not-its-shape](../blog/a-range-is-not-its-shape.md), port area 1); the
consumer is v0.8 tiling ([#412](https://github.com/ahrefs/ocannl/issues/412)).

## Goal

A schedule layer for `Low_level.t`: loop-nest transforms (Split-with-retype,
Swap/interchange, Unroll, Stage/tile-staging, later Upcast/vectorize and Padto)
represented as *values* — a list of `optop`s — applied as a pure
`Low_level.optimized -> Low_level.optimized` pass at the `?lowered_transform` seam,
Halide-style, rather than tinygrad's destructive mid-pipeline rewrite. A schedule is
then searchable (BEAM over schedule prefixes with on-device timing), cacheable, and
testable independently of the kernel it acts on.

## What the landed groundwork provides

Everything downstream of the transform now exists; only the transform itself is missing:

- **A type to retype to**: `axis_type = Serial | Grid | Workgroup | Workgroup_reduce |
  Unrolled` on `For_loop` (`low_level.ml:42,59`). Annotated kernels render as hardware
  index bindings, launch with real grid/block dims on CUDA and Metal, and fall back to
  serial loops on cc (`c_syntax.ml:484-540`, hooks at `c_syntax.ml:73-84`).
- **A guarded-write statement**: `If of { cond : scalar_arg; body : t }`
  (`low_level.ml:83`) — exactly what Split's remainder tiles and Padto's masks need.
  `simplify_llc` folds interval-decided conditions, so guards that the loop extents
  prove (the divisible-tile common case) cost nothing.
- **Shared tiles and barriers**: `optimized.workgroup_shared : Set.M(Tnode).t`
  (`low_level.ml:227`) renders as `__shared__`/`threadgroup` declarations;
  `Workgroup_barrier` renders per backend; cooperative loads are ordinary
  Workgroup-annotated `For_loop`/`Set`/`Get` code — no new IR vocabulary needed.
- **A safety net**: `compile_proc` runs `validate_parallel` (`low_level.ml:2574`) —
  slot arity, barrier divergence, materialized-write coverage of every active hardware
  dimension — then injects extent guards (`guard_annotated_extents`,
  `low_level.ml:2672`) and computes `launch_dims` (`low_level.ml:2532`). An ill-formed
  schedule fails at backend-compile time with a named error, not a race.
- **A test template**: `test/operations/hardware_axes_parity.ml` and
  `hardware_workgroup_reduce.ml` — run the transformed kernel against its `Serial`
  twin, assert structure from `build_files/` source, dispatch expectations per backend.
- **Signed, bounded indices**: hardware bindings and loop counters are `int32`
  (`int64` under `large_models`), and the per-node numel contract bounds every extent —
  Split's `Affine` index arithmetic cannot overflow the index width.

## Design

### 1. The `optop` vocabulary

```ocaml
(* new module arrayjit/lib/schedule.ml *)
type optop =
  | Split of {
      axis : Indexing.symbol;       (* the loop to split, identified by its index symbol *)
      factor : int;                 (* inner extent *)
      outer : Low_level.axis_type;  (* retype of the new outer loop (Serial = no retype) *)
      inner : Low_level.axis_type;  (* retype of the new inner loop *)
    }
  | Swap of { outer : Indexing.symbol; inner : Indexing.symbol }
      (* interchange two perfectly-nested loops *)
  | Retype of { axis : Indexing.symbol; ty : Low_level.axis_type }
  | Unroll of { axis : Indexing.symbol; materialize : bool }
      (* materialize=false: set axis=Unrolled, codegen repeats the body;
         materialize=true: unroll in the IR so simplify/CSE see the copies (§4) *)
  | Stage of {
      source : Tnode.t;             (* tensor whose accesses to stage through a tile *)
      tile_loops : Indexing.symbol list;  (* loops whose extents size the tile *)
      shared : bool;                (* true: workgroup_shared tile + cooperative load +
                                       barriers; false: per-thread/Local tile (CPU packing) *)
    }

type schedule = optop list  (* applied left to right *)

val apply : schedule -> Low_level.optimized -> Low_level.optimized
```

Axes are referenced by their index symbol: symbols are unique within one lowered
routine (`optimize_proc` keys `reverse_node_map` by symbol), and `Split` mints fresh
symbols via `Indexing.get_symbol` for the outer/inner pair, returning them to the
caller (schedules built programmatically) or recording them in a name environment
(schedules built by search, which addresses loops positionally and resolves to symbols
when applying).

Deliberately absent from v1, mirroring the axis-types enum's restraint: `Upcast`
(vectorize — waits on the `Set_from_vec`/`vec_unop` growth path and a `Vectorized`
axis type), `Padto` (masking is representable today via `If` + interval discharge, but
its profitability case is tensor-core alignment, which is far), and tensor-core ops.
`Cpu_parallel` retyping is **retired** (decision 2026-07-05, recorded in
[gh-ocannl-164](gh-ocannl-164.md)): within-routine CPU threading binds `Grid` axes to a
task pool in the C backend's rendering — `Grid`'s contract is exactly the task-pool
contract, so the same schedules serve GPU and CPU. The exhaustive-match language keeps
late additions safe.

### 2. Pass ordering — the seam is *post*-optimization (decision)

The stub's open question was ordering against virtualization. The landed seam settles
it: `lowered_transform` runs after the whole `optimize_proc` pipeline (virtualize →
cleanup → `simplify_llc` → one-hot rewrite → CSE → `hoist_cross_statement_cse`,
`low_level.ml:3109-3128`) and before `compile_proc`'s validate/guard/launch steps. So
the realized order is:

```
lower → virtualize → simplify → CSE/hoist → SCHEDULE → (local re-simplify) → validate → guard → render
```

Consequences, recorded as the normative contract:

- **The schedule sees fused code.** Virtual (inlined) nodes are already folded into
  the loop nests, so tiling a matmul automatically tiles its inlined elementwise
  epilogue — fusion composes with tiling for free.
- **Transforms must fold their own guards.** `Split` with a non-dividing factor and
  `Stage`'s edge tiles emit `If` guards; since `simplify_llc` already ran,
  `Schedule.apply` finishes by re-running `simplify_llc` (and, when `Unroll
  ~materialize:true` or `Stage` created duplication, `eliminate_common_subexpressions`
  + `hoist_cross_statement_cse`) on its output. All three are pure and idempotent;
  this is the interval proposal's "introducing pass invokes the folder" rule applied
  wholesale. Cheaper and simpler than moving the seam inside `optimize_proc`.
- **No re-virtualization.** `traced_store` is fixed by the time the schedule runs; a
  schedule cannot flip an inlining decision (the bounded re-virtualization iteration
  the stub contemplated is dropped — nothing in the v1 vocabulary needs it; Padto, the
  one op that could, is deferred). Tiles created by `Stage` are fresh `Local`-mode
  nodes born after virtualization, so there is no interaction to police. Note
  `traced_store` must gain entries for staged tiles (they are materialized-per-kernel
  locals); `apply` updates the store the same way `optimize_proc`'s tracing would.

### 3. `Split` mechanics

`For_loop {index=i; from_=0; to_=N-1; body}` with factor `T` becomes

```
For_loop {index=i_o; from_=0; to_=ceil(N/T)-1; axis=outer; body=
  For_loop {index=i_i; from_=0; to_=T-1; axis=inner; body=
    [If (T*i_o + i_i < N)]   (* only when T ∤ N *)
      subst(body, i := T*i_o + i_i)}}
```

Substitution is total over the two places a symbol can occur:

- **Index vectors** (`Get`/`Set`/`Set_from_vec` `idcs`): `Iterator i` becomes
  `Affine {symbols=[(T, i_o); (1, i_i)]; offset=0}`; an existing `Affine` term
  `(c, i)` expands to `(c*T, i_o); (c, i_i)` (`indexing.ml:139-146` — the
  representation was built for this).
- **Scalar expressions**: `Embed_index` (`low_level.ml:112`) carries an `axis_index`,
  so the same rewrite applies.

The remainder guard is `Binop (Cmplt, Embed_index (Affine ...), Constant N)` wrapped
in `If` — and when `T | N`, the interval environment (`i_o ∈ [0, N/T-1]`,
`i_i ∈ [0, T-1]` ⇒ `T*i_o + i_i ∈ [0, N-1]`) folds it away; `apply`'s trailing
simplify does this. Annotated loops require `from_ = 0`, which lowering guarantees
today; `Split` asserts it.

**Legality**: splitting a serial loop is always legal (same iterations, same order
when the nest order is outer-then-inner). Retyping to `Grid`/`Workgroup` carries the
independence obligation — that is the annotating caller's responsibility exactly as
`validate_parallel`'s docs state; the structural half is checked downstream.

`Swap` requires perfect nesting (the outer loop's body is exactly the inner loop);
`apply` fails loudly otherwise. Interchanging serial loops reorders iterations —
legal for the accumulation patterns lowering emits (associative-commutative accums;
bitwise-reproducibility caveats belong to the search harness, which compares against
the unscheduled twin).

### 4. `Unroll`: annotation flavor vs. IR flavor

The landed `Unrolled` axis type repeats the body at *codegen* time
(`c_syntax.ml`, per-block constant bindings) — after simplify and CSE have run. That
is right for freeing an index register, but wrong for register tiling, where the
payoff comes from the optimizer seeing the copies: constant-folding `Affine` indices
into `Fixed_idx`, and CSE deduplicating the shared-tile loads that the TM×TN
outer-product form repeats across copies. Hence the `materialize` flag: the IR flavor
substitutes constants (per the `unroll_dims` precedent, `low_level.ml:3467`) and
relies on `apply`'s trailing simplify+CSE to produce the register-blocked form. This
is the "optimized via inlining and CSE" leg of the plan: **register blocktiling is
not a bespoke transform — it is Split + materializing Unroll + the existing CSE.**

### 5. `Stage`: tile staging (the one genuinely new pass)

Split/Swap/Retype/Unroll are structural rewrites. `Stage` synthesizes code:

1. Mint a tile `Tnode.t` (`Local` memory mode — the "uniquely unobservable
   routine-scoped scratch" contract fits exactly), dims = extents of `tile_loops`,
   precision = source's.
2. If `shared`: add it to `optimized.workgroup_shared`; emit a cooperative-load nest
   (Workgroup-annotated loops striding the tile so each thread loads its portion),
   a `Workgroup_barrier`, then the consumer nest with `Get source` rewritten to
   `Get tile` under remapped (tile-local) indices, then a second barrier before the
   next outer-tile iteration. Edge tiles get `If` guards on the loads
   (construct-then-fold as usual).
3. If not `shared` (CPU packing): a plain serial copy nest into a stack tile —
   Böhm's operand packing — no barriers; profitable because the repacked layout is
   contiguous for the micro-kernel.

This is where tinygrad pays with GROUP_REDUCE/LOCAL machinery; OCANNL pays with one
transform that emits ordinary IR. The index remapping is the delicate part: within
the consumer scope, `source`'s index vector restricted to `tile_loops` becomes the
tile's index vector; accesses mixing staged and unstaged symbols in one axis are
rejected in v1.

### 6. Heuristic presets — the default annotator

The schedule layer is also how *every* kernel stops running single-threaded, not just
matmul. A preset (`Schedule.default_gpu : optimized -> schedule`) computes, on the
lowered IR:

- **Parallelizable loops**: outermost loops whose index appears in the index vector of
  every materialized `Set` beneath them — the same coverage property
  `validate_parallel` enforces, used generatively. (Projection information is *not*
  needed post-lowering: output vs. contraction structure is recoverable from where
  indices appear in writes.)
- **Elementwise / outer-parallel kernels**: `Split` the parallel loop(s) by the block
  size (default 256, clamped by the backend's max-threads query), retype outer=`Grid`,
  inner=`Workgroup`. Up to 3 of each kind (slot limit); fold or leave-serial beyond.
- **Reduction loops stay serial** in the preset (one thread per output element);
  `Workgroup_reduce` staging is the opt-in matmul/attention path.
- **Small kernels** (total iteration count below a threshold, default 1024): empty
  schedule — the all-`Serial` kernel launches 1×1 as today.

The matmul preset is the composition from
[gh-ocannl-412](gh-ocannl-412.md): `Split i (BM)` → `Split j (BN)` → `Split k (BK)` →
retype `i_o,j_o=Grid`, `i_i,j_i=Workgroup` → `Swap` k_o outward → `Stage A ~shared` +
`Stage B ~shared` → `Split i_i (TM)`/`Split j_i (TN)` + materializing `Unroll` for the
register tile. Each prefix of that schedule is independently runnable and
parity-testable — which is exactly the property BEAM search needs.

### 7. Search and caching (scoped, deferred)

- BEAM over schedule prefixes, timing candidates on-device; contexts-as-values means
  candidate compiles are sibling `Context.compile` calls from one frontier — no global
  device state to isolate. Correctness gate per candidate: executed parity against the
  unscheduled twin (the `hardware_axes_parity` harness generalized).
- Cache winners keyed by (canonicalized kernel hash, device identity); OCANNL's
  runtime compilation makes per-shape specialization natural.
- Cost models and search policy belong to
  [gh-ocannl-261](gh-ocannl-261.md) follow-ups; v1 ships the presets of §6 plus the
  hand-written matmul schedule, no search.

## Phasing

- **Phase S1 — structural ops**: `schedule.ml` with `Split`/`Swap`/`Retype`/`Unroll`,
  substitution, guard construction, trailing simplify+CSE; unit tests at the
  `Low_level.t` level (structural) plus executed parity via `?lowered_transform`
  (Metal + cc fallback locally, CUDA in CI). Deliverable: the default GPU annotator
  preset — every elementwise kernel launches parallel.
- **Phase S2 — `Stage`**: shared tiles + cooperative loads; the SMEM matmul schedule;
  parity + a first speedup measurement vs. the S1 (parallel, un-tiled) kernel.
- **Phase S3 — register tiling**: materializing `Unroll`, the full matmul preset;
  benchmark against S2 and the naive baseline (the #412 >10× criterion lives here).
- **Phase S4 — CPU**: non-shared `Stage` (packing) + cache-sized `Split` presets for
  cc ([watch-ocannl-README-md-347818d3](watch-ocannl-README-md-347818d3.md) scope);
  vectorization/`Upcast` remains a follow-up (`Cpu_parallel` is retired — CPU
  parallelism is pool-backed `Grid` rendering; see [gh-ocannl-164](gh-ocannl-164.md)).

## Acceptance criteria

- [x] `optop` type and `Schedule.apply` implemented against `Low_level.optimized`,
      with substitution over `axis_index` and `Embed_index`, remainder guards via
      `If`, and the trailing simplify/CSE contract of §2.
- [x] Pass-ordering decision recorded (§2: post-optimization seam, transforms
      re-simplify locally, no re-virtualization) — done in this document; the
      code comments in `schedule.ml` point here.
- [x] Default GPU annotator preset: all-`Serial` elementwise kernels above the
      threshold get Grid×Workgroup schedules automatically; parity suite green on
      cc/Metal locally (CUDA via CI).
- [x] Hand-written SMEM+register matmul schedule passes executed parity
      (`test/operations/schedule_smem_matmul.ml`, `schedule_register_matmul.ml`) and
      beats the unscheduled kernel by far more than the #412 target on Metal
      (~1800–3000×; see the status note — the margin comes from parallelization,
      with SMEM/register tiles at parity pending accumulator privatization).
- [x] Search harness scoped (BEAM first; cost models deferred to #261 follow-ups) —
      §7; implementation deferred.

## Relations

[axis-types-for-loops](axis-types-for-loops.md) (**landed** — supplies the types,
barrier, shared placement, `If`, validation, rendering, launch),
[interval-analysis-scalar-t](interval-analysis-scalar-t.md) (**landed** — guard
discharge), [signed-index-precision](signed-index-precision.md) (**landed** — index
width and extent bounds),
[#412](https://github.com/ahrefs/ocannl/issues/412) (GPU tiling — the consumer;
[gh-ocannl-412](gh-ocannl-412.md) is now the requirements/benchmarks doc, this is the
mechanism doc), [watch-ocannl-README-md-347818d3](watch-ocannl-README-md-347818d3.md)
(CPU matmul — Phase S4), [gh-ocannl-164](gh-ocannl-164.md) (CPU SIMD floor),
[gh-ocannl-267](gh-ocannl-267.md) (Tiramisu — thesis is this layer),
[gh-ocannl-242](gh-ocannl-242.md) (TVM/Ansor lineage),
[gh-ocannl-261](gh-ocannl-261.md) (search/cost functions),
[gh-ocannl-263](gh-ocannl-263.md) (Flash attention — eventual `Stage` consumer).
