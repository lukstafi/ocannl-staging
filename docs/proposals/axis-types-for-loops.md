# Axis-type annotations on Low_level loops

**Date**: 2026-06-12 (stub); elaborated 2026-07-03
**Status**: Elaborated proposal — seeded by the tinygrad deep dive
([a-range-is-not-its-shape](../blog/a-range-is-not-its-shape.md), port area 2); first
site of the decision is [#412](https://github.com/ahrefs/ocannl/issues/412)'s
grid/block mapping.

## Goal

Make the loop-to-hardware mapping explicit in the IR instead of a backend convention:
an axis-type annotation on `For_loop` in the spirit of tinygrad's `AxisType`
(GLOBAL / LOCAL / THREAD / LOOP / REDUCE / GROUP_REDUCE / UPCAST / UNROLL —
`tinygrad/uop/ops.py`), so that backends emit grid/block/thread index bindings for
annotated axes rather than loops, and so that schedule transforms
([schedule-ir-optops](schedule-ir-optops.md)) have a type to assign when splitting.

Companion representation decisions bundled here because they share the annotation's
lifetime and consumers: a workgroup-scoped barrier statement in `Low_level.t`, and a
workgroup-shared placement refinement for `Local`-mode tensor nodes (the
tinygrad-`LOCAL`-addrspace analogue).

## Acceptance criteria

- [ ] `axis_type` field added to `For_loop` with `Serial` default at every
      construction site; all-`Serial` programs compile to byte-identical kernels
      (modulo the removed single-thread guard) on cc, CUDA, and Metal.
- [ ] `Workgroup_barrier` statement and shared-placement refinement represented in
      `Low_level.t` / `optimized`, rendered on CUDA and Metal, rejected with a clear
      error on cc.
- [ ] Launch dimensions derived from annotations and plumbed to `launch_kernel`
      (CUDA) / `dispatch_threadgroups` (Metal); the `kernel_prep_line` single-thread
      guard is gone.
- [ ] A hand-annotated executable parity test: the same `Assignments.comp` lowered
      once with all-`Serial` loops and once with `Grid`/`Workgroup` annotations
      produces equal results (Metal locally, CUDA in CI, cc via serial fallback).
- [ ] A structural expected-file test showing the generated `.cu`/`.metal` source for
      an annotated kernel (index bindings, `__shared__`/`threadgroup` declaration,
      barrier).

## Context

### Where the code stands

Every backend today emits single-threaded kernels; the loop-to-hardware question is
unanswered anywhere in the tree:

- `For_loop { index; from_; to_; body; trace_it }` (`low_level.ml:38`) carries no
  hardware information. `pp_ll` (`c_syntax.ml:450-467`) renders every loop as a C
  `for` statement, identically for all backends.
- CUDA guards every kernel with `kernel_prep_line = "if (threadIdx.x != 0 ||
  blockIdx.x != 0) { return; }"` (`cuda_backend.ml:360`) and launches
  `~grid_dim_x:1 ~block_dim_x:1 ~shared_mem_bytes:0` (`cuda_backend.ml:1043`).
- Metal already passes `uint3 gid [[threadgroup_position_in_grid]]` and `uint3 lid
  [[thread_position_in_threadgroup]]` to every kernel (`extra_args`,
  `metal_backend.ml:444`) but dispatches one threadgroup of one thread
  (`metal_backend.ml:900-902`).
- The IR has no barrier, no shared-memory declaration, no address-space notion.
  `Local_scope` scalars are the `REG` end; `Local`-mode tensor nodes are emitted as
  kernel-local stack arrays (`c_syntax.ml:1179-1195`).

The choice is whether to answer the mapping question as a backend convention
("backends parallelize outer loops they deem profitable") or as IR annotation.
tinygrad's experience argues for annotation: the schedule layer's Split-and-retype
needs a type to retype *to*; the kernel before and after scheduling are both
printable, diffable values; and the backend renderers stay dumb — they translate
annotations, they don't decide them.

### Key code pointers

| Location | Description |
|----------|-------------|
| `arrayjit/lib/low_level.ml:38` | `For_loop` definition; `trace_it` is the precedent for loop-level metadata |
| `arrayjit/lib/low_level.ml:1006-1012` | virtualization's fresh-symbol loop reconstruction (explicit record rebuild) |
| `arrayjit/lib/low_level.ml:1756` | structural loop comparison used by CSE — must include the new field |
| `arrayjit/lib/low_level.ml:2657` | `loop_over_dims` — main `For_loop` construction site |
| `arrayjit/lib/low_level.ml:182-188` | `optimized` record — carrier for the shared-placement set and launch dims |
| `arrayjit/lib/assignments.ml:311,489` | `loop_accum` / `loop_accum_rev` — the other construction sites (projection-driven nests) |
| `arrayjit/lib/c_syntax.ml:27-94` | `C_syntax_config` — where the new rendering hooks go |
| `arrayjit/lib/c_syntax.ml:450-467` | `pp_ll` `For_loop` case — the rendering branch point |
| `arrayjit/lib/c_syntax.ml:1019-1210` | `compile_proc` — kernel prologue, `kernel_prep_line`, local declarations |
| `arrayjit/lib/cuda_backend.ml:360,1043` | single-thread guard and hardcoded 1×1 launch |
| `arrayjit/lib/metal_backend.ml:444,900` | `gid`/`lid` extra args and 1-threadgroup dispatch |
| `arrayjit/lib/tnode.ml:12-34` | the 5-constructor `memory_mode` lattice (kept untouched by this proposal) |

## Design

### 1. Annotation carrier: a `For_loop` field, not symbol metadata

```ocaml
(* low_level.ml *)
type axis_type =
  | Serial            (* plain for-loop; the default — today's behavior *)
  | Grid              (* one grid dimension: blockIdx.* / threadgroup_position_in_grid *)
  | Workgroup         (* one block/threadgroup dimension: threadIdx.* / thread_position_in_threadgroup *)
  | Workgroup_reduce  (* a Workgroup axis participating in a shared-memory reduction *)
  | Unrolled          (* fully unrolled at codegen *)
[@@deriving sexp, compare, equal]

  | For_loop of {
      index : Indexing.symbol;
      from_ : int;
      to_ : int;
      body : t;
      trace_it : bool;
      axis : axis_type;
    }
```

Why the loop and not the symbol:

- `Indexing.symbol` is `Symbol of int` (`indexing.ml:8`) — context-free, compared and
  hashed pervasively. Attaching metadata means either changing equality semantics or
  maintaining a global side table.
- A side table is fragile exactly where it matters: virtualization inlines by
  substituting *fresh* symbols (`low_level.ml:1006-1012`), and every renaming would
  have to migrate table entries. A record field travels for free through the
  `For_loop { for_config with body }` idiom used throughout the optimizer.
- The annotation is a property of the *loop* (how these iterations map to hardware),
  not of the index value. Body occurrences need nothing: an annotated loop renders as
  a binding `const unsigned int i42 = blockIdx.x;`, so `pp_symbol` and every
  `axis_index` in the body are untouched.
- `trace_it` is the precedent: loop-level metadata lives on the loop.

Deliberately absent from the initial enum, to be added when their codegen exists (an
exhaustive-match language makes late additions safe): `Vectorized` (UPCAST — reserved
for the `Set_from_vec`/`vec_unop` growth path), `Cpu_parallel` (THREAD — within-routine
CPU parallelism; today cc parallelism is across streams via the Multicore scheduler),
and any warp-level type. tinygrad's `REDUCE` is *not* ported: reduction-ness of a
serial loop is recoverable from the body (accumulation into a scope or node), and
carrying it would create a redundancy the optimizer must keep consistent.

**Hardware slot assignment** is positional, not stored: among a kernel's `Grid` loops,
the innermost binds `.x`, the next `.y`, then `.z` (innermost-fastest for coalescing);
same rule independently for `Workgroup`/`Workgroup_reduce` loops. More than 3 of a
kind per kernel is a validation error in v1 (the schedule layer can fold axes before
annotating, later). Keeping slots out of the IR means Split/Swap transforms never
have to renumber anything.

### 2. Semantics and well-formedness

An annotated loop means: *iterations are executed by distinct hardware threads; the
loop binds its index to the hardware index rather than iterating*. The loop's extent
(`to_ - from_ + 1`; `from_` must be 0 for annotated loops) contributes to the kernel's
launch dimensions.

Obligations on whatever pass writes annotations (initially hand-written tests and
[#412](https://github.com/ahrefs/ocannl/issues/412)'s heuristics, then the schedule
layer), collected in a `validate_parallel : t -> unit` check run at backend-compile
time:

- **Independence**: iterations of a `Grid`/`Workgroup` loop must have no
  cross-iteration dependencies. `Workgroup_reduce` is the labelled exception: its
  cross-iteration communication must be staged through shared placements and
  barriers.
- **Thread-locality across `Seq`**: a producer statement and consumer statement in
  the same kernel are legal only if the dependence is thread-local under the
  annotation (the hardware thread reads what it itself wrote — e.g. the zero-init
  nest and accumulation nest of one `Accum_op` annotated with identical extents and
  slot order), or workgroup-local and separated by a barrier. Cross-workgroup
  dependencies inside one kernel are ill-formed — there is no grid sync; that
  boundary belongs to kernel splitting (megakernel territory,
  [#318](https://github.com/ahrefs/ocannl/issues/318)).
- **Extent consistency**: sibling nests binding the same slot should have equal
  extents. Launch dims take the per-slot max; a smaller-extent binding is wrapped in
  an `if (i < extent)` guard — but a kernel containing barriers must have *equal*
  workgroup extents everywhere (a barrier under divergent control flow is UB), which
  `validate_parallel` enforces.
- **Barrier placement**: `Workgroup_barrier` must not appear lexically inside a
  guarded/divergent region or inside a `Serial` loop nested under differing-extent
  workgroup axes.

Serialization is always a legal implementation of `Grid`/`Workgroup` *in the absence
of barriers* (running the iterations in order refines any parallel interleaving).
This is what keeps the cc backend sound with zero new machinery — see §5. Barriers
break serialization-by-outer-loop (thread 0 would run past the barrier before thread
1 starts), which is why cc rejects them rather than no-ops them.

### 3. Barrier and shared placement in the IR

**Barrier** — one new statement constructor:

```ocaml
(* low_level.ml, type t *)
| Workgroup_barrier
```

Grid-scoped synchronization is deliberately not representable — that is the honest
boundary tinygrad also draws (its `Barrier` is workgroup-scoped; kernels split at
reduction edges). If cooperative-groups grid sync ever earns its keep, it arrives as
a new constructor, not a parameter.

Optimizer contract: `Workgroup_barrier` is an opaque effectful statement — no CSE, no
hoisting, no code motion across it (conservatively: a full memory fence). Since the
pipeline order is virtualize → schedule → simplify (the
[schedule-ir-optops](schedule-ir-optops.md) decision), virtualization never sees a
barrier — only `simplify_llc`-era passes need the conservative treatment, and they
already keep unknown statements in place; the new constructor's match arms are
`Noop`-like for analysis, atomic for motion.

**Shared placement** — *not* a new memory mode. The `memory_mode` lattice
(`tnode.ml:12-34`) stays at its five constructors; `Local`'s contract ("uniquely
unobservable routine-scoped scratch") already describes a shared tile exactly. What
differs is placement within the kernel, which is a *schedule decision* about
compiler-generated tiles that never escape the kernel — so it rides with the lowered
code, not with the tensor node:

```ocaml
(* low_level.ml *)
type optimized = {
  traced_store : traced_store;
  optimize_ctx : optimize_ctx;
  llc : t;
  merge_node : Tnode.t option;
  workgroup_shared : Set.M(Tnode).t;  (* Local-mode nodes to place in shared memory *)
}
```

`compile_proc`'s local-declaration pass (`c_syntax.ml:1179-1195`) consults the set:
a `Local` node in `workgroup_shared` is declared with the backend's shared prefix
(`__shared__ float t[256];` / `threadgroup float t[256];`) instead of a plain stack
array. Sizes are static (dims are known at lowering), so CUDA's
`shared_mem_bytes` stays 0 (static `__shared__` doesn't use the dynamic pool).
Zero-initialization of shared arrays cannot use `= {0}` (not allowed for
`__shared__`); nodes needing init get a cooperative-store loop generated by the
annotating pass, which is what it generates for tile loads anyway.

No `Cooperative_load` construct: cooperative tile loading is ordinary
`For_loop`(Workgroup-annotated)/`Set`/`Get` code plus a barrier, generated by the
schedule pass. The IR grows two things (a field's worth of annotation and a barrier),
not a vocabulary.

### 4. Launch-dimension plumbing

A pure analysis in `low_level.ml`:

```ocaml
type launch_dims = { grid : int array; block : int array }  (* each length ≤ 3 *)
val launch_dims : t -> launch_dims  (* all-1s for all-Serial code *)
```

computed at `compile_proc` time (it also performs slot assignment, shared with
`pp_ll` via an environment). Each backend's `code` record grows a `launch :
launch_dims` field, consumed at link/run time:

- CUDA: `S.launch_kernel func ~grid_dim_x:g.(0) ... ~block_dim_x:b.(0) ...
  ~shared_mem_bytes:0` replacing the hardcoded 1×1 (`cuda_backend.ml:1043`);
  `kernel_prep_line` is deleted (an all-Serial kernel launches 1×1, making the guard
  redundant).
- Metal: `threadgroups_per_grid`/`threads_per_threadgroup` from the same record
  (`metal_backend.ml:900-902`). Validate `block` product against
  `maxTotalThreadsPerThreadgroup` at pipeline creation.
- cc: after `validate_parallel` confirms no barriers/shared placements, launch dims
  are ignored (serial fallback).

Device-limit checks (max threads per block, max grid dims, shared-memory capacity)
live in the backends where device properties are queryable; `validate_parallel`
checks only backend-independent structure.

### 5. Backend mapping rules

Rendering is driven by three new `C_syntax_config` items (defaults in `c_syntax.ml`,
overridden per backend):

```ocaml
val hardware_index : kind:[ `Grid | `Workgroup ] -> slot:int -> string option
  (* None = backend cannot bind this axis in hardware *)
val barrier_syntax : string option
val shared_decl_prefix : string option
```

| IR construct | CUDA | Metal | cc (sync/multicore) |
|---|---|---|---|
| `Serial` loop | `for (...)` | `for (...)` | `for (...)` |
| `Grid` loop, slot k | `const unsigned int iN = blockIdx.{x,y,z};` binding, no loop; extent → `grid_dim_{x,y,z}` | binding from `gid.{x,y,z}` (already in `extra_args`); extent → `threadgroups_per_grid` | serial `for` (legal fallback) |
| `Workgroup` loop, slot k | binding from `threadIdx.{x,y,z}`; extent → `block_dim_{x,y,z}` | binding from `lid.{x,y,z}`; extent → `threads_per_threadgroup` | serial `for`, legal only barrier-free (validated) |
| `Workgroup_reduce` loop | as `Workgroup` (the distinction is for the schedule layer and validation, not the renderer) | as `Workgroup` | rejected |
| `Unrolled` loop | emitted as repeated body with substituted constants (per `unroll_dims` precedent) | same | same |
| `Workgroup_barrier` | `__syncthreads();` | `threadgroup_barrier(mem_flags::mem_threadgroup);` | rejected: `barrier_syntax = None` |
| shared placement | `__shared__ T name[n];` | `threadgroup T name[n];` | rejected: `shared_decl_prefix = None` |

A `None` from a hook is a compile-time error for barriers/shared, and a serial
fallback for `Grid`/`Workgroup` — matching §2's soundness argument. The schedule
layer will eventually consult a per-backend capability record before annotating; in
v1 the annotating test code just knows.

### 6. Migration sketch

Mechanical, in one PR (Phase A below), no behavior change:

1. Add `axis_type` and the `axis` field. Construction sites get `axis = Serial`:
   `loop_over_dims` (`low_level.ml:2657`), `loop_over_padding_region`
   (`low_level.ml:2705`), `loop_accum`/`loop_accum_rev` (`assignments.ml:311,489`),
   the fresh-symbol
   rebuild (`low_level.ml:1006-1012` — explicit record literal, so the compiler
   flags it). The ~8 `{ for_config with body }` reconstructions are untouched.
2. Include `axis` in the structural comparison at `low_level.ml:1756` (CSE
   alpha-equivalence: differently-annotated loops are different code) and in
   sexp/printing (expect-test churn: promote).
3. `Workgroup_barrier` constructor: match arms added everywhere the compiler
   exhausts `t` — analysis arms treat it as an effectful leaf (like `Zero_out`
   without a node), motion arms refuse to cross it. `optimized` grows
   `workgroup_shared` (empty everywhere initially).
4. New `C_syntax_config` hooks with cc/CUDA/Metal instances; `pp_ll` `For_loop` case
   branches on `axis`; `compile_proc` computes slots + launch dims; backend `code`
   records carry `launch`; guard deleted.

Steps 1–3 are Phase A (zero-diff on generated code); step 4 is Phase B.

## Approach / phasing

Implementation order across backends is **Metal first** (the primary dev box is a
Mac), then cc's serial fallback, then CUDA (CI-verified; not buildable locally). The
[#412 writeup](gh-ocannl-412.md)'s CUDA-centric phase list predates this decision and
is outdated on ordering — treat it as a catalogue of transforms, not a sequence.

- **Phase A — representation** *(landed 2026-07-03)*. IR field + barrier constructor
  + `workgroup_shared` field, all defaults preserving today's output byte-for-byte.
  Tests: existing suite green, promote sexp-printing churn.
- **Phase B — rendering and launch.** Hooks, bindings, slot assignment,
  `launch_dims`, guard removal, `validate_parallel` — Metal dispatch first, cc serial
  fallback alongside, CUDA following. Tests: structural expected-file test of a
  hand-built annotated `Low_level.t` (per the established pattern, built directly —
  lowering from the high level never produces annotations yet); executed parity test
  of a `Grid`+`Workgroup`-annotated element-wise kernel vs. its `Serial` twin on
  Metal (locally), the serial fallback on cc, and CUDA in CI.
- **Phase C — workgroup structure.** Shared declarations + barrier rendering (Metal
  `threadgroup` first); a hand-built shared-memory reduction (the GROUP_REDUCE
  pattern: partial sums in a shared tile, barrier, tree or serial combine) as an
  executed parity test vs. the serial reduction.
- **Consumers.** [#412](https://github.com/ahrefs/ocannl/issues/412) Phase 2+
  annotates via heuristics/tiling; [schedule-ir-optops](schedule-ir-optops.md)'s
  Split-with-retype makes annotation a schedule value;
  [gh-ocannl-263](gh-ocannl-263.md) (Flash attention) eventually consumes the
  workgroup structure. The `__constant__` end of address spaces
  ([#195](https://github.com/ahrefs/ocannl/issues/195)) can reuse the
  placement-refinement pattern (§3) rather than a memory mode, but is out of scope
  here.

## Risks and non-goals

- **Not solved here**: grid-level synchronization, persistent blocks, megakernels
  ([#318](https://github.com/ahrefs/ocannl/issues/318)) — tinygrad's spec doesn't
  solve them either; this is the staircase, not the summit. Kernel splitting at
  cross-workgroup dependence edges is future work the well-formedness rules make
  nameable.
- **Validation is structural, not semantic**: `validate_parallel` cannot prove
  iteration independence; that is the annotating pass's obligation (like `trace_it`
  correctness today). An executed-parity test harness is the practical check —
  every annotated kernel in the test suite runs against its `Serial` twin.
- **Simplify-after-schedule motion bugs**: the conservative-fence contract for
  `Workgroup_barrier` must be audited in the CSE/hoisting passes when Phase C lands
  (the passes recurse structurally, so the audit is bounded to their match arms).
- **Expect-test churn** from sexp changes in Phase A is benign but wide; promote in
  the same PR.

## Relations

[#412](https://github.com/ahrefs/ocannl/issues/412) (consumer and first site),
[#195](https://github.com/ahrefs/ocannl/issues/195) (addrspace cousin),
[#318](https://github.com/ahrefs/ocannl/issues/318) (megakernel exploration, landed),
[gh-ocannl-263](gh-ocannl-263.md) (Flash attention — eventual consumer),
[schedule-ir-optops](schedule-ir-optops.md) (depends on this),
[interval-analysis-scalar-t](interval-analysis-scalar-t.md) (sibling port; discharges
the guards that differing-extent bindings introduce).

## Original acceptance criteria (from the stub)

- [x] Annotation carrier decided (`For_loop` field vs. symbol metadata) with
      migration sketch for `to_low_level`/backends — §1, §6.
- [x] Barrier + LOCAL buffer representation in `Low_level.t` sketched — §3.
- [x] Backend mapping rules (CUDA grid/block, Metal threadgroups, CC threads)
      drafted — §5.
