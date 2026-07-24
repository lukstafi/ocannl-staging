# Agent notes: distilled cross-session knowledge

Promoted from coding agents' local session memory so that fresh clones inherit the design
history and trap knowledge that is not derivable from the code alone. Scope discipline:
machine-specific facts (which backends are installed, local paths, remote-benchmarking setups)
deliberately stay out — each machine's agents keep those locally. When a note disagrees with the
code, the interfaces, or the primary docs, those win; treat entries as leads, and verify the
named symbols still exist. Workflow rules live in CLAUDE.md; this file is subsystem lore.

## Shape inference and einsum

- Operator gotchas: unary sum-reduction is `++` (einsum1, e.g. `a ++ "i => 0"`); `+++` is the
  BINARY `outer_sum`, not a reduction. `m ++ "ii => i"` extracts diagonals. The `%op` `O` scope
  exposes only `<`, `=`, `<>` (plus `not`, `where`); derive `>`, `<=`, `>=` by swapping/negating.
  A NUMBER in an axis spec is a fixed-index placement (size k+1 axis, value at slot k), not "axis
  N": per-axis index grids are `range n ++ "i=>i0"` ([n,1]) and `range n ++ "j=>0j"` ([1,n]),
  and comparing them broadcasts to an [n,n] grid; `range_of_shape` gives row-major flattened
  offsets, not per-axis indices. In `%op` only numeric literals auto-lift to tensors — a
  let-bound float needs `!.x`.
- Solver deferral discipline: `unify_dim` solves eagerly (recursing internally), so any
  constraint it RETURNS is a deferral that cannot progress in the current env — e.g. a conv axis
  whose kernel dim is still a variable. Deferrals are idempotent: re-solving one in place
  livelocks; hand it to the driver's next pass, where substitution can make it actionable
  (`solve_inequalities` is the pattern). Executable probe for output-only windowed specs:
  `test/einsum/test_conv_output_only.ml`.
- `Row.Concat` dimensions (block tensors, `^` in specs) are solved as ARITHMETIC SUMS, not
  structural lists: equality lives in `unify_dim` (single-component unwrap, `Concat=Dim`
  subtraction, cancel-common residuals, nested-`Concat` flattening), and the equal-arity
  all-`Var` pairing must fire at ANY solver stage or incremental Stage-1 solving livelocks.
  Solver-level facts are testable directly via `Row.solve_inequalities` on hand-built
  constraints (`test/operations/test_concat_dim_solver.ml`).
- `Shape.compute_row_product` must not latch a product while the row still ends in an open
  `Row_var` — the historical bug silently resolved xavier/kaiming fan products to 1 for every
  `{ w }` param with an inferred input row (fixed; regression `test/operations/layer_norm_1d.ml`,
  `test_default_init_std.ml`). Fixed-index output axes (`=> ...|0`, recognized via their
  `At_least_dim`) must not be equated by the stage-2 singleton-bounds heuristic, so that it
  broadcasts at stage-7.
- Implicit (omitted) row specs are shared and safe-to-guess; an explicit `...` stays strict; row
  variables in kernel slots need the `|` prefix. Symbolic extents (`Row.Sym`, gh-490) are rigid:
  equal only to themselves, materialized at their declared maximum; `Total_elems` against them
  fails fast, and `Total_elems` cannot resolve conv/affine dims either — give leaves explicit
  dims (e.g. `Operation.init ~o:[n] ~grad_spec:Require_grad`) instead of `Tensor.term_init` when
  the consumer is a windowed spec.
- Padded (`=`-mode) windows: max/tropical-family accumulations lower with clamped window bounds
  and demand NO margins (gh-504: `Row.proj_env.clamp_padded`; interval-based range guards in
  `Assignments.to_low_level`); add-family keeps physical halos committed as tensor-node identity
  (padded convs lower offset-free). The padded dim equation requires the input divisible by the
  stride, and a stride-2 window only clips on the right for window ≥ 5.

## Syntax extensions (%op / %cd)

- Block-tensor delimiters map array `[|…|]` → batch, list `[…]` → output, tuple `(…)` → input
  axis; canonical nesting is array ⊃ list ⊃ tuple. Function-argument and einsum-operand tuples
  keep their OCaml meaning (distinct ppx arms), so `(a,b) ++^ …` is an operand pair, not a stack.
- `%op` inline-record init expressions (`{ w = kaiming normal1 () }`) resolve identifiers at the
  lifted binding's scope, which is the enclosing UNIT parameter: a `let%op f x = ...` with no
  `()` param fails with "Unbound value" (gh-511 tracks the scoping fix) — write
  `let%op mk_f () x = ...` and apply `mk_f ()`, or fully qualify the initializer's DSL idents
  (`kaiming TDSL.O.normal1 ()`). Only the applied head is auto-qualified to `NTDSL.*`; argument
  idents rely on the `open TDSL.O` scope introduced at the unit parameter. Design caveat: the
  no-unit-param form is not generative — `f` closes over ONE shared param created at definition
  time; the `()` idiom makes the construction point explicit.

## Graph construction and autodiff

- A shared subtensor's forward code is embedded in its first-CONSTRUCTED consumer, and
  `Tensor.consume_forward_code` rejects consuming a root whose non-embedded reads are embedded
  in another live root. Consequences: construct (and compile) the consumer that should own a
  shared computation FIRST; with padding in play, the margin-demanding consumer (conv) must also
  compile before margin-blind ones, or the operand's layout locks unpadded. Sibling fragments
  are topologically ordered (owner-first forward, owner-last backward) — the old id-ascending
  order silently zeroed shared paths (regressions `forward_fragment_order.ml`,
  `backprop_fragment_order.ml`).
- The tropical einsum's gradient gate (`cond_rhs1` in `Operation.tropical`, same pattern in
  `einmax1`) is last-write-wins per input position: exact only when windows don't overlap
  (stride ≥ window) — `max_pool2d` with stride < window trains on silently wrong gradients
  (gh-512: verified repro, root-cause analysis — the flat Assignments IR forces a slot-shaped
  gate tensor — and fix routes, argmax-record being the most tractable). Write gradient oracles
  accordingly.
- Silent numeric divergence recipe: build a minimal numpy oracle reading the same safetensors,
  probe stage-by-stage (`Train.set_materialized` intermediates BEFORE `forward_once`, then
  `Context.get_values`), shrink to a tiny `NTDSL.init` repro, and read `build_files/*.cd` —
  use-before-def is visible at the .cd level.

## Lowering, virtualization, indexing

- `Ir.Ops.index_prec ()` is SIGNED (int32; int64 under `large_models`): negative index
  intermediates are well-defined; emit guards in natural signed form. The `a <= b` as
  `Cmplt (a, b+1)` idiom survives only because there is no `Cmple`. Per-node element counts must
  fit int32 unless `large_models`; launch params are bind-validated.
- Value-rewriting passes need executed parity tests, not just structural pins (see CLAUDE.md).
  To exercise a virtualized affine-LHS producer end-to-end, hand-build an `Assignments.comp`
  (einsum result-side scatter specs don't parse; gradients accumulate → stay materialized): wrap
  in `Asgns.Block_comment`, set `embedded_nodes`, force the output materialized, seed inputs
  with `Context.set_values`, then compile→run→`get_values`.
- Big-reduction producers are forced `Never_virtual` by `virtualize_max_inline_reduction`
  (default 16) — remember it when a structural expectation assumes inlining.

## Backends

- Metal shader compiler miscompiles serial `acc[k] = acc[k] + f(i)` loops when pointers derive
  from dynamically-loaded offsets (the pooled `__pool_slots` binding): result = last iteration
  only; hides under API validation (`MTL_SHADER_VALIDATION=1` makes it vanish). Fingerprint:
  loss ≈ correct/batch_size. Workaround shipped as `volatile_scalar_rmw` in
  `arrayjit/lib/c_syntax.ml` (metal config sets it); standalone repro
  `benchmarks/runners/ocannl/bench_metal_bug.ml`, guard `scalar_rmw_accumulation.ml`. Suspect
  this class first for any new metal-only numeric bug with the 1/batch smell.
- Metal `Where` must stay a short-circuiting ternary: MSL `select` is a function call that
  evaluates BOTH branches, so any range guard's deliberately out-of-range read (clamped windows,
  inlined-concat component guards) would still be evaluated. Codegen pins:
  `test_where_precision.metal.expected`, `test_metal_guarded_gather_codegen`.
- Metal buffer binding is the pooled slot-table (`__pools` + `__pool_slots`); raw `gpuAddress`
  casts segfault at dispatch and argument encoders don't fit the binding model. Same-queue
  command buffers overlap over untracked resources: back-to-back runs of the SAME routine need
  the FIFO wait, pipelined (no-sync) timing is unreliable, and `get_values`/`set_values` do FULL
  awaits by design.
- Convention for parallel-codegen work: land Metal → cc → CUDA/HIP; CUDA-only codegen snapshots
  (`.cu.expected`) go stale until that hardware next runs the suite — expect re-promotes.

## Training and performance

- Metal training recipe: `Train.every_non_literal_materialized loss` (kernel fission then cuts
  every cross-nest edge) + `Autotune.tune ~rounds:0 ~timing_ctx:scratch`. `~rounds:0` keeps
  .expected files schedule-invariant (preset seeds preserve reduction order); `?timing_ctx` on a
  scratch lineage is MANDATORY for training comps fed via `set_values` — timing on all-zero
  inputs poisons params through log(0)→NaN, and reinit-after-tune is not airtight (ledger trips,
  Metal device-side races).
- benchmarks/ is the cross-framework parity+timing suite (self-describing safetensors fixtures,
  one-JSON-line runners, loss-trajectory parity gate ~1e-7 fp32 vs pytorch/cpu). The gate
  doubles as a gradient oracle. tinygrad: realize the loss BEFORE `opt.step()` or it recomputes
  from updated weights. The autotune schedule cache persists across processes; compiler changes
  invalidate it by digest.

## Conventions

- Releases use lightweight, un-prefixed git tags (`0.8`, not `v0.8`).
- Prefer the minimal targeted fix over speculative hardening: offer hardening separately as an
  option with its costs, don't fold it into the fix.
