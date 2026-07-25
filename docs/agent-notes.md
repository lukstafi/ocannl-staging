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
  has the full comparison set `<`, `<=`, `>`, `>=`, `=`, `<>` (plus `not`, `where`) since the
  `Cmple` primitive (PR #216); `<=` is IEEE-exact (NaN-false), not a `not <` composite.
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
- `einsum_n_constraints` (the shared constraint-generation helper for binary Einsum, Block/concat
  and ternary Einsum) takes `?lhs_constraints_first` because the branches need OPPOSITE orders and
  no global choice works: binary Einsum passes `true` (the LHS constraint must precede RHS Affine
  constraints so a conv `over` variable resolves against an LHS-specified axis), while Block/concat
  and ternary keep the `false` default (RHS before LHS, or concat rows get prematurely solved to a
  concrete dimension). If a change makes one family's tests go green and another's red, the fix is
  the per-caller flag, not flipping the default.
- Non-termination in the fixpoint solver does not always trip a consecutive-iteration equality
  check — it can oscillate between two states or loop one non-converging constraint. Pin it by
  instrumenting the suspect arm with a counter that prints its operands and exits past a cap.
  Separately, a normalization can be correct-by-spec yet have no `%op`-level "fails-without"
  fixture: forcing enough sibling components to reach the branch makes the equation determinate, so
  re-substitution routes it through a simpler arm first. Cover the reachable determinate sub-case
  with a solver-level fixture and write the reachability gap down rather than contriving a vacuous
  one (`Concat = Dim` multi-residual normalization, PRs #64/#66, is the worked case).
- `test/einsum/test_basis_total_order.expected` prints PASS/FAIL for a battery of `d1 ⊑ d2`
  accept/reject cases, which makes it the direct decision-set-unchanged gate for any
  behavior-preserving relabel or order flip: if it stays all-PASS, no case crossed the boundary, so
  a reversed comparison cannot have slipped through. Freeze it before starting such a change and
  treat any non-wording diff as stop-and-flag.
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
  intermediates are well-defined; emit guards in natural signed form. Guard shapes are
  canonicalized to ONE shape per role: upper bounds are strict `Cmplt` (`idx < bound`, the natural
  operator), lower bounds are direct `Cmple` (`0 <= idx`). Construct new guards in exactly these
  shapes — recognizers match structurally (`schedule.ml`'s Tensorize mask parser and
  breakpoints-from-guards affine view, `c_syntax.ml`'s launch-guard strip), so an off-canon
  encoding silently stops contributing breakpoints or gets rejected. When adding a comparison
  operator to a guard, give every such recognizer an arm for it. Per-node element counts must fit
  int32 unless `large_models`; launch params are bind-validated.
- Value-rewriting passes need executed parity tests, not just structural pins (see CLAUDE.md).
  To exercise a virtualized affine-LHS producer end-to-end, hand-build an `Assignments.comp`
  (einsum result-side scatter specs don't parse; gradients accumulate → stay materialized): pass
  `~name` to `Context.compile` (or wrap in `Asgns.Block_comment` for labeled debug dumps), set
  `embedded_nodes`, force the output materialized, seed inputs with `Context.set_values`, then
  compile→run→`get_values`.
- For a codegen-SHAPE regression (this IR pattern emits this output sequence), build the
  `Low_level.t` plus `traced_store` by hand and instantiate `C_syntax(Pure_C_config(...))` directly
  rather than reaching the shape through `%op`/`%cd`: the DSL virtualizes intermediates and turns
  einsums surjective, and `Tn.update_memory_mode tn Never_virtual` only partly restores control over
  occurrence count and loop nesting. Precedents: `arrayjit/test/test_cross_cse.ml`,
  `arrayjit/test/test_zero_out_codegen.ml`. A DSL-level fixture is a smoke test, not the regression.
- A node-level "what happened at first touch" flag (`zero_initialized_by_code` and friends) cannot
  soundly drive a PER-OCCURRENCE codegen decision, because nothing clears it across the traversal: a
  guard keyed on it alone collapses `Zero_out; Set; Zero_out` to one zero and drops a `Zero_out`
  inside a `For_loop` on every iteration. The shape that works is per-traversal state — a `seen` set
  cleared at the single reset point (`compile_proc`) plus a positional `~in_loop` threaded through
  the recursion, defaulting to `true` for mutually-recursive callers that don't carry it. When a
  codegen decision consults a `traced_array`-style boolean, ask whether it is node-level or
  occurrence-level; they coincide only at first touch on the linear path.
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
- An AC of the form "buffers are allocated with storage mode X" is untestable through the abstract
  cross-backend surface (`type buffer_ptr` hides the representation, and no public operation reads
  the property back). The unlock is a `module type of` refinement in the CONCRETE backend's `.mli` —
  `with type buffer_ptr = Metal.Buffer.t`, sealing the `.ml` to match — which lets a test call
  `Metal.Resource.get_storage_mode` on real `alloc_array`/`alloc_zeros` results without weakening
  the surface other backends rely on (`test_metal_alloc.ml`). A unit test of the
  `storage_mode_for_memory_mode` classifier alone passes whether or not the call sites thread
  `?mode`, so it is not evidence for the integration.
- Before committing a design to a new MSL construct, settle "does it actually run?" with a
  throwaway dune exe calling `Metal.Library.on_device` on candidate snippets and dispatching them:
  passing the Metal compiler proves syntax and types, not the runtime contract (residency,
  addressing model, lifetime). This is how the pooled slot-table binding was chosen over the
  raw-`gpuAddress` table, which compiled cleanly and segfaulted at dispatch.
- Enum and type-mapping changes in the GPU BINDING libraries (ocaml-cudajit and friends) are fully
  verifiable without the hardware, since the conversions are pure: a `test_no_device/` target that
  feeds the conversion well-known plus out-of-range/legacy values through a sexp diff pins the
  behavior on any host (`test_no_device/test_computemode.ml` covers `computemode_of_int` on
  `{0,1,2,3,42}` after a removed legacy CUDA value started raising). Reserve real devices for
  genuinely runtime-dependent behavior. For a binding repo you have no checkout of,
  `gh pr diff --repo <owner>/<repo>` reviews the change by inspection.
- Parallel-codegen work often lands Metal → cc → CUDA/HIP, but that is a default reflecting
  which machine is booted first and used most (the Mac Studio), not a rule — tasks can start on
  CUDA or HIP for load balancing across machines. The durable part: codegen snapshots for a
  backend whose hardware isn't attached (`.cu.expected` etc.) go stale until that hardware next
  runs the suite — expect re-promotes.

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
