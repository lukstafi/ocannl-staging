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
- Padded (`=`-mode) windows: max/tropical-family accumulations lower with clamped window bounds
  and demand NO margins (gh-504: `Row.proj_env.clamp_padded`; interval-based range guards in
  `Assignments.to_low_level`); add-family keeps physical halos committed as tensor-node identity
  (padded convs lower offset-free). The padded dim equation requires the input divisible by the
  stride, and a stride-2 window only clips on the right for window ≥ 5.

## Syntax extensions (%op / %cd)

- Block-tensor delimiters map array `[|…|]` → batch, list `[…]` → output, tuple `(…)` → input
  axis; canonical nesting is array ⊃ list ⊃ tuple. Function-argument and einsum-operand tuples
  keep their OCaml meaning (distinct ppx arms), so `(a,b) ++^ …` is an operand pair, not a stack.
- `%op` inline-record init expressions (`{ w = kaiming normal1 () }`) are bound under the
  generated `open TDSL.O`, including when there is no unit parameter (gh-511). The no-unit-param
  form is not generative: `let%op f x = ...` closes over ONE shared param created at definition
  time. Use `let%op mk_f () x = ...` and apply `mk_f ()` when each model instance needs fresh
  parameters; the `()` idiom makes that construction point explicit.

## Graph construction and autodiff

- A shared subtensor's forward code is embedded in its first-CONSTRUCTED consumer, and
  `Tensor.consume_forward_code` rejects consuming a root whose non-embedded reads are embedded
  in another live root. Consequences: construct (and compile) the consumer that should own a
  shared computation FIRST; with padding in play, the margin-demanding consumer (conv) must also
  compile before margin-blind ones, or the operand's layout locks unpadded. Sibling fragments
  are topologically ordered (owner-first forward, owner-last backward) — the old id-ascending
  order silently zeroed shared paths (regressions `forward_fragment_order.ml`,
  `backprop_fragment_order.ml`).
- Per-(result, reduced-position) facts in `%cd` grad code belong in a product-space intermediate
  (`*_pspace` suffix): its shape unifies with `Shape.product_space_shape` (result rows +
  contracted axes appended; fixed-index axes pinned to 1 — pinned projection means never a
  product axis even at extent > 1; ≤ 1 reduced-over row variable, else Shape_error), and its
  identity projection is `Indexing.prod_project_for`, which skips dim-1 axes and pairs the rest
  with product components by extent (first-fit — layout order need not match product order, since
  all accesses go through the same pure pairing); leftovers on either side raise at lowering. An operand-slot-shaped gate (`_rhs1`) is last-write-wins under
  overlapping windows — the gh-512 wrong-gradients bug; `tropical`/`einmax1` gates are now exact
  for stride < window and independent RHS2 indices, with an `=:|| eq (t1, t1)` validity mask for
  clamped windows whose output is genuinely -inf (executed oracles:
  `test/operations/overlapping_window_grads.ml`). Unary einsum specs with conv indices
  (`@^^ "o<+k => o"`, `++` alike) fail projection solving before any of this — pre-existing
  gh-515 — so overlap coverage goes through binary `@^+` with a zero kernel.
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
- A GPU schedule must cover EVERY materialized-writing nest of the routine, not only the one the
  pipeline builds. Launch dimensions are kernel-global, so `Low_level.validate_parallel` rejects any
  companion write (a bias/relu tail; the elementwise statements an aligned-merged fission segment
  carries) not nested under loops covering every active `(kind, slot)` pair — and on GPU there is no
  all-serial fallback, so the whole candidate fails to compile. Do not relax the rule: an uncovered
  dimension means every hardware index executes that write. Annotate instead, from
  `Schedule.aligned_chains` — the default annotators' cross-nest analysis exposed as data (which
  loops may carry geometry, already trimmed so that chain position *k* denotes the same thread
  coordinate in every linked nest), which lets a pipeline supply its own per-position geometry while
  alignment stays `schedule.ml`'s rule. A pipeline that instead leaves companions bare and counts on
  `Fuse_epilogue` absorbing them has NO surviving form when the fusion declines; that cascade
  (gh-ocannl-521) had every GPU backend seeding tensorized candidates in bulk and timing none of
  them. Residual, shared with the shipped zeroing geometry: a tensorized nest's workgroup slot is the
  opaque `Tensorize` lane, so a per-lane companion reads cells other lanes of the same simdgroup
  produced — safe only because the threadgroup is exactly one simd width; a cross-nest simdgroup
  barrier is the formal fix.
- "Seeded" is not "timed". An autotune family can be enumerated in bulk and rejected in bulk at
  candidate compile, and a count of proposals then reads as coverage it does not have — assert on
  the *timed* counter (`report.mma_timed`, `fiss_sketch_timed`, `split_reduce_timed`), and follow it
  with an executed value check, since a candidate that compiles is not yet one that computes.

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
- Reduced-precision *literals* are dialect-specific and do not transpose between backends. `0.0h`
  is a clang extension and valid MSL, but not CUDA C++ — nvrtc rejects it with "user-defined
  literal operator not found" (gh-ocannl-518, the half `Relu_gate`). On CUDA/HIP write the zero as
  `__ushort_as_half((unsigned short)0x0000U)` (bf16: `__ushort_as_bfloat16`), and prefer the
  intrinsic comparisons (`__hgt`/`__hlt`) over operators: mixing a `__half`/`__nv_bfloat16` with a
  literal of another arithmetic type is separately ambiguous under nvrtc/hiprtc, since the type's
  implicit conversion operators make the overload sequences indistinguishable (see the bf16
  comments in `cuda_backend.ml`/`hip_backend.ml`). Same family as the MSL `bfloat` trap below —
  a reduced-precision literal or overload that is fine in one dialect is a hard error, or worse a
  silent truncation, in another. Such bugs only surface with that vendor's hardware attached; the
  executed guards are `test/operations/half_ops.ml` and `test/operations/bf16_ops.ml`, plus
  `test/training/mixed_prec_parity.ml`.
- MSL's math library has no `bfloat` overloads (`max`, `fma`, ...). Render bf16 arithmetic by
  promoting to `float` and casting back — `(bfloat)max(0.0f, (float)(v))`, as `FMA` already did.
  The trap is that an *untyped* literal does not fail loudly: `max(0, v)` makes an integer overload
  unambiguous, so it compiles and silently truncates every sub-unit activation to 0, whereas
  `max((bfloat)0.0, v)` is a clean "call to 'max' is ambiguous" error. Fingerprint of the silent
  form: loss pinned at exactly ln(#classes) with NO batch-to-batch variation (a frozen-weights bug
  would still vary per batch; an input-independent forward does not). Found by the gh-ocannl-476
  sweep; `Relu` at `Bfloat16_prec` had fallen through to a catch-all commented `Byte_prec,
  Void_prec`. When adding a precision, audit every `unop_syntax`/`binop_syntax` catch-all arm.
- `test/config/ocannl_config` pins `backend=cc`, so `dune runtest` never exercises GPU codegen —
  a Metal/CUDA-only rendering bug passes a fully green suite. The bf16 bug above was already
  covered by `test/training/mixed_prec_parity.ml` (its "loss trajectory parity within 0.1" check
  would have caught a zeroed forward); it had simply never run on a GPU backend. Run
  `OCANNL_BACKEND=metal dune runtest` (the env var is an explicit dune dependency, so it re-runs)
  before trusting a backend-specific codegen change.
- Parallel-codegen work often lands Metal → cc → CUDA/HIP, but that is a default reflecting
  which machine is booted first and used most (the Mac Studio), not a rule — tasks can start on
  CUDA or HIP for load balancing across machines. The durable part: codegen snapshots for a
  backend whose hardware isn't attached (`.cu.expected` etc.) go stale until that hardware next
  runs the suite — expect re-promotes.
- HIP scratch (private segment) is budgeted **per work-item, independent of launch geometry**, and
  a kernel over budget aborts the HSA queue instead of failing cleanly (gh-ocannl-533). The
  post-link validator in `hip_backend.ml` (`validate_scratch_budget`) declines it first, as
  `Resource_exceeded Thread_scratch`. Measured on gfx1151/ROCm 7.14/WSL2: the cutoff is
  `ceil(pss/64)*64 * max_threads_per_multiprocessor * multiprocessor_count <= 4 GiB` — 104832 B
  accepted, 104848 B rejected; #533's 163856 B is far over. Traps worth remembering: the 4 GiB cap
  is NOT queryable (it is enforced by the WSL WDDM thunk, `wsl::thunk::ComputeQueue::UpdateScratch`),
  `hipLimitStackSize` is 1024 and has nothing to do with it, and hipcc separately refuses frames
  over 262136 B. Disable with `ocannl_hip_scratch_validation=false` where the model doesn't hold.
  Guard: `test/operations/hip_scratch_budget.ml` (`slow` alias).
- Building a test kernel that actually *has* a big scratch frame takes care: write the `Local`
  array in one loop and read it back in REVERSE in another. A forward read in the same order lets
  the compiler forward each store to its load and delete the array, leaving nothing to reject.

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

## Build and test mechanics

CLAUDE.md holds the workflow rules; these are the dune/OCaml mechanics behind them, narrow enough
that they earn a lookup rather than always-loaded space.

- `(copy_files ...)` creates PASSIVE rules: they do not fire just because you build a sibling target
  in the same directory — only when listed in that target's `(deps ...)` or requested explicitly. A
  rule consuming copy_files output must therefore declare it. And validate a `(mode promote)` target
  from a clean state (`dune clean && dune build @alias`): stale `_build/` intermediates can satisfy
  an undeclared dep, so an incomplete build passes while the artifact is wrong. Assert content
  (size, object counts), not mere existence.
- Dune roots at the OUTERMOST ancestor holding a `dune-workspace` (failing that, a `dune-project`)
  and ignores dot-directories, so from a worktree under `.claude/worktrees/` the main checkout wins
  and the worktree is invisible to dune: targeted commands fail with `Don't know about directory
  .claude/worktrees/...`, while a bare `dune build`/`dune runtest` quietly builds and tests the
  PARENT branch. `scripts/setup-ocaml-env.sh` writes a one-line `dune-workspace` at the worktree
  root, restoring it as the root with its own `_build`. The step tests the ancestor DIRECTORIES
  rather than git topology, since a checkout can nest inside another checkout that is itself a
  linked worktree living anywhere, and `--git-common-dir` then names the primary checkout, not the
  one dune would root at. That file is generated per worktree and gitignored, never committed —
  being the outermost, a tracked copy at the repo root would shadow every worktree's and pin them
  all back to the parent (the script reports `FAIL` for a `dune-workspace` in any ancestor, which
  it cannot override from below).
  With it in place, `--root .` and `dune promotion apply` are no longer needed from a worktree;
  `tools/promote.sh` remains the Windows path, for the CRLF stripping. Worktrees placed outside the
  repo need none of this, but see no `ocannl_config` on their ancestor path.
- A record with `[@@deriving sexp]` makes every `.expected` file that prints the parent a hidden
  consumer of its FIELD NAMES, and `rg "\.field_name"` over sources is vacuous against that (sexp
  prints `(field_name value)`, not member access). Before claiming a rename has no serialization
  consumers, grep the sexp shape: `rg -F "(field_name " --glob '*.expected'` (the trailing space
  disambiguates longer identifiers). Budget the resulting promote as expected work, in its own
  commit, after diff-confirming the delta is rename-only.

## Conventions

- Releases use lightweight, un-prefixed git tags (`0.8`, not `v0.8`).
- Prefer the minimal targeted fix over speculative hardening: offer hardening separately as an
  option with its costs, don't fold it into the fix.
