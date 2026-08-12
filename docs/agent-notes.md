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
- Use-site row resolution is opt-in (gh-ocannl-544): an operation result's open row variables
  close DOWN to its arguments' shapes; a use site broadcasts the result in but cannot widen it.
  The resolve-at-use mark (`Row.add_resolve_at_use`, propagated through unification) grants the
  old widening behavior and is registered for: terminal rows (leaves and params), parameter
  initialization cones (`Tensor.mark_init_cone` — init intermediates must FILL the param's shape,
  or broadcasting repeats random values), sampler results (`uint4x32_to_prec_uniform{,1}` — a
  draw fills whatever shape context demands), and the explicit `stretch` identity wrapper (which
  replaced the `0.5 + 0.5` shape-inferred-constant idiom, e.g. pooling kernels `stretch 0.0`).
  The motivating trap (gh-ocannl-540): `Mixed_prec`'s cast twin read as the weight operand of a
  *batched* matmul was widened by the use site to `[batch, out, in]` — a per-batch-row copy of
  one weight. Nothing catches that downstream: every slice holds the same value, so results,
  gradients and loss trajectories are all correct, and the only symptoms are `batch`x memory/work
  and the row symbol appearing in an operand index. A parity oracle whose model has no batch axis
  cannot see a batch-broadcast bug — `mixed_prec_parity.ml` had covered this recipe for a whole
  release; `mixed_prec_twin_shape.ml` and `stretch_resolution.ml` pin the shapes directly.
  Dim variables follow the same policy (`Row.add_resolve_at_use_dim`): unmarked
  `Unconstrained_dim` axes are guessed minimal instead of widening to the use-site GLB — but only
  at stage 7, because stage-6 row closings still push concrete dims through einsum equality
  chains and an eager stage-6 guess-to-1 conflicts with a dim arriving in the same stage
  (box_muller's interior axes were the canary). `At_least_dim` axes always meet their use sites
  (direct indexing is a dim-carrying use).
- That spurious axis is also how a shape defect reaches the SCHEDULER, which is where it actually
  got noticed: `Tensorize`'s role check rejects an operand mentioning the third micro symbol, and
  `Stage`'s insertion point L\* is "the deepest loop carrying an outer-part symbol", so an operand
  that spuriously depends on the row loop pins the cooperative load nest INSIDE it and breaks the
  perfect nest. One shape bug, two unrelated-looking decline families, split exactly by whether the
  seed stages operands (`sk_bk > 0`) or not. When a whole autotune family declines structurally,
  check the operands' index expressions before suspecting the scheduler.

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

- Both digests over lowered code — the analysis cache's key (`Low_level.analysis_digest`) and the
  schedule cache's canonical identity (`Schedule_cache.canonicalize`) — share ONE walk,
  `Low_level.Canonical_render.emit` (gh-ocannl-563). Render a newly added `Low_level.t` /
  `scalar_t` construct there and nowhere else (the matches are exhaustive, so the build breaks);
  a new digest-relevant *fact* is a `Canonical_render.policy` field if it is an identity choice, or
  a consumer's own preamble/companion section if only one of them consults it. The walk owns what
  the two agree on: loop-binder tokens `b<n>`, first-occurrence local-scope alpha, comment
  skipping, `mark_incomplete` on opaque statements. Golden + seam test:
  `test/operations/canonical_render.ml`.
- Why a bespoke renderer rather than `to_doc` / `sexp_of_t`: (1) the digests must be
  ALPHA-INVARIANT — `Indexing.symbol` and `scope_id` are global counters, so sibling lowerings of
  one routine share no symbol numbers and any structural print misses 100% of the time; (2)
  `to_doc`'s node names are NOT CONTEXT-FREE — `get_ident_within_code` pre-passes the whole code
  array to find labels claimed by more than one uid and only disambiguates those, so a node prints
  as bare `x` when alone, which COLLIDES two different routines rendered separately (a wrong hit,
  and for the analysis cache a correctness failure) and makes a fragment's digest shift when a
  sibling fragment changes (per-segment schedule matching needs the opposite); it also mutates
  (`Tn.update_code_name`) and is layout- and config-dependent (`PPrint` width,
  `ll_ident_style`/`output_prec_in_ll_files`). (3) `Low_level.t` derives no `compare`/`hash`
  (`Staged_compilation` holds a closure), and the schedule cache's key becomes an on-disk FILENAME
  (`Schedule_cache.cache_key`/`cache_file`) — so a string digest, not a structural key.
- The `Low_level` analysis cache (gh-ocannl-560) makes sibling candidate compiles share one
  `analyze_proc` result keyed by that digest of the raw lowered code. Its identity policy is the
  OPPOSITE of `Schedule_cache.canonicalize`'s: tensor nodes and static symbols enter by identity
  (`Tn.uid`; symbol ident + the mutable `static_range`/`used_as_extent` facts) because a hit
  reuses the stored code verbatim, while `canonicalize` alpha-renames everything so schedules
  replay across sessions. Anything the analysis consults beyond the code must enter the key —
  `inline_complex_computations` does, since the rmw exemption changes what the coverage /
  multiplicity queries count. Two traps: (1) caches
  that retain lowered code keep tensor nodes (and, via pool finalizers, buffers) alive — register
  a clearer in `Tnode.before_accessibility_snapshot`, or `print_accessible_headers` goldens grow
  phantom "accessible" nodes (this is how the cache was caught); (2) on a hit, still re-run
  `pin_device_written_bounds` — its raising writer-after-settled-reader guard must fire regardless
  of caching.
- `Affine.access.a_path` components are TYPED (`Affine.path_comp`, gh-ocannl-561): `Stmt` indices
  interleaved with `Cond`/`Body` (`If`) and `Rhs`/`Write` (`Set` family), constructor order =
  execution order, so lexicographic comparison is program order within a statement too. Every
  access path ends in `Cond`/`Rhs`/`Write` and nothing extends past `Write`. Consumers must not
  compare bare paths for "same statement" — use `Affine.same_statement` (agreement above the
  final component; the path-level twin of `a_stmt_write` subordination) — and must take top-level
  statement identity via `Affine.stmt_head`, never `List.hd` (a single-statement routine's paths
  start with a marker, not a `Stmt`). History: with bare positions an `If` condition's read
  aliased its guarded body's write (gh-554 round 3), and `read_covered_before` needed a
  prefix-exclusion hack for enclosing writes vs their inlined `Local_scope` bodies — both
  unrepresentable now. Each `Local_scope` occurrence additionally extends the path with an `Arg`
  evaluation position (per-statement counter), and `path_before` deliberately does NOT order
  across sibling `Arg`s: two scope bodies inlined into one statement must neither interleave
  their interior components (a `Seq`-bodied sibling's `Stmt` sorts before a bare-bodied one's
  `Rhs` — a later operand's write would pose as prior to an earlier operand's read; Codex P1 on
  PR #297) nor claim cross-operand evaluation order at all (it would silently depend on codegen's
  scope emission order).
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
- `check_half_prec_constants_cutoff` (`Ops.exceeds_fp16_cutoff`, enforced from
  `Low_level.simplify_llc.check_constant` during lowering, hence backend-independently) is a
  HEADROOM policy, not a representability check: its default 2^14 sits far below fp16's 65504 max
  finite, so a constant it rejects may be perfectly representable. Read "too big for FP16" twice
  before believing it names an overflow — the one message covered two opposite defects at once
  (gh-ocannl-547/548), and fixing either alone just moves the failure to the other. Reduction
  identities are out of scope by construction (`Ops.neutral_elem`: `Max` → `-inf`, `Min` → `+inf`),
  exempted via `Float.is_finite`: they are sentinels arithmetic consumes, exactly representable, and
  every backend converts them per IEEE (`__float2half` / `__double2half` / a `(half)` cast). Attention
  masks fill with `Nn_blocks.default_mask_fill` = `-inf` (per-call `?mask_fill` for the one case that
  needs finite: a mask that can cover a whole row, where `-inf - -inf` would give NaN) rather than a
  large finite magic number, so the fill needs no per-precision tuning.
- The `bf16_ops`/`half_ops` convention of picking inputs that are exactly representable in the
  reduced precision, so printed numbers stay backend-uniform, does NOT extend to transcendental
  results: no choice of inputs makes an `exp` output exactly representable, and backend libm
  implementations disagree in the last mantissa bit (HIP's `exp` gives `2.1215e-1` where cc, metal
  and cuda give `2.1228e-1` — one ulp at 2^-13, found only by running `half_softmax` on real ROCm
  hardware). A reduced-precision golden containing `exp`/`log`/`tanh` output must therefore print
  coarsely enough to sit above an ulp and carry its numeric content in a tolerance comparison against
  a double-precision reference computed in the test, not in the printed digits.
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
  barrier is the formal fix. Query the analysis at the SITE'S arity (`aligned_chains ?max_chain`,
  default 2 = the presets' Grid+Workgroup shape): a batched matmul's chain is batch loops + row +
  column, and under the default cap a rank-3+ site can never match its full chain, so every seed for
  such a site declines on companion coverage — that single decline held gpt2_mini's five FFN-class
  kernels at a 1024-thread launch, 70% of the CUDA step at 1.3% of fp32 peak (gh-ocannl-569). A
  companion that reduces OVER the site's minor axis (the lm_head's max-logits row) trims the common
  prefix below the site's arity and correctly still declines — that one needs fission, not coverage.
- "`Tile_mma` is a barrier" is only half true, and the half that fails is the one barrier elision
  wants. Every rendering form ENDS the intrinsic block with a workgroup barrier, so a staging
  barrier that follows one is always redundant (`Schedule.elide_staged_barriers` drops it, and the
  after-loop one, at any pipeline depth). The LEADING bracket is form-dependent: the fragment-scope
  form (`render_mma_fragment_scope`, the crowned Metal/CUDA shape) emits it ONCE, on the scope
  wrapping the whole anchor loop, not per iteration — so a barrier may never be elided against a
  *following* `Tile_mma`. That is why a depth-1 staged k-block keeps exactly one explicit barrier
  (between its loads and its compute) while the pipelined form keeps none: the pipelined prefetch
  writes the *next* iteration's buffer copy, so the previous iteration's trailing bracket is what
  separates it from its reads. Elision is a synchronization-only transform, so the test that pins it
  is an executed BITWISE comparison against the same schedule with barriers re-inserted
  (`schedule_pipelined_matmul`) — adding barriers is always conservative, which makes that reference
  sound no matter what the elision does.
- The CUDA `cp.async` arm (gh-ocannl-487 phase 2, `C_syntax_config.async_copy`) keeps that
  discipline by never touching the intrinsic's brackets: an async copy is only complete for the
  issuing thread after a wait, and only visible to the workgroup after a barrier that FOLLOWS the
  wait — so the rotor loop's body is uniformly prefixed with `ocannl_cp_async_wait_all();
  __syncthreads();`, re-inserting for the async arm exactly the phase opener
  `elide_staged_barriers` drops for synchronous stores (those are published by the previous
  iteration's trailing bracket; an async copy waited AFTER a barrier is published to no one).
  Wait-all (PTX `cp.async.wait_all` = commit_group + wait_group 0) instead of commit/wait-group
  bookkeeping is what makes the emission per-`Set` opportunistic and safe: any staging statement
  the arm declines (precision conversion, surviving fringe ternary, non-global source, 2-byte
  elements — cp.async needs 4/8/16) falls back to a plain store published by the same barrier,
  and correctness never depends on which statements were accepted. It is also why depth stays 2:
  deeper lookahead needs per-group waits. Eligibility is per tile in `compile_proc`
  (`current_async_tiles`; kernel logging disables it — a logged `Set` reads back what an
  in-flight copy cannot provide). Measured on the RTX 5070 Ti (paired in-process pd1/pd2, 9
  replicates, tf32 fragment-scope form): 512³ f32 pd2/pd1 = 0.97 median within a ~14% spread;
  deep-K 256×256×2048 = 0.92 with all 9 replicates in 0.906–0.946 against ≤4.4% arm spread — the
  overlap genuinely pays where the k_o loop dominates, reversing the portable form's Metal
  ~1.4–1.5× cost and HIP's null.
- A schedule can pass `Schedule.apply`'s validation and still be one the RENDERER cannot express:
  the pipelined-tile checks in `c_syntax.ml` (a read reached outside its rotor loop; a rotor loop no
  longer `Serial`) are positional facts about the final IR, which schedule application does not
  re-derive. Such a check must raise `Schedule_outcome.Cause_at (Backend_codegen, Unsupported …)`,
  not `invalid_arg` — an untyped exception at a compile-side phase is `Fatal` under
  `strict_failure_classification`, so one composed candidate ends the whole search (seen on Metal
  searches over `tile_acr_ma`). `raise_cause` re-renders the same `Invalid_argument` at the public
  `Context.compile` boundary, so typing one costs nothing for hand-written schedules. To probe a
  renderer precondition in a test, do surgery on the applied `optimized` (re-point `pt_rotor`)
  rather than retyping loops — a retyped anchor trips `Low_level.validate_parallel` first and never
  reaches the check under test.
- Autotune fault injection keyed by ATTEMPT INDEX (`Autotune.on_candidate_attempt`) is
  backend-dependent and silently vacuous: how many attempts precede an arm's first *timed* candidate
  varies. On Metal a small matmul's materialize-all arm has a baseline binding no hardware dimension
  (gh-532), and its whole `W_preset` block then dedups against that same digest — six attempts, none
  timed. Count timing runs with `Autotune.on_candidate_preflight` (one per timing run) and inject
  relative to that. Relatedly, hoisted (link-time packed) `Stage` candidates are a CPU family only —
  `matmul_seed_params` proposes `sk_hoist` from its `is_cpu` branch — so any test precondition about
  packed-constant pools is false on GPU backends and has to be stated as an equivalence.
- "Seeded" is not "timed". An autotune family can be enumerated in bulk and rejected in bulk at
  candidate compile, and a count of proposals then reads as coverage it does not have — assert on
  the *timed* counter (`report.mma_timed`, `fiss_sketch_timed`, `split_reduce_timed`), and follow it
  with an executed value check, since a candidate that compiles is not yet one that computes.
- "Timed" is not "tensorized" either, and that failure is worse: a declined `Tile_mma` renders its
  scalar fallback, which compiles and runs, so the candidate is timed, ranked and possibly crowned
  under an `mma-*` label (gh-ocannl-545: 20 of 20 timed bf16 candidates on CUDA were scalar). The
  emission is the source of truth — grep the emitted kernel for the intrinsic
  (`wmma::`/`mma.sync`/`simdgroup_`), or read `C_syntax.mma_census`; `schedule_log_declines=true`
  names the rule that fired. When seeding and emission can disagree, fix the seeding side too, or the
  measurement budget keeps going to schedules that never tensorize: `mma_format_tiles` is keyed on
  the whole `(a, b, accumulator)` format triple, with per-entry arch floors, precisely so that a
  combination a backend supports at one accumulator width but not the other cannot be seeded.
  `mma_staged_layouts` (gh-ocannl-481) is keyed the same way for the same reason: the swizzled
  staged twin is seeded only where the emission can actually read that layout, which on CUDA is
  the uniform-bf16 combination and not fp8 (whose B side has no 16-bit `ldmatrix` form at the
  orientation the staged sketches mint). The census distinguishes `Mma_intrinsics_ldmatrix` from
  `Mma_intrinsics`, so "tensorized" and "fed at rate" are separable in a sweep.
- "Crowned" is not "shipped", and neither is reproducible on a small routine. `Train.tune_placements`
  runs two searches and keeps one artifact, so a family can win the arm that is then discarded whole
  — read `report.best_label` / `best_tensorized` / `mma_best_ms` per arm (the A/B calls `?report`
  for arm A first and ships the smaller `best_ms`), never the fact that some search crowned it.
  Below GEMM-dominated sizes the crown is a lottery: on `mlp_small`/metal five identical cold-cache
  searches crowned four different families in one arm with a 4.5% spread of best times, while the
  arm gap stayed at 57–95% (gh-ocannl-546, benchmarks/report-gh546-metal.md). Conclusions of the
  form "family X wins/never wins here" need repeats; the arm-level verdict does not.
  The arms are independent experiments and are contained as such since gh-ocannl-550: an arm whose
  search raises is a LOSING arm (ranked `infinity`), the other arm's winner ships and stays cached,
  and the failed arm's partial report still arrives in position carrying `terminal_failure` — read
  that (or `partial`) before `best_ms`, because a partial arm's best is a time whose routine was
  never compiled. Before that fix, one arm's late failure destroyed the other arm's finished work
  in-process (the cache entry survived, since `SC.store` precedes the winner replay — so a warm
  cache could still replay it; five of five tf32 `gpt2_mini` runs lost arm A this way). Note where
  it escaped: NOT at the failing candidate — candidate-grade protection absorbed those OOMs as
  `Backend_link` declines — but after the search concluded, when the exhausted device defeated both
  the winner replay and the untuned-default fallback compile behind it. Containment tests do not
  need a device that can fail: `Autotune.on_candidate_attempt` injects one
  (`test/operations/autotune_arm_containment`).
- A timing failure's *phase* decides whether the lineage is condemned, so pre-dispatch validation
  needs its own. `Context.run` validates (poisoned lineage, uninitialized inputs, unsatisfied
  execution dependencies, out-of-range static bindings) before dispatching; inside a `Launch`-tagged
  boundary those failures were unattributable — `classify_failure` returns `None` on every C
  backend — so `classify_raw` made them `Fatal` and the handler poisoned the lineage. A one-line
  user mistake (a `timing_ctx` scratch context missing one of the caller's initializations, which
  its own docs warn about) thus condemned the search *and* the context, with no restore
  (gh-ocannl-564; #536 for why there is no restore). `Schedule_outcome.Preflight` now tags that
  region and classifies as a contained `No_device_writes` decline **without consulting the
  backend** — host-side validation, so a classifier guessing `Writes_may_have_occurred` would
  escalate a failure that provably wrote nothing. Rule for any new boundary around `Context.run`:
  tag what precedes `Ir.Task.run` as `Preflight`, or a fixable mistake reads as device damage —
  **but only the per-candidate half of it may be contained** (gh-ocannl-569, found on HIP). The
  split is `Context.check_lineage_runnable` (poisoned lineage, uninitialized inputs, unexecuted
  dependencies) against `Context.check_launch_bindings` (out-of-range static bindings), and it is
  the difference between a condition that belongs to the *lineage* and one that belongs to *this
  candidate's* bindings. Only the second can fail one candidate while its siblings time cleanly;
  the first fails every candidate of every arm identically, so containing it is silent —
  a search whose serial baseline is not dispatched (**every GPU search**, gh-ocannl-532) then
  declines every candidate for the one reason, times nothing, and `tune_placements` *returns
  normally* shipping the untuned default out of an unusable lineage, with no exception and no
  `terminal_failure`. On the C backends the dispatched serial baseline hits the condition first and
  takes the arm down with the caller's message, which is why every golden encodes the CPU shape and
  CI never saw it. So: **contain the per-candidate half, raise the lineage-wide half outside the
  boundary** — at the candidate site before `Outcome.protect`, and inside the baseline's match
  scrutinee so `condemn` still reads it as pre-dispatch and leaves the lineage usable. Tag the
  hoisted raise `Preflight` anyway: it carries no boundary there, but `search`'s fallback handler
  would otherwise report a validation error under its `Transform` default.
  These causes also resist injection: they belong to the lineage and the bindings, not to a
  candidate, so a genuine one fails *every* candidate at once, which is why
  `Autotune.on_candidate_preflight` exists — though since the lineage-wide half now escapes the
  region, injecting one of *its* exceptions through that hook exercises the containment machinery
  with a realistic payload rather than mirroring where a real one is raised.
- Placement decides which tensorized candidates *exist*, not just how they rank, because
  `mma_tile_for_precisions` keys on the storage precisions of the nodes the site actually reads.
  Under the mixed-precision recipe on a uniform-format backend (Metal's `simdgroup_matrix`: no mixed
  multiply-accumulate) the default-placement arm seeds **zero** mma candidates — the reduced-precision
  cast twins are virtual, so the site reads f32 masters into an f16 destination and no advertised
  tile matches. `Mixed_prec.Twin_materialized` (three small weight casts) restores the whole family
  at the default arm's cost, whereas materialize-all buys it by doubling the kernel count. If a
  reduced-precision cell reports `mma_candidates = 0`, look at the twins before the seeding rules.
- Supplying a `?lowered_transform` bypasses the default annotator entirely (`backends.ml` `compile`
  only calls `Schedule.maybe_default_schedules` in the `None, None` arm), so **any** code that goes
  through that seam is the unscheduled serial form unless it schedules itself. The autotuner's base
  compile is exactly that, which is why its baseline candidate binds no hardware dimension on GPU:
  the whole routine in one work-item. Such a dispatch is unbounded in cost and uninterruptible on a
  device shared with the display — measured 6.9 s/run on Metal for LeNet (winner 35.7 ms) and hours
  on gfx1151, with driver timeouts and a lost display (gh-ocannl-532). `Autotune.tune` therefore
  does not dispatch an unparallelized candidate on a GPU backend at all: `dispatchable` gates both
  the baseline and every candidate, `baseline_ms` is `infinity` there, and if nothing at all gets
  timed the search stores no cache entry and returns the untuned default compile rather than the
  serial incumbent. On CPU backends the serial form runs at full single-core speed and is still
  timed. When timing anything else through this seam, price the serial form before dispatching it.
- The base compile staying unscheduled is a settled decision, not an oversight (gh-ocannl-552): the
  default pipeline is `maybe_default_schedules` — fission then per-segment annotation, so several
  kernels in general, not one `optimized` to rebase candidates on; every candidate family assumes
  the serial zero point; and annotation reads `hardware_limits`, which would bake per-device
  decisions into the cache's `source_digest`. The "did tuning beat the shipped default?" reference
  is `report.default_ms` instead — the config-thresholds fissioned seed reproduces the untuned
  pipeline exactly and its time is attributed by digest (so a seed that dedups against a timed twin,
  the CPU serial baseline included, still reports).

## Backends

- **Device buffers are not GC-reclaimable, and the reason is a table, not GC pressure**
  (gh-ocannl-550). Each backend's private `Slab.pools` (`(device_id, pool_id) -> base`) holds a
  strong reference to every slab it allocated, so no finalizer on a pool can ever run; the one
  function that drops an entry, `Backends.finalize`, had **no caller anywhere in the repo** until
  `Context.release` was added. Measured consequence, from the per-candidate census of a cold tf32
  `gpt2_mini` search: +1 working pool and ~102 MiB per candidate *attempted* — dedups and
  `Backend_link` declines pay in full, because the pool is allocated at link, before the dedup check
  — monotone to 11.9 GB of a 12,227 MiB card, `pools_freed = 0` the whole way. Contrast the same
  run's code modules: 35 loaded, 31 unloaded, live count flat at 3–4. Modules sit behind no such
  table, so cudajit's `cuModuleUnload` finalizer fires fine on an ordinary host heap. So do not
  reach for `Gc.full_major` when device memory grows — it cannot help a rooted table — and do not
  assume a class leaks just because another one does. `Ir.Alloc_census` (config `autotune_log`
  prints it per candidate) separates the four classes: working pools, constant pools, contexts,
  modules.
- A CAS-guarded cleanup must not commit the flag before the cleanup succeeds. `Backends.finalize`'s
  `ctx.finalized` means "the pools were freed", not "a free was attempted": `Backend.await` inside it
  can raise (a device still reporting an asynchronous error, a dead worker domain), and committing
  first made every later release of that context a silent no-op with its pools rooted for the process
  — i.e. it reinstated gh-550's growth on precisely the failure paths where callers catch a failed
  release and carry on. It resets the flag on exception instead; that is safe only because freeing is
  idempotent per `pool_id` on every backend, so a partially completed cleanup does not double-free on
  retry. Any new atomic "done" flag around fallible cleanup wants the same shape.
- **On WSL2 a device-memory bug does not look like one.** The CUDA driver there backs allocations past
  VRAM with host memory, so the same unfixed search that OOMs promptly elsewhere reached **28.8 GB
  requested** on a 12,227 MiB card while `nvidia-smi` sat pinned at 11,879 MiB and reported headroom
  throughout; the observable symptom was thrashing (candidate times 105 ms → 3,563 ms, search wall
  time 767 s vs 105 s fixed), and `CUDA_ERROR_OUT_OF_MEMORY` arrived only at the very end. Two
  consequences: an OOM's *position* in a candidate stream is a property of the box's spare host RAM,
  not of the bug (gh-550's "arm-B candidate 47" landmark reproduced at arm-B ~135 here), and
  `nvidia-smi` is the wrong instrument — `Context.get_used_memory` sums the pool table's requested
  bytes and matched the census to the decimal at every sample.
- Anything that knows an artifact's exact lifetime should call `Context.release` (idempotent, eager,
  finalizer-independent); `Autotune.tune` does it per candidate, bounding a search at
  `beam_width + 2` live candidates instead of one per attempt. Not calling it is never a
  correctness bug, only a memory one. What `release` frees is precisely the pools a context holds
  that its parent does not and that are not per-device constants — so sibling contexts are
  independent (each `compile` mints its own `pool_id`s) but a released context is a dead handle, and
  **release leaves, never interior nodes**: a context compiled from another inherits its buffer
  locations, so releasing an ancestor leaves the descendant resolving a dropped `pool_id`. Unchecked
  precondition, deliberately (refcounting persistent context values would defeat their point).
- **Two classes `release` cannot reach, so "bounded" always needs a qualifier.** (a) Per-device
  constants: it skips every `constant_buffer_cache` key by design. That is right for a shared weight
  and wrong for a hoisted `Stage` candidate, whose `apply_stage` mints a FRESH packed-constant tnode
  per application (`fresh_tile_id ()`), so a CPU search seeding `hoist` sketches grows one constant
  pool per such candidate — measured on `cc` at 1 → 109 constant pools over 181 candidates while
  working pools stayed within 2–6. Not safely fixable in place, because constants are bump-packed
  several to a pool and the first candidate's pool mixes its private tile with the shared operand
  weights later candidates reuse; a safe rule is per-pool purity, i.e. gh-ocannl-565's eviction-policy
  work. (b) A link that RAISES after `allocate_delta` — now handled (`Backends.with_delta` frees the
  delta on the way out), but the shape is worth knowing: allocation precedes backend linking, so any
  new failure point between them leaks a whole routine footprint with no context to release it
  through. When asserting a memory bound, assert on `live_working_pools`, not `live_pools`: summing the
  constant class in makes the assertion fail for a reason it is not about, and on a workload with no
  hoisted candidates it passes while proving less than it looks like.
- Four facts about the allocation seams that each cost a review round to learn, and that any further
  release work will meet again. (1) There are **two** shared allocation sites, not one:
  `Backends.allocate_delta` for a compile's delta, and the `allocate` inside
  `Add_buffer_retrieval_and_syncing` for a `from_host`/`copy` destination not yet in the context. Both
  land in the same pool tables and are freed by the same context `finalize`. (2) `allocate_delta` is
  **not atomic** — it schedules host uploads and can allocate several segments — so a guard wrapped
  around it from outside cannot see a partial delta; the unwind has to live inside, and must `await`
  before freeing because those uploads are asynchronous. (3) Constant-cache entries **point into**
  pools, so unwinding an allocation must drop the entries that allocation inserted before freeing,
  while leaving pre-existing ones (they belong to earlier compiles). (4) Retain-then-raise is the
  standing bug shape in this area: decide what ships *after* the last thing that can raise, or the one
  artifact you deliberately kept is the one nobody can reach. Corollary for reviewing such a change:
  each fix adds a container, a guard or a retention decision, i.e. a new path with the same obligation
  — re-examine the failure paths the fix itself created, not just the ones it closed.
- Fissioned-step segment batches go through the `sequence_segments` seam
  (`Backend_impl.Lowered_backend`): Metal encodes one serial-dispatch command buffer; CUDA/HIP
  stream-capture the launch loop into a graph replayed as one `cuGraphLaunch`/`hipGraphLaunch`
  per step (gh-ocannl-488, config `gpu_graph_capture`). Graph capture bakes kernel arguments, so
  instantiated graphs are cached keyed on every launch-time-varying argument: static-index
  binding values and the merge-buffer position. Two traps encoded there: the merge pool is the
  one pool that can be REALLOCATED IN PLACE (same `pool_id`, new base), so its key component must
  be pointer identity, not `buffer_loc`; and a failed capture leaves the stream in capture mode —
  always terminate via `end_capture` before falling back or re-raising. The legacy NULL stream
  cannot be captured (OCANNL streams are all non-default, so this only bites standalone repros).
  Fallback paths (logging on, capture rejected, config off) are plain per-segment launches —
  same-stream FIFO makes the generic event chain redundant on CUDA/HIP.
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
- A tensor node's precision is its **storage** precision; the precision its arithmetic runs at is a
  separate thing, `C_syntax_config.compute_prec` (gh-ocannl-517). They coincide on the GPU backends
  (native `__nv_bfloat16` / MSL `bfloat`/`half`, and the 16-bit tensor-core shapes that consume
  them) and diverge on `cc`, where every narrow-float operator was a widen/op/narrow round-trip
  anyway: there the narrow floats compute in f32 (`Ir.Numerics.narrow_compute_f32`, on by default),
  so a load widens once and a store narrows once, and an assignment's intermediates keep f32
  mantissa. The rule when touching `c_syntax.ml`: a **declaration**, a kernel parameter or a buffer
  element type takes the storage precision; a **rendered expression** takes `comp_prec` of it. Two
  exceptions are load-bearing. The RNG lane conversions (`uint4x32_to_<prec>_uniform*`) pick both
  their result type and which random bits they consume from the precision they render at — the fp8
  generator is not a rounding of the f32 one — so they stay at storage precision, and the
  scope-locals they write are excluded by a whole-proc scan (`rng_scope_local_uids`), because a
  `Declare_local` carries no value to test. And a `Set` whose value contains no operator renders at
  storage precision, so a copy loop stays a copy instead of a round-trip through f32.
- Convert-on-load/store is what makes the `Vectorized` renderings reachable for 16-bit nodes: the
  lane count comes from the **compute** vector, so the narrow side is a half-width vector, and the
  conversion happens at the memory boundary rather than per lane inside the body (per-lane
  conversion would give the traffic win straight back). Bitwise parity with the serial remainder
  loop is by construction — every fallback arm calls the same scalar conversion the serial path
  does — and `test/operations/narrow_storage_compute.ml` asserts it with `=`, not a tolerance.
- **fp16 is the one narrow format a CPU can compute in natively** (gh-ocannl-516), and whether it
  can is a C-preprocessor fact the OCaml renderer cannot see. `cc` probes the configured compiler
  once per process and reports three states — no `_Float16`, `_Float16` with arithmetic *promoted*
  to float (correct, no lane-count win: x86 without AVX512-FP16), and genuinely native
  (ARMv8.2-FP16, AVX512-FP16) — surfacing the last as `hardware_limits.native_fp16_arithmetic`.
  `Ir.Numerics.fp16_arithmetic` (off by default: it trades mantissa for throughput, unlike
  `narrow_compute_f32`) then makes `compute_prec` leave `Half_prec` alone, so `vec_ext_typ` mints a
  `HALF_T` vector and the lane count doubles. The middle state is why the probe is not a boolean:
  seeding and the cost model must not expect a lane-count win where only the type exists.
- The fp16 FMA is where parity nearly breaks: `fmaf` on `_Float16` operands promotes to float and
  rounds **twice**, while `__builtin_elementwise_fma` on an fp16 vector rounds once. The scalar
  rendering and the vector rendering's per-lane fallback therefore both go through one builtin
  macro, `OCANNL_HALF_FMA`, defined by the same `#if` — so both configurations agree by
  construction rather than by inspection. Any new fp16 op admitted to the vector path needs the
  same treatment.
- **`-march=native` is the wrong flag on ARM and was silently downgrading every CPU kernel.** Apple
  clang accepts it on arm64 and targets a *lower* baseline than passing nothing: 22
  `__ARM_FEATURE_*` macros against 26 with no flag and 33 with `-mcpu=native`, losing
  `__ARM_FEATURE_FP16_VECTOR_ARITHMETIC` among them — so a machine with native 16-bit arithmetic
  probed as one without. `cc_backend_arch_flags` now defaults to `auto`, which asks the target
  which family it is in and probes that family's spelling (`-mcpu=native` on ARM, `-march=native`
  on x86 — where `-mcpu=` is merely an alias for `-mtune=` and would not select the ISA at all).
- The traffic win is real but it favors **fp16, not bf16**, the reverse of gh-ocannl-517's
  expectation. On an M-series at n = 2^22, a bandwidth-bound elementwise add measures 131 GB/s at
  f32, 1.97x that at half storage, and **0.91x** at bf16 (`bin/narrow_storage_bench.exe`); a
  compute-bound control stays below 1x for both, as it must. bf16's round-to-nearest-even narrowing
  is four vector ops against fp16's single NEON instruction, and at stream speed that costs more
  than halving the bytes saves. The route to competitive bf16 is a hardware convert (`BFCVT` on
  ARMv8.6-A, AVX512-BF16 on x86) — but only if it can be shown to agree with `single_to_bfloat16`
  bitwise, NaN payloads included, or the vectorized rendering stops matching its serial twin.
- Benchmark trap: `Context.get_values` walks the whole buffer into an OCaml `float array`, an O(n)
  host-side cost that does **not** depend on storage precision. Timing it inside the measured region
  (as `bin/cpu_vectorization_bench.ml` did until gh-ocannl-517) makes every kernel look equally slow
  — an order of magnitude below the machine's stream bandwidth — and exactly masks any traffic
  difference. Keep readbacks outside the timed region; the `cc` scheduler is synchronous, so no
  separate await is needed.
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
  executed guards are `test/operations/half_ops.ml`, `test/operations/bf16_ops.ml` (operand
  ambiguity) and `test/operations/bf16_builtins.ml` (builtin return types), plus
  `test/training/mixed_prec_parity.ml`.
- MSL's math library has **no `bfloat` overload of any builtin** — `sqrt`, `exp`, `log`, `pow`,
  `fmax`, `fmin`, `fmod`, `trunc`, `rsqrt`, `tanh`, `fma` all promote to `float` and return
  `float`, and unlike C, MSL then rejects the narrowing assignment back to a `bfloat` destination
  ("assigning to 'bfloat' from incompatible type 'float'"). So the bridge belongs on the whole
  math-builtin family, not per operator: `metal_backend.ml`'s `bf16_from_builtin` casts the result
  back (gh-ocannl-549). What is *not* affected, and needs no bridge: arithmetic operators,
  comparisons, the ternary, `!`, and the `0.0bf` literal suffix — MSL's `bfloat` is a native scalar
  type. `half` has the full overload set, so f16 has no such gap. Verify claims like these by
  compiling one-line kernels through `Metal.Library.on_device` rather than by reasoning about the
  spec; there is no `xcrun metal` without full Xcode.
- The same bf16 emission fails *differently* per GPU dialect, which is why one backend's evidence
  misleads about another's (gh-ocannl-549). A float-returning builtin at bf16 is the single root
  site; where the dialect complains depends on where that float lands. MSL rejects the assignment,
  so every placement fails. CUDA/HIP accept it (`__nv_bfloat16`/`__hip_bfloat16` have an implicit
  converting constructor from float), so the materialized placement — which stores each result in
  its own bf16 node — compiles, and only the placement that *inlines* the builtin into a consuming
  bf16 binop fails, on the operand: nvrtc reports a mixed-operand `__hadd` (its bf16 `Add` arm is
  `func "__hadd"`), hiprtc reports `operator '+' is ambiguous ('__hip_bfloat16' and 'float')` (its
  `Add` falls through to plain `+`). A placement-dependent bf16 compile error is therefore a clue
  about *inlining*, not about a fission-introduced mixed type — nothing introduces a float, the op
  table's own `expf`/`sqrtf`/... arms return one.
- The MSL bf16 trap's older half: an *untyped* literal does not fail loudly. `max(0, v)` makes an
  integer overload unambiguous, so it compiles and silently truncates every sub-unit activation to
  0, whereas `max((bfloat)0.0, v)` is a clean "call to 'max' is ambiguous" error. Fingerprint of
  the silent form: loss pinned at exactly ln(#classes) with NO batch-to-batch variation (a
  frozen-weights bug would still vary per batch; an input-independent forward does not). Found by
  the gh-ocannl-476 sweep; `Relu` at `Bfloat16_prec` had fallen through to a catch-all commented
  `Byte_prec, Void_prec`. When adding a precision, audit every `unop_syntax`/`binop_syntax`
  catch-all arm.
- Tensor-node debug names become identifiers verbatim in the emitted kernel, so anything the
  backend also emits as a *name* must be reserved (`ident_blacklist`). Reserve it from the
  backend's own syntax functions, never from the C spellings: `C_syntax.op_syntax_idents` renders
  every (precision, operator) pair over a placeholder and harvests the identifiers, so an override
  cannot drift out of the list. Deriving from `Ops.*_c_syntax` instead described C only and left
  MSL's unsuffixed `tanh`/`exp`/`log`/`sqrt`/`sin`/`cos`/`trunc` free — and those are exactly the
  `Tensor.unop ~op_label` labels, so a GPT-2 gelu declared `device float *__restrict tanh` and the
  call on the next line resolved to the pointer (gh-ocannl-553). A backend's builtins-table keys
  belong in the list too: a node taking one shadows the definition *and* drags it into a kernel
  that never calls it, since `filter_and_prepend_builtins` selects entries by searching the
  rendered kernel for their key. The collision only bites when one kernel holds both the
  declaration and the call, so which backend it fires on depends on fissioning — the guard is
  `test/operations/test_ident_blacklist.ml`, and its section 3 only has teeth under
  `OCANNL_BACKEND=metal` (C spells these with an `f` suffix, so no C compile can exhibit it).
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
- A typed decline is only half of gh-ocannl-533: what the issue asked for is that the SEARCH
  survive it. The rejection fired on `Autotune.tune`'s own base compile — the identity-transform
  capture, historically the one compile in `tune` that raised instead of returning an outcome — so
  it bypassed `try_spec`, the decline census and the partial report, and killed the run with a
  perfectly-classified cause. Two facts worth carrying: the baseline is the one candidate not
  compiled *as* a candidate, and passing `?lowered_transform` bypasses the default annotator, so
  what gets validated there is the unscheduled serial form — the worst case for per-work-item
  scratch, and on GPU never dispatched anyway. It is now declined (`report.baseline_declined`,
  `baseline_ms = infinity`) and the scheduled candidates carry the search; fission plus
  `promote_locals` is what brings a large softmax/CE head back within budget, which is why
  `gpt2_mini hip/tuned` completes while every whole-routine preset declines. In the census it
  carries its own cause and NOT gh-ocannl-543's `Not_dispatched_key "baseline"` — a declined
  baseline is never dispatched either, but recording both would double-count it under a reason that
  is not the one. One refusal, one entry. Guard:
  `test/operations/hip_scratch_tune_survives.ml` (`slow` alias).
- Building a test kernel that actually *has* a big scratch frame takes care: write the `Local`
  array in one loop and read it back in REVERSE in another. A forward read in the same order lets
  the compiler forward each store to its load and delete the array, leaving nothing to reject.
- `Context.get_used_memory` must report OCANNL's OWN allocation (`Slab.used_memory`, or the
  backend's atomic counter) — never the driver's `total - free`. That is device-global: it counts
  other processes and moves in allocation granules, so it cannot see sub-granule effects like the
  liveness planner's arena savings. gh-ocannl-289 fixed this for CUDA; HIP kept asking the driver
  until gh-ocannl-542, where on a gfx1151 APU sharing memory with the display it made
  `buffer_aliasing` report the planner INCREASING the footprint 106496 -> 2072576 B, against
  1556896 -> 1425668 B once measured properly (cc: 1556640 -> 1425540). When one backend's
  numeric assertion inverts while its parity assertions pass, suspect the measurement API before
  the pass under test — and check whether a sibling backend already fixed the same thing.
- Rematerialization (gh-ocannl-498) is a PLANNING pass, not a search: `Context.plan_memory_budget`
  picks `Inline` flips from `Backends.score_footprint` (the arena layout's own bytes) versus the
  recompute-cost bound `flip_candidates` already carries, and compiles nothing to decide. The trap
  it is built around: footprint relief is NOT per-node-local under aliasing. A node whose live span
  was already shared frees nothing by leaving, and inlining one node moves the others' spans — so
  both the solo pass and the cumulative prefix are scored against a real lowering, and a candidate
  with zero marginal relief is skipped rather than taken for free (the gh-ocannl-558 enablement
  lesson, in reverse). Score the layout, never the node's own size.
- The footprint scorer deliberately scores the routine's WHOLE in-context node set, not a context's
  allocation delta, and enumerates by uid rather than in `traced_store` order — otherwise the
  selector's choices would drift with how much of the graph a particular context had already
  allocated, and with hash order across processes. On `cc` the model came out equal to the measured
  `Context.get_used_memory` delta to the byte (1392772 both ways in
  `test/operations/memory_budget`), but treat that as a coincidence of a single-context run: the
  real allocator skips already-held nodes and the driver page-rounds pool bases.
- `Inline`-direction candidates want the CHEAPEST recompute cost first; `flip_candidates` is sorted
  most-expensive-first because the `Materialize` chain of `Train.tune_placements` wants that end. A
  pre-filter cut that forgets to reverse keeps exactly the flips a budget would least want to pay
  for.

## Training and performance

- Metal training recipe: `Train.every_non_literal_materialized loss` (kernel fission then cuts
  every cross-nest edge) + `Autotune.tune ~rounds:0 ~timing_ctx:scratch`. `~rounds:0` keeps
  .expected files schedule-invariant (preset seeds preserve reduction order); `?timing_ctx` on a
  scratch lineage is MANDATORY for training comps fed via `set_values` — timing on all-zero
  inputs poisons params through log(0)→NaN, and reinit-after-tune is not airtight (ledger trips,
  Metal device-side races).
- `Autotune.report.candidates_timed` is NOT comparable across a CPU and a GPU backend, so an
  assertion on it recorded from `cc` is a portability trap (gh-ocannl-543 —
  `autotune_fission_sketch` timed 19 candidates on cc and 1 on HIP for the same chain). The GPU
  rule of gh-ocannl-532 refuses every candidate that binds no hardware dimension, and on a
  two-nest chain that is most of the space: the whole-routine presets dedup to the unscheduled
  base, and the beam's `Split`/`Swap`/`Retype-Vectorized` moves off that base cannot introduce a
  `Grid`/`Workgroup` loop (only `Tensorize` and placement retypes can), so the fissioned preset is
  the only thing left to measure. The refusals now carry a typed census key,
  `Not_dispatched_key of origin` (`baseline` | `candidate` | `beam_move`) — read it (or
  `BENCH_TUNE_REPORT=1`) before concluding a GPU search "found nothing".
- benchmarks/ is the cross-framework parity+timing suite (self-describing safetensors fixtures,
  one-JSON-line runners, loss-trajectory parity gate ~1e-7 fp32 vs pytorch/cpu). The gate
  doubles as a gradient oracle. tinygrad: realize the loss BEFORE `opt.step()` or it recomputes
  from updated weights. The autotune schedule cache persists across processes; compiler changes
  invalidate it by digest.
- That digest covers the compiled result, NOT the diagnostics. When re-measuring a search's
  decline census after changing only an error message or a log line, `rm benchmarks/autotune_cache/*.sexp`
  first — otherwise the second run reports `cache_hit=true`, `timed=0`, `declines=[]` and every
  counter reads zero, which looks exactly like "the problem went away". Fixtures are also not in a
  fresh checkout (`benchmarks/fixtures/*.safetensors` is generated); `gen_fixtures.py` recreates
  them, and they are only valid while `gen_fixtures.py` and `benchmarks/workloads/` are unchanged.
- A benchmark leg belongs to a WORKLOAD, not to a runner (gh-ocannl-551). `BENCH_STATIC_SCALE` /
  `BENCH_GATE_INTERVAL` lived in `bench_mlp` alone, so the gate-cost contract silently had no
  answer for `gpt2_mini` — and a forward-only workload has no loss scale to gate in the first
  place. The resolution to copy when adding a leg: put the flag parsing and step shapes in
  `Bench_harness` (shared), dispatch the step shape on the fixture's `mode` (as the Python
  runners do), add a training workload when the question needs one (`gpt2_mini_train`), and make
  orchestrate report an inexpressible cell as NOT APPLICABLE with its reason instead of omitting
  it. In OCANNL specifics: a parameter has `batch_dims = []` pinned by `Tensor.param`, so a
  positional-embedding table must be output-axis-shaped and placed with an einsum-add. The gpt
  f16 cells additionally needed gh-ocannl-548 (`-inf` mask fill) and gh-ocannl-547 (reduction
  identities out of the fp16 constant guard's scope); the runner-side workaround they replaced —
  pinning the softmax at f32 and materializing the masked scores — is gone, so do not
  reintroduce a pin for a constant the library now keeps representable.

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
- Deleting a file target out from under dune is not a way to force it to re-run: `dune build
  <that target>` afterwards exits 0 having produced nothing (observed on dune 3.23.1 with
  `test/operations/<name>.exe.output`), and `-f/--force` does not rescue it — `--force` only
  re-runs actions attached to ALIASES. Either force the alias (`dune build --force
  @<dir>/runtest`), or run the built exe directly with its cwd set to `_build/default/<dir>`, which
  is exactly the environment dune gives it — the same cwd, hence the same `ocannl_config` search
  root, that makes `dune exec` unusable (CLAUDE.md).
- A record with `[@@deriving sexp]` makes every `.expected` file that prints the parent a hidden
  consumer of its FIELD NAMES, and `rg "\.field_name"` over sources is vacuous against that (sexp
  prints `(field_name value)`, not member access). Before claiming a rename has no serialization
  consumers, grep the sexp shape: `rg -F "(field_name " --glob '*.expected'` (the trailing space
  disambiguates longer identifiers). Budget the resulting promote as expected work, in its own
  commit, after diff-confirming the delta is rename-only.

## Conventions

- Releases use lightweight, un-prefixed git tags (`0.8`, not `v0.8`).
- `ocannl_config.reference` ships with every setting COMMENTED OUT, and the two forms are
  load-bearing: a commented-out setting is `#key=value` with NO space after the `#`, while prose
  (and the verbatim profile-payload blocks at the end of the file) always uses `# `. That is how
  `test_config_consistency` tells documented keys from prose, so a new key documented as `# key=…`
  reads as undocumented and fails the test. Config values may not contain `=` — the parser splits
  on it and rejects a line with two, which rules out payload/config values like `-mcpu=native`; a
  setting that needs one gets a word spelling instead (`cc_backend_arch_flags=none`). A value can
  never be the empty string either: empty means "unset" at every source.
- Prefer the minimal targeted fix over speculative hardening: offer hardening separately as an
  option with its costs, don't fold it into the fix.
- A backend-gated leg must never print a bare `p "<claim>" true` on the backend that cannot run it:
  the golden line is then byte-identical to a verified run's, so neither the transcript nor a
  reviewer can tell the claim was never evaluated (this is how a `Tensorize` leg came to "cover" the
  gh-528 interior-batch bug). The scheduling tests use a file-local `skipped name`, which prints the
  same stdout line — the golden must stay backend-uniform, and dune's `(test)` stanza diffs stdout
  ONLY, so stderr is free — and announces the skip on stderr. `grep SKIPPED` over a run then
  enumerates exactly what that hardware did not verify. The other honest form is putting the
  condition in the label itself (`"… (skipped: non-C backend)"`), which distinguishes the golden
  line; a bare `true` whose label is indistinguishable is the one to reject.
- Executed parity checks need a nonzero guard on the REFERENCE, not just the comparison: a fragment
  mapping that reads outside a staged block, a candidate kernel that never ran, and a reference
  whose own setup collapsed all produce all-zeros, and zeros compare equal to zeros. The convention
  is a file-local `nonzero name a` that raises, applied where each reference array is produced —
  once per producer, not per comparison. Guard the reference side only: a zero candidate against a
  nonzero reference is already a `false` in the golden, which is more diagnosable than an exception.
- A tolerance cannot reject an input-independent forward if the reference itself does not move:
  every leg sits at one constant and every parity line reads `true`. `benchmarks/orchestrate.py`'s
  `loss_moved` is the model; in-tree, require the reference's own spread to exceed the tolerance it
  gates (`mixed_prec_parity`, `precision_policy_parity`) or that distinct inputs give distinct
  outputs (`gpt2_dry_run`'s positions-differ). "All finite" is not such a guard — all-zeros is
  finite.
