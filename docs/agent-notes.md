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
  barrier is the formal fix.
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
