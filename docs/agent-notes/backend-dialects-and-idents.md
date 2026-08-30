# Backend dialects, identifiers and bindings

Where one GPU dialect differs from another, what a generated name may collide with, and how the
backend config functor binds its overrides.

Part of the agent notes; the [index](../agent-notes.md) carries the scope discipline and the other
files.

- Metal shader compiler miscompiles serial accumulations, and API validation hides it
  (`MTL_SHADER_VALIDATION=1` makes it vanish). Two spellings are affected: the original
  `acc[k] = acc[k] + f(i)` device-memory RMW can leave only the last iteration (fingerprint: loss
  ≈ correct/batch_size), and gh-ocannl-731 showed that codegen's replacement scope-local
  accumulator can instead diverge by a data-independent additive constant.
  `volatile_serial_accumulation` in `arrayjit/lib/c_syntax.ml` (Metal sets it) therefore renders
  device reads inside the accumulating update and its controlling guards through expression-level
  `volatile` pointer casts; accumulator declarations, opening reads, vectorized/packed paths and MMA
  reads stay plain (gh-ocannl-820). The volatility census (`Context.routine.volatility`) reports per
  routine which localized sites and device-memory RMWs received that form. Standalone repros:
  `bench_metal_bug.ml` (RMW form) and `bench_metal_bug_local.ml` (localized form, plus a
  one-factor-at-a-time matrix and the tax measurement) under `benchmarks/runners/ocannl/`; executed
  guards: `scalar_rmw_accumulation.ml`, `reduction_accumulator_residency.ml`, and `rope_test.ml`.
  Suspect this class first for a Metal-only accumulation bug that disappears under shader
  validation.
- What that matrix refuted is worth knowing before proposing a narrower predicate (gh-ocannl-782,
  M4 Max / macOS 26): the pooled slot table is NOT the trigger — building the pointers from literal
  offsets straight off a kernel parameter, with no dynamic load anywhere, miscompiles identically —
  and neither is `__restrict`, nor a device memory barrier, nor where the preceding device store
  lands (moving it to an unrelated cell keeps the defect; removing it altogether is what stops it,
  which no kernel can rely on). The two things that do stop it are the `volatile` accumulator and a
  `volatile` READ pointer; an accumulating loop that dereferences no node pointer at all never
  miscompiled. So the only reproducer-backed narrowing axis excludes accumulations that read no
  device memory, which are rare and usually constant-folded — measured, the tax is 1.06x on a
  memory-bound per-thread reduction, 2.15x accumulator-bound, 4.1x on a single-threaded
  scalar-loss reduction. The `volatile`-source form costs 1.03x on that last shape; gh-ocannl-820
  adopted its expression-level equivalent after the matrix stayed green row-for-row.
- Metal `Where` must stay a short-circuiting ternary: MSL `select` is a function call that
  evaluates BOTH branches, so any range guard's deliberately out-of-range read (clamped windows,
  inlined-concat component guards) would still be evaluated. Codegen pins:
  `test_where_precision.metal.expected`, `test_metal_guarded_gather_codegen`.
- Metal has no `double`, and f64 stays rejected in `typ_of_prec` — a declared double buffer must
  fail rather than be silently degraded; only scalar expression casts render as `float`. So a
  backend-agnostic test must never reach for `double` just to have a second precision:
  `digest_identity_flips` flipped `default_value_prec` to f64 to probe a code-borne cache-key knob,
  and aborted the whole metal `test/operations` run (gh-ocannl-632). Half is the portable choice.

- Metal buffer binding is the pooled slot-table (`__pools` + `__pool_slots`); raw `gpuAddress`
  casts segfault at dispatch and argument encoders don't fit the binding model. Same-queue
  command buffers overlap over untracked resources: back-to-back runs of the SAME routine need
  the FIFO wait, pipelined (no-sync) timing is unreliable, and `get_values`/`set_values` do FULL
  awaits by design.

- Emitted float constants go through `C_syntax.c_float_literal`, the single site that turns a
  `Low_level.Constant` into kernel text, and it owes three things at once (gh-ocannl-623): a radix
  point or exponent, so the token is a *floating* literal and not an integer one the cast happens to
  convert (`%.16g (-0.)` was `"-0"`, hence `+0.0`, a live corruption on every C-family backend);
  enough digits to round-trip, since 16 do not recover every double (`0.1 +. 0.2` and `max_float`
  both print as a *different* double at `%.16g`, and hosted constant inits inline arbitrary host
  values through here); and spellings for `INFINITY` / `(-INFINITY)` / `NAN`, which MSL provides and
  the CUDA and HIP preludes `#ifndef`-define. What it deliberately does NOT emit is a precision
  suffix: the literal stays double-typed and `convert_precision`'s cast narrows it, which is one
  rounding of the exact host double — an `f`-suffixed decimal rounds the decimal straight to float
  and disagrees on a value sitting on a float tie. The one place that reasoning has a hole is worth
  knowing, because it is invisible from the C backends: the cast only IS a narrowing where the
  dialect has a `double`, and **MSL does not** — Metal rounds the emitted decimal itself, at parse
  time. Round-tripping is not exactness, so "round the value to f32" and "round a decimal near it
  to f32" can differ, and they differ exactly on an f32 tie (host takes ties-to-even; the dialect
  follows whichever side the decimal fell on), which makes the emitted digit count decide the
  value. Hence `is_f32_tie` -> a `%h` hexadecimal literal, exact by construction so neither reading
  has anything to round; C99, CUDA, HIP and MSL all accept the form, and only ties are spelled that
  way. Guard: `test/operations/float_literal_forms`, bitwise at f64/f32/f16 plus the emitted
  tokens, and its three tie cases fail on Metal without the hex spelling.
- Two of those three obligations are not C's: an IR dump wants a floating literal that round-trips
  for the same reasons a kernel does, and the `.ll`/`.cd` printers (`Low_level.to_doc_cstyle`,
  `Low_level.to_doc`) used to render a `Constant` with a bare `%.16g` of their own — so `2.` showed
  as `2`, `-0.` as the integer `-0`, and a constant whose 17th digit mattered was displayed as its
  16-digit neighbour. That is the debug surface you reach for when chasing this exact class of bug,
  which is what made it worth more than tidiness (gh-ocannl-713). The radix-forcing and the
  `%.17g` retry now live in `Utils.decimal_float_literal`, shared by `c_float_literal` and both
  printers; the C-dialect spellings (`INFINITY`, the tie's `%h`) stay in `c_float_literal`, since a
  dump is not C and its specials keep `%.16g`'s `inf`/`nan` words. Guard:
  `test/operations/ll_printer_constants`, the dump-side twin of `float_literal_forms`.
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
  The GPU dialects reserve the C++ keywords on top of the C ones: CUDA, HIP and MSL are all parsed
  by a C++ front end, so `C_syntax.cpp_keywords` (one shared table, added to all three) covers
  plausible labels like `class`, `new`, `operator`, `bool` and `this` that plain C leaves free.
- Routine names are the *other* identifier class reaching the emitted source, and they do not go
  through the node-name machinery at all — a node's colliding label is forced into `n<id>_<label>`
  and every local carries a `v<scope>_` / `wred_` / `__rmw_` prefix, but a routine name was emitted
  verbatim, so `Block_comment ("asm", ...)` produced `void asm(` and the compiler's rejection
  surfaced as "this is a bug in OCANNL … file an issue with the generated .c file", naming
  everything except the name that caused it (gh-ocannl-686). `C_syntax.kernel_ident` is the fix and
  the rule: apply it once at each backend's `compile`/`compile_batch` entry, *before* anything
  derives a file name or a symbol, so the emitted header, the `dlsym` / `cuModuleGetFunction` /
  `hipModuleGetFunction` / `new_function_with_name` lookup and the `.c`/`.cu`/`.hip`/`.metal`
  artifact all name the same thing; `compile_proc` refuses a name its caller did not mangle rather
  than emitting one. It is the identity on any legal non-reserved identifier, which is what keeps
  schedule-cache identities and existing goldens from churning. The name a routine is *known* by —
  the routine record, the `.cd`/`.ll` sources, calibration rows — deliberately keeps the caller's
  spelling; only the C-family artifacts carry the mangled one. Guard:
  `test/operations/reserved_routine_names.ml` (executes each colliding routine, not just parses it).
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

- A backend's `C_syntax_config` binds what it inherits at `include Pure_C_config` time, and that has
  bitten three ways: emission code defined ABOVE the backend's `typ_of_prec` override captures
  Pure_C's C spelling (half renders as `HALF_T`); a module-level function whose name matches a config
  field is shadowed by the include (hence `cc_backend.ml`'s `_setting` suffix convention); and
  overriding one member of a paired default (`compute_prec` without `accum_prec`) silently keeps the
  other's default pairing. Define overrides before the code that reads them, and restate both halves
  of a pair.
