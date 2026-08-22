# Precision, narrow formats and CPU SIMD

The storage-vs-compute seam, the software codecs, accumulator widths, and vector codegen quality.

Part of the agent notes; the [index](../agent-notes.md) carries the scope discipline and the other
files.

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

- Metal has no fp8 *type* either, but fp8 works: e5m2 stores as a byte and computes in f32 through
  the `Builtins_metal` software codec, i.e. Metal takes gh-ocannl-517's storage/compute seam for
  that one format (`compute_prec`), the way `cc` takes it for every narrow float. The codec is
  bit-identical to `builtins.c`'s for all 2^32 floats and all 256 codes — verified exhaustively
  off-tree, which is how a hand-written float codec should be checked, not by sampling — and it is
  written in integer/bitcast form on purpose: Metal compiles with fast math by default, under which
  the infinity and NaN branches of a float-arithmetic codec are not reliable. fp8 rounding is portable
  too, but only because ours was changed to theirs — see the next entry.
- fp8 (e5m2) NARROWING is one rounding everywhere, and getting there meant changing ours, not
  theirs (gh-ocannl-646). A float-to-e5m2 conversion decides four things, and the software codec
  (`builtins.c`, `Builtins_cc`, `Builtins_metal` — and, through `Ops.single_to_fp8`, the HOST side
  of every backend) decided all four differently from `__nv_fp8_e5m2` / `__hip_fp8_e5m2`: ties away
  from zero instead of to even, subnormals flushed instead of rounded into (so code `0x01` was
  unreachable by narrowing), the sign of a zero dropped, and finite overflow going to infinity
  instead of saturating to 57344. Since the host side is backend-independent, CUDA and HIP each had
  a host-vs-device split inside one tensor. The codec now does what both vendors do. They disagree
  with each other on exactly two inputs — an already-infinite input (CUDA saturates, HIP keeps it
  infinite) and the sign of a NaN (CUDA drops it) — so those are asserted only up to what all four
  agree on.
  Two reusable lessons from how this was settled. Verify a codec against the HARDWARE, not the
  vendor docs: a kernel sweeping all 2^32 float bit patterns against a candidate runs in seconds on
  either GPU box, and that is what turned "I believe both use RNE" into an exact difference set.
  And when both sides of a parity check run the same code — cc and Metal narrow on the host and on
  the device through this one codec — the check is vacuous by construction, so
  `test_fp8_codec_parity` also pins a GOLDEN of the narrowed values, which is what those two
  backends can actually fail.
- A test that means to exercise a NARROWING conversion has to pin the source precision, or it
  exercises nothing. Two separate mechanisms make the conversion disappear, and both were hit
  writing `test_fp8_codec_parity`. Precision inference flows the destination's precision BACKWARDS
  into the source, so `dst:fp8 =: src` lowers to a byte copy between two fp8 buffers with the
  narrowing having happened on the host when the source was filled — pass `~top_down_prec:false`
  AND `Tnode.update_prec src single`. And a virtualized source inlines its cells as literals, at
  which point the conversion is a constant expression the BACKEND COMPILER folds on the host, so
  a device-side defect cannot show — `Train.set_materialized` the source. Both failure modes look
  identical from the outside: the test passes on every backend, which is what one wants to see.
  The only reliable checks are to read the emitted kernel (`dst[i] = single_to_fp8(src[i])`, source
  declared `float *`) and to run the mutation — with HIP's guard forced off the leg must FAIL, and
  it did not until both mechanisms were closed.
- Narrowing f64 to fp8 must not go through f32 (gh-ocannl-648). Rounding twice moves a double
  that sits just off an f32 tie ONTO that tie, and the second rounding then breaks it by a rule the
  first has already made wrong — so `single_to_fp8((float)x)` disagreed with the GPU fp8 types,
  which convert straight from the double. Not a tie-rule question and not new: under the old
  tie-away rule the same seam disagreed on the mirror inputs (`x-eps` instead of `x+eps`).
  `Ops.double_to_fp8` is the one-step codec, used by the C backends' `Double_prec` arm and by the
  host side — where it matters most, since an OCaml float IS a double, so every host-side fp8 write
  was double-rounding. Verified against `__nv_fp8_e5m2` over 17.2 billion finite doubles (all 2^32
  top-halves crossed with four low-half patterns, ties included) and against the f32 codec over all
  2^32 f32-exact doubles. Metal needs none of it: its `double` is `float`.
- ROCm miscompiles `(__hip_fp8_e5m2)(float)` for magnitudes around 4e-25 to 3.3e-24, returning up
  to 2^-14 where the answer is zero (gh-ocannl-647) — an out-of-range shift, `shift mod 32` landing
  back in range; the exhaustive sweep localizes it to exactly four f32 exponents. CUDA is correct
  there, and so is our software codec. It IS guarded, but opt-in: under
  `prefer_backend_uniformity` HIP's float-to-fp8 narrowings route through
  `ocannl_single_to_fp8_uniform`, which pre-rounds everything below half the smallest subnormal to
  a signed zero — exact, since those magnitudes round to zero anyway, so outside ROCm's window it
  changes nothing. The default still emits the platform's own cast (working around a vendor bug in
  our codegen means carrying it until someone remembers to remove it). `test/config/ocannl_config`
  sets the flag, so `test_fp8_codec_parity`'s two underflow legs are REAL assertions in the suite,
  announced as `SKIPPED on hip` only in a flag-off run. The guard covers both narrowing sites — the
  conversions AND the operator bridges, which narrow an f32 result back to fp8 — through one funnel
  (`fp8_from_float`); guarding only the conversions was the first version, and a review caught it.

- A tensor node's precision is its **storage** precision; the precision its arithmetic runs at is a
  separate thing, `C_syntax_config.compute_prec` (gh-ocannl-517). They coincide on the GPU backends
  for the formats those have as types (native `__nv_bfloat16` / MSL `bfloat`/`half`, and the 16-bit
  tensor-core shapes that consume them), and diverge on `cc`, where every narrow-float operator was
  a widen/op/narrow round-trip
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
- The register-tiled `Tile_mma` rendering is on the same seam (gh-ocannl-575): gates, lane count
  and the C-tile registers follow `comp_prec`, operands bridge at the memory boundary
  (`vec_bridge` identity = the old memcpys, so f32 emission is unchanged), and the accumulator
  narrows ONCE per cell after the whole k extent. Since gh-ocannl-639 the serial fallback narrows
  once per nest too (below), so narrow-storage parity vs the fallback no longer needs narrow-exact
  inputs when the reduction scopes coincide. Packing `Stage`s
  take `tile_prec` (exact widenings only) to fold the widening into the pack; seeding resolves the
  same `Numerics.cpu_compute_prec` the emission uses — change either side only through that
  helper, or "timed is not tensorized" returns for narrow sites.
- **A reduction accumulator's WIDTH is policy and its RESIDENCY is unconditional; its narrowing
  POINTS are schedule** (gh-ocannl-639, gh-ocannl-693). The plain serial fallback of an
  accumulation nest holds the accumulator in a scope LOCAL at `acc_prec` and stores once after the
  nest, implemented not by new emission but by locally rewriting the nest at codegen into the
  `Local_scope` form virtualization gives virtual accumulators
  (`C_syntax.try_localize_serial_reduce`) and rendering that — `scope_prec_of` and the
  `Set_local`/`Get_local` arms already carry the widening. The rewrite is NOT precision-gated: it
  used to bail when `acc_prec` was the identity on the storage precision, which left every f32
  reduction no schedule op reached accumulating in the output node's global memory, one
  read-modify-write per step — and on Metal `volatile_scalar_rmw` pinned it there by construction,
  since its trigger predicate is verbatim the localization opportunity. At identity precision the
  widening half is vacuous and the rewrite is exactly value-neutral, so there is nothing for a gate
  to protect. Codegen is the ONLY localizer of a materialized accumulator (`optimize` rejects such
  a scope, gh-ocannl-681).
- **The rendered forms ONE reduction loop can take**, which is the space a property test over a
  single reduction has to cover (gh-ocannl-664). `pp_ll`'s `For_loop` arm dispatches on the loop's
  axis kind, and every arm that can serialize falls back through `try_localize_serial_reduce`:
  `Serial` -> localized scope, else plain serial loop; `Unrolled` -> localized scope (the repeated
  bodies update the scope local), else repeated bodies each doing a global RMW; `Vectorized` ->
  `try_vectorize_reduce`'s SIMD accumulator grid + epilogue, else (accumulating body) the localized
  scope or plain serial loop — never the pragma'd loop, whose independence assertion a loop-carried
  accumulation does not satisfy; `Workgroup_reduce` -> `try_warp_reduce`'s shuffle tree, else the
  hardware binding, else localized scope / serial; `Workgroup` -> hardware binding, else localized
  scope / serial; `Grid` -> the pool-rendered parallel loop or the hardware binding, whose fallback
  is the plain `serial_loop` and NOT the localizing one — the one arm that can serialize without
  localizing. Deliberately left so: an unbindable, non-parallel-eligible `Grid` level over a
  reduction axis would be a cross-thread race wherever it is reachable at all, and no schedule op in
  the tree produces one, so making the dispatch uniform there would be an untestable change.
  Orthogonally,
  the `Set` arm's `try_register_tile` C-tile and `Tile_mma` hold accumulators too. Every form except
  the two RMW ones (plain serial, unrolled repetition) keeps the accumulator in a register for the
  whole nest, so a property test's invariant is: all forms agree on the value, and only the two RMW
  forms touch the node more than twice.
- **What forces one of the two RMW forms**, i.e. the declines a property test must be able to
  provoke: `debug_log_from_routines` (a `Local_scope` body renders with `log_set_locals:false`, so
  localizing would silence the per-iteration trace — the SIMD and tensorized renderings bail under
  logging for the same reason); an update mentioning an RNG conversion (gh-ocannl-517: the
  conversion picks which random bits it consumes from the precision it renders at); and
  `peel_accum_nest`'s structural refusals — a level holding more than one statement, a guard that is
  not `pure_index_guard`, a cell that is not invariant across the peeled levels, an update outside
  the `accum_update_parts` grammar (only `Add`/`Mul`/`Max`/`Min` and the `FMA` form, with the
  accumulator read a DIRECT operand of the top operator), and, for a scope-form base, updates under
  mixed operators or a write to another node inside the body.
- **A DEAD level (`to_ < from_`) is never peeled**, and that refusal is load-bearing rather than
  tidiness. A dead loop's body performs no accesses at all — the routine-interface walk propagates
  liveness as `live && to_ >= from_`, so a node reached only under one is absent from the parameters
  and need not be allocated — while every form the peel licenses reads and writes the accumulated
  cell OUTSIDE the levels, unconditionally. Peeling one would invent accesses the program does not
  make, possibly naming an identifier the interface never declared; it is the same convention
  `drop_dead_loop_accesses` keeps for the affine metrics and virtualization keeps by dropping dead
  loops outright ("mint phantom parameters for identifiers only dead code renders"). Because
  `optimize` drops them, ordinary lowering cannot deliver a dead loop to codegen — a post-optimize
  transform can, which is why the refusal lives in the shared `peel_accum_nest` (covering the
  schedule mints) plus the rendered level's own bounds in `try_localize_serial_reduce`, which the
  peel never sees. Pinned by `test/operations/peel_dead_level.ml`, live twin included.
  Refusing to peel is only half of it: the level then has to RENDER, and the `Unrolled` arm repeats
  its body `to_ - from_ + 1` times through `Base.List.init`, which RAISES on a negative length
  rather than answering the empty list — so the count is clamped at zero, zero repetitions being
  exactly a dead level's access-free meaning. That abort was reachable before gh-ocannl-693 for a
  dead `Unrolled` level whose body is not a recognized accumulation; refusing to peel the
  accumulating ones widened its reach, which is how it surfaced.
- **A peeled guard must be CONFINED to the levels being peeled** — every symbol it mentions is
  one of them, and it mentions at least one. `rebuild` keeps a guard around the
  accumulating update only, so the localized form performs its opening load and closing store
  OUTSIDE it — right when the guard's truth is not fixed for the whole nest, wrong when it is.
  A guard invariant across the peeled levels is fixed for the whole nest, so hoisting the accesses
  out of it turns "this instance performs no access" into a load and a store; across an enclosing
  HARDWARE axis that is a data race, not merely wasted work. For
  `Workgroup w -> Serial k -> If (w < 1) (acc[0] += x[k])` every lane would load `acc[0]` and write
  its unchanged local back, so a lane reading before lane 0's store and writing after it silently
  discards the reduction — the same 1/N fingerprint as the Metal RMW miscompile, from a different
  cause. "Varies with a peeled symbol" is NOT the right predicate: a MIXED guard like `w + k <= 0`
  varies with the peeled `k` and still selects among lanes. Nor is an empty symbol set safe — a
  guard mentioning no peeled symbol is fixed for the whole nest. **The gh-490 runtime-extent guard
  is NOT constant-bounded** — worth knowing, because assuming it was cost a review round:
  `Assignments.extent_guard` (assignments.ml:225) emits
  `Cmplt (Embed_index (Iterator index), Embed_index (Iterator sym.static_symbol))`, whose bound is a
  STATIC symbol, a kernel parameter bound at launch. (`Schedule`'s Pad guards ARE constant-bounded;
  the two shapes are easy to conflate.) A static symbol cannot select among enclosing loop
  iterations, but the peel cannot tell it from a loop index on its own, so the caller certifies via
  `~invariant`: codegen passes its `idx_params`, and a caller passing none merely declines more —
  which for narrow storage would also drop the gh-639 accumulator width, not just the residency.
  `peel_dead_level.ml` carries all four shapes. Keeping such a guard
  around the whole scope instead of declining would also be sound and would localize more; it needs
  the peel to report its outer guards separately, which is wider than the correctness fix.
- The interaction with Metal's `volatile_scalar_rmw` needs no special case, and that is worth
  knowing before adding one: the shadow keys on the emitted `Set` reading its own node at a cell
  invariant across an enclosing SERIAL loop, and localization lifts the `Set` out of exactly those
  loops (`peel_accum_nest` runs outermost-first, since `pp_ll` recurses top-down), so at a fully
  localized site the predicate is already false. Where the peel was blocked at an outer level — a
  sibling statement, a data-dependent guard — the store stays inside an invariant-address loop and
  the shadow still fires, correctly: the per-iteration read-modify-write is genuinely still there at
  that level. It joins virtual scopes,
  `try_vectorize_reduce`'s epilogue and `try_register_tile`'s C-tile. The peel accepts
  Serial/`Unrolled`/`Vectorized` levels (autotune proposes `Unroll` over any Serial loop of extent
  <= 8 and Retype-`Vectorized` over reductions; a `Vectorized` level rides into the scope and
  `try_vectorize_reduce` recognizes the `Set_local` update form, folding its chains into the scope
  local with no storage round-trip — its scalar TAIL also folds into the wide total before the
  single store), sees through the pure-index guard shape `If (i < bound)` (gh-490
  symbolic extents — data-dependent guards are NOT transparent), hoists through a scope-form base
  (a `Set` whose value is already the accumulation `Local_scope`, reusing its id — accepted ONLY
  when every update fits the mint grammar under ONE reduction operator,
  `Low_level.scope_updates_reduce_op`: hoisting is licensed by the reduction shape, and both a
  general recurrence like `local := local - x` and an individually-valid-but-MIXED sequence like
  `local += x; local *= y` must keep their per-iteration narrowing), and also serves
  hardware-annotated loops the backend serializes for lack of a hardware index (cc's
  `Workgroup_reduce`) — standalone via the dispatch fallback, and NESTED via the peel's
  `extra_level` predicate, which is codegen-only (a schedule mint wrapping a hardware-annotated
  loop in a scope would break backends that bind the dimension). A merge-buffer read
  (`Get_merge_buffer`) is NOT self-dependence of its node — it is a separate read-only staging
  buffer — so `p =+ p.merge`-style updates stay recognizable accumulations. `Schedule.Partition` of a recognized nest mints ONE scope spanning its
  segment loops (an index-set specialization is not a partial-reduction boundary), and
  `rewrite_loop` descends into `Local_scope` bodies so minted-scope interiors (partition segments,
  an outer materialized unroll's inner loops) stay addressable by later schedule ops. The walk that
  LOCATES a loop has to reach exactly as far as the one that REWRITES it, or a probe reports absent
  a loop the very next op happily rewrites (`partition_breakpoints` did, gh-ocannl-668): both come
  from one walker, `Schedule.find_loops_env` — `find_loop`/`find_loops` are its unit-environment
  instances and `partition_breakpoints` threads the enclosing loop ranges through it — and the
  reach has TWO dimensions, both of which cost a review round. As deep: `~in_scopes` (default true)
  descends into scopes. As WIDE: `rewrite_loop` rewrites every copy, so a first-match probe speaks
  for a fraction of the op — a materializing `Unroll` leaves one copy of the inner loop per step,
  each with the outer index substituted by a different constant, so copies of ONE source guard flip
  at different points and `partition_breakpoints` must return their union (a first-copy answer
  leaves the siblings' guards mixed after the `Partition` rewrites them all). Same rule for
  legality: `loops_independent` combines the verdicts of every binding, worst-of. The autotuner's
  own enumeration obeys the same law (gh-ocannl-666): `Autotune.collect_loops` descends scope
  bodies and treats binder-sharing mint copies as ONE decision (one proposal per binder, exactly
  as `rewrite_loop` rewrites every copy), so a materialized unroll or partition no longer hides
  the inner loops from the beam's later rounds; `collect_serial_triples` stays statement-level on
  purpose — `Tensorize`'s `Workgroup` lane loop is refused inside a scope by `validate_parallel`
  (which `op_legality` does not decide: races, not scope nesting), and mint interiors reduce into
  one loop-invariant cell, so no viable micro-kernel triple can sit there anyway.
  `apply_split_reduce` is the one caller taking the first statement-level match, and for a contract
  reason: it inserts its combine statement at statement level, which only sequences correctly if
  the reduction runs there too. A MATERIALIZED unroll never reaches codegen as bare copies:
  `Sched.Unroll ~materialize:true` itself rewrites a recognized accumulation nest into the scope
  form — that is where the provenance lives, since a codegen pass looking at adjacent same-cell
  `Set`s cannot tell unrolled copies of one assignment from two user-authored assignments, whose
  separate stores (and separate narrowings) are their semantics (`accum_width`'s 256+1+1 leg pins
  the boundary). The whole nest recognition — Serial/Unrolled levels, pure-index guards,
  raw-update or scope-form base — is ONE function, `Low_level.peel_accum_nest` (over
  `accum_update_parts`), shared by the transform and the emission precisely because three review
  rounds found them drifting one capability at a time (guards, scope-form bases, the logging
  decline); extend nest recognition only there. The
  widening is inert wherever `comp_prec` is the identity (f32/f64,
  GPU backends, `narrow_compute_f32=false`, native fp16), and it declines twice more: under
  `debug_log_from_routines` (a `Local_scope` body renders with `log_set_locals:false`, so the
  rewrite would silence the per-iteration trace — the per-step `Set` form is the traceable one,
  and every tensorized rendering already declines under logging), and on updates mentioning an
  RNG conversion (the conversion picks its result type AND which random bits it consumes from the
  precision it renders at, so an f32-precision scope would change the draw, not widen it —
  `narrow_rng_nesting`'s reduced-uniform leg pins this). Two traps for tests in this area: (1) cross-schedule BITWISE parity on
  non-storage-exact sums additionally needs the same narrowing points — a k-blocked schedule
  stores storage-precision partials at every `bk` boundary by construction, so give the packed
  leg a whole-k tile (`tile_mma_narrow`'s gh-639 leg uses `bk = n`); (2) discriminating inputs
  must DRIFT out of storage exactness — a zero-mean operand random-walks small enough that every
  bf16 partial sum stays exact and per-step narrowing is invisible (`accum_width.ml`'s policy-off
  negative control is the canary).
- **On GPU the accumulator residency follows the backend's tensor-unit formats, per backend**
  (gh-ocannl-663): `C_syntax_config.accum_prec` — the width a recognized reduction accumulator
  resides at given the storage precision — feeds the try_widen gate AND `scope_prec_of`, whose
  reduction-shaped scopes a codegen census (`C_syntax.accum_scope_ids`, per `scope_id`, verdicts
  from `Low_level.accum_local_update_parts`) resolves at accumulator width, so schedule-minted
  scopes (materialized `Unroll`, `Partition`) and virtual accumulators match the widened serial
  fallback on every backend; the codegen-minted scope registers itself there. The per-backend
  table: CUDA widens bf16→f32 (its mma legs hold f32 per-lane registers whole-k — NVIDIA has no
  bf16 accumulate) and fp8→f32; HIP widens only fp8 (RDNA WMMA has genuine bf16/f16 accumulator
  variants and the uniform triples are seeded, so bf16 serial legs deliberately stay narrow —
  width-uniform with the mma legs); Metal's `accum_prec = compute_prec` (fp8→f32); cc's likewise
  (the CPU accumulator IS a compute intermediate). `narrow_compute_f32` (already in the
  schedule-cache key, gh-ocannl-568) reaches a GPU accumulator only where policy-off can restore
  per-step narrowing SCHEDULE-UNIFORMLY: fp8 on CUDA/HIP (nothing tensorizes fp8 destinations).
  CUDA's bf16 residency is structural like Metal's fp8 — the mma accumulate is hardware-f32, so a
  policy-narrowed serial leg would resurrect the schedule dependence. Scope INITS (a `Set_local`
  not reading its own local — the inlined image of a separate source assignment) render at
  `comp_prec` and convert once into the residency, preserving each source assignment's own
  narrowing (the adjacent-accumulations provenance boundary); virtualization's guarded-read
  updates `Where (index-only cond, update, Get_local id)` classify as reductions via
  `Low_level.accum_local_update_op` — a recognizer deliberately separate from both
  `accum_local_update_parts` (whose `(op, contrib)` licenses rebuilding an unguarded update) and
  `scope_updates_reduce_op` (the hoist license). fp16 is
  everywhere width-uniform at f16 (native accumulate in every seeded triple); aligning it with
  `fp16_arithmetic` would need seeding restrictions or an f32-accumulate uniform-f16 emission and
  remains open. Two traps: `compute_prec`/`accum_prec` bind at `include Pure_C_config` time, so
  overriding one without restating the other silently keeps the default pairing — a startup
  width assert in the `C_syntax` functor catches the narrow direction; and `Workgroup_reduce`'s
  warp-shuffle rendering still hard-errors on narrow accumulators (an error, not a width
  divergence — extending it to accumulate at `accum_prec` is a separable follow-up).
- **A "packmma" timing is not evidence that anything tensorized.** A `Tile_mma` whose register-tile
  preconditions fail renders the scalar fallback and the run still reports under whatever the
  variant was named — the column extent below the compute vector width is the easiest way in (at
  f32/16-byte vectors the crossover is n = 4; at f16 it is n = 8; on a wider machine the crossover
  follows the DEGRADED width, i.e. the 32-byte floor, not the configured one), but a narrow
  `vector_bytes`,
  mixed operand compute precisions, transposed-B storage, and `debug_log_from_routines` all decline
  too. Check `C_syntax.mma_census` (flip `mma_census_enabled` around the compile) rather than
  trusting the label; `bin/narrow_gebp_bench` prints the census per line and warns when any
  statement declined, and `schedule_log_declines=true` gives the per-rule reason. This is the same
  "timed is not tensorized" hazard the seeding note above raises, seen from the bench side.
- `bin/narrow_gebp_bench` takes its blocking factors as arguments (`[bm] [bk]` positionally, or
  `--bm=`/`--bk=`), defaulting to 64/256. The packed variants need `n mod bm = 0` and `n mod bk = 0`
  — an n meeting neither still runs the unblocked naive variant, so an arbitrary extent (the sort a
  register-tiling review actually asks about) can be measured against something.
- **Negative zero is what breaks a "bitwise equal to the scalar twin" claim** (gh-ocannl-615). Two
  spellings normalized it, both fixed but both easy to reintroduce: a scalar-to-vector splat written
  `((vtyp){0} + x)` returns `+0.0` for `x = -0.0` (IEEE `(+0.0) + (-0.0) = +0.0`), so use
  `C_syntax.vec_splat` — `x - (vtyp){0}`, the exact identity on every input; and a float constant
  printed with `%.16g` alone comes out as the C *integer* literal `-0`, i.e. `+0.0` once cast, so
  print through `C_syntax.c_float_literal`. Neither shows up in ordinary parity tests: `Float.equal`
  reports `-0. = +0.`, and a divergence in the sign of a zero product only survives into a result
  whose accumulator is itself a signed zero. `test/operations/vec_signed_zero` is the regression, and
  its splat legs need an accumulator preloaded with `-0.0` (a zeroed one absorbs the sign) — which is
  why they are an explicit `=+` into an initialized tensor rather than a `schedule_mma_matmul` leg.
  The literal is the CROSS-BACKEND half — confirmed to have corrupted CUDA host data too, not just
  cc — so its leg deliberately runs on every backend while the splat legs are CPU-gated; a
  `Vec_extensions`-only test would have missed it. **A splat must be arithmetic-free, not merely
  value-preserving**: `vec_splat`'s first fix, `x - (vtyp){0}`, is the identity on both zeros but
  still quiets a signaling NaN (`0x7f800001 -> 0x7fc00001` under gcc `-fsignaling-nans`; that it
  usually survives is only the optimizer folding the subtraction away, which nothing requires — the
  same "the compiler will probably do the right thing" dependency `vec_fma_builtin` refuses for
  `a * b + c`). It renders an initializer of `lanes` copies of a bound scalar temp instead, which is
  why the callers bind the operand first. sNaN cannot be tested end to end — host values cross as
  OCaml doubles and the narrowing quiets it — so the structural pins keep both arithmetic spellings
  out by name.
- **gcc rounded fp16 FMAs twice, and that is visible, not theoretical** (gh-ocannl-621).
  `OCANNL_HALF_FMA` had two arms: clang's `__builtin_elementwise_fma`, which rounds once at fp16,
  and everyone else's `FLOAT_TO_HALF(fmaf(...))`, which rounds at float and again at fp16 — so on a
  target with genuine fp16 arithmetic, gcc disagreed with clang and with every GPU backend's
  single-rounding `__hfma` / `fma(half, …)`. It now takes `__builtin_fmaf16` where the ISA has the
  instruction (`__AVX512FP16__` or `__ARM_FEATURE_FP16_VECTOR_ARITHMETIC` — exactly what
  `cc_backend`'s fp16 probe calls `Native`), which is also 3-5x fewer instructions, because the
  promoting arm widens and narrows every lane inside the loop. The divergence is not a corner case:
  a single-rounded fp16 FMA and one promoted through float differ on about one triple in ten
  thousand (42039 of 4.1e8 searched), and on one in ~29000 even when all three operands are
  *normal* (3393 of 9.9e7). The tempting dismissal is wrong — `float`'s 24 bits are the `2p + 2`
  that makes double rounding innocuous for a single multiply or add, but an FMA's exact `a*b + c`
  can need far more than 24 bits and the guarantee does not extend to it. A search over well-scaled
  inputs near 1.0 finds zero divergences and reads as a false all-clear. Guard it on the ISA macro,
  never on `__has_builtin`, which always answers yes for `__builtin_fmaf16`: without the
  instruction gcc emits a call to `fmaf16()`, which a glibc need not export at all (verified here
  as a link error).
- **A vector accumulator update must reach the compiler as ONE vector operation** (gh-ocannl-614,
  gh-ocannl-621, fixed). gcc -O3 register-allocated the per-lane `fmaf` loop catastrophically: on
  the packed GEBP shape it unrolled the k-loop and spilled the whole C-tile — 18 insns / 8 vector
  FMAs / 0 stack refs per k step at `-O2` becoming 200 / 4 vector + 48 scalar / 8 at `-O3`, a ~9x
  throughput loss at every storage precision (`bin/narrow_gebp_bench` packmma: 112 → 12.6 GFLOP/s
  at f32, 94 → 10.4 at f16; the earlier claim that f32/bf16 were spared predates the extent-adapted
  tile width). The cause is the lane subscripting itself, not the unrolling: straight-line lane
  stores, a lane-indexed scratch array, a `(vtyp){...}` constructor of per-lane `fmaf` calls, and
  `#pragma GCC unroll 1` all spill the same way, while every form that arrives as one vector op
  costs the `-O2` count at `-O3`. `C_syntax.vec_fma_builtin` therefore adds middle arms to
  `vec_acc_fma` — target whole-vector FMA builtins mirroring clang's `__builtin_elementwise_fma`,
  with the per-lane loop still last. Not `dst = a * b + dst`, which allocates just as well but is
  only maybe-contracted: under `cc_backend_fp_contract=off` gcc emits a mul and an add, and
  `schedule_mma_matmul`'s full-mantissa leg (the only bitwise leg whose products are inexact, hence
  the only one that can see this) then fails.
- **The per-lane arm fails in TWO ways, and the second one is silent.** Besides spilling it can
  SCALARIZE: SLP declines to reassemble the lane loop, so one lanes-wide FMA becomes `lanes` scalar
  ones — a lane-count arithmetic loss with no stack traffic to give it away, which is why the
  census below counts vector and scalar FMAs separately rather than "FMAs". gh-ocannl-621 swept
  every reachable width this way (static census of the innermost FMA-carrying loop of the emitted
  micro-kernel, `-O2` vs `-O3`, insns / vector FMAs / scalar FMAs / stack refs), and the verdict is
  not uniform:
  - **gcc/aarch64 is the worst-affected target and it is every Linux ARM box's default** (Apple
    clang takes the `__builtin_elementwise_fma` arm and never sees any of this). f32 × 4 lanes:
    294 / 8 / 48 / 58 at `-O3` against the builtin's 26 / 16 / 0. f64 × 2 lanes scalarizes
    completely at *both* levels — 0 vector FMAs, 32 scalar.
  - x86 f64 × 2 lanes (16-byte vectors) scalarizes the same way at both levels; f64 × 8 (AVX-512)
    goes 28 / 16 / 0 → 417 / 8 / 96 at `-O3`.
  - f32 × 16 (AVX-512) is the one width where the per-lane arm is *fine*: 31 insns against the
    builtin's 28, no spill, no scalarization. Its row is robustness, not a measured win.
  - At fp16 the vector arm is not where the cost was; see the `OCANNL_HALF_FMA` note below. Once
    that macro rounds once, SLP reassembles the per-lane fp16 loop into exactly what the explicit
    builtin emits, so the fp16 rows are robustness rather than a measured win — but their guards
    have to be the *same* target question as the macro's, or the vector body would round once
    against a scalar peel rounding twice.
  - `vec_acc_combine`'s `Max`/`Min` loop still keeps the per-lane form (no builtin matches
    `fmaxf`'s NaN semantics) — if a `Vectorized` reduction or a wide-vector kernel benches ~10x
    off, this is the first thing to check, and `cc_backend_optimization_level=2` still confirms it
    in one run.
- **`-U__FMA__` does NOT disable the builtin arm of a generated kernel** — `<immintrin.h>`, which
  the prelude includes under `__AVX2__`, re-defines `__FMA__` through `#pragma GCC target("fma")`
  in `fmaintrin.h`, and the definition survives the matching `pop_options`. An A/B run that way
  measures the same arm twice and reads as "the spill is gone"; it cost gh-ocannl-621 a false
  negative before the preprocessed output was compared. Force the per-lane arm by deleting the
  `#elif` lines from the emitted `.c` instead, and keep a positive control: building
  `bin/narrow_gebp_bench` at 97e7d286 (gh-614's parent) still reproduces 12.57 GFLOP/s against
  HEAD's ~128, and its kernel censuses 199 / 52 / 6 at `-O3`.
- **Widths that no local hardware can execute still get checked, three ways.** gh-ocannl-621's
  AVX-512, AVX512-FP16 and aarch64 rows were written on a machine with none of them (QEMU's TCG
  implements neither AVX-512 nor AVX512-FP16 — `query-cpu-model-expansion` on `-cpu max` reports
  every `avx512*` flag false — and no qemu-user was installable without root). What is checkable
  without the hardware: (1) the arm compiles under the target its guard names, and renders exactly
  one fused instruction at `-ffp-contract=off`, with the operand order readable off the asm;
  (2) the builtin is the same call gcc's own `<immintrin.h>` inlines for `_mm512_fmadd_ps` and
  friends, so its semantics are the ISA's; (3) the register-allocation census above, which is a
  compile-time property and needs no hardware at all. A cross toolchain for (1) and (3) on ARM is
  two commands without root: `apt-get download gcc-aarch64-linux-gnu …` then `dpkg-deb -x` into a
  prefix. What that leaves unverified is a wrong *signature*, which is why the new rows carry a
  second guard the shipped AVX/AVX2 rows do not — `__has_builtin`, which on gcc tracks enabled
  target features exactly, so a compiler that spells the builtin differently falls through to the
  per-lane arm instead of failing the kernel compile. (`__has_builtin` is useless for
  `__builtin_fmaf16`: it always answers yes, and without the ISA feature gcc emits a call to
  `fmaf16()`, a symbol glibc does not necessarily export — verified here as a link error. That one
  is guarded on the feature macro.)
- **"No such hardware" is a claim about a machine, not about the project — so name the machine.**
  The rows above were written from an Arrow Lake-HX box, where AVX-512 is fused off across the
  whole hybrid part, and the note recorded that as if it held everywhere; the machine that actually
  benchmarks CPU work here is Zen 5, which has AVX-512 F/VL/BW/DQ/BF16 and ran both AVX-512 rows
  correctly on the first attempt. Agents work in worktrees on several boxes, and this file
  deliberately carries no machine facts, so an unqualified "cannot be run here" reads as a
  project-wide limit and suppresses exactly the check it should have prompted. Check with
  `gcc -march=native -E -dM -x c /dev/null | grep AVX512` and say which host answered. What DOES
  hold fleet-wide: no AMD part implements AVX512-FP16, so those rows stay compile-checked until an
  Intel Sapphire-Rapids-or-newer P-core part joins the fleet — ARM reaches native fp16 through the
  NEON rows instead.
- **The auto SIMD width is 64 bytes on an AVX-512 target** (gh-ocannl-621 follow-up), probed the
  way `simd_flags`' first stage probes AVX2 — asserting what the configured target already has,
  never requesting more — and gated on that AVX2 stage having fired, so `cc_backend_simd_flags=none`
  still answers 16. Worth 1.7x on the packed f32 GEBP of `bin/narrow_gebp_bench` at n = 512 on
  Zen 5 (225.7 vs 130.5 GFLOP/s; f16 189.6 vs 108.1, bf16 140.6 vs 95.1), checksums identical.
- **A single width per machine would make a wider machine emit LESS vector code, and "the widest
  that fits" is still not enough.** Every explicit-SIMD rendering declines below one full vector,
  so at 64 bytes an f32 loop of extent 8..15 — and a `Tile_mma` column extent in that range — falls
  to the serial or scalar path that the same code vectorizes at 32. And at extent 40, 16 lanes
  cover 32 columns and peel 8 while 8 lanes divide it evenly: the wide machine running the narrow
  one's kernel plus a scalar tail. So the width is RANKED over a ladder
  (`Backend_intf.simd_lane_ladder`, halving to a floor of `min vector_bytes 32`): the loop
  renderings minimize trips (`extent / lanes` vector steps plus `extent mod lanes` scalar
  iterations — one instruction per body op either way, so no fitted constant is needed), and the
  register tiling folds the ladder into the peel-cost search it already ran over `rn`. The
  register-pressure cap stays keyed on the MACHINE's width: stepping down does not shrink the
  register file, which is why n = 40 still comes out ahead on the wider machine (118.0 GFLOP/s at
  a 4x5 tile of 8-lane vectors against the 32-byte machine's 103.3 at 4x1). The floor is not
  cosmetic — degrading below the machine's pre-widening width would newly vectorize loops that
  render serially today, and a vector accumulation reassociates, so that is a numerics change
  rather than a scheduling one. Autotune's seeding pre-filter calls the same function, or it would
  withhold candidates the renderer would in fact tile.
- Computing fp16 in fp16 on a *promoted* target is a ~18x loss against f32-compute-over-fp16
  (measured, same bench) — the reason `fp16_arithmetic` is ignored off-native and pure-f16 seeds
  gate on `hardware_limits.native_fp16_arithmetic`. The decisive pure-f16-vs-f32-GEBP measurement
  on genuinely native hardware (NEON) has NOT run yet; see the gh-575 proposal doc's pending note.
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

- **Every GPU backend compiles with fast math**: CUDA passes `--use_fast_math`, HIP `-ffast-math`,
  and MSL defaults it on — `cc` is the exception (opt-in `cc_backend_fast_math`). So a device-side
  non-finiteness test must be a RANGE COMPARE of a runtime value (`-3e38 < x && x < 3e38`); `x <> x`
  and `x - x = 0` fold to a constant, silently disabling an overflow gate (the shape
  `Mixed_prec.gated_scaled_update` needs). It is the same reason `Builtins_metal`'s fp8 codec is
  written in integer/bitcast form rather than float arithmetic.

- **An `external` typed `int array` carries no length, so a stub reading fixed fields must check
  `Wosize_val` first** (gh-ocannl-688). `builtins.c`'s uint4x32 helpers take an OCaml array and read
  lanes 0..3 out of it; nothing in the OCaml type says the caller passed four. An under-length array
  is then an out-of-bounds read that is *usually invisible* — it picks up adjacent heap words and the
  wrong random number is discarded — which is why such a mismatch can sit in the tree indefinitely.
  `ocaml_array_to_uint4x32` now raises `Invalid_argument` on any arity but 4; `arrayjit_copy_with_padding`
  is the older instance of the same check. When adding a stub that reads a fixed number of fields off
  an OCaml block, validate the arity at the boundary — the type system is not doing it for you.
- **Fingerprint: SIGBUS / `KERN_PROTECTION_FAILURE` inside `caml_c_call` under a `camlFoo$entry`
  frame is an FFI over-read at module-initialization time, not a JIT or W^X problem.** The macOS
  crash report names the faulting address; if it equals the top of a 2 MB `rw-` `VM_ALLOCATE` region
  followed by a `---` one (OCaml 5 commits domain 0's minor heap at the low end of a 256 MB
  PROT_NONE reservation), the stub read past a block that happened to be the topmost allocation in
  the minor heap. Allocation-layout dependence is what makes it present as flakiness correlated with
  nothing meaningful — load, launcher, concurrency — so do not chase the correlation. To reproduce
  deterministically, put the short block at the top of a fresh minor heap: `Gc.minor (); let a =
  Array.make 1 0 in <call>`. That turns a 3-in-5 flake into 5-in-5 (`test/operations/uint4x32_stub_bounds.ml`).
