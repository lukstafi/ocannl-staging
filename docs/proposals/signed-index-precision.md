# Signed index precision

**Date**: 2026-07-03
**Status**: Stub — seeded by the interval-analysis discussion
([interval-analysis-scalar-t](interval-analysis-scalar-t.md)) and the integer-ids gather work.
Decision: migrate and accept the churn — signedness simplifies correctness.

## Goal

Migrate index arithmetic from unsigned (`Ops.index_prec ()` = uint32/uint64) to signed
(int32, int64 under `large_models`). Unsigned precisions remain for *data* domains only:
uint4x32 RNG state, uint8/uint32 stored values (e.g. class IDs).

## Motivation

The unsigned choice has charged a recurring correctness tax; three incidents so far:

1. `Fixed_idx (-1)` wraps to MAX (see the unsigned-index-precision lesson).
2. Guards must be written `a < b + 1` because `a <= b` with a possibly `-1` operand is wrong.
3. The gh-343 gather guard could not run in index precision at all (`-1 < idx` vacuously
   false after wrap) and resorted to `double` — since split into integer/float flavors, but
   the signed branch still had to pick int64 rather than the index precision.

Each was survivable because physical padding keeps all emitted index arithmetic
non-negative today. Logical padding / masks (the Padto direction) end that:
`Affine { offset = -padding }` makes negative intermediates *routine* rather than
exceptional, and `0 <= i` is inexpressible in unsigned arithmetic.

tinygrad precedent: index computation is signed throughout (int32 default, upcast to
int64 for large buffers; recent versions use an abstract index dtype that lowers to a
signed concrete type). Unsigned dtypes appear only as data (uint8 images/quantization,
bit-manipulation, threefry counters) — the exact split this proposal adopts.

## Design points

- Signed int32/int64 everywhere index arithmetic is emitted (loop counters, `Embed_index`,
  `Get_dynamic` casts), with width chosen at codegen — see the tnode-granularity bullet;
  the `large_models` switch is retired rather than flipped to signed.
- Int32 overflow is excluded by contract, not by widening: per-node (padded) element count
  must fit int32 unless 64-bit indices are in use. This is sufficient — every index
  intermediate is bounded by some node's extent by projection construction (axis indices by
  their dims, conv affine forms by the padded input dim, flat offsets by numel); loop
  *counts* may exceed 2^31 without any index expression doing so. Enforce once per node by
  wrapping the dims lazy in `Tnode.create` (forcing dims validates padded numel; every
  consumer inherits the guarantee). Violation is a hard error naming the node — auto-setting
  the global flag would be inconsistent across already-compiled routines.
- Width selection at **tnode granularity**, interval-driven (this task now *blocks on*
  [interval-analysis-scalar-t](interval-analysis-scalar-t.md) — see Sequencing): an index
  expression's width is the join over the extents and settled `vmin`/`vmax` bounds of the
  tensor nodes it indexes or reads. Static offsets into a node are bounded by its (padded)
  numel from projections; dynamic index values (`Get_dynamic.dyn_value`) are bounded by
  the Tnode interprocedural bounds layer — the same summaries that discharge guards also
  serve as the width oracle. This is tinygrad's actual mechanism (verified against the
  2026-06 source): index expressions are built in an abstract signed dtype
  (`dtypes.weakint`), and `pm_lower_index_dtype` in `tinygrad/uop/ops.py` selects width
  per expression at codegen — `select_dtype u = int64 if u.overflows(int32) else int32`,
  where `overflows` is one vmin/vmax query; binary ops join widths via
  `least_upper_dtype` with casts at boundaries; GPU special dims pinned int32. All
  produced widths are signed — unsigned never appears in tinygrad indexing. Mixed widths
  within a routine are safe *because* the choices derive from proven bounds; without
  intervals this would be ad-hoc casting, which is why the migration waits for them. The
  `large_models` switch disappears; `Ops.index_prec ()` becomes a codegen-time resolution
  instead of a global read. (A per-routine max-numel fold remains the trivial fallback
  for expressions the analysis cannot bound, and the correct width for launch-parameter
  FFI types.)
- The unsigned single-compare trick stays available *at guard sites* as a codegen choice:
  `(uint32_t)(i) < size` implements `0 <= i < size` in one comparison. Signed core
  arithmetic + unsigned-cast guards is strictly more expressive than either pure regime.
  Caveat carried over from the masking discussion: the trick is sound only when the
  possibly-negative value flows directly into the bounds compare — never through
  div/mod/multiply first — so guards must be per-axis, before flat-offset computation.
- Simplifies the planned interval analysis: machine integers and mathematical integers
  agree, so no wrap modeling and no "refuse when the lower bound could cross zero" rule.
- C compilers treat signed overflow as UB, hence assume no wrap — friendlier to loop
  strength-reduction and vectorization than defined-wrap unsigned.

## IR representation: abstract `Index_prec`

The OCANNL translation of tinygrad's `weakint`: a payload-less constructor in `Ops.prec`
(precedent: `Void_prec`), defined as *signed, integral, width unspecified*. Discipline:

- Storable, with canonical widths (revised from an earlier "annotation-only" draft —
  virtual nodes are retrievable by recomputation, so materialization concerns don't bar
  storage): host representation is OCaml native `int` (Bigarray `int` kind), device
  storage is int64; `prec_in_bytes` = 8. Width-abstractness applies to *arithmetic*
  (resolved per routine at codegen); storage width never depends on a routine's
  resolution, so cross-routine buffer sharing is consistent. 63-bit OCaml int reaches
  every index even for large models (byte offsets of any allocatable buffer are far
  below 2^62; the per-node numel contract keeps device values inside it).
- Payoff at the root: `TDSL.range` (and future argmax-style producers) become
  `Index_prec` by construction — virtual in the common case (inlined width-abstractly,
  as the `Embed_index` c_syntax arm already does), recomputed into native-int host
  arrays when observed, int64 when genuinely materialized. This retires the
  range-producer-in-model-precision trap without inference gymnastics, and
  `class_ids_of_int_list` can fill a plain `int` bigarray directly. Explicitly narrowed
  concrete storage (e.g. uint32 ids at half the footprint) remains an ordinary
  precision choice; both read into index arithmetic via the usual boundary casts.
- Lowering/optimization need only signedness + integrality, which the abstract prec
  carries by definition; width is exclusively a codegen concern. `Embed_index` is
  *already* width-abstract in the IR (`axis_index` has no precision; the c_syntax arm
  resolves `index_prec ()` at codegen) — the only early width commitments today are
  optimization-created guards (the gather's int64/uint64 branches become `Index_prec`)
  and the `Get_dynamic` cast target.
- Backends resolve it at emission, tnode-granularly via `interval_of` + Tnode bounds (see
  design points), with a per-routine max-numel fold as the fallback for unbounded
  expressions and as the launch-parameter FFI width; kernel-parameter and launch-site
  types agree automatically since both come from the same resolution. `Ops.index_prec ()`
  becomes this resolution rather than a settings read. (An earlier draft deferred
  per-expression narrowing as bug-prone; that held only without intervals — widths derived
  from proven bounds make the mixed-width joins mechanical, per tinygrad.)

## Migration inventory (churn accepted)

- `ops.ml`: `index_prec`.
- `c_syntax.ml`: loop counter emission, `Embed_index` / `Get_dynamic` cast targets.
- Backend builtins (`builtins.c`, `builtins_cuda.ml`, `builtins_metal.ml`): conversions
  involving the index precision.
- `low_level.ml`: guards using the `a < b + 1` idiom can revert to `a <= b` -like forms;
  the gather's signed branch can use index precision directly.
- Test goldens: `.expected` files asserting `((uint32_t)(` / `((uint64_t)(` casts
  (e.g. `test/operations/test_one_hot_embedding_lookup`), plus any `build_files` snapshots.
- Audit: pool slot types and buffer-offset arithmetic stay 64-bit-safe.

## Sequencing

**Blocks on [interval-analysis-scalar-t](interval-analysis-scalar-t.md)** (revised
2026-07-03; previously "independent"). The two are independent at their cores — intervals
over the current IR need no wrap modeling because physical padding keeps all emitted index
arithmetic non-negative by construction — but implementing this migration before intervals
would mean an interim per-routine width scheme and paying the `.expected` golden churn
twice. Landing after intervals lets width selection ship in its final tnode-granular form
in one pass. Nothing correctness-urgent is delayed: the unsigned wrap traps only bite when
logical-padding masks introduce negative offsets, and masks are sequenced after intervals
regardless. Still before or together with masks-replacing-physical-padding (negative
offsets are the first real consumer of signedness).

## Relations

[interval-analysis-scalar-t](interval-analysis-scalar-t.md) (blocks this; also simplified
by this once masks introduce negative offsets);
[schedule-ir-optops](schedule-ir-optops.md) (Padto masks are the first consumer of
negative index arithmetic); gh-343 integer-guard flavors in `build_guarded_gather`
(2026-07-03) as the transitional state.

## Acceptance criteria (for the elaborated proposal)

- [ ] Decided: int32 default, overflow excluded by the per-node numel contract (see design
      points); width selection is interval-driven at tnode granularity with a per-routine
      max-numel fallback, deprecating `large_models`. Blocks on
      [interval-analysis-scalar-t](interval-analysis-scalar-t.md).
- [ ] Enforcement site specified: dims-lazy wrapper in `Tnode.create` validating padded
      numel once per node.
- [ ] Inventory verified by grep: every `index_prec` consumer and every unsigned-idiom
      guard listed with its replacement.
- [ ] Guard-form policy specified: when to emit signed two-compare vs unsigned-cast
      single-compare (per-axis, pre-flattening restriction stated).
- [ ] Golden-churn estimate: count of `.expected` files touched.
