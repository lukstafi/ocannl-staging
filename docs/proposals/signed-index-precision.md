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

- `index_prec ()` returns int32 (int64 under `large_models`). Loop counters, `Embed_index`,
  and `Get_dynamic` index casts follow.
- Int32 overflow is excluded by contract, not by widening: per-node (padded) element count
  must fit int32 unless 64-bit indices are in use. This is sufficient — every index
  intermediate is bounded by some node's extent by projection construction (axis indices by
  their dims, conv affine forms by the padded input dim, flat offsets by numel); loop
  *counts* may exceed 2^31 without any index expression doing so. Enforce once per node by
  wrapping the dims lazy in `Tnode.create` (forcing dims validates padded numel; every
  consumer inherits the guarantee). Violation is a hard error naming the node — auto-setting
  the global flag would be inconsistent across already-compiled routines.
- Stronger endpoint: choose index width *per compiled routine* as int32/int64 by max
  numel over touched nodes — index width is routine-internal (loop counters and casts;
  indices are never stored in tensors), and the `large_models` switch then disappears.
  `Ops.index_prec ()` becomes a codegen parameter instead of a global read.
- Strongest endpoint (tinygrad's actual mechanism, verified against the 2026-06 source):
  index expressions are built in an abstract signed dtype (`dtypes.weakint`) and width is
  selected **per expression node** at codegen ("lower all index dtypes",
  `pm_lower_index_dtype` in `tinygrad/uop/ops.py`): `select_dtype u = int64 if
  u.overflows(int32) else int32`, where `overflows` is one vmin/vmax interval query
  (`u.vmin < dtype.min or dtype.max < u.vmax`); binary ops join widths via
  `least_upper_dtype` with casts at the boundaries; GPU special dims are pinned int32.
  All produced widths are signed — unsigned never appears in indexing. For OCANNL this
  means per-expression width selection falls out of
  [interval-analysis-scalar-t](interval-analysis-scalar-t.md) for free (one
  `interval_of` query per node at codegen); the per-routine max-numel rule above is the
  interim approximation implementable before intervals land.
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
- Backends resolve it per routine (max numel over touched nodes → int32/int64) before
  emission; kernel-parameter and launch-site FFI types agree automatically since both
  come from the same resolution. `Ops.index_prec ()` becomes this resolution rather than
  a settings read.
- Per-expression narrowing (tinygrad's `least_upper_dtype` joins) remains available later
  as a pure backend optimization powered by `interval_of` — no IR change — but is
  deferred: it only pays when a >2^31-element tensor coexists with small hot loops in one
  kernel, and mixed-width index arithmetic invites conversion bugs.

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

Before or together with masks-replacing-physical-padding (negative offsets are the first
real consumer). Independent of the interval analysis (which works under either signedness
but gets simpler after this lands).

## Relations

[interval-analysis-scalar-t](interval-analysis-scalar-t.md) (simplified by this);
[schedule-ir-optops](schedule-ir-optops.md) (Padto masks are the first consumer of
negative index arithmetic); gh-343 integer-guard flavors in `build_guarded_gather`
(2026-07-03) as the transitional state.

## Acceptance criteria (for the elaborated proposal)

- [ ] Decided: int32 default, overflow excluded by the per-node numel contract (see design
      points). Remaining choice: global `large_models` error-enforcement vs per-routine
      width selection (preferred; deprecates the switch).
- [ ] Enforcement site specified: dims-lazy wrapper in `Tnode.create` validating padded
      numel once per node.
- [ ] Inventory verified by grep: every `index_prec` consumer and every unsigned-idiom
      guard listed with its replacement.
- [ ] Guard-form policy specified: when to emit signed two-compare vs unsigned-cast
      single-compare (per-axis, pre-flattening restriction stated).
- [ ] Golden-churn estimate: count of `.expected` files touched.
