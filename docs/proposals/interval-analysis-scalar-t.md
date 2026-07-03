# Interval (min/max) analysis over scalar_t

**Date**: 2026-06-12
**Status**: Stub — seeded by the tinygrad deep dive
([a-range-is-not-its-shape](../blog/a-range-is-not-its-shape.md), port area 3). Judged
there the best effort-to-payoff item of the six ports; no blocking dependency.

## Goal

An interval lattice over `Low_level.scalar_t` (and the integer analogue over
`Indexing.axis_index`), in the spirit of tinygrad's `vmin`/`vmax` derived property
(`tinygrad/uop/ops.py`, `_min_max`): `Constant` exact, `Embed_index` bounded by its
loop's statically-known extent, `Get` bounded by dtype range, arithmetic by interval
rules (add endpoints; extremal products; careful div/mod-by-constant cases). Exposed as
an analysis `simplify_llc`'s rewrite arms can consult to discharge validity masks,
prove indices in-bounds, and fold comparisons.

## Key points from the article's analysis

- Four existing efforts independently approximate this one analysis: stage-B
  injectivity in [#133](https://github.com/ahrefs/ocannl/issues/133) (range arithmetic
  over loop extents), read-before-write tracking in
  [#340](https://github.com/ahrefs/ocannl/issues/340), matcher side-conditions in
  [#343](https://github.com/ahrefs/ocannl/issues/343), and the landed surjectivity
  reasoning of [#420](https://github.com/ahrefs/ocannl/issues/420). Intervals are the
  unifying upgrade.
- Sequencing: before the schedule layer
  ([schedule-ir-optops](schedule-ir-optops.md)) — Padto is affordable in tinygrad
  *because* interval reasoning discharges most of the masks it introduces.
- Receiving site exists: `interval_of : scalar_t -> bounds` slots into `simplify_llc`'s
  world; loop extents are statically known from projections.

## Interprocedural layer: vmin/vmax on `Tnode.t`

The `scalar_t`-level analysis is intra-routine; without more, every `Get`/`Get_dynamic`
degrades to dtype range at routine boundaries (a reader routine lowers its own comp; the
writer's defining expression is in a different comp — the tensor node is the only artifact
crossing between them). Store per-tensor scalar bounds (float `vmin`/`vmax` pair, exact for
integer values below 2^53) on `Tnode.t` as writer-derived summaries. Payoff: with ids
settled at `[0, vocab)`, the gh-343 gather guard folds *entirely* for genuinely runtime
ids, not just range-produced ones; same for data-dependent mask discharge.

Soundness design (third instance of the `delayed_prec` lifecycle pattern):

- **Multi-writer staleness**: a guard discharged against bounds, followed by a
  later-compiled wider writer, would leave already-generated code unsound (an OOB read,
  not a wrong zero). So: writers *propose* bounds (joined, like `Inferred` promotion);
  a reader that discharges a guard *settles* them (like forcing the prec lazy); a
  post-settlement writer whose interval does not fit is a compile-time error (like
  `update_prec` on a settled precision).
- **Host writes**: symmetric with compiled writers around the settlement point.
  *Pre-settlement*, `set_values`/`from_host` (and ndarray-backed `init_data` at creation)
  act as writers: scan the uploaded values (O(n), trivial next to the copy) and *propose*
  the observed `[min, max]` into the join — or pin the tensor to top if scanning is
  disabled. Otherwise a later reader could settle narrow writer-derived bounds and fold a
  guard against host contents that were never inspected. A bonus: tensors created via
  `class_ids_of_int_list`-style host arrays get tight bounds automatically.
  *Post-settlement*, host writes validate against the settled bounds or error.
- **Self-referential writers** (read-modify-write across executions: param updates,
  accumulators) default to top — a fixpoint over unbounded repetitions is not attempted.
  The profitable cases (ids, range producers, one-hots) are write-once/functional.

## Relations

[#133](https://github.com/ahrefs/ocannl/issues/133),
[#340](https://github.com/ahrefs/ocannl/issues/340),
[#343](https://github.com/ahrefs/ocannl/issues/343), landed #420;
[schedule-ir-optops](schedule-ir-optops.md) (downstream consumer);
[signed-index-precision](signed-index-precision.md) (removes the need to model unsigned
wrap in the integer lattice — with signed indices, machine and mathematical integers
agree and the "lower bound could cross zero" refusal rule disappears).

## Acceptance criteria (for the elaborated proposal)

- [ ] Lattice and rules specified (float vs. index-integer variants; widening not
      needed — loop extents are finite and static).
- [ ] At least one existing approximation (#133's range arithmetic or #343's
      side-conditions) re-expressed through the shared analysis.
- [ ] Caching strategy decided (per-node memo during a `simplify_llc` run).
- [ ] `Tnode` bounds lifecycle specified (propose/settle/conflict semantics mirroring
      `delayed_prec`; host-write validation; self-referential writers pinned to top).
