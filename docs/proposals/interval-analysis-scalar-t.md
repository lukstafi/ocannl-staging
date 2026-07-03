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
- **Symbol environment tracks all symbols.** `Embed_index` is where a symbol crosses from
  the index world into the value world, so the analysis threads `symbol → interval` —
  the abstract twin of `visit_llc`'s concrete `symbol → int` env. Completeness is by
  construction: every in-scope symbol at lowering time comes from a `For_loop`
  (`[from_, to_]`) or a static binding (`[0, static_range)`; top when the range is
  `None` — see [signed-index-precision](signed-index-precision.md) on bind-time
  validation). The one env serves both value analysis (`Embed_index` into guards and
  comparisons) and position analysis (`Get`/`Set` `idcs`, the #133/#420-style facts).
  Wiring is already in place: `optimize_proc` receives the full
  `static_indices : static_symbol list` (ranges included) and passes it to every
  candidate host pass; bounds are lost only to the per-pass idiom that strips the list
  to a bare-symbol membership `Set` (low_level.ml lines 613/796/1479 as of 2026-07-03).
  Seeding = building a `symbol → interval` map from the same list before the strip.
  Verify at implementation time that `static_range` (a `mutable int option`) is settled
  before `optimize_proc` runs — dims are forced by lowering, so ranges derived from dims
  should be; assert rather than assume.
- **The memo must be env-scoped.** The same `scalar_t` subtree can be physically shared
  under different loop nests (immutable-tree sharing after rewrites; CSE/`Local_scope`
  reuse), and its interval differs per enclosing scope. Key the memo by (expression,
  symbol-env) or scope it per loop body — a global identity-keyed memo would return
  another scope's bounds.

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
[signed-index-precision](signed-index-precision.md) (**blocks on this**, revised
2026-07-03: intervals need no wrap modeling over the current IR — physical padding keeps
emitted index arithmetic non-negative by construction, so the "lower bound could cross
zero" rule is a cheap assert until masks land — while the signed migration waits for
intervals so tnode-granular width selection ships in its final form and the golden churn
is paid once).

## Acceptance criteria (for the elaborated proposal)

- [ ] Lattice and rules specified (float vs. index-integer variants; widening not
      needed — loop extents are finite and static).
- [ ] At least one existing approximation (#133's range arithmetic or #343's
      side-conditions) re-expressed through the shared analysis.
- [ ] Caching strategy decided (per-node memo during a `simplify_llc` run).
- [ ] `Tnode` bounds lifecycle specified (propose/settle/conflict semantics mirroring
      `delayed_prec`; host-write validation; self-referential writers pinned to top).
