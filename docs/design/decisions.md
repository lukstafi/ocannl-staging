# Design decisions taken without the maintainer in the loop

A ledger, not a design record: one row per design decision that a wave coordinator or a worker
took so that work could go on, for the maintainer to review afterwards. Waves run without design
holds, so a decision that would once have waited for an answer is taken, written up as a comment on
its issue, and listed here. The comment is the record; the row is a pointer to it.

- **A row is added** when its decision comment is posted and the issue gets the `design-joint`
  label. A wave's rows land together, in the one close-out commit its coordinator makes, so workers
  do not all edit this file; a session outside a wave adds its row in the PR that acts on the
  decision.
- **Newest first**: a new block of rows goes directly under the header row.
- **Who decided** is `maintainer`, `coordinator of <wave>`, or `worker` with the wave or session it
  worked in.
- **Status** is the maintainer's to change: `unreviewed` until read, then `ok`, or `revisit` when
  the call should be reopened, with the reason on the issue.

| Date | Issue | The call | Who decided | Decision | Status |
| --- | --- | --- | --- | --- | --- |
| 2026-11-13 | [ahrefs/ocannl#2193](https://github.com/ahrefs/ocannl/issues/2193) | Keep `test/operations/dune` one file with its test stanzas sorted by name and a scan holding the order, rather than one dune file per test; the agent notes get a place-by-topic rule, not a re-split. | worker (issue-wave v2, Track D) | [comment](https://github.com/ahrefs/ocannl/issues/2193#issuecomment-6856387993) | unreviewed |
| 2026-11-11 | [ahrefs/ocannl#2154](https://github.com/ahrefs/ocannl/issues/2154) | Part 2 may add a bounded replay over the stage-6 balanced-flank trial, which adds a search to the row solver. | coordinator of wave-20261111-ocannl | [comment](https://github.com/ahrefs/ocannl/issues/2154#issuecomment-6809946796) | unreviewed |
| 2026-11-11 | [ahrefs/ocannl#2134](https://github.com/ahrefs/ocannl/issues/2134) | The bounded wait is public API on every backend (`Backend.await_within`, `Context.await_within`), not an optional Metal-only capability. | coordinator of wave-20261111-ocannl | [comment](https://github.com/ahrefs/ocannl/issues/2134#issuecomment-6799923154) | unreviewed |
| 2026-11-11 | [ahrefs/ocannl#2167](https://github.com/ahrefs/ocannl/issues/2167) | Refuse one label over two concatenated axes of one tensor, and list it among what the einsum notation cannot say. | coordinator of wave-20261111-ocannl | [comment](https://github.com/ahrefs/ocannl/issues/2167#issuecomment-6799177304) | unreviewed |
| 2026-11-11 | [ahrefs/ocannl#2163](https://github.com/ahrefs/ocannl/issues/2163) | Both parts: a ppx compile-time refusal for a capture label absent from its spec, and the user's source spec text carried into `Shape` so every einsum refusal cites what was written. | coordinator of wave-20261111-ocannl | [comment](https://github.com/ahrefs/ocannl/issues/2163#issuecomment-6799176978) | unreviewed |
| 2026-11-11 | [ahrefs/ocannl#2155](https://github.com/ahrefs/ocannl/issues/2155) | A `~persist` flag on `Context.set_values`, rather than a separate seeding verb. | coordinator of wave-20261111-ocannl | [comment](https://github.com/ahrefs/ocannl/issues/2155#issuecomment-6799178644) | unreviewed |
| 2026-11-11 | [ahrefs/ocannl#2154](https://github.com/ahrefs/ocannl/issues/2154) | Reproduce first: keep the refusal if it follows from gh-544's close-down and make the message name the closing; fix it only if the store is satisfiable. | coordinator of wave-20261111-ocannl | [comment](https://github.com/ahrefs/ocannl/issues/2154#issuecomment-6799176641) | unreviewed |
| 2026-11-11 | [ahrefs/ocannl#2144](https://github.com/ahrefs/ocannl/issues/2144) | A typed refusal at the host fill, through `Ops.integer_constant_converts`, rather than a defined wrap. | coordinator of wave-20261111-ocannl | [comment](https://github.com/ahrefs/ocannl/issues/2144#issuecomment-6799178294) | unreviewed |
| 2026-11-11 | [ahrefs/ocannl#2143](https://github.com/ahrefs/ocannl/issues/2143) | Unary `~-` at unsigned precisions renders as a wrapping `0 - t`, agreeing with `0. - a`. | coordinator of wave-20261111-ocannl | [comment](https://github.com/ahrefs/ocannl/issues/2143#issuecomment-6799177986) | unreviewed |
| 2026-11-11 | [ahrefs/ocannl#2135](https://github.com/ahrefs/ocannl/issues/2135) | Count a finals `Writes_may_have_occurred` failure as a decline, so the search has one accounting across phases. | coordinator of wave-20261111-ocannl | [comment](https://github.com/ahrefs/ocannl/issues/2135#issuecomment-6799176327) | unreviewed |
| 2026-11-11 | [ahrefs/ocannl#2134](https://github.com/ahrefs/ocannl/issues/2134) | Bound a candidate's timing window on the host as a multiple of the incumbent's time with an absolute floor, its expiry a typed decline, rather than adding a configuration key. | coordinator of wave-20261111-ocannl | [comment](https://github.com/ahrefs/ocannl/issues/2134#issuecomment-6799175945) | unreviewed |
| 2026-11-11 | [ahrefs/ocannl#2128](https://github.com/ahrefs/ocannl/issues/2128) | Charge the consumer the max over the select alternatives joined with its own loads, rather than carrying the narrower alternatives downstream. | coordinator of wave-20261111-ocannl | [comment](https://github.com/ahrefs/ocannl/issues/2128#issuecomment-6799177618) | unreviewed |
