# Context-scoped memory modes: split observability facts from placement decisions

**Date**: 2026-07-05
**Status**: Stub — design discussion captured from the schedule-layer work
(PR #90 follow-up); not scheduled.

## Motivation

`Tnode.memory_mode` conflates two things that have different natural scopes:

1. **Observability facts** — is the node read by the host, shared across routines or
   streams, persistent, effectively constant? These are properties of how the *program*
   (the set of comps plus host code) uses the node. They are inherently relative to a
   routine set — which today is approximated by "the whole process" via a global mutable
   field with provenance-tracked settlement.
2. **Placement decisions** — inlined ([Virtual]), routine-local scratch ([Local]), or a
   device buffer ([On_device]/hosted). These are per-compilation choices *constrained* by
   the observability facts. The contract already hints at the split: nothing says a
   [Local] node cannot be inlined — Virtual-vs-Local is a pure performance choice for any
   node that is unobservable, yet today it is settled once, globally, per node.

Now that contexts are values (`Context.t` carries the compile frontier, execution ledger,
and per-context buffers), observability questions always have a context in hand — and the
buffer side is *already* per-context: whether a node has storage is a fact about a
context, not about the node. The decision layer just doesn't match yet.

## The forcing function: autotuning isolation

The BEAM/search plan (schedule-ir-optops §7, gh-ocannl-261) compiles sibling candidates
from one frontier. With a global `memory_mode`, a candidate compile that forces or settles
a mode **poisons its sibling candidates** through shared mutable state — mode updates are
process-global, and conflicting settlements raise. Contexts-as-values was supposed to make
candidate compiles independent; the memory-mode global is the remaining shared-state leak
(the same class of problem as `Tnode.bounds_state`, which the interval work handled by
pinning at lowering time). Per-context placement resolution is what actually makes
sibling candidate compiles hermetic.

Secondary benefits: per-context specialization (the same tensor virtual inside a fused
inference kernel but device-resident in the training lineage), and making Virtual↔Local
officially a per-compilation choice — the "unobservable middle" where a future
`Materialize`/`Devirtualize` optop could operate without touching global state.

## Sketch

- Keep on `Tnode.t` the *requests and facts with global meaning*: hosted-required,
  effectively-constant, `Never_virtual` requests — the API-level intent, monotonic as
  today.
- Move the *resolution* (Virtual | Local | On_device) into the compilation lineage:
  either a `placements : placement Map.M(Tnode).t` on `Context.t` threaded through
  compile (matching how buffers already live per-context), or on the routine/`optimized`
  result keyed by the context that produced it.
- `is_virtual_force`/`is_materialized_force ~provenance` call sites (c_syntax, validate,
  schedule) become lookups against the resolution carried by the compile in progress.

## Honest caveats

- **Temporal monotonicity does not go away.** A later compile in the same lineage can
  still discover a reader of a node an earlier compile already resolved as Local — the
  frontier only knows the past, so late materialization must still raise (as bounds
  settlement does). Context scoping shrinks the blast radius to a lineage; it does not
  make placement retroactively revisable.
- **Wide touch surface**: the provenance-int force sites are scattered (c_syntax 333–338,
  validate 160, schedule 172/175/176, backend pool/transfer logic, slice aliasing,
  host-init upload paths). This is a refactor to schedule deliberately, not to fold into
  a feature PR — in particular, `Privatize` did not need it (its tiles are born
  per-kernel with fresh ids, so global `Local` mode is correct for them).

## Relations

[schedule-ir-optops](schedule-ir-optops.md) (§7 search needs hermetic sibling compiles),
[gh-ocannl-261](gh-ocannl-261.md) (autotuning consumer),
[memory-mode-streamlined; Local contract] (the 2026-07-03 lattice this splits),
[interval-analysis-scalar-t](interval-analysis-scalar-t.md) (the bounds-settlement
precedent for lineage-scoped monotonic facts).
