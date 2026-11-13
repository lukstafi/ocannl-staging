## OCANNL: what this project asks its second reviewer to look for

The rulebook is `AGENTS.md` at the root of the tree, and the subsystem notes it points to under
`docs/agent-notes/` (index: `docs/agent-notes.md`). Read the note that matches what the PR touches
before judging it, at the base:
`git -C @@CHECKOUT@@ show origin/@@MAIN@@:docs/agent-notes/<file>.md`. Hold the PR to those rules
as written there; when a rule and the code disagree, say which one the PR followed.

Look especially for:

- **A `Verdict` claim that cannot fail.** `p` over a `List.for_all`, or a quantified claim with no
  population floor, is true of an empty collection. So is a claim phrased so that `false` passes,
  one whose expected value is computed by the code under test, and a skip gate that reads the
  value it gates. Ask of each new claim what input would make it print `false`.
- **A negative control that was never run.** A scan, a refusal or a guard ships with a control
  that shows it refusing. A control fed an input derived from the constant it checks holds for any
  constant; one the PR body describes and no golden or harness contains was not run. A golden that
  could only have been promoted from a red run (a `false` line, a `FAILED` line) was not read.
- **A golden that pins the wrong thing.** A float a device reduction produced, on stdout at any
  precision, where the rule is digits to stderr and a two-sided claim on stdout. A backend-uniform
  golden offered as evidence of which backend ran. A count or tally that moves with every unrelated
  test. A snapshot of emitted code where the claim is about a value. A second copy of a set another
  part of the system owns, asserted to still say what it says, where the rule is to derive it or
  assert the two equal.
- **An executed-value claim checked only structurally.** A pass that changes what value a cell
  holds (virtualization guards, index solving, init elision, an algebraic rewrite) needs executed
  output compared against a materialized run, with every producer discriminating: varying with
  every index of its iteration and clear of the init value. An assertion on generated text or on
  the shape of the lowered code is necessary and is not that. Generated code is read through
  `Test_utils.Generated`, never from `build_files/` directly.
- **A missing backend leg.** A test of values that every backend computes has to have run on Metal
  and on CUDA or HIP as well as on cc, and the PR body says on which box each leg ran. A leg that
  could not run is named as not run, not left out. Claims only some backends can evaluate are
  written once, in a `Verdict.leg`, not as a skip list restating them. Text goldens of another
  backend (`.cu.expected`, `.metal.expected`) that the change reaches and the PR did not re-promote
  on that hardware are stale.
- **A numerics-changing gate outside the record.** A new configuration key is documented in
  `ocannl_config.reference`, registered in `Utils.known_config_keys` and classified for the schedule
  cache's identity; one that changes numerics lands in the `approximate` profile and is pinned at
  its default in `reproducible`, in the PR that adds it.
- **Work that is not this PR's.** `CHANGES.md` touched in feature work, a `(test)` stanza without
  `ocannl_config` in its `(deps ...)`, a test run with `dune exec`, a status read through a pipe.

Cite the rule you hold a finding to (`AGENTS.md`, or `docs/agent-notes/<file>.md#<heading>`), so the
worker can check your reading of it.
