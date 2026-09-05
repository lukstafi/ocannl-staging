# `verdict_ratchet` synthetic controls: the mutation-run manifest

`test/operations/verdict_ratchet.ml` carries its own negative and near-miss controls (the
`quantified_helper_controls` list and the `run_*_control` families). A control earns its place by
a **mutation run**: the scanner mechanism it pins is deliberately disabled, the focused alias
`dune build @test/operations/runtest-verdict_ratchet` is run, and exactly the intended control(s)
fail while their accepting neighbours stay green. Those runs were made through `tools/test-run.sh`
while landing staging PR #633 (gh-ocannl-887, gh-ocannl-891), and their evidence was cited only in
the PR's inline review replies. This file is the one place that collects them, so a scanner change
can find every control that guards the mechanism it touches and re-run the same mutation.

Conventions:

- **Round** is the `Review fixes round N` commit on `master` that introduced the mechanism and its
  control (commit ids as merged through PR #633).
- **Mutation** is what was disabled or reverted in the scanner for the run; the control names are
  the exact labels in `quantified_helper_controls`, which is also how they print in
  `verdict_ratchet.expected` under `Synthetic helper-rule controls:`.
- **Run** is the `tools/test-run.sh` run id. The log lives at `~/.ocannl-test-runs/<run>/log` on
  the machine that ran the review (`lukstafi`'s macOS box, worktree `ocannl-887`); the id is the
  durable citation, the path is machine-local. Every mutation run exited 1 with exactly the listed
  controls reporting `false` (verified 2026-09-05 against the retained logs); the confirmation
  runs exited 0.
- A control listed here is **maintained**: rename it in the `.ml` and here together, and when a
  scanner change makes a mutation no longer meaningful, replace the row rather than deleting it.

## Mutation runs with a retained log

| Round | Mutation applied to the scanner | Control(s) that failed under the mutation | Run |
|---|---|---|---|
| 5 `d2bacbc7c` | Match-bound return values not mapped back to the scrutinee producer | `refuses a helper that returns a match-bound quantified value` | `20260904T165054Z-79197` |
| 5 `d2bacbc7c` | Recursive binding group analysed as non-recursive (no sibling fixed point) | `refuses a quantified helper reached through a mutually recursive sibling` | `20260904T165128Z-90221` |
| 5 `d2bacbc7c` | Function parameters left in the outer helper lookup while the body is analysed | `does not resolve an outer quantified binding shadowed by a function parameter` | `20260904T165229Z-31857` |
| 6 `ceef88daf` | `Fn.id`/`Fun.id`/`Stdlib.Fun.id` not treated as transparent Boolean wrappers | `refuses a fully applied quantifier through a transparent Boolean wrapper` | `20260904T171029Z-46614` |
| 6 `ceef88daf` | `returned_binding_polarities` resolves names past an intervening `let` | `does not return a quantified binding shadowed by a later local` | `20260904T171113Z-53808` |
| 6 `ceef88daf` | `filter`/`filter_map` views collapsed to the source name as population identity | `refuses a guard on a differently filtered population` (the same-filter control stays accepted) | `20260904T171139Z-60506` |
| 7 `de5a4ac09` | Dependency collection skips `if` conditions and protected `try` bodies | `refuses a bound quantifier returned through an if condition`; `refuses a bound quantifier returned from a protected try body`; also flags the `shell_scripts_parse.ml:line_enables_errexit` exemption as no longer earned, proving that exemption live | `20260904T172446Z-69949` |
| 8 `33aeb7057` | Match/try case patterns not removed from the outer helper environment | `does not resolve an outer binding shadowed by a match pattern` | `20260904T173912Z-15580` |
| 8 `33aeb7057` | Optional-default dependency edges disabled | `resolves an outer quantified binding used by an optional default` | `20260904T173938Z-31569` |
| 8 `33aeb7057` | Match-case guard traversal disabled | `refuses a bound quantifier returned through a match guard` | `20260904T174007Z-50752` |
| 9 `f3f9ee19f` | Supplied-optional-label set cleared at helper calls | `does not use an optional default dependency when the caller supplies the argument`; `suppresses an earlier default inside a later default when the caller supplies it` | `20260904T175423Z-47445` |
| 9 `f3f9ee19f` | Earlier optional defaults removed from the environment of later defaults | `resolves a quantified binding through chained optional defaults` | `20260904T175458Z-67152` |
| 9 `f3f9ee19f` | Direct condition quantifiers not attributed through complementary Boolean `if` branches | `refuses a direct quantifier returned through an if condition` | `20260904T175530Z-87146` |
| 10 `b9d7bc7f3` | Every `?label` forwarding treated as supplying the optional | `uses an optional default when a forwarded argument is None` (forwarded `Some` stays accepted) | `20260904T180755Z-76190` |
| 10 `b9d7bc7f3` | Direct quantifiers in match guards not collected | `refuses a direct quantifier returned through a match guard` | `20260904T180830Z-85990` |
| 10 `b9d7bc7f3` | `Pfunction_cases` bodies not traversed for returned quantifiers | `refuses a quantified helper written with function-case syntax` (its guarded neighbour stays accepted) | `20260904T180904Z-95866` |
| 11 `d7afcb3fb` | Non-empty witnesses unioned across `function` cases | `does not share a function-case guard with another case` | `20260904T182203Z-82716` |
| 11 `d7afcb3fb` | Match/try patterns not shadowed in `returned_binding_polarities` | `does not return an outer quantified local shadowed by a match pattern` | `20260904T182252Z-93990` |
| 12 `3a17c695c` | Scrutinee attribution through complementary `true`/`false` constructor matches disabled | `refuses a direct quantifier forwarded by a Boolean constructor match` | `20260904T183621Z-80904` |
| 13 `51932d7cd` | Local Verdict claim wrappers not recognised | `refuses a quantified binding passed through a Verdict wrapper`; `refuses a direct quantifier passed through a Verdict wrapper` | `20260904T185240Z-12119` |
| 14 `dfae272f6` | Wrapper slot signature replaced by an empty one | the two round-13 wrapper controls plus `refuses a direct exists negated by a labeled Verdict wrapper parameter`; `refuses a bound exists negated by a labeled Verdict wrapper parameter` | `20260904T190912Z-77325` |
| 14 `dfae272f6` | `try` handler guards not analysed for direct quantifiers | `refuses a direct quantifier returned through a try-case guard` | `20260904T190946Z-84558` |
| 15 `c27671af8` | `boolean_match_polarity` not composed with the claim polarity for bound scrutinees | `refuses a bound exists inverted by a Boolean constructor match` | `20260904T192620Z-99615` |
| 15 `c27671af8` | `try` handler guard dependencies not signed by the handler result | `refuses a bound exists inverted through a try-case guard` | `20260904T192722Z-34041` |
| 15 `c27671af8` | Wrapper optional-default traversal disabled | `uses an omitted optional default that feeds a Verdict wrapper claim` | `20260904T192814Z-66163` |
| 15 `c27671af8` | Tuple/record bindings do not retain unguarded quantified components | `refuses a quantified component destructured from an intermediate aggregate` | `20260904T192852Z-77163` |
| 16 `38c9986ea` | Unknown forwarded `?opt:expr` treated as `None` | `inspects the possible payload of an unknown forwarded wrapper option` | `20260904T195049Z-54546` |
| 16 `38c9986ea` | Under-applied `Verdict.p label` records no owed Boolean slot | `refuses a direct quantifier passed to a partially applied Verdict claim` | `20260904T195124Z-76928` |
| 16 `38c9986ea` | Nested-`let` witness boundary not sealed against outer same-spelled witnesses | `keeps an outer witness from guarding a shadowed nested population` | `20260904T195208Z-90599` |
| 17 `ca64d7dd3` | Owed Boolean slot created only for parameterless partial wrappers | `refuses a direct quantifier passed to a curried partial Verdict wrapper` | `20260904T200334Z-43193` |
| 17 `ca64d7dd3` | Case-guard polarity falls back to positive-only when the result is not a literal | `refuses a direct match guard whose false result is a Boolean alias`; `refuses a bound match guard whose false result is a Boolean alias` | `20260904T200408Z-50550` |
| 24 `d230df573` | Match-case returned quantifiers not sealed before re-entering an outer Boolean | `does not let an outer guard witness a match-bound population` | `20260904T215904Z-75791` |
| 24 `d230df573` | Wrapper-return polarity extraction ignores the `if` condition | `refuses a quantified condition used as a wrapper claim value` | `20260904T215941Z-83437` |
| 24 `d230df573` | Qualified lookup of file-local module wrappers broken | `refuses a quantified argument passed to a qualified local-module wrapper` | `20260904T220149Z-91421` |
| 25 `8c090d393` | Opened-module prefix corrupted on import of local wrapper exports | `refuses a quantified argument passed through an opened local module` | `20260904T221532Z-96093` |
| 25 `8c090d393` | Returned match-pattern bindings not mapped to the scrutinee for wrapper slots | `refuses a quantified argument forwarded through a match wrapper` | `20260904T221610Z-4324` |
| 25 `8c090d393` | Wildcard cases ignored when deriving Boolean match polarity | `refuses a direct quantifier forwarded by a wildcard Boolean match` | `20260904T221501Z-88511` |
| 26 `5b5cba542` | Local modules export only claim wrappers, not quantified helpers | `refuses a quantified helper called through a local module`; `refuses a quantified helper called through an opened local module` | `20260904T222946Z-71923` |
| 26 `5b5cba542` | Immediately invoked function body not traversed on the direct-quantifier path | `refuses a direct quantifier returned by an immediately invoked function` | `20260904T223017Z-79747` |
| 26 `5b5cba542` | Immediately invoked function body not traversed on the named-dependency path | `refuses a quantified binding returned by an immediately invoked function` | `20260904T223101Z-87728` |
| 26 `5b5cba542` | Quantifier-function alias lookup broken | `refuses a direct quantifier called through a function alias` (the inverted near-miss stays accepted) | `20260904T223205Z-12199` |
| 27 `1e14b1368` | Boolean-match result not applied to the scrutinee before deriving wrapper slots | `refuses a quantified argument forwarded by a Boolean match wrapper` | `20260904T224356Z-80717` |
| 27 `1e14b1368` | Callback bodies not traversed during wrapper claim discovery | `refuses a quantified argument claimed inside a callback` | `20260904T224435Z-88688` |
| 27 `1e14b1368` | Callback parameter shadowing done by filtering instead of lexical tombstones | `does not connect a callback-shadowed parameter to its wrapper` | `20260904T224511Z-96391` |

### The deferred prototype

| Round | Prototype | What it showed | Run |
|---|---|---|---|
| 25 `8c090d393` | Native `Verdict.*` calls scanned for direct quantifiers (not merged) | Caught the proposed fixture, and also 65 clean-tree corpus sites plus three claim-local controls now reporting `Verdict.p` as the helper. Deferred to ahrefs/ocannl#908 as an audited migration. | `20260904T221248Z-50429` |

## Mutation runs recorded only in the PR thread

These rounds report the mutation in their inline review reply on staging PR #633 but retain no run
id. The control names are the ones the round added; re-running the mutation is the way to
re-establish them.

| Round | Mutation reported | Control(s) |
|---|---|---|
| 1 `b3890bc4b` | Explicit true/false comparison polarity disabled | `refuses a fully applied quantifier compared with true`; `accepts a fully applied quantifier compared with false` |
| 1 `b3890bc4b` | Lexical environment for `let` bindings nested inside a claim argument removed | `refuses a binding nested directly inside a claim argument` (guarded and negated neighbours stay accepted) |
| 2 `6eb3dc479` | Direct `Bool.equal` dispatch removed | `refuses a direct Bool.equal true around a fully applied quantifier`; `accepts a direct Bool.equal false around a fully applied quantifier` |
| 2 `6eb3dc479` | Intermediate dependencies collected without polarity; inherited guards not propagated | `accepts a negated intermediate binding`; `accepts a guarded intermediate binding` |
| 2 `6eb3dc479` | Literal tuple/record pattern mapping to producers disabled | `refuses a quantified component of a destructured tuple binding`; `conservatively refuses a quantified component of a record binding` |
| 3 `a2398f95b` | Inherited guards forwarded across helper calls | `conservatively refuses an outer guard across a helper call`; `refuses a mismatched actual hidden by equal formal names` |
| 4 `f96942416` | Inherited guard identity forwarded into a nested `let` | `refuses a shadowed guard identity across a nested alias` |
| 4 `f96942416` | Pipeline-position `not`/`Bool.not` not recognised | `accepts a piped negated intermediate binding`; `accepts a directly quantified value piped through not` |
| 4 `f96942416` | Negative dependency edges dropped | `refuses a negated bound exists` (`accepts a positive bound exists` stays accepted) |
| 4 `f96942416` | Signed local returns removed, in each direction | `accepts a helper that negates a quantified local binding`; `refuses a double negation around a quantified local binding` |
| 18 `eb72b88a0` | Constant Boolean alias resolution for `if` polarity disabled | `refuses a direct if condition whose false outcome is a Boolean alias`; `refuses a bound if condition whose false outcome is a Boolean alias` (`does not attribute a condition whose branches return the same literal` stays accepted) |
| 18 `eb72b88a0` | Tail-position setup unwrapping (let/sequence/local open/constraint/coercion) disabled | `refuses a direct quantifier passed through a wrapper with tail setup`; `refuses a returned quantifier behind a local open` |
| 18 `eb72b88a0` | `not @@ value` polarity flip removed | `refuses a negated exists written with the application operator` |
| 18 `eb72b88a0` | Direct wrapper quantifiers keyed by the wrapper definition offset | the `reused wrapper call` shadowed-quantified controls (`run_shadowed_quantified_controls`) |
| 19 `5a04464da` | Scoped alias map dropped from wrapper tail traversal | `refuses a direct quantifier passed through a wrapper setup alias` (`does not connect a wrapper parameter hidden by a setup constant` stays accepted) |
| 20 `539804bc7` | Constructor-match polarity without constant-alias resolution | `refuses a bound exists inverted by aliased Boolean match outcomes`; `refuses a direct exists inverted by aliased Boolean match outcomes` |
| 20 `539804bc7` | Boolean comparison without constant-alias resolution | `refuses a direct exists compared with a false Boolean alias`; `refuses a bound exists compared with a false Boolean alias` |
| 21 `16839405f` | Unsupplied claim slots not forwarded through a partially applied local wrapper | `refuses a direct quantifier passed to a partially applied local wrapper` |
| 22 `b9cdffab2` | Direct wrapper quantifiers keyed by the call offset instead of the argument offset | the `call slots` shadowed-quantified controls (`run_shadowed_quantified_controls`) |
| 23 `5e92676b0` | Local-module/local-exception setup not recursed into for returned quantifiers | `refuses a returned quantifier behind local module setup` |

## Confirmation runs

Green runs cited beside the mutations, on the rebased head of the round: the focused ratchet alias,
then `@test/operations/scans`, then `@check`.

| Round | Ratchet (and co-migrated aliases) | Scans | `@check` |
|---|---|---|---|
| 6 | `20260904T171421Z-23314` (with `launch_predicate_parity`, `config_usage_scan`, `dead_export_scan_cases`, `env_var_deps`) | | |
| 13 | `20260904T185518Z-88373` (with `test_random_histograms`, `threefry4x32_demo`) | `20260904T185630Z-26420` | `20260904T185651Z-35075` |
| 14 | `20260904T191034Z-91792` | `20260904T191059Z-98768` | `20260904T191118Z-6656` |
| 15 | `20260904T193638Z-84803` (with `autotune_arm_containment`, `test_random_histograms`, `threefry4x32_demo`) | `20260904T193649Z-91822` | `20260904T193704Z-8077` |
| 16 | `20260904T195243Z-886` | `20260904T195304Z-9381` | `20260904T195321Z-18572` |
| 17 | `20260904T200448Z-57923` | `20260904T200503Z-65115` | `20260904T200518Z-72328` |

## The other control families

These predate PR #633 and are exercised on every run rather than by a one-off mutation; they are
listed so the inventory is complete.

- `run_refusal_control` (`--quantified-helper-refusal-control`): the ratchet re-executes itself on a
  synthetic offending fixture and checks the exact refusal diagnostic and exit status. Introduced
  by `cf1874075` (bind refusal markers to exercised controls).
- `run_shadowed_quantified_controls`: an exempted quantified helper key must name exactly one
  definition, including two definitions on one physical line, one call site, or one call slot.
  Introduced by `ad983d863`; the wrapper call-site and call-slot members by rounds 18 and 22 above.
- `run_stale_quantified_control`: an exemption whose key no longer matches a live claim is refused.
  Introduced by `ad983d863`.
- `run_colliding_site_controls`: literal-label and computed-label exemption keys that resolve to
  two sites, on separate lines or twice on one line, are refused. Introduced by `7c4e8a826`
  (gh-ocannl-891 offset-based identities).
