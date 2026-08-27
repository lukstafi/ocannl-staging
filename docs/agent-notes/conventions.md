# Conventions

Releases, configuration spellings, git and PR mechanics, and the honesty rules for test output
and measurement reports.

Part of the agent notes; the [index](../agent-notes.md) carries the scope discipline and the other
files.

- Releases use lightweight, un-prefixed git tags (`0.8`, not `v0.8`).
- `ocannl_config.reference` ships with every setting COMMENTED OUT, and the two forms are
  load-bearing: a commented-out setting is `#key=value` with NO space after the `#`, while prose
  (and the verbatim profile-payload blocks at the end of the file) always uses `# `. That is how
  `test_config_consistency` tells documented keys from prose, so a new key documented as `# key=…`
  reads as undocumented and fails the test. Config values may not contain `=` — the parser splits
  on it and rejects a line with two, which rules out payload/config values like `-mcpu=native`; a
  setting that needs one gets a word spelling instead (`cc_backend_arch_flags=none`). A value can
  never be the empty string either: empty means "unset" at every source.
- A configuration key has exactly TWO environment spellings, `ocannl_<key>` and `OCANNL_<KEY>`
  (gh-ocannl-605 dropped the dash-prefixed pair). A dune rule whose output depends on an ambient
  OCANNL setting must declare both as `(env_var …)` deps — dune tracks none but `OCANNL_BACKEND`,
  so an undeclared one leaves a stale target in place and the test green without having run. The
  lowercase spelling is not the redundant one: `Utils.read_env_var` consults it FIRST, so
  `ocannl_profile` beats `OCANNL_PROFILE`. The commandline is the permissive side and always has
  been (`Utils.cmdline_var_names`), but along fixed axes rather than per separator: prefixed or
  not, either case, one leading dash or two (never zero — a bare argument is the host's
  positional), value separator `=`, `_`, `-` or nothing, and the dashing is TWO choices — the
  prefix separator on its own, the key's own separators as a group. So `--ocannl-log_level=1` and
  `--ocannl_log-level=1` are the same setting while a halfway-dashed key
  (`--ocannl-print_decimals-precision=1`) is not a spelling at all. That table
  (`Utils.cmdline_var_prefixes`) is also what the unknown-argument warning matches against — do not
  give it a parser of its own, which is how it came to warn about arguments the reader applied and
  stay silent about ones it ignored.
- A `bin/` tool's own arguments are POSITIONAL and share the commandline with the library's
  `--ocannl_*` settings, so every one of them splits argv the same way — through `Bench_args`
  (`bin/bench_args.ml`, gh-ocannl-634), not a hand-rolled filter. An option is `--`-prefixed or a
  `-` followed by a non-digit, a lone `--` ends the options, and `Bench_args.int` range-checks each
  argument where it is read (`~least:0` for a documented zero); a repeated `--flag=` resolves to the
  FIRST spelling, as `Utils.read_cmdline_var` does. The `--` terminator governs that split only —
  the library scans the whole of `Sys.argv` for its own settings with no terminator and prefix-free
  spellings accepted, so a post-`--` positional spelling a known key (`-- --backend=cuda` as a
  prompt) has already been applied as configuration and cannot be unset from a tool; `Bench_args`
  cannot fix that, so it warns (`shadowing_config`) rather than letting it pass silently. The filter that reads naturally —
  drop everything starting with `-` — is the bug this replaced: it eats a negative extent, and with
  several positionals that shifts every later argument one slot left, so the bench measures a
  geometry nobody asked for and reports a plausible number for it. Review found that same defect in
  two separate copies of the idiom. `test/operations/bench_args_parsing` pins the predicate, the
  slot alignment and the refusals; it links `bench_args` alone, no backend.
- An OCANNL-linked executable's stdout belongs to the program, not to the library: the config
  startup chatter (welcome message, `log_config_sourcing` trace, profile banner) and every other
  library diagnostic go to stderr. That is what lets a tool make stdout a data channel — the
  benchmark one-JSON-line runners, `tools/fit_envelope.exe`'s config-pasteable fits — without
  suppression flags. Keep new library-side reporting on stderr. `test/operations/startup_streams`
  pins both halves: stdout stays empty with the chatter turned all the way up, and a default run's
  stderr is the welcome banner plus whatever warnings there are. The second half is why
  `log_config_sourcing` and `log_level` default to off/0 (gh-ocannl-595): a stream that carries
  eighty lines of routine trace cannot carry a warning, and the unknown-config-key warning is the
  one startup message that means the user made a mistake.
- Prefer the minimal targeted fix over speculative hardening: offer hardening separately as an
  option with its costs, don't fold it into the fix.
- Git refuses to check out or update a branch that ANOTHER worktree has checked out. This is
  checked-out-branch protection, not `git worktree lock` (which is about pruning and moving a
  worktree, so `git worktree unlock` does nothing for these refusals). While the main checkout holds
  `master`, a linked worktree cannot `git checkout master` ("'master' is already used by worktree
  at …"), `git branch -f master …` ("cannot force update the branch"), or `git fetch origin
  master:master` ("refusing to fetch into branch"). It CAN write the remote ref, which is untouched
  by any of this: `git push origin HEAD:master` — after `git fetch origin && git rebase
  origin/master`, since the push has to fast-forward — lands a commit straight on master from a
  worktree. What that leaves behind is local: the main checkout's `master` is now stale, and only
  that checkout can advance it (`git -C <main> merge --ff-only origin/master`). Either do so, or
  give every later branch an EXPLICIT start point — `git worktree add -b next <path> origin/master`,
  `git checkout -b next origin/master` — because an omitted start point takes the current HEAD, and
  from a stale main checkout that silently drops the commits just landed.
- The same protection makes `gh pr merge --delete-branch` misleading from a worktree: the merge
  LANDS and only the cleanup fails ("fatal: 'master' is already used by worktree"), so the command
  exits nonzero over an already-merged PR — check the PR's state over REST (`gh api
  repos/<owner>/<repo>/pulls/<n> --jq '"merged=\(.merged) state=\(.state)"'`) before reacting to that
  status, and again before any cleanup. Use REST because `gh pr view --json` and `gh pr checks` ride
  the GraphQL endpoint, which degrades independently of it, and a nonzero `gh pr merge` whose state
  query then 503s is exactly the shape of a merge that DID land; the REST substitute for the checks
  is `gh api repos/<owner>/<repo>/commits/<sha>/check-runs --jq '[.check_runs[]|"\(.name):
  \(.conclusion // .status)"]|.[]'`. `gh pr merge` also returns WITHOUT merging when the base has
  required checks or a merge queue, enabling auto-merge instead, and the steps below would then tear
  down a PR still waiting to land. This repo has neither, so here the flag's own cleanup failure is
  the only way that command misleads.
  Merge without the flag and clean up in this order, every command anchored with `git -C <main>` so
  that none of them depends on the current directory: `push origin --delete <branch>`; `fetch
  --prune origin`; `merge --ff-only origin/master`; `worktree remove <path>`; `branch -d <branch>`
  (`-D` after a squash or rebase merge, whose commits are not ancestors of `master`). The sequence
  runs green from inside the worktree it deletes, verified end to end in a scratch repo, and both
  ordering constraints are load-bearing. `-d` tests the branch's UPSTREAM, falling back to HEAD only
  when there is none, so deleting the remote branch FIRST is what makes it a real "merged into
  master" check instead of a tautology about `origin/<branch>` — left in place, it deletes an
  unmerged topic with only a warning. With the upstream gone the check lands on the main checkout's
  `master`, which the merge commit on `origin` does not advance, hence the fast-forward before it.
  Anchoring matters because `worktree remove` deletes the current directory when run from inside it,
  and any later unanchored command dies with "fatal: Unable to read current working directory".
- A backend-gated leg must never print a bare `p "<claim>" true` on the backend that cannot run it:
  the golden line is then byte-identical to a verified run's, so neither the transcript nor a
  reviewer can tell the claim was never evaluated (this is how a `Tensorize` leg came to "cover" the
  gh-528 interior-batch bug). Report it with `Verdict.skipped ~backend "claim"`, which prints the
  same stdout line — the golden must stay backend-uniform, and dune's `(test)` stanza diffs stdout
  ONLY, so stderr is free — and announces the skip on stderr. `grep SKIPPED` over a run then
  enumerates exactly what that hardware did not verify. Backend-specific goldens are the wrong tool
  for this: the leg is *absent* on that backend rather than differently-valued, and a golden per
  backend would have to be regenerated on hardware the author usually does not have. The other honest form is putting the
  condition in the label itself (`"… (skipped: non-C backend)"`), which distinguishes the golden
  line; a bare `true` whose label is indistinguishable is the one to reject.
- The skip helper keeps the CLAIM line uniform; it says nothing about the descriptive `printf`s
  beside it. Guarding the whole leg — device run, dump, and claim — behind the `else` branch leaves
  the dump printed on some backends and not others, which is a golden diff on exactly the backend
  that skips. Hoist the descriptive output out of the branch, computing it from something that runs
  everywhere (usually the host-side reference the claim compares against), and gate only the
  device-dependent comparison.
- Executed parity checks need a nonzero guard on the REFERENCE, not just the comparison: a fragment
  mapping that reads outside a staged block, a candidate kernel that never ran, and a reference
  whose own setup collapsed all produce all-zeros, and zeros compare equal to zeros. The convention
  is a file-local `nonzero name a` that raises, applied where each reference array is produced —
  once per producer, not per comparison. Guard the reference side only: a zero candidate against a
  nonzero reference is already a `false` in the golden, which is more diagnosable than an exception.
- A tolerance cannot reject an input-independent forward if the reference itself does not move:
  every leg sits at one constant and every parity line reads `true`. `benchmarks/orchestrate.py`'s
  `loss_moved` is the model; in-tree, require the reference's own spread to exceed the tolerance it
  gates (`mixed_prec_parity`, `precision_policy_parity`) or that distinct inputs give distinct
  outputs (`gpt2_dry_run`'s positions-differ). "All finite" is not such a guard — all-zeros is
  finite.
- A new configuration key must not have an existing key as a NAME PREFIX. `Utils.read_cmdline_var`
  matches an argument against `key ^ ("_" | "-" | "=" | "")` and takes whatever follows as the value,
  so `--ocannl_cc_parallel_grid_private_bytes_cap=N` was read as `cc_parallel_grid` with the value
  `private_bytes_cap=N` and crashed the run — the key was renamed `cc_grid_private_bytes_cap`.
  Nothing checks this: the consistency tests check documentation, classification and read sites, not
  name disjointness.
- Stacked PRs: once the base PR merges, RETARGET the stacked one to master BEFORE merging it. A merge
  into the now-stale base branch lands the work on that branch and nowhere else while GitHub still
  reports the PR as "merged" — staging#168's conv sketches were stranded exactly that way and had to
  be re-landed as #170. The same audit question ("is this on master?") is worth asking of any PR whose
  base was not master.
- For a measurement or report PR, substance stabilizes early: two full review rounds plus one
  verdict-stability check, after which findings are answered rather than actioned unless they touch
  validity, consequence, or arithmetic (validated on the gh-ocannl-530 campaign, where rounds 5–7 were
  framing churn). The failure mode review is actually there to catch runs in both directions — quoting
  whichever control flatters the story — so a report shows ALL matched contrasts side by side rather
  than the decisive one.
- When the next review finding is "leg X missed guard Y", look for the unfactored duplication instead
  of patching leg X. In `tools/sweep.sh`'s nine rounds every point-wise guard had a leg, a path or a
  machine it had not been applied to, and the fixes that actually closed a class REMOVED or UNIFIED
  mechanism: one `flock` replacing a directory-plus-pid-file dance with its reclaim races and
  `kill -0` pid-reuse hole, one `run_capped` replacing three hand-rolled background-and-publish-pid
  call sites.
- Any file OCANNL publishes for a later process to read — a schedule-cache entry, a checkpoint, the
  cc probe cache — goes through `Utils.Atomic_file` (`arrayjit/lib/atomic_file.mli`), never through
  a hand-rolled `<path>.tmp`. Three parts, and a hand-rolled copy usually has one or two: a staging
  name carrying the writer's pid AND a per-process counter (a fixed `.tmp` lets two writers stream
  into one file and commit a mixture), removal on every failing path, and `cleanup_stale` for the
  writer killed inside its commit window, which cannot clean up after itself. The Windows halves
  are measured, not inferred (gh-ocannl-588): a live mapping blocks neither the rename nor a delete
  but PINS the file's size, so reopening the target for writing is the one operation Windows
  refuses — publish by rename, never by truncation; close the staging file before renaming or
  removing it, since the C runtime opens without `FILE_SHARE_DELETE`; and expect a rename to fail
  transiently while another process holds the target open, which is why the commit retries a
  bounded number of times.
