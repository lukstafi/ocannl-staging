# Build and test mechanics

The dune/OCaml mechanics behind CLAUDE.md's workflow rules, and what CI actually covers.

Part of the agent notes; the [index](../agent-notes.md) carries the scope discipline and the other
files.

CLAUDE.md holds the workflow rules; these are the dune/OCaml mechanics behind them, narrow enough
that they earn a lookup rather than always-loaded space.

- A repository-wide scanning check (`config_dep_completeness`, `env_var_deps`, `cache_dir_ignores`)
  is only as good as the distance between what it asserts and what it claims, and a proxy that
  coincides with the property today reads exactly like the property. The working test is: name a
  change that makes the claim false while the check stays green. Each of these was a real gap, and
  each looked airtight until asked that way — "the `.gitignore` line is present" is not "git ignores
  this name" (last-match-wins, so a later `!` takes the coverage away with the line intact); "the
  literal starts with the prefix" is not "the glob covers it" (a glob segment stops at a separator,
  `Schedule_cache.ensure_dir` does not); "every `~cache_dir` carries the prefix" is not "every cache
  directory does" (a direct `Schedule_cache.store ~dir` creates one too, and `""` disables the cache
  only where `Autotune.tune` reads it that way). Two corollaries. A scan that excludes files by
  filename suffix has built an escape hatch from its own "every source" claim unless the exclusion
  is anchored to the directory that generates them. And where the premise is "this value happens to
  equal that constant", read the constant from where it is defined rather than restating the
  coincidence: an unchecked default is the one name no call site spells.
- The repository's PROSE is checkable where its structure carries meaning, and the agent notes are
  the case in point: `agent_notes_structure` (gh-ocannl-691) reads `docs/agent-notes.md` and
  `docs/agent-notes/` as structure rather than as text. It holds five things true — no bullet cut
  short (each ends in sentence-terminating punctuation, which is what a merge splice destroys), no
  index hook absent from the file its row links, no table row wrapped across two physical lines
  (that ends the table and renders every row below it as pipe-delimited prose), no file unreachable
  from the index, and no bullet promoted into two files. Three of those were review findings on the
  split that created the files, each caught by a human reading carefully. What it costs when you
  append here: end the bullet with punctuation, indent continuations exactly two spaces past their
  bullet, keep one nesting level, keep every index row on one physical line, and give a new file its
  index row the day it appears.
- A scan over a tree that is USUALLY CLEAN needs synthetic negative controls more than one over a
  tree full of findings does, because green-because-intact and green-because-blind are the same
  output. So the rules live in `test/support/agent_notes_scan.ml` as pure functions over strings and
  `agent_notes_scan_cases` runs each of them on a violation it must flag AND on the nearest
  legitimate text it must not — a pipe inside a code span, a bullet ending in `.)` or in bold, two
  bullets that merely open with the same few words. The second half is not decoration: a rule that
  fires on ordinary writing gets switched off rather than obeyed, so every rule owes a demonstration
  that it does not.
- The `.expected` golden of such a repository-wide check should hold what is TRUE of the repository,
  not how much of it there is. A tally — "170 tests in this directory", "241 test stanzas declare
  the config" — moves on every correct addition anywhere, so every unrelated contributor has to
  promote a file they did not touch, and the promote is indistinguishable from blessing a real
  regression in that same file; worse, one hot line in one file collects a textual conflict from
  every parallel branch at once (gh-ocannl-665, where it was an arc's only rebase conflict). The
  counts are usually there as the "the scan did not go blind" signal, which is a real thing to keep
  — so keep it, elsewhere. Three places it can live, all churn-free: PRESENCE in the golden (which
  kinds of stanza a directory has, changing only when it gains its first or loses its last), a floor
  read by a SECOND reader that shares no machinery with the first (`Dune_stanza_scan.raw_stanzas`
  reads the stanzas and what each runs off the raw text, and a sexp walk going blind cannot take it
  down too), and the exact numbers on STDERR, which a `(test)` stanza does not diff. Assert the floor through `Verdict`
  rather than as a golden line, so a scan that goes blind cannot be promoted back to green. And
  compare the floor to the walk STANZA BY STANZA, never as two totals over a file: the second reader
  recognises fewer shapes than the walk — today an external command handed something the workspace
  builds (`(run python3 %{dep:x.py})`) — so every stanza the walk places and the floor misses is a
  unit of SLACK that can absorb a different stanza silently dropping out of enforcement. Not
  theoretical — the tree stood at 296 placed against a floor of 295, one whole stanza of cover.
  Print the gap ITEMISED rather than as arithmetic: "one short" does not say which stanza is
  standing on the walk alone, and the class it belongs to is what decides whether closing it is
  worth the loss of independence. Asked per stanza
  the two answers are about the same stanza and cannot be traded against a third, and the narrower
  vocabulary degrades to a weaker floor just where it is narrow instead of to a hole elsewhere. Note
  what this licenses: once the comparison is per stanza, the two readers may share the TRAVERSAL
  that pairs them, because the independence being protected was only ever in the classification. Put the
  independence in the CLASSIFICATION and nowhere else: what can go blind is the checker's own
  traversal and command-recognition, so answer those questions a second way — but PARSE the input
  with the same reader the format admits rather than re-deriving its grammar. Seven review rounds on
  gh-ocannl-665 went into a floor that hand-scanned dune's text, and the same lesson kept arriving
  from new directions: quoted atoms in a `chdir` destination, then in a command, then in a binding's
  value, then in a form's HEAD; whitespace after an open paren; comments where an atom may begin.
  Each was either a false failure or silent under-coverage, and the module's own opening line had
  already said why — an approximation of a grammar has no natural stopping point. Rewriting the
  floor on `Sexplib` dissolved four findings at once and shrank it.
  What the second reader still must agree about is SCOPE, and getting it wrong fails correct scans
  rather than blind ones: (a) STANZA POSITION — only top level and inside `subdir`, else `(env (test
  (flags …)))` reads as a test stanza when it names a build profile; (b) WHAT RUNS THINGS — only
  tests, rules and aliases, else a library's `(preprocess (action (run …)))` counts; (c) WHERE it
  runs — `subdir` and `chdir` compose into the directory whose config the process finds, and the
  comparison key is that composition, for a test's own `%{test}` as much as for a helper; (c')
  what CANNOT be resolved — under a `chdir` holding a pform the walk emits a site with no
  executables, so never report a literal `%{…}` as a directory; (d) GROUPING and IDENTITY —
  one site per distinct executable per directory, so raw occurrence counts fail a `progn` running
  one executable twice while a flat set lets five of six rules be answered for by the sixth, and
  identities come from a structured field (`site.executables`), never from splitting a display name.
  And where a floor matches against a CLASS of sites, check that the class is not wider than the
  thing being protected: "any unnameable site" let a `(bash …)` answer for a dropped
  PATH-rewritten one, and the fix was the same as for the executables — give the site a structured
  reason of its own (`site.path_rewritten`) instead of grouping by a shared symptom.
  A last trap in the other direction: declining what you cannot resolve is the SAFE direction for a
  floor and therefore the easy thing to over-use — declining `./%{pp}` left all three `test/ppx`
  rules with no floor at all, and the fix was to resolve the `(:pp pp.exe)` binding from the same
  stanza, which in turn means resolving commands only once the stanza has been read.
  The general form of that trap, and the one worth reaching for first: a shape the second reader
  cannot RESOLVE is usually still one it can COUNT. It does not have to parse a `(bash …)` line, or
  resolve a `(chdir %{…} …)`, to record that the stanza runs SOMETHING — which is the entire
  question a subject-or-not floor asks. Tag such a shape rather than dropping it (gh-ocannl-690),
  and carry the tag WITHOUT the piece that is unknown: a command under an unresolvable `chdir` goes
  into a directory-less list, because the per-directory floors would otherwise hold the walk's
  refusal to guess against the floor's guess. Then watch the OTHER direction, which is where tagging
  puts the risk: a floor that sees a stanza the walk does not fails a correct scan. Mirror the walk's
  own dropping rule at the tag site — a PATH tool is external wherever a `chdir` sends it, so the
  walk places no site and the floor must record nothing.
  And keep the rule's DECISION somewhere synthetic text can reach it. `env_var_deps`' XOR lived in
  its main loop, so "a rule running its test through `bash` is subject to the rule" could be argued
  and not asserted; `Scan.backend_rule_of` is that decision alone, with the diagnostics and tallies
  left in the check that owns their wording, and `dune_scan_cases` states the rule over stanzas the
  repository does not contain. Render the floor's answer ALONGSIDE the verdict in such cases
  ("declares neither +floor" versus the same line without it): the pairing is what makes a false
  green visible as a golden line rather than as an absence.
  The
  sibling checks are worth a glance when touching this genre and were both fine: `env_var_deps` lists
  names only, and `digest_completeness`'s key count moves only alongside its own enumerated key list
  — a number in the same commit as the change it describes costs nothing.
- Where a check needs an EXEMPTION per site, prefer an in-place marker comment to a central list,
  and give it a grammar rigid enough to be wrong out loud (gh-ocannl-659, the XOR between
  `(env_var OCANNL_BACKEND)` and `; ocannl-backend: <word> -- <reason>`). Two reasons, and the
  second is the one that decides it. A central per-site list is the churn and conflict magnet the
  bullet above is about. But the recurrence mechanism is that the next author copies the stanza
  NEXT TO the one they are writing — so the classification has to live there, in the file they are
  already editing, not in a list they will never open. Three things make such a marker checkable
  rather than decorative. (i) ATTRIBUTION BY CONTAINMENT, not adjacency: a comment counts for the
  stanza whose parentheses it sits between, because this repository's dune files habitually leave a
  blank line between a comment block and the stanza it introduces, and "the comment above" would
  have to guess how far above — and would hand a marker to the wrong stanza the first time someone
  left a note between two rules. (ii) A MALFORMED MARKER IS A FAILURE, never a shrug: a grammar that
  silently declines to parse leaves its stanza declaring nothing and reports it as if the author had
  written none, which is the worst of both. So the vocabulary is closed (`none|cc|multidev_cc|cuda|
  hip|metal`), the reason is required and must be more than one word, and the separator is the
  EARLIEST of the spellings admitted rather than the first one that occurs anywhere. The sharper
  form of the same rule, and the one a first draft gets wrong: NEVER NORMALISE WHAT YOU COULD
  REJECT. Filtering empty elements out of a comma list reads `cc,` and `cc,,metal` as a clean
  `cc`/`cc,metal`; deduplicating reads `cc,cc` as `cc`; reading from the earliest sentinel absorbs a
  second declaration on the same line into the first one's reason, where even the accounting check
  below cannot see it, both occurrences being in a comment the scan did place. Each of those is a
  tidy-looking answer to a marker its author got WRONG — and a construct whose entire purpose is to
  be checkable cannot afford to repair its own input. (iii) An
  ACCOUNTING CHECK over the sentinel: every occurrence of `ocannl-backend:` in the file, found by
  the dumbest possible substring scan, must have been read as a marker attributed to a stanza that
  runs something. That is what catches the marker written into a quoted argument, into a field, or
  into a stanza that runs nothing — cases where the author believed they had declared something.
  (iv) READ THE DECLARATION FROM THE FIELD THE ACTION RUNS UNDER, never from the stanza at large. A
  stanza can carry several dependency fields and dune reruns an action under exactly one of them —
  an inline-test library declares under `(inline_tests (deps …))`, and `(preprocessor_deps (env_var
  OCANNL_BACKEND))` in the same stanza reruns nothing that matters while looking, to any
  whole-stanza search, exactly like a declaration. So `site` carries `declares_backend` scoped to
  the same deps field as `declares_config`, and the XOR reads it from the sites; the two answers
  cannot drift because they come from one place. The general form: when a check pairs a claim with
  the thing dune will act on, scope the claim the way dune scopes it.
  The mechanical cost is that comments are what a sexp reader throws away, so the scan needs
  positions: `Dune_stanza_scan.read_raw` returns forms with their byte ranges plus every `;`
  comment, and its tree is compared SHAPE FOR SHAPE against sexplib's, which is strictly stronger
  than the flat form count it replaced and is what keeps a hand-written lexer honest.
- A ratchet whose corpus is the repository's own test sources will scan its OWN fixture, and a
  fixture that pins a shape has to spell that shape out to pin it — so the fixture matches, every
  time, by construction (gh-ocannl-668). Exempt the FILE rather than its matches one at a time: the
  case table grows a row whenever the reader learns a distinction, and a list of its labels
  elsewhere is a second copy of it, maintained by whoever adds the row and read by nobody. What
  keeps that blanket exemption from being a hole is a CANARY — name two of the fixture's literals
  and fail if the scan stops finding them — which is also the churn-free anti-blindness signal for
  this genre, since an empty offender list and an unread corpus are otherwise the same result.
  Spell one canary in a form only the real reader can see (a string literal broken over a line
  continuation, whose decoded value spans no single line of the file): it fails a scan that has
  quietly regressed to matching text, which the plain spelling would not.
- Scan OCaml sources through **ppxlib's** parse tree (`Ppxlib.Parse.implementation`,
  `Ppxlib.Ast_traverse.iter`), never `compiler-libs`'. The compiler's `Parsetree` moves between
  releases, and the breakage lands in the scanner as a compile error rather than in anything it
  scans: 5.5 alone gave `Ldot` located components and stopped spelling `let module M = … in …` as
  `Pexp_letmodule`, making it an ordinary structure item inside the expression — so no single arm
  compiles on both sides of that boundary. ppxlib parses with the compiler's own parser and then
  migrates the tree to an AST of ppxlib's version, which is what decouples a scanner from the
  compiler it is built by. That AST is `Astlib.Ast_502` — BELOW the declared OCaml floor, so every
  parse is a downgrade, and the reassurance is not "nothing newer can arrive" but that the 5.x
  downgrade chain performs no `migration_error`: a construct the older AST spells differently is
  mapped onto the older constructor (5.5's structure-item-in-an-expression becomes
  `Pexp_letmodule`) or carried in an attribute encoding. The corollary is that the coupling moves
  to ppxlib — **`dune-project` bounds ppxlib above** for that reason, since the ppx matches the
  same selected AST at some sixty sites. Match with
  ordinary patterns; metaquot quotations (`[%expr [%e? f] ()]`, `[%expr ()]`) are preferred wherever
  they express the WHOLE shape, since they ignore locations and attributes while keeping arity and
  labels exact. What they cannot reach — a variable-length argument list, a string constant's value,
  a module binding — stays written against the constructors.
- A documentation comment survives into the parse tree as an `[@@@ocaml.doc "…"]` attribute holding
  a STRING, so an iterator hands it to an expression hook exactly like code would (verified by
  removing the guard — the prose cases flip to findings). Any scan over string literals must
  therefore override `method! attribute _ = ()`, or this file's own documentation of the pattern it
  hunts becomes a finding. Ordinary `(* … *)` comments need no such
  care, the parser drops them outright. Extension payloads are the opposite call and must stay
  visited: `[%cd …]`, `[%expect {|…|}]` carry code, or the golden text of some.
- `git` strips TRAILING spaces from a `.gitignore` pattern (unless backslash-quoted) and keeps
  LEADING ones, so an accidentally indented ` /foo/` is a pattern beginning with a space and matches
  nothing. Any code reading that file must not `String.strip` both ends, or it reports coverage git
  does not give; `#` likewise opens a comment only at column 0. A backslash escapes the next
  character, in both directions that matter: `\_` is an underscore, so `!/foo\_bar/` really does
  un-ignore `foo_bar` while a matcher reading the backslash literally sees no match and reports it
  still ignored, and `\*` is a literal asterisk rather than a wildcard, so ignoring the escape
  over-matches as readily as it under-matches. A leading `**/` matches any number of directories
  INCLUDING ZERO, so `**/foo` and `/**/foo` both reach a root-level `foo` and cannot be dismissed as
  "contains a slash, so it is anchored elsewhere"; consecutive asterisks anywhere else are ordinary
  ones. All of this verified against `git check-ignore`
  rather than read off the documentation — which is the cheaper move whenever the question is what
  git actually does.
- `tools/test-run.sh` is the one way to run `dune runtest` / `dune build @slow` from a session;
  its header documents usage. It exists because every hand-rolled alternative has failed in
  practice, each differently: piping dune to `tail` reports tail's status (no pipefail), so
  promotion diffs read as green; a wrapper variable named `status` is read-only in zsh, so the
  assignment dies before the sentinel prints and a green suite looks failed; waiter loops on
  `pgrep -x dune` match the editor's immortal `dune ocaml-merlin` daemons and spin forever (one
  review session accumulated ten stranded waiter shells); `kill -0 $pid` can latch onto a
  recycled pid; and an uncapped hung run (macOS XProtect stalling a fresh exe, a wedged backend)
  strands whatever waits on it. The script runs dune unpiped under a wall-clock cap that kills
  the whole process group, records the verdict in a FILE (`wait` polls that file under a hard
  timeout — it structurally cannot strand), and holds a per-worktree flock so a second
  invocation refuses loudly instead of queueing behind dune's lock — "I lost track of a run so
  I started another" being the usual start of the spiral. Prefer foreground `run` launched
  through the agent harness's background mode (the harness notifies on exit); `start`/`status`/
  `wait`/`stop` are only for runs that must outlive the launching session.
- `(copy_files ...)` creates PASSIVE rules: they do not fire just because you build a sibling target
  in the same directory — only when listed in that target's `(deps ...)` or requested explicitly. A
  rule consuming copy_files output must therefore declare it. And validate a `(mode promote)` target
  from a clean state (`dune clean && dune build @alias`): stale `_build/` intermediates can satisfy
  an undeclared dep, so an incomplete build passes while the artifact is wrong. Assert content
  (size, object counts), not mere existence.
- A test asserting on generated code must establish that the artifact it reads is the one THIS run
  emitted; `Test_utils.Generated` is how (gh-ocannl-655). `build_files/<exe>/<routine>.<ext>` is a
  side effect of a compile, not a value the test holds, and two things detach it from the compile it
  describes: `test/config/ocannl_config` keeps `clean_up_build_files_on_startup=false`, so an
  artifact outlives its run indefinitely, and a second compile under the same routine name overwrites
  it within a run. Either way the assertion outlives the kernel it asserts on — it keeps passing, and
  keeps counting as coverage, after that kernel stopped being emitted at all (folded to a constant,
  erased by precision inference, fissioned into a differently-named routine). `Generated.init
  ~backend_name`, called before the first compile, empties this executable's own subdirectory, so
  existence IS freshness — no mtime, no clock granularity. What licenses that sweep is narrower than
  "the directory is scoped": only the DEFAULT, executable-derived subdirectory is inherently
  process-private, since dune runs one process per executable. Any configured `build_files_prefix`
  is refused outright — a second executable can be given the same prefix, so deleting there is
  unsafe, and without deletion a deterministic compile's re-emitted identical kernel is
  indistinguishable from a stale one (deletion is the only write signal that does not depend on
  timestamp granularity). Tests that assert on generated code therefore leave the prefix at its
  default. `Generated.read` fails through `Verdict` on a
  missing artifact instead of answering `None` — the arm that some call sites recorded as `false` and
  others forgot. `Generated.arm` deletes one routine's artifact before a candidate's compile, which
  is what a loop reusing a routine name needs in order to attribute what it reads; reading one
  routine twice across changed contents is otherwise reported as an unattributed overwrite. Corollary
  for a leg this backend cannot evaluate: gate it and report `Verdict.skipped` rather than letting it
  reach the read, because an absent artifact is a failure here by design.
- Dune roots at the OUTERMOST ancestor holding a `dune-workspace` (failing that, a `dune-project`)
  and ignores dot-directories, so from a worktree under `.claude/worktrees/` the main checkout wins
  and the worktree is invisible to dune: targeted commands fail with `Don't know about directory
  .claude/worktrees/...`, while a bare `dune build`/`dune runtest` quietly builds and tests the
  PARENT branch. `scripts/setup-ocaml-env.sh` writes a one-line `dune-workspace` at the worktree
  root, restoring it as the root with its own `_build`. The step tests the ancestor DIRECTORIES
  rather than git topology, since a checkout can nest inside another checkout that is itself a
  linked worktree living anywhere, and `--git-common-dir` then names the primary checkout, not the
  one dune would root at. That file is generated per worktree and gitignored, never committed —
  being the outermost, a tracked copy at the repo root would shadow every worktree's and pin them
  all back to the parent (the script reports `FAIL` for a `dune-workspace` in any ancestor, which
  it cannot override from below).
  With it in place, `--root .` and `dune promotion apply` are no longer needed from a worktree;
  `tools/promote.sh` remains the Windows path, for the CRLF stripping. Worktrees placed outside the
  repo need none of this, but see no `ocannl_config` on their ancestor path.
- Dune tracks an environment variable only where a stanza declares it, and the tracking reaches
  further than the stanza: `dune rules test/operations/<name>.exe.output` shows the `(Env
  OCANNL_BACKEND)` dependency travelling from the `(test)` stanza's `(deps ...)` into the
  `.exe.output` rule dune generates from it, so `OCANNL_BACKEND=cuda dune build …exe.output`
  really does re-run the test on cuda. What it does NOT do is tell you it ran there: a
  backend-uniform golden (GPU legs announcing themselves on stderr while printing the same
  `<claim>: true` on stdout) makes the cuda `.exe.output` byte-identical to the cc one, which is
  how gh-ocannl-622 came to read a cc-looking file as proof the recipe was broken. It was the
  inference that was broken; the recipe holds for DECLARED variables, and gh-ocannl-628 is the
  hole that was real — the lowercase spelling `read_env_var` consulted first was declared nowhere,
  so `ocannl_backend=metal` decided the backend while invalidating nothing. gh-ocannl-652 closed it
  from the other end: the environment has ONE spelling, `OCANNL_<KEY>`, and setting a lowercase or
  dashed spelling of a known key aborts the run with a message naming the spelling that works, so
  the variable cannot quietly decide nothing either.
- Deleting a file target out from under dune is not a way to force it to re-run: `dune build
  <that target>` afterwards exits 0 having produced nothing (observed on dune 3.23.1 with
  `test/operations/<name>.exe.output`), and `-f/--force` does not rescue it — `--force` only
  re-runs actions attached to ALIASES. Either force the alias (`dune build --force
  @<dir>/runtest`), or run the built exe directly with its cwd set to `_build/default/<dir>`, which
  is exactly the environment dune gives it — the same cwd, hence the same `ocannl_config` search
  root, that makes `dune exec` unusable (CLAUDE.md). The cause is that dune trusts its own digest
  database and never stats a rule's targets, so a hand-deleted one is recorded as built forever;
  that also rules out the two other reflexes, since touching a source changes no CONTENT digest and
  deleting `_build/.digest-db` does not restore the memo either. For a target with no alias of its
  own — an `(executable)` plus a `(rule)` producing `<name>.actual`, which is how every scanning
  check is built — the recovery is `dune build --sandbox=copy <that target>`: sandboxing changes
  how the rule executes, which invalidates the memo and re-runs it. `dune clean` works too and buys
  a full rebuild, which on macOS means every fresh executable queueing behind XProtect again. Worth
  knowing before it bites, because the failure is silent in the dangerous direction: the missing
  target leaves whatever `.actual` was there before, so a probe that only diffs the file reads
  green while nothing has run.
- A record with `[@@deriving sexp]` makes every `.expected` file that prints the parent a hidden
  consumer of its FIELD NAMES, and `rg "\.field_name"` over sources is vacuous against that (sexp
  prints `(field_name value)`, not member access). Before claiming a rename has no serialization
  consumers, grep the sexp shape: `rg -F "(field_name " --glob '*.expected'` (the trailing space
  disambiguates longer identifiers). Budget the resulting promote as expected work, in its own
  commit, after diff-confirming the delta is rename-only.

- `dune build @check` type-checks; it does NOT link executables. A "rebuilt" test or benchmark exe
  after a library change is therefore the STALE binary — this has produced a false verdict three
  separate times (a timing rerun, a negative control, a guard "verified" against the old code). Build
  the thing you are about to run (`dune build <dir>/<name>.exe`, or the test's alias / `.exe.output`
  rule) and check its mtime against the sources you edited. In the other direction, a plain
  `dune build` RUNS every cram-style test executable, so it must not overlap a GPU timing window.
- Most `test/operations` stanzas preprocess with `(pps ppx_here ppx_ocannl)` and nothing else, so
  `[%equal: …]` / `[%compare: …]` are unavailable — spell the comparison out (`Option.equal
  Int.equal`) rather than extending the stanza for one line.
- OCaml/Base traps that each cost a debugging session here: `Base.List.init` applies its function in
  DECREASING index order, so building a list of runs with it executes them backwards (use an explicit
  loop when the elements have effects); OCaml 5's `Lazy.is_val` returns TRUE for a lazy that is
  mid-force (forcing it then raises `Lazy.Undefined`), so it cannot guard a force reachable from
  inside that lazy's own computation; two record types sharing a field name resolve to the
  LAST-defined type, silently mistyping `x.a.b` (which is why the scheduler's event field is
  `dev_state`); and `Base.Float.max_value` is INFINITY — the finite maximum is `max_finite_value`.

### What CI actually covers

- GitHub CI exercises exactly ONE backend. `test/config/ocannl_config` pins `backend=cc` and the
  runners have no GPU, so a green `ci` run says nothing whatever about Metal, CUDA or HIP. Do not
  read a green PR check as cross-backend validation; it is a CPU-backend and portability check.
- Windows is off the per-PR matrix (62-74min against 20 on macOS and 29 on ubuntu) and runs on a
  twice-weekly schedule, together with an ubuntu job on the OCaml floor the opam files claim
  (`>= 5.3.0`, against 5.5 everywhere else). Both are reachable on demand through
  `workflow_dispatch` with `extended: true` — dispatch from a branch when touching `.expected`
  goldens or the cc backend's toolchain handling rather than waiting for the sweep, since those are
  the changes that actually break on Windows (line endings, float formatting, mingw). Twice weekly
  rather than weekly because actions/cache evicts entries unread for 7 days, and an exactly-weekly
  cadence would pay the cold-switch cost every time. The two ride the same cadence because they
  fail the same way: slowly, and through the dependency cone or the toolchain rather than through
  a change under review.
- The Windows job ends with a smoke of `tools/test-run.sh` itself, from Git Bash: `run`, `status
  last`, a deliberately failing target, `start`/`wait last`, `list`, each asserted against its
  documented exit code. That script's MSYS branches (unconditional and fatal `opam-env.sh` sourcing,
  dune routed through `dune-quiet.sh`, the tokenless liveness degradation where MSYS `ps` has no
  lstart/state columns, the flock through MSYS perl) execute nowhere else, and are otherwise reached
  only from rare hand-run Windows sessions — where their rot is discovered mid-task. It builds
  `tools/promote.sh`, a source file dune copies rather than compiles, so it costs a workspace scan;
  the failing-target leg is the load-bearing one, since it is where a dropped `PIPESTATUS` in
  `dune-quiet.sh` would report red runs as green. It runs even when `dune runtest` above went red
  (Windows runs twice a week; a golden drift must not mask the runner's health for three days) and
  it runs last, so a broken runner cannot abort the sweep's only Windows test coverage.
- That smoke step has itself been run on real MSYS (rog, Git Bash, under GitHub's exact
  `bash --noprofile --norc -eo pipefail`), so it is a confirmed test rather than an untested one.
  All four MSYS branches were checked directly and hold: `opam-env.sh` sourced from a PATH with
  neither `dune` nor `ocamlfind` yields a toolchain that LINKS (not merely compiles), and refuses
  fatally with exit 2 when opam is absent; `dune-quiet.sh` preserves dune's status through its
  stderr filter (1, 0, and an arbitrary 7 from a shim) while dropping only the `.drectve` lines;
  MSYS perl's flock genuinely excludes a second process; and `proc_alive` does not in fact need
  its tokenless fallback there — MSYS `ps` has no `-o` at all, but MSYS *does* provide
  `/proc/<pid>/stat`, so `ps_token` takes the Linux branch and records real tokens.
- What that run FOUND: under MSYS with the default `winsymlinks` mode, `ln -s` does not create a
  link, it silently COPIES — so `ln -sfn <run-dir> last-<key>` left a full copy of the run
  directory where the pointer belonged, and every `last` lookup afterwards resolved to nothing.
  `test-run.sh` now points `last` with a plain file holding the path (written through a temporary
  and renamed, so still atomic), which needs no symlink privilege on any platform. Reach for a
  pointer file, not a symlink, in anything that must work from Git Bash.
- `tools/sweep.sh` is the GPU-backend coverage: cc and metal locally, cuda on `rog-nv-wsl`, hip on
  `minix-amd-wsl`, all pinned to ONE resolved commit so a mid-sweep merge cannot leave the machines
  testing different trees. It records a row per unit in `~/.ocannl-sweep/history.tsv` and never
  exits non-zero for test failures — its exit code is not a verdict, the history file is. A daily
  scheduled task drives it.
- The GPU boxes are usually powered off, so `skip (unreachable)` is the normal outcome and a sweep
  of skips is not a failure. What IS a failure is silent non-coverage: track the age of the last
  `pass` per backend, because nothing else in the project tests CUDA or HIP at all.
- Report changes in the failure set, not the presence of failures. A backend's suite goes red in
  bursts and comes back (Metal's `test/operations` was red for a stretch, green again after
  gh-ocannl-632), so a sweep that shouts on every red is one that gets ignored inside a week;
  `sweep.sh` writes a sorted `.fingerprint` next to each non-pass log precisely so the previous
  run's can be diffed against it.
- The per-machine worktrees are reused, not recreated, so a sweep is incremental against an
  existing `_build` — seconds rather than minutes when little changed. That is what makes a daily
  cadence affordable; it also means a sweep is not a clean-tree build, and a genuine
  from-scratch check still wants `dune clean` or a fresh CI run.
