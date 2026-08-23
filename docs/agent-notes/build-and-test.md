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
- GitHub builds a pull request's MERGE COMMIT, so a repository-wide scan that is green on your
  branch is not evidence about the tree CI will scan. `agent_notes_structure` (gh-ocannl-691,
  staging#413) survived nine review rounds, `dune build @check`, its targeted aliases and a
  deliberately corrupted copy of the notes, and the merge gate then refused it on nine defects in
  `docs/agent-notes/scheduling-and-autotune.md` — a correctly written note that landed on master
  while the PR was in review, so it existed only in the merged tree (187 bullets against the
  branch's 179). Nothing runnable locally sees that tree unless you bring the base in yourself. This
  matters more for a scan than for an ordinary test, and the difference is the blast radius: a unit
  test fails when its own subject regresses, and its subject is in the diff, whereas a scan fails
  when anyone, anywhere, writes something it did not anticipate — and the population it scans keeps
  growing under it for as long as the PR is open. So fetch the STAGING remote — whose name is local
  and need not be `origin` (CLAUDE.md, Pull Requests) — rebase onto its `master`, or merge it in
  where the branch is shared and rewriting is not yours to do (as #413 did), and re-run the scan
  BEFORE opening such a PR and again before merging it; where neither is welcome, build the merge
  commit on a scratch branch and run it there. What the omission buys is a false failure on a
  colleague's correct work, which is the outcome that gets a check disabled rather than fixed.
- A negative control written FROM the corpus can encode the ABSENCE of a shape rather than a rule
  about it, and that is the more expensive half of the same story. #413's fixtures came from a
  survey of the notes as they stood; the survey found no bullet continued after a blank line, so
  that shape reached no fixture, and the control that came nearest — `- A finished fact.`, a blank
  line, then an indented line — asserted a finding, because a finding is what the scanner gave. In
  Markdown that is a second PARAGRAPH of the same bullet and is correct, which is precisely the
  writing the merge commit failed. A control that asserts the buggy reading is worse than no control
  at all: it makes the wrong behaviour look deliberate, and the next reviewer reads it as the
  specification. So write controls from the RULE, not from the tree — state what the format admits,
  SYNTHESIZE the violating text and the nearest legitimate text beside it, and check the reading
  against the format rather than against what the code answers today. A shape the corpus does not
  contain is the one most in need of a fixture, since nothing in the tree will contradict the
  implementation's guess about it; a survey reporting zero of something is a hole in the fixtures,
  not permission to leave that case undefined.
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
  repository does not contain.
  gh-ocannl-723 is the pattern's second instance and shows what a scanner rule costs once the seam
  exists: the rule is `Scan.artifact_subjects`, `dune_scan_cases` puts it to fifteen stanza/source
  pairs the repository has no member of, and `config_scan_lexing` does the same for the source side
  — where the hostile input is the repository's own, since `test/support/generated.ml` names
  `Generated.init` in half a dozen doc comments and `generated_provenance.ml` asserts on a string
  literal quoting one, so a text scan would read the module that DEFINES the initializer as its
  heaviest caller. Two lessons beyond the pattern. First, a decision-table control still leaves the
  wiring from decision to failure unexercised, and `env_var_deps --control` closes it: it builds a
  synthetic tree containing the violating pair, hands it to `env_var_deps.exe` as a CHILD, and
  claims that the child names the stanza and exits 1 without the declaration and exits 0 with it —
  the `generated_provenance` capture pattern, and worth the fixture because the tree is DERIVED from
  the check's own exemption and gateless lists and so cannot drift from them. Second, a vacuity
  floor over a repository census cannot be asked of an arbitrary tree: the artifact caller floor
  applies only when `Config_key_scan.floor_violations` says the run was handed the repository's scan
  roots, and WHICH mode the run was in goes into the golden, so a glob that breaks flips that line
  rather than quietly retiring the floor.
  Four shapes such a scan gets wrong quietly, all found in review, and each is a member of a genre
  rather than a one-off. **Identifiers**: a scan that matches a function through the module it
  belongs to must collect the module's local names from BOTH grammars — OCaml spells binding,
  opening and including twice (`module G = M` / `let module G = M in …`, `open M` /
  `let open M in …`, plus `include M`) — and must consult what it has already recorded, since
  `module H = G` names the module as surely as `module G = Test_utils.Generated` does.
  **Ownership**: a converse check ("this declaration has nothing behind it") must say which stanza
  owns each verdict, or the same fact is reported twice — a rule that runs an executable is judged
  through that executable, one that runs something unnameable is not judged at all, and only a
  stanza that runs nothing whatever answers for its own declaration. **Path identity**: match a
  runner to an executable by the path AS WRITTEN, never by basename, or a rule running
  `../support/probe.exe` credits a local `probe` with a declaration made elsewhere — the same
  collapse the config scanner's duplicate-basename check exists to prevent. **Dune's defaults**: a
  stanza with no `(modules …)` field, or one naming `:standard`, owns the directory less what other
  stanzas claim; reading either as "names no modules" makes a required declaration come out stale.
  **Dune's identities**: an executable is run under `<name>.exe` and under its `public_name` via
  `%{bin:…}`, and a runner named the second way is still its runner. **Subdirectories**: ask the
  question per `(subdir …)` group, not per dune file — the stanzas inside one name modules that live
  there, and a walk that stays at the top level reports a nested source as claimed by nobody.
  **Wrappers over paths**: a signature constraint (`module G : module type of M = M`) wraps the path
  without changing what it names, so unwrap `Pmod_constraint` recursively before resolving an alias.
  **Cross-group relations**: descending into `(subdir …)` is only half of it — a rule outside the
  wrapper runs the executable inside it under the qualified path, so match runners over the whole
  file while resolving modules per group, or descending finds both stanzas and then discards the
  relation between them.
  Two more, about the rule rather than the scan. A declaration is justified by ANY read of the key it
  tracks, so phrase the rule over the key (`Config_key_scan.source_reads_key`) and not over the one
  function that prompted the check — otherwise the documented way of pinning a key becomes unusable
  for that key. And having widened it that far, widen it in BOTH directions: permitting a
  declaration for a direct reader while requiring one only of the function's callers leaves exactly
  the stale run the check exists to prevent. A third: `inline_tests` on a library does not make its
  modules test-only, so a rule that is about what a module DOES when linked (here, emptying the
  artifact directory) must not accept a declaration that invalidates the inline-test runner alone.
  One more trap that costs a debugging round: dune's `glob_files_rec` runs over the BUILD tree,
  where a `<name>.pp.ml` sits beside every ppx-using `<name>.ml` — and a `.pp.ml` is not input the
  compiler's own parser accepts. A scan that parses everything it is handed (rather than only the
  modules a stanza names) must filter through `Config_key_scan.sources_among` first. Render the floor's answer ALONGSIDE the verdict in such cases
  ("declares neither +floor" versus the same line without it): the pairing is what makes a false
  green visible as a golden line rather than as an absence.
  The
  sibling checks are worth a glance when touching this genre: `env_var_deps` lists
  names only, and `digest_completeness`'s key count moves only alongside its own enumerated key list
  — a number in the same commit as the change it describes costs nothing. Its count of SOURCE files
  was a different matter, and the bullet below is what became of it.
- The same genre had a second instance in the two CONFIGURATION scanners, `test_config_consistency`
  and `digest_completeness` (gh-ocannl-701): their goldens pinned the size of the globbed corpus
  (`Sources scanned: 89 -- arrayjit/lib 47, …`), so every PR adding a source file anywhere under the
  six library directories owed a promote round it had no other reason to make. Note which way round
  the danger ran, because it is the opposite of the noise: two branches adding files in DIFFERENT
  directories write different text on that one line and conflict, which is loud and mechanical,
  while two adding files in the SAME directory write identical text, merge cleanly, and leave the
  total wrong by one for the next unrelated PR to inherit as a red test it did not cause. What
  replaced it is `Config_key_scan.scan_root_floors`, a hand-written lower bound per globbed ROOT,
  asserted through `Verdict` and itemised on failure so the message names the root standing on
  nothing; the goldens keep the root NAMES, since the globs' reach is a fact about the repository
  and a new or vanished root still reads as a diff, and the counts go to stderr.
  Set such a floor well below the day's count and leave it there — the number to raise it to is
  never "today's count", which is the tally coming back in a slower form. Bucket by the configured
  ROOT rather than by a path's own dirname, which is the trap a per-directory tally sets for its
  replacement: these rules glob with `glob_files_rec`, so a source in a subdirectory would otherwise
  open a bucket of its own — a new golden line, which is the promote round being removed, and under
  no floor at all.
- That floor machinery is `Test_utils.Scan_floors`, shared by the configuration scanners and by
  `codegen_text_inventory`: pass it the root table, and get the census-by-root, the itemised
  diagnostic and the stderr report. Each scan keeps its own table, since which roots it globs and
  how far each may fall are facts about that scan.
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
  The dune side of that: `init` READS `build_files_prefix`, so the stanza dune runs the test under
  must declare `(env_var OCANNL_BUILD_FILES_PREFIX)` — otherwise dune serves the previous run's
  result when the variable changes, which is gh-ocannl-628's hole one key over. `env_var_deps`
  requires it of every stanza whose `(modules …)` name a source that calls the initializer, and
  reports a declaration with no caller behind it as well (gh-ocannl-723). Where the declaration goes
  is dune's semantics and not one rule: a `(test)` runs under its own `(deps …)`, an inline-test
  library under `(inline_tests (deps …))`, and an `(executable)` has no `deps` field at all, so it is
  the rule that RUNS it that carries the declaration — the same placement as the `ocannl_config` dep
  and the backend marker, and checked as such (a declaration on a NEIGHBOUR of that rule reruns the
  neighbour, so it does not count).
- `Verdict` gates a claim by exit status, and a claim whose LABEL is computed needed an entry point of
  its own: `Verdict.pf fmt … b` is `p` with the label rendered from arguments
  (`Verdict.pf "%s gradients match the oracle" leg ok`), and `Verdict.claimf` is `claim` the same
  way. The boolean comes last, after the format's arguments, so the call reads in `p name b` order
  and forgetting it is a type error rather than a silent no-op. Before they existed every
  computed-label claim was a bare `printf`, which exits 0 on `false` and lets the failure be
  `dune promote`d into the golden — the whole of gh-ocannl-624. `verdict_ratchet` now holds both
  shapes, its reader having widened twice: the separator vocabulary is `:`, `=` and `->`, since
  reading only a colon left the entire `… = %b` population outside a check written to catch exactly
  it; and a format may carry other conversions, since a computed label carries one by construction.
  That second widening costs the old escape hatch — a descriptive `%b` print can no longer excuse
  itself by interpolating what it describes — so a census row that ends on its boolean takes a named
  exemption, keyed by the format HEAD (everything up to the `%b`) rather than the whole format,
  because the whole format is itself the claim shape and a list of them would force the check to
  exempt its own file. Every exemption in the tree is a row whose assertion is claimed separately
  beside it through `Verdict.claim`/`claimf` on the same bound boolean; an exemption without that is
  one that should not have been granted.
- A quantified claim needs a non-emptiness guard, and the guard has to be the SHORTEST thing to
  write or it does not get written. `p "every seed spreads j" (List.for_all seeds ~f:…)` is `true`
  when `seeds` is empty, and the line it prints is byte-identical to one a hundred seeds passed —
  the gh-ocannl-601 hazard arriving through `Verdict.p` itself, invisible in the golden by
  construction (gh-ocannl-729). It bites hardest where the claim is worth most: quantified over a
  DERIVED collection — the seeds a family tree yields, the refutations a gate raises, the
  candidates that validated — where empty is a plausible regression rather than an impossible
  state. So `Verdict.p_all name xs ~f`, `p_none name xs ~f` (the mirror: filtering an empty list
  also yields nothing, so `List.is_empty (List.filter …)` has the same hole), `p_exists`, and
  `p_empty name ~over:population derived` for the sites that keep the derived subset around to
  report it. A non-empty collection prints exactly what `p` prints, which is what let ~44 files
  convert with their goldens unmoved; an empty one prints `<claim> (empty): false`, and `?min:n`
  prints `<claim> (only 1 of 4): false`. Arrays go through `Array.to_list` rather than growing a
  second family. What stays on the unguarded spelling is the claim whose passing reading IS
  emptiness — "no candidate declines", "no key is undocumented", a scan over a tree that is usually
  clean — and those want a companion claim that the population was there at all, which is what the
  `p_empty ~over` form is.
- Guarantees that fire only on an empty collection are never exercised by a green suite, so
  `verdict_quantified` stages them: the satisfied forms run directly, and each refusal runs as a
  CHILD process whose streams the parent captures. Capturing is not tidiness — a refusal prints
  `FAIL:`/`FAILED:`, this repository's failure marker, so an inherited stream puts those words in a
  green run's log and costs `grep FAIL` its meaning (the same argument `generated_provenance`
  makes). The second half of that test is the one that makes a wide sweep safe: it runs `p` and
  `p_all` in two children and requires their stdout to be equal, which is the property "converting
  a site does not move its golden" stated as a check rather than as a hope.
- `Ll_test`'s traversal is the one place a new `Ir.Low_level` constructor is handled, and it now
  carries the queries the hand-built-IR tests used to write for themselves. `walk` takes a record of
  hooks: the construct-specific ones, a generic `?on_stmt`/`?on_scalar` for a counter that names its
  own shape, and an `?on_exit` — which is what makes an enclosing-context query derivable from the
  same walk instead of a second one. `?in_scopes:false` selects the statement-positions-only reading
  (the walk `Schedule.find_loop` had before gh-ocannl-668); every other query leaves the
  `Local_scope` descent on, and deciding it here once is the point, since `schedule_partition` alone
  had grown six walks each deciding it independently while scope nesting was the property under
  test. Above them sit `loop_sites`, `find_loop`, `find_loop_with_extent`, `find_nest`, `binds_loop`,
  `count_loops`, `first_binding` and `census`. Reach for `find_nest ~outer_n ~inner_n` rather than
  locating by extent alone wherever an initialization loop can share the reduction nest's extent:
  matching on extent takes the earlier one in preorder, which is how a `Partition` leg came to unroll
  an init loop and report `copies = 1` while exercising nothing. The walk also descends into
  `Tile_mma`'s `fallback` rather than stopping at the tile, and reports the operands through the
  fallback alone — reporting the tile's own `d`/`a`/`b` as well would count every operand of a
  tensorized nest twice.
- A test operand minted from a FLATTENED offset stops discriminating at sizes that divide its
  modulus, and does it while still looking right: `(row * stride + col) mod p` collapses to
  `col mod p` whenever `p` divides the row stride, so every row becomes identical. That makes a
  whole class of bugs invisible — a transform substituting or repeating the wrong row, panel or
  K-block computes the correct output, which no whole-output check, checksum included, can see, so
  it is found by review rather than by a red test. `Ll_test.cycle` (multi-index) and
  `Ll_test.cycle_flat` (flat offset) compute exactly the values the hand-written idiom did and raise
  `Invalid_argument` when the modulus is blind to an axis of the `~dims` handed to them, which turns
  a latent trap into a loud failure the moment a size arms it. Converting a site is therefore free:
  `Float.of_int (i % 13) *. 0.25` is `~modulus:13 ~offset:0. ~stride:0.25`, and `(x *. s) -. c` is
  `~offset:(-. c /. s) ~stride:s`, so no golden moves. The care is in `~dims`, which must be the
  operand's real row-major shape read off its `NTDSL.init`/`TDSL.ndarray` call (`~batch_dims` then
  `~output_dims` then `~input_dims`), not the `Array.init` argument. What this does not buy is
  aperiodicity: the values repeat with period `modulus`, so a shift by `modulus` is a symmetry, and
  where the blocking factors are searchable a packed panel can repeat under `k -> k + p` and hide a
  panel-substitution bug just as well; the recipe with no shift symmetry at any lag is
  `bin/narrow_gebp_bench.ml`'s `mix`. A formula that is NOT a flat index —
  `(i0 + i1 + 2*i2 + 3*i3) mod 7` — is not this class and needs no conversion, as long as no
  coefficient is a multiple of the modulus.
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
  The same hook also fetches `origin master` (bounded, best-effort: offline prints `skip`) and
  prints a `WARNING` with the commit count and the recovery (`git merge --ff-only
  refs/remotes/origin/master`, or a rebase when there are local commits) whenever HEAD is behind it — because a worktree is
  created from the MAIN checkout's HEAD, whose `master` only moves when someone fast-forwards it
  after a merge, so a new worktree can start dozens of commits stale (79 on 2026-08-22) and a
  full suite run then tests old code. Read the checklist before the first build.
  That section has a hand-runnable harness, `scripts/test-setup-ocaml-env.sh` — run it after
  editing the section; it is on no dune alias, since its `bounded` legs sit out watchdog timeouts.
  It copies the WORKING-TREE hook into throwaway clones under a `mktemp -d` (never touching this
  repository's refs or config) and covers the watchdog (TERM at the bound, KILL 5s later, the
  process GROUP, rc preservation, no orphans), the counting wording and its two recovery commands,
  the offline `skip` with the count taken as of the last successful fetch, a branch and a tag both
  named `origin/master` not displacing the tracking ref, `FETCH_HEAD` left byte-identical, and
  which SSH launcher git ends up invoking with or without the appended OpenSSH options. Both
  harness bugs it exists to prevent were live during PR #430's review rounds: a throwaway clone
  that silently tested the COMMITTED script, and a `run` helper that executed its label as a
  command. When adding a leg, add the negative control too — mutate the hook and check that leg,
  and only that leg, goes red. The harness found one bug on its first outing: `bounded` decided
  whether to wait for its watchdog with a `kill -0` on the command's process group, and `kill -0`
  counts a ZOMBIE as present — git's ssh child is one, reparented when git exits and not yet
  reaped — so a fetch that had already failed in milliseconds read as still running and sat out the
  whole 30s bound, on every session start with an unreachable ssh remote. Emptiness is therefore
  not a signal question: `group_alive` reads process STATES, from `/proc` where there is one and
  from `ps -A -o pgid=,stat=` otherwise, and only a non-zombie member counts as work. Where the
  reaper is a PID 1 that does not reap — the ordinary container case — the zombie is PERMANENT, so
  the first attempt at this, a short retry loop around the same `kill -0`, would not have helped;
  that is the shape to keep in mind before reaching for a timing fix here again.
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
  deleting `_build/.digest-db` does not restore the memo either. Every golden-diff rule now has an
  alias to force (`dune build --force @<dir>/runtest-<name>`, see below); for a target with no alias
  at all the recovery is `dune build --sandbox=copy <that target>`: sandboxing changes
  how the rule executes, which invalidates the memo and re-runs it. `dune clean` works too and buys
  a full rebuild, which on macOS means every fresh executable queueing behind XProtect again. Worth
  knowing before it bites, because the failure is silent in the dangerous direction: the missing
  target leaves whatever `.actual` was there before, so a probe that only diffs the file reads
  green while nothing has run.
- **Before changing code generation, read the inventory**: `dune build
  @test/operations/runtest-codegen_text_inventory` prints, as its golden, every file in the tree
  that pins the TEXT of emitted code (gh-ocannl-712). Two populations, and no single search finds
  both. Goldens holding emitted kernel or IR source live in `test/` and in `arrayjit/test/` — a
  scan of one tree is how gh-ocannl-623's first CI run went red, since three `arrayjit/test`
  goldens quote emitted constants. And some tests pin emitted text from a string literal in the
  `.ml` rather than from a golden (`Generated.assert_emits ~contains:…`, or `Generated.read`
  followed by a substring test), which no `.expected` scan can see; those are the expensive miss,
  because they are `Verdict` claims and so exit nonzero, failing a plain `dune build` rather than
  only `dune runtest`. Each source entry itemises the fragments it pins, `sprintf` formats and
  concatenations included with the hole shown (`"(float)(" ^ ... ^ ")"`, `"< (int)(%d.0))) {"`) —
  a range guard's bound is a float `Constant` at index precision, so gh-ocannl-623 turned
  `(int)(33)` into `(int)(33.0)`, a context nobody would think to grep for. Grep the inventory for
  the spelling you are moving, re-run what it names, and promote its own golden last.
- Each golden in that inventory carries the family that must re-record it, and the DECLARING
  extension wins over the markers: a `.hip.expected` spells CUDA's `__global__` launch vocabulary
  and is still HIP. A fragment the scan cannot name at its call site — text a helper computes —
  marks that file's itemisation partial rather than dropping it, so the file is still listed and
  the re-run is still called for. Markers are read only outside `Verdict` claim lines, since a
  claim label is prose ABOUT a kernel and freely quotes its vocabulary ("padded GPU intrinsics fire
  against the threadgroup fragment").
- Every rule that diffs a golden — the repo-wide scans, the codegen snapshots, the config-precedence
  rules, the ppx-output diffs — carries its own `runtest-<name>` alias (gh-ocannl-726), so
  `dune build @test/operations/runtest-verdict_ratchet` runs that one test, applies its diff and is
  promotable, and a misspelling exits 1 (`Alias … is empty`) rather than doing nothing. The seven
  repo-wide scans additionally share `dune build @test/operations/scans` (gh-ocannl-703): the family
  runs in seconds, and is what to run after touching a config key, a dune stanza, a printed claim or
  an agent note. Two properties of dune shape how those aliases are written, and both are silent
  when got wrong:
  - **A rule attached to two aliases makes building EITHER build BOTH.** So a golden-diff rule sits
    on `runtest-<name>` *alone*, never on `(aliases runtest runtest-<name>)` — that spelling type-checks,
    passes its own test, and quietly puts the whole directory behind every per-test alias (measured:
    `@test/operations/runtest-verdict_ratchet` ran the entire `test/operations` suite). What keeps
    those rules in plain `dune runtest` is an `(alias (name runtest) (deps (alias runtest-<name>) …))`
    stanza per directory — the same shape PR #431 gave `slow`. `env_var_deps` fails on a member the
    stanza omits, and on a golden diff attached to `runtest` itself.
  - **`runtest-<name>` is dune's own namespace for `(test)`/`(tests)` stanzas and inline-test
    libraries** (dune >= 3.20). Reusing such a name for a rule is a dependency cycle as soon as the
    rule also names `runtest`, and is confusing in any case, so a hand-written per-test alias names
    the GOLDEN rather than the stanza: `runtest-zero_out_local_decl-unoptimized` beside the
    `zero_out_local_decl` test, `runtest-test_ppx_op-ppx` beside `test_ppx_op`. Adding an action to a
    generated alias is legal, and the per-directory `runtest-env_spelling_gate` rule does exactly
    that so every per-test rule can depend on the ambient gate -- and is the one rule `env_var_deps`
    lets share such a name, recognized by its `(universe)` dependency rather than by name. `env_var_deps` compares the two
    literally: the alias must begin with the golden's name, since what a reader has in hand when
    they reach for the alias is the golden that just failed, and an alias that renames it is one
    they construct empty.
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
- `tools/sweep.sh` is the coverage for every backend CI does not run: cc, multidev_cc and metal
  locally, cuda on `rog-nv-wsl`, hip on `minix-amd-wsl`, all pinned to ONE resolved commit so a
  mid-sweep merge cannot leave the machines testing different trees. multidev_cc needs no hardware
  and is there anyway: it keeps its own `micrograd_demo_logging` debug-log golden, which
  `dune runtest` diffs only under `OCANNL_BACKEND=multidev_cc` — a spelling neither test/config's
  pinned `backend=cc` nor CI ever sets. gh-ocannl-700 is the cost of that gap: `eefa827e`
  (gh-ocannl-461) reordered backprop fragments, re-promoted the cc golden, and left multidev's
  stale and master red under that backend for six weeks. **A backend with its own goldens and no
  leg here is a silent regression channel, hardware or not** — add the leg when you add the
  goldens. It records a row per unit in `~/.ocannl-sweep/history.tsv` and never
  exits non-zero for test failures — its exit code is not a verdict, the history file is. A daily
  scheduled task drives it.
- `timeout(1)` is not a portable group-killing bound, and the failure is silent in both directions.
  macOS ships none at all, which is why the repo reaches for `perl -e 'alarm N; exec @ARGV'`; and
  where one exists it is not necessarily GNU's — uutils coreutils (Rust, Ubuntu's default since
  25.10, and what BOTH GPU sweep boxes run as 0.8.0) accepts `-k` and delivers the TERM phase to the
  process group, but escalates the KILL to the DIRECT CHILD only. A descendant that ignores or
  outlives TERM is reparented and keeps running while `timeout` cheerfully reports 137. Measured
  there: `timeout -k 2 1 sh -c 'trap "" TERM; sleep 987654'` returns 137 and leaves the `sleep`
  alive. For a remote unit that means the cap says "coverage lost" while the run still holds the
  GPU and the worktree lock the next sweep must take. Both sides of `tools/sweep.sh` therefore run
  the unit under the same perl supervisor (`capped` locally, `remote_capped` emitting it as far-side
  shell text): it forks, `setpgrp`s the child, and signals the GROUP on expiry, TERM then KILL, so
  the bound holds whatever `timeout` is on the far side's PATH. Exit 142 (128+SIGALRM) is that
  supervisor's expiry, on either side of the ssh (gh-ocannl-727).
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
- A golden line printed at a FIXED decimal precision is not made portable by lowering the
  precision: it only moves the boundary. `cifar_conv`'s epoch-30 mean loss sat at ~1.05, so its
  `%.1f` print — introduced to absorb reduction-order drift — read `1.0` on cc and `1.1` on cuda at
  the same commit, and no promotion could serve both backends. The fix that holds is the one the
  test metrics already used (`9defc92f`): exact digits to stderr, where dune does not diff them, and
  on stdout a `Verdict` claim about the property the trajectory was there to show (the loss fell
  from the first logged epoch to the last). The `@slow` goldens get this treatment the day they
  flip, because CI never runs `@slow` — only `tools/sweep.sh --slow` (the Sunday sweep) and hand runs do — so a knife-edge
  there stays hidden until a GPU run lands on the other side of it.
- gh-ocannl-725 swept that genre out of `test/training` and `test/gpt2` rather than waiting for the
  next flip, and the audit produced a rule worth reusing whenever a training golden gains a number.
  A float may sit in a stdout golden only when its value is EXACT by construction: a threshold
  constant (`moons_demo`'s convergence epsilon), a power-of-two loss scale (`mixed_prec_parity`), a
  host-side schedule evaluated in closed form (`loop_utils`' LR table and z-scores), or a small sum
  of exactly representable dyadic terms (`data_parallel`'s 67.5 / 11.25 batch losses). Anything a
  device reduction produced — a trained loss, an epoch mean, a probability, a stepped parameter —
  goes to stderr tagged `(not part of the golden)`, and stdout gets a `Verdict` claim about the
  property the number was showing. The claim must still DISCRIMINATE: a threshold the trained value
  clears and an untrained one does not, a fall between the first and last logged epoch, an argmax or
  a ranking (`mlp_bn_names`' top-3 next characters — their 0.07–0.11 probability gaps are hundreds
  of times the cross-build drift that the two printed decimals were one tie away from), or an
  absolute closed-form value within a tolerance. "Is finite" is not by itself such a claim; pair it
  with one that a wrong number fails. Nothing lints this — a `%.Nf` scan's false-positive rate on
  constants and dataset sizes did not justify another exemption list — so it is an audit rule for
  whoever adds the next training golden.
- Two details of that conversion are what the digits were quietly doing, and both are easy to drop
  (Codex found all seven instances of them in one round on PR #447). First: a pinned number is a
  TWO-SIDED constraint, and the claim that replaces it usually is not. Every loss bound in the
  makemore and conv tests was an upper bound, so a dropped negation or a backend sign error — a
  finite NEGATIVE cross-entropy — cleared all of them while the trajectory still fell; cross-entropy
  is `-log p >= 0` by construction, so each of those tests now claims validity over the whole
  trajectory beside the thresholds. The same applies to a claim about SHAPE: `mlp_bn_names`' top-3
  ranking and separation both hold for a head of 0.70 / 0.20 / 0.08, so coarse magnitude bands went
  in alongside them. Ask what range the removed digits excluded, and claim that range. Second: mark
  every relocated line `(not part of the golden)` — stderr and stdout interleave in a terminal, and
  without the tag a reader cannot tell an informational number from an asserted one.
