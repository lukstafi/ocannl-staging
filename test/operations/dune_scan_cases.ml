(** How {!Test_utils.Dune_stanza_scan} reads a dune file, exercised directly.

    The scan decides which stanzas are subject to the dependency rule, so every one of its mistakes
    is silent in the same way [Config_key_scan]'s are: a stanza it fails to recognise looks exactly
    like a stanza that is not there. The cases below are therefore written against the shapes the
    repository's dune files actually take — a dep buried under a named group, an executable reached
    through a variable, a stanza quoted in a comment — rather than against the one shape a checker
    is easiest to write for. *)

open Base
open Stdio
module Scan = Test_utils.Dune_stanza_scan

let render (site : Scan.site) =
  let where = Scan.in_subdir site.Scan.subdir site.Scan.cwd in
  Printf.sprintf "%s%s %s%s"
    (if String.is_empty where then "" else "in " ^ where ^ ": ")
    (Scan.kind_name site.Scan.kind) site.Scan.name
    (if site.Scan.declares_config then " [declares]" else "")

(* Each case pairs a dune-file body with the sites the scan should report, in order. *)
let cases =
  [
    ( "a test declaring the dep",
      {dune|(test (name t) (deps ocannl_config) (libraries base))|dune},
      [ "test t [declares]" ] );
    ("a test omitting the dep", {dune|(test (name t) (libraries base))|dune}, [ "test t" ]);
    ( "a test with deps but not this one",
      {dune|(test (name t) (deps (env_var OCANNL_BACKEND) t.expected))|dune},
      [ "test t" ] );
    (* The dep may sit anywhere in the field, including inside a group, and a directory reaching
       for a config elsewhere still declares one. *)
    ( "the dep is found wherever in deps it sits",
      {dune|(test (name t) (deps (glob_files *.data) (env_var OCANNL_BACKEND) ocannl_config))|dune},
      [ "test t [declares]" ] );
    ( "an explicit file dependency declares it",
      {dune|(test (name t) (deps (file ocannl_config)))|dune},
      [ "test t [declares]" ] );
    (* A config elsewhere is still a dependency on a file -- WHICH file it has to be is settled by
       the check, which knows where configs exist; see declared_paths_cases below. *)
    ( "a config elsewhere is a dependency too",
      {dune|(test (name t) (deps ../config/ocannl_config))|dune},
      [ "test t [declares]" ] );
    ( "so does one bound to a name",
      {dune|(test (name t) (deps (:cfg ocannl_config)))|dune},
      [ "test t [declares]" ] );
    (* Dune's dependency language names other things than files, and depending on an ALIAS or an
       environment variable of that name is not depending on the file (Codex P2, round 3). *)
    ( "an alias of the same name is not the file",
      {dune|(test (name t) (deps (alias ocannl_config)))|dune},
      [ "test t" ] );
    ( "nor is an environment variable of that name",
      {dune|(test (name t) (deps (env_var ocannl_config)))|dune},
      [ "test t" ] );
    (* A glob matches the source tree, and in a directory whose config arrives through
       `(copy_files …)` the file is a generated target -- so the glob matches nothing and builds
       nothing, while looking like a declaration (Codex P2, round 7). *)
    ( "a glob does not depend on the copied config",
      {dune|(test (name t) (deps (glob_files ocannl_config)))|dune},
      [ "test t" ] );
    ( "a multi-test stanza is one site naming both",
      {dune|(tests (names a b) (deps ocannl_config))|dune},
      [ "test a, b [declares]" ] );
    (* An inline-test library runs tests too, and its deps live one level in. *)
    ( "inline tests declare the dep inside their own field",
      {dune|(library (name l) (inline_tests (deps ocannl_config (env_var OCANNL_BACKEND))))|dune},
      [ "inline tests l [declares]" ] );
    ( "an inline-test field with no deps at all",
      {dune|(library (name l) (inline_tests))|dune},
      [ "inline tests l" ] );
    ( "a library's own deps are not the inline tests' deps",
      {dune|(library (name l) (deps ocannl_config) (inline_tests (deps (env_var X))))|dune},
      [ "inline tests l" ] );
    ("a plain library is not a site", {dune|(library (name l) (libraries base))|dune}, []);
    (* An executable is not a site: dune runs it only where a rule says so, which is what keeps
       diagnostic and tutorial executables off the list without an exemption. *)
    ("an executable is not a site", {dune|(executable (name bench) (libraries base))|dune}, []);
    ( "a rule running an executable is a site",
      {dune|(rule (target x.actual) (deps ocannl_config)
 (action (with-stdout-to %{target} (run %{dep:bench.exe}))))|dune},
      [ "rule running bench.exe [declares]" ] );
    ( "a rule running an executable through a named dep is one too",
      {dune|(rule (targets out.ml) (deps (:pp pp.exe) (:input in.ml))
 (action (run ./%{pp} --impl %{input} -o %{targets})))|dune},
      [ "rule running pp.exe" ] );
    (* A dependency form wrapping the path binds the executable just as surely; keeping only bare
       atoms lost it, after which the binding looked empty and the command external (Codex P2,
       round 6). *)
    ( "a named dep may wrap its path in a dependency form",
      {dune|(rule (deps ocannl_config (:runner (file probe.exe))) (action (run %{runner})))|dune},
      [ "rule running probe.exe [declares]" ] );
    ( "a named dep resolving to no executable is not evidence of an external tool",
      {dune|(rule (deps (:script run.sh)) (action (run %{script})))|dune},
      [ "rule whose command this scan cannot read: %{script}" ] );
    (* dune runs a chdir'd action elsewhere, and OCANNL searches upward from THERE, so that
       directory's config is the one that has to be built (Codex P2, round 6). *)
    ( "a chdir moves the directory whose config matters",
      {dune|(rule (deps ocannl_config) (action (chdir ../sibling (run %{dep:probe.exe}))))|dune},
      [ "in ../sibling: rule running probe.exe [declares]" ] );
    ( "and the dependency that satisfies it is the one reaching that directory",
      {dune|(rule (deps ../sibling/ocannl_config)
 (action (chdir ../sibling (run %{dep:probe.exe}))))|dune},
      [ "in ../sibling: rule running probe.exe [declares]" ] );
    (* A `(test)` may carry a custom action, and a chdir in one moves the test's own process
       (Codex P2, round 8). *)
    ( "a test's own action can move where it runs",
      {dune|(test (name t) (deps ocannl_config) (action (chdir ../sibling (run %{test}))))|dune},
      [ "in ../sibling: test t [declares]" ] );
    ( "a test with an ordinary action still runs where it is",
      {dune|(test (name t) (deps ocannl_config) (action (run %{test} --flag)))|dune},
      [ "test t [declares]" ] );
    (* The test's own command is what says where the TEST runs; a helper sent elsewhere in the same
       action is a site of its own, not a relocation of the test (Codex P2, round 10). *)
    ( "a helper chdir'd elsewhere does not move the test",
      {dune|(test (name t) (deps ocannl_config)
 (action (progn (chdir ../scratch (run diff a b)) (run %{test}))))|dune},
      [ "test t [declares]" ] );
    ( "but a helper running an executable elsewhere is its own site",
      {dune|(test (name t) (deps ocannl_config)
 (action (progn (chdir ../scratch (run %{dep:probe.exe})) (run %{test}))))|dune},
      [ "test t [declares]"; "in ../scratch: rule running probe.exe [declares]" ] );
    ( "a test's shell action leaves its directory unestablished too",
      {dune|(test (name t) (deps ocannl_config) (action (bash "cd ../sibling && ./%{test}")))|dune},
      [
        "test t [declares]";
        "rule whose working directory this scan cannot establish: shell: cd ../sibling && \
         ./%{test} [declares]";
      ] );
    (* OCANNL searches UPWARD, so an action that chdirs into a child finds the stanza's own config;
       demanding the child have one rejects a correctly configured rule (Codex P2, round 9). *)
    ( "a chdir into a child is answered by the stanza's own config",
      {dune|(rule (deps ocannl_config) (action (chdir child (run %{dep:probe.exe}))))|dune},
      [ "in child: rule running probe.exe [declares]" ] );
    ( "the child's own config answers it as well",
      {dune|(rule (deps child/ocannl_config) (action (chdir child (run %{dep:probe.exe}))))|dune},
      [ "in child: rule running probe.exe [declares]" ] );
    ( "a sibling chdir is reported at the sibling",
      {dune|(rule (deps ocannl_config) (action (chdir ../sibling (run %{dep:probe.exe}))))|dune},
      [ "in ../sibling: rule running probe.exe [declares]" ] );
    (* An explicit path names something this repository produced, whatever its extension; stripping
       the `./` before asking loses what distinguishes it from `python3` (Codex P2, round 8). *)
    ( "an explicit relative program is a site without an extension",
      {dune|(rule (deps ocannl_config) (action (run ./probe)))|dune},
      [ "rule running probe [declares]" ] );
    ( "a bare word is still a tool on PATH",
      {dune|(rule (action (run diff a b)))|dune},
      [] );
    (* An absolute path names something the system provides -- no more ours than a bare word is
       (Codex P2, round 18). *)
    ( "an absolute path is a system tool",
      {dune|(rule (action (run /usr/bin/python3 --version)))|dune},
      [] );
    (* A PATH tool handed something this repository builds may be launching it (`env probe.exe`)
       or reading it (`diff old.exe new.exe`), and dune's grammar does not say which -- so neither
       is guessed: the pair is reported, and the check settles it the way it settles every other
       unreadable command (Codex P2, rounds 12 and 13). *)
    ( "a launcher's target is reported without guessing that it is one",
      {dune|(rule (deps ocannl_config) (action (run env %{dep:probe.exe} --flag)))|dune},
      [
        "rule whose working directory this scan cannot establish: env, handed %{dep:probe.exe} \
         [declares]";
      ] );
    ( "and so is a tool that is only reading them",
      {dune|(rule (action (run diff old.exe new.exe)))|dune},
      [
        "rule whose working directory this scan cannot establish: diff, handed old.exe, new.exe";
      ] );
    (* Fail closed on an argument the scan cannot resolve either: `./%{runner}` names whatever the
       binding does, and dropping it would hide a launched executable (Codex P2, round 14). *)
    ( "an unresolvable argument counts as much as a resolved one",
      {dune|(rule (deps (:runner probe.exe)) (action (run env ./%{runner})))|dune},
      [ "rule whose working directory this scan cannot establish: env, handed ./%{runner}" ] );
    ( "ordinary arguments are not executables",
      {dune|(rule (action (run %{dep:probe.exe} --input data.csv --out %{targets})))|dune},
      [ "rule running probe.exe" ] );
    (* Where a chdir destination is built out of a pform, the directory cannot be resolved, and
       taking it literally would search a directory that does not exist (Codex P2, round 12). *)
    ( "a chdir to a pform leaves the directory unestablished",
      {dune|(rule (deps ocannl_config) (action (chdir %{workspace_root} (run %{dep:probe.exe}))))|dune},
      [
        "rule whose working directory this scan cannot establish: %{dep:probe.exe}, under `(chdir \
         %{workspace_root} ...)` [declares]";
      ] );
    (* But only for something that could read a configuration there: a PATH tool reads none
       wherever it runs, so an unresolvable destination over one is not a finding (round 13). *)
    ( "a chdir to a pform over a PATH tool is not",
      {dune|(rule (action (chdir %{workspace_root} (run diff a b))))|dune},
      [] );
    (* A rewritten PATH is the other way a command name stops saying what it names: there even a
       bare word may be a local executable, so the External verdict is the one that cannot be
       trusted under it (Codex P2, round 16). *)
    (* An action head the scan cannot place keeps the directory it sits in, since it may run an
       OCANNL program there (Codex P2, round 18). *)
    ( "an unclassified action under a chdir keeps that directory",
      {dune|(rule (deps child/ocannl_config) (action (chdir child (invent-an-action probe.exe))))|dune},
      [ "in child: rule with an action this scan cannot place: invent-an-action [declares]" ] );
    ( "and an unclassified action under a pform chdir is unplaceable, like a command",
      {dune|(rule (deps ocannl_config)
 (action (chdir %{workspace_root} (invent-an-action probe.exe))))|dune},
      [
        "rule whose working directory this scan cannot establish: invent-an-action, under \
         `(chdir %{workspace_root} ...)` [declares]";
      ] );
    ( "a rewritten PATH makes a bare command unplaceable",
      {dune|(rule (deps ocannl_config) (action (setenv PATH . (run env probe))))|dune},
      [
        "rule whose working directory this scan cannot establish: env, under `(setenv PATH . \
         ...)` [declares]";
      ] );
    ( "but a path-qualified command still names what it names",
      {dune|(rule (deps ocannl_config) (action (setenv PATH . (run %{dep:probe.exe}))))|dune},
      [ "rule running probe.exe [declares]" ] );
    (* And the directory a nested chdir chose is still where the process runs: PATH says nothing
       about that (Codex P2, round 17). *)
    ( "a chdir under a PATH rewrite keeps its directory",
      {dune|(rule (deps child/ocannl_config)
 (action (setenv PATH . (chdir child (run ./probe.exe)))))|dune},
      [ "in child: rule running probe.exe [declares]" ] );
    ( "a chdir to the stanza's own directory changes nothing",
      {dune|(rule (deps ocannl_config) (action (chdir . (run %{dep:probe.exe}))))|dune},
      [ "rule running probe.exe [declares]" ] );
    ( "a nested action does not hide the executable",
      {dune|(rule (alias slow) (deps ocannl_config)
 (action (no-infer (progn (with-stdout-to x.actual (run %{dep:mnist_conv.exe}))
  (diff x.expected x.actual)))))|dune},
      [ "rule running mnist_conv.exe [declares]" ] );
    ( "a rule running no executable is not a site",
      {dune|(rule (alias runtest) (action (diff a.expected a.actual)))|dune},
      [] );
    ( "a PATH tool with no workspace argument is not a site",
      {dune|(rule (alias runtest) (action (run python3 --version)))|dune},
      [] );
    (* But one handed something from the workspace is: `env probe.exe` launches it, `diff old.exe
       new.exe` reads it, and `env -C ../s probe.exe` launches it elsewhere -- three readings dune's
       grammar does not tell apart (Codex P2, rounds 12 to 14). The strongest is assumed, so the
       check asks for an exemption with a reason rather than guessing. *)
    ( "a PATH tool handed something from the workspace is unplaceable",
      {dune|(rule (alias runtest) (deps helper.py) (action (run python3 %{dep:test_it.py})))|dune},
      [
        "rule whose working directory this scan cannot establish: python3, handed \
         %{dep:test_it.py}";
      ] );
    (* An `(alias ...)` stanza took an action of its own before dune 2.0 and can still depend on an
       executable, so it is read like a rule rather than passed over (Codex P2, round 1). *)
    ( "an alias stanza is read like a rule",
      {dune|(alias (name runtest) (deps ocannl_config) (action (run %{dep:probe.exe})))|dune},
      [ "rule running probe.exe [declares]" ] );
    (* What the command names, in the spellings that do not say `.exe` (Codex P2, round 2). A
       `%{bin:…}` resolves a public executable of this workspace before it looks at PATH, so it
       counts; an external tool that reads no configuration is what the exemption list is for. *)
    ( "a public executable through %{bin:...}",
      {dune|(rule (deps ocannl_config) (action (run %{bin:probe} --read=backend)))|dune},
      [ "rule running probe [declares]" ] );
    ( "%{exe:...} names one too",
      {dune|(rule (action (run %{exe:probe.exe})))|dune},
      [ "rule running probe.exe" ] );
    (* This scan does not parse shell, and splitting a command line on whitespace only looked like
       reading it: `if ready; then ./probe.exe; fi` yields `./probe.exe;`, which passes for an
       external tool (Codex P2, round 5). A shell line is reported as unreadable, which the check
       settles by requiring the dependency. *)
    ( "a shell action is not parsed, it is reported",
      {dune|(rule (deps ocannl_config) (action (bash "./probe.exe --flag > out")))|dune},
      [
        "rule whose working directory this scan cannot establish: shell: ./probe.exe --flag > out \
         [declares]";
      ] );
    ( "including one where a word split would have lost the executable",
      {dune|(rule (action (system "if ready; then ./probe.exe; fi")))|dune},
      [
        "rule whose working directory this scan cannot establish: shell: if ready; then \
         ./probe.exe; fi";
      ] );
    (* Command position, not "an .exe somewhere in the stanza": a rule that only moves an
       executable around runs nothing. *)
    ( "a rule that copies an executable does not run it",
      {dune|(rule (target probe.copy) (action (copy %{dep:probe.exe} %{target})))|dune},
      [] );
    ( "a command this scan cannot place is reported, not ignored",
      {dune|(rule (deps ocannl_config) (action (run %{dep:helper.sh})))|dune},
      [ "rule whose command this scan cannot read: %{dep:helper.sh} [declares]" ] );
    (* dynamic-run executes a program too (Codex P2, round 3), and an action head on neither list
       is reported rather than passed over -- which is what makes a fourth such action harmless. *)
    ( "dynamic-run executes a program as much as run does",
      {dune|(rule (deps ocannl_config) (action (dynamic-run %{dep:probe.exe} --flag)))|dune},
      [ "rule running probe.exe [declares]" ] );
    ( "an action head on neither list is reported",
      {dune|(rule (deps ocannl_config) (action (invent-an-action probe.exe)))|dune},
      [ "rule with an action this scan cannot place: invent-an-action [declares]" ] );
    ( "a program action's arguments are not actions",
      {dune|(rule (action (run %{dep:probe.exe} --diff --copy)))|dune},
      [ "rule running probe.exe" ] );
    (* Identity is the path as written: a basename would let one directory's exemption cover a
       different executable of the same name (Codex P2, round 3). *)
    ( "an executable elsewhere keeps its path",
      {dune|(rule (action (run %{dep:../../tools/pp.exe} --impl x.ml)))|dune},
      [ "rule running ../../tools/pp.exe" ] );
    (* And the whole filename, punctuation included: splitting on "characters a path may contain"
       cut `helper+pp.exe` down to `pp.exe`, which is a different executable and a different
       exemption (Codex P2, round 4). Dune's own `%{` and `}` are the only boundaries. *)
    ( "a filename's punctuation is part of its identity",
      {dune|(rule (action (run %{dep:helper+pp.exe} --impl x.ml)))|dune},
      [ "rule running helper+pp.exe" ] );
    ( "a toolchain pform runs a compiler, not a test",
      {dune|(rule (action (run %{ocaml} script.ml)))|dune},
      [] );
    ( "a path built out of a pform is not guessed at",
      {dune|(rule (action (run %{dep:tools}/probe.exe)))|dune},
      [ "rule whose command this scan cannot read: %{dep:tools}/probe.exe" ] );
    (* What the file says, not what its prose says. *)
    ( "a stanza inside a comment is not a stanza",
      {dune|; (test (name phantom))
(test (name t) (deps ocannl_config))|dune},
      [ "test t [declares]" ] );
    ( "a config named in a comment does not declare it",
      {dune|; this one reads ocannl_config at startup
(test (name t))|dune},
      [ "test t" ] );
    ( "a config named in a string is not a dep",
      {dune|(test (name t) (action (run %{test} "ocannl_config")))|dune},
      [ "test t" ] );
    ( "an executable named in a comment does not make a rule a site",
      {dune|(rule (alias runtest)
 ; unlike bench.exe, this one only diffs
 (action (diff a.expected a.actual)))|dune},
      [] );
    (* A `(subdir …)` wrapper configures another directory from this dune file, and
       test/operations/dune runs two rules in test/operations/config that way. The stanzas inside
       are subject to the same rules, and the directory they name is the one whose config they
       need. *)
    ( "a subdir's stanzas are the same kinds of site",
      {dune|(subdir config
 (rule (deps ocannl_config) (action (run %{dep:reader.exe})))
 (test (name inner)))|dune},
      [ "in config: rule running reader.exe [declares]"; "in config: test inner" ] );
    ( "a nested subdir names the path it applies to",
      {dune|(subdir a (subdir b (test (name t) (deps ocannl_config))))|dune},
      [ "in a/b: test t [declares]" ] );
    ( "stanzas outside the subdir keep the dune file's own directory",
      {dune|(subdir config (test (name inner) (deps ocannl_config)))
(test (name outer) (deps ocannl_config))|dune},
      [ "in config: test inner [declares]"; "test outer [declares]" ] );
    ( "stanzas are reported in file order",
      {dune|(test (name a) (deps ocannl_config))
(executable (name e))
(rule (deps ocannl_config) (action (run %{dep:e.exe})))
(test (name b))|dune},
      [ "test a [declares]"; "rule running e.exe [declares]"; "test b" ] );
  ]

(* Stanzas that rewrite the executable search path for a whole directory, which the check refuses
   rather than modelling. Only the NAME position of an `env-vars` binding counts: setting some
   other variable to the literal value `PATH` rewrites nothing (Codex P2, round 18). *)
let path_rewriting_cases =
  [
    ("an env stanza setting PATH", {dune|(env (_ (env-vars (PATH .))))|dune}, [ "env" ]);
    ("one setting something else to PATH", {dune|(env (_ (env-vars (OTHER PATH))))|dune}, []);
    ("an env stanza that touches neither", {dune|(env (_ (flags (:standard))))|dune}, []);
  ]

(* WHICH config file a stanza depends on, as written. The scan reports the paths; which of them is
   the one the process will actually read is the check's decision, since only it knows where
   configs exist -- OCANNL walks up from the process directory and reads the first it finds, so an
   ancestor's answers only while no nearer directory has its own (Codex P2, rounds 9 to 11). *)
let declared_paths_cases =
  [
    ("the local one", {dune|(test (name t) (deps ocannl_config))|dune}, [ "ocannl_config" ]);
    ( "one in a sibling",
      {dune|(rule (deps ../sibling/ocannl_config) (action (run %{dep:probe.exe})))|dune},
      [ "../sibling/ocannl_config" ] );
    ( "one in a common parent, which a sibling chdir may legitimately need",
      {dune|(rule (deps ../ocannl_config) (action (chdir ../b (run %{dep:probe.exe}))))|dune},
      [ "../ocannl_config" ] );
    ( "several, reported in full",
      {dune|(test (name t) (deps ocannl_config child/ocannl_config))|dune},
      [ "child/ocannl_config"; "ocannl_config" ] );
    ( "a config the dependency language does not depend on",
      {dune|(test (name t) (deps (alias ocannl_config) (glob_files ocannl_config)))|dune},
      [] );
  ]

(* Which directories the file materializes a config into: "." stands for its own. *)
let copy_cases =
  [
    ("a copy_files stanza", {dune|(copy_files (files ../config/ocannl_config))|dune}, [ "." ]);
    ("the short spelling", {dune|(copy_files ../config/ocannl_config)|dune}, [ "." ]);
    ( "a copy_files inside a subdir names that directory",
      {dune|(subdir config (copy_files (files ../../config/ocannl_config)))|dune},
      [ "config" ] );
    ("copying something else", {dune|(copy_files (files ../data/*.csv))|dune}, []);
    (* A wildcard that could cover the config counts as materializing it: this decides where a
       config EXISTS, so guessing wide risks accepting a directory that has one anyway, while
       guessing narrow rejects a correctly configured one (Codex P2, round 12). *)
    ("a wildcard that covers it", {dune|(copy_files (files ../config/*))|dune}, [ "." ]);
    ("a wildcard that cannot", {dune|(copy_files (files ../config/*.expected))|dune}, []);
    ("dune's set syntax is taken as possibly matching",
     {dune|(copy_files (files ../config/{ocannl_config,other}))|dune}, [ "." ]);
    ("no copy_files at all", {dune|(test (name t) (deps ocannl_config))|dune}, []);
  ]

(* Which stanza kinds the scan has no classification for. One it cannot place might carry an
   action that runs a test executable, so it is reported rather than counted as nothing. *)
let unclassified_cases =
  [
    ("a classified stanza", {dune|(test (name t) (deps ocannl_config))|dune}, []);
    ("a stanza that runs nothing", {dune|(ocamllex lexer)|dune}, []);
    (* Dune runs cram tests; this repository has none, and the day one appears the scan says so. *)
    ("a cram stanza", {dune|(cram (applies_to my_test) (deps helper.exe))|dune}, [ "cram" ]);
    ( "an include, whose contents the scan never sees",
      {dune|(include stanzas.inc)|dune},
      [ "include" ] );
    ( "inside a subdir, too",
      {dune|(subdir gen (cram (applies_to x)))|dune},
      [ "cram" ] );
    ("something that is not a stanza at all", {dune|bare_atom|dune}, [ "<not a stanza>" ]);
  ]

(* A stanza dune reads whole and sexplib reads with a hole in it: the markers are atoms to the one
   and comments to the other, and counting only the file's top-level forms would have found both
   readers reporting one stanza (Codex P2, round 17). *)
let nested_marker_case =
  ( "a marker nested inside a stanza",
    {dune|(rule (deps ocannl_config)
 (action (progn (echo #|) (run %{dep:probe.exe}) (echo |#))))|dune} )

(* The two sequences sexplib reads as comments and dune does not. Reading such a file would drop
   whatever they enclose, so the scan refuses it -- but only when something was actually dropped,
   which a second count of the top-level forms is what establishes. Inside a string or after a
   `;`, sexplib does not treat them as comments either, and refusing there would take the whole
   suite down over an unrelated argument (Codex P2, round 12). *)
let refused_cases =
  [
    ("a block comment", {dune|#| (test (name hidden)) |#
(test (name t))|dune});
    ("a datum comment", {dune|#;(test (name hidden))
(test (name t))|dune});
  ]

let accepted_marker_cases =
  [
    ( "the marker inside a string is not a comment to either reader",
      {dune|(rule (deps ocannl_config) (action (echo "#| not a comment |#")))|dune},
      [] );
    ( "nor is one in a line comment",
      {dune|; dune would read #| as an atom; sexplib would not
(test (name t) (deps ocannl_config))|dune},
      [ "test t [declares]" ] );
  ]

let () =
  let ok = ref true in
  let check name expected found =
    if List.equal String.equal found expected then printf "ok: %s\n" name
    else (
      ok := false;
      printf "FAIL: %s -- expected [%s], found [%s]\n" name
        (String.concat ~sep:"; " expected)
        (String.concat ~sep:"; " found))
  in
  List.iter cases ~f:(fun (name, source, expected) ->
      let found =
        try List.map (Scan.sites source) ~f:render
        with exn ->
          ok := false;
          printf "FAIL: %s -- the scan raised: %s\n" name (Exn.to_string exn);
          []
      in
      check name expected found);
  List.iter path_rewriting_cases ~f:(fun (name, source, expected) ->
      check ("path-rewriting stanzas -- " ^ name) expected (Scan.path_rewriting_stanzas source));
  List.iter declared_paths_cases ~f:(fun (name, source, expected) ->
      let found =
        List.concat_map (Scan.sites source) ~f:(fun site -> site.Scan.declared_config_paths)
        |> List.dedup_and_sort ~compare:String.compare
      in
      check ("declared config paths -- " ^ name) expected found);
  List.iter copy_cases ~f:(fun (name, source, expected) ->
      let found =
        List.map (Scan.config_copy_dirs source) ~f:(fun dir ->
            if String.is_empty dir then "." else dir)
      in
      check ("copies the config -- " ^ name) expected found);
  List.iter unclassified_cases ~f:(fun (name, source, expected) ->
      check ("unclassified heads -- " ^ name) expected (Scan.unclassified_heads source));
  List.iter accepted_marker_cases ~f:(fun (name, source, expected) ->
      let found =
        try List.map (Scan.sites source) ~f:render
        with exn ->
          ok := false;
          printf "FAIL: accepted marker -- %s: the scan refused it: %s\n" name (Exn.to_string exn);
          []
      in
      check ("accepted marker -- " ^ name) expected found);
  List.iter (nested_marker_case :: refused_cases) ~f:(fun (name, source) ->
      match Scan.sites source with
      | exception _ -> printf "ok: refused -- %s\n" name
      | sites ->
          ok := false;
          printf "FAIL: refused -- %s: read the file as %d sites instead of refusing it\n" name
            (List.length sites));
  if not !ok then Stdlib.exit 1
