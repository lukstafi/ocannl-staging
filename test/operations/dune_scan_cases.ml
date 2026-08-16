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
  Printf.sprintf "%s%s %s%s"
    (if String.is_empty site.Scan.subdir then "" else "in " ^ site.Scan.subdir ^ ": ")
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
      {dune|(test (name t) (deps (glob_files *.data) ../config/ocannl_config))|dune},
      [ "test t [declares]" ] );
    ( "an explicit file dependency declares it",
      {dune|(test (name t) (deps (file ../config/ocannl_config)))|dune},
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
    ( "a nested action does not hide the executable",
      {dune|(rule (alias slow) (deps ocannl_config)
 (action (no-infer (progn (with-stdout-to x.actual (run %{dep:mnist_conv.exe}))
  (diff x.expected x.actual)))))|dune},
      [ "rule running mnist_conv.exe [declares]" ] );
    ( "a rule running no executable is not a site",
      {dune|(rule (alias runtest) (action (diff a.expected a.actual)))|dune},
      [] );
    ( "a rule running a non-OCaml tool is not a site",
      {dune|(rule (alias runtest) (deps helper.py) (action (run python3 %{dep:test_it.py})))|dune},
      [] );
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
    ( "a shell action's command line is read the same way",
      {dune|(rule (deps ocannl_config) (action (bash "./probe.exe --flag > out")))|dune},
      [ "rule running probe.exe [declares]" ] );
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
      {dune|(test (name t) (action (run %{exe} "ocannl_config")))|dune},
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

(* Which directories the file materializes a config into: "." stands for its own. *)
let copy_cases =
  [
    ("a copy_files stanza", {dune|(copy_files (files ../config/ocannl_config))|dune}, [ "." ]);
    ("the short spelling", {dune|(copy_files ../config/ocannl_config)|dune}, [ "." ]);
    ( "a copy_files inside a subdir names that directory",
      {dune|(subdir config (copy_files (files ../../config/ocannl_config)))|dune},
      [ "config" ] );
    ("copying something else", {dune|(copy_files (files ../data/*.csv))|dune}, []);
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

(* The two sequences sexplib reads as comments and dune does not. Reading such a file would drop
   whatever they enclose, so the scan refuses it instead of reporting a shorter file. *)
let refused_cases =
  [
    ("a block comment", {dune|#| (test (name hidden)) |#
(test (name t))|dune});
    ("a datum comment", {dune|#;(test (name hidden))
(test (name t))|dune});
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
  List.iter copy_cases ~f:(fun (name, source, expected) ->
      let found =
        List.map (Scan.config_copy_dirs source) ~f:(fun dir ->
            if String.is_empty dir then "." else dir)
      in
      check ("copies the config -- " ^ name) expected found);
  List.iter unclassified_cases ~f:(fun (name, source, expected) ->
      check ("unclassified heads -- " ^ name) expected (Scan.unclassified_heads source));
  List.iter refused_cases ~f:(fun (name, source) ->
      match Scan.sites source with
      | exception _ -> printf "ok: refused -- %s\n" name
      | sites ->
          ok := false;
          printf "FAIL: refused -- %s: read the file as %d sites instead of refusing it\n" name
            (List.length sites));
  if not !ok then Stdlib.exit 1
