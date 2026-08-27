(* gh-ocannl-628: the ambient variables a dune rule declares, against the ones a run reads.

   A dune rule is invalidated by an environment variable only if it says so. Where that matters is
   everywhere: 213 stanzas in this repository declare `(env_var OCANNL_BACKEND)` so that changing
   the backend re-runs the test rather than serving the previous backend's output as a pass.

   Two ways that goes wrong, both silent, both checked here.

   {1 A spelling nothing reads}

   `Utils.read_env_var` consults ONE name per key, `OCANNL_<KEY>` (gh-ocannl-652). Before that it
   consulted the lowercase `ocannl_<key>` FIRST -- so `ocannl_backend=cuda` outranked
   `OCANNL_BACKEND` and decided which backend every test compiled and ran on, while not one dune
   file in the repository declared it: a developer who exported the lowercase form got stale targets
   served as passes, from rules written precisely to prevent that. gh-ocannl-628 swept the second
   spelling INTO 213 stanzas; gh-ocannl-652 dropped the spelling instead, and made setting one fatal
   so that the demotion could not be silent. What is left to check here is that a declaration names
   the spelling that is read -- a stanza declaring `(env_var ocannl_backend)` tracks a variable no
   run will consult, and invalidates nothing.

   {2 Suites a rejected spelling never reaches}

   Nothing declares the REJECTED spellings, by design, so nothing reruns for one either: a cached
   `@test/einsum/runtest` would serve its previous passes with `ocannl_backend=cuda` ambient and the
   fatal startup check never reached (gh-ocannl-652, Codex P1 round 2). Each test directory carries
   an `env_spelling_gate` whose `(universe)` dependency makes dune rerun it every invocation; this
   check is what keeps the set whole, since dune aliases are per directory and the next test
   directory would otherwise be added without one.

   Per ALIAS, not per directory (Codex P1 round 3). A `(test)` stanza is a runtest action and
   nothing else, so a directory whose gate is a test stanza is ungated for `@slow` -- a separately
   documented entry point that `dune build @slow` reaches without building one `(test)`. Asking the
   question per directory let a runtest gate vouch for the slow rules beside it, which is the shape
   of the hole it was written to prevent. And per EVERY alias a build can start from, not a fixed
   list of two: each slow rule sits on its own `slow-<name>` alias, so that one slow test can be
   rerun after a change without the ~30 minutes the whole `@slow` suite takes, and
   `ocannl_backend=cuda dune build @test/training/slow-mlp_names` is the same entry point with the
   same hole. A gate "reaches" an alias when building the alias builds the gate: the gate's own
   alias, a rule whose `deps` name it (which runs the gate BEFORE the rule, so a rejected spelling
   fails before the slow run rather than beside it), or an `(alias (name …) (deps …))` stanza
   aggregating either. That last shape is what keeps `dune build @slow` the whole suite, and the
   check here is also what keeps the aggregation whole: a `slow-<name>` rule the `slow` alias does
   not list is one `@slow` skips, silently.

   {2 Gates the build does not see}

   The other direction: a variable that IS read and is declared nowhere. ppx_minidebug's per-module
   tracing gates (`OCANNL_LOG_LEVEL_ROW` and eighteen siblings) are read while PREPROCESSING, so a
   library that does not declare them hands back the modules it built with the trace statements
   stripped -- `OCANNL_LOG_LEVEL_ROW=9 dune build` returning a silent binary, which reads as "the
   trace shows nothing" rather than as "the trace was never compiled in". Each gate is checked
   against the library whose modules read it.

   {2 A stanza that declares neither spelling}

   The pairing check above sees a stanza that declares ONE spelling. What it cannot see is a stanza
   that declares NEITHER: a backend-sensitive test added with no `(env_var ...)` at all is invisible
   to a check phrased over the declarations that exist, and dune then serves the previous backend's
   output as a pass under `OCANNL_BACKEND=cuda dune build @…` -- the exact failure this file exists
   to prevent, arrived at from the other side (gh-ocannl-659, found on lukstafi/ocannl-staging#374
   where `simd_lane_choice` was added with no declaration; it is genuinely backend-free, and nothing
   in the build would have said otherwise had it been a `Context.auto` test).

   So the rule below is an exclusive or, asked of every stanza that runs an executable: either it
   declares `(env_var OCANNL_BACKEND)`, or it carries a marker comment inside its parentheses naming
   the backend it is pinned to -- or `none` -- and why. Both absent is the hole; both present is
   contradictory intent, since a stanza that names its backend has nothing to invalidate on.

   The conditional rule is kept, deliberately. Declaring the variable universally would be simpler
   to check and would claim a sensitivity most of these stanzas do not have: `test_einsum_parser`
   calls a parser, `test_metal_pool_bindings` pins Metal's emission, and the seven `Context.auto`
   rules in this directory pin `--ocannl_backend=cc` on the command line, which outranks the
   environment. A declaration on those would be noise, and noise is what the next reader learns to
   skip. What the marker costs instead is a sentence of classification per stanza, written where the
   next author will copy it from.

   {1 What decides "addressed to the configuration"}

   `Utils.classify_env_var`, the same function the startup check uses (gh-ocannl-629), so a name a
   rule tracks and a name a run reports cannot be classified two ways. It also supplies the reserved
   namespaces: `OCANNL_TOOL_...` is the tooling's, and `OCANNL_LOG_LEVEL_<MODULE>` is a gate,
   checked by the other half of this test. Both are uppercase-only, as configuration keys now are
   too. *)

open Base
open Stdio
module Scan = Test_utils.Dune_stanza_scan
module Sources = Test_utils.Config_key_scan

(* Declarations of a name OCANNL does not read as a configuration key. Keyed by "<dune
   file>:<name>", and each entry earns its place on every run (see the staleness check below): a
   rule tracking a variable no key would be read from is normally a typo, which is the whole point
   of asking. *)
let exempt_declarations =
  [
    ( "test/operations/dune:ocannl_backedn",
      "the fixture behind the `config_var_warnings` golden, which captures the warning a mistyped \
       key draws; the rule tracks the name so that an ambient one arriving does not leave the \
       golden stale" );
    ( "test/operations/dune:OCANNL_BACKEDN",
      "the same fixture, in the casing OCANNL reads -- a lowercase one is reported rather than \
       read, so both write to the stream the golden holds" );
    ( "test/operations/dune:ocannl-log_level",
      "the `config_var_fatal_spelling` fixture: a known key in the dashed spelling that \
       gh-ocannl-605 dropped, which since gh-ocannl-652 aborts the run rather than warning" );
  ]

(* Directories with runtest actions that carry no ambient gate, and why. Same shape as the
   declaration exemptions above: each is checked for still being needed. *)
let gateless_dirs =
  [
    ( "benchmarks/dune",
      "its one runtest action runs python3 over the benchmark orchestrator's own unit tests, which \
       import no OCANNL executable -- there is no startup check in reach to gate, the same reason \
       `config_dep_completeness` exempts it from the ocannl_config dependency" );
  ]

(* A gate is a stanza that depends on the state of the world: that is what makes dune rerun it
   rather than serve the previous run. Matched structurally rather than by the stanza's name, so a
   gate that is renamed or rewritten still counts. *)
let rec depends_on_universe = function
  | Sexp.List [ Sexp.Atom "universe" ] -> true
  | Sexp.List l -> List.exists l ~f:depends_on_universe
  | Sexp.Atom _ -> false

(* The lock a directory's actions share, where it has one. A gate added to such a directory has to
   take it too (Codex P1 round 4): not because the gate contends -- it links `arrayjit.utils` and
   starts no OpenMP pool -- but because the one unlocked action in a file of locked ones is what the
   next person copies when writing a real training test. Asked of the file rather than hard-coded
   per directory, so a directory that adopts the lock later brings its gate along. *)
let training_lock = "ocannl_training_test"

let rec takes_training_lock = function
  | Sexp.List (Sexp.Atom "locks" :: args) ->
      List.exists args ~f:(function Sexp.Atom a -> String.equal a training_lock | _ -> false)
  | Sexp.List l -> List.exists l ~f:takes_training_lock
  | Sexp.Atom _ -> false

(* The aliases a stanza's actions attach to: a rule's own `(alias …)` field -- not an `(alias …)`
   inside its `deps`, which is a dependency on another alias and is read by {!alias_deps} -- and
   `runtest` for a `(test)`/`(tests)` stanza, dune's shorthand for a runtest action that names no
   alias of its own. Asked per alias rather than per directory (Codex P1 round 3): a `(test)` stanza
   contributes to `runtest` alone, so a directory whose gate is a test stanza is ungated for `@slow`
   -- which is a separately documented entry point, and which `dune build @slow` reaches without
   building a single `(test)`. A gate on one alias must not vouch for another. *)
let aliases_of stanza =
  match stanza with
  | Sexp.List (Sexp.Atom ("test" | "tests") :: _) -> [ "runtest" ]
  | Sexp.List (Sexp.Atom "rule" :: fields) ->
      (* Both spellings: `(alias A)` and dune's plural `(aliases A B)`, which is how a rule sits on
         `runtest` and on its own per-test alias at once (gh-ocannl-726). Reading only the singular
         would take those rules for attaching to nothing, and a rule attached to nothing is a rule
         no gate has to reach. *)
      List.concat_map fields ~f:(function
        | Sexp.List [ Sexp.Atom "alias"; Sexp.Atom n ] -> [ n ]
        | Sexp.List (Sexp.Atom "aliases" :: args) ->
            List.filter_map args ~f:(function Sexp.Atom n -> Some n | _ -> None)
        | _ -> [])
  | _ -> []

(* The aliases a stanza's `deps` field names, which dune builds before it: before a rule's action
   runs, or whenever the alias an `(alias (name A) (deps …))` stanza defines is built. *)
let alias_deps stanza =
  match Scan.field stanza "deps" with
  | None -> []
  | Some args ->
      List.filter_map args ~f:(function
        | Sexp.List [ Sexp.Atom "alias"; Sexp.Atom n ] -> Some n
        | _ -> None)

(* The alias an `(alias (name A) …)` stanza defines: since dune 2.0 it carries no action of its own
   and only aggregates what its `deps` name. *)
let alias_stanza_name = function
  | Sexp.List (Sexp.Atom "alias" :: _) as stanza -> (
      match Scan.names_of stanza with [ n ] -> Some n | _ -> None)
  | _ -> None

(* A gate is a stanza with an action that depends on the state of the world. *)
let is_gate stanza = depends_on_universe stanza && not (List.is_empty (aliases_of stanza))

(* Every alias a build can start from: those rules and tests attach to, and those `(alias …)`
   stanzas define. *)
let entry_points stanzas =
  List.concat_map stanzas ~f:(fun s -> aliases_of s @ Option.to_list (alias_stanza_name s))
  |> List.dedup_and_sort ~compare:String.compare

(* The aliases `dune build @<alias>` runs a gate for: those a gate attaches to, closed under what
   building an alias builds -- the `deps` of every rule attached to it and of the `(alias …)` stanza
   defining it. A slow rule whose `deps` name the gate's alias is gated as much as one the gate sits
   beside, and runs it first. *)
let gated_aliases stanzas =
  let rec close gated =
    let next =
      List.fold stanzas ~init:gated ~f:(fun gated stanza ->
          if is_gate stanza || List.exists (alias_deps stanza) ~f:(Set.mem gated) then
            List.fold
              (aliases_of stanza @ Option.to_list (alias_stanza_name stanza))
              ~init:gated ~f:Set.add
          else gated)
    in
    if Set.equal next gated then gated else close next
  in
  close (Set.empty (module String))

(* The aliases building `@<alias>` reaches, through the same two routes -- what a suite
   aggregates. *)
let aliases_reached_from stanzas alias =
  let rec close reached =
    let next =
      List.fold stanzas ~init:reached ~f:(fun reached stanza ->
          let attached = aliases_of stanza @ Option.to_list (alias_stanza_name stanza) in
          if List.exists attached ~f:(Set.mem reached) then
            List.fold (alias_deps stanza) ~init:reached ~f:Set.add
          else reached)
    in
    if Set.equal next reached then reached else close next
  in
  close (Set.singleton (module String) alias)

(* The suite a per-test alias belongs to, by the naming convention: `<suite>-<name>` is one test and
   `<suite>` the suite, an `(alias (name <suite>) (deps …))` stanza listing its members -- so a
   member the list omits is one `dune build @<suite>` skips, silently. Two suites are built this
   way. `slow` is the one gh-ocannl-667 introduced. `runtest` is the same arrangement for the
   golden-diff rules (gh-ocannl-726): dune generates `runtest-<name>` for `(test)`/`(tests)` stanzas
   and inline-test libraries only, so an `(executable)` plus a `(rule)` that diffs has to be given
   one -- and it has to be given it ALONE, since a rule attached to two aliases makes building
   either build both, which would put the whole directory behind every per-test alias. Aggregating
   the members here is then the only thing that keeps them in `dune runtest`. *)
let suites = [ "slow"; "runtest" ]
let member_of suite alias = String.is_prefix alias ~prefix:(suite ^ "-")

(* The repo-wide scans and the family alias that runs them (gh-ocannl-703). Which rules are in the
   family is DERIVED rather than listed here: a rule that globs the repository recursively is
   reading the repository, which is what makes a scan repo-wide -- so a scan lands in the family the
   day it lands in the file, and this check cannot go stale against the stanza it is checking the
   way a second copy of the list would. *)
let scans_suite = "scans"

let rec globs_repository = function
  | Sexp.List (Sexp.Atom "glob_files_rec" :: _) -> true
  | Sexp.List l -> List.exists l ~f:globs_repository
  | Sexp.Atom _ -> false

let is_repo_wide_scan stanza =
  match stanza with
  | Sexp.List (Sexp.Atom "rule" :: _) ->
      Option.value_map (Scan.field stanza "deps") ~default:false ~f:(fun args ->
          List.exists args ~f:globs_repository)
  | _ -> false

(* ... and the glob has to LEAVE the directory to be reading the repository (Codex P2, round 6). A
   test that recursively reads a fixture tree of its own uses the same form and is nobody's
   repo-wide scan; classifying it as one would demand a `scans` family beside it and fail the check
   until an unrelated suite appeared. *)
let rec escapes_directory = function
  | Sexp.List (Sexp.Atom "glob_files_rec" :: args) ->
      List.exists args ~f:(function
        | Sexp.Atom pattern -> String.is_prefix pattern ~prefix:"../"
        | _ -> false)
  | Sexp.List l -> List.exists l ~f:escapes_directory
  | Sexp.Atom _ -> false

let is_repo_wide_scan stanza =
  is_repo_wide_scan stanza
  && Option.value_map (Scan.field stanza "deps") ~default:false ~f:(fun args ->
      List.exists args ~f:escapes_directory)

(* The focused aggregates beyond `scans` (gh-ocannl-783), and the same completeness question asked
   of them. A family is an `(alias (name <family>) (deps …))` stanza per directory, aggregating the
   per-test aliases of the tests that belong to it, so that `dune build @<family>` from the
   workspace root runs that class of test wherever it lives; a member the stanza omits is one the
   family skips, silently, which is exactly the failure gh-ocannl-703 closed for `scans`.

   Membership is DERIVED from what the member stanza itself declares -- never from a second copy of
   the list here, which could only confirm that the copy still says what it says. The derivation is
   a FLOOR: a family may list more (see `test_slab_free_on_grow` in `arrayjit/test/dune`), because
   what this check is for is the member that silently falls out, not the member someone chose to
   include. *)
type family = {
  family_alias : string;  (** the alias, spelled identically in every directory with members *)
  family_is : string;  (** what makes a stanza a member, for the diagnostic *)
}

let metal_family =
  {
    family_alias = "metal-codegen";
    family_is =
      "names `metal` as its backend, so no other box can judge it -- the executed Metal-only \
       guards and the emitted-MSL structural tests";
  }

(* The library modules that exist to make resource lifetimes observable: a test that names one is
   asking about allocation, free or context lifetime across a cleanup seam, which is what this
   family collects. Derived from the instrumentation rather than from the test's name or directory,
   for the reason the whole arrangement is derived -- a probe added tomorrow is asked about the day
   it lands. *)
let lifecycle_modules = [ "Resource_fault_injection"; "Alloc_census" ]

(* The one source that names those modules without probing anything: this scan, which has to spell
   them in order to look for them. Excluded by name rather than by contorting the spelling, and the
   exclusion is checked for staleness below -- an exemption the census does not exercise has stopped
   protecting anything, and every other exemption in this file is checked the same way. *)
let lifecycle_scanner = "test/operations/env_var_deps.ml"

let lifecycle_family =
  {
    family_alias = "lifecycle";
    family_is =
      "has a module that reads the resource-lifecycle instrumentation ("
      ^ String.concat ~sep:", " lifecycle_modules
      ^ "), so it answers for allocation, free or context lifetime across a cleanup seam";
  }

(* A lower bound on how many members the two derivations find across the repository, for the reason
   `artifact_caller_floor` has one: the relationship below holds vacuously over an empty family, and
   a derivation that silently stopped finding members would print the line a complete repository
   prints. A floor rather than a count, so that it does not move when a Metal test or a lifecycle
   probe is added. There were 7 on 2026-08-27. *)
let family_member_floor = 4

(* A family member is not a STANZA but one of the things a stanza builds: `(tests (names a b))` is
   two tests with an alias each, and `(executables (names a b))` two executables with their own
   runners. Reading a stanza as one member accepts a family that reaches half of it, and asks a
   family to reach the half that is no member at all (Codex P2, round 2). So the unit of this whole
   check is the named test/executable/inline-test library -- or, for a rule, the rule itself, which
   is what an `(executable)`'s backend marker sits on. *)
type member_unit = {
  unit_subdir : string;  (** the `(subdir …)` it sits under, relative to the dune file *)
  unit_stanza : Sexp.t;
  unit_name : string option;  (** the name dune builds it under, where it has one *)
  unit_identity : string;  (** how the diagnostics and the report name it *)
  unit_aliases : string list;
      (** the aliases that reach THIS unit; any one of them will do, since two rules running one
          executable are two ways of running it *)
}

(* Path arithmetic, so that a runner is credited to the executable it actually runs. `(chdir other
   (run ./probe.exe))` runs `other/probe.exe`, and a comparison that dropped either the cwd or the
   command's own directory would credit the LOCAL `probe` with it (Codex P2, rounds 1 and 2). *)
let normalize_path path =
  String.split path ~on:'/'
  |> List.fold ~init:[] ~f:(fun acc component ->
      match component with
      | "" | "." -> acc
      | ".." -> (
          match acc with above :: rest when not (String.equal above "..") -> rest | _ -> ".." :: acc)
      | component -> component :: acc)
  |> List.rev |> String.concat ~sep:"/"

(* The aliases of the rules in this directory that run `<name>.exe` as declared in `subdir`. An
   `(executable)` has no alias of its own, so its runner's is the one a family lists -- the same
   placement the `ocannl_config` dep and the backend marker take. A path this comparison declines
   leaves the unit unaggregated, which is reported: the direction that asks the author to say what
   they meant rather than passing on a coincidence of names. *)
let runner_aliases stanzas ~subdir ~name =
  let wanted = normalize_path (Scan.in_subdir subdir (name ^ ".exe")) in
  List.concat_map stanzas ~f:(fun stanza ->
      if
        List.exists (Scan.executables_run stanza) ~f:(fun (cwd, command) ->
            match command with
            | Scan.Runs path ->
                String.equal (normalize_path (Scan.in_subdir subdir (Scan.in_subdir cwd path))) wanted
            | _ -> false)
      then aliases_of stanza
      else [])

(* The units a stanza contributes, with the aliases that reach each. Deliberately never `runtest`: a
   family listing the directory's whole `runtest` would run the directory, which is the run these
   aggregates exist to avoid. *)
let family_units stanzas ~subdir stanza =
  let head = Option.value (Scan.head stanza) ~default:"<not a stanza>" in
  let unit ?name aliases =
    {
      unit_subdir = subdir;
      unit_stanza = stanza;
      unit_name = name;
      unit_identity = Printf.sprintf "%s %s" head (Option.value name ~default:"<unnamed>");
      unit_aliases = aliases;
    }
  in
  let generated name = unit ~name [ "runtest-" ^ name ] in
  match Scan.head stanza with
  (* dune >= 3.20 generates `runtest-<name>` per `(test)`/`(tests)` name AND per inline-test
     library -- the namespace `generated_runtest_names` already knows. A Metal-marked inline-test
     library reaches its family through exactly that alias (Codex P2, round 2). *)
  | Some ("test" | "tests") -> List.map (Scan.names_of stanza) ~f:generated
  | Some "library" when Option.is_some (Scan.field stanza "inline_tests") ->
      List.map (Scan.names_of stanza) ~f:generated
  | Some ("executable" | "executables") ->
      List.map (Scan.names_of stanza) ~f:(fun name ->
          unit ~name (runner_aliases stanzas ~subdir ~name))
  | Some _ -> [ unit (aliases_of stanza) ]
  | None -> []

(* The stanza kinds whose units can be asked whether their own modules read the lifecycle
   instrumentation. A plain `(library)` whose modules read it is the instrumentation's own
   implementation or a shared helper, not a probe anything runs; an inline-test library is a probe,
   and is included for that reason. *)
let has_family_modules stanza =
  match Scan.head stanza with
  | Some ("test" | "tests" | "executable" | "executables") -> true
  | Some "library" -> Option.is_some (Scan.field stanza "inline_tests")
  | _ -> false

(* Dune's named dependencies: `(deps (:golden foo.expected))` binds `%{golden}` to that path. A
   pform naming one carries no colon, so without the binding `golden_stem` would take the BINDING's
   name for the golden's -- rejecting the alias a reader would write and accepting one that names
   nothing (Codex P2, round 6). *)
let named_deps stanza =
  match Scan.field stanza "deps" with
  | None -> []
  | Some args ->
      List.filter_map args ~f:(function
        | Sexp.List (Sexp.Atom name :: Sexp.Atom path :: _) when String.is_prefix name ~prefix:":"
          ->
            Some (String.drop_prefix name 1, path)
        | _ -> None)

(* A file NAMED by a pform -- `%{dep:verdict_ratchet.expected}`, dune's ordinary way of writing a
   dependency inline, or `%{golden}` for a named one -- still names a file. Unwrap the leading
   pform: what follows the last `:` inside the braces is the path, and a pform with no colon is a
   named dependency, resolvable only from the stanza that bound it (Codex P2, rounds 4 and 6).
   Shared by the alias check and by the scans family's target matching (round 7), so the two agree
   about what a name is. *)
let unwrap_pform ?(named = []) file =
  match String.chop_prefix file ~prefix:"%{" with
  | None -> file
  | Some rest -> (
      match String.substr_index rest ~pattern:"}" with
      | None -> rest
      | Some close ->
          let inside = String.prefix rest close in
          let after = String.drop_prefix rest (close + 1) in
          let path =
            match String.rindex inside ':' with
            | Some colon -> String.drop_prefix inside (colon + 1)
            (* An unbound named dependency resolves to nothing, which the callers refuse by name
               rather than guessing. *)
            | None -> Option.value (List.Assoc.find named inside ~equal:String.equal) ~default:""
          in
          path ^ after)

(* What a rule writes, so that the rule DIFFING it can be found: a scan produces `<name>.actual` and
   a second rule holds it against the golden. It is that second rule the family alias has to
   aggregate, since it is the one that fails when the scan reports something. *)
let targets_of stanza =
  List.concat_map [ "target"; "targets" ] ~f:(fun field ->
      match Scan.field stanza field with
      | None -> []
      | Some args -> List.filter_map args ~f:(function Sexp.Atom a -> Some a | _ -> None))

let rec diffs_file ?(named = []) target = function
  | Sexp.List (Sexp.Atom ("diff" | "diff?") :: args) ->
      (* Modulo the pform spelling: `(diff foo.expected %{dep:foo.actual})` is dune's ordinary way
         of naming the dependency, and comparing it literally against the producer's `foo.actual`
         would report a correctly aggregated scan as missing (Codex P2, round 7). *)
      List.exists args ~f:(function
        | Sexp.Atom a -> String.equal (unwrap_pform ~named a) target
        | _ -> false)
  | Sexp.List l -> List.exists l ~f:(diffs_file ~named target)
  | Sexp.Atom _ -> false

(* A rule whose action holds a golden against a run's output: the shape that has no alias of its own
   unless someone writes one, since dune generates `runtest-<name>` for `(test)`/`(tests)` stanzas
   and for inline-test libraries, and for nothing else (gh-ocannl-726). *)
let rec is_diff_action = function
  | Sexp.List (Sexp.Atom ("diff" | "diff?") :: _) -> true
  | Sexp.List l -> List.exists l ~f:is_diff_action
  | Sexp.Atom _ -> false

(* The golden a diff rule holds a run's output against: the first operand of its `diff`. What the
   alias is checked against below -- an alias whose name has nothing to do with its golden is an
   entry point nobody constructs, since what a reader has in hand is the golden that just failed
   (Codex P2, round 3, after `runtest-n3_fwd_with_prec` had shortened away the `-unoptimized` its
   golden carries). *)
let rec goldens_in = function
  | Sexp.List (Sexp.Atom ("diff" | "diff?") :: Sexp.Atom golden :: _) -> [ golden ]
  | Sexp.List l -> List.concat_map l ~f:goldens_in
  | Sexp.Atom _ -> []

(* The part of a golden's name a reader would type: everything before the first dune pform or
   extension, less a trailing `_expected` and any separator left dangling. So
   `verdict_ratchet.expected` is `verdict_ratchet`, `n3_fwd_with_prec-unoptimized.ll.expected` is
   `n3_fwd_with_prec-unoptimized`, `top_down_prec.%{read:…}.expected` is `top_down_prec`,
   `micrograd_demo_logging-%{read:…}-0-0.log.expected` is `micrograd_demo_logging`, and the ppx
   convention's `test_ppx_op_expected.ml` is `test_ppx_op`. The alias may go on to say WHICH golden
   of a subject it checks -- `-extension`, `-unoptimized`, `-ppx` -- which is why the relation asked
   for is a prefix rather than equality: one run can write several goldens, and each needs an alias
   of its own. *)
let golden_stem ?(named = []) golden =
  let cut_at s ~on =
    match String.substr_index s ~pattern:on with None -> s | Some i -> String.prefix s i
  in
  let golden = unwrap_pform ~named golden in
  (* A pform INSIDE the name goes the other way, and is cut before the basename is taken: a
     `%{read:config/…}` carries a path, so taking the basename first would leave the CONFIG file's
     name as the stem. *)
  let stem = cut_at (Stdlib.Filename.basename (cut_at golden ~on:"%{")) ~on:"." in
  let stem = Option.value (String.chop_suffix stem ~suffix:"_expected") ~default:stem in
  String.rstrip stem ~drop:(fun c -> Char.equal c '-' || Char.equal c '_')

let is_golden_diff stanza =
  match stanza with
  | Sexp.List (Sexp.Atom "rule" :: _) ->
      Option.value_map (Scan.field stanza "action") ~default:false ~f:(fun args ->
          List.exists args ~f:is_diff_action)
  | _ -> false

(* The names dune generates a `runtest-<name>` alias for by itself (>= 3.20): every `(test)`/
   `(tests)` stanza name, and every inline-test library's name. A hand-written alias must not reuse
   one (Codex P2, round 5): the two aliases MERGE, so the targeted entry point runs that test as
   well as the rule, which is the isolation this whole arrangement is for -- and once the rule also
   names `runtest`, dune calls the pair a dependency cycle. The `runtest-env_spelling_gate` rule is
   the deliberate exception, and is recognized structurally rather than by name: it is the ambient
   gate, whose whole purpose is to run the same binary the `(test)` stanza does. *)
let is_test_stanza = function Sexp.List (Sexp.Atom ("test" | "tests") :: _) -> true | _ -> false

(* The generated names that belong to an ambient gate: a `(test)` stanza depending on `(universe)`
   is the gate, so a rule sharing ITS alias runs the same gate binary, which is the one deliberate
   collision. Recognized this way rather than by the literal name `env_spelling_gate` (which a
   rename would silently unexempt) and rather than by the rule alone (which let any
   universe-dependent rule claim the exemption -- Codex P2, rounds 6 and 7). *)
let gate_generated_names stanzas =
  List.concat_map stanzas ~f:(fun stanza ->
      if is_test_stanza stanza && depends_on_universe stanza then Scan.names_of stanza else [])
  |> Set.of_list (module String)

let generated_runtest_names stanzas =
  List.concat_map stanzas ~f:(fun stanza ->
      match stanza with
      | Sexp.List (Sexp.Atom ("test" | "tests") :: _) -> Scan.names_of stanza
      | Sexp.List (Sexp.Atom "library" :: _) when Option.is_some (Scan.field stanza "inline_tests")
        ->
          Scan.names_of stanza
      | _ -> [])
  |> Set.of_list (module String)

(* The prefix `Utils.classify_env_var` reports for a per-module tracing gate. *)
let gate_prefix = "ocannl_log_level_"

(* A lower bound on how many sources call `Test_utils.Generated.init` (gh-ocannl-723). The rule
   below is a relationship between two answers, and if the source-side answer silently became "none"
   -- a ppxlib upgrade, a rename of the module, a glob that stopped reaching the test sources -- the
   relationship would hold vacuously over an empty set and the check would go green having stopped
   checking. A floor rather than a count, for the reason gh-ocannl-665 took the counts out of the
   sibling goldens: it must not move when a test is added or removed. There were 37 on
   2026-08-23. *)
let artifact_caller_floor = 20

(* The configuration key `OCANNL_BUILD_FILES_PREFIX` addresses, which is what a module reading it by
   name reads. *)
let artifact_config_key = "build_files_prefix"

(* The stanza kinds that name their own modules, and so can be asked what those modules read. *)
let module_stanzas = [ "library"; "test"; "tests"; "executable"; "executables" ]

let main () =
  if Array.length Stdlib.Sys.argv < 2 then (
    eprintf "Usage: %s <workspace_root> <dune file or .ml source...>\n" Stdlib.Sys.argv.(0);
    Stdlib.exit 1);
  let base = Scan.base_dir Stdlib.Sys.argv.(1) in
  let paths =
    Array.to_list (Array.subo Stdlib.Sys.argv ~pos:2)
    |> List.map ~f:(fun path -> (Scan.repo_relative base path, path))
    |> List.dedup_and_sort ~compare:(fun (a, _) (b, _) -> String.compare a b)
  in
  let dune_files =
    List.filter paths ~f:(fun (path, _) -> String.equal (Stdlib.Filename.basename path) "dune")
  in
  (* Through `Sources.sources_among`, the same filter the sibling scans apply: dune's globs run over
     the BUILD tree, where a preprocessed `<name>.pp.ml` sits beside every ppx-using `<name>.ml`.
     Reading both would double the census, and a `.pp.ml` is not OCaml the compiler's own parser
     accepts -- it carries the ppx's output verbatim -- so the pair has to be resolved to the
     source, not merely deduplicated. *)
  let source_files =
    let all = List.filter paths ~f:(fun (path, _) -> String.is_suffix path ~suffix:".ml") in
    let kept = Set.of_list (module String) (Sources.sources_among (List.map all ~f:fst)) in
    List.filter all ~f:(fun (path, _) -> Set.mem kept path)
  in
  let sources =
    List.map source_files ~f:(fun (path, on_disk) -> (String.lowercase path, on_disk))
  in
  if List.is_empty dune_files || List.is_empty sources then (
    Verdict.fail "no dune files or no sources among the arguments -- the rule's globs match nothing";
    Stdlib.exit 1);
  (* Directories whose sources this scan was handed. A library elsewhere cannot have its gates
     checked, and says so rather than passing for lack of evidence. *)
  let scanned_dirs =
    List.map sources ~f:(fun (path, _) -> Stdlib.Filename.dirname path)
    |> Set.of_list (module String)
  in
  let source_of ~dir module_name =
    List.Assoc.find sources ~equal:String.equal
      (String.lowercase (Scan.in_subdir dir (module_name ^ ".ml")))
  in
  (* gh-ocannl-783: every source that NAMES one of the resource-lifecycle instrumentation modules,
     keyed the way `source_of` looks a module up. A substring search over the repository and
     nothing more: the module has to be named for any use of it to reach the source, and what makes
     a test a lifecycle probe is that it asks the instrumentation something. *)
  let lifecycle_named =
    List.filter_map source_files ~f:(fun (path, on_disk) ->
        let content = In_channel.read_all on_disk in
        if List.exists lifecycle_modules ~f:(fun m -> String.is_substring content ~substring:m) then
          Some (String.lowercase path)
        else None)
  in
  let lifecycle_sources =
    List.filter lifecycle_named ~f:(fun path ->
        not (String.equal path (String.lowercase lifecycle_scanner)))
    |> Set.of_list (module String)
  in
  (* gh-ocannl-723: every source that calls `Test_utils.Generated.init`, keyed the way `source_of`
     looks a module up, so that a stanza's `(modules …)` field answers for its own sources. Narrowed
     textually before parsing -- the module has to be NAMED for any spelling of the call to reach it
     -- so the census costs a substring search over the repository and a parse of the few dozen
     files that could contain one. *)
  let artifact_callers =
    List.filter_map source_files ~f:(fun (path, on_disk) ->
        let content = In_channel.read_all on_disk in
        if not (Sources.could_call_generated_init content) then None
        else
          match Sources.generated_init_calls_in_source content with
          | [] -> None
          | _ :: _ -> Some path
          | exception exn ->
              (* A source this scan cannot read is one it cannot answer for, and answering "no
                 calls" for it would be the silent failure the whole check is against. *)
              Verdict.fail
                (Printf.sprintf
                   "%s names `Test_utils.Generated` and does not parse, so whether it calls the \
                    initializer cannot be established: %s"
                   path (Exn.to_string exn));
              None)
  in
  let artifact_caller_keys =
    Set.of_list (module String) (List.map artifact_callers ~f:String.lowercase)
  in
  (* Whether this run was handed the repository, established the way the sibling scans establish it:
     every scan root the globs are written for contributed its floor of sources. The relationship
     below is about whatever tree is in front of the scan and is checked either way; the CENSUS
     floor is a statement about the repository, so it is asked only of a run that has it. Which mode
     a run was in goes into the golden, so a glob that breaks flips that line rather than quietly
     retiring the floor. *)
  let repository_census = List.is_empty (Sources.floor_violations (List.map source_files ~f:fst)) in
  (* The lifecycle derivation's one exclusion, checked against the census that would otherwise trip
     over it: over the repository this scan's own source names the instrumentation modules, and if
     it stopped doing so the exclusion would be silently protecting nothing. Asked only of a run
     that has the repository, since the control's trees do not contain this file. *)
  if repository_census && not (List.mem lifecycle_named (String.lowercase lifecycle_scanner) ~equal:String.equal)
  then
    Verdict.fail
      (Printf.sprintf
         "%s no longer names %s, so excluding it from the resource-lifecycle derivation protects \
          nothing -- drop the exclusion"
         lifecycle_scanner
         (String.concat ~sep:" or " lifecycle_modules));
  let artifact_claimed = ref (Set.empty (module String)) in
  let artifact_violations = ref 0 in
  (* One line per subject for stderr, and per dune file for the golden: what the golden holds is
     that a file still declares the variable where its modules call the initializer, not how many
     stanzas do -- the gh-ocannl-665 argument, since a count moves whenever a test is added. *)
  let artifact_table = ref [] in
  let artifact_by_file = ref [] in
  let fail = Verdict.fail in
  let exemptions = Map.of_alist_exn (module String) exempt_declarations in
  let exemptions_used = ref (Set.empty (module String)) in
  let gateless = Map.of_alist_exn (module String) gateless_dirs in
  let gateless_used = ref (Set.empty (module String)) in
  let gated = ref [] in
  (* The gh-ocannl-659 half: one line per stanza that runs an executable, and the per-file summary
     the golden holds. *)
  let classification = ref [] in
  let by_file = ref [] in
  (* gh-ocannl-783: one entry per derived family member -- the file, the family, the stanza, and
     whether that family's alias reaches it. The claim below is quantified over this population, so
     a derivation that stopped finding members reports an empty family rather than passing. *)
  let family_table = ref [] in
  let placed_subjects = ref 0 in
  let subject_floor = ref 0 in
  (* The two readers' POPULATIONS, each stanza by the file and line it opens at, so that the
     relationship checked below is "the same stanzas" and not "as many stanzas". Two totals compared
     as numbers say nothing about WHICH stanza either reader is alone on, and a gap of one absorbs a
     different stanza dropping out of enforcement (Codex P2, round 2), which is why gh-ocannl-690
     itemised the gap on stderr rather than leaving it as arithmetic. Identities rather than counts
     is also what keeps this off the churn treadmill gh-ocannl-701 took the scans off: a test added
     anywhere moves both lists together. *)
  let walk_places = ref [] in
  let floor_names = ref [] in
  (* Kept apart on purpose: a stanza declaring neither is the hole gh-ocannl-659 is about, while a
     marker the scan could not place is the scan going blind to it -- and a claim that conflated the
     two would pass while the second was true. *)
  let xor_violations = ref 0 in
  let marker_holes = ref 0 in
  let tracked_keys = ref (Set.empty (module String)) in
  let gate_table = ref [] in
  let read_table = ref [] in
  (* Every `(env_var ...)` under a sexp, at any depth. *)
  let rec env_vars_in = function
    | Sexp.List [ Sexp.Atom "env_var"; Sexp.Atom name ] -> [ name ]
    | Sexp.List l -> List.concat_map l ~f:env_vars_in
    | Sexp.Atom _ -> []
  in
  (* The dependency fields, at any depth -- `(deps ...)` of a rule inside a `(subdir ...)` as much
     as of a top-level test, and `(preprocessor_deps ...)` of a library. Not recursing INTO one
     keeps the count below a partition of the file's declarations rather than a double count. *)
  let rec dep_fields = function
    | Sexp.List (Sexp.Atom (("deps" | "preprocessor_deps") as field) :: args) -> [ (field, args) ]
    | Sexp.List l -> List.concat_map l ~f:dep_fields
    | Sexp.Atom _ -> []
  in
  List.iter dune_files ~f:(fun (dune_file, on_disk) ->
      let dir = match Stdlib.Filename.dirname dune_file with "." -> "" | dir -> dir in
      let content = In_channel.read_all on_disk in
      let stanzas = Scan.stanzas content in
      (* gh-ocannl-659: the exclusive or, over every stanza that runs an executable. *)
      let marked = Scan.marked_stanzas content in
      let attributed = ref [] in
      let words = ref (Set.empty (module String)) in
      let any_declared = ref false in
      List.iter marked ~f:(fun stanza ->
          (* A `(rule ...)` has no `(name ...)`, so it is named by what it runs -- which is what the
             reader has to go and look at anyway. *)
          let what =
            if not (String.is_empty stanza.Scan.marked_name) then stanza.Scan.marked_name
            else
              match
                List.map stanza.Scan.marked_sites ~f:(fun s -> s.Scan.name)
                |> List.dedup_and_sort ~compare:String.compare
              with
              | [] -> "<unnamed>"
              | names -> "running " ^ String.concat ~sep:", " names
          in
          let where =
            Printf.sprintf "%s:%d, the %s %s" dune_file stanza.Scan.marked_line
              stanza.Scan.marked_head what
          in
          (* The rule itself is [Scan.backend_rule_of]: this check owns the wording of the
             diagnostics and the tallies, and the DECISION lives with the scan so that it can be put
             to a stanza the repository does not contain (gh-ocannl-690). *)
          List.iter (Scan.marker_lines stanza) ~f:(fun line -> attributed := line :: !attributed);
          List.iter (Scan.malformed_markers stanza) ~f:(fun (line, text, why) ->
              Int.incr xor_violations;
              fail
                (Printf.sprintf
                   "%s:%d has a `%s` comment that does not parse as a marker: %s. The line reads \
                    `;%s`"
                   dune_file line Scan.marker_sentinel why text));
          match Scan.backend_rule_of stanza with
          (* A marker on a stanza that runs nothing declares nothing. An `(executable)` has no
             `deps` field at all, which is why its companion rule is where both the `ocannl_config`
             dep and this marker go -- putting it on the executable reads as a declaration and is
             not one. *)
          | Scan.Marker_without_run line ->
              Int.incr xor_violations;
              fail
                (Printf.sprintf
                   "%s carries a backend marker at line %d and runs no executable -- the marker \
                    belongs on the stanza that RUNS it, which for an `(executable)` is its \
                    companion rule, the same placement as the `%s` dep"
                   where line Scan.config_file)
          | Scan.Runs_nothing -> ()
          | Scan.Names_twice line ->
              Int.incr xor_violations;
              fail
                (Printf.sprintf
                   "%s carries more than one backend marker (the second at line %d) -- one stanza \
                    runs on one backend, so say so once"
                   where line)
          | Scan.Declares_and_names (line, m) ->
              Int.incr xor_violations;
              fail
                (Printf.sprintf
                   "%s both declares `(env_var %s)` and carries a marker at line %d saying `%s` -- \
                    those are contradictory: a stanza that names its backend has nothing for the \
                    variable to invalidate, and a stanza that selects one has no business claiming \
                    otherwise. Keep whichever is true"
                   where Scan.backend_env_var line m.Scan.backend)
          | Scan.Declares_variable ->
              Int.incr placed_subjects;
              any_declared := true;
              classification :=
                Printf.sprintf "  %-58s declares %s" where Scan.backend_env_var :: !classification
          | Scan.Names_backend (_, m) ->
              Int.incr placed_subjects;
              List.iter (String.split m.Scan.backend ~on:',') ~f:(fun word ->
                  words := Set.add !words word);
              classification :=
                Printf.sprintf "  %-58s %s -- %s" where m.Scan.backend m.Scan.reason
                :: !classification
          | Scan.Names_neither ->
              Int.incr placed_subjects;
              Int.incr xor_violations;
              fail
                (Printf.sprintf
                   "%s runs an executable and declares neither `(env_var %s)` nor a backend marker \
                    -- so `%s=cuda dune build @…` would serve this stanza's previous result as a \
                    pass. Add the declaration if the run SELECTS a backend, or the marker `; %s \
                    <%s> -- <reason>` if it names one or links none"
                   where Scan.backend_env_var Scan.backend_env_var Scan.marker_sentinel
                   (String.concat ~sep:"|" Scan.marker_backends)));
      (* Every marker in the file, against the ones a stanza claimed. A marker the walk attributed
         to nothing is one whose author believed they had declared something. *)
      let attributed = Set.of_list (module Int) !attributed in
      List.iter (Scan.marker_comments content) ~f:(fun (line, text) ->
          if not (Set.mem attributed line) then (
            Int.incr marker_holes;
            fail
              (Printf.sprintf
                 "%s:%d has a backend marker that sits inside no stanza -- a comment between \
                  stanzas declares nothing; move it inside the parentheses of the stanza it is \
                  about. The line reads `;%s`"
                 dune_file line text)));
      (* And every marker in the file against every occurrence of the sentinel ANYWHERE in it: the
         difference between the two is a marker the comment lexer did not place -- written into a
         quoted argument, into a stanza field, or into a comment shape this scan reads differently
         than dune does. *)
      let in_comments =
        List.sum
          (module Int)
          (Scan.marker_comments content)
          ~f:(fun (_, text) ->
            let rec count from found =
              match String.substr_index text ~pos:from ~pattern:Scan.marker_sentinel with
              | None -> found
              | Some at -> count (at + 1) (found + 1)
            in
            count 0 0)
      in
      let in_text = Scan.sentinel_occurrences content in
      if in_text <> in_comments then (
        Int.incr marker_holes;
        fail
          (Printf.sprintf
             "%s spells `%s` %d times and only %d of them are in a comment this scan places -- a \
              marker outside a comment declares nothing, and one in a comment this scan cannot see \
              is one it will not read"
             dune_file Scan.marker_sentinel in_text in_comments));
      (* The floor under the walk, read by the second reader that shares none of its classification
         machinery: a stanza the walk stops seeing is a stanza the rule above stops applying to,
         which looks exactly like a file with nothing to check (the gh-ocannl-665 argument, and
         config_dep_completeness' floors).

         Checked STANZA BY STANZA, not as two totals over the file. A total has slack in it, and the
         slack is not hypothetical: the raw reader recognises fewer shapes than `sites_of_stanza`
         does -- no `bash`/`system`, nothing under an unresolvable `chdir` -- so a stanza the walk
         places and this reader misses adds one to the walk's count and nothing to the floor, which
         is exactly enough to absorb a DIFFERENT stanza silently dropping out of enforcement (Codex
         P2, round 2). Asked per stanza, the two answers are about the same stanza and cannot be
         traded against a third; and the raw reader's narrower vocabulary degrades to a weaker floor
         for the stanzas it cannot see, rather than to a hole somewhere else in the file. *)
      List.iter marked ~f:(fun stanza ->
          let identity =
            Printf.sprintf "%s:%d, the %s %s" dune_file stanza.Scan.marked_line
              stanza.Scan.marked_head
              (if String.is_empty stanza.Scan.marked_name then "<unnamed>"
               else stanza.Scan.marked_name)
          in
          if stanza.Scan.marked_raw_subject && List.is_empty stanza.Scan.marked_sites then (
            Int.incr marker_holes;
            fail
              (Printf.sprintf
                 "%s: the raw text shows it running an executable and the walk placed no site for \
                  it -- it is reading the file with a hole in it, and a stanza it stops seeing is \
                  one this rule stops applying to"
                 identity));
          if stanza.Scan.marked_raw_subject then (
            Int.incr subject_floor;
            floor_names := identity :: !floor_names);
          if not (List.is_empty stanza.Scan.marked_sites) then
            walk_places :=
              ( identity,
                Printf.sprintf "%s running %s" identity
                  (String.concat ~sep:", "
                     (List.map stanza.Scan.marked_sites ~f:(fun s -> s.Scan.name))) )
              :: !walk_places);
      by_file :=
        ( dune_file,
          (if !any_declared then [ "declares " ^ Scan.backend_env_var ] else [])
          @
          match Set.to_list !words with
          | [] -> []
          | words -> [ "markers: " ^ String.concat ~sep:", " words ] )
        :: !by_file;
      (* gh-ocannl-723: the artifact-directory declaration, against the modules that read the key
         needing it. Both directions, since a declaration nothing reads for is the restatement this
         replaces rather than the relationship.

         Per SUBDIRECTORY, not per dune file (Codex P2, round 3). `(subdir gen …)` applies its
         stanzas to another directory, so the modules they name live there and the executables they
         run are theirs -- and a walk that only saw the top level reported a nested stanza's source
         as claimed by nobody. `Scan.walk` is the same descent the marker and site scans make. *)
      let artifact_groups =
        Scan.walk "" stanzas ~f:(fun subdir stanza -> [ (subdir, stanza) ])
        |> List.stable_sort ~compare:(fun (a, _) (b, _) -> String.compare a b)
        |> List.group ~break:(fun (a, _) (b, _) -> not (String.equal a b))
        |> List.map ~f:(fun group ->
            let subdir = fst (List.hd_exn group) in
            (subdir, Scan.in_subdir dir subdir, List.map group ~f:snd))
      in
      (* Runner identities are written relative to the DUNE FILE, so the raw `(subdir …)` path is
         what qualifies them -- not the repository-relative directory the modules are looked up
         in. *)
      let all_group_stanzas = List.concat_map artifact_groups ~f:(fun (_, _, group) -> group) in
      let subjects =
        List.concat_map artifact_groups ~f:(fun (subdir, here, group) ->
            let key module_name = String.lowercase (Scan.in_subdir here (module_name ^ ".ml")) in
            let calls module_name = Set.mem artifact_caller_keys (key module_name) in
            (* A module that reads `build_files_prefix` some other way needs the variable tracked
               for the same reason a caller does, and is subject to the same rule (Codex P2, rounds
               2 and 3). Narrowed textually first, the way the caller census is: the key has to be
               SPELLED for either spelling of a read to name it, so only the sources that mention it
               are parsed. *)
            let reads_prefix module_name =
              match List.Assoc.find sources ~equal:String.equal (key module_name) with
              | None -> false
              | Some on_disk -> (
                  let content = In_channel.read_all on_disk in
                  String.is_substring content ~substring:artifact_config_key
                  && try Sources.source_reads_key content ~key:artifact_config_key with _ -> false)
            in
            (* Dune's default module set is the directory less what other stanzas claim, so the scan
               needs to know what the directory holds -- a `(test (name t))` with no `(modules …)`
               builds `t.ml` (Codex P2, round 2). Only the sources this scan was handed, which is
               what it can answer for; the census check below is what catches a caller no stanza
               claims either way. *)
            let directory = if String.is_empty here then "." else here in
            let directory_modules =
              List.filter_map source_files ~f:(fun (path, _) ->
                  if String.equal (Stdlib.Filename.dirname path) directory then
                    Some (Stdlib.Filename.remove_extension (Stdlib.Filename.basename path))
                  else None)
            in
            List.map
              (Scan.artifact_subjects ~directory_modules ~subdir ~runner_stanzas:all_group_stanzas
                 group ~calls ~reads_prefix) ~f:(fun subject -> (here, subject)))
      in
      List.iter subjects ~f:(fun (here, subject) ->
          let where =
            Printf.sprintf "%s, the %s %s" dune_file subject.Scan.artifact_head
              subject.Scan.artifact_name
          in
          let needs = subject.Scan.artifact_callers @ subject.Scan.artifact_readers in
          let what =
            match (subject.Scan.artifact_callers, subject.Scan.artifact_readers) with
            | [], readers ->
                String.concat ~sep:", " readers ^ " reads `" ^ artifact_config_key ^ "` by name"
            | callers, [] -> String.concat ~sep:", " callers ^ " calls `Test_utils.Generated.init`"
            | callers, readers ->
                String.concat ~sep:", " callers ^ " calls `Test_utils.Generated.init` and "
                ^ String.concat ~sep:", " readers ^ " reads `" ^ artifact_config_key ^ "` by name"
          in
          List.iter needs ~f:(fun m ->
              artifact_claimed :=
                Set.add !artifact_claimed (String.lowercase (Scan.in_subdir here (m ^ ".ml"))));
          artifact_table :=
            Printf.sprintf "  %-58s %s (%s)" where
              (Scan.artifact_verdict_name subject.Scan.artifact_verdict)
              (if List.is_empty needs then subject.Scan.artifact_deps_site
               else String.concat ~sep:", " needs)
            :: !artifact_table;
          match subject.Scan.artifact_verdict with
          | Scan.Artifact_declared | Scan.Artifact_other_reader -> ()
          | Scan.Artifact_undeclared ->
              Int.incr artifact_violations;
              fail
                (Printf.sprintf
                   "%s: %s -- `%s` decides which directory the run's generated artifacts are read \
                    from, and %s does not declare `(env_var %s)`, so dune serves the previous \
                    run's result across a change of it. Add the declaration there"
                   where what artifact_config_key subject.Scan.artifact_deps_site
                   Scan.artifact_env_var)
          | Scan.Artifact_stale_declaration ->
              Int.incr artifact_violations;
              fail
                (Printf.sprintf
                   "%s declares `(env_var %s)` in %s and no module of it reads `%s` at all -- \
                    neither through `Test_utils.Generated.init` nor by name. A declaration with \
                    nothing behind it is a restatement, not a relationship, and the next author \
                    copies it. Drop it, or read the generated artifacts through \
                    `Test_utils.Generated`, which is the one supported way to read them"
                   where Scan.artifact_env_var subject.Scan.artifact_deps_site artifact_config_key)
          | Scan.Artifact_unrun ->
              Int.incr artifact_violations;
              fail
                (Printf.sprintf
                   "%s: %s, and no stanza in this file runs the executable -- an `(executable)` \
                    has no `deps` field, so the `(env_var %s)` declaration goes on the rule that \
                    RUNS it, the same placement as the `%s` dep and the backend marker. This scan \
                    can find neither"
                   where what Scan.artifact_env_var Scan.config_file)
          | Scan.Artifact_in_library ->
              Int.incr artifact_violations;
              fail
                (Printf.sprintf
                   "%s: %s, from a library module -- the initializer empties the artifact \
                    directory of the process that owns it, so it belongs to an executable's own \
                    modules. Called through a library it puts the `(env_var %s)` requirement on \
                    every stanza that links the library, where nothing follows it"
                   where what Scan.artifact_env_var));
      artifact_by_file := (dune_file, List.map subjects ~f:snd) :: !artifact_by_file;
      (* The ambient gate, per directory AND per alias (gh-ocannl-652). *)
      let gated_here = gated_aliases stanzas in
      let entry_points = entry_points stanzas in
      (* The lock (Codex P1 round 4): a gate in a file whose actions take it has to take it too. *)
      if List.exists stanzas ~f:takes_training_lock then
        List.iter stanzas ~f:(fun s ->
            if is_gate s && not (takes_training_lock s) then
              fail
                (Printf.sprintf
                   "%s serializes its actions on `%s` and its gate on `%s` does not take the lock \
                    -- the one unlocked action in a file of locked ones is what the next training \
                    test gets copied from; add `(locks %s)` to it"
                   dune_file training_lock
                   (String.concat ~sep:", " (aliases_of s))
                   training_lock));
      List.iter entry_points ~f:(fun alias ->
          if Set.mem gated_here alias then gated := (dune_file, alias) :: !gated
          else if Map.mem gateless dune_file then gateless_used := Set.add !gateless_used dune_file
          else
            fail
              (Printf.sprintf
                 "%s has actions on the `%s` alias and no ambient gate reaches it -- nothing here \
                  declares a rejected environment spelling, so `ocannl_backend=cuda dune build \
                  @%s` would serve this directory's cached results with the fatal startup check \
                  never reached; copy the `env_spelling_gate` stanza for that alias from a \
                  neighbour, depend on the gate's alias from the rule, or exempt the directory by \
                  name with the reason"
                 dune_file alias
                 (if String.equal alias "runtest" then
                    Stdlib.Filename.dirname dune_file ^ "/" ^ alias
                  else alias)));
      (* Each suite's members, against what it aggregates. *)
      List.iter suites ~f:(fun suite ->
          let reaches = aliases_reached_from stanzas suite in
          List.iter entry_points ~f:(fun alias ->
              if member_of suite alias && not (Set.mem reaches alias) then
                fail
                  (Printf.sprintf
                     "%s attaches a rule to `%s` that the `%s` alias does not aggregate -- `dune \
                      build @%s` would skip it silently; list `(alias %s)` in the `(alias (name \
                      %s) (deps …))` stanza"
                     dune_file alias suite suite alias suite)));
      (* The scans family, the same question asked of the repo-wide scans (gh-ocannl-703): the rule
         that diffs a scan's output against its golden is the one that fails, so it is the one
         `@scans` has to reach. The producers are recognized by what they read rather than by name,
         so a scan added tomorrow is asked about too. *)
      let scans_reaches = aliases_reached_from stanzas scans_suite in
      List.iter stanzas ~f:(fun producer ->
          if is_repo_wide_scan producer then
            (* A producer the family already aggregates directly needs no separate checker at all:
               declaring a target says where its diagnostics go, not that a second rule judges them
               -- a scan can assert by exit status and still write one (Codex P2, round 8). Asked
               before the target hunt, so both shapes are accepted the same way. *)
            if List.exists (aliases_of producer) ~f:(Set.mem scans_reaches) then ()
            else if List.is_empty (targets_of producer) then
              (* A scan that declares no target writes and checks in one action -- the `no-infer`
                 shape this repository uses elsewhere, or an assertion by exit status. There is no
                 second rule to look for, so the family has to aggregate THIS rule's own alias;
                 iterating over its (empty) target list would have passed it silently, which is the
                 fail-open Codex found in round 4. *)
              if not (List.exists (aliases_of producer) ~f:(Set.mem scans_reaches)) then
                fail
                  (Printf.sprintf
                     "%s has a rule that globs the repository -- a repo-wide scan -- and declares \
                      no target, so it checks its own output in its action, and the `%s` alias \
                      does not aggregate the alias it sits on (%s): `dune build @%s/%s` would skip \
                      it silently. List its alias in the `(alias (name %s) (deps …))` stanza"
                     dune_file scans_suite
                     (match aliases_of producer with
                     | [] -> "none"
                     | aliases -> String.concat ~sep:", " aliases)
                     (Stdlib.Filename.dirname dune_file)
                     scans_suite scans_suite)
              else ()
            else
              List.iter (targets_of producer) ~f:(fun target ->
                  let checkers =
                    List.filter stanzas ~f:(fun s ->
                        is_golden_diff s
                        && Option.value_map (Scan.field s "action") ~default:false ~f:(fun args ->
                            List.exists args ~f:(diffs_file ~named:(named_deps s) target)))
                  in
                  let aliases = List.concat_map checkers ~f:aliases_of in
                  if not (List.exists aliases ~f:(Set.mem scans_reaches)) then
                    fail
                      (Printf.sprintf
                         "%s globs the repository to produce `%s` -- a repo-wide scan -- and no \
                          rule the `%s` alias aggregates diffs it against its golden: `dune build \
                          @%s/%s` would skip it silently. Give the diff rule `(alias \
                          runtest-<name>)` -- that alias and no other -- and list `(alias \
                          runtest-<name>)` in BOTH the `(alias (name runtest) (deps …))` and the \
                          `(alias (name %s) (deps …))` stanzas"
                         dune_file target scans_suite
                         (Stdlib.Filename.dirname dune_file)
                         scans_suite scans_suite)));
      (* The focused aggregates (gh-ocannl-783): the same completeness question, asked of two more
         families whose membership is derived from what the member stanza declares. *)
      (* The stanzas as dune applies them, each with the `(subdir …)` it sits under. `marked_stanzas`
         makes the same descent and carries the same pairing, so the two readings of one file agree
         about which stanzas it holds and where. *)
      let file_stanzas = List.map marked ~f:(fun m -> (m.Scan.marked_subdir, m.Scan.marked_sexp)) in
      (* Dune resolves a default module set within ONE directory, so the stanzas that can narrow a
         stanza's claim are that subdirectory's, not the file's: an explicit `(modules probe)` in
         `(subdir a …)` must not remove `b/probe.ml` from what a default-module stanza in `(subdir b
         …)` builds (Codex P2, round 1). *)
      let stanzas_in subdir =
        List.filter_map file_stanzas ~f:(fun (sub, stanza) ->
            if String.equal sub subdir then Some stanza else None)
      in
      (* Every unit the file builds, each in the subdirectory dune applies it to. *)
      let units =
        List.concat_map file_stanzas ~f:(fun (subdir, stanza) ->
            family_units (stanzas_in subdir) ~subdir stanza)
      in
      (* The modules of ONE unit. Dune builds each name of a plural stanza as its own executable --
         its main module plus the stanza's shared ones, and NOT the other names' mains -- so asking
         the question of the stanza's whole module list makes one main's use of the instrumentation
         a claim about its neighbour too (Codex P2, round 2). *)
      let unit_module_sources { unit_subdir; unit_stanza; unit_name; _ } =
        let here = Scan.in_subdir dir unit_subdir in
        let directory = if String.is_empty here then "." else here in
        let directory_modules =
          List.filter_map source_files ~f:(fun (path, _) ->
              if String.equal (Stdlib.Filename.dirname path) directory then
                Some (Stdlib.Filename.remove_extension (Stdlib.Filename.basename path))
              else None)
        in
        let siblings =
          match unit_name with
          | None -> []
          | Some name ->
              List.filter (Scan.names_of unit_stanza) ~f:(fun other ->
                  not (String.equal other name))
              |> List.map ~f:String.lowercase
        in
        Scan.modules_of ~directory_modules (stanzas_in unit_subdir) unit_stanza
        |> List.filter ~f:(fun module_name ->
            not (List.mem siblings (String.lowercase module_name) ~equal:String.equal))
        |> List.map ~f:(fun module_name ->
            String.lowercase (Scan.in_subdir here (module_name ^ ".ml")))
      in
      (* The Metal derivation reads the MARKED stanza, whatever kind it is. For an `(executable)` the
         marker's required placement is the rule that runs it (gh-ocannl-659), so a Metal test in
         that form carries its marker on an unnamed `(rule …)`: asking the question of the executable
         stanza instead would find no marker and let the family omit the test while this scan passed
         (Codex P2, round 1). Whatever stanza carries the marker is the member -- every unit of it,
         since a marker is a statement about the stanza. *)
      let metal_stanzas =
        List.filter marked ~f:(fun stanza ->
            match Scan.backend_rule_of stanza with
            | Scan.Names_backend (_, body) | Scan.Declares_and_names (_, body) ->
                (* The marker admits several words, comma-separated, where a stanza honestly names
                   two backends -- so membership is "names metal among them", not "is spelled
                   metal". *)
                List.mem (String.split body.Scan.backend ~on:',') "metal" ~equal:String.equal
            | Scan.Runs_nothing | Scan.Marker_without_run _ | Scan.Declares_variable
            | Scan.Names_twice _ | Scan.Names_neither ->
                false)
        |> List.map ~f:(fun m -> (m.Scan.marked_subdir, m.Scan.marked_sexp))
      in
      let metal_members =
        List.filter units ~f:(fun u ->
            List.exists metal_stanzas ~f:(fun (subdir, stanza) ->
                String.equal subdir u.unit_subdir && phys_equal stanza u.unit_stanza))
      in
      let lifecycle_members =
        List.filter units ~f:(fun u ->
            has_family_modules u.unit_stanza
            && List.exists (unit_module_sources u) ~f:(Set.mem lifecycle_sources))
      in
      List.iter
        [ (metal_family, metal_members); (lifecycle_family, lifecycle_members) ]
        ~f:(fun (family, members) ->
          (* Reachability PER DIRECTORY, from the stanzas dune applies there: a `(subdir child …)`
             group may carry its own `(alias (name <family>) …)`, and the root recursive
             `@<family>` build reaches it, so a member there is correctly wired (Codex P2, round
             2). What is never right is reaching a member from another directory's alias, which is
             what asking the file as a whole would have allowed. *)
          let reaches subdir = aliases_reached_from (stanzas_in subdir) family.family_alias in
          List.iter members ~f:(fun u ->
              let reaches = reaches u.unit_subdir in
              let aggregated = List.exists u.unit_aliases ~f:(Set.mem reaches) in
              family_table :=
                (dune_file, family.family_alias, u.unit_identity, aggregated) :: !family_table;
              if not aggregated then
                fail
                  (Printf.sprintf
                     "%s: the %s%s %s, and the `%s` alias does not reach it -- `dune build @%s` \
                      would skip it silently. List %s in %s `(alias (name %s) (deps …))` stanza, \
                      adding the stanza if that directory has no member yet"
                     dune_file u.unit_identity
                     (if String.is_empty u.unit_subdir then ""
                      else Printf.sprintf " in `(subdir %s …)`" u.unit_subdir)
                     family.family_is family.family_alias family.family_alias
                     (match u.unit_aliases with
                     | [] ->
                         "the alias of the rule that runs it -- it has none, being an \
                          `(executable)` no rule in that directory runs"
                     | aliases ->
                         String.concat ~sep:" or "
                           (List.map aliases ~f:(fun alias -> Printf.sprintf "`(alias %s)`" alias)))
                     (if String.is_empty u.unit_subdir then "this file's"
                      else "that group's")
                     family.family_alias)));
      (* A hand-written per-test alias must not reuse a name dune generates one for: the aliases
         merge, and the targeted run stops being one test (Codex P2, round 5). Asked of every rule
         in the file rather than of the golden diffs alone, since any rule can be given such an
         alias; the ambient gate is the one rule that means to share it. *)
      let generated = generated_runtest_names stanzas in
      let gate_names = gate_generated_names stanzas in
      List.iter stanzas ~f:(fun stanza ->
          List.iter (aliases_of stanza) ~f:(fun alias ->
              match String.chop_prefix alias ~prefix:"runtest-" with
              (* The one deliberate collision: a rule sharing the alias dune generates for a
                 `(test)` stanza BECAUSE IT RUNS THE SAME BINARY -- the ambient gate, whose rule
                 exists so that every per-test alias can depend on it. Three things have to hold,
                 and the third is what the earlier rounds were missing: the rule is a gate, the name
                 belongs to a gate `(test)` stanza, and the rule's action runs that stanza's
                 executable, so the merged alias runs one program either way (Codex P2, rounds 6 to
                 8). *)
              | Some name
                when is_gate stanza && Set.mem gate_names name
                     && List.exists (Scan.executables_run stanza) ~f:(fun (_cwd, command) ->
                         match command with
                         | Scan.Runs path ->
                             String.equal (Stdlib.Filename.basename path) (name ^ ".exe")
                         | _ -> false) ->
                  ()
              | Some name when Set.mem generated name ->
                  fail
                    (Printf.sprintf
                       "%s attaches a rule to `%s`, the alias dune generates for the `%s` stanza \
                        in this directory -- the two merge, so `dune build @%s/%s` would run that \
                        test as well as this rule, and naming `runtest` beside it is a dependency \
                        cycle. Name the alias after the GOLDEN this rule checks, qualified where a \
                        run writes several"
                       dune_file alias name
                       (Stdlib.Filename.dirname dune_file)
                       alias)
              | _ -> ()));
      (* And one alias checks one golden: two golden diffs sharing an alias make the targeted run
         two tests, which is the isolation this arrangement is for -- and the prefix relation above
         admits the pair, since `foo.expected` and `foo-extension.expected` both accept
         `runtest-foo-extension` (Codex P2, round 7). A producer rule sharing its checker's alias is
         a different thing and stays allowed: it is what MAKES the output the checker reads. *)
      List.iter
        (List.concat_map stanzas ~f:(fun s ->
             (* One entry per GOLDEN, not per rule: a single rule whose `progn` diffs two goldens
                puts two checks behind one alias just as two rules would (Codex P2, round 8). *)
             if is_golden_diff s then
               List.concat_map (aliases_of s) ~f:(fun alias ->
                   List.map (goldens_in s) ~f:(fun _ -> alias))
             else [])
        |> List.sort ~compare:String.compare
        |> List.group ~break:(fun a b -> not (String.equal a b))
        |> List.filter ~f:(fun group -> List.length group > 1))
        ~f:(fun group ->
          fail
            (Printf.sprintf
               "%s checks %d goldens on the alias `%s` -- `dune build @%s/%s` would run them all, \
                so the alias no longer names one test. Give each its own, named after the golden \
                it checks"
               dune_file (List.length group) (List.hd_exn group)
               (Stdlib.Filename.dirname dune_file)
               (List.hd_exn group)));
      (* Every golden diff sits on a per-test alias, and on that alias ALONE (gh-ocannl-726). Dune
         generates `runtest-<name>` for `(test)`/`(tests)` stanzas and inline-test libraries and for
         nothing else, so a rule that diffs a golden and names `runtest` itself can only be run by
         running the whole directory -- and validating it targeted then means building its `.actual`
         and diffing by hand, outside dune, which fails open. Naming BOTH aliases does not fix it
         either, and is the trap worth checking for: a rule attached to two aliases makes building
         either one build both, so the per-test alias would drag the directory in behind it, and
         where the name is one dune generates the pair is a dependency cycle outright. *)
      List.iter stanzas ~f:(fun stanza ->
          if is_golden_diff stanza then
            match aliases_of stanza with
            (* One alias, and it names a member of a suite: that is the whole convention. Asked of
               the alias SET rather than of the name `runtest` alone (Codex P2, round 1), since a
               rule with no alias at all, or with an alias of its own invention, has neither the
               targeted entry point nor a place in a suite -- and its golden stops being checked
               without anything saying so. *)
            | [ alias ]
              when List.exists suites ~f:(fun suite -> member_of suite alias)
                   && List.for_all (goldens_in stanza) ~f:(fun golden ->
                       let suffix =
                         List.find_map suites ~f:(fun suite ->
                             String.chop_prefix alias ~prefix:(suite ^ "-"))
                         |> Option.value ~default:alias
                       in
                       let stem = golden_stem ~named:(named_deps stanza) golden in
                       (* The stem itself, or the stem and then a qualifier saying WHICH golden of
                          the run this is. A bare prefix would accept `runtest-foobar` for
                          `foo.expected`, leaving the alias a reader constructs empty (Codex P2,
                          round 8). *)
                       (not (String.is_empty stem))
                       && (String.equal suffix stem || String.is_prefix suffix ~prefix:(stem ^ "-")))
              ->
                ()
            | aliases ->
                let what =
                  match aliases with
                  | [] -> "attaches a golden diff to no alias at all"
                  | [ alias ] when not (List.exists suites ~f:(fun s -> member_of s alias)) ->
                      Printf.sprintf "attaches a golden diff to `%s`, which is no suite's member"
                        alias
                  | [ alias ]
                    when List.exists (goldens_in stanza) ~f:(fun g ->
                             String.is_empty (golden_stem ~named:(named_deps stanza) g)) ->
                      Printf.sprintf
                        "attaches `%s` to a golden this check cannot name -- %s reduces to an \
                         empty stem, so nothing constrains the alias. Spell the golden as a plain \
                         path, or teach `golden_stem` the form"
                        alias
                        (String.concat ~sep:", "
                           (List.filter (goldens_in stanza) ~f:(fun g ->
                                String.is_empty (golden_stem ~named:(named_deps stanza) g))))
                  | [ alias ] ->
                      Printf.sprintf
                        "attaches `%s` to a golden its name does not name: the goldens are %s, so \
                         the alias should begin `<suite>-%s`. A reader reaches for the alias with \
                         the failing GOLDEN in hand, and an alias that renames it is one they \
                         construct empty"
                        alias
                        (String.concat ~sep:", " (goldens_in stanza))
                        (String.concat ~sep:"` or `<suite>-"
                           (List.map (goldens_in stanza)
                              ~f:(golden_stem ~named:(named_deps stanza))))
                  | aliases ->
                      Printf.sprintf
                        "attaches a golden diff to %d aliases (%s) -- and a rule on two aliases \
                         makes building either one build both"
                        (List.length aliases) (String.concat ~sep:", " aliases)
                in
                fail
                  (Printf.sprintf
                     "%s %s. Give it `(alias <suite>-<name>)` -- that alias and no other, with \
                      <suite> one of %s -- and list it in this file's `(alias (name <suite>) (deps \
                      …))` stanza, which is what runs it as part of the suite. <name> is the \
                      golden the rule checks, and for `runtest` must not be the name of a `(test)` \
                      stanza in this directory, since dune generates `runtest-<name>` for those"
                     dune_file what (String.concat ~sep:", " suites)));
      let fields = List.concat_map stanzas ~f:dep_fields in
      let declared =
        List.concat_map fields ~f:(fun (_, args) -> List.concat_map args ~f:env_vars_in)
      in
      (* A declaration this scan did not look inside a dependency field for is one it cannot check,
         so the two counts have to agree. Dune admits `(env_var ...)` only in a dependency
         specification, so a disagreement means a field spelling this scan does not know. *)
      let all = List.concat_map stanzas ~f:env_vars_in in
      if List.length all <> List.length declared then
        fail
          (Printf.sprintf
             "%s declares %d `(env_var ...)` dependencies but only %d of them are in a `deps` or \
              `preprocessor_deps` field -- teach this check the field that holds the others"
             dune_file (List.length all) (List.length declared));
      List.iter fields ~f:(fun (_field, args) ->
          let names = List.concat_map args ~f:env_vars_in in
          List.iter names ~f:(fun name ->
              let key = dune_file ^ ":" ^ name in
              match Utils.classify_env_var name with
              (* Someone else's variable entirely (`PATH`, `HOME`): not this check's business. *)
              | Utils.Env_not_addressed -> ()
              | Utils.Env_config_key config_key -> tracked_keys := Set.add !tracked_keys config_key
              (* The gates are checked below, against the modules that read them. *)
              | Utils.Env_reserved _ -> ()
              | Utils.Env_unread_spelling _ | Utils.Env_unknown_key _ | Utils.Env_unread_reserved _
                ->
                  if Map.mem exemptions key then exemptions_used := Set.add !exemptions_used key
                  else
                    let what =
                      match Utils.classify_env_var name with
                      | Utils.Env_unread_reserved prefix ->
                          "is in the reserved " ^ prefix
                          ^ " namespace in a casing its reader does not consult"
                      | _ -> "addresses OCANNL and names no configuration key it reads"
                    in
                    fail
                      (Printf.sprintf
                         "%s declares `(env_var %s)`, which %s -- a variable nothing consults \
                          invalidates nothing; fix the spelling, or exempt it by name with the \
                          reason"
                         dune_file name what)));
      (* What a stanza's own modules read, against what the stanza declares: the gates, read while
         PREPROCESSING them, and the ambient variables they read by name at RUN time. *)
      List.iter stanzas ~f:(fun stanza ->
          match (Scan.head stanza, Scan.field stanza "modules") with
          | Some kind, Some modules when List.mem module_stanzas kind ~equal:String.equal ->
              let name = match Scan.names_of stanza with name :: _ -> name | [] -> "<unnamed>" in
              let where = Printf.sprintf "%s, %s %s" dune_file kind name in
              let env_vars_of field =
                match Scan.field stanza field with
                | None -> []
                | Some args -> List.concat_map args ~f:env_vars_in
              in
              let declared_gates =
                List.filter (env_vars_of "preprocessor_deps") ~f:(fun name ->
                    match Utils.classify_env_var name with
                    | Utils.Env_reserved prefix -> String.equal prefix gate_prefix
                    | _ -> false)
              in
              let sources =
                List.filter_map modules ~f:(function
                  | Sexp.Atom module_name ->
                      Option.map (source_of ~dir module_name) ~f:(fun on_disk ->
                          (module_name ^ ".ml", In_channel.read_all on_disk))
                  | _ -> None)
              in
              let read_gates =
                List.concat_map sources ~f:(fun (source, content) ->
                    Sources.tracing_gates_in_source content
                    |> List.map ~f:(fun gate -> (gate, source)))
              in
              if (not (List.is_empty declared_gates)) && not (Set.mem scanned_dirs dir) then
                fail
                  (Printf.sprintf
                     "%s declares tracing gates, and this check was handed no sources from %s to \
                      check them against -- add the directory to the rule\'s globs"
                     where
                     (if String.is_empty dir then "the repository root" else dir))
              else (
                List.iter read_gates ~f:(fun (gate, source) ->
                    if not (List.mem declared_gates gate ~equal:String.equal) then
                      fail
                        (Printf.sprintf
                           "%s/%s reads the tracing gate %s while being preprocessed, and %s does \
                            not declare it -- setting the variable then returns the modules \
                            already built without the trace statements"
                           dir source gate where)
                    else gate_table := (where, gate, dir ^ "/" ^ source) :: !gate_table);
                List.iter declared_gates ~f:(fun gate ->
                    if not (List.Assoc.mem read_gates gate ~equal:String.equal) then
                      fail
                        (Printf.sprintf
                           "%s declares the tracing gate %s, which none of its modules reads -- \
                            drop it, or move it to the library whose modules do"
                           where gate)));
              (* The run-time half (Codex P2, round 2). A gate is read while the module is built;
                 `Sys.getenv "OCANNL_TOOL_TEST_RESTRICT_MASK"` is read while it RUNS, and the
                 consequence of not declaring it is the same one this whole check is about --
                 `test_cpu_topology` was reusable across a change of the mask that decides what it
                 does. Presence is checkable here and not for configuration keys at large, which a
                 test reaches through the library rather than by name: what makes the difference is
                 the literal in the source, which says exactly which variable this module reads.

                 An `(executable)` stanza has no `deps` field at all -- its companion rule carries
                 them -- so for those the declaration is looked for anywhere in the dune file, the
                 same latitude `config_dep_completeness` gives the `ocannl_config` dep. *)
              let declared_for_stanza =
                match Scan.field stanza "deps" with
                | Some _ -> env_vars_of "deps"
                | None -> declared
              in
              List.iter sources ~f:(fun (source, content) ->
                  Sources.env_var_reads_in_source content
                  |> List.iter ~f:(fun read ->
                      match Utils.classify_env_var read with
                      | Utils.Env_not_addressed -> ()
                      | _ ->
                          if not (List.mem declared_for_stanza read ~equal:String.equal) then
                            fail
                              (Printf.sprintf
                                 "%s/%s reads the environment variable %s by name, and %s does not \
                                  declare it -- dune then reuses the previous result across a \
                                  change of the variable that decides what the run does"
                                 dir source read where)
                          else read_table := (where, read, dir ^ "/" ^ source) :: !read_table))
          | _ -> ()));
  let stale =
    Set.diff (Set.of_list (module String) (List.map exempt_declarations ~f:fst)) !exemptions_used
  in
  if not (Set.is_empty stale) then
    fail
      (Printf.sprintf
         "exempted declarations no dune file makes any more -- drop them from the exemption list: \
          %s"
         (String.concat ~sep:", " (Set.to_list stale)));
  (* The coverage of the gate half, stated rather than assumed: a library outside these directories
     has no sources here to check its `preprocessor_deps` against, and says so above if it declares
     a gate at all -- but a gate it FAILS to declare is invisible from here, so the boundary belongs
     in the golden where widening it is a reviewable diff. *)
  printf "Directories whose sources this scan reads:\n";
  Set.iter scanned_dirs ~f:(fun dir -> printf "  %s\n" (if String.is_empty dir then "." else dir));
  printf "\nConfiguration keys tracked as ambient dependencies, in the one spelling OCANNL reads:\n";
  Set.iter !tracked_keys ~f:(printf "  %s\n");
  printf "\nPer-module tracing gates, and the library whose preprocessor_deps declares each:\n";
  List.sort !gate_table ~compare:(fun (_, a, _) (_, b, _) -> String.compare a b)
  |> List.iter ~f:(fun (where, gate, source) -> printf "  %-30s %s (%s)\n" gate where source);
  printf
    "\nAmbient variables a module reads by name at run time, and the stanza that declares each:\n";
  List.sort !read_table ~compare:(fun (_, a, _) (_, b, _) -> String.compare a b)
  |> List.iter ~f:(fun (where, read, source) -> printf "  %-30s %s (%s)\n" read where source);
  let stale_gateless =
    Set.diff (Set.of_list (module String) (List.map gateless_dirs ~f:fst)) !gateless_used
  in
  if not (Set.is_empty stale_gateless) then
    fail
      (Printf.sprintf
         "directories exempted from the ambient gate that no longer run tests -- drop them from \
          the exemption list: %s"
         (String.concat ~sep:", " (Set.to_list stale_gateless)));
  printf "\nAmbient environment gates, by dune file and every alias whose build runs one:\n";
  List.sort !gated ~compare:(fun (a, x) (b, y) ->
      match String.compare a b with 0 -> String.compare x y | c -> c)
  |> List.iter ~f:(fun (dune_file, alias) -> printf "  %-40s @%s\n" dune_file alias);
  List.iter gateless_dirs ~f:(fun (dir, why) -> printf "  %s -- no gate: %s\n" dir why);
  printf "\nDeclarations of a name OCANNL does not read as a configuration key, exempt by design:\n";
  List.iter exempt_declarations ~f:(fun (key, why) -> printf "  %s -- %s\n" key why);
  printf
    "\n\
     Artifact-directory declarations, by dune file. A stanza whose modules call\n\
     `Test_utils.Generated.init` declares `(env_var %s)`\n\
     where dune runs it -- in its own `(deps ...)`, or, for an `(executable)` which has none, in\n\
     the rule that runs it (gh-ocannl-723). What is held here is which VERDICTS a file's stanzas\n\
     draw, not how many draw each: a tally would move whenever a test is added, and the per-stanza\n\
     table goes to stderr.\n"
    Scan.artifact_env_var;
  printf "  %s\n"
    (if repository_census then
       Printf.sprintf "the census covers every scan root, so the floor of %d callers applies to it"
         artifact_caller_floor
     else
       "the census does not cover the repository's scan roots, so only the relationship is asked \
        of it");
  List.sort !artifact_by_file ~compare:(fun (a, _) (b, _) -> String.compare a b)
  |> List.iter ~f:(fun (dune_file, subjects) ->
      if not (List.is_empty subjects) then
        printf "  %s: %s\n" dune_file
          (List.map subjects ~f:(fun s -> Scan.artifact_verdict_name s.Scan.artifact_verdict)
          |> List.dedup_and_sort ~compare:String.compare
          |> String.concat ~sep:", "));
  (* gh-ocannl-659. The golden holds which backend WORDS a dune file's markers use, not how many
     stanzas carry each: a tally there would move on every test added anywhere in the repository,
     which is the churn gh-ocannl-665 took out of `config_dep_completeness` for the same reason. The
     per-stanza classification is not centralized here at all -- it lives in the marker comment next
     to the stanza, which is the point of putting it there; what this file pins is that a directory
     did not quietly lose a whole class of them. *)
  printf
    "\n\
     Backend declarations, by dune file. Every stanza that runs an executable either declares\n\
     `(env_var %s)`, or carries the marker comment\n\
    \    ; %s <%s> -- <reason>\n\
     inside its parentheses. What is held here is which WORDS a file's markers use, not how many\n\
     stanzas carry each: the per-stanza reasons live next to the stanzas, where the next author\n\
     will copy them from, and the tallies go to stderr (gh-ocannl-659, gh-ocannl-665).\n"
    Scan.backend_env_var Scan.marker_sentinel
    (String.concat ~sep:"|" Scan.marker_backends);
  List.sort !by_file ~compare:(fun (a, _) (b, _) -> String.compare a b)
  |> List.iter ~f:(fun (dune_file, present) ->
      printf "  %s: %s\n" dune_file
        (if List.is_empty present then "nothing that runs a test executable"
         else String.concat ~sep:"; " present));
  (* gh-ocannl-783. The golden holds which FAMILIES a dune file has derived members of, not how many
     stanzas each has -- the same reason the two sections above hold words and verdicts rather than
     tallies: a count would move whenever a Metal test or a lifecycle probe is added, and what is
     worth a reviewable diff is a directory acquiring or losing a family. The per-member table, with
     the alias that reaches each, goes to stderr. *)
  printf
    "\n\
     Focused aggregate families, by dune file (gh-ocannl-783). Each family alias is spelled\n\
     identically in every directory that has members, so `dune build @<family>` from the workspace\n\
     root runs the whole family; membership is derived from what a member stanza declares, and a\n\
     member the family's `(alias (name <family>) (deps ...))` stanza omits fails this scan.\n";
  List.map !family_table ~f:(fun (dune_file, family, _, _) -> (dune_file, family))
  |> List.dedup_and_sort ~compare:Poly.compare
  |> List.group ~break:(fun (a, _) (b, _) -> not (String.equal a b))
  |> List.iter ~f:(fun group ->
      printf "  %s: %s\n"
        (fst (List.hd_exn group))
        (String.concat ~sep:", " (List.map group ~f:snd)));
  eprintf
    "Focused-aggregate members, and the family alias that reaches each (not diffed -- see \
     gh-ocannl-665):\n\
     %s\n"
    (String.concat ~sep:"\n"
       (List.map
          (List.sort !family_table ~compare:Poly.compare)
          ~f:(fun (dune_file, family, identity, aggregated) ->
            Printf.sprintf "  %-24s %-40s the %s%s" ("@" ^ family) dune_file identity
              (if aggregated then "" else " -- NOT AGGREGATED"))));
  eprintf
    "Backend classification of every stanza that runs an executable (not diffed -- see \
     gh-ocannl-665):\n\
     %s\n"
    (String.concat ~sep:"\n" (List.rev !classification));
  eprintf "Totals: %d such stanzas, against a raw-text floor of %d.\n" !placed_subjects
    !subject_floor;
  (* The two populations, compared as SORTED LISTS OF STANZAS rather than as totals. A number would
     say "one short" and leave which stanza to arithmetic; these say which file and line each reader
     is alone on, and the claim below is about the stanzas themselves (gh-ocannl-708). Both
     directions, because they fail differently: a stanza only the floor names is the walk going
     blind, and one only the walk places is a stanza whose enforcement nothing independent vouches
     for. *)
  let placed_identities = List.sort (List.map !walk_places ~f:fst) ~compare:String.compare in
  let floored_identities = List.sort !floor_names ~compare:String.compare in
  let detail = Map.of_alist_multi (module String) !walk_places in
  let describe identity =
    match Map.find detail identity with Some (what :: _) -> what | _ -> identity
  in
  (* As MULTISETS, not sets. Two stanzas opening on the same line of the same file share an
     identity, and comparing deduplicated lists would let the floored one answer for the other --
     the collapse `config_dep_completeness` compares (directory, executable) pairs with multiplicity
     to avoid (Codex P2, rounds 2 and 3 of PR #343). *)
  let counts identities =
    List.fold identities
      ~init:(Map.empty (module String))
      ~f:(fun tally identity ->
        Map.update tally identity ~f:(fun n -> 1 + Option.value n ~default:0))
  in
  let excess these those =
    let those = counts those in
    Map.to_alist (counts these)
    |> List.concat_map ~f:(fun (identity, mine) ->
        let theirs = Option.value (Map.find those identity) ~default:0 in
        List.init (Int.max 0 (mine - theirs)) ~f:(fun _ -> identity))
  in
  let walk_only = excess placed_identities floored_identities in
  let floor_only = excess floored_identities placed_identities in
  (* The claim is quantified over the UNION, through `Verdict.p_all`, which carries the
     non-emptiness guard with it: a scan that stopped reading dune files altogether would leave two
     empty populations, and "the same stanzas" holds vacuously of nothing (gh-ocannl-729). *)
  let population =
    List.dedup_and_sort (placed_identities @ floored_identities) ~compare:String.compare
  in
  let placed_counts = counts placed_identities and floored_counts = counts floored_identities in
  let named_by_both identity =
    Option.value (Map.find placed_counts identity) ~default:0
    = Option.value (Map.find floored_counts identity) ~default:0
  in
  (match walk_only with
  | [] -> eprintf "Every one of them has a second reader's floor under it.\n"
  | walk_only ->
      eprintf
        "The %d standing on the walk alone -- a site is placed and the raw reader names nothing \
         there. Teach `Scan.raw_stanza_of` the shape, or the rule applies to a stanza nothing \
         independent vouches for:\n\
         %s\n"
        (List.length walk_only)
        (String.concat ~sep:"\n" (List.map walk_only ~f:(fun i -> "  " ^ describe i))));
  (match floor_only with
  | [] -> ()
  | floor_only ->
      eprintf
        "And the %d the raw reader names with no site placed -- the walk reading the file with a \
         hole in it:\n\
         %s\n"
        (List.length floor_only)
        (String.concat ~sep:"\n" (List.map floor_only ~f:(fun i -> "  " ^ i))));
  printf "\n";
  Verdict.p
    "every stanza that runs an executable either declares the backend variable or says in place \
     why it does not"
    (!xor_violations = 0);
  Verdict.p
    "every marker the text spells was read as one, and every stanza the raw text shows running an \
     executable has a site placed for it"
    (!marker_holes = 0);
  Verdict.p_all
    "every stanza either reader names as running an executable is named by the other too" population
    ~f:named_by_both;
  (* The relationship, and a floor under the population that carries it (gh-ocannl-729): a
     derivation that stopped finding members -- a marker grammar change, a renamed instrumentation
     module, a glob that stopped reaching the sources -- would leave "every member is aggregated"
     true of nothing and print the line a healthy repository prints. The floor is a statement about
     the REPOSITORY, so it is asked only of a run that has it: the control's synthetic trees
     legitimately contain one member or none, and the relationship is checked over them either way.
     Same shape, and the same reasoning, as the artifact census floor below. *)
  let family_members = List.sort !family_table ~compare:Poly.compare in
  let family_unaggregated = List.filter family_members ~f:(fun (_, _, _, reached) -> not reached) in
  let family_floor_met =
    (not repository_census) || List.length family_members >= family_member_floor
  in
  if not family_floor_met then
    fail
      (Printf.sprintf
         "the repository's derivation finds %d focused-aggregate member%s, against a floor of %d -- \
          the derivation has stopped finding them, and a family whose membership is empty is \
          aggregated completely by an empty stanza"
         (List.length family_members)
         (if List.length family_members = 1 then "" else "s")
         family_member_floor);
  eprintf "Focused-aggregate members derived: %d, against a floor of %d.\n"
    (List.length family_members) family_member_floor;
  Verdict.p
    "every focused-aggregate member is reached by its family alias, and a repository-wide \
     derivation finds enough of them for the rule to be about something"
    (List.is_empty family_unaggregated && family_floor_met);
  eprintf
    "Artifact-directory verdict of every stanza whose modules call Test_utils.Generated.init, or \
     which declares %s without one (not diffed -- see gh-ocannl-665):\n\
     %s\n"
    Scan.artifact_env_var
    (String.concat ~sep:"\n" (List.rev !artifact_table));
  let unclaimed =
    List.filter artifact_callers ~f:(fun path ->
        not (Set.mem !artifact_claimed (String.lowercase path)))
  in
  List.iter unclaimed ~f:(fun path ->
      fail
        (Printf.sprintf
           "%s calls `Test_utils.Generated.init` and no stanza's `(modules ...)` claims it -- the \
            rule that would require `(env_var %s)` of it never reaches it. Name the module in the \
            stanza that builds it, or hand this scan that directory's dune file"
           path Scan.artifact_env_var));
  let floor_met =
    (not repository_census) || List.length artifact_callers >= artifact_caller_floor
  in
  if not floor_met then
    fail
      (Printf.sprintf
         "the repository's census finds %d source%s calling `Test_utils.Generated.init`, against a \
          floor of %d -- the census has stopped finding them, and a relationship checked over an \
          empty set holds for nothing"
         (List.length artifact_callers)
         (if List.length artifact_callers = 1 then "" else "s")
         artifact_caller_floor);
  eprintf "Sources calling Test_utils.Generated.init: %d, against a floor of %d.\n"
    (List.length artifact_callers) artifact_caller_floor;
  Verdict.p
    "every source that calls Test_utils.Generated.init is claimed by some stanza's modules, and a \
     repository-wide census finds enough of them for the rule to be about something"
    (List.is_empty unclaimed && floor_met);
  Verdict.p
    "every stanza whose modules call Test_utils.Generated.init declares OCANNL_BUILD_FILES_PREFIX \
     where dune runs it, and every declaration of it has a caller behind it"
    (!artifact_violations = 0);
  if not (Verdict.any_failed ()) then
    printf
      "\n\
       OK: every `(env_var ...)` addressed to OCANNL names a spelling a run reads, every test \
       directory carries the ambient gate, and every per-module tracing gate is declared by the \
       library whose modules read it.\n"

(* gh-ocannl-723's negative control, and why it runs the checker rather than inspecting it.

   A control written from today's corpus encodes the ABSENCE of the shape it is about: every stanza
   in this repository that calls `Test_utils.Generated.init` declares the variable, so a check that
   reported nothing and a check that decided nothing would both pass over it, and the second is what
   an unexercised rule quietly becomes. So the rule is put to a stanza/source pair the repository
   does not contain: a synthetic tree of four dune files and one source, handed to THIS executable
   in a child process, once with the declaration and once without.

   Everything but the one declaration is held fixed between the two runs, so the difference in
   verdict is the rule's and nothing else's. The tree is built to satisfy the file's other rules --
   the fixtures for the exemption and gateless lists are DERIVED from those lists, so they cannot
   drift from them -- which buys the sharper claim: the violating tree exits 1 and names the stanza,
   and the legitimate one exits 0. *)

let control_root_paths = [ "t/dune"; "t/probe.ml"; "t/nested/probe2.ml" ]
let control_probe = "let () = Test_utils.Generated.init ~backend_name:\"cc\"\n"

(* The subject: an `(executable)` whose one module calls the initializer, plus the rule that runs it
   -- an executable has no `deps` field, so the rule is where the declaration has to go, and putting
   the pair in the control is what keeps that placement checked. *)
let control_subject ~declares =
  Printf.sprintf
    {dune|(executable
 (name probe)
 (modules probe)
 (libraries test_utils))

(rule
 ; ocannl-backend: none -- a synthetic control fixture, which runs on no device at all.
 (target probe.actual)
 (deps
  ocannl_config
%s  %%{dep:probe.exe})
 (action
  (with-stdout-to
   %%{target}
   (run ./probe.exe))))

; A second, always-correct pair inside a `(subdir …)`. It is not the pair under control -- it
; declares in both runs -- but it is what makes the LEGITIMATE run's exit status depend on the walk
; descending: a scan that only read the top level would leave `nested/probe2.ml` claimed by nobody,
; which the census check reports (Codex P2, round 3).
(subdir
 nested
 (executable
  (name probe2)
  (modules probe2)
  (libraries test_utils))
 (rule
  ; ocannl-backend: none -- the same fixture, one directory down.
  (target probe2.actual)
  (deps
   ocannl_config
   (env_var %s)
   %%{dep:probe2.exe})
  (action
   (with-stdout-to
    %%{target}
    (run ./probe2.exe)))))
|dune}
    (if declares then Printf.sprintf "  (env_var %s)\n" Scan.artifact_env_var else "")
    Scan.artifact_env_var

(* The rest of the tree exists only so that the two runs differ in ONE verdict. Both lists below are
   checked for staleness against the files that make them necessary, so a tree without those files
   fails for reasons that have nothing to do with the rule under control. *)
let control_context () =
  let by_file =
    List.map exempt_declarations ~f:(fun (key, _) ->
        match String.lsplit2 key ~on:':' with
        | Some (file, name) -> (file, name)
        | None -> (key, key))
    |> Map.of_alist_multi (module String)
  in
  let exempt_files =
    List.map (Map.to_alist by_file) ~f:(fun (file, names) ->
        let declarations =
          String.concat ~sep:"" (List.map names ~f:(Printf.sprintf "  (env_var %s)\n"))
        in
        ( file,
          "(rule\n (target exempt.fixture)\n (deps\n" ^ declarations
          ^ " )\n (action\n  (with-stdout-to\n   %{target}\n   (echo \"\"))))\n" ))
  in
  let gateless_files =
    List.map gateless_dirs ~f:(fun (file, _) ->
        ( file,
          Printf.sprintf "(test\n (name gateless)\n (deps\n  (env_var %s))\n (modules gateless))\n"
            Scan.backend_env_var ))
  in
  exempt_files @ gateless_files

let write_file path data =
  let dir = Stdlib.Filename.dirname path in
  let rec mkdirs dir =
    if not (String.equal dir Stdlib.Filename.current_dir_name || Stdlib.Sys.file_exists dir) then (
      mkdirs (Stdlib.Filename.dirname dir);
      try Unix.mkdir dir 0o755 with Unix.Unix_error (Unix.EEXIST, _, _) -> ())
  in
  mkdirs dir;
  Out_channel.write_all path ~data

(* The tree is this process's to remove: it was made by `Filename.temp_dir`, which creates a fresh
   directory nothing else holds. Removal is best effort -- a control that failed to tidy up is not a
   control that decided wrongly, and the temporary directory is the operating system's to reclaim
   either way. *)
let rec remove_tree path =
  match Unix.lstat path with
  | { Unix.st_kind = Unix.S_DIR; _ } ->
      Array.iter (Stdlib.Sys.readdir path) ~f:(fun entry ->
          remove_tree (Stdlib.Filename.concat path entry));
      Unix.rmdir path
  | _ -> Unix.unlink path
  | exception Unix.Unix_error _ -> ()

let describe_status = function
  | Unix.WEXITED n -> Printf.sprintf "exited %d" n
  | Unix.WSIGNALED n -> Printf.sprintf "was killed by signal %d" n
  | Unix.WSTOPPED n -> Printf.sprintf "was stopped by signal %d" n

(* Through temporary FILES rather than pipes, for the reason `generated_provenance` gives: the child
   writes to both streams, and reading two pipes in sequence deadlocks as soon as the one not being
   read fills its buffer. *)
let run_checker ~root ~exe args =
  let capture suffix = Stdlib.Filename.temp_file "evd_control" suffix in
  let out_path = capture ".out" and err_path = capture ".err" in
  let open_capture p = Unix.openfile p [ Unix.O_WRONLY; Unix.O_TRUNC ] 0o600 in
  let out = open_capture out_path and err = open_capture err_path in
  let here = Stdlib.Sys.getcwd () in
  Stdlib.Sys.chdir root;
  let pid = Unix.create_process exe (Array.of_list (exe :: args)) Unix.stdin out err in
  let _, status = Unix.waitpid [] pid in
  Stdlib.Sys.chdir here;
  Unix.close out;
  Unix.close err;
  let text = In_channel.read_all out_path ^ In_channel.read_all err_path in
  (try Unix.unlink out_path with Unix.Unix_error _ -> ());
  (try Unix.unlink err_path with Unix.Unix_error _ -> ());
  (status, text)

let control () =
  (* Absolute before the chdir: `Sys.executable_name` resolves the name this process was started
     with, and a relative one would name nothing from inside the temporary tree. *)
  let exe =
    let name = Stdlib.Sys.executable_name in
    if Stdlib.Filename.is_relative name then Stdlib.Filename.concat (Stdlib.Sys.getcwd ()) name
    else name
  in
  let root = Stdlib.Filename.temp_dir "evd_control" "" in
  let context = control_context () in
  List.iter context ~f:(fun (file, content) ->
      write_file (Stdlib.Filename.concat root file) content);
  write_file (Stdlib.Filename.concat root "t/probe.ml") control_probe;
  write_file (Stdlib.Filename.concat root "t/nested/probe2.ml") control_probe;
  let paths = control_root_paths @ List.map context ~f:fst in
  let run ~declares =
    write_file (Stdlib.Filename.concat root "t/dune") (control_subject ~declares);
    run_checker ~root ~exe ("." :: paths)
  in
  (* The exact sentence the rule produces, so that the control observes THIS rule failing and not
     merely the child's misfortune -- the argument gh-ocannl-692 made for `generated_provenance`. *)
  let diagnostic = "calls `Test_utils.Generated.init`" in
  let report label (status, text) =
    eprintf "the control's %s run %s. Its captured output:\n%s\n" label (describe_status status)
      text
  in
  let violating = run ~declares:false in
  let legitimate = run ~declares:true in
  let violating_reported =
    (match fst violating with Unix.WEXITED 1 -> true | _ -> false)
    && String.is_substring (snd violating) ~substring:diagnostic
    && String.is_substring (snd violating) ~substring:Scan.artifact_env_var
  in
  let legitimate_passed =
    (match fst legitimate with Unix.WEXITED 0 -> true | _ -> false)
    && not (String.is_substring (snd legitimate) ~substring:diagnostic)
  in
  if not violating_reported then report "violating" violating;
  if not legitimate_passed then report "legitimate" legitimate;
  printf
    "The rule is put to a stanza this repository does not contain: an `(executable)` whose one\n\
     module calls `Test_utils.Generated.init`, run by a rule that does or does not declare\n\
     `(env_var %s)`. Nothing else differs between the two runs.\n\n"
    Scan.artifact_env_var;
  Verdict.p
    "the checker reports the stanza and exits 1 when the rule running it omits the declaration"
    violating_reported;
  Verdict.p "the same tree with the declaration added passes and says nothing about it"
    legitimate_passed;
  try remove_tree root with Unix.Unix_error _ -> ()

(* gh-ocannl-708's control, and why it is a second tree rather than a stanza added to the one above.

   The rule under control here is the RELATIONSHIP between the two readers: a stanza the walk places
   a site for is a stanza the raw-text floor names. This repository contains exactly one stanza of
   the shape that used to break it -- `benchmarks/dune` running its orchestrator as `(run python3
   %{dep:test_orchestrate.py})` -- so a control read off today's corpus would pass whether the floor
   learned the shape or the shape left the repository. Put to a tree of its own, the claim is about
   the rule: the same tool handed a file this workspace builds is seen by both readers, and handed
   nothing of ours is seen by neither. *)

let floor_subject ~handed ~declares =
  Printf.sprintf
    {dune|(rule
%s (target orchestrated.actual)
 (deps ocannl_config %s)
 (action
  (with-stdout-to
   %%{target}
   (run python3 %s))))
|dune}
    (if declares then
       " ; ocannl-backend: none -- hands a script to python3, which links no backend.\n"
     else "")
    (if handed then "%{dep:orchestrate.py}" else "orchestrate.py")
    (if handed then "%{dep:orchestrate.py}" else "orchestrate.py")

let floor_control () =
  let exe =
    let name = Stdlib.Sys.executable_name in
    if Stdlib.Filename.is_relative name then Stdlib.Filename.concat (Stdlib.Sys.getcwd ()) name
    else name
  in
  let root = Stdlib.Filename.temp_dir "evd_floor" "" in
  let context = control_context () in
  List.iter context ~f:(fun (file, content) ->
      write_file (Stdlib.Filename.concat root file) content);
  (* One source, so that the run is handed both a dune file and a source: the checker refuses a tree
     with neither, since its globs matching nothing is the failure it reports first. It calls
     nothing and reads no configuration key, which keeps the tree's only subject the rule above. *)
  write_file (Stdlib.Filename.concat root "t/noop.ml") "let () = ()\n";
  let paths = "t/dune" :: "t/noop.ml" :: List.map context ~f:fst in
  let run ~handed ~declares =
    write_file (Stdlib.Filename.concat root "t/dune") (floor_subject ~handed ~declares);
    run_checker ~root ~exe ("." :: paths)
  in
  let report label (status, text) =
    eprintf "the floor control's %s run %s. Its captured output:\n%s\n" label
      (describe_status status) text
  in
  let exited n (status, _) = match status with Unix.WEXITED m -> m = n | _ -> false in
  (* The rule reaches the stanza at all: without a marker or a declaration, the checker reports it
     by name. That is the walk's half. *)
  let reported = run ~handed:true ~declares:false in
  (* And the floor's half: with the marker, the run passes AND says every stanza it placed has a
     second reader's floor under it -- the sentence that named this stanza as the exception before
     the two readers shared the pform lists. *)
  let floored = run ~handed:true ~declares:true in
  (* The negative control. The same command handed nothing this workspace provides is a stanza
     NEITHER reader sees, so the rule does not apply and no marker is asked for. A floor that
     over-claimed here would fail this correct tree. *)
  let invisible = run ~handed:false ~declares:false in
  let unfloored_sentence = "standing on the walk alone" in
  let floored_sentence = "second reader's floor under it" in
  let reported_ok =
    exited 1 reported
    && String.is_substring (snd reported) ~substring:"runs an executable and declares neither"
    && String.is_substring (snd reported) ~substring:"t/dune"
  in
  let floored_ok =
    exited 0 floored
    && String.is_substring (snd floored) ~substring:floored_sentence
    && not (String.is_substring (snd floored) ~substring:unfloored_sentence)
  in
  let invisible_ok =
    exited 0 invisible
    && (not
          (String.is_substring (snd invisible) ~substring:"runs an executable and declares neither"))
    && not (String.is_substring (snd invisible) ~substring:unfloored_sentence)
  in
  if not reported_ok then report "reported" reported;
  if not floored_ok then report "floored" floored;
  if not invisible_ok then report "invisible" invisible;
  printf
    "\n\
     The relationship is put to a tree of one rule, which runs `python3` on a file that is or is\n\
     not one this workspace builds. Nothing else differs between the three runs.\n\n";
  Verdict.p
    "an external command handed a file this workspace builds is a stanza the rule reaches, \
     reported by name when it declares neither"
    reported_ok;
  Verdict.p "the same stanza, declaring its backend, passes with the raw-text floor naming it too"
    floored_ok;
  Verdict.p "the same command handed nothing of this workspace is a stanza neither reader sees"
    invisible_ok;
  try remove_tree root with Unix.Unix_error _ -> ()

(* gh-ocannl-783's control, and why it is a third tree.

   Every family in this repository is complete -- that is what the rule is for -- so a control read
   off today's corpus would pass whether the rule decides anything or not, which is the argument
   both controls above make. Put to a tree of its own, what is asserted is the rule: a stanza the
   derivation calls a member is reported when its family alias does not reach it, the same tree with
   the alias listing it passes, and a stanza the derivation calls no member is asked for nothing.

   Both derivations are exercised, because they are independent: the Metal one reads the stanza's
   backend marker, the lifecycle one reads its modules' sources, and a control that ran only the
   first would leave the second able to stop finding anything. *)

let family_gate =
  Printf.sprintf
    {dune|(test
 ; ocannl-backend: none -- the ambient gate of this synthetic tree; it runs on no device.
 (name gate)
 (modules gate)
 (deps ocannl_config (universe))
 (libraries base))

(rule
 ; ocannl-backend: none -- the same gate, on the alias the family stanza depends on.
 (alias runtest-gate)
 (deps ocannl_config (universe))
 (action
  (run %%{dep:gate.exe})))

(alias
 (name runtest)
 (deps (alias runtest-gate)))
|dune}

(* The three shapes a member stanza takes, because the derivation reads each of them differently and
   the differences are where it went wrong (Codex P2, round 1). A `(test)` carries its marker itself
   and dune generates its alias; an `(executable)` runs nothing, so its marker is REQUIRED to sit on
   the rule that runs it and the alias to list is that rule's; and a `(tests)` is several tests
   behind one stanza, each with an alias the family has to reach. *)
type family_shape =
  | Single_test  (** `(test (name probe) …)` *)
  | Exe_with_runner  (** `(executable (name probe) …)` plus the `(rule)` that runs it *)
  | Runner_elsewhere  (** the same pair, whose rule `(chdir nested …)` runs ANOTHER `probe.exe` *)
  | Plural_tests  (** `(tests (names probe probe2) …)` -- two units behind one stanza *)
  | Inline_library  (** `(library (name probelib) (inline_tests …))`, whose alias dune generates *)
  | Subdir_member  (** the member, and its family alias, inside a `(subdir child …)` group *)

let family_marker ~metal =
  Printf.sprintf
    " ; ocannl-backend: %s -- a synthetic control fixture, judged on this marker alone.\n"
    (if metal then "metal" else "cc")

let family_member_stanza ~shape ~metal =
  let marker = family_marker ~metal in
  match shape with
  | Single_test ->
      Printf.sprintf
        "(test\n%s (name probe)\n (modules probe)\n (deps ocannl_config)\n (libraries base))\n"
        marker
  | Plural_tests ->
      Printf.sprintf
        "(tests\n%s (names probe probe2)\n (modules probe probe2)\n (deps ocannl_config)\n \
         (libraries base))\n"
        marker
  | Inline_library ->
      Printf.sprintf
        "(library\n%s (name probelib)\n (modules probe)\n (inline_tests\n  (deps \
         ocannl_config))\n (libraries base))\n"
        marker
  | Subdir_member ->
      (* Dune lets a `(subdir …)` group carry its own alias stanza, and the recursive `@<family>`
         build from the root reaches it. The family stanza therefore goes INSIDE the group here,
         which is the arrangement the check used to reject. *)
      Printf.sprintf
        "(subdir\n child\n (test\n%s  (name probe)\n  (modules probe)\n  (deps ocannl_config)\n  \
         (libraries base))\n (alias\n  (name %s)\n  (deps (alias runtest-probe))))\n"
        (String.substr_replace_all marker ~pattern:" ; " ~with_:"  ; ")
        metal_family.family_alias
  | Exe_with_runner | Runner_elsewhere ->
      (* The runner depends on the gate's alias, since its own alias is a build entry point like any
         other -- the same reason the family stanzas below do. *)
      Printf.sprintf
        "(executable\n (name probe)\n (modules probe)\n (libraries base))\n\n(rule\n%s (alias \
         probe_run)\n (deps ocannl_config (alias runtest-gate) %%{dep:probe.exe})\n (action\n  \
         %s))\n"
        marker
        (match shape with
        | Runner_elsewhere -> "(chdir nested (run ./probe.exe))"
        | _ -> "(run ./probe.exe)")

(* Which aliases the family stanza lists, when there is one. `Every` is what a correct dune file
   writes; `First_only` lists one alias of a plural stanza, which is both the half-listed error and
   -- when only one of the two mains is a member -- the correct listing. *)
type family_listing = Every | First_only

let family_listed_aliases ~shape ~listing =
  match (shape, listing) with
  | Subdir_member, _ -> []
  | (Exe_with_runner | Runner_elsewhere), _ -> [ "probe_run" ]
  | Single_test, _ -> [ "runtest-probe" ]
  | Inline_library, _ -> [ "runtest-probelib" ]
  | Plural_tests, First_only -> [ "runtest-probe" ]
  | Plural_tests, Every -> [ "runtest-probe"; "runtest-probe2" ]

(* The subject: one member stanza which is, or is not, a member of the family named -- by its
   backend marker for `metal-codegen`, by what `probe.ml` names for `lifecycle` -- and, optionally,
   the family alias stanza that aggregates it. Everything else is held fixed across the runs. *)
let family_subject ~shape ~metal ~family ~listing =
  Printf.sprintf "%s\n%s%s" family_gate
    (family_member_stanza ~shape ~metal)
    (match (family, family_listed_aliases ~shape ~listing) with
    | None, _ | _, [] -> ""
    | Some family, aliases ->
        Printf.sprintf "\n(alias\n (name %s)\n (deps\n  (alias runtest-gate)\n%s))\n" family
          (String.concat ~sep:"\n" (List.map aliases ~f:(Printf.sprintf "  (alias %s)"))))

(* The lifecycle derivation reads the module's SOURCE, so the two spellings of `probe.ml` are what
   makes the stanza a member or not. Syntactically valid OCaml either way: the checker parses the
   sources it is handed for the variables they read, and a source it cannot read is one it reports
   rather than passes over. *)
let family_probe ~lifecycle =
  if lifecycle then "let () = ignore (Ir.Alloc_census.snapshot ())\n" else "let () = ()\n"

let family_control () =
  let exe =
    let name = Stdlib.Sys.executable_name in
    if Stdlib.Filename.is_relative name then Stdlib.Filename.concat (Stdlib.Sys.getcwd ()) name
    else name
  in
  let root = Stdlib.Filename.temp_dir "evd_family" "" in
  let context = control_context () in
  List.iter context ~f:(fun (file, content) ->
      write_file (Stdlib.Filename.concat root file) content);
  write_file (Stdlib.Filename.concat root "t/gate.ml") "let () = ()\n";
  (* The plural shape's second module. Written once and always present: which stanza CLAIMS it is
     what differs between the shapes, and an unclaimed source is not itself a finding here. *)
  write_file (Stdlib.Filename.concat root "t/probe2.ml") "let () = ()\n";
  let paths =
    "t/dune" :: "t/gate.ml" :: "t/probe.ml" :: "t/probe2.ml" :: List.map context ~f:fst
  in
  let run ?(shape = Single_test) ?(listing = Every) ~metal ~lifecycle ~family () =
    write_file (Stdlib.Filename.concat root "t/dune")
      (family_subject ~shape ~metal ~family ~listing);
    write_file (Stdlib.Filename.concat root "t/probe.ml") (family_probe ~lifecycle);
    run_checker ~root ~exe ("." :: paths)
  in
  let report label (status, text) =
    eprintf "the family control's %s run %s. Its captured output:\n%s\n" label
      (describe_status status) text
  in
  let exited n (status, _) = match status with Unix.WEXITED m -> m = n | _ -> false in
  let unreached = "alias does not reach" in
  let metal = Some metal_family.family_alias and lifecycle = Some lifecycle_family.family_alias in
  let metal_omitted = run ~metal:true ~lifecycle:false ~family:None () in
  let metal_listed = run ~metal:true ~lifecycle:false ~family:metal () in
  let lifecycle_omitted = run ~metal:false ~lifecycle:true ~family:None () in
  let lifecycle_listed = run ~metal:false ~lifecycle:true ~family:lifecycle () in
  (* The marker's placement on an `(executable)` is the RULE that runs it, so the derivation has to
     read it there and the alias to list is the rule's. *)
  let runner_omitted = run ~shape:Exe_with_runner ~metal:true ~lifecycle:false ~family:None () in
  let runner_listed = run ~shape:Exe_with_runner ~metal:true ~lifecycle:false ~family:metal () in
  (* A plural stanza is several units: a marker covers both of them, so listing one alias leaves the
     other out of the family -- while the lifecycle derivation belongs to whichever main actually
     names the instrumentation, so listing that one alias is COMPLETE. Same stanza, same listing,
     opposite verdicts: what differs is which derivation put it in the family. *)
  let plural_half =
    run ~shape:Plural_tests ~listing:First_only ~metal:true ~lifecycle:false ~family:metal ()
  in
  let plural_whole = run ~shape:Plural_tests ~metal:true ~lifecycle:false ~family:metal () in
  let plural_one_main =
    run ~shape:Plural_tests ~listing:First_only ~metal:false ~lifecycle:true ~family:lifecycle ()
  in
  (* An inline-test library's alias is generated too, so a Metal-marked one reaches its family
     through `runtest-<library-name>`. *)
  let library_omitted = run ~shape:Inline_library ~metal:true ~lifecycle:false ~family:None () in
  let library_listed = run ~shape:Inline_library ~metal:true ~lifecycle:false ~family:metal () in
  (* A `(subdir …)` group carrying its own family alias is correctly wired, and the recursive build
     from the root reaches it. *)
  let subdir_listed = run ~shape:Subdir_member ~metal:true ~lifecycle:false ~family:None () in
  (* And the runner that runs SOMEONE ELSE's executable: listing its alias does not put this
     directory's lifecycle probe in the family, however alike the two basenames are. *)
  let runner_elsewhere =
    run ~shape:Runner_elsewhere ~metal:false ~lifecycle:true ~family:lifecycle ()
  in
  let runner_here = run ~shape:Exe_with_runner ~metal:false ~lifecycle:true ~family:lifecycle () in
  (* The negative control: neither derivation calls this stanza a member, so no family alias is
     asked for. A derivation that over-claimed would fail this correct tree. *)
  let no_member = run ~metal:false ~lifecycle:false ~family:None () in
  let omitted_ok family (result : Unix.process_status * string) =
    exited 1 result
    && String.is_substring (snd result) ~substring:unreached
    && String.is_substring (snd result) ~substring:family
    && String.is_substring (snd result) ~substring:"t/dune"
  in
  let listed_ok result =
    exited 0 result && not (String.is_substring (snd result) ~substring:unreached)
  in
  let metal_omitted_ok = omitted_ok metal_family.family_alias metal_omitted in
  let lifecycle_omitted_ok = omitted_ok lifecycle_family.family_alias lifecycle_omitted in
  let runner_omitted_ok = omitted_ok metal_family.family_alias runner_omitted in
  let plural_half_ok =
    omitted_ok metal_family.family_alias plural_half
    && String.is_substring (snd plural_half) ~substring:"runtest-probe2"
  in
  let metal_listed_ok = listed_ok metal_listed in
  let lifecycle_listed_ok = listed_ok lifecycle_listed in
  let runner_listed_ok = listed_ok runner_listed in
  let plural_whole_ok = listed_ok plural_whole in
  let plural_one_main_ok = listed_ok plural_one_main in
  let library_omitted_ok = omitted_ok metal_family.family_alias library_omitted in
  let library_listed_ok = listed_ok library_listed in
  let subdir_listed_ok = listed_ok subdir_listed in
  let runner_elsewhere_ok = omitted_ok lifecycle_family.family_alias runner_elsewhere in
  let runner_here_ok = listed_ok runner_here in
  let no_member_ok = listed_ok no_member in
  if not metal_omitted_ok then report "metal, family stanza omitted" metal_omitted;
  if not metal_listed_ok then report "metal, family stanza listing it" metal_listed;
  if not lifecycle_omitted_ok then report "lifecycle, family stanza omitted" lifecycle_omitted;
  if not lifecycle_listed_ok then report "lifecycle, family stanza listing it" lifecycle_listed;
  if not runner_omitted_ok then report "executable + runner, family stanza omitted" runner_omitted;
  if not runner_listed_ok then report "executable + runner, runner listed" runner_listed;
  if not plural_half_ok then report "plural stanza, one alias listed" plural_half;
  if not plural_whole_ok then report "plural stanza, both aliases listed" plural_whole;
  if not plural_one_main_ok then report "plural stanza, one main is the probe" plural_one_main;
  if not library_omitted_ok then report "inline-test library, family stanza omitted" library_omitted;
  if not library_listed_ok then report "inline-test library, generated alias listed" library_listed;
  if not subdir_listed_ok then report "member and family alias inside a subdir group" subdir_listed;
  if not runner_elsewhere_ok then report "runner running another directory's exe" runner_elsewhere;
  if not runner_here_ok then report "runner running this directory's exe" runner_here;
  if not no_member_ok then report "neither derivation's member" no_member;
  printf
    "\n\
     The focused-aggregate rule is put to a tree of one member stanza and a family alias that does\n\
     or does not list it. Nothing else differs between the runs of a pair; the member takes each of\n\
     the shapes dune builds a member in -- a test, an executable with its runner, a plural stanza,\n\
     an inline-test library, a `(subdir …)` group -- and the last run is a stanza neither\n\
     derivation calls a member.\n\n";
  Verdict.p
    "a stanza whose backend marker names metal is reported, naming its family, when the \
     metal-codegen alias does not reach it"
    metal_omitted_ok;
  Verdict.p "the same tree with the member listed in the family stanza passes" metal_listed_ok;
  Verdict.p
    "a stanza whose modules read the resource-lifecycle instrumentation is reported the same way \
     when the lifecycle alias does not reach it"
    lifecycle_omitted_ok;
  Verdict.p "the same tree with that member listed passes too" lifecycle_listed_ok;
  Verdict.p
    "an executable whose RUNNER carries the metal marker is a member too, reported when the family \
     does not reach that rule"
    runner_omitted_ok;
  Verdict.p "the same tree with the runner's own alias listed passes" runner_listed_ok;
  Verdict.p
    "a plural stanza with only one of its two generated aliases listed is reported, naming the one \
     still missing"
    plural_half_ok;
  Verdict.p "the same plural stanza with both listed passes" plural_whole_ok;
  Verdict.p
    "the same one-alias listing is COMPLETE when only that main reads the instrumentation, so the \
     lifecycle family is not asked for its neighbour"
    plural_one_main_ok;
  Verdict.p
    "an inline-test library carrying the metal marker is a member, reported when the family does \
     not reach the alias dune generates for it"
    library_omitted_ok;
  Verdict.p "the same library listed under its generated alias passes" library_listed_ok;
  Verdict.p
    "a member inside a `(subdir …)` group is aggregated by a family alias in that same group, \
     which the recursive build from the root reaches"
    subdir_listed_ok;
  Verdict.p
    "a runner that runs another directory's executable of the same name does not aggregate this \
     directory's member"
    runner_elsewhere_ok;
  Verdict.p "the same runner running this directory's executable does" runner_here_ok;
  Verdict.p "a stanza neither derivation calls a member is asked for no family alias" no_member_ok;
  try remove_tree root with Unix.Unix_error _ -> ()

let () =
  match Array.to_list Stdlib.Sys.argv with
  | _ :: [ "--control" ] ->
      control ();
      floor_control ();
      family_control ()
  | _ -> main ()
