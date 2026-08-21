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
   of the hole it was written to prevent.

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

(* The aliases that RUN things, and so can serve a cached result. Asked per alias rather than per
   directory (Codex P1 round 3): a `(test)` stanza contributes to `runtest` alone, so a directory
   whose gate is a test stanza is ungated for `@slow` -- which is a separately documented entry
   point, and which `dune build @slow` reaches without building a single `(test)`. A gate on one
   alias must not vouch for another. *)
let run_aliases = [ "runtest"; "slow" ]

let rec names_alias alias = function
  | Sexp.List [ Sexp.Atom "alias"; Sexp.Atom n ] -> String.equal n alias
  | Sexp.List l -> List.exists l ~f:(names_alias alias)
  | Sexp.Atom _ -> false

(* Which of {!run_aliases} this stanza attaches to. A `(test)`/`(tests)` stanza is dune's shorthand
   for a runtest action and names no alias of its own. *)
let aliases_of stanza =
  match stanza with
  | Sexp.List (Sexp.Atom ("test" | "tests") :: _) -> [ "runtest" ]
  | Sexp.List (Sexp.Atom "rule" :: _) ->
      List.filter run_aliases ~f:(fun alias -> names_alias alias stanza)
  | _ -> []

(* The prefix `Utils.classify_env_var` reports for a per-module tracing gate. *)
let gate_prefix = "ocannl_log_level_"

(* The stanza kinds that name their own modules, and so can be asked what those modules read. *)
let module_stanzas = [ "library"; "test"; "tests"; "executable"; "executables" ]

let () =
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
  let sources =
    List.filter paths ~f:(fun (path, _) -> String.is_suffix path ~suffix:".ml")
    |> List.map ~f:(fun (path, on_disk) -> (String.lowercase path, on_disk))
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
  let placed_subjects = ref 0 in
  let subject_floor = ref 0 in
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
          let markers =
            List.filter_map stanza.Scan.marked_comments ~f:(fun (line, text) ->
                Option.map (Scan.parse_marker text) ~f:(fun marker -> (line, text, marker)))
          in
          List.iter markers ~f:(fun (line, _, _) -> attributed := line :: !attributed);
          let subject = not (List.is_empty stanza.Scan.marked_sites) in
          let declares = stanza.Scan.marked_declares_backend in
          List.iter markers ~f:(function
            | line, text, Scan.Malformed why ->
                Int.incr xor_violations;
                fail
                  (Printf.sprintf
                     "%s:%d has a `%s` comment that does not parse as a marker: %s. The line reads \
                      `;%s`"
                     dune_file line Scan.marker_sentinel why text)
            | _ -> ());
          let well_formed =
            List.filter_map markers ~f:(function
              | line, _, Scan.Marker m -> Some (line, m)
              | _, _, Scan.Malformed _ -> None)
          in
          match (subject, declares, well_formed) with
          (* A marker on a stanza that runs nothing declares nothing. An `(executable)` has no
             `deps` field at all, which is why its companion rule is where both the `ocannl_config`
             dep and this marker go -- putting it on the executable reads as a declaration and is
             not one. *)
          | false, _, (line, _) :: _ ->
              Int.incr xor_violations;
              fail
                (Printf.sprintf
                   "%s carries a backend marker at line %d and runs no executable -- the marker \
                    belongs on the stanza that RUNS it, which for an `(executable)` is its \
                    companion rule, the same placement as the `%s` dep"
                   where line Scan.config_file)
          | false, _, [] -> ()
          | true, _, _ :: (line, _) :: _ ->
              Int.incr xor_violations;
              fail
                (Printf.sprintf
                   "%s carries more than one backend marker (the second at line %d) -- one stanza \
                    runs on one backend, so say so once"
                   where line)
          | true, true, [ (line, m) ] ->
              Int.incr xor_violations;
              fail
                (Printf.sprintf
                   "%s both declares `(env_var %s)` and carries a marker at line %d saying `%s` -- \
                    those are contradictory: a stanza that names its backend has nothing for the \
                    variable to invalidate, and a stanza that selects one has no business claiming \
                    otherwise. Keep whichever is true"
                   where Scan.backend_env_var line m.Scan.backend)
          | true, true, [] ->
              Int.incr placed_subjects;
              any_declared := true;
              classification :=
                Printf.sprintf "  %-58s declares %s" where Scan.backend_env_var :: !classification
          | true, false, [ (_, m) ] ->
              Int.incr placed_subjects;
              List.iter (String.split m.Scan.backend ~on:',') ~f:(fun word ->
                  words := Set.add !words word);
              classification :=
                Printf.sprintf "  %-58s %s -- %s" where m.Scan.backend m.Scan.reason
                :: !classification
          | true, false, [] ->
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
      (* The floor under the walk, read by the second reader that shares none of its machinery: a
         stanza it stops seeing is a stanza the rule above stops applying to, which looks exactly
         like a file with nothing to check (the gh-ocannl-665 argument, and config_dep_completeness'
         floors).

         The predicate has to admit EVERY kind of subject `sites_of_stanza` places, or the floor is
         blind to exactly the classes it omits: a regression in one of them would drop those stanzas
         out of `marked_sites` while leaving this count unchanged, and the blindness check would
         stay green over a class that had stopped being enforced (Codex P2, round 1). Beyond the
         `run` actions and a test stanza's own binary, that means an `(inline_tests)` library --
         which `sites_of_stanza` reports as an `Inline_tests` site -- and a bare command under a
         rewritten `PATH`, which it reports as a `path_rewritten` site precisely because it refuses
         to name the program. *)
      let raw_subjects =
        List.count (Scan.raw_stanzas content) ~f:(fun r ->
            (not (List.is_empty r.Scan.raw_runs))
            || (not (List.is_empty r.Scan.raw_test_cwds))
            || r.Scan.raw_inline_tests
            || not (List.is_empty r.Scan.raw_unnameable))
      in
      let placed_here = List.count marked ~f:(fun s -> not (List.is_empty s.Scan.marked_sites)) in
      subject_floor := !subject_floor + raw_subjects;
      if placed_here < raw_subjects then (
        Int.incr marker_holes;
        fail
          (Printf.sprintf
             "%s: the raw text shows %d %s running an executable and the walk placed only %d -- it \
              is reading the file with a hole in it, and a stanza it stops seeing is one this rule \
              stops applying to"
             dune_file raw_subjects
             (if raw_subjects = 1 then "stanza" else "stanzas")
             placed_here));
      by_file :=
        ( dune_file,
          (if !any_declared then [ "declares " ^ Scan.backend_env_var ] else [])
          @ (match Set.to_list !words with
            | [] -> []
            | words -> [ "markers: " ^ String.concat ~sep:", " words ]) )
        :: !by_file;
      (* The ambient gate, per directory AND per alias (gh-ocannl-652). *)
      List.iter run_aliases ~f:(fun alias ->
          let attaches stanza = List.mem (aliases_of stanza) alias ~equal:String.equal in
          if List.exists stanzas ~f:attaches then
            if List.exists stanzas ~f:(fun s -> attaches s && depends_on_universe s) then (
              gated := (dune_file, alias) :: !gated;
              if
                List.exists stanzas ~f:takes_training_lock
                && not
                     (List.exists stanzas ~f:(fun s ->
                          attaches s && depends_on_universe s && takes_training_lock s))
              then
                fail
                  (Printf.sprintf
                     "%s serializes its actions on `%s` and its `%s` gate does not take the lock \
                      -- the one unlocked action in a file of locked ones is what the next \
                      training test gets copied from; add `(locks %s)` to it"
                     dune_file training_lock alias training_lock))
            else if Map.mem gateless dune_file then
              gateless_used := Set.add !gateless_used dune_file
            else
              fail
                (Printf.sprintf
                   "%s has actions on the `%s` alias and no ambient gate attached to it -- nothing \
                    here declares a rejected environment spelling, so `ocannl_backend=cuda dune \
                    build @%s` would serve this directory's cached results with the fatal startup \
                    check never reached; copy the `env_spelling_gate` stanza for that alias from a \
                    neighbour, or exempt the directory by name with the reason"
                   dune_file alias
                   (if String.equal alias "slow" then "slow"
                    else Stdlib.Filename.dirname dune_file ^ "/" ^ alias)));
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
  printf "\nAmbient environment gates, by dune file and the alias each is attached to:\n";
  List.sort !gated ~compare:(fun (a, x) (b, y) ->
      match String.compare a b with 0 -> String.compare x y | c -> c)
  |> List.iter ~f:(fun (dune_file, alias) -> printf "  %-40s @%s\n" dune_file alias);
  List.iter gateless_dirs ~f:(fun (dir, why) -> printf "  %s -- no gate: %s\n" dir why);
  printf "\nDeclarations of a name OCANNL does not read as a configuration key, exempt by design:\n";
  List.iter exempt_declarations ~f:(fun (key, why) -> printf "  %s -- %s\n" key why);
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
  eprintf
    "Backend classification of every stanza that runs an executable (not diffed -- see \
     gh-ocannl-665):\n\
     %s\n"
    (String.concat ~sep:"\n" (List.rev !classification));
  eprintf "Totals: %d such stanzas, against a raw-text floor of %d.\n" !placed_subjects
    !subject_floor;
  printf "\n";
  Verdict.p
    "every stanza that runs an executable either declares the backend variable or says in place \
     why it does not"
    (!xor_violations = 0);
  Verdict.p
    "every marker the text spells was read as one, and the walk places at least as many stanzas as \
     a second reader finds"
    (!marker_holes = 0 && !placed_subjects >= !subject_floor && !subject_floor > 0);
  if not (Verdict.any_failed ()) then
    printf
      "\n\
       OK: every `(env_var ...)` addressed to OCANNL names a spelling a run reads, every test \
       directory carries the ambient gate, and every per-module tracing gate is declared by the \
       library whose modules read it.\n"
