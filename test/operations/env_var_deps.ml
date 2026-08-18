(* gh-ocannl-628: the ambient variables a dune rule declares, against the ones a run reads.

   A dune rule is invalidated by an environment variable only if it says so. Where that matters is
   everywhere: 213 stanzas in this repository declare `(env_var OCANNL_BACKEND)` so that changing
   the backend re-runs the test rather than serving the previous backend's output as a pass.

   Two ways that goes wrong, both silent, both checked here.

   {1 One spelling of two}

   `Utils.read_env_var` consults the LOWERCASE spelling FIRST: `ocannl_backend=cuda` outranks
   `OCANNL_BACKEND` and decides which backend every test compiles and runs on. Before gh-ocannl-628
   not one dune file in the repository declared it -- all 213 declared the uppercase spelling
   alone, the one that loses -- so a developer who exported the lowercase form got stale targets
   served as passes, from rules written precisely to prevent that. A sweep fixes the 213; only a
   check keeps the 214th, copied next month from a neighbour, from reintroducing the hole.

   {2 Gates the build does not see}

   The other direction: a variable that IS read and is declared nowhere. ppx_minidebug's per-module
   tracing gates (`OCANNL_LOG_LEVEL_ROW` and eighteen siblings) are read while PREPROCESSING, so a
   library that does not declare them hands back the modules it built with the trace statements
   stripped -- `OCANNL_LOG_LEVEL_ROW=9 dune build` returning a silent binary, which reads as "the
   trace shows nothing" rather than as "the trace was never compiled in". Each gate is checked
   against the library whose modules read it.

   {1 What decides "addressed to the configuration"}

   `Utils.classify_env_var`, the same function the startup warning uses (gh-ocannl-629), so a name
   a rule tracks and a name a run warns about cannot be classified two ways. It also supplies the
   reserved namespaces: `OCANNL_TOOL_...` is the tooling's and has no second spelling to pair with,
   and `OCANNL_LOG_LEVEL_<MODULE>` is a gate, checked by the other half of this test. *)

open Base
open Stdio
module Scan = Test_utils.Dune_stanza_scan
module Sources = Test_utils.Config_key_scan

(* Declarations of a name OCANNL does not read as a configuration key. Keyed by
   "<dune file>:<name>", and each entry earns its place on every run (see the staleness check
   below): a rule tracking a variable no key would be read from is normally a typo, which is the
   whole point of asking. *)
let exempt_declarations =
  [
    ( "test/operations/dune:ocannl_backedn",
      "the fixture behind the `config_var_warnings` golden, which captures the warning a mistyped \
       key draws; the rule tracks the name so that an ambient one arriving does not leave the \
       golden stale" );
    ("test/operations/dune:OCANNL_BACKEDN", "the same fixture, in the other spelling");
    ( "test/operations/dune:ocannl-log_level",
      "the same fixture's second case: a known key in the dashed spelling that gh-ocannl-605 \
       dropped, which draws the other of the two warnings" );
  ]

(* The prefix `Utils.classify_env_var` reports for a per-module tracing gate. *)
let gate_prefix = "ocannl_log_level_"

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
  let tracked_keys = ref (Set.empty (module String)) in
  let gate_table = ref [] in
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
      let stanzas = Scan.stanzas (In_channel.read_all on_disk) in
      let fields = List.concat_map stanzas ~f:dep_fields in
      let declared = List.concat_map fields ~f:(fun (_, args) -> List.concat_map args ~f:env_vars_in) in
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
      List.iter fields ~f:(fun (field, args) ->
          let names = List.concat_map args ~f:env_vars_in in
          List.iter names ~f:(fun name ->
              let key = dune_file ^ ":" ^ name in
              match Utils.classify_env_var name with
              (* Someone else's variable entirely (`PATH`, `HOME`): not this check's business. *)
              | Utils.Env_not_addressed -> ()
              | Utils.Env_config_key config_key ->
                  tracked_keys := Set.add !tracked_keys config_key;
                  List.iter (Utils.env_var_names config_key) ~f:(fun spelling ->
                      if not (List.mem names spelling ~equal:String.equal) then
                        fail
                          (Printf.sprintf
                             "%s declares `(env_var %s)` without `(env_var %s)` in the same `%s` \
                              field -- `Utils.read_env_var` reads both spellings of %s, and the \
                              undeclared one invalidates nothing"
                             dune_file name spelling field config_key))
              (* A reserved namespace has no second spelling to pair with. The gates are checked
                 below, against the modules that read them. *)
              | Utils.Env_reserved _ -> ()
              | Utils.Env_unread_spelling _ | Utils.Env_unknown_key _
              | Utils.Env_unread_reserved _ ->
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
      (* The gates: what a library's modules read while being preprocessed, against what its
         `preprocessor_deps` declares. *)
      List.iter stanzas ~f:(fun stanza ->
          match (Scan.head stanza, Scan.field stanza "modules") with
          | Some "library", Some modules ->
              let library =
                match Scan.names_of stanza with name :: _ -> name | [] -> "<unnamed>"
              in
              let where = Printf.sprintf "%s, library %s" dune_file library in
              let declared_gates =
                match Scan.field stanza "preprocessor_deps" with
                | None -> []
                | Some args ->
                    List.concat_map args ~f:env_vars_in
                    |> List.filter ~f:(fun name ->
                           match Utils.classify_env_var name with
                           | Utils.Env_reserved prefix -> String.equal prefix gate_prefix
                           | _ -> false)
              in
              let modules =
                List.filter_map modules ~f:(function Sexp.Atom m -> Some m | _ -> None)
              in
              let read_gates =
                List.concat_map modules ~f:(fun module_name ->
                    match source_of ~dir module_name with
                    | None -> []
                    | Some on_disk ->
                        Sources.tracing_gates_in_source (In_channel.read_all on_disk)
                        |> List.map ~f:(fun gate -> (gate, module_name ^ ".ml")))
              in
              if (not (List.is_empty declared_gates)) && not (Set.mem scanned_dirs dir) then
                fail
                  (Printf.sprintf
                     "%s declares tracing gates, and this check was handed no sources from %s to \
                      check them against -- add the directory to the rule's globs"
                     where (if String.is_empty dir then "the repository root" else dir))
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
                           where gate)))
          | _ -> ()));
  let stale =
    Set.diff
      (Set.of_list (module String) (List.map exempt_declarations ~f:fst))
      !exemptions_used
  in
  if not (Set.is_empty stale) then
    fail
      (Printf.sprintf
         "exempted declarations no dune file makes any more -- drop them from the exemption list: \
          %s"
         (String.concat ~sep:", " (Set.to_list stale)));
  (* The coverage of the gate half, stated rather than assumed: a library outside these
     directories has no sources here to check its `preprocessor_deps` against, and says so above
     if it declares a gate at all -- but a gate it FAILS to declare is invisible from here, so the
     boundary belongs in the golden where widening it is a reviewable diff. *)
  printf "Directories whose sources this scan reads:\n";
  Set.iter scanned_dirs ~f:(fun dir ->
      printf "  %s\n" (if String.is_empty dir then "." else dir));
  printf "\nConfiguration keys tracked as ambient dependencies, in both spellings everywhere:\n";
  Set.iter !tracked_keys ~f:(printf "  %s\n");
  printf "\nPer-module tracing gates, and the library whose preprocessor_deps declares each:\n";
  List.sort !gate_table ~compare:(fun (_, a, _) (_, b, _) -> String.compare a b)
  |> List.iter ~f:(fun (where, gate, source) -> printf "  %-30s %s (%s)\n" gate where source);
  printf "\nDeclarations of a name OCANNL does not read as a configuration key, exempt by design:\n";
  List.iter exempt_declarations ~f:(fun (key, why) -> printf "  %s -- %s\n" key why);
  if not (Verdict.any_failed ()) then
    printf
      "\n\
       OK: every `(env_var ...)` naming a configuration key declares both spellings in the same \
       field, and every per-module tracing gate is declared by the library whose modules read it.\n"
