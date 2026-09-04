(** gh-ocannl-874: every public executable declared in [bin/dune] is run exactly once by
    [@bin-smoke].

    The public declarations and the smoke action are independent lists in one dune file. A new
    executable therefore compiles under [@check] even when its runtime canary is accidentally
    omitted. This scan relates those two lists from the parsed dune structure. It accepts either the
    executable's local [%{exe:...}] identity or its [%{bin:...}] public identity, and requires exact
    membership so a stale or duplicated smoke command is visible too. Every repository dune file is
    inspected because [@bin-smoke] is recursive from the workspace root. Conditional executable or
    smoke stanzas, opaque smoke actions, and unexpanded [(include ...)] forms are refused: without
    evaluating or expanding them, a syntactic census could claim commands CI does not execute or
    miss additional commands that it does.

    [--negative-control] applies the production scan to a synthetic dune file with two public
    executables and only one smoke command. Its dune rule accepts exit 1 and no other status, so a
    checker that stops detecting the omission makes the control fail rather than go green. *)

open Base
open Stdio
open Verdict.Claims
module Dune_scan = Test_utils.Dune_stanza_scan

type declaration = { local : string; public : string }
type smoke_target = Local of string | Public of string

type result = {
  declarations : declaration list;
  smoked : string list;
  missing : string list;
  unexpected : string list;
  errors : string list;
}

type alias_node = {
  id : string;
  dune_path : string;
  subdir : string;
  stanza : Sexp.t;
  aliases : string list;
  is_bin_smoke : bool;
  dependencies : string list;
  target_dependencies : string list;
  dependency_errors : string list;
}

let sorted = List.sort ~compare:String.compare
let verified_helper_alias = "bin/bin-smoke-env_spelling_gate"
let verified_helper_local = "bin/env_spelling_gate.exe"
let verified_helper_command = "%{dep:env_spelling_gate.exe}"

let path_dirname path =
  match String.rsplit2 path ~on:'/' with Some (directory, _) -> directory | None -> ""

let path_basename path =
  match String.rsplit2 path ~on:'/' with Some (_, basename) -> basename | None -> path

let duplicates strings =
  List.sort_and_group strings ~compare:String.compare
  |> List.filter_map ~f:(function first :: _ :: _ -> Some first | _ -> None)

let aliases_of stanza =
  match Dune_scan.head stanza with
  | Some "alias" -> (
      match Dune_scan.field stanza "name" with Some [ Sexp.Atom name ] -> [ name ] | _ -> [])
  | _ ->
      List.concat_map [ "alias"; "aliases" ] ~f:(fun field ->
          match Dune_scan.field stanza field with
          | None -> []
          | Some args ->
              List.filter_map args ~f:(function Sexp.Atom name -> Some name | _ -> None))

let has_enabled_if stanza = Option.is_some (Dune_scan.field stanza "enabled_if")
let alias_key ~subdir name = Dune_scan.normalize_path (Dune_scan.in_subdir subdir name)

let alias_dependencies ~subdir stanza =
  let rec collect = function
    | Sexp.List [ Sexp.Atom "alias"; Sexp.Atom name ] -> ([ alias_key ~subdir name ], [])
    | Sexp.List (Sexp.Atom "alias_rec" :: _) ->
        ([], [ "@bin-smoke reaches alias_rec, whose recursive dependency cannot be derived" ])
    | Sexp.List children ->
        List.fold children ~init:([], []) ~f:(fun (dependencies, errors) child ->
            let found, found_errors = collect child in
            (List.rev_append found dependencies, List.rev_append found_errors errors))
    | Sexp.Atom _ -> ([], [])
  in
  match Dune_scan.field stanza "deps" with
  | None -> ([], [])
  | Some deps -> collect (Sexp.List deps)

let dependency_path ~subdir atom =
  match Dune_scan.pieces atom with
  | [ Dune_scan.Literal path ] -> Some (Dune_scan.normalize_path (Dune_scan.in_subdir subdir path))
  | [ Dune_scan.Pform pform ] -> (
      match String.lsplit2 pform ~on:':' with
      | Some ("dep", path) -> Some (Dune_scan.normalize_path (Dune_scan.in_subdir subdir path))
      | _ -> None)
  | _ -> None

let target_dependencies ~subdir stanza =
  let rec collect = function
    | Sexp.Atom atom -> Option.to_list (dependency_path ~subdir atom)
    | Sexp.List (Sexp.Atom ("alias" | "alias_rec" | "env_var" | "universe") :: _) -> []
    | Sexp.List (Sexp.Atom name :: values) when String.is_prefix name ~prefix:":" ->
        List.concat_map values ~f:collect
    | Sexp.List children -> List.concat_map children ~f:collect
  in
  let field_dependencies =
    match Dune_scan.field stanza "deps" with
    | None -> []
    | Some deps -> List.concat_map deps ~f:collect
  in
  let action_dependencies =
    match Dune_scan.field stanza "action" with
    | None -> []
    | Some action ->
        List.concat_map action ~f:Dune_scan.atoms
        |> List.filter_map ~f:(fun atom ->
            match Dune_scan.pieces atom with
            | [ Dune_scan.Pform pform ] -> (
                match String.lsplit2 pform ~on:':' with
                | Some (("dep" | "read" | "read-lines" | "read-strings" | "path" | "file"), path) ->
                    Some (Dune_scan.normalize_path (Dune_scan.in_subdir subdir path))
                | _ -> None)
            | _ -> None)
  in
  List.rev_append field_dependencies action_dependencies
  |> List.dedup_and_sort ~compare:String.compare

let resolve_target ~subdir target = Dune_scan.normalize_path (Dune_scan.in_subdir subdir target)

let rec inferred_action_targets ~subdir = function
  | Sexp.Atom _ -> []
  | Sexp.List (Sexp.Atom "no-infer" :: _) -> []
  | Sexp.List
      (Sexp.Atom ("with-stdout-to" | "with-stderr-to" | "with-outputs-to")
      :: Sexp.Atom target
      :: nested) ->
      resolve_target ~subdir target :: List.concat_map nested ~f:(inferred_action_targets ~subdir)
  | Sexp.List (Sexp.Atom ("write-file" | "touch" | "mkdir") :: Sexp.Atom target :: _) ->
      [ resolve_target ~subdir target ]
  | Sexp.List
      (Sexp.Atom ("copy" | "copy#" | "copy-and-add-line-directive" | "format-dune-file")
      :: _source
      :: Sexp.Atom target
      :: _) ->
      [ resolve_target ~subdir target ]
  | Sexp.List children -> List.concat_map children ~f:(inferred_action_targets ~subdir)

let produced_targets ~subdir stanza =
  match Dune_scan.head stanza with
  | Some "rule" ->
      let declared =
        List.concat_map [ "target"; "targets" ] ~f:(fun field ->
            match Dune_scan.field stanza field with
            | None -> []
            | Some targets ->
                List.filter_map targets ~f:(function
                  | Sexp.Atom target -> Some (resolve_target ~subdir target)
                  | Sexp.List [ Sexp.Atom "dir"; Sexp.Atom target ] ->
                      Some (resolve_target ~subdir target)
                  | Sexp.List _ -> None))
      in
      let inferred =
        match Dune_scan.field stanza "action" with
        | None -> []
        | Some action -> inferred_action_targets ~subdir (Sexp.List action)
      in
      List.rev_append declared inferred |> List.dedup_and_sort ~compare:String.compare
  | _ -> []

let rec contains_head expected = function
  | Sexp.Atom _ -> false
  | Sexp.List (Sexp.Atom head :: children) ->
      String.equal head expected || List.exists children ~f:(contains_head expected)
  | Sexp.List children -> List.exists children ~f:(contains_head expected)

let accepts_nonstandard_exit stanza =
  match Dune_scan.field stanza "action" with
  | None -> false
  | Some action -> contains_head "with-accepted-exit-codes" (Sexp.List action)

let public_condition_error names =
  Printf.sprintf
    "public executable `%s` uses enabled_if, so its active membership cannot be derived statically"
    (String.concat ~sep:", " names)

let smoke_condition_error dune_path =
  Printf.sprintf
    "%s: @bin-smoke uses enabled_if, so its commands are not unconditional runtime coverage"
    dune_path

let include_error dune_path =
  Printf.sprintf "%s uses an unexpanded include stanza, so @bin-smoke membership may be incomplete"
    dune_path

let opaque_smoke_error dune_path command =
  Printf.sprintf "%s: @bin-smoke contains an opaque command: %s" dune_path command

let external_smoke_error dune_path =
  Printf.sprintf "%s: @bin-smoke reaches an external command whose effects are opaque" dune_path

let accepted_exit_error dune_path =
  Printf.sprintf "%s: @bin-smoke reaches with-accepted-exit-codes, so failure is not a canary"
    dune_path

let dynamic_run_error dune_path =
  Printf.sprintf "%s: @bin-smoke reaches dynamic-run, whose execution count is not fixed" dune_path

let directory_path_error dune_path =
  Printf.sprintf "%s rewrites PATH for a directory containing an @bin-smoke definition" dune_path

let implicit_alias_error dune_path dependency =
  Printf.sprintf "%s: @bin-smoke reaches implicit alias %s, whose contributors are not explicit"
    dune_path dependency

let may_have_implicit_contributors dependency =
  let name = path_basename dependency in
  String.equal name "runtest" || String.is_prefix name ~prefix:"runtest-"

let cached_alias_error dune_path targets =
  Printf.sprintf "%s: @bin-smoke reaches a target-bearing rule that may be cached: %s" dune_path
    (String.concat ~sep:", " targets)

let declarations_of_stanza ~subdir stanza =
  match Dune_scan.head stanza with
  | Some ("executable" | "executables") ->
      let names = Dune_scan.names_of stanza in
      let public_names = Dune_scan.public_names stanza in
      let condition_errors =
        if has_enabled_if stanza then [ public_condition_error names ] else []
      in
      if List.is_empty public_names then ([], [])
      else if List.length names <> List.length public_names then
        ( [],
          condition_errors
          @ [
              Printf.sprintf "public executable stanza has %d local name(s) but %d public name(s)"
                (List.length names) (List.length public_names);
            ] )
      else
        ( List.zip_exn names public_names
          |> List.filter_map ~f:(fun (name, public) ->
              if String.equal public "-" then None
              else
                Some
                  {
                    local = Dune_scan.normalize_path (Dune_scan.in_subdir subdir (name ^ ".exe"));
                    public;
                  }),
          condition_errors )
  | _ -> ([], [])

let smoke_targets_of_stanza ~allow_verified_helper ~dune_path ~subdir stanza =
  Dune_scan.classified_command_sites_with_pins_preserving_multiplicity stanza
  |> List.map ~f:(fun (cwd, _pins, site, command) ->
      match command with
      | Dune_scan.Runs path ->
          let local =
            Dune_scan.normalize_path (Dune_scan.in_subdir subdir (Dune_scan.in_subdir cwd path))
          in
          if
            allow_verified_helper
            && String.equal local verified_helper_local
            &&
            match site with
            | Dune_scan.Program (command, []) -> String.equal command verified_helper_command
            | _ -> false
          then Ok None
          else Ok (Some (Local local))
      | Dune_scan.Runs_public name -> Ok (Some (Public name))
      | Dune_scan.Unrecognized command
      | Dune_scan.Unknown_directory command
      | Dune_scan.Path_rewritten command ->
          Error (opaque_smoke_error dune_path command)
      | Dune_scan.External -> Error (external_smoke_error dune_path))

let target_name = function Local path -> path | Public name -> "%{bin:" ^ name ^ "}"

let scan dune_files =
  let declarations, _executable_locals, alias_nodes, generated_targets, scan_errors =
    List.fold dune_files ~init:([], [], [], [], [])
      ~f:(fun
          (all_declarations, all_locals, all_nodes, all_targets, all_errors) (dune_path, content) ->
        try
          let stanzas = Dune_scan.stanzas content in
          let directory = path_dirname dune_path in
          let walked =
            Dune_scan.walk "" stanzas ~f:(fun subdir stanza ->
                [ (Dune_scan.in_subdir directory subdir, stanza) ])
          in
          let declarations, executable_locals, declaration_errors =
            if String.equal dune_path "bin/dune" then
              List.fold walked ~init:([], [], [])
                ~f:(fun (declarations, locals, errors) (subdir, stanza) ->
                  let found_declarations, found_errors = declarations_of_stanza ~subdir stanza in
                  let locals =
                    match Dune_scan.head stanza with
                    | Some ("executable" | "executables") ->
                        List.rev_append
                          (List.map (Dune_scan.names_of stanza) ~f:(fun name ->
                               Dune_scan.normalize_path (Dune_scan.in_subdir subdir (name ^ ".exe"))))
                          locals
                    | _ -> locals
                  in
                  ( List.rev_append found_declarations declarations,
                    locals,
                    List.rev_append found_errors errors ))
            else ([], [], [])
          in
          let alias_nodes =
            List.filter_mapi walked ~f:(fun index (subdir, stanza) ->
                let names = aliases_of stanza in
                if List.is_empty names then None
                else
                  let dependencies, dependency_errors = alias_dependencies ~subdir stanza in
                  Some
                    {
                      id = Printf.sprintf "%s:%d" dune_path index;
                      dune_path;
                      subdir;
                      stanza;
                      aliases = List.map names ~f:(alias_key ~subdir);
                      is_bin_smoke = List.mem names "bin-smoke" ~equal:String.equal;
                      dependencies;
                      target_dependencies = target_dependencies ~subdir stanza;
                      dependency_errors;
                    })
          in
          let generated_targets =
            List.concat_map walked ~f:(fun (subdir, stanza) -> produced_targets ~subdir stanza)
          in
          let include_errors =
            List.concat_map walked ~f:(fun (_subdir, stanza) ->
                match Dune_scan.head stanza with
                | Some "include" -> [ include_error dune_path ]
                | _ -> [])
          in
          ( List.rev_append declarations all_declarations,
            List.rev_append executable_locals all_locals,
            List.rev_append alias_nodes all_nodes,
            List.rev_append generated_targets all_targets,
            List.rev_append declaration_errors (List.rev_append include_errors all_errors) )
        with exn ->
          ( all_declarations,
            all_locals,
            all_nodes,
            all_targets,
            Printf.sprintf "cannot parse %s: %s" dune_path (Exn.to_string exn) :: all_errors ))
  in
  let path_rewriting_directories =
    List.filter_map dune_files ~f:(fun (dune_path, content) ->
        try
          if List.is_empty (Dune_scan.path_rewriting_stanzas content) then None
          else Some (path_dirname dune_path, dune_path)
        with _ -> None)
  in
  let smoke_roots = List.filter alias_nodes ~f:(fun node -> node.is_bin_smoke) in
  let directory_path_errors node =
    List.filter_map path_rewriting_directories ~f:(fun (directory, dune_path) ->
        if
          String.is_empty directory
          || String.equal directory node.subdir
          || String.is_prefix node.subdir ~prefix:(directory ^ "/")
        then Some (directory_path_error dune_path)
        else None)
  in
  let rec visit visited targets errors = function
    | [] -> (targets, errors)
    | node :: rest when Set.mem visited node.id -> visit visited targets errors rest
    | node :: rest ->
        let visited = Set.add visited node.id in
        let allow_verified_helper =
          List.equal String.equal node.aliases [ verified_helper_alias ]
        in
        let targets, command_errors =
          smoke_targets_of_stanza ~allow_verified_helper ~dune_path:node.dune_path
            ~subdir:node.subdir node.stanza
          |> List.fold ~init:(targets, []) ~f:(fun (targets, errors) -> function
            | Ok (Some target) -> (target :: targets, errors)
            | Ok None -> (targets, errors)
            | Error error -> (targets, error :: errors))
        in
        let condition_errors =
          if has_enabled_if node.stanza then [ smoke_condition_error node.dune_path ] else []
        in
        let exit_errors =
          if accepts_nonstandard_exit node.stanza then [ accepted_exit_error node.dune_path ]
          else []
        in
        let dynamic_run_errors =
          if contains_head "dynamic-run" node.stanza then [ dynamic_run_error node.dune_path ]
          else []
        in
        let directory_path_errors = directory_path_errors node in
        let generated_dependency_errors =
          List.filter node.target_dependencies ~f:(fun dependency ->
              List.mem generated_targets dependency ~equal:String.equal)
          |> List.map ~f:(fun dependency ->
              Printf.sprintf "%s: @bin-smoke reaches generated target dependency %s" node.dune_path
                dependency)
        in
        let alias_targets = produced_targets ~subdir:node.subdir node.stanza in
        let cached_alias_errors =
          if List.is_empty alias_targets then []
          else [ cached_alias_error node.dune_path alias_targets ]
        in
        let dependencies, missing_dependency_errors =
          List.fold node.dependencies ~init:([], []) ~f:(fun (nodes, errors) dependency ->
              let errors =
                if may_have_implicit_contributors dependency then
                  implicit_alias_error node.dune_path dependency :: errors
                else errors
              in
              let found =
                List.filter alias_nodes ~f:(fun candidate ->
                    List.mem candidate.aliases dependency ~equal:String.equal)
              in
              if List.is_empty found then
                ( nodes,
                  Printf.sprintf "%s: @bin-smoke depends on undefined alias %s" node.dune_path
                    dependency
                  :: errors )
              else (List.rev_append found nodes, errors))
        in
        let errors =
          List.rev_append node.dependency_errors
            (List.rev_append command_errors
               (List.rev_append condition_errors
                  (List.rev_append exit_errors
                     (List.rev_append dynamic_run_errors
                        (List.rev_append directory_path_errors
                           (List.rev_append generated_dependency_errors
                              (List.rev_append cached_alias_errors
                                 (List.rev_append missing_dependency_errors errors))))))))
        in
        visit visited targets errors (List.rev_append dependencies rest)
  in
  let targets, smoke_errors = visit (Set.empty (module String)) [] [] smoke_roots in
  let smoke_stanza_count = List.length smoke_roots in
  let scan_errors = List.rev_append smoke_errors scan_errors in
  let declarations = List.sort declarations ~compare:(fun a b -> String.compare a.local b.local) in
  let duplicate_locals =
    duplicates (List.map declarations ~f:(fun declaration -> declaration.local))
  in
  let duplicate_public =
    duplicates (List.map declarations ~f:(fun declaration -> declaration.public))
  in
  let canonicalize target =
    let matches =
      List.filter declarations ~f:(fun declaration ->
          match target with
          | Local path -> String.equal declaration.local path
          | Public name -> String.equal declaration.public name)
    in
    match matches with
    | [ declaration ] -> Ok declaration.local
    | [] -> Error (target_name target)
    | _ -> Error (target_name target ^ " (ambiguous public executable identity)")
  in
  let smoked, unexpected =
    List.fold targets ~init:([], []) ~f:(fun (smoked, unexpected) target ->
        match canonicalize target with
        | Ok local -> (local :: smoked, unexpected)
        | Error target -> (smoked, target :: unexpected))
  in
  let smoked = sorted smoked in
  let expected = List.map declarations ~f:(fun declaration -> declaration.local) |> sorted in
  let missing =
    List.filter expected ~f:(fun local -> not (List.mem smoked local ~equal:String.equal))
  in
  let duplicate_smoked = duplicates smoked in
  let errors =
    List.rev scan_errors
    @ (if Int.equal smoke_stanza_count 0 then [ "the repository defines no @bin-smoke action" ]
       else [])
    @ List.map duplicate_locals ~f:(Printf.sprintf "duplicate public local identity: %s")
    @ List.map duplicate_public ~f:(Printf.sprintf "duplicate public installed identity: %s")
    @ List.map duplicate_smoked ~f:(Printf.sprintf "@bin-smoke runs more than once: %s")
  in
  { declarations; smoked; missing; unexpected = sorted unexpected; errors }

let scan_bin_content content = scan [ ("bin/dune", content) ]

let complete result =
  List.length result.declarations > 0
  && Int.equal (List.length result.missing) 0
  && Int.equal (List.length result.unexpected) 0
  && Int.equal (List.length result.errors) 0
  && List.equal String.equal
       (List.map result.declarations ~f:(fun declaration -> declaration.local) |> sorted)
       result.smoked

let complete_fixture =
  {dune|(executable (name alpha) (public_name alpha-tool))
(executable (name beta) (public_name beta-tool))
(executable (name env_spelling_gate))
(rule
 (alias bin-smoke-env_spelling_gate)
 (action (run %{dep:env_spelling_gate.exe})))
(rule
 (alias bin-smoke)
 (deps (alias bin-smoke-env_spelling_gate))
 (action
  (progn
   (run %{exe:alpha.exe})
   (run %{bin:beta-tool}))))|dune}

let missing_fixture =
  {dune|(executable (name alpha) (public_name alpha-tool))
(executable (name beta) (public_name beta-tool))
(rule
 (alias bin-smoke)
 (action (run %{exe:alpha.exe})))|dune}

let duplicate_fixture =
  {dune|(executable (name alpha) (public_name alpha-tool))
(executable (name beta) (public_name beta-tool))
(rule
 (alias bin-smoke)
 (action
  (progn
   (run %{exe:alpha.exe})
   (run %{exe:alpha.exe})
   (run %{exe:beta.exe}))))|dune}

let conditional_smoke_fixture =
  {dune|(executable (name alpha) (public_name alpha-tool))
(rule
 (alias bin-smoke)
 (enabled_if false)
 (action (run %{exe:alpha.exe})))|dune}

let conditional_public_fixture =
  {dune|(executable
 (name alpha)
 (public_name alpha-tool)
 (enabled_if false))
(rule
 (alias bin-smoke)
 (action (run %{exe:alpha.exe})))|dune}

let included_fixture = complete_fixture ^ "\n(include extra-stanzas.inc)\n"

let opaque_smoke_fixture =
  {dune|(executable (name alpha) (public_name alpha-tool))
(rule
 (alias bin-smoke)
 (action
  (progn
   (run %{exe:alpha.exe})
   (bash "./alpha.exe --again"))))|dune}

let competing_alias_fixture =
  {dune|(rule
 (alias bin-smoke)
 (action (run %{bin:alpha-tool})))|dune}

let transitive_duplicate_fixture =
  {dune|(executable (name alpha) (public_name alpha-tool))
(rule
 (alias helper-smoke)
 (action (run %{exe:alpha.exe})))
(rule
 (alias bin-smoke)
 (deps (alias helper-smoke))
 (action (run %{exe:alpha.exe})))|dune}

let external_smoke_fixture =
  {dune|(executable (name alpha) (public_name alpha-tool))
(rule
 (alias bin-smoke)
 (deps runner.py)
 (action
  (progn
   (run %{exe:alpha.exe})
   (run python3 runner.py))))|dune}

let accepted_exit_fixture =
  {dune|(executable (name alpha) (public_name alpha-tool))
(rule
 (alias bin-smoke)
 (action
  (with-accepted-exit-codes 1
   (run %{exe:alpha.exe}))))|dune}

let generated_target_fixture =
  {dune|(executable (name alpha) (public_name alpha-tool))
(rule
 (target smoke.stamp)
 (action
  (progn
   (run %{exe:alpha.exe})
   (touch %{target}))))
(rule
 (alias bin-smoke)
 (deps smoke.stamp)
 (action (run %{exe:alpha.exe})))|dune}

let inferred_target_fixture =
  {dune|(executable (name alpha) (public_name alpha-tool))
(rule
 (action
  (with-stdout-to smoke.stamp
   (run %{exe:alpha.exe}))))
(rule
 (alias bin-smoke)
 (deps smoke.stamp)
 (action (run %{exe:alpha.exe})))|dune}

let target_bearing_alias_fixture =
  {dune|(executable (name alpha) (public_name alpha-tool))
(rule
 (alias bin-smoke)
 (target smoke.stamp)
 (action
  (progn
   (run %{exe:alpha.exe})
   (touch smoke.stamp))))|dune}

let directory_target_fixture =
  {dune|(executable (name alpha) (public_name alpha-tool))
(rule
 (targets (dir smoke-output))
 (action
  (progn
   (run %{exe:alpha.exe})
   (mkdir smoke-output))))
(rule
 (alias bin-smoke)
 (deps smoke-output)
 (action (run %{exe:alpha.exe})))|dune}

let directory_target_alias_fixture =
  {dune|(executable (name alpha) (public_name alpha-tool))
(rule
 (alias bin-smoke)
 (targets (dir smoke-output))
 (action (run %{exe:alpha.exe})))|dune}

let rewritten_path_fixture =
  {dune|(executable (name alpha) (public_name alpha-tool))
(rule
 (alias bin-smoke)
 (action
  (setenv Path elsewhere
   (run alpha.exe))))|dune}

let dynamic_run_fixture =
  {dune|(executable (name alpha) (public_name alpha-tool))
(rule
 (alias bin-smoke)
 (action (dynamic-run %{exe:alpha.exe})))|dune}

let directory_path_fixture =
  {dune|(env
 (_
  (env-vars
   (Path elsewhere))))
(executable (name alpha) (public_name alpha-tool))
(rule
 (alias bin-smoke)
 (action (run alpha.exe)))|dune}

let private_launcher_fixture =
  {dune|(executable (name alpha) (public_name alpha-tool))
(executable (name helper))
(rule
 (alias helper-smoke)
 (action (run %{exe:helper.exe} %{exe:alpha.exe})))
(rule
 (alias bin-smoke)
 (deps (alias helper-smoke))
 (action (run %{exe:alpha.exe})))|dune}

let verified_helper_launcher_fixture =
  {dune|(executable (name alpha) (public_name alpha-tool))
(executable (name env_spelling_gate))
(rule
 (alias bin-smoke-env_spelling_gate)
 (action (run %{dep:env_spelling_gate.exe} %{exe:alpha.exe})))
(rule
 (alias bin-smoke)
 (deps (alias bin-smoke-env_spelling_gate))
 (action (run %{exe:alpha.exe})))|dune}

let action_dependency_fixture =
  {dune|(executable (name alpha) (public_name alpha-tool))
(rule
 (target smoke.stamp)
 (action (run %{exe:alpha.exe})))
(rule
 (alias bin-smoke)
 (action (run %{exe:alpha.exe} %{dep:smoke.stamp})))|dune}

let transitive_path_bin_fixture =
  {dune|(executable (name alpha) (public_name alpha-tool))
(rule
 (alias bin-smoke)
 (deps (alias ../test/helper-smoke))
 (action (run %{exe:alpha.exe})))|dune}

let transitive_path_helper_fixture =
  {dune|(env (_ (env-vars (Path elsewhere))))
(rule
 (alias helper-smoke)
 (action (chdir ../bin (run alpha.exe))))|dune}

let implicit_alias_fixture =
  {dune|(executable (name alpha) (public_name alpha-tool))
(test
 (name hidden)
 (action (run %{exe:alpha.exe})))
(alias (name runtest))
(rule
 (alias bin-smoke)
 (deps (alias runtest))
 (action (run %{exe:alpha.exe})))|dune}

let controls_hold () =
  let accepted = scan_bin_content complete_fixture in
  let refused = scan_bin_content missing_fixture in
  let duplicated = scan_bin_content duplicate_fixture in
  let conditional_smoke = scan_bin_content conditional_smoke_fixture in
  let conditional_public = scan_bin_content conditional_public_fixture in
  let included = scan_bin_content included_fixture in
  let opaque = scan_bin_content opaque_smoke_fixture in
  let competing =
    scan [ ("bin/dune", complete_fixture); ("test/synthetic/dune", competing_alias_fixture) ]
  in
  let transitive = scan_bin_content transitive_duplicate_fixture in
  let external_result = scan_bin_content external_smoke_fixture in
  let accepted_exit = scan_bin_content accepted_exit_fixture in
  let generated_target = scan_bin_content generated_target_fixture in
  let inferred_target = scan_bin_content inferred_target_fixture in
  let target_bearing_alias = scan_bin_content target_bearing_alias_fixture in
  let directory_target = scan_bin_content directory_target_fixture in
  let directory_target_alias = scan_bin_content directory_target_alias_fixture in
  let rewritten_path = scan_bin_content rewritten_path_fixture in
  let dynamic_run = scan_bin_content dynamic_run_fixture in
  let directory_path = scan_bin_content directory_path_fixture in
  let private_launcher = scan_bin_content private_launcher_fixture in
  let verified_helper_launcher = scan_bin_content verified_helper_launcher_fixture in
  let action_dependency = scan_bin_content action_dependency_fixture in
  let transitive_path =
    scan
      [ ("bin/dune", transitive_path_bin_fixture); ("test/dune", transitive_path_helper_fixture) ]
  in
  let implicit_alias = scan_bin_content implicit_alias_fixture in
  complete accepted
  && (not (complete refused))
  && List.equal String.equal refused.missing [ "bin/beta.exe" ]
  && List.is_empty refused.unexpected && List.is_empty refused.errors
  && (not (complete duplicated))
  && List.mem duplicated.errors "@bin-smoke runs more than once: bin/alpha.exe" ~equal:String.equal
  && (not (complete conditional_smoke))
  && List.mem conditional_smoke.errors (smoke_condition_error "bin/dune") ~equal:String.equal
  && (not (complete conditional_public))
  && List.mem conditional_public.errors (public_condition_error [ "alpha" ]) ~equal:String.equal
  && (not (complete included))
  && List.mem included.errors (include_error "bin/dune") ~equal:String.equal
  && (not (complete opaque))
  && List.mem opaque.errors
       (opaque_smoke_error "bin/dune" "shell: ./alpha.exe --again")
       ~equal:String.equal
  && (not (complete competing))
  && List.mem competing.errors "@bin-smoke runs more than once: bin/alpha.exe" ~equal:String.equal
  && (not (complete transitive))
  && List.mem transitive.errors "@bin-smoke runs more than once: bin/alpha.exe" ~equal:String.equal
  && (not (complete external_result))
  && List.mem external_result.errors (external_smoke_error "bin/dune") ~equal:String.equal
  && (not (complete accepted_exit))
  && List.mem accepted_exit.errors (accepted_exit_error "bin/dune") ~equal:String.equal
  && (not (complete generated_target))
  && List.mem generated_target.errors
       "bin/dune: @bin-smoke reaches generated target dependency bin/smoke.stamp"
       ~equal:String.equal
  && (not (complete inferred_target))
  && List.mem inferred_target.errors
       "bin/dune: @bin-smoke reaches generated target dependency bin/smoke.stamp"
       ~equal:String.equal
  && (not (complete target_bearing_alias))
  && List.mem target_bearing_alias.errors
       (cached_alias_error "bin/dune" [ "bin/smoke.stamp" ])
       ~equal:String.equal
  && (not (complete directory_target))
  && List.mem directory_target.errors
       "bin/dune: @bin-smoke reaches generated target dependency bin/smoke-output"
       ~equal:String.equal
  && (not (complete directory_target_alias))
  && List.mem directory_target_alias.errors
       (cached_alias_error "bin/dune" [ "bin/smoke-output" ])
       ~equal:String.equal
  && (not (complete rewritten_path))
  && List.mem rewritten_path.errors
       (opaque_smoke_error "bin/dune" "alpha.exe, under `(setenv PATH elsewhere ...)`")
       ~equal:String.equal
  && (not (complete dynamic_run))
  && List.mem dynamic_run.errors (dynamic_run_error "bin/dune") ~equal:String.equal
  && (not (complete directory_path))
  && List.mem directory_path.errors (directory_path_error "bin/dune") ~equal:String.equal
  && (not (complete private_launcher))
  && List.mem private_launcher.unexpected "bin/helper.exe" ~equal:String.equal
  && (not (complete verified_helper_launcher))
  && List.mem verified_helper_launcher.unexpected "bin/env_spelling_gate.exe" ~equal:String.equal
  && (not (complete action_dependency))
  && List.mem action_dependency.errors
       "bin/dune: @bin-smoke reaches generated target dependency bin/smoke.stamp"
       ~equal:String.equal
  && (not (complete transitive_path))
  && List.mem transitive_path.errors (directory_path_error "test/dune") ~equal:String.equal
  && (not (complete implicit_alias))
  && List.mem implicit_alias.errors
       (implicit_alias_error "bin/dune" "bin/runtest")
       ~equal:String.equal

let report_diagnostics result =
  eprintf "Public bin executables (%d): [%s]\n" (List.length result.declarations)
    (List.map result.declarations ~f:(fun declaration ->
         Printf.sprintf "%s -> %s" declaration.local declaration.public)
    |> String.concat ~sep:"; ");
  eprintf "@bin-smoke members (%d): [%s]\n" (List.length result.smoked)
    (String.concat ~sep:"; " result.smoked);
  List.iter result.missing ~f:(fun local -> eprintf "missing from @bin-smoke: %s\n" local);
  List.iter result.unexpected ~f:(fun target ->
      eprintf "not a public bin executable but run by @bin-smoke: %s\n" target);
  List.iter result.errors ~f:(eprintf "%s\n")

let report result =
  report_diagnostics result;
  p "bin-smoke covers every public bin executable exactly once" (complete result)

let () =
  match Array.to_list Stdlib.Sys.argv with
  | [ _; "--negative-control" ] -> report (scan_bin_content missing_fixture)
  | _ :: workspace_root :: dune_paths ->
      let base = Dune_scan.base_dir workspace_root in
      let dune_files =
        List.filter_map dune_paths ~f:(fun path ->
            let relative = Dune_scan.repo_relative base path in
            if String.equal (List.last_exn (String.split relative ~on:'/')) "dune" then
              Some (relative, In_channel.read_all path)
            else None)
      in
      let result = scan dune_files in
      let controls_hold = controls_hold () in
      let result =
        if controls_hold then result
        else { result with errors = "synthetic membership controls failed" :: result.errors }
      in
      report result;
      Test_utils.Refusal_control_manifest.print "bin_smoke_membership_scan.ml"
  | argv ->
      eprintf "Usage: %s <workspace-root> <dune-file...>|--negative-control\n" (List.hd_exn argv);
      Stdlib.exit 2
