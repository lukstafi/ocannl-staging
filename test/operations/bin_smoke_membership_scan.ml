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

let sorted = List.sort ~compare:String.compare

let path_dirname path =
  match String.rsplit2 path ~on:'/' with Some (directory, _) -> directory | None -> ""

let duplicates strings =
  List.sort_and_group strings ~compare:String.compare
  |> List.filter_map ~f:(function first :: _ :: _ -> Some first | _ -> None)

let aliases_of stanza =
  List.concat_map [ "alias"; "aliases" ] ~f:(fun field ->
      match Dune_scan.field stanza field with
      | None -> []
      | Some args -> List.filter_map args ~f:(function Sexp.Atom name -> Some name | _ -> None))

let is_bin_smoke_stanza stanza = List.mem (aliases_of stanza) "bin-smoke" ~equal:String.equal
let has_enabled_if stanza = Option.is_some (Dune_scan.field stanza "enabled_if")

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

let smoke_targets_of_stanza ~dune_path ~subdir stanza =
  if not (is_bin_smoke_stanza stanza) then []
  else
    Dune_scan.executables_run_with_pins_preserving_multiplicity stanza
    |> List.map ~f:(fun (cwd, _pins, command) ->
        match command with
        | Dune_scan.Runs path ->
            Ok
              (Local
                 (Dune_scan.normalize_path
                    (Dune_scan.in_subdir subdir (Dune_scan.in_subdir cwd path))))
        | Dune_scan.Runs_public name -> Ok (Public name)
        | Dune_scan.Unrecognized command
        | Dune_scan.Unknown_directory command
        | Dune_scan.Path_rewritten command ->
            Error (opaque_smoke_error dune_path command)
        | Dune_scan.External -> assert false)

let target_name = function Local path -> path | Public name -> "%{bin:" ^ name ^ "}"

let scan dune_files =
  let declarations, targets, smoke_stanza_count, scan_errors =
    List.fold dune_files ~init:([], [], 0, [])
      ~f:(fun (all_declarations, all_targets, smoke_count, all_errors) (dune_path, content) ->
        try
          let stanzas = Dune_scan.stanzas content in
          let directory = path_dirname dune_path in
          let declarations, declaration_errors =
            if String.equal dune_path "bin/dune" then
              Dune_scan.walk "" stanzas ~f:(fun subdir stanza ->
                  let subdir = Dune_scan.in_subdir directory subdir in
                  let declarations, errors = declarations_of_stanza ~subdir stanza in
                  [ (declarations, errors) ])
              |> List.fold ~init:([], []) ~f:(fun (all, errors) (found, found_errors) ->
                  (List.rev_append found all, List.rev_append found_errors errors))
            else ([], [])
          in
          let smoke_stanzas =
            Dune_scan.walk "" stanzas ~f:(fun subdir stanza ->
                if is_bin_smoke_stanza stanza then
                  [ (Dune_scan.in_subdir directory subdir, stanza) ]
                else [])
          in
          let targets, opaque_errors =
            List.concat_map smoke_stanzas ~f:(fun (subdir, stanza) ->
                smoke_targets_of_stanza ~dune_path ~subdir stanza)
            |> List.fold ~init:([], []) ~f:(fun (targets, errors) -> function
              | Ok target -> (target :: targets, errors)
              | Error error -> (targets, error :: errors))
          in
          let smoke_condition_errors =
            List.filter_map smoke_stanzas ~f:(fun (_subdir, stanza) ->
                if has_enabled_if stanza then Some (smoke_condition_error dune_path) else None)
          in
          let include_errors =
            Dune_scan.walk "" stanzas ~f:(fun _subdir stanza ->
                match Dune_scan.head stanza with
                | Some "include" -> [ include_error dune_path ]
                | _ -> [])
          in
          ( List.rev_append declarations all_declarations,
            List.rev_append targets all_targets,
            smoke_count + List.length smoke_stanzas,
            List.rev_append declaration_errors
              (List.rev_append opaque_errors
                 (List.rev_append smoke_condition_errors
                    (List.rev_append include_errors all_errors))) )
        with exn ->
          ( all_declarations,
            all_targets,
            smoke_count,
            Printf.sprintf "cannot parse %s: %s" dune_path (Exn.to_string exn) :: all_errors ))
  in
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
(executable (name helper))
(rule
 (alias bin-smoke)
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
