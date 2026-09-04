(** gh-ocannl-874: every public executable declared in [bin/dune] is run exactly once by
    [@bin-smoke].

    The public declarations and the smoke action are independent lists in one dune file. A new
    executable therefore compiles under [@check] even when its runtime canary is accidentally
    omitted. This scan relates those two lists from the parsed dune structure. It accepts either the
    executable's local [%{exe:...}] identity or its [%{bin:...}] public identity, and requires exact
    membership so a stale or duplicated smoke command is visible too. Conditional executable or
    smoke stanzas and unexpanded [(include ...)] forms are refused: without evaluating Dune's
    condition language or loading the included dependency, either would make a syntactic member look
    like a command CI necessarily executes.

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

let smoke_condition_error =
  "@bin-smoke uses enabled_if, so its commands are not unconditional runtime coverage"

let include_error =
  "bin/dune uses an unexpanded include stanza, so public and smoke membership may be incomplete"

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

let smoke_targets_of_stanza ~subdir stanza =
  if not (is_bin_smoke_stanza stanza) then []
  else
    Dune_scan.runs_of_with_multiplicity ~subdir stanza
    |> List.map ~f:(fun (identity, _pins) ->
        match identity with `File path -> Local path | `Public name -> Public name)

let target_name = function Local path -> path | Public name -> "%{bin:" ^ name ^ "}"

let scan content =
  let parsed =
    try
      let stanzas = Dune_scan.stanzas content in
      let declarations, declaration_errors =
        Dune_scan.walk "" stanzas ~f:(fun subdir stanza ->
            let declarations, errors = declarations_of_stanza ~subdir stanza in
            [ (declarations, errors) ])
        |> List.fold ~init:([], []) ~f:(fun (all, errors) (found, found_errors) ->
            (List.rev_append found all, List.rev_append found_errors errors))
      in
      let smoke_stanzas =
        Dune_scan.walk "" stanzas ~f:(fun subdir stanza ->
            if is_bin_smoke_stanza stanza then [ (subdir, stanza) ] else [])
      in
      let targets =
        List.concat_map smoke_stanzas ~f:(fun (subdir, stanza) ->
            smoke_targets_of_stanza ~subdir stanza)
      in
      let smoke_condition_errors =
        List.filter_map smoke_stanzas ~f:(fun (_subdir, stanza) ->
            if has_enabled_if stanza then Some smoke_condition_error else None)
      in
      let include_errors =
        Dune_scan.walk "" stanzas ~f:(fun _subdir stanza ->
            match Dune_scan.head stanza with Some "include" -> [ include_error ] | _ -> [])
      in
      Ok
        ( declarations,
          List.rev declaration_errors @ smoke_condition_errors @ include_errors,
          smoke_stanzas,
          targets )
    with exn -> Error (Exn.to_string exn)
  in
  match parsed with
  | Error error ->
      {
        declarations = [];
        smoked = [];
        missing = [];
        unexpected = [];
        errors = [ "cannot parse bin/dune: " ^ error ];
      }
  | Ok (declarations, declaration_errors, smoke_stanzas, targets) ->
      let declarations =
        List.sort declarations ~compare:(fun a b -> String.compare a.local b.local)
      in
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
        declaration_errors
        @ (if List.is_empty smoke_stanzas then [ "bin/dune defines no @bin-smoke action" ] else [])
        @ List.map duplicate_locals ~f:(Printf.sprintf "duplicate public local identity: %s")
        @ List.map duplicate_public ~f:(Printf.sprintf "duplicate public installed identity: %s")
        @ List.map duplicate_smoked ~f:(Printf.sprintf "@bin-smoke runs more than once: %s")
      in
      { declarations; smoked; missing; unexpected = sorted unexpected; errors }

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

let controls_hold () =
  let accepted = scan complete_fixture in
  let refused = scan missing_fixture in
  let duplicated = scan duplicate_fixture in
  let conditional_smoke = scan conditional_smoke_fixture in
  let conditional_public = scan conditional_public_fixture in
  let included = scan included_fixture in
  complete accepted
  && (not (complete refused))
  && List.equal String.equal refused.missing [ "beta.exe" ]
  && List.is_empty refused.unexpected && List.is_empty refused.errors
  && (not (complete duplicated))
  && List.mem duplicated.errors "@bin-smoke runs more than once: alpha.exe" ~equal:String.equal
  && (not (complete conditional_smoke))
  && List.mem conditional_smoke.errors smoke_condition_error ~equal:String.equal
  && (not (complete conditional_public))
  && List.mem conditional_public.errors (public_condition_error [ "alpha" ]) ~equal:String.equal
  && (not (complete included))
  && List.mem included.errors include_error ~equal:String.equal

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
  | [ _; "--negative-control" ] -> report (scan missing_fixture)
  | [ _; dune_file ] ->
      let result = scan (In_channel.read_all dune_file) in
      let controls_hold = controls_hold () in
      let result =
        if controls_hold then result
        else { result with errors = "synthetic membership controls failed" :: result.errors }
      in
      report result;
      Test_utils.Refusal_control_manifest.print "bin_smoke_membership_scan.ml"
  | argv ->
      eprintf "Usage: %s <bin/dune>|--negative-control\n" (List.hd_exn argv);
      Stdlib.exit 2
