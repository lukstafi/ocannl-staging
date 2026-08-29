(* A per-backend golden is useful only when its family covers every backend OCANNL has.

   Family membership comes from filenames: a backend name delimited by [-] on the left and [-] or
   [.] on the right is replaced by [<backend>] to form the family identity. The backend vocabulary
   comes from [Backends.all_of_backend] and [Backends.backend_name], so adding a backend makes an
   incomplete family fail here before dune reports a raw missing-rule error on that backend.

   A member copied from a golden recorded on another backend is deliberately not a failure: the
   daily backend sweep is what can replace it. It must, however, be visible in-tree. Put this rigid
   comment in the [dune] file beside the family rule:

   ; ocannl-golden-recorded-on: <member>.expected <- <backend> -- <reason>

   The scan validates the target, source backend, uniqueness and reason, then prints every such
   member into its own golden. Remove the marker when the member is recorded on its own backend. *)

open Base
open Stdio
module Backends = Context.Backends

type member = { path : string; family : string; backend : string }
type provenance = { member : string; recorded_on : string; reason : string }

let marker = "ocannl-golden-recorded-on:"
let sorted = List.sort ~compare:String.compare

let find_occurrences text ~pattern =
  let pattern_length = String.length pattern in
  let rec loop from acc =
    match String.substr_index text ~pos:from ~pattern with
    | None -> List.rev acc
    | Some index -> loop (index + pattern_length) (index :: acc)
  in
  if Int.equal pattern_length 0 then [] else loop 0 []

let member_of_path ~backends path =
  let basename = Stdlib.Filename.basename path in
  let dirname = Stdlib.Filename.dirname path in
  let candidates =
    List.concat_map backends ~f:(fun backend ->
        let token = "-" ^ backend in
        find_occurrences basename ~pattern:token
        |> List.filter_map ~f:(fun index ->
            let suffix_at = index + String.length token in
            if
              suffix_at < String.length basename
              && (Char.equal basename.[suffix_at] '-' || Char.equal basename.[suffix_at] '.')
            then
              let family_basename =
                String.prefix basename index ^ "-<backend>" ^ String.drop_prefix basename suffix_at
              in
              Some { path; family = Stdlib.Filename.concat dirname family_basename; backend }
            else None))
  in
  match candidates with
  | [] -> Ok None
  | [ member ] -> Ok (Some member)
  | _ ->
      Error
        (Printf.sprintf "%s names more than one backend position: [%s]" path
           (List.map candidates ~f:(fun member -> member.backend)
           |> sorted |> String.concat ~sep:"; "))

let split_once text ~on =
  match String.substr_index text ~pattern:on with
  | None -> None
  | Some index -> Some (String.prefix text index, String.drop_prefix text (index + String.length on))

let marker_of_line ~backends ~dune_path ~line_number line =
  let occurrences = find_occurrences line ~pattern:marker in
  match occurrences with
  | [] -> Ok None
  | [ _ ] -> (
      let stripped = String.strip line in
      let prefix = "; " ^ marker ^ " " in
      match String.chop_prefix stripped ~prefix with
      | None ->
          Error
            (Printf.sprintf
               "%s:%d: malformed provenance marker; expected `; %s <member>.expected <- <backend> \
                -- <reason>`"
               dune_path line_number marker)
      | Some body -> (
          match split_once body ~on:" -- " with
          | None ->
              Error
                (Printf.sprintf "%s:%d: provenance marker has no ` -- <reason>`" dune_path
                   line_number)
          | Some (relationship, reason) -> (
              match split_once relationship ~on:" <- " with
              | None ->
                  Error
                    (Printf.sprintf
                       "%s:%d: provenance marker has no ` <member>.expected <- <backend>`" dune_path
                       line_number)
              | Some (basename, recorded_on) ->
                  let basename = String.strip basename in
                  let recorded_on = String.strip recorded_on in
                  let reason = String.strip reason in
                  let member =
                    Stdlib.Filename.concat (Stdlib.Filename.dirname dune_path) basename
                  in
                  let errors =
                    List.filter_opt
                      [
                        (if
                           String.is_empty basename
                           || (not (String.equal basename (Stdlib.Filename.basename basename)))
                           || not (String.is_suffix basename ~suffix:".expected")
                         then
                           Some
                             (Printf.sprintf
                                "%s:%d: provenance target must be an .expected basename in the \
                                 dune file's directory"
                                dune_path line_number)
                         else None);
                        (if List.mem backends recorded_on ~equal:String.equal then None
                         else
                           Some
                             (Printf.sprintf "%s:%d: recorded-on backend `%s` is not one OCANNL has"
                                dune_path line_number recorded_on));
                        (if String.is_empty reason then
                           Some
                             (Printf.sprintf "%s:%d: provenance marker reason is empty" dune_path
                                line_number)
                         else None);
                      ]
                  in
                  if List.is_empty errors then Ok (Some { member; recorded_on; reason })
                  else Error (String.concat ~sep:"\n" errors))))
  | _ ->
      Error
        (Printf.sprintf "%s:%d: more than one `%s` marker occurs on this line" dune_path line_number
           marker)

let () =
  if Array.length Stdlib.Sys.argv < 2 then (
    eprintf "Usage: %s <workspace_root> <expected-or-dune-file...>\n" Stdlib.Sys.argv.(0);
    Stdlib.exit 1);
  let base = Test_utils.Dune_stanza_scan.base_dir Stdlib.Sys.argv.(1) in
  let arguments =
    Array.to_list (Array.subo Stdlib.Sys.argv ~pos:2)
    |> List.map ~f:(fun path -> (Test_utils.Dune_stanza_scan.repo_relative base path, path))
  in
  let backends = List.map Backends.all_of_backend ~f:Backends.backend_name |> sorted in
  let expected_paths =
    List.filter_map arguments ~f:(fun (relative, _) ->
        if String.is_suffix relative ~suffix:".expected" then Some relative else None)
  in
  let members, member_errors =
    List.fold expected_paths ~init:([], []) ~f:(fun (members, errors) path ->
        match member_of_path ~backends path with
        | Ok None -> (members, errors)
        | Ok (Some member) -> (member :: members, errors)
        | Error error -> (members, error :: errors))
  in
  let families =
    List.fold members
      ~init:(Map.empty (module String))
      ~f:(fun families member -> Map.add_multi families ~key:member.family ~data:member)
  in
  let incomplete =
    Map.to_alist families
    |> List.filter_map ~f:(fun (family, members) ->
        let actual = List.map members ~f:(fun member -> member.backend) |> sorted in
        if List.equal String.equal actual backends then None else Some (family, actual))
  in
  List.iter incomplete ~f:(fun (family, actual) ->
      eprintf "%s: backend golden family is incomplete\n" family;
      eprintf "  expected backends: [%s]\n" (String.concat ~sep:"; " backends);
      eprintf "  actual backends:   [%s]\n" (String.concat ~sep:"; " actual));
  let provenance, marker_errors =
    List.filter arguments ~f:(fun (relative, _) ->
        String.equal (Stdlib.Filename.basename relative) "dune")
    |> List.fold ~init:([], []) ~f:(fun (provenance, errors) (relative, path) ->
        In_channel.read_lines path
        |> List.foldi ~init:(provenance, errors) ~f:(fun line_index (provenance, errors) line ->
            match
              marker_of_line ~backends ~dune_path:relative ~line_number:(line_index + 1) line
            with
            | Ok None -> (provenance, errors)
            | Ok (Some marker) -> (marker :: provenance, errors)
            | Error error -> (provenance, error :: errors)))
  in
  let members_by_path =
    List.fold members
      ~init:(Map.empty (module String))
      ~f:(fun by_path member -> Map.add_exn by_path ~key:member.path ~data:member)
  in
  let marker_targets =
    List.fold provenance
      ~init:(Map.empty (module String))
      ~f:(fun targets marker -> Map.add_multi targets ~key:marker.member ~data:marker)
  in
  let provenance_errors =
    List.concat_map (Map.to_alist marker_targets) ~f:(fun (path, markers) ->
        let duplicate =
          if List.length markers > 1 then
            [ Printf.sprintf "%s: more than one provenance marker names this member" path ]
          else []
        in
        match Map.find members_by_path path with
        | None ->
            Printf.sprintf "%s: provenance marker target is not a backend golden family member" path
            :: duplicate
        | Some member when String.equal member.backend (List.hd_exn markers).recorded_on ->
            Printf.sprintf
              "%s: provenance marker records the member on its own backend; remove the marker" path
            :: duplicate
        | Some _ -> duplicate)
  in
  let errors = List.rev_append member_errors (List.rev_append marker_errors provenance_errors) in
  List.iter errors ~f:(eprintf "%s\n");
  printf "Backend golden families:\n";
  Map.keys families |> List.iter ~f:(printf "  %s\n");
  printf "\nMembers not yet recorded on their own backend:\n";
  List.sort provenance ~compare:(fun a b -> String.compare a.member b.member)
  |> List.iter ~f:(fun { member; recorded_on; reason } ->
      printf "  %s <- %s -- %s\n" member recorded_on reason);
  printf "\n";
  let complete =
    (not (Map.is_empty families)) && List.is_empty incomplete && List.is_empty errors
  in
  Verdict.p "backend golden families are complete and provenance markers are valid" complete
