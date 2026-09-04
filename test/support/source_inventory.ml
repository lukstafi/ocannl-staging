(** Source-only file inventories for repository scans (gh-ocannl-871).

    A repository scan declares [(sandbox always)] and [(source_tree ../..)] on its Dune rule, then
    calls {!of_dune_sandbox} with [%{workspace_root}] and the action-created files Dune placed in
    that sandbox (the scanner executable, redirected target, and copied [ocannl_config]). The clean
    sandbox is the boundary: it contains the declared source tree without checkout Git metadata or
    stale build outputs, keeps path boundaries in the filesystem instead of a whitespace-sensitive
    manifest, and puts no source list on Windows' length-limited command line.

    Callers select their semantic corpus from {!files}; they do not rediscover source membership
    with their own recursive globs. A new source directory therefore enters every migrated scan's
    inventory without extending a root allowlist. *)

open Base

type file = { path : string; on_disk : string }
type t = file list

let path_components path =
  String.split_on_chars path ~on:[ '/'; '\\' ] |> List.filter ~f:(Fn.non String.is_empty)

let normalized_components path =
  let path =
    if Stdlib.Filename.is_relative path then Stdlib.Filename.concat (Stdlib.Sys.getcwd ()) path
    else path
  in
  List.fold (path_components path) ~init:[] ~f:(fun reversed component ->
      match (component, reversed) with
      | ".", _ -> reversed
      | "..", _ :: rest -> rest
      | "..", [] -> []
      | component, _ -> component :: reversed)
  |> List.rev

let rec drop_prefix path prefix =
  match (path, prefix) with
  | path, [] -> Some path
  | p :: path, q :: prefix when String.equal p q -> drop_prefix path prefix
  | _ -> None

let relative_to_root ~root path =
  match drop_prefix (normalized_components path) (normalized_components root) with
  | Some relative -> String.concat ~sep:"/" relative
  | None ->
      invalid_arg
        (Printf.sprintf "source inventory path `%s` is outside its sandbox root `%s`" path root)

let vcs_metadata_component = function ".git" | ".hg" | ".svn" -> true | _ -> false

let rec regular_files path =
  if vcs_metadata_component (Stdlib.Filename.basename path) then []
  else
    match Unix.lstat path with
    | { Unix.st_kind = Unix.S_DIR; _ } ->
        Array.to_list (Stdlib.Sys.readdir path)
        |> List.concat_map ~f:(fun entry -> regular_files (Stdlib.Filename.concat path entry))
    | { Unix.st_kind = Unix.S_REG; _ } -> [ path ]
    | { Unix.st_kind = Unix.S_LNK; _ } -> (
        (* Dune preserves checked-in file symlinks in the source tree. They are source paths too;
           directory symlinks are not descended, avoiding an inventory walk outside the sandbox or
           around a cycle. *)
        match Unix.stat path with
        | { Unix.st_kind = Unix.S_REG; _ } -> [ path ]
        | _ -> []
        | exception Unix.Unix_error _ -> [])
    | _ -> []
    | exception Unix.Unix_error _ -> []

let of_dune_sandbox ~workspace_root ~generated =
  let generated =
    List.map generated ~f:(relative_to_root ~root:workspace_root) |> Set.of_list (module String)
  in
  regular_files workspace_root
  |> List.filter_map ~f:(fun on_disk ->
      let path = relative_to_root ~root:workspace_root on_disk in
      Option.some_if (not (Set.mem generated path)) { path; on_disk })
  |> List.dedup_and_sort ~compare:(fun a b -> String.compare a.path b.path)

let files inventory = inventory
let select inventory ~f = List.filter inventory ~f:(fun file -> f file.path)
let mem inventory path = List.exists inventory ~f:(fun file -> String.equal file.path path)
