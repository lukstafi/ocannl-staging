(* Whether a dune invocation can reach a stanza that names a GPU backend: the reachability half of
   tools/fleet-slot-run.sh's `--cpu`/`--gpu` decision (gh-ocannl-1004; the reasoning is
   Test_utils.Slot_kind's). Run from the repository root with the dune argv as its arguments; prints
   `cpu` when the run cannot reach one, or `gpu: <why>` when it can or when the tree cannot be
   read. *)

open Base
open Stdio

(* Every dune file dune itself would read: it skips directories whose name starts with `.` or `_`
   (_build, _opam, .git, ...). Reading more than dune does only widens the answer. *)
let rec dune_files dir =
  let path = if String.is_empty dir then "." else dir in
  let entries = Stdlib.Sys.readdir path |> Array.to_list |> List.sort ~compare:String.compare in
  let here =
    if List.mem entries "dune" ~equal:String.equal then
      [ (dir, In_channel.read_all (Stdlib.Filename.concat path "dune")) ]
    else []
  in
  here
  @ List.concat_map entries ~f:(fun e ->
      let sub = if String.is_empty dir then e else dir ^ "/" ^ e in
      if String.is_prefix e ~prefix:"." || String.is_prefix e ~prefix:"_" then []
      else if Stdlib.Sys.is_directory sub then dune_files sub
      else [])

let () =
  let argv = Array.to_list Stdlib.Sys.argv |> List.tl_exn in
  match dune_files "" with
  | exception exn -> printf "gpu: the source tree is unreadable here (%s)\n" (Exn.to_string exn)
  | files -> (
      match Test_utils.Slot_kind.verdict ~dune_files:files argv with
      | Ok None -> printf "cpu\n"
      | Ok (Some why) -> printf "gpu: %s\n" why
      | Error why -> printf "gpu: %s\n" why)
