(** Response files for the repository-wide scans (Windows command-line limit).

    A scan is handed every file it reads, on the command line, by a dune rule whose [(deps ...)] are
    recursive globs. That is what makes the dependency edges right — dune reruns the scan when any
    scanned file changes — but it also makes the command line grow with the repository, and Windows
    caps a whole command line at 32,767 characters. Past it [CreateProcess] fails, and dune reports
    it as [Error: CreateProcess(): No such file or directory]: an error that names neither the
    length nor the executable, on whichever scan the last few merges happened to push over. Measured
    on the tree that first hit it, three scans were between 37,000 and 40,000 characters and a
    fourth was 255 short of the cap, so this is not a one-off to be waited out.

    A dune [(echo %{deps})] action runs inside dune and spawns nothing, so a rule can write its own
    dependency list to a file of any length and pass that file instead. This is the reading half: an
    argument spelled [@<path>] stands for the whitespace-separated words in that file, spliced in
    where it stood. The [@] convention is the one gcc, javac and ld already use, and splicing in
    place is what lets a scan keep its fixed leading arguments — [<root> @list] and [<root> a b c]
    reach the rest of the program as the same argv. *)

open Base

let expand argv =
  Array.to_list argv
  |> List.concat_map ~f:(fun arg ->
      match String.chop_prefix arg ~prefix:"@" with
      | None -> [ arg ]
      | Some path ->
          (* A response file this process cannot read is a broken rule, not an empty file list:
             answering with no arguments would make a scan report on nothing and pass. *)
          let contents =
            try Stdio.In_channel.read_all path
            with Stdlib.Sys_error message ->
              raise (Stdlib.Sys_error (Printf.sprintf "response file `%s`: %s" path message))
          in
          String.split_on_chars contents ~on:[ ' '; '\t'; '\n'; '\r' ]
          |> List.filter ~f:(Fn.non String.is_empty))
  |> Array.of_list
