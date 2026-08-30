(** Live inventory enforcing caller-visible honesty for optional arguments in [lib/]
    (gh-ocannl-811). The classifier and its synthetic negative controls live in
    [Test_utils.Optional_arg_scan] and [optional_arg_scan_cases.ml]. *)

open Base
open Stdio
module Scan = Test_utils.Optional_arg_scan
module Dune = Test_utils.Dune_stanza_scan
module Sources = Test_utils.Config_key_scan

let read path = Stdlib.In_channel.with_open_bin path Stdlib.In_channel.input_all

let () =
  if Array.length Stdlib.Sys.argv < 3 then (
    eprintf "Usage: %s <workspace_root> <lib source...>\n" Stdlib.Sys.argv.(0);
    Stdlib.exit 1);
  let base = Dune.base_dir Stdlib.Sys.argv.(1) in
  let arguments =
    Array.to_list (Array.subo Stdlib.Sys.argv ~pos:2)
    |> List.map ~f:(fun path -> (Dune.repo_relative base path, path))
  in
  let on_disk = Map.of_alist_reduce (module String) arguments ~f:(fun first _ -> first) in
  let sources = Sources.sources_among (List.map arguments ~f:fst) in
  if List.is_empty sources then Verdict.fail "the lib/ source glob handed the inventory nothing";
  let args =
    List.concat_map sources ~f:(fun source ->
        let interface =
          String.chop_suffix source ~suffix:".ml"
          |> Option.bind ~f:(fun stem -> Map.find on_disk (stem ^ ".mli"))
          |> Option.map ~f:read
        in
        Scan.args_in_source ?interface ~source (read (Map.find_exn on_disk source)))
    |> List.sort ~compare:(fun a b -> String.compare (Scan.render a) (Scan.render b))
  in
  eprintf "Scanned %d lib/ sources and found %d public optional arguments.\n" (List.length sources)
    (List.length args);
  print_endline
    "Optional arguments accepted by lib/ entry points. `implemented` means the bound value has a\n\
     non-discard use; `unimplemented` must therefore be visible to callers as an underscore label.\n";
  List.iter args ~f:(fun arg -> printf "  %s\n" (Scan.render arg));
  print_endline "";
  Verdict.p_exists "the inventory found lib/ optional arguments" args ~f:(fun _ -> true);
  Verdict.p_all
    "every implemented option has an ordinary label and every unimplemented option an underscore"
    args ~f:Scan.honest
