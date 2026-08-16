(* gh-ocannl-597: every stanza that runs a test executable declares the shared ocannl_config.

   A test resolves its configuration by walking UP from its working directory, which under dune is
   [_build/default/<test dir>]; `(copy_files ../config/ocannl_config)` puts the shared config file
   there, but only for the stanzas that DEPEND on it. Nothing is sandboxed, so a stanza that omits
   the dep does not fail -- it reads whatever is in the directory when it happens to run. The test
   then gets default `print_decimals_precision`/`fixed_state_for_init` and a config-sourcing trace
   on stderr, or not, depending on run order.

   gh-ocannl-586 fixed seven stanzas that had drifted that way and left a sentence in CLAUDE.md
   standing between the project and the eighth -- which is the same guarantee the seven had, none.
   Seventeen candidates had accumulated unnoticed, which is the measure of how quiet the failure
   is. This test is the forcing function instead.

   What it checks, over every dune file in the repository:

   - every `(test)`/`(tests)` stanza, and every `(library)` with `(inline_tests)`, lists
     `ocannl_config` in the deps dune uses when it runs them;
   - every `(rule ...)` that runs an executable does too -- the predicate that carries the content,
     because an `(executable)` stanza has no `deps` field at all, so the dep belongs on its
     companion rule;
   - and a directory with any such stanza has an `ocannl_config` to depend on in the first place.

   An `(executable)` no rule runs is structurally not a site and needs no exemption: that is how
   the diagnostic and tutorial executables (`bench_circles_step`, `gpt2_generate`, the `@slow`
   runners' companions) stay off this list without anyone maintaining one. The one genuine
   exemption is named below, with its reason. *)

open Base
open Stdio
module Scan = Test_utils.Dune_stanza_scan

(* A rule that runs an executable which reads no configuration. Keyed by "<dir>:<exe>", and each
   entry has to earn its place on every run (see the staleness check): an exemption is a claim
   about what an executable links, and a claim that stops being true is not a free pass. *)
let exempt_sites =
  [
    ( "test/ppx:pp.exe",
      "the ppx driver, run to expand a source file and diff the expansion: `ppx_ocannl` links \
       base, ppxlib, str and einsum_parser -- no configuration reader -- so no `ocannl_config` \
       can reach its output" );
  ]

(* Dune runs this from the rule's own directory inside the build tree and hands it paths relative
   to that directory. `%{workspace_root}` arrives as the way back out ("../.."), so the number of
   its components says how many of the working directory's trailing components name the rule's
   directory -- which turns those paths into repository-relative ones without this test having to
   assume what the build directory is called. *)
let split_path path = String.split_on_chars path ~on:[ '/'; '\\' ]

let base_dir workspace_root =
  let depth = List.count (split_path workspace_root) ~f:(String.equal "..") in
  let cwd = List.filter (split_path (Stdlib.Sys.getcwd ())) ~f:(Fn.non String.is_empty) in
  List.drop cwd (max 0 (List.length cwd - depth))

let repo_relative base path =
  let components =
    List.fold (base @ split_path path) ~init:[] ~f:(fun acc component ->
        match component with
        | "" | "." -> acc
        | ".." -> ( match acc with _ :: rest -> rest | [] -> [])
        | component -> component :: acc)
  in
  String.concat ~sep:"/" (List.rev components)

let () =
  if Array.length Stdlib.Sys.argv < 2 then (
    eprintf "Usage: %s <workspace_root> <dune file or ocannl_config...>\n" Stdlib.Sys.argv.(0);
    Stdlib.exit 1);
  let base = base_dir Stdlib.Sys.argv.(1) in
  (* Reported repository-relative, opened as dune handed them over: the working directory is the
     rule's own, deep in the build tree. *)
  let paths =
    Array.to_list (Array.subo Stdlib.Sys.argv ~pos:2)
    |> List.map ~f:(fun path -> (repo_relative base path, path))
    |> List.dedup_and_sort ~compare:(fun (a, _) (b, _) -> String.compare a b)
  in
  let of_basename name =
    List.filter paths ~f:(fun (path, _) -> String.equal (Stdlib.Filename.basename path) name)
  in
  let dune_files = of_basename "dune" in
  (* Where an `ocannl_config` resolves: a file checked in next to the dune file, or one a
     `(copy_files ...)` puts there -- the glob that supplies these paths matches dune's targets as
     well as its sources, so both arrive the same way. The stanza is read too, so that a directory
     whose config has simply never been built still counts as having one. *)
  let config_dirs =
    of_basename Scan.config_file
    |> List.map ~f:(fun (path, _) -> Stdlib.Filename.dirname path)
    |> Set.of_list (module String)
  in
  if List.is_empty dune_files then (
    printf "FAIL: no dune files among the arguments -- the rule's globs match nothing\n";
    Stdlib.exit 1);
  let ok = ref true in
  let fail msg =
    ok := false;
    printf "FAIL: %s\n" msg
  in
  let exemptions = Map.of_alist_exn (module String) exempt_sites in
  let exemptions_used = ref (Set.empty (module String)) in
  let counts = Hashtbl.create (module String) in
  let bump kind = Hashtbl.update counts kind ~f:(fun n -> 1 + Option.value n ~default:0) in
  List.iter dune_files ~f:(fun (dune_file, on_disk) ->
      let dir = Stdlib.Filename.dirname dune_file in
      let content = In_channel.read_all on_disk in
      let sites = Scan.sites content in
      (* A stanza kind the scan cannot place might carry an action that runs a test executable, so
         it is reported rather than counted as nothing (Codex P2, round 1). *)
      List.iter (Scan.unclassified_heads content) ~f:(fun head ->
          fail
            (Printf.sprintf
               "%s has a `(%s ...)` stanza, which this check has no classification for -- add it \
                to Dune_stanza_scan.action_heads if it can run an executable, or to inert_heads if \
                it cannot"
               dune_file head));
      (* A `(subdir …)` stanza runs elsewhere, so it is that directory's config it reaches for. *)
      let copies = Set.of_list (module String) (Scan.config_copy_dirs content) in
      let directory_of site = Scan.in_subdir dir site.Scan.subdir in
      List.map sites ~f:(fun site -> (site.Scan.subdir, directory_of site))
      |> List.dedup_and_sort ~compare:Poly.compare
      |> List.iter ~f:(fun (subdir, directory) ->
             if not (Set.mem config_dirs directory || Set.mem copies subdir) then
               fail
                 (Printf.sprintf
                    "%s runs test executables in %s, which has no %s to depend on -- add \
                     `(copy_files ../config/%s)`"
                    dune_file directory Scan.config_file Scan.config_file));
      let described =
        List.map sites ~f:(fun ({ Scan.kind; name; declares_config; subdir = _ } as site) ->
            let key = directory_of site ^ ":" ^ name in
            (* An exemption is spent only where the dep is actually absent, so declaring it anyway
               makes the entry stale and the list gets pruned rather than growing quietly. *)
            let exempt =
              match kind with
              | Scan.Runs_executable when (not declares_config) && Map.mem exemptions key ->
                  exemptions_used := Set.add !exemptions_used key;
                  true
              | _ -> false
            in
            (match kind with
            (* Declaring the dep does not settle this one: what the rule runs is unknown, so
               whether it needs the dep is unknown too (Codex P2, round 2). *)
            | Scan.Unreadable_command ->
                fail
                  (Printf.sprintf
                     "%s runs `%s`, which this check cannot read -- name the executable in a way \
                      Dune_stanza_scan.classify_command places (a path, `%%{bin:…}`, or a named \
                      dependency), or teach it that spelling"
                     dune_file name)
            | Scan.Unclassified_action ->
                fail
                  (Printf.sprintf
                     "%s has a `(%s ...)` action, which this check has no classification for -- \
                      add it to Dune_stanza_scan.program_actions if it executes a program, or to \
                      inert_actions if it does not"
                     dune_file name)
            | _ ->
                if exempt then bump "exempt"
                else if declares_config then bump (Scan.kind_name kind)
                else
                  fail
                    (Printf.sprintf "%s: the %s %s does not declare %s in its deps" dune_file
                       (Scan.kind_name kind) name Scan.config_file));
            (kind, name, exempt))
      in
      let tally kind =
        List.count described ~f:(fun (k, _, exempt) -> Poly.equal k kind && not exempt)
      in
      let exempted = List.filter described ~f:(fun (_, _, exempt) -> exempt) in
      let exempt_names =
        List.map exempted ~f:(fun (_, name, _) -> name)
        |> List.dedup_and_sort ~compare:String.compare
        |> String.concat ~sep:", "
      in
      let counted =
        [
          (tally Scan.Test, "test", "tests");
          (tally Scan.Inline_tests, "inline-test library", "inline-test libraries");
          (tally Scan.Runs_executable, "exe-running rule", "exe-running rules");
          ( List.length exempted,
            "exempt rule (" ^ exempt_names ^ ")",
            "exempt rules (" ^ exempt_names ^ ")" );
        ]
        |> List.filter_map ~f:(fun (n, singular, plural) ->
               if n = 0 then None
               else Some (Printf.sprintf "%d %s" n (if n = 1 then singular else plural)))
      in
      printf "%s: %s\n" dune_file
        (if List.is_empty counted then "nothing that runs a test executable"
         else String.concat ~sep:", " counted));
  let stale =
    Set.diff (Set.of_list (module String) (List.map exempt_sites ~f:fst)) !exemptions_used
  in
  if not (Set.is_empty stale) then
    fail
      (Printf.sprintf
         "exempted sites that no rule runs any more -- drop them from the exemption list: %s"
         (String.concat ~sep:", " (Set.to_list stale)));
  (* The reviewable part: an exemption is a claim about what an executable links, which no scan can
     check, so it is printed rather than merely held. *)
  printf "\nStanza kinds that can run a test executable: %s (plus `test`, `tests` and a `library`'s \
          `inline_tests`, which dune runs itself).\n"
    (String.concat ~sep:", " Scan.action_heads);
  printf "Stanza kinds that cannot: %s. Anything else fails above.\n"
    (String.concat ~sep:", " Scan.inert_heads);
  printf "Actions that execute a program: %s.\n" (String.concat ~sep:", " Scan.program_actions);
  printf "Actions that do not: %s. Anything else fails above.\n"
    (String.concat ~sep:", " Scan.inert_actions);
  printf "\nExempt from the dependency, running an executable that reads no configuration:\n";
  List.iter exempt_sites ~f:(fun (key, why) -> printf "  %s -- %s\n" key why);
  let count kind = Option.value (Hashtbl.find counts kind) ~default:0 in
  if !ok then
    printf
      "\nOK: %d dune files; %d test stanzas, %d inline-test libraries and %d exe-running rules \
       declare %s; %d exempt.\n"
      (List.length dune_files) (count "test") (count "inline tests") (count "rule running")
      Scan.config_file (count "exempt")
