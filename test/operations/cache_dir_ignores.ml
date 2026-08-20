(* Every schedule cache directory a source names carries the prefix the ignore list globs on.

   An autotune search saves its schedules into a directory named relative to the working directory.
   Under `dune runtest` that is inside `_build/`, so a test naming one leaves nothing behind in the
   repository -- until someone runs the test executable from the repository root by hand, which is
   how the tuning tests are actually developed, and the directory appears there untracked. The root
   `.gitignore` used to carry an entry per directory, and nothing but a person noticing kept the two
   in step: two entries had drifted out and were added only when a hand run on one machine surfaced
   them, one of them months after the test landed.

   A list that has to be maintained is the wrong shape for that. The ignore list is one glob over a
   name prefix instead, and what this check holds is the premise the glob rests on: every
   `~cache_dir` names a directory carrying the prefix, so a new tuning test is ignored the day it
   is written and neither the ignore list nor this test's golden has anything to say about it.

   Two requirements, over every OCaml source in the repository:

   - every `~cache_dir` argument resolves to a string literal -- a directory carrying the prefix,
     or the empty string, which turns the cache off -- or to a parameter forwarded from a call site
     scanned in its own right. A spelling that resolves to none of those is reported rather than
     assumed harmless;
   - and the root `.gitignore` still carries the glob, so that the first requirement is not being
     enforced in support of a rule that has been edited away.

   The built-in default needs no special case: the prefix IS its name, so a library that stops
   naming it fails the second requirement rather than slipping past this one.

   How a directory name is recovered from a source -- and why only a parse can do it -- is
   {!Cache_dir_scan}. *)

open Base
open Stdio
module Scan = Test_utils.Cache_dir_scan

let base_dir = Test_utils.Dune_stanza_scan.base_dir
let repo_relative = Test_utils.Dune_stanza_scan.repo_relative

let () =
  if Array.length Stdlib.Sys.argv < 2 then (
    eprintf "Usage: %s <workspace_root> <.gitignore and source files...>\n" Stdlib.Sys.argv.(0);
    Stdlib.exit 1);
  let base = base_dir Stdlib.Sys.argv.(1) in
  (* Reported repository-relative, opened as dune handed them over: the working directory is the
     rule's own, deep in the build tree. *)
  let paths =
    Array.to_list (Array.subo Stdlib.Sys.argv ~pos:2)
    |> List.map ~f:(fun path -> (repo_relative base path, path))
    |> List.dedup_and_sort ~compare:(fun (a, _) (b, _) -> String.compare a b)
  in
  (* The ROOT ignore file, by its repository-relative path rather than its basename: the glob is
     root-anchored, and a `.gitignore` in a subdirectory anchors to that subdirectory instead. *)
  let ignore_file = List.Assoc.find paths ".gitignore" ~equal:String.equal in
  (* The globs reach into the build tree, which holds derived copies beside the sources dune staged:
     a `.pp.ml` is a source after preprocessing, and a `*_actual.ml` under test/ppx is a rule's
     captured output -- sometimes a compiler error message rather than OCaml at all. Neither is a
     place a directory can be named that its own source does not already name, and whether either
     exists depends on what has been built. Anything else that fails to parse is reported by name
     below rather than taking the run down.

     This check's own source is NOT excluded: a check that skipped it would carve out the one place
     a directory could be named without the prefix rule being applied to it. *)
  let derived path =
    String.is_suffix path ~suffix:".pp.ml" || String.is_suffix path ~suffix:"_actual.ml"
  in
  let sources =
    List.filter paths ~f:(fun (path, _) ->
        String.is_suffix path ~suffix:".ml" && not (derived path))
  in
  (* Failures go through [Verdict]: reported on both channels, and the run exits nonzero from its
     teardown -- so the exit status, not the promotable golden diff, carries the verdict
     (gh-ocannl-601). *)
  let fail = Verdict.fail in
  let ignore_file =
    match ignore_file with
    | Some on_disk -> on_disk
    | None ->
        fail
          "the repository-root .gitignore is not among the arguments -- the rule's dependency on \
           it is missing";
        Stdlib.exit 1
  in
  if List.is_empty sources then (
    fail "no OCaml sources among the arguments -- the rule's globs match nothing";
    Stdlib.exit 1);
  if not (Scan.declares_required_glob (In_channel.read_all ignore_file)) then
    fail
      (Printf.sprintf
         "the root .gitignore no longer carries `%s` -- without it nothing ignores the cache \
          directories the sources name, and the prefix this check enforces buys nothing"
         Scan.required_glob);
  let naming = ref [] in
  List.iter sources ~f:(fun (source, on_disk) ->
      let content = In_channel.read_all on_disk in
      match Or_error.try_with (fun () -> Scan.uses content) with
      | Error error ->
          fail
            (Printf.sprintf
               "%s is among the sources this check scans and does not parse as OCaml (%s) -- if it \
                is a build artifact rather than a source, exclude it beside the `.pp.ml` \
                expansions"
               source
               (String.concat ~sep:" " (String.split_lines (Error.to_string_hum error))))
      | Ok uses ->
          List.iter uses ~f:(fun { Scan.resolution; line } ->
              match resolution with
              | Scan.Names name when String.is_prefix name ~prefix:Scan.required_prefix ->
                  naming := source :: !naming
              | Scan.Names name ->
                  fail
                    (Printf.sprintf
                       "%s:%d names the cache directory %s, which does not start with `%s` -- \
                        rename it to `%s_…` so that the root .gitignore's `%s` covers it"
                       source line name Scan.required_prefix Scan.required_prefix Scan.required_glob)
              | Scan.Disabled | Scan.Forwarded _ -> ()
              | Scan.Unresolved _ ->
                  fail
                    (Printf.sprintf
                       "%s:%d: this `~%s` argument %s, which this scan cannot resolve to a literal \
                        -- pass the directory as a literal here, or bind it to a name mentioning \
                        `%s` whose right-hand side is one, so that the prefix can be checked"
                       source line Scan.label (Scan.describe resolution) Scan.label)));
  (* The sources that name one, not the names themselves: what the check establishes is a property
     every name has, so listing them would only make the golden churn on each new tuning test --
     the maintenance burden this check exists to remove. *)
  let naming = List.dedup_and_sort !naming ~compare:String.compare in
  printf "Sources naming a schedule cache directory, all with the `%s` prefix that `%s` covers:\n"
    Scan.required_prefix Scan.required_glob;
  List.iter naming ~f:(fun source -> printf "  %s\n" source);
  (* No count of sources SCANNED: the globs reach into the build tree, so how many `.ml` files they
     match depends on which `(select …)` copies have been built by the time this rule runs, and a
     golden pinning that would fail on build order rather than on anything true. *)
  if not (Verdict.any_failed ()) then
    printf "\nOK: %d sources name one; the ignore list needs no entry per directory.\n"
      (List.length naming)
