(* Every shell script in the repository parses, checked by the shell its shebang names.

   Nothing else in `dune build @check` or `dune runtest` looks at shell at all, and the repository
   runs a fair amount of it in places where a syntax error is expensive and late-discovered:
   `scripts/setup-ocaml-env.sh` is the SessionStart hook, so a broken one greets every session with
   a failing hook rather than a failing test; `tools/test-run.sh` fronts every suite run, and its
   Windows-only branches are exercised so rarely that CI added a step of its own to keep them from
   rotting. During PR #438 `bash -n` was run by hand about a dozen times and caught real breakage
   twice -- an edit that produced `D_IGN_PARENT="93.$$"# comment` (a `#` that starts no comment
   there, so the line is not the assignment it reads as), and a scratch harness whose sed-driven
   mutations kept emitting invalid shell until it grew a `bash -n` guard of its own. Both were found
   because someone thought to run the parser; nothing made them run it.

   `bash -n` / `sh -n` parse and execute nothing, so this costs one short-lived process per script
   and cannot run anything the scripts do.

   {1 How the scripts are found}

   By the rule's `(glob_files_rec ../../*.sh)` dependency, not by `git ls-files`. The glob is what
   makes a new script covered the day it lands, and -- the part git cannot do from inside a dune
   action -- it is also what makes dune RERUN this check when a script changes. A list recovered
   from git would leave the rule depending on nothing that moves when a script is edited, so the
   first run's pass would be served from cache forever after, which is the failure mode this
   repository keeps rediscovering. (`(universe)` would force the reruns, at the price of running
   every check on every dune invocation; the glob gets both properties for free.)

   The glob is over dune's view of the source tree, which skips dot-directories and `_build`. It
   therefore also reaches scripts git does not track -- deliberately: a scratch harness that keeps
   generating invalid shell is exactly the case PR #438 hit, and one that lives in `tools/` for an
   afternoon is worth the same parse. What it cannot see is a tracked script under a dot-directory
   (`.github/`); the floor below is what keeps a scan that has gone blind from passing quietly.

   {1 What the golden holds}

   One line per script, so that a script dropping out of the scan is visible in the diff, plus two
   claims that are NOT promotable: a floor on how many scripts were reached, and the presence of the
   two scripts named above. The diff alone would not do -- a scan that finds nothing produces an
   empty golden, and `dune promote` would record it.

   Which shell checked which script goes to stderr, which a `(test)` diff does not read, because it
   is machine-dependent: a host without `sh` falls back to `bash` and vice versa. *)

open Base
open Stdio

let base_dir = Test_utils.Dune_stanza_scan.base_dir
let repo_relative = Test_utils.Dune_stanza_scan.repo_relative

(* The scripts tracked when this check was written, and the floor it holds the scan to. A floor
   rather than a count: adding a script must not make anyone promote a number, while a scan that
   stops reaching a directory has to fail rather than shrink the golden. *)
let script_floor = 12

(* Named because their breakage is not discovered by a test run: the first is the SessionStart hook,
   the second is what runs the suite. If the scan can no longer see these two it is not seeing the
   repository, whatever else it found. *)
let must_be_scanned = [ "scripts/setup-ocaml-env.sh"; "tools/test-run.sh" ]

(** Run [prog -n path] with stdin closed and both output streams captured, and return the exit
    status together with what the shell said. A missing [prog] arrives as exit 127 on Unix (the
    forked child cannot exec and exits with it) and as [Unix_error] on Windows; both mean the same
    thing here, so both become [None]. *)
let parse_check prog path =
  let tmp = Stdlib.Filename.temp_file "ocannl_shell_parse" ".log" in
  let devnull = if Stdlib.Sys.win32 then "NUL" else "/dev/null" in
  Exn.protect
    ~finally:(fun () -> try Stdlib.Sys.remove tmp with _ -> ())
    ~f:(fun () ->
      let out = Unix.openfile tmp [ Unix.O_WRONLY; Unix.O_TRUNC ] 0o600 in
      let inp = Unix.openfile devnull [ Unix.O_RDONLY ] 0o400 in
      let status =
        Exn.protect
          ~finally:(fun () ->
            Unix.close out;
            Unix.close inp)
          ~f:(fun () ->
            match Unix.create_process prog [| prog; "-n"; path |] inp out out with
            | pid -> Some (snd (Unix.waitpid [] pid))
            | exception Unix.Unix_error _ -> None)
      in
      match status with
      | None | Some (Unix.WEXITED 127) -> None
      | Some status -> Some (status, String.strip (In_channel.read_all tmp)))

(** Whether [prog] can be used as a parse checker at all, decided by handing it a file that is a
    valid script in every shell there is. Probed once per shell name; a host missing one of them
    (Git Bash installs where only `bash` is on PATH) falls back to the other rather than failing. *)
let available =
  let cache = Hashtbl.create (module String) in
  fun prog ->
    Hashtbl.find_or_add cache prog ~default:(fun () ->
        let tmp = Stdlib.Filename.temp_file "ocannl_shell_probe" ".sh" in
        Exn.protect
          ~finally:(fun () -> try Stdlib.Sys.remove tmp with _ -> ())
          ~f:(fun () ->
            Out_channel.write_all tmp ~data:":\n";
            match parse_check prog tmp with Some (Unix.WEXITED 0, _) -> true | _ -> false))

(** The shells a script must parse under, from its first line.

    A shebang naming `env` defers to its argument (`#!/usr/bin/env bash`). A file with no shebang is
    one that is SOURCED rather than run -- `tools/opam-env.sh` says so in its own header -- so it has
    no interpreter of its own and must parse under whichever shell sources it: both, then. An
    interpreter this check does not know is reported rather than waved through, since guessing wrong
    would mean checking a zsh script against bash's grammar. *)
let checkers_for first_line =
  let interpreter =
    match String.chop_prefix (String.strip first_line) ~prefix:"#!" with
    | None -> None
    | Some rest -> (
        match List.filter (String.split_on_chars rest ~on:[ ' '; '\t' ]) ~f:(Fn.non String.is_empty)
        with
        | [] -> None
        | first :: args ->
            let name p = List.last_exn (String.split_on_chars p ~on:[ '/'; '\\' ]) in
            (* `env` may carry options of its own (`env -S`, `env -i`); the interpreter is the first
               argument that is not one. *)
            if String.equal (name first) "env" then
              List.find args ~f:(fun a -> not (String.is_prefix a ~prefix:"-"))
              |> Option.map ~f:name
            else Some (name first))
  in
  match interpreter with
  | None -> Ok [ "sh"; "bash" ]
  | Some "bash" -> Ok [ "bash" ]
  | Some ("sh" | "dash" | "ash") -> Ok [ "sh" ]
  | Some other -> Error other

let () =
  if Array.length Stdlib.Sys.argv < 2 then (
    eprintf "Usage: %s <workspace_root> <ocannl_config and shell scripts...>\n" Stdlib.Sys.argv.(0);
    Stdlib.exit 1);
  let base = base_dir Stdlib.Sys.argv.(1) in
  (* Reported repository-relative, opened as dune handed them over: the working directory is deep in
     the build tree and the paths arrive relative to it. *)
  let scripts =
    Array.to_list Stdlib.Sys.argv
    |> Fn.flip List.drop 2
    |> List.filter ~f:(fun path -> String.is_suffix path ~suffix:".sh")
    |> List.map ~f:(fun path -> (repo_relative base path, path))
    |> List.sort ~compare:(fun (a, _) (b, _) -> String.compare a b)
  in
  List.iter scripts ~f:(fun (rel, path) ->
      let first_line =
        In_channel.with_file path ~f:In_channel.input_line |> Option.value ~default:""
      in
      match checkers_for first_line with
      | Error interpreter ->
          Verdict.fail
            (Printf.sprintf "%s runs under %s, which this check does not know how to parse with"
               rel interpreter)
      | Ok wanted ->
          (* The fallback keeps a host with only one of the two shells honest rather than red: every
             script this repository has parses under bash, and a POSIX one that parses under bash is
             not thereby proven POSIX -- which is why the substitution is announced. *)
          let usable =
            List.filter_map wanted ~f:(fun prog ->
                if available prog then Some prog
                else
                  let substitute = if String.equal prog "sh" then "bash" else "sh" in
                  if available substitute then (
                    eprintf "%s: no `%s` on this host, parsing with `%s` instead\n" rel prog
                      substitute;
                    Some substitute)
                  else None)
            |> List.dedup_and_sort ~compare:String.compare
          in
          if List.is_empty usable then
            Verdict.fail
              (Printf.sprintf "no shell on this host can parse %s (tried %s)" rel
                 (String.concat ~sep:", " wanted))
          else
            let complaints =
              List.filter_map usable ~f:(fun prog ->
                  match parse_check prog path with
                  | None ->
                      Some (Printf.sprintf "`%s` disappeared between the probe and the check" prog)
                  | Some (Unix.WEXITED 0, _) -> None
                  | Some (_, said) -> Some (Printf.sprintf "`%s -n` said: %s" prog said))
            in
            eprintf "%s: parsed with %s\n" rel (String.concat ~sep:", " usable);
            List.iter complaints ~f:(fun complaint -> eprintf "  %s: %s\n" rel complaint);
            Verdict.pf "%s parses" rel (List.is_empty complaints));
  eprintf "shell scripts scanned: %d\n" (List.length scripts);
  Verdict.pf "the scan reached at least the %d shell scripts this repository is known to have"
    script_floor
    (List.length scripts >= script_floor);
  let scanned = Set.of_list (module String) (List.map scripts ~f:fst) in
  let missing = List.filter must_be_scanned ~f:(Fn.non (Set.mem scanned)) in
  List.iter missing ~f:(fun rel -> eprintf "not reached by the scan: %s\n" rel);
  Verdict.p "the scan reached the session hook and the suite runner" (List.is_empty missing)
