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

   `-n` parses and executes nothing, so this costs one short-lived process per script and cannot run
   anything the scripts do. That property is load-bearing rather than incidental, and two rules
   below exist to keep it: only shells whose `-n` means "parse only" are ever invoked, and only
   shebang options that cannot redirect what is read are carried through.

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

   {1 The shebang is a command line, not a name}

   The first version of this check read the shebang for a word and threw the rest away, which is
   wrong in three ways that all have the same shape (Codex review round 2 on PR #454, three P2s):
   `#!/usr/bin/env -S -u FOO bash` names `FOO` as the interpreter if you take the first token
   without a leading dash, since `-u` has an operand; `#!/usr/bin/env -S bash -O extglob` parses
   only with `-O extglob`, so dropping it turns a valid script into a reported syntax error; and
   `#!/bin/dash` collapsed to `sh` is checked by whatever the host calls `sh`, which on macOS is
   bash -- so `function f() { :; }`, which dash rejects, would be reported as parsing.

   So {!Shebang.parse} reads the line as the command line it is: `env`'s own options, operands and
   `NAME=VALUE` assignments are consumed, the interpreter is whatever `env` would exec, and the
   interpreter's own options are carried into the check. What it will not do is guess. An
   interpreter outside {!parse_only_shells} is refused rather than run, because `-n` means "parse
   only" for shells and something else entirely for a `python3` that a `.sh` file might name; an
   option outside {!carried_options}/{!dropped_options} is refused too, because `-c` and `-s` make
   the shell read its program from somewhere other than the file, which would turn this check into
   an execution. Refused means a failing verdict naming the token, not a silent pass.

   {!shebang_cases} pins that parser on synthetic lines -- the three above among them -- since the
   repository's own scripts exercise two shapes of the grammar and would not notice the rest
   regressing.

   {1 What the golden holds}

   The parser table, then one line per script, so that a script dropping out of the scan is visible
   in the diff, plus two claims that are NOT promotable: a floor on how many scripts were reached,
   and the presence of the two scripts named above. The diff alone would not do -- a scan that finds
   nothing produces an empty golden, and `dune promote` would record it.

   Which shell checked which script goes to stderr, which a `(test)` diff does not read, because it
   is machine-dependent: a host without `dash` checks a dash script with another POSIX shell. *)

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

module Shebang = struct
  (** What a script's first line says about how to parse it. *)
  type t =
    | Sourced
        (** No shebang: the file is sourced rather than run, so it has no interpreter of its own and
            must parse under whichever shell sources it. *)
    | Interp of string * string list
        (** The interpreter the kernel (or `env`) would exec, and the options that have to be given
            to it for the file to parse as written. *)

  (** Shells whose `-n` parses without executing. The list is a whitelist rather than a fallthrough
      because `-n` is not a shared convention outside this family -- `python3 -n` is an error and
      `perl -n` wraps the program in a read loop and RUNS it -- and a `.sh` file is free to name one
      of those. *)
  let parse_only_shells = [ "sh"; "bash"; "dash"; "ash"; "ksh"; "mksh"; "zsh" ]

  (** POSIX shells that can stand in for one another when the named one is not installed. A stand-in
      is a weaker check, never a wrong one: these accept a subset of what bash does, so a script
      written for one of them and parsed by another in the family is being held to about the same
      grammar. bash is deliberately absent -- see [stand_ins]. *)
  let posix_family = [ "sh"; "dash"; "ash" ]

  (** Interpreter options carried into the parse check, because they change what parses. `-O`/`+O`
      is bash's shopt (`extglob` being the one that matters here), `-o`/`+o` is `set`'s (`posix`),
      and `--posix` is the same switch spelled long. Each entry says whether the option takes a
      separate operand. *)
  let carried_options = [ ("-O", true); ("+O", true); ("-o", true); ("+o", true); ("--posix", false) ]

  (** Interpreter options dropped, with a note on stderr: `#!/bin/bash -eu` is a common idiom and
      none of these reaches the grammar. Single letters, so that a cluster (`-eu`) can be taken
      apart. Anything outside this set and [carried_options] is refused rather than assumed
      harmless -- `-c` and `-s` in particular make the shell read its program from the command line
      or stdin, which would make `-n` a parse of something other than the file. *)
  let dropped_options = [ 'e'; 'u'; 'x'; 'v'; 'f'; 'h'; 'k'; 'm'; 'b'; 'B'; 'C'; 'E'; 'H'; 'T'; 'p' ]

  let basename p = List.last_exn (String.split_on_chars p ~on:[ '/'; '\\' ])

  let words line =
    List.filter (String.split_on_chars line ~on:[ ' '; '\t' ]) ~f:(Fn.non String.is_empty)

  (** Consume `env`'s own arguments and return what it would exec. GNU `env` takes `-i`/`-0` and
      friends without an operand, `-u NAME` and `-C DIR` with one (attached, separate, or as
      `--unset=NAME`), `-S`/`--split-string` whose payload is the command line itself, `--` to end
      its options, and any number of `NAME=VALUE` assignments before the command. Only the last of
      those was handled before, which is the whole of the first review finding.

      `-S`'s payload arrives already split here, because a shebang is one argument to the kernel and
      this reads the line by whitespace either way; quoting inside `-S` is therefore not honoured,
      which is noted rather than fixed since no shebang in this repository uses it. *)
  let rec skip_env_arguments = function
    | [] -> Error "`env` with no command"
    | "--" :: rest -> (
        match rest with [] -> Error "`env --` with no command" | cmd :: args -> Ok (cmd, args))
    | ("-S" | "--split-string") :: rest -> skip_env_arguments rest
    | ("-i" | "--ignore-environment" | "-0" | "--null" | "-v" | "--debug") :: rest ->
        skip_env_arguments rest
    | ("-u" | "--unset" | "-C" | "--chdir") :: rest -> (
        (* Operand in the next word. *)
        match rest with [] -> Error "`env` option with no operand" | _ :: rest -> skip_env_arguments rest)
    | token :: rest when String.is_prefix token ~prefix:"--split-string=" -> (
        match words (String.drop_prefix token (String.length "--split-string=")) @ rest with
        | [] -> Error "`env --split-string=` with no command"
        | split -> skip_env_arguments split)
    | token :: rest
      when List.exists [ "--unset="; "--chdir=" ] ~f:(fun p -> String.is_prefix token ~prefix:p)
           || (String.is_prefix token ~prefix:"-"
              && (not (String.is_prefix token ~prefix:"--"))
              && String.length token > 2
              && List.mem [ 'u'; 'C' ] token.[1] ~equal:Char.equal) ->
        (* `--unset=NAME`, `--chdir=DIR`, `-uNAME`, `-CDIR`: operand attached. *)
        skip_env_arguments rest
    | token :: rest when String.contains token '=' && not (String.is_prefix token ~prefix:"-") ->
        (* A `NAME=VALUE` assignment, which may precede the command. *)
        skip_env_arguments rest
    | token :: _ when String.is_prefix token ~prefix:"-" ->
        Error (Printf.sprintf "`env` option this check does not know: %s" token)
    | cmd :: args -> Ok (cmd, args)

  (** Classify one interpreter option: carried through, dropped, or refused. *)
  let rec classify_options acc = function
    | [] -> Ok (List.rev acc)
    | token :: rest -> (
        match List.Assoc.find carried_options token ~equal:String.equal with
        | Some true -> (
            match rest with
            | [] -> Error (Printf.sprintf "%s with no operand" token)
            | operand :: rest -> classify_options (operand :: token :: acc) rest)
        | Some false -> classify_options (token :: acc) rest
        | None ->
            if
              String.is_prefix token ~prefix:"-"
              && (not (String.is_prefix token ~prefix:"--"))
              && String.length token > 1
              && String.for_all (String.drop_prefix token 1) ~f:(fun c ->
                     List.mem dropped_options c ~equal:Char.equal)
            then (* A cluster of runtime-only flags, `-eu`: none of them reaches the grammar. *)
              classify_options acc rest
            else Error (Printf.sprintf "interpreter option this check does not know: %s" token))

  let parse first_line =
    match String.chop_prefix (String.strip first_line) ~prefix:"#!" with
    | None -> Ok Sourced
    | Some rest -> (
        match words rest with
        | [] -> Ok Sourced
        | first :: args ->
            let resolved =
              if String.equal (basename first) "env" then skip_env_arguments args
              else Ok (first, args)
            in
            Result.bind resolved ~f:(fun (cmd, args) ->
                let shell = basename cmd in
                if not (List.mem parse_only_shells shell ~equal:String.equal) then
                  Error
                    (Printf.sprintf "interpreter whose `-n` this check cannot vouch for: %s" shell)
                else Result.map (classify_options [] args) ~f:(fun opts -> Interp (shell, opts))))

  (** How a parse reads in a verdict label, so that the golden is a table of what the parser does
      rather than a column of booleans. *)
  let render = function
    | Ok Sourced -> "sourced"
    | Ok (Interp (shell, [])) -> shell
    | Ok (Interp (shell, opts)) -> String.concat ~sep:" " (shell :: opts)
    | Error reason -> "refused (" ^ reason ^ ")"

  (** The grammar, as a table of lines and what {!parse} must make of them -- rendered by
      {!render}, so a case reads as the specification rather than as a boolean.

      The first three entries are the review findings of PR #454 verbatim. The rest cover what
      those three imply: every `env` option form (attached operand, separate operand, long
      `--unset=`, `--` terminator, `-S` with an assignment), the interpreter options carried and
      dropped, and the two refusals that keep this check from ever executing a script -- an
      interpreter whose `-n` is not a parse, and an option that would redirect what is read. *)
  let shebang_cases =
    [
      (* The three findings. *)
      ("#!/usr/bin/env -S -u FOO bash", "bash");
      ("#!/usr/bin/env -S bash -O extglob", "bash -O extglob");
      ("#!/bin/dash", "dash");
      (* The shapes this repository's own scripts use. *)
      ("#!/bin/bash", "bash");
      ("#!/usr/bin/env bash", "bash");
      ("#!/bin/sh", "sh");
      ("", "sourced");
      (* The rest of `env`'s argument grammar. *)
      ("#!/usr/bin/env -S FOO=bar bash", "bash");
      ("#!/usr/bin/env -u FOO bash", "bash");
      ("#!/usr/bin/env -uFOO bash", "bash");
      ("#!/usr/bin/env --unset=FOO bash", "bash");
      ("#!/usr/bin/env -C /tmp bash", "bash");
      ("#!/usr/bin/env -i bash", "bash");
      ("#!/usr/bin/env -- bash", "bash");
      ("#!/usr/bin/env --split-string=bash -o posix", "bash -o posix");
      (* Interpreter options: carried when they reach the grammar, dropped when they cannot. *)
      ("#!/bin/bash --posix", "bash --posix");
      ("#!/bin/bash -eu", "bash");
      ("#!/bin/bash -O extglob -e", "bash -O extglob");
      (* Refusals. Both would otherwise be silent: the first parses a Python file with a shell's
         grammar, the second makes `-n` a parse of the command line rather than of the file. *)
      ("#!/usr/bin/env python3", "refused (interpreter whose `-n` this check cannot vouch for: python3)");
      ("#!/bin/bash -c true", "refused (interpreter option this check does not know: -c)");
      ("#!/usr/bin/env -Z bash", "refused (`env` option this check does not know: -Z)");
    ]
end

(** Run [prog args… -n path] with stdin closed and both output streams captured, and return the exit
    status together with what the shell said. A missing [prog] arrives as exit 127 on Unix (the
    forked child cannot exec and exits with it) and as [Unix_error] on Windows; both mean the same
    thing here, so both become [None]. *)
let parse_check prog args path =
  let tmp = Stdlib.Filename.temp_file "ocannl_shell_parse" ".log" in
  let devnull = if Stdlib.Sys.win32 then "NUL" else "/dev/null" in
  Exn.protect
    ~finally:(fun () -> try Stdlib.Sys.remove tmp with _ -> ())
    ~f:(fun () ->
      let out = Unix.openfile tmp [ Unix.O_WRONLY; Unix.O_TRUNC ] 0o600 in
      let inp = Unix.openfile devnull [ Unix.O_RDONLY ] 0o400 in
      let argv = Array.of_list ((prog :: args) @ [ "-n"; path ]) in
      let status =
        Exn.protect
          ~finally:(fun () ->
            Unix.close out;
            Unix.close inp)
          ~f:(fun () ->
            match Unix.create_process prog argv inp out out with
            | pid -> Some (snd (Unix.waitpid [] pid))
            | exception Unix.Unix_error _ -> None)
      in
      match status with
      | None | Some (Unix.WEXITED 127) -> None
      | Some status -> Some (status, String.strip (In_channel.read_all tmp)))

(** Whether [prog] can be used as a parse checker at all, decided by handing it a file that is a
    valid script in every shell there is. Probed once per shell name. *)
let available =
  let cache = Hashtbl.create (module String) in
  fun prog ->
    Hashtbl.find_or_add cache prog ~default:(fun () ->
        let tmp = Stdlib.Filename.temp_file "ocannl_shell_probe" ".sh" in
        Exn.protect
          ~finally:(fun () -> try Stdlib.Sys.remove tmp with _ -> ())
          ~f:(fun () ->
            Out_channel.write_all tmp ~data:":\n";
            match parse_check prog [] tmp with Some (Unix.WEXITED 0, _) -> true | _ -> false))

(** The shells that may stand in for [shell] when it is not installed, in preference order.

    Only within the POSIX family, and bash is not a member of it in either direction. Checking a
    dash script with bash is what the review objected to (bash accepts what dash rejects, so the
    check passes vacuously); checking a BASH script with dash is worse in the other direction --
    dash rejects arrays, `[[`, and process substitution, so every bashism becomes a reported syntax
    error in a script that is perfectly valid. bash's absence is therefore a failure, not a
    substitution, and so is ksh's and zsh's. The narrow remaining case -- `sh` where the host has
    only `dash`, or the reverse -- is a genuine equivalence, and it is still announced. *)
let stand_ins shell =
  if List.mem Shebang.posix_family shell ~equal:String.equal then
    List.filter Shebang.posix_family ~f:(Fn.non (String.equal shell))
  else []

(** Resolve one wanted shell to one that exists here, announcing any substitution. *)
let resolve ~rel shell =
  if available shell then Some shell
  else
    match List.find (stand_ins shell) ~f:available with
    | Some substitute ->
        eprintf "%s: no `%s` on this host, parsing with `%s` instead\n" rel shell substitute;
        Some substitute
    | None -> None

let () =
  if Array.length Stdlib.Sys.argv < 2 then (
    eprintf "Usage: %s <workspace_root> <ocannl_config and shell scripts...>\n" Stdlib.Sys.argv.(0);
    Stdlib.exit 1);
  (* The shebang grammar, on lines built to break it rather than on the two shapes this
     repository's scripts happen to use. Each case reads as what the parser must make of the line,
     so the golden is the specification. *)
  List.iter Shebang.shebang_cases ~f:(fun (line, expected) ->
      let actual = Shebang.render (Shebang.parse line) in
      if not (String.equal actual expected) then
        eprintf "shebang %S read as `%s`\n" line actual;
      Verdict.pf "shebang %S reads as `%s`" line expected (String.equal actual expected));
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
      match Shebang.parse first_line with
      | Error reason -> Verdict.fail (Printf.sprintf "%s: %s" rel reason)
      | Ok parsed ->
          let wanted =
            match parsed with
            | Shebang.Sourced -> [ ("sh", []); ("bash", []) ]
            | Shebang.Interp (shell, opts) -> [ (shell, opts) ]
          in
          let usable =
            List.filter_map wanted ~f:(fun (shell, opts) ->
                Option.map (resolve ~rel shell) ~f:(fun shell -> (shell, opts)))
          in
          if List.is_empty usable then
            Verdict.fail
              (Printf.sprintf "no shell on this host can parse %s (wanted %s)" rel
                 (String.concat ~sep:", " (List.map wanted ~f:fst)))
          else
            let complaints =
              List.filter_map usable ~f:(fun (prog, opts) ->
                  match parse_check prog opts path with
                  | None ->
                      Some (Printf.sprintf "`%s` disappeared between the probe and the check" prog)
                  | Some (Unix.WEXITED 0, _) -> None
                  | Some (_, said) -> Some (Printf.sprintf "`%s -n` said: %s" prog said))
            in
            eprintf "%s: parsed with %s\n" rel
              (String.concat ~sep:", "
                 (List.map usable ~f:(fun (prog, opts) ->
                      String.concat ~sep:" " (prog :: opts))));
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
