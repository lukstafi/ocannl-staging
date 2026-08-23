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

   The glob is over dune's view of the source tree, which also reaches scripts git does not track --
   deliberately: a scratch harness that keeps generating invalid shell is exactly the case PR #438
   hit, and one that lives in `tools/` for an afternoon is worth the same parse.

   Dune's scan skips dot-directories, which used to leave `.github/scripts/*.sh` uncovered, and the
   floor below did NOT catch that -- the twelve visible scripts keep it satisfied however many
   hidden ones are broken (Codex review round 3). The root `dune` names `.github` into the scan
   instead, and says there why it names that one and not `.claude/`. The floor is still worth having
   for what it does cover: a directory that drops out of the scan entirely.

   {1 The shebang is a command line, not a name}

   The first version of this check read the shebang for a word and threw the rest away, which is
   wrong in three ways that all have the same shape (Codex review round 2 on PR #454, three P2s):
   `#!/usr/bin/env -S -u FOO bash` names `FOO` as the interpreter if you take the first token
   without a leading dash, since `-u` has an operand; `#!/usr/bin/env -S bash -O extglob` parses
   only with `-O extglob`, so dropping it turns a valid script into a reported syntax error; and
   `#!/bin/dash` collapsed to `sh` is checked by whatever the host calls `sh`, which on macOS is
   bash -- so `function f() { :; }`, which dash rejects, would be reported as parsing.

   So {!Shebang.parse} reads the line as the command line it is -- but as the KERNEL builds it, which
   is the correction of round 3: everything after the interpreter path is ONE argument, not a word
   list. Only `env`'s `-S`/`--split-string` splits it, which is what that option exists for. The
   difference is not academic in either direction. `#!/usr/bin/env -u FOO bash` hands env the single
   argument `-u FOO bash`, so env unsets a variable named " FOO bash" and execs the script itself --
   measured here, it does not reach bash, it HANGS (the kernel re-enters the same shebang until it
   gives up). And `#!/bin/bash -O extglob` hands bash the single argument `-O extglob`, which bash
   rejects with "invalid option". Reading either as a word list made this check pin a broken script
   as valid; both are now refused, naming the kernel semantics and pointing at `-S`.

   Beyond that, what the parser will not do is guess. An interpreter outside {!parse_only_shells} is
   refused rather than run, because `-n` means "parse only" for shells and something else entirely
   for a `python3` that a `.sh` file might name -- `perl -n` wraps the program in a read loop and
   RUNS it. An option in {!Shebang.refused_short}/{!Shebang.refused_long} is refused too, because
   `-c` and `-s` make the shell read its program from somewhere other than the file and
   `--version` exits before reading it at all. Every other option is carried through to the shell,
   which is the authority on its own: a universal "harmless flags" list is a guess, and it was wrong
   for `#!/bin/dash -B`. An `env` shebang that assigns a variable or asks for `-0` output is refused
   for the same reason -- the first changes the environment the lookup happens in, and the second is
   an `env` invocation that exits 125 rather than running anything. Refused means a failing verdict
   naming the token, not a silent pass.

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
    | Interp of launch * string list
        (** The interpreter the kernel (or `env`) would exec, and the options that have to be given
            to it for the file to parse as written. *)

  (** How the interpreter is found, which is not the same question for the two shebang forms and
      cannot be flattened to a name (Codex review round 4).

      A direct `#!/bin/bash` names a FILE, and the kernel execs that file; resolving `bash` on PATH
      instead can run a different build entirely -- on macOS `/bin/bash` is 3.2 while a Homebrew
      `bash` 5 sits earlier on PATH, and 5 accepts `declare -A` and `${v^^}` that 3.2 rejects, so a
      script the kernel-selected shell would refuse passes. That skew is on this repository's own
      macOS CI leg, not hypothetical. A shebang going through `env`, by contrast, is a PATH lookup
      by definition -- that is what `env` is for -- so resolving the name is the faithful thing
      there. *)
  and launch =
    | Path of string  (** A direct shebang: exec this exact file, as the kernel would. *)
    | Name of string  (** Selected through `env`: resolve on PATH, as `env` would. *)

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

  (** Interpreter options that would make `-n` a parse of something other than this file, so a
      shebang carrying one is refused rather than checked. `-c` takes the program from the command
      line and `-s` from stdin; `--version`/`--help` exit before reading anything, which would make
      an unparsable script pass with status 0.

      Everything else is carried through verbatim, which is round 5's correction. The previous
      version dropped a fixed list of "runtime-only" flags as harmless, and that list was universal
      while the shells are not: `#!/bin/dash -B` had its `-B` dropped and was checked as plain
      `dash -n`, where the real invocation exits 2 with "Illegal option -B". Encoding each shell's
      option table here would be the same guess one level down, so the shell is left to be the
      authority on its own options -- with `-n` in play the carried flags cannot execute anything,
      so there is nothing to be gained by filtering them. *)
  let refused_short = [ 'c'; 's' ]

  let refused_long = [ "--version"; "--help" ]

  let basename p = List.last_exn (String.split_on_chars p ~on:[ '/'; '\\' ])

  let words line =
    List.filter (String.split_on_chars line ~on:[ ' '; '\t' ]) ~f:(Fn.non String.is_empty)

  (** Consume `env`'s own arguments and return what it would exec. GNU `env` takes `-i`/`-0` and
      friends without an operand, `-u NAME` and `-C DIR` with one (attached, separate, or as
      `--unset=NAME`), `-S`/`--split-string` whose payload is the command line itself, `--` to end
      its options, and any number of `NAME=VALUE` assignments before the command. Only the last of
      those was handled before, which is the whole of the first review finding.

      This runs only over an argument list `-S` produced, since without `-S` the kernel hands env a
      single argument and {!parse} refuses that shape before getting here. Quoting inside `-S` is not
      honoured -- the payload is split on whitespace -- which is noted rather than fixed, since no
      shebang in this repository uses `-S` at all. *)
  let rec skip_env_arguments = function
    | [] -> Error "`env` with no command"
    | "--" :: rest -> (
        match rest with [] -> Error "`env --` with no command" | cmd :: args -> Ok (Name cmd, args))
    | ("-S" | "--split-string") :: rest -> skip_env_arguments rest
    | (("-0" | "--null") as token) :: _ ->
        (* `env` refuses this with a command -- "cannot specify --null (-0) with command", exit
           125 -- so a shebang carrying it launches nothing. *)
        Error (Printf.sprintf "`env %s` cannot be combined with a command (env exits 125)" token)
    | ("-i" | "--ignore-environment" | "-v" | "--debug") :: rest -> skip_env_arguments rest
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
    | token :: _ when String.contains token '=' && not (String.is_prefix token ~prefix:"-") ->
        (* A `NAME=VALUE` assignment. `env` applies it BEFORE looking the command up, so
           `#!/usr/bin/env -S PATH=/definitely/missing bash` exits 127 having found no bash at all --
           measured. Skipping the assignment and then resolving the command on the scan's own PATH
           passes a shebang that cannot launch. Reproducing env's environment here is not something
           this check can do honestly (PATH decides the lookup, and ENV/BASH_ENV/SHELLOPTS reach the
           shell's own startup), so an assignment is refused instead. *)
        Error
          (Printf.sprintf
             "`env` assignment `%s`: the command is looked up in an environment this check cannot \
              reproduce"
             token)
    | token :: _ when String.is_prefix token ~prefix:"-" ->
        Error (Printf.sprintf "`env` option this check does not know: %s" token)
    | cmd :: args -> Ok (Name cmd, args)

  (** Refuse the options above; carry the rest, operands included, in the order written. *)
  let check_options args =
    let refused token =
      if String.is_prefix token ~prefix:"--" then List.mem refused_long token ~equal:String.equal
      else if String.is_prefix token ~prefix:"-" && String.length token > 1 then
        String.exists (String.drop_prefix token 1) ~f:(fun c ->
            List.mem refused_short c ~equal:Char.equal)
      else false
    in
    match List.find args ~f:refused with
    | Some token ->
        Error (Printf.sprintf "interpreter option that would not parse this file: %s" token)
    | None -> Ok args

  (* The message both kernel-semantics refusals share: the shape is wrong on the shebang line
     itself, so naming the offending text and the remedy is more use than naming a token. *)
  let one_argument_error who arg =
    Printf.sprintf
      "the kernel passes `%s` to %s as ONE argument -- no command or option there; use `env -S`"
      arg who

  let parse first_line =
    (* `#!` must be the file's first two BYTES: the kernel does not look past anything, so
       ` #!/bin/bash` is not a shebang -- exec fails with ENOEXEC and the caller's shell runs the
       file with `sh` instead (measured: such a file executes, under sh, not bash). Stripping the
       line before testing the prefix classified it as a bash script and skipped the no-shebang
       checks, so a bash-only construct in it would pass here and fail for a real launcher. Only the
       tail is stripped, which is where a CRLF checkout's `\r` would sit. *)
    match
      if String.is_prefix first_line ~prefix:"#!" then
        Some (String.strip (String.drop_prefix first_line 2))
      else None
    with
    | None -> Ok Sourced
    | Some rest -> (
        (* The kernel splits a shebang in exactly one place: the interpreter path, then the whole
           remainder as a single argument. Everything below follows from that. *)
        let rest = String.strip rest in
        match words rest with
        | [] -> Ok Sourced
        | first :: _ ->
            let argument = String.strip (String.drop_prefix rest (String.length first)) in
            let resolved =
              if String.equal (basename first) "env" then
                match argument with
                | "" -> Error "`env` with no command"
                | arg
                  when String.is_prefix arg ~prefix:"-S "
                       || String.is_prefix arg ~prefix:"--split-string=" ->
                    (* The one option that turns the single argument into a word list. *)
                    skip_env_arguments (words arg)
                | arg when String.exists arg ~f:Char.is_whitespace ->
                    Error (one_argument_error "`env`" arg)
                | arg -> skip_env_arguments [ arg ]
              else if String.exists argument ~f:Char.is_whitespace then
                Error (one_argument_error ("`" ^ basename first ^ "`") argument)
              else
                (* A direct shebang: the kernel execs this path, so keep it. *)
                Ok (Path first, if String.is_empty argument then [] else [ argument ])
            in
            Result.bind resolved ~f:(fun (launch, args) ->
                let shell = basename (match launch with Path p -> p | Name n -> n) in
                if not (List.mem parse_only_shells shell ~equal:String.equal) then
                  Error
                    (Printf.sprintf "interpreter whose `-n` this check cannot vouch for: %s" shell)
                else Result.map (check_options args) ~f:(fun opts -> Interp (launch, opts))))

  (** How a parse reads in a verdict label, so that the golden is a table of what the parser does
      rather than a column of booleans. *)
  let launched = function Path p -> p | Name n -> n

  let render = function
    | Ok Sourced -> "sourced"
    | Ok (Interp (launch, opts)) -> String.concat ~sep:" " (launched launch :: opts)
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
      (* The shapes this repository's own scripts use. A direct shebang renders as the PATH it
         names and an `env` one as a bare name, which is the round-4 distinction: the kernel execs
         the file `#!/bin/bash` names, while `env` performs a PATH lookup. Flattening both to
         "bash" is what let a macOS `/bin/bash` 3.2 script be accepted by a Homebrew bash 5. *)
      ("#!/bin/bash", "/bin/bash");
      ("#!/usr/bin/env bash", "bash");
      ("#!/bin/sh", "/bin/sh");
      ("#!/bin/dash", "/bin/dash");
      ("", "sourced");
      (* Kernel semantics: everything after the interpreter path is ONE argument. Both of these
         were asserted to read as `bash` before round 3, and both are broken shebangs -- the first
         measurably HANGS (env execs the script, which re-enters the same shebang), the second dies
         with bash's "invalid option". Pinning them as valid is what the review caught. *)
      ("#!/usr/bin/env -u FOO bash",
       "refused (the kernel passes `-u FOO bash` to `env` as ONE argument -- no command or option \
        there; use `env -S`)");
      ("#!/bin/bash -O extglob",
       "refused (the kernel passes `-O extglob` to `bash` as ONE argument -- no command or option \
        there; use `env -S`)");
      (* A single argument is fine when it really is one token, which is why `-eu` works in a
         shebang and `-O extglob` cannot. *)
      ("#!/bin/bash -eu", "/bin/bash -eu");
      (* Carried, not dropped: the shell is the authority on its own options, and a universal
         "harmless" list was wrong here -- `dash -B -n` exits 2 with "Illegal option -B", so the
         real invocation fails and the check must too. *)
      ("#!/bin/dash -B", "/bin/dash -B");
      ("#!/bin/bash --posix", "/bin/bash --posix");
      (* `env -S` is the mechanism that DOES produce a word list, so env's own grammar is reachable
         only inside it. The first entry is the review finding of round 2 verbatim. *)
      ("#!/usr/bin/env -S -u FOO bash", "bash");
      ("#!/usr/bin/env -S bash -O extglob", "bash -O extglob");
      ("#!/usr/bin/env -S FOO=bar bash",
       "refused (`env` assignment `FOO=bar`: the command is looked up in an environment this \
        check cannot reproduce)");
      ("#!/usr/bin/env -S -0 bash",
       "refused (`env -0` cannot be combined with a command (env exits 125))");
      ("#!/usr/bin/env -S -uFOO bash", "bash");
      ("#!/usr/bin/env -S --unset=FOO bash", "bash");
      ("#!/usr/bin/env -S -C /tmp bash", "bash");
      ("#!/usr/bin/env -S -i bash", "bash");
      ("#!/usr/bin/env -S -- bash", "bash");
      ("#!/usr/bin/env --split-string=bash -o posix", "bash -o posix");
      (* Refusals. Each would otherwise be silent: the first parses a Python file with a shell's
         grammar, the second makes `-n` a parse of the command line rather than of the file, the
         third is an `env` option whose operand rule this check does not know. *)
      ("#!/usr/bin/env python3",
       "refused (interpreter whose `-n` this check cannot vouch for: python3)");
      ("#!/bin/bash -c", "refused (interpreter option that would not parse this file: -c)");
      ("#!/bin/bash --version",
       "refused (interpreter option that would not parse this file: --version)");
      (* Not a shebang: `#!` must be the first two bytes, so this file is run by the caller's
         shell and gets the no-shebang treatment. *)
      (" #!/bin/bash", "sourced");
      ("#!/usr/bin/env -S -Z bash", "refused (`env` option this check does not know: -Z)");
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

(** Whether [prog] exists here and can be executed, probed once per name by handing it a file that
    is a valid script in every shell there is.

    "Exists" is the question, deliberately, and it is not the same as "passed the probe". An earlier
    version answered false whenever the probe exited nonzero, which quietly conflated a missing
    binary with a present one that rejected `:` -- and since a false answer sends [resolve] to a
    substitute, a broken or non-shell interpreter would have been swapped out and the script judged
    by a DIFFERENT shell than its shebang names, silently. That is the same class of defect as the
    stand-in the round-2 review caught. So only a failure to exec at all (see [parse_check]) counts
    as absent; anything that ran keeps its verdict, and a `bash` on PATH that cannot parse `:` is a
    reported failure rather than a reason to consult another shell. *)
let available =
  let cache = Hashtbl.create (module String) in
  fun prog ->
    Hashtbl.find_or_add cache prog ~default:(fun () ->
        let tmp = Stdlib.Filename.temp_file "ocannl_shell_probe" ".sh" in
        Exn.protect
          ~finally:(fun () -> try Stdlib.Sys.remove tmp with _ -> ())
          ~f:(fun () ->
            Out_channel.write_all tmp ~data:":\n";
            Option.is_some (parse_check prog [] tmp)))

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

(** Resolve one wanted interpreter to something that exists here, announcing any substitution.

    A [Path] is tried as written first, which is the whole point of keeping it: the kernel would exec
    that file, and a same-named binary earlier on PATH can be a different major version accepting a
    different grammar. Its fallback is the basename on PATH, because a direct shebang's path is a
    Unix path and Windows has no `/bin/bash` to exec -- under Git Bash `bash` resolves and the
    literal path does not, so refusing there would fail every `#!/bin/sh` script in this repository
    for a reason that is about the host rather than about the script. Announced, like every other
    substitution here. *)
let rec resolve ~rel = function
  | Shebang.Path path ->
      if available path then Some path
      else
        let name = Shebang.basename path in
        eprintf "%s: no `%s` on this host, resolving `%s` on PATH instead\n" rel path name;
        resolve ~rel (Shebang.Name name)
  | Shebang.Name shell -> (
      if available shell then Some shell
      else
        match List.find (stand_ins shell) ~f:available with
        | Some substitute ->
            eprintf "%s: no `%s` on this host, parsing with `%s` instead\n" rel shell substitute;
            Some substitute
        | None -> None)

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
            | Shebang.Sourced -> [ (Shebang.Name "sh", []); (Shebang.Name "bash", []) ]
            | Shebang.Interp (launch, opts) -> [ (launch, opts) ]
          in
          let usable =
            List.filter_map wanted ~f:(fun (launch, opts) ->
                Option.map (resolve ~rel launch) ~f:(fun prog -> (prog, opts)))
          in
          if List.is_empty usable then
            Verdict.fail
              (Printf.sprintf "no shell on this host can parse %s (wanted %s)" rel
                 (String.concat ~sep:", "
                    (List.map wanted ~f:(fun (l, _) -> Shebang.launched l))))
          else
            let complaints =
              List.filter_map usable ~f:(fun (prog, opts) ->
                  match parse_check prog opts path with
                  | None ->
                      Some (Printf.sprintf "`%s` disappeared between the probe and the check" prog)
                  | Some (Unix.WEXITED 0, _) -> None
                  | Some (_, said) ->
                      (* Naming the invocation as it was actually made, carried options included:
                         with `#!/bin/dash -B` the option is the whole reason it failed. *)
                      Some
                        (Printf.sprintf "`%s -n` said: %s"
                           (String.concat ~sep:" " (prog :: opts))
                           said))
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
