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
  (** What a script's first line says about how to parse it.

      {1 Why this grammar is narrow}

      Rounds 2 to 5 of the review on PR #454 each found a case where reconstructing what the kernel,
      `env` and the shell would do got it wrong, and round 5's answer -- carry every option through
      and let the shell be the authority -- turned out to be the worst of them: a shebang with a
      positional operand, `#!/usr/bin/env -S bash helper.sh`, produced the invocation
      `bash helper.sh -n target`, and a shell stops processing options at the first operand. So `-n`
      became an argument and bash EXECUTED helper.sh. Measured, with a marker file. That is the one
      thing this check promises never to do.

      The lesson is not that the emulation needed another patch. Every capability added to it has
      cost a defect, the repository has no script that uses any of them, and the ambiguity at the
      centre is not resolvable without each shell's option-arity table: given `-O extglob`, nothing
      short of bash's own tables says whether `extglob` is an operand of `-O` or the script to run.

      So the accepted grammar is now the small sound core, and everything else is REFUSED -- a
      failing verdict naming what it could not vouch for, never a silent pass:

      - no shebang: the file is sourced, checked under both `sh` and `bash`;
      - [#!<path>] and [#!/usr/bin/env <shell>], optionally [-S<shell>] / [-S <shell>] /
        [--split-string=<shell>], where <shell> is in {!parse_only_shells};
      - shebang options are allowed only when every token begins with `-`, none is the `--`
        terminator, and none carries a letter or long spelling that would stop `-n` from being a
        parse of this file ({!refused_short}, {!refused_long}). A bare word is refused outright,
        because that is exactly the operand-or-option-argument ambiguity above;
      - any other `env` option, and any `NAME=VALUE` assignment, is refused: they change the
        environment the lookup and the shell's own startup happen in, which this check cannot
        reproduce (see the `BASH_ENV` handling in {!parse_check}).

      Two independent guards keep the promise, rather than one. This grammar is the first;
      {!parse_check} placing `-n` immediately after the program, before anything from the shebang,
      is the second -- with `-n` already set, even a token this parser wrongly admitted would be
      PARSED as a script rather than run. *)
  type t =
    | Sourced
        (** No shebang: the file is sourced rather than run, so it has no interpreter of its own and
            must parse under whichever shell sources it. *)
    | Interp of launch * string list
        (** The interpreter the kernel (or `env`) would exec, and the options -- all of them
            option-shaped, see above -- that have to be given to it for the file to parse. *)

  (** How the interpreter is found, which is not the same question for the two shebang forms and
      cannot be flattened to a name (round 4).

      A direct `#!/bin/bash` names a FILE, and the kernel execs that file; resolving `bash` on PATH
      instead can run a different build entirely -- on macOS `/bin/bash` is 3.2 while a Homebrew
      `bash` 5 sits earlier on PATH, and 5 accepts `declare -A` and `${v^^}` that 3.2 rejects, so a
      script the kernel-selected shell would refuse passes. That skew is on this repository's own
      macOS CI leg. A shebang going through `env`, by contrast, is a PATH lookup by definition. *)
  and launch =
    | Path of string  (** A direct shebang: exec this exact file, as the kernel would. *)
    | Name of string  (** Selected through `env`: resolve on PATH, as `env` would. *)

  (** Shells whose `-n` parses without executing. A whitelist rather than a fallthrough because `-n`
      is not a shared convention outside this family -- `python3 -n` is an error and `perl -n` wraps
      the program in a read loop and RUNS it -- and a `.sh` file is free to name one of those. *)
  let parse_only_shells = [ "sh"; "bash"; "dash"; "ash"; "ksh"; "mksh"; "zsh" ]

  (** POSIX shells that can stand in for one another when the named one is not installed. A stand-in
      is a weaker check, never a wrong one: these accept a subset of what bash does. bash is
      deliberately absent -- see [stand_ins]. *)
  let posix_family = [ "sh"; "dash"; "ash" ]

  (** Short options that stop `-n` from being a parse of this whole file. `-c` takes the program
      from the command line and `-s` from stdin, so `-n` would parse something else; `-t` makes the
      shell stop after one command, and `bash -t -n` exits 0 on a file whose second line is a syntax
      error (measured) -- a silent pass, which is the worst failure this check has. *)
  let refused_short = [ 'c'; 's'; 't' ]

  (** Long options that exit before reading the file, so an unparsable script would pass with status
      0. *)
  let refused_long = [ "--version"; "--help" ]

  let basename p = List.last_exn (String.split_on_chars p ~on:[ '/'; '\\' ])

  let words line =
    List.filter (String.split_on_chars line ~on:[ ' '; '\t' ]) ~f:(Fn.non String.is_empty)

  (** Accept a shebang's interpreter options, or say why not.

      Every token must be option-shaped. A bare word is refused rather than guessed at: it is either
      an operand of the option before it or the script the shell would run instead of this one, and
      telling those apart needs the shell's own option-arity tables. The `--` terminator is refused
      for the same reason -- everything after it is positional, and `bash -- -n path` makes `-n` the
      filename (bash 5.2 exits 127, measured). *)
  let check_options args =
    let refusal token =
      if String.equal token "--" then
        Some "the `--` terminator makes everything after it positional"
      else if not (String.is_prefix token ~prefix:"-") then
        Some
          "a bare word is either an option's operand or the script the shell would run instead of \
           this one, and this check cannot tell which"
      else if String.is_prefix token ~prefix:"--" then
        if List.mem refused_long token ~equal:String.equal then
          Some "this option exits before reading the file"
        else None
      else if
        String.exists (String.drop_prefix token 1) ~f:(fun c ->
            List.mem refused_short c ~equal:Char.equal)
      then Some "this option stops `-n` from being a parse of the whole file"
      else None
    in
    match List.find_map args ~f:(fun t -> Option.map (refusal t) ~f:(fun why -> (t, why))) with
    | Some (token, why) ->
        Error (Printf.sprintf "shebang option this check cannot vouch for: %s -- %s" token why)
    | None -> Ok args

  (* The message the kernel-semantics refusals share: the shape is wrong on the shebang line
     itself, so naming the offending text and the remedy is more use than naming a token. *)
  let one_argument_error who arg =
    Printf.sprintf
      "the kernel passes `%s` to %s as ONE argument -- no command or option there; use `env -S`" arg
      who

  (** What `env` would exec, from the single argument the kernel hands it.

      Only `-S`/`--split-string` is accepted, in its three spellings, because that is the option
      whose whole purpose is to make a shebang carry a word list. Every other `env` option and every
      `NAME=VALUE` assignment is refused: they build an environment for the command, and the
      command's parsing depends on that environment in ways this check cannot reproduce -- `PATH`
      decides which binary is found at all (`env -S PATH=/definitely/missing bash` exits 127,
      measured), and `BASH_ENV`/`ENV` are parsed by the shell before the script (measured: a broken
      `BASH_ENV` makes `bash -n` fail a valid file). *)
  let env_command argument =
    let payload =
      if String.is_prefix argument ~prefix:"-S " then
        Some (String.drop_prefix argument (String.length "-S "))
      else if String.is_prefix argument ~prefix:"--split-string=" then
        Some (String.drop_prefix argument (String.length "--split-string="))
      else if String.is_prefix argument ~prefix:"-S" then
        (* `-Sbash`: GNU env accepts the payload attached to the short option. *)
        Some (String.drop_prefix argument 2)
      else None
    in
    match payload with
    | Some payload -> (
        match words payload with
        | [] -> Error "`env -S` with no command"
        | cmd :: args ->
            (* Inside `-S` the payload really is a word list, so env's own options and assignments
               are reachable here -- and refused here, for the reason below. *)
            if String.is_prefix cmd ~prefix:"-" then
              Error
                (Printf.sprintf
                   "`env` option this check cannot vouch for: %s -- it builds an environment the \
                    command's parsing depends on"
                   cmd)
            else if String.contains cmd '=' then
              Error
                (Printf.sprintf
                   "`env` assignment `%s`: the command is looked up, and parses, in an environment \
                    this check cannot reproduce"
                   cmd)
            else Ok (Name cmd, args))
    | None ->
        if String.exists argument ~f:Char.is_whitespace then
          Error (one_argument_error "`env`" argument)
        else if String.is_prefix argument ~prefix:"-" then
          Error
            (Printf.sprintf
               "`env` option this check cannot vouch for: %s -- it builds an environment the \
                command's parsing depends on"
               argument)
        else if String.contains argument '=' then
          Error
            (Printf.sprintf
               "`env` assignment `%s`: the command is looked up, and parses, in an environment this \
                check cannot reproduce"
               argument)
        else Ok (Name argument, [])

  let parse first_line =
    (* `#!` must be the file's first two BYTES: the kernel does not look past anything, so
       ` #!/bin/bash` is not a shebang -- exec fails with ENOEXEC and the caller's shell runs the
       file with `sh` instead (measured: such a file executes, under sh, not bash). Only the tail is
       stripped, which is where a CRLF checkout's `\r` would sit. *)
    match
      if String.is_prefix first_line ~prefix:"#!" then
        Some (String.strip (String.drop_prefix first_line 2))
      else None
    with
    | None -> Ok Sourced
    | Some rest -> (
        match words rest with
        | [] -> Ok Sourced
        | first :: _ ->
            (* The kernel splits a shebang in exactly one place: the interpreter path, then the
               whole remainder as a single argument. *)
            let argument = String.strip (String.drop_prefix rest (String.length first)) in
            let resolved =
              if String.equal (basename first) "env" then
                if String.is_empty argument then Error "`env` with no command"
                else env_command argument
              else if String.exists argument ~f:Char.is_whitespace then
                Error (one_argument_error ("`" ^ basename first ^ "`") argument)
              else Ok (Path first, if String.is_empty argument then [] else [ argument ])
            in
            Result.bind resolved ~f:(fun (launch, args) ->
                let shell = basename (match launch with Path p -> p | Name n -> n) in
                if not (List.mem parse_only_shells shell ~equal:String.equal) then
                  Error
                    (Printf.sprintf "interpreter whose `-n` this check cannot vouch for: %s" shell)
                else Result.map (check_options args) ~f:(fun opts -> Interp (launch, opts))))

  let launched = function Path p -> p | Name n -> n

  (** How a parse reads in a verdict label, so that the golden is a table of what the parser does
      rather than a column of booleans. *)
  let render = function
    | Ok Sourced -> "sourced"
    | Ok (Interp (launch, opts)) -> String.concat ~sep:" " (launched launch :: opts)
    | Error reason -> "refused (" ^ reason ^ ")"

  (** The grammar, as a table of lines and what {!parse} must make of them -- rendered by {!render},
      so a case reads as the specification rather than as a boolean. The repository's own scripts
      exercise two shapes of it; everything else here comes from a review round that found the
      parser wrong about it, and is kept so the same case cannot regress quietly. *)
  let shebang_cases =
    [
      (* The shapes this repository's own scripts use. A direct shebang renders as the PATH it
         names and an `env` one as a bare name (round 4): the kernel execs the file `#!/bin/bash`
         names, while `env` performs a PATH lookup. Flattening both to "bash" let a macOS
         `/bin/bash` 3.2 script be accepted by a Homebrew bash 5. *)
      ("#!/bin/bash", "/bin/bash");
      ("#!/usr/bin/env bash", "bash");
      ("#!/bin/sh", "/bin/sh");
      ("#!/bin/dash", "/bin/dash");
      ("", "sourced");
      (* Not a shebang: `#!` must be the first two bytes, so the file is run by the caller's shell
         and gets the no-shebang treatment (round 5). *)
      (" #!/bin/bash", "sourced");
      (* Kernel semantics (round 3): everything after the interpreter path is ONE argument. Both of
         these were asserted to read as `bash` before; both are broken shebangs -- the first
         measurably HANGS (env execs the script, which re-enters the same shebang), the second dies
         with bash's "invalid option". *)
      ("#!/usr/bin/env -u FOO bash",
       "refused (the kernel passes `-u FOO bash` to `env` as ONE argument -- no command or option \
        there; use `env -S`)");
      ("#!/bin/bash -O extglob",
       "refused (the kernel passes `-O extglob` to `bash` as ONE argument -- no command or option \
        there; use `env -S`)");
      (* A single option-shaped token is fine, which is why `-eu` works in a shebang where
         `-O extglob` cannot. *)
      ("#!/bin/bash -eu", "/bin/bash -eu");
      ("#!/bin/bash --posix", "/bin/bash --posix");
      (* `env -S` is the mechanism that produces a word list, in all three spellings -- the attached
         form is round 6's. *)
      ("#!/usr/bin/env -S bash", "bash");
      ("#!/usr/bin/env -Sbash", "bash");
      ("#!/usr/bin/env --split-string=bash --posix", "bash --posix");
      (* The P1 of round 6, and the reason the grammar is narrow: a positional operand makes the
         shell stop processing options, so the constructed `bash helper.sh -n target` EXECUTED
         helper.sh. Measured, with a marker file. *)
      ("#!/usr/bin/env -S bash helper.sh",
       "refused (shebang option this check cannot vouch for: helper.sh -- a bare word is either an \
        option's operand or the script the shell would run instead of this one, and this check \
        cannot tell which)");
      ("#!/usr/bin/env -S bash -O extglob",
       "refused (shebang option this check cannot vouch for: extglob -- a bare word is either an \
        option's operand or the script the shell would run instead of this one, and this check \
        cannot tell which)");
      (* Options that would make `-n` mean something other than "parse this whole file". `-t` exits
         after one command (`bash -t -n` returns 0 on a file whose second line is a syntax error);
         `--` makes `-n` positional (bash 5.2: "-n: No such file or directory", 127). *)
      ("#!/bin/bash -t",
       "refused (shebang option this check cannot vouch for: -t -- this option stops `-n` from \
        being a parse of the whole file)");
      ("#!/bin/bash -c",
       "refused (shebang option this check cannot vouch for: -c -- this option stops `-n` from \
        being a parse of the whole file)");
      ("#!/bin/bash --",
       "refused (shebang option this check cannot vouch for: -- -- the `--` terminator makes \
        everything after it positional)");
      ("#!/bin/bash --version",
       "refused (shebang option this check cannot vouch for: --version -- this option exits before \
        reading the file)");
      (* `env` builds an environment, and both the lookup and the shell's own startup depend on it
         (rounds 5 and 6). Refused rather than emulated. *)
      ("#!/usr/bin/env -S PATH=/missing bash",
       "refused (`env` assignment `PATH=/missing`: the command is looked up, and parses, in an \
        environment this check cannot reproduce)");
      ("#!/usr/bin/env -S -i bash",
       "refused (`env` option this check cannot vouch for: -i -- it builds an environment the \
        command's parsing depends on)");
      (* An interpreter whose `-n` is not a parse at all. *)
      ("#!/usr/bin/env python3",
       "refused (interpreter whose `-n` this check cannot vouch for: python3)");
    ]
end


(** Variables that make a shell parse something other than -- or in addition to -- the file it is
    handed, cleared for the child. `BASH_ENV` is the sharp one: bash expands and PARSES it before a
    non-interactive script, so an exported `BASH_ENV` pointing at a file with a syntax error makes
    `bash -n` fail every valid script in the repository (measured). `ENV` is its POSIX-shell
    counterpart, and `SHELLOPTS`/`BASHOPTS` set shell options at startup, which can reach the
    grammar. Clearing them is better than declaring them as dune dependencies: the check should not
    depend on the ambient environment at all, and a variable that cannot influence the run needs no
    dependency edge (Codex review round 6). *)
let cleared_variables = [ "BASH_ENV"; "ENV"; "SHELLOPTS"; "BASHOPTS" ]

let isolated_environment =
  lazy
    (Array.filter (Unix.environment ()) ~f:(fun binding ->
         let name = List.hd_exn (String.split binding ~on:'=') in
         not (List.mem cleared_variables name ~equal:String.equal)))

(** Run [prog -n args… path] with stdin closed, an isolated environment, and both output streams
    captured; return the exit status together with what the shell said.

    `-n` goes FIRST, before anything the shebang carried. That is the second of the two guards on
    this check's promise to execute nothing (the first being {!Shebang}'s grammar): a shell stops
    processing options at its first operand, so a token this parser wrongly admitted as an option
    could otherwise become the script and be RUN -- which is exactly what happened before round 6,
    where `bash helper.sh -n target` executed helper.sh. With `-n` already set, the same mistake
    parses the wrong file instead of running it: still a bug, no longer an execution.

    A missing [prog] arrives as exit 127 on Unix (the forked child cannot exec and exits with it)
    and as [Unix_error] on Windows; both mean the same thing here, so both become [None]. *)
let parse_check prog args path =
  let tmp = Stdlib.Filename.temp_file "ocannl_shell_parse" ".log" in
  let devnull = if Stdlib.Sys.win32 then "NUL" else "/dev/null" in
  Exn.protect
    ~finally:(fun () -> try Stdlib.Sys.remove tmp with _ -> ())
    ~f:(fun () ->
      let out = Unix.openfile tmp [ Unix.O_WRONLY; Unix.O_TRUNC ] 0o600 in
      let inp = Unix.openfile devnull [ Unix.O_RDONLY ] 0o400 in
      let argv = Array.of_list ((prog :: "-n" :: args) @ [ path ]) in
      let status =
        Exn.protect
          ~finally:(fun () ->
            Unix.close out;
            Unix.close inp)
          ~f:(fun () ->
            match
              Unix.create_process_env prog argv (force isolated_environment) inp out out
            with
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

    A [Path] is tried as written, which is the whole point of keeping it: the kernel would exec that
    file, and a same-named binary earlier on PATH can be a different major version accepting a
    different grammar.

    When that path is absent, the two platforms differ and so does the answer (round 6). On Unix an
    absent path means the kernel could not launch this script AT ALL, so resolving the basename on
    PATH would report success for something unrunnable -- it is a failure, and reported as one.
    Windows has no `/bin/sh` to exec in the first place: under Git Bash the shebang is honoured by
    the shell rather than the kernel, and the name resolves where the literal path never will, so
    the fallback there is the faithful reading rather than a papering-over. Hence the platform test,
    which the earlier unconditional fallback lacked while claiming this same rationale. *)
let rec resolve ~rel = function
  | Shebang.Path path ->
      if available path then Some path
      else if Stdlib.Sys.win32 then (
        let name = Shebang.basename path in
        eprintf "%s: no `%s` on this host (Windows resolves the shebang itself), using `%s`\n" rel
          path name;
        resolve ~rel (Shebang.Name name))
      else None
  | Shebang.Name shell -> (
      if available shell then Some shell
      else
        match List.find (stand_ins shell) ~f:available with
        | Some substitute ->
            eprintf "%s: no `%s` on this host, parsing with `%s` instead\n" rel shell substitute;
            Some substitute
        | None -> None)

(** The first line of a file, or [None] if it cannot be read as text at all. Binary files reach
    here through the directory globs, so a failure to read one is "not a script", not an error. *)
let first_line_of path =
  try In_channel.with_file path ~f:In_channel.input_line with _ -> None

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
  (* A `.sh` file is in scope by its name; anything else the globs hand over is in scope only if it
     actually starts with a shell shebang, which is how `tools/run-tests` gets covered without the
     rule depending on every file in the repository (round 6). Reading two bytes off each candidate
     is the price, over the few hundred files in `tools/` and `scripts/`. *)
  let in_scope path =
    String.is_suffix path ~suffix:".sh"
    ||
    match Shebang.parse (Option.value ~default:"" (first_line_of path)) with
    | Ok (Shebang.Interp _) -> true
    | Ok Shebang.Sourced | Error _ -> false
  in
  let scripts =
    Array.to_list Stdlib.Sys.argv
    |> Fn.flip List.drop 2
    |> List.filter ~f:(fun path -> (not (Stdlib.Sys.is_directory path)) && in_scope path)
    |> List.map ~f:(fun path -> (repo_relative base path, path))
    (* The `*.sh` glob and the two directory globs overlap, so the same script arrives twice; one
       entry per repository-relative path, which is also what the golden is keyed by. *)
    |> List.dedup_and_sort ~compare:(fun (a, _) (b, _) -> String.compare a b)
  in
  List.iter scripts ~f:(fun (rel, path) ->
      let first_line = Option.value (first_line_of path) ~default:"" in
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
