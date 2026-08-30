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

   So {!Shebang.parse} reads the line as the command line it is -- but as the KERNEL builds it,
   which is the correction of round 3: everything after the interpreter path is ONE argument, not a
   word list. Only `env`'s `-S`/`--split-string` splits it, which is what that option exists for.
   The difference is not academic in either direction. `#!/usr/bin/env -u FOO bash` hands env the
   single argument `-u FOO bash`, so env unsets a variable named " FOO bash" and execs the script
   itself -- measured here, it does not reach bash, it HANGS (the kernel re-enters the same shebang
   until it gives up). And `#!/bin/bash -O extglob` hands bash the single argument `-O extglob`,
   which bash rejects with "invalid option". Reading either as a word list made this check pin a
   broken script as valid; both are now refused, naming the kernel semantics and pointing at `-S`.

   Beyond that, what the parser will not do is guess. An interpreter outside {!parse_only_shells} is
   refused rather than run, because `-n` means "parse only" for shells and something else entirely
   for a `python3` that a `.sh` file might name -- `perl -n` wraps the program in a read loop and
   RUNS it. An option in {!Shebang.refused_short}/{!Shebang.refused_long} is refused too, because
   `-c` and `-s` make the shell read its program from somewhere other than the file and `--version`
   exits before reading it at all. Every other option is carried through to the shell, which is the
   authority on its own: a universal "harmless flags" list is a guess, and it was wrong for
   `#!/bin/dash -B`. An `env` shebang that assigns a variable or asks for `-0` output is refused for
   the same reason -- the first changes the environment the lookup happens in, and the second is an
   `env` invocation that exits 125 rather than running anything. Refused means a failing verdict
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

      {1 Why this grammar accepts no arguments at all}

      Seven review rounds on PR #454 went into reconstructing what the kernel, `env` and the shell
      would do with a shebang, and the option surface was wrong in both directions to the end:

      - round 6's P1 -- `#!/usr/bin/env -S bash helper.sh` built `bash helper.sh -n target`, and a
        shell stops processing options at its first operand, so `-n` became an argument and bash
        EXECUTED helper.sh (measured, with a marker file);
      - round 7's P1 -- `#!/usr/bin/env -S zsh --exec` builds `zsh -n --exec path`, and zsh's named
        options make `--exec` turn execution back ON after `-n` turned it off;
      - and round 7 again, in the other direction: `bash -n --posix path` exits 2, "invalid option",
        because bash wants its GNU long options BEFORE the short ones -- so the round-6 guard of
        putting `-n` first turned a shebang the table explicitly accepts into a reported syntax
        error (measured, both orderings).

      Placing `-n` is thus not safe in either position, and no ordering rule fixes an option that
      re-enables execution. Each of those was a fix for the round before it. So the grammar accepts
      a shell and NOTHING ELSE: the invocation is always exactly [<shell> -n <path>], with nothing
      from the shebang between the program and the file, and there is no argument surface left to be
      wrong about. What is accepted:

      - no shebang: the file is sourced, checked under both `sh` and `bash`;
      - [#!<path>], where the basename is in {!parse_only_shells};
      - [#!/usr/bin/env <shell>], and the same through [-S <shell>] / [-S<shell>] /
        [--split-string=<shell>].

      Everything else -- any argument to the shell, any `env` option, any `NAME=VALUE` assignment,
      an interpreter outside the whitelist -- is REFUSED: a failing verdict naming what could not be
      vouched for, never a silent pass. Nothing in this repository uses any of it, and a script that
      later wants to gets a message saying exactly why this check will not speak for it.

      The whitelist itself is load-bearing: `-n` means "parse only" for shells and something else
      entirely elsewhere -- `python3 -n` is an error and `perl -n` wraps the program in a read loop
      and RUNS it -- and a `.sh` file is free to name one of those. *)
  type t =
    | Sourced
        (** No shebang: the file is sourced rather than run, so it has no interpreter of its own and
            must parse under whichever shell sources it. *)
    | Interp of launch  (** The interpreter the kernel, or `env`, would exec. *)

  (** How the interpreter is found, which is not the same question for the two shebang forms and
      cannot be flattened to a name (round 4).

      A direct `#!/bin/bash` names a FILE, and the kernel execs that file; resolving `bash` on PATH
      instead can run a different build entirely -- on macOS `/bin/bash` is 3.2 while a Homebrew
      `bash` 5 sits earlier on PATH, and 5 accepts `declare -A` and `${v^^}` that 3.2 rejects. That
      skew is on this repository's own macOS CI leg. A shebang going through `env` is a PATH lookup
      by definition, so a name is the faithful reading there. *)
  and launch =
    | Path of string  (** A direct shebang: exec this exact file, as the kernel would. *)
    | Via_env of { env_path : string; command : string }
        (** Selected through `env`: the kernel execs [env_path], which then resolves [command] on
            PATH. BOTH have to exist for the script to launch, and the env binary was the half this
            check did not verify (round 9) -- it accepted `/bin/env` from the table and probed only
            the command, so on a host with just `/usr/bin/env` an unlaunchable script passed. Same
            rule as [Path] now, applied to the binary the kernel actually execs. *)

  let parse_only_shells = [ "sh"; "bash"; "dash"; "ash"; "ksh"; "mksh"; "zsh" ]

  (** POSIX shells that can stand in for one another. Consulted ONLY for the no-shebang case, where
      `sh` and `bash` are this check's own choice of checkers rather than anything the file asked
      for; a shell a shebang actually names is never substituted (round 7). bash is not a member in
      either direction: standing in for dash it accepts what dash rejects, and standing in for bash
      it rejects every bashism. *)
  let posix_family = [ "sh"; "dash"; "ash" ]

  let basename p = List.last_exn (String.split_on_chars p ~on:[ '/'; '\\' ])

  (** The interpreter paths that get `env` semantics. A basename test is not enough (round 8):
      `#!/opt/custom/env bash` was handed env's meaning and resolved `bash` on PATH, while the
      kernel would have tried to exec `/opt/custom/env` -- a path that may not exist, and if it does
      is not necessarily GNU env. These two are what every `env` shebang in the wild names; anything
      else basenamed `env` falls through to the direct-interpreter path, where it is refused for not
      being a shell. *)
  let env_paths = [ "/usr/bin/env"; "/bin/env" ]

  let words line =
    List.filter (String.split_on_chars line ~on:[ ' '; '\t' ]) ~f:(Fn.non String.is_empty)

  let no_arguments who args =
    Printf.sprintf
      "%s is given %s, and this check runs `<shell> -n <file>` and nothing else -- no placement of \
       `-n` is safe among shell options"
      who (String.concat ~sep:" " args)

  (** What `env` would exec, from the single argument the kernel hands it.

      That argument must be a bare command name. Nothing else is accepted, and `-S`/`--split-string`
      is the notable removal (round 10): whether an `env` implementation supports it is a property
      of the BINARY, not of its path -- BSD env and coreutils before 8.30 have no `-S` -- so
      accepting it meant either probing the feature or passing a shebang that fails before the shell
      starts. Neither was necessary, because since round 7 removed shell arguments the payload can
      only ever be a bare command name, which plain `env` resolves without `-S`. The option had
      become pure surface, so it is gone and `#!/usr/bin/env -S bash` is refused pointing at the
      plain spelling.

      `env` options and `NAME=VALUE` assignments are refused for the older reason: they build an
      environment that both the lookup and the shell's own startup depend on, which this check
      cannot reproduce -- `env -S PATH=/definitely/missing bash` exits 127, and a broken `BASH_ENV`
      makes `bash -n` fail a valid file (both measured). *)
  let env_command env_path argument =
    if String.is_prefix argument ~prefix:"-S" || String.is_prefix argument ~prefix:"--split-string"
    then
      Error
        "`env -S` support is a property of the env binary, not its path, and buys nothing here \
         since no shell arguments are accepted -- write `#!/usr/bin/env <shell>`"
    else if String.exists argument ~f:Char.is_whitespace then
      Error
        (Printf.sprintf
           "the kernel passes `%s` to `env` as ONE argument -- there is no command by that name"
           argument)
    else if String.is_prefix argument ~prefix:"-" then
      Error
        (Printf.sprintf
           "`env` option this check cannot vouch for: %s -- it builds an environment the command's \
            parsing depends on"
           argument)
    else if String.contains argument '=' then
      Error
        (Printf.sprintf
           "`env` assignment `%s`: the command is looked up, and parses, in an environment this \
            check cannot reproduce"
           argument)
    else Ok (Via_env { env_path; command = argument })

  let parse first_line =
    (* `#!` must be the file's first two BYTES: the kernel does not look past anything, so `
       #!/bin/bash` is not a shebang -- exec fails ENOEXEC and the caller's shell runs the file with
       `sh` (measured: such a file executes, under sh, not bash). *)
    if not (String.is_prefix first_line ~prefix:"#!") then Ok Sourced
    else
      let body = String.drop_prefix first_line 2 in
      (* A CR belongs to the interpreter PATH as far as the kernel is concerned: a CRLF
         `#!/bin/bash` file fails to exec with "/bin/bash^M: bad interpreter" (126) while `bash -n`
         on it exits 0 -- so normalising the byte away here would report a script that cannot run as
         parsing (round 7, measured). `.gitattributes` pins `*.sh` to LF, which is what makes this a
         guard rather than a routine path. *)
      if String.exists body ~f:(fun c -> Char.equal c '\r') then
        Error
          "the shebang line ends CRLF, and the kernel keeps the CR in the interpreter path (`bad \
           interpreter`); this file needs LF endings"
      else
        match words body with
        | [] -> Ok Sourced
        | first :: _ ->
            (* The kernel splits a shebang in exactly one place: the interpreter path, then the
               whole remainder as a single argument. *)
            let body = String.strip body in
            let argument = String.strip (String.drop_prefix body (String.length first)) in
            let resolved =
              if List.mem env_paths first ~equal:String.equal then
                if String.is_empty argument then Error "`env` with no command"
                else env_command first argument
              else if
                (* Whitelist first on this branch, so that a path this check gives no special
                   meaning to -- `/opt/custom/env`, a `python3` -- is refused for WHAT IT IS rather
                   than for the arguments it happens to carry. *)
                not (List.mem parse_only_shells (basename first) ~equal:String.equal)
              then
                Error
                  (Printf.sprintf "interpreter whose `-n` this check cannot vouch for: %s"
                     (basename first))
              else if String.is_empty argument then Ok (Path first)
              else Error (no_arguments (Printf.sprintf "`%s`" (basename first)) [ argument ])
            in
            Result.bind resolved ~f:(fun launch ->
                let shell =
                  basename (match launch with Path p -> p | Via_env { command; _ } -> command)
                in
                if List.mem parse_only_shells shell ~equal:String.equal then Ok (Interp launch)
                else
                  Error
                    (Printf.sprintf "interpreter whose `-n` this check cannot vouch for: %s" shell))

  (** Whether a first line looks like a shell shebang at all, decided WITHOUT regard to whether
      {!parse} accepts its arguments.

      This is what puts an unsuffixed file into the scan, and it has to be the looser question
      (round 7): `tools/run-tests` carrying `#!/bin/bash -c` is a file this check must report on,
      and keying scope off a successful parse silently dropped exactly those -- the twelve suffixed
      scripts kept the floor satisfied, so a broken executable could vanish from both the golden and
      the check. Mentioning a shell anywhere in the line is deliberately generous: over-including
      costs a refusal that names the file, while under-including costs silence. *)
  let mentions_a_shell first_line =
    (* A word can carry the shell attached to `env`'s split-string option, and both spellings are
       forms {!parse} accepts -- so a predicate that only basenamed the raw word filtered
       `#!/usr/bin/env -Sbash` out of the scan entirely, before anything could report on it (round
       8). Stripping those prefixes here keeps the two in step. *)
    let shell_of word =
      let word = String.strip word in
      let word =
        match String.chop_prefix word ~prefix:"--split-string=" with
        | Some rest -> rest
        | None -> (
            match String.chop_prefix word ~prefix:"-S" with Some rest -> rest | None -> word)
      in
      basename word
    in
    String.is_prefix first_line ~prefix:"#!"
    && List.exists
         (words (String.drop_prefix first_line 2))
         ~f:(fun word -> List.mem parse_only_shells (shell_of word) ~equal:String.equal)

  (** Lines whose SCOPE must hold whatever {!parse} makes of them, pinned separately because the two
      questions come apart: round 7 found scope keyed off a successful parse, which dropped the
      broken files this check exists for, and round 8 found the looser predicate blind to two
      spellings `parse` accepts. Neither bug is visible in the parse table. *)
  let scope_cases =
    [
      ("#!/bin/bash", true);
      ("#!/usr/bin/env bash", true);
      ("#!/usr/bin/env -Sbash", true);
      ("#!/usr/bin/env -S bash", true);
      ("#!/usr/bin/env --split-string=bash", true);
      (* In scope precisely BECAUSE parse refuses them: these are the files that must be
         reported. *)
      ("#!/bin/bash -c", true);
      ("#!/usr/bin/env -S bash helper.sh", true);
      (* Not shell scripts, and not this check's business. *)
      ("#!/usr/bin/env python3", false);
      ("#!/usr/bin/perl", false);
      ("", false);
      (" #!/bin/bash", false);
    ]

  (** How a parse reads in a verdict label, so that the golden is a table of what the parser does
      rather than a column of booleans. *)
  let render = function
    | Ok Sourced -> "sourced"
    | Ok (Interp (Path p)) -> p
    | Ok (Interp (Via_env { env_path; command })) -> command ^ " via " ^ env_path
    | Error reason -> "refused (" ^ reason ^ ")"

  (** The grammar, as a table of lines and what {!parse} must make of them. The repository's own
      scripts exercise two shapes of it; every other entry comes from a review round that found the
      parser wrong about that line, and is kept so the case cannot regress quietly. *)
  let shebang_cases =
    [
      (* What this repository's scripts use. A direct shebang renders as the PATH it names and an
         `env` one as a bare name (round 4): the kernel execs the file `#!/bin/bash` names, while
         `env` performs a PATH lookup. Flattening both to "bash" let a macOS `/bin/bash` 3.2 script
         be accepted by a Homebrew bash 5. *)
      ("#!/bin/bash", "/bin/bash");
      ("#!/usr/bin/env bash", "bash via /usr/bin/env");
      ("#!/bin/sh", "/bin/sh");
      ("#!/bin/dash", "/bin/dash");
      ("", "sourced");
      (* `env -S`, all three spellings (the attached one is round 6's). *)
      ( "#!/usr/bin/env -S bash",
        "refused (`env -S` support is a property of the env binary, not its path, and buys nothing \
         here since no shell arguments are accepted -- write `#!/usr/bin/env <shell>`)" );
      ( "#!/usr/bin/env -Sbash",
        "refused (`env -S` support is a property of the env binary, not its path, and buys nothing \
         here since no shell arguments are accepted -- write `#!/usr/bin/env <shell>`)" );
      ( "#!/usr/bin/env --split-string=bash",
        "refused (`env -S` support is a property of the env binary, not its path, and buys nothing \
         here since no shell arguments are accepted -- write `#!/usr/bin/env <shell>`)" );
      (* Not a shebang: `#!` must be the first two bytes, so the file is run by the caller's shell
         and gets the no-shebang treatment (round 5). *)
      (" #!/bin/bash", "sourced");
      (* CRLF: the kernel keeps the CR in the interpreter path, so the file cannot exec (126) even
         though `bash -n` on it exits 0 (round 7). *)
      ( "#!/bin/bash\r",
        "refused (the shebang line ends CRLF, and the kernel keeps the CR in the interpreter path \
         (`bad interpreter`); this file needs LF endings)" );
      (* No arguments, in either direction and for every reason rounds 5 to 7 found: an operand is
         EXECUTED (round 6's P1, measured with a marker), `--exec` turns execution back on after
         `-n` turned it off (round 7's P1), `-t` makes `bash -n` exit 0 on a file whose second line
         is a syntax error, `--` makes `-n` the filename (127), and `bash -n --posix` is itself
         rejected (2) because bash wants long options first. No placement of `-n` is safe among
         them, so none of them is accepted. *)
      ( "#!/bin/bash helper.sh",
        "refused (`bash` is given helper.sh, and this check runs `<shell> -n <file>` and nothing \
         else -- no placement of `-n` is safe among shell options)" );
      ( "#!/bin/zsh --exec",
        "refused (`zsh` is given --exec, and this check runs `<shell> -n <file>` and nothing else \
         -- no placement of `-n` is safe among shell options)" );
      ( "#!/bin/bash --posix",
        "refused (`bash` is given --posix, and this check runs `<shell> -n <file>` and nothing \
         else -- no placement of `-n` is safe among shell options)" );
      ( "#!/bin/bash -eu",
        "refused (`bash` is given -eu, and this check runs `<shell> -n <file>` and nothing else -- \
         no placement of `-n` is safe among shell options)" );
      ( "#!/bin/bash -O extglob",
        "refused (`bash` is given -O extglob, and this check runs `<shell> -n <file>` and nothing \
         else -- no placement of `-n` is safe among shell options)" );
      (* Kernel semantics (round 3): everything after the interpreter path is ONE argument, so an
         `env` shebang without `-S` names no command. This one measurably HANGS -- env execs the
         script, which re-enters the same shebang. *)
      ( "#!/usr/bin/env -u FOO bash",
        "refused (the kernel passes `-u FOO bash` to `env` as ONE argument -- there is no command \
         by that name)" );
      (* `env` builds an environment, and both the lookup and the shell's startup depend on it
         (rounds 5 and 6). Refused rather than emulated. *)
      ( "#!/usr/bin/env PATH=/missing",
        "refused (`env` assignment `PATH=/missing`: the command is looked up, and parses, in an \
         environment this check cannot reproduce)" );
      ( "#!/usr/bin/env -i",
        "refused (`env` option this check cannot vouch for: -i -- it builds an environment the \
         command's parsing depends on)" );
      (* `env` semantics belong to the canonical paths, not to every basename `env` (round 8): the
         kernel would try to exec `/opt/custom/env`, which may not exist and need not be GNU env,
         while this check was resolving `bash` on PATH and passing the file. *)
      ("#!/opt/custom/env bash", "refused (interpreter whose `-n` this check cannot vouch for: env)");
      (* Both accepted env paths are named in the rendering, because BOTH have to exist for the
         script to launch and they are not equally present -- most distributions ship only
         /usr/bin/env, and round 9 caught this check accepting /bin/env without asking. *)
      ("#!/bin/env bash", "bash via /bin/env");
      (* An interpreter whose `-n` is not a parse at all. *)
      ( "#!/usr/bin/env python3",
        "refused (interpreter whose `-n` this check cannot vouch for: python3)" );
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

    The argument vector is fixed -- program, `-n`, file -- and nothing from the shebang goes into
    it. That is what makes the promise to execute nothing structural rather than a property of the
    parser: round 6's P1 (`bash helper.sh -n target` executed helper.sh) and round 7's (`zsh -n
    --exec path` turns execution back on) both needed a shebang token to reach this vector, and none
    can. It also removes the ordering question round 7 raised in the other direction, where `bash -n
    --posix path` exits 2 because bash wants long options first.

    A missing [prog] arrives as exit 127 on Unix (the forked child cannot exec and exits with it)
    and as [Unix_error] on Windows; both mean the same thing here, so both become [None]. *)
let parse_check prog path =
  let tmp = Stdlib.Filename.temp_file "ocannl_shell_parse" ".log" in
  let devnull = if Stdlib.Sys.win32 then "NUL" else "/dev/null" in
  Exn.protect
    ~finally:(fun () -> try Stdlib.Sys.remove tmp with _ -> ())
    ~f:(fun () ->
      let out = Unix.openfile tmp [ Unix.O_WRONLY; Unix.O_TRUNC ] 0o600 in
      let inp = Unix.openfile devnull [ Unix.O_RDONLY ] 0o400 in
      let argv = [| prog; "-n"; path |] in
      let status =
        Exn.protect
          ~finally:(fun () ->
            Unix.close out;
            Unix.close inp)
          ~f:(fun () ->
            match Unix.create_process_env prog argv (force isolated_environment) inp out out with
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
            Option.is_some (parse_check prog tmp)))

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

(** Whether a path names something this host can execute. Existence is not enough -- a present but
    non-executable interpreter fails exec just as surely -- so this asks the kernel's question. *)
let executable path =
  try
    Unix.access path [ Unix.X_OK ];
    true
  with Unix.Unix_error _ -> false

(** Resolve one wanted interpreter to a program that exists here, or say why it does not.

    Two rules, and round 9 is what made them one rule rather than two. Every path the KERNEL would
    exec must exist: the interpreter of a direct shebang, and the `env` binary of an `env` one. An
    absent one means the script cannot launch at all on this host, so reporting success for it would
    be reporting on something unrunnable. Windows is the exception throughout, and the only one: the
    shebang is honoured there by the shell rather than the kernel, so `/bin/sh` never resolves as a
    literal path and the basename on PATH is the faithful reading rather than a papering-over.

    A shell that a shebang NAMES is never substituted (round 7). Stand-ins apply only when [ours] is
    set -- the no-shebang case, where `sh` and `bash` are this check's own choice of checkers rather
    than anything the file asked for. *)
let rec resolve ~rel ?(ours = false) launch =
  let kernel_path_missing path =
    Error
      (Printf.sprintf "the kernel cannot exec `%s` on this host, so this script cannot run at all"
         path)
  in
  match launch with
  | Shebang.Path path ->
      if available path then Ok path
      else if Stdlib.Sys.win32 then (
        let name = Shebang.basename path in
        eprintf "%s: no `%s` on this host (Windows resolves the shebang itself), using `%s`\n" rel
          path name;
        resolve ~rel ~ours (Shebang.Via_env { env_path = ""; command = name }))
      else kernel_path_missing path
  | Shebang.Via_env { env_path; command } -> (
      if
        (* [env_path] is empty only for the Windows fallback above and the no-shebang checkers,
           where there is no env binary in the picture. *)
        (not (String.is_empty env_path)) && (not Stdlib.Sys.win32) && not (executable env_path)
      then kernel_path_missing env_path
      else if available command then Ok command
      else if not ours then
        Error
          (Printf.sprintf
             "`%s` is not installed here, and a shell a shebang names is not substituted" command)
      else
        match List.find (stand_ins command) ~f:available with
        | Some substitute ->
            eprintf "%s: no `%s` on this host, parsing with `%s` instead\n" rel command substitute;
            Ok substitute
        | None -> Error (Printf.sprintf "neither `%s` nor a stand-in is installed here" command))

(** The first line of a file, or [None] if it cannot be read as text at all. Binary files reach here
    through the directory globs, so a failure to read one is "not a script", not an error.

    [~fix_win_eol:false] is load-bearing, not tidiness: Stdio strips a trailing CR by DEFAULT, which
    silently repaired exactly the CRLF shebang round 7 is about -- the parser refused
    ["#!/bin/bash\r"] in its own table while the file it was handed arrived already normalised, so
    the check passed a script the kernel cannot exec. Read the bytes the kernel would read. *)
let first_line_of path =
  try In_channel.with_file path ~f:(In_channel.input_line ~fix_win_eol:false) with _ -> None

let () =
  if Array.length Stdlib.Sys.argv < 2 then (
    eprintf "Usage: %s <workspace_root> <ocannl_config and shell scripts...>\n" Stdlib.Sys.argv.(0);
    Stdlib.exit 1);
  (* The shebang grammar, on lines built to break it rather than on the two shapes this repository's
     scripts happen to use. Each case reads as what the parser must make of the line, so the golden
     is the specification. *)
  List.iter Shebang.shebang_cases ~f:(fun (line, expected) ->
      let actual = Shebang.render (Shebang.parse line) in
      if not (String.equal actual expected) then eprintf "shebang %S read as `%s`\n" line actual;
      Verdict.pf "shebang %S reads as `%s`" line expected (String.equal actual expected));
  (* The scope predicate, pinned separately from the parse table: a line can be one this check must
     report on precisely BECAUSE its arguments are refused, so the two questions do not answer each
     other. Phrased so that `true` is the passing reading either way. *)
  List.iter Shebang.scope_cases ~f:(fun (line, expected) ->
      let actual = Shebang.mentions_a_shell line in
      Verdict.pf "shebang %S is %s" line
        (if expected then "a shell script this check reports on" else "outside this check's scope")
        (Bool.equal actual expected));
  let base = base_dir Stdlib.Sys.argv.(1) in
  (* Reported repository-relative, opened as dune handed them over: the working directory is deep in
     the build tree and the paths arrive relative to it. *)
  (* A `.sh` file is in scope by its name; anything else the globs hand over is in scope only if it
     actually starts with a shell shebang, which is how `tools/run-tests` gets covered without the
     rule depending on every file in the repository (round 6). Reading two bytes off each candidate
     is the price, over the few hundred files in `tools/` and `scripts/`. *)
  let in_scope path =
    String.is_suffix path ~suffix:".sh"
    || Shebang.mentions_a_shell (Option.value ~default:"" (first_line_of path))
  in
  let scripts =
    Array.to_list Stdlib.Sys.argv |> Fn.flip List.drop 2
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
          let ours = match parsed with Shebang.Sourced -> true | Shebang.Interp _ -> false in
          let wanted =
            match parsed with
            | Shebang.Sourced ->
                [
                  Shebang.Via_env { env_path = ""; command = "sh" };
                  Shebang.Via_env { env_path = ""; command = "bash" };
                ]
            | Shebang.Interp launch -> [ launch ]
          in
          let resolutions = List.map wanted ~f:(resolve ~rel ~ours) in
          let usable = List.filter_map resolutions ~f:Result.ok in
          (* EVERY wanted checker has to resolve, not merely one of them (round 10). A sourced file
             is required to parse under both `sh` and `bash`, and reporting only when the whole list
             failed meant that a host missing `sh` and its stand-ins checked `tools/opam-env.sh`
             under bash alone -- silently, since the surviving checker kept `usable` non-empty and
             the golden line still read `parses: true`. A skipped grammar is not a passing one. *)
          if List.exists resolutions ~f:Result.is_error then
            List.iter resolutions ~f:(function
              | Error reason -> Verdict.fail (Printf.sprintf "%s: %s" rel reason)
              | Ok _ -> ())
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
  Verdict.p "the scan reached the session hook and the suite runner" (List.is_empty missing);
  Test_utils.Refusal_control_manifest.print "shell_scripts_parse.ml"
