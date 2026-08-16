(** Reading dune files for the places that run a test executable.

    A test executable resolves its configuration by walking UP from its working directory, which
    under dune is [_build/default/<test dir>] — so it reads the [ocannl_config] that
    [(copy_files ../config/ocannl_config)] materialized there, but only if the stanza that runs it
    DEPENDS on that file. Nothing is sandboxed, so a stanza that omits the dep does not fail: it
    reads whatever happens to be in the directory when it runs, which makes both the run and any
    [.exe.output] probe order-dependent (gh-ocannl-586, and gh-ocannl-597 for why a convention in
    prose was not enough).

    This module answers, for one dune file, the two questions that check has to ask: which of its
    stanzas run a test executable, and whether each of them declares the dependency.

    {1 Reading dune files as s-expressions}

    Dune's syntax is s-expressions with [;] line comments and quoted strings, which is what sexplib
    reads — so this parses rather than splitting on parens. The lesson is
    {!Config_key_scan}'s: an approximation of a grammar has no natural stopping point, and every
    mistake here is silent in the same way, since a stanza the scan fails to recognise looks exactly
    like a stanza that does not exist.

    Sexplib is not dune's own reader, and the two disagree in exactly two places, both handled:

    - [#|…|#] and [#;] are comments to sexplib and ordinary atom characters to dune. A file
      containing either would be read with a hole in it, so {!stanzas} refuses it instead.
    - Dune's multi-line string blocks (a quoted string opening with a backslash and a bar) use an
      escape sexplib rejects, so such a file raises rather than being misread. No dune file in this
      repository uses one; if one appears, the failure names the file. *)

open Base

let config_file = "ocannl_config"

(** [in_subdir parent child] joins two relative directories, either of which may be empty or [.] —
    [(chdir . …)] and a plain [./] say "here", and saying it must not make a directory look like a
    different one. *)
let in_subdir parent child =
  let clean d =
    if String.is_empty d || String.equal d "." then ""
    else Option.value (String.chop_prefix d ~prefix:"./") ~default:d
  in
  match (clean parent, clean child) with
  | "", child -> child
  | parent, "" -> parent
  | parent, child -> parent ^ "/" ^ child

(* How many stanzas a dune file has, counted by nothing but parentheses, skipping `;` comments and
   quoted strings. This is a CROSS-CHECK, not a second parser: it says how many top-level forms are
   there, and sexplib has to agree.

   The disagreement it exists to catch is `#|…|#` and `#;`, which sexplib reads as comments and
   dune does not, so sexplib could swallow a whole stanza. An earlier version refused any file
   CONTAINING those two characters, which is wrong in the ordinary case: inside a quoted argument
   or after a `;`, sexplib does not treat them as comments either, and refusing there would take
   the whole suite down over an unrelated string (Codex P2, round 12 of PR #343). Counting says
   precisely when something was swallowed. *)
let top_level_form_count content =
  let count = ref 0 and depth = ref 0 in
  let i = ref 0 in
  let length = String.length content in
  let delimiter c = Char.is_whitespace c || List.mem [ '('; ')'; ';'; '"' ] c ~equal:Char.equal in
  while !i < length do
    (match content.[!i] with
    | ';' ->
        while !i < length && not (Char.equal content.[!i] '\n') do
          Int.incr i
        done
    | '"' ->
        (* A quoted string is one form where a form can start. *)
        if !depth = 0 then Int.incr count;
        Int.incr i;
        let closed = ref false in
        while (not !closed) && !i < length do
          (match content.[!i] with
          | '\\' -> Int.incr i
          | '"' -> closed := true
          | _ -> ());
          Int.incr i
        done;
        Int.decr i
    | '(' ->
        if !depth = 0 then Int.incr count;
        Int.incr depth
    | ')' -> if !depth > 0 then Int.decr depth
    | c when Char.is_whitespace c -> ()
    | _ ->
        (* A bare atom is a form too, even at the top level, where dune would reject it and sexplib
           happily returns it -- so the two still agree on the count and the caller reports it. *)
        if !depth = 0 then Int.incr count;
        while !i < length && not (delimiter content.[!i]) do
          Int.incr i
        done;
        Int.decr i);
    Int.incr i
  done;
  !count

(** Raises if [content] is not something both readers agree on: a scan that cannot read its input
    must say so rather than report a file with a hole in it. *)
let stanzas content =
  let parsed = Sexplib.Sexp.scan_sexps (Lexing.from_string content) in
  let counted = top_level_form_count content in
  if List.length parsed <> counted then
    failwith
      (Printf.sprintf
         "dune file parses as %d stanzas but has %d top-level forms -- sexplib and dune disagree \
          about what is a comment here (`#|…|#` and `#;` are comments to the one and atoms to the \
          other), so this scan would read the file with a hole in it"
         (List.length parsed) counted);
  parsed

let head = function Sexp.List (Sexp.Atom h :: _) -> Some h | _ -> None

(** The arguments of a stanza's [(<name> …)] field, where it has one. *)
let field_in fields name =
  List.find_map fields ~f:(function
    | Sexp.List (Sexp.Atom f :: args) when String.equal f name -> Some args
    | _ -> None)

let field stanza name =
  match stanza with Sexp.List (_ :: fields) -> field_in fields name | _ -> None

let rec atoms = function Sexp.Atom a -> [ a ] | Sexp.List l -> List.concat_map l ~f:atoms

(** Whether a [copy_files] pattern could name [name] — the literal spelling, or a wildcard that
    covers it. Only [*] and [?] are interpreted; a pattern using dune's set syntax ([{a,b}], [[ab]])
    is taken as possibly matching, since this decides where a config EXISTS and guessing wide only
    risks accepting a directory that has one anyway, while guessing narrow rejects a correctly
    configured one (Codex P2, round 12 of PR #343). *)
let glob_could_match pattern ~name =
  if String.exists pattern ~f:(fun c -> List.mem [ '{'; '[' ] c ~equal:Char.equal) then true
  else
    let pattern = String.to_array pattern and name = String.to_array name in
    let rec go p n =
      if p = Array.length pattern then n = Array.length name
      else
        match pattern.(p) with
        | '*' -> go (p + 1) n || (n < Array.length name && go p (n + 1))
        | '?' -> n < Array.length name && go (p + 1) (n + 1)
        | c -> n < Array.length name && Char.equal c name.(n) && go (p + 1) (n + 1)
    in
    go 0 0

(** Dependency-specification forms that really depend on the file they name, so that the config
    named inside one is built before the test runs.

    Only [(file …)] and a bare path. Dune's dependency language has forms that name something else
    entirely — [(alias ocannl_config)], [(env_var ocannl_config)], [(package ocannl_config)] — and
    reading those as a file dependency lets a stanza pass while depending on no config at all
    (Codex P2, round 3 of PR #343). The GLOB forms are excluded for a subtler reason of the same
    kind (round 7): they match the source tree, and in a directory whose config arrives through
    [(copy_files …)] the file is a generated target, so [(deps (glob_files ocannl_config))] matches
    nothing and builds nothing. Checked against dune 3.18 in a cleaned build directory, where such
    a rule fails with "cat: ocannl_config: No such file or directory" exactly as a rule with no
    deps at all does.

    Anything not listed does not count, which fails safe: a form that does depend on the file fails
    loudly and is added, rather than one that does not silently passing. *)
let file_dep_forms = [ "file" ]

(** Forms that name paths, for finding what a [(:name …)] binding might point at. Wider than
    {!file_dep_forms} on purpose: here the question is what an executable could be called, and
    recognising more spellings only means recognising more of the executables that get run. *)
let path_bearing_forms = [ "file"; "glob_files"; "glob_files_rec"; "source_tree" ]

(** Every config file a [(deps …)] field really depends on, as the path is written — relative to
    the stanza's directory.

    Which of them is the one that MATTERS this module cannot say. OCANNL walks UP from the process
    directory and reads the first config it finds ([Utils.config_file_args]), so the dependency has
    to name that one: an ancestor's will do while no nearer directory has its own, and a rule that
    chdirs out of the stanza's subtree may legitimately name a common parent (Codex P2, rounds 9
    to 11 of PR #343). Where the files exist, and hence which directory wins, is the caller's
    knowledge. *)
let declared_config_paths args =
  let rec collect sexp =
    match sexp with
    | Sexp.Atom atom ->
        if String.equal (Stdlib.Filename.basename atom) config_file then [ atom ] else []
    | Sexp.List (Sexp.Atom head :: rest)
      when String.is_prefix head ~prefix:":" || List.mem file_dep_forms head ~equal:String.equal ->
        List.concat_map rest ~f:collect
    | Sexp.List _ -> []
  in
  match args with
  | None -> []
  | Some args -> List.concat_map args ~f:collect |> List.dedup_and_sort ~compare:String.compare

(** The names a [(name …)] or [(names …)] field gives, in order. *)
let names_of stanza =
  match field stanza "name" with
  | Some [ Sexp.Atom name ] -> [ name ]
  | _ -> (
      match field stanza "names" with
      | Some args -> List.filter_map args ~f:(function Sexp.Atom n -> Some n | _ -> None)
      | _ -> [])

(** A command atom, split at dune's [%{…}] boundaries and nowhere else.

    An earlier version tokenized on an allowlist of "characters a path may contain", which quietly
    cut valid filenames in half: [%{dep:helper+pp.exe}] came out as [pp.exe] and matched an
    exemption written for a different executable (Codex P2, round 4 of PR #343). A filename
    allowlist is a guess about the filesystem; [%{] and [}] are dune's own delimiters, and they are
    all this needs to know. *)
type piece = Literal of string | Pform of string

let pieces atom =
  let length = String.length atom in
  let rec go acc position =
    let literal_from start = if start < length then [ Literal (String.subo atom ~pos:start) ] else [] in
    match String.substr_index atom ~pos:position ~pattern:"%{" with
    | None -> List.rev acc @ literal_from position
    | Some start -> (
        let acc =
          if start > position then Literal (String.sub atom ~pos:position ~len:(start - position)) :: acc
          else acc
        in
        match String.index_from atom (start + 2) '}' with
        (* Unterminated: not a pform at all, so the rest is text. *)
        | None -> List.rev acc @ literal_from start
        | Some stop ->
            go (Pform (String.sub atom ~pos:(start + 2) ~len:(stop - start - 2)) :: acc) (stop + 1))
  in
  go [] 0

(** What a [(run …)] action's command names.

    {1 Why command position, and why an unrecognized command fails}

    Looking for a [.exe] anywhere in the stanza is the tempting rule and the wrong one in both
    directions: it counts a rule that merely copies an executable, and it misses every way of
    naming one that does not spell the extension — [%{bin:probe}] the first among them (Codex P2,
    round 2 of PR #343), which is the same fall-through as round 1's [(alias …)] one spelling
    further in. Patching spellings one at a time invites a further round, so what this reads is the
    command position of a program action, and a command it cannot place FAILS rather than counting
    as nothing.

    The spellings it places, all of them present in this repository or in dune's own manual: a
    literal path ending in [.exe]; [%{dep:…}] and [%{exe:…}], which name a path the same way;
    [%{bin:name}], which resolves a PUBLIC executable — conservatively a site, because dune resolves
    it from this workspace before PATH, and an external tool that reads no configuration is what the
    exemption list is for; [%{name}] bound by a named dependency [(:name pp.exe)], which the action
    reaches without ever spelling the file; and the toolchain pforms, which run a compiler. A bare
    word is a tool on PATH ([python3], [diff]): not something this repository builds, so not a
    site. *)
type command =
  | Runs of string  (** the executable, by the path written or the name [%{bin:…}] gave *)
  | External  (** a tool on PATH or in the toolchain, which this repository does not build *)
  | Unrecognized of string  (** command position this scan cannot read — reported, never ignored *)
  | Unknown_directory of string
      (** a shell command line: not only is the program unreadable, the directory it runs in is
          too, because [cd] inside the line moves it without dune knowing (Codex P2, round 8) *)

(** Pforms naming a program that is part of the toolchain rather than of this repository. *)
let toolchain_pforms = [ "ocaml"; "ocamlc"; "ocamlopt"; "cc"; "cxx"; "make" ]

(** How a [(test)] stanza's custom action names the test binary. *)
let test_pform = "%{test}"

(* The path AS WRITTEN, less a leading [./] that only says "here". Reducing it to a basename would
   make `%{dep:../../tools/pp.exe}` and a local `pp.exe` the same identity, and an exemption
   naming one would cover the other -- the same collapse the config scanner's duplicate-basename
   check exists to prevent (Codex P2, round 3, and #340 round 10 before it). *)
let program_path path = Option.value (String.chop_prefix path ~prefix:"./") ~default:path

let is_executable path = String.is_suffix path ~suffix:".exe"

(* An explicit path is not a PATH lookup. `./probe` and `../tools/probe` name something this
   repository produced, whatever their extension, and stripping the `./` before asking loses the
   one thing that distinguishes them from `python3` (Codex P2, round 8). *)
let is_explicit_path path = String.is_prefix path ~prefix:"./" || String.contains path '/'

let classify_command ~named_deps cmd =
  let explicit = is_explicit_path cmd in
  let cmd = program_path cmd in
  match pieces cmd with
  | [ Literal path ] -> if is_executable path || explicit then Runs path else External
  | [ Pform pform ] -> (
      match String.lsplit2 pform ~on:':' with
      | Some (("dep" | "exe" | "path" | "file"), path) ->
          if is_executable path then Runs (program_path path) else Unrecognized cmd
      | Some ("bin", name) -> Runs name
      | Some _ -> Unrecognized cmd
      | None ->
          if String.equal pform "test" then Runs test_pform
          else if List.mem toolchain_pforms pform ~equal:String.equal then External
          else (
            (* [./%{pp}] with [(deps (:pp pp.exe))]: the action names the dependency, not the
               file. *)
            match List.Assoc.find named_deps pform ~equal:String.equal with
            | Some paths -> (
                match List.find paths ~f:is_executable with
                | Some path -> Runs (program_path path)
                (* A binding that resolves to no executable is not evidence of an external tool:
                   the action runs whatever it binds, and this scan did not recognise it. *)
                | None -> Unrecognized cmd)
            | None -> Unrecognized cmd))
  (* Text and pforms mixed, or a pform inside a path: rather than guess where the program's name
     begins, say so. *)
  | _ -> Unrecognized cmd

(** The [(:name …)] bindings of a stanza's [(deps …)] field, with the paths each one binds.

    The paths are collected through the dependency forms that wrap them, because
    [(:runner (file probe.exe))] binds an executable as surely as [(:runner probe.exe)] does, and
    keeping only the bare atoms lost it — after which the binding looked empty and the command
    reading it looked external (Codex P2, round 6 of PR #343). *)
let named_deps_of stanza =
  let rec paths sexp =
    match sexp with
    | Sexp.Atom a -> [ a ]
    | Sexp.List (Sexp.Atom head :: rest) when List.mem path_bearing_forms head ~equal:String.equal
      ->
        List.concat_map rest ~f:paths
    | Sexp.List _ -> []
  in
  match field stanza "deps" with
  | None -> []
  | Some args ->
      List.filter_map args ~f:(function
        | Sexp.List (Sexp.Atom name :: values) when String.is_prefix name ~prefix:":" ->
            Some (String.drop_prefix name 1, List.concat_map values ~f:paths)
        | _ -> None)

(** Actions that execute a program. [dynamic-run] is here because dune runs one there too (Codex
    P2, round 3); [system] and [bash] hand a command line to a shell. *)
let program_actions = [ "run"; "dynamic-run"; "system"; "bash" ]

(** Every other head dune's action language admits, including the predicate heads that appear in a
    [(with-accepted-exit-codes …)] test. None of them executes a program of its own — they nest
    actions, move bytes around, or compare files.

    The list is here so that {!unclassified_action_heads} can report a head on neither list. Three
    review rounds found this scan missing a way to run something — a stanza kind, then a command
    spelling, then [dynamic-run] — each patched instance leaving the next one waiting. What ends
    that is the fall-through, not the third patch: dune's action vocabulary is closed and short, so
    the scan can say what it knows and fail on the rest. *)
let inert_actions =
  [
    "progn"; "concurrent"; "chdir"; "setenv"; "with-stdout-to"; "with-stderr-to";
    "with-outputs-to"; "with-stdin-from"; "with-accepted-exit-codes"; "ignore-stdout";
    "ignore-stderr"; "ignore-outputs"; "no-infer"; "echo"; "write-file"; "cat"; "copy"; "copy#";
    "copy-and-add-line-directive"; "diff"; "diff?"; "cmp"; "pipe-stdout"; "pipe-stderr";
    "pipe-outputs"; "format-dune-file"; "or"; "and"; "not";
  ]

(** What an action puts in a position where a program could be named. [Elsewhere] wraps another
    of these with the destination of a [chdir] this scan cannot resolve. *)
type command_site =
  | Program of string
  | Argument of string * string  (** the command, and one of its arguments *)
  | Shell of string
  | Elsewhere of string * command_site

(* Every command an action runs, at any depth: [(with-stdout-to … (run …))],
   [(no-infer (progn (run …) …))] and the rest nest the one that matters. Each comes with the
   directory the process will run in, relative to the stanza's own: [chdir] moves it, and the
   configuration an OCANNL executable finds is the one it searches upward from THERE, not the one
   next to the dune file (Codex P2, round 6 of PR #343). *)
let rec commands_in ?(cwd = "") sexp =
  let nested =
    match sexp with Sexp.List l -> List.concat_map l ~f:(commands_in ~cwd) | _ -> []
  in
  match sexp with
  (* A destination built out of a pform ([%{workspace_root}]) is not a path this scan can resolve,
     and treating it as a literal directory name would have the process searching from a directory
     that does not exist -- possibly landing back on the stanza's own config (Codex P2, round 12).
     Everything under it is reported as running somewhere unestablished. *)
  | Sexp.List (Sexp.Atom "chdir" :: Sexp.Atom dir :: rest)
    when String.is_substring dir ~substring:"%{" ->
      List.concat_map rest ~f:(commands_in ~cwd)
      |> List.map ~f:(fun (_, command) ->
             let named =
               match command with
               | Program cmd -> cmd
               | Argument (_, arg) -> arg
               | Shell line -> "shell: " ^ line
               | Elsewhere (what, _) -> what
             in
             (* Kept as the command it is, tagged with the destination: an unresolvable directory
                only matters for something that might read configuration there, and a PATH tool
                reads none wherever it runs (Codex P2, round 13). *)
             (cwd, Elsewhere (Printf.sprintf "%s, under `(chdir %s ...)`" named dir, command)))
  | Sexp.List (Sexp.Atom "chdir" :: Sexp.Atom dir :: rest) ->
      List.concat_map rest ~f:(commands_in ~cwd:(in_subdir cwd dir))
  | Sexp.List (Sexp.Atom ("run" | "dynamic-run") :: Sexp.Atom cmd :: args) ->
      (* A PATH tool handed something this repository builds may be launching it (`env probe.exe`)
         or may be reading it (`diff old.exe new.exe`) -- dune's grammar does not say which, and
         nothing structural distinguishes them (Codex P2, rounds 12 and 13). So neither guess is
         made: the pair is reported as a command this scan cannot place, which the check settles
         the way it settles every other one -- the rule declares the dependency, or names an
         exemption with the reason. *)
      ((cwd, Program cmd)
      :: List.filter_map args ~f:(function
           | Sexp.Atom arg -> Some (cwd, Argument (cmd, arg))
           | _ -> None))
      @ nested
  (* A shell action hands a command line to a shell, and this scan does not parse shell. Splitting
     it on whitespace looked like reading it and was not: `if ready; then ./probe.exe; fi` yields
     `./probe.exe;`, which ends in no extension and passes for an external tool -- so the rule runs
     a test executable and the check says nothing (Codex P2, round 5). A shell line is reported
     whole instead, and as something stronger than an unreadable command: `cd ../sibling &&
     ./probe.exe` moves the working directory with no dune `chdir` to show for it, so the directory
     whose config the process will find is unknown too (round 8). *)
  | Sexp.List (Sexp.Atom ("bash" | "system") :: args) ->
      List.filter_map args ~f:(function Sexp.Atom a -> Some (cwd, Shell a) | _ -> None) @ nested
  | _ -> nested

(** Heads inside a stanza's [(action …)] that are on neither action list, each once. *)
let unclassified_action_heads stanza =
  let rec walk_action sexp =
    match sexp with
    | Sexp.Atom _ -> []
    | Sexp.List (Sexp.Atom head :: args) ->
        let nested =
          (* A program action's arguments are its command line, not further actions. *)
          if List.mem program_actions head ~equal:String.equal then []
          else List.concat_map args ~f:walk_action
        in
        if
          List.mem program_actions head ~equal:String.equal
          || List.mem inert_actions head ~equal:String.equal
        then nested
        else head :: nested
    | Sexp.List l -> List.concat_map l ~f:walk_action
  in
  match field stanza "action" with
  | None -> []
  | Some args -> List.concat_map args ~f:walk_action |> List.dedup_and_sort ~compare:String.compare

(** What a stanza runs, each with the directory it runs in. *)
let executables_run stanza =
  let named_deps = named_deps_of stanza in
  let rec classify command =
    match command with
    | Program cmd -> classify_command ~named_deps cmd
    | Shell line -> Unknown_directory ("shell: " ^ line)
    (* Only what could read a configuration cares which directory it runs in: a PATH tool reads
       none wherever it is (Codex P2, round 13). *)
    | Elsewhere (what, command) -> (
        match classify command with External -> External | _ -> Unknown_directory what)
    (* An argument matters only when it names an executable -- everything else a command line
       carries (flags, inputs, targets) is not something being run. Which of the two it is, this
       scan does not guess. *)
    | Argument (cmd, arg) -> (
        match classify_command ~named_deps arg with
        | Runs name -> Unrecognized (Printf.sprintf "%s, handed %s" cmd name)
        | External | Unrecognized _ | Unknown_directory _ -> External)
  in
  List.filter_map (commands_in stanza) ~f:(fun (cwd, command) ->
      match classify command with External -> None | classified -> Some (cwd, classified))
  |> List.dedup_and_sort ~compare:Poly.compare

type kind =
  | Test  (** a [(test)] or [(tests)] stanza, which dune runs itself *)
  | Inline_tests  (** a [(library)] with an [(inline_tests)] field, ditto *)
  | Runs_executable  (** a [(rule)] that runs an executable — where an [(executable)] stanza's
                         dependencies have to live, there being no [deps] field on one *)
  | Unreadable_command  (** a [(run …)] whose command this scan cannot place: reported, so that
                            what it runs is settled by a reader rather than by silence *)
  | Unreadable_directory  (** a shell action: the directory the process ends up in is unknown, so
                              no dependency of this stanza's can be shown to be the right one *)
  | Unclassified_action  (** an action head on neither {!program_actions} nor {!inert_actions} —
                             it might run a program, so it is reported too *)

(** [subdir] is the directory the stanza applies to, relative to the dune file's own: empty at the
    top level, and the path a [(subdir …)] wrapper names inside one. A wrapped stanza runs
    elsewhere, so it is that directory's config it needs — test/operations/dune configures
    test/operations/config this way.

    [cwd] is the directory the process runs in, relative to the stanza's: empty unless a [chdir]
    moved it. The two compose — [in_subdir subdir cwd] is the directory whose configuration the
    executable will actually find, and the one that has to have one. *)
type site = {
  kind : kind;
  name : string;
  declares_config : bool;
      (** whether the deps depend on any config file at all; WHICH one had to be named is
          {!declared_config_paths}, since only the caller knows where configs exist *)
  declared_config_paths : string list;
  subdir : string;
  cwd : string;
}

(* [(subdir <dir> <stanza>…)] applies its body to another directory. Descending into it is what
   keeps its stanzas subject to the same rules; ignoring it would drop them silently, which is the
   one thing this scan must not do. *)
let rec walk dir stanzas ~f =
  List.concat_map stanzas ~f:(fun stanza ->
      match stanza with
      | Sexp.List (Sexp.Atom "subdir" :: Sexp.Atom sub :: body) -> walk (in_subdir dir sub) body ~f
      | stanza -> f dir stanza)

let kind_name = function
  | Test -> "test"
  | Inline_tests -> "inline tests"
  | Runs_executable -> "rule running"
  | Unreadable_command -> "rule whose command this scan cannot read:"
  | Unreadable_directory -> "rule whose working directory this scan cannot establish:"
  | Unclassified_action -> "rule with an action this scan cannot place:"

(** Stanza heads that carry an action, so one of them may run a test executable. [alias] is here
    for the same reason [rule] is: it took an [action] field before dune 2.0, and it can still
    depend on an executable. *)
let action_heads = [ "rule"; "alias" ]

(** Stanza heads classified as running no test executable. Declaring things to build ([executable],
    [ocamllex]), placing files ([copy_files], [install]), or describing the directory ([env],
    [dirs]) — none of them run anything of their own.

    The list exists so that {!unclassified_heads} can report what is on neither list. A head nobody
    has classified might carry an action, and passing over it in silence is the failure this whole
    scan exists to end (Codex P2, round 1 of PR #343) — [cram] is the live example: dune runs cram
    tests, this repository has none, and the day one appears the scan says so rather than counting
    it as nothing. *)
let inert_heads =
  [
    "executable"; "executables"; "copy_files"; "copy_files#"; "install"; "env"; "dirs";
    "data_only_dirs"; "vendored_dirs"; "include_subdirs"; "documentation"; "ocamllex"; "ocamlyacc";
    "menhir"; "toplevel"; "deprecated_library_name";
  ]

(** Every place in [content] that runs a test executable.

    An [(executable)] stanza is not one: it declares something to build, and dune runs it only
    where a rule says so — which is why a diagnostic executable such as [bench_circles_step] or a
    tutorial such as [gpt2_generate] needs no exemption from the check built on this. It is
    structurally not a site, rather than a name on a list someone has to keep true.

    What a rule runs is read from the command position of its [run] actions, in every spelling
    {!classify_command} places — and a command it cannot place becomes an {!Unreadable_command}
    site, which the caller fails on. *)
let sites content =
  walk "" (stanzas content) ~f:(fun subdir stanza ->
      let deps () = field stanza "deps" in
      let site ?(cwd = "") ?deps:(deps_field = deps ()) kind name =
        [
          {
            kind;
            name;
            declares_config = not (List.is_empty (declared_config_paths deps_field));
            declared_config_paths = declared_config_paths deps_field;
            subdir;
            cwd;
          };
        ]
      in
      let stanza_name () = String.concat ~sep:", " (names_of stanza) in
      (* Everything a stanza's actions run, each with the directory it runs in. A `(test)` may
         carry a custom action, so this serves both branches; the difference is only WHICH of the
         commands is the test itself. *)
      let run = executables_run stanza in
      let sites_for ~is_test =
        List.map run ~f:fst |> List.dedup_and_sort ~compare:String.compare
        |> List.concat_map ~f:(fun cwd ->
               let for_cwd f =
                 List.filter_map run ~f:(fun (c, command) ->
                     if String.equal c cwd then f command else None)
               in
               let exes =
                 for_cwd (function
                   (* In a test stanza, `%{test}` is the test binary itself, reported as the Test
                      site rather than as something the action also runs. *)
                   | Runs name when is_test && String.equal name test_pform -> None
                   | Runs name -> Some name
                   | _ -> None)
               in
               let unreadable = for_cwd (function Unrecognized cmd -> Some cmd | _ -> None) in
               let unlocatable = for_cwd (function Unknown_directory cmd -> Some cmd | _ -> None) in
               (if List.is_empty exes then []
                else site ~cwd Runs_executable (String.concat ~sep:", " exes))
               @ List.concat_map unreadable ~f:(fun cmd -> site ~cwd Unreadable_command cmd)
               @ List.concat_map unlocatable ~f:(fun cmd -> site ~cwd Unreadable_directory cmd))
      in
      match head stanza with
      | Some ("test" | "tests") ->
          (* Where the TEST runs, which is where its own command runs -- not where a helper in the
             same action happens to be sent (Codex P2, round 10). With no custom action, dune runs
             it in the stanza's directory. *)
          let test_cwds =
            List.filter_map run ~f:(function
              | cwd, Runs name when String.equal name test_pform -> Some cwd
              | _ -> None)
            |> List.dedup_and_sort ~compare:String.compare
          in
          let test_cwds = if List.is_empty test_cwds then [ "" ] else test_cwds in
          List.concat_map test_cwds ~f:(fun cwd -> site ~cwd Test (stanza_name ()))
          @ sites_for ~is_test:true
      | Some "library" -> (
          match field stanza "inline_tests" with
          | None -> []
          | Some inline -> site ~deps:(field_in inline "deps") Inline_tests (stanza_name ()))
      | Some h when List.mem action_heads h ~equal:String.equal ->
          (* One site per directory the rule runs something in: what each needs is that
             directory's config, declared by the path that reaches it from here. *)
          sites_for ~is_test:false
          @ List.concat_map (unclassified_action_heads stanza) ~f:(fun head ->
                site Unclassified_action head)
      | _ -> [])

(** Stanza heads in [content] that {!sites} has no classification for, each once. The caller fails
    on them: see {!inert_heads} for why silence is not an option here. *)
let unclassified_heads content =
  walk "" (stanzas content) ~f:(fun _subdir stanza ->
      match head stanza with
      | Some h
        when List.mem action_heads h ~equal:String.equal
             || List.mem inert_heads h ~equal:String.equal
             || List.mem [ "test"; "tests"; "library"; "subdir" ] h ~equal:String.equal ->
          []
      | Some h -> [ h ]
      (* A bare atom or a list that does not start with one is not a stanza dune would accept, so
         it is reported the same way rather than passed over. *)
      | None -> [ "<not a stanza>" ])
  |> List.dedup_and_sort ~compare:String.compare

(** The directories this dune file materializes the shared configuration into with a
    [(copy_files …ocannl_config)] stanza, relative to its own as in {!site}. The other way for a
    directory to have one is a file checked in next to the dune file, which this cannot see and the
    caller supplies. *)
let config_copy_dirs content =
  walk "" (stanzas content) ~f:(fun subdir stanza ->
      match head stanza with
      (* [copy_files#] is the preprocessing spelling of the same stanza. A wildcard is read as
         possibly matching the config: this decides where a config EXISTS, so guessing wide only
         risks accepting a directory that has one by another route, while guessing narrow would
         reject a correctly configured one (Codex P2, round 12). *)
      | Some ("copy_files" | "copy_files#")
        when List.exists (atoms stanza) ~f:(fun atom ->
                 glob_could_match (Stdlib.Filename.basename atom) ~name:config_file) ->
          [ subdir ]
      | _ -> [])
