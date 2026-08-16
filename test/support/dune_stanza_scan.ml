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

(* Sexplib reads these as comments; dune does not. See the header. *)
let sexp_only_comment_markers = [ "#|"; "#;" ]

(** Raises if [content] is not something both readers agree on: a scan that cannot read its input
    must say so rather than report an empty file. *)
let stanzas content =
  List.iter sexp_only_comment_markers ~f:(fun marker ->
      if String.is_substring content ~substring:marker then
        failwith
          (Printf.sprintf
             "dune file uses %S, which sexplib reads as a comment and dune does not -- this scan \
              would read the file with a hole in it"
             marker));
  Sexplib.Sexp.scan_sexps (Lexing.from_string content)

let head = function Sexp.List (Sexp.Atom h :: _) -> Some h | _ -> None

(** The arguments of a stanza's [(<name> …)] field, where it has one. *)
let field_in fields name =
  List.find_map fields ~f:(function
    | Sexp.List (Sexp.Atom f :: args) when String.equal f name -> Some args
    | _ -> None)

let field stanza name =
  match stanza with Sexp.List (_ :: fields) -> field_in fields name | _ -> None

let rec atoms = function Sexp.Atom a -> [ a ] | Sexp.List l -> List.concat_map l ~f:atoms

(** Dependency-specification forms that carry FILE dependencies, so that the config file named
    inside one is really depended upon. Dune's dependency language also has forms that name
    something else entirely — [(alias ocannl_config)], [(env_var ocannl_config)],
    [(package ocannl_config)] — and reading those as a file dependency would let a stanza pass this
    check while depending on no config at all (Codex P2, round 3 of PR #343). Anything not listed
    here does not count, which fails safe: an exotic form that does name the file fails loudly and
    is added, rather than an exotic form that does not silently passing. *)
let file_dep_forms = [ "file"; "glob_files"; "glob_files_rec"; "source_tree"; "include" ]

(** Whether a [(deps …)] field really depends on the shared configuration file. Path-insensitive: a
    directory reaching for a config elsewhere ([../config/ocannl_config]) still declares it. *)
let rec dep_names_config sexp =
  match sexp with
  | Sexp.Atom atom -> String.equal (Stdlib.Filename.basename atom) config_file
  | Sexp.List (Sexp.Atom head :: rest) ->
      (* [(:name <deps>)] binds a name to ordinary dependencies; the forms above take paths. *)
      (String.is_prefix head ~prefix:":" || List.mem file_dep_forms head ~equal:String.equal)
      && List.exists rest ~f:dep_names_config
  | Sexp.List _ -> false

let declares_config args =
  match args with None -> false | Some args -> List.exists args ~f:dep_names_config

(** The names a [(name …)] or [(names …)] field gives, in order. *)
let names_of stanza =
  match field stanza "name" with
  | Some [ Sexp.Atom name ] -> [ name ]
  | _ -> (
      match field stanza "names" with
      | Some args -> List.filter_map args ~f:(function Sexp.Atom n -> Some n | _ -> None)
      | _ -> [])

(* An atom carries a path inside dune's own punctuation: [%{dep:mlp_names.exe}] and [./%{pp}] are
   one atom each. Split on everything a path cannot contain, so the parts can be read separately. *)
let path_char c =
  Char.is_alphanum c || List.mem [ '_'; '.'; '-'; '/'; '\\' ] c ~equal:Char.equal

let words atom =
  String.map atom ~f:(fun c -> if path_char c then c else ' ')
  |> String.split ~on:' '
  |> List.filter ~f:(Fn.non String.is_empty)

(** What a [(run …)] action's command names.

    {1 Why command position, and why an unrecognized command fails}

    Looking for a [.exe] anywhere in the stanza is the tempting rule and the wrong one in both
    directions: it counts a rule that merely copies an executable, and it misses every way of
    naming one that does not spell the extension — [%{bin:probe}] the first among them (Codex P2,
    round 2 of PR #343), which is the same fall-through as round 1's [(alias …)] one spelling
    further in. Patching spellings one at a time invites a third round, so what this reads is the
    command position of a [run] action, and a command it cannot place FAILS rather than counting
    as nothing.

    The spellings it does place, all of them present in this repository or in dune's own manual:
    a path ending in [.exe] however it is wrapped ([%{dep:foo.exe}], [%{exe:foo.exe}], [./foo.exe]);
    [%{bin:name}], which resolves a PUBLIC executable — conservatively a site, because dune resolves
    it from this workspace before PATH, and an external tool that reads no configuration is what the
    exemption list is for; and [%{name}] bound by a named dependency [(:name pp.exe)], which the
    action reaches without ever spelling the file. A bare word is a tool on PATH ([python3],
    [diff]): not something this repository builds, so not a site. *)
type command =
  | Runs of string  (** the executable, by basename or by the name [%{bin:…}] gave *)
  | External  (** a tool on PATH, which this repository does not build *)
  | Unrecognized of string  (** command position this scan cannot read — reported, never ignored *)

let pform_payload atom ~prefix =
  match String.substr_index atom ~pattern:("%{" ^ prefix) with
  | None -> None
  | Some start -> (
      let rest = String.drop_prefix atom (start + String.length prefix + 2) in
      match String.lsplit2 rest ~on:'}' with Some (payload, _) -> Some payload | None -> None)

(* The path AS WRITTEN, less a leading [./] that only says "here". Reducing it to a basename would
   make `%{dep:../../tools/pp.exe}` and a local `pp.exe` the same identity, and an exemption
   naming one would cover the other -- the same collapse the config scanner's duplicate-basename
   check exists to prevent (Codex P2, round 3, and #340 round 10 before it). *)
let program_path path = Option.value (String.chop_prefix path ~prefix:"./") ~default:path

let classify_command ~named_deps cmd =
  match List.find (words cmd) ~f:(String.is_suffix ~suffix:".exe") with
  | Some path -> Runs (program_path path)
  | None -> (
      match
        List.find_map [ "bin:"; "exe:" ] ~f:(fun prefix -> pform_payload cmd ~prefix)
      with
      | Some name -> Runs (program_path name)
      | None -> (
          (* [./%{pp}] with [(deps (:pp pp.exe))]: the action names the dependency, not the file. *)
          match pform_payload cmd ~prefix:"" with
          | None -> External
          | Some name -> (
              match List.Assoc.find named_deps name ~equal:String.equal with
              | Some paths -> (
                  match List.find paths ~f:(String.is_suffix ~suffix:".exe") with
                  | Some path -> Runs (program_path path)
                  | None -> External)
              | None -> Unrecognized cmd)))

(** The [(:name …)] bindings of a stanza's [(deps …)] field. *)
let named_deps_of stanza =
  match field stanza "deps" with
  | None -> []
  | Some args ->
      List.filter_map args ~f:(function
        | Sexp.List (Sexp.Atom name :: values) when String.is_prefix name ~prefix:":" ->
            Some
              ( String.drop_prefix name 1,
                List.concat_map values ~f:(function Sexp.Atom a -> words a | _ -> []) )
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

(* Every command an action runs, at any depth: [(with-stdout-to … (run …))],
   [(no-infer (progn (run …) …))] and the rest nest the one that matters. A shell action carries
   its command line as a string, which the same word split reads. *)
let rec commands_in sexp =
  let nested = match sexp with Sexp.List l -> List.concat_map l ~f:commands_in | _ -> [] in
  match sexp with
  | Sexp.List (Sexp.Atom ("run" | "dynamic-run") :: Sexp.Atom cmd :: _) -> cmd :: nested
  | Sexp.List (Sexp.Atom ("bash" | "system") :: args) ->
      List.concat_map args ~f:(function Sexp.Atom a -> [ a ] | _ -> []) @ nested
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

let executables_run stanza =
  let named_deps = named_deps_of stanza in
  List.map (commands_in stanza) ~f:(classify_command ~named_deps)
  |> List.dedup_and_sort ~compare:Poly.compare

type kind =
  | Test  (** a [(test)] or [(tests)] stanza, which dune runs itself *)
  | Inline_tests  (** a [(library)] with an [(inline_tests)] field, ditto *)
  | Runs_executable  (** a [(rule)] that runs an executable — where an [(executable)] stanza's
                         dependencies have to live, there being no [deps] field on one *)
  | Unreadable_command  (** a [(run …)] whose command this scan cannot place: reported, so that
                            what it runs is settled by a reader rather than by silence *)
  | Unclassified_action  (** an action head on neither {!program_actions} nor {!inert_actions} —
                             it might run a program, so it is reported too *)

(** [subdir] is the directory the stanza applies to, relative to the dune file's own: empty at the
    top level, and the path a [(subdir …)] wrapper names inside one. A wrapped stanza runs
    elsewhere, so it is that directory's config it needs — test/operations/dune configures
    test/operations/config this way. *)
type site = { kind : kind; name : string; declares_config : bool; subdir : string }

(** [in_subdir parent child] joins two relative directories, either of which may be empty. *)
let in_subdir parent child =
  if String.is_empty parent then child
  else if String.is_empty child then parent
  else parent ^ "/" ^ child

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
      let site kind name declares_config = [ { kind; name; declares_config; subdir } ] in
      let stanza_name () = String.concat ~sep:", " (names_of stanza) in
      match head stanza with
      | Some ("test" | "tests") ->
          site Test (stanza_name ()) (declares_config (field stanza "deps"))
      | Some "library" -> (
          match field stanza "inline_tests" with
          | None -> []
          | Some inline ->
              site Inline_tests (stanza_name ()) (declares_config (field_in inline "deps")))
      | Some h when List.mem action_heads h ~equal:String.equal ->
          let declares = declares_config (field stanza "deps") in
          let run = executables_run stanza in
          let exes = List.filter_map run ~f:(function Runs name -> Some name | _ -> None) in
          let unreadable =
            List.filter_map run ~f:(function Unrecognized cmd -> Some cmd | _ -> None)
          in
          (if List.is_empty exes then []
           else site Runs_executable (String.concat ~sep:", " exes) declares)
          @ List.concat_map unreadable ~f:(fun cmd -> site Unreadable_command cmd declares)
          @ List.concat_map (unclassified_action_heads stanza) ~f:(fun head ->
                site Unclassified_action head declares)
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
      (* [copy_files#] is the preprocessing spelling of the same stanza. *)
      | Some ("copy_files" | "copy_files#")
        when List.exists (atoms stanza) ~f:(fun atom ->
                 String.equal (Stdlib.Filename.basename atom) config_file) ->
          [ subdir ]
      | _ -> [])
