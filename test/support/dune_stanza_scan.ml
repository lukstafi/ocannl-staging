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

(** Whether a [(deps …)] field mentions the shared configuration file. Path-insensitive: a
    directory reaching for a config elsewhere ([../config/ocannl_config]) still declares it. *)
let declares_config args =
  match args with
  | None -> false
  | Some args ->
      List.exists args ~f:(fun arg ->
          List.exists (atoms arg) ~f:(fun atom ->
              String.equal (Stdlib.Filename.basename atom) config_file))

(** The names a [(name …)] or [(names …)] field gives, in order. *)
let names_of stanza =
  match field stanza "name" with
  | Some [ Sexp.Atom name ] -> [ name ]
  | _ -> (
      match field stanza "names" with
      | Some args -> List.filter_map args ~f:(function Sexp.Atom n -> Some n | _ -> None)
      | _ -> [])

(* An atom carries a path inside dune's own punctuation: [%{dep:mlp_names.exe}] and [./%{pp}] are
   one atom each. Split on everything a path cannot contain and keep what ends in [.exe]. *)
let path_char c =
  Char.is_alphanum c || List.mem [ '_'; '.'; '-'; '/'; '\\' ] c ~equal:Char.equal

let executables_mentioned stanza =
  atoms stanza
  |> List.concat_map ~f:(fun atom ->
         String.map atom ~f:(fun c -> if path_char c then c else ' ')
         |> String.split ~on:' '
         |> List.filter ~f:(String.is_suffix ~suffix:".exe")
         |> List.map ~f:Stdlib.Filename.basename)
  |> List.dedup_and_sort ~compare:String.compare

type kind =
  | Test  (** a [(test)] or [(tests)] stanza, which dune runs itself *)
  | Inline_tests  (** a [(library)] with an [(inline_tests)] field, ditto *)
  | Runs_executable  (** a [(rule)] that runs an executable — where an [(executable)] stanza's
                         dependencies have to live, there being no [deps] field on one *)

type site = { kind : kind; name : string; declares_config : bool }

let kind_name = function
  | Test -> "test"
  | Inline_tests -> "inline tests"
  | Runs_executable -> "rule running"

(** Every place in [content] that runs a test executable.

    An [(executable)] stanza is not one: it declares something to build, and dune runs it only
    where a rule says so — which is why a diagnostic executable such as [bench_circles_step] or a
    tutorial such as [gpt2_generate] needs no exemption from the check built on this. It is
    structurally not a site, rather than a name on a list someone has to keep true.

    A rule counts as running an executable when it mentions one at all. That covers both spellings
    the repository uses — [%{dep:foo.exe}] inline in the action, and a named dependency
    [(:pp pp.exe)] the action reaches through [%{pp}] — without the scan having to model dune's
    variable expansion. *)
let sites content =
  List.filter_map (stanzas content) ~f:(fun stanza ->
      let named kind =
        Some
          {
            kind;
            name = String.concat ~sep:", " (names_of stanza);
            declares_config = declares_config (field stanza "deps");
          }
      in
      match head stanza with
      | Some ("test" | "tests") -> named Test
      | Some "library" -> (
          match field stanza "inline_tests" with
          | None -> None
          | Some inline ->
              Some
                {
                  kind = Inline_tests;
                  name = String.concat ~sep:", " (names_of stanza);
                  declares_config = declares_config (field_in inline "deps");
                })
      | Some "rule" -> (
          match executables_mentioned stanza with
          | [] -> None
          | exes ->
              Some
                {
                  kind = Runs_executable;
                  name = String.concat ~sep:", " exes;
                  declares_config = declares_config (field stanza "deps");
                })
      | _ -> None)

(** Whether the directory materializes the shared configuration for itself, i.e. has a
    [(copy_files …ocannl_config)] stanza. The other way to have one is to check a file in next to
    the dune file, which this cannot see and the caller supplies. *)
let copies_config content =
  List.exists (stanzas content) ~f:(fun stanza ->
      match head stanza with
      | Some "copy_files" ->
          List.exists (atoms stanza) ~f:(fun atom ->
              String.equal (Stdlib.Filename.basename atom) config_file)
      | _ -> false)
