(** How both configuration scanners read a source file, exercised directly.

    [Config_key_scan] decides what counts as code, so every one of its mistakes is silent by
    construction: keys that vanish from a scan look exactly like keys that were never read. Four
    rounds of review on PR #340 found such bugs one at a time -- prose read as a call site, a
    quoted string inside a comment read as a nested comment, a character literal opening a string,
    an escaped literal losing its value -- which is the argument for testing on hostile input
    rather than on the library sources that happen to exist today.

    The scanner now runs the compiler's own lexer, so most of these are regression cases rather
    than live hazards. They are kept because that is the point: they are what says so. *)

open Base
open Stdio
module Scan = Test_utils.Config_key_scan

(* A key that MUST be found is spelled with the marker; a key that must NOT be found sits in prose
   or in a blanked body. Each case pairs a snippet with the keys the scan should report. *)
let cases =
  [
    ("plain call site", {ocaml|let x = get ~arg_name:"alpha" ~default:""|ocaml}, [ "alpha" ]);
    ( "optional-parameter default",
      {ocaml|let f ?(arg_name = "beta") () = get ~arg_name ~default:""|ocaml},
      [ "beta" ] );
    ( "prose in a comment is not a call site",
      {ocaml|(* pass ~arg_name:"phantom" here *) let x = get ~arg_name:"gamma" ~default:""|ocaml},
      [ "gamma" ] );
    ( "nested comments close in the right order",
      {ocaml|(* outer (* inner ~arg_name:"phantom" *) still comment *) let x = get ~arg_name:"delta" ~default:""|ocaml},
      [ "delta" ] );
    ( "a string inside a comment does not swallow the file",
      {ocaml|(* the "(*" spelling *) let x = get ~arg_name:"epsilon" ~default:""|ocaml},
      [ "epsilon" ] );
    ( "a quoted string inside a comment is not a nested comment",
      {ocaml|(* example: {| (* text |} *) let x = get ~arg_name:"zeta" ~default:""|ocaml},
      [ "zeta" ] );
    ( "a tagged quoted string inside a comment",
      {ocaml|(* example: {sql| (* |sql} *) let x = get ~arg_name:"eta" ~default:""|ocaml},
      [ "eta" ] );
    ( "a double quote as a character literal does not open a string",
      {ocaml|let q = '"'
let x = get ~arg_name:"theta" ~default:""|ocaml},
      [ "theta" ] );
    ( "an escaped quote inside a string does not end it",
      {ocaml|let s = "a \" (* not a comment *)"
let x = get ~arg_name:"iota" ~default:""|ocaml},
      [ "iota" ] );
    ( "a backslash-backslash char literal does not eat the closing quote",
      {ocaml|let c = '\\'
let x = get ~arg_name:"kappa" ~default:""|ocaml},
      [ "kappa" ] );
    ( "a type variable is not a character literal",
      {ocaml|let f (x : 'a list) = x let y = get ~arg_name:"lambda" ~default:""|ocaml},
      [ "lambda" ] );
    ( "a record literal is not a quoted string",
      {ocaml|let r = { field } let x = get ~arg_name:"mu" ~default:""|ocaml},
      [ "mu" ] );
    (* Round 3. The first two are forms the compiler accepts that a textual scan mis-read; the
       third is one the review proposed which OCaml does not actually accept as a delimiter --
       `{foo2|` lexes as `{`, `foo2`, `|`, so the tag rule is lowercase and underscore only. *)
    ( "a character literal inside a comment does not open a string",
      {ocaml|(* the quote character is '"' *) let x = get ~arg_name:"nu" ~default:""|ocaml},
      [ "nu" ] );
    ( "an extension quoted string inside a comment",
      {ocaml|(* example: {%ext| (* text |} *) let x = get ~arg_name:"xi" ~default:""|ocaml},
      [ "xi" ] );
    (* Optional-argument application, in every spelling: with a literal it IS a read, punned or
       applied to an expression it is not. *)
    ( "optional application with a literal is a read",
      {ocaml|let x = get_style ?arg_name:"pi" ()|ocaml},
      [ "pi" ] );
    ( "optional application of a variable reads no literal key",
      {ocaml|let g name = get_style ?arg_name:(Some name) () let x = get ~arg_name:"rho" ~default:""|ocaml},
      [ "rho" ] );
    ( "a punned optional argument reads no literal key",
      {ocaml|let g ?arg_name () = get_style ?arg_name () let x = get ~arg_name:"sigma" ~default:""|ocaml},
      [ "sigma" ] );
    (* A call site spelled inside a STRING is not a call site either -- the token stream never
       looks inside a literal, where the old textual scan happily found a phantom key. *)
    (* Round 4. The lexer's decoded value, not the source slice: a continuation line makes the two
       differ, and the slice would be dropped as malformed while the check still saw a literal --
       an unregistered key slipping past both scanners. *)
    ( "an escaped-newline continuation decodes to the real key",
      "let x = get ~arg_name:\"undocu\\\n   mented\" ~default:\"\"",
      [ "undocumented" ] );
    ( "an escape sequence decodes to the real key",
      {ocaml|let x = get ~arg_name:"tab	here" ~default:""|ocaml},
      [ "tab	here" ] );
    ( "a typed optional default is still a literal default",
      {ocaml|let get_style ?(arg_name : string = "typed_default") () = arg_name|ocaml},
      [ "typed_default" ] );
    ( "an empty literal names no key, so no key is read",
      {ocaml|let x = get ~arg_name:"" ~default:""|ocaml},
      [] );
    ( "a call site quoted inside a string literal is not a read",
      {ocaml|let doc = "pass ~arg_name:\"phantom\" here" let x = get ~arg_name:"tau" ~default:""|ocaml},
      [ "tau" ] );
  ]

(* The other half of what the check needs from the lexer: which uses are NOT literals. A helper
   forwarding the key -- in any spelling -- must show up here, or it hides every key routed
   through it. *)
let non_literal_cases =
  [
    ("labelled variable", {ocaml|let f name = get ~arg_name:name ~default:""|ocaml}, 1);
    (* An empty literal is a literal: it is NOT reported here, which is why the consistency test
       reports it separately rather than trusting the census to notice its absence. *)
    ("empty literal is still a literal", {ocaml|let x = get ~arg_name:"" ~default:""|ocaml}, 0);
    ("punned label", {ocaml|let f ~arg_name = get ~arg_name ~default:""|ocaml}, 2);
    ("optional application of an expression", {ocaml|let f name = g ?arg_name:(Some name)|ocaml}, 1);
    ("punned optional", {ocaml|let f ?arg_name () = g ?arg_name ()|ocaml}, 2);
    ("literals are not reported", {ocaml|let x = get ~arg_name:"k" ~default:""|ocaml}, 0);
    ("prose is not reported", {ocaml|(* ~arg_name and ?arg_name *) let x = 1|ocaml}, 0);
  ]

(* Where a use sits, which is what decides whether an exemption covers it. A documentation
   comment is not a definition, however much its text looks like one. *)
let definition_cases =
  [
    ("a plain definition", {ocaml|let get_global_arg x = x
let other y = get ~arg_name:y|ocaml}, Some "other");
    ("a rec definition", {ocaml|let rec loop x = loop x
let other y = get ~arg_name:y|ocaml}, Some "other");
    ( "a doc comment quoting a column-zero binding is not a definition",
      {ocaml|let real x = x
(** an example:
let get_global_arg
    is only prose *)
let other y = get ~arg_name:y|ocaml},
      Some "other" );
    ( "a nameless binding lends no name, so nothing can be exempt in it",
      {ocaml|let get_global_arg x = x
let () = ignore (get ~arg_name:name)|ocaml},
      None );
    ( "a named local helper is its own definition, not its host",
      {ocaml|let get_global_arg x =
  let hidden name = get ~arg_name:name in
  hidden x|ocaml},
      Some "hidden (nested)" );
    ( "an aliased local helper is still a helper",
      {ocaml|let get_global_arg x =
  let (hidden as alias) = fun name -> get ~arg_name:name in
  ignore alias;
  hidden x|ocaml},
      Some "hidden (nested)" );
    (* No binding at all, so no name, so exempt nowhere. Keying on the lambda is what makes this
       case exist; a rule about binding forms could not have reached it. *)
    ( "an inline anonymous function can be exempt nowhere",
      {ocaml|let get_global_arg x = List.map (fun name -> get ~arg_name:name) x|ocaml},
      Some "<anonymous function> (nested)" );
    ( "a nameless local binding is transparent, so its host keeps the use",
      {ocaml|let get_global_arg x =
  let a, b = get ~arg_name:x in
  (a, b)|ocaml},
      Some "get_global_arg" );
    ( "a binding inside a module is qualified, so it cannot borrow an exemption",
      {ocaml|let get_global_arg x = x
module M = struct
  let inner () = get ~arg_name:name
end|ocaml},
      Some "M.inner (nested)" );
    ( "a nested binding reusing an exempt name stays qualified",
      {ocaml|let real x = x
module Sneaky = struct
  let get_global_arg name = get ~arg_name:name
end|ocaml},
      Some "Sneaky.get_global_arg (nested)" );
    ( "a local module is qualified too",
      {ocaml|let real x = x
let f () =
  let module M = struct let get_global_arg name = get ~arg_name:name end in
  M.get_global_arg|ocaml},
      Some "M.get_global_arg (nested)" );
    ( "a binding inside open struct is not bare",
      {ocaml|let real x = x
open struct
  let get_global_arg name = get ~arg_name:name
end|ocaml},
      Some "_.get_global_arg (nested)" );
    ( "a binding inside include struct is not bare",
      {ocaml|let real x = x
include struct
  let get_global_arg name = get ~arg_name:name
end|ocaml},
      Some "_.get_global_arg (nested)" );
    ( "a binding inside a structure-level extension is not bare",
      {ocaml|let real x = x
[%%ext
let get_global_arg name = get ~arg_name:name]|ocaml},
      Some "_.get_global_arg (nested)" );
    (* The name here is NOT prefixed -- a packed module is not one of the forms the path machinery
       knows -- and that is the point: the exemption reads `top_level`, which is false because this
       binding is not one of the root structure's own. A form the reader-facing path misses still
       cannot widen an exemption. *)
    ( "a binding inside a first-class module expression is not bare",
      {ocaml|let real x = x
let m = (module struct
  let get_global_arg name = get ~arg_name:name
end : S)|ocaml},
      Some "get_global_arg (nested)" );
    ( "siblings of a let-and group are told apart",
      {ocaml|let get_global_arg x = x
and other name = get ~arg_name:name|ocaml},
      Some "other" );
    ( "an ordinary comment quoting a binding is not a definition",
      {ocaml|let real x = x
(* let get_global_arg *)
let other y = get ~arg_name:y|ocaml},
      Some "other" );
  ]

(* The other spelling of a read: a field of the resolved settings record. Prose naming one is not
   a read of it. *)
let settings_cases =
  [
    ("a field read", {ocaml|let x = Utils.settings.large_models|ocaml}, [ "large_models" ]);
    ( "a field named in a doc comment is not a read",
      {ocaml|(** see Utils.settings.large_models *) let x = 1|ocaml},
      [] );
    ( "a module-qualified field label names the same read",
      {ocaml|let x = Utils.settings.Utils.large_models|ocaml},
      [ "large_models" ] );
    ( "a functor-application elsewhere does not crash the scan",
      {ocaml|let e = Set.Make(String).empty
let x = Utils.settings.large_models|ocaml},
      [ "large_models" ] );
    (* An applied path in expression position is NOT a settings read, for the parser's reason
       rather than the scanner's: OCaml reads `F(X)` as a constructor application, so this is a
       field access over an expression and its receiver is no identifier at all. Pinned because a
       review round asked the scanner to preserve a shape the language does not produce. *)
    ( "an applied path in expression position is not an identifier, so not a read",
      {ocaml|let x = F(X).Utils.settings.large_models|ocaml},
      [] );
    ( "an unqualified record of the same shape is not a read",
      {ocaml|let x = Low_level.virtualize_settings.max_visits|ocaml},
      [] );
    ( "a predicate call folds in its threshold",
      {ocaml|let x = Utils.debug_log_from_routines ()|ocaml},
      [ "debug_log_from_routines"; "log_level" ] );
    ( "a predicate merely mentioned in prose is not a call",
      {ocaml|(* debug_log_from_routines () decides *) let x = 1|ocaml},
      [] );
  ]

let () =
  let ok = ref true in
  List.iter cases ~f:(fun (name, source, expected) ->
      let found =
        try List.sort ~compare:String.compare (Scan.keys_in_source source)
        with _ ->
          ok := false;
          printf "FAIL: %s -- the snippet does not parse\n" name;
          []
      in
      let expected = List.sort ~compare:String.compare expected in
      if List.equal String.equal found expected then printf "ok: %s\n" name
      else (
        ok := false;
        printf "FAIL: %s -- expected [%s], found [%s]\n" name
          (String.concat ~sep:"; " expected)
          (String.concat ~sep:"; " found)));
  List.iter non_literal_cases ~f:(fun (name, source, expected) ->
      let found =
        List.count (Scan.label_uses source) ~f:(fun u -> Option.is_none u.Scan.key)
      in
      if found = expected then printf "ok: non-literal uses -- %s\n" name
      else (
        ok := false;
        printf "FAIL: non-literal uses -- %s: expected %d, found %d\n" name expected found));
  List.iter definition_cases ~f:(fun (name, source, expected) ->
      let definitions = Scan.definitions source in
      let render (d : Scan.definition) =
        Option.value d.Scan.name ~default:"<anonymous function>"
        ^ if d.Scan.top_level then "" else " (nested)"
      in
      let found =
        List.filter_map (Scan.label_uses source) ~f:(fun u ->
            Option.map (Scan.definition_at definitions u.Scan.offset) ~f:render)
        |> List.hd
      in
      if Option.equal String.equal found expected then printf "ok: enclosing definition -- %s\n" name
      else (
        ok := false;
        printf "FAIL: enclosing definition -- %s: expected %s, found %s\n" name
          (Option.value expected ~default:"<none>")
          (Option.value found ~default:"<none>")));
  List.iter settings_cases ~f:(fun (name, source, expected) ->
      let found = List.sort ~compare:String.compare (Scan.settings_keys_in_source source) in
      let expected = List.sort ~compare:String.compare expected in
      if List.equal String.equal found expected then printf "ok: settings read -- %s\n" name
      else (
        ok := false;
        printf "FAIL: settings read -- %s: expected [%s], found [%s]\n" name
          (String.concat ~sep:"; " expected)
          (String.concat ~sep:"; " found)));
  if not !ok then Stdlib.exit 1
