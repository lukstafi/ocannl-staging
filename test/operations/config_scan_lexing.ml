(** How both configuration scanners read a source file, exercised directly.

    [Config_key_scan] decides what counts as code, so every one of its mistakes is silent by
    construction: keys that vanish from a scan look exactly like keys that were never read. Four
    rounds of review on PR #340 found such bugs one at a time -- prose read as a call site, a quoted
    string inside a comment read as a nested comment, a character literal opening a string, an
    escaped literal losing its value -- which is the argument for testing on hostile input rather
    than on the library sources that happen to exist today.

    The scanner parses, with the compiler's own parser behind ppxlib, so most of these are
    regression cases rather than live hazards. They are kept because that is the point: they are
    what says so. *)

open Base
open Stdio
module Scan = Test_utils.Config_key_scan

(* Failures go through [Verdict], so that a regression exits nonzero instead of being `dune
   promote`d into the golden as the expected output (gh-ocannl-601). *)
let fail fmt = Printf.ksprintf Verdict.fail fmt

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
    (* Round 3. The first two are forms the compiler accepts that a textual scan mis-read; the third
       is one the review proposed which OCaml does not actually accept as a delimiter -- `{foo2|`
       lexes as `{`, `foo2`, `|`, so the tag rule is lowercase and underscore only. *)
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
      [ "tab\there" ] );
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
   forwarding the key -- in any spelling -- must show up here, or it hides every key routed through
   it. *)
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

(* Where a use sits, which is what decides whether an exemption covers it. A documentation comment
   is not a definition, however much its text looks like one. *)
let definition_cases =
  [
    ( "a plain definition",
      {ocaml|let get_global_arg x = x
let other y = get ~arg_name:y|ocaml},
      Some "other" );
    ( "a rec definition",
      {ocaml|let rec loop x = loop x
let other y = get ~arg_name:y|ocaml},
      Some "other" );
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
    (* `let module M = … in …` is `Pexp_letmodule`, which the path machinery does not descend, so
       the binding's name arrives without M's prefix. What the case pins is the second half: the
       binding is `(nested)`, and `top_level` is what an exemption reads, so a name introduced this
       way is exempt nowhere.

       That the first half is stable enough to pin at all is the scanner reading ppxlib's parse tree
       rather than the compiler's -- 5.5 spells this construct as a structure item inside an
       expression, and ppxlib migrates it back to the constructor matched here. *)
    ( "a local module's binding is nested, so it is exempt nowhere",
      {ocaml|let real x = x
let f () =
  let module M = struct let get_global_arg name = get ~arg_name:name end in
  M.get_global_arg|ocaml},
      Some "get_global_arg (nested)" );
  ]

(* The other spelling of a read: a field of the resolved settings record. Prose naming one is not a
   read of it. *)
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
    (* An applied path in expression position is NOT a settings read, for the parser's reason rather
       than the scanner's: OCaml reads `F(X)` as a constructor application, so this is a field
       access over an expression and its receiver is no identifier at all. Pinned because a review
       round asked the scanner to preserve a shape the language does not produce. *)
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

(* A predicate is named in two places -- the census that folds its keys out of a CALL, and the check
   that refuses it handed around as a VALUE -- and while those were two hand-written lists, a
   predicate reaching only the second lost its keys from the census in silence (gh-ocannl-750). One
   table names them now, so what is left to pin is that every entry of it is honoured at both
   positions. Generated from the table for that reason: a restated copy would check that the copy
   still says what it says, while a case per entry fails the moment a site stops consulting the
   table. The keys an entry carries are anchored by the literal case above; here they come from the
   table, since what is under test is the reach of the entries and not their contents. *)
let predicate_position_cases = Scan.settings_predicates

(* gh-ocannl-723: which sources call [Test_utils.Generated.init], the source side of the rule that
   requires OCANNL_BUILD_FILES_PREFIX of the stanza that runs them.

   The hostile input here is the repository's own: [test/support/generated.ml] names its own
   [Generated.init] in half a dozen doc comments and error messages, and [generated_provenance.ml]
   asserts on a string literal quoting one of them -- so a text scan would read the module that
   DEFINES the initializer as its heaviest caller. The spellings on the other side are the three the
   tests actually use, and the alias is the one most of them take. *)
let generated_init_cases =
  [
    ( "written out",
      {ocaml|let () = Test_utils.Generated.init ~backend_name|ocaml},
      [ "Test_utils.Generated.init" ] );
    ( "through the module alias the tests use",
      {ocaml|module Generated = Test_utils.Generated
let () = Generated.init ~backend_name|ocaml},
      [ "Generated.init" ] );
    ( "through an alias of any other name",
      {ocaml|module G = Test_utils.Generated
let () = G.init ~backend_name|ocaml},
      [ "G.init" ] );
    ( "under an open of the module",
      {ocaml|open Test_utils.Generated
let () = init ~backend_name|ocaml},
      [ "init" ] );
    (* Each of those has an expression spelling, and a pass that knew only the structure ones read
       the bare `init` below as somebody else's function -- an unrecognised caller, which looks
       exactly like a stanza with nothing to declare (Codex P2, round 1). *)
    ( "under an expression-scoped open",
      {ocaml|let () = let open Test_utils.Generated in init ~backend_name|ocaml},
      [ "init" ] );
    ( "through a locally bound module",
      {ocaml|let () = let module G = Test_utils.Generated in G.init ~backend_name|ocaml},
      [ "G.init" ] );
    ( "under an include, which puts init in scope under no name of its own",
      {ocaml|include Test_utils.Generated
let () = init ~backend_name|ocaml},
      [ "init" ] );
    (* A bare `init` is one only under that open: nothing else makes it this function, and reading
       it as one would make a caller of every module with an initializer. *)
    ( "a bare init without the open is somebody else's function",
      {ocaml|let init () = () 
let () = init ()|ocaml},
      [] );
    (* The match is by NAME, and a module called `Generated` is read as the one wherever it was
       bound. Over-reading is the safe direction of the two: a declaration too many makes dune rerun
       a stanza it need not have, while one too few is the stale run this rule exists to prevent. *)
    ( "a module called Generated is read by its name, whatever it was bound to",
      {ocaml|module Generated = Somewhere.Else
let () = Generated.init ~backend_name|ocaml},
      [ "Generated.init" ] );
    (* The one this check exists for: the module that defines the initializer names it in prose and
       in the message it raises, and calls it nowhere. *)
    (* An alias of an alias reaches the same function, and a first pass that recorded `G` without
       consulting what it had recorded read `H.init` as somebody else's (Codex P2, round 2). *)
    ( "through a chain of aliases",
      {ocaml|module G = Test_utils.Generated
module H = G
let () = H.init ~backend_name|ocaml},
      [ "H.init" ] );
    (* A signature constraint wraps the path without changing which module it names (Codex P2, round
       4). *)
    ( "through an alias carrying a signature constraint",
      {ocaml|module G : module type of Test_utils.Generated = Test_utils.Generated
let () = G.init ~backend_name|ocaml},
      [ "G.init" ] );
    ( "a doc comment naming it is not a call",
      {ocaml|(** [Test_utils.Generated.init] must be called before any compile. *)
let uninitialized () = ()|ocaml},
      [] );
    ( "a string literal quoting it is not a call",
      {ocaml|let message = "Test_utils.Generated.init ~backend_name must be called before any compile"|ocaml},
      [] );
    ( "a different function of the module is not the initializer",
      {ocaml|module Generated = Test_utils.Generated
let src = Generated.read "nz_mma"|ocaml},
      [] );
    (* Two spellings in one file are two answers, and the census only needs one -- but a scan that
       reported the first and stopped would go quiet the day a file changed which one it opens
       with. *)
    ( "both spellings in one file",
      {ocaml|module Generated = Test_utils.Generated
let () = Test_utils.Generated.init ~backend_name
let () = Generated.init ~backend_name|ocaml},
      [ "Test_utils.Generated.init"; "Generated.init" ] );
  ]

(* gh-ocannl-749: which configuration keys a source reads STRAIGHT from the environment, and whether
   it does so somewhere this scan cannot follow.

   The shape that matters is a guard: a list of key names and an iteration handing each to
   `Utils.read_env_var`. Its keys never sit next to the call, so a scan that reported only what the
   call site spells would answer "none" for the very sources the rule is about -- and answering
   "none" is what an unread guard looks like. The dynamic flag is the answer instead, and the caller
   falls back to the file's string literals, intersected with the configuration registry.

   Each case is the pair: the keys named literally, and whether some reach is dynamic. *)
let env_reader_cases =
  [
    ( "a literal argument names its key",
      {ocaml|let x = Utils.read_env_var "profile"|ocaml},
      ([ "profile" ], false) );
    ( "the receiver is matched by its last component, so an alias counts",
      {ocaml|module U = Utils
let x = U.read_env_var "profile"|ocaml},
      ([ "profile" ], false) );
    ( "and so does a bare call under an open",
      {ocaml|open Utils
let x = read_env_var "profile"|ocaml},
      ([ "profile" ], false) );
    ( "prose naming the reader is not a read",
      {ocaml|(* Utils.read_env_var "profile" decides *) let x = 1|ocaml},
      ([], false) );
    ( "a string literal quoting a call is not a call",
      {ocaml|let message = "Utils.read_env_var \"profile\" is how a guard reads"|ocaml},
      ([], false) );
    (* The guard, which is the shape the rule exists for: the key arrives as a parameter, and the
       LIST it arrives from is what the scan resolves (see the key-list family below). *)
    ( "a key taken from a resolvable list is the keys of that list",
      {ocaml|let guarded = [ "log_level"; "profile" ]
let () = List.iter (fun arg_name -> ignore (Utils.read_env_var arg_name)) guarded|ocaml},
      ([ "log_level"; "profile" ], false) );
    (* Handing the function around as a value is the same loss one layer up: whatever calls it is
       out of reach, so the source cannot be answered for either (the settings predicates above take
       the same treatment, for the same reason). *)
    ( "the reader handed on as a value is a dynamic reach",
      {ocaml|let () = List.iter (fun k -> ignore (Utils.read_env_var k)) []
let also = Utils.read_env_var|ocaml},
      ([], true) );
    ( "a partial application names no key here",
      {ocaml|let f = Utils.read_env_var ~x:1|ocaml},
      ([], true) );
    (* Both spellings in one file, since a guard commonly sits beside a direct read. *)
    ( "a literal read and a guard in one file",
      {ocaml|let () = ignore (Utils.read_env_var "profile")
let guarded = [ "log_level" ]
let () = List.iter (fun arg_name -> ignore (Utils.read_env_var arg_name)) guarded|ocaml},
      ([ "log_level"; "profile" ], false) );
    (* A different function of the same module is not the reader: `read_cmdline_or_env_var` consults
       the commandline first, which an ambient variable cannot outrank, so it is not the
       unconditional dependency this scan reports. *)
    ( "a longer name ending differently is not the reader",
      {ocaml|let x = Utils.read_cmdline_or_env_var "profile"|ocaml},
      ([], false) );
    (* The RECEIVER decides, not the basename. Over-reading is the safe direction for most of the
       scans in this file, and it is the wrong one here: what the rule built on this asks for is an
       `(env_var …)` declaration, so a function of the file's own read as `Utils.read_env_var` would
       fail a correct stanza out loud (Codex P2, round 2 of PR #484). *)
    ( "a function of the file's own that happens to share the name is not the reader",
      {ocaml|let read_env_var _ = None
let x = read_env_var "profile"|ocaml},
      ([], false) );
    ( "nor is a bare call under an open of somebody else",
      {ocaml|open Elsewhere
let x = read_env_var "profile"|ocaml},
      ([], false) );
    ( "an alias of an alias still reaches it",
      {ocaml|module U = Utils
module V = U
let x = V.read_env_var "profile"|ocaml},
      ([ "profile" ], false) );
    (* And both halves are LEXICAL, not file-wide. An `open Utils` that a local binding shadows does
       not make the shadowed call the library's -- which for a check that asks for a declaration is
       the difference between a correct stanza passing and failing (Codex P2, round 3 of PR
       #484). *)
    ( "a local binding shadows the opened name",
      {ocaml|open Utils
let read_env_var _ = None
let x = read_env_var "profile"|ocaml},
      ([], false) );
    ( "an expression-scoped open does not reach past its body",
      {ocaml|let a = let open Utils in read_env_var "log_level"
let b = read_env_var "profile"|ocaml},
      ([ "log_level" ], false) );
    ( "a qualified call is the reader whatever the file binds locally",
      {ocaml|open Utils
let read_env_var _ = None
let x = Utils.read_env_var "profile"|ocaml},
      ([ "profile" ], false) );
  ]

(* The LIST a guard iterates, which is what the scan resolves so that everything else can be
   refused. These are the shapes the guards in this repository are written in; an expression outside
   them is reported, not approximated. The pair is the keys resolved and whether anything was left
   unresolved. *)
let key_list_cases =
  [
    ( "a list bound at top level and iterated",
      {ocaml|let guarded = [ "log_level"; "profile" ]
let () = List.iter guarded ~f:(fun k -> ignore (Utils.read_env_var k))|ocaml},
      ([ "log_level"; "profile" ], false) );
    ( "the stdlib argument order too",
      {ocaml|let guarded = [ "log_level" ]
let () = List.iter (fun k -> ignore (Utils.read_env_var k)) guarded|ocaml},
      ([ "log_level" ], false) );
    ( "a list written at the iteration",
      {ocaml|let () = List.iter [ "log_level" ] ~f:(fun k -> ignore (Utils.read_env_var k))|ocaml},
      ([ "log_level" ], false) );
    (* `List.map keys ~f:fst @ [ … ]` is how `profile_precedence` builds its guard list out of the
       table it also prints. *)
    ( "a projection of a table, appended to a literal",
      {ocaml|let keys = [ ("autotune_rounds", "2"); ("tf32_matmuls", "false") ]
let guarded = List.map keys ~f:fst @ [ "no_config_file" ]
let () = List.iter guarded ~f:(fun k -> ignore (Utils.read_env_var k))|ocaml},
      ([ "autotune_rounds"; "no_config_file"; "tf32_matmuls" ], false) );
    ( "the reader handed straight to the iteration is answered as well",
      {ocaml|let guarded = [ "log_level" ]
let () = List.iter guarded ~f:Utils.read_env_var|ocaml},
      ([ "log_level" ], false) );
    (* And the refusals. A list this scan cannot follow is REPORTED, not approximated from whatever
       literals the file happens to contain -- an incidental `"profile"` elsewhere in the source
       made an unresolved reach look answered (Codex P2, round 4 of PR #484). *)
    ( "a list from another compilation unit is unresolved",
      {ocaml|let () = List.iter Shared.guarded ~f:(fun k -> ignore (Utils.read_env_var k))|ocaml},
      ([], true) );
    (* And it stays unresolved when a LOCAL binding shares its basename: resolving `Shared.guarded`
       through a local `guarded` would answer with the wrong keys and swallow the refusal that
       reports it (Codex P2, round 6 of PR #484). *)
    ( "a qualified list does not resolve through a local binding of the same name",
      {ocaml|let guarded = [ "profile" ]
let () = List.iter Shared.guarded ~f:(fun k -> ignore (Utils.read_env_var k))|ocaml},
      ([], true) );
    (* Only the combinators whose argument semantics this scan knows establish an iteration. A
       wrapper carrying a decoy list otherwise supplied the keys, blessing the reader with a list it
       is never handed (Codex P2, round 6). *)
    ( "an unknown higher-order call establishes nothing, decoy list or not",
      {ocaml|let guarded = [ "log_level" ]
let decoy = [ "profile" ]
let () = apply decoy guarded ~f:(fun k -> ignore (Utils.read_env_var k))|ocaml},
      ([], true) );
    ( "an incidental literal does not answer for an unresolved reach",
      {ocaml|let label = "profile"
let () = ignore (Utils.read_env_var Sys.argv.(1))|ocaml},
      ([], true) );
    ( "a list whose elements are not literals is unresolved",
      {ocaml|let guarded = [ some_key; other_key ]
let () = List.iter guarded ~f:(fun k -> ignore (Utils.read_env_var k))|ocaml},
      ([], true) );
    (* The parameter carries its keys over the lambda's BODY and nowhere else, so a same-named
       variable elsewhere is not silently answered by it. *)
    (* A name resolves to the binding VISIBLE at the use, not to the file's last one: taking the
       latest read `let guarded = […] … let guarded = []` as the empty list and asked for no
       declaration, while the guard really iterates the first (Codex P2, round 5 of PR #484). *)
    ( "a later rebinding does not reach backwards",
      {ocaml|let guarded = [ "log_level" ]
let () = List.iter guarded ~f:(fun k -> ignore (Utils.read_env_var k))
let guarded = []|ocaml},
      ([ "log_level" ], false) );
    ( "and a use after the rebinding sees the new one",
      {ocaml|let guarded = [ "log_level" ]
let guarded = [ "profile" ]
let () = List.iter guarded ~f:(fun k -> ignore (Utils.read_env_var k))|ocaml},
      ([ "profile" ], false) );
    (* A parameter REBOUND inside the callback is not the iterated one: answering it with the
       iterated list certifies a program that can read any key at all (Codex P2, round 7 of PR
       #484). *)
    ( "a parameter rebound inside the callback is not the iterated one",
      {ocaml|let () = List.iter [ "profile" ] ~f:(fun k -> let k = Sys.argv.(1) in ignore (Utils.read_env_var k))|ocaml},
      ([], true) );
    ( "and an inner iteration binds its own",
      {ocaml|let () =
  List.iter [ "profile" ] ~f:(fun k ->
      ignore (Utils.read_env_var k);
      List.iter [ "log_level" ] ~f:(fun k -> ignore (Utils.read_env_var k)))|ocaml},
      ([ "log_level"; "profile" ], false) );
    (* The projection is `List.map` and not any callee whose basename is `map`: a local one that
       ignores its argument had its input projected as though it were the standard function. *)
    ( "a local map does not project a table",
      {ocaml|let map _ ~f:_ = [ "virtualize_max_visits" ]
let keys = [ ("profile", "x") ]
let guarded = map keys ~f:fst
let () = List.iter guarded ~f:(fun k -> ignore (Utils.read_env_var k))|ocaml},
      ([], true) );
    (* Everything the resolver follows it matches by NAME, which is sound only while the file has
       not taken the name for something else -- the one direction a whitelist does not close by
       itself, and a silent one. A source that rebinds a trusted name gets no resolution at all. *)
    ( "a file that rebinds List resolves nothing",
      {ocaml|module List = Other
let guarded = [ "profile" ]
let () = List.iter guarded ~f:(fun k -> ignore (Utils.read_env_var k))|ocaml},
      ([], true) );
    ( "and one that rebinds fst does not project a table",
      {ocaml|let fst _ = "profile"
let keys = [ ("log_level", "0") ]
let guarded = List.map keys ~f:fst
let () = List.iter guarded ~f:(fun k -> ignore (Utils.read_env_var k))|ocaml},
      ([], true) );
    (* An `open` is not a rebinding: `Base.List.map` is `List.map`, and this repository opens Base
       everywhere. *)
    (* A nested module ending in `List` is not the standard one: a custom iterator may call the
       callback with keys the list does not hold (Codex P2, round 8 of PR #484). *)
    ( "a nested Other.List.iter is not the standard combinator",
      {ocaml|let guarded = [ "profile" ]
let () = Other.List.iter guarded ~f:(fun k -> ignore (Utils.read_env_var k))|ocaml},
      ([], true) );
    ( "but a standard root in front of it is",
      {ocaml|let guarded = [ "profile" ]
let () = Base.List.iter guarded ~f:(fun k -> ignore (Utils.read_env_var k))|ocaml},
      ([ "profile" ], false) );
    (* The map callee being right does not make its argument right: `~f:Other.fst` may return the
       other column (Codex P2, round 9 of PR #484). *)
    ( "a qualified projector that is not the standard fst does not project",
      {ocaml|let keys = [ ("profile", "virtualize_max_visits") ]
let guarded = List.map keys ~f:Other.fst
let () = List.iter guarded ~f:(fun k -> ignore (Utils.read_env_var k))|ocaml},
      ([], true) );
    (* Rebinding an approved ROOT leaves the whitelisted path intact and changes what it means. *)
    ( "a rebound standard root is a rebound trusted name",
      {ocaml|module Base = Shared
let guarded = [ "profile" ]
let () = Base.List.iter guarded ~f:(fun k -> ignore (Utils.read_env_var k))|ocaml},
      ([], true) );
    ( "a qualified concatenation operator is not the standard one",
      {ocaml|let left = [ "profile" ]
let right = [ "log_level" ]
let guarded = Shared.( @ ) left right
let () = List.iter guarded ~f:(fun k -> ignore (Utils.read_env_var k))|ocaml},
      ([], true) );
    ( "an open of a library providing List is not a rebinding",
      {ocaml|open Base
let guarded = [ "profile" ]
let () = List.iter guarded ~f:(fun k -> ignore (Utils.read_env_var k))|ocaml},
      ([ "profile" ], false) );
    (* A rebinding this scan cannot resolve is a TOMBSTONE, not an absence: reaching past it to the
       earlier list answers with keys that no longer hold (Codex P2, round 11 of PR #484). *)
    ( "an unresolvable rebinding is not reached past",
      {ocaml|let guarded = [ "profile" ]
let guarded = [ Sys.argv.(1) ]
let () = List.iter guarded ~f:(fun k -> ignore (Utils.read_env_var k))|ocaml},
      ([], true) );
    ( "and a use BEFORE it still sees the resolvable one",
      {ocaml|let guarded = [ "profile" ]
let () = List.iter guarded ~f:(fun k -> ignore (Utils.read_env_var k))
let guarded = [ Sys.argv.(1) ]|ocaml},
      ([ "profile" ], false) );
    (* A tombstone is recorded for ANY later binding of a name that once denoted a key list --
       inferring "list-shaped" from the AST form let this one past, being neither a constructor nor
       an application (Codex P2, round 12 of PR #484). *)
    ( "a conditional rebinding is tombstoned too",
      {ocaml|let guarded = [ "profile" ]
let guarded = if enabled then [ "virtualize_max_visits" ] else []
let () = List.iter guarded ~f:(fun k -> ignore (Utils.read_env_var k))|ocaml},
      ([], true) );
    ( "and a name that never denoted a list is not tombstoned by an unrelated binding",
      {ocaml|let guarded = [ "profile" ]
let other = 3
let () = List.iter guarded ~f:(fun k -> ignore (Utils.read_env_var k))|ocaml},
      ([ "profile" ], false) );
    ( "the binding does not escape the lambda it was established at",
      {ocaml|let guarded = [ "log_level" ]
let () = List.iter guarded ~f:(fun k -> ignore (Utils.read_env_var k))
let elsewhere k = Utils.read_env_var k|ocaml},
      ([ "log_level" ], true) );
  ]

let could_read_cases =
  [
    ("names the reader", {ocaml|let x = Utils.read_env_var "profile"|ocaml}, true);
    ( "does not name it at all",
      {ocaml|let x = Utils.get_global_arg ~arg_name:"profile"|ocaml},
      false );
  ]

(* And the textual filter the census narrows with, which is only safe while naming the module is a
   NECESSARY condition for calling it: a file the filter drops is never parsed, so a call it hid
   would be invisible rather than reported. *)
let could_call_cases =
  [
    ("names the module", {ocaml|let () = Test_utils.Generated.init ~backend_name|ocaml}, true);
    ("names it only to alias it", {ocaml|module G = Test_utils.Generated|ocaml}, true);
    ("does not name it at all", {ocaml|let () = print_string "hello"|ocaml}, false);
  ]

let () =
  List.iter cases ~f:(fun (name, source, expected) ->
      let found =
        try List.sort ~compare:String.compare (Scan.keys_in_source source)
        with _ ->
          fail "%s -- the snippet does not parse" name;
          []
      in
      let expected = List.sort ~compare:String.compare expected in
      if List.equal String.equal found expected then printf "ok: %s\n" name
      else
        fail "%s -- expected [%s], found [%s]" name
          (String.concat ~sep:"; " expected)
          (String.concat ~sep:"; " found));
  List.iter non_literal_cases ~f:(fun (name, source, expected) ->
      let found = List.count (Scan.label_uses source) ~f:(fun u -> Option.is_none u.Scan.key) in
      if found = expected then printf "ok: non-literal uses -- %s\n" name
      else fail "non-literal uses -- %s: expected %d, found %d" name expected found);
  let enclosing_definition source =
    let definitions = Scan.definitions source in
    let render (d : Scan.definition) =
      Option.value d.Scan.name ~default:"<anonymous function>"
      ^ if d.Scan.top_level then "" else " (nested)"
    in
    List.filter_map (Scan.label_uses source) ~f:(fun u ->
        Option.map (Scan.definition_at definitions u.Scan.offset) ~f:render)
    |> List.hd
  in
  let show = Option.value ~default:"<none>" in
  List.iter definition_cases ~f:(fun (name, source, expected) ->
      let found = enclosing_definition source in
      if Option.equal String.equal found expected then
        printf "ok: enclosing definition -- %s\n" name
      else
        fail "enclosing definition -- %s: expected %s, found %s" name (show expected) (show found));
  List.iter settings_cases ~f:(fun (name, source, expected) ->
      let found = List.sort ~compare:String.compare (Scan.settings_keys_in_source source) in
      let expected = List.sort ~compare:String.compare expected in
      if List.equal String.equal found expected then printf "ok: settings read -- %s\n" name
      else
        fail "settings read -- %s: expected [%s], found [%s]" name
          (String.concat ~sep:"; " expected)
          (String.concat ~sep:"; " found));
  List.iter predicate_position_cases ~f:(fun (predicate, implied) ->
      let call = Printf.sprintf "let x = Utils.%s ()" predicate in
      let handed_on = Printf.sprintf "let f = Utils.%s" predicate in
      let found = List.sort ~compare:String.compare (Scan.settings_keys_in_source call) in
      let expected = List.sort ~compare:String.compare implied in
      if List.equal String.equal found expected then
        printf "ok: predicate call contributes its keys -- %s\n" predicate
      else
        fail "predicate call -- %s: expected [%s], found [%s]" predicate
          (String.concat ~sep:"; " expected)
          (String.concat ~sep:"; " found);
      (* The call is the position the census reads, so it is not a finding; the bare value is the
         position it cannot follow, so it is. Both directions, because a predicate the second site
         does not know is silently readable as a value, and one the first site blesses too eagerly
         would let a genuinely escaping read through. *)
      let at_call = List.length (Scan.unqualified_settings_reads call) in
      if at_call = 0 then printf "ok: predicate call is not an escaping read -- %s\n" predicate
      else fail "predicate call -- %s: expected no escaping read, found %d" predicate at_call;
      let handed_on_count = List.length (Scan.unqualified_settings_reads handed_on) in
      if handed_on_count = 1 then
        printf "ok: predicate handed on as a value is an escaping read -- %s\n" predicate
      else
        fail "predicate handed on -- %s: expected 1 escaping read, found %d" predicate
          handed_on_count);
  List.iter generated_init_cases ~f:(fun (name, source, expected) ->
      let found =
        try Scan.generated_init_calls_in_source source
        with _ ->
          fail "Generated.init -- %s: the snippet does not parse" name;
          []
      in
      if List.equal String.equal found expected then printf "ok: Generated.init -- %s\n" name
      else
        fail "Generated.init -- %s: expected [%s], found [%s]" name
          (String.concat ~sep:"; " expected)
          (String.concat ~sep:"; " found));
  List.iter could_call_cases ~f:(fun (name, source, expected) ->
      let found = Scan.could_call_generated_init source in
      if Bool.equal found expected then printf "ok: could call -- %s\n" name
      else fail "could call -- %s: expected %b, found %b" name expected found);
  List.iter env_reader_cases ~f:(fun (name, source, (expected_keys, expected_dynamic)) ->
      let found =
        try Some (Scan.env_reader_reads_in_source source)
        with _ ->
          fail "environment read -- %s: the snippet does not parse" name;
          None
      in
      Option.iter found ~f:(fun found ->
          let keys = found.Scan.reader_keys in
          let dynamic = not (List.is_empty found.Scan.reader_unresolved) in
          let expected_keys = List.sort ~compare:String.compare expected_keys in
          if List.equal String.equal keys expected_keys && Bool.equal dynamic expected_dynamic then
            printf "ok: environment read -- %s\n" name
          else
            fail
              "environment read -- %s: expected dynamic %b with keys [%s], found dynamic %b with \
               keys [%s]"
              name expected_dynamic
              (String.concat ~sep:"; " expected_keys)
              dynamic (String.concat ~sep:"; " keys)));
  List.iter key_list_cases ~f:(fun (name, source, (expected_keys, expected_unresolved)) ->
      let found = Scan.env_reader_reads_in_source source in
      let unresolved = not (List.is_empty found.Scan.reader_unresolved) in
      let expected_keys = List.sort ~compare:String.compare expected_keys in
      if
        List.equal String.equal found.Scan.reader_keys expected_keys
        && Bool.equal unresolved expected_unresolved
      then printf "ok: key list -- %s\n" name
      else
        fail
          "key list -- %s: expected unresolved %b with keys [%s], found unresolved %b with keys \
           [%s]"
          name expected_unresolved
          (String.concat ~sep:"; " expected_keys)
          unresolved
          (String.concat ~sep:"; " found.Scan.reader_keys));
  List.iter could_read_cases ~f:(fun (name, source, expected) ->
      let found = Scan.could_read_env_var source in
      if Bool.equal found expected then printf "ok: could read the environment -- %s\n" name
      else fail "could read the environment -- %s: expected %b, found %b" name expected found);
  List.iter [ "digest_completeness.ml"; "test_config_consistency.ml" ] ~f:(fun source ->
      Test_utils.Refusal_control_manifest.print source)
