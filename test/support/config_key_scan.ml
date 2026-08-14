(** Scanning OCaml sources for the configuration keys they read.

    Shared by the two consistency tests that hold the configuration honest:
    [test_config_consistency] (every key documented and registered) and [digest_completeness] (every
    key classified against the cache identity, gh-ocannl-572). Both need the same fact — which keys
    a source file reads — so they read it the same way. *)

open Base

(** {1 Reading these sources as OCaml, not as text}

    Everything below rests on one pass of the compiler's own lexer. Three rounds of review on
    PR #340 found the same class of defect in a hand-rolled scanner -- prose read as a call site, a
    quoted string inside a comment read as a nested comment, a character literal inside a comment
    opening a string -- and each fix left the next divergence from OCaml's real lexing rules
    waiting. Every one of them is silent by construction: a desynchronised scan blanks live code,
    and keys that vanish look exactly like keys that were never read.

    So the approximation is gone. [Lexer.token] decides what is a comment, a string, a character
    literal and a quoted string, because it is the same code that decides it for the compiler. *)

let fst3 (a, _, _) = a

(** The tokens of [content], each with the source range it covers. Raises if the text does not lex:
    a scan that cannot read its input must say so rather than report an empty census. *)
let tokens content =
  let lexbuf = Lexing.from_string content in
  Lexer.init ();
  let rec loop acc =
    match Lexer.token lexbuf with
    | Parser.EOF -> List.rev acc
    | tok -> loop ((tok, Lexing.lexeme_start lexbuf, Lexing.lexeme_end lexbuf) :: acc)
  in
  loop []

(** Every top-level definition in [content], as (offset where its [let] or [and] begins, name),
    in source order.

    Read from the token stream rather than the text, so nothing that merely looks like a definition
    can pass for one: a documentation comment quoting a column-zero [let get_global_arg] is not a
    token at all, and cannot lend its name to the code that follows it (Codex P2, round 4). The
    same argument covers whatever else a comment might contain, without enumerating the cases. *)
let definitions content =
  let at_line_start start = start = 0 || Char.equal content.[start - 1] '\n' in
  let rec scan toks acc =
    match toks with
    | [] -> List.rev acc
    | ((Parser.LET | Parser.AND), start, _) :: rest when at_line_start start -> name rest start acc
    | _ :: rest -> scan rest acc
  and name toks start acc =
    match toks with
    | (Parser.REC, _, _) :: rest -> name rest start acc
    | (Parser.LIDENT id, _, _) :: rest -> scan rest ((start, id) :: acc)
    | _ -> scan toks acc
  in
  scan (tokens content) []

(** Every place the [arg_name] label names a configuration key, and what it names it with. [key] is
    [Some k] when the argument is a string literal — the convention both consistency tests rely on
    to find a read — and [None] when it is anything else: a variable, a punned parameter, an
    expression. [offset] is where the label starts, for a caller that wants to report the line.

    All the spellings OCaml accepts are covered, because the lexer supplies them rather than a
    pattern guessing at them: [~arg_name:"key"] and [?arg_name:"key"] (a [LABEL] / [OPTLABEL] token
    followed by a [STRING]), the punned [~arg_name] and [?arg_name] (never a literal), and the
    optional-parameter default [?(arg_name = "key")]. *)
type label_use = { key : string option; offset : int }

let label = "arg_name"

let label_uses content =
  let toks = Array.of_list (tokens content) in
  ignore content;
  let literal_at i =
    if i < Array.length toks then
      match toks.(i) with
      | Parser.STRING (value, _, _), _, _ -> Some value
      | _ -> None
    else None
  in
  let is_named i =
    i < Array.length toks
    && match toks.(i) with Parser.LIDENT id, _, _ -> String.equal id label | _ -> false
  in
  let rec walk i acc =
    if i >= Array.length toks then List.rev acc
    else
      match toks.(i) with
      (* [~arg_name:…] and [?arg_name:…]: the label and its colon are one token, so whatever
         follows IS the argument. *)
      | (Parser.LABEL id | Parser.OPTLABEL id), offset, _ when String.equal id label ->
          walk (i + 1) ({ key = literal_at (i + 1); offset } :: acc)
      (* [?(arg_name = "key")], the optional-parameter default. *)
      | Parser.QUESTION, offset, _
        when i + 3 < Array.length toks
             && (match toks.(i + 1) with Parser.LPAREN, _, _ -> true | _ -> false)
             && is_named (i + 2) ->
          let key =
            match toks.(i + 3) with
            | Parser.EQUAL, _, _ -> literal_at (i + 4)
            | _ -> None
          in
          walk (i + 4) ({ key; offset } :: acc)
      (* Punned: [~arg_name] / [?arg_name], in an application or a parameter list. Never a
         literal — this is the shape that hides keys. *)
      | (Parser.TILDE | Parser.QUESTION), offset, _ when is_named (i + 1) ->
          walk (i + 2) ({ key = None; offset } :: acc)
      | _ -> walk (i + 1) acc
  in
  walk 0 []

(** The [arg_name] literals of the [get_global_arg] / [get_global_flag] calls in [content].

    A key that reaches the lookup any other way — through a helper taking the name as a
    parameter — is invisible to this scan, and hence to both tests built on it. That is why
    [test_config_consistency] separately fails any non-literal use of the label outside the handful
    of named functions that implement the lookup. *)
let keys_in_source content =
  List.filter_map (label_uses content) ~f:(fun u -> u.key)
  |> List.filter ~f:(fun s -> not (String.is_empty s))

(** [keys_in_source] over each file, as a set. Call sites only — this is what
    [test_config_consistency] means by "every key a source file asks for is documented and
    registered". *)
let keys_in_files files =
  List.concat_map files ~f:(fun fname -> keys_in_source (Stdio.In_channel.read_all fname))
  |> Set.of_list (module String)

(** The other spelling of a configuration read: a field of the startup-resolved [Utils.settings]
    record, whose field names {e are} the config keys, and the two predicates over it that fold in
    the [log_level > 1] threshold ([debug_log_from_routines], [with_runtime_debug]). A census built
    from [arg_name] literals alone would miss every one of these — [large_models] is read as
    [Utils.settings.large_models] in the codegen, so a future misclassification of it could pass
    unchallenged (Codex P2 on PR #337). *)
let settings_keys_in_source content =
  let toks = Array.of_list (tokens content) in
  let tok i = if i < Array.length toks then Some (fst3 toks.(i)) else None in
  let is_lident i name =
    match tok i with Some (Parser.LIDENT id) -> String.equal id name | _ -> false
  in
  let rec walk i acc =
    if i >= Array.length toks then acc
    else
      match tok i with
      (* Qualified: [Low_level.virtualize_settings] and friends are records of the same shape whose
         field names are NOT config keys ([max_visits] against [virtualize_max_visits]), so an
         unqualified match would attribute reads to keys that do not exist. *)
      | Some (Parser.UIDENT "Utils")
        when (match tok (i + 1) with Some Parser.DOT -> true | _ -> false)
             && is_lident (i + 2) "settings"
             && (match tok (i + 3) with Some Parser.DOT -> true | _ -> false) -> (
          match tok (i + 4) with
          | Some (Parser.LIDENT field) -> walk (i + 5) (field :: acc)
          | _ -> walk (i + 1) acc)
      (* The two predicates that fold the [log_level > 1] threshold into a flag. A call, not a
         mention: [LPAREN RPAREN] must follow. *)
      | Some (Parser.LIDENT name)
        when (match tok (i + 1) with Some Parser.LPAREN -> true | _ -> false)
             && (match tok (i + 2) with Some Parser.RPAREN -> true | _ -> false) -> (
          match name with
          | "debug_log_from_routines" -> walk (i + 3) ("log_level" :: name :: acc)
          | "with_runtime_debug" ->
              walk (i + 3) ("log_level" :: "output_debug_files_in_build_directory" :: acc)
          | _ -> walk (i + 1) acc)
      | _ -> walk (i + 1) acc
  in
  walk 0 []

(** Every configuration read of a file — [arg_name] call sites and {!settings_keys_in_source} —
    keyed by file basename, for tests that care {e where} a key is read. Field names that are not
    config keys come along; callers intersect with the registry. *)
let keys_by_file files =
  List.map files ~f:(fun fname ->
      let content = Stdio.In_channel.read_all fname in
      ( Stdlib.Filename.basename fname,
        Set.of_list (module String) (keys_in_source content @ settings_keys_in_source content) ))
