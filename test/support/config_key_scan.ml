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

(** The text a [STRING] token stands for, sliced out of the source rather than taken from the
    token's payload, whose shape has changed across compiler releases. Handles both spellings:
    ["key"] and the quoted forms ([{|key|}], [{tag|key|tag}]). *)
let string_literal_value content start stop =
  if start >= stop || not (Char.equal content.[start] '{') then
    if stop - start >= 2 then String.sub content ~pos:(start + 1) ~len:(stop - start - 2) else ""
  else
    match String.index_from content (start + 1) '|' with
    | None -> ""
    | Some open_bar -> (
        match String.rindex_from content (stop - 2) '|' with
        | Some close_bar when close_bar > open_bar ->
            String.sub content ~pos:(open_bar + 1) ~len:(close_bar - open_bar - 1)
        | _ -> "")

(** [content] with everything that is not a token replaced by spaces — comments, in other words —
    and with [~strings:true] the bodies of string literals too, delimiters kept. Every offset and
    newline is preserved, so a position found here still indexes the original text.

    Used by the scans that remain textual by nature: a [Utils.settings] field read, and finding the
    top-level definition an offset sits in. *)
let blank_bodies ?(strings = false) content =
  let buf = Bytes.make (String.length content) ' ' in
  String.iteri content ~f:(fun i c -> if Char.equal c '\n' then Bytes.set buf i '\n');
  let copy start stop =
    for i = start to min stop (String.length content) - 1 do
      Bytes.set buf i content.[i]
    done
  in
  List.iter (tokens content) ~f:(fun (tok, start, stop) ->
      match tok with
      | Parser.STRING _ when strings ->
          (* Keep the delimiters, drop the body: a scanner may still need to see that a literal is
             there without seeing what is in it. *)
          let opening =
            if Char.equal content.[start] '{' then
              match String.index_from content (start + 1) '|' with
              | Some bar -> bar + 1 - start
              | None -> 1
            else 1
          in
          let closing =
            if Char.equal content.[start] '{' then
              match String.rindex_from content (stop - 2) '|' with
              | Some bar when bar > start -> stop - bar
              | _ -> 1
            else 1
          in
          copy start (start + opening);
          copy (stop - closing) stop
      | _ -> copy start stop);
  Bytes.to_string buf

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
  let literal_at i =
    if i < Array.length toks then
      match toks.(i) with
      | Parser.STRING _, start, stop -> Some (string_literal_value content start stop)
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
  |> List.filter ~f:(fun s -> (not (String.is_empty s)) && not (String.contains s '\n'))

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
  let content = blank_bodies content in
  let ident_at pos =
    let n = String.length content in
    let stop =
      Option.value
        (String.lfindi content ~pos ~f:(fun _ c ->
             not (Char.is_alphanum c || Char.equal c '_')))
        ~default:n
    in
    String.sub content ~pos ~len:(stop - pos)
  in
  (* Qualified: [Low_level.virtualize_settings] and friends are records of the same shape whose
     field names are NOT config keys ([max_visits] against [virtualize_max_visits]), so an
     unqualified match would attribute reads to keys that do not exist. *)
  let marker = "Utils.settings." in
  let rec fields i acc =
    match String.substr_index ~pos:i content ~pattern:marker with
    | None -> acc
    | Some start ->
        let field = ident_at (start + String.length marker) in
        fields (start + 1) (if String.is_empty field then acc else field :: acc)
  in
  let predicates =
    List.concat_map
      [
        ("debug_log_from_routines ()", [ "debug_log_from_routines"; "log_level" ]);
        ("with_runtime_debug ()", [ "output_debug_files_in_build_directory"; "log_level" ]);
      ]
      ~f:(fun (call, keys) ->
        if Option.is_some (String.substr_index content ~pattern:call) then keys else [])
  in
  fields 0 [] @ predicates

(** Every configuration read of a file — [arg_name] call sites and {!settings_keys_in_source} —
    keyed by file basename, for tests that care {e where} a key is read. Field names that are not
    config keys come along; callers intersect with the registry. *)
let keys_by_file files =
  List.map files ~f:(fun fname ->
      let content = Stdio.In_channel.read_all fname in
      ( Stdlib.Filename.basename fname,
        Set.of_list (module String) (keys_in_source content @ settings_keys_in_source content) ))
