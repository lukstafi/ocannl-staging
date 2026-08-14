(** Scanning OCaml sources for the configuration keys they read.

    Shared by the two consistency tests that hold the configuration honest:
    [test_config_consistency] (every key documented and registered) and [digest_completeness] (every
    key classified against the cache identity, gh-ocannl-572). Both need the same fact — which keys
    a source file reads — so they read it the same way. *)

open Base

(** [content] with the bodies of comments — and, with [~strings:true], of string literals —
    replaced by spaces, every offset and newline preserved so positions still line up.

    These scans read text, but they mean to describe {e code}. Prose that spells a marker is not a
    use of it: a comment writing the call-site form produced a phantom key called "literal" on
    staging PR #337, and a comment writing the label alone would report a configuration read that
    does not exist. Blanking is what makes "a comment is not a call site" true rather than merely
    intended. String bodies are left alone by default because that is where the keys live; the
    non-literal check blanks them too, since it cares only about the delimiters.

    Nested comments, string literals inside comments, character literals (['"'] would otherwise
    open a string) and quoted-string literals ([{|…|}], [{tag|…|tag}]) are all tracked, because a
    desynchronised scan fails silently rather than loudly. *)
let blank_bodies ?(strings = false) content =
  let n = String.length content in
  let buf = Bytes.of_string content in
  let blank i = if not (Char.equal content.[i] '\n') then Bytes.set buf i ' ' in
  let at i s = i + String.length s <= n && String.is_substring_at content ~pos:i ~substring:s in
  (* A character literal, not the start of a string: ['a'], ['\n'], ['\\'], ['"']. A type variable
     (['a t]) has no closing quote and must not be skipped. *)
  let char_literal_len i =
    if i + 2 < n && Char.equal content.[i + 1] '\\' then
      let rec close j = if j < n && j <= i + 6 then if Char.equal content.[j] '\'' then Some (j - i + 1) else close (j + 1) else None in
      close (i + 2)
    else if i + 2 < n && Char.equal content.[i + 2] '\'' then Some 3
    else None
  in
  (* [{tag|…|tag}]: the tag is a (possibly empty) lowercase identifier. *)
  let quoted_string_end i =
    let rec tag j =
      if j < n && (Char.is_lowercase content.[j] || Char.equal content.[j] '_') then tag (j + 1)
      else if j < n && Char.equal content.[j] '|' then Some (String.sub content ~pos:(i + 1) ~len:(j - i - 1))
      else None
    in
    match tag (i + 1) with
    | None -> None
    | Some tg -> (
        let closing = "|" ^ tg ^ "}" in
        let body_start = i + 1 + String.length tg + 1 in
        match String.substr_index content ~pos:body_start ~pattern:closing with
        | None -> None
        | Some j -> Some (body_start, j, j + String.length closing))
  in
  (* [depth] is the comment nesting level; 0 is code. Inside a comment everything is blanked. *)
  let rec walk i depth =
    if i >= n then ()
    else if at i "(*" then (
      blank i;
      blank (i + 1);
      walk (i + 2) (depth + 1))
    else if depth > 0 && at i "*)" then (
      blank i;
      blank (i + 1);
      walk (i + 2) (depth - 1))
    else if Char.equal content.[i] '"' then (
      if depth > 0 then blank i;
      in_string (i + 1) depth)
    else
      (* Quoted strings are lexed INSIDE comments too, exactly as OCaml lexes them (Codex P2, round
         2): in [(* {| (* |} *)] the inner opener belongs to the quoted string, and taking it for a
         nested comment would leave the walk one level deep for the rest of the file -- blanking
         live code, and dropping its keys from every scan without a word. *)
      match if Char.equal content.[i] '{' then quoted_string_end i else None with
      | Some (body_start, body_end, after) ->
          if depth > 0 then
            for j = i to after - 1 do
              blank j
            done
          else if strings then
            for j = body_start to body_end - 1 do
              blank j
            done;
          walk after depth
      | None ->
          if depth > 0 then (
            blank i;
            walk (i + 1) depth)
          else (
            match char_literal_len i with
            | Some len when Char.equal content.[i] '\'' -> walk (i + len) depth
            | _ -> walk (i + 1) depth)
  and in_string i depth =
    if i >= n then ()
    else if Char.equal content.[i] '\\' && i + 1 < n then (
      if depth > 0 || strings then (
        blank i;
        blank (i + 1));
      in_string (i + 2) depth)
    else if Char.equal content.[i] '"' then (
      if depth > 0 then blank i;
      walk (i + 1) depth)
    else (
      if depth > 0 || strings then blank i;
      in_string (i + 1) depth)
  in
  walk 0 0;
  Bytes.to_string buf

(** The [arg_name] literals of the [get_global_arg] / [get_global_flag] calls in [content]. Two
    forms appear in the codebase: [~arg_name:"key"] (direct call sites) and [?(arg_name = "key")]
    (optional parameter defaults, e.g. [get_style] in tnode.ml).

    A key that reaches the lookup any other way — through a helper taking the name as a
    parameter — is invisible to this scan, and hence to both tests built on it. That is why
    [test_config_consistency] separately fails any non-literal use of the label outside the handful
    of named functions that implement the lookup. *)
let keys_in_source content =
  let content = blank_bodies content in
  let find_all marker =
    let mlen = String.length marker in
    let n = String.length content in
    let rec loop i acc =
      match String.substr_index ~pos:i content ~pattern:marker with
      | None -> acc
      | Some start ->
          let key_start = start + mlen in
          let key_end =
            match String.lfindi content ~pos:key_start ~f:(fun _ c -> Char.equal c '"') with
            | None -> n
            | Some j -> j
          in
          let key = String.sub content ~pos:key_start ~len:(key_end - key_start) in
          loop (key_end + 1) (key :: acc)
    in
    loop 0 []
  in
  find_all {|arg_name:"|} @ find_all {|arg_name = "|}
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
