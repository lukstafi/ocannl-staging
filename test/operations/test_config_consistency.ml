open Base
open Stdio
module Config_key_scan = Test_utils.Config_key_scan

(* The reference file ships with every setting COMMENTED OUT (gh-ocannl-559), so that copying it
   wholesale to an `ocannl_config` states no settings. A commented-out setting is spelled `#key=…`
   with no space after the `#`; prose comments (and the verbatim profile-payload blocks at the end
   of the file) always use `# `, so the two are told apart mechanically. *)
let uncomment line =
  match String.chop_prefix line ~prefix:"#" with
  | Some rest -> (
      match String.lsplit2 rest ~on:'=' with
      | Some (key, _)
        when (not (String.is_empty key))
             && String.for_all key ~f:(fun c ->
                    Char.is_alphanum c || Char.equal '_' c || Char.equal '-' c) ->
          rest
      | _ -> line)
  | None -> line

let extract_keys filename =
  In_channel.read_lines filename
  |> List.map ~f:(fun line -> uncomment (String.strip line))
  |> List.filter_map ~f:(fun line ->
      if
        String.is_empty line || String.is_prefix ~prefix:"#" line
        || String.is_prefix ~prefix:"~~" line
      then None
      else
        match String.lsplit2 line ~on:'=' with
        | Some (key, _) ->
            let key =
              String.lowercase
              @@ String.strip ~drop:(fun c -> Char.equal '-' c || Char.equal ' ' c) key
            in
            let key =
              if String.is_prefix key ~prefix:"ocannl" then
                String.drop_prefix key 6 |> String.strip ~drop:(Char.equal '_')
              else key
            in
            if String.is_empty key then None else Some key
        | None -> None)
  |> Set.of_list (module String)

let () =
  if Array.length Stdlib.Sys.argv < 3 then (
    eprintf "Usage: %s <reference_file> <source_file...>\n" Stdlib.Sys.argv.(0);
    Stdlib.exit 1);
  let reference_file = Stdlib.Sys.argv.(1) in
  let source_files =
    Array.to_list (Array.sub Stdlib.Sys.argv ~pos:2 ~len:(Array.length Stdlib.Sys.argv - 2))
  in
  let file_keys = extract_keys reference_file in
  let code_keys = Utils.known_config_keys in
  let source_keys = Config_key_scan.keys_in_files source_files in
  let ok = ref true in
  let fail msg =
    ok := false;
    printf "FAIL: %s\n" msg
  in
  (* 1. Source call-site keys must all appear in the reference file *)
  let missing_in_ref = Set.diff source_keys file_keys in
  if not (Set.is_empty missing_in_ref) then
    fail
      (Printf.sprintf "call-site keys missing from %s: %s" reference_file
         (String.concat ~sep:", " @@ Set.to_list missing_in_ref));
  (* 2. Source call-site keys must all appear in known_config_keys registry *)
  let missing_in_registry = Set.diff source_keys code_keys in
  if not (Set.is_empty missing_in_registry) then
    fail
      (Printf.sprintf "call-site keys missing from known_config_keys registry: %s"
         (String.concat ~sep:", " @@ Set.to_list missing_in_registry));
  (* 3. known_config_keys and reference file must agree (bidirectional) *)
  let extra_in_ref = Set.diff file_keys code_keys in
  let extra_in_registry = Set.diff code_keys file_keys in
  if not (Set.is_empty extra_in_ref) then
    fail
      (Printf.sprintf "reference file has keys not in known_config_keys: %s"
         (String.concat ~sep:", " @@ Set.to_list extra_in_ref));
  if not (Set.is_empty extra_in_registry) then
    fail
      (Printf.sprintf "known_config_keys has keys not in reference file: %s"
         (String.concat ~sep:", " @@ Set.to_list extra_in_registry));
  (* 4. Profile payloads (gh-ocannl-559) are partial config files: every key they set must be a
     known, documented setting -- a payload is not a place a new key can hide. *)
  let payload_keys =
    List.map Utils.profile_payloads ~f:(fun (name, text) ->
        let keys =
          Utils.parse_config_lines ~source:("profile " ^ name) (String.split_lines text)
          |> List.map ~f:fst
          |> Set.of_list (module String)
        in
        let unknown = Set.diff keys code_keys in
        if not (Set.is_empty unknown) then
          fail
            (Printf.sprintf "profile %S sets keys missing from known_config_keys registry: %s" name
               (String.concat ~sep:", " @@ Set.to_list unknown));
        let undocumented = Set.diff keys file_keys in
        if not (Set.is_empty undocumented) then
          fail
            (Printf.sprintf "profile %S sets keys missing from %s: %s" name reference_file
               (String.concat ~sep:", " @@ Set.to_list undocumented));
        (name, keys))
  in
  (* 5. The payload texts are reproduced in the reference file between BEGIN/END markers, each line
     prefixed with "# ". Documentation of a value that can drift from the value is worse than no
     documentation, so the copy is checked rather than trusted. *)
  let reference_lines = In_channel.read_lines reference_file in
  List.iter Utils.profile_payloads ~f:(fun (name, text) ->
      let marker kind = Printf.sprintf "# --- %s PROFILE PAYLOAD %s ---" kind name in
      let index_of m =
        List.findi reference_lines ~f:(fun _ l -> String.equal (String.strip l) m)
        |> Option.map ~f:fst
      in
      match (index_of (marker "BEGIN"), index_of (marker "END")) with
      | Some b, Some e when e > b ->
          let quoted =
            List.sub reference_lines ~pos:(b + 1) ~len:(e - b - 1)
            |> List.map ~f:(fun l ->
                let l = Option.value (String.chop_prefix l ~prefix:"#") ~default:l in
                Option.value (String.chop_prefix l ~prefix:" ") ~default:l)
          in
          let expected = String.split_lines text in
          if not (List.equal String.equal quoted expected) then
            fail
              (Printf.sprintf
                 "the %s payload quoted in %s differs from the embedded one in \
                  arrayjit/lib/utils.ml"
                 name reference_file)
      | _ ->
          fail
            (Printf.sprintf "%s has no '%s' … '%s' block" reference_file (marker "BEGIN")
               (marker "END")));
  (* 6. Both consistency tests (this one and digest_completeness) find a configuration read by
     scanning sources for the key spelled as a string literal at the call site
     (Test_utils.Config_key_scan). That convention is load-bearing: a helper that takes the key as a
     parameter hides every key routed through it from BOTH scanners -- on staging PR #337 one such
     helper hid three real keys, and a deliberately unregistered fake key stayed green until the
     helper was removed. So the label must carry a literal everywhere except in the functions that
     legitimately move a key around, named one by one below:
     - in utils.ml, the lookup plumbing itself, which forwards the name between get_global_arg,
       get_global_flag and get_global_arg_with_source and on into the boolean parser;
     - in tnode.ml, get_style, which takes the key as an optional parameter defaulting to a
       literal, re-passed by its callers with literals of their own.
     Naming the functions rather than exempting their files is the difference between "this
     forwarding is known" and "nothing in this file is ever checked" (Codex P2, round 1): utils.ml
     and tnode.ml go on reading configuration in the ordinary way everywhere else, and a new
     forwarding helper in either of them fails like anywhere else.
     Only the labelled-argument spellings count, and only in code: comment and string bodies are
     blanked first, so prose about the convention -- including this comment's neighbours in the
     scanned files -- is not a call site. *)
  let forwarding_sites =
    Map.of_alist_exn
      (module String)
      [
        ( "utils.ml",
          Set.of_list
            (module String)
            [
              "bool_of_config_string";
              "resolve_config_value";
              "get_global_arg_with_source";
              "get_global_arg";
              "get_global_flag";
            ] );
        ("tnode.ml", Set.of_list (module String) [ "get_style" ]);
      ]
  in
  (* Both halves of this check are the lexer's answer rather than a pattern's
     (Config_key_scan): which uses of the label are not literals -- covering every spelling OCaml
     accepts, including the optional forms a textual matcher missed (Codex P2, round 3) -- and
     which top-level definition an offset sits in, so that a documentation comment quoting a
     column-zero `let get_global_arg` cannot lend an exempt name to unrelated code (round 4). Only
     the line quoted back to the reader is sliced from the text, at the offset the lexer reported. *)
  let line_at original i =
    let line_start =
      match String.rfindi original ~pos:i ~f:(fun _ c -> Char.equal c '\n') with
      | Some k -> k + 1
      | None -> 0
    in
    let line_end =
      match String.lfindi original ~pos:i ~f:(fun _ c -> Char.equal c '\n') with
      | Some k -> k
      | None -> String.length original
    in
    String.strip (String.sub original ~pos:line_start ~len:(line_end - line_start))
  in
  let forwarding_hit = ref (Set.empty (module String)) in
  List.iter source_files ~f:(fun fname ->
      let base = Stdlib.Filename.basename fname in
      let known =
        Option.value (Map.find forwarding_sites base) ~default:(Set.empty (module String))
      in
      let original = In_channel.read_all fname in
      let definitions = Config_key_scan.definitions original in
      let enclosing offset = Config_key_scan.definition_at definitions offset in
      List.iter (Config_key_scan.label_uses original) ~f:(fun use ->
          match use.Config_key_scan.key with
          | Some _ -> ()
          | None -> (
              let offset = use.Config_key_scan.offset in
              match enclosing offset with
              | Some d when Set.mem known d ->
                  forwarding_hit := Set.add !forwarding_hit (base ^ ":" ^ d)
              | definition ->
                  fail
                  @@ Printf.sprintf "%s does not spell the config key as a string literal, in %s: %s"
                       base
                       (Option.value definition ~default:"<no enclosing definition>")
                       (line_at original offset))));
  (* An exemption is a claim that a named function forwards a key; a claim that stops being true
     is stale, not a free pass, so each one has to earn its place on every run. *)
  let exempted_sites =
    Map.to_alist forwarding_sites
    |> List.concat_map ~f:(fun (base, fns) ->
           List.map (Set.to_list fns) ~f:(fun fn -> base ^ ":" ^ fn))
    |> Set.of_list (module String)
  in
  let stale = Set.diff exempted_sites !forwarding_hit in
  if not (Set.is_empty stale) then
    fail
      (Printf.sprintf
         "exempted functions that no longer forward a config key -- drop them from the exemption \
          list: %s"
         (String.concat ~sep:", " @@ Set.to_list stale));
  if !ok then (
    printf
      "OK: %d call-site keys, all in reference file and registry; registry and reference agree on \
       %d keys.\n"
      (Set.length source_keys) (Set.length code_keys);
    printf "OK: %d profile payloads, documented and quoted verbatim: %s.\n"
      (List.length payload_keys)
      (String.concat ~sep:", "
      @@ List.map payload_keys ~f:(fun (name, keys) ->
             Printf.sprintf "%s (%d keys)" name (Set.length keys)));
    printf
      "OK: %d files spell every config key as a string literal, outside %d forwarding functions: \
       %s.\n"
      (List.length source_files) (Set.length exempted_sites)
      (String.concat ~sep:", " @@ Set.to_list exempted_sites))
