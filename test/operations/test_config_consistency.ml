open Base
open Stdio

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

(* Scan OCaml source files for arg_name literal strings in get_global_arg/get_global_flag calls. Two
   forms appear in the codebase: - ~arg_name:"key" (direct call sites) - ?(arg_name = "key")
   (optional parameter defaults, e.g. get_style in tnode.ml) *)
let extract_source_keys source_files =
  let find_all_in content marker =
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
  List.concat_map source_files ~f:(fun fname ->
      let content = In_channel.read_all fname in
      find_all_in content {|arg_name:"|} @ find_all_in content {|arg_name = "|})
  |> List.filter ~f:(fun s -> (not (String.is_empty s)) && not (String.contains s '\n'))
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
  let source_keys = extract_source_keys source_files in
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
  if !ok then (
    printf
      "OK: %d call-site keys, all in reference file and registry; registry and reference agree on \
       %d keys.\n"
      (Set.length source_keys) (Set.length code_keys);
    printf "OK: %d profile payloads, documented and quoted verbatim: %s.\n"
      (List.length payload_keys)
      (String.concat ~sep:", "
      @@ List.map payload_keys ~f:(fun (name, keys) ->
             Printf.sprintf "%s (%d keys)" name (Set.length keys))))
