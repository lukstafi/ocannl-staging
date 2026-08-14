(** Scanning OCaml sources for the configuration keys they read.

    Shared by the two consistency tests that hold the configuration honest:
    [test_config_consistency] (every key documented and registered) and [digest_completeness] (every
    key classified against the cache identity, gh-ocannl-572). Both need the same fact — which keys
    a source file reads — so they read it the same way. *)

open Base

(** The [arg_name] literals of the [get_global_arg] / [get_global_flag] calls in [content]. Two
    forms appear in the codebase: [~arg_name:"key"] (direct call sites) and [?(arg_name = "key")]
    (optional parameter defaults, e.g. [get_style] in tnode.ml). *)
let keys_in_source content =
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
