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

(** [keys_in_source] over each file, as a set. *)
let keys_in_files files =
  List.concat_map files ~f:(fun fname -> keys_in_source (Stdio.In_channel.read_all fname))
  |> Set.of_list (module String)

(** The same, keyed by file basename — for tests that care {e where} a key is read, not only that it
    is. *)
let keys_by_file files =
  List.map files ~f:(fun fname ->
      ( Stdlib.Filename.basename fname,
        Set.of_list (module String) (keys_in_source (Stdio.In_channel.read_all fname)) ))
