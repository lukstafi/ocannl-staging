open Base
open Stdio

let extract_config arg =
  let prefixes = [ "--read="; "--read-"; "--read_" ] in
  List.find_map prefixes ~f:(fun prefix ->
      Option.map (String.chop_prefix arg ~prefix) ~f:(fun config -> config))

let extract_output arg =
  let prefixes = [ "--output="; "--output-"; "--output_" ] in
  List.find_map prefixes ~f:(fun prefix ->
      Option.map (String.chop_prefix arg ~prefix) ~f:(fun output -> output))

type output_destination = File | Stdout | Env

let parse_output_destination = function
  | "file" -> File
  | "stdout" -> Stdout
  | "env" -> Env
  | _ -> File (* Default to file *)

let output_config config_name value = function
  | File -> (
      let filename = Config_reads.file_of_key config_name in
      try
        Out_channel.write_all filename ~data:value;
        printf "Wrote value of '%s' to %s\n" config_name filename
      with exn -> eprintf "Error writing to %s: %s\n" filename (Exn.to_string exn))
  | Stdout -> printf "%s\n" value
  | Env ->
      let env_var = "CURRENT_OCANNL_" ^ String.uppercase config_name in
      printf "export %s='%s'\n" env_var value

let () =
  let config_opt = Array.find_map Stdlib.Sys.argv ~f:extract_config in
  let output_opt = Array.find_map Stdlib.Sys.argv ~f:extract_output in
  let output_dest = parse_output_destination (Option.value output_opt ~default:"file") in
  match config_opt with
  | Some key when List.mem Config_reads.backend_keys key ~equal:String.equal ->
      (* What the backend setting decides is decided in [Config_reads], where the scans reading
         these files back find what each value means (gh-ocannl-1581). *)
      let setting = Utils.get_global_arg ~default:"" ~arg_name:"backend" in
      output_config key (Option.value_exn (Config_reads.written ~key setting)) output_dest
  | Some config ->
      (* The raw resolved text, or nothing: a registered key's plain read is refused
         (gh-ocannl-1332), and this tool reports what is stated, not a parse of it. *)
      let value = Option.value_map (Utils.profile_key_source config) ~default:"" ~f:fst in
      output_config config value output_dest
  | None -> printf "No --read=<config>, --read-<config>, or --read_<config> argument found.\n"
