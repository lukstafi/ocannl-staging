open Base

type settings = {
  mutable log_level : int;
  mutable debug_log_from_routines : bool;
      (** If the [debug_log_from_routines] flag is true _and_ the flag [log_level > 1], backends
          should generate code (e.g. fprintf statements) to log the execution, and arrange for the
          logs to be emitted via ppx_minidebug. *)
  mutable output_debug_files_in_build_directory : bool;
      (** Writes compilation related files in the [build_files] subdirectory of the run directory
          (additional files, or files that would otherwise be in temp directory). When both
          [output_debug_files_in_build_directory = true] and [log_level > 1], compilation should
          also preserve debug and line information for runtime debugging. *)
  mutable fixed_state_for_init : int option;
  mutable print_decimals_precision : int;
      (** When rendering arrays etc., outputs this many decimal digits. *)
  mutable check_half_prec_constants_cutoff : float option;
      (** If given, generic code optimization should fail if a half precision FP16 constant exceeds
          the cutoff. *)
  mutable default_prng_variant : string;
      (** The default variant of threefry4x32 PRNG to use. Options: "crypto" (20 rounds) or "light"
          (2 rounds). Defaults to "light" for better performance. *)
  mutable large_models : bool;
      (** If true, use uint64 for indexing arithmetic. If false, use uint32 for indexing arithmetic.
          This affects all backends' kernel index parameters and local index variables, and gates
          the per-pool offset width (uint32 caps a pool at 4 GB; see the pool allocator). Decoupled
          in intent from element indexing within a single tensor, though both currently follow it.
      *)
}
[@@deriving sexp]

let settings =
  {
    log_level = 0;
    debug_log_from_routines = false;
    output_debug_files_in_build_directory = false;
    fixed_state_for_init = None;
    print_decimals_precision = 2;
    check_half_prec_constants_cutoff = Some (2. **. 14.);
    default_prng_variant = "light";
    large_models = false;
  }

let accessed_global_args = Hash_set.create (module String)
let str_nonempty ~f s = if String.is_empty s then None else Some (f s)
let pair a b = (a, b)

let known_config_keys =
  Set.of_list
    (module String)
    [
      (* Bootstrap keys (read before config file via read_cmdline_or_env_var directly) *)
      "suppress_welcome_message";
      "no_config_file";
      "log_config_sourcing";
      (* The preset-bundle picker (gh-ocannl-559); resolved before any ordinary key *)
      "profile";
      (* Utils.settings *)
      "log_level";
      "debug_log_from_routines";
      "output_debug_files_in_build_directory";
      "fixed_state_for_init";
      "print_decimals_precision";
      "check_half_prec_constants_cutoff";
      "default_prng_variant";
      "large_models";
      "big_models" (* deprecated alias for large_models *);
      (* Cleanup / startup *)
      "build_files_prefix";
      "clean_up_build_files_on_startup";
      "clean_up_log_files_on_startup";
      "never_capture_stdout";
      (* ppx_minidebug *)
      "snapshot_every_sec";
      "time_tagged";
      "elapsed_times";
      "location_format";
      "debug_backend";
      "hyperlink_prefix";
      "logs_print_scope_ids";
      "logs_verbose_scope_ids";
      "log_main_domain_to_stdout";
      "log_file_stem";
      "prev_run_prefix";
      "toc_entry_minimal_depth";
      "toc_entry_minimal_size";
      "toc_entry_minimal_span";
      "debug_highlights";
      "debug_highlight_pcre";
      "diff_ignore_pattern_pcre";
      "diff_max_distance_factor";
      "debug_scope_id_pairs";
      "debug_log_truncate_children";
      "debug_log_prune_upto";
      "debug_log_to_stream_files";
      (* Backends *)
      "backend";
      "prefer_backend_uniformity";
      "cc_backend_optimization_level";
      "cc_backend_compiler_command";
      "cc_backend_arch_flags";
      "cc_backend_simd_flags";
      "cc_backend_fp_contract";
      "cc_backend_probe_cache";
      "cc_backend_fast_math";
      "cc_backend_post_compile_timeout";
      "cc_backend_verify_codesign";
      "output_dlls_in_build_directory";
      "cuda_printf_fifo_size";
      "hip_printf_fifo_size";
      "hip_scratch_validation";
      "gpu_graph_capture";
      "multidev_num_devices";
      "buffer_aliasing";
      "log_buffer_aliasing";
      "memory_budget";
      "log_memory_budget";
      (* Low-level / optimization *)
      "virtualize_max_visits";
      "virtualize_max_inline_reduction";
      "enable_device_only";
      "inline_scalar_constexprs";
      "inline_simple_computations";
      "inline_complex_computations";
      "output_prec_in_ll_files";
      "stack_threshold_in_bytes";
      (* Schedule layer (docs/proposals/schedule-ir-optops.md §6) *)
      "automatic_gpu_schedule";
      "gpu_schedule_block_size";
      "gpu_schedule_min_parallel";
      "automatic_cpu_schedule";
      "cpu_schedule_min_parallel";
      "schedule_fission";
      "schedule_log_launches";
      "schedule_log_declines";
      "legality_crosscheck";
      "cc_parallel_grid";
      "cc_parallel_chunks";
      "cc_pool_core_class";
      "cc_grid_private_bytes_cap";
      "cc_vector_bytes";
      "cc_fp16_arithmetic";
      (* Autotuning (Autotune.tune) *)
      "autotune_search";
      "autotune_beam_width";
      "autotune_rounds";
      "autotune_repeats";
      "autotune_cache_dir";
      "autotune_split_reduce_max_sites";
      "autotune_log";
      "tune_inline_flips";
      "tune_flip_ordering";
      "strict_failure_classification";
      (* Analytic cost model (gh-ocannl-491) *)
      "autotune_keep_fraction";
      "autotune_bound_pruning";
      "autotune_calibration_file";
      "model_default_schedule";
      "model_default_placements";
      "model_default_geometry_lattice";
      "model_peak_flops";
      "model_peak_memory_bandwidth";
      (* Numerics policy (gh-ocannl-478) *)
      "tf32_matmuls";
      "narrow_compute_f32";
      "fp16_arithmetic";
      (* Identifiers and other *)
      "ll_ident_style";
      "cd_ident_style";
      "default_prec";
      "limit_constant_fill_size";
      "max_shape_error_origins";
    ]

let bool_of_config_string ~arg_name s =
  match String.lowercase @@ String.strip s with
  | "true" | "1" -> true
  | "false" | "0" -> false
  | _ -> invalid_arg @@ "ocannl_" ^ arg_name ^ " setting should be a boolean; found: " ^ s

(** Whether to print where each configuration setting comes from (commandline, environment, config
    file, or the hard-coded default). It gates the logging of the config-reading functions
    themselves, so it is bootstrapped directly rather than via {!get_global_arg}: the initial value
    only reflects the commandline and the environment, and the config file setting is applied once
    the config file is read. Can also be set programmatically, e.g. to trace configs a test reads. *)
let log_config_sourcing = ref true

let env_var_names n =
  let env_variants = [ "ocannl_" ^ n; "ocannl-" ^ n ] in
  List.concat_map env_variants ~f:(fun n -> [ n; String.uppercase n ])

(** The commandline sublevel of {!get_global_arg}: returns the setting's value and the [Sys.argv]
    element it came from. Pure -- the sourcing log lives at the resolution seam, which is the only
    place that knows which sublevel actually won.

    [qualified_only] drops the prefix-free spellings, leaving the [ocannl_]-qualified ones. OCANNL
    is a library: it scans the host executable's [Sys.argv], so a prefix-free key claims an
    application's own option of that name. That is tolerable for keys nobody else would spell
    ([--virtualize_max_visits]) and not for [--profile], which is a common application flag and
    which OCANNL treats as fatal when it does not name a known bundle -- a host passing
    [--profile=prod] would die during module initialization (Codex P2 on PR #291). *)
let read_cmdline_var ?(qualified_only = false) n =
  let n_dash = String.tr ~target:'_' ~replacement:'-' n in
  (* Prefixed commandline variants first (backward compat), then prefix-free *)
  let cmd_prefixed = List.concat_map (env_var_names n) ~f:(fun n -> [ "-" ^ n; "--" ^ n; n ]) in
  let cmd_unprefixed =
    if qualified_only then []
    else
      let keys = if String.equal n n_dash then [ n ] else [ n; n_dash ] in
      List.concat_map keys ~f:(fun k -> [ "--" ^ k; "-" ^ k ])
  in
  let cmd_variants =
    List.concat_map (cmd_prefixed @ cmd_unprefixed) ~f:(fun n -> [ n ^ "_"; n ^ "-"; n ^ "="; n ])
  in
  Array.find_map Stdlib.Sys.argv ~f:(fun arg ->
      List.find_map cmd_variants ~f:(fun p ->
          Option.some_if (String.is_prefix ~prefix:p arg)
            (String.drop_prefix arg (String.length p), arg)))

(** The environment sublevel of {!get_global_arg}: returns the setting's value and the variable it
    came from. An empty value counts as unset. *)
let read_env_var n =
  match
    List.find_map (env_var_names n) ~f:(fun env_n ->
        Option.(join @@ map (Stdlib.Sys.getenv_opt env_n) ~f:(str_nonempty ~f:(pair env_n))))
  with
  | None | Some (_, "") -> None
  | Some (env_n, result) -> Some (result, env_n)

(** The bootstrap reader: the few keys that are consulted before the config file exists (and hence
    before profiles are resolved) come from the commandline or the environment only. *)
let read_cmdline_or_env_var n =
  let with_debug =
    !log_config_sourcing
    && (settings.log_level > 0 || equal_string n "log_level")
    && not (Hash_set.mem accessed_global_args n)
  in
  match read_cmdline_var n with
  | Some (result, arg) ->
      if with_debug then Stdio.printf "Found %s, commandline %s\n%!" result arg;
      Some result
  | None ->
      Option.map (read_env_var n) ~f:(fun (result, env_n) ->
          if with_debug then Stdio.printf "Found %s, environment %s\n%!" result env_n;
          result)

(* Originally from the library core.filename_base. *)
let filename_parts filename =
  let rec loop acc filename =
    match (Stdlib.Filename.dirname filename, Stdlib.Filename.basename filename) with
    | ("." as base), "." -> base :: acc
    | ("/" as base), "/" -> base :: acc
    | disk, base when String.is_suffix disk ~suffix:":\\" -> disk :: base :: acc
    | rest, dir -> loop (dir :: acc) rest
  in
  loop [] filename

(* Originally from the library core.filename_base. *)
let filename_of_parts = function
  | [] -> invalid_arg "Utils.filename_of_parts: empty parts list"
  | root :: rest -> List.fold rest ~init:root ~f:Stdlib.Filename.concat

let log_config_sourcing_arg = read_cmdline_or_env_var "log_config_sourcing"

let () =
  Option.iter log_config_sourcing_arg ~f:(fun v ->
      log_config_sourcing := bool_of_config_string ~arg_name:"log_config_sourcing" v)

(** Parses the [ocannl_config] syntax: one [key=value] per line, [#] and [~~] lines are comments,
    empty values mean "unset", the [ocannl_] key prefix is optional and keys are case-insensitive.
    Shared by the config file and by the embedded profile payloads (which are literally partial
    config files); [source] names the origin in error messages. *)
let parse_config_lines ~source lines =
  lines
  |> List.filter ~f:(fun l ->
      not (String.is_prefix ~prefix:"~~" l || String.is_prefix ~prefix:"#" l))
  |> List.map ~f:(String.split ~on:'=')
  |> List.filter_map ~f:(function
    | [] -> None
    | [ s ] when String.is_empty (String.strip s) -> None
    | key :: [ v ] ->
        let key =
          String.(lowercase @@ strip ~drop:(fun c -> equal_char '-' c || equal_char ' ' c) key)
        in
        let key =
          if String.is_prefix key ~prefix:"ocannl" then
            String.drop_prefix key 6 |> String.strip ~drop:(equal_char '_')
          else key
        in
        str_nonempty ~f:(pair key) v
    | l ->
        failwith @@ "OCANNL: invalid syntax in " ^ source
        ^ ", should have a single '=' on each non-empty line, found: " ^ String.concat l)

let config_table_of_lines ~source lines =
  match Hashtbl.of_alist (module String) (parse_config_lines ~source lines) with
  | `Ok h -> h
  | `Duplicate_key key -> failwith @@ "OCANNL: duplicate key in " ^ source ^ ": " ^ key

let config_file_args =
  let suppress_welcome_message () =
    Option.value_map ~default:false ~f:Bool.of_string
    @@ read_cmdline_or_env_var "suppress_welcome_message"
  in
  match read_cmdline_or_env_var "no_config_file" with
  | None | Some "false" ->
      let read = Stdio.In_channel.read_lines in
      let fname, config_lines =
        let rev_dirs = List.rev @@ filename_parts @@ Stdlib.Sys.getcwd () in
        let rec find_up = function
          | [] ->
              if not (suppress_welcome_message ()) then
                Stdio.printf
                  "\nWelcome to OCANNL! No ocannl_config file found along current path.\n%!";
              ("", [])
          | _ :: tl as rev_dirs -> (
              let fname = filename_of_parts (List.rev @@ ("ocannl_config" :: rev_dirs)) in
              try (fname, read fname) with Sys_error _ -> find_up tl)
        in
        find_up rev_dirs
      in
      let result = config_table_of_lines ~source:("the config file " ^ fname) config_lines in
      if String.length fname > 0 then
        Hashtbl.iter_keys result ~f:(fun key ->
            if not (Set.mem known_config_keys key) then
              Stdio.eprintf "OCANNL warning: unknown config key %S in %s\n%!" key fname);
      if
        String.length fname > 0
        && (not (suppress_welcome_message ()))
        && not
             (Option.value_map ~default:false ~f:Bool.of_string
             @@ Hashtbl.find result "suppress_welcome_message")
      then Stdio.printf "\nWelcome to OCANNL! Reading configuration defaults from %s.\n%!" fname;
      result
  | Some _ ->
      if not (suppress_welcome_message ()) then
        Stdio.printf "\nWelcome to OCANNL! Configuration defaults file is disabled.\n%!";
      Hashtbl.create (module String)

let () =
  (* The commandline and the environment take precedence, and were already applied above. *)
  if Option.is_none log_config_sourcing_arg then
    Option.iter (Hashtbl.find config_file_args "log_config_sourcing") ~f:(fun v ->
        log_config_sourcing := bool_of_config_string ~arg_name:"log_config_sourcing" v)

(** {2 Configuration profiles (gh-ocannl-559)} *)

(** The source levels a setting can come from, in decreasing priority. Each level splits into two
    sublevels: the keys stated explicitly at that level, then the payload of a profile {e picked} at
    that level -- so a specific setting always beats an aggregate one of equal immediacy, and a
    profile named on the commandline still overrides an exhaustive config file. *)
type config_level = Cmdline_level | Env_level | Config_file_level

let equal_config_level l1 l2 =
  match (l1, l2) with
  | Cmdline_level, Cmdline_level | Env_level, Env_level | Config_file_level, Config_file_level ->
      true
  | (Cmdline_level | Env_level | Config_file_level), _ -> false

let describe_config_level = function
  | Cmdline_level -> "the commandline"
  | Env_level -> "the environment"
  | Config_file_level -> "the config file"

(** Where {!get_global_arg} found a value; reported by the [log_config_sourcing] trace. *)
type config_source =
  | From_cmdline of string  (** the matching [Sys.argv] element *)
  | From_env of string  (** the matching environment variable *)
  | From_config_file
  | From_profile of config_level * string  (** the level that picked the profile, and its name *)
  | From_default

(** A short provenance tag, e.g. ["profile 'reproducible' via the commandline"]. *)
let config_source_label = function
  | From_cmdline _ -> "commandline"
  | From_env _ -> "environment"
  | From_config_file -> "config file"
  | From_profile (level, name) ->
      Printf.sprintf "profile '%s' via %s" name (describe_config_level level)
  | From_default -> "default"

let describe_config_source ~value ~default = function
  | From_cmdline arg -> Printf.sprintf "Found %s, commandline %s" value arg
  | From_env var -> Printf.sprintf "Found %s, environment %s" value var
  | From_config_file -> Printf.sprintf "Found %s, in the config file" value
  | From_profile (level, name) ->
      Printf.sprintf "Found %s, in the profile %S picked via %s" value name
        (describe_config_level level)
  | From_default -> Printf.sprintf "Not found, using default %s" default

(** The precedence walk, factored out of {!get_global_arg} so it can be exercised on synthetic
    sources (see test/operations/config_profiles.ml). The lookups are pure; logging happens at the
    call site, which is the only place that knows which sublevel won. *)
let resolve_config_value ~cmdline ~env ~file ~profile ~default ~arg_name =
  let from_profile level =
    match profile with
    | Some (l, name, lookup) when equal_config_level l level ->
        Option.map (lookup arg_name) ~f:(fun v -> (v, From_profile (level, name)))
    | _ -> None
  in
  let sublevels =
    [
      (fun () -> Option.map (cmdline arg_name) ~f:(fun (v, arg) -> (v, From_cmdline arg)));
      (fun () -> from_profile Cmdline_level);
      (fun () -> Option.map (env arg_name) ~f:(fun (v, var) -> (v, From_env var)));
      (fun () -> from_profile Env_level);
      (fun () -> Option.map (file arg_name) ~f:(fun v -> (v, From_config_file)));
      (fun () -> from_profile Config_file_level);
    ]
  in
  Option.value (List.find_map sublevels ~f:(fun f -> f ())) ~default:(default, From_default)

(** The keys a profile payload may not set: they are read before profiles are resolved (or would
    make the resolution recursive). *)
let profile_ineligible_keys =
  Set.of_list (module String) [ "profile"; "no_config_file"; "log_config_sourcing" ]

(* The payloads are embedded rather than installed as files: the config search walks up from the
   working directory and would find the USER's config, so a shipped preset file would need
   share-directory machinery -- painful on Windows, and wrong under `dune runtest` where the working
   directory is _build/default/<test dir>. Embedded text costs no distribution machinery and stays
   inspectable: it is reproduced verbatim in ocannl_config.reference and its provenance is reported
   by the log_config_sourcing trace. *)

let reproducible_profile_payload =
  {|# Deterministic, and identical across machines wherever reasonable. Cross-BACKEND
# reproducibility is explicitly out of scope: this profile does not try to make the cc
# and the cuda backends agree bit for bit.

# The largest cross-machine determinism leak: schedule identity pins numerics (loop order,
# split reductions, tensorization), so two machines crowning two schedules compute two
# results. Replaying stays allowed -- a pinned schedule is deterministic -- but only from a
# CHOSEN autotune_cache_dir: with the search off the built-in default is treated as no cache,
# so an earlier local search's leftovers in ./autotune_cache cannot silently pin a schedule.
autotune_search=false
# The analytic cost model picks schedules from per-backend (and, once calibrated, per-machine)
# envelope constants -- the same leak without the timing runs.
model_default_schedule=false
# The greedy inlining refinement of Train.tune_placements accepts a flip when the timing
# improves, so which nodes end up inlined is machine-dependent -- and inlining is not
# numerics-neutral where storage precision is narrower than compute precision: a materialized
# node rounds to its storage precision, an inlined one does not.
tune_inline_flips=0

# `-mcpu=native` / `-march=native` changes which FMA and vector instructions the compiler may
# use, hence the results, per machine. "none" passes no architecture flag.
cc_backend_arch_flags=none
# Whether the SIMD probe fires depends on what the toolchain's DEFAULT target already exposes,
# which is itself a per-machine fact -- so pin the flags off rather than reason about which of
# them could have changed a result.
cc_backend_simd_flags=none
# Contraction the codegen did not ask for (the explicit `fmaf` selections stay): whether a*b+c
# becomes one fused op is compiler- and target-discretionary.
cc_backend_fp_contract=off
cc_backend_fast_math=false
# Explicit SIMD rendering reassociates strict-FP reductions into per-lane accumulator chains,
# so the summation order would follow the probed vector width. 0 keeps the serial order.
cc_vector_bytes=0

# The numerics gates at their exact defaults, so that a profile switch never silently changes
# what math is being done (they are the orthogonal axis; see the issue).
tf32_matmuls=false
fp16_arithmetic=false
narrow_compute_f32=true
|}

let performance_profile_payload =
  {|# The fastest configuration AT UNCHANGED SEMANTICS. Result-changing gates (tf32_matmuls,
# fp16_arithmetic's accuracy-for-throughput trade is the one exception below, and the
# algebraic-rewrite tiers) belong to the numerics axis, not to this one.

# Empirical schedule search on, with a wider beam and more rounds than the everyday defaults.
autotune_search=true
autotune_beam_width=4
autotune_rounds=4
# Untuned compiles still get the cost model's pick instead of the plain default pipeline.
model_default_schedule=true

# Target the host CPU, and let the SIMD probe add what that target already supports.
cc_backend_arch_flags=auto
cc_backend_simd_flags=auto

# Native 16-bit arithmetic where the target has it; ignored on targets that promote fp16 to
# float, and on the GPU backends.
fp16_arithmetic=true
|}

(** The embedded profile payloads, by name. Each is literally a partial [ocannl_config] file: same
    syntax, same parser, setting only the keys where the profiles' goals disagree. *)
let profile_payloads =
  [ ("reproducible", reproducible_profile_payload); ("performance", performance_profile_payload) ]

let parse_profile_payload ~name text =
  let source = Printf.sprintf "the built-in profile %S" name in
  let table = config_table_of_lines ~source (String.split_lines text) in
  Hashtbl.iter_keys table ~f:(fun key ->
      if Set.mem profile_ineligible_keys key then
        failwith @@ "OCANNL: " ^ source ^ " sets the key " ^ key
        ^ ", which is resolved before profiles are"
      else if not (Set.mem known_config_keys key) then
        failwith @@ "OCANNL: " ^ source ^ " sets the unknown config key " ^ key);
  table

(** The profile picked for this run, if any: its level (which decides the priority of its payload),
    its name, and the parsed payload. *)
let active_profile =
  (* An EMPTY value is unset, at each level independently: everywhere else in the configuration
     "" means "as if absent", and a launcher expanding [--ocannl_profile=$PROFILE] with an unset
     variable must not thereby disable a profile the environment or the config file names (Codex P2
     on PR #291). So the fall-through tests each level's value, not just its presence. *)
  let normalize name = str_nonempty ~f:Fn.id (String.lowercase (String.strip name)) in
  let picked =
    List.find_map
      [
        (* [--ocannl_profile=...], not [--profile=...]: see [read_cmdline_var]'s [qualified_only]. *)
        (Cmdline_level, Option.map (read_cmdline_var ~qualified_only:true "profile") ~f:fst);
        (Env_level, Option.map (read_env_var "profile") ~f:fst);
        (Config_file_level, Hashtbl.find config_file_args "profile");
      ]
      ~f:(fun (level, value) ->
        Option.map (Option.bind value ~f:normalize) ~f:(fun name -> (level, name)))
  in
  Option.map picked ~f:(fun (level, name) ->
      match List.Assoc.find profile_payloads name ~equal:String.equal with
      | None ->
          invalid_arg @@ "OCANNL: unknown ocannl_profile " ^ name ^ " (picked via "
          ^ describe_config_level level ^ "); known profiles: "
          ^ String.concat ~sep:", " (List.map profile_payloads ~f:fst)
      | Some text ->
          if !log_config_sourcing then
            Stdio.printf "\nOCANNL: using the configuration profile %S, picked via %s.\n%!" name
              (describe_config_level level);
          (level, name, parse_profile_payload ~name text))

let profile_lookup =
  Option.map active_profile ~f:(fun (level, name, table) ->
      (level, name, fun key -> Hashtbl.find table key))

(** Retrieves the [arg_name] setting from the commandline, the environment, the config file, or the
    payload of the profile picked at one of those levels; returns [default] if none has it, together
    with where the value came from. *)
let get_global_arg_with_source ~default ~arg_name:n =
  let with_debug =
    !log_config_sourcing
    && (settings.log_level > 0 || equal_string n "log_level")
    && not (Hash_set.mem accessed_global_args n)
  in
  if with_debug then
    Stdio.printf "Retrieving commandline, environment, or config file variable ocannl_%s\n%!" n;
  let result, source =
    resolve_config_value ~cmdline:read_cmdline_var ~env:read_env_var
      ~file:(Hashtbl.find config_file_args) ~profile:profile_lookup ~default ~arg_name:n
  in
  if with_debug then Stdio.printf "%s\n%!" (describe_config_source ~value:result ~default source);
  Hash_set.add accessed_global_args n;
  (result, source)

let get_global_arg ~default ~arg_name = fst (get_global_arg_with_source ~default ~arg_name)

let get_global_flag ~default ~arg_name:n =
  bool_of_config_string ~arg_name:n
  @@ get_global_arg ~default:(if default then "true" else "false") ~arg_name:n

let original_log_level =
  let log_level =
    let s = String.strip @@ get_global_arg ~default:"1" ~arg_name:"log_level" in
    match Int.of_string_opt s with
    | Some ll -> ll
    | None -> invalid_arg @@ "ocannl_log_level setting should be an integer; found: " ^ s
  in
  settings.log_level <- log_level;
  log_level

(* Originally from the library core.filename_base. *)
let filename_concat p1 p2 =
  if String.is_empty p1 then
    invalid_arg
    @@ "Utils.filename_concat called with an empty string as its first argument, second argument: "
    ^ p2;
  let rec collapse_trailing s =
    match String.rsplit2 s ~on:'/' with
    | Some ("", ("." | "")) -> ""
    | Some (s, ("." | "")) -> collapse_trailing s
    | None | Some _ -> s
  in
  let rec collapse_leading s =
    match String.lsplit2 s ~on:'/' with
    | Some (("." | ""), s) -> collapse_leading s
    | Some _ | None -> s
  in
  collapse_trailing p1 ^ "/" ^ collapse_leading p2

let clean_filename fname =
  let fname = String.strip fname in
  let fname =
    (* Beyond the path separators, the rest of the set Windows reserves: a routine named after a
       diagnostic can carry any of them (gh-ocannl-481's "... unit count > 1" reached [open_out] as
       a filename and killed the test process on Windows with [Invalid argument], while passing on
       POSIX where only '/' is special). Replaced on every platform rather than under [Sys.win32],
       so that a debug artifact has one name everywhere and a rule naming it cannot be
       platform-dependent. *)
    String.map
      ~f:(fun c ->
        if
          List.exists ~f:(equal_char c) [ '/'; '\\'; ':'; '<'; '>'; '"'; '|'; '?'; '*' ]
          || Char.to_int c < 0x20
        then '-'
        else c)
      fname
  in
  (* Reject bare "."/".." (and the empty string): otherwise filename_concat "build_files" cleaned
     can resolve to build_files/.. and the startup cleanup (clean_up_build_files_on_startup=true)
     would recursively delete the parent directory. *)
  match fname with
  | "" | "." | ".." -> "_"
  | _ -> fname

(* Concurrently running programs (e.g. dune tests) share the working directory, so a flat
   [build_files/] (or [log_files/]) would race on same-named artifacts. Unless overridden by
   [build_files_prefix], every process gets its own subdirectory derived from the executable name;
   the sentinel value "." restores the flat legacy layout. *)
let artifacts_subdir () =
  match get_global_arg ~default:"" ~arg_name:"build_files_prefix" with
  | "" ->
      Some
        (clean_filename @@ Stdlib.Filename.remove_extension
        @@ Stdlib.Filename.basename Stdlib.Sys.executable_name)
  | "." -> None
  | prefix -> Some (clean_filename prefix)

(* Tolerates mkdir races between concurrently running tests. *)
let ensure_artifacts_dir base subdir =
  let dir = match subdir with None -> base | Some p -> filename_concat base p in
  (try assert (Stdlib.Sys.is_directory dir)
   with Stdlib.Sys_error _ | Assert_failure _ -> (
     (try assert (Stdlib.Sys.is_directory base)
      with Stdlib.Sys_error _ | Assert_failure _ -> (
        try Stdlib.Sys.mkdir base 0o777 with Stdlib.Sys_error _ -> ()));
     if Option.is_some subdir then try Stdlib.Sys.mkdir dir 0o777 with Stdlib.Sys_error _ -> ()));
  dir

(** The directory generated-code debug files are written to (created if missing):
    [build_files/<prefix>/], where the prefix defaults to the executable's base name. *)
let build_files_dir () = ensure_artifacts_dir "build_files" (artifacts_subdir ())

let build_file fname = filename_concat (build_files_dir ()) @@ clean_filename fname

(** The directory diagnostic and routine-debug logs are written to (created if missing):
    [log_files/<prefix>/], sharing the prefix resolution of {!build_files_dir}. *)
let log_files_dir () = ensure_artifacts_dir "log_files" (artifacts_subdir ())

let diagn_log_file fname = filename_concat (log_files_dir ()) @@ clean_filename fname

let () =
  (* Cleanup needs to happen before get_local_debug_runtime (or any other code is run). *)
  (* Use Unix.lstat to distinguish symlinks from real directories: Sys.is_directory
     follows symlinks, so a symlink inside build_files/log_files pointing outside
     the tree would cause recursion to delete unrelated files. We unlink the link
     itself instead of descending. *)
  let lstat_kind path = try Some (Unix.lstat path).st_kind with Unix.Unix_error _ -> None in
  let rec remove_dir_if_exists dirname =
    match lstat_kind dirname with
    | None -> ()
    | Some Unix.S_DIR -> (
        try
          Array.iter (Stdlib.Sys.readdir dirname) ~f:(fun fname ->
              let path = Stdlib.Filename.concat dirname fname in
              match lstat_kind path with
              | Some Unix.S_DIR -> remove_dir_if_exists path
              | Some _ | None -> (
                  (* Regular file, symlink (to file or dir), socket, etc.: unlink the entry, do not
                     follow. *)
                  try Stdlib.Sys.remove path with Stdlib.Sys_error _ -> ()));
          Stdlib.Sys.rmdir dirname
        with exn ->
          Stdio.eprintf "Failed to delete directory %s: %s\n%!" dirname (Exn.to_string exn))
    | Some _ -> (
        (* Symlink (even to a dir), regular file, etc.: unlink the entry. *)
        try Stdlib.Sys.remove dirname
        with exn ->
          Stdio.eprintf "Failed to delete %s (expected a directory): %s\n%!" dirname
            (Exn.to_string exn))
  in
  (* Cleanup is scoped to this process's own subdirectory (see [artifacts_subdir]), so a starting
     test cannot delete the in-flight artifacts of a concurrently running one. If the artifact root
     itself is a symlink, skip the scoped cleanup rather than follow it: appending the subdirectory
     would resolve through the link, deleting files outside the working tree. *)
  let remove_scoped_dir base =
    match artifacts_subdir () with
    | None -> remove_dir_if_exists base
    | Some p -> (
        match lstat_kind base with
        | Some Unix.S_DIR -> remove_dir_if_exists (filename_concat base p)
        | Some _ | None -> ())
  in
  let clean_up_log_files_on_startup =
    get_global_flag ~default:true ~arg_name:"clean_up_log_files_on_startup"
  in
  if clean_up_log_files_on_startup then remove_scoped_dir "log_files";
  let clean_up_build_files_on_startup =
    get_global_flag ~default:true ~arg_name:"clean_up_build_files_on_startup"
  in
  if clean_up_build_files_on_startup then remove_scoped_dir "build_files"

let get_local_debug_runtime =
  let snapshot_every_sec =
    Option.join
    @@ str_nonempty ~f:Float.of_string_opt
    @@ get_global_arg ~default:"" ~arg_name:"snapshot_every_sec"
  in
  let time_tagged =
    match String.lowercase @@ get_global_arg ~default:"elapsed" ~arg_name:"time_tagged" with
    | "not_tagged" -> Minidebug_runtime.Not_tagged
    | "clock" -> Clock
    | "elapsed" -> Elapsed
    | s -> invalid_arg @@ "ocannl_time_tagged setting should be none, clock or elapsed; found: " ^ s
  in
  let elapsed_times =
    match String.lowercase @@ get_global_arg ~default:"not_reported" ~arg_name:"elapsed_times" with
    | "not_reported" -> Minidebug_runtime.Not_reported
    | "seconds" -> Seconds
    | "milliseconds" -> Milliseconds
    | "microseconds" -> Microseconds
    | "nanoseconds" -> Nanoseconds
    | s ->
        invalid_arg
        @@ "ocannl_elapsed_times setting should be not_reported, seconds or milliseconds, \
            microseconds or nanoseconds; found: " ^ s
  in
  let location_format =
    match String.lowercase @@ get_global_arg ~default:"beg_pos" ~arg_name:"location_format" with
    | "no_location" -> Minidebug_runtime.No_location
    | "file_only" -> File_only
    | "beg_line" -> Beg_line
    | "beg_pos" -> Beg_pos
    | "range_line" -> Range_line
    | "range_pos" -> Range_pos
    | s ->
        invalid_arg
        @@ "ocannl_location_format setting should be one of: no_location, file_only, beg_line, \
            beg_pos, range_line, range_pos; found: " ^ s
  in
  let backend, toc_flame_graph, printbox_backend =
    match
      String.lowercase @@ String.strip @@ get_global_arg ~default:"db" ~arg_name:"debug_backend"
    with
    | "text" -> (`Printbox, false, `Text)
    | "html" -> (`Printbox, true, `Html Minidebug_runtime.default_html_config)
    | "markdown" -> (`Printbox, false, `Markdown Minidebug_runtime.default_md_config)
    | "flushing" -> (`Flushing, true, `Text)
    | "db" -> (`Db, false, `Text)
    | s ->
        invalid_arg
        @@ "ocannl_debug_backend setting should be text, html, markdown, flushing or db; found: "
        ^ s
  in
  let hyperlink = get_global_arg ~default:"./" ~arg_name:"hyperlink_prefix" in
  let print_scope_ids = get_global_flag ~default:false ~arg_name:"logs_print_scope_ids" in
  let verbose_scope_ids = get_global_flag ~default:false ~arg_name:"logs_verbose_scope_ids" in
  let log_main_domain_to_stdout =
    get_global_flag ~default:false ~arg_name:"log_main_domain_to_stdout"
  in
  let file_stem =
    if log_main_domain_to_stdout then None
    else Some (get_global_arg ~default:"debug" ~arg_name:"log_file_stem")
  in
  let filename = Option.map file_stem ~f:diagn_log_file in
  let prev_run_file =
    let prefix = str_nonempty ~f:Fn.id @@ get_global_arg ~default:"" ~arg_name:"prev_run_prefix" in
    Option.map2 prefix file_stem ~f:(fun prefix stem -> diagn_log_file @@ prefix ^ stem ^ ".raw")
  in
  let toc_entry_minimal_depth =
    let arg = get_global_arg ~default:"" ~arg_name:"toc_entry_minimal_depth" in
    if String.is_empty arg then [] else [ Minidebug_runtime.Minimal_depth (Int.of_string arg) ]
  in
  let toc_entry_minimal_size =
    let arg = get_global_arg ~default:"" ~arg_name:"toc_entry_minimal_size" in
    if String.is_empty arg then [] else [ Minidebug_runtime.Minimal_size (Int.of_string arg) ]
  in
  let toc_entry_minimal_span =
    let arg = get_global_arg ~default:"" ~arg_name:"toc_entry_minimal_span" in
    if String.is_empty arg then []
    else
      let arg, period = (String.prefix arg (String.length arg - 2), String.suffix arg 2) in
      let period =
        match period with
        | "ns" -> Mtime.Span.ns
        | "us" -> Mtime.Span.us
        | "ms" -> Mtime.Span.ms
        | _ ->
            invalid_arg
            @@ "ocannl_toc_entry_minimal_span setting should end with one of: ns, us, ms; found: "
            ^ period
      in
      [ Minidebug_runtime.Minimal_span Mtime.Span.(Int.of_string arg * period) ]
  in
  let toc_entry =
    Minidebug_runtime.And (toc_entry_minimal_depth @ toc_entry_minimal_size @ toc_entry_minimal_span)
  in
  let debug_highlights =
    let arg = get_global_arg ~default:"" ~arg_name:"debug_highlights" in
    if String.is_empty arg then [] else String.split arg ~on:'|'
  in
  let highlight_re =
    let arg = get_global_arg ~default:"" ~arg_name:"debug_highlight_pcre" in
    Option.to_list @@ str_nonempty ~f:Re.Pcre.re arg
  in
  let highlight_terms = Re.(alt (highlight_re @ List.map debug_highlights ~f:str)) in
  let diff_ignore_pattern =
    str_nonempty ~f:Re.Pcre.re @@ get_global_arg ~default:"" ~arg_name:"diff_ignore_pattern_pcre"
  in
  let max_distance_factor =
    str_nonempty ~f:Int.of_string @@ get_global_arg ~default:"" ~arg_name:"diff_max_distance_factor"
  in
  let scope_id_pairs =
    let pairs_str = get_global_arg ~default:"" ~arg_name:"debug_scope_id_pairs" in
    if String.is_empty pairs_str then []
    else
      String.split pairs_str ~on:';'
      |> List.filter_map ~f:(fun pair_str ->
          match String.split pair_str ~on:',' with
          | [ id1; id2 ] ->
              Option.try_with (fun () ->
                  (Int.of_string (String.strip id1), Int.of_string (String.strip id2)))
          | _ -> None)
  in
  let truncate_children =
    let arg = get_global_arg ~default:"" ~arg_name:"debug_log_truncate_children" in
    if String.is_empty arg then None else Some (Int.of_string arg)
  in
  let prune_upto =
    let arg = get_global_arg ~default:"" ~arg_name:"debug_log_prune_upto" in
    if String.is_empty arg then None else Some (Int.of_string arg)
  in
  let name = get_global_arg ~default:"debug" ~arg_name:"log_file_stem" in
  match (backend, filename) with
  | `Flushing, None ->
      Minidebug_runtime.prefixed_runtime_flushing ~time_tagged ~elapsed_times ~print_scope_ids
        ~verbose_scope_ids ~global_prefix:name ~for_append:false ~log_level:original_log_level ()
  | `Flushing, Some filename ->
      Minidebug_runtime.local_runtime_flushing ~time_tagged ~elapsed_times ~print_scope_ids
        ~verbose_scope_ids ~global_prefix:name ~for_append:false ~log_level:original_log_level
        filename
  | `Printbox, None ->
      Minidebug_runtime.prefixed_runtime ~time_tagged ~elapsed_times ~location_format
        ~print_scope_ids ~verbose_scope_ids ~global_prefix:name ~toc_entry
        ~toc_specific_hyperlink:"" ~highlight_terms ?truncate_children
        ~exclude_on_path:Re.(str "env")
        ~log_level:original_log_level ?snapshot_every_sec ()
  | `Printbox, Some filename ->
      Minidebug_runtime.local_runtime ~time_tagged ~elapsed_times ~location_format ~print_scope_ids
        ~verbose_scope_ids ~global_prefix:name ~toc_flame_graph ~flame_graph_separation:50
        ~toc_entry ~for_append:false ~max_inline_sexp_length:120 ~hyperlink
        ~toc_specific_hyperlink:"" ~highlight_terms ?truncate_children
        ~exclude_on_path:Re.(str "env")
        ?prune_upto ~backend:printbox_backend ~log_level:original_log_level ?snapshot_every_sec
        ?prev_run_file ?diff_ignore_pattern ?max_distance_factor ~scope_id_pairs filename
  | `Db, _ ->
      let filename = Option.value ~default:(diagn_log_file name) filename in
      let db =
        Minidebug_db.debug_db_file ~time_tagged ~elapsed_times ~print_scope_ids ~verbose_scope_ids
          ~run_name:name ~log_level:original_log_level filename
      in
      fun () -> db

let _get_local_debug_runtime = get_local_debug_runtime

[%%global_debug_log_level 0]

(* export OCANNL_LOG_LEVEL_UTILS=9 to enable debugging into the log_files/ directory. *)
[%%global_debug_log_level_from_env_var "OCANNL_LOG_LEVEL_UTILS"]

(* [%%global_debug_interrupts { max_nesting_depth = 100; max_num_children = 1000 }] *)

let%diagn_sexp set_log_level level =
  settings.log_level <- level;
  [%log
    "Set log_level to",
    (Debug_runtime.log_level := level;
     level
      : int)]

let restore_settings () =
  set_log_level original_log_level;
  settings.debug_log_from_routines <-
    get_global_flag ~default:false ~arg_name:"debug_log_from_routines";
  settings.output_debug_files_in_build_directory <-
    get_global_flag ~default:false ~arg_name:"output_debug_files_in_build_directory";
  settings.fixed_state_for_init <-
    (let seed = get_global_arg ~arg_name:"fixed_state_for_init" ~default:"" in
     if String.is_empty seed then None else Some (Int.of_string seed));
  settings.print_decimals_precision <-
    Int.of_string @@ get_global_arg ~arg_name:"print_decimals_precision" ~default:"2";
  settings.check_half_prec_constants_cutoff <-
    Float.of_string_opt
    @@ get_global_arg ~arg_name:"check_half_prec_constants_cutoff" ~default:"16384.0";
  settings.default_prng_variant <- get_global_arg ~default:"light" ~arg_name:"default_prng_variant";
  (* [big_models] is the pre-gh-ocannl-344 name; read it as a fallback so existing configs / CLI /
     OCANNL_BIG_MODELS still enable 64-bit index and pool-offset widths when [large_models] is
     unset. *)
  settings.large_models <-
    get_global_flag
      ~default:(get_global_flag ~default:false ~arg_name:"big_models")
      ~arg_name:"large_models"

let () = restore_settings ()

let () =
  let ocannl_prefixes = [ "--ocannl_"; "--ocannl-"; "-ocannl_"; "-ocannl-" ] in
  Array.iter Stdlib.Sys.argv ~f:(fun arg ->
      match List.find ocannl_prefixes ~f:(fun p -> String.is_prefix ~prefix:p arg) with
      | None -> ()
      | Some prefix ->
          let rest = String.drop_prefix arg (String.length prefix) in
          let raw_key = match String.lsplit2 rest ~on:'=' with Some (k, _) -> k | None -> rest in
          let key = String.tr ~target:'-' ~replacement:'_' @@ String.lowercase raw_key in
          if not (Set.mem known_config_keys key) then
            Stdio.eprintf "OCANNL warning: unknown commandline argument %S\n%!" arg)

let with_runtime_debug () = settings.output_debug_files_in_build_directory && settings.log_level > 1
let debug_log_from_routines () = settings.debug_log_from_routines && settings.log_level > 1
let never_capture_stdout () = get_global_flag ~default:false ~arg_name:"never_capture_stdout"

let enable_runtime_debug () =
  settings.output_debug_files_in_build_directory <- true;
  set_log_level @@ max 2 settings.log_level

let rec union_find ~equal map ~key ~rank =
  match Map.find map key with
  | None -> (key, rank)
  | Some data ->
      if equal key data then (key, rank) else union_find ~equal map ~key:data ~rank:(rank + 1)

let union_add ~equal map k1 k2 =
  if equal k1 k2 then map
  else
    let root1, rank1 = union_find ~equal map ~key:k1 ~rank:0
    and root2, rank2 = union_find ~equal map ~key:k2 ~rank:0 in
    if rank1 < rank2 then Map.update map root1 ~f:(fun _ -> root2)
    else Map.update map root2 ~f:(fun _ -> root1)

(** Filters the list keeping the first occurrence of each element. *)
let unique_keep_first ~equal l =
  let rec loop acc = function
    | [] -> List.rev acc
    | hd :: tl -> if List.mem acc hd ~equal then loop acc tl else loop (hd :: acc) tl
  in
  loop [] l

(** Returns the multiset difference of [l1] and [l2], where [l1] and [l2] must be sorted in
    increasing order. *)
let sorted_diff ~compare l1 l2 =
  let rec loop acc l1 l2 =
    match (l1, l2) with
    | [], _ -> List.rev acc
    | l1, [] -> List.rev_append acc l1
    | h1 :: t1, h2 :: t2 -> (
        match compare h1 h2 with
        | c when c < 0 -> loop (h1 :: acc) t1 l2
        | 0 ->
            (* Depending on this line this can be either a set diff or a multiset: currently it's
               multiset diff. *)
            loop acc t1 t2
        | _ -> loop acc l1 t2)
  in
  (loop [] l1 l2 [@nontail])

(** Removes the first occurrence of an element from the list that is equal to the given element. *)
let remove_elem ~equal elem l =
  let rec loop acc = function
    | [] -> List.rev acc
    | hd :: tl -> if equal elem hd then List.rev_append acc tl else loop (hd :: acc) tl
  in
  loop [] l

(** [parallel_merge merge num_devices] progressively invokes the pairwise [merge] callback,
    converging on the 0th position, with [from] ranging from [1] to [num_devices - 1], and
    [to_ < from]. *)
let parallel_merge merge (num_devices : int) =
  let rec loop (upper : int) : unit =
    let is_even = (upper + 1) % 2 = 0 in
    let lower = if is_even then 0 else 1 in
    let half : int = (upper - (lower - 1)) / 2 in
    if half > 0 then (
      let midpoint : int = half + lower - 1 in
      for i = lower to midpoint do
        (* Maximal [from] is [2 * half + lower - 1 = upper]. *)
        merge ~from:(half + i) ~to_:i
      done;
      loop midpoint)
  in
  loop (num_devices - 1)

let ( !@ ) = Atomic.get

type atomic_bool = bool Atomic.t

let sexp_of_atomic_bool flag = sexp_of_bool @@ Atomic.get flag

type atomic_int = int Atomic.t

let sexp_of_atomic_int flag = sexp_of_int @@ Atomic.get flag

let sexp_append ~elem = function
  | Sexp.List l -> Sexp.List (elem :: l)
  | Sexp.Atom _ as e2 -> Sexp.List [ elem; e2 ]

let sexp_mem ~elem = function
  | Sexp.Atom _ as e2 -> Sexp.equal elem e2
  | Sexp.List l -> Sexp.(List.mem ~equal l elem)

let rec sexp_deep_mem ~elem = function
  | Sexp.Atom _ as e2 -> Sexp.equal elem e2
  | Sexp.List l -> Sexp.(List.mem ~equal l elem) || List.exists ~f:(sexp_deep_mem ~elem) l

let split_with_seps sep s =
  let tokens = Re.split_full sep s in
  List.map tokens ~f:(function `Text tok -> tok | `Delim sep -> Re.Group.get sep 0)

module Lazy = struct
  include Lazy

  let sexp_of_t = Minidebug_runtime.sexp_of_lazy_t
  let sexp_of_lazy_t = Minidebug_runtime.sexp_of_lazy_t
end

type requirement =
  | Skip
  | Required
  | Optional of { callback_if_missing : unit -> unit [@sexp.opaque] [@compare.ignore] }
[@@deriving compare, sexp]

let default_indent = ref 2

let doc_of_sexp sexp =
  let open Sexp in
  let open Int in
  let module Bytes = Stdlib.Bytes in
  let must_escape str =
    let len = String.length str in
    len = 0
    ||
    let rec loop str ix =
      match str.[ix] with
      | '"' | '(' | ')' | ';' | '\\' -> true
      | '|' ->
          ix > 0
          &&
          let next = ix - 1 in
          Char.equal str.[next] '#' || loop str next
      | '#' ->
          ix > 0
          &&
          let next = ix - 1 in
          Char.equal str.[next] '|' || loop str next
      | '\000' .. '\032' | '\127' .. '\255' -> true
      | _ -> ix > 0 && loop str (ix - 1)
    in
    loop str (len - 1)
  in

  let escaped s =
    let n = ref 0 in
    for i = 0 to String.length s - 1 do
      n :=
        !n
        +
        match String.unsafe_get s i with
        | '\"' | '\\' | '\n' | '\t' | '\r' | '\b' -> 2
        | ' ' .. '~' -> 1
        | _ -> 4
    done;
    if !n = String.length s then s
    else
      let s' = Bytes.create !n in
      n := 0;
      for i = 0 to String.length s - 1 do
        (match String.unsafe_get s i with
        | ('\"' | '\\') as c ->
            Bytes.unsafe_set s' !n '\\';
            incr n;
            Bytes.unsafe_set s' !n c
        | '\n' ->
            Bytes.unsafe_set s' !n '\\';
            incr n;
            Bytes.unsafe_set s' !n 'n'
        | '\t' ->
            Bytes.unsafe_set s' !n '\\';
            incr n;
            Bytes.unsafe_set s' !n 't'
        | '\r' ->
            Bytes.unsafe_set s' !n '\\';
            incr n;
            Bytes.unsafe_set s' !n 'r'
        | '\b' ->
            Bytes.unsafe_set s' !n '\\';
            incr n;
            Bytes.unsafe_set s' !n 'b'
        | ' ' .. '~' as c -> Bytes.unsafe_set s' !n c
        | c ->
            let a = Stdlib.Char.code c in
            Bytes.unsafe_set s' !n '\\';
            incr n;
            Bytes.unsafe_set s' !n (Stdlib.Char.chr (48 + (a / 100)));
            incr n;
            Bytes.unsafe_set s' !n (Stdlib.Char.chr (48 + (a / 10 % 10)));
            incr n;
            Bytes.unsafe_set s' !n (Stdlib.Char.chr (48 + (a % 10))));
        incr n
      done;
      Bytes.unsafe_to_string s'
  in

  let esc_str str =
    let estr = escaped str in
    let elen = String.length estr in
    let res = Bytes.create (elen + 2) in
    Bytes.blit_string estr 0 res 1 elen;
    Bytes.unsafe_set res 0 '"';
    Bytes.unsafe_set res (elen + 1) '"';
    Bytes.unsafe_to_string res
  in

  let index_of_newline str start = Stdlib.String.index_from_opt str start '\n' in

  let get_substring str index end_pos_opt =
    let end_pos = match end_pos_opt with None -> String.length str | Some end_pos -> end_pos in
    String.sub str ~pos:index ~len:(end_pos - index)
  in

  let is_one_line str =
    match index_of_newline str 0 with
    | None -> true
    | Some index -> Int.(index + 1 = String.length str)
  in

  let open PPrint in
  let doc_maybe_esc_str str =
    if not (must_escape str) then string str
    else if is_one_line str then string (esc_str str)
    else
      let rec loop index acc =
        let next_newline = index_of_newline str index in
        let next_line = get_substring str index next_newline in
        let acc = acc ^^ string (escaped next_line) in
        match next_newline with
        | None -> acc
        | Some newline_index ->
            loop (newline_index + 1) (acc ^^ string "\\" ^^ hardline ^^ string "\\n")
      in
      (* the leading space is to line up the lines *)
      string " \"" ^^ loop 0 empty ^^ string "\""
  in

  let rec doc_of_sexp_indent indent = function
    | Atom str -> doc_maybe_esc_str str
    | List (h :: t) ->
        group (string "(" ^^ nest indent (doc_of_sexp_indent indent h ^^ doc_of_sexp_rest indent t))
    | List [] -> string "()"
  and doc_of_sexp_rest indent = function
    | h :: t -> space ^^ doc_of_sexp_indent indent h ^^ doc_of_sexp_rest indent t
    | [] -> string ")"
  in

  doc_of_sexp_indent !default_indent sexp

let output_to_build_file ~fname =
  if settings.output_debug_files_in_build_directory then
    let f = Stdio.Out_channel.create @@ build_file fname in
    let print doc =
      PPrint.ToChannel.pretty 0.7 100 f doc;
      Stdio.Out_channel.flush f
    in
    Some print
  else None

let get_debug_output_channel ~fname =
  if settings.output_debug_files_in_build_directory then
    Some (Stdio.Out_channel.create @@ build_file fname)
  else None

exception User_error of string

let header_sep =
  let open Re in
  compile (seq [ str " "; opt any; str "="; str " " ])

let%diagn_sexp log_trace_tree _logs =
  [%log_block
    "trace tree";
    let sep s = String.concat ~sep:"\n" @@ String.split ~on:'$' s in
    let rec loop = function
      | [] -> []
      | line :: more when String.is_empty line -> loop more
      | "COMMENT: end" :: more -> more
      | comment :: more when String.is_prefix comment ~prefix:"COMMENT: " ->
          let more =
            [%log_entry
              sep @@ String.chop_prefix_exn ~prefix:"COMMENT: " comment;
              loop more]
          in
          loop more
      | source :: trace :: more when String.is_prefix source ~prefix:"# " ->
          (let source = sep @@ String.chop_prefix_exn ~prefix:"# " source in
           match split_with_seps header_sep @@ sep trace with
           | [] | [ "" ] -> [%log source]
           | header1 :: assign1 :: header2 :: body ->
               let header = String.concat [ header1; assign1; header2 ] in
               let body = String.concat body in
               let _message = Sexp.(List [ Atom header; Atom source; Atom body ]) in
               [%log (_message : Sexp.t)]
           | _ -> [%log source, trace]);
          loop more
      | _line :: more ->
          [%log sep _line];
          loop more
    in
    let rec loop_logs logs =
      let output = loop logs in
      if not (List.is_empty output) then
        [%log_block
          "TRAILING LOGS:";
          loop_logs output]
    in
    loop_logs _logs]

include Datatypes
module Cpu_topology = Cpu_topology

type build_file_channel = { f_path : string; oc : Stdlib.out_channel; finalize : unit -> unit }

let open_build_file ~base_name ~extension : build_file_channel =
  let f_path =
    if settings.output_debug_files_in_build_directory then build_file @@ base_name ^ extension
    else Stdlib.Filename.temp_file (base_name ^ "_") extension
  in
  (* (try Stdlib.Sys.remove f_path with _ -> ()); *)
  let oc = Out_channel.open_text f_path in
  let finalize () =
    Stdio.Out_channel.flush oc;
    Stdio.Out_channel.close oc
  in
  { f_path; oc; finalize }

let captured_log_prefix = ref "!@#"

type captured_log_processor = { log_processor_prefix : string; process_logs : string list -> unit }

let captured_log_processors : captured_log_processor list ref = ref []

let add_log_processor ~prefix process_logs =
  captured_log_processors :=
    { log_processor_prefix = prefix; process_logs } :: !captured_log_processors

external input_scan_line : Stdlib.in_channel -> int = "caml_ml_input_scan_line"
external flush_c_streams : unit -> unit = "ocannl_flush_c_streams"

let input_line chan =
  let n = input_scan_line chan in
  if n = 0 then raise End_of_file;
  let line = Stdlib.really_input_string chan (abs n) in
  ( n > 0,
    String.chop_suffix_if_exists ~suffix:"\n" @@ String.chop_suffix_if_exists line ~suffix:"\r\n" )

let capture_stdout_logs arg =
  if never_capture_stdout () || not (debug_log_from_routines ()) then arg ()
  else (
    Stdlib.flush Stdlib.stdout;
    flush_c_streams ();
    (* Ensure previous stdout is flushed *)
    let original_stdout_fd = Unix.dup Unix.stdout in

    let pipe_read_fd, pipe_write_fd = Unix.pipe ~cloexec:true () in
    Unix.dup2 pipe_write_fd Unix.stdout;

    (* pipe_write_fd is now the new Stdlib.stdout, do not close it in parent until done. *)
    (* The reader domain will close pipe_read_fd. *)
    let collected_logs_ref = ref [] in
    let reader_domain_failed = Atomic.make false in

    let reader_domain_logic () =
      let in_channel = Unix.in_channel_of_descr pipe_read_fd in
      (* Create an output channel to the original stdout for immediate passthrough *)
      let orig_out = Unix.out_channel_of_descr (Unix.dup original_stdout_fd) in
      try
        while true do
          let _is_endlined, line = input_line in_channel in
          match String.chop_prefix ~prefix:!captured_log_prefix line with
          | Some logline -> collected_logs_ref := logline :: !collected_logs_ref
          | None ->
              (* Forward non-log lines to original stdout immediately *)
              Stdlib.output_string orig_out (line ^ "\n");
              Stdlib.flush orig_out
        done;
        Stdlib.close_out_noerr orig_out;
        Stdlib.close_in_noerr in_channel (* This closes pipe_read_fd *)
      with
      | End_of_file -> () (* Normal termination of the reader *)
      | exn ->
          Stdlib.close_out_noerr orig_out;
          Atomic.set reader_domain_failed true;
          Stdio.eprintf "Exception in stdout reader domain: %s\\nBacktrace:\\n%s\\n%!"
            (Exn.to_string exn)
            (Stdlib.Printexc.get_backtrace ());
          Stdlib.close_in_noerr in_channel (* This closes pipe_read_fd *);
          Stdlib.Printexc.raise_with_backtrace exn (Stdlib.Printexc.get_raw_backtrace ())
    in

    let reader_domain = Domain.spawn reader_domain_logic in

    let result =
      try arg ()
      with exn ->
        (* Ensure cleanup even if arg() fails *)
        Stdlib.flush Stdlib.stdout;
        flush_c_streams ();
        (* Flush to pipe_write_fd *)
        Unix.close pipe_write_fd;
        (* Signal EOF to reader domain *)
        (* Restore stdout before waiting for the reader domain so that the write end of the pipe is
           effectively closed (both the explicit [pipe_write_fd] descriptor above _and_ the
           descriptor 1 obtained via [dup2] earlier). Otherwise the reader domain would never see an
           EOF and [Domain.join] would block indefinitely. *)
        Unix.dup2 original_stdout_fd Unix.stdout;
        (* Restore stdout *)
        Unix.close original_stdout_fd;

        (* Now that all write descriptors for the pipe are closed, we can wait for the reader domain
           to finish. *)
        (try Domain.join reader_domain
         with e ->
           Stdio.eprintf "Exception while joining reader domain (arg failed): %s\\n%!"
             (Exn.to_string e));

        (if not (Atomic.get reader_domain_failed) then
           let captured_output = List.rev !collected_logs_ref in
           List.iter (List.rev !captured_log_processors)
             ~f:(fun { log_processor_prefix; process_logs } ->
               process_logs
               @@ List.filter_map captured_output
                    ~f:(String.chop_prefix ~prefix:log_processor_prefix)));
        captured_log_processors := [];
        (* Clear processors *)
        Stdlib.Printexc.raise_with_backtrace exn (Stdlib.Printexc.get_raw_backtrace ())
    in

    (* Normal path: arg() completed successfully *)
    Stdlib.flush Stdlib.stdout;
    flush_c_streams ();
    (* Flush to pipe_write_fd *)
    Unix.close pipe_write_fd;

    (* Signal EOF to reader domain *)

    (* Restore stdout before waiting for the reader domain so that the write end of the pipe is
       effectively closed and the reader can finish properly. *)
    Unix.dup2 original_stdout_fd Unix.stdout;
    (* Restore stdout *)
    Unix.close original_stdout_fd;

    (try Domain.join reader_domain
     with e ->
       Stdio.eprintf "Exception while joining reader domain (arg succeeded): %s\\n%!"
         (Exn.to_string e);
       if Atomic.get reader_domain_failed then
         Stdlib.Printexc.raise_with_backtrace e (Stdlib.Printexc.get_raw_backtrace ()));

    if not (Atomic.get reader_domain_failed) then
      let captured_output = List.rev !collected_logs_ref in
      Exn.protect
        ~f:(fun () ->
          (* Process captured logs by processors first. *)
          List.iter (List.rev !captured_log_processors)
            ~f:(fun { log_processor_prefix; process_logs } ->
              process_logs
              @@ List.filter_map captured_output
                   ~f:(String.chop_prefix ~prefix:log_processor_prefix)))
        ~finally:(fun () -> captured_log_processors := [])
    else captured_log_processors := [] (* Clear processors if reader failed *);
    result)

let log_debug_routine_logs ~log_contents ~stream_name =
  if get_global_flag ~default:false ~arg_name:"debug_log_to_stream_files" then
    let stream_file_name = diagn_log_file @@ stream_name ^ ".log" in
    Stdio.Out_channel.with_file stream_file_name ~append:true ~f:(fun oc ->
        List.iter log_contents ~f:(fun line -> Stdio.Out_channel.output_line oc line))
  else log_trace_tree log_contents

let log_debug_routine_file ~log_file_name ~stream_name =
  let log_contents = Stdio.In_channel.read_lines log_file_name in
  log_debug_routine_logs ~log_contents ~stream_name;
  Stdlib.Sys.remove log_file_name

let gcd a b =
  let rec loop a b = if b = 0 then a else loop b (a % b) in
  loop (abs a) (abs b)
