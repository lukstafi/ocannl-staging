(* Configuration names used outside OCaml stay tied to [Utils.known_config_keys].

   The source forms are deliberately narrow and syntactic. Shell and Python scripts contribute every
   qualified command-line token accepted through [Utils.cmdline_var_prefixes], and every
   [OCANNL_<KEY>=] token, including tokens in comments and quoted command strings -- those are still
   instructions or commands a reader can reuse. Markdown contributes an inline code span whose whole
   rendered content is one assignment in a bare or prefixed form. Fenced code and longer snippets
   describe arbitrary APIs and expression languages, so treating every equals sign in them as
   configuration would make the scan unusable.

   gh-ocannl-790. *)

open Base
open Stdio
module Markdown = Test_utils.Agent_notes_scan

type kind =
  | Cli_flag
  | Prefix_free_cli_flag
  | Environment_assignment
  | Markdown_assignment
  | Config_file_assignment

type occurrence = {
  path : string;
  line : int;
  key : string;
  spelling : string;
  kind : kind;
  spaced_bare : bool;
}

let lowercase_key_char c = Char.is_lowercase c || Char.is_digit c || Char.equal c '_'
let uppercase_key_char c = Char.is_uppercase c || Char.is_digit c || Char.equal c '_'
let normalize_key key = Option.value (String.chop_prefix key ~prefix:"ocannl_") ~default:key

(* [OCANNL_TOOL_*] is the repository's explicit tool-private namespace, and
   [OCANNL_LOG_LEVEL_<MODULE>] is ppx_minidebug's compile-time tracing-gate namespace. Neither is
   OCANNL runtime configuration, and both are open vocabularies by design. *)
let non_config_env_key key =
  String.is_prefix key ~prefix:"TOOL_" || String.is_prefix key ~prefix:"LOG_LEVEL_"

(* The command-line spelling grammar belongs to the runtime reader. Derive its qualified prefixes
   with a sentinel key, then ask the same function whether each observed name is one it accepts.
   This still finds a REMOVED key: only the grammar comes from the registry owner, never the key
   population. *)
let cli_name_prefixes =
  let sentinel = "configusagescankey" in
  Utils.cmdline_var_names ~qualified_only:true sentinel
  |> List.filter_map ~f:(fun name ->
      let suffix =
        if String.equal name (String.uppercase name) then String.uppercase sentinel else sentinel
      in
      String.chop_suffix name ~suffix)
  |> List.dedup_and_sort ~compare:String.compare

let cli_key_char c = Char.is_alpha c || Char.is_digit c || Char.equal c '_' || Char.equal c '-'

let cli_token_char c =
  cli_key_char c || List.mem [ '='; '.'; '/'; ':'; '+'; '$'; '{'; '}' ] c ~equal:Char.equal

let syntactic_cli_key_of_name name =
  List.find_map cli_name_prefixes ~f:(fun name_prefix ->
      Option.bind (String.chop_prefix name ~prefix:name_prefix) ~f:(fun raw_key ->
          if String.is_empty raw_key || not (String.for_all raw_key ~f:cli_key_char) then None
          else
            let key = String.lowercase raw_key |> String.tr ~target:'-' ~replacement:'_' in
            Option.some_if
              (List.mem (Utils.cmdline_var_names ~qualified_only:true key) name ~equal:String.equal)
              key))

let known_cli_key token =
  Set.to_list Utils.known_config_keys
  |> List.filter ~f:(fun key ->
      List.exists (Utils.cmdline_var_prefixes ~qualified_only:true key) ~f:(fun prefix ->
          String.is_prefix token ~prefix))
  (* The runtime accepts prefix-overlapping keys too. Attribute the token to the most specific one;
     existence is the property this scan needs, and a removed token with no surviving parse still
     reaches the unknown branch below. *)
  |> List.max_elt ~compare:(fun a b -> Int.compare (String.length a) (String.length b))

let unknown_cli_key ~name_prefix token =
  let rest = String.drop_prefix token (String.length name_prefix) in
  let stop = ref 0 in
  while !stop < String.length rest && cli_key_char rest.[!stop] do
    Int.incr stop
  done;
  String.prefix rest !stop |> String.lowercase |> String.tr ~target:'-' ~replacement:'_'

let cli_key_of_token token =
  List.find_map cli_name_prefixes ~f:(fun name_prefix ->
      Option.bind (String.chop_prefix token ~prefix:name_prefix) ~f:(fun rest ->
          if String.is_empty rest || not (String.for_all rest ~f:cli_token_char) then None
          else
            match String.lsplit2 token ~on:'=' with
            | Some (name, _) -> syntactic_cli_key_of_name name
            | None ->
                Some
                  (Option.value (known_cli_key token) ~default:(unknown_cli_key ~name_prefix token))))

(* Prefix-free flags belong to the host application's namespace, so they cannot be discovered
   globally without claiming flags such as [--profile=prod]. Counted site judgments identify the
   places that deliberately document OCANNL's supported prefix-free form. The spelling grammar still
   comes from the runtime helper, and the key remains literal here so removing it from the registry
   makes the old occurrence fail. *)
let prefix_free_config_mentions =
  [
    ("docs/agent-notes/conventions.md", "backend", 1);
    ("docs/proposals/gh-ocannl-409.md", "backend", 3);
  ]

let prefix_free_names key =
  let qualified = Utils.cmdline_var_names ~qualified_only:true key in
  Utils.cmdline_var_names key
  |> List.filter ~f:(fun name -> not (List.mem qualified name ~equal:String.equal))

let prefix_free_occurrences_for ~path ~keys content =
  String.split_lines content
  |> List.mapi ~f:(fun index line ->
      List.concat_map keys ~f:(fun key ->
          List.concat_map (prefix_free_names key) ~f:(fun name ->
              let rec from pos found =
                match String.substr_index line ~pos ~pattern:name with
                | None -> List.rev found
                | Some start
                  when start > 0
                       && (Char.is_alphanum line.[start - 1]
                          || Char.equal line.[start - 1] '_'
                          || Char.equal line.[start - 1] '-') ->
                    from (start + 1) found
                | Some start ->
                    let stop = ref (start + String.length name) in
                    while !stop < String.length line && cli_token_char line.[!stop] do
                      Int.incr stop
                    done;
                    let spelling = String.sub line ~pos:start ~len:(!stop - start) in
                    from
                      (max (start + 1) !stop)
                      ({
                         path;
                         line = index + 1;
                         key;
                         spelling;
                         kind = Prefix_free_cli_flag;
                         spaced_bare = false;
                       }
                      :: found)
              in
              from 0 [])))
  |> List.concat

let tracked_prefix_free_keys path =
  List.filter_map prefix_free_config_mentions ~f:(fun (tracked_path, key, _) ->
      Option.some_if (String.equal path tracked_path) key)
  |> List.dedup_and_sort ~compare:String.compare

let prefixed_occurrences ?(start_ok = fun _ _ -> true) ~path ~prefix ~key_char ~normalize ~kind
    content =
  String.split_lines content
  |> List.mapi ~f:(fun index line ->
      let rec from pos found =
        match String.substr_index line ~pos ~pattern:prefix with
        | None -> List.rev found
        | Some start when not (start_ok line start) -> from (start + 1) found
        | Some start ->
            let key_start = start + String.length prefix in
            let key_stop = ref key_start in
            while !key_stop < String.length line && key_char line.[!key_stop] do
              Int.incr key_stop
            done;
            let next = max (start + 1) !key_stop in
            if
              !key_stop > key_start
              && !key_stop < String.length line
              && Char.equal line.[!key_stop] '='
            then
              let raw_key = String.sub line ~pos:key_start ~len:(!key_stop - key_start) in
              let spelling = String.sub line ~pos:start ~len:(!key_stop - start + 1) in
              match normalize raw_key with
              | None -> from next found
              | Some key ->
                  from next
                    ({ path; line = index + 1; key; spelling; kind; spaced_bare = false } :: found)
            else from next found
      in
      from 0 [])
  |> List.concat

let script_occurrences ~path content =
  let cli_occurrences =
    String.split_lines content
    |> List.mapi ~f:(fun index line ->
        List.concat_map cli_name_prefixes ~f:(fun name_prefix ->
            let rec from pos found =
              match String.substr_index line ~pos ~pattern:name_prefix with
              | None -> List.rev found
              | Some start
                when start > 0
                     && (Char.is_alphanum line.[start - 1]
                        || Char.equal line.[start - 1] '_'
                        || Char.equal line.[start - 1] '-') ->
                  from (start + 1) found
              | Some start -> (
                  let stop = ref (start + String.length name_prefix) in
                  while !stop < String.length line && cli_token_char line.[!stop] do
                    Int.incr stop
                  done;
                  let next = max (start + 1) !stop in
                  if !stop = start + String.length name_prefix then from next found
                  else
                    let spelling = String.sub line ~pos:start ~len:(!stop - start) in
                    match cli_key_of_token spelling with
                    | None -> from next found
                    | Some key ->
                        from next
                          ({
                             path;
                             line = index + 1;
                             key;
                             spelling;
                             kind = Cli_flag;
                             spaced_bare = false;
                           }
                          :: found))
            in
            from 0 []))
    |> List.concat
  in
  cli_occurrences
  @ prefixed_occurrences
      ~start_ok:(fun line start ->
        start = 0 || not (Char.is_alphanum line.[start - 1] || Char.equal line.[start - 1] '_'))
      ~path ~prefix:"OCANNL_" ~key_char:uppercase_key_char
      ~normalize:(fun key -> if non_config_env_key key then None else Some (String.lowercase key))
      ~kind:Environment_assignment content

let same_range (a, b) (x, y) = Int.equal a x && Int.equal b y

let assignment_key ~key_char ~normalize rendered =
  match String.lsplit2 rendered ~on:'=' with
  | Some (raw_key, value)
    when (not (String.is_empty raw_key))
         && key_char raw_key.[0]
         && String.for_all raw_key ~f:key_char
         && not (String.exists value ~f:Char.is_whitespace) ->
      normalize raw_key
  | _ -> None

(* These sites deliberately retain or exercise spellings known to be invalid. Counts stop a second
   obsolete instruction or test input in the same file from borrowing the intended exception. This
   list lives beside the spaced-form classifier so a spaced repetition is admitted to the scan and
   then changes its count, rather than disappearing as ordinary non-config prose. *)
let historical_invalid_config_mentions =
  [
    ("docs/agent-notes/conventions.md", "cc_parallel_grid_private_bytes_cap", 1);
    ("docs/agent-notes/conventions.md", "private_bytes_cap", 1);
    ("docs/proposals/gh-ocannl-409.md", "bacend", 1);
    ("docs/proposals/gh-ocannl-409.md", "output_debug_files_in_run_directory", 1);
    ("docs/proposals/gh-ocannl-409.md", "randomness_lib", 4);
    ("test/operations/dune", "backedn", 1);
    ("test/operations/dune", "not_a_real_key", 1);
    ("test/operations/startup_streams/ocannl_config", "definitely_not_a_config_key", 1);
  ]

let tracked_historical_config path key =
  List.exists historical_invalid_config_mentions ~f:(fun (tracked_path, tracked_key, _) ->
      String.equal path tracked_path && String.equal key tracked_key)

(* Whitespace makes bare assignments common in non-config prose. Pin every CURRENT config use of
   that form by file/key/count: a later key removal still scans the old site, while a newly added
   registered use must declare itself here. Prefixed CLI and environment forms are unambiguous and
   do not need this judgment list. *)
let spaced_config_mentions =
  [
    ( "docs/proposals/fix-inline-complex-computations-default-doc.md",
      "inline_complex_computations",
      1 );
    ("docs/proposals/gh-ocannl-344.md", "large_models", 2);
    ("docs/proposals/gh-ocannl-351.md", "inline_complex_computations", 2);
    ("docs/proposals/task-73617488.md", "virtualize_max_visits", 3);
  ]

let tracked_spaced_config path key =
  List.exists spaced_config_mentions ~f:(fun (tracked_path, tracked_key, _) ->
      String.equal path tracked_path && String.equal key tracked_key)

let control_fixture path = String.equal path "config_usage_scan_bogus.fixture"

let one_assignment ~path rendered =
  match String.lsplit2 rendered ~on:'=' with
  | Some (raw_name, raw_value) -> (
      let name = String.strip raw_name in
      let value = String.strip raw_value in
      let spaced = not (String.equal raw_name name && String.equal raw_value value) in
      if String.is_empty name || String.exists value ~f:Char.is_whitespace then None
      else
        match cli_key_of_token (name ^ "=" ^ value) with
        | Some key -> Some (key, false)
        | None -> (
            match String.chop_prefix name ~prefix:"OCANNL_" with
            | Some key
              when (not (String.is_empty key))
                   && String.for_all key ~f:uppercase_key_char
                   && not (non_config_env_key key) ->
                Some (String.lowercase key, false)
            | _ ->
                Option.bind
                  (assignment_key ~key_char:lowercase_key_char
                     ~normalize:(fun key -> Some (normalize_key key))
                     (name ^ "=" ^ value))
                  ~f:(fun key ->
                    if
                      (not spaced)
                      || Set.mem Utils.known_config_keys key
                      || tracked_spaced_config path key || control_fixture path
                      || tracked_historical_config path key
                    then
                      Some
                        ( key,
                          spaced
                          && (not (control_fixture path))
                          && not (tracked_historical_config path key) )
                    else None)))
  | None -> ( match cli_key_of_token rendered with Some key -> Some (key, false) | None -> None)

let markdown_occurrences ~allow_bare ~path content =
  let scan = Markdown.inert_by_line content in
  let comments line = Markdown.spans_at scan.comment_ranges line in
  let fences line = Markdown.spans_at scan.fence_ranges line in
  let as_markdown lineno occurrence =
    { occurrence with line = lineno; kind = Markdown_assignment; spaced_bare = false }
  in
  let lines = Markdown.lines content in
  let inline =
    lines
    |> List.concat_map ~f:(fun (lineno, line) ->
        Markdown.spans_at scan.ranges lineno
        |> List.filter ~f:(fun range ->
            (not (List.mem (comments lineno) range ~equal:same_range))
            && not (List.mem (fences lineno) range ~equal:same_range))
        |> List.concat_map ~f:(fun (start, stop) ->
            (* A multiline span arrives as one range per physical line. A complete [key=value]
               mention is one inline span, so both delimiters must be on this line. *)
            if stop <= start || not (Char.equal line.[start] '`' && Char.equal line.[stop - 1] '`')
            then []
            else
              let spelling = String.sub line ~pos:start ~len:(stop - start) in
              let rendered = Markdown.code_span_content spelling in
              let prefixed =
                script_occurrences ~path rendered |> List.map ~f:(as_markdown lineno)
              in
              if not (List.is_empty prefixed) then prefixed
              else if allow_bare then
                Option.to_list
                @@ Option.map (one_assignment ~path rendered) ~f:(fun (key, spaced_bare) ->
                    {
                      path;
                      line = lineno;
                      key;
                      spelling = rendered;
                      kind = Markdown_assignment;
                      spaced_bare;
                    })
              else []))
  in
  let fenced =
    lines
    |> List.concat_map ~f:(fun (lineno, line) ->
        if List.is_empty (fences lineno) then []
        else script_occurrences ~path line |> List.map ~f:(as_markdown lineno))
  in
  inline @ fenced

(* Keep this normalization beside [Utils.parse_config_lines]: config files accept case-insensitive
   keys, leading dashes/spaces, and an optional [ocannl_] prefix. The registry population remains
   independent so a removed normalized key still fails here. *)
let normalize_config_file_key raw_key =
  let key =
    String.lowercase @@ String.strip raw_key ~drop:(fun c -> Char.equal c '-' || Char.equal c ' ')
  in
  if String.is_prefix key ~prefix:"ocannl" then
    String.drop_prefix key 6 |> String.strip ~drop:(fun c -> Char.equal c '_')
  else key

let config_file_occurrences ~path content =
  String.split_lines content
  |> List.filter_mapi ~f:(fun index line ->
      let rendered = String.strip line in
      if
        String.is_empty rendered || String.is_prefix line ~prefix:"#"
        || String.is_prefix line ~prefix:"~~"
      then None
      else
        Option.bind (String.lsplit2 line ~on:'=') ~f:(fun (raw_key, value) ->
            let key = normalize_config_file_key raw_key in
            Option.some_if
              (not (String.is_empty key || String.is_empty value))
              {
                path;
                line = index + 1;
                key;
                spelling = String.strip raw_key ^ "=";
                kind = Config_file_assignment;
                spaced_bare = false;
              }))

(* These are assignments in other languages or report formats, not configuration. This is a judgment
   list rather than a restatement of a vocabulary owned elsewhere, and it is scoped by FILE as well
   as key: exempting [state] everywhere would let a future stale config mention called [state] pass
   in silence. Each entry pins its occurrence count: becoming real, disappearing, or gaining a
   second same-file use all fail. *)
let non_config_assignment_mentions =
  [
    (".claude/skills/slipshow/SKILL.md", "title", 1);
    ("README.md", "i", 1);
    ("benchmarks/README.md", "lr", 1);
    ("docs/agent-notes/build-and-test.md", "execution", 3);
    ("docs/agent-notes/scheduling-and-autotune.md", "max_chain", 1);
    ("docs/agent-notes/training-and-performance.md", "declines", 1);
    ("docs/agent-notes/training-and-performance.md", "state", 1);
    ("docs/agent-notes/training-and-performance.md", "timed", 1);
    ("docs/precision_inference.md", "top_down_prec", 6);
    ("docs/proposals/axis-labels.md", "hidden", 1);
    ("docs/proposals/axis-labels.md", "name", 4);
    ("docs/proposals/axis-labels.md", "rgb", 1);
    ("docs/proposals/concat-forward-component-data-propagation.md", "a", 2);
    ("docs/proposals/concat-forward-component-data-propagation.md", "a_mc", 1);
    ("docs/proposals/concat-forward-component-data-propagation.md", "b", 2);
    ("docs/proposals/concat-forward-component-data-propagation.md", "b_mc", 1);
    ("docs/proposals/concat-forward-component-data-propagation.md", "c", 2);
    ("docs/proposals/concat-forward-component-data-propagation.md", "c_mc", 1);
    ("docs/proposals/fix-centered-init-test-fallout.md", "epsilon", 1);
    ("docs/proposals/gh-ocannl-255.md", "d", 2);
    ("docs/proposals/gh-ocannl-263.md", "seq_q", 1);
    ("docs/proposals/gh-ocannl-308-comment.md", "n", 1);
    ("docs/proposals/gh-ocannl-420.md", "i", 1);
    ("docs/proposals/gh-ocannl-536.md", "private_seg_size", 1);
    ("docs/proposals/total-basis-bcast-if-1.md", "d", 1);
    ("docs/proposals/watch-ocannl-README-md-369aadb4.md", "batch", 2);
    ("docs/research/llmc-lessons.md", "accumulate", 1);
    ("docs/research/llmc-lessons.md", "beta", 1);
    ("docs/research/lean-attention-feasibility.md", "seq_q", 1);
    ("docs/syntax_extensions.md", "dilation", 2);
    ("docs/syntax_extensions.md", "kernel_size", 2);
    ("docs/syntax_extensions.md", "stride", 2);
  ]

let mention_site path key = path ^ "\000" ^ key

let non_config_assignment_sites =
  Set.of_list (module String)
  @@ List.map non_config_assignment_mentions ~f:(fun (path, key, _) -> mention_site path key)

let historical_invalid_config_sites =
  Set.of_list (module String)
  @@ List.map historical_invalid_config_mentions ~f:(fun (path, key, _) -> mention_site path key)

let spaced_config_sites =
  Set.of_list (module String)
  @@ List.map spaced_config_mentions ~f:(fun (path, key, _) -> mention_site path key)

let prefix_free_config_sites =
  Set.of_list (module String)
  @@ List.map prefix_free_config_mentions ~f:(fun (path, key, _) -> mention_site path key)

let kind_name = function
  | Cli_flag -> "command-line flag"
  | Prefix_free_cli_flag -> "prefix-free command-line flag"
  | Environment_assignment -> "environment assignment"
  | Markdown_assignment -> "documentation assignment"
  | Config_file_assignment -> "configuration-file assignment"

let check ~repository_census occurrences =
  let seen_non_config = Hashtbl.create (module String) in
  let seen_historical = Hashtbl.create (module String) in
  let seen_spaced_config = Hashtbl.create (module String) in
  let seen_prefix_free = Hashtbl.create (module String) in
  List.iter occurrences ~f:(fun occurrence ->
      (if occurrence.spaced_bare then
         let site = mention_site occurrence.path occurrence.key in
         if Set.mem spaced_config_sites site then Hashtbl.incr seen_spaced_config site
         else
           Verdict.fail
             (Printf.sprintf
                "%s:%d: spaced bare config mention `%s` lacks a file/key/count entry in \
                 spaced_config_mentions"
                occurrence.path occurrence.line occurrence.spelling));
      (match occurrence.kind with
      | Prefix_free_cli_flag ->
          let site = mention_site occurrence.path occurrence.key in
          if Set.mem prefix_free_config_sites site then Hashtbl.incr seen_prefix_free site
          else if repository_census then
            Verdict.fail
              (Printf.sprintf
                 "%s:%d: prefix-free config flag `%s` lacks a file/key/count entry in \
                  prefix_free_config_mentions"
                 occurrence.path occurrence.line occurrence.spelling)
      | _ -> ());
      if Set.mem Utils.known_config_keys occurrence.key then ()
      else if Set.mem non_config_assignment_sites (mention_site occurrence.path occurrence.key) then
        Hashtbl.incr seen_non_config (mention_site occurrence.path occurrence.key)
      else if Set.mem historical_invalid_config_sites (mention_site occurrence.path occurrence.key)
      then Hashtbl.incr seen_historical (mention_site occurrence.path occurrence.key)
      else
        Verdict.fail
          (Printf.sprintf "%s:%d: %s `%s` names `%s`, absent from Utils.known_config_keys"
             occurrence.path occurrence.line (kind_name occurrence.kind) occurrence.spelling
             occurrence.key));
  if repository_census then (
    let newly_real =
      List.filter non_config_assignment_mentions ~f:(fun (_, key, _) ->
          Set.mem Utils.known_config_keys key)
    in
    if not (List.is_empty newly_real) then
      Verdict.fail
        (Printf.sprintf
           "non-config assignment exemptions now name registered config keys -- remove: %s"
           (newly_real
           |> List.map ~f:(fun (path, key, _) -> path ^ ":" ^ key)
           |> String.concat ~sep:", "));
    let drifted_non_config =
      List.filter_map non_config_assignment_mentions ~f:(fun (path, key, expected) ->
          let actual =
            Hashtbl.find seen_non_config (mention_site path key) |> Option.value ~default:0
          in
          Option.some_if (not (Int.equal actual expected)) (path, key, expected, actual))
    in
    if not (List.is_empty drifted_non_config) then
      Verdict.fail
        (Printf.sprintf "non-config assignment exemption occurrence counts drifted: %s"
           (drifted_non_config
           |> List.map ~f:(fun (path, key, expected, actual) ->
               Printf.sprintf "%s:%s expected %d, saw %d" path key expected actual)
           |> String.concat ~sep:", "));
    let drifted_historical =
      List.filter_map historical_invalid_config_mentions ~f:(fun (path, key, expected) ->
          let actual =
            Hashtbl.find seen_historical (mention_site path key) |> Option.value ~default:0
          in
          Option.some_if (not (Int.equal actual expected)) (path, key, expected, actual))
    in
    if not (List.is_empty drifted_historical) then
      Verdict.fail
        (Printf.sprintf "historical invalid-config exemption occurrence counts drifted: %s"
           (drifted_historical
           |> List.map ~f:(fun (path, key, expected, actual) ->
               Printf.sprintf "%s:%s expected %d, saw %d" path key expected actual)
           |> String.concat ~sep:", "));
    let drifted_spaced =
      List.filter_map spaced_config_mentions ~f:(fun (path, key, expected) ->
          let actual =
            Hashtbl.find seen_spaced_config (mention_site path key) |> Option.value ~default:0
          in
          Option.some_if (not (Int.equal actual expected)) (path, key, expected, actual))
    in
    if not (List.is_empty drifted_spaced) then
      Verdict.fail
        (Printf.sprintf "spaced config-mention occurrence counts drifted: %s"
           (drifted_spaced
           |> List.map ~f:(fun (path, key, expected, actual) ->
               Printf.sprintf "%s:%s expected %d, saw %d" path key expected actual)
           |> String.concat ~sep:", "));
    let drifted_prefix_free =
      List.filter_map prefix_free_config_mentions ~f:(fun (path, key, expected) ->
          let actual =
            Hashtbl.find seen_prefix_free (mention_site path key) |> Option.value ~default:0
          in
          Option.some_if (not (Int.equal actual expected)) (path, key, expected, actual))
    in
    if not (List.is_empty drifted_prefix_free) then
      Verdict.fail
        (Printf.sprintf "prefix-free config-mention occurrence counts drifted: %s"
           (drifted_prefix_free
           |> List.map ~f:(fun (path, key, expected, actual) ->
               Printf.sprintf "%s:%s expected %d, saw %d" path key expected actual)
           |> String.concat ~sep:", ")))

let file_kind path =
  let basename = Stdlib.Filename.basename path in
  if String.is_suffix path ~suffix:".md" then `Markdown
  else if String.equal path "ocannl_config.reference" then `Reference
  else if String.equal basename "dune" then `Dune
  else if String.equal basename "ocannl_config" || String.equal basename "ocannl_config.for_debug"
  then `Config
  else `Script

let occurrences_of_file ~reported_path path =
  let content = In_channel.read_all path in
  let primary =
    match file_kind reported_path with
    | `Markdown ->
        let allow_bare =
          (not (String.is_prefix reported_path ~prefix:"benchmarks/"))
          || String.equal reported_path "benchmarks/README.md"
        in
        markdown_occurrences ~allow_bare ~path:reported_path content
    | `Config -> config_file_occurrences ~path:reported_path content
    | `Dune | `Reference | `Script -> script_occurrences ~path:reported_path content
  in
  primary
  @ prefix_free_occurrences_for ~path:reported_path
      ~keys:(tracked_prefix_free_keys reported_path)
      content

let fixture path config_path =
  let reported_path = Stdlib.Filename.basename path in
  let content = In_channel.read_all path in
  check ~repository_census:false
    (script_occurrences ~path:reported_path content
    @ markdown_occurrences ~allow_bare:true ~path:reported_path content
    @ prefix_free_occurrences_for ~path:reported_path
        ~keys:[ "definitely_not_a_prefix_free_config_key" ]
        content
    @ config_file_occurrences
        ~path:(Stdlib.Filename.basename config_path)
        (In_channel.read_all config_path))

let live workspace_root paths =
  let base = Test_utils.Dune_stanza_scan.base_dir workspace_root in
  let files =
    List.filter paths ~f:(fun path -> not (Stdlib.Sys.is_directory path))
    |> List.filter_map ~f:(fun path ->
        let relative = Test_utils.Dune_stanza_scan.repo_relative base path in
        if
          String.equal relative "AGENTS.md" || String.equal relative "README.md"
          || String.equal relative "ocannl_config.reference"
          || String.equal relative "ocannl_config.for_debug"
          || String.is_prefix relative ~prefix:".claude/skills/"
             && String.is_suffix relative ~suffix:".md"
          || (String.is_prefix relative ~prefix:"docs/" && String.is_suffix relative ~suffix:".md")
          || String.is_prefix relative ~prefix:"benchmarks/"
             && String.is_suffix relative ~suffix:".md"
          || String.equal (Stdlib.Filename.basename relative) "dune"
          || String.equal (Stdlib.Filename.basename relative) "ocannl_config"
          || (String.is_prefix relative ~prefix:"tools/"
             || String.is_prefix relative ~prefix:"scripts/"
             || String.is_prefix relative ~prefix:"benchmarks/")
             && (String.is_suffix relative ~suffix:".sh" || String.is_suffix relative ~suffix:".py")
        then Some (relative, path)
        else None)
    |> List.dedup_and_sort ~compare:(fun (a, _) (b, _) -> String.compare a b)
  in
  let scripts =
    List.filter files ~f:(fun (path, _) ->
        String.is_suffix path ~suffix:".sh" || String.is_suffix path ~suffix:".py")
  in
  let scripts_under root =
    List.filter scripts ~f:(fun (path, _) -> String.is_prefix path ~prefix:root)
  in
  let markdown = List.filter files ~f:(fun (path, _) -> String.is_suffix path ~suffix:".md") in
  let roots = [ "tools/"; "scripts/"; "benchmarks/" ] in
  Verdict.p_all "the scan reaches script files under tools, scripts, and benchmarks" roots
    ~f:(fun root -> not (List.is_empty (scripts_under root)));
  Verdict.p "the scan reaches AGENTS.md, skill docs, root README, docs, and benchmark Markdown"
    (List.exists markdown ~f:(fun (path, _) -> String.equal path "AGENTS.md")
    && List.exists markdown ~f:(fun (path, _) -> String.is_prefix path ~prefix:".claude/skills/")
    && List.exists markdown ~f:(fun (path, _) -> String.equal path "README.md")
    && List.exists markdown ~f:(fun (path, _) -> String.is_prefix path ~prefix:"docs/")
    && List.exists markdown ~f:(fun (path, _) ->
        String.is_prefix path ~prefix:"benchmarks/"
        && not (String.equal path "benchmarks/README.md")));
  Verdict.p "the scan reaches Dune actions"
    (List.exists files ~f:(fun (path, _) -> String.equal (Stdlib.Filename.basename path) "dune"));
  let config_roots =
    [
      "arrayjit/test/";
      "benchmarks/";
      "test/config/";
      "test/operations/profiles/";
      "test/operations/startup_streams/";
    ]
  in
  Verdict.p_all "the scan reaches every checked-in ocannl_config root" config_roots ~f:(fun root ->
      List.exists files ~f:(fun (path, _) ->
          String.is_prefix path ~prefix:root
          && String.equal (Stdlib.Filename.basename path) "ocannl_config"));
  Verdict.p "the scan reaches ocannl_config.reference examples"
    (List.exists files ~f:(fun (path, _) -> String.equal path "ocannl_config.reference"));
  Verdict.p "the scan reaches the checked-in debug config template"
    (List.exists files ~f:(fun (path, _) -> String.equal path "ocannl_config.for_debug"));
  let occurrences =
    List.concat_map files ~f:(fun (reported_path, path) -> occurrences_of_file ~reported_path path)
  in
  check ~repository_census:true occurrences;
  let cli_count =
    List.count occurrences ~f:(fun o ->
        Poly.equal o.kind Cli_flag || Poly.equal o.kind Prefix_free_cli_flag)
  in
  let env_count = List.count occurrences ~f:(fun o -> Poly.equal o.kind Environment_assignment) in
  let markdown_count = List.count occurrences ~f:(fun o -> Poly.equal o.kind Markdown_assignment) in
  let config_count =
    List.count occurrences ~f:(fun o -> Poly.equal o.kind Config_file_assignment)
  in
  eprintf
    "config usage scan: %d files, %d command-line flags, %d environment assignments, %d \
     documentation assignments, %d config-file assignments\n"
    (List.length files) cli_count env_count markdown_count config_count;
  if not (Verdict.any_failed ()) then (
    printf
      "OK: qualified OCANNL command-line flags and OCANNL_<KEY>= assignments in scripts under \
       tools/, scripts/, and benchmarks/ name registered keys.\n";
    printf
      "OK: inline key=value assignments in scanned Markdown name registered keys or explicit \
       non-config notation.\n";
    printf
      "OK: configuration tokens in Dune actions and checked-in ocannl_config files name registered \
       keys or counted intentional-invalid controls.\n")

let () =
  match Array.to_list Stdlib.Sys.argv with
  | _ :: [ "--fixture"; path; config_path ] -> fixture path config_path
  | _ :: workspace_root :: paths when not (List.is_empty paths) -> live workspace_root paths
  | argv ->
      eprintf "Usage: %s <workspace_root> <files...> | %s --fixture <file> <config-file>\n"
        (List.hd_exn argv) (List.hd_exn argv);
      Stdlib.exit 1
