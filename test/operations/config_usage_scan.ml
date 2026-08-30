(* Configuration names used outside OCaml stay tied to [Utils.known_config_keys].

   The two source forms are deliberately narrow and syntactic. Shell scripts contribute every
   [--ocannl_<key>=] token, including tokens in comments and quoted command strings -- those are
   still instructions or commands a reader can reuse. Markdown contributes an inline code span whose
   whole rendered content is one [key=value] assignment. Fenced code and longer snippets describe
   arbitrary APIs and expression languages, so treating every equals sign in them as configuration
   would make the scan unusable.

   gh-ocannl-790. *)

open Base
open Stdio
module Markdown = Test_utils.Agent_notes_scan

type kind = Shell_flag | Markdown_assignment
type occurrence = { path : string; line : int; key : string; spelling : string; kind : kind }

let key_char c = Char.is_lowercase c || Char.is_digit c || Char.equal c '_'
let normalize_key key = Option.value (String.chop_prefix key ~prefix:"ocannl_") ~default:key

let shell_occurrences ~path content =
  let prefix = "--ocannl_" in
  String.split_lines content
  |> List.mapi ~f:(fun index line ->
      let rec from pos found =
        match String.substr_index line ~pos ~pattern:prefix with
        | None -> List.rev found
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
              let key = String.sub line ~pos:key_start ~len:(!key_stop - key_start) in
              let spelling = String.sub line ~pos:start ~len:(!key_stop - start + 1) in
              from next ({ path; line = index + 1; key; spelling; kind = Shell_flag } :: found)
            else from next found
      in
      from 0 [])
  |> List.concat

let same_range (a, b) (x, y) = Int.equal a x && Int.equal b y

let one_assignment rendered =
  match String.lsplit2 rendered ~on:'=' with
  | Some (raw_key, value)
    when (not (String.is_empty raw_key))
         && (not (String.is_empty value))
         && Char.is_lowercase raw_key.[0]
         && String.for_all raw_key ~f:key_char
         && not (String.exists value ~f:Char.is_whitespace) ->
      Some (normalize_key raw_key)
  | _ -> None

let markdown_occurrences ~path content =
  let scan = Markdown.inert_by_line content in
  let comments line = Markdown.spans_at scan.comment_ranges line in
  let fences line = Markdown.spans_at scan.fence_ranges line in
  Markdown.lines content
  |> List.concat_map ~f:(fun (lineno, line) ->
      Markdown.spans_at scan.ranges lineno
      |> List.filter ~f:(fun range ->
          (not (List.mem (comments lineno) range ~equal:same_range))
          && not (List.mem (fences lineno) range ~equal:same_range))
      |> List.filter_map ~f:(fun (start, stop) ->
          (* A multiline span arrives as one range per physical line. A complete [key=value] mention
             is one inline span, so both delimiters must be on this line. *)
          if stop <= start || not (Char.equal line.[start] '`' && Char.equal line.[stop - 1] '`')
          then None
          else
            let spelling = String.sub line ~pos:start ~len:(stop - start) in
            let rendered = Markdown.code_span_content spelling in
            Option.map (one_assignment rendered) ~f:(fun key ->
                { path; line = lineno; key; spelling = rendered; kind = Markdown_assignment })))

(* These are assignments in other languages or report formats, not configuration. This is a judgment
   list rather than a restatement of a vocabulary owned elsewhere, and it is scoped by FILE as well
   as key: exempting [state] everywhere would let a future stale config mention called [state] pass
   in silence. It is checked in both directions below: an entry whose key becomes real is stale, and
   every entry must still explain at least one scanned occurrence. *)
let non_config_assignment_mentions =
  [
    ("docs/agent-notes/build-and-test.md", "execution");
    ("docs/agent-notes/scheduling-and-autotune.md", "max_chain");
    ("docs/agent-notes/training-and-performance.md", "declines");
    ("docs/agent-notes/training-and-performance.md", "state");
    ("docs/agent-notes/training-and-performance.md", "timed");
    ("docs/precision_inference.md", "top_down_prec");
    ("docs/proposals/axis-labels.md", "hidden");
    ("docs/proposals/axis-labels.md", "name");
    ("docs/proposals/axis-labels.md", "rgb");
    ("docs/proposals/concat-forward-component-data-propagation.md", "a");
    ("docs/proposals/concat-forward-component-data-propagation.md", "a_mc");
    ("docs/proposals/concat-forward-component-data-propagation.md", "b");
    ("docs/proposals/concat-forward-component-data-propagation.md", "b_mc");
    ("docs/proposals/concat-forward-component-data-propagation.md", "c");
    ("docs/proposals/concat-forward-component-data-propagation.md", "c_mc");
    ("docs/proposals/fix-centered-init-test-fallout.md", "epsilon");
    ("docs/proposals/gh-ocannl-255.md", "d");
    ("docs/proposals/gh-ocannl-263.md", "seq_q");
    ("docs/proposals/gh-ocannl-308-comment.md", "n");
    ("docs/proposals/gh-ocannl-420.md", "i");
    ("docs/proposals/gh-ocannl-536.md", "private_seg_size");
    ("docs/proposals/total-basis-bcast-if-1.md", "d");
    ("docs/proposals/watch-ocannl-README-md-369aadb4.md", "batch");
    ("docs/research/llmc-lessons.md", "accumulate");
    ("docs/research/llmc-lessons.md", "beta");
    ("docs/research/lean-attention-feasibility.md", "seq_q");
    ("docs/syntax_extensions.md", "dilation");
    ("docs/syntax_extensions.md", "kernel_size");
    ("docs/syntax_extensions.md", "stride");
  ]

(* These files discuss spellings known to be invalid. The exception is site-specific so repeating
   one in current documentation still fails, and it is usage-checked so deleting the historical
   mention makes the exception stale rather than leaving an escape hatch. *)
let historical_invalid_config_mentions =
  [
    ("docs/agent-notes/conventions.md", "private_bytes_cap");
    ("docs/proposals/gh-ocannl-409.md", "bacend");
    ("docs/proposals/gh-ocannl-409.md", "output_debug_files_in_run_directory");
    ("docs/proposals/gh-ocannl-409.md", "randomness_lib");
  ]

let mention_site path key = path ^ "\000" ^ key

let non_config_assignment_sites =
  Set.of_list (module String)
  @@ List.map non_config_assignment_mentions ~f:(fun (path, key) -> mention_site path key)

let historical_invalid_config_sites =
  Set.of_list (module String)
  @@ List.map historical_invalid_config_mentions ~f:(fun (path, key) -> mention_site path key)

let kind_name = function
  | Shell_flag -> "shell flag"
  | Markdown_assignment -> "documentation assignment"

let check ~repository_census occurrences =
  let seen_non_config = ref (Set.empty (module String)) in
  let seen_historical = ref (Set.empty (module String)) in
  List.iter occurrences ~f:(fun occurrence ->
      if Set.mem Utils.known_config_keys occurrence.key then ()
      else if Set.mem non_config_assignment_sites (mention_site occurrence.path occurrence.key) then
        seen_non_config := Set.add !seen_non_config (mention_site occurrence.path occurrence.key)
      else if Set.mem historical_invalid_config_sites (mention_site occurrence.path occurrence.key)
      then seen_historical := Set.add !seen_historical (mention_site occurrence.path occurrence.key)
      else
        Verdict.fail
          (Printf.sprintf "%s:%d: %s `%s` names `%s`, absent from Utils.known_config_keys"
             occurrence.path occurrence.line (kind_name occurrence.kind) occurrence.spelling
             occurrence.key));
  if repository_census then (
    let newly_real =
      List.filter non_config_assignment_mentions ~f:(fun (_, key) ->
          Set.mem Utils.known_config_keys key)
    in
    if not (List.is_empty newly_real) then
      Verdict.fail
        (Printf.sprintf
           "non-config assignment exemptions now name registered config keys -- remove: %s"
           (newly_real
           |> List.map ~f:(fun (path, key) -> path ^ ":" ^ key)
           |> String.concat ~sep:", "));
    let stale_non_config =
      List.filter non_config_assignment_mentions ~f:(fun (path, key) ->
          not (Set.mem !seen_non_config (mention_site path key)))
    in
    if not (List.is_empty stale_non_config) then
      Verdict.fail
        (Printf.sprintf "non-config assignment exemptions no scanned mention uses: %s"
           (stale_non_config
           |> List.map ~f:(fun (path, key) -> path ^ ":" ^ key)
           |> String.concat ~sep:", "));
    let stale_historical =
      List.filter historical_invalid_config_mentions ~f:(fun (path, key) ->
          not (Set.mem !seen_historical (mention_site path key)))
    in
    if not (List.is_empty stale_historical) then
      Verdict.fail
        (Printf.sprintf "historical invalid-config exemptions no scanned mention uses: %s"
           (stale_historical
           |> List.map ~f:(fun (path, key) -> path ^ ":" ^ key)
           |> String.concat ~sep:", ")))

let file_kind path = if String.is_suffix path ~suffix:".md" then `Markdown else `Shell

let occurrences_of_file ~reported_path path =
  let content = In_channel.read_all path in
  match file_kind path with
  | `Markdown -> markdown_occurrences ~path:reported_path content
  | `Shell -> shell_occurrences ~path:reported_path content

let fixture path =
  let reported_path = Stdlib.Filename.basename path in
  let content = In_channel.read_all path in
  check ~repository_census:false
    (shell_occurrences ~path:reported_path content
    @ markdown_occurrences ~path:reported_path content)

let live workspace_root paths =
  let base = Test_utils.Dune_stanza_scan.base_dir workspace_root in
  let files =
    List.filter paths ~f:(fun path -> not (Stdlib.Sys.is_directory path))
    |> List.filter_map ~f:(fun path ->
        let relative = Test_utils.Dune_stanza_scan.repo_relative base path in
        if
          String.equal relative "AGENTS.md"
          || (String.is_prefix relative ~prefix:"docs/" && String.is_suffix relative ~suffix:".md")
          || (String.is_prefix relative ~prefix:"tools/"
             || String.is_prefix relative ~prefix:"scripts/"
             || String.is_prefix relative ~prefix:"benchmarks/")
             && String.is_suffix relative ~suffix:".sh"
        then Some (relative, path)
        else None)
    |> List.dedup_and_sort ~compare:(fun (a, _) (b, _) -> String.compare a b)
  in
  let under root = List.filter files ~f:(fun (path, _) -> String.is_prefix path ~prefix:root) in
  let markdown =
    List.filter files ~f:(fun (path, _) ->
        String.equal path "AGENTS.md" || String.is_prefix path ~prefix:"docs/")
  in
  let roots = [ "tools/"; "scripts/"; "benchmarks/" ] in
  Verdict.p_all "the scan reaches shell scripts under tools, scripts, and benchmarks" roots
    ~f:(fun root -> not (List.is_empty (under root)));
  Verdict.p "the scan reaches AGENTS.md and documentation"
    (List.exists markdown ~f:(fun (path, _) -> String.equal path "AGENTS.md")
    && List.exists markdown ~f:(fun (path, _) -> String.is_prefix path ~prefix:"docs/"));
  let occurrences =
    List.concat_map files ~f:(fun (reported_path, path) -> occurrences_of_file ~reported_path path)
  in
  check ~repository_census:true occurrences;
  let shell_count = List.count occurrences ~f:(fun o -> Poly.equal o.kind Shell_flag) in
  let markdown_count = List.length occurrences - shell_count in
  eprintf "config usage scan: %d files, %d shell flags, %d documentation assignments\n"
    (List.length files) shell_count markdown_count;
  if not (Verdict.any_failed ()) then (
    printf
      "OK: shell --ocannl_<key>= flags under tools/, scripts/, and benchmarks/ name registered keys.\n";
    printf
      "OK: inline key=value assignments in AGENTS.md and docs/ name registered keys or explicit \
       non-config notation.\n")

let () =
  match Array.to_list Stdlib.Sys.argv with
  | _ :: [ "--fixture"; path ] -> fixture path
  | _ :: workspace_root :: paths when not (List.is_empty paths) -> live workspace_root paths
  | argv ->
      eprintf "Usage: %s <workspace_root> <files...> | %s --fixture <file>\n" (List.hd_exn argv)
        (List.hd_exn argv);
      Stdlib.exit 1
