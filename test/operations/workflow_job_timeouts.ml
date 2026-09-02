(* gh-ocannl-605 follow-up: every GitHub Actions job bounds its own runtime.

   A job with no [timeout-minutes] runs to GitHub's 6-hour default, so one hung step spends six
   hours of runner time before anything notices -- a sibling repository lost a month of Actions
   minutes to a single such job. Every job in this repository's workflows was given a ceiling by
   hand; nothing held the next job someone adds to the same rule, and the failure is invisible until
   the bill or the queue shows it.

   This scan reads the workflow files as text, the way [config_usage_scan] does: no YAML library is
   a dependency of this repository, and a scan is not a reason to add one. The grammar it decides is
   the one these files are written in -- block mappings indented with spaces -- and it is
   deliberately narrow: the top-level [jobs:] key, the job keys one level under it, and the keys one
   level under each job. Block scalars ([run: |]) are skipped wholesale, so a shell script that
   happens to contain a colon decides nothing.

   The value is not interpreted. [timeout-minutes: 45] and [timeout-minutes: ${{ matrix.os ==
   'windows-latest' && 240 || 45 }}] are both a ceiling; which ceiling is right for a job is a
   judgment the job's author makes, and this scan only holds that one was made -- before [steps:],
   where the reader of the job looks for it. *)

open Base
open Stdio
open Verdict.Claims

let printf = Test_utils.Refusal_control_manifest.printf

type line = { number : int; indent : int; text : string }

type job = {
  file : string;
  name : string;
  timeout : (int * string) option;  (** line number and rendered value *)
  steps_line : int option;
}

let strip_cr text = Option.value (String.chop_suffix text ~suffix:"\r") ~default:text

let indent_of text =
  let rec count index =
    if index < String.length text && Char.equal text.[index] ' ' then count (index + 1) else index
  in
  count 0

(* Content lines only: blanks and whole-line comments carry no structure, and a document marker
   ([---]) is not a mapping key. *)
let content_lines text =
  String.split_lines text
  |> List.mapi ~f:(fun index raw ->
      let raw = strip_cr raw in
      { number = index + 1; indent = indent_of raw; text = String.rstrip raw })
  |> List.filter ~f:(fun { text; _ } ->
      let stripped = String.strip text in
      (not (String.is_empty stripped))
      && (not (String.is_prefix stripped ~prefix:"#"))
      && not (String.is_prefix stripped ~prefix:"---"))

(* A key line is [<name>:] or [<name>: <value>]. The name is unquoted in every workflow this
   repository has, and an unrecognized line is simply not a key -- it cannot become one by being
   read more generously. *)
let key_of { text; _ } =
  let stripped = String.strip text in
  match String.lsplit2 stripped ~on:':' with
  | Some (name, rest)
    when (not (String.is_empty name))
         && String.for_all name ~f:(fun c ->
             Char.is_alphanum c || List.mem [ '_'; '-'; '.' ] c ~equal:Char.equal) ->
      Some (name, String.strip rest)
  | _ -> None

let strip_trailing_comment value =
  match String.substr_index value ~pattern:" #" with
  | Some position -> String.strip (String.prefix value position)
  | None -> value

(* [run: |] and its relatives introduce a literal block: everything indented under such a key is
   text, not structure. Dropping those lines is what keeps a step's shell script from contributing
   keys of its own. *)
let is_block_scalar value =
  match String.chop_prefix (strip_trailing_comment value) ~prefix:"|" with
  | Some rest -> String.for_all rest ~f:(fun c -> Char.equal c '-' || Char.equal c '+')
  | None -> (
      match String.chop_prefix (strip_trailing_comment value) ~prefix:">" with
      | Some rest -> String.for_all rest ~f:(fun c -> Char.equal c '-' || Char.equal c '+')
      | None -> false)

let drop_block_scalars lines =
  let skip_below = ref None in
  List.filter lines ~f:(fun line ->
      let inside =
        match !skip_below with
        | Some indent when line.indent > indent -> true
        | Some _ ->
            skip_below := None;
            false
        | None -> false
      in
      if inside then false
      else (
        (match key_of line with
        | Some (_, value) when is_block_scalar value -> skip_below := Some line.indent
        | _ -> ());
        true))

(* The lines strictly under [head], i.e. up to the next line at [head]'s indentation or
   shallower. *)
let body_of ~head rest = List.take_while rest ~f:(fun line -> line.indent > head.indent)

let jobs_block lines =
  match
    List.findi lines ~f:(fun _ line ->
        line.indent = 0 && match key_of line with Some ("jobs", _) -> true | _ -> false)
  with
  | None -> None
  | Some (position, head) -> Some (body_of ~head (List.drop lines (position + 1)))

let minimum_indent lines =
  List.min_elt (List.map lines ~f:(fun { indent; _ } -> indent)) ~compare:Int.compare

let jobs_of ~file text =
  let lines = drop_block_scalars (content_lines text) in
  match jobs_block lines with
  | None -> None
  | Some block ->
      let job_indent = Option.value (minimum_indent block) ~default:0 in
      let rec collect found = function
        | [] -> List.rev found
        | head :: rest when head.indent = job_indent -> (
            let body = body_of ~head rest in
            match key_of head with
            | None -> collect found rest
            | Some (name, _) ->
                let key_indent = Option.value (minimum_indent body) ~default:0 in
                let job_keys =
                  List.filter_map body ~f:(fun line ->
                      if line.indent = key_indent then
                        Option.map (key_of line) ~f:(fun (key, value) -> (line.number, key, value))
                      else None)
                in
                let timeout =
                  List.find_map job_keys ~f:(fun (number, key, value) ->
                      Option.some_if
                        (String.equal key "timeout-minutes")
                        (number, strip_trailing_comment value))
                in
                let steps_line =
                  List.find_map job_keys ~f:(fun (number, key, _) ->
                      Option.some_if (String.equal key "steps") number)
                in
                collect ({ file; name; timeout; steps_line } :: found) rest)
        | _ :: rest -> collect found rest
      in
      Some (collect [] block)

let bounded job =
  match job.timeout with None -> false | Some (_, value) -> not (String.is_empty value)

let ahead_of_steps job =
  match (job.timeout, job.steps_line) with
  | Some (number, _), Some steps -> number < steps
  | Some _, None -> true
  | None, _ -> false

let scan workspace_root paths =
  let base = Test_utils.Dune_stanza_scan.base_dir workspace_root in
  let files =
    List.filter paths ~f:(fun path -> not (Stdlib.Sys.is_directory path))
    |> List.filter_map ~f:(fun path ->
        let relative = Test_utils.Dune_stanza_scan.repo_relative base path in
        if
          String.is_prefix relative ~prefix:".github/workflows/"
          && (String.is_suffix relative ~suffix:".yml" || String.is_suffix relative ~suffix:".yaml")
        then Some (relative, path)
        else None)
    |> List.dedup_and_sort ~compare:(fun (a, _) (b, _) -> String.compare a b)
  in
  let scanned =
    List.map files ~f:(fun (relative, path) ->
        (relative, jobs_of ~file:relative (In_channel.read_all path)))
  in
  let jobs = List.concat_map scanned ~f:(fun (_, jobs) -> Option.value jobs ~default:[]) in
  printf "GitHub workflow jobs and the runtime ceiling each one declares:\n";
  List.iter scanned ~f:(fun (relative, jobs) ->
      match jobs with
      | None -> printf "  %s: no top-level jobs block\n" relative
      | Some jobs ->
          List.iter jobs ~f:(fun job ->
              printf "  %s %s: %s\n" relative job.name
                (match job.timeout with
                | Some (_, value) -> "timeout-minutes: " ^ value
                | None -> "NO timeout-minutes")));
  printf "\n";
  List.iter scanned ~f:(fun (relative, jobs) ->
      if Option.is_none jobs then
        eprintf "%s: no top-level `jobs:` key; this scan cannot answer for its jobs\n" relative);
  List.iter jobs ~f:(fun job ->
      if not (bounded job) then
        eprintf
          "%s: job `%s` declares no job-level `timeout-minutes`; it would run to GitHub's 6-hour \
           default\n"
          job.file job.name
      else if not (ahead_of_steps job) then
        eprintf "%s: job `%s` declares `timeout-minutes` after its `steps:`\n" job.file job.name);
  p_all ~min:4 "every checked-in GitHub workflow file declares a top-level jobs mapping" scanned
    ~f:(fun (_, jobs) -> Option.is_some jobs);
  p_all "every workflow job declares a job-level runtime ceiling of its own" jobs ~f:bounded;
  p_all "every workflow job states that ceiling ahead of the steps it bounds" jobs ~f:ahead_of_steps;
  eprintf "Scanned %d workflow files and %d jobs.\n" (List.length files) (List.length jobs);
  Test_utils.Refusal_control_manifest.print "workflow_job_timeouts.ml"

let () =
  match Array.to_list Stdlib.Sys.argv with
  | _ :: workspace_root :: paths -> scan workspace_root paths
  | _ ->
      eprintf "Usage: %s <workspace root> <workflow files...>\n" Stdlib.Sys.argv.(0);
      Stdlib.exit 2
