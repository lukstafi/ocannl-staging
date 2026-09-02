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
   where the reader of the job looks for it.

   A CALLER JOB is the exception, and a two-sided one: a job whose body is a job-level [uses:]
   calling a reusable workflow may not carry [timeout-minutes] at all -- GitHub rejects the workflow
   if it does -- and the jobs it runs are bounded in the workflow it calls. So a caller job is
   required NOT to declare one, rather than exempted from the question.

   {1 What the reader refuses}

   The two levels this scan reads -- the job keys under [jobs:], and the keys under each job -- must
   present a shape it can DECIDE. Anything else is refused by name rather than read generously: a
   file with no [jobs:] key, an inline [jobs: {}] value, a [jobs:] block with no entries, a job
   whose body is inline, a job with no keys, a line at either level this reader cannot parse as a
   key, and a file indented with tabs. The alternative -- accepting what it cannot parse -- makes a
   file that was never examined indistinguishable from one that passed, which is the failure mode
   this whole scan exists to prevent.

   The live tree is valid, which is exactly why its refusals need synthetic controls: green because
   intact and green because blind are the same output. [control] builds workflow trees that differ
   from a legitimate one in one respect each, runs this same executable over them as a CHILD, and
   claims both the exit status and the diagnostic text -- for every refusal, and for the nearest
   accepted counterpart of each. Capturing the children's streams keeps their [FAIL:] markers out of
   a green suite log, following [ocamlformat_ignore_scan] and [generated_provenance]. *)

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
  calls_reusable : bool;
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

(* A comment is not a value: `timeout-minutes: # decide later` declares the key and no ceiling, and
   so does a bare `timeout-minutes:`. Both reduce to the empty string here, which is what
   [declared_timeout] refuses. *)
let strip_trailing_comment value =
  if String.is_prefix (String.strip value) ~prefix:"#" then ""
  else
    match String.substr_index value ~pattern:" #" with
    | Some position -> String.strip (String.prefix value position)
    | None -> value

(* [run: |] and its relatives introduce a literal block: everything indented under such a key is
   text, not structure. Dropping those lines is what keeps a step's shell script from contributing
   keys of its own. *)
let is_block_scalar value =
  let value = strip_trailing_comment value in
  let after_indicator =
    Option.first_some (String.chop_prefix value ~prefix:"|") (String.chop_prefix value ~prefix:">")
  in
  Option.value_map after_indicator ~default:false
    ~f:(String.for_all ~f:(fun c -> Char.equal c '-' || Char.equal c '+'))

(* A block scalar is introduced by a KEY, and that key may itself sit in a sequence entry -- [- run:
   |] is how every step in this repository spells it. Strip the entry markers before asking whether
   the line opens a block, or the scripts under `steps:` are never dropped and their contents are
   read as workflow structure. *)
let block_scalar_opener line =
  let rec strip_markers text =
    let stripped = String.lstrip text in
    match String.chop_prefix stripped ~prefix:"-" with
    | Some rest when String.is_prefix rest ~prefix:" " || String.is_empty rest -> strip_markers rest
    | _ -> stripped
  in
  match key_of { line with text = strip_markers line.text } with
  | Some (_, value) -> is_block_scalar value
  | None -> false

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
        if block_scalar_opener line then skip_below := Some line.indent;
        true))

(* The lines strictly under [head], i.e. up to the next line at [head]'s indentation or
   shallower. *)
let body_of ~head rest = List.take_while rest ~f:(fun line -> line.indent > head.indent)

(* Why a file cannot be judged, in the words its refusal will use. Keeping the reasons in one type
   is what makes "the reader refuses what it cannot decide" checkable: a new shape adds a
   constructor here and an arm to the controls, rather than a silent acceptance nobody sees. *)
type unreadable =
  | Tab_indentation
  | No_jobs_key
  | Inline_jobs_value
  | Empty_jobs_block
  | Unreadable_key of int
  | Inline_job_body of string
  | Job_without_keys of string

let explain = function
  | Tab_indentation -> "a line indented with tabs, which this reader measures in spaces"
  | No_jobs_key -> "no top-level `jobs:` key"
  | Inline_jobs_value -> "an inline `jobs:` value rather than a block of job entries"
  | Empty_jobs_block -> "a `jobs:` key with no job entries under it"
  | Unreadable_key line -> Printf.sprintf "line %d, which is not a key this reader can parse" line
  | Inline_job_body name -> Printf.sprintf "an inline body for job `%s`" name
  | Job_without_keys name -> Printf.sprintf "no keys at all under job `%s`" name

type reading = Jobs of job list | Unreadable of unreadable

let leading_tab text =
  let rec scan index =
    index < String.length text
    && (Char.equal text.[index] '\t' || (Char.equal text.[index] ' ' && scan (index + 1)))
  in
  scan 0

let jobs_head lines =
  List.findi lines ~f:(fun _ line ->
      line.indent = 0 && match key_of line with Some ("jobs", _) -> true | _ -> false)

let minimum_indent lines =
  List.min_elt (List.map lines ~f:(fun { indent; _ } -> indent)) ~compare:Int.compare

(* Reads the two levels this scan decides, and refuses -- by name -- every shape it cannot. The
   refusal is raised rather than returned so that no arm can degrade into an empty job list, which
   is the shape that would make an unexamined file look like a passing one. *)
exception Refuse of unreadable

let jobs_of ~file text =
  let refuse reason = raise (Refuse reason) in
  let required option ~reason = match option with Some value -> value | None -> refuse reason in
  let read () =
    (* Block scalars first: a tab is legal CONTENT (a Makefile recipe in a `run: |` body), and only
       a tab this reader would have to measure as indentation is a refusal. *)
    let lines = drop_block_scalars (content_lines text) in
    if List.exists lines ~f:(fun { text; _ } -> leading_tab text) then refuse Tab_indentation;
    let position, head = required (jobs_head lines) ~reason:No_jobs_key in
    (match key_of head with
    | Some (_, value) when not (String.is_empty (strip_trailing_comment value)) ->
        refuse Inline_jobs_value
    | _ -> ());
    let block = body_of ~head (List.drop lines (position + 1)) in
    let job_indent = required (minimum_indent block) ~reason:Empty_jobs_block in
    let rec collect found = function
      | [] -> List.rev found
      | head :: rest when head.indent = job_indent ->
          let body = body_of ~head rest in
          let name, inline = required (key_of head) ~reason:(Unreadable_key head.number) in
          if not (String.is_empty (strip_trailing_comment inline)) then
            refuse (Inline_job_body name);
          let key_indent = required (minimum_indent body) ~reason:(Job_without_keys name) in
          let job_keys =
            (* A block sequence may be written at its key's own indentation ([steps:] then [- name:
               …] both at four), and those entries are the key's VALUE, not further keys of the job.
               `.github/workflows/ci.yml` is written that way. *)
            List.filter body ~f:(fun line ->
                line.indent = key_indent
                && not (String.is_prefix (String.strip line.text) ~prefix:"-"))
            |> List.map ~f:(fun line ->
                let key, value = required (key_of line) ~reason:(Unreadable_key line.number) in
                (line.number, key, value))
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
          let calls_reusable =
            List.exists job_keys ~f:(fun (_, key, _) -> String.equal key "uses")
          in
          collect ({ file; name; timeout; steps_line; calls_reusable } :: found) rest
      | line :: rest ->
          if line.indent < job_indent then refuse (Unreadable_key line.number)
          else collect found rest
    in
    match collect [] block with [] -> refuse Empty_jobs_block | jobs -> jobs
  in
  match read () with jobs -> Jobs jobs | exception Refuse reason -> Unreadable reason

let declared_timeout job =
  match job.timeout with
  | None -> None
  | Some (_, value) -> Option.some_if (not (String.is_empty value)) value

(* Two-sided: an ordinary job must declare a ceiling, and a caller job must not. GitHub rejects
   `timeout-minutes` on the job-level `uses:` form, so requiring it there would demand a workflow
   that cannot run -- and for a caller it is the KEY that GitHub rejects, whatever value follows it,
   which is why this side asks about [job.timeout] rather than about a value that survived. *)
let bounded job =
  if job.calls_reusable then Option.is_none job.timeout else Option.is_some (declared_timeout job)

let ahead_of_steps job =
  match (job.timeout, job.steps_line) with
  | Some (number, _), Some steps -> number < steps
  | Some _, None -> true
  | None, _ -> job.calls_reusable

let describe job =
  if job.calls_reusable then
    match job.timeout with
    | None -> "calls a reusable workflow, which carries its own ceilings"
    | Some (_, value) ->
        "calls a reusable workflow AND declares timeout-minutes: "
        ^ if String.is_empty value then "(no value)" else value
  else
    match declared_timeout job with
    | Some value -> "timeout-minutes: " ^ value
    | None -> "NO timeout-minutes"

(* GitHub discovers workflows directly inside `.github/workflows`; a YAML file in a subdirectory
   under it is a fixture or a supporting file that no job ever runs. The globs are recursive because
   that is what makes this a repository-wide scan in dune's eyes, so the depth is decided here. *)
let is_workflow relative =
  match String.chop_prefix relative ~prefix:".github/workflows/" with
  | None -> false
  | Some basename ->
      (not (String.mem basename '/'))
      && (String.is_suffix basename ~suffix:".yml" || String.is_suffix basename ~suffix:".yaml")

let scan workspace_root paths =
  let base = Test_utils.Dune_stanza_scan.base_dir workspace_root in
  let files =
    List.filter paths ~f:(fun path -> not (Stdlib.Sys.is_directory path))
    |> List.filter_map ~f:(fun path ->
        let relative = Test_utils.Dune_stanza_scan.repo_relative base path in
        Option.some_if (is_workflow relative) (relative, path))
    |> List.dedup_and_sort ~compare:(fun (a, _) (b, _) -> String.compare a b)
  in
  let scanned =
    List.map files ~f:(fun (relative, path) ->
        (relative, jobs_of ~file:relative (In_channel.read_all path)))
  in
  let jobs =
    List.concat_map scanned ~f:(fun (_, reading) ->
        match reading with Jobs jobs -> jobs | Unreadable _ -> [])
  in
  (* The inventory goes to stderr, where the other repo-wide scans put their censuses: a bounded job
     renamed, or a ceiling tuned from 30 to 45, changes none of the claims below, and a golden that
     moved with it would make every workflow edit promote this file -- a conflict point for branches
     that have nothing to do with the rule. Stdout keeps what is stably true. *)
  eprintf
    "GitHub workflow jobs and the runtime ceiling each one declares (not part of the golden):\n";
  List.iter scanned ~f:(fun (relative, reading) ->
      match reading with
      | Unreadable reason -> eprintf "  %s: UNREADABLE -- %s\n" relative (explain reason)
      | Jobs jobs ->
          List.iter jobs ~f:(fun job -> eprintf "  %s %s: %s\n" relative job.name (describe job)));
  List.iter scanned ~f:(fun (relative, reading) ->
      match reading with
      | Jobs _ -> ()
      | Unreadable reason ->
          eprintf "%s: this scan cannot answer for its jobs -- it found %s\n" relative
            (explain reason));
  List.iter jobs ~f:(fun job ->
      if not (bounded job) then
        if job.calls_reusable then
          eprintf
            "%s: job `%s` calls a reusable workflow and declares `timeout-minutes`, which GitHub \
             rejects on that form\n"
            job.file job.name
        else
          eprintf
            "%s: job `%s` declares no job-level `timeout-minutes`; it would run to GitHub's 6-hour \
             default\n"
            job.file job.name
      else if not (ahead_of_steps job) then
        eprintf "%s: job `%s` declares `timeout-minutes` after its `steps:`\n" job.file job.name);
  p_all ~min:4 "every checked-in GitHub workflow file presents jobs this scan can read" scanned
    ~f:(fun (_, reading) -> match reading with Jobs _ -> true | Unreadable _ -> false);
  p_all
    "every workflow job declares a job-level runtime ceiling, unless it calls a reusable workflow"
    jobs ~f:bounded;
  p_all "every workflow job states that ceiling ahead of the steps it bounds" jobs ~f:ahead_of_steps;
  eprintf "Scanned %d workflow files and %d jobs.\n" (List.length files) (List.length jobs)

(* {1 Synthetic controls} *)

let write_file path data =
  let rec mkdirs dir =
    if not (String.equal dir Stdlib.Filename.current_dir_name || Stdlib.Sys.file_exists dir) then (
      mkdirs (Stdlib.Filename.dirname dir);
      try Unix.mkdir dir 0o755 with Unix.Unix_error (Unix.EEXIST, _, _) -> ())
  in
  mkdirs (Stdlib.Filename.dirname path);
  Out_channel.write_all path ~data

let rec remove_tree path =
  match Unix.lstat path with
  | { Unix.st_kind = Unix.S_DIR; _ } ->
      Array.iter (Stdlib.Sys.readdir path) ~f:(fun entry ->
          remove_tree (Stdlib.Filename.concat path entry));
      Unix.rmdir path
  | _ -> Unix.unlink path
  | exception Unix.Unix_error _ -> ()

let describe_status = function
  | Unix.WEXITED n -> Printf.sprintf "exited %d" n
  | Unix.WSIGNALED n -> Printf.sprintf "was killed by signal %d" n
  | Unix.WSTOPPED n -> Printf.sprintf "was stopped by signal %d" n

let sound_job ~name =
  Printf.sprintf
    "  %s:\n    runs-on: ubuntu-latest\n    timeout-minutes: 10\n    steps:\n      - run: echo hi\n"
    name

let workflow jobs = "name: fixture\non: workflow_dispatch\njobs:\n" ^ String.concat jobs

(* The child is run FROM the fixture root with the workflow paths spelled relative to it, which is
   the shape dune's own action has: [%{workspace_root}] and paths that resolve against it. Handing
   absolute paths instead would make [repo_relative] answer with a filesystem path that starts at
   the temporary directory, and every fixture would look like a tree containing no workflows at
   all. *)
let run_child ~exe ~root ~files =
  let capture suffix = Stdlib.Filename.temp_file "workflow_timeouts_control" suffix in
  let out_path = capture ".out" and err_path = capture ".err" in
  let open_capture path = Unix.openfile path [ Unix.O_WRONLY; Unix.O_TRUNC ] 0o600 in
  let out = open_capture out_path and err = open_capture err_path in
  (* [--scan-only] suppresses the control driver in the child; without it every child would stage
     another generation of children. *)
  let argv = Array.of_list (exe :: "--scan-only" :: "." :: files) in
  let here = Stdlib.Sys.getcwd () in
  Unix.chdir root;
  let pid = Unix.create_process exe argv Unix.stdin out err in
  let _, status = Unix.waitpid [] pid in
  Unix.chdir here;
  Unix.close out;
  Unix.close err;
  let text = In_channel.read_all out_path ^ In_channel.read_all err_path in
  (try Unix.unlink out_path with Unix.Unix_error _ -> ());
  (try Unix.unlink err_path with Unix.Unix_error _ -> ());
  (status, text)

(* Every fixture tree carries the same three sound workflows beside the one under test, so the
   file-population floor the live run makes is the floor the controls run against too: a control
   that quietly ran a weaker claim than the shipping one would prove nothing about the shipping
   one. *)
let control () =
  let exe =
    let name = Stdlib.Sys.executable_name in
    if Stdlib.Filename.is_relative name then Stdlib.Filename.concat (Stdlib.Sys.getcwd ()) name
    else name
  in
  let fixture = Stdlib.Filename.temp_dir "workflow timeouts control " "" in
  let case_index = ref 0 in
  (* One filler is a `.yaml`, so the accepted arms exercise both discovered spellings; a refusing
     arm names its subject `.yaml` too, which is what makes discovery of that spelling load-bearing
     rather than incidental. *)
  let run ?(extra = []) ?(subject_name = "subject.yml") subject =
    Int.incr case_index;
    let root = Stdlib.Filename.concat fixture (Printf.sprintf "case%d" !case_index) in
    let workflows = Stdlib.Filename.concat root ".github/workflows" in
    let fillers = [ "filler1.yml"; "filler2.yml"; "filler3.yaml" ] in
    List.iter fillers ~f:(fun name ->
        write_file (Stdlib.Filename.concat workflows name) (workflow [ sound_job ~name:"build" ]));
    write_file (Stdlib.Filename.concat workflows subject_name) subject;
    List.iter extra ~f:(fun (path, content) ->
        write_file (Stdlib.Filename.concat root path) content);
    let files =
      (subject_name :: fillers |> List.map ~f:(Stdlib.Filename.concat ".github/workflows"))
      @ List.map extra ~f:fst
    in
    run_child ~exe ~root ~files
  in
  let report label (status, text) =
    eprintf "the %s control %s. Its captured output:\n%s\n" label (describe_status status) text
  in
  let passed label (status, text) =
    let ok = match status with Unix.WEXITED 0 -> true | _ -> false in
    if not ok then report label (status, text);
    ok
  in
  (* Split so the emptiness guard sits in the boolean this returns: a refusal nobody named is not
     one this control observed, so an empty message list fails rather than satisfying [for_all]
     vacuously. *)
  let matched_refusal ~messages (status, text) =
    (not (List.is_empty messages))
    && (match status with Unix.WEXITED 1 -> true | _ -> false)
    && List.for_all messages ~f:(fun message -> String.is_substring text ~substring:message)
  in
  let refused label ~messages outcome =
    let ok = matched_refusal ~messages outcome in
    if not ok then report label outcome;
    ok
  in
  let missing_timeout ?(file = "subject.yml") name =
    Printf.sprintf ".github/workflows/%s: job `%s` declares no job-level `timeout-minutes`" file
      name
  in
  let unreadable reason =
    Printf.sprintf "this scan cannot answer for its jobs -- it found %s" reason
  in
  let unbounded_job name =
    Printf.sprintf "  %s:\n    runs-on: ubuntu-latest\n    steps:\n      - run: echo hi\n" name
  in
  let legitimate =
    run
      (workflow
         [
           "  matrixed:\n\
           \    runs-on: ${{ matrix.os }}\n\
           \    timeout-minutes: ${{ matrix.os == 'windows-latest' && 240 || 45 }}\n\
           \    steps:\n\
           \      - run: |\n\
           \          echo 'a block scalar mentioning timeout-minutes: 1'\n";
           "  caller:\n    uses: ./.github/workflows/filler1.yml\n";
           (* The shape `ci.yml` is written in: the sequence under `steps:` starts at the key's own
              indentation rather than deeper. *)
           "  flush_sequence:\n\
           \    runs-on: ubuntu-latest\n\
           \    timeout-minutes: 15\n\
           \    steps:\n\
           \    - name: one\n\
           \      run: echo hi\n";
           sound_job ~name:"plain";
         ])
  in
  let no_timeout =
    run
      (workflow [ "  unbounded:\n    runs-on: ubuntu-latest\n    steps:\n      - run: echo hi\n" ])
  in
  let after_steps =
    run
      (workflow
         [
           "  late:\n\
           \    runs-on: ubuntu-latest\n\
           \    steps:\n\
           \      - run: echo hi\n\
           \    timeout-minutes: 9\n";
         ])
  in
  let block_scalar_decoy =
    run
      (workflow
         [
           "  decoy:\n\
           \    runs-on: ubuntu-latest\n\
           \    steps:\n\
           \      - run: |\n\
           \          echo 'timeout-minutes: 30'\n";
         ])
  in
  let step_level_decoy =
    run
      (workflow
         [
           "  stepwise:\n\
           \    runs-on: ubuntu-latest\n\
           \    steps:\n\
           \      - name: bounded step\n\
           \        timeout-minutes: 5\n\
           \        run: echo hi\n";
         ])
  in
  let caller_with_timeout =
    run
      (workflow
         [ "  caller:\n    uses: ./.github/workflows/filler1.yml\n    timeout-minutes: 30\n" ])
  in
  (* The key present with nothing behind it: a declaration that names no ceiling, against its
     nearest counterpart, a real ceiling wearing a trailing comment. *)
  let empty_value =
    run
      (workflow
         [
           "  hollow:\n\
           \    runs-on: ubuntu-latest\n\
           \    timeout-minutes:\n\
           \    steps:\n\
           \      - run: echo hi\n";
         ])
  in
  let comment_only_value =
    run
      (workflow
         [
           "  deferred:\n\
           \    runs-on: ubuntu-latest\n\
           \    timeout-minutes: # decide later\n\
           \    steps:\n\
           \      - run: echo hi\n";
         ])
  in
  let commented_ceiling =
    run
      (workflow
         [
           "  annotated:\n\
           \    runs-on: ubuntu-latest\n\
           \    timeout-minutes: 20 # ~2min normally; a ceiling only a hang can reach\n\
           \    steps:\n\
           \      - run: echo hi\n";
         ])
  in
  let caller_with_empty_timeout =
    run
      (workflow [ "  caller:\n    uses: ./.github/workflows/filler1.yml\n    timeout-minutes:\n" ])
  in
  let yaml_subject = run ~subject_name:"subject.yaml" (workflow [ unbounded_job "unbounded" ]) in
  let flush_sequence_unbounded =
    run
      (workflow
         [
           "  flush_sequence:\n\
           \    runs-on: ubuntu-latest\n\
           \    steps:\n\
           \    - name: one\n\
           \      run: echo hi\n";
         ])
  in
  let no_jobs = run "name: fixture\non: workflow_dispatch\n" in
  (* The shapes the reader refuses rather than reads generously. Each is a file that would otherwise
     be examined by nobody while every claim stayed green. *)
  let inline_jobs = run "name: fixture\non: workflow_dispatch\njobs: {}\n" in
  let empty_jobs_block = run "name: fixture\non: workflow_dispatch\njobs:\n" in
  let inline_job_body =
    run "name: fixture\non: workflow_dispatch\njobs:\n  build: { runs-on: ubuntu-latest }\n"
  in
  let job_without_keys = run "name: fixture\non: workflow_dispatch\njobs:\n  build:\n" in
  let unreadable_job_key =
    run
      ("name: fixture\non: workflow_dispatch\njobs:\n  \"quoted job\":\n"
     ^ "    runs-on: ubuntu-latest\n    timeout-minutes: 10\n    steps:\n      - run: echo hi\n")
  in
  (* An unreadable line INSIDE an otherwise valid job -- ceiling and all. A reader that skipped it
     would report the job as fully examined, so the refusal is the point; the counterpart is the
     same job with the same key spelled plainly. *)
  let unreadable_key_in_job =
    run
      (workflow
         [
           "  build:\n\
           \    runs-on: ubuntu-latest\n\
           \    timeout-minutes: 10\n\
           \    \"quoted key\": value\n\
           \    steps:\n\
           \      - run: echo hi\n";
         ])
  in
  let readable_key_in_job =
    run
      (workflow
         [
           "  build:\n\
           \    runs-on: ubuntu-latest\n\
           \    timeout-minutes: 10\n\
           \    continue-on-error: false\n\
           \    steps:\n\
           \      - run: echo hi\n";
         ])
  in
  let tab_inside_block_scalar =
    run
      (workflow
         [
           "  build:\n\
           \    runs-on: ubuntu-latest\n\
           \    timeout-minutes: 10\n\
           \    steps:\n\
           \      - run: |\n\
           \          cat > Makefile <<'EOF'\n\
           \          all:\n\
           \          \techo hi\n\
           \          EOF\n";
         ])
  in
  let tab_indented =
    run "name: fixture\non: workflow_dispatch\njobs:\n\tbuild:\n\t\truns-on: ubuntu-latest\n"
  in
  (* A YAML file in a subdirectory of `.github/workflows` is not a workflow: GitHub discovers only
     the files sitting directly there. The recursive glob hands this one over anyway, which is what
     makes the depth check load-bearing rather than decorative. *)
  let nested_fixture =
    run
      ~extra:
        [
          ( ".github/workflows/fixtures/not_a_workflow.yml",
            workflow
              [ "  unbounded:\n    runs-on: ubuntu-latest\n    steps:\n      - run: echo hi\n" ] );
        ]
      (workflow [ sound_job ~name:"plain" ])
  in
  printf
    "\n\
     Synthetic controls run this same scanner over workflow trees differing from a legitimate one\n\
     in one respect each; refusal output is captured and matched below.\n\n";
  p
    "a tree whose jobs carry an expression ceiling, a plain one, and a reusable-workflow call \
     passes"
    (passed "legitimate" legitimate);
  p "a job with no ceiling at all is refused, named, with GitHub's 6-hour default spelled out"
    (refused "unbounded job" ~messages:[ missing_timeout "unbounded" ] no_timeout);
  p "a ceiling declared after the steps it would bound is refused as out of order"
    (refused "late ceiling"
       ~messages:[ "job `late` declares `timeout-minutes` after its `steps:`" ]
       after_steps);
  p "a ceiling that exists only inside a run block scalar leaves the job unbounded"
    (refused "block scalar decoy" ~messages:[ missing_timeout "decoy" ] block_scalar_decoy);
  p "a per-step ceiling is not the job's own, and the job is still refused"
    (refused "step-level decoy" ~messages:[ missing_timeout "stepwise" ] step_level_decoy);
  p "a reusable-workflow call that declares a ceiling GitHub would reject is refused too"
    (refused "caller with a ceiling"
       ~messages:
         [
           "job `caller` calls a reusable workflow and declares `timeout-minutes`, which GitHub \
            rejects on that form";
         ]
       caller_with_timeout);
  p "a timeout-minutes key with nothing behind it declares no ceiling and is refused"
    (refused "empty ceiling value" ~messages:[ missing_timeout "hollow" ] empty_value);
  p "a timeout-minutes key whose whole value is a comment is refused the same way"
    (refused "comment-only ceiling value"
       ~messages:[ missing_timeout "deferred" ]
       comment_only_value);
  p "a real ceiling wearing a trailing comment is a ceiling, and its job passes"
    (passed "commented ceiling" commented_ceiling);
  p "a reusable-workflow call carrying the key with no value is refused on the key, not the value"
    (refused "caller with an empty ceiling key"
       ~messages:
         [
           "job `caller` calls a reusable workflow and declares `timeout-minutes`, which GitHub \
            rejects on that form";
         ]
       caller_with_empty_timeout);
  p "a .yaml workflow is discovered and judged like a .yml one"
    (refused "yaml-spelled subject"
       ~messages:[ missing_timeout ~file:"subject.yaml" "unbounded" ]
       yaml_subject);
  p
    "a job whose steps sequence starts at its own key indentation is still judged, and refused \
     unbounded"
    (refused "flush sequence, no ceiling"
       ~messages:[ missing_timeout "flush_sequence" ]
       flush_sequence_unbounded);
  p "an inline jobs value is refused as a shape this reader cannot decide"
    (refused "inline jobs value"
       ~messages:[ unreadable "an inline `jobs:` value rather than a block of job entries" ]
       inline_jobs);
  p "a jobs key with no entries under it is refused rather than read as a file of no jobs"
    (refused "empty jobs block"
       ~messages:[ unreadable "a `jobs:` key with no job entries under it" ]
       empty_jobs_block);
  p "an inline job body is refused, its keys being unreadable to this reader"
    (refused "inline job body"
       ~messages:[ unreadable "an inline body for job `build`" ]
       inline_job_body);
  p "a job with no keys under it is refused rather than read as a job without a ceiling"
    (refused "job without keys"
       ~messages:[ unreadable "no keys at all under job `build`" ]
       job_without_keys);
  p "an unreadable line inside an otherwise valid job is refused, ceiling and all"
    (refused "unreadable key inside a job"
       ~messages:[ "which is not a key this reader can parse" ]
       unreadable_key_in_job);
  p "the same job with that key spelled plainly is read and passes"
    (passed "readable key inside a job" readable_key_in_job);
  p "a quoted job key is refused rather than passed over, ceiling and all"
    (refused "unreadable job key"
       ~messages:[ "which is not a key this reader can parse" ]
       unreadable_job_key);
  p "a tab inside a run block scalar is content, and its workflow passes"
    (passed "tab inside a block scalar" tab_inside_block_scalar);
  p "a tab-indented workflow is refused, this reader measuring indentation in spaces"
    (refused "tab indentation"
       ~messages:[ unreadable "a line indented with tabs, which this reader measures in spaces" ]
       tab_indented);
  p "a workflow file with no jobs mapping is refused rather than read as having no jobs"
    (refused "no jobs mapping" ~messages:[ unreadable "no top-level `jobs:` key" ] no_jobs);
  p "a YAML file nested under .github/workflows is no workflow, and its unbounded job is not judged"
    (passed "nested fixture" nested_fixture);
  Test_utils.Refusal_control_manifest.print "workflow_job_timeouts.ml";
  remove_tree fixture

(* Every file this scan reads arrives as a `@<path>` response file, because the list a repository-
   wide glob produces may be longer than a Windows command line; see [Test_utils.Scan_argv]. The
   synthetic controls pass their handful of paths directly, which reaches the same argv. *)
let () =
  match Array.to_list (Test_utils.Scan_argv.expand Stdlib.Sys.argv) with
  | _ :: "--scan-only" :: workspace_root :: paths -> scan workspace_root paths
  | _ :: workspace_root :: paths ->
      scan workspace_root paths;
      control ()
  | _ ->
      eprintf "Usage: %s [--scan-only] <workspace root> <workflow files...>\n" Stdlib.Sys.argv.(0);
      Stdlib.exit 2
