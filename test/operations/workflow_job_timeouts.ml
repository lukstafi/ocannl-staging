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

let strip_trailing_comment value =
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
                let calls_reusable =
                  List.exists job_keys ~f:(fun (_, key, _) -> String.equal key "uses")
                in
                collect ({ file; name; timeout; steps_line; calls_reusable } :: found) rest)
        | _ :: rest -> collect found rest
      in
      Some (collect [] block)

let declared_timeout job =
  match job.timeout with
  | None -> None
  | Some (_, value) -> Option.some_if (not (String.is_empty value)) value

(* Two-sided: an ordinary job must declare a ceiling, and a caller job must not -- GitHub rejects
   `timeout-minutes` on the job-level `uses:` form, so requiring it there would demand a workflow
   that cannot run. *)
let bounded job =
  if job.calls_reusable then Option.is_none (declared_timeout job)
  else Option.is_some (declared_timeout job)

let ahead_of_steps job =
  match (job.timeout, job.steps_line) with
  | Some (number, _), Some steps -> number < steps
  | Some _, None -> true
  | None, _ -> job.calls_reusable

let describe job =
  if job.calls_reusable then
    match declared_timeout job with
    | None -> "calls a reusable workflow, which carries its own ceilings"
    | Some value -> "calls a reusable workflow AND declares timeout-minutes: " ^ value
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
  let jobs = List.concat_map scanned ~f:(fun (_, jobs) -> Option.value jobs ~default:[]) in
  printf "GitHub workflow jobs and the runtime ceiling each one declares:\n";
  List.iter scanned ~f:(fun (relative, jobs) ->
      match jobs with
      | None -> printf "  %s: no top-level jobs block\n" relative
      | Some jobs ->
          List.iter jobs ~f:(fun job -> printf "  %s %s: %s\n" relative job.name (describe job)));
  printf "\n";
  List.iter scanned ~f:(fun (relative, jobs) ->
      if Option.is_none jobs then
        eprintf "%s: no top-level `jobs:` key; this scan cannot answer for its jobs\n" relative);
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
  p_all ~min:4 "every checked-in GitHub workflow file declares a top-level jobs mapping" scanned
    ~f:(fun (_, jobs) -> Option.is_some jobs);
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
  let run ?(extra = []) subject =
    Int.incr case_index;
    let root = Stdlib.Filename.concat fixture (Printf.sprintf "case%d" !case_index) in
    let workflows = Stdlib.Filename.concat root ".github/workflows" in
    let filler index = Printf.sprintf "filler%d.yml" index in
    List.iter [ 1; 2; 3 ] ~f:(fun index ->
        write_file
          (Stdlib.Filename.concat workflows (filler index))
          (workflow [ sound_job ~name:"build" ]));
    write_file (Stdlib.Filename.concat workflows "subject.yml") subject;
    List.iter extra ~f:(fun (path, content) ->
        write_file (Stdlib.Filename.concat root path) content);
    let files =
      ("subject.yml" :: List.map [ 1; 2; 3 ] ~f:filler
      |> List.map ~f:(Stdlib.Filename.concat ".github/workflows"))
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
  let missing_timeout name =
    Printf.sprintf ".github/workflows/subject.yml: job `%s` declares no job-level `timeout-minutes`"
      name
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
  let no_jobs = run "name: fixture\non: workflow_dispatch\n" in
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
  p "a workflow file with no jobs mapping is refused rather than read as having no jobs"
    (refused "no jobs mapping"
       ~messages:[ "no top-level `jobs:` key; this scan cannot answer for its jobs" ]
       no_jobs);
  p "a YAML file nested under .github/workflows is no workflow, and its unbounded job is not judged"
    (passed "nested fixture" nested_fixture);
  Test_utils.Refusal_control_manifest.print "workflow_job_timeouts.ml";
  remove_tree fixture

let () =
  match Array.to_list Stdlib.Sys.argv with
  | _ :: "--scan-only" :: workspace_root :: paths -> scan workspace_root paths
  | _ :: workspace_root :: paths ->
      scan workspace_root paths;
      control ()
  | _ ->
      eprintf "Usage: %s [--scan-only] <workspace root> <workflow files...>\n" Stdlib.Sys.argv.(0);
      Stdlib.exit 2
