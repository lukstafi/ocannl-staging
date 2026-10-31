(* gh-ocannl-1718: config [autotune_log=crown] prints exactly one crown-and-finals line per [tune]
   call, on stderr, and nothing else of the [autotune_log] stream.

   The dune action passes [--ocannl_autotune_log=crown], since the level is read once per process.
   Each call's line is claimed to be exactly [autotune-crown: routine=<name> <fields>], where the
   fields are [Autotune.crown_fields] of the report the same call delivered to its [~report], so the
   line and a caller's own rendering cannot drift apart. The calls cover the paths that report
   differently: a search, a cache replay of it, a search-less call, and a deferred call whose
   compile runs after its report (the [sealed] path, which must not report twice). A call's [?log]
   overrides the level: [~log:false] silences the line, [~log:true] keeps it beside the full
   diagnostics. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
open Verdict.Claims

let prefix = "autotune-crown: "
let crown_lines = List.filter ~f:(String.is_prefix ~prefix)

let comp label =
  let a = TDSL.ndarray [| 1.; 2.; 3.; 4. |] ~label:[ label ^ "_a" ] ~output_dims:[ 4 ] () in
  let b = TDSL.ndarray [| 4.; 3.; 2.; 1. |] ~label:[ label ^ "_b" ] ~output_dims:[ 4 ] () in
  let%op sum = a + b in
  Train.forward sum

(* One call under a stderr capture: the lines it wrote and the report it delivered. *)
let captured ?log ?(search = true) ?(cache_dir = "") ?(deferred = false) ~name comp =
  let report = ref None in
  let on_report r = report := Some r in
  let run () =
    Autotune.with_uncontended_test_windows (fun () ->
        if deferred then
          let d =
            Autotune.tune_deferred ~name ~search ~beam_width:1 ~rounds:0 ~repeats:1 ~cache_dir ?log
              ~progress:false ~report:on_report (Context.cpu ()) comp Ir.Indexing.Empty
          in
          ignore (Autotune.compile_deferred d : Context.t * Context.routine)
        else
          ignore
            (Autotune.tune ~name ~search ~beam_width:1 ~rounds:0 ~repeats:1 ~cache_dir ?log
               ~progress:false ~report:on_report (Context.cpu ()) comp Ir.Indexing.Empty
              : Context.t * Context.routine))
  in
  let (), lines = Stderr_capture.lines run in
  (lines, !report)

let expected_line ~name r = Printf.sprintf "%sroutine=%S %s" prefix name (Autotune.crown_fields r)

(* A call under the ambient level: exactly its one line, matching its report, and no full
   diagnostics. *)
let one_line ~what ~name (lines, report) =
  match report with
  | None -> p (what ^ ": the call delivered a report") false
  | Some r ->
      p
        (what ^ ": exactly one crown line, the report's own fields")
        (List.equal String.equal (crown_lines lines) [ expected_line ~name r ]);
      p_none (what ^ ": no full autotune_log line") lines ~f:(String.is_prefix ~prefix:"autotune: ")

let () =
  p "the crown level leaves the full diagnostics off" (not (Autotune.log_enabled ()));
  (* Emptied first, so the first call searches and the second replays what it saved. *)
  let cache_dir = "autotune_cache_crown_log" in
  if Stdlib.Sys.file_exists cache_dir then
    Array.iter (Stdlib.Sys.readdir cache_dir) ~f:(fun f ->
        Stdlib.Sys.remove (Stdlib.Filename.concat cache_dir f));
  let searched = captured ~name:"crown_search" ~cache_dir (comp "cs") in
  one_line ~what:"a search" ~name:"crown_search" searched;
  p "the search was searched"
    (Option.exists (snd searched) ~f:(fun r ->
         String.equal (Autotune.outcome_name r.Autotune.outcome) "searched"));
  let replayed = captured ~name:"crown_search" ~cache_dir (comp "cs") in
  one_line ~what:"a cache replay" ~name:"crown_search" replayed;
  p "the second call replayed the cache"
    (Option.exists (snd replayed) ~f:(fun r ->
         String.equal (Autotune.outcome_name r.Autotune.outcome) "cache-replay"));
  one_line ~what:"a search-less call" ~name:"crown_nosearch"
    (captured ~search:false ~name:"crown_nosearch" (comp "ns"));
  one_line ~what:"a deferred call, its compile included" ~name:"crown_deferred"
    (captured ~deferred:true ~name:"crown_deferred" (comp "df"));
  (* The call delivered its report, and wrote no line for it. *)
  let silenced, silenced_report = captured ~log:false ~name:"crown_silenced" (comp "sl") in
  p_empty "~log:false silences the crown line" ~over:(Option.to_list silenced_report)
    (crown_lines silenced);
  let full, full_report = captured ~log:true ~name:"crown_full" (comp "fl") in
  (match full_report with
  | None -> p "~log:true: the call delivered a report" false
  | Some r ->
      p "~log:true keeps exactly the one crown line"
        (List.equal String.equal (crown_lines full) [ expected_line ~name:"crown_full" r ]));
  p_exists "~log:true prints the full diagnostics too" full
    ~f:(String.is_prefix ~prefix:"autotune: ")
