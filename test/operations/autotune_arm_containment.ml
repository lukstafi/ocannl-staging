(* gh-ocannl-550: a candidate failure in one placement arm must not destroy the other arm's
   completed result.

   The reproduction that motivated this needs a 12 GB GPU and a half-hour search
   (benchmarks/report-gh528-gpt2-cuda.md §3: five of five tf32 gpt2_mini runs OOMed at arm-B
   candidate 47, and the exception took arm A's already-crowned winner out of the process with it).
   Here the failure is injected instead, through [Autotune.on_candidate_attempt], at a chosen
   position within arm B — after arm B has timed candidates of its own, so this also pins that an
   unshippable partial best does not win the A/B.

   Asserted, backend-independently: arm A's winner ships (the returned routine computes the right
   values), it is cached (a second, injection-free tune replays that same schedule from the disk
   cache instead of re-searching), and the failed arm is reported honestly — its partial report
   arrives in position carrying the terminal failure, rather than being silently downgraded to
   "arm B lost". *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module SC = Ir.Schedule_cache

let p name b = Stdio.printf "%s: %b\n" name b
let approx a b = Float.(abs (a - b) < 1e-4)
let n = 8

let clean_cache dir =
  if Stdlib.Sys.file_exists dir && Stdlib.Sys.is_directory dir then
    Array.iter (Stdlib.Sys.readdir dir) ~f:(fun f ->
        Stdlib.Sys.remove (Stdlib.Filename.concat dir f))

(* The injection is global state on a library ref, so it is restored unconditionally: a leaked
   raiser would fail every later autotune call in this process. *)
let with_injected_failure ~arms_reported ~at ~message f =
  let attempts = ref 0 in
  (Autotune.on_candidate_attempt :=
     fun label ->
       (* Arm A reports exactly once, when its search ends, so this fires within arm B only. *)
       if !arms_reported >= 1 then (
         Int.incr attempts;
         if !attempts = at then
           raise (Failure (Printf.sprintf "%s at candidate %s" message label))));
  Exn.protect ~f ~finally:(fun () -> Autotune.on_candidate_attempt := fun _ -> ())

let () =
  let mav = Array.init (n * n) ~f:(fun i -> Float.of_int (i % 7) *. 0.5) in
  let mbv = Array.init (n * n) ~f:(fun i -> Float.of_int (i % 5) -. 2.) in
  let ma = TDSL.ndarray mav ~label:[ "ac_ma" ] ~input_dims:[ n ] ~output_dims:[ n ] () in
  let mb = TDSL.ndarray mbv ~label:[ "ac_mb" ] ~input_dims:[ n ] ~output_dims:[ n ] () in
  let%op mc = ma * mb in
  let%op t2 = relu mc in
  ignore mc;
  let comp = Train.forward t2 in
  (* Reference values from a plain compile. *)
  let ctx_ref, routine_ref = Context.compile (Context.auto ()) comp Ir.Indexing.Empty in
  let ctx_ref = Context.run ctx_ref routine_ref in
  let expected = Context.get_values ctx_ref t2.Tensor.value in

  (* A cache directory of this test's own, emptied first: "arm A's winner is cached" is a claim
     about what run 1 stores, so run 1 has to be a genuine miss. *)
  let cache_dir = "autotune_arm_containment_cache" in
  clean_cache cache_dir;

  (* --- Run 1: arm B dies at its third candidate --- *)
  let arms_reported = ref 0 in
  let reports = ref [] in
  let report r =
    Int.incr arms_reported;
    reports := r :: !reports
  in
  let message = "injected candidate failure" in
  (* Attempt 1 of an arm is its baseline compile; injecting later leaves arm B with timed
     candidates of its own, which is the point of this scenario. *)
  let ctx_t, routine_t =
    with_injected_failure ~arms_reported ~at:4 ~message (fun () ->
        Train.tune_placements ~beam_width:2 ~rounds:0 ~repeats:1 ~cache_dir ~report
          (Context.auto ()) t2 comp Ir.Indexing.Empty)
  in
  let reports1 = List.rev !reports in
  p "the failing arm did not take the tune down with it" true;
  p "both arms reported, in position" (List.length reports1 = 2);
  let arm_a = List.nth_exn reports1 0 and arm_b = List.nth_exn reports1 1 in
  p "arm A completed" (not arm_a.Autotune.partial);
  p "arm A crowned a timed winner" (not (Float.is_inf arm_a.Autotune.best_ms));
  p "arm B is reported as partial" arm_b.Autotune.partial;
  p "arm B's report carries the terminal failure"
    (Option.value_map arm_b.Autotune.terminal_failure ~default:false ~f:(fun tf ->
         String.is_substring tf.Autotune.detail ~substring:message));
  p "arm B had timed candidates before failing"
    (arm_b.Autotune.candidates_timed > 0 && not (Float.is_inf arm_b.Autotune.best_ms));
  let ctx_t = Context.run ctx_t routine_t in
  let got = Context.get_values ctx_t t2.Tensor.value in
  p "the surviving arm's routine ships and computes the right values"
    (Array.for_all2_exn got expected ~f:approx);

  (* --- Run 2, no injection: arm A's winner survived to the disk cache --- *)
  let reports = ref [] in
  let ctx_2, routine_2 =
    Train.tune_placements ~beam_width:2 ~rounds:0 ~repeats:1 ~cache_dir
      ~report:(fun r -> reports := r :: !reports)
      (Context.auto ()) t2 comp Ir.Indexing.Empty
  in
  let arm_a2 = List.nth_exn (List.rev !reports) 0 in
  p "arm A's winner was cached by the run its sibling arm failed in" arm_a2.Autotune.cache_hit;
  p "the replay is the very schedule run 1 crowned"
    (SC.equal_saved_schedule arm_a.Autotune.best_schedule arm_a2.Autotune.best_schedule);
  let ctx_2 = Context.run ctx_2 routine_2 in
  let got_2 = Context.get_values ctx_2 t2.Tensor.value in
  p "the cached winner replays to the right values" (Array.for_all2_exn got_2 expected ~f:approx);

  (* --- Run 3: arm B dies at its FIRST attempt, before its search reports anything. [?report] is
     positional, so consumers name arms by arrival order; the failed arm must still occupy its slot
     rather than let the surviving arm's report be attributed to it. --- *)
  let arms_reported = ref 0 in
  let reports = ref [] in
  let report r =
    Int.incr arms_reported;
    reports := r :: !reports
  in
  let ctx_3, routine_3 =
    with_injected_failure ~arms_reported ~at:1 ~message (fun () ->
        Train.tune_placements ~beam_width:2 ~rounds:0 ~repeats:1 ~cache_dir ~report
          (Context.auto ()) t2 comp Ir.Indexing.Empty)
  in
  let reports3 = List.rev !reports in
  p "an arm failing before it reports still occupies its slot" (List.length reports3 = 2);
  let arm_a3 = List.nth_exn reports3 0 and arm_b3 = List.nth_exn reports3 1 in
  p "the slot report is arm B's, not arm A's misattributed one"
    (arm_a3.Autotune.cache_hit && arm_b3.Autotune.partial);
  p "the slot report says the arm died before reporting"
    (Option.value_map arm_b3.Autotune.terminal_failure ~default:false ~f:(fun tf ->
         String.is_substring tf.Autotune.detail ~substring:"before the search reported"));
  let ctx_3 = Context.run ctx_3 routine_3 in
  p "arm A still ships when arm B dies before reporting"
    (Array.for_all2_exn (Context.get_values ctx_3 t2.Tensor.value) expected ~f:approx);

  (* --- A report-callback exception is the caller's, not the search's: it propagates instead of
     being reclassified as an arm failure. --- *)
  let raised =
    try
      let _ =
        Train.tune_placements ~beam_width:2 ~rounds:0 ~repeats:1 ~cache_dir
          ~report:(fun _ -> failwith "injected report callback failure")
          (Context.auto ()) t2 comp Ir.Indexing.Empty
      in
      false
    with Failure msg -> String.is_substring msg ~substring:"injected report callback failure"
  in
  p "a report-callback exception propagates instead of losing an arm" raised;

  (* --- A compiler invariant violation is a tuner bug, not a schedule that lost: containing it
     would let the process exit 0 with a shipped winner and the bug unmentioned. Same classes
     [Ir.Schedule_outcome.classify_raw] refuses to classify one level down. --- *)
  let attempts = ref 0 in
  let assertion_propagated =
    Exn.protect
      ~f:(fun () ->
        (Autotune.on_candidate_attempt :=
           fun _ ->
             Int.incr attempts;
             if !attempts = 2 then assert false);
        try
          let _ =
            Train.tune_placements ~beam_width:2 ~rounds:0 ~repeats:1 ~cache_dir
              (Context.auto ()) t2 comp Ir.Indexing.Empty
          in
          false
        with Assert_failure _ -> true)
      ~finally:(fun () -> Autotune.on_candidate_attempt := fun _ -> ())
  in
  p "a compiler assertion propagates instead of losing an arm" assertion_propagated
