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
   "arm B lost".

   gh-ocannl-564 extends the same suite one level down, to what a timing run does BEFORE it
   dispatches: a failure of [Context.run]'s pre-dispatch validation writes nothing, so it must be a
   per-candidate decline that the search survives — not the fatal that condemned the lineage when it
   arrived tagged [Launch] with no backend able to attribute it. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module SC = Ir.Schedule_cache
module SO = Ir.Schedule_outcome

let p name b = Stdio.printf "%s: %b\n" name b
let approx a b = Float.(abs (a - b) < 1e-4)
let n = 8

let clean_cache dir =
  if Stdlib.Sys.file_exists dir && Stdlib.Sys.is_directory dir then
    Array.iter (Stdlib.Sys.readdir dir) ~f:(fun f ->
        Stdlib.Sys.remove (Stdlib.Filename.concat dir f))

(* The injection is global state on a library ref, so it is restored unconditionally: a leaked
   raiser would fail every later autotune call in this process. *)
let with_injected_failure ?exn ~arms_reported ~at ~message f =
  let attempts = ref 0 in
  (Autotune.on_candidate_attempt :=
     fun label ->
       (* Arm A reports exactly once, when its search ends, so this fires within arm B only. *)
       if !arms_reported >= 1 then (
         Int.incr attempts;
         if !attempts = at then
           raise
             (Option.value exn
                ~default:(Failure (Printf.sprintf "%s at candidate %s" message label)))));
  Exn.protect ~f ~finally:(fun () -> Autotune.on_candidate_attempt := fun _ -> ())

(* gh-ocannl-564: the same discipline at the pre-dispatch validation seam. [at] counts preflights
   across the whole call — 1 is arm A's baseline timing, so 2 is the first candidate arm A times —
   and the count is returned so a scenario can assert the injection actually fired rather than
   passing vacuously on a search that timed fewer candidates than expected. *)
let with_injected_preflight_failure ~at ~raise_it f =
  let preflights = ref 0 in
  (Autotune.on_candidate_preflight :=
     fun _routine_name ->
       Int.incr preflights;
       if !preflights = at then raise_it ());
  let result = Exn.protect ~f ~finally:(fun () -> Autotune.on_candidate_preflight := fun _ -> ()) in
  (result, !preflights)

let preflight_declines (r : Autotune.report) =
  List.filter r.Autotune.declines ~f:(fun d ->
      match d.Autotune.key with SO.Unclassified_key (SO.Preflight, _) -> true | _ -> false)

let declined_with (reports : Autotune.report list) ~substring =
  List.exists reports ~f:(fun r ->
      List.exists (preflight_declines r) ~f:(fun d ->
          d.Autotune.count >= 1
          && List.exists d.Autotune.sample_details ~f:(String.is_substring ~substring)))

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

  (* --- Run 3: arm B dies at its FIRST attempt — its base compile, before a search exists. [?report]
     is positional, so consumers name arms by arrival order; the failed arm must still occupy its
     slot rather than let the surviving arm's report be attributed to it. The report is the tuner's
     own (it reports on every path), so it carries a structured phase rather than a guess. --- *)
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
  p "an arm failing before its search starts still occupies its slot" (List.length reports3 = 2);
  let arm_a3 = List.nth_exn reports3 0 and arm_b3 = List.nth_exn reports3 1 in
  p "the slot report is arm B's, not arm A's misattributed one"
    (arm_a3.Autotune.cache_hit && arm_b3.Autotune.partial);
  p "the pre-search report names the injected failure at a structured phase"
    (Option.value_map arm_b3.Autotune.terminal_failure ~default:false ~f:(fun tf ->
         String.is_substring tf.Autotune.detail ~substring:message
         && Ir.Schedule_outcome.equal_phase tf.Autotune.phase Ir.Schedule_outcome.Transform));
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
  p "a compiler assertion propagates instead of losing an arm" assertion_propagated;

  (* --- The callback failure and the arm failure are the SAME nullary exception. [Exit] is a
     singleton value, so "was this the callback's exception?" cannot be answered by physical
     identity: here the tuner swallows the callback's [Exit] on its partial-report path and raises
     the arm's own [Exit], which identity would misread as the callback's and propagate, losing the
     completed arm. Cache off so arm B searches instead of replaying. --- *)
  let arms_reported = ref 0 in
  let collision_contained =
    with_injected_failure ~exn:Stdlib.Exit ~arms_reported ~at:4 ~message:"unused" (fun () ->
        match
          Train.tune_placements ~beam_width:2 ~rounds:0 ~repeats:1 ~cache_dir:""
            ~report:(fun r ->
              Int.incr arms_reported;
              if r.Autotune.partial then raise Stdlib.Exit)
            (Context.auto ()) t2 comp Ir.Indexing.Empty
        with
        | ctx_c, routine_c ->
            let ctx_c = Context.run ctx_c routine_c in
            Array.for_all2_exn (Context.get_values ctx_c t2.Tensor.value) expected ~f:approx
        | exception Stdlib.Exit -> false)
  in
  p "an arm failing with the same exception its callback raised is still contained"
    collision_contained;

  (* --- Containment stops where the damage is shared: a failure that poisons the lineage the arms
     search in leaves the sibling unable to execute a single timing run, so it must propagate rather
     than burn a search proving that. The injected failure poisons the lineage the way an
     unattributed launch failure would. --- *)
  let ctx_p = Context.auto () in
  let reports = ref 0 in
  let attempts = ref 0 in
  let poisoned_propagated =
    Exn.protect
      ~f:(fun () ->
        (Autotune.on_candidate_attempt :=
           fun _ ->
             Int.incr attempts;
             if !attempts = 1 then (
               Context.poison_lineage ctx_p ~routine_name:"injected"
                 (Failure "injected lineage poisoning");
               raise (Failure "injected lineage poisoning")));
        match
          Train.tune_placements ~beam_width:2 ~rounds:0 ~repeats:1 ~cache_dir:""
            ~report:(fun _ -> Int.incr reports)
            ctx_p t2 comp Ir.Indexing.Empty
        with
        | _ -> false
        | exception Failure msg -> String.is_substring msg ~substring:"injected lineage poisoning")
      ~finally:(fun () -> Autotune.on_candidate_attempt := fun _ -> ())
  in
  p "a failure that poisons the shared lineage propagates instead of trying the sibling"
    poisoned_propagated;
  p "the sibling arm was not attempted on a poisoned lineage" (!reports = 1);

  (* --- The mirror case: the arm that poisons is the LATER one, so an earlier arm's winner is in
     hand — but without a [timing_ctx] the arms searched the caller's own lineage, so that winner
     can never run. Handing it back would report success for a routine guaranteed to raise. --- *)
  let ctx_q = Context.auto () in
  let arms_reported = ref 0 in
  let attempts = ref 0 in
  let poisoned_winner_refused =
    Exn.protect
      ~f:(fun () ->
        (Autotune.on_candidate_attempt :=
           fun _ ->
             if !arms_reported >= 1 then (
               Int.incr attempts;
               if !attempts = 3 then (
                 Context.poison_lineage ctx_q ~routine_name:"injected"
                   (Failure "injected late poisoning");
                 raise (Failure "injected late poisoning"))));
        match
          Train.tune_placements ~beam_width:2 ~rounds:0 ~repeats:1 ~cache_dir:""
            ~report:(fun _ -> Int.incr arms_reported)
            ctx_q t2 comp Ir.Indexing.Empty
        with
        | _ -> false
        | exception Failure msg -> String.is_substring msg ~substring:"poisoned")
      ~finally:(fun () -> Autotune.on_candidate_attempt := fun _ -> ())
  in
  p "a winner is not shipped out of a lineage a later arm poisoned" poisoned_winner_refused;

  (* --- An interrupt raised inside a report callback is about the process, not the arm: the
     best-effort reporting on the failure path must not swallow it. --- *)
  let attempts = ref 0 in
  let interrupt_propagated =
    Exn.protect
      ~f:(fun () ->
        (Autotune.on_candidate_attempt :=
           fun _ ->
             Int.incr attempts;
             if !attempts = 3 then raise (Failure "injected failure under interrupt"));
        match
          Train.tune_placements ~beam_width:2 ~rounds:0 ~repeats:1 ~cache_dir:""
            ~report:(fun r -> if r.Autotune.partial then raise Stdlib.Sys.Break)
            (Context.auto ()) t2 comp Ir.Indexing.Empty
        with
        | _ -> false
        | exception Stdlib.Sys.Break -> true)
      ~finally:(fun () -> Autotune.on_candidate_attempt := fun _ -> ())
  in
  p "an interrupt raised by a report callback is not swallowed" interrupt_propagated;

  (* --- The timing runs condemn a lineage they cannot attribute a failure in, which is only sound
     if pre-dispatch validation happens outside that judgement: an unsatisfied dependency writes
     nothing and is the caller's to fix and retry, so it must leave the lineage usable. --- *)
  let ctx_r = Context.auto () in
  let ctx_r1, r1 = Context.compile ctx_r comp Ir.Indexing.Empty in
  let _ctx_r2, r2 = Context.compile ctx_r1 comp Ir.Indexing.Empty in
  let preflight_rejected =
    match Context.check_runnable ctx_r1 r2 with
    | () -> false
    | exception Failure _ -> true
  in
  p "pre-dispatch validation rejects an unexecuted dependency" preflight_rejected;
  p "a pre-dispatch rejection leaves the lineage usable"
    (Option.is_none (Context.poisoned_failure ctx_r1));
  let ctx_r1 = Context.run ctx_r1 r1 in
  p "and the retry succeeds once the dependency has executed"
    (match Context.check_runnable ctx_r1 r2 with () -> true | exception _ -> false);

  (* --- gh-ocannl-564: the candidate half of the same claim. A pre-dispatch rejection arrives
     inside the candidate timing's failure boundary, which is tagged [Launch] and — with no backend
     verdict, which is every C backend — used to make it fatal: the lineage was condemned and the
     search died for a mistake nothing had yet acted on. It is now its own phase, so it is a
     per-candidate decline: census-visible, lineage intact, search completes.

     The rejections themselves are produced by real validation of real routines (the injection only
     decides WHICH candidate meets one): the causes are properties of the lineage and the bindings,
     so a genuine one fails every candidate of every arm at once and could not show a search that
     survived it. The whole-search form of that is the last scenario below. --- *)
  let negative_control () =
    let reports = ref [] in
    let ctx_n, routine_n =
      Train.tune_placements ~beam_width:2 ~rounds:0 ~repeats:1 ~cache_dir:""
        ~report:(fun r -> reports := r :: !reports)
        (Context.auto ()) t2 comp Ir.Indexing.Empty
    in
    let ctx_n = Context.run ctx_n routine_n in
    ( List.rev !reports,
      Array.for_all2_exn (Context.get_values ctx_n t2.Tensor.value) expected ~f:approx )
  in
  let control_reports, control_ships = negative_control () in
  p "control: an uninjected search declines nothing at pre-dispatch validation"
    (List.for_all control_reports ~f:(fun r -> List.is_empty (preflight_declines r)));
  p "control: an uninjected search completes and ships"
    (control_ships && List.for_all control_reports ~f:(fun r -> not r.Autotune.partial));

  (* An unsatisfied execution dependency, taken from a routine that genuinely has one: [dep_r2]
     reads what [dep_r1] writes and [dep_r1] has not run, so [check_runnable] raises exactly what
     the timing run's own validation would. *)
  let dep_ctx = Context.auto () in
  let dep_ctx1, _dep_r1 = Context.compile dep_ctx comp Ir.Indexing.Empty in
  let _dep_ctx2, dep_r2 = Context.compile dep_ctx1 comp Ir.Indexing.Empty in
  let ctx_d = Context.auto () in
  let reports_d = ref [] in
  let (ctx_d', routine_d), preflights_d =
    with_injected_preflight_failure ~at:2
      ~raise_it:(fun () -> Context.check_runnable dep_ctx1 dep_r2)
      (fun () ->
        Train.tune_placements ~beam_width:2 ~rounds:0 ~repeats:1 ~cache_dir:""
          ~report:(fun r -> reports_d := r :: !reports_d)
          ctx_d t2 comp Ir.Indexing.Empty)
  in
  let reports_d = List.rev !reports_d in
  p "an unsatisfied dependency reached a candidate's timing run" (preflights_d >= 2);
  p "it is a decline in the census, at the pre-dispatch phase"
    (declined_with reports_d ~substring:"unexecuted dependencies");
  p "the search that declined it completed"
    (List.for_all reports_d ~f:(fun r -> not r.Autotune.partial));
  p "and did not condemn the lineage" (Option.is_none (Context.poisoned_failure ctx_d));
  let ctx_d' = Context.run ctx_d' routine_d in
  p "a winner still ships and computes the right values"
    (Array.for_all2_exn (Context.get_values ctx_d' t2.Tensor.value) expected ~f:approx);

  (* An out-of-range static binding, likewise from real bind-time validation. [set_test_bindings]
     only ever writes in-range values, so this is the shape of the mistake a caller makes: a binding
     the tuner leaves alone because it declares no range, carrying a value the index width rejects.
     A different exception constructor from the dependency case, so it also pins that the phase
     rather than the exception is what classifies these. *)
  let osym, _ = Ir.Indexing.get_static_symbol ~static_range:4 Ir.Indexing.Empty in
  let ctx_b = Context.auto () in
  let reports_b = ref [] in
  let (ctx_b', routine_b), preflights_b =
    with_injected_preflight_failure ~at:2
      ~raise_it:(fun () -> Ir.Indexing.validate_lowered_bindings [ (osym, ref 9) ])
      (fun () ->
        Train.tune_placements ~beam_width:2 ~rounds:0 ~repeats:1 ~cache_dir:""
          ~report:(fun r -> reports_b := r :: !reports_b)
          ctx_b t2 comp Ir.Indexing.Empty)
  in
  let reports_b = List.rev !reports_b in
  p "an out-of-range static binding reached a candidate's timing run" (preflights_b >= 2);
  p "it too is a decline in the census, at the pre-dispatch phase"
    (declined_with reports_b ~substring:"exceeds its declared range");
  p "the search that declined it completed"
    (List.for_all reports_b ~f:(fun r -> not r.Autotune.partial));
  p "and did not condemn the lineage" (Option.is_none (Context.poisoned_failure ctx_b));
  let ctx_b' = Context.run ctx_b' routine_b in
  p "a winner still ships and computes the right values"
    (Array.for_all2_exn (Context.get_values ctx_b' t2.Tensor.value) expected ~f:approx);

  (* --- Containment stops at the one pre-dispatch condition that is not fixable. A poisoned lineage
     has no restore (gh-ocannl-536), so every later timing run in it is dead too: declining the
     candidate would decline every remaining candidate for the same terminal reason and then report
     a completed search. The poisoning happens AFTER the baseline was timed, so a winner is in hand
     and the search has something to wrongly report success with. --- *)
  let ctx_z = Context.auto () in
  let attempts = ref 0 in
  let reports_z = ref [] in
  let poisoned_stops_the_search =
    Exn.protect
      ~f:(fun () ->
        (Autotune.on_candidate_attempt :=
           fun _ ->
             Int.incr attempts;
             (* Poisons without raising: the lineage's own state must stop the search, through the
                timing run's pre-dispatch check rather than through an injected exception. *)
             if !attempts = 2 then
               Context.poison_lineage ctx_z ~routine_name:"injected"
                 (Failure "injected prior poisoning"));
        match
          Train.tune_placements ~beam_width:2 ~rounds:0 ~repeats:1 ~cache_dir:""
            ~report:(fun r -> reports_z := r :: !reports_z)
            ctx_z t2 comp Ir.Indexing.Empty
        with
        | _ -> false
        | exception Failure msg -> String.is_substring msg ~substring:"poisoned")
      ~finally:(fun () -> Autotune.on_candidate_attempt := fun _ -> ())
  in
  p "a poisoned lineage is not declined like a fixable pre-dispatch failure"
    poisoned_stops_the_search;
  (* Positionally arm A — the arm the poisoning happened inside. Read as "some report", this passes
     without the fix too: arm B's baseline timing hits the same poisoned lineage and reports it,
     while arm A goes on declining every remaining candidate, times nothing, and reports a COMPLETED
     search that shipped an untuned fallback out of a dead lineage. *)
  p "the arm it happened in reports a terminal failure, not a completed search"
    (Option.value_map (List.hd (List.rev !reports_z)) ~default:false ~f:(fun r ->
         r.Autotune.partial
         && Option.value_map r.Autotune.terminal_failure ~default:false ~f:(fun tf ->
                String.is_substring tf.Autotune.detail ~substring:"poisoned")));

  (* --- The genuine whole-search form, injection-free: the caller hands the tuner a lineage holding
     an unexecuted compile of the same computation, so every routine the tuner compiles inherits an
     unsatisfied dependency and both arms fail at their baseline's validation. That failure is the
     caller's to fix — which it can only be if the lineage survives it. --- *)
  let ctx_u = Context.auto () in
  let ctx_u1, r_unrun = Context.compile ctx_u comp Ir.Indexing.Empty in
  let unrunnable_raised =
    match
      Train.tune_placements ~beam_width:2 ~rounds:0 ~repeats:1 ~cache_dir:"" ctx_u1 t2 comp
        Ir.Indexing.Empty
    with
    | _ -> false
    | exception Failure msg -> String.is_substring msg ~substring:"unexecuted dependencies"
  in
  p "a tune whose lineage cannot run its own baseline fails with the validation error"
    unrunnable_raised;
  p "the lineage it failed in is not condemned" (Option.is_none (Context.poisoned_failure ctx_u1));
  let ctx_u2 = Context.run ctx_u1 r_unrun in
  let ctx_u3, routine_u =
    Train.tune_placements ~beam_width:2 ~rounds:0 ~repeats:1 ~cache_dir:"" ctx_u2 t2 comp
      Ir.Indexing.Empty
  in
  let ctx_u3 = Context.run ctx_u3 routine_u in
  p "and the retry in it succeeds once the dependency has executed"
    (Array.for_all2_exn (Context.get_values ctx_u3 t2.Tensor.value) expected ~f:approx)
