(* gh-ocannl-755: [Autotune.time_routine] times a candidate against one of two objectives, and they
   do not crown the same candidate. [Isolated] is one launch plus one host synchronization -- the
   latency of a lone dispatch. [Queued] dispatches a calibrated batch back to back, synchronizes
   once and divides, so what is left is what the kernel sustains inside a stream that already has
   work in it, which is what a training step presents to every kernel of a layer.

   What is pinned here is the MECHANISM, not the ranking (the ranking is a device measurement; the
   tables are in the issue and the harness that produced them is [bin/projection_shape_bench.ml]).
   Two failures would silently undo the change while every timing still looked plausible: a batch
   depth that collapses to 1, which turns a queued search back into an isolated one, and a reading
   that is per BATCH rather than per launch, which inflates every candidate by the same factor and
   so leaves the ranking -- and only the ranking -- looking right.

   The dispatch counts are read off the computation itself: the routine is [n[0] += 1] on a
   materialized node, so after a timing call [n[0]] IS the number of launches that call made. That
   is an exact count of the thing under test, not a proxy for it. *)

open Base
module LL = Ir.Low_level
module Tn = Ir.Tnode
module Idx = Ir.Indexing
module SC = Ir.Schedule_cache

let p = Verdict.p

let backend () = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")

(* {1 The calibration policy, without a device} *)

(* [queued_batch_depth] is what decides whether queued timing queues anything at all, and its two
   boundaries are the ones a regression crosses silently. Written as the estimate a caller could
   plausibly measure, paired with the depth the policy owes it -- a routine at or above the batch
   target must batch at 1 (there is nothing to amortize, and the two modes then agree by
   construction), a microsecond routine must be capped, and everything between must scale. *)
let depth_cases =
  [
    ("a routine far slower than the batch target", 100., 1);
    ("a routine at the batch target", 10., 1);
    ("a routine at half the batch target", 5., 2);
    ("a 0.1 ms routine", 0.1, 100);
    ("a 0.05 ms routine, exactly at the cap", 0.05, 200);
    ("a 1 us routine, past the cap", 0.001, 200);
    ("an infinitely slow routine", Float.infinity, 1);
    ("a clock that resolved nothing (zero)", 0., 200);
    ("a clock that resolved nothing (nan)", Float.nan, 200);
    (* Saturates rather than raising: the ratio here is past the integer range, so a cap applied
       after the float-to-int conversion would raise instead of capping. *)
    ("a subnormal estimate", Float.min_positive_subnormal_value, 200);
  ]

let () =
  Stdio.printf "== queued batch depth ==\n";
  Verdict.p_all "every calibration estimate gets the depth the policy owes it" depth_cases
    ~f:(fun (what, est_ms, want) ->
      let got = Autotune.queued_batch_depth ~est_ms in
      if got <> want then
        Stdio.eprintf "  %s: est %g ms -> depth %d, expected %d\n%!" what est_ms got want;
      got = want);
  (* The floor and the cap are the two claims a scaling-only implementation would still pass, so
     they are also asserted as the properties they are, over the same population. *)
  Verdict.p_all "no calibration estimate ever yields a depth below 1" depth_cases
    ~f:(fun (_, est_ms, _) -> Autotune.queued_batch_depth ~est_ms >= 1);
  Verdict.p_all "no calibration estimate ever yields a depth above the cap" depth_cases
    ~f:(fun (_, est_ms, _) -> Autotune.queued_batch_depth ~est_ms <= 200)

(* {1 The setting's spelling} *)

let () =
  Stdio.printf "\n== autotune_timing spelling ==\n";
  let reads s want =
    match Autotune.timing_of_setting s with
    | got -> Poly.equal got want
    | exception _ -> false
  in
  p "\"queued\" selects the queued objective" (reads "queued" Autotune.Queued);
  p "\"isolated\" selects the isolated objective" (reads "isolated" Autotune.Isolated);
  p "the spelling is case- and space-insensitive" (reads " ISOLATED\n" Autotune.Isolated);
  (* The negative control: a misspelling must be refused rather than falling back to a mode the
     caller did not ask for -- silently timing under the other objective is the failure this whole
     issue is about. *)
  p "a misspelling is refused rather than defaulted"
    (match Autotune.timing_of_setting "batched" with
    | _ -> false
    | exception Invalid_argument _ -> true
    | exception _ -> false)

(* {1 The instrument, on a routine that counts its own launches} *)

(* [n[0] += 1] over a one-element materialized node: every dispatch adds exactly one, and f32 counts
   integers exactly far past any dispatch count these loops can reach. Deliberately trivial, so that
   the same source is a scalar kernel on a GPU backend too -- this test runs on whichever backend is
   configured, and a nest heavy enough to be interesting would run as a single work item there. *)
let counter_node = Ll_test.node_factory ~first_id:9900 ~dims:[| 1 |] () "gh755_counter"

let counter_routine () =
  Ll_test.materialize counter_node;
  let idx = Ll_test.fixed 0 in
  let bump =
    Ll_test.set_at counter_node idx
      (Ll_test.add (Ll_test.get counter_node [| idx |]) (Ll_test.c 1.))
  in
  let o = Ll_test.optimize ~materialized:[ counter_node ] ~name:"gh755_counter" bump in
  let ctx, routine = Ll_test.link ~name:"gh755_counter" o in
  (o, ctx, routine)

type reading = { ms : float; wall_ms : float; dispatches : int }

(* Held so the cache-key section below asks about the SAME lowering the instrument measured, rather
   than minting a second one whose canonical form it would have to argue is equivalent. *)
let measured : (LL.optimized * Context.t) option ref = ref None

let () =
  Stdio.printf "\n== the instrument's two modes ==\n";
  let opt, ctx, routine = counter_routine () in
  let ctx = Context.set_values ctx counter_node [| 0. |] in
  measured := Some (opt, ctx);
  let count () = Int.of_float (Context.get_values ctx counter_node).(0) in
  let measure timing =
    let before = count () in
    let c0 = Mtime_clock.counter () in
    let ms = Autotune.time_routine ~repeats:3 ~timing ctx routine in
    let wall_ms = Mtime.Span.to_float_ns (Mtime_clock.count c0) /. 1e6 in
    { ms; wall_ms; dispatches = count () - before }
  in
  (* Isolated first: its reading is also the estimate the gate below asks the policy about. *)
  let iso = measure Autotune.Isolated in
  let que = measure Autotune.Queued in
  Stdio.eprintf
    "  (not part of the golden) isolated %.6f ms over %d dispatches in %.1f ms wall; queued %.6f ms \
     over %d dispatches in %.1f ms wall\n\
     %!"
    iso.ms iso.dispatches iso.wall_ms que.ms que.dispatches que.wall_ms;
  let finite r = Float.is_finite r.ms && Float.is_positive r.ms in
  p "both modes returned a positive finite per-launch time" (finite iso && finite que);
  (* One launch per timed run, at least [repeats] runs and at most the 64-run top-up cap, plus the
     warmup. Two-sided: the upper bound would also admit a loop that stopped at the warmup. *)
  p "isolated timing dispatched one launch per timed run, warmup included"
    (iso.dispatches >= 4 && iso.dispatches <= 65);
  (* Depth > 1 is what queued mode IS. Gated on the policy's own answer for this machine's isolated
     reading rather than on a threshold restated here: on a machine where one dispatch already costs
     a whole batch target the claim is vacuously true, and a vacuous [true] must not read like a
     verified one. *)
  let batches_here = Autotune.queued_batch_depth ~est_ms:iso.ms > 1 in
  let claim = "queued timing dispatched more launches than isolated timing did" in
  if batches_here then p claim (que.dispatches > iso.dispatches)
  else Verdict.skipped ~backend:(backend ()) claim;
  (* Per launch, not per batch. A reading that forgot to divide by the depth would be about [depth]
     times the mean cost of a launch in that call -- up to 200x -- so the two-sided factor of 3
     leaves ample room for the difference between a minimum and a mean while refusing that. *)
  let mean r = r.wall_ms /. Float.of_int (max 1 r.dispatches) in
  let per_launch r = Float.(r.ms <= 3. * mean r && r.ms >= mean r / 3.) in
  Verdict.pass_fail "isolated reading is a per-launch time" (per_launch iso)
    ~detail:(fun () -> Printf.sprintf "%.6f ms vs mean %.6f ms" iso.ms (mean iso));
  Verdict.pass_fail "queued reading is a per-launch time" (per_launch que)
    ~detail:(fun () -> Printf.sprintf "%.6f ms vs mean %.6f ms" que.ms (mean que));
  (* Amortizing a round trip can only remove time, so a queued reading above the isolated one is the
     instrument reporting the wrong quantity, not a slow machine. The factor absorbs the noise a
     min-of-N leaves; the point of the claim is the direction. *)
  Verdict.pass_fail "queued timing does not read above isolated timing"
    Float.(que.ms <= iso.ms * 2.)
    ~detail:(fun () -> Printf.sprintf "queued %.6f ms vs isolated %.6f ms" que.ms iso.ms);
  (* Queued timing costs MORE than isolated timing on a fast routine -- isolated stops at the
     64-run cap, queued runs to the wall budget -- but it is bounded by that budget rather than
     scaling with the batch depth. A loop that counted launches instead of wall time would spend up
     to [max_queue_depth] times longer, which is what this refuses. *)
  Verdict.pass_fail "queued timing's wall cost is bounded by the budget, not by the batch depth"
    Float.(que.wall_ms <= (iso.wall_ms * 3.) + 100.)
    ~detail:(fun () -> Printf.sprintf "queued %.1f ms vs isolated %.1f ms wall" que.wall_ms iso.wall_ms)

(* {1 The objective is part of the cache identity} *)

(* gh-ocannl-755, Codex P1 on PR #512: the two objectives crown different candidates, so an entry
   stored under one is not the answer to a search asking the other -- and its stored times are
   readings of a different quantity that a replay would copy into the reading process's report under
   that process's label. Keying on the objective is what keeps the regimes apart, and it is also
   what stops a warm cache from replaying isolated-crowned winners forever and defeating the new
   default outright. Pinned at the key rather than by driving a search: the key is the mechanism --
   a mismatched entry lives in a different file and is never looked up at all. *)
let () =
  Stdio.printf "\n== the objective in the cache key ==\n";
  let opt, ctx = Option.value_exn !measured in
  let canon = SC.canonicalize ~static_indices:[] opt in
  let limits = Context.hardware_limits ctx and backend = Context.backend_name ctx in
  let key objective = SC.cache_key ~objective ~limits canon ~backend in
  p "the cache key is stable within one objective"
    (String.equal (key "queued") (key "queued"));
  p "the cache key separates the two timing objectives"
    (not (String.equal (key "isolated") (key "queued")));
  (* Derived, not restated: a caller that resolved no mode of its own must key exactly as one that
     resolved the configured mode, or a test's hand-built entry would sit under a key no search
     looks up. *)
  p "an omitted objective keys as the configured one"
    (String.equal (SC.cache_key ~limits canon ~backend) (key (SC.objective_tag ())));
  (* The tag a key carries is the mode's own spelling, so a report's objective and the entry that
     stored its times name the same thing. *)
  p "the key's objective spelling round-trips through the mode"
    (List.for_all [ Autotune.Isolated; Autotune.Queued ] ~f:(fun m ->
         Poly.equal (Autotune.timing_of_setting (Autotune.timing_string m)) m))
