(* gh-ocannl-550, accumulation half: a schedule search's live artifacts must be bounded by the beam,
   not by candidates processed.

   The failure this pins needs no GPU even though its consequence does. On CUDA the accumulation
   exhausted a 12 GB card a fifth of the way through a cold tf32 gpt2_mini search
   (benchmarks/report-gh550-cuda.md); on [cc] the very same artifacts accumulate — one context and
   one working pool per candidate — and {!Ir.Alloc_census} counts them, so the growth is
   host-observable here while the out-of-memory is not.

   The observation point is [Autotune.on_candidate_attempt], which fires BEFORE a candidate's compile
   and therefore after the previous candidate has been released: exactly "between candidates". What
   is asserted is the shape of the sequence, not an absolute number — pools per candidate and
   contexts per compile are implementation details, but "the k-th sample does not exceed the first
   sample by more than the beam holds" is the property gh-ocannl-550 is about. The census deltas are
   taken against the first sample so that the caller's own contexts and the reference compile do not
   enter the count.

   The two sequences this separates, measured on cc over an 18-candidate search (beam width 2):
   released, the live-pool delta reads 0,1,2,2,2,2,2,3,4,4,4,... and flattens at the beam's
   steady state; not released, it reads 0,1,2,3,...,17 — exactly one per candidate, to the end. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules

let p name b = Stdio.printf "%s: %b\n" name b
let n = 16
let beam_width = 2

let clean_cache dir =
  if Stdlib.Sys.file_exists dir && Stdlib.Sys.is_directory dir then
    Array.iter (Stdlib.Sys.readdir dir) ~f:(fun f ->
        Stdlib.Sys.remove (Stdlib.Filename.concat dir f))

(* Samples of the census taken between candidates, oldest first. *)
let sample_between_candidates f =
  let samples = ref [] in
  (Autotune.on_candidate_attempt :=
     fun _label -> samples := Ir.Alloc_census.snapshot () :: !samples);
  let result =
    Exn.protect ~f ~finally:(fun () -> Autotune.on_candidate_attempt := fun _ -> ())
  in
  (result, List.rev !samples)

let () =
  let mav = Array.init (n * n) ~f:(fun i -> Float.of_int (i % 7) *. 0.5) in
  let mbv = Array.init (n * n) ~f:(fun i -> Float.of_int (i % 5) -. 2.) in
  let ma = TDSL.ndarray mav ~label:[ "acr_ma" ] ~input_dims:[ n ] ~output_dims:[ n ] () in
  let mb = TDSL.ndarray mbv ~label:[ "acr_mb" ] ~input_dims:[ n ] ~output_dims:[ n ] () in
  let%op mc = ma * mb in
  let%op t2 = relu mc in
  ignore mc;
  let comp = Train.forward t2 in

  (* Reference values from a plain compile, so the release cannot be "correct" by having broken the
     search: a tuner that freed the winner's buffers would still satisfy every count below. *)
  let ctx_ref, routine_ref = Context.compile (Context.auto ()) comp Ir.Indexing.Empty in
  let ctx_ref = Context.run ctx_ref routine_ref in
  let expected = Context.get_values ctx_ref t2.Tensor.value in

  let cache_dir = "autotune_candidate_release_cache" in
  clean_cache cache_dir;
  let (ctx_t, routine_t), samples =
    sample_between_candidates (fun () ->
        Autotune.tune ~beam_width ~rounds:1 ~repeats:1 ~cache_dir (Context.auto ()) comp
          Ir.Indexing.Empty)
  in
  (* A search that timed one or two candidates would satisfy every bound below vacuously. *)
  p "the search attempted enough candidates for accumulation to show"
    (List.length samples >= 8);

  let first = List.hd_exn samples in
  let deltas ~f = List.map samples ~f:(fun s -> f s - f first) in
  let pool_deltas = deltas ~f:Ir.Alloc_census.live_pools in
  let ctx_deltas = deltas ~f:Ir.Alloc_census.live_contexts in
  let max_of = List.fold ~init:0 ~f:max in

  (* The absolute bound: one candidate in flight, plus what the beam retains, plus the running best.
     Deliberately generous — how many pools one compile needs is an implementation detail, and a
     backend needing two per candidate is not a regression of this property. *)
  let bound = 4 * (beam_width + 2) in
  p "live pools between candidates stay bounded" (max_of pool_deltas <= bound);
  p "live contexts between candidates stay bounded" (max_of ctx_deltas <= bound);

  (* And the scale-free form, which is the one that actually discriminates: the second half of the
     search must not hold measurably more than the first half did. Accumulation fails this whatever
     the per-candidate footprint is, and whatever the constant above is loosened to. *)
  let halve l =
    let k = List.length l / 2 in
    (List.take l k, List.drop l k)
  in
  let pools_early, pools_late = halve pool_deltas in
  let ctx_early, ctx_late = halve ctx_deltas in
  let slack = beam_width + 2 in
  p "live pools do not grow across the search" (max_of pools_late <= max_of pools_early + slack);
  p "live contexts do not grow across the search" (max_of ctx_late <= max_of ctx_early + slack);

  (* Freeing has to be actual freeing, not just a dropped reference: the census counts a pool freed
     only where the backend's [free_pool] ran. *)
  let last = List.last_exn samples in
  p "candidates were released, not merely dropped"
    (last.Ir.Alloc_census.pools_freed > 0
    && last.Ir.Alloc_census.contexts_released >= List.length samples - bound);

  let ctx_t = Context.run ctx_t routine_t in
  let got = Context.get_values ctx_t t2.Tensor.value in
  p "the winner still ships and computes the right values"
    (Array.for_all2_exn got expected ~f:(fun a b -> Float.(abs (a - b) < 1e-4)));

  (* And the exit sweep: with the search over and its routine in hand, nothing else it compiled is
     still held. [live_pools] is read against the reference compile's own footprint, so this compares
     the state after tuning against the state before the search's first candidate. *)
  let after = Ir.Alloc_census.snapshot () in
  p "the search holds nothing beyond its winner once it has returned"
    (Ir.Alloc_census.live_pools after - Ir.Alloc_census.live_pools first <= bound)
