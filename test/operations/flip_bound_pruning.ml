(* gh-ocannl-514: bound pruning in [Train.tune_placements]' flip chain, one level above the
   phase-4b sketch gate. The dune rule pins the cc backend, [autotune_bound_pruning=true] and a
   tiny envelope (model_peak_* = 1e3), making the partial-placement-vector roofline floor
   ([Autotune.placement_surface.ps_floor_ms]) astronomically larger than any measured time — so
   every [`Materialize] flip candidate is fathomed before its nested search, deterministically on
   any machine, without consuming the budget. [`Inline] flips are never floor-pruned (committing
   to inline tightens nothing), so the number of flip searches observed through [flip_report]
   equals the number of [`Inline] candidates the surface reports, and the shipped routine still
   computes correct values (it is the plain A/B winner or an inline refinement of it).

   The control for "the same flips are measured when pruning is off" is the existing
   inline_flip_tune test, which runs the same driver without the gate. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module LL = Ir.Low_level
module Asgns = Ir.Assignments

let p name b = Stdio.printf "%s: %b\n" name b
let approx a b = Float.(abs (a -. b) < 1e-4)

let named name (comp : Asgns.comp) : Asgns.comp =
  { comp with asgns = Asgns.Block_comment (name, comp.asgns) }

let n = 8

let () =
  let mav = Array.init (n * n) ~f:(fun i -> Float.of_int (i % 7) *. 0.5) in
  let mbv = Array.init (n * n) ~f:(fun i -> Float.of_int (i % 5) -. 2.) in
  let ma = TDSL.ndarray mav ~label:[ "ma" ] ~input_dims:[ n ] ~output_dims:[ n ] () in
  let mb = TDSL.ndarray mbv ~label:[ "mb" ] ~input_dims:[ n ] ~output_dims:[ n ] () in
  let%op mc = ma * mb in
  let%op t2 = relu mc in
  ignore mc;
  let comp = named "fbp" (Train.forward t2) in
  (* Reference values from a plain compile. *)
  let ctx_ref, routine_ref = Context.compile (Context.auto ()) comp Ir.Indexing.Empty in
  let ctx_ref = Context.run ctx_ref routine_ref in
  let expected = Context.get_values ctx_ref t2.Tensor.value in
  (* The surface the chain will walk: how many flips of each kind exist. *)
  let surface = Autotune.placement_surface (Context.auto ()) comp Ir.Indexing.Empty in
  let mat_flips, inline_flips_on_surface =
    List.partition_tf surface.Autotune.ps_candidates ~f:(fun fc ->
        match fc.LL.fc_flip with `Materialize -> true | `Inline -> false)
  in
  p "the surface reports at least one materialize flip" (List.length mat_flips >= 1);
  (* Budget above the whole surface: every candidate is either measured or fathomed. *)
  let budget = List.length surface.Autotune.ps_candidates + 1 in
  let arm_reports = ref [] in
  let flip_reports = ref [] in
  let ctx_t, routine_t =
    Train.tune_placements ~beam_width:2 ~rounds:0 ~repeats:1 ~cache_dir:""
      ~report:(fun r -> arm_reports := r :: !arm_reports)
      ~flip_report:(fun r -> flip_reports := r :: !flip_reports)
      ~inline_flips:budget (Context.auto ()) t2 comp Ir.Indexing.Empty
  in
  let ctx_t = Context.run ctx_t routine_t in
  let got = Context.get_values ctx_t t2.Tensor.value in
  p "tuned routine values match the plain compile" (Array.for_all2_exn got expected ~f:approx);
  p "the public report callback keeps the positional A/B contract" (List.length !arm_reports = 2);
  p "every materialize flip was fathomed: only the inline flips reached a search"
    (List.length !flip_reports = List.length inline_flips_on_surface)
