(* gh-ocannl-514, the placement decision surface: the enablement prior, the ranking, and the
   partial-vector floor.

   The matmul reads a policy-virtual operand ([mbs], a pointwise scale of [mb]) and its result
   ([mc]) inlines into the relu consumer, so the default-placement lowering carries no
   recognizable matmul site at all — the site exists only in the all-materialized specialization
   of the decision surface, reading [mbs] into [mc]. With synthetic GPU limits advertising an
   (f32, f32, f32) mma format tile (classification is a pure function of the lowerings, so no GPU
   is needed — the sketch_family_tree harness), the enablement prior must promote exactly the
   site's flip candidates: materializing them is what makes the tensorized family expressible.

   The decoy [us] (a pointwise scale read with a 32-fold per-cell multiplicity by a broadcast
   consumer) carries a larger recompute-cost bound than the site candidates — the gh-558 shape,
   where cost ordering buries the family-unlocking flips below a candidate that unlocks nothing
   and enablement ordering does not.

   The floor closure is asserted monotone in the committed materializations, under the envelope
   constants pinned by the rule's command line (cc carries none of its own).

   Printed candidate lists carry names, flip kinds, costs and enablement marks: the ranking is
   deterministic given the computation and the synthetic limits. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module LL = Ir.Low_level
module Tn = Ir.Tnode
module Asgns = Ir.Assignments

let p name b = Stdio.printf "%s: %b\n" name b

let named name (comp : Asgns.comp) : Asgns.comp =
  { comp with asgns = Asgns.Block_comment (name, comp.asgns) }

let f32 = Ir.Backend_intf.Mma_f32

let gpu_limits =
  {
    Ir.Backend_intf.no_hardware_limits with
    mma =
      Some
        {
          Ir.Backend_intf.mma_simd_width = 32;
          mma_tile = (8, 8, 8);
          mma_format_tiles = [ ((f32, f32, f32), (8, 8, 8)) ];
          mma_staged_layouts = [];
          mma_pipeline_depths = [];
        };
  }

let n = 8
let m = 32

let () =
  let mav = Array.init (n * n) ~f:(fun i -> Float.of_int (i % 7) *. 0.5) in
  let mbv = Array.init (n * n) ~f:(fun i -> Float.of_int (i % 5) -. 2.) in
  let ma = TDSL.ndarray mav ~label:[ "ma" ] ~input_dims:[ n ] ~output_dims:[ n ] () in
  let mb = TDSL.ndarray mbv ~label:[ "mb" ] ~input_dims:[ n ] ~output_dims:[ n ] () in
  let uv = Array.init m ~f:(fun i -> Float.of_int (i % 3) *. 0.25) in
  let wv = Array.init (m * m) ~f:(fun i -> Float.of_int (i % 11) *. 0.125) in
  let u = TDSL.ndarray uv ~label:[ "u" ] ~output_dims:[ m ] () in
  let w = TDSL.ndarray wv ~label:[ "w" ] ~output_dims:[ m; m ] () in
  (* The mma-site half: mbs and mc are policy-virtual. *)
  let%op mbs = mb *. 0.5 in
  let%op mc = ma * mbs in
  let%op t2 = relu mc in
  (* The decoy half: us is policy-virtual, read broadcast by every row of w. *)
  let%op us = u *. 2.0 in
  let%op d2 = w +* "ij; j => ij" us in
  let%op total = (t2 ++ "ij => 0") + (d2 ++ "ij => 0") in
  let comp = named "ps" (Train.forward total) in
  let ctx = Context.auto () in
  let base = Context.lowered_for_decisions ctx comp Ir.Indexing.Empty in
  let candidates = base.LL.flip_candidates in
  let to_materialize =
    List.filter_map candidates ~f:(fun fc ->
        match fc.LL.fc_flip with `Materialize -> Some fc.LL.fc_tn | `Inline -> None)
  in
  let allmat =
    Context.lowered_for_decisions ~materialized:to_materialize ctx comp Ir.Indexing.Empty
  in
  let enablement, disablement =
    Autotune.placement_enablement ~limits:gpu_limits ~static_indices:[] ~base ~allmat
  in
  let mem set (t : Tensor.t) = Set.mem set t.Tensor.value in
  p "the enablement set is nonempty" (not (Set.is_empty enablement));
  p "it contains the site operand mbs" (mem enablement mbs);
  p "it contains the site destination mc" (mem enablement mc);
  p "it does not contain the decoy us" (not (mem enablement us));
  p "no site is eligible under default placements (empty disablement)" (Set.is_empty disablement);
  let show ordering =
    let ranked =
      Autotune.rank_flip_candidates ~ordering ~enablement ~disablement candidates
    in
    List.iter ranked ~f:(fun fc ->
        Stdio.printf "  %-11s %-12s cost %-5d%s\n"
          (match fc.LL.fc_flip with `Materialize -> "materialize" | `Inline -> "inline")
          (Tn.debug_name fc.LL.fc_tn) fc.LL.fc_recompute_cost
          (if Set.mem enablement fc.LL.fc_tn then "  [enablement]" else ""));
    ranked
  in
  Stdio.printf "cost ranking:\n";
  let by_cost = show `Cost in
  Stdio.printf "enablement ranking:\n";
  let by_enablement = show `Enablement in
  let is_en fc = Set.mem enablement fc.LL.fc_tn in
  p "cost ranking buries the enablement candidates below the decoy"
    (match by_cost with fc :: _ -> not (is_en fc) | [] -> false);
  p "enablement ranking puts the family-unlocking materialize flip first"
    (match by_enablement with
    | fc :: _ -> is_en fc && (match fc.LL.fc_flip with `Materialize -> true | `Inline -> false)
    | [] -> false);
  p "enablement ranking puts the family-breaking inline flip last"
    (match List.last by_enablement with
    | Some fc ->
        is_en fc && (match fc.LL.fc_flip with `Inline -> true | `Materialize -> false)
    | None -> false);
  (* The floor closure, under the rule's pinned envelope: monotone in the commitments, and
     strictly above the empty commitment once the site nodes' traffic is certain. *)
  let surface = Autotune.placement_surface ~ordering:`Enablement ctx comp Ir.Indexing.Empty in
  let f0 = surface.Autotune.ps_floor_ms ~materialized:[] in
  let f1 = surface.Autotune.ps_floor_ms ~materialized:[ mbs.Tensor.value ] in
  let f2 = surface.Autotune.ps_floor_ms ~materialized:[ mbs.Tensor.value; mc.Tensor.value ] in
  let ge a b = match (a, b) with Some a, Some b -> Float.(a >= b) | _ -> false in
  p "the floor is present under the pinned envelope" (Option.is_some f0);
  p "committing mbs does not lower the floor" (ge f1 f0);
  p "committing mc on top does not lower it either" (ge f2 f1);
  p "the two commitments strictly raise the floor"
    (match (f2, f0) with Some f2, Some f0 -> Float.(f2 > f0) | _ -> false)
