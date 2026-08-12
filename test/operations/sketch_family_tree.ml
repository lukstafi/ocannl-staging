(* gh-ocannl-514 phase 1: the matmul sketch family as a refinement tree.

   The flat sections below pin the family's seed enumeration — recorded against the hand-written
   enumeration BEFORE the tree refactor, so the factoring must reproduce it list-for-list (order
   included: enumeration order reaches candidate timing order and dedup keep-first). Synthetic
   limits keep every leg machine-independent — seeding is a pure function of the lowering, so the
   GPU legs enumerate (and twin: swizzle, pipeline depth) without any GPU present.

   The site is a 64x64x64 f32 matmul: every tile geometry in the curated menus divides it, so the
   menus enumerate in full. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module LL = Ir.Low_level
module Asgns = Ir.Assignments

let named name (comp : Asgns.comp) : Asgns.comp =
  { comp with asgns = Asgns.Block_comment (name, comp.asgns) }

(* Compile [tensor], capturing the base lowering for the (pure) seeding calls. *)
let with_lowering ~name tensor =
  let captured = ref None in
  let _ctx, _routine =
    Context.compile
      ~lowered_transform:(fun opt ->
        captured := Some opt;
        opt)
      (Context.auto ())
      (named name (Train.forward tensor))
      Ir.Indexing.Empty
  in
  Option.value_exn !captured

let cpu_limits = { Ir.Backend_intf.no_hardware_limits with simd_vector_bytes = 32 }

let f32 = Ir.Backend_intf.Mma_f32

let gpu_plain_limits =
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

(* Staged-layout and pipelining advertisements switch on the twin seeding: a swizzled twin per
   staged seed whose tile rows span power-of-two 16-byte units, and a depth twin per fully
   dividing staged seed. *)
let gpu_full_limits =
  {
    Ir.Backend_intf.no_hardware_limits with
    mma =
      Some
        {
          Ir.Backend_intf.mma_simd_width = 32;
          mma_tile = (8, 8, 8);
          mma_format_tiles = [ ((f32, f32, f32), (8, 8, 8)) ];
          mma_staged_layouts = [ ((f32, f32, f32), Ir.Backend_intf.Mma_swizzled_b128) ];
          mma_pipeline_depths = [ 2 ];
        };
  }

let show (p : Autotune.sketch_params) =
  Printf.sprintf
    "gpu:%b mma:%b simd:%-2d bm:%-3d bn:%-3d bk:%-3d tm:%d tn:%d hoist:%b grid:%b packrest:%b \
     epi:%b swz:%s depth:%d"
    p.Autotune.sk_gpu p.Autotune.sk_mma p.Autotune.sk_simd p.Autotune.sk_bm p.Autotune.sk_bn
    p.Autotune.sk_bk p.Autotune.sk_tm p.Autotune.sk_tn p.Autotune.sk_hoist p.Autotune.sk_grid
    p.Autotune.sk_pack_rest p.Autotune.sk_epilogue
    (match p.Autotune.sk_swizzle with
    | None -> "-"
    | Some LL.Swizzle_b128 -> "b128"
    | Some LL.Swizzle_elem -> "elem")
    p.Autotune.sk_depth

let section name ~is_gpu ~is_cpu ~limits opt =
  let seeds = Autotune.sketch_seed_params ~is_gpu ~is_cpu ~limits opt in
  Stdio.printf "== %s: %d seeds ==\n" name (List.length seeds);
  List.iteri seeds ~f:(fun i p -> Stdio.printf "%2d  %s\n" i (show p));
  seeds

(* The refinement-tree view of the same family (gh-ocannl-514 phases 1-2): decision levels with
   commitment-dependent domains, whose leaves are exactly the flat enumeration above, and whose
   children carry verdicts — [refuted]/[excluded] branches print their witness and contribute no
   leaves, [unknown] branches never fathom. *)
module Sspace = Ir.Schedule_space

let rec print_tree ~indent tree =
  match tree with
  | Sspace.Leaf p -> Stdio.printf "%s* %s\n" indent (show p)
  | Sspace.Choice { level; children } ->
      if List.is_empty children then Stdio.printf "%s%s: infeasible\n" indent level
      else
        List.iter children ~f:(fun (label, child) ->
            match child with
            | Sspace.Child sub ->
                Stdio.printf "%s%s = %s\n" indent level label;
                print_tree ~indent:(indent ^ "  ") (Lazy.force sub)
            | Sspace.Unknown (w, sub) ->
                Stdio.printf "%s%s = %s  [unknown: %s]\n" indent level label w;
                print_tree ~indent:(indent ^ "  ") (Lazy.force sub)
            | Sspace.Excluded w -> Stdio.printf "%s%s = %s  [excluded: %s]\n" indent level label w
            | Sspace.Refuted w -> Stdio.printf "%s%s = %s  [refuted: %s]\n" indent level label w)

(* The three verdict collectors: a shape's pre-compilation decline explanations (gh-ocannl-479),
   the policy-suppressed branches a driver could re-propose, and the branches only candidate
   compilation settles. *)
let verdict_reports tree =
  let pp (path, w) =
    Stdio.printf "  %s: %s\n"
      (String.concat ~sep:" > " (List.map path ~f:(fun (l, v) -> l ^ "=" ^ v)))
      w
  in
  let section name entries =
    if not (List.is_empty entries) then (
      Stdio.printf "-- %s --\n" name;
      List.iter entries ~f:pp)
  in
  section "refuted" (Sspace.refutations tree);
  section "excluded" (Sspace.exclusions tree);
  section "unknown" (Sspace.unknowns tree)

let tree_section name ~is_gpu ~is_cpu ~limits opt seeds =
  match Autotune.matmul_sketch_tree ~is_gpu ~is_cpu ~limits opt with
  | None -> Stdio.printf "== %s tree: no site detected ==\n" name
  | Some tree ->
      Stdio.printf "== %s tree: %d choice nodes, depth %d ==\n" name (Sspace.count_choices tree)
        (Sspace.depth tree);
      print_tree ~indent:"" tree;
      let paths = Sspace.enumerate tree in
      (match List.last paths with
      | Some (path, _) ->
          Stdio.printf "last leaf's decision path: %s\n"
            (String.concat ~sep:" > "
               (List.map path ~f:(fun (level, label) -> level ^ "=" ^ label)))
      | None -> Stdio.printf "no leaves\n");
      Stdio.printf "tree leaves = flat enumeration: %b\n"
        (List.equal
           (fun a b -> String.equal (show a) (show b))
           (Sspace.leaves tree) seeds)

(* Awkward sites (phase 2): the tree's verdicts explain, before any compilation, why branches
   propose nothing — where the flat enumeration silently dropped them. Only the collector reports
   print here; the leaves are still the seeds the flat API proposes. *)
let awkward_section name ~is_gpu ~is_cpu ~limits opt =
  match Autotune.matmul_sketch_tree ~is_gpu ~is_cpu ~limits opt with
  | None -> Stdio.printf "== %s: no site detected ==\n" name
  | Some tree ->
      let seeds = Autotune.sketch_seed_params ~is_gpu ~is_cpu ~limits opt in
      Stdio.printf "== %s: %d seeds ==\n" name (List.length seeds);
      Stdio.printf "tree leaves = flat enumeration: %b\n"
        (List.equal
           (fun a b -> String.equal (show a) (show b))
           (Sspace.leaves tree) seeds);
      verdict_reports tree

let () =
  let nn = 64 in
  (* A non-hoistable, B hoistable (host-init-backed constant): the exactly-one-hoistable case, so
     the hoisted, hoisted-grid AND mixed grid-outermost ([sk_pack_rest]) packing shapes all
     enumerate. *)
  let av =
    NTDSL.init ~l:"av" ~prec:Ir.Ops.single ~o:[ nn; nn ]
      ~f:(fun idcs -> (Float.of_int (((idcs.(0) * nn) + idcs.(1)) % 13) -. 6.) *. 0.25)
      ()
  in
  let bvv = Array.init (nn * nn) ~f:(fun x -> (Float.of_int (x % 17) -. 8.) *. 0.125) in
  let bv = TDSL.ndarray bvv ~label:[ "bv" ] ~output_dims:[ nn; nn ] () in
  let%op mm = av +* "ik;kj=>ij" bv in
  let opt = with_lowering ~name:"sft_mm" mm in
  let cpu_seeds = section "cpu simd32" ~is_gpu:false ~is_cpu:true ~limits:cpu_limits opt in
  let _ = section "gpu plain" ~is_gpu:true ~is_cpu:false ~limits:gpu_plain_limits opt in
  let gpu_seeds = section "gpu staged+depth" ~is_gpu:true ~is_cpu:false ~limits:gpu_full_limits opt in
  tree_section "cpu simd32" ~is_gpu:false ~is_cpu:true ~limits:cpu_limits opt cpu_seeds;
  tree_section "gpu staged+depth" ~is_gpu:true ~is_cpu:false ~limits:gpu_full_limits opt gpu_seeds;
  (* --- Awkward sites: witnesses for what is NOT proposed --- *)
  (* 20^3: no curated blocktile geometry divides it; unstaged mma is refuted while padded staged
     mma survives (gh-ocannl-485 zero-fringe pads); CPU Grid shapes at bm=64 lack row blocks. *)
  let wa =
    NTDSL.init ~l:"wa" ~prec:Ir.Ops.single ~o:[ 20; 20 ]
      ~f:(fun idcs -> Float.of_int (((idcs.(0) * 20) + idcs.(1)) % 7) *. 0.5)
      ()
  in
  let wb =
    NTDSL.init ~l:"wb" ~prec:Ir.Ops.single ~o:[ 20; 20 ]
      ~f:(fun idcs -> Float.of_int (((idcs.(0) * 20) + idcs.(1)) % 5) -. 2.)
      ()
  in
  let%op awk = wa +* "ik;kj=>ij" wb in
  let opt_awk = with_lowering ~name:"sft_awk" awk in
  awkward_section "awkward 20^3 gpu" ~is_gpu:true ~is_cpu:false ~limits:gpu_full_limits opt_awk;
  awkward_section "awkward 20^3 cpu" ~is_gpu:false ~is_cpu:true ~limits:cpu_limits opt_awk;
  (* Half-precision operands: the CPU register tiling requires uniform f32/f64, and the synthetic
     GPU capability advertises only the f32 format triple. *)
  let ha =
    NTDSL.init ~l:"ha" ~prec:Ir.Ops.half ~o:[ nn; nn ]
      ~f:(fun idcs -> Float.of_int (((idcs.(0) * nn) + idcs.(1)) % 7) *. 0.5)
      ()
  in
  let hb =
    NTDSL.init ~l:"hb" ~prec:Ir.Ops.half ~o:[ nn; nn ]
      ~f:(fun idcs -> Float.of_int (((idcs.(0) * nn) + idcs.(1)) % 5) -. 2.)
      ()
  in
  let%op hmm = ha +* "ik;kj=>ij" hb in
  let opt_h = with_lowering ~name:"sft_half" hmm in
  awkward_section "half-prec cpu" ~is_gpu:false ~is_cpu:true ~limits:cpu_limits opt_h;
  awkward_section "half-prec gpu" ~is_gpu:true ~is_cpu:false ~limits:gpu_full_limits opt_h;
  (* Transposed B (k on its minor axis): whole-triple and the hoisted-only Grid shape read B in
     place, which the register tiling statically declines; packing shapes normalize the layout. *)
  let tb =
    NTDSL.init ~l:"tb" ~prec:Ir.Ops.single ~o:[ nn; nn ]
      ~f:(fun idcs -> Float.of_int (((idcs.(0) * nn) + idcs.(1)) % 11) *. 0.25)
      ()
  in
  let%op tmm = av +* "ik;jk=>ij" tb in
  let opt_t = with_lowering ~name:"sft_tb" tmm in
  awkward_section "transposed-B cpu" ~is_gpu:false ~is_cpu:true ~limits:cpu_limits opt_t
