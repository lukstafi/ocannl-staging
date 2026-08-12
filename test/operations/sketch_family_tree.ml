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

(* The refinement-tree view of the same family (gh-ocannl-514 phase 1): decision levels with
   commitment-dependent domains, whose leaves are exactly the flat enumeration above. An empty
   choice is an infeasible node — every completion was filtered out. *)
let rec print_tree ~indent tree =
  match tree with
  | Ir.Schedule_space.Leaf p -> Stdio.printf "%s* %s\n" indent (show p)
  | Ir.Schedule_space.Choice { level; children } ->
      if List.is_empty children then Stdio.printf "%s%s: infeasible\n" indent level
      else
        List.iter children ~f:(fun (label, sub) ->
            Stdio.printf "%s%s = %s\n" indent level label;
            print_tree ~indent:(indent ^ "  ") (Lazy.force sub))

let tree_section name ~is_gpu ~is_cpu ~limits opt seeds =
  match Autotune.matmul_sketch_tree ~is_gpu ~is_cpu ~limits opt with
  | None -> Stdio.printf "== %s tree: no site detected ==\n" name
  | Some tree ->
      Stdio.printf "== %s tree: %d choice nodes, depth %d ==\n" name
        (Ir.Schedule_space.count_choices tree)
        (Ir.Schedule_space.depth tree);
      print_tree ~indent:"" tree;
      let paths = Ir.Schedule_space.enumerate tree in
      (match List.last paths with
      | Some (path, _) ->
          Stdio.printf "last leaf's decision path: %s\n"
            (String.concat ~sep:" > "
               (List.map path ~f:(fun (level, label) -> level ^ "=" ^ label)))
      | None -> Stdio.printf "no leaves\n");
      Stdio.printf "tree leaves = flat enumeration: %b\n"
        (List.equal
           (fun a b -> String.equal (show a) (show b))
           (Ir.Schedule_space.leaves tree) seeds)

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
  tree_section "gpu staged+depth" ~is_gpu:true ~is_cpu:false ~limits:gpu_full_limits opt gpu_seeds
