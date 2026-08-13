(* gh-ocannl-514 phase 1: the matmul sketch family as a refinement tree.

   The flat sections below pin the family's seed enumeration — originally recorded against the
   hand-written enumeration before the tree refactor (which reproduced it list-for-list, order
   included: enumeration order reaches candidate timing order and dedup keep-first); the phase-2
   pre-compile refutations have since deliberately removed statically-doomed entries (e.g. the
   single-row-block whole-triple Grid form), pinned here as behavior. Synthetic limits keep every
   leg machine-independent — seeding is a pure function of the lowering, so the GPU legs
   enumerate (and twin: swizzle, pipeline depth) without any GPU present.

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
            | Sspace.Excluded (w, _) ->
                Stdio.printf "%s%s = %s  [excluded: %s]\n" indent level label w
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
  awkward_section "transposed-B cpu" ~is_gpu:false ~is_cpu:true ~limits:cpu_limits opt_t;
  (* Tight hardware limits: the staged operand tiles are a sound workgroup-memory floor, so
     geometries whose depth-1 tiles exceed the cap refute outright and dividing geometries whose
     doubled tiles exceed it refute their depth twins; a blocktile geometry's launch size
     (bm/tm * bn/tn threads) is statically known, so the thread cap refutes bm64 bn64 tm4 tn4
     (256 threads) at 128 — all pre-compile, where
     Schedule.check_hardware_limits_classified would otherwise reject candidate by candidate.
     6144 bytes fits bm16 bn32 bk32 (2048 + 4096) exactly at depth 1. *)
  let gpu_tight_limits =
    {
      gpu_full_limits with
      Ir.Backend_intf.max_workgroup_memory_bytes = Some 6144;
      max_threads_per_workgroup = Some 128;
      (* An advertised depth outside Schedule.apply_stage's implemented 1..2 range refutes
         rather than enumerating twins that fail every candidate compile. *)
      mma =
        Option.map gpu_full_limits.Ir.Backend_intf.mma ~f:(fun m ->
            { m with Ir.Backend_intf.mma_pipeline_depths = [ 2; 3 ] });
    }
  in
  awkward_section "gpu tight smem" ~is_gpu:true ~is_cpu:false ~limits:gpu_tight_limits opt;
  (* A wide output column extent: the unsplit B~ panel (bn = 0) of the (64, 0, 256) packed
     geometry spans 589824 bytes of tiles — above the 256 KiB stack/cache-economy threshold,
     which EXCLUDES (a driver may lift the policy) rather than refutes. *)
  let wn =
    NTDSL.init ~l:"wn" ~prec:Ir.Ops.single ~o:[ 64; 512 ]
      ~f:(fun idcs -> Float.of_int (((idcs.(0) * 512) + idcs.(1)) % 9) *. 0.25)
      ()
  in
  let%op wmm = av +* "ik;kj=>ij" wn in
  let opt_w = with_lowering ~name:"sft_wide" wmm in
  awkward_section "wide-N cpu" ~is_gpu:false ~is_cpu:true ~limits:cpu_limits opt_w;
  (* The lift operation: an Excluded child's payload is the same judgment with only that policy
     lifted, still subject to legality — the serial shape's economy-capped geometry recovers a
     leaf, while the Grid shapes' lifted payloads re-refute on the single-row-block rule. *)
  (match Autotune.matmul_sketch_tree ~is_gpu:false ~is_cpu:true ~limits:cpu_limits opt_w with
  | None -> Stdio.printf "wide-N: no site\n"
  | Some tree ->
      let rec lift_all t =
        match t with
        | Sspace.Leaf p -> Sspace.Leaf p
        | Sspace.Choice { level; children } ->
            Sspace.Choice
              {
                level;
                children =
                  List.map children ~f:(fun (l, c) ->
                      ( l,
                        match Sspace.lift_excluded c with
                        | Sspace.Child sub -> Sspace.Child (lazy (lift_all (Lazy.force sub)))
                        | c -> c ));
              }
      in
      Stdio.printf "wide-N leaves: %d; with exclusions lifted: %d\n"
        (List.length (Sspace.leaves tree))
        (List.length (Sspace.leaves (lift_all tree))));
  (* --- The B&B driver's contract on a synthetic tree (phase 4): the bound fathoms Child
     subtrees against the tightening threshold (equality fathoms — displacement needs strict
     improvement), Unknown children are NEVER fathomed even when their bound would, Refuted and
     Excluded children are never entered (an excluded 0-cost leaf must not leak into the
     minimum), and without a bound the walk degenerates to the flat first-best scan. --- *)
  let leafv name v = Sspace.Child (lazy (Sspace.Leaf (name, v))) in
  let choice level children = Sspace.Choice { level; children } in
  let syn =
    choice "top"
      [
        ("a", leafv "a" 5.);
        ("b", Sspace.Child (lazy (choice "bsub" [ ("b1", leafv "b1" 3.); ("b2", leafv "b2" 2.) ])));
        ( "u",
          Sspace.Unknown
            ("compile-settled", lazy (choice "usub" [ ("u1", leafv "u1" 1.) ])) );
        ("r", Sspace.Refuted "illegal");
        ("x", Sspace.Excluded ("policy", lazy (leafv "x" 0.)));
        ("c", Sspace.Child (lazy (choice "csub" [ ("c1", leafv "c1" 2.) ])));
      ]
  in
  let bound ~path:_ sub =
    match sub with
    | Sspace.Choice { level = "bsub"; _ } -> Some 2.5
    | Sspace.Choice { level = "usub"; _ } -> Some 100.
    | Sspace.Choice { level = "csub"; _ } -> Some 10.
    | _ -> None
  in
  let score (_, v) = Some v in
  let show_run name (best, stats) =
    Stdio.printf "%s: best=%s stats=%s\n" name
      (match best with
      | Some ((n, _), v) -> Printf.sprintf "%s@%.1f" n v
      | None -> "none")
      (Sexp.to_string_hum (Sspace.sexp_of_search_stats stats))
  in
  show_run "search with bounds" (Sspace.search ~bound ~incumbent:5.5 ~score syn);
  show_run "search without bounds" (Sspace.search ~incumbent:5.5 ~score syn);
  show_run "ties: first best wins"
    (Sspace.search ~score
       (choice "tie" [ ("t1", leafv "t1" 2.); ("t2", leafv "t2" 2.) ]));
  (* Cost-fathoming vs verdict-fathoming: a bounded Child subtree CONTAINING an Unknown below is
     correctly fathomed by cost — bound soundness quantifies over every completion independently
     of how verdicts resolve, so a subtree that cannot beat the incumbent even where legal is
     pruned. Only a directly encountered Unknown skips the bound (its dominant unknown is
     legality, where a bound over possibly-nonexistent completions is vacuous). *)
  let nested =
    choice "top"
      [
        ("good", leafv "good" 3.);
        ( "deep",
          Sspace.Child
            (lazy
              (choice "dsub"
                 [ ("du", Sspace.Unknown ("compile-settled", lazy (Sspace.Leaf ("du", 1.)))) ]))
        );
      ]
  in
  let bound ~path:_ = function Sspace.Choice { level = "dsub"; _ } -> Some 5. | _ -> None in
  show_run "cost bound fathoms above a nested Unknown" (Sspace.search ~bound ~score nested);
  (* Path-dependent bounds (gh-ocannl-514, the placement-space search): the bound receives the
     committed (level, label) vector down to and including the judged child — the partial vector
     the subtree stands for — so a floor that tightens as commitments accumulate can price the
     exact node being judged. Here the bound prices by the committed labels alone: committing
     "mat" costs 4 per level, so the mat/mat subtree (floor 8) is fathomed against the incumbent
     7 while mat/keep (floor 4) and keep/* (floor 0) are entered. (The fathomed leaf scores 1 —
     this synthetic bound is deliberately dishonest about it, pinning the mechanics and the
     documented caveat that an unsound bound prunes true winners.) *)
  let placementish =
    choice "n1"
      [
        ( "mat",
          Sspace.Child
            (lazy (choice "n2" [ ("mat", leafv "mm" 1.); ("keep", leafv "mk" 6.) ])) );
        ( "keep",
          Sspace.Child
            (lazy (choice "n2" [ ("mat", leafv "km" 6.5); ("keep", leafv "kk" 6.9) ])) );
      ]
  in
  let bound ~path _sub =
    Some (4. *. Float.of_int (List.count path ~f:(fun (_, label) -> String.equal label "mat")))
  in
  show_run "path-dependent bound fathoms the doubly-committed subtree"
    (Sspace.search ~bound ~incumbent:7. ~score placementish)
;
  (* --- Phase 5: the tile-size lattice and the non-uniform family bound --- *)
  (* The lattice hides behind its exclusion: the un-lifted tree keeps the curated leaves (the
     seed-list identity above already pinned that), and lifting turns the exclusion into interval
     boxes over every intrinsic-tile multiple — 8 bm values x 8 staged bk values on the 64^3
     site — whose leaves join the enumeration without disturbing the curated ones' order. *)
  let mm2 =
    (* A fresh lowering of the same 64^3 site: [opt] is still live above, but a dedicated name
       keeps this section's output self-contained. *)
    with_lowering ~name:"sft_lattice"
      (let%op l2 = av +* "ik;kj=>ij" bv in
       l2)
  in
  (match Autotune.matmul_sketch_tree ~is_gpu:true ~is_cpu:false ~limits:gpu_full_limits mm2 with
  | None -> Stdio.printf "lattice: no site detected\n"
  | Some tree ->
      let curated = List.length (Sspace.leaves tree) in
      let lifted = Autotune.lift_geometry_lattice tree in
      let lifted_leaves = List.length (Sspace.leaves lifted) in
      Stdio.printf "== lattice (64^3, tile 8x8x8) ==\n";
      Stdio.printf "curated leaves %d, lifted leaves %d (+%d lattice singletons), depth %d -> %d\n"
        curated lifted_leaves (lifted_leaves - curated) (Sspace.depth tree) (Sspace.depth lifted);
      Stdio.printf "the lattice exclusion carries the lift instructions: %b\n"
        (List.exists (Sspace.exclusions tree) ~f:(fun (_, w) ->
             String.equal w Autotune.geometry_lattice_witness));
      (* The certain-traffic increments that make the bound non-uniform: a committed staged
         geometry prices its operand tiles, an unstaged one prices nothing, and a lattice box
         prices its most favorable corner — monotone in refinement. *)
      let inc = Autotune.sketch_path_traffic_floor ~is_gpu:true ~limits:gpu_full_limits mm2 in
      let show_inc name path = Stdio.printf "  %-46s -> %d bytes\n" name (inc path) in
      Stdio.printf "certain-traffic increments along paths:\n";
      show_inc "curated staged bm16 bn32 bk32"
        [ ("pipeline", "tensorized"); ("geometry", "bm16 bn32 bk32") ];
      show_inc "curated unstaged bm16 bn32 bk0"
        [ ("pipeline", "tensorized"); ("geometry", "bm16 bn32 bk0") ];
      show_inc "lattice box bm 8..32, bk open"
        [ ("pipeline", "tensorized"); ("geometry", "lattice"); ("bm", "bm 8..32") ];
      show_inc "lattice box bm=32, bk 16..32"
        [ ("pipeline", "tensorized"); ("geometry", "lattice"); ("bm", "bm=32"); ("bk", "bk 16..32") ];
      show_inc "lattice singleton bm=32 bk=64"
        [ ("pipeline", "tensorized"); ("geometry", "lattice"); ("bm", "bm=32"); ("bk", "bk=64") ];
      (* Search over the lifted tree with the increment itself as the bound and an incumbent
         between the small and large boxes' floors: large-tile boxes fathom without expansion, so
         the walk scores a fraction of the lattice — the logarithmic-effective regime. The score
         never displaces (infinity), pinning that fathoming alone dispatches the boxes. *)
      let bound ~path _sub = Some (Float.of_int (inc path)) in
      let score _p = Some Float.infinity in
      let _best, stats = Sspace.search ~bound ~incumbent:6000. ~score lifted in
      Stdio.printf
        "lifted search at incumbent 6000: %d expanded, %d scored, %d fathomed, %d refuted, %d \
         excluded\n"
        stats.Sspace.st_expanded stats.Sspace.st_scored stats.Sspace.st_fathomed
        stats.Sspace.st_refuted stats.Sspace.st_excluded);
  (* Corner-judged box refutations: a workgroup-memory cap that admits only the smallest staged
     tiles refutes the large-tile half-boxes at their most favorable corner, pre-expansion — the
     "tile-size interval whose minimum footprint exceeds shared memory" fathom of the issue. *)
  let gpu_tiny_smem =
    { gpu_full_limits with Ir.Backend_intf.max_workgroup_memory_bytes = Some 2048 }
  in
  match Autotune.matmul_sketch_tree ~is_gpu:true ~is_cpu:false ~limits:gpu_tiny_smem mm2 with
  | None -> Stdio.printf "lattice under 2KB smem: no site detected\n"
  | Some tree ->
      let lifted = Autotune.lift_geometry_lattice tree in
      let box_refutations =
        List.filter (Sspace.refutations lifted) ~f:(fun (path, _) ->
            List.exists path ~f:(fun (_, label) ->
                String.is_substring label ~substring:"..")
            && List.exists path ~f:(fun (_, label) -> String.equal label "lattice"))
      in
      Stdio.printf "== lattice under a 2048-byte workgroup-memory cap ==\n";
      Stdio.printf "surviving lattice leaves %d; box-level refutations %d, e.g.:\n"
        (List.length
           (List.filter (Sspace.enumerate lifted) ~f:(fun (path, _) ->
                List.exists path ~f:(fun (_, label) -> String.equal label "lattice"))))
        (List.length box_refutations);
      List.iter (List.take box_refutations 2) ~f:(fun (path, w) ->
          Stdio.printf "  %s: %s\n"
            (String.concat ~sep:" > " (List.map path ~f:(fun (l, v) -> l ^ "=" ^ v)))
            w)
;
  (* Review round (Codex P1 on PR #327): the lattice minima and the open-corner pricing must come
     from the same per-format tile selection the tree builds with — a canonical [mma_tile]
     coarser than the selected format's (CUDA's TF32 shape) must not inflate the open-axis
     corner. Canonical 16x16x16, f32 format tile 8x8x8: the open-corner increment prices 8s, and
     the lattice enumerates 8-step multiples. *)
  let gpu_coarse_canonical =
    {
      Ir.Backend_intf.no_hardware_limits with
      mma =
        Some
          {
            Ir.Backend_intf.mma_simd_width = 32;
            mma_tile = (16, 16, 16);
            mma_format_tiles = [ ((f32, f32, f32), (8, 8, 8)) ];
            mma_staged_layouts = [];
            mma_pipeline_depths = [];
          };
    }
  in
  match
    Autotune.matmul_sketch_tree ~is_gpu:true ~is_cpu:false ~limits:gpu_coarse_canonical mm2
  with
  | None -> Stdio.printf "coarse-canonical: no site detected\n"
  | Some tree ->
      let lifted = Autotune.lift_geometry_lattice tree in
      let lattice_leaves =
        List.filter (Sspace.enumerate lifted) ~f:(fun (path, _) ->
            List.exists path ~f:(fun (_, label) -> String.equal label "lattice"))
      in
      let inc =
        Autotune.sketch_path_traffic_floor ~is_gpu:true ~limits:gpu_coarse_canonical mm2
      in
      Stdio.printf "== coarse canonical mma_tile 16^3, format tile 8^3 ==\n";
      Stdio.printf "lattice leaves %d (8-step multiples of both axes: %b)\n"
        (List.length lattice_leaves)
        (List.for_all lattice_leaves ~f:(fun (_, p) ->
             p.Autotune.sk_bm % 8 = 0 && p.Autotune.sk_bk % 8 = 0)
        && List.exists lattice_leaves ~f:(fun (_, p) -> p.Autotune.sk_bm = 8)
        && List.exists lattice_leaves ~f:(fun (_, p) -> p.Autotune.sk_bk = 8));
      Stdio.printf "  open-corner lattice increment (format 8s, not canonical 16s) -> %d bytes\n"
        (inc [ ("pipeline", "tensorized"); ("geometry", "lattice") ])
