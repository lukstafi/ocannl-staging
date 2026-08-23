(* gh-ocannl-728: does splitting a matmul site's axis into a pre-split pair cost throughput, at a
   fixed tile geometry and fixed total work?

   The gpt2_mini q/k/v and out projections run at ~37% of gfx1151's sgemm peak while the FFN
   up-projection runs at ~85%. Two explanations were on the table: (H1) the batch loop is a
   pre-split ROW loop the tile cannot span, so every block sees M = 128 where the FFN sites were
   believed to see M = 1024; (H2) the projection kernels are simply small (134 MFLOP each against
   the FFN's 537) and launch/occupancy-bound at any tiling.

   This bench builds the shapes by hand at the SAME total FLOPs, on the SAME device, under the
   SAME sketch geometries, and times them warm. Each site is
   [d[b.., m, n..] += w[n.., k..] * x[b.., m, k..]]; the groups are:

   - A: 134.2 MFLOP, N = K = 256, 1024 rows split as B x M for B in {1,2,4,8} -- equal work, equal
     block count at every geometry, only the row split differs. This is the leg that isolates
     M-per-block from everything else.
   - B: the FFN up-projection's own shape (N = 1024, 537 MFLOP), merged and batched.
   - C: 134 / 268 / 537 / 1074 MFLOP at a fixed shape, batched and merged -- the size leg.
   - D: the out projection, whose weight carries two input axes (a multi-axis contraction,
     gh-ocannl-683), merged and batched.
   - E: the 2x2 factorial over the two pre-splits at constant FLOPs -- rows merged (1024) or split
     (8 x 128) against the column axis merged (256) or split into 8 heads of 32, which is the
     column structure the gpt2_mini lowering actually gives q/k/v.
   - P: a 1024^3 square GEMM -- the family's own ceiling, where no shape question arises.

   Two sites appear twice: A_rows1024 = E_qkv_rows1_heads1 and A_b8x128 = E_qkv_rows8_heads1.
   That is deliberate -- each is measured twice within one run, so a run reports its own session
   drift and a shape effect can be judged against it.

   Candidates per site: the untuned shipped default, then every geometry of the GPU blocktile
   sketch family applied as the pure IR transform it is; mode [tune] instead runs a full
   [Autotune.tune] search per site with [~search:true] and the disk cache DISABLED, so neither
   shape can replay the other's cached winner and a configuration that disabled searching cannot
   return the untuned default under a tuned label.

   Timing is ROUND-INTERLEAVED, not site-by-site: one round per base tile geometry ([bgrid] twins
   included, since a merged site's plain arm and a batched site's [bgrid] arm are the pair being
   compared), inside which every arm is timed batch by batch in a rotation. A monotone session
   drift then lands on all the arms of a comparison alike; timing one site to completion before
   the next puts the drift straight into the difference under test, and reversing the site order
   only moves that bias. Two statistics per arm: [repeats] dispatches queued back-to-back with one
   sync (what a kernel sustains inside a step, and the summary's statistic, taken at the MEDIAN
   batch), and the tuner's own one-dispatch-one-sync minimum, which reads up to 2.6x higher.

   Every candidate's whole output is compared cell by cell against a host-computed oracle -- built
   straight from the input formulas, so it is independent of the compiler under test -- and the
   inputs are chosen so that f32 and f64 accumulation agree exactly whatever order either uses.
   Each line carries the launch dimensions the schedule actually produced, so "same geometry" is
   read off the kernel. Any cell that fails parity, fails to compile or run, or (in [tune] mode)
   did not actually search, is counted and the process exits nonzero: a blank in the column the
   caller asked for is a failed experiment, not a missing number.

   Usage (bin/ cwd trap: pin the backend, and run from a directory holding an ocannl_config):
     OCANNL_BACKEND=hip _build/default/bin/projection_shape_bench.exe \
       [repeats] [batches] [group] [order] [mode]
   Defaults 50 repeats, 5 timing batches, group "all" (a/b/c/d/e/p/abd/abde/all), order fwd (or
   rev), mode seeds (or tune, or both). *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module LL = Ir.Low_level
module Sched = Ir.Schedule
module Asgns = Ir.Assignments

(* Flushed per line: a long remote run should be readable while it is still going. *)
let p fmt =
  Printf.ksprintf
    (fun s ->
      Stdio.print_string s;
      Stdio.Out_channel.flush Stdio.stdout)
    fmt

let named name (comp : Asgns.comp) : Asgns.comp =
  { comp with asgns = Asgns.Block_comment (name, comp.asgns) }

(* [offset + stride * (flat index mod modulus)] over row-major [dims]: varies along every axis
   whose extent is not a multiple of [modulus]. Products are multiples of 1/8 and partial sums
   stay far below 2^24, so f32 addition is exact in any order and parity is bitwise. *)
let cycle ~dims ~modulus ~offset ~stride idcs =
  let flat = Array.foldi dims ~init:0 ~f:(fun i acc d -> (acc * d) + (idcs.(i) % d)) in
  offset +. (stride *. Float.of_int (flat % modulus))

(* The lowering alone; the context this mints is released rather than left to the pool tables,
   which strongly retain device slabs (docs/agent-notes/backend-memory.md). *)
let capture fwd =
  let captured = ref None in
  let ctx, _r =
    Context.compile
      ~lowered_transform:(fun opt ->
        captured := Some opt;
        opt)
      (Context.auto ()) fwd Ir.Indexing.Empty
  in
  Context.release ctx;
  Option.value_exn ~here:[%here] !captured

let geom_label (q : Autotune.sketch_params) =
  Printf.sprintf "%s%s %dx%dx%d/%dx%d%s%s%s%s%s"
    (if q.sk_mma then "mma-" else "")
    (if q.sk_gpu then "gpu" else "cpu")
    q.sk_bm q.sk_bn q.sk_bk q.sk_tm q.sk_tn
    (if q.sk_depth > 1 then Printf.sprintf " pd%d" q.sk_depth else "")
    (if q.sk_hoist then " hoist" else "")
    (if q.sk_grid then " grid" else "")
    (if q.sk_pack_rest then " packrest" else "")
    (if q.sk_batch_grid then " bgrid" else "")

let now () = Float.of_int63 (Time_now.nanoseconds_since_unix_epoch ()) /. 1e9

(* One site: a batched matmul [d[bs.., m, j] += a[bs.., m, kk..] * w[j, kk..]]. [ks] is the
   weight's input-axis list -- a singleton for the q/k/v shape, a pair for the out projection's
   multi-axis contraction. *)
type site = { tag : string; bs : int list; m : int; ns : int list; ks : int list }

let prod = List.fold ~init:1 ~f:( * )

let flops s =
  2.0 *. Float.of_int (prod s.bs) *. Float.of_int s.m *. Float.of_int (prod s.ns)
  *. Float.of_int (prod s.ks)

let build s =
  let wdims = Array.of_list (s.ns @ s.ks) in
  let adims = Array.of_list (s.bs @ [ s.m ] @ s.ks) in
  let w =
    NTDSL.init ~l:("w_" ^ s.tag) ~prec:Ir.Ops.single ~o:s.ns ~i:s.ks
      ~f:(cycle ~dims:wdims ~modulus:11 ~offset:(-5.5) ~stride:0.5)
      ()
  in
  let a =
    NTDSL.init ~l:("a_" ^ s.tag) ~prec:Ir.Ops.single ~b:(s.bs @ [ s.m ]) ~o:s.ks
      ~f:(cycle ~dims:adims ~modulus:13 ~offset:0.25 ~stride:0.25)
      ()
  in
  let%op d = w * a in
  d


(* The independent oracle: the site's whole output computed on the host straight from the [cycle]
   formulas, in the same row-major layout the device writes. Every candidate is compared against
   it cell by cell, not through a scalar digest -- a checksum with a repeating coefficient cannot
   see a permutation of cells one period apart, and a reference taken from the default pipeline
   cannot see a defect the default pipeline shares. Products are multiples of 1/8 and every
   partial sum stays below 2^24, so the f64 accumulation here and the device's f32 accumulation
   agree exactly whatever order either uses, and the comparison is [Float.equal]. *)
let oracle s =
  let nn = prod s.ns and kk = prod s.ks and rows = prod s.bs * s.m in
  let wv =
    Array.init (nn * kk) ~f:(fun i -> -5.5 +. (0.5 *. Float.of_int (i % 11)))
  and av = Array.init (rows * kk) ~f:(fun i -> 0.25 +. (0.25 *. Float.of_int (i % 13))) in
  let d = Array.create ~len:(rows * nn) 0.0 in
  for r = 0 to rows - 1 do
    for j = 0 to nn - 1 do
      let acc = ref 0.0 in
      for k = 0 to kk - 1 do
        acc := !acc +. (av.((r * kk) + k) *. wv.((j * kk) + k))
      done;
      d.((r * nn) + j) <- !acc
    done
  done;
  d

(* Everything one timed row needs, kept alive only for the round that times it. *)
type live = {
  lv_tag : string;
  lv_flops : float;
  lv_label : string;
  lv_ctx : Context.t ref;
  lv_routine : Context.routine;
  lv_out : Ir.Tnode.t;
  lv_launch : string;
  lv_parity : bool;
  lv_times : float list ref;
  lv_single : float ref;
}

let () =
  let args = Bench_args.create "projection_shape_bench" in
  let repeats = Bench_args.int args 0 ~name:"repeats" ~default:50 in
  let nbatches = Bench_args.int args 1 ~name:"batches" ~default:5 in
  let group = String.lowercase (Bench_args.string args 2 ~default:"all") in
  (* [fwd]/[rev] reverses the rotation the interleaved rounds start from, so a residual
     first-arm-is-cold bias can be shown not to carry the conclusion. *)
  let order = String.lowercase (Bench_args.string args 3 ~default:"fwd") in
  (* [seeds] times the sketch seeds, one interleaved round per geometry; [tune] runs the full
     search per site (which cannot be interleaved -- a search is minutes of its own dispatches)
     with the disk cache disabled, so neither shape can replay the other's winner. *)
  let mode = String.lowercase (Bench_args.string args 4 ~default:"seeds") in
  let do_seeds = not (String.equal mode "tune") in
  let do_tune = List.mem [ "tune"; "both" ] mode ~equal:String.equal in
  let backend = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc") in
  let on_gpu =
    List.exists [ "metal"; "cuda"; "hip" ] ~f:(fun s -> String.is_substring backend ~substring:s)
  in
  let limits = Context.hardware_limits (Context.auto ()) in
  p "backend %s, repeats %d, batches %d, group %s, order %s, mode %s\n" backend repeats nbatches
    group order mode;
  p "%s\n" (if on_gpu then "GPU blocktile family" else "CPU families");
  let sites =
    let a =
      [
        { tag = "A_rows1024"; bs = []; m = 1024; ns = [ 256 ]; ks = [ 256 ] };
        { tag = "A_b2x512"; bs = [ 2 ]; m = 512; ns = [ 256 ]; ks = [ 256 ] };
        { tag = "A_b4x256"; bs = [ 4 ]; m = 256; ns = [ 256 ]; ks = [ 256 ] };
        { tag = "A_b8x128"; bs = [ 8 ]; m = 128; ns = [ 256 ]; ks = [ 256 ] };
      ]
    and b =
      [
        { tag = "B_ffn_rows1024"; bs = []; m = 1024; ns = [ 1024 ]; ks = [ 256 ] };
        { tag = "B_ffn_b8x128"; bs = [ 8 ]; m = 128; ns = [ 1024 ]; ks = [ 256 ] };
      ]
    and c =
      List.concat_map [ 8; 16; 32; 64 ] ~f:(fun bb ->
          [
            { tag = Printf.sprintf "C_b%dx128" bb; bs = [ bb ]; m = 128; ns = [ 256 ]; ks = [ 256 ] };
            {
              tag = Printf.sprintf "C_rows%d" (bb * 128);
              bs = [];
              m = bb * 128;
              ns = [ 256 ];
              ks = [ 256 ];
            };
          ])
    and d =
      [
        { tag = "D_outproj_rows1024"; bs = []; m = 1024; ns = [ 256 ]; ks = [ 8; 32 ] };
        { tag = "D_outproj_b8x128"; bs = [ 8 ]; m = 128; ns = [ 256 ]; ks = [ 8; 32 ] };
      ]
    and e =
      [
        { tag = "E_qkv_rows8_heads8"; bs = [ 8 ]; m = 128; ns = [ 8; 32 ]; ks = [ 256 ] };
        { tag = "E_qkv_rows1_heads8"; bs = []; m = 1024; ns = [ 8; 32 ]; ks = [ 256 ] };
        { tag = "E_qkv_rows8_heads1"; bs = [ 8 ]; m = 128; ns = [ 256 ]; ks = [ 256 ] };
        { tag = "E_qkv_rows1_heads1"; bs = []; m = 1024; ns = [ 256 ]; ks = [ 256 ] };
      ]
    and pk = [ { tag = "P_square1024"; bs = []; m = 1024; ns = [ 1024 ]; ks = [ 1024 ] } ] in
    match group with
    | "a" -> a
    | "b" -> b
    | "c" -> c
    | "d" -> d
    | "e" -> e
    | "p" -> pk
    | "abd" -> a @ b @ d
    | "abde" -> a @ b @ d @ e
    | "all" -> a @ b @ c @ d @ e @ pk
    | g -> invalid_arg ("unknown group " ^ g)
  in
  let sites = match order with
    | "fwd" -> sites
    | "rev" -> List.rev sites
    | o -> invalid_arg ("unknown order " ^ o)
  in
  (* Any cell of the experiment that could not be measured is a failure of the run, not a blank in
     a table: a harness that exits 0 with "n/a" in the column the caller asked for lets automation
     accept an invalid experiment. *)
  let failures = ref 0 in
  let fail fmt = Printf.ksprintf (fun m -> Int.incr failures; p "   !! %s\n" m) fmt in
  (* Phase 1: per site, the lowering, the detected shape, the seed list and the host oracle. *)
  let prepared =
    List.map sites ~f:(fun s ->
        let fl = flops s in
        p "\n== %s : rows %s | cols %s | contract %s  (%.1f MFLOP)\n" s.tag
          (String.concat ~sep:"x" (List.map (s.bs @ [ s.m ]) ~f:Int.to_string))
          (String.concat ~sep:"x" (List.map s.ns ~f:Int.to_string))
          (String.concat ~sep:"x" (List.map s.ks ~f:Int.to_string))
          (fl /. 1e6);
        let opt = capture (named (s.tag ^ "_probe") (Train.forward (build s))) in
        (match Autotune.detect_matmul opt.LL.llc with
        | None -> fail "%s: NOT DETECTED as a matmul site" s.tag
        | Some site ->
            p "   site: m=%d n=%d k=%d  outer-k [%s]  batch-outer [%s]  batch-inner [%s]\n"
              site.Autotune.m_ni site.Autotune.m_nj site.Autotune.m_nk
              (String.concat ~sep:","
                 (List.map site.Autotune.m_ko ~f:(fun (_, e) -> Int.to_string e)))
              (String.concat ~sep:","
                 (List.map site.Autotune.m_bo ~f:(fun (_, e) -> Int.to_string e)))
              (String.concat ~sep:","
                 (List.map site.Autotune.m_bi ~f:(fun (_, e) -> Int.to_string e))));
        let seeds =
          Autotune.sketch_seed_params ~is_gpu:on_gpu ~is_cpu:(not on_gpu) ~limits opt
          |> List.filter ~f:(fun (q : Autotune.sketch_params) ->
                 (not q.sk_epilogue) && not q.sk_mma)
        in
        (s, fl, seeds, lazy (oracle s)))
  in
  (* Compile one candidate and check its parity; [idx] makes the routine name unique so that under
     [output_debug_files_in_build_directory] each candidate's .cd/.ll/backend source survives its
     neighbours instead of being overwritten by them. *)
  let counter = ref 0 in
  let arm (s, fl, _, orc) ~label ~compile =
    Int.incr counter;
    let name = Printf.sprintf "%s_c%d" s.tag !counter in
    match
      let d = build s in
      let fwd = named name (Train.forward d) in
      let dims = ref None in
      let record o =
        dims := Some (LL.launch_dims o.LL.llc);
        o
      in
      let ctx, routine = compile ~record ~name fwd in
      let ctx = ref ctx in
      for _ = 1 to 3 do
        ctx := Context.run !ctx routine
      done;
      let got = Context.get_values !ctx d.Tensor.value in
      let want = Lazy.force orc in
      let parity =
        Array.length got = Array.length want && Array.for_all2_exn got want ~f:Float.equal
      in
      let launch =
        match !dims with
        | None -> "(default annotators)"
        | Some dm ->
            let pr a = String.concat ~sep:"x" (Array.to_list (Array.map a ~f:Int.to_string)) in
            Printf.sprintf "grid %s block %s" (pr dm.LL.grid) (pr dm.LL.block)
      in
      {
        lv_tag = s.tag;
        lv_flops = fl;
        lv_label = label;
        lv_ctx = ctx;
        lv_routine = routine;
        lv_out = d.Tensor.value;
        lv_launch = launch;
        lv_parity = parity;
        lv_times = ref [];
        lv_single = ref Float.infinity;
      }
    with
    | lv ->
        if not lv.lv_parity then
          fail "%s / %s: PARITY FAILED against the host oracle" s.tag label;
        Some lv
    | exception exn ->
        fail "%s / %s: FAILED: %s" s.tag label
          (List.hd_exn (String.split_lines (Exn.to_string exn)));
        None
  in
  (* One interleaved round: every arm of [lives] is timed batch by batch in rotation, so a monotone
     session drift lands on all of them alike instead of on whichever site came later in the run.
     A whole-site loop cannot do that -- reversing the site order only moves the bias, it does not
     cancel it (docs/agent-notes/training-and-performance.md's A/B protocol: alternate the arms RUN
     BY RUN). Each arm's own batch is [repeats] dispatches queued back-to-back with one sync. *)
  let time_round lives =
    let arr = Array.of_list lives in
    let n = Array.length arr in
    if n > 0 then begin
      for b = 0 to nbatches - 1 do
        for i = 0 to n - 1 do
          let lv = arr.((i + b) % n) in
          let t0 = now () in
          for _ = 1 to repeats do
            lv.lv_ctx := Context.run !(lv.lv_ctx) lv.lv_routine
          done;
          let _ = Context.get_values !(lv.lv_ctx) lv.lv_out in
          let t1 = now () in
          lv.lv_times := ((t1 -. t0) /. Float.of_int repeats) :: !(lv.lv_times)
        done
      done;
      (* The tuner's own statistic ([Autotune.time_routine]): one dispatch, one sync, minimum over
         the iterations -- what a min-of-N per-kernel profile reports, and up to 2.6x above the
         steady-state figure on the same routine. Interleaved for the same reason. *)
      for b = 0 to repeats - 1 do
        for i = 0 to n - 1 do
          let lv = arr.((i + b) % n) in
          let t0 = now () in
          lv.lv_ctx := Context.run !(lv.lv_ctx) lv.lv_routine;
          Context.sync !(lv.lv_ctx);
          let dt = now () -. t0 in
          if Float.(dt < !(lv.lv_single)) then lv.lv_single := dt
        done
      done
    end
  in
  (* The median is the summary and the winner-selection statistic: with a handful of batches on a
     loaded device the minimum is systematically optimistic and crowns whichever geometry drew the
     luckiest batch. The minimum is kept beside it as a diagnostic. *)
  let report lv =
    let sorted = Array.of_list !(lv.lv_times) in
    Array.sort sorted ~compare:Float.compare;
    let n = Array.length sorted in
    let best = sorted.(0) and median = sorted.(n / 2) and worst = sorted.(n - 1) in
    let g t = lv.lv_flops /. t /. 1e9 in
    p "   %-22s %-26s %8.1f GFLOP/s med (min %7.1f, min1 %7.1f)  spread %4.1f%%  %s%s\n" lv.lv_tag
      lv.lv_label (g median) (g best) (g !(lv.lv_single))
      ((worst -. best) /. best *. 100.)
      lv.lv_launch
      (if lv.lv_parity then "" else "  *** PARITY FAILED ***");
    g median
  in
  let release lv = Context.release !(lv.lv_ctx) in
  let results : (string, (string * float) list) Hashtbl.t = Hashtbl.create (module String) in
  let record_result lv g =
    Hashtbl.update results lv.lv_tag ~f:(function
      | None -> [ (lv.lv_label, g) ]
      | Some l -> (lv.lv_label, g) :: l)
  in
  let run_round ~label lives =
    if not (List.is_empty lives) then begin
      p "\n-- round: %s (%d arms interleaved)\n" label (List.length lives);
      time_round lives;
      List.iter lives ~f:(fun lv -> record_result lv (report lv));
      List.iter lives ~f:release
    end
  in
  (* Round 0: the untuned shipped default, one arm per site. *)
  run_round ~label:"default (untuned)"
    (List.filter_map prepared ~f:(fun pr ->
         arm pr ~label:"default (untuned)" ~compile:(fun ~record:_ ~name:_ fwd ->
             Context.compile (Context.auto ()) fwd Ir.Indexing.Empty)));
  (* One round per geometry, over the sites whose seed list offers it: that is the comparison the
     experiment makes, so that is the set that has to be interleaved. Menu order is preserved. *)
  if do_seeds then begin
    (* A round is a BASE geometry, [sk_batch_grid] twins included, because a merged site's
       [32x32x8/4x4] and a batched site's [32x32x8/4x4 bgrid] are the two arms the experiment
       actually compares -- putting the twins in separate rounds would leave exactly that pair
       un-interleaved. A batched site therefore contributes both of its arms to the round. *)
    let base_geom g = String.chop_suffix_if_exists g ~suffix:" bgrid" in
    let geometries =
      List.concat_map prepared ~f:(fun (_, _, seeds, _) ->
          List.map seeds ~f:(fun q -> base_geom (geom_label q)))
      |> List.dedup_and_sort ~compare:String.compare
    in
    List.iter geometries ~f:(fun g ->
        let lives =
          List.concat_map prepared ~f:(fun ((_, _, seeds, _) as pr) ->
              List.filter seeds ~f:(fun q -> String.equal (base_geom (geom_label q)) g)
              |> List.filter_map ~f:(fun q ->
                     arm pr ~label:(geom_label q) ~compile:(fun ~record ~name:_ fwd ->
                         Context.compile
                           ~lowered_transform:(fun o ->
                             record (Sched.apply (Autotune.sketch_schedule ~p:q o) o))
                           (Context.auto ()) fwd Ir.Indexing.Empty)))
        in
        run_round ~label:g lives)
  end;
  (* The full search is per site by construction -- it is minutes of its own dispatches, so there
     is nothing to interleave with. [~search:true] because this mode was asked for explicitly: a
     resolved [autotune_search=false] (the reproducible profile) would otherwise return the untuned
     default under a "TUNED" label, and any outcome other than [Searched] is a failed cell here. *)
  let tuned = Hashtbl.create (module String) in
  if do_tune then
    List.iter prepared ~f:(fun ((s, _, _, _) as pr) ->
        let lbl = ref "" and ms = ref Float.nan and outcome = ref None in
        let lives =
          List.filter_opt
            [
              arm pr ~label:"TUNED (full search)" ~compile:(fun ~record:_ ~name fwd ->
                  Autotune.tune ~name ~search:true ~cache_dir:""
                    ~report:(fun (r : Autotune.report) ->
                      lbl := r.Autotune.best_label;
                      ms := r.Autotune.best_ms;
                      outcome := Some r.Autotune.outcome)
                    (Context.auto ()) fwd Ir.Indexing.Empty);
            ]
        in
        (match !outcome with
        | Some Autotune.Searched -> ()
        | Some o ->
            fail "%s: the tune cell did not search (%s) -- its number is not a tuned measurement"
              s.tag (Autotune.outcome_name o)
        | None -> if not (List.is_empty lives) then fail "%s: the tune cell reported nothing" s.tag);
        run_round ~label:("TUNED " ^ s.tag) lives;
        p "      crowned: %s  (search best_ms %.4f)\n" !lbl !ms;
        Hashtbl.set tuned ~key:s.tag ~data:!lbl);
  let f_opt = function None -> "     n/a" | Some g -> Printf.sprintf "%8.1f" g in
  let of_site tag lbl =
    Option.bind (Hashtbl.find results tag) ~f:(fun l ->
        List.find l ~f:(fun (l', _) -> String.equal l' lbl) |> Option.map ~f:snd)
  in
  (* The two geometries gh-ocannl-728 quotes as crowned on gfx1151. A merged site has no batch
     axis, so its bgrid twin does not exist and the plain sibling is the same kernel. *)
  let crowned tag want =
    match of_site tag (want ^ " bgrid") with Some g -> Some g | None -> of_site tag want
  in
  p "\n\n== summary (GFLOP/s, median timing batch) ==\n";
  p "%-22s %9s  %8s  %8s  %8s  %8s  %8s  %s\n" "site" "MFLOP" "untuned" "32x32x8" "16x16x8"
    "bestseed" "tuned" "winner";
  List.iter sites ~f:(fun s ->
      let all = Option.value (Hashtbl.find results s.tag) ~default:[] in
      let seeds_only =
        List.filter all ~f:(fun (l, _) ->
            (not (String.is_prefix l ~prefix:"default"))
            && not (String.is_prefix l ~prefix:"TUNED"))
      in
      let best_of =
        List.fold seeds_only ~init:None ~f:(fun acc (l, g) ->
            match acc with Some (_, bg) when Float.(bg >= g) -> acc | _ -> Some (l, g))
      in
      let tn = of_site s.tag "TUNED (full search)" in
      p "%-22s %9.1f  %s  %s  %s  %s  %s  %s\n" s.tag (flops s /. 1e6)
        (f_opt (of_site s.tag "default (untuned)"))
        (f_opt (crowned s.tag "gpu 32x32x8/4x4"))
        (f_opt (crowned s.tag "gpu 16x16x8/2x2"))
        (f_opt (Option.map best_of ~f:snd))
        (f_opt tn)
        (match Hashtbl.find tuned s.tag with
        | Some l when Option.is_some tn -> l
        | _ -> Option.value_map best_of ~default:"-" ~f:fst));
  if !failures > 0 then (
    p "\n%d cell(s) of this experiment could not be measured or failed parity -- see the !! lines.\n"
      !failures;
    Stdlib.exit 1)
