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

   Two sites appear twice: A_rows1024 = E_qkv_rows1_heads1 and A_b8x128 = E_qkv_rows8_heads1. That
   is deliberate. They sit ten positions apart in group [abde], so every run measures its own
   session drift and a shape effect can be compared against it.

   Candidates per site: the untuned shipped default, then every geometry of the GPU blocktile
   sketch family applied as the pure IR transform it is; mode [tune] instead runs a full
   [Autotune.tune] search per site with the disk cache DISABLED, so neither shape can replay the
   other's cached winner, and times the crowned routine the same way.

   Every candidate's whole-output checksum is checked against the untuned baseline's, bitwise --
   the inputs are built so that every partial sum is exact in f32 -- before its time is believed,
   and each line carries the launch dimensions the schedule actually produced, so "same geometry"
   is read off the kernel rather than assumed.

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

let capture fwd =
  let captured = ref None in
  let _ctx, _r =
    Context.compile
      ~lowered_transform:(fun opt ->
        captured := Some opt;
        opt)
      (Context.auto ()) fwd Ir.Indexing.Empty
  in
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

let checksum values =
  Array.foldi values ~init:0.0 ~f:(fun i acc v -> acc +. (v *. Float.of_int (1 + (i % 251))))

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

let () =
  (* No tf32: the scalar blocktile family this bench times does no tensorization, and leaving the
     policy alone keeps f32 parity bitwise. *)
  let args = Bench_args.create "projection_shape_bench" in
  let repeats = Bench_args.int args 0 ~name:"repeats" ~default:50 in
  let nbatches = Bench_args.int args 1 ~name:"batches" ~default:5 in
  let group = String.lowercase (Bench_args.string args 2 ~default:"all") in
  (* Sites are timed one after another, so a monotone session drift (clocks, thermals) confounds
     the treatment with position in the run -- the A/B trap docs/agent-notes/training-and-performance.md
     names. Running the same group in both orders is the control: a conclusion that survives the
     reversal is not drift. *)
  let order = String.lowercase (Bench_args.string args 3 ~default:"fwd") in
  (* [seeds] times the raw sketch seeds; [tune] additionally runs the full search (beam refinements
     included) per site with the disk cache DISABLED, so neither shape can replay the other's
     winner, and times the crowned routine the same warm way. The seed table alone cannot answer
     "what is the best schedule the system can find for this shape" -- the beam's own moves
     (vectorized loads, unrolls, swizzles) sit above the seeds. *)
  let mode = String.lowercase (Bench_args.string args 4 ~default:"seeds") in
  let do_seeds = not (String.equal mode "tune") in
  let do_tune = List.mem [ "tune"; "both" ] mode ~equal:String.equal in
  let backend = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc") in
  let on_gpu =
    List.exists [ "metal"; "cuda"; "hip" ] ~f:(fun s -> String.is_substring backend ~substring:s)
  in
  let limits = Context.hardware_limits (Context.auto ()) in
  p "backend %s, repeats %d, batches %d, group %s, order %s\n" backend repeats nbatches group order;
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
      (* Size sweep: the projection shape scaled by batch count, and the merged shape scaled by
         rows -- same tile geometries, same aspect ratio, only total work changes. *)
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
      (* Workload-faithful sites, from the gpt2_mini forward lowering (bench_gpt_diag census, cc):
         q/k/v is [8s,128s,8s,32s,256s] -- the heads are a SEPARATE output axis, so the column
         extent is 32, not 256 -- while the out projection is [8s,128s,256s,8s,32s] and the FFN
         GEMM1 is [8s,128s,1024s,256s]. A 2x2 factorial over the two pre-splits at constant FLOPs:
         the row axis merged (1024) or split (8 x 128), the column axis merged (256) or split into
         8 heads of 32. *)
      [
        { tag = "E_qkv_rows8_heads8"; bs = [ 8 ]; m = 128; ns = [ 8; 32 ]; ks = [ 256 ] };
        { tag = "E_qkv_rows1_heads8"; bs = []; m = 1024; ns = [ 8; 32 ]; ks = [ 256 ] };
        { tag = "E_qkv_rows8_heads1"; bs = [ 8 ]; m = 128; ns = [ 256 ]; ks = [ 256 ] };
        { tag = "E_qkv_rows1_heads1"; bs = []; m = 1024; ns = [ 256 ]; ks = [ 256 ] };
      ]
    and pk =
      (* The family's own ceiling at a size where nothing is launch-bound: what this scalar
         blocktile family reaches when the shape is not in question. *)
      [ { tag = "P_square1024"; bs = []; m = 1024; ns = [ 1024 ]; ks = [ 1024 ] } ]
    in
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
  let sites =
    match order with
    | "fwd" -> sites
    | "rev" -> List.rev sites
    | o -> invalid_arg ("unknown order " ^ o)
  in
  let failures = ref 0 in
  (* Per-site: the untuned baseline, the two crowned geometries the issue names, and the best of
     the whole blocktile menu -- the four numbers the summary table carries. *)
  let summary = ref [] in
  List.iter sites ~f:(fun s ->
      let fl = flops s in
      p "\n== %s : rows %s | cols %s | contract %s  (%.1f MFLOP)\n" s.tag
        (String.concat ~sep:"x" (List.map (s.bs @ [ s.m ]) ~f:Int.to_string))
        (String.concat ~sep:"x" (List.map s.ns ~f:Int.to_string))
        (String.concat ~sep:"x" (List.map s.ks ~f:Int.to_string))
        (fl /. 1e6);
      let d = build s in
      let fwd = named (s.tag ^ "_sched") (Train.forward d) in
      let opt = capture fwd in
      (match Autotune.detect_matmul opt.LL.llc with
      | None -> p "   NOT DETECTED as a matmul site\n"
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
      (* The reference is the SHIPPED default (the default annotators, i.e. no [lowered_transform]):
         an identity transform would compile the unscheduled nest, which on a GPU is a single-thread
         kernel and minutes per run at these sizes. It is also the untuned baseline every scheduled
         candidate is compared against. *)
      let want_chk = ref Float.nan in
      let time_one ~label ~compile =
        try
          let d = build s in
          let fwd = named (s.tag ^ "_t") (Train.forward d) in
          let dims = ref None in
          let record o =
            dims := Some (LL.launch_dims o.LL.llc);
            o
          in
          let ctx, routine = compile ~record fwd in
          let ctx = ref ctx in
          for _ = 1 to 3 do
            ctx := Context.run !ctx routine
          done;
          let v = Context.get_values !ctx d.Tensor.value in
          let chk = checksum v in
          let ok =
            if Float.is_nan !want_chk then (
              want_chk := chk;
              true)
            else Float.equal chk !want_chk
          in
          if not ok then Int.incr failures;
          let times =
            Array.init nbatches ~f:(fun _ ->
                let t0 = now () in
                for _ = 1 to repeats do
                  ctx := Context.run !ctx routine
                done;
                let _ = Context.get_values !ctx d.Tensor.value in
                let t1 = now () in
                (t1 -. t0) /. Float.of_int repeats)
          in
          (* Two statistics, because they answer different questions and disagree by up to 2.5x on
             a loaded WSL box. [times] above is steady-state: [repeats] dispatches queued
             back-to-back with ONE sync, so the mean is what the kernel sustains when a step keeps
             the queue full. [single] is the tuner's own statistic ([Autotune.time_routine]): one
             dispatch, one sync, minimum over the iterations -- the luckiest single launch, which
             is what a min-of-N per-kernel profile reports. A conclusion should hold under both. *)
          let single = ref Float.infinity in
          for _ = 1 to repeats do
            let t0 = now () in
            ctx := Context.run !ctx routine;
            Context.sync !ctx;
            let dt = now () -. t0 in
            if Float.(dt < !single) then single := dt
          done;
          let sorted = Array.sorted_copy times ~compare:Float.compare in
          let best = sorted.(0) in
          let median = sorted.(Array.length sorted / 2) in
          let worst = sorted.(Array.length sorted - 1) in
          let launch =
            match !dims with
            | None -> "(default annotators)"
            | Some dm ->
                let pr a = String.concat ~sep:"x" (Array.to_list (Array.map a ~f:Int.to_string)) in
                Printf.sprintf "grid %s block %s" (pr dm.LL.grid) (pr dm.LL.block)
          in
          p
            "   %-26s %9.4f ms  %8.1f GFLOP/s (med %7.1f, min1 %7.1f)  spread %4.1f%%  %s%s\n"
            label (best *. 1e3) (fl /. best /. 1e9) (fl /. median /. 1e9)
            (fl /. !single /. 1e9)
            ((worst -. best) /. best *. 100.)
            launch
            (if ok then "" else "  *** PARITY FAILED (chk " ^ Printf.sprintf "%.10g" chk ^ ") ***");
          Some (fl /. best /. 1e9)
        with e ->
          p "   %-26s FAILED: %s\n" label (List.hd_exn (String.split_lines (Exn.to_string e)));
          None
      in
      let of_transform t ~record fwd =
        Context.compile
          ~lowered_transform:(fun o -> record (t o))
          (Context.auto ()) fwd Ir.Indexing.Empty
      in
      let base =
        time_one ~label:"default (untuned)" ~compile:(fun ~record:_ fwd ->
            Context.compile (Context.auto ()) fwd Ir.Indexing.Empty)
      in
      let measured =
        if not do_seeds then []
        else
          List.filter_map seeds ~f:(fun q ->
              let lbl = geom_label q in
              Option.map
                (time_one ~label:lbl
                   ~compile:
                     (of_transform (fun o -> Sched.apply (Autotune.sketch_schedule ~p:q o) o)))
                ~f:(fun g -> (lbl, g)))
      in
      let tuned =
        if not do_tune then None
        else begin
          let lbl = ref "" in
          let ms = ref Float.nan in
          let g =
            time_one ~label:"TUNED (full search)" ~compile:(fun ~record:_ fwd ->
                Autotune.tune ~name:(s.tag ^ "_t") ~cache_dir:""
                  ~report:(fun (r : Autotune.report) ->
                    lbl := r.Autotune.best_label;
                    ms := r.Autotune.best_ms)
                  (Context.auto ()) fwd Ir.Indexing.Empty)
          in
          p "      crowned: %s  (search best_ms %.4f)\n" !lbl !ms;
          Option.map g ~f:(fun g -> (!lbl, g))
        end
      in
      let named_geom want =
        List.find measured ~f:(fun (l, _) -> String.equal l want) |> Option.map ~f:snd
      in
      (* The two geometries gh-ocannl-728 quotes as crowned on gfx1151. On a merged site there is
         no batch axis, so the bgrid twin does not exist and the plain sibling is the same kernel. *)
      let crowned want =
        match named_geom (want ^ " bgrid") with Some g -> Some g | None -> named_geom want
      in
      let best_of =
        List.fold measured ~init:None ~f:(fun acc (l, g) ->
            match acc with Some (_, bg) when Float.(bg >= g) -> acc | _ -> Some (l, g))
      in
      summary :=
        ( s.tag,
          fl,
          base,
          crowned "gpu 32x32x8/4x4",
          crowned "gpu 16x16x8/2x2",
          best_of,
          tuned )
        :: !summary);
  let f_opt = function None -> "     n/a" | Some g -> Printf.sprintf "%8.1f" g in
  p "\n\n== summary (GFLOP/s, min over timing batches) ==\n";
  p "%-22s %9s  %8s  %8s  %8s  %8s  %8s  %s\n" "site" "MFLOP" "untuned" "32x32x8" "16x16x8"
    "bestseed" "tuned" "winner";
  List.iter (List.rev !summary) ~f:(fun (tag, fl, base, c32, c16, best, tuned) ->
      p "%-22s %9.1f  %s  %s  %s  %s  %s  %s\n" tag (fl /. 1e6) (f_opt base) (f_opt c32)
        (f_opt c16)
        (f_opt (Option.map best ~f:snd))
        (f_opt (Option.map tuned ~f:snd))
        (match tuned with
        | Some (l, _) -> l
        | None -> Option.value_map best ~default:"-" ~f:fst));
  if !failures > 0 then (
    p "\n%d candidate(s) failed parity against the serial reference.\n" !failures;
    Stdlib.exit 1)
