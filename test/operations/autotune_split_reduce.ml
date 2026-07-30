(* Split-reduce autotune seeding, gh-ocannl-484 task 3: the deterministic two-pass split reduction
   ([Sched.Split_reduce], landed as tasks 1/2/4) becomes reachable by the tuner. Covered here,
   backend-independent (all printed booleans hold on every backend):

   - [Autotune.split_reduce_sites] detects reduction-dominated sites: an rmw accumulation whose
   target has few cells while a long serial reduction loop feeds it (the skinny matvec below — the
   same shape as conv bias/weight gradients and split-K GEMMs). Elementwise code and short
   reductions yield no sites; the gh-466 embedding-backward scatter yields a dynamic site. -
   [Autotune.tune] seeds the sites as [F_split] candidates (report [split_reduce_candidates]),
   compiles and times them ([split_reduce_timed]) — the prelude applies whole-routine before
   fission, so the two passes land in separate kernels — and the tuned routine's values match the
   untuned serial reference (approximately: the split reassociates the reduction). - A hand-crafted
   split-reduce cache entry — whole-routine prelude in [saved] alongside post-prelude per-segment
   [segments] — replays through tune's cache-hit path ([F_split_saved]: [of_saved] re-mints the
   partials node, the per-segment lookup and drift guard pass), with correct values. - Multi-site:
   two skinny matvec accumulations in one routine seed per-site singles plus one composite
   recombining each site's best-timed [num_blocks]. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module LL = Ir.Low_level
module Sched = Ir.Schedule
module SC = Ir.Schedule_cache
module Asgns = Ir.Assignments

let p name b = Stdio.printf "%s: %b\n" name b
let approx a b = Float.(abs (a - b) < 1e-3 *. (1. +. abs b))

let named name (comp : Asgns.comp) : Asgns.comp =
  { comp with asgns = Asgns.Block_comment (name, comp.asgns) }

let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")
let is_gpu = Sched.backend_is_gpu backend_name
let is_cpu = Sched.backend_is_cpu backend_name

(* The dune sandbox persists across runs: stale entries written by an older binary would break
   miss-then-hit assertions. *)
let clean_cache dir =
  if Stdlib.Sys.file_exists dir && Stdlib.Sys.is_directory dir then
    Array.iter (Stdlib.Sys.readdir dir) ~f:(fun f ->
        Stdlib.Sys.remove (Stdlib.Filename.concat dir f))

let preset opt =
  if is_gpu then Sched.default_gpu ~min_parallel:1 ~limits:Ir.Backend_intf.no_hardware_limits opt
  else if is_cpu then Sched.default_cpu ~min_parallel:1 opt
  else []

let zero_sched _tns = []

(* Skinny matvec — the reduction-dominated shape: y[o] = Σ_k m[o,k] * v[k], few output cells, long
   reduction. Mixed magnitudes so reassociation is numerically observable but small. *)
let out_d = 8
let red_d = 128

let mav =
  Array.init (out_d * red_d) ~f:(fun x ->
      Float.sin (Float.of_int x) *. (10. **. Float.of_int (x % 3)))

let mvv = Array.init red_d ~f:(fun x -> Float.cos (Float.of_int x) *. (10. **. Float.of_int (x % 2)))

let make_matvec label =
  let m =
    TDSL.ndarray mav ~label:[ "sr_m" ^ label ] ~input_dims:[ red_d ] ~output_dims:[ out_d ] ()
  in
  let v = TDSL.ndarray mvv ~label:[ "sr_v" ^ label ] ~output_dims:[ red_d ] () in
  let%op y = m * v in
  y

let capture_base comp =
  let capture = ref None in
  let ctx = Context.auto () in
  let _ctx, _routine =
    Context.compile
      ~lowered_transform:(fun opt ->
        capture := Some opt;
        opt)
      ctx comp Ir.Indexing.Empty
  in
  Option.value_exn ~here:[%here] !capture

let run_forward ~name (t : Tensor.t) =
  let ctx = Context.auto () in
  let ctx, routine = Context.compile ctx (named name (Train.forward t)) Ir.Indexing.Empty in
  let ctx = Context.run ctx routine in
  Context.get_values ctx t.Tensor.value

(* === Leg 1: site detection. === *)

let () =
  Tensor.unsafe_reinitialize ();
  let y = make_matvec "det" in
  let base_opt = capture_base (named "asr_detect" (Train.forward y)) in
  let sites = Autotune.split_reduce_sites base_opt in
  p "skinny matvec yields one static split-reduce site"
    (match sites with
    | [ s ] ->
        (not s.Autotune.sr_dynamic) && s.Autotune.sr_red = red_d && s.Autotune.sr_out = out_d
    | _ -> false);

  (* Elementwise code has no rmw accumulation: no sites. *)
  let a = TDSL.ndarray mvv ~label:[ "sr_a" ] ~output_dims:[ red_d ] () in
  let b = TDSL.ndarray mvv ~label:[ "sr_b" ] ~output_dims:[ red_d ] () in
  let%op w = a + b in
  let ew_opt = capture_base (named "asr_elemwise" (Train.forward w)) in
  p "elementwise code yields no split-reduce sites"
    (List.is_empty (Autotune.split_reduce_sites ew_opt));

  (* A short reduction (below the extent threshold) is not worth a second pass: no sites. *)
  let short = 32 in
  let ms =
    TDSL.ndarray
      (Array.init (out_d * short) ~f:(fun x -> Float.of_int (x % 5)))
      ~label:[ "sr_ms" ] ~input_dims:[ short ] ~output_dims:[ out_d ] ()
  in
  let vs =
    TDSL.ndarray (Array.init short ~f:(fun x -> Float.of_int (x % 3))) ~label:[ "sr_vs" ]
      ~output_dims:[ short ] ()
  in
  let%op ys = ms * vs in
  let short_opt = capture_base (named "asr_short" (Train.forward ys)) in
  p "short reductions yield no split-reduce sites"
    (List.is_empty (Autotune.split_reduce_sites short_opt))

(* === Leg 2: the gh-466 embedding-backward scatter yields a dynamic site. === *)

let () =
  let vocab = 5 in
  let embed = 4 in
  let positions = 96 in
  let classes = TDSL.range vocab in
  let ids =
    TDSL.ndarray
      (Array.init positions ~f:(fun i -> Float.of_int (i % vocab)))
      ~label:[ "asr_ids" ] ~batch_dims:[ positions ] ~output_dims:[] ()
  in
  let%op one_hot = classes = ids in
  let c =
    TDSL.param
      ~values:(Array.init (embed * vocab) ~f:Float.of_int)
      "asr_c" ~input_dims:[ vocab ] ~output_dims:[ embed ] ()
  in
  let%op embedded = c * one_hot in
  let%op loss = embedded ++ "...|... => |->0" in
  let update = Train.grad_update loss in
  let ctx = Context.auto () in
  let ctx = Train.init_params ctx Train.IDX.empty loss in
  let capture = ref None in
  let _ctx, _routine =
    Context.compile
      ~lowered_transform:(fun opt ->
        capture := Some opt;
        opt)
      ctx (named "asr_emb" update) Ir.Indexing.Empty
  in
  let emb_opt = Option.value_exn ~here:[%here] !capture in
  let sites = Autotune.split_reduce_sites emb_opt in
  p "embedding backward yields a dynamic (scatter) split-reduce site"
    (List.exists sites ~f:(fun s -> s.Autotune.sr_dynamic))

(* === Leg 3: tune seeds the site; tuned values match the serial reference. === *)

let () =
  let y0 = make_matvec "ref" in
  let want = run_forward ~name:"asr_serial" y0 in
  let y1 = make_matvec "tuned" in
  let report = ref None in
  let ctx = Context.auto () in
  let ctx, routine =
    Autotune.tune ~beam_width:2 ~rounds:0 ~repeats:1 ~cache_dir:""
      ~report:(fun r -> report := Some r)
      ctx
      (named "asr_tuned" (Train.forward y1))
      Ir.Indexing.Empty
  in
  let ctx = Context.run ctx routine in
  let got = Context.get_values ctx y1.Tensor.value in
  (match !report with
  | Some r ->
      (* [2*num_blocks <= red] admits b ∈ {8, 32} of the CPU sweep, b ∈ {32} of the GPU sweep. *)
      p "split-reduce candidates seeded"
        (if is_cpu then r.Autotune.split_reduce_candidates = 2
         else r.Autotune.split_reduce_candidates >= 1);
      p "split-reduce candidates timed" (r.Autotune.split_reduce_timed >= 1)
  | None ->
      p "split-reduce candidates seeded" false;
      p "split-reduce candidates timed" false);
  p "tuned skinny matvec matches the serial reference"
    (Array.for_all2_exn got want ~f:approx)

(* === Leg 4: a hand-crafted split-reduce cache entry replays (the F_split_saved path). === *)

let () =
  let cache_dir = "autotune_split_cache" in
  clean_cache cache_dir;
  let y0 = make_matvec "hcref" in
  let want = run_forward ~name:"asr_hc_serial" y0 in
  let y1 = make_matvec "hc" in
  let hc_comp = named "asr_hc" (Train.forward y1) in
  let base_opt = capture_base hc_comp in
  let base_canon = SC.canonicalize base_opt in
  let prelude_saved = ref [] in
  let segments_assoc = ref [] in
  let sctx = Context.auto () in
  let _sctx, _sroutine =
    Context.compile
      ~lowered_transforms:(fun opt ->
        let sites = Autotune.split_reduce_sites opt in
        let s = List.hd_exn sites in
        let op, _, _, _ =
          Sched.split_reduce ~axis:s.Autotune.sr_axis ~target:s.Autotune.sr_target ~num_blocks:8
        in
        (prelude_saved := fst (SC.to_saved (SC.base_registry (SC.canonicalize opt)) [ op ]));
        let opt' = Sched.apply [ op ] opt in
        let tuples = Sched.fission_scheduled ~preset ~zero_sched ~static_indices:[] opt' in
        segments_assoc :=
          List.filter_map tuples ~f:(fun (kind, pre, sched, _post) ->
              match kind with
              | `Normal ->
                  let pre_canon = SC.canonicalize ~with_placements:false pre in
                  Some (SC.digest pre_canon, fst (SC.to_saved (SC.base_registry pre_canon) sched))
              | _ -> None);
        List.map tuples ~f:(fun (_, _, _, post) -> post))
      sctx hc_comp Ir.Indexing.Empty
  in
  p "the hand-crafted entry has a prelude and per-segment schedules"
    ((not (List.is_empty !prelude_saved)) && List.length !segments_assoc >= 2);
  SC.store ~dir:cache_dir
    ~key:(SC.cache_key base_canon ~backend:backend_name)
    {
      SC.version = SC.entry_version;
      backend = backend_name;
      source_digest = SC.digest base_canon;
      saved = !prelude_saved;
      segments = Some !segments_assoc;
      best_ms = 0.;
      baseline_ms = 0.;
    };
  let hit_report = ref None in
  let hctx = Context.auto () in
  let hctx, hroutine =
    Autotune.tune ~beam_width:1 ~rounds:0 ~repeats:1 ~cache_dir
      ~report:(fun r -> hit_report := Some r)
      hctx hc_comp Ir.Indexing.Empty
  in
  let hctx = Context.run hctx hroutine in
  let got = Context.get_values hctx y1.Tensor.value in
  (match !hit_report with
  | Some r ->
      p "split-reduce entry hits the cache" r.Autotune.cache_hit;
      p "split-reduce cache-hit replay is fissioned" r.Autotune.fissioned
  | None ->
      p "split-reduce entry hits the cache" false;
      p "split-reduce cache-hit replay is fissioned" false);
  p "split-reduce cache-hit replay computes correctly" (Array.for_all2_exn got want ~f:approx)

(* === Leg 5: multi-site — per-site singles plus the recombined composite. === *)

let () =
  let red2 = 256 in
  let m2v =
    Array.init (out_d * red2) ~f:(fun x -> Float.cos (Float.of_int x) *. Float.of_int ((x % 4) + 1))
  in
  let v2v = Array.init red2 ~f:(fun x -> Float.sin (Float.of_int (3 * x))) in
  let make_pair label =
    let m1 =
      TDSL.ndarray mav ~label:[ "ms_m1" ^ label ] ~input_dims:[ red_d ] ~output_dims:[ out_d ] ()
    in
    let v1 = TDSL.ndarray mvv ~label:[ "ms_v1" ^ label ] ~output_dims:[ red_d ] () in
    let m2 =
      TDSL.ndarray m2v ~label:[ "ms_m2" ^ label ] ~input_dims:[ red2 ] ~output_dims:[ out_d ] ()
    in
    let v2 = TDSL.ndarray v2v ~label:[ "ms_v2" ^ label ] ~output_dims:[ red2 ] () in
    let%op y1 = m1 * v1 in
    Train.set_materialized y1.Tensor.value;
    let%op y2 = m2 * v2 in
    Train.set_materialized y2.Tensor.value;
    let%op z = y1 + y2 in
    z
  in
  let z0 = make_pair "ref" in
  let want = run_forward ~name:"asr_ms_serial" z0 in
  let z1 = make_pair "tuned" in
  let base_opt = capture_base (named "asr_ms_probe" (Train.forward (make_pair "probe"))) in
  p "both accumulations are detected as split-reduce sites"
    (List.length (Autotune.split_reduce_sites base_opt) = 2);
  let report = ref None in
  let ctx = Context.auto () in
  let ctx, routine =
    Autotune.tune ~beam_width:2 ~rounds:0 ~repeats:1 ~cache_dir:""
      ~report:(fun r -> report := Some r)
      ctx
      (named "asr_ms_tuned" (Train.forward z1))
      Ir.Indexing.Empty
  in
  let ctx = Context.run ctx routine in
  let got = Context.get_values ctx z1.Tensor.value in
  (match !report with
  | Some r ->
      (* CPU sweep {8, 32, 128}: site 1 (red 128) admits {8, 32}, site 2 (red 256) admits
         {8, 32, 128} — five singles; the composite recombines the two sites' best-timed singles.
         On cc every candidate compiles and times, so the composite is exactly one extra timing. *)
      p "multi-site split-reduce singles seeded"
        (if is_cpu then r.Autotune.split_reduce_candidates = 5
         else r.Autotune.split_reduce_candidates >= 2);
      p "best-timed singles recombined into a composite"
        (if is_cpu then r.Autotune.split_reduce_timed = r.Autotune.split_reduce_candidates + 1
         else r.Autotune.split_reduce_timed > 0)
  | None ->
      p "multi-site split-reduce singles seeded" false;
      p "best-timed singles recombined into a composite" false);
  p "tuned multi-site routine matches the serial reference"
    (Array.for_all2_exn got want ~f:approx);
  Stdio.printf "\nDone.\n%!"
