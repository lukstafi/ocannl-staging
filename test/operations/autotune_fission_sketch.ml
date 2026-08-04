(* Autotune follow-ups: per-fission-segment tuning and the matmul sketch generator.

   Covered here, backend-independent (all printed booleans hold on every backend):

   - [Schedule.fission_scheduled] (the exposed fission pipeline with caller-supplied per-segment
   schedules) splits the canonical two-nest chain with a forced-materialized intermediate into two
   [`Normal] segments; compiling them through the plural [?lowered_transforms] seam executes
   correctly. - A hand-crafted fissioned cache entry (per-segment schedules keyed by pre-schedule
   segment digests) replays through [Autotune.tune]'s cache-hit path: the report says [fissioned],
   and the values are correct — exercising [F_saved] rebinding of per-segment saved schedules. -
   [Autotune.tune] on a fissionable computation searches whole-routine and fissioned candidates and
   returns correct values; the second call hits the cache; its search reaches several candidates,
   measuring the ones the backend can dispatch and accounting for the rest in the decline census
   (gh-ocannl-543 — on GPU backends only the fissioned preset is measured). - The matmul sketch
   generator detects a
   32x32 matmul and seeds tile-size instantiations of the register-blocktiling (GPU) /
   operand-packing (CPU) pipelines, plus the tensorized (tile-MMA) pipelines (unstaged and
   cooperatively staged [Tensorize] on backends with an mma capability; whole-triple and Grid-split
   register-tiled [Tile_mma] on the C backends); the tuned routine matches the serial twin, and the
   schedules round-trip through the saved form when a sketch wins. - Per-fission-segment sketches
   ([F_sketch]): on a fissionable chain whose consumer is a matmul, the matmul's [Zero_out] fissions
   into its own [`Zeros] segment, so the whole-routine sketches never fit the segment's (unzeroed)
   site — the per-segment seeds must apply instead ([fiss_sketch_candidates]), get timed
   ([fiss_sketch_timed]), and the tuned routine matches the serial twin. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module LL = Ir.Low_level
module Sched = Ir.Schedule
module SC = Ir.Schedule_cache
module Asgns = Ir.Assignments

let p name b = Stdio.printf "%s: %b\n" name b
let approx a b = Float.(abs (a - b) < 1e-2)

let named name (comp : Asgns.comp) : Asgns.comp =
  { comp with asgns = Asgns.Block_comment (name, comp.asgns) }

let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")
let is_gpu = Sched.backend_is_gpu backend_name
let is_cpu = Sched.backend_is_cpu backend_name

(* The dune sandbox persists across runs: stale entries written by an older binary (digest
   ingredients and the saved-schedule format evolve) would break miss-then-hit assertions. *)
let clean_cache dir =
  if Stdlib.Sys.file_exists dir && Stdlib.Sys.is_directory dir then
    Array.iter (Stdlib.Sys.readdir dir) ~f:(fun f ->
        Stdlib.Sys.remove (Stdlib.Filename.concat dir f))

let () =
  List.iter
    [ "autotune_fission_cache"; "autotune_fission_cache2"; "autotune_sketch_cache" ]
    ~f:clean_cache;
  (* === The fissionable chain: d = a + b (forced materialized), e = d *. d^T. The transposed read
     keeps the cross-nest edge misaligned, so the aligned cross-nest rule cannot merge the pair and
     the chain still fissions (a plain pointwise consumer would now stay one kernel). === *)
  let n = 16 in
  let av = Array.init (n * n) ~f:(fun i -> Float.of_int (i % 7) *. 0.5) in
  let bv = Array.init (n * n) ~f:(fun i -> Float.of_int (i % 5) -. 2.) in
  let expected_e =
    Array.init (n * n) ~f:(fun idx ->
        let i = idx / n and j = idx % n in
        let dv k l = av.((k * n) + l) +. bv.((k * n) + l) in
        dv i j *. dv j i)
  in
  let a = TDSL.ndarray av ~label:[ "a" ] ~output_dims:[ n; n ] () in
  let b = TDSL.ndarray bv ~label:[ "b" ] ~output_dims:[ n; n ] () in
  let%op d = a + b in
  Train.set_materialized d.Tensor.value;
  let%op e = d *. (d ++ "ij=>ji") in
  let chain_comp = named "af_chain" (Train.forward e) in

  (* --- fission_scheduled + the plural transforms seam, directly --- *)
  let seg_kinds = ref [] in
  let ctx = Context.auto () in
  let limits = Context.hardware_limits ctx in
  let ctx, routine =
    Context.compile
      ~lowered_transforms:(fun opt ->
        let preset o =
          if is_gpu then Sched.default_gpu ~min_parallel:1 ~limits o
          else if is_cpu then Sched.default_cpu ~min_parallel:1 o
          else []
        in
        let zero_sched tns = if is_gpu then Sched.zero_expansion ~limits tns else [] in
        let tuples = Sched.fission_scheduled ~preset ~zero_sched ~static_indices:[] opt in
        seg_kinds := List.map tuples ~f:(fun (kind, _, _, _) -> kind);
        List.map tuples ~f:(fun (_, _, _, post) -> post))
      ctx chain_comp Ir.Indexing.Empty
  in
  let ctx = Context.run ctx routine in
  let got_e = Context.get_values ctx e.Tensor.value in
  p "fission_scheduled splits the chain into two normal segments"
    (match !seg_kinds with [ `Normal; `Normal ] -> true | _ -> false);
  p "fissioned segments through lowered_transforms compute correctly"
    (Array.for_all2_exn got_e expected_e ~f:approx);

  (* --- a hand-crafted fissioned cache entry replays through tune's cache-hit path --- *)
  let cache_dir = "autotune_fission_cache" in
  let base_capture = ref None in
  let bctx = Context.auto () in
  let _bctx, _br =
    Context.compile
      ~lowered_transform:(fun opt ->
        base_capture := Some opt;
        opt)
      bctx chain_comp Ir.Indexing.Empty
  in
  let base_opt = Option.value_exn ~here:[%here] !base_capture in
  let base_canon = SC.canonicalize base_opt in
  let segments_assoc = ref [] in
  let fctx = Context.auto () in
  let _fctx, _fr =
    Context.compile
      ~lowered_transforms:(fun opt ->
        let preset o =
          if is_gpu then Sched.default_gpu ~min_parallel:1 ~limits o
          else if is_cpu then Sched.default_cpu ~min_parallel:1 o
          else []
        in
        let zero_sched tns = if is_gpu then Sched.zero_expansion ~limits tns else [] in
        let tuples = Sched.fission_scheduled ~preset ~zero_sched ~static_indices:[] opt in
        segments_assoc :=
          List.filter_map tuples ~f:(fun (kind, pre, sched, _post) ->
              match kind with
              | `Normal ->
                  (* Per-segment replay matching keys on the structural canon (see
                     Schedule_cache.canonicalize's [with_placements]). *)
                  let pre_canon = SC.canonicalize ~with_placements:false pre in
                  let saved, _reg = SC.to_saved (SC.base_registry pre_canon) sched in
                  Some (SC.digest pre_canon, saved)
              | _ -> None);
        List.map tuples ~f:(fun (_, _, _, post) -> post))
      fctx chain_comp Ir.Indexing.Empty
  in
  SC.store ~dir:cache_dir
    ~key:(SC.cache_key base_canon ~backend:(Context.backend_name bctx))
    {
      SC.version = SC.entry_version;
      backend = Context.backend_name bctx;
      source_digest = SC.digest base_canon;
      saved = [];
      segments = Some !segments_assoc;
      best_ms = 0.;
      baseline_ms = 0.;
      default_ms = None;
    };
  let hit_report = ref None in
  let hctx = Context.auto () in
  let hctx, hroutine =
    Autotune.tune ~beam_width:1 ~rounds:0 ~repeats:1 ~cache_dir
      ~report:(fun r -> hit_report := Some r)
      hctx chain_comp Ir.Indexing.Empty
  in
  let hctx = Context.run hctx hroutine in
  let got_hit = Context.get_values hctx e.Tensor.value in
  (match !hit_report with
  | Some r ->
      p "hand-crafted fissioned entry hits the cache" r.Autotune.cache_hit;
      p "cache-hit replay is fissioned" r.Autotune.fissioned
  | None ->
      p "hand-crafted fissioned entry hits the cache" false;
      p "cache-hit replay is fissioned" false);
  p "fissioned cache-hit replay computes correctly"
    (Array.for_all2_exn got_hit expected_e ~f:approx);

  (* --- tune end-to-end on the fissionable chain (fresh cache dir: search, then hit) --- *)
  let cache_dir2 = "autotune_fission_cache2" in
  let reports = ref [] in
  let tune_chain () =
    let ctx = Context.auto () in
    let ctx, routine =
      Autotune.tune ~beam_width:2 ~rounds:1 ~repeats:1 ~cache_dir:cache_dir2
        ~report:(fun r -> reports := r :: !reports)
        ctx chain_comp Ir.Indexing.Empty
    in
    let ctx = Context.run ctx routine in
    Context.get_values ctx e.Tensor.value
  in
  let got_t1 = tune_chain () in
  let got_t2 = tune_chain () in
  let r2, r1 =
    match !reports with [ r2; r1 ] -> (r2, r1) | _ -> failwith "expected two reports"
  in
  p "tuned fissionable chain values correct" (Array.for_all2_exn got_t1 expected_e ~f:approx);
  p "chain tune searches then hits the cache" ((not r1.Autotune.cache_hit) && r2.Autotune.cache_hit);
  (* gh-ocannl-552: the untuned-default reference is measured by the search — the config-thresholds
     fissioned seed is the first candidate that binds a hardware dimension on GPU, and on CPU it is
     timed or dedups against a timed twin — and persists through the cache entry, so a cache-hit
     report still answers "did tuning beat the default?". *)
  p "the default reference is measured and survives the cache round-trip (gh-ocannl-552)"
    (match (r1.Autotune.default_ms, r2.Autotune.default_ms) with
    | Some d1, Some d2 ->
        Float.is_finite d1 && Float.is_finite d2 && Float.(r1.Autotune.best_ms <= d1)
    | _ -> false);
  (* gh-ocannl-543: [candidates_timed >= 2] is a cc-shaped assertion. This chain's candidate space
     is the same on every backend, but most of it is serial forms — the whole-routine presets dedup
     to the unscheduled base, and the beam's Split/Swap/Vectorize moves off that base cannot bind a
     hardware dimension — so a GPU backend times exactly one candidate (the fissioned preset) and
     refuses the rest under gh-ocannl-532, where cc times all of them at full single-core speed. The
     portable statement is over the population the census now covers: at least one candidate was
     measured, and the search reached several, whether measured or refused. *)
  let not_dispatched (r : Autotune.report) =
    List.sum
      (module Int)
      r.Autotune.declines
      ~f:(fun d ->
        match d.Autotune.key with
        | Ir.Schedule_outcome.Not_dispatched_key _ -> d.Autotune.count
        | _ -> 0)
  in
  p "chain tune timed a dispatchable candidate" (r1.Autotune.candidates_timed >= 1);
  p "chain tune reached multiple candidates (timed, or refused as unparallelized)"
    (r1.Autotune.candidates_timed + not_dispatched r1 >= 2);
  p "chain tune: candidates refused as unparallelized exactly on GPU"
    (if is_gpu then not_dispatched r1 > 0 else not_dispatched r1 = 0);
  p "chain cache-hit values correct" (Array.for_all2_exn got_t2 expected_e ~f:approx);

  (* === The matmul sketch: 32x32 times 32x32 === *)
  let m = 32 in
  let mav = Array.init (m * m) ~f:(fun i -> Float.of_int (i % 13) *. 0.25) in
  let mbv = Array.init (m * m) ~f:(fun i -> Float.of_int (i % 17) -. 8.) in
  let ma = TDSL.ndarray mav ~label:[ "ma" ] ~input_dims:[ m ] ~output_dims:[ m ] () in
  let mb = TDSL.ndarray mbv ~label:[ "mb" ] ~input_dims:[ m ] ~output_dims:[ m ] () in
  let%op mc0 = ma * mb in
  let serial_comp = named "af_mm_serial" (Train.forward mc0) in
  let sctx = Context.auto () in
  let sctx, sroutine =
    Context.compile ~lowered_transform:(fun opt -> opt) sctx serial_comp Ir.Indexing.Empty
  in
  let sctx = Context.run sctx sroutine in
  let got_serial = Context.get_values sctx mc0.Tensor.value in
  let%op mc1 = ma * mb in
  let mm_comp = named "af_mm_tuned" (Train.forward mc1) in
  let cache_dir3 = "autotune_sketch_cache" in
  let mm_reports = ref [] in
  let tune_mm () =
    let ctx = Context.auto () in
    (* [timing_ctx]: candidates compile and time against a scratch lineage; only the winner is
       compiled from [ctx] (the caller's buffers are never touched by timing runs). *)
    let ctx, routine =
      Autotune.tune ~beam_width:2 ~rounds:0 ~repeats:1 ~cache_dir:cache_dir3
        ~timing_ctx:(Context.auto ())
        ~report:(fun r -> mm_reports := r :: !mm_reports)
        ctx mm_comp Ir.Indexing.Empty
    in
    let ctx = Context.run ctx routine in
    Context.get_values ctx mc1.Tensor.value
  in
  let got_mm1 = tune_mm () in
  let got_mm2 = tune_mm () in
  let mr2, mr1 =
    match !mm_reports with [ r2; r1 ] -> (r2, r1) | _ -> failwith "expected two mm reports"
  in
  p "matmul sketch instantiations seeded" (mr1.Autotune.sketch_candidates > 0);
  (* Tensorized (tile-MMA) sketch families ride along the blocktiling/packing ones: for the 32x32x32
     f32 site, cc seeds 2 packing + 2 hoisted-packing + 2 whole-triple/Grid-split [Tensorize]
     pipelines + 1 cache-blocked packed [Tile_mma] composition, its hoisted, Grid-parallel
     hoisted-only, and Grid-parallel in-kernel variants (gh-ocannl-469/470); Metal seeds 4
     blocktiling + 5 simdgroup pipelines (2 unstaged full-K + 3 cooperatively staged); other GPU
     backends seed at least the 4 blocktiling ones (plus 5 wmma pipelines when the device reports an
     mma capability). *)
  p "tensorized (mma) sketch instantiations seeded"
    (mr1.Autotune.sketch_candidates
    >= if is_cpu then 6 else if String.is_substring backend_name ~substring:"metal" then 9 else 4);
  p "tuned matmul matches the serial twin" (Array.for_all2_exn got_mm1 got_serial ~f:approx);
  p "matmul tune searches then hits the cache"
    ((not mr1.Autotune.cache_hit) && mr2.Autotune.cache_hit);
  p "matmul cache-hit values match the serial twin"
    (Array.for_all2_exn got_mm2 got_serial ~f:approx);

  (* === Per-fission-segment sketches (F_sketch): qd = qa + qb (forced materialized), then the
     matmul qe = qd * qc. The chain fissions; the matmul's [Zero_out] lands in its own [`Zeros]
     segment, so the matmul segment's site is unzeroed — [detect_matmul] must fire per segment and
     the sketch pipelines apply without the zero-expansion geometry. === *)
  let q = 32 in
  let qav = Array.init (q * q) ~f:(fun i -> Float.of_int (i % 11) *. 0.125) in
  let qbv = Array.init (q * q) ~f:(fun i -> Float.of_int (i % 7) -. 3.) in
  let qcv = Array.init (q * q) ~f:(fun i -> (Float.of_int (i % 5) *. 0.5) -. 1.) in
  let qa = TDSL.ndarray qav ~label:[ "qa" ] ~input_dims:[ q ] ~output_dims:[ q ] () in
  let qb = TDSL.ndarray qbv ~label:[ "qb" ] ~input_dims:[ q ] ~output_dims:[ q ] () in
  let qc = TDSL.ndarray qcv ~label:[ "qc" ] ~input_dims:[ q ] ~output_dims:[ q ] () in
  let%op qd0 = qa + qb in
  Train.set_materialized qd0.Tensor.value;
  let%op qe0 = qd0 * qc in
  let fs_serial_comp = named "af_fs_serial" (Train.forward qe0) in
  let qsctx = Context.auto () in
  let qsctx, qsroutine =
    Context.compile ~lowered_transform:(fun opt -> opt) qsctx fs_serial_comp Ir.Indexing.Empty
  in
  let qsctx = Context.run qsctx qsroutine in
  let got_fs_serial = Context.get_values qsctx qe0.Tensor.value in
  let%op qd1 = qa + qb in
  Train.set_materialized qd1.Tensor.value;
  let%op qe1 = qd1 * qc in
  let fs_comp = named "af_fs_tuned" (Train.forward qe1) in
  let fs_report = ref None in
  let fsctx = Context.auto () in
  let fsctx, fsroutine =
    Autotune.tune ~beam_width:2 ~rounds:0 ~repeats:1 ~cache_dir:""
      ~report:(fun r -> fs_report := Some r)
      fsctx fs_comp Ir.Indexing.Empty
  in
  let fsctx = Context.run fsctx fsroutine in
  let got_fs = Context.get_values fsctx qe1.Tensor.value in
  (match !fs_report with
  | Some r ->
      (* For the 32x32x32 f32 segment site (unzeroed), cc seeds 2 packing + 2 hoisted-packing (qc is
         a hoistable constant) + 2 whole-triple tensorized pipelines + 1 packed [Tile_mma]
         composition, its hoisted, Grid-parallel hoisted-only (qd in place x hoisted-packed qc — the
         inference-GEMM shape), and Grid-parallel in-kernel variants; Metal seeds 4 blocktiling + 5
         simdgroup pipelines; other GPU backends at least the 4 blocktiling ones. *)
      p "per-segment sketch candidates seeded"
        (r.Autotune.fiss_sketch_candidates
        >=
        if is_cpu then 6 else if String.is_substring backend_name ~substring:"metal" then 9 else 4);
      p "per-segment sketch candidates timed" (r.Autotune.fiss_sketch_timed > 0)
  | None ->
      p "per-segment sketch candidates seeded" false;
      p "per-segment sketch candidates timed" false);
  p "tuned fissioned matmul matches the serial twin"
    (Array.for_all2_exn got_fs got_fs_serial ~f:approx);

  (* === Multi-site F_sketch enumeration: two matmul segments with different geometries in one
     fissioned chain. Each parameter set of each site must be proposed ALONE (the other segment
     falling back to its default preset), so the combined candidate count is [site_a + site_b] — no
     site's seed can mask another site's seeds from being timed. Zipped combos fail this: index
     pairing (n-th combo = every segment's n-th set) gives [max site_a site_b], and pinning the
     others to their first set adds masking through an invalid first seed (observed on cifar_conv,
     PR #174: the fc matmul's invalid packrest-grid seed masked the conv segments' row-block seed).
     Cross-segment combination is recovered by one extra composite candidate that applies each
     site's best-timed single simultaneously — counted in [fiss_sketch_timed] but not in the seeded
     [fiss_sketch_candidates]. === *)
  let q2 = 16 in
  let qc16v = Array.init (q2 * q) ~f:(fun i -> (Float.of_int (i % 9) *. 0.25) -. 1.) in
  let qc16 = TDSL.ndarray qc16v ~label:[ "qc16" ] ~input_dims:[ q ] ~output_dims:[ q2 ] () in
  let tune_candidates tag comp y =
    let report = ref None in
    let ctx = Context.auto () in
    let ctx, routine =
      Autotune.tune ~beam_width:2 ~rounds:0 ~repeats:1 ~cache_dir:""
        ~report:(fun r -> report := Some r)
        ctx (named tag comp) Ir.Indexing.Empty
    in
    let ctx = Context.run ctx routine in
    let got = Context.get_values ctx y.Tensor.value in
    match !report with
    | Some r -> ((r.Autotune.fiss_sketch_candidates, r.Autotune.fiss_sketch_timed), got)
    | None -> ((-1, -1), got)
  in
  let%op qd2 = qa + qb in
  Train.set_materialized qd2.Tensor.value;
  let%op qe2 = qd2 * qc in
  let (cand_a, _), _ = tune_candidates "af_ms_a" (Train.forward qe2) qe2 in
  let%op qd3 = qa + qb in
  Train.set_materialized qd3.Tensor.value;
  let%op qf3 = qc16 * qd3 in
  let (cand_b, _), _ = tune_candidates "af_ms_b" (Train.forward qf3) qf3 in
  let%op qd4 = qa + qb in
  Train.set_materialized qd4.Tensor.value;
  let%op qe4 = qd4 * qc in
  Train.set_materialized qe4.Tensor.value;
  let%op qg4 = qc16 * qe4 in
  let (cand_ab, timed_ab), got_ms = tune_candidates "af_ms_ab" (Train.forward qg4) qg4 in
  let%op qd5 = qa + qb in
  Train.set_materialized qd5.Tensor.value;
  let%op qe5 = qd5 * qc in
  Train.set_materialized qe5.Tensor.value;
  let%op qg5 = qc16 * qe5 in
  let ms_serial_comp = named "af_ms_serial" (Train.forward qg5) in
  let msctx = Context.auto () in
  let msctx, msroutine =
    Context.compile ~lowered_transform:(fun opt -> opt) msctx ms_serial_comp Ir.Indexing.Empty
  in
  let msctx = Context.run msctx msroutine in
  let got_ms_serial = Context.get_values msctx qg5.Tensor.value in
  p "multi-site: both sites seed per-segment sketches" (cand_a > 1 && cand_b > 1);
  p "multi-site: unmasked singles combo count (a + b)" (cand_ab = cand_a + cand_b);
  p "multi-site: best-timed singles recombined into a composite candidate"
    (* On cc every single compiles and times, so the composite is exactly one extra timing; on
       backends where some singles fail validation only the looser bound is stable. *)
    (if is_cpu then timed_ab = cand_ab + 1 else timed_ab > 0);
  p "multi-site: tuned two-matmul chain matches the serial twin"
    (Array.for_all2_exn got_ms got_ms_serial ~f:approx);

  (* --- timing_ctx on a different backend is rejected (Codex P2 on PR #109): candidates timed
     elsewhere do not predict the target device, and the winner would be cached under the target
     backend's key without ever having been timed on it. sync_cc vs multicore_cc are both always
     available. --- *)
  p "timing_ctx on a different backend rejected"
    (match
       Autotune.tune ~rounds:0 ~repeats:1 ~cache_dir:"" ~timing_ctx:(Context.cpu ~threads:4 ())
         (Context.cpu ()) mm_comp Ir.Indexing.Empty
     with
    | _ -> false
    | exception Invalid_argument msg -> String.is_substring msg ~substring:"same backend")
