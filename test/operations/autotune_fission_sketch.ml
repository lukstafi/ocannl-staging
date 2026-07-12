(* Autotune follow-ups: per-fission-segment tuning and the matmul sketch generator.

   Covered here, backend-independent (all printed booleans hold on every backend):

   - [Schedule.fission_scheduled] (the exposed fission pipeline with caller-supplied per-segment
     schedules) splits the canonical two-nest chain with a forced-materialized intermediate into
     two [`Normal] segments; compiling them through the plural [?lowered_transforms] seam executes
     correctly.
   - A hand-crafted fissioned cache entry (per-segment schedules keyed by pre-schedule segment
     digests) replays through [Autotune.tune]'s cache-hit path: the report says [fissioned], and
     the values are correct — exercising [F_saved] rebinding of per-segment saved schedules.
   - [Autotune.tune] on a fissionable computation searches whole-routine and fissioned candidates
     and returns correct values; the second call hits the cache.
   - The matmul sketch generator detects a 32x32 matmul and seeds tile-size instantiations of the
     register-blocktiling (GPU) / operand-packing (CPU) pipelines ([sketch_candidates > 0]); the
     tuned routine matches the serial twin, and Stage/Privatize round-trip through the saved form
     when a sketch wins. *)

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
  List.iter [ "autotune_fission_cache"; "autotune_fission_cache2"; "autotune_sketch_cache" ]
    ~f:clean_cache;
  (* === The fissionable chain: d = a + b (forced materialized), e = d *. d^T. The transposed
     read keeps the cross-nest edge misaligned, so the aligned cross-nest rule cannot merge the
     pair and the chain still fissions (a plain pointwise consumer would now stay one kernel). === *)
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
  p "chain tune timed multiple candidates" (r1.Autotune.candidates_timed >= 2);
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
  p "tuned matmul matches the serial twin" (Array.for_all2_exn got_mm1 got_serial ~f:approx);
  p "matmul tune searches then hits the cache"
    ((not mr1.Autotune.cache_hit) && mr2.Autotune.cache_hit);
  p "matmul cache-hit values match the serial twin"
    (Array.for_all2_exn got_mm2 got_serial ~f:approx);

  (* --- timing_ctx on a different backend is rejected (Codex P2 on PR #109): candidates timed
     elsewhere do not predict the target device, and the winner would be cached under the target
     backend's key without ever having been timed on it. sync_cc vs multicore_cc are both always
     available. --- *)
  p "timing_ctx on a different backend rejected"
    (match
       Autotune.tune ~rounds:0 ~repeats:1 ~cache_dir:""
         ~timing_ctx:(Context.cpu ~threads:4 ())
         (Context.cpu ()) mm_comp Ir.Indexing.Empty
     with
    | _ -> false
    | exception Invalid_argument msg -> String.is_substring msg ~substring:"same backend")
