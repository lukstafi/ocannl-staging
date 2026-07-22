(* The analytic cost model's selection half (gh-ocannl-491 tasks 3-4): candidate scoring
   ([Autotune.model_score]), the order-preserving keep-fraction pre-filter
   ([Autotune.model_prefilter]), the seed pre-filter inside [Autotune.tune] (report fields
   [model_scored] / [model_pruned]), and the model-picked untuned default
   ([Autotune.model_default]).

   Run by a dune rule (not a test stanza) with --ocannl_backend=cc and the model_peak_* envelope
   overrides pinned on the command line: the C backends carry no advisory envelope constants, so
   the overrides are what make roofline scoring live here — deterministically, on every entry of
   the backend test matrix. The override values are chosen bandwidth-dominant (peak_flops huge,
   peak_memory_bandwidth tiny), so candidate ranking is decided by the byte counts, which differ
   across tile geometries. All printed values are booleans that hold regardless of the machine's
   SIMD width (which changes which sketch seeds exist, but not the properties asserted). *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module LL = Ir.Low_level
module Sched = Ir.Schedule
module Asgns = Ir.Assignments

let p name b = Stdio.printf "%s: %b\n" name b
let approx a b = Float.(abs (a - b) < 1e-4)

let named name (comp : Asgns.comp) : Asgns.comp =
  { comp with asgns = Asgns.Block_comment (name, comp.asgns) }

let rec first_loop (llc : LL.t) =
  match llc with
  | LL.Seq (a, b) -> ( match first_loop a with Some r -> Some r | None -> first_loop b)
  | LL.For_loop { index; body; _ } -> Some (index, body)
  | _ -> None

let first_loop_exn llc = Option.value_exn ~here:[%here] (first_loop llc)

let capture comp =
  let captured = ref None in
  let ctx = Context.auto () in
  let _ctx, _routine =
    Context.compile
      ~lowered_transform:(fun opt ->
        captured := Some opt;
        opt)
      ctx comp Ir.Indexing.Empty
  in
  Option.value_exn ~here:[%here] !captured

(* 16x16 x 16x16 matmul values, for correctness checks below. *)
let n = 16
let mav = Array.init (n * n) ~f:(fun i -> Float.of_int (i % 7) *. 0.5)
let mbv = Array.init (n * n) ~f:(fun i -> Float.of_int (i % 11) -. 4.)

let mm_expected =
  Array.init (n * n) ~f:(fun idx ->
      let i = idx / n and j = idx % n in
      let acc = ref 0. in
      for k = 0 to n - 1 do
        acc := !acc +. (mav.((i * n) + k) *. mbv.((k * n) + j))
      done;
      !acc)

let () =
  (* --- The pre-filter over fabricated scores: pure, envelope-independent --- *)
  let items = [ ("a", Some 3.); ("b", None); ("c", Some 1.); ("d", Some 2.); ("e", None) ] in
  let kept = Autotune.model_prefilter ~keep_fraction:0.5 items in
  let names = List.map kept ~f:fst in
  let mem x = List.mem names x ~equal:String.equal in
  p "prefilter keeps unscored candidates (no-coverage exemption)" (mem "b" && mem "e");
  p "prefilter keeps ceil(0.5 * 3) = 2 scored candidates, dropping the worst"
    (mem "c" && mem "d" && not (mem "a"));
  p "prefilter preserves candidate order" (List.equal String.equal names [ "b"; "c"; "d"; "e" ]);
  p "keep_fraction 1 keeps everything"
    (List.length (Autotune.model_prefilter ~keep_fraction:1. items) = 5);
  let ties = [ (1, Some 2.); (2, Some 1.); (3, Some 1.) ] in
  p "ties at the cutoff are all kept (order-independence)"
    (List.equal Int.equal (List.map (Autotune.model_prefilter ~keep_fraction:0.34 ties) ~f:fst)
       [ 2; 3 ]);
  p "at least one scored candidate always survives"
    (List.length (Autotune.model_prefilter ~keep_fraction:0.0001 ties) >= 1)

let () =
  (* --- model_score on a real lowering (envelope from the command-line overrides) --- *)
  let ma = TDSL.ndarray mav ~label:[ "sc_a" ] ~input_dims:[ n ] ~output_dims:[ n ] () in
  let mb = TDSL.ndarray mbv ~label:[ "sc_b" ] ~input_dims:[ n ] ~output_dims:[ n ] () in
  let%op mc = ma * mb in
  let opt = capture (named "score_mm" (Train.forward mc)) in
  let limits = Ir.Backend_intf.no_hardware_limits in
  let s_empty = Autotune.model_score ~static_indices:[] ~limits opt [] in
  p "empty schedule scoreable under the override envelope"
    (match s_empty with Some s -> Float.(s > 0.) | None -> false);
  let sym_i, body = first_loop_exn opt.LL.llc in
  let split_op, _o, _i = Sched.split ~axis:sym_i ~factor:4 ~outer:LL.Serial ~inner:LL.Serial in
  let s_split = Autotune.model_score ~static_indices:[] ~limits opt [ split_op ] in
  p "pure loop restructuring (Split) leaves the score unchanged"
    (match (s_empty, s_split) with Some a, Some b -> approx a b | _ -> false);
  let sym_j, _ = first_loop_exn body in
  (* Swapping a pair in the wrong nesting order fails to apply: no coverage, not an error. *)
  let bad = [ Sched.Swap { outer = sym_j; inner = sym_i } ] in
  p "an inapplicable schedule has no score (kept, only measured)"
    (Option.is_none (Autotune.model_score ~static_indices:[] ~limits opt bad))

let () =
  (* --- The seed pre-filter inside tune --- *)
  let ta = TDSL.ndarray mav ~label:[ "tn_a" ] ~input_dims:[ n ] ~output_dims:[ n ] () in
  let tb = TDSL.ndarray mbv ~label:[ "tn_b" ] ~input_dims:[ n ] ~output_dims:[ n ] () in
  let%op tc = ta * tb in
  let comp = named "prefilter_mm" (Train.forward tc) in
  let reports = ref [] in
  let tune keep_fraction =
    let ctx = Context.auto () in
    let ctx, routine =
      Autotune.tune ~beam_width:1 ~rounds:0 ~repeats:1 ~cache_dir:"" ~keep_fraction
        ~report:(fun r -> reports := r :: !reports)
        ctx comp Ir.Indexing.Empty
    in
    let ctx = Context.run ctx routine in
    Context.get_values ctx tc.Tensor.value
  in
  let got_all = tune 1.0 in
  let got_cut = tune 0.01 in
  let r_cut, r_all =
    match !reports with [ b; a ] -> (b, a) | _ -> failwith "expected two reports"
  in
  p "keep_fraction 1 turns the pre-filter off"
    (r_all.Autotune.model_scored = 0 && r_all.Autotune.model_pruned = 0);
  p "pre-filter scored the sketch candidates" (r_cut.Autotune.model_scored >= 2);
  p "pre-filter pruned the worse-scored sketches before timing"
    (r_cut.Autotune.model_pruned >= 1
    && r_cut.Autotune.model_pruned < r_cut.Autotune.model_scored);
  p "pruned search timed fewer candidates than the full search"
    (r_cut.Autotune.candidates_timed <= r_all.Autotune.candidates_timed);
  p "baseline still timed under aggressive pruning" (r_cut.Autotune.candidates_timed >= 1);
  p "tuned values correct with and without the pre-filter"
    (Array.for_all2_exn got_all mm_expected ~f:approx
    && Array.for_all2_exn got_cut mm_expected ~f:approx)

let () =
  (* --- The model-picked untuned default --- *)
  let da = TDSL.ndarray mav ~label:[ "df_a" ] ~input_dims:[ n ] ~output_dims:[ n ] () in
  let db = TDSL.ndarray mbv ~label:[ "df_b" ] ~input_dims:[ n ] ~output_dims:[ n ] () in
  let%op dc = da * db in
  let comp = named "model_default_mm" (Train.forward dc) in
  let choice = ref None in
  let ctx = Context.auto () in
  let ctx, routine =
    Autotune.model_default ~report:(fun r -> choice := Some r) ctx comp Ir.Indexing.Empty
  in
  let ctx = Context.run ctx routine in
  let got = Context.get_values ctx dc.Tensor.value in
  p "model_default returns a working routine with correct values"
    (Array.for_all2_exn got mm_expected ~f:approx);
  let r = Option.value_exn ~here:[%here] !choice in
  p "model_default scored candidates (default pipeline included)" (r.Autotune.mc_scored >= 1);
  p "model_default reports a choice with its model score"
    ((not (String.is_empty r.Autotune.mc_label))
    && match r.Autotune.mc_model_ms with Some m -> Float.(m > 0.) | None -> false)
