(* gh-ocannl-669: [Autotune.tune]'s [?name] — naming parity with [Context.compile], and the
   calibration column that reads it.

   Two facts, each invisible before this parameter existed:

   - A comp carrying no [Assignments.Block_comment] — what [Context.compile ?name] exists for, and
   how [lib/parallel.ml] builds its broadcast and all-reduce routines — can be TUNED, not merely
   compiled. Without a name the search died at its first candidate's lowering with the
   [Invalid_argument] of [Assignments.get_name_exn], deep inside the search and saying nothing about
   names. [Autotune.model_default] and [Autotune.placement_surface], the other two drop-ins for
   [Context.compile] here, are pinned on the same nameless comp. - The [routine] column of the
   calibration rows (gh-ocannl-635) READS the name the compiles were given instead of deriving it a
   second time from the block comment. Pinned with a comp that has BOTH — a block comment saying one
   thing, a [~name] saying another — so a reading and a re-derivation disagree. That divergence was
   unreachable while nothing could pass a name, and would have been silent the day something could:
   every row naming the block comment while the kernels it measured, and their generated sources,
   carried the other name.

   The calibration file is pinned by the companion dune rule
   (--ocannl_autotune_calibration_file=autotune_routine_name.tsv) and truncated here at start, so
   reruns in the same _build directory stay self-contained. Timings never appear: only names. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module Asgns = Ir.Assignments
module Cal = Ir.Cost_model.Calibration

let p = Verdict.p
let p_all = Verdict.p_all
let p_all2 = Verdict.p_all2
let approx a b = Float.(abs (a - b) < 1e-4)

(* Small, but wide enough that the preset candidates bind a hardware dimension and are therefore
   dispatched (and hence timed, and hence emit rows) on GPU backends too. *)
let dim = 64
let n = dim * dim

let () =
  let file =
    String.strip (Utils.get_global_arg ~arg_name:"autotune_calibration_file" ~default:"")
  in
  assert (not (String.is_empty file));
  if Stdlib.Sys.file_exists file then Stdlib.Sys.remove file;
  let av = Array.init n ~f:(fun i -> Float.of_int (i % 13) *. 0.5) in
  let bv = Array.init n ~f:(fun i -> Float.of_int (i % 7) -. 3.) in
  let expected = Array.init n ~f:(fun i -> av.(i) +. bv.(i)) in

  (* --- The nameless comp: [Tensor.consume_forward_code] is what [Train.forward] wraps in a block
     comment, so taking it directly is exactly the shape a caller who names the routine at the
     compile site works with. --- *)
  let a = TDSL.ndarray av ~label:[ "gh669_a" ] ~output_dims:[ dim; dim ] () in
  let b = TDSL.ndarray bv ~label:[ "gh669_b" ] ~output_dims:[ dim; dim ] () in
  let%op c = a + b in
  Train.set_materialized c.Tensor.value;
  let nameless = Tensor.consume_forward_code c in
  (* The precondition the whole test rests on: this comp has no name of its own. *)
  p "the comp carries no derivable name"
    (match Asgns.get_name_exn nameless.Asgns.asgns with
    | _ -> false
    | exception Invalid_argument _ -> true);

  let tuned_name = "gh669_nameless_tuned" in
  let ctx = Context.auto () in
  let ctx, routine =
    Autotune.tune ~name:tuned_name ~beam_width:1 ~rounds:0 ~repeats:1 ~cache_dir:"" ctx nameless
      Ir.Indexing.Empty
  in
  let ctx = Context.run ctx routine in
  p_all2 "tune ~name searches and runs a comp with no block comment"
    (Context.get_values ctx c.Tensor.value)
    expected ~f:approx;
  p "the tuned routine carries the name the search was given"
    (String.equal routine.Context.name tuned_name);

  let md_name = "gh669_nameless_model_default" in
  let mctx = Context.auto () in
  let mctx, mroutine = Autotune.model_default ~name:md_name mctx nameless Ir.Indexing.Empty in
  let mctx = Context.run mctx mroutine in
  p_all2 "model_default ~name compiles and runs the same comp"
    (Context.get_values mctx c.Tensor.value)
    expected ~f:approx;
  p "model_default's routine carries the name it was given"
    (String.equal mroutine.Context.name md_name);

  (* Analyze-only, so this pins the naming of the hermetic lowerings alone — the half of
     [Train.tune_placements] that is not a compile. *)
  let sctx = Context.auto () in
  p "placement_surface ~name reads a nameless comp's decision surface"
    (match
       Autotune.placement_surface ~name:"gh669_nameless_surface" sctx nameless Ir.Indexing.Empty
     with
    | _ -> true
    | exception Invalid_argument _ -> false);

  (* --- The bookkeeping half: a comp that DOES carry a name, tuned under a different one. --- *)
  let a2 = TDSL.ndarray av ~label:[ "gh669_a2" ] ~output_dims:[ dim; dim ] () in
  let b2 = TDSL.ndarray bv ~label:[ "gh669_b2" ] ~output_dims:[ dim; dim ] () in
  let%op e = a2 + b2 in
  Train.set_materialized e.Tensor.value;
  let block_name = "gh669_block_comment_name" in
  let passed_name = "gh669_passed_name" in
  let block_named =
    let comp = Tensor.consume_forward_code e in
    { comp with Asgns.asgns = Asgns.Block_comment (block_name, comp.Asgns.asgns) }
  in
  p "the second comp does carry a derivable name"
    (String.equal (Asgns.get_name_exn block_named.Asgns.asgns) block_name);
  let ereport = ref None in
  let ectx = Context.auto () in
  let ectx, eroutine =
    Autotune.tune ~name:passed_name ~beam_width:1 ~rounds:0 ~repeats:1 ~cache_dir:""
      ~report:(fun r -> ereport := Some r)
      ectx block_named Ir.Indexing.Empty
  in
  let ectx = Context.run ectx eroutine in
  p "the overriding name wins over the block comment for the routine"
    (String.equal eroutine.Context.name passed_name);
  p_all2 "the overridingly named search still computes correct values"
    (Context.get_values ectx e.Tensor.value)
    expected ~f:approx;
  let r = Option.value_exn ~here:[%here] !ereport in
  p "the search timed at least one candidate" (r.Autotune.candidates_timed >= 1);

  (* --- The rows both searches appended. --- *)
  let rows =
    if Stdlib.Sys.file_exists file then
      List.filter_map (Stdio.In_channel.read_lines file) ~f:Cal.of_line
    else []
  in
  p "calibration rows were appended" (not (List.is_empty rows));
  p "the overriding name reached the calibration rows"
    (List.exists rows ~f:(fun row -> String.equal row.Cal.routine passed_name));
  (* The discriminating claim: re-deriving the column would have named the block comment here. *)
  p "no calibration row names the block comment the search overrode"
    (not (List.exists rows ~f:(fun row -> String.equal row.Cal.routine block_name)));
  p_all "every calibration row names one of the searches by the name it was given" rows
    ~f:(fun row ->
      String.equal row.Cal.routine tuned_name || String.equal row.Cal.routine passed_name)
