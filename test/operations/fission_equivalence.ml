(* The autotuner's config-threshold fission candidate must reproduce the untuned default
   pipeline exactly (PR #140: "tuned >= untuned by construction" rests on it): given the same
   optimized lowering, [Sched.maybe_default_schedules] (what an untuned [Context.compile]
   applies) and the autotuner's candidate pipeline ([Sched.fission_scheduled] with the
   config-default presets — mirroring [Autotune.compile_candidate]'s [F_preset
   { block_size = None; config_thresholds = true; privatize = false }]) must produce the same
   segments. The CUDA benchmark runs observed a 26-kernel candidate against a 12-kernel untuned
   pipeline on mlp_small; this test localizes any such divergence to the pipeline invocation
   (runs everywhere) rather than benchmark-only conditions.

   The computation is a small MLP training step (two layers + relu, softmax-CE-like loss,
   gradient update + SGD) — enough structure to fission into multiple segments on GPU
   backends; on cc the comparison is trivially about the (single-segment) serial pipeline. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module LL = Ir.Low_level
module SC = Ir.Schedule_cache
module Sched = Ir.Schedule
module Asgns = Ir.Assignments

let p name b = Stdio.printf "%s: %b\n" name b
let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")

let () =
  let batch = 16 and d_in = 8 and d_hid = 16 and d_out = 4 in
  let xs =
    NTDSL.init ~l:"xs" ~prec:Ir.Ops.single ~b:[ batch ] ~i:[] ~o:[ d_in ]
      ~f:(fun idcs -> Float.of_int ((idcs.(0) + idcs.(1)) % 5) *. 0.1)
      ()
  in
  let w1v = Array.init (d_hid * d_in) ~f:(fun i -> Float.of_int (i % 7) *. 0.05) in
  let w2v = Array.init (d_out * d_hid) ~f:(fun i -> Float.of_int (i % 5) *. 0.04) in
  let w1 =
    Operation.init ~l:"w1" ~prec:Ir.Ops.single ~b:[] ~i:[ d_in ] ~o:[ d_hid ]
      ~f:(fun idcs -> w1v.((idcs.(0) * d_in) + idcs.(1)))
      ~grad_spec:Tensor.Require_grad ()
  in
  let w2 =
    Operation.init ~l:"w2" ~prec:Ir.Ops.single ~b:[] ~i:[ d_hid ] ~o:[ d_out ]
      ~f:(fun idcs -> w2v.((idcs.(0) * d_hid) + idcs.(1)))
      ~grad_spec:Tensor.Require_grad ()
  in
  let%op logits = w2 * relu (w1 * xs) in
  let%op loss = (logits *. logits) ++ "...|... => 0" in
  let update = Train.grad_update loss in
  let sgd = Train.sgd_update ~learning_rate:(TDSL.O.( !. ) 0.01) loss in
  let step = Asgns.sequence [ update; sgd ] in
  let ctx = Context.auto () in
  let ctx = Train.init_params ctx Ir.Indexing.Empty loss in
  let opt_capture = ref None in
  let _ctx, _routine =
    Context.compile
      ~lowered_transform:(fun o ->
        opt_capture := Some o;
        o)
      ctx step Ir.Indexing.Empty
  in
  let opt = Option.value_exn !opt_capture in
  let limits = Context.hardware_limits ctx in
  let static_indices = [] in
  (* Both pipelines mutate the placements fork (fission's Local promotions), so each runs on
     its own hermetic copy. *)
  let copy () =
    {
      opt with
      LL.traced_store = Hashtbl.copy opt.LL.traced_store;
      LL.optimize_ctx = LL.copy_optimize_ctx opt.LL.optimize_ctx;
    }
  in
  let untuned = Sched.maybe_default_schedules ~backend_name ~limits ~static_indices (copy ()) in
  let is_gpu = Sched.backend_is_gpu backend_name in
  let is_cpu = Sched.backend_is_cpu backend_name in
  let candidate =
    (* Mirrors Autotune.compile_candidate's preset for
       [F_preset { block_size = None; privatize = false; config_thresholds = true }]. *)
    let preset seg =
      if is_gpu then Sched.default_gpu ~limits seg
      else if is_cpu then Sched.default_cpu seg
      else []
    in
    let zero_sched tns = if is_gpu then Sched.zero_expansion ~limits tns else [] in
    List.map
      (Sched.fission_scheduled ~promote_locals:is_gpu ~preset ~zero_sched ~static_indices
         (copy ()))
      ~f:(fun (_, _, _, post) -> post)
  in
  let digests posts =
    List.map posts ~f:(fun post -> SC.digest (SC.canonicalize ~static_indices post))
  in
  (* Counts are backend-dependent (cc: single serial segment; GPU backends fission), so print
     them only on divergence to keep the expected output backend-stable. *)
  if List.length untuned <> List.length candidate then
    Stdio.printf "DIVERGENCE — untuned pipeline: %d segments; candidate pipeline: %d segments\n"
      (List.length untuned) (List.length candidate);
  p "segment counts agree" (List.length untuned = List.length candidate);
  p "per-segment digests agree"
    (List.equal String.equal (digests untuned) (digests candidate))
