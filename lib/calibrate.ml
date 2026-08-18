(** Memory-bandwidth calibration (gh-ocannl-578): STREAM-style streaming kernels whose byte counts
    are exact by construction — one access site per tensor node and direction, no guards — so the
    calibration rows they append under config [autotune_calibration_file] satisfy the fitter's
    per-leg exactness rule and can constrain the envelope's memory leg
    ([model_peak_memory_bandwidth]). Matmul-family tuning data cannot: its rows are compute-bound
    (their bytes-over-time understates achievable bandwidth no matter how exactly counted) and
    typically carry guards-taken/union upper bounds, so the fit of the memory leg needs exactly
    this kind of calibration diversity — see [tools/calibrate_bandwidth.ml] for the command-line
    entry point and [Ir.Cost_model.Calibration] for the fit semantics. *)

open Base
module Asgns = Ir.Assignments
module Idx = Ir.Indexing
open Ocannl_tensor.Operation.DSL_modules

let named name (comp : Asgns.comp) : Asgns.comp =
  { comp with asgns = Asgns.Block_comment (name, comp.asgns) }

(** [stream ?elems ?repeats ctx] times the four STREAM kernels — copy [c =: a], scale
    [b =: 0.5 *. a], add [c =: a + b], triad [a =: b + 0.5 *. c] — over 1-D single-precision
    tensors of [elems] cells each (default [2^26], 256 MiB per tensor: well past any last-level
    cache, so the measured rates are main-memory rates). Each kernel goes through
    {!Autotune.tune} with seed candidates only ([~rounds:0]) and the schedule cache disabled, so
    every timed candidate appends a calibration row through the ordinary emission path; the
    search is forced on ([~search:true]) — timing is the point of a calibration pass, whatever
    profile the config picked. Choose [elems] a power of two: an extent every workgroup size
    divides evenly keeps the parallelized candidates free of range guards, hence their rows
    bytes-exact.

    Returns the kernels' tuning reports in order, [(name, report)]. Raises {!Utils.User_error}
    when config [autotune_calibration_file] is not set — the pass would time kernels with
    nowhere to record them. *)
let stream ?(elems = 1 lsl 26) ?repeats ctx =
  let file =
    String.strip (Utils.get_global_arg ~arg_name:"autotune_calibration_file" ~default:"")
  in
  if String.is_empty file then
    raise
      (Utils.User_error
         "Calibrate.stream: config autotune_calibration_file is not set, so the calibration rows \
          would have nowhere to go");
  if elems <= 0 then raise (Utils.User_error "Calibrate.stream: elems must be positive");
  let a = NTDSL.term ~label:[ "stream_a" ] ~output_dims:[ elems ] () in
  let b = NTDSL.term ~label:[ "stream_b" ] ~output_dims:[ elems ] () in
  let c = NTDSL.term ~label:[ "stream_c" ] ~output_dims:[ elems ] () in
  List.iter [ a; b; c ] ~f:(fun t -> Train.set_materialized t.Tensor.value);
  (* The fills initialize the streams on device; values are irrelevant to the timings, and this
     routine is compiled outside [tune], so it contributes no calibration rows. *)
  let init =
    named "stream_init"
      [%cd
        a =: 2;
        b =: 0.5;
        c =: 0]
  in
  let routine = Train.to_routine ctx Idx.Empty init in
  let init_ctx = routine.Context.context in
  Train.run init_ctx routine;
  let kernels =
    [
      ("stream_copy", [%cd c =: a]);
      ("stream_scale", [%cd b =: !.0.5 *. a]);
      ("stream_add", [%cd c =: a + b]);
      ("stream_triad", [%cd a =: b + (!.0.5 *. c)]);
    ]
  in
  (* Winner contexts are released as soon as their report is captured, and the initialization
     context — which owns the stream tensors' buffers — after the last kernel, exceptional paths
     included: a long-lived caller must not accumulate three large streams (plus winners) per
     calibration call. Leaves first, then their parent, per {!Context.release}'s precondition. *)
  Exn.protect
    ~finally:(fun () -> Context.release init_ctx)
    ~f:(fun () ->
      List.map kernels ~f:(fun (name, comp) ->
          let report = ref None in
          let tuned_ctx, (_routine : Context.routine) =
            Autotune.tune ~search:true ~rounds:0 ?repeats ~cache_dir:""
              ~report:(fun r -> report := Some r)
              init_ctx (named name comp) Idx.Empty
          in
          Context.release tuned_ctx;
          match !report with
          | Some r -> (name, r)
          | None ->
              (* [tune] reports exactly once per call on every path that does any work. *)
              raise
                (Utils.User_error
                   (Printf.sprintf "Calibrate.stream: no tuning report for %s" name))))
