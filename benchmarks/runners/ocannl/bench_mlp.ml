(* OCANNL runner for the cross-framework benchmark suite (see benchmarks/README.md).

   Reads a self-describing safetensors fixture (initial weights, dataset, and hyperparameters in the
   metadata map), builds an n-layer relu MLP with softmax cross-entropy loss, and trains with plain
   SGD. Prints a single JSON result line on stdout: parity losses for the first [parity_steps]
   steps, then steady-state per-step wall times (per-step synced percentiles and queued mean).

   The backend is selected the usual OCANNL way (e.g. [--ocannl_backend=metal]). Environment:
   BENCH_FIXTURE is the fixture path; BENCH_TUNE=1 enables the autotuned variant
   ([Train.tune_placements]: placement A/B — the default virtual-plus-promotion graph and the
   materialize-all graph are both tuned and the measured winner kept).

   BENCH_PRECISION=bf16|f16 (gh-ocannl-492 task 4) trains under the mixed-precision recipe:
   fixture weights stay f32 masters with reduced-precision cast twins feeding the graph
   (Mixed_prec.with_master_weights), activations and gradients storage-assigned to the reduced
   precision over the logits subtree (Precision_policy.apply — the softmax-CE loss head stays
   f32), and — f16 only — dynamic loss scaling: the step becomes a gradient routine plus an
   optimizer routine gated on the host-read gradient checksum, so the reported step times include
   the per-step device sync that the inf/nan gate costs. Composing BENCH_PRECISION with BENCH_NO_SGD
   is not supported.

   BENCH_PRECISION composes with BENCH_TUNE (gh-ocannl-529): under a reduced precision it is the
   operand storage precision that decides whether a tensorized candidate is seeded at all, and on
   RDNA3/3.5 — whose WMMA has no f32-input shape — bf16 is the only tensor-core route there is. On
   the dynamic-loss-scaling legs only the gradient (or fused) routine is tuned; the step shape is
   what those legs measure, so the small optimizer routine keeps its plain compile.

   The gh-ocannl-492 task-5 gate-cost experiment legs (f16 only): BENCH_STATIC_SCALE=1 uses a
   fixed loss scale with no gate and no host read (the discriminating experiment — if f16's
   penalty collapses toward bf16's under it, the dynamic gate is confirmed as the GPU-side cost);
   BENCH_GATE_INTERVAL=N uses the fused on-device gate (Mixed_prec.gated_scaled_update) with the
   host sampling the sticky window checksum every N steps. The reported "precision" field becomes
   f16-static / f16-gatedN respectively. *)

open Base
open Ocannl
module IDX = Train.IDX
open Nn_blocks.DSL_modules
module Asgns = Ir.Assignments
module St = Safetensors

let cross_entropy_loss = Nn_blocks.cross_entropy_loss

let percentile sorted p =
  let n = Array.length sorted in
  let idx = Float.to_int (Float.round_nearest (p /. 100. *. Float.of_int (n - 1))) in
  sorted.(idx)

let () =
  let fixture = Stdlib.Sys.getenv "BENCH_FIXTURE" in
  let tune = match Stdlib.Sys.getenv_opt "BENCH_TUNE" with Some "1" -> true | _ -> false in
  let precision, mp_prec =
    match Stdlib.Sys.getenv_opt "BENCH_PRECISION" with
    | None | Some "" | Some "0" | Some "f32" -> ("f32", None)
    | Some "bf16" -> ("bf16", Some Ir.Ops.bfloat16)
    | Some "f16" -> ("f16", Some Ir.Ops.half)
    | Some other -> failwith ("bench_mlp: unknown BENCH_PRECISION: " ^ other)
  in
  (* f16 needs loss scaling (its exponent range underflows on gradients); bf16 does not. *)
  let scaled = String.equal precision "f16" in
  (* The gh-ocannl-492 task-5 experiment legs, f16 only. BENCH_STATIC_SCALE=1: a fixed loss scale
     with no inf/nan gate at all — one fused routine per step, no host read; the discriminating
     experiment for how much of f16's step cost is the dynamic gate. BENCH_GATE_INTERVAL=N: the
     fused gated recipe (Mixed_prec.gated_scaled_update) — the gate evaluated on device, the host
     sampling the sticky window checksum every N steps. Default: the two-routine host-read gate. *)
  let static_scale =
    match Stdlib.Sys.getenv_opt "BENCH_STATIC_SCALE" with Some "1" -> true | _ -> false
  in
  let gate_interval =
    Stdlib.Sys.getenv_opt "BENCH_GATE_INTERVAL" |> Option.map ~f:Int.of_string
  in
  Option.iter gate_interval ~f:(fun n ->
      if n <= 0 then failwith "bench_mlp: BENCH_GATE_INTERVAL must be a positive integer");
  if (static_scale || Option.is_some gate_interval) && not scaled then
    failwith "bench_mlp: BENCH_STATIC_SCALE / BENCH_GATE_INTERVAL require BENCH_PRECISION=f16";
  if static_scale && Option.is_some gate_interval then
    failwith "bench_mlp: BENCH_STATIC_SCALE and BENCH_GATE_INTERVAL are mutually exclusive";
  let precision =
    if static_scale then precision ^ "-static"
    else
      match gate_interval with Some n -> Printf.sprintf "%s-gated%d" precision n | None -> precision
  in
  let st = St.read fixture in
  let meta = St.metadata st in
  let get k = List.Assoc.find_exn meta ~equal:String.equal k in
  let workload = get "name" in
  let n_layers = Int.of_string (get "n_layers") in
  let batch_size = Int.of_string (get "batch_size") in
  let lr = Float.of_string (get "lr") in
  let parity_steps = Int.of_string (get "parity_steps") in
  let warmup_steps = Int.of_string (get "warmup_steps") in
  let timed_steps = Int.of_string (get "timed_steps") in
  let x_nd = St.to_ndarray st "x" in
  let y_nd = St.to_ndarray st "y" in
  let total = (Ir.Ndarray.dims x_nd).(0) in
  let n_batches = total / batch_size in
  let no_slice =
    match Stdlib.Sys.getenv_opt "BENCH_NO_SLICE" with Some "1" -> true | _ -> false
  in
  let xs = TDSL.rebatch ~l:"xs" x_nd () in
  let ys = TDSL.rebatch ~l:"ys" y_nd () in
  let batch_x, batch_y, batch_n, bindings =
    if no_slice then (
      assert (n_batches = 1);
      (xs, ys, None, IDX.empty))
    else
      let batch_n, bindings = IDX.get_static_symbol ~static_range:n_batches IDX.empty in
      let%op batch_x = xs @| batch_n in
      let%op batch_y = ys @| batch_n in
      (batch_x, batch_y, Some batch_n, bindings)
  in
  let make_params () =
    List.init n_layers ~f:(fun idx ->
        let i = idx + 1 in
        let w_name = Printf.sprintf "w%d" i in
        let b_name = Printf.sprintf "b%d" i in
        let info = Option.value_exn ~here:[%here] (St.info st w_name) in
        let din, dout =
          match info.St.shape with
          | [ o; i_ ] -> (i_, o)
          | _ -> failwith ("bench_mlp: weight is not a matrix: " ^ w_name)
        in
        let w = TDSL.wrap_param ~l:w_name ~i:[ din ] ~o:[ dout ] (St.to_ndarray st w_name) () in
        let b = TDSL.wrap_param ~l:b_name ~o:[ dout ] (St.to_ndarray st b_name) () in
        (w, b))
  in
  (* Under a reduced precision the fixture weights stay f32 masters (the sgd targets); the graph
     reads their cast twins. wrap_param routes through Tensor.param, so the hook applies. *)
  let params =
    match mp_prec with
    | Some prec -> Mixed_prec.with_master_weights ~prec make_params
    | None -> make_params ()
  in
  let materialize =
    match Stdlib.Sys.getenv_opt "BENCH_MATERIALIZE" with Some "1" -> true | _ -> false
  in
  let logits =
    List.foldi params ~init:batch_x ~f:(fun idx acc (w, b) ->
        let last = Int.equal idx (n_layers - 1) in
        let open TDSL.O in
        let z = b + (w * acc) in
        (* Explicit variant: store pre-activations for the backward pass (what the other frameworks
           do) instead of the default fully-Virtual recompute-in-backward. *)
        if materialize then Train.set_materialized z.Tensor.value;
        if last then z else relu z)
  in
  let%op batch_loss =
    cross_entropy_loss ~spec:"...|v" ~normalize_by:!..batch_size () ~logits ~targets:batch_y
  in
  (* Storage-precision policy: activations and gradients of the MLP body (the logits subtree) go
     reduced; the softmax-CE loss head and its gradients are PINNED at f32 via [except] — the
     "softmax and losses stay f32" AMP default. Merely not assigning the head is not enough:
     bottom-up precision inference would promote it to the reduced precision from the logits
     operand (e.g. the stable-softmax max_logits, whose -inf init the f16 constant guard rejects).
     Masters and twins already carry Specified precisions and are untouched. *)
  Option.iter mp_prec ~f:(fun prec ->
      let body = Hash_set.create (module Int) in
      let rec walk t =
        if not (Hash_set.mem body t.Tensor.value.Ir.Tnode.id) then (
          Hash_set.add body t.Tensor.value.Ir.Tnode.id;
          Option.iter t.Tensor.diff ~f:(fun d -> Hash_set.add body d.Tensor.grad.Ir.Tnode.id);
          List.iter t.Tensor.children ~f:(fun c -> walk c.Tensor.subtensor))
      in
      walk logits;
      let except tn = not (Hash_set.mem body tn.Ir.Tnode.id) in
      Precision_policy.apply ~except
        { Precision_policy.param_prec = None; activation_prec = Some prec; grad_prec = Some prec }
        batch_loss);
  let debug = match Stdlib.Sys.getenv_opt "BENCH_DEBUG" with Some "1" -> true | _ -> false in
  let learning_rate = TDSL.O.( !. ) lr in
  let no_sgd = match Stdlib.Sys.getenv_opt "BENCH_NO_SGD" with Some "1" -> true | _ -> false in
  if no_sgd && Option.is_some mp_prec then
    failwith "bench_mlp: BENCH_NO_SGD with BENCH_PRECISION is not supported";
  (* The f16 leg's step is a gradient routine plus an optimizer routine gated on the host-read
     gradient checksum (dynamic loss scaling); the other legs are a single fused routine. Note:
     grad_update consumes the loss's forward code, so the two step shapes must not both be
     built. *)
  let scaled_parts =
    if scaled && not static_scale then
      let scaler = Mixed_prec.Loss_scaler.create () in
      match gate_interval with
      | Some interval ->
          let wflag, comp =
            Mixed_prec.gated_scaled_update ~setup_for_parallel:debug scaler ~learning_rate
              batch_loss
          in
          Some (`Fused (scaler, wflag, comp, interval))
      | None ->
          let checksum, grad_comp =
            Mixed_prec.scaled_grad_update ~setup_for_parallel:debug scaler batch_loss
          in
          let sgd_comp = Mixed_prec.scaled_sgd_update scaler ~learning_rate batch_loss in
          Some (`HostGate (scaler, checksum, grad_comp, sgd_comp))
    else None
  in
  let step_comp =
    match scaled_parts with
    | Some _ -> Asgns.empty_comp
    | None when static_scale ->
        (* The static-scale leg: scaled backprop and unscaled optimizer as ONE routine, no
           checksum, no gate, no host read — the scale scalars are set once and never adjusted. *)
        let scaler = Mixed_prec.Loss_scaler.create () in
        Asgns.sequence
          [
            Train.grad_update ~setup_for_parallel:debug
              ~loss_scale:scaler.Mixed_prec.Loss_scaler.scale batch_loss;
            Train.sgd_update ~learning_rate ~grad_unscale:scaler.Mixed_prec.Loss_scaler.unscale
              batch_loss;
          ]
    | None ->
        let update = Train.grad_update ~setup_for_parallel:debug batch_loss in
        if no_sgd then update
        else Asgns.sequence [ update; Train.sgd_update ~learning_rate batch_loss ]
  in
  let ctx = Context.auto () in
  let backend = Context.backend_name ctx in
  let ctx = Train.init_params ctx bindings batch_loss in
  let t0 = Unix.gettimeofday () in
  (* BENCH_TUNE_REPORT=1: print both placement arms' search reports on stderr. *)
  let report =
    match Stdlib.Sys.getenv_opt "BENCH_TUNE_REPORT" with
    | Some "1" ->
        Some
          (fun (r : Autotune.report) ->
            let declines =
              String.concat ~sep:","
                (List.map r.declines ~f:(fun d ->
                     Sexp.to_string (Ir.Schedule_outcome.sexp_of_rejection_key d.key)
                     ^ "=" ^ Int.to_string d.count))
            in
            let terminal =
              Option.value_map r.terminal_failure ~default:"none" ~f:(fun failure ->
                  Sexp.to_string (Ir.Schedule_outcome.sexp_of_phase failure.phase)
                  ^ Option.value_map failure.candidate ~default:""
                      ~f:(fun candidate -> ":" ^ candidate))
            in
            Stdlib.Printf.eprintf
              "tune arm: cache_hit=%b partial=%b timed=%d failed=%d declines=[%s] terminal=%s \
               rounds=%d sketch=%d mma_candidates=%d mma_timed=%d fissioned=%b baseline_ms=%.4f \
               default_ms=%s best_ms=%.4f\n\
               %!"
              r.cache_hit r.partial r.candidates_timed r.candidates_failed declines terminal
              r.rounds_run r.sketch_candidates r.mma_candidates r.mma_timed r.fissioned
              r.baseline_ms
              (Option.value_map r.default_ms ~default:"none" ~f:(Printf.sprintf "%.4f"))
              r.best_ms)
    | _ -> None
  in
  (* BENCH_TUNE composes with every precision leg (gh-ocannl-529). It used to be rejected outright
     with BENCH_PRECISION, which made bf16 unmeasurable under autotuning — and bf16 is the ONLY
     tensor-core route on RDNA3/3.5, whose WMMA has no f32-input shape, so whether HIP even seeds a
     tensorized candidate could not be asked. (Which candidates are seeded depends on the operand
     AND accumulator storage precisions, via mma_input_formats_of_prec / mma_acc_format_of_prec
     against the backend's mma_format_tiles — gh-ocannl-545, where keying on the operands alone had
     CUDA seeding and timing 20 bf16 candidates that every emission rendered as scalar.)

     Each leg tunes the routine that carries the work. The dynamic-loss-scaling legs keep their
     step SHAPE — the gate is what is being measured — so only the gradient/fused routine is tuned
     and the tiny optimizer routine is compiled plainly, from the tuned context so the lineage's
     compile order is unchanged. Tuning runs on a scratch lineage ([timing_ctx]), so the repeated
     candidate executions never touch the benchmark's own weights or the host-side scaler state. *)
  let tuned ctx comp =
    let scratch = Train.init_params (Context.auto ()) bindings batch_loss in
    Train.tune_placements ?report ~rounds:0 ~timing_ctx:scratch ctx batch_loss comp bindings
  in
  let ctx, routines =
    match scaled_parts with
    | Some (`HostGate (scaler, checksum, grad_comp, sgd_comp)) ->
        let ctx, grad_routine =
          if tune then tuned ctx grad_comp else Context.compile ctx grad_comp bindings
        in
        let ctx, sgd_routine = Context.compile ctx sgd_comp bindings in
        (ctx, `Scaled (scaler, checksum, grad_routine, sgd_routine))
    | Some (`Fused (scaler, wflag, comp, interval)) ->
        let ctx, routine = if tune then tuned ctx comp else Context.compile ctx comp bindings in
        (ctx, `Fused (scaler, wflag, routine, interval))
    | None ->
        let ctx, routine =
          if tune then tuned ctx step_comp
          else if Lazy.force Autotune.model_default_enabled then
            (* gh-ocannl-491: the model-picked untuned default (config
               [model_default_schedule=true]) — run the same benchmark with the gate off vs on for
               the before/after comparison. *)
            Autotune.model_default ctx step_comp bindings
          else Context.compile ctx step_comp bindings
        in
        (ctx, `Plain routine)
  in
  let compile_s = Unix.gettimeofday () -. t0 in
  (* The scaled step threads the context (Loss_scaler.update overwrites the scale tensors). *)
  let ctx_ref = ref ctx in
  let step_bindings =
    match routines with
    | `Plain routine -> Context.bindings routine
    | `Scaled (_, _, grad_routine, _) -> Context.bindings grad_routine
    | `Fused (_, _, routine, _) -> Context.bindings routine
  in
  let batch_ref = Option.map batch_n ~f:(fun bn -> IDX.find_exn step_bindings bn) in
  let step_count = ref 0 in
  let run_step () =
    Option.iter batch_ref ~f:(fun r -> r := !step_count % n_batches);
    (match routines with
    | `Plain routine -> Train.run !ctx_ref routine
    | `Scaled (scaler, checksum, grad_routine, sgd_routine) ->
        let ctx', _ran =
          Mixed_prec.scaled_step ~scaler ~grad_routine ~sgd_routine ~checksum !ctx_ref
        in
        ctx_ref := ctx'
    | `Fused (scaler, wflag, routine, interval) ->
        let ctx', _window_finite =
          Mixed_prec.gated_step ~scaler ~routine ~window_checksum:wflag ~check_interval:interval
            ~step:!step_count !ctx_ref
        in
        ctx_ref := ctx');
    Int.incr step_count
  in
  let open Operation.At in
  if debug then (
    run_step ();
    List.iteri params ~f:(fun idx (_w, b) ->
        let dout = (Ir.Ndarray.dims (St.to_ndarray st (Printf.sprintf "b%d" (idx + 1)))).(0) in
        let k = min dout 4 in
        Stdio.printf "b%d grad:" (idx + 1);
        for j = 0 to k - 1 do
          Stdio.printf " %.9g" (!ctx_ref, b).@%[j]
        done;
        Stdio.printf "  value after 1 step:";
        for j = 0 to k - 1 do
          Stdio.printf " %.9g" (!ctx_ref, b).@[j]
        done;
        Stdio.printf "\n");
    Stdlib.exit 0);
  let losses =
    Array.init parity_steps ~f:(fun _ ->
        run_step ();
        (!ctx_ref, batch_loss).@[0])
  in
  for _ = 1 to warmup_steps do
    run_step ()
  done;
  Context.sync !ctx_ref;
  (* Monotonic high-resolution clock (not [Unix.gettimeofday]): on Windows the latter ticks at ~1
     ms, which floors sub-millisecond step times to 0 (see bench_harness.ml). *)
  let elapsed_ms c0 = Mtime.Span.to_float_ns (Mtime_clock.count c0) /. 1e6 in
  let synced =
    Array.init timed_steps ~f:(fun _ ->
        let c0 = Mtime_clock.counter () in
        run_step ();
        Context.sync !ctx_ref;
        elapsed_ms c0)
  in
  let c0 = Mtime_clock.counter () in
  for _ = 1 to timed_steps do
    run_step ()
  done;
  Context.sync !ctx_ref;
  let queued_ms = elapsed_ms c0 /. Float.of_int timed_steps in
  Array.sort synced ~compare:Float.compare;
  let json_floats arr =
    String.concat ~sep:"," (Array.to_list (Array.map arr ~f:(Printf.sprintf "%.9g")))
  in
  Stdio.printf
    {|{"framework":"ocannl","backend":"%s","variant":"%s","precision":"%s","workload":"%s","compile_s":%.3f,"step_ms":{"p10":%.6g,"p50":%.6g,"p90":%.6g},"queued_step_ms":%.6g,"timed_steps":%d,"losses":[%s]}|}
    backend
    (* Scheduling variant only: the storage precision is the "precision" field's business
       (gh-ocannl-539). They are independent axes, and folding a reduced precision into the
       variant made a tuned bf16 cell unnameable. *)
    (if tune then "tuned" else if materialize then "materialized" else "default")
    precision workload compile_s (percentile synced 10.) (percentile synced 50.) (percentile synced 90.)
    queued_ms timed_steps (json_floats losses);
  Stdio.printf "\n"
