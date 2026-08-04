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
   f16-static / f16-gatedN respectively. Since gh-ocannl-551 the flag parsing and the step shapes
   they select live in Bench_harness, shared with every other training runner. *)

open Base
open Ocannl
module IDX = Train.IDX
open Nn_blocks.DSL_modules
module St = Safetensors
module H = Bench_harness

let cross_entropy_loss = Nn_blocks.cross_entropy_loss

let () =
  let fixture = Stdlib.Sys.getenv "BENCH_FIXTURE" in
  let tune = H.env_flag "BENCH_TUNE" in
  let st = St.read fixture in
  let leg = H.precision_leg ~runner:"bench_mlp" ~training:(H.is_training st) ~st () in
  let mp_prec = leg.H.prec in
  let meta = St.metadata st in
  let get k = List.Assoc.find_exn meta ~equal:String.equal k in
  let n_layers = Int.of_string (get "n_layers") in
  let batch_size = Int.of_string (get "batch_size") in
  let lr = Float.of_string (get "lr") in
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
  let debug = H.env_flag "BENCH_DEBUG" in
  let learning_rate = TDSL.O.( !. ) lr in
  let no_sgd = H.env_flag "BENCH_NO_SGD" in
  if no_sgd && Option.is_some mp_prec then
    failwith "bench_mlp: BENCH_NO_SGD with BENCH_PRECISION is not supported";
  (* The step shape is the leg's business (Bench_harness.train_step_parts): the f16 default is a
     gradient routine plus an optimizer routine gated on the host-read gradient checksum (dynamic
     loss scaling), the other legs are a single fused routine. Note: grad_update consumes the
     loss's forward code, so exactly one shape may be built. *)
  let parts =
    H.train_step_parts ~setup_for_parallel:debug ~no_sgd ~leg ~learning_rate batch_loss
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
               best_ms=%.4f\n\
               %!"
              r.cache_hit r.partial r.candidates_timed r.candidates_failed declines terminal
              r.rounds_run r.sketch_candidates r.mma_candidates r.mma_timed r.fissioned
              r.baseline_ms r.best_ms)
    | _ -> None
  in
  (* BENCH_TUNE composes with every precision leg (gh-ocannl-529). It used to be rejected outright
     with BENCH_PRECISION, which made bf16 unmeasurable under autotuning — and bf16 is the ONLY
     tensor-core route on RDNA3/3.5, whose WMMA has no f32-input shape, so whether HIP even seeds a
     tensorized candidate could not be asked. (Which candidates are seeded depends on the operand
     storage precision, via mma_input_formats_of_prec against the backend's mma_format_tiles.)

     Each leg tunes the routine that carries the work. The dynamic-loss-scaling legs keep their
     step SHAPE — the gate is what is being measured — so only the gradient/fused routine is tuned
     and the tiny optimizer routine is compiled plainly, from the tuned context so the lineage's
     compile order is unchanged. Tuning runs on a scratch lineage ([timing_ctx]), so the repeated
     candidate executions never touch the benchmark's own weights or the host-side scaler state. *)
  let tuned ctx comp =
    let scratch = Train.init_params (Context.auto ()) bindings batch_loss in
    Train.tune_placements ?report ~rounds:0 ~timing_ctx:scratch ctx batch_loss comp bindings
  in
  let ctx, routines = H.compile_train_step ~tune ~tuned ctx bindings parts in
  let compile_s = Unix.gettimeofday () -. t0 in
  (* The scaled step threads the context (Loss_scaler.update overwrites the scale tensors). *)
  let ctx_ref = ref ctx in
  let batch_ref =
    Option.map batch_n ~f:(fun bn -> IDX.find_exn (H.train_step_bindings routines) bn)
  in
  let step_count = ref 0 in
  let run_step () =
    Option.iter batch_ref ~f:(fun r -> r := !step_count % n_batches);
    H.run_train_step routines ctx_ref ~step:!step_count;
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
  H.measure_and_emit ~st ~backend
    ~variant:
      (* Scheduling variant only: the storage precision is the "precision" field's business
         (gh-ocannl-539). They are independent axes, and folding a reduced precision into the
         variant made a tuned bf16 cell unnameable. *)
      (if tune then "tuned" else if materialize then "materialized" else "default")
    ~precision:leg.H.label ~compile_s ~run_step
    ~read_loss:(fun () -> (!ctx_ref, batch_loss).@[0])
    ~sync:(fun () -> Context.sync !ctx_ref)
    ()
