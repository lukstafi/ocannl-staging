(* OCANNL runner for the cross-framework benchmark suite (see benchmarks/README.md).

   Reads a self-describing safetensors fixture (initial weights, dataset, and hyperparameters in
   the metadata map), builds an n-layer relu MLP with softmax cross-entropy loss, and trains with
   plain SGD. Prints a single JSON result line on stdout: parity losses for the first
   [parity_steps] steps, then steady-state per-step wall times (per-step synced percentiles and
   queued mean).

   The backend is selected the usual OCANNL way (e.g. [--ocannl_backend=metal]). Environment:
   BENCH_FIXTURE is the fixture path; BENCH_TUNE=1 enables the autotuned variant
   ([Train.tune_placements]: placement A/B — the default virtual-plus-promotion graph and the
   materialize-all graph are both tuned and the measured winner kept). *)

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
  let params =
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
  let materialize =
    match Stdlib.Sys.getenv_opt "BENCH_MATERIALIZE" with Some "1" -> true | _ -> false
  in
  let logits =
    List.foldi params ~init:batch_x ~f:(fun idx acc (w, b) ->
        let last = Int.equal idx (n_layers - 1) in
        let open TDSL.O in
        let z = b + (w * acc) in
        (* Explicit variant: store pre-activations for the backward pass (what the other
           frameworks do) instead of the default fully-Virtual recompute-in-backward. *)
        if materialize then Train.set_materialized z.Tensor.value;
        if last then z else relu z)
  in
  let%op batch_loss =
    cross_entropy_loss ~spec:"...|v" ~normalize_by:!..batch_size () ~logits ~targets:batch_y
  in
  let debug = match Stdlib.Sys.getenv_opt "BENCH_DEBUG" with Some "1" -> true | _ -> false in
  let update = Train.grad_update ~setup_for_parallel:debug batch_loss in
  let learning_rate = TDSL.O.( !. ) lr in
  let sgd = Train.sgd_update ~learning_rate batch_loss in
  let no_sgd = match Stdlib.Sys.getenv_opt "BENCH_NO_SGD" with Some "1" -> true | _ -> false in
  let step_comp = if no_sgd then update else Asgns.sequence [ update; sgd ] in
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
            Stdlib.Printf.eprintf
              "tune arm: cache_hit=%b timed=%d failed=%d rounds=%d sketch=%d fissioned=%b \
               baseline_ms=%.4f best_ms=%.4f\n\
               %!"
              r.cache_hit r.candidates_timed r.candidates_failed r.rounds_run
              r.sketch_candidates r.fissioned r.baseline_ms r.best_ms)
    | _ -> None
  in
  let ctx, routine =
    if tune then
      let scratch = Train.init_params (Context.auto ()) bindings batch_loss in
      Train.tune_placements ?report ~rounds:0 ~timing_ctx:scratch ctx batch_loss step_comp
        bindings
    else Context.compile ctx step_comp bindings
  in
  let compile_s = Unix.gettimeofday () -. t0 in
  let batch_ref =
    Option.map batch_n ~f:(fun bn -> IDX.find_exn (Context.bindings routine) bn)
  in
  let step_count = ref 0 in
  let run_step () =
    Option.iter batch_ref ~f:(fun r -> r := !step_count % n_batches);
    Train.run ctx routine;
    Int.incr step_count
  in
  let open Operation.At in
  (if debug then (
     run_step ();
     List.iteri params ~f:(fun idx (_w, b) ->
         let dout = (Ir.Ndarray.dims (St.to_ndarray st (Printf.sprintf "b%d" (idx + 1)))).(0) in
         let k = min dout 4 in
         Stdio.printf "b%d grad:" (idx + 1);
         for j = 0 to k - 1 do
           Stdio.printf " %.9g" (ctx, b).@%[j]
         done;
         Stdio.printf "  value after 1 step:";
         for j = 0 to k - 1 do
           Stdio.printf " %.9g" (ctx, b).@[j]
         done;
         Stdio.printf "\n");
     Stdlib.exit 0));
  let losses =
    Array.init parity_steps ~f:(fun _ ->
        run_step ();
        (ctx, batch_loss).@[0])
  in
  for _ = 1 to warmup_steps do
    run_step ()
  done;
  Context.sync ctx;
  let synced =
    Array.init timed_steps ~f:(fun _ ->
        let t0 = Unix.gettimeofday () in
        run_step ();
        Context.sync ctx;
        (Unix.gettimeofday () -. t0) *. 1000.)
  in
  let t0 = Unix.gettimeofday () in
  for _ = 1 to timed_steps do
    run_step ()
  done;
  Context.sync ctx;
  let queued_ms = (Unix.gettimeofday () -. t0) /. Float.of_int timed_steps *. 1000. in
  Array.sort synced ~compare:Float.compare;
  let json_floats arr =
    String.concat ~sep:"," (Array.to_list (Array.map arr ~f:(Printf.sprintf "%.9g")))
  in
  Stdio.printf
    {|{"framework":"ocannl","backend":"%s","variant":"%s","workload":"%s","compile_s":%.3f,"step_ms":{"p10":%.6g,"p50":%.6g,"p90":%.6g},"queued_step_ms":%.6g,"timed_steps":%d,"losses":[%s]}|}
    backend
    (if tune then "tuned" else if materialize then "materialized" else "default")
    workload compile_s (percentile synced 10.) (percentile synced 50.) (percentile synced 90.)
    queued_ms timed_steps (json_floats losses);
  Stdio.printf "\n"
