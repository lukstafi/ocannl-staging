(* Shared scaffolding for the OCANNL benchmark runners: fixture metadata access, weight injection
   into block-created params by debug-name tokens, the reduced-precision legs and training-step
   shapes, the measurement protocol (parity losses, warmup, per-step-synced percentiles, queued
   mean), and JSON emission. See benchmarks/README.md for the protocol. *)

open Base
open Ocannl
module St = Safetensors
module Tn = Ir.Tnode
module Asgns = Ir.Assignments

let get_meta st k = List.Assoc.find_exn (St.metadata st) ~equal:String.equal k
let meta_int st k = Int.of_string (get_meta st k)

let meta_default st k ~default =
  match List.Assoc.find (St.metadata st) ~equal:String.equal k with
  | Some v -> v
  | None -> default

(** Whether the fixture describes a training workload ([mode: train], the generator's default) as
    opposed to a forward-only one ([mode: infer]). Every runner dispatches its step shape on this,
    the same way the Python runners do. *)
let is_training st = String.equal (meta_default st "mode" ~default:"train") "train"

(* Keys added after fixtures existed (e.g. stride1/stride2) default rather than fail, so
   pre-existing fixtures keep working. *)
let meta_int_default st k ~default =
  match List.Assoc.find (St.metadata st) ~equal:String.equal k with
  | Some v -> Int.of_string v
  | None -> default
let env_flag name = match Stdlib.Sys.getenv_opt name with Some "1" -> true | _ -> false

(** {1 Reduced-precision legs and training-step shapes (gh-ocannl-492 tasks 4 and 5)}

    These live here rather than in a runner because a flag implemented in {e one} runner is
    indistinguishable, from the report, from a cell nobody ran: [BENCH_STATIC_SCALE] and
    [BENCH_GATE_INTERVAL] existed in [bench_mlp] alone, which silently made the gate-cost
    experiment unavailable for every other workload — including [gpt2_mini], the matmul-dominated
    one where reduced precision matters most (gh-ocannl-551). A runner now opts in by calling
    {!precision_leg} with whether its fixture is a training one, and every leg is either available
    or refused with a message naming why. *)

type precision_leg = {
  label : string;
      (** The report's [precision] field: [f32 | bf16 | f16 | f16-static | f16-gatedN]. *)
  base : string;  (** [f32 | bf16 | f16] — the storage precision, without the gate-leg suffix. *)
  prec : Ir.Ops.prec option;  (** [None] at f32. *)
  static_scale : bool;  (** f16 with a fixed scale: no gate, no host read. *)
  gate_interval : int option;  (** f16 with the fused on-device gate, sampled every N steps. *)
  init_scale : float;
      (** The f16 loss scale to start from — the fixture's [loss_scale] metadata (torch's 65536
          when absent), overridable with [BENCH_LOSS_SCALE]. It is a workload property: a scale
          that overflows on the first step costs the dynamic legs a few backoff steps (which the
          parity window then sees) and makes the static leg — which never adjusts — diverge
          outright. *)
}

(** Parses [BENCH_PRECISION] / [BENCH_STATIC_SCALE] / [BENCH_GATE_INTERVAL]. [runner] prefixes
    error messages; [training] is {!is_training} of the fixture — the gate legs measure the cost of
    the loss-scaling gate, which only a training step has, so on a forward-only fixture they are
    refused rather than silently ignored. *)
let precision_leg ~runner ~training ?st () =
  let base, prec =
    match Stdlib.Sys.getenv_opt "BENCH_PRECISION" with
    | None | Some "" | Some "0" | Some "f32" -> ("f32", None)
    | Some "bf16" -> ("bf16", Some Ir.Ops.bfloat16)
    | Some "f16" -> ("f16", Some Ir.Ops.half)
    | Some other -> failwith (runner ^ ": unknown BENCH_PRECISION: " ^ other)
  in
  let static_scale = env_flag "BENCH_STATIC_SCALE" in
  let gate_interval =
    match Stdlib.Sys.getenv_opt "BENCH_GATE_INTERVAL" with
    | None | Some "" | Some "0" -> None
    | Some n -> Some (Int.of_string n)
  in
  Option.iter gate_interval ~f:(fun n ->
      if n <= 0 then failwith (runner ^ ": BENCH_GATE_INTERVAL must be a positive integer"));
  let gate_leg = static_scale || Option.is_some gate_interval in
  if gate_leg && not (String.equal base "f16") then
    failwith (runner ^ ": BENCH_STATIC_SCALE / BENCH_GATE_INTERVAL require BENCH_PRECISION=f16");
  if static_scale && Option.is_some gate_interval then
    failwith (runner ^ ": BENCH_STATIC_SCALE and BENCH_GATE_INTERVAL are mutually exclusive");
  if gate_leg && not training then
    failwith
      (runner
     ^ ": BENCH_STATIC_SCALE / BENCH_GATE_INTERVAL measure the loss-scaling gate, which only a \
        training step has; this fixture is forward-only (metadata mode=infer)");
  let label =
    if static_scale then base ^ "-static"
    else
      match gate_interval with Some n -> Printf.sprintf "%s-gated%d" base n | None -> base
  in
  let init_scale =
    match Stdlib.Sys.getenv_opt "BENCH_LOSS_SCALE" with
    | Some s when not (String.is_empty s) -> Float.of_string s
    | _ -> (
        match st with
        | Some st -> Float.of_string (meta_default st "loss_scale" ~default:"65536.")
        | None -> 65536.)
  in
  if Float.(init_scale <= 0.) then failwith (runner ^ ": loss scale must be positive");
  { label; base; prec; static_scale; gate_interval; init_scale }

(** The step shapes of the training legs, before compilation. *)
type train_parts =
  | Plain_step of Asgns.comp  (** One fused routine: f32, bf16, and the f16 static-scale leg. *)
  | Host_gated of Mixed_prec.Loss_scaler.t * Tensor.t * Asgns.comp * Asgns.comp
      (** Gradient routine, host-read checksum gate, optimizer routine (the f16 default). *)
  | Device_gated of Mixed_prec.Loss_scaler.t * Tensor.t * Asgns.comp * int
      (** One routine with the gate on device; the host samples the window checksum every N. *)

(** The step shape of [leg]. f16 without a gate-leg flag is the dynamic host-read gate (its
    per-step device sync is part of what the leg measures); [no_sgd] builds the gradient update
    alone (f32/bf16 only — a debugging shape, not a comparable cell). *)
let train_step_parts ?(setup_for_parallel = false) ?(no_sgd = false) ~leg ~learning_rate loss =
  match leg with
  | { base = "f16"; static_scale = false; gate_interval = Some interval; _ } ->
      let scaler = Mixed_prec.Loss_scaler.create ~init_scale:leg.init_scale () in
      let wflag, comp =
        Mixed_prec.gated_scaled_update ~setup_for_parallel scaler ~learning_rate loss
      in
      Device_gated (scaler, wflag, comp, interval)
  | { base = "f16"; static_scale = false; _ } ->
      let scaler = Mixed_prec.Loss_scaler.create ~init_scale:leg.init_scale () in
      let checksum, grad_comp = Mixed_prec.scaled_grad_update ~setup_for_parallel scaler loss in
      let sgd_comp = Mixed_prec.scaled_sgd_update scaler ~learning_rate loss in
      Host_gated (scaler, checksum, grad_comp, sgd_comp)
  | { base = "f16"; static_scale = true; _ } ->
      (* Scaled backprop and unscaled optimizer as ONE routine, no checksum, no gate, no host
         read — the scale scalars are set once and never adjusted. *)
      let scaler = Mixed_prec.Loss_scaler.create ~init_scale:leg.init_scale () in
      Plain_step
        (Asgns.sequence
           [
             Train.grad_update ~setup_for_parallel
               ~loss_scale:scaler.Mixed_prec.Loss_scaler.scale loss;
             Train.sgd_update ~learning_rate
               ~grad_unscale:scaler.Mixed_prec.Loss_scaler.unscale loss;
           ])
  | _ ->
      let update = Train.grad_update ~setup_for_parallel loss in
      Plain_step
        (if no_sgd then update
         else Asgns.sequence [ update; Train.sgd_update ~learning_rate loss ])

(** The compiled counterpart of {!train_parts}. A forward-only runner reuses [Plain] for its
    forward routine, so one step driver ({!run_train_step}) serves both modes. *)
type train_routines =
  | Plain of Context.routine
  | Host_gate of Mixed_prec.Loss_scaler.t * Tensor.t * Context.routine * Context.routine
  | Device_gate of Mixed_prec.Loss_scaler.t * Tensor.t * Context.routine * int

(** Compiles a step shape. [tuned] is the runner's autotuning compile (it needs the loss tensor,
    so the runner supplies it). Each leg tunes the routine that carries the work: the
    dynamic-loss-scaling legs keep their step SHAPE — the gate is what they measure — so only the
    gradient/fused routine is tuned and the tiny optimizer routine is compiled plainly, from the
    tuned context so the lineage's compile order is unchanged. *)
let compile_train_step ~tune ~tuned ctx bindings parts =
  match parts with
  | Plain_step comp ->
      let ctx, routine =
        if tune then tuned ctx comp
        else if Lazy.force Autotune.model_default_enabled then
          (* gh-ocannl-491: the model-picked untuned default (config
             [model_default_schedule=true]) — run the same benchmark with the gate off vs on for
             the before/after comparison. *)
          Autotune.model_default ctx comp bindings
        else Context.compile ctx comp bindings
      in
      (ctx, Plain routine)
  | Host_gated (scaler, checksum, grad_comp, sgd_comp) ->
      let ctx, grad_routine =
        if tune then tuned ctx grad_comp else Context.compile ctx grad_comp bindings
      in
      let ctx, sgd_routine = Context.compile ctx sgd_comp bindings in
      (ctx, Host_gate (scaler, checksum, grad_routine, sgd_routine))
  | Device_gated (scaler, wflag, comp, interval) ->
      let ctx, routine = if tune then tuned ctx comp else Context.compile ctx comp bindings in
      (ctx, Device_gate (scaler, wflag, routine, interval))

let train_step_bindings = function
  | Plain routine -> Context.bindings routine
  | Host_gate (_, _, grad_routine, _) -> Context.bindings grad_routine
  | Device_gate (_, _, routine, _) -> Context.bindings routine

(** Runs one step of [routines] — a training step, or a forward pass when the runner compiled its
    forward code as [Plain]. The scaled legs thread the context (the scaler overwrites the scale
    tensors), hence the reference. [step] is 0-based. *)
let run_train_step routines ctx_ref ~step =
  match routines with
  | Plain routine -> Train.run !ctx_ref routine
  | Host_gate (scaler, checksum, grad_routine, sgd_routine) ->
      let ctx, _ran = Mixed_prec.scaled_step ~scaler ~grad_routine ~sgd_routine ~checksum !ctx_ref in
      ctx_ref := ctx
  | Device_gate (scaler, wflag, routine, interval) ->
      let ctx, _window_finite =
        Mixed_prec.gated_step ~scaler ~routine ~window_checksum:wflag ~check_interval:interval ~step
          !ctx_ref
      in
      ctx_ref := ctx

let percentile sorted p =
  let n = Array.length sorted in
  let idx = Float.to_int (Float.round_nearest (p /. 100. *. Float.of_int (n - 1))) in
  sorted.(idx)

(** {1 Placement A/B arms in the emitted result (gh-ocannl-546)}

    A per-arm search outcome that never reaches the result line is invisible in every end-to-end
    number the sweep reports: a tensorized candidate can win its arm and then be discarded whole
    when the other arm ships, and the only trace is an [OCANNL_AUTOTUNE_LOG] stderr stream that a
    successful cell throws away. So each arm's crowned candidate — its label, whether it tensorizes,
    and how the best {e timed} tensorized candidate compares — is collected here and emitted with
    the measurement, where `results.jsonl` keeps it.

    {!Train.tune_placements} calls [report] once per arm, arm A (default placements) first, and
    ships the arm with the smaller [best_ms]; that is the whole attribution rule and it is applied
    below rather than guessed at. Arms are named in arrival order, so one collector describes one
    placement A/B — every step shape in {!compile_train_step} tunes exactly one routine. *)

type tune_arms = { mutable arm_reports : Autotune.report list (* reverse order *) }

let tune_arms () = { arm_reports = [] }
let collect_arm t (r : Autotune.report) = t.arm_reports <- r :: t.arm_reports

(** The [tune] JSON object, or [None] when no arm reported (an untuned cell). Times are milliseconds,
    and a time that was never measured is [null], not [inf]: [best_ms] is [infinity] when an arm
    timed nothing at all (every candidate failed and the GPU baseline was not dispatched) and
    [mma_best_ms] when it timed no tensorized candidate. Those are exactly the runs whose evidence
    this object exists to preserve, so they must not be the runs whose result line fails to parse. *)
let tune_json t =
  let ms_json v = if Float.is_inf v then "null" else Printf.sprintf "%.6g" v in
  match List.rev t.arm_reports with
  | [] -> None
  | reports ->
      let named = List.mapi reports ~f:(fun i r -> (Printf.sprintf "%c" (Char.of_int_exn (65 + i)), r)) in
      let shipped =
        List.fold named ~init:None ~f:(fun acc (name, r) ->
            match acc with
            | Some (_, best) when Float.( <= ) best r.Autotune.best_ms -> acc
            | _ -> Some (name, r.Autotune.best_ms))
        |> Option.value_map ~default:"?" ~f:fst
      in
      let arm (name, (r : Autotune.report)) =
        Printf.sprintf
          {|{"arm":"%s","best_ms":%s,"best_label":"%s","tensorized":%b,"mma_scalar_fallbacks":%d,"mma_seeded":%d,"mma_timed":%d,"mma_best_ms":%s}|}
          name
          (ms_json r.Autotune.best_ms)
          (String.substr_replace_all r.Autotune.best_label ~pattern:{|"|} ~with_:"'")
          r.Autotune.best_tensorized r.Autotune.best_mma_scalar_fallbacks r.Autotune.mma_candidates
          r.Autotune.mma_timed
          (ms_json r.Autotune.mma_best_ms)
      in
      Some
        (Printf.sprintf {|{"shipped":"%s","arms":[%s]}|} shipped
           (String.concat ~sep:"," (List.map named ~f:arm)))

let floats_of_gen g =
  let n = Array.fold (Bigarray.Genarray.dims g) ~init:1 ~f:( * ) in
  let a1 = Bigarray.reshape_1 g n in
  Array.init n ~f:(Bigarray.Array1.get a1)

(* Debug-name token matching: a param matches a fixture key when every required token appears among
   the underscore-separated tokens of its debug name. *)
let tokens_of dn = String.split dn ~on:'_' |> List.filter ~f:(Fn.non String.is_empty)

let matches ~required dn =
  let toks = tokens_of dn in
  List.for_all required ~f:(fun t -> List.mem toks t ~equal:String.equal)

(** [inject ctx st loss mapping] overwrites each param of [loss] with the fixture tensor whose
    required tokens all appear in the param's debug name. [mapping]: (fixture_key, required tokens).
    Every param must match exactly one mapping entry (and sizes must agree). Params matching no
    entry are left at their initialization (pass them deliberately!). *)
let inject ctx st loss mapping =
  Set.fold loss.Tensor.params ~init:ctx ~f:(fun ctx p ->
      let tn = p.Tensor.value in
      let dn = Tn.debug_name tn in
      match List.filter mapping ~f:(fun (_, required) -> matches ~required dn) with
      | [] -> failwith ("bench: no fixture entry matches param " ^ dn)
      | [ (key, _) ] ->
          let values = floats_of_gen (St.to_float32 st key) in
          let n = Tn.num_elems tn in
          if n <> Array.length values then
            failwith
              (Printf.sprintf "bench: %s has %d elems but fixture %s has %d" dn n key
                 (Array.length values));
          Context.set_values ctx tn values
      | ms ->
          failwith
            ("bench: param " ^ dn ^ " matches multiple fixture entries: "
            ^ String.concat ~sep:", " (List.map ms ~f:fst)))

let dump_params loss =
  Set.iter loss.Tensor.params ~f:(fun p ->
      let tn = p.Tensor.value in
      Stdio.printf "param %s dims [%s]\n" (Tn.debug_name tn)
        (String.concat ~sep:";"
           (Array.to_list (Array.map (Lazy.force tn.Tn.dims) ~f:Int.to_string))))

(** Captures [comp]'s optimized lowering, the input every diagnostic below works from. Supplying a
    [?lowered_transform] bypasses the default annotator, so the routine this links is the unscheduled
    serial form — for a large graph that is the whole working set in one work-item's stack frame, and
    on HIP the gh-ocannl-533 scratch validator declines it (gpt2_mini's forward asks for 163,856 B).
    The lowering is captured inside the transform, i.e. before codegen and link, so a typed rejection
    costs nothing here: the routine is discarded either way. An untyped failure still propagates. *)
let capture_lowering ctx comp bindings =
  let stash = ref None in
  let outcome =
    Context.compile_outcome
      ~lowered_transform:(fun opt ->
        stash := Some opt;
        opt)
      ~provenance:Ir.Schedule_outcome.User_schedule ctx comp bindings
  in
  (match (outcome, !stash) with
  | Ok _, Some _ -> ()
  | Ok _, None -> failwith "capture_lowering: the backend did not invoke lowered_transform"
  (* Failed before the transform ran: no lowering to keep, so nothing to continue with. *)
  | Error failure, None -> Ir.Schedule_outcome.raise_failure failure
  | Error (Ir.Schedule_outcome.Fatal _ as failure), Some _ ->
      Ir.Schedule_outcome.raise_failure failure
  | Error (Ir.Schedule_outcome.Classified classified), Some _ ->
      Stdio.printf "note: the unscheduled whole-routine form does not link here (%s)\n%!"
        (Ir.Schedule_outcome.detail_of_cause classified.Ir.Schedule_outcome.cause));
  Option.value_exn ~here:[%here] !stash

(** Diagnostic: prints the default fission-pipeline segment census for the captured lowered routine
    — per segment its kind, launch geometry and schedule size, and per top-level nest the loop
    extents (with axis-type letters) and written tensor nodes ([!] materialized, [~] routine-local).
    Used by the [bench_*_diag] runners; not part of the benchmark protocol. *)
let print_census ?promote_locals ~backend ~limits ~static_indices opt =
  let module LL = Ir.Low_level in
  let module Sched = Ir.Schedule in
  let stmt_detail plc stmt =
    let loops = ref [] and writes = ref [] and zeros = ref [] in
    let rec code (llc : LL.t) =
      match llc with
      | LL.Noop | LL.Comment _ | LL.Declare_local _ | LL.Staged_compilation _ | LL.Workgroup_barrier
      | LL.Tile_mma _ ->
          ()
      | LL.Seq (a, b) ->
          code a;
          code b
      | LL.For_loop { from_; to_; body; axis; _ } ->
          loops :=
            ( to_ - from_ + 1,
              match axis with
              | LL.Serial -> "s"
              | LL.Grid -> "G"
              | LL.Workgroup -> "W"
              | LL.Workgroup_reduce -> "R"
              | LL.Vectorized -> "v"
              | LL.Unrolled -> "u" )
            :: !loops;
          code body
      | LL.Zero_out tn -> zeros := tn :: !zeros
      | LL.Set { tn; _ } -> writes := tn :: !writes
      | LL.Set_dynamic { tn; _ } -> writes := tn :: !writes
      | LL.Set_from_vec { tn; _ } -> writes := tn :: !writes
      | LL.Set_local _ -> ()
      | LL.If { body; _ } -> code body
    in
    code stmt;
    let tn_s tn =
      Printf.sprintf "%s%s(%d)" (Tn.debug_name tn)
        (if Tn.Placements.is_materialized_peek plc tn then "!" else "~")
        (Tn.num_elems tn)
    in
    let loops_s =
      String.concat ~sep:"," (List.rev_map !loops ~f:(fun (n, k) -> Printf.sprintf "%d%s" n k))
    in
    let ws = List.dedup_and_sort ~compare:Tn.compare (!writes @ !zeros) in
    if List.is_empty ws && String.is_empty loops_s then None
    else
      Some (Printf.sprintf "loops[%s] w:%s" loops_s (String.concat ~sep:" " (List.map ws ~f:tn_s)))
  in
  let gpu = Sched.backend_is_gpu backend in
  let promote_locals = Option.value promote_locals ~default:gpu in
  let preset o = if gpu then Sched.default_gpu ~limits o else Sched.default_cpu o in
  let zero_sched tns = if gpu then Sched.zero_expansion ~limits tns else [] in
  let segs = Sched.fission_scheduled ~promote_locals ~preset ~zero_sched ~static_indices opt in
  Stdio.printf "default pipeline: %d segments\n" (List.length segs);
  List.iteri segs ~f:(fun i (kind, pre, sched, post) ->
      let dims = LL.launch_dims post.LL.llc in
      let np = Array.fold dims.grid ~init:1 ~f:( * ) * Array.fold dims.block ~init:1 ~f:( * ) in
      let stmts = List.length (LL.flat_lines [ post.LL.llc ]) in
      let kind_s = match kind with `Normal -> "N" | `Zeros -> "Z" | `Solo -> "S" in
      Stdio.printf "  seg%-3d %s threads=%-8d grid=[%d;%d;%d] block=[%d;%d;%d] ops=%d stmts=%d\n" i
        kind_s np dims.grid.(0) dims.grid.(1) dims.grid.(2) dims.block.(0) dims.block.(1)
        dims.block.(2) (List.length sched) stmts;
      let plc = pre.LL.optimize_ctx.LL.placements in
      List.iter (LL.flat_lines [ pre.LL.llc ]) ~f:(fun stmt ->
          match stmt_detail plc stmt with Some s -> Stdio.printf "        %s\n" s | None -> ()));
  Stdio.Out_channel.flush Stdio.stdout

(** Diagnostic companion to the [BENCH_SR_SITES] site listing (gh-ocannl-484 task 3): {e why} the
    accumulations that are absent from it were rejected. [Autotune.split_reduce_sites] returns only
    the reduction loops that clear its extent floor {e and} probe [Op_legal], and the two are
    indistinguishable from the listing alone — which left "find out why the conv-gradient
    accumulations are rejected" as the open question of the CUDA leg. This prints every low-output
    write with its enclosing loop nest and, per enclosing serial loop, the
    {!Ir.Schedule.op_legality} verdict of splitting that loop, so a missing site names the
    recognizer rule that rejected it. Used by the [bench_*_diag] runners; not part of the benchmark
    protocol. *)
let print_split_reduce_verdicts opt =
  let module LL = Ir.Low_level in
  let module Sched = Ir.Schedule in
  let module Idx = Ir.Indexing in
  (* The same output-parallelism bound the detector uses, so this probe covers exactly the writes it
     considers — a write above the bound is out of the family's scope by design, not by rejection. *)
  let out_max = 4096 in
  Stdio.printf "split-reduce probe (writes with <= %d cells):\n" out_max;
  let rec walk enclosing (llc : LL.t) =
    match llc with
    | LL.Seq (a, b) ->
        walk enclosing a;
        walk enclosing b
    | LL.If { body; _ } -> walk enclosing body
    | LL.For_loop { index; from_; to_; body; axis; _ } ->
        walk (enclosing @ [ (index, to_ - from_ + 1, axis) ]) body
    | LL.Set { tn; _ } | LL.Set_dynamic { tn; _ } ->
        let cells = try Tn.num_elems tn with _ -> 0 in
        if cells >= 1 && cells <= out_max then (
          Stdio.printf "  w:%s(%d) loops[%s]\n" (Tn.debug_name tn) cells
            (String.concat ~sep:","
               (List.map enclosing ~f:(fun (s, n, ty) ->
                    Printf.sprintf "%s=%d%s" (Idx.symbol_ident s) n
                      (match ty with LL.Serial -> "s" | _ -> "p"))));
          List.iter enclosing ~f:(fun (s, n, ty) ->
              if LL.equal_axis_type ty LL.Serial then
                let verdict =
                  match Sched.split_reduce ~axis:s ~target:tn ~num_blocks:2 with
                  | op, _, _, _ -> (
                      (* gh-ocannl-537: distinguish the rejection an interchange removes (seeding
                         hoists these and re-probes) from the ones that end the site. *)
                      let hoist () =
                        match Sched.split_reduce_hoist opt op with
                        | [] -> ""
                        | syms ->
                            " [hoistable: "
                            ^ String.concat ~sep:"," (List.map syms ~f:Idx.symbol_ident)
                            ^ "]"
                      in
                      match Sched.op_legality opt op with
                      | Sched.Op_legal -> "LEGAL"
                      | Sched.Op_illegal m -> "illegal: " ^ m ^ hoist ()
                      | Sched.Op_unknown m -> "unknown: " ^ m ^ hoist ())
                  | exception Invalid_argument m -> "raised: " ^ m
                in
                Stdio.printf "      axis %s extent %d -> %s\n" (Idx.symbol_ident s) n
                  (String.substr_replace_all verdict ~pattern:"\n" ~with_:" ")))
    | _ -> ()
  in
  walk [] opt.LL.llc;
  Stdio.Out_channel.flush Stdio.stdout

(** Diagnostic: per-segment (approximately per-layer) wall times of the default fission pipeline.
    Each segment's post-schedule code is compiled as its own routine through the
    [lowered_transform] seam (hermetic, autotune-style: the compile's fresh lowering of [comp] is
    discarded and the stashed segment substituted), then timed min-of-[repeats] with a device sync
    per run — so each number is one kernel's wall time including launch overhead. Run the full step
    once before calling so segment inputs are populated; timing mutates segment outputs (and
    re-accumulates accumulators), so restore any state that matters afterwards. [bind] binds the
    routine's static indices (e.g. the batch index). Used by the [bench_*_diag] runners; not part
    of the benchmark protocol. *)
let time_segments ?promote_locals ?(repeats = 20) ~backend ~limits ~static_indices ~ctx ~comp
    ~bindings ~bind opt =
  let module LL = Ir.Low_level in
  let module Sched = Ir.Schedule in
  let gpu = Sched.backend_is_gpu backend in
  let promote_locals = Option.value promote_locals ~default:gpu in
  let preset o = if gpu then Sched.default_gpu ~limits o else Sched.default_cpu o in
  let zero_sched tns = if gpu then Sched.zero_expansion ~limits tns else [] in
  let segs = Sched.fission_scheduled ~promote_locals ~preset ~zero_sched ~static_indices opt in
  let elapsed_ms c0 = Mtime.Span.to_float_ns (Mtime_clock.count c0) /. 1e6 in
  let writes_of llc =
    let writes = ref [] in
    let rec code (l : LL.t) =
      match l with
      | LL.Noop | LL.Comment _ | LL.Declare_local _ | LL.Staged_compilation _
      | LL.Workgroup_barrier | LL.Tile_mma _ | LL.Set_local _ ->
          ()
      | LL.Seq (a, b) ->
          code a;
          code b
      | LL.For_loop { body; _ } | LL.If { body; _ } -> code body
      | LL.Zero_out tn | LL.Set { tn; _ } | LL.Set_dynamic { tn; _ } | LL.Set_from_vec { tn; _ } ->
          writes := tn :: !writes
    in
    code llc;
    List.dedup_and_sort ~compare:Tn.compare !writes
  in
  Stdio.printf "segment times (min of %d runs, ms):\n" repeats;
  let total = ref 0. in
  let declined = ref 0 in
  List.iteri segs ~f:(fun i (kind, pre, _sched, post) ->
      let kind_s = match kind with `Normal -> "N" | `Zeros -> "Z" | `Solo -> "S" in
      let ws = String.concat ~sep:" " (List.map (writes_of pre.LL.llc) ~f:Tn.debug_name) in
      (* A segment compiled hermetically is not the segment as the full routine runs it: alone it
         keeps the whole per-thread working set that the full pipeline's promotions relieve, and on
         HIP the gh-ocannl-533 scratch validator declines it (gpt2_mini's cross-entropy head asks
         for 163,856 B per work-item). That is a limitation of this instrument, not of the workload
         — so the segment is reported as declined and the remaining ones are still timed. An
         untyped failure still propagates: [User_schedule] provenance keeps this diagnostic honest
         about compiler bugs. *)
      match
        Context.compile_outcome ~lowered_transform:(fun _ -> post)
          ~provenance:Ir.Schedule_outcome.User_schedule ctx comp bindings
      with
      | Error (Ir.Schedule_outcome.Fatal _ as failure) -> Ir.Schedule_outcome.raise_failure failure
      | Error (Ir.Schedule_outcome.Classified classified) ->
          Int.incr declined;
          Stdio.printf "  seg%-3d %s DECLINED (%s)  w:%s\n" i kind_s
            (Ir.Schedule_outcome.detail_of_cause classified.Ir.Schedule_outcome.cause)
            ws
      | Ok (_ctx', routine) ->
          bind routine;
          Train.run ctx routine;
          Context.sync ctx;
          let best = ref Float.infinity in
          for _ = 1 to repeats do
            let c0 = Mtime_clock.counter () in
            Train.run ctx routine;
            Context.sync ctx;
            best := Float.min !best (elapsed_ms c0)
          done;
          total := !total +. !best;
          Stdio.printf "  seg%-3d %s %8.4f ms  w:%s\n" i kind_s !best ws);
  Stdio.printf "  total (sum of per-segment minima): %.4f ms%s\n" !total
    (if !declined = 0 then ""
     else Printf.sprintf " (INCOMPLETE: %d of %d segments declined)" !declined (List.length segs));
  Stdio.Out_channel.flush Stdio.stdout

(** Runs the measurement protocol and prints the JSON result line. [run_step] advances the batch
    binding and enqueues one step; [read_loss] returns the current loss value (awaits the device);
    [sync] awaits all queued work. *)
let measure_and_emit ~st ~backend ~variant ?(precision = "f32") ~compile_s ?tokens_per_step ?tune
    ~run_step ~read_loss ~sync ()
    =
  let workload = get_meta st "name" in
  let parity_steps = meta_int st "parity_steps" in
  let warmup_steps = meta_int st "warmup_steps" in
  let timed_steps = meta_int st "timed_steps" in
  Stdio.eprintf "bench: compiled in %.1fs, starting %d parity steps\n%!" compile_s parity_steps;
  (* Monotonic high-resolution clock (not [Unix.gettimeofday]): on Windows the latter ticks at ~1
     ms, which floors sub-millisecond step times to 0. *)
  let elapsed_ms c0 = Mtime.Span.to_float_ns (Mtime_clock.count c0) /. 1e6 in
  let losses =
    Array.init parity_steps ~f:(fun i ->
        let c0 = Mtime_clock.counter () in
        run_step ();
        let l = read_loss () in
        Stdio.eprintf "bench: parity step %d loss %.6g (%.2fs)\n%!" i l (elapsed_ms c0 /. 1000.);
        l)
  in
  for _ = 1 to warmup_steps do
    run_step ()
  done;
  sync ();
  let synced =
    Array.init timed_steps ~f:(fun _ ->
        let c0 = Mtime_clock.counter () in
        run_step ();
        sync ();
        elapsed_ms c0)
  in
  let c0 = Mtime_clock.counter () in
  for _ = 1 to timed_steps do
    run_step ()
  done;
  sync ();
  let queued_ms = elapsed_ms c0 /. Float.of_int timed_steps in
  Array.sort synced ~compare:Float.compare;
  let json_floats arr =
    String.concat ~sep:"," (Array.to_list (Array.map arr ~f:(Printf.sprintf "%.9g")))
  in
  let tokens_field =
    match tokens_per_step with Some t -> Printf.sprintf {|"tokens_per_step":%d,|} t | None -> ""
  in
  let tune_field =
    match Option.bind tune ~f:tune_json with
    | Some j -> Printf.sprintf {|"tune":%s,|} j
    | None -> ""
  in
  Stdio.printf
    {|{"framework":"ocannl","backend":"%s","variant":"%s","precision":"%s","workload":"%s","compile_s":%.3f,%s%s"step_ms":{"p10":%.6g,"p50":%.6g,"p90":%.6g},"queued_step_ms":%.6g,"timed_steps":%d,"losses":[%s]}|}
    backend variant precision workload compile_s tokens_field tune_field (percentile synced 10.)
    (percentile synced 50.)
    (percentile synced 90.) queued_ms timed_steps (json_floats losses);
  Stdio.printf "\n"
