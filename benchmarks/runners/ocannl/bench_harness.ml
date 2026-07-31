(* Shared scaffolding for the OCANNL benchmark runners: fixture metadata access, weight injection
   into block-created params by debug-name tokens, the measurement protocol (parity losses, warmup,
   per-step-synced percentiles, queued mean), and JSON emission. See benchmarks/README.md for the
   protocol. *)

open Base
open Ocannl
module St = Safetensors
module Tn = Ir.Tnode

let get_meta st k = List.Assoc.find_exn (St.metadata st) ~equal:String.equal k
let meta_int st k = Int.of_string (get_meta st k)

(* Keys added after fixtures existed (e.g. stride1/stride2) default rather than fail, so
   pre-existing fixtures keep working. *)
let meta_int_default st k ~default =
  match List.Assoc.find (St.metadata st) ~equal:String.equal k with
  | Some v -> Int.of_string v
  | None -> default
let env_flag name = match Stdlib.Sys.getenv_opt name with Some "1" -> true | _ -> false

let percentile sorted p =
  let n = Array.length sorted in
  let idx = Float.to_int (Float.round_nearest (p /. 100. *. Float.of_int (n - 1))) in
  sorted.(idx)

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
  List.iteri segs ~f:(fun i (kind, pre, _sched, post) ->
      let _ctx', routine = Context.compile ~lowered_transform:(fun _ -> post) ctx comp bindings in
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
      let kind_s = match kind with `Normal -> "N" | `Zeros -> "Z" | `Solo -> "S" in
      let ws = String.concat ~sep:" " (List.map (writes_of pre.LL.llc) ~f:Tn.debug_name) in
      Stdio.printf "  seg%-3d %s %8.4f ms  w:%s\n" i kind_s !best ws);
  Stdio.printf "  total (sum of per-segment minima): %.4f ms\n" !total;
  Stdio.Out_channel.flush Stdio.stdout

(** Runs the measurement protocol and prints the JSON result line. [run_step] advances the batch
    binding and enqueues one step; [read_loss] returns the current loss value (awaits the device);
    [sync] awaits all queued work. *)
let measure_and_emit ~st ~backend ~variant ?(precision = "f32") ~compile_s ?tokens_per_step ~run_step
    ~read_loss ~sync ()
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
  Stdio.printf
    {|{"framework":"ocannl","backend":"%s","variant":"%s","precision":"%s","workload":"%s","compile_s":%.3f,%s"step_ms":{"p10":%.6g,"p50":%.6g,"p90":%.6g},"queued_step_ms":%.6g,"timed_steps":%d,"losses":[%s]}|}
    backend variant precision workload compile_s tokens_field (percentile synced 10.) (percentile synced 50.)
    (percentile synced 90.) queued_ms timed_steps (json_floats losses);
  Stdio.printf "\n"
