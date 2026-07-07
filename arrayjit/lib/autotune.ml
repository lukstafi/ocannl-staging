open Base
module SC = Ir.Schedule_cache
module Sched = Ir.Schedule
module LL = Ir.Low_level
module Idx = Ir.Indexing

type report = {
  cache_hit : bool;
  candidates_timed : int;
  candidates_failed : int;
  rounds_run : int;
  baseline_ms : float;
  best_ms : float;
  best_schedule : SC.saved_schedule;
}

let int_arg ~arg_name ~default =
  let s = Utils.get_global_arg ~arg_name ~default:(Int.to_string default) in
  try Int.of_string (String.strip s) with _ -> default

(* A candidate round-improvement below this fraction of the incumbent ends the search. *)
let min_progress = 0.01

(** {2 Timing} *)

let set_test_bindings routine =
  List.iter (Context.bindings routine) ~f:(fun (ss, r) ->
      match ss.Idx.static_range with Some range when range > 0 -> r := range / 2 | _ -> ())

let time_routine ~repeats cctx routine =
  set_test_bindings routine;
  (* Warmup run: absorbs lazy initialization and fills caches like a steady-state iteration. *)
  let ctx = ref (Context.run cctx routine) in
  Context.sync !ctx;
  let best = ref Float.infinity in
  for _ = 1 to max 1 repeats do
    let t0 = Unix.gettimeofday () in
    ctx := Context.run !ctx routine;
    Context.sync !ctx;
    let dt = (Unix.gettimeofday () -. t0) *. 1000. in
    if Float.(dt < !best) then best := dt
  done;
  !best

(** {2 Candidate compilation}

    A candidate is a recipe producing a schedule against a {e fresh} lowering: backend [compile]
    re-lowers (with fresh symbols) on every call, so the schedule is rebound structurally inside
    the transform closure, after checking the fresh code's canonical digest against the base
    compile's. *)

type spec =
  | Saved of SC.saved_schedule  (** Replay this saved schedule. *)
  | Preset of { block_size : int option }
      (** The default annotator ({!Sched.default_gpu} / {!Sched.default_cpu}) with
          [min_parallel:1] — the tuner measures instead of guessing whether parallelism pays. *)

type compiled = {
  saved : SC.saved_schedule;
  cctx : Context.t;
  routine : Context.routine;
  opt_after : LL.optimized;
  registry : SC.registry;
  digest_after : string;
}

let compile_candidate ~static_indices ~base_digest ~limits ~is_gpu ~is_cpu ctx comp bindings spec :
    (compiled, string) Result.t =
  let captured = ref None in
  let transform opt =
    let canon = SC.canonicalize ~static_indices opt in
    if not (String.equal (SC.digest canon) base_digest) then
      invalid_arg "Autotune: fresh lowering does not match the tuned code (digest mismatch)";
    let sched, saved, registry =
      match spec with
      | Saved saved ->
          let sched, registry = SC.of_saved canon saved in
          (sched, saved, registry)
      | Preset { block_size } ->
          let sched =
            if is_gpu then Sched.default_gpu ?block_size ~min_parallel:1 ~limits opt
            else if is_cpu then Sched.default_cpu ~min_parallel:1 opt
            else []
          in
          let saved, registry = SC.to_saved (SC.base_registry canon) sched in
          (sched, saved, registry)
    in
    let opt' = Sched.apply ~static_indices sched opt in
    let digest_after = SC.digest (SC.canonicalize ~static_indices opt') in
    captured := Some (saved, registry, opt', digest_after);
    opt'
  in
  try
    let cctx, routine = Context.compile ~lowered_transform:transform ctx comp bindings in
    match !captured with
    | Some (saved, registry, opt_after, digest_after) ->
        Ok { saved; cctx; routine; opt_after; registry; digest_after }
    | None -> Error "Autotune: lowered_transform was not invoked"
  with exn -> Error (Exn.to_string exn)

(** {2 The action menu} *)

type loop_desc = {
  ld_ref : SC.sym_ref;
  ld_extent : int;
  ld_axis : LL.axis_type;
  ld_innermost : bool;
  ld_accumulating : bool;
  ld_perfect_child : (SC.sym_ref * LL.axis_type) option;
}

let rec contains_loop = function
  | LL.Seq (a, b) -> contains_loop a || contains_loop b
  | LL.If { body; _ } -> contains_loop body
  | LL.For_loop _ -> true
  | _ -> false

(* Whether the body carries a read-modify-write accumulation (a loop-carried dependency through
   memory when the written cell does not vary with the loop). Retype-to-[Vectorized] asserts
   iteration independence with no structural check downstream (the C backends emit e.g.
   [#pragma GCC ivdep] when the explicit-SIMD eligibility check falls back to pragmas), so the
   menu must not propose it over an accumulation. Conservative: [Local_scope] and [Tile_mma]
   bodies count as accumulating. *)
let rec accumulates (llc : LL.t) =
  match llc with
  | LL.Seq (a, b) -> accumulates a || accumulates b
  | LL.If { body; _ } -> accumulates body
  | LL.For_loop { body; _ } -> accumulates body
  | LL.Set { tn; llsc; _ } -> scalar_reads ~read:(`Tn tn) llsc
  | LL.Set_local (id, sc) -> scalar_reads ~read:(`Local id) sc
  | LL.Tile_mma _ -> true
  | LL.Set_from_vec _ | LL.Zero_out _ | LL.Declare_local _ | LL.Workgroup_barrier | LL.Noop
  | LL.Comment _ | LL.Staged_compilation _ ->
      false

and scalar_reads ~read (sc : LL.scalar_t) =
  let arg (s, _prec) = scalar_reads ~read s in
  match sc with
  | LL.Get (tn2, _) -> ( match read with `Tn tn -> phys_equal tn tn2 | `Local _ -> false)
  | LL.Get_local id2 -> (
      match read with `Local id -> LL.equal_scope_id id id2 | `Tn _ -> false)
  | LL.Local_scope _ -> true (* Conservative: opaque nested computation. *)
  | LL.Get_dynamic { tn = tn2; dyn_value; _ } ->
      (match read with `Tn tn -> phys_equal tn tn2 | `Local _ -> false) || arg dyn_value
  | LL.Get_merge_buffer _ -> false
  | LL.Ternop (_, a, b, c) -> arg a || arg b || arg c
  | LL.Binop (_, a, b) -> arg a || arg b
  | LL.Unop (_, a) -> arg a
  | LL.Constant _ | LL.Constant_bits _ | LL.Embed_index _ -> false

(* Loops proposable for schedule ops: the statement-level nest structure (we do not descend into
   [Local_scope] bodies or [Tile_mma] fallbacks — transforming those is never profitable and
   often invalid), restricted to loops whose binder the registry can name (Stage-internal copy
   loops cannot be referenced by a persisted schedule). *)
let collect_loops registry llc =
  let acc = ref [] in
  let rec walk = function
    | LL.Seq (a, b) ->
        walk a;
        walk b
    | LL.If { body; _ } -> walk body
    | LL.For_loop { index; from_; to_; body; axis; _ } ->
        (match SC.resolve registry index with
        | Some ld_ref when from_ = 0 ->
            let ld_perfect_child =
              match body with
              | LL.For_loop { index = ci; from_ = 0; axis = cax; _ } ->
                  Option.map (SC.resolve registry ci) ~f:(fun r -> (r, cax))
              | _ -> None
            in
            acc :=
              {
                ld_ref;
                ld_extent = to_ + 1;
                ld_axis = axis;
                ld_innermost = not (contains_loop body);
                ld_accumulating = accumulates body;
                ld_perfect_child;
              }
              :: !acc
        | _ -> ());
        walk body
    | _ -> ()
  in
  walk llc;
  List.rev !acc

(* Perfectly nested serial triples (with extents), for Tensorize proposals. *)
let collect_serial_triples registry llc =
  let acc = ref [] in
  let rec walk = function
    | LL.Seq (a, b) ->
        walk a;
        walk b
    | LL.If { body; _ } -> walk body
    | LL.For_loop { index = i; from_ = 0; to_ = ti; axis = LL.Serial; body; _ } ->
        (match body with
        | LL.For_loop
            {
              index = j;
              from_ = 0;
              to_ = tj;
              axis = LL.Serial;
              body = LL.For_loop { index = k; from_ = 0; to_ = tk; axis = LL.Serial; body = b3; _ };
              _;
            }
          when not (contains_loop b3) -> (
            match (SC.resolve registry i, SC.resolve registry j, SC.resolve registry k) with
            | Some ri, Some rj, Some rk ->
                acc := ((ri, ti + 1), (rj, tj + 1), (rk, tk + 1)) :: !acc
            | _ -> ())
        | _ -> ());
        walk body
    | LL.For_loop { body; _ } -> walk body
    | _ -> ()
  in
  walk llc;
  List.rev !acc

let split_factors = [ 2; 4; 8; 16; 32 ]
let max_actions_per_elem = 48

let menu ~is_cpu ~(limits : Ir.Backend_intf.hardware_limits) (elem : compiled) :
    SC.saved_optop list =
  let loops = collect_loops elem.registry elem.opt_after.LL.llc in
  let splits =
    List.concat_map loops ~f:(fun ld ->
        if not (LL.equal_axis_type ld.ld_axis LL.Serial) then []
        else
          List.filter_map split_factors ~f:(fun factor ->
              if factor < ld.ld_extent && ld.ld_extent % factor = 0 then
                Some
                  (SC.Split { axis = ld.ld_ref; factor; outer = LL.Serial; inner = LL.Serial })
              else None))
  in
  let swaps =
    List.filter_map loops ~f:(fun ld ->
        match (ld.ld_axis, ld.ld_perfect_child) with
        | LL.Serial, Some (child, LL.Serial) -> Some (SC.Swap { outer = ld.ld_ref; inner = child })
        | _ -> None)
  in
  let unrolls =
    List.concat_map loops ~f:(fun ld ->
        if LL.equal_axis_type ld.ld_axis LL.Serial && ld.ld_extent <= 8 then
          [
            SC.Unroll { axis = ld.ld_ref; materialize = true };
            SC.Unroll { axis = ld.ld_ref; materialize = false };
          ]
        else [])
  in
  let vectorizes =
    if not is_cpu then []
    else
      List.filter_map loops ~f:(fun ld ->
          if
            LL.equal_axis_type ld.ld_axis LL.Serial
            && ld.ld_innermost
            && not ld.ld_accumulating
          then Some (SC.Retype { axis = ld.ld_ref; ty = LL.Vectorized })
          else None)
  in
  let tensorizes =
    match limits.Ir.Backend_intf.mma with
    | None -> []
    | Some { Ir.Backend_intf.mma_simd_width; mma_tile = tm, tn, tk } ->
        (* The nesting order need not match the (i, j, k) roles — the roles are fixed by the
           accumulation pattern, which [Schedule.apply] validates (invalid permutations fail the
           candidate compile and are skipped). Propose role assignments compatible with the
           intrinsic tile's divisibility per role. *)
        List.concat_map (collect_serial_triples elem.registry elem.opt_after.LL.llc)
          ~f:(fun ((r1, e1), (r2, e2), (r3, e3)) ->
            List.filter_map
              [
                (r1, e1, r2, e2, r3, e3);
                (r1, e1, r3, e3, r2, e2);
                (r2, e2, r1, e1, r3, e3);
                (r2, e2, r3, e3, r1, e1);
                (r3, e3, r1, e1, r2, e2);
                (r3, e3, r2, e2, r1, e1);
              ]
              ~f:(fun (i, ei, j, ej, k, ek) ->
                if ei % tm = 0 && ej % tn = 0 && ek % tk = 0 then
                  Some (SC.Tensorize { i; j; k; simd_width = mma_simd_width })
                else None))
  in
  List.take
    (tensorizes @ splits @ swaps @ unrolls @ vectorizes)
    max_actions_per_elem

(** {2 The search} *)

let tune ?beam_width ?rounds ?repeats ?seed_block_sizes ?cache_dir ?report ctx comp bindings =
  let beam_width =
    max 1 (Option.value beam_width ~default:(int_arg ~arg_name:"autotune_beam_width" ~default:2))
  in
  let rounds = Option.value rounds ~default:(int_arg ~arg_name:"autotune_rounds" ~default:2) in
  let repeats = Option.value repeats ~default:(int_arg ~arg_name:"autotune_repeats" ~default:3) in
  let seed_block_sizes = Option.value seed_block_sizes ~default:[ 64; 128; 256; 512 ] in
  let cache_dir =
    Option.value cache_dir
      ~default:(Utils.get_global_arg ~arg_name:"autotune_cache_dir" ~default:"autotune_cache")
  in
  let static_indices = Idx.bound_symbols bindings in
  let backend = Context.backend_name ctx in
  let is_gpu = Sched.backend_is_gpu backend and is_cpu = Sched.backend_is_cpu backend in
  let limits = Context.hardware_limits ctx in
  (* The base compile: identity transform (= the serial baseline candidate), capturing the
     optimized code for canonicalization. *)
  let base_opt = ref None in
  let bctx, broutine =
    Context.compile
      ~lowered_transform:(fun opt ->
        base_opt := Some opt;
        opt)
      ctx comp bindings
  in
  let base_opt =
    match !base_opt with
    | Some o -> o
    | None -> failwith "Autotune.tune: backend compile did not invoke lowered_transform"
  in
  let canon = SC.canonicalize ~static_indices base_opt in
  let base_digest = SC.digest canon in
  let use_cache = (not (String.is_empty cache_dir)) && SC.complete canon in
  let key = SC.cache_key canon ~backend in
  let compile_spec = compile_candidate ~static_indices ~base_digest ~limits ~is_gpu ~is_cpu ctx comp bindings in
  let emit_report r = Option.iter report ~f:(fun f -> f r) in
  let cached =
    if use_cache then
      match SC.lookup ~dir:cache_dir ~key with
      | Some entry when String.equal entry.SC.source_digest base_digest -> (
          match compile_spec (Saved entry.SC.saved) with
          | Ok c ->
              emit_report
                {
                  cache_hit = true;
                  candidates_timed = 0;
                  candidates_failed = 0;
                  rounds_run = 0;
                  baseline_ms = entry.SC.baseline_ms;
                  best_ms = entry.SC.best_ms;
                  best_schedule = entry.SC.saved;
                };
              Some (c.cctx, c.routine)
          | Error _ -> (* Stale or corrupt entry: fall through to a fresh search. *) None)
      | _ -> None
    else None
  in
  match cached with
  | Some result -> result
  | None ->
      let seen = Hash_set.create (module String) in
      Hash_set.add seen base_digest;
      (* Baseline timing runs uncaught: its failures (e.g. uninitialized inputs) are the user's
         bug, with the same message [Context.run] would give. *)
      let baseline_ms = time_routine ~repeats bctx broutine in
      let baseline =
        {
          saved = [];
          cctx = bctx;
          routine = broutine;
          opt_after = base_opt;
          registry = SC.base_registry canon;
          digest_after = base_digest;
        }
      in
      let n_timed = ref 1 and n_failed = ref 0 in
      let try_spec spec =
        match compile_spec spec with
        | Error _ ->
            Int.incr n_failed;
            None
        | Ok c ->
            if Hash_set.mem seen c.digest_after then None
            else (
              Hash_set.add seen c.digest_after;
              match time_routine ~repeats c.cctx c.routine with
              | ms ->
                  Int.incr n_timed;
                  Some (c, ms)
              | exception _ ->
                  Int.incr n_failed;
                  None)
      in
      let seed_specs =
        Preset { block_size = None }
        :: (if is_gpu then
              List.map seed_block_sizes ~f:(fun bs -> Preset { block_size = Some bs })
            else [])
      in
      let by_time (_, a) (_, b) = Float.compare a b in
      let pool = (baseline, baseline_ms) :: List.filter_map seed_specs ~f:try_spec in
      let beam = ref (List.take (List.sort pool ~compare:by_time) beam_width) in
      let best = ref (List.hd_exn !beam) in
      let rounds_run = ref 0 in
      let continue_ = ref true in
      while !continue_ && !rounds_run < rounds do
        Int.incr rounds_run;
        let cands =
          List.concat_map !beam ~f:(fun (elem, _) ->
              List.map (menu ~is_cpu ~limits elem) ~f:(fun op -> Saved (elem.saved @ [ op ])))
        in
        let results = List.sort (List.filter_map cands ~f:try_spec) ~compare:by_time in
        match results with
        | [] -> continue_ := false
        | (_, round_best_ms) :: _ ->
            let _, incumbent_ms = !best in
            if Float.(round_best_ms < incumbent_ms *. (1. -. min_progress)) then (
              beam := List.take results beam_width;
              best := List.hd_exn !beam)
            else continue_ := false
      done;
      let best_c, best_ms = !best in
      if use_cache then
        SC.store ~dir:cache_dir ~key
          {
            SC.version = SC.entry_version;
            backend;
            source_digest = base_digest;
            saved = best_c.saved;
            best_ms;
            baseline_ms;
          };
      emit_report
        {
          cache_hit = false;
          candidates_timed = !n_timed;
          candidates_failed = !n_failed;
          rounds_run = !rounds_run;
          baseline_ms;
          best_ms;
          best_schedule = best_c.saved;
        };
      (best_c.cctx, best_c.routine)
