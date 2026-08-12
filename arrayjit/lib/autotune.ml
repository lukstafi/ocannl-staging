open Base
module SC = Ir.Schedule_cache
module Sched = Ir.Schedule
module Sspace = Ir.Schedule_space
module LL = Ir.Low_level
module Idx = Ir.Indexing
module Outcome = Ir.Schedule_outcome

type decline_summary = {
  key : Outcome.rejection_key;
  count : int;
  sample_details : string list;
}

type terminal_failure = {
  phase : Outcome.phase;
  candidate : string option;
  detail : string;
}

type report = {
  cache_hit : bool;
  candidates_timed : int;
  candidates_failed : int;
  partial : bool;
  baseline_declined : bool;
  declines : decline_summary list;
  terminal_failure : terminal_failure option;
  rounds_run : int;
  sketch_candidates : int;
  epilogue_sketch_candidates : int;
  fiss_sketch_candidates : int;
  fiss_sketch_timed : int;
  split_reduce_candidates : int;
  split_reduce_timed : int;
  mma_candidates : int;
      (** Candidates whose label promises a tensorized pipeline ([spec_expects_mma]) that the search
          put through candidate compile: whole-routine and per-fission-segment seeds, the
          cross-segment recombination composite, and beam-expansion candidates. *)
  mma_timed : int;
      (** How many of [mma_candidates] survived candidate compile far enough to be TIMED. A search
          with [mma_candidates > 0] and [mma_timed = 0] never measured a tensorized pipeline at all
          — the state gh-ocannl-521 records for every GPU backend. Dedup'd candidates do not count:
          a duplicate digest means an identical candidate was already timed. *)
  model_scored : int;
  model_pruned : int;
  fissioned : bool;
  baseline_ms : float;
  default_ms : float option;
  best_ms : float;
  best_label : string;
  best_tensorized : bool;
  best_mma_statements : int;
  best_mma_scalar_fallbacks : int;
  mma_best_ms : float;
      (** The best timed tensorized candidate's time (gh-ocannl-546), [infinity] when none was
          timed. Its margin against [best_ms] is what tells a crowned tensorization apart from one
          that lost by 1% and one that lost by 40%. *)
  best_schedule : SC.saved_schedule;
}

(** The report of a [tune] call that never searched (config [autotune_search=false], gh-ocannl-559):
    every counter zero and every time [infinity], like a search whose candidates all failed. The
    caller gets the untuned default compile; [best_label] says why. *)
let no_search_report =
  {
    cache_hit = false;
    candidates_timed = 0;
    candidates_failed = 0;
    partial = false;
    baseline_declined = false;
    declines = [];
    terminal_failure = None;
    rounds_run = 0;
    sketch_candidates = 0;
    epilogue_sketch_candidates = 0;
    fiss_sketch_candidates = 0;
    fiss_sketch_timed = 0;
    split_reduce_candidates = 0;
    split_reduce_timed = 0;
    mma_candidates = 0;
    mma_timed = 0;
    model_scored = 0;
    model_pruned = 0;
    fissioned = false;
    baseline_ms = Float.infinity;
    default_ms = None;
    best_ms = Float.infinity;
    best_label = "search disabled";
    best_tensorized = false;
    best_mma_statements = 0;
    best_mma_scalar_fallbacks = 0;
    mma_best_ms = Float.infinity;
    best_schedule = [];
  }

(* Best-effort reporting must stay best-effort for ordinary callback errors and NOT for these: an
   interrupt or a runtime-fatal condition raised inside a [report] callback is about the process, and
   swallowing it (on a path that is already failing) would, for a caller that CONTAINS the failure
   per arm, let a long search carry on through a Ctrl-C (gh-ocannl-550). Same set
   {!Ir.Schedule_outcome.classify_raw} refuses to classify. *)
let process_fatal_exn = function
  | Out_of_memory | Stdlib.Sys.Break | Stack_overflow | Assert_failure _ -> true
  | _ -> false

type decline_acc = { mutable da_count : int; mutable da_details : string list }

(* Where the candidate died, for the per-candidate log line. Compile-side phases are already
   apparent from the message; the launch/sync split is not, and it is the difference between "this
   schedule could never run" and "it ran and the device complained". *)
let phase_label (phase : Outcome.phase) = Sexp.to_string (Outcome.sexp_of_phase phase)

let record_decline declines (classified : Outcome.classified_cause) =
  let key = Outcome.key_of_cause classified.cause in
  let detail = Outcome.detail_of_cause classified.cause in
  let first_for_key = not (Hashtbl.mem declines key) in
  Hashtbl.update declines key ~f:(function
    | None -> { da_count = 1; da_details = [ detail ] }
    | Some acc ->
        acc.da_count <- acc.da_count + 1;
        if
          List.length acc.da_details < 3
          && not (List.mem acc.da_details detail ~equal:String.equal)
        then acc.da_details <- acc.da_details @ [ detail ];
        acc);
  if first_for_key then
    match classified.cause with
    (* Unclassified by construction, and contained under strict classification too (gh-ocannl-564),
       so the warning below — about a compile-side failure only permissiveness absorbed — would be
       saying something false about it. *)
    | Outcome.Unclassified { phase = Outcome.Preflight; _ } -> ()
    | Outcome.Unclassified _ ->
        Stdio.eprintf
          "autotune: WARNING: permissive failure classification contained an unclassified \
           compiler failure (%s); strict_failure_classification=true would stop the search\n%!"
          detail
    | _ -> ()

let decline_summaries declines =
  Hashtbl.to_alist declines
  |> List.sort ~compare:(fun (a, _) (b, _) -> Outcome.compare_rejection_key a b)
  |> List.map ~f:(fun (key, acc) ->
         { key; count = acc.da_count; sample_details = acc.da_details })

let failed_count declines =
  Hashtbl.fold declines ~init:0 ~f:(fun ~key:_ ~data:acc count -> count + acc.da_count)

let int_arg ~arg_name ~default =
  let s = Utils.get_global_arg ~arg_name ~default:(Int.to_string default) in
  try Int.of_string (String.strip s) with _ -> default

let float_arg ~arg_name ~default =
  let s = Utils.get_global_arg ~arg_name ~default:(Float.to_string default) in
  try Float.of_string (String.strip s) with _ -> default

(* A candidate round-improvement below this fraction of the incumbent ends the search. *)
let min_progress = 0.01

(* The beam holds no compiled candidate exactly when nothing was timed, which every consumer of the
   winner tests first ([nothing_timed]). *)
let timed_winner_exists = "Autotune.tune: a finite best time without a compiled candidate"

(** {2 Timing} *)

let set_test_bindings routine =
  List.iter (Context.bindings routine) ~f:(fun (ss, r) ->
      match ss.Idx.static_range with
      | Some range when range > 0 && ss.Idx.used_as_extent ->
          (* gh-490 symbolic extents: tune at the upper bound. The schedule digest is
             extent-value-independent (the extent is a kernel parameter), so one tuned entry serves
             every extent; measuring at the maximum makes the tuned schedule's cost model
             conservative for smaller runtime extents. *)
          r := range
      | Some range when range > 0 -> r := range / 2
      | _ -> ())

(* Fast routines get extra timed runs beyond [repeats], until this much total measured time (or
   [max_timing_runs]): on sub-millisecond kernels a min-of-3 is dominated by launch jitter, and the
   winner selection becomes a lottery — a heavier candidate can be crowned by one lucky sample while
   the true winner's few samples all landed under contention. Noise only ever adds time, so min-of-N
   converges monotonically to the true best case and more samples strictly reduce mis-selection; for
   routines slower than [min_timing_ms / repeats] per run nothing changes. *)
let min_timing_ms = 25.
let max_timing_runs = 64

(* Sibling fault-injection seam to [on_candidate_attempt], at a timing run's pre-dispatch validation
   rather than at a candidate's compile (gh-ocannl-564). Default no-op, no config key selects it.
   Needed because the causes this phase contains — an unsatisfied dependency, an out-of-range
   binding — belong to the lineage and the bindings, so a genuine one hits every candidate at once
   and cannot express "this one declined, the search went on". *)
let on_candidate_preflight : (string -> unit) ref = ref (fun _routine_name -> ())

(* [Context.bindings] exposes the routine's live binding refs — restore them after timing (Codex P2
   on PR #103), or the returned winner would stay bound to the tuner's midpoint test values. *)
let time_routine ?(tag_failures = false) ~repeats cctx routine =
  let saved_bindings = List.map (Context.bindings routine) ~f:(fun (_ss, r) -> (r, !r)) in
  let run ctx =
    if tag_failures then Outcome.tag Outcome.Launch (fun () -> Context.run ctx routine)
    else Context.run ctx routine
  in
  let sync ctx =
    if tag_failures then Outcome.tag Outcome.Sync (fun () -> Context.sync ctx) else Context.sync ctx
  in
  Exn.protect
    ~finally:(fun () -> List.iter saved_bindings ~f:(fun (r, v) -> r := v))
    ~f:(fun () ->
      set_test_bindings routine;
      (* The runs' pre-dispatch validation, in its own phase so an unattributed failure of it is
         contained rather than condemning the lineage (gh-ocannl-564). Here and once: what it checks
         (lineage, initialized nodes, dependencies, the bindings just written) is settled before the
         warmup and only becomes more satisfied as the loop dispatches. [Context.run] re-validates
         per iteration inside the [Launch] tag, where it can no longer fail. *)
      (* Only the PER-CANDIDATE half of the pre-dispatch validation is contained here. The
         lineage-wide half ({!Context.check_lineage_runnable}) is run by the callers below, outside
         their failure boundaries, because it fails every candidate of every arm identically —
         see the comments at those two sites (gh-ocannl-569). *)
      if tag_failures then
        Outcome.tag Outcome.Preflight (fun () ->
            !on_candidate_preflight (Context.routine_name routine);
            Context.check_launch_bindings routine);
      (* Warmup run: absorbs lazy initialization and fills caches like a steady-state iteration. *)
      let ctx = ref (run cctx) in
      sync !ctx;
      let best = ref Float.infinity in
      let total = ref 0. in
      let count = ref 0 in
      while
        !count < max 1 repeats || (Float.(!total < min_timing_ms) && !count < max_timing_runs)
      do
        (* Monotonic high-resolution clock: on Windows, [Unix.gettimeofday] ticks at ~1 ms, which
           makes sub-millisecond candidates indistinguishable (they all measure 0). *)
        let c0 = Mtime_clock.counter () in
        ctx := run !ctx;
        sync !ctx;
        let dt = Mtime.Span.to_float_ns (Mtime_clock.count c0) /. 1e6 in
        total := !total +. dt;
        Int.incr count;
        if Float.(dt < !best) then best := dt
      done;
      !best)

(* gh-ocannl-532: on a GPU backend, code that binds no hardware dimension runs the whole routine in
   a single work-item — every nest a serial scalar loop, at one lane's throughput. Such a candidate
   cannot win a search whose other candidates are parallel, so dispatching it is pure cost, and the
   cost is unbounded: a training step of a few GFLOP is minutes to hours per run, and
   [time_routine] does four of them (a warmup plus [autotune_repeats]). The dispatch is also
   uninterruptible and shares the device with the display — the sessions in gh-ocannl-532 produced
   driver-timeout reports and, once, loss of display output. So an unparallelized GPU candidate is
   never dispatched: not timed, and not eligible to win. This covers the identity-transform serial
   baseline, which is where it bites (the default annotator that parallelizes an untuned compile is
   bypassed whenever a [?lowered_transform] is supplied, so the tuner's base compile is always the
   unscheduled form). On CPU backends the serial form runs at full single-core speed and stays a
   legitimate competitor — the rule is GPU-only. *)
let binds_hardware_dims (opt : LL.optimized) = not (List.is_empty (LL.hardware_axes opt.LL.llc))

(* A candidate is dispatchable when it is on a CPU backend, or at least one of its kernels binds a
   hardware dimension. Whole-candidate rather than per-kernel: a fissioned candidate legitimately
   leaves small segments serial next to parallel ones, and only an entirely serial routine has the
   unbounded single-work-item cost. *)
let dispatchable ~is_gpu (opts : LL.optimized list) =
  (not is_gpu) || List.exists opts ~f:binds_hardware_dims

let axis_type_is_hardware = function
  | LL.Grid | LL.Workgroup | LL.Workgroup_reduce -> true
  | LL.Serial | LL.Unrolled | LL.Vectorized -> false

(* Whether a menu move could turn a form that binds no hardware dimension into one that does. Only
   two families can: a placement retype (or a [Split] whose halves are hardware-typed), and
   [Tensorize], whose lane loop is a fresh [Workgroup] axis — which is exactly the move the seeding
   comments call the beam's one path out of the serial baseline. The moves [menu] actually proposes
   otherwise rewrite serial loops into serial loops ([Split] Serial/Serial, [Swap], [Unroll],
   [Retype] to [Vectorized]), so extending an undispatchable incumbent with them yields another
   undispatchable candidate — provable without compiling it (gh-ocannl-543). Families [menu] does
   not emit answer [true]: not pruning is the conservative side, so a future menu addition is never
   silently dropped. *)
let optop_can_bind_hardware (op : SC.saved_optop) =
  match op with
  | SC.Split { outer; inner; _ } -> axis_type_is_hardware outer || axis_type_is_hardware inner
  | SC.Retype { ty; _ } -> axis_type_is_hardware ty
  | SC.Swap _ | SC.Unroll _ -> false
  | SC.Tensorize _ | SC.Partition _ | SC.Pad _ | SC.Stage _ | SC.Privatize _ | SC.Expand_zero _
  | SC.Fuse_epilogue _ | SC.Split_reduce _ ->
      true

let optop_family (op : SC.saved_optop) =
  match op with
  | SC.Split _ -> "Split"
  | SC.Swap _ -> "Swap"
  | SC.Retype _ -> "Retype"
  | SC.Unroll _ -> "Unroll"
  | SC.Partition _ -> "Partition"
  | SC.Pad _ -> "Pad"
  | SC.Stage _ -> "Stage"
  | SC.Privatize _ -> "Privatize"
  | SC.Expand_zero _ -> "Expand_zero"
  | SC.Tensorize _ -> "Tensorize"
  | SC.Fuse_epilogue _ -> "Fuse_epilogue"
  | SC.Split_reduce _ -> "Split_reduce"

(** {2 Matmul detection and sketch schedules}

    Sketch candidates instantiate the composed matmul pipelines pinned by
    test/operations/schedule_register_matmul.ml (GPU register blocktiling: Split + Swap + shared
    Stage + Privatize + materializing Unroll) and schedule_cpu_pack_matmul.ml (CPU operand packing:
    Split + Swap + non-shared Stage + Privatize), parameterized by tile sizes. Detection is
    permissive — a mis-detected site fails its candidate compile (op preconditions,
    [validate_parallel], hardware limits) and is skipped like any other invalid candidate. *)

type sketch_params = {
  sk_gpu : bool;  (** Register blocktiling with shared staging vs. CPU operand packing. *)
  sk_mma : bool;
      (** Tensorized (tile-MMA) pipeline instead of the scalar blocktiling/packing one: on GPU,
          Split → (optional cooperative shared Stage) → Tensorize targeting [simdgroup_matrix] /
          tensor cores; on cc, the whole-triple [Tile_mma] rendered register-tiled (gh-ocannl-469),
          optionally Grid-parallel over row blocks — or, with [sk_bk > 0], the cache-blocked packed
          composition (packing Stages feeding the register-tiled kernel;
          [cpu_mma_pack_sketch_schedule]), itself optionally Grid-parallel ([sk_grid]: hoisted
          packing runs Grid-outermost; in-kernel packing relies on the renderer's per-chunk tile
          privatization). Seeded directly because the greedy menu cannot reach the composition: a
          bare [Tensorize] from the serial baseline (one simdgroup, everything else serial) loses
          round 1 and the beam discards it before Grid retypes could join it. *)
  sk_simd : int;  (** MMA lane width ([hardware_limits.mma_simd_width]); 0 when [not sk_mma]. *)
  sk_bm : int;
  sk_bn : int;
  sk_bk : int;
      (** For GPU MMA sketches, [sk_bk = 0] = unstaged (one full-K [Tile_mma] block). For conv GPU
          seeds, [sk_bn]/[sk_bk] are re-purposed as the pad-to multiples of the column/reduction
          extents (gh-ocannl-485; 0 = already an intrinsic-tile multiple). *)
  sk_tm : int;
      (** Register-tile factors; unused on CPU. For conv GPU seeds, [sk_tm] is re-purposed as the
          row pad-to multiple of the unblocked flavor (gh-ocannl-485; 0 = no pad). *)
  sk_tn : int;
  sk_hoist : bool;
      (** CPU packing only: pack compile-time-constant operands out of the routine, into the
          per-device constant pool (gh-ocannl-470). Proposed alongside the in-kernel packing variant
          so the choice stays measured; applied per operand, only to hoistable (known-constant,
          host-init-backed) sources. *)
  sk_grid : bool;
      (** CPU packed composition only ([sk_mma] with [sk_bk > 0]): split [i] into pool-parallel
          [Grid] row blocks instead of Serial ones. Four shapes, keyed by [sk_hoist] and
          [sk_pack_rest]:

          - With [sk_hoist] alone, hoisted-only packing: only hoistable operands are packed (at link
            time, into the constant pool) and the rest are read in place, leaving the kernel body
            all-materialized; the Grid loop stays outermost (one dispatch spanning the whole GEBP
            triple). The typical inference GEMM: activations (in place) x constant weights.
          - With [sk_hoist] and [sk_pack_rest], the mixed grid-outermost shape (gh-ocannl-473):
            hoistable operands still pack at link time, but a non-hoistable operand gets an
            in-kernel packing Stage instead of being read in place — its tile lands inside the Grid
            body and is privatized to per-chunk block-scope storage by the renderer. For the
            inference GEMM this recovers the A~ pack the hoisted-only shape forfeits (a per-chunk
            [bm x bk] tile) while keeping the single outermost dispatch.
          - With [sk_pack_rest] alone, grid-outermost in-kernel packing (gh-ocannl-475): both
            operands pack inside the Grid body and privatize per chunk — each chunk re-packs its own
            B~ panel (redundant copies, but one dispatch instead of one per k-block). Needs the
            tiles under the renderer's per-chunk privatization cap (config
            [cc_grid_private_bytes_cap]).
          - Without [sk_hoist] or [sk_pack_rest], in-kernel packing: the per-row-block A~ packing
            Stage lands inside the Grid body — its tile is privatized to per-chunk block-scope
            storage by the renderer ([C_syntax.parallel_grid_safe]'s privatization rule) — while the
            B~ panel packs at the k-block loop outside the Grid and is read-only inside (shared
            across the row-block chunks, behind a pointer alias under the blocks extension),
            re-entering the parallel construct once per k-block.

          Proposed alongside the serial flavors so the choice stays measured. *)
  sk_pack_rest : bool;
      (** Grid-outermost packed compositions only (with [sk_grid]): give non-hoistable operands a
          non-hoisted in-kernel packing Stage instead of reading them in place, relying on the
          renderer's per-chunk tile privatization. With [sk_hoist], the mixed shape of gh-ocannl-473
          (hoisted constant panel + per-chunk pack of the rest); without [sk_hoist], the per-chunk
          B~ re-packing shape of gh-ocannl-475 — the Grid loop stays outermost (one dispatch
          spanning the GEBP triple) and every operand packs inside the Grid body. No effect on the
          serial flavors or the hoisted-only Grid flavor, whose stages are already determined. *)
  sk_conv : bool;
      (** Convolution site (gh-ocannl-493): the seed instantiates the implicit-GEMM conv pipeline
          ([cpu_conv_sketch_schedule] / [gpu_conv_sketch_schedule] via [detect_conv]) instead of a
          matmul one. The packing [Stage] serves as im2col and the micro-kernel is the ordinary
          [Tile_mma] ([sk_mma] is set so the census expectations apply). On CPU, [sk_grid]
          pool-parallelizes the outermost batch/spatial loop — on merged segments with the aligned
          whole-segment geometry of the default preset ([conv_aligned_grid]). On GPU backends with
          an mma capability ([sk_gpu] with [sk_simd] the lane width), the staged pipeline: outer
          loops [Grid]-typed, cooperative shared-tile staging, the accumulator fragment resident
          across the kernel window (gh-ocannl-480). *)
  sk_epilogue : bool;
      (** Epilogue fusion (gh-ocannl-486): append [Sched.Fuse_epilogue] on the site's output, so the
          sole-consumer elementwise tail (bias add / activation / residual) folds into the
          store-back and the whole routine is one kernel — the fused competitor to the fissioned
          two-kernel form. Seeded only when [Sched.can_fuse_epilogue] holds on the base code; a
          candidate whose scheduled form no longer admits the fusion (e.g. materializing unrolls
          duplicating the store-back) fails its compile and is skipped like any other invalid
          candidate. On GPU the accumulator moves to workgroup-shared memory (the [shared] flag) so
          the Metal fragment intrinsics keep firing after placement makes it routine-local. *)
  sk_swizzle : LL.swizzle_kind option;
      (** Staged GPU mma sketches only ([sk_mma] with [sk_bk > 0]): store both cooperative operand
          tiles in this XOR layout (gh-ocannl-481 item 3, D3). Seeded as a {e twin} of each staged
          seed — same tile sizes, both operands marked — and only for format triples the backend
          advertises in {!Ir.Backend_intf.mma_capability.mma_staged_layouts}, so a twin is never
          proposed where the emission would decline it back to the scalar fallback (gh-ocannl-479).
          The tuner, not a heuristic, decides whether the bank-conflict fix beats the plain tile:
          the same "propose both, measure" pattern as hoisted packing. Unstaged seeds have no shared
          tile to swizzle and are never twinned. *)
  sk_depth : int;
      (** Staged GPU mma/conv sketches: the cooperative stages' software-pipelining depth
          ([Schedule.Stage ~pipeline_depth], gh-ocannl-487); 1 = unpipelined. Depths > 1 are seeded
          as {e twins} of each staged seed — same tile sizes, same pipeline, so a timing
          difference between the two is the prefetch overlap's (against the halved occupancy from
          the doubled shared-memory footprint), and nothing else's — for exactly the depths the
          backend advertises in {!Ir.Backend_intf.mma_capability.mma_pipeline_depths}, and only
          for staged operands of at least 4-byte storage — the async arms' element floor
          ([C_syntax_config.async_copy]); a narrower twin could only render the portable
          synchronous form, whose occupancy cost phase 1 measured. The rendering is bitwise
          identical to the plain sibling, so the tuner's choice is free of numerics concerns.
          Unstaged seeds have no cooperative copy to pipeline and are never twinned. *)
}

(* Resolve the tensor-core input format from storage precision before seeding a typed matmul/conv
   site. Single-precision storage has two possible compute formats: prefer tf32 when the numerics
   policy enables it and the backend advertises that pair, then fall back to genuine f32 (Metal).
   Backends remain the emission source of truth; this only prevents the autotuner from rejecting a
   supported divergent tile up front, or timing a format the capability does not advertise. *)
let mma_input_formats_of_prec (prec : Ir.Ops.prec) : Ir.Backend_intf.mma_input_format list =
  match prec with
  | Ir.Ops.Half_prec _ -> [ Ir.Backend_intf.Mma_f16 ]
  | Ir.Ops.Bfloat16_prec _ -> [ Ir.Backend_intf.Mma_bf16 ]
  | Ir.Ops.Fp8_prec _ -> [ Ir.Backend_intf.Mma_fp8_e5m2 ]
  | Ir.Ops.Single_prec _ ->
      if (Ir.Numerics.get ()).Ir.Numerics.tf32_matmuls then
        [ Ir.Backend_intf.Mma_tf32; Ir.Backend_intf.Mma_f32 ]
      else [ Ir.Backend_intf.Mma_f32 ]
  | _ -> []

(* The accumulator format of a destination's storage precision (gh-ocannl-545). Unlike the
   multiplicands, this admits no policy choice: the accumulator is read back from and written to the
   node, so its format is its storage layout. In particular f32 storage accumulates as [Mma_f32]
   even under the tf32 policy — tf32 truncates the multiplicands, never the accumulator. *)
let mma_acc_format_of_prec (prec : Ir.Ops.prec) : Ir.Backend_intf.mma_input_format option =
  match prec with
  | Ir.Ops.Half_prec _ -> Some Ir.Backend_intf.Mma_f16
  | Ir.Ops.Bfloat16_prec _ -> Some Ir.Backend_intf.Mma_bf16
  | Ir.Ops.Single_prec _ -> Some Ir.Backend_intf.Mma_f32
  | _ -> None

let equal_mma_format_triple (a1, b1, d1) (a2, b2, d2) =
  Ir.Backend_intf.equal_mma_input_format a1 a2
  && Ir.Backend_intf.equal_mma_input_format b1 b2
  && Ir.Backend_intf.equal_mma_input_format d1 d2

(* The site's resolved format triples, in [mma_input_formats_of_prec]'s preference order. *)
let mma_format_triples ~a_prec ~b_prec ~d_prec =
  match mma_acc_format_of_prec d_prec with
  | None -> []
  | Some d_format ->
      List.concat_map (mma_input_formats_of_prec a_prec) ~f:(fun a_format ->
          List.map (mma_input_formats_of_prec b_prec) ~f:(fun b_format ->
              (a_format, b_format, d_format)))

let mma_tile_for_precisions (mma : Ir.Backend_intf.mma_capability) ~a_prec ~b_prec ~d_prec =
  List.find_map (mma_format_triples ~a_prec ~b_prec ~d_prec) ~f:(fun key ->
      List.Assoc.find mma.Ir.Backend_intf.mma_format_tiles key ~equal:equal_mma_format_triple)

(* The swizzled staged layout, if any, that the backend can read for this site's formats
   (gh-ocannl-481 item 3, D3). [None] leaves the staged seeds untwinned. *)
let mma_staged_layout_for_precisions (mma : Ir.Backend_intf.mma_capability) ~a_prec ~b_prec ~d_prec :
    LL.swizzle_kind option =
  List.find_map (mma_format_triples ~a_prec ~b_prec ~d_prec) ~f:(fun key ->
      List.Assoc.find mma.Ir.Backend_intf.mma_staged_layouts key ~equal:equal_mma_format_triple)
  |> Option.map ~f:(function Ir.Backend_intf.Mma_swizzled_b128 -> LL.Swizzle_b128)

type matmul_site = {
  m_i : Idx.symbol;
  m_j : Idx.symbol;
  m_k : Idx.symbol;
  m_ni : int;
  m_nj : int;
  m_nk : int;
  m_bo : (Idx.symbol * int) list;
      (** Batch loops enclosing the [m_i] loop, in nest order (gh-ocannl-528): loops beyond the
          [i x j x k] triple that carry their own output axis. They stay [Serial] in the sketch
          pipelines (grid slots are budgeted for the row/column blocks) and join the cross-nest
          alignment chain ([matmul_site_chain]). Empty on plain rank-2 sites. *)
  m_bi : (Idx.symbol * int) list;
      (** Batch loops nested {e between} [m_i] and [m_j] (nest order) — attention's interior head
          axis. The sketch pipelines hoist them above [m_i] with [Swap]s ([batch_hoist_swaps]) so
          the micro-kernel is perfectly nested for [Tensorize]. Empty on plain rank-2 sites. *)
  m_row_axis : int;
      (** The axis of [m_d]'s index map carrying [m_i] (the 2-D tile row). [rank - 2] on plain
          sites; smaller when interior batch axes sit between the roles. [m_j] is always on the
          minor axis. *)
  m_d : Ir.Tnode.t;
  m_a : Ir.Tnode.t;
  m_b : Ir.Tnode.t;
  m_zeroed : bool;  (** A whole-node [Zero_out] of [m_d] is present (needed by [expand_zero]). *)
  m_tb : bool option;
      (** [m_b]'s stored layout: [Some false] = [m_j] on its minor axis ([..., k, ..., j]),
          [Some true] = transposed ([m_k] on the minor axis), [None] = neither cleanly (the
          candidate then fails [Tensorize]'s own role check at compile). Feeds the seeding
          pre-filter (gh-ocannl-479): a rendering that reads B {e in place} inherits this
          orientation — which the register tiling declines when transposed — while a packing
          [Stage] normalizes it. A transposed A never declines (its feeds are scalar splats either
          way), so it is not tracked. *)
  m_fma : bool;
      (** The accumulation is in fused ([Ops.FMA]) form, as [optimize]'s simplify leaves it — the
          form the register-tiled [Tile_mma] rendering requires (its vector twin promises bitwise
          equality only for fused rounding). Candidate schedules rewrite operand reads but never the
          accumulation form, so this is decidable at seeding time. *)
}

let idcs_mention idcs s =
  Array.exists idcs ~f:(function
    | Idx.Iterator s2 -> Idx.equal_symbol s s2
    | Idx.Affine { symbols; _ } -> List.exists symbols ~f:(fun (_, s2) -> Idx.equal_symbol s s2)
    | _ -> false)

let strip_stmts stmts =
  List.filter stmts ~f:(function LL.Noop | LL.Comment _ -> false | _ -> true)

(* The perfectly nested serial prefix of a statement: (symbol, extent) per loop, plus the leaf. *)
let rec serial_nest_of (llc : LL.t) : (Idx.symbol * int) list * LL.t =
  match llc with
  | LL.For_loop { index; body; from_ = 0; to_; axis = LL.Serial; _ } -> (
      match strip_stmts (LL.flat_lines [ body ]) with
      | [ single ] ->
          let rest, leaf = serial_nest_of single in
          ((index, to_ + 1) :: rest, leaf)
      | _ -> ([ (index, to_ + 1) ], body))
  | LL.If { body; _ } -> serial_nest_of body
  | _ -> ([], llc)

let rec collect_gets (sc : LL.scalar_t) : (Ir.Tnode.t * Idx.axis_index array) list =
  let arg (s, _prec) = collect_gets s in
  match sc with
  | LL.Get (tn, idcs) -> [ (tn, idcs) ]
  | LL.Ternop (_, a, b, c) -> arg a @ arg b @ arg c
  | LL.Binop (_, a, b) -> arg a @ arg b
  | LL.Unop (_, a) -> arg a
  | LL.Get_dynamic { dyn_value; _ } -> arg dyn_value
  | LL.Local_scope _ | LL.Get_local _ | LL.Get_merge_buffer _ | LL.Constant _ | LL.Constant_bits _
  | LL.Embed_index _ ->
      []

let idx_mentions (idx : Idx.axis_index) s =
  match idx with
  | Idx.Iterator s2 -> Idx.equal_symbol s s2
  | Idx.Affine { symbols; _ } -> List.exists symbols ~f:(fun (_, s2) -> Idx.equal_symbol s s2)
  | _ -> false

let idx_coeff (idx : Idx.axis_index) sym =
  match idx with
  | Idx.Iterator s when Idx.equal_symbol s sym -> 1
  | Idx.Affine { symbols; _ } ->
      List.sum (module Int) symbols ~f:(fun (c, s) -> if Idx.equal_symbol s sym then c else 0)
  | _ -> 0

(* The unique axis of [idcs] owning [s]: [s] appears in exactly one component, with coefficient 1
   there. Mirrors [Schedule.Tensorize]'s ownership discipline. *)
let unit_axis (idcs : Idx.axis_index array) s : int option =
  let ps = Array.filter_mapi idcs ~f:(fun p idx -> Option.some_if (idx_mentions idx s) p) in
  match Array.to_list ps with [ p ] when idx_coeff idcs.(p) s = 1 -> Some p | _ -> None

(* Batched-site classification shared by the relation-based and procedural matchers
   (gh-ocannl-528). Inputs: the perfectly nested serial accumulation statement's loops in nest
   order (with extents), the accumulator's index map [di], and the two operand reads. Roles:

   - [k] is the innermost loop and the only one absent from [di].
   - Every other loop must own a distinct axis of [di] (unit coefficient, sole occurrence).
   - [j] owns [di]'s minor axis and must be the innermost of the write loops (how lowering orders
     them — the sketch pipelines' hoisting normalization only handles batch loops above [j]).
   - Per operand order, [a] must own [k], must not read [j]; [b] must own [j] and [k]; [i] is the
     {e deepest} write loop owned by [a] and absent from [b] — the 2-D tile row. The exclusions
     are what keep variance-style self-products [d[b,s] += x[b,s,k] * x[b,s,k]] — whose reads
     mention every loop — from masquerading as matmuls: they seeded (and always failed candidate
     compile) before.
   - Everything else is batch: [m_bo] outside [i], [m_bi] between [i] and [j]; batch symbols may
     appear in the operands freely (their occurrences form the tile block base).

   Detection remains permissive about everything else — a mis-detected site fails its candidate
   compile (op preconditions, [validate_parallel], hardware limits) and is skipped. *)
let classify_matmul ~(loops : (Idx.symbol * int) list) ~(d : Ir.Tnode.t)
    ~(di : Idx.axis_index array) ~(o1 : Ir.Tnode.t * Idx.axis_index array)
    ~(o2 : Ir.Tnode.t * Idx.axis_index array) ~(zeroed : bool) ~(fma : bool) : matmul_site option =
  let rank = Array.length di in
  match List.rev loops with
  | (k, nk) :: ((_ :: _ :: _ as rev_ws) : (Idx.symbol * int) list)
    when rank >= 2 && not (idcs_mention di k) ->
      let ws = List.rev rev_ws in
      let d_axes = List.map ws ~f:(fun (s, _) -> unit_axis di s) in
      if List.exists d_axes ~f:Option.is_none then None
      else
        let axes = List.zip_exn ws (List.filter_opt d_axes) in
        let distinct =
          let ps = List.map axes ~f:snd in
          List.length (List.dedup_and_sort ps ~compare:Int.compare) = List.length ps
        in
        if not distinct then None
        else
          let (j, nj), pj = List.last_exn axes in
          if pj <> rank - 1 then None
          else
            let front = List.drop_last_exn axes in
            let try_order ((a, ai) : Ir.Tnode.t * Idx.axis_index array)
                ((b, bi) : Ir.Tnode.t * Idx.axis_index array) : matmul_site option =
              if
                idcs_mention ai j
                || Option.is_none (unit_axis ai k)
                || Option.is_none (unit_axis bi j)
                || Option.is_none (unit_axis bi k)
              then None
              else
                let eligible =
                  List.filter front ~f:(fun ((s, _), _) ->
                      Option.is_some (unit_axis ai s) && not (idcs_mention bi s))
                in
                Option.map (List.last eligible) ~f:(fun ((i, ni), p_row) ->
                    let before_i = ref true in
                    let m_bo = ref [] and m_bi = ref [] in
                    List.iter front ~f:(fun ((s, n), _) ->
                        if Idx.equal_symbol s i then before_i := false
                        else if !before_i then m_bo := (s, n) :: !m_bo
                        else m_bi := (s, n) :: !m_bi);
                    let rank_b = Array.length bi in
                    let m_tb =
                      match (unit_axis bi j, unit_axis bi k) with
                      | Some p, _ when p = rank_b - 1 -> Some false
                      | _, Some p when p = rank_b - 1 -> Some true
                      | _ -> None
                    in
                    {
                      m_i = i;
                      m_j = j;
                      m_k = k;
                      m_ni = ni;
                      m_nj = nj;
                      m_nk = nk;
                      m_bo = List.rev !m_bo;
                      m_bi = List.rev !m_bi;
                      m_row_axis = p_row;
                      m_d = d;
                      m_a = a;
                      m_b = b;
                      m_zeroed = zeroed;
                      m_tb;
                      m_fma = fma;
                    })
            in
            (match try_order o1 o2 with Some _ as r -> r | None -> try_order o2 o1)
  | _ -> None

let detect_matmul_procedural (llc : LL.t) : matmul_site option =
  let stmts = strip_stmts (LL.flat_lines [ llc ]) in
  let zeroed = List.filter_map stmts ~f:(function LL.Zero_out tn -> Some tn | _ -> None) in
  List.find_map stmts ~f:(fun stmt ->
      match serial_nest_of stmt with
      | (_ :: _ :: _ :: _ as loops), LL.Set { tn = d; idcs = di; llsc; _ } -> (
          let gets = collect_gets llsc in
          let is_d_read (tn, idcs) = phys_equal tn d && Array.equal Idx.equal_axis_index idcs di in
          let d_reads, others = List.partition_tf gets ~f:is_d_read in
          match (d_reads, others) with
          | _ :: _, [ o1; o2 ] ->
              let fma =
                match llsc with
                | LL.Ternop (Ir.Ops.FMA, _, _, (LL.Get (tn, idcs), _)) -> is_d_read (tn, idcs)
                | _ -> false
              in
              classify_matmul ~loops ~d ~di ~o1 ~o2
                ~zeroed:(List.exists zeroed ~f:(phys_equal d))
                ~fma
          | _ -> None)
      | _ -> None)

(** {2 Relation-based micro-kernel recognition (gh-494 waypoint-2 remainder)}

    Detection reads off the same extracted artifact the op-legality oracle consumes —
    [LL.affine_accesses]: the rmw markers, index maps, loop boxes and program paths — instead of
    re-walking the code with a procedural structural matcher, so detection and legality share one
    source of access truth. The procedural matchers above are kept for the [legality_crosscheck]
    soak, which raises on any divergence (detection feeds sketch seeding, so changes must be
    behavior-preserving). Known corners where the relations see more than the old walkers (an [If]
    guard whose condition reads a tensor node; an interior statement with no tensor accesses):
    optimized code does not produce them, and the crosscheck guards the claim. *)

module A = Ir.Affine

(* Axis types by loop binder — the one nest discipline the access records do not carry (their
   [a_loops] carry the bounds). Statement-level loops only: an access whose enclosing loops are not
   all found here is inside a [Local_scope] body or [Tile_mma] fallback, which the recognizers
   reject anyway. *)
let rec loop_axis_types acc (llc : LL.t) =
  match llc with
  | LL.Seq (a, b) -> loop_axis_types (loop_axis_types acc a) b
  | LL.For_loop { index; axis; body; _ } -> loop_axis_types ((index, axis) :: acc) body
  | LL.If { body; _ } -> loop_axis_types acc body
  | _ -> acc

let path_head = A.stmt_head

(* Access records per top-level statement, in program order (the extraction fires in program order
   and top-level statement indices are nondecreasing). *)
let accesses_by_statement (accs : Ir.Tnode.t A.access list) =
  List.group accs ~break:(fun a b -> path_head a.A.a_path <> path_head b.A.a_path)

(* The accumulation form (fused [Ops.FMA] vs add-of-product) is scalar structure the access records
   do not carry; probe the recognized statement's leaf assignment directly. *)
let fma_form (llc : LL.t) ~stmt_path ~d ~(di : Idx.axis_index array) : bool =
  let stmt =
    match path_head stmt_path with -1 -> llc | h -> List.nth_exn (LL.flat_lines [ llc ]) h
  in
  let is_d_read tn idcs = phys_equal tn d && Array.equal Idx.equal_axis_index idcs di in
  let rec find = function
    | LL.Seq (a, b) -> ( match find a with Some _ as r -> r | None -> find b)
    | LL.For_loop { body; _ } | LL.If { body; _ } -> find body
    | LL.Set { tn; idcs; llsc; _ } when is_d_read tn idcs -> Some llsc
    | _ -> None
  in
  match find stmt with
  | Some (LL.Ternop (Ir.Ops.FMA, _, _, (LL.Get (tn, idcs), _))) -> is_d_read tn idcs
  | _ -> false

(* The perfect all-serial from-0 accumulation statement, as the relations express it: a single
   interpretable write whose enclosing statement's accesses are all the statement's own direct
   reads ([Affine.same_statement]: paths agreeing above the final [Rhs]/[Write] component — a
   sibling statement inside the nest, or a read nested in a [Local_scope] body, breaks the
   agreement) and share its loop box. Returns the write and the non-write accesses split into the
   write's own same-cell reads (the rmw carrier) and the operand reads, in program order. *)
let serial_kernel_of axes (g : Ir.Tnode.t A.access list) =
  let writes = List.filter g ~f:(fun a -> a.A.a_write) in
  match writes with
  | [ w ] when (not w.A.a_whole) && (not w.A.a_dynamic) && not w.A.a_vec_last ->
      let serial0 (s, (lo, _)) =
        lo = 0
        &&
        match List.Assoc.find axes s ~equal:Idx.equal_symbol with
        | Some LL.Serial -> true
        | Some _ | None -> false
      in
      let loops_equal =
        List.equal (fun (s1, (l1, h1)) (s2, (l2, h2)) ->
            Idx.equal_symbol s1 s2 && l1 = l2 && h1 = h2)
      in
      if
        List.for_all w.A.a_loops ~f:serial0
        && List.for_all g ~f:(fun a ->
            A.same_statement a.A.a_path w.A.a_path && loops_equal a.A.a_loops w.A.a_loops)
      then
        let same_d a =
          phys_equal a.A.a_tn w.A.a_tn && Array.equal Idx.equal_axis_index a.A.a_map w.A.a_map
        in
        let reads = List.filter g ~f:(fun a -> not a.A.a_write) in
        let d_reads, others = List.partition_tf reads ~f:same_d in
        Some (w, d_reads, others)
      else None
  | _ -> None

let detect_matmul_affine (llc : LL.t) : matmul_site option =
  let accs = LL.affine_accesses llc in
  let axes = loop_axis_types [] llc in
  let zeroed = List.filter accs ~f:(fun a -> a.A.a_write && a.A.a_whole) in
  List.find_map (accesses_by_statement accs) ~f:(fun g ->
      match serial_kernel_of axes g with
      | Some (w, (_ :: _ as _d_reads), [ o1; o2 ]) ->
          let loops = List.map w.A.a_loops ~f:(fun (s, (_, hi)) -> (s, hi + 1)) in
          classify_matmul ~loops ~d:w.A.a_tn ~di:w.A.a_map ~o1:(o1.A.a_tn, o1.A.a_map)
            ~o2:(o2.A.a_tn, o2.A.a_map)
            ~zeroed:(List.exists zeroed ~f:(fun z -> phys_equal z.A.a_tn w.A.a_tn))
            ~fma:(fma_form llc ~stmt_path:w.A.a_path ~d:w.A.a_tn ~di:w.A.a_map)
      | _ -> None)

let matmul_site_equal (x : matmul_site) (y : matmul_site) =
  let batch_equal = List.equal (fun (s1, n1) (s2, n2) -> Idx.equal_symbol s1 s2 && n1 = n2) in
  Idx.equal_symbol x.m_i y.m_i && Idx.equal_symbol x.m_j y.m_j && Idx.equal_symbol x.m_k y.m_k
  && x.m_ni = y.m_ni && x.m_nj = y.m_nj && x.m_nk = y.m_nk && batch_equal x.m_bo y.m_bo
  && batch_equal x.m_bi y.m_bi && x.m_row_axis = y.m_row_axis && phys_equal x.m_d y.m_d
  && phys_equal x.m_a y.m_a && phys_equal x.m_b y.m_b && Bool.equal x.m_zeroed y.m_zeroed
  && Option.equal Bool.equal x.m_tb y.m_tb
  && Bool.equal x.m_fma y.m_fma

let detect_matmul (llc : LL.t) : matmul_site option =
  let site = detect_matmul_affine llc in
  (if Lazy.force A.crosscheck_enabled then
     let procedural = detect_matmul_procedural llc in
     match (procedural, site) with
     | None, None -> ()
     | Some p, Some n when matmul_site_equal p n -> ()
     | _ ->
         invalid_arg
           "Autotune.detect_matmul crosscheck: the relation-based and procedural matchers diverge \
            — detection must be behavior-preserving");
  site

let sink sym below = List.map below ~f:(fun inner -> Sched.Swap { outer = sym; inner })

(* gh-ocannl-485 (PADTO): pad [axis] to the next multiple of [f] when [f] does not divide its
   [extent]. Identity pads are omitted, so divisible sites keep byte-identical schedules (and
   schedule-cache keys). Only sound in pipelines that stage every operand the padded axis reaches
   ([Tensorize] enforces the zero-fringe requirement at apply). *)
let pad_to ~axis ~extent f =
  if f > 0 && extent % f <> 0 then [ Sched.Pad { axis; to_multiple_of = f } ] else []

(* Blocks of size [b] covering a possibly padded extent [n]. *)
let blocks_of n b = (n + b - 1) / b

(** {2 Convolution detection and the implicit-GEMM sketch (gh-ocannl-493)}

    A convolution is a matmul over a virtual im2col operand. Conv einsums lower to affine-indexed
    accumulation nests —
    [d[b.., oh.., oc] += a[b.., s*oh + t*kh + off.., ic] * w[perm(oc, kh.., ic)]] — so the
    implicit-GEMM mapping is a re-association of loops that already exist: reorder to
    [outer..; kernel..; row; oc; ic], pack the strided-window [row × ic] slice of [a] (the packing
    [Stage] {e is} im2col — same copy nest, conv index arithmetic) and the [ic × oc] slice of [w]
    (normalizing any stored layout) at the kernel-window anchor, then [Tensorize (row, oc, ic)]
    exactly as for matmuls: the register tiling / tensor cores and the accumulator contraction
    (gh-ocannl-480; resident across the whole kernel-window chain since gh-ocannl-501) apply
    unchanged.

    Unlike the matmul pipelines, the reorder moves the [ic] reduction inside the kernel loops, so
    the per-element reduction order changes: conv sketch candidates match the unscheduled form
    within float-reassociation tolerance (like the GPU fragment paths), while the tensorized
    pipeline stays bitwise against the reorder-only form on the C backends. *)

type conv_axis = {
  cx_o : Idx.symbol;  (** Output spatial symbol (appears in [d] as a plain iterator). *)
  cx_no : int;
  cx_k : Idx.symbol;  (** Kernel-window symbol (appears in [w], not in [d]). *)
  cx_nk : int;
  cx_stride : int;
  cx_dilation : int;
  cx_offset : int;
      (** Padding offset on the input access. Healthy graphs lower padded convs offset-free: the
          source is physically padded and buffer indices absorb the halo shift, while halo-lost
          operands (layout committed before the padded consumer) are rejected at shape-inference
          time. A nonzero offset can still reach detection from hand-built [Low_level] code. *)
}

type conv_site = {
  c_loops : Idx.symbol list;  (** The accumulation nest's loops, outermost first. *)
  c_outer : (Idx.symbol * int) list;
      (** Loops kept outer, in nest order: batch axes and the non-row output spatial axes. *)
  c_kernel : Idx.symbol list;  (** Kernel-window symbols in nest order (the [k_o] tier). *)
  c_axes : conv_axis list;
  c_row : Idx.symbol;  (** The GEMM row: the conv axis at [d]'s rank-2 position. *)
  c_nrow : int;
  c_oc : Idx.symbol;  (** The GEMM column: [d]'s rank-1 symbol, read by [w] only. *)
  c_noc : int;
  c_red : Idx.symbol;  (** The GEMM reduction: the channel symbol read by both operands. *)
  c_nred : int;
  c_d : Ir.Tnode.t;
  c_a : Ir.Tnode.t;
  c_b : Ir.Tnode.t;
  c_zeroed : bool;
  c_fma : bool;
}

let detect_conv_procedural (llc : LL.t) : conv_site option =
  let stmts = strip_stmts (LL.flat_lines [ llc ]) in
  let zeroed = List.filter_map stmts ~f:(function LL.Zero_out tn -> Some tn | _ -> None) in
  List.find_map stmts ~f:(fun stmt ->
      let loops, leaf = serial_nest_of stmt in
      match leaf with
      | LL.Set { tn = d; idcs = di; llsc; _ } when List.length loops >= 4 -> (
          let extent s = List.Assoc.find loops s ~equal:Idx.equal_symbol in
          let gets = collect_gets llsc in
          let is_d_read (tn, idcs) = phys_equal tn d && Array.equal Idx.equal_axis_index idcs di in
          let d_reads, others = List.partition_tf gets ~f:is_d_read in
          match (d_reads, others) with
          | _ :: _, [ o1; o2 ] -> (
              (* [d] written at plain, distinct iterators — its symbols are the GEMM output
                 space. *)
              let d_syms =
                Array.to_list di
                |> List.map ~f:(function Idx.Iterator s -> Some s | _ -> None)
                |> Option.all
              in
              match d_syms with
              | Some d_syms
                when List.length d_syms = Array.length di
                     && (not (List.contains_dup d_syms ~compare:Idx.compare_symbol))
                     && List.length d_syms >= 2 -> (
                  let is_out s = List.mem d_syms s ~equal:Idx.equal_symbol in
                  (* The input operand carries the conv fingerprint: an affine component mixing an
                     output symbol with a kernel symbol. *)
                  let conv_component (idx : Idx.axis_index) =
                    match idx with
                    | Idx.Affine { symbols = [ (c1, s1); (c2, s2) ]; offset } -> (
                        match (is_out s1, is_out s2) with
                        | true, false ->
                            Some
                              {
                                cx_o = s1;
                                cx_no = 0;
                                cx_k = s2;
                                cx_nk = 0;
                                cx_stride = c1;
                                cx_dilation = c2;
                                cx_offset = offset;
                              }
                        | false, true ->
                            Some
                              {
                                cx_o = s2;
                                cx_no = 0;
                                cx_k = s1;
                                cx_stride = c2;
                                cx_dilation = c1;
                                cx_offset = offset;
                                cx_nk = 0;
                              }
                        | _ -> None)
                    | _ -> None
                  in
                  let classify (tn, idcs) =
                    let axes = Array.to_list idcs |> List.filter_map ~f:conv_component in
                    (tn, idcs, axes)
                  in
                  let (a, a_idcs, a_axes), (b, b_idcs, b_axes) =
                    let c1 = classify o1 and c2 = classify o2 in
                    match (c1, c2) with
                    | (_, _, _ :: _), (_, _, []) -> (c1, c2)
                    | (_, _, []), (_, _, _ :: _) -> (c2, c1)
                    | _ -> (c1, c1)
                    (* Both-or-neither convolutional: rejected below (b_axes <> []). *)
                  in
                  let b_plain =
                    Array.to_list b_idcs
                    |> List.map ~f:(function Idx.Iterator s -> Some s | _ -> None)
                    |> Option.all
                  in
                  match b_plain with
                  | Some b_syms when List.is_empty b_axes && not (phys_equal a b) -> (
                      let kernel_syms = List.map a_axes ~f:(fun cx -> cx.cx_k) in
                      let in_b s = List.mem b_syms s ~equal:Idx.equal_symbol in
                      let oc_candidates = List.filter d_syms ~f:in_b in
                      (* Reduction symbols: read by both operands, not output, not kernel. *)
                      let a_plain_syms =
                        Array.to_list a_idcs
                        |> List.filter_map ~f:(function Idx.Iterator s -> Some s | _ -> None)
                      in
                      let red_candidates =
                        List.filter a_plain_syms ~f:(fun s -> in_b s && not (is_out s))
                      in
                      let rank = Array.length di in
                      match (oc_candidates, red_candidates, extent (List.last_exn d_syms)) with
                      | [ oc ], [ red ], Some noc
                        when Idx.equal_symbol oc (List.last_exn d_syms)
                             && (not (List.exists a_plain_syms ~f:(Idx.equal_symbol oc)))
                             && List.for_all kernel_syms ~f:(fun k -> in_b k && not (is_out k))
                             && List.for_all b_syms ~f:(fun s ->
                                 Idx.equal_symbol s oc || Idx.equal_symbol s red
                                 || List.mem kernel_syms s ~equal:Idx.equal_symbol) -> (
                          (* The GEMM row: the conv axis sitting at [d]'s rank-2 position. *)
                          let row_sym =
                            match di.(rank - 2) with Idx.Iterator s -> s | _ -> assert false
                          in
                          match
                            ( List.find a_axes ~f:(fun cx -> Idx.equal_symbol cx.cx_o row_sym),
                              extent row_sym,
                              extent red )
                          with
                          | Some _, Some nrow, Some nred ->
                              let with_extents cx =
                                match (extent cx.cx_o, extent cx.cx_k) with
                                | Some no, Some nk -> Some { cx with cx_no = no; cx_nk = nk }
                                | _ -> None
                              in
                              let axes = Option.all (List.map a_axes ~f:with_extents) in
                              let loop_syms = List.map loops ~f:fst in
                              let m_fma =
                                match llsc with
                                | LL.Ternop (Ir.Ops.FMA, _, _, (LL.Get (tn, idcs), _)) ->
                                    is_d_read (tn, idcs)
                                | _ -> false
                              in
                              let is_kernel s = List.mem kernel_syms s ~equal:Idx.equal_symbol in
                              let outer =
                                List.filter loops ~f:(fun (s, _) ->
                                    is_out s
                                    && (not (Idx.equal_symbol s row_sym))
                                    && not (Idx.equal_symbol s oc))
                              in
                              let kernel_order = List.filter loop_syms ~f:is_kernel in
                              Option.map axes ~f:(fun axes ->
                                  {
                                    c_loops = loop_syms;
                                    c_outer = outer;
                                    c_kernel = kernel_order;
                                    c_axes = axes;
                                    c_row = row_sym;
                                    c_nrow = nrow;
                                    c_oc = oc;
                                    c_noc = noc;
                                    c_red = red;
                                    c_nred = nred;
                                    c_d = d;
                                    c_a = a;
                                    c_b = b;
                                    c_zeroed = List.exists zeroed ~f:(phys_equal d);
                                    c_fma = m_fma;
                                  })
                          | _ -> None)
                      | _ -> None)
                  | _ -> None)
              | _ -> None)
          | _ -> None)
      | _ -> None)

let detect_conv_affine (llc : LL.t) : conv_site option =
  let accs = LL.affine_accesses llc in
  let axes = loop_axis_types [] llc in
  let zeroed = List.filter accs ~f:(fun a -> a.A.a_write && a.A.a_whole) in
  List.find_map (accesses_by_statement accs) ~f:(fun g ->
      match serial_kernel_of axes g with
      | Some (w, _ :: _, [ ro1; ro2 ]) when List.length w.A.a_loops >= 4 -> (
          let loops = List.map w.A.a_loops ~f:(fun (s, (_, hi)) -> (s, hi + 1)) in
          let extent s = List.Assoc.find loops s ~equal:Idx.equal_symbol in
          let d = w.A.a_tn and di = w.A.a_map in
          let o1 = (ro1.A.a_tn, ro1.A.a_map) and o2 = (ro2.A.a_tn, ro2.A.a_map) in
          (* From here on, the classification is the same role logic as the procedural matcher, fed
             from the extracted maps. *)
          let d_syms =
            Array.to_list di
            |> List.map ~f:(function Idx.Iterator s -> Some s | _ -> None)
            |> Option.all
          in
          match d_syms with
          | Some d_syms
            when List.length d_syms = Array.length di
                 && (not (List.contains_dup d_syms ~compare:Idx.compare_symbol))
                 && List.length d_syms >= 2 -> (
              let is_out s = List.mem d_syms s ~equal:Idx.equal_symbol in
              let conv_component (idx : Idx.axis_index) =
                match idx with
                | Idx.Affine { symbols = [ (c1, s1); (c2, s2) ]; offset } -> (
                    match (is_out s1, is_out s2) with
                    | true, false ->
                        Some
                          {
                            cx_o = s1;
                            cx_no = 0;
                            cx_k = s2;
                            cx_nk = 0;
                            cx_stride = c1;
                            cx_dilation = c2;
                            cx_offset = offset;
                          }
                    | false, true ->
                        Some
                          {
                            cx_o = s2;
                            cx_no = 0;
                            cx_k = s1;
                            cx_stride = c2;
                            cx_dilation = c1;
                            cx_offset = offset;
                            cx_nk = 0;
                          }
                    | _ -> None)
                | _ -> None
              in
              let classify (tn, idcs) =
                let axes = Array.to_list idcs |> List.filter_map ~f:conv_component in
                (tn, idcs, axes)
              in
              let (a, a_idcs, a_axes), (b, b_idcs, b_axes) =
                let c1 = classify o1 and c2 = classify o2 in
                match (c1, c2) with
                | (_, _, _ :: _), (_, _, []) -> (c1, c2)
                | (_, _, []), (_, _, _ :: _) -> (c2, c1)
                | _ -> (c1, c1)
                (* Both-or-neither convolutional: rejected below (b_axes <> []). *)
              in
              let b_plain =
                Array.to_list b_idcs
                |> List.map ~f:(function Idx.Iterator s -> Some s | _ -> None)
                |> Option.all
              in
              match b_plain with
              | Some b_syms when List.is_empty b_axes && not (phys_equal a b) -> (
                  let kernel_syms = List.map a_axes ~f:(fun cx -> cx.cx_k) in
                  let in_b s = List.mem b_syms s ~equal:Idx.equal_symbol in
                  let oc_candidates = List.filter d_syms ~f:in_b in
                  let a_plain_syms =
                    Array.to_list a_idcs
                    |> List.filter_map ~f:(function Idx.Iterator s -> Some s | _ -> None)
                  in
                  let red_candidates =
                    List.filter a_plain_syms ~f:(fun s -> in_b s && not (is_out s))
                  in
                  let rank = Array.length di in
                  match (oc_candidates, red_candidates, extent (List.last_exn d_syms)) with
                  | [ oc ], [ red ], Some noc
                    when Idx.equal_symbol oc (List.last_exn d_syms)
                         && (not (List.exists a_plain_syms ~f:(Idx.equal_symbol oc)))
                         && List.for_all kernel_syms ~f:(fun k -> in_b k && not (is_out k))
                         && List.for_all b_syms ~f:(fun s ->
                             Idx.equal_symbol s oc || Idx.equal_symbol s red
                             || List.mem kernel_syms s ~equal:Idx.equal_symbol) -> (
                      let row_sym =
                        match di.(rank - 2) with Idx.Iterator s -> s | _ -> assert false
                      in
                      match
                        ( List.find a_axes ~f:(fun cx -> Idx.equal_symbol cx.cx_o row_sym),
                          extent row_sym,
                          extent red )
                      with
                      | Some _, Some nrow, Some nred ->
                          let with_extents cx =
                            match (extent cx.cx_o, extent cx.cx_k) with
                            | Some no, Some nk -> Some { cx with cx_no = no; cx_nk = nk }
                            | _ -> None
                          in
                          let caxes = Option.all (List.map a_axes ~f:with_extents) in
                          let loop_syms = List.map loops ~f:fst in
                          let is_kernel s = List.mem kernel_syms s ~equal:Idx.equal_symbol in
                          let outer =
                            List.filter loops ~f:(fun (s, _) ->
                                is_out s
                                && (not (Idx.equal_symbol s row_sym))
                                && not (Idx.equal_symbol s oc))
                          in
                          let kernel_order = List.filter loop_syms ~f:is_kernel in
                          Option.map caxes ~f:(fun caxes ->
                              {
                                c_loops = loop_syms;
                                c_outer = outer;
                                c_kernel = kernel_order;
                                c_axes = caxes;
                                c_row = row_sym;
                                c_nrow = nrow;
                                c_oc = oc;
                                c_noc = noc;
                                c_red = red;
                                c_nred = nred;
                                c_d = d;
                                c_a = a;
                                c_b = b;
                                c_zeroed = List.exists zeroed ~f:(fun z -> phys_equal z.A.a_tn d);
                                c_fma = fma_form llc ~stmt_path:w.A.a_path ~d ~di;
                              })
                      | _ -> None)
                  | _ -> None)
              | _ -> None)
          | _ -> None)
      | _ -> None)

let conv_axis_equal (x : conv_axis) (y : conv_axis) =
  Idx.equal_symbol x.cx_o y.cx_o && x.cx_no = y.cx_no && Idx.equal_symbol x.cx_k y.cx_k
  && x.cx_nk = y.cx_nk && x.cx_stride = y.cx_stride && x.cx_dilation = y.cx_dilation
  && x.cx_offset = y.cx_offset

let conv_site_equal (x : conv_site) (y : conv_site) =
  List.equal Idx.equal_symbol x.c_loops y.c_loops
  && List.equal (fun (s1, n1) (s2, n2) -> Idx.equal_symbol s1 s2 && n1 = n2) x.c_outer y.c_outer
  && List.equal Idx.equal_symbol x.c_kernel y.c_kernel
  && List.equal conv_axis_equal x.c_axes y.c_axes
  && Idx.equal_symbol x.c_row y.c_row && x.c_nrow = y.c_nrow && Idx.equal_symbol x.c_oc y.c_oc
  && x.c_noc = y.c_noc && Idx.equal_symbol x.c_red y.c_red && x.c_nred = y.c_nred
  && phys_equal x.c_d y.c_d && phys_equal x.c_a y.c_a && phys_equal x.c_b y.c_b
  && Bool.equal x.c_zeroed y.c_zeroed && Bool.equal x.c_fma y.c_fma

let detect_conv (llc : LL.t) : conv_site option =
  let site = detect_conv_affine llc in
  (if Lazy.force A.crosscheck_enabled then
     let procedural = detect_conv_procedural llc in
     match (procedural, site) with
     | None, None -> ()
     | Some p, Some n when conv_site_equal p n -> ()
     | _ ->
         invalid_arg
           "Autotune.detect_conv crosscheck: the relation-based and procedural matchers diverge — \
            detection must be behavior-preserving");
  site

(* Zero-geometry ops shared by the sketch pipelines: expand the whole-node [Zero_out] of the output
   and give the resulting nest a compatible parallel geometry, via [mk_zops] on its two fresh loop
   symbols. When the site is NOT zeroed — a fission segment's site never is, the [Zero_out] lands in
   its own [`Zeros] segment — there is nothing to expand and the pipelines are correct without it:
   [Privatize] init-loads the accumulator tile from the (pre-zeroed) target, and [Tile_mma] loads
   the accumulator fragment before the reduction. *)
let zero_geometry (site : matmul_site) ~(mk_zops : zi:Idx.symbol -> zj:Idx.symbol -> Sched.schedule)
    : Sched.schedule =
  if not site.m_zeroed then []
  else (
    let rank = Array.length (Lazy.force site.m_d.Ir.Tnode.dims) in
    if rank < 2 || site.m_row_axis >= rank - 1 then
      (* This is a known limitation of the generated sketch, not an arbitrary exception from a
         user transform. Preserve that distinction at the narrow site so strict candidate failure
         classification records a decline and continues trying the remaining seeds. *)
      raise
        (Outcome.Cause_at
           ( Outcome.Transform,
             Outcome.Unsupported
               {
                 feature = "autotune_sketch_output_rank";
                 detail = "Autotune sketch: zero expansion needs a row axis before the minor axis";
               } ));
    let ez, zsyms = Sched.expand_zero ~tn:site.m_d in
    (* Batched outputs (gh-ocannl-528): the row/column zero loops get the accumulation's geometry;
       the batch-axis zero loops stay [Serial], like the batch loops of the accumulation nest. The
       row loop precedes the column loop in the zero nest ([m_row_axis < rank - 1]), matching the
       accumulation's positional hardware-slot order. *)
    let zi = List.nth_exn zsyms site.m_row_axis and zj = List.last_exn zsyms in
    ez :: mk_zops ~zi ~zj)

(* The would-be epilogue tail's loop symbols: the first real statement after the last statement
   writing [target] — the nest [Sched.Fuse_epilogue] consumes (its perfect-Serial-nest and
   sole-consumer vetting happens in the op itself). Used by the fused twins to leave that nest
   unannotated (fuse-before-annotate, gh-ocannl-501): [sketch_schedule] appends the fusion op last,
   by which point an annotated tail nest — [Fuse_epilogue] requires a perfect Serial tail — would be
   rejected. The relocated tail write lands under the accumulation nest's own geometry instead, so
   coverage is preserved without the dropped annotation. *)
let epilogue_tail_loop_syms ~(target : Ir.Tnode.t) (opt : LL.optimized) : Idx.symbol list =
  let stmts =
    List.filter (LL.flat_lines [ opt.LL.llc ]) ~f:(function
      | LL.Noop | LL.Comment _ -> false
      | _ -> true)
  in
  let rec writes_target = function
    | LL.Set { tn; _ } | LL.Zero_out tn | LL.Set_dynamic { tn; _ } | LL.Set_from_vec { tn; _ } ->
        Ir.Tnode.equal tn target
    | LL.Tile_mma { d = tn, _; _ } -> Ir.Tnode.equal tn target
    | LL.Seq (a, b) -> writes_target a || writes_target b
    | LL.For_loop { body; _ } | LL.If { body; _ } -> writes_target body
    | _ -> false
  in
  let rec loop_syms acc = function
    | LL.For_loop { index; body; _ } -> loop_syms (index :: acc) body
    | LL.Seq (a, b) -> loop_syms (loop_syms acc a) b
    | LL.If { body; _ } -> loop_syms acc body
    | _ -> acc
  in
  match List.filter_mapi stmts ~f:(fun i s -> Option.some_if (writes_target s) i) |> List.last with
  | None -> []
  | Some r -> ( match List.nth stmts (r + 1) with Some tail -> loop_syms [] tail | None -> [])

let rec nest_loop_syms acc (llc : LL.t) =
  match llc with
  | LL.For_loop { index; body; _ } -> nest_loop_syms (index :: acc) body
  | LL.Seq (a, b) -> nest_loop_syms (nest_loop_syms acc a) b
  | LL.If { body; _ } -> nest_loop_syms acc body
  | _ -> acc

(* Companion geometry for the GPU matmul sketches (gh-ocannl-521).

   A GPU sketch builds hardware geometry for the accumulation nest alone. Launch dimensions are
   global to the kernel, so every OTHER materialized-writing nest in the same routine — the bias/relu
   tail of a classifier head, the elementwise companions an aligned-merged fission segment carries —
   must be nested under loops covering the same active slots, or [Low_level.validate_parallel]
   rejects it and the whole candidate fails to compile. The GPU seeds used to leave those nests bare
   and depend on [Sched.Fuse_epilogue] absorbing the companion into the accumulation nest; when the
   fusion declines (a guarded reduction output, a whole-K [Tile_mma] accumulator), the seed had no
   surviving form at all — the cascade that left every GPU backend with tensorized candidates seeded
   in bulk and none ever timed.

   The precedent is [conv_aligned_grid], which reuses the default CPU preset's aligned cross-nest
   analysis rather than re-proving alignment. The same analysis extends to workgroup geometry: it is
   {!Sched.aligned_chains} that decides WHICH loops may be annotated and that chain position [k]
   means the same thread coordinate in every linked nest; only the geometry per position is the
   sketch's own, which is what a preset cannot supply for a tensorized nest (two [Grid] slots plus a
   [Workgroup] lane). Emitting a positionally identical geometry on each companion nest therefore
   preserves the alignment the analysis proved, and covers the same slots the site's nest binds.

   [site_syms] is the accumulation nest's chain, [annotate pos sym] the ops for chain position [pos]
   of a companion, [skip] the loop symbols of a nest to leave alone (the fused twins' epilogue tail,
   which the fusion relocates under the accumulation nest's geometry — annotating it would make
   [Fuse_epilogue] reject the candidate for the wrong reason), and [expanded_zeros] the nodes whose
   whole-node [Zero_out] the caller expands with the same geometry.

   [None] when the analysis bails, when the site's own chain was trimmed below [site_syms] (the nests
   could not be aligned at this arity — a companion annotated anyway would read cells another thread
   wrote, with no intra-kernel synchronization to order them), or when a companion's chain does not
   match the site's in arity and extents. A [None] must fail the candidate rather than fall back to a
   bare companion: on GPU there is no all-serial fallback.

   The query runs at the site's own arity ([max_chain = length site_syms], gh-ocannl-569): the
   analysis' default cap of 2 is the preset annotators' Grid+Workgroup shape, and under it a batched
   (rank-3+) site could never match its full chain — every seed for gpt2's FFN-class kernels
   declined here, serializing the minor output axis. A companion that genuinely cannot follow the
   full arity (a reduction over the site's minor axis, e.g. the lm_head's max-logits row) still
   trims the component's common prefix below [site_syms] and correctly declines.

   Residual, shared with the zeroing geometry this reuses: a tensorized nest's workgroup slot is the
   [Tensorize] lane, whose per-lane element ownership is architecture-opaque, so a per-lane companion
   reads cells other lanes of the same simdgroup produced. The threadgroup is exactly one simd width
   here (a single [Workgroup] slot of extent [sk_simd]), which is what makes that safe in practice; a
   cross-nest simdgroup barrier would be the formal fix. *)
let companion_geometry ~(site_syms : (Idx.symbol * int) list) ~(skip : Idx.symbol list)
    ~(expanded_zeros : Ir.Tnode.t list) ~(annotate : int -> Idx.symbol -> Sched.schedule)
    (opt : LL.optimized) : (Sched.schedule, string) Result.t =
  let plc = opt.LL.optimize_ctx.LL.placements in
  let rec writes_materialized (llc : LL.t) =
    match llc with
    | LL.Set { tn; _ } | LL.Set_dynamic { tn; _ } | LL.Set_from_vec { tn; _ } | LL.Zero_out tn
    | LL.Tile_mma { d = tn, _; _ } ->
        Ir.Tnode.Placements.is_materialized_peek plc tn
    | LL.Seq (a, b) -> writes_materialized a || writes_materialized b
    | LL.For_loop { body; _ } | LL.If { body; _ } -> writes_materialized body
    | _ -> false
  in
  let mentions syms stmt =
    List.exists (nest_loop_syms [] stmt) ~f:(fun s -> List.mem syms s ~equal:Idx.equal_symbol)
  in
  let site_sym_list = List.map site_syms ~f:fst in
  (* Only nests that write a MATERIALIZED node need covering — [validate_parallel]'s rule is about
     shared memory, and routine-local scratch is per-thread by construction. Restricting the demand
     this way also keeps the query out of the way of the pipelines it never needed to constrain: a
     site with nothing to cover neither consults the analysis nor can be failed by it. *)
  let needs =
    List.filter (LL.flat_lines [ opt.LL.llc ]) ~f:(fun stmt ->
        match stmt with
        | LL.Noop | LL.Comment _ -> false
        | LL.Zero_out tn when List.exists expanded_zeros ~f:(Ir.Tnode.equal tn) -> false
        | _ ->
            writes_materialized stmt
            && (not (mentions site_sym_list stmt))
            && not (mentions skip stmt))
  in
  if List.is_empty needs then Ok []
  else
    let shape cs = String.concat ~sep:"x" (List.map cs ~f:(fun (_, e) -> Int.to_string e)) in
    let same_shape cs =
      List.length cs = List.length site_syms
      && List.for_all2_exn cs site_syms ~f:(fun (_, e) (_, e') -> e = e')
    in
    let written stmt =
      let acc = ref [] in
      let rec go (llc : LL.t) =
        match llc with
        | LL.Set { tn; _ } | LL.Set_dynamic { tn; _ } | LL.Set_from_vec { tn; _ } | LL.Zero_out tn
        | LL.Tile_mma { d = tn, _; _ } ->
            if Ir.Tnode.Placements.is_materialized_peek plc tn then
              acc := Ir.Tnode.debug_name tn :: !acc
        | LL.Seq (a, b) ->
            go a;
            go b
        | LL.For_loop { body; _ } | LL.If { body; _ } -> go body
        | _ -> ()
      in
      go stmt;
      String.concat ~sep:"," (List.dedup_and_sort ~compare:String.compare !acc)
    in
    match Sched.aligned_chains ~max_chain:(List.length site_syms) ~expanded_zeros opt with
    | None ->
        Error
          (Printf.sprintf
             "the cross-nest race analysis bails on this routine, so the %d companion nest(s) (%s) \
              cannot be given aligned geometry"
             (List.length needs)
             (String.concat ~sep:"; " (List.map needs ~f:written)))
    | Some chains ->
        (* The site's own nest must keep the analysis' full chain: a trimmed one means the nests
           could not be aligned at this arity, and a companion annotated anyway would read cells
           another thread wrote, with no intra-kernel synchronization to order them. *)
        let own_ok =
          List.exists chains ~f:(fun (_, cs) ->
              List.equal (fun (a, _) (b, _) -> Idx.equal_symbol a b) cs site_syms && same_shape cs)
        in
        (* Loop symbols are unique per loop construct, so a nest is identified by its chain's
           outermost symbol occurring among the statement's loops. *)
        let chain_of stmt =
          let syms = nest_loop_syms [] stmt in
          List.find_map chains ~f:(fun (_, cs) ->
              match cs with
              | (s, _) :: _ when List.mem syms s ~equal:Idx.equal_symbol -> Some cs
              | _ -> None)
        in
        if not own_ok then
          Error
            (Printf.sprintf
               "the accumulation nest's aligned chain was trimmed below its %s geometry, so its \
                companions cannot share it"
               (shape site_syms))
        else
          List.fold_until needs ~init:[]
            ~f:(fun acc stmt ->
              match chain_of stmt with
              | Some cs when same_shape cs ->
                  Continue (acc @ List.concat (List.mapi cs ~f:(fun pos (s, _) -> annotate pos s)))
              | Some cs ->
                  Stop
                    (Error
                       (Printf.sprintf
                          "companion nest writing %s has aligned chain %s, the accumulation nest %s"
                          (written stmt) (shape cs) (shape site_syms)))
              | None ->
                  Stop
                    (Error
                       (Printf.sprintf
                          "companion nest writing %s has no aligned parallel chain" (written stmt))))
            ~finish:(fun acc -> Ok acc)

(* A companion nest that cannot take the accumulation nest's aligned geometry is a limitation of the
   generated sketch, not an arbitrary exception from a user transform — the same distinction
   [zero_geometry] draws for non-rank-2 outputs. Raise it at the narrow site as a typed [Unsupported]
   cause so strict candidate failure classification records a decline and keeps trying the remaining
   seeds; a plain [invalid_arg] here aborts the whole search under the default
   [strict_failure_classification=true]. *)
let companion_coverage_unsupported ~tensorized why =
  raise
    (Outcome.Cause_at
       ( Outcome.Transform,
         Outcome.Unsupported
           {
             feature = "autotune_sketch_companion_coverage";
             detail =
               Printf.sprintf "Autotune sketch: %sGPU matmul companion coverage (gh-521): %s"
                 (if tensorized then "tensorized " else "")
                 why;
           } ))

(* The chain the GPU matmul sketches annotate, as [companion_geometry] wants it: the accumulation
   nest's own outer loops in nest order — batch loops included (gh-ocannl-528) — which is exactly
   what {!Sched.aligned_chains} reports for that nest when the site is parallelizable at full
   arity. *)
let matmul_site_chain (site : matmul_site) =
  site.m_bo @ ((site.m_i, site.m_ni) :: site.m_bi) @ [ (site.m_j, site.m_nj) ]

(* Chain-position roles matching [matmul_site_chain]: batch positions get no annotation (batch
   loops stay [Serial] — the 3-slot grid budget is spent on the row/column blocks, and serial
   loops above hardware loops are legal: hardware loops bind, not iterate), row/column positions
   get the pipeline's geometry. *)
let matmul_chain_roles (site : matmul_site) : [ `Batch | `Row | `Col ] list =
  List.map site.m_bo ~f:(fun _ -> `Batch)
  @ (`Row :: List.map site.m_bi ~f:(fun _ -> `Batch))
  @ [ `Col ]

(* Hoist interior batch loops above the [m_i] loop (gh-ocannl-528), making the [i x j x k]
   micro-kernel perfectly nested for the splits, sinks and [Tensorize] below. Sequential adjacent
   [Swap]s: after each, [m_i] is directly above the next interior batch loop. *)
let batch_hoist_swaps (site : matmul_site) : Sched.schedule =
  List.map site.m_bi ~f:(fun (g, _) -> Sched.Swap { outer = site.m_i; inner = g })

(* The register-blocktiled GPU matmul (schedule_register_matmul.ml): each output dimension split
   twice (block tile -> Grid, register tile -> Workgroup), register loops sunk innermost, operands
   staged through workgroup-shared tiles at the k-block loop, output privatized, register loops
   materially unrolled. The zeroing nest gets the same geometry (barriers need slot-uniform
   workgroup extents), and companion nests the matching per-position split pair
   ([companion_geometry], gh-ocannl-521). *)
let gpu_sketch_schedule ~(opt : LL.optimized) (site : matmul_site)
    { sk_bm = bm; sk_bn = bn; sk_bk = bk; sk_tm = tm; sk_tn = tn; sk_epilogue; _ } : Sched.schedule =
  (* One geometry description drives the accumulation nest, the expanded zeroing nest and the
     companion nests: per row/column chain position, the block split (Grid) and the register split
     (Workgroup), which is what makes their slots and workgroup extents agree by construction;
     batch positions stay [Serial] (gh-ocannl-528). *)
  let annotate_role role sym =
    match role with
    | `Batch -> []
    | (`Row | `Col) as rc ->
        let blk, reg = match rc with `Row -> (bm, tm) | `Col -> (bn, tn) in
        let sp, _, inner = Sched.split ~axis:sym ~factor:blk ~outer:LL.Grid ~inner:LL.Serial in
        let sp2, _, _ = Sched.split ~axis:inner ~factor:reg ~outer:LL.Workgroup ~inner:LL.Serial in
        [ sp; sp2 ]
  in
  let roles = Array.of_list (matmul_chain_roles site) in
  let annotate pos sym = annotate_role roles.(pos) sym in
  let cops =
    match
      companion_geometry ~site_syms:(matmul_site_chain site)
        ~skip:(if sk_epilogue then epilogue_tail_loop_syms ~target:site.m_d opt else [])
        ~expanded_zeros:(if site.m_zeroed then [ site.m_d ] else [])
        ~annotate opt
    with
    | Ok ops -> ops
    | Error why -> companion_coverage_unsupported ~tensorized:false why
  in
  let zops =
    zero_geometry site ~mk_zops:(fun ~zi ~zj -> annotate_role `Row zi @ annotate_role `Col zj)
  in
  let zops = cops @ zops in
  let sp_i, _, i_i = Sched.split ~axis:site.m_i ~factor:bm ~outer:LL.Grid ~inner:LL.Serial in
  let sp_i2, i_w, i_t = Sched.split ~axis:i_i ~factor:tm ~outer:LL.Workgroup ~inner:LL.Serial in
  let sp_j, j_o, j_i = Sched.split ~axis:site.m_j ~factor:bn ~outer:LL.Grid ~inner:LL.Serial in
  let sp_j2, j_w, j_t = Sched.split ~axis:j_i ~factor:tn ~outer:LL.Workgroup ~inner:LL.Serial in
  let sp_k, k_o, k_i = Sched.split ~axis:site.m_k ~factor:bk ~outer:LL.Serial ~inner:LL.Serial in
  let swaps = sink i_t [ j_o; j_w; j_t; k_o; k_i ] @ sink j_t [ k_o; k_i ] in
  batch_hoist_swaps site @ zops
  @ [ sp_i; sp_i2; sp_j; sp_j2; sp_k ]
  @ swaps
  @ [
      Sched.Stage
        {
          source = site.m_a;
          tile_loops = [ i_w; i_t; k_i ];
          shared = true;
          cooperative = None;
          hoisted = false;
          swizzle = None;
          pad_stride = None;
          pipeline_depth = 1;
        };
      Sched.Stage
        {
          source = site.m_b;
          tile_loops = [ k_i; j_w; j_t ];
          shared = true;
          cooperative = None;
          hoisted = false;
          swizzle = None;
          pad_stride = None;
          pipeline_depth = 1;
        };
      Sched.Privatize { target = site.m_d; over = k_o };
      Sched.Unroll { axis = i_t; materialize = true };
      Sched.Unroll { axis = j_t; materialize = true };
    ]

(* A constant operand eligible for hoisted (out-of-routine) packing (gh-ocannl-470). The same
   predicate enters the canonical digest ([Schedule_cache.canonicalize]), so a cached winner for a
   same-shape program of different operand constancy never replays here — hoisted candidates are
   always measured for constant sites. *)
let hoistable = Sched.hoistable_constant

(* The CPU operand-packing matmul (schedule_cpu_pack_matmul.ml): all-serial tiling with the tile
   loops sunk to [i_o j_o k_o k_i i_i j_i], operands packed into contiguous stack scratch, output
   privatized across the k-block loop. With [sk_hoist], constant operands are instead packed once at
   link time into the per-device constant pool. *)
let cpu_sketch_schedule (site : matmul_site) { sk_bm = bm; sk_bn = bn; sk_bk = bk; sk_hoist; _ } :
    Sched.schedule =
  let sp_i, _, i_i = Sched.split ~axis:site.m_i ~factor:bm ~outer:LL.Serial ~inner:LL.Serial in
  let sp_j, j_o, j_i = Sched.split ~axis:site.m_j ~factor:bn ~outer:LL.Serial ~inner:LL.Serial in
  let sp_k, k_o, k_i = Sched.split ~axis:site.m_k ~factor:bk ~outer:LL.Serial ~inner:LL.Serial in
  batch_hoist_swaps site
  @ [ sp_i; sp_j; sp_k ]
  @ sink i_i [ j_o; j_i; k_o; k_i ]
  @ sink j_i [ k_o; k_i; i_i ]
  @ [
      Sched.Stage
        {
          source = site.m_a;
          tile_loops = [ i_i; k_i ];
          shared = false;
          cooperative = None;
          hoisted = sk_hoist && hoistable site.m_a;
          swizzle = None;
          pad_stride = None;
          pipeline_depth = 1;
        };
      Sched.Stage
        {
          source = site.m_b;
          tile_loops = [ k_i; j_i ];
          shared = false;
          cooperative = None;
          hoisted = sk_hoist && hoistable site.m_b;
          swizzle = None;
          pad_stride = None;
          pipeline_depth = 1;
        };
      Sched.Privatize { target = site.m_d; over = k_o };
    ]

(* Tensorized (tile-MMA) GPU matmul (docs/proposals/tensorize-mma.md; the pinned pipelines of
   schedule_mma_matmul.ml): Split the output dims into Grid blocks, then [Tensorize] the inner
   micro-kernel into a [Tile_mma] block statement. Stage-only composition — [Privatize] must NOT
   join it: it would relocate the accumulator into thread-local scratch, which the MMA loads cannot
   address ([mma_syntax] declines thread-space operands, silently costing the whole tensorization),
   and [Tile_mma]'s block semantics already keep the accumulator fragments register-resident across
   the reduction. With [sk_bk = 0] the single block statement spans the full reduction, streaming
   operand tiles from device memory and amortizing [d] traffic entirely; with [sk_bk > 0] both
   operands are staged through cooperative shared tiles at the k-block loop (lane-aware Stage),
   costing one [d] fragment load/store per k-block. The zeroing nest mirrors the accumulation's grid
   geometry, with an inner Workgroup loop of extent [sk_simd] covering the lane slot
   (barrier-strength uniformity: every workgroup extent must equal the lane width once a [Tile_mma]
   is present) — the seeds constrain [sk_bn = sk_simd] so the zeroing's grid blocks align with
   [j]'s. Companion nests (a bias/relu tail; the elementwise statements an aligned-merged fission
   segment carries) get the same geometry, which is what lets an UNFUSED tensorized candidate compile
   at all (gh-ocannl-521): before, only the [Fuse_epilogue] twin could survive a companion, and when
   the fusion declined the seed had no surviving form. *)
let gpu_mma_sketch_schedule ~(opt : LL.optimized) (site : matmul_site)
    { sk_bm = bm; sk_bn = bn; sk_bk = bk; sk_simd = w; sk_epilogue; sk_swizzle; sk_depth; _ } :
    Sched.schedule =
  (* The column role splits at the lane width, not at [bn]: the inner loop IS the workgroup slot
     the [Tile_mma]'s lane occupies, and a barrier-carrying kernel requires equal extents at a
     slot. The seeds constrain [sk_bn = sk_simd], so this is also the accumulation nest's column
     block. Batch positions stay [Serial] (gh-ocannl-528). *)
  let annotate_role role sym =
    match role with
    | `Batch -> []
    | `Row ->
        let sp, _, _ = Sched.split ~axis:sym ~factor:bm ~outer:LL.Grid ~inner:LL.Serial in
        [ sp ]
    | `Col ->
        let sp, _, _ = Sched.split ~axis:sym ~factor:w ~outer:LL.Grid ~inner:LL.Workgroup in
        [ sp ]
  in
  let roles = Array.of_list (matmul_chain_roles site) in
  let annotate pos sym = annotate_role roles.(pos) sym in
  let cops =
    match
      companion_geometry ~site_syms:(matmul_site_chain site)
        ~skip:(if sk_epilogue then epilogue_tail_loop_syms ~target:site.m_d opt else [])
        ~expanded_zeros:(if site.m_zeroed then [ site.m_d ] else [])
        ~annotate opt
    with
    | Ok ops -> ops
    | Error why -> companion_coverage_unsupported ~tensorized:true why
  in
  let zops =
    zero_geometry site ~mk_zops:(fun ~zi ~zj -> annotate_role `Row zi @ annotate_role `Col zj)
  in
  let zops = cops @ zops in
  let sp_i, _, i_i = Sched.split ~axis:site.m_i ~factor:bm ~outer:LL.Grid ~inner:LL.Serial in
  let sp_j, j_o, j_i = Sched.split ~axis:site.m_j ~factor:bn ~outer:LL.Grid ~inner:LL.Serial in
  if bk = 0 then
    let tz, _lane = Sched.tensorize ~i:i_i ~j:j_i ~k:site.m_k ~simd_width:w in
    batch_hoist_swaps site @ zops @ [ sp_i; sp_j ] @ sink i_i [ j_o ] @ [ tz ]
  else
    let sp_k, k_o, k_i = Sched.split ~axis:site.m_k ~factor:bk ~outer:LL.Serial ~inner:LL.Serial in
    let tz, _lane = Sched.tensorize ~i:i_i ~j:j_i ~k:k_i ~simd_width:w in
    (* Pad-composition seeding (gh-ocannl-485): with both operands staged through zero-fringe
       cooperative tiles, non-multiple extents pad to the block sizes — the guards land on the leaf
       accumulation, [Tensorize] moves the row/column masks to the fragment transfers and
       discharges the reduction mask against the staged tiles. *)
    batch_hoist_swaps site
    @ pad_to ~axis:site.m_i ~extent:site.m_ni bm
    @ pad_to ~axis:site.m_j ~extent:site.m_nj bn
    @ pad_to ~axis:site.m_k ~extent:site.m_nk bk
    @ zops
    @ [ sp_i; sp_j; sp_k ] @ sink i_i [ j_o ] @ sink j_i [ k_o ] @ sink i_i [ k_o ]
    @ [
        (* The swizzled twin (gh-ocannl-481 item 3, D3) marks BOTH operand tiles: the tile sizes and
           the whole rest of the pipeline are identical to its plain sibling, so a timing difference
           between the two is the layout's, and nothing else's. *)
        Sched.Stage
          {
            source = site.m_a;
            tile_loops = [ i_i; k_i ];
            shared = true;
            cooperative = Some w;
            hoisted = false;
            swizzle = sk_swizzle;
            pad_stride = None;
            pipeline_depth = sk_depth;
          };
        Sched.Stage
          {
            source = site.m_b;
            tile_loops = [ k_i; j_i ];
            shared = true;
            cooperative = Some w;
            hoisted = false;
            swizzle = sk_swizzle;
            pad_stride = None;
            pipeline_depth = sk_depth;
          };
        tz;
      ]

(* Whole-triple tensorized CPU matmul (gh-ocannl-469; bin/schedule_bench.ml's [tensorize] variant):
   one [Tile_mma] statement the C backends render tinyBLAS-style — the C-tile in an RM×RN grid of
   vector registers held across the k-loop, edges peeled. The zeroing's column loop becomes the
   Workgroup axis with the lane width matching its extent (coverage rule; the lane loop renders
   serially on the C backends). With [sk_bm > 0] the row loops split into pool-parallel Grid blocks;
   [sk_bm = 0] keeps the single-statement form. *)
let cpu_mma_sketch_schedule (site : matmul_site) { sk_bm = bm; _ } : Sched.schedule =
  let zops =
    zero_geometry site ~mk_zops:(fun ~zi ~zj ->
        let rz = Sched.Retype { axis = zj; ty = LL.Workgroup } in
        if bm = 0 then [ rz ]
        else
          let sp_zi, _, _ = Sched.split ~axis:zi ~factor:bm ~outer:LL.Grid ~inner:LL.Serial in
          [ sp_zi; rz ])
  in
  if bm = 0 then
    let tz, _lane = Sched.tensorize ~i:site.m_i ~j:site.m_j ~k:site.m_k ~simd_width:site.m_nj in
    batch_hoist_swaps site @ zops @ [ tz ]
  else
    let sp_i, _, i_i = Sched.split ~axis:site.m_i ~factor:bm ~outer:LL.Grid ~inner:LL.Serial in
    let tz, _lane = Sched.tensorize ~i:i_i ~j:site.m_j ~k:site.m_k ~simd_width:site.m_nj in
    batch_hoist_swaps site @ zops @ [ sp_i; tz ]

(* Cache-blocked, operand-packed tensorized CPU matmul: [Tile_mma] composed with the S4 packing
   pipeline (the remaining piece of gh-ocannl-469). GEBP loop structure, all-Serial: [j_o? { k_o {
   pack B~[bk x bn]; i_o { pack A~[bm x bk]; Tile_mma(bm, bn, bk) } } }] — the packing [Stage]s land
   at their own anchors (B~ at [k_o], once per (j_o, k_o) block; A~ at [i_o]) and the register-tiled
   micro-kernel streams the contiguous, cache-resident tiles ([lda = bk], [ldb = bn]). [tile_loops]
   are passed in micro-kernel order ([k_i; j_i] for B), so a transposed source packs into the
   normalized layout and [Tensorize] sees [ta = tb = false]. [sk_bn = 0] leaves [j] unsplit (one B~
   row panel of [bk x nj] per k-block). The lane width is 1: the C backends render the lane loop
   serially, and a unit lane keeps the kernel's parallel geometry trivial. Hoisted packing (constant
   operands, gh-ocannl-470) is proposed per operand like the scalar S4 pipeline.

   With [sk_grid], the row-block loop [i_o] is [Grid]-typed and pool-parallelizes; the whole-node
   [Zero_out] of the output — no longer legal beside a hardware-annotated loop ([validate_parallel])
   — expands into a nest whose row loop Grid-splits with the same [bm] geometry ([zero_geometry];
   the unit-lane Workgroup axis has extent 1, stays inactive, and needs no coverage from the zeroing
   nest). Four shapes (see [sk_grid]):

   - [sk_hoist]: hoisted-only packing — hoistable operands are packed at link time into the constant
   pool, the rest are read in place, so the kernel body touches only materialized buffers; the Grid
   loop stays outermost (one dispatch spanning the whole GEBP triple). The typical inference GEMM:
   activations (in place) x constant weights (hoisted-packed panel). - [sk_hoist] with
   [sk_pack_rest]: the mixed grid-outermost shape (gh-ocannl-473) — same loop structure, but
   non-hoistable operands get a non-hoisted in-kernel Stage; their tiles land inside the Grid body
   and rely on the renderer's per-chunk privatization (an in-place read forfeits the pack entirely;
   an A~ tile is [bm x bk], comfortably per-chunk). - [sk_pack_rest] alone: grid-outermost in-kernel
   packing (gh-ocannl-475) — both operands pack inside the Grid body, each chunk re-packing its own
   B~ panel; one dispatch, tiles under the renderer's per-chunk cap. - Otherwise, in-kernel packing:
   [i_o] sinks under [j_o]/[k_o] exactly as in the serial shape, so the B~ panel packs outside the
   Grid body (read-only inside, shared across the row-block chunks) while the per-row-block A~ tile
   is privatized to per-chunk block-scope storage by the renderer ([C_syntax.parallel_grid_safe]'s
   privatization rule). *)
let cpu_mma_pack_sketch_schedule (site : matmul_site)
    { sk_bm = bm; sk_bn = bn; sk_bk = bk; sk_hoist; sk_grid; sk_pack_rest; _ } : Sched.schedule =
  let outer_i = if sk_grid then LL.Grid else LL.Serial in
  let grid_outermost = sk_grid && (sk_hoist || sk_pack_rest) in
  let sp_i, i_o, i_i = Sched.split ~axis:site.m_i ~factor:bm ~outer:outer_i ~inner:LL.Serial in
  let sp_k, k_o, k_i = Sched.split ~axis:site.m_k ~factor:bk ~outer:LL.Serial ~inner:LL.Serial in
  let splits, j_col, j_swaps =
    if bn = 0 then ([ sp_i; sp_k ], site.m_j, [])
    else
      let sp_j, j_o, j_i =
        Sched.split ~axis:site.m_j ~factor:bn ~outer:LL.Serial ~inner:LL.Serial
      in
      ([ sp_i; sp_j; sp_k ], j_i, sink i_i [ j_o ] @ if grid_outermost then [] else sink i_o [ j_o ])
  in
  let stage ~hoisted source tile_loops =
    Sched.Stage { source; tile_loops; shared = false; cooperative = None; hoisted; swizzle = None; pad_stride = None; pipeline_depth = 1 }
  in
  let stages =
    if grid_outermost then
      List.filter_map
        [ (site.m_b, [ k_i; j_col ]); (site.m_a, [ i_i; k_i ]) ]
        ~f:(fun (src, tls) ->
          if hoistable src then Some (stage ~hoisted:true src tls)
          else if sk_pack_rest then Some (stage ~hoisted:false src tls)
          else None)
    else
      [
        stage ~hoisted:(sk_hoist && hoistable site.m_b) site.m_b [ k_i; j_col ];
        stage ~hoisted:(sk_hoist && hoistable site.m_a) site.m_a [ i_i; k_i ];
      ]
  in
  let zops =
    if not sk_grid then []
    else
      zero_geometry site ~mk_zops:(fun ~zi ~zj:_ ->
          let sp_zi, _, _ = Sched.split ~axis:zi ~factor:bm ~outer:LL.Grid ~inner:LL.Serial in
          [ sp_zi ])
  in
  (* Pad-composition seeding (gh-ocannl-485), only when both operands are staged (hoisted packing
     zero-fills its pad slots, so it qualifies): non-multiple extents pad to the block sizes and
     [Tensorize] masks the fragment transfers. An in-place operand read cannot absorb a pad, so the
     grid-outermost hoisted-only shape pads only if every operand packs. *)
  let both_staged = List.length stages = 2 in
  let pads =
    if not both_staged then []
    else
      pad_to ~axis:site.m_i ~extent:site.m_ni bm
      @ (if bn = 0 then [] else pad_to ~axis:site.m_j ~extent:site.m_nj bn)
      @ pad_to ~axis:site.m_k ~extent:site.m_nk bk
  in
  let tz, _lane = Sched.tensorize ~i:i_i ~j:j_col ~k:k_i ~simd_width:1 in
  batch_hoist_swaps site @ pads @ zops @ splits @ j_swaps @ sink j_col [ k_o ] @ sink i_i [ k_o ]
  @ (if grid_outermost then [] else sink i_o [ k_o ])
  @ stages @ [ tz ]

(* Adjacent-transposition reorder of a perfect serial nest: [Swap]s that turn [current] (nest order,
   outermost first) into [target]. Selection sort — for each target position, bubble the wanted loop
   outward one level at a time (each [Swap] exchanges a directly-nested pair). *)
let reorder_swaps ~current ~target : Sched.schedule =
  let cur = Array.of_list current in
  let swaps = ref [] in
  List.iteri target ~f:(fun p want ->
      match Array.findi cur ~f:(fun _ s -> Idx.equal_symbol s want) with
      | (None | Some (0, _)) when p > 0 -> invalid_arg "Autotune.reorder_swaps: not a permutation"
      | None -> invalid_arg "Autotune.reorder_swaps: not a permutation"
      | Some (q, _) ->
          if q < p then invalid_arg "Autotune.reorder_swaps: not a permutation";
          for r = q downto p + 1 do
            swaps := Sched.Swap { outer = cur.(r - 1); inner = cur.(r) } :: !swaps;
            let tmp = cur.(r - 1) in
            cur.(r - 1) <- cur.(r);
            cur.(r) <- tmp
          done);
  List.rev !swaps

(* The segment's real top-level statements as the conv seeding counts them: glue excluded, and the
   conv site's own [Zero_out] excluded (the pipeline's zero geometry handles it); every other
   statement is a companion nest. *)
let conv_real_stmts (site : conv_site) (opt : LL.optimized) : LL.t list =
  List.filter (LL.flat_lines [ opt.LL.llc ]) ~f:(function
    | LL.Noop | LL.Comment _ -> false
    | LL.Zero_out tn -> not (Ir.Tnode.equal tn site.c_d)
    | _ -> true)

(* The conv output's epilogue tail (see [epilogue_tail_loop_syms]): the fused twins on aligned-merged
   segments omit the preset's [Retype] on that nest (fuse-before-annotate, gh-ocannl-501). *)
let conv_tail_loop_syms (site : conv_site) (opt : LL.optimized) : Idx.symbol list =
  epilogue_tail_loop_syms ~target:site.c_d opt

(* Whole-segment Grid alignment for the conv pipeline on merged segments (gh-ocannl-493): the
   pipeline's own [Retype] covers only the conv nest, so on an aligned-merged segment (lenet's
   conv+bias/relu+pooling) the companions' materialized writes would fail [validate_parallel]. Reuse
   the default CPU preset's aligned cross-nest analysis instead of re-proving alignment: a non-empty
   [Sched.default_cpu] schedule Grid-retypes the outermost qualifying loop of {e every}
   materialized-writing nest, with the equal-extent common-prefix trims applied — exactly the
   whole-segment geometry the fissioned default runs. Accept it as the conv sketch's grid ops when
   it covers the conv nest at its outermost loop (which must be an outer output loop of extent >= 2,
   so the pipeline's reorder keeps it outermost and pool chunking has work to split). *)
let conv_aligned_grid (site : conv_site) (opt : LL.optimized) : Sched.schedule option =
  match (site.c_outer, site.c_loops) with
  | (outermost, n) :: _, first :: _ when n >= 2 && Idx.equal_symbol outermost first -> (
      match Sched.default_cpu ~min_parallel:1 opt with
      | [] -> None
      | sched ->
          if
            List.exists sched ~f:(function
              | Sched.Retype { axis; ty = LL.Grid } -> Idx.equal_symbol axis outermost
              | _ -> false)
          then Some sched
          else None)
  | _ -> None

(* The current nest order after a [Split] of the GEMM row into [row_o { row_i }] (in place, as
   [rewrite_loop] produces it), for feeding [reorder_swaps]. A dividing block factor keeps the split
   guard-free (see [Schedule.apply_op]'s [Split]), so the nest stays a perfect serial nest and the
   subsequent [Swap]s are well-formed. *)
let conv_split_row_current (site : conv_site) ~row_o ~row_i : Idx.symbol list =
  List.concat_map site.c_loops ~f:(fun s ->
      if Idx.equal_symbol s site.c_row then [ row_o; row_i ] else [ s ])

(* The implicit-GEMM conv pipeline (gh-ocannl-493), CPU route: reorder the accumulation nest to
   [outer..; kernel..; row; oc; ic], pack the input's [row × ic] strided-window slice and the
   kernel's [ic × oc] slice (both anchor under the innermost kernel-window loop; the packing IS
   im2col, one window slice at a time, and normalizes the kernel's stored layout), then [Tensorize
   (row, oc, ic)] — the register-tiled [Tile_mma] micro-kernel, with the accumulator contracted to a
   fragment resident across the whole kernel-window chain (gh-ocannl-480, gh-ocannl-501: one
   fragment init/store per output tile). With [sk_grid], the outermost output loop is [Grid]-typed
   and pool-parallelizes; a whole-node [Zero_out] of the output then expands with the matching
   geometry. On a segment with more than one companion statement the grid ops come from
   [conv_aligned_grid] instead, so every companion nest is annotated with the aligned whole-segment
   geometry (such segments carry no [Zero_out] — the preset's analysis bails on those, and the seeds
   gate accordingly).

   With [sk_bm > 0] (gh-ocannl-500) the GEMM row is split into panels of [sk_bm] rows before the
   reorder — cache-blocked GEBP-style panels, the conv analog of [cpu_mma_sketch_schedule]'s
   row-block split — and the in-panel [row_i × oc] micro-kernel is tensorized (the register tiling
   peels its own sub-tile edges). [sk_bm] must divide [c_nrow] so the split stays guard-free (a
   remainder guard would break the reorder's perfect nesting). The panel loop's parallelism source
   depends on the segment: on a conv-alone segment the panel loop is [Grid]-typed directly (one pool
   chunk per row-block); on an aligned-merged segment (conv + materialized companions, e.g. lenet's
   conv+bias/relu+pooling) the whole-segment [Grid] geometry comes from [conv_aligned_grid] as for
   the unblocked flavor and the panel loop stays [Serial] — pure cache blocking within each pool
   chunk. Both cases are unzeroed (the seeds gate accordingly): the [Zero_out] lives in its own
   [`Zeros] segment, so no zero geometry is needed. *)
let cpu_conv_sketch_schedule ~(opt : LL.optimized) (site : conv_site)
    { sk_grid; sk_bm; sk_epilogue; _ } : Sched.schedule =
  let stage source tile_loops =
    Sched.Stage
      { source; tile_loops; shared = false; cooperative = None; hoisted = false; swizzle = None; pad_stride = None; pipeline_depth = 1 }
  in
  (* Fuse-before-annotate (gh-ocannl-501): the fused twin of an aligned-merged seed omits the preset
     [Retype] on the tail nest [Fuse_epilogue] consumes — see [conv_tail_loop_syms]. *)
  let drop_tail_retypes sched =
    if not sk_epilogue then sched
    else
      let tail_syms = conv_tail_loop_syms site opt in
      List.filter sched ~f:(function
        | Sched.Retype { axis; _ } -> not (List.mem tail_syms axis ~equal:Idx.equal_symbol)
        | _ -> true)
  in
  if sk_bm > 0 then (
    (* A non-dividing row block pads the row to the block size (gh-ocannl-485): the pad's
       leaf-statement guards keep the nest perfectly nested for the reorder's [Swap]s, and both
       operands pack through zero-fringe tiles, so [Tensorize] masks the fragment transfers. *)
    let row_pads = pad_to ~axis:site.c_row ~extent:site.c_nrow sk_bm in
    (* On a merged segment the aligned whole-segment [Grid] annotation parallelizes; the panel loop
       is a serial cache block. On a conv-alone segment the panel loop is the parallel [Grid]. *)
    let grid_ops, panel_axis =
      if List.length (conv_real_stmts site opt) > 2 then
        match conv_aligned_grid site opt with
        | Some sched -> (drop_tail_retypes sched, LL.Serial)
        | None -> invalid_arg "Autotune conv sketch: companion nests do not align for Grid"
      else ([], LL.Grid)
    in
    let sp_row, row_o, row_i =
      Sched.split ~axis:site.c_row ~factor:sk_bm ~outer:panel_axis ~inner:LL.Serial
    in
    let current = conv_split_row_current site ~row_o ~row_i in
    let target =
      List.map site.c_outer ~f:fst @ [ row_o ] @ site.c_kernel @ [ row_i; site.c_oc; site.c_red ]
    in
    let tz, _lane = Sched.tensorize ~i:row_i ~j:site.c_oc ~k:site.c_red ~simd_width:1 in
    row_pads @ grid_ops
    @ (sp_row :: reorder_swaps ~current ~target)
    @ [ stage site.c_a [ row_i; site.c_red ]; stage site.c_b [ site.c_red; site.c_oc ]; tz ])
  else
    let loop_syms =
      List.map site.c_outer ~f:fst @ site.c_kernel @ [ site.c_row; site.c_oc; site.c_red ]
    in
    let tz, _lane = Sched.tensorize ~i:site.c_row ~j:site.c_oc ~k:site.c_red ~simd_width:1 in
    let zops, grid_ops =
      if not sk_grid then ([], [])
      else if List.length (conv_real_stmts site opt) > 2 then
        match conv_aligned_grid site opt with
        | Some sched -> ([], drop_tail_retypes sched)
        | None -> invalid_arg "Autotune conv sketch: companion nests do not align for Grid"
      else
        match site.c_outer with
        | [] -> invalid_arg "Autotune conv sketch: no outer loop to Grid-parallelize"
        | (outermost, _) :: _ ->
            let zops =
              if not site.c_zeroed then []
              else
                let ez, zsyms = Sched.expand_zero ~tn:site.c_d in
                match zsyms with
                | z0 :: _ -> [ ez; Sched.Retype { axis = z0; ty = LL.Grid } ]
                | [] -> [ ez ]
            in
            (zops, [ Sched.Retype { axis = outermost; ty = LL.Grid } ])
    in
    zops @ grid_ops
    @ reorder_swaps ~current:site.c_loops ~target:loop_syms
    @ [ stage site.c_a [ site.c_row; site.c_red ]; stage site.c_b [ site.c_red; site.c_oc ]; tz ]

(* The GPU staged leg of the implicit-GEMM conv pipeline (gh-ocannl-493): the same loop
   re-association as the CPU route, with the outer output loops [Grid]-typed (one threadgroup per
   outer coordinate, the kernel-window loops serial inside it) and both slices staged through
   cooperative workgroup-shared tiles at the kernel-window anchor (lane-aware [Stage], the lane
   width matching [Tensorize]'s — barrier-strength uniformity). Reusing [Tensorize] inherits the
   accumulator contraction (gh-ocannl-480) unchanged: the [row × oc] fragment stays resident across
   the whole kernel-window chain (gh-ocannl-501), on Metal in simdgroup registers. Zeroed sites are
   gated off at the seeds — the GPU leg targets fission segments, whose [Zero_out] lives in its own
   [`Zeros] segment.

   With [sk_bm > 0] (gh-ocannl-500) the GEMM row is additionally split into [Grid] blocks of [sk_bm]
   rows: one threadgroup per (outer.., row-block) coordinate instead of one per outer coordinate, so
   small-spatial sites fill the device better. [sk_bm] must divide [c_nrow] (a remainder guard would
   push the cooperative-load barriers under divergent control flow, rejected by
   [validate_parallel]). Only the row is blocked — a 2-D conv already binds two outer [Grid] loops
   (batch, the non-row output spatial axis), so a second [Grid] block on [oc] would exceed the
   three-slot budget; [oc] stays the tensorized column extent. The block loop [row_o] carries no
   companion nest here (unzeroed segments), so no cross-nest zero geometry is needed. *)
let gpu_conv_sketch_schedule (site : conv_site)
    { sk_simd = w; sk_bm; sk_bn; sk_bk; sk_tm; sk_depth; _ } : Sched.schedule =
  let stage source tile_loops =
    Sched.Stage
      { source; tile_loops; shared = true; cooperative = Some w; hoisted = false; swizzle = None; pad_stride = None; pipeline_depth = sk_depth }
  in
  let outer_grid =
    List.map site.c_outer ~f:(fun (s, _) -> Sched.Retype { axis = s; ty = LL.Grid })
  in
  (* Pad-composition seeding (gh-ocannl-485): the conv seeds carry the intrinsic-tile pad multiples
     for the column ([sk_bn]) and reduction ([sk_bk]) extents, and — in the unblocked flavor — the
     row ([sk_tm]); a non-dividing row block pads to the block size. Both operand slices stage
     through zero-fringe cooperative tiles, so [Tensorize] masks the fragment transfers and
     discharges the reduction mask. *)
  let col_red_pads =
    (if sk_bn > 0 then pad_to ~axis:site.c_oc ~extent:site.c_noc sk_bn else [])
    @ if sk_bk > 0 then pad_to ~axis:site.c_red ~extent:site.c_nred sk_bk else []
  in
  if sk_bm > 0 then (
    let pads = pad_to ~axis:site.c_row ~extent:site.c_nrow sk_bm @ col_red_pads in
    let sp_row, row_o, row_i =
      Sched.split ~axis:site.c_row ~factor:sk_bm ~outer:LL.Grid ~inner:LL.Serial
    in
    let current = conv_split_row_current site ~row_o ~row_i in
    let target =
      List.map site.c_outer ~f:fst @ [ row_o ] @ site.c_kernel @ [ row_i; site.c_oc; site.c_red ]
    in
    let tz, _lane = Sched.tensorize ~i:row_i ~j:site.c_oc ~k:site.c_red ~simd_width:w in
    pads
    @ (outer_grid @ [ sp_row ])
    @ reorder_swaps ~current ~target
    @ [ stage site.c_a [ row_i; site.c_red ]; stage site.c_b [ site.c_red; site.c_oc ]; tz ])
  else
    let pads =
      (if sk_tm > 0 then pad_to ~axis:site.c_row ~extent:site.c_nrow sk_tm else [])
      @ col_red_pads
    in
    let loop_syms =
      List.map site.c_outer ~f:fst @ site.c_kernel @ [ site.c_row; site.c_oc; site.c_red ]
    in
    let tz, _lane = Sched.tensorize ~i:site.c_row ~j:site.c_oc ~k:site.c_red ~simd_width:w in
    pads @ outer_grid
    @ reorder_swaps ~current:site.c_loops ~target:loop_syms
    @ [ stage site.c_a [ site.c_row; site.c_red ]; stage site.c_b [ site.c_red; site.c_oc ]; tz ]

(* Building a sketch is a narrow phase seam of its own (gh-ocannl-536). Every [invalid_arg] above is
   an applicability precondition — no matmul site, a companion nest whose geometry the family cannot
   cover — i.e. the same verdict as a [Schedule.apply] precondition: this candidate is not
   applicable, and the search is better off recording a decline. Escaping untyped they were
   unclassified and therefore FATAL under strict classification, so a single inapplicable GPU sketch
   family ended the whole search (reproducible on Metal with test/operations/autotune_fission_sketch
   before this). Typing them here rather than around the whole transform closure keeps the boundary
   narrow, which is the point: an arbitrary exception escaping a transform stays fatal. *)
let sketch_schedule_unchecked ~p (opt : LL.optimized) : Sched.schedule =
  let sched, d =
    if p.sk_conv then
      match detect_conv opt.LL.llc with
      | None -> invalid_arg "Autotune sketch: no convolution site detected"
      | Some site ->
          ( (if p.sk_gpu then gpu_conv_sketch_schedule site p
             else cpu_conv_sketch_schedule ~opt site p),
            site.c_d )
    else
      match detect_matmul opt.LL.llc with
      | None -> invalid_arg "Autotune sketch: no matmul micro-kernel detected"
      | Some site ->
          let sched =
            if p.sk_mma then
              if p.sk_gpu then gpu_mma_sketch_schedule ~opt site p
              else if p.sk_bk > 0 then cpu_mma_pack_sketch_schedule site p
              else cpu_mma_sketch_schedule site p
            else if p.sk_gpu then gpu_sketch_schedule ~opt site p
            else cpu_sketch_schedule site p
          in
          (sched, site.m_d)
  in
  if p.sk_epilogue then
    (* [shared] is the fragment-site knob: only the GPU MMA sketches store through the contracted
       fragment; the block-tiling pipeline stores through [Privatize], where [Fuse_epilogue] rejects
       [shared] outright and the twin would fail for the wrong reason. *)
    sched @ [ Sched.Fuse_epilogue { target = d; shared = p.sk_gpu && p.sk_mma } ]
  else sched

let sketch_schedule ~p (opt : LL.optimized) : Sched.schedule =
  match sketch_schedule_unchecked ~p opt with
  | sched -> sched
  | exception Invalid_argument detail ->
      raise
        (Outcome.Cause_at
           (Outcome.Transform, Outcome.Illegal_schedule { check = "Autotune.sketch"; detail }))

(* Sketch seed parameters compatible with the site's extents. Fully staged tensorized pipelines no
   longer require dividing tiles: non-multiple extents seed [(pad, tensorize)] compositions
   (gh-ocannl-485) whose masked edges the tuner measures against scalar alternatives — pipelines
   that read an operand in place keep their divisibility gates. Unzeroed sites — the norm for fission segments,
   whose [Zero_out] lives in its own [`Zeros] segment — are proposable too: the pipelines skip the
   zero geometry (see [zero_geometry]), and a site whose kernel-mates cannot share the parallel
   geometry merely fails its candidate compile. *)
(* Conv seeds (gh-ocannl-493). CPU: the serial implicit-GEMM pipeline plus its Grid-parallel
   variant, pre-filtered by the register tiling's statically decidable rules like the matmul
   seeds (gh-ocannl-479): uniform f32/f64, fused accumulation form, and the micro-kernel column
   extent (the out-channel count) at least one vector of lanes. Layout orientation needs no
   pre-filter: both operands are packed, which normalizes any stored layout. GPU (backends with an
   mma capability): the staged pipeline ([gpu_conv_sketch_schedule]), pre-filtered by the
   intrinsic-tile divisibility of the micro-kernel extents (like the mma matmul seeds) and the
   shared-tile footprint against the workgroup-memory limit. Strided rows (stride-2 stems and
   downsample blocks) are seeded on both legs since the compacting [Stage] (gh-ocannl-502). *)
let conv_seed_params ~is_gpu ~is_cpu ~(limits : Ir.Backend_intf.hardware_limits)
    (opt : LL.optimized) : (sketch_params list * Ir.Tnode.t) option =
  match detect_conv opt.LL.llc with
  | None -> None
  | Some site ->
      let prec = Lazy.force site.c_d.Ir.Tnode.storage_prec in
      (* Strided rows are seeded (gh-ocannl-502): the row's tile part in the input access is the
         single term [stride*row] — the kernel-window symbol lands in the outer part, at the staging
         anchor — so the compacting [Stage] packs the window densely (tile axis sized by the loop
         extent, tile store/read at coefficient 1, only the load's source index and its edge guard
         keeping the stride), which satisfies [Tensorize]'s unit-coefficient index discipline.
         Stride-2 downsampling stems and blocks therefore reach the implicit-GEMM pipeline like
         unit-stride convs.

         Every axis must still be offset-free — which padded convs from the tensor front end are:
         the halo is part of the physically padded buffer, buffer indices absorb the shift, and
         [Stage]'s edge guards compare against the padded [Tn.dims], so staging padded convs is
         sound (pinned by the cvp pipeline leg of schedule_conv_gemm). The gate is retained as
         defense-in-depth against hand-built [Low_level] sites with genuine offsets, where the
         packing anchor would mispack (Codex P1 on PR #168). Candidates are timed, not
         value-checked, so unsound seeds must not be proposed at all. The gate applies to both legs:
         the GPU pipeline packs through the same [Stage] decomposition. *)
      let offset_free = List.for_all site.c_axes ~f:(fun cx -> cx.cx_offset = 0) in
      let real_stmts = conv_real_stmts site opt in
      let base =
        {
          sk_gpu = false;
          sk_mma = true;
          sk_simd = 0;
          sk_bm = 0;
          sk_bn = 0;
          sk_bk = 0;
          sk_tm = 0;
          sk_tn = 0;
          sk_hoist = false;
          sk_grid = false;
          sk_pack_rest = false;
          sk_conv = true;
          sk_epilogue = false;
          sk_swizzle = None;
          sk_depth = 1;
        }
      in
      let cpu_seeds =
        if not is_cpu then []
        else
          let uniform_f32_64 =
            (match prec with Ir.Ops.Single_prec _ | Ir.Ops.Double_prec _ -> true | _ -> false)
            && Ir.Ops.equal_prec (Lazy.force site.c_a.Ir.Tnode.storage_prec) prec
            && Ir.Ops.equal_prec (Lazy.force site.c_b.Ir.Tnode.storage_prec) prec
          in
          let lanes =
            limits.Ir.Backend_intf.simd_vector_bytes / max 1 (Ir.Ops.prec_in_bytes prec)
          in
          if
            not
              (limits.Ir.Backend_intf.simd_vector_bytes >= 8
              && lanes >= 2 && uniform_f32_64 && site.c_fma && site.c_noc >= lanes && offset_free)
          then []
          else
            (* Grid flavors need every materialized write in the routine covered by the Grid axis
               ([validate_parallel]), and the conv pipeline's own [Retype] only annotates the conv
               nest: seed them when the conv statement is alone — or has exactly one companion
               statement, the would-be epilogue tail, whose fused twin ([sk_grid] with
               [sk_epilogue]) relocates the tail write under the Grid loop (the unfused [sk_grid]
               candidate then fails validation and is skipped; the twin carries the Grid flavor —
               multi-window convs included, since the whole-window contraction, gh-ocannl-501, lands
               the store-back after the full kernel window). Segments with more companions (an
               aligned-merged segment, e.g. lenet's conv+bias/relu+pooling) are seeded when the
               default preset's aligned cross-nest analysis Grid-annotates the whole segment
               ([conv_aligned_grid]) — the pipeline then adopts that whole-segment geometry. The
               fused twin of an aligned-grid seed omits the preset [Retype] on the tail nest the
               fusion consumes (fuse-before-annotate, gh-ocannl-501; see [conv_tail_loop_syms]), so
               the twin compiles on merged segments too. *)
            let grid_ok =
              (match site.c_outer with (_, n) :: _ -> n >= 2 | [] -> false)
              && (List.length real_stmts <= 2 || Option.is_some (conv_aligned_grid site opt))
            in
            (* Cache-blocked row-panel flavors (gh-ocannl-500): split the GEMM row into panels of
               [sk_bm] rows ([cpu_conv_sketch_schedule]'s [sk_bm] leg). Dividing blocks only — the
               split must stay guard-free so the reorder's [Swap]s are well-formed — and at least
               two panels. Proposed on any unzeroed segment: a conv-alone segment
               [Grid]-parallelizes the panel loop, an aligned-merged segment adopts the
               whole-segment [Grid] geometry ([conv_aligned_grid]) and blocks the row serially for
               cache residency. The whole-routine zeroed graph keeps the serial/aligned flavors
               above (its [Zero_out] is in the routine, so the aligned analysis bails and the block
               flavor is not seeded). *)
            let block_ok =
              (not site.c_zeroed)
              && (List.length real_stmts <= 1 || Option.is_some (conv_aligned_grid site opt))
            in
            let row_blocks =
              if not block_ok then []
              else
                (* Non-dividing blocks pad the row (gh-ocannl-485, the builder emits the [Pad]);
                   require at least two (possibly padded) panels. *)
                List.filter_map [ 8; 16; 32 ] ~f:(fun bm ->
                    if blocks_of site.c_nrow bm >= 2 then Some { base with sk_bm = bm } else None)
            in
            (base :: (if grid_ok then [ { base with sk_grid = true } ] else [])) @ row_blocks
      in
      let gpu_seeds =
        match (is_gpu, limits.Ir.Backend_intf.mma) with
        | true, Some ({ Ir.Backend_intf.mma_simd_width = w; _ } as mma) -> (
            match
              mma_tile_for_precisions mma
                ~a_prec:(Lazy.force site.c_a.Ir.Tnode.storage_prec)
                ~b_prec:(Lazy.force site.c_b.Ir.Tnode.storage_prec)
                ~d_prec:(Lazy.force site.c_d.Ir.Tnode.storage_prec)
            with
            | None -> []
            | Some (tm_t, tn_t, tk_t) ->
                (* Zeroed sites are gated off: the GPU leg targets fission segments, whose [Zero_out]
               lives in its own [`Zeros] segment (a whole-routine zeroed GPU flavor would need the
               zero nest annotated with matching workgroup geometry — a follow-up). Companion
               gating mirrors the CPU grid flavors: on GPU there is no all-serial fallback, so any
               uncovered companion write fails [validate_parallel] — the one-companion seed only
               survives through its fused twin. *)
                (* The intrinsic-tile divisibility is now a PER-BLOCK property (gh-ocannl-500): the
               tensorized micro-kernel row is [sk_bm] (the block), not the whole [c_nrow], so a
               staged block flavor is proposable whenever [sk_bm] — a multiple of the intrinsic row
               tile that divides [c_nrow] — exists, and the whole-extent flavor ([sk_bm = 0]) only
               when [c_nrow] itself is a multiple. (Column and reduction stay whole-extent tensorized,
               so [c_noc] / [c_nred] keep their intrinsic-tile gates; blocking those would exceed the
               three-[Grid]-slot budget on 2-D convs — a follow-up.) A dividing block that is a
               multiple of the tile implies whole divisibility, so on ordinary shapes the block
               flavors add [Grid] device-fill splits rather than waking new sites; genuine edge
               peeling of the cooperative micro-kernel — a tensorized bulk beside a scalar remainder,
               which [Stage]'s single-index-vector rule blocks in v1 — is a recorded follow-up. *)
                (* Pad-composition seeding (gh-ocannl-485): non-multiple column/reduction/row
                   extents no longer gate the seeds — the builder pads them to the intrinsic tile
                   (the seed carries the multiples in [sk_bn]/[sk_bk]/[sk_tm], 0 = already a
                   multiple) and [Tensorize] masks the edges. The shared-tile footprint is computed
                   on the padded extents. *)
                let noc_p = blocks_of site.c_noc tn_t * tn_t in
                let nred_p = blocks_of site.c_nred tk_t * tk_t in
                let pad_n = if site.c_noc % tn_t = 0 then 0 else tn_t in
                let pad_k = if site.c_nred % tk_t = 0 then 0 else tk_t in
                let shared_bytes rows =
                  ((rows * nred_p) + (nred_p * noc_p)) * Ir.Ops.prec_in_bytes prec
                in
                let shared_fits ?(copies = 1) rows =
                  match limits.Ir.Backend_intf.max_workgroup_memory_bytes with
                  | Some cap -> copies * shared_bytes rows <= cap
                  | None -> true
                in
                let base_ok = offset_free && (not site.c_zeroed) && List.length real_stmts <= 2 in
                if not base_ok then []
                else
                  let rows_p = blocks_of site.c_nrow tm_t * tm_t in
                  let whole =
                    let pad_m = if site.c_nrow % tm_t = 0 then 0 else tm_t in
                    if shared_fits rows_p then
                      [ { base with sk_gpu = true; sk_simd = w; sk_tm = pad_m; sk_bn = pad_n; sk_bk = pad_k } ]
                    else []
                  in
                  let blocked =
                    List.filter_map [ 8; 16; 32 ] ~f:(fun bm ->
                        if bm % tm_t = 0 && blocks_of site.c_nrow bm >= 2 && shared_fits bm then
                          Some
                            { base with sk_gpu = true; sk_simd = w; sk_bm = bm; sk_bn = pad_n; sk_bk = pad_k }
                        else None)
                  in
                  (* The pipelined twins (gh-ocannl-487): every conv GPU flavor stages
                     cooperatively, so each unmasked flavor gets a twin per advertised depth —
                     gated on the [copies]-multiplied footprint, since the rotation allocates that
                     many tile copies. Masked flavors (any pad multiple set, or a row block that
                     does not divide the row extent) are not twinned: their pad masks keep
                     [Tensorize] on the barrier-bracketed per-call arm, whose leading bracket sits
                     between the prefetch and the compute — the copy must complete there, so the
                     twin could only pay the doubled footprint (Codex P2 on PR #303). *)
                  let masked p0 =
                    p0.sk_bn > 0 || p0.sk_bk > 0
                    ||
                    if p0.sk_bm = 0 then p0.sk_tm > 0 else site.c_nrow % p0.sk_bm <> 0
                  in
                  (* Depth twins additionally ride the async arms' element floor (Codex P2 on PR
                     #317, as in the matmul sketch): staged tiles of sub-4-byte elements render
                     the portable synchronous form only, so their twin could only pay the doubled
                     footprint. *)
                  let async_wide =
                    Ir.Ops.prec_in_bytes (Lazy.force site.c_a.Ir.Tnode.storage_prec) >= 4
                    && Ir.Ops.prec_in_bytes (Lazy.force site.c_b.Ir.Tnode.storage_prec) >= 4
                  in
                  let depth_twins =
                    List.concat_map (whole @ blocked) ~f:(fun p0 ->
                        if masked p0 || not async_wide then []
                        else
                          let rows = if p0.sk_bm = 0 then rows_p else p0.sk_bm in
                          List.filter_map mma.Ir.Backend_intf.mma_pipeline_depths ~f:(fun d ->
                              if shared_fits ~copies:d rows then Some { p0 with sk_depth = d }
                              else None))
                  in
                  whole @ blocked @ depth_twins)
        | _ -> []
      in
      let seeds = cpu_seeds @ gpu_seeds in
      if List.is_empty seeds then None
      else
        (* Fused-epilogue twins are proposed for every conv seed (gh-ocannl-501):
           [contract_tensorized_accumulator] contracts across the whole kernel-window chain, so the
           fragment store-back lands after the full window unconditionally and [Fuse_epilogue]'s
           exactly-once check passes by construction — multi-window (2-D) convs included. *)
        Some (seeds, site.c_d)

(* The matmul family as a refinement tree (gh-ocannl-514 phase 1): the hand-written seed
   enumeration factored into decision levels — pipeline, then per-pipeline shape/geometry levels,
   twins as their own level — with the seed list recovered as the tree's {!Sspace.leaves}, in the
   exact order the flat enumeration produced (enumeration order reaches candidate timing order and
   dedup keep-first, so the factoring must preserve it; levels therefore appear in emission order,
   e.g. the packing shape ABOVE its geometries). Children domains depend on earlier commitments —
   a packing shape constrains which geometries remain, a twin exists only for staged geometries —
   which is what makes this a tree of staged choices rather than a product of independent domains.
   Subtrees are lazy so a future fathom (phase 4) can prune a choice without forcing what is below
   it; a choice whose every child was filtered out is an infeasible node with no leaves. Pinned by
   test/operations/sketch_family_tree.ml against the pre-factoring golden. *)
(* CPU Grid shapes render on the pool only when the configuration allows it: an explicit
   [cc_parallel_grid=none] or [cc_parallel_chunks=1] makes [C_syntax.collect_parallel_grid]
   deterministically collect nothing, so the candidate runs serially under a Grid label. Only the
   explicit settings are mirrored here — [auto] resolves through the backend's compiler probe and
   pool sizing, which stay render-settled (the same seeding-vs-builder boundary as companion
   coverage). *)
let cpu_grid_rendering_disabled =
  lazy
    (let mode =
       String.lowercase
         (String.strip (Utils.get_global_arg ~arg_name:"cc_parallel_grid" ~default:"auto"))
     in
     if String.equal mode "none" then
       Some "cc_parallel_grid=none: Grid loops render serially, the shape would time under a Grid label"
     else
       match
         Int.of_string
           (String.strip (Utils.get_global_arg ~arg_name:"cc_parallel_chunks" ~default:"0"))
       with
       | 1 ->
           Some
             "cc_parallel_chunks=1: a single chunk renders serially, the shape would time under a \
              Grid label"
       | _ -> None
       | exception _ -> None)

let matmul_family_tree ~is_gpu ~is_cpu ~(limits : Ir.Backend_intf.hardware_limits) site :
    sketch_params Sspace.tree =
  let divides c n = c <= n && n % c = 0 in
  let leaf p = Sspace.Child (lazy (Sspace.Leaf p)) in
  let choice level children = Sspace.Choice { level; children } in
  (* [subt] defers subtree construction into the child's lazy: verdicts are decided at
     parent construction (fathoming needs them without expansion), subtrees are not. *)
  let subt t = Sspace.Child (lazy (t ())) in
  (* The first failing conjunct is the witness: a refutation names the one constraint whose
     violation already refutes every completion, not the whole gate. *)
  let refute_unless (conds : (bool * string) list) (ok : unit -> sketch_params Sspace.child) :
      sketch_params Sspace.child =
    match List.find conds ~f:(fun (c, _) -> not c) with
    | Some (_, witness) -> Sspace.Refuted witness
    | None -> ok ()
  in
  let ndiv what c ~into n =
    (divides c n, Printf.sprintf "%s=%d does not divide %s=%d" what c into n)
  in
  let base_params =
    {
      sk_gpu = false;
      sk_mma = false;
      sk_simd = 0;
      sk_bm = 0;
      sk_bn = 0;
      sk_bk = 0;
      sk_tm = 0;
      sk_tn = 0;
      sk_hoist = false;
      sk_grid = false;
      sk_pack_rest = false;
      sk_conv = false;
      sk_epilogue = false;
      sk_swizzle = None;
      sk_depth = 1;
    }
  in
  let blocktile_child =
    if is_gpu then
      let a_prec = Lazy.force site.m_a.Ir.Tnode.storage_prec in
      let b_prec = Lazy.force site.m_b.Ir.Tnode.storage_prec in
      subt (fun () -> choice "geometry"
           (List.map
              [ (64, 64, 8, 4, 4); (32, 32, 8, 4, 4); (16, 16, 8, 4, 4); (32, 32, 16, 2, 2); (16, 16, 8, 2, 2) ]
              ~f:(fun (bm, bn, bk, tm, tn) ->
                ( Printf.sprintf "bm%d bn%d bk%d tm%d tn%d" bm bn bk tm tn,
                  refute_unless
                    ([
                       ndiv "bm" bm ~into:"m" site.m_ni;
                       ndiv "bn" bn ~into:"n" site.m_nj;
                       ndiv "bk" bk ~into:"k" site.m_nk;
                       ndiv "tm" tm ~into:"bm" bm;
                       ndiv "tn" tn ~into:"bn" bn;
                     ]
                    @ (* The launch size is statically known — two Workgroup dimensions of [bm/tm]
                         and [bn/tn] threads — so a known thread cap refutes pre-compile what
                         [Schedule.check_hardware_limits_classified] would reject per candidate;
                         likewise the two [shared] operand stages' workgroup-memory floor. *)
                    (match limits.Ir.Backend_intf.max_threads_per_workgroup with
                    | Some cap when tm > 0 && tn > 0 && bm / tm * (bn / tn) > cap ->
                        [
                          ( false,
                            Printf.sprintf
                              "block tile launches %d threads per workgroup (bm/tm * bn/tn), \
                               exceeding the %d-thread limit"
                              (bm / tm * (bn / tn))
                              cap );
                        ]
                    | _ -> [])
                    @
                    match limits.Ir.Backend_intf.max_workgroup_memory_bytes with
                    | Some cap
                      when (bm * bk * Ir.Ops.prec_in_bytes a_prec)
                           + (bk * bn * Ir.Ops.prec_in_bytes b_prec)
                           > cap ->
                        [
                          ( false,
                            Printf.sprintf
                              "staged operand tiles need %d bytes of workgroup memory, exceeding \
                               the %d-byte limit"
                              ((bm * bk * Ir.Ops.prec_in_bytes a_prec)
                              + (bk * bn * Ir.Ops.prec_in_bytes b_prec))
                              cap );
                        ]
                    | _ -> [])
                    (fun () ->
                      leaf
                        {
                          base_params with
                          sk_gpu = true;
                          sk_bm = bm;
                          sk_bn = bn;
                          sk_bk = bk;
                          sk_tm = tm;
                          sk_tn = tn;
                        }) ))))
    else if is_cpu then
      (* Hoisted vs in-kernel packing stays a measured choice (gh-ocannl-470): when a constant
         operand can be packed at link time, propose each tiling in both flavors. The packing
         level sits ABOVE its geometries: the flat enumeration emitted all in-kernel tilings
         before the hoisted twins. *)
      let geoms hoist =
        choice "geometry"
          (List.map [ 16; 8 ] ~f:(fun b ->
               ( Printf.sprintf "b%d" b,
                 refute_unless
                   [
                     ndiv "b" b ~into:"m" site.m_ni;
                     ndiv "b" b ~into:"n" site.m_nj;
                     ndiv "b" b ~into:"k" site.m_nk;
                   ]
                   (fun () ->
                     leaf { base_params with sk_bm = b; sk_bn = b; sk_bk = b; sk_hoist = hoist }) )))
      in
      subt (fun () -> choice "packing"
           [
             ("in-kernel", subt (fun () -> geoms false));
             ( "hoisted",
               if hoistable site.m_a || hoistable site.m_b then subt (fun () -> geoms true)
               else
                 Sspace.Refuted
                   "hoisted packing needs a host-init-backed constant operand; neither operand is \
                    one" );
           ])
    else Sspace.Refuted "backend kind seeds no scalar blocktile pipeline"
  in
  let mma_child =
    match (is_gpu, limits.Ir.Backend_intf.mma) with
    | true, Some _ when Utils.debug_log_from_routines () ->
        (* Same predicate the GPU [mma_syntax] paths consult: under routine logging the emission
           skips the intrinsic and renders the scalar fallback, so every leaf would be timed (and
           cached) under a tensorized label. *)
        Sspace.Refuted
          "routine logging is active (debug_log_from_routines): the mma emission renders the \
           scalar fallback, so every leaf would time under a tensorized label"
    | true, Some { Ir.Backend_intf.mma_simd_width = w; _ }
      when Option.value_map limits.Ir.Backend_intf.max_threads_per_workgroup ~default:false
             ~f:(fun cap -> w > cap) ->
        (* The tensorization lane is a Workgroup axis of extent [w] in every geometry. *)
        Sspace.Refuted
          (Printf.sprintf
             "mma lane width %d exceeds the %d-thread workgroup limit" w
             (Option.value_exn limits.Ir.Backend_intf.max_threads_per_workgroup))
    | true, Some ({ Ir.Backend_intf.mma_simd_width = w; _ } as mma) -> (
        let a_prec = Lazy.force site.m_a.Ir.Tnode.storage_prec in
        let b_prec = Lazy.force site.m_b.Ir.Tnode.storage_prec in
        let d_prec = Lazy.force site.m_d.Ir.Tnode.storage_prec in
        match mma_tile_for_precisions mma ~a_prec ~b_prec ~d_prec with
        | None ->
            Sspace.Refuted
              (Printf.sprintf
                 "backend advertises no mma format tile for operands (%s, %s) with accumulator %s"
                 (Ir.Ops.prec_string a_prec) (Ir.Ops.prec_string b_prec)
                 (Ir.Ops.prec_string d_prec))
        | Some (tm_t, tn_t, tk_t) ->
            (* [bn = w] keeps the zeroing's column grid blocks aligned with [j]'s (see
               [gpu_mma_sketch_schedule]); [bk = 0] = unstaged full-K block. Staged seeds
               ([bk > 0]) no longer require the extents to be block multiples: the builder pads the
               non-multiple axes and [Tensorize] masks the edges (gh-ocannl-485) — block sizes must
               still be intrinsic-tile multiples. Unstaged seeds read the operands in place, so a
               pad cannot be absorbed and the full divisibility gates remain. *)
            (* The swizzled layout the emission can read for these formats, if any, and the tile
               extents it needs (gh-ocannl-481 item 3, D3): [Swizzle_b128] permutes whole 16-byte
               units, so each staged tile's minor extent — [bk] elements of A, [bn] of B — must span
               a power-of-two count > 1 of them. Judged here rather than left to raise inside
               [Schedule.apply]: an inapplicable twin is refuted with its witness, not merely
               failed. *)
            let staged_layout = mma_staged_layout_for_precisions mma ~a_prec ~b_prec ~d_prec in
            let b128_units_ok prec extent =
              let bytes = extent * Ir.Ops.prec_in_bytes prec in
              bytes % 16 = 0
              &&
              let units = bytes / 16 in
              units > 1 && units land (units - 1) = 0
            in
            let nmul what c ~of_ n =
              (c % n = 0, Printf.sprintf "%s=%d is not a multiple of the intrinsic tile %s=%d" what c of_ n)
            in
            (* A sound workgroup-memory floor for staged geometries: any completion allocates at
               least the cooperative operand tiles ([bm x bk] of A, [bk x bn] of B), [depth]-fold
               under software pipelining — other shared allocations only add. Exceeding the
               advertised limit refutes every completion below the child pre-compile, where
               [Schedule.check_hardware_limits_classified] would otherwise reject it one candidate
               compile at a time. [None]/unknown limit refutes nothing. *)
            let staged_tiles_exceed ~bm ~bn ~bk ~depth =
              match limits.Ir.Backend_intf.max_workgroup_memory_bytes with
              | Some cap when bk > 0 ->
                  let bytes =
                    ((bm * bk * Ir.Ops.prec_in_bytes a_prec)
                    + (bk * bn * Ir.Ops.prec_in_bytes b_prec))
                    * depth
                  in
                  if bytes > cap then
                    Some
                      (Printf.sprintf
                         "staged operand tiles need %d bytes of workgroup memory%s, exceeding the \
                          %d-byte limit"
                         bytes
                         (if depth > 1 then Printf.sprintf " at pipeline depth %d" depth else "")
                         cap)
                  else None
              | _ -> None
            in
            subt (fun () -> choice "geometry"
                 (List.map
                    [ (16, w, 0); (32, w, 0); (16, w, 32); (32, w, 32); (32, w, 16) ]
                    ~f:(fun (bm, bn, bk) ->
                      ( Printf.sprintf "bm%d bn%d bk%d" bm bn bk,
                        refute_unless
                          ([ nmul "bm" bm ~of_:"m" tm_t; nmul "bn" bn ~of_:"n" tn_t ]
                          @ (match staged_tiles_exceed ~bm ~bn ~bk ~depth:1 with
                            | Some w -> [ (false, w) ]
                            | None -> [])
                          @
                          if bk = 0 then
                            [
                              ndiv "bm" bm ~into:"m" site.m_ni;
                              ndiv "bn" bn ~into:"n" site.m_nj;
                              ( site.m_nk % tk_t = 0,
                                Printf.sprintf
                                  "unstaged full-K block: k=%d is not a multiple of the intrinsic \
                                   tile k=%d"
                                  site.m_nk tk_t );
                            ]
                          else [ nmul "bk" bk ~of_:"k" tk_t ])
                          (fun () ->
                            let base =
                              { base_params with sk_gpu = true; sk_mma = true; sk_simd = w; sk_bm = bm; sk_bn = bn; sk_bk = bk }
                            in
                            (* The twins level (per staged geometry): the swizzled layout and the
                               pipelined depths, each measured against the shared plain sibling —
                               see the field docs on [sk_swizzle]/[sk_depth]. Unstaged geometries
                               have no cooperative copy, so the twin choices do not arise at all;
                               ineligible staged twins are refuted (emission constraint) or
                               excluded (measured-cost policy) with their witnesses (gh-ocannl-481
                               item 3 D3; Codex P2 on PRs #303 and #317). *)
                            let swizzle_twins =
                              match staged_layout with
                              | None -> []
                              | Some LL.Swizzle_elem -> []
                              | Some LL.Swizzle_b128 when bk = 0 -> []
                              | Some LL.Swizzle_b128 ->
                                  [
                                    ( "swizzled",
                                      if b128_units_ok a_prec bk && b128_units_ok b_prec bn then
                                        leaf { base with sk_swizzle = Some LL.Swizzle_b128 }
                                      else
                                        Sspace.Refuted
                                          "Swizzle_b128 permutes whole 16-byte units: each staged \
                                           tile's minor extent must span a power-of-two count > 1 \
                                           of them" );
                                  ]
                            in
                            let depth_twins =
                              if List.is_empty mma.Ir.Backend_intf.mma_pipeline_depths || bk = 0
                              then []
                              else
                                List.map mma.Ir.Backend_intf.mma_pipeline_depths ~f:(fun d ->
                                    ( Printf.sprintf "depth%d" d,
                                      if d < 1 || d > 2 then
                                        (* The capability list is advisory; the implemented range
                                           is [Schedule.apply_stage]'s — the wait-all emission has
                                           single-step lookahead, deeper pipelines need
                                           commit_group/wait_group N. *)
                                        Sspace.Refuted
                                          (Printf.sprintf
                                             "pipeline depth %d is outside the implemented range \
                                              1..2 (Schedule.apply_stage)"
                                             d)
                                      else
                                      match staged_tiles_exceed ~bm ~bn ~bk ~depth:d with
                                      | Some w ->
                                          (* Legality beats policy: the multiplied footprint
                                             refutes before the measured-cost exclusions apply. *)
                                          Sspace.Refuted w
                                      | None ->
                                      if
                                        not
                                          (divides bm site.m_ni && divides bn site.m_nj
                                          && divides bk site.m_nk)
                                      then
                                        Sspace.Excluded
                                          "pad-masked site: Tensorize stays on the \
                                           barrier-bracketed arm, so the twin could only pay the \
                                           doubled shared-memory footprint (Codex P2 on PR #303)"
                                      else if
                                        Ir.Ops.prec_in_bytes a_prec < 4
                                        || Ir.Ops.prec_in_bytes b_prec < 4
                                      then
                                        Sspace.Excluded
                                          "below the async arms' 4-byte element floor: only the \
                                           synchronous form would render — the occupancy cost \
                                           phase 1 measured, with no overlap to buy back (Codex \
                                           P2 on PR #317)"
                                      else leaf { base with sk_depth = d } ))
                            in
                            subt (fun () -> choice "twin"
                                 ((("plain", leaf base) :: swizzle_twins) @ depth_twins)) )
                      )))))
    | true, None -> Sspace.Refuted "backend advertises no mma capability"
    | _ when is_cpu ->
        (* The register-tiled [Tile_mma] rendering needs no MMA units (cc's [limits.mma] is a token
           1x1x1 capability). Statement rules the renderer checks per emission
           ([C_syntax.try_register_tile]) that are already decidable here judge the branch
           (gh-ocannl-479): a candidate that statically must render the scalar fallback refutes
           the family's tensorized promise — it would otherwise be timed, and possibly crowned and
           cached, under a tensorized label, making "the tensorized candidate lost"
           indistinguishable from "it never ran tensorized". Statically decidable: operand-precision
           uniformity (f32/f64 only), the fused accumulation form, the micro-kernel column extent
           vs. the vector lane count, and transposed-B storage for renderings that read B {e in
           place} (a packing Stage normalizes the layout, so the packed flavors are exempt). What is
           only knowable at emission (address spaces, footprint interactions with other locals) is
           covered by the decline diagnostics and the [C_syntax.mma_census]. *)
        let prec = Lazy.force site.m_d.Ir.Tnode.storage_prec in
        let uniform_f32_64 =
          (match prec with Ir.Ops.Single_prec _ | Ir.Ops.Double_prec _ -> true | _ -> false)
          && Ir.Ops.equal_prec (Lazy.force site.m_a.Ir.Tnode.storage_prec) prec
          && Ir.Ops.equal_prec (Lazy.force site.m_b.Ir.Tnode.storage_prec) prec
        in
        let lanes = limits.Ir.Backend_intf.simd_vector_bytes / max 1 (Ir.Ops.prec_in_bytes prec) in
        let tb_in_place = Option.value site.m_tb ~default:false in
        refute_unless
          [
            ( limits.Ir.Backend_intf.simd_vector_bytes >= 8,
              Printf.sprintf "no usable SIMD vector file (simd_vector_bytes=%d < 8)"
                limits.Ir.Backend_intf.simd_vector_bytes );
            ( lanes >= 2,
              Printf.sprintf "fewer than two vector lanes at %s (simd_vector_bytes=%d)"
                (Ir.Ops.prec_string prec) limits.Ir.Backend_intf.simd_vector_bytes );
            ( uniform_f32_64,
              "register tiling requires uniform f32/f64 operand and accumulator precisions" );
            (site.m_fma, "register tiling requires the fused accumulation form");
            ( not (Utils.debug_log_from_routines ()),
              "routine logging is active (debug_log_from_routines): [C_syntax.try_register_tile] \
               deterministically declines, so every leaf would time the scalar fallback under a \
               tensorized label" );
          ]
          (fun () ->
            let whole () =
              (* Whole-triple [Tile_mma] reads both operands in place over the full column extent:
                 the stored B orientation and [n = m_nj] reach the renderer as-is. *)
              choice "row-block"
                (List.map [ 0; 64; 16 ] ~f:(fun bm ->
                     ( Printf.sprintf "bm%d" bm,
                       refute_unless
                         ([
                            ( bm = 0 || divides bm site.m_ni,
                              Printf.sprintf "bm=%d does not divide m=%d" bm site.m_ni );
                          ]
                         @
                         (* [bm > 0] splits the rows into pool-rendered Grid blocks. *)
                         match (bm > 0, Lazy.force cpu_grid_rendering_disabled) with
                         | true, Some w -> [ (false, w) ]
                         | _ -> [])
                         (fun () -> leaf { base_params with sk_mma = true; sk_bm = bm }) )))
            in
            let whole_child =
              refute_unless
                [
                  ( not tb_in_place,
                    "stored B is transposed: whole-triple reads B in place, which the register \
                     tiling statically declines" );
                  ( site.m_nj >= lanes,
                    Printf.sprintf "column extent n=%d is below one vector of lanes (%d)"
                      site.m_nj lanes );
                ]
                (fun () ->
                  match site.m_tb with
                  | None ->
                      (* Per [m_tb]'s own contract, [None] means no role symbol occupies B's minor
                         axis — in-place reads inherit that layout and Tensorize's role check
                         rejects every one of them at candidate compile, deterministically. The
                         packed forms stay available: a packing Stage normalizes the layout. *)
                      Sspace.Refuted
                        "stored B has no role symbol on its minor axis: whole-triple reads B in \
                         place and cannot satisfy Tensorize's role check"
                  | Some _ -> subt (fun () -> whole ()))
            in
            (* Cache-blocked packed composition ([cpu_mma_pack_sketch_schedule]; [bk > 0] selects
               it): [bn = 0] = unsplit column panel. The packed tiles are function-scope stack
               arrays, so their combined footprint is capped — which is also roughly the L2
               residency the blocking aims for. Non-multiple extents no longer gate the packed
               composition (gh-ocannl-485): both operands pack through zero-fringe tiles, so the
               builder pads the axes to the block sizes and [Tensorize] masks the edges. Shapes
               that read an operand in place cannot absorb a pad, so each geometry carries whether
               the extents divide outright ([full_div]) and those shapes refute non-dividing
               geometries. *)
            let packed () =
              let prec_bytes = Ir.Ops.prec_in_bytes (Lazy.force site.m_a.Ir.Tnode.storage_prec) in
              let tile_bytes_cap = 256 * 1024 in
              let menu =
                List.map
                  [ (64, 0, 64); (64, 0, 256); (128, 128, 128); (64, 128, 256); (16, 0, 16) ]
                  ~f:(fun (bm, bn, bk) ->
                    let bn_eff = if bn = 0 then site.m_nj else bn in
                    let tiles_bytes = ((bm * bk) + (bk * bn_eff)) * prec_bytes in
                    let verdict =
                      (* The packed micro-kernel's column extent is the B~ panel width — a
                         legality floor. The footprint threshold is search economy (roughly the
                         L2 residency the blocking aims for — an oversized tile still compiles,
                         and a hoisted panel is not even a stack array), so it excludes rather
                         than refutes: a driver may lift it. *)
                      if bn_eff < lanes then
                        Some
                          (`Refute
                            (Printf.sprintf
                               "B~ panel width %d is below one vector of lanes (%d)" bn_eff lanes))
                      else if tiles_bytes > tile_bytes_cap then
                        Some
                          (`Exclude
                            (Printf.sprintf
                               "packed tiles (%d bytes) exceed the %d-byte stack/cache-economy \
                                threshold (heuristic, not a compiler limit)"
                               tiles_bytes tile_bytes_cap))
                      else None
                    in
                    let full_div =
                      divides bm site.m_ni
                      && (bn = 0 || divides bn site.m_nj)
                      && divides bk site.m_nk
                    in
                    ( Printf.sprintf "bm%d bn%d bk%d" bm bn bk,
                      { base_params with sk_mma = true; sk_bm = bm; sk_bn = bn; sk_bk = bk },
                      verdict,
                      full_div,
                      tiles_bytes ))
              in
              let grid_ok p = blocks_of site.m_ni p.sk_bm >= 2 in
              let too_few_blocks p =
                Printf.sprintf "bm=%d gives %d row block(s); a Grid split needs at least 2"
                  p.sk_bm (blocks_of site.m_ni p.sk_bm)
              in
              (* Per-chunk privatized-tile floor for the Grid shapes ([C_syntax]'s
                 [per_chunk_private_bytes_cap], config [cc_grid_private_bytes_cap]): a known
                 in-kernel tile exceeding the cap makes [parallel_grid_safe] decline the Grid
                 rendering — the candidate would run serially under a Grid label. Other per-chunk
                 locals can still trip the cap at render; passing here is necessary, not
                 sufficient (the census and decline diagnostics cover the rest). *)
              let chunk_cap =
                match
                  Int.of_string
                    (String.strip
                       (Utils.get_global_arg ~arg_name:"cc_grid_private_bytes_cap"
                          ~default:"262144"))
                with
                | c when c > 0 -> c
                | _ -> 256 * 1024
                | exception _ -> 256 * 1024
              in
              let over_chunk_cap ~what bytes =
                if bytes > chunk_cap then
                  Some
                    (Printf.sprintf
                       "%s (%d bytes) exceeds the per-chunk privatization cap (%d, config \
                        cc_grid_private_bytes_cap)"
                       what bytes chunk_cap)
                else None
              in
              (* One packing shape's geometries: the shared menu judged per shape. The shape level
                 sits ABOVE the geometries, matching the flat enumeration's variant-major emission
                 order. *)
              let geoms ~f =
                choice "geometry"
                  (List.map menu ~f:(fun (label, p, verdict, full_div, tiles_bytes) ->
                       ( label,
                         match verdict with
                         | Some (`Refute w) -> Sspace.Refuted w
                         | Some (`Exclude w) -> Sspace.Excluded w
                         | None -> f p full_div tiles_bytes )))
              in
              let any_hoistable = hoistable site.m_a || hoistable site.m_b in
              let no_constant = "no host-init-backed constant operand to pack at link time" in
              (* See the flat enumeration's rationale, now attached to the shapes it judges:
                 hoisted packing (gh-ocannl-470) needs a constant operand; the hoisted-only Grid
                 shape reads non-hoistable operands in place (no pad absorption, and a transposed
                 non-hoistable B statically declines the register tiling); the mixed
                 grid-outermost shape (gh-ocannl-473) exists exactly when one operand is hoistable
                 and the other is not; grid-outermost per-chunk re-packing (gh-ocannl-475) is
                 proposed only where no hoistable operand leaves a one-dispatch alternative, and
                 its tiles must fit the renderer's per-chunk privatization cap. Grid shapes need
                 at least two row blocks (c_syntax.ml [collect_parallel_grid]). *)
              choice "packing-shape"
                [
                  ("serial", subt (fun () -> geoms ~f:(fun p _ _ -> leaf p)));
                  ( "hoisted",
                    if any_hoistable then subt (fun () -> geoms ~f:(fun p _ _ -> leaf { p with sk_hoist = true }))
                    else Sspace.Refuted no_constant );
                  ( "hoisted-grid",
                    if not any_hoistable then Sspace.Refuted no_constant
                    else if Option.is_some (Lazy.force cpu_grid_rendering_disabled) then
                      Sspace.Refuted
                        (Option.value_exn (Lazy.force cpu_grid_rendering_disabled))
                    else if
                      (* A non-hoistable B is read in place by this shape (its stage is omitted),
                         so only the clean untransposed orientation survives: transposed B
                         statically declines the register tiling, and [m_tb = None] means no role
                         symbol occupies B's minor axis — Tensorize's role check rejects every
                         in-place read deterministically. *)
                      (not (hoistable site.m_b))
                      && not (Option.value_map site.m_tb ~default:false ~f:not)
                    then
                      Sspace.Refuted
                        (match site.m_tb with
                        | Some true ->
                            "non-hoistable transposed B would be read in place, which the \
                             register tiling statically declines"
                        | _ ->
                            "stored B has no role symbol on its minor axis: the hoisted-grid \
                             shape reads non-hoistable B in place and cannot satisfy Tensorize's \
                             role check")
                    else
                      subt (fun () -> geoms ~f:(fun p full_div _ ->
                             if not (full_div || (hoistable site.m_a && hoistable site.m_b)) then
                               Sspace.Refuted
                                 "extents do not divide the blocks: the non-hoistable operand is \
                                  read in place and cannot absorb the zero-fringe pad"
                             else if not (grid_ok p) then Sspace.Refuted (too_few_blocks p)
                             else leaf { p with sk_hoist = true; sk_grid = true })) );
                  ( "hoisted-grid-pack-rest",
                    if not any_hoistable then Sspace.Refuted no_constant
                    else if Option.is_some (Lazy.force cpu_grid_rendering_disabled) then
                      Sspace.Refuted
                        (Option.value_exn (Lazy.force cpu_grid_rendering_disabled))
                    else if hoistable site.m_a && hoistable site.m_b then
                      Sspace.Excluded
                        "both operands are hoistable: nothing is left to pack in-kernel, the \
                         shape degenerates to hoisted-grid"
                    else
                      subt (fun () -> geoms ~f:(fun p _ _ ->
                             let bn_eff = if p.sk_bn = 0 then site.m_nj else p.sk_bn in
                             (* The non-hoistable operand's in-kernel packing Stage lands inside
                                the Grid body and privatizes per chunk. *)
                             let rest_tile =
                               if hoistable site.m_a then p.sk_bk * bn_eff * prec_bytes
                               else p.sk_bm * p.sk_bk * prec_bytes
                             in
                             if not (grid_ok p) then Sspace.Refuted (too_few_blocks p)
                             else
                               match over_chunk_cap ~what:"per-chunk packed tile" rest_tile with
                               | Some w -> Sspace.Refuted w
                               | None ->
                                   leaf { p with sk_hoist = true; sk_grid = true; sk_pack_rest = true }))
                  );
                  ( "grid-pack-rest",
                    if Option.is_some (Lazy.force cpu_grid_rendering_disabled) then
                      Sspace.Refuted
                        (Option.value_exn (Lazy.force cpu_grid_rendering_disabled))
                    else if any_hoistable then
                      Sspace.Excluded
                        "a hoistable operand exists: the hoisted shapes cover the one-dispatch \
                         role without per-chunk re-packing"
                    else
                      subt (fun () -> geoms ~f:(fun p _ tiles_bytes ->
                             if not (grid_ok p) then Sspace.Refuted (too_few_blocks p)
                             else
                               match over_chunk_cap ~what:"per-chunk packed tiles" tiles_bytes with
                               | Some w -> Sspace.Refuted w
                               | None -> leaf { p with sk_grid = true; sk_pack_rest = true })) );
                  ( "grid",
                    match Lazy.force cpu_grid_rendering_disabled with
                    | Some w -> Sspace.Refuted w
                    | None ->
                    subt (fun () -> geoms ~f:(fun p _ _ ->
                           (* The per-row-block A~ tile privatizes per chunk; the read-only B~
                              panel is shared and does not count against the cap. *)
                           let a_tile = p.sk_bm * p.sk_bk * prec_bytes in
                           if not (grid_ok p) then Sspace.Refuted (too_few_blocks p)
                           else
                             match over_chunk_cap ~what:"per-chunk privatized A~ tile" a_tile with
                             | Some w -> Sspace.Refuted w
                             | None -> leaf { p with sk_grid = true })) );
                ]
            in
            subt (fun () -> choice "tensorized-form"
                 [ ("whole-triple", whole_child); ("packed", subt (fun () -> packed ())) ]))
    | _ -> Sspace.Refuted "backend kind seeds no tensorized pipeline"
  in
  choice "pipeline" [ ("blocktile", blocktile_child); ("tensorized", mma_child) ]

let matmul_seed_params ~is_gpu ~is_cpu ~(limits : Ir.Backend_intf.hardware_limits) site :
    sketch_params list =
  Sspace.leaves (matmul_family_tree ~is_gpu ~is_cpu ~limits site)

let sketch_seed_params ~is_gpu ~is_cpu ~(limits : Ir.Backend_intf.hardware_limits)
    (opt : LL.optimized) : sketch_params list =
  let seeds, fuse_target =
    match detect_matmul opt.LL.llc with
    | None -> (
        match conv_seed_params ~is_gpu ~is_cpu ~limits opt with
        | Some (seeds, d) -> (seeds, Some d)
        | None -> ([], None))
    | Some site -> (matmul_seed_params ~is_gpu ~is_cpu ~limits site, Some site.m_d)
  in
  match (seeds, fuse_target) with
  | [], _ -> []
  | seeds, Some d when Sched.can_fuse_epilogue ~target:d opt ->
      (* Fused-epilogue variants (gh-ocannl-486): when the site's output feeds an eligible
         elementwise tail, every seed gets a fused twin — the tuner measures fused (one kernel) vs.
         unfused (the fissioned two-kernel form). The check runs on the base code where the plain
         accumulation-nest fusion site applies; seeds whose scheduled form no longer admits the
         fusion fail their candidate compile and are skipped. *)
      seeds @ List.map seeds ~f:(fun p -> { p with sk_epilogue = true })
  | seeds, _ -> seeds

(* The exported tree view of the matmul family (site detection included); the conv family and the
   epilogue-twin level factor the same way as mechanical follow-ups. *)
let matmul_sketch_tree ~is_gpu ~is_cpu ~(limits : Ir.Backend_intf.hardware_limits)
    (opt : LL.optimized) : sketch_params Sspace.tree option =
  Option.map (detect_matmul opt.LL.llc) ~f:(matmul_family_tree ~is_gpu ~is_cpu ~limits)

(** {2 The privatized fission flavor}

    A variant of the per-segment preset that contracts each materialized read-modify-write
    accumulator into a per-thread register tile ({!Sched.optop.Privatize}) over its serial reduction
    loop. A routine-local accumulator beats a device-memory RMW on every backend, and on Metal it
    additionally sidesteps the volatile-RMW miscompile workaround tax (c_syntax.ml
    [volatile_scalar_rmw]). Detection is permissive: each proposal is validated by try-applying
    against the segment (Privatize's own preconditions — single index vector, uniform
    iteration-invariant guards, etc.), and dropped rather than failing the candidate. *)

let rec subtree_has_hardware_loop (llc : LL.t) =
  match llc with
  | LL.For_loop { axis = LL.Grid | LL.Workgroup | LL.Workgroup_reduce; _ } -> true
  | LL.For_loop { body; _ } -> subtree_has_hardware_loop body
  | LL.Seq (a, b) -> subtree_has_hardware_loop a || subtree_has_hardware_loop b
  | LL.If { body; _ } -> subtree_has_hardware_loop body
  | _ -> false

(* Materialized RMW accumulation sites of the (post-preset) scheduled segment, each paired with the
   outermost enclosing Serial loop eligible to privatize over: the access vector must not mention
   its symbol (so the accumulation is carried across it), and no hardware-typed loop may sit inside
   its subtree (the private tile is per-thread; spanning other threads' iterations would store back
   their elements). *)
let privatize_proposals (post : LL.optimized) : (Ir.Tnode.t * Idx.symbol) list =
  let plc = post.LL.optimize_ctx.LL.placements in
  let proposals = ref [] in
  let rec walk stack (llc : LL.t) =
    match llc with
    | LL.Seq (a, b) ->
        walk stack a;
        walk stack b
    | LL.If { body; _ } -> walk stack body
    | LL.For_loop { index; from_; body; axis; _ } -> walk ((index, from_, axis, body) :: stack) body
    | LL.Set { tn; idcs; llsc; _ }
      when Ir.Tnode.Placements.is_materialized_peek plc tn
           && List.exists (collect_gets llsc) ~f:(fun (t, i) ->
               phys_equal t tn && Array.equal Idx.equal_axis_index i idcs) ->
        List.find (List.rev stack) ~f:(fun (index, from_, axis, body) ->
            LL.equal_axis_type axis LL.Serial && from_ = 0
            && (not (idcs_mention idcs index))
            && not (subtree_has_hardware_loop body))
        |> Option.iter ~f:(fun (index, _, _, _) ->
            if
              not
                (List.exists !proposals ~f:(fun (t, s) ->
                     Ir.Tnode.equal t tn && Idx.equal_symbol s index))
            then proposals := (tn, index) :: !proposals)
    | _ -> ()
  in
  walk [] post.LL.llc;
  List.rev !proposals

(** The preset schedule extended with a [Privatize] per detected accumulator. Proposals are detected
    on the preset-scheduled segment and validated one at a time by re-applying the growing schedule;
    a proposal violating an op precondition is dropped. The exploratory applies run against a
    hermetic copy of the segment: [Privatize] registers its (fresh) tile in the traced store and
    placements, and abandoned tiles would otherwise be emitted as dead local declarations when the
    caller applies the returned schedule to the real segment. *)
let extend_with_privatize ~static_indices sched (seg : LL.optimized) : Sched.schedule =
  let scratch () =
    {
      seg with
      LL.traced_store = Hashtbl.copy seg.LL.traced_store;
      LL.optimize_ctx = LL.copy_optimize_ctx seg.LL.optimize_ctx;
    }
  in
  match Sched.apply_classified ~static_indices sched (scratch ()) with
  | exception Outcome.Cause_at _ -> sched
  | post ->
      List.fold (privatize_proposals post) ~init:sched ~f:(fun acc (target, over) ->
          let acc' = acc @ [ Sched.Privatize { target; over } ] in
          match Sched.apply_classified ~static_indices acc' (scratch ()) with
          | (_ : LL.optimized) -> acc'
          | exception Outcome.Cause_at _ -> acc)

(** {2 Split-reduce site detection (gh-ocannl-484 task 3)}

    Reduction-dominated sites: an accumulation whose target has few cells (little output
    parallelism) fed by a long serial reduction loop — the bias/weight-gradient reductions of the
    conv benchmarks, softmax denominators, and skinny (split-K) GEMMs alike. The gh-476 sweep
    attribution: on both Metal and CUDA one such fission segment is 60-95% of the default conv
    training step, and the tuner had no move into the split-reduction region of the schedule space —
    [Sched.Split_reduce] existed but nothing seeded it. Detection is deliberately cheap and
    over-approximate: any rmw [Set] (or gh-466 [Set_dynamic] scatter) qualifies structurally, and
    each candidate axis is settled by the hermetic {!Sched.op_legality} probe — the op's own
    recognizer decides the static-form pinning discipline, never a re-implementation here.

    {3 The enabling interchange (gh-ocannl-537)}

    A bare [Split_reduce] reaches none of the conv-gradient accumulations it was filed for: OCANNL
    lowers them with the accumulated channel loop {e innermost} and the reduction loops (batch, y, x)
    outside it, so every axis is rejected for "the accumulation cell mentions a symbol not bound by a
    loop enclosing the reduction loop" — measured on HIP lenet, where that one segment is 89% of the
    step. That cause, and only that cause, a loop interchange removes. So a rejected candidate is
    re-probed after hoisting exactly the symbols {!Sched.split_reduce_hoist} names, each bubbled
    outside the reduction loop by adjacent [Swap]s (relative order preserved); the site records the
    chain and the [F_split] prelude replays it before the split. Every [Swap] is confirmed
    [Op_legal] on the code it is applied to — [Swap]'s reassociation license covers the accumulation
    it reorders, but it is checked per site, not assumed — and the [Split_reduce] is re-probed on the
    interchanged code, so a returned site is still seedable exactly as proposed. *)

type sr_site = {
  sr_axis : Idx.symbol;  (** The reduction loop to split: the largest-extent legal candidate. *)
  sr_target : Ir.Tnode.t;  (** The accumulated node. *)
  sr_red : int;  (** The [sr_axis] loop's extent. *)
  sr_out : int;  (** The target's cell count — the site's whole output parallelism. *)
  sr_cost : int;
      (** Estimated segment cost: the accumulation statement's trip count — the product of every
          enclosing loop extent, i.e. how many accumulate steps the serial nest spends on this
          site. Ranks the sites (gh-ocannl-541): the earlier [sr_red / sr_out] integer-division
          ratio sent every large-output site to 0, silently excluding the very sites (conv weight
          gradients) with the most serial work to recover. *)
  sr_dynamic : bool;  (** The gh-466 scatter form ([Set_dynamic]). *)
  sr_swaps : (Idx.symbol * Idx.symbol) list;
      (** The enabling interchange (gh-ocannl-537), as [(outer, inner)] pairs applied {e in order}
          before the [Split_reduce]: each hoists an accumulation-cell loop outside [sr_axis]. Empty
          when the site is splittable as lowered. *)
}

(* Sites with more output cells than this have enough output parallelism that the default presets
   already fill a device; splitting the reduction would only add combine traffic. *)
let sr_out_max = 4096

(* Reduction extents below this are not worth a second kernel pass (the combine reads
   [num_blocks] partial cells per output cell). *)
let sr_red_min = 64

(* The adjacent-interchange chain hoisting [needed] outside [axis] within the write's loop [path]
   (outermost first), or [None] when some symbol is not a loop of that path — e.g. a static index —
   and hence not hoistable. Each symbol is bubbled up one loop at a time until it encloses [axis];
   taking them in path order leaves their relative order intact, so the resulting enclosing prefix
   iterates the accumulation cell exactly as the original nest did. *)
let sr_hoist_swaps ~path ~axis ~needed : (Idx.symbol * Idx.symbol) list option =
  let pos order s = List.findi order ~f:(fun _ x -> Idx.equal_symbol x s) |> Option.map ~f:fst in
  match
    (pos path axis, List.map needed ~f:(fun s -> Option.map (pos path s) ~f:(fun i -> (i, s))))
  with
  | None, _ -> None
  | Some _, indexed -> (
      match Option.all indexed with
      | None -> None
      | Some indexed ->
          let ordered =
            List.sort indexed ~compare:(fun (a, _) (b, _) -> Int.compare a b) |> List.map ~f:snd
          in
          let order = ref path and swaps = ref [] in
          List.iter ordered ~f:(fun h ->
              let continue_ = ref true in
              while !continue_ do
                (* Both are in [order] by construction and interchange only permutes it. *)
                let ih = Option.value_exn (pos !order h) in
                let ia = Option.value_exn (pos !order axis) in
                if ih <= ia then continue_ := false
                else
                  let parent = List.nth_exn !order (ih - 1) in
                  swaps := (parent, h) :: !swaps;
                  order :=
                    List.mapi !order ~f:(fun i x ->
                        if i = ih - 1 then h else if i = ih then parent else x)
              done);
          Some (List.rev !swaps))

let split_reduce_sites ?(static_indices = []) (opt : LL.optimized) : sr_site list =
  let acc = ref [] in
  let hermetic (o : LL.optimized) =
    {
      o with
      LL.traced_store = Hashtbl.copy o.LL.traced_store;
      LL.optimize_ctx = LL.copy_optimize_ctx o.LL.optimize_ctx;
    }
  in
  (* The interchanged code, once every [Swap] of the chain is confirmed [Op_legal] against the code
     it is applied to ({!Sched.schedule_legality} walks the chain exactly as application will —
     [Swap]'s reassociation license covers accumulations, but each site is checked, not assumed).
     Anything short of all-legal drops the site. *)
  let apply_swaps swaps =
    let ops = List.map swaps ~f:(fun (outer, inner) -> Sched.Swap { outer; inner }) in
    let verdicts = Sched.schedule_legality opt ops in
    if
      List.length verdicts <> List.length ops
      || not
           (List.for_all verdicts ~f:(fun (_, v) -> Sched.equal_op_verdict v Sched.Op_legal))
    then None
    else
      match Sched.apply ~static_indices ops (hermetic opt) with
      | opt' -> Some opt'
      | exception Invalid_argument _ -> None
  in
  let splittable o ~axis ~tn =
    let op, _, _, _ = Sched.split_reduce ~axis ~target:tn ~num_blocks:2 in
    match Sched.op_legality o op with
    | Sched.Op_legal -> `Legal
    | Sched.Op_illegal _ | Sched.Op_unknown _ -> (
        (* The one rejection an interchange removes; empty for every other cause. *)
        match Sched.split_reduce_hoist o op with [] -> `No | needed -> `Hoist needed)
  in
  let consider ~enclosing ~tn ~idcs ~dynamic =
    let out = try Ir.Tnode.num_elems tn with _ -> 0 in
    if out >= 1 && out <= sr_out_max then
      let path = List.map enclosing ~f:(fun (s, _, _) -> s) in
      let candidates =
        List.filter enclosing ~f:(fun (s, n, ty) ->
            LL.equal_axis_type ty LL.Serial && n >= sr_red_min && not (idcs_mention idcs s))
        (* Largest extent first: the probe stops at the first legal candidate, and loops enclosing
           an inner reduction loop fail the pinning discipline anyway (an enclosing reduction loop
           pins no component), so outer/larger candidates dominate. *)
        |> List.sort ~compare:(fun (_, a, _) (_, b, _) -> Int.compare b a)
      in
      let legal =
        List.find_map candidates ~f:(fun (s, n, _) ->
            match splittable opt ~axis:s ~tn with
            | `Legal -> Some (s, n, [])
            | `No -> None
            | `Hoist needed -> (
                (* gh-537: hoist and re-probe. Both the interchange and the split are settled on the
                   code they act on, so the recorded chain is replayable as recorded. *)
                match sr_hoist_swaps ~path ~axis:s ~needed with
                | None -> None
                | Some swaps -> (
                    match apply_swaps swaps with
                    | None -> None
                    | Some swapped -> (
                        match splittable swapped ~axis:s ~tn with
                        | `Legal -> Some (s, n, swaps)
                        | `No | `Hoist _ -> None))))
      in
      Option.iter legal ~f:(fun (s, n, swaps) ->
          if not (List.exists !acc ~f:(fun site -> Idx.equal_symbol site.sr_axis s)) then
            acc :=
              {
                sr_axis = s;
                sr_target = tn;
                sr_red = n;
                sr_out = out;
                sr_cost = List.fold enclosing ~init:1 ~f:(fun c (_, n, _) -> c * max 1 n);
                sr_dynamic = dynamic;
                sr_swaps = swaps;
              }
              :: !acc)
  in
  let rec walk enclosing (llc : LL.t) =
    match llc with
    | LL.Seq (a, b) ->
        walk enclosing a;
        walk enclosing b
    | LL.If { body; _ } -> walk enclosing body
    | LL.For_loop { index; from_; to_; body; axis; _ } ->
        walk (enclosing @ [ (index, to_ - from_ + 1, axis) ]) body
    | LL.Set { tn; idcs; llsc; _ } ->
        (* rmw accumulation: the value re-reads the written node ([op_legality] then enforces the
           exact same-cell and operator discipline). *)
        if List.exists (collect_gets llsc) ~f:(fun (t, _) -> Ir.Tnode.equal t tn) then
          consider ~enclosing ~tn ~idcs ~dynamic:false
    | LL.Set_dynamic { tn; idcs; _ } -> consider ~enclosing ~tn ~idcs ~dynamic:true
    | _ -> ()
  in
  walk [] opt.LL.llc;
  (* Estimated segment cost, descending — the site with the most serial work to recover ranks
     first (gh-ocannl-541). Stable, so equal-cost sites keep detection (program) order. The
     candidate-volume cap is NOT applied here: it belongs to the search ([tune]'s
     [max_split_reduce_sites]), which records the sites it evicts in the decline census. *)
  List.stable_sort (List.rev !acc) ~compare:(fun a b -> Int.compare b.sr_cost a.sr_cost)

(** {2 Analytic cost-model scoring (gh-ocannl-491, the selection half)}

    The extraction half lives in {!Ir.Cost_model}; here it is consumed for ranking candidate
    schedules — the beam pre-filter of {!tune} and the untuned-default selection of
    {!model_default}. The model is advisory throughout: a candidate class without model coverage
    (opaque code, a schedule the model cannot apply, missing envelope constants) is never dropped,
    only measured — consistent with never overriding a measured result, and keeping the search
    independent of enumeration order. *)

module CM = Ir.Cost_model

let scratch_of (opt : LL.optimized) =
  {
    opt with
    LL.traced_store = Hashtbl.copy opt.LL.traced_store;
    LL.optimize_ctx = LL.copy_optimize_ctx opt.LL.optimize_ctx;
  }

(* Per-machine calibrated envelope constants from the config beat the backend's class-level advisory
   constants ([Backend_intf.hardware_limits]'s [peak_flops] / [peak_memory_bandwidth]) — fitting
   them from [autotune_calibration_file] data is the intended workflow. *)
let peak_override ~arg_name =
  lazy
    (let s = String.strip (Utils.get_global_arg ~arg_name ~default:"") in
     if String.is_empty s then None
     else
       match Float.of_string s with
       | f when Float.(f > 0.) -> Some f
       | _ -> None
       | exception _ -> None)

let peak_flops_override = peak_override ~arg_name:"model_peak_flops"
let peak_bandwidth_override = peak_override ~arg_name:"model_peak_memory_bandwidth"

let envelope ~(limits : Ir.Backend_intf.hardware_limits) =
  ( Option.first_some (Lazy.force peak_flops_override) limits.Ir.Backend_intf.peak_flops,
    Option.first_some
      (Lazy.force peak_bandwidth_override)
      limits.Ir.Backend_intf.peak_memory_bandwidth )

(* The roofline lower bound summed over a candidate's kernels; [None] — no model coverage — when any
   kernel is opaque (its counts may UNDER-estimate, so ranking on them could prune the true winner)
   or when no envelope constant is present. The kernels run sequentially, so the bound is per-kernel
   max-of-legs, summed — aggregating flops/bytes first and applying the roofline once would
   under-price a compute-bound + bandwidth-bound mix to roughly its larger leg. *)
let summaries_roofline ~peak_flops ~peak_memory_bandwidth (summaries : CM.summary list) :
    float option =
  if List.exists summaries ~f:(fun s -> s.CM.opaque) then None
  else
    (* [roofline_seconds] is [None] exactly when no envelope constant is given, uniformly across the
       folds — the [~flops:0 ~bytes:0] seed keeps that contract for the empty list. *)
    List.fold summaries
      ~init:(CM.roofline_seconds ?peak_flops ?peak_memory_bandwidth ~flops:0 ~bytes:0 ())
      ~f:(fun acc s ->
        Option.both acc
          (CM.roofline_seconds ?peak_flops ?peak_memory_bandwidth ~flops:s.CM.flops
             ~bytes:(CM.total_bytes s) ())
        |> Option.map ~f:(fun (a, b) -> a +. b))

let model_score ~static_indices ~limits (opt : LL.optimized) (sched : Sched.schedule) : float option
    =
  let peak_flops, peak_memory_bandwidth = envelope ~limits in
  match Sched.apply_classified ~static_indices sched (scratch_of opt) with
  | exception Outcome.Cause_at _ -> None
  | post -> summaries_roofline ~peak_flops ~peak_memory_bandwidth [ CM.analyze post.LL.llc ]

let model_prefilter ~keep_fraction (scored : ('a * float option) list) : ('a * float option) list =
  let scores = List.filter_map scored ~f:snd in
  let n = List.length scores in
  if Float.(keep_fraction >= 1.) || n <= 1 then scored
  else
    let n_keep =
      Int.min n (Int.max 1 (Int.of_float (Float.round_up (keep_fraction *. Float.of_int n))))
    in
    let cutoff = List.nth_exn (List.sort scores ~compare:Float.compare) (n_keep - 1) in
    (* Ties at the cutoff are all kept: which of two equal-scored candidates survives must not
       depend on enumeration order. Unscored candidates ([None]) always pass — the no-coverage
       exemption. *)
    List.filter scored ~f:(fun (_, s) ->
        match s with None -> true | Some v -> Float.(v <= cutoff))

(** {2 Candidate compilation}

    A candidate is a recipe producing schedules against a {e fresh} lowering: backend [compile]
    re-lowers (with fresh symbols) on every call, so schedules are rebound structurally inside the
    transform closure, after checking the fresh code's canonical digest against the base compile's.
    Whole-routine candidates go through the singular [?lowered_transform] seam; fissioned candidates
    through the plural [?lowered_transforms] seam, with per-segment schedules keyed by the
    pre-schedule segment's canonical digest. *)

type whole_flavor =
  | W_saved of SC.saved_schedule
  | W_preset of { block_size : int option }
  | W_sketch of sketch_params

type fiss_flavor =
  | F_preset of {
      block_size : int option;
      privatize : bool;
      config_thresholds : bool;
          (** Use the config-default [min_parallel] thresholds instead of the search's
              [min_parallel:1] — with [block_size = None] this reproduces the untuned default
              pipeline ({!Sched.maybe_default_schedules}) exactly, so the candidate pool always
              contains the behavior the user gets without tuning: on launch-overhead-bound workloads
              the aggressive [min_parallel:1] presets can all lose to it. *)
    }
  | F_saved of (string * SC.saved_schedule) list
  | F_sketch of (string * sketch_params) list
      (** Per-segment matmul sketches: for each listed segment (keyed by its pre-schedule structural
          digest, like [F_saved]), the composed sketch pipeline instantiated with the given
          parameters; every other segment gets the plain default preset — the same pipeline the
          seed-time segment enumeration ran, so the segmentation converges. On a key miss
          (segmentation drift) the candidate degrades to the plain fissioned preset and dedups away
          by digest; unlike [F_saved] it never replays a cache entry, so no loud drift guard is
          needed. *)
  | F_split of { sites : (sr_site * int) list }
      (** Split-reduce seeds (gh-ocannl-484 task 3): per listed site, a
          [Sched.Split_reduce { axis = sr_axis; target = sr_target; num_blocks }] — applied {e
          whole-routine, before fission}, unlike the per-segment flavors: the two passes must
          compile as separate kernels (annotating the block loop with both passes in one kernel
          races — the combine needs grid-wide synchronization), and the partials producer/consumer
          pair is exactly the materialized cross-nest edge fission cuts at. Each resulting segment
          then gets the aggressive default preset — the block loop parallelizes pass 1, the combine
          nest annotates like any small kernel. *)
  | F_split_saved of SC.saved_schedule * (string * SC.saved_schedule) list
      (** Replay of a split-reduce winner: the whole-routine prelude (resolved against the base
          canonical form, re-minting the partials node and fresh symbols via [SC.of_saved]), then
          per-segment saved schedules over the {e post-prelude} segmentation, keyed and
          drift-guarded exactly like [F_saved]. *)

type spec = Whole of whole_flavor | Fiss of fiss_flavor

(* The replayable/cacheable description of a compiled candidate. *)
type form =
  | Whole_saved of SC.saved_schedule
  | Fiss_saved of (string * SC.saved_schedule) list
  | Split_saved of SC.saved_schedule * (string * SC.saved_schedule) list

type unit_gen = {
  u_key : string option;  (** [Some pre_digest] for a fission segment; [None] whole-routine. *)
  u_saved : SC.saved_schedule;
  u_registry : SC.registry;
  u_opt : LL.optimized;  (** The transformed unit, for menu generation. *)
}

type compiled = {
  form : form;
  cctx : Context.t;
  routine : Context.routine;
  units : unit_gen list;
  all_opts : LL.optimized list;
      (** Every compiled segment ([`Zeros] and [`Solo] segments included, unlike [units]) — the code
          the timing runs actually execute, for calibration analysis. *)
  digest_after : string;
  mma_renders : (string * Ir.C_syntax.mma_rendering) list;
      (** The [Ir.C_syntax.mma_census] of this candidate's compile: how each [Tile_mma] statement
          actually rendered (gh-ocannl-479) — a tensorized candidate whose statements all fell back
          to the scalar path never ran tensorized, and the tuning log must say so next to the
          timing. *)
}

(* Per-candidate search diagnostics on stderr, gated by config [autotune_log]. *)
let log_enabled =
  lazy
    (match
       String.lowercase
         (String.strip (Utils.get_global_arg ~arg_name:"autotune_log" ~default:"false"))
     with
    | "true" | "1" -> true
    | _ -> false)

let logf fmt =
  Printf.ksprintf (fun s -> if Lazy.force log_enabled then Stdio.eprintf "autotune: %s\n%!" s) fmt

(* Log tag for a (possibly '+'-concatenated, fissioned) digest: a plain prefix only reflects the
   first segment — two fissioned programs identical in segment 1 would read as "the same digest"
   (misled the CUDA round-4 analysis on PR #140) — so fold the whole string into the tag. *)
let dshort d =
  String.prefix d 8 ^ "/" ^ String.prefix (Stdlib.Digest.to_hex (Stdlib.Digest.string d)) 8

let bs_label = function None -> "cfg" | Some b -> Int.to_string b

(* Calibration output (gh-ocannl-491 task 4) and the bound-agreement invariant (gh-ocannl-514
   phase 0): the model score next to the measured time — every tuning run is free calibration data
   for the envelope constants, and every timed candidate is a test of the roofline bound's
   soundness. Human-readable stderr lines under config [autotune_log]; durable tab-separated rows
   (schema owned by {!CM.Calibration}) appended under config [autotune_calibration_file].

   The analysis runs on the candidate's actual compiled segments ([compiled.all_opts]), so a row
   prices exactly the code that was timed. For an exact-count candidate, a roofline LOWER bound
   exceeding a measured time can only mean the envelope constants understate this machine's
   achievable peaks — a search fathoming on that bound would prune true winners — so the violation
   warns unconditionally, not gated by [autotune_log]: per the gh-ocannl-498 lesson, an invariant
   between a scorer and reality is checked continuously against every sample, never spot-checked.
   Approximate counts ([CM.approximate]: guards-taken / union upper bounds) make an exceedance
   ambiguous — mostly-failing guards over-count without implicating the envelope — so those log as
   diagnostics and their rows are flagged for the fitter to exclude. Refitting the constants from the
   accumulated rows ([CM.Calibration.fit], [tools/fit_envelope.exe]) restores soundness. The
   analysis therefore also runs whenever envelope constants are present, even with logging and the
   calibration file off — one [CM.analyze] per compiled segment, trivial next to the compile and
   timing runs the candidate already paid for. *)
let calibration_file =
  lazy (String.strip (Utils.get_global_arg ~arg_name:"autotune_calibration_file" ~default:""))

let emit_calibration ~backend ~limits ~label ~digest ~measured_ms (opts : LL.optimized list) =
  let file = Lazy.force calibration_file in
  let peak_flops, peak_memory_bandwidth = envelope ~limits in
  let have_envelope = Option.is_some peak_flops || Option.is_some peak_memory_bandwidth in
  if Lazy.force log_enabled || (not (String.is_empty file)) || have_envelope then (
    let summaries = List.map opts ~f:(fun o -> CM.analyze o.LL.llc) in
    let flops = List.sum (module Int) summaries ~f:(fun s -> s.CM.flops) in
    let bytes = List.sum (module Int) summaries ~f:CM.total_bytes in
    let opaque = List.exists summaries ~f:(fun s -> s.CM.opaque) in
    let flops_approx = List.exists summaries ~f:(fun s -> s.CM.flops_approx) in
    let bytes_approx = List.exists summaries ~f:CM.footprint_approximate in
    let model_ms =
      Option.map (summaries_roofline ~peak_flops ~peak_memory_bandwidth summaries) ~f:(fun s ->
          s *. 1e3)
    in
    let dtag = dshort digest in
    (let seconds = Float.max 1e-12 (measured_ms *. 1e-3) in
     (* Per-leg audit: an exact aggregate leg exceeding the measurement indicts the envelope no
        matter what the other leg's counts are (the aggregate leg lower-bounds the per-kernel
        sum). The whole-bound check additionally catches the fully-exact multi-kernel case where
        the per-kernel max-of-legs sum exceeds the measurement without either aggregate leg
        doing so. The implied minima name only legs that are configured AND exact — an absent
        leg cannot have caused the exceedance, and an approximate one is not evidence. *)
     let leg_exceeds exact count peak =
       match peak with
       | Some p -> exact && Float.(Float.of_int count /. seconds > p)
       | None -> false
     in
     let flops_leg = leg_exceeds (not flops_approx) flops peak_flops in
     let bytes_leg = leg_exceeds (not bytes_approx) bytes peak_memory_bandwidth in
     let bound_exceeds =
       match model_ms with Some m -> Float.(m > measured_ms) | None -> false
     in
     if flops_leg || bytes_leg || (bound_exceeds && (not flops_approx) && not bytes_approx) then
       let minima =
         String.concat ~sep:" and "
           (List.filter_opt
              [
                (if Option.is_some peak_flops && not flops_approx then
                   Some (Printf.sprintf "model_peak_flops >= %.6g" (Float.of_int flops /. seconds))
                 else None);
                (if Option.is_some peak_memory_bandwidth && not bytes_approx then
                   Some
                     (Printf.sprintf "model_peak_memory_bandwidth >= %.6g"
                        (Float.of_int bytes /. seconds))
                 else None);
              ])
       in
       Stdio.eprintf
         "autotune: BOUND VIOLATION: roofline lower bound %s ms > measured %.4f ms for %s \
          (digest %s) on %s — the envelope constants understate this machine's peaks (this row \
          implies %s as necessary minima); refit with tools/fit_envelope.exe over \
          autotune_calibration_file data\n\
          %!"
         (match model_ms with Some m -> Printf.sprintf "%.6f" m | None -> "?")
         measured_ms label dtag backend minima
     else if bound_exceeds then
       (* Only an approximate leg can explain the exceedance: possibly over-counting
          (guards-taken / union upper bounds), not the envelope — a diagnostic, no
          unconditional warning, no implied-minima claim. *)
       logf
         "model bound %.6f ms > measured %.4f ms for %s (digest %s), but its counts are \
          approximate upper bounds (guarded/masked code) — possibly over-counting, not the \
          envelope"
         (Option.value_exn model_ms) measured_ms label dtag);
    let n_kernels = List.length summaries in
    logf "calibration: %s measured %.4f ms, model %s, %d kernel%s, flops %d, bytes %d%s" label
      measured_ms
      (match model_ms with Some m -> Printf.sprintf "%.6f ms" m | None -> "n/a")
      n_kernels
      (if n_kernels = 1 then "" else "s")
      flops bytes
      (if opaque then " (opaque: counts may under-estimate)" else "");
    if not (String.is_empty file) then
      let line =
        CM.Calibration.to_line
          {
            CM.Calibration.backend;
            digest = dtag;
            label;
            measured_ms;
            model_ms;
            kernels = n_kernels;
            flops;
            bytes;
            flops_approx;
            bytes_approx;
            opaque;
          }
        ^ "\n"
      in
      try
        Stdio.Out_channel.with_file file ~append:true ~f:(fun oc ->
            Stdio.Out_channel.output_string oc line)
      with _ -> logf "calibration: cannot append to %s" file)

(* Whether the spec's label promises a tensorized pipeline — used to flag "no Tile_mma emitted"
   census anomalies (gh-ocannl-479). *)
let spec_expects_mma = function
  | Whole (W_sketch p) -> p.sk_mma
  | Fiss (F_sketch entries) -> List.exists entries ~f:(fun (_, p) -> p.sk_mma)
  | _ -> false

(* The swizzled staged twin is labeled apart from its plain sibling (gh-ocannl-481 item 3, D3): the
   two are otherwise identical, so a timing report that could not name which is which would be
   reporting the same candidate twice. *)
let swz_label p =
  match p.sk_swizzle with
  | None -> ""
  | Some LL.Swizzle_elem -> " swz-elem"
  | Some LL.Swizzle_b128 -> " swz-b128"

(* The pipelined staged twin likewise (gh-ocannl-487): identical to its plain sibling except the
   cooperative-stage depth, so the label must carry it. *)
let depth_label p = if p.sk_depth > 1 then Printf.sprintf " pd%d" p.sk_depth else ""

let spec_label = function
  | Whole (W_saved s) -> Printf.sprintf "W_saved[%d ops]" (List.length s)
  | Whole (W_preset { block_size }) -> Printf.sprintf "W_preset[bs=%s]" (bs_label block_size)
  | Whole (W_sketch p) when p.sk_mma ->
      Printf.sprintf "W_sketch[%smma-%s %dx%dx%d%s%s%s%s%s%s%s]"
        (if p.sk_conv then "conv-" else "")
        (if p.sk_gpu then "gpu" else "cpu")
        p.sk_bm p.sk_bn p.sk_bk
        (if p.sk_bk > 0 then if p.sk_gpu then " staged" else " pack" else "")
        (swz_label p) (depth_label p)
        (if p.sk_hoist then " hoist" else "")
        (if p.sk_grid then " grid" else "")
        (if p.sk_pack_rest then " packrest" else "")
        (if p.sk_epilogue then " ep" else "")
  | Whole (W_sketch p) ->
      Printf.sprintf "W_sketch[%s %dx%dx%d/%dx%d%s%s]"
        (if p.sk_gpu then "gpu" else "cpu")
        p.sk_bm p.sk_bn p.sk_bk p.sk_tm p.sk_tn
        (if p.sk_hoist then " hoist" else "")
        (if p.sk_epilogue then " ep" else "")
  | Fiss (F_preset { block_size; privatize; config_thresholds }) ->
      Printf.sprintf "F_preset[bs=%s%s%s]" (bs_label block_size)
        (if privatize then " priv" else "")
        (if config_thresholds then " cfg-thresh" else "")
  | Fiss (F_saved assoc) -> Printf.sprintf "F_saved[%d segs]" (List.length assoc)
  | Fiss (F_sketch entries) ->
      Printf.sprintf "F_sketch[%s]"
        (String.concat ~sep:","
           (List.map entries ~f:(fun (_, p) ->
                Printf.sprintf "%s%s%s %dx%dx%d%s%s%s%s%s%s%s"
                  (if p.sk_conv then "conv-" else "")
                  (if p.sk_mma then "mma-" else "")
                  (if p.sk_gpu then "gpu" else "cpu")
                  p.sk_bm p.sk_bn p.sk_bk
                  (if p.sk_mma then "" else Printf.sprintf "/%dx%d" p.sk_tm p.sk_tn)
                  (swz_label p) (depth_label p)
                  (if p.sk_hoist then " hoist" else "")
                  (if p.sk_grid then " grid" else "")
                  (if p.sk_pack_rest then " packrest" else "")
                  (if p.sk_epilogue then " ep" else ""))))
  | Fiss (F_split { sites }) ->
      Printf.sprintf "F_split[%s]"
        (String.concat ~sep:","
           (List.map sites ~f:(fun (s, b) ->
                Printf.sprintf "%s%s red%d out%d b%d%s"
                  (Ir.Tnode.debug_name s.sr_target)
                  (if s.sr_dynamic then " dyn" else "")
                  s.sr_red s.sr_out b
                  (match List.length s.sr_swaps with
                  | 0 -> ""
                  | n -> Printf.sprintf " swap%d" n))))
  | Fiss (F_split_saved (prelude, assoc)) ->
      Printf.sprintf "F_split_saved[%d prelude ops, %d segs]" (List.length prelude)
        (List.length assoc)

(* Every candidate derives its CODE from the ONE base lowering ([base_opt] with [canon] its
   canonical form, captured together in [tune]) rather than from the compile's own fresh lowering,
   whose llc the transform ignores. Re-lowering per candidate was subtly unsound: timing runs settle
   tensor-node value bounds, so later fresh lowerings can fold guards (and even re-segment fission)
   differently from the base — failing digest checks at best (the CUDA rounds on PR #140: whole arms
   degenerating to their serial baselines) and silently replaying the winner with empty per-segment
   schedules at worst (a 296 ms winner returning as a 2614 ms routine). Deriving from the base makes
   candidates and the winner replay drift-immune and byte-comparable by construction; the
   fresh-lowering digest check survives only in spirit via the disk cache's [source_digest] guard
   (cross-process compatibility).

   The rebased code keeps the fresh compile's OWN [optimize_ctx] (the per-compile fork of the
   context's lineage): link-time buffer allocation consults that fork, so placement mutations by
   schedule ops — fission's Local promotions above all — must land there or the allocator would miss
   buffers the kernels reference. Candidate hermeticity is unchanged: each compile forks the lineage
   table anew. The traced store is copied from the base (schedule ops register their tiles in
   it). *)
let compile_candidate ~static_indices ~base_opt ~canon ~limits ~is_gpu ~is_cpu ~provenance ctx comp
    bindings spec : compiled Outcome.outcome =
  let candidate = spec_label spec in
  let rebase (fresh : LL.optimized) =
    {
      base_opt with
      LL.traced_store = Hashtbl.copy base_opt.LL.traced_store;
      LL.optimize_ctx = fresh.LL.optimize_ctx;
    }
  in
  let preset_sched ?block_size ?(config_thresholds = false) opt =
    let min_parallel = if config_thresholds then None else Some 1 in
    if is_gpu then Sched.default_gpu ?block_size ?min_parallel ~limits opt
    else if is_cpu then Sched.default_cpu ?min_parallel opt
    else []
  in
  let captured = ref None in
  let compile_ctx () =
    match spec with
    | Whole flavor ->
        let transform fresh =
          let opt = rebase fresh in
          let sched, saved, registry =
            match flavor with
            | W_saved saved ->
                let sched, registry = SC.of_saved canon saved in
                (sched, saved, registry)
            | W_preset { block_size } ->
                let sched = preset_sched ?block_size opt in
                let saved, registry = SC.to_saved (SC.base_registry canon) sched in
                (sched, saved, registry)
            | W_sketch p ->
                let sched = sketch_schedule ~p opt in
                let saved, registry = SC.to_saved (SC.base_registry canon) sched in
                (sched, saved, registry)
          in
          let opt' = Sched.apply_classified ~static_indices sched opt in
          let digest_after = SC.digest (SC.canonicalize ~static_indices opt') in
          captured :=
            Some
              ( Whole_saved saved,
                [ { u_key = None; u_saved = saved; u_registry = registry; u_opt = opt' } ],
                [ opt' ],
                digest_after );
          opt'
        in
        Context.compile_outcome ~lowered_transform:transform ~provenance ~candidate ctx comp bindings
    | Fiss flavor ->
        let transforms fresh =
          let opt = rebase fresh in
          (* The split-reduce prelude (gh-ocannl-484 task 3) applies whole-routine BEFORE fission:
             the partials edge it mints is what fission cuts at, giving the two passes separate
             kernels and the event-chain synchronization the combine needs. *)
          let prelude, prelude_saved =
            match flavor with
            | F_preset _ | F_saved _ | F_sketch _ -> ([], [])
            | F_split { sites } ->
                let sched =
                  (* Per site: the gh-537 enabling interchange (empty for a site splittable as
                     lowered), then the split itself. Sites are distinct statements, so their
                     preludes compose. *)
                  List.concat_map sites ~f:(fun (s, num_blocks) ->
                      let op, _, _, _ =
                        Sched.split_reduce ~axis:s.sr_axis ~target:s.sr_target ~num_blocks
                      in
                      List.map s.sr_swaps ~f:(fun (outer, inner) -> Sched.Swap { outer; inner })
                      @ [ op ])
                in
                let saved, _ = SC.to_saved (SC.base_registry canon) sched in
                (sched, saved)
            | F_split_saved (psaved, _) ->
                let sched, _ = SC.of_saved canon psaved in
                (sched, psaved)
          in
          let opt =
            if List.is_empty prelude then opt
            else Sched.apply_classified ~static_indices prelude opt
          in
          let zero_sched tns = if is_gpu then Sched.zero_expansion ~limits tns else [] in
          (* Per-segment schedule matching keys on the STRUCTURAL canon ([with_placements:false]):
             placement classes can render differently across compilation lineages on byte-identical
             segments (decided in one, undecided in the other — e.g. tuning with [timing_ctx]),
             which used to fail winner replays wholesale. A lookup miss returns the empty schedule:
             [fission_scheduled] probes {e fine} (pre-coalescing) segments through this closure, and
             only the empty-on-miss answer lets coalescing re-converge to the saved segmentation,
             where every final [`Normal] segment's digest hits (the verification after fission below
             catches genuine drift loudly instead of silently replaying unscheduled segments). *)
          let seg_key seg =
            SC.digest (SC.canonicalize ~static_indices ~with_placements:false seg)
          in
          let preset seg =
            match flavor with
            | F_preset { block_size; privatize; config_thresholds } ->
                let sched = preset_sched ?block_size ~config_thresholds seg in
                if privatize then extend_with_privatize ~static_indices sched seg else sched
            | F_saved entries | F_split_saved (_, entries) -> (
                let seg_canon = SC.canonicalize ~static_indices ~with_placements:false seg in
                match List.Assoc.find entries ~equal:String.equal (SC.digest seg_canon) with
                | Some saved -> fst (SC.of_saved seg_canon saved)
                | None -> [])
            | F_sketch entries -> (
                match List.Assoc.find entries ~equal:String.equal (seg_key seg) with
                | Some p -> sketch_schedule ~p seg
                | None -> preset_sched seg)
            | F_split _ -> preset_sched seg
          in
          let tuples =
            (* Match the default pipeline's placements (statement-crossing [Local]s promoted on
               GPU), so fissioned candidates and the untuned baseline schedule the same code. *)
            Sched.fission_scheduled ~promote_locals:is_gpu ~preset ~zero_sched ~static_indices opt
          in
          (* Genuine-drift guard for saved replays (cross-process cache entries): with the
             empty-on-miss closure above, a saved winner whose segmentation no longer matches would
             coalesce differently and silently replay some segments unscheduled. Verify instead that
             every final [`Normal] segment found its saved schedule. *)
          (match flavor with
          | F_preset _ | F_sketch _ | F_split _ -> ()
          | F_saved entries | F_split_saved (_, entries) ->
              List.iter tuples ~f:(fun (kind, pre, _, _) ->
                  match kind with
                  | `Zeros | `Solo -> ()
                  | `Normal ->
                      if not (List.Assoc.mem entries ~equal:String.equal (seg_key pre)) then
                        invalid_arg
                          "Autotune: fissioned replay: no saved schedule for a segment \
                           (segmentation drifted)"));
          let posts = List.map tuples ~f:(fun (_, _, _, post) -> post) in
          let units =
            List.filter_map tuples ~f:(fun (kind, pre, sched, post) ->
                match kind with
                | `Zeros | `Solo -> None
                | `Normal ->
                    (* The structural canon: [u_key] must match the replay closure's lookup, and
                       [of_saved] at replay resolves against the same (placement-independent)
                       binder/tnode numbering. *)
                    let pre_canon = SC.canonicalize ~static_indices ~with_placements:false pre in
                    let saved, registry = SC.to_saved (SC.base_registry pre_canon) sched in
                    Some
                      {
                        u_key = Some (SC.digest pre_canon);
                        u_saved = saved;
                        u_registry = registry;
                        u_opt = post;
                      })
          in
          let assoc =
            (* One entry per [`Normal] segment in segment order; structurally identical segments
               share a key and their saved forms are interchangeable, so duplicates are harmless. *)
            List.map units ~f:(fun u -> (Option.value_exn u.u_key, u.u_saved))
          in
          let digest_after =
            String.concat ~sep:"+"
              (List.map posts ~f:(fun post -> SC.digest (SC.canonicalize ~static_indices post)))
          in
          let form =
            if List.is_empty prelude_saved then Fiss_saved assoc
            else Split_saved (prelude_saved, assoc)
          in
          captured := Some (form, units, posts, digest_after);
          posts
        in
        Context.compile_outcome ~lowered_transforms:transforms ~provenance ~candidate ctx comp
          bindings
  in
  (* Collect the Tile_mma rendering census across this candidate's kernel compiles (fissioned
     segments included); [mma_census_enabled] keeps the census from growing in non-tuning
     processes. Compiles are sequential on the main domain, so save-and-restore suffices. *)
  Ir.C_syntax.mma_census := [];
  Ir.C_syntax.mma_census_enabled := true;
  let compile_result =
    Exn.protect ~f:compile_ctx ~finally:(fun () -> Ir.C_syntax.mma_census_enabled := false)
  in
  match compile_result with
  | Error failure -> Error failure
  | Ok (cctx, routine) -> (
      let mma_renders = !Ir.C_syntax.mma_census in
      match !captured with
      | Some (form, units, all_opts, digest_after) ->
          Ok { form; cctx; routine; units; all_opts; digest_after; mma_renders }
      | None ->
          Outcome.protect ~classify_backend:(fun _ _ -> None) ~provenance
            ~phase:Outcome.Transform ~candidate (fun () ->
              failwith "Autotune: the transform was not invoked"))

(** {2 The action menu} *)

type loop_desc = {
  ld_ref : SC.sym_ref;
  ld_sym : Idx.symbol;  (** The raw binder, for consulting {!Sched.op_legality}. *)
  ld_extent : int;
  ld_axis : LL.axis_type;
  ld_innermost : bool;
  ld_accumulating : bool;
  ld_perfect_child : (SC.sym_ref * Idx.symbol * LL.axis_type) option;
}

let rec contains_loop = function
  | LL.Seq (a, b) -> contains_loop a || contains_loop b
  | LL.If { body; _ } -> contains_loop body
  | LL.For_loop _ -> true
  | _ -> false

(* Loops proposable for schedule ops: the statement-level nest structure (we do not descend into
   [Local_scope] bodies or [Tile_mma] fallbacks — transforming those is never profitable and often
   invalid), restricted to loops whose binder the registry can name (Stage-internal copy loops
   cannot be referenced by a persisted schedule). *)
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
                  Option.map (SC.resolve registry ci) ~f:(fun r -> (r, ci, cax))
              | _ -> None
            in
            acc :=
              {
                ld_ref;
                ld_sym = index;
                ld_extent = to_ + 1;
                ld_axis = axis;
                ld_innermost = not (contains_loop body);
                ld_accumulating = LL.has_accumulation body;
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
                acc := ((ri, i, ti + 1), (rj, j, tj + 1), (rk, k, tk + 1)) :: !acc
            | _ -> ())
        | _ -> ());
        walk body
    | LL.For_loop { body; _ } -> walk body
    | _ -> ()
  in
  walk llc;
  List.rev !acc

let split_factors = [ 2; 4; 8; 16; 32 ]
let max_actions_per_unit = 48

let menu ~is_cpu ~is_gpu ~(limits : Ir.Backend_intf.hardware_limits) (u : unit_gen) :
    SC.saved_optop list =
  let loops = collect_loops u.u_registry u.u_opt.LL.llc in
  (* Menu proposals carry their raw-symbol counterpart so the op-legality oracle (gh-494 waypoint 3)
     can veto proven-illegal ones before they cost a candidate compile; [Op_unknown] proposals
     proceed to compile-and-time exactly as before (the oracle's Unknown is never a rejection). *)
  let gate (saved, raw) =
    match Sched.op_legality u.u_opt raw with
    | Sched.Op_illegal witness ->
        logf "menu prune (illegal): %s" witness;
        None
    | Sched.Op_legal | Sched.Op_unknown _ -> Some saved
  in
  let splits =
    List.concat_map loops ~f:(fun ld ->
        if not (LL.equal_axis_type ld.ld_axis LL.Serial) then []
        else
          List.filter_map split_factors ~f:(fun factor ->
              if factor < ld.ld_extent && ld.ld_extent % factor = 0 then
                let raw, _, _ =
                  Sched.split ~axis:ld.ld_sym ~factor ~outer:LL.Serial ~inner:LL.Serial
                in
                gate
                  (SC.Split { axis = ld.ld_ref; factor; outer = LL.Serial; inner = LL.Serial }, raw)
              else None))
  in
  let swaps =
    List.filter_map loops ~f:(fun ld ->
        match (ld.ld_axis, ld.ld_perfect_child) with
        | LL.Serial, Some (child, child_sym, LL.Serial) ->
            gate
              ( SC.Swap { outer = ld.ld_ref; inner = child },
                Sched.Swap { outer = ld.ld_sym; inner = child_sym } )
        | _ -> None)
  in
  let unrolls =
    List.concat_map loops ~f:(fun ld ->
        if LL.equal_axis_type ld.ld_axis LL.Serial && ld.ld_extent <= 8 then
          List.filter_map [ true; false ] ~f:(fun materialize ->
              gate
                ( SC.Unroll { axis = ld.ld_ref; materialize },
                  Sched.Unroll { axis = ld.ld_sym; materialize } ))
        else [])
  in
  let vectorizes =
    (* CPU renders eligible retyped loops via vector extensions (or vectorization pragmas); GPU
       backends render them as 128-bit packed loads/stores (gh-ocannl-463). Ineligible candidates
       fall back to plain serial loops, so a proposal that fails codegen eligibility merely times
       like the baseline. Accumulating bodies are proposable on CPU (gh-ocannl-468): the renderer
       either emits the reduction-chains rendering or falls back to a plain serial loop — never to a
       vectorization pragma, which would assert iteration independence the loop-carried accumulation
       does not satisfy. On GPU the reduction rendering does not exist (reductions parallelize via
       [Workgroup_reduce] instead), so accumulations stay excluded. *)
    if not (is_cpu || is_gpu) then []
    else
      List.filter_map loops ~f:(fun ld ->
          if
            LL.equal_axis_type ld.ld_axis LL.Serial
            && ld.ld_innermost
            && ((not ld.ld_accumulating) || is_cpu)
          then
            gate
              ( SC.Retype { axis = ld.ld_ref; ty = LL.Vectorized },
                Sched.Retype { axis = ld.ld_sym; ty = LL.Vectorized } )
          else None)
  in
  let triples = collect_serial_triples u.u_registry u.u_opt.LL.llc in
  let tensorizes =
    match limits.Ir.Backend_intf.mma with
    | None -> []
    | Some { Ir.Backend_intf.mma_simd_width; mma_tile = tm, tn, tk; _ } ->
        (* The nesting order need not match the (i, j, k) roles — the roles are fixed by the
           accumulation pattern. The op-legality oracle decides role-assignment validity (gh-494
           waypoint 3 follow-up): invalid permutations — most of the 6 per triple — are proven
           [Op_illegal] by the probe of apply's micro-kernel recognition and pruned before they cost
           a candidate compile, instead of failing at compile time. Propose role assignments
           compatible with the intrinsic tile's divisibility per role. *)
        List.concat_map triples ~f:(fun (t1, t2, t3) ->
            List.filter_map
              [ (t1, t2, t3); (t1, t3, t2); (t2, t1, t3); (t2, t3, t1); (t3, t1, t2); (t3, t2, t1) ]
              ~f:(fun ((i, si, ei), (j, sj, ej), (k, sk, ek)) ->
                if ei % tm = 0 && ej % tn = 0 && ek % tk = 0 then
                  let raw, _lane = Sched.tensorize ~i:si ~j:sj ~k:sk ~simd_width:mma_simd_width in
                  gate (SC.Tensorize { i; j; k; simd_width = mma_simd_width }, raw)
                else None))
  in
  logf
    "menu: %d serial triple(s) -> %d tensorize proposal(s); %d split, %d swap, %d unroll, %d \
     vectorize"
    (List.length triples) (List.length tensorizes) (List.length splits) (List.length swaps)
    (List.length unrolls) (List.length vectorizes);
  List.take (tensorizes @ splits @ swaps @ unrolls @ vectorizes) max_actions_per_unit

(* Extend one unit of a compiled candidate with a menu action. The fissioned entries stay in segment
   order (the positional replay fallback relies on it); extending by key updates every structurally
   identical segment — they carry interchangeable saved forms, so extending them uniformly keeps the
   digest lookup and the positional entries consistent. *)
let extend_spec (elem : compiled) (u : unit_gen) (op : SC.saved_optop) : spec option =
  match (elem.form, u.u_key) with
  | Whole_saved _, None -> Some (Whole (W_saved (u.u_saved @ [ op ])))
  | Fiss_saved assoc, Some key ->
      Some
        (Fiss
           (F_saved
              (List.map assoc ~f:(fun (k, s) ->
                   if String.equal k key then (k, u.u_saved @ [ op ]) else (k, s)))))
  | Split_saved (prelude, assoc), Some key ->
      Some
        (Fiss
           (F_split_saved
              ( prelude,
                List.map assoc ~f:(fun (k, s) ->
                    if String.equal k key then (k, u.u_saved @ [ op ]) else (k, s)) )))
  | _ -> None

(** {2 Model-picked untuned defaults (gh-ocannl-491 task 3)}

    A drop-in for [Context.compile] that raises the untuned floor: with no measurement at all, the
    default pipeline and the sketch families are scored with the roofline model inside the compile's
    own transform seam, and the model-argmin schedule is applied. Advisory by construction — a
    candidate without model coverage is never picked over the default, ties go to the default, and
    any scoring or application failure falls back to the ordinary default pipeline. *)

type model_choice = {
  mc_label : string;
      (** ["default"] or the winning candidate's spec label (matching {!tune}'s [autotune_log]
          labels). *)
  mc_model_ms : float option;
      (** The winner's roofline lower bound in ms — a ranking score, not a runtime prediction;
          [None] when selection did not run (no envelope constants, automatic scheduling disabled,
          or the default itself had no model coverage). *)
  mc_scored : int;
      (** Model evaluations that produced a score (the default pipeline included; the fissioned flow
          also scores per segment). *)
  mc_skipped : int;  (** Model evaluations without coverage, excluded from ranking. *)
  mc_rejected : int;
      (** Candidates excluded from the ranking because their scheduled form fails
          {!Ir.Low_level.validate_parallel} — it could not have compiled (gh-ocannl-522). *)
}

let model_default_enabled =
  lazy (Utils.get_global_flag ~default:false ~arg_name:"model_default_schedule")

(* The model ranks several scheduled forms without compiling them. Keep this eager validation in
   that ranking loop: removing it made an invalid tensorized argmin displace a viable schedule and
   then fall back all the way to the default. This is no longer needed for exception attribution or
   advisory containment -- codegen carries the same typed cause -- only to preserve "best viable
   model candidate" selection without compiling every contender. *)
let validate_segments_for_model (segs : LL.optimized list) =
  List.iter segs ~f:(fun (o : LL.optimized) ->
      LL.validate_parallel_classified o.LL.optimize_ctx.LL.placements o.LL.llc);
  segs

let compile_advisory ?on_fallback ?fallback_if lowered_transforms ctx comp bindings =
  match
    Context.compile_outcome ~lowered_transforms ~provenance:Outcome.Advisory ctx comp bindings
  with
  | Ok result -> result
  | Error (Outcome.Fatal _ as failure) -> Outcome.raise_failure failure
  | Error (Outcome.Classified classified as failure) ->
      (* [fallback_if] is what keeps the retry from duplicating a genuine failure: a transform that
         already degraded to the default pipeline has nothing to fall back TO, so recompiling would
         just repeat the same failing compile (and, on a resource failure, aggravate it) before
         raising the same exception. Such callers say [false] here and the original exception
         propagates through the public exception contract. *)
      if not (Option.value_map fallback_if ~default:true ~f:(fun f -> f ())) then
        Outcome.raise_failure failure;
      (* Typed compiler rejection is the advisory fallback boundary. Fatal failures are propagated
         above without paying for a second compile. *)
      Option.iter on_fallback ~f:(fun f -> f (Outcome.exception_of_cause classified.cause));
      Context.compile ctx comp bindings

let model_default ?report ctx comp bindings =
  let backend = Context.backend_name ctx in
  let is_gpu = Sched.backend_is_gpu backend and is_cpu = Sched.backend_is_cpu backend in
  let limits = Context.hardware_limits ctx in
  let static_indices = Idx.bound_symbols bindings in
  let peak_flops, peak_memory_bandwidth = envelope ~limits in
  let emit r = Option.iter report ~f:(fun f -> f r) in
  let no_selection =
    { mc_label = "default"; mc_model_ms = None; mc_scored = 0; mc_skipped = 0; mc_rejected = 0 }
  in
  if
    (Option.is_none peak_flops && Option.is_none peak_memory_bandwidth)
    || not (Sched.automatic_schedule_active ~backend_name:backend)
  then (
    emit no_selection;
    Context.compile ctx comp bindings)
  else
    let choice = ref no_selection in
    (* Whether the segments the compile actually received came from a model pick rather than the
       default pipeline — the condition for the compile-level fallback below to have anywhere to
       fall back to. *)
    let applied_pick = ref false in
    let transforms (opt : LL.optimized) : LL.optimized list =
      let n_scored = ref 0 and n_skipped = ref 0 and n_rejected = ref 0 in
      let score opts =
        match
          summaries_roofline ~peak_flops ~peak_memory_bandwidth
            (List.map opts ~f:(fun o -> CM.analyze o.LL.llc))
        with
        | Some s ->
            Int.incr n_scored;
            Some s
        | None ->
            Int.incr n_skipped;
            None
      in
      (* The model must rank the best viable schedule, not crown an invalid argmin and fall all the
         way back to default. The validator is typed, so only an expected schedule rejection is
         excluded; compiler assertions and other failures still escape. *)
      let score_valid opts =
        match validate_segments_for_model opts with
        | opts -> score opts
        | exception Outcome.Cause_at _ ->
            Int.incr n_rejected;
            None
      in
      let default_segs () =
        Sched.maybe_default_schedules ~backend_name:backend ~limits ~static_indices opt
      in
      let preset seg = if is_gpu then Sched.default_gpu ~limits seg else Sched.default_cpu seg in
      let zero_sched tns = if is_gpu then Sched.zero_expansion ~limits tns else [] in
      let seg_key seg = SC.digest (SC.canonicalize ~static_indices ~with_placements:false seg) in
      let label, model_s, action =
        try
          (* The untuned default pipeline, scored on a hermetic copy — it is both the anchor
             candidate and the fallback. *)
          let default_scratch =
            Sched.maybe_default_schedules ~backend_name:backend ~limits ~static_indices
              (scratch_of opt)
          in
          let default_score = score default_scratch in
          match default_score with
          | None ->
              (* No coverage of the default itself: nothing to honestly compare against. *)
              ("default", None, `Default)
          | Some ds -> (
              (* Whole-routine sketch candidates. A candidate without coverage is skipped — it is
                 never picked over the default without a measured run ({!tune} covers that). *)
              let whole =
                List.filter_map (sketch_seed_params ~is_gpu ~is_cpu ~limits opt) ~f:(fun p ->
                    match
                      Sched.apply_classified ~static_indices (sketch_schedule ~p opt)
                        (scratch_of opt)
                    with
                    | exception Outcome.Cause_at _ ->
                        Int.incr n_rejected;
                        None
                    | post -> Option.map (score_valid [ post ]) ~f:(fun s -> (p, s)))
              in
              let contenders =
                List.map whole ~f:(fun (p, s) -> (spec_label (Whole (W_sketch p)), s, `Whole p))
              in
              (* Per-segment sketch substitution over the default fission segmentation (only when
                 the default actually fissioned; otherwise the whole-routine sketches cover the
                 site). Mirrors [tune]'s [F_sketch] flavor: segments keyed by their structural
                 pre-schedule digest, a key miss degrading to the default preset. *)
              let fiss =
                if List.length default_scratch <= 1 then None
                else
                  match
                    Sched.fission_scheduled ~promote_locals:is_gpu ~preset ~zero_sched
                      ~static_indices (scratch_of opt)
                  with
                  | exception Outcome.Cause_at _ ->
                      Int.incr n_rejected;
                      None
                  | tuples -> (
                      let entries =
                        List.filter_map tuples ~f:(fun (kind, pre, _sched, post) ->
                            match kind with
                            | `Zeros | `Solo -> None
                            | `Normal -> (
                                let base_score = score [ post ] in
                                let best_sketch =
                                  List.filter_map (sketch_seed_params ~is_gpu ~is_cpu ~limits pre)
                                    ~f:(fun p ->
                                      match
                                        Sched.apply_classified ~static_indices
                                          (sketch_schedule ~p pre)
                                          (scratch_of pre)
                                      with
                                      | exception Outcome.Cause_at _ ->
                                          Int.incr n_rejected;
                                          None
                                      | sp -> Option.map (score_valid [ sp ]) ~f:(fun s -> (p, s)))
                                  |> List.min_elt ~compare:(fun (_, a) (_, b) -> Float.compare a b)
                                in
                                match (base_score, best_sketch) with
                                | Some bs, Some (p, s) when Float.(s < bs) -> Some (seg_key pre, p)
                                | _ -> None))
                      in
                      if List.is_empty entries then None
                      else
                        let subst_preset seg =
                          match List.Assoc.find entries ~equal:String.equal (seg_key seg) with
                          | Some p -> sketch_schedule ~p seg
                          | None -> preset seg
                        in
                        (* Score the substituted pipeline whole, so it competes on the same footing
                           as the other candidates. *)
                        match
                          Sched.fission_scheduled ~promote_locals:is_gpu ~preset:subst_preset
                            ~zero_sched ~static_indices (scratch_of opt)
                        with
                        | exception Outcome.Cause_at _ ->
                            Int.incr n_rejected;
                            None
                        | tuples2 ->
                            let posts = List.map tuples2 ~f:(fun (_, _, _, post) -> post) in
                            Option.map (score_valid posts) ~f:(fun s -> (entries, s)))
              in
              let contenders =
                contenders
                @
                match fiss with
                | Some (entries, s) -> [ (spec_label (Fiss (F_sketch entries)), s, `Fiss entries) ]
                | None -> []
              in
              (* Argmin with ties to the default: the model only displaces the honest default on a
                 strict improvement. *)
              let best =
                List.min_elt contenders ~compare:(fun (_, a, _) (_, b, _) -> Float.compare a b)
              in
              match best with
              | Some (lbl, s, act) when Float.(s < ds) -> (lbl, Some s, act)
              | _ -> ("default", Some ds, `Default))
        with Outcome.Cause_at (_, cause) ->
          logf "model_default: scoring declined (%s); using the default pipeline"
            (Outcome.detail_of_cause cause);
          ("default", None, `Default)
      in
      choice :=
        {
          mc_label = label;
          mc_model_ms = Option.map model_s ~f:(fun s -> s *. 1e3);
          mc_scored = !n_scored;
          mc_skipped = !n_skipped;
          mc_rejected = !n_rejected;
        };
      let apply_action () =
        (* Schedule application uses the typed seam. Backend validation is deliberately left to
           [compile_advisory], which now receives its classified cause directly from codegen. *)
        match action with
        | `Default -> default_segs ()
        | `Whole p ->
            validate_segments_for_model
              [ Sched.apply_classified ~static_indices (sketch_schedule ~p opt) opt ]
        | `Fiss entries ->
            let subst_preset seg =
              match List.Assoc.find entries ~equal:String.equal (seg_key seg) with
              | Some p -> sketch_schedule ~p seg
              | None -> preset seg
            in
            validate_segments_for_model
              (List.map
                 (Sched.fission_scheduled ~promote_locals:is_gpu ~preset:subst_preset ~zero_sched
                    ~static_indices opt) ~f:(fun (_, _, _, post) -> post))
      in
      match apply_action () with
      | segs ->
          logf "model_default: chose %s (model %s; %d scored, %d without coverage, %d unbuildable)"
            label
            (match model_s with Some s -> Printf.sprintf "%.6f ms" (s *. 1e3) | None -> "n/a")
            !n_scored !n_skipped !n_rejected;
          (applied_pick := match action with `Default -> false | `Whole _ | `Fiss _ -> true);
          segs
      | exception Outcome.Cause_at (_, cause) ->
          logf
            "model_default: winner %s FAILED to apply or validate (%s); using the default pipeline"
            label (Outcome.detail_of_cause cause);
          choice :=
            {
              no_selection with
              mc_scored = !n_scored;
              mc_skipped = !n_skipped;
              mc_rejected = !n_rejected;
            };
          applied_pick := false;
          default_segs ()
    in
    let on_fallback exn =
      logf "model_default: compiling the pick %s FAILED (%s); recompiling the default pipeline"
        !choice.mc_label (Exn.to_string exn);
      choice := { !choice with mc_label = "default"; mc_model_ms = None }
    in
    (* With the model on the default pipeline (no sketch strictly improved on it, or the pick failed
       validation above), the compile that just failed IS the fallback: retrying it would duplicate
       an expensive failure and delay the honest error, so the exception propagates instead. *)
    let result =
      compile_advisory ~on_fallback ~fallback_if:(fun () -> !applied_pick) transforms ctx comp
        bindings
    in
    emit !choice;
    result

(** {2 The search} *)

(* gh-ocannl-550: the containment properties of the search — a failed candidate costs that
   candidate, a failed search costs that search and not its sibling arm — are only testable with a
   candidate that fails, and the reproduction that motivated them needs a 12 GB GPU and a
   half-hour search. This seam manufactures the failure instead. It is called with the candidate's
   label before each candidate compile; raising from it emulates the shape the device OOM had, a
   failure that is NOT contained as a candidate decline (there it escaped after the search had
   concluded, when the exhausted device defeated both the winner replay and its untuned fallback).
   Not a production seam: default no-op, and no config key selects it. Called for the baseline
   compile too — it is a candidate (gh-ocannl-533) — which is what makes a failure BEFORE the
   search has reported anything injectable, the case the positional-arm-slot handling in
   [Train.tune_placements] exists for. *)
let on_candidate_attempt : (string -> unit) ref = ref (fun _label -> ())

let tune ?search ?beam_width ?rounds ?repeats ?seed_block_sizes ?cache_dir ?keep_fraction
    ?max_split_reduce_sites ?timing_ctx ?report ctx comp bindings =
  (* gh-ocannl-559: with the search off, [tune] still replays an explicitly provided cache -- a
     pinned schedule is deterministic, and committing one is how a reproducible run keeps a tuned
     schedule -- but never times candidates, whose crowning is the largest cross-machine
     determinism leak. A miss compiles the untuned default pipeline, exactly like the
     nothing-was-timed fallback below. *)
  let search =
    Option.value search ~default:(Utils.get_global_flag ~arg_name:"autotune_search" ~default:true)
  in
  let beam_width =
    max 1 (Option.value beam_width ~default:(int_arg ~arg_name:"autotune_beam_width" ~default:2))
  in
  let rounds = Option.value rounds ~default:(int_arg ~arg_name:"autotune_rounds" ~default:2) in
  let repeats = Option.value repeats ~default:(int_arg ~arg_name:"autotune_repeats" ~default:3) in
  let max_split_reduce_sites =
    max 0
      (Option.value max_split_reduce_sites
         ~default:(int_arg ~arg_name:"autotune_split_reduce_max_sites" ~default:8))
  in
  let seed_block_sizes = Option.value seed_block_sizes ~default:[ 64; 128; 256; 512 ] in
  (* Whether the cache directory was CHOSEN, as opposed to being the built-in default: passed by the
     caller, or set at some config source (a profile payload included). Only relevant with the
     search off, where it is the difference between replaying a cache someone committed and
     replaying whatever an earlier local search happened to leave in ./autotune_cache
     (gh-ocannl-559; Codex P2 on PR #291) -- the latter would make two reproducible runs differ on
     local state, which is the leak that turning the search off exists to close. *)
  let cache_dir_chosen =
    Option.is_some cache_dir
    ||
    match snd (Utils.get_global_arg_with_source ~arg_name:"autotune_cache_dir" ~default:"") with
    | Utils.From_default -> false
    | _ -> true
  in
  let cache_dir =
    Option.value cache_dir
      ~default:(Utils.get_global_arg ~arg_name:"autotune_cache_dir" ~default:"autotune_cache")
  in
  (* A search-less [tune] replays only a cache someone asked for. *)
  let cache_dir = if search || cache_dir_chosen then cache_dir else "" in
  let keep_fraction =
    Option.value keep_fraction ~default:(float_arg ~arg_name:"autotune_keep_fraction" ~default:1.)
  in
  let static_indices = Idx.bound_symbols bindings in
  let backend = Context.backend_name ctx in
  let emit_report r = Option.iter report ~f:(fun f -> f r) in
  (* [tune] reports exactly once per call, on every path (gh-ocannl-550). The failures that happen
     before (or instead of) the search proper — the base compile failing before its lowering is
     captured, a fatal baseline link, a fatal cache replay, a baseline timing failure, and either
     untuned fallback compile of a search-less call — used to raise with no report at all, which
     leaves a caller that attributes arms by arrival order (the positional [?report] of
     [Train.tune_placements]) with no slot for this search. The phase reported is the one the
     failure itself carries, so the diagnostic names where it actually died — at codegen, at link,
     at launch, at sync — instead of guessing. Reporting is best-effort here, as on the search's own
     fatal path: it must not replace the compiler failure. [base] carries whatever the call did
     learn before failing (e.g. a decline census). *)
  let emit_pre_search_failure ?(base = no_search_report) ~phase ~candidate ~detail () =
    let r =
      {
        base with
        partial = true;
        best_label = "";
        terminal_failure = Some { phase; candidate; detail };
      }
    in
    try emit_report r
    with report_exn when not (process_fatal_exn report_exn) ->
      Stdio.eprintf "autotune: pre-search failure report callback failed: %s\n%!"
        (Exn.to_string report_exn)
  in
  (* gh-ocannl-550: every [raise_pre_search] leaves [tune] without returning a routine, so the base
     compile's artifact is dead on all of them — but the base is linked further down, after this is
     defined, so the release action arrives by hook rather than by reference. A hook rather than a call
     at each raise site on purpose: the previous rounds of this work fixed such sites one at a time and
     each new one was a fresh leak (the fatal cache replay was the last of them), whereas a family with
     one member cannot be partially updated. Harmless where nothing is linked yet — the two raises
     above the base compile invoke the no-op default. *)
  let release_baseline_hook = ref (fun () -> ()) in
  (* The ONE way this function releases anything (gh-ocannl-550). Releasing is best-effort everywhere:
     it runs on failure paths where the device may already be refusing work, and a failure to give
     memory back must never replace the outcome the caller has to act on. Process-fatal conditions
     still propagate. A helper rather than the ad-hoc guard this started as, because "is this call
     wrapped?" produced its own review finding once already. *)
  let release_quietly ~what ctx =
    try Context.release ctx
    with exn when not (process_fatal_exn exn) ->
      logf "release of %s failed: %s" what (Exn.to_string exn)
  in
  (* [emit_report] on a path that then hands a routine back: the callback's exception propagates by
     design, so the caller never receives [result] and its buffers become unreachable while the pool
     table goes on rooting them. Every site that reports a compiled routine reports through this. *)
  let report_or_release r ~result =
    match emit_report r with
    | () -> ()
    | exception exn ->
        let backtrace = Stdlib.Printexc.get_raw_backtrace () in
        release_quietly ~what:"the routine of a failed completion report" (fst result);
        Stdlib.Printexc.raise_with_backtrace exn backtrace
  in
  let raise_pre_search ?base (failure : Outcome.failure) =
    !release_baseline_hook ();
    (match failure with
    | Outcome.Classified c ->
        emit_pre_search_failure ?base ~phase:c.Outcome.phase ~candidate:None
          ~detail:(Outcome.detail_of_cause c.Outcome.cause) ()
    | Outcome.Fatal f ->
        emit_pre_search_failure ?base ~phase:f.Outcome.phase ~candidate:f.Outcome.candidate
          ~detail:(Exn.to_string f.Outcome.exn) ());
    Outcome.raise_failure failure
  in
  (* The untuned fallback of a search-less call, through the containment-aware form so a failure
     reports the phase it carries. [Context.compile] is exactly this plus [raise_failure], which is
     what [raise_pre_search] ends with, so the caller sees the same exception either way. *)
  let compile_untuned_default ?base () =
    match
      Context.compile_outcome ~provenance:Ir.Schedule_outcome.User_schedule ctx comp bindings
    with
    | Ok result -> result
    | Error failure -> raise_pre_search ?base failure
  in
  (* Without a cache to replay there is nothing for a search-less [tune] to do, so it does not even
     take the base compile that computes the cache key: the caller gets the untuned default compile
     it would have gotten from [Context.compile]. *)
  if (not search) && String.is_empty cache_dir then (
    logf "search disabled (autotune_search=false) and no chosen cache: compiling the untuned default";
    (* Report AFTER the fallback compile: a report is a record of what this call achieved, and
       [no_search_report] says the untuned default shipped. Emitting it first would leave a
       consumer holding a clean, non-partial report for a call that then raised (Codex P2 on PR
       #291); a compile that raises reports its own failure instead (gh-ocannl-550). *)
    let result = compile_untuned_default () in
    report_or_release no_search_report ~result;
    result)
  else
  let is_gpu = Sched.backend_is_gpu backend and is_cpu = Sched.backend_is_cpu backend in
  (* With [timing_ctx], the search (candidate compiles and timing runs) happens against that scratch
     lineage's buffers, and only the winner is compiled from [ctx] — so the timing runs never mutate
     the caller's live state (parameters, accumulators). The scratch context must contain the nodes
     the computation requires from a prior context (e.g. initialized parameters), typically by
     repeating the caller's initialization on a fresh root context. It must live on the same backend
     and device as [ctx] (Codex P2 on PR #109): candidates timed elsewhere do not predict this
     device, and the winner would be cached under this backend's key without ever having been timed
     on it. *)
  Option.iter timing_ctx ~f:(fun tctx ->
      if
        (not (String.equal (Context.backend_name tctx) backend))
        || Context.device_id tctx <> Context.device_id ctx
      then
        invalid_arg
          (Printf.sprintf
             "Autotune.tune: timing_ctx must be on the same backend and device as the target \
              context (timing: %s device %d, target: %s device %d)"
             (Context.backend_name tctx) (Context.device_id tctx) backend (Context.device_id ctx)));

  (* Device work, not a pure query: the GPU backends lazily initialize the device and read driver
     attributes here, so a driver or enumeration error surfaces at this line — the first thing this
     call does that can fail, and squarely inside the reporting contract. *)
  let limits =
    match Context.hardware_limits ctx with
    | limits -> limits
    | exception exn ->
        let backtrace = Stdlib.Printexc.get_raw_backtrace () in
        emit_pre_search_failure ~phase:Outcome.Hardware_limits ~candidate:None
          ~detail:(Exn.to_string exn) ();
        Stdlib.Printexc.raise_with_backtrace exn backtrace
  in
  let search_ctx = Option.value timing_ctx ~default:ctx in
  (* The base compile: identity transform (= the serial baseline candidate), capturing the optimized
     code every candidate derives from (see [compile_candidate]) and its canonical form.
     Canonicalize INSIDE the transform: after the transform returns, codegen forces the remaining
     undecided placements into the very placements table the captured [opt] references, and
     placement classes enter the digest (Schedule_cache.canonicalize) — the disk-cache key must be
     the deterministic transform-time form so that storing and replaying processes agree.

     The baseline is a candidate, so its compile is protected like every other candidate's
     (gh-ocannl-533): a typed rejection — the HIP scratch validator declining the unscheduled serial
     form at [Backend_link] is the case that motivated this — declines the baseline and lets the
     search proceed with the scheduled candidates, instead of killing the run before a single
     candidate has been tried. This is sound because the capture happens INSIDE the transform, which
     runs before codegen and link: [base_opt] survives the rejection, so every candidate still
     derives from the same base lowering. Only the timing of the serial form is lost, and on a GPU
     backend it was never going to be timed anyway ([dispatchable] below). Unclassified failures
     stay fatal: provenance [Candidate] under strict classification.

     gh-ocannl-552 settled whether this base compile should instead be the default-annotated
     pipeline (the shared cause behind gh-ocannl-532 and gh-ocannl-533): it cannot be. The default
     form is [maybe_default_schedules] — fission, then per-segment annotation — so in general it is
     several kernels, not one [optimized] to rebase candidates on; every candidate family (presets,
     the sketch detectors, fission enumeration, beam menu moves) assumes the serial zero point; and
     annotation consults [hardware_limits], which would bake per-device decisions into
     [source_digest]. The consequences that motivated the question are each handled where they
     arise: the scratch hazard by this compile's candidate-grade protection (gh-ocannl-533), the
     GPU dispatch hazard by [dispatchable] (gh-ocannl-532), and the missing "did tuning beat the
     default?" reference by [report.default_ms] — the [config_thresholds] seed's measurement, not a
     new baseline. *)
  let base_capture = ref None in
  let base_outcome =
    Context.compile_outcome
      ~lowered_transform:(fun opt ->
        (* Inside the transform, so an injected fault is classified by the ordinary machinery
           (phase [Transform], provenance [Candidate]) and reaches [raise_pre_search] below with a
           real phase and a report — rather than escaping the whole call unreported, which would
           break the exactly-once contract for direct [tune] callers. *)
        !on_candidate_attempt "baseline";
        base_capture := Some (opt, SC.canonicalize ~static_indices opt);
        opt)
      ~provenance:Outcome.Candidate ~candidate:"baseline" search_ctx comp bindings
  in
  let base_opt, canon =
    match (!base_capture, base_outcome) with
    | Some oc, _ -> oc
    (* Failed before reaching the transform: there is no base lowering, hence no search. *)
    | None, Error failure -> raise_pre_search failure
    | None, Ok _ -> failwith "Autotune.tune: backend compile did not invoke lowered_transform"
  in
  let baseline_linked, baseline_decline =
    match base_outcome with
    | Ok (bctx, broutine) -> (Some (bctx, broutine), None)
    | Error (Outcome.Classified classified) -> (None, Some classified)
    | Error (Outcome.Fatal _ as failure) -> raise_pre_search failure
  in
  (* gh-ocannl-550: the base compile runs BEFORE the cache is consulted — its lowering is what every
     candidate and every replay derives from — so on the two paths that do not search, its linked
     artifact is dead as soon as that decision is taken, and nothing downstream can reach it. On the
     search path it enters the beam instead and is released there. Without this, a warm-cache process
     leaked one full base-candidate pool per [tune] call, permanently (the pool table roots it), which
     for a repeatedly-tuning process is the very accumulation this issue is about. *)
  let release_baseline () =
    Option.iter baseline_linked ~f:(fun (bctx, _) ->
        release_quietly ~what:"the baseline compile" bctx)
  in
  release_baseline_hook := release_baseline;
  let base_digest = SC.digest canon in
  let use_cache = (not (String.is_empty cache_dir)) && SC.complete canon in
  let key = SC.cache_key ?pool_tag:limits.Ir.Backend_intf.worker_pool_tag canon ~backend in
  let compile_spec =
    compile_candidate ~static_indices ~base_opt ~canon ~limits ~is_gpu ~is_cpu
      ~provenance:Outcome.Candidate search_ctx comp bindings
  in
  (* Winner (and cache-hit) compiles target the caller's context; they replay against the same base
     lowering as the search's candidates. *)
  let compile_spec_real provenance =
    compile_candidate ~static_indices ~base_opt ~canon ~limits ~is_gpu ~is_cpu ~provenance ctx comp
      bindings
  in
  let flat_schedule = function
    | Whole_saved saved -> saved
    | Fiss_saved assoc -> List.concat_map assoc ~f:snd
    | Split_saved (prelude, assoc) -> prelude @ List.concat_map assoc ~f:snd
  in
  let is_fissioned = function
    | Whole_saved _ -> false
    | Fiss_saved _ | Split_saved _ -> true
  in
  (* Whether the crowned schedule tensorizes is read off the schedule, not off the winning spec's
     label (gh-ocannl-546): the beam can extend a plainly-labeled incumbent with a [Tensorize] move,
     and a sketch label promises tensorization the transform may not have kept. *)
  let saved_is_tensorized (saved : SC.saved_schedule) =
    List.exists saved ~f:(function SC.Tensorize _ -> true | _ -> false)
  in
  let mma_scalar_fallbacks c =
    List.count c.mma_renders ~f:(fun (_, r) ->
        Ir.C_syntax.equal_mma_rendering r Ir.C_syntax.Mma_scalar_fallback)
  in
  (* The decline census outlives the cache branch: the baseline compile happens before the lookup and
     can be declined whether or not a cached winner then replays (gh-ocannl-533), so a cache-hit
     report has to carry that rejection too — [baseline_declined] with an empty census would be an
     internally inconsistent diagnostic on exactly the warm-cache runs of the workload that motivated
     the fix (Codex review, PR #271). *)
  let declines : (Outcome.rejection_key, decline_acc) Hashtbl.Poly.t = Hashtbl.Poly.create () in
  (* A declined baseline is an ordinary entry in the census: it is the same evidence about the same
     device as any candidate's rejection, and dropping it would report a smaller [candidates_failed]
     than the work actually attempted. It is recorded HERE and NOT as a [Not_dispatched] refusal
     below — the two are mutually exclusive accounts of one baseline, and the gh-ocannl-532 refusal
     asserts a reason ("binds no hardware dimension") that is not why this baseline did not run. *)
  Option.iter baseline_decline ~f:(record_decline declines);
  (* What the call has learned by now: everything after this point reports on top of it, success or
     failure, so a pre-search failure never understates the work already attempted (a declined
     baseline in particular must not read back as [baseline_declined = false], [declines = []]). *)
  let census () =
    {
      no_search_report with
      candidates_failed = failed_count declines;
      baseline_declined = Option.is_some baseline_decline;
      declines = decline_summaries declines;
    }
  in
  let cached =
    if use_cache then
      match SC.lookup ~dir:cache_dir ~key with
      (* The numerics check is belt-and-braces: [key] already carries the same tag (gh-ocannl-568),
         so a policy-mismatched entry normally lives in a different file and is never looked up.
         It catches a hand-moved or hand-written entry, which is the shape of the misdirection this
         guards against — a tf32-vs-default A/B whose cache directories got crossed. *)
      | Some entry
        when String.equal entry.SC.source_digest base_digest
             && String.equal entry.SC.numerics (SC.numerics_tag ()) -> (
          let spec =
            match entry.SC.segments with
            (* A fissioned entry with a non-empty [saved] is a split-reduce winner: [saved] is the
               whole-routine prelude, [segments] the post-prelude per-segment schedules. *)
            | Some assoc when not (List.is_empty entry.SC.saved) ->
                Fiss (F_split_saved (entry.SC.saved, assoc))
            | Some assoc -> Fiss (F_saved assoc)
            | None -> Whole (W_saved entry.SC.saved)
          in
          match compile_spec_real Outcome.Cache_replay spec with
          | Ok c when not (dispatchable ~is_gpu c.all_opts) ->
              (* An entry written before the gh-ocannl-532 rule can name the serial baseline as the
                 winner: it was timed then, and it won by default whenever every candidate failed to
                 compile — the state gh-ocannl-521 recorded for every GPU backend. Replaying it
                 would reintroduce the single-work-item dispatch through the cache, permanently and
                 without ever timing anything. Rejected like a stale entry: the fresh search below
                 overwrites it. Rejecting the replay (rather than bumping [entry_version]) keeps
                 every sound entry, on this backend and on the CPU backends, where an empty schedule
                 is a legitimate winner. *)
              logf "cache entry replays to an unparallelized routine, re-searching: %s"
                (spec_label spec);
              (* gh-ocannl-550: rejected, so its buffers are dead — and the fresh search below is
                 about to want them. *)
              release_quietly ~what:"a rejected cache replay" c.cctx;
              None
          | Ok c ->
              logf "cache hit: %s (best %.4f ms, baseline %.4f ms)" (spec_label spec)
                entry.SC.best_ms entry.SC.baseline_ms;
              (* gh-ocannl-550: the report happens INSIDE the construction of [cached], so a [report]
                 callback that raises here never reaches the [Some result] arm below that releases the
                 baseline, and abandons the replayed winner too — two rooted routine footprints per
                 call, for a caller that retries. Both released before the callback's exception
                 propagates; the exception and its backtrace are unchanged. *)
              let emit_report report =
                match emit_report report with
                | () -> ()
                | exception exn ->
                    let backtrace = Stdlib.Printexc.get_raw_backtrace () in
                    release_quietly ~what:"the replay of a failed cache-hit report" c.cctx;
                    release_baseline ();
                    Stdlib.Printexc.raise_with_backtrace exn backtrace
              in
              emit_report
                {
                  cache_hit = true;
                  candidates_timed = 0;
                  (* No search ran, so the only rejection this can carry is the baseline's. *)
                  candidates_failed = failed_count declines;
                  partial = false;
                  baseline_declined = Option.is_some baseline_decline;
                  declines = decline_summaries declines;
                  terminal_failure = None;
                  rounds_run = 0;
                  sketch_candidates = 0;
                  epilogue_sketch_candidates = 0;
                  fiss_sketch_candidates = 0;
                  fiss_sketch_timed = 0;
                  split_reduce_candidates = 0;
                  split_reduce_timed = 0;
                  mma_candidates = 0;
                  mma_timed = 0;
                  model_scored = 0;
                  model_pruned = 0;
                  fissioned = is_fissioned c.form;
                  baseline_ms = entry.SC.baseline_ms;
                  default_ms =
                    (* The entry's [default_ms] describes the default pipeline under the config
                       that ran the search; the cache key covers neither, so a config change can
                       redefine the default without missing the cache. Fingerprint mismatch drops
                       the stale diagnostic — the winner replay itself stays valid (Codex P2 on
                       PR #279). *)
                    (match (entry.SC.default_ms, entry.SC.default_fingerprint) with
                    | (Some _ as d), Some fp
                      when String.equal fp
                             (Sched.default_schedule_fingerprint ~backend_name:backend) ->
                        d
                    | _ -> None);
                  best_ms = entry.SC.best_ms;
                  best_label = spec_label spec;
                  best_tensorized = saved_is_tensorized (flat_schedule c.form);
                  best_mma_statements = List.length c.mma_renders;
                  best_mma_scalar_fallbacks = mma_scalar_fallbacks c;
                  (* Nothing was timed in this process ([mma_timed = 0] like every other counter
                     here), so there is no measured tensorized candidate to report — including the
                     replayed winner, whose [best_ms] was measured by the process that searched.
                     [best_tensorized] still describes the artifact, which is what it is for. *)
                  mma_best_ms = Float.infinity;
                  best_schedule = flat_schedule c.form;
                };
              Some (c.cctx, c.routine)
          | Error (Outcome.Classified classified) ->
              (* Stale or corrupt entry: fall through to a fresh search. *)
              logf "cache entry replay FAILED, re-searching: %s"
                (Outcome.detail_of_cause classified.cause);
              None
          | Error (Outcome.Fatal _ as failure) -> raise_pre_search ~base:(census ()) failure)
      | _ -> None
    else None
  in
  match cached with
  | Some result ->
      release_baseline ();
      result
  | None when not search ->
      logf "search disabled (autotune_search=false) and no cache entry: compiling the untuned default";
      (* Before the fallback compile, which wants the memory. *)
      release_baseline ();
      (* After the compile, as in the no-cache branch above. The census the base compile already
         produced is carried whether this succeeds or fails. *)
      let reached = census () in
      let result = compile_untuned_default ~base:reached () in
      report_or_release reached ~result;
      result
  | None -> (
      let seen = Hash_set.create (module String) in
      Hash_set.add seen base_digest;
      (* Every gh-ocannl-532 refusal enters the same decline census (gh-ocannl-543). Without it a GPU
         search that timed a single candidate reports [candidates_timed = 1] with an empty census —
         the same report a computation with a one-element schedule space would give — and the
         difference (how many candidates existed and were refused, and why) was only ever visible in
         the [autotune_log] stderr stream. *)
      let record_not_dispatched ~origin ~detail =
        record_decline declines
          {
            Outcome.phase = Outcome.Transform;
            cause = Outcome.Not_dispatched { origin; detail };
            execution_effect = Outcome.No_device_writes;
          }
      in
      (* [None] when the baseline compile was declined (gh-ocannl-533): there is no routine to time
         and none to return, and the search runs on the scheduled candidates alone. *)
      let baseline =
        Option.map baseline_linked ~f:(fun (bctx, broutine) ->
            {
              form = Whole_saved [];
              cctx = bctx;
              routine = broutine;
              units =
                [
                  {
                    u_key = None;
                    u_saved = [];
                    u_registry = SC.base_registry canon;
                    u_opt = base_opt;
                  };
                ];
              all_opts = [ base_opt ];
              digest_after = base_digest;
              mma_renders = [];
            })
      in
      (* Baseline timing failures are the user's bug (e.g. uninitialized inputs) and propagate as
         the exception [Context.run] would give — reported first, with the phase they carry, so the
         arm still occupies its slot (gh-ocannl-550). On a GPU backend the baseline is the
         unscheduled serial form and is not dispatched at all (see [dispatchable]); [infinity] is
         its rank, so every timed candidate beats it and the search never returns it (see the
         fallback at the end), and a declined baseline ranks the same way. *)
      let baseline_dispatched = Option.is_some baseline && dispatchable ~is_gpu [ base_opt ] in
      let baseline_ms =
        match baseline with
        | Some b when baseline_dispatched -> (
            (* Still uncaught in the sense that matters — the caller sees the same exception
               [Context.run] would raise, unwrapped and with its own backtrace. The tagging is only
               so the report can name the phase (pre-dispatch validation vs. launch vs. sync) before
               it propagates.

               The lineage effect is NOT optional, though, and it is why this consults the backend's
               classifier like the candidate timing below (gh-ocannl-550): a baseline launch that may
               have written buffers leaves the lineage unusable, and a caller that CONTAINS this
               failure — [Train.tune_placements] does, per arm — would otherwise go on to time its
               other arm against buffers the failed baseline had already modified. Proven
               write-free, the routine's execution claim is withdrawn instead; unattributed, the
               device's state is unknown and the lineage is condemned, exactly as an unattributed
               candidate launch failure condemns it. *)
            let condemn phase exn =
              match phase with
              (* Nothing to judge and nothing to withdraw (gh-ocannl-564): the routine never ran,
                 and the execution claim is only made after a dispatch. Without this arm an
                 unsatisfied dependency would fall to [None] below on every C backend and condemn
                 the lineage the caller is meant to fix and retry in. *)
              | Outcome.Preflight -> ()
              | _ -> (
                  match Context.failure_classifier b.cctx phase exn with
                  | Some { Ir.Schedule_outcome.execution_effect = Outcome.No_device_writes; _ } ->
                      Context.rollback_execution b.cctx (Context.routine_id b.routine)
                  | Some _ | None ->
                      Context.poison_lineage b.cctx
                        ~routine_name:(Context.routine_name b.routine)
                        exn)
            in
            match
              (* Lineage-wide validation, tagged so [condemn] above reads it for what it is —
                 pre-dispatch, nothing to withdraw — and raised here rather than inside the timing
                 so a baseline failure keeps propagating as the pre-search failure it is. This is
                 the site that made the containment gap invisible on the C backends: the serial
                 baseline is dispatched there, hits this first, and takes the search down with the
                 caller's error, where a GPU backend refuses the baseline outright (gh-ocannl-532)
                 and never reaches it (gh-ocannl-569). *)
              Outcome.tag Outcome.Preflight (fun () ->
                  Context.check_lineage_runnable b.cctx b.routine);
              time_routine ~tag_failures:true ~repeats b.cctx b.routine
            with
            | ms -> ms
            | exception Outcome.Raised_at (phase, exn, backtrace) ->
                condemn phase exn;
                emit_pre_search_failure ~base:(census ()) ~phase ~candidate:(Some "baseline")
                  ~detail:(Exn.to_string exn) ();
                (* gh-ocannl-550: it never reaches the beam, so nothing downstream can release it —
                   and a caller that CONTAINS this (a write-free preflight decline, a
                   backend-classified failure) goes on to another arm or retries. *)
                release_baseline ();
                Stdlib.Printexc.raise_with_backtrace exn backtrace
            | exception exn ->
                let backtrace = Stdlib.Printexc.get_raw_backtrace () in
                condemn Outcome.Launch exn;
                emit_pre_search_failure ~base:(census ()) ~phase:Outcome.Launch
                  ~candidate:(Some "baseline") ~detail:(Exn.to_string exn) ();
                release_baseline ();
                Stdlib.Printexc.raise_with_backtrace exn backtrace)
        | _ -> Float.infinity
      in
      (match baseline_decline with
      | Some classified ->
          logf "baseline: DECLINED at %s %s"
            (phase_label classified.phase)
            (Outcome.detail_of_cause classified.cause)
      | None ->
          if baseline_dispatched then (
            logf "baseline: %.4f ms (digest %s)" baseline_ms (dshort base_digest);
            emit_calibration ~backend ~limits ~label:"baseline" ~digest:base_digest
              ~measured_ms:baseline_ms [ base_opt ])
          else (
            (* No calibration row: the model column is only meaningful next to a measurement. *)
            logf
              "baseline: NOT DISPATCHED, binds no hardware dimension on %s -- the whole routine \
               would run in one work-item (gh-ocannl-532) (digest %s)"
              backend (dshort base_digest);
            record_not_dispatched ~origin:"baseline"
              ~detail:
                (Printf.sprintf "the serial baseline binds no hardware dimension on %s (gh-ocannl-532)"
                   backend)));
      let n_timed = ref (if baseline_dispatched then 1 else 0) in
      (* Live search state for an honest partial report. Each counter starts at the amount of work
         completed so far and is updated at its ordinary accounting site below. [best_so_far] is
         updated after every successful timing, including midway through seed enumeration. *)
      let n_mma_proposed = ref 0 and n_mma_timed = ref 0 in
      (* gh-ocannl-546: the crowned candidate's identity, and how close tensorization came to it.
         Labels are keyed by digest rather than carried on the candidate, because the winner is
         picked from the beam pool (and the beam's own expansions time through the same site), so
         the timing site is the one place every timed candidate passes exactly once. *)
      let label_by_digest = Hashtbl.create (module String) in
      if baseline_dispatched then Hashtbl.set label_by_digest ~key:base_digest ~data:"baseline";
      let mma_best_ms = ref Float.infinity in
      let winner_label best_c =
        Option.value_map best_c ~default:"" ~f:(fun c ->
            Option.value (Hashtbl.find label_by_digest c.digest_after) ~default:"")
      in
      let winner_tensorized best_c =
        Option.exists best_c ~f:(fun c -> saved_is_tensorized (flat_schedule c.form))
      in
      let n_model_scored = ref 0 and n_model_pruned = ref 0 in
      let n_fiss_sketch_timed = ref 0 and n_sr_timed = ref 0 in
      let rounds_run = ref 0 in
      let n_sketch_candidates = ref 0
      and n_epilogue_sketch_candidates = ref 0
      and n_fiss_sketch_candidates = ref 0
      and n_split_reduce_candidates = ref 0 in
      let best_so_far = ref (baseline, baseline_ms) in
      let by_time (_, a) (_, b) = Float.compare a b in
      (* gh-ocannl-550: the search's live artifacts are bounded by [beam_width], not by candidates
         processed. [beam] IS the candidate pool — it holds the fastest [beam_width] entries seen so
         far, and [admit] releases whatever falls out of it. It starts with the baseline when the
         baseline is eligible; a declined one contributes no entry, so the beam can be empty and every
         consumer below takes that as "nothing was timed" (gh-ocannl-533).

         Bounding as we go is equivalent to the old "keep every timed candidate, sort, then take
         [beam_width]" — keeping the k smallest incrementally keeps the k smallest overall — with one
         difference: a tie between exactly equal times now resolves by arrival rather than by seed
         order.

         Why bound it at all: a candidate's device buffers are invisible to the OCaml GC, because the
         backends' pool tables root every slab they allocate (see {!Context.release}), so a pool
         holding every ranked candidate holds its device memory too — a cold tf32 gpt2_mini search
         filled a 12 GB card a fifth of the way through and then ran the remaining candidates, its
         winner replay and its fallback compile against a full device. The tune loop is the one place
         that needs no allocator to fix that: it knows each candidate's exact lifetime — timed, then
         dead unless it is a beam survivor. *)
      let beam = ref (Option.to_list (Option.map baseline ~f:(fun b -> (b, baseline_ms)))) in
      (* The beam-expansion round's own bounded accumulator, hoisted to this scope for one reason: the
         exit sweep has to be able to see it. A fatal launch/sync failure part way through a round used
         to abandon up to [beam_width] already-timed survivors that were in neither [beam] nor
         [best_so_far] (gh-ocannl-550, round-three review). Reset at the top of each round. *)
      let round = ref [] in
      (* Set by the exit sweep: past it there is no reader left for any candidate the search compiled,
         so retention stops applying. A flag rather than clearing [best_so_far], which the reports
         still read for the winner's label after the sweep has freed its buffers. *)
      let search_over = ref false in
      let release_candidate c =
        (* Physical identity, not digest: the beam is the authority on what is live, and a released
           candidate's digest deliberately STAYS in [seen] — it must keep deduplicating, and dedup
           cannot resurrect an artifact, since [seen], [timed_ms_by_digest] and [label_by_digest] hold
           strings and floats and never a [compiled]. [best_so_far] is normally the beam's head, but
           it can lag one round behind it (a sub-threshold improvement updates the former and not the
           latter), so it is checked separately. *)
        if
          !search_over
          || not
               (List.exists !beam ~f:(fun (c', _) -> phys_equal c c')
               || List.exists !round ~f:(fun (c', _) -> phys_equal c c')
               || Option.exists (fst !best_so_far) ~f:(phys_equal c))
        then
          (* Best-effort: a failure to free must not replace the candidate's own outcome, and this
             runs on failure paths too, where the device may already be refusing work. Process-fatal
             conditions still propagate. *)
          release_quietly ~what:("candidate " ^ dshort c.digest_after) c.cctx
      in
      let admit entry =
        let kept, evicted = List.split_n (List.sort (entry :: !beam) ~compare:by_time) beam_width in
        beam := kept;
        List.iter evicted ~f:(fun (c, _) -> release_candidate c)
      in
      (* The exit sweep. Once the search has produced its report, the beam survivors and the running
         best have no reader left either — and on the [timing_ctx] path not even the winner does,
         since it is recompiled from the caller's context out of its saved schedule, which is data.
         Ordering matters twice: the sweep must run AFTER the report record has been built (it reads
         [best_so_far]) and BEFORE the compiles that follow it, which are the two the exhausted device
         used to defeat (the winner replay and the untuned-default fallback behind it). *)
      let release_all_candidates ~keep () =
        search_over := true;
        let live =
          List.map !beam ~f:fst @ List.map !round ~f:fst @ Option.to_list (fst !best_so_far)
        in
        beam := [];
        round := [];
        List.iter live ~f:(fun c ->
            if not (List.exists keep ~f:(phys_equal c)) then release_candidate c)
      in
      (* The gh-ocannl-552 reference point. [baseline_ms] is the serial form's time ([infinity] on
         GPU), so it cannot answer "did tuning beat what the user gets without tuning?". The
         untuned default pipeline is already in the pool — the [config_thresholds] seed reproduces
         it exactly — and its measurement is attributed by digest, so a seed that dedups against an
         identical earlier candidate (the timed baseline included, on CPU backends whose config
         thresholds leave the code unparallelized) still reports the time of that code.

         The attribution honors the scheduling gates (Codex P1 on PR #279): the seed reproduces
         [maybe_default_schedules] only on its main path. With automatic scheduling inactive
         ([automatic_gpu_schedule]/[automatic_cpu_schedule] off, or [debug_log_from_routines] on),
         the untuned default IS the unscheduled serial form, so the reference is the base digest —
         timed on CPU, deliberately unmeasured on GPU (gh-ocannl-532). With [schedule_fission]
         off, the untuned default is the whole-routine config-thresholds annotation, which no
         candidate reproduces (the whole-routine presets use [min_parallel:1]): no attribution,
         rather than labeling a differently-scheduled pipeline as the default. *)
      let auto_sched = Sched.automatic_schedule_active ~backend_name:backend in
      let config_seed_is_default = auto_sched && Sched.default_pipeline_fissions () in
      let timed_ms_by_digest = Hashtbl.create (module String) in
      if baseline_dispatched then
        Hashtbl.set timed_ms_by_digest ~key:base_digest ~data:baseline_ms;
      let default_seed_digest = ref (if auto_sched then None else Some base_digest) in
      let default_ms () =
        Option.bind !default_seed_digest ~f:(Hashtbl.find timed_ms_by_digest)
      in
      let partial_emitted = ref false in
      let emit_partial_and_raise (fatal : Outcome.fatal) =
        let summaries = decline_summaries declines in
        let best_c, best_ms = !best_so_far in
        let terminal_failure =
          Some
            {
              phase = fatal.phase;
              candidate = fatal.candidate;
              detail = Exn.to_string fatal.exn;
            }
        in
        let partial_report =
          {
            cache_hit = false;
            candidates_timed = !n_timed;
            candidates_failed = failed_count declines;
            partial = true;
            baseline_declined = Option.is_some baseline_decline;
            declines = summaries;
            terminal_failure;
            rounds_run = !rounds_run;
            sketch_candidates = !n_sketch_candidates;
            epilogue_sketch_candidates = !n_epilogue_sketch_candidates;
            fiss_sketch_candidates = !n_fiss_sketch_candidates;
            fiss_sketch_timed = !n_fiss_sketch_timed;
            split_reduce_candidates = !n_split_reduce_candidates;
            split_reduce_timed = !n_sr_timed;
            mma_candidates = !n_mma_proposed;
            mma_timed = !n_mma_timed;
            model_scored = !n_model_scored;
            model_pruned = !n_model_pruned;
            fissioned = Option.exists best_c ~f:(fun c -> is_fissioned c.form);
            baseline_ms;
            default_ms = default_ms ();
            best_ms;
            best_label = winner_label best_c;
            best_tensorized = winner_tensorized best_c;
            best_mma_statements =
              Option.value_map best_c ~default:0 ~f:(fun c -> List.length c.mma_renders);
            best_mma_scalar_fallbacks = Option.value_map best_c ~default:0 ~f:mma_scalar_fallbacks;
            mma_best_ms = !mma_best_ms;
            best_schedule = Option.value_map best_c ~default:[] ~f:(fun c -> flat_schedule c.form);
          }
        in
        (* Reporting is best-effort on the exceptional path and must not replace the compiler
           failure or its raw backtrace. *)
        partial_emitted := true;
        (try emit_report partial_report
         with report_exn when not (process_fatal_exn report_exn) ->
           Stdio.eprintf "autotune: partial-report callback failed: %s\n%!"
             (Exn.to_string report_exn));
        (* gh-ocannl-550: this arm is over and returns no routine, so every artifact it still holds
           is dead. It matters most exactly here: a caller that CONTAINS this failure per arm
           ([Train.tune_placements]) goes on to search its other arm, and used to do so against a
           device still holding everything this arm had compiled. *)
        release_all_candidates ~keep:[] ();
        Outcome.raise_failure (Outcome.Fatal fatal)
      in
      (* The post-search fallbacks to the untuned default (nothing timed; the winner replay failed
         or degenerated). Through the containment-aware form, so a failure here reports the phase it
         carries — the outer catch-all would otherwise record every one of them as [Transform],
         which for a link failure is simply wrong. The exception the caller sees is unchanged:
         [emit_partial_and_raise] ends in [raise_failure], exactly as [Context.compile] does. *)
      let untuned_default_or_raise () =
        match
          Context.compile_outcome ~provenance:Ir.Schedule_outcome.User_schedule ctx comp bindings
        with
        | Ok result -> result
        | Error (Outcome.Fatal fatal) -> emit_partial_and_raise fatal
        | Error (Outcome.Classified classified) ->
            emit_partial_and_raise
              (Outcome.fatal_of_classified ~candidate:"untuned default fallback" classified)
      in
      let search () =
      (* gh-ocannl-521: tensorized candidates are counted where they are TIMED, not where they are
         enumerated — a family can be seeded in bulk and rejected in bulk at candidate compile, and
         the enumerated count alone reads as coverage it does not have. Both counters are taken HERE
         rather than off [seed_specs], so they cover the same population by construction: the
         cross-segment recombination composite and the beam-expansion candidates also reach
         [try_spec] without appearing in the seed list, and counting only seeds in the denominator
         would let [mma_timed] exceed [mma_candidates] on a multi-segment routine. *)
      let try_spec spec =
        !on_candidate_attempt (spec_label spec);
        if spec_expects_mma spec then Int.incr n_mma_proposed;
        match compile_spec spec with
        | Error (Outcome.Classified classified) ->
            record_decline declines classified;
            logf "%s: FAILED at %s %s" (spec_label spec)
              (phase_label classified.phase)
              (Outcome.detail_of_cause classified.cause);
            None
        | Error (Outcome.Fatal fatal) -> emit_partial_and_raise fatal
        | Ok c ->
            (* Recorded whether or not this compile goes on to be timed: on dedup the code was (or
               will not be) timed under the same digest, and the [default_ms] lookup follows the
               digest, not the seed (gh-ocannl-552). Guarded: the seed is the untuned default only
               when the default pipeline is active and fissions (Codex P1 on PR #279). *)
            (match spec with
            | Fiss (F_preset { block_size = None; privatize = false; config_thresholds = true })
              when config_seed_is_default ->
                default_seed_digest := Some c.digest_after
            | _ -> ());
            if Hash_set.mem seen c.digest_after then (
              logf "%s: dedup (digest %s)" (spec_label spec) (dshort c.digest_after);
              (* gh-ocannl-550: a dedup still PAID for a compile and a link, so it holds a
                 candidate's worth of device buffers — and its identical twin, already in the beam or
                 already released, is the one the search reasons about. This one is dead on arrival.
                 The digest stays in [seen]. *)
              release_candidate c;
              None)
            else if not (dispatchable ~is_gpu c.all_opts) then (
              (* Degenerated to the serial form (gh-ocannl-532): recorded as seen, so an equivalent
                 later candidate dedups rather than re-deriving the same skip. *)
              Hash_set.add seen c.digest_after;
              logf "%s: NOT DISPATCHED, binds no hardware dimension (digest %s)" (spec_label spec)
                (dshort c.digest_after);
              record_not_dispatched ~origin:"candidate"
                ~detail:
                  (Printf.sprintf "%s degenerated to a form binding no hardware dimension"
                     (spec_label spec));
              release_candidate c;
              None)
            else (
              Hash_set.add seen c.digest_after;
              match
                (* The backend's own classifier decides whether a launch or sync failure is this
                   candidate's fault: the driver error is all the evidence there is, and only the
                   backend can read it. With the always-[None] classifier this used to pass, no
                   backend could ever declare one, so every launch failure was fatal by phase
                   default and there was nowhere for a backend to plug one in (gh-ocannl-536; the
                   HIP scratch-overflow arm of gh-ocannl-533 is what fills this seam). The phase
                   reaching the report is the tagged one inside [time_routine], so a report
                   distinguishes a launch refusal from an asynchronous failure at sync.

                   And from the third case, which is not the backend's to judge: [time_routine]'s
                   pre-dispatch validation carries [Preflight] and is contained without asking the
                   classifier (gh-ocannl-564). Tagged [Launch] it was fatal on every C backend, so a
                   scratch context missing one of the caller's initializations condemned the search
                   instead of declining a candidate. *)
                (* The lineage-wide validation is OUTSIDE the boundary (gh-ocannl-569): a poisoned
                   lineage, an uninitialized input and an unexecuted dependency are properties of
                   the context and the computation, so a genuine one fails every candidate of every
                   arm at once. Contained as a decline it is silent — on a backend whose serial
                   baseline is not dispatched (every GPU backend) every candidate declines for the
                   one reason, nothing is timed, and the search ships the untuned default out of an
                   unusable lineage under a report that says it completed. It reaches the caller
                   instead, which is the only party that can fix it.

                   Tagged, though not contained: the tag carries no boundary here, it only labels
                   the phase so the fallback handler at the end of [search] reports a pre-dispatch
                   validation failure as [Preflight] rather than as its [Transform] default. *)
                Outcome.tag Outcome.Preflight (fun () ->
                    Context.check_lineage_runnable c.cctx c.routine);
                Outcome.protect ~classify_backend:(Context.failure_classifier c.cctx)
                  ~provenance:Outcome.Candidate ~phase:Outcome.Launch
                  ~candidate:(spec_label spec) (fun () ->
                    time_routine ~tag_failures:true ~repeats c.cctx c.routine)
              with
              | Ok ms ->
                  Int.incr n_timed;
                  Hashtbl.set timed_ms_by_digest ~key:c.digest_after ~data:ms;
                  Hashtbl.set label_by_digest ~key:c.digest_after ~data:(spec_label spec);
                  if spec_expects_mma spec then Int.incr n_mma_timed;
                  (* Structural, not label-promised, and deliberately a different population from
                     [n_mma_timed]: with [rounds > 0] the beam menu appends a [Tensorize] to a saved
                     or preset incumbent, and the resulting [W_saved]/[F_saved] spec promises
                     nothing in its label — yet it is exactly as tensorized as a sketch seed, and it
                     can win. Keying this on the label would let the placement A/B report "no
                     tensorized candidate was timed" about a search whose winner tensorizes. *)
                  if saved_is_tensorized (flat_schedule c.form) && Float.(ms < !mma_best_ms) then
                    mma_best_ms := ms;
                  logf "%s: %.4f ms (digest %s)" (spec_label spec) ms (dshort c.digest_after);
                  emit_calibration ~backend ~limits ~label:(spec_label spec) ~digest:c.digest_after
                    ~measured_ms:ms c.all_opts;
                  (* The rendering census next to the timing (gh-ocannl-479): a candidate labeled
                     tensorized whose [Tile_mma] statements all declined at emission timed the
                     scalar fallback — report it, or every number off this tuning run inherits the
                     ambiguity. *)
                  let scalar =
                    List.count c.mma_renders ~f:(fun (_, r) ->
                        Ir.C_syntax.equal_mma_rendering r Ir.C_syntax.Mma_scalar_fallback)
                  in
                  let total = List.length c.mma_renders in
                  if scalar > 0 then
                    logf
                      "%s: NOTE %d/%d Tile_mma statement(s) rendered as the lane-0 scalar \
                       fallback                        (config schedule_log_declines=true names \
                       the failed rule)"
                      (spec_label spec) scalar total
                  else if total = 0 && spec_expects_mma spec then
                    logf "%s: NOTE tensorized candidate emitted no Tile_mma statement"
                      (spec_label spec);
                  if Float.(ms < snd !best_so_far) then best_so_far := (Some c, ms);
                  Some (c, ms)
              | Error (Outcome.Classified classified) -> (
                  record_decline declines classified;
                  logf "%s: RUN FAILED at %s %s" (spec_label spec)
                    (phase_label classified.phase)
                    (Outcome.detail_of_cause classified.cause);
                  match classified.execution_effect with
                  | Outcome.No_device_writes ->
                      (* [Context.run] marks a routine executed before the later [sync] can report
                         an asynchronous failure. A rejection the backend proved wrote nothing
                         withdraws that claim, so the next candidate compiled in this lineage does
                         not wait on a routine that never completed. A no-op for a [Preflight]
                         decline, which precedes the dispatch that makes the claim. *)
                      Context.rollback_execution c.cctx (Context.routine_id c.routine);
                      (* gh-ocannl-550: a candidate that failed to run is as dead as one that lost,
                         and on the failure that motivated all of this it is deader — an
                         out-of-memory decline is exactly when the freed buffers are worth most. *)
                      release_candidate c;
                      None
                  | Outcome.Writes_may_have_occurred ->
                      (* Counted once as a decline (its cause is real evidence about the candidate)
                         and then escalated: the timing lineage may hold partially written buffers,
                         and there is no restore API to rebuild its inputs and parameters, so
                         timing the next candidate on it would score suspect data. *)
                      Context.poison_lineage c.cctx
                        ~routine_name:(Context.routine_name c.routine)
                        (Outcome.exception_of_cause classified.cause);
                      (* gh-ocannl-550: the exit sweep in [emit_partial_and_raise] can only reach
                         what the beam or [best_so_far] holds, and this candidate is in neither — it
                         failed before being admitted. Releasing it here is what keeps the in-flight
                         one from outliving the arm, which matters precisely because
                         [Train.tune_placements] CONTAINS this failure and goes on to search its
                         sibling arm on the same device. *)
                      release_candidate c;
                      emit_partial_and_raise
                        (Outcome.fatal_of_classified ~candidate:(spec_label spec) classified))
              | Error (Outcome.Fatal fatal) ->
                  (* An unattributed launch/sync failure says nothing about what the device did, so
                     the lineage is condemned before the exception unwinds — a caller that catches
                     it cannot reuse a ledger claiming the failed routine completed. *)
                  Context.poison_lineage c.cctx ~routine_name:(Context.routine_name c.routine)
                    fatal.Outcome.exn;
                  (* Not in the beam either (see above). *)
                  release_candidate c;
                  emit_partial_and_raise fatal)
      in
      (* gh-ocannl-550: the per-candidate allocation census, on the same [autotune_log] stream as the
         candidate lines it follows, so a growth curve can be read against the classes that produce
         it instead of against wall-clock samples from outside the process. One line per attempt,
         whether the candidate was timed, declined or deduped — a class that grows on the DECLINE
         path is a different bug from one that grows on the timed path, and only per-attempt lines
         distinguish them. The device figure is the backend's own accounting, which the census does
         not replace: it covers pools the shared seam does not allocate (the merge buffer) and, on
         [cc], counts host allocations whose GC finalizer has not yet run. *)
      let try_spec spec =
        let result = try_spec spec in
        (* Gated explicitly, not just by [logf]: [logf]'s arguments are evaluated whether or not the
           flag is on, and both readings here fold a hashtable. *)
        if Lazy.force log_enabled then
          logf "census after %s: %s | device %.1f MiB" (spec_label spec)
            (Ir.Alloc_census.to_string (Ir.Alloc_census.snapshot ()))
            (Float.of_int (Context.get_used_memory search_ctx) /. 1048576.);
        result
      in
      let block_size_presets mk =
        mk None :: (if is_gpu then List.map seed_block_sizes ~f:(fun bs -> mk (Some bs)) else [])
      in
      (* The model pre-filter of the sketch seeding (gh-ocannl-491 task 3): rank each candidate
         family (the whole-routine sketches; each fission segment's sketches) with the roofline
         model and keep the best [keep_fraction] of the scored candidates before any compilation or
         timing. Only candidates the model fully covers are droppable — a candidate without model
         coverage (opaque code, a schedule the model cannot apply, missing envelope constants) is
         always kept, only measured — so the pre-filter never precludes a measured result and its
         outcome is independent of enumeration order. Presets, saved schedules and the baseline are
         never pruned. *)
      let model_prefilter_params ~seg_opt ~family params =
        if Float.(keep_fraction >= 1.) || List.length params <= 1 then params
        else
          let scored =
            List.map params ~f:(fun p ->
                let score =
                  model_score ~static_indices ~limits seg_opt (sketch_schedule ~p seg_opt)
                in
                (p, score))
          in
          n_model_scored := !n_model_scored + List.count scored ~f:(fun (_, s) -> Option.is_some s);
          let kept = model_prefilter ~keep_fraction scored in
          List.iter scored ~f:(fun ((p, s) as entry) ->
              if not (List.mem kept entry ~equal:phys_equal) then (
                Int.incr n_model_pruned;
                logf "model prune (%s, keep %.2f): %s scored %.3e s" family keep_fraction
                  (spec_label (Whole (W_sketch p))) (Option.value_exn s)));
          List.map kept ~f:fst
      in
      let sketch_params =
        model_prefilter_params ~seg_opt:base_opt ~family:"whole-routine"
          (sketch_seed_params ~is_gpu ~is_cpu ~limits base_opt)
      in
      n_sketch_candidates := List.length sketch_params;
      n_epilogue_sketch_candidates := List.count sketch_params ~f:(fun p -> p.sk_epilogue);
      (* Per-fission-segment sketch seeds (the [F_sketch] flavor): heavily fissioned graphs tune per
         segment, where the whole-routine sketches never apply. Enumerate the fission segmentation
         once, on a hermetic copy of the base lowering with the same pipeline settings the candidate
         transform uses ([preset_sched]'s defaults), and detect a matmul site per [`Normal] segment
         — keyed by the segment's structural pre-schedule digest, like [F_saved]. *)
      let fiss_sketch_entries =
        if not (is_gpu || is_cpu) then []
        else
          let scratch =
            {
              base_opt with
              LL.traced_store = Hashtbl.copy base_opt.LL.traced_store;
              LL.optimize_ctx = LL.copy_optimize_ctx base_opt.LL.optimize_ctx;
            }
          in
          let preset seg =
            if is_gpu then Sched.default_gpu ~min_parallel:1 ~limits seg
            else Sched.default_cpu ~min_parallel:1 seg
          in
          let zero_sched tns = if is_gpu then Sched.zero_expansion ~limits tns else [] in
          match
            Sched.fission_scheduled ~promote_locals:is_gpu ~preset ~zero_sched ~static_indices
              scratch
          with
          | exception Outcome.Cause_at _ -> []
          | [] | [ _ ] -> [] (* Unfissioned: the whole-routine sketches cover the site. *)
          | tuples ->
              List.filter_map tuples ~f:(fun (kind, pre, _, _) ->
                  match kind with
                  | `Zeros | `Solo -> None
                  | `Normal -> (
                      match sketch_seed_params ~is_gpu ~is_cpu ~limits pre with
                      | [] -> None
                      | params ->
                          Some
                            ( SC.digest (SC.canonicalize ~static_indices ~with_placements:false pre),
                              pre,
                              params )))
      in
      let fiss_sketch_entries =
        (* Structurally identical segments share a digest — and thus, at apply time, a schedule — so
           keep one entry per digest. *)
        List.fold fiss_sketch_entries ~init:[] ~f:(fun acc ((key, _, _) as e) ->
            if List.exists acc ~f:(fun (k, _, _) -> String.equal k key) then acc else e :: acc)
        |> List.rev
      in
      let fiss_sketch_entries =
        (* Per-segment pre-filtering: each segment's sketches are their own family — cross-segment
           scores are incomparable (different code volumes), and the singles below are also ranked
           per segment by the recombination step. *)
        List.map fiss_sketch_entries ~f:(fun (key, pre, ps) ->
            (key, model_prefilter_params ~seg_opt:pre ~family:("segment " ^ dshort key) ps))
      in
      let fiss_sketch_specs =
        (* Single-segment specs: each parameter set of each keyed segment is proposed alone, every
           other segment falling back to its default preset (an absent key degrades to the preset in
           the transform closure). Any zipping of segments' seeds into shared combos — index
           pairing, or pinning the other segments to their first set — lets one segment's invalid
           seed mask another segment's seeds from ever being timed (observed on cifar_conv: the fc
           matmul's invalid packrest-grid seed masked the conv segments' row-block seed; and a
           segment's FIRST seed can itself be the invalid one, e.g. GPU conv seeds with a companion
           tail). Cross-segment combination is recovered below by recombining each segment's
           best-timed single into one composite candidate. *)
        List.concat_map fiss_sketch_entries ~f:(fun (key, ps) ->
            List.map ps ~f:(fun p -> Fiss (F_sketch [ (key, p) ])))
      in
      n_fiss_sketch_candidates := List.length fiss_sketch_specs;
      (* Split-reduce seeds (gh-ocannl-484 task 3), detected on the base lowering — the prelude
         applies whole-routine, so no segment enumeration is needed first — and proposed as
         single-site candidates over a few [num_blocks] values (the tunable of the family;
         [2*b <= extent] keeps chunks at least two elements, below which the split is all combine
         overhead). On GPU the block loop is the bulk of pass 1's launch parallelism at these
         low-output sites, so the sweep leans larger; the CPU pool saturates at core counts.
         Multi-site combination is recovered below by recombining the best-timed singles. *)
      let sr_ranked =
        if is_gpu || is_cpu then split_reduce_sites ~static_indices base_opt else []
      in
      let sr_sites = List.take sr_ranked max_split_reduce_sites in
      (* The candidate-volume cap binding is an eviction, not a judgement about the site: it was
         reachable and ranked, and lost only to the cap. Record each evicted site in the decline
         census — the gh-ocannl-541 blind spot was exactly a previously-seeded site silently
         dropping out of the proposal set when newly-reachable sites filled the cap. *)
      List.iter (List.drop sr_ranked max_split_reduce_sites) ~f:(fun s ->
          let detail =
            Printf.sprintf "site %s red%d out%d cost%d%s evicted by autotune_split_reduce_max_sites=%d"
              (Ir.Tnode.debug_name s.sr_target) s.sr_red s.sr_out s.sr_cost
              (match List.length s.sr_swaps with
              | 0 -> ""
              | n -> Printf.sprintf " swap%d" n)
              max_split_reduce_sites
          in
          logf "split_reduce: %s" detail;
          record_decline declines
            {
              Outcome.phase = Outcome.Transform;
              cause = Outcome.Seed_evicted { family = "split_reduce"; detail };
              execution_effect = Outcome.No_device_writes;
            });
      let sr_num_blocks = if is_gpu then [ 32; 128; 512 ] else [ 8; 32; 128 ] in
      let sr_specs =
        List.concat_map sr_sites ~f:(fun s ->
            List.filter_map sr_num_blocks ~f:(fun b ->
                if 2 * b <= s.sr_red then Some (Fiss (F_split { sites = [ (s, b) ] })) else None))
      in
      n_split_reduce_candidates := List.length sr_specs;
      let seed_specs =
        block_size_presets (fun block_size -> Whole (W_preset { block_size }))
        @ (if is_gpu || is_cpu then
             (* Each fissioned preset is seeded plain and privatized (the latter dedups away by
                digest when no accumulator is eligible). The [config_thresholds] seeds reproduce the
                untuned default pipeline exactly (plus its privatized variant), so the winner is
                never worse than not tuning — the aggressive [min_parallel:1] presets can all lose
                to it on launch-overhead-bound workloads. *)
             List.concat_map [ false; true ] ~f:(fun privatize ->
                 Fiss (F_preset { block_size = None; privatize; config_thresholds = true })
                 :: block_size_presets (fun block_size ->
                     Fiss (F_preset { block_size; privatize; config_thresholds = false })))
           else [])
        @ List.map sketch_params ~f:(fun p -> Whole (W_sketch p))
        @ fiss_sketch_specs @ sr_specs
      in
      let fiss_single_results = ref [] in
      let sr_single_results = ref [] in
      List.iter seed_specs ~f:(fun spec ->
          let result = try_spec spec in
          (match (spec, result) with
          | Fiss (F_sketch [ (key, p) ]), Some (_, ms) ->
              Int.incr n_fiss_sketch_timed;
              fiss_single_results := (key, (p, ms)) :: !fiss_single_results
          | Fiss (F_sketch _), Some _ -> Int.incr n_fiss_sketch_timed
          | Fiss (F_split { sites = [ (s, b) ] }), Some (_, ms) ->
              Int.incr n_sr_timed;
              sr_single_results := (s, b, ms) :: !sr_single_results
          | Fiss (F_split _), Some _ -> Int.incr n_sr_timed
          | _ -> ());
          Option.iter result ~f:admit);
      (match default_ms () with
      | Some ms -> logf "untuned-default pipeline: %.4f ms (gh-ocannl-552 reference)" ms
      | None ->
          logf
            "untuned-default pipeline: not timed (gated to a form outside the pool, not seeded, \
             failed, or not dispatched)");
      (* Cross-segment recombination: the singles time every parameter set unmasked, but the best
         full routine may sketch several segments at once. One extra composite candidate applies
         each keyed segment's best-timed single simultaneously — informed by the singles' own
         timings, where the full cartesian product would be exponential. *)
      let recombined =
        List.filter_map fiss_sketch_entries ~f:(fun (key, _) ->
            List.filter !fiss_single_results ~f:(fun (k, _) -> String.equal k key)
            |> List.min_elt ~compare:(fun (_, (_, a)) (_, (_, b)) -> Float.compare a b)
            |> Option.map ~f:(fun (_, (p, _)) -> (key, p)))
      in
      if List.length recombined >= 2 then
        Option.iter (try_spec (Fiss (F_sketch recombined))) ~f:(fun timed ->
            Int.incr n_fiss_sketch_timed;
            admit timed);
      (* Multi-site split-reduce recombination: apply each detected site's best-timed [num_blocks]
         simultaneously — the sites are distinct statements, so their preludes compose. Same
         rationale as the sketch recombination above: singles keep every value unmasked, one
         composite recovers the combination. *)
      let recombined =
        List.filter_map sr_sites ~f:(fun s ->
            List.filter !sr_single_results ~f:(fun (s2, _, _) ->
                Idx.equal_symbol s2.sr_axis s.sr_axis)
            |> List.min_elt ~compare:(fun (_, _, a) (_, _, b) -> Float.compare a b)
            |> Option.map ~f:(fun (s2, b, _) -> (s2, b)))
      in
      if List.length recombined >= 2 then
        Option.iter (try_spec (Fiss (F_split { sites = recombined }))) ~f:(fun timed ->
            Int.incr n_sr_timed;
            admit timed);
      (* [None] iff the beam is empty: no candidate timed and the baseline was not eligible (an
         undispatched GPU baseline never enters the beam with a finite rank; a declined one does not
         enter it at all). *)
      let best = ref (List.hd !beam) in
      let continue_ = ref true in
      while !continue_ && !rounds_run < rounds do
        Int.incr rounds_run;
        let cands =
          List.concat_map !beam ~f:(fun (elem, _) ->
              (* On a GPU backend the beam can hold an incumbent that was never dispatched — the
                 serial baseline, whose [infinity] rank keeps it in the pool when fewer than
                 [beam_width] candidates were timed. Expanding it is worthwhile only through the
                 moves that can bind a hardware dimension (the [Tensorize] path the sketch comments
                 describe); every other move provably yields another undispatchable candidate, which
                 [try_spec]'s dispatchability skip drops after paying for its transform, codegen,
                 compile and link (16 such compiles per round on the gh-ocannl-543 chain). Pruned
                 moves are still counted in the census, so the refusal stays visible where it was
                 before. *)
              let elem_dispatchable = dispatchable ~is_gpu elem.all_opts in
              List.concat_map elem.units ~f:(fun u ->
                  List.filter_map (menu ~is_cpu ~is_gpu ~limits u) ~f:(fun op ->
                      if elem_dispatchable || optop_can_bind_hardware op then extend_spec elem u op
                      else (
                        logf "menu prune (cannot parallelize an undispatched incumbent): %s"
                          (optop_family op);
                        record_not_dispatched ~origin:"beam_move"
                          ~detail:
                            (Printf.sprintf
                               "%s on an incumbent binding no hardware dimension cannot bind one \
                                either"
                               (optop_family op));
                        None))))
        in
        (* gh-ocannl-550: bounded like the seed pass, but in a SECOND accumulator, because a round's
           decision compares its own best against the incumbent and, if it wins, replaces the beam
           wholesale — so the previous beam has to stay alive until that decision is taken, and this
           round's also-rans must not (16 compiles per round on the gh-ocannl-543 chain). An evicted
           entry is provably outside [!round] by the time it is released, so [release_candidate]'s
           beam/best check is the whole guard it needs. *)
        round := [];
        let round_admit entry =
          let kept, evicted =
            List.split_n (List.sort (entry :: !round) ~compare:by_time) beam_width
          in
          round := kept;
          List.iter evicted ~f:(fun (c, _) -> release_candidate c)
        in
        List.iter cands ~f:(fun spec -> Option.iter (try_spec spec) ~f:round_admit);
        match !round with
        | [] -> continue_ := false
        | (_, round_best_ms) :: _ ->
            let incumbent_ms = Option.value_map !best ~default:Float.infinity ~f:snd in
            let previous = !beam in
            if Float.(round_best_ms < incumbent_ms *. (1. -. min_progress)) then (
              beam := !round;
              best := List.hd !beam;
              (* The displaced incumbents are dead. *)
              List.iter previous ~f:(fun (c, _) -> release_candidate c))
            else (
              continue_ := false;
              (* The round did not beat the incumbent by enough: the beam is unchanged, so everything
                 this round produced is dead — except a sub-threshold improvement that became
                 [best_so_far], which [release_candidate] keeps and the exit cleanup releases. *)
              let produced = !round in
              round := [];
              List.iter produced ~f:(fun (c, _) -> release_candidate c))
      done;
      let best_c, best_ms =
        match !best with Some (c, ms) -> (Some c, ms) | None -> (None, Float.infinity)
      in
      (* Nothing was timed exactly when every candidate failed and (on GPU) the serial baseline was
         never run — or, since gh-ocannl-533, was itself declined. Nothing measured means nothing to
         cache: a stored entry would pin future processes to a never-timed schedule. *)
      let nothing_timed = Float.is_inf best_ms in
      (if use_cache then
         if nothing_timed then
           logf "nothing was timed: storing no cache entry (gh-ocannl-532)"
         else
           let saved, segments =
             let best_c = Option.value_exn best_c ~message:timed_winner_exists in
             match best_c.form with
             | Whole_saved saved -> (saved, None)
             | Fiss_saved assoc -> ([], Some assoc)
             | Split_saved (prelude, assoc) -> (prelude, Some assoc)
           in
           SC.store ~dir:cache_dir ~key
             {
               SC.version = SC.entry_version;
               backend;
               numerics = SC.numerics_tag ();
               source_digest = base_digest;
               saved;
               segments;
               best_ms;
               baseline_ms;
               default_ms = default_ms ();
               default_fingerprint =
                 Option.map (default_ms ()) ~f:(fun _ ->
                     Sched.default_schedule_fingerprint ~backend_name:backend);
             });
      (* Diagnostic control (config [autotune_log]): compile and time the UNTUNED default pipeline
         in this very process, on the search context — discriminates a genuinely slow winner from
         process-state effects when the winner's code nominally equals the untuned program yet a
         separately-run untuned process measures faster (PR #140 round 6: same digest, 3.4x runtime
         difference across processes on cuda). *)
      (if Lazy.force log_enabled then
         match Context.compile search_ctx comp bindings with
         | cctx, croutine ->
             (match time_routine ~repeats cctx croutine with
             | ms -> logf "untuned-default in-process control: %.4f ms" ms
             | exception exn -> logf "untuned-default control run failed: %s" (Exn.to_string exn));
             (* A diagnostic's artifacts are dead the moment it has printed its number
                (gh-ocannl-550) — and the diagnostic is on exactly when the memory question is being
                measured, so leaving them behind would show up in the very census that reads it.
                Best-effort, like [release_candidate]: this runs after a timing failure the control
                deliberately swallowed, and [release] awaits the device, so a backend still reporting
                that failure must not be allowed to turn a completed search with a valid winner into
                a fatal one. *)
             release_quietly ~what:"the untuned-default control" cctx
         | exception exn -> logf "untuned-default control compile failed: %s" (Exn.to_string exn));
      let completed_report =
        {
          cache_hit = false;
          candidates_timed = !n_timed;
          candidates_failed = failed_count declines;
          partial = false;
          baseline_declined = Option.is_some baseline_decline;
          declines = decline_summaries declines;
          terminal_failure = None;
          rounds_run = !rounds_run;
          sketch_candidates = List.length sketch_params;
          epilogue_sketch_candidates = List.count sketch_params ~f:(fun p -> p.sk_epilogue);
          fiss_sketch_candidates = List.length fiss_sketch_specs;
          fiss_sketch_timed = !n_fiss_sketch_timed;
          split_reduce_candidates = List.length sr_specs;
          split_reduce_timed = !n_sr_timed;
          mma_candidates = !n_mma_proposed;
          mma_timed = !n_mma_timed;
          model_scored = !n_model_scored;
          model_pruned = !n_model_pruned;
          fissioned = Option.exists best_c ~f:(fun c -> is_fissioned c.form);
          baseline_ms;
          default_ms = default_ms ();
          best_ms;
          best_label = winner_label best_c;
          best_tensorized = winner_tensorized best_c;
          best_mma_statements =
            Option.value_map best_c ~default:0 ~f:(fun c -> List.length c.mma_renders);
          best_mma_scalar_fallbacks = Option.value_map best_c ~default:0 ~f:mma_scalar_fallbacks;
          mma_best_ms = !mma_best_ms;
          best_schedule = Option.value_map best_c ~default:[] ~f:(fun c -> flat_schedule c.form);
        }
      in
      let result =
        if nothing_timed then (
          (* Returning the incumbent here would hand the caller the very serial routine this search
             refused to dispatch (gh-ocannl-532) — slower than not tuning at all, and on GPU
             unbounded. The untuned default pipeline is the honest fallback: the same code the
             caller would have compiled without the tuner. *)
          logf "nothing was timed: falling back to the untuned default compile (gh-ocannl-532)";
          release_all_candidates ~keep:[] ();
          untuned_default_or_raise ())
        else
          (* [nothing_timed] is false, so the beam holds a timed winner. *)
          let best_c = Option.value_exn best_c ~message:timed_winner_exists in
          if Option.is_none timing_ctx then (
            (* The winner's own artifacts ARE the return value here; every other candidate is dead. *)
            release_all_candidates ~keep:[ best_c ] ();
            (best_c.cctx, best_c.routine))
          else
          (* The search ran against the scratch lineage; compile the winner from the caller's
             context (like the cache-hit path). Digest mismatch or replay failure falls back to the
             production default schedule. *)
          let spec =
            match best_c.form with
            | Whole_saved saved -> Whole (W_saved saved)
            | Fiss_saved assoc -> Fiss (F_saved assoc)
            | Split_saved (prelude, assoc) -> Fiss (F_split_saved (prelude, assoc))
          in
          (* Nothing the replay needs is an artifact — [spec] above is the winner's saved schedule —
             so the whole beam goes before the compile that reproduces it (gh-ocannl-550). *)
          release_all_candidates ~keep:[] ();
          match compile_spec_real Outcome.Candidate spec with
          | Ok c when not (dispatchable ~is_gpu c.all_opts) ->
              (* Completes the invariant rather than fixing an observed bug: the winner was timed,
                 so it was dispatchable when measured, and the replay is digest-guarded. But this is
                 the last of the three ways [tune] hands back a routine, and none of them may return
                 an unparallelized GPU routine (gh-ocannl-532). The default compile is the same
                 fallback a failed replay takes. *)
              logf "winner replay produced an unparallelized routine, falling back: %s"
                (spec_label spec);
              (* gh-ocannl-550: rejected, so dead — and the fallback compile below wants the memory.
                 Same one-liner as the rejected cache replay above; the pre-replay sweep could not
                 cover this context, which did not exist yet. *)
              release_quietly ~what:"the rejected winner replay" c.cctx;
              untuned_default_or_raise ()
          | Ok c ->
              logf "winner replay ok: %s" (spec_label spec);
              (c.cctx, c.routine)
          | Error (Outcome.Classified classified) ->
              logf "winner replay FAILED (%s), falling back to the default compile: %s"
                (spec_label spec) (Outcome.detail_of_cause classified.cause);
              untuned_default_or_raise ()
          | Error (Outcome.Fatal fatal) -> emit_partial_and_raise fatal
      in
      (result, completed_report)
      in
      let result, completed_report =
        let escaped ~phase exn backtrace =
          if !partial_emitted then Stdlib.Printexc.raise_with_backtrace exn backtrace
          else emit_partial_and_raise { exn; backtrace; phase; candidate = None }
        in
        try search () with
        (* A raise that carries its phase keeps it: the lineage-wide pre-dispatch validation is
           deliberately raised outside the candidate loop's failure boundary (gh-ocannl-569), so it
           arrives here rather than at a classifier, and reporting it under the [Transform] default
           below would tell the caller a validation error was a transform failure. The original
           exception is re-raised, not the wrapper, so the caller still sees its message. *)
        | Outcome.Raised_at (phase, exn, backtrace) -> escaped ~phase exn backtrace
        | exn -> escaped ~phase:Outcome.Transform exn (Stdlib.Printexc.get_raw_backtrace ())
      in
      (* A callback failure on the ordinary completion path is the callback's own exception and
         propagates normally; only fatal-path callbacks are best-effort. But propagating means the
         caller never receives [result], so its buffers become unreachable while the pool table keeps
         rooting them (gh-ocannl-550) — one full winner's footprint per aborted report, which for a
         caller that retries would accumulate exactly like the candidates used to. The exit sweep
         above deliberately kept this one; nothing is keeping it now. *)
      report_or_release completed_report ~result;
      result)
