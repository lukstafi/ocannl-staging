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
  sketch_candidates : int;
  fiss_sketch_candidates : int;
  fiss_sketch_timed : int;
  fissioned : bool;
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

(* Fast routines get extra timed runs beyond [repeats], until this much total measured time (or
   [max_timing_runs]): on sub-millisecond kernels a min-of-3 is dominated by launch jitter, and
   the winner selection becomes a lottery — a heavier candidate can be crowned by one lucky
   sample while the true winner's few samples all landed under contention. Noise only ever adds
   time, so min-of-N converges monotonically to the true best case and more samples strictly
   reduce mis-selection; for routines slower than [min_timing_ms / repeats] per run nothing
   changes. *)
let min_timing_ms = 25.
let max_timing_runs = 64

(* [Context.bindings] exposes the routine's live binding refs — restore them after timing
   (Codex P2 on PR #103), or the returned winner would stay bound to the tuner's midpoint test
   values. *)
let time_routine ~repeats cctx routine =
  let saved_bindings = List.map (Context.bindings routine) ~f:(fun (_ss, r) -> (r, !r)) in
  Exn.protect
    ~finally:(fun () -> List.iter saved_bindings ~f:(fun (r, v) -> r := v))
    ~f:(fun () ->
      set_test_bindings routine;
      (* Warmup run: absorbs lazy initialization and fills caches like a steady-state
         iteration. *)
      let ctx = ref (Context.run cctx routine) in
      Context.sync !ctx;
      let best = ref Float.infinity in
      let total = ref 0. in
      let count = ref 0 in
      while
        !count < max 1 repeats
        || (Float.(!total < min_timing_ms) && !count < max_timing_runs)
      do
        (* Monotonic high-resolution clock: on Windows, [Unix.gettimeofday] ticks at ~1 ms, which
           makes sub-millisecond candidates indistinguishable (they all measure 0). *)
        let c0 = Mtime_clock.counter () in
        ctx := Context.run !ctx routine;
        Context.sync !ctx;
        let dt = Mtime.Span.to_float_ns (Mtime_clock.count c0) /. 1e6 in
        total := !total +. dt;
        Int.incr count;
        if Float.(dt < !best) then best := dt
      done;
      !best)

(** {2 Matmul detection and sketch schedules}

    Sketch candidates instantiate the composed matmul pipelines pinned by
    test/operations/schedule_register_matmul.ml (GPU register blocktiling: Split + Swap + shared
    Stage + Privatize + materializing Unroll) and schedule_cpu_pack_matmul.ml (CPU operand
    packing: Split + Swap + non-shared Stage + Privatize), parameterized by tile sizes. Detection
    is permissive — a mis-detected site fails its candidate compile (op preconditions,
    [validate_parallel], hardware limits) and is skipped like any other invalid candidate. *)

type sketch_params = {
  sk_gpu : bool;  (** Register blocktiling with shared staging vs. CPU operand packing. *)
  sk_mma : bool;
      (** Tensorized (tile-MMA) pipeline instead of the scalar blocktiling/packing one: on GPU,
          Split → (optional cooperative shared Stage) → Tensorize targeting [simdgroup_matrix] /
          tensor cores; on cc, the whole-triple [Tile_mma] rendered register-tiled
          (gh-ocannl-469), optionally Grid-parallel over row blocks — or, with [sk_bk > 0], the
          cache-blocked packed composition (packing Stages feeding the register-tiled kernel;
          [cpu_mma_pack_sketch_schedule]), itself optionally Grid-parallel ([sk_grid]: hoisted
          packing runs Grid-outermost; in-kernel packing relies on the renderer's per-chunk tile
          privatization). Seeded directly because the greedy menu cannot reach
          the composition: a bare [Tensorize] from the serial baseline (one simdgroup, everything
          else serial) loses round 1 and the beam discards it before Grid retypes could join
          it. *)
  sk_simd : int;  (** MMA lane width ([hardware_limits.mma_simd_width]); 0 when [not sk_mma]. *)
  sk_bm : int;
  sk_bn : int;
  sk_bk : int;  (** For GPU MMA sketches, [sk_bk = 0] = unstaged (one full-K [Tile_mma] block). *)
  sk_tm : int;  (** Register-tile factors; unused on CPU. *)
  sk_tn : int;
  sk_hoist : bool;
      (** CPU packing only: pack compile-time-constant operands out of the routine, into the
          per-device constant pool (gh-ocannl-470). Proposed alongside the in-kernel packing
          variant so the choice stays measured; applied per operand, only to hoistable
          (known-constant, host-init-backed) sources. *)
  sk_grid : bool;
      (** CPU packed composition only ([sk_mma] with [sk_bk > 0]): split [i] into pool-parallel
          [Grid] row blocks instead of Serial ones. Two shapes, keyed by [sk_hoist]:

          - With [sk_hoist], hoisted-only packing: only hoistable operands are packed (at link
            time, into the constant pool) and the rest are read in place, leaving the kernel body
            all-materialized; the Grid loop stays outermost (one dispatch spanning the whole GEBP
            triple). The typical inference GEMM: activations (in place) x constant weights.
          - Without [sk_hoist], in-kernel packing: the per-row-block A~ packing Stage lands
            inside the Grid body — its tile is privatized to per-chunk block-scope storage by the
            renderer ([C_syntax.parallel_grid_safe]'s privatization rule) — while the B~ panel
            packs at the k-block loop outside the Grid and is read-only inside (shared across the
            row-block chunks, behind a pointer alias under the blocks extension).

          Proposed alongside the serial flavors so the choice stays measured. *)
}

type matmul_site = {
  m_i : Idx.symbol;
  m_j : Idx.symbol;
  m_k : Idx.symbol;
  m_ni : int;
  m_nj : int;
  m_nk : int;
  m_d : Ir.Tnode.t;
  m_a : Ir.Tnode.t;
  m_b : Ir.Tnode.t;
  m_zeroed : bool;  (** A whole-node [Zero_out] of [m_d] is present (needed by [expand_zero]). *)
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

let detect_matmul (llc : LL.t) : matmul_site option =
  let stmts = strip_stmts (LL.flat_lines [ llc ]) in
  let zeroed = List.filter_map stmts ~f:(function LL.Zero_out tn -> Some tn | _ -> None) in
  List.find_map stmts ~f:(fun stmt ->
      match serial_nest_of stmt with
      | [ (i, ni); (j, nj); (k, nk) ], LL.Set { tn = d; idcs = di; llsc; _ } -> (
          let gets = collect_gets llsc in
          let d_reads, others =
            List.partition_tf gets ~f:(fun (tn, idcs) ->
                phys_equal tn d && Array.equal Idx.equal_axis_index idcs di)
          in
          match (d_reads, others) with
          | _ :: _, [ (t1, i1); (t2, i2) ]
            when idcs_mention di i && idcs_mention di j && not (idcs_mention di k) ->
              let role_a (idcs : Idx.axis_index array) = idcs_mention idcs i && idcs_mention idcs k
              and role_b idcs = idcs_mention idcs k && idcs_mention idcs j in
              let assign =
                if role_a i1 && role_b i2 then Some (t1, t2)
                else if role_a i2 && role_b i1 then Some (t2, t1)
                else None
              in
              Option.map assign ~f:(fun (m_a, m_b) ->
                  {
                    m_i = i;
                    m_j = j;
                    m_k = k;
                    m_ni = ni;
                    m_nj = nj;
                    m_nk = nk;
                    m_d = d;
                    m_a;
                    m_b;
                    m_zeroed = List.exists zeroed ~f:(phys_equal d);
                  })
          | _ -> None)
      | _ -> None)

let sink sym below = List.map below ~f:(fun inner -> Sched.Swap { outer = sym; inner })

(* Zero-geometry ops shared by the sketch pipelines: expand the whole-node [Zero_out] of the
   output and give the resulting nest a compatible parallel geometry, via [mk_zops] on its two
   fresh loop symbols. When the site is NOT zeroed — a fission segment's site never is, the
   [Zero_out] lands in its own [`Zeros] segment — there is nothing to expand and the pipelines
   are correct without it: [Privatize] init-loads the accumulator tile from the (pre-zeroed)
   target, and [Tile_mma] loads the accumulator fragment before the reduction. *)
let zero_geometry (site : matmul_site)
    ~(mk_zops : zi:Idx.symbol -> zj:Idx.symbol -> Sched.schedule) : Sched.schedule =
  if not site.m_zeroed then []
  else (
    if Array.length (Lazy.force site.m_d.Ir.Tnode.dims) <> 2 then
      invalid_arg "Autotune sketch: only rank-2 outputs in v1";
    let ez, zsyms = Sched.expand_zero ~tn:site.m_d in
    let zi, zj =
      match zsyms with [ zi; zj ] -> (zi, zj) | _ -> invalid_arg "Autotune sketch: non-2d zero"
    in
    ez :: mk_zops ~zi ~zj)

(* The register-blocktiled GPU matmul (schedule_register_matmul.ml): each output dimension split
   twice (block tile -> Grid, register tile -> Workgroup), register loops sunk innermost, operands
   staged through workgroup-shared tiles at the k-block loop, output privatized, register loops
   materially unrolled. The zeroing nest gets the same geometry (barriers need slot-uniform
   workgroup extents). *)
let gpu_sketch_schedule (site : matmul_site) { sk_bm = bm; sk_bn = bn; sk_bk = bk; sk_tm = tm; sk_tn = tn; _ }
    : Sched.schedule =
  let zops =
    zero_geometry site ~mk_zops:(fun ~zi ~zj ->
        let sp_zi, _, zi_i = Sched.split ~axis:zi ~factor:bm ~outer:LL.Grid ~inner:LL.Serial in
        let sp_zi2, _, _ = Sched.split ~axis:zi_i ~factor:tm ~outer:LL.Workgroup ~inner:LL.Serial in
        let sp_zj, _, zj_i = Sched.split ~axis:zj ~factor:bn ~outer:LL.Grid ~inner:LL.Serial in
        let sp_zj2, _, _ = Sched.split ~axis:zj_i ~factor:tn ~outer:LL.Workgroup ~inner:LL.Serial in
        [ sp_zi; sp_zi2; sp_zj; sp_zj2 ])
  in
  let sp_i, _, i_i = Sched.split ~axis:site.m_i ~factor:bm ~outer:LL.Grid ~inner:LL.Serial in
  let sp_i2, i_w, i_t = Sched.split ~axis:i_i ~factor:tm ~outer:LL.Workgroup ~inner:LL.Serial in
  let sp_j, j_o, j_i = Sched.split ~axis:site.m_j ~factor:bn ~outer:LL.Grid ~inner:LL.Serial in
  let sp_j2, j_w, j_t = Sched.split ~axis:j_i ~factor:tn ~outer:LL.Workgroup ~inner:LL.Serial in
  let sp_k, k_o, k_i = Sched.split ~axis:site.m_k ~factor:bk ~outer:LL.Serial ~inner:LL.Serial in
  let swaps = sink i_t [ j_o; j_w; j_t; k_o; k_i ] @ sink j_t [ k_o; k_i ] in
  zops
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
        };
      Sched.Stage
        {
          source = site.m_b;
          tile_loops = [ k_i; j_w; j_t ];
          shared = true;
          cooperative = None;
          hoisted = false;
        };
      Sched.Privatize { target = site.m_d; over = k_o };
      Sched.Unroll { axis = i_t; materialize = true };
      Sched.Unroll { axis = j_t; materialize = true };
    ]

(* A constant operand eligible for hoisted (out-of-routine) packing (gh-ocannl-470). The same
   predicate enters the canonical digest ([Schedule_cache.canonicalize]), so a cached winner for
   a same-shape program of different operand constancy never replays here — hoisted candidates
   are always measured for constant sites. *)
let hoistable = Sched.hoistable_constant

(* The CPU operand-packing matmul (schedule_cpu_pack_matmul.ml): all-serial tiling with the tile
   loops sunk to [i_o j_o k_o k_i i_i j_i], operands packed into contiguous stack scratch, output
   privatized across the k-block loop. With [sk_hoist], constant operands are instead packed once
   at link time into the per-device constant pool. *)
let cpu_sketch_schedule (site : matmul_site)
    { sk_bm = bm; sk_bn = bn; sk_bk = bk; sk_hoist; _ } : Sched.schedule =
  let sp_i, _, i_i = Sched.split ~axis:site.m_i ~factor:bm ~outer:LL.Serial ~inner:LL.Serial in
  let sp_j, j_o, j_i = Sched.split ~axis:site.m_j ~factor:bn ~outer:LL.Serial ~inner:LL.Serial in
  let sp_k, k_o, k_i = Sched.split ~axis:site.m_k ~factor:bk ~outer:LL.Serial ~inner:LL.Serial in
  [ sp_i; sp_j; sp_k ]
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
        };
      Sched.Stage
        {
          source = site.m_b;
          tile_loops = [ k_i; j_i ];
          shared = false;
          cooperative = None;
          hoisted = sk_hoist && hoistable site.m_b;
        };
      Sched.Privatize { target = site.m_d; over = k_o };
    ]

(* Tensorized (tile-MMA) GPU matmul (docs/proposals/tensorize-mma.md; the pinned pipelines of
   schedule_mma_matmul.ml): Split the output dims into Grid blocks, then [Tensorize] the inner
   micro-kernel into a [Tile_mma] block statement. Stage-only composition — [Privatize] must NOT
   join it: it would relocate the accumulator into thread-local scratch, which the MMA loads
   cannot address ([mma_syntax] declines thread-space operands, silently costing the whole
   tensorization), and [Tile_mma]'s block semantics already keep the accumulator fragments
   register-resident across the reduction. With [sk_bk = 0] the single block statement spans the
   full reduction, streaming operand tiles from device memory and amortizing [d] traffic
   entirely; with [sk_bk > 0] both operands are staged through cooperative shared tiles at the
   k-block loop (lane-aware Stage), costing one [d] fragment load/store per k-block. The zeroing
   nest mirrors the accumulation's grid geometry, with an inner Workgroup loop of extent
   [sk_simd] covering the lane slot (barrier-strength uniformity: every workgroup extent must
   equal the lane width once a [Tile_mma] is present) — the seeds constrain [sk_bn = sk_simd] so
   the zeroing's grid blocks align with [j]'s. *)
let gpu_mma_sketch_schedule (site : matmul_site)
    { sk_bm = bm; sk_bn = bn; sk_bk = bk; sk_simd = w; _ } : Sched.schedule =
  let zops =
    zero_geometry site ~mk_zops:(fun ~zi ~zj ->
        let sp_zi, _, _ = Sched.split ~axis:zi ~factor:bm ~outer:LL.Grid ~inner:LL.Serial in
        let sp_zj, _, _ = Sched.split ~axis:zj ~factor:w ~outer:LL.Grid ~inner:LL.Workgroup in
        [ sp_zi; sp_zj ])
  in
  let sp_i, _, i_i = Sched.split ~axis:site.m_i ~factor:bm ~outer:LL.Grid ~inner:LL.Serial in
  let sp_j, j_o, j_i = Sched.split ~axis:site.m_j ~factor:bn ~outer:LL.Grid ~inner:LL.Serial in
  if bk = 0 then
    let tz, _lane = Sched.tensorize ~i:i_i ~j:j_i ~k:site.m_k ~simd_width:w in
    zops @ [ sp_i; sp_j ] @ sink i_i [ j_o ] @ [ tz ]
  else
    let sp_k, k_o, k_i = Sched.split ~axis:site.m_k ~factor:bk ~outer:LL.Serial ~inner:LL.Serial in
    let tz, _lane = Sched.tensorize ~i:i_i ~j:j_i ~k:k_i ~simd_width:w in
    zops
    @ [ sp_i; sp_j; sp_k ]
    @ sink i_i [ j_o ]
    @ sink j_i [ k_o ]
    @ sink i_i [ k_o ]
    @ [
        Sched.Stage
          {
            source = site.m_a;
            tile_loops = [ i_i; k_i ];
            shared = true;
            cooperative = Some w;
            hoisted = false;
          };
        Sched.Stage
          {
            source = site.m_b;
            tile_loops = [ k_i; j_i ];
            shared = true;
            cooperative = Some w;
            hoisted = false;
          };
        tz;
      ]

(* Whole-triple tensorized CPU matmul (gh-ocannl-469; bin/schedule_bench.ml's [tensorize]
   variant): one [Tile_mma] statement the C backends render tinyBLAS-style — the C-tile in an
   RM×RN grid of vector registers held across the k-loop, edges peeled. The zeroing's column
   loop becomes the Workgroup axis with the lane width matching its extent (coverage rule; the
   lane loop renders serially on the C backends). With [sk_bm > 0] the row loops split into
   pool-parallel Grid blocks; [sk_bm = 0] keeps the single-statement form. *)
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
    zops @ [ tz ]
  else
    let sp_i, _, i_i = Sched.split ~axis:site.m_i ~factor:bm ~outer:LL.Grid ~inner:LL.Serial in
    let tz, _lane = Sched.tensorize ~i:i_i ~j:site.m_j ~k:site.m_k ~simd_width:site.m_nj in
    zops @ [ sp_i; tz ]

(* Cache-blocked, operand-packed tensorized CPU matmul: [Tile_mma] composed with the S4 packing
   pipeline (the remaining piece of gh-ocannl-469). GEBP loop structure, all-Serial:
   [j_o? { k_o { pack B~[bk x bn]; i_o { pack A~[bm x bk]; Tile_mma(bm, bn, bk) } } }] — the
   packing [Stage]s land at their own anchors (B~ at [k_o], once per (j_o, k_o) block; A~ at
   [i_o]) and the register-tiled micro-kernel streams the contiguous, cache-resident tiles
   ([lda = bk], [ldb = bn]). [tile_loops] are passed in micro-kernel order ([k_i; j_i] for B),
   so a transposed source packs into the normalized layout and [Tensorize] sees
   [ta = tb = false]. [sk_bn = 0] leaves [j] unsplit (one B~ row panel of [bk x nj] per k-block).
   The lane width is 1: the C backends render the lane loop serially, and a unit lane keeps the
   kernel's parallel geometry trivial. Hoisted packing (constant operands, gh-ocannl-470) is
   proposed per operand like the scalar S4 pipeline.

   With [sk_grid], the row-block loop [i_o] is [Grid]-typed and pool-parallelizes; the whole-node
   [Zero_out] of the output — no longer legal beside a hardware-annotated loop
   ([validate_parallel]) — expands into a nest whose row loop Grid-splits with the same [bm]
   geometry ([zero_geometry]; the unit-lane Workgroup axis has extent 1, stays inactive, and
   needs no coverage from the zeroing nest). Two shapes (see [sk_grid]):

   - [sk_hoist]: hoisted-only packing — hoistable operands are packed at link time into the
     constant pool, the rest are read in place, so the kernel body touches only materialized
     buffers; the Grid loop stays outermost (one dispatch spanning the whole GEBP triple). The
     typical inference GEMM: activations (in place) x constant weights (hoisted-packed panel).
   - Otherwise, in-kernel packing: [i_o] sinks under [j_o]/[k_o] exactly as in the serial shape,
     so the B~ panel packs outside the Grid body (read-only inside, shared across the row-block
     chunks) while the per-row-block A~ tile is privatized to per-chunk block-scope storage by
     the renderer ([C_syntax.parallel_grid_safe]'s privatization rule). *)
let cpu_mma_pack_sketch_schedule (site : matmul_site)
    { sk_bm = bm; sk_bn = bn; sk_bk = bk; sk_hoist; sk_grid; _ } : Sched.schedule =
  let outer_i = if sk_grid then LL.Grid else LL.Serial in
  let grid_outermost = sk_grid && sk_hoist in
  let sp_i, i_o, i_i = Sched.split ~axis:site.m_i ~factor:bm ~outer:outer_i ~inner:LL.Serial in
  let sp_k, k_o, k_i = Sched.split ~axis:site.m_k ~factor:bk ~outer:LL.Serial ~inner:LL.Serial in
  let splits, j_col, j_swaps =
    if bn = 0 then ([ sp_i; sp_k ], site.m_j, [])
    else
      let sp_j, j_o, j_i = Sched.split ~axis:site.m_j ~factor:bn ~outer:LL.Serial ~inner:LL.Serial in
      ( [ sp_i; sp_j; sp_k ],
        j_i,
        sink i_i [ j_o ] @ if grid_outermost then [] else sink i_o [ j_o ] )
  in
  let stage ~hoisted source tile_loops =
    Sched.Stage { source; tile_loops; shared = false; cooperative = None; hoisted }
  in
  let stages =
    if grid_outermost then
      List.filter_map
        [ (site.m_b, [ k_i; j_col ]); (site.m_a, [ i_i; k_i ]) ]
        ~f:(fun (src, tls) -> if hoistable src then Some (stage ~hoisted:true src tls) else None)
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
  let tz, _lane = Sched.tensorize ~i:i_i ~j:j_col ~k:k_i ~simd_width:1 in
  zops @ splits @ j_swaps
  @ sink j_col [ k_o ]
  @ sink i_i [ k_o ]
  @ (if grid_outermost then [] else sink i_o [ k_o ])
  @ stages @ [ tz ]

let sketch_schedule ~p (opt : LL.optimized) : Sched.schedule =
  match detect_matmul opt.LL.llc with
  | None -> invalid_arg "Autotune sketch: no matmul micro-kernel detected"
  | Some site ->
      if p.sk_mma then
        if p.sk_gpu then gpu_mma_sketch_schedule site p
        else if p.sk_bk > 0 then cpu_mma_pack_sketch_schedule site p
        else cpu_mma_sketch_schedule site p
      else if p.sk_gpu then gpu_sketch_schedule site p
      else cpu_sketch_schedule site p

(* Sketch seed parameters compatible with the site's extents (dividing tiles: every constructed
   guard folds, and shared staging requires them). Unzeroed sites — the norm for fission
   segments, whose [Zero_out] lives in its own [`Zeros] segment — are proposable too: the
   pipelines skip the zero geometry (see [zero_geometry]), and a site whose kernel-mates cannot
   share the parallel geometry merely fails its candidate compile. *)
let sketch_seed_params ~is_gpu ~is_cpu ~(limits : Ir.Backend_intf.hardware_limits)
    (opt : LL.optimized) : sketch_params list =
  match detect_matmul opt.LL.llc with
  | None -> []
  | Some site ->
      let divides c n = c <= n && n % c = 0 in
      let blocktile =
        if is_gpu then
          List.filter_map
            [
              (64, 64, 8, 4, 4); (32, 32, 8, 4, 4); (16, 16, 8, 4, 4); (32, 32, 16, 2, 2);
              (16, 16, 8, 2, 2);
            ]
            ~f:(fun (bm, bn, bk, tm, tn) ->
              if
                divides bm site.m_ni && divides bn site.m_nj && divides bk site.m_nk
                && divides tm bm && divides tn bn
              then
                Some
                  {
                    sk_gpu = true;
                    sk_mma = false;
                    sk_simd = 0;
                    sk_bm = bm;
                    sk_bn = bn;
                    sk_bk = bk;
                    sk_tm = tm;
                    sk_tn = tn;
                    sk_hoist = false;
                    sk_grid = false;
                  }
              else None)
        else if is_cpu then
          let base =
            List.filter_map [ 16; 8 ] ~f:(fun b ->
                if divides b site.m_ni && divides b site.m_nj && divides b site.m_nk then
                  Some
                    {
                      sk_gpu = false;
                      sk_mma = false;
                      sk_simd = 0;
                      sk_bm = b;
                      sk_bn = b;
                      sk_bk = b;
                      sk_tm = 0;
                      sk_tn = 0;
                      sk_hoist = false;
                      sk_grid = false;
                    }
                else None)
          in
          (* Hoisted vs in-kernel packing stays a measured choice (gh-ocannl-470): when a constant
             operand can be packed at link time, propose each tiling in both flavors. *)
          if hoistable site.m_a || hoistable site.m_b then
            base @ List.map base ~f:(fun p -> { p with sk_hoist = true })
          else base
        else []
      in
      let mma =
        match (is_gpu, limits.Ir.Backend_intf.mma) with
          | true, Some { Ir.Backend_intf.mma_simd_width = w; mma_tile = tm_t, tn_t, tk_t } ->
              (* [bn = w] keeps the zeroing's column grid blocks aligned with [j]'s (see
                 [gpu_mma_sketch_schedule]); [bk = 0] = unstaged full-K block. *)
              List.filter_map
                [ (16, w, 0); (32, w, 0); (16, w, 32); (32, w, 32); (32, w, 16) ]
                ~f:(fun (bm, bn, bk) ->
                  if
                    divides bm site.m_ni && divides bn site.m_nj
                    && (bk = 0 || (divides bk site.m_nk && bk % tk_t = 0))
                    && bm % tm_t = 0 && bn % tn_t = 0 && site.m_nk % tk_t = 0
                  then
                    Some
                      {
                        sk_gpu = true;
                        sk_mma = true;
                        sk_simd = w;
                        sk_bm = bm;
                        sk_bn = bn;
                        sk_bk = bk;
                        sk_tm = 0;
                        sk_tn = 0;
                        sk_hoist = false;
                        sk_grid = false;
                      }
                  else None)
          | _ when is_cpu ->
              (* The register-tiled [Tile_mma] rendering needs no MMA units ([limits.mma] is
                 [None] on cc): seed the whole-triple form plus Grid-parallel row-block splits.
                 Ineligible statements (non-f32/f64, transposed B) render the scalar fallback —
                 correct, merely timing like the baseline. *)
              let whole =
                List.filter_map [ 0; 64; 16 ] ~f:(fun bm ->
                    if bm = 0 || divides bm site.m_ni then
                      Some
                        {
                          sk_gpu = false;
                          sk_mma = true;
                          sk_simd = 0;
                          sk_bm = bm;
                          sk_bn = 0;
                          sk_bk = 0;
                          sk_tm = 0;
                          sk_tn = 0;
                          sk_hoist = false;
                          sk_grid = false;
                        }
                    else None)
              in
              (* Cache-blocked packed composition ([cpu_mma_pack_sketch_schedule]; [bk > 0]
                 selects it): [bn = 0] = unsplit column panel. The packed tiles are function-scope
                 stack arrays, so cap their combined footprint — which is also roughly the L2
                 residency the blocking aims for. *)
              let prec_bytes = Ir.Ops.prec_in_bytes (Lazy.force site.m_a.Ir.Tnode.prec) in
              let tile_bytes_cap = 256 * 1024 in
              let packed =
                List.filter_map
                  [ (64, 0, 64); (64, 0, 256); (128, 128, 128); (64, 128, 256); (16, 0, 16) ]
                  ~f:(fun (bm, bn, bk) ->
                    let bn_eff = if bn = 0 then site.m_nj else bn in
                    if
                      divides bm site.m_ni
                      && (bn = 0 || divides bn site.m_nj)
                      && divides bk site.m_nk
                      && ((bm * bk) + (bk * bn_eff)) * prec_bytes <= tile_bytes_cap
                    then
                      Some
                        {
                          sk_gpu = false;
                          sk_mma = true;
                          sk_simd = 0;
                          sk_bm = bm;
                          sk_bn = bn;
                          sk_bk = bk;
                          sk_tm = 0;
                          sk_tn = 0;
                          sk_hoist = false;
                          sk_grid = false;
                        }
                    else None)
              in
              (* Hoisted (link-time) packing stays a measured choice for constant operands, like
                 the scalar S4 pipeline (gh-ocannl-470). And when a hoistable operand exists, the
                 hoisted-only composition pool-parallelizes ([sk_grid && sk_hoist]): hoisted
                 packing emits no in-kernel pack writes, so an outermost Grid split over the row
                 blocks is trivially race-free — pack only the hoistable operand(s), read the
                 rest in place ([cpu_mma_pack_sketch_schedule]). Grid seeds need at least two row
                 blocks (c_syntax.ml [collect_parallel_grid] wants extent >= 2). *)
              let base = packed in
              let packed =
                if hoistable site.m_a || hoistable site.m_b then
                  packed
                  @ List.map base ~f:(fun p -> { p with sk_hoist = true })
                  @ List.filter_map base ~f:(fun p ->
                        if site.m_ni / p.sk_bm >= 2 then
                          Some { p with sk_hoist = true; sk_grid = true }
                        else None)
                else packed
              in
              (* Pool-parallel Grid over the in-kernel packed composition: the renderer
                 privatizes the per-row-block A~ tile to per-chunk storage and shares the
                 read-only B~ panel ([C_syntax.parallel_grid_safe]). A measured choice against
                 the all-Serial and hoisted-only Grid flavors. *)
              let packed =
                packed
                @ List.filter_map base ~f:(fun p ->
                      if site.m_ni / p.sk_bm >= 2 then Some { p with sk_grid = true } else None)
              in
              whole @ packed
          | _ -> []
      in
      blocktile @ mma

(** {2 The privatized fission flavor}

    A variant of the per-segment preset that contracts each materialized read-modify-write
    accumulator into a per-thread register tile ({!Sched.optop.Privatize}) over its serial
    reduction loop. A routine-local accumulator beats a device-memory RMW on every backend, and
    on Metal it additionally sidesteps the volatile-RMW miscompile workaround tax
    (c_syntax.ml [volatile_scalar_rmw]). Detection is permissive: each proposal is validated by
    try-applying against the segment (Privatize's own preconditions — single index vector,
    uniform iteration-invariant guards, etc.), and dropped rather than failing the candidate. *)

let rec subtree_has_hardware_loop (llc : LL.t) =
  match llc with
  | LL.For_loop { axis = LL.Grid | LL.Workgroup | LL.Workgroup_reduce; _ } -> true
  | LL.For_loop { body; _ } -> subtree_has_hardware_loop body
  | LL.Seq (a, b) -> subtree_has_hardware_loop a || subtree_has_hardware_loop b
  | LL.If { body; _ } -> subtree_has_hardware_loop body
  | _ -> false

(* Materialized RMW accumulation sites of the (post-preset) scheduled segment, each paired with
   the outermost enclosing Serial loop eligible to privatize over: the access vector must not
   mention its symbol (so the accumulation is carried across it), and no hardware-typed loop may
   sit inside its subtree (the private tile is per-thread; spanning other threads' iterations
   would store back their elements). *)
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
            LL.equal_axis_type axis LL.Serial
            && from_ = 0
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

(** The preset schedule extended with a [Privatize] per detected accumulator. Proposals are
    detected on the preset-scheduled segment and validated one at a time by re-applying the
    growing schedule; a proposal violating an op precondition is dropped. The exploratory applies
    run against a hermetic copy of the segment: [Privatize] registers its (fresh) tile in the
    traced store and placements, and abandoned tiles would otherwise be emitted as dead local
    declarations when the caller applies the returned schedule to the real segment. *)
let extend_with_privatize ~static_indices sched (seg : LL.optimized) : Sched.schedule =
  let scratch () =
    {
      seg with
      LL.traced_store = Hashtbl.copy seg.LL.traced_store;
      LL.optimize_ctx = LL.copy_optimize_ctx seg.LL.optimize_ctx;
    }
  in
  match Sched.apply ~static_indices sched (scratch ()) with
  | exception _ -> sched
  | post ->
      List.fold (privatize_proposals post) ~init:sched ~f:(fun acc (target, over) ->
          let acc' = acc @ [ Sched.Privatize { target; over } ] in
          match Sched.apply ~static_indices acc' (scratch ()) with
          | (_ : LL.optimized) -> acc'
          | exception _ -> acc)

(** {2 Candidate compilation}

    A candidate is a recipe producing schedules against a {e fresh} lowering: backend [compile]
    re-lowers (with fresh symbols) on every call, so schedules are rebound structurally inside
    the transform closure, after checking the fresh code's canonical digest against the base
    compile's. Whole-routine candidates go through the singular [?lowered_transform] seam;
    fissioned candidates through the plural [?lowered_transforms] seam, with per-segment
    schedules keyed by the pre-schedule segment's canonical digest. *)

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
              contains the behavior the user gets without tuning: on launch-overhead-bound
              workloads the aggressive [min_parallel:1] presets can all lose to it. *)
    }
  | F_saved of (string * SC.saved_schedule) list
  | F_sketch of (string * sketch_params) list
      (** Per-segment matmul sketches: for each listed segment (keyed by its pre-schedule
          structural digest, like [F_saved]), the composed sketch pipeline instantiated with the
          given parameters; every other segment gets the plain default preset — the same
          pipeline the seed-time segment enumeration ran, so the segmentation converges. On a
          key miss (segmentation drift) the candidate degrades to the plain fissioned preset
          and dedups away by digest; unlike [F_saved] it never replays a cache entry, so no
          loud drift guard is needed. *)

type spec = Whole of whole_flavor | Fiss of fiss_flavor

(* The replayable/cacheable description of a compiled candidate. *)
type form = Whole_saved of SC.saved_schedule | Fiss_saved of (string * SC.saved_schedule) list

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
  digest_after : string;
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
let dshort d = String.prefix d 8 ^ "/" ^ String.prefix (Stdlib.Digest.to_hex (Stdlib.Digest.string d)) 8

let bs_label = function None -> "cfg" | Some b -> Int.to_string b

let spec_label = function
  | Whole (W_saved s) -> Printf.sprintf "W_saved[%d ops]" (List.length s)
  | Whole (W_preset { block_size }) -> Printf.sprintf "W_preset[bs=%s]" (bs_label block_size)
  | Whole (W_sketch p) when p.sk_mma ->
      Printf.sprintf "W_sketch[mma-%s %dx%dx%d%s%s%s]"
        (if p.sk_gpu then "gpu" else "cpu")
        p.sk_bm p.sk_bn p.sk_bk
        (if p.sk_bk > 0 then if p.sk_gpu then " staged" else " pack" else "")
        (if p.sk_hoist then " hoist" else "")
        (if p.sk_grid then " grid" else "")
  | Whole (W_sketch p) ->
      Printf.sprintf "W_sketch[%s %dx%dx%d/%dx%d%s]"
        (if p.sk_gpu then "gpu" else "cpu")
        p.sk_bm p.sk_bn p.sk_bk p.sk_tm p.sk_tn
        (if p.sk_hoist then " hoist" else "")
  | Fiss (F_preset { block_size; privatize; config_thresholds }) ->
      Printf.sprintf "F_preset[bs=%s%s%s]" (bs_label block_size)
        (if privatize then " priv" else "")
        (if config_thresholds then " cfg-thresh" else "")
  | Fiss (F_saved assoc) -> Printf.sprintf "F_saved[%d segs]" (List.length assoc)
  | Fiss (F_sketch entries) ->
      Printf.sprintf "F_sketch[%s]"
        (String.concat ~sep:","
           (List.map entries ~f:(fun (_, p) ->
                Printf.sprintf "%s%s %dx%dx%d%s%s%s"
                  (if p.sk_mma then "mma-" else "")
                  (if p.sk_gpu then "gpu" else "cpu")
                  p.sk_bm p.sk_bn p.sk_bk
                  (if p.sk_mma then "" else Printf.sprintf "/%dx%d" p.sk_tm p.sk_tn)
                  (if p.sk_hoist then " hoist" else "")
                  (if p.sk_grid then " grid" else ""))))

(* Every candidate derives its CODE from the ONE base lowering ([base_opt] with [canon] its
   canonical form, captured together in [tune]) rather than from the compile's own fresh
   lowering, whose llc the transform ignores. Re-lowering per candidate was subtly unsound:
   timing runs settle tensor-node value bounds, so later fresh lowerings can fold guards (and
   even re-segment fission) differently from the base — failing digest checks at best (the CUDA
   rounds on PR #140: whole arms degenerating to their serial baselines) and silently replaying
   the winner with empty per-segment schedules at worst (a 296 ms winner returning as a 2614 ms
   routine). Deriving from the base makes candidates and the winner replay drift-immune and
   byte-comparable by construction; the fresh-lowering digest check survives only in spirit via
   the disk cache's [source_digest] guard (cross-process compatibility).

   The rebased code keeps the fresh compile's OWN [optimize_ctx] (the per-compile fork of the
   context's lineage): link-time buffer allocation consults that fork, so placement mutations by
   schedule ops — fission's Local promotions above all — must land there or the allocator would
   miss buffers the kernels reference. Candidate hermeticity is unchanged: each compile forks
   the lineage table anew. The traced store is copied from the base (schedule ops register their
   tiles in it). *)
let compile_candidate ~static_indices ~base_opt ~canon ~limits ~is_gpu ~is_cpu ctx comp bindings
    spec : (compiled, string) Result.t =
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
          let opt' = Sched.apply ~static_indices sched opt in
          let digest_after = SC.digest (SC.canonicalize ~static_indices opt') in
          captured :=
            Some
              ( Whole_saved saved,
                [ { u_key = None; u_saved = saved; u_registry = registry; u_opt = opt' } ],
                digest_after );
          opt'
        in
        Context.compile ~lowered_transform:transform ctx comp bindings
    | Fiss flavor ->
        let transforms fresh =
          let opt = rebase fresh in
          let zero_sched tns = if is_gpu then Sched.zero_expansion ~limits tns else [] in
          (* Per-segment schedule matching keys on the STRUCTURAL canon ([with_placements:false]):
             placement classes can render differently across compilation lineages on
             byte-identical segments (decided in one, undecided in the other — e.g. tuning with
             [timing_ctx]), which used to fail winner replays wholesale. A lookup miss returns
             the empty schedule: [fission_scheduled] probes {e fine} (pre-coalescing) segments
             through this closure, and only the empty-on-miss answer lets coalescing re-converge
             to the saved segmentation, where every final [`Normal] segment's digest hits (the
             verification after fission below catches genuine drift loudly instead of silently
             replaying unscheduled segments). *)
          let seg_key seg = SC.digest (SC.canonicalize ~static_indices ~with_placements:false seg) in
          let preset seg =
            match flavor with
            | F_preset { block_size; privatize; config_thresholds } ->
                let sched = preset_sched ?block_size ~config_thresholds seg in
                if privatize then extend_with_privatize ~static_indices sched seg else sched
            | F_saved entries -> (
                let seg_canon =
                  SC.canonicalize ~static_indices ~with_placements:false seg
                in
                match List.Assoc.find entries ~equal:String.equal (SC.digest seg_canon) with
                | Some saved -> fst (SC.of_saved seg_canon saved)
                | None -> [])
            | F_sketch entries -> (
                match List.Assoc.find entries ~equal:String.equal (seg_key seg) with
                | Some p -> sketch_schedule ~p seg
                | None -> preset_sched seg)
          in
          let tuples =
            (* Match the default pipeline's placements (statement-crossing [Local]s promoted on
               GPU), so fissioned candidates and the untuned baseline schedule the same code. *)
            Sched.fission_scheduled ~promote_locals:is_gpu ~preset ~zero_sched ~static_indices
              opt
          in
          (* Genuine-drift guard for saved replays (cross-process cache entries): with the
             empty-on-miss closure above, a saved winner whose segmentation no longer matches
             would coalesce differently and silently replay some segments unscheduled. Verify
             instead that every final [`Normal] segment found its saved schedule. *)
          (match flavor with
          | F_preset _ | F_sketch _ -> ()
          | F_saved entries ->
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
                    let pre_canon =
                      SC.canonicalize ~static_indices ~with_placements:false pre
                    in
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
               share a key and their saved forms are interchangeable, so duplicates are
               harmless. *)
            List.map units ~f:(fun u -> (Option.value_exn u.u_key, u.u_saved))
          in
          let digest_after =
            String.concat ~sep:"+"
              (List.map posts ~f:(fun post -> SC.digest (SC.canonicalize ~static_indices post)))
          in
          captured := Some (Fiss_saved assoc, units, digest_after);
          posts
        in
        Context.compile ~lowered_transforms:transforms ctx comp bindings
  in
  try
    let cctx, routine = compile_ctx () in
    match !captured with
    | Some (form, units, digest_after) -> Ok { form; cctx; routine; units; digest_after }
    | None -> Error "Autotune: the transform was not invoked"
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
let max_actions_per_unit = 48

let menu ~is_cpu ~is_gpu ~(limits : Ir.Backend_intf.hardware_limits) (u : unit_gen) :
    SC.saved_optop list =
  let loops = collect_loops u.u_registry u.u_opt.LL.llc in
  let splits =
    List.concat_map loops ~f:(fun ld ->
        if not (LL.equal_axis_type ld.ld_axis LL.Serial) then []
        else
          List.filter_map split_factors ~f:(fun factor ->
              if factor < ld.ld_extent && ld.ld_extent % factor = 0 then
                Some (SC.Split { axis = ld.ld_ref; factor; outer = LL.Serial; inner = LL.Serial })
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
    (* CPU renders eligible retyped loops via vector extensions (or vectorization pragmas); GPU
       backends render them as 128-bit packed loads/stores (gh-ocannl-463). Ineligible candidates
       fall back to plain serial loops, so a proposal that fails codegen eligibility merely times
       like the baseline. Accumulating bodies are proposable on CPU (gh-ocannl-468): the renderer
       either emits the reduction-chains rendering or falls back to a plain serial loop — never
       to a vectorization pragma, which would assert iteration independence the loop-carried
       accumulation does not satisfy. On GPU the reduction rendering does not exist (reductions
       parallelize via [Workgroup_reduce] instead), so accumulations stay excluded. *)
    if not (is_cpu || is_gpu) then []
    else
      List.filter_map loops ~f:(fun ld ->
          if
            LL.equal_axis_type ld.ld_axis LL.Serial
            && ld.ld_innermost
            && ((not ld.ld_accumulating) || is_cpu)
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
        List.concat_map (collect_serial_triples u.u_registry u.u_opt.LL.llc)
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
  List.take (tensorizes @ splits @ swaps @ unrolls @ vectorizes) max_actions_per_unit

(* Extend one unit of a compiled candidate with a menu action. The fissioned entries stay in
   segment order (the positional replay fallback relies on it); extending by key updates every
   structurally identical segment — they carry interchangeable saved forms, so extending them
   uniformly keeps the digest lookup and the positional entries consistent. *)
let extend_spec (elem : compiled) (u : unit_gen) (op : SC.saved_optop) : spec option =
  match (elem.form, u.u_key) with
  | Whole_saved _, None -> Some (Whole (W_saved (u.u_saved @ [ op ])))
  | Fiss_saved assoc, Some key ->
      Some
        (Fiss
           (F_saved
              (List.map assoc ~f:(fun (k, s) ->
                   if String.equal k key then (k, u.u_saved @ [ op ]) else (k, s)))))
  | _ -> None

(** {2 The search} *)

let tune ?beam_width ?rounds ?repeats ?seed_block_sizes ?cache_dir ?timing_ctx ?report ctx comp
    bindings =
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
  (* With [timing_ctx], the search (candidate compiles and timing runs) happens against that
     scratch lineage's buffers, and only the winner is compiled from [ctx] — so the timing runs
     never mutate the caller's live state (parameters, accumulators). The scratch context must
     contain the nodes the computation requires from a prior context (e.g. initialized
     parameters), typically by repeating the caller's initialization on a fresh root context. It
     must live on the same backend and device as [ctx] (Codex P2 on PR #109): candidates timed
     elsewhere do not predict this device, and the winner would be cached under this backend's
     key without ever having been timed on it. *)
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
  let search_ctx = Option.value timing_ctx ~default:ctx in
  (* The base compile: identity transform (= the serial baseline candidate), capturing the
     optimized code every candidate derives from (see [compile_candidate]) and its canonical
     form. Canonicalize INSIDE the transform: after the transform returns, codegen forces the
     remaining undecided placements into the very placements table the captured [opt]
     references, and placement classes enter the digest (Schedule_cache.canonicalize) — the
     disk-cache key must be the deterministic transform-time form so that storing and
     replaying processes agree. *)
  let base_capture = ref None in
  let bctx, broutine =
    Context.compile
      ~lowered_transform:(fun opt ->
        base_capture := Some (opt, SC.canonicalize ~static_indices opt);
        opt)
      search_ctx comp bindings
  in
  let base_opt, canon =
    match !base_capture with
    | Some oc -> oc
    | None -> failwith "Autotune.tune: backend compile did not invoke lowered_transform"
  in
  let base_digest = SC.digest canon in
  let use_cache = (not (String.is_empty cache_dir)) && SC.complete canon in
  let key = SC.cache_key canon ~backend in
  let compile_spec =
    compile_candidate ~static_indices ~base_opt ~canon ~limits ~is_gpu ~is_cpu search_ctx comp
      bindings
  in
  (* Winner (and cache-hit) compiles target the caller's context; they replay against the same
     base lowering as the search's candidates. *)
  let compile_spec_real =
    compile_candidate ~static_indices ~base_opt ~canon ~limits ~is_gpu ~is_cpu ctx comp bindings
  in
  let emit_report r = Option.iter report ~f:(fun f -> f r) in
  let flat_schedule = function
    | Whole_saved saved -> saved
    | Fiss_saved assoc -> List.concat_map assoc ~f:snd
  in
  let is_fissioned = function Whole_saved _ -> false | Fiss_saved _ -> true
  in
  let cached =
    if use_cache then
      match SC.lookup ~dir:cache_dir ~key with
      | Some entry when String.equal entry.SC.source_digest base_digest -> (
          let spec =
            match entry.SC.segments with
            | Some assoc -> Fiss (F_saved assoc)
            | None -> Whole (W_saved entry.SC.saved)
          in
          match compile_spec_real spec with
          | Ok c ->
              logf "cache hit: %s (best %.4f ms, baseline %.4f ms)" (spec_label spec)
                entry.SC.best_ms entry.SC.baseline_ms;
              emit_report
                {
                  cache_hit = true;
                  candidates_timed = 0;
                  candidates_failed = 0;
                  rounds_run = 0;
                  sketch_candidates = 0;
                  fiss_sketch_candidates = 0;
                  fiss_sketch_timed = 0;
                  fissioned = is_fissioned c.form;
                  baseline_ms = entry.SC.baseline_ms;
                  best_ms = entry.SC.best_ms;
                  best_schedule = flat_schedule c.form;
                };
              Some (c.cctx, c.routine)
          | Error msg ->
              (* Stale or corrupt entry: fall through to a fresh search. *)
              logf "cache entry replay FAILED, re-searching: %s" msg;
              None)
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
      logf "baseline: %.4f ms (digest %s)" baseline_ms (dshort base_digest);
      let baseline =
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
          digest_after = base_digest;
        }
      in
      let n_timed = ref 1 and n_failed = ref 0 in
      let try_spec spec =
        match compile_spec spec with
        | Error msg ->
            Int.incr n_failed;
            logf "%s: FAILED %s" (spec_label spec) msg;
            None
        | Ok c ->
            if Hash_set.mem seen c.digest_after then (
              logf "%s: dedup (digest %s)" (spec_label spec) (dshort c.digest_after);
              None)
            else (
              Hash_set.add seen c.digest_after;
              match time_routine ~repeats c.cctx c.routine with
              | ms ->
                  Int.incr n_timed;
                  logf "%s: %.4f ms (digest %s)" (spec_label spec) ms
                    (dshort c.digest_after);
                  Some (c, ms)
              | exception exn ->
                  Int.incr n_failed;
                  logf "%s: RUN FAILED %s" (spec_label spec) (Exn.to_string exn);
                  None)
      in
      let block_size_presets mk =
        mk None :: (if is_gpu then List.map seed_block_sizes ~f:(fun bs -> mk (Some bs)) else [])
      in
      let sketch_params = sketch_seed_params ~is_gpu ~is_cpu ~limits base_opt in
      (* Per-fission-segment sketch seeds (the [F_sketch] flavor): heavily fissioned graphs tune
         per segment, where the whole-routine sketches never apply. Enumerate the fission
         segmentation once, on a hermetic copy of the base lowering with the same pipeline
         settings the candidate transform uses ([preset_sched]'s defaults), and detect a matmul
         site per [`Normal] segment — keyed by the segment's structural pre-schedule digest,
         like [F_saved]. *)
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
          | exception _ -> []
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
                              params )))
      in
      let fiss_sketch_specs =
        (* Index pairing: the n-th spec applies each keyed segment's n-th compatible parameter
           set (its first, when it has fewer) — every parameter set of every segment gets
           proposed while the other segments stay pinned to their preferred tiling. *)
        let n =
          List.fold fiss_sketch_entries ~init:0 ~f:(fun acc (_, ps) -> max acc (List.length ps))
        in
        List.init n ~f:(fun idx ->
            Fiss
              (F_sketch
                 (List.map fiss_sketch_entries ~f:(fun (key, ps) ->
                      (key, Option.value (List.nth ps idx) ~default:(List.hd_exn ps))))))
      in
      let seed_specs =
        block_size_presets (fun block_size -> Whole (W_preset { block_size }))
        @ (if is_gpu || is_cpu then
             (* Each fissioned preset is seeded plain and privatized (the latter dedups away by
                digest when no accumulator is eligible). The [config_thresholds] seeds reproduce
                the untuned default pipeline exactly (plus its privatized variant), so the
                winner is never worse than not tuning — the aggressive [min_parallel:1] presets
                can all lose to it on launch-overhead-bound workloads. *)
             List.concat_map [ false; true ] ~f:(fun privatize ->
                 Fiss (F_preset { block_size = None; privatize; config_thresholds = true })
                 :: block_size_presets (fun block_size ->
                        Fiss (F_preset { block_size; privatize; config_thresholds = false })))
           else [])
        @ List.map sketch_params ~f:(fun p -> Whole (W_sketch p))
        @ fiss_sketch_specs
      in
      let by_time (_, a) (_, b) = Float.compare a b in
      let n_fiss_sketch_timed = ref 0 in
      let pool =
        (baseline, baseline_ms)
        :: List.filter_map seed_specs ~f:(fun spec ->
               let result = try_spec spec in
               (match (spec, result) with
               | Fiss (F_sketch _), Some _ -> Int.incr n_fiss_sketch_timed
               | _ -> ());
               result)
      in
      let beam = ref (List.take (List.sort pool ~compare:by_time) beam_width) in
      let best = ref (List.hd_exn !beam) in
      let rounds_run = ref 0 in
      let continue_ = ref true in
      while !continue_ && !rounds_run < rounds do
        Int.incr rounds_run;
        let cands =
          List.concat_map !beam ~f:(fun (elem, _) ->
              List.concat_map elem.units ~f:(fun u ->
                  List.filter_map (menu ~is_cpu ~is_gpu ~limits u) ~f:(extend_spec elem u)))
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
      (if use_cache then
         let saved, segments =
           match best_c.form with
           | Whole_saved saved -> (saved, None)
           | Fiss_saved assoc -> ([], Some assoc)
         in
         SC.store ~dir:cache_dir ~key
           {
             SC.version = SC.entry_version;
             backend;
             source_digest = base_digest;
             saved;
             segments;
             best_ms;
             baseline_ms;
           });
      (* Diagnostic control (config [autotune_log]): compile and time the UNTUNED default
         pipeline in this very process, on the search context — discriminates a genuinely slow
         winner from process-state effects when the winner's code nominally equals the untuned
         program yet a separately-run untuned process measures faster (PR #140 round 6: same
         digest, 3.4x runtime difference across processes on cuda). *)
      (if Lazy.force log_enabled then
         match Context.compile search_ctx comp bindings with
         | cctx, croutine -> (
             match time_routine ~repeats cctx croutine with
             | ms -> logf "untuned-default in-process control: %.4f ms" ms
             | exception exn -> logf "untuned-default control run failed: %s" (Exn.to_string exn))
         | exception exn -> logf "untuned-default control compile failed: %s" (Exn.to_string exn));
      emit_report
        {
          cache_hit = false;
          candidates_timed = !n_timed;
          candidates_failed = !n_failed;
          rounds_run = !rounds_run;
          sketch_candidates = List.length sketch_params;
          fiss_sketch_candidates = List.length fiss_sketch_specs;
          fiss_sketch_timed = !n_fiss_sketch_timed;
          fissioned = is_fissioned best_c.form;
          baseline_ms;
          best_ms;
          best_schedule = flat_schedule best_c.form;
        };
      if Option.is_none timing_ctx then (best_c.cctx, best_c.routine)
      else
        (* The search ran against the scratch lineage; compile the winner from the caller's
           context (like the cache-hit path). Digest mismatch or replay failure falls back to the
           production default schedule. *)
        let spec =
          match best_c.form with
          | Whole_saved saved -> Whole (W_saved saved)
          | Fiss_saved assoc -> Fiss (F_saved assoc)
        in
        match compile_spec_real spec with
        | Ok c ->
            logf "winner replay ok: %s" (spec_label spec);
            (c.cctx, c.routine)
        | Error msg ->
            logf "winner replay FAILED (%s), falling back to the default compile: %s"
              (spec_label spec) msg;
            Context.compile ctx comp bindings
