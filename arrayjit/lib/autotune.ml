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
      for _ = 1 to max 1 repeats do
        let t0 = Unix.gettimeofday () in
        ctx := Context.run !ctx routine;
        Context.sync !ctx;
        let dt = (Unix.gettimeofday () -. t0) *. 1000. in
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
  sk_bm : int;
  sk_bn : int;
  sk_bk : int;
  sk_tm : int;  (** Register-tile factors; unused on CPU. *)
  sk_tn : int;
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

(* The register-blocktiled GPU matmul (schedule_register_matmul.ml): each output dimension split
   twice (block tile -> Grid, register tile -> Workgroup), register loops sunk innermost, operands
   staged through workgroup-shared tiles at the k-block loop, output privatized, register loops
   materially unrolled. The zeroing nest gets the same geometry (barriers need slot-uniform
   workgroup extents). *)
let gpu_sketch_schedule (site : matmul_site) { sk_bm = bm; sk_bn = bn; sk_bk = bk; sk_tm = tm; sk_tn = tn; _ }
    : Sched.schedule =
  if not site.m_zeroed then
    invalid_arg "Autotune sketch: no whole-node Zero_out of the matmul output";
  if Array.length (Lazy.force site.m_d.Ir.Tnode.dims) <> 2 then
    invalid_arg "Autotune sketch: only rank-2 outputs in v1";
  let ez, zsyms = Sched.expand_zero ~tn:site.m_d in
  let zi, zj =
    match zsyms with [ zi; zj ] -> (zi, zj) | _ -> invalid_arg "Autotune sketch: non-2d zero"
  in
  let sp_zi, _, zi_i = Sched.split ~axis:zi ~factor:bm ~outer:LL.Grid ~inner:LL.Serial in
  let sp_zi2, _, _ = Sched.split ~axis:zi_i ~factor:tm ~outer:LL.Workgroup ~inner:LL.Serial in
  let sp_zj, _, zj_i = Sched.split ~axis:zj ~factor:bn ~outer:LL.Grid ~inner:LL.Serial in
  let sp_zj2, _, _ = Sched.split ~axis:zj_i ~factor:tn ~outer:LL.Workgroup ~inner:LL.Serial in
  let sp_i, _, i_i = Sched.split ~axis:site.m_i ~factor:bm ~outer:LL.Grid ~inner:LL.Serial in
  let sp_i2, i_w, i_t = Sched.split ~axis:i_i ~factor:tm ~outer:LL.Workgroup ~inner:LL.Serial in
  let sp_j, j_o, j_i = Sched.split ~axis:site.m_j ~factor:bn ~outer:LL.Grid ~inner:LL.Serial in
  let sp_j2, j_w, j_t = Sched.split ~axis:j_i ~factor:tn ~outer:LL.Workgroup ~inner:LL.Serial in
  let sp_k, k_o, k_i = Sched.split ~axis:site.m_k ~factor:bk ~outer:LL.Serial ~inner:LL.Serial in
  let swaps = sink i_t [ j_o; j_w; j_t; k_o; k_i ] @ sink j_t [ k_o; k_i ] in
  [ ez; sp_zi; sp_zi2; sp_zj; sp_zj2; sp_i; sp_i2; sp_j; sp_j2; sp_k ]
  @ swaps
  @ [
      Sched.Stage
        { source = site.m_a; tile_loops = [ i_w; i_t; k_i ]; shared = true; cooperative = None };
      Sched.Stage
        { source = site.m_b; tile_loops = [ k_i; j_w; j_t ]; shared = true; cooperative = None };
      Sched.Privatize { target = site.m_d; over = k_o };
      Sched.Unroll { axis = i_t; materialize = true };
      Sched.Unroll { axis = j_t; materialize = true };
    ]

(* The CPU operand-packing matmul (schedule_cpu_pack_matmul.ml): all-serial tiling with the tile
   loops sunk to [i_o j_o k_o k_i i_i j_i], operands packed into contiguous stack scratch, output
   privatized across the k-block loop. *)
let cpu_sketch_schedule (site : matmul_site) { sk_bm = bm; sk_bn = bn; sk_bk = bk; _ } :
    Sched.schedule =
  let sp_i, _, i_i = Sched.split ~axis:site.m_i ~factor:bm ~outer:LL.Serial ~inner:LL.Serial in
  let sp_j, j_o, j_i = Sched.split ~axis:site.m_j ~factor:bn ~outer:LL.Serial ~inner:LL.Serial in
  let sp_k, k_o, k_i = Sched.split ~axis:site.m_k ~factor:bk ~outer:LL.Serial ~inner:LL.Serial in
  [ sp_i; sp_j; sp_k ]
  @ sink i_i [ j_o; j_i; k_o; k_i ]
  @ sink j_i [ k_o; k_i; i_i ]
  @ [
      Sched.Stage
        { source = site.m_a; tile_loops = [ i_i; k_i ]; shared = false; cooperative = None };
      Sched.Stage
        { source = site.m_b; tile_loops = [ k_i; j_i ]; shared = false; cooperative = None };
      Sched.Privatize { target = site.m_d; over = k_o };
    ]

let sketch_schedule ~p (opt : LL.optimized) : Sched.schedule =
  match detect_matmul opt.LL.llc with
  | None -> invalid_arg "Autotune sketch: no matmul micro-kernel detected"
  | Some site -> if p.sk_gpu then gpu_sketch_schedule site p else cpu_sketch_schedule site p

(* Sketch seed parameters compatible with the site's extents (dividing tiles: every constructed
   guard folds, and shared staging requires them). *)
let sketch_seed_params ~is_gpu ~is_cpu (opt : LL.optimized) : sketch_params list =
  match detect_matmul opt.LL.llc with
  | None -> []
  | Some site ->
      let divides c n = c <= n && n % c = 0 in
      if is_gpu && site.m_zeroed then
        List.filter_map
          [
            (64, 64, 8, 4, 4); (32, 32, 8, 4, 4); (16, 16, 8, 4, 4); (32, 32, 16, 2, 2);
            (16, 16, 8, 2, 2);
          ]
          ~f:(fun (bm, bn, bk, tm, tn) ->
            if
              divides bm site.m_ni && divides bn site.m_nj && divides bk site.m_nk
              && divides tm bm && divides tn bn
            then Some { sk_gpu = true; sk_bm = bm; sk_bn = bn; sk_bk = bk; sk_tm = tm; sk_tn = tn }
            else None)
      else if is_cpu then
        List.filter_map [ 16; 8 ] ~f:(fun b ->
            if divides b site.m_ni && divides b site.m_nj && divides b site.m_nk then
              Some { sk_gpu = false; sk_bm = b; sk_bn = b; sk_bk = b; sk_tm = 0; sk_tn = 0 }
            else None)
      else []

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

type fiss_flavor = F_preset of { block_size : int option } | F_saved of (string * SC.saved_schedule) list

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

let compile_candidate ~static_indices ~base_digest ~limits ~is_gpu ~is_cpu ctx comp bindings spec :
    (compiled, string) Result.t =
  let check_digest opt =
    let canon = SC.canonicalize ~static_indices opt in
    if not (String.equal (SC.digest canon) base_digest) then
      invalid_arg "Autotune: fresh lowering does not match the tuned code (digest mismatch)";
    canon
  in
  let preset_sched ?block_size opt =
    if is_gpu then Sched.default_gpu ?block_size ~min_parallel:1 ~limits opt
    else if is_cpu then Sched.default_cpu ~min_parallel:1 opt
    else []
  in
  let captured = ref None in
  let compile_ctx () =
    match spec with
    | Whole flavor ->
        let transform opt =
          let canon = check_digest opt in
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
        let transforms opt =
          let (_ : SC.canonical) = check_digest opt in
          let zero_sched tns = if is_gpu then Sched.zero_expansion ~limits tns else [] in
          let preset seg =
            match flavor with
            | F_preset { block_size } -> preset_sched ?block_size seg
            | F_saved assoc -> (
                let seg_canon = SC.canonicalize ~static_indices seg in
                match List.Assoc.find assoc ~equal:String.equal (SC.digest seg_canon) with
                | Some saved -> fst (SC.of_saved seg_canon saved)
                | None -> [])
          in
          let tuples = Sched.fission_scheduled ~preset ~zero_sched ~static_indices opt in
          let posts = List.map tuples ~f:(fun (_, _, _, post) -> post) in
          let units =
            List.filter_map tuples ~f:(fun (kind, pre, sched, post) ->
                match kind with
                | `Zeros | `Solo -> None
                | `Normal ->
                    let pre_canon = SC.canonicalize ~static_indices pre in
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
            (* Structurally identical segments share a digest and hence a schedule; dedup keys. *)
            List.dedup_and_sort
              ~compare:(fun (k1, _) (k2, _) -> String.compare k1 k2)
              (List.map units ~f:(fun u -> (Option.value_exn u.u_key, u.u_saved)))
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
       like the baseline. *)
    if not (is_cpu || is_gpu) then []
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

(* Extend one unit of a compiled candidate with a menu action. *)
let extend_spec (elem : compiled) (u : unit_gen) (op : SC.saved_optop) : spec option =
  match (elem.form, u.u_key) with
  | Whole_saved _, None -> Some (Whole (W_saved (u.u_saved @ [ op ])))
  | Fiss_saved assoc, Some key ->
      let assoc = List.Assoc.remove assoc ~equal:String.equal key in
      Some (Fiss (F_saved ((key, u.u_saved @ [ op ]) :: assoc)))
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
     optimized code for canonicalization. *)
  let base_opt = ref None in
  let bctx, broutine =
    Context.compile
      ~lowered_transform:(fun opt ->
        base_opt := Some opt;
        opt)
      search_ctx comp bindings
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
  let compile_spec =
    compile_candidate ~static_indices ~base_digest ~limits ~is_gpu ~is_cpu search_ctx comp bindings
  in
  (* Winner (and cache-hit) compiles target the caller's context. *)
  let compile_spec_real =
    compile_candidate ~static_indices ~base_digest ~limits ~is_gpu ~is_cpu ctx comp bindings
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
              emit_report
                {
                  cache_hit = true;
                  candidates_timed = 0;
                  candidates_failed = 0;
                  rounds_run = 0;
                  sketch_candidates = 0;
                  fissioned = is_fissioned c.form;
                  baseline_ms = entry.SC.baseline_ms;
                  best_ms = entry.SC.best_ms;
                  best_schedule = flat_schedule c.form;
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
      let block_size_presets mk =
        mk None :: (if is_gpu then List.map seed_block_sizes ~f:(fun bs -> mk (Some bs)) else [])
      in
      let sketch_params = sketch_seed_params ~is_gpu ~is_cpu base_opt in
      let seed_specs =
        block_size_presets (fun block_size -> Whole (W_preset { block_size }))
        @ (if is_gpu || is_cpu then
             block_size_presets (fun block_size -> Fiss (F_preset { block_size }))
           else [])
        @ List.map sketch_params ~f:(fun p -> Whole (W_sketch p))
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
      emit_report
        {
          cache_hit = false;
          candidates_timed = !n_timed;
          candidates_failed = !n_failed;
          rounds_run = !rounds_run;
          sketch_candidates = List.length sketch_params;
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
        | Ok c -> (c.cctx, c.routine)
        | Error _ -> Context.compile ctx comp bindings
