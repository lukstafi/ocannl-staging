open Base
module Tn = Tnode
module Idx = Indexing
module LL = Low_level

type mint_role =
  | Split_outer
  | Split_inner
  | Expand_axis of int
  | Tensorize_lane
  | Partition_seg of int
  | Split_reduce_block
  | Split_reduce_inner
  | Split_reduce_combine of int
[@@deriving sexp, compare, equal, hash]

type sym_ref = Base of int | Static of int | Minted of int * mint_role
[@@deriving sexp, compare, equal]

module Sym_ref = struct
  module T = struct
    type t = sym_ref [@@deriving sexp, compare]
  end

  include T
  include Comparator.Make (T)
end

type saved_optop =
  | Split of { axis : sym_ref; factor : int; outer : LL.axis_type; inner : LL.axis_type }
  | Swap of { outer : sym_ref; inner : sym_ref }
  | Retype of { axis : sym_ref; ty : LL.axis_type }
  | Unroll of { axis : sym_ref; materialize : bool }
  | Partition of { axis : sym_ref; breakpoints : int list }
  | Pad of { axis : sym_ref; to_multiple_of : int }
  | Stage of {
      source : int;
      tile_loops : sym_ref list;
      shared : bool;
      cooperative : int option;
      hoisted : bool;
      swizzle : bool; [@sexp.bool]
    }
  | Privatize of { target : int; over : sym_ref }
  | Expand_zero of { tn : int }
  | Tensorize of { i : sym_ref; j : sym_ref; k : sym_ref; simd_width : int }
  | Fuse_epilogue of { target : int; shared : bool }
  | Split_reduce of { axis : sym_ref; target : int; num_blocks : int }
[@@deriving sexp, compare, equal]

type saved_schedule = saved_optop list [@@deriving sexp, compare, equal]

type canonical = {
  digest : string;
  complete : bool;
  base_syms : sym_ref Map.M(Idx.Symbol).t;
      (** Base and Static entries; binder symbols bound by more than one loop are excluded (their
          references would be ambiguous). *)
  tn_refs : int Map.M(Tn).t;
  ref_tns : Tn.t array;
}

let digest c = c.digest
let complete c = c.complete

let tn_of_ref c i =
  if i < 0 || i >= Array.length c.ref_tns then
    invalid_arg
      (Printf.sprintf "Schedule_cache.tn_of_ref: index %d out of range (%d tensor nodes)" i
         (Array.length c.ref_tns))
  else c.ref_tns.(i)

let canonicalize ?(static_indices = []) ?(with_placements = true) (opt : LL.optimized) : canonical =
  let buf = Buffer.create 4096 in
  let add = Buffer.add_string buf in
  let complete = ref true in
  (* The in-scope rendering token per symbol (shadow-aware: inner binders overwrite). *)
  let tokens = Hashtbl.create (module Idx.Symbol) in
  (* The exported resolution: the first binding of each symbol; duplicated binders are dropped. *)
  let first_bind = Hashtbl.create (module Idx.Symbol) in
  let dup_binders = Hash_set.create (module Idx.Symbol) in
  List.iteri static_indices ~f:(fun k ss ->
      let s = ss.Idx.static_symbol in
      Hashtbl.set tokens ~key:s ~data:(Printf.sprintf "s%d" k);
      Hashtbl.set first_bind ~key:s ~data:(Static k));
  let base_count = ref 0 in
  let tn_refs = Hashtbl.create (module Tn) in
  let rev_tns = ref [] in
  let bind_loop s =
    let id = !base_count in
    Int.incr base_count;
    let tok = "b" ^ Int.to_string id in
    if Hashtbl.mem first_bind s then (
      Hash_set.add dup_binders s;
      complete := false)
    else Hashtbl.set first_bind ~key:s ~data:(Base id);
    Hashtbl.set tokens ~key:s ~data:tok;
    tok
  in
  let emit_sym s =
    match Hashtbl.find tokens s with
    | Some tok -> add tok
    | None ->
        complete := false;
        add "?"
  in
  let plc = opt.LL.optimize_ctx.LL.placements in
  let emit_tn tn =
    match Hashtbl.find tn_refs tn with
    | Some i -> add ("t" ^ Int.to_string i)
    | None ->
        let i = Hashtbl.length tn_refs in
        Hashtbl.set tn_refs ~key:tn ~data:i;
        rev_tns := tn :: !rev_tns;
        let dims = Lazy.force tn.Tn.dims in
        (* Hoistability enters the digest alongside dims and precision (gh-ocannl-470, Codex P2 on
           PR #123): a hoisted-[Stage] winner is only valid against constant operands, and a
           non-hoisted winner cached for a same-shape non-constant program must not mask a constant
           program's hoisted candidates — so such programs must not share cache keys. *)
        let hc = if Schedule.hoistable_constant tn then ";const" else "" in
        (* The effective placement class enters the digest too (Codex P1 on PR #140): the optimized
           code can be identical while placements differ — [Local] scratch vs an [On_device] buffer
           — and the generated backend code then differs in kind and performance, so such programs
           must not share cache keys. In particular the placement-A/B arms of
           [Train.tune_placements] would otherwise cache-hit each other's entries whenever their
           code diverges only in placements, skipping the second arm's measurement. Placements of
           nodes reaching the optimized code are settled by the end of the pipeline; render an
           undecided node defensively rather than assert. *)
        let pc =
          (* [with_placements = false] gives the structural identity: placement classes can render
             differently across compilation lineages on byte-identical code (decided in one,
             undecided in the other), so per-segment schedule matching in fissioned replays keys on
             structure only, while disk-cache keys and post-schedule dedup keep the placement-aware
             form. *)
          if not with_placements then ""
          else
            match Tn.Placements.get plc tn with
            | None -> ";u"
            | Some (m, _) -> ";" ^ Sexp.to_string (Tn.sexp_of_memory_mode m)
        in
        add
          (Printf.sprintf "t%d=[%s;%s%s%s]" i
             (String.concat_array ~sep:"," (Array.map dims ~f:Int.to_string))
             (Sexp.to_string (Ops.sexp_of_prec (Lazy.force tn.Tn.storage_prec)))
             hc pc)
  in
  (* Local scope ids come from a process-global counter freshly on each lowering (like loop
     symbols), so digest their first-occurrence alpha index, not the raw id — otherwise sibling
     compiles of local-heavy routines would never agree on a digest (Codex P2 on PR #103). *)
  let scope_alpha = Hashtbl.create (module LL.Scope_id) in
  let emit_scope (id : LL.scope_id) =
    emit_tn id.tn;
    let a = Hashtbl.find_or_add scope_alpha id ~default:(fun () -> Hashtbl.length scope_alpha) in
    add ("." ^ Int.to_string a)
  in
  let emit_idx = function
    | Idx.Fixed_idx i -> add ("#" ^ Int.to_string i)
    | Idx.Iterator s -> emit_sym s
    | Idx.Affine { symbols; offset } ->
        add "(";
        List.iter symbols ~f:(fun (c, s) ->
            add (Int.to_string c ^ "*");
            emit_sym s;
            add "+");
        add (Int.to_string offset ^ ")")
    | Idx.Sub_axis -> add "_"
    | Idx.Concat syms ->
        add "cat(";
        List.iter syms ~f:(fun s ->
            emit_sym s;
            add ",");
        add ")"
  in
  let emit_idcs idcs =
    add "[";
    Array.iter idcs ~f:(fun idx ->
        emit_idx idx;
        add ",");
    add "]"
  in
  let rec emit_scalar (sc : LL.scalar_t) =
    match sc with
    | LL.Local_scope { id; body; orig_indices } ->
        add "scope(";
        emit_scope id;
        add ")";
        emit_idcs orig_indices;
        add "{";
        emit body;
        add "}"
    | LL.Get_local id ->
        add "getl(";
        emit_scope id;
        add ")"
    | LL.Get (tn, idcs) ->
        add "get ";
        emit_tn tn;
        emit_idcs idcs
    | LL.Get_dynamic { tn; idcs; dyn_axis; dyn_value } ->
        add "getd ";
        emit_tn tn;
        emit_idcs idcs;
        add ("/" ^ Int.to_string dyn_axis ^ ":");
        emit_arg dyn_value
    | LL.Get_merge_buffer (tn, idcs) ->
        add "getm ";
        emit_tn tn;
        emit_idcs idcs
    | LL.Ternop (op, a, b, c) ->
        add (Sexp.to_string (Ops.sexp_of_ternop op));
        add "(";
        emit_arg a;
        add ",";
        emit_arg b;
        add ",";
        emit_arg c;
        add ")"
    | LL.Binop (op, a, b) ->
        add (Sexp.to_string (Ops.sexp_of_binop op));
        add "(";
        emit_arg a;
        add ",";
        emit_arg b;
        add ")"
    | LL.Unop (op, a) ->
        add (Sexp.to_string (Ops.sexp_of_unop op));
        add "(";
        emit_arg a;
        add ")"
    | LL.Constant f -> add (Float.to_string f)
    | LL.Constant_bits b -> add ("bits:" ^ Int64.to_string b)
    | LL.Embed_index idx ->
        add "ix:";
        emit_idx idx
  and emit_arg ((sc, prec) : LL.scalar_arg) =
    emit_scalar sc;
    add ("@" ^ Sexp.to_string (Ops.sexp_of_prec prec))
  and emit (llc : LL.t) =
    match llc with
    | LL.Noop -> add "nop;"
    (* Comment text is presentational (routine names, debug names): identical code under different
       names should share one cache entry. *)
    | LL.Comment _ -> add "c;"
    | LL.Staged_compilation _ ->
        complete := false;
        add "staged;"
    | LL.Seq (a, b) ->
        emit a;
        emit b
    | LL.For_loop { index; from_; to_; body; trace_it; axis } ->
        let tok = bind_loop index in
        add
          (Printf.sprintf "for %s=%d..%d@%s%s{" tok from_ to_
             (Sexp.to_string (LL.sexp_of_axis_type axis))
             (if trace_it then "" else "!"));
        emit body;
        add "}"
    | LL.Zero_out tn ->
        add "zero ";
        emit_tn tn;
        add ";"
    | LL.Set { tn; idcs; llsc; debug = _ } ->
        add "set ";
        emit_tn tn;
        emit_idcs idcs;
        add ":=";
        emit_scalar llsc;
        add ";"
    | LL.Set_dynamic { tn; idcs; dyn_axis; dyn_value; llsc; debug = _ } ->
        add "setdyn ";
        emit_tn tn;
        emit_idcs idcs;
        add ("@" ^ Int.to_string dyn_axis ^ "=");
        emit_arg dyn_value;
        add ":=";
        emit_scalar llsc;
        add ";"
    | LL.Set_from_vec { tn; idcs; length; vec_unop; arg; debug = _ } ->
        add ("setv" ^ Int.to_string length ^ " ");
        emit_tn tn;
        emit_idcs idcs;
        add (":=" ^ Sexp.to_string (Ops.sexp_of_vec_unop vec_unop));
        add "(";
        emit_arg arg;
        add ");"
    | LL.Set_local (id, sc) ->
        add "setl ";
        emit_scope id;
        add ":=";
        emit_scalar sc;
        add ";"
    | LL.Declare_local { id; needs_init } ->
        add "decl ";
        emit_scope id;
        add (if needs_init then "0;" else ";")
    | LL.Workgroup_barrier -> add "bar;"
    | LL.If { cond; body } ->
        add "if(";
        emit_arg cond;
        add "){";
        emit body;
        add "}"
    | LL.Tile_mma { d; a; b; ta; tb; m; n; k; ldd; lda; ldb; lane; fallback } ->
        let operand (tn, idcs) =
          emit_tn tn;
          emit_idcs idcs
        in
        add "mma ";
        operand d;
        operand a;
        operand b;
        add (Printf.sprintf " %b %b %d %d %d %d %d %d " ta tb m n k ldd lda ldb);
        emit_sym lane;
        add "{";
        emit fallback;
        add "}"
  in
  emit opt.LL.llc;
  (* Codegen-relevant companions of the code. *)
  add "shared:[";
  Set.iter opt.LL.workgroup_shared ~f:(fun tn ->
      emit_tn tn;
      add ",");
  add "];fragments:[";
  Set.iter opt.LL.simdgroup_fragments ~f:(fun tn ->
      emit_tn tn;
      add ",");
  (* Emitted only when non-empty so pre-swizzle canonical digests (and their caches) stay valid. *)
  if not (Set.is_empty opt.LL.swizzled) then begin
    add "];swizzled:[";
    Set.iter opt.LL.swizzled ~f:(fun tn ->
        emit_tn tn;
        add ",")
  end;
  add "];merge:";
  (match opt.LL.merge_node with None -> add "-" | Some tn -> emit_tn tn);
  add ";";
  let base_syms =
    Hashtbl.fold first_bind
      ~init:(Map.empty (module Idx.Symbol))
      ~f:(fun ~key ~data acc ->
        if Hash_set.mem dup_binders key then acc else Map.set acc ~key ~data)
  in
  {
    digest = Stdlib.Digest.to_hex (Stdlib.Digest.string (Buffer.contents buf));
    complete = !complete;
    base_syms;
    tn_refs =
      Hashtbl.fold tn_refs
        ~init:(Map.empty (module Tn))
        ~f:(fun ~key ~data acc -> Map.set acc ~key ~data);
    ref_tns = Array.of_list_rev !rev_tns;
  }

(** {2 Registries} *)

type registry = {
  canonical : canonical;
  fwd : sym_ref Map.M(Idx.Symbol).t;
  bwd : Idx.symbol Map.M(Sym_ref).t;
  n_ops : int;
}

let base_registry canonical =
  let bwd =
    Map.fold canonical.base_syms
      ~init:(Map.empty (module Sym_ref))
      ~f:(fun ~key ~data acc -> Map.set acc ~key:data ~data:key)
  in
  { canonical; fwd = canonical.base_syms; bwd; n_ops = 0 }

let resolve r s = Map.find r.fwd s
let resolve_tn r tn = Map.find r.canonical.tn_refs tn

let record r sym sref =
  { r with fwd = Map.set r.fwd ~key:sym ~data:sref; bwd = Map.set r.bwd ~key:sref ~data:sym }

let resolve_exn r s =
  match resolve r s with
  | Some sref -> sref
  | None ->
      invalid_arg
        ("Schedule_cache.to_saved: cannot resolve symbol " ^ Idx.symbol_ident s
       ^ " (not a canonical loop binder, static index, or schedule-minted symbol)")

let resolve_tn_exn r tn =
  match resolve_tn r tn with
  | Some i -> i
  | None ->
      invalid_arg
        ("Schedule_cache.to_saved: cannot resolve tensor node " ^ Tn.debug_name tn
       ^ " (absent from the canonicalized code)")

let unresolve_exn r sref =
  match Map.find r.bwd sref with
  | Some s -> s
  | None ->
      invalid_arg
        (Printf.sprintf "Schedule_cache.of_saved: dangling reference %s"
           (Sexp.to_string (sexp_of_sym_ref sref)))

let to_saved r (sched : Schedule.schedule) : saved_schedule * registry =
  let r, rev_saved =
    List.fold sched ~init:(r, []) ~f:(fun (r, acc) op ->
        let idx = r.n_ops in
        let r, saved =
          match op with
          | Schedule.Split { axis; factor; outer; inner; outer_index; inner_index } ->
              let saved = Split { axis = resolve_exn r axis; factor; outer; inner } in
              let r = record r outer_index (Minted (idx, Split_outer)) in
              let r = record r inner_index (Minted (idx, Split_inner)) in
              (r, saved)
          | Schedule.Swap { outer; inner } ->
              (r, Swap { outer = resolve_exn r outer; inner = resolve_exn r inner })
          | Schedule.Retype { axis; ty } -> (r, Retype { axis = resolve_exn r axis; ty })
          | Schedule.Unroll { axis; materialize } ->
              (r, Unroll { axis = resolve_exn r axis; materialize })
          | Schedule.Partition { axis; breakpoints; segment_indices } ->
              let saved = Partition { axis = resolve_exn r axis; breakpoints } in
              let r =
                List.foldi segment_indices ~init:r ~f:(fun j r s ->
                    record r s (Minted (idx, Partition_seg j)))
              in
              (r, saved)
          | Schedule.Pad { axis; to_multiple_of } ->
              (r, Pad { axis = resolve_exn r axis; to_multiple_of })
          | Schedule.Stage { source; tile_loops; shared; cooperative; hoisted; swizzle } ->
              ( r,
                Stage
                  {
                    source = resolve_tn_exn r source;
                    tile_loops = List.map tile_loops ~f:(resolve_exn r);
                    shared;
                    cooperative;
                    hoisted;
                    swizzle;
                  } )
          | Schedule.Privatize { target; over } ->
              (r, Privatize { target = resolve_tn_exn r target; over = resolve_exn r over })
          | Schedule.Expand_zero { tn; indices } ->
              let r =
                List.foldi indices ~init:r ~f:(fun j r s ->
                    record r s (Minted (idx, Expand_axis j)))
              in
              (r, Expand_zero { tn = resolve_tn_exn r tn })
          | Schedule.Tensorize { i; j; k; lane; simd_width } ->
              let saved =
                Tensorize
                  { i = resolve_exn r i; j = resolve_exn r j; k = resolve_exn r k; simd_width }
              in
              (record r lane (Minted (idx, Tensorize_lane)), saved)
          | Schedule.Fuse_epilogue { target; shared } ->
              (r, Fuse_epilogue { target = resolve_tn_exn r target; shared })
          | Schedule.Split_reduce
              { axis; target; num_blocks; block_index; inner_index; combine_indices } ->
              let saved =
                Split_reduce
                  { axis = resolve_exn r axis; target = resolve_tn_exn r target; num_blocks }
              in
              let r = record r block_index (Minted (idx, Split_reduce_block)) in
              let r = record r inner_index (Minted (idx, Split_reduce_inner)) in
              let r =
                List.foldi combine_indices ~init:r ~f:(fun j r s ->
                    record r s (Minted (idx, Split_reduce_combine j)))
              in
              (r, saved)
        in
        ({ r with n_ops = idx + 1 }, saved :: acc))
  in
  (List.rev rev_saved, r)

let of_saved canonical (saved : saved_schedule) : Schedule.schedule * registry =
  let r, rev_sched =
    List.fold saved
      ~init:(base_registry canonical, [])
      ~f:(fun (r, acc) saved_op ->
        let idx = r.n_ops in
        let r, op =
          match saved_op with
          | Split { axis; factor; outer; inner } ->
              let op, outer_index, inner_index =
                Schedule.split ~axis:(unresolve_exn r axis) ~factor ~outer ~inner
              in
              let r = record r outer_index (Minted (idx, Split_outer)) in
              let r = record r inner_index (Minted (idx, Split_inner)) in
              (r, op)
          | Swap { outer; inner } ->
              (r, Schedule.Swap { outer = unresolve_exn r outer; inner = unresolve_exn r inner })
          | Retype { axis; ty } -> (r, Schedule.Retype { axis = unresolve_exn r axis; ty })
          | Unroll { axis; materialize } ->
              (r, Schedule.Unroll { axis = unresolve_exn r axis; materialize })
          | Partition { axis; breakpoints } ->
              let op, segment_indices =
                Schedule.partition ~axis:(unresolve_exn r axis) ~breakpoints
              in
              let r =
                List.foldi segment_indices ~init:r ~f:(fun j r s ->
                    record r s (Minted (idx, Partition_seg j)))
              in
              (r, op)
          | Pad { axis; to_multiple_of } ->
              (r, Schedule.Pad { axis = unresolve_exn r axis; to_multiple_of })
          | Stage { source; tile_loops; shared; cooperative; hoisted; swizzle } ->
              ( r,
                Schedule.Stage
                  {
                    source = tn_of_ref canonical source;
                    tile_loops = List.map tile_loops ~f:(unresolve_exn r);
                    shared;
                    cooperative;
                    hoisted;
                    swizzle;
                  } )
          | Privatize { target; over } ->
              ( r,
                Schedule.Privatize
                  { target = tn_of_ref canonical target; over = unresolve_exn r over } )
          | Expand_zero { tn } ->
              let op, indices = Schedule.expand_zero ~tn:(tn_of_ref canonical tn) in
              let r =
                List.foldi indices ~init:r ~f:(fun j r s ->
                    record r s (Minted (idx, Expand_axis j)))
              in
              (r, op)
          | Tensorize { i; j; k; simd_width } ->
              let op, lane =
                Schedule.tensorize ~i:(unresolve_exn r i) ~j:(unresolve_exn r j)
                  ~k:(unresolve_exn r k) ~simd_width
              in
              (record r lane (Minted (idx, Tensorize_lane)), op)
          | Fuse_epilogue { target; shared } ->
              (r, Schedule.Fuse_epilogue { target = tn_of_ref canonical target; shared })
          | Split_reduce { axis; target; num_blocks } ->
              let op, block_index, inner_index, combine_indices =
                Schedule.split_reduce ~axis:(unresolve_exn r axis)
                  ~target:(tn_of_ref canonical target) ~num_blocks
              in
              let r = record r block_index (Minted (idx, Split_reduce_block)) in
              let r = record r inner_index (Minted (idx, Split_reduce_inner)) in
              let r =
                List.foldi combine_indices ~init:r ~f:(fun j r s ->
                    record r s (Minted (idx, Split_reduce_combine j)))
              in
              (r, op)
        in
        ({ r with n_ops = idx + 1 }, op :: acc))
  in
  (List.rev rev_sched, r)

(** {2 The disk cache} *)

type entry = {
  version : int;
  backend : string;
  source_digest : string;
  saved : saved_schedule;
  segments : (string * saved_schedule) list option; [@sexp.option]
      (** A fissioned winner: per-segment schedules keyed by the {e pre-schedule} segment's
          canonical digest ([saved] is then empty). [None] for whole-routine schedules. *)
  best_ms : float;
  baseline_ms : float;
  default_ms : float option; [@sexp.option]
      (** The untuned default pipeline's measured time from the search that wrote the entry, for
          diagnostics (gh-ocannl-552). [None] when the default seed was not timed, or for entries
          written before this field existed — optional so such entries stay readable without an
          [entry_version] bump. *)
  default_fingerprint : string option; [@sexp.option]
      (** {!Schedule.default_schedule_fingerprint} at store time, present iff [default_ms] is: the
          cache key covers only the source digest and the backend, so a config change can redefine
          what "the default pipeline" means without missing the cache. A replaying process
          compares fingerprints and drops a stale [default_ms] (the schedule itself stays valid —
          only this diagnostic is config-relative). *)
}
[@@deriving sexp]

let entry_version = 3

let sanitize name =
  String.map name ~f:(fun c ->
      if Char.is_alphanum c || Char.equal c '-' || Char.equal c '_' then c else '_')

let cache_key canonical ~backend = canonical.digest ^ "-" ^ sanitize backend

let rec ensure_dir dir =
  if String.is_empty dir || String.equal dir "." || String.equal dir "/" then ()
  else if Stdlib.Sys.file_exists dir then ()
  else (
    ensure_dir (Stdlib.Filename.dirname dir);
    (* Concurrent creators race benignly. *)
    try Stdlib.Sys.mkdir dir 0o777 with Stdlib.Sys_error _ -> ())

let cache_file ~dir ~key = Stdlib.Filename.concat dir (sanitize key ^ ".sexp")

let store ~dir ~key entry =
  ensure_dir dir;
  let file = cache_file ~dir ~key in
  let tmp = file ^ ".tmp" in
  Stdio.Out_channel.write_all tmp ~data:(Sexp.to_string_hum (sexp_of_entry entry));
  try Stdlib.Sys.rename tmp file with Stdlib.Sys_error _ -> ()

let lookup ~dir ~key =
  let file = cache_file ~dir ~key in
  if not (Stdlib.Sys.file_exists file) then None
  else
    try
      let entry = entry_of_sexp (Sexplib.Sexp.load_sexp file) in
      if entry.version = entry_version then Some entry else None
    with _ -> None
