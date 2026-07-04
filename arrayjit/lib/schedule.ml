(** Schedule IR: loop-nest transforms as values, applied to [Low_level.optimized] at the
    [?lowered_transform] seam of backend [compile]. See docs/proposals/schedule-ir-optops.md — that
    document records the normative pass-ordering contract (§2): the schedule runs after the whole
    [optimize_proc] pipeline, transforms fold their own guards by re-running [simplify_llc] (plus
    CSE/hoisting when a transform duplicated code), and there is no re-virtualization. *)

open Base
module Tn = Tnode

type optop =
  | Split of {
      axis : Indexing.symbol;  (** The loop to split, identified by its index symbol. *)
      factor : int;  (** Extent of the new inner loop. *)
      outer : Low_level.axis_type;  (** Axis type of the new outer loop. *)
      inner : Low_level.axis_type;  (** Axis type of the new inner loop. *)
      outer_index : Indexing.symbol;
      inner_index : Indexing.symbol;
    }
  | Swap of { outer : Indexing.symbol; inner : Indexing.symbol }
  | Retype of { axis : Indexing.symbol; ty : Low_level.axis_type }
  | Unroll of { axis : Indexing.symbol; materialize : bool }
[@@deriving sexp_of]

type schedule = optop list [@@deriving sexp_of]

let split ~axis ~factor ~outer ~inner =
  let outer_index = Indexing.get_symbol () and inner_index = Indexing.get_symbol () in
  (Split { axis; factor; outer; inner; outer_index; inner_index }, outer_index, inner_index)

(** {2 Index substitution}

    A symbol is replaced by an affine combination [Σ terms + offset]: [Split] substitutes
    [i := factor*i_o + i_i], materializing [Unroll] substitutes [i := k]. The two places a loop
    symbol can occur are index vectors ([axis_index array]) and [Embed_index] scalars; the
    traversals below cover both. *)

type affine_subst = { terms : (int * Indexing.symbol) list; offset : int }

let rec add_term acc (c, s) =
  match acc with
  | [] -> [ (c, s) ]
  | (c', s') :: tl when Indexing.equal_symbol s s' -> (c + c', s') :: tl
  | hd :: tl -> hd :: add_term tl (c, s)

(* Merge duplicate symbols, drop zero coefficients, and restore the [Fixed_idx] / [Iterator] /
   [Affine] canonical forms ([Affine] must have >1 term or a coefficient other than 0/1). *)
let normalize_affine ~terms ~offset : Indexing.axis_index =
  let terms = List.fold terms ~init:[] ~f:add_term |> List.filter ~f:(fun (c, _) -> c <> 0) in
  match (terms, offset) with
  | [], _ -> Indexing.Fixed_idx offset
  | [ (1, s) ], 0 -> Indexing.Iterator s
  | _ -> Indexing.Affine { symbols = terms; offset }

let subst_axis_index ~sym ~(by : affine_subst) (idx : Indexing.axis_index) : Indexing.axis_index =
  match idx with
  | Indexing.Fixed_idx _ | Indexing.Sub_axis -> idx
  | Indexing.Iterator s ->
      if Indexing.equal_symbol s sym then normalize_affine ~terms:by.terms ~offset:by.offset
      else idx
  | Indexing.Affine { symbols; offset } ->
      if List.exists symbols ~f:(fun (_, s) -> Indexing.equal_symbol s sym) then
        let terms, offset =
          List.fold symbols ~init:([], offset) ~f:(fun (acc, off) (c, s) ->
              if Indexing.equal_symbol s sym then
                (acc @ List.map by.terms ~f:(fun (bc, bs) -> (c * bc, bs)), off + (c * by.offset))
              else (acc @ [ (c, s) ], off))
        in
        normalize_affine ~terms ~offset
      else idx
  | Indexing.Concat _ -> idx (* Eliminated during lowering; scheduled loops never carry these. *)

let rec map_code ~fidx (llc : Low_level.t) : Low_level.t =
  let open Low_level in
  match llc with
  | Noop | Comment _ | Staged_compilation _ | Zero_out _ | Declare_local _ | Workgroup_barrier ->
      llc
  | Seq (a, b) -> Seq (map_code ~fidx a, map_code ~fidx b)
  | For_loop fc -> For_loop { fc with body = map_code ~fidx fc.body }
  | Set { tn; idcs; llsc; debug } ->
      Set { tn; idcs = Array.map idcs ~f:fidx; llsc = map_scalar ~fidx llsc; debug }
  | Set_from_vec { tn; idcs; length; vec_unop; arg = a, p; debug } ->
      Set_from_vec
        { tn; idcs = Array.map idcs ~f:fidx; length; vec_unop; arg = (map_scalar ~fidx a, p); debug }
  | Set_local (id, llsc) -> Set_local (id, map_scalar ~fidx llsc)
  | If { cond = c, p; body } -> If { cond = (map_scalar ~fidx c, p); body = map_code ~fidx body }

and map_scalar ~fidx (llsc : Low_level.scalar_t) : Low_level.scalar_t =
  let open Low_level in
  match llsc with
  | Local_scope { id; body; orig_indices } ->
      Local_scope { id; body = map_code ~fidx body; orig_indices = Array.map orig_indices ~f:fidx }
  | Get_local _ | Constant _ | Constant_bits _ -> llsc
  | Get (tn, idcs) -> Get (tn, Array.map idcs ~f:fidx)
  | Get_dynamic { tn; idcs; dyn_axis; dyn_value = v, p } ->
      Get_dynamic { tn; idcs = Array.map idcs ~f:fidx; dyn_axis; dyn_value = (map_scalar ~fidx v, p) }
  | Get_merge_buffer (tn, idcs) -> Get_merge_buffer (tn, Array.map idcs ~f:fidx)
  | Ternop (op, (a, pa), (b, pb), (c, pc)) ->
      Ternop (op, (map_scalar ~fidx a, pa), (map_scalar ~fidx b, pb), (map_scalar ~fidx c, pc))
  | Binop (op, (a, pa), (b, pb)) -> Binop (op, (map_scalar ~fidx a, pa), (map_scalar ~fidx b, pb))
  | Unop (op, (a, pa)) -> Unop (op, (map_scalar ~fidx a, pa))
  | Embed_index idx -> Embed_index (fidx idx)

(** {2 The transforms} *)

type floop = {
  index : Indexing.symbol;
  from_ : int;
  to_ : int;
  body : Low_level.t;
  trace_it : bool;
  axis : Low_level.axis_type;
}
(* A copy of [For_loop]'s inlined record (which cannot escape its match). *)

let for_loop { index; from_; to_; body; trace_it; axis } =
  Low_level.For_loop { index; from_; to_; body; trace_it; axis }

(* Rewrites the unique statement-level [For_loop] whose index is [sym]. Loops inside [Local_scope]
   bodies are deliberately out of scope: annotated loops there are rejected by
   [validate_parallel], and splitting them has no v1 use case. *)
let rewrite_loop ~what ~sym ~(f : floop -> Low_level.t) (llc : Low_level.t) : Low_level.t =
  let open Low_level in
  let found = ref false in
  let rec go llc =
    match llc with
    | For_loop { index; from_; to_; body; trace_it; axis } when Indexing.equal_symbol index sym ->
        found := true;
        f { index; from_; to_; body; trace_it; axis }
    | For_loop fc -> For_loop { fc with body = go fc.body }
    | Seq (a, b) -> Seq (go a, go b)
    | If { cond; body } -> If { cond; body = go body }
    | other -> other
  in
  let result = go llc in
  if not !found then
    invalid_arg
      (what ^ ": no statement-level For_loop with index " ^ Indexing.symbol_ident sym
     ^ " in this routine");
  result

let apply_op (llc : Low_level.t) (op : optop) : Low_level.t =
  let open Low_level in
  match op with
  | Split { axis; factor; outer; inner; outer_index; inner_index } ->
      rewrite_loop ~what:"Schedule.Split" ~sym:axis llc ~f:(fun fc ->
          if factor <= 0 then invalid_arg "Schedule.Split: factor must be positive";
          if fc.from_ <> 0 then
            invalid_arg
              ("Schedule.Split: loop " ^ Indexing.symbol_ident axis
             ^ " must start at 0 (lowering guarantees this)");
          let n = fc.to_ + 1 in
          let terms = [ (factor, outer_index); (1, inner_index) ] in
          let by = { terms; offset = 0 } in
          let body = map_code ~fidx:(subst_axis_index ~sym:axis ~by) fc.body in
          let body =
            if n % factor = 0 then body
            else
              (* Remainder guard, construct-then-fold: [apply]'s trailing [simplify_llc] erases it
                 when the loop extents prove it (i.e. exactly when [factor] divides [n]; here it
                 does not, so the guard survives interval folding — but a later transform of an
                 enclosing loop can still change the environment, so we keep the uniform
                 discipline of always folding rather than special-casing). *)
              let iprec = Ops.index_prec () in
              let cond =
                Binop
                  ( Ops.Cmplt,
                    (Embed_index (Indexing.Affine { symbols = terms; offset = 0 }), iprec),
                    (Constant (Float.of_int n), iprec) )
              in
              If { cond = (cond, iprec); body }
          in
          For_loop
            {
              index = outer_index;
              from_ = 0;
              to_ = ((n + factor - 1) / factor) - 1;
              axis = outer;
              trace_it = fc.trace_it;
              body =
                For_loop
                  {
                    index = inner_index;
                    from_ = 0;
                    to_ = factor - 1;
                    axis = inner;
                    trace_it = fc.trace_it;
                    body;
                  };
            })
  | Swap { outer; inner } ->
      rewrite_loop ~what:"Schedule.Swap" ~sym:outer llc ~f:(fun ofc ->
          match ofc.body with
          | For_loop { index; from_; to_; body; trace_it; axis }
            when Indexing.equal_symbol index inner ->
              for_loop { index; from_; to_; trace_it; axis; body = for_loop { ofc with body } }
          | _ ->
              invalid_arg
                ("Schedule.Swap: loops " ^ Indexing.symbol_ident outer ^ " and "
               ^ Indexing.symbol_ident inner
               ^ " are not perfectly nested (the outer body must be exactly the inner loop)"))
  | Retype { axis; ty } ->
      rewrite_loop ~what:"Schedule.Retype" ~sym:axis llc ~f:(fun fc ->
          (match ty with
          | Grid | Workgroup | Workgroup_reduce ->
              if fc.from_ <> 0 then
                invalid_arg
                  ("Schedule.Retype: hardware-annotated loop " ^ Indexing.symbol_ident axis
                 ^ " must start at 0")
          | Serial | Unrolled -> ());
          for_loop { fc with axis = ty })
  | Unroll { axis; materialize = false } ->
      rewrite_loop ~what:"Schedule.Unroll" ~sym:axis llc ~f:(fun fc ->
          for_loop { fc with axis = Unrolled })
  | Unroll { axis; materialize = true } ->
      rewrite_loop ~what:"Schedule.Unroll" ~sym:axis llc ~f:(fun fc ->
          unflat_lines
            (List.init
               (fc.to_ - fc.from_ + 1)
               ~f:(fun k ->
                 let v = fc.from_ + k in
                 map_code ~fidx:(subst_axis_index ~sym:axis ~by:{ terms = []; offset = v }) fc.body)))

let apply ?(static_indices = []) (sched : schedule) (opt : Low_level.optimized) :
    Low_level.optimized =
  if List.is_empty sched then opt
  else
    let llc = List.fold sched ~init:opt.Low_level.llc ~f:apply_op in
    (* Transforms fold their own guards (schedule-ir-optops §2): the pipeline's simplify already
       ran, so re-run it here; and when a transform duplicated code, re-run CSE + hoisting too. *)
    let llc = Low_level.simplify_llc static_indices llc in
    let llc =
      if List.exists sched ~f:(function Unroll { materialize = true; _ } -> true | _ -> false)
      then Low_level.hoist_cross_statement_cse @@ Low_level.eliminate_common_subexpressions llc
      else llc
    in
    { opt with llc }

(** {2 The default GPU annotator}

    [default_gpu] computes a schedule that makes elementwise / outer-parallel kernels launch with
    real grid and workgroup dimensions (schedule-ir-optops §6). It is deliberately conservative:
    parallelizing is only proposed when the analysis below proves that annotating cannot introduce
    a race, otherwise the empty schedule is returned and the kernel runs 1×1 as before.

    Thread identity after annotation is the tuple of annotated-loop index values. The safety
    argument, per kernel:

    - A written node's every write vector contains each of its nest's chosen parallel symbols as a
      plain [Iterator] component, so equal components imply the same thread (injectivity) — and
      [Split]'s [factor*i_o + i_i] substitution preserves injectivity because [i_i < factor].
    - All accesses to a written node agree on every component that mentions a parallel symbol, so
      reads only ever hit the reading thread's own elements.
    - No node written in one top-level nest is accessed in another: sibling nests execute with no
      global synchronization between them, so any cross-nest producer/consumer pair (or WAW/WAR
      pair) is a race once threads interleave. Non-materialized (routine-local) scratch is
      per-thread, hence exempt — unless its writes mention parallel symbols, in which case each
      thread only writes a slice of its private copy and cross-nest reads would see garbage.
    - Whole-node [Zero_out] of a materialized node, barriers, opaque [Staged_compilation],
      pre-existing hardware annotations, and materialized writes outside every nest all bail. *)

type access = {
  a_tn : Tn.t;
  a_idcs : Indexing.axis_index array;
  a_write : bool;
  a_dynamic : bool;  (** [Get_dynamic]: the effective index is not statically known. *)
}

exception Bail

(* Collects accesses of tensor nodes (not scalar scope-locals) in [llc]. Raises [Bail] on opaque
   or clearly unschedulable constructs. [depth] counts enclosing [Local_scope] bodies:
   materialized writes there are invisible to [validate_parallel]'s coverage check, so bail. *)
let scan_accesses (llc : Low_level.t) : access list =
  let open Low_level in
  let acc = ref [] in
  let add ~depth:_ ~write ~dynamic tn idcs =
    acc := { a_tn = tn; a_idcs = idcs; a_write = write; a_dynamic = dynamic } :: !acc
  in
  let rec code ~depth llc =
    match llc with
    | Noop | Comment _ | Declare_local _ -> ()
    | Staged_compilation _ -> raise Bail
    | Workgroup_barrier -> raise Bail
    | Seq (a, b) ->
        code ~depth a;
        code ~depth b
    | For_loop { axis; body; _ } ->
        if not (equal_axis_type axis Serial) then raise Bail;
        code ~depth body
    | Zero_out tn ->
        if Tn.is_materialized_force tn 172 then raise Bail
        (* Zeroing per-thread scratch is safe: each thread zeroes its own copy. *)
    | Set { tn; idcs; llsc; _ } ->
        if depth > 0 && Tn.is_materialized_force tn 172 then raise Bail;
        add ~depth ~write:true ~dynamic:false tn idcs;
        scalar ~depth llsc
    | Set_from_vec { tn; idcs; arg = a, _; _ } ->
        if depth > 0 && Tn.is_materialized_force tn 172 then raise Bail;
        add ~depth ~write:true ~dynamic:false tn idcs;
        scalar ~depth a
    | Set_local (_, llsc) -> scalar ~depth llsc
    | If { cond = c, _; body } ->
        scalar ~depth c;
        code ~depth body
  and scalar ~depth (llsc : scalar_t) =
    match llsc with
    | Local_scope { body; _ } -> code ~depth:(depth + 1) body
    | Get_local _ | Constant _ | Constant_bits _ | Embed_index _ -> ()
    | Get (tn, idcs) -> add ~depth ~write:false ~dynamic:false tn idcs
    | Get_dynamic { tn; idcs; dyn_value = v, _; _ } ->
        add ~depth ~write:false ~dynamic:true tn idcs;
        scalar ~depth v
    | Get_merge_buffer (_, _) -> () (* The merge buffer is a separate read-only input buffer. *)
    | Ternop (_, (a, _), (b, _), (c, _)) ->
        scalar ~depth a;
        scalar ~depth b;
        scalar ~depth c
    | Binop (_, (a, _), (b, _)) ->
        scalar ~depth a;
        scalar ~depth b
    | Unop (_, (a, _)) -> scalar ~depth a
  in
  code ~depth:0 llc;
  !acc

type nest_info = {
  n_loops : Low_level.t;  (** The nest statement itself ([For_loop] possibly under [If]). *)
  n_accesses : access list;
}

(* Top-level statements of the kernel: [For_loop] (possibly [If]-wrapped) statements become nests;
   everything else contributes to the "bare" pseudo-nest (executed unconditionally by every
   thread of the launch). *)
let split_nests (llc : Low_level.t) : nest_info list * access list =
  let open Low_level in
  let rec is_nest = function
    | For_loop _ -> true
    | If { body; _ } -> is_nest body
    | _ -> false
  in
  let stmts = flat_lines [ llc ] in
  let nests, bare =
    List.partition_map stmts ~f:(fun stmt ->
        if is_nest stmt then First { n_loops = stmt; n_accesses = scan_accesses stmt }
        else Second (scan_accesses stmt))
  in
  (nests, List.concat bare)

let mentions_sym syms (idx : Indexing.axis_index) =
  match idx with
  | Indexing.Iterator s -> List.mem syms s ~equal:Indexing.equal_symbol
  | Indexing.Affine { symbols; _ } ->
      List.exists symbols ~f:(fun (_, s) -> List.mem syms s ~equal:Indexing.equal_symbol)
  | Indexing.Fixed_idx _ | Indexing.Sub_axis | Indexing.Concat _ -> false

(* The single-child chain of [For_loop]s from the top of a nest, descending through [If] wrappers
   and comments; stops at the first branching ([Seq] with more than one non-comment statement). *)
let path_loops (nest : Low_level.t) : Low_level.t list =
  let open Low_level in
  let strip stmts =
    List.filter stmts ~f:(function Noop | Comment _ -> false | _ -> true)
  in
  let rec go llc acc =
    match llc with
    | For_loop fc -> (
        let acc = llc :: acc in
        match strip (flat_lines [ fc.body ]) with [ single ] -> go single acc | _ -> List.rev acc)
    | If { body; _ } -> go body acc
    | _ -> List.rev acc
  in
  go nest []

let default_gpu ?block_size ?min_parallel (opt : Low_level.optimized) : schedule =
  let open Low_level in
  let block_size =
    Option.value block_size
      ~default:
        (Int.of_string @@ Utils.get_global_arg ~arg_name:"gpu_schedule_block_size" ~default:"256")
  in
  let min_parallel =
    Option.value min_parallel
      ~default:
        (Int.of_string @@ Utils.get_global_arg ~arg_name:"gpu_schedule_min_parallel" ~default:"1024")
  in
  try
    let nests, bare = split_nests opt.llc in
    (* Bare materialized writes cannot be covered by annotated loops. *)
    if List.exists bare ~f:(fun a -> a.a_write && Tn.is_materialized_force a.a_tn 172) then
      raise Bail;
    (* Chains: per nest, the outermost (up to two) Serial path loops whose index occurs as a plain
       [Iterator] component in every materialized write vector of the nest. *)
    let chains =
      List.map nests ~f:(fun n ->
          let mat_writes =
            List.filter n.n_accesses ~f:(fun a ->
                a.a_write && Tn.is_materialized_force a.a_tn 172)
          in
          let qualifies s =
            (not (List.is_empty mat_writes))
            && List.for_all mat_writes ~f:(fun a ->
                   Array.exists a.a_idcs ~f:(fun idx ->
                       Indexing.equal_axis_index idx (Indexing.Iterator s)))
          in
          let chain =
            List.filter (path_loops n.n_loops) ~f:(function
              | For_loop fc -> fc.from_ = 0 && qualifies fc.index
              | _ -> false)
            |> fun l -> List.take l 2
          in
          if List.is_empty chain && not (List.is_empty mat_writes) then raise Bail;
          chain)
    in
    let chain_syms chain =
      List.filter_map chain ~f:(function For_loop fc -> Some fc.index | _ -> None)
    in
    (* Per-nest hazard analysis (see the module comment for the safety argument). *)
    List.iter2_exn nests chains ~f:(fun n chain ->
        let syms = chain_syms chain in
        let by_tn = Hashtbl.create (module Int) in
        List.iter n.n_accesses ~f:(fun a ->
            Hashtbl.add_multi by_tn ~key:a.a_tn.Tn.id ~data:a);
        Hashtbl.iter by_tn ~f:(fun accs ->
            let written = List.exists accs ~f:(fun a -> a.a_write) in
            if written then (
              let is_mat = Tn.is_materialized_force (List.hd_exn accs).a_tn 172 in
              let chain_relevant =
                List.exists accs ~f:(fun a -> Array.exists a.a_idcs ~f:(mentions_sym syms))
              in
              if is_mat || chain_relevant then (
                if List.exists accs ~f:(fun a -> a.a_dynamic) then raise Bail;
                (* All accesses must agree on every component that mentions a parallel symbol. *)
                let rank =
                  List.fold accs ~init:0 ~f:(fun m a -> max m (Array.length a.a_idcs))
                in
                for p = 0 to rank - 1 do
                  let comps =
                    List.map accs ~f:(fun a ->
                        if p < Array.length a.a_idcs then a.a_idcs.(p) else Indexing.Fixed_idx 0)
                  in
                  if List.exists comps ~f:(mentions_sym syms) then
                    match comps with
                    | [] -> ()
                    | c0 :: rest ->
                        if not (List.for_all rest ~f:(Indexing.equal_axis_index c0)) then
                          raise Bail
                done))));
    (* Cross-nest interference: nothing written in one nest may be touched in another (or in bare
       statements), except fully-per-thread scratch. *)
    let groups =
      List.map2_exn nests chains ~f:(fun n chain -> (n.n_accesses, chain_syms chain))
      @ [ (bare, []) ]
    in
    List.iteri groups ~f:(fun i (accs_i, syms_i) ->
        let writes_i = List.filter accs_i ~f:(fun a -> a.a_write) in
        if not (List.is_empty writes_i) then
          List.iteri groups ~f:(fun j (accs_j, _) ->
              if i <> j then
                List.iter writes_i ~f:(fun w ->
                    let touched_elsewhere =
                      List.exists accs_j ~f:(fun a -> a.a_tn.Tn.id = w.a_tn.Tn.id)
                    in
                    if touched_elsewhere then
                      if Tn.is_materialized_force w.a_tn 172 then raise Bail
                      else if
                        (* Local scratch: safe only if every thread writes its whole private
                           copy, i.e. the writes do not depend on parallel symbols. *)
                        Array.exists w.a_idcs ~f:(mentions_sym syms_i)
                      then raise Bail)));
    (* Threshold: skip kernels whose largest parallelizable nest is too small to pay for a
       launch. *)
    let nest_parallel_size chain =
      List.fold chain ~init:1 ~f:(fun sz -> function
        | For_loop fc -> sz * (fc.to_ + 1) | _ -> sz)
    in
    let max_parallel =
      List.fold chains ~init:0 ~f:(fun m chain -> max m (nest_parallel_size chain))
    in
    if max_parallel < min_parallel then []
    else
      (* Emit per-nest ops. Every annotated nest contributes exactly one Grid and one Workgroup
         loop, so hardware slots are uniform ([.x] of each kind) across nests and every
         materialized write covers all active dimensions ([validate_parallel]'s requirement). *)
      List.concat_map chains ~f:(fun chain ->
          match chain with
          | [] -> []
          | [ For_loop fc ] ->
              let n0 = fc.to_ + 1 in
              let op, _, _ =
                split ~axis:fc.index ~factor:(min block_size n0) ~outer:Grid ~inner:Workgroup
              in
              [ op ]
          | For_loop fc0 :: For_loop fc1 :: _ ->
              let n1 = fc1.to_ + 1 in
              if n1 <= block_size then
                [ Retype { axis = fc0.index; ty = Grid }; Retype { axis = fc1.index; ty = Workgroup } ]
              else
                let op, _, _ =
                  split ~axis:fc1.index ~factor:block_size ~outer:Serial ~inner:Workgroup
                in
                [ Retype { axis = fc0.index; ty = Grid }; op ]
          | _ -> [])
  with Bail -> []

(** {2 Wiring: the implicit transform for GPU backends} *)

let automatic_gpu_schedule =
  lazy (Utils.get_global_flag ~default:true ~arg_name:"automatic_gpu_schedule")

let backend_is_gpu name =
  String.is_substring name ~substring:"cuda" || String.is_substring name ~substring:"metal"

let maybe_default_gpu ~backend_name ~static_indices (opt : Low_level.optimized) :
    Low_level.optimized =
  if
    backend_is_gpu backend_name
    && Lazy.force automatic_gpu_schedule
    (* Runtime kernel logging is line-interleaved under parallel execution; keep logged runs
       serial so the logs stay deterministic and readable. *)
    && not (Utils.debug_log_from_routines ())
  then apply ~static_indices (default_gpu opt) opt
  else opt
