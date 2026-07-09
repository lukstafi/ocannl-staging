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
  | Stage of {
      source : Tn.t;
      tile_loops : Indexing.symbol list;
      shared : bool;
      cooperative : int option;
      hoisted : bool;
    }
  | Privatize of { target : Tn.t; over : Indexing.symbol }
  | Expand_zero of { tn : Tn.t; indices : Indexing.symbol list }
  | Tensorize of {
      i : Indexing.symbol;
      j : Indexing.symbol;
      k : Indexing.symbol;
      lane : Indexing.symbol;
      simd_width : int;
    }
[@@deriving sexp_of]

type schedule = optop list [@@deriving sexp_of]

let split ~axis ~factor ~outer ~inner =
  let outer_index = Indexing.get_symbol () and inner_index = Indexing.get_symbol () in
  (Split { axis; factor; outer; inner; outer_index; inner_index }, outer_index, inner_index)

let tensorize ~i ~j ~k ~simd_width =
  let lane = Indexing.get_symbol () in
  (Tensorize { i; j; k; lane; simd_width }, lane)

let expand_zero ~tn =
  let rank = Array.length (Lazy.force tn.Tn.dims) in
  let indices = List.init rank ~f:(fun _ -> Indexing.get_symbol ()) in
  (Expand_zero { tn; indices }, indices)

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
  | Tile_mma { d = d_tn, d_idcs; a = a_tn, a_idcs; b = b_tn, b_idcs; m; n; k; lane; fallback } ->
      Tile_mma
        {
          d = (d_tn, Array.map d_idcs ~f:fidx);
          a = (a_tn, Array.map a_idcs ~f:fidx);
          b = (b_tn, Array.map b_idcs ~f:fidx);
          m;
          n;
          k;
          lane;
          (* The fallback's loop symbols are fresh and bound inside it; only free (outer) symbols
             are substituted, exactly like the base indices. *)
          fallback = map_code ~fidx fallback;
        }
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

(* Fresh scope ids for scalar locals declared within [llc] (binders: [Declare_local],
   [Local_scope]). Materializing [Unroll] duplicates its body: without refreshing, sibling copies
   would declare the same scope id — rendered as duplicate declarations in one C block — and
   confuse the scope-id-keyed CSE/hoisting passes. References to locals declared outside [llc]
   are left alone. *)
let refresh_scopes (llc : Low_level.t) : Low_level.t =
  let open Low_level in
  let mapping = ref [] in
  let bind id =
    if not (List.Assoc.mem !mapping ~equal:equal_scope_id id) then
      mapping := (id, get_scope id.tn) :: !mapping
  in
  let rec collect llc =
    match llc with
    | Noop | Comment _ | Staged_compilation _ | Zero_out _ | Workgroup_barrier -> ()
    | Tile_mma { fallback; _ } -> collect fallback
    | Declare_local { id; _ } -> bind id
    | Seq (a, b) ->
        collect a;
        collect b
    | For_loop { body; _ } | If { body; _ } -> collect body
    | Set { llsc; _ } | Set_local (_, llsc) -> collect_scalar llsc
    | Set_from_vec { arg = a, _; _ } -> collect_scalar a
  and collect_scalar (llsc : scalar_t) =
    match llsc with
    | Local_scope { id; body; _ } ->
        bind id;
        collect body
    | Get_local _ | Get _ | Get_merge_buffer _ | Constant _ | Constant_bits _ | Embed_index _ -> ()
    | Get_dynamic { dyn_value = v, _; _ } -> collect_scalar v
    | Ternop (_, (a, _), (b, _), (c, _)) ->
        collect_scalar a;
        collect_scalar b;
        collect_scalar c
    | Binop (_, (a, _), (b, _)) ->
        collect_scalar a;
        collect_scalar b
    | Unop (_, (a, _)) -> collect_scalar a
  in
  collect llc;
  if List.is_empty !mapping then llc
  else
    let subst id =
      match List.Assoc.find !mapping ~equal:equal_scope_id id with Some id' -> id' | None -> id
    in
    let rec code llc =
      match llc with
      | Noop | Comment _ | Staged_compilation _ | Zero_out _ | Workgroup_barrier -> llc
      | Tile_mma ({ fallback; _ } as tm) -> Tile_mma { tm with fallback = code fallback }
      | Declare_local { id; needs_init } -> Declare_local { id = subst id; needs_init }
      | Seq (a, b) -> Seq (code a, code b)
      | For_loop fc -> For_loop { fc with body = code fc.body }
      | If { cond = c, p; body } -> If { cond = (scalar c, p); body = code body }
      | Set ({ llsc; _ } as s) -> Set { s with llsc = scalar llsc }
      | Set_local (id, llsc) -> Set_local (subst id, scalar llsc)
      | Set_from_vec ({ arg = a, p; _ } as sv) -> Set_from_vec { sv with arg = (scalar a, p) }
    and scalar (llsc : scalar_t) : scalar_t =
      match llsc with
      | Local_scope { id; body; orig_indices } ->
          Local_scope { id = subst id; body = code body; orig_indices }
      | Get_local id -> Get_local (subst id)
      | Get _ | Get_merge_buffer _ | Constant _ | Constant_bits _ | Embed_index _ -> llsc
      | Get_dynamic ({ dyn_value = v, p; _ } as gd) -> Get_dynamic { gd with dyn_value = (scalar v, p) }
      | Ternop (op, (a, pa), (b, pb), (c, pc)) ->
          Ternop (op, (scalar a, pa), (scalar b, pb), (scalar c, pc))
      | Binop (op, (a, pa), (b, pb)) -> Binop (op, (scalar a, pa), (scalar b, pb))
      | Unop (op, (a, pa)) -> Unop (op, (scalar a, pa))
    in
    code llc

let apply_op (llc : Low_level.t) (op : optop) : Low_level.t =
  let open Low_level in
  match op with
  | Stage _ | Privatize _ ->
      assert false (* Handled by [apply_opt_op]: they need the whole [optimized]. *)
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
          | Serial | Unrolled | Vectorized -> ());
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
                 refresh_scopes
                 @@ map_code ~fidx:(subst_axis_index ~sym:axis ~by:{ terms = []; offset = v })
                      fc.body)))
  | Tensorize { i; j; k; lane; simd_width } ->
      (* docs/proposals/tensorize-mma.md §3: replace the innermost serial [i × j × k] matmul
         micro-kernel — whose body is a single accumulation [d[...] += a[...] * b[...]] (plain-add
         or FMA form, as [optimize]'s simplify leaves it) — with a [Tile_mma] block statement
         wrapped in a fresh extent-[simd_width] [Workgroup] lane loop. The statement covers the
         whole [m×n×k] block, so fragment residency across the reduction is an intra-statement
         codegen concern; the original nest becomes the scalar [fallback]. Divisibility by the
         backend's intrinsic tile is checked at emission ([mma_syntax] declines per call and the
         fallback runs), since the schedule layer is backend-agnostic. *)
      rewrite_loop ~what:"Schedule.Tensorize" ~sym:i llc ~f:(fun ifc ->
          if simd_width <= 0 then invalid_arg "Schedule.Tensorize: simd_width must be positive";
          let strip body =
            List.filter (flat_lines [ body ]) ~f:(function Noop | Comment _ -> false | _ -> true)
          in
          let nested ~of_ sym body =
            match strip body with
            | [ For_loop { index; from_; to_; body; trace_it; axis } ]
              when Indexing.equal_symbol index sym ->
                { index; from_; to_; body; trace_it; axis }
            | _ ->
                invalid_arg
                  ("Schedule.Tensorize: loop " ^ Indexing.symbol_ident sym
                 ^ " must be exactly the body of loop " ^ Indexing.symbol_ident of_
                 ^ " (a perfectly nested i x j x k micro-kernel)")
          in
          let jfc = nested ~of_:i j ifc.body in
          let kfc = nested ~of_:j k jfc.body in
          List.iter
            [ { ifc with body = Noop }; { jfc with body = Noop }; { kfc with body = Noop } ]
            ~f:(fun fc ->
              if not (equal_axis_type fc.axis Serial) || fc.from_ <> 0 then
                invalid_arg
                  ("Schedule.Tensorize: loop " ^ Indexing.symbol_ident fc.index
                 ^ " must be Serial starting at 0"));
          let d_tn, d_idcs, llsc =
            match strip kfc.body with
            | [ Set { tn; idcs; llsc; _ } ] -> (tn, idcs, llsc)
            | _ ->
                invalid_arg
                  "Schedule.Tensorize: the micro-kernel body must be a single accumulation Set"
          in
          let is_d_read (sc : Low_level.scalar_t) =
            match sc with
            | Get (tn, idcs) ->
                Tn.equal tn d_tn && Array.equal Indexing.equal_axis_index idcs d_idcs
            | _ -> false
          in
          let get_operand (sc : Low_level.scalar_t) =
            match sc with Get (tn, idcs) -> Some (tn, idcs) | _ -> None
          in
          let operands =
            match llsc with
            | Ternop (Ops.FMA, (x, _), (y, _), (acc, _)) when is_d_read acc ->
                Option.both (get_operand x) (get_operand y)
            | Binop (Ops.Add, (acc, _), (Binop (Ops.Mul, (x, _), (y, _)), _)) when is_d_read acc ->
                Option.both (get_operand x) (get_operand y)
            | Binop (Ops.Add, (Binop (Ops.Mul, (x, _), (y, _)), _), (acc, _)) when is_d_read acc ->
                Option.both (get_operand x) (get_operand y)
            | _ -> None
          in
          let x_op, y_op =
            match operands with
            | Some ops -> ops
            | None ->
                invalid_arg
                  ("Schedule.Tensorize: the micro-kernel must accumulate a product of reads into "
                 ^ Tn.debug_name d_tn ^ " (d[...] += a[...] * b[...], plain-add or FMA form)")
          in
          let mentions sym (idx : Indexing.axis_index) =
            match idx with
            | Indexing.Iterator s -> Indexing.equal_symbol s sym
            | Indexing.Affine { symbols; _ } ->
                List.exists symbols ~f:(fun (_, s) -> Indexing.equal_symbol s sym)
            | Indexing.Fixed_idx _ | Indexing.Sub_axis | Indexing.Concat _ -> false
          in
          let coeff sym (idx : Indexing.axis_index) =
            match idx with
            | Indexing.Iterator s when Indexing.equal_symbol s sym -> 1
            | Indexing.Affine { symbols; _ } ->
                List.sum
                  (module Int)
                  symbols
                  ~f:(fun (c, s) -> if Indexing.equal_symbol s sym then c else 0)
            | _ -> 0
          in
          (* Index discipline: the tile spans the operand's last two axes with unit strides in the
             micro-kernel symbols — [row] appears with coefficient 1 exactly in component
             [rank-2], [col] in component [rank-1], and the third symbol not at all. Outer-loop
             terms (the block base) may appear anywhere. *)
          let role_ok (_tn, idcs) ~row ~col =
            let rank = Array.length idcs in
            rank >= 2
            && coeff row idcs.(rank - 2) = 1
            && coeff col idcs.(rank - 1) = 1
            && List.for_all [ i; j; k ] ~f:(fun s ->
                   Array.for_alli idcs ~f:(fun p idx ->
                       let allowed =
                         if Indexing.equal_symbol s row then p = rank - 2
                         else if Indexing.equal_symbol s col then p = rank - 1
                         else false
                       in
                       (not (mentions s idx)) || allowed))
          in
          if not (role_ok (d_tn, d_idcs) ~row:i ~col:j) then
            invalid_arg
              ("Schedule.Tensorize: accumulator " ^ Tn.debug_name d_tn
             ^ " must be indexed [..., i, j] over its last two axes (unit coefficients)");
          let a_op, b_op =
            if role_ok x_op ~row:i ~col:k && role_ok y_op ~row:k ~col:j then (x_op, y_op)
            else if role_ok y_op ~row:i ~col:k && role_ok x_op ~row:k ~col:j then (y_op, x_op)
            else
              invalid_arg
                ("Schedule.Tensorize: operands of the product must be indexed [..., i, k] and \
                  [..., k, j] over their last two axes (unit coefficients; transposed layouts are \
                  not supported in v1): "
                ^ Tn.debug_name (fst x_op)
                ^ ", "
                ^ Tn.debug_name (fst y_op))
          in
          let zero = { terms = []; offset = 0 } in
          let base idcs =
            Array.map idcs ~f:(fun idx ->
                subst_axis_index ~sym:i ~by:zero
                  (subst_axis_index ~sym:j ~by:zero (subst_axis_index ~sym:k ~by:zero idx)))
          in
          For_loop
            {
              index = lane;
              from_ = 0;
              to_ = simd_width - 1;
              axis = Workgroup;
              trace_it = false;
              body =
                Tile_mma
                  {
                    d = (d_tn, base d_idcs);
                    a = (fst a_op, base (snd a_op));
                    b = (fst b_op, base (snd b_op));
                    m = ifc.to_ + 1;
                    n = jfc.to_ + 1;
                    k = kfc.to_ + 1;
                    lane;
                    fallback = for_loop ifc;
                  };
            })
  | Expand_zero { tn; indices } ->
      (* Whole-node [Zero_out] is never distributed across hardware threads ([validate_parallel]
         rejects it in multi-threaded kernels); expand it into an ordinary loop nest — over the
         caller-supplied symbols, so subsequent ops in the schedule can split and annotate the
         zeroing with the same geometry as the computation. *)
      let dims = Lazy.force tn.Tn.dims in
      if Array.length dims <> List.length indices then
        invalid_arg
          ("Schedule.Expand_zero: " ^ Int.to_string (List.length indices) ^ " indices for a rank-"
          ^ Int.to_string (Array.length dims)
          ^ " node");
      let idcs = Array.of_list_map indices ~f:(fun s -> Indexing.Iterator s) in
      let nest =
        List.fold_right
          (List.zip_exn indices (Array.to_list dims))
          ~init:(Set { tn; idcs; llsc = Constant 0.; debug = "" })
          ~f:(fun (s, d) body ->
            For_loop { index = s; from_ = 0; to_ = d - 1; body; trace_it = false; axis = Serial })
      in
      let found = ref false in
      let rec go llc =
        match llc with
        | Zero_out tn' when Tn.equal tn tn' ->
            if !found then
              invalid_arg
                ("Schedule.Expand_zero: multiple Zero_out statements for " ^ Tn.debug_name tn);
            found := true;
            nest
        | Seq (a, b) -> Seq (go a, go b)
        | For_loop fc -> For_loop { fc with body = go fc.body }
        | If { cond; body } -> If { cond; body = go body }
        | other -> other
      in
      let result = go llc in
      if not !found then
        invalid_arg ("Schedule.Expand_zero: no Zero_out of " ^ Tn.debug_name tn);
      result

(** {2 [Stage]: tile staging (schedule-ir-optops §5)}

    The one transform that synthesizes code. [Stage { source; tile_loops; shared }] requires all
    reads of [source] to use one index vector (v1). Each source axis's index is decomposed into a
    tile part (terms over [tile_loops], positive coefficients) and an outer part; source axes with
    a nonempty tile part become tile axes, sized by the tile part's range over the tile loops'
    extents. The load nest is inserted at the deepest loop that must stay outside the tile — the
    innermost loop carrying an outer-part symbol or a reused [Workgroup] tile axis — by replacing
    that loop's body with [loads; barrier; body-with-reads-remapped; barrier] ([shared]) or
    [loads; remapped body] (packing). Cooperative loads reuse [Workgroup]-typed tile loops as the
    cooperating thread indices, iterate [Serial] tile loops under fresh symbols, guard each tile
    axis with an [If (index < dim)] edge guard (construct-then-fold), and — for [shared] — restrict
    redundant loading along non-participating workgroup axes with [If (w == 0)] guards. The tile is
    a fresh [Local]-mode node registered in the traced store (and in [workgroup_shared] when
    [shared]). *)

(* Schedule-minted tile/accumulator nodes live in their own reserved namespace, so their session
   ids are independent of tensor-land ids (allocated from 0 by the [Tensor] session counter). *)
let tile_namespace = "tile"

let fresh_tile_id =
  let c = ref (-1) in
  fun () ->
    Int.incr c;
    !c

let terms_of_index (idx : Indexing.axis_index) : ((int * Indexing.symbol) list * int) option =
  match idx with
  | Indexing.Fixed_idx k -> Some ([], k)
  | Indexing.Iterator s -> Some ([ (1, s) ], 0)
  | Indexing.Affine { symbols; offset } -> Some (symbols, offset)
  | Indexing.Sub_axis -> Some ([], 0)
  | Indexing.Concat _ -> None

(* All reads [Get (source, idcs)] with their enclosing statement-level loop stacks
   (outermost-first; [floop.body] is dummied out). Writes to [source] are rejected. *)
let collect_source_accesses ~source (llc : Low_level.t) :
    (Indexing.axis_index array * floop list) list =
  let open Low_level in
  let acc = ref [] in
  let reject_write tn =
    if Tn.equal tn source then
      invalid_arg ("Schedule.Stage: source " ^ Tn.debug_name source ^ " is written in the routine")
  in
  let rec code stack llc =
    match llc with
    | Noop | Comment _ | Staged_compilation _ | Declare_local _ | Workgroup_barrier -> ()
    | Tile_mma _ ->
        (* Cooperative-tile operands are not remappable reads in v1. *)
        invalid_arg "Schedule.Stage: apply Stage before Tensorize"
    | Zero_out tn -> reject_write tn
    | Seq (a, b) ->
        code stack a;
        code stack b
    | For_loop { index; from_; to_; body; trace_it; axis } ->
        code ({ index; from_; to_; body = Noop; trace_it; axis } :: stack) body
    | Set { tn; llsc; _ } ->
        reject_write tn;
        scalar stack llsc
    | Set_from_vec { tn; arg = a, _; _ } ->
        reject_write tn;
        scalar stack a
    | Set_local (_, llsc) -> scalar stack llsc
    | If { cond = c, _; body } ->
        scalar stack c;
        code stack body
  and scalar stack (llsc : scalar_t) =
    match llsc with
    | Local_scope { body; _ } -> code stack body
    | Get_local _ | Constant _ | Constant_bits _ | Embed_index _ -> ()
    | Get (tn, idcs) -> if Tn.equal tn source then acc := (idcs, List.rev stack) :: !acc
    | Get_dynamic { tn; dyn_value = v, _; _ } ->
        if Tn.equal tn source then
          invalid_arg "Schedule.Stage: dynamically indexed source reads are unsupported";
        scalar stack v
    | Get_merge_buffer (_, _) -> ()
    | Ternop (_, (a, _), (b, _), (c, _)) ->
        scalar stack a;
        scalar stack b;
        scalar stack c
    | Binop (_, (a, _), (b, _)) ->
        scalar stack a;
        scalar stack b
    | Unop (_, (a, _)) -> scalar stack a
  in
  code [] llc;
  !acc

(* Replaces reads [Get (source, idcs)] — and, when [writes] is set, writes
   [Set { tn = source; idcs; _ }] — with [idcs] equal to [from_idcs] by accesses of [tile] at
   [tile_idcs]. *)
let remap_reads ?(writes = false) ~source ~from_idcs ~tile ~tile_idcs (llc : Low_level.t) :
    Low_level.t =
  let open Low_level in
  let rec code llc =
    match llc with
    | Noop | Comment _ | Staged_compilation _ | Zero_out _ | Declare_local _ | Workgroup_barrier ->
        llc
    (* Unreachable: [collect_source_accesses] / Privatize's scan reject [Tile_mma] first. *)
    | Tile_mma _ -> llc
    | Seq (a, b) -> Seq (code a, code b)
    | For_loop fc -> For_loop { fc with body = code fc.body }
    | Set { tn; idcs; llsc; debug }
      when writes && Tn.equal tn source && Array.equal Indexing.equal_axis_index idcs from_idcs ->
        Set { tn = tile; idcs = tile_idcs; llsc = scalar llsc; debug }
    | Set { tn; idcs; llsc; debug } -> Set { tn; idcs; llsc = scalar llsc; debug }
    | Set_from_vec ({ arg = a, p; _ } as sv) -> Set_from_vec { sv with arg = (scalar a, p) }
    | Set_local (id, llsc) -> Set_local (id, scalar llsc)
    | If { cond = c, p; body } -> If { cond = (scalar c, p); body = code body }
  and scalar (llsc : scalar_t) : scalar_t =
    match llsc with
    | Get (tn, idcs)
      when Tn.equal tn source && Array.equal Indexing.equal_axis_index idcs from_idcs ->
        Get (tile, tile_idcs)
    | Get _ | Get_local _ | Get_merge_buffer _ | Constant _ | Constant_bits _ | Embed_index _ ->
        llsc
    | Local_scope ({ body; _ } as ls) -> Local_scope { ls with body = code body }
    | Get_dynamic ({ dyn_value = v, p; _ } as gd) -> Get_dynamic { gd with dyn_value = (scalar v, p) }
    | Ternop (op, (a, pa), (b, pb), (c, pc)) ->
        Ternop (op, (scalar a, pa), (scalar b, pb), (scalar c, pc))
    | Binop (op, (a, pa), (b, pb)) -> Binop (op, (scalar a, pa), (scalar b, pb))
    | Unop (op, (a, pa)) -> Unop (op, (scalar a, pa))
  in
  code llc

(* Host-side packing for hoisted Stage (gh-ocannl-470): materialize the packed layout of a
   constant operand from its host-init data. Forced at link/upload time — through the packed
   node's own [Host_inits] lazy — and uploaded into the per-device constant pool like any other
   host-initialized constant. [src_prog]/[packed_prog] are the per-axis affine index programs:
   [(coefficient, position in sym_extents) array * offset], evaluated over the odometer
   enumeration of the tile and outer loop symbols. *)
let pack_constant_tile ~debug ~(src_nd : Ndarray.t) ~(src_dims : int array) ~(prec : Ops.prec)
    ~(packed_dims : int array) ~(sym_extents : int array)
    ~(src_prog : ((int * int) array * int) array)
    ~(packed_prog : ((int * int) array * int) array) : Ndarray.t =
  if not (Ops.equal_prec (Ndarray.get_prec src_nd) prec) then
    invalid_arg ("Schedule.Stage: hoisted staging: host-init precision mismatch for " ^ debug);
  if not (Array.equal Int.equal (Ndarray.dims src_nd) src_dims) then
    invalid_arg ("Schedule.Stage: hoisted staging: host-init dims mismatch for " ^ debug);
  (* Zero-filled at creation: pad slots of edge tiles (never written below) must read as zeros. *)
  let dst = Ndarray.create_array ~debug prec ~dims:packed_dims ~padding:(Some ([||], Some 0.0)) in
  let n_syms = Array.length sym_extents in
  let vals = Array.create ~len:n_syms 0 in
  let eval (terms, off) = Array.fold terms ~init:off ~f:(fun acc (c, p) -> acc + (c * vals.(p))) in
  let src_idx = Array.create ~len:(Array.length src_dims) 0 in
  let dst_idx = Array.create ~len:(Array.length packed_dims) 0 in
  let f2 src dstb =
    let rec go d =
      if d = n_syms then (
        Array.iteri src_prog ~f:(fun a pr -> src_idx.(a) <- eval pr);
        (* The edge guard: out-of-range coordinates arise only from edge tiles. *)
        if Array.for_alli src_idx ~f:(fun a i -> i < src_dims.(a)) then (
          Array.iteri packed_prog ~f:(fun a pr -> dst_idx.(a) <- eval pr);
          Stdlib.Bigarray.Genarray.set dstb dst_idx (Stdlib.Bigarray.Genarray.get src src_idx)))
      else
        for v = 0 to sym_extents.(d) - 1 do
          vals.(d) <- v;
          go (d + 1)
        done
    in
    go 0
  in
  Ndarray.apply2 { f2 } src_nd dst;
  dst

let apply_stage ~source ~tile_loops ~shared ~cooperative ~hoisted (opt : Low_level.optimized) :
    Low_level.optimized =
  let open Low_level in
  if List.is_empty tile_loops then invalid_arg "Schedule.Stage: empty tile_loops";
  Option.iter cooperative ~f:(fun w ->
      if not shared then invalid_arg "Schedule.Stage: cooperative staging requires shared = true";
      if w <= 0 then invalid_arg "Schedule.Stage: cooperative simd width must be positive");
  if hoisted && shared then
    invalid_arg "Schedule.Stage: hoisted staging requires shared = false (it emits no load nest)";
  let accesses = collect_source_accesses ~source opt.llc in
  let idcs0, stack0 =
    match accesses with
    | [] -> invalid_arg ("Schedule.Stage: no reads of " ^ Tn.debug_name source ^ " in the routine")
    | hd :: _ -> hd
  in
  List.iter accesses ~f:(fun (idcs, _) ->
      if not (Array.equal Indexing.equal_axis_index idcs idcs0) then
        invalid_arg
          ("Schedule.Stage: v1 requires all reads of " ^ Tn.debug_name source
         ^ " to use identical index vectors"));
  let stack0 = Array.of_list stack0 (* outermost-first *) in
  let is_tile s = List.mem tile_loops s ~equal:Indexing.equal_symbol in
  let depth_of s = Array.findi stack0 ~f:(fun _ fl -> Indexing.equal_symbol fl.index s) in
  let floop_of_exn s =
    match depth_of s with
    | Some (_, fl) -> fl
    | None ->
        invalid_arg
          ("Schedule.Stage: tile loop " ^ Indexing.symbol_ident s
         ^ " does not enclose the source access")
  in
  (* Per source axis: tile part (terms over tile loops), outer part, offset. *)
  let decomp =
    Array.map idcs0 ~f:(fun idx ->
        match terms_of_index idx with
        | None -> invalid_arg "Schedule.Stage: Concat indices are unsupported"
        | Some (terms, offset) ->
            let tile_part, outer_part = List.partition_tf terms ~f:(fun (_, s) -> is_tile s) in
            (tile_part, outer_part, offset))
  in
  Array.iter decomp ~f:(fun (tp, _, _) ->
      List.iter tp ~f:(fun (c, s) ->
          if c <= 0 then
            invalid_arg "Schedule.Stage: nonpositive coefficient on a tile loop index";
          let fl = floop_of_exn s in
          if fl.from_ <> 0 then invalid_arg "Schedule.Stage: tile loops must start at 0"));
  List.iter tile_loops ~f:(fun s ->
      if
        not
          (Array.exists decomp ~f:(fun (tp, _, _) ->
               List.exists tp ~f:(fun (_, s') -> Indexing.equal_symbol s s')))
      then
        invalid_arg
          ("Schedule.Stage: tile loop " ^ Indexing.symbol_ident s
         ^ " does not occur in the source access"));
  let extent s =
    let fl = floop_of_exn s in
    fl.to_ - fl.from_ + 1
  in
  (* Tile axes: source axes with a nonempty tile part; dim = the tile part's range. *)
  let tile_axes =
    Array.filter_mapi decomp ~f:(fun a (tp, _, _) ->
        if List.is_empty tp then None
        else Some (a, List.fold tp ~init:1 ~f:(fun acc (c, s) -> acc + (c * (extent s - 1)))))
  in
  (* Classify tile loops: Workgroup-typed loops are reused as cooperating thread indices in the
     load nest; Serial ones are iterated under fresh symbols. *)
  let reused, iterated =
    List.partition_tf tile_loops ~f:(fun s ->
        match (floop_of_exn s).axis with
        | Workgroup | Workgroup_reduce -> true
        | Serial | Vectorized -> false
        | Grid -> invalid_arg "Schedule.Stage: a Grid-typed loop cannot be a tile loop"
        | Unrolled ->
            invalid_arg "Schedule.Stage: apply Stage before (materializing) Unroll of a tile loop")
  in
  if (not shared) && not (List.is_empty reused) then
    invalid_arg "Schedule.Stage: non-shared (packing) staging requires Serial tile loops";
  if Option.is_some cooperative && not (List.is_empty reused) then
    invalid_arg
      "Schedule.Stage: cooperative staging requires Serial tile loops (the fresh lane loop is the \
       cooperating axis; reusing Workgroup tile loops is the non-cooperative mode)";
  if hoisted then (
    (* Hoisted (out-of-routine) packing for a compile-time-constant source (gh-ocannl-470): the
       packed layout covers the whole source — one packed axis per outer coordinate followed by
       the tile axes — so no load nest is emitted; the reads are remapped to a fresh
       host-initialized constant whose buffer is packed on the host at link time and uploaded
       once per device into the constant pool. *)
    if not (Host_inits.mem source) then
      invalid_arg
        ("Schedule.Stage: hoisted staging requires registered host-init data for "
       ^ Tn.debug_name source);
    if
      not
        (Tn.known_host_constant source
        || Tn.Placements.known_constant opt.Low_level.optimize_ctx.placements source)
    then
      invalid_arg
        ("Schedule.Stage: hoisted staging requires a known-constant source, got "
       ^ Tn.debug_name source);
    (match Lazy.force source.Tn.padding with
    | Some _ ->
        invalid_arg
          ("Schedule.Stage: hoisted staging does not support a padded source: "
         ^ Tn.debug_name source)
    | None -> ());
    (* Packing runs at link time, outside the routine: every outer-part symbol must be an
       enclosing loop with a known extent (static/dynamic indices have no binding there), and the
       affine maps below need nonnegative source coordinates. *)
    Array.iter decomp ~f:(fun (_, op_, off) ->
        if off < 0 then
          invalid_arg "Schedule.Stage: hoisted staging requires nonnegative index offsets";
        List.iter op_ ~f:(fun (c, s) ->
            if c <= 0 then
              invalid_arg
                "Schedule.Stage: hoisted staging requires positive outer-part coefficients";
            if Option.is_none (depth_of s) then
              invalid_arg
                ("Schedule.Stage: hoisted staging requires outer-part symbol "
                ^ Indexing.symbol_ident s
                ^ " to be bound by an enclosing loop")));
    let tile_dim a = Array.find_map tile_axes ~f:(fun (a', d) -> Option.some_if (a = a') d) in
    (* Packed outer axes, in source-axis order: on a tiled axis the outer part (plus offset)
       divided by the tile dim is the tile-count coordinate (divisibility required — the standard
       blocked decomposition [k := bk*KT + k_i] satisfies it); an axis without a tile part keeps
       its outer index expression as-is. Raw [(terms, offset, extent)] — the terms drive both the
       consumer's remapped reads and the host-side packing program. *)
    let outer_axes =
      Array.filter_mapi decomp ~f:(fun a (_tp, op_, off) ->
          if List.is_empty op_ && off = 0 then None
          else
            match tile_dim a with
            | None ->
                let ext =
                  List.fold op_ ~init:(off + 1) ~f:(fun acc (c, s) -> acc + (c * (extent s - 1)))
                in
                Some (op_, off, ext)
            | Some t_a ->
                List.iter op_ ~f:(fun (c, _) ->
                    if c % t_a <> 0 then
                      invalid_arg
                        "Schedule.Stage: hoisted staging requires outer-part coefficients \
                         divisible by the tile dim of their axis");
                if off % t_a <> 0 then
                  invalid_arg
                    "Schedule.Stage: hoisted staging requires the index offset divisible by the \
                     tile dim of its axis";
                let q = List.map op_ ~f:(fun (c, s) -> (c / t_a, s)) in
                let qoff = off / t_a in
                let ext =
                  List.fold q ~init:(qoff + 1) ~f:(fun acc (c, s) -> acc + (c * (extent s - 1)))
                in
                Some (q, qoff, ext))
    in
    let packed_dims =
      Array.append (Array.map outer_axes ~f:(fun (_, _, ext) -> ext)) (Array.map tile_axes ~f:snd)
    in
    let packed_read_idcs =
      Array.append
        (Array.map outer_axes ~f:(fun (terms, off, _) -> normalize_affine ~terms ~offset:off))
        (Array.map tile_axes ~f:(fun (a, _) ->
             let tp, _, _ = decomp.(a) in
             normalize_affine ~terms:tp ~offset:0))
    in
    let prec = Lazy.force source.Tn.prec in
    let tile =
      Tn.create ~namespace:tile_namespace (Tn.Specified prec) ~id:(fresh_tile_id ())
        ~label:("packed" :: source.Tn.label)
        ~unpadded_dims:(lazy packed_dims)
        ~padding:(lazy None) ()
    in
    (* A host-initialized constant: [Effectively_constant] intent, materialized in this lineage —
       [allocate_delta] routes it (read-only, host-init-backed) into the per-device constant
       pool, and [Host_inits.mem] keeps it out of the routine's required inputs. *)
    Tn.update_memory_mode tile Effectively_constant 176;
    Tn.Placements.update opt.Low_level.optimize_ctx.placements tile Tn.On_device 176;
    let traced = get_node opt.traced_store tile in
    traced.read_only <- true;
    (* The packing program: odometer enumeration of every tile and outer symbol, evaluated
       through positional [(coefficient, symbol slot)] compiles of the affine maps. Captured
       eagerly — the lazy below must not reference the pre-remap loop structure. *)
    let syms =
      Array.to_list decomp
      |> List.concat_map ~f:(fun (tp, op_, _) -> List.map (tp @ op_) ~f:snd)
      |> List.dedup_and_sort ~compare:Indexing.Symbol.compare
      |> List.map ~f:(fun s -> (s, extent s))
      |> Array.of_list
    in
    let pos_of s =
      fst
        (Option.value_exn
           (Array.findi syms ~f:(fun _ (s', _) -> Indexing.equal_symbol s s')))
    in
    let compile_terms terms = Array.of_list_map terms ~f:(fun (c, s) -> (c, pos_of s)) in
    let src_prog = Array.map decomp ~f:(fun (tp, op_, off) -> (compile_terms (tp @ op_), off)) in
    let packed_prog =
      Array.append
        (Array.map outer_axes ~f:(fun (terms, off, _) -> (compile_terms terms, off)))
        (Array.map tile_axes ~f:(fun (a, _) ->
             let tp, _, _ = decomp.(a) in
             (compile_terms tp, 0)))
    in
    let sym_extents = Array.map syms ~f:snd in
    let src_dims = Lazy.force source.Tn.dims in
    let debug = Tn.debug_name tile in
    Host_inits.register tile
      (lazy
        (let src_nd =
           match Host_inits.find source with
           | Some l -> Lazy.force l
           | None ->
               invalid_arg
                 ("Schedule.Stage: host-init data for " ^ Tn.debug_name source
                ^ " disappeared before link")
         in
         pack_constant_tile ~debug ~src_nd ~src_dims ~prec ~packed_dims ~sym_extents ~src_prog
           ~packed_prog));
    let llc = remap_reads ~source ~from_idcs:idcs0 ~tile ~tile_idcs:packed_read_idcs opt.llc in
    { opt with llc })
  else
  (* The insertion point L*: the deepest loop that must stay outside the tile — carrying an
     outer-part symbol or a reused workgroup tile axis. *)
  let outer_sym_depths =
    Array.to_list decomp
    |> List.concat_map ~f:(fun (_, op_, _) -> op_)
    |> List.filter_map ~f:(fun (_, s) -> Option.map (depth_of s) ~f:fst)
  in
  let reused_depths = List.map reused ~f:(fun s -> fst (Option.value_exn (depth_of s))) in
  let lstar_depth = List.max_elt (outer_sym_depths @ reused_depths) ~compare:Int.compare in
  (* Serial tile loops may sit above or below L*: the load nest iterates them under fresh symbols,
     so only outer-part symbols and reused workgroup axes pin the staging point. *)
  (* Mint the tile. *)
  let prec = Lazy.force source.Tn.prec in
  let tile_dims = Array.map tile_axes ~f:snd in
  let tile =
    Tn.create ~namespace:tile_namespace (Tn.Specified prec) ~id:(fresh_tile_id ())
      ~label:("tile" :: source.Tn.label)
      ~unpadded_dims:(lazy tile_dims)
      ~padding:(lazy None) ()
  in
  Tn.Placements.update opt.Low_level.optimize_ctx.placements tile Tn.Local 175;
  ignore (get_node opt.traced_store tile : traced_array);
  (* The load nest. *)
  let fresh = List.map iterated ~f:(fun s -> (s, Indexing.get_symbol ())) in
  let load_sym s =
    match List.Assoc.find fresh ~equal:Indexing.equal_symbol s with Some s' -> s' | None -> s
  in
  let subst_terms terms = List.map terms ~f:(fun (c, s) -> (c, load_sym s)) in
  let load_src_idcs =
    Array.map decomp ~f:(fun (tp, op_, off) ->
        normalize_affine ~terms:(subst_terms tp @ op_) ~offset:off)
  in
  let tile_store_idcs =
    Array.map tile_axes ~f:(fun (a, _) ->
        let tp, _, _ = decomp.(a) in
        normalize_affine ~terms:(subst_terms tp) ~offset:0)
  in
  let tile_read_idcs =
    Array.map tile_axes ~f:(fun (a, _) ->
        let tp, _, _ = decomp.(a) in
        normalize_affine ~terms:tp ~offset:0)
  in
  let iprec = Ops.index_prec () in
  let src_dims = Lazy.force source.Tn.dims in
  let load_stmt =
    Set { tn = tile; idcs = tile_store_idcs; llsc = Get (source, load_src_idcs); debug = "" }
  in
  (* Edge guards per tile axis (construct-then-fold: [apply]'s trailing simplify erases the ones
     the loop extents prove, i.e. whenever the tile sizes divide the source extents). *)
  let load_stmt =
    Array.fold tile_axes ~init:load_stmt ~f:(fun stmt (a, _) ->
        let cond =
          Binop
            ( Ops.Cmplt,
              (Embed_index load_src_idcs.(a), iprec),
              (Constant (Float.of_int src_dims.(a)), iprec) )
        in
        If { cond = (cond, iprec); body = stmt })
  in
  (* Lane-aware cooperative staging (docs/proposals/tensorize-mma.md, "Lane-aware Stage"): a
     fresh extent-[w] [Workgroup] lane loop will wrap the load nest, so the loads are partitioned
     (or restricted) along the same hardware slot the tensorized micro-kernel's lane loop binds —
     positional slot assignment aligns the two loops at slot 0, and [validate_parallel]'s
     barrier-strength extent-uniformity rejects a width mismatch, so the only coordination with
     [Tensorize] is passing the same simd width. The partition folds the lane linearly into the
     innermost fresh copy loop (consecutive lanes touch consecutive minor-axis elements):
     division/modulo are not in the affine index algebra, so an extent that neither divides nor
     is bounded by the width falls back to the representative-lane ([w == 0]) discipline. *)
  let lane = Option.map cooperative ~f:(fun w -> (Indexing.get_symbol (), w)) in
  let lane_plan =
    match (lane, List.last fresh) with
    | None, _ | _, None -> `Serial_all
    | Some (w_sym, w), Some (s_orig, s_inner) ->
        let e = extent s_orig in
        if e <= w then `Drop_inner (s_inner, e, w_sym)
        else if e % w = 0 then `Divide_inner (s_inner, e / w, w_sym, w)
        else `Restrict_lane0
  in
  let load_stmt =
    match lane_plan with
    | `Serial_all | `Restrict_lane0 -> load_stmt
    | `Drop_inner (s_inner, e, w_sym) ->
        (* The lane replaces the innermost copy loop; the edge guard folds exactly when the
           extent equals the width (construct-then-fold, as everywhere in this transform). *)
        let stmt =
          map_code
            ~fidx:(subst_axis_index ~sym:s_inner ~by:{ terms = [ (1, w_sym) ]; offset = 0 })
            load_stmt
        in
        let cond =
          Binop
            ( Ops.Cmplt,
              (Embed_index (Indexing.Iterator w_sym), iprec),
              (Constant (Float.of_int e), iprec) )
        in
        If { cond = (cond, iprec); body = stmt }
    | `Divide_inner (s_inner, _, w_sym, w) ->
        (* [s_inner := w * s_inner + w_sym]: per copy step, the [w] lanes cover a contiguous
           chunk of the minor axis. *)
        map_code
          ~fidx:
            (subst_axis_index ~sym:s_inner ~by:{ terms = [ (w, s_inner); (1, w_sym) ]; offset = 0 })
          load_stmt
  in
  let load_nest =
    List.fold (List.rev fresh) ~init:load_stmt ~f:(fun body (s, s') ->
        match lane_plan with
        | `Drop_inner (si, _, _) when Indexing.equal_symbol s' si -> body
        | `Divide_inner (si, ext', _, _) when Indexing.equal_symbol s' si ->
            For_loop { index = s'; from_ = 0; to_ = ext' - 1; body; trace_it = false; axis = Serial }
        | _ ->
            For_loop
              { index = s'; from_ = 0; to_ = extent s - 1; body; trace_it = false; axis = Serial })
  in
  (* The splice target. With an anchor L* (an outer-part symbol or reused workgroup axis), the
     load nest goes at the start of L*'s body. A shared stage with no anchor (e.g. staging a
     broadcast vector indexed only by serial tile loops) must NOT go to the routine root: every
     hardware thread executes top-level statements, and no workgroup index symbol is bound there
     to restrict the loads — so wrap the outermost tile loop instead, where the workgroup axes
     enclosing the consumer also enclose (and can guard) the loads (PR #90 review). Packing
     (shared = false) writes per-thread scratch, so the root is fine. *)
  let outermost_tile_depth =
    List.map tile_loops ~f:(fun s -> fst (Option.value_exn (depth_of s)))
    |> List.min_elt ~compare:Int.compare
    |> Option.value_exn
  in
  let wrap_outermost_tile = shared && Option.is_none lstar_depth in
  (* Loops enclosing the staging point, outermost-first. *)
  let in_scope =
    match lstar_depth with
    | Some ld -> Array.to_list (Array.sub stack0 ~pos:0 ~len:(ld + 1))
    | None when wrap_outermost_tile ->
        Array.to_list (Array.sub stack0 ~pos:0 ~len:outermost_tile_depth)
    | None -> []
  in
  (* Every workgroup slot active in the kernel's launch must be either reused by this tile's
     cooperative load or bound by an enclosing loop at the staging point (restricted to [w == 0]
     below): hardware threads differing only in an uncovered slot would all execute the loads,
     writing the same shared addresses concurrently — a same-value race is still a race. *)
  (if shared then
     let axes = hardware_axes opt.llc in
     let wg_axes = List.filter axes ~f:(fun a -> match a.ha_kind with `Workgroup -> true | `Grid -> false) in
     let active_slots =
       List.filter_map wg_axes ~f:(fun a -> if a.ha_extent > 1 then Some a.ha_slot else None)
       |> List.dedup_and_sort ~compare:Int.compare
     in
     let covered =
       List.filter_map in_scope ~f:(fun fl ->
           match fl.axis with
           | Workgroup | Workgroup_reduce ->
               List.find_map wg_axes ~f:(fun a ->
                   if Indexing.equal_symbol a.ha_index fl.index then Some a.ha_slot else None)
           | Serial | Grid | Unrolled | Vectorized -> None)
     in
     (* Cooperative staging covers slot 0 by construction: the fresh lane loop is the innermost
        Workgroup loop of the load nest, and extent agreement with the kernel's other slot-0 loops
        is enforced downstream by [validate_parallel]'s barrier-strength uniformity. *)
     let covered = if Option.is_some cooperative then 0 :: covered else covered in
     let missing = List.filter active_slots ~f:(fun s -> not (List.mem covered s ~equal:Int.equal)) in
     if not (List.is_empty missing) then
       invalid_arg
         ("Schedule.Stage: workgroup slot(s) "
         ^ String.concat ~sep:", " (List.map missing ~f:Int.to_string)
         ^ " are active in this kernel but no loop binding them encloses the staging point for "
         ^ Tn.debug_name source
         ^ ": their threads would race on the shared tile; restructure the schedule so the \
            workgroup loops enclose the staging point (or reuse them as tile loops)"));
  (* Restrict redundant cooperative loading along in-scope workgroup axes that do not participate
     in this tile: one representative thread ([w == 0]) loads for all. *)
  let load_nest =
    if not shared then load_nest
    else
      List.fold in_scope ~init:load_nest ~f:(fun body fl ->
          match fl.axis with
          | (Workgroup | Workgroup_reduce)
            when not (List.mem reused fl.index ~equal:Indexing.equal_symbol) ->
              If
                {
                  cond =
                    ( Binop
                        ( Ops.Cmpeq,
                          (Embed_index (Indexing.Iterator fl.index), iprec),
                          (Constant 0., iprec) ),
                      iprec );
                  body;
                }
          | _ -> body)
  in
  (* The lane loop wraps the (possibly lane-restricted) load nest, including the [w == 0]
     restrictions along other in-scope workgroup axes; the barriers stay its siblings at the
     staging point — hardware [Workgroup] loops bind rather than iterate, so they remain
     uniformly reached. *)
  let load_nest =
    match lane with
    | None -> load_nest
    | Some (w_sym, w) ->
        let body =
          match lane_plan with
          | `Restrict_lane0 ->
              If
                {
                  cond =
                    ( Binop
                        ( Ops.Cmpeq,
                          (Embed_index (Indexing.Iterator w_sym), iprec),
                          (Constant 0., iprec) ),
                      iprec );
                  body = load_nest;
                }
          | `Serial_all | `Drop_inner _ | `Divide_inner _ -> load_nest
        in
        For_loop { index = w_sym; from_ = 0; to_ = w - 1; axis = Workgroup; trace_it = false; body }
  in
  let build inner =
    let remapped =
      remap_reads ~source ~from_idcs:idcs0 ~tile ~tile_idcs:tile_read_idcs inner
    in
    if shared then unflat_lines [ load_nest; Workgroup_barrier; remapped; Workgroup_barrier ]
    else unflat_lines [ load_nest; remapped ]
  in
  let llc =
    match lstar_depth with
    | Some ld ->
        rewrite_loop ~what:"Schedule.Stage" ~sym:stack0.(ld).index opt.llc ~f:(fun fc ->
            for_loop { fc with body = build fc.body })
    | None when wrap_outermost_tile ->
        rewrite_loop ~what:"Schedule.Stage" ~sym:stack0.(outermost_tile_depth).index opt.llc
          ~f:(fun fc -> build (for_loop fc))
    | None -> build opt.llc
  in
  {
    opt with
    llc;
    workgroup_shared =
      (if shared then Set.add opt.workgroup_shared tile else opt.workgroup_shared);
  }

(** {2 [Privatize]: accumulator privatization}

    Virtualization already privatizes accumulators — that is what [Local_scope] is — but it is
    forbidden from touching materialized nodes (they are observable). [Privatize { target; over }]
    recovers the same form inside one kernel for a materialized accumulation: the read-modify-write
    of [target] across the [over] loop's whole subtree is contracted to a per-thread [Local]
    accumulator tile — initialized from [target] before the loop, accumulated in place, stored
    back after — keeping a single final write per element. Because the tile is routine-local
    scratch whose address cannot alias the kernel's device pointers, downstream C/CUDA/MSL
    compilers can register-allocate it without [restrict] (gh-ocannl-164).

    Tile shape: per [target] axis, the index terms over loops nested inside [over] (they must be
    [Serial] — a workgroup-indexed private tile would make the store-back nest write other
    threads' elements) form the tile part sizing that axis; terms over loops outside [over] are
    the per-thread element selection, kept in the init-load and store-back indices. No tile part
    at all yields a scalar accumulator (dims [|1|]). Init/store nests iterate fresh serial
    symbols with per-axis edge guards (construct-then-fold, as in [Stage]). Any [Zero_out] of
    [target] elsewhere in the routine is left in place: the init-load observes its effect, so
    semantics are preserved without a surjectivity analysis (dropping the redundant zeroing is a
    follow-up). *)

let apply_privatize ~target ~over (opt : Low_level.optimized) : Low_level.optimized =
  let open Low_level in
  let iprec = Ops.index_prec () in
  let tgt_dims = Lazy.force target.Tn.dims in
  rewrite_loop ~what:"Schedule.Privatize" ~sym:over opt.llc ~f:(fun fc ->
      if not (equal_axis_type fc.axis Serial) then
        invalid_arg "Schedule.Privatize: the accumulation loop must be Serial";
      (* Accesses of [target] within the loop's subtree, with their loop stacks and enclosing
         [If] guard chains relative to it. Guards matter (PR #91 review): the remap keeps a guard
         on the update itself, but a thread-identifying guard (e.g. [w == 0] lane restriction)
         means only some threads' private accumulators receive updates — the init-load and
         store-back must then run under the {e same} predicate, or stale lanes clobber the
         result. *)
      let accesses = ref [] in
      let has_write = ref false in
      let rec scan stack conds llc =
        match llc with
        | Noop | Comment _ | Staged_compilation _ | Declare_local _ | Workgroup_barrier -> ()
        | Tile_mma _ -> invalid_arg "Schedule.Privatize: apply Privatize before Tensorize"
        | Zero_out tn ->
            if Tn.equal tn target then
              invalid_arg "Schedule.Privatize: Zero_out of the target inside the accumulation loop"
        | Seq (a, b) ->
            scan stack conds a;
            scan stack conds b
        | For_loop { index; from_; to_; body; trace_it; axis } ->
            scan ({ index; from_; to_; body = Noop; trace_it; axis } :: stack) conds body
        | Set { tn; idcs; llsc; _ } ->
            if Tn.equal tn target then (
              has_write := true;
              accesses := (idcs, stack, conds) :: !accesses);
            scan_scalar stack conds llsc
        | Set_from_vec { tn; arg = a, _; _ } ->
            if Tn.equal tn target then
              invalid_arg "Schedule.Privatize: vector writes to the target are unsupported";
            scan_scalar stack conds a
        | Set_local (_, llsc) -> scan_scalar stack conds llsc
        | If { cond = (c, _) as cond; body } ->
            scan_scalar stack conds c;
            scan stack (cond :: conds) body
      and scan_scalar stack conds (llsc : scalar_t) =
        match llsc with
        | Local_scope { body; _ } -> scan stack conds body
        | Get_local _ | Constant _ | Constant_bits _ | Embed_index _ -> ()
        | Get (tn, idcs) ->
            if Tn.equal tn target then accesses := (idcs, stack, conds) :: !accesses
        | Get_dynamic { tn; dyn_value = v, _; _ } ->
            if Tn.equal tn target then
              invalid_arg "Schedule.Privatize: dynamically indexed target accesses are unsupported";
            scan_scalar stack conds v
        | Get_merge_buffer (_, _) -> ()
        | Ternop (_, (a, _), (b, _), (c, _)) ->
            scan_scalar stack conds a;
            scan_scalar stack conds b;
            scan_scalar stack conds c
        | Binop (_, (a, _), (b, _)) ->
            scan_scalar stack conds a;
            scan_scalar stack conds b
        | Unop (_, (a, _)) -> scan_scalar stack conds a
      in
      scan [] [] fc.body;
      let idcs0 =
        match !accesses with
        | [] ->
            invalid_arg
              ("Schedule.Privatize: no accesses of " ^ Tn.debug_name target
             ^ " under the accumulation loop")
        | (idcs, _, _) :: _ -> idcs
      in
      if not !has_write then
        invalid_arg ("Schedule.Privatize: " ^ Tn.debug_name target ^ " is not written (no accumulation)");
      List.iter !accesses ~f:(fun (idcs, _, _) ->
          if not (Array.equal Indexing.equal_axis_index idcs idcs0) then
            invalid_arg
              ("Schedule.Privatize: v1 requires all accesses of " ^ Tn.debug_name target
             ^ " under the loop to use identical index vectors"));
      (* Guard chains must agree across accesses, and must be iteration-invariant — free of
         memory reads and of symbols bound by [over] or any loop inside its subtree — so the
         same predicate can gate the init-load and store-back. A per-iteration (data- or
         index-dependent) guard cannot be contracted: rejected. *)
      let conds0 = match !accesses with (_, _, conds) :: _ -> conds | [] -> [] in
      List.iter !accesses ~f:(fun (_, _, conds) ->
          if not (List.equal equal_scalar_arg conds conds0) then
            invalid_arg
              ("Schedule.Privatize: accesses of " ^ Tn.debug_name target
             ^ " sit under differing If guards"));
      (if not (List.is_empty conds0) then
         let bound_inside =
           let acc = ref [ over ] in
           let rec go llc =
             match llc with
             | For_loop { index; body; _ } ->
                 acc := index :: !acc;
                 go body
             | Seq (a, b) ->
                 go a;
                 go b
             | If { body; _ } -> go body
             | Set { llsc; _ } | Set_local (_, llsc) -> go_scalar llsc
             | Set_from_vec { arg = a, _; _ } -> go_scalar a
             | Noop | Comment _ | Staged_compilation _ | Zero_out _ | Declare_local _
             | Workgroup_barrier | Tile_mma _ ->
                 ()
           and go_scalar (llsc : scalar_t) =
             match llsc with
             | Local_scope { body; _ } -> go body
             | Get_dynamic { dyn_value = v, _; _ } -> go_scalar v
             | Ternop (_, (a, _), (b, _), (c, _)) ->
                 go_scalar a;
                 go_scalar b;
                 go_scalar c
             | Binop (_, (a, _), (b, _)) ->
                 go_scalar a;
                 go_scalar b
             | Unop (_, (a, _)) -> go_scalar a
             | Get_local _ | Get _ | Get_merge_buffer _ | Constant _ | Constant_bits _
             | Embed_index _ ->
                 ()
           in
           go fc.body;
           !acc
         in
         let sym_free s = not (List.mem bound_inside s ~equal:Indexing.equal_symbol) in
         let idx_invariant = function
           | Indexing.Fixed_idx _ | Indexing.Sub_axis -> true
           | Indexing.Iterator s -> sym_free s
           | Indexing.Affine { symbols; _ } -> List.for_all symbols ~f:(fun (_, s) -> sym_free s)
           | Indexing.Concat _ -> false
         in
         let rec invariant (llsc : scalar_t) =
           match llsc with
           | Constant _ | Constant_bits _ -> true
           | Embed_index idx -> idx_invariant idx
           | Ternop (_, (a, _), (b, _), (c, _)) -> invariant a && invariant b && invariant c
           | Binop (_, (a, _), (b, _)) -> invariant a && invariant b
           | Unop (_, (a, _)) -> invariant a
           | Get _ | Get_local _ | Get_dynamic _ | Get_merge_buffer _ | Local_scope _ -> false
         in
         List.iter conds0 ~f:(fun (c, _) ->
             if not (invariant c) then
               invalid_arg
                 ("Schedule.Privatize: accesses of " ^ Tn.debug_name target
                ^ " sit under an If guard that varies across the accumulation (mentions memory \
                   or a symbol bound inside the loop); cannot gate the init/store-back with it")));
      (* Loops bound inside the subtree, by index symbol (union over access paths). *)
      let inner_loop s =
        List.find_map !accesses ~f:(fun (_, stack, _) ->
            List.find stack ~f:(fun fl -> Indexing.equal_symbol fl.index s))
      in
      (* Per axis: tile part (terms over inner loops) and outer part. The [over] symbol itself
         must not occur — the accumulator is carried across that loop. *)
      let decomp =
        Array.map idcs0 ~f:(fun idx ->
            match terms_of_index idx with
            | None -> invalid_arg "Schedule.Privatize: Concat indices are unsupported"
            | Some (terms, offset) ->
                if List.exists terms ~f:(fun (_, s) -> Indexing.equal_symbol s over) then
                  invalid_arg
                    ("Schedule.Privatize: the target's indices mention the accumulation loop "
                    ^ Indexing.symbol_ident over ^ " — nothing to carry the accumulator across");
                let tile_part, outer_part =
                  List.partition_tf terms ~f:(fun (_, s) -> Option.is_some (inner_loop s))
                in
                List.iter tile_part ~f:(fun (c, s) ->
                    if c <= 0 then
                      invalid_arg "Schedule.Privatize: nonpositive coefficient on an inner index";
                    let fl = Option.value_exn (inner_loop s) in
                    if not (equal_axis_type fl.axis Serial) || fl.from_ <> 0 then
                      invalid_arg
                        ("Schedule.Privatize: inner index loop " ^ Indexing.symbol_ident s
                       ^ " must be Serial starting at 0 (a workgroup-indexed private tile would \
                          store back other threads' elements)"));
                (tile_part, outer_part, offset))
      in
      let extent s =
        let fl = Option.value_exn (inner_loop s) in
        fl.to_ - fl.from_ + 1
      in
      let tile_axes =
        Array.filter_mapi decomp ~f:(fun a (tp, _, _) ->
            if List.is_empty tp then None
            else Some (a, List.fold tp ~init:1 ~f:(fun acc (c, s) -> acc + (c * (extent s - 1)))))
      in
      let scalar_acc = Array.is_empty tile_axes in
      let tile_dims = if scalar_acc then [| 1 |] else Array.map tile_axes ~f:snd in
      let prec = Lazy.force target.Tn.prec in
      let tile =
        Tn.create ~namespace:tile_namespace (Tn.Specified prec) ~id:(fresh_tile_id ())
          ~label:("acc" :: target.Tn.label)
          ~unpadded_dims:(lazy tile_dims)
          ~padding:(lazy None) ()
      in
      Tn.Placements.update opt.Low_level.optimize_ctx.placements tile Tn.Local 176;
      ignore (get_node opt.traced_store tile : traced_array);
      let tile_read_idcs =
        if scalar_acc then [| Indexing.Fixed_idx 0 |]
        else
          Array.map tile_axes ~f:(fun (a, _) ->
              let tp, _, _ = decomp.(a) in
              normalize_affine ~terms:tp ~offset:0)
      in
      (* Init-load and store-back nests over fresh serial symbols (two independent sets). *)
      let transfer ~into_tile =
        let fresh_syms =
          Array.fold_right tile_axes
            ~init:(Map.empty (module Indexing.Symbol))
            ~f:(fun (a, _) m ->
              let tp, _, _ = decomp.(a) in
              List.fold tp ~init:m ~f:(fun m (_, s) ->
                  if Map.mem m s then m else Map.set m ~key:s ~data:(Indexing.get_symbol ())))
        in
        let load_sym s = Option.value (Map.find fresh_syms s) ~default:s in
        let subst_terms terms = List.map terms ~f:(fun (c, s) -> (c, load_sym s)) in
        let src_idcs =
          Array.map decomp ~f:(fun (tp, op_, off) ->
              normalize_affine ~terms:(subst_terms tp @ op_) ~offset:off)
        in
        let t_idcs =
          if scalar_acc then [| Indexing.Fixed_idx 0 |]
          else
            Array.map tile_axes ~f:(fun (a, _) ->
                let tp, _, _ = decomp.(a) in
                normalize_affine ~terms:(subst_terms tp) ~offset:0)
        in
        let stmt =
          if into_tile then
            Set { tn = tile; idcs = t_idcs; llsc = Get (target, src_idcs); debug = "" }
          else Set { tn = target; idcs = src_idcs; llsc = Get (tile, t_idcs); debug = "" }
        in
        (* Per-axis edge guards (construct-then-fold; they survive only for non-dividing tiles). *)
        let stmt =
          Array.fold tile_axes ~init:stmt ~f:(fun stmt (a, _) ->
              let cond =
                Binop
                  ( Ops.Cmplt,
                    (Embed_index src_idcs.(a), iprec),
                    (Constant (Float.of_int tgt_dims.(a)), iprec) )
              in
              If { cond = (cond, iprec); body = stmt })
        in
        let nest =
          Map.fold fresh_syms ~init:stmt ~f:(fun ~key:s ~data:s' body ->
              For_loop
                { index = s'; from_ = 0; to_ = extent s - 1; body; trace_it = false; axis = Serial })
        in
        (* Carry the accesses' (uniform, iteration-invariant) guard chain onto the transfers:
           only the lanes that update their private accumulator may load and store it back
           (PR #91 review). *)
        List.fold conds0 ~init:nest ~f:(fun body cond -> If { cond; body })
      in
      let remapped =
        remap_reads ~writes:true ~source:target ~from_idcs:idcs0 ~tile ~tile_idcs:tile_read_idcs
          fc.body
      in
      unflat_lines
        [ transfer ~into_tile:true; for_loop { fc with body = remapped }; transfer ~into_tile:false ])
  |> fun llc -> { opt with llc }

let apply_opt_op (opt : Low_level.optimized) (op : optop) : Low_level.optimized =
  match op with
  | Stage { source; tile_loops; shared; cooperative; hoisted } ->
      apply_stage ~source ~tile_loops ~shared ~cooperative ~hoisted opt
  | Privatize { target; over } -> apply_privatize ~target ~over opt
  | (Split _ | Swap _ | Retype _ | Unroll _ | Expand_zero _ | Tensorize _) as op ->
      { opt with llc = apply_op opt.Low_level.llc op }

let apply ?(static_indices = []) (sched : schedule) (opt : Low_level.optimized) :
    Low_level.optimized =
  if List.is_empty sched then opt
  else
    let opt = List.fold sched ~init:opt ~f:apply_opt_op in
    let llc = opt.Low_level.llc in
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
    - Cross-nest pairs over a written node (producer/consumer, WAW, WAR) are races once threads
      interleave — sibling nests execute with no global synchronization between them — unless the
      accesses are {e aligned}: each thread then touches only its own elements in both nests, and
      a thread's writes are visible to its own later reads by program order, so values match the
      serial execution. Nests transitively linked by such pairs form a dependency component;
      alignment of a component means: (a) every member's chain is trimmed to one common prefix
      with pointwise-equal extents — the annotator then emits identical geometry for all members,
      so a hardware thread maps to the same chain-index tuple in every member nest; and (b) per
      axis position of every write/access pair over a shared node, either neither side's
      component mentions its own nest's (trimmed) parallel symbols, or both are plain [Iterator]s
      of symbols at the same chain position. Statically-unknown ([Get_dynamic]) accesses never
      align. If no common trim depth >= 1 aligns every pair of the component, the analysis bails.
      Bare statements (outside every nest, executed unconditionally by every thread) have no
      parallel symbols, so a bare access to a nest-written node bails by the same rule.
      Non-materialized (routine-local) scratch is per-thread, hence exempt when its writes
      mention no parallel symbols (each thread writes its whole private copy); otherwise it needs
      alignment like a materialized node (each thread reads back exactly the slice of its private
      copy that it wrote).
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
let scan_accesses plc (llc : Low_level.t) : access list =
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
    (* Already-scheduled cooperative code: the default annotator leaves it alone. *)
    | Tile_mma _ -> raise Bail
    | Seq (a, b) ->
        code ~depth a;
        code ~depth b
    | For_loop { axis; body; _ } ->
        if not (equal_axis_type axis Serial) then raise Bail;
        code ~depth body
    | Zero_out tn ->
        if Tn.Placements.is_materialized_peek plc tn then raise Bail
        (* Zeroing per-thread scratch is safe: each thread zeroes its own copy. *)
    | Set { tn; idcs; llsc; _ } ->
        if depth > 0 && Tn.Placements.is_materialized_peek plc tn then raise Bail;
        add ~depth ~write:true ~dynamic:false tn idcs;
        scalar ~depth llsc
    | Set_from_vec { tn; idcs; arg = a, _; _ } ->
        if depth > 0 && Tn.Placements.is_materialized_peek plc tn then raise Bail;
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
let split_nests plc (llc : Low_level.t) : nest_info list * access list =
  let open Low_level in
  let rec is_nest = function
    | For_loop _ -> true
    | If { body; _ } -> is_nest body
    | _ -> false
  in
  let stmts = flat_lines [ llc ] in
  let nests, bare =
    List.partition_map stmts ~f:(fun stmt ->
        if is_nest stmt then First { n_loops = stmt; n_accesses = scan_accesses plc stmt }
        else Second (scan_accesses plc stmt))
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

(* Shared analysis of the default annotator presets (schedule-ir-optops §6): per top-level nest,
   the parallelizable chain of outermost Serial path loops, validated by the conservative race
   analysis (see {!default_gpu}'s doc). Nests linked by cross-nest dependencies keep a common
   aligned prefix of their chains (see the module comment). Raises [Bail] when any check fails. *)
let analyze_parallel_chains (opt : Low_level.optimized) : Low_level.t list list =
  let open Low_level in
  let plc = opt.optimize_ctx.placements in
  let nests, bare = split_nests plc opt.llc in
  (* Bare materialized writes cannot be covered by annotated loops. *)
  if List.exists bare ~f:(fun a -> a.a_write && Tn.Placements.is_materialized_peek plc a.a_tn) then
    raise Bail;
  (* Chains: per nest, the outermost (up to two) Serial path loops whose index occurs as a plain
     [Iterator] component in every materialized write vector of the nest. *)
  let chains =
    List.map nests ~f:(fun n ->
        let mat_writes =
          List.filter n.n_accesses ~f:(fun a ->
              a.a_write && Tn.Placements.is_materialized_peek plc a.a_tn)
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
  (* Access groups: the nests plus the bare pseudo-group (executed unconditionally, no chain). *)
  let groups =
    Array.of_list (List.map2_exn nests chains ~f:(fun n c -> (n.n_accesses, c)) @ [ (bare, []) ])
  in
  let n_groups = Array.length groups in
  let group_tbls =
    Array.map groups ~f:(fun (accs, _) ->
        let tbl = Hashtbl.create (module Int) in
        List.iter accs ~f:(fun a -> Hashtbl.add_multi tbl ~key:a.a_tn.Tn.uid ~data:a);
        tbl)
  in
  let full_syms = Array.map groups ~f:(fun (_, c) -> chain_syms c) in
  let sym_arr = Array.map full_syms ~f:Array.of_list in
  let extent_arr =
    Array.map groups ~f:(fun (_, c) ->
        Array.of_list (List.filter_map c ~f:(function For_loop fc -> Some (fc.to_ + 1) | _ -> None)))
  in
  (* Cross-group dependency edges (module comment, aligned cross-nest parallelism): a node
     written in one group and touched in another must have all its write/access pairs aligned —
     unless it is per-thread scratch whose writes never mention the writer's parallel symbols
     (every thread then writes its whole private copy, and cross-nest reads are per-thread
     coherent regardless of geometry). *)
  let edges = ref [] in
  for i = 0 to n_groups - 1 do
    for j = i + 1 to n_groups - 1 do
      Hashtbl.iteri group_tbls.(i) ~f:(fun ~key:uid ~data:accs_i ->
          match Hashtbl.find group_tbls.(j) uid with
          | None -> ()
          | Some accs_j ->
              let writes_mention accs syms =
                List.exists accs ~f:(fun a ->
                    a.a_write && Array.exists a.a_idcs ~f:(mentions_sym syms))
              in
              if
                (List.exists accs_i ~f:(fun a -> a.a_write)
                || List.exists accs_j ~f:(fun a -> a.a_write))
                && (Tn.Placements.is_materialized_peek plc (List.hd_exn accs_i).a_tn
                   || writes_mention accs_i full_syms.(i)
                   || writes_mention accs_j full_syms.(j))
              then edges := (i, j, uid) :: !edges)
    done
  done;
  (* [trims.(g)]: how many chain loops group [g] keeps; alignment may lower it (uniformly per
     dependency component, so the annotator emits identical geometry for linked nests). *)
  let trims = Array.map groups ~f:(fun (_, c) -> List.length c) in
  (if not (List.is_empty !edges) then
     let parent = Array.init n_groups ~f:Fn.id in
     let rec find x =
       if parent.(x) = x then x
       else (
         parent.(x) <- find parent.(x);
         parent.(x))
     in
     List.iter !edges ~f:(fun (i, j, _) ->
         let ri = find i and rj = find j in
         if ri <> rj then parent.(ri) <- rj);
     (* One write/access pair at trim depth [l]: statically-known indices; per axis position,
        parallel symbols are mentioned on both sides or neither; where both, plain [Iterator]s at
        the same chain position (extents are equal by the [l_max] prefix rule below). *)
     let pair_aligned ~l gi gj (a : access) (b : access) =
       (not a.a_dynamic) && (not b.a_dynamic)
       &&
       let syms_i = List.take full_syms.(gi) l and syms_j = List.take full_syms.(gj) l in
       let pos g s =
         Array.findi sym_arr.(g) ~f:(fun k s' -> k < l && Indexing.equal_symbol s s')
         |> Option.map ~f:fst
       in
       let comp idcs p = if p < Array.length idcs then idcs.(p) else Indexing.Fixed_idx 0 in
       let rank = max (Array.length a.a_idcs) (Array.length b.a_idcs) in
       let aligned_at p =
         let ci = comp a.a_idcs p and cj = comp b.a_idcs p in
         match (mentions_sym syms_i ci, mentions_sym syms_j cj) with
         | false, false -> true
         | true, true -> (
             match (ci, cj) with
             | Indexing.Iterator si, Indexing.Iterator sj -> (
                 match (pos gi si, pos gj sj) with
                 | Some ki, Some kj -> ki = kj
                 | _ -> false)
             (* An [Affine] over a parallel symbol is never a plain aligned slice. *)
             | _ -> false)
         | _ -> false
       in
       List.for_all (List.init rank ~f:Fn.id) ~f:aligned_at
     in
     let edge_aligned ~l (i, j, uid) =
       let accs_i = Hashtbl.find_multi group_tbls.(i) uid
       and accs_j = Hashtbl.find_multi group_tbls.(j) uid in
       List.for_all accs_i ~f:(fun a ->
           List.for_all accs_j ~f:(fun b ->
               (not (a.a_write || b.a_write)) || pair_aligned ~l i j a b))
     in
     let comps = Hashtbl.create (module Int) in
     Array.iteri parent ~f:(fun g _ -> Hashtbl.add_multi comps ~key:(find g) ~data:g);
     Hashtbl.iteri comps ~f:(fun ~key:root ~data:members ->
         if List.length members > 1 then (
           let comp_edges = List.filter !edges ~f:(fun (i, _, _) -> find i = root) in
           (* Longest prefix of the members' chains with pointwise-equal extents: the geometry
              must be identical for the thread->iteration maps to coincide across the component
              (a 1-loop chain splits Grid x Workgroup while a 2-loop chain retypes in place, so
              even a shared axis maps differently across different chain shapes). *)
           let l_max =
             List.fold members ~init:2 ~f:(fun m g -> min m (Array.length extent_arr.(g)))
           in
           let ext_eq k =
             match members with
             | [] -> true
             | g0 :: rest -> List.for_all rest ~f:(fun g -> extent_arr.(g).(k) = extent_arr.(g0).(k))
           in
           let rec agree k = if k >= l_max || not (ext_eq k) then k else agree (k + 1) in
           let l_max = agree 0 in
           let rec search l =
             if l < 1 then raise Bail
             else if List.for_all comp_edges ~f:(edge_aligned ~l) then
               List.iter members ~f:(fun g -> trims.(g) <- l)
             else search (l - 1)
           in
           search l_max)));
  let chains = List.mapi chains ~f:(fun i c -> List.take c trims.(i)) in
  (* Per-nest hazard analysis over the final (possibly trimmed) parallel symbols (see the module
     comment for the safety argument). *)
  List.iter2_exn nests chains ~f:(fun n chain ->
      let syms = chain_syms chain in
      let by_tn = Hashtbl.create (module Int) in
      List.iter n.n_accesses ~f:(fun a -> Hashtbl.add_multi by_tn ~key:a.a_tn.Tn.uid ~data:a);
      Hashtbl.iter by_tn ~f:(fun accs ->
          let written = List.exists accs ~f:(fun a -> a.a_write) in
          if written then (
            let is_mat = Tn.Placements.is_materialized_peek plc (List.hd_exn accs).a_tn in
            let chain_relevant =
              List.exists accs ~f:(fun a -> Array.exists a.a_idcs ~f:(mentions_sym syms))
            in
            if is_mat || chain_relevant then (
              if List.exists accs ~f:(fun a -> a.a_dynamic) then raise Bail;
              (* All accesses must agree on every component that mentions a parallel symbol. *)
              let rank = List.fold accs ~init:0 ~f:(fun m a -> max m (Array.length a.a_idcs)) in
              for p = 0 to rank - 1 do
                let comps =
                  List.map accs ~f:(fun a ->
                      if p < Array.length a.a_idcs then a.a_idcs.(p) else Indexing.Fixed_idx 0)
                in
                if List.exists comps ~f:(mentions_sym syms) then
                  match comps with
                  | [] -> ()
                  | c0 :: rest ->
                      if not (List.for_all rest ~f:(Indexing.equal_axis_index c0)) then raise Bail
              done))));
  chains

(* Parallel iterations a chain covers. *)
let chain_size chain =
  List.fold chain ~init:1 ~f:(fun sz -> function
    | Low_level.For_loop fc -> sz * (fc.to_ + 1) | _ -> sz)

(* Threshold helper: skip kernels whose largest parallelizable nest is too small to pay for a
   launch (GPU) or a task fan-out (CPU). *)
let max_parallel_size chains = List.fold chains ~init:0 ~f:(fun m chain -> max m (chain_size chain))

let default_gpu ?block_size ?min_parallel ?(limits = Backend_intf.no_hardware_limits)
    (opt : Low_level.optimized) : schedule =
  let open Low_level in
  let block_size =
    Option.value block_size
      ~default:
        (Int.of_string @@ Utils.get_global_arg ~arg_name:"gpu_schedule_block_size" ~default:"256")
  in
  (* The configured block size is a target, the device's workgroup capacity is a hard cap. *)
  let block_size =
    Option.value_map limits.Backend_intf.max_threads_per_workgroup ~default:block_size
      ~f:(min block_size)
  in
  let min_parallel =
    Option.value min_parallel
      ~default:
        (Int.of_string @@ Utils.get_global_arg ~arg_name:"gpu_schedule_min_parallel" ~default:"1024")
  in
  try
    let chains = analyze_parallel_chains opt in
    if max_parallel_size chains < min_parallel then []
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

let default_cpu ?min_parallel (opt : Low_level.optimized) : schedule =
  let min_parallel =
    Option.value min_parallel
      ~default:
        (Int.of_string
        @@ Utils.get_global_arg ~arg_name:"cpu_schedule_min_parallel" ~default:"16384")
  in
  try
    let chains = analyze_parallel_chains opt in
    if max_parallel_size chains < min_parallel then []
    else
      (* Retype the outermost chain loop to [Grid], nothing else: pool-backed Grid rendering
         (gh-ocannl-164) partitions that loop into contiguous chunks, and a [Workgroup] split
         would only add loop structure that executes serially inside a chunk. *)
      List.concat_map chains ~f:(function
        | Low_level.For_loop fc :: _ -> [ Retype { axis = fc.index; ty = Low_level.Grid } ]
        | _ -> [])
  with Bail -> []

(** {2 Kernel fission at cross-workgroup edges}

    The default annotator's cross-nest interference check (see {!analyze_parallel_chains}) is a
    fact about hardware threads: sibling nests of one kernel execute with no grid-wide
    synchronization between them, so a producer/consumer (or WAW/WAR) pair of top-level statements
    over a materialized node is a race once threads interleave — and the whole routine used to
    fall back to a 1×1 launch. Fission recovers the synchronization from the stream instead:
    top-level statements are partitioned into {e segments} at those dependency edges, each
    segment compiles to its own kernel, and the routine's task launches them in order on the
    routine's stream with a device-side event chained at each boundary (queue FIFO alone does not
    order overlapping command buffers over Metal's untracked resources; see [Raise_backend.link]).
    Each segment then gets the default schedule independently, on its own launch geometry.
    Dependency edges the aligned cross-nest rule proves race-free do {e not} cut, provided the
    shared kernel keeps every nest's standalone parallelism (see {!aligned_merge}): elementwise
    chains over one intermediate stay a single kernel, while parallelism switches (a batch-parallel
    producer feeding a reduce-over-batch consumer) and alignment-trimming merges still cut.

    Segmentation is conservative and total (no [Bail]): a statement opaque to the analysis
    ([Staged_compilation], barriers, pre-annotated loops) or one the annotator can never cover
    (bare materialized writes, materialized writes inside [Local_scope] bodies, non-injective
    nests) is isolated into its own serial segment rather than poisoning its neighbors' schedules.
    Materialized whole-node [Zero_out]s are likewise isolated and — on GPU — expanded
    ({!optop.Expand_zero}) and annotated with the same geometry policy as ordinary nests.

    Two constructs cross segment boundaries and need repair, because a kernel's locals die at
    launch end:

    - Scalar scope-locals hoisted by [Low_level.hoist_cross_statement_cse] (top-level
      [Declare_local] + defining statements): the defining statements are {e replicated} at the
      head of every consuming segment. This is valid exactly when nothing written between the
      definition's original position and the segment start overlaps the definition's reads (the
      replica then computes the same value); otherwise the offending segments are merged back and
      run serially, preserving today's behavior.
    - [Local]-placed scratch tensor nodes whose accesses end up split across segments are promoted
      to [On_device] ({!Tnode.Placements.promote_local_to_device}): [Local] is always a compiler
      decision premised on single-kernel lifetime, and fission runs before any consumer of the
      decision (codegen parameter lists, context allocation) has read it.

    Adjacent segments that both end up unannotated are coalesced back (fewer launches); if
    everything coalesces, the routine compiles to a single kernel exactly as before. *)

(* Per top-level statement: tensor-node access sets, scope-local references crossing statement
   boundaries, and the statement kind for segmentation. [None] = opaque to the analysis. *)
type stmt_summary = {
  s_reads : Set.M(Tn).t;
  s_writes : Set.M(Tn).t;  (** Includes [Zero_out] targets. *)
  s_top_zero : Tn.t option;
      (** The statement is exactly [Zero_out tn] of a materialized node (expandable). *)
  s_scope_reads : Low_level.scope_id list;
      (** [Get_local]s of scope ids bound outside this statement. *)
  s_scope_writes : Low_level.scope_id list;
      (** [Set_local]s of scope ids bound outside this statement. *)
  s_scope_declares : Low_level.scope_id list;  (** [Declare_local]s within this statement. *)
}

exception Opaque_stmt

let summarize_stmt plc (stmt : Low_level.t) : stmt_summary option =
  let open Low_level in
  let reads = ref (Set.empty (module Tn)) and writes = ref (Set.empty (module Tn)) in
  let top_zero = ref None in
  let scope_reads = ref [] and scope_writes = ref [] in
  let bound = ref [] and declares = ref [] in
  let rec code ~top llc =
    match llc with
    | Noop | Comment _ -> ()
    | Staged_compilation _ | Workgroup_barrier | Tile_mma _ -> raise Opaque_stmt
    | Declare_local { id; _ } -> declares := id :: !declares
    | Seq (a, b) ->
        code ~top a;
        code ~top b
    | For_loop { axis; body; _ } ->
        if not (equal_axis_type axis Serial) then raise Opaque_stmt;
        code ~top:false body
    | Zero_out tn ->
        writes := Set.add !writes tn;
        if top && Tn.Placements.is_materialized_peek plc tn then top_zero := Some tn
    | Set { tn; llsc; _ } ->
        writes := Set.add !writes tn;
        scalar llsc
    | Set_from_vec { tn; arg = a, _; _ } ->
        writes := Set.add !writes tn;
        scalar a
    | Set_local (id, llsc) ->
        scope_writes := id :: !scope_writes;
        scalar llsc
    | If { cond = c, _; body } ->
        scalar c;
        code ~top:false body
  and scalar (llsc : scalar_t) =
    match llsc with
    | Local_scope { id; body; _ } ->
        bound := id :: !bound;
        code ~top:false body
    | Get_local id -> scope_reads := id :: !scope_reads
    | Get (tn, _) -> reads := Set.add !reads tn
    | Get_dynamic { tn; dyn_value = v, _; _ } ->
        reads := Set.add !reads tn;
        scalar v
    | Get_merge_buffer (_, _) -> () (* A separate read-only input buffer: never a hazard. *)
    | Constant _ | Constant_bits _ | Embed_index _ -> ()
    | Ternop (_, (a, _), (b, _), (c, _)) ->
        scalar a;
        scalar b;
        scalar c
    | Binop (_, (a, _), (b, _)) ->
        scalar a;
        scalar b
    | Unop (_, (a, _)) -> scalar a
  in
  match code ~top:true stmt with
  | exception Opaque_stmt -> None
  | () ->
      let internal = !bound @ !declares in
      let externals ids =
        List.filter ids ~f:(fun id ->
            not (List.mem internal id ~equal:Low_level.equal_scope_id))
        |> List.dedup_and_sort ~compare:Low_level.compare_scope_id
      in
      Some
        {
          s_reads = !reads;
          s_writes = !writes;
          s_top_zero = !top_zero;
          s_scope_reads = externals !scope_reads;
          s_scope_writes = externals !scope_writes;
          s_scope_declares = List.dedup_and_sort !declares ~compare:Low_level.compare_scope_id;
        }

type funit = {
  f_stmts : Low_level.t list;  (** Leading comments/noops + the statement, original order. *)
  f_index : int;  (** Position among units, for scope-replication validity ranges. *)
  f_sum : stmt_summary option;
  f_kind : [ `Normal | `Zeros | `Solo ];
  f_chains : Low_level.t list list;
      (** The statement's chains analyzed standalone ([[]] for non-[`Normal] units): the
          no-parallelism-loss baseline for aligned segment merging (see {!aligned_merge}). *)
}

(* Units: each real statement with the comments preceding it. Trailing comments attach to the last
   unit. *)
let collect_units plc (opt : Low_level.optimized) (stmts : Low_level.t list) : funit list =
  let open Low_level in
  let is_glue = function Noop | Comment _ -> true | _ -> false in
  let mk index glue stmt =
    let f_sum = summarize_stmt plc stmt in
    let f_kind, f_chains =
      match f_sum with
      | None -> (`Solo, [])
      | Some { s_top_zero = Some _; _ } -> (`Zeros, [])
      | Some _ -> (
          (* Isolate statements the annotator could never share a parallel kernel with: it raises
             [Bail] on them even in isolation (bare materialized writes, materialized writes
             inside [Local_scope] bodies, dynamic accesses or non-injective/uncovered writes).
             In its own segment such a statement runs as a serial 1×1 launch and its neighbors
             keep their parallelism. Read-only statements never bail; their chains are empty. *)
          match analyze_parallel_chains { opt with llc = stmt } with
          | chains -> (`Normal, chains)
          | exception Bail -> (`Solo, []))
    in
    { f_stmts = List.rev (stmt :: glue); f_index = index; f_sum; f_kind; f_chains }
  in
  let rec go index glue acc = function
    | [] -> (
        (* Trailing glue: attach to the last unit. *)
        match (acc, glue) with
        | _, [] | [], _ -> List.rev acc
        | last :: rest, glue ->
            List.rev ({ last with f_stmts = last.f_stmts @ List.rev glue } :: rest))
    | stmt :: tl when is_glue stmt -> go index (stmt :: glue) acc tl
    | stmt :: tl -> go (index + 1) [] (mk index glue stmt :: acc) tl
  in
  go 0 [] [] stmts

type segment = {
  g_units : funit list;
  g_kind : [ `Normal | `Zeros | `Solo ];
  g_reads : Set.M(Tn).t;
  g_writes : Set.M(Tn).t;
  g_chains : Low_level.t list list;
      (** Concatenation of the units' standalone chains, in statement order (the merged code's
          nest order): the baseline for {!aligned_merge}'s no-parallelism-loss guard. *)
}

let seg_of_unit u =
  let reads, writes =
    match u.f_sum with
    | Some s -> (s.s_reads, s.s_writes)
    | None -> (Set.empty (module Tn), Set.empty (module Tn))
  in
  { g_units = [ u ]; g_kind = u.f_kind; g_reads = reads; g_writes = writes; g_chains = u.f_chains }

let merge_segs ~kind a b =
  {
    g_units = a.g_units @ b.g_units;
    g_kind = kind;
    g_reads = Set.union a.g_reads b.g_reads;
    g_writes = Set.union a.g_writes b.g_writes;
    g_chains = a.g_chains @ b.g_chains;
  }

(* The within-kernel interference rule of {!analyze_parallel_chains}, reformulated as a cut
   criterion: a materialized node written on one side and touched on the other must not share a
   kernel — unless the merged statements pass the aligned cross-nest analysis without losing
   parallelism ({!aligned_merge}). Local-scratch pairs deliberately do NOT cut — the per-segment
   annotator retains the finer per-thread-privacy analysis for those (and simply declines to
   annotate when it fails). *)
let mat_conflict plc seg (s : stmt_summary) =
  let mat tn = Tn.Placements.is_materialized_peek plc tn in
  let hits w touched = Set.exists w ~f:(fun tn -> mat tn && Set.mem touched tn) in
  hits seg.g_writes (Set.union s.s_reads s.s_writes)
  || hits s.s_writes (Set.union seg.g_reads seg.g_writes)

(* Whether extending [seg] with [u] keeps one parallelizable kernel: the merged statements pass
   {!analyze_parallel_chains} (its aligned cross-nest rule proves the dependency race-free) AND no
   nest's parallel size shrinks versus its standalone analysis. Alignment may trim linked chains
   to a common prefix; when it would, cutting (a launch boundary, today's behavior) is the better
   default than trading parallelism for one launch — and keeps the fissioned candidates of the
   schedule search at full per-segment parallelism. *)
let aligned_merge (opt : Low_level.optimized) seg (u : funit) : bool =
  let llc =
    Low_level.unflat_lines (List.concat_map (seg.g_units @ [ u ]) ~f:(fun u' -> u'.f_stmts))
  in
  match analyze_parallel_chains { opt with llc } with
  | exception Bail -> false
  | merged -> (
      match
        List.for_all2 merged (seg.g_chains @ u.f_chains) ~f:(fun m s ->
            chain_size m >= chain_size s)
      with
      | List.Or_unequal_lengths.Ok ok -> ok
      | Unequal_lengths -> false)

let group_units (opt : Low_level.optimized) (units : funit list) : segment list =
  let plc = opt.Low_level.optimize_ctx.placements in
  let close cur acc = match cur with None -> acc | Some seg -> seg :: acc in
  let rec go cur acc = function
    | [] -> List.rev (close cur acc)
    | u :: tl -> (
        match (cur, u.f_kind, u.f_sum) with
        | Some ({ g_kind = `Normal; _ } as seg), `Normal, Some s
          when (not (mat_conflict plc seg s)) || aligned_merge opt seg u ->
            go (Some (merge_segs ~kind:`Normal seg (seg_of_unit u))) acc tl
        | Some ({ g_kind = `Zeros; _ } as seg), `Zeros, Some s when not (mat_conflict plc seg s) ->
            go (Some (merge_segs ~kind:`Zeros seg (seg_of_unit u))) acc tl
        | _ -> go (Some (seg_of_unit u)) (close cur acc) tl)
  in
  go None [] units

(** {3 Scope-local replication across segments (option (b) v2)} *)

exception Unfissionable
(* Malformed scope-local shape (a scope id referenced with no defining statement before it, or a
   merge cascade reaching the first segment): the driver falls back to single-kernel compilation.
   Repairable crossings do not raise — they return [None] from [plan_replicas] and merge their
   segment range instead. *)

(* Def units of scope id [id] strictly before unit index [limit]: the declaring unit and every
   unit that [Set_local]s it from outside. *)
let def_units_of (units : funit array) ~limit id =
  Array.to_list units
  |> List.filter ~f:(fun u ->
         u.f_index < limit
         &&
         match u.f_sum with
         | None -> false
         | Some s ->
             List.mem s.s_scope_declares id ~equal:Low_level.equal_scope_id
             || List.mem s.s_scope_writes id ~equal:Low_level.equal_scope_id)

(* The replica set for one segment: transitive closure of def units over the scope ids they read
   themselves. Returns [None] when replication is invalid — a def unit writes tensors or is
   opaque, or a unit between the (earliest) def and the segment start writes a tensor some def
   reads (the replica would compute a different value than the original definition did). *)
let plan_replicas (units : funit array) ~seg_start (ext_ids : Low_level.scope_id list) :
    funit list option =
  let module SId = struct
    let equal = Low_level.equal_scope_id
  end in
  let rec closure needed_ids collected =
    match needed_ids with
    | [] -> collected
    | id :: rest ->
        let defs = def_units_of units ~limit:seg_start id in
        if List.is_empty defs then raise Unfissionable;
        let fresh = List.filter defs ~f:(fun d -> not (List.mem collected d ~equal:phys_equal)) in
        let more_ids =
          List.concat_map fresh ~f:(fun d ->
              match d.f_sum with None -> [] | Some s -> s.s_scope_reads)
          |> List.filter ~f:(fun id' ->
                 (not (List.mem rest id' ~equal:SId.equal))
                 && not
                      (List.exists collected ~f:(fun d ->
                           match d.f_sum with
                           | None -> false
                           | Some s -> List.mem s.s_scope_declares id' ~equal:SId.equal)))
        in
        closure (rest @ more_ids) (collected @ fresh)
  in
  match closure ext_ids [] with
  | [] -> Some []
  | defs ->
      let defs = List.sort defs ~compare:(fun a b -> Int.compare a.f_index b.f_index) in
      let valid =
        List.for_all defs ~f:(fun d ->
            match d.f_sum with None -> false | Some s -> Set.is_empty s.s_writes)
        &&
        let def_reads =
          List.fold defs
            ~init:(Set.empty (module Tn))
            ~f:(fun acc d ->
              match d.f_sum with None -> acc | Some s -> Set.union acc s.s_reads)
        in
        let earliest = (List.hd_exn defs).f_index in
        Array.for_all units ~f:(fun u ->
            u.f_index <= earliest || u.f_index >= seg_start
            ||
            match u.f_sum with
            | None -> false
            | Some s -> Set.is_empty (Set.inter s.s_writes def_reads))
      in
      if valid then Some defs else None

(* External scope ids a segment references: read or written by its units but not declared by an
   earlier unit of the same segment (statement order is preserved, so any in-segment declaration
   precedes in-segment uses of well-formed code). *)
let seg_external_ids seg =
  let declared =
    List.concat_map seg.g_units ~f:(fun u ->
        match u.f_sum with None -> [] | Some s -> s.s_scope_declares)
  in
  List.concat_map seg.g_units ~f:(fun u ->
      match u.f_sum with None -> [] | Some s -> s.s_scope_reads @ s.s_scope_writes)
  |> List.filter ~f:(fun id -> not (List.mem declared id ~equal:Low_level.equal_scope_id))
  |> List.dedup_and_sort ~compare:Low_level.compare_scope_id

(* Resolve scope-local crossings: per segment, either a replica plan or a forced merge of the
   def..use segment range (the merged segment runs serially — exactly today's behavior for that
   region). Restarts after each merge; terminates because the segment count strictly decreases. *)
let rec resolve_scope_crossings (units : funit array) (segs : segment list) :
    segment list * funit list list =
  let plans =
    List.map segs ~f:(fun seg ->
        let ext = seg_external_ids seg in
        if List.is_empty ext then `Replicas []
        else
          let seg_start = (List.hd_exn seg.g_units).f_index in
          match plan_replicas units ~seg_start ext with
          | Some defs -> `Replicas defs
          | None -> `Merge_back seg_start)
  in
  match
    List.findi plans ~f:(fun _ -> function `Merge_back _ -> true | `Replicas _ -> false)
  with
  | None ->
      ( segs,
        List.map plans ~f:(function `Replicas defs -> defs | `Merge_back _ -> assert false) )
  | Some (j, _) ->
      (* Merge from the segment holding the earliest def through segment [j]. We do not know the
         def segment without re-running the failed plan; merging [j] with its predecessor is the
         minimal step that makes progress and re-checks. *)
      if j = 0 then raise Unfissionable
      else
        let before = List.take segs (j - 1) in
        let merged =
          match List.drop segs (j - 1) with
          | a :: b :: rest -> merge_segs ~kind:`Solo a b :: rest
          | _ -> assert false
        in
        resolve_scope_crossings units (before @ merged)

(* Per segment (units + replicas): the (reads, writes) tensor-node footprint. *)
let seg_footprints (segs_with_replicas : (segment * funit list) list) :
    (Set.M(Tn).t * Set.M(Tn).t) list =
  List.map segs_with_replicas ~f:(fun (seg, replicas) ->
      List.fold (replicas @ seg.g_units)
        ~init:(Set.empty (module Tn), Set.empty (module Tn))
        ~f:(fun (r, w) u ->
          match u.f_sum with
          | None -> (r, w)
          | Some s -> (Set.union r s.s_reads, Set.union w s.s_writes)))

let crosses_segments segs_with_replicas tn =
  List.count (seg_footprints segs_with_replicas) ~f:(fun (r, w) ->
      Set.mem r tn || Set.mem w tn)
  >= 2

(* Promote [Local]-placed scratch whose accesses (including replicas') span segments: kernel-local
   arrays do not survive a launch boundary. Runs before per-segment schedules are computed, so the
   annotator sees the promoted (materialized) status consistently — the converse order would let a
   segment annotate without covering the node's writes. Returns the undo list: coalescing may
   later remove a crossing, and a promotion without a surviving crossing must be restored (it
   would otherwise leak an observable placement change out of an all-serial routine). *)
let promote_crossing plc (segs_with_replicas : (segment * funit list) list) :
    (Tn.t * (Tn.memory_mode * int) option) list =
  let footprints = seg_footprints segs_with_replicas in
  let written =
    List.fold footprints ~init:(Set.empty (module Tn)) ~f:(fun acc (_, w) -> Set.union acc w)
  in
  let touched = List.map footprints ~f:(fun (r, w) -> Set.union r w) in
  Set.fold written ~init:[] ~f:(fun undo tn ->
      if
        (not (Tn.Placements.is_materialized_peek plc tn))
        && List.count touched ~f:(fun t -> Set.mem t tn) >= 2
      then (
        let prior = Tn.Placements.raw_entry plc tn in
        Tn.Placements.promote_local_to_device plc tn 177;
        (tn, prior) :: undo)
      else undo)

(* All tensor nodes a segment's code references — including scope ids' backing tnodes, so the
   filtered traced store retains every entry codegen consults — and whether it reads the merge
   buffer. *)
let code_footprint (llc : Low_level.t) : Set.M(Tn).t * bool =
  let open Low_level in
  let tns = ref (Set.empty (module Tn)) and merge = ref false in
  let add tn = tns := Set.add !tns tn in
  let rec code llc =
    match llc with
    | Noop | Comment _ | Staged_compilation _ | Workgroup_barrier -> ()
    | Tile_mma { d = d_tn, _; a = a_tn, _; b = b_tn, _; _ } ->
        add d_tn;
        add a_tn;
        add b_tn
    | Declare_local { id; _ } -> add id.tn
    | Seq (a, b) ->
        code a;
        code b
    | For_loop { body; _ } -> code body
    | Zero_out tn -> add tn
    | Set { tn; llsc; _ } ->
        add tn;
        scalar llsc
    | Set_from_vec { tn; arg = a, _; _ } ->
        add tn;
        scalar a
    | Set_local (id, llsc) ->
        add id.tn;
        scalar llsc
    | If { cond = c, _; body } ->
        scalar c;
        code body
  and scalar (llsc : scalar_t) =
    match llsc with
    | Local_scope { id; body; _ } ->
        add id.tn;
        code body
    | Get_local id -> add id.tn
    | Get (tn, _) -> add tn
    | Get_dynamic { tn; dyn_value = v, _; _ } ->
        add tn;
        scalar v
    | Get_merge_buffer (tn, _) ->
        add tn;
        merge := true
    | Constant _ | Constant_bits _ | Embed_index _ -> ()
    | Ternop (_, (a, _), (b, _), (c, _)) ->
        scalar a;
        scalar b;
        scalar c
    | Binop (_, (a, _), (b, _)) ->
        scalar a;
        scalar b
    | Unop (_, (a, _)) -> scalar a
  in
  code llc;
  (!tns, !merge)

let segment_optimized (full : Low_level.optimized) (llc : Low_level.t) : Low_level.optimized =
  let tns, reads_merge = code_footprint llc in
  {
    Low_level.traced_store =
      Hashtbl.filteri full.Low_level.traced_store ~f:(fun ~key ~data:_ -> Set.mem tns key);
    optimize_ctx = full.Low_level.optimize_ctx;
    llc;
    merge_node = (if reads_merge then full.Low_level.merge_node else None);
    workgroup_shared = Set.filter full.Low_level.workgroup_shared ~f:(Set.mem tns);
  }

(* Expand-and-annotate schedule for a segment of materialized whole-node [Zero_out]s (GPU): the
   expanded nests get the same geometry policy as {!default_gpu}'s chains. Below [min_parallel]
   (largest node) the zeros stay whole-node — a serial kernel renders them as [memset]. *)
let zero_expansion ?block_size ?min_parallel ~(limits : Backend_intf.hardware_limits)
    (tns : Tn.t list) : schedule =
  let block_size =
    Option.value block_size
      ~default:
        (Int.of_string @@ Utils.get_global_arg ~arg_name:"gpu_schedule_block_size" ~default:"256")
  in
  let block_size =
    Option.value_map limits.Backend_intf.max_threads_per_workgroup ~default:block_size
      ~f:(min block_size)
  in
  let min_parallel =
    Option.value min_parallel
      ~default:
        (Int.of_string @@ Utils.get_global_arg ~arg_name:"gpu_schedule_min_parallel" ~default:"1024")
  in
  let dims tn = Lazy.force tn.Tn.dims in
  let numel tn = Array.fold (dims tn) ~init:1 ~f:( * ) in
  if
    List.exists tns ~f:(fun tn -> Array.is_empty (dims tn))
    (* A rank-0 expansion is a bare write: uncoverable in a multi-threaded kernel. *)
    || List.fold tns ~init:0 ~f:(fun m tn -> max m (numel tn)) < min_parallel
  then []
  else
    List.concat_map tns ~f:(fun tn ->
        let op, syms = expand_zero ~tn in
        let ds = dims tn in
        let annots =
          match syms with
          | [] -> assert false
          | [ s0 ] ->
              let n0 = ds.(0) in
              let sp, _, _ =
                split ~axis:s0 ~factor:(min block_size n0) ~outer:Low_level.Grid
                  ~inner:Low_level.Workgroup
              in
              [ sp ]
          | s0 :: s1 :: _ ->
              let n1 = ds.(1) in
              if n1 <= block_size then
                [
                  Retype { axis = s0; ty = Low_level.Grid };
                  Retype { axis = s1; ty = Low_level.Workgroup };
                ]
              else
                let sp, _, _ =
                  split ~axis:s1 ~factor:block_size ~outer:Low_level.Serial
                    ~inner:Low_level.Workgroup
                in
                [ Retype { axis = s0; ty = Low_level.Grid }; sp ]
        in
        op :: annots)

let seg_llc replicas seg =
  Low_level.unflat_lines (List.concat_map (replicas @ seg.g_units) ~f:(fun u -> u.f_stmts))

let fission_scheduled ~(preset : Low_level.optimized -> schedule)
    ~(zero_sched : Tn.t list -> schedule) ~static_indices (opt : Low_level.optimized) :
    ([ `Normal | `Zeros | `Solo ] * Low_level.optimized * schedule * Low_level.optimized) list =
  let plc = opt.Low_level.optimize_ctx.placements in
  let fallback () =
    let sched = preset opt in
    [ (`Normal, opt, sched, apply ~static_indices sched opt) ]
  in
  let units = collect_units plc opt (Low_level.flat_lines [ opt.Low_level.llc ]) in
  let segs = group_units opt units in
  if List.length segs <= 1 then fallback ()
  else
    match resolve_scope_crossings (Array.of_list units) segs with
    | exception Unfissionable -> fallback ()
    | segs, replicas when List.length segs <= 1 -> ignore replicas; fallback ()
    | segs, replicas ->
        let segs_with_replicas = List.zip_exn segs replicas in
        let promoted = promote_crossing plc segs_with_replicas in
        let undo_promotions which =
          List.iter which ~f:(fun (tn, prior) -> Tn.Placements.unsafe_restore plc tn prior)
        in
        let scheduled =
          List.map segs_with_replicas ~f:(fun (seg, replicas) ->
              let sched =
                match seg.g_kind with
                | `Solo -> []
                | `Zeros ->
                    zero_sched
                      (List.filter_map seg.g_units ~f:(fun u ->
                           Option.bind u.f_sum ~f:(fun s -> s.s_top_zero)))
                | `Normal -> preset (segment_optimized opt (seg_llc replicas seg))
              in
              (seg, replicas, sched))
        in
        (* Coalesce adjacent unannotated segments: consecutive serial kernels gain nothing from a
           launch boundary. Merged segments are rebuilt from original units and their replicas
           recomputed for the new boundary (the def..start gap only shrinks under merging, so
           feasibility is preserved). *)
        let coalesced =
          List.fold scheduled ~init:[] ~f:(fun acc (seg, replicas, sched) ->
              match acc with
              | (pseg, _, []) :: rest when List.is_empty sched ->
                  let merged = merge_segs ~kind:`Solo pseg seg in
                  let replicas =
                    match
                      plan_replicas (Array.of_list units)
                        ~seg_start:(List.hd_exn merged.g_units).f_index
                        (seg_external_ids merged)
                    with
                    | Some defs -> defs
                    | None -> assert false (* Merging only shrinks the validity range. *)
                  in
                  (merged, replicas, []) :: rest
              | _ -> (seg, replicas, sched) :: acc)
          |> List.rev
        in
        if List.length coalesced <= 1 then (
          (* Everything merged back: single kernel, exactly as before fission — including
             placements, so undo every promotion (an all-serial small routine must not leak
             observable placement changes; zero2hero's virtual-neuron printouts pinned this). *)
          undo_promotions promoted;
          fallback ())
        else (
          (* Coalescing may have absorbed a crossing: promotions without a surviving crossing
             are restored. Sound in this direction — the segments' schedules were computed under
             the stricter materialized view (see [promote_crossing]). *)
          let final_swr = List.map coalesced ~f:(fun (seg, replicas, _) -> (seg, replicas)) in
          undo_promotions
            (List.filter promoted ~f:(fun (tn, _) -> not (crosses_segments final_swr tn)));
          List.map coalesced ~f:(fun (seg, replicas, sched) ->
              let pre = segment_optimized opt (seg_llc replicas seg) in
              (seg.g_kind, pre, sched, apply ~static_indices sched pre)))

let fission_default ~preset ~zero_sched ~static_indices (opt : Low_level.optimized) :
    Low_level.optimized list =
  List.map (fission_scheduled ~preset ~zero_sched ~static_indices opt)
    ~f:(fun (_kind, _pre, _sched, post) -> post)

(** {2 Wiring: the implicit transform for GPU and CPU backends} *)

let automatic_gpu_schedule =
  lazy (Utils.get_global_flag ~default:true ~arg_name:"automatic_gpu_schedule")

let automatic_cpu_schedule =
  lazy (Utils.get_global_flag ~default:true ~arg_name:"automatic_cpu_schedule")

let backend_is_gpu name =
  String.is_substring name ~substring:"cuda" || String.is_substring name ~substring:"metal"

let backend_is_cpu name = String.is_substring name ~substring:"cc"

let schedule_fission = lazy (Utils.get_global_flag ~default:true ~arg_name:"schedule_fission")

let maybe_default_schedule ~backend_name ?(limits = Backend_intf.no_hardware_limits)
    ~static_indices (opt : Low_level.optimized) : Low_level.optimized =
  (* Runtime kernel logging is line-interleaved under parallel execution; keep logged runs
     serial so the logs stay deterministic and readable. *)
  if Utils.debug_log_from_routines () then opt
  else if backend_is_gpu backend_name && Lazy.force automatic_gpu_schedule then
    apply ~static_indices (default_gpu ~limits opt) opt
  else if backend_is_cpu backend_name && Lazy.force automatic_cpu_schedule then
    apply ~static_indices (default_cpu opt) opt
  else opt

let maybe_default_schedules ~backend_name ?(limits = Backend_intf.no_hardware_limits)
    ~static_indices (opt : Low_level.optimized) : Low_level.optimized list =
  if Utils.debug_log_from_routines () then [ opt ]
  else
    let gpu = backend_is_gpu backend_name && Lazy.force automatic_gpu_schedule in
    let cpu = backend_is_cpu backend_name && Lazy.force automatic_cpu_schedule in
    if not (gpu || cpu) then [ opt ]
    else if not (Lazy.force schedule_fission) then
      [ maybe_default_schedule ~backend_name ~limits ~static_indices opt ]
    else
      let preset o = if gpu then default_gpu ~limits o else default_cpu o in
      (* CPU zero segments stay whole-node: a serial kernel renders them as [memset], which is
         hard to beat below many-megabyte sizes; parallel zero expansion on CPU is a follow-up. *)
      let zero_sched tns = if gpu then zero_expansion ~limits tns else [] in
      fission_default ~preset ~zero_sched ~static_indices opt

let check_hardware_limits ~name ~(limits : Backend_intf.hardware_limits)
    (opt : Low_level.optimized) : unit =
  Option.iter limits.max_threads_per_workgroup ~f:(fun max_threads ->
      let block = (Low_level.launch_dims opt.llc).block in
      let block_product = Array.fold block ~init:1 ~f:( * ) in
      if block_product > max_threads then
        raise
        @@ Utils.User_error
             [%string
               "Schedule: kernel %{name} requests a workgroup of %{block_product#Int} threads, \
                exceeding the device limit of %{max_threads#Int} threads per workgroup"]);
  Option.iter limits.max_workgroup_memory_bytes ~f:(fun max_bytes ->
      let shared_bytes =
        Set.fold opt.workgroup_shared ~init:0 ~f:(fun acc tn ->
            acc + Lazy.force tn.Tn.size_in_bytes)
      in
      if shared_bytes > max_bytes then
        raise
        @@ Utils.User_error
             [%string
               "Schedule: kernel %{name} stages %{shared_bytes#Int} bytes of workgroup-shared \
                tiles, exceeding the device limit of %{max_bytes#Int} bytes"])
