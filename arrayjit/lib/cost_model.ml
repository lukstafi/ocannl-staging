open Base
module Idx = Indexing
module Tn = Tnode

(* Analytic cost model, extraction half (gh-ocannl-491 task 1). See cost_model.mli for the
   approximation contract — every count here is an upper bound (exactness tracked per node) except
   under [opaque], the flagged under-counting escape hatch. *)

type node_footprint = {
  fp_read_bytes : int;
  fp_write_bytes : int;
  fp_rmw_bytes : int;
  fp_approx : bool;
}
[@@deriving sexp_of]

type summary = {
  per_node : (Tn.t * node_footprint) list;
  read_bytes : int;
  write_bytes : int;
  flops : int;
  flops_approx : bool;
  opaque : bool;
}
[@@deriving sexp_of]

(* Distinct cells one access can touch, with exactness. Interpretable maps: image cardinality =
   loop-box size / fiber size ({!Affine.fiber_cardinality}); an [`At_least] fiber (non-injective
   map) makes that an upper bound on the image. Uninterpretable components fall back to the whole
   node. Guarded accesses are counted guards-taken. All biases over-count. *)
let access_cells (a : Tn.t Affine.access) : int * bool =
  let node_cells = Tn.num_elems a.a_tn in
  let uninterpretable =
    a.a_dynamic
    || Array.exists a.a_map ~f:(function Idx.Sub_axis | Idx.Concat _ -> true | _ -> false)
  in
  if a.a_whole then (node_cells, a.a_guarded)
  else if uninterpretable then (node_cells, true)
  else
    let domain = List.map a.a_loops ~f:(fun (s, (lo, hi)) -> (s, hi - lo + 1)) in
    let box = List.fold domain ~init:1 ~f:(fun acc (_, w) -> acc * w) in
    let image, exact_image =
      match Affine.fiber_cardinality ~domain a.a_map with
      | `Exact f -> (box / max 1 f, true)
      | `At_least f -> (box / max 1 f, false)
    in
    if a.a_vec_last then
      (* Each map instance is the base of a run along the minor axis: runs may overlap for strided
         bases, so the product is an upper bound. *)
      (min node_cells (image * max 1 a.a_vec_len), true)
    else (min node_cells image, (not exact_image) || a.a_guarded)

let footprints (accesses : Tn.t Affine.access list) : (Tn.t * node_footprint) list =
  (* Per node and direction: sum of per-access cell counts (a union upper bound, capped by the
     node's size); exact only for a single exact access in the direction. *)
  let tbl = Hashtbl.create (module Tn) in
  let order = ref [] in
  List.iter accesses ~f:(fun a ->
      let cells, approx = access_cells a in
      let cur =
        Hashtbl.find_or_add tbl a.a_tn ~default:(fun () ->
            order := a.a_tn :: !order;
            (0, 0, 0, 0, 0, false))
      in
      let reads, writes, rmws, nreads, nwrites, apx = cur in
      let next =
        if a.a_write then
          ( reads,
            writes + cells,
            (rmws + if a.a_rmw then cells else 0),
            nreads,
            nwrites + 1,
            apx || approx )
        else (reads + cells, writes, rmws, nreads + 1, nwrites, apx || approx)
      in
      Hashtbl.set tbl ~key:a.a_tn ~data:next);
  List.rev_map !order ~f:(fun tn ->
      let reads, writes, rmws, nreads, nwrites, apx = Hashtbl.find_exn tbl tn in
      let width = Ops.prec_in_bytes (Lazy.force tn.Tn.storage_prec) in
      let node_bytes = Tn.num_elems tn * width in
      let cap n = min node_bytes (n * width) in
      ( tn,
        {
          fp_read_bytes = cap reads;
          fp_write_bytes = cap writes;
          fp_rmw_bytes = cap rmws;
          fp_approx = apx || nreads > 1 || nwrites > 1;
        } ))

let analyze (code : Low_level.t) : summary =
  let flops_approx = ref false and opaque = ref false in
  (* [scale] is the product of enclosing loop extents, [env] their (symbol, extent) bindings —
     needed by [Tile_mma], whose 2*m*n*k multiply-adds are cooperative across its [lane] loop, not
     repeated per lane. *)
  let rec go ~scale ~env (c : Low_level.t) : int =
    match c with
    | Low_level.Noop | Comment _ | Zero_out _ | Declare_local _ | Workgroup_barrier -> 0
    | Staged_compilation _ ->
        opaque := true;
        0
    | Seq (c1, c2) -> go ~scale ~env c1 + go ~scale ~env c2
    | For_loop { index; from_; to_; body; _ } ->
        let extent = max 0 (to_ - from_ + 1) in
        go ~scale:(scale * extent) ~env:((index, extent) :: env) body
    | Set { llsc; _ } -> scale * sc_flops llsc
    | Set_dynamic { dyn_value = dv, _; llsc; _ } -> scale * (sc_flops dv + sc_flops llsc)
    | Set_from_vec { length; arg = a, _; _ } -> scale * (length + sc_flops a)
    | Set_local (_, llsc) -> scale * sc_flops llsc
    | If { cond = cnd, _; body } ->
        (* Guards-taken: the body is charged as if the guard always passes. *)
        flops_approx := true;
        (scale * sc_flops cnd) + go ~scale ~env body
    | Tile_mma { m; n; k; lane; _ } ->
        let lane_extent =
          List.Assoc.find env lane ~equal:Idx.equal_symbol |> Option.value ~default:1
        in
        scale / max 1 lane_extent * (2 * m * n * k)
  and sc_flops (sc : Low_level.scalar_t) : int =
    match sc with
    | Low_level.Local_scope { body; _ } -> go ~scale:1 ~env:[] body
    | Get_local _ | Get _ | Constant _ | Constant_bits _ | Embed_index _ -> 0
    | Get_merge_buffer _ ->
        (* Merge-buffer traffic is not represented by [affine_accesses] either: flag the
           under-count. *)
        opaque := true;
        0
    | Get_dynamic { dyn_value = dv, _; _ } -> sc_flops dv
    | Ternop ((Ops.FMA | Ops.Mul3), a1, a2, a3) ->
        (* Two arithmetic operations each — matching [peak_flops]' FMA-counted-as-two convention, so
           an FMA-form kernel scores the same compute leg as its mul+add form. *)
        2 + arg a1 + arg a2 + arg a3
    | Ternop (Ops.Where, a1, a2, a3) -> 1 + arg a1 + arg a2 + arg a3
    | Binop ((Ops.Arg1 | Ops.Arg2), a1, a2) -> arg a1 + arg a2
    | Binop (_, a1, a2) -> 1 + arg a1 + arg a2
    | Unop (Ops.Identity, a1) -> arg a1
    | Unop (_, a1) -> 1 + arg a1
  and arg (sc, _prec) = sc_flops sc in
  let flops = go ~scale:1 ~env:[] code in
  let per_node = footprints (Low_level.affine_accesses code) in
  {
    per_node;
    read_bytes = List.fold per_node ~init:0 ~f:(fun acc (_, fp) -> acc + fp.fp_read_bytes);
    write_bytes = List.fold per_node ~init:0 ~f:(fun acc (_, fp) -> acc + fp.fp_write_bytes);
    flops;
    flops_approx = !flops_approx;
    opaque = !opaque;
  }

let total_bytes s = s.read_bytes + s.write_bytes
let arithmetic_intensity s = Float.of_int s.flops /. Float.of_int (max 1 (total_bytes s))

let roofline_seconds ?peak_flops ?peak_memory_bandwidth ~flops ~bytes () : float option =
  let legs =
    List.filter_opt
      [
        Option.map peak_flops ~f:(fun p -> Float.of_int flops /. p);
        Option.map peak_memory_bandwidth ~f:(fun p -> Float.of_int bytes /. p);
      ]
  in
  match legs with [] -> None | l -> Some (List.reduce_exn l ~f:Float.max)
