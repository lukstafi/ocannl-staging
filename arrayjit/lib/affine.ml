open Base
module Idx = Indexing

(** Affine legality queries over access index vectors (gh-ocannl-494, manifesto §6 waypoints 1-2).

    Loop nests are generated from einsum projections, so every access in optimized [Low_level.t]
    code carries its affine index map ([Indexing.axis_index array]) natively. This module is the
    query engine over those maps: the procedural schedule-legality analyses (the default annotator's
    per-nest hazard agreement rule, the cross-nest alignment rule, [C_syntax.parallel_grid_safe]'s
    shared rule, the covering checks) are conservative special cases of the decision procedures
    here, and are being re-derived as queries — one implementation of "is this access pattern
    race-free" instead of three.

    Scope: linear-integer reasoning over box domains — per-axis linear Diophantine equations with
    gcd/interval infeasibility and forced-equality derivation via the mixed-radix injectivity
    criterion (the same criterion as {!Indexing.affine_injective}). No full Presburger machinery:
    every query form needed so far is decided (or conservatively declined) at this level. Any
    component the engine cannot interpret ([Sub_axis], [Concat], dynamic indices) contributes no
    information, which errs on the side of declining — soundness is preserved by construction. *)

(** {2 The pair-conflict query}

    Can two accesses touch a common cell from different "threads"? Thread identity is a tuple of
    parallel loop indices; the query works over two copies of the iteration space (the [`Left] and
    [`Right] access's enclosing loops), sharing the symbols that are equal across concurrently
    executing threads (static indices, loops enclosing the parallel region — a join separates their
    iterations). *)

type var = Shared of Idx.symbol | Left_v of Idx.symbol | Right_v of Idx.symbol
[@@deriving compare, sexp_of]

module Var = struct
  module T = struct
    type t = var [@@deriving compare, sexp_of]
  end

  include T
  include Comparator.Make (T)
end

type verdict =
  | Disjoint  (** The two accesses never touch a common cell at all. *)
  | Same_thread
      (** Common cells occur only when every paired parallel symbol is equal — conflicts are
          confined to a single thread, where program order applies. *)
  | Cross_thread of string
      (** A cross-thread conflict is possible, or the engine cannot rule one out; the payload is the
          witness/explanation (the axis or symbol pair that failed). *)
[@@deriving sexp_of]

let equal_verdict a b =
  match (a, b) with
  | Disjoint, Disjoint | Same_thread, Same_thread | Cross_thread _, Cross_thread _ -> true
  | _ -> false

let rec gcd a b = if b = 0 then abs a else gcd b (Int.rem a b)

(* Linear terms of one axis component on one side. [None] = uninterpretable component (no
   information; conservative). Width-1 symbols are substituted by their lower bound: they are
   constants, and removing them unclutters the matched-pair forcing below. *)
let terms_of ~range ~dup ~(side : Idx.symbol -> var) (idx : Idx.axis_index) :
    ((int * var) list * int) option =
  let tag s = if dup s then side s else Shared s in
  let term (c, s) =
    match range s with
    | Some (lo, hi) when lo = hi -> Either.Second (c * lo)
    | _ -> Either.First (c, tag s)
  in
  match idx with
  | Idx.Fixed_idx c -> Some ([], c)
  | Idx.Iterator s -> (
      match term (1, s) with Either.First t -> Some ([ t ], 0) | Either.Second c -> Some ([], c))
  | Idx.Affine { symbols; offset } ->
      let ts, cs = List.partition_map (Idx.coalesce_affine_terms symbols) ~f:term in
      Some (ts, offset + List.fold cs ~init:0 ~f:( + ))
  | Idx.Sub_axis | Idx.Concat _ -> None

(* Combine left and right components into [Σ d·v = δ] with per-var coalescing (a [Shared] symbol
   appearing with equal coefficients on both sides cancels). *)
let equation_of ~range ~dup_left ~dup_right (l : Idx.axis_index) (r : Idx.axis_index) :
    ((int * var) list * int) option =
  match
    ( terms_of ~range ~dup:dup_left ~side:(fun s -> Left_v s) l,
      terms_of ~range ~dup:dup_right ~side:(fun s -> Right_v s) r )
  with
  | Some (lt, lc), Some (rt, rc) ->
      let combined =
        List.fold
          (lt @ List.map rt ~f:(fun (c, v) -> (-c, v)))
          ~init:(Map.empty (module Var))
          ~f:(fun acc (c, v) -> Map.update acc v ~f:(function None -> c | Some c0 -> c0 + c))
        |> Map.filter ~f:(fun c -> c <> 0)
      in
      Some (Map.to_alist combined |> List.map ~f:(fun (v, c) -> (c, v)), rc - lc)
  | _ -> None

let range_of_var ~range = function Shared s | Left_v s | Right_v s -> range s

(* No solution within bounds: the gcd of the coefficients does not divide the constant, or the
   interval of the left-hand side (when all variables are bounded) misses it. *)
let infeasible ~range (terms, rhs) =
  match terms with
  | [] -> rhs <> 0
  | _ -> (
      let g = List.fold terms ~init:0 ~f:(fun g (c, _) -> gcd g c) in
      rhs % g <> 0
      ||
      let bounds =
        List.fold terms
          ~init:(Some (0, 0))
          ~f:(fun acc (c, v) ->
            match (acc, range_of_var ~range v) with
            | Some (lo, hi), Some (vlo, vhi) ->
                Some (lo + min (c * vlo) (c * vhi), hi + max (c * vlo) (c * vhi))
            | _ -> None)
      in
      match bounds with Some (lo, hi) -> rhs < lo || rhs > hi | None -> false)

(* Forced equalities: an equation [Σ_k c_k·(x_k − y_k) = 0] with [x_k] all-[Left_v], [y_k]
   all-[Right_v], matched pairwise by coefficient, forces [x_k = y_k] for every k when the linear
   form [Σ c_k·z_k] is injective on the (per-pair combined) bounding box — the mixed-radix criterion
   of {!Indexing.affine_injective}: sorted by ascending [abs c], every term satisfies [abs c_k >= 1
   + Σ_{i<k} abs c_i·(w_i − 1)]. Equal values of an injective form imply equal arguments, and the
   forcing holds in every solution of the conjunction it is part of. *)
let forced_pairs ~range (terms, rhs) : (Idx.symbol * Idx.symbol) list =
  if rhs <> 0 then []
  else
    let ls, rest =
      List.partition_tf terms ~f:(fun (_, v) -> match v with Left_v _ -> true | _ -> false)
    in
    let rs, shared =
      List.partition_tf rest ~f:(fun (_, v) -> match v with Right_v _ -> true | _ -> false)
    in
    if (not (List.is_empty shared)) || List.length ls <> List.length rs || List.is_empty ls then []
    else
      let cmp (c1, _) (c2, _) = Int.compare c1 c2 in
      let ls = List.sort ls ~compare:cmp in
      let rs = List.sort rs ~compare:(fun (c1, _) (c2, _) -> Int.compare (-c1) (-c2)) in
      let paired =
        List.zip_exn ls rs
        |> List.map ~f:(fun ((cl, vl), (cr, vr)) ->
            let sym = function Left_v s | Right_v s | Shared s -> s in
            if cl <> -cr then None
            else
              match (range_of_var ~range vl, range_of_var ~range vr) with
              | Some (llo, lhi), Some (rlo, rhi) ->
                  let w = max lhi rhi - min llo rlo + 1 in
                  Some (abs cl, w, (sym vl, sym vr))
              | _ -> None)
      in
      match Option.all paired with
      | None -> []
      | Some triples ->
          let sorted =
            List.sort triples ~compare:(fun (c1, _, _) (c2, _, _) -> Int.compare c1 c2)
          in
          let rec radix acc = function
            | [] -> true
            | (c, w, _) :: tl -> c >= 1 + acc && radix (acc + (c * (w - 1))) tl
          in
          if radix 0 sorted then List.map triples ~f:(fun (_, _, p) -> p) else []

let axis_index_to_string (idx : Idx.axis_index) =
  Sexp.to_string_hum ([%sexp_of: Idx.axis_index] idx)

(** [pair_conflict ~range ~dup_left ~dup_right ~pairs ~left ~right]: verdict on whether the accesses
    with index vectors [left] and [right] (over the same tensor node; rank-padded with
    [Fixed_idx 0]) can touch a common cell from different threads. [range s] gives the inclusive
    iteration bounds of loop symbol [s] ([None] for static/unknown symbols). [dup_left]/[dup_right]
    select the symbols iterated independently by each side (its enclosing loops within the analyzed
    parallel region); other symbols are shared — equal across concurrently executing threads.
    [pairs] is the thread identity: the parallel symbols of the left copy paired with those of the
    right copy (for same-nest analyses, pairs of the form [(p, p)]).

    Sound and conservative: [Disjoint] and [Same_thread] are proven; everything else is
    [Cross_thread]. *)
let pair_conflict ~range ~dup_left ~dup_right ~(pairs : (Idx.symbol * Idx.symbol) list)
    ~(left : Idx.axis_index array) ~(right : Idx.axis_index array) : verdict =
  let rank = max (Array.length left) (Array.length right) in
  let comp idcs p = if p < Array.length idcs then idcs.(p) else Idx.Fixed_idx 0 in
  let eqs =
    List.init rank ~f:(fun p ->
        (p, equation_of ~range ~dup_left ~dup_right (comp left p) (comp right p)))
  in
  if List.exists eqs ~f:(fun (_, eq) -> Option.value_map eq ~default:false ~f:(infeasible ~range))
  then Disjoint
  else
    let forced =
      List.concat_map eqs ~f:(fun (_, eq) ->
          Option.value_map eq ~default:[] ~f:(forced_pairs ~range))
    in
    let pair_forced (p, p') =
      (* A width-1 parallel symbol has a single thread coordinate, so equality across threads holds
         by definition — necessary because {!terms_of} substitutes width-1 symbols away before the
         equation-level forcing can see them. *)
      (match (range p, range p') with
        | Some (lo, hi), Some (lo', hi') -> lo = hi && lo' = hi' && lo = lo'
        | _ -> false)
      || List.exists forced ~f:(fun (a, b) -> Idx.equal_symbol a p && Idx.equal_symbol b p')
    in
    if (not (List.is_empty pairs)) && List.for_all pairs ~f:pair_forced then Same_thread
    else
      let witness =
        match List.find pairs ~f:(Fn.non pair_forced) with
        | Some (p, p') when Idx.equal_symbol p p' ->
            Printf.sprintf "parallel symbol %s is not forced equal across threads"
              (Idx.symbol_ident p)
        | Some (p, p') ->
            Printf.sprintf "paired parallel symbols %s ~ %s are not forced equal across threads"
              (Idx.symbol_ident p) (Idx.symbol_ident p')
        | None -> "no parallel symbols to confine the conflict"
      in
      Cross_thread
        (Printf.sprintf "%s (left %s, right %s)" witness
           (String.concat_array ~sep:"," (Array.map left ~f:axis_index_to_string))
           (String.concat_array ~sep:"," (Array.map right ~f:axis_index_to_string)))

(** {2 The covering query} *)

(** [covers_box ~range ~dims idcs]: whether the index vector [idcs], as its symbols range over their
    (loop) bounds, enumerates every cell of the [dims] box exactly once — a bijection onto the box.
    This is the write-dominance building block: a covering unguarded write rewrites the whole array.
    Requirements: each symbol used at most once across the vector; per axis, a zero-based
    full-extent iterator, a mixed-radix affine combination of zero-based symbols whose radix chain
    exactly composes to the axis dimension, or [Fixed_idx 0] on a unit axis. Generalizes (and is
    checked against) the procedural per-axis rule of [C_syntax.first_access_standalone_covering]. *)
let covers_box ~range ~(dims : int array) (idcs : Idx.axis_index array) : bool =
  Array.length idcs = Array.length dims
  &&
  let used = ref [] in
  let fresh s =
    if List.mem !used s ~equal:Idx.equal_symbol then false
    else (
      used := s :: !used;
      true)
  in
  let extent_of s = match range s with Some (0, hi) -> Some (hi + 1) | _ -> None in
  Array.for_alli idcs ~f:(fun a idx ->
      let dim = dims.(a) in
      match idx with
      | Idx.Fixed_idx 0 -> dim = 1
      | Idx.Iterator s -> ( fresh s && match extent_of s with Some e -> e = dim | None -> false)
      | Idx.Affine { symbols; offset = 0 } ->
          let sorted =
            List.sort (Idx.coalesce_affine_terms symbols) ~compare:(fun (c1, _) (c2, _) ->
                Int.compare c1 c2)
          in
          let rec radix r = function
            | [] -> r = dim
            | (c, s) :: tl -> (
                c = r && fresh s
                && match extent_of s with Some e -> radix (r * e) tl | None -> false)
          in
          radix 1 sorted
      | Idx.Fixed_idx _ | Idx.Affine _ | Idx.Sub_axis | Idx.Concat _ -> false)

(** {2 Counting}

    [fiber_cardinality ~domain idcs]: how many points of the loop box [domain] (symbol, width pairs)
    map to one given cell in the image of the access map [idcs] — the per-cell visit count of a read
    access, and the recompute cost per read site of inlining a setter (the retired [Low_level] concrete tracer's
    reduction extent, [virtualize_max_inline_reduction]'s subject). Domain symbols absent from the
    map contribute the product of their widths; when the map is injective on its mentioned symbols
    ({!Indexing.affine_injective}) that product is the exact fiber size of every image cell (cells
    outside the image have zero), otherwise it is a lower bound. *)
let fiber_cardinality ~(domain : (Idx.symbol * int) list) (idcs : Idx.axis_index array) :
    [ `Exact of int | `At_least of int ] =
  let mentions s =
    Array.exists idcs ~f:(function
      | Idx.Iterator s' -> Idx.equal_symbol s s'
      | Idx.Affine { symbols; _ } ->
          (* Coalesced, so that zero or cancelling terms do not count as mentions —
             [Indexing.affine_injective] coalesces too, and a symbol it never sees must contribute
             its width to the fiber (Codex P2 on PR #181). *)
          List.exists (Idx.coalesce_affine_terms symbols) ~f:(fun (_, s') -> Idx.equal_symbol s s')
      | Idx.Concat syms -> List.exists syms ~f:(Idx.equal_symbol s)
      | Idx.Fixed_idx _ | Idx.Sub_axis -> false)
  in
  let base = List.fold domain ~init:1 ~f:(fun acc (s, w) -> if mentions s then acc else acc * w) in
  let symbol_range s =
    List.Assoc.find domain s ~equal:Idx.equal_symbol |> Option.value ~default:1
  in
  if Idx.affine_injective ~symbol_range idcs then `Exact base else `At_least base

(** Upper-bound companion of {!fiber_cardinality}: at most how many points of the loop box [domain]
    map to any single cell of the image of [idcs]. Domain symbols absent from the map contribute the
    product of their widths exactly; when the map is injective on its mentioned symbols that is the
    whole fiber and the bound is exact. Otherwise the mentioned symbols' contribution is bounded per
    component: the solutions of [Σ c_k·s_k = const] over a box are pinned once all symbols but one
    are fixed, so one component admits at most the product of its symbols' widths divided by the
    largest of them, times the widths of the mentioned symbols it does not constrain; the smallest
    component-wise bound is taken. Uninterpretable components ([Sub_axis], [Concat]) constrain
    nothing. *)
let fiber_cardinality_ub ~(domain : (Idx.symbol * int) list) (idcs : Idx.axis_index array) :
    [ `Exact of int | `At_most of int ] =
  let mentioned_in (idx : Idx.axis_index) : Idx.symbol list =
    match idx with
    | Idx.Iterator s -> [ s ]
    | Idx.Affine { symbols; _ } -> List.map (Idx.coalesce_affine_terms symbols) ~f:snd
    | Idx.Concat syms -> syms
    | Idx.Fixed_idx _ | Idx.Sub_axis -> []
  in
  let in_domain s = List.Assoc.mem domain s ~equal:Idx.equal_symbol in
  let width s = List.Assoc.find domain s ~equal:Idx.equal_symbol |> Option.value ~default:1 in
  let mentioned =
    Array.to_list idcs |> List.concat_map ~f:mentioned_in
    |> List.dedup_and_sort ~compare:Idx.compare_symbol
    |> List.filter ~f:in_domain
  in
  let base =
    List.fold domain ~init:1 ~f:(fun acc (s, w) ->
        if List.mem mentioned s ~equal:Idx.equal_symbol then acc else acc * w)
  in
  if Idx.affine_injective ~symbol_range:width idcs then `Exact base
  else
    let product syms = List.fold syms ~init:1 ~f:(fun acc s -> acc * width s) in
    let component_bound (idx : Idx.axis_index) : int option =
      match idx with
      | Idx.Fixed_idx _ | Idx.Sub_axis | Idx.Concat _ -> None
      | Idx.Iterator _ | Idx.Affine _ -> (
          let syms = List.filter (mentioned_in idx) ~f:in_domain in
          match syms with
          | [] -> None
          | _ ->
              let widths = List.map syms ~f:width in
              let wmax = List.fold widths ~init:1 ~f:max in
              let own = List.fold widths ~init:1 ~f:( * ) / wmax in
              let others =
                List.filter mentioned ~f:(fun s -> not (List.mem syms s ~equal:Idx.equal_symbol))
              in
              Some (own * product others))
    in
    let best =
      Array.fold idcs ~init:(product mentioned) ~f:(fun acc idx ->
          match component_bound idx with Some b -> min acc b | None -> acc)
    in
    `At_most (base * best)

(** {2 Projection-level predicates}

    Moved from [Indexing] (they are queries about the affine LHS map of a projection, this module's
    home turf): [is_surjective] decides whether every LHS position is written — used to elide
    zero-initialization before assignments; [is_injective] whether no LHS position is written twice
    — used with [is_surjective] to elide initialization entirely. *)

let is_surjective (proj : Idx.projections) =
  (* For surjectivity, we check if all target (LHS) positions will be written to. This is used to
     determine if we need to zero-initialize before assignment. *)

  (* Check if there are any fixed indices (except Fixed_idx 0 when dim is 1) *)
  let has_non_trivial_fixed =
    Array.exists2_exn proj.Idx.project_lhs proj.Idx.lhs_dims ~f:(fun idx dim ->
        match idx with
        | Idx.Fixed_idx i -> not (i = 0 && dim <= 1) (* Fixed_idx 0 is OK only when dim is 0 or 1 *)
        | _ -> false)
  in
  if has_non_trivial_fixed then false
  else
    (* Collect symbols used in LHS *)
    let lhs_symbols, has_affine, has_sub_axis, num_concat_axes =
      Array.fold proj.Idx.project_lhs ~init:([], false, false, 0)
        ~f:(fun (syms, has_aff, has_sub, num_concat) idx ->
          match idx with
          | Idx.Iterator s -> (s :: syms, has_aff, has_sub, num_concat)
          | Idx.Fixed_idx _ -> (syms, has_aff, has_sub, num_concat)
          | Idx.Affine { symbols; _ } ->
              let coeff1_syms =
                List.filter_map symbols ~f:(fun (coeff, s) -> if coeff = 1 then Some s else None)
              in
              (coeff1_syms @ syms, true, has_sub, num_concat)
          | Idx.Sub_axis -> (syms, has_aff, true, num_concat)
          | Idx.Concat syms_list -> (syms_list @ syms, has_aff, has_sub, num_concat + 1))
    in
    if num_concat_axes > 1 then
      (* With multiple LHS Concat axes, we either have a block tensor (disjoint symbols in Concats),
         or a partially-diagonal tensor (overlapping symbols in Concats). *)
      false
    else
      let lhs_symbol_set = Set.of_list (module Idx.Symbol) lhs_symbols in
      let product_symbol_set =
        Set.of_list (module Idx.Symbol) (Array.to_list proj.Idx.product_iterators |> List.concat)
      in
      (* Count only LHS axes that need coverage by iterator symbols. [Fixed_idx 0] on a trivial (dim
         <= 1) axis is already covered: there is a single position and it is written. Counting such
         axes would spuriously fail the symbol-count check below for scalar / all-dims-1 tensors,
         falsely reporting them as non-surjective. *)
      let non_trivial_lhs_count =
        Array.foldi proj.Idx.project_lhs ~init:0 ~f:(fun i acc idx ->
            match idx with Idx.Fixed_idx 0 when proj.Idx.lhs_dims.(i) <= 1 -> acc | _ -> acc + 1)
      in

      (* All lhs symbols must be from product iterators (no bound symbols) *)
      if not (Set.is_subset lhs_symbol_set ~of_:product_symbol_set) then false
      else if has_sub_axis then
        (* Conservative: Sub_axis case is complex, so assume non-surjective. This is pessimistic but
           safe - Sub_axis would require comparing lhs_dims and product_space dimensions
           carefully. *)
        false
      else if has_affine then
        (* For Affine indices with strides: check coefficient compatibility. A strided access
           pattern may skip elements. *)
        let symbol_dims =
          Array.foldi proj.Idx.product_iterators ~init:[] ~f:(fun i acc syms ->
              let dims = proj.Idx.product_space.(i) in
              let pairs = List.zip_exn syms dims in
              List.fold pairs ~init:acc ~f:(fun acc (sym, d) ->
                  if Set.mem lhs_symbol_set sym then (sym, d) :: acc else acc))
          |> Map.of_alist_exn (module Idx.Symbol)
        in
        let check_affine_surjective =
          Array.for_all proj.Idx.project_lhs ~f:(function
            | Idx.Affine { symbols; _ } ->
                (* Find max dimension of coeff=1 symbols *)
                let max_coeff1_dim =
                  List.filter_map symbols ~f:(fun (coeff, s) ->
                      if coeff = 1 then Map.find symbol_dims s else None)
                  |> List.max_elt ~compare:Int.compare
                  |> Option.value ~default:Int.max_value
                in
                (* Check that coeff=1 dimension is not smaller than any stride *)
                List.for_all symbols ~f:(fun (coeff, _) -> coeff = 1 || max_coeff1_dim >= coeff)
            | _ -> true)
        in
        if not check_affine_surjective then false
        else
          (* Check that we have enough unique symbols to cover all LHS dimensions *)
          Set.length lhs_symbol_set >= non_trivial_lhs_count
      else
        (* Simple case: only Iterator and Fixed_idx *)
        (* Need enough unique symbols to cover all dimensions *)
        Set.length lhs_symbol_set >= non_trivial_lhs_count

let is_injective (proj : Idx.projections) =
  let all_product_iterators =
    Set.of_list (module Idx.Symbol) (Array.to_list proj.Idx.product_iterators |> List.concat)
  in
  let product_iterator_sets =
    Array.fold proj.Idx.product_iterators ~init:[ [] ] ~f:(fun acc syms ->
        List.concat_map acc ~f:(fun combination -> List.map syms ~f:(fun s -> s :: combination)))
    |> List.map ~f:(Set.of_list (module Idx.Symbol))
  in
  (* Per-symbol loop width (range), derived from the product space. Symbols absent from the product
     space (e.g. static indices) default to range 1 and are treated as pinned/static. *)
  let symbol_range_map =
    Array.fold2_exn proj.Idx.product_iterators proj.Idx.product_space
      ~init:(Map.empty (module Idx.Symbol))
      ~f:(fun acc syms dims ->
        match List.zip syms dims with
        | Ok pairs -> List.fold pairs ~init:acc ~f:(fun acc (s, d) -> Map.set acc ~key:s ~data:d)
        | Unequal_lengths -> acc)
  in
  let symbol_range s = Map.find symbol_range_map s |> Option.value ~default:1 in
  (* gh-133 Stage B: the affine LHS map must be injective over its non-static symbols (mixed-radix
     per-position criterion + whole-LHS pinning fixpoint). Previously any [Affine] position with
     more than one product iterator was rejected outright. *)
  let is_injective_mapping = Idx.affine_injective ~symbol_range proj.Idx.project_lhs in
  (* Symbols (product iterators only) appearing on the LHS, for the block-coverage check below. *)
  let lhs_symbols =
    Array.fold proj.Idx.project_lhs ~init:[] ~f:(fun syms idx ->
        match idx with
        | Idx.Iterator s -> s :: syms
        | Idx.Fixed_idx _ | Idx.Sub_axis -> syms
        | Idx.Affine { symbols; _ } ->
            List.filter_map symbols ~f:(fun (_coeff, s) ->
                Option.some_if (Set.mem all_product_iterators s) s)
            @ syms
        | Idx.Concat syms_list -> syms_list @ syms)
  in

  if not is_injective_mapping then false
  else
    let lhs_symbol_set = Set.of_list (module Idx.Symbol) lhs_symbols in
    (* For injectivity, each product iterator of a valid input block must map to at most one
       position *)
    let good, bad =
      List.partition_tf product_iterator_sets ~f:(Set.is_subset ~of_:lhs_symbol_set)
    in
    if List.is_empty good then false
    else List.for_all bad ~f:(fun s -> not @@ Set.is_subset ~of_:s lhs_symbol_set)

(** {2 Access records}

    The extraction target for [Low_level.affine_accesses] (gh-494 waypoint 1): each tensor-node
    access as an explicit affine relation — the enclosing loop box, the index map into the node's
    cells, and the program placement. ['tn] abstracts the tensor-node type to keep this module below
    [Tnode] in the dependency order. *)

type 'tn access = {
  a_tn : 'tn;
  a_map : Idx.axis_index array;
      (** The affine map from the loop box into the node's cells. Empty and standing for every cell
          when [a_whole]. *)
  a_write : bool;
  a_dynamic : bool;
      (** The effective cell is not statically known (dynamic gather/scatter): the map has a
          placeholder component, so queries must not interpret it. *)
  a_whole : bool;  (** A whole-node access ([Zero_out]). *)
  a_vec_last : bool;
      (** A vectorized write ([Set_from_vec]): the last map component is the base of a run along the
          minor axis, not a single cell — queries must treat that component as opaque. *)
  a_vec_len : int;
      (** The run length of a vectorized write along the minor axis; [0] unless [a_vec_last]. *)
  a_guarded : bool;  (** Under an [If] guard: executes conditionally, never a definite write. *)
  a_rmw : bool;
      (** The statement also reads [a_tn] on its right-hand side (an accumulation): the write
          carries a reduction dependence — an order-sensitive legality dimension (the determinism
          contract): a loop carrying only reduction edges may be reassociated (vectorized) under an
          explicit license, but never parallelized. *)
  a_val_syms : Idx.symbol list;
      (** Writes only: loop symbols the written value depends on syntactically (index symbols of rhs
          reads, embedded indices, dynamic-index sub-expressions). Direct dependence only — a chain
          through another node's cells is not tracked. *)
  a_stmt_write : Idx.axis_index array option;
      (** Reads only: the index map of the enclosing [Set]/[Set_from_vec]/[Set_dynamic] statement's
          write when the read occurs in that statement's right-hand side; [None] elsewhere ([If]
          conditions and [Local_scope] inner statements carry their own statements' writes). The
          subject of the read-modify-write exemption ([Low_level.rmw_exempt]): matching by
          statement subordination rather than by program path, so a guarded body's write cannot
          alias its [If] condition's read (they share a path). *)
  a_loops : (Idx.symbol * (int * int)) list;
      (** Enclosing loops, outermost first, with inclusive iteration bounds. *)
  a_path : int list;
      (** Lexicographic program-order position: statement indices per [Seq] nesting level. *)
}
[@@deriving sexp_of]

(** Whether two accesses (of the same node) can touch a common cell, each access taken over its
    whole loop box — the two sides' iterations paired independently, including iterations of loops
    the sides share (the accesses need not be simultaneous, so a shared loop's symbol varies
    independently between one side's visit and the other's). Symbols bound by neither side's loops
    (static indices) are shared parameters, equal on both sides, bounded by [static_range] when
    known. Conservative: [false] only when {!pair_conflict} proves disjointness; uninterpretable
    access kinds (dynamic, whole-node, vectorized) count as overlapping. *)
let may_touch_same_cell ?(static_range = fun _ -> None) (a : 'tn access) (b : 'tn access) : bool =
  if a.a_dynamic || b.a_dynamic || a.a_whole || b.a_whole || a.a_vec_last || b.a_vec_last then true
  else
    let range s =
      match List.Assoc.find a.a_loops s ~equal:Idx.equal_symbol with
      | Some bounds -> Some bounds
      | None -> (
          match List.Assoc.find b.a_loops s ~equal:Idx.equal_symbol with
          | Some bounds -> Some bounds
          | None -> static_range s)
    in
    let dup_left s = List.Assoc.mem a.a_loops s ~equal:Idx.equal_symbol in
    let dup_right s = List.Assoc.mem b.a_loops s ~equal:Idx.equal_symbol in
    match pair_conflict ~range ~dup_left ~dup_right ~pairs:[] ~left:a.a_map ~right:b.a_map with
    | Disjoint -> false
    | Same_thread | Cross_thread _ -> true

(** {2 The containment query}

    [read_covered_before ~read ~writes ()]: is every cell the [read] access can touch necessarily
    written before the read executes — the dominance side of dependence analysis, and the fifth
    decision procedure (gh-494 waypoint 2). Unlike the ∃-flavored {!pair_conflict} (negated to prove
    disjointness), containment is a ∀∃ query — for every read instance there must exist a covering
    write instance — so the variable treatment differs: the read's own loop symbols are universal, a
    write's own loop symbols (below the loops common with the read) are existential, and symbols
    shared by both sides (common enclosing loops, thread identity under [?thread], static indices)
    are parameters that must cancel per axis. A residual parameter on the write side declines that
    write (it would pin the write to one parameter value, or — for a common loop — refer to other
    iterations of that loop); a residual common-loop or static parameter on the read side is soundly
    universalized over its range; a residual thread parameter on the read side declines against a
    thread-bound write (the thread would read a cell another thread wrote), but universalizes
    against a write not under that thread loop — such a write executes redundantly on every thread,
    so each thread's copy receives it.

    Visibility is same-common-iteration program order: a write is usable when its statement path is
    lexicographically before the read's, and coverage is proven within the current iteration of the
    common enclosing loops (the write's whole subtree, including its own inner loops, has then
    executed). Loop-carried coverage — a read covered only by earlier iterations of a shared loop —
    is declined, conservatively.

    With [?thread] naming the parallel (thread-identity) symbols, [`Covered] proves the cell side of
    the per-thread-copy transform: the thread reads only cells it wrote itself, earlier in its own
    serial chunk. For reads covered within the same top-level statement that also proves the VALUE
    correct — chunks are serially contiguous there, so the thread's own write is the serial
    last-writer. A cross-statement covering write needs a side condition the caller must supply:
    other chunks of the writing statement run serially after the reader's own chunk, so the values
    coincide only when the written value cannot vary across the chunks writing the same cell — every
    thread symbol feeding the value ([a_val_syms]) must also pin the written cell (appear in the
    write's map).

    Guarded writes are the caller's choice: include them to mirror guards-taken analyses
    ([Low_level.trace_node_facts] and the coverage queries take guards unconditionally), pre-filter [a_guarded] for
    execution-accurate coverage. [writes] must be accesses of the same node as [read]. *)

(* Value set of one axis component, abstracted as an arithmetic progression {ap_lo, ap_lo + ap_step,
   ..., ap_hi} (ap_step = 0 iff singleton). For the write (superset) side the form must be dense —
   actually attaining every progression point — for the read (subset) side a hull suffices (superset
   of the actual set, sound on the left of ⊆). *)
type ap = { ap_lo : int; ap_hi : int; ap_step : int } [@@deriving sexp_of]

let floor_div a b = if a >= 0 then a / b else -((-a + b - 1) / b)
let ceil_div a b = if a >= 0 then (a + b - 1) / b else -(-a / b)

(* [terms]: [(coeff, (lo, hi))] with nonzero coeffs and nondegenerate ranges (width-1 symbols are
   folded into the offset by the caller). *)
let ap_of_form ~exact terms offset : ap option =
  match terms with
  | [] -> Some { ap_lo = offset; ap_hi = offset; ap_step = 0 }
  | _ ->
      let lo, hi =
        List.fold terms ~init:(offset, offset) ~f:(fun (lo, hi) (c, (vlo, vhi)) ->
            (lo + min (c * vlo) (c * vhi), hi + max (c * vlo) (c * vhi)))
      in
      let g = List.fold terms ~init:0 ~f:(fun g (c, _) -> gcd g c) in
      let dense =
        (* Sorted by ascending magnitude, each coefficient must not out-jump the span already
           reachable plus one step: then every multiple of [g] in [lo, hi] is attained. *)
        let sorted =
          List.sort
            (List.map terms ~f:(fun (c, (vlo, vhi)) -> (abs c, vhi - vlo)))
            ~compare:(fun (c1, _) (c2, _) -> Int.compare c1 c2)
        in
        let rec go span = function
          | [] -> true
          | (c, w) :: tl -> c <= g + span && go (span + (c * w)) tl
        in
        go 0 sorted
      in
      if exact && not dense then None else Some { ap_lo = lo; ap_hi = hi; ap_step = g }

(* r ⊆ w, where w is dense. *)
let ap_subset r w =
  r.ap_lo >= w.ap_lo && r.ap_hi <= w.ap_hi
  &&
  if w.ap_step = 0 then r.ap_lo = r.ap_hi
  else (r.ap_lo - w.ap_lo) % w.ap_step = 0 && r.ap_step % w.ap_step = 0

(* The contiguous run of r's lattice indices k (cell [r.ap_lo + k * r.ap_step]) that dense w covers;
   [None] when w's lattice does not include r's points wholesale. *)
let ap_covered_chunk r w : (int * int) option =
  let k_max = if r.ap_step = 0 then 0 else (r.ap_hi - r.ap_lo) / r.ap_step in
  let k_of v = if r.ap_step = 0 then 0 else (v - r.ap_lo) / r.ap_step in
  if w.ap_step = 0 then
    if
      w.ap_lo >= r.ap_lo && w.ap_lo <= r.ap_hi
      && (r.ap_step = 0 || (w.ap_lo - r.ap_lo) % r.ap_step = 0)
    then
      let k = k_of w.ap_lo in
      Some (k, k)
    else None
  else if (r.ap_lo - w.ap_lo) % w.ap_step = 0 && r.ap_step % w.ap_step = 0 then
    let k_lo =
      if r.ap_step = 0 then if w.ap_lo <= r.ap_lo then 0 else 1
      else max 0 (ceil_div (w.ap_lo - r.ap_lo) r.ap_step)
    and k_hi =
      if r.ap_step = 0 then if w.ap_hi >= r.ap_lo then 0 else -1
      else min k_max (floor_div (w.ap_hi - r.ap_lo) r.ap_step)
    in
    if k_lo <= k_hi then Some (k_lo, k_hi) else None
  else None

(* Linear view of one axis component: terms plus offset, [None] for uninterpretable ones. *)
let linear_terms (idx : Idx.axis_index) : ((int * Idx.symbol) list * int) option =
  match idx with
  | Idx.Fixed_idx c -> Some ([], c)
  | Idx.Iterator s -> Some ([ (1, s) ], 0)
  | Idx.Affine { symbols; offset } -> Some (Idx.coalesce_affine_terms symbols, offset)
  | Idx.Sub_axis | Idx.Concat _ -> None

let read_covered_before ?(thread = fun _ -> false) ?(static_range = fun _ -> None)
    ~(read : 'tn access) ~(writes : 'tn access list) () : [ `Covered | `Unknown of string ] =
  let exception Fail of string in
  let path_before p q =
    (* Lexicographic statement order, except that a prefix path is an ENCLOSING statement
       ([Local_scope] bodies extend the enclosing statement's path): the enclosing write happens
       after its rhs body executes, so it is not prior to reads inside the body. *)
    List.compare Int.compare p q < 0 && not (List.is_prefix q ~prefix:p ~equal:Int.equal)
  in
  let has_opaque m =
    Array.exists m ~f:(function Idx.Sub_axis | Idx.Concat _ -> true | _ -> false)
  in
  try
    if read.a_dynamic then raise (Fail "dynamic read");
    if read.a_whole || read.a_vec_last then raise (Fail "uninterpretable read kind");
    if has_opaque read.a_map then raise (Fail "opaque read component");
    let usable =
      List.filter writes ~f:(fun w ->
          w.a_write && (not w.a_dynamic) && path_before w.a_path read.a_path)
    in
    if List.is_empty usable then raise (Fail "no prior writes");
    if List.exists usable ~f:(fun w -> w.a_whole) then `Covered
    else begin
      let read_range s = List.Assoc.find read.a_loops s ~equal:Idx.equal_symbol in
      let rank =
        List.fold usable ~init:(Array.length read.a_map) ~f:(fun m w ->
            max m (Array.length w.a_map))
      in
      let comp m p = if p < Array.length m then m.(p) else Idx.Fixed_idx 0 in
      (* Per write: per-axis relation to the read's cells, in the residual coordinate frame left by
         parameter cancellation. [None] = this write proves nothing. *)
      let analyze (w : 'tn access) :
          (string * ap array * [ `Full | `Chunk of int * int | `Nope ] array) option =
        let exception Skip in
        try
          if has_opaque w.a_map && not w.a_vec_last then raise Skip;
          let c_len =
            let rec go n rl wl =
              match (rl, wl) with
              | (s1, b1) :: rt, (s2, b2) :: wt
                when Idx.equal_symbol s1 s2 && [%equal: int * int] b1 b2 ->
                  go (n + 1) rt wt
              | _ -> n
            in
            go 0 read.a_loops w.a_loops
          in
          let common = List.map (List.take read.a_loops c_len) ~f:fst in
          let is_common s = List.mem common s ~equal:Idx.equal_symbol in
          let w_own_loops = List.drop w.a_loops c_len in
          let w_own_range s = List.Assoc.find w_own_loops s ~equal:Idx.equal_symbol in
          (* Existential symbols used so far, per write: a symbol tying two axes of the write makes
             the per-axis image factorization an overapproximation of the written set — unsound on
             the superset side — so reuse declines the write. *)
          let used_exist = ref [] in
          let sig_parts = ref [] in
          let vec_axis = if w.a_vec_last then Array.length w.a_map - 1 else -1 in
          let r_aps = Array.create ~len:rank { ap_lo = 0; ap_hi = 0; ap_step = 0 } in
          let rels =
            Array.init rank ~f:(fun p ->
                match (linear_terms (comp read.a_map p), linear_terms (comp w.a_map p)) with
                | None, _ | _, None -> raise Skip
                | Some (rts, ro), Some (wts, wo) -> (
                    (* Split each side into shared-parameter terms and own terms. *)
                    let shared_of side_own_range (c, s) =
                      if thread s || is_common s || Option.is_none (side_own_range s) then
                        Either.First (c, s)
                      else Either.Second (c, s)
                    in
                    let r_shared, r_own = List.partition_map rts ~f:(shared_of read_range) in
                    (* A read-own symbol is one of the read's loops below the common prefix. *)
                    let r_own =
                      List.map r_own ~f:(fun (c, s) ->
                          match read_range s with
                          | Some b -> (c, b)
                          | None -> raise Skip (* cannot happen: partition used read_range *))
                    in
                    let w_shared, w_own = List.partition_map wts ~f:(shared_of w_own_range) in
                    let w_own =
                      List.map w_own ~f:(fun (c, s) ->
                          match w_own_range s with
                          | Some b ->
                              if List.mem !used_exist s ~equal:Idx.equal_symbol then raise Skip;
                              used_exist := s :: !used_exist;
                              (c, s, b)
                          | None -> raise Skip)
                    in
                    (* Cancel parameters matched by coefficient; residuals: write side declines,
                       read side universalizes over its range when one is known. *)
                    let w_par = ref w_shared in
                    let r_resid =
                      List.filter_map r_shared ~f:(fun (c, s) ->
                          match
                            List.findi !w_par ~f:(fun _ (c', s') -> Idx.equal_symbol s s' && c = c')
                          with
                          | Some (i, _) ->
                              w_par := List.filteri !w_par ~f:(fun j _ -> j <> i);
                              None
                          | None -> Some (c, s))
                    in
                    if not (List.is_empty !w_par) then raise Skip;
                    let r_univ =
                      List.map r_resid ~f:(fun (c, s) ->
                          (* A residual thread parameter declines only against a thread-bound write
                             (the thread would read a cell another thread wrote); a write not under
                             the thread loop executes redundantly on every thread — each thread's
                             copy receives it — so the read side universalizes. *)
                          if thread s && List.Assoc.mem w.a_loops s ~equal:Idx.equal_symbol then
                            raise Skip;
                          match
                            if is_common s || thread s then read_range s else static_range s
                          with
                          | Some b -> (c, b)
                          | None -> raise Skip)
                    in
                    (* Fold width-1 ranges into offsets; drop zero-width... widths >= 1 always. *)
                    let fold_const terms off =
                      List.fold terms ~init:([], off) ~f:(fun (ts, off) (c, (lo, hi)) ->
                          if lo = hi then (ts, off + (c * lo)) else ((c, (lo, hi)) :: ts, off))
                    in
                    let r_terms, r_off = fold_const (r_own @ r_univ) ro in
                    let w_terms, w_off =
                      fold_const (List.map w_own ~f:(fun (c, _, b) -> (c, b))) wo
                    in
                    sig_parts :=
                      (p, List.sort r_resid ~compare:[%compare: int * Idx.symbol]) :: !sig_parts;
                    let r_ap = Option.value_exn (ap_of_form ~exact:false r_terms r_off) in
                    r_aps.(p) <- r_ap;
                    let w_ap =
                      match ap_of_form ~exact:true w_terms w_off with
                      | None -> None
                      | Some w_ap when p = vec_axis ->
                          (* The vectorized run extends the base progression along the minor axis;
                             contiguous only when runs at least abut. *)
                          let len = w.a_vec_len in
                          if w_ap.ap_step = 0 then
                            Some { ap_lo = w_ap.ap_lo; ap_hi = w_ap.ap_lo + len - 1; ap_step = 1 }
                          else if w_ap.ap_step <= len then
                            Some { ap_lo = w_ap.ap_lo; ap_hi = w_ap.ap_hi + len - 1; ap_step = 1 }
                          else None
                      | some -> some
                    in
                    match w_ap with
                    | None -> `Nope
                    | Some w_ap -> (
                        if ap_subset r_ap w_ap then `Full
                        else
                          match ap_covered_chunk r_ap w_ap with
                          | Some (kl, kh) -> `Chunk (kl, kh)
                          | None -> `Nope)))
          in
          (* The residual coordinate frame: which parameters were cancelled vs. universalized shifts
             what the chunk indices mean, so unionable writes must agree on both the residual
             parameter lists and the resulting read-side progressions. *)
          let signature =
            Sexp.to_string
              ([%sexp_of: (int * (int * Idx.symbol) list) list * ap array]
                 (List.sort !sig_parts ~compare:(fun (p, _) (q, _) -> Int.compare p q), r_aps))
          in
          Some (signature, r_aps, rels)
        with Skip -> None
      in
      let analyzed = List.filter_map usable ~f:analyze in
      if
        List.exists analyzed ~f:(fun (_, _, rels) ->
            Array.for_all rels ~f:(function `Full -> true | _ -> false))
      then `Covered
      else begin
        (* Union rule: writes sharing a residual frame contribute product boxes of per-axis chunks
           (sound: per-axis independence of each write's existentials makes its image the product of
           its per-axis sets); exact box-union coverage of the read's lattice box by
           coordinate-compression sweep — the first axis is cut at every box boundary, boxes
           spanning an elementary strip recurse on the remaining axes. This is what covers padded
           tensors (margin strips plus the interior tile the box) and literal initializations
           (pointwise writes tile it). *)
        let by_sig = Hashtbl.create (module String) in
        List.iter analyzed ~f:(fun ((s, _, _) as a) -> Hashtbl.add_multi by_sig ~key:s ~data:a);
        let rec boxes_cover (target : (int * int) list) (boxes : (int * int) list list) =
          match target with
          | [] -> not (List.is_empty boxes)
          | (lo, hi) :: rest_t ->
              if lo > hi then true
              else
                let cuts =
                  lo
                  :: List.concat_map boxes ~f:(function
                    | (bl, bh) :: _ -> [ bl; bh + 1 ]
                    | [] -> [])
                  |> List.filter ~f:(fun x -> x >= lo && x <= hi)
                  |> List.dedup_and_sort ~compare:Int.compare
                in
                List.for_all cuts ~f:(fun x ->
                    let spanning =
                      List.filter_map boxes ~f:(function
                        | (bl, bh) :: rest when bl <= x && x <= bh -> Some rest
                        | _ -> None)
                    in
                    boxes_cover rest_t spanning)
        in
        let union_covers group =
          match group with
          | [] -> false
          | (_, r_aps, _) :: _ ->
              let k_max r = if r.ap_step = 0 then 0 else (r.ap_hi - r.ap_lo) / r.ap_step in
              let target = Array.to_list (Array.map r_aps ~f:(fun r -> (0, k_max r))) in
              let boxes =
                List.filter_map group ~f:(fun (_, r_aps', rels) ->
                    Array.to_list
                      (Array.mapi rels ~f:(fun p rel ->
                           match rel with
                           | `Full -> Some (0, k_max r_aps'.(p))
                           | `Chunk (kl, kh) -> Some (kl, kh)
                           | `Nope -> None))
                    |> Option.all)
              in
              boxes_cover target boxes
        in
        if Hashtbl.exists by_sig ~f:union_covers then `Covered
        else
          raise
            (Fail
               (Printf.sprintf "read cells not covered by prior writes (read %s)"
                  (String.concat_array ~sep:"," (Array.map read.a_map ~f:axis_index_to_string))))
      end
    end
  with Fail witness -> `Unknown witness

(** {2 Crosscheck}

    Config [legality_crosscheck]: when enabled, the call sites swapped onto the queries also run the
    legacy procedural analysis and compare. A query stricter than the procedural answer raises —
    either a query precision regression or a latent unsoundness of the procedural rule, both needing
    eyes. A query more permissive than the procedural answer is the expected precision gain, logged
    to stderr for review. *)

let crosscheck_enabled = lazy (Utils.get_global_flag ~default:false ~arg_name:"legality_crosscheck")

let crosscheck ~site ~context ~(procedural_safe : unit -> bool) ~query_safe ~witness =
  if Lazy.force crosscheck_enabled && Bool.(procedural_safe () <> query_safe) then
    if not query_safe then
      invalid_arg
        (Printf.sprintf
           "Affine.crosscheck %s: the procedural analysis accepts but the affine query declines \
            (%s) — query precision regression or latent procedural unsoundness; context: %s"
           site witness context)
    else
      Stdio.eprintf
        "[legality_crosscheck] %s: the affine query accepts where the procedural analysis \
         declined; context: %s\n\
         %!"
        site context
