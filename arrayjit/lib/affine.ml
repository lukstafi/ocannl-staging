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
      match term (1, s) with
      | Either.First t -> Some ([ t ], 0)
      | Either.Second c -> Some ([], c))
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
  | _ ->
      let g = List.fold terms ~init:0 ~f:(fun g (c, _) -> gcd g c) in
      rhs % g <> 0
      ||
      let bounds =
        List.fold terms ~init:(Some (0, 0)) ~f:(fun acc (c, v) ->
            match (acc, range_of_var ~range v) with
            | Some (lo, hi), Some (vlo, vhi) ->
                Some (lo + min (c * vlo) (c * vhi), hi + max (c * vlo) (c * vhi))
            | _ -> None)
      in
      (match bounds with Some (lo, hi) -> rhs < lo || rhs > hi | None -> false)

(* Forced equalities: an equation [Σ_k c_k·(x_k − y_k) = 0] with [x_k] all-[Left_v], [y_k]
   all-[Right_v], matched pairwise by coefficient, forces [x_k = y_k] for every k when the linear
   form [Σ c_k·z_k] is injective on the (per-pair combined) bounding box — the mixed-radix
   criterion of {!Indexing.affine_injective}: sorted by ascending [abs c], every term satisfies
   [abs c_k >= 1 + Σ_{i<k} abs c_i·(w_i − 1)]. Equal values of an injective form imply equal
   arguments, and the forcing holds in every solution of the conjunction it is part of. *)
let forced_pairs ~range (terms, rhs) : (Idx.symbol * Idx.symbol) list =
  if rhs <> 0 then []
  else
    let ls, rest = List.partition_tf terms ~f:(fun (_, v) -> match v with Left_v _ -> true | _ -> false) in
    let rs, shared = List.partition_tf rest ~f:(fun (_, v) -> match v with Right_v _ -> true | _ -> false) in
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
          let sorted = List.sort triples ~compare:(fun (c1, _, _) (c2, _, _) -> Int.compare c1 c2) in
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
      List.exists forced ~f:(fun (a, b) -> Idx.equal_symbol a p && Idx.equal_symbol b p')
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

(** [covers_box ~range ~dims idcs]: whether the index vector [idcs], as its symbols range over
    their (loop) bounds, enumerates every cell of the [dims] box exactly once — a bijection onto
    the box. This is the write-dominance building block: a covering unguarded write rewrites the
    whole array. Requirements: each symbol used at most once across the vector; per axis, a
    zero-based full-extent iterator, a mixed-radix affine combination of zero-based symbols whose
    radix chain exactly composes to the axis dimension, or [Fixed_idx 0] on a unit axis.
    Generalizes (and is checked against) the procedural per-axis rule of
    [C_syntax.first_access_standalone_covering]. *)
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
      | Idx.Iterator s -> (
          fresh s && match extent_of s with Some e -> e = dim | None -> false)
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
    cells, and the program placement. ['tn] abstracts the tensor-node type to keep this module
    below [Tnode] in the dependency order. *)

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
      (** A vectorized write ([Set_from_vec]): the last map component is the base of a run along
          the minor axis, not a single cell — queries must treat that component as opaque. *)
  a_guarded : bool;  (** Under an [If] guard: executes conditionally, never a definite write. *)
  a_rmw : bool;
      (** The statement also reads [a_tn] on its right-hand side (an accumulation): the write
          carries a reduction dependence — an order-sensitive legality dimension (the determinism
          contract): a loop carrying only reduction edges may be reassociated (vectorized) under an
          explicit license, but never parallelized. *)
  a_loops : (Idx.symbol * (int * int)) list;
      (** Enclosing loops, outermost first, with inclusive iteration bounds. *)
  a_path : int list;
      (** Lexicographic program-order position: statement indices per [Seq] nesting level. *)
}
[@@deriving sexp_of]

(** {2 Crosscheck}

    Config [legality_crosscheck]: when enabled, the call sites swapped onto the queries also run
    the legacy procedural analysis and compare. A query stricter than the procedural answer raises
    — either a query precision regression or a latent unsoundness of the procedural rule, both
    needing eyes. A query more permissive than the procedural answer is the expected precision
    gain, logged to stderr for review. *)

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
