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
