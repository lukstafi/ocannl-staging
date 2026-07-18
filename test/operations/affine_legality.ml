(* gh-494 waypoint 1-2: unit + brute-force-oracle coverage for the [Ir.Affine] query engine.

   [pair_conflict] and [covers_box] are sound-and-conservative decision procedures; on small
   extents their answers are checkable by exhaustive enumeration of the iteration boxes. Every case
   asserts soundness (a proven verdict must agree with the enumeration) and prints both the query
   verdict and the oracle classification, so precision gaps (query [Cross_thread] where the oracle
   found no cross-thread conflict) are visible and reviewable rather than silent. *)

open Base
module Idx = Ir.Indexing
module Aff = Ir.Affine

let sym = Idx.get_symbol
let aff terms offset = Idx.Affine { symbols = terms; offset }

let find_range ranges s = List.Assoc.find ranges s ~equal:Idx.equal_symbol

(* Enumerate all assignments of [syms] within [ranges] (inclusive bounds). *)
let envs ranges syms =
  List.fold syms ~init:[ [] ] ~f:(fun acc s ->
      let lo, hi = Option.value_exn (find_range ranges s) in
      List.concat_map acc ~f:(fun env ->
          List.init (hi - lo + 1) ~f:(fun k -> (s, lo + k) :: env)))

let eval_idx env (idx : Idx.axis_index) : int option =
  let v s = List.Assoc.find env s ~equal:Idx.equal_symbol in
  match idx with
  | Idx.Fixed_idx c -> Some c
  | Idx.Iterator s -> v s
  | Idx.Affine { symbols; offset } ->
      List.fold symbols ~init:(Some offset) ~f:(fun acc (c, s) ->
          match (acc, v s) with Some a, Some x -> Some (a + (c * x)) | _ -> None)
  | Idx.Sub_axis | Idx.Concat _ -> None

let eval_vec env (idcs : Idx.axis_index array) rank : int list option =
  List.init rank ~f:(fun p ->
      eval_idx env (if p < Array.length idcs then idcs.(p) else Idx.Fixed_idx 0))
  |> Option.all

(* Oracle for [pair_conflict]: enumerate the shared symbols once and each side's duplicated symbols
   independently; classify the conflicts (common cells) by whether all [pairs] agree. [`Opaque] when
   some component cannot be evaluated. *)
let oracle_conflict ~oracle_ranges ~dup_left ~dup_right ~pairs ~left ~right =
  let mentioned idcs =
    Array.to_list idcs
    |> List.concat_map ~f:(function
         | Idx.Iterator s -> [ s ]
         | Idx.Affine { symbols; _ } -> List.map symbols ~f:snd
         | _ -> [])
  in
  let dedup = List.dedup_and_sort ~compare:Idx.compare_symbol in
  let lsyms = dedup (mentioned left) and rsyms = dedup (mentioned right) in
  let shared = List.filter (dedup (lsyms @ rsyms)) ~f:(fun s -> not (dup_left s || dup_right s)) in
  let l_only = List.filter lsyms ~f:dup_left and r_only = List.filter rsyms ~f:dup_right in
  let rank = max (Array.length left) (Array.length right) in
  let conflicts = ref [] and opaque = ref false in
  List.iter (envs oracle_ranges shared) ~f:(fun senv ->
      List.iter (envs oracle_ranges l_only) ~f:(fun lenv ->
          List.iter (envs oracle_ranges r_only) ~f:(fun renv ->
              match (eval_vec (senv @ lenv) left rank, eval_vec (senv @ renv) right rank) with
              | Some lv, Some rv ->
                  if List.equal Int.equal lv rv then conflicts := (senv @ lenv, senv @ renv) :: !conflicts
              | _ -> opaque := true)));
  if !opaque then `Opaque
  else if List.is_empty !conflicts then `Disjoint
  else if
    List.for_all !conflicts ~f:(fun (lenv, renv) ->
        List.for_all pairs ~f:(fun (p, p') ->
            match
              ( List.Assoc.find lenv p ~equal:Idx.equal_symbol,
                List.Assoc.find renv p' ~equal:Idx.equal_symbol )
            with
            | Some a, Some b -> a = b
            (* A parallel symbol absent from an access's environment takes every value: only a
               conflict-free pairing may treat it as confined. *)
            | _ -> false))
  then `Same_thread
  else `Cross_thread

let show_oracle = function
  | `Disjoint -> "disjoint"
  | `Same_thread -> "same-thread"
  | `Cross_thread -> "cross-thread"
  | `Opaque -> "opaque"

let show_verdict = function
  | Aff.Disjoint -> "Disjoint"
  | Aff.Same_thread -> "Same_thread"
  | Aff.Cross_thread _ -> "Cross_thread"

(* Soundness: a proven query verdict must not overclaim relative to the enumeration. An [`Opaque]
   oracle (some component it cannot evaluate) falsifies nothing: the query's proven verdicts remain
   sound because skipping an uninterpretable axis only enlarges the conflict set it reasons over. *)
let sound verdict oracle =
  match (verdict, oracle) with
  | _, `Opaque -> true
  | Aff.Disjoint, `Disjoint -> true
  | Aff.Disjoint, _ -> false
  | Aff.Same_thread, (`Disjoint | `Same_thread) -> true
  | Aff.Same_thread, _ -> false
  | Aff.Cross_thread _, _ -> true

let unsound_count = ref 0

let check_conflict ~name ~query_ranges ~oracle_ranges ~dup_left ~dup_right ~pairs ~left ~right =
  let range s = find_range query_ranges s in
  let verdict = Aff.pair_conflict ~range ~dup_left ~dup_right ~pairs ~left ~right in
  let oracle = oracle_conflict ~oracle_ranges ~dup_left ~dup_right ~pairs ~left ~right in
  let ok = sound verdict oracle in
  if not ok then Int.incr unsound_count;
  Stdio.printf "%-34s query %-12s oracle %-12s%s\n" name (show_verdict verdict) (show_oracle oracle)
    (if ok then "" else "  UNSOUND")

(* Oracle for [covers_box]: enumerate the mentioned symbols, count visits per cell. *)
let oracle_covers ~oracle_ranges ~dims idcs =
  let mentioned =
    Array.to_list idcs
    |> List.concat_map ~f:(function
         | Idx.Iterator s -> [ s ]
         | Idx.Affine { symbols; _ } -> List.map symbols ~f:snd
         | _ -> [])
    |> List.dedup_and_sort ~compare:Idx.compare_symbol
  in
  let cells = Hashtbl.create (module String) in
  let opaque = ref false in
  List.iter (envs oracle_ranges mentioned) ~f:(fun env ->
      match eval_vec env idcs (Array.length dims) with
      | Some v ->
          if List.existsi v ~f:(fun a x -> x < 0 || x >= dims.(a)) then opaque := true
          else Hashtbl.incr cells (String.concat ~sep:"," (List.map v ~f:Int.to_string))
      | None -> opaque := true);
  (not !opaque)
  && Hashtbl.length cells = Array.fold dims ~init:1 ~f:( * )
  && Hashtbl.for_all cells ~f:(fun n -> n = 1)

let check_covers ~name ~query_ranges ~oracle_ranges ~dims idcs =
  let range s = find_range query_ranges s in
  let query = Aff.covers_box ~range ~dims idcs in
  let oracle = oracle_covers ~oracle_ranges ~dims idcs in
  let ok = (not query) || oracle in
  if not ok then Int.incr unsound_count;
  Stdio.printf "%-34s query %-12b oracle %-12b%s\n" name query oracle (if ok then "" else "  UNSOUND")

let () =
  Stdio.printf "=== pair_conflict: named cases ===\n";
  let p = sym () and q = sym () and j = sym () and u = sym () and v = sym () and g = sym () in
  let i0 = sym () and j0 = sym () and s = sym () and t = sym () in
  let dup_of syms sm = List.mem syms sm ~equal:Idx.equal_symbol in
  let r3 = (0, 2) in
  (* Same-nest sites: both sides iterate the same loop symbols. *)
  let same_nest ~name ?(ranges = []) ~syms ~pairs left right =
    let ranges = if List.is_empty ranges then List.map syms ~f:(fun s -> (s, r3)) else ranges in
    check_conflict ~name ~query_ranges:ranges ~oracle_ranges:ranges ~dup_left:(dup_of syms)
      ~dup_right:(dup_of syms)
      ~pairs:(List.map pairs ~f:(fun s -> (s, s)))
      ~left ~right
  in
  same_nest ~name:"iterator agreement" ~syms:[ p; j ] ~pairs:[ p ]
    [| Idx.Iterator p; Idx.Iterator j |]
    [| Idx.Iterator p; Idx.Iterator j |];
  same_nest ~name:"stencil read p-1" ~syms:[ p ] ~pairs:[ p ] [| Idx.Iterator p |]
    [| aff [ (1, p) ] (-1) |];
  same_nest ~name:"strided disjoint 2p vs 2p+1" ~syms:[ p ] ~pairs:[ p ] [| aff [ (2, p) ] 0 |]
    [| aff [ (2, p) ] 1 |];
  same_nest ~name:"non-injective match p+q" ~syms:[ p; q ] ~pairs:[ p ] [| aff [ (1, p); (1, q) ] 0 |]
    [| aff [ (1, p); (1, q) ] 0 |];
  same_nest ~name:"mixed radix s+4t (fits)" ~syms:[ s; t ]
    ~ranges:[ (s, (0, 3)); (t, (0, 2)) ]
    ~pairs:[ s; t ]
    [| aff [ (1, s); (4, t) ] 0 |]
    [| aff [ (1, s); (4, t) ] 0 |];
  same_nest ~name:"mixed radix s+4t (overlaps)" ~syms:[ s; t ]
    ~ranges:[ (s, (0, 4)); (t, (0, 2)) ]
    ~pairs:[ s; t ]
    [| aff [ (1, s); (4, t) ] 0 |]
    [| aff [ (1, s); (4, t) ] 0 |];
  same_nest ~name:"fixed disjoint slices" ~syms:[ j ] ~pairs:[ j ] [| Idx.Fixed_idx 3; Idx.Iterator j |]
    [| Idx.Fixed_idx 5; Idx.Iterator j |];
  same_nest ~name:"transposed access" ~syms:[ p; q ] ~pairs:[ p; q ]
    [| Idx.Iterator p; Idx.Iterator q |]
    [| Idx.Iterator q; Idx.Iterator p |];
  same_nest ~name:"read row 0" ~syms:[ p; j ] ~pairs:[ p ]
    [| Idx.Iterator p; Idx.Iterator j |]
    [| Idx.Fixed_idx 0; Idx.Iterator j |];
  same_nest ~name:"offset loop bounds [2,5]" ~syms:[ p ] ~ranges:[ (p, (2, 5)) ] ~pairs:[ p ]
    [| Idx.Iterator p |] [| Idx.Iterator p |];
  same_nest ~name:"rank padding" ~syms:[ p; j ] ~pairs:[ p ]
    [| Idx.Iterator p; Idx.Fixed_idx 0 |]
    [| Idx.Iterator p |];
  same_nest ~name:"sub_axis is opaque" ~syms:[ p ] ~pairs:[ p ] [| Idx.Sub_axis; Idx.Iterator p |]
    [| Idx.Sub_axis; Idx.Iterator p |];
  (* Static (shared) symbol: unknown range to the query, enumerated by the oracle. *)
  check_conflict ~name:"shared static symbol"
    ~query_ranges:[ (p, r3); (j, r3) ]
    ~oracle_ranges:[ (p, r3); (j, r3); (g, r3) ]
    ~dup_left:(dup_of [ p; j ]) ~dup_right:(dup_of [ p; j ]) ~pairs:[ (p, p) ]
    ~left:[| Idx.Iterator g; Idx.Iterator p |]
    ~right:[| Idx.Iterator g; Idx.Iterator p |];
  (* Cross-nest: distinct symbol sets, thread identity via pairing. *)
  check_conflict ~name:"cross-nest aligned"
    ~query_ranges:[ (i0, r3); (j0, r3); (u, r3); (v, r3) ]
    ~oracle_ranges:[ (i0, r3); (j0, r3); (u, r3); (v, r3) ]
    ~dup_left:(dup_of [ i0; u ]) ~dup_right:(dup_of [ j0; v ]) ~pairs:[ (i0, j0) ]
    ~left:[| Idx.Iterator i0; Idx.Iterator u |]
    ~right:[| Idx.Iterator j0; Idx.Iterator v |];
  check_conflict ~name:"cross-nest chain-position swap"
    ~query_ranges:[ (p, r3); (q, r3); (i0, r3); (j0, r3) ]
    ~oracle_ranges:[ (p, r3); (q, r3); (i0, r3); (j0, r3) ]
    ~dup_left:(dup_of [ p; q ]) ~dup_right:(dup_of [ i0; j0 ])
    ~pairs:[ (p, i0); (q, j0) ]
    ~left:[| Idx.Iterator p; Idx.Iterator q |]
    ~right:[| Idx.Iterator j0; Idx.Iterator i0 |];

  Stdio.printf "\n=== pair_conflict: exhaustive sweep (same-nest, pairs=[p]) ===\n";
  let components =
    [
      ("F0", Idx.Fixed_idx 0);
      ("F1", Idx.Fixed_idx 1);
      ("p", Idx.Iterator p);
      ("q", Idx.Iterator q);
      ("2p", aff [ (2, p) ] 0);
      ("p+1", aff [ (1, p) ] 1);
      ("p+3q", aff [ (1, p); (3, q) ] 0);
      ("p+q", aff [ (1, p); (1, q) ] 0);
    ]
  in
  let ranges = [ (p, r3); (q, r3) ] in
  let tally = Hashtbl.create (module String) in
  List.iter components ~f:(fun (ln, lc) ->
      List.iter components ~f:(fun (rn, rc) ->
          let range s = find_range ranges s in
          let dup = dup_of [ p; q ] in
          let verdict =
            Aff.pair_conflict ~range ~dup_left:dup ~dup_right:dup ~pairs:[ (p, p) ] ~left:[| lc |]
              ~right:[| rc |]
          in
          let oracle =
            oracle_conflict ~oracle_ranges:ranges ~dup_left:dup ~dup_right:dup ~pairs:[ (p, p) ]
              ~left:[| lc |] ~right:[| rc |]
          in
          if not (sound verdict oracle) then (
            Int.incr unsound_count;
            Stdio.printf "UNSOUND: %s vs %s: query %s oracle %s\n" ln rn (show_verdict verdict)
              (show_oracle oracle));
          Hashtbl.incr tally (show_verdict verdict ^ "/" ^ show_oracle oracle)));
  Hashtbl.to_alist tally
  |> List.sort ~compare:(fun (a, _) (b, _) -> String.compare a b)
  |> List.iter ~f:(fun (k, n) -> Stdio.printf "%-28s %d\n" k n);

  Stdio.printf "\n=== covers_box ===\n";
  let a = sym () and b = sym () and i = sym () in
  let qr = [ (i, (0, 2)); (j, (0, 3)); (a, (0, 2)); (b, (0, 3)) ] in
  let cov ~name ?(query_ranges = qr) ?(oracle_ranges = qr) ~dims idcs =
    check_covers ~name ~query_ranges ~oracle_ranges ~dims idcs
  in
  cov ~name:"iterators cover box" ~dims:[| 3; 4 |] [| Idx.Iterator i; Idx.Iterator j |];
  cov ~name:"iterator wrong extent" ~dims:[| 4 |] [| Idx.Iterator i |];
  cov ~name:"mixed radix a+3b" ~dims:[| 12 |] [| aff [ (1, a); (3, b) ] 0 |];
  cov ~name:"mixed radix wrong product" ~dims:[| 13 |] [| aff [ (1, a); (3, b) ] 0 |];
  cov ~name:"repeated symbol (diagonal)" ~dims:[| 3; 3 |] [| Idx.Iterator i; Idx.Iterator i |];
  cov ~name:"offset misses zero" ~dims:[| 3 |] [| aff [ (1, i) ] 1 |];
  cov ~name:"stride gap 2i" ~dims:[| 6 |] [| aff [ (2, i) ] 0 |];
  cov ~name:"fixed on unit axis" ~dims:[| 1; 3 |] [| Idx.Fixed_idx 0; Idx.Iterator i |];
  cov ~name:"fixed on wide axis" ~dims:[| 2; 3 |] [| Idx.Fixed_idx 0; Idx.Iterator i |];
  cov ~name:"non-zero-based loop"
    ~query_ranges:[ (i, (1, 3)) ]
    ~oracle_ranges:[ (i, (1, 3)) ]
    ~dims:[| 3 |] [| Idx.Iterator i |];

  Stdio.printf "\nunsound cases: %d\n" !unsound_count
