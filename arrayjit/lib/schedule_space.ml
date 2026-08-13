open Base

(* The partial-schedule decision space (gh-ocannl-514 phases 1-2). See schedule_space.mli. *)

type placement = Pl_inline | Pl_stage_at of Indexing.symbol | Pl_materialize
[@@deriving sexp_of, compare, equal]

type 'a child =
  | Child of 'a tree Lazy.t
  | Unknown of string * 'a tree Lazy.t
  | Excluded of string * 'a child Lazy.t
  | Refuted of string

and 'a tree = Leaf of 'a | Choice of { level : string; children : (string * 'a child) list }

let subtree = function
  | Child sub | Unknown (_, sub) -> Some sub
  | Excluded _ | Refuted _ -> None

let lift_excluded = function Excluded (_, c) -> Lazy.force c | c -> c

let rec leaves = function
  | Leaf a -> [ a ]
  | Choice { children; _ } ->
      List.concat_map children ~f:(fun (_, c) ->
          match subtree c with Some sub -> leaves (Lazy.force sub) | None -> [])

let enumerate tree =
  let rec go rev_path = function
    | Leaf a -> [ (List.rev rev_path, a) ]
    | Choice { level; children } ->
        List.concat_map children ~f:(fun (label, c) ->
            match subtree c with
            | Some sub -> go ((level, label) :: rev_path) (Lazy.force sub)
            | None -> [])
  in
  go [] tree

(* One collector for the three witness-carrying verdicts; each report's path ends at the judged
   (level, label). [Unknown] children are also descended into — their subtrees can carry further
   verdicts. *)
let collect ~f tree =
  let rec go rev_path = function
    | Leaf _ -> []
    | Choice { level; children } ->
        List.concat_map children ~f:(fun (label, c) ->
            let path = List.rev ((level, label) :: rev_path) in
            let here = match f c with Some w -> [ (path, w) ] | None -> [] in
            here
            @
            match subtree c with
            | Some sub -> go ((level, label) :: rev_path) (Lazy.force sub)
            | None -> [])
  in
  go [] tree

let refutations tree = collect tree ~f:(function Refuted w -> Some w | _ -> None)
let exclusions tree = collect tree ~f:(function Excluded (w, _) -> Some w | _ -> None)
let unknowns tree = collect tree ~f:(function Unknown (w, _) -> Some w | _ -> None)

type search_stats = {
  st_expanded : int;
  st_scored : int;
  st_unscored : int;
  st_fathomed : int;
  st_refuted : int;
  st_excluded : int;
  st_unknown : int;
}
[@@deriving sexp_of]

let no_search_stats =
  {
    st_expanded = 0;
    st_scored = 0;
    st_unscored = 0;
    st_fathomed = 0;
    st_refuted = 0;
    st_excluded = 0;
    st_unknown = 0;
  }

let search ?bound ?incumbent ~score tree =
  let stats = ref no_search_stats in
  let best = ref None in
  let threshold = ref (Option.value incumbent ~default:Float.infinity) in
  let fathoms rev_path sub =
    (* An optimistic bound at or above the threshold refutes strict improvement for every
       completion below — equality fathoms because displacing the incumbent needs [<]. *)
    match Option.bind bound ~f:(fun b -> b ~path:(List.rev rev_path) sub) with
    | Some b -> Float.(b >= !threshold)
    | None -> false
  in
  let rec leaf_or_choice rev_path t =
    match t with
    | Leaf a -> (
        match score a with
        | Some s ->
            stats := { !stats with st_scored = !stats.st_scored + 1 };
            if Float.(s < !threshold) then (
              best := Some (a, s);
              threshold := s)
        | None -> stats := { !stats with st_unscored = !stats.st_unscored + 1 })
    | Choice { level; children } ->
        stats := { !stats with st_expanded = !stats.st_expanded + 1 };
        List.iter children ~f:(fun (label, c) ->
            let rev_path = (level, label) :: rev_path in
            match c with
            | Refuted _ -> stats := { !stats with st_refuted = !stats.st_refuted + 1 }
            | Excluded _ -> stats := { !stats with st_excluded = !stats.st_excluded + 1 }
            | Unknown (_, sub) ->
                (* Never fathomed: an unknown verdict must reach scoring (or, in a measured
                   search, compilation). *)
                stats := { !stats with st_unknown = !stats.st_unknown + 1 };
                leaf_or_choice rev_path (Lazy.force sub)
            | Child sub ->
                let sub = Lazy.force sub in
                if fathoms rev_path sub then
                  stats := { !stats with st_fathomed = !stats.st_fathomed + 1 }
                else leaf_or_choice rev_path sub)
  in
  (if fathoms [] tree then stats := { !stats with st_fathomed = 1 } else leaf_or_choice [] tree);
  (!best, !stats)

let rec count_choices = function
  | Leaf _ -> 0
  | Choice { children; _ } ->
      1
      + List.sum
          (module Int)
          children
          ~f:(fun (_, c) ->
            match subtree c with Some sub -> count_choices (Lazy.force sub) | None -> 0)

let rec depth = function
  | Leaf _ -> 0
  | Choice { children; _ } ->
      1
      + List.fold children ~init:0 ~f:(fun acc (_, c) ->
            match subtree c with
            | Some sub -> Int.max acc (depth (Lazy.force sub))
            | None -> acc)
