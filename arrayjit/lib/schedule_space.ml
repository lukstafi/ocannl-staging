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
