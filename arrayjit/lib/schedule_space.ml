open Base

(* The partial-schedule decision space (gh-ocannl-514 phase 1). See schedule_space.mli. *)

type placement = Pl_inline | Pl_stage_at of Indexing.symbol | Pl_materialize
[@@deriving sexp_of, compare, equal]

type 'a tree = Leaf of 'a | Choice of { level : string; children : (string * 'a tree Lazy.t) list }

let rec leaves = function
  | Leaf a -> [ a ]
  | Choice { children; _ } ->
      List.concat_map children ~f:(fun (_, sub) -> leaves (Lazy.force sub))

let enumerate tree =
  let rec go rev_path = function
    | Leaf a -> [ (List.rev rev_path, a) ]
    | Choice { level; children } ->
        List.concat_map children ~f:(fun (label, sub) ->
            go ((level, label) :: rev_path) (Lazy.force sub))
  in
  go [] tree

let rec count_choices = function
  | Leaf _ -> 0
  | Choice { children; _ } ->
      1 + List.sum (module Int) children ~f:(fun (_, sub) -> count_choices (Lazy.force sub))

let rec depth = function
  | Leaf _ -> 0
  | Choice { children; _ } ->
      1
      + List.fold children ~init:0 ~f:(fun acc (_, sub) -> Int.max acc (depth (Lazy.force sub)))
