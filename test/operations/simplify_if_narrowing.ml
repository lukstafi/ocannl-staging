(* gh-ocannl-566: [simplify_llc] narrows its interval environment with the bounds an enclosing [If]
   condition implies for its body, so a guard the condition proves is folded away.

   Exposed by gh-ocannl-487 phase 1: the depth-2 pipelined prefetch loads block [k+1] under
   [If (k < to_)], and the [k := k+1] substitution widens its zero-fringe edge guard's index
   interval past the extent -- the loop range alone cannot decide it, the enclosing condition can.
   That end-to-end case is pinned (structurally and by executed bitwise parity) in
   [schedule_pipelined_matmul]; here each narrowing rule is exercised in isolation against a
   discriminating control that must NOT fold, since a folded guard silently changes the value a cell
   holds.

   High-level lowering does not produce these shapes on demand, so the cases are built directly as
   [Ir.Low_level.t] and run through [Ir.Low_level.simplify_llc] -- the pass under test, called with
   no static indices, exactly as [Schedule.apply]'s trailing simplify calls it. *)

open Base
module LL = Ir.Low_level
module Tn = Ir.Tnode
module Ops = Ir.Ops
module Idx = Ir.Indexing

let p name b = Stdio.printf "%s: %b\n" name b
let single = Ops.single
let iprec = Ops.index_prec ()
let next_id = ref 7000

let mk ?(dims = [| 16 |]) label =
  Int.incr next_id;
  Tn.create (Tn.Specified single) ~id:!next_id ~label:[ label ]
    ~unpadded_dims:(lazy dims)
    ~padding:(lazy None)
    ()

(* --- low-level builders --- *)
let sym () = Idx.get_symbol ()
let iter s = Idx.Iterator s
let embed idx : LL.scalar_t = LL.Embed_index idx
let ivar s = embed (iter s)
let ic i : LL.scalar_t = LL.Constant (Float.of_int i)

(* [Σ coeff*sym + offset] as an index expression. *)
let affine symbols offset = embed (Idx.Affine { symbols; offset })
let cmp op a b : LL.scalar_t = LL.Binop (op, (a, iprec), (b, iprec))
let lt = cmp Ops.Cmplt
let le = cmp Ops.Cmple
let eq = cmp Ops.Cmpeq
let ne = cmp Ops.Cmpne
let conj a b : LL.scalar_t = LL.Binop (Ops.And, (a, iprec), (b, iprec))
let disj a b : LL.scalar_t = LL.Binop (Ops.Or, (a, iprec), (b, iprec))

(* The zero-fringe shape: [cond ? src[i] : 0]. *)
let where_zero cond src s : LL.scalar_t =
  LL.Ternop
    (Ops.Where, (cond, iprec), (LL.Get (src, [| iter s |]), single), (LL.Constant 0., single))

let set dst s llsc : LL.t = LL.Set { tn = dst; idcs = [| iter s |]; llsc; debug = "" }
let loop ?(from_ = 0) ~to_ s body : LL.t = LL.For_loop { index = s; from_; to_; body; axis = Serial }
let if_ cond body : LL.t = LL.If { cond = (cond, iprec); body }

(* --- structural probes on the simplified form --- *)
let rec count ~f_stmt ~f_scalar (llc : LL.t) : int =
  (if f_stmt llc then 1 else 0)
  +
  match llc with
  | LL.Seq (a, b) -> count ~f_stmt ~f_scalar a + count ~f_stmt ~f_scalar b
  | LL.For_loop { body; _ } | LL.If { body; _ } -> count ~f_stmt ~f_scalar body
  | LL.Set { llsc; _ } | LL.Set_local (_, llsc) -> count_s ~f_stmt ~f_scalar llsc
  | _ -> 0

and count_s ~f_stmt ~f_scalar (llsc : LL.scalar_t) : int =
  (if f_scalar llsc then 1 else 0)
  +
  let loop = count_s ~f_stmt ~f_scalar in
  match llsc with
  | LL.Ternop (_, (a, _), (b, _), (c, _)) -> loop a + loop b + loop c
  | LL.Binop (_, (a, _), (b, _)) -> loop a + loop b
  | LL.Unop (_, (a, _)) -> loop a
  | LL.Local_scope { body; _ } -> count ~f_stmt ~f_scalar body
  | _ -> 0

let never _ = false
let wheres llc = count llc ~f_stmt:never ~f_scalar:(function LL.Ternop (Ops.Where, _, _, _) -> true | _ -> false)
let ifs llc = count llc ~f_stmt:(function LL.If _ -> true | _ -> false) ~f_scalar:never

let zeros llc =
  count llc ~f_stmt:never ~f_scalar:(function LL.Constant 0. -> true | _ -> false)

let simplify llc = LL.simplify_llc [] llc

(* The motivating shape, parameterized by the guard wrapping the inner nest: two k-blocks of 16 over
   an extent of 32, the prefetch's index shifted a block ahead ([i + 16*k + 16]), edge-guarded
   against the extent. Under [k <= 0] the index stays in [16, 31] and the guard folds; over the full
   loop range it reaches 47 and must not. *)
let block_case ~guard_of =
  let dst = mk "sin_dst" and src = mk ~dims:[| 32 |] "sin_src" in
  let k = sym () and i = sym () in
  let idx = affine [ (1, i); (16, k) ] 16 in
  let body = loop ~to_:15 i (set dst i (where_zero (lt idx (ic 32)) src i)) in
  loop ~to_:1 k (match guard_of k with None -> body | Some cond -> if_ cond body)

let () =
  (* --- Cmplt: the pipelined prefetch's own shape. --- *)
  let folded = simplify (block_case ~guard_of:(fun k -> Some (lt (ivar k) (ic 1)))) in
  let unguarded = simplify (block_case ~guard_of:(fun _ -> None)) in
  let weak = simplify (block_case ~guard_of:(fun k -> Some (lt (ivar k) (ic 2)))) in
  p "Cmplt guard folds the edge guard it proves" (wheres folded = 0 && zeros folded = 0);
  p "without the guard the same edge guard survives" (wheres unguarded = 1);
  p "a guard weaker than the loop range narrows nothing" (wheres weak = 1);

  (* --- Cmple. --- *)
  let folded_le = simplify (block_case ~guard_of:(fun k -> Some (le (ivar k) (ic 0)))) in
  let weak_le = simplify (block_case ~guard_of:(fun k -> Some (le (ivar k) (ic 1)))) in
  p "Cmple guard folds the edge guard it proves" (wheres folded_le = 0);
  p "a Cmple guard at the loop's top narrows nothing" (wheres weak_le = 1);

  (* --- Cmpeq narrows both ways; Cmpne (its negation) is not a narrowing rule. --- *)
  let folded_eq = simplify (block_case ~guard_of:(fun k -> Some (eq (ivar k) (ic 0)))) in
  let unfolded_ne = simplify (block_case ~guard_of:(fun k -> Some (ne (ivar k) (ic 0)))) in
  p "Cmpeq guard folds the edge guard it proves" (wheres folded_eq = 0);
  p "Cmpne guard narrows nothing" (wheres unfolded_ne = 1);

  (* --- Conjunction: both conjuncts contribute, and neither alone suffices. The mixed-radix index
     [4*k + m] over [0, 3] x [0, 3] reaches 15; narrowing only [k] leaves 7, narrowing both leaves
     5, so the guard [< 6] discriminates the [And] rule from either half. A disjunction implies
     neither and must narrow nothing. --- *)
  let mixed_case ~guard_of =
    let dst = mk "sin_dst2" and src = mk ~dims:[| 16 |] "sin_src2" in
    let k = sym () and m = sym () in
    let idx = affine [ (4, k); (1, m) ] 0 in
    loop ~to_:3 k
      (loop ~to_:3 m (if_ (guard_of k m) (set dst m (where_zero (lt idx (ic 6)) src m))))
  in
  let both = simplify (mixed_case ~guard_of:(fun k m -> conj (lt (ivar k) (ic 2)) (lt (ivar m) (ic 2)))) in
  let one = simplify (mixed_case ~guard_of:(fun k _ -> lt (ivar k) (ic 2))) in
  let either = simplify (mixed_case ~guard_of:(fun k m -> disj (lt (ivar k) (ic 2)) (lt (ivar m) (ic 2)))) in
  p "And guard narrows both conjuncts' symbols" (wheres both = 0);
  p "only one conjunct's narrowing does not decide it" (wheres one = 1);
  p "Or guard narrows nothing" (wheres either = 1);

  (* --- The reversed operand order narrows the symbol's LOWER bound (a negative coefficient in the
     difference), and the decided guard folds to its else-branch. --- *)
  let low_case ~guarded =
    let dst = mk "sin_dst3" and src = mk ~dims:[| 16 |] "sin_src3" in
    let k = sym () in
    let body = set dst k (where_zero (lt (ivar k) (ic 6)) src k) in
    loop ~to_:9 k (if guarded then if_ (lt (ic 5) (ivar k)) body else body)
  in
  let low = simplify (low_case ~guarded:true) in
  let low_ref = simplify (low_case ~guarded:false) in
  p "a lower-bound guard folds a refuted edge guard to its else-branch"
    (wheres low = 0 && zeros low = 1);
  p "without it the same edge guard survives" (wheres low_ref = 1);

  (* --- An inner guard the outer one implies is erased entirely (statement-level). --- *)
  let nested =
    let dst = mk "sin_dst4" and src = mk ~dims:[| 16 |] "sin_src4" in
    let k = sym () in
    simplify
      (loop ~to_:9 k
         (if_ (lt (ivar k) (ic 3))
            (if_ (lt (ivar k) (ic 5)) (set dst k (LL.Get (src, [| iter k |]))))))
  in
  p "an inner guard implied by the outer one is erased" (ifs nested = 1);

  (* --- Negative controls on soundness: a condition that is not an integer-affine index comparison
     contributes no facts, and a guard that is unsatisfiable in conjunction (a dead body) is left
     alone rather than fabricating an empty interval. --- *)
  let data_dep =
    let dst = mk "sin_dst5" and src = mk ~dims:[| 32 |] "sin_src5" and sel = mk "sin_sel" in
    let k = sym () and i = sym () in
    let idx = affine [ (1, i); (16, k) ] 16 in
    simplify
      (loop ~to_:1 k
         (if_
            (LL.Binop (Ops.Cmplt, (LL.Get (sel, [| Idx.Fixed_idx 0 |]), single), (LL.Constant 1., single)))
            (loop ~to_:15 i (set dst i (where_zero (lt idx (ic 32)) src i)))))
  in
  p "a data-dependent guard narrows nothing" (wheres data_dep = 1 && ifs data_dep = 1);
  let dead =
    let dst = mk "sin_dst6" and src = mk ~dims:[| 16 |] "sin_src6" in
    let k = sym () in
    simplify
      (loop ~to_:1 k
         (if_
            (conj (lt (ivar k) (ic 1)) (lt (ic 0) (ivar k)))
            (set dst k (LL.Get (src, [| iter k |])))))
  in
  p "an unsatisfiable conjunction is left alone" (ifs dead = 1);

  (* --- Evaluation-precision soundness: past a float precision's exact-integer range the machine
     comparison is not the mathematical one -- single represents [2^24 + 1] as [2^24], so a
     single-precision guard [k <= 2^24] is machine-true at [k = 2^24 + 1] too, and narrowing from
     it would fold the (index-precision) edge guard to the wrong branch there. The identical guard
     evaluated at the index precision is exact and narrows. --- *)
  let lim = 16777216 (* 2^24, [Interval.float_exact_int_limit] of single *) in
  let prec_case ~cond_prec =
    let dst = mk "sin_dst7" and src = mk "sin_src7" in
    let k = sym () in
    simplify
      (loop ~to_:(lim + 1) k
         (LL.If
            {
              cond = (LL.Binop (Ops.Cmple, (ivar k, cond_prec), (ic lim, cond_prec)), cond_prec);
              body = set dst k (where_zero (le (ivar k) (ic lim)) src k);
            }))
  in
  p "a float-precision guard past its exact-integer range narrows nothing"
    (wheres (prec_case ~cond_prec:single) = 1);
  p "the same guard at the index precision narrows" (wheres (prec_case ~cond_prec:iprec) = 0)
