(* gh-ocannl-514 phase 3: [Ir.Cost_model.completion_floor] — the dual (lower-bound) extraction
   over hand-built programs where every number is checkable by hand (single precision, 4
   bytes/cell). Each case also asserts the duality invariant on the same code: the floor never
   exceeds the upper extraction ([analyze]).

   - pointwise map: all accesses exact, floor = upper on both legs;
   - matmul with an rmw accumulator: exact; opening the accumulator's placement drops exactly its
     traffic, and [node_floor_bytes] is the Materialize refinement delta that adds it back;
   - guarded write: the floor counts guards-never-taken (condition ops only, no write traffic)
     where the upper counts guards-taken — [fr_exact] false;
   - two reads of one node: the floor takes the larger exact image (a union is at least its
     largest member) where the upper takes the capped sum;
   - dynamic gather: the uninterpretable access floors to zero where the upper falls back to the
     whole node. *)

open Base
open Ocannl.Operation.DSL_modules
module LL = Ir.Low_level
module Idx = Ir.Indexing
module Tn = Ir.Tnode
module Ops = Ir.Ops
module CM = Ir.Cost_model

let fresh_tn =
  let c = ref 980_000_000 in
  fun label dims ->
    Int.incr c;
    Tn.create (Tn.Specified Ops.single) ~id:!c ~label:[ label ]
      ~unpadded_dims:(lazy dims)
      ~padding:(lazy None)
      ()

let sp = Ops.single

let for_over ?(extent = 4) sym body =
  LL.For_loop { index = sym; from_ = 0; to_ = extent - 1; body; axis = LL.Serial }

let get tn idcs = LL.Get (tn, idcs)
let it s = Idx.Iterator s

let show name ?open_placement code =
  let s = CM.analyze code in
  let f = CM.completion_floor ?open_placement code in
  Stdio.printf "== %s ==\n  floor: flops=%d bytes=%d %s\n  upper: flops=%d bytes=%d\n  floor <= upper: %b\n"
    name f.CM.fr_flops f.CM.fr_bytes
    (if f.CM.fr_exact then "exact" else "inexact")
    s.CM.flops (CM.total_bytes s)
    (f.CM.fr_flops <= s.CM.flops && f.CM.fr_bytes <= CM.total_bytes s);
  f

let () =
  let i = Idx.get_symbol () and j = Idx.get_symbol () and k = Idx.get_symbol () in
  (* Pointwise 4x5 map: C[i][j] = A[i][j] + B[j]. All accesses exact: floor = upper.
     A rd 80 B, B rd 20 B, C wr 80 B; 20 adds. *)
  let a = fresh_tn "A" [| 4; 5 |] in
  let b = fresh_tn "B" [| 5 |] in
  let c = fresh_tn "C" [| 4; 5 |] in
  let pointwise =
    for_over i
      (for_over ~extent:5 j
         (LL.Set
            {
              tn = c;
              idcs = [| it i; it j |];
              llsc = LL.Binop (Ops.Add, (get a [| it i; it j |], sp), (get b [| it j |], sp));
              debug = "";
            }))
  in
  let _ = show "pointwise map" pointwise in
  (* 4x5x6 matmul, rmw accumulator: D[i][j] += A2[i][k] * B2[k][j]. Exact both ways: flops =
     2*4*5*6 = 240; bytes = A2 96 + B2 120 + D rd 80 + D wr 80 = 376. Opening D's placement
     drops its 160 bytes; node_floor_bytes D restores them — the Materialize delta. *)
  let a2 = fresh_tn "A2" [| 4; 6 |] in
  let b2 = fresh_tn "B2" [| 6; 5 |] in
  let d = fresh_tn "D" [| 4; 5 |] in
  let matmul =
    for_over i
      (for_over ~extent:5 j
         (for_over ~extent:6 k
            (LL.Set
               {
                 tn = d;
                 idcs = [| it i; it j |];
                 llsc =
                   LL.Binop
                     ( Ops.Add,
                       (get d [| it i; it j |], sp),
                       ( LL.Binop
                           (Ops.Mul, (get a2 [| it i; it k |], sp), (get b2 [| it k; it j |], sp)),
                         sp ) );
                 debug = "";
               })))
  in
  let closed = show "matmul (rmw accumulator)" matmul in
  let opened =
    show "matmul, D's placement open" ~open_placement:(fun tn -> Tn.equal tn d) matmul
  in
  Stdio.printf "  open + Materialize delta restores the closed floor: %b\n"
    (opened.CM.fr_bytes + CM.node_floor_bytes matmul d = closed.CM.fr_bytes);
  (* Guarded write: if (E[i] > 0) F[i] = E[i] * 2. Upper counts guards-taken (cmp + mul per
     iteration, F written); the floor counts only the certain condition ops and no F traffic. *)
  let e = fresh_tn "E" [| 4 |] in
  let fq = fresh_tn "F" [| 4 |] in
  let guarded =
    for_over i
      (LL.If
         {
           cond = (LL.Binop (Ops.Cmplt, (LL.Constant 0., sp), (get e [| it i |], sp)), sp);
           body =
             LL.Set
               {
                 tn = fq;
                 idcs = [| it i |];
                 llsc = LL.Binop (Ops.Mul, (get e [| it i |], sp), (LL.Constant 2., sp));
                 debug = "";
               };
         })
  in
  let _ = show "guarded write" guarded in
  (* Two exact reads of one node with disjoint images: G[i][j] = H[0][j] + H[1][j]. The upper
     sums the images (10 cells = 40 rd bytes, a union bound); the floor takes only the larger
     exact image (5 cells = 20 bytes) — a union is at least its largest member, no more is
     certain. G writes 80 bytes both ways: floor 100 vs upper 120. *)
  let g = fresh_tn "G" [| 4; 5 |] in
  let h = fresh_tn "H" [| 4; 5 |] in
  let two_reads =
    for_over i
      (for_over ~extent:5 j
         (LL.Set
            {
              tn = g;
              idcs = [| it i; it j |];
              llsc =
                LL.Binop
                  ( Ops.Add,
                    (get h [| Idx.Fixed_idx 0; it j |], sp),
                    (get h [| Idx.Fixed_idx 1; it j |], sp) );
              debug = "";
            }))
  in
  let _ = show "two reads, union floor = larger image" two_reads in
  (* Dynamic gather: P[i] = Q[R[i]]. Q's access is uninterpretable — the upper falls back to the
     whole node, the floor to zero (inexact). *)
  let p = fresh_tn "P" [| 4 |] in
  let q = fresh_tn "Q" [| 8 |] in
  let r = fresh_tn "R" [| 4 |] in
  let gather =
    for_over i
      (LL.Set
         {
           tn = p;
           idcs = [| it i |];
           llsc =
             LL.Get_dynamic
               {
                 tn = q;
                 idcs = [| Idx.Fixed_idx 0 |];
                 dyn_axis = 0;
                 dyn_value = (get r [| it i |], sp);
               };
           debug = "";
         })
  in
  let _ = show "dynamic gather" gather in
  ()
