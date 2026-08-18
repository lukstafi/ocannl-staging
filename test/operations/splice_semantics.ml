(* gh-617 + gh-618: the splice-semantics design pair surfaced by review of the gh-610/611 fixes.

   Phase 1 (gh-618) pins the raw-side split of the read-modify-write exemption by consumer. A read
   at its enclosing statement's write position is exempt from the visit-counting placement
   heuristics ([rmw_exempt] — the statement's own store is not a visit, and virtual_chain_fanin
   relies on it for residual chains), but for a node that owns a buffer it is NOT interface
   coverage: in [ell[0] = 5000; out[i] = ell[i]] the copy-position reads consume ell's incoming
   cells 1.., and an accumulation with no preceding definite initialization consumes its own entry
   values. Both must classify [read_before_write] — a routine input, excluded from buffer
   aliasing — where pre-split they read as output-only. Two controls pin the boundaries: coverage
   by a real prior write still suppresses the flag (routine-complete lowered flows emit the
   initialization before accumulations, so they are unaffected), and an UNDECIDED node keeps its
   virtualization eligibility — a virtual node has no interface, and exemption-dependent coverage
   is exactly the shape of the virtualizer's partial-write producers (affine_lowering.ml AC6), so
   the strict verdict binds only once a node is known non-virtual. *)

open Base
open Ll_test
module LL = Ir.Low_level
module Tn = Ir.Tnode

let dim = 4

let inputs (o : LL.optimized) =
  let (ins, _), _ = LL.input_and_output_nodes o in
  ins

(* === Phase 1: gh-618 — the rmw exemption is not interface coverage (raw reads) === *)

let phase1 () =
  let mk = node_factory ~first_id:3600 ~dims:[| dim |] () in
  (* Copy-position reads of a DIFFERENT node after a partial write: the exemption matches on the
     index map alone, whichever nodes are involved, so pre-split ell read as covered and
     classified output-only — its incoming cells 1.. feed out yet the interface said the entry
     value was ignorable (aliasing-eligible, dropped from link-time input verification). *)
  let ell = mk "ell" in
  materialize ell;
  let out = mk "cpout" in
  materialize out;
  let llc_copy =
    let s = sym () in
    seq
      (set ell [| fixed 0 |] (c 5000.))
      (loop_n s dim (set out [| iter s |] (get ell [| iter s |])))
  in
  let o_copy = optimize ~name:"ssem_copypos" llc_copy in
  p "copy-position raw reads after a partial write: ell is read-before-write"
    (read_before_write o_copy ell);
  p "copy-position raw reads: ell is an input of the routine" (Set.mem (inputs o_copy) ell);
  let ell_old = Array.init dim ~f:(fun i -> 11. +. Float.of_int i) in
  let got =
    execute ~name:"ssem_copypos" o_copy ~seed:[ (ell, ell_old); (out, blank dim) ] ~read:[ out ]
  in
  p "copy-position raw reads: the uncovered cells read the incoming values"
    (same got [ Array.init dim ~f:(fun i -> if i = 0 then 5000. else ell_old.(i)) ]);
  (* True read-modify-write with no preceding initialization: x[i] = x[i] + a[i] consumes x's
     entry values, so x is an input — the tracer-mirroring exemption must not classify it
     output-only. *)
  let x = mk "x" in
  materialize x;
  let a = mk "a" in
  materialize a;
  let llc_rmw =
    let s = sym () in
    loop_n s dim (set x [| iter s |] (add (get x [| iter s |]) (get a [| iter s |])))
  in
  let o_rmw = optimize ~name:"ssem_rmw" llc_rmw in
  p "uninitialized accumulation: x is read-before-write" (read_before_write o_rmw x);
  p "uninitialized accumulation: x is an input of the routine" (Set.mem (inputs o_rmw) x);
  let x_old = Array.init dim ~f:(fun i -> 21. +. Float.of_int i) in
  let a_vals = Array.init dim ~f:(fun i -> 100. *. (1. +. Float.of_int i)) in
  let got_rmw =
    execute ~name:"ssem_rmw" o_rmw ~seed:[ (x, x_old); (a, a_vals) ] ~read:[ x ]
  in
  p "uninitialized accumulation: updated in place over the incoming values"
    (same got_rmw [ Array.init dim ~f:(fun i -> x_old.(i) +. a_vals.(i)) ]);
  (* Control: coverage by a real prior write still suppresses the flag — the strict interface
     verdict must not overreach onto routine-complete accumulations, whose lowering emits the
     initialization first. *)
  let y = mk "y" in
  materialize y;
  let b = mk "b" in
  materialize b;
  let llc_ctrl =
    let s = sym () in
    seq (zero y) (loop_n s dim (set y [| iter s |] (add (get y [| iter s |]) (get b [| iter s |]))))
  in
  let o_ctrl = optimize ~name:"ssem_zeroed" llc_ctrl in
  p "zero-initialized accumulation: covered by the real write, not read-before-write"
    (not (read_before_write o_ctrl y));
  p "zero-initialized accumulation: y is not an input" (not (Set.mem (inputs o_ctrl) y));
  let b_vals = Array.init dim ~f:(fun i -> 31. +. Float.of_int i) in
  let got_ctrl =
    execute ~name:"ssem_zeroed" o_ctrl ~seed:[ (y, blank dim); (b, b_vals) ] ~read:[ y ]
  in
  p "zero-initialized accumulation: the sentinel seed is overwritten" (same got_ctrl [ b_vals ]);
  (* Placement-freedom control: an UNDECIDED partial-write producer consumed at the copy position
     — the triangular-scatter genre of affine_lowering.ml's AC6 (injective, so no neutral init is
     emitted; off-region cells fall back to the init the INLINING prepends) — must keep its
     virtualization eligibility: a virtual node has no interface, so the strict verdict binds
     only once a node is known non-virtual. The materialized reading of this genre is ell above;
     AC6 executes both readings. *)
  let z = mk ~dims:[| 3; 4 |] "z" in
  let src = mk ~dims:[| 3; 2 |] "zsrc" in
  materialize src;
  let zout = mk ~dims:[| 3; 4 |] "zout" in
  materialize zout;
  let llc_und =
    let s1 = sym () and s2 = sym () and a = sym () and b = sym () in
    seq
      (loop_n s1 3
         (loop_n s2 2
            (set z [| iter s1; aff [ (1, s1); (1, s2) ] 0 |] (get src [| iter s1; iter s2 |]))))
      (loop_n a 3 (loop_n b 4 (set zout [| iter a; iter b |] (get z [| iter a; iter b |]))))
  in
  let o_und = optimize ~name:"ssem_undecided" llc_und in
  p "undecided partial-write producer: keeps its virtualization eligibility"
    (known_virtual o_und z)

let () = phase1 ()
