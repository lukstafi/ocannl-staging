(* gh-573: the transitive inline-fanin guard. A running sum (a transformer's residual stream) has
   per-cell read multiplicity within the visit cap — its consumers' copy-position reads are
   read-modify-write-exempt — and no reduction loops, so neither per-node cap
   ([virtualize_max_visits], [virtualize_max_inline_reduction]) ever materializes it; yet inlining
   it replays the entire prefix of the chain at every consumer, quadratic in depth. The guard caps
   the fan-in of the fully-inlined computation (the number of distinct materialized nodes it
   loads, accumulated through chains of virtual producers): the first chain node whose fan-in
   exceeds [virtualize_max_inline_fanin] is materialized (provenance 41), which resets the fan-in
   of everything downstream.

   Phase 1 pins the decision structurally on a hand-built [Ir.Low_level.t] chain through the
   [Ll_test] harness: with the default cap 8, a 10-link add chain materializes exactly the link
   whose fan-in first reaches 9 (x8), the links before and after stay virtual, and the
   materialization is reported as an [`Inline] flip whose recompute cost carries the fan-in
   (extent 1 × multiplicity 1 × fan-in 9 — so the memory-budget planner does not rank the node
   among the cheapest to re-inline). Disabling the cap reproduces the old behavior (whole chain
   virtual). Both readings execute and must agree cell for cell with the OCaml reference
   (gh-ocannl-589: placement decisions need an executed leg, not just structural pins).

   Phase 1b pins that reads inside a [Local_scope] body in a setter's right-hand side count toward
   that setter's fan-in (review round 1): a producer computed through a scope body loading 9
   materialized nodes materializes under the cap.

   Phase 2 exercises the real [Assignments] pipeline on a tensor-graph chain built with
   [TDSL.O.( + )], asserting the same placement pattern through [Context.placements] and value
   parity between the default and cap-disabled compiles.

   Phase 3 pins the cross-routine reading (review round 1): the same chain split across two
   [LL.optimize] calls sharing one [optimize_ctx] — the second routine reads a node the first
   committed [Virtual], whose fan-in is derived from the stored computation rather than from
   (absent) local setters, so the chain cannot escape the cap by being compiled in pieces.
   Structural pins only for now: the executed leg is blocked on gh-ocannl-610 (kernel parameters
   miss leaves that reach a routine only through a cross-routine inlined computation). *)

open Base
open Ll_test
module LL = Ir.Low_level
module Tn = Ir.Tnode

let n_links = 10
let dim = 4

(* === The chain, shared by phases 1 and 3 ===

   x0, w1..w10 materialized; x_k = x_{k-1} + w_k; out = x_10. Fan-in of x_k is k+1
   ({x0, w1..wk}); the default cap 8 trips first at x8 (fan-in 9). *)

type chain = {
  x0 : Tn.t;
  ws : Tn.t array;
  xs : Tn.t array;
  out : Tn.t;
  links : LL.t list;
  consumer : LL.t;
  mk : string -> Tn.t;
}

let build_chain ~first_id () =
  let mk = node_factory ~first_id ~dims:[| dim |] () in
  let x0 = mk "x0" in
  materialize x0;
  let ws =
    Array.init n_links ~f:(fun k ->
        let w = mk (Printf.sprintf "w%d" (k + 1)) in
        materialize w;
        w)
  in
  let xs = Array.init n_links ~f:(fun k -> mk (Printf.sprintf "x%d" (k + 1))) in
  let out = mk "out" in
  materialize out;
  let link k =
    let prev = if k = 0 then x0 else xs.(k - 1) in
    let s = sym () in
    loop_n s dim (set xs.(k) [| iter s |] (add (get prev [| iter s |]) (get ws.(k) [| iter s |])))
  in
  let links = List.init n_links ~f:link in
  let consumer =
    let s = sym () in
    loop_n s dim (set out [| iter s |] (get xs.(n_links - 1) [| iter s |]))
  in
  { x0; ws; xs; out; links; consumer; mk }

let x0_vals = Array.init dim ~f:(fun i -> Float.of_int (i + 1))
let w_vals k = Array.init dim ~f:(fun i -> Float.of_int ((100 * (k + 1)) + (10 * (i + 1))))

let expected_out =
  Array.init dim ~f:(fun i ->
      Array.foldi (Array.init n_links ~f:w_vals) ~init:x0_vals.(i) ~f:(fun _ acc w -> acc +. w.(i)))

let chain_seed { x0; ws; out; _ } =
  ((x0, x0_vals) :: List.init n_links ~f:(fun k -> (ws.(k), w_vals k))) @ [ (out, blank dim) ]

let find_flip (o : LL.optimized) tn =
  List.find_map o.LL.flip_candidates ~f:(fun fc ->
      if Tn.equal fc.LL.fc_tn tn then Some (fc.LL.fc_flip, fc.LL.fc_recompute_cost) else None)

(* === Phase 1: hand-built chain, one routine === *)

let phase1 () =
  assert (LL.virtualize_settings.max_inline_fanin = 8);
  let c = build_chain ~first_id:3000 () in
  let llc = List.reduce_exn ~f:seq (c.links @ [ c.consumer ]) in
  let o = optimize ~name:"vcf_chain" llc in
  p "chain: x7 stays virtual (fan-in 8, at the cap)" (known_virtual o c.xs.(6));
  p "chain: x8 materialized by the fan-in cap" (known_non_virtual o c.xs.(7));
  p "chain: x9 and x10 virtual again past the reset"
    (known_virtual o c.xs.(8) && known_virtual o c.xs.(9));
  p "chain: x8 written once, read as a buffer downstream"
    (count_set o c.xs.(7) = 1 && count_get o c.xs.(7) >= 1);
  p "chain: x8 is an Inline flip charged its fan-in (cost 9)"
    (match find_flip o c.xs.(7) with Some (`Inline, 9) -> true | _ -> false);
  (* Executed parity: the guard is a placement decision, so both readings of the same program must
     produce the same cells. Discriminating producer values: vary with the link and the cell, off
     the zero-init and the sentinel. *)
  let seed = chain_seed c and read = [ c.out ] in
  let got = execute ~name:"vcf_chain" o ~seed ~read in
  p "chain: executed values match the reference" (same got [ expected_out ]);
  LL.virtualize_settings.max_inline_fanin <- -1;
  let o_off = optimize ~name:"vcf_chain_off" llc in
  LL.virtualize_settings.max_inline_fanin <- 8;
  p "cap disabled: whole chain virtual (the pre-fix behavior)"
    (Array.for_all c.xs ~f:(known_virtual o_off));
  let got_off = execute ~name:"vcf_chain_off" o_off ~seed ~read in
  p "cap disabled: executed values agree with the capped arm" (same got got_off)

(* === Phase 1b: scope-body reads count toward the enclosing setter's fan-in === *)

let phase1b () =
  let mk = node_factory ~first_id:3100 ~dims:[| dim |] () in
  let ws =
    Array.init 9 ~f:(fun k ->
        let w = mk (Printf.sprintf "sw%d" (k + 1)) in
        materialize w;
        w)
  in
  let y = mk "y" and out = mk "sout" and lv = mk "lv" in
  materialize out;
  virtualize lv;
  let i = sym () and j = sym () in
  let id = LL.get_scope lv in
  (* y[i] = Local_scope{ lv := w1[i] + ... + w9[i] }: nine loads, all inside the scope body. *)
  let body =
    LL.Set_local
      ( id,
        Array.fold ws ~init:(c 0.) ~f:(fun acc w -> add acc (get w [| iter i |])) )
  in
  let producer =
    loop_n i dim
      (set y [| iter i |] (LL.Local_scope { id; body; orig_indices = [| iter i |] }))
  in
  let consumer = loop_n j dim (set out [| iter j |] (get y [| iter j |])) in
  let o = optimize ~name:"vcf_scope" (seq producer consumer) in
  p "scope: producer materialized (scope-body loads count, fan-in 9)" (known_non_virtual o y);
  let seed = (out, blank dim) :: (Array.to_list ws |> List.mapi ~f:(fun k w -> (w, w_vals k))) in
  let expected =
    Array.init dim ~f:(fun i ->
        Array.foldi (Array.init 9 ~f:w_vals) ~init:0. ~f:(fun _ acc w -> acc +. w.(i)))
  in
  let got = execute ~name:"vcf_scope" o ~seed ~read:[ out ] in
  p "scope: executed values match the reference" (same got [ expected ])

(* === Phase 2: the Assignments pipeline === *)

open Ocannl
open Ocannl.Operation.DSL_modules

let phase2 () =
  let build () =
    Utils.settings.fixed_state_for_init <- Some 42;
    Tensor.unsafe_reinitialize ();
    let init l k =
      NTDSL.init ~l ~prec:Ir.Ops.single ~o:[ dim ]
        ~f:(fun idcs -> Float.of_int ((100 * k) + idcs.(0) + 1))
        ()
    in
    let x0 = init "vcf_x0" 0 in
    let xs =
      List.folding_map
        (List.init n_links ~f:(fun k -> k + 1))
        ~init:x0
        ~f:(fun x k ->
          let x' = TDSL.O.( + ) x (init (Printf.sprintf "vcf_w%d" k) k) in
          (x', x'))
    in
    let last = List.last_exn xs in
    let%op loss = last ++ "i=>0" in
    Train.set_materialized loss.Tensor.value;
    Tn.set_observable loss.Tensor.value;
    (xs, loss)
  in
  let run ~cap =
    LL.virtualize_settings.max_inline_fanin <- cap;
    let xs, loss = build () in
    let ctx = Context.auto () in
    let ctx, routine = Context.compile ctx (Train.forward loss) Ir.Indexing.Empty in
    let ctx = Context.run ctx routine in
    LL.virtualize_settings.max_inline_fanin <- 8;
    (Context.get_values ctx loss.Tensor.value, Context.placements ctx, xs)
  in
  let lv_default, plc_default, xs_default = run ~cap:8 in
  let lv_off, plc_off, xs_off = run ~cap:(-1) in
  let value k xs = (List.nth_exn xs (k - 1)).Tensor.value in
  p "pipeline: x7 stays virtual" (Tn.Placements.known_virtual plc_default (value 7 xs_default));
  p "pipeline: x8 materialized by the fan-in cap"
    (Tn.Placements.known_non_virtual plc_default (value 8 xs_default));
  p "pipeline: x9 and x10 virtual again past the reset"
    (Tn.Placements.known_virtual plc_default (value 9 xs_default)
    && Tn.Placements.known_virtual plc_default (value 10 xs_default));
  p "pipeline, cap disabled: whole chain virtual"
    (List.for_all xs_off ~f:(fun x -> Tn.Placements.known_virtual plc_off x.Tensor.value));
  let expected =
    Array.init dim ~f:(fun i ->
        List.fold
          (List.init n_links ~f:(fun k -> k + 1))
          ~init:(Float.of_int (i + 1))
          ~f:(fun acc k -> acc +. Float.of_int ((100 * k) + i + 1)))
    |> Array.fold ~init:0. ~f:( +. )
  in
  p "pipeline: loss matches the reference"
    (Array.length lv_default = 1 && Float.(abs (lv_default.(0) -. expected) <= 1e-3));
  p "pipeline: default and cap-disabled arms agree"
    (Array.length lv_off = 1 && Float.(abs (lv_default.(0) -. lv_off.(0)) <= 1e-3))

(* === Phase 3: the chain split across two routines in one lineage === *)

let phase3 () =
  let c = build_chain ~first_id:3200 () in
  let split = 5 in
  (* Routine A needs a materialized output of its own or cleanup eliminates it wholesale (a
     fully-virtual routine has nothing to keep); a copy-out of x5 mirrors a real routine's
     observable root, and its same-position read leaves x5 virtual. *)
  let out_a = c.mk "out_a" in
  materialize out_a;
  let copy_out =
    let s = sym () in
    loop_n s dim (set out_a [| iter s |] (get c.xs.(split - 1) [| iter s |]))
  in
  let llc_a = List.reduce_exn ~f:seq (List.take c.links split @ [ copy_out ]) in
  let llc_b = List.reduce_exn ~f:seq (List.drop c.links split @ [ c.consumer ]) in
  let ctx = LL.empty_optimize_ctx () in
  let o_a = LL.optimize ctx ~unoptim_ll_source:None ~ll_source:None ~name:"vcf_xchain_a" [] llc_a in
  p "cross-routine: routine A leaves x1..x5 virtual"
    (Array.for_alli c.xs ~f:(fun k tn -> k >= split || known_virtual o_a tn));
  let o_b = LL.optimize ctx ~unoptim_ll_source:None ~ll_source:None ~name:"vcf_xchain_b" [] llc_b in
  (* Routine B never sets x5, yet x5's stored computation is replayed when inlined: its fan-in
     (6: {x0, w1..w5}) is derived from the computation, so x6 sees 7, x7 sees 8, and x8 trips the
     cap exactly as in the single-routine reading. *)
  p "cross-routine: x7 stays virtual" (known_virtual o_b c.xs.(6));
  p "cross-routine: x8 materialized by the fan-in cap" (known_non_virtual o_b c.xs.(7));
  p "cross-routine: x9 and x10 virtual"
    (known_virtual o_b c.xs.(8) && known_virtual o_b c.xs.(9))
(* No executed leg for this phase yet: routine B's kernel parameters miss the leaves reaching it
   only through the inlined cross-routine computation ([input_and_output_nodes] folds over the
   traced store, which is built from the raw code) — a pre-existing gap, gh-ocannl-610.
   Reinstating [execute ~name:"vcf_xchain_b" o_b ~seed:(chain_seed c) ~read:[ c.out ]] against
   [expected_out] is that issue's acceptance test. *)

let () =
  phase1 ();
  phase1b ();
  phase2 ();
  phase3 ()
