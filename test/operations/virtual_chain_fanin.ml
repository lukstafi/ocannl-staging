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
   materialization is reported as an [`Inline] flip (searchable, like the other heuristic caps).
   Disabling the cap reproduces the old behavior (whole chain virtual). Both readings execute and
   must agree cell for cell with the OCaml reference (gh-ocannl-589: placement decisions need an
   executed leg, not just structural pins).

   Phase 2 exercises the real [Assignments] pipeline on a tensor-graph chain built with
   [TDSL.O.( + )], asserting the same placement pattern through [Context.placements] and value
   parity between the default and cap-disabled compiles. *)

open Base
open Ll_test
module LL = Ir.Low_level
module Tn = Ir.Tnode

let n_links = 10
let dim = 4

(* === Phase 1: hand-built chain === *)

let mk = node_factory ~first_id:3000 ~dims:[| dim |] ()

let phase1 () =
  assert (LL.virtualize_settings.max_inline_fanin = 8);
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
  let consumer =
    let s = sym () in
    loop_n s dim (set out [| iter s |] (get xs.(n_links - 1) [| iter s |]))
  in
  let llc = List.reduce_exn ~f:seq (List.init n_links ~f:link @ [ consumer ]) in
  let o = optimize ~name:"vcf_chain" llc in
  (* Fan-in of x_k is k+1 ({x0, w1..wk}); the cap 8 trips first at x8 (fan-in 9). *)
  p "chain: x7 stays virtual (fan-in 8, at the cap)" (known_virtual o xs.(6));
  p "chain: x8 materialized by the fan-in cap" (known_non_virtual o xs.(7));
  p "chain: x9 and x10 virtual again past the reset"
    (known_virtual o xs.(8) && known_virtual o xs.(9));
  p "chain: x8 written once, read as a buffer downstream"
    (count_set o xs.(7) = 1 && count_get o xs.(7) >= 1);
  p "chain: x8 reported as an Inline flip"
    (List.exists o.LL.flip_candidates ~f:(fun fc ->
         Tn.equal fc.LL.fc_tn xs.(7)
         && match fc.LL.fc_flip with `Inline -> true | `Materialize -> false));
  (* Executed parity: the guard is a placement decision, so both readings of the same program must
     produce the same cells. Discriminating producer values: vary with the link and the cell, off
     the zero-init and the sentinel. *)
  let x0_vals = Array.init dim ~f:(fun i -> Float.of_int (i + 1)) in
  let w_vals k = Array.init dim ~f:(fun i -> Float.of_int ((100 * (k + 1)) + (10 * (i + 1)))) in
  let expected =
    Array.init dim ~f:(fun i ->
        Array.foldi (Array.init n_links ~f:w_vals) ~init:x0_vals.(i) ~f:(fun _ acc w ->
            acc +. w.(i)))
  in
  let seed =
    ((x0, x0_vals) :: List.init n_links ~f:(fun k -> (ws.(k), w_vals k))) @ [ (out, blank dim) ]
  in
  let read = [ out ] in
  let got = execute ~name:"vcf_chain" o ~seed ~read in
  p "chain: executed values match the reference" (same got [ expected ]);
  LL.virtualize_settings.max_inline_fanin <- -1;
  let o_off = optimize ~name:"vcf_chain_off" llc in
  LL.virtualize_settings.max_inline_fanin <- 8;
  p "cap disabled: whole chain virtual (the pre-fix behavior)"
    (Array.for_all xs ~f:(known_virtual o_off));
  let got_off = execute ~name:"vcf_chain_off" o_off ~seed ~read in
  p "cap disabled: executed values agree with the capped arm" (same got got_off)

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

let () =
  phase1 ();
  phase2 ()
