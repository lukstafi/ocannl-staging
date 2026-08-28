open Base
open Ocannl.Operation.DSL_modules

(* [Tensor.op] can only learn an operation's neutral element by reading it off the assignments
   [op_asn] built, so it sets the update step's [neutral_elem] afterwards. The field is consumed
   later still, when the projections are derived. An [op_asn] that derives them early gets them
   derived against [neutral_elem = None], which silently changes the padding and guard decisions in
   [Shape.derive_projections] -- a wrong value, not a crash. [Tensor.op] rejects that rather than
   assuming nobody does it.

   The rejection is by staleness, not by earliness, and both sides of that are pinned here: an
   early derivation whose operation turns out to have no neutral element after all was not misled
   and is accepted. *)

let%cd accumulating_op_asn ~t ~t1 ~projections = v =:+ relu v1
let%cd grad_asn ~t:_ ~g ~t1 ~projections = g1 =+ relu_gate (v1, g)

let force ~projections =
  ignore (Lazy.force projections.Tensor.projections : Ir.Indexing.projections)

(* Derives early AND has a neutral element to install: the derivation is stale. *)
let stale_op_asn ~t ~t1 ~projections =
  force ~projections;
  accumulating_op_asn ~t ~t1 ~projections

(* Derives early, but [collect_neutral_elem] finds no [Accum_op] leaf, so the value the derivation
   read is the value that would be installed and nothing is stale. *)
let harmless_op_asn ~t:_ ~t1:_ ~projections =
  force ~projections;
  Ir.Assignments.empty_comp

let unop ~op_asn x =
  Tensor.unop ~op_label:"guard_probe" ~op_asn ~grad_asn ~grad_spec:Tensor.Prohibit_grad x ()

let () =
  (* The accepted leg first: the rejected one leaves the shapes it mutated behind. *)
  let accepted =
    match unop ~op_asn:harmless_op_asn (TDSL.range 4) with
    | (_ : Tensor.t) -> true
    | exception Tensor.Session_error (msg, _) ->
        Stdio.prerr_endline ("(not part of the golden) unexpectedly rejected: " ^ msg);
        false
  in
  Verdict.p "an early derivation with no neutral element to install is accepted" accepted;
  let rejected =
    match unop ~op_asn:stale_op_asn (TDSL.range 4) with
    | (_ : Tensor.t) -> false
    | exception Tensor.Session_error (msg, _) ->
        Stdio.prerr_endline ("(not part of the golden) rejected with: " ^ msg);
        (* Pin THIS guard, not any construction failure: an unrelated [Session_error] from
           [Tensor.op] would otherwise keep the test green with the guard removed. *)
        String.is_substring msg ~substring:"derived before its neutral element was known"
  in
  Verdict.p "op_asn deriving the projections early with a stale neutral element is rejected"
    rejected
