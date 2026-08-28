open Base
open Ocannl.Operation.DSL_modules

(* [Tensor.op] can only learn an operation's neutral element by reading it off the assignments
   [op_asn] built, so it sets the update step's [neutral_elem] afterwards. The field is consumed
   later still, when the projections are derived. The whole scheme rests on nothing deriving them in
   between: an [op_asn] that does gets them derived against [neutral_elem = None], and where that
   changes a decision the padding and guard choices in [Shape.derive_projections] come out wrong --
   a wrong value, not a crash. [Tensor.op] rejects that rather than assuming nobody does it.

   The rejection is deliberately on the invariant and not on whether breaking it happened to matter
   for the operation at hand, so the legs below span the cases a narrower rule would let through --
   no neutral element to install, a finite one over a pointwise step, a non-finite one where no
   padded window consults the clamping flag, a finite one whose margins are touched. All are
   rejected, and this test is what a future narrowing has to argue with: the reasoning is at the
   guard in tensor.ml, and it turns on the cost of the call being non-local (forcing the projections
   runs [Shape.finish_inference] over every active update step in the session) rather than on any
   of these cases being individually harmful.

   The control leg is what keeps that honest: the same padded operation, built without the early
   derivation, constructs fine. *)

let%cd summing_op_asn ~t ~t1 ~projections = v =:+ relu v1
let%cd maxing_op_asn ~t ~t1 ~projections = v =:@^ relu v1
let%cd grad_asn ~t:_ ~g ~t1 ~projections = g1 =+ relu_gate (v1, g)
let%cd conv_op_asn ~t ~t1 ~t2 ~projections = v =:+ v1 * v2
let%cd conv_grad_asn ~t:_ ~g ~t1 ~t2 ~projections = g1 =+ g * v2

let force ~projections =
  ignore (Lazy.force projections.Tensor.projections : Ir.Indexing.projections)

(* No [Accum_op] leaf, so [collect_neutral_elem] yields [None] and the mutation is a no-op. *)
let no_neutral_op_asn ~t:_ ~t1:_ ~projections =
  force ~projections;
  Ir.Assignments.empty_comp

(* A finite neutral (0.) over a pointwise step: [clamp_padded] is [false] under both [None] and
   [Some 0.], and there are no margins to commit. *)
let finite_neutral_op_asn ~t ~t1 ~projections =
  force ~projections;
  summing_op_asn ~t ~t1 ~projections

(* A non-finite neutral ([Ops.neutral_elem Max] is [neg_infinity]) but still a pointwise step, so no
   padded window reaches the place [Row.solve_proj_equations] consults the clamping flag. *)
let nonfinite_neutral_op_asn ~t ~t1 ~projections =
  force ~projections;
  maxing_op_asn ~t ~t1 ~projections

(* A padded ([=]-mode) window, whose margin demand is what [update_padding_elem] would commit. *)
let margin_touching_op_asn ~t ~t1 ~t2 ~projections =
  force ~projections;
  conv_op_asn ~t ~t1 ~t2 ~projections

let vec n = TDSL.range_of_shape ~batch_dims:[] ~input_dims:[] ~output_dims:[ n ] ()

let unop ~op_asn =
  Tensor.unop ~op_label:"guard_probe" ~op_asn ~grad_asn ~grad_spec:Tensor.Prohibit_grad

let padded_conv ~op_asn =
  Tensor.binop ~op_label:"guard_probe"
    ~compose_op:(Shape.Einsum ("1*oh= + kh; kh => oh", []))
    ~op_asn ~grad_asn:conv_grad_asn ~grad_spec:Tensor.Prohibit_grad (vec 8) (vec 3) ()

let verdict ~claim ~expect_rejected build =
  match build () with
  | (_ : Tensor.t) ->
      if expect_rejected then Stdio.prerr_endline "(not part of the golden) unexpectedly accepted";
      Verdict.p claim (not expect_rejected)
  | exception Tensor.Session_error (msg, _) ->
      Stdio.prerr_endline ("(not part of the golden) Session_error: " ^ msg);
      (* Pin THIS guard, not any construction failure: an unrelated [Session_error] from [Tensor.op]
         would otherwise keep a rejection leg green with the guard removed. *)
      Verdict.p claim
        (expect_rejected
        && String.is_substring msg ~substring:"derived before its neutral element was set")

let () =
  (* The accepted control first: a rejected leg leaves behind the shapes its derivation mutated. *)
  verdict ~claim:"the padded operation itself constructs when nothing derives early"
    ~expect_rejected:false (fun () -> padded_conv ~op_asn:conv_op_asn);
  verdict ~claim:"an early derivation is rejected when there is no neutral element to install"
    ~expect_rejected:true (fun () -> unop ~op_asn:no_neutral_op_asn (vec 4) ());
  verdict ~claim:"an early derivation is rejected for a finite pointwise neutral" ~expect_rejected:true
    (fun () -> unop ~op_asn:finite_neutral_op_asn (vec 4) ());
  verdict ~claim:"an early derivation is rejected for a non-finite neutral without a padded window"
    ~expect_rejected:true (fun () -> unop ~op_asn:nonfinite_neutral_op_asn (vec 4) ());
  verdict ~claim:"an early derivation is rejected for a padded window that reads margins"
    ~expect_rejected:true (fun () -> padded_conv ~op_asn:margin_touching_op_asn)
