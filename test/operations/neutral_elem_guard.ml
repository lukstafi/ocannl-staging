open Base
open Ocannl.Operation.DSL_modules

(* [Tensor.op] can only learn an operation's neutral element by reading it off the assignments
   [op_asn] built, and [Shape.derive_projections] consumes it (the clamped-window decision and the
   margins' neutral). So the operation's shape update step, whose [neutral_elem] is immutable, is
   only created once [op_asn] has returned, and the [projections] handle [op_asn] receives is a
   promise for it. Forcing the handle from within [op_asn] is rejected before anything happens: the
   step is not registered, nothing is derived, no operand shape is touched.

   The rejection is on the invariant and not on whether the missing neutral element would have
   changed a decision, so the legs below span the cases a narrower rule would let through -- no
   neutral element at all, a finite one over a pointwise step, a non-finite one where no padded
   window consults the clamping flag, a finite one whose margins are touched. All are rejected, and
   this test is what a future narrowing has to argue with: the reasoning is at the seam in
   tensor.ml, and it turns on the cost of the call being non-local (forcing the projections runs
   [Shape.finish_inference] over every active update step in the session) rather than on any of
   these cases being individually harmful.

   The last two legs pin what rejecting early buys. The operand of the rejected padded-window leg
   carries no committed padding afterwards -- a derivation that had run against an unknown neutral
   element would have committed margins on it -- and the very same operands then serve the accepted
   control, whose derivation commits their padding with the operation's actual neutral. *)

let%cd summing_op_asn ~t ~t1 ~projections = v =:+ relu v1
let%cd maxing_op_asn ~t ~t1 ~projections = v =:@^ relu v1
let%cd grad_asn ~t:_ ~g ~t1 ~projections = g1 =+ relu_gate (v1, g)
let%cd conv_op_asn ~t ~t1 ~t2 ~projections = v =:+ v1 * v2
let%cd conv_grad_asn ~t:_ ~g ~t1 ~t2 ~projections = g1 =+ g * v2

let force ~projections =
  ignore (Lazy.force projections.Tensor.projections : Ir.Indexing.projections)

(* No [Accum_op] leaf, so [collect_neutral_elem] would yield [None]. *)
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

(* A padded ([=]-mode) window, whose margin demand is what [update_padding_elem] commits. *)
let margin_touching_op_asn ~t ~t1 ~t2 ~projections =
  force ~projections;
  conv_op_asn ~t ~t1 ~t2 ~projections

let vec n = TDSL.range_of_shape ~batch_dims:[] ~input_dims:[] ~output_dims:[ n ] ()

let unop ~op_asn =
  Tensor.unop ~op_label:"guard_probe" ~op_asn ~grad_asn ~grad_spec:Tensor.Prohibit_grad

let padded_conv ~op_asn a k =
  Tensor.binop ~op_label:"guard_probe"
    ~compose_op:(Shape.Einsum ("1*oh= + kh; kh => oh", []))
    ~op_asn ~grad_asn:conv_grad_asn ~grad_spec:Tensor.Prohibit_grad a k ()

let verdict ~claim ~expect_rejected build =
  match build () with
  | (_ : Tensor.t) ->
      if expect_rejected then Stdio.prerr_endline "(not part of the golden) unexpectedly accepted";
      Verdict.p claim (not expect_rejected)
  | exception Tensor.Session_error (msg, _) ->
      Stdio.prerr_endline ("(not part of the golden) Session_error: " ^ msg);
      (* Pin THIS rejection, not any construction failure: an unrelated [Session_error] from
         [Tensor.op] would otherwise keep a rejection leg green with the rejection removed. *)
      Verdict.p claim
        (expect_rejected && String.is_substring msg ~substring:"forced from within [op_asn]")

let () =
  verdict ~claim:"an early derivation is rejected when there is no neutral element to install"
    ~expect_rejected:true (fun () -> unop ~op_asn:no_neutral_op_asn (vec 4) ());
  verdict ~claim:"an early derivation is rejected for a finite pointwise neutral"
    ~expect_rejected:true (fun () -> unop ~op_asn:finite_neutral_op_asn (vec 4) ());
  verdict ~claim:"an early derivation is rejected for a non-finite neutral without a padded window"
    ~expect_rejected:true (fun () -> unop ~op_asn:nonfinite_neutral_op_asn (vec 4) ());
  let a = vec 8 and k = vec 3 in
  verdict ~claim:"an early derivation is rejected for a padded window that reads margins"
    ~expect_rejected:true (fun () -> padded_conv ~op_asn:margin_touching_op_asn a k);
  (* [Shape.to_padding] finalizes inference for every active step: had the rejected leg registered
     and derived its step, [a] would carry the window's margins here. *)
  Verdict.p "the operand of the rejected padded window carries no padding"
    (Option.is_none (Shape.to_padding a.Tensor.shape));
  verdict ~claim:"the same operands then construct the padded operation when nothing derives early"
    ~expect_rejected:false (fun () -> padded_conv ~op_asn:conv_op_asn a k);
  Verdict.p "the accepted operation commits the operand's margins with its neutral element, 0"
    (match Shape.to_padding a.Tensor.shape with
    | Some (_, padded_value) -> Float.equal padded_value 0.
    | None -> false)
