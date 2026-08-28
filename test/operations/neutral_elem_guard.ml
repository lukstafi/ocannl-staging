open Base
open Ocannl.Operation.DSL_modules

(* [Tensor.op] can only learn an operation's neutral element by reading it off the assignments
   [op_asn] built, so it sets the update step's [neutral_elem] afterwards. The field is consumed
   later still, when the projections are derived. An [op_asn] that derives them early gets them
   derived against [neutral_elem = None]; where that changes a decision, the padding and guard
   choices in [Shape.derive_projections] come out wrong -- a wrong value, not a crash -- and
   [Tensor.op] rejects it rather than assuming nobody does it.

   The rejection is by staleness, not by earliness. [Shape.derivation_is_stale_for] enumerates the
   two decisions that read the field, and there is a leg here per outcome: the two that are stale
   (a non-finite neutral, which flips clamping; a finite one over a step that reads margins, which
   leaves the margin neutral uncommitted) and the two that are not (no neutral element to install;
   a finite one over an ordinary pointwise step, which derives identically either way). *)

let%cd summing_op_asn ~t ~t1 ~projections = v =:+ relu v1
let%cd maxing_op_asn ~t ~t1 ~projections = v =:@^ relu v1
let%cd grad_asn ~t:_ ~g ~t1 ~projections = g1 =+ relu_gate (v1, g)
let%cd conv_op_asn ~t ~t1 ~t2 ~projections = v =:+ v1 * v2
let%cd conv_grad_asn ~t:_ ~g ~t1 ~t2 ~projections = g1 =+ g * v2

let force ~projections =
  ignore (Lazy.force projections.Tensor.projections : Ir.Indexing.projections)

(* No [Accum_op] leaf at all, so [collect_neutral_elem] yields [None]: the value the derivation read
   is the value that would be installed. *)
let no_neutral_op_asn ~t:_ ~t1:_ ~projections =
  force ~projections;
  Ir.Assignments.empty_comp

(* A finite neutral (0.) over a pointwise step that reads no margins: [clamp_padded] is [false]
   under both [None] and [Some 0.], and there is no margin neutral to commit, so the derivation is
   identical either way. This is the ordinary shape of a custom [op_asn]. *)
let finite_neutral_op_asn ~t ~t1 ~projections =
  force ~projections;
  summing_op_asn ~t ~t1 ~projections

(* A max accumulation: [Ops.neutral_elem Max] is [neg_infinity], so [clamp_padded] would be [true]
   where the derivation computed [false] -- and it fed [solve_proj_equations], so the whole
   derivation differs. *)
let nonfinite_neutral_op_asn ~t ~t1 ~projections =
  force ~projections;
  maxing_op_asn ~t ~t1 ~projections

(* Finite neutral, but the padded ([=]-mode) window demands margins on the operand, so
   [update_padding_elem] had a commitment to make and made it under [None]: installing [Some 0.]
   afterwards leaves the shape's margin neutral disagreeing with the operation that reads it. This
   is the second disjunct, and the one a pointwise probe cannot reach. *)
let margin_touching_op_asn ~t ~t1 ~t2 ~projections =
  force ~projections;
  conv_op_asn ~t ~t1 ~t2 ~projections

let vec n = TDSL.range_of_shape ~batch_dims:[] ~input_dims:[] ~output_dims:[ n ] ()
let unop ~op_asn = Tensor.unop ~op_label:"guard_probe" ~op_asn ~grad_asn ~grad_spec:Tensor.Prohibit_grad

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
        && String.is_substring msg ~substring:"derived before its neutral element was known")

let () =
  (* The accepted legs first: a rejected one leaves behind the shapes its derivation mutated. *)
  verdict ~claim:"an early derivation with no neutral element to install is accepted"
    ~expect_rejected:false (fun () -> unop ~op_asn:no_neutral_op_asn (vec 4) ());
  verdict ~claim:"an early derivation whose finite neutral reads no margins is accepted"
    ~expect_rejected:false (fun () -> unop ~op_asn:finite_neutral_op_asn (vec 4) ());
  (* The control for the margin leg below: the same padded operation without the early derivation
     constructs fine, so its rejection is about the staleness and not about the padded spec. *)
  verdict ~claim:"the padded operation itself constructs when nothing derives early"
    ~expect_rejected:false (fun () -> padded_conv ~op_asn:conv_op_asn);
  verdict ~claim:"an early derivation whose non-finite neutral flips clamping is rejected"
    ~expect_rejected:true (fun () -> unop ~op_asn:nonfinite_neutral_op_asn (vec 4) ());
  verdict ~claim:"an early derivation whose finite neutral reads margins is rejected"
    ~expect_rejected:true (fun () -> padded_conv ~op_asn:margin_touching_op_asn)
