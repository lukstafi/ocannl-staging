open Base
open Ocannl.Operation.DSL_modules

(* [Tensor.op] can only learn an operation's neutral element by reading it off the assignments
   [op_asn] built, so it sets the update step's [neutral_elem] afterwards. The field is consumed
   later still, when the projections are derived. An [op_asn] that derives them early gets them
   derived against [neutral_elem = None], which silently changes the padding and guard decisions in
   [Shape.derive_projections] -- a wrong value, not a crash. [Tensor.op] rejects that rather than
   assuming nobody does it; this is the negative control for the rejection. *)

let%cd sound_op_asn ~t ~t1 ~projections = v =:+ relu v1
let%cd grad_asn ~t:_ ~g ~t1 ~projections = g1 =+ relu_gate (v1, g)

let deriving_op_asn ~t ~t1 ~projections =
  (* The hazard, provoked: force the projections while [neutral_elem] is still [None]. *)
  ignore (Lazy.force projections.Tensor.projections : Ir.Indexing.projections);
  sound_op_asn ~t ~t1 ~projections

let () =
  let x = TDSL.range 4 in
  let rejected =
    try
      let (_ : Tensor.t) =
        Tensor.unop ~op_label:"guard_probe" ~op_asn:deriving_op_asn ~grad_asn
          ~grad_spec:Tensor.Prohibit_grad x ()
      in
      false
    with Tensor.Session_error (msg, _) ->
      Stdio.prerr_endline ("(not part of the golden) rejected with: " ^ msg);
      true
  in
  Verdict.p "op_asn deriving the projections early is rejected" rejected
