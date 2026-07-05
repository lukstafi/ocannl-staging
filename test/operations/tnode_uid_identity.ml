(* Tnode identity is the process-unique [uid], not the presentational [id]: [id]s restart at 0 on
   [Tensor.unsafe_reinitialize] (for deterministic printing), so a tnode-keyed table surviving a
   reinitialization must NOT alias a fresh node that reuses a stale node's [id]. This class of
   aliasing was the root cause of a backend cache leaking between tests (the bug that motivated
   the per-call generative backend functors, retired in favor of backend singletons). *)

open Base
open Stdio
open Ocannl
open Operation.DSL_modules
module Tn = Ir.Tnode

let make () =
  let%op x = [ 1.; 2. ] in
  x.Tensor.value

let () =
  let tn1 = make () in
  let table = Hashtbl.create (module Tn) in
  Hashtbl.set table ~key:tn1 ~data:"stale";
  let set1 = Set.singleton (module Tn) tn1 in
  Tensor.unsafe_reinitialize ();
  let tn2 = make () in
  printf "printed ids equal across reinitialize: %b (%s = %s)\n"
    (String.equal (Tn.id tn1) (Tn.id tn2))
    (Tn.id tn1) (Tn.id tn2);
  printf "stale node still hits its own table entry: %b\n"
    (Option.is_some (Hashtbl.find table tn1));
  printf "fresh same-id node misses the stale table entry: %b\n"
    (Option.is_none (Hashtbl.find table tn2));
  printf "fresh same-id node is not Tn.equal to the stale node: %b\n" (not (Tn.equal tn1 tn2));
  printf "fresh same-id node is not a member of the stale set: %b\n" (not (Set.mem set1 tn2))
