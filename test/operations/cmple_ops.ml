(* Functional and NaN-semantics test for the Cmple (less-or-equal) primitive and the derived
   comparisons: [<=] lowers to Cmple, while [>] and [>=] are operand swaps of Cmplt/Cmple.

   The NaN block pins that Cmple is a true primitive and not a not-(b < a) composite: any
   comparison with a NaN operand must yield 0, whereas the negation rewrite would yield 1. This
   guards against a future "simplification" of Cmple into Cmplt-plus-Not. *)

open Base
open Ocannl
open Stdio
open Ocannl.Operation.DSL_modules

let print_row name got =
  printf "%s: [%s]\n" name
    (String.concat ~sep:"; " (Array.to_list got |> List.map ~f:(Printf.sprintf "%.1f")))

let () =
  let a = TDSL.ndarray [| 1.; 2.; 3.; 4. |] ~label:[ "a" ] ~output_dims:[ 4 ] () in
  let b = TDSL.ndarray [| 2.; 2.; 1.; 4. |] ~label:[ "b" ] ~output_dims:[ 4 ] () in
  let%op r_lt = a < b in
  let%op r_le = a <= b in
  let%op r_gt = a > b in
  let%op r_ge = a >= b in
  let ctx = Context.auto () in
  let ctx = Train.forward_once ctx r_lt in
  let ctx = Train.forward_once ctx r_le in
  let ctx = Train.forward_once ctx r_gt in
  let ctx = Train.forward_once ctx r_ge in
  (* a = [1;2;3;4], b = [2;2;1;4]. *)
  print_row "a <  b" (Context.get_values ctx r_lt.Tensor.value);
  print_row "a <= b" (Context.get_values ctx r_le.Tensor.value);
  print_row "a >  b" (Context.get_values ctx r_gt.Tensor.value);
  print_row "a >= b" (Context.get_values ctx r_ge.Tensor.value);
  (* NaN semantics: every comparison with a NaN operand is false (0), on both sides. *)
  let n = TDSL.ndarray [| Float.nan; Float.nan; 5. |] ~label:[ "n" ] ~output_dims:[ 3 ] () in
  let m = TDSL.ndarray [| 1.; Float.nan; 5. |] ~label:[ "m" ] ~output_dims:[ 3 ] () in
  let%op nan_le = n <= m in
  let%op nan_le_rev = m <= n in
  let%op nan_lt = n < m in
  let%op nan_ge = n >= m in
  let ctx = Train.forward_once ctx nan_le in
  let ctx = Train.forward_once ctx nan_le_rev in
  let ctx = Train.forward_once ctx nan_lt in
  let ctx = Train.forward_once ctx nan_ge in
  (* n = [nan;nan;5], m = [1;nan;5]. *)
  print_row "n <= m" (Context.get_values ctx nan_le.Tensor.value);
  print_row "m <= n" (Context.get_values ctx nan_le_rev.Tensor.value);
  print_row "n <  m" (Context.get_values ctx nan_lt.Tensor.value);
  print_row "n >= m" (Context.get_values ctx nan_ge.Tensor.value)
