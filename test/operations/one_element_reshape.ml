(* Regression test for gh-460: a 1-element Reshape/rebatch terminal must keep a single dim-1 axis
   instead of collapsing to a scalar (identity-reshape preference), and rank-0 tensors must not
   crash [Tnode.create_with_reshape]. *)

open Base
open Ocannl
open Operation.DSL_modules
open Stdio

let nd_of_floats vals =
  let open Bigarray in
  let genarray = Genarray.create Float32 c_layout [| Array.length vals |] in
  Array.iteri vals ~f:(fun i v -> Genarray.set genarray [| i |] v);
  Ir.Ndarray.as_array Ir.Ops.Single genarray

let dims_string (t : Tensor.t) =
  ignore (Lazy.force t.value.Ir.Tnode.dims);
  Ir.Tnode.dims_to_string t.value

let () =
  (* The issue repro: a single-token sequence must produce a [1] batch, and one-hot [1; 5]. *)
  let ids = Nn_blocks.class_ids_of_int_list [ 2 ] in
  let oh = Nn_blocks.one_hot_of_ids ~num_classes:5 ids in
  let ctx = Train.forward_once (Context.auto ()) oh in
  printf "single-token ids dims: %s\n" (dims_string ids);
  printf "single-token one-hot dims: %s\n" (dims_string oh);
  Train.printf ~here:[%here] ~with_code:false ~with_grad:false ctx oh;

  (* Sanity: the multi-token case is unchanged. *)
  let ids2 = Nn_blocks.class_ids_of_int_list [ 2; 4 ] in
  let oh2 = Nn_blocks.one_hot_of_ids ~num_classes:5 ids2 in
  let _ctx = Train.forward_once (Context.auto ()) oh2 in
  printf "two-token ids dims: %s\n" (dims_string ids2);
  printf "two-token one-hot dims: %s\n" (dims_string oh2);

  (* rebatch of a 1-element array keeps the [1] batch axis. *)
  let t1 = TDSL.rebatch ~l:"one_elem" (nd_of_floats [| 7. |]) () in
  let _ctx = Train.forward_once (Context.auto ()) t1 in
  printf "rebatch of 1 element dims: %s\n" (dims_string t1);

  (* Explicit dims consuming the whole total still close the batch row to empty. *)
  let t6 = TDSL.reshape ~l:"six" ~o:[ 2; 3 ] (nd_of_floats [| 1.; 2.; 3.; 4.; 5.; 6. |]) () in
  let _ctx = Train.forward_once (Context.auto ()) t6 in
  printf "reshape 6 elements ~o:[2;3] dims: %s\n" (dims_string t6);

  (* An explicitly requested scalar reshape still works; exercises rank-0 create_with_reshape. *)
  let t0 = TDSL.reshape ~l:"one_scalar" ~b:[] ~o:[] (nd_of_floats [| 3. |]) () in
  let ctx = Train.forward_once (Context.auto ()) t0 in
  printf "reshape 1 element ~b:[] ~o:[] dims: %s\n" (dims_string t0);
  Train.printf ~here:[%here] ~with_code:false ~with_grad:false ctx t0;

  (* Fully inferred 1-element reshape keeps one (output) axis. *)
  let ta = TDSL.reshape ~l:"one_auto" (nd_of_floats [| 5. |]) () in
  let _ctx = Train.forward_once (Context.auto ()) ta in
  printf "reshape 1 element inferred dims: %s\n" (dims_string ta)
