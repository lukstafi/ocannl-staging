open Base
open Ocannl.Operation.DSL_modules
module Tn = Ir.Tnode
module Nd = Ir.Ndarray
module Ops = Ir.Ops
module Train = Ocannl.Train
module Persistence = Ocannl.Persistence

(* Tests for gh-ocannl-372: namespaces for tensor node IDs. A session's ambient namespace is set
   only by [Tensor.unsafe_reinitialize ~namespace] (default "ocannl", elided from renderings);
   [Persistence.load ~prefix_namespace] rewrites loaded namespaces while preserving session ids,
   which keeps [Embed_self_id] values invariant. *)

let tmp_dir = Stdlib.Filename.get_temp_dir_name ()
let tmp_file name = Stdlib.Filename.concat tmp_dir ("test_tnode_namespaces_" ^ name ^ ".ckpt")

let cleanup name =
  let path = tmp_file name in
  if Stdlib.Sys.file_exists path then Stdlib.Sys.remove path

(* Create a tnode with given values and upload it into [ctx], returning the updated context. *)
let make_tn ctx ~id ~label prec dims values =
  let nd = Nd.create_array ~debug:"test" prec ~dims ~padding:None in
  Nd.set_flat_values nd values;
  let tn, _init = Tn.create_from_padded ~id ~label ~ndarray:nd ~padding:None () in
  (Context.from_host ctx tn nd, tn)

let show ctx tn =
  String.concat ~sep:"; "
    (Array.to_list
       (Array.map (Context.get_values ctx tn) ~f:(fun v -> Stdlib.Printf.sprintf "%.1f" v)))

let () =
  (* === Test 1: Namespace rendering === *)
  Stdio.printf "=== Test 1: Namespace rendering ===\n";
  Tensor.unsafe_reinitialize ();
  let ctx = Context.cpu () in
  let _ctx, tn0 = make_tn ctx ~id:0 ~label:[] Ops.single [| 2 |] [| 1.0; 2.0 |] in
  Stdio.printf "default: id=%s debug_name=%s namespace=%s\n" (Tn.id tn0) (Tn.debug_name tn0)
    tn0.Tn.namespace;
  Tensor.unsafe_reinitialize ~namespace:"snap1" ();
  let ctx = Context.cpu () in
  let _ctx, tn1 = make_tn ctx ~id:0 ~label:[] Ops.single [| 2 |] [| 1.0; 2.0 |] in
  Stdio.printf "snap1: id=%s debug_name=%s namespace=%s\n" (Tn.id tn1) (Tn.debug_name tn1)
    tn1.Tn.namespace;
  (* Plain reinitialization resets the namespace back to the default. *)
  Tensor.unsafe_reinitialize ();
  Stdio.printf "after plain reinit: current namespace=%s\n" (Tn.get_current_namespace ());

  (* === Test 2: Namespace validation === *)
  Stdio.printf "=== Test 2: Namespace validation ===\n";
  List.iter [ "1bad"; "a.b"; "with space"; "" ] ~f:(fun ns ->
      try
        Tensor.unsafe_reinitialize ~namespace:ns ();
        Stdio.printf "ERROR: accepted %S\n" ns
      with Invalid_argument msg -> Stdio.printf "Caught: %s\n" msg);
  Tensor.unsafe_reinitialize ();

  (* === Test 3: Model surgery - loading one checkpoint under two prefixes === *)
  Stdio.printf "=== Test 3: Prefixed loads ===\n";
  Tensor.unsafe_reinitialize ();
  let ctx = Context.cpu () in
  let ctx, w = make_tn ctx ~id:0 ~label:[ "w" ] Ops.single [| 2 |] [| 1.0; 2.0 |] in
  let ctx, b = make_tn ctx ~id:1 ~label:[ "b" ] Ops.single [| 2 |] [| 3.0; 4.0 |] in
  let path = tmp_file "surgery" in
  Persistence.save ~ctx ~appending:false (Set.of_list (module Tn) [ w; b ]) path;
  Tensor.unsafe_reinitialize ();
  let ctx = Context.cpu () in
  let next_before = Tensor.get_next_id () in
  let ctx, left = Persistence.load ~ctx ~prefix_namespace:"left" path in
  let ctx, right = Persistence.load ~ctx ~prefix_namespace:"right" path in
  Set.iter left ~f:(fun tn -> Stdio.printf "  %s values=[%s]\n" (Tn.id tn) (show ctx tn));
  Set.iter right ~f:(fun tn -> Stdio.printf "  %s values=[%s]\n" (Tn.id tn) (show ctx tn));
  Stdio.printf "next_id before=%d after=%d (prefixed loads do not bump)\n" next_before
    (Tensor.get_next_id ());
  (* An unprefixed load lands in the checkpoint's own (default) namespace and bumps the floor. *)
  let ctx, plain = Persistence.load ~ctx path in
  Set.iter plain ~f:(fun tn -> Stdio.printf "  %s values=[%s]\n" (Tn.id tn) (show ctx tn));
  Stdio.printf "next_id after unprefixed load=%d\n" (Tensor.get_next_id ());
  (* Re-loading under an already-used prefix clashes. *)
  (try
     let _ = Persistence.load ~ctx ~prefix_namespace:"left" path in
     Stdio.printf "ERROR: should have raised\n"
   with Failure msg -> Stdio.printf "Caught: %s\n" msg);
  (* Invalid prefixes are rejected up front. *)
  (try
     let _ = Persistence.load ~ctx ~prefix_namespace:"not ok" path in
     Stdio.printf "ERROR: should have raised\n"
   with Invalid_argument msg -> Stdio.printf "Caught: %s\n" msg);
  cleanup "surgery";

  (* === Test 4: Embed_self_id is namespace-invariant === *)
  Stdio.printf "=== Test 4: Embed_self_id invariance ===\n";
  let run_self_id () =
    let ctx = Context.cpu () in
    let%op filler = embed_self_id () in
    let%op hey = embed_self_id () in
    Train.set_materialized hey.value;
    let ctx = Train.forward_once ctx hey in
    ignore filler;
    (Context.get_values ctx hey.value).(0)
  in
  Tensor.unsafe_reinitialize ();
  let v_default = run_self_id () in
  Tensor.unsafe_reinitialize ~namespace:"other_ns" ();
  let v_other = run_self_id () in
  Tensor.unsafe_reinitialize ();
  Stdio.printf "self_id default=%.1f other_ns=%.1f equal=%b\n" v_default v_other
    (Float.equal v_default v_other);

  Stdio.printf "=== All namespace tests completed ===\n"
