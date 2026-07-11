(* Omitted row specs read as the context ellipsis: [x] is equivalent to [...|...->x].
   Writing a kind separator with an empty row spec ([| ->x]) denotes explicitly empty
   (closed) rows, preserving the pre-2026-07 reading of omitted rows. *)
open Ocannl
open Nn_blocks.DSL_modules

let shape_case name f =
  (* Shape inference constraint state is global: reinitialize so a failed case does not
     poison subsequent ones. *)
  Tensor.unsafe_reinitialize ();
  (try
     let ctx = Context.auto () in
     let t = f () in
     let _ctx = Train.forward_once ctx t in
     Stdio.printf "%s: %s\n%!" name (Shape.to_string_hum t.Tensor.shape)
   with Row.Shape_error (msg, _) -> Stdio.printf "%s: Shape_error %s\n%!" name msg);
  Stdio.printf "\n%!"

let () =
  shape_case "batch passes through omitted rows: batched ++ \"i=>i0\"" (fun () ->
      let x = TDSL.range_of_shape ~batch_dims:[ 2 ] ~output_dims:[ 3 ] () in
      let%op y = x ++ "i=>i0" in
      y);
  shape_case "explicit spelling is equivalent: batched ++ \"...|...->i => ...|...->i0\"" (fun () ->
      let x = TDSL.range_of_shape ~batch_dims:[ 2 ] ~output_dims:[ 3 ] () in
      let%op y = x ++ "...|...->i => ...|...->i0" in
      y);
  shape_case "input passes through omitted rows: matrix ++ \"o=>o0\"" (fun () ->
      let x = TDSL.range_of_shape ~input_dims:[ 4 ] ~output_dims:[ 3 ] () in
      let%op y = x ++ "o=>o0" in
      y);
  shape_case "empty rows still inferred when nothing forces axes: vector ++ \"i=>i0\"" (fun () ->
      let x = TDSL.range_of_shape ~output_dims:[ 3 ] () in
      let%op y = x ++ "i=>i0" in
      y);
  shape_case "bare separators close rows: batched ++ \"|->i => |->i0\" errors" (fun () ->
      let x = TDSL.range_of_shape ~batch_dims:[ 2 ] ~output_dims:[ 3 ] () in
      let%op y = x ++ "|->i => |->i0" in
      y);
  shape_case "reduce-all needs closed result rows: batched ++ \"...|... => |->0\"" (fun () ->
      let x = TDSL.range_of_shape ~batch_dims:[ 2 ] ~output_dims:[ 3 ] () in
      let%op y = x ++ "...|... => |->0" in
      y);
  shape_case "omitted result rows keep batch: batched ++ \"...|... => 0\"" (fun () ->
      let x = TDSL.range_of_shape ~batch_dims:[ 2 ] ~output_dims:[ 3 ] () in
      let%op y = x ++ "...|... => 0" in
      y);
  shape_case "terse sum over non-batch: batched ++ \"... => 0\"" (fun () ->
      let x = TDSL.range_of_shape ~batch_dims:[ 2 ] ~output_dims:[ 3 ] () in
      let%op y = x ++ "... => 0" in
      y);
  shape_case "binary einsum with omitted rows: batched matmul \"ij;jk=>ik\"" (fun () ->
      let a = TDSL.range_of_shape ~batch_dims:[ 2 ] ~output_dims:[ 3; 4 ] () in
      let b = TDSL.range_of_shape ~output_dims:[ 4; 5 ] () in
      let%op c = a +* "ij;jk=>ik" b in
      c);
  (* The implicit context row variables use reserved names, so [batch]/[input]/[output] stay
     available as ordinary dimension labels in terse multichar specs. *)
  shape_case "\"batch\" as a dimension label: \"batch, seq, d; batch, time, d => batch, seq, time\""
    (fun () ->
      let a = TDSL.range_of_shape ~output_dims:[ 2; 3; 4 ] () in
      let b = TDSL.range_of_shape ~output_dims:[ 2; 5; 4 ] () in
      let%op c = a +* "batch, seq, d; batch, time, d => batch, seq, time" b in
      c)
