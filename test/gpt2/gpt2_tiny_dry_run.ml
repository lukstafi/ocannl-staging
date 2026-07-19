(* Fast dry run of the GPT-2 architecture at tiny dimensions. Regression test for the Embed_dim
   lowering-order bug: in a deep model the (dim d) fetch of the final layer_norm is the first
   statement lowered, and its variable_ref is only filled in when shape inference completes -- so
   Embed_dim lowering must force the fetched node's dims before reading the solved dimension. *)

open Base
open Ocannl
open Stdio
open Nn_blocks.DSL_modules

let () =
  let config =
    Gpt2_model.
      {
        vocab_size = 11;
        n_positions = 8;
        d_model = 8;
        n_layers = 2;
        n_heads = 2;
        d_head = 4;
        d_ff = 16;
        epsilon = 1e-5;
      }
  in
  let seq_len = 4 in
  let model = Gpt2_model.gpt2 ~config ~src:Gpt2_model.Random ~seq_len () in
  let ids = Nn_blocks.token_ids_of_array ~max_len:seq_len [| 10; 3; 1; 2 |] in
  let logits = model ids in
  let ctx = Train.forward_once (Context.auto ()) logits in
  let dims = Lazy.force logits.Tensor.value.Ir.Tnode.dims in
  printf "logits dims: %s\n"
    (String.concat ~sep:"x" (Array.to_list (Array.map dims ~f:Int.to_string)));
  let vals = Context.get_values ctx logits.Tensor.value in
  printf "logits count: %d\n" (Array.length vals);
  printf "all finite: %b\n" (Array.for_all vals ~f:Float.is_finite)
