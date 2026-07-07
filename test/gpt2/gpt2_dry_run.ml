(* Dry run of the full GPT-2 124M architecture with deterministic host-generated random weights:
   builds the 12-layer graph at real dimensions (vocab 50257, d_model 768), runs one forward pass
   over a short token sequence, and checks shapes and numeric sanity. This exercises shape
   inference and the compilation pipeline at real-model scale without any network access; the
   pretrained-weights path is the gpt2_generate tutorial executable. *)

open Base
open Ocannl
open Stdio
open Nn_blocks.DSL_modules

let () =
  let config = Gpt2_model.gpt2_small in
  let seq_len = 8 in
  let model = Gpt2_model.gpt2 ~config ~src:Gpt2_model.Random ~seq_len () in
  (* Arbitrary fixed token IDs, including a large one near the vocabulary edge. *)
  let ids = Nn_blocks.token_ids_of_array ~max_len:seq_len [| 50256; 15496; 11; 995; 0; 1; 2; 3 |] in
  let logits = model ids in
  let ctx = Train.forward_once (Context.auto ()) logits in
  let dims = Lazy.force logits.Tensor.value.Ir.Tnode.dims in
  printf "logits dims: %s\n"
    (String.concat ~sep:"x" (Array.to_list (Array.map dims ~f:Int.to_string)));
  let vals = Context.get_values ctx logits.Tensor.value in
  printf "logits count: %d\n" (Array.length vals);
  printf "all finite: %b\n" (Array.for_all vals ~f:Float.is_finite);
  (* Positions must produce different distributions (the model is not collapsing). *)
  let row p = Array.sub vals ~pos:(p * config.vocab_size) ~len:config.vocab_size in
  let first = row 0 and last = row (seq_len - 1) in
  let differ = Array.existsi first ~f:(fun i v -> Float.(abs (v - last.(i)) > 1e-6)) in
  printf "positions differ: %b\n" differ;
  (* Values should be in a sane range for random weights (LayerNorm keeps activations bounded). *)
  let max_abs = Array.fold vals ~init:0. ~f:(fun acc v -> Float.max acc (Float.abs v)) in
  printf "max |logit| below 100: %b\n" Float.(max_abs < 100.)
