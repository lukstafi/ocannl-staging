(* Debugging probe: compares OCANNL GPT-2 stages against the numpy reference (scratchpad
   ref_gpt2.py) on the same pretrained weights and the fixed prompt [464; 3139; 286; 4881; 318].
   Not a test; run manually while diagnosing weight-mapping issues. *)

open Base
open Ocannl
open Stdio
open Nn_blocks.DSL_modules

let () =
  let config = Gpt2_model.gpt2_small in
  let st =
    Safetensors.read
      (Dataprep.Dataset_utils.get_cache_dir "models" ^ "gpt2/model.safetensors")
  in
  let src = Gpt2_model.Pretrained st in
  let ids_arr = [| 464; 3139; 286; 4881; 318 |] in
  let seq_len = Array.length ids_arr in
  let ids = Nn_blocks.token_ids_of_array ids_arr in
  let print_rows name t ~rows ~width =
    let ctx = Train.forward_once (Context.auto ()) t in
    let vals = Context.get_values ctx t.Tensor.value in
    List.iter rows ~f:(fun row ->
        let off = row * width in
        printf "%s[%d][:3]: %.6f %.6f %.6f\n" name row vals.(off) vals.(off + 1) vals.(off + 2))
  in

  (* Stage 1: embeddings. *)
  let w = Gpt2_model.weight src in
  let wte_embed =
    w ~name:"wte_embed" ~b:[] ~o:[ config.d_model ] ~i:[ config.vocab_size ]
      ~of_hf:(Gpt2_model.hf_transpose "wte.weight") ()
  in
  let pos_embed =
    w ~name:"wpe" ~b:[ seq_len ] ~o:[ config.d_model ] ~i:[]
      ~of_hf:(fun st ->
        Bigarray.Genarray.sub_left (Safetensors.to_float32 st "wpe.weight") 0 seq_len)
      ()
  in
  let one_hot = Nn_blocks.one_hot_of_ids ~num_classes:config.vocab_size ids in
  let embed = [%op (wte_embed * one_hot) + pos_embed] in
  print_rows "embed" embed ~rows:[ 0; seq_len - 1 ] ~width:config.d_model;

  (* Stage 2: embeddings + block 0. *)
  let mask =
    NTDSL.init ~l:"causal_mask" ~prec:Ir.Ops.single ~b:[ seq_len ] ~i:[ seq_len ] ~o:[]
      ~f:(function
        | [| s; t |] -> if s >= t then 1. else 0.
        | _ -> assert false)
      ()
  in
  let ids1 = Nn_blocks.token_ids_of_array ids_arr in
  let one_hot1 = Nn_blocks.one_hot_of_ids ~num_classes:config.vocab_size ids1 in
  let wte_embed1 =
    w ~name:"wte_embed1" ~b:[] ~o:[ config.d_model ] ~i:[ config.vocab_size ]
      ~of_hf:(Gpt2_model.hf_transpose "wte.weight") ()
  in
  let pos_embed1 =
    w ~name:"wpe1" ~b:[ seq_len ] ~o:[ config.d_model ] ~i:[]
      ~of_hf:(fun st ->
        Bigarray.Genarray.sub_left (Safetensors.to_float32 st "wpe.weight") 0 seq_len)
      ()
  in
  let embed1 = [%op (wte_embed1 * one_hot1) + pos_embed1] in
  let block0 = Gpt2_model.block ~config ~src ~layer:0 in
  let x0 = block0 ~mask embed1 in
  print_rows "after block 0" x0 ~rows:[ seq_len - 1 ] ~width:config.d_model;

  (* Stage 3: full model logits. *)
  let model = Gpt2_model.gpt2 ~config ~src ~seq_len () in
  let ids2 = Nn_blocks.token_ids_of_array ids_arr in
  let logits = model ids2 in
  let ctx = Train.forward_once (Context.auto ()) logits in
  let vals = Context.get_values ctx logits.Tensor.value in
  let off = (seq_len - 1) * config.vocab_size in
  printf "logits[-1][:5]: %.4f %.4f %.4f %.4f %.4f\n" vals.(off) vals.(off + 1) vals.(off + 2)
    vals.(off + 3) vals.(off + 4);
  let idx = Array.init config.vocab_size ~f:(fun i -> i) in
  Array.sort idx ~compare:(fun a b -> Float.compare vals.(off + b) vals.(off + a));
  printf "top-5 ids: %d %d %d %d %d\n" idx.(0) idx.(1) idx.(2) idx.(3) idx.(4);
  printf "top-5 logits: %.4f %.4f %.4f %.4f %.4f\n" vals.(off + idx.(0)) vals.(off + idx.(1))
    vals.(off + idx.(2)) vals.(off + idx.(3)) vals.(off + idx.(4))
