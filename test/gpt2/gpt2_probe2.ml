(* Debugging probe, stage 2: bisect inside block 0 against ref_block0.py. Not a test. *)

open Base
open Ocannl
open Stdio
open Nn_blocks.DSL_modules

let () =
  let config = Gpt2_model.gpt2_small in
  let { Gpt2_model.d_model; n_heads; d_head; epsilon; vocab_size; _ } = config in
  let st =
    Safetensors.read (Dataprep.Dataset_utils.get_cache_dir "models" ^ "gpt2/model.safetensors")
  in
  let src = Gpt2_model.Pretrained st in
  let ids_arr = [| 464; 3139; 286; 4881; 318 |] in
  let seq_len = Array.length ids_arr in
  let row = seq_len - 1 in
  let w = Gpt2_model.weight src in
  let fresh_embed suffix =
    let ids = Nn_blocks.token_ids_of_array ids_arr in
    let one_hot = Nn_blocks.one_hot_of_ids ~num_classes:vocab_size ids in
    let wte =
      w ~name:("wte_embed" ^ suffix) ~b:[] ~o:[ d_model ] ~i:[ vocab_size ]
        ~of_hf:(Gpt2_model.hf_transpose "wte.weight") ()
    in
    let wpe =
      w ~name:("wpe" ^ suffix) ~b:[ seq_len ] ~o:[ d_model ] ~i:[]
        ~of_hf:(fun st ->
          Bigarray.Genarray.sub_left (Safetensors.to_float32 st "wpe.weight") 0 seq_len)
        ()
    in
    [%op (wte * one_hot) + wpe]
  in
  let ln1_weights suffix =
    ( w ~name:("ln1g" ^ suffix) ~center:1.0 ~scale:0.0 ~b:[] ~o:[ d_model ] ~i:[]
        ~of_hf:(Gpt2_model.hf_id "h.0.ln_1.weight") (),
      w ~name:("ln1b" ^ suffix) ~b:[] ~o:[ d_model ] ~i:[]
        ~of_hf:(Gpt2_model.hf_id "h.0.ln_1.bias") () )
  in
  let attn_weights suffix =
    let qkv_w which =
      Gpt2_model.hf_qkv_w "h.0.attn.c_attn.weight" ~which ~n_heads ~d_head ~d_model
    in
    let qkv_b which = Gpt2_model.hf_qkv_b "h.0.attn.c_attn.bias" ~which ~n_heads ~d_head in
    ( w ~name:("wq" ^ suffix) ~b:[] ~o:[ n_heads; d_head ] ~i:[ d_model ] ~of_hf:(qkv_w 0) (),
      w ~name:("bq" ^ suffix) ~b:[] ~o:[ n_heads; d_head ] ~i:[] ~of_hf:(qkv_b 0) (),
      w ~name:("wk" ^ suffix) ~b:[] ~o:[ n_heads; d_head ] ~i:[ d_model ] ~of_hf:(qkv_w 1) (),
      w ~name:("bk" ^ suffix) ~b:[] ~o:[ n_heads; d_head ] ~i:[] ~of_hf:(qkv_b 1) (),
      w ~name:("wv" ^ suffix) ~b:[] ~o:[ n_heads; d_head ] ~i:[ d_model ] ~of_hf:(qkv_w 2) (),
      w ~name:("bv" ^ suffix) ~b:[] ~o:[ n_heads; d_head ] ~i:[] ~of_hf:(qkv_b 2) (),
      w
        ~name:("wo" ^ suffix)
        ~b:[] ~o:[ d_model ] ~i:[ n_heads; d_head ]
        ~of_hf:(fun st ->
          let src = Safetensors.to_float32 st "h.0.attn.c_proj.weight" in
          Gpt2_model.permuted src
            ~dims:[ d_model; n_heads; d_head ]
            ~f:(fun idx -> [| (idx.(1) * d_head) + idx.(2); idx.(0) |]))
        (),
      w ~name:("bo" ^ suffix) ~b:[] ~o:[ d_model ] ~i:[]
        ~of_hf:(Gpt2_model.hf_id "h.0.attn.c_proj.bias") () )
  in
  let mask () =
    NTDSL.init ~l:"causal_mask" ~prec:Ir.Ops.single ~b:[ seq_len ] ~i:[ seq_len ] ~o:[]
      ~f:(function [| s; t |] -> if s >= t then 1. else 0. | _ -> assert false)
      ()
  in
  let print3 name t =
    let ctx = Train.forward_once (Context.auto ()) t in
    let vals = Context.get_values ctx t.Tensor.value in
    let off = row * d_model in
    printf "%s[4][:3]: %.6f %.6f %.6f\n%!" name vals.(off) vals.(off + 1) vals.(off + 2)
  in

  (* ln1 output *)
  let g1, b1 = ln1_weights "_a" in
  print3 "ln1" (Gpt2_model.layer_norm ~epsilon () ~gamma:g1 ~beta:b1 (fresh_embed "_a"));

  (* attention output (on a fresh ln1) *)
  let g1b, b1b = ln1_weights "_b" in
  let wq, bq, wk, bk, wv, bv, wo, bo = attn_weights "_b" in
  let attn_out =
    Gpt2_model.attention ~n_heads ~d_head () ~w_q:wq ~b_q:bq ~w_k:wk ~b_k:bk ~w_v:wv ~b_v:bv
      ~w_o:wo ~b_o:bo ~mask:(mask ())
      (Gpt2_model.layer_norm ~epsilon () ~gamma:g1b ~beta:b1b (fresh_embed "_b"))
  in
  print3 "attn" attn_out

(* Granular attention internals: q, scores, weights (head 0, row 4). *)
let () =
  let config = Gpt2_model.gpt2_small in
  let { Gpt2_model.d_model; n_heads; d_head; epsilon; vocab_size; _ } = config in
  let st =
    Safetensors.read (Dataprep.Dataset_utils.get_cache_dir "models" ^ "gpt2/model.safetensors")
  in
  let src = Gpt2_model.Pretrained st in
  let ids_arr = [| 464; 3139; 286; 4881; 318 |] in
  let seq_len = Array.length ids_arr in
  let row = seq_len - 1 in
  let w = Gpt2_model.weight src in
  let ids = Nn_blocks.token_ids_of_array ids_arr in
  let one_hot = Nn_blocks.one_hot_of_ids ~num_classes:vocab_size ids in
  let wte =
    w ~name:"wte_c" ~b:[] ~o:[ d_model ] ~i:[ vocab_size ]
      ~of_hf:(Gpt2_model.hf_transpose "wte.weight") ()
  in
  let wpe =
    w ~name:"wpe_c" ~b:[ seq_len ] ~o:[ d_model ] ~i:[]
      ~of_hf:(fun st ->
        Bigarray.Genarray.sub_left (Safetensors.to_float32 st "wpe.weight") 0 seq_len)
      ()
  in
  let g1 =
    w ~name:"ln1g_c" ~b:[] ~o:[ d_model ] ~i:[] ~of_hf:(Gpt2_model.hf_id "h.0.ln_1.weight") ()
  in
  let b1 =
    w ~name:"ln1b_c" ~b:[] ~o:[ d_model ] ~i:[] ~of_hf:(Gpt2_model.hf_id "h.0.ln_1.bias") ()
  in
  let qkv_w which = Gpt2_model.hf_qkv_w "h.0.attn.c_attn.weight" ~which ~n_heads ~d_head ~d_model in
  let qkv_b which = Gpt2_model.hf_qkv_b "h.0.attn.c_attn.bias" ~which ~n_heads ~d_head in
  let wq = w ~name:"wq_c" ~b:[] ~o:[ n_heads; d_head ] ~i:[ d_model ] ~of_hf:(qkv_w 0) () in
  let bq = w ~name:"bq_c" ~b:[] ~o:[ n_heads; d_head ] ~i:[] ~of_hf:(qkv_b 0) () in
  let wk = w ~name:"wk_c" ~b:[] ~o:[ n_heads; d_head ] ~i:[ d_model ] ~of_hf:(qkv_w 1) () in
  let bk = w ~name:"bk_c" ~b:[] ~o:[ n_heads; d_head ] ~i:[] ~of_hf:(qkv_b 1) () in
  let mask =
    NTDSL.init ~l:"causal_mask_c" ~prec:Ir.Ops.single ~b:[ seq_len ] ~i:[ seq_len ] ~o:[]
      ~f:(function [| s; t |] -> if s >= t then 1. else 0. | _ -> assert false)
      ()
  in
  let embed = [%op (wte * one_hot) + wpe] in
  let l1 = Gpt2_model.layer_norm ~epsilon () ~gamma:g1 ~beta:b1 embed in
  let%op q = (wq * l1) + bq in
  let%op k = (wk * l1) + bk in
  let%op scores =
    (q +* k " ... s | h d; ... t | h d => ... s | t -> h" [ "h"; "d" ]) /. sqrt (dim d)
  in
  Shape.set_dim h n_heads;
  Shape.set_dim d d_head;
  let%op weights = Nn_blocks.softmax ~spec:" ... | t -> ..." () (where mask scores !.(-1e9)) in
  (* q dims [seq; heads; d_head]; scores dims [seq(s); heads(out); seq(t, input rightmost)]. *)
  Train.set_materialized q.Tensor.value;
  Train.set_materialized scores.Tensor.value;
  let ctx = Train.forward_once (Context.auto ()) weights in
  let qv = Context.get_values ctx q.Tensor.value in
  let qoff = row * n_heads * d_head in
  Stdio.printf "q[4][:3]: %.6f %.6f %.6f\n%!" qv.(qoff) qv.(qoff + 1) qv.(qoff + 2);
  let sv = Context.get_values ctx scores.Tensor.value in
  let soff = (row * n_heads * seq_len) + (0 * seq_len) in
  Stdio.printf "scores h0 row4: %.6f %.6f %.6f %.6f %.6f\n%!" sv.(soff) sv.(soff + 1)
    sv.(soff + 2) sv.(soff + 3) sv.(soff + 4);
  let wv = Context.get_values ctx weights.Tensor.value in
  Stdio.printf "weights h0 row4: %.6f %.6f %.6f %.6f %.6f\n%!" wv.(soff) wv.(soff + 1)
    wv.(soff + 2) wv.(soff + 3) wv.(soff + 4)

(* Pre-projection attention output: weights . v, head 0, row 4. *)
let () =
  let config = Gpt2_model.gpt2_small in
  let { Gpt2_model.d_model; n_heads; d_head; epsilon; vocab_size; _ } = config in
  let st =
    Safetensors.read (Dataprep.Dataset_utils.get_cache_dir "models" ^ "gpt2/model.safetensors")
  in
  let src = Gpt2_model.Pretrained st in
  let ids_arr = [| 464; 3139; 286; 4881; 318 |] in
  let seq_len = Array.length ids_arr in
  let row = seq_len - 1 in
  let w = Gpt2_model.weight src in
  let ids = Nn_blocks.token_ids_of_array ids_arr in
  let one_hot = Nn_blocks.one_hot_of_ids ~num_classes:vocab_size ids in
  let wte =
    w ~name:"wte_d" ~b:[] ~o:[ d_model ] ~i:[ vocab_size ]
      ~of_hf:(Gpt2_model.hf_transpose "wte.weight") ()
  in
  let wpe =
    w ~name:"wpe_d" ~b:[ seq_len ] ~o:[ d_model ] ~i:[]
      ~of_hf:(fun st ->
        Bigarray.Genarray.sub_left (Safetensors.to_float32 st "wpe.weight") 0 seq_len)
      ()
  in
  let g1 =
    w ~name:"ln1g_d" ~b:[] ~o:[ d_model ] ~i:[] ~of_hf:(Gpt2_model.hf_id "h.0.ln_1.weight") ()
  in
  let b1 =
    w ~name:"ln1b_d" ~b:[] ~o:[ d_model ] ~i:[] ~of_hf:(Gpt2_model.hf_id "h.0.ln_1.bias") ()
  in
  let qkv_w which = Gpt2_model.hf_qkv_w "h.0.attn.c_attn.weight" ~which ~n_heads ~d_head ~d_model in
  let qkv_b which = Gpt2_model.hf_qkv_b "h.0.attn.c_attn.bias" ~which ~n_heads ~d_head in
  let wq = w ~name:"wq_d" ~b:[] ~o:[ n_heads; d_head ] ~i:[ d_model ] ~of_hf:(qkv_w 0) () in
  let bq = w ~name:"bq_d" ~b:[] ~o:[ n_heads; d_head ] ~i:[] ~of_hf:(qkv_b 0) () in
  let wk = w ~name:"wk_d" ~b:[] ~o:[ n_heads; d_head ] ~i:[ d_model ] ~of_hf:(qkv_w 1) () in
  let bk = w ~name:"bk_d" ~b:[] ~o:[ n_heads; d_head ] ~i:[] ~of_hf:(qkv_b 1) () in
  let wv = w ~name:"wv_d" ~b:[] ~o:[ n_heads; d_head ] ~i:[ d_model ] ~of_hf:(qkv_w 2) () in
  let bv = w ~name:"bv_d" ~b:[] ~o:[ n_heads; d_head ] ~i:[] ~of_hf:(qkv_b 2) () in
  let mask =
    NTDSL.init ~l:"causal_mask_d" ~prec:Ir.Ops.single ~b:[ seq_len ] ~i:[ seq_len ] ~o:[]
      ~f:(function [| s; t |] -> if s >= t then 1. else 0. | _ -> assert false)
      ()
  in
  let embed = [%op (wte * one_hot) + wpe] in
  let l1 = Gpt2_model.layer_norm ~epsilon () ~gamma:g1 ~beta:b1 embed in
  let%op q = (wq * l1) + bq in
  let%op k = (wk * l1) + bk in
  let%op v = (wv * l1) + bv in
  let%op scores =
    (q +* k " ... s | h d; ... t | h d => ... s | t -> h" [ "h"; "d" ]) /. sqrt (dim d)
  in
  Shape.set_dim h n_heads;
  Shape.set_dim d d_head;
  let%op attn_weights = Nn_blocks.softmax ~spec:" ... | t -> ..." () (where mask scores !.(-1e9)) in
  let%op preproj = attn_weights +* v " ... s | t -> h; ... t | h e => ... s | h e" [ "e" ] in
  Shape.set_dim e d_head;
  Train.set_materialized v.Tensor.value;
  let ctx = Train.forward_once (Context.auto ()) preproj in
  let vv = Context.get_values ctx v.Tensor.value in
  let voff = row * n_heads * d_head in
  Stdio.printf "v[4][:3]: %.6f %.6f %.6f\n%!" vv.(voff) vv.(voff + 1) vv.(voff + 2);
  let pv = Context.get_values ctx preproj.Tensor.value in
  Stdio.printf "preproj[4][:3]: %.6f %.6f %.6f\n%!" pv.(voff) pv.(voff + 1) pv.(voff + 2)

(* Discriminate: are wv's stored values correct, and is the bare matmul zero? *)
let () =
  let config = Gpt2_model.gpt2_small in
  let { Gpt2_model.d_model; n_heads; d_head; epsilon; vocab_size; _ } = config in
  let st =
    Safetensors.read (Dataprep.Dataset_utils.get_cache_dir "models" ^ "gpt2/model.safetensors")
  in
  let src = Gpt2_model.Pretrained st in
  let ids_arr = [| 464; 3139; 286; 4881; 318 |] in
  let seq_len = Array.length ids_arr in
  let row = seq_len - 1 in
  let w = Gpt2_model.weight src in
  let ids = Nn_blocks.token_ids_of_array ids_arr in
  let one_hot = Nn_blocks.one_hot_of_ids ~num_classes:vocab_size ids in
  let wte =
    w ~name:"wte_e" ~b:[] ~o:[ d_model ] ~i:[ vocab_size ]
      ~of_hf:(Gpt2_model.hf_transpose "wte.weight") ()
  in
  let wpe =
    w ~name:"wpe_e" ~b:[ seq_len ] ~o:[ d_model ] ~i:[]
      ~of_hf:(fun st ->
        Bigarray.Genarray.sub_left (Safetensors.to_float32 st "wpe.weight") 0 seq_len)
      ()
  in
  let g1 =
    w ~name:"ln1g_e" ~b:[] ~o:[ d_model ] ~i:[] ~of_hf:(Gpt2_model.hf_id "h.0.ln_1.weight") ()
  in
  let b1 =
    w ~name:"ln1b_e" ~b:[] ~o:[ d_model ] ~i:[] ~of_hf:(Gpt2_model.hf_id "h.0.ln_1.bias") ()
  in
  let qkv_w which = Gpt2_model.hf_qkv_w "h.0.attn.c_attn.weight" ~which ~n_heads ~d_head ~d_model in
  let wv = w ~name:"wv_e" ~b:[] ~o:[ n_heads; d_head ] ~i:[ d_model ] ~of_hf:(qkv_w 2) () in
  let embed = [%op (wte * one_hot) + wpe] in
  let l1 = Gpt2_model.layer_norm ~epsilon () ~gamma:g1 ~beta:b1 embed in
  let%op mm = wv * l1 in
  Train.set_materialized wv.Tensor.value;
  let ctx = Train.forward_once (Context.auto ()) mm in
  let wvv = Context.get_values ctx wv.Tensor.value in
  (* wv dims [h; e; m]: [0,0,m=0..2] should equal c_attn.weight[m, 1536]..: 
     numpy W[0,1536], W[1,1536], W[2,1536]. *)
  Stdio.printf "wv[0][0][m=0..2]: %.6f %.6f %.6f\n%!" wvv.(0) wvv.(1) wvv.(2);
  let mmv = Context.get_values ctx mm.Tensor.value in
  let moff = row * n_heads * d_head in
  Stdio.printf "wv*l1[4][:3]: %.6f %.6f %.6f\n%!" mmv.(moff) mmv.(moff + 1) mmv.(moff + 2)
