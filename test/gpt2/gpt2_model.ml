(* GPT-2 (decoder-only transformer, pre-LN, HuggingFace GPT2LMHeadModel semantics) assembled from
   OCANNL operations, parameterized by the weights source: host-generated random weights (for dry
   runs exercising shape inference and compilation at full model scale) or a pretrained HuggingFace
   [model.safetensors] checkpoint.

   Differences from the generic Nn_blocks transformer blocks, needed to match pretrained weights
   exactly: pre-LN residual structure with a final layer norm, biases on the attention and FFN
   projections, textbook LayerNorm (mean subtraction and mean-of-squares variance; Nn_blocks
   [layer_norm] computes a different normalizer), and the tanh-approximate GeLU.

   Weight layouts: OCANNL tensor dims are [batch @ output @ input] with input axes rightmost, so
   e.g. the query projection is [n_heads; d_head; d_model] and maps from the HF Conv1D convention (y
   = x W + b with W of shape [d_in; d_out], fused qkv in c_attn) by transposition/slicing. *)

open Base
open Ocannl
module Nb = Nn_blocks
open Nb.DSL_modules

type config = {
  vocab_size : int;
  n_positions : int;  (** Maximum sequence length (the wpe table height). *)
  d_model : int;
  n_layers : int;
  n_heads : int;
  d_head : int;
  d_ff : int;
  epsilon : float;
}

let gpt2_small =
  {
    vocab_size = 50257;
    n_positions = 1024;
    d_model = 768;
    n_layers = 12;
    n_heads = 12;
    d_head = 64;
    d_ff = 3072;
    epsilon = 1e-5;
  }

type source =
  | Random  (** Deterministic host-generated pseudo-random weights (backend-independent). *)
  | Pretrained of Safetensors.t

(* --- Deterministic host-side pseudo-random init (splitmix64) --- *)

let splitmix64 state =
  let open Int64 in
  let z = state + 0x9E3779B97F4A7C15L in
  let z = z lxor shift_right_logical z 30 * 0xBF58476D1CE4E5B9L in
  let z = z lxor shift_right_logical z 27 * 0x94D049BB133111EBL in
  z lxor shift_right_logical z 31

let random_genarray ~salt ~center ~scale dims =
  let open Bigarray in
  let ga = Genarray.create Float32 c_layout (Array.of_list dims) in
  let numel = List.fold dims ~init:1 ~f:( * ) in
  let flat = reshape_1 ga numel in
  for i = 0 to numel - 1 do
    let bits = splitmix64 Int64.((of_int salt * 0x2545F4914F6CDD1DL) + of_int i) in
    let unit_centered =
      (Float.of_int64 Int64.(bits land 0xFFFFFFL) /. Float.of_int (1 lsl 24)) -. 0.5
    in
    flat.{i} <- center +. (scale *. unit_centered)
  done;
  ga

(* --- HF layout transforms (all produce float32 genarrays with OCANNL dims) --- *)

(** Fresh genarray of [dims] with [dst.(idx) = src.(f idx)]. *)
let permuted src ~dims ~f =
  let open Bigarray in
  let dst = Genarray.create Float32 c_layout (Array.of_list dims) in
  let rec iter idx d =
    if d = List.length dims then Genarray.set dst idx (Genarray.get src (f idx))
    else
      for i = 0 to List.nth_exn dims d - 1 do
        idx.(d) <- i;
        iter idx (d + 1)
      done
  in
  iter (Array.create ~len:(List.length dims) 0) 0;
  dst

(** A weight tensor with dims [b @ o @ i]: [Random] fills it host-side deterministically from
    [name]; [Pretrained] reads and lays out the HF checkpoint via [of_hf]. *)
let weight src ~name ?(center = 0.0) ?(scale = 0.02) ~b ~o ~i ~of_hf () =
  let dims = b @ o @ i in
  let ga =
    match src with
    | Random -> random_genarray ~salt:(String.hash name) ~center ~scale dims
    | Pretrained st ->
        let ga = of_hf st in
        let actual = Bigarray.Genarray.dims ga |> Array.to_list in
        if not (List.equal Int.equal actual dims) then
          failwith
            [%string
              "gpt2 weight %{name}: expected dims %{String.concat ~sep:\"x\" (List.map dims \
               ~f:Int.to_string)}, got %{String.concat ~sep:\"x\" (List.map actual \
               ~f:Int.to_string)}"];
        ga
  in
  TDSL.wrap ~l:name ~b ~i ~o (Ir.Ndarray.as_array Ir.Ops.Single ga) ()

let hf_id name st = Safetensors.to_float32 st name

let hf_transpose name st =
  let src = Safetensors.to_float32 st name in
  let d = Bigarray.Genarray.dims src in
  permuted src ~dims:[ d.(1); d.(0) ] ~f:(fun idx -> [| idx.(1); idx.(0) |])

(** Slice one of q/k/v out of the fused [c_attn.weight] [d_model; 3*d_model] and lay it out as
    [n_heads; d_head; d_model]. *)
let hf_qkv_w name ~which ~n_heads ~d_head ~d_model st =
  let src = Safetensors.to_float32 st name in
  let off = which * n_heads * d_head in
  permuted src ~dims:[ n_heads; d_head; d_model ] ~f:(fun idx ->
      [| idx.(2); off + (idx.(0) * d_head) + idx.(1) |])

(** The matching bias slice of [c_attn.bias], laid out as [n_heads; d_head]. *)
let hf_qkv_b name ~which ~n_heads ~d_head st =
  let src = Safetensors.to_float32 st name in
  let off = which * n_heads * d_head in
  permuted src ~dims:[ n_heads; d_head ] ~f:(fun idx -> [| off + (idx.(0) * d_head) + idx.(1) |])

(* --- Model blocks (%op with weights passed explicitly; no inline params) --- *)

(** Textbook LayerNorm: gamma * (x - mean) / sqrt (mean ((x - mean)^2) + eps) + beta. *)
let%op layer_norm ~epsilon () ~gamma ~beta x =
  let mean = (x ++ " ... | ..d.. => ... | 0 " [ "d" ]) /. dim d in
  let centered = x - mean in
  let variance = ((centered *. centered) ++ " ... | ... => ... | 0 ") /. dim d in
  (gamma *. (centered /. sqrt (variance + !.epsilon))) + beta

(** Causal multi-head self-attention with projection biases (GPT-2 style). The sequence axis is the
    rightmost batch axis of [x]; [mask] is 1 where position [s] may attend to [t]. *)
let%op attention ~n_heads ~d_head () ~w_q ~b_q ~w_k ~b_k ~w_v ~b_v ~w_o ~b_o ~mask x =
  let q = (w_q * x) + b_q in
  let k = (w_k * x) + b_k in
  let v = (w_v * x) + b_v in
  let scores =
    (q +* k " ... s | h d; ... t | h d => ... s | t -> h" [ "h"; "d" ]) /. sqrt (dim d)
  in
  Shape.set_dim h n_heads;
  Shape.set_dim d d_head;
  Shape.set_dim e d_head;
  let attn_weights = Nb.softmax ~spec:" ... | t -> ..." () (where mask scores !.(-1e9)) in
  (w_o * (attn_weights +* v " ... s | t -> h; ... t | h e => ... s | h e" [ "e" ])) + b_o

let%op ffn () ~w_fc ~b_fc ~w_proj ~b_proj x = (w_proj * Nb.gelu ((w_fc * x) + b_fc)) + b_proj

(** One pre-LN decoder block: x + attn (ln1 x), then x + ffn (ln2 x). *)
let block ~config ~src ~layer =
  let { d_model; n_heads; d_head; d_ff; epsilon; _ } = config in
  let p suffix = [%string "h.%{layer#Int}.%{suffix}"] in
  let w = weight src in
  let qkv_w which = hf_qkv_w (p "attn.c_attn.weight") ~which ~n_heads ~d_head ~d_model in
  let qkv_b which = hf_qkv_b (p "attn.c_attn.bias") ~which ~n_heads ~d_head in
  let ln_1_g =
    w ~name:(p "ln_1.weight") ~center:1.0 ~scale:0.0 ~b:[] ~o:[ d_model ] ~i:[]
      ~of_hf:(hf_id (p "ln_1.weight"))
      ()
  in
  let ln_1_b =
    w ~name:(p "ln_1.bias") ~center:0.0 ~scale:0.0 ~b:[] ~o:[ d_model ] ~i:[]
      ~of_hf:(hf_id (p "ln_1.bias"))
      ()
  in
  let ln_2_g =
    w ~name:(p "ln_2.weight") ~center:1.0 ~scale:0.0 ~b:[] ~o:[ d_model ] ~i:[]
      ~of_hf:(hf_id (p "ln_2.weight"))
      ()
  in
  let ln_2_b =
    w ~name:(p "ln_2.bias") ~center:0.0 ~scale:0.0 ~b:[] ~o:[ d_model ] ~i:[]
      ~of_hf:(hf_id (p "ln_2.bias"))
      ()
  in
  let w_q =
    w ~name:(p "attn.w_q") ~b:[] ~o:[ n_heads; d_head ] ~i:[ d_model ] ~of_hf:(qkv_w 0) ()
  in
  let w_k =
    w ~name:(p "attn.w_k") ~b:[] ~o:[ n_heads; d_head ] ~i:[ d_model ] ~of_hf:(qkv_w 1) ()
  in
  let w_v =
    w ~name:(p "attn.w_v") ~b:[] ~o:[ n_heads; d_head ] ~i:[ d_model ] ~of_hf:(qkv_w 2) ()
  in
  let b_q =
    w ~name:(p "attn.b_q") ~scale:0.0 ~b:[] ~o:[ n_heads; d_head ] ~i:[] ~of_hf:(qkv_b 0) ()
  in
  let b_k =
    w ~name:(p "attn.b_k") ~scale:0.0 ~b:[] ~o:[ n_heads; d_head ] ~i:[] ~of_hf:(qkv_b 1) ()
  in
  let b_v =
    w ~name:(p "attn.b_v") ~scale:0.0 ~b:[] ~o:[ n_heads; d_head ] ~i:[] ~of_hf:(qkv_b 2) ()
  in
  let w_o =
    w ~name:(p "attn.c_proj.weight") ~b:[] ~o:[ d_model ] ~i:[ n_heads; d_head ]
      ~of_hf:(fun st ->
        let src = Safetensors.to_float32 st (p "attn.c_proj.weight") in
        permuted src ~dims:[ d_model; n_heads; d_head ] ~f:(fun idx ->
            [| (idx.(1) * d_head) + idx.(2); idx.(0) |]))
      ()
  in
  let b_o =
    w ~name:(p "attn.c_proj.bias") ~scale:0.0 ~b:[] ~o:[ d_model ] ~i:[]
      ~of_hf:(hf_id (p "attn.c_proj.bias"))
      ()
  in
  let w_fc =
    w ~name:(p "mlp.c_fc.weight") ~b:[] ~o:[ d_ff ] ~i:[ d_model ]
      ~of_hf:(hf_transpose (p "mlp.c_fc.weight"))
      ()
  in
  let b_fc =
    w ~name:(p "mlp.c_fc.bias") ~scale:0.0 ~b:[] ~o:[ d_ff ] ~i:[]
      ~of_hf:(hf_id (p "mlp.c_fc.bias"))
      ()
  in
  let w_proj =
    w ~name:(p "mlp.c_proj.weight") ~b:[] ~o:[ d_model ] ~i:[ d_ff ]
      ~of_hf:(hf_transpose (p "mlp.c_proj.weight"))
      ()
  in
  let b_proj =
    w ~name:(p "mlp.c_proj.bias") ~scale:0.0 ~b:[] ~o:[ d_model ] ~i:[]
      ~of_hf:(hf_id (p "mlp.c_proj.bias"))
      ()
  in
  fun ~mask x ->
    let x =
      [%op
        x
        + attention ~n_heads ~d_head () ~w_q ~b_q ~w_k ~b_k ~w_v ~b_v ~w_o ~b_o ~mask
            (layer_norm ~epsilon () ~gamma:ln_1_g ~beta:ln_1_b x)]
    in
    [%op
      x + ffn () ~w_fc ~b_fc ~w_proj ~b_proj (layer_norm ~epsilon () ~gamma:ln_2_g ~beta:ln_2_b x)]

(** Build the full model for a fixed [seq_len]. Returns a function from a token-ID tensor (as
    produced by {!Nb.token_ids_of_array}: a [seq_len] batch of uint32 IDs) to the logits tensor
    (batch [seq_len], output [vocab_size]). *)
let gpt2 ~config ~src ~seq_len () =
  let { vocab_size; n_positions; d_model; n_layers; epsilon; _ } = config in
  assert (seq_len <= n_positions);
  let w = weight src in
  (* Embedding table as [d_model (output); vocab (input)]: transposed wte. *)
  let wte_embed =
    w ~name:"wte_embed" ~b:[] ~o:[ d_model ] ~i:[ vocab_size ] ~of_hf:(hf_transpose "wte.weight") ()
  in
  (* Tied LM head as [vocab (output); d_model (input)]: HF wte layout as-is. *)
  let lm_head =
    w ~name:"lm_head" ~b:[] ~o:[ vocab_size ] ~i:[ d_model ] ~of_hf:(hf_id "wte.weight") ()
  in
  let pos_embed =
    w ~name:"wpe" ~b:[ seq_len ] ~o:[ d_model ] ~i:[]
      ~of_hf:(fun st ->
        let src = Safetensors.to_float32 st "wpe.weight" in
        Bigarray.Genarray.sub_left src 0 seq_len)
      ()
  in
  let ln_f_g =
    w ~name:"ln_f.weight" ~center:1.0 ~scale:0.0 ~b:[] ~o:[ d_model ] ~i:[]
      ~of_hf:(hf_id "ln_f.weight") ()
  in
  let ln_f_b =
    w ~name:"ln_f.bias" ~center:0.0 ~scale:0.0 ~b:[] ~o:[ d_model ] ~i:[] ~of_hf:(hf_id "ln_f.bias")
      ()
  in
  let blocks = List.init n_layers ~f:(fun layer -> block ~config ~src ~layer) in
  let mask =
    NTDSL.init ~l:"causal_mask" ~prec:Ir.Ops.single ~b:[ seq_len ] ~i:[ seq_len ] ~o:[]
      ~f:(function
        | [| s; t |] -> if s >= t then 1. else 0. | _ -> failwith "gpt2: unexpected mask indices")
      ()
  in
  fun ids ->
    let one_hot = Nb.one_hot_of_ids ~num_classes:vocab_size ids in
    let x = [%op (wte_embed * one_hot) + pos_embed] in
    let x = List.fold blocks ~init:x ~f:(fun x blk -> blk ~mask x) in
    let x = layer_norm ~epsilon () ~gamma:ln_f_g ~beta:ln_f_b x in
    [%op lm_head * x]
