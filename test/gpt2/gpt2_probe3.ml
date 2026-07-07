(* Minimal repro hunt: the GPT-2 attention pattern at tiny dimensions with known values.
   Three matmuls share one operand; in the full graph the third one (v) evaluated as zero.
   Compare each stage against hand-computable expectations. Not a test. *)

open Base
open Ocannl
open Stdio
open Nn_blocks.DSL_modules

let () =
  let seq = 3 and dm = 4 and heads = 2 and dh = 2 in
  (* x[t][m] = 0.1*(t+1) if m = t mod 4 else 0.05 -- arbitrary but position-dependent. *)
  let x =
    NTDSL.init ~l:"x" ~prec:Ir.Ops.single ~b:[ seq ] ~i:[] ~o:[ dm ]
      ~f:(function
        | [| t; m |] -> if m = t % dm then 0.1 *. Float.of_int (t + 1) else 0.05
        | _ -> assert false)
      ()
  in
  (* Weights: w?[h][j][m] distinct per matrix. *)
  let mk_w name c =
    NTDSL.init ~l:name ~prec:Ir.Ops.single ~b:[] ~i:[ dm ] ~o:[ heads; dh ]
      ~f:(function
        | [| h; j; m |] -> c *. (1. +. (0.1 *. Float.of_int ((h * 2) + j)) +. (0.01 *. Float.of_int m))
        | _ -> assert false)
      ()
  in
  let mk_b name c =
    NTDSL.init ~l:name ~prec:Ir.Ops.single ~b:[] ~i:[] ~o:[ heads; dh ]
      ~f:(function [| h; j |] -> c *. Float.of_int ((h * 2) + j + 1) | _ -> assert false)
      ()
  in
  let wq = mk_w "wq" 0.1 and bq = mk_b "bq" 0.01 in
  let wk = mk_w "wk" 0.2 and bk = mk_b "bk" 0.02 in
  let wv = mk_w "wv" 0.3 and bv = mk_b "bv" 0.03 in
  let mask =
    NTDSL.init ~l:"m" ~prec:Ir.Ops.single ~b:[ seq ] ~i:[ seq ] ~o:[]
      ~f:(function [| s; t |] -> if s >= t then 1. else 0. | _ -> assert false)
      ()
  in
  let%op q = (wq * x) + bq in
  let%op k = (wk * x) + bk in
  let%op v = (wv * x) + bv in
  let%op scores =
    (q +* k " ... s | h d; ... t | h d => ... s | t -> h" [ "h"; "d" ]) /. sqrt (dim d)
  in
  Shape.set_dim h heads;
  Shape.set_dim d dh;
  let%op weights = Nn_blocks.softmax ~spec:" ... | t -> ..." () (where mask scores !.(-1e9)) in
  let%op preproj = weights +* v " ... s | t -> h; ... t | h e => ... s | h e" [ "e" ] in
  Shape.set_dim e dh;
  Train.set_materialized v.Tensor.value;
  Train.set_materialized q.Tensor.value;
  let ctx = Train.forward_once (Context.auto ()) preproj in
  let p name t width =
    let vals = Context.get_values ctx t.Tensor.value in
    printf "%s:" name;
    Array.iteri vals ~f:(fun i v -> if i < width then printf " %.6f" v);
    printf "\n%!"
  in
  p "q[all]" q (seq * heads * dh);
  p "v[all]" v (seq * heads * dh);
  p "preproj" preproj (seq * heads * dh)

(* Same tiny attention, but x goes through layer_norm (stage b) and then also through the one-hot
   embedding gather (stage c), mirroring the real pipeline. In the failing full model, v evaluated
   to broadcast bias only (matmul contribution zeroed). *)
let tiny_attention ~tag x =
  let seq = 3 and dm = 4 and heads = 2 and dh = 2 in
  ignore dm;
  let mk_w name c =
    NTDSL.init ~l:(name ^ tag) ~prec:Ir.Ops.single ~b:[] ~i:[ 4 ] ~o:[ heads; dh ]
      ~f:(function
        | [| h; j; m |] ->
            c *. (1. +. (0.1 *. Float.of_int ((h * 2) + j)) +. (0.01 *. Float.of_int m))
        | _ -> assert false)
      ()
  in
  let mk_b name c =
    NTDSL.init ~l:(name ^ tag) ~prec:Ir.Ops.single ~b:[] ~i:[] ~o:[ heads; dh ]
      ~f:(function [| h; j |] -> c *. Float.of_int ((h * 2) + j + 1) | _ -> assert false)
      ()
  in
  let wq = mk_w "wq" 0.1 and bq = mk_b "bq" 0.01 in
  let wk = mk_w "wk" 0.2 and bk = mk_b "bk" 0.02 in
  let wv = mk_w "wv" 0.3 and bv = mk_b "bv" 0.03 in
  ignore bv;
  let mask =
    NTDSL.init ~l:("m" ^ tag) ~prec:Ir.Ops.single ~b:[ seq ] ~i:[ seq ] ~o:[]
      ~f:(function [| s; t |] -> if s >= t then 1. else 0. | _ -> assert false)
      ()
  in
  let%op q = (wq * x) + bq in
  let%op k = (wk * x) + bk in
  let%op v = (wv * x) + bv in
  let%op scores =
    (q +* k " ... s | h d; ... t | h d => ... s | t -> h" [ "h"; "d" ]) /. sqrt (dim d)
  in
  Shape.set_dim h heads;
  Shape.set_dim d dh;
  let%op weights = Nn_blocks.softmax ~spec:" ... | t -> ..." () (where mask scores !.(-1e9)) in
  let%op preproj = weights +* v " ... s | t -> h; ... t | h e => ... s | h e" [ "e" ] in
  Shape.set_dim e dh;
  Train.set_materialized v.Tensor.value;
  let ctx = Train.forward_once (Context.auto ()) preproj in
  let p name t width =
    let vals = Context.get_values ctx t.Tensor.value in
    printf "%s%s:" name tag;
    Array.iteri vals ~f:(fun i v -> if i < width then printf " %.6f" v);
    printf "\n%!"
  in
  p "v" v (seq * heads * dh);
  p "preproj" preproj (seq * heads * dh)

let () =
  let seq = 3 and dm = 4 in
  (* Stage b: layer_norm-fed attention. *)
  let x0 =
    NTDSL.init ~l:"x0_b" ~prec:Ir.Ops.single ~b:[ seq ] ~i:[] ~o:[ dm ]
      ~f:(function
        | [| t; m |] -> if m = t % dm then 0.1 *. Float.of_int (t + 1) else 0.05
        | _ -> assert false)
      ()
  in
  let g =
    NTDSL.init ~l:"g_b" ~prec:Ir.Ops.single ~b:[] ~i:[] ~o:[ dm ]
      ~f:(fun _ -> 1.)
      ()
  in
  let bta =
    NTDSL.init ~l:"bta_b" ~prec:Ir.Ops.single ~b:[] ~i:[] ~o:[ dm ]
      ~f:(fun _ -> 0.)
      ()
  in
  tiny_attention ~tag:"_b" (Gpt2_model.layer_norm ~epsilon:1e-5 () ~gamma:g ~beta:bta x0);

  (* Stage c: one-hot embedding gather + positional add + layer_norm, like the real model. *)
  let vocab = 7 in
  let ids = Nn_blocks.token_ids_of_array [| 2; 5; 1 |] in
  let one_hot = Nn_blocks.one_hot_of_ids ~num_classes:vocab ids in
  let wte =
    NTDSL.init ~l:"wte_c3" ~prec:Ir.Ops.single ~b:[] ~i:[ vocab ] ~o:[ dm ]
      ~f:(function
        | [| e; vv |] -> (0.1 *. Float.of_int (vv + 1)) +. (0.01 *. Float.of_int e)
        | _ -> assert false)
      ()
  in
  let wpe =
    NTDSL.init ~l:"wpe_c3" ~prec:Ir.Ops.single ~b:[ seq ] ~i:[] ~o:[ dm ]
      ~f:(function
        | [| t; e |] -> 0.02 *. Float.of_int ((t * 4) + e)
        | _ -> assert false)
      ()
  in
  let g =
    NTDSL.init ~l:"g_c3" ~prec:Ir.Ops.single ~b:[] ~i:[] ~o:[ dm ] ~f:(fun _ -> 1.) ()
  in
  let bta =
    NTDSL.init ~l:"bta_c3" ~prec:Ir.Ops.single ~b:[] ~i:[] ~o:[ dm ] ~f:(fun _ -> 0.) ()
  in
  let embed = [%op (wte * one_hot) + wpe] in
  tiny_attention ~tag:"_c" (Gpt2_model.layer_norm ~epsilon:1e-5 () ~gamma:g ~beta:bta embed)

(* Ultra-minimal: three matmuls consuming one layer_norm output, no attention machinery. *)
let () =
  let seq = 3 and dm = 4 and heads = 2 and dh = 2 in
  let x0 =
    NTDSL.init ~l:"x0_d" ~prec:Ir.Ops.single ~b:[ seq ] ~i:[] ~o:[ dm ]
      ~f:(function
        | [| t; m |] -> if m = t % dm then 0.1 *. Float.of_int (t + 1) else 0.05
        | _ -> assert false)
      ()
  in
  let g = NTDSL.init ~l:"g_d" ~prec:Ir.Ops.single ~b:[] ~i:[] ~o:[ dm ] ~f:(fun _ -> 1.) () in
  let bta = NTDSL.init ~l:"bta_d" ~prec:Ir.Ops.single ~b:[] ~i:[] ~o:[ dm ] ~f:(fun _ -> 0.) () in
  let mk_w name c =
    NTDSL.init ~l:name ~prec:Ir.Ops.single ~b:[] ~i:[ dm ] ~o:[ heads; dh ]
      ~f:(function
        | [| h; j; m |] ->
            c *. (1. +. (0.1 *. Float.of_int ((h * 2) + j)) +. (0.01 *. Float.of_int m))
        | _ -> assert false)
      ()
  in
  let wq = mk_w "wq_d" 0.1 and wk = mk_w "wk_d" 0.2 and wv = mk_w "wv_d" 0.3 in
  let l = Gpt2_model.layer_norm ~epsilon:1e-5 () ~gamma:g ~beta:bta x0 in
  let%op q = wq * l in
  let%op k = wk * l in
  let%op v = wv * l in
  Train.set_materialized q.Tensor.value;
  Train.set_materialized k.Tensor.value;
  Train.set_materialized v.Tensor.value;
  (* Root the three results in one graph. *)
  let%op total = (q ++ "...|... => 0") + (k ++ "...|... => 0") + (v ++ "...|... => 0") in
  let ctx = Train.forward_once (Context.auto ()) total in
  let p name t =
    let vals = Context.get_values ctx t.Tensor.value in
    printf "%s_d:" name;
    Array.iter vals ~f:(fun v -> printf " %.6f" v);
    printf "\n%!"
  in
  p "q" q;
  p "k" k;
  p "v" v
