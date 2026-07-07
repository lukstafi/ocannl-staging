(* Regression test for forward-code fragment ordering (computed-after-use): three matmuls
   consume one shared layer_norm output. The layer_norm chain is embedded in the forward code of
   its first-constructed consumer (q), while sibling subtree ids follow construction order -- which
   for `(f q + f k) + f v` is v, k, q (OCaml applications evaluate right-to-left). Before the
   topological ordering of forward fragments in Tensor.raw, the id-ascending emission ran the k and
   v matmuls before the layer_norm chain, so they read zeros: k and v printed all zeros. Correct
   output: k = 2*q and v = 3*q (the weights are scaled copies), with q nonzero. *)
open Base
open Ocannl
open Stdio
open Nn_blocks.DSL_modules

let%op layer_norm ~epsilon () ~gamma ~beta x =
  let mean = (x ++ " ... | ..d.. => ... | 0 " [ "d" ]) /. dim d in
  let centered = x - mean in
  let variance = ((centered *. centered) ++ " ... | ... => ... | 0 ") /. dim d in
  (gamma *. (centered /. sqrt (variance + !.epsilon))) + beta

let () =
  let seq = 3 and dm = 4 and heads = 2 and dh = 2 in
  let x0 =
    NTDSL.init ~l:"x0" ~prec:Ir.Ops.single ~b:[ seq ] ~i:[] ~o:[ dm ]
      ~f:(function
        | [| t; m |] -> if m = t % dm then 0.1 *. Float.of_int (t + 1) else 0.05
        | _ -> assert false)
      ()
  in
  let g = NTDSL.init ~l:"g" ~prec:Ir.Ops.single ~b:[] ~i:[] ~o:[ dm ] ~f:(fun _ -> 1.) () in
  let bta = NTDSL.init ~l:"bta" ~prec:Ir.Ops.single ~b:[] ~i:[] ~o:[ dm ] ~f:(fun _ -> 0.) () in
  let mk_w name c =
    NTDSL.init ~l:name ~prec:Ir.Ops.single ~b:[] ~i:[ dm ] ~o:[ heads; dh ]
      ~f:(function
        | [| h; j; m |] ->
            c *. (1. +. (0.1 *. Float.of_int ((h * 2) + j)) +. (0.01 *. Float.of_int m))
        | _ -> assert false)
      ()
  in
  let wq = mk_w "wq" 0.1 and wk = mk_w "wk" 0.2 and wv = mk_w "wv" 0.3 in
  let l = layer_norm ~epsilon:1e-5 () ~gamma:g ~beta:bta x0 in
  let%op q = wq * l in
  let%op k = wk * l in
  let%op v = wv * l in
  Train.set_materialized q.Tensor.value;
  Train.set_materialized k.Tensor.value;
  Train.set_materialized v.Tensor.value;
  let%op total = (q ++ "...|... => 0") + (k ++ "...|... => 0") + (v ++ "...|... => 0") in
  let ctx = Train.forward_once (Context.auto ()) total in
  let p name t =
    let vals = Context.get_values ctx t.Tensor.value in
    printf "%s:" name;
    Array.iter vals ~f:(fun v -> printf " %.6f" v);
    printf "\n%!"
  in
  p "q" q;
  p "k" k;
  p "v" v
