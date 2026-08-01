(* OCANNL GPT-2-style inference runner: pre-LN decoder blocks built from the idiomatic nn_blocks
   pieces (multi_head_attention, layer_norm — fixture weights injected by name) and a tanh-gelu FFN
   from fixture-wrapped weight tensors. Token embedding is the logical one-hot gather (gh-343); the
   lm_head is tied to wte via an einsum that reads it transposed. Forward-only: the parity metric is
   softmax-CE of the logits against fixture target ids, recorded per batch with no updates. Layouts
   documented in gen_fixtures.py build_gpt. *)

open Base
open Ocannl
module IDX = Train.IDX
open Nn_blocks.DSL_modules
module St = Safetensors
module H = Bench_harness

let cross_entropy_loss = Nn_blocks.cross_entropy_loss
let gelu = Nn_blocks.gelu

let ids_of_gen g =
  let dims = Bigarray.Genarray.dims g in
  Array.init dims.(0) ~f:(fun s ->
      Array.init dims.(1) ~f:(fun i -> Int.of_float (Bigarray.Genarray.get g [| s; i |])))

(* Token ids as a uint32 tensor with explicit [n_batches; batch_size; seq] batch dims, so that [@|
   batch_n] indexes the leading batch axis. *)
let ids_tensor ~label ints ~n_batches ~batch_size ~seq =
  let open Bigarray in
  let g = Genarray.create Int32 c_layout [| n_batches; batch_size; seq |] in
  Array.iteri ints ~f:(fun idx row ->
      let b = idx / batch_size and s = idx % batch_size in
      Array.iteri row ~f:(fun t id -> Genarray.set g [| b; s; t |] (Int32.of_int_trunc id)));
  TDSL.wrap ~l:label ~b:[ n_batches; batch_size; seq ] ~o:[]
    (Ir.Ndarray.as_array Ir.Ops.Uint32 g)
    ()

let () =
  let fixture = Stdlib.Sys.getenv "BENCH_FIXTURE" in
  let tune = H.env_flag "BENCH_TUNE" in
  let materialize = H.env_flag "BENCH_MATERIALIZE" in
  let debug = H.env_flag "BENCH_DEBUG" in
  (* BENCH_PRECISION=bf16|f16 (gh-ocannl-492 task 4, the forward-only leg): reduced precision
     enters by LOAD-TIME CONVERSION, not cast twins — inference has no optimizer, so there is no
     master copy for a twin to preserve, and keeping f32 storage would pay exactly the weight
     bandwidth the leg measures away (torch's [model.half ()]). Data-backed weights (wte/wpe/ffn)
     convert at wrap; the attention-projection params take the reduced precision through the
     storage policy and [H.inject]'s [set_values] converts the f32 fixture values at load;
     layer-norm gains/biases and the softmax-CE head stay f32 (the AMP default). No loss scaling:
     there are no gradients. *)
  let precision, mp_prec =
    match Stdlib.Sys.getenv_opt "BENCH_PRECISION" with
    | None | Some "" | Some "0" | Some "f32" -> ("f32", None)
    | Some "bf16" -> ("bf16", Some Ir.Ops.bfloat16)
    | Some "f16" -> ("f16", Some Ir.Ops.half)
    | Some other -> failwith ("bench_gpt: unknown BENCH_PRECISION: " ^ other)
  in
  let st = St.read fixture in
  let batch_size = H.meta_int st "batch_size" in
  let n_layer = H.meta_int st "n_layer" in
  let n_head = H.meta_int st "n_head" in
  let d_model = H.meta_int st "d_model" in
  let vocab = H.meta_int st "vocab" in
  let seq = H.meta_int st "seq_len" in
  let d_head = d_model / n_head in
  let ids_all = ids_of_gen (St.to_float32 st "ids") in
  let tgt_all = ids_of_gen (St.to_float32 st "tgt") in
  let total = Array.length ids_all in
  let n_batches = total / batch_size in
  let ids_t = ids_tensor ~label:"ids" ids_all ~n_batches:(total / batch_size) ~batch_size ~seq in
  let tgt_t = ids_tensor ~label:"tgt" tgt_all ~n_batches:(total / batch_size) ~batch_size ~seq in
  let batch_n, bindings = IDX.get_static_symbol ~static_range:n_batches IDX.empty in
  let%op ids_b = ids_t @| batch_n in
  let%op tgt_b = tgt_t @| batch_n in
  let wrap name ~i ~o = TDSL.wrap ~l:name ?prec:mp_prec ~b:[] ~i ~o (St.to_ndarray st name) () in
  let wte = wrap "wte" ~i:[ vocab ] ~o:[ d_model ] in
  let wpe =
    TDSL.wrap ~l:"wpe" ?prec:mp_prec ~b:[ seq ] ~i:[] ~o:[ d_model ] (St.to_ndarray st "wpe") ()
  in
  let mask =
    NTDSL.init ~l:"mask" ~prec:Ir.Ops.single ~b:[ seq ] ~i:[ seq ] ~o:[]
      ~f:(function [| s; t |] -> if s >= t then 1. else 0. | _ -> assert false)
      ()
  in
  let onehot_x = Nn_blocks.one_hot_of_ids ~num_classes:vocab ids_b in
  let%op embedded = (wte * onehot_x) + wpe in
  let layers =
    List.init n_layer ~f:(fun i ->
        let name fmt = Printf.sprintf fmt i in
        let lbl = Printf.sprintf "l%d" i in
        let mha =
          Nn_blocks.multi_head_attention ~label:[ lbl ] ~num_heads:n_head ~d_k:d_head ~d_v:d_head ()
        in
        let ln1 = Nn_blocks.layer_norm ~label:[ "ln1"; lbl ] () in
        let ln2 = Nn_blocks.layer_norm ~label:[ "ln2"; lbl ] () in
        let fw1 = wrap (name "l%d_ffn_w1") ~i:[ d_model ] ~o:[ H.meta_int st "d_ff" ] in
        let fb1 = wrap (name "l%d_ffn_b1") ~i:[] ~o:[ H.meta_int st "d_ff" ] in
        let fw2 = wrap (name "l%d_ffn_w2") ~i:[ H.meta_int st "d_ff" ] ~o:[ d_model ] in
        let fb2 = wrap (name "l%d_ffn_b2") ~i:[] ~o:[ d_model ] in
        fun x ->
          let%op x1 = x + mha ~train_step:None ~mask (ln1 x) in
          let%op x2 = x1 + ((fw2 * gelu ((fw1 * ln2 x1) + fb1)) + fb2) in
          x2)
  in
  let lnf = Nn_blocks.layer_norm ~label:[ "lnf" ] () in
  let hfinal = lnf (List.fold layers ~init:embedded ~f:(fun x layer -> layer x)) in
  (* Tied lm_head: read wte transposed (contract d, produce v). *)
  let%op logits = wte +* "|v -> d; ... | d => ... | v" hfinal in
  let targets = Nn_blocks.one_hot_of_ids ~num_classes:vocab tgt_b in
  let n_positions = batch_size * seq in
  let%op batch_loss =
    cross_entropy_loss ~spec:"...|v" ~normalize_by:!..n_positions () ~logits ~targets
  in
  (* Storage policy for the reduced-precision leg: the decoder body (the logits subtree) computes
     in the reduced precision — attention-projection params included ([param_prec]; injection
     converts the fixture values at load) — while layer-norm gains/biases and the softmax-CE head
     are PINNED at f32 via [except] (merely not assigning them is not enough: precision inference
     would pull them to the reduced precision from their neighbors). *)
  Option.iter mp_prec ~f:(fun prec ->
      let body = Hash_set.create (module Int) in
      let rec walk t =
        if not (Hash_set.mem body t.Tensor.value.Ir.Tnode.id) then (
          Hash_set.add body t.Tensor.value.Ir.Tnode.id;
          List.iter t.Tensor.children ~f:(fun c -> walk c.Tensor.subtensor))
      in
      walk logits;
      let is_ln tn =
        List.exists tn.Ir.Tnode.label ~f:(fun l ->
            String.equal l "gamma" || String.equal l "beta")
      in
      let except tn = (not (Hash_set.mem body tn.Ir.Tnode.id)) || is_ln tn in
      Precision_policy.apply ~except
        { Precision_policy.param_prec = Some prec; activation_prec = Some prec; grad_prec = None }
        batch_loss);
  if materialize then Train.every_non_literal_materialized batch_loss;
  let fwd = Train.forward batch_loss in
  let ctx = Context.auto () in
  let backend = Context.backend_name ctx in
  let ctx = Train.init_params ctx bindings batch_loss in
  if debug then (
    H.dump_params batch_loss;
    Stdlib.exit 0);
  let mapping =
    ("lnf_g", [ "gamma"; "lnf" ])
    :: ("lnf_b", [ "beta"; "lnf" ])
    :: List.concat_map (List.range 0 n_layer) ~f:(fun i ->
        let l = Printf.sprintf "l%d" i in
        [
          (Printf.sprintf "l%d_wq" i, [ l; "w"; "q" ]);
          (Printf.sprintf "l%d_wk" i, [ l; "w"; "k" ]);
          (Printf.sprintf "l%d_wv" i, [ l; "w"; "v" ]);
          (Printf.sprintf "l%d_wo" i, [ l; "w"; "o" ]);
          (Printf.sprintf "l%d_ln1_g" i, [ l; "gamma"; "ln1" ]);
          (Printf.sprintf "l%d_ln1_b" i, [ l; "beta"; "ln1" ]);
          (Printf.sprintf "l%d_ln2_g" i, [ l; "gamma"; "ln2" ]);
          (Printf.sprintf "l%d_ln2_b" i, [ l; "beta"; "ln2" ]);
        ])
  in
  let ctx = H.inject ctx st batch_loss mapping in
  let t0 = Unix.gettimeofday () in
  let ctx, routine =
    if tune then
      let scratch = Train.init_params (Context.auto ()) bindings batch_loss in
      (* Placement A/B: tune the default (virtual + promotion) graph and the materialize-all graph,
         keep the measured winner. *)
      Train.tune_placements ~rounds:0 ~timing_ctx:scratch ctx batch_loss fwd bindings
    else if Lazy.force Autotune.model_default_enabled then
      (* gh-ocannl-491: the model-picked untuned default (config [model_default_schedule=true]). *)
      Autotune.model_default ctx fwd bindings
    else Context.compile ctx fwd bindings
  in
  let compile_s = Unix.gettimeofday () -. t0 in
  let ctx = if tune then H.inject ctx st batch_loss mapping else ctx in
  let batch_ref = IDX.find_exn (Context.bindings routine) batch_n in
  let step_count = ref 0 in
  let run_step () =
    batch_ref := !step_count % n_batches;
    Train.run ctx routine;
    Int.incr step_count
  in
  let open Operation.At in
  H.measure_and_emit ~st ~backend
    ~variant:
      (* Mirror bench_mlp: the scheduling variant alone. orchestrate renders precision as its own
         report column and composes the two axes itself (gh-ocannl-539), so a reduced-precision
         cell is distinguished by the precision field rather than by overloading this one. *)
      (if tune then "tuned" else if materialize then "materialized" else "default")
    ~precision ~compile_s ~tokens_per_step:(batch_size * seq) ~run_step
    ~read_loss:(fun () -> (ctx, batch_loss).@[0])
    ~sync:(fun () -> Context.sync ctx)
    ()
