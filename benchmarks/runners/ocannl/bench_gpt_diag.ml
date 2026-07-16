(* Diagnostic for the gpt2_mini forward step's default schedule: prints the fission segment census
   (kinds, launch geometry, statement counts) on the configured backend, then times a few individual
   steps. Not part of the benchmark protocol; run manually. *)

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
  let materialize = H.env_flag "BENCH_MATERIALIZE" in
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
  let ids_t = ids_tensor ~label:"ids" ids_all ~n_batches ~batch_size ~seq in
  let tgt_t = ids_tensor ~label:"tgt" tgt_all ~n_batches ~batch_size ~seq in
  let batch_n, bindings = IDX.get_static_symbol ~static_range:n_batches IDX.empty in
  let%op ids_b = ids_t @| batch_n in
  let%op tgt_b = tgt_t @| batch_n in
  let wrap name ~i ~o = TDSL.wrap ~l:name ~b:[] ~i ~o (St.to_ndarray st name) () in
  let wte = wrap "wte" ~i:[ vocab ] ~o:[ d_model ] in
  let wpe = TDSL.wrap ~l:"wpe" ~b:[ seq ] ~i:[] ~o:[ d_model ] (St.to_ndarray st "wpe") () in
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
  let%op logits = wte +* "|v -> d; ... | d => ... | v" hfinal in
  let targets = Nn_blocks.one_hot_of_ids ~num_classes:vocab tgt_b in
  let n_positions = batch_size * seq in
  let%op batch_loss =
    cross_entropy_loss ~spec:"...|v" ~normalize_by:!..n_positions () ~logits ~targets
  in
  ignore tgt_t;
  if materialize then Train.every_non_literal_materialized batch_loss;
  let fwd = Train.forward batch_loss in
  let ctx = Context.auto () in
  let backend = Context.backend_name ctx in
  let limits = Context.hardware_limits ctx in
  let ctx = Train.init_params ctx bindings batch_loss in
  (* First compile only stashes the lowered code for the census (an explicit transform replaces the
     default pipeline, so this routine is discarded); the timed routine is compiled with the regular
     default pipeline below. *)
  let stash = ref None in
  let _census_ctx, _census_routine =
    Context.compile
      ~lowered_transform:(fun opt ->
        stash := Some opt;
        opt)
      ctx fwd bindings
  in
  let opt = Option.value_exn ~here:[%here] !stash in
  let promote_locals =
    match Stdlib.Sys.getenv_opt "BENCH_PROMOTE" with Some "0" -> Some false | _ -> None
  in
  H.print_census ?promote_locals ~backend ~limits ~static_indices:[ batch_n ] opt;
  let t0 = Unix.gettimeofday () in
  let ctx, routine = Context.compile ctx fwd bindings in
  let compile_s = Unix.gettimeofday () -. t0 in
  Stdio.printf "backend: %s  compile_s: %.3f\n" backend compile_s;
  (* Time a few individual steps with a full sync each. *)
  if H.env_flag "BENCH_STEPS" then
    let batch_ref = IDX.find_exn (Context.bindings routine) batch_n in
    for step = 0 to 2 do
      batch_ref := step % n_batches;
      let t0 = Unix.gettimeofday () in
      Train.run ctx routine;
      Context.sync ctx;
      Stdio.printf "step %d: %.1f ms\n" step ((Unix.gettimeofday () -. t0) *. 1000.);
      Stdio.Out_channel.flush Stdio.stdout
    done
