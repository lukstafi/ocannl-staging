(* Warp-shuffle rendering of [Workgroup_reduce] accumulation loops (gh-ocannl-462, llm.c's
   warpReduce/blockReduce idiom): executed parity of shuffle-rendered reductions against
   OCaml-computed references, plus structural checks on the generated source.

   Unlike test/operations/hardware_workgroup_reduce.ml, which stages the tree reduction explicitly
   (shared tile + barriers), here the [?lowered_transform] replaces the lowered serial reduction
   with a single [Workgroup_reduce]-typed loop whose body stays the plain accumulation statement
   [acc = op(acc, contrib(i))] — the renderer owns the communication. On GPU backends (Metal
   locally, CUDA in CI) that renders as the two-phase warp-shuffle pattern ([ocannl_shfl_xor] tree
   within each warp, one shared slot per warp, barrier, first-warp combine); on the C backends the
   same body legally renders as the ordinary serial loop, so every printed boolean holds on every
   backend.

   Covered: a 4-warp sum (two-phase, with the shared per-warp staging), a single-warp FMA
   dot-product (pure shuffles, no staging), a 2-warp max-reduce (non-Add combine), and the clean
   rejection of a recognized accumulation whose extent does not cover whole warps (GPU) vs. its
   serial execution (CPU). *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module Tn = Ir.Tnode
module LL = Ir.Low_level
module Asgns = Ir.Assignments
module Idx = Ir.Indexing

let () = Utils.settings.output_debug_files_in_build_directory <- true
let p name b = Stdio.printf "%s: %b\n" name b
let approx a b = Float.(abs (a - b) < 1e-3)
let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")

let on_gpu =
  String.is_substring backend_name ~substring:"metal"
  || String.is_substring backend_name ~substring:"cuda"
  || String.is_substring backend_name ~substring:"hip"

let single = Ir.Ops.single

let read_generated base_name =
  let ext = if String.is_substring backend_name ~substring:"metal" then ".metal" else ".c" in
  let ext = if String.is_substring backend_name ~substring:"cuda" then ".cu" else ext in
  let ext = if String.is_substring backend_name ~substring:"hip" then ".hip" else ext in
  let path = Utils.build_file (base_name ^ ext) in
  if Stdlib.Sys.file_exists path then Some (Stdio.In_channel.read_all path) else None

let named name (comp : Asgns.comp) : Asgns.comp =
  { comp with asgns = Asgns.Block_comment (name, comp.asgns) }

(* Replace the lowered serial reduction with a single [Workgroup_reduce] accumulation loop over a
   fresh index; the renderer owns the communication. The transform drops the lowered [Zero_out] of
   the accumulator, so un-mark [zero_initialized_by_code]: allocation then zeroes the buffer, giving
   the accumulation the same all-zeros starting point as the serial lowering. *)
let reduce_transform ~n ~body_of (s : Tn.t) (opt : LL.optimized) : LL.optimized =
  (LL.get_node opt.traced_store s).LL.zero_initialized_by_code <- false;
  let i = Idx.get_symbol () in
  {
    opt with
    llc =
      LL.For_loop
        {
          index = i;
          from_ = 0;
          to_ = n - 1;
          body = body_of i;
          axis = Workgroup_reduce;
        };
  }

let run ~name ~transform t =
  let comp = named name (Train.forward t) in
  let ctx = Context.auto () in
  let ctx, routine = Context.compile ~lowered_transform:transform ctx comp Ir.Indexing.Empty in
  let ctx = Context.run ctx routine in
  (Context.get_values ctx t.Tensor.value).(0)

let it i = Idx.Iterator i
let f0 = Idx.Fixed_idx 0

let () =
  (* --- Two-phase sum: 128 = 4 warps of 32. --- *)
  let n = 128 in
  let vv = Array.init n ~f:(fun k -> (Float.of_int (k % 21) *. 0.25) -. 2.) in
  let expected_sum = Array.fold vv ~init:0. ~f:( +. ) in
  let v = TDSL.ndarray vv ~label:[ "v" ] ~output_dims:[ n ] () in
  let%op s0 = v ++ "i=>0" in
  let got_serial = run ~name:"wshfl_sum_serial" ~transform:(fun opt -> opt) s0 in
  p "serial sum correct" (approx got_serial expected_sum);
  let%op s1 = v ++ "i=>0" in
  let got =
    run ~name:"sum_wshfl"
      ~transform:
        (reduce_transform ~n s1.Tensor.value ~body_of:(fun i ->
             LL.Set
               {
                 tn = s1.Tensor.value;
                 idcs = [| f0 |];
                 llsc =
                   Binop
                     ( Ir.Ops.Add,
                       (Get (s1.Tensor.value, [| f0 |]), single),
                       (Get (v.Tensor.value, [| it i |]), single) );
                 debug = "";
               }))
      s1
  in
  p "warp-shuffle sum parity" (approx got expected_sum);
  (match read_generated "sum_wshfl" with
  | None -> p "two-phase shuffle rendering (GPU) or serial fallback (CPU)" false
  | Some src ->
      let has sub = String.is_substring src ~substring:sub in
      let ok =
        if on_gpu then has "ocannl_shfl_xor" && has "wred_partials_"
        else (not (has "ocannl_shfl_xor")) && not (has "wred_partials_")
      in
      p "two-phase shuffle rendering (GPU) or serial fallback (CPU)" ok);

  (* --- Single-warp FMA dot-product: 32 = 1 warp (no staging, no barrier). --- *)
  let m = 32 in
  let av = Array.init m ~f:(fun k -> (Float.of_int (k % 7) *. 0.5) -. 1.) in
  let bv = Array.init m ~f:(fun k -> Float.of_int (k % 5) -. 2.) in
  let expected_dot = Array.fold2_exn av bv ~init:0. ~f:(fun acc a b -> acc +. (a *. b)) in
  let va = TDSL.ndarray av ~label:[ "va" ] ~output_dims:[ m ] () in
  let vb = TDSL.ndarray bv ~label:[ "vb" ] ~output_dims:[ m ] () in
  let%op d1 = va +* "i;i=>0" vb in
  let got_dot =
    run ~name:"dot_wshfl"
      ~transform:
        (reduce_transform ~n:m d1.Tensor.value ~body_of:(fun i ->
             LL.Set
               {
                 tn = d1.Tensor.value;
                 idcs = [| f0 |];
                 llsc =
                   Ternop
                     ( Ir.Ops.FMA,
                       (Get (va.Tensor.value, [| it i |]), single),
                       (Get (vb.Tensor.value, [| it i |]), single),
                       (Get (d1.Tensor.value, [| f0 |]), single) );
                 debug = "";
               }))
      d1
  in
  p "warp-shuffle fma dot parity" (approx got_dot expected_dot);
  (match read_generated "dot_wshfl" with
  | None -> p "single-warp shuffle rendering (GPU) or serial fallback (CPU)" false
  | Some src ->
      let has sub = String.is_substring src ~substring:sub in
      let ok =
        if on_gpu then has "ocannl_shfl_xor" && not (has "wred_partials_")
        else not (has "ocannl_shfl_xor")
      in
      p "single-warp shuffle rendering (GPU) or serial fallback (CPU)" ok);

  (* --- 2-warp max-reduce: a non-Add combine. The max is positive, so the allocation-zeroed
     accumulator start does not affect the result. --- *)
  let q = 64 in
  let wv = Array.init q ~f:(fun k -> Float.of_int (k * 13 % 29) -. 5.) in
  let expected_max = Array.fold wv ~init:Float.neg_infinity ~f:Float.max in
  let w = TDSL.ndarray wv ~label:[ "w" ] ~output_dims:[ q ] () in
  let%op x1 = w @^^ "i=>0" in
  let got_max =
    run ~name:"max_wshfl"
      ~transform:
        (reduce_transform ~n:q x1.Tensor.value ~body_of:(fun i ->
             LL.Set
               {
                 tn = x1.Tensor.value;
                 idcs = [| f0 |];
                 llsc =
                   Binop
                     ( Ir.Ops.Max,
                       (Get (x1.Tensor.value, [| f0 |]), single),
                       (Get (w.Tensor.value, [| it i |]), single) );
                 debug = "";
               }))
      x1
  in
  p "warp-shuffle max parity" (approx got_max expected_max);

  (* --- A recognized accumulation whose extent (48) does not cover whole warps: the GPU renderer
     must reject it cleanly (binding the index would race); the C backends run it serially. --- *)
  let r = 48 in
  let uv = Array.init r ~f:(fun k -> Float.of_int (k % 11) *. 0.125) in
  let expected_u = Array.fold uv ~init:0. ~f:( +. ) in
  let u = TDSL.ndarray uv ~label:[ "u" ] ~output_dims:[ r ] () in
  let%op y1 = u ++ "i=>0" in
  let transform =
    reduce_transform ~n:r y1.Tensor.value ~body_of:(fun i ->
        LL.Set
          {
            tn = y1.Tensor.value;
            idcs = [| f0 |];
            llsc =
              Binop
                ( Ir.Ops.Add,
                  (Get (y1.Tensor.value, [| f0 |]), single),
                  (Get (u.Tensor.value, [| it i |]), single) );
            debug = "";
          })
  in
  if on_gpu then
    match
      try
        ignore (run ~name:"odd_extent_wshfl" ~transform y1 : float);
        None
      with Invalid_argument msg -> Some msg
    with
    | Some msg ->
        p "non-warp-multiple extent rejected (GPU) or runs serially (CPU)"
          (String.is_substring msg ~substring:"multiple of the warp size")
    | None -> p "non-warp-multiple extent rejected (GPU) or runs serially (CPU)" false
  else
    p "non-warp-multiple extent rejected (GPU) or runs serially (CPU)"
      (approx (run ~name:"odd_extent_wshfl" ~transform y1) expected_u);

  (* --- A recognized accumulation sharing workgroup slot 0 with a LARGER sibling extent: on GPU
     [guard_annotated_extents] wraps the reduce body in the synthetic [If (i < 64)] launch guard,
     and the renderer must still see through it and reject (a plain binding would race the
     accumulator; PR #119 review). The sibling nest is a benign self-copy of the input, so on the C
     backends the whole kernel runs serially with a partial (first-64) sum. --- *)
  let t = 128 in
  let tv = Array.init t ~f:(fun k -> (Float.of_int (k % 17) *. 0.5) -. 3.) in
  let expected_partial = Array.fold (Array.sub tv ~pos:0 ~len:64) ~init:0. ~f:( +. ) in
  let tt = TDSL.ndarray tv ~label:[ "tt" ] ~output_dims:[ t ] () in
  let%op z1 = tt ++ "i=>0" in
  let sibling_transform (opt : LL.optimized) : LL.optimized =
    (LL.get_node opt.traced_store z1.Tensor.value).LL.zero_initialized_by_code <- false;
    let j = Idx.get_symbol () in
    let i = Idx.get_symbol () in
    let copy_nest =
      LL.For_loop
        {
          index = j;
          from_ = 0;
          to_ = t - 1;
          axis = Workgroup;
          body =
            LL.Set
              {
                tn = tt.Tensor.value;
                idcs = [| it j |];
                llsc = Get (tt.Tensor.value, [| it j |]);
                debug = "";
              };
        }
    in
    let reduce_nest =
      LL.For_loop
        {
          index = i;
          from_ = 0;
          to_ = 63;
          axis = Workgroup_reduce;
          body =
            LL.Set
              {
                tn = z1.Tensor.value;
                idcs = [| f0 |];
                llsc =
                  Binop
                    ( Ir.Ops.Add,
                      (Get (z1.Tensor.value, [| f0 |]), single),
                      (Get (tt.Tensor.value, [| it i |]), single) );
                debug = "";
              };
        }
    in
    { opt with llc = LL.Seq (copy_nest, reduce_nest) }
  in
  if on_gpu then
    match
      try
        ignore (run ~name:"guarded_extent_wshfl" ~transform:sibling_transform z1 : float);
        None
      with Invalid_argument msg -> Some msg
    with
    | Some msg ->
        p "guarded smaller-extent accumulation rejected (GPU) or runs serially (CPU)"
          (String.is_substring msg ~substring:"cover the whole workgroup")
    | None -> p "guarded smaller-extent accumulation rejected (GPU) or runs serially (CPU)" false
  else
    p "guarded smaller-extent accumulation rejected (GPU) or runs serially (CPU)"
      (approx (run ~name:"guarded_extent_wshfl" ~transform:sibling_transform z1) expected_partial)
