open Base
module Idx = Ir.Indexing
module LL = Ir.Low_level
module Ops = Ir.Ops
module Tn = Ir.Tnode

let next_id = ref 9000

let mk ?(prec = Ops.single) ?(dims = [| 4 |]) label =
  Int.incr next_id;
  Tn.create (Tn.Specified prec) ~id:!next_id ~label:[ label ]
    ~unpadded_dims:(lazy dims)
    ~padding:(lazy None)
    ()

let materialize tn = Tn.update_memory_mode tn Tn.On_device 901

let optimize ?(static_indices = []) llc =
  let ctx : LL.optimize_ctx = { computations = Hashtbl.create (module Tn) } in
  (LL.optimize ctx ~unoptim_ll_source:None ~ll_source:None ~name:"interval_analysis" static_indices
     llc)
    .llc

let count_scalar_ops (llc : LL.t) =
  let where = ref 0 and cmplt = ref 0 and trunc = ref 0 in
  let rec proc = function
    | LL.Noop | LL.Comment _ | LL.Staged_compilation _ | LL.Zero_out _ | LL.Declare_local _
    | LL.Workgroup_barrier ->
        ()
    | LL.Seq (a, b) ->
        proc a;
        proc b
    | LL.For_loop { body; _ } -> proc body
    | LL.Set { llsc; _ } -> scal llsc
    | LL.Set_from_vec { arg = s, _; _ } -> scal s
    | LL.Set_local (_, s) -> scal s
  and scal = function
    | LL.Ternop (op, (a, _), (b, _), (c, _)) ->
        (match op with Ops.Where -> Int.incr where | Ops.FMA | Ops.Mul3 -> ());
        scal a;
        scal b;
        scal c
    | LL.Binop (op, (a, _), (b, _)) ->
        (match op with Ops.Cmplt -> Int.incr cmplt | _ -> ());
        scal a;
        scal b
    | LL.Unop (op, (a, _)) ->
        (match op with Ops.Trunc -> Int.incr trunc | _ -> ());
        scal a
    | LL.Get_dynamic { dyn_value = v, _; tn = _; idcs = _; dyn_axis = _ } -> scal v
    | LL.Local_scope { body; _ } -> proc body
    | LL.Get _ | LL.Get_local _ | LL.Get_merge_buffer _ | LL.Constant _ | LL.Constant_bits _
    | LL.Embed_index _ ->
        ()
  in
  proc llc;
  (!where, !cmplt, !trunc)

let assert_no_guard_ops llc =
  let where, cmplt, trunc = count_scalar_ops llc in
  assert (where = 0);
  assert (cmplt = 0);
  assert (trunc = 0)

let test_loop_interval_folds_where () =
  let i = Idx.get_symbol () in
  let out = mk "loop_out" in
  materialize out;
  let guard =
    LL.Binop (Ops.Cmplt, (LL.Embed_index (Idx.Iterator i), Ops.int64), (LL.Constant 4., Ops.int64))
  in
  let llc =
    LL.For_loop
      {
        index = i;
        from_ = 0;
        to_ = 3;
        axis = Serial;
        trace_it = true;
        body =
          LL.Set
            {
              tn = out;
              idcs = [| Idx.Iterator i |];
              llsc =
                LL.Ternop
                  ( Ops.Where,
                    (guard, Ops.int64),
                    (LL.Constant 7., Ops.single),
                    (LL.Constant 9., Ops.single) );
              debug = "";
            };
      }
  in
  assert_no_guard_ops (optimize llc)

let test_static_interval_folds_where () =
  let static_idx, _ = Idx.get_static_symbol ~static_range:4 Idx.Empty in
  let out = mk "static_out" in
  materialize out;
  let s = static_idx.Idx.static_symbol in
  let guard =
    LL.Binop (Ops.Cmplt, (LL.Embed_index (Idx.Iterator s), Ops.int64), (LL.Constant 4., Ops.int64))
  in
  let llc =
    LL.Set
      {
        tn = out;
        idcs = [| Idx.Fixed_idx 0 |];
        llsc =
          LL.Ternop
            ( Ops.Where,
              (guard, Ops.int64),
              (LL.Constant 3., Ops.single),
              (LL.Constant 5., Ops.single) );
        debug = "";
      }
  in
  assert_no_guard_ops (optimize ~static_indices:[ static_idx ] llc)

let make_one_hot_reduction ?ids ~ids_prec () =
  let vocab = 4 and embed = 3 in
  let table = mk ~dims:[| vocab; embed |] "table" in
  let ids = Option.value ids ~default:(mk ~prec:ids_prec ~dims:[| 2 |] "ids") in
  let result = mk ~dims:[| 2; embed |] "emb" in
  let b = Idx.get_symbol () and d = Idx.get_symbol () and k = Idx.get_symbol () in
  let iprec = Lazy.force ids.Tn.prec in
  let vprec = Lazy.force table.Tn.prec in
  let cmpeq =
    LL.Binop
      ( Ops.Cmpeq,
        (LL.Embed_index (Idx.Iterator k), iprec),
        (LL.Get (ids, [| Idx.Iterator b |]), iprec) )
  in
  let contribution =
    LL.Ternop
      ( Ops.Where,
        (cmpeq, iprec),
        (LL.Get (table, [| Idx.Iterator k; Idx.Iterator d |]), vprec),
        (LL.Constant 0., vprec) )
  in
  let id : LL.scope_id = { tn = result; scope_id = !next_id * 10 } in
  let acc = LL.Binop (Ops.Add, (LL.Get_local id, vprec), (contribution, vprec)) in
  LL.Set
    {
      tn = result;
      idcs = [| Idx.Iterator b; Idx.Iterator d |];
      llsc =
        LL.Local_scope
          {
            id;
            orig_indices = [| Idx.Iterator b; Idx.Iterator d |];
            body =
              LL.Seq
                ( LL.Set_local (id, LL.Constant 0.),
                  LL.For_loop
                    {
                      index = k;
                      from_ = 0;
                      to_ = vocab - 1;
                      trace_it = false;
                      axis = Serial;
                      body = LL.Set_local (id, acc);
                    } );
          };
      debug = "";
    }

let test_generic_gather_guard_folding () =
  let _, _, float_truncs =
    make_one_hot_reduction ~ids_prec:Ops.single ()
    |> LL.rewrite_one_hot_reductions |> count_scalar_ops
  in
  assert (float_truncs > 0);
  let _, uint_cmplt, uint_truncs =
    make_one_hot_reduction ~ids_prec:Ops.uint32 ()
    |> LL.rewrite_one_hot_reductions |> count_scalar_ops
  in
  assert (uint_cmplt = 1);
  assert (uint_truncs = 0);
  let _, int_cmplt, int_truncs =
    make_one_hot_reduction ~ids_prec:Ops.int32 ()
    |> LL.rewrite_one_hot_reductions |> count_scalar_ops
  in
  assert (int_cmplt = 2);
  assert (int_truncs = 0)

let test_compiled_writer_bounds_feed_gather_guard () =
  let ids = mk ~prec:Ops.uint32 ~dims:[| 4 |] "bounded_ids" in
  materialize ids;
  let b = Idx.get_symbol () in
  let writer =
    LL.For_loop
      {
        index = b;
        from_ = 0;
        to_ = 3;
        axis = Serial;
        trace_it = true;
        body =
          LL.Set
            {
              tn = ids;
              idcs = [| Idx.Iterator b |];
              llsc = LL.Embed_index (Idx.Iterator b);
              debug = "";
            };
      }
  in
  ignore (optimize writer : LL.t);
  (match Tn.value_bounds_candidate ids with
  | Some { lo; hi; integral; exact } ->
      assert (Float.(lo = 0.));
      assert (Float.(hi = 3.));
      assert integral;
      assert exact
  | None -> assert false);
  let rewritten =
    make_one_hot_reduction ~ids ~ids_prec:Ops.uint32 () |> LL.rewrite_one_hot_reductions
  in
  assert_no_guard_ops rewritten;
  assert (Option.is_some (Tn.settled_value_bounds ids));
  let fractional : Tn.value_bounds = { lo = 1.5; hi = 1.5; integral = false; exact = true } in
  (match Tn.propose_value_bounds ids fractional with
  | exception Utils.User_error _ -> ()
  | _ -> assert false);
  let wider : Tn.value_bounds = { lo = 0.; hi = 4.; integral = true; exact = true } in
  match Tn.propose_value_bounds ids wider with
  | exception Utils.User_error _ -> ()
  | _ -> assert false

let test_settled_top_accepts_narrow_write () =
  let tn = mk ~prec:Ops.uint32 "settled_top" in
  Tn.propose_value_bounds_top tn;
  Tn.settle_value_bounds tn;
  let narrow : Tn.value_bounds = { lo = 0.; hi = 1.; integral = true; exact = true } in
  Tn.propose_value_bounds tn narrow

let () =
  test_loop_interval_folds_where ();
  test_static_interval_folds_where ();
  test_generic_gather_guard_folding ();
  test_compiled_writer_bounds_feed_gather_guard ();
  test_settled_top_accepts_narrow_write ()
