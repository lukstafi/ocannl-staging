(* Partition / index-set splitting (gh-ocannl-508): [Sched.Partition] splits a Serial loop's
   iteration range at static affine breakpoints into separate, individually specialized segment
   nests. The narrowed per-segment ranges let [Sched.apply]'s trailing simplify interval-fold the
   in-loop guards each segment decides, replacing the two guard workarounds:

   - [Split]'s construct-then-fold remainder guard: partitioning at the last tile-multiple first and
   then splitting the dividing main segment yields a guard-free main nest plus a serial epilogue (no
   [If] anywhere).

   - the virtualizer's per-component [Where] range guards of an inlined concatenation: partitioning
   the consumer loop at the component boundaries (derived by [Sched.partition_breakpoints] from the
   guards themselves) folds every [Where], converging the inlined-concat rendering with the segment
   nests that materialized concatenation already lowers to.

   The fresh segment symbols returned by [Sched.partition] make each segment individually
   addressable by subsequent ops (per-segment scheduling), demonstrated by unrolling just the tail
   segment. Structural checks pair with executed-output parity against untransformed references on
   every backend. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module LL = Ir.Low_level
module Sched = Ir.Schedule
module Idx = Ir.Indexing
module Asgns = Ir.Assignments

let p name b = Stdio.printf "%s: %b\n" name b

let named name (comp : Asgns.comp) : Asgns.comp =
  { comp with asgns = Asgns.Block_comment (name, comp.asgns) }

(* The first [For_loop] (in preorder) with the given iteration count. *)
let find_loop_with_extent ~n (llc : LL.t) : Idx.symbol option =
  let found = ref None in
  let rec go (llc : LL.t) =
    if Option.is_none !found then
      match llc with
      | LL.For_loop { index; from_; to_; body; _ } ->
          if to_ - from_ + 1 = n then found := Some index else go body
      | LL.Seq (a, b) ->
          go a;
          go b
      | LL.If { body; _ } -> go body
      | _ -> ()
  in
  go llc;
  !found

(* Structural census: statement [If] guards, scalar [Where] guards, and [For_loop]s. *)
let census (llc : LL.t) : int * int * int =
  let ifs = ref 0 and wheres = ref 0 and loops = ref 0 in
  let rec go (llc : LL.t) =
    match llc with
    | LL.Noop | LL.Comment _ | LL.Staged_compilation _ | LL.Zero_out _ | LL.Declare_local _
    | LL.Workgroup_barrier ->
        ()
    | LL.Seq (a, b) ->
        go a;
        go b
    | LL.For_loop { body; _ } ->
        Int.incr loops;
        go body
    | LL.If { cond = c, _; body } ->
        Int.incr ifs;
        scan c;
        go body
    | LL.Tile_mma { fallback; _ } -> go fallback
    | LL.Set { llsc; _ } | LL.Set_local (_, llsc) -> scan llsc
    | LL.Set_dynamic { dyn_value = v, _; llsc; _ } ->
        scan v;
        scan llsc
    | LL.Set_from_vec { arg = a, _; _ } -> scan a
  and scan (sc : LL.scalar_t) =
    match sc with
    | LL.Ternop (op, (a, _), (b, _), (c, _)) ->
        if Ir.Ops.equal_ternop op Ir.Ops.Where then Int.incr wheres;
        scan a;
        scan b;
        scan c
    | LL.Binop (_, (a, _), (b, _)) ->
        scan a;
        scan b
    | LL.Unop (_, (a, _)) -> scan a
    | LL.Local_scope { body; _ } -> go body
    | LL.Get_dynamic { dyn_value = v, _; _ } -> scan v
    | LL.Get_local _ | LL.Get _ | LL.Get_merge_buffer _ | LL.Constant _ | LL.Constant_bits _
    | LL.Embed_index _ ->
        ()
  in
  go llc;
  (!ifs, !wheres, !loops)

let run_with name transform (t : Tensor.t) =
  let ctx = Context.auto () in
  let ctx, routine =
    Context.compile ~lowered_transform:transform ctx
      (named name (Train.forward t))
      Ir.Indexing.Empty
  in
  let ctx = Context.run ctx routine in
  Context.get_values ctx t.Tensor.value

let close a b = Array.for_all2_exn a b ~f:(fun x y -> Float.(abs (x - y) < 1e-5))

let () =
  Tensor.unsafe_reinitialize ();

  (* === 1: Split-remainder replacement — partition at the last tile-multiple, then split the
     dividing main segment: guard-free main nest + serial epilogue. === *)
  let xv = Array.init 10 ~f:(fun i -> Float.of_int (i - 4)) in
  let make_graph1 () =
    let x = TDSL.ndarray xv ~label:[ "sp_x" ] ~output_dims:[ 10 ] () in
    let%op y = relu x in
    y
  in
  let want1 = run_with "sp_ref1" (fun opt -> opt) (make_graph1 ()) in

  (* Reference discipline: a plain non-dividing Split leaves its remainder guard in-loop. *)
  let split_ifs = ref (-1) in
  let transform_split (opt : LL.optimized) =
    let axis = Option.value_exn ~here:[%here] (find_loop_with_extent ~n:10 opt.LL.llc) in
    let op, _, _ = Sched.split ~axis ~factor:4 ~outer:LL.Serial ~inner:LL.Serial in
    let opt = Sched.apply [ op ] opt in
    let ifs, _, _ = census opt.LL.llc in
    split_ifs := ifs;
    opt
  in
  let got_split = run_with "sp_split_guarded" transform_split (make_graph1 ()) in
  p "plain non-dividing split keeps a remainder guard" (!split_ifs > 0);
  p "guarded split matches reference" (close got_split want1);

  let part_census = ref (-1, -1, -1) in
  let transform_part (opt : LL.optimized) =
    let axis = Option.value_exn ~here:[%here] (find_loop_with_extent ~n:10 opt.LL.llc) in
    let op1, segs = Sched.partition ~axis ~breakpoints:[ 8 ] in
    let main = List.hd_exn segs in
    let op2, _, _ = Sched.split ~axis:main ~factor:4 ~outer:LL.Serial ~inner:LL.Serial in
    let opt = Sched.apply [ op1; op2 ] opt in
    part_census := census opt.LL.llc;
    opt
  in
  let got_part = run_with "sp_partition_split" transform_part (make_graph1 ()) in
  let ifs, _, loops = !part_census in
  p "partition+split is guard-free" (ifs = 0);
  p "partition+split has main outer+inner and tail loops" (loops = 3);
  p "partition+split matches reference" (close got_part want1);

  (* === 2: Inlined-concat consumer — partition at the component boundaries derived from the
     virtualizer's [Where] range guards; every guard folds. === *)
  let make_graph2 () =
    let a = TDSL.ndarray [| 1.0; 2.0; 3.0 |] ~label:[ "sp_ca" ] ~output_dims:[ 3 ] () in
    let b = TDSL.ndarray [| 4.0; 5.0 |] ~label:[ "sp_cb" ] ~output_dims:[ 2 ] () in
    let%op r = sin ((a, b) ++^ "a; b => a^b") in
    r
  in
  let ref_wheres = ref (-1) in
  let observe (opt : LL.optimized) =
    let _, wheres, _ = census opt.LL.llc in
    ref_wheres := wheres;
    opt
  in
  let want2 = run_with "sp_ref2" observe (make_graph2 ()) in
  p "inlined concat consumer carries Where range guards" (!ref_wheres > 0);
  p "concat values are sin of the concatenation"
    (close want2 (Array.map [| 1.0; 2.0; 3.0; 4.0; 5.0 |] ~f:Float.sin));

  let bps = ref [] in
  let part2_census = ref (-1, -1, -1) in
  let transform_concat (opt : LL.optimized) =
    let axis = Option.value_exn ~here:[%here] (find_loop_with_extent ~n:5 opt.LL.llc) in
    bps := Sched.partition_breakpoints ~axis opt.LL.llc;
    let op, _segs = Sched.partition ~axis ~breakpoints:!bps in
    let opt = Sched.apply [ op ] opt in
    part2_census := census opt.LL.llc;
    opt
  in
  let got2 = run_with "sp_partition_concat" transform_concat (make_graph2 ()) in
  p "breakpoints derived from the guards are the component boundary"
    (List.equal Int.equal !bps [ 3 ]);
  let ifs2, wheres2, loops2 = !part2_census in
  p "partitioned concat consumer is guard-free" (ifs2 = 0 && wheres2 = 0);
  p "partitioned concat consumer has one nest per component" (loops2 = 2);
  p "partitioned concat matches reference" (close got2 want2);

  (* === 3: Per-segment scheduling — the fresh segment symbols are individually addressable: unroll
     just the tail segment. === *)
  let part3_census = ref (-1, -1, -1) in
  let transform_tail_unrolled (opt : LL.optimized) =
    let axis = Option.value_exn ~here:[%here] (find_loop_with_extent ~n:5 opt.LL.llc) in
    let op, segs = Sched.partition ~axis ~breakpoints:[ 3 ] in
    let tail = List.last_exn segs in
    let opt = Sched.apply [ op; Sched.Unroll { axis = tail; materialize = true } ] opt in
    part3_census := census opt.LL.llc;
    opt
  in
  let got3 = run_with "sp_partition_tail_unroll" transform_tail_unrolled (make_graph2 ()) in
  let ifs3, wheres3, loops3 = !part3_census in
  p "tail-unrolled partition is guard-free" (ifs3 = 0 && wheres3 = 0);
  p "only the main segment remains a loop" (loops3 = 1);
  p "tail-unrolled partition matches reference" (close got3 want2);

  (* === 4: Pattern discipline — targeted errors. === *)
  let expect_error name transform t =
    match
      try
        ignore
          (Context.compile ~lowered_transform:transform (Context.auto ())
             (named name (Train.forward t))
             Ir.Indexing.Empty
            : Context.t * Context.routine);
        None
      with Invalid_argument msg -> Some msg
    with
    | Some msg ->
        p
          (name ^ " rejected with a targeted error")
          (String.is_substring msg ~substring:"Schedule.")
    | None -> p (name ^ " rejected with a targeted error") false
  in
  expect_error "sp_err_out_of_range"
    (fun (opt : LL.optimized) ->
      let axis = Option.value_exn ~here:[%here] (find_loop_with_extent ~n:10 opt.LL.llc) in
      let op, _ = Sched.partition ~axis ~breakpoints:[ 12 ] in
      Sched.apply [ op ] opt)
    (make_graph1 ());
  expect_error "sp_err_not_increasing"
    (fun (opt : LL.optimized) ->
      let axis = Option.value_exn ~here:[%here] (find_loop_with_extent ~n:10 opt.LL.llc) in
      let op, _ = Sched.partition ~axis ~breakpoints:[ 3; 3 ] in
      Sched.apply [ op ] opt)
    (make_graph1 ());
  expect_error "sp_err_no_such_loop"
    (fun (opt : LL.optimized) ->
      let op, _ = Sched.partition ~axis:(Idx.get_symbol ()) ~breakpoints:[ 3 ] in
      Sched.apply [ op ] opt)
    (make_graph1 ());

  (* === 5: Guard-shape canonicity — index guards use one canonical shape per role (upper bounds
     strict [Cmplt], lower bounds direct [Cmple]). [partition_breakpoints] must derive the same
     transition points from a [Cmple] guard as from the strict encoding it replaced; without a
     [Cmple] arm the guard silently contributes nothing. === *)
  let iprec = Ir.Ops.index_prec () in
  let bps_of ~to_ mk =
    let axis = Idx.get_symbol () in
    let llc =
      LL.For_loop
        {
          index = axis;
          from_ = 0;
          to_;
          body = LL.If { cond = (mk axis, iprec); body = LL.Noop };
          trace_it = false;
          axis = LL.Serial;
        }
    in
    Sched.partition_breakpoints ~axis llc
  in
  let ivar i = (LL.Embed_index (Idx.Iterator i), iprec) in
  let fixed n = (LL.Embed_index (Idx.Fixed_idx n), iprec) in
  (* [3 <= i] and [2 < i] both flip at 3; [i <= 6] and [i < 7] both flip at 7. *)
  let le_lower = bps_of ~to_:9 (fun i -> LL.Binop (Ir.Ops.Cmple, fixed 3, ivar i)) in
  let lt_lower = bps_of ~to_:9 (fun i -> LL.Binop (Ir.Ops.Cmplt, fixed 2, ivar i)) in
  let le_upper = bps_of ~to_:9 (fun i -> LL.Binop (Ir.Ops.Cmple, ivar i, fixed 6)) in
  let lt_upper = bps_of ~to_:9 (fun i -> LL.Binop (Ir.Ops.Cmplt, ivar i, fixed 7)) in
  let is bps want = List.equal Int.equal bps want in
  p "Cmple lower bound breaks where its Cmplt encoding did"
    (is le_lower [ 3 ] && is lt_lower [ 3 ]);
  p "Cmple upper bound breaks where its Cmplt encoding did"
    (is le_upper [ 7 ] && is lt_upper [ 7 ]);
  (* A non-unit coefficient exercises the rounding: [2i <= 5] flips at [i = 3], as does [2i < 6]. *)
  let coef2 n = (LL.Embed_index (Idx.Affine { symbols = [ (2, n) ]; offset = 0 }), iprec) in
  let le_scaled = bps_of ~to_:9 (fun i -> LL.Binop (Ir.Ops.Cmple, coef2 i, fixed 5)) in
  let lt_scaled = bps_of ~to_:9 (fun i -> LL.Binop (Ir.Ops.Cmplt, coef2 i, fixed 6)) in
  p "Cmple with a scaled axis rounds like its Cmplt encoding"
    (is le_scaled [ 3 ] && is lt_scaled [ 3 ]);

  Stdio.printf "\nDone.\n%!"
