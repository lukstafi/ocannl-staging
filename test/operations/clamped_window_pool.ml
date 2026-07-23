(* Clamped-window lowering for padded max-family pooling (gh-ocannl-504): a padded ([=]-mode)
   window spec whose accumulation identity is non-finite (max / tropical) demands NO margins on its
   operand. Shape inference skips the margin registration ([Row.solve_proj_equations]'s
   [clamp_padded]), and the assignments lowering clamps the window to the operand's valid range
   with range guards — a scalar [Where] falling back to the accumulation identity on gathered
   reads, a statement [If] on a scatter's write target — so an out-of-range window position
   contributes the identity, which is the same as not visiting it.

   Pinned here:

   - The operand and result of a padded max-pool stay unpadded; executed values match the
     -inf-margins semantics ("same" pooling), on an all-negative input that would expose 0-margin
     corruption.

   - The clamp guards are [Sched.partition_breakpoints] flip points (the guards mention the window
     symbol, exercising the interval-offset extension): partitioning the output loop at them gives
     guard-free interior segments with specialized boundary segments, matching the unpartitioned
     reference (executed parity).

   - Backward: the argmax scatter transposes the clamp — guarded writes ([If]) — and gradients are
     correct.

   - Inception-style sharing: one tensor feeding both a padded 0-neutral conv (which commits
     0-margins on the shared buffer) and a padded max-pool reading it clamped, never seeing the
     0-margins. Previously rejected ("Conflicting padding neutral elements") with a
     materialized-copy remedy.

   - The neutral-element gate: a padded add-family window spec ([+++], neutral 0) still demands
     margins — physical halos remain the add-family mechanism. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module LL = Ir.Low_level
module Sched = Ir.Schedule
module Idx = Ir.Indexing
module Asgns = Ir.Assignments

let p name b = Stdio.printf "%s: %b\n" name b
let pr fmt = Stdio.printf fmt

let named name (comp : Asgns.comp) : Asgns.comp =
  { comp with asgns = Asgns.Block_comment (name, comp.asgns) }

let padding_to_string (tn : Ir.Tnode.t) =
  if not (Lazy.is_val tn.Ir.Tnode.padding) then "UNFORCED"
  else
    match Lazy.force tn.Ir.Tnode.padding with
    | None -> "None"
    | Some (arr, elem) ->
        Printf.sprintf "[%s] elem=%s"
          (String.concat ~sep:"; "
             (Array.to_list arr
             |> List.map ~f:(fun Ir.Ops.{ left; right } -> Printf.sprintf "%d/%d" left right)))
          (Float.to_string elem)

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

(* The pool's output loop: the first (preorder) statement-level [For_loop] of extent [n] whose
   subtree carries a clamp guard — distinguishing it from the result's initialization loop of the
   same extent. *)
let find_pool_loop ~n (llc : LL.t) : Idx.symbol option =
  let rec has_guard (llc : LL.t) =
    match llc with
    | LL.For_loop { body; _ } -> has_guard body
    | LL.Seq (a, b) -> has_guard a || has_guard b
    | LL.If _ -> true
    | LL.Set { llsc; _ } -> scan llsc
    | _ -> false
  and scan (sc : LL.scalar_t) =
    match sc with
    | LL.Ternop (op, (a, _), (b, _), (c, _)) ->
        Ir.Ops.equal_ternop op Ir.Ops.Where || scan a || scan b || scan c
    | LL.Binop (_, (a, _), (b, _)) -> scan a || scan b
    | LL.Unop (_, (a, _)) -> scan a
    | _ -> false
  in
  let found = ref None in
  let rec go (llc : LL.t) =
    if Option.is_none !found then
      match llc with
      | LL.For_loop { index; from_; to_; body; _ } ->
          if to_ - from_ + 1 = n && has_guard body then found := Some index else go body
      | LL.Seq (a, b) ->
          go a;
          go b
      | LL.If { body; _ } -> go body
      | _ -> ()
  in
  go llc;
  !found

let close a b = Array.for_all2_exn a b ~f:(fun x y -> Float.(abs (x - y) < 1e-5))
let fa a = String.concat ~sep:" " (Array.to_list a |> List.map ~f:(fun v -> Printf.sprintf "%g" v))

let () =
  Tensor.unsafe_reinitialize ();

  (* === 1: Clamped padded max-pool, 1-D: N=8, stride 2, window 5 (left margin 2, right 3).
     Windows clip at both ends: y_o = max x[2o-2 .. 2o+2]. === *)
  let xv = Array.init 8 ~f:(fun i -> Float.of_int i -. 16.) in
  let make_pool () =
    let x = TDSL.ndarray xv ~label:[ "cw_x" ] ~output_dims:[ 8 ] () in
    let%op y = x @^+ "2*o=+w; w => o" [ "w" ] (0.0 + 0.0) in
    Shape.set_dim w 5;
    (x, y)
  in
  let observed = ref None in
  let run_pool name transform =
    Tensor.unsafe_reinitialize ();
    let x, y = make_pool () in
    let ctx = Context.auto () in
    Train.set_materialized x.Tensor.value;
    Train.set_materialized y.Tensor.value;
    let ctx, routine =
      Context.compile ~lowered_transform:transform ctx (named name (Train.forward y)) Idx.Empty
    in
    let ctx = Context.run ctx routine in
    (x, y, Context.get_values ctx y.Tensor.value)
  in
  let x, y, want =
    run_pool "cw_ref" (fun opt ->
        observed := Some (census opt.LL.llc);
        opt)
  in
  pr "clamped pool = [%s]\n" (fa want);
  p "clamped values match the -inf-margins semantics"
    (close want [| -14.; -12.; -10.; -9. |]);
  p "operand stays unpadded"
    (match Lazy.force x.Tensor.value.Ir.Tnode.padding with None -> true | Some _ -> false);
  p "result stays unpadded"
    (match Lazy.force y.Tensor.value.Ir.Tnode.padding with None -> true | Some _ -> false);
  (match !observed with
  | Some (ifs, wheres, _) ->
      p "clamp is a single Where range guard, no statement If" (ifs = 0 && wheres = 1)
  | None -> p "reference lowering observed" false);

  (* === 2: Partition the output loop at the breakpoints derived from the clamp guard: guard-free
     interior segment, specialized boundary segments, executed parity. === *)
  let bps = ref [] in
  let part_census = ref (-1, -1, -1) in
  let transform_part (opt : LL.optimized) =
    let axis = Option.value_exn ~here:[%here] (find_pool_loop ~n:4 opt.LL.llc) in
    bps := Sched.partition_breakpoints ~axis opt.LL.llc;
    let op, _segs = Sched.partition ~axis ~breakpoints:!bps in
    let opt = Sched.apply [ op ] opt in
    part_census := census opt.LL.llc;
    opt
  in
  let _, _, got_part = run_pool "cw_partitioned" transform_part in
  p "breakpoints delimit the left- and right-truncated boundary segments"
    (List.equal Int.equal !bps [ 1; 3 ]);
  (let ifs, wheres, _ = !part_census in
   p "interior segment is guard-free (one Where per boundary segment)" (ifs = 0 && wheres = 2));
  p "partitioned pool matches reference" (close got_part want);

  (* === 3: Backward — the argmax scatter transposes the clamp (guarded writes). Non-overlapping
     windows (stride 2 = window 2, left margin 1): y_0 = x_0 (left-clipped), y_1 = max(x_1, x_2),
     y_2 = max(x_3, x_4); increasing values put the argmax at the window's last element. === *)
  Tensor.unsafe_reinitialize ();
  let bx =
    Operation.init ~l:"cw_bx" ~prec:Ir.Ops.single ~b:[] ~o:[ 6 ]
      ~f:(fun idcs -> Float.of_int idcs.(0) -. 16.)
      ~grad_spec:Tensor.Require_grad ()
  in
  let%op by = bx @^+ "2*o=+w; w => o" [ "w" ] (0.0 + 0.0) in
  Shape.set_dim w 2;
  let%op loss = by ++ "o => 0" in
  let update = Train.grad_update loss in
  let bwd_census = ref (-1, -1, -1) in
  let ctx = Context.auto () in
  Train.set_materialized bx.Tensor.value;
  Train.set_materialized (Option.value_exn ~here:[%here] bx.Tensor.diff).grad;
  Train.set_materialized by.Tensor.value;
  let ctx = Train.init_params ctx Train.IDX.empty loss in
  let ctx, routine =
    Context.compile
      ~lowered_transform:(fun opt ->
        bwd_census := census opt.LL.llc;
        opt)
      ctx update Idx.Empty
  in
  let ctx = Context.run ctx routine in
  let bxg = Context.get_values ctx (Option.value_exn ~here:[%here] bx.Tensor.diff).grad in
  pr "pool input gradient = [%s]\n" (fa bxg);
  p "gradient lands on each clipped window's argmax"
    (close bxg [| 1.; 0.; 1.; 0.; 1.; 0. |]);
  (let ifs, _, _ = !bwd_census in
   p "backward scatter writes are If-guarded" (ifs > 0));
  p "differentiable operand stays unpadded"
    (match Lazy.force bx.Tensor.value.Ir.Tnode.padding with None -> true | Some _ -> false);

  (* === 4: Inception-style sharing — a padded 0-neutral conv commits margins on the shared
     operand; the padded max-pool reads the same buffer clamped and never sees the 0-margins
     (all-negative input: a 0-margin max would surface as 0s / wrong edge maxima). === *)
  Tensor.unsafe_reinitialize ();
  let%op sx = TDSL.range_of_shape ~output_dims:[ 8 ] () - 16. in
  (* Created (and compiled) first: [conv_out] embeds [sx]'s computation. *)
  let%op conv_out = sx +* "o=+k; k => o" [ "k" ] (1.0 + 0.0) in
  Shape.set_dim k 3;
  let%op pooled = sx @^+ "2*o=+w; w => o" [ "w" ] (0.0 + 0.0) in
  Shape.set_dim w 3;
  let ctx = Context.auto () in
  Train.set_materialized sx.Tensor.value;
  Train.set_materialized pooled.Tensor.value;
  Train.set_materialized conv_out.Tensor.value;
  (* [conv_out] embeds [sx]'s computation and compiles first, committing the conv's margins on
     [sx]'s buffer; the pool then compiles against the committed (0-neutral, padded) layout. *)
  let ctx = Train.forward_once ctx conv_out in
  let ctx = Train.forward_once ctx pooled in
  p "shared operand carries the conv's committed margins"
    (String.equal (padding_to_string sx.Tensor.value) "[1/2] elem=0.");
  let pv = Context.get_values ctx pooled.Tensor.value in
  let cv = Context.get_values ctx conv_out.Tensor.value in
  pr "shared pooled = [%s]\n" (fa pv);
  pr "shared conv = [%s]\n" (fa cv);
  p "pool never reads the 0-margins" (close pv [| -15.; -13.; -11.; -9. |]);
  p "conv sums with 0-margins"
    (close cv [| -31.; -45.; -42.; -39.; -36.; -33.; -30.; -19. |]);

  (* === 5: The gate is the accumulation identity — a padded add-family window ([+++], neutral 0)
     still demands margins (the physical-halo mechanism). === *)
  Tensor.unsafe_reinitialize ();
  let%op ax = TDSL.range_of_shape ~output_dims:[ 8 ] () - 16. in
  let%op asum = ax +++ "2*o=+w; w => o" [ "w" ] (1.0 + 0.0) in
  Shape.set_dim w 3;
  let ctx = Context.auto () in
  Train.set_materialized ax.Tensor.value;
  Train.set_materialized asum.Tensor.value;
  let ctx = Train.init_params ctx Train.IDX.empty asum in
  let _ctx = Train.forward_once ctx asum in
  p "padded add-family window still commits margins"
    (String.equal (padding_to_string ax.Tensor.value) "[1/2] elem=0.");

  Stdio.printf "\nDone.\n%!"
