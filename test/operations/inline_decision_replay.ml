(* gh-555: inlining as a first-class, searchable schedule decision.

   Phase 1 exercises the analysis/specialization split directly on hand-built [Ir.Low_level.t]
   (like virtual_shared_loop.ml): [analyze_proc] runs once, [specialize_proc] replays per candidate
   over the shared analysis. Pinned: - the default policy materializes a complex producer read
   twice (the [virtualize_max_visits] cap); - an [inline_preferences] entry (the Inline half of the
   decision vector) exempts the producer from the cap, and the virtualizer inlines it — same
   analysis, different decisions, hermetic results; - replaying with a fresh default context
   reproduces the first result exactly (candidate hermeticity of the shared traced store).

   Phase 2 exercises the [Context.decide_inline] API end-to-end on a tensor graph: executed values
   must agree between the default (materialized intermediate) and the inline-forced compile.

   Printed facts are booleans/PASS lines so the expected output stays backend-stable. *)

open Base
module LL = Ir.Low_level
module Tn = Ir.Tnode
module Ops = Ir.Ops

let single = Ir.Ops.single
let next_id = ref 2000

let mk ?(dims = [| 3 |]) label =
  Int.incr next_id;
  Tn.create (Tn.Specified single) ~id:!next_id ~label:[ label ]
    ~unpadded_dims:(lazy dims)
    ~padding:(lazy None)
    ()

let materialize tn = Tn.update_memory_mode tn Tn.On_device 99
let sym () = Ir.Indexing.get_symbol ()
let iter s = Ir.Indexing.Iterator s
let set s tn llsc : LL.t = LL.Set { tn; idcs = [| iter s |]; llsc; debug = "" }
let get s tn : LL.scalar_t = LL.Get (tn, [| iter s |])
let mul a b : LL.scalar_t = LL.Binop (Ops.Mul, (a, single), (b, single))
let c x : LL.scalar_t = LL.Constant x

let loop s body : LL.t =
  LL.For_loop { index = s; from_ = 0; to_ = 2; body; trace_it = true; axis = Serial }

let seq a b : LL.t = LL.Seq (a, b)

let rec walk_t ~on_set ~on_get (llc : LL.t) =
  match llc with
  | LL.Noop | LL.Declare_local _ | LL.Comment _ | LL.Staged_compilation _ | LL.Workgroup_barrier
  | LL.Tile_mma _ ->
      ()
  | LL.Seq (a, b) ->
      walk_t ~on_set ~on_get a;
      walk_t ~on_set ~on_get b
  | LL.For_loop { body; _ } -> walk_t ~on_set ~on_get body
  | LL.Zero_out tn -> on_set tn
  | LL.Set { tn; llsc; _ } ->
      on_set tn;
      walk_s ~on_set ~on_get llsc
  | LL.Set_dynamic { tn; dyn_value = v, _; llsc; _ } ->
      on_set tn;
      walk_s ~on_set ~on_get v;
      walk_s ~on_set ~on_get llsc
  | LL.Set_from_vec { tn; arg = s, _; _ } ->
      on_set tn;
      walk_s ~on_set ~on_get s
  | LL.Set_local (_, s) -> walk_s ~on_set ~on_get s
  | LL.If { cond = c, _; body } ->
      walk_s ~on_set ~on_get c;
      walk_t ~on_set ~on_get body

and walk_s ~on_set ~on_get (s : LL.scalar_t) =
  match s with
  | LL.Constant _ | LL.Constant_bits _ | LL.Get_local _ | LL.Embed_index _ | LL.Get_merge_buffer _
    ->
      ()
  | LL.Get (tn, _) -> on_get tn
  | LL.Get_dynamic { tn; dyn_value = v, _; _ } ->
      on_get tn;
      walk_s ~on_set ~on_get v
  | LL.Local_scope { body; _ } -> walk_t ~on_set ~on_get body
  | LL.Ternop (_, (a, _), (b, _), (d, _)) ->
      walk_s ~on_set ~on_get a;
      walk_s ~on_set ~on_get b;
      walk_s ~on_set ~on_get d
  | LL.Binop (_, (a, _), (b, _)) ->
      walk_s ~on_set ~on_get a;
      walk_s ~on_set ~on_get b
  | LL.Unop (_, (a, _)) -> walk_s ~on_set ~on_get a

let count_set (o : LL.optimized) tn =
  let n = ref 0 in
  walk_t ~on_set:(fun t -> if t.Tn.id = tn.Tn.id then Int.incr n) ~on_get:(fun _ -> ()) o.llc;
  !n

let count_get (o : LL.optimized) tn =
  let n = ref 0 in
  walk_t ~on_set:(fun _ -> ()) ~on_get:(fun t -> if t.Tn.id = tn.Tn.id then Int.incr n) o.llc;
  !n

let p name b = Stdio.printf "%s: %b\n" name b

(* A reversed read [prod[2-j]]: position differs from the enclosing statement's write position, so
   unlike a plain copy it is not read-modify-write-exempt and counts toward the multiplicity. *)
let get_rev s tn : LL.scalar_t =
  LL.Get (tn, [| Ir.Indexing.Affine { symbols = [ (-1, s) ]; offset = 2 } |])

let phase1 () =
  let x = mk "x" and prod = mk "prod" and oa = mk "oa" and ob = mk "ob" in
  materialize x;
  materialize oa;
  materialize ob;
  let i = sym () and j = sym () and k = sym () in
  (* [prod] is complex (reads the materialized [x]) and read by two overlapping consumer sites: the
     default policy trips the visit cap (multiplicity 2 > virtualize_max_visits = 1) and
     materializes it. *)
  let producer = loop i (set i prod (mul (get i x) (c 2.))) in
  let use_a = loop j (set j oa (get_rev j prod)) in
  let use_b = loop k (set k ob (get_rev k prod)) in
  let llc = seq producer (seq use_a use_b) in
  assert (LL.virtualize_settings.max_visits = 1);
  let an = LL.analyze_proc [] llc in
  let o_default = LL.specialize_proc (LL.empty_optimize_ctx ()) an in
  let ctx_inline = LL.empty_optimize_ctx () in
  Hash_set.add ctx_inline.LL.inline_preferences prod;
  let o_inline = LL.specialize_proc ctx_inline an in
  let o_replay = LL.specialize_proc (LL.empty_optimize_ctx ()) an in
  let plc o = o.LL.optimize_ctx.LL.placements in
  p "default: producer materialized by the visit cap"
    (Tn.Placements.known_non_virtual (plc o_default) prod);
  p "default: producer setter survives, both consumers read it"
    (count_set o_default prod = 1 && count_get o_default prod = 2);
  p "inline-preferred: producer virtual" (Tn.Placements.known_virtual (plc o_inline) prod);
  p "inline-preferred: setter dropped, no residual reads"
    (count_set o_inline prod = 0 && count_get o_inline prod = 0);
  p "replay over shared analysis reproduces the default result"
    (Sexp.equal (LL.sexp_of_t o_default.LL.llc) (LL.sexp_of_t o_replay.LL.llc));
  p "hermeticity: inline candidate did not leak into the default traced store"
    (Tn.Placements.known_non_virtual (plc o_replay) prod);
  (* The searchable decision surface (gh-555): each arm reports the producer as flippable in the
     opposite direction, with the recompute-cost bound (extent 1 × multiplicity 2). *)
  let find_flip o tn =
    List.find_map o.LL.flip_candidates ~f:(fun fc ->
        if fc.LL.fc_tn.Tn.id = tn.Tn.id then Some (fc.LL.fc_flip, fc.LL.fc_recompute_cost)
        else None)
  in
  p "default arm reports the producer as an Inline flip of cost 2"
    (match find_flip o_default prod with Some (`Inline, 2) -> true | _ -> false);
  p "inline arm reports the producer as a Materialize flip"
    (match find_flip o_inline prod with Some (`Materialize, 2) -> true | _ -> false)

open Ocannl
open Ocannl.Operation.DSL_modules

let phase2 () =
  (* [t] is complex (reads [x]) and read by two reduction sites (reductions read at positions
     differing from their write position, so unlike copies they count toward the multiplicity):
     default-materialized by the visit cap; [Context.decide_inline] keeps it virtual. *)
  let build () =
    Utils.settings.fixed_state_for_init <- Some 42;
    Tensor.unsafe_reinitialize ();
    let x =
      NTDSL.init ~l:"idr_x" ~prec:Ir.Ops.single ~o:[ 4 ]
        ~f:(fun idcs -> Float.of_int (idcs.(0) + 1) *. 0.5)
        ()
    in
    let%op t = x *. x in
    let%op y = (t ++ "i=>0") + (t ++ "i=>0") in
    Train.set_materialized y.Tensor.value;
    Tn.set_observable y.Tensor.value;
    (t, y)
  in
  let run ~inline =
    let t, y = build () in
    let ctx = Context.auto () in
    let ctx = if inline then Context.decide_inline ctx [ t.Tensor.value ] else ctx in
    let ctx, routine = Context.compile ctx (Train.forward y) Ir.Indexing.Empty in
    let ctx = Context.run ctx routine in
    (Context.get_values ctx y.Tensor.value, Context.placements ctx, t.Tensor.value)
  in
  let yv_default, plc_default, t_default = run ~inline:false in
  let yv_inline, plc_inline, t_inline = run ~inline:true in
  p "phase2 default: intermediate is non-virtual"
    (Tn.Placements.known_non_virtual plc_default t_default);
  p "phase2 inline-forced: intermediate is virtual" (Tn.Placements.known_virtual plc_inline t_inline);
  p "phase2 parity: same result length" (Array.length yv_default = Array.length yv_inline);
  let max_diff =
    Array.foldi yv_default ~init:0.0 ~f:(fun i acc v ->
        Float.max acc (Float.abs (v -. yv_inline.(i))))
  in
  if Float.(max_diff <= 1e-6) then Stdio.printf "phase2 parity: PASS\n"
  else Stdio.printf "phase2 parity: FAIL (max diff %.8f)\n" max_diff

let () =
  phase1 ();
  phase2 ()
