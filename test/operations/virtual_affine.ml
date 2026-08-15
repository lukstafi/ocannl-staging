(* Regression test for gh-ocannl-133 Stage B: virtualize injective affine producers (multi-symbol
   affine LHS indices).

   High-level lowering never produces these shapes directly in a controllable way, so -- like
   [virtual_shared_loop.ml] -- the cases are built directly as [Ir.Low_level.t] and run through
   [Ir.Low_level.optimize] (the same trace_node_facts -> virtual_llc -> cleanup -> simplify pipeline the
   backends use). We assert structurally on the optimized form (which producers virtualize, that no
   intermediate array read/setter survives, and which stay materialized).

   Cases: - structural affine match: producer [2*oh+wh] consumed at the same affine structure
   inlines with no intermediate buffer; - unit-coefficient solving at a plain iterator: producer
   [2*oh+wh] consumed at [t] inlines (the residual [oh] loop is kept and range-guarded); -
   triangular [(s1, s1+s2)]: unit-coefficient solving after pinning s1; - non-injective [i+j] (both
   ranges > 1) stays non-virtual, preserving a producer array read; - Stage A diagonal [i;i] still
   virtualizes (no regression).

   Structural pins say what the optimizer BUILT; virtualization rewrites what value a cell holds, so
   every case also has an EXECUTED leg (gh-ocannl-589): the very same [optimized] record is compiled
   through the [?prelowered] seam (gh-ocannl-562, worked out in [prelowered_seam.ml]), seeded, run,
   and its outputs checked against an OCaml reference, plus — where the producer virtualizes —
   against a second arm that re-specializes the same code with the producer's placement pre-decided
   [On_device]. That second arm is what pins the solved index and its range guard to the
   materialized reading of the same program: exactly the cells the scatter would have written must
   carry the scattered value, and the rest the init value. *)

open Base
module LL = Ir.Low_level
module Tn = Ir.Tnode
module Ops = Ir.Ops
module Idx = Ir.Indexing

let single = Ir.Ops.single
let next_id = ref 2000

let mk ?(dims = [| 6 |]) label =
  Int.incr next_id;
  Tn.create (Tn.Specified single) ~id:!next_id ~label:[ label ]
    ~unpadded_dims:(lazy dims)
    ~padding:(lazy None)
    ()

(* Materialized nodes are also the ones the executed legs seed and read back, so they are marked
   observable: the buffer-aliasing planner may not hand their bytes to another node. Both facts are
   declared intent, settled before optimization, so neither perturbs the structural pins. *)
let materialize tn =
  Tn.update_memory_mode tn Tn.On_device 99;
  Tn.set_observable tn

(* --- low-level builders --- *)
let sym () = Idx.get_symbol ()
let iter s = Idx.Iterator s
let aff terms offset = Idx.Affine { symbols = terms; offset }
let set tn idcs llsc : LL.t = LL.Set { tn; idcs; llsc; debug = "" }
let get tn idcs : LL.scalar_t = LL.Get (tn, idcs)
let c x : LL.scalar_t = LL.Constant x
let zero tn : LL.t = LL.Zero_out tn

(* [from_ = 0, to_ = n - 1] gives a loop of width [n]. *)
let loop_r s n body : LL.t =
  LL.For_loop { index = s; from_ = 0; to_ = n - 1; body; axis = Serial }

let seq a b : LL.t = LL.Seq (a, b)

(* [materialized] pre-decides those nodes' placement in the lineage, exactly as
   [Context.decide_materialized] does for the [Assignments] pipeline: it is what gives each case a
   materialized arm to compare its inlined scatter against. *)
let optimize ?(materialized = []) llc : LL.optimized =
  let ctx : LL.optimize_ctx = LL.empty_optimize_ctx () in
  List.iter materialized ~f:(fun tn -> Tn.Placements.update ctx.LL.placements tn Tn.On_device 589);
  LL.optimize ctx ~unoptim_ll_source:None ~ll_source:None ~name:"virtual_affine" [] llc

(* --- the executed leg --- Hand-built code is compiled AS WRITTEN: the identity
   [lowered_transform] takes the place of the default schedule annotator, which would otherwise
   parallelize or fission the loop nest. *)
let base_ctx = lazy (Context.auto ())

let execute ~name (o : LL.optimized) ~(seed : (Tn.t * float array) list) ~(read : Tn.t list) =
  let ctx = Lazy.force base_ctx in
  let ctx, routine =
    Context.compile ~name ~prelowered:o
      ~lowered_transform:(fun x -> x)
      ctx Ir.Assignments.empty_comp Idx.Empty
  in
  let ctx = List.fold seed ~init:ctx ~f:(fun ctx (tn, vs) -> Context.set_values ctx tn vs) in
  let ctx = Context.run ctx routine in
  List.map read ~f:(Context.get_values ctx)

(* Cells no writer covers keep this sentinel, so "wrote the wrong cells" fails the value check
   instead of reading whatever the buffer happened to hold. *)
let blank n = Array.create ~len:n (-1.)

let close values expected =
  Array.length values = Array.length expected
  && Array.for_alli values ~f:(fun i v -> Float.(abs (v -. expected.(i)) <= 1e-5))

let same got expected = List.for_all2_exn got expected ~f:close

(* Post-optimization placement probes: decisions live on the optimize_ctx's placements
   (context-scoped memory modes), not on the tnode (which now holds only declared intent). *)
let known_virtual (o : LL.optimized) tn = Tn.Placements.known_virtual o.optimize_ctx.placements tn

let known_non_virtual (o : LL.optimized) tn =
  Tn.Placements.known_non_virtual o.optimize_ctx.placements tn

(* --- structural probes on the optimized form --- *)
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

(* Count [Where] guards and the two canonical bound comparisons in the optimized form: range guards
   emitted by unit-coefficient solving render as [Where (And (Cmple _, Cmplt _), value, Get_local)]
   — one shape per role, a direct [Cmple] lower bound and a strict [Cmplt] upper bound — whereas a
   pure structural affine match introduces neither. *)
let count_guard_ops (o : LL.optimized) =
  let wh = ref 0 and le = ref 0 and lt = ref 0 in
  let rec t (llc : LL.t) =
    match llc with
    | LL.Seq (a, b) ->
        t a;
        t b
    | LL.For_loop { body; _ } -> t body
    | LL.Set { llsc; _ } -> s llsc
    | LL.Set_from_vec { arg = a, _; _ } -> s a
    | LL.Set_local (_, x) -> s x
    | _ -> ()
  and s (sc : LL.scalar_t) =
    match sc with
    | LL.Ternop (op, (a, _), (b, _), (d, _)) ->
        (match op with Ops.Where -> Int.incr wh | _ -> ());
        s a;
        s b;
        s d
    | LL.Binop (op, (a, _), (b, _)) ->
        (match op with
        | Ops.Cmple -> Int.incr le
        | Ops.Cmplt -> Int.incr lt
        | _ -> ());
        s a;
        s b
    | LL.Unop (_, (a, _)) -> s a
    | LL.Local_scope { body; _ } -> t body
    | _ -> ()
  in
  t o.LL.llc;
  (!wh, !le, !lt)

let p name b = Stdio.printf "%s: %b\n" name b

(* === Case 1: structural affine match === *)
let case_structural_match () =
  let tgt = mk ~dims:[| 6 |] "smatch" and out = mk ~dims:[| 6 |] "out1" in
  materialize out;
  let oh = sym () and wh = sym () and a = sym () and b = sym () in
  (* Injective + surjective scatter over [0, 6): no zero-init (lowering payoff). *)
  let prod = loop_r oh 3 (loop_r wh 2 (set tgt [| aff [ (2, oh); (1, wh) ] 0 |] (c 5.))) in
  let cons =
    loop_r a 3
      (loop_r b 2 (set out [| aff [ (2, a); (1, b) ] 0 |] (get tgt [| aff [ (2, a); (1, b) ] 0 |])))
  in
  let llc = seq prod cons in
  let o = optimize llc in
  p "structural-match producer virtual" (known_virtual o tgt);
  p "structural-match producer inlined (no array reads survive)" (count_get o tgt = 0);
  p "structural-match producer setter dropped" (count_set o tgt = 0);
  p "structural-match consumer setter kept" (count_set o out = 1);
  (* Same affine structure on both sides: bound pairwise, no range/equality guard. *)
  let wh, le, lt = count_guard_ops o in
  p "structural-match has no guard ops" (wh = 0 && le = 0 && lt = 0);
  (* The scatter is surjective onto [0, 6), so every cell carries the scattered value — the
     sentinel survives anywhere the inlined nest lost a cell the producer covered. *)
  let seed = [ (out, blank 6) ] and read = [ out ] in
  let virt = execute ~name:"va_structural" o ~seed ~read in
  let mat = execute ~name:"va_structural_mat" (optimize ~materialized:[ tgt ] llc) ~seed ~read in
  p "structural-match: every scattered cell holds the producer value"
    (same virt [ Array.create ~len:6 5. ]);
  p "structural-match: virtual and materialized arms agree" (same virt mat)

(* === Case 2: unit-coefficient solving at a plain iterator === *)
let case_unit_solve_plain () =
  let tgt = mk ~dims:[| 6 |] "usolve" and out = mk ~dims:[| 6 |] "out2" in
  materialize out;
  let oh = sym () and wh = sym () and t = sym () in
  let prod = loop_r oh 3 (loop_r wh 2 (set tgt [| aff [ (2, oh); (1, wh) ] 0 |] (c 7.))) in
  let cons = loop_r t 6 (set out [| iter t |] (get tgt [| iter t |])) in
  let llc = seq prod cons in
  let o = optimize llc in
  p "unit-solve(plain) producer virtual" (known_virtual o tgt);
  p "unit-solve(plain) producer inlined (no array reads survive)" (count_get o tgt = 0);
  p "unit-solve(plain) producer setter dropped" (count_set o tgt = 0);
  p "unit-solve(plain) consumer setter kept" (count_set o out = 1);
  (* Solving [wh = t - 2*oh] keeps the [oh] loop and range-guards [0 <= t-2*oh < 2]: a [Where] over
     an [And] of a [Cmple] lower bound and a [Cmplt] upper bound. *)
  let wh, le, lt = count_guard_ops o in
  p "unit-solve(plain) emits a range guard (Where + Cmple lower + Cmplt upper)"
    (wh >= 1 && le >= 1 && lt >= 1);
  (* The residual [oh] loop folds over the producer's fiber, so the range guard is what keeps each
     [t] to the single [oh] that scattered it: a guard admitting more than one iteration or none
     would show up here as a wrong value, not as a wrong shape. *)
  let seed = [ (out, blank 6) ] and read = [ out ] in
  let virt = execute ~name:"va_unit_solve" o ~seed ~read in
  let mat = execute ~name:"va_unit_solve_mat" (optimize ~materialized:[ tgt ] llc) ~seed ~read in
  p "unit-solve(plain): every cell holds the producer value once"
    (same virt [ Array.create ~len:6 7. ]);
  p "unit-solve(plain): virtual and materialized arms agree" (same virt mat)

(* === Case 3: triangular (s1, s1 + s2), unit-coefficient solving after pinning s1 === *)
let case_triangular () =
  let tgt = mk ~dims:[| 3; 4 |] "tri" and out = mk ~dims:[| 3; 4 |] "out3" in
  materialize out;
  let s1 = sym () and s2 = sym () and a = sym () and b = sym () in
  (* Triangular map is injective but not surjective, so it carries a zero-init. *)
  let prod =
    seq (zero tgt)
      (loop_r s1 3 (loop_r s2 2 (set tgt [| iter s1; aff [ (1, s1); (1, s2) ] 0 |] (c 9.))))
  in
  let cons =
    loop_r a 3 (loop_r b 4 (set out [| iter a; iter b |] (get tgt [| iter a; iter b |])))
  in
  let llc = seq prod cons in
  let o = optimize llc in
  p "triangular producer virtual" (known_virtual o tgt);
  p "triangular producer inlined (no array reads survive)" (count_get o tgt = 0);
  p "triangular consumer setter kept" (count_set o out = 1);
  (* s2 = b - a is solved and range-guarded [0 <= b-a < 2]. *)
  let wh, le, lt = count_guard_ops o in
  p "triangular emits a range guard (Where + Cmple lower + Cmplt upper)"
    (wh >= 1 && le >= 1 && lt >= 1);
  (* The scatter is injective but not surjective, so the guard has to separate the band it covers
     from the cells that keep the zero-init: [out.(a, b)] is 9 exactly on [0 <= b - a < 2]. *)
  let expected =
    Array.init 12 ~f:(fun n ->
        let a = n / 4 and b = n % 4 in
        if b - a >= 0 && b - a < 2 then 9. else 0.)
  in
  let seed = [ (out, blank 12) ] and read = [ out ] in
  let virt = execute ~name:"va_triangular" o ~seed ~read in
  let mat = execute ~name:"va_triangular_mat" (optimize ~materialized:[ tgt ] llc) ~seed ~read in
  p "triangular: executed values are the scattered band over the zero-init"
    (same virt [ expected ]);
  p "triangular: virtual and materialized arms agree" (same virt mat)

(* === Case 4: non-injective i+j (both ranges > 1) stays non-virtual === *)
let case_noninjective () =
  let tgt = mk ~dims:[| 5 |] "ni" and out = mk ~dims:[| 5 |] "out4" in
  materialize out;
  let i = sym () and j = sym () and a = sym () and b = sym () in
  let prod = loop_r i 3 (loop_r j 3 (set tgt [| aff [ (1, i); (1, j) ] 0 |] (c 1.))) in
  let cons =
    loop_r a 3
      (loop_r b 3 (set out [| aff [ (1, a); (1, b) ] 0 |] (get tgt [| aff [ (1, a); (1, b) ] 0 |])))
  in
  let o = optimize (seq prod cons) in
  (* i+j with both ranges > 1 is not injective: the dropped producer loops fold over a fiber, so the
     producer must stay materialized (the reason is the injectivity soundness line). *)
  p "non-injective producer stays non-virtual" (known_non_virtual o tgt);
  p "non-injective producer array read preserved" (count_get o tgt >= 1);
  (* Staying materialized is already the safe reading, so this case has no second arm — the
     executed leg pins that the fiber the producer folds over reaches the consumer through a
     buffer. Only [out] is read back: [tgt] is non-virtual but unobservable, so the pipeline places
     it [Local] (routine-scoped scratch with no context buffer), which is exactly the point. *)
  let got = execute ~name:"va_noninjective" o ~seed:[ (out, blank 5) ] ~read:[ out ] in
  p "non-injective: the scattered value reached the consumer through the buffer"
    (same got [ Array.create ~len:5 1. ])

(* === Case 5: Stage A diagonal [i;i] still virtualizes (no regression) === *)
let case_stage_a_diagonal () =
  let d = mk ~dims:[| 3; 3 |] "diag" and out = mk ~dims:[| 3; 3 |] "out5" in
  materialize out;
  let i = sym () and a = sym () and b = sym () in
  let prod = seq (zero d) (loop_r i 3 (set d [| iter i; iter i |] (c 4.))) in
  let cons = loop_r a 3 (loop_r b 3 (set out [| iter a; iter b |] (get d [| iter a; iter b |]))) in
  let llc = seq prod cons in
  let o = optimize llc in
  p "stage-a diagonal producer virtual" (known_virtual o d);
  p "stage-a diagonal inlined (no array reads survive)" (count_get o d = 0);
  let expected = Array.init 9 ~f:(fun n -> if n / 3 = n % 3 then 4. else 0.) in
  let seed = [ (out, blank 9) ] and read = [ out ] in
  let virt = execute ~name:"va_stage_a" o ~seed ~read in
  let mat = execute ~name:"va_stage_a_mat" (optimize ~materialized:[ d ] llc) ~seed ~read in
  p "stage-a diagonal: executed values are the diagonal over the zero-init"
    (same virt [ expected ]);
  p "stage-a diagonal: virtual and materialized arms agree" (same virt mat)

let () =
  case_structural_match ();
  case_unit_solve_plain ();
  case_triangular ();
  case_noninjective ();
  case_stage_a_diagonal ();
  Stdio.printf "%!"
