(* Regression test for gh-ocannl-134: multiple virtual tensors sharing one traced for-loop.

   High-level lowering never places two distinct tensors in the same for-loop (each assignment
   lowers to its own [loop_over_dims]), so these cases are built directly as [Ir.Low_level.t] and
   run through [Ir.Low_level.optimize] -- the same pipeline (trace_node_facts -> decide_placements -> virtual_llc ->
   cleanup_virtual_llc -> simplify -> CSE -> hoist) the backends use. We assert structurally on the
   optimized form and on the resulting [traced_array]/memory-mode facts, which precisely pin the
   #134 invariants:

   - shared loop symbols no longer force [is_complex]; - each candidate tensor in a shared loop gets
   its own stored computation and inlines downstream; - a surviving (materialized) sibling setter
   inlines a virtualized provider from the same loop; - a forward virtual->virtual chain is fully
   inlined, leaving no read of a dropped virtual node; - a reverse/read-before-write sibling read
   stays materialized (existing safety mechanism); - cleanup keeps non-virtual residual setters
   instead of dropping the whole loop.

   Structural pins say what the optimizer BUILT; virtualization rewrites what value a cell holds, so
   every case also has an EXECUTED leg (gh-ocannl-589): the very same [optimized] record is compiled
   through the [?prelowered] seam (gh-ocannl-562, worked out in [prelowered_seam.ml]), seeded, run,
   and its outputs checked against an OCaml reference. Where the case has a virtualization
   candidate, a second arm re-specializes the same code with that candidate's placement pre-decided
   [On_device] — the materialized reading of the same program — and the two arms must agree. *)

open Base
module LL = Ir.Low_level
module Tn = Ir.Tnode
module Ops = Ir.Ops

let single = Ir.Ops.single
let next_id = ref 1000

let mk ?(dims = [| 3 |]) label =
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
let sym () = Ir.Indexing.get_symbol ()
let iter s = Ir.Indexing.Iterator s
let set s tn llsc : LL.t = LL.Set { tn; idcs = [| iter s |]; llsc; debug = "" }
let get s tn : LL.scalar_t = LL.Get (tn, [| iter s |])
let add a b : LL.scalar_t = LL.Binop (Ops.Add, (a, single), (b, single))
let mul a b : LL.scalar_t = LL.Binop (Ops.Mul, (a, single), (b, single))
let c x : LL.scalar_t = LL.Constant x
let embed s : LL.scalar_t = LL.Embed_index (iter s)

(* Sibling providers write [base + i] rather than a constant: a constant makes the executed legs
   blind to WHICH cell of the provider an inlined read stood for, which is the half of the rewrite
   that the shared loop symbol puts at risk. [Embed_index] is not an array access, so this does not
   make the provider [is_complex]. *)
let ramp base s = add (c base) (embed s)

let loop s body : LL.t =
  LL.For_loop { index = s; from_ = 0; to_ = 2; body; axis = Serial }

let seq a b : LL.t = LL.Seq (a, b)

(* [materialized] pre-decides those nodes' placement in the lineage, exactly as
   [Context.decide_materialized] does for the [Assignments] pipeline: it is what gives each case a
   materialized arm to compare its inlined values against. *)
let optimize ?(materialized = []) llc : LL.optimized =
  let ctx : LL.optimize_ctx = LL.empty_optimize_ctx () in
  List.iter materialized ~f:(fun tn -> Tn.Placements.update ctx.LL.placements tn Tn.On_device 589);
  LL.optimize ctx ~unoptim_ll_source:None ~ll_source:None ~name:"shared_loop" [] llc

(* --- the executed leg --- Hand-built code is compiled AS WRITTEN: the identity
   [lowered_transform] takes the place of the default schedule annotator, which would otherwise
   parallelize or fission the loop nest. *)
let base_ctx = lazy (Context.auto ())

let execute ~name (o : LL.optimized) ~(seed : (Tn.t * float array) list) ~(read : Tn.t list) =
  let ctx = Lazy.force base_ctx in
  let ctx, routine =
    Context.compile ~name ~prelowered:o
      ~lowered_transform:(fun x -> x)
      ctx Ir.Assignments.empty_comp Ir.Indexing.Empty
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

let is_complex (o : LL.optimized) tn = (Hashtbl.find_exn o.traced_store tn).LL.is_complex
let p name b = Stdio.printf "%s: %b\n" name b

(* === Case 1: two independent virtual siblings in one loop, read downstream === *)
let case_independent () =
  let a = mk "a" and b = mk "b" and oa = mk "oa" and ob = mk "ob" in
  materialize oa;
  materialize ob;
  let i = sym () and j = sym () and k = sym () in
  let shared = loop i (seq (set i a (ramp 2. i)) (set i b (ramp 3. i))) in
  let use_a = loop j (set j oa (get j a)) in
  let use_b = loop k (set k ob (get k b)) in
  let llc = seq shared (seq use_a use_b) in
  let o = optimize llc in
  p "independent siblings both virtual" (known_virtual o a && known_virtual o b);
  p "independent siblings setters dropped" (count_set o a = 0 && count_set o b = 0);
  p "independent siblings inlined at use sites (no array reads survive)"
    (count_get o a = 0 && count_get o b = 0);
  (* Sharing symbol [i] alone must not make either sibling complex. *)
  p "is_complex from sharing alone" (is_complex o a || is_complex o b);
  let seed = [ (oa, blank 3); (ob, blank 3) ] and read = [ oa; ob ] in
  let virt = execute ~name:"vsl_independent" o ~seed ~read in
  let mat = execute ~name:"vsl_independent_mat" (optimize ~materialized:[ a; b ] llc) ~seed ~read in
  p "independent siblings: each use site got its own sibling's cell"
    (same virt [ [| 2.; 3.; 4. |]; [| 3.; 4.; 5. |] ]);
  p "independent siblings: virtual and materialized arms agree" (same virt mat)

(* === Case 2: mixed loop -- one sibling virtual, one materialized === *)
let case_mixed () =
  let a = mk "a" and b = mk "b" and oa = mk "oa" in
  materialize b;
  materialize oa;
  let i = sym () and j = sym () in
  let shared = loop i (seq (set i a (ramp 2. i)) (set i b (ramp 3. i))) in
  let use_a = loop j (set j oa (get j a)) in
  let llc = seq shared use_a in
  let o = optimize llc in
  p "mixed cleanup keeps b setter" (count_set o b = 1);
  p "mixed drops virtual a setter" (count_set o a = 0);
  p "mixed a virtual, b non-virtual" (known_virtual o a && known_non_virtual o b);
  let seed = [ (b, blank 3); (oa, blank 3) ] and read = [ b; oa ] in
  let virt = execute ~name:"vsl_mixed" o ~seed ~read in
  let mat = execute ~name:"vsl_mixed_mat" (optimize ~materialized:[ a ] llc) ~seed ~read in
  p "mixed: the surviving setter still stores, the virtual sibling still reaches its use"
    (same virt [ [| 3.; 4.; 5. |]; [| 2.; 3.; 4. |] ]);
  p "mixed: virtual and materialized arms agree" (same virt mat)

(* === Case 3: forward sibling provider inlined into a surviving materialized reader === *)
let case_forward_provider () =
  let a = mk "a" and b = mk "b" in
  materialize b;
  let i = sym () in
  (* a written, then b reads a -- both in one loop; b survives (materialized), a virtual. *)
  let shared = loop i (seq (set i a (ramp 2. i)) (set i b (add (get i a) (c 1.)))) in
  let o = optimize shared in
  p "forward provider inlined into materialized reader"
    (known_virtual o a && count_set o a = 0 && count_get o a = 0 && count_set o b = 1);
  let seed = [ (b, blank 3) ] and read = [ b ] in
  let virt = execute ~name:"vsl_forward" o ~seed ~read in
  let mat = execute ~name:"vsl_forward_mat" (optimize ~materialized:[ a ] shared) ~seed ~read in
  p "forward provider: the reader stored the provider's own cell, not a stale one"
    (same virt [ [| 3.; 4.; 5. |] ]);
  p "forward provider: virtual and materialized arms agree" (same virt mat)

(* === Case 4: forward virtual->virtual chain consumed downstream === *)
let case_chain () =
  let a = mk "a" and b = mk "b" and out = mk "out" in
  materialize out;
  let i = sym () and j = sym () in
  (* a = f; b = g(a); both virtual. out = h(b), materialized, read downstream. *)
  let shared = loop i (seq (set i a (ramp 2. i)) (set i b (add (get i a) (c 1.)))) in
  let use_b = loop j (set j out (mul (get j b) (c 2.))) in
  let llc = seq shared use_b in
  let o = optimize llc in
  p "forward virtual-to-virtual chain both virtual" (known_virtual o a && known_virtual o b);
  p "forward virtual-to-virtual chain fully inlined"
    (count_set o a = 0
    && count_set o b = 0
    && count_get o a = 0
    && count_get o b = 0
    && known_non_virtual o out);
  let seed = [ (out, blank 3) ] and read = [ out ] in
  let virt = execute ~name:"vsl_chain" o ~seed ~read in
  let mat = execute ~name:"vsl_chain_mat" (optimize ~materialized:[ a; b ] llc) ~seed ~read in
  p "forward chain: the doubly-inlined value is (2 + i + 1) * 2"
    (same virt [ [| 6.; 8.; 10. |] ]);
  p "forward chain: virtual and materialized arms agree" (same virt mat)

(* === Case 5: loop-carried / read-before-write sibling read stays materialized === [a] is written
   at [i] but read at [i+1] in the same loop, so the read of [a[i+1]] precedes its write in trace
   order (read-before-write). The existing access analysis records this as a recurrent access and
   forces [a] materialized; the later writer must NOT be used to rewrite the earlier read. This is
   the safety mechanism the proposal relies on (#134). *)
let case_reverse () =
  (* [a] is one cell wider than the loop so that the read-ahead of the last iteration stays in
     bounds: the executed leg reads this array for real. *)
  let a = mk ~dims:[| 4 |] "a" and b = mk "b" in
  materialize b;
  let i = sym () in
  let read_ahead = LL.Get (a, [| Ir.Indexing.Affine { symbols = [ (1, i) ]; offset = 1 } |]) in
  let shared = loop i (seq (set i a (c 2.)) (set i b (add read_ahead (c 1.)))) in
  let o = optimize shared in
  p "loop-carried provider kept materialized" (known_non_virtual o a);
  p "loop-carried provider read NOT rewritten (array read preserved)" (count_get o a >= 1);
  (* The values make the safety mechanism observable: each [b.(i)] must come from the INCOMING
     [a.(i+1)], never from the [2.] the same loop stores one iteration later (which would give
     [3.] throughout). *)
  let seed = [ (a, [| 10.; 20.; 30.; 40. |]); (b, blank 3) ] and read = [ b; a ] in
  let got = execute ~name:"vsl_reverse" o ~seed ~read in
  p "loop-carried: reads saw the incoming values, writes landed"
    (same got [ [| 21.; 31.; 41. |]; [| 2.; 2.; 2.; 40. |] ])

(* === Case 6: is_complex still set by a genuine complex scalar computation === *)
let case_complex () =
  let x = mk "x" and y = mk "y" and z = mk "z" in
  materialize x;
  materialize y;
  materialize z;
  let i = sym () in
  let l = loop i (set i z (mul (get i x) (get i y))) in
  let o = optimize l in
  p "is_complex from genuine complex scalar" (is_complex o z);
  let got =
    execute ~name:"vsl_complex" o
      ~seed:[ (x, [| 1.; 2.; 3. |]); (y, [| 4.; 5.; 6. |]); (z, blank 3) ]
      ~read:[ z ]
  in
  p "genuine complex scalar: executed elementwise product" (same got [ [| 4.; 10.; 18. |] ])

(* === Case 7: two virtual providers + an in-loop materialized consumer (Codex P1) === c
   (materialized) reads BOTH a and b in the same loop. The storage pass for the first candidate (a)
   walks c's setter, which reads the not-yet-stored b; it must not call inline_computation on b
   before b is stored (that raised a stale optimize_ctx error). Both providers must virtualize and
   inline into c, and c's setter must survive. *)
let case_inloop_consumer () =
  let a = mk "a" and b = mk "b" and cons = mk "cons" in
  materialize cons;
  let i = sym () in
  let shared =
    loop i
      (seq (set i a (ramp 2. i))
         (seq (set i b (ramp 3. i)) (set i cons (add (get i a) (get i b)))))
  in
  let o = optimize shared in
  p "in-loop consumer: both providers virtual" (known_virtual o a && known_virtual o b);
  p "in-loop consumer: providers inlined (no array reads survive)"
    (count_get o a = 0 && count_get o b = 0);
  p "in-loop consumer: consumer setter kept" (count_set o cons = 1);
  let seed = [ (cons, blank 3) ] and read = [ cons ] in
  let virt = execute ~name:"vsl_inloop" o ~seed ~read in
  let mat = execute ~name:"vsl_inloop_mat" (optimize ~materialized:[ a; b ] shared) ~seed ~read in
  p "in-loop consumer: both inlined providers reached the sum" (same virt [ [| 5.; 7.; 9. |] ]);
  p "in-loop consumer: virtual and materialized arms agree" (same virt mat)

(* === Case 9: a write under a dead loop ([to_ < from_]) never executes === The retired tracer
   never enumerated dead loops; the structural facts pass and the metric views must likewise record
   nothing from them: the node stays read-only (a routine input, not a spurious output). *)
let case_dead_loop () =
  let d = mk "dead" and out = mk "dlo" in
  materialize out;
  let i = sym () and j = sym () in
  let dead_write =
    LL.For_loop
      { index = i; from_ = 0; to_ = -1; body = set i d (c 7.); axis = Serial }
  in
  let consume = loop j (set j out (get j d)) in
  let o = optimize (seq dead_write consume) in
  let traced = Base.Hashtbl.find_exn o.LL.traced_store d in
  p "dead loop: node stays read-only" traced.LL.read_only;
  let (inputs, outputs), _merge = LL.input_and_output_nodes o in
  p "dead loop: node is a routine input, not an output"
    (Set.mem inputs d && not (Set.mem outputs d));
  (* Had the dead loop run, the consumer would have copied [7.] instead of the seeded values. *)
  let got =
    execute ~name:"vsl_dead_loop" o
      ~seed:[ (d, [| 11.; 12.; 13. |]); (out, blank 3) ]
      ~read:[ out; d ]
  in
  p "dead loop: the consumer read the seeded values and the dead write never happened"
    (same got [ [| 11.; 12.; 13. |]; [| 11.; 12.; 13. |] ])

(* === Case 9b: a dead write supplies no coverage === The coverage-side companion of case 9: the
   dead write is dropped from the metric views, so the fixed-position read is read-before-write and
   the node is a routine input. *)
let case_dead_non_traced () =
  let d = mk "deadnt" and out = mk "dnto" in
  materialize out;
  let i = sym () and j = sym () in
  let dead_write =
    LL.For_loop { index = i; from_ = 0; to_ = -1; body = set i d (c 7.); axis = Serial }
  in
  let consume =
    loop j (LL.Set { tn = out; idcs = [| iter j |]; llsc = LL.Get (d, [| Ir.Indexing.Fixed_idx 0 |]); debug = "" })
  in
  let o = optimize (seq dead_write consume) in
  let traced = Base.Hashtbl.find_exn o.LL.traced_store d in
  p "dead-write coverage: read_before_write set" traced.LL.read_before_write;
  let (inputs, _outputs), _merge = LL.input_and_output_nodes o in
  p "dead-write coverage: node is a routine input" (Set.mem inputs d);
  let got =
    execute ~name:"vsl_dead_coverage" o
      ~seed:[ (d, [| 11.; 12.; 13. |]); (out, blank 3) ]
      ~read:[ out ]
  in
  p "dead-write coverage: the fixed-position read served the seeded cell"
    (same got [ [| 11.; 11.; 11. |] ])

(* === Case 10: an If condition's read is not a read-modify-write self-read === The condition reads
   [a] at the same position the guarded body writes it, and shares the body's program path; the
   exemption must not fire (the read executes before the write), so [a] is read-before-write — a
   routine input whose prior contents must be preserved. *)
let case_if_cond_read () =
  let a = mk "guarded" in
  let i = sym () in
  let guarded_update =
    loop i
      (LL.If
         {
           cond = (LL.Binop (Ops.Cmplt, (get i a, single), (c 1., single)), single);
           body = set i a (c 1.);
         })
  in
  let o = optimize guarded_update in
  let traced = Base.Hashtbl.find_exn o.LL.traced_store a in
  p "if-cond read: read_before_write set" traced.LL.read_before_write;
  let (inputs, _outputs), _merge = LL.input_and_output_nodes o in
  p "if-cond read: node is a routine input" (Set.mem inputs a);
  (* Only the cells whose incoming value is below 1 are clamped; the rest keep what was seeded, so
     the executed leg fails if the input contents were not preserved. *)
  let got = execute ~name:"vsl_if_cond" o ~seed:[ (a, [| 0.5; 2.; -1. |]) ] ~read:[ a ] in
  p "if-cond read: only the cells failing the guard were updated" (same got [ [| 1.; 2.; 1. |] ])

let () =
  case_independent ();
  case_mixed ();
  case_forward_provider ();
  case_chain ();
  case_reverse ();
  case_complex ();
  case_inloop_consumer ();
  case_dead_loop ();
  case_dead_non_traced ();
  case_if_cond_read ();
  Stdio.printf "%!"
