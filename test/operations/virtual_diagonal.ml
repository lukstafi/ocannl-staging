(* Regression test for gh-ocannl-133 Stage A: virtualize producers whose index vector repeats a
   non-static symbol (diagonal [i;i] / partially-diagonal [i;j;i]) plus covered single-symbol affine
   positions.

   High-level lowering never produces a [Get] of a virtual diagonal with two distinct call-site
   symbols in one place (each assignment lowers to its own loop nest), so -- like
   [virtual_shared_loop] -- these cases are built directly as [Ir.Low_level.t] and run through
   [Ir.Low_level.optimize], the same pipeline (trace_node_facts -> virtual_llc -> cleanup_virtual_llc ->
   simplify -> CSE -> hoist) the backends use. We assert structurally on the optimized form: that
   the diagonal producer virtualizes, that its reads are inlined, and that an equality guard ([Where
   (Cmpeq ...)]) is emitted exactly when the read uses distinct/dynamic indices and folded away when
   the read indices are syntactically equal.

   [Concat] virtualization stays out of scope: a [Concat] index is eliminated during lowering and
   [trace_node_facts] raises if one ever reaches this pass, so it cannot be exercised through [optimize]
   here. The "Concat remains rejected" criterion is the unchanged [check_idcs] [Non_virtual 52]
   branch plus the existing test_concat_graph / test_block_tensor coverage.

   Structural pins say what the optimizer BUILT; virtualization rewrites what value a cell holds, so
   every case also has an EXECUTED leg (gh-ocannl-589): the very same [optimized] record is compiled
   through the [?prelowered] seam (gh-ocannl-562, worked out in [prelowered_seam.ml]), seeded, run,
   and its outputs checked against an OCaml reference, plus against a second arm that
   re-specializes the same code with the producer's placement pre-decided [On_device]. That second
   arm is what pins the guarded inline to the materialized reading of the same program: the guard
   must reproduce, cell by cell, what the producer's buffer would have held. *)

open Base
module LL = Ir.Low_level
module Tn = Ir.Tnode
module Ops = Ir.Ops
module Idx = Ir.Indexing

let single = Ir.Ops.single
let next_id = ref 2000

let mk ?(dims = [| 3; 3 |]) label =
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
let embed s : LL.scalar_t = LL.Embed_index (iter s)
let c x : LL.scalar_t = LL.Constant x
let add a b : LL.scalar_t = LL.Binop (Ops.Add, (a, single), (b, single))
let mul a b : LL.scalar_t = LL.Binop (Ops.Mul, (a, single), (b, single))
let zero tn : LL.t = LL.Zero_out tn

(* A producer value has to identify the producer iteration -- EVERY symbol of it -- and stay clear
   of the zero-init, or the executed legs inherit blind spots the structural probes cannot cover
   either. A bare [embed i] has two: the write at index 0 carries the init value, so dropping the
   first iteration is invisible; and a symbol missing from the value (the non-repeated [j] of a
   partial diagonal) makes a wrong substitution on that axis invisible. Hence [1 + i] for a
   one-symbol producer and [1 + 10*outer + inner] for a two-symbol one. *)
let tick s = add (c 1.) (embed s)
let tag outer inner = add (c 1.) (add (mul (c 10.) (embed outer)) (embed inner))

(* a setter writing [tn] at the given index array *)
let set_at idcs tn llsc : LL.t = LL.Set { tn; idcs; llsc; debug = "" }
let get_at idcs tn : LL.scalar_t = LL.Get (tn, idcs)

let loop s body : LL.t =
  LL.For_loop { index = s; from_ = 0; to_ = 2; body; axis = Serial }

let seq a b : LL.t = LL.Seq (a, b)

(* [materialized] pre-decides those nodes' placement in the lineage, exactly as
   [Context.decide_materialized] does for the [Assignments] pipeline: it is what gives each case a
   materialized arm to compare its guarded inline against. *)
let optimize ?(materialized = []) llc : LL.optimized =
  let ctx : LL.optimize_ctx = LL.empty_optimize_ctx () in
  List.iter materialized ~f:(fun tn -> Tn.Placements.update ctx.LL.placements tn Tn.On_device 589);
  LL.optimize ctx ~unoptim_ll_source:None ~ll_source:None ~name:"virtual_diagonal" [] llc

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
let sentinel = -1.
let blank n = Array.create ~len:n sentinel

let close values expected =
  Array.length values = Array.length expected
  && Array.for_alli values ~f:(fun i v -> Float.(abs (v -. expected.(i)) <= 1e-5))

let same got expected = List.for_all2_exn got expected ~f:close

(* Post-optimization placement probes: decisions live on the optimize_ctx's placements
   (context-scoped memory modes), not on the tnode (which now holds only declared intent). *)
let known_virtual (o : LL.optimized) tn = Tn.Placements.known_virtual o.optimize_ctx.placements tn

(* --- structural probes on the optimized form --- *)
let rec walk_t ~on_get ~on_where (llc : LL.t) =
  match llc with
  | LL.Noop | LL.Declare_local _ | LL.Comment _ | LL.Staged_compilation _ | LL.Workgroup_barrier
  | LL.Tile_mma _ ->
      ()
  | LL.Seq (a, b) ->
      walk_t ~on_get ~on_where a;
      walk_t ~on_get ~on_where b
  | LL.For_loop { body; _ } -> walk_t ~on_get ~on_where body
  | LL.Zero_out _ -> ()
  | LL.Set { llsc; _ } -> walk_s ~on_get ~on_where llsc
  | LL.Set_dynamic { dyn_value = v, _; llsc; _ } ->
      walk_s ~on_get ~on_where v;
      walk_s ~on_get ~on_where llsc
  | LL.Set_from_vec { arg = s, _; _ } -> walk_s ~on_get ~on_where s
  | LL.Set_local (_, s) -> walk_s ~on_get ~on_where s
  | LL.If { cond = c, _; body } ->
      walk_s ~on_get ~on_where c;
      walk_t ~on_get ~on_where body

and walk_s ~on_get ~on_where (s : LL.scalar_t) =
  match s with
  | LL.Constant _ | LL.Constant_bits _ | LL.Get_local _ | LL.Embed_index _ | LL.Get_merge_buffer _
    ->
      ()
  | LL.Get (tn, _) -> on_get tn
  | LL.Get_dynamic { tn; dyn_value = v, _; _ } ->
      on_get tn;
      walk_s ~on_get ~on_where v
  | LL.Local_scope { body; _ } -> walk_t ~on_get ~on_where body
  | LL.Ternop (op, (a, _), (b, _), (d, _)) ->
      on_where op;
      walk_s ~on_get ~on_where a;
      walk_s ~on_get ~on_where b;
      walk_s ~on_get ~on_where d
  | LL.Binop (_, (a, _), (b, _)) ->
      walk_s ~on_get ~on_where a;
      walk_s ~on_get ~on_where b
  | LL.Unop (_, (a, _)) -> walk_s ~on_get ~on_where a

let count_get (o : LL.optimized) tn =
  let n = ref 0 in
  walk_t ~on_get:(fun t -> if t.Tn.id = tn.Tn.id then Int.incr n) ~on_where:(fun _ -> ()) o.llc;
  !n

let count_where (o : LL.optimized) =
  let n = ref 0 in
  walk_t ~on_get:(fun _ -> ()) ~on_where:(function Ops.Where -> Int.incr n | _ -> ()) o.llc;
  !n

let p name b = Stdio.printf "%s: %b\n" name b

(* === Case 1: diagonal producer read by a generic (distinct-symbol) consumer === d[i,i] = i
   (off-diagonal zero); a materialized consumer reads d[j,k]. The diagonal must virtualize, its read
   must be inlined, and exactly one equality guard must survive (j = k). *)
let case_diagonal_generic () =
  let d = mk "d" and o = mk "o" in
  materialize o;
  let i = sym () and j = sym () and k = sym () in
  let producer = seq (zero d) (loop i (set_at [| iter i; iter i |] d (tick i))) in
  let consumer = loop j (loop k (set_at [| iter j; iter k |] o (get_at [| iter j; iter k |] d))) in
  let llc = seq producer consumer in
  let opt = optimize llc in
  p "diagonal-generic: producer virtual" (known_virtual opt d);
  p "diagonal-generic: read inlined (no array read of d)" (count_get opt d = 0);
  p "diagonal-generic: one equality guard survives" (count_where opt >= 1);
  p "diagonal-generic: consumer read inlined under guard" (count_get opt d = 0);
  (* The guard is the whole content of the case: [o] must be the diagonal matrix, not the producer
     value smeared over every cell (guard dropped) and not all zeros (guard always failing). *)
  let expected =
    Array.init 9 ~f:(fun n -> if n / 3 = n % 3 then Float.of_int ((n / 3) + 1) else 0.)
  in
  let seed = [ (o, blank 9) ] and read = [ o ] in
  let virt = execute ~name:"vd_generic" opt ~seed ~read in
  let mat = execute ~name:"vd_generic_mat" (optimize ~materialized:[ d ] llc) ~seed ~read in
  p "diagonal-generic: executed values are the diagonal" (same virt [ expected ]);
  p "diagonal-generic: virtual and materialized arms agree" (same virt mat)

(* === Case 2: diagonal producer read at equal indices -> guard simplifies away === *)
let case_diagonal_equal () =
  let d = mk "d" and o = mk "o" in
  materialize o;
  let i = sym () and j = sym () in
  let producer = seq (zero d) (loop i (set_at [| iter i; iter i |] d (tick i))) in
  (* read d[j,j]: the two call-site indices are syntactically equal, so no guard. *)
  let consumer = loop j (set_at [| iter j; iter j |] o (get_at [| iter j; iter j |] d)) in
  let llc = seq producer consumer in
  let opt = optimize llc in
  p "diagonal-equal: producer virtual" (known_virtual opt d);
  p "diagonal-equal: read inlined (no array read of d)" (count_get opt d = 0);
  p "diagonal-equal: guard folded away (no Where)" (count_where opt = 0);
  (* Folding the guard away must not fold the producer's index dependence away with it: the
     consumer writes only its own diagonal, and each cell there holds that row's producer value. *)
  let expected =
    Array.init 9 ~f:(fun n -> if n / 3 = n % 3 then Float.of_int ((n / 3) + 1) else sentinel)
  in
  let seed = [ (o, blank 9) ] and read = [ o ] in
  let virt = execute ~name:"vd_equal" opt ~seed ~read in
  let mat = execute ~name:"vd_equal_mat" (optimize ~materialized:[ d ] llc) ~seed ~read in
  p "diagonal-equal: executed values are the diagonal, off-diagonal untouched"
    (same virt [ expected ]);
  p "diagonal-equal: virtual and materialized arms agree" (same virt mat)

(* === Case 3: partially-diagonal producer [i;j;i] read generically === d[i,j,i] = i; consumer reads
   d[a,b,cc]. Repeated i guards (a = cc); j substituted normally. *)
let case_partial_diagonal () =
  let d = mk ~dims:[| 3; 3; 3 |] "pd" and o = mk ~dims:[| 3; 3; 3 |] "po" in
  materialize o;
  let i = sym () and j = sym () in
  let a = sym () and b = sym () and cc = sym () in
  let producer =
    seq (zero d) (loop i (loop j (set_at [| iter i; iter j; iter i |] d (tag i j))))
  in
  let consumer =
    loop a
      (loop b
         (loop cc (set_at [| iter a; iter b; iter cc |] o (get_at [| iter a; iter b; iter cc |] d))))
  in
  let llc = seq producer consumer in
  let opt = optimize llc in
  p "partial-diagonal: producer virtual" (known_virtual opt d);
  p "partial-diagonal: read inlined (no array read of d)" (count_get opt d = 0);
  p "partial-diagonal: one equality guard survives" (count_where opt >= 1);
  (* Guarded on the repeated axis only, and the non-repeated [b] is substituted normally, so it
     shows through in the value: [o.(a, b, cc)] is [1 + 10a + b] when [a = cc], and the init
     otherwise. *)
  let expected =
    Array.init 27 ~f:(fun n ->
        let a = n / 9 and b = n / 3 % 3 and cc = n % 3 in
        if a = cc then Float.of_int (1 + (10 * a) + b) else 0.)
  in
  let seed = [ (o, blank 27) ] and read = [ o ] in
  let virt = execute ~name:"vd_partial" opt ~seed ~read in
  let mat = execute ~name:"vd_partial_mat" (optimize ~materialized:[ d ] llc) ~seed ~read in
  p "partial-diagonal: executed values guard the repeated axis only" (same virt [ expected ]);
  p "partial-diagonal: virtual and materialized arms agree" (same virt mat)

(* === Case 4: static-vs-dynamic read of a diagonal producer === Read d[0, j] (a row slice): the
   first position is bound to Fixed_idx 0, the second is dynamic; the consistency must become a
   guard (0 = j), NOT a Non_virtual 13 rejection. *)
let case_static_dynamic () =
  let d = mk "d" and o = mk ~dims:[| 3 |] "o" in
  materialize o;
  let i = sym () and j = sym () in
  let producer = seq (zero d) (loop i (set_at [| iter i; iter i |] d (tick i))) in
  let consumer = loop j (set_at [| iter j |] o (get_at [| Idx.Fixed_idx 0; iter j |] d)) in
  let llc = seq producer consumer in
  let opt = optimize llc in
  p "static-dynamic: producer virtual" (known_virtual opt d);
  p "static-dynamic: read inlined (no array read of d)" (count_get opt d = 0);
  p "static-dynamic: one equality guard survives" (count_where opt >= 1);
  let seed = [ (o, blank 3) ] and read = [ o ] in
  let virt = execute ~name:"vd_static_dynamic" opt ~seed ~read in
  let mat = execute ~name:"vd_static_dynamic_mat" (optimize ~materialized:[ d ] llc) ~seed ~read in
  p "static-dynamic: executed values are row 0 of the diagonal" (same virt [ [| 1.; 0.; 0. |] ]);
  p "static-dynamic: virtual and materialized arms agree" (same virt mat)

(* === Case 5: covered single-symbol affine producer position === Producer d[i, i+1] (single-symbol
   affine, covered by the bare iterator i) read at d[j, j+1]; this must validate after substitution
   (no Non_virtual 13) and inline with no surviving guard. *)
let case_single_symbol_affine () =
  (* One column wider than the diagonal cases: the producer's [i + 1] position reaches column 3, and
     the materialized arm writes it for real. *)
  let d = mk ~dims:[| 3; 4 |] "d" and o = mk ~dims:[| 3 |] "o" in
  materialize o;
  let i = sym () and j = sym () in
  let aff s : Idx.axis_index = Idx.Affine { symbols = [ (1, s) ]; offset = 1 } in
  let producer = seq (zero d) (loop i (set_at [| iter i; aff i |] d (tick i))) in
  let consumer = loop j (set_at [| iter j |] o (get_at [| iter j; aff j |] d)) in
  let llc = seq producer consumer in
  let opt = optimize llc in
  p "single-affine: producer virtual" (known_virtual opt d);
  p "single-affine: read inlined (no array read of d)" (count_get opt d = 0);
  p "single-affine: no guard for matching affine" (count_where opt = 0);
  (* Unguarded, but still per-index: [o.(j)] is the producer value at [i = j]. *)
  let seed = [ (o, blank 3) ] and read = [ o ] in
  let virt = execute ~name:"vd_affine" opt ~seed ~read in
  let mat = execute ~name:"vd_affine_mat" (optimize ~materialized:[ d ] llc) ~seed ~read in
  p "single-affine: executed values track the producer index" (same virt [ [| 1.; 2.; 3. |] ]);
  p "single-affine: virtual and materialized arms agree" (same virt mat)

(* === Case 5b: covered single-symbol affine producer read at a MISMATCHED offset === Producer d[i,
   i+1] read at d[j, j+2]: the substituted producer index (j+1) does not equal the call-site index
   (j+2), but the position is covered (symbol j is bound), so it must virtualize with a SURVIVING
   equality guard (j+1 = j+2, which folds to the init value) -- NOT a Non_virtual 13 deferral to
   materialization. *)
let case_single_symbol_affine_mismatch () =
  (* Wide enough for both the producer's [j + 1] writes and the consumer's [j + 2] reads, which the
     materialized arm performs against a real buffer. *)
  let d = mk ~dims:[| 3; 5 |] "d" and o = mk ~dims:[| 3 |] "o" in
  materialize o;
  let i = sym () and j = sym () in
  let aff ofs s : Idx.axis_index = Idx.Affine { symbols = [ (1, s) ]; offset = ofs } in
  let producer = seq (zero d) (loop i (set_at [| iter i; aff 1 i |] d (tick i))) in
  let consumer = loop j (set_at [| iter j |] o (get_at [| iter j; aff 2 j |] d)) in
  let llc = seq producer consumer in
  let opt = optimize llc in
  p "single-affine-mismatch: producer virtual" (known_virtual opt d);
  p "single-affine-mismatch: read inlined (no array read of d)" (count_get opt d = 0);
  p "single-affine-mismatch: equality guard survives" (count_where opt >= 1);
  (* The guard can never hold, so every cell falls back on the init value, and the materialized arm
     agrees, because [d(j, j + 2)] is off the written diagonal and holds the zero-init. *)
  let seed = [ (o, blank 3) ] and read = [ o ] in
  let virt = execute ~name:"vd_affine_mismatch" opt ~seed ~read in
  let mat = execute ~name:"vd_affine_mismatch_mat" (optimize ~materialized:[ d ] llc) ~seed ~read in
  p "single-affine-mismatch: executed values are the init value" (same virt [ [| 0.; 0.; 0. |] ]);
  p "single-affine-mismatch: virtual and materialized arms agree" (same virt mat)

(* === Case 6: single-symbol (non-repeated) producer still virtualizes -- regression guard === *)
let case_single_symbol () =
  let d = mk ~dims:[| 3 |] "d" and o = mk ~dims:[| 3 |] "o" in
  materialize o;
  let i = sym () and j = sym () in
  let producer = loop i (set_at [| iter i |] d (tick i)) in
  let consumer = loop j (set_at [| iter j |] o (get_at [| iter j |] d)) in
  let llc = seq producer consumer in
  let opt = optimize llc in
  p "single-symbol: producer virtual" (known_virtual opt d);
  p "single-symbol: read inlined (no array read of d)" (count_get opt d = 0);
  p "single-symbol: no guard" (count_where opt = 0);
  let seed = [ (o, blank 3) ] and read = [ o ] in
  let virt = execute ~name:"vd_single" opt ~seed ~read in
  let mat = execute ~name:"vd_single_mat" (optimize ~materialized:[ d ] llc) ~seed ~read in
  p "single-symbol: executed values track the producer index" (same virt [ [| 1.; 2.; 3. |] ]);
  p "single-symbol: virtual and materialized arms agree" (same virt mat)

let () =
  case_diagonal_generic ();
  case_diagonal_equal ();
  case_partial_diagonal ();
  case_static_dynamic ();
  case_single_symbol_affine ();
  case_single_symbol_affine_mismatch ();
  case_single_symbol ();
  Stdio.printf "%!"
