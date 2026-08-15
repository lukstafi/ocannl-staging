(* gh-ocannl-562: the [?prelowered] test seam — compiling and LINKING a hand-built
   [Ir.Low_level.optimized], so that IR shapes the [Assignments] pipeline never emits can be pinned
   by executed values and not only by structural assertions.

   Before the seam, backend [compile] always re-lowered from [Assignments]: [?lowered_transform]
   substitutes the codegen input only, while [code.lowered] — which drives I/O classification
   ([input_and_output_nodes]), liveness planning and the context-buffer delta — stayed the compile's
   own lowering, so hand-built code could reach [analyze_proc]/[specialize_proc] and no further.

   Phase 1 is the seam itself on ordinary code: a hand-written pointwise kernel is seeded with
   [set_values], run, and read back with [get_values].

   Phase 2 is the pattern the seam exists for (the gh-ocannl-561 review's sibling-[Local_scope]
   case, unreachable through [Assignments] because the optimizer never emits scopes reading a node
   another statement overwrites):

   [Y[i] = scopeA{ la := 0; for k: la := la + X[i] } + scopeB{ lb := X[i] * 3 }] then [X[i] := 5]

   The reads in the sibling scopes precede the later write, so the coverage query must NOT report
   X's reads as covered: X is read before written, hence a routine input whose incoming buffer
   contents are preserved. Under a wrong [`Covered] verdict X's placement stays undecided, the
   virtualizer inlines its trivial [5.0] setter into the scope reads, and the seeded values silently
   vanish — which shows up here as executed values, the leg structural pins cannot reach (verified
   by forcing the verdict during development: the value checks fail alongside the classification
   ones).

   Phase 3 pins the scope-purity contract (gh-ocannl-584) the phase-2 program is written to respect:
   the overwrite of X lives in a statement of its own, not inside [scopeB]'s body, because a scope
   body renders HOISTED ahead of the enclosing statement and ordered by [scope_id] rather than by
   the operand's syntactic position. A body write is therefore malformed IR rather than
   under-specified IR, and codegen rejects it.

   Printed facts are booleans/PASS lines so the expected output stays backend-stable. *)

open Base
open Stdio
module LL = Ir.Low_level
module Tn = Ir.Tnode
module Idx = Ir.Indexing
module Ops = Ir.Ops

let single = Ops.single
let p name b = printf "%s: %b\n" name b
let next_id = ref 5620

let mk ?(dims = [| 4 |]) label =
  Int.incr next_id;
  Tn.create (Tn.Specified single) ~id:!next_id ~label:[ label ]
    ~unpadded_dims:(lazy dims)
    ~padding:(lazy None)
    ()

let sym () = Idx.get_symbol ()
let iter s = Idx.Iterator s
let get s tn : LL.scalar_t = LL.Get (tn, [| iter s |])
let set s tn llsc : LL.t = LL.Set { tn; idcs = [| iter s |]; llsc; debug = "" }
let binop op a b : LL.scalar_t = LL.Binop (op, (a, single), (b, single))
let c x : LL.scalar_t = LL.Constant x
let loop ~upto s body : LL.t = LL.For_loop { index = s; from_ = 0; to_ = upto; body; axis = Serial }
let seq a b : LL.t = LL.Seq (a, b)

(* Hand-built code is compiled AS WRITTEN: the identity [lowered_transform] takes the place of the
   default schedule annotator, which would otherwise parallelize or fission the loop nest. *)
let compile_and_run ~name (llc : LL.t) ~(seed : (Tn.t * float array) list) ~(read : Tn.t list) =
  let opt =
    LL.optimize (LL.empty_optimize_ctx ()) ~unoptim_ll_source:None ~ll_source:None ~name [] llc
  in
  let ctx = Context.auto () in
  let ctx, routine =
    Context.compile ~name ~prelowered:opt
      ~lowered_transform:(fun o -> o)
      ctx Ir.Assignments.empty_comp Idx.Empty
  in
  let ctx = List.fold seed ~init:ctx ~f:(fun ctx (tn, values) -> Context.set_values ctx tn values) in
  let ctx = Context.run ctx routine in
  (opt, List.map read ~f:(Context.get_values ctx))

let close values expected =
  Array.length values = Array.length expected
  && Array.for_alli values ~f:(fun i v -> Float.(abs (v -. expected.(i)) <= 1e-5))

(* A hand-written [Y[i] = X[i] * 2 + 1] over 4 cells: seeded, executed and read back. *)
let phase1 () =
  let x = mk "pls_x" and y = mk "pls_y" in
  Tn.update_memory_mode y Tn.On_device 99;
  Tn.set_observable y;
  let i = sym () in
  let llc = loop ~upto:3 i (set i y (binop Ops.Add (binop Ops.Mul (get i x) (c 2.)) (c 1.))) in
  let xv = [| 1.5; 2.5; 3.5; 4.5 |] in
  let opt, results = compile_and_run ~name:"pls_pointwise" llc ~seed:[ (x, xv) ] ~read:[ y ] in
  let (inputs, outputs), _merge = LL.input_and_output_nodes opt in
  p "phase1: the hand-built read-only node is a routine input" (Set.mem inputs x);
  p "phase1: the written node is a routine output" (Set.mem outputs y);
  match results with
  | [ yv ] ->
      p "phase1: executed values match the hand-built formula"
        (close yv (Array.map xv ~f:(fun v -> (v *. 2.) +. 1.)))
  | _ -> assert false

(* The two sibling [Local_scope] operands of the gh-ocannl-561 sketch, with the overwrite of X where
   the scope-purity contract puts it — a statement of its own, after the reading statement:

   [Y[i] = scopeA{ la := 0; for k: la := la + X[i] } + scopeB{ lb := X[i] * 3 }; X[i] := 5]

   X's placement is left to the pipeline, so the read-before-write classification is what keeps X a
   materialized input.

   [scopeA] is an inlined accumulation (the virtualizer's own shape for an inlined reduction) while
   [scopeB] is a single assignment, which [simplify_llc] collapses into the enclosing expression —
   so the program covers both fates a scope operand can meet, and under purity the two agree: no
   read can observe whether it was hoisted or collapsed. *)
let phase2 () =
  let x = mk "pls2_x" and y = mk "pls2_y" in
  let la = mk ~dims:[| 1 |] "pls2_la" and lb = mk ~dims:[| 1 |] "pls2_lb" in
  Tn.update_memory_mode y Tn.On_device 99;
  Tn.set_observable y;
  (* The scope-local scalars stand for inlined nodes, exactly as the virtualizer's own scopes do. *)
  Tn.update_memory_mode la Tn.Virtual 99;
  Tn.update_memory_mode lb Tn.Virtual 99;
  let i = sym () and k = sym () and j = sym () in
  let id_a = LL.get_scope la in
  let id_b = LL.get_scope lb in
  let scope_a : LL.scalar_t =
    LL.Local_scope
      {
        id = id_a;
        body =
          seq
            (LL.Set_local (id_a, c 0.))
            (loop ~upto:1 k (LL.Set_local (id_a, binop Ops.Add (LL.Get_local id_a) (get i x))));
        orig_indices = [| iter i |];
      }
  in
  let scope_b : LL.scalar_t =
    LL.Local_scope
      {
        id = id_b;
        body = LL.Set_local (id_b, binop Ops.Mul (get i x) (c 3.));
        orig_indices = [| iter i |];
      }
  in
  let llc =
    seq (loop ~upto:3 i (set i y (binop Ops.Add scope_a scope_b))) (loop ~upto:3 j (set j x (c 5.)))
  in
  let xv = [| 1.5; 2.5; 3.5; 4.5 |] in
  let opt, results =
    compile_and_run ~name:"pls_sibling_scopes" llc ~seed:[ (x, xv) ] ~read:[ y; x ]
  in
  (match Hashtbl.find opt.LL.traced_store x with
  | None -> p "phase2: X traced" false
  | Some traced -> p "phase2: X is classified read-before-write" traced.LL.read_before_write);
  let (inputs, outputs), _merge = LL.input_and_output_nodes opt in
  p "phase2: X is a routine input" (Set.mem inputs x);
  p "phase2: X is also a routine output" (Set.mem outputs x);
  match results with
  | [ yv; xv_out ] ->
      (* Both scopes' reads see X's INCOMING value; the later statement's write lands. *)
      p "phase2: Y = 5*X_in (both scope operands read the input)"
        (close yv (Array.map xv ~f:(fun v -> 5. *. v)));
      p "phase2: X was overwritten by the following statement" (close xv_out [| 5.; 5.; 5.; 5. |])
  | _ -> assert false

(* gh-ocannl-584: the same program with the overwrite moved INTO [scopeB]'s body,

   [Y[i] = scopeA{ la := 0; for k: la := la + X[i] } + scopeB{ X[i] := 5; lb := X[i] * 3 }]

   is out of contract, and codegen says so instead of silently picking an order. The write would
   render hoisted ahead of the whole statement, so [scopeA]'s reads — and, had [scopeA] been a
   collapsible single assignment, its in-expression reads too — would see 5 rather than the incoming
   values, regardless of the operands' syntactic positions. *)
let phase3 () =
  let x = mk "pls3_x" and y = mk "pls3_y" in
  let la = mk ~dims:[| 1 |] "pls3_la" and lb = mk ~dims:[| 1 |] "pls3_lb" in
  Tn.update_memory_mode y Tn.On_device 99;
  Tn.set_observable y;
  Tn.update_memory_mode la Tn.Virtual 99;
  Tn.update_memory_mode lb Tn.Virtual 99;
  let i = sym () and k = sym () in
  let id_a = LL.get_scope la in
  let id_b = LL.get_scope lb in
  let scope_a : LL.scalar_t =
    LL.Local_scope
      {
        id = id_a;
        body =
          seq
            (LL.Set_local (id_a, c 0.))
            (loop ~upto:1 k (LL.Set_local (id_a, binop Ops.Add (LL.Get_local id_a) (get i x))));
        orig_indices = [| iter i |];
      }
  in
  let scope_b : LL.scalar_t =
    LL.Local_scope
      {
        id = id_b;
        body = seq (set i x (c 5.)) (LL.Set_local (id_b, binop Ops.Mul (get i x) (c 3.)));
        orig_indices = [| iter i |];
      }
  in
  let llc = loop ~upto:3 i (set i y (binop Ops.Add scope_a scope_b)) in
  let opt =
    LL.optimize (LL.empty_optimize_ctx ()) ~unoptim_ll_source:None ~ll_source:None
      ~name:"pls_impure_scope" [] llc
  in
  let rejected f =
    try
      f ();
      false
    with Invalid_argument msg -> String.is_substring msg ~substring:"validate_scope_bodies"
  in
  p "phase3: the validator rejects a tensor-node write in a scope body"
    (rejected (fun () -> LL.validate_scope_bodies opt.LL.llc));
  (* And it rejects there, not merely in a standalone check: the same program driven through the
     backend's [compile] — the path the phase-2 program takes to execution — never reaches
     codegen. *)
  p "phase3: compiling the out-of-contract routine is rejected"
    (rejected (fun () ->
         ignore
           (Context.compile ~name:"pls_impure_scope" ~prelowered:opt
              ~lowered_transform:(fun o -> o)
              (Context.auto ()) Ir.Assignments.empty_comp Idx.Empty)))

let () =
  phase1 ();
  phase2 ();
  phase3 ();
  printf "prelowered seam: PASS\n"
