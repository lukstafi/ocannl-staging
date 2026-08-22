(* [Low_level.peel_accum_nest] must refuse a DEAD level (gh-ocannl-693, Codex P2 on PR #421).

   Every form the peel licenses — codegen's accumulator localization and [Schedule]'s
   [Unroll ~materialize:true] / [Partition] mints — reads and writes the accumulated cell OUTSIDE
   the peeled levels, unconditionally:

     { float v; v = acc[0]; for (k = 0; k <= -1; ++k) { v = v + src[k]; } acc[0] = v; }

   whereas the loop it replaces performs no accesses at all. That is not merely wasteful. The
   routine-interface walk propagates liveness as [live && to_ >= from_], so a node reached only
   under a dead loop is absent from the routine's parameters and need not be allocated at all —
   hoisting its access out of the level can therefore name an identifier that was never declared.
   The same convention is kept by [drop_dead_loop_accesses] for the affine metrics and by
   virtualization's dead-loop drop, whose comment gives the reason in the same terms: not to "mint
   phantom parameters for identifiers only dead code renders".

   Pinned at [peel_accum_nest] rather than end to end, and deliberately: [LL.optimize] drops dead
   loops, so ordinary lowering cannot deliver one to codegen — which is why this never bit. What
   CAN deliver one is a post-optimize transform, i.e. exactly the schedule mints that share this
   peel, so the refusal belongs to the shared definition and is asserted there. The live twin is
   the control: same nest, one bound changed, and it must still peel — a refusal that also refused
   live nests would pass a one-sided test while disabling the feature. *)

open Base
module LL = Ir.Low_level
module Idx = Ir.Indexing
module Tn = Ir.Tnode
module Ops = Ir.Ops

let single = Ops.single

let node ~id ~label ~dims =
  Tn.create (Tn.Specified single) ~id ~label:[ label ]
    ~unpadded_dims:(lazy dims)
    ~padding:(lazy None)
    ()

let acc = node ~id:940_100_001 ~label:"pdl_acc" ~dims:[| 1 |]
let src = node ~id:940_100_002 ~label:"pdl_src" ~dims:[| 4 |]

(* [for k = 0 to upto: acc[0] = acc[0] + src[k]] — an accumulation nest whose cell is invariant
   across [k], i.e. exactly what the peel recognizes. [upto = -1] makes the level dead. *)
let nest ~upto =
  let k = Idx.get_symbol () in
  LL.For_loop
    {
      index = k;
      from_ = 0;
      to_ = upto;
      axis = LL.Serial;
      body =
        LL.Set
          {
            tn = acc;
            idcs = [| Idx.Fixed_idx 0 |];
            llsc =
              LL.Binop
                ( Ops.Add,
                  (LL.Get (acc, [| Idx.Fixed_idx 0 |]), single),
                  (LL.Get (src, [| Idx.Iterator k |]), single) );
            debug = "";
          };
    }

let peels ~upto =
  Option.is_some (LL.peel_accum_nest ~free_of:[] (nest ~upto))

let () =
  (* The control: the identical nest over a live range is still recognized. Without this, a peel
     that refused everything would satisfy the claim below. *)
  Verdict.p "a live accumulation level is peeled" (peels ~upto:3);
  Verdict.p "a single-iteration level is peeled" (peels ~upto:0);
  Verdict.p "a dead accumulation level is refused" (not (peels ~upto:(-1)));
  Verdict.p "an emptier dead level is refused" (not (peels ~upto:(-5)))

(* A peeled guard must VARY with the levels being peeled (gh-ocannl-693, Codex P1 on PR #421).

   [rebuild] keeps a guard around the accumulating update only, so the localized form performs its
   opening load and its closing store OUTSIDE it. That matches the original exactly when the guard's
   truth is not fixed for the whole nest. A guard invariant across the peeled levels is fixed for the
   whole nest, and hoisting the accesses out of it turns "this instance performs no access" into a
   load and a store. Across an enclosing HARDWARE axis that is a data race rather than wasted work:
   for [Workgroup w -> Serial k -> If (w < 1) (acc[0] += x[k])] every lane loads [acc[0]] and writes
   its unchanged local back, so a lane reading before lane 0's store and writing after it silently
   discards the reduction.

   The control is the gh-490 symbolic-extent shape, whose guard mentions the peeled index: it must
   still peel, or the refusal has disabled the feature for every padded window rather than closing
   the race. *)
let guarded_nest ~guard_sym =
  let k = Idx.get_symbol () in
  (* The guard's left-hand index: the peeled [k], an enclosing symbol, or an affine MIX of both —
     the mixed one varies with [k] and still selects among enclosing lanes. *)
  let g : Idx.axis_index =
    match guard_sym with
    | `Peeled -> Idx.Iterator k
    | `Enclosing s -> Idx.Iterator s
    | `Mixed s -> Idx.Affine { symbols = [ (1, s); (1, k) ]; offset = 0 }
  in
  LL.For_loop
    {
      index = k;
      from_ = 0;
      to_ = 3;
      axis = LL.Serial;
      body =
        LL.If
          {
            cond =
              ( LL.Binop
                  (Ops.Cmplt, (LL.Embed_index g, single), (LL.Constant 1.0, single)),
                single );
            body =
              LL.Set
                {
                  tn = acc;
                  idcs = [| Idx.Fixed_idx 0 |];
                  llsc =
                    LL.Binop
                      ( Ops.Add,
                        (LL.Get (acc, [| Idx.Fixed_idx 0 |]), single),
                        (LL.Get (src, [| Idx.Iterator k |]), single) );
                  debug = "";
                };
          };
    }

let () =
  let enclosing = Idx.get_symbol () in
  Verdict.p "a guard varying with the peeled index still peels"
    (Option.is_some (LL.peel_accum_nest ~free_of:[] (guarded_nest ~guard_sym:`Peeled)));
  Verdict.p "a guard invariant across the peeled levels is refused"
    (Option.is_none
       (LL.peel_accum_nest ~free_of:[] (guarded_nest ~guard_sym:(`Enclosing enclosing))));
  (* Varying with a peeled symbol is NOT enough: a mixed guard still selects among enclosing lanes,
     so hoisting the accesses out of it lets every lane write the cell only lane 0 touched. *)
  Verdict.p "a guard mixing a peeled and an enclosing symbol is refused"
    (Option.is_none
       (LL.peel_accum_nest ~free_of:[] (guarded_nest ~guard_sym:(`Mixed enclosing))));
  (* gh-490's runtime-extent guard is NOT constant-bounded: [Assignments.extent_guard] emits
     [i < s] with [s] a STATIC symbol -- a kernel parameter bound at launch, never a loop index. It
     therefore cannot select among enclosing iterations, and the peel must still take it, or every
     runtime-bounded reduction loses accumulator localization (and, on narrow storage, its
     gh-ocannl-639 accumulator width along with it). The caller certifies such symbols through
     [~invariant]; codegen passes its [idx_params]. *)
  let extent_sym = Idx.get_symbol () in
  let extent_nest =
    let k = Idx.get_symbol () in
    LL.For_loop
      {
        index = k;
        from_ = 0;
        to_ = 3;
        axis = LL.Serial;
        body =
          LL.If
            {
              cond =
                ( LL.Binop
                    ( Ops.Cmplt,
                      (LL.Embed_index (Idx.Iterator k), single),
                      (LL.Embed_index (Idx.Iterator extent_sym), single) ),
                  single );
              body =
                LL.Set
                  {
                    tn = acc;
                    idcs = [| Idx.Fixed_idx 0 |];
                    llsc =
                      LL.Binop
                        ( Ops.Add,
                          (LL.Get (acc, [| Idx.Fixed_idx 0 |]), single),
                          (LL.Get (src, [| Idx.Iterator k |]), single) );
                    debug = "";
                  };
            };
      }
  in
  Verdict.p "a runtime-extent guard is refused without the caller's certification"
    (Option.is_none (LL.peel_accum_nest ~free_of:[] extent_nest));
  Verdict.p "a runtime-extent guard peels when its bound is certified loop-invariant"
    (Option.is_some
       (LL.peel_accum_nest ~invariant:[ extent_sym ] ~free_of:[] extent_nest));
  (* Certification is not a blanket permit: an ENCLOSING loop symbol wrongly certified would be the
     caller's error, but a guard with no peeled symbol stays refused however it is certified. *)
  Verdict.p "certification does not admit a guard with no peeled symbol"
    (Option.is_none
       (LL.peel_accum_nest ~invariant:[ enclosing ] ~free_of:[]
          (guarded_nest ~guard_sym:(`Enclosing enclosing))))

(* Having refused to peel it, codegen must still RENDER a dead level — and the [Unrolled] arm is
   where that is not free: it repeats the body [to_ - from_ + 1] times, and [Base.List.init] RAISES
   on a negative length rather than answering the empty list. Zero repetitions is exactly a dead
   level's access-free meaning, so the count is clamped at zero.

   Driven through the [?prelowered] seam, which is the supported route for post-optimize IR
   ([LL.optimize] drops dead loops, so nothing else can deliver one to codegen) and is also how the
   schedule mints' output reaches a backend. The claim is that compiling succeeds at all: before the
   clamp this raised [Invalid_argument] out of code generation. The accumulator is seeded and read
   back, so a level that silently ran anyway would fail too. *)
let () =
  let node = Ll_test.node_factory ~first_id:940_200_000 ~dims:[| 4 |] () in
  let dacc = node ~dims:[| 1 |] "pdl_dacc" in
  let dsrc = node "pdl_dsrc" in
  Ll_test.materialize dacc;
  Ll_test.materialize dsrc;
  let u = Ll_test.sym () in
  let update =
    Ll_test.set dacc
      [| Ll_test.fixed 0 |]
      (Ll_test.add (Ll_test.get dacc [| Ll_test.fixed 0 |]) (Ll_test.get dsrc [| Ll_test.iter u |]))
  in
  (* [from_ = 0, to_ = -3]: dead, and by more than one, so the repetition count is genuinely
     negative rather than zero — [List.init 0] would have been harmless. *)
  let dead_unrolled =
    LL.For_loop { index = u; from_ = 0; to_ = -3; body = update; axis = LL.Unrolled }
  in
  (* The raw twin supplies the traced store and placements; [scoped] replaces the schedule. *)
  let raw = LL.For_loop { index = u; from_ = 0; to_ = 3; body = update; axis = LL.Serial } in
  let seeded = 7.5 in
  let outcome =
    match
      Ll_test.optimize_scoped ~materialized:[ dacc; dsrc ] ~name:"pdl_dead_unrolled" ~raw
        dead_unrolled
    with
    | o ->
        Ll_test.execute ~name:"pdl_dead_unrolled" o
          ~seed:[ (dacc, [| seeded |]); (dsrc, [| 1.0; 2.0; 3.0; 4.0 |]) ]
          ~read:[ dacc ]
        |> fun vals -> Ok (List.hd_exn vals).(0)
    | exception e -> Error (Exn.to_string e)
  in
  Verdict.p "a dead Unrolled level compiles instead of aborting codegen"
    (Result.is_ok outcome);
  Verdict.p "a dead Unrolled level performs no accumulation"
    (match outcome with Ok v -> Float.equal v seeded | Error _ -> false)
