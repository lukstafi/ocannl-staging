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
