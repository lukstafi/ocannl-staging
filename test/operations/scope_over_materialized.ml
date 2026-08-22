(* gh-ocannl-681: a [Local_scope] over a MATERIALIZED node is out of contract for
   [Low_level.optimize], and is REJECTED rather than silently normalized to a plain [Get].

   The shape is legal on the OTHER side of the pipeline: [Schedule]'s materializing [Unroll] and
   [Partition] mints, and [C_syntax.try_widen_serial_reduce], build exactly it over a materialized
   accumulator AFTER optimization, and codegen renders it. So one IR meant two different things
   depending on which side of [optimize] it was handed to, and nothing told a test author or a
   future transform writer which side they were on. The optimizer used to answer by discarding the
   body and reading the buffer instead, which is how two [accum_width.ml] legs ran kernels literally
   spelling [acc[0] = acc[0]] while claiming to pin an accumulation width: the identity copy
   happened to reproduce the expected value (256 stays 256), so they passed without executing what
   they claimed.

   The legs below are the contract. Materialized-accumulator localization belongs to codegen's
   accumulator peel (gh-ocannl-693), which this rejection makes the ONE route: until it lands, the
   only way to hand a backend the post-optimize scope form is past the optimizer, through the
   [?prelowered] seam that {!Ll_test.optimize_scoped} wraps -- leg 4. *)

open Base
module LL = Ir.Low_level
module Tn = Ir.Tnode

let p = Ll_test.p

let () =
  let node = Ll_test.node_factory ~first_id:9800 ~dims:[| 8 |] () in
  (* Eight values that identify their iteration and stay off both the zero-init and the seed, so a
     dropped, replayed or reordered iteration changes the sum: 1 + 2 + ... + 8 = 36. *)
  let xs_values = Array.init 8 ~f:(fun k -> Float.of_int (k + 1)) in
  let seeded = 100.0 in
  let accumulated = 136.0 in
  (* === the shape that used to collapse === *)
  let acc = node ~dims:[| 1 |] "som_acc" in
  let xs = node "som_xs" in
  Ll_test.materialize acc;
  Ll_test.materialize xs;
  let i = Ll_test.sym () in
  let id = LL.get_scope acc in
  let body =
    LL.Seq
      ( LL.Set_local (id, Ll_test.get acc [| Ll_test.fixed 0 |]),
        Ll_test.loop_n i 8
          (LL.Set_local (id, Ll_test.add (LL.Get_local id) (Ll_test.get xs [| Ll_test.iter i |]))) )
  in
  let scoped =
    Ll_test.set acc
      [| Ll_test.fixed 0 |]
      (LL.Local_scope { id; body; orig_indices = [| Ll_test.fixed 0 |] })
  in
  let rejection =
    match Ll_test.optimize ~materialized:[ acc; xs ] ~name:"som_reject" scoped with
    | (_ : LL.optimized) -> None
    | exception Invalid_argument msg -> Some msg
  in
  p "optimizing a Local_scope over a materialized node is refused" (Option.is_some rejection);
  p "the refusal names the node"
    (Option.value_map rejection ~default:false ~f:(fun msg ->
         String.is_substring msg ~substring:(Tn.debug_name acc)));
  (* The refusal is about the SCOPE, not about the computation: the same accumulation spelled
     without one optimizes and executes. This is also the raw twin the prelowered leg needs. *)
  let raw =
    Ll_test.loop_n i 8
      (Ll_test.set acc
         [| Ll_test.fixed 0 |]
         (Ll_test.add
            (Ll_test.get acc [| Ll_test.fixed 0 |])
            (Ll_test.get xs [| Ll_test.iter i |])))
  in
  let raw_vals =
    let o = Ll_test.optimize ~materialized:[ acc; xs ] ~name:"som_raw" raw in
    Ll_test.execute ~name:"som_raw" o
      ~seed:[ (acc, [| seeded |]); (xs, xs_values) ]
      ~read:[ acc ]
  in
  p "the same accumulation without the scope optimizes and executes"
    (Float.equal (List.hd_exn raw_vals).(0) accumulated);
  (* === negative control: a scope over a VIRTUAL node is the legal form === *)
  (* [out.(r) = sum_c src.(r, c)], with the reduction held in a scope local over a virtual node --
     the shape the virtualizer itself mints. The producer values identify BOTH symbols
     ([1 + 10*r + c]), so a substitution that drops [r] or [c], or a range that over- or
     under-covers, changes the row sums; [out] is seeded with the sentinel, so an uncovered row
     fails rather than reading whatever the buffer held. *)
  let out = node ~dims:[| 4 |] "som_out" in
  let src = node ~dims:[| 4; 3 |] "som_src" in
  let tmp = node ~dims:[| 1 |] "som_tmp" in
  Ll_test.materialize out;
  Ll_test.materialize src;
  Ll_test.virtualize tmp;
  let r = Ll_test.sym () in
  let c = Ll_test.sym () in
  let vid = LL.get_scope tmp in
  let vbody =
    LL.Seq
      ( LL.Set_local (vid, Ll_test.c 0.0),
        Ll_test.loop_n c 3
          (LL.Set_local
             ( vid,
               Ll_test.add (LL.Get_local vid)
                 (Ll_test.get src [| Ll_test.iter r; Ll_test.iter c |]) )) )
  in
  let vllc =
    Ll_test.loop_n r 4
      (Ll_test.set out
         [| Ll_test.iter r |]
         (LL.Local_scope { id = vid; body = vbody; orig_indices = [| Ll_test.fixed 0 |] }))
  in
  let vo = Ll_test.optimize ~materialized:[ out; src ] ~name:"som_virtual" vllc in
  p "a Local_scope over a virtual node survives optimization" (Ll_test.count_scopes vo.LL.llc = 1);
  let src_values =
    Array.init 12 ~f:(fun k -> 1.0 +. (10.0 *. Float.of_int (k / 3)) +. Float.of_int (k % 3))
  in
  let vvals =
    Ll_test.execute ~name:"som_virtual" vo
      ~seed:[ (out, Ll_test.blank 4); (src, src_values) ]
      ~read:[ out ]
  in
  p "the virtual-node scope executes to the reference row sums"
    (Ll_test.same vvals [ [| 6.0; 36.0; 66.0; 96.0 |] ]);
  (* === the supported route for the post-optimize form === *)
  let so = Ll_test.optimize_scoped ~materialized:[ acc; xs ] ~name:"som_prelowered" ~raw scoped in
  let svals =
    Ll_test.execute ~name:"som_prelowered" so
      ~seed:[ (acc, [| seeded |]); (xs, xs_values) ]
      ~read:[ acc ]
  in
  p "the prelowered seam still delivers the materialized-accumulator scope"
    (Float.equal (List.hd_exn svals).(0) accumulated)
