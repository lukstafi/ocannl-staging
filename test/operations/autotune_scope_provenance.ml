(* gh-ocannl-687: the action menu enumerates loops inside SCHEDULE-minted [Local_scope] bodies only.

   gh-ocannl-666 taught [Autotune.collect_loops] to descend scope bodies, because since
   gh-ocannl-639 the accumulation mints of [Sched.Unroll { materialize = true }] and [Partition]
   move the per-step / per-segment loops inside the accumulator's scope, where
   [Schedule.rewrite_loop] still reaches them. The IR carried no provenance, so the descent could
   not be aimed: virtualization mints the same construct at every inlined read, and the menu began
   proposing splits, swaps, unrolls and vectorize retypes for inlined interpolation and reduction
   loops that no schedule op has ever been able to reach — descriptors that cost a candidate
   compile each and, under the per-unit cap, displace proposals for the main nest.

   [Low_level.scope_mint] is the fact that separates them. The probe is a controlled pair: ONE
   program shape — [out[i] = scope{ acc := out[i]; for k: acc := acc + x[i,k] }] — built twice,
   the two copies differing in nothing but the scope's [mint] field. The schedule-minted copy is
   the shape gh-ocannl-666 was for and its [k] must be proposable; the inlined copy is the
   widening's collateral and its [k] must not be. The control that makes the second claim
   discriminating: [k] is registry-nameable in the inlined copy too, so what keeps it out of the
   menu is the provenance filter and not an unresolvable binder — the pre-687 descend-everything
   walk would have enumerated it, which is exactly the behaviour under test.

   The last pair of claims is the one the exclusion has to earn (PR #424 review, P2). Narrowing the
   enumeration WITHOUT narrowing [contains_loop] in step would leave this shape with no [Vectorized]
   retype at either level: not at [k] (no longer enumerated) and not at [i] (still reading as
   non-innermost because of [k]). Before gh-666 the outer loop drew that retype and after it the
   inner one did; losing it at both levels would be a regression neither state had. So the two walks
   share one provenance filter, and the two copies here must differ in WHICH loop is vectorizable,
   never in whether one is. *)

open Base
open Ocannl.Operation.DSL_modules
module LL = Ir.Low_level
module Idx = Ir.Indexing
module SC = Ir.Schedule_cache

let p = Ll_test.p
let node = Ll_test.node_factory ~first_id:7100 ~dims:[| 4 |] ()
let ni, nk = (4, 4)

(* Synthetic CPU limits: a vector width, so [Vectorized] retypes are proposed at all. *)
let limits = { Ir.Backend_intf.no_hardware_limits with simd_vector_bytes = 32 }

let () =
  let out = node ~dims:[| ni |] "asp_out" in
  let x = node ~dims:[| ni; nk |] "asp_x" in
  Ll_test.materialize out;
  Ll_test.materialize x;
  let i = Ll_test.sym () and k = Ll_test.sym () in
  let acc = LL.get_scope out in
  let scope_body =
    Ll_test.seq
      (LL.Set_local (acc, Ll_test.get out [| Ll_test.iter i |]))
      (Ll_test.loop_n k nk
         (LL.Set_local
            ( acc,
              Ll_test.add (LL.Get_local acc)
                (Ll_test.get x [| Ll_test.iter i; Ll_test.iter k |]) )))
  in
  (* The one variable of this experiment. *)
  let program mint =
    Ll_test.loop_n i ni
      (Ll_test.set out
         [| Ll_test.iter i |]
         (LL.Local_scope { id = acc; body = scope_body; orig_indices = [| Ll_test.iter i |]; mint }))
  in
  (* The scope-free twin the post-optimize seam needs (gh-ocannl-681): same nodes, same reads and
     writes, no scope — it supplies the traced store and placements. *)
  let raw =
    Ll_test.loop_n i ni
      (Ll_test.loop_n k nk
         (Ll_test.set out
            [| Ll_test.iter i |]
            (Ll_test.add
               (Ll_test.get out [| Ll_test.iter i |])
               (Ll_test.get x [| Ll_test.iter i; Ll_test.iter k |]))))
  in
  let menu_of mint =
    let o =
      Ll_test.optimize_scoped ~materialized:[ out; x ] ~name:"asp_probe" ~raw (program mint)
    in
    let canon = SC.canonicalize ~static_indices:[] ~with_placements:false o in
    let registry = SC.base_registry canon in
    (registry, Autotune.menu ~is_cpu:true ~is_gpu:false ~limits ~registry o)
  in
  let reg_mint, menu_mint = menu_of LL.Schedule_minted in
  let reg_inline, menu_inline = menu_of LL.Inlined_computation in
  let targets rf op =
    match op with
    | SC.Split { axis; _ } | SC.Unroll { axis; _ } | SC.Retype { axis; _ } ->
        SC.equal_sym_ref axis rf
    | SC.Swap { outer; inner } -> SC.equal_sym_ref outer rf || SC.equal_sym_ref inner rf
    | _ -> false
  in
  let count reg menu sym =
    match SC.resolve reg sym with
    | None -> 0
    | Some rf -> List.count menu ~f:(targets rf)
  in
  (* The control: the inlined copy's inner binder IS nameable by a persisted schedule, so its
     absence from the menu is the provenance decision and nothing else. *)
  p "control: the inlined scope's inner loop is registry-nameable"
    (Option.is_some (SC.resolve reg_inline k));
  p "the schedule mint's scope-nested loop draws proposals" (count reg_mint menu_mint k > 0);
  p "the virtualization inline's scope-nested loop draws none"
    (count reg_inline menu_inline k = 0);
  (* Only the scope-nested descriptors differ: the statement-level nest is untouched by the
     provenance filter. *)
  p "the statement-level loop keeps its proposals under either mint (and gains one under the inline)"
    (count reg_inline menu_inline i = count reg_mint menu_mint i + 1
    && count reg_mint menu_mint i > 0);
  p "the inlined copy's menu is the schedule mint's, minus its scope-nested part plus the outer \
     retype that becomes available"
    (List.length menu_inline + count reg_mint menu_mint k = List.length menu_mint + 1);
  (* Innermost-ness is judged over the same loops the enumeration covers, so each copy offers the
     retype at exactly one level: the mint at its scope-nested [k] (gh-ocannl-666's purpose), the
     inline at the enclosing [i] (the pre-666 reading, restored for scopes no schedule op targets).
     Neither copy is left without one. *)
  let vectorizes reg menu sym =
    match SC.resolve reg sym with
    | None -> 0
    | Some rf ->
        List.count menu ~f:(function
          | SC.Retype { axis; ty = LL.Vectorized } -> SC.equal_sym_ref axis rf
          | _ -> false)
  in
  p "the schedule mint offers the Vectorized retype at its scope-nested loop"
    (vectorizes reg_mint menu_mint k = 1 && vectorizes reg_mint menu_mint i = 0);
  p "the inlined copy offers it at the enclosing loop instead"
    (vectorizes reg_inline menu_inline i = 1 && vectorizes reg_inline menu_inline k = 0);
  p "neither copy is left with no Vectorized candidate at all"
    (vectorizes reg_mint menu_mint k + vectorizes reg_mint menu_mint i > 0
    && vectorizes reg_inline menu_inline k + vectorizes reg_inline menu_inline i > 0);
  Stdio.printf "\nDone.\n%!"
