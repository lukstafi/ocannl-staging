(* gh-494 waypoint 3: the op-legality oracle over hand-built programs.

   A schedule op's obligation is decided by the affine queries: pairing the op's loop symbols as
   thread identity must leave every write-involving access pair Disjoint or Same_thread, with the
   reduction-reassociation license for Vectorized retypes and Swaps of accumulations. Verdicts are
   three-valued; both proven directions are sound (Op_unknown means "compile and see"). *)

open Base
module Tn = Ir.Tnode
module LL = Ir.Low_level
module Idx = Ir.Indexing
module Sched = Ir.Schedule

let fresh_tn =
  let c = ref 970_000_000 in
  fun label dims ->
    Int.incr c;
    Tn.create (Tn.Specified Ir.Ops.single) ~id:!c ~label:[ label ]
      ~unpadded_dims:(lazy dims)
      ~padding:(lazy None)
      ()

let sp = Ir.Ops.single

let for_over ?(extent = 8) sym body =
  LL.For_loop { index = sym; from_ = 0; to_ = extent - 1; body; trace_it = false; axis = LL.Serial }

let hand_built ~stmts ~tns_on_device ~tns_local =
  let optimize_ctx = LL.empty_optimize_ctx () in
  let plc = optimize_ctx.LL.placements in
  List.iter tns_on_device ~f:(fun tn -> Tn.Placements.update plc tn Tn.On_device 49);
  List.iter tns_local ~f:(fun tn -> Tn.Placements.update plc tn Tn.Local 49);
  let traced_store = Hashtbl.create (module Tn) in
  let llc = LL.unflat_lines stmts in
  List.iter (tns_on_device @ tns_local) ~f:(fun tn ->
      ignore (LL.get_node traced_store tn : LL.traced_array));
  {
    LL.traced_store;
    optimize_ctx;
    llc;
    merge_node = None;
    workgroup_shared = Set.empty (module Tn);
    simdgroup_fragments = Set.empty (module Tn);
    swizzled = Set.empty (module Tn);
  }

let show = function
  | Sched.Op_legal -> "Legal"
  | Sched.Op_illegal _ -> "Illegal"
  | Sched.Op_unknown _ -> "Unknown"

let p name v = Stdio.printf "%-44s %s\n" name (show v)

let () =
  (* Matmul: zero-init nest plus an accumulation nest. *)
  let a = fresh_tn "a" [| 8; 8 |]
  and b = fresh_tn "b" [| 8; 8 |]
  and c = fresh_tn "c" [| 8; 8 |] in
  let i = Idx.get_symbol () and j = Idx.get_symbol () in
  let i2 = Idx.get_symbol () and j2 = Idx.get_symbol () and k = Idx.get_symbol () in
  let get tn idcs = LL.Get (tn, idcs) in
  let zero_nest =
    for_over i
      (for_over j
         (LL.Set
            { tn = c; idcs = [| Idx.Iterator i; Idx.Iterator j |]; llsc = LL.Constant 0.; debug = "" }))
  in
  let acc_nest =
    for_over i2
      (for_over j2
         (for_over k
            (LL.Set
               {
                 tn = c;
                 idcs = [| Idx.Iterator i2; Idx.Iterator j2 |];
                 llsc =
                   LL.Binop
                     ( Ir.Ops.Add,
                       (get c [| Idx.Iterator i2; Idx.Iterator j2 |], sp),
                       ( LL.Binop
                           ( Ir.Ops.Mul,
                             (get a [| Idx.Iterator i2; Idx.Iterator k |], sp),
                             (get b [| Idx.Iterator k; Idx.Iterator j2 |], sp) ),
                         sp ) );
                 debug = "";
               })))
  in
  (* Self-contained kernel: the accumulation nest alone (its rmw serves as the init-carrying
     form); every node is accessed only under this statement, so the intra-nest proof is the whole
     obligation for hardware retypes too. *)
  let opt = hand_built ~stmts:[ acc_nest ] ~tns_on_device:[ a; b; c ] ~tns_local:[] in
  (* Two sibling statements sharing [c]: hardware annotations interleave their threads with no
     grid-wide synchronization, so hardware retypes must downgrade to Unknown (Codex P1) while
     within-nest reorderings (Vectorized, Swap) keep their intra-nest verdicts. *)
  let opt2 = hand_built ~stmts:[ zero_nest; acc_nest ] ~tns_on_device:[ a; b; c ] ~tns_local:[] in
  let check name op = p name (Sched.op_legality opt op) in
  let check2 name op = p name (Sched.op_legality opt2 op) in
  check "retype outer parallel loop to Grid" (Sched.Retype { axis = i2; ty = LL.Grid });
  check "retype inner parallel loop to Workgroup" (Sched.Retype { axis = j2; ty = LL.Workgroup });
  check "retype reduction loop to Grid" (Sched.Retype { axis = k; ty = LL.Grid });
  check "retype reduction loop to Vectorized" (Sched.Retype { axis = k; ty = LL.Vectorized });
  check2 "cross-statement retype to Grid" (Sched.Retype { axis = i2; ty = LL.Grid });
  check2 "cross-statement zero-init loop to Grid" (Sched.Retype { axis = i; ty = LL.Grid });
  check2 "cross-statement Vectorized keeps scope" (Sched.Retype { axis = k; ty = LL.Vectorized });
  check2 "cross-statement swap keeps scope" (Sched.Swap { outer = i2; inner = j2 });
  check "swap parallel pair" (Sched.Swap { outer = i2; inner = j2 });
  check "swap parallel with reduction" (Sched.Swap { outer = j2; inner = k });
  check "swap non-nested loops" (Sched.Swap { outer = i2; inner = k });
  (let op, _, _ =
     Sched.split ~axis:k ~factor:4 ~outer:LL.Serial ~inner:LL.Workgroup
   in
   check "split reduction with Workgroup inner" op);
  (let op, _, _ = Sched.split ~axis:j2 ~factor:4 ~outer:LL.Serial ~inner:LL.Workgroup in
   check "split parallel with Workgroup inner" op);
  (let op, _, _ = Sched.split ~axis:k ~factor:4 ~outer:LL.Serial ~inner:LL.Serial in
   check "split reduction serial/serial" op);
  check "unroll reduction" (Sched.Unroll { axis = k; materialize = false });
  check "stage is not modeled"
    (Sched.Stage
       {
         source = a;
         tile_loops = [ k ];
         shared = false;
         cooperative = None;
         hoisted = false;
         swizzle = false;
       });

  (* A genuinely order-sensitive serial loop: lagged self-reference. *)
  let x = fresh_tn "x" [| 9 |] in
  let t = Idx.get_symbol () in
  let lag_nest =
    for_over t
      (LL.Set
         {
           tn = x;
           idcs = [| Idx.Affine { symbols = [ (1, t) ]; offset = 1 } |];
           llsc =
             LL.Binop
               ( Ir.Ops.Add,
                 (get x [| Idx.Iterator t |], sp),
                 (get x [| Idx.Affine { symbols = [ (1, t) ]; offset = 1 } |], sp) );
           debug = "";
         })
  in
  let opt_lag = hand_built ~stmts:[ lag_nest ] ~tns_on_device:[ x ] ~tns_local:[] in
  p "retype lagged recurrence to Grid" (Sched.op_legality opt_lag (Sched.Retype { axis = t; ty = LL.Grid }));

  (* Sequential checking: verdicts against progressively applied code, stopping at illegal. *)
  let sched =
    [
      Sched.Retype { axis = i2; ty = LL.Grid };
      Sched.Retype { axis = k; ty = LL.Workgroup };
      Sched.Retype { axis = j2; ty = LL.Workgroup };
    ]
  in
  Stdio.printf "schedule_legality stops at the illegal op:\n";
  List.iter (Sched.schedule_legality opt sched) ~f:(fun (op, v) ->
      let label =
        match op with
        | Sched.Retype { axis; ty } ->
            Printf.sprintf "  Retype %s %s" (Idx.symbol_ident axis)
              (Sexp.to_string (LL.sexp_of_axis_type ty))
        | _ -> "  <op>"
      in
      p label v)
