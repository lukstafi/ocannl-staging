(* gh-494: per-thread scratch value variance vs. the default annotator.

   A statement-crossing [Local] scratch node whose written cells mention no chain symbols used to be
   exempt from the cross-nest dependency edge ("every thread writes its whole private copy") — but
   that exemption implicitly assumed the written VALUE is thread-invariant. When the value depends
   on a chain symbol that does not also pin the written cell (here [tmp[u] := x[s1, u]] under
   chain [s0; s1]), a consumer thread's private copy holds its own chunk's last value while the
   serial reference holds the last chunk's: annotating both loops as hardware axes miscompiles on
   per-thread-copy backends (GPU registers).

   The fix makes such writes count as chain-mentioning (the edge forms) and adds a per-edge
   value-invariance condition at each trim level, so the alignment search serializes the offending
   chain loop instead: the outer loop stays parallel, the value-feeding loop becomes serial, and
   per-thread copies then match the serial semantics. A scratch node whose value is
   thread-invariant keeps the full two-loop chain (regression guard). *)

open Base
module Tn = Ir.Tnode
module LL = Ir.Low_level
module Idx = Ir.Indexing
module Sched = Ir.Schedule

let p name b = Stdio.printf "%s: %b\n" name b

let fresh_tn =
  let c = ref 960_000_000 in
  fun label dims ->
    Int.incr c;
    Tn.create (Tn.Specified Ir.Ops.single) ~id:!c ~label:[ label ]
      ~unpadded_dims:(lazy dims)
      ~padding:(lazy None)
      ()

let sp = Ir.Ops.single

let for_over ?(extent = 64) sym body =
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
  }

let hardware_syms (sched : Sched.schedule) : Idx.symbol list =
  List.filter_map sched ~f:(function
    | Sched.Retype { axis; ty = LL.Grid | LL.Workgroup } -> Some axis
    | Sched.Split { axis; outer = LL.Grid | LL.Workgroup; _ }
    | Sched.Split { axis; inner = LL.Grid | LL.Workgroup; _ } ->
        Some axis
    | _ -> None)

(* Two nests sharing a [Local] scratch [tmp]: nest 1 rewrites [tmp] under its chain and also makes
   a materialized write qualifying the chain; nest 2 reads [tmp]. [~variant] selects whether the
   scratch value depends on nest 1's inner chain symbol [s1]. *)
let build ~variant =
  let x = fresh_tn "x" [| 64; 4 |] in
  let out1 = fresh_tn "out1" [| 64; 64 |] in
  let out2 = fresh_tn "out2" [| 64; 64 |] in
  let tmp = fresh_tn "tmp" [| 4 |] in
  let s0 = Idx.get_symbol () and s1 = Idx.get_symbol () and u = Idx.get_symbol () in
  let t0 = Idx.get_symbol () and t1 = Idx.get_symbol () in
  let tmp_value =
    if variant then LL.Get (x, [| Idx.Iterator s1; Idx.Iterator u |])
    else LL.Get (x, [| Idx.Fixed_idx 0; Idx.Iterator u |])
  in
  let nest1 =
    for_over s0
      (for_over s1
         (LL.unflat_lines
            [
              for_over ~extent:4 u
                (LL.Set { tn = tmp; idcs = [| Idx.Iterator u |]; llsc = tmp_value; debug = "" });
              LL.Set
                {
                  tn = out1;
                  idcs = [| Idx.Iterator s0; Idx.Iterator s1 |];
                  llsc = LL.Get (tmp, [| Idx.Fixed_idx 0 |]);
                  debug = "";
                };
            ]))
  in
  let nest2 =
    for_over t0
      (for_over t1
         (LL.Set
            {
              tn = out2;
              idcs = [| Idx.Iterator t0; Idx.Iterator t1 |];
              llsc = LL.Get (tmp, [| Idx.Fixed_idx 1 |]);
              debug = "";
            }))
  in
  let opt = hand_built ~stmts:[ nest1; nest2 ] ~tns_on_device:[ x; out1; out2 ] ~tns_local:[ tmp ] in
  (opt, [ s0; t0 ], [ s1; t1 ])

let () =
  let opt, outer, inner = build ~variant:true in
  let sched = Sched.default_gpu ~min_parallel:4 opt in
  let hw = hardware_syms sched in
  let mem s = List.mem hw s ~equal:Idx.equal_symbol in
  p "variant scratch: schedule is nonempty" (not (List.is_empty sched));
  p "variant scratch: outer chain loops stay parallel" (List.for_all outer ~f:mem);
  p "variant scratch: value-feeding chain loops are serialized"
    (List.for_all inner ~f:(fun s -> not (mem s)));
  let opt, outer, inner = build ~variant:false in
  let sched = Sched.default_gpu ~min_parallel:4 opt in
  let hw = hardware_syms sched in
  let mem s = List.mem hw s ~equal:Idx.equal_symbol in
  p "invariant scratch: full two-loop chains kept"
    (List.for_all (outer @ inner) ~f:mem);
  ignore sp
