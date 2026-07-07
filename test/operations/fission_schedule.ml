(* Kernel fission at cross-workgroup edges (docs/proposals/schedule-ir-optops.md §7): routines
   whose top-level statements form materialized producer/consumer pairs split into segment kernels
   launched back-to-back on the routine's stream, each with its own default schedule.

   Covered here:

   - Executed: a two-nest chain with a forced-materialized intermediate — before fission the whole
     routine ran 1×1; now both nests parallelize in separate kernels (values checked, source
     checked for [__seg0]/[__seg1] and per-backend parallel constructs).
   - Executed: a backward pass ([Train.grad_update]) — the [Zero_out] of the gradient and the
     bare/reduction statements segment away from the accumulation nest (gradient values checked).
   - Structural, backend-independent (analysis run for [backend_name:"metal"] on captured lowered
     code): segment count, per-segment hardware annotation, traced-store filtering, identity on
     backends without automatic scheduling.
   - Structural, hand-built [Low_level.t]: replication of hoisted scope-locals into consuming
     segments (option (b) v2), the merge-back fallback when a write between the definition and the
     consumer invalidates replication, and promotion of [Local] scratch stranded across a cut. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module LL = Ir.Low_level
module Sched = Ir.Schedule
module Asgns = Ir.Assignments
module Tn = Ir.Tnode
module Idx = Ir.Indexing
module IDX = Train.IDX

let () = Utils.settings.output_debug_files_in_build_directory <- true
let p name b = Stdio.printf "%s: %b\n" name b
let approx a b = Float.(abs (a - b) < 1e-4)
let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")
let on_cpu = String.is_substring backend_name ~substring:"cc"

let read_generated base_name =
  let ext = if String.is_substring backend_name ~substring:"metal" then ".metal" else ".c" in
  let ext = if String.is_substring backend_name ~substring:"cuda" then ".cu" else ext in
  let path = Stdlib.Filename.concat "build_files" (base_name ^ ext) in
  if Stdlib.Sys.file_exists path then Some (Stdio.In_channel.read_all path) else None

let has_hardware_regs src =
  String.is_substring src ~substring:"gid." || String.is_substring src ~substring:"blockIdx."

let has_parallel_construct src =
  String.is_substring src ~substring:"dispatch_apply"
  || String.is_substring src ~substring:"#pragma omp parallel for"

let count_substr src pattern =
  List.length (String.substr_index_all src ~may_overlap:false ~pattern)

let named name (comp : Asgns.comp) : Asgns.comp =
  { comp with asgns = Asgns.Block_comment (name, comp.asgns) }

let grad_of t = (Option.value_exn t.Tensor.diff).Tensor.grad

let rec has_declare_local (llc : LL.t) =
  match llc with
  | LL.Declare_local _ -> true
  | LL.Seq (a, b) -> has_declare_local a || has_declare_local b
  | LL.For_loop { body; _ } | LL.If { body; _ } -> has_declare_local body
  | _ -> false

let annotated seg = not (List.is_empty (LL.hardware_axes seg.LL.llc))

(* --- 1. Executed: materialized-intermediate chain --- *)
let () =
  let n = 512 in
  let av = Array.init (n * n) ~f:(fun i -> Float.of_int (i % 19) *. 0.5) in
  let bv = Array.init (n * n) ~f:(fun i -> Float.of_int (i % 23) -. 11.) in
  let expected = Array.init (n * n) ~f:(fun i -> (av.(i) +. bv.(i)) *. (av.(i) +. bv.(i))) in
  let a = TDSL.ndarray av ~label:[ "a" ] ~output_dims:[ n; n ] () in
  let b = TDSL.ndarray bv ~label:[ "b" ] ~output_dims:[ n; n ] () in
  let%op d = a + b in
  Train.set_materialized d.Tensor.value;
  let%op e = d *. d in
  let comp = named "fission_chain" (Train.forward e) in
  let ctx = Context.auto () in
  let ctx, routine = Context.compile ctx comp Ir.Indexing.Empty in
  let ctx = Context.run ctx routine in
  let got = Context.get_values ctx e.Tensor.value in
  p "chain values correct" (Array.for_all2_exn got expected ~f:approx);
  (* Run again: segment kernels share bindings/buffers; results must be stable. *)
  let ctx = Context.run ctx routine in
  let got2 = Context.get_values ctx e.Tensor.value in
  p "chain rerun deterministic" (Array.equal Float.( = ) got got2);
  (match read_generated "fission_chain__seg" with
  | None -> p "chain fissioned into two kernels" false
  | Some src ->
      let two_kernels =
        count_substr src "__seg0" >= 1 && count_substr src "__seg1" >= 1
      in
      p "chain fissioned into two kernels" two_kernels;
      let parallel_ok =
        if on_cpu then count_substr src "Pool-backed Grid rendering" >= 2 && has_parallel_construct src
        else has_hardware_regs src
      in
      p "both chain segments parallelize" parallel_ok);

  (* --- 2. Structural: the same chain analyzed for the metal backend, on captured lowered code
     (runs identically under every configured backend). --- *)
  let%op d2 = a + b in
  Train.set_materialized d2.Tensor.value;
  let%op e2 = d2 *. d2 in
  let comp2 = named "fission_capture" (Train.forward e2) in
  let stash = ref None in
  let _ctx, _routine =
    Context.compile
      ~lowered_transform:(fun opt ->
        stash := Some opt;
        opt)
      (Context.auto ()) comp2 Ir.Indexing.Empty
  in
  let opt = Option.value_exn ~here:[%here] !stash in
  let segs = Sched.maybe_default_schedules ~backend_name:"metal" ~static_indices:[] opt in
  p "capture: two segments" (List.length segs = 2);
  p "capture: both segments hardware-annotated" (List.for_all segs ~f:annotated);
  (match segs with
  | [ seg0; seg1 ] ->
      let mem seg tn = Hashtbl.mem seg.LL.traced_store tn in
      p "capture: traced stores are filtered"
        (mem seg0 d2.Tensor.value
        && (not (mem seg0 e2.Tensor.value))
        && mem seg1 d2.Tensor.value && mem seg1 e2.Tensor.value)
  | _ -> p "capture: traced stores are filtered" false);
  let identity = Sched.maybe_default_schedules ~backend_name:"interpreter" ~static_indices:[] opt in
  p "capture: identity on non-scheduling backends"
    (match identity with [ o ] -> phys_equal o opt | _ -> false)

(* --- 3. Structural, hand-built: scope-local replication, merge-back, Local promotion --- *)

let fresh_tn =
  let c = ref 950_000_000 in
  fun label dims ->
    Int.incr c;
    Tn.create (Tn.Specified Ir.Ops.single) ~id:!c ~label:[ label ]
      ~unpadded_dims:(lazy dims) ~padding:(lazy None) ()

let sp = Ir.Ops.single

let for_over ?(extent = 4096) sym body =
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
  }

let () =
  (* Shared hoisted scope-local [v]: two nests over a materialized producer/consumer edge, both
     reading [v]. The consumer segment must receive a replica of the definition. *)
  let a = fresh_tn "ha" [| 4096 |] in
  let m1 = fresh_tn "hm1" [| 4096 |] in
  let m2 = fresh_tn "hm2" [| 4096 |] in
  let vtn = fresh_tn "hv" [| 1 |] in
  let v = LL.get_scope vtn in
  let get tn idx = LL.Get (tn, [| idx |]) in
  let def_stmts =
    [
      LL.Declare_local { id = v; needs_init = false };
      LL.Set_local (v, get a (Idx.Fixed_idx 0));
    ]
  in
  let i = Idx.get_symbol () and j = Idx.get_symbol () in
  let nest1 =
    for_over i
      (LL.Set
         {
           tn = m1;
           idcs = [| Idx.Iterator i |];
           llsc = LL.Binop (Ir.Ops.Mul, (get a (Idx.Iterator i), sp), (LL.Get_local v, sp));
           debug = "";
         })
  in
  let nest2 ?(read_a = false) () =
    let rhs =
      if read_a then
        LL.Binop
          ( Ir.Ops.Add,
            (LL.Binop (Ir.Ops.Add, (get m1 (Idx.Iterator j), sp), (get a (Idx.Iterator j), sp)), sp),
            (LL.Get_local v, sp) )
      else LL.Binop (Ir.Ops.Add, (get m1 (Idx.Iterator j), sp), (LL.Get_local v, sp))
    in
    for_over j (LL.Set { tn = m2; idcs = [| Idx.Iterator j |]; llsc = rhs; debug = "" })
  in
  let opt =
    hand_built
      ~stmts:(def_stmts @ [ nest1; nest2 () ])
      ~tns_on_device:[ a; m1; m2 ] ~tns_local:[]
  in
  let segs = Sched.maybe_default_schedules ~backend_name:"metal" ~static_indices:[] opt in
  p "replication: two segments" (List.length segs = 2);
  p "replication: both segments annotated" (List.for_all segs ~f:annotated);
  p "replication: consumer segment carries a replica of the local's definition"
    (match segs with [ _; seg1 ] -> has_declare_local seg1.LL.llc | _ -> false);

  (* Invalid replication: a write to [a] (the definition's read) sits between the definition and
     the consumer, in its own forced segment. The consumer range merges back and runs serially,
     still carrying a valid replica computed before the offending write. *)
  let k = Idx.get_symbol () in
  let a_writer =
    for_over k (LL.Set { tn = a; idcs = [| Idx.Iterator k |]; llsc = LL.Constant 1.; debug = "" })
  in
  let opt =
    hand_built
      ~stmts:(def_stmts @ [ nest1; a_writer; nest2 ~read_a:true () ])
      ~tns_on_device:[ a; m1; m2 ] ~tns_local:[]
  in
  let segs = Sched.maybe_default_schedules ~backend_name:"metal" ~static_indices:[] opt in
  p "merge-back: two segments" (List.length segs = 2);
  (match segs with
  | [ seg0; seg1 ] ->
      p "merge-back: producer segment annotated, merged range serial"
        (annotated seg0 && not (annotated seg1));
      p "merge-back: merged segment still carries the replica" (has_declare_local seg1.LL.llc)
  | _ ->
      p "merge-back: producer segment annotated, merged range serial" false;
      p "merge-back: merged segment still carries the replica" false);

  (* Local scratch stranded across a cut: nest1 writes Local [d]; nest2 writes materialized [m];
     nest3 reads both — the [m] edge forces the cut, stranding [d], which must be promoted. *)
  let d = fresh_tn "hd" [| 4096 |] in
  let m = fresh_tn "hm" [| 4096 |] in
  let m3 = fresh_tn "hm3" [| 4096 |] in
  let s1 = Idx.get_symbol () and s2 = Idx.get_symbol () and s3 = Idx.get_symbol () in
  let w1 =
    for_over s1
      (LL.Set { tn = d; idcs = [| Idx.Iterator s1 |]; llsc = get a (Idx.Iterator s1); debug = "" })
  in
  let w2 =
    for_over s2
      (LL.Set
         {
           tn = m;
           idcs = [| Idx.Iterator s2 |];
           llsc = LL.Binop (Ir.Ops.Mul, (get a (Idx.Iterator s2), sp), (LL.Constant 2., sp));
           debug = "";
         })
  in
  let w3 =
    for_over s3
      (LL.Set
         {
           tn = m3;
           idcs = [| Idx.Iterator s3 |];
           llsc = LL.Binop (Ir.Ops.Add, (get d (Idx.Iterator s3), sp), (get m (Idx.Iterator s3), sp));
           debug = "";
         })
  in
  let opt = hand_built ~stmts:[ w1; w2; w3 ] ~tns_on_device:[ a; m; m3 ] ~tns_local:[ d ] in
  let plc = opt.LL.optimize_ctx.LL.placements in
  p "promotion: scratch starts unmaterialized" (not (Tn.Placements.is_materialized_peek plc d));
  let segs = Sched.maybe_default_schedules ~backend_name:"metal" ~static_indices:[] opt in
  p "promotion: two segments, both annotated"
    (List.length segs = 2 && List.for_all segs ~f:annotated);
  p "promotion: stranded Local scratch promoted to On_device"
    (Tn.Placements.is_materialized_peek plc d)

(* --- 4. Executed: backward pass — Zero_out and reduction statements segment away from the
   gradient accumulation nest. --- *)
let () =
  let n = 192 in
  let xv = Array.init (n * n) ~f:(fun i -> Float.of_int (i % 29) *. 0.125) in
  let x = TDSL.ndarray xv ~label:[ "x" ] ~output_dims:[ n; n ] () in
  let%op l = ({ w = uniform (); o = [ 192; 192 ] } *. x) ++ "...|... => 0" in
  let update = named "fission_bwd" (Train.grad_update l) in
  let ctx = Train.init_params (Context.auto ()) IDX.empty l in
  let routine = Train.to_routine ctx IDX.empty update in
  let ctx = Context.run (Context.context routine) routine in
  let w =
    List.find_exn (Set.to_list l.Tensor.params) ~f:(fun t ->
        String.equal (Tn.debug_name t.Tensor.value) "w")
  in
  let wg = Context.get_values ctx (grad_of w) in
  (* dl/dw = x. *)
  p "backward gradient values correct" (Array.for_all2_exn wg xv ~f:approx);
  match read_generated "fission_bwd__seg" with
  | None -> p "backward pass fissioned" false
  | Some src -> p "backward pass fissioned" (count_substr src "__seg0" >= 1)
