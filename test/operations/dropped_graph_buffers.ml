(* gh-ocannl-1747: what a dropped graph's buffers come back through. gh-ocannl-1745 made the session
   release the OCaml graph (dropped_graph_reclaimed); this test follows the memory behind it.

   - A lineage the caller releases, leaves first, gives back its buffers: the pool table's working
   pools and the device's used-memory gauge return to the baseline (on cc a GC finalizer frees the
   host buffer once the table lets go of it, so there the gauge is the GC's own witness), every pool
   the window allocated is counted freed, and the graph's nodes and the parameter's host-init array
   are reclaimed. Before gh-ocannl-1747 the device's writer-event table ([updating_for]) held every
   node a routine or an upload ever wrote, for the life of the device, so no dropped graph's nodes
   -- nor, through them, its host-init data -- were ever collected. - The negative control: an
   unreleased lineage keeps its pools in the pool table, exactly their bytes and nothing else
   (gh-ocannl-550: release is the only path back), though no longer its nodes. - Two retentions
   deferred to gh-ocannl-1762, pinned as they stand so that its fix flips them:
   [Train.forward_once]'s initialization context, which the caller never holds, keeps the
   parameter's pool after the returned context is released; and a fresh graph's host literal keeps a
   device-wide constant pool, and its node, until device teardown.

   Witnesses are [Weak] observers taken in [@inline never] builders that return no other reference
   to the graph; every reading clears the bounded analysis cache and follows [Gc.full_major] twice
   and [Gc.compact]. A graph's writer events are pruned at the next write, so each case runs one
   more graph before it reads.

   Every memory claim reads the process's own accounting ([Alloc_census], the backend's used gauge),
   never the driver's free bytes (gh-ocannl-1798): those are device-wide, so any other process on
   the GPU -- the suite's concurrent GPU tests included -- moves them between two readings. They are
   printed on stderr, as information. *)
open Base
open Ocannl
open Nn_blocks.DSL_modules
open Verdict.Claims
module IDX = Train.IDX
module Census = Ir.Alloc_census

let n = 1 lsl 21
let graphs = 3

let settle () =
  Stdlib.Gc.full_major ();
  Stdlib.Gc.full_major ();
  Stdlib.Gc.compact ()

let watch x =
  let w = Stdlib.Weak.create 1 in
  Stdlib.Weak.set w 0 (Some (Stdlib.Obj.repr x));
  w

let reclaimed w = not (Stdlib.Weak.check w 0)

type reading = {
  working : int;
  working_bytes : int;
  constant_bytes : int;
  pool_bytes : int;
  allocated : int;
  freed : int;
  used : int;
  free : int option;
}

(* The analysis cache keeps the last eight compiles' lowered code, hence their nodes: a bounded
   retention by design ([Low_level.clear_analysis_cache]), cleared so the readings see past it. *)
let read () =
  Ir.Low_level.clear_analysis_cache ();
  settle ();
  let c = Census.snapshot () in
  let probe = Context.auto () in
  {
    working = c.live_working_pools;
    working_bytes = c.live_working_bytes;
    constant_bytes = c.live_constant_bytes;
    pool_bytes = Census.live_pool_bytes c;
    allocated = c.working_pools_allocated + c.constant_pools_allocated;
    freed = c.pools_freed;
    used = Context.get_used_memory probe;
    free = Option.map (Context.get_free_and_total_memory probe) ~f:fst;
  }

let show name r =
  Stdio.eprintf
    "%s: working pools %d (%d bytes), constant %d bytes, pools allocated %d freed %d, used %d, \
     driver free %s (not part of the golden)\n\
     %!"
    name r.working r.working_bytes r.constant_bytes r.allocated r.freed r.used
    (Option.value_map r.free ~default:"-" ~f:Int.to_string)

(* How far the driver's free bytes moved from [before] to [after]: device-wide, so information
   only. *)
let show_driver name ~before ~after =
  Option.iter (Option.both before.free after.free) ~f:(fun (b, a) ->
      Stdio.eprintf "%s: driver free bytes moved by %d (not part of the golden)\n%!" name (a - b))

let backend = Context.backend_name (Context.auto ())

type built = {
  nodes : Stdlib.Obj.t Stdlib.Weak.t list;
  host_init : Stdlib.Obj.t Stdlib.Weak.t;
  bytes : int;
  computed : bool;
}

let value i j = Float.of_int ((i + j) % 7)

let param i =
  TDSL.param ~values:(Array.init n ~f:(value i)) (Printf.sprintf "dgb_w%d" i) ~output_dims:[ n ] ()

let host_init (w : Tensor.t) =
  Lazy.force (Option.value_exn ~here:[%here] (Ir.Host_inits.find w.value))

(* [t = w *. 2.] over a fresh parameter, initialized and run through the explicit lineage. *)
let[@inline never] lineage ~release i =
  let w = param i in
  let%op t = w *. 2. in
  let c1 = Train.init_params (Context.auto ()) IDX.empty t in
  let c2, r = Train.to_routine c1 IDX.empty (Train.forward t) in
  let c2 = Context.run c2 r in
  let got = Context.get_values c2 t.value in
  let nd = host_init w in
  if release then (
    Context.release c2;
    Context.release c1);
  {
    nodes = [ watch w.value; watch t.value ];
    host_init = watch nd;
    bytes = Ir.Ndarray.size_in_bytes nd;
    computed =
      Array.length got = n && Array.for_alli got ~f:(fun j v -> Float.equal v (2. *. value i j));
  }

let[@inline never] forward_once_released i =
  let w = param i in
  let%op t = w *. 2. in
  let ctx = Train.forward_once (Context.auto ()) t in
  let bytes = Ir.Ndarray.size_in_bytes (host_init w) in
  Context.release ctx;
  bytes

let[@inline never] literal_released i =
  let x =
    TDSL.ndarray
      (Array.init n ~f:(value i))
      ~label:[ Printf.sprintf "dgb_x%d" i ]
      ~output_dims:[ n ] ()
  in
  let%op t = x *. 2. in
  let ctx, r = Train.to_routine (Context.auto ()) IDX.empty (Train.forward t) in
  let ctx = Context.run ctx r in
  let bytes = Ir.Ndarray.size_in_bytes (host_init x) in
  Context.release ctx;
  (watch x.value, bytes)

let builds ~from f = List.init graphs ~f:(fun i -> f (from + i))

(* One more released graph, whose writes prune the events of the graphs before it. *)
let[@inline never] one_more i = ignore (lineage ~release:true i : built)

let () =
  Stdio.eprintf "backend: %s (not part of the golden)\n%!" backend;
  (* Warm-up: the device, its merge buffer and the code caches. *)
  one_more 0;
  case "a released lineage gives its buffers back" (fun () ->
      let before = read () in
      show "released: before" before;
      let built = builds ~from:1 (lineage ~release:true) in
      one_more 9;
      let after = read () in
      show "released: after" after;
      p_all "released: every graph computed t = 2 w" built ~f:(fun b -> b.computed);
      p "released: the live working pools are back to the baseline"
        (after.working = before.working && after.working_bytes = before.working_bytes);
      p "released: the device's used-memory gauge is back to the baseline" (after.used = before.used);
      show_driver "released" ~before ~after;
      p "released: every pool the window allocated was freed, and the pool bytes are back"
        (after.allocated > before.allocated
        && after.freed - before.freed = after.allocated - before.allocated
        && after.pool_bytes = before.pool_bytes);
      p_all "released: every graph's nodes are reclaimed" built ~f:(fun b ->
          List.for_all b.nodes ~f:reclaimed);
      p_all "released: every parameter's host-init array is reclaimed" built ~f:(fun b ->
          reclaimed b.host_init));
  case "an unreleased lineage keeps its pools, not its graph" (fun () ->
      let before = read () in
      show "unreleased: before" before;
      let built = builds ~from:11 (lineage ~release:false) in
      one_more 19;
      let after = read () in
      show "unreleased: after" after;
      let bytes = (List.hd_exn built).bytes in
      p_all "unreleased: every graph computed t = 2 w" built ~f:(fun b -> b.computed);
      p "unreleased: two working pools per graph stay in the pool table (gh-ocannl-550)"
        (after.working - before.working = 2 * graphs
        && after.working_bytes - before.working_bytes = 2 * graphs * bytes);
      show_driver "unreleased" ~before ~after;
      p "unreleased: exactly those pools are left unfreed, and the pool bytes grew by theirs alone"
        (after.allocated - before.allocated - (after.freed - before.freed) = 2 * graphs
        && after.pool_bytes - before.pool_bytes = 2 * graphs * bytes);
      p_all "unreleased: every graph's nodes are reclaimed all the same" built ~f:(fun b ->
          List.for_all b.nodes ~f:reclaimed));
  case "gh-ocannl-1762: forward_once's initialization context keeps the parameter's pool" (fun () ->
      let before = read () in
      let bytes = builds ~from:21 forward_once_released in
      one_more 29;
      let after = read () in
      show "forward_once: after" after;
      p "forward_once: one parameter pool per graph stays after releasing the returned context"
        (after.working - before.working = graphs
        && after.working_bytes - before.working_bytes = List.sum (module Int) bytes ~f:Fn.id));
  case "gh-ocannl-1762: a fresh graph's host literal keeps a device-wide constant pool" (fun () ->
      let before = read () in
      let built = builds ~from:31 literal_released in
      one_more 39;
      let after = read () in
      show "literal: after" after;
      p "literal: the constant bytes grew by each graph's literal"
        (after.constant_bytes - before.constant_bytes
        = List.sum (module Int) built ~f:(fun (_, bytes) -> bytes));
      p_all "literal: every literal's node is kept by the device's constant cache" built
        ~f:(fun (w, _) -> not (reclaimed w)))
