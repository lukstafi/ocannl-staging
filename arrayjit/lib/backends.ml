open Base
open Ir
module Tn = Tnode
module Schedulers = Schedulers
open Backend_intf
open Backend_impl

let _get_local_debug_runtime = Utils.get_local_debug_runtime

[%%global_debug_log_level 0]

(* export OCANNL_LOG_LEVEL_BACKENDS=9 to enable debugging into the log_files/ directory. *)
[%%global_debug_log_level_from_env_var "OCANNL_LOG_LEVEL_BACKENDS"]

(* gh-ocannl-344: pure planner for the pool allocator. Lays out a sequence of [(size, alignment)]
   allocations (in order) into one or more pools so that no pool's bumped extent exceeds [cap] bytes
   -- the per-pool 4 GB ceiling for uint32 offsets when [large_models = false]. Returns, per input
   item, its [(segment_index, byte_offset)], plus the byte size of each segment (pool). Raises
   [Utils.User_error] (naming [what] and [debug_name i]) if a single item exceeds [cap], since no
   pool can hold it without uint64 offsets. Factored out so the segmenting/cap behavior is unit
   testable with synthetic sizes (it does not need real device memory). *)
let plan_pool_segments ~(cap : int) ~(what : string) ~(debug_name : int -> string)
    (items : (int * int) list) : (int * int) list * int list =
  let align_up off a = if a <= 1 then off else (off + a - 1) / a * a in
  let seg = ref 0 and bump = ref 0 in
  let closed =
    ref []
    (* completed segment sizes, reversed *)
  in
  let assignments =
    List.mapi items ~f:(fun i (size, align) ->
        if size > cap then
          raise
          @@ Utils.User_error
               (Printf.sprintf
                  "%s: tensor node %s needs %d bytes, over the %d-byte per-pool cap; set \
                   large_models=true for uint64 offsets"
                  what (debug_name i) size cap);
        let offset = align_up !bump align in
        if offset + size > cap then (
          (* Close the current pool and open a new one starting this item at offset 0. *)
          closed := !bump :: !closed;
          Int.incr seg;
          bump := size;
          (!seg, 0))
        else (
          bump := offset + size;
          (!seg, offset)))
  in
  let segment_sizes = if List.is_empty items then [] else List.rev (!bump :: !closed) in
  (assignments, segment_sizes)

(* gh-ocannl-489 liveness-based buffer aliasing: the config gates. Read per call (not cached in a
   lazy) so tests can flip them via the environment between compilation sections of one process. *)
let buffer_aliasing () = Utils.get_global_flag ~default:false ~arg_name:"buffer_aliasing"
let log_buffer_aliasing () = Utils.get_global_flag ~default:false ~arg_name:"log_buffer_aliasing"

(* gh-ocannl-489: liveness-aware companion of [plan_pool_segments] -- lays out a sequence of [(size,
   alignment, precision_class, live_span)] allocations into ONE pool where two allocations may
   occupy overlapping byte ranges iff both have a live span (planner-eligible), the spans are
   disjoint as closed intervals, and the precision classes are equal (sharing bytes across effective
   types would be C strict-aliasing UB in the single-procedure CPU backends). A [None] span means
   always-live: the node conflicts with everything, including other always-live nodes. Placement is
   greedy by decreasing size (ties broken by input order, keeping the layout deterministic): each
   item lands at the lowest suitably-aligned offset that avoids all conflicting placed items.
   Returns per-item byte offsets in input order plus the pool's total size, or [None] when the
   layout exceeds [cap] (the caller falls back to [plan_pool_segments]' bump packing, which segments
   at the cap). Pure, so the coloring is unit testable with synthetic sizes. *)
let plan_arena_offsets ~(cap : int) (items : (int * int * string * (int * int) option) list) :
    (int list * int) option =
  let arr = Array.of_list items in
  let n = Array.length arr in
  let order =
    List.sort (List.init n ~f:Fn.id) ~compare:(fun i j ->
        let si, _, _, _ = arr.(i) and sj, _, _, _ = arr.(j) in
        match compare_int sj si with 0 -> compare_int i j | c -> c)
  in
  let conflicts i j =
    let _, _, ci, spi = arr.(i) and _, _, cj, spj = arr.(j) in
    match (spi, spj) with
    | Some (lo1, hi1), Some (lo2, hi2) when String.equal ci cj -> not (hi1 < lo2 || hi2 < lo1)
    | _ -> true
  in
  let offsets = Array.create ~len:n 0 in
  let placed = ref [] in
  let total = ref 0 in
  List.iter order ~f:(fun i ->
      let size, align, _, _ = arr.(i) in
      let align_up off = if align <= 1 then off else (off + align - 1) / align * align in
      let ranges =
        List.filter_map !placed ~f:(fun j ->
            if conflicts i j then
              let sj, _, _, _ = arr.(j) in
              Some (offsets.(j), offsets.(j) + sj)
            else None)
        |> List.sort ~compare:(fun (a, _) (b, _) -> compare_int a b)
      in
      let off =
        List.fold ranges ~init:(align_up 0) ~f:(fun off (lo, hi) ->
            if off + size <= lo then off else max off (align_up hi))
      in
      offsets.(i) <- off;
      placed := i :: !placed;
      total := max !total (off + size));
  if !total > cap then None else Some (Array.to_list offsets, !total)

let size_in_bytes_of (key : Tn.t) =
  let prec = Lazy.force key.Tn.storage_prec in
  Array.fold (Lazy.force key.Tn.dims) ~init:1 ~f:( * ) * Ops.prec_in_bytes prec

(* gh-ocannl-489 liveness-based buffer aliasing, the per-compile planning half: live spans over the
   FINAL segment sequence (post-schedule, post-fission -- cross-nest merges and kernel cuts change
   which accesses can interleave, so pre-schedule spans would be unsound), filtered down to the
   aliasing-eligible nodes. Eligible = in-context working nodes that the routine writes before any
   read: not read-only, not read-before-write (excludes recurrent nodes and reliance on alloc-time
   zeros), not observable (host reads must stay valid), not host-initialized (live before the
   program), not constants, not the merge node. Callers run this BEFORE backend codegen: the
   candidates registered on [optimize_ctx.alias_candidates] make codegen drop the [restrict]
   qualifier on their kernel parameters (an actually-aliased [restrict] pair is a miscompile).
   Position granularity: statements on backends that synchronize between top-level statements (the C
   backends -- no hardware workgroup binding, parallel dispatches join), segments otherwise (GPU
   kernels lack grid-wide sync between statements). [lowered] is the whole-routine (pre-fission)
   code, whose traced store and placements decide eligibility; [segments] is the final kernel
   sequence in execution order. *)
let plan_alias_spans ~(name : string) ~(limits : hardware_limits) ~(lowered : Low_level.optimized)
    ~(segments : Low_level.optimized list) : (Tn.t, int * int) Base.Hashtbl.t option =
  if not (buffer_aliasing ()) then None
  else
    let stmt_serial = Option.is_none limits.max_threads_per_workgroup in
    match
      Low_level.buffer_access_spans ~stmt_serial
        (List.map segments ~f:(fun seg -> seg.Low_level.llc))
    with
    | None -> None
    | Some spans ->
        let plc = lowered.Low_level.optimize_ctx.placements in
        Hashtbl.filter_keys_inplace spans ~f:(fun tn ->
            match Hashtbl.find lowered.Low_level.traced_store tn with
            | None -> false
            | Some node ->
                Tn.Placements.is_in_context_force plc tn 45
                && (not node.Low_level.read_only)
                && (not node.Low_level.read_before_write)
                && (node.Low_level.zeroed_out || node.Low_level.has_assignment)
                (* Written but never consumed in-routine means the write is an EXPORT for later
                   routines (parameter initialization is the ubiquitous case) -- its lifetime
                   extends past this routine, so it must keep dedicated bytes. *)
                && node.Low_level.read_by_other
                && (not (Tn.is_observable tn))
                && (not (Host_inits.mem tn))
                && (not (Tn.Placements.known_constant plc tn))
                && not (Option.exists lowered.Low_level.merge_node ~f:(Tn.equal tn)));
        if Hashtbl.is_empty spans then None
        else (
          Hashtbl.iter_keys spans ~f:(Hash_set.add lowered.Low_level.optimize_ctx.alias_candidates);
          if log_buffer_aliasing () then
            List.iter
              (Hashtbl.to_alist spans
              |> List.sort ~compare:(fun (_, (a, _)) (_, (b, _)) -> compare_int a b))
              ~f:(fun (tn, (lo, hi)) ->
                Stdlib.Printf.eprintf "buffer aliasing: %s: candidate %s live [%d, %d]\n%!" name
                  (Tn.debug_name tn) lo hi);
          Some spans)

type footprint = {
  fp_total : int;
  fp_working : int;
  fp_constants : int;
  fp_dedicated : int;
  fp_planned : int;
  fp_nodes : int;
}
[@@deriving sexp_of, equal]

(* gh-ocannl-498: the byte footprint a routine's placement vector implies, scored with the same
   liveness/arena machinery the allocator uses ([plan_alias_spans] + [plan_arena_offsets]) but
   without a device or a context. This is the cost side [Low_level.flip_candidates] does not carry:
   the recompute-cost bound says what inlining a node COSTS, this says what it SAVES.

   Scored over the routine's whole in-context node set, not a context's allocation delta, so the
   number depends only on the code and the placements -- the precondition for a deterministic budget
   selector ([Context.plan_memory_budget]) whose choices do not drift with how much of the graph a
   particular context has already allocated. It is therefore a MODEL of the peak, not a prediction
   of [Context.get_used_memory]: the real allocator skips nodes a prior context already holds, and
   pool bases are page-rounded by the driver.

   Enumeration is canonical ([Tn.compare], i.e. by uid) rather than [traced_store] order, so the
   greedy arena coloring is reproducible across processes; [allocate_delta] uses store order to keep
   pool ids stable and can therefore break size ties differently. *)
let score_footprint ~(backend_name : string) ~(limits : hardware_limits)
    ~(static_indices : Indexing.static_symbol list) (lowered : Low_level.optimized) : footprint =
  let lowered =
    if buffer_aliasing () then
      { lowered with Low_level.llc = Low_level.sink_zero_outs lowered.Low_level.llc }
    else lowered
  in
  let segments = Schedule.maybe_default_schedules ~backend_name ~limits ~static_indices lowered in
  let spans = plan_alias_spans ~name:"<footprint>" ~limits ~lowered ~segments in
  (* Schedule ops applied per segment can CREATE tnodes the pre-fission store has never seen (a
     hoisted [Stage] registers its packed-constant tile), and [allocate_delta] enumerates the store
     it is handed -- so score the union, as the compile's own fold-back does. *)
  let store = Hashtbl.copy lowered.Low_level.traced_store in
  List.iter segments ~f:(fun seg ->
      Hashtbl.iteri seg.Low_level.traced_store ~f:(fun ~key ~data ->
          if not (Hashtbl.mem store key) then Hashtbl.add_exn store ~key ~data));
  let plc = lowered.Low_level.optimize_ctx.placements in
  let working = ref [] and constants = ref [] in
  Hashtbl.iteri store ~f:(fun ~key ~data:node ->
      if Tn.Placements.is_in_context_force plc key 47 then
        if node.Low_level.read_only || Tn.Placements.known_constant plc key then
          constants := key :: !constants
        else working := key :: !working);
  let canonical l = List.sort !l ~compare:Tn.compare in
  let working = canonical working and constants = canonical constants in
  let items group =
    List.map group ~f:(fun key ->
        ( size_in_bytes_of key,
          max (Ops.prec_in_bytes (Lazy.force key.Tn.storage_prec)) Ops.buffer_alignment ))
  in
  let cap = if Utils.settings.large_models then Int.max_value else 0x1_0000_0000 in
  let bump ~what group =
    let _, segment_sizes =
      plan_pool_segments ~cap ~what
        ~debug_name:(fun i -> Tn.debug_name (List.nth_exn group i))
        (items group)
    in
    List.fold segment_sizes ~init:0 ~f:( + )
  in
  let fp_dedicated = bump ~what:"Backends.score_footprint" working in
  let fp_planned =
    Option.value_map spans ~default:0 ~f:(fun spans -> List.count working ~f:(Hashtbl.mem spans))
  in
  let arena =
    Option.bind spans ~f:(fun spans ->
        plan_arena_offsets ~cap
          (List.map2_exn working (items working) ~f:(fun key (size, align) ->
               ( size,
                 align,
                 Ops.prec_string (Lazy.force key.Tn.storage_prec),
                 Hashtbl.find spans key ))))
  in
  let fp_working = match arena with Some (_, total) -> total | None -> fp_dedicated in
  let fp_constants = bump ~what:"Backends.score_footprint" constants in
  {
    fp_total = fp_working + fp_constants;
    fp_working;
    fp_constants;
    fp_dedicated;
    fp_planned;
    fp_nodes = List.length working + List.length constants;
  }

(* gh-ocannl-489: whether [tn]'s buffer shares bytes with another node's in [ctx_buffers] -- i.e.
   the liveness planner ([plan_arena_offsets] via [allocate_delta]) placed it at an overlapping
   range. Exact and layout-derived: bump-packed tenants of one pool have disjoint ranges and never
   trigger this. Used by the read guards: an aliased node's values are not preserved past its last
   in-routine read, so host reads, cross-device reads and later-routine inputs must fail loudly. *)
let buffer_overlaps (ctx_buffers : Backend_intf.ctx_buffers) tn (loc : Backend_intf.buffer_loc) :
    bool =
  let size = size_in_bytes_of tn in
  Map.existsi ctx_buffers ~f:(fun ~key ~data:(l : Backend_intf.buffer_loc) ->
      (not (Tn.equal key tn))
      && l.pool_id = loc.pool_id
      && l.offset < loc.offset + size
      && loc.offset < l.offset + size_in_bytes_of key)

let aliased_read_error ~what tn =
  raise
  @@ Utils.User_error
       (Printf.sprintf
          "%s: tensor node %s was buffer-aliased by the liveness memory planner (config \
           buffer_aliasing): its buffer is shared with other nodes and its values are not \
           preserved past its last read within the routine that computes it. Mark it as observable \
           or materialized-and-read (e.g. via Tnode.set_observable) before compiling that routine, \
           or disable buffer_aliasing."
          what (Tnode.debug_name tn))

(* Dynamic backstop for merge-buffer verification: runs as the first work of a consumer's schedule
   and checks the node most recently scheduled into the stream's merge buffer. The primary check is
   now the static [check_merge_buffer_static] performed at link time (gh-ocannl-288); this remains
   as a defensive backstop for transfers that are scheduled without a downstream link. *)
let check_merge_buffer device ~code_node =
  let name = function Some tn -> Tnode.debug_name tn | None -> "none" in
  match (device.updating_for_merge_buffer, code_node) with
  | _, None -> ()
  | Some (actual, _), Some expected when Tnode.equal actual expected -> ()
  | _ ->
      raise
      @@ Utils.User_error
           ("Merge buffer mismatch, on device: "
           ^ name (Option.map ~f:fst device.updating_for_merge_buffer)
           ^ ", expected by code: " ^ name code_node)

(* Static counterpart of [check_merge_buffer]: verifies at link time -- before any schedule runs --
   that the merge-buffer node statically recorded on the linked [context] (by a [device_to_device]
   transfer routine, see {!Add_buffer_retrieval_and_syncing.device_to_device}) matches the node the
   linked [code] expects. This is the "static verification in the right direction" of gh-ocannl-288:
   the transfer routine's context chains naturally into the consumer's link. *)
let check_merge_buffer_static ~merge_buffer_node ~code_node =
  let name = function Some tn -> Tnode.debug_name tn | None -> "none" in
  match (merge_buffer_node, code_node) with
  | _, None -> ()
  | Some actual, Some expected when Tnode.equal actual expected -> ()
  | _ ->
      raise
      @@ Utils.User_error
           ("Merge buffer mismatch at link time: the linked context provides "
          ^ name merge_buffer_node ^ ", but the linked code expects " ^ name code_node)

module Add_buffer_retrieval_and_syncing (Backend : No_buffer_retrieval_or_syncing) = struct
  let wait_for_ready ~dst ~src tn =
    let s = src.device in
    let d = dst.device in
    (* TODO: maybe it's worthwhile to clean up s.updating_for every now and then. *)
    Hashtbl.find s.updating_for tn
    |> Option.iter ~f:(fun upd_e ->
        if not (equal_device s d || Backend.is_done upd_e) then Backend.will_wait_for dst upd_e)

  (* Shared allocator seam: mints a deterministic per-device [pool_id] (advancing
     [device.next_pool_id] in the caller's tnode-iteration order), allocates the slab through the
     backend's int-in/int-out API, and returns the [buffer_loc]. Phase-1 policy is one pool per
     tnode at offset 0 -- byte-for-byte equivalent to the old per-tnode allocation. [zero_init]
     selects the old [alloc_zeros] vs [alloc_array] behavior. *)
  let allocate (device : _ Backend_intf.device) (tn : Tn.t) ~zero_init : Backend_intf.buffer_loc =
    let pool_id = device.next_pool_id in
    device.next_pool_id <- pool_id + 1;
    let prec = Lazy.force tn.Tn.storage_prec in
    (* Compute the byte size from dims*prec rather than forcing [tn.size_in_bytes], to keep the
       node's debug printout (and lazy-forcing behavior) byte-for-byte as before. *)
    let size_in_bytes =
      Array.fold (Lazy.force tn.Tn.dims) ~init:1 ~f:( * ) * Ops.prec_in_bytes prec
    in
    let mode = Option.map tn.Tn.memory_mode_intent ~f:fst in
    Backend.alloc_pool ?mode device ~pool_id ~size_in_bytes ~alignment:(Ops.prec_in_bytes prec);
    (* gh-ocannl-550: the OTHER shared allocation site — a [from_host] or [copy] whose destination node
       is not in the context yet allocates here, not through [allocate_delta]. Its slabs go into the
       same backend pool tables and are freed by the same context [finalize], so leaving them
       uncounted made the census silently underreport in data-loading and context-copy workflows. Not
       working-vs-constant: this path is a working buffer by construction (a host transfer's
       destination). *)
    Alloc_census.record_pool ~device_id:device.device_id ~pool_id ~constant:false ~size_in_bytes;
    if zero_init then Backend.memset_zero device ~pool_id ~offset:0 ~size_in_bytes;
    { pool_id; offset = 0 }

  let%track3_sexp to_host (ctx : Backend.context) (tn : Tn.t) (hosted : Ndarray.t) =
    match Map.find ctx.ctx_buffers tn with
    | Some loc ->
        if buffer_overlaps ctx.ctx_buffers tn loc then aliased_read_error ~what:"reading to host" tn;
        [%log "copying", Tn.debug_name tn, "at", (loc : Backend_intf.buffer_loc), "to host"];
        (* No cross-stream writer synchronization needed: multi-streaming was removed
           (gh-ocannl-341). Only one stream exists per device, so there are no concurrent
           cross-stream writes to wait for before this device-to-host copy. *)
        Backend.to_host ~src:ctx ~src_loc:loc hosted;
        true
    | None -> false

  let update_writer_event ?e ctx tn =
    let s = ctx.device in
    let e = Option.value_or_thunk e ~default:(fun () -> Backend.all_work s) in
    match tn with
    | Assignments.Node tn -> Hashtbl.update s.updating_for tn ~f:(fun _ -> e)
    | Assignments.Merge_buffer tn ->
        (* Note: the previous event does not need to be done! *)
        s.updating_for_merge_buffer <- Some (tn, Some e)

  let%track3_sexp from_host (ctx : Backend.context) (tn : Tn.t) (hosted : Ndarray.t) =
    match Map.find ctx.ctx_buffers tn with
    | Some dst ->
        (* A host write to an aliased buffer would be clobbered by the next run of the aliasing
           routine (and would clobber co-tenants meanwhile) -- surely a mistake; fail loudly. *)
        if buffer_overlaps ctx.ctx_buffers tn dst then
          aliased_read_error ~what:"writing from host" tn;
        (* No cross-stream reader synchronization needed: multi-streaming was removed
           (gh-ocannl-341). Only one stream exists per device, so there are no concurrent
           cross-stream readers to wait for before this host-to-device upload. *)
        [%log "copying", Tn.debug_name tn, "to", (dst : Backend_intf.buffer_loc), "from host"];
        Backend.from_host ~dst:ctx ~dst_loc:dst hosted;
        update_writer_event ctx @@ Node tn;
        true
    | None -> false

  (* gh-ocannl-550: [allocate] roots a pool in the backend table, and the transfer that follows adds
     its location to the context only on success — so a failing upload leaves a pool no context can
     ever reach, and therefore no [Context.release] can reclaim. Frees the one pool this operation
     minted; unlike [allocate_delta]'s unwind there is no constant-cache involvement here (a transfer
     destination is a working buffer by construction), so this needs nothing beyond the free. *)
  let with_transfer_pool device (loc : Backend_intf.buffer_loc) ~f =
    match f () with
    | result -> result
    | exception exn ->
        let backtrace = Stdlib.Printexc.get_raw_backtrace () in
        (try
           Backend.await device;
           Option.iter Backend.free_pool ~f:(fun free -> free device ~pool_id:loc.pool_id);
           Alloc_census.forget_pool ~device_id:device.device_id ~pool_id:loc.pool_id
         with _ -> ());
        Stdlib.Printexc.raise_with_backtrace exn backtrace

  let%track3_sexp init_from_host (ctx : Backend.context) (tn : Tn.t) (hosted : Ndarray.t) =
    match Map.find ctx.ctx_buffers tn with
    | None ->
        (* No zero-init: we are immediately copying from host. *)
        let dst = allocate ctx.device tn ~zero_init:false in
        with_transfer_pool ctx.device dst ~f:(fun () ->
            [%log "copying", Tn.debug_name tn, "to", (dst : Backend_intf.buffer_loc), "from host"];
            Backend.from_host ~dst:ctx ~dst_loc:dst hosted;
            update_writer_event ctx @@ Node tn;
            { ctx with ctx_buffers = Map.add_exn ctx.ctx_buffers ~key:tn ~data:dst })
    | Some _ ->
        raise
        @@ Utils.User_error
             ("init_from_host: input context already contains tensor node " ^ Tn.debug_name tn
            ^ ", for device " ^ Backend.get_name ctx.device)

  (* [device_to_device] builds a transfer routine instead of scheduling the copy directly. The
     caller schedules it (via [Task.run r.schedule]) or links a consumer against [r.context]. For
     the [Copy] case, [r.context]'s [merge_buffer_node] records the produced node statically, so
     that [link] can verify it against a consumer's [expected_merge_node] at link time -- the
     "static verification in the right direction" of gh-ocannl-288. *)
  let%track3_sexp device_to_device (tn : Tn.t) ~into_merge_buffer ~(dst : Backend.context)
      ~(src : Backend.context) : Backend.context routine option =
    match Map.find src.ctx_buffers tn with
    | None -> None
    | Some s_loc -> (
        if buffer_overlaps src.ctx_buffers tn s_loc then
          aliased_read_error ~what:"device_to_device source" tn;
        match into_merge_buffer with
        | No -> (
            match Map.find dst.ctx_buffers tn with
            | None -> None
            | Some d_loc ->
                (* Same device + same location => physically the same buffer; nothing to copy. *)
                if equal_device src.device dst.device && [%equal: buffer_loc] s_loc d_loc then None
                else
                  let context = Backend.make_child dst in
                  let description =
                    "device_to_device " ^ Tn.debug_name tn ^ " from " ^ Backend.get_name src.device
                    ^ " to " ^ Backend.get_name dst.device
                  in
                  let work () =
                    wait_for_ready ~dst ~src tn;
                    Backend.(
                      device_to_device tn ~into_merge_buffer ~dst_loc:(Some d_loc) ~dst
                        ~src_loc:s_loc ~src);
                    update_writer_event dst @@ Node tn;
                    [%log
                      "copying",
                      Tn.debug_name tn,
                      "from",
                      Backend.get_name src.device,
                      "to",
                      Backend.get_name dst.device]
                  in
                  let schedule = Task.Task { context_lifetime = (src, dst); description; work } in
                  Some
                    {
                      context;
                      schedule;
                      bindings = [];
                      name = description;
                      inputs = Set.singleton (module Tnode) tn;
                      merge_buffer_input = None;
                      outputs = Set.singleton (module Tnode) tn;
                    })
        | Copy ->
            let context = Backend.make_child dst ~merge_buffer_node:(Some tn) in
            let description =
              "device_to_device " ^ Tn.debug_name tn ^ " into merge buffer from "
              ^ Backend.get_name src.device
            in
            let work () =
              wait_for_ready ~dst ~src tn;
              Backend.(
                device_to_device tn ~into_merge_buffer ~dst_loc:None ~dst ~src_loc:s_loc ~src);
              update_writer_event dst @@ Merge_buffer tn;
              [%log "copy into merge buffer", Tn.debug_name tn, "from", Backend.get_name src.device]
            in
            let schedule = Task.Task { context_lifetime = (src, dst); description; work } in
            Some
              {
                context;
                schedule;
                bindings = [];
                name = description;
                inputs = Set.singleton (module Tnode) tn;
                merge_buffer_input = None;
                outputs = Set.empty (module Tnode);
              })

  let%track3_sexp init_from_device (tn : Tn.t) ~(dst : Backend.context) ~(src : Backend.context) =
    match Map.find src.ctx_buffers tn with
    | None ->
        raise
        @@ Utils.User_error
             ("init_from_device: tensor node " ^ Tn.debug_name tn ^ " is not in input context "
            ^ Backend.get_name src.device ^ ", for device " ^ Backend.get_name dst.device)
    | Some s_loc -> (
        (* gh-ocannl-489: same source-read guard as [device_to_device] -- an aliased node's bytes
           are clobbered, so copying them into a fresh context would silently preserve the wrong
           value. *)
        if buffer_overlaps src.ctx_buffers tn s_loc then
          aliased_read_error ~what:"init_from_device source" tn;
        wait_for_ready ~dst ~src tn;
        match Map.find dst.ctx_buffers tn with
        | Some _ ->
            raise
            @@ Utils.User_error
                 ("init_from_device: tensor node " ^ Tn.debug_name tn
                ^ " already in output context " ^ Backend.get_name dst.device ^ ", for device "
                ^ Backend.get_name src.device)
        | None ->
            (* No zero-init: we are immediately copying from another device. *)
            let d_loc = allocate dst.device tn ~zero_init:false in
            with_transfer_pool dst.device d_loc ~f:(fun () ->
                Backend.(
                  device_to_device tn ~into_merge_buffer:No ~dst_loc:(Some d_loc) ~dst
                    ~src_loc:s_loc ~src);
                update_writer_event dst @@ Node tn;
                [%log
                  "copying",
                  Tn.debug_name tn,
                  "from",
                  Backend.get_name src.device,
                  "to",
                  Backend.get_name dst.device];
                { dst with ctx_buffers = Map.add_exn dst.ctx_buffers ~key:tn ~data:d_loc }))

  type r = Backend.context routine [@@deriving sexp_of]

  let sync_routine (r : r) : r =
    (* Host transfers are no longer automatic (gh-ocannl-333): all CPU-side access goes through
       explicit, on-demand [Context] transfers. [sync_routine] now only records the post-execution
       writer event for the routine's outputs (used for device-side ordering and merge buffers). *)
    let s = r.context.device in
    let post () =
      let e = Backend.all_work s in
      Set.iter r.outputs ~f:(fun tn -> update_writer_event ~e r.context @@ Node tn)
    in
    { r with schedule = Task.(append ~work:post r.schedule) }

  let sync_device device =
    Backend.await device;
    device.updating_for_merge_buffer <- None;
    Hashtbl.clear device.updating_for
end

let%track6_sexp lower_assignments optim_ctx ?name bindings asgns =
  (* Fork the lineage state (computations and placements) so this compile's decisions do not leak
     into the incoming context or into sibling compiles from the same context
     (docs/proposals/context-scoped-memory-modes.md). The forked state travels with the code and
     reaches the child context at link time. *)
  let optim_ctx = Low_level.copy_optimize_ctx optim_ctx in
  let name : string =
    Option.value_or_thunk name ~default:(fun () -> Assignments.get_name_exn asgns)
  in
  let unoptim_ll_source = Utils.output_to_build_file ~fname:(name ^ "-unoptimized.ll") in
  let ll_source = Utils.output_to_build_file ~fname:(name ^ ".ll") in
  let cd_source = Utils.output_to_build_file ~fname:(name ^ ".cd") in
  ( name,
    Assignments.lower optim_ctx ~unoptim_ll_source ~ll_source ~cd_source ~name
      (Indexing.bound_symbols bindings) asgns )

let lower_batch_assignments optim_ctx ?names ?occupancy bindings asgns_l =
  (* One fork for the whole batch: the batch is a single compilation unit, so its members share the
     lineage state, but the batch as a whole stays hermetic w.r.t. sibling compiles. *)
  let optim_ctx = Low_level.copy_optimize_ctx optim_ctx in
  let names =
    Option.value_or_thunk names ~default:(fun () ->
        Array.map asgns_l ~f:(fun asgns -> Assignments.get_name_exn asgns))
  in
  let prefix_name = String.(strip ~drop:(equal_char '_') @@ common_prefix @@ Array.to_list names) in
  let unoptim_ll_source = Utils.output_to_build_file ~fname:(prefix_name ^ "-unoptimized.ll") in
  let ll_source = Utils.output_to_build_file ~fname:(prefix_name ^ ".ll") in
  let cd_source = Utils.output_to_build_file ~fname:(prefix_name ^ ".cd") in
  let bound = Indexing.bound_symbols bindings in
  let occupancy = Option.value occupancy ~default:(fun ~name:_ ~src_n:_ -> true) in
  Array.unzip
  @@ Array.mapi names ~f:(fun src_n name ->
      let asgns = asgns_l.(src_n) in
      if occupancy ~name ~src_n then
        ( Some name,
          Some
            (Assignments.lower optim_ctx ~unoptim_ll_source ~ll_source ~cd_source ~name bound asgns)
        )
      else (None, None))

let%debug3_sexp verify_prior_context ~(plc : Tn.Placements.t) ~ctx_arrays ~from_prior_context : unit
    =
  Set.iter from_prior_context ~f:(fun tn ->
      if
        Tn.Placements.is_in_context_force plc tn 42
        && (not (Option.is_some @@ Map.find ctx_arrays tn))
        (* Nodes with registered host initialization data (ndarray-backed literals, loaded tensors)
           self-initialize in this context at link time from [Host_inits] (gh-ocannl-333), so they
           need not be present in a prior context. *)
        && not (Host_inits.mem tn)
      then raise @@ Utils.User_error ("The linked context lacks node " ^ Tnode.debug_name tn))

let%debug3_sexp from_prior_context_batch ~(plc : Tn.Placements.t)
    (comps : Assignments.comp option array) : Tn.t_set =
  Array.filter_map comps ~f:(fun comp ->
      Option.map comp ~f:(fun comp ->
          Set.diff (Assignments.context_nodes ~plc comp.Assignments.asgns) comp.embedded_nodes))
  |> Array.fold ~init:(Set.empty (module Tnode)) ~f:Set.union

(** Adds a scheduler and brings a lowered no-device backend on par with lowered device backends. *)
module Add_device
    (Add_scheduler : functor
      (Impl : For_add_scheduler)
      -> With_scheduler with type buffer_ptr = Impl.buffer_ptr)
    (Backend : Lowered_no_device_backend)
(* : Lowered_backend *) =
struct
  include Backend

  include Add_scheduler (struct
    include Backend
  end)

  type code = { lowered : Low_level.optimized; proc : Backend.procedure } [@@deriving sexp_of]

  type code_batch = {
    lowereds : Low_level.optimized option array;
    procs : Backend.procedure option array;
    bindings : Indexing.unit_bindings;
        (** Kept for {!link_batch}: the batch's procedures share one set of static-index refs. *)
  }
  [@@deriving sexp_of]

  let compile ~(name : string) bindings lowered : code =
    let proc = compile ~name bindings lowered in
    { lowered; proc }

  let compile_batch ~names bindings lowereds : code_batch =
    let procs = compile_batch ~names bindings lowereds in
    { lowereds; procs; bindings }

  let link context (code : code) ctx_buffers : Indexing.lowered_bindings * Task.t =
    let runner_label = get_name context.device in
    let merge_buffer = context.device.merge_buffer in
    (* [resolve] is the device's backend-private [buffer_loc -> base] lookup; [link_compiled] does
       the (eager) [ctx_buffers] and (lazy) merge-buffer resolution with it, backend-side. The
       generic shared layer never sees a raw pointer. *)
    let resolve = resolve_pool context.device in
    let bindings, to_schedule =
      link_compiled ~merge_buffer ~resolve ~runner_label ctx_buffers code.proc
    in
    let schedule =
      Task.enschedule ~schedule_task ~get_stream_name:get_name context.device to_schedule
    in
    (bindings, schedule)

  let link_batch context (code_batch : code_batch) ctx_buffers =
    let runner_label = get_name context.device in
    let merge_buffer = context.device.merge_buffer in
    let resolve = resolve_pool context.device in
    (* One shared bindings assoc for the whole batch (mirroring the CUDA/Metal backends): the
       batch's procedures — in particular the segment kernels of one fissioned routine — must see
       the same static-index refs, or setting a binding through the routine would reach only one of
       them. *)
    let lowered_bindings : Indexing.lowered_bindings =
      List.map (Indexing.bound_symbols code_batch.bindings) ~f:(fun s -> (s, ref 0))
    in
    let schedules =
      Array.mapi code_batch.procs ~f:(fun i -> function
        | Some proc ->
            let ctx_buffers = Option.value_exn ~here:[%here] ctx_buffers.(i) in
            let bindings', to_schedule =
              link_compiled ~lowered_bindings ~merge_buffer ~resolve ~runner_label ctx_buffers proc
            in
            assert (phys_equal bindings' lowered_bindings);
            Some
              (Task.enschedule ~schedule_task ~get_stream_name:get_name context.device to_schedule)
        | None -> None)
    in
    (lowered_bindings, schedules)

  (* CPU segment tasks are host closures the stream runner executes in order; the generic event
     chain degenerates to no-ops there, so there is nothing cheaper to provide. *)
  let sequence_segments _context ~name:_ ~bindings:_ ~uses_merge_buffer:_ _tasks = None

  (* Transfers take {!Backend_intf.buffer_loc} and resolve to the backend pointer here, against the
     device's private pool table -- the resolution is backend-side, not in the generic shared
     layer. *)
  let from_host ~dst ~dst_loc hosted =
    let dst_ptr = resolve_pool dst.device dst_loc in
    let work () = host_to_buffer hosted ~dst:dst_ptr in
    schedule_task dst.device
      (Task.Task
         { context_lifetime = dst; description = "from_host on " ^ get_name dst.device; work })

  let to_host ~src ~src_loc hosted =
    let src_ptr = resolve_pool src.device src_loc in
    let work () = buffer_to_host hosted ~src:src_ptr in
    schedule_task src.device
      (Task.Task { context_lifetime = src; description = "to_host on " ^ get_name src.device; work })

  let device_to_device tn ~into_merge_buffer ~dst_loc ~dst ~src_loc ~src =
    let s = dst.device in
    let size_in_bytes = Lazy.force tn.Tnode.size_in_bytes in
    let src_ptr = resolve_pool src.device src_loc in
    let work =
      (* TODO: log the operation if [Utils.settings.with_log_level > 1]. *)
      match (into_merge_buffer, dst_loc) with
      | No, None -> invalid_arg "Add_device.device_to_device: missing dst_loc"
      | No, Some dst_loc ->
          let dst_ptr = resolve_pool dst.device dst_loc in
          fun () -> buffer_to_buffer ~dst:dst_ptr ~src:src_ptr ~size_in_bytes
      | Copy, _ ->
          fun () ->
            (* The merge buffer is the device's reserved single-tenant pool; grow it in place when a
               larger node arrives ([alloc_pool] overwrites the reserved pool-id entry). *)
            if s.merge_buffer_capacity < size_in_bytes then (
              alloc_pool
                ?mode:(Option.map tn.Tnode.memory_mode_intent ~f:fst)
                s ~pool_id:merge_buffer_pool_id ~size_in_bytes
                ~alignment:(Ops.prec_in_bytes (Lazy.force tn.Tnode.storage_prec));
              s.merge_buffer_capacity <- size_in_bytes);
            let loc = { pool_id = merge_buffer_pool_id; offset = 0 } in
            s.merge_buffer := Some loc;
            buffer_to_buffer ~dst:(resolve_pool s loc) ~src:src_ptr ~size_in_bytes
    in
    let description =
      "device_to_device " ^ Tnode.debug_name tn ^ " dst " ^ get_name s ^ " src "
      ^ get_name src.device
    in
    schedule_task s (Task.Task { context_lifetime = (src, dst); description; work })
end

module Raise_backend (Device : Lowered_backend) : Backend = struct
  include Device
  include Add_buffer_retrieval_and_syncing (Device)

  type fissioned = { batch : Device.code_batch; count : int } [@@deriving sexp_of]

  type nonrec code = {
    from_prior_context : Set.M(Tnode).t;
    name : string;
    lowered : Low_level.optimized;
        (** The whole-routine lowered code, used for allocation and I/O analysis. When the routine
            is fissioned this is the pre-fission form; the per-segment kernels live in [proc]. *)
    proc : (code, fissioned) Either.t;
        (** [First]: a single kernel, as before fission. [Second]: the segment kernels of one
            routine, compiled as a batch and launched back-to-back on the routine's stream. *)
    expected_merge_node : Tnode.t option;
    alias_spans : (Tnode.t, int * int) Base.Hashtbl.t option;
        (** gh-ocannl-489: the liveness planner's per-compile facts -- live spans of the
            aliasing-eligible nodes over the final segment sequence (see
            {!Low_level.buffer_access_spans}). Consumed by [allocate_delta] at link time to lay the
            working pool out with overlapping offsets ([plan_arena_offsets]). [None] when the
            [buffer_aliasing] config is off or the code is opaque to the liveness fold. *)
  }
  [@@deriving sexp_of]

  type nonrec code_batch = {
    from_prior_context : Set.M(Tnode).t;
    lowereds : Low_level.optimized option array;
    code_batch : code_batch;
    names : string option array;
    expected_merge_nodes : Tnode.t option array;
  }
  [@@deriving sexp_of]

  let empty_optimize_ctx = Low_level.empty_optimize_ctx
  let get_optimize_ctx (code : code) = code.lowered.Low_level.optimize_ctx

  let get_optimize_ctx_batch (code_batch : code_batch) =
    Array.find_map code_batch.lowereds ~f:(Option.map ~f:(fun l -> l.Low_level.optimize_ctx))
    |> Option.value_or_thunk ~default:Low_level.empty_optimize_ctx

  let%debug3_sexp compile optim_ctx ?name ?lowered_transform ?lowered_transforms bindings
      (comp : Assignments.comp) : code =
    let (name : string), (lowered : Low_level.optimized) =
      lower_assignments optim_ctx ?name bindings comp.asgns
    in
    (* gh-ocannl-489 follow-up: with the liveness planner on, sink whole-node initializations toward
       their first use so live spans start there instead of at an up-front zeroing block (which
       nests the backprop gradient chain's intervals and defeats [plan_arena_offsets]). Reordering
       only -- values are unchanged; gated to keep the planner-off pipeline byte-identical. Before
       scheduling: segment cuts and cross-nest merges see the sunk order. *)
    let lowered =
      if buffer_aliasing () then
        { lowered with Low_level.llc = Low_level.sink_zero_outs lowered.Low_level.llc }
      else lowered
    in
    let limits = Device.hardware_limits () in
    let lowereds =
      Schedule_outcome.tag Schedule_outcome.Transform (fun () ->
          match (lowered_transform, lowered_transforms) with
          | Some _, Some _ ->
              invalid_arg
                "Backend.compile: pass at most one of lowered_transform, lowered_transforms"
          | Some transform, None -> [ transform lowered ]
          | None, Some transforms -> (
              match transforms lowered with
              | [] -> invalid_arg "Backend.compile: lowered_transforms returned an empty list"
              | segments -> segments)
          | None, None ->
              (* No explicit schedule: the default annotator parallelizes kernels it can prove safe
                 (docs/proposals/schedule-ir-optops.md §6) -- Grid x Workgroup on GPU backends,
                 pool-rendered Grid on CPU backends; the identity otherwise. Kernel fission may
                 split the routine into several kernels at cross-workgroup dependency edges; they
                 run back-to-back on the routine's stream (see [link]). *)
              Schedule.maybe_default_schedules ~backend_name:Device.name ~limits
                ~static_indices:(Indexing.bound_symbols bindings) lowered)
    in
    (* Per-compile launch-geometry trace (config [schedule_log_launches]): one line per segment with
       its grid/block dims — for diffing what two compiles of nominally identical code actually emit
       (PR #140 round 6). *)
    (if Lazy.force Schedule.log_launches then
       let n_segs = List.length lowereds in
       List.iteri lowereds ~f:(fun i seg ->
           let d = Low_level.launch_dims seg.Low_level.llc in
           Stdlib.Printf.eprintf
             "schedule: %s seg %d/%d grid=[%d;%d;%d] block=[%d;%d;%d] stmts=%d\n%!" name i n_segs
             d.grid.(0) d.grid.(1) d.grid.(2) d.block.(0) d.block.(1) d.block.(2)
             (List.length (Low_level.flat_lines [ seg.Low_level.llc ]))));
    (* gh-ocannl-489 liveness-based buffer aliasing: the per-compile planning half runs BEFORE
       [Device.compile] below, because the candidates it registers on
       [optimize_ctx.alias_candidates] make codegen drop the [restrict] qualifier on their kernel
       parameters (an actually-aliased [restrict] pair is a miscompile). *)
    let alias_spans : (Tn.t, int * int) Base.Hashtbl.t option =
      plan_alias_spans ~name ~limits ~lowered ~segments:lowereds
    in
    let (proc : (Device.code, fissioned) Either.t), (lowered : Low_level.optimized) =
      match lowereds with
      | [] -> assert false
      | [ single ] ->
          Schedule.check_hardware_limits_classified ~name ~limits single;
          let compiled =
            Schedule_outcome.tag Schedule_outcome.Backend_compile (fun () ->
                compile ~name bindings single)
          in
          (Either.First compiled, single)
      | segments ->
          let seg_names = List.mapi segments ~f:(fun i _ -> name ^ "__seg" ^ Int.to_string i) in
          List.iter2_exn seg_names segments ~f:(fun seg_name seg ->
              Schedule.check_hardware_limits_classified ~name:seg_name ~limits seg);
          let batch =
            Schedule_outcome.tag Schedule_outcome.Backend_compile (fun () ->
                compile_batch
                  ~names:(Array.of_list_map seg_names ~f:Option.some)
                  bindings
                  (Array.of_list_map segments ~f:Option.some))
          in
          (* Keep the whole-routine (pre-fission) lowered code: context allocation and I/O analysis
             need the union footprint, and each segment's [optimized] carries only its filtered
             slice of the traced store. Schedule ops applied per segment can CREATE tnodes the
             pre-fission store has never seen — a hoisted [Stage] registers its packed-constant tile
             in the segment's filtered store (its placement lands in the shared lineage fork, but
             [allocate_delta] enumerates the traced store) — so fold segment-added entries back in.
             Pre-existing keys are shared mutable records (filtered slices alias them), so only
             genuinely new keys need copying. *)
          List.iter segments ~f:(fun seg ->
              Hashtbl.iteri seg.Low_level.traced_store ~f:(fun ~key ~data ->
                  if not (Hashtbl.mem lowered.Low_level.traced_store key) then
                    Hashtbl.add_exn lowered.Low_level.traced_store ~key ~data));
          (Either.Second { batch; count = List.length segments }, lowered)
    in
    (* Placements of all context nodes are settled by codegen (the [compile] just above), so this
       query resolves against the code's own lineage fork. *)
    let from_prior_context : Tn.t_set =
      Set.diff
        (Assignments.context_nodes ~plc:lowered.Low_level.optimize_ctx.placements comp.asgns)
        comp.embedded_nodes
    in
    {
      from_prior_context;
      name;
      lowered;
      proc;
      expected_merge_node = lowered.Low_level.merge_node;
      alias_spans;
    }

  let%debug3_sexp compile_batch optim_ctx ?names ?occupancy bindings
      (comps : Assignments.comp array) : code_batch =
    let names, lowereds =
      lower_batch_assignments optim_ctx ?names ?occupancy bindings
      @@ Array.map comps ~f:(fun c -> c.asgns)
    in
    let lowereds =
      Array.map lowereds
        ~f:
          (Option.map
             ~f:
               (Schedule.maybe_default_schedule ~backend_name:Device.name
                  ~limits:(Device.hardware_limits ())
                  ~static_indices:(Indexing.bound_symbols bindings)))
    in
    Array.iter2_exn names lowereds ~f:(fun name lowered ->
        Option.iter lowered ~f:(fun lowered ->
            Schedule.check_hardware_limits_classified
              ~name:(Option.value name ~default:"<unnamed>")
              ~limits:(Device.hardware_limits ()) lowered));
    let code_batch =
      Schedule_outcome.tag Schedule_outcome.Backend_compile (fun () ->
          compile_batch ~names bindings lowereds)
    in
    let batch_plc =
      (Array.find_map lowereds ~f:(Option.map ~f:(fun l -> l.Low_level.optimize_ctx))
      |> Option.value_or_thunk ~default:Low_level.empty_optimize_ctx)
        .placements
    in
    let from_prior_context =
      from_prior_context_batch ~plc:batch_plc
      @@ Array.mapi lowereds ~f:(fun i -> Option.map ~f:(fun _ -> comps.(i)))
    in
    {
      from_prior_context;
      names;
      lowereds;
      code_batch;
      expected_merge_nodes =
        Array.map lowereds ~f:(fun lowered ->
            Option.(join @@ map lowered ~f:(fun optim -> optim.Low_level.merge_node)));
    }

  (* gh-ocannl-344 Phase B/C: allocate a context's delta -- the in-context tnodes not already
     present in [context.ctx_buffers]. Working (non-constant) and constant/read-only nodes are EACH
     packed into pools sized to their group and bump-assigned increasing byte offsets, replacing the
     one-pool-per-tnode policy. Working pools belong to the context (freed at its [finalize]);
     constant pools are deduped per-device via [constant_buffer_cache] and outlive the context
     (freed at device teardown). Enumeration follows [traced_store] order so pool ids and offsets
     stay deterministic across runs. The per-pool 4 GB cap (uint32 offsets unless large_models) is
     enforced by {!Backend_utils.plan_pool_segments}. *)
  let%track3_sexp allocate_delta (context : context) ~name
      ~(alias_spans : (Tn.t, int * int) Base.Hashtbl.t option) (lowered : Low_level.optimized) :
      ctx_buffers =
    let traced_store = lowered.Low_level.traced_store in
    let device = context.device in
    let cap = if Utils.settings.large_models then Int.max_value else 0x1_0000_0000 in
    (* Pass 1: partition the delta, preserving [traced_store] iteration order. Slice-alias views own
       no buffer and are excluded automatically: [is_in_context_force] returns false for them
       (gh-ocannl-293 293a). Their parent is materialized and is allocated here (or already present
       from a prior context) like any other node, since the alias's redirected reads/writes
       reference the parent in the lowered code. *)
    let working = ref [] and constants = ref [] in
    Hashtbl.iteri traced_store ~f:(fun ~key ~data:node ->
        if
          Tnode.Placements.is_in_context_force lowered.Low_level.optimize_ctx.placements key 43
          && not (Map.mem context.ctx_buffers key)
        then
          if
            node.Low_level.read_only
            || Tn.Placements.known_constant lowered.Low_level.optimize_ctx.placements key
          then constants := (key, node) :: !constants
          else working := (key, node) :: !working);
    let working = List.rev !working and constants = List.rev !constants in
    let ctx_buffers = ref context.ctx_buffers in
    (* Pack a group of (key, node) into one or more pools, segmenting at the cap. [register] decides
       how the resulting [buffer_loc] is recorded (directly into [ctx_buffers] for working nodes, or
       deduped through [constant_buffer_cache] for constants). [base_pool_id] of each segment is a
       freshly minted [next_pool_id]; offsets and pool sizes come from the pure planner. *)
    let place (key, node) ~pool_id ~offset ~register =
      let size_in_bytes = size_in_bytes_of key in
      let alloc () : buffer_loc =
        let host_init = Host_inits.find key in
        (* Zero-initialize unless the node will be copied from host immediately, or the lowered code
           already zero-initializes it. *)
        let zero_init = not (Option.is_some host_init || node.Low_level.zero_initialized_by_code) in
        if zero_init then memset_zero device ~pool_id ~offset ~size_in_bytes;
        let loc = { pool_id; offset } in
        Option.iter host_init ~f:(fun nd ->
            let nd = Lazy.force nd in
            (* Interval analysis, Phase B: [Host_inits] uploads are host writes; propose the scanned
               bounds lazily -- here, where the buffer is forced at link/upload time -- so [Reshape]
               inits wait for shape and padding inference as designed. *)
            Tnode.propose_bounds_from_host key nd;
            Device.from_host ~dst:context ~dst_loc:loc nd);
        loc
      in
      register key ~alloc
    in
    (* gh-ocannl-550: the shared seam is where a pool's byte size is known, so it is where the
       per-class census records it. [~constant] separates the two groups below, whose lifetimes
       differ: a working pool belongs to the context and dies with it, a constant pool is deduped
       per device and outlives it. *)
    (* What THIS call minted, so a failure part way through can give it back (gh-ocannl-550,
       round-four review). [with_delta] in [link] covers failures after the delta is returned; it
       cannot cover a failure inside this function, where the caller never receives [ctx_buffers] and
       the slabs already allocated are rooted with nothing to reach them. Pool ids are fresh from
       [device.next_pool_id], so every id here is unambiguously this call's. *)
    let minted = ref [] in
    let cache_inserts = ref [] in
    let alloc_pool_counted ~constant device ~pool_id ~size_in_bytes ~alignment =
      alloc_pool device ~pool_id ~size_in_bytes ~alignment;
      minted := pool_id :: !minted;
      Alloc_census.record_pool ~device_id:device.device_id ~pool_id ~constant ~size_in_bytes
    in
    let unwind_partial_delta () =
      (* The uploads scheduled above are asynchronous, so the pools must not be freed under them. *)
      (try Device.await device with _ -> ());
      (* Constant-cache entries this call inserted point INTO the pools about to go, so they have to
         come out first or a later context would resolve a freed slab. Entries that were already
         there belong to earlier compiles and are untouched. *)
      List.iter !cache_inserts ~f:(Hashtbl.remove device.constant_buffer_cache);
      Option.iter free_pool ~f:(fun free_pool ->
          List.dedup_and_sort !minted ~compare:Int.compare
          |> List.iter ~f:(fun pool_id ->
                 free_pool device ~pool_id;
                 Alloc_census.forget_pool ~device_id:device.device_id ~pool_id))
    in
    let pack ?arena ~constant (group : (Tn.t * Low_level.traced_array) list)
        ~(register : Tn.t -> alloc:(unit -> buffer_loc) -> unit) : unit =
      if not (List.is_empty group) then begin
        (* gh-ocannl-498: lay every group out in CANONICAL (uid) order, which is the order
           [score_footprint] scores. Both planners are order-sensitive -- the arena's greedy
           coloring breaks equal-size ties by input order, and bump packing's alignment padding and
           cap segmentation depend on the running offset (sizes 4 then 64 at alignment 32 occupy 96
           bytes, reversed 68). Scoring one order and allocating another would let a plan report
           itself under budget while linking asks for a larger pool. [traced_store] order was merely
           a deterministic order; uid order is deterministic too, and shared with the scorer. Only
           the layout input is reordered: pool ids are minted per segment before any placement, and
           registration is order-independent. *)
        let group = List.sort group ~compare:(fun (a, _) (b, _) -> Tn.compare a b) in
        let items =
          (* Within-pool offsets are padded to [Ops.buffer_alignment] (not just the element size) so
             that every node's buffer — not only each pool's base — is SIMD-aligned (gh-ocannl-164);
             ≤31 bytes of padding per node. *)
          List.map group ~f:(fun (key, _) ->
              ( size_in_bytes_of key,
                max (Ops.prec_in_bytes (Lazy.force key.Tn.storage_prec)) Ops.buffer_alignment ))
        in
        (* gh-ocannl-489: with a liveness plan (the working group under [buffer_aliasing]), lay the
           group out as one arena where liveness-disjoint same-precision nodes overlap. Falls back
           to bump packing when the arena would exceed the per-pool cap. *)
        let arena_layout =
          Option.bind arena ~f:(fun spans ->
              plan_arena_offsets ~cap
                (List.map2_exn group items ~f:(fun (key, _) (size, align) ->
                     ( size,
                       align,
                       Ops.prec_string (Lazy.force key.Tn.storage_prec),
                       Hashtbl.find spans key ))))
        in
        match arena_layout with
        | Some (offsets, total) ->
            let pool_id = device.next_pool_id in
            device.next_pool_id <- pool_id + 1;
            let alignment =
              List.fold group ~init:1 ~f:(fun a (key, _) ->
                  max a (Ops.prec_in_bytes (Lazy.force key.Tn.storage_prec)))
            in
            alloc_pool_counted ~constant device ~pool_id ~size_in_bytes:total ~alignment;
            List.iter2_exn group offsets ~f:(fun entry offset ->
                place entry ~pool_id ~offset ~register);
            if log_buffer_aliasing () then
              let _, bump_sizes =
                plan_pool_segments ~cap ~what:"Backends.allocate_delta"
                  ~debug_name:(fun i -> Tn.debug_name (fst (List.nth_exn group i)))
                  items
              in
              let dedicated = List.fold bump_sizes ~init:0 ~f:( + ) in
              let planned =
                Option.value_map arena ~default:0 ~f:(fun spans ->
                    List.count group ~f:(fun (key, _) -> Hashtbl.mem spans key))
              in
              Stdlib.Printf.eprintf
                "buffer aliasing: %s: working pool %d bytes, dedicated packing %d bytes (%.1f%% \
                 saved; %d/%d nodes liveness-planned)\n\
                 %!"
                name total dedicated
                (100. *. Float.of_int (dedicated - total) /. Float.of_int (max 1 dedicated))
                planned (List.length group)
        | None ->
            let assignments, segment_sizes =
              plan_pool_segments ~cap ~what:"Backends.allocate_delta"
                ~debug_name:(fun i -> Tn.debug_name (fst (List.nth_exn group i)))
                items
            in
            (* Mint a pool id per segment up front, sized from the planner. *)
            let seg_pool_ids =
              List.map segment_sizes ~f:(fun size_in_bytes ->
                  let pool_id = device.next_pool_id in
                  device.next_pool_id <- pool_id + 1;
                  (pool_id, size_in_bytes))
              |> Array.of_list
            in
            (* Allocate each segment's slab, padding alignment to the max element precision it
               holds. *)
            let seg_align = Array.map seg_pool_ids ~f:(fun _ -> ref 1) in
            List.iter2_exn group assignments ~f:(fun (key, _) (seg, _) ->
                let a = Ops.prec_in_bytes (Lazy.force key.Tn.storage_prec) in
                if a > !(seg_align.(seg)) then seg_align.(seg) := a);
            Array.iteri seg_pool_ids ~f:(fun seg (pool_id, size_in_bytes) ->
                alloc_pool_counted ~constant device ~pool_id ~size_in_bytes
                  ~alignment:!(seg_align.(seg)));
            (* Place each node at its planned (segment, offset). *)
            List.iter2_exn group assignments ~f:(fun entry (seg, offset) ->
                let pool_id, _ = seg_pool_ids.(seg) in
                place entry ~pool_id ~offset ~register)
      end
    in
    (* Pass 2a: working delta -> context-owned pool(s), recorded directly. Only this group is
       liveness-planned (gh-ocannl-489): constants are deduped per-device and outlive the context,
       so their lifetimes are not routine intervals. *)
    let passes () =
      pack ?arena:alias_spans ~constant:false working ~register:(fun key ~alloc ->
          ctx_buffers := Map.add_exn !ctx_buffers ~key ~data:(alloc ()));
    (* Pass 2b: constants / read-only -> per-device constant pool(s). Constants already allocated on
       this device (a hit in [constant_buffer_cache], possibly from another context tree) resolve
       directly and are excluded from the new slab, so the freshly-minted constant pool holds
       exactly this device's genuinely-new constants -- no wasted holes. The remaining new constants
       pack into one constant pool (or more, past the cap), deduped into the cache. Constant pools
       outlive the context and are skipped by context [finalize] (freed at device teardown). *)
    let new_constants = ref [] in
    List.iter constants ~f:(fun (key, node) ->
        match Hashtbl.find device.constant_buffer_cache key with
        | Some data -> ctx_buffers := Map.add_exn !ctx_buffers ~key ~data
        | None -> new_constants := (key, node) :: !new_constants);
      pack ~constant:true (List.rev !new_constants) ~register:(fun key ~alloc ->
          let data =
            Hashtbl.find_or_add device.constant_buffer_cache key ~default:(fun () ->
                cache_inserts := key :: !cache_inserts;
                alloc ())
          in
          ctx_buffers := Map.add_exn !ctx_buffers ~key ~data)
    in
    (match passes () with
    | () -> ()
    | exception exn ->
        let backtrace = Stdlib.Printexc.get_raw_backtrace () in
        (* Best-effort: giving the slabs back must not replace the allocation failure the caller has
           to classify (an out-of-memory decline being the whole point). *)
        (try unwind_partial_delta () with _ -> ());
        Stdlib.Printexc.raise_with_backtrace exn backtrace);
    !ctx_buffers

  (* gh-ocannl-550: [allocate_delta] runs BEFORE the backend link, and the pool table roots whatever
     it allocated — so a link that raises (HIP's scratch-budget validator, a driver refusal, an
     aliased-read rejection) used to leave that routine's pools behind with no context through which
     anyone could ever release them. Those are the [Backend_link] declines an autotune search absorbs,
     so they accumulated exactly like the candidates that succeeded.

     Frees the delta of [ctx_buffers] against [context]: keyed by [pool_id] (one pool holds several
     nodes) and skipping per-device constants, i.e. the same rule the context [finalize] applies —
     this stands in for it on the path where no context was ever built. *)
  let free_delta context (ctx_buffers : ctx_buffers) =
    (* Sync first, for the same reason [unwind_partial_delta] and the context [finalize] do
       (gh-ocannl-550, round-five review): [allocate_delta] queues [Host_inits] uploads through
       [Device.from_host], so a delta being discarded after a failed link can still have writes in
       flight — and freeing the slab under them is device corruption, on a path that is otherwise a
       contained candidate decline the search carries on from. Best-effort: the device may already be
       refusing work, and that must not replace the link failure the caller has to classify. *)
    (try Device.await context.device with _ -> ());
    Option.iter free_pool ~f:(fun free_pool ->
        Map.fold ctx_buffers ~init:(Set.empty (module Int))
          ~f:(fun ~key ~data:(loc : buffer_loc) freed ->
            if
              (not (Map.mem context.ctx_buffers key))
              && (not (Hashtbl.mem context.device.constant_buffer_cache key))
              && not (Set.mem freed loc.pool_id)
            then (
              free_pool context.device ~pool_id:loc.pool_id;
              Alloc_census.forget_pool ~device_id:context.device.device_id ~pool_id:loc.pool_id;
              Set.add freed loc.pool_id)
            else freed)
        |> (ignore : Set.M(Int).t -> unit))

  (* Runs [f] on a freshly allocated delta, freeing that delta if [f] raises. Everything after the
     allocation belongs inside: a failure past [make_child] discards the child too, so its pools are
     just as unreachable as if the child had never existed. *)
  let with_delta context ctx_buffers ~f =
    match f () with
    | result -> result
    | exception exn ->
        let backtrace = Stdlib.Printexc.get_raw_backtrace () in
        (* Best-effort, and it must not replace the link failure the caller has to classify. *)
        (try free_delta context ctx_buffers with _ -> ());
        Stdlib.Printexc.raise_with_backtrace exn backtrace

  let%debug3_sexp link context (code : code) =
    verify_prior_context ~plc:code.lowered.Low_level.optimize_ctx.placements
      ~ctx_arrays:context.ctx_buffers ~from_prior_context:code.from_prior_context;
    (* Static merge-buffer verification "in the right direction" (gh-ocannl-288): the linked context
       carries the merge-buffer node of the producing [device_to_device] transfer routine; a
       mismatch with the consuming code raises here, at link time, before any schedule runs. *)
    check_merge_buffer_static ~merge_buffer_node:context.merge_buffer_node
      ~code_node:code.expected_merge_node;
    let (inputs, outputs), merge_buffer_input = Low_level.input_and_output_nodes code.lowered in
    let ctx_buffers =
      allocate_delta context ~name:code.name ~alias_spans:code.alias_spans code.lowered
    in
    with_delta context ctx_buffers ~f:(fun () ->
    (* gh-ocannl-489: a routine reading a node whose buffer an earlier routine of this lineage
       aliased would read clobbered values -- fail at link time, before any schedule runs. Writes
       (outputs) are allowed: the aliasing routine rewrites everything it reads on each run. This
       code's own aliased nodes are never its inputs (aliasing-eligible nodes are not
       read-before-write), so the check only fires on genuinely cross-routine reads. *)
    Set.iter inputs ~f:(fun tn ->
        Option.iter (Map.find ctx_buffers tn) ~f:(fun loc ->
            if buffer_overlaps ctx_buffers tn loc then
              aliased_read_error ~what:("linking " ^ code.name ^ ", input") tn));
    let optimize_ctx = code.lowered.Low_level.optimize_ctx in
    let bindings, schedule =
      match code.proc with
      | Either.First single -> link context single ctx_buffers
      | Either.Second { batch; count } ->
          (* Fissioned routine: every segment kernel links against the routine's one ctx_buffers
             delta and shares the one bindings assoc; the combined task launches the segments in
             order on the routine's stream, whose FIFO ordering supplies the grid-wide
             synchronization at each segment boundary (the same contract consecutive routines on one
             stream already rely on). *)
          let bindings, tasks =
            link_batch context batch (Array.create ~len:count (Some ctx_buffers))
          in
          let tasks = Array.to_list (Array.filter_opt tasks) in
          assert (List.length tasks = count);
          (* Device-side ordering at each segment boundary: the cut is where the kernel-internal
             code lacks grid-wide synchronization, so the stream must provide it. Queue FIFO alone
             is not enough on Metal — command buffers over untracked resources may overlap in
             execution (caught by test_random_histograms). Backends that can order the batch
             device-side more cheaply (one Metal command buffer with a serial compute pass) provide
             [sequence_segments]; the fallback chains an event per boundary: schedule each next
             segment to wait for all work enqueued so far. No host blocking. *)
          let schedule =
            match
              sequence_segments context ~name:code.name ~bindings
                ~uses_merge_buffer:(Option.is_some code.expected_merge_node)
                tasks
            with
            | Some fused -> fused
            | None ->
                Task.Task
                  {
                    context_lifetime = tasks;
                    description = "fissioned segments of " ^ code.name;
                    work =
                      (fun () ->
                        List.iteri tasks ~f:(fun i t ->
                            if i > 0 then will_wait_for context (all_work context.device);
                            Task.run t));
                  }
          in
          (bindings, schedule)
    in
    let context = make_child ~ctx_buffers ~optimize_ctx context in
    let schedule =
      Task.prepend schedule ~work:(fun () ->
          check_merge_buffer context.device ~code_node:code.expected_merge_node)
    in
    sync_routine
      { context; schedule; bindings; name = code.name; inputs; merge_buffer_input; outputs })

  let%debug3_sexp link_batch context code_batch =
    verify_prior_context ~plc:(get_optimize_ctx_batch code_batch).Low_level.placements
      ~ctx_arrays:context.ctx_buffers ~from_prior_context:code_batch.from_prior_context;
    (* gh-ocannl-550: the same unwind [link] gets, extended over the whole batch. Every member's delta
       is allocated before the backend linker runs, so a later member's allocation or the link itself
       raising used to abandon every completed member's pools -- rooted, with no context to reach them,
       since the member contexts are only derived in the fold below. [free_delta] is applied per member
       and skips per-device constants exactly as the context [finalize] does, so a partial batch gives
       back its working pools and leaves the shared constants alone. *)
    let allocated = ref [] in
    let unwind_batch () =
      List.iter !allocated ~f:(fun cb -> try free_delta context cb with _ -> ())
    in
    let ctx_buffers, bindings, schedules =
      match
        let ctx_buffers =
          Array.mapi code_batch.lowereds ~f:(fun i ->
              Option.map ~f:(fun l ->
                  let name = Option.value code_batch.names.(i) ~default:"<unnamed>" in
                  (* Batch compiles are not liveness-planned in v1 (they do not go through the
                     fission/schedule seam of [compile]); [alias_spans:None] keeps bump packing. *)
                  let cb = allocate_delta context ~name ~alias_spans:None l in
                  allocated := cb :: !allocated;
                  cb))
        in
        let bindings, schedules = link_batch context code_batch.code_batch ctx_buffers in
        (ctx_buffers, bindings, schedules)
      with
      | result -> result
      | exception exn ->
          let backtrace = Stdlib.Printexc.get_raw_backtrace () in
          unwind_batch ();
          Stdlib.Printexc.raise_with_backtrace exn backtrace
    in
    Array.fold_mapi schedules ~init:context ~f:(fun i context -> function
      | None -> (context, None)
      | Some schedule ->
          let ctx_buffers = Option.value_exn ctx_buffers.(i) in
          let optimize_ctx = (Option.value_exn code_batch.lowereds.(i)).Low_level.optimize_ctx in
          let expected_merge_node = code_batch.expected_merge_nodes.(i) in
          (* Static merge-buffer verification at link time (gh-ocannl-288): check the node provided
             by the fold-current context before deriving the consumer's child context. *)
          check_merge_buffer_static ~merge_buffer_node:context.merge_buffer_node
            ~code_node:expected_merge_node;
          let context = make_child ~ctx_buffers ~optimize_ctx context in
          let (inputs, outputs), merge_buffer_input =
            Low_level.input_and_output_nodes @@ Option.value_exn code_batch.lowereds.(i)
          in
          (* gh-ocannl-489: same cross-routine read guard as in [link]. *)
          Set.iter inputs ~f:(fun tn ->
              Option.iter (Map.find ctx_buffers tn) ~f:(fun loc ->
                  if buffer_overlaps ctx_buffers tn loc then
                    aliased_read_error ~what:"linking batch member, input" tn));
          let schedule =
            Task.prepend schedule ~work:(fun () ->
                check_merge_buffer context.device ~code_node:expected_merge_node)
          in
          let r =
            sync_routine { context; schedule; bindings; name; inputs; merge_buffer_input; outputs }
          in
          (context, Some r))
end

module Make_device_backend_from_lowered
    (Add_scheduler : functor
      (Impl : For_add_scheduler)
      -> With_scheduler with type buffer_ptr = Impl.buffer_ptr)
    (Backend_impl : Lowered_no_device_backend) =
struct
  module Lowered_device = Add_device (Add_scheduler) (Backend_impl)
  module Backend_device = Raise_backend (Lowered_device)
  include Backend_device
end

let finalize (type dev runner event)
    (module Backend : Backend with type dev = dev and type runner = runner and type event = event)
    (ctx : Backend.context) : unit =
  (* The flag means "this context's pools have been freed", NOT "a free was attempted" — so cleanup
     that RAISES resets it (gh-ocannl-550). [Backend.await] is the realistic raiser: a device still
     reporting an asynchronous error, or a dead worker domain. Left set, every later release of this
     context would be a silent no-op and its pools would stay rooted for the process — restoring
     exactly the unbounded growth this exists to end, and on the failure paths where it matters most,
     since the tuner catches a failed release and carries on with the next candidate or arm. A retry
     is safe: freeing is idempotent per [pool_id] on every backend (the table entry is gone after the
     first success, and [Alloc_census.forget_pool] ignores an absent key), so a cleanup that got part
     way through does not double-free on the next attempt. *)
  let cleanup () =
    Option.iter Backend.free_pool ~f:(fun free_pool ->
        Backend.await ctx.device;
        (* One pool holds several nodes (gh-ocannl-344 bump packing / gh-ocannl-489 arenas), so the
           same [pool_id] is reached through several keys; dedup before freeing, or the second visit
           frees an already-freed slab. [Alloc_census.forget_pool] is idempotent for the same
           reason, but the backend's [free_pool] is the one that must not run twice. *)
        Map.fold ctx.ctx_buffers ~init:(Set.empty (module Int))
          ~f:(fun ~key ~data:(loc : Ir.Backend_intf.buffer_loc) freed ->
            if
              (not (Option.exists ctx.parent ~f:(fun pc -> Map.mem pc.ctx_buffers key)))
              && (not (Hashtbl.mem ctx.device.constant_buffer_cache key))
              && not (Set.mem freed loc.pool_id)
            then (
              free_pool ctx.device ~pool_id:loc.pool_id;
              Alloc_census.forget_pool ~device_id:ctx.device.device_id ~pool_id:loc.pool_id;
              Set.add freed loc.pool_id)
            else freed)
        |> (ignore : Set.M(Int).t -> unit))
  in
  if Atomic.compare_and_set ctx.finalized false true then
    match cleanup () with
    | () -> Alloc_census.count_context_released ()
    | exception exn ->
        Atomic.set ctx.finalized false;
        raise exn

(* {2 The implemented backends, as singletons}

   One instantiation per backend for the whole process, so backend context types are nameable
   ([Cc_b.context], ...) and two independently-created contexts on the same backend unify -- the
   precondition for [Context.copy] dispatching to backend-specific [device_to_device] via
   {!wrapped_context}. The retired [fresh_backend] applied these functors per call to isolate
   tnode-keyed backend caches between tests (reinitialization reuses tnode ids); tnode identity is
   now the never-reused [Tnode.uid], so stale cache entries cannot alias fresh nodes and the
   isolation is unnecessary. Instantiating a module here must not touch any driver or hardware:
   device backends keep discovery/driver-init lazy, forced at first device use (cudajit is a depopt
   -- the library being installed does not imply a usable driver -- and a CPU-only run must not
   depend on GPU runtimes). On platforms without the corresponding library the dune [select]ed
   [Lowered_backend_missing] stub is instantiated instead, likewise harmless at init and raising on
   use. Either failure mode surfaces at [get_device], where [Context.auto]'s fallback catches it per
   call, as with the retired per-call instantiation. *)

module Cc_b : Backend = Make_device_backend_from_lowered (Schedulers.Sync) (Cc_backend)
module Multidev_cc_b : Backend = Make_device_backend_from_lowered (Schedulers.Multidev) (Cc_backend)
module Cuda_b : Backend = Raise_backend (Cuda_backend_impl.Impl : Lowered_backend)
module Hip_b : Backend = Raise_backend (Hip_backend_impl.Impl : Lowered_backend)
module Metal_b : Backend = Raise_backend (Metal_backend_impl.Impl : Lowered_backend)

type backend = Cc | Multidev_cc | Cuda | Hip | Metal [@@deriving sexp, equal]

let get_backend ?backend_name () =
  match
    Option.value_or_thunk backend_name ~default:(fun () ->
        Utils.get_global_arg ~arg_name:"backend" ~default:"cc")
    |> String.lowercase
  with
  (* "sync_cc" and "multicore_cc" are accepted as deprecated aliases of the renamed backends. *)
  | "cc" | "sync_cc" -> Cc
  | "multidev_cc" | "multicore_cc" -> Multidev_cc
  | "cuda" -> Cuda
  | "hip" -> Hip
  | "metal" -> Metal
  | backend -> invalid_arg [%string "Backends.get_backend: unknown backend %{backend}"]

let backend_name = function
  | Cc -> "cc"
  | Multidev_cc -> "multidev_cc"
  | Cuda -> "cuda"
  | Hip -> "hip"
  | Metal -> "metal"

let backend_module : backend -> (module Backend) = function
  | Cc -> (module Cc_b)
  | Multidev_cc -> (module Multidev_cc_b)
  | Cuda -> (module Cuda_b)
  | Hip -> (module Hip_b)
  | Metal -> (module Metal_b)

type wrapped_context =
  | Cc_ctx of Cc_b.context
  | Multidev_cc_ctx of Multidev_cc_b.context
  | Cuda_ctx of Cuda_b.context
  | Hip_ctx of Hip_b.context
  | Metal_ctx of Metal_b.context

let wrapped_backend = function
  | Cc_ctx _ -> Cc
  | Multidev_cc_ctx _ -> Multidev_cc
  | Cuda_ctx _ -> Cuda
  | Hip_ctx _ -> Hip
  | Metal_ctx _ -> Metal

let make_context ?(device_id = 0) backend =
  let fresh (type dev runner event)
      (module B : Backend with type dev = dev and type runner = runner and type event = event) =
    let device = B.get_device ~ordinal:device_id in
    B.make_context ~optimize_ctx:(B.empty_optimize_ctx ()) device
  in
  match backend with
  | Cc -> Cc_ctx (fresh (module Cc_b))
  | Multidev_cc -> Multidev_cc_ctx (fresh (module Multidev_cc_b))
  | Cuda -> Cuda_ctx (fresh (module Cuda_b))
  | Hip -> Hip_ctx (fresh (module Hip_b))
  | Metal -> Metal_ctx (fresh (module Metal_b))

type 'a ctx_op = {
  f :
    'dev 'runner 'event.
    (module Backend with type dev = 'dev and type runner = 'runner and type event = 'event) ->
    ('dev, 'runner, 'event) Backend_intf.context ->
    ('dev, 'runner, 'event) Backend_intf.context * 'a;
}
(** A context-transforming backend operation, polymorphic over the backend's type components so
    {!with_backend} can rebuild the same {!wrapped_context} constructor around the result. *)

let with_backend (w : wrapped_context) { f } =
  match w with
  | Cc_ctx c ->
      let c, r = f (module Cc_b) c in
      (Cc_ctx c, r)
  | Multidev_cc_ctx c ->
      let c, r = f (module Multidev_cc_b) c in
      (Multidev_cc_ctx c, r)
  | Cuda_ctx c ->
      let c, r = f (module Cuda_b) c in
      (Cuda_ctx c, r)
  | Hip_ctx c ->
      let c, r = f (module Hip_b) c in
      (Hip_ctx c, r)
  | Metal_ctx c ->
      let c, r = f (module Metal_b) c in
      (Metal_ctx c, r)

type 'a ctx_query = {
  q :
    'dev 'runner 'event.
    (module Backend with type dev = 'dev and type runner = 'runner and type event = 'event) ->
    ('dev, 'runner, 'event) Backend_intf.context ->
    'a;
}
(** A read-only backend operation; like {!ctx_op} but leaves the context untouched. *)

let query (w : wrapped_context) { q } =
  match w with
  | Cc_ctx c -> q (module Cc_b) c
  | Multidev_cc_ctx c -> q (module Multidev_cc_b) c
  | Cuda_ctx c -> q (module Cuda_b) c
  | Hip_ctx c -> q (module Hip_b) c
  | Metal_ctx c -> q (module Metal_b) c
