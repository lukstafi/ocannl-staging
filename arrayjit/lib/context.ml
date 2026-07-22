open Base
module Asgns = Ir.Assignments
module Tn = Ir.Tnode
module Nd = Ir.Ndarray
module Idx = Ir.Indexing
module BI = Ir.Backend_intf
module Backends_deprecated = Backends

(* The backend context rides in [Backends.wrapped_context] -- a closed disjunction over the backend
   singletons' context types (no existential): [Backends.query]/[Backends.with_backend] dispatch
   generic operations, and [copy] pair-matches the constructors to recover type equality for
   same-backend transfers. *)

type compile_frontier = {
  last_writer : int Map.M(Tn).t;
      (** For each tnode, the routine_id of the most recent routine that writes it. *)
  last_readers : Set.M(Int).t Map.M(Tn).t;
      (** For each tnode, the set of routine_ids that read it since the last write. *)
}
(** Immutable compile-time frontier for execution dependency tracking. Each context carries its own
    frontier; only the context returned by [compile] receives the updated frontier. The original
    context is unchanged. This ensures that sibling compiles (from the same context) produce
    independent routines. *)

type execution_ledger = {
  mutable next_id : int;
  routine_names : string Hashtbl.M(Int).t;
  mutable executed : Set.M(Int).t;
}
(** Shared mutable state for execution tracking, allocated once per root context. Shared by
    reference across all contexts derived from the same root. *)

let empty_frontier = { last_writer = Map.empty (module Tn); last_readers = Map.empty (module Tn) }

let create_ledger () =
  { next_id = 0; routine_names = Hashtbl.create (module Int); executed = Set.empty (module Int) }

type t = {
  wrapped : (Backends.wrapped_context[@sexp.opaque]);
  device_id : int;
  initialized_nodes : Set.M(Tn).t;
  frontier : (compile_frontier[@sexp.opaque]);
  ledger : (execution_ledger[@sexp.opaque]);
}
[@@deriving sexp_of]

let backend_name ctx = Backends.backend_name (Backends.wrapped_backend ctx.wrapped)

type routine = {
  context : t;
  task : Ir.Task.t;
  bindings : Idx.lowered_bindings;
  name : string;
  inputs : Set.M(Tn).t;
  outputs : Set.M(Tn).t;
  routine_id : int;
  execution_deps : Set.M(Int).t;
}

let bindings r = r.bindings
let context r = r.context
let routine_id r = r.routine_id
let routine_name r = r.name
let execution_deps r = Set.to_list r.execution_deps
let can_run ctx routine = Set.is_subset routine.execution_deps ~of_:ctx.ledger.executed

(** Create a context from a backend name *)
let create_from_backend_name ~device_id backend_name =
  let backend = Backends.get_backend ~backend_name () in
  {
    wrapped = Backends.make_context ~device_id backend;
    device_id;
    initialized_nodes = Set.empty (module Tn);
    frontier = empty_frontier;
    ledger = create_ledger ();
  }

let cuda ?device_id () =
  create_from_backend_name ~device_id:(Option.value device_id ~default:0) "cuda"

let hip ?device_id () =
  create_from_backend_name ~device_id:(Option.value device_id ~default:0) "hip"

let metal ?device_id () =
  create_from_backend_name ~device_id:(Option.value device_id ~default:0) "metal"

let cpu ?threads () =
  (* Kernel-level CPU parallelism is automatic on both cc backends (pool-rendered Grid loops, see
     [automatic_cpu_schedule]); [threads] > 1 selects the multidev_cc debugging backend, which
     exposes multiple worker-domain devices. *)
  let backend_name = match threads with None | Some 1 -> "cc" | Some _ -> "multidev_cc" in
  create_from_backend_name ~device_id:0 backend_name

let auto () =
  (* First check if a backend is configured globally *)
  match Utils.get_global_arg ~arg_name:"backend" ~default:"" with
  | "" ->
      (* No global config, try backends in order of preference *)
      let backends_to_try = [ "metal"; "cuda"; "hip"; "cc" ] in
      let rec try_backends = function
        | [] -> failwith "No backend available"
        | name :: rest -> (
            try create_from_backend_name ~device_id:0 name with _ -> try_backends rest)
      in
      try_backends backends_to_try
  | backend_name -> (
      (* Use the configured backend *)
      try create_from_backend_name ~device_id:0 backend_name
      with _ -> invalid_arg ("Unknown backend: " ^ backend_name))

let compile ?lowered_transform ?lowered_transforms ctx comp bindings =
  (* Compile and link on the wrapped backend context; only backend-independent routine components
     (and, via [with_backend]'s rebuilt constructor, the updated context) escape the dispatch. *)
  let wrapped, (task, lowered_bindings, name, backend_inputs, backend_outputs) =
    Backends.with_backend ctx.wrapped
      {
        f =
          (fun (type dev runner event)
            (module Backend : BI.Backend
              with type dev = dev
               and type runner = runner
               and type event = event)
            bctx
          ->
            let code =
              Backend.compile ?lowered_transform ?lowered_transforms bctx.BI.optimize_ctx bindings
                comp
            in
            let r = Backend.link bctx code in
            (r.BI.context, (r.BI.schedule, r.BI.bindings, r.BI.name, r.BI.inputs, r.BI.outputs)));
      }
  in

  (* Allocate unique ID from shared ledger *)
  let id = ctx.ledger.next_id in
  ctx.ledger.next_id <- id + 1;

  (* Use the backend routine's precise access sets for dependency tracking. [backend_inputs] =
     materialized read-only and read-before-write nodes. [backend_outputs] = all materialized
     written-to nodes. *)
  let frontier = ctx.frontier in
  let empty_int_set = Set.empty (module Int) in

  (* RAW: for each backend input, depend on its last writer *)
  let deps =
    Set.fold backend_inputs ~init:empty_int_set ~f:(fun deps tn ->
        match Map.find frontier.last_writer tn with
        | Some writer_id -> Set.add deps writer_id
        | None -> deps)
  in

  (* WAW + WAR: for each backend output, depend on last writer and all last readers *)
  let deps =
    Set.fold backend_outputs ~init:deps ~f:(fun deps tn ->
        let deps =
          match Map.find frontier.last_writer tn with
          | Some writer_id -> Set.add deps writer_id
          | None -> deps
        in
        match Map.find frontier.last_readers tn with
        | Some readers -> Set.union deps readers
        | None -> deps)
  in

  (* Build updated frontier (immutable — only in returned context) *)
  let new_last_writer =
    Set.fold backend_outputs ~init:frontier.last_writer ~f:(fun lw tn ->
        Map.set lw ~key:tn ~data:id)
  in
  let new_last_readers =
    Set.fold backend_outputs ~init:frontier.last_readers ~f:(fun lr tn -> Map.remove lr tn)
  in
  let pure_inputs = Set.diff backend_inputs backend_outputs in
  let new_last_readers =
    Set.fold pure_inputs ~init:new_last_readers ~f:(fun lr tn ->
        let existing = Option.value (Map.find lr tn) ~default:empty_int_set in
        Map.set lr ~key:tn ~data:(Set.add existing id))
  in
  let new_frontier = { last_writer = new_last_writer; last_readers = new_last_readers } in

  (* Register in shared ledger *)
  Hashtbl.set ctx.ledger.routine_names ~key:id ~data:name;

  (* Required inputs for the initialization check below: the backend routine's materialized
     read-only / read-before-write nodes, resolved against this compile's placements (the
     context-scoped memory-modes split removed the pre-lowering [context_nodes] settlement). Nodes
     with registered host initialization data (ndarray-backed literals, loaded tensors)
     self-initialize at link time from [Host_inits] (gh-ocannl-333), so they are excluded. *)
  let inputs =
    Set.filter (Set.diff backend_inputs comp.Asgns.embedded_nodes) ~f:(fun tn ->
        not (Ir.Host_inits.mem tn))
  in

  (* Outputs are all nodes written by the computation *)
  let outputs = backend_outputs in

  let updated_ctx = { ctx with wrapped; frontier = new_frontier } in

  let routine =
    {
      context = updated_ctx;
      task;
      bindings = lowered_bindings;
      name;
      inputs;
      outputs;
      routine_id = id;
      execution_deps = deps;
    }
  in

  (updated_ctx, routine)

let run ctx routine =
  (* Check that all required inputs are initialized. A node counts as initialized if it was produced
     by a prior routine ([initialized_nodes]) or is already allocated in the running context's
     device buffers ([in_backend]): such inputs are either user-set via [set_values]/[from_host]
     (which write the allocated buffer in place) or zero-initialized at allocation, which is the
     correct identity for read-only accumulators (e.g. gradients). NOTE (Codex P1): this does not
     distinguish a forgotten non-zero data input from a zero-valid accumulator — both are
     [alloc_zeros]'d read-only buffers — so a forgotten data input reads zeros rather than failing.
     Catching that precisely needs per-node "needs-nonzero-init" metadata OCANNL does not currently
     carry; a stricter check produces false positives on read-only accumulator gradients
     (zero2hero_1of7, primitive_ops). *)
  let ctx_buffers = Backends.query ctx.wrapped { q = (fun _ c -> c.BI.ctx_buffers) } in
  let in_backend tn = Map.mem ctx_buffers tn in
  let missing_inputs =
    Set.filter routine.inputs ~f:(fun tn -> not (Set.mem ctx.initialized_nodes tn || in_backend tn))
  in
  (if not (Set.is_empty missing_inputs) then
     let missing_names =
       Set.to_list missing_inputs |> List.map ~f:Tn.debug_name |> String.concat ~sep:", "
     in
     failwith (Printf.sprintf "Context.run: required input nodes not initialized: %s" missing_names));

  (* Check execution dependencies *)
  let missing_deps = Set.diff routine.execution_deps ctx.ledger.executed in
  (if not (Set.is_empty missing_deps) then
     let dep_names =
       Set.to_list missing_deps
       |> List.filter_map ~f:(fun dep_id ->
           Option.map (Hashtbl.find ctx.ledger.routine_names dep_id) ~f:(fun n ->
               Printf.sprintf "%s (id=%d)" n dep_id))
       |> String.concat ~sep:", "
     in
     failwith
       (Printf.sprintf "Context.run: routine %s (id=%d) has unexecuted dependencies: %s"
          routine.name routine.routine_id dep_names));

  (* Bind-time validation of launch parameters (docs/proposals/signed-index-precision.md): each
     bound value must be non-negative, within its declared static range, and within the index
     width. *)
  Idx.validate_lowered_bindings ~width64:Utils.settings.large_models routine.bindings;

  (* Run the routine's task/schedule *)
  Ir.Task.run routine.task;

  (* Mark executed in shared ledger *)
  ctx.ledger.executed <- Set.add ctx.ledger.executed routine.routine_id;

  (* Mark outputs as initialized and return updated context *)
  let initialized_nodes = Set.union ctx.initialized_nodes routine.outputs in
  { ctx with initialized_nodes }

let sync ctx =
  Backends.query ctx.wrapped
    {
      q =
        (fun (type dev runner event)
          (module Backend : BI.Backend
            with type dev = dev
             and type runner = runner
             and type event = event)
          c
        -> Backend.await c.BI.device);
    }

let hardware_limits ctx =
  Backends.query ctx.wrapped
    {
      q =
        (fun (type dev runner event)
          (module Backend : BI.Backend
            with type dev = dev
             and type runner = runner
             and type event = event)
          _c
        -> Backend.hardware_limits ());
    }

(* Internal helper - not exposed in interface to maintain invariants *)
let mark_initialized ctx nodes =
  { ctx with initialized_nodes = Set.union ctx.initialized_nodes nodes }

(* {2 On-demand host access (gh-ocannl-333)}

   All CPU-side value access goes through these context-mediated transfers. There is no host copy
   stored on the tensor node, and there is no cache: each call allocates a fresh temporary host
   buffer and performs a device-to-host (or host-to-device) transfer. This is intentionally
   expensive on non-unified-memory backends — callers should batch access rather than poll. *)

(* A fresh temporary host buffer matching the node's (padded) device buffer. *)
let host_buffer (tn : Tn.t) =
  Nd.create_array
    ~debug:("Context host buffer for " ^ Tn.debug_name tn)
    (Lazy.force tn.Tn.storage_prec) ~dims:(Lazy.force tn.Tn.dims) ~padding:(Lazy.force tn.Tn.padding)

(** Whether [tn] has a device buffer allocated in this context. *)
let mem ctx (tn : Tn.t) : bool =
  Backends.query ctx.wrapped { q = (fun _ c -> Map.mem c.BI.ctx_buffers tn) }

(* For-print proxies (gh-ocannl-333 AC 5): when a tensor's node is not materialized in a context,
   [Train.printf] recompiles a copy ([%cd "for_print" =: t]) into a fresh node and registers it here
   as a proxy for the source node, so {!to_host} can read the source's value through the copy. The
   table is keyed by the source node and holds the proxy node; it is read-only from [to_host]'s
   point of view and is for printing only — never a general host cache. (Keyed by the tnode, whose
   identity is the never-reused [uid]: an id-keyed table here would resolve stale proxies for reused
   ids after [Tensor.unsafe_reinitialize].) *)
let for_print_proxies : Tn.t Hashtbl.M(Tn).t = Hashtbl.create (module Tn)

let register_for_print ~(src : Tn.t) ~(proxy : Tn.t) =
  Hashtbl.set for_print_proxies ~key:src ~data:proxy

(* A deep copy of a host [Ndarray] (same precision, dims, and layout). Used so reads of shared
   initialization buffers hand the caller a private buffer it may mutate. *)
let copy_nd (src : Nd.t) : Nd.t =
  Nd.apply_with_prec
    {
      f =
        (fun prec arr ->
          let dst =
            Bigarray.Genarray.create (Bigarray.Genarray.kind arr) Bigarray.c_layout
              (Bigarray.Genarray.dims arr)
          in
          Bigarray.Genarray.blit arr dst;
          Nd.as_array prec dst);
    }
    src

(** Transfers [tn]'s device buffer into a fresh host [Ndarray] and returns it. Raises if the node is
    not present in the context (and has no host-init data or for-print proxy). *)
let to_host ctx (tn : Tn.t) : Nd.t =
  (* An [\@|] slice view is addressed through its parent (gh-ocannl-293 293a): an eligible slice
     owns no buffer, and an ineligible (copy) slice's value is recomputed from the parent each run.
     Reject direct host reads uniformly -- read the parent tensor instead. [slice_of] is set eagerly
     at construction, so this also covers the window before lowering decides eligibility. *)
  (match Tn.slice_of tn with
  | Some (parent, _) ->
      raise
      @@ Utils.User_error
           (Printf.sprintf
              "Context.to_host: node %s is an @| slice view; read its parent %s instead"
              (Tn.debug_name tn) (Tn.debug_name parent))
  | None -> ());
  let nd = host_buffer tn in
  (* [transfer] awaits pending device writes feeding the node, attempts the device-to-host copy, and
     awaits its completion before the host buffer is read. *)
  let transfer node =
    Backends.query ctx.wrapped
      {
        q =
          (fun (type dev runner event)
            (module Backend : BI.Backend
              with type dev = dev
               and type runner = runner
               and type event = event)
            c
          ->
            Backend.await c.BI.device;
            if Backend.to_host c node nd then (
              Backend.await c.BI.device;
              true)
            else false);
      }
  in
  if transfer tn then nd
  else
    match Ir.Host_inits.find tn with
    | Some init ->
        (* An ndarray-backed literal that is not part of any computation in this context (so it was
           never allocated on the device): its value is its registered host initialization data.
           Return a private copy so a mutating caller (e.g. [set_value]'s read-modify-write) cannot
           corrupt the shared initialization buffer used to initialize other contexts. *)
        copy_nd (Lazy.force init)
    | None -> (
        (* Read through a for-print proxy, if a copy of [tn] was materialized for printing. *)
        match Hashtbl.find for_print_proxies tn with
        | Some proxy when transfer proxy -> nd
        | _ ->
            raise
            @@ Utils.User_error
                 (Printf.sprintf "Context.to_host: node %s is not present in context (backend %s)"
                    (Tn.debug_name tn) (backend_name ctx)))

(** Uploads the host buffer [nd] into [tn]'s device buffer, allocating it if needed, and returns a
    context in which [tn] is marked initialized (so a subsequent {!run} reading [tn] succeeds). *)
let from_host ctx (tn : Tn.t) (nd : Nd.t) : t =
  (* Reject direct host writes to an [\@|] slice view (gh-ocannl-293 293a). Critically, [slice_of]
     is set eagerly at construction, so this fires even when the slice has not been lowered yet and
     [alias_of] is still [None] -- without it the [init_from_host] fallback below would allocate a
     fresh detached buffer for the slice that later alias lowering orphans (the host write would
     update neither the parent nor any buffer the kernels read). Write the parent instead. *)
  (match Tn.slice_of tn with
  | Some (parent, _) ->
      raise
      @@ Utils.User_error
           (Printf.sprintf
              "Context.from_host: node %s is an @| slice view; write its parent %s instead"
              (Tn.debug_name tn) (Tn.debug_name parent))
  | None -> ());
  (* Interval analysis, Phase B: a host write acts as a writer around the bounds-settlement point --
     pre-settlement it proposes the scanned [min, max] into the node's bounds candidate,
     post-settlement it validates against the settled bounds (or raises). See
     [Tnode.bounds_state]. *)
  Tn.propose_bounds_from_host tn nd;
  let wrapped, () =
    Backends.with_backend ctx.wrapped
      {
        f =
          (fun (type dev runner event)
            (module Backend : BI.Backend
              with type dev = dev
               and type runner = runner
               and type event = event)
            c
          ->
            (* Await pending device work BEFORE the upload, mirroring [to_host]: backends with
               host-visible (Shared) buffers implement [from_host] as a direct CPU memcpy, which
               already-queued kernels writing [tn] (e.g. a just-scheduled parameter initialization)
               would otherwise execute after and overwrite — [set_values] right after
               [Train.init_params] silently lost its writes on Metal. *)
            Backend.await c.BI.device;
            let c = if Backend.from_host c tn nd then c else Backend.init_from_host c tn nd in
            Backend.await c.BI.device;
            (c, ()));
      }
  in
  mark_initialized { ctx with wrapped } (Set.singleton (module Tn) tn)

(** Copies [tn]'s device buffer from [src] into [dst] (or into [dst]'s stream's merge buffer for
    [~into_merge_buffer:Copy]), returning the updated destination context. When both contexts come
    from the same backend, the pair match on {!Backends.wrapped_context} recovers type equality and
    the copy dispatches to the backend's [device_to_device] transfer machinery; otherwise it falls
    back to a host round-trip. *)
let copy ?(into_merge_buffer = BI.No) ~src ~dst tn =
  (* The fallback also serves nodes with no device buffer in [src]: [to_host] reads host-init
     literals and for-print proxies. A merge buffer cannot be filled host-side, so [Copy] raises
     where the fallback would engage. *)
  let host_roundtrip what =
    match into_merge_buffer with
    | BI.No -> from_host dst tn (to_host src tn)
    | BI.Copy ->
        raise
        @@ Utils.User_error
             (Printf.sprintf "Context.copy: cannot fill the merge buffer with node %s: %s"
                (Tn.debug_name tn) what)
  in
  let same (type dev runner event)
      (module Backend : BI.Backend
        with type dev = dev
         and type runner = runner
         and type event = event)
      ~(rewrap : (dev, runner, event) BI.context -> Backends.wrapped_context)
      (sctx : (dev, runner, event) BI.context) (dctx : (dev, runner, event) BI.context) =
    match Backend.device_to_device tn ~into_merge_buffer ~dst:dctx ~src:sctx with
    | Some r -> (
        (* The transfer routine's schedule is ordered on [dst]'s stream; host reads await the device
           as usual. For [Copy], the rewrapped [r.context] is what carries [merge_buffer_node = Some
           tn] into the next [compile]'s static merge-node check (gh-ocannl-288). *)
        Ir.Task.run r.BI.schedule;
        let dst = { dst with wrapped = rewrap r.BI.context } in
        match into_merge_buffer with
        | BI.No -> mark_initialized dst (Set.singleton (module Tn) tn)
        | BI.Copy -> dst)
    | None ->
        if not (Map.mem sctx.BI.ctx_buffers tn) then
          host_roundtrip "the node is absent from the source context"
        else if not (Map.mem dctx.BI.ctx_buffers tn) then
          (* Present in [src], absent in [dst]: allocate in [dst] and schedule the copy. *)
          mark_initialized
            { dst with wrapped = rewrap (Backend.init_from_device tn ~dst:dctx ~src:sctx) }
            (Set.singleton (module Tn) tn)
        else
          (* The source and destination buffers are physically the same: nothing to transfer. *)
          mark_initialized dst (Set.singleton (module Tn) tn)
  in
  match (src.wrapped, dst.wrapped) with
  | Backends.Cc_ctx s, Backends.Cc_ctx d ->
      same (module Backends.Cc_b) ~rewrap:(fun c -> Backends.Cc_ctx c) s d
  | Backends.Multidev_cc_ctx s, Backends.Multidev_cc_ctx d ->
      same (module Backends.Multidev_cc_b) ~rewrap:(fun c -> Backends.Multidev_cc_ctx c) s d
  | Backends.Cuda_ctx s, Backends.Cuda_ctx d ->
      same (module Backends.Cuda_b) ~rewrap:(fun c -> Backends.Cuda_ctx c) s d
  | Backends.Hip_ctx s, Backends.Hip_ctx d ->
      same (module Backends.Hip_b) ~rewrap:(fun c -> Backends.Hip_ctx c) s d
  | Backends.Metal_ctx s, Backends.Metal_ctx d ->
      same (module Backends.Metal_b) ~rewrap:(fun c -> Backends.Metal_ctx c) s d
  | ( ( Backends.Cc_ctx _ | Backends.Multidev_cc_ctx _ | Backends.Cuda_ctx _ | Backends.Hip_ctx _
      | Backends.Metal_ctx _ ),
      _ ) ->
      host_roundtrip
        (Printf.sprintf "cross-backend transfer (%s to %s)" (backend_name src) (backend_name dst))

let get_values ctx (tn : Tn.t) : float array =
  let nd = to_host ctx tn in
  let padding = Option.map ~f:fst (Lazy.force tn.Tn.padding) in
  Nd.retrieve_flat_values ?padding nd

let set_values ctx (tn : Tn.t) (values : float array) : t =
  let nd = host_buffer tn in
  let padding = Option.map ~f:fst (Lazy.force tn.Tn.padding) in
  Nd.set_flat_values ?padding nd values;
  from_host ctx tn nd

let get_value ctx (tn : Tn.t) (idx : int array) : float =
  let nd = to_host ctx tn in
  let padding = Option.map ~f:fst (Lazy.force tn.Tn.padding) in
  let idx =
    if Array.length (Lazy.force tn.Tn.dims) = 0 && Array.length idx = 1 then
      if idx.(0) = 0 then [||] else invalid_arg "Context.get_value: index out of bounds"
    else idx
  in
  Nd.get_as_float ?padding nd idx

(* Reads the current device buffer, sets one element, and uploads the whole buffer back, so that the
   other elements are preserved. *)
let set_value ctx (tn : Tn.t) (idx : int array) (v : float) : t =
  let nd = to_host ctx tn in
  let padding = Option.map ~f:fst (Lazy.force tn.Tn.padding) in
  Nd.set_from_float ?padding nd idx v;
  from_host ctx tn nd

let points_1d ?from_axis ~xdim ctx (tn : Tn.t) =
  let nd = to_host ctx tn in
  let padding = Option.map ~f:fst (Lazy.force tn.Tn.padding) in
  Nd.retrieve_1d_points ?from_axis ?padding ~xdim nd

let points_2d ?from_axis ~xdim ~ydim ctx (tn : Tn.t) =
  let nd = to_host ctx tn in
  let padding = Option.map ~f:fst (Lazy.force tn.Tn.padding) in
  Nd.retrieve_2d_points ?from_axis ?padding ~xdim ~ydim nd

let is_initialized ctx node = Set.mem ctx.initialized_nodes node
let device_id ctx = ctx.device_id

let get_used_memory ctx =
  Backends.query ctx.wrapped
    {
      q =
        (fun (type dev runner event)
          (module Backend : BI.Backend
            with type dev = dev
             and type runner = runner
             and type event = event)
          c
        -> Backend.get_used_memory c.BI.device);
    }

let placements ctx =
  Backends.query ctx.wrapped { q = (fun _ c -> c.BI.optimize_ctx.Ir.Low_level.placements) }

let decide_materialized ctx tns =
  let wrapped, () =
    Backends.with_backend ctx.wrapped
      {
        f =
          (fun (type dev runner event)
            (module Backend : BI.Backend
              with type dev = dev
               and type runner = runner
               and type event = event)
            bctx
          ->
            (* Fork the lineage state exactly like a compile would, then record the decisions in the
               fork: the argument context and its other descendants are unaffected. *)
            let optimize_ctx = Ir.Low_level.copy_optimize_ctx bctx.BI.optimize_ctx in
            let plc = optimize_ctx.Ir.Low_level.placements in
            List.iter tns ~f:(fun tn ->
                match Tn.Placements.get plc tn with
                | None | Some ((Tn.Never_virtual | Tn.On_device), _) ->
                    Tn.Placements.update plc tn Tn.On_device 31
                | Some ((Tn.Virtual | Tn.Local | Tn.Effectively_constant), _) -> ());
            (Backend.make_child ~optimize_ctx bctx, ()));
      }
  in
  { ctx with wrapped }
