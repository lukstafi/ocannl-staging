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
  mutable poisoned : (string * exn) option;
      (** Set when a launch or synchronization failed in a way that may have left device buffers of
          this lineage partially written (gh-ocannl-536). [Context.run] marks a routine executed
          before the later [sync] can report an asynchronous failure, so the ledger would otherwise
          claim a routine completed that did not. There is no restore API, so the lineage stays
          unusable: every entrypoint re-raises the stored failure, naming the routine. Scratch
          lineages ([timing_ctx]) are separate, so poisoning one does not condemn the caller's
          context. *)
}
(** Shared mutable state for execution tracking, allocated once per root context. Shared by
    reference across all contexts derived from the same root. *)

let empty_frontier = { last_writer = Map.empty (module Tn); last_readers = Map.empty (module Tn) }

let create_ledger () =
  {
    next_id = 0;
    routine_names = Hashtbl.create (module Int);
    executed = Set.empty (module Int);
    poisoned = None;
  }

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

(* gh-ocannl-536 landing step 5: backend selection is not candidate compilation, so it does not use
   the compile-phase policy — but it used to catch everything, which turned a broken driver, an
   assertion failure, or an interrupt into a silent downgrade to another backend (and, on the
   configured arm, into the misleading "Unknown backend"). Only {!BI.Backend_unavailable} — raised
   by device discovery when the library is not linked in or the driver reports no devices — advances
   to the next candidate; everything else propagates with its original backtrace. *)
let advances_to_next_backend = function BI.Backend_unavailable _ -> true | _ -> false

let auto () =
  (* First check if a backend is configured globally *)
  match Utils.get_global_arg ~arg_name:"backend" ~default:"" with
  | "" ->
      (* No global config, try backends in order of preference *)
      let backends_to_try = [ "metal"; "cuda"; "hip"; "cc" ] in
      let rec try_backends unavailable = function
        | [] ->
            failwith
              ("Context.auto: no backend available; tried "
              ^ String.concat ~sep:", " (List.rev unavailable))
        | name :: rest -> (
            match create_from_backend_name ~device_id:0 name with
            | ctx -> ctx
            | exception exn when advances_to_next_backend exn ->
                try_backends (Exn.to_string exn :: unavailable) rest)
      in
      try_backends [] backends_to_try
  | backend_name ->
      (* Use the configured backend. An unknown name already raises a message naming it
         ([Backends.get_backend]); an unusable one keeps its own failure rather than being relabeled
         as a spelling mistake. *)
      create_from_backend_name ~device_id:0 backend_name

let compile_outcome ?name ?lowered_transform ?lowered_transforms ~provenance ?candidate ctx comp
    bindings =
  (* Compile and link on the wrapped backend context; only backend-independent routine components
     (and, via [with_backend]'s rebuilt constructor, the updated context) escape the dispatch. *)
  let wrapped, backend_outcome =
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
            match
              Ir.Schedule_outcome.protect ~classify_backend:Backend.classify_failure ~provenance
                ~phase:Ir.Schedule_outcome.Transform ?candidate (fun () ->
                  let code =
                    Backend.compile ?name ?lowered_transform ?lowered_transforms
                      bctx.BI.optimize_ctx bindings comp
                  in
                  Ir.Schedule_outcome.tag Ir.Schedule_outcome.Backend_link (fun () ->
                      Backend.link bctx code))
            with
            | Ok r ->
                ( r.BI.context,
                  Ok (r.BI.schedule, r.BI.bindings, r.BI.name, r.BI.inputs, r.BI.outputs) )
            | Error failure -> (bctx, Error failure));
      }
  in
  match backend_outcome with
  | Error failure -> Error failure
  | Ok (task, lowered_bindings, name, backend_inputs, backend_outputs) ->
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

  Ok (updated_ctx, routine)

let compile ?name ?lowered_transform ?lowered_transforms ctx comp bindings =
  match
    compile_outcome ?name ?lowered_transform ?lowered_transforms
      ~provenance:Ir.Schedule_outcome.User_schedule ctx comp bindings
  with
  | Ok result -> result
  | Error failure -> Ir.Schedule_outcome.raise_failure failure

(* {2 Failure classification at the launch/sync boundary (gh-ocannl-536)}

   The compile path passes [Backend.classify_failure] into [protect] through the same first-class
   module dispatch as compilation itself. Launch and sync need it too — that is where a driver
   reports a candidate's kernel as unrunnable — but they are not compile-shaped: the public
   [run]/[sync] contracts stay raising APIs, and callers that want typed outcomes (the autotuner)
   wrap them with the classifier this accessor hands out. Without it, a backend's classifier is
   simply never consulted for a launch failure, and every such failure is fatal by phase default. *)
let failure_classifier ctx :
    Ir.Schedule_outcome.phase -> exn -> Ir.Schedule_outcome.classified_cause option =
  Backends.query ctx.wrapped
    {
      q =
        (fun (type dev runner event)
          (module Backend : BI.Backend
            with type dev = dev
             and type runner = runner
             and type event = event)
          _c
        -> Backend.classify_failure);
    }

let poisoned_failure ctx =
  Option.map ctx.ledger.poisoned ~f:(fun (name, exn) ->
      Failure
        (Printf.sprintf
           "Context: this execution lineage was poisoned by a failure of routine %s that may have \
            written device buffers; there is no restore API, so it cannot be reused. Original \
            failure: %s"
           name (Exn.to_string exn)))

let check_not_poisoned ctx = Option.iter (poisoned_failure ctx) ~f:raise

(** Undoes [run]'s optimistic execution marking for a routine whose failure is known not to have
    written device buffers, so the next candidate does not inherit a dependency that never ran. *)
let rollback_execution ctx routine_id =
  ctx.ledger.executed <- Set.remove ctx.ledger.executed routine_id

let poison_lineage ctx ~routine_name exn =
  if Option.is_none ctx.ledger.poisoned then ctx.ledger.poisoned <- Some (routine_name, exn)

(* The pre-dispatch validation of {!run}, callable on its own (gh-ocannl-550): everything here
   happens BEFORE [Ir.Task.run], so a failure it raises proves the routine was never dispatched and
   the device wrote nothing. A caller that wraps [run] in a launch-tagged failure boundary (the
   autotuner's timing runs) validates through this in its own [Schedule_outcome.Preflight] region,
   so an unattributed failure at [Launch] means dispatch was attempted — which makes condemning the
   lineage there sound, and keeps a mere unsatisfied dependency or an out-of-range binding from
   condemning it (gh-ocannl-564). *)
let check_runnable ctx routine =
  check_not_poisoned ctx;
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
  Idx.validate_lowered_bindings ~width64:Utils.settings.large_models routine.bindings

let run ctx routine =
  check_runnable ctx routine;

  (* Run the routine's task/schedule *)
  Ir.Task.run routine.task;

  (* Mark executed in shared ledger *)
  ctx.ledger.executed <- Set.add ctx.ledger.executed routine.routine_id;

  (* Mark outputs as initialized and return updated context *)
  let initialized_nodes = Set.union ctx.initialized_nodes routine.outputs in
  { ctx with initialized_nodes }

let sync ctx =
  check_not_poisoned ctx;
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
    (Lazy.force tn.Tn.storage_prec) ~dims:(Lazy.force tn.Tn.dims)
    ~padding:(Lazy.force tn.Tn.padding)

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
  check_not_poisoned ctx;
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
  check_not_poisoned ctx;
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
  (* Both lineages, and BEFORE dispatch: the same-backend path runs the transfer schedule directly
     rather than through [to_host], so checking only the host round-trip would let a poisoned source
     export a possibly partially written buffer into a clean lineage — exactly what poisoning is for
     (Codex P2 on PR #256). Reading from a condemned lineage and running transfer work on one are
     both refused. *)
  check_not_poisoned src;
  check_not_poisoned dst;
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

(* gh-560: the analyze-only entry point — lowering and optimization without backend codegen or
   linking. [Backends.lower_assignments] forks the lineage state itself, so the surface is read off
   a hermetic sibling: the argument context, its ledger and frontier are unaffected. With the
   analysis cache (gh-560), a context that already compiled this routine (e.g. the tuner's arms)
   pays only the [specialize_proc] replay here. *)
let decision_surface ?name ctx comp bindings =
  let optim_ctx = Backends.query ctx.wrapped { q = (fun _ c -> c.BI.optimize_ctx) } in
  let _name, (lowered : Ir.Low_level.optimized) =
    Backends.lower_assignments optim_ctx ?name bindings comp.Asgns.asgns
  in
  lowered.Ir.Low_level.flip_candidates

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

let decide_inline ctx tns =
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
            (* Fork like [decide_materialized]; the preference is recorded rather than a placement
               decided, because inlining legality is settled only during optimization
               ([check_and_store_virtual]) — a preferred node the virtualizer rejects still
               materializes. A node whose placement THIS lineage already decided (e.g. a cap
               materialization from an earlier compile of a routine setting it) keeps that
               decision: decisions are final within a lineage — already-compiled routines depend on
               them (a consumer compiled against the node's buffer must find it written) — so the
               preference only steers placements not yet decided. Callers wanting the exemption to
               take effect fork a pre-compile sibling, as [Train.tune_placements] does. *)
            let optimize_ctx = Ir.Low_level.copy_optimize_ctx bctx.BI.optimize_ctx in
            List.iter tns
              ~f:(Hash_set.add optimize_ctx.Ir.Low_level.inline_preferences);
            (Backend.make_child ~optimize_ctx bctx, ()));
      }
  in
  { ctx with wrapped }

(* {2 gh-ocannl-498: budget-driven recompute-vs-store} *)

type memory_budget = Bytes of int | Minimize [@@deriving sexp_of]

type budget_plan = {
  bp_baseline : Backends.footprint;
  bp_final : Backends.footprint;
  bp_flips : (Tn.t * int * int) list;
  bp_considered : int;
  bp_dropped : int;
  bp_within_budget : bool;
}
[@@deriving sexp_of]

(* gh-ocannl-498: compare the rationals [ra/ca] and [rb/cb] EXACTLY, for ranking candidates by
   footprint relief per unit of recompute cost. [ca] and [cb] must be positive; the numerators are
   byte counts and may be negative, since inlining a node can lengthen other nodes' spans and cost
   footprint rather than free it.

   Cross-multiplying would be the obvious comparison and is wrong: both factors are legitimately
   large (bytes against reduction extent x read multiplicity), so the products can wrap and silently
   invert the order. The Euclidean/continued-fraction descent uses only division and remainder, so
   it cannot overflow, and stays bit-reproducible unlike a float ratio -- but it assumes
   NON-NEGATIVE numerators: OCaml's division truncates toward zero, so a negative numerator inverts
   the very comparison the descent is making ([-1/10] would rank above [1/10], and [0/1] above
   [-1/5]). The sign is therefore settled first, and two negatives are compared by reversed
   magnitude. *)
let compare_relief_ratio ra ca rb cb =
  let rec nonneg ra ca rb cb =
    (* Both numerators non-negative here; denominators positive. *)
    let qa = ra / ca and qb = rb / cb in
    if qa <> qb then Int.compare qa qb
    else
      let ma = ra - (qa * ca) and mb = rb - (qb * cb) in
      if ma = 0 then if mb = 0 then 0 else -1
      else if mb = 0 then 1
      else (* both fractional parts nonzero: compare ca/ma with cb/mb, inverted. *)
        nonneg cb mb ca ma
  in
  match (ra >= 0, rb >= 0) with
  | true, true -> nonneg ra ca rb cb
  | true, false -> 1
  | false, true -> -1
  (* Both negative: |ra|/ca vs |rb|/cb with the order reversed. *)
  | false, false -> nonneg (-rb) cb (-ra) ca

let log_memory_budget () = Utils.get_global_flag ~default:false ~arg_name:"log_memory_budget"

(* One hermetic analysis of [comp] from [ctx]'s lineage with [inline] additionally preferred inline:
   the footprint the resulting placement vector implies, plus the decision surface it reports.
   [Backends.lower_assignments] forks the lineage state, so nothing here reaches [ctx] -- and the
   gh-560 analysis cache makes every call after the first one a specialization replay. *)
let analyze_footprint ?name ~(inline : Tn.t list) ctx comp bindings :
    Backends.footprint * Ir.Low_level.flip_candidate list =
  let optim_ctx = Backends.query ctx.wrapped { q = (fun _ c -> c.BI.optimize_ctx) } in
  let optim_ctx = Ir.Low_level.copy_optimize_ctx optim_ctx in
  List.iter inline ~f:(Hash_set.add optim_ctx.Ir.Low_level.inline_preferences);
  let _name, (lowered : Ir.Low_level.optimized) =
    Backends.lower_assignments optim_ctx ?name bindings comp.Asgns.asgns
  in
  ( Backends.score_footprint ~backend_name:(backend_name ctx) ~limits:(hardware_limits ctx)
      ~static_indices:(Idx.bound_symbols bindings) lowered,
    lowered.Ir.Low_level.flip_candidates )

let footprint ?name ctx comp bindings = fst (analyze_footprint ?name ~inline:[] ctx comp bindings)

let plan_memory_budget ?name ?max_candidates ~budget ctx comp bindings =
  (* [Minimize] promises every flip that still relieves footprint, so it must not silently stop at a
     default cut -- and a config-only user (memory_budget=minimize) has no way to raise one. It
     therefore defaults to unbounded, paying two lowerings per candidate; a caller who wants that
     bounded passes [max_candidates] explicitly, which applies to both budget kinds. A byte budget
     stops as soon as it is met, so its default cut is a cost guard, not a semantic one. *)
  let max_candidates =
    match (max_candidates, budget) with
    | Some n, _ -> n
    | None, Minimize -> Int.max_value
    | None, Bytes _ -> 32
  in
  if not (Utils.get_global_flag ~default:false ~arg_name:"buffer_aliasing") then
    raise
    @@ Utils.User_error
         "Context.plan_memory_budget: a memory budget needs the liveness memory planner (config \
          buffer_aliasing=true) -- without it every node is always-live, the footprint score \
          degenerates to bump packing, and the relief of demoting an intermediate is unrelated to \
          what the allocator would do"
  else begin
    let logf fmt =
      Stdlib.Printf.ksprintf
        (fun s -> if log_memory_budget () then Stdio.eprintf "memory budget: %s\n%!" s)
        fmt
    in
    let score inline = fst (analyze_footprint ?name ~inline ctx comp bindings) in
    let bp_baseline, surface = analyze_footprint ?name ~inline:[] ctx comp bindings in
    (* The acceptance-stopping predicate: [Minimize] is never satisfied, so it keeps taking flips
       that still help. [within] is the reported outcome, where a target-less [Minimize] trivially
       holds -- there is no budget for it to miss. *)
    let met (fp : Backends.footprint) =
      match budget with Minimize -> false | Bytes b -> fp.Backends.fp_total <= b
    in
    let within (fp : Backends.footprint) =
      match budget with Minimize -> true | Bytes b -> fp.Backends.fp_total <= b
    in
    let done_ () =
      {
        bp_baseline;
        bp_final = bp_baseline;
        bp_flips = [];
        bp_considered = 0;
        bp_dropped = 0;
        bp_within_budget = within bp_baseline;
      }
    in
    if met bp_baseline then (
      logf "baseline %d bytes is already within budget; no flips" bp_baseline.Backends.fp_total;
      (ctx, done_ ()))
    else
      (* Only the [`Inline] direction: demoting a materialized intermediate to recompute-at-use is
         what relieves footprint. Ranked CHEAPEST-recompute-first for the pre-filter (the surface's
         own order is most-expensive-first, which the [Materialize]-direction search wants), so a
         [max_candidates] cut keeps the flips a budget would most want to pay for. *)
      let all =
        List.fold surface ~init:[] ~f:(fun acc fc ->
            match fc.Ir.Low_level.fc_flip with
            | `Materialize -> acc
            | `Inline ->
                if List.exists acc ~f:(fun c -> Tn.equal c.Ir.Low_level.fc_tn fc.Ir.Low_level.fc_tn)
                then acc
                else fc :: acc)
        |> List.sort ~compare:(fun a b ->
            match Int.compare a.Ir.Low_level.fc_recompute_cost b.Ir.Low_level.fc_recompute_cost with
            | 0 -> Tn.compare a.Ir.Low_level.fc_tn b.Ir.Low_level.fc_tn
            | c -> c)
      in
      let considered = List.take all max_candidates in
      let bp_dropped = List.length all - List.length considered in
      if bp_dropped > 0 then
        logf "%d of %d inline candidates dropped by max_candidates=%d (cheapest recompute kept)"
          bp_dropped (List.length all) max_candidates;
      (* Round 1: each candidate's relief against the ACTUAL baseline layout. A node whose span was
         already shared relieves nothing on its own (the gh-ocannl-558 lesson in reverse: relief is
         not a function of the node's own size). Solo relief only RANKS here -- a zero-relief
         candidate is kept, at the back, because relief is not additive in either direction: two
         nodes pinning the same arena peak each free nothing alone and the whole range together, so
         dropping them outright would report an otherwise reachable budget unreachable. Round 2
         picks those up jointly. *)
      let scored =
        List.map considered ~f:(fun fc ->
            let fp = score [ fc.Ir.Low_level.fc_tn ] in
            let relief = bp_baseline.Backends.fp_total - fp.Backends.fp_total in
            logf "candidate %s: recompute cost %d, solo relief %d bytes"
              (Tn.debug_name fc.Ir.Low_level.fc_tn)
              fc.Ir.Low_level.fc_recompute_cost relief;
            (fc, relief))
      in
      let ranked =
        List.sort scored ~compare:(fun (a, ra) (b, rb) ->
            let ca = max 1 a.Ir.Low_level.fc_recompute_cost
            and cb = max 1 b.Ir.Low_level.fc_recompute_cost in
            (* Descending by ratio, so [b] against [a]. *)
            match compare_relief_ratio rb cb ra ca with
            | 0 -> (
                match Int.compare rb ra with
                | 0 -> Tn.compare a.Ir.Low_level.fc_tn b.Ir.Low_level.fc_tn
                | c -> c)
            | c -> c)
      in
      (* Round 2: accept a prefix, re-scoring the CUMULATIVE vector each time. Inlining one node
         moves the others' live spans, so a candidate's solo relief is not what it is worth here. A
         candidate that adds nothing is held SPECULATIVELY rather than dropped: if a later one then
         relieves bytes on top of it, the whole speculative group is committed together (the
         two-nodes-at-one-peak case).

         Every candidate is therefore scored BOTH ways, with and without the held group, and the
         three outcomes are treated differently. Held flips are not merely unpaid, they can be
         actively HARMFUL — a flip whose marginal was negative moved someone's span the wrong way —
         and judging every later candidate only in their company would let one bad hold mask a
         candidate that pays on its own, losing it and, with it, a reachable budget.

         - joint strictly better: the group is load-bearing, so commit it with the candidate. -
         joint strictly worse: the group is harmful here, so commit the candidate alone and DISCARD
         the group (no group is reconsidered once discarded — this is a bounded planner, not a
         search over subsets). - equal: the group is merely neutral. Commit the candidate alone but
         KEEP holding it: committing it would pay recompute for zero bytes, and discarding it would
         throw away a flip that may still be half of a later pair. Dropping neutral holds eagerly
         measurably costs relief (on test/operations/memory_budget's step, 1196164 -> 1228932
         bytes).

         Speculatives never joined by a paying flip are discarded at the end, so no recompute is
         ever paid for zero bytes. The relief of a joint commit is reported on the flip that closed
         it, and the sum over [bp_flips] is exactly [bp_baseline - bp_final]. *)
      let accepted = ref [] and flips = ref [] and cur = ref bp_baseline in
      (* Held (node, recompute cost) pairs, most recently held first. *)
      let speculative = ref [] in
      let names l = String.concat ~sep:", " (List.map l ~f:(fun (tn, _) -> Tn.debug_name tn)) in
      List.iter ranked ~f:(fun (fc, solo) ->
          if not (met !cur) then begin
            let tn = fc.Ir.Low_level.fc_tn and cost = fc.Ir.Low_level.fc_recompute_cost in
            let held = !speculative in
            let cand_alone = tn :: !accepted in
            let fp_alone = score cand_alone in
            let cand_joint, fp_joint =
              if List.is_empty held then (cand_alone, fp_alone)
              else
                let c = (tn :: List.map held ~f:fst) @ !accepted in
                (c, score c)
            in
            let verdict =
              match compare_int fp_joint.Backends.fp_total fp_alone.Backends.fp_total with
              | c when c < 0 -> `Load_bearing
              | 0 -> `Neutral
              | _ -> `Harmful
            in
            let cand = match verdict with `Load_bearing -> cand_joint | _ -> cand_alone in
            let fp = match verdict with `Load_bearing -> fp_joint | _ -> fp_alone in
            let marginal = !cur.Backends.fp_total - fp.Backends.fp_total in
            if marginal > 0 then (
              logf "accept %s: %d bytes (solo %d), cost %d%s, footprint now %d" (Tn.debug_name tn)
                marginal solo cost
                (match (verdict, held) with
                | _, [] -> ""
                | `Load_bearing, _ -> Printf.sprintf " jointly with %s" (names held)
                | `Neutral, _ -> Printf.sprintf " alone, still holding %s" (names held)
                | `Harmful, _ -> Printf.sprintf " alone, dropping harmful held %s" (names held))
                fp.Backends.fp_total;
              (* [flips] is reverse-chronological until the final [List.rev]. A joint commit's held
                 flips carry 0 and the group's relief lands on the flip that made it pay. *)
              flips :=
                (tn, marginal, cost)
                ::
                (match verdict with
                | `Load_bearing -> List.map held ~f:(fun (h, c) -> (h, 0, c))
                | `Neutral | `Harmful -> [])
                @ !flips;
              accepted := cand;
              (* A neutral group stays held: committing it would pay recompute for zero bytes, and
                 dropping it would discard a flip that may still be half of a later pair. *)
              (speculative := match verdict with `Neutral -> held | _ -> []);
              cur := fp)
            else (
              logf "hold %s: no marginal relief yet (solo was %d); speculative" (Tn.debug_name tn)
                solo;
              speculative := (tn, cost) :: !speculative)
          end);
      (match !speculative with
      | [] -> ()
      | held ->
          logf "dropping %d speculative flip(s) that never paid: %s" (List.length held) (names held));
      let bp_final = !cur in
      let bp_within_budget = within bp_final in
      (match budget with
      | Minimize ->
          logf "minimized: %d -> %d bytes with %d flip(s)" bp_baseline.Backends.fp_total
            bp_final.Backends.fp_total (List.length !flips)
      | Bytes b ->
          logf "budget %d bytes: %d -> %d bytes with %d flip(s), %s" b bp_baseline.Backends.fp_total
            bp_final.Backends.fp_total (List.length !flips)
            (if bp_within_budget then "within budget" else "STILL OVER BUDGET"));
      let ctx = if List.is_empty !accepted then ctx else decide_inline ctx !accepted in
      ( ctx,
        {
          bp_baseline;
          bp_final;
          bp_flips = List.rev !flips;
          bp_considered = List.length considered;
          bp_dropped;
          bp_within_budget;
        } )
  end
