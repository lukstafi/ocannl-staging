open Base

(* The live table mirrors the backends' private pool tables, keyed the same way, because the shared
   allocator seam is the only place that knows a pool's byte size AND is backend-independent: the
   per-backend [Slab] tables are private by design (the concrete pointer type never appears in a
   shared signature), and [free_pool] is passed only a [pool_id].

   A mutex, like {!Backend_impl.Make_slab}'s [pools_mutex] and for the same reason: with the
   [Multidev] scheduler the allocator sites run on worker domains concurrently with link-time
   allocation on the main domain, and a Base hashtable is not domain-safe. *)
let live : (int * int, int * bool) Hashtbl.Poly.t = Hashtbl.Poly.create ()
let live_mutex = Stdlib.Mutex.create ()
let with_live f = Stdlib.Mutex.protect live_mutex f

let working_pools_allocated : Utils.atomic_int = Atomic.make 0
let constant_pools_allocated : Utils.atomic_int = Atomic.make 0
let pools_freed : Utils.atomic_int = Atomic.make 0
let contexts_created : Utils.atomic_int = Atomic.make 0
let contexts_released : Utils.atomic_int = Atomic.make 0
let modules_loaded : Utils.atomic_int = Atomic.make 0
let modules_unloaded : Utils.atomic_int = Atomic.make 0

let record_pool ~device_id ~pool_id ~constant ~size_in_bytes =
  if constant then ignore (Atomic.fetch_and_add constant_pools_allocated 1 : int)
  else ignore (Atomic.fetch_and_add working_pools_allocated 1 : int);
  with_live (fun () -> Hashtbl.set live ~key:(device_id, pool_id) ~data:(size_in_bytes, constant))

let forget_pool ~device_id ~pool_id =
  with_live (fun () ->
      let key = (device_id, pool_id) in
      if Hashtbl.mem live key then (
        Hashtbl.remove live key;
        ignore (Atomic.fetch_and_add pools_freed 1 : int)))

let count_context_created () = ignore (Atomic.fetch_and_add contexts_created 1 : int)
let count_context_released () = ignore (Atomic.fetch_and_add contexts_released 1 : int)
let count_module_loaded () = ignore (Atomic.fetch_and_add modules_loaded 1 : int)
let count_module_unloaded () = ignore (Atomic.fetch_and_add modules_unloaded 1 : int)

type t = {
  live_working_pools : int;
  live_working_bytes : int;
  live_constant_pools : int;
  live_constant_bytes : int;
  working_pools_allocated : int;
  constant_pools_allocated : int;
  pools_freed : int;
  contexts_created : int;
  contexts_released : int;
  modules_loaded : int;
  modules_unloaded : int;
}
[@@deriving sexp_of, equal]

let snapshot () =
  let live_working_pools, live_working_bytes, live_constant_pools, live_constant_bytes =
    with_live (fun () ->
        Hashtbl.fold live ~init:(0, 0, 0, 0)
          ~f:(fun ~key:_ ~data:(bytes, constant) (wp, wb, cp, cb) ->
            if constant then (wp, wb, cp + 1, cb + bytes) else (wp + 1, wb + bytes, cp, cb)))
  in
  {
    live_working_pools;
    live_working_bytes;
    live_constant_pools;
    live_constant_bytes;
    working_pools_allocated = Atomic.get working_pools_allocated;
    constant_pools_allocated = Atomic.get constant_pools_allocated;
    pools_freed = Atomic.get pools_freed;
    contexts_created = Atomic.get contexts_created;
    contexts_released = Atomic.get contexts_released;
    modules_loaded = Atomic.get modules_loaded;
    modules_unloaded = Atomic.get modules_unloaded;
  }

let live_pools c = c.live_working_pools + c.live_constant_pools
let live_pool_bytes c = c.live_working_bytes + c.live_constant_bytes
let live_contexts c = c.contexts_created - c.contexts_released
let live_modules c = c.modules_loaded - c.modules_unloaded
let mib bytes = Float.of_int bytes /. 1048576.

let to_string c =
  Printf.sprintf
    "pools %d live = %d working (%.1f MiB) + %d constant (%.1f MiB); %d allocated, %d freed | \
     contexts %d live (%d created, %d released) | modules %d live (%d loaded, %d unloaded)"
    (live_pools c) c.live_working_pools (mib c.live_working_bytes) c.live_constant_pools
    (mib c.live_constant_bytes)
    (c.working_pools_allocated + c.constant_pools_allocated)
    c.pools_freed (live_contexts c) c.contexts_created c.contexts_released (live_modules c)
    c.modules_loaded c.modules_unloaded
