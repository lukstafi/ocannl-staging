(* Regression test for the reserved merge-pool grow path (AC6/AC7): when [alloc_pool] is called for
   a pool id that already has a slab (only the reserved merge pool, id 0, is ever re-allocated in
   place), the previous backend allocation must be freed before it is replaced -- otherwise device
   memory grows without bound on every merge-buffer grow.

   The CUDA backend ([cuda_backend.ml] [Slab.alloc_pool]) and the shared [Make_slab.alloc_pool] use
   the identical free-on-overwrite pattern. CUDA is not buildable in this harness (no cudajit), so
   we pin the invariant through [Make_slab] with a mock raw backend whose [free_pool_raw] is [Some]
   (i.e. a backend that owns explicitly-freed pointers, like CUDA). The assertion "grow freed the
   old pool" would fail if [alloc_pool] overwrote the table entry without freeing -- the exact bug
   this fixes. A unique tnode pool id (never pre-existing) must free nothing. *)

open Base
module Backend_impl = Ir.Backend_impl
module Backend_intf = Ir.Backend_intf
module Tn = Ir.Tnode

(* A raw backend whose "pointers" are integer ids and whose [free_pool_raw] records frees --
   standing in for a backend (like CUDA) that owns explicitly-released device pointers. *)
module Mock_raw = struct
  type buffer_ptr = int

  let sexp_of_buffer_ptr = Int.sexp_of_t
  let get_used_memory () = 0
  let next = ref 0
  let freed : int list ref = ref []
  let fail_next_alloc = ref false

  let alloc_pool_raw ~size_in_bytes:_ =
    if !fail_next_alloc then (
      fail_next_alloc := false;
      failwith "injected merge-pool allocation failure")
    else (
      Int.incr next;
      !next)

  let free_pool_raw = Some (fun ptr -> freed := ptr :: !freed)
  let memset_zero_raw _ptr ~offset:_ ~size_in_bytes:_ = ()

  (* A "pointer" is an integer id; advancing it by [bytes] models sub-region addressing. *)
  let offset_buffer base ~bytes = base + bytes
  let buffer_to_buffer ~dst:_ ~src:_ ~size_in_bytes:_ = ()
  let host_to_buffer _nd ~dst:_ = ()
  let buffer_to_host _nd ~src:_ = ()
end

module Mock_config = struct
  type dev = unit
  type runner = unit
  type event = unit

  let sexp_of_dev = Base.sexp_of_unit
  let sexp_of_runner = Base.sexp_of_unit
  let sexp_of_event = Base.sexp_of_unit
  let name = "mock"
end

module Mock_dt = Backend_impl.Device_types_ll (Mock_config)
module Mock_slab = Backend_impl.Make_slab (Mock_dt) (Mock_raw)
module Mock_dev = Backend_impl.Device (Mock_dt) (Mock_slab)

(* A raw backend that relies on GC (no explicit deallocator), like the CPU backends. *)
module Mock_raw_gc = struct
  type buffer_ptr = int

  let sexp_of_buffer_ptr = Int.sexp_of_t
  let get_used_memory () = 0
  let next = ref 100

  let alloc_pool_raw ~size_in_bytes:_ =
    Int.incr next;
    !next

  let free_pool_raw = None (* relies on GC + the dropped table entry *)
  let memset_zero_raw _ptr ~offset:_ ~size_in_bytes:_ = ()
  let offset_buffer base ~bytes = base + bytes
  let buffer_to_buffer ~dst:_ ~src:_ ~size_in_bytes:_ = ()
  let host_to_buffer _nd ~dst:_ = ()
  let buffer_to_host _nd ~src:_ = ()
end

module Mock_gc_slab = Backend_impl.Make_slab (Mock_dt) (Mock_raw_gc)
module Mock_gc_dev = Backend_impl.Device (Mock_dt) (Mock_gc_slab)

let loc pool_id : Backend_intf.buffer_loc = { pool_id; offset = 0 }

let () =
  let device = Mock_dev.make_device () () ~ordinal:0 in
  (* Reserved merge pool (id 0): allocate, then grow it in place (re-allocate the same key). *)
  Mock_slab.alloc_pool device ~pool_id:0 ~size_in_bytes:16 ~alignment:1;
  let p1 = Mock_slab.resolve_pool device (loc 0) in
  Mock_slab.alloc_pool device ~pool_id:0 ~size_in_bytes:32 ~alignment:1;
  let p2 = Mock_slab.resolve_pool device (loc 0) in
  Verdict.p "grow freed the old pool" (List.mem !Mock_raw.freed p1 ~equal:Int.equal);
  Verdict.p "grow installed a new pool" (not (p1 = p2));
  Stdio.printf "freed count after grow = %d\n" (List.length !Mock_raw.freed);
  (* Failure control for the same seam: growing must not require old+new bytes to coexist. The old
     slab is released first, but its table entry and device capacity claim must be invalidated
     before any fallible action. Thus a failed replacement leaves an honestly absent merge pool,
     rather than the old bug's table entry pointing at released memory. *)
  let frees_before_failed_grow = List.length !Mock_raw.freed in
  let stale_writer =
    Tn.create (Tn.Default Ir.Ops.single) ~id:571 ~label:[ "stale merge writer" ]
      ~unpadded_dims:(lazy [| 1 |])
      ~padding:(lazy None)
      ()
  in
  device.merge_buffer := Some (loc 0);
  device.merge_buffer_capacity <- 32;
  device.updating_for_merge_buffer <- Some (stale_writer, None);
  Mock_raw.fail_next_alloc := true;
  let grow_failed =
    match Mock_slab.alloc_pool device ~pool_id:0 ~size_in_bytes:64 ~alignment:1 with
    | () -> false
    | exception Failure msg -> String.is_substring msg ~substring:"injected"
  in
  Verdict.p "injected grow failure fired" grow_failed;
  let old_pool_absent =
    match Mock_slab.resolve_pool device (loc 0) with _ -> false | exception _ -> true
  in
  Verdict.p "failed grow invalidated the released pool, capacity, and writer"
    (old_pool_absent
    && List.length !Mock_raw.freed = frees_before_failed_grow + 1
    && List.mem !Mock_raw.freed p2 ~equal:Int.equal
    && Option.is_none !(device.merge_buffer)
    && device.merge_buffer_capacity = 0
    && Option.is_none device.updating_for_merge_buffer);
  Mock_slab.alloc_pool device ~pool_id:0 ~size_in_bytes:64 ~alignment:1;
  let p3 = Mock_slab.resolve_pool device (loc 0) in
  Verdict.p "grow retry installs a fresh pool without another free"
    ((not (p2 = p3)) && List.length !Mock_raw.freed = frees_before_failed_grow + 1);
  (* A unique tnode pool id never pre-exists, so allocating it frees nothing. *)
  Mock_slab.alloc_pool device ~pool_id:1 ~size_in_bytes:16 ~alignment:1;
  Stdio.printf "freed count after unique-id alloc = %d\n" (List.length !Mock_raw.freed);

  (* Pooled addressing: resolving { pool_id; offset } must advance the slab base by [offset] bytes
     (the multi-tenant-pool invariant). resolve_pool at offset 0 returns the base; at offset N
     returns base + N. If [resolve_pool] reverted to asserting offset = 0, the second call below
     would raise instead of returning base + 8. *)
  let base1 = Mock_slab.resolve_pool device (loc 1) in
  let base1_at8 = Mock_slab.resolve_pool device { Backend_intf.pool_id = 1; offset = 8 } in
  Verdict.p "resolve_pool offset 0 = base"
    (base1_at8 - base1 = 8 && Mock_slab.resolve_pool device (loc 1) = base1);
  Verdict.p "resolve_pool offset 8 = base + 8" (base1_at8 = base1 + 8);

  (* free_pool must drop the private table entry even for a GC-reliant backend (free_pool_raw =
     None), so the strong reference is released and the buffer can be reclaimed. If free_pool were
     [None] (the bug), [finalize] would never remove these entries. *)
  let gc_device = Mock_gc_dev.make_device () () ~ordinal:0 in
  Mock_gc_slab.alloc_pool gc_device ~pool_id:7 ~size_in_bytes:16 ~alignment:1;
  Verdict.p "gc backend exposes free_pool (not None)" (Option.is_some Mock_gc_slab.free_pool);
  let present_before =
    try
      ignore (Mock_gc_slab.resolve_pool gc_device (loc 7) : int);
      true
    with _ -> false
  in
  Option.iter Mock_gc_slab.free_pool ~f:(fun free -> free gc_device ~pool_id:7);
  let present_after =
    try
      ignore (Mock_gc_slab.resolve_pool gc_device (loc 7) : int);
      true
    with _ -> false
  in
  Verdict.p "gc backend entry present before free" present_before;
  Verdict.p "gc backend entry absent after free" (not present_after)
