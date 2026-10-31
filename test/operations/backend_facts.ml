(* gh-ocannl-1723: what a compiler decision asks of a backend is the backend's stated capability
   facts ([Ir.Backend_intf.backend_facts]), never a test of its name.

   Two relationships, each from where its link cost is paid. [marker_backend_vocabulary] holds the
   facts over every backend without a device (names reach their module's facts; the facts are
   consistent; a non-backend name is refused). This test adds the two that need more:

   - the running backend: the facts a live context reports are the facts its name parses to, and its
   [grid_execution] agrees with the device's own limits, which bound a workgroup's threads exactly
   on a backend that binds hardware axes ([Backend_device_common.hardware_limits]). An agreement
   check only: the facts are never inferred from the limits. - the consumers: holding every other
   fact fixed, varying one fact moves the decision that reads it. A migration that threaded the
   facts through and kept deciding on something else would leave these claims false. *)

open Base
open Verdict.Claims
module BI = Ir.Backend_intf
module Sched = Ir.Schedule
module Tn = Ir.Tnode
module LL = Ir.Low_level

let () =
  let ctx = Context.auto () in
  let name = Context.backend_name ctx in
  (* A backend-uniform golden cannot say which backend ran it (gh-ocannl-622). *)
  Stdio.eprintf "backend_facts: on the %s backend (not part of the golden)\n%!" name;
  let facts = Context.backend_facts ctx in
  p "a live context's facts are the facts its backend's name parses to"
    (BI.equal_backend_facts facts (Context.backend_facts_of_name name));
  let limits = Context.hardware_limits ctx in
  p "the facts bind a hardware grid exactly where the device bounds a workgroup's threads"
    (Bool.equal (BI.binds_hardware_grid facts) (Option.is_some limits.max_threads_per_workgroup))

(* Each fact varied alone, from both a host-pool and a hardware-grid record, so a consumer that read
   another fact in its place -- one that happens to agree on today's backends -- moves where it
   should not, or stays where it should move. *)
let bases = [ BI.cc_facts; BI.cuda_facts; BI.metal_facts ]

let moves what ~vary ~decide =
  p_all what bases ~f:(fun base ->
      let a, b = vary base in
      not (Poly.equal (decide a) (decide b)))

let () =
  moves "grid_execution alone moves the GPU and CPU search-shaping key reads"
    ~vary:(fun f ->
      ({ f with BI.grid_execution = Hardware_grid }, { f with BI.grid_execution = Host_pool }))
    ~decide:(fun facts ->
      ( Sched.search_shaping_key_read ~facts "gpu_schedule_block_size",
        Sched.search_shaping_key_read ~facts "cpu_schedule_min_work",
        Autotune.search_shaping_key_read ~facts "autotune_register_tile_rm_twin" ));
  (* The premise of the device-memory floor (gh-ocannl-1554): a node written, read by another
     statement and not settled on the device, under limits that bound nothing. *)
  let tn =
    Tn.create (Tn.Specified Ir.Ops.single) ~id:960_000_001 ~label:[ "bf_fused" ]
      ~unpadded_dims:(lazy [| 4 |])
      ~padding:(lazy None)
      ()
  in
  let node = LL.get_node (Hashtbl.create (module Tn)) tn in
  node.has_assignment <- true;
  node.read_by_other <- true;
  let plc = Tn.Placements.create () in
  moves "workgroup_shared_placement alone moves whether a node may be kept off the device"
    ~vary:(fun f ->
      ( { f with BI.workgroup_shared_placement = true },
        { f with BI.workgroup_shared_placement = false } ))
    ~decide:(fun facts ->
      Sched.may_keep_off_device ~facts ~limits:BI.no_hardware_limits plc tn node);
  moves "seeds_untuned_default alone moves whether the seeded plan's keys are read"
    ~vary:(fun f ->
      ({ f with BI.seeds_untuned_default = true }, { f with BI.seeds_untuned_default = false }))
    ~decide:(fun facts ->
      ( Autotune.seeded_default_backend facts,
        Autotune.search_shaping_key_read ~facts "gpu_default_sketch_min_reduction" ));
  moves "queued_timing alone moves the queued batch depth cap"
    ~vary:(fun f ->
      ({ f with BI.queued_timing = Queued_deep }, { f with BI.queued_timing = Queued_legacy }))
    ~decide:Autotune.queue_depth_cap_for_backend;
  moves "retries_contended_window alone moves the contended-window retry"
    ~vary:(fun f ->
      ({ f with BI.retries_contended_window = true }, { f with BI.retries_contended_window = false }))
    ~decide:Autotune.retry_contended_window_for_backend;
  moves "host_load_attributable alone moves the dispersion policy"
    ~vary:(fun f ->
      ({ f with BI.host_load_attributable = true }, { f with BI.host_load_attributable = false }))
    ~decide:Autotune.dispersion_policy_for_backend
