(** CPU topology facts for the worker-pool uniformity policy (gh-ocannl-530,
    docs/proposals/gh-ocannl-530-pool-uniformity.md). Facts only — the policy that consumes them
    lives in [Cc_backend]. Every probe degrades to an "unknown" answer ([[]], [`Unknown], a
    fallback count) rather than raising; the consumers treat unknown conservatively. *)

open Base

external native_core_classes_str : unit -> string = "ocannl_core_classes_str"
external effective_cpu_count_stub : unit -> int = "ocannl_effective_cpu_count"
external hypervisor_present_stub : unit -> int = "ocannl_hypervisor_present"
external set_process_affinity_stub : int64 -> int = "ocannl_set_process_affinity"

type core_class = {
  perf_rank : int;
      (** Relative performance of this class among the machine's classes: higher = faster cores.
          Absolute values are platform-native (Windows [EfficiencyClass], Darwin perflevel order
          reversed, list position for Linux) — only the ordering is meaningful. *)
  count : int;  (** Logical CPUs in the class (SMT siblings included). *)
  mask : int64;
      (** Affinity mask over logical CPUs (bit i = CPU i); [0L] when the platform has no affinity
          masks (Darwin) — such classes are informational and cannot be restricted to. *)
}
[@@deriving sexp, compare, equal]

(* --- Pure parsing, exposed for tests. --- *)

(** Parses a sysfs-style CPU list, e.g. ["0-1,10-13,22-23"]. [None] on malformed input. *)
let parse_cpu_list s =
  let parse_range part =
    match String.lsplit2 part ~on:'-' with
    | None -> Option.map (Int.of_string_opt (String.strip part)) ~f:(fun c -> [ c ])
    | Some (lo, hi) -> (
        match (Int.of_string_opt (String.strip lo), Int.of_string_opt (String.strip hi)) with
        | Some lo, Some hi when lo <= hi -> Some (List.range lo (hi + 1))
        | _ -> None)
  in
  let s = String.strip s in
  if String.is_empty s then None
  else
    String.split s ~on:','
    |> List.map ~f:parse_range
    |> Option.all
    |> Option.map ~f:List.concat

(** Affinity mask of a CPU list; [None] when empty or any CPU is outside the 64-bit mask (v1 does
    not cross Windows processor groups, and the policy no-ops on such machines). *)
let mask_of_cpu_list cpus =
  if List.is_empty cpus then None
  else if List.exists cpus ~f:(fun c -> c < 0 || c >= 64) then None
  else
    Some
      (List.fold cpus ~init:0L ~f:(fun acc c -> Int64.(acc lor shift_left 1L c)))

(** Parses the stub's class string ["rank:count:maskhex;..."]. [[]] on any malformed item. *)
let parse_classes_str s =
  let parse_item item =
    match String.split item ~on:':' with
    | [ rank; count; mask ] -> (
        match (Int.of_string_opt rank, Int.of_string_opt count) with
        | Some perf_rank, Some count when count > 0 -> (
            try Some { perf_rank; count; mask = Int64.of_string ("0x" ^ mask) }
            with _ -> None)
        | _ -> None)
    | _ -> None
  in
  if String.is_empty s then []
  else
    match String.split s ~on:';' |> List.map ~f:parse_item |> Option.all with
    | None -> []
    | Some classes ->
        List.sort classes ~compare:(fun a b -> Int.compare b.perf_rank a.perf_rank)

(** Builds perf-ranked classes from (rank, cpu list) pairs; [[]] if any list fails to fit a
    64-bit mask. *)
let classes_of_ranked_cpu_lists ranked =
  ranked
  |> List.map ~f:(fun (perf_rank, cpus) ->
      Option.map (mask_of_cpu_list cpus) ~f:(fun mask ->
          { perf_rank; count = List.length cpus; mask }))
  |> Option.all
  |> Option.value_map ~default:[] ~f:(fun classes ->
      List.sort classes ~compare:(fun a b -> Int.compare b.perf_rank a.perf_rank))

(* --- Platform probes. --- *)

let read_sys_file path =
  try Some (String.strip (Stdio.In_channel.read_all path)) with _ -> None

(* Intel hybrid exposes the two classes directly; ARM big.LITTLE (and DynamIQ) exposes per-CPU
   relative capacity instead, so group CPUs by capacity and rank the groups by it. Any missing or
   unparsable piece yields [[]] (read as: uniform or unknown). *)
let linux_classes () =
  match
    ( read_sys_file "/sys/devices/cpu_core/cpus",
      read_sys_file "/sys/devices/cpu_atom/cpus" )
  with
  | Some core, Some atom -> (
      match (parse_cpu_list core, parse_cpu_list atom) with
      | Some core, Some atom -> classes_of_ranked_cpu_lists [ (1, core); (0, atom) ]
      | _ -> [])
  | _ -> (
      match Option.bind (read_sys_file "/sys/devices/system/cpu/present") ~f:parse_cpu_list with
      | None -> []
      | Some cpus -> (
          let capacities =
            List.map cpus ~f:(fun c ->
                Option.bind
                  (read_sys_file (Printf.sprintf "/sys/devices/system/cpu/cpu%d/cpu_capacity" c))
                  ~f:Int.of_string_opt
                |> Option.map ~f:(fun cap -> (cap, c)))
            |> Option.all
          in
          match capacities with
          | None -> []
          | Some pairs ->
              let groups =
                Hashtbl.of_alist_multi (module Int) pairs
                |> Hashtbl.to_alist
                |> List.sort ~compare:(fun (a, _) (b, _) -> Int.compare a b)
              in
              if List.length groups < 2 then []
              else classes_of_ranked_cpu_lists (List.mapi groups ~f:(fun i (_, cpus) -> (i, cpus)))))

let core_classes =
  let classes =
    lazy
      (let s = native_core_classes_str () in
       if not (String.is_empty s) then parse_classes_str s
       else if Stdlib.Sys.file_exists "/sys/devices/system/cpu" then linux_classes ()
       else [])
  in
  fun () -> Lazy.force classes

let total_logical_count classes = List.sum (module Int) classes ~f:(fun c -> c.count)

(** Logical CPUs available to this process — affinity-respecting, unlike
    [Domain.recommended_domain_count] (affinity-blind on Windows, [_SC_NPROCESSORS_ONLN]-based
    elsewhere). Queried fresh on each call: the answer changes when the process restricts
    itself. *)
let effective_cpu_count () =
  let n = effective_cpu_count_stub () in
  if n > 0 then n else max 1 (Stdlib.Domain.recommended_domain_count ())

(** Whether the process runs as a hypervisor {e guest} (WSL2, Hyper-V VMs, KVM, ...), where
    guest-visible topology is fabricated and must not drive geometry decisions (gh-ocannl-530).
    A native Windows host with the Hyper-V role (or WSL2/VBS) enabled sets the CPUID hypervisor
    bit yet sees real topology, and reads as [`No] here: on Windows a "Microsoft Hv" hypervisor
    is the host platform itself, while Hyper-V guests see fabricated {e uniform} topology, making
    the core-classes check the operative gate for them (see the stub for why the root-partition
    privilege check does not work). [`Unknown] on platforms without a detection path (non-x86
    Linux). *)
let hypervisor_present () =
  match hypervisor_present_stub () with 1 -> `Yes | 0 -> `No | _ -> `Unknown

type pool_decision = {
  pool_restrict : core_class option;
      (** The class to confine the process to; [None] = leave the pool as found. *)
  pool_width : int;  (** Worker-pool width after the decision, feeding the auto chunk count. *)
  pool_tag : string;
      (** Compact pool signature ([w8P], [w24], ...) for the autotune disk-cache key: crowned
          schedules do not transfer across pools (the gh-530 cross-arm replay), so a policy flip
          or a different external pinning must re-tune, not replay. *)
}
[@@deriving sexp_of]

(** The pure pool-uniformity decision (config [cc_pool_core_class]; the applying wrapper lives in
    [Cc_backend], the rationale in docs/proposals/gh-ocannl-530-pool-uniformity.md). [classes] is
    perf-ranked fastest-first; [effective] the process's affinity-respecting logical CPU count.
    Explicit [`Performance]/[`Efficiency] override an external pinning (the user asked for the
    class); [`Auto] respects it (the pinning is itself the user's pool decision). [`Unknown]
    hypervisor state reads as "cannot rule virtualization out" and blocks restriction: the one
    hybrid family it leaves unrestricted in practice is big.LITTLE ARM Linux, which no
    measurement covers yet. *)
let decide_pool_restriction ~openmp ~setting ~classes ~hypervisor ~effective =
  let keep =
    { pool_restrict = None; pool_width = effective; pool_tag = "w" ^ Int.to_string effective }
  in
  let restrict_to (cls : core_class) suffix =
    {
      pool_restrict = Some cls;
      pool_width = cls.count;
      pool_tag = "w" ^ Int.to_string cls.count ^ suffix;
    }
  in
  if not openmp then keep
  else
    match setting with
    | `All -> keep
    | (`Auto | `Performance | `Efficiency) as s ->
        let hybrid_restrictable =
          List.length classes >= 2
          && List.for_all classes ~f:(fun (c : core_class) -> Int64.(c.mask <> 0L))
        in
        let native = match hypervisor with `No -> true | `Yes | `Unknown -> false in
        let externally_pinned = effective < total_logical_count classes in
        if not (hybrid_restrictable && native) then keep
        else if externally_pinned && Poly.equal s `Auto then keep
        else (
          match s with
          | `Auto | `Performance -> restrict_to (List.hd_exn classes) "P"
          | `Efficiency -> restrict_to (List.last_exn classes) "E")

(** Restricts the whole process (all threads and future children) to the CPUs of [mask]. *)
let restrict_process_to_mask mask =
  match set_process_affinity_stub mask with
  | 0 -> Ok ()
  | -1 -> Error "process affinity is not supported on this platform"
  | _ -> Error (Printf.sprintf "setting the process affinity mask 0x%Lx failed" mask)
