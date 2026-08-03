open Base
open Ir
module Tn = Tnode
module Lazy = Utils.Lazy
module H = Hip
open Backend_intf

let _get_local_debug_runtime = Utils.get_local_debug_runtime

[%%global_debug_log_level 0]

(* export OCANNL_LOG_LEVEL_HIP_BACKEND=9 to enable debugging into the log_files/ directory. *)
[%%global_debug_log_level_from_env_var "OCANNL_LOG_LEVEL_HIP_BACKEND"]

let () =
  H.hip_call_hook :=
    Some
      (fun ~message:_message ~status:_status ->
        [%debug_sexp
          [%log5_block
            _message;
            if not @@ H.is_success _status then [%log (_status : H.result)]]])

let _suspended () =
  H.hip_call_hook := Some (fun ~message ~status:_ -> Stdlib.Printf.printf "HIP %s\n" message)

module Backend_buffer = struct
  type buffer_ptr = H.Deviceptr.t

  let sexp_of_buffer_ptr ptr = Sexp.Atom (H.Deviceptr.string_of ptr)
end

module Device_config = struct
  include Backend_buffer

  type dev = {
    dev : H.Device.t;
    primary_context : H.Context.t;
    set_builtins_in : H.Module.t -> unit;
  }
  [@@deriving sexp_of]

  type runner = H.Stream.t [@@deriving sexp_of]
  type event = H.Delimited_event.t [@@deriving sexp_of]

  let name = "hip"
end

module Device_stream = Backend_impl.Device_types_ll (Device_config)
open Device_config

let set_ctx ctx = H.Context.set_current ctx

(* The HIP slab allocator: a private [(device_id, pool_id) -> hipDeviceptr_t] table backing the
   shared {!Backend_intf.Slab_alloc}. *)
module Slab = struct
  open Backend_intf

  type device = Device_stream.device
  type buffer_ptr = H.Deviceptr.t

  (* Requested sizes are tracked alongside the pointers so [get_used_memory] can report the bytes
     OCANNL allocated on the device (gh-ocannl-289, mirroring the CUDA backend): the driver's
     [get_free_and_total_mem] moves in allocation granules, which hides sub-granule effects such as
     the liveness planner's arena savings (gh-ocannl-489) and counts other processes' memory. On an
     APU sharing memory with the display that second effect dominates outright — it is what made
     test/operations/buffer_aliasing report the planner INCREASING the training footprint here
     while decreasing it everywhere else (gh-ocannl-542). *)
  let pools : (int * int, buffer_ptr * int) Hashtbl.Poly.t = Hashtbl.Poly.create ()

  let alloc_pool ?mode:_ (device : device) ~pool_id ~size_in_bytes ~alignment:_ =
    set_ctx device.dev.primary_context;
    let key = (device.device_id, pool_id) in
    (* Free any prior allocation under this key before replacing it, so device memory stays
       equivalent to the pre-refactor path. Unique tnode pool ids never pre-exist; this only fires
       on the reserved merge pool growing in place. *)
    Option.iter (Hashtbl.find pools key) ~f:(fun (ptr, _) -> H.Deviceptr.mem_free ptr);
    let size_in_bytes = max 1 size_in_bytes in
    let ptr = H.Deviceptr.mem_alloc ~size_in_bytes in
    Hashtbl.set pools ~key ~data:(ptr, size_in_bytes)

  let free_pool =
    Some
      (fun (device : device) ~pool_id ->
        let key = (device.device_id, pool_id) in
        Option.iter (Hashtbl.find pools key) ~f:(fun (ptr, _) -> H.Deviceptr.mem_free ptr);
        Hashtbl.remove pools key)

  let resolve_pool (device : device) { pool_id; offset = _ } : buffer_ptr =
    (* Return the slab base. The byte offset is NOT folded into the handle here; callers apply it
       via the hipjit ?offset / ?dst_offset / ?src_offset params or via H.Deviceptr.offset. *)
    fst (Hashtbl.find_exn pools (device.device_id, pool_id))

  let used_memory (device : device) =
    Hashtbl.fold pools ~init:0 ~f:(fun ~key:(dev_id, _) ~data:(_, size) acc ->
        if dev_id = device.device_id then acc + size else acc)

  let memset_zero (device : device) ~pool_id ~offset ~size_in_bytes =
    let base = resolve_pool device { pool_id; offset } in
    if size_in_bytes > 0 then
      H.Stream.memset_d8 ~offset base Unsigned.UChar.zero ~length:size_in_bytes device.runner
end

(* [initialized_devices] never forgets its entries. *)
let initialized_devices = Hash_set.create (module Int)
let initialized = ref false

module Impl : Ir.Backend_impl.Lowered_backend = struct
  include Backend_impl.Device (Device_stream) (Slab)

  (* The concrete [buffer_ptr]/[buffer] + sexps for the impl-facing interface (no longer carried by
     the shared [Device_config_common]). *)
  include Backend_buffer

  let ctx_of (context : context) = context.device.dev.primary_context
  let is_done event = H.Delimited_event.query event
  let will_wait_for context event = H.Delimited_event.wait context.device.runner event
  let sync event = H.Delimited_event.synchronize event
  let all_work device = H.Delimited_event.record device.runner

  (* The inline-test harness ([ppx_inline_test]'s runner, which hosts ppx_expect tests) is detected
     by its command line, the same way [Ppx_inline_test_lib] switches to test mode. *)
  let am_running_inline_tests =
    Array.length Stdlib.Sys.argv > 1 && String.equal Stdlib.Sys.argv.(1) "inline-test-runner"

  (* On Windows, amdhip64_N.dll prints a "HIP Library Path: <dll>" banner to stdout when the runtime
     first initializes (rocclr os_win32.cpp; unconditional — not gated by AMD_LOG_LEVEL), which
     would pollute expect-test and .expected-test outputs. Silence OS-level stdout (fd 1) across the
     first HIP call; the redirect is harmless on platforms without the banner. Inside a ppx_expect
     capture the descriptor games corrupt the harness's bookkeeping (removing its capture file fails
     with a sharing violation), so skip there — the inline-test case is instead handled by the eager
     module-init forcing below, which runs before any capture starts. *)
  let quiet_first_call f =
    if am_running_inline_tests then f ()
    else (
      Stdlib.flush Stdlib.stdout;
      let original = Unix.dup Unix.stdout in
      let devnull =
        Unix.openfile (if Stdlib.Sys.win32 then "NUL" else "/dev/null") [ Unix.O_WRONLY ] 0o666
      in
      Unix.dup2 devnull Unix.stdout;
      Exn.protect ~f ~finally:(fun () ->
          Stdlib.flush Stdlib.stdout;
          Unix.dup2 original Unix.stdout;
          Unix.close original;
          Unix.close devnull))

  (* Driver initialization and device discovery are lazy: the singleton [Impl] module initializes at
     program startup (Backends instantiates it eagerly for nameable types), and hipjit is a depopt
     -- the library being installed does not imply a usable driver/GPU. Forcing here, at first
     device use, keeps CPU-only runs from touching the driver and lets [Context.auto] catch
     unusable-HIP failures per call. *)
  let ensure_initialized =
    lazy
      (if not !initialized then (
         quiet_first_call (fun () -> H.init ());
         initialized := true))

  (* Under the inline-test harness, when the session's backend is [hip], force runtime
     initialization at module init: the "HIP Library Path" banner then goes to the runner's own
     stdout instead of the first [%expect] block (no ppx_expect capture is active yet, so the banner
     cannot corrupt or pollute test output). Failures are swallowed — an unusable driver/GPU
     surfaces at [get_device], where [Context.auto]'s fallback can catch it. *)
  let () =
    if
      am_running_inline_tests
      && String.equal "hip"
           (String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:""))
    then try Lazy.force ensure_initialized with _ -> ()

  let num_devices () =
    Lazy.force ensure_initialized;
    H.Device.get_count ()

  (* [devices] is mutable to support plugging in new devices. *)
  let devices = lazy (ref @@ Array.create ~len:(num_devices ()) None)

  (* Bytes OCANNL has allocated on this device via [Slab], exact rather than the driver's
     granule-quantized [total - free] (gh-ocannl-289). Device-wide across contexts, matching the
     [Context.get_used_memory] contract. *)
  let get_used_memory (device : device) = Slab.used_memory device

  (* The merge buffer is the device's reserved single-tenant pool (id [merge_buffer_pool_id]); grow
     it in place when a larger node arrives ([Slab.alloc_pool] overwrites the reserved entry). *)
  let opt_alloc_merge_buffer ~size_in_bytes (device : device) : unit =
    if device.merge_buffer_capacity < size_in_bytes then (
      Slab.alloc_pool device ~pool_id:merge_buffer_pool_id ~size_in_bytes ~alignment:1;
      device.merge_buffer_capacity <- size_in_bytes);
    device.merge_buffer := Some { pool_id = merge_buffer_pool_id; offset = 0 }

  let%track4_sexp finalize_device (device : device) =
    H.Context.set_current device.dev.primary_context;
    H.Context.synchronize ();
    (* gh-ocannl-344: constants are bump-packed, so several cache entries share one constant pool
       slab; free each distinct [pool_id] exactly once (freeing per-entry would double-free / free a
       sub-region pointer). [Slab.free_pool] frees the slab and drops its table entry. *)
    Hashtbl.data device.constant_buffer_cache
    |> List.map ~f:(fun (loc : Backend_intf.buffer_loc) -> loc.pool_id)
    |> List.dedup_and_sort ~compare:Int.compare
    |> List.iter ~f:(fun pool_id ->
        Option.iter Slab.free_pool ~f:(fun free -> free device ~pool_id))

  (* --- Cooperative tile-MMA (rocWMMA) capability, shared by [hardware_limits] and [mma_syntax].
     Requires BOTH the RDNA3/RDNA3.5+ (gfx11/gfx12) wave32 architecture AND discoverable rocWMMA
     headers. Gating both sites matters: [hardware_limits] keeps autotune/scheduling from selecting
     [Tile_mma] where it cannot work, and [mma_syntax] makes a manual [Sched.tensorize] on an
     unsupported device (CDNA gfx9 wave64, a mixed fleet) or a host without rocWMMA decline to the
     scalar fallback rather than emit an uncompilable kernel. Memoized behind [lazy]: device
     enumeration and filesystem probes must not run at module init. *)
  let all_rdna_wave32 =
    lazy
      (let n = num_devices () in
       n > 0
       && Array.for_all
            (Array.init n ~f:(fun ordinal -> H.Device.get_attributes (H.Device.get ~ordinal)))
            ~f:(fun (a : H.Device.attributes) ->
              (String.is_prefix a.gcn_arch_name ~prefix:"gfx11"
              || String.is_prefix a.gcn_arch_name ~prefix:"gfx12")
              && a.warp_size = 32))

  (* The HIP SDK include dir (no-spaces junction on Windows / HIP_PATH / /opt/rocm), forward-slashed
     for the clang command line. [None] when no SDK is found (the Linux built-in-headers path). *)
  let hip_sdk_include_dir =
    lazy
      (let candidates =
         (match Sys.getenv "LOCALAPPDATA" with Some l -> [ l ^ "/hip_path_link" ] | None -> [])
         @ (match Sys.getenv "HIP_PATH" with Some p -> [ p ] | None -> [])
         @ [ "/opt/rocm" ]
       in
       List.find_map candidates ~f:(fun p ->
           if Stdlib.Sys.file_exists (p ^ "/include/hip/hip_fp16.h") then
             Some (String.map ~f:(fun c -> if Char.(c = '\\') then '/' else c) (p ^ "/include"))
           else None))

  (* A directory directly containing [rocwmma/rocwmma.hpp], if any: [ROCWMMA_PATH] variants, a clone
     under [%LOCALAPPDATA%/rocwmma], or the HIP include tree (rocWMMA installs there on Linux).
     rocWMMA is header-only and is NOT in the ROCm Windows SDK, hence the extra search paths. *)
  let rocwmma_include_dir =
    lazy
      (let candidates =
         (match Sys.getenv "ROCWMMA_PATH" with
           | Some p -> [ p; p ^ "/include"; p ^ "/library/include" ]
           | None -> [])
         @ (match Sys.getenv "LOCALAPPDATA" with
           | Some l -> [ l ^ "/rocwmma/library/include" ]
           | None -> [])
         @ match Lazy.force hip_sdk_include_dir with Some d -> [ d ] | None -> []
       in
       List.find candidates ~f:(fun p -> Stdlib.Sys.file_exists (p ^ "/rocwmma/rocwmma.hpp"))
       |> Option.map ~f:(String.map ~f:(fun c -> if Char.(c = '\\') then '/' else c)))

  let mma_supported () =
    Lazy.force all_rdna_wave32 && Option.is_some (Lazy.force rocwmma_include_dir)

  let%diagn2_sexp hip_to_code ~name hip_src =
    let name_hip = name ^ ".hip" in
    (* Tile-MMA kernels (emitted by [mma_syntax]) need the rocWMMA header and C++17; injected only
       when actually used, so kernels without tensor cores compile exactly as before and do not
       require rocWMMA to be present. Mirrors the CUDA backend's <mma.h> injection. The header's
       location is [rocwmma_include_dir] (the same probe that gates [mma_supported]); since
       [mma_syntax] only emits [rocwmma::] when [mma_supported] holds, a kernel that reaches here
       with rocWMMA in it always has a discoverable header. *)
    let uses_rocwmma = String.is_substring hip_src ~substring:"rocwmma::" in
    let hip_src = if uses_rocwmma then "#include <rocwmma/rocwmma.hpp>\n" ^ hip_src else hip_src in
    if Utils.settings.output_debug_files_in_build_directory then (
      let build_file = Utils.open_build_file ~base_name:name ~extension:".hip" in
      Stdio.Out_channel.output_string build_file.oc hip_src;
      build_file.finalize ());
    [%log "compiling to a code object"];
    let with_debug =
      Utils.settings.output_debug_files_in_build_directory || Utils.settings.log_level > 0
    in
    (* hiprtc targets the architecture of the current default device when no [--offload-arch] is
       given. On Linux hiprtc ships built-in HIP headers; on Windows (observed with ROCm 7.1)
       [#include <hip/hip_fp16.h>] is not found without an include path, so point at the SDK's
       include directory ([hip_sdk_include_dir]: the no-spaces junction created by ocaml-hipjit,
       falling back to HIP_PATH or /opt/rocm). The -I is only added when the directory exists, so
       the Linux built-in-headers path is unaffected. *)
    let hip_include_opt =
      match Lazy.force hip_sdk_include_dir with Some d -> [ "-I" ^ d ] | None -> []
    in
    (* rocWMMA include dir, only for tensor-core kernels ([rocwmma_include_dir] finds the dir
       holding [rocwmma/rocwmma.hpp]). *)
    let rocwmma_include_opt =
      if not uses_rocwmma then []
      else match Lazy.force rocwmma_include_dir with Some d -> [ "-I" ^ d ] | None -> []
    in
    let options =
      hip_include_opt @ rocwmma_include_opt
      @ (if uses_rocwmma then [ "-std=c++17" ] else [])
      @ ("-ffast-math" :: (if Utils.with_runtime_debug () then [ "-g" ] else []))
    in
    let code = Hiprtc.compile_to_code ~hip_src ~name:name_hip ~options ~with_debug in
    if Utils.settings.output_debug_files_in_build_directory then (
      let oc = Stdio.Out_channel.create ~binary:true @@ Utils.build_file @@ name ^ ".hsaco" in
      Stdio.Out_channel.output_string oc @@ Hiprtc.string_from_code code;
      Stdio.Out_channel.flush oc;
      Stdio.Out_channel.close oc;
      let oc = Out_channel.open_text @@ Utils.build_file @@ name ^ ".hip_log" in
      Stdio.Out_channel.output_string oc @@ Option.value ~default:"" (Hiprtc.compilation_log code);
      Stdio.Out_channel.flush oc;
      Stdio.Out_channel.close oc);
    code

  let run_options () =
    (* NOTE: on the AMD platform these are accepted for CUDA-driver-API compatibility but mostly
       ignored by [hipModuleLoadDataEx]. *)
    if Utils.with_runtime_debug () then
      H.Module.[ GENERATE_DEBUG_INFO true; GENERATE_LINE_INFO true ]
    else []

  (* No runtime linking needed since Threefry is included directly in each kernel *)
  let set_builtins_for_device ~primary_context:_ _kernel_module = assert !initialized

  let%track3_sexp get_device ~(ordinal : int) : device =
    let n = num_devices () in
    (* See the corresponding note in [Cuda_backend.get_device]. *)
    if n = 0 then
      raise
      @@ Backend_intf.Backend_unavailable
           { backend = name; detail = "the driver reports no HIP devices" };
    if n <= ordinal then
      invalid_arg [%string "Exec_as_hip.get_device %{ordinal#Int}: not enough devices"];
    let devices = Lazy.force devices in
    (if Array.length !devices <= ordinal then
       let old, len = (!devices, Array.length !devices) in
       devices := Array.init (ordinal + 1) ~f:(fun i -> if i < len then old.(i) else None));
    let default () =
      let dev = H.Device.get ~ordinal in
      let primary_context : H.Context.t = H.Context.get_primary dev in
      let set_builtins_in = set_builtins_for_device ~primary_context in
      let dev = { dev; primary_context; set_builtins_in } in
      set_ctx primary_context;
      if Utils.debug_log_from_routines () && not (Hash_set.mem initialized_devices ordinal) then
        Int.of_string_opt @@ Utils.get_global_arg ~arg_name:"hip_printf_fifo_size" ~default:""
        |> Option.iter ~f:H.Context.(set_limit PRINTF_FIFO_SIZE);
      Hash_set.add initialized_devices ordinal;
      (* With one compute stream per device, the runner (HIP stream) is created with the device. *)
      let hip_stream = H.Stream.create ~non_blocking:true () in
      let result = make_device dev hip_stream ~ordinal in
      Stdlib.Gc.finalise finalize_device result;
      !devices.(ordinal) <- Some result;
      result
    in
    Option.value_or_thunk !devices.(ordinal) ~default

  let _hip_properties =
    let cache =
      let%debug2_sexp f (ordinal : int) =
        let dev = get_device ~ordinal in
        lazy (H.Device.get_attributes dev.dev.dev)
      in
      lazy (Array.init (num_devices ()) ~f)
    in
    let%debug2_sexp get_props (device : device) : H.Device.attributes =
      let cache = Lazy.force cache in
      Lazy.force cache.(device.ordinal)
    in
    get_props

  let await (device : device) : unit =
    set_ctx device.dev.primary_context;
    (* Device-side [printf] is buffered outside the stream. On ROCm, stream synchronization can
       return while the printf FIFO is still being copied to host stdout; use device synchronization
       while routine logging is enabled so callers may safely close or restore stdout afterward. *)
    if Utils.debug_log_from_routines () then H.Context.synchronize ()
    else H.Stream.synchronize device.runner

  let is_idle (device : device) = H.Stream.is_ready device.runner

  (* Transfers take {!Backend_intf.buffer_loc} and resolve to the concrete device pointer here,
     against the device's private pool table. We pass [~length] explicitly to [memcpy_H_to_D] /
     [memcpy_D_to_H]: without it the hipjit impl computes [size_in_bytes = full_size - offset],
     which would reduce the copy to 0 bytes when the tensor is placed at an offset equal to its own
     size (the common bump-packed case). *)
  let from_host ~dst ~dst_loc hosted =
    set_ctx @@ ctx_of dst;
    let base = Slab.resolve_pool dst.device dst_loc in
    let f src =
      let full_bytes = Bigarray.Genarray.size_in_bytes src in
      let elem_bytes = Bigarray.kind_size_in_bytes (Bigarray.Genarray.kind src) in
      H.Stream.memcpy_H_to_D ~length:(full_bytes / elem_bytes) ~dst_offset:dst_loc.offset ~dst:base
        ~src dst.device.runner
    in
    Ndarray.apply { f } hosted

  let to_host ~src ~src_loc hosted =
    set_ctx @@ ctx_of src;
    let base = Slab.resolve_pool src.device src_loc in
    let f dst =
      let full_bytes = Bigarray.Genarray.size_in_bytes dst in
      let elem_bytes = Bigarray.kind_size_in_bytes (Bigarray.Genarray.kind dst) in
      H.Stream.memcpy_D_to_H ~length:(full_bytes / elem_bytes) ~src_offset:src_loc.offset ~dst
        ~src:base src.device.runner
    in
    Ndarray.apply { f } hosted

  let device_to_device tn ~into_merge_buffer ~dst_loc ~dst ~src_loc ~src =
    let dev = dst.device in
    let same_device = dev.ordinal = src.device.ordinal in
    let size_in_bytes = Lazy.force tn.Tn.size_in_bytes in
    let src_base = Slab.resolve_pool src.device src_loc in
    let src_offset = src_loc.offset in
    let memcpy ~dst_base ~dst_offset =
      if same_device then
        H.Stream.memcpy_D_to_D ~size_in_bytes ~dst_offset ~src_offset ~dst:dst_base ~src:src_base
          dst.device.runner
      else
        (* Note: unlike CUDA, HIP identifies peers by device rather than by context. *)
        H.Stream.memcpy_peer ~size_in_bytes ~dst_offset ~src_offset ~dst:dst_base
          ~dst_device:dst.device.dev.dev ~src:src_base ~src_device:src.device.dev.dev
          dst.device.runner
    in
    match (into_merge_buffer, dst_loc) with
    | No, None -> invalid_arg "Hip_backend.device_to_device: missing dst_loc"
    | No, Some dst_loc ->
        set_ctx @@ ctx_of dst;
        let dst_base = Slab.resolve_pool dst.device dst_loc in
        memcpy ~dst_base ~dst_offset:dst_loc.offset
    | Copy, _ ->
        set_ctx @@ ctx_of dst;
        opt_alloc_merge_buffer ~size_in_bytes dst.device;
        let loc = Option.value_exn ~here:[%here] !(dst.device.merge_buffer) in
        let dst_base = Slab.resolve_pool dst.device loc in
        memcpy ~dst_base ~dst_offset:loc.offset

  type code = {
    traced_store : Low_level.traced_store;
    code : Hiprtc.compile_to_code_result;
    kparams : (string * kparam_source) list;
    bindings : Indexing.unit_bindings;
    name : string;
    launch : Low_level.launch_dims;
  }
  [@@deriving sexp_of]

  type code_batch = {
    traced_stores : Low_level.traced_store option array;
    code : Hiprtc.compile_to_code_result;
    bindings : Indexing.unit_bindings;
    kparams_and_names :
      ((string * kparam_source) list * string * Low_level.launch_dims) option array;
  }
  [@@deriving sexp_of]

  module Hip_syntax_config (Input : sig
    val procs : Low_level.optimized array
  end) =
  struct
    include C_syntax.Pure_C_config (struct
      type nonrec buffer_ptr = buffer_ptr

      let procs = Input.procs

      let full_printf_support =
        not @@ Utils.get_global_flag ~default:false ~arg_name:"prefer_backend_uniformity"
    end)

    let ident_blacklist =
      ident_blacklist
      @ [
          (* HIP built-in variables — would shadow per-thread or per-block context *)
          "threadIdx";
          "blockIdx";
          "blockDim";
          "gridDim";
          "warpSize";
        ]

    let main_kernel_prefix = "extern \"C\" __global__"

    (* An all-Serial kernel launches 1x1x1, so no single-thread guard is needed; annotated kernels
       need every thread (axis-types proposal §4). *)
    let kernel_prep_line = ""

    (* Use native types for loop indices and arguments instead of stdint.h types. Signed index
       arithmetic (docs/proposals/signed-index-precision.md). *)
    let loop_index_type = if Utils.settings.large_models then "long long " else "int "
    let arg_int_prefix = if Utils.settings.large_models then "const long long " else "const int "

    (* Hardware axis bindings (docs/proposals/axis-types-for-loops.md §5); the binding site casts
       the unsigned register to the signed [loop_index_type] (values fit by device limits and the
       per-node numel contract). *)
    let hardware_index ~kind ~slot =
      let base = match kind with `Grid -> "blockIdx" | `Workgroup -> "threadIdx" in
      match slot with
      | 0 -> Some (base ^ ".x")
      | 1 -> Some (base ^ ".y")
      | 2 -> Some (base ^ ".z")
      | _ -> None

    let barrier_syntax = Some "__syncthreads();"
    let shared_decl_prefix = Some "__shared__ "
    let restrict_keyword = Some "__restrict__"

    (* Warp-shuffle rendering of [Workgroup_reduce] accumulation loops (gh-ocannl-462):
       [ocannl_shfl_xor] wraps [__shfl_xor] with an explicit width of 32 (builtins_hip.ml). RDNA
       GPUs have 32-wide wavefronts natively; on wave64 (CDNA/GCN) devices the explicit width makes
       the shuffles reduce the same 32-lane groups, so 32 is correct everywhere. *)
    let warp_size = 32

    (* No vectorization pragmas in device code — SIMD-style gains on GPU come from memory
       transactions: eligible [Vectorized] loops render 128-bit packed loads/stores through the
       [__align__(16)] pack structs (gh-ocannl-463), and everything else falls back to plain serial
       loops. Local arrays live in registers/local memory; no alignment attribute needed (packed
       accesses require device-resident nodes). *)
    let vectorize_pragma = []
    let aligned_local_attr = None
    let vector_bytes = 16
    let vector_style = `Packed_struct

    let typ_of_prec = function
      | Ops.Byte_prec _ -> "unsigned char"
      | Ops.Uint16_prec _ -> "unsigned short"
      | Ops.Int32_prec _ -> "int"
      | Ops.Int64_prec _ -> "long long"
      | Ops.Uint4x32_prec _ -> "uint4x32_t"
      | Ops.Half_prec _ -> "__half"
      | Ops.Bfloat16_prec _ -> "__hip_bfloat16" (* HIP bfloat16 type *)
      | Ops.Fp8_prec _ -> "__hip_fp8_e5m2" (* HIP FP8 type (E5M2 format) *)
      | Ops.Single_prec _ -> "float"
      | Ops.Double_prec _ -> "double"
      | Ops.Void_prec -> "void"
      | Ops.Uint32_prec _ -> "unsigned int"
      | Ops.Uint64_prec _ -> "unsigned long long"

    let vec_typ_of_prec ~length prec =
      match (prec, length) with
      | Ops.Single_prec _, 4 -> "float4_t"
      | Ops.Double_prec _, 2 -> "double2_t"
      | Ops.Int32_prec _, 4 -> "int32x4_t"
      | Ops.Int64_prec _, 2 -> "int64x2_t"
      | Ops.Byte_prec _, 16 -> "int8x16_t"
      (* Fp8 needs [__hip_fp8_e5m2] elements: [Set_from_vec] assigns them to the fp8 array cells
         without a cast, and [__hip_fp8_e5m2] has no assignment from integer types. *)
      | Ops.Fp8_prec _, 16 -> "fp8x16_t"
      | Ops.Uint16_prec _, 8 -> "uint16x8_t"
      | Ops.Uint32_prec _, 4 -> "uint32x4_t"
      | Ops.Uint64_prec _, 2 -> "uint64x2_t"
      (* Like fp8, bfloat16 needs [__hip_bfloat16] elements rather than raw [unsigned short] bits:
         [Set_from_vec] assigns them to the array cells without a cast. Mirrors the CUDA backend. *)
      | Ops.Bfloat16_prec _, 8 -> "bfloat16x8_t"
      | Ops.Half_prec _, 8 -> "half8_t"
      | _, 1 -> typ_of_prec prec
      | _ -> invalid_arg "Hip_backend.vec_typ_of_prec: invalid combination"

    (* DRAFT (tensorize-mma T3, HIP counterpart of the CUDA wmma draft): cooperative tile-MMA
       emission for [Low_level.Tile_mma] via rocWMMA -- ROCm's header-compatible analogue of
       nvcuda::wmma, so the fragment/load/mma/store shape mirrors cuda_backend.ml almost verbatim.
       The extent-32 lane loop binds threadIdx.x, so the 32 consecutive .x threads reaching the
       statement form the cooperating RDNA wavefront (wave32); 16x16x16 fragment blocks stay
       resident across the whole [k] extent. Supported combinations on RDNA3 / RDNA3.5 WMMA: f16 x
       f16 -> f32 (flagship), f16 x f16 -> f16, bf16 x bf16 -> f32, bf16 x bf16 -> bf16. RDNA WMMA
       has no f32-input (tf32-like) shape, so uniform f32 stays on the scalar path -- unlike Metal
       [simdgroup_matrix], which does f32. Declines (the barrier-bracketed lane-0 fallback renders
       instead) on: other precision combinations, extents not multiples of 16, leading dimensions
       violating the 16-element-tile stride constraint, and thread-space operands (per-thread stacks
       are not a jointly-owned tile). Also declines (via [mma_supported] in the guard below) when
       the target is not RDNA3+/wave32 or rocWMMA headers are absent: a manual [Sched.tensorize]
       reaches this hook even where [hardware_limits.mma] is [None], so the capability check cannot
       live in [hardware_limits] alone. Verified on gfx1151 (Radeon 8060S, RDNA3.5) under hiprtc via
       schedule_mma_matmul: the f16 -> f16 combination compiles and executes and matches the serial
       twin bitwise; the bf16 and f16 -> f32 combinations take the same rocWMMA template path,
       differing only in fragment element type. rocWMMA (header-only) is cloned under
       %LOCALAPPDATA%/rocwmma since it is not in the ROCm 7.1 Windows SDK. [hip_to_code] injects the
       header and -std=c++17 only when a kernel actually uses it, so non-tensor-core kernels are
       unaffected and do not require rocWMMA to be present. *)
    let mma_syntax =
      Some
        (fun ~d_prec
          ~a_prec
          ~b_prec
          ~ta
          ~tb
          ~m
          ~n
          ~k
          ~d:(d_ptr, ldd, d_space)
          ~a:(a_ptr, lda, a_space)
          ~b:(b_ptr, ldb, b_space)
        ->
          let tile = 16 in
          (* (a/b fragment element type, accumulator fragment element type, ld multiple for a/b, ld
             multiple for d). rocWMMA element types [rocwmma::float16_t] / [rocwmma::bfloat16_t] /
             [float] need not be textually identical to the node's own C type ([__half] /
             [__hip_bfloat16]), so the operand pointers are [reinterpret_cast] to them below. *)
          let combo =
            match (a_prec, b_prec, d_prec) with
            | Ops.Half_prec _, Ops.Half_prec _, Ops.Single_prec _ ->
                Some ("rocwmma::float16_t", "float", 8, 4)
            | Ops.Half_prec _, Ops.Half_prec _, Ops.Half_prec _ ->
                Some ("rocwmma::float16_t", "rocwmma::float16_t", 8, 8)
            | Ops.Bfloat16_prec _, Ops.Bfloat16_prec _, Ops.Single_prec _ ->
                Some ("rocwmma::bfloat16_t", "float", 8, 4)
            | Ops.Bfloat16_prec _, Ops.Bfloat16_prec _, Ops.Bfloat16_prec _ ->
                Some ("rocwmma::bfloat16_t", "rocwmma::bfloat16_t", 8, 8)
            | _ -> None
          in
          let loadable = function
            | `Device | `Shared -> true (* generic-address loads cover both *)
            | `Thread | `Fragment _ -> false
          in
          match d_space with
          | `Fragment fragment -> (
              let (* gh-ocannl-480 (cross-[k_o] accumulator residency): the accumulator fragment
                     array [fragment] was declared and loaded once by [mma_fragment_syntax]. Here
                     each [Tile_mma] at a serial [k_o] emits update-only mma steps into it -- no
                     per-[k_o] load or store of [d]. The trailing barrier keeps the next k-block's
                     cooperative staging from overwriting the shared tiles still being read. The
                     acceptance guard matches [mma_fragment_syntax]'s (both see the same
                     [lda]/[ldb]), so whenever the fragment scope accepts this branch does too. *)
                open
                PPrint
              in
              match combo with
              | Some (ab_typ, _acc_typ, ab_ld_mult, _d_ld_mult)
                when mma_supported ()
                     && m % tile = 0
                     && n % tile = 0
                     && k % tile = 0
                     && lda % ab_ld_mult = 0
                     && ldb % ab_ld_mult = 0
                     && loadable a_space && loadable b_space ->
                  let mt = m / tile and nt = n / tile and kt = k / tile in
                  let frag kind typ layout =
                    Printf.sprintf "rocwmma::fragment<rocwmma::%s, %d, %d, %d, %s%s>" kind tile tile
                      tile typ
                      (match layout with Some l -> ", rocwmma::" ^ l | None -> "")
                  in
                  let ptr_decl name typ ptr =
                    string (Printf.sprintf "%s *%s = reinterpret_cast<%s *>(" typ name typ)
                    ^^ ptr ^^ string ");"
                  in
                  let a_layout = if ta then "col_major" else "row_major" in
                  let b_layout = if tb then "col_major" else "row_major" in
                  let barrier = "__syncthreads();" in
                  let body_lines =
                    [
                      Printf.sprintf "for (int __ki = 0; __ki < %d; ++__ki) {" kt;
                      Printf.sprintf "  %s __mma_bf[%d];"
                        (frag "matrix_b" ab_typ (Some b_layout))
                        nt;
                      Printf.sprintf "  for (int __ni = 0; __ni < %d; ++__ni) {" nt;
                      (if tb then
                         Printf.sprintf
                           "    rocwmma::load_matrix_sync(__mma_bf[__ni], __mma_bp + __ni * %d * \
                            %d + __ki * %d, %d);"
                           tile ldb tile ldb
                       else
                         Printf.sprintf
                           "    rocwmma::load_matrix_sync(__mma_bf[__ni], __mma_bp + __ki * %d * \
                            %d + __ni * %d, %d);"
                           tile ldb tile ldb);
                      "  }";
                      Printf.sprintf "  for (int __mi = 0; __mi < %d; ++__mi) {" mt;
                      Printf.sprintf "    %s __mma_af;" (frag "matrix_a" ab_typ (Some a_layout));
                      (if ta then
                         Printf.sprintf
                           "    rocwmma::load_matrix_sync(__mma_af, __mma_ap + __ki * %d * %d + \
                            __mi * %d, %d);"
                           tile lda tile lda
                       else
                         Printf.sprintf
                           "    rocwmma::load_matrix_sync(__mma_af, __mma_ap + __mi * %d * %d + \
                            __ki * %d, %d);"
                           tile lda tile lda);
                      Printf.sprintf "    for (int __ni = 0; __ni < %d; ++__ni) {" nt;
                      Printf.sprintf
                        "      rocwmma::mma_sync(%s[__mi][__ni], __mma_af, __mma_bf[__ni], \
                         %s[__mi][__ni]);"
                        fragment fragment;
                      "    }";
                      "  }";
                      "}";
                      barrier;
                    ]
                  in
                  let body =
                    ptr_decl "__mma_ap" ("const " ^ ab_typ) a_ptr
                    ^^ hardline
                    ^^ ptr_decl "__mma_bp" ("const " ^ ab_typ) b_ptr
                    ^^ hardline
                    ^^ separate_map hardline string body_lines
                  in
                  Some
                    (group
                       (string
                          (Printf.sprintf "{ /* tile_mma fragment update %dx%dx%d (rocwmma) */" m n
                             k)
                       ^^ nest 2 (hardline ^^ body)
                       ^^ hardline ^^ rbrace))
              | _ -> None)
          | `Device | `Shared | `Thread -> (
              match combo with
              | Some (ab_typ, acc_typ, ab_ld_mult, d_ld_mult)
                when mma_supported ()
                     && m % tile = 0
                     && n % tile = 0
                     && k % tile = 0
                     && lda % ab_ld_mult = 0
                     && ldb % ab_ld_mult = 0
                     && ldd % d_ld_mult = 0
                     && loadable d_space && loadable a_space && loadable b_space ->
                  let open PPrint in
                  let mt = m / tile and nt = n / tile and kt = k / tile in
                  let frag kind typ layout =
                    Printf.sprintf "rocwmma::fragment<rocwmma::%s, %d, %d, %d, %s%s>" kind tile tile
                      tile typ
                      (match layout with Some l -> ", rocwmma::" ^ l | None -> "")
                  in
                  (* [reinterpret_cast] bridges the node's C element type to the rocWMMA fragment
                     type. *)
                  let ptr_decl name typ ptr =
                    string (Printf.sprintf "%s *%s = reinterpret_cast<%s *>(" typ name typ)
                    ^^ ptr ^^ string ");"
                  in
                  let a_layout = if ta then "col_major" else "row_major" in
                  let b_layout = if tb then "col_major" else "row_major" in
                  let barrier = "__syncthreads();" in
                  let body_lines =
                    [
                      barrier;
                      Printf.sprintf "%s __mma_acc[%d][%d];" (frag "accumulator" acc_typ None) mt nt;
                      Printf.sprintf "for (int __mi = 0; __mi < %d; ++__mi) {" mt;
                      Printf.sprintf "  for (int __ni = 0; __ni < %d; ++__ni) {" nt;
                      Printf.sprintf
                        "    rocwmma::load_matrix_sync(__mma_acc[__mi][__ni], __mma_dp + __mi * %d \
                         * %d + __ni * %d, %d, rocwmma::mem_row_major);"
                        tile ldd tile ldd;
                      "  }";
                      "}";
                      Printf.sprintf "for (int __ki = 0; __ki < %d; ++__ki) {" kt;
                      Printf.sprintf "  %s __mma_bf[%d];"
                        (frag "matrix_b" ab_typ (Some b_layout))
                        nt;
                      Printf.sprintf "  for (int __ni = 0; __ni < %d; ++__ni) {" nt;
                      (* Transposed storage ([tb]): the stored matrix is the role's transpose --
                         index it at (col, row) and declare the fragment [col_major]; the leading
                         dimension stays the operand's own. Same for [ta] below. *)
                      (if tb then
                         Printf.sprintf
                           "    rocwmma::load_matrix_sync(__mma_bf[__ni], __mma_bp + __ni * %d * \
                            %d + __ki * %d, %d);"
                           tile ldb tile ldb
                       else
                         Printf.sprintf
                           "    rocwmma::load_matrix_sync(__mma_bf[__ni], __mma_bp + __ki * %d * \
                            %d + __ni * %d, %d);"
                           tile ldb tile ldb);
                      "  }";
                      Printf.sprintf "  for (int __mi = 0; __mi < %d; ++__mi) {" mt;
                      Printf.sprintf "    %s __mma_af;" (frag "matrix_a" ab_typ (Some a_layout));
                      (if ta then
                         Printf.sprintf
                           "    rocwmma::load_matrix_sync(__mma_af, __mma_ap + __ki * %d * %d + \
                            __mi * %d, %d);"
                           tile lda tile lda
                       else
                         Printf.sprintf
                           "    rocwmma::load_matrix_sync(__mma_af, __mma_ap + __mi * %d * %d + \
                            __ki * %d, %d);"
                           tile lda tile lda);
                      Printf.sprintf "    for (int __ni = 0; __ni < %d; ++__ni) {" nt;
                      "      rocwmma::mma_sync(__mma_acc[__mi][__ni], __mma_af, __mma_bf[__ni], \
                       __mma_acc[__mi][__ni]);";
                      "    }";
                      "  }";
                      "}";
                      Printf.sprintf "for (int __mi = 0; __mi < %d; ++__mi) {" mt;
                      Printf.sprintf "  for (int __ni = 0; __ni < %d; ++__ni) {" nt;
                      Printf.sprintf
                        "    rocwmma::store_matrix_sync(__mma_dp + __mi * %d * %d + __ni * %d, \
                         __mma_acc[__mi][__ni], %d, rocwmma::mem_row_major);"
                        tile ldd tile ldd;
                      "  }";
                      "}";
                      barrier;
                    ]
                  in
                  let body =
                    ptr_decl "__mma_dp" acc_typ d_ptr ^^ hardline
                    ^^ ptr_decl "__mma_ap" ("const " ^ ab_typ) a_ptr
                    ^^ hardline
                    ^^ ptr_decl "__mma_bp" ("const " ^ ab_typ) b_ptr
                    ^^ hardline
                    ^^ separate_map hardline string body_lines
                  in
                  Some
                    (group
                       (string (Printf.sprintf "{ /* tile_mma %dx%dx%d (rocwmma) */" m n k)
                       ^^ nest 2 (hardline ^^ body)
                       ^^ hardline ^^ rbrace))
              | _ -> None))

    (* Cross-[k_o] accumulator residency (gh-ocannl-480): the marked local accumulator tile becomes
       a persistent rocWMMA accumulator-fragment array whose load/store bracket the whole serial
       reduction. Loaded once from [target] before the [k_o] loop, updated in place by the nested
       [Tile_mma]s (which see [`Fragment fragment] and take the update-only branch of [mma_syntax]),
       stored once after. Mirrors the Metal [simdgroup_matrix] rendering; the guard matches the
       [`Fragment] branch of [mma_syntax] so both accept together. *)
    let mma_fragment_syntax =
      Some
        (fun ~d_prec
          ~a_prec
          ~b_prec
          ~m
          ~n
          ~k
          ~fragment
          ~target:(d_ptr, ldd, d_space)
          ~a:(_, lda, a_space)
          ~b:(_, ldb, b_space)
          ~body
        ->
          let tile = 16 in
          let combo =
            match (a_prec, b_prec, d_prec) with
            | Ops.Half_prec _, Ops.Half_prec _, Ops.Single_prec _ ->
                Some ("rocwmma::float16_t", "float", 8, 4)
            | Ops.Half_prec _, Ops.Half_prec _, Ops.Half_prec _ ->
                Some ("rocwmma::float16_t", "rocwmma::float16_t", 8, 8)
            | Ops.Bfloat16_prec _, Ops.Bfloat16_prec _, Ops.Single_prec _ ->
                Some ("rocwmma::bfloat16_t", "float", 8, 4)
            | Ops.Bfloat16_prec _, Ops.Bfloat16_prec _, Ops.Bfloat16_prec _ ->
                Some ("rocwmma::bfloat16_t", "rocwmma::bfloat16_t", 8, 8)
            | _ -> None
          in
          let loadable = function `Device | `Shared -> true | `Thread | `Fragment _ -> false in
          match combo with
          | Some (_ab_typ, acc_typ, ab_ld_mult, d_ld_mult)
            when mma_supported ()
                 && m % tile = 0
                 && n % tile = 0
                 && k % tile = 0
                 && lda % ab_ld_mult = 0
                 && ldb % ab_ld_mult = 0
                 && ldd % d_ld_mult = 0
                 && loadable d_space && loadable a_space && loadable b_space ->
              let open PPrint in
              let mt = m / tile and nt = n / tile in
              let frag kind typ layout =
                Printf.sprintf "rocwmma::fragment<rocwmma::%s, %d, %d, %d, %s%s>" kind tile tile
                  tile typ
                  (match layout with Some l -> ", rocwmma::" ^ l | None -> "")
              in
              let ptr_decl name typ ptr =
                string (Printf.sprintf "%s *%s = reinterpret_cast<%s *>(" typ name typ)
                ^^ ptr ^^ string ");"
              in
              let barrier = "__syncthreads();" in
              let lines_before =
                [
                  barrier;
                  Printf.sprintf "%s %s[%d][%d];" (frag "accumulator" acc_typ None) fragment mt nt;
                  Printf.sprintf "for (int __mi = 0; __mi < %d; ++__mi) {" mt;
                  Printf.sprintf "  for (int __ni = 0; __ni < %d; ++__ni) {" nt;
                  Printf.sprintf
                    "    rocwmma::load_matrix_sync(%s[__mi][__ni], __mma_dp + __mi * %d * %d + \
                     __ni * %d, %d, rocwmma::mem_row_major);"
                    fragment tile ldd tile ldd;
                  "  }";
                  "}";
                  "/* rocwmma fragment reduction body begins */";
                ]
              in
              let lines_after =
                [
                  "/* rocwmma fragment reduction body ends */";
                  Printf.sprintf "for (int __mi = 0; __mi < %d; ++__mi) {" mt;
                  Printf.sprintf "  for (int __ni = 0; __ni < %d; ++__ni) {" nt;
                  Printf.sprintf
                    "    rocwmma::store_matrix_sync(__mma_dp + __mi * %d * %d + __ni * %d, \
                     %s[__mi][__ni], %d, rocwmma::mem_row_major);"
                    tile ldd tile fragment ldd;
                  "  }";
                  "}";
                  barrier;
                ]
              in
              let d_decl = ptr_decl "__mma_dp" acc_typ d_ptr in
              Some
                (group
                   (string (Printf.sprintf "{ /* rocwmma fragment %dx%d across k_o */" m n)
                   ^^ nest 2
                        (hardline ^^ d_decl ^^ hardline
                        ^^ separate_map hardline string lines_before
                        ^^ hardline ^^ body () ^^ hardline
                        ^^ separate_map hardline string lines_after)
                   ^^ hardline ^^ rbrace))
          | _ -> None)

    let rec binop_syntax prec v =
      (* The match stays exhaustive over (op, prec) -- that is what catches a newly added operator
         here -- but arms whose spelling is plain C delegate to {!C_syntax.default_binop_syntax}
         rather than restating the token. *)
      let open PPrint in
      let f op_str v1 v2 =
        group
          (parens (v1 ^^ string (" " ^ op_str) ^^ ifflat (space ^^ v2) (nest 2 (break 1 ^^ v2))))
      in
      let func fn v1 v2 =
        group (string fn ^^ parens (v1 ^^ comma ^^ ifflat (space ^^ v2) (nest 2 (break 1 ^^ v2))))
      in
      match (v, prec) with
      | Ops.Arg1, _ -> invalid_arg "Hip_backend.binop_syntax: Arg1 is not an operator"
      | Arg2, _ -> invalid_arg "Hip_backend.binop_syntax: Arg2 is not an operator"
      | _, Ops.Void_prec -> invalid_arg "Hip_backend.binop_syntax: Void precision"
      (* The RNG ops call the same builtins under the same precision contract on every C-family
         backend, so they render through the shared helper. Must precede the fp8 bridge: the
         Threefry errors should name the actual target precision, and the lane conversion's builtin
         already yields the target precision. *)
      | ((Threefry4x32_crypto | Threefry4x32_light | Uint4x32_to_prec_uniform_lane) as op), _ ->
          C_syntax.rng_binop_syntax ~backend:"HIP" ~call:func prec op
      | _, Fp8_prec _ ->
          (* __hip_fp8_e5m2 defines no arithmetic operators, and its implicit conversion operators
             (float, double, int, char, ... in amd_hip_fp8.h) make the built-in operators ambiguous,
             so bridge fp8 math through float, mirroring the CC backend's fp8 handling. *)
          fun v1 v2 ->
            let fl v = string "(float)" ^^ parens v in
            group (string "(__hip_fp8_e5m2)" ^^ parens (binop_syntax Ops.single v (fl v1) (fl v2)))
      | Add, Half_prec _ -> func "__hadd"
      | Sub, Half_prec _ -> func "__hsub"
      | Mul, Half_prec _ -> func "__hmul"
      | Div, Half_prec _ -> func "__hdiv"
      | Add, _ -> f "+"
      | Sub, _ -> f "-"
      | Mul, _ -> f "*"
      | Div, _ -> f "/"
      | ToPowOf, Double_prec _ -> func "pow"
      | ToPowOf, Single_prec _ -> func "powf"
      | ToPowOf, Half_prec _ ->
          fun v1 v2 ->
            group
              (string "hexp2(hlog2(" ^^ v1 ^^ string "),"
              ^^ ifflat (space ^^ v2) (nest 2 (break 1 ^^ v2))
              ^^ string ")")
      | ToPowOf, (Byte_prec _ | Uint16_prec _ | Int32_prec _ | Int64_prec _ | Uint4x32_prec _) ->
          invalid_arg "Hip_backend.binop_syntax: ToPowOf not supported for integer precisions"
      | ToPowOf, Bfloat16_prec _ ->
          fun v1 v2 ->
            group
              (string "__float2bfloat16(powf(__bfloat162float("
              ^^ v1 ^^ string "), __bfloat162float(" ^^ v2 ^^ string ")))")
      | Relu_gate, (Byte_prec _ | Uint16_prec _ | Int32_prec _ | Int64_prec _) ->
          fun v1 v2 ->
            group
              (parens
                 (group (parens (v1 ^^ string " > 0"))
                 ^^ ifflat
                      (space ^^ string "?" ^^ space ^^ v2 ^^ space ^^ string ":" ^^ space
                     ^^ string "0")
                      (nest 2
                         (break 1 ^^ string "?" ^^ space ^^ v2 ^^ break 1 ^^ string ":" ^^ space
                        ^^ string "0"))))
      | Relu_gate, Bfloat16_prec _ ->
          fun v1 v2 ->
            group
              (parens
                 (group (parens (string "__bfloat162float(" ^^ v1 ^^ string ") > 0.0f"))
                 ^^ ifflat
                      (space ^^ string "?" ^^ space ^^ v2 ^^ space ^^ string ":" ^^ space
                     ^^ string "__float2bfloat16(0.0f)")
                      (nest 2
                         (break 1 ^^ string "?" ^^ space ^^ v2 ^^ break 1 ^^ string ":" ^^ space
                        ^^ string "__float2bfloat16(0.0f)"))))
      | Relu_gate, Half_prec _ ->
          (* HIP's clang has no [0.0h] half literal; compare via [__hgt] against a bitcast zero. *)
          fun v1 v2 ->
            group
              (parens
                 (group
                    (parens
                       (string "__hgt(" ^^ v1
                       ^^ string ", __ushort_as_half((unsigned short)0x0000U))"))
                 ^^ ifflat
                      (space ^^ string "?" ^^ space ^^ v2 ^^ space ^^ string ":" ^^ space
                      ^^ string "__ushort_as_half((unsigned short)0x0000U)")
                      (nest 2
                         (break 1 ^^ string "?" ^^ space ^^ v2 ^^ break 1 ^^ string ":" ^^ space
                         ^^ string "__ushort_as_half((unsigned short)0x0000U)"))))
      | Relu_gate, Single_prec _ ->
          fun v1 v2 ->
            group
              (parens
                 (group (parens (v1 ^^ string " > 0.0f"))
                 ^^ ifflat
                      (space ^^ string "?" ^^ space ^^ v2 ^^ space ^^ string ":" ^^ space
                     ^^ string "0.0f")
                      (nest 2
                         (break 1 ^^ string "?" ^^ space ^^ v2 ^^ break 1 ^^ string ":" ^^ space
                        ^^ string "0.0f"))))
      | Relu_gate, Double_prec _ ->
          fun v1 v2 ->
            group
              (parens
                 (group (parens (v1 ^^ string " > 0.0"))
                 ^^ ifflat
                      (space ^^ string "?" ^^ space ^^ v2 ^^ space ^^ string ":" ^^ space
                     ^^ string "0.0")
                      (nest 2
                         (break 1 ^^ string "?" ^^ space ^^ v2 ^^ break 1 ^^ string ":" ^^ space
                        ^^ string "0.0"))))
      | Relu_gate, Uint4x32_prec _ ->
          fun v1 v2 ->
            group
              (parens
                 (group (parens (v1 ^^ string " > 0"))
                 ^^ ifflat
                      (space ^^ string "?" ^^ space ^^ v2 ^^ space ^^ string ":" ^^ space
                     ^^ string "0")
                      (nest 2
                         (break 1 ^^ string "?" ^^ space ^^ v2 ^^ break 1 ^^ string ":" ^^ space
                        ^^ string "0"))))
      | Satur01_gate, Byte_prec _ ->
          fun v1 v2 ->
            group
              (parens
                 (group
                    (parens
                       (string "(float)" ^^ v1 ^^ string " > 0.0f && (float)" ^^ v1
                      ^^ string " < 1.0f"))
                 ^^ ifflat
                      (space ^^ string "?" ^^ space ^^ v2 ^^ space ^^ string ":" ^^ space
                     ^^ string "(unsigned char)0")
                      (nest 2
                         (break 1 ^^ string "?" ^^ space ^^ v2 ^^ break 1 ^^ string ":" ^^ space
                        ^^ string "(unsigned char)0"))))
      | Satur01_gate, Half_prec _ ->
          fun v1 v2 ->
            group
              (parens
                 (group
                    (parens
                       (string "__hgt(" ^^ v1 ^^ comma
                       ^^ string " __ushort_as_half((unsigned short)0x0000U)) && __hlt("
                       ^^ v1 ^^ comma
                       ^^ string " __ushort_as_half((unsigned short)0x3C00U))"))
                 ^^ ifflat
                      (space ^^ string "?" ^^ space ^^ v2 ^^ space ^^ string ":" ^^ space
                      ^^ string "__ushort_as_half((unsigned short)0x0000U)")
                      (nest 2
                         (break 1 ^^ string "?" ^^ space ^^ v2 ^^ break 1 ^^ string ":" ^^ space
                         ^^ string "__ushort_as_half((unsigned short)0x0000U)"))))
      | Satur01_gate, Single_prec _ ->
          fun v1 v2 ->
            group
              (parens
                 (group (parens (v1 ^^ string " > 0.0f && " ^^ v1 ^^ string " < 1.0f"))
                 ^^ ifflat
                      (space ^^ string "?" ^^ space ^^ v2 ^^ space ^^ string ":" ^^ space
                     ^^ string "0.0f")
                      (nest 2
                         (break 1 ^^ string "?" ^^ space ^^ v2 ^^ break 1 ^^ string ":" ^^ space
                        ^^ string "0.0f"))))
      | Satur01_gate, Double_prec _ ->
          fun v1 v2 ->
            group
              (parens
                 (group (parens (v1 ^^ string " > 0.0 && " ^^ v1 ^^ string " < 1.0"))
                 ^^ ifflat
                      (space ^^ string "?" ^^ space ^^ v2 ^^ space ^^ string ":" ^^ space
                     ^^ string "0.0")
                      (nest 2
                         (break 1 ^^ string "?" ^^ space ^^ v2 ^^ break 1 ^^ string ":" ^^ space
                        ^^ string "0.0"))))
      | Satur01_gate, Uint16_prec _ ->
          fun v1 v2 ->
            group
              (parens
                 (group
                    (parens
                       (string "(float)" ^^ v1 ^^ string " > 0.0f && (float)" ^^ v1
                      ^^ string " < 1.0f"))
                 ^^ ifflat
                      (space ^^ string "?" ^^ space ^^ v2 ^^ space ^^ string ":" ^^ space
                     ^^ string "(unsigned short)0")
                      (nest 2
                         (break 1 ^^ string "?" ^^ space ^^ v2 ^^ break 1 ^^ string ":" ^^ space
                        ^^ string "(unsigned short)0"))))
      | Satur01_gate, Int32_prec _ ->
          fun v1 v2 ->
            group
              (parens
                 (group
                    (parens
                       (string "(float)" ^^ v1 ^^ string " > 0.0f && (float)" ^^ v1
                      ^^ string " < 1.0f"))
                 ^^ ifflat
                      (space ^^ string "?" ^^ space ^^ v2 ^^ space ^^ string ":" ^^ space
                     ^^ string "0")
                      (nest 2
                         (break 1 ^^ string "?" ^^ space ^^ v2 ^^ break 1 ^^ string ":" ^^ space
                        ^^ string "0"))))
      | Satur01_gate, Int64_prec _ ->
          fun v1 v2 ->
            group
              (parens
                 (group
                    (parens
                       (string "(double)" ^^ v1 ^^ string " > 0.0 && (double)" ^^ v1
                      ^^ string " < 1.0"))
                 ^^ ifflat
                      (space ^^ string "?" ^^ space ^^ v2 ^^ space ^^ string ":" ^^ space
                     ^^ string "0LL")
                      (nest 2
                         (break 1 ^^ string "?" ^^ space ^^ v2 ^^ break 1 ^^ string ":" ^^ space
                        ^^ string "0LL"))))
      | Satur01_gate, Uint4x32_prec _ ->
          fun v1 v2 ->
            group
              (parens
                 (group
                    (parens
                       (string "(float)" ^^ v1 ^^ string " > 0.0f && (float)" ^^ v1
                      ^^ string " < 1.0f"))
                 ^^ ifflat
                      (space ^^ string "?" ^^ space ^^ v2 ^^ space ^^ string ":" ^^ space
                     ^^ string "0u")
                      (nest 2
                         (break 1 ^^ string "?" ^^ space ^^ v2 ^^ break 1 ^^ string ":" ^^ space
                        ^^ string "0u"))))
      | Satur01_gate, Bfloat16_prec _ ->
          fun v1 v2 ->
            group
              (parens
                 (group
                    (parens
                       (string "__bfloat162float(" ^^ v1
                       ^^ string ") > 0.0f && __bfloat162float("
                       ^^ v1 ^^ string ") < 1.0f"))
                 ^^ ifflat
                      (space ^^ string "?" ^^ space ^^ v2 ^^ space ^^ string ":" ^^ space
                     ^^ string "__float2bfloat16(0.0f)")
                      (nest 2
                         (break 1 ^^ string "?" ^^ space ^^ v2 ^^ break 1 ^^ string ":" ^^ space
                        ^^ string "__float2bfloat16(0.0f)"))))
      | Max, Byte_prec _ -> func "max"
      | Max, Half_prec _ -> func "__hmax"
      | Max, Double_prec _ -> func "fmax"
      | Max, Single_prec _ -> func "fmaxf"
      | Max, Uint16_prec _ -> func "max"
      | Max, Int32_prec _ -> func "max"
      | Max, Int64_prec _ -> func "max"
      | Max, Uint4x32_prec _ -> func "max"
      | Max, Bfloat16_prec _ -> func "__hmax"
      | Min, Byte_prec _ -> func "min"
      | Min, Half_prec _ -> func "__hmin"
      | Min, Double_prec _ -> func "fmin"
      | Min, Single_prec _ -> func "fminf"
      | Min, Uint16_prec _ -> func "min"
      | Min, Int32_prec _ -> func "min"
      | Min, Int64_prec _ -> func "min"
      | Min, Uint4x32_prec _ -> func "min"
      | Min, Bfloat16_prec _ -> func "__hmin"
      | ( Mod,
          ( Byte_prec _ | Uint16_prec _ | Int32_prec _ | Uint32_prec _ | Int64_prec _
          | Uint64_prec _ ) ) ->
          f "%"
      (* Like the libm calls in [unop_syntax]: [fmod] on bfloat16 operands returns float, which
         only fails once the placement inlines it into a bfloat16 binop (gh-ocannl-549). *)
      | Mod, Bfloat16_prec _ ->
          fun v1 v2 ->
            group
              (string "__float2bfloat16(fmodf(__bfloat162float(" ^^ v1
              ^^ string "), __bfloat162float(" ^^ v2 ^^ string ")))")
      | Mod, _ -> func "fmod"
      (* Comparisons and logical connectives are precision-independent and spelled the same in HIP
         C++ as in C, so they render through the shared default -- fp8 already bridged above. The
         constructors stay listed to keep the match exhaustiveness-checked. *)
      | ((Cmplt | Cmple | Cmpne | Cmpeq | Or | And) as op), _ ->
          C_syntax.default_binop_syntax prec op
      | ToPowOf, (Uint32_prec _ | Uint64_prec _) ->
          invalid_arg "Hip_backend.binop_syntax: ToPowOf not supported for integer precisions"
      | Relu_gate, Uint32_prec _ ->
          fun v1 v2 ->
            group
              (parens
                 (group (parens (v1 ^^ string " > 0u"))
                 ^^ ifflat
                      (space ^^ string "?" ^^ space ^^ v2 ^^ space ^^ string ":" ^^ space
                     ^^ string "0u")
                      (nest 2
                         (break 1 ^^ string "?" ^^ space ^^ v2 ^^ break 1 ^^ string ":" ^^ space
                        ^^ string "0u"))))
      | Relu_gate, Uint64_prec _ ->
          fun v1 v2 ->
            group
              (parens
                 (group (parens (v1 ^^ string " > 0ULL"))
                 ^^ ifflat
                      (space ^^ string "?" ^^ space ^^ v2 ^^ space ^^ string ":" ^^ space
                     ^^ string "0ULL")
                      (nest 2
                         (break 1 ^^ string "?" ^^ space ^^ v2 ^^ break 1 ^^ string ":" ^^ space
                        ^^ string "0ULL"))))
      | Satur01_gate, Uint32_prec _ ->
          fun v1 v2 ->
            group
              (parens
                 (group (parens (v1 ^^ string " > 0u && " ^^ v1 ^^ string " < 1u"))
                 ^^ ifflat
                      (space ^^ string "?" ^^ space ^^ v2 ^^ space ^^ string ":" ^^ space
                     ^^ string "0u")
                      (nest 2
                         (break 1 ^^ string "?" ^^ space ^^ v2 ^^ break 1 ^^ string ":" ^^ space
                        ^^ string "0u"))))
      | Satur01_gate, Uint64_prec _ ->
          fun v1 v2 ->
            group
              (parens
                 (group (parens (v1 ^^ string " > 0ULL && " ^^ v1 ^^ string " < 1ULL"))
                 ^^ ifflat
                      (space ^^ string "?" ^^ space ^^ v2 ^^ space ^^ string ":" ^^ space
                     ^^ string "0ULL")
                      (nest 2
                         (break 1 ^^ string "?" ^^ space ^^ v2 ^^ break 1 ^^ string ":" ^^ space
                        ^^ string "0ULL"))))
      | Max, Uint32_prec _ -> func "max"
      | Max, Uint64_prec _ -> func "max"
      | Min, Uint32_prec _ -> func "min"
      | Min, Uint64_prec _ -> func "min"

    let rec unop_syntax prec v =
      let open PPrint in
      let f prefix suffix expr = group (string prefix ^^ expr ^^ string suffix) in
      let func fn expr = group (string fn ^^ parens expr) in
      (* A libm call on a bfloat16 operand resolves (the operand converts to float) but *returns
         float*. Assigning that back to a bfloat16 cell is accepted -- __hip_bfloat16's converting
         constructor is implicit -- so it goes unnoticed until the placement that inlines the call
         instead makes the float an operand of a bfloat16 binop, where hiprtc reports
         "operator '+' is ambiguous ('__hip_bfloat16' and 'float')" (gh-ocannl-549). Bridge the
         result back the way [ToPowOf], [Relu], [Recip] and [Satur01] already do, so the emission
         is bfloat16-typed wherever it lands. *)
      let bf16_func fn = f ("__float2bfloat16(" ^ fn ^ "(__bfloat162float(") ")))" in
      match (v, prec) with
      | Ops.Identity, _ -> f "" ""
      | Uint4x32_to_prec_uniform1, Ops.Uint4x32_prec _ ->
          invalid_arg
            "Hip_backend.unop_syntax: Uint4x32_to_prec_uniform1 not supported for Uint4x32"
      (* Heterogeneous op: the argument is uint4x32 whatever the result precision, so it must stay
         ahead of the fp8 float-bridging below; the fp8 builtin returns __hip_fp8_e5m2. *)
      | Uint4x32_to_prec_uniform1, _ -> func ("uint4x32_to_" ^ Ops.prec_string prec ^ "_uniform")
      | _, Ops.Fp8_prec _ ->
          (* __hip_fp8_e5m2 defines no arithmetic operators, and its implicit conversion operators
             (float, double, int, char, ... in amd_hip_fp8.h) make the built-in operators ambiguous,
             so bridge fp8 math through float, mirroring the CC backend's fp8 handling. *)
          fun expr ->
            group
              (string "(__hip_fp8_e5m2)"
              ^^ parens (unop_syntax Ops.single v (string "(float)" ^^ parens expr)))
      | Relu, Ops.Single_prec _ -> f "fmaxf(0.0, " ")"
      | Relu, Ops.Half_prec _ -> f "__hmax_nan(__ushort_as_half((unsigned short)0x0000U), " ")"
      | Relu, Ops.Byte_prec _ -> f "fmax(0, " ")"
      (* Mixing a [__hip_bfloat16] with a literal of another arithmetic type is ambiguous under
         hiprtc for the same reason as [fma] on bfloat16 operands (see [ternop_syntax] below), so
         [fmax(0.0, bf16)] does not compile. Bridge through float like the bf16 binops above, and
         produce the result with [__float2bfloat16]. *)
      | Relu, Ops.Bfloat16_prec _ -> f "__float2bfloat16(fmaxf(0.0f, __bfloat162float(" ")))"
      | Relu, _ -> f "fmax(0.0, " ")"
      | Satur01, Byte_prec _ -> f "fmax(0, fmin(1, " "))"
      | Satur01, Bfloat16_prec _ ->
          f "__float2bfloat16(fmaxf(0.0f, fminf(1.0f, __bfloat162float(" "))))"
      | Satur01, Half_prec _ ->
          f
            "__hmax_nan(__ushort_as_half((unsigned short)0x0000U), \
             __hmin_nan(__ushort_as_half((unsigned short)0x3C00U), "
            "))"
      | Satur01, Single_prec _ -> f "fmaxf(0.0f, fminf(1.0f, " "))"
      | Satur01, _ -> f "fmax(0.0, fmin(1.0, " "))"
      | Exp, Half_prec _ -> func "hexp"
      | Exp, Double_prec _ -> func "exp"
      | Exp, Bfloat16_prec _ -> bf16_func "expf"
      | Exp, _ -> func "expf"
      | Log, Half_prec _ -> func "hlog"
      | Log, Double_prec _ -> func "log"
      | Log, Bfloat16_prec _ -> bf16_func "logf"
      | Log, _ -> func "logf"
      | Exp2, Half_prec _ -> func "hexp2"
      | Exp2, Double_prec _ -> func "exp2"
      | Exp2, Bfloat16_prec _ -> bf16_func "exp2f"
      | Exp2, _ -> func "exp2f"
      | Log2, Half_prec _ -> func "hlog2"
      | Log2, Double_prec _ -> func "log2"
      | Log2, Bfloat16_prec _ -> bf16_func "log2f"
      | Log2, _ -> func "log2f"
      | Sin, Half_prec _ -> func "hsin"
      | Sin, Double_prec _ -> func "sin"
      | Sin, Bfloat16_prec _ -> bf16_func "sinf"
      | Sin, _ -> func "sinf"
      | Cos, Half_prec _ -> func "hcos"
      | Cos, Double_prec _ -> func "cos"
      | Cos, Bfloat16_prec _ -> bf16_func "cosf"
      | Cos, _ -> func "cosf"
      | Sqrt, Half_prec _ -> func "hsqrt"
      | Sqrt, Double_prec _ -> func "sqrt"
      | Sqrt, Bfloat16_prec _ -> bf16_func "sqrtf"
      | Sqrt, _ -> func "sqrtf"
      | Recip, Byte_prec _ ->
          invalid_arg "Hip_backend.unop_syntax: Recip not supported for byte/integer precisions"
      | Recip, Half_prec _ -> func "hrcp"
      | Recip, Single_prec _ -> f "(1.0f / (" "))"
      | Recip, Double_prec _ -> f "(1.0 / (" "))"
      (* [1 / bf16] is ambiguous: the int operand can pair with any of the bfloat16 conversions. *)
      | Recip, Bfloat16_prec _ -> f "__float2bfloat16(1.0f / __bfloat162float(" "))"
      | Recip, _ -> f "(1 / (" "))"
      | Recip_sqrt, Byte_prec _ ->
          invalid_arg
            "Hip_backend.unop_syntax: Recip_sqrt not supported for byte/integer precisions"
      | Recip_sqrt, Half_prec _ -> func "hrsqrt"
      | Recip_sqrt, Double_prec _ -> f "(1.0 / sqrt(" "))"
      | Recip_sqrt, Single_prec _ -> f "(1.0f / sqrtf(" "))"
      | Recip_sqrt, Bfloat16_prec _ ->
          f "__float2bfloat16(1.0f / sqrtf(__bfloat162float(" ")))"
      | Recip_sqrt, _ -> f "(1 / sqrtf(" "))"
      | Neg, _ -> f "(-(" "))"
      | Trunc, Double_prec _ -> func "trunc"
      | Trunc, Bfloat16_prec _ -> bf16_func "truncf"
      | Trunc, _ -> func "truncf"
      | Tanh_approx, Byte_prec _ ->
          invalid_arg
            "Hip_backend.unop_syntax: Tanh_approx not supported for byte/integer precisions"
      | Tanh_approx, Half_prec _ -> func "htanh_approx"
      | Tanh_approx, Single_prec _ -> func "tanhf"
      | Tanh_approx, Bfloat16_prec _ -> bf16_func "tanhf"
      | Tanh_approx, _ -> func "tanh"
      (* [bf16 == 0.0] is ambiguous for the same reason as [1 / bf16] above. *)
      | Not, Bfloat16_prec _ -> f "__float2bfloat16(__bfloat162float(" ") == 0.0f ? 1.0f : 0.0f)"
      | Not, _ -> f "(" " == 0.0 ? 1.0 : 0.0)"

    let vec_unop_syntax prec op v =
      let open PPrint in
      match (op, prec) with
      | Ops.Uint4x32_to_prec_uniform, _ ->
          group (string ("uint4x32_to_" ^ Ops.prec_string prec ^ "_uniform_vec(") ^^ v ^^ rparen)

    let rec ternop_syntax prec v =
      let open PPrint in
      let func fn v1 v2 v3 = group (string fn ^^ parens (separate comma [ v1; v2; v3 ])) in
      match (v, prec) with
      | _, Ops.Fp8_prec _ ->
          (* __hip_fp8_e5m2 defines no arithmetic operators, and its implicit conversion operators
             (float, double, int, char, ... in amd_hip_fp8.h) make the built-in operators ambiguous,
             so bridge fp8 math through float, mirroring the CC backend's fp8 handling. *)
          fun v1 v2 v3 ->
            let fl v = string "(float)" ^^ parens v in
            group
              (string "(__hip_fp8_e5m2)"
              ^^ parens (ternop_syntax Ops.single v (fl v1) (fl v2) (fl v3)))
      | Ops.Where, _ ->
          (* The whole ternary must be parenthesized, not just the condition: C's [?:] binds looser
             than the surrounding arithmetic, so for an expression like [where(c,a,b) + 1] the
             trailing [+ 1] would otherwise be absorbed into the else-branch, silently dropping it
             from the then-branch (see the CUDA backend, task-04f97340). *)
          fun v1 v2 v3 -> group (parens (parens v1 ^^ string " ? " ^^ v2 ^^ string " : " ^^ v3))
      | FMA, Ops.Half_prec _ -> func "__hfma"
      (* [__hip_bfloat16] has implicit conversion operators to float, __bf16, int, char, ... , so a
         plain [fma] call on bfloat16 operands is ambiguous under hiprtc: its float, double and
         _Float16 overloads (hiprtc_runtime.h) are reached through different conversion operators,
         which makes their conversion sequences indistinguishable. [__hfma] from amd_hip_bf16.h
         takes bfloat16 operands exactly, mirroring [__hmax] / [__hmin] above. *)
      | FMA, Ops.Bfloat16_prec _ -> func "__hfma"
      | FMA, Ops.Single_prec _ -> func "fmaf"
      | FMA, _ -> func "fma"
      | Mul3, _ -> fun v1 v2 v3 -> group (parens (v1 ^^ string " * " ^^ v2 ^^ string " * " ^^ v3))

    let convert_precision ~from ~to_ =
      match (from, to_) with
      | Ops.Double_prec _, Ops.Double_prec _
      | Single_prec _, Single_prec _
      | Half_prec _, Half_prec _
      | Byte_prec _, Byte_prec _
      | Uint16_prec _, Uint16_prec _
      | Int32_prec _, Int32_prec _
      | Int64_prec _, Int64_prec _
      | Uint4x32_prec _, Uint4x32_prec _
      | Bfloat16_prec _, Bfloat16_prec _
      | Fp8_prec _, Fp8_prec _
      | Void_prec, Void_prec ->
          ("", "")
      (* hip_fp16.h has no [__double2half]; route through float. *)
      | Double_prec _, Half_prec _ -> ("__float2half((float)(", "))")
      | Single_prec _, Half_prec _ -> ("__float2half(", ")")
      | Byte_prec _, Half_prec _ -> ("__ushort2half_rn((unsigned short int)", ")")
      | Double_prec _, Uint4x32_prec _ -> ("double_to_uint4x32(", ")")
      | Single_prec _, Uint4x32_prec _ -> ("single_to_uint4x32(", ")")
      | Uint4x32_prec _, _ -> ("", ".v[0]")
      | Byte_prec _, Uint4x32_prec _ -> ("byte_to_uint4x32(", ")")
      | Uint16_prec _, Uint4x32_prec _ -> ("uint16_to_uint4x32(", ")")
      | Bfloat16_prec _, Uint4x32_prec _ -> ("bfloat16_to_uint4x32(", ")")
      | Half_prec _, Uint4x32_prec _ -> ("half_to_uint4x32(", ")")
      | Fp8_prec _, Uint4x32_prec _ -> ("fp8_to_uint4x32(", ")")
      (* The integer counter conversions MUST call the builtins, which spread the bits across all
         four uint4x32 lanes (golden-ratio / MMIX / rotation mixing). The raw struct literal below
         only fills lane 0, leaving lanes 1-3 zero; with the 2-round light threefry used for
         parameter init that produces near-identical outputs for consecutive counters (periodicity),
         so random inits diverge from CC/Metal. [Ops.index_prec] is signed [int32] (or [int64] under
         [large_models]), so the signed arms are the conversion hit by every PRNG init loop, e.g.
         centered [uniform1] parameter initialization (task-04f97340). *)
      | Int32_prec _, Uint4x32_prec _ -> ("int32_to_uint4x32(", ")")
      | Int64_prec _, Uint4x32_prec _ -> ("int64_to_uint4x32(", ")")
      | Uint32_prec _, Uint4x32_prec _ -> ("uint32_to_uint4x32(", ")")
      | Uint64_prec _, Uint4x32_prec _ -> ("uint64_to_uint4x32(", ")")
      | _, Uint4x32_prec _ -> ("{(unsigned int)(", "), 0, 0, 0}")
      (* [__hip_bfloat16] has constructors from float, double, short, unsigned short, ... and
         [__half] has several conversion operators, so C-style casts between them are ambiguous
         (observed with ROCm 7.1 hiprtc); route through float explicitly. Same precaution for the
         half <-> fp8 pairs. *)
      | Half_prec _, Bfloat16_prec _ -> ("__float2bfloat16(__half2float(", "))")
      | Bfloat16_prec _, Half_prec _ -> ("__float2half(__bfloat162float(", "))")
      | _, Bfloat16_prec _ -> ("__float2bfloat16((float)(", "))")
      | Bfloat16_prec _, _ -> ("(" ^ typ_of_prec to_ ^ ")(__bfloat162float(", "))")
      | Half_prec _, Fp8_prec _ -> ("(__hip_fp8_e5m2)(__half2float(", "))")
      | Fp8_prec _, Half_prec _ -> ("__float2half((float)(", "))")
      | ( Fp8_prec _,
          (Byte_prec _ | Uint16_prec _ | Int32_prec _ | Uint32_prec _ | Int64_prec _ | Uint64_prec _)
        ) ->
          (* __hip_fp8_e5m2's integer conversion operators saturate (wrong for negative values into
             unsigned types) and overlap enough to make direct casts ambiguity-prone; convert via
             float, like the CC backend. *)
          ("(" ^ typ_of_prec to_ ^ ")((float)(", "))")
      | _ -> ("(" ^ typ_of_prec to_ ^ ")(", ")")

    let kernel_log_param = Some ("int", "log_id")
    let log_involves_file_management = false

    let pp_log_statement ~log_param_c_expr_doc ~base_message_literal ~args_docs =
      let open PPrint in
      let format_string_literal =
        let res = String.substr_replace_all base_message_literal ~pattern:"\n" ~with_:"$" in
        let res =
          if for_log_trace_tree && String.is_suffix res ~suffix:"$" then
            String.drop_suffix res 1 ^ "\\n"
          else res
        in
        !Utils.captured_log_prefix ^ "%d: " ^ res
      in
      let all_args =
        match log_param_c_expr_doc with
        | Some doc -> doc :: args_docs
        | None -> args_docs (* Should not happen if kernel_log_param is Some *)
      in
      group
        (string "printf("
        ^^ dquotes (string format_string_literal)
        ^^ comma
        ^^ nest 4 (break 1 ^^ separate (comma ^^ break 1) all_args)
        ^^ rparen ^^ semi)
  end

  (* hiprtc ships built-in HIP headers, and device-side printf needs no declaration on ROCm. *)
  let hip_includes =
    {|#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
/* hip_fp8.h ships with ROCm >= 6.2 (hiprtc is clang, so __has_include is available); guarding
   keeps non-fp8 kernels compiling on older SDKs, where fp8 kernels still fail with unknown type
   __hip_fp8_e5m2. */
#if __has_include(<hip/hip_fp8.h>)
#include <hip/hip_fp8.h>
#endif

/* Define math constants that would normally come from <math.h> */
#ifndef INFINITY
#define INFINITY __builtin_huge_valf()
#endif
#ifndef NAN
#define NAN __builtin_nanf("")
#endif|}

  let%diagn2_sexp compile ~name bindings ({ Low_level.traced_store; _ } as lowered) =
    let module Syntax = C_syntax.C_syntax (Hip_syntax_config (struct
      let procs = [| lowered |]
    end))
    in
    let idx_params = Indexing.bound_symbols bindings in
    let kparams, proc_doc, launch = Syntax.compile_proc ~name idx_params lowered in
    let source =
      Syntax.filter_and_prepend_builtins ~includes:hip_includes ~builtins:Builtins_hip.builtins
        ~proc_doc
    in
    let code = hip_to_code ~name source in
    { traced_store; code; kparams; bindings; name; launch }

  let%diagn2_sexp compile_batch ~names bindings lowereds =
    let module Syntax = C_syntax.C_syntax (Hip_syntax_config (struct
      let procs = Array.filter_opt lowereds
    end))
    in
    let idx_params = Indexing.bound_symbols bindings in
    let kparams_and_docs =
      Array.map2_exn names lowereds
        ~f:
          (Option.map2 ~f:(fun name lowered ->
               let kparams, doc, launch = Syntax.compile_proc ~name idx_params lowered in
               ((kparams, name, launch), doc)))
    in
    let all_proc_docs = List.filter_map (Array.to_list kparams_and_docs) ~f:(Option.map ~f:snd) in
    let final_doc = PPrint.(separate hardline all_proc_docs) in
    let source =
      Syntax.filter_and_prepend_builtins ~includes:hip_includes ~builtins:Builtins_hip.builtins
        ~proc_doc:final_doc
    in
    let name : string =
      String.(
        strip ~drop:(equal_char '_')
        @@ common_prefix (Array.to_list names |> List.concat_map ~f:Option.to_list))
    in
    let code = hip_to_code ~name source in
    let traced_stores = Array.map lowereds ~f:(Option.map ~f:(fun l -> l.Low_level.traced_store)) in
    let kparams_and_names = Array.map kparams_and_docs ~f:(Option.map ~f:fst) in
    { traced_stores; code; kparams_and_names; bindings }

  let get_global_run_id =
    let next_id = ref 0 in
    fun () ->
      Int.incr next_id;
      if !next_id < 0 then next_id := 0;
      !next_id

  (* {2 Post-link scratch validation (gh-ocannl-533)}

     A kernel's private (scratch) segment is sized by the compiler and budgeted by the runtime only
     at dispatch. When the dispatch asks for more scratch than the device can back, ROCm aborts the
     QUEUE -- "[UpdateScratch] scratch_size overflow!" / [HSA_STATUS_ERROR_INVALID_ARGUMENT] -- and
     what reaches OCaml out of synchronize is a bare [hipErrorInvalidValue], the same code an
     uninitialized input yields. There is nothing to classify on, and the stream is already dead:
     gh-ocannl-533 saw one autotune candidate take the whole benchmark process down with it.

     So this is prediction, not recovery (see docs/proposals/gh-ocannl-536.md): read the linked
     kernel's private segment size and decline an over-budget kernel BEFORE it is ever launched,
     as a typed [Resource_exceeded Thread_scratch]. To a tuner candidate that is an ordinary
     decline the blocker census tabulates; a hand-written schedule gets the usual
     [Utils.User_error] rendering at the public [Context.compile] boundary.

     The budget model, established experimentally on gfx1151 (Radeon 8060S, ROCm 7.14, WSL2) -- see
     the gh-ocannl-533 writeup:

     - The rejection is a function of the per-work-item size ALONE. A kernel at 98320 B/work-item
       launches at 204800 work-items; one at 114704 B is rejected at a SINGLE work-item. So the
       runtime backs the worst-case fully-occupied device, not the requested grid, and the check
       needs no launch geometry.
     - The cutoff sits where [private_seg_size] rounded up to a 64-byte granule, times the device's
       maximum resident work-items ([max_threads_per_multiprocessor * multiprocessor_count]),
       crosses 4 GiB. Measured boundary: 104832 B accepted, 104848 B rejected; the model reproduces
       every one of the ~70 sampled points, including both sides of that 16-byte step.
     - The compiler separately refuses a stack frame over 262136 B, so it never emits a kernel far
       above this; #533's 163856 B is comfortably inside what compiles and outside what launches.

     The 4 GiB cap is not a value any HIP or HSA query exposes -- the abort comes from the WSL WDDM
     thunk ([wsl::thunk::ComputeQueue::UpdateScratch]) -- so it is a documented constant from this
     experiment, while the multiplier is genuinely queried. Where the model is unverified the right
     answer is silence, not a guess: [ocannl_hip_scratch_validation=false] disables the check
     entirely, and a device that reports no usable occupancy figures is never rejected. *)

  let hip_scratch_validation =
    lazy (Utils.get_global_flag ~default:true ~arg_name:"hip_scratch_validation")

  (* Total scratch the runtime backs = per-work-item size, rounded up to the allocation granule,
     times the device's maximum resident work-items. *)
  let scratch_granule_bytes = 64
  let scratch_total_cap_bytes = 4 * 1024 * 1024 * 1024

  let scratch_limit_per_work_item (attrs : H.Device.attributes) =
    let resident = attrs.max_threads_per_multiprocessor * attrs.multiprocessor_count in
    if resident <= 0 then None
    else
      (* Largest granule-aligned size whose full-occupancy total still fits the cap. *)
      let granules_per_work_item = scratch_total_cap_bytes / resident / scratch_granule_bytes in
      if granules_per_work_item <= 0 then None
      else Some (granules_per_work_item * scratch_granule_bytes)

  (* Memoized per ordinal, like [_hip_properties]: this runs on EVERY link, and re-entering the
     driver for static device properties once per routine is pure overhead — it showed up as
     contention with several HIP processes sharing one iGPU. *)
  let scratch_budget_of_device =
    let cache =
      lazy
        (Array.init (num_devices ()) ~f:(fun ordinal ->
             lazy
               (let attrs = H.Device.get_attributes (H.Device.get ~ordinal) in
                (attrs, scratch_limit_per_work_item attrs))))
    in
    fun (device : device) -> Lazy.force (Lazy.force cache).(device.ordinal)

  let validate_scratch_budget ~(device : device) ~name func =
    if Lazy.force hip_scratch_validation then
      let attrs, limit = scratch_budget_of_device device in
      Option.iter limit ~f:(fun limit ->
          let requested =
            H.Module.get_function_attribute func H.Module.HIP_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES
          in
          let rounded =
            (requested + scratch_granule_bytes - 1) / scratch_granule_bytes * scratch_granule_bytes
          in
          if rounded > limit then
            raise
            @@ Schedule_outcome.Cause_at
                 ( Schedule_outcome.Backend_link,
                   Schedule_outcome.Resource_exceeded
                     {
                       resource = Schedule_outcome.Thread_scratch;
                       requested;
                       limit = Some limit;
                       detail =
                         [%string
                           "HIP: kernel %{name} needs %{requested#Int} bytes of private (scratch) \
                            memory per work-item, above the %{limit#Int} bytes this device can \
                            back at full occupancy (%{attrs.max_threads_per_multiprocessor#Int} \
                            work-items x %{attrs.multiprocessor_count#Int} CUs against a \
                            %{scratch_total_cap_bytes#Int}-byte scratch allocation). Launching it \
                            would abort the queue rather than fail cleanly (gh-ocannl-533)"];
                     } ))

  let link_proc ~prior_context ~name ~(kparams : (string * kparam_source) list)
      ~(launch : Low_level.launch_dims) ~ctx_buffers lowered_bindings run_module =
    let func = H.Module.get_function run_module ~name in
    let device = prior_context.device in
    validate_scratch_budget ~device ~name func;
    let stream_name = get_name device in
    (* Pre-resolve slab bases to keep the owning [Deviceptr.t]'s alive for the lifetime of the task
       closure; region views ([Tensor_at]) are non-owning and must not outlive the slab base. *)
    let ctx_bases = Map.map ctx_buffers ~f:(Slab.resolve_pool device) in
    let%diagn3_sexp work () : unit =
      let log_id = get_global_run_id () in
      let log_id_prefix = Int.to_string log_id ^ ": " in
      [%log_result
        "Launching",
        name,
        "on",
        stream_name,
        (log_id : int),
        (kparams : (string * kparam_source) list)];
      let module S = H.Stream in
      let args : S.kernel_param list =
        (* TODO: should we prohibit or warn about local-only tensors that are in
           prior_context.ctx_buffers? *)
        List.map kparams ~f:(function
          | _name, Kparam_ptr tn ->
              let loc = Option.value_exn ~here:[%here] @@ Map.find ctx_buffers tn in
              let base = Map.find_exn ctx_bases tn in
              S.Tensor_at (H.Deviceptr.offset base ~bytes:loc.offset)
          | _name, Log_file_name -> S.Int log_id
          | _name, Merge_buffer ->
              let loc = Option.value_exn ~here:[%here] !(device.merge_buffer) in
              let base = Slab.resolve_pool device loc in
              S.Tensor_at (H.Deviceptr.offset base ~bytes:loc.offset)
          | _name, Static_idx s ->
              let i = Indexing.find_exn lowered_bindings s in
              (* Shared bind-time validation: negativity, range -- inclusive [0, range] for symbolic
                 extents (gh-490), strict [0, range) for indices -- and index width. *)
              Indexing.validate_bound_value ~width64:Utils.settings.large_models s !i;
              S.Int !i
          | _name, (Kparam_pool_slab _ | Kparam_pool_slots _) ->
              (* The HIP backend uses per-tnode pointer params ([`Per_param] codegen); only the
                 Metal backend emits the pooled slab / slot parameters. *)
              invalid_arg "Hip_backend.link: unexpected pooled kparam (HIP uses per-tnode pointers)")
      in
      set_ctx @@ ctx_of prior_context;
      [%log "launching the kernel"];
      (if Utils.debug_log_from_routines () then
         Utils.add_log_processor ~prefix:log_id_prefix @@ fun log_contents ->
         Utils.log_debug_routine_logs ~log_contents ~stream_name);
      (* Launch dimensions derived from hardware-annotated loops (axis-types proposal §4);
         all-Serial kernels launch 1x1x1, as before. Static [__shared__] declarations do not use the
         dynamic pool, so [shared_mem_bytes] stays 0. *)
      S.launch_kernel func ~grid_dim_x:launch.Low_level.grid.(0)
        ~grid_dim_y:launch.Low_level.grid.(1) ~grid_dim_z:launch.Low_level.grid.(2)
        ~block_dim_x:launch.Low_level.block.(0) ~block_dim_y:launch.Low_level.block.(1)
        ~block_dim_z:launch.Low_level.block.(2) ~shared_mem_bytes:0 device.runner args;
      [%log "kernel launched"]
    in
    Task.Task
      {
        context_lifetime = (run_module, ctx_bases);
        description = "launches " ^ name ^ " on " ^ stream_name;
        work;
      }

  let%track3_sexp link prior_context (code : code) ctx_buffers =
    let ctx = ctx_of prior_context in
    set_ctx ctx;
    let run_module = H.Module.load_data_ex code.code (run_options ()) in
    prior_context.device.dev.set_builtins_in run_module;
    let idx_params = Indexing.bound_symbols code.bindings in
    let lowered_bindings : Indexing.lowered_bindings =
      List.map idx_params ~f:(fun s -> (s, ref 0))
    in
    let task =
      link_proc ~prior_context ~name:code.name ~kparams:code.kparams ~launch:code.launch
        ~ctx_buffers lowered_bindings run_module
    in
    (lowered_bindings, task)

  let%track3_sexp link_batch prior_context (code_batch : code_batch) ctx_buffers =
    let idx_params = Indexing.bound_symbols code_batch.bindings in
    let lowered_bindings : Indexing.lowered_bindings =
      List.map idx_params ~f:(fun s -> (s, ref 0))
    in
    let ctx = ctx_of prior_context in
    set_ctx ctx;
    let run_module = H.Module.load_data_ex code_batch.code (run_options ()) in
    prior_context.device.dev.set_builtins_in run_module;
    let procs =
      Array.mapi code_batch.kparams_and_names ~f:(fun i pns ->
          Option.value ~default:None
          @@ Option.map2 pns ctx_buffers.(i) ~f:(fun (kparams, name, launch) ctx_buffers ->
              let task =
                link_proc ~prior_context ~name ~kparams ~launch ~ctx_buffers lowered_bindings
                  run_module
              in
              Some task))
    in
    (lowered_bindings, procs)

  (* HIP kernel launches on one stream already execute in FIFO order, so the generic event chain is
     correct; a cheaper plain-sequence task is a possible follow-up (events are ~free there). *)
  let sequence_segments _context ~name:_ _tasks = None

  let get_global_debug_info () =
    Sexp.message "hip_global_debug"
      [ ("live_streams", [%sexp_of: int] @@ H.Stream.get_total_live_streams ()) ]

  let static_properties () =
    let device_properties =
      Array.init (num_devices ()) ~f:(fun ordinal ->
          let dev = H.Device.get ~ordinal in
          let attributes = H.Device.get_attributes dev in
          let props =
            [
              ("device_name", Sexp.Atom attributes.name);
              ("device_ordinal", [%sexp_of: int] ordinal);
              ("gcn_arch_name", Sexp.Atom attributes.gcn_arch_name);
              ("multiprocessor_count", [%sexp_of: int] attributes.multiprocessor_count);
              ("clock_rate", [%sexp_of: int] attributes.clock_rate);
              ("warp_size", [%sexp_of: int] attributes.warp_size);
              ("async_engine_count", [%sexp_of: int] attributes.async_engine_count);
              ("compute_capability_major", [%sexp_of: int] attributes.compute_capability_major);
              ("compute_capability_minor", [%sexp_of: int] attributes.compute_capability_minor);
              ("max_threads_per_block", [%sexp_of: int] attributes.max_threads_per_block);
              ("unified_addressing", [%sexp_of: bool] attributes.unified_addressing);
            ]
          in
          Sexp.message "device" props)
    in
    Sexp.List (Sexp.Atom "hip_devices" :: Array.to_list device_properties)

  (* Conservative per-workgroup device limits for the schedule layer (schedule-ir-optops §6):
     minimum across devices, so code compiled once is valid wherever it links. *)
  (* Memoized behind [lazy]: driver init and device enumeration must not run at backend-module
     initialization ([num_devices] forces [ensure_initialized]). *)
  let hardware_limits =
    let limits =
      lazy
        (let attrs =
           Array.init (num_devices ()) ~f:(fun ordinal ->
               H.Device.get_attributes (H.Device.get ~ordinal))
         in
         let min_over f = Array.map attrs ~f |> Array.min_elt ~compare:Int.compare in
         {
           Backend_intf.max_threads_per_workgroup =
             min_over (fun (a : H.Device.attributes) -> a.max_threads_per_block);
           max_workgroup_memory_bytes =
             min_over (fun (a : H.Device.attributes) -> a.shared_mem_per_block);
           (* Cooperative tile-MMA via rocWMMA, gated on [mma_supported]: RDNA3/RDNA3.5+
              (gfx11/gfx12) wave32 across ALL devices AND discoverable rocWMMA headers. CDNA (gfx9,
              wave64, MFMA) and header-less hosts stay on the scalar path -- reporting [Some] there
              would let autotune pick [Tile_mma] and then fail to compile. [None] unless EVERY
              device qualifies: limits are min-over-devices, so code compiled once must be valid
              wherever it links. Precision combinations are decided per call by [mma_syntax]. *)
           mma =
             (if mma_supported () then
                Some
                  {
                    Backend_intf.mma_simd_width = 32;
                    mma_tile = (16, 16, 16);
                    mma_format_tiles =
                      [
                        ((Backend_intf.Mma_f16, Backend_intf.Mma_f16), (16, 16, 16));
                        ((Backend_intf.Mma_bf16, Backend_intf.Mma_bf16), (16, 16, 16));
                      ];
                  }
              else None);
           simd_vector_bytes = 0;
           (* Advisory roofline envelope (gh-ocannl-491): documented rough constants for the
              RDNA3-class targets this backend is exercised on (dGPU/APU: ~10 fp32 TFLOP/s, ~250
              GB/s — Strix-Halo-class LPDDR5X). Per-device queries are calibration follow-up work;
              the model only ranks, so class-level numbers suffice. *)
           peak_flops = Some 1.0e13;
           peak_memory_bandwidth = Some 2.5e11;
         })
    in
    fun () -> Lazy.force limits

  let get_debug_info (device : device) =
    let tot, unr, unf = H.Stream.total_unreleased_unfinished_delimited_events device.runner in
    let i2s = [%sexp_of: int] in
    Sexp.message "hip_stream_debug"
      [ ("total_events", i2s tot); ("unreleased_events", i2s unr); ("unfinished_events", i2s unf) ]
end
