open Base
open Ir
module Tn = Tnode
module Lazy = Utils.Lazy
module Cu = Cuda
open Backend_intf

let _get_local_debug_runtime = Utils.get_local_debug_runtime

[%%global_debug_log_level 0]

(* export OCANNL_LOG_LEVEL_CUDA_BACKEND=9 to enable debugging into the log_files/ directory. *)
[%%global_debug_log_level_from_env_var "OCANNL_LOG_LEVEL_CUDA_BACKEND"]

let () =
  Cu.cuda_call_hook :=
    Some
      (fun ~message:_message ~status:_status ->
        [%debug_sexp
          [%log5_block
            _message;
            if not @@ Cu.is_success _status then [%log (_status : Cu.result)]]])

let _suspended () =
  Cu.cuda_call_hook := Some (fun ~message ~status:_ -> Stdlib.Printf.printf "CUDA %s\n" message)

module Backend_buffer = struct
  type buffer_ptr = Cu.Deviceptr.t

  let sexp_of_buffer_ptr ptr = Sexp.Atom (Cu.Deviceptr.string_of ptr)
end

module Device_config = struct
  include Backend_buffer

  type dev = {
    dev : Cu.Device.t;
    primary_context : Cu.Context.t;
    set_builtins_in : Cu.Module.t -> unit;
  }
  [@@deriving sexp_of]

  type runner = Cu.Stream.t [@@deriving sexp_of]
  type event = Cu.Delimited_event.t [@@deriving sexp_of]

  let name = "cuda"
end

module Device_stream = Backend_impl.Device_types_ll (Device_config)
open Device_config

let set_ctx ctx = Cu.Context.set_current ctx

(* The CUDA slab allocator: a private [(device_id, pool_id) -> CUdeviceptr] table backing the shared
   {!Backend_intf.Slab_alloc}. *)
module Slab = struct
  open Backend_intf

  type device = Device_stream.device
  type buffer_ptr = Cu.Deviceptr.t

  (* Requested sizes are tracked alongside the pointers so [get_used_memory] can report the bytes
     OCANNL allocated on the device (gh-ocannl-289): the driver's [get_free_and_total_mem] moves in
     allocation granules (~2 MiB), which hides sub-granule effects such as the liveness planner's
     arena savings (gh-ocannl-489) and counts other processes' memory. *)
  let pools : (int * int, buffer_ptr * int) Hashtbl.Poly.t = Hashtbl.Poly.create ()

  let alloc_pool ?mode:_ (device : device) ~pool_id ~size_in_bytes ~alignment:_ =
    set_ctx device.dev.primary_context;
    let key = (device.device_id, pool_id) in
    (* Free any prior allocation under this key before replacing it, so device memory stays
       equivalent to the pre-refactor path. Unique tnode pool ids never pre-exist; this only fires
       on the reserved merge pool growing in place. *)
    Option.iter (Hashtbl.find pools key) ~f:(fun (ptr, _) -> Cu.Deviceptr.mem_free ptr);
    let size_in_bytes = max 1 size_in_bytes in
    let ptr = Cu.Deviceptr.mem_alloc ~size_in_bytes in
    Hashtbl.set pools ~key ~data:(ptr, size_in_bytes)

  let free_pool =
    Some
      (fun (device : device) ~pool_id ->
        let key = (device.device_id, pool_id) in
        Option.iter (Hashtbl.find pools key) ~f:(fun (ptr, _) -> Cu.Deviceptr.mem_free ptr);
        Hashtbl.remove pools key)

  let resolve_pool (device : device) { pool_id; offset = _ } : buffer_ptr =
    (* Return the slab base. The byte offset is NOT folded into the handle here; callers apply it
       via the cudajit ?offset / ?dst_offset / ?src_offset params or via Cu.Deviceptr.offset. *)
    fst (Hashtbl.find_exn pools (device.device_id, pool_id))

  let used_memory (device : device) =
    Hashtbl.fold pools ~init:0 ~f:(fun ~key:(dev_id, _) ~data:(_, size) acc ->
        if dev_id = device.device_id then acc + size else acc)

  let memset_zero (device : device) ~pool_id ~offset ~size_in_bytes =
    let base = resolve_pool device { pool_id; offset } in
    if size_in_bytes > 0 then
      Cu.Stream.memset_d8 ~offset base Unsigned.UChar.zero ~length:size_in_bytes device.runner
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
  let is_done event = Cu.Delimited_event.query event
  let will_wait_for context event = Cu.Delimited_event.wait context.device.runner event
  let sync event = Cu.Delimited_event.synchronize event
  let all_work device = Cu.Delimited_event.record device.runner

  (* Driver initialization and device discovery are lazy: the singleton [Impl] module initializes at
     program startup (Backends instantiates it eagerly for nameable types), and cudajit is a depopt
     -- the library being installed does not imply a usable driver/GPU. Forcing here, at first
     device use, keeps CPU-only runs from touching the driver and lets [Context.auto] catch
     unusable-CUDA failures per call, as the retired per-call [fresh_backend] did. *)
  let ensure_initialized =
    lazy
      (if not !initialized then (
         Cu.init ();
         initialized := true))

  let num_devices () =
    Lazy.force ensure_initialized;
    Cu.Device.get_count ()

  (* [devices] is mutable to support plugging in new devices. *)
  let devices = lazy (ref @@ Array.create ~len:(num_devices ()) None)

  (* Minimum compute capability across devices as [major * 10 + minor] (e.g. 89 for sm_89, 120 for
     sm_120), memoized behind [lazy] (driver init and device enumeration must not run at
     backend-module initialization). Tensor cores (wmma) need sm_70+, bf16 wmma sm_80+, fp8 mma.sync
     sm_89+ (docs/proposals/tensorize-mma.md T3). *)
  let min_compute_capability =
    let cc =
      lazy
        (let n = num_devices () in
         if n = 0 then 0
         else
           Array.init n ~f:(fun ordinal ->
               let a : Cu.Device.attributes = Cu.Device.get_attributes (Cu.Device.get ~ordinal) in
               (a.compute_capability_major * 10) + a.compute_capability_minor)
           |> Array.min_elt ~compare:Int.compare
           |> Option.value ~default:0)
    in
    fun () -> Lazy.force cc

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
    Cu.Context.set_current device.dev.primary_context;
    Cu.Context.synchronize ();
    (* Note: this is not necessary as releasing the primary context by GC will reset the context.
       gh-ocannl-344: constants are bump-packed, so several cache entries share one constant pool
       slab; free each distinct [pool_id] exactly once (freeing per-entry would double-free / free a
       sub-region pointer). [Slab.free_pool] frees the slab and drops its table entry. *)
    Hashtbl.data device.constant_buffer_cache
    |> List.map ~f:(fun (loc : Backend_intf.buffer_loc) -> loc.pool_id)
    |> List.dedup_and_sort ~compare:Int.compare
    |> List.iter ~f:(fun pool_id ->
        Option.iter Slab.free_pool ~f:(fun free -> free device ~pool_id))

  let%diagn2_sexp cuda_to_ptx ~name cu_src =
    (* Tensorize-mma T3: kernels containing wmma intrinsics need <mma.h> and an explicit arch
       (nvrtc's default is below sm_70). Injected only when used, so kernels without tensor cores
       compile exactly as before even where the toolkit headers are absent. The [(wmma-bf16)] and
       [(wmma-tf32)] markers are emitted by [mma_syntax] for bf16 resp. tf32 fragments (both
       sm_80+); the [(mma-fp8)] marker for the
       inline-PTX fp8 [mma.sync] path (sm_89+, no header needed). *)
    let uses_wmma = String.is_substring cu_src ~substring:"nvcuda::wmma" in
    let cu_src = if uses_wmma then "#include <mma.h>\n" ^ cu_src else cu_src in
    (* Half/bf16 ARITHMETIC intrinsics (unlike the conversions, which cuda_fp16.h/cuda_bf16.h
       emulate on any arch) are only declared for __CUDA_ARCH__ >= 530 (halfs) resp. >= 800
       (bfloat16s), while nvrtc's default target is compute_52 — e.g. [__hfma] in a serial half
       matmul fails with "identifier undefined" unless we raise the floor. The bf16 overloads share
       the half intrinsics' names, so a bf16 kernel is recognized by the type name appearing
       alongside the arithmetic tokens; a kernel mixing half arithmetic with bf16 storage-only is
       conservatively floored at compute_80 too. *)
    let has s = String.is_substring cu_src ~substring:s in
    let uses_h_arith =
      (* Every half-arith token [unop_syntax]/[binop_syntax]/[ternop_syntax] can emit; [hexp]/[hlog]
         also cover their [2]-suffixed variants and [__hmax]/[__hmin] the [_nan] variants as
         substrings. *)
      List.exists ~f:has
        [
          "__hadd";
          "__hsub";
          "__hmul";
          "__hdiv";
          "__hmax";
          "__hmin";
          "__hgt";
          "__hfma";
          "hexp";
          "hlog";
          "hsin";
          "hcos";
          "hsqrt";
          "hrcp";
          "hrsqrt";
          "htanh_approx";
        ]
    in
    (* [compile_batch] concatenates several kernels into one source, so the floors can trigger
       independently (e.g. a half-wmma kernel batched with scalar bf16 arithmetic needs compute_80
       even without the [(wmma-bf16)] marker): take the max, not the first match. *)
    let arch_floor =
      List.filter_opt
        [
          (if uses_wmma then Some (if has "(wmma-bf16)" || has "(wmma-tf32)" then 80 else 70)
           else None);
          (if has "(mma-fp8)" then Some 89 else None);
          (if uses_h_arith then Some (if has "__nv_bfloat16" then 80 else 53) else None);
        ]
      |> List.max_elt ~compare:Int.compare
    in
    let arch_opts =
      match arch_floor with
      | Some floor ->
          (* CUDA 13 dropped offline compilation below compute_75 (Maxwell through Volta), so nvrtc
             13 rejects the compute_53/compute_70 floors outright: raise a triggered floor to
             compute_75 whenever every attached device can load such PTX (a device below sm_75 keeps
             the literal floor — it must be paired with an nvrtc 12.x that still accepts it). We
             deliberately do NOT raise all the way to the device arch (e.g. compute_120 on Blackwell
             GeForce): PTX targeted at a floor arch is forward-JIT-compiled by the driver on every
             later GPU, and instruction variants like the sm_89 fp8 [mma.sync] remain valid under a
             compute_89 target where a compute_120 target would demand the family-specific [kind::]
             forms. *)
          let arch = max floor (min 75 (min_compute_capability ())) in
          [ Printf.sprintf "--gpu-architecture=compute_%d" arch ]
      | None -> []
    in
    let name_cu = name ^ ".cu" in
    if Utils.settings.output_debug_files_in_build_directory then (
      let build_file = Utils.open_build_file ~base_name:name ~extension:".cu" in
      Stdio.Out_channel.output_string build_file.oc cu_src;
      build_file.finalize ());
    [%log "compiling to PTX"];
    let with_debug =
      Utils.settings.output_debug_files_in_build_directory || Utils.settings.log_level > 0
    in
    let cuda_include_opt =
      (* On Windows, check for the no-spaces junction created by ocaml-cudajit *)
      let cuda_path =
        if String.(Stdlib.Sys.os_type = "Win32" || Stdlib.Sys.os_type = "Cygwin") then
          let junction_path =
            match Sys.getenv "LOCALAPPDATA" with
            | Some local_appdata -> local_appdata ^ "/cuda_path_link"
            | None -> ( match Sys.getenv "CUDA_PATH" with Some p -> p | None -> "")
          in
          if Stdlib.Sys.file_exists (junction_path ^ "/include") then Some junction_path
          else Sys.getenv "CUDA_PATH"
        else Sys.getenv "CUDA_PATH"
      in
      match cuda_path with
      | Some cuda_path ->
          (* Normalize path separators for Windows *)
          let include_path =
            if String.(Stdlib.Sys.os_type = "Win32" || Stdlib.Sys.os_type = "Cygwin") then
              String.map ~f:(fun c -> if Char.(c = '\\') then '/' else c) (cuda_path ^ "/include")
            else cuda_path ^ "/include"
          in
          [ "-I" ^ include_path ]
      | None ->
          if
            (* Fallback to common location if CUDA_PATH is not set *)
            Stdlib.Sys.file_exists "/usr/local/cuda/include"
          then [ "-I/usr/local/cuda/include" ]
          else []
    in
    let options =
      cuda_include_opt @ arch_opts
      @ ("--use_fast_math" :: (if Utils.with_runtime_debug () then [ "--device-debug" ] else []))
    in
    let ptx = Nvrtc.compile_to_ptx ~cu_src ~name:name_cu ~options ~with_debug in
    if Utils.settings.output_debug_files_in_build_directory then (
      let oc = Out_channel.open_text @@ Utils.build_file @@ name ^ ".ptx" in
      Stdio.Out_channel.output_string oc @@ Nvrtc.string_from_ptx ptx;
      Stdio.Out_channel.flush oc;
      Stdio.Out_channel.close oc;
      let oc = Out_channel.open_text @@ Utils.build_file @@ name ^ ".cu_log" in
      Stdio.Out_channel.output_string oc
      @@ Option.value_exn ~here:[%here] (Nvrtc.compilation_log ptx);
      Stdio.Out_channel.flush oc;
      Stdio.Out_channel.close oc);
    ptx

  let run_options () =
    if Utils.with_runtime_debug () then
      Cu.Module.[ GENERATE_DEBUG_INFO true; GENERATE_LINE_INFO true ]
    else []

  (* No longer need runtime linking since Threefry is included directly in each kernel *)
  let set_builtins_for_device ~primary_context:_ _kernel_module = assert !initialized

  let%track3_sexp get_device ~(ordinal : int) : device =
    if num_devices () <= ordinal then
      invalid_arg [%string "Exec_as_cuda.get_device %{ordinal#Int}: not enough devices"];
    let devices = Lazy.force devices in
    (if Array.length !devices <= ordinal then
       let old, len = (!devices, Array.length !devices) in
       devices := Array.init (ordinal + 1) ~f:(fun i -> if i < len then old.(i) else None));
    let default () =
      let dev = Cu.Device.get ~ordinal in
      let primary_context : Cu.Context.t = Cu.Context.get_primary dev in
      let set_builtins_in = set_builtins_for_device ~primary_context in
      let dev = { dev; primary_context; set_builtins_in } in
      set_ctx primary_context;
      if Utils.debug_log_from_routines () && not (Hash_set.mem initialized_devices ordinal) then
        Int.of_string_opt @@ Utils.get_global_arg ~arg_name:"cuda_printf_fifo_size" ~default:""
        |> Option.iter ~f:Cu.Context.(set_limit PRINTF_FIFO_SIZE);
      Hash_set.add initialized_devices ordinal;
      (* With one compute stream per device, the runner (CUDA stream) is created with the device. *)
      let cu_stream = Cu.Stream.create ~non_blocking:true () in
      let result = make_device dev cu_stream ~ordinal in
      Stdlib.Gc.finalise finalize_device result;
      !devices.(ordinal) <- Some result;
      result
    in
    Option.value_or_thunk !devices.(ordinal) ~default

  let _cuda_properties =
    let cache =
      let%debug2_sexp f (ordinal : int) =
        let dev = get_device ~ordinal in
        lazy (Cu.Device.get_attributes dev.dev.dev)
      in
      lazy (Array.init (num_devices ()) ~f)
    in
    let%debug2_sexp get_props (device : device) : Cu.Device.attributes =
      let cache = Lazy.force cache in
      Lazy.force cache.(device.ordinal)
    in
    get_props

  let await (device : device) : unit =
    set_ctx device.dev.primary_context;
    (* Match the HIP logging contract: device-side [printf] must be fully drained before a caller
       closes or restores captured stdout. The stronger synchronization is debug-only. *)
    if Utils.debug_log_from_routines () then Cu.Context.synchronize ()
    else Cu.Stream.synchronize device.runner

  let is_idle (device : device) = Cu.Stream.is_ready device.runner

  (* Transfers take {!Backend_intf.buffer_loc} and resolve to the concrete [CUdeviceptr] here,
     against the device's private pool table. We pass [~length] explicitly to [memcpy_H_to_D] /
     [memcpy_D_to_H]: without it the cudajit impl computes [size_in_bytes = full_size - offset],
     which would reduce the copy to 0 bytes when the tensor is placed at an offset equal to its own
     size (the common bump-packed case). *)
  let from_host ~dst ~dst_loc hosted =
    set_ctx @@ ctx_of dst;
    let base = Slab.resolve_pool dst.device dst_loc in
    let f src =
      let full_bytes = Bigarray.Genarray.size_in_bytes src in
      let elem_bytes = Bigarray.kind_size_in_bytes (Bigarray.Genarray.kind src) in
      Cu.Stream.memcpy_H_to_D ~length:(full_bytes / elem_bytes) ~dst_offset:dst_loc.offset ~dst:base
        ~src dst.device.runner
    in
    Ndarray.apply { f } hosted

  let to_host ~src ~src_loc hosted =
    set_ctx @@ ctx_of src;
    let base = Slab.resolve_pool src.device src_loc in
    let f dst =
      let full_bytes = Bigarray.Genarray.size_in_bytes dst in
      let elem_bytes = Bigarray.kind_size_in_bytes (Bigarray.Genarray.kind dst) in
      Cu.Stream.memcpy_D_to_H ~length:(full_bytes / elem_bytes) ~src_offset:src_loc.offset ~dst
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
        Cu.Stream.memcpy_D_to_D ~size_in_bytes ~dst_offset ~src_offset ~dst:dst_base ~src:src_base
          dst.device.runner
      else
        Cu.Stream.memcpy_peer ~size_in_bytes ~dst_offset ~src_offset ~dst:dst_base
          ~dst_ctx:(ctx_of dst) ~src:src_base ~src_ctx:(ctx_of src) dst.device.runner
    in
    match (into_merge_buffer, dst_loc) with
    | No, None -> invalid_arg "Cuda_backend.device_to_device: missing dst_loc"
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
    ptx : Nvrtc.compile_to_ptx_result;
    kparams : (string * kparam_source) list;
    bindings : Indexing.unit_bindings;
    name : string;
    launch : Low_level.launch_dims;
  }
  [@@deriving sexp_of]

  type code_batch = {
    traced_stores : Low_level.traced_store option array;
    ptx : Nvrtc.compile_to_ptx_result;
    bindings : Indexing.unit_bindings;
    kparams_and_names :
      ((string * kparam_source) list * string * Low_level.launch_dims) option array;
  }
  [@@deriving sexp_of]

  module Cuda_syntax_config (Input : sig
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
          (* CUDA built-in variables — would shadow per-thread or per-block context *)
          "threadIdx";
          "blockIdx";
          "blockDim";
          "gridDim";
          "warpSize";
        ]

    let main_kernel_prefix = "extern \"C\" __global__"

    (* The pre-Phase-B single-thread guard is gone (axis-types proposal §4): an all-Serial kernel
       launches 1x1x1, making the guard redundant; annotated kernels need every thread. *)
    let kernel_prep_line = ""

    (* Use native CUDA types for loop indices and arguments instead of stdint.h types. Signed index
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
       [ocannl_shfl_xor] wraps [__shfl_xor_sync] (builtins_cuda.ml). All supported devices have
       32-wide warps. *)
    let warp_size = 32

    (* No vectorization pragmas in device code — SIMD-style gains on GPU come from memory
       transactions: eligible [Vectorized] loops render 128-bit packed loads/stores through the
       [__align__(16)] pack structs (gh-ocannl-463; llm.c's Packed128 shows LDG.128/STS.128 are the
       baseline for bandwidth-bound kernels), and everything else falls back to plain serial loops.
       Local arrays live in registers/local memory; no alignment attribute needed (packed accesses
       require device-resident nodes). *)
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
      | Ops.Bfloat16_prec _ -> "__nv_bfloat16" (* CUDA bfloat16 type *)
      | Ops.Fp8_prec _ -> "__nv_fp8_e5m2" (* CUDA FP8 type (E5M2 format) *)
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
      (* Fp8 needs [__nv_fp8_e5m2] elements: [Set_from_vec] assigns them to the fp8 array cells
         without a cast, and [__nv_fp8_e5m2] has no assignment from integer types. *)
      | Ops.Fp8_prec _, 16 -> "fp8x16_t"
      | (Ops.Uint16_prec _ | Ops.Bfloat16_prec _), 8 -> "uint16x8_t"
      | Ops.Half_prec _, 8 -> "half8_t"
      | _, 1 -> typ_of_prec prec
      | _ -> invalid_arg "Cuda_backend.vec_typ_of_prec: invalid combination"

    (* The wmma-supported precision combinations (tensorize-mma T3). Shared by [mma_syntax] and
       [mma_fragment_syntax] so a fragment scope accepts exactly when its nested update-only MMA
       calls would — including the numerics-policy gate on the tf32 arm, which lives here for the
       same reason. *)
    type wmma_combo_info = {
      wc_ab_typ : string;
          (** The fragment element type for [matrix_a]/[matrix_b] — a C++ type for the 16-bit
              combinations, the tag type [nvcuda::wmma::precision::tf32] for tf32 (storage stays
              [float]; only the fragments are tagged). *)
      wc_acc_typ : string;  (** The accumulator fragment element type. *)
      wc_tm : int;
      wc_tn : int;
      wc_tk : int;
          (** The intrinsic tile shape: 16×16×16 for the 16-bit combinations, 16×16×8 for tf32
              (mirrors [mma_format_tiles] in the capability descriptor). *)
      wc_ab_ld_mult : int;  (** wmma stride constraint: a/b leading-dim multiple, in elements. *)
      wc_d_ld_mult : int;  (** wmma stride constraint: d leading-dim multiple, in elements. *)
      wc_min_cc : int;
      wc_marker : string;
          (** Marker suffix for the rendering comment (["" | "-bf16" | "-tf32"]); [cuda_to_ptx]
              greps it to select the arch floor. *)
      wc_cvt_tf32 : bool;
          (** Convert loaded a/b fragment elements with [__float_to_tf32]: tf32 fragments load raw
              f32 bits, the explicit conversion performs the mantissa truncation (per the CUDA
              programming guide; the intrinsic requires already-converted inputs). *)
    }

    let wmma_combo ~a_prec ~b_prec ~d_prec =
      let mk ?(tile = (16, 16, 16)) ?(marker = "") ?(cvt_tf32 = false) ab_typ acc_typ ab_ld d_ld cc
          =
        let wc_tm, wc_tn, wc_tk = tile in
        Some
          {
            wc_ab_typ = ab_typ;
            wc_acc_typ = acc_typ;
            wc_tm;
            wc_tn;
            wc_tk;
            wc_ab_ld_mult = ab_ld;
            wc_d_ld_mult = d_ld;
            wc_min_cc = cc;
            wc_marker = marker;
            wc_cvt_tf32 = cvt_tf32;
          }
      in
      match (a_prec, b_prec, d_prec) with
      | Ops.Half_prec _, Ops.Half_prec _, Ops.Single_prec _ -> mk "__half" "float" 8 4 70
      | Ops.Half_prec _, Ops.Half_prec _, Ops.Half_prec _ -> mk "__half" "__half" 8 8 70
      | Ops.Bfloat16_prec _, Ops.Bfloat16_prec _, Ops.Single_prec _ ->
          mk ~marker:"-bf16" "__nv_bfloat16" "float" 8 4 80
      | Ops.Single_prec _, Ops.Single_prec _, Ops.Single_prec _
        when (Numerics.get ()).Numerics.tf32_matmuls ->
          (* gh-ocannl-478: uniform-f32 GEMMs compute in tf32 (m16n16k8, sm_80+) when the numerics
             policy opts in; with the policy off this arm is [None] and the scalar fallback keeps
             full f32 numerics. f32's wmma stride constraint is 4 elements. *)
          mk ~tile:(16, 16, 8) ~marker:"-tf32" ~cvt_tf32:true "nvcuda::wmma::precision::tf32"
            "float" 4 4 80
      | _ -> None

    (* Tensorize-mma T3: cooperative tile-MMA emission for [Low_level.Tile_mma]. Two renderings:

       - The CUDA wmma C++ API for the 16-bit combinations: f16 x f16 -> f32 (the flagship), f16 x
       f16 -> f16, and bf16 x bf16 -> f32 (sm_80+; the [(wmma-bf16)] marker makes [cuda_to_ptx]
       select the arch). 16x16x16 fragment blocks are resident across the whole [k] extent,
       mirroring the Metal emission. - Inline-PTX
       [mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32] for fp8 x fp8 -> f32 (OCANNL's fp8 is
       e5m2), which wmma cannot express. sm_89+ (Ada); the [(mma-fp8)] marker makes [cuda_to_ptx]
       target compute_89, whose PTX the driver forward-JIT-compiles on newer GPUs (e.g. sm_120
       Blackwell GeForce) — deliberately NOT the device arch, where the plain e5m2 encoding gives
       way to the family-specific [kind::f8f6f4] forms.

       The extent-32 lane loop binds threadIdx.x, so the 32 consecutive .x threads reaching the
       statement form the cooperating warp. Transposed operand storage ([ta]/[tb]) loads wmma
       fragments as [col_major] with swapped offset arithmetic; the fp8 path declines it in v1.
       Uniform f32 targets the wmma tf32 shape m16n16k8 on sm_80+ (the [(wmma-tf32)] marker selects
       the arch) when the numerics policy opts in ([Numerics.t.tf32_matmuls], gh-ocannl-478): tf32
       truncates the mantissa to 10 bits, so with the policy off — the default — uniform f32 stays
       on the scalar path with full f32 numerics. Declines (the barrier-bracketed lane-0 fallback
       renders instead) on: other precision combinations, extents not multiples of the intrinsic
       tile, leading dimensions violating wmma's stride constraint (a multiple of 8 elements for
       16-bit types, 4 for f32; the fp8 path loads per-lane bytes and has no stride constraint),
       thread-space operands (per-thread stacks are not a jointly-owned tile), and devices below
       the arch floor. *)
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
          (* Pointer declarations use [typ_of_prec] of the operand's own precision, which coincides
             with [wmma_combo]'s fragment element types (tf32 fragments load from plain [float]
             pointers). *)
          let combo = wmma_combo ~a_prec ~b_prec ~d_prec in
          let loadable = function
            | `Device | `Shared -> true (* generic-address loads cover both *)
            | `Thread | `Fragment _ -> false
          in
          let is_fp8_combo =
            match (a_prec, b_prec, d_prec) with
            | Ops.Fp8_prec _, Ops.Fp8_prec _, Ops.Single_prec _ -> true
            | _ -> false
          in
          if
            is_fp8_combo
            (* Transposed operand storage ([ta]/[tb]) is declined in v1: the per-lane byte gathers
               below assume the roles' own layouts (A row-major, B row-major source gathered into
               the col-major fragment); the scalar fallback keeps transposed fp8 correct. Swapping
               the gather arithmetic per flag is the natural extension. *)
            && (not ta)
            && (not tb)
            && m % 16 = 0
            && n % 8 = 0
            && k % 32 = 0
            && loadable d_space && loadable a_space && loadable b_space
            && min_compute_capability () >= 89
          then
            (* Raw [mma.sync] with the architecture-defined per-lane fragment layouts of m16n8k32
               (PTX ISA "Matrix Fragments for mma.m16n8k32"; fp8 shares the .s8/.u8 layouts). Thread
               [lane], with groupID g = lane>>2 and threadID-in-group t = lane&3, holds: A (16x32,
               row-major source) regs a0..a3 = 4 consecutive bytes at rows {g, g+8} x column groups
               {4t, 4t+16}; B (32x8, column-major fragment from our row-major source) regs b0,b1 =
               rows {4t..4t+3, 16+4t..16+4t+3} down column g, gathered bytewise (stride ldb);
               accumulator D (16x8 f32) regs d0..d3 = rows {g, g+8} x columns {2t, 2t+1}.
               Low-indexed elements sit in low-order bytes of each .b32 register. Byte loads impose
               no alignment or stride constraints and read generic addresses, covering both
               __shared__ tiles and device pointers. *)
            let open PPrint in
            let mt = m / 16 and nt = n / 8 and kt = k / 32 in
            let pack4 name base offs =
              Printf.sprintf
                "unsigned %s = (unsigned)%s[%s] | ((unsigned)%s[%s] << 8) | ((unsigned)%s[%s] << \
                 16) | ((unsigned)%s[%s] << 24);"
                name base offs.(0) base offs.(1) base offs.(2) base offs.(3)
            in
            let consec p0 = [| p0 ^ " + 0"; p0 ^ " + 1"; p0 ^ " + 2"; p0 ^ " + 3" |] in
            let strided p0 stride =
              [|
                p0 ^ " + 0 * " ^ stride;
                p0 ^ " + 1 * " ^ stride;
                p0 ^ " + 2 * " ^ stride;
                p0 ^ " + 3 * " ^ stride;
              |]
            in
            let barrier = "__syncthreads();" in
            let body_lines =
              [
                barrier;
                "unsigned __mma_lid;";
                "asm(\"mov.u32 %0, %%laneid;\" : \"=r\"(__mma_lid));";
                "const int __mma_g = (int)(__mma_lid >> 2);";
                "const int __mma_t = (int)(__mma_lid & 3);";
                Printf.sprintf "for (int __mi = 0; __mi < %d; ++__mi) {" mt;
                Printf.sprintf "  for (int __ni = 0; __ni < %d; ++__ni) {" nt;
                Printf.sprintf
                  "    float *__mma_dr0 = __mma_dp + (__mi * 16 + __mma_g) * %d + __ni * 8 + 2 * \
                   __mma_t;"
                  ldd;
                Printf.sprintf "    float *__mma_dr1 = __mma_dr0 + 8 * %d;" ldd;
                "    float __mma_d0 = __mma_dr0[0], __mma_d1 = __mma_dr0[1];";
                "    float __mma_d2 = __mma_dr1[0], __mma_d3 = __mma_dr1[1];";
                Printf.sprintf "    for (int __ki = 0; __ki < %d; ++__ki) {" kt;
                Printf.sprintf
                  "      const unsigned char *__mma_ar0 = __mma_ap + (__mi * 16 + __mma_g) * %d + \
                   __ki * 32 + 4 * __mma_t;"
                  lda;
                Printf.sprintf "      const unsigned char *__mma_ar1 = __mma_ar0 + 8 * %d;" lda;
                "      " ^ pack4 "__mma_a0" "__mma_ar0" (consec "0");
                "      " ^ pack4 "__mma_a1" "__mma_ar1" (consec "0");
                "      " ^ pack4 "__mma_a2" "__mma_ar0" (consec "16");
                "      " ^ pack4 "__mma_a3" "__mma_ar1" (consec "16");
                Printf.sprintf
                  "      const unsigned char *__mma_br0 = __mma_bp + (__ki * 32 + 4 * __mma_t) * \
                   %d + __ni * 8 + __mma_g;"
                  ldb;
                Printf.sprintf "      const unsigned char *__mma_br1 = __mma_br0 + 16 * %d;" ldb;
                "      " ^ pack4 "__mma_b0" "__mma_br0" (strided "0" (Int.to_string ldb));
                "      " ^ pack4 "__mma_b1" "__mma_br1" (strided "0" (Int.to_string ldb));
                "      asm(\"mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32 \"";
                "          \"{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\"";
                "          : \"+f\"(__mma_d0), \"+f\"(__mma_d1), \"+f\"(__mma_d2), \"+f\"(__mma_d3)";
                "          : \"r\"(__mma_a0), \"r\"(__mma_a1), \"r\"(__mma_a2), \"r\"(__mma_a3), \
                 \"r\"(__mma_b0), \"r\"(__mma_b1));";
                "    }";
                "    __mma_dr0[0] = __mma_d0; __mma_dr0[1] = __mma_d1;";
                "    __mma_dr1[0] = __mma_d2; __mma_dr1[1] = __mma_d3;";
                "  }";
                "}";
                barrier;
              ]
            in
            let cast_ptr name ptr =
              string (Printf.sprintf "const unsigned char *%s = (const unsigned char *)(" name)
              ^^ ptr ^^ string ");"
            in
            let body =
              string "float *__mma_dp = " ^^ d_ptr ^^ semi ^^ hardline ^^ cast_ptr "__mma_ap" a_ptr
              ^^ hardline ^^ cast_ptr "__mma_bp" b_ptr ^^ hardline
              ^^ separate_map hardline string body_lines
            in
            Some
              (group
                 (string (Printf.sprintf "{ /* tile_mma %dx%dx%d (mma-fp8) e5m2 */" m n k)
                 ^^ nest 2 (hardline ^^ body)
                 ^^ hardline ^^ rbrace))
          else
            match combo with
            | None -> None
            | Some
                {
                  wc_ab_typ = ab_typ;
                  wc_acc_typ = acc_typ;
                  wc_tm;
                  wc_tn;
                  wc_tk;
                  wc_ab_ld_mult = ab_ld_mult;
                  wc_d_ld_mult = d_ld_mult;
                  wc_min_cc = min_cc;
                  wc_marker = marker;
                  wc_cvt_tf32 = cvt_tf32;
                } -> (
                let open PPrint in
                let mt = m / wc_tm and nt = n / wc_tn and kt = k / wc_tk in
                let frag kind typ layout =
                  Printf.sprintf "nvcuda::wmma::fragment<nvcuda::wmma::%s, %d, %d, %d, %s%s>" kind
                    wc_tm wc_tn wc_tk typ
                    (match layout with Some l -> ", nvcuda::wmma::" ^ l | None -> "")
                in
                let ptr_decl name typ ptr =
                  string (Printf.sprintf "%s *%s = " typ name) ^^ ptr ^^ semi
                in
                let a_layout = if ta then "col_major" else "row_major" in
                let b_layout = if tb then "col_major" else "row_major" in
                let barrier = "__syncthreads();" in
                (* tf32 fragments load raw f32 bits; the explicit elementwise [__float_to_tf32]
                   performs the mantissa truncation the intrinsic expects (CUDA programming guide
                   requirement — without it the multiplicand bits are implementation-defined). *)
                let cvt_lines frag_expr =
                  if cvt_tf32 then
                    [
                      Printf.sprintf
                        "    for (int __t = 0; __t < %s.num_elements; ++__t) %s.x[__t] = \
                         nvcuda::wmma::__float_to_tf32(%s.x[__t]);"
                        frag_expr frag_expr frag_expr;
                    ]
                  else []
                in
                (* The serial [k] extent of one block statement: per step, the B-row fragment
                   loads, the A fragment loads, and the mma updates of the accumulator array [acc].
                   Shared between the self-contained rendering ([acc] = the block-local
                   [__mma_acc]) and the update-only rendering against a resident fragment array
                   (gh-ocannl-480). *)
                let k_loop_lines ~ab_typ ~acc =
                  [
                    Printf.sprintf "for (int __ki = 0; __ki < %d; ++__ki) {" kt;
                    Printf.sprintf "  %s __mma_bf[%d];" (frag "matrix_b" ab_typ (Some b_layout)) nt;
                    Printf.sprintf "  for (int __ni = 0; __ni < %d; ++__ni) {" nt;
                    (* Transposed storage ([tb]): the stored matrix is the role's transpose —
                       index it at (col, row) and declare the fragment [col_major]; the leading
                       dimension stays the operand's own. Same for [ta] below. *)
                    (if tb then
                       Printf.sprintf
                         "    nvcuda::wmma::load_matrix_sync(__mma_bf[__ni], __mma_bp + __ni * %d \
                          * %d + __ki * %d, %d);"
                         wc_tn ldb wc_tk ldb
                     else
                       Printf.sprintf
                         "    nvcuda::wmma::load_matrix_sync(__mma_bf[__ni], __mma_bp + __ki * %d \
                          * %d + __ni * %d, %d);"
                         wc_tk ldb wc_tn ldb);
                  ]
                  @ cvt_lines "__mma_bf[__ni]"
                  @ [
                      "  }";
                      Printf.sprintf "  for (int __mi = 0; __mi < %d; ++__mi) {" mt;
                      Printf.sprintf "    %s __mma_af;" (frag "matrix_a" ab_typ (Some a_layout));
                      (if ta then
                         Printf.sprintf
                           "    nvcuda::wmma::load_matrix_sync(__mma_af, __mma_ap + __ki * %d * \
                            %d + __mi * %d, %d);"
                           wc_tk lda wc_tm lda
                       else
                         Printf.sprintf
                           "    nvcuda::wmma::load_matrix_sync(__mma_af, __mma_ap + __mi * %d * \
                            %d + __ki * %d, %d);"
                           wc_tm lda wc_tk lda);
                    ]
                  @ cvt_lines "__mma_af"
                  @ [
                      Printf.sprintf "    for (int __ni = 0; __ni < %d; ++__ni) {" nt;
                      Printf.sprintf
                        "      nvcuda::wmma::mma_sync(%s[__mi][__ni], __mma_af, __mma_bf[__ni], \
                         %s[__mi][__ni]);"
                        acc acc;
                      "    }";
                      "  }";
                      "}";
                    ]
                in
                let ab_ok =
                  m % wc_tm = 0
                  && n % wc_tn = 0
                  && k % wc_tk = 0
                  && lda % ab_ld_mult = 0
                  && ldb % ab_ld_mult = 0
                  && loadable a_space && loadable b_space
                  && min_compute_capability () >= min_cc
                in
                match d_space with
                | `Fragment fragment when ab_ok ->
                    (* Update-only steps against the accumulator-fragment array [fragment] declared
                       by the enclosing [mma_fragment_syntax] scope (gh-ocannl-480): no
                       per-statement load/store of [d]. The trailing barrier releases the staged
                       shared tiles for the next serial iteration's cooperative loads. *)
                    let body =
                      ptr_decl "__mma_ap" ("const " ^ typ_of_prec a_prec) a_ptr
                      ^^ hardline
                      ^^ ptr_decl "__mma_bp" ("const " ^ typ_of_prec b_prec) b_ptr
                      ^^ hardline
                      ^^ separate_map hardline string
                           (k_loop_lines ~ab_typ ~acc:fragment @ [ barrier ])
                    in
                    Some
                      (group
                         (string
                            (Printf.sprintf "{ /* tile_mma fragment update %dx%dx%d (wmma%s) */" m
                               n k marker)
                         ^^ nest 2 (hardline ^^ body)
                         ^^ hardline ^^ rbrace))
                | _ when ab_ok && ldd % d_ld_mult = 0 && loadable d_space ->
                    let body_lines =
                      [
                        barrier;
                        Printf.sprintf "%s __mma_acc[%d][%d];" (frag "accumulator" acc_typ None) mt
                          nt;
                        Printf.sprintf "for (int __mi = 0; __mi < %d; ++__mi) {" mt;
                        Printf.sprintf "  for (int __ni = 0; __ni < %d; ++__ni) {" nt;
                        Printf.sprintf
                          "    nvcuda::wmma::load_matrix_sync(__mma_acc[__mi][__ni], __mma_dp + \
                           __mi * %d * %d + __ni * %d, %d, nvcuda::wmma::mem_row_major);"
                          wc_tm ldd wc_tn ldd;
                        "  }";
                        "}";
                      ]
                      @ k_loop_lines ~ab_typ ~acc:"__mma_acc"
                      @ [
                          Printf.sprintf "for (int __mi = 0; __mi < %d; ++__mi) {" mt;
                          Printf.sprintf "  for (int __ni = 0; __ni < %d; ++__ni) {" nt;
                          Printf.sprintf
                            "    nvcuda::wmma::store_matrix_sync(__mma_dp + __mi * %d * %d + __ni \
                             * %d, __mma_acc[__mi][__ni], %d, nvcuda::wmma::mem_row_major);"
                            wc_tm ldd wc_tn ldd;
                          "  }";
                          "}";
                          barrier;
                        ]
                    in
                    let body =
                      ptr_decl "__mma_dp" (typ_of_prec d_prec) d_ptr
                      ^^ hardline
                      ^^ ptr_decl "__mma_ap" ("const " ^ typ_of_prec a_prec) a_ptr
                      ^^ hardline
                      ^^ ptr_decl "__mma_bp" ("const " ^ typ_of_prec b_prec) b_ptr
                      ^^ hardline
                      ^^ separate_map hardline string body_lines
                    in
                    Some
                      (group
                         (string
                            (Printf.sprintf "{ /* tile_mma %dx%dx%d (wmma%s) */" m n k marker)
                         ^^ nest 2 (hardline ^^ body)
                         ^^ hardline ^^ rbrace))
                | _ -> None))

    (* Cross-[k_o] accumulator residency (gh-ocannl-480), following the Metal emission: the marked
       local tile becomes a wmma accumulator-fragment array declared once, loaded from the backing
       target before the serial reduction body and stored after it; the nested [Tile_mma] sees
       [`Fragment] and emits update-only mma steps. The acceptance conditions mirror [mma_syntax]'s
       wmma arm exactly (via [wmma_combo] and the same extent/stride/space/arch checks), so an
       accepted scope never strands its inner call. The fp8 inline-PTX combination declines: its
       accumulator lives in per-lane f32 registers with the m16n8k32 layout, not wmma fragments, so
       it keeps the per-[k_o] rendering through the caller's target-aliasing path. *)
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
          let loadable = function
            | `Device | `Shared -> true (* generic-address loads cover both *)
            | `Thread | `Fragment _ -> false
          in
          match wmma_combo ~a_prec ~b_prec ~d_prec with
          | Some
              {
                wc_acc_typ = acc_typ;
                wc_tm;
                wc_tn;
                wc_tk;
                wc_ab_ld_mult = ab_ld_mult;
                wc_d_ld_mult = d_ld_mult;
                wc_min_cc = min_cc;
                _;
              }
            when m % wc_tm = 0
                 && n % wc_tn = 0
                 && k % wc_tk = 0
                 && lda % ab_ld_mult = 0
                 && ldb % ab_ld_mult = 0
                 && ldd % d_ld_mult = 0
                 && loadable d_space && loadable a_space && loadable b_space
                 && min_compute_capability () >= min_cc ->
              let open PPrint in
              let mt = m / wc_tm and nt = n / wc_tn in
              let barrier = "__syncthreads();" in
              let acc_frag =
                Printf.sprintf "nvcuda::wmma::fragment<nvcuda::wmma::accumulator, %d, %d, %d, %s>"
                  wc_tm wc_tn wc_tk acc_typ
              in
              (* Bracketing barriers: sibling statements (the zeroing of [d], cooperative staging)
                 execute lane-partitioned, so the fragment loads must observe the other lanes'
                 writes, and later statements the stores. *)
              let lines_before =
                [
                  barrier;
                  Printf.sprintf "%s %s[%d][%d];" acc_frag fragment mt nt;
                  Printf.sprintf "for (int __mi = 0; __mi < %d; ++__mi) {" mt;
                  Printf.sprintf "  for (int __ni = 0; __ni < %d; ++__ni) {" nt;
                  Printf.sprintf
                    "    nvcuda::wmma::load_matrix_sync(%s[__mi][__ni], __mma_dp + __mi * %d * %d \
                     + __ni * %d, %d, nvcuda::wmma::mem_row_major);"
                    fragment wc_tm ldd wc_tn ldd;
                  "  }";
                  "}";
                  "/* wmma fragment reduction body begins */";
                ]
              in
              let lines_after =
                [
                  "/* wmma fragment reduction body ends */";
                  Printf.sprintf "for (int __mi = 0; __mi < %d; ++__mi) {" mt;
                  Printf.sprintf "  for (int __ni = 0; __ni < %d; ++__ni) {" nt;
                  Printf.sprintf
                    "    nvcuda::wmma::store_matrix_sync(__mma_dp + __mi * %d * %d + __ni * %d, \
                     %s[__mi][__ni], %d, nvcuda::wmma::mem_row_major);"
                    wc_tm ldd wc_tn fragment ldd;
                  "  }";
                  "}";
                  barrier;
                ]
              in
              let d_decl =
                string (Printf.sprintf "%s *__mma_dp = " (typ_of_prec d_prec)) ^^ d_ptr ^^ semi
              in
              Some
                (group
                   (string (Printf.sprintf "{ /* wmma fragment %dx%d across k_o */" m n)
                   ^^ nest 2
                        (hardline ^^ d_decl ^^ hardline
                        ^^ separate_map hardline string lines_before
                        ^^ hardline ^^ body () ^^ hardline
                        ^^ separate_map hardline string lines_after)
                   ^^ hardline ^^ rbrace))
          | _ -> None)

    let rec binop_syntax prec v =
      (* TODO: consider using binop_syntax inherited from Pure_C_config and overriding only where
         different. *)
      let open PPrint in
      let f op_str v1 v2 =
        group
          (parens (v1 ^^ string (" " ^ op_str) ^^ ifflat (space ^^ v2) (nest 2 (break 1 ^^ v2))))
      in
      let func fn v1 v2 =
        group (string fn ^^ parens (v1 ^^ comma ^^ ifflat (space ^^ v2) (nest 2 (break 1 ^^ v2))))
      in
      match (v, prec) with
      | Ops.Arg1, _ -> invalid_arg "Cuda_backend.binop_syntax: Arg1 is not an operator"
      | Arg2, _ -> invalid_arg "Cuda_backend.binop_syntax: Arg2 is not an operator"
      | _, Ops.Void_prec -> invalid_arg "Cuda_backend.binop_syntax: Void precision"
      | Threefry4x32_crypto, _ -> (
          (* Threefry4x32_crypto must output to uint4x32 precision *)
          match prec with
          | Ops.Uint4x32_prec _ -> func "arrayjit_threefry4x32_crypto"
          | _ ->
              raise
              @@ Utils.User_error
                   (Printf.sprintf
                      "CUDA backend: Threefry4x32_crypto requires target precision to be uint4x32, \
                       but got %s"
                      (Ops.prec_string prec)))
      | Threefry4x32_light, _ -> (
          (* Threefry4x32_light must output to uint4x32 precision *)
          match prec with
          | Ops.Uint4x32_prec _ -> func "arrayjit_threefry4x32_light"
          | _ ->
              raise
              @@ Utils.User_error
                   (Printf.sprintf
                      "CUDA backend: Threefry4x32_light requires target precision to be uint4x32, \
                       but got %s"
                      (Ops.prec_string prec)))
      | _, Fp8_prec _ ->
          (* __nv_fp8_e5m2 defines no arithmetic operators and all its constructors and conversion
             operators are explicit (cuda_fp8.hpp), so bridge fp8 math through float, mirroring the
             CC backend's fp8 handling. *)
          fun v1 v2 ->
            let fl v = string "(float)" ^^ parens v in
            group (string "(__nv_fp8_e5m2)" ^^ parens (binop_syntax Ops.single v (fl v1) (fl v2)))
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
          invalid_arg "Cuda_backend.binop_syntax: ToPowOf not supported for integer precisions"
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
          fun v1 v2 ->
            group
              (parens
                 (group (parens (v1 ^^ string " > 0.0h"))
                 ^^ ifflat
                      (space ^^ string "?" ^^ space ^^ v2 ^^ space ^^ string ":" ^^ space
                     ^^ string "0.0h")
                      (nest 2
                         (break 1 ^^ string "?" ^^ space ^^ v2 ^^ break 1 ^^ string ":" ^^ space
                        ^^ string "0.0h"))))
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
      | Max, Bfloat16_prec _ ->
          (* FIXME: This might be wrong, definitely verify and maybe fix, here and elsewhere *)
          func "__hmax"
      | Min, Byte_prec _ -> func "min"
      | Min, Half_prec _ -> func "__hmin"
      | Min, Double_prec _ -> func "fmin"
      | Min, Single_prec _ -> func "fminf"
      | Min, Uint16_prec _ -> func "min"
      | Min, Int32_prec _ -> func "min"
      | Min, Int64_prec _ -> func "min"
      | Min, Uint4x32_prec _ -> func "min"
      | Min, Bfloat16_prec _ -> func "__hmin"
      | Mod, Byte_prec _ -> f "%"
      | Mod, _ -> func "fmod"
      | Cmplt, _ -> f "<"
      | Cmpne, _ -> f "!="
      | Cmpeq, _ -> f "=="
      | Or, _ -> f "||"
      | And, _ -> f "&&"
      | ToPowOf, (Uint32_prec _ | Uint64_prec _) ->
          invalid_arg "Cuda_backend.binop_syntax: ToPowOf not supported for integer precisions"
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
      match (v, prec) with
      | Ops.Identity, _ -> f "" ""
      | Uint4x32_to_prec_uniform1, Ops.Uint4x32_prec _ ->
          invalid_arg
            "Cuda_backend.unop_syntax: Uint4x32_to_prec_uniform1 not supported for Uint4x32"
      (* Heterogeneous op: the argument is uint4x32 whatever the result precision, so it must stay
         ahead of the fp8 float-bridging below; the fp8 builtin returns __nv_fp8_e5m2. *)
      | Uint4x32_to_prec_uniform1, _ -> func ("uint4x32_to_" ^ Ops.prec_string prec ^ "_uniform")
      | _, Ops.Fp8_prec _ ->
          (* __nv_fp8_e5m2 defines no arithmetic operators and all its constructors and conversion
             operators are explicit (cuda_fp8.hpp), so bridge fp8 math through float, mirroring the
             CC backend's fp8 handling. *)
          fun expr ->
            group
              (string "(__nv_fp8_e5m2)"
              ^^ parens (unop_syntax Ops.single v (string "(float)" ^^ parens expr)))
      | Relu, Ops.Single_prec _ -> f "fmaxf(0.0, " ")"
      | Relu, Ops.Half_prec _ -> f "__hmax_nan(__ushort_as_half((unsigned short)0x0000U), " ")"
      | Relu, Ops.Byte_prec _ -> f "fmax(0, " ")"
      | Relu, _ -> f "fmax(0.0, " ")"
      | Satur01, Byte_prec _ -> f "fmax(0, fmin(1, " "))"
      | Satur01, Half_prec _ ->
          f
            "__hmax_nan(__ushort_as_half((unsigned short)0x0000U), \
             __hmin_nan(__ushort_as_half((unsigned short)0x3C00U), "
            "))"
      | Satur01, Single_prec _ -> f "fmaxf(0.0f, fminf(1.0f, " "))"
      | Satur01, _ -> f "fmax(0.0, fmin(1.0, " "))"
      | Exp, Half_prec _ -> func "hexp"
      | Exp, Double_prec _ -> func "exp"
      | Exp, _ -> func "expf"
      | Log, Half_prec _ -> func "hlog"
      | Log, Double_prec _ -> func "log"
      | Log, _ -> func "logf"
      | Exp2, Half_prec _ -> func "hexp2"
      | Exp2, Double_prec _ -> func "exp2"
      | Exp2, _ -> func "exp2f"
      | Log2, Half_prec _ -> func "hlog2"
      | Log2, Double_prec _ -> func "log2"
      | Log2, _ -> func "log2f"
      | Sin, Half_prec _ -> func "hsin"
      | Sin, Double_prec _ -> func "sin"
      | Sin, _ -> func "sinf"
      | Cos, Half_prec _ -> func "hcos"
      | Cos, Double_prec _ -> func "cos"
      | Cos, _ -> func "cosf"
      | Sqrt, Half_prec _ -> func "hsqrt"
      | Sqrt, Double_prec _ -> func "sqrt"
      | Sqrt, _ -> func "sqrtf"
      | Recip, Byte_prec _ ->
          invalid_arg "Cuda_backend.unop_syntax: Recip not supported for byte/integer precisions"
      | Recip, Half_prec _ -> func "hrcp"
      | Recip, Single_prec _ -> f "(1.0f / (" "))"
      | Recip, Double_prec _ -> f "(1.0 / (" "))"
      | Recip, _ -> f "(1 / (" "))"
      | Recip_sqrt, Byte_prec _ ->
          invalid_arg
            "Cuda_backend.unop_syntax: Recip_sqrt not supported for byte/integer precisions"
      | Recip_sqrt, Half_prec _ -> func "hrsqrt"
      | Recip_sqrt, Double_prec _ -> f "(1.0 / sqrt(" "))"
      | Recip_sqrt, Single_prec _ -> f "(1.0f / sqrtf(" "))"
      | Recip_sqrt, _ -> f "(1 / sqrtf(" "))"
      | Neg, _ -> f "(-(" "))"
      | Trunc, Double_prec _ -> func "trunc"
      | Trunc, _ -> func "truncf"
      | Tanh_approx, Byte_prec _ ->
          invalid_arg
            "Cuda_backend.unop_syntax: Tanh_approx not supported for byte/integer precisions"
      | Tanh_approx, Half_prec _ -> func "htanh_approx"
      | Tanh_approx, Single_prec _ -> func "__tanhf"
      | Tanh_approx, _ -> func "tanh"
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
          (* __nv_fp8_e5m2 defines no arithmetic operators and all its constructors and conversion
             operators are explicit (cuda_fp8.hpp), so bridge fp8 math through float, mirroring the
             CC backend's fp8 handling. *)
          fun v1 v2 v3 ->
            let fl v = string "(float)" ^^ parens v in
            group
              (string "(__nv_fp8_e5m2)"
              ^^ parens (ternop_syntax Ops.single v (fl v1) (fl v2) (fl v3)))
      | Ops.Where, _ ->
          (* The whole ternary must be parenthesized, not just the condition: C's [?:] binds looser
             than the surrounding arithmetic, so for an expression like [where(c,a,b) + 1] the
             trailing [+ 1] would otherwise be absorbed into the else-branch ([c ? a : b + 1]),
             silently dropping it from the then-branch. This off-by-one surfaced only on CUDA
             (task-04f97340): CC wraps the conditional in [(... ? ... : ...)] via
             [Ops.ternop_c_syntax] and Metal emits a fully-bracketed [select(...)] call. *)
          fun v1 v2 v3 -> group (parens (parens v1 ^^ string " ? " ^^ v2 ^^ string " : " ^^ v3))
      | FMA, Ops.Half_prec _ -> func "__hfma"
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
      | Double_prec _, Half_prec _ -> ("__double2half(", ")")
      | Single_prec _, Half_prec _ -> ("__float2half(", ")")
      | Byte_prec _, Half_prec _ -> ("__ushort2half_rn((unsigned short int)", ")")
      | Double_prec _, Uint4x32_prec _ -> ("double_to_uint4x32(", ")")
      | Single_prec _, Uint4x32_prec _ -> ("single_to_uint4x32(", ")")
      (* __nv_fp8_e5m2's constructors are all explicit, so the [.v[0]] arm below would not
         implicitly convert on assignment; spell out the (numeric) constructor call. *)
      | Uint4x32_prec _, Fp8_prec _ -> ("(__nv_fp8_e5m2)((", ").v[0])")
      | Uint4x32_prec _, _ -> ("", ".v[0]")
      | Byte_prec _, Uint4x32_prec _ -> ("byte_to_uint4x32(", ")")
      | Uint16_prec _, Uint4x32_prec _ -> ("uint16_to_uint4x32(", ")")
      | Bfloat16_prec _, Uint4x32_prec _ -> ("bfloat16_to_uint4x32(", ")")
      | Half_prec _, Uint4x32_prec _ -> ("half_to_uint4x32(", ")")
      | Fp8_prec _, Uint4x32_prec _ -> ("fp8_to_uint4x32(", ")")
      | ( Fp8_prec _,
          (Byte_prec _ | Uint16_prec _ | Int32_prec _ | Uint32_prec _ | Int64_prec _ | Uint64_prec _)
        ) ->
          (* __nv_fp8_e5m2 has no integer conversion operators: a C-style cast to an integer type
             would resolve through the saturating [operator unsigned char] (wrong for negative
             values). Convert via float, like the CC backend. *)
          ("(" ^ typ_of_prec to_ ^ ")((float)(", "))")
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

  let%diagn2_sexp compile ~name bindings ({ Low_level.traced_store; _ } as lowered) =
    (* TODO: The following link seems to claim it's better to expand into loops than use memset.
       https://stackoverflow.com/questions/23712558/how-do-i-best-initialize-a-local-memory-array-to-0 *)
    let module Syntax = C_syntax.C_syntax (Cuda_syntax_config (struct
      let procs = [| lowered |]
    end))
    in
    let idx_params = Indexing.bound_symbols bindings in
    let kparams, proc_doc, launch = Syntax.compile_proc ~name idx_params lowered in
    let cuda_includes =
      {|#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

/* Define math constants that would normally come from <math.h> */
#ifndef INFINITY
#define INFINITY __int_as_float(0x7f800000)
#endif
#ifndef NAN
#define NAN __int_as_float(0x7fffffff)
#endif|}
      ^
      if Utils.debug_log_from_routines () then
        "\n__device__ int printf (const char * format, ... );"
      else ""
    in
    let source =
      Syntax.filter_and_prepend_builtins ~includes:cuda_includes ~builtins:Builtins_cuda.builtins
        ~proc_doc
    in
    let ptx = cuda_to_ptx ~name source in
    { traced_store; ptx; kparams; bindings; name; launch }

  let%diagn2_sexp compile_batch ~names bindings lowereds =
    let module Syntax = C_syntax.C_syntax (Cuda_syntax_config (struct
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
    let cuda_includes =
      {|#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

/* Define math constants that would normally come from <math.h> */
#ifndef INFINITY
#define INFINITY __int_as_float(0x7f800000)
#endif
#ifndef NAN
#define NAN __int_as_float(0x7fffffff)
#endif|}
      ^
      if Utils.debug_log_from_routines () then
        "\n__device__ int printf (const char * format, ... );"
      else ""
    in
    let source =
      Syntax.filter_and_prepend_builtins ~includes:cuda_includes ~builtins:Builtins_cuda.builtins
        ~proc_doc:final_doc
    in

    let name : string =
      String.(
        strip ~drop:(equal_char '_')
        @@ common_prefix (Array.to_list names |> List.concat_map ~f:Option.to_list))
    in
    let ptx = cuda_to_ptx ~name source in
    let traced_stores = Array.map lowereds ~f:(Option.map ~f:(fun l -> l.Low_level.traced_store)) in
    let kparams_and_names = Array.map kparams_and_docs ~f:(Option.map ~f:fst) in
    { traced_stores; ptx; kparams_and_names; bindings }

  let get_global_run_id =
    let next_id = ref 0 in
    fun () ->
      Int.incr next_id;
      if !next_id < 0 then next_id := 0;
      !next_id

  let link_proc ~prior_context ~name ~(kparams : (string * kparam_source) list)
      ~(launch : Low_level.launch_dims) ~ctx_buffers lowered_bindings run_module =
    let func = Cu.Module.get_function run_module ~name in
    let device = prior_context.device in
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
      let module S = Cu.Stream in
      let args : S.kernel_param list =
        (* TODO: should we prohibit or warn about local-only tensors that are in
           prior_context.ctx_buffers? *)
        List.map kparams ~f:(function
          | _name, Kparam_ptr tn ->
              let loc = Option.value_exn ~here:[%here] @@ Map.find ctx_buffers tn in
              let base = Map.find_exn ctx_bases tn in
              S.Tensor_at (Cu.Deviceptr.offset base ~bytes:loc.offset)
          | _name, Log_file_name -> S.Int log_id
          | _name, Merge_buffer ->
              let loc = Option.value_exn ~here:[%here] !(device.merge_buffer) in
              let base = Slab.resolve_pool device loc in
              S.Tensor_at (Cu.Deviceptr.offset base ~bytes:loc.offset)
          | _name, Static_idx s ->
              let i = Indexing.find_exn lowered_bindings s in
              (* Shared bind-time validation: negativity, range -- inclusive [0, range] for symbolic
                 extents (gh-490), strict [0, range) for indices -- and index width. *)
              Indexing.validate_bound_value ~width64:Utils.settings.large_models s !i;
              S.Int !i
          | _name, (Kparam_pool_slab _ | Kparam_pool_slots _) ->
              (* The CUDA backend uses per-tnode pointer params ([`Per_param] codegen); only the
                 Metal backend emits the pooled slab / slot parameters. *)
              invalid_arg
                "Cuda_backend.link: unexpected pooled kparam (CUDA uses per-tnode pointers)")
      in
      set_ctx @@ ctx_of prior_context;
      [%log "launching the kernel"];
      (* Stdio.printf "launching %s\n" name; *)
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
    let run_module = Cu.Module.load_data_ex code.ptx (run_options ()) in
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
    let run_module = Cu.Module.load_data_ex code_batch.ptx (run_options ()) in
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

  (* CUDA kernel launches on one stream already execute in FIFO order, so the generic event chain is
     correct; a cheaper plain-sequence task is a possible follow-up (events are ~free there). *)
  let sequence_segments _context ~name:_ _tasks = None

  let get_global_debug_info () =
    Sexp.message "cuda_global_debug"
      [ ("live_streams", [%sexp_of: int] @@ Cu.Stream.get_total_live_streams ()) ]

  let static_properties () =
    let device_properties =
      Array.init (num_devices ()) ~f:(fun ordinal ->
          let dev = Cu.Device.get ~ordinal in
          let attributes = Cu.Device.get_attributes dev in
          let props =
            [
              ("device_name", Sexp.Atom attributes.name);
              ("device_ordinal", [%sexp_of: int] ordinal);
              ("multiprocessor_count", [%sexp_of: int] attributes.multiprocessor_count);
              ("clock_rate", [%sexp_of: int] attributes.clock_rate);
              ("async_engine_count", [%sexp_of: int] attributes.async_engine_count);
              ("compute_capability_major", [%sexp_of: int] attributes.compute_capability_major);
              ("compute_capability_minor", [%sexp_of: int] attributes.compute_capability_minor);
              ("max_threads_per_block", [%sexp_of: int] attributes.max_threads_per_block);
              ("unified_addressing", [%sexp_of: bool] attributes.unified_addressing);
            ]
          in
          Sexp.message "device" props)
    in
    Sexp.List (Sexp.Atom "cuda_devices" :: Array.to_list device_properties)

  (* Conservative per-workgroup device limits for the schedule layer (schedule-ir-optops §6):
     minimum across devices, so code compiled once is valid wherever it links. *)
  (* Memoized behind [lazy]: driver init and device enumeration must not run at backend-module
     initialization ([num_devices] forces [ensure_initialized]). *)
  let hardware_limits =
    let limits =
      lazy
        (let attrs =
           Array.init (num_devices ()) ~f:(fun ordinal ->
               Cu.Device.get_attributes (Cu.Device.get ~ordinal))
         in
         let min_over f = Array.map attrs ~f |> Array.min_elt ~compare:Int.compare in
         {
           Backend_intf.max_threads_per_workgroup =
             min_over (fun (a : Cu.Device.attributes) -> a.max_threads_per_block);
           max_workgroup_memory_bytes =
             min_over (fun (a : Cu.Device.attributes) -> a.max_shared_memory_per_block);
           (* Tensor cores (tensorize-mma T3): the 32-thread warp cooperates on 16x16x16 wmma tiles
              from sm_70 up (fp8 [mma.sync] tiles are 16x8x32, sm_89+ — [mma_tile] is the
              autotuner's divisibility filter and stays at the wmma shape). Precision combinations
              are decided per call by [mma_syntax]. *)
           mma =
             (if min_compute_capability () >= 70 then
                Some
                  {
                    Backend_intf.mma_simd_width = 32;
                    mma_tile = (16, 16, 16);
                    mma_format_tiles =
                      [
                        ((Backend_intf.Mma_f16, Backend_intf.Mma_f16), (16, 16, 16));
                        ((Backend_intf.Mma_bf16, Backend_intf.Mma_bf16), (16, 16, 16));
                        ((Backend_intf.Mma_fp8_e5m2, Backend_intf.Mma_fp8_e5m2), (16, 8, 32));
                        ((Backend_intf.Mma_tf32, Backend_intf.Mma_tf32), (16, 16, 8));
                      ];
                  }
              else None);
           simd_vector_bytes = 0;
           (* Advisory roofline envelope (gh-ocannl-491): documented rough constants for the sm_70+
              discrete-GPU class (RTX-30/40 mid-range: ~15 fp32 TFLOP/s, ~450 GB/s). Per-device
              queries (SM count x clock, memory clock x bus width) are calibration follow-up work;
              the model only ranks, so class-level numbers suffice. *)
           peak_flops = Some 1.5e13;
           peak_memory_bandwidth = Some 4.5e11;
         })
    in
    fun () -> Lazy.force limits

  let get_debug_info (device : device) =
    let tot, unr, unf = Cu.Stream.total_unreleased_unfinished_delimited_events device.runner in
    let i2s = [%sexp_of: int] in
    Sexp.message "cuda_stream_debug"
      [ ("total_events", i2s tot); ("unreleased_events", i2s unr); ("unfinished_events", i2s unf) ]
end
