(* The CUDA arm of the fp8 soak (gh-ocannl-657): narrow every input with [__nv_fp8_e5m2] and hand
   the codes back. Nothing here knows what the answers are compared against -- fp8_soak.ml owns the
   comparison, so the HIP arm beside this one is the same shape with a different vendor type.

   The kernel casts to the vendor type exactly as [Cuda_backend]'s [convert_precision] emits it:
   [(__nv_fp8_e5m2)x] for a float and for a double, which is a saturating round-to-nearest-even
   conversion done by the hardware. Compiled WITHOUT [--use_fast_math] (which the backend does pass)
   because fast math is about arithmetic and this program does none: the conversion is the whole
   kernel, and the soak should not have to reason about what a relaxed floating-point mode does to
   the thing being measured. *)

open Base
module Cu = Cuda

let name = "cuda"
let vendor_type = "__nv_fp8_e5m2"

type bytes_buf = (int, Stdlib.Bigarray.int8_unsigned_elt, Stdlib.Bigarray.c_layout) Stdlib.Bigarray.Array1.t

let source =
  {|
#include <cuda_fp8.h>

extern "C" __global__ void ocannl_vendor_narrow_f32(unsigned long long base,
                                                    unsigned long long count,
                                                    unsigned char *out) {
  unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
  for (unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
       i < count; i += stride) {
    unsigned int u = (unsigned int)(base + i);
    float x = __uint_as_float(u);
    __nv_fp8_e5m2 v = (__nv_fp8_e5m2)x;
    out[i] = v.__x;
  }
}

extern "C" __global__ void ocannl_vendor_narrow_f64(unsigned long long base,
                                                    unsigned long long count,
                                                    unsigned int l0, unsigned int l1,
                                                    unsigned int l2, unsigned int l3,
                                                    unsigned char *out) {
  unsigned int lows[4] = {l0, l1, l2, l3};
  unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
  for (unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
       i < count; i += stride) {
    unsigned long long hi = base + i;
    for (int k = 0; k < 4; ++k) {
      unsigned long long u = (hi << 32) | (unsigned long long)lows[k];
      double d = __longlong_as_double((long long)u);
      __nv_fp8_e5m2 v = (__nv_fp8_e5m2)d;
      out[(4 * i) + k] = v.__x;
    }
  }
}
|}

(* Everything the sweep needs, held for the whole of it. The two lifetime fields are not
   decoration: [Cu.Context.get_primary] finalizes with [cuDevicePrimaryCtxRelease], and the
   underlying context is RESET once the last reference goes -- so a context dropped after [init]
   returns can be collected mid-sweep and take the device allocation and the loaded code with it.
   [Cuda_backend] keeps its [primary_context] in the device record for exactly this reason.
   [kernel_module] is belt and braces: cudajit's [get_function] already documents that the returned
   [func] retains its module, so this one is about saying the lifetime rather than inferring it from
   a binding's internals -- and about the HIP arm, where the same reasoning has to hold. *)
type state = {
  context : Cu.Context.t;
  kernel_module : Cu.Module.t;
  attrs : Cu.Device.attributes;
  stream : Cu.Stream.t;
  narrow_f32 : Cu.Module.func;
  narrow_f64 : Cu.Module.func;
  options : string list; (* what nvrtc was given, so a run record says what it compiled *)
  mutable buffer : (Cu.Deviceptr.t * int) option; (* pointer and its size in bytes *)
}

(* Which [--gpu-architecture] the kernel is built for, which decides WHAT IS BEING MEASURED.
   [cuda_fp8.hpp] guards its conversions with `#if __CUDA_ARCH__ >= 890`: at or above sm_89 the cast
   becomes the hardware `cvt` instruction, below it the header's own software emulation. So the two
   policies sweep two different things, and both are worth having:

   - [`Device] targets this GPU's own compute capability, which is what "verify the codec against
     the HARDWARE" means -- gh-ocannl-646's lesson, and the only setting that exercises the
     instruction a kernel actually runs.
   - [`Backend] takes [Cuda_backend.gpu_arch_options], the repo's marker-driven policy, which for a
     source with no arch markers (like this one) passes NO architecture at all and so gets nvrtc's
     default target -- below 890, hence the software path. That is what an OCANNL fp8 kernel without
     tensor-core markers is compiled with today, so it is the honest answer to "does the codec agree
     with what the backend emits".

   Default [`Device]: the issue this program exists for asks about the hardware. *)
type arch_policy = [ `Device | `Backend ]

let arch_policy : arch_policy ref = ref `Device

(* Must be called before the first sweep: the module is compiled once, on first use. *)
let set_arch_policy p = arch_policy := p

let state = ref None

(* Whether this BOX can run the arm, not whether the build has it: an opam switch with both
   `cudajit` and `hipjit` in it has both arms compiled, and on a machine with one kind of GPU the
   default selection must not pick the other -- least of all after finishing the first vendor's
   several-minute sweep. The reason travels with the answer so a skip is never silent. *)
let probe () =
  match
    Cu.init ();
    Cu.Device.get_count ()
  with
  | 0 -> Error "cudajit is linked, but the CUDA driver reports no device"
  | _ -> Ok ()
  | exception e -> Error ("cudajit is linked, but CUDA initialization failed: " ^ Exn.to_string e)

let init () =
  match !state with
  | Some st -> st
  | None ->
      Cu.init ();
      let device = Cu.Device.get ~ordinal:0 in
      let ctx = Cu.Context.get_primary device in
      Cu.Context.set_current ctx;
      let attrs = Cu.Device.get_attributes device in
      (* Both option groups come from [Cuda_backend], not from a local guess. The include discovery
         especially: this kernel `#include`s <cuda_fp8.h>, and a soak that looked only in
         /usr/local/cuda would probe "ready" and then fail to compile on every Windows installation
         and every Linux one outside that prefix (Codex P2 on PR #463). The arch policy is shared
         for the same reason -- one nvrtc caller disagreeing with the backend about which
         [--gpu-architecture] a source needs is how a soak comes to measure something the backend
         never emits. *)
      let device_cc =
        (attrs.compute_capability_major * 10) + attrs.compute_capability_minor
      in
      let options =
        Cuda_backend.cuda_include_options ()
        @
        match !arch_policy with
        | `Device -> [ Printf.sprintf "--gpu-architecture=compute_%d" device_cc ]
        | `Backend -> Cuda_backend.gpu_arch_options ~device_cc source
      in
      let ptx =
        Nvrtc.compile_to_ptx ~cu_src:source ~name:"ocannl_fp8_soak.cu" ~options ~with_debug:false
      in
      let kernel_module = Cu.Module.load_data_ex ptx [] in
      let st =
        {
          context = ctx;
          kernel_module;
          attrs;
          options;
          stream = Cu.Stream.create ();
          narrow_f32 = Cu.Module.get_function kernel_module ~name:"ocannl_vendor_narrow_f32";
          narrow_f64 = Cu.Module.get_function kernel_module ~name:"ocannl_vendor_narrow_f64";
          buffer = None;
        }
      in
      state := Some st;
      st

let describe () =
  let st = init () in
  (* The nvrtc options are part of the answer, not decoration: which [--gpu-architecture] the kernel
     was built for decides whether [(__nv_fp8_e5m2)x] became a hardware conversion or the header's
     software fallback, and a soak whose record does not say which was measured is asking to be
     re-derived later. They come from [Cuda_backend], so what is printed is also what the backend
     would use for a source with these markers. *)
  Printf.sprintf "%s (compute capability %d.%d, %d SMs); nvrtc options: %s" st.attrs.name
    st.attrs.compute_capability_major st.attrs.compute_capability_minor
    st.attrs.multiprocessor_count
    (if List.is_empty st.options then "(none)" else String.concat ~sep:" " st.options)

(* One allocation, grown on demand and reused across chunks: 64 chunk-sized [cuMemAlloc]s in a row
   is a way to run into fragmentation on a GPU another process is also using. *)
let device_buffer st bytes =
  match st.buffer with
  | Some (ptr, size) when size >= bytes -> ptr
  | prev ->
      Option.iter prev ~f:(fun (ptr, _) -> Cu.Deviceptr.mem_free ptr);
      let ptr = Cu.Deviceptr.mem_alloc ~size_in_bytes:bytes in
      st.buffer <- Some (ptr, bytes);
      ptr

let block_dim = 256

(* Enough blocks to fill the device several times over, but not one per element: the kernels are
   grid-stride loops, so the launch shape is a scheduling choice rather than a correctness one. *)
let grid_dim st = Int.max 1 (st.attrs.multiprocessor_count * 32)

let fetch st ptr (out : bytes_buf) count =
  Cu.Stream.memcpy_D_to_H ~length:count
    ~dst:(Stdlib.Bigarray.genarray_of_array1 out)
    ~src:ptr st.stream;
  Cu.Stream.synchronize st.stream

let narrow_f32 ~base ~count (out : bytes_buf) =
  let st = init () in
  let ptr = device_buffer st count in
  Cu.Stream.launch_kernel st.narrow_f32 ~grid_dim_x:(grid_dim st) ~block_dim_x:block_dim
    ~shared_mem_bytes:0 st.stream
    [ Cu.Stream.Size_t (Unsigned.Size_t.of_int base); Cu.Stream.Size_t (Unsigned.Size_t.of_int count); Cu.Stream.Tensor ptr ];
  fetch st ptr out count

let narrow_f64 ~base ~count ~lows (out : bytes_buf) =
  let st = init () in
  let bytes = 4 * count in
  let ptr = device_buffer st bytes in
  Cu.Stream.launch_kernel st.narrow_f64 ~grid_dim_x:(grid_dim st) ~block_dim_x:block_dim
    ~shared_mem_bytes:0 st.stream
    [
      Cu.Stream.Size_t (Unsigned.Size_t.of_int base);
      Cu.Stream.Size_t (Unsigned.Size_t.of_int count);
      Cu.Stream.Int lows.(0);
      Cu.Stream.Int lows.(1);
      Cu.Stream.Int lows.(2);
      Cu.Stream.Int lows.(3);
      Cu.Stream.Tensor ptr;
    ];
  fetch st ptr out bytes
