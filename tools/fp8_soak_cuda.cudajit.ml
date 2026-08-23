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
let available = true

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

type state = {
  device : Cu.Device.t;
  attrs : Cu.Device.attributes;
  stream : Cu.Stream.t;
  narrow_f32 : Cu.Module.func;
  narrow_f64 : Cu.Module.func;
  mutable buffer : (Cu.Deviceptr.t * int) option; (* pointer and its size in bytes *)
}

let state = ref None

let init () =
  match !state with
  | Some st -> st
  | None ->
      Cu.init ();
      let device = Cu.Device.get ~ordinal:0 in
      let ctx = Cu.Context.get_primary device in
      Cu.Context.set_current ctx;
      let attrs = Cu.Device.get_attributes device in
      let arch =
        Printf.sprintf "--gpu-architecture=compute_%d%d" attrs.compute_capability_major
          attrs.compute_capability_minor
      in
      let includes =
        if Stdlib.Sys.file_exists "/usr/local/cuda/include" then [ "-I/usr/local/cuda/include" ]
        else []
      in
      let ptx =
        Nvrtc.compile_to_ptx ~cu_src:source ~name:"ocannl_fp8_soak.cu"
          ~options:(includes @ [ arch ])
          ~with_debug:false
      in
      let m = Cu.Module.load_data_ex ptx [] in
      let st =
        {
          device;
          attrs;
          stream = Cu.Stream.create ();
          narrow_f32 = Cu.Module.get_function m ~name:"ocannl_vendor_narrow_f32";
          narrow_f64 = Cu.Module.get_function m ~name:"ocannl_vendor_narrow_f64";
          buffer = None;
        }
      in
      state := Some st;
      st

let describe () =
  let st = init () in
  Printf.sprintf "%s (compute capability %d.%d, %d SMs)" st.attrs.name
    st.attrs.compute_capability_major st.attrs.compute_capability_minor
    st.attrs.multiprocessor_count

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
