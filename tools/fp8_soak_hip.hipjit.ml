(* The HIP arm of the fp8 soak (gh-ocannl-657): the same sweep against [__hip_fp8_e5m2].

   UNVERIFIED. Written on the CUDA box, where the `select` in tools/dune resolves to
   fp8_soak_hip.missing.ml, so this file has never been compiled: hipjit was not installed there.
   It is a mechanical mirror of fp8_soak_cuda.cudajit.ml against the hipjit API as
   arrayjit/lib/hip_backend.ml uses it, and the first ROCm box to build it should expect to fix
   names rather than to write the sweep. See the follow-up issue linked from gh-ocannl-657.

   This arm is also what re-localizes gh-ocannl-647: ROCm miscompiles [(__hip_fp8_e5m2)(float)] for
   magnitudes around 4e-25 to 3.3e-24, returning up to 2^-14 where every other implementation
   returns zero, and the exhaustive sweep is what narrowed that to exactly four f32 exponents. So a
   clean run here is NOT expected on an affected ROCm: the disagreement list is the finding. Adding
   a second pair of kernels that call [ocannl_single_to_fp8_uniform] /
   [ocannl_double_to_fp8_uniform] (Builtins_hip) would then show the guard closing the window. *)

open Base
module H = Hip

let name = "hip"
let vendor_type = "__hip_fp8_e5m2"

type bytes_buf =
  (int, Stdlib.Bigarray.int8_unsigned_elt, Stdlib.Bigarray.c_layout) Stdlib.Bigarray.Array1.t

(* hiprtc ships HIP's own headers, and targets the current default device when given no
   [--offload-arch] -- the same reliance hip_backend.ml's [compile] has. *)
let source =
  {|
extern "C" __global__ void ocannl_vendor_narrow_f32(unsigned long long base,
                                                    unsigned long long count,
                                                    unsigned char *out) {
  unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
  for (unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
       i < count; i += stride) {
    unsigned int u = (unsigned int)(base + i);
    float x = __uint_as_float(u);
    __hip_fp8_e5m2 v = (__hip_fp8_e5m2)x;
    out[i] = (unsigned char)v.__x;
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
      __hip_fp8_e5m2 v = (__hip_fp8_e5m2)d;
      out[(4 * i) + k] = (unsigned char)v.__x;
    }
  }
}
|}

(* [context] and [kernel_module] are held for the same reason the CUDA arm holds them: a primary
   context released by its finalizer resets the device under the sweep. See that arm's comment. *)
type state = {
  context : H.Context.t;
  kernel_module : H.Module.t;
  attrs : H.Device.attributes;
  stream : H.Stream.t;
  narrow_f32 : H.Module.func;
  narrow_f64 : H.Module.func;
  mutable buffer : (H.Deviceptr.t * int) option;
}

(* Present so the arms share one signature. hiprtc targets the current default device when given no
   [--offload-arch], and ROCm's fp8 conversion has no `__CUDA_ARCH__`-style compile-time split
   between a hardware and a software path, so there is nothing here for the policy to select
   between: both settings compile the same kernel for this device. Kept rather than dropped because
   the driver sets it uniformly, and because a future ROCm that does split should have somewhere
   obvious to put the distinction. *)
type arch_policy = [ `Device | `Backend ]

let set_arch_policy (_ : arch_policy) = ()

let state = ref None

(* Whether this box has an AMD device, not merely whether hipjit is linked. See the CUDA arm. *)
let probe () =
  match
    H.init ();
    H.Device.get_count ()
  with
  | 0 -> Error "hipjit is linked, but the HIP runtime reports no device"
  | _ -> Ok ()
  | exception e -> Error ("hipjit is linked, but HIP initialization failed: " ^ Exn.to_string e)

let init () =
  match !state with
  | Some st -> st
  | None ->
      H.init ();
      let device = H.Device.get ~ordinal:0 in
      let ctx = H.Context.get_primary device in
      H.Context.set_current ctx;
      let code =
        Hiprtc.compile_to_code ~hip_src:source ~name:"ocannl_fp8_soak.hip" ~options:[]
          ~with_debug:false
      in
      let kernel_module = H.Module.load_data_ex code [] in
      let st =
        {
          context = ctx;
          kernel_module;
          attrs = H.Device.get_attributes device;
          stream = H.Stream.create ();
          narrow_f32 = H.Module.get_function kernel_module ~name:"ocannl_vendor_narrow_f32";
          narrow_f64 = H.Module.get_function kernel_module ~name:"ocannl_vendor_narrow_f64";
          buffer = None;
        }
      in
      state := Some st;
      st

let describe () =
  let st = init () in
  Printf.sprintf "%s (%s)" st.attrs.name st.attrs.gcn_arch_name

let device_buffer st bytes =
  match st.buffer with
  | Some (ptr, size) when size >= bytes -> ptr
  | prev ->
      Option.iter prev ~f:(fun (ptr, _) -> H.Deviceptr.mem_free ptr);
      let ptr = H.Deviceptr.mem_alloc ~size_in_bytes:bytes in
      st.buffer <- Some (ptr, bytes);
      ptr

let block_dim = 256
let grid_dim = 4096

let fetch st ptr (out : bytes_buf) count =
  H.Stream.memcpy_D_to_H ~length:count
    ~dst:(Stdlib.Bigarray.genarray_of_array1 out)
    ~src:ptr st.stream;
  H.Stream.synchronize st.stream

let narrow_f32 ~base ~count (out : bytes_buf) =
  let st = init () in
  let ptr = device_buffer st count in
  H.Stream.launch_kernel st.narrow_f32 ~grid_dim_x:grid_dim ~block_dim_x:block_dim
    ~shared_mem_bytes:0 st.stream
    [
      H.Stream.Size_t (Unsigned.Size_t.of_int base);
      H.Stream.Size_t (Unsigned.Size_t.of_int count);
      H.Stream.Tensor ptr;
    ];
  fetch st ptr out count

let narrow_f64 ~base ~count ~lows (out : bytes_buf) =
  let st = init () in
  let bytes = 4 * count in
  let ptr = device_buffer st bytes in
  H.Stream.launch_kernel st.narrow_f64 ~grid_dim_x:grid_dim ~block_dim_x:block_dim
    ~shared_mem_bytes:0 st.stream
    [
      H.Stream.Size_t (Unsigned.Size_t.of_int base);
      H.Stream.Size_t (Unsigned.Size_t.of_int count);
      H.Stream.Int lows.(0);
      H.Stream.Int lows.(1);
      H.Stream.Int lows.(2);
      H.Stream.Int lows.(3);
      H.Stream.Tensor ptr;
    ];
  fetch st ptr out bytes
