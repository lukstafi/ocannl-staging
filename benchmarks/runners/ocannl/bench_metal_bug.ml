(* Standalone Metal check for the serial RMW-accumulation miscompile in the fissioned CE kernel:
   verbatim seg4 source from bench_min_repro build_files (b=64), buffers filled with the same
   values as the OCANNL run. Expected ~5.02; the bug prints ~0.029 (last sample only). *)

module Me = Metal

let source =
  {|
#include <metal_stdlib>
using namespace metal;
kernel void cross_entropy_loss_fwd__seg4(
    device char* __pool0 [[buffer(0)]],
    device char* __pool1 [[buffer(1)]],
    device char* __pool10 [[buffer(2)]],
    device char* __pool11 [[buffer(3)]],
    device char* __pool12 [[buffer(4)]],
    device char* __pool13 [[buffer(5)]],
    device char* __pool14 [[buffer(6)]],
    device char* __pool15 [[buffer(7)]],
    device char* __pool2 [[buffer(8)]],
    device char* __pool3 [[buffer(9)]],
    device char* __pool4 [[buffer(10)]],
    device char* __pool5 [[buffer(11)]],
    device char* __pool6 [[buffer(12)]],
    device char* __pool7 [[buffer(13)]],
    device char* __pool8 [[buffer(14)]],
    device char* __pool9 [[buffer(15)]],
    device const uint* __pool_slots [[buffer(16)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) {

  /* Pool base pointers. */
  device char* __pools[16] = { __pool0, __pool1, __pool2, __pool3, __pool4, __pool5, __pool6, __pool7, __pool8, __pool9, __pool10, __pool11, __pool12, __pool13, __pool14, __pool15 };
  device float* __restrict cross_entropy_loss = (device float*)(__pools[__pool_slots[0]] + __pool_slots[1]);
  device float* __restrict n4 = (device float*)(__pools[__pool_slots[2]] + __pool_slots[3]);
  device float* __restrict n13_log = (device float*)(__pools[__pool_slots[4]] + __pool_slots[5]);
  device float* __restrict one_hot = (device float*)(__pools[__pool_slots[6]] + __pool_slots[7]);
  device float* __restrict cross_entropy = (device float*)(__pools[__pool_slots[8]] + __pool_slots[9]);
  device float* __restrict max_logits = (device float*)(__pools[__pool_slots[10]] + __pool_slots[11]);
  /* Local declarations and initialization. */

  /* Main logic. */
  for (int32_t i55 = 0; i55 <= 63; ++i55) {
    {
      float v19_n16;
      v19_n16 = (float)(0);
      for (int32_t i59 = 0; i59 <= 9; ++i59) {
        v19_n16 = fma(one_hot[(i55) * 10 + i59], (n4[(i55) * 10 + i59] -
          (max_logits[(i55) * 1 + 0] + n13_log[(i55) * 1 + 0])), v19_n16);
      }
      cross_entropy[0] = (cross_entropy[0] + -v19_n16);
    }
  }
  cross_entropy_loss[0] = (cross_entropy[0] / (float)(64));
  /* end */
}
|}

let () =
  let b = 64 and d = 84 and v = 10 in
  let device = Me.Device.create_system_default () in
  let queue = Me.CommandQueue.on_device device in
  let options = Me.CompileOptions.init () in
  let library = Me.Library.on_device device ~source options in
  let func = Me.Library.new_function_with_name library "cross_entropy_loss_fwd__seg4" in
  let pso, _ = Me.ComputePipelineState.on_device_with_function device func in
  let ropts =
    Me.ResourceOptions.(
      storage_mode_shared + cpu_cache_mode_write_combined + hazard_tracking_mode_untracked)
  in
  (* Layout within one pool: cel@0, ce@256, max@512, log@1024, one_hot@2048, n4@8192. *)
  let pool = Me.Buffer.on_device device ~length:65536 ropts in
  let slots = Me.Buffer.on_device device ~length:64 ropts in
  let open Ctypes in
  let fp = coerce (ptr void) (ptr float) (Me.Buffer.contents pool) in
  let logit s c =
    let acc = ref 0. in
    for k = 0 to d - 1 do
      acc :=
        !acc +. (0.1 *. Float.cos (Float.of_int ((c * d) + k)) *. Float.sin (Float.of_int ((s * d) + k)))
    done;
    !acc
  in
  (* n4 = logits, max_logits, n13_log, one_hot as in the OCANNL run. *)
  for s = 0 to b - 1 do
    let mx = ref neg_infinity in
    for c = 0 to v - 1 do
      fp +@ (2048 + (s * v) + c) <-@ logit s c;
      if logit s c > !mx then mx := logit s c
    done;
    fp +@ (128 + s) <-@ !mx;
    let sum = ref 0. in
    for c = 0 to v - 1 do
      sum := !sum +. exp (logit s c -. !mx)
    done;
    fp +@ (256 + s) <-@ log !sum;
    for c = 0 to v - 1 do
      fp +@ (512 + (s * v) + c) <-@ (if c = s mod v then 1.0 else 0.0)
    done
  done;
  fp <-@ 0.0;
  (* cel *)
  fp +@ 64 <-@ 0.0;
  (* ce *)
  let sp = coerce (ptr void) (ptr uint32_t) (Me.Buffer.contents slots) in
  let set i v = sp +@ i <-@ Unsigned.UInt32.of_int v in
  (* (pool_idx, byte_offset) pairs: cel, n4, n13_log, one_hot, ce, max_logits *)
  set 0 0;
  set 1 0;
  set 2 0;
  set 3 (2048 * 4);
  set 4 0;
  set 5 (256 * 4);
  set 6 0;
  set 7 (512 * 4);
  set 8 0;
  set 9 (64 * 4);
  set 10 0;
  set 11 (128 * 4);
  let cb = Me.CommandBuffer.on_queue queue in
  let enc = Me.ComputeCommandEncoder.on_buffer cb in
  Me.ComputeCommandEncoder.set_compute_pipeline_state enc pso;
  for i = 0 to 15 do
    Me.ComputeCommandEncoder.set_buffer enc ~index:i pool
  done;
  Me.ComputeCommandEncoder.set_buffer enc ~index:16 slots;
  Me.ComputeCommandEncoder.dispatch_threadgroups enc
    ~threadgroups_per_grid:{ width = 1; height = 1; depth = 1 }
    ~threads_per_threadgroup:{ width = 1; height = 1; depth = 1 };
  Me.ComputeCommandEncoder.end_encoding enc;
  Me.CommandBuffer.commit cb;
  Me.CommandBuffer.wait_until_completed cb;
  Printf.printf "loss: %.6f  ce: %.6f (expected loss ~5.0198; bug ~0.029263)\n"
    !@fp
    !@(fp +@ 64)
