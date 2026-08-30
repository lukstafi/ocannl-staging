(* Does Metal's compile-time math policy reassociate scalar reductions, preserve a runtime infinity
   mask, and what does the safe policy cost? gh-ocannl-848.

   Standalone on purpose: this links Metal and ctypes, not OCANNL, so it measures the compiler API
   directly and remains useful if backend lowering changes. Run on a Metal Mac with:

   dune exec benchmarks/runners/ocannl/metal_reassoc_probe.exe

   The reduction detector is the one used by tools/nvrtc_reassoc_probe.c: 1.0f followed by values
   smaller than half an ulp at 1.0. Strict left-to-right addition remains exactly 1.0; grouping any
   of the small terms first makes the result larger. Three source shapes are used because hiprtc's
   reassociation bug depended on spelling: a fixed counted loop, a runtime-bound loop, and 128
   repeated statements. The infinity leg reads [-INFINITY] and the mask from device memory so the
   compiler must preserve a runtime value rather than merely fold a literal.

   Runtime timings use the command buffer's GPU clock. Arms are interleaved and rotated within each
   repeat, every pipeline is warmed first, and medians (not minima) are reported. *)

module Me = Metal

type math_policy =
  | Default
  | Legacy_fast of bool
  | Mode of Me.CompileOptions.MathMode.t
  | Production_safe_fast

type variant = { label : string; policy : math_policy }

let variants =
  [
    { label = "default"; policy = Default };
    { label = "legacy-fast=true"; policy = Legacy_fast true };
    { label = "legacy-fast=false"; policy = Legacy_fast false };
    { label = "mode=fast"; policy = Mode Fast };
    { label = "mode=relaxed"; policy = Mode Relaxed };
    { label = "mode=safe"; policy = Mode Safe };
    { label = "production-safe-fast"; policy = Production_safe_fast };
  ]

let apply_policy options = function
  | Default -> ()
  | Legacy_fast enabled -> Me.CompileOptions.set_fast_math_enabled options enabled
  | Mode mode -> Me.CompileOptions.set_math_mode options mode
  | Production_safe_fast ->
      Me.CompileOptions.set_math_mode options Safe;
      Me.CompileOptions.set_math_floating_point_functions options Fast

let repeated_statements =
  String.concat "" (List.init 128 (fun i -> Printf.sprintf "  acc += in[%d];\n" i))

let source =
  Printf.sprintf
    {|#include <metal_stdlib>
using namespace metal;

kernel void counted(
    device const float* in [[buffer(0)]], device float* out [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
  float acc = 0.0f;
  for (uint i = 0; i < 128; ++i) acc += in[i];
  out[gid] = acc;
}

kernel void runtime_n(
    device const float* in [[buffer(0)]], device float* out [[buffer(1)]],
    constant uint& n [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
  float acc = 0.0f;
  for (uint i = 0; i < n; ++i) acc += in[i];
  out[gid] = acc;
}

kernel void repeated(
    device const float* in [[buffer(0)]], device float* out [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
  float acc = 0.0f;
%s  out[gid] = acc;
}

kernel void cancel(
    device const float* in [[buffer(0)]], device float* out [[buffer(1)]]) {
  float a = in[0], b = in[1];
  out[0] = (a + b) - a;
}

kernel void infinity_mask(
    device const float* in [[buffer(0)]], device float* out [[buffer(1)]]) {
  float masked = in[2] != 0.0f ? in[1] : in[0];
  out[0] = exp(masked - in[1]);
}
|}
    repeated_statements

let ropts =
  Me.ResourceOptions.(
    storage_mode_shared + cpu_cache_mode_write_combined + hazard_tracking_mode_untracked)

let float_bits x = Int32.bits_of_float x

let median xs =
  let a = Array.of_list xs in
  Array.sort Float.compare a;
  let n = Array.length a in
  if n mod 2 = 1 then a.(n / 2) else (a.((n / 2) - 1) +. a.(n / 2)) /. 2.0

let rotate n xs =
  let len = List.length xs in
  let n = n mod len in
  List.filteri (fun i _ -> i >= n) xs @ List.filteri (fun i _ -> i < n) xs

let () =
  let device = Me.Device.create_system_default () in
  let queue = Me.CommandQueue.on_device device in
  let states = Hashtbl.create 32 in
  let effective = Hashtbl.create 8 in
  List.iter
    (fun variant ->
      let options = Me.CompileOptions.init () in
      Me.CompileOptions.set_language_version options Me.CompileOptions.LanguageVersion.version_3_1;
      apply_policy options variant.policy;
      Hashtbl.add effective variant.label
        ( Me.CompileOptions.get_fast_math_enabled options,
          Me.CompileOptions.get_math_mode options,
          Me.CompileOptions.get_math_floating_point_functions options );
      let library = Me.Library.on_device device ~source options in
      List.iter
        (fun name ->
          let fn = Me.Library.new_function_with_name library name in
          Hashtbl.add states (variant.label, name)
            (fst (Me.ComputePipelineState.on_device_with_function device fn)))
        [ "counted"; "runtime_n"; "repeated"; "cancel"; "infinity_mask" ])
    variants;

  let max_n = 1 lsl 20 in
  let timed_threads = 1 lsl 18 in
  let input = Me.Buffer.on_device device ~length:(4 * max_n) ropts in
  let output = Me.Buffer.on_device device ~length:(4 * timed_threads) ropts in
  let nbuf = Me.Buffer.on_device device ~length:4 ropts in
  let open Ctypes in
  let input_f = coerce (ptr void) (ptr float) (Me.Buffer.contents input) in
  let output_f = coerce (ptr void) (ptr float) (Me.Buffer.contents output) in
  let n_u32 = coerce (ptr void) (ptr uint32_t) (Me.Buffer.contents nbuf) in
  input_f <-@ 1.0;
  for i = 1 to max_n - 1 do
    input_f +@ i <-@ 3e-8
  done;
  n_u32 <-@ Unsigned.UInt32.of_int 128;

  let dispatch variant name ~threads ~n =
    n_u32 <-@ Unsigned.UInt32.of_int n;
    let cb = Me.CommandBuffer.on_queue queue in
    let enc = Me.ComputeCommandEncoder.on_buffer cb in
    Me.ComputeCommandEncoder.set_compute_pipeline_state enc
      (Hashtbl.find states (variant.label, name));
    Me.ComputeCommandEncoder.set_buffer enc ~index:0 input;
    Me.ComputeCommandEncoder.set_buffer enc ~index:1 output;
    if String.equal name "runtime_n" then Me.ComputeCommandEncoder.set_buffer enc ~index:2 nbuf;
    let width = min 256 threads in
    Me.ComputeCommandEncoder.dispatch_threadgroups enc
      ~threadgroups_per_grid:{ width = (threads + width - 1) / width; height = 1; depth = 1 }
      ~threads_per_threadgroup:{ width; height = 1; depth = 1 };
    Me.ComputeCommandEncoder.end_encoding enc;
    Me.CommandBuffer.commit cb;
    Me.CommandBuffer.wait_until_completed cb;
    Me.CommandBuffer.get_gpu_end_time cb -. Me.CommandBuffer.get_gpu_start_time cb
  in
  let read variant name ~n =
    output_f <-@ -1.0;
    ignore (dispatch variant name ~threads:1 ~n : float);
    !@output_f
  in

  Printf.printf "device: %s\n" (Me.Device.get_attributes device).name;
  Printf.printf "MSL language: 3.1; reduction detector: 1.0f + 127 x 3e-8f\n";
  List.iter
    (fun variant ->
      let fast, mode, funcs = Hashtbl.find effective variant.label in
      Printf.printf "  %-19s effective: fastMathEnabled=%b mathMode=%s functions=%s\n" variant.label
        fast
        (match mode with Safe -> "safe" | Relaxed -> "relaxed" | Fast -> "fast")
        (match funcs with Fast -> "fast" | Precise -> "precise"))
    variants;
  Printf.printf "%-19s %12s %12s %12s %12s %12s\n" "policy" "counted" "runtime" "repeated" "cancel"
    "inf-mask";
  let safety_failures = ref 0 in
  List.iter
    (fun variant ->
      let counted = read variant "counted" ~n:128 in
      let runtime = read variant "runtime_n" ~n:128 in
      let repeated = read variant "repeated" ~n:128 in
      input_f <-@ 1e10;
      input_f +@ 1 <-@ 1.0;
      let cancel = read variant "cancel" ~n:1 in
      input_f <-@ neg_infinity;
      input_f +@ 1 <-@ 0.5;
      input_f +@ 2 <-@ 0.0;
      let inf_mask = read variant "infinity_mask" ~n:1 in
      Printf.printf "%-19s %12lx %12lx %12lx %12lx %12lx\n" variant.label (float_bits counted)
        (float_bits runtime) (float_bits repeated) (float_bits cancel) (float_bits inf_mask);
      (match variant.policy with
      | Legacy_fast false | Mode Safe | Production_safe_fast ->
          if counted <> 1.0 || runtime <> 1.0 || repeated <> 1.0 || cancel <> 0.0 || inf_mask <> 0.0
          then incr safety_failures
      | Default | Legacy_fast true | Mode Fast | Mode Relaxed -> ());
      input_f <-@ 1.0;
      input_f +@ 1 <-@ 3e-8;
      input_f +@ 2 <-@ 3e-8)
    variants;

  let repeats = 21 in
  let shapes =
    [
      ("counted", timed_threads, 128, "262144 threads x 128 fixed-count terms");
      ("runtime_n", 1, max_n, "1 thread x 1048576 runtime-bound terms");
    ]
  in
  Printf.printf "\nGPU-clock medians, %d rotated interleaved repeats (one warm-up per arm):\n"
    repeats;
  List.iter
    (fun (name, threads, n, shape) ->
      List.iter (fun variant -> ignore (dispatch variant name ~threads ~n : float)) variants;
      let samples = Hashtbl.create 8 in
      List.iter (fun variant -> Hashtbl.add samples variant.label []) variants;
      for r = 0 to repeats - 1 do
        List.iter
          (fun variant ->
            let t = dispatch variant name ~threads ~n in
            Hashtbl.replace samples variant.label (t :: Hashtbl.find samples variant.label))
          (rotate r variants)
      done;
      let baseline = median (Hashtbl.find samples "default") in
      Printf.printf "  %s:\n" shape;
      List.iter
        (fun variant ->
          let t = median (Hashtbl.find samples variant.label) in
          Printf.printf "    %-19s %9.4f ms  %7.3fx default\n" variant.label (t *. 1000.0)
            (t /. baseline))
        variants)
    shapes;
  if !safety_failures = 0 then Printf.printf "\nVERDICT: all safe spellings preserved all values\n"
  else (
    Printf.eprintf "\nVERDICT: %d safe spelling(s) changed a required value\n" !safety_failures;
    exit 1)
