(* Standalone discriminator for gh-ocannl-828: do two long Metal command buffers become much more
   expensive merely because both are submitted before the host waits, or only when OCANNL's
   SharedEvent ordering shape is present?

   This links the Metal bindings directly and contains no OCANNL lowering, scheduling, context or
   stream code. It measures three shapes over the same pipeline and two independent output buffers:

   - [sync-between]: commit one kernel and await it before committing the next; - [raw-queued]:
   commit both kernels, then await both (no intervening host synchronization); - [event-chain]:
   reproduce Metal_backend's command-buffer sequence exactly: kernel, signal, wait+kernel, signal,
   followed by one host wait on the final SharedEvent value.

   The kernel's loop bound is runtime data and every iteration performs a volatile device read, so
   the compiler cannot fold the long loop away. The first argument is a target duration in
   milliseconds for one kernel (default 1000); calibration measures a short kernel and scales its
   iteration count to the target. Run only on an otherwise idle Metal Mac, recording [uptime] beside
   the output:

   dune exec benchmarks/runners/ocannl/metal_queue_probe.exe -- 2000 *)

module Me = Metal

let source =
  {|
#include <metal_stdlib>
using namespace metal;

kernel void spin(
    device const volatile float* in [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& iterations [[buffer(2)]]) {
  float x = out[0];
  for (uint i = 0; i < iterations; ++i) {
    x = fma(x, 1.00000011920928955078125f, in[i & 1023] * 0.00000011920928955078125f);
  }
  out[0] = x;
}
|}

let ropts =
  Me.ResourceOptions.(
    storage_mode_shared + cpu_cache_mode_write_combined + hazard_tracking_mode_untracked)

let now () = Unix.gettimeofday ()
let elapsed_ms start = (now () -. start) *. 1000.

let command_buffer ~queue ~pso ~input ~output ~iterations_buf =
  let cb = Me.CommandBuffer.on_queue queue in
  let enc = Me.ComputeCommandEncoder.on_buffer cb in
  Me.ComputeCommandEncoder.set_compute_pipeline_state enc pso;
  Me.ComputeCommandEncoder.set_buffer enc ~index:0 input;
  Me.ComputeCommandEncoder.set_buffer enc ~index:1 output;
  Me.ComputeCommandEncoder.set_buffer enc ~index:2 iterations_buf;
  Me.ComputeCommandEncoder.dispatch_threadgroups enc
    ~threadgroups_per_grid:{ width = 1; height = 1; depth = 1 }
    ~threads_per_threadgroup:{ width = 1; height = 1; depth = 1 };
  Me.ComputeCommandEncoder.end_encoding enc;
  cb

let commit_signal queue event value =
  let cb = Me.CommandBuffer.on_queue queue in
  Me.CommandBuffer.encode_signal_event cb (Me.SharedEvent.super event) value;
  Me.CommandBuffer.commit cb

let check_completed label cb =
  match Me.CommandBuffer.get_status cb with
  | Completed -> ()
  | Error ->
      Printf.eprintf "%s command buffer failed: %s\n" label
        (Option.value (Me.CommandBuffer.get_error cb) ~default:"unknown Metal error");
      exit 1
  | status ->
      Printf.eprintf "%s command buffer ended in status %s\n" label
        (Sexplib0.Sexp.to_string_hum (Me.CommandBuffer.Status.sexp_of_t status));
      exit 1

let () =
  let target_ms =
    match Array.to_list Sys.argv with
    | [ _ ] -> 1000.
    | [ _; value ] -> (
        match Float.of_string_opt value with
        | Some value when Float.is_finite value && value > 0. -> value
        | _ ->
            Printf.eprintf "metal_queue_probe: target_ms must be a positive finite number\n";
            exit 2)
    | _ ->
        Printf.eprintf "usage: metal_queue_probe [target_ms]\n";
        exit 2
  in
  let device = Me.Device.create_system_default () in
  let queue = Me.CommandQueue.on_device device in
  let options = Me.CompileOptions.init () in
  Me.CompileOptions.set_language_version options Me.CompileOptions.LanguageVersion.version_3_1;
  let library = Me.Library.on_device device ~source options in
  let fn = Me.Library.new_function_with_name library "spin" in
  let pso, _ = Me.ComputePipelineState.on_device_with_function device fn in
  let input = Me.Buffer.on_device device ~length:(1024 * 4) ropts in
  let output_a = Me.Buffer.on_device device ~length:4 ropts in
  let output_b = Me.Buffer.on_device device ~length:4 ropts in
  let iterations_buf = Me.Buffer.on_device device ~length:4 ropts in
  let open Ctypes in
  let input_f = coerce (ptr void) (ptr float) (Me.Buffer.contents input) in
  let output_a_f = coerce (ptr void) (ptr float) (Me.Buffer.contents output_a) in
  let output_b_f = coerce (ptr void) (ptr float) (Me.Buffer.contents output_b) in
  let iterations_u32 = coerce (ptr void) (ptr uint32_t) (Me.Buffer.contents iterations_buf) in
  for i = 0 to 1023 do
    input_f +@ i <-@ Float.of_int ((i mod 29) + 1) /. 29.
  done;
  let set_iterations n =
    iterations_u32 <-@ Unsigned.UInt32.of_int n;
    output_a_f <-@ 0.25;
    output_b_f <-@ 0.5
  in
  let run_one output =
    let cb = command_buffer ~queue ~pso ~input ~output ~iterations_buf in
    let start = now () in
    Me.CommandBuffer.commit cb;
    Me.CommandBuffer.wait_until_completed cb;
    check_completed "calibration" cb;
    (elapsed_ms start, cb)
  in
  (* Warm compilation/dispatch before asking the calibration clock a question. *)
  set_iterations 1;
  ignore (run_one output_a);
  let calibration_iterations = 1 lsl 18 in
  set_iterations calibration_iterations;
  let seed_ms, _ = run_one output_a in
  if (not (Float.is_finite seed_ms)) || seed_ms <= 0. then (
    Printf.eprintf "metal_queue_probe: calibration clock returned %.9g ms\n" seed_ms;
    exit 1);
  let iterations =
    let scaled =
      Float.ceil (Float.of_int calibration_iterations *. target_ms /. seed_ms) |> Int.of_float
    in
    Int.max 1 (Int.min (1 lsl 30) scaled)
  in
  set_iterations iterations;
  let calibration_ms, _ = run_one output_a in
  Printf.printf "device: %s\n" (Me.Device.get_attributes device).name;
  Printf.printf "target: %.0f ms; iterations: %d; calibrated single: %.3f ms\n" target_ms iterations
    calibration_ms;

  let sync_between () =
    set_iterations iterations;
    let start = now () in
    let a = command_buffer ~queue ~pso ~input ~output:output_a ~iterations_buf in
    Me.CommandBuffer.commit a;
    Me.CommandBuffer.wait_until_completed a;
    check_completed "sync-between first" a;
    let b = command_buffer ~queue ~pso ~input ~output:output_b ~iterations_buf in
    Me.CommandBuffer.commit b;
    Me.CommandBuffer.wait_until_completed b;
    check_completed "sync-between second" b;
    elapsed_ms start
  in
  let raw_queued () =
    set_iterations iterations;
    let a = command_buffer ~queue ~pso ~input ~output:output_a ~iterations_buf in
    let b = command_buffer ~queue ~pso ~input ~output:output_b ~iterations_buf in
    let start = now () in
    Me.CommandBuffer.commit a;
    Me.CommandBuffer.commit b;
    (* Both waits are after both submissions; the second is normally already complete. *)
    Me.CommandBuffer.wait_until_completed a;
    Me.CommandBuffer.wait_until_completed b;
    check_completed "raw-queued first" a;
    check_completed "raw-queued second" b;
    elapsed_ms start
  in
  let event_chain () =
    set_iterations iterations;
    let event = Me.SharedEvent.on_device device in
    let one = Unsigned.ULLong.one in
    let two = Unsigned.ULLong.of_int 2 in
    let a = command_buffer ~queue ~pso ~input ~output:output_a ~iterations_buf in
    let b = command_buffer ~queue ~pso ~input ~output:output_b ~iterations_buf in
    Me.CommandBuffer.encode_wait_for_event b (Me.SharedEvent.super event) one;
    let start = now () in
    Me.CommandBuffer.commit a;
    commit_signal queue event one;
    Me.CommandBuffer.commit b;
    commit_signal queue event two;
    let completed =
      Me.SharedEvent.wait_until_signaled_value event ~value:two ~timeout_ms:Unsigned.ULLong.max_int
    in
    if not completed then (
      Printf.eprintf "event-chain final SharedEvent wait timed out\n";
      exit 1);
    Me.CommandBuffer.wait_until_completed a;
    Me.CommandBuffer.wait_until_completed b;
    check_completed "event-chain first" a;
    check_completed "event-chain second" b;
    elapsed_ms start
  in
  (* Warm every submission shape once, then rotate their measurement order across three passes. *)
  ignore (sync_between ());
  ignore (raw_queued ());
  ignore (event_chain ());
  let arms =
    [ ("sync-between", sync_between); ("raw-queued", raw_queued); ("event-chain", event_chain) ]
  in
  let rotate n values =
    let rec split i left = function
      | rest when i = 0 -> (List.rev left, rest)
      | [] -> (List.rev left, [])
      | value :: rest -> split (i - 1) (value :: left) rest
    in
    let left, right = split n [] values in
    right @ left
  in
  let samples = Hashtbl.create 3 in
  List.iter (fun (label, _) -> Hashtbl.add samples label []) arms;
  for pass = 0 to 2 do
    List.iter
      (fun (label, run) ->
        let ms = run () in
        Hashtbl.replace samples label (ms :: Hashtbl.find samples label))
      (rotate pass arms)
  done;
  let median values =
    let values = Array.of_list values in
    Array.sort Float.compare values;
    values.(Array.length values / 2)
  in
  let baseline = median (Hashtbl.find samples "sync-between") in
  Printf.printf "%-14s %12s %10s\n" "submission" "two-kernel ms" "vs synced";
  List.iter
    (fun (label, _) ->
      let ms = median (Hashtbl.find samples label) in
      Printf.printf "%-14s %12.3f %9.3fx\n" label ms (ms /. baseline))
    arms;
  Printf.printf "outputs: %.9g %.9g\n" !@output_a_f !@output_b_f
