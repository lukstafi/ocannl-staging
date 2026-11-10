(* gh-ocannl-2082: what macOS does to a Metal command buffer that does not end.

   Raw Metal, no OCANNL: one cell per process, chosen by the first argument, so that a cell's GPU
   state cannot leak into the next one's except through the OS. Every line goes to stdout, prefixed
   with the cell and the seconds since its commit.

   - [stall THREADS MAX_S]: commits a kernel of THREADS threads (one threadgroup of one thread is
     what OCANNL dispatches for an all-serial kernel), each looping 10^15 times, then a second
     buffer that only signals an event, as OCANNL's [sync] waits on. Polls both buffers' statuses,
     the event and the IORegistry's GPU [recoveryCount] every 250 ms until the kernel's buffer
     reads a final status or MAX_S passes. At 5 s it commits a one-thread kernel on a second queue
     and reports when that one completes: whether the GPU still serves other work under the stall.
     Exits without waiting for anything.
   - [spin ITERS]: a bounded kernel of one thread, ITERS dependent multiply-adds; reports its GPU
     time and final status. For whether a long legitimate kernel is killed.
   - [plain]: one trivial kernel, waited for with a 60 s bound. Run after a stall cell, it says
     whether the GPU came back once the stalling process exited.
   - [oob]: a one-thread kernel writing 256 MiB past the end of a 32-byte buffer. With
     [MTL_SHADER_VALIDATION=1] in the environment, whether Metal fails its command buffer. *)

module Me = Metal

let source =
  {|
#include <metal_stdlib>
using namespace metal;

kernel void stall(device float *x [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
  for (int i = 0; i < 100000; i++)
    for (int j = 0; j < 100000; j++)
      for (int k = 0; k < 100000; k++)
        x[tid % 8] = x[tid % 8] + 1.0f;
}

kernel void spin(device float *x [[buffer(0)]], constant uint &n [[buffer(1)]],
                 uint tid [[thread_position_in_grid]]) {
  float acc = x[tid];
  for (uint i = 0; i < n; i++) acc = fma(acc, 0.99999994f, 1.0f);
  x[tid] = acc;
}

kernel void plain(device float *x [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
  x[tid] = x[tid] + 1.0f;
}

kernel void oob(device float *x [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
  x[tid + (1u << 26)] = 1.0f;
}
|}

let device = Me.Device.create_system_default ()
let library = Me.Library.on_device device ~source (Me.CompileOptions.init ())

let pso name =
  fst (Me.ComputePipelineState.on_device_with_function device (Me.Library.new_function_with_name library name))

let buffer bytes =
  Me.Buffer.on_device device ~length:bytes
    (Me.ResourceOptions.make ~storage_mode:Me.ResourceOptions.storage_mode_shared ())

let cell = if Array.length Sys.argv > 1 then Sys.argv.(1) else "plain"
let t0 = ref (Unix.gettimeofday ())
let say fmt = Printf.ksprintf (fun s -> Printf.printf "cell=%s t=%.2f %s\n%!" cell (Unix.gettimeofday () -. !t0) s) fmt

let recovery () =
  let ic =
    Unix.open_process_in
      "/usr/sbin/ioreg -r -c IOAccelerator -l | /usr/bin/grep -o '\"recoveryCount\"=[0-9]*' | head -1"
  in
  let line = try input_line ic with End_of_file -> "?" in
  ignore (Unix.close_process_in ic);
  line

let status cb = Me.CommandBuffer.(Status.sexp_of_t (get_status cb)) |> Sexplib0.Sexp.to_string
let final cb = Me.CommandBuffer.(match get_status cb with Completed | Error -> true | _ -> false)
let error cb = Option.value (Me.CommandBuffer.get_error cb) ~default:"-"

let launch ?(bytes = []) queue ~name ~threads x =
  let cb = Me.CommandBuffer.on_queue queue in
  let enc = Me.ComputeCommandEncoder.on_buffer cb in
  Me.ComputeCommandEncoder.set_compute_pipeline_state enc (pso name);
  Me.ComputeCommandEncoder.set_buffer enc ~index:0 x;
  List.iteri
    (fun i v ->
      let p = Ctypes.(allocate uint32_t (Unsigned.UInt32.of_int v) |> to_voidp) in
      Me.ComputeCommandEncoder.set_bytes enc ~bytes:p ~length:4 ~index:(i + 1))
    bytes;
  let per = min threads 256 in
  Me.ComputeCommandEncoder.dispatch_threadgroups enc
    ~threadgroups_per_grid:{ width = (threads + per - 1) / per; height = 1; depth = 1 }
    ~threads_per_threadgroup:{ width = per; height = 1; depth = 1 };
  Me.ComputeCommandEncoder.end_encoding enc;
  Me.CommandBuffer.commit cb;
  cb

let gpu_time cb = Me.CommandBuffer.(get_gpu_end_time cb -. get_gpu_start_time cb)

let wait_bounded ~max_s cb =
  let deadline = Unix.gettimeofday () +. max_s in
  while (not (final cb)) && Unix.gettimeofday () < deadline do
    Unix.sleepf 0.01
  done

let stall threads max_s =
  let queue = Me.CommandQueue.on_device device and side = Me.CommandQueue.on_device device in
  let x = buffer (4 * max threads 8) and y = buffer 32 in
  let event = Me.SharedEvent.on_device device in
  say "recovery_before %s" (recovery ());
  t0 := Unix.gettimeofday ();
  let k = launch queue ~name:"stall" ~threads x in
  let s = Me.CommandBuffer.on_queue queue in
  Me.CommandBuffer.encode_signal_event s (Me.SharedEvent.super event) (Unsigned.ULLong.of_int 1);
  Me.CommandBuffer.commit s;
  let side_cb = ref None and side_done = ref false in
  let last = ref "" and last_rc = ref "" and beat = ref 0. in
  let stop = ref false in
  while not !stop do
    let now = Unix.gettimeofday () -. !t0 in
    let rc = if now -. !beat >= 2. then (beat := now; recovery ()) else !last_rc in
    let line =
      Printf.sprintf "kernel=%s signal=%s event=%s" (status k) (status s)
        (Unsigned.ULLong.to_string (Me.SharedEvent.get_signaled_value event))
    in
    if line <> !last || rc <> !last_rc then say "%s %s" line rc;
    last := line;
    last_rc := rc;
    if !side_cb = None && now >= 5. then (
      side_cb := Some (launch side ~name:"plain" ~threads:1 y);
      say "side kernel committed on a second queue");
    (match !side_cb with
    | Some cb when (not !side_done) && final cb ->
        side_done := true;
        say "side kernel %s gpu=%.4fs error=%s" (status cb) (gpu_time cb) (error cb)
    | _ -> ());
    if final k then (
      say "KERNEL FINAL %s error=%s gpu=%.2fs" (status k) (error k) (gpu_time k);
      wait_bounded ~max_s:2. s;
      say "signal buffer %s error=%s event=%s" (status s) (error s)
        (Unsigned.ULLong.to_string (Me.SharedEvent.get_signaled_value event));
      stop := true)
    else if now >= max_s then (
      say "NOT FINAL after %.0fs kernel=%s side_done=%b" max_s (status k) !side_done;
      stop := true)
    else Unix.sleepf 0.25
  done;
  say "recovery_after %s" (recovery ());
  say "RESULT stall threads=%d final=%b kernel=%s side_completed=%b" threads (final k) (status k)
    !side_done

let spin iters =
  let queue = Me.CommandQueue.on_device device in
  let x = buffer 32 in
  say "recovery_before %s" (recovery ());
  t0 := Unix.gettimeofday ();
  let k = launch queue ~bytes:[ iters ] ~name:"spin" ~threads:1 x in
  Me.CommandBuffer.wait_until_completed k;
  say "RESULT spin iters=%d %s gpu=%.3fs error=%s %s" iters (status k) (gpu_time k) (error k)
    (recovery ())

let plain () =
  let queue = Me.CommandQueue.on_device device in
  let x = buffer 32 in
  say "recovery_before %s" (recovery ());
  t0 := Unix.gettimeofday ();
  let k = launch queue ~name:"plain" ~threads:1 x in
  wait_bounded ~max_s:60. k;
  say "RESULT plain %s gpu=%.4fs error=%s %s" (status k) (gpu_time k) (error k) (recovery ())

let oob () =
  let queue = Me.CommandQueue.on_device device in
  let x = buffer 32 in
  say "MTL_SHADER_VALIDATION=%s"
    (Option.value (Sys.getenv_opt "MTL_SHADER_VALIDATION") ~default:"unset");
  t0 := Unix.gettimeofday ();
  let k = launch queue ~name:"oob" ~threads:1 x in
  wait_bounded ~max_s:60. k;
  say "RESULT oob %s gpu=%.4fs error=%s" (status k) (gpu_time k) (error k)

let () =
  let arg i d = if Array.length Sys.argv > i then Sys.argv.(i) else d in
  match cell with
  | "stall" -> stall (int_of_string (arg 2 "1")) (float_of_string (arg 3 "60"))
  | "spin" -> spin (int_of_string (arg 2 "1000000"))
  | "plain" -> plain ()
  | "oob" -> oob ()
  | other -> failwith ("unknown cell " ^ other)
