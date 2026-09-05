let () =
  if Array.length Sys.argv <> 5 then (
    prerr_endline "usage: sweep_harness_driver HARNESS SWEEP AGGREGATE_SKIPS VERDICT_PROBE";
    exit 2);
  let metal_options =
    Ir.Compiler_options.metal ~routine_logging:false ~math_api:Modern_split
    |> Ir.Compiler_options.render_metal
  in
  let hip_options =
    Ir.Compiler_options.hiprtc ~hip_include_options:[] ~rocwmma_include_options:[]
      ~uses_rocwmma:false ~with_debug:false
    |> Ir.Compiler_options.render
  in
  (* The two slots [Cuda_backend] discovers at compile time (the CUDA_PATH include directory and
     [gpu_arch_options]' architecture target) are sentinels here, as in
     arrayjit/test/test_cuda_compile_options.ml: the harness pins that the fingerprint carries the
     rendered vector whole, and a value no installed toolkit produces cannot match by
     coincidence. *)
  let nvrtc_options =
    Ir.Compiler_options.nvrtc ~cuda_include_options:[ "-I/sentinel/cuda/include" ]
      ~arch_options:[ "--gpu-architecture=compute_999" ]
      ~with_device_debug:false
    |> Ir.Compiler_options.render
  in
  let argv =
    [|
      "bash";
      Sys.argv.(1);
      Sys.argv.(2);
      Sys.argv.(3);
      Sys.argv.(4);
      metal_options;
      hip_options;
      nvrtc_options;
    |]
  in
  let pid = Unix.create_process "bash" argv Unix.stdin Unix.stdout Unix.stderr in
  match snd (Unix.waitpid [] pid) with
  | Unix.WEXITED code -> exit code
  | Unix.WSIGNALED signal | Unix.WSTOPPED signal -> exit (128 + signal)
