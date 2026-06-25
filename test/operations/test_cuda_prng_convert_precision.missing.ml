(* Off-platform stub. cudajit / arrayjit.cuda_backend is absent on this host, so dune's [select] (in
   this test's [libraries]) picks this module instead of test_cuda_prng_convert_precision.real.ml. It
   reproduces the recorded golden output so the test passes trivially; the real assertions over
   [Cuda_backend.convert_precision] run only where cudajit is available (e.g. minipc-wsl). *)
let () =
  Stdio.In_channel.read_all "test_cuda_prng_convert_precision.expected" |> Stdio.print_string
