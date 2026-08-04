(* Stub selected wherever cudajit (and hence the optional `arrayjit.cuda_backend` library) is
   unavailable — see test_cuda_classify_failure.missing.ml for the pattern and why an `enabled_if`
   cannot replace it. Reproduces the golden so the test passes; the real checks run wherever cudajit
   is installed (no GPU needed: the policy under test is a pure function). *)
(* Binary mode keeps the echo byte-identical on Windows, where text-mode stdout would rewrite the
   CRLF line endings git's autocrlf checkout puts in the .expected file. *)
let () =
  Stdio.Out_channel.set_binary_mode Stdio.stdout true;
  Stdio.In_channel.read_all "test_cuda_arch_flags.expected" |> Stdio.print_string
