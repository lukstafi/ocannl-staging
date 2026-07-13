(* Off-platform stub — see test_metal_storage_mode.missing.ml. Selected wherever the macOS-only
   `metal` / `arrayjit.metal_backend` libraries are unavailable; reproduces the golden so the test
   passes; real paths run only on macOS. *)
(* Binary mode keeps the echo byte-identical on Windows, where text-mode stdout would rewrite the
   CRLF line endings git's autocrlf checkout puts in the .expected file. *)
let () =
  Stdio.Out_channel.set_binary_mode Stdio.stdout true;
  Stdio.In_channel.read_all "test_metal_alloc.expected" |> Stdio.print_string
