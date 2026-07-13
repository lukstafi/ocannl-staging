(* Off-platform stub. The Metal backend (`metal` / `arrayjit.metal_backend`) is macOS-only, so
   dune's `select` (in this test's `libraries`) picks this module wherever those libraries are
   unavailable. It reproduces the recorded golden output so the test passes trivially; the real
   Metal paths run only on macOS, where `test_metal_storage_mode.real.ml` is selected instead. *)
(* Binary mode keeps the echo byte-identical on Windows, where text-mode stdout would rewrite the
   CRLF line endings git's autocrlf checkout puts in the .expected file. *)
let () =
  Stdio.Out_channel.set_binary_mode Stdio.stdout true;
  Stdio.In_channel.read_all "test_metal_storage_mode.expected" |> Stdio.print_string
