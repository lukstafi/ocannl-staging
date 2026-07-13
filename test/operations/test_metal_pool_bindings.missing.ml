(* Off-platform stub. The Metal backend is macOS-only, so dune's [select] (in this test's
   [libraries]) picks this module wherever [metal] is unavailable. It reproduces the recorded golden
   output so the test passes trivially; the real Metal pooled-binding path runs only on macOS, where
   [test_metal_pool_bindings.real.ml] is selected instead. *)
(* Binary mode keeps the echo byte-identical on Windows, where text-mode stdout would rewrite the
   CRLF line endings git's autocrlf checkout puts in the .expected file. *)
let () =
  Stdio.Out_channel.set_binary_mode Stdio.stdout true;
  Stdio.In_channel.read_all "test_metal_pool_bindings.expected" |> Stdio.print_string
