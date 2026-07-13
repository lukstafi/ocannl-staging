(* Off-platform stub. The real Metal source check runs only when the Metal backend is available. *)
(* Binary mode keeps the echo byte-identical on Windows, where text-mode stdout would rewrite the
   CRLF line endings git's autocrlf checkout puts in the .expected file. *)
let () =
  Stdio.Out_channel.set_binary_mode Stdio.stdout true;
  Stdio.In_channel.read_all "test_metal_guarded_gather_codegen.expected" |> Stdio.print_string
