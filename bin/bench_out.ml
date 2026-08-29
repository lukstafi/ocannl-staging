(** The line printer every tool in [bin/] reports through.

    OCaml's stdout is block-buffered whether or not it is a terminal, and nothing here flushes it
    until the process exits. On a bench that is minutes per leg -- the normal case for these, and
    the whole point of the naive legs -- that means a run in progress is indistinguishable from a
    hang: no header, no completed rows, nothing at all until the last variant finishes. A
    [schedule_bench 512 20] on Metal was diagnosed as a hang for exactly this reason, its header
    line having been printed before the first kernel and still sitting in the buffer ten minutes
    later (gh-ocannl-829).

    So every line is flushed as it is produced. The cost is one write syscall per row, against runs
    measured in seconds per row.

    Its own library rather than a second module of {!Bench_args}, because dune allows a module in
    only one stanza and a wrapped library named after its single module is what makes [Bench_out.p]
    spell directly; separate from {!Bench_args} because the tools that print are not the tools that
    take positional geometry ([device_props] prints and parses nothing). Depends on stdio alone. *)

(** [flush ()] pushes whatever is buffered on stdout out now. For the writers that do not go through
    {!p}: [Train.printf_tree], [PrintBox_text.output Stdio.stdout] and friends render straight to
    the channel, so a tool that prints a tree or a table and then compiles for a minute needs this
    after the write, or that output waits for exit exactly as an unflushed row would. *)
let flush () = Stdio.Out_channel.flush Stdio.stdout

(** [p fmt] is [Stdio.printf fmt] followed by a flush. Bind it eta-expanded --
    [let p fmt = Bench_out.p fmt] -- so the format type stays polymorphic across a tool's several
    formats. *)
let p fmt =
  Printf.ksprintf
    (fun s ->
      Stdio.print_string s;
      flush ())
    fmt
