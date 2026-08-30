let () =
  if Array.length Sys.argv <> 5 then (
    prerr_endline "usage: sweep_harness_driver HARNESS SWEEP AGGREGATE_SKIPS VERDICT_PROBE";
    exit 2);
  let argv = [| "bash"; Sys.argv.(1); Sys.argv.(2); Sys.argv.(3); Sys.argv.(4) |] in
  let pid = Unix.create_process "bash" argv Unix.stdin Unix.stdout Unix.stderr in
  match snd (Unix.waitpid [] pid) with
  | Unix.WEXITED code -> exit code
  | Unix.WSIGNALED signal | Unix.WSTOPPED signal -> exit (128 + signal)
