let () =
  if Array.length Sys.argv <> 2 then (
    prerr_endline "usage: verdict_skip_probe BACKEND";
    exit 2);
  Verdict.skipped ~backend:Sys.argv.(1) "common unevaluated claim";
  Verdict.skipped ~aggregation:`Environment ~backend:"fixture gate" "common environment-gated claim"
