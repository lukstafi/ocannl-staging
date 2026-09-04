let () =
  if Array.length Sys.argv < 2 || Array.length Sys.argv > 3 then (
    prerr_endline "usage: verdict_skip_probe BACKEND [execute-environment]";
    exit 2);
  Verdict.skipped ~backend:Sys.argv.(1) "common unevaluated claim";
  if Array.length Sys.argv = 3 && String.equal Sys.argv.(2) "execute-environment" then
    Verdict.p "common environment-gated claim" true
  else
    Verdict.skipped ~aggregation:`Environment ~backend:"fixture gate"
      "common environment-gated claim"
