let () =
  if Array.length Sys.argv < 2 || Array.length Sys.argv > 3 then (
    prerr_endline
      "usage: verdict_skip_probe BACKEND [execute-environment|environment-as-backend]";
    exit 2);
  Verdict.skipped ~backend:Sys.argv.(1) "common unevaluated claim";
  match if Array.length Sys.argv = 3 then Some Sys.argv.(2) else None with
  | Some "execute-environment" -> Verdict.p "common environment-gated claim" true
  | Some "environment-as-backend" ->
      Verdict.skipped ~backend:Sys.argv.(1) "common environment-gated claim"
  | Some mode -> failwith ("unknown verdict probe mode: " ^ mode)
  | None ->
      Verdict.skipped ~aggregation:`Environment ~backend:"fixture gate"
        "common environment-gated claim"
