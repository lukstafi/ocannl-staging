(* Fit per-machine roofline envelope constants (config [model_peak_flops] /
   [model_peak_memory_bandwidth]) from calibration data recorded by [Autotune.tune] under config
   [autotune_calibration_file] (gh-ocannl-514 phase 0; schema and fit semantics in
   [Ir.Cost_model.Calibration]).

   Usage: dune exec tools/fit_envelope.exe -- [--backend=NAME] [--margin=F] <calibration.tsv>...

   Prints config-pasteable [model_peak_*] fits to stdout. The keys are global, so only a
   single-backend report is pasteable as-is: pass [--backend] to select one when the data spans
   several (without it, multi-backend output is printed for inspection with a stderr warning).
   [--margin] multiplies the printed constants — headroom against candidates faster than any yet
   measured (fitted peaks are floors, not certified maxima), at the cost of proportionally
   looser bounds. Files from several tuning runs on the same machine can be concatenated or
   passed together — the fit only tightens with more rows.

   Linking [ir] makes this a config consumer like any OCANNL executable, so config startup
   chatter can land on stdout ahead of the report; [--ocannl_*] flags are left to the config
   machinery (pass [--ocannl_suppress_welcome_message=true --ocannl_log_config_sourcing=false]
   for clean redirectable output). *)

open Base
module Cal = Ir.Cost_model.Calibration

let usage () =
  Stdio.eprintf "usage: fit_envelope [--backend=NAME] [--margin=F] <calibration.tsv>...\n";
  Stdlib.exit 2

let () =
  let args =
    List.tl_exn (Array.to_list Stdlib.Sys.argv)
    |> List.filter ~f:(fun a -> not (String.is_prefix a ~prefix:"--ocannl_"))
  in
  let backend = ref None and margin = ref 1.0 and files = ref [] in
  List.iter args ~f:(fun a ->
      match String.chop_prefix a ~prefix:"--backend=" with
      | Some b -> backend := Some b
      | None -> (
          match String.chop_prefix a ~prefix:"--margin=" with
          | Some m -> (
              match Float.of_string m with
              | f when Float.(f >= 1.) -> margin := f
              | _ | (exception _) ->
                  (* Below 1 the "headroom" would shrink the peaks — reintroducing exactly the
                     bound violations the fit just removed. *)
                  Stdio.eprintf "fit_envelope: --margin needs a float >= 1.0, got %s\n" m;
                  Stdlib.exit 2)
          | None ->
              if String.is_prefix a ~prefix:"--" then usage () else files := a :: !files));
  let files = List.rev !files in
  if List.is_empty files then usage ();
  let malformed = ref 0 and legacy = ref 0 in
  let rows =
    List.concat_map files ~f:(fun file ->
        List.filter_map (Stdio.In_channel.read_lines file) ~f:(fun line ->
            if String.is_empty (String.strip line) then None
            else
              match Cal.of_line line with
              | Some _ as r -> r
              | None ->
                  (* Rows recorded before the approx column (the original 9-column schema of
                     gh-ocannl-491) cannot prove their counts exact, so they cannot enter the
                     fit — but dropping them as generic garbage would hide that the file's
                     fastest candidates may be missing. Name them explicitly. *)
                  if List.length (String.split line ~on:'\t') = 9 then Int.incr legacy
                  else Int.incr malformed;
                  None))
  in
  if !legacy > 0 then
    Stdio.eprintf
      "fit_envelope: skipped %d legacy 9-column row(s) (recorded before the approx-count \
       column): they cannot prove exact counts, so the fit may miss the fastest previously \
       observed candidates — prefer re-recording calibration data with the current build\n"
      !legacy;
  if !malformed > 0 then Stdio.eprintf "fit_envelope: skipped %d malformed line(s)\n" !malformed;
  if List.is_empty rows then (
    Stdio.eprintf "fit_envelope: no calibration rows\n";
    Stdlib.exit 1);
  let fits = Cal.fit rows in
  let fits =
    match !backend with
    | None -> fits
    | Some b -> (
        match List.filter fits ~f:(fun f -> String.equal f.Cal.fit_backend b) with
        | [] ->
            Stdio.eprintf "fit_envelope: no rows for backend %s (backends present: %s)\n" b
              (String.concat ~sep:", " (List.map fits ~f:(fun f -> f.Cal.fit_backend)));
            Stdlib.exit 1
        | selected -> selected)
  in
  (* The model_peak_* keys are global, so concatenated per-backend sections repeat them and are
     not pasteable into one config file. *)
  if List.length fits > 1 then
    Stdio.eprintf
      "fit_envelope: %d backends in the data (%s) — sections below repeat the global \
       model_peak_* keys; pass --backend=NAME for a config-pasteable single section\n"
      (List.length fits)
      (String.concat ~sep:", " (List.map fits ~f:(fun f -> f.Cal.fit_backend)));
  let with_margin f =
    let scale = Option.map ~f:(fun (v, w) -> (v *. !margin, w)) in
    {
      f with
      Cal.fit_peak_flops = scale f.Cal.fit_peak_flops;
      fit_peak_memory_bandwidth = scale f.Cal.fit_peak_memory_bandwidth;
    }
  in
  List.iter fits ~f:(fun f -> Stdio.print_string (Cal.report (with_margin f)))
