(* Fit per-machine roofline envelope constants (config [model_peak_flops] /
   [model_peak_memory_bandwidth]) from calibration data recorded by [Autotune.tune] under config
   [autotune_calibration_file] (gh-ocannl-514 phase 0; schema and fit semantics in
   [Ir.Cost_model.Calibration]).

   Usage: dune exec tools/fit_envelope.exe -- <calibration.tsv> [<more.tsv> ...]

   Prints config-pasteable [model_peak_*] fits, grouped per backend, to stdout. Files from
   several tuning runs on the same machine can be concatenated or passed together — the fit only
   tightens with more rows. Linking [ir] makes this a config consumer like any OCANNL
   executable, so config startup chatter can land on stdout ahead of the report; [--ocannl_*]
   flags are left to the config machinery (pass [--ocannl_suppress_welcome_message=true
   --ocannl_log_config_sourcing=false] for clean redirectable output). *)

open Base
module Cal = Ir.Cost_model.Calibration

let () =
  let files =
    List.tl_exn (Array.to_list Stdlib.Sys.argv)
    |> List.filter ~f:(fun a -> not (String.is_prefix a ~prefix:"--ocannl_"))
  in
  if List.is_empty files then (
    Stdio.eprintf "usage: fit_envelope <calibration.tsv> [<more.tsv> ...]\n";
    Stdlib.exit 2);
  let malformed = ref 0 in
  let rows =
    List.concat_map files ~f:(fun file ->
        List.filter_map (Stdio.In_channel.read_lines file) ~f:(fun line ->
            if String.is_empty (String.strip line) then None
            else
              match Cal.of_line line with
              | Some _ as r -> r
              | None ->
                  Int.incr malformed;
                  None))
  in
  if !malformed > 0 then Stdio.eprintf "fit_envelope: skipped %d malformed line(s)\n" !malformed;
  if List.is_empty rows then (
    Stdio.eprintf "fit_envelope: no calibration rows\n";
    Stdlib.exit 1);
  List.iter (Cal.fit rows) ~f:(fun f -> Stdio.print_string (Cal.report f))
