(* End-to-end streaming memory-bandwidth calibration (gh-ocannl-578): [Ocannl.Calibrate.stream]
   over tiny tensors, on the configured backend. The pass's whole point is producing calibration
   rows with EXACT byte counts — the fitter's per-leg exactness rule bars approximate rows from
   the memory leg, which is why matmul-family tuning data alone leaves
   [model_peak_memory_bandwidth] unfittable. Timings are machine-dependent, so only structural
   facts are printed: rows were appended through the ordinary tuning emission path, bytes-exact
   rows are among them, and the fit over just these rows yields a memory-leg constant.

   The calibration file is pinned by the companion dune rule
   (--ocannl_autotune_calibration_file=bandwidth_calibration.tsv) and truncated here at start, so
   reruns in the same _build directory stay bounded and self-contained. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module Cal = Ir.Cost_model.Calibration

let () =
  let file =
    String.strip (Utils.get_global_arg ~arg_name:"autotune_calibration_file" ~default:"")
  in
  assert (not (String.is_empty file));
  if Stdlib.Sys.file_exists file then Stdlib.Sys.remove file;
  let ctx = Context.auto () in
  let reports = Calibrate.stream ~elems:65536 ~repeats:1 ctx in
  Stdio.printf "kernels tuned: %s\n" (String.concat ~sep:" " (List.map reports ~f:fst));
  let rows =
    if Stdlib.Sys.file_exists file then
      List.filter_map (Stdio.In_channel.read_lines file) ~f:Cal.of_line
    else []
  in
  Stdio.printf "rows appended: %b\n" (not (List.is_empty rows));
  let exact_bytes =
    List.filter rows ~f:(fun r ->
        (not r.Cal.bytes_approx) && (not r.Cal.opaque) && r.Cal.bytes > 0
        && Float.(r.Cal.measured_ms > 0.))
  in
  Stdio.printf "bytes-exact rows present: %b\n" (not (List.is_empty exact_bytes));
  let fits = Cal.fit rows in
  Stdio.printf "single-backend fit: %b\n" (List.length fits = 1);
  Stdio.printf "memory leg fitted from these rows: %b\n"
    (List.exists fits ~f:(fun f -> Option.is_some f.Cal.fit_peak_memory_bandwidth))
