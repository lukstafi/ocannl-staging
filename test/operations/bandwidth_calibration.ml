(* End-to-end streaming memory-bandwidth calibration (gh-ocannl-578): [Ocannl.Calibrate.stream] over
   tiny tensors, on the configured backend. The pass's whole point is producing calibration rows
   with EXACT byte counts — the fitter's per-leg exactness rule bars approximate rows from the
   memory leg, which is why matmul-family tuning data alone leaves [model_peak_memory_bandwidth]
   unfittable. Timings are machine-dependent, so only structural facts are printed: rows were
   appended through the ordinary tuning emission path, bytes-exact rows are among them, and the fit
   over just these rows yields a memory-leg constant.

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
  Verdict.p "rows appended" (not (List.is_empty rows));
  (* Every row names the computation it timed (gh-ocannl-635) — the writer-side half of the schema,
     which only an end-to-end tuning run exercises: the name comes from the comp's block comment
     through [Autotune.tune]'s compiles. Without it the fitted memory-leg floor cannot say which
     stream kernel demonstrated it, and per-kernel rates have to be reconstructed outside the
     rows. *)
  let kernels = List.map reports ~f:fst in
  Verdict.p_all "every row names its routine" rows ~f:(fun r ->
      List.mem kernels r.Cal.routine ~equal:String.equal);
  Stdio.printf "routines named, in order: %s\n"
    (String.concat ~sep:" "
       (List.filter kernels ~f:(fun k ->
            List.exists rows ~f:(fun r -> String.equal r.Cal.routine k))));
  let exact_bytes =
    List.filter rows ~f:(fun r ->
        (not r.Cal.bytes_approx) && (not r.Cal.opaque) && r.Cal.bytes > 0
        && Float.(r.Cal.measured_ms > 0.))
  in
  Verdict.p "bytes-exact rows present" (not (List.is_empty exact_bytes));
  let fits = Cal.fit rows in
  Verdict.p "single-backend fit" (List.length fits = 1);
  Verdict.p "memory leg fitted from these rows"
    (List.exists fits ~f:(fun f -> Option.is_some f.Cal.fit_peak_memory_bandwidth))
