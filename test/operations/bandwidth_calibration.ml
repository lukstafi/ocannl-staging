(* End-to-end streaming memory-bandwidth calibration (gh-ocannl-578): [Ocannl.Calibrate.stream] over
   tiny tensors, on the configured backend. The pass's whole point is producing calibration rows
   with EXACT byte counts — the fitter's per-leg exactness rule bars approximate rows from the
   memory leg, which is why matmul-family tuning data alone leaves [model_peak_memory_bandwidth]
   unfittable. Timings are machine-dependent, so only structural facts are printed: rows were
   appended through the ordinary tuning emission path, bytes-exact rows are among them, and the fit
   over just these rows yields a memory-leg constant.

   The calibration file is pinned by the companion dune rule
   (--ocannl_autotune_calibration_file=bandwidth_calibration.tsv) and truncated here at start, so
   reruns in the same _build directory stay bounded and self-contained.

   WHICH kernels contribute rows is a property of the host, not of the pass (gh-ocannl-892). A
   timing window whose samples are mostly host stalls is refused ([Autotune.admitted_timing_ms]),
   and a refused candidate emits no row; on a busy machine a whole kernel's candidates can be
   refused, and then that kernel has no rows at all. So the golden pins the RELATIONSHIP -- a
   kernel contributed rows exactly when its search timed a candidate, and a kernel with no rows
   shows refused timings as the evidence -- rather than an ordered list of the four names, which on
   the GPU backends made the golden a function of the load the sweep happened to be under. *)

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
  let contributed name = List.exists rows ~f:(fun r -> String.equal r.Cal.routine name) in
  List.iter reports ~f:(fun (name, rep) ->
      Stdio.eprintf
        "%s (not part of the golden): %d candidate(s) timed, %d timing(s) refused, %d candidate(s) \
         failed, %s\n"
        name rep.Autotune.candidates_timed rep.Autotune.timings_contended
        rep.Autotune.candidates_failed
        (if contributed name then "contributed rows" else "NO ROWS"));
  (* The pass emits one row per admitted candidate timing and nothing anywhere else. Stated as the
     biconditional it is: a kernel whose rows went missing while its search did time something is
     the emission defect this test exists to catch, and a kernel with rows it never timed for would
     mean the [routine] column named the wrong computation. Neither is what a contended host
     produces -- that moves both sides at once. *)
  Verdict.p_all "a kernel contributed rows exactly when its search timed a candidate" reports
    ~f:(fun (name, rep) -> Bool.equal (contributed name) (rep.Autotune.candidates_timed > 0));
  (* The biconditional alone would also accept a kernel whose every candidate failed compile or
     dispatch ([candidates_failed]): nothing timed, nothing contributed, both sides false. That is
     the loss of coverage the ordered list used to catch, and it is not what load does -- a
     refused timing window increments [timings_contended]. So a kernel may go row-less only on
     that evidence. Cache replay would zero both counters, but [Calibrate.stream] passes
     [~cache_dir:""], so every search here times live. *)
  Verdict.p_all "a kernel with no rows was refused its timings, not silently lost" reports
    ~f:(fun (name, rep) -> contributed name || rep.Autotune.timings_contended > 0);
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
