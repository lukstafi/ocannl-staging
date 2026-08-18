(* The calibration-TSV schema round-trip and the envelope fitter (gh-ocannl-514 phase 0):
   [Ir.Cost_model.Calibration]. Pure string/arithmetic work — no backend, no config sensitivity.
   Floats destined for the output go through [to_line]'s %f formats and [report]'s
   [Ndarray.concise_float], both exponent-portable across platforms. *)

open Base
open Ocannl.Operation.DSL_modules
module Cal = Ir.Cost_model.Calibration

let row ~backend ~digest ?(routine = "") ~label ~measured_ms ~model_ms ~kernels ~flops ~bytes
    ?(flops_approx = false) ?(bytes_approx = false) ?(opaque = false) () =
  {
    Cal.backend;
    digest;
    routine;
    label;
    measured_ms;
    model_ms;
    kernels;
    flops;
    bytes;
    flops_approx;
    bytes_approx;
    opaque;
  }

let () =
  let rows =
    [
      (* Bandwidth-bound copy: zero flops, so it must not constrain the compute leg. Recorded
         before any envelope constants were set: empty model column. Measured time is floored at
         the 6th decimal on serialization (never rounded up), so the sub-ns tail vanishes. *)
      row ~backend:"cc" ~digest:"aaaaaaaa/11111111" ~routine:"stream_copy" ~label:"copy_preset"
        ~measured_ms:2.0000004999 ~model_ms:None ~kernels:1 ~flops:0 ~bytes:16_000_000 ();
      (* Compute-bound matmul whose recorded bound exceeds its measured time: a violation row —
         the envelope in force at recording time understated the machine. *)
      row ~backend:"cc" ~digest:"bbbbbbbb/22222222" ~routine:"matmul_fwd" ~label:"matmul bs=64"
        ~measured_ms:4.0 ~model_ms:(Some 5.0) ~kernels:1 ~flops:2_000_000_000
        ~bytes:3_000_000 ();
      (* Approx-flops row (guards-taken over-counting): its fake 100x compute throughput must
         not drive the compute leg, and its recorded model > measured exceedance must not count
         as a bound violation (possible over-count, not an understated envelope). Exactness is
         per leg: the exact bytes count still feeds the memory leg — with the highest achieved
         bandwidth here, this row must become the memory-leg binding row. *)
      row ~backend:"cc" ~digest:"eeeeeeee/55555555" ~routine:"conv_fwd" ~label:"masked fringe"
        ~measured_ms:1.0 ~model_ms:(Some 2.5) ~kernels:1 ~flops:50_000_000_000
        ~bytes:9_000_000 ~flops_approx:true ();
      (* Opaque row: excluded from the fit (its counts may under-estimate), still counted. *)
      row ~backend:"cc" ~digest:"cccccccc/33333333" ~routine:"staged_gemm" ~label:"staged"
        ~measured_ms:1.0 ~model_ms:None ~kernels:1 ~flops:1_000_000 ~bytes:1_000_000
        ~opaque:true ();
      (* Second backend, multi-kernel aggregate row: fits are grouped per backend. With both
         legs binding on this single row, each leg's necessary maximum alone equals the measured
         time, so the aggregate sufficient condition (flops/pf + bytes/pb <= t) forces fission
         slack 2 — the recomputed per-kernel-summed bound then respects the row. *)
      row ~backend:"toy" ~digest:"dddddddd/44444444" ~routine:"mlp_step" ~label:"fissioned fused"
        ~measured_ms:10.0 ~model_ms:(Some 9.0) ~kernels:3 ~flops:5_000_000 ~bytes:400_000_000
        ();
    ]
  in
  let lines = List.map rows ~f:Cal.to_line in
  Stdio.printf "emitted TSV (tabs shown as ' | '):\n";
  List.iter lines ~f:(fun l ->
      Stdio.printf "  %s\n" (String.substr_replace_all l ~pattern:"\t" ~with_:" | "));
  let parsed = List.filter_map lines ~f:Cal.of_line in
  Stdio.printf "\nround-trip: %d/%d rows parsed back, re-emission identical: %b\n"
    (List.length parsed) (List.length lines)
    (List.equal String.equal lines (List.map parsed ~f:Cal.to_line));
  (* Unparseable numbers are rejected at either column count — these carry the 11 columns of the
     legacy schema below, so they also pin that the legacy arm validates as strictly. *)
  let bad =
    [
      "cc\tonly\tthree";
      "cc\td\tl\tnot_a_number\t\t1\t0\t0\tfalse\tfalse\tfalse";
      "cc\td\tl\t1.0\t\t1\t0\t0\tfalse\tfalse\tmaybe";
    ]
  in
  List.iter bad ~f:(fun l ->
      Verdict.p (Printf.sprintf "malformed %S rejected" l) (Option.is_none (Cal.of_line l)));
  (* A row from before the routine column (gh-ocannl-635) keeps fitting: it carries every
     exactness flag a leg needs, so a file accumulated across builds does not lose its history —
     such a row only stops naming its computation, and its witness falls back to the bare
     candidate label. *)
  let legacy =
    Cal.of_line
      "cc\tffffffff/66666666\tlegacy_preset\t3.000000\t\t1\t0\t12000000\tfalse\tfalse\tfalse"
  in
  Verdict.p "legacy 11-column row parses" (Option.is_some legacy);
  Option.iter legacy ~f:(fun r ->
      Stdio.printf "\nlegacy row re-emitted in the current schema:\n  %s\n"
        (String.substr_replace_all (Cal.to_line r) ~pattern:"\t" ~with_:" | ");
      Verdict.p "legacy row carries no routine" (String.is_empty r.Cal.routine);
      Verdict.p "an unnamed row names itself by its label alone"
        (String.equal (Cal.row_name r) "legacy_preset"));
  (* A tab in a name would otherwise split its row into fragments no reader can parse: the cell
     loses the tab, the row survives. *)
  let tabbed =
    row ~backend:"cc" ~digest:"99999999/99999999" ~routine:"odd\tname" ~label:"pre\nset"
      ~measured_ms:1.0 ~model_ms:None ~kernels:1 ~flops:0 ~bytes:1_000 ()
  in
  Verdict.p "tabs and newlines in names do not split the row"
    (match Cal.of_line (Cal.to_line tabbed) with
    | Some r -> String.equal (Cal.row_name r) "odd name/pre set"
    | None -> false);
  let for_fit = parsed @ Option.to_list legacy in
  Stdio.printf "\nfits:\n";
  List.iter (Cal.fit for_fit) ~f:(fun f ->
      Stdio.printf "%s" (Cal.report f);
      Stdio.printf
        "(timed %d, opaque %d, approx-flops %d, approx-bytes %d, multi-kernel %d, violations \
         %d)\n\n"
        f.Cal.fit_rows f.Cal.fit_opaque f.Cal.fit_flops_approx f.Cal.fit_bytes_approx
        f.Cal.fit_multi_kernel f.Cal.fit_violations)
