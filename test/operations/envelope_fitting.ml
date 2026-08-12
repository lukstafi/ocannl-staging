(* The calibration-TSV schema round-trip and the envelope fitter (gh-ocannl-514 phase 0):
   [Ir.Cost_model.Calibration]. Pure string/arithmetic work — no backend, no config sensitivity.
   Floats destined for the output go through [to_line]'s %f formats and [report]'s
   [Ndarray.concise_float], both exponent-portable across platforms. *)

open Base
open Ocannl.Operation.DSL_modules
module Cal = Ir.Cost_model.Calibration

let row ~backend ~digest ~label ~measured_ms ~model_ms ~kernels ~flops ~bytes ~opaque =
  { Cal.backend; digest; label; measured_ms; model_ms; kernels; flops; bytes; opaque }

let () =
  let rows =
    [
      (* Bandwidth-bound copy: zero flops, so it must not constrain the compute leg. Recorded
         before any envelope constants were set: empty model column. *)
      row ~backend:"cc" ~digest:"aaaaaaaa/11111111" ~label:"copy_preset" ~measured_ms:2.0
        ~model_ms:None ~kernels:1 ~flops:0 ~bytes:16_000_000 ~opaque:false;
      (* Compute-bound matmul whose recorded bound exceeds its measured time: a violation row —
         the envelope in force at recording time understated the machine. *)
      row ~backend:"cc" ~digest:"bbbbbbbb/22222222" ~label:"matmul bs=64" ~measured_ms:4.0
        ~model_ms:(Some 5.0) ~kernels:1 ~flops:2_000_000_000 ~bytes:3_000_000 ~opaque:false;
      (* Opaque row: excluded from the fit (its counts may under-estimate), still counted. *)
      row ~backend:"cc" ~digest:"cccccccc/33333333" ~label:"staged" ~measured_ms:1.0
        ~model_ms:None ~kernels:1 ~flops:1_000_000 ~bytes:1_000_000 ~opaque:true;
      (* Second backend, multi-kernel aggregate row: fits are grouped per backend. *)
      row ~backend:"toy" ~digest:"dddddddd/44444444" ~label:"fissioned fused" ~measured_ms:10.0
        ~model_ms:(Some 9.0) ~kernels:3 ~flops:5_000_000 ~bytes:400_000_000 ~opaque:false;
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
  let bad =
    [ "cc\tonly\tthree"; "cc\td\tl\tnot_a_number\t\t1\t0\t0\tfalse"; "cc\td\tl\t1.0\t\t1\t0\t0\tmaybe" ]
  in
  List.iter bad ~f:(fun l ->
      Stdio.printf "malformed %S parses: %b\n" l (Option.is_some (Cal.of_line l)));
  Stdio.printf "\nfits:\n";
  List.iter (Cal.fit parsed) ~f:(fun f ->
      Stdio.printf "%s" (Cal.report f);
      Stdio.printf "(scoreable %d, opaque %d, multi-kernel %d, violations %d)\n\n" f.Cal.fit_rows
        f.Cal.fit_opaque f.Cal.fit_multi_kernel f.Cal.fit_violations)
