(* The benchmark runners' JSON result line, on fabricated values (gh-ocannl-676).

   [orchestrate.py] reads a cell's measurement by taking the last '{'-prefixed line of its output
   and [json.loads]ing it; a line that does not parse is reported as `!!! <label> failed` and the
   cell is dropped, after its whole measurement has been paid for. The line is one ~300-character
   format with two optional pre-formatted fragments and a nested object, and until this test
   nothing anywhere parsed it.

   The values here are the ones a happy-path run never produces and a report needs most: a diverged
   loss trajectory (OCaml's [%g] spells a non-finite float [nan] / [inf] / [-inf], none of which is
   JSON), a time that was never measured ([infinity]), and a backend diagnostic carrying quotes and
   control characters. The parse oracle is Yojson, which rejects exactly those OCaml spellings —
   the negative control at the end shows it does — while accepting [null]. *)

open Base
module V = Verdict

let parses s = match Yojson.Safe.from_string s with _ -> true | exception _ -> false
let member k j = Yojson.Safe.Util.member k j

let () =
  Stdio.printf "=== scalars ===\n";
  List.iter
    [ ("nan", Float.nan); ("inf", Float.infinity); ("-inf", Float.neg_infinity) ]
    ~f:(fun (name, v) ->
      V.p (Printf.sprintf "num %s is null" name) (String.equal (Bench_json.num v) "null");
      V.p (Printf.sprintf "fixed %s is null" name) (String.equal (Bench_json.fixed v) "null"));
  V.p "num of a finite value is the number"
    (String.equal (Bench_json.num 1.25) "1.25" && String.equal (Bench_json.fixed 1.25) "1.250");
  Stdio.printf "nums of a diverged trajectory: [%s]\n"
    (Bench_json.nums ~prec:9 [| 1.5; Float.nan; Float.infinity; Float.neg_infinity |])

(* A tuned cell whose arm A timed nothing at all and terminated on a failure whose message carries
   the characters that would invalidate the record: a quote, a backslash, a NUL and an ESC.

   The three arms are the three provenance buckets the [tune] object totals (gh-ocannl-677), so the
   line carries one of each: a search that died mid-way, a replayed cache entry, and an arm that
   neither searched nor replayed because the search was off. That last one is why [no_searches]
   exists — before it, a reader had to infer the case from [searches] and [replays] both being
   zero, which is exactly the derivation the outcome type replaced. *)
let tune =
  Bench_json.tune_object ~shipped:"B" ~searches:1 ~replays:1 ~no_searches:1
    ~arms:
      [
        Bench_json.tune_arm ~name:"A" ~state:"search-died" ~searched:true ~cache_hit:false
          ~best_ms:Float.infinity ~best_label:"tile 32x32" ~tensorized:false
          ~mma_scalar_fallbacks:0 ~mma_seeded:4 ~mma_timed:0 ~mma_best_ms:Float.infinity
          ~terminal_failure:
            (Some
               (Printf.sprintf "compile failed: \"kernel\" \\ path%c%c ESC" (Char.of_int_exn 0)
                  (Char.of_int_exn 27)));
        Bench_json.tune_arm ~name:"B" ~state:"cache-replay" ~searched:false ~cache_hit:true
          ~best_ms:0.75 ~best_label:"grid 128" ~tensorized:true ~mma_scalar_fallbacks:2
          ~mma_seeded:6 ~mma_timed:3 ~mma_best_ms:0.8 ~terminal_failure:None;
        (* Neither searched nor replayed: every counter zero, no winner to name. *)
        Bench_json.tune_arm ~name:"C" ~state:"search-disabled" ~searched:false ~cache_hit:false
          ~best_ms:Float.infinity ~best_label:"" ~tensorized:false ~mma_scalar_fallbacks:0
          ~mma_seeded:0 ~mma_timed:0 ~mma_best_ms:Float.infinity ~terminal_failure:None;
      ]

let ordinary =
  Bench_json.result_line ~backend:"cc" ~variant:"default" ~precision:"f32" ~workload:"mlp3"
    ~compile_s:2.5 ~searched:false ~p10:0.5 ~p50:0.75 ~p90:1.25 ~queued_ms:0.625 ~timed_steps:20
    ~losses:[| 2.5; 1.75; 1.25 |] ()

(* Everything a diverged, half-measured, tuned cell reports at once. *)
let diverged =
  Bench_json.result_line ~backend:"metal" ~variant:"tuned" ~precision:"f16" ~workload:"gpt2_mini"
    ~compile_s:Float.nan ~searched:true ~tokens_per_step:4096 ~tune
    ~p10:Float.infinity ~p50:Float.nan ~p90:Float.neg_infinity ~queued_ms:Float.nan ~timed_steps:0
    ~losses:[| 1.5; Float.nan; Float.infinity; Float.neg_infinity |] ()

let () =
  Stdio.printf "\n=== ordinary cell ===\n%s\n" ordinary;
  Stdio.printf "\n=== diverged cell ===\n%s\n" diverged;
  Stdio.printf "\n=== verdicts ===\n";
  List.iter
    [ ("ordinary", ordinary); ("diverged", diverged) ]
    ~f:(fun (name, line) ->
      V.p (Printf.sprintf "%s line parses as JSON" name) (parses line);
      V.p
        (Printf.sprintf "%s line is one line" name)
        (not (String.exists line ~f:(fun c -> Char.equal c '\n' || Char.equal c '\r')));
      V.p
        (Printf.sprintf "%s line has no byte below U+0020" name)
        (not (String.exists line ~f:(fun c -> Char.to_int c < 0x20))))

let () =
  let j = Yojson.Safe.from_string diverged in
  let losses = member "losses" j in
  V.p "a diverged loss trajectory keeps its finite steps and nulls the rest"
    (Yojson.Safe.equal losses (`List [ `Float 1.5; `Null; `Null; `Null ]));
  V.p "an unmeasured time is null, not a number"
    (List.for_all [ "p10"; "p50"; "p90" ] ~f:(fun p ->
         Yojson.Safe.equal (member p (member "step_ms" j)) `Null)
    && Yojson.Safe.equal (member "queued_step_ms" j) `Null
    && Yojson.Safe.equal (member "compile_s" j) `Null);
  let arm_a = List.hd_exn (Yojson.Safe.Util.to_list (member "arms" (member "tune" j))) in
  V.p "an arm that timed nothing reports null times"
    (Yojson.Safe.equal (member "best_ms" arm_a) `Null
    && Yojson.Safe.equal (member "mma_best_ms" arm_a) `Null);
  V.p "a diagnostic survives as a scrubbed string"
    (match member "terminal_failure" arm_a with
    | `String s -> String.is_prefix s ~prefix:"compile failed: 'kernel' / path"
    | _ -> false)

(* The negative control: without the mapping the line carries OCaml's own spellings, and this
   oracle rejects each of them — which is what makes the verdicts above evidence rather than
   ceremony. (A JSON parser that admits `NaN` as an extension still rejects `nan`.) *)
let () =
  V.p "the pre-fix spellings do not parse"
    (List.for_all [ "nan"; "inf"; "-inf" ] ~f:(fun spelling ->
         not (parses (Printf.sprintf {|{"losses":[%s]}|} spelling))));
  V.p "OCaml's own float conversion spells them the way this oracle rejects"
    (List.for_all [ Float.nan; Float.infinity; Float.neg_infinity ] ~f:(fun v ->
         not (parses (Printf.sprintf {|{"losses":[%.9g]}|} v))))
