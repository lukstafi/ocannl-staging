(* Exhaustive verification of the e5m2 codecs, CPU-only (gh-ocannl-657).

   gh-ocannl-646 and gh-ocannl-648 settled what [single_to_fp8] and [double_to_fp8] do -- ties to
   even, subnormals rounded rather than flushed, signed zero kept, finite overflow saturating, and
   f64 narrowed in ONE step -- and they were settled by sweeping every input, not by sampling: every
   defect found lived in the tails (the negative-NaN sign, the NaN payload, the subnormal boundary,
   the tie-to-even carry). Those sweeps ran in a scratch directory and were thrown away. This is the
   half of them that needs no GPU; tools/fp8_soak.ml is the half that does.

   Three things are checked, each over its whole input set:

   - [single_to_fp8] against a rounding ORACLE, over all 2^32 f32 bit patterns. The oracle is not a
   second codec: it is the format's own decode table plus the rule that a code owns the interval
   between the midpoints to its neighbours, with a midpoint going to the even code. That makes
   "correctly rounded" a local property, cheap enough to evaluate on every input. - [double_to_fp8]
   against [single_to_fp8] on all 2^32 f32-exact doubles -- the cross-check that found the two
   codecs disagreeing on the sign of a NaN -- and against the same oracle over 17.2e9 doubles that
   are NOT f32-exact (every top half crossed with four low halves, the midpoint pattern among them,
   which is the shape gh-ocannl-648 lived in). - [fp8_to_single] over all 256 codes: the exact
   decoded value, strict monotonicity (which is what licenses the oracle above), and the round trip
   back through both narrowing codecs.

   The sweeps are in fp8_codec_exhaustive_stubs.c, calling the very functions [Ops] exposes, so what
   runs here is the shipped codec rather than a transcription of it. Under [@slow] because it takes
   seconds rather than milliseconds; [@check] still compiles it. *)

open Base

type sweep_buf =
  (int64, Stdlib.Bigarray.int64_elt, Stdlib.Bigarray.c_layout) Stdlib.Bigarray.Array1.t

external sweep_init : unit -> unit = "ocannl_fp8_sweep_init"
external sweep_f32 : int64 -> int64 -> sweep_buf -> unit = "ocannl_fp8_sweep_f32"
external sweep_f64 : int64 -> int64 -> sweep_buf -> unit = "ocannl_fp8_sweep_f64"
external reference_decode : int -> float = "ocannl_fp8_reference_decode"

(* Mirrors the OUT_* offsets in fp8_codec_exhaustive_stubs.c. *)
let out_cross = 0
let out_rounding = 1
let out_sign = 2
let out_overflow = 3
let out_special = 4
let out_reached = 5
let out_reported = 9
let out_records = 10
let max_records = 8
let out_len = out_records + (3 * max_records)
let two_pow_32 = 0x1_0000_0000

let fresh_buf () : sweep_buf =
  let b = Stdlib.Bigarray.Array1.create Stdlib.Bigarray.int64 Stdlib.Bigarray.c_layout out_len in
  Stdlib.Bigarray.Array1.fill b 0L;
  b

(* [run sweep total] splits [0, total) into one contiguous range per domain and adds the results up.
   Capped at 8 domains rather than [Domain.recommended_domain_count ()]: the sweep is memory-free
   and scales linearly, so the cap only decides how much of a shared machine it takes, and the slow
   rule already holds the training lock against the rest of the suite. *)
let domains = Int.min 8 (Domain.recommended_domain_count ())

let run sweep total =
  let chunk = (total + domains - 1) / domains in
  let bufs = Array.init domains ~f:(fun _ -> fresh_buf ()) in
  let spawned =
    Array.init domains ~f:(fun i ->
        let base = i * chunk in
        let count = Int.max 0 (Int.min chunk (total - base)) in
        Domain.spawn (fun () -> sweep (Int64.of_int base) (Int64.of_int count) bufs.(i)))
  in
  Array.iter spawned ~f:Domain.join;
  bufs

let sum bufs i = Array.fold bufs ~init:0 ~f:(fun acc b -> acc + Int64.to_int_exn b.{i})

let reached bufs =
  Array.init 4 ~f:(fun w ->
      Array.fold bufs ~init:0L ~f:(fun acc b -> Int64.bit_or acc b.{out_reached + w}))

(* The offender records, in chunk order, so a failing run names the same inputs every time. *)
let records bufs =
  Array.concat_map bufs ~f:(fun b ->
      Array.init
        (Int64.to_int_exn b.{out_reported})
        ~f:(fun k ->
          let base = out_records + (3 * k) in
          (b.{base}, Int64.to_int_exn b.{base + 1}, Int64.to_int_exn b.{base + 2})))

let reason_name = function
  | 1 -> "single_to_fp8 and double_to_fp8 disagree"
  | 2 -> "not the nearest code (ties to even)"
  | 3 -> "sign not carried over"
  | 4 -> "a finite input narrowed to infinity or NaN"
  | 5 -> "an infinity or NaN narrowed to the wrong code"
  | r -> Printf.sprintf "unknown reason %d" r

(* Offenders go to stderr with their exact bit patterns: they are what a failing run needs and what
   a passing golden must not carry. A record's second field is the produced code, except for a
   cross-codec disagreement, where it packs both ([single] << 8 | [double]). *)
let report_records label rs =
  Array.iter rs ~f:(fun (bits, code, reason) ->
      if reason = 1 then
        Stdio.eprintf "%s: input 0x%Lx -> single_to_fp8 0x%02x, double_to_fp8 0x%02x (%s)\n" label
          bits (code lsr 8) (code land 0xFF) (reason_name reason)
      else Stdio.eprintf "%s: input 0x%Lx -> 0x%02x (%s)\n" label bits code (reason_name reason))

let report_counts label bufs =
  let named =
    [
      ("cross-codec disagreements", out_cross);
      ("misroundings", out_rounding);
      ("sign losses", out_sign);
      ("finite inputs narrowed to a non-finite code", out_overflow);
      ("infinities or NaNs on the wrong code", out_special);
    ]
  in
  List.iter named ~f:(fun (what, i) ->
      let n = sum bufs i in
      if n > 0 then Stdio.eprintf "%s: %d %s (not part of the golden)\n" label n what)

let bits v = Int64.bits_of_float v
let same a b = Int64.equal (bits a) (bits b)

let () =
  sweep_init ();
  let started = Unix.gettimeofday () in
  let f32 = run sweep_f32 two_pow_32 in
  let f32_done = Unix.gettimeofday () in
  let f64 = run sweep_f64 two_pow_32 in
  let f64_done = Unix.gettimeofday () in
  Stdio.printf "fp8 (e5m2) exhaustive codec sweep\n";
  Stdio.printf "  f32 bit patterns swept: %d\n" two_pow_32;
  Stdio.printf "  doubles swept: %d\n" (4 * two_pow_32);
  Stdio.eprintf "f32 sweep: %.1fs; f64 sweep: %.1fs; %d domains (not part of the golden)\n"
    (f32_done -. started) (f64_done -. f32_done) domains;
  report_counts "f32 sweep" f32;
  report_counts "f64 sweep" f64;
  report_records "f32 sweep" (records f32);
  report_records "f64 sweep" (records f64);

  Verdict.p "single_to_fp8 and double_to_fp8 agree on every one of the 2^32 f32-exact doubles"
    (sum f32 out_cross = 0);
  Verdict.p "single_to_fp8 rounds every f32 input to the nearest e5m2 code, ties to even"
    (sum f32 out_rounding = 0);
  Verdict.p "single_to_fp8 carries the sign of every f32 input over, zeros and NaNs included"
    (sum f32 out_sign = 0);
  Verdict.p "no finite f32 input narrows to an infinity or a NaN code" (sum f32 out_overflow = 0);
  Verdict.p "every infinite f32 input narrows to 0x7C and every NaN to 0x7F, keeping its sign"
    (sum f32 out_special = 0);
  Verdict.p "double_to_fp8 rounds all 17.2e9 swept doubles to the nearest e5m2 code, ties to even"
    (sum f64 out_rounding = 0 && sum f64 out_overflow = 0);
  Verdict.p "double_to_fp8 carries the sign of every swept double over"
    (sum f64 out_sign = 0 && sum f64 out_special = 0);

  (* Non-vacuity: the codec this replaced could not emit an e5m2 subnormal from any narrowing at
     all, so "no disagreement" has to be read together with "and every code is reachable". Codes
     0x7D and 0x7E (and their negatives) are the non-canonical NaN payloads, which narrowing never
     produces -- a NaN takes 0x7F. *)
  let reachable = reached f32 in
  let is_reached c =
    not (Int64.equal 0L (Int64.bit_and reachable.(c / 64) (Int64.shift_left 1L (c % 64))))
  in
  let expected_reachable c = c land 0x7F <> 0x7D && c land 0x7F <> 0x7E in
  let n_reached = List.count (List.range 0 256) ~f:is_reached in
  Stdio.printf "  distinct e5m2 codes produced by narrowing: %d\n" n_reached;
  Verdict.p_all "narrowing f32 reaches every e5m2 code but the non-canonical NaN payloads 0x7D/0x7E"
    (List.range 0 256) ~min:256 ~f:(fun c -> Bool.equal (is_reached c) (expected_reachable c));

  (* All 256 codes, widened. The reference decode is the format read off its fields, computed in the
     stub next to the oracle -- so this is also what licenses the oracle's use of midpoints. *)
  let widened = Array.init 256 ~f:Ir.Ops.fp8_to_single in
  let decode_ok c =
    let m = c land 0x7F in
    let signed v = if c land 0x80 <> 0 then Float.neg v else v in
    if m <= 0x7B then same widened.(c) (signed (reference_decode m))
    else if m = 0x7C then Float.equal widened.(c) (signed Float.infinity)
    else Float.is_nan widened.(c)
  in
  Verdict.p_all
    "fp8_to_single decodes all 256 codes to their exact e5m2 values, signed zero included"
    (List.range 0 256) ~min:256 ~f:decode_ok;
  Verdict.p_all "fp8_to_single is strictly increasing over the 124 non-negative finite codes"
    (List.range 0 0x7B) ~min:0x7B ~f:(fun c -> Float.( < ) widened.(c) widened.(c + 1));
  Verdict.p_all "every non-NaN code survives a widen-and-narrow round trip through both codecs"
    (List.range 0 256) ~min:256 ~f:(fun c ->
      let m = c land 0x7F in
      m >= 0x7D || (Ir.Ops.single_to_fp8 widened.(c) = c && Ir.Ops.double_to_fp8 widened.(c) = c));
  Verdict.p_all "every NaN code widens to a NaN and narrows back to the canonical NaN code"
    (List.range 0 256) ~min:256 ~f:(fun c ->
      let m = c land 0x7F in
      m < 0x7D
      || Float.is_nan widened.(c)
         && Ir.Ops.single_to_fp8 widened.(c) land 0x7F = 0x7F
         && Ir.Ops.double_to_fp8 widened.(c) land 0x7F = 0x7F);

  (* The C entry points' argument guards, exercised. They exist because an [external] is a hole in
     the type system -- nothing in [int64 -> int64 -> sweep_buf -> unit] says how long the buffer
     must be, and the decode table holds only the 0x7C finite magnitudes -- and a guard that has
     never fired is a claim about nothing. Cheap enough to check every run. *)
  let refuses f =
    try
      f ();
      false
    with Invalid_argument _ -> true
  in
  let one_word : sweep_buf =
    Stdlib.Bigarray.Array1.create Stdlib.Bigarray.int64 Stdlib.Bigarray.c_layout 1
  in
  Verdict.p "the reference decode refuses a code that is not a finite e5m2 magnitude"
    (refuses (fun () -> ignore (reference_decode 0x7C : float))
    && refuses (fun () -> ignore (reference_decode 0x7F : float)));
  Verdict.p "a sweep refuses a counters buffer too short to hold its results"
    (refuses (fun () -> sweep_f32 0L 1L one_word) && refuses (fun () -> sweep_f64 0L 1L one_word));
  Verdict.p "a sweep refuses a negative base or count"
    (refuses (fun () -> sweep_f32 (-1L) 1L (fresh_buf ()))
    && refuses (fun () -> sweep_f64 0L (-1L) (fresh_buf ())));

  (* The format's landmarks, as hex floats so the golden is platform-independent: what the codes at
     the boundaries the tails live on actually mean. *)
  Stdio.printf "  smallest subnormal (0x01): %s\n" (Test_utils.hex_float widened.(0x01));
  Stdio.printf "  largest subnormal (0x03): %s\n" (Test_utils.hex_float widened.(0x03));
  Stdio.printf "  smallest normal (0x04): %s\n" (Test_utils.hex_float widened.(0x04));
  Stdio.printf "  largest finite (0x7B): %s\n" (Test_utils.hex_float widened.(0x7B));
  Stdio.printf "  negative zero (0x80): %s\n" (Test_utils.hex_float widened.(0x80))
