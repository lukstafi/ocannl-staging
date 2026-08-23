(* The GPU half of the e5m2 codec verification (gh-ocannl-657): OCANNL's software codec against the
   vendor's fp8 type, over every input rather than a sample.

   Why this is a tool and not a test: it needs a GPU, and it sweeps 2^32 float bit patterns and
   17.2e9 doubles. The pure-CPU half -- which pins the codec against a rounding oracle and pins the
   two codecs against each other -- is test/operations/fp8_codec_exhaustive.ml, on the `slow` alias.
   What only hardware can answer is whether the codec still agrees with `__nv_fp8_e5m2` /
   `__hip_fp8_e5m2`, which is the property gh-ocannl-646 established by changing OUR rounding to
   match THEIRS, and which a new CUDA or ROCm release can take away.

   The comparison matters because both conversions live inside one tensor: a kernel narrows with the
   vendor cast, while every host-side write of the same fp8 node goes through [Ops.single_to_fp8] /
   [Ops.double_to_fp8]. So the host side here IS the shipped codec (builtins.c, reached through
   fp8_soak_stubs.c) and the device side IS what the backend emits.

   Usage:
     dune exec tools/fp8_soak.exe                  -- every arm the box has, both sweeps
     dune exec tools/fp8_soak.exe -- --arm=cuda --sweep=f32
   Runs in a couple of minutes on an RTX 5070 Ti; see docs/agent-notes/backend-precision-and-simd.md.

   Exit status is a verdict: nonzero if any claim failed, so it can gate a release check. *)

open Base

type bytes_buf =
  (int, Stdlib.Bigarray.int8_unsigned_elt, Stdlib.Bigarray.c_layout) Stdlib.Bigarray.Array1.t

type counts_buf =
  (int64, Stdlib.Bigarray.int64_elt, Stdlib.Bigarray.c_layout) Stdlib.Bigarray.Array1.t

external soak_f32 : int64 -> int64 -> bytes_buf -> counts_buf -> unit = "ocannl_fp8_soak_f32"

external soak_f64 : int64 -> int64 -> int array -> bytes_buf -> counts_buf -> unit
  = "ocannl_fp8_soak_f64"

(* An arm narrows inputs with ONE vendor's fp8 type and says nothing about what the answer should
   be. Adding a vendor is adding a module of this shape and one `select` clause in tools/dune -- not
   a second program, which is how the CUDA sweep and the HIP sweep drifted apart the first time. *)
module type ARM = sig
  val name : string
  val vendor_type : string

  val probe : unit -> (unit, string) Result.t
  (** Whether THIS BOX can run the arm — the vendor's jit library linked AND its driver reporting a
      device — with the reason when it cannot. "Compiled in" is not the same question: a switch
      carrying both `cudajit` and `hipjit` compiles both arms, and on a machine with one kind of GPU
      the default selection must skip the other rather than raise partway through a run, possibly
      after the first vendor's several-minute sweep has already completed. An explicit [--arm] runs
      regardless, so a box whose hardware is missing or misconfigured still gets the vendor's own
      diagnosis instead of a silent skip. *)

  val describe : unit -> string

  val narrow_f32 : base:int -> count:int -> bytes_buf -> unit
  (** Fills [out.{i}] with the vendor's e5m2 code for the f32 whose bit pattern is [base + i]. *)

  val narrow_f64 : base:int -> count:int -> lows:int array -> bytes_buf -> unit
  (** Fills [out.{4*i + k}] with the vendor's e5m2 code for the double whose bit pattern is
      [(base + i) << 32 | lows.(k)]. *)
end

let arms : (module ARM) list = [ (module Fp8_soak_cuda); (module Fp8_soak_hip) ]

(* Mirrors the S_* offsets in fp8_soak_stubs.c. *)
let s_finite = 0
let s_inf = 1
let s_nan = 2
let s_inf_seen = 3
let s_nan_seen = 4
let s_inf_codes = 5
let s_nan_codes = 9
let s_all_codes = 13
let s_reported = 17
let s_records = 18
let s_max_records = 8
let s_len = s_records + (3 * s_max_records)
let two_pow_32 = 0x1_0000_0000

(* The four low halves the f64 sweep crosses every top half with: zero, one ulp up, the mantissa's
   own midpoint bit, and all ones. The midpoint is the point -- gh-ocannl-648 was a double sitting
   just off an f32 tie, and a codec that narrows via f32 first moves it ONTO the tie. *)
let low_halves = [| 0x00000000; 0x00000001; 0x80000000; 0xFFFFFFFF |]

let fresh_counts () : counts_buf =
  let b = Stdlib.Bigarray.Array1.create Stdlib.Bigarray.int64 Stdlib.Bigarray.c_layout s_len in
  Stdlib.Bigarray.Array1.fill b 0L;
  b

let get c i = Int64.to_int_exn c.{i}

let code_set (c : counts_buf) slot =
  List.filter (List.range 0 256) ~f:(fun code ->
      not
        (Int64.equal 0L
           (Int64.bit_and c.{slot + (code / 64)} (Int64.shift_left 1L (code % 64)))))

let show_codes codes =
  if List.is_empty codes then "none"
  else String.concat ~sep:" " (List.map codes ~f:(fun c -> Printf.sprintf "0x%02x" c))

(* Progress on stderr, and only to a terminal: piped into a log the carriage returns would be one
   line per percent of a sweep that already reports its wall clock. *)
let interactive = Unix.isatty Unix.stderr
let progress = ref (-1)

let show_progress label done_ total =
  let pct = 100 * done_ / total in
  if interactive && pct > !progress then (
    progress := pct;
    Stdio.eprintf "\r%s: %d%%%!" label pct)

let end_progress () =
  progress := -1;
  if interactive then Stdio.eprintf "\r%!"

(* Chunk sizes, in device bytes: 64 MiB each, so the soak keeps one modest allocation on a GPU that
   may be shared, and the transfer is large enough that neither launch nor copy latency shows. *)
let f32_chunk = 1 lsl 26
let f64_chunk = 1 lsl 24 (* top halves; four bytes of output each *)

let report_records (c : counts_buf) ~arm ~vendor =
  for k = 0 to get c s_reported - 1 do
    let base = s_records + (3 * k) in
    Stdio.eprintf "%s: input 0x%Lx -> codec 0x%02x, %s 0x%02x\n" arm c.{base} (get c (base + 1)) vendor
      (get c (base + 2))
  done

let elapsed since = Unix.gettimeofday () -. since

(* Two landmark narrowings, printed once: a tie that must go to even (1.125 -> 1.0, code 0x3C) and
   the tie between zero and the smallest subnormal (2^-17 -> +0). They are also what puts
   builtins.o on the link line -- the sweep reaches [single_to_fp8] and [double_to_fp8] through an
   `extern` declaration in C, and a member of the [ir] library's stub archive is only pulled in once
   something has referenced it, which the OCaml externals here do. *)
let codec_landmarks () =
  Printf.sprintf "single_to_fp8 1.125 -> 0x%02x, double_to_fp8 2^-17 -> 0x%02x"
    (Ir.Ops.single_to_fp8 1.125)
    (Ir.Ops.double_to_fp8 (Float.( ** ) 2. (-17.)))

let run_f32 (module A : ARM) =
  let counts = fresh_counts () in
  let buf : bytes_buf =
    Stdlib.Bigarray.Array1.create Stdlib.Bigarray.int8_unsigned Stdlib.Bigarray.c_layout f32_chunk
  in
  let started = Unix.gettimeofday () in
  let base = ref 0 in
  while !base < two_pow_32 do
    let count = Int.min f32_chunk (two_pow_32 - !base) in
    A.narrow_f32 ~base:!base ~count buf;
    soak_f32 (Int64.of_int !base) (Int64.of_int count) buf counts;
    base := !base + count;
    show_progress (A.name ^ " f32") !base two_pow_32
  done;
  end_progress ();
  (counts, elapsed started)

let run_f64 (module A : ARM) =
  let counts = fresh_counts () in
  let buf : bytes_buf =
    Stdlib.Bigarray.Array1.create Stdlib.Bigarray.int8_unsigned Stdlib.Bigarray.c_layout
      (4 * f64_chunk)
  in
  let started = Unix.gettimeofday () in
  let base = ref 0 in
  while !base < two_pow_32 do
    let count = Int.min f64_chunk (two_pow_32 - !base) in
    A.narrow_f64 ~base:!base ~count ~lows:low_halves buf;
    soak_f64 (Int64.of_int !base) (Int64.of_int count) low_halves buf counts;
    base := !base + count;
    show_progress (A.name ^ " f64") !base two_pow_32
  done;
  end_progress ();
  (counts, elapsed started)

(* Every finite e5m2 code, both signs: 0x00-0x7B and 0x80-0xFB. A sweep of all 2^32 float patterns
   must reach all of them, and a kernel that silently wrote nothing would not. *)
let all_finite_codes =
  List.filter (List.range 0 256) ~f:(fun c -> c land 0x7F <= 0x7B)

let report (module A : ARM) ~sweep ~inputs (counts, seconds) =
  let vendor = A.vendor_type in
  Stdio.printf "\n%s %s sweep: %d inputs, %.1fs\n" A.name sweep inputs seconds;
  Stdio.printf "  non-finite inputs: %d infinite, %d NaN\n" (get counts s_inf_seen)
    (get counts s_nan_seen);
  Stdio.printf "  %s codes on infinite inputs: %s (%d disagree with the codec)\n" vendor
    (show_codes (code_set counts s_inf_codes))
    (get counts s_inf);
  Stdio.printf "  %s codes on NaN inputs: %s (%d disagree with the codec)\n" vendor
    (show_codes (code_set counts s_nan_codes))
    (get counts s_nan);
  Stdio.printf "  distinct %s codes produced overall: %d\n" vendor
    (List.length (code_set counts s_all_codes));
  report_records counts ~arm:A.name ~vendor;
  Verdict.pf "the software codec and %s agree on every finite %s input the sweep covers" vendor
    sweep
    (get counts s_finite = 0);
  let produced = code_set counts s_all_codes in
  Verdict.p_all
    (Printf.sprintf "the %s %s sweep produced every signed finite e5m2 code" vendor sweep)
    all_finite_codes ~min:248
    ~f:(fun c -> List.mem produced c ~equal:Int.equal)

let usage () =
  Stdio.printf
    "fp8_soak: sweep OCANNL's e5m2 codec against a GPU vendor's fp8 type.\n\
       --arm=cuda|hip   which vendor (default: every arm this build has)\n\
       --sweep=f32|f64|both   which input set (default: both)\n"

let () =
  let arm_filter = ref None in
  let sweep = ref "both" in
  Array.iteri (Stdlib.Sys.argv) ~f:(fun i s ->
      if i > 0 then
        match String.lsplit2 s ~on:'=' with
        | Some ("--arm", v) -> arm_filter := Some (String.lowercase v)
        | Some ("--sweep", v) -> sweep := String.lowercase v
        | _ ->
            usage ();
            Stdlib.exit (if String.equal s "--help" then 0 else 2));
  (* The default takes the arms that probe clean and says out loud why it passed over each of the
     others. An arm named EXPLICITLY is never skipped silently: a box whose driver is missing or
     whose GPU is not visible gets the probe's own diagnosis as a failed verdict — which carries the
     nonzero exit an uncaught exception would, and the reason without the backtrace. *)
  let selected =
    match !arm_filter with
    | Some n ->
        List.filter arms ~f:(fun (module A : ARM) ->
            String.equal n A.name
            &&
            match A.probe () with
            | Ok () -> true
            | Error why ->
                Verdict.fail (Printf.sprintf "the %s arm can run on this box (%s)" A.name why);
                false)
    | None ->
        List.filter arms ~f:(fun (module A : ARM) ->
            match A.probe () with
            | Ok () -> true
            | Error why ->
                Stdio.eprintf "fp8_soak: skipping the %s arm: %s\n%!" A.name why;
                false)
  in
  (match (selected, !arm_filter) with
  | [], Some _ when Verdict.any_failed () -> () (* the probe already said why, as a verdict *)
  | [], _ ->
      Stdio.eprintf "fp8_soak: no arm selected; this build has: %s\n%!"
        (String.concat ~sep:", "
           (List.map arms ~f:(fun (module A : ARM) ->
                match A.probe () with
                | Ok () -> Printf.sprintf "%s (ready)" A.name
                | Error why -> Printf.sprintf "%s (%s)" A.name why)));
      Verdict.fail "at least one GPU arm can run on this box"
  | _ -> ());
  List.iter selected ~f:(fun ((module A : ARM) as arm) ->
      (* Flushed, so the arm header cannot land after a skip notice on the other stream. *)
      Stdio.printf "%s arm: %s\n%!" A.name (A.describe ());
      Stdio.printf "  codec: builtins.c single_to_fp8 / double_to_fp8 (%s); vendor: %s\n"
        (codec_landmarks ()) A.vendor_type;
      if String.(!sweep = "f32" || !sweep = "both") then
        report arm ~sweep:"f32" ~inputs:two_pow_32 (run_f32 arm);
      if String.(!sweep = "f64" || !sweep = "both") then
        report arm ~sweep:"f64" ~inputs:(4 * two_pow_32) (run_f64 arm))
