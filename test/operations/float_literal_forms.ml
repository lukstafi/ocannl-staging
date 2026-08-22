(* Every emitted float constant is a floating literal that parses back to exactly the host value
   (gh-ocannl-623).

   [C_syntax.c_float_literal] renders each [Low_level.Constant] into the kernel source. Three
   properties have to hold of the token it produces, and [%.16g] alone gets all three wrong:

   - It has to be a FLOATING literal. A value with no fractional part printed by [%.16g] has no
     radix point, so it is a C INTEGER literal that the surrounding cast happens to convert. That is
     value-preserving in today's cast contexts for every value but one: [-0.] printed as ["-0"] is
     the integer zero, hence [+0.0] once cast — a live cross-backend data corruption fixed for that
     single value by gh-ocannl-615. The rest of the class is latent, not inert (integer division,
     integer promotion, a digit run outgrowing [long long]), which is what this test closes.
   - It has to ROUND-TRIP. Sixteen significant digits do not recover every double: [0.1 +. 0.2]
     prints as ["0.3"], a different double, and so does [max_float]. Arbitrary host values reach
     this printer (hosted constant inits inline them as scalar stores), so this is a live
     value-changing bug rather than a hypothetical one, and it is invisible to a tolerance-based
     comparison — hence the bitwise checks below.
   - The specials have to be SPELLED. [INFINITY] / [(-INFINITY)] / [NAN] are C99 macros MSL also
     provides and the CUDA and HIP preludes define under [#ifndef]; a NaN's payload and sign do not
     survive any of them, so NaN is checked as "is a NaN" rather than on the bits.

   Deliberately absent: a precision suffix. The literal is always double-typed and the narrowing is
   [convert_precision]'s cast — one rounding of the exact host double, the same conversion the host
   itself performs. An [f]-suffixed decimal would round the decimal straight to float instead, which
   disagrees for a value on a float tie. The f32 and f16 legs are what pin that the cast is really
   there and really does the narrowing: their oracles are the host's own conversions.

   The IR is hand-built so that each constant is stored, untouched, into its own cell of a
   materialized node: through the [Assignments] pipeline a bare constant fill is the backend's
   business to inline or upload, and only an inlined one is about this printer. *)

open Base
module LL = Ir.Low_level
module Ops = Ir.Ops
module Tn = Ir.Tnode

let () = Utils.settings.output_debug_files_in_build_directory <- true
let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")

module Generated = Test_utils.Generated

let () = Generated.init ~backend_name

let p = Verdict.p
let skipped = Verdict.skipped ~backend:backend_name

(* On the bits, because the whole point is values a tolerance — or [Float.equal], which reports
   [-0. = +0.] — cannot tell apart. NaN is the one exception: no dialect spells a payload. *)
let bitwise a b =
  if Float.is_nan a || Float.is_nan b then Float.is_nan a && Float.is_nan b
  else Int64.equal (Int64.bits_of_float a) (Int64.bits_of_float b)

(* The host's own double -> f32 and double -> f32 -> f16 narrowings: the oracles for the legs whose
   store precision is narrower than the literal's type. [FLOAT_TO_HALF((float)c)] is what the
   backends emit, so the half oracle goes through f32 too. *)
let to_f32 c = Int32.float_of_bits (Int32.bits_of_float c)
let to_f16 c = Ops.half_to_single (Ops.single_to_half (to_f32 c))

(* [value] is what the host holds; [spelling] is the exact literal the printer must produce for it,
   and [was] the pre-fix spelling that must NOT survive — checked with the cast's closing paren
   attached, since ["(float)(7)"] is otherwise a prefix of ["(float)(7.0)"]. [narrow] marks the
   values a half-precision node may hold: [Ops.exceeds_fp16_cutoff] refuses the large ones during
   lowering (the infinities are exempt there by construction, being reduction identities). *)
type case = { value : float; spelling : string; was : string option; narrow : bool }

let case ?was ?(narrow = true) value spelling = { value; spelling; was; narrow }

let cases =
  [
    (* Whole numbers: the class that lost its radix point. *)
    case 7. "7.0" ~was:"7";
    case (-13.) "-13.0" ~was:"-13";
    case 4096. "4096.0" ~was:"4096";
    case 0. "0.0" ~was:"0";
    (* The one value the missing radix point already corrupted (gh-ocannl-615). *)
    case (-0.) "-0.0" ~was:"-0";
    (* Large whole magnitudes: still all digits at [%.16g] (it switches to an exponent at 1e16), so
       this is where an integer literal's type would eventually overflow. *)
    case 1e15 "1000000000000000.0" ~was:"1000000000000000" ~narrow:false;
    case 1e20 "1e+20" ~narrow:false;
    (* Tiny: the smallest subnormal double, which flushes to zero in both narrow legs. *)
    case 5e-324 "4.940656458412465e-324" ~narrow:false;
    (* Values sixteen digits cannot recover. The last is the midpoint between [1.0f] and its
       successor: exact as a double, so [(float)(...)] takes it to [1.0f] by round-to-even, while
       its (necessarily 17-digit) decimal sits just above the midpoint and would round the other way
       if it were parsed straight to float. That is what an [f] suffix on the literal would do, so
       this case is also what pins the decision to leave the narrowing to the cast. *)
    case (0.1 +. 0.2) "0.30000000000000004";
    case Float.max_finite_value "1.7976931348623157e+308" ~narrow:false;
    case Float.(1. + (2. ** -24.)) "1.0000000596046448";
    (* Ordinary values, pinning that their spelling did not move. *)
    case 0.5 "0.5";
    case (1. /. 3.) "0.3333333333333333";
    (* The specials. *)
    case Float.infinity "INFINITY";
    case Float.neg_infinity "(-INFINITY)";
    case Float.nan "NAN";
  ]

(* One store per constant, into its own cell of a materialized node. No arithmetic anywhere: an
   identity like [c *. 1.0] would be one more place for a value to be normalized, and the store is
   what the issue is about. Cells are seeded with a sentinel none of the cases equals, so a write
   the lowering dropped fails the value check rather than reading back a plausible number. *)
let leg ~prec ~name ~oracle ~selected =
  let selected = List.filter cases ~f:selected in
  let n = List.length selected in
  let node = Ll_test.node_factory ~prec ~first_id:9600 ~dims:[| n |] () in
  let out = node "flit_out" in
  Ll_test.materialize out;
  let llc =
    List.foldi selected ~init:LL.Noop ~f:(fun i acc { value; _ } ->
        Ll_test.seq acc (Ll_test.set out [| Ll_test.fixed i |] (Ll_test.c value)))
  in
  let o = Ll_test.optimize ~name llc in
  let got = List.hd_exn (Ll_test.execute ~name o ~seed:[ (out, Ll_test.blank n) ] ~read:[ out ]) in
  let want = Array.of_list_map selected ~f:(fun { value; _ } -> oracle value) in
  (selected, got, want)

let report ~claim (selected, got, want) =
  let ok = Array.length got = Array.length want && Array.for_all2_exn got want ~f:bitwise in
  if not ok then
    Array.iteri got ~f:(fun i g ->
        if not (bitwise g want.(i)) then
          Stdio.eprintf "  %s: emitted %h, host %h\n"
            (List.nth_exn selected i).spelling
            g want.(i));
  p claim ok

(* === The values, at each store precision === *)

(* f64: the literal stands bare, with no cast at all ([convert_precision] is the identity between
   equal precisions) — so this leg is the one where the literal's own C type is the value. Metal has
   no [double] and rejects an f64 node outright, so it is gated rather than run there. *)
let () =
  if String.equal backend_name "metal" then skipped "f64 constants reach the kernel bit-exactly"
  else
    report ~claim:"f64 constants reach the kernel bit-exactly"
      (leg ~prec:Ops.double ~name:"flit_f64" ~oracle:Fn.id ~selected:(fun _ -> true))

(* f32: the literal is double-typed and [(float)(...)] narrows it, so the oracle is the host's own
   double -> f32 rounding. *)
let f32_leg =
  leg ~prec:Ops.single ~name:"flit_f32" ~oracle:to_f32 ~selected:(fun _ -> true)

let () = report ~claim:"f32 constants reach the kernel bit-exactly" f32_leg

(* f16, the representative of the narrow-float family (bf16 and fp8 differ only in which codec the
   cast names): the literal is still a plain double literal, never a dialect-specific half literal —
   [0.0h] is valid MSL and a hard error under nvrtc, so no backend may spell one here. *)
let () =
  report ~claim:"f16 constants reach the kernel as the host's own half conversion"
    (leg ~prec:Ops.half ~name:"flit_f16" ~oracle:to_f16 ~selected:(fun c -> c.narrow))

(* === The emitted tokens === *)

(* The executed legs above establish the values; this establishes that they are values of literals
   with the intended shape, so that a future value which happens to be inert cannot quietly go back
   to being an integer literal. Read off the f32 kernel: [(float)(...)] is the double -> f32 cast on
   every C-family backend, which makes the negative checks exact. *)
let () =
  let src = Generated.read "flit_f32" in
  let has s = String.is_substring src ~substring:s in
  let missing =
    List.filter_map cases ~f:(fun { spelling; _ } ->
        if has ("(float)(" ^ spelling ^ ")") then None else Some spelling)
  in
  List.iter missing ~f:(fun s -> Stdio.eprintf "  not emitted as (float)(%s)\n" s);
  p "every constant is emitted as the intended floating literal" (List.is_empty missing);
  let surviving =
    List.filter_map cases ~f:(fun { was; _ } ->
        Option.bind was ~f:(fun w -> if has ("(float)(" ^ w ^ ")") then Some w else None))
  in
  List.iter surviving ~f:(fun s -> Stdio.eprintf "  still emitted as (float)(%s)\n" s);
  p "no constant is emitted as an integer literal" (List.is_empty surviving)
