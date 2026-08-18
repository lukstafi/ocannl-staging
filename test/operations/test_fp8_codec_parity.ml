(* The fp8 (e5m2) narrowing codec is ONE rounding, on the host and in every backend's kernels
   (gh-ocannl-638).

   A tensor's fp8 cells get written from two places: the host, through [Ndarray]'s
   [Ops.single_to_fp8] (the C stub compiled from builtins.c), and a kernel, through whatever the
   backend emits — the same software codec on cc and Metal, the native [__nv_fp8_e5m2] /
   [__hip_fp8_e5m2] cast on CUDA and HIP. Nothing makes those agree by construction, so this
   compares them directly: the same values narrowed each way, read back and compared.

   The values are the ones where a codec has a decision to make, because everywhere else agreement
   is uninformative — ties in both directions (round-to-nearest-EVEN, not away from zero), the
   subnormal range (rounded into, not flushed — an e5m2 subnormal is unreachable for a codec that
   flushes), the sign of a zero, and overflow of a finite value (saturating to the largest finite,
   not going to infinity). Those four were the software codec's arbitrary choices and the native
   types' considered ones; they are the native types' now.

   Two inputs are deliberately NOT pinned, because the vendors disagree with each other and
   OCANNL does not emit the conversion itself on those backends: an already-INFINITE input, which
   CUDA saturates to the largest finite while HIP keeps it infinite, and the SIGN of a NaN, which
   CUDA drops and HIP keeps. Each is asserted below only up to the property all four agree on. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules

let p = Verdict.p
let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")
let on_hip = String.is_substring backend_name ~substring:"hip"

(* A leg this backend cannot run still prints its line, so the golden stays backend-uniform, and
   the skip is announced on stderr (the convention of schedule_mma_matmul.ml). *)
let skipped name =
  Stdio.eprintf "SKIPPED on %s (vacuous): %s\n%!" backend_name name;
  p name true

(* Narrow on the DEVICE. [*. 1.] is the exact identity on every value below — signed zeros,
   infinities and NaNs included — so the only thing between the source and the fp8 buffer is the
   backend's float-to-fp8 conversion. Reading back applies the HOST's widening, which is exact and
   agrees everywhere (all 256 codes, checked against both GPUs), so a difference in what comes
   back is a difference in the narrowing. *)
let narrow_on_device values =
  Tensor.unsafe_reinitialize ();
  let ctx = Context.auto () in
  let src =
    TDSL.ndarray values ~label:[ "codec_src" ] ~output_dims:[ Array.length values ] ()
  in
  let dst = TDSL.O.( *. ) src (TDSL.number 1.0) in
  Ir.Tnode.update_prec dst.Tensor.value Ir.Ops.fp8;
  Train.set_materialized dst.Tensor.value;
  let ctx = Train.forward_once ctx dst in
  Context.get_values ctx dst.Tensor.value

(* The same values narrowed on the host: [Ndarray.set_from_float] runs [Ops.single_to_fp8]. *)
let narrow_on_host values =
  Array.map values ~f:(fun v -> Ir.Ops.fp8_to_single (Ir.Ops.single_to_fp8 v))

let two_pow n = Float.(2. ** of_int n)

let decisive =
  [|
    (* Ties between adjacent e5m2 values: to even, not away from zero. *)
    1.125; 1.375; 1.625; 2.25; -1.125; -1.625; 0.140625;
    (* The subnormal range: a flushing codec returns zero for all but the first. *)
    two_pow (-14); two_pow (-15); two_pow (-16); 2.5 *. two_pow (-16); 1e-5; -1e-5;
    (* The tie between zero and the smallest subnormal, which goes to zero (even). *)
    two_pow (-17);
    (* Signed zeros. *)
    0.0; -0.0;
    (* Overflow of a FINITE value: saturates to the largest finite. *)
    57344.0; 60000.0; 61440.0; 65504.0; 1e30; -1e30;
    (* Ordinary values, as controls. *)
    1.0; -2.0; 0.5; 100.0; -0.001;
  |]

let () =
  let host = narrow_on_host decisive in
  let device = narrow_on_device decisive in
  (* The values themselves, not just their agreement: on cc and Metal both sides of the comparison
     above run the SAME software codec, so a tie rule changed in both would keep the comparison
     green. The golden is what pins the rule, and it is backend-uniform — every entry here was
     checked against [__nv_fp8_e5m2] and [__hip_fp8_e5m2] on real hardware. Hex float notation
     keeps it platform-independent (CLAUDE.md). *)
  Stdio.printf "narrowed (input -> e5m2):\n";
  Array.iteri decisive ~f:(fun i v -> Stdio.printf "  %h -> %h\n" v device.(i));
  let same a b = Int64.equal (Int64.bits_of_float a) (Int64.bits_of_float b) in
  let disagreements =
    Array.filter_mapi decisive ~f:(fun i v ->
        if same host.(i) device.(i) then None
        else Some (Printf.sprintf "%h: host %h, device %h" v host.(i) device.(i)))
  in
  Array.iter disagreements ~f:(fun d -> Stdio.eprintf "fp8 narrowing disagreement: %s\n" d);
  p "the device narrows every decisive value exactly as the host does"
    (Array.is_empty disagreements);

  (* Reachability, so that "no disagreement" cannot hold vacuously: the flushing codec this
     replaced could never emit the smallest subnormal from a narrowing, whichever side ran it. *)
  p "narrowing reaches the smallest e5m2 subnormal"
    (Array.existsi decisive ~f:(fun i v ->
         Float.(v > 0.) && Float.(device.(i) = two_pow (-16))));
  p "a finite input above the range saturates rather than going infinite"
    (Array.for_alli decisive ~f:(fun i v ->
         (not (Float.is_finite v)) || Float.is_finite device.(i)));
  p "the sign of a zero survives narrowing"
    (Array.existsi decisive ~f:(fun i v ->
         Float.(v = 0.) && Float.ieee_negative v && Float.ieee_negative device.(i)));

  (* Infinities and NaNs: only the part all four backends agree on. *)
  let sdev = narrow_on_device [| Float.infinity; Float.neg_infinity; Float.nan |] in
  p "an infinite input narrows to the largest finite magnitude or beyond, keeping its sign"
    (Float.(sdev.(0) >= 57344.) && Float.(sdev.(1) <= -57344.));
  p "a NaN input narrows to a NaN" (Float.is_nan sdev.(2));

  (* Magnitudes far below the smallest subnormal must vanish. HIP miscompiles exactly this range
     (gh-ocannl-639: an out-of-range shift returns values as large as 2^-14), so the leg is
     announced as skipped there rather than pinning the defect as expected behavior. *)
  let claim = "magnitudes far below the smallest subnormal narrow to zero" in
  if on_hip then skipped claim
  else
    let tiny = [| 4.1359e-25; 8.27e-25; 1.65e-24; 3.31e-24; -4.1359e-25 |] in
    let tdev = narrow_on_device tiny in
    p claim (Array.for_all tdev ~f:(fun x -> Float.(x = 0.)))
