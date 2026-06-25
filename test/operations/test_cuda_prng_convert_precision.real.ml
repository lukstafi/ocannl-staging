(* Regression test for task-7d2ed931: CUDA threefry/PRNG distribution.

   The on-device PRNG was statistically broken because the CUDA-local [convert_precision] dropped the
   bit-spreading on the per-element threefry counter: a [uint32]/[uint64] index offset coerced to
   [Uint4x32] fell through to a no-spread struct literal [{(unsigned int)(offset), 0, 0, 0}], so
   consecutive offsets produced correlated 2-round-light-threefry outputs. The fix routes
   [X -> Uint4x32] through the [<prec>_to_uint4x32(] bit-spreading helpers and [Uint4x32 -> scalar]
   through [uint4x32_to_<prec>_uniform(] instead of raw [.v[0]].

   This asserts the conversion codegen directly (no CUDA device required, only the [cudajit]-gated
   [arrayjit.cuda_backend] library), covering both the default [uint32] and the [large_models]
   [uint64] index-precision widths. *)

open Base
module Ops = Ir.Ops
module Utils = Utils

let show (pre, post) = Printf.sprintf "%s_%s" pre post

let assert_eq ~msg expected actual =
  if not (String.equal expected actual) then (
    Stdio.printf "FAIL %s: expected %S got %S\n" msg expected actual;
    assert false)

(* The no-spread fallback that the bug emitted; must never appear for any conversion now. *)
let no_spread_prefix = "{(unsigned int)("

let check_not_no_spread ~msg (pre, _post) =
  if String.is_substring pre ~substring:no_spread_prefix then (
    Stdio.printf "FAIL %s: still emits the no-spread struct literal %S\n" msg pre;
    assert false)

let () =
  (* AC3 (uint32 width): the implicit threefry counter is the loop offset of precision
     [Ops.index_prec ()], which is [uint32] under default [large_models=false]. It is coerced to the
     [Uint4x32] binop precision, so the conversion that must bit-spread is [index_prec -> uint4x32].
     The invariant that would break under the bug: this emits the no-spread struct literal instead of
     a [_to_uint4x32] helper call. *)
  let old_setting = Utils.settings.large_models in
  Utils.settings.large_models <- false;
  let idx_prec = Ops.index_prec () in
  assert (String.equal (Ops.prec_string idx_prec) "uint32");
  let conv = Cuda_backend.convert_precision ~from:idx_prec ~to_:Ops.uint4x32 in
  Stdio.printf "uint32 -> uint4x32: %s\n" (show conv);
  assert_eq ~msg:"index_prec(uint32) -> uint4x32" "uint32_to_uint4x32(_)" (show conv);
  check_not_no_spread ~msg:"index_prec(uint32) -> uint4x32" conv;

  (* AC3 (uint64 width): under [large_models=true], [Ops.index_prec () = uint64]; the same offset
     path must spread via [uint64_to_uint4x32(]. This is the harness condition that instantiates the
     wide-index case — without flipping the setting, this assertion would never exercise it. *)
  Utils.settings.large_models <- true;
  let idx_prec64 = Ops.index_prec () in
  assert (String.equal (Ops.prec_string idx_prec64) "uint64");
  let conv64 = Cuda_backend.convert_precision ~from:idx_prec64 ~to_:Ops.uint4x32 in
  Stdio.printf "uint64 -> uint4x32: %s\n" (show conv64);
  assert_eq ~msg:"index_prec(uint64) -> uint4x32" "uint64_to_uint4x32(_)" (show conv64);
  check_not_no_spread ~msg:"index_prec(uint64) -> uint4x32" conv64;
  Utils.settings.large_models <- old_setting;

  (* Explicit width coverage, independent of [index_prec ()]. *)
  let u32 = Cuda_backend.convert_precision ~from:Ops.uint32 ~to_:Ops.uint4x32 in
  assert_eq ~msg:"uint32 -> uint4x32" "uint32_to_uint4x32(_)" (show u32);
  let u64 = Cuda_backend.convert_precision ~from:Ops.uint64 ~to_:Ops.uint4x32 in
  assert_eq ~msg:"uint64 -> uint4x32" "uint64_to_uint4x32(_)" (show u64);

  (* Secondary fix: [Uint4x32 -> scalar] must call a [_uniform] conversion, not raw-extract [.v[0]].
     The invariant that would break under the bug: [show] would be ["_.v[0]"]. NB: the buildable
     CUDA/Metal helpers are the [_uniform] variants, so this diverges from the bare
     [uint4x32_to_<prec>(] text of [Ops.c_convert_precision] (accepted assumption gap). *)
  let to_single = Cuda_backend.convert_precision ~from:Ops.uint4x32 ~to_:Ops.single in
  Stdio.printf "uint4x32 -> single: %s\n" (show to_single);
  assert_eq ~msg:"uint4x32 -> single" "uint4x32_to_single_uniform(_)" (show to_single);
  let to_double = Cuda_backend.convert_precision ~from:Ops.uint4x32 ~to_:Ops.double in
  assert_eq ~msg:"uint4x32 -> double" "uint4x32_to_double_uniform(_)" (show to_double);

  (* Identity arms preserved (no-op conversion). *)
  let id32 = Cuda_backend.convert_precision ~from:Ops.uint32 ~to_:Ops.uint32 in
  assert_eq ~msg:"uint32 -> uint32 identity" "_" (show id32);

  Stdio.printf "CUDA PRNG convert_precision tests passed!\n"
