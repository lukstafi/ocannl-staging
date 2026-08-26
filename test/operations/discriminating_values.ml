(* The contract {!Ll_test.cycle} and {!Ll_test.drift} document, pinned (gh-ocannl-639, Codex round 1
   on the extraction PR): the recipe's two halves are conditions, not rules of thumb, and a future
   caller reusing the family gets both checked rather than assumed.

   Blindness is mechanical, so [cycle] raises on it: a modulus dividing an axis's row-major stride
   makes the value constant along that axis, and coprimality with the reduction EXTENT does not rule
   that out — [|2; 4; 3|] with modulus 3 has strides [12; 3; 1], so despite 3 and 4 being coprime
   only the innermost index moves the value.

   Exactness is numeric, so it is exhibited here rather than argued: the cells are bf16-exact, some
   partial sum of the reductions accum_width actually builds is NOT (which is what makes per-step
   narrowing visible — a nonzero mean over too few terms would be the zero-mean trap wearing a
   different hat), and every partial sum is f32-exact (which is what lets the f64 host-side
   reference reproduce the widened kernel bitwise). *)

open Base
open Ocannl.Operation.DSL_modules

let p = Verdict.p
let p_all = Verdict.p_all
let bf16_exact x = Float.equal x (Ir.Ops.bfloat16_to_single (Ir.Ops.single_to_bfloat16 x))
let f32_exact x = Float.equal x (Stdlib.Int32.float_of_bits (Stdlib.Int32.bits_of_float x))

(* The shapes accum_width's drift operands take: the two-axis reduction, its vectorized twin, and
   the 16-wide row the Workgroup_reduce legs sum. *)
let shapes = [ [| 4; 6; 6 |]; [| 4; 4; 32 |]; [| 4; 16 |] ]
let row_axis = [| 4; 16 |]

(* Every multi-index of [dims], in row-major order. *)
let all_indices dims =
  Array.fold dims ~init:[ [||] ] ~f:(fun acc extent ->
      List.concat_map acc ~f:(fun idcs -> List.init extent ~f:(fun j -> Array.append idcs [| j |])))

(* The reduction over [dims]'s trailing axes at outer index 0, as running partial sums. *)
let partials ~dims =
  List.folding_map
    (all_indices (Array.subo dims ~pos:1))
    ~init:0.0
    ~f:(fun acc rest ->
      let sum = acc +. Ll_test.drift ~dims (Array.append [| 0 |] rest) in
      (sum, sum))

let () =
  p "cycle rejects a modulus blind to an axis, which coprimality with the extent does not rule out"
    (try
       ignore (Ll_test.cycle ~dims:[| 2; 4; 3 |] ~modulus:3 ~offset:1. ~stride:0.5 [| 0; 0; 0 |]);
       false
     with Invalid_argument _ -> true);
  p_all "cycle accepts the shapes the accumulator-width legs use" shapes ~f:(fun dims ->
      Option.is_none (Ll_test.blind_axis ~dims ~modulus:13));
  p_all "drift varies with every index of every shape used" shapes ~f:(fun dims ->
      let base = Array.map dims ~f:(fun _ -> 0) in
      Array.for_alli dims ~f:(fun ax _ ->
          let bumped = Array.copy base in
          bumped.(ax) <- 1;
          not (Float.equal (Ll_test.drift ~dims base) (Ll_test.drift ~dims bumped))));
  p_all "every drift cell is exact in bf16" shapes ~f:(fun dims ->
      List.for_all (all_indices dims) ~f:(fun idcs -> bf16_exact (Ll_test.drift ~dims idcs)));
  (* The fact the accumulator-width legs rest on: the sums leave bf16 exactness within the extents
     they reduce, so an accumulator narrowing per step diverges from one narrowing at the store. *)
  p_all "a drift reduction leaves bf16 exactness within the extents these tests reduce" shapes
    ~f:(fun dims -> List.exists (partials ~dims) ~f:(fun s -> not (bf16_exact s)));
  p_all "every drift partial sum stays exact in f32" shapes ~f:(fun dims ->
      List.for_all (partials ~dims) ~f:f32_exact);
  (* The crossing drift's doc names: the eleventh term of the 16-wide row reaches 275/64, odd and
     above bf16's guaranteed-exact bound of 2^8 units of 1/64. *)
  p "the 16-wide row's first bf16-inexact partial sum is the eleventh, at 275/64"
    (match List.findi (partials ~dims:row_axis) ~f:(fun _ s -> not (bf16_exact s)) with
    | Some (i, s) -> i = 10 && Float.equal s (275.0 /. 64.0)
    | None -> false)
