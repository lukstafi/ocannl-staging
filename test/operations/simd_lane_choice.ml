(* The lane-count choice behind explicit SIMD rendering (gh-ocannl-621 follow-up): which vector
   width an extent gets on a register file of a given size.

   Pure arithmetic over [Ir.Backend_intf.simd_lane_ladder] / [simd_lanes_for], so unlike every other
   vector-width test here it says the same thing on every machine — which is the point: the property
   that matters is a comparison BETWEEN widths ("a wider register file never renders worse than a
   narrower one"), and no single machine can exhibit both sides of it. *)

open Base

let p = Verdict.p
let p_all = Verdict.p_all
let ladder = Ir.Backend_intf.simd_lane_ladder
let lanes_for = Ir.Backend_intf.simd_lanes_for
let reduce_lanes_for = Ir.Backend_intf.simd_reduce_lanes_for

(* Vector steps plus scalar remainder iterations, the quantity the choice minimizes; an extent no
   width can fill runs serially, which is [extent] trips. *)
let trips ~vector_bytes ~elt_bytes ~extent =
  match lanes_for ~vector_bytes ~elt_bytes ~extent with
  | None -> extent
  | Some lanes -> (extent / lanes) + (extent % lanes)

let () =
  (* The ladder halves from the register width down to a floor of 32 bytes — never below the width a
     pre-AVX-512 machine used, because a narrower rendering would newly vectorize (and thereby
     reassociate) loops that render serially today. *)
  p "f32 ladders: 64B halves to 32B, narrower widths offer one rung"
    (List.equal Int.equal (ladder ~vector_bytes:64 ~elt_bytes:4) [ 16; 8 ]
    && List.equal Int.equal (ladder ~vector_bytes:32 ~elt_bytes:4) [ 8 ]
    && List.equal Int.equal (ladder ~vector_bytes:16 ~elt_bytes:4) [ 4 ]);
  p "f16 and f64 ladders halve the same way"
    (List.equal Int.equal (ladder ~vector_bytes:64 ~elt_bytes:2) [ 32; 16 ]
    && List.equal Int.equal (ladder ~vector_bytes:64 ~elt_bytes:8) [ 8; 4 ]
    && List.equal Int.equal (ladder ~vector_bytes:16 ~elt_bytes:8) [ 2 ]);
  p "a backend that renders no vectors offers no rungs"
    (List.is_empty (ladder ~vector_bytes:0 ~elt_bytes:4));

  (* The three regimes of the choice at 64 bytes, f32. *)
  p "a long loop takes the full width (517 -> 16 lanes)"
    (Option.equal Int.equal (lanes_for ~vector_bytes:64 ~elt_bytes:4 ~extent:517) (Some 16));
  p "an extent the narrower width divides steps down (40 -> 8 lanes)"
    (Option.equal Int.equal (lanes_for ~vector_bytes:64 ~elt_bytes:4 ~extent:40) (Some 8));
  p "an extent below the wide vector still vectorizes narrow (12 -> 8 lanes)"
    (Option.equal Int.equal (lanes_for ~vector_bytes:64 ~elt_bytes:4 ~extent:12) (Some 8));
  p "an extent below every rung declines (7 -> none)"
    (Option.is_none (lanes_for ~vector_bytes:64 ~elt_bytes:4 ~extent:7));
  p "the remainder is not chased at scale (1000 -> 16 lanes, not 8)"
    (Option.equal Int.equal (lanes_for ~vector_bytes:64 ~elt_bytes:4 ~extent:1000) (Some 16));

  (* The claim the whole ladder exists to make, over every extent a kernel might carry. *)
  let extents = List.range 1 2049 in
  let never_worse ~elt_bytes =
    List.for_all extents ~f:(fun extent ->
        trips ~vector_bytes:64 ~elt_bytes ~extent <= trips ~vector_bytes:32 ~elt_bytes ~extent)
  in
  p "a 64-byte register file never renders more trips than a 32-byte one, f32"
    (never_worse ~elt_bytes:4);
  p "likewise at f16 and f64" (never_worse ~elt_bytes:2 && never_worse ~elt_bytes:8);
  (* Strictly better somewhere, or the ladder would be an elaborate no-op. *)
  p "and strictly fewer on some extents"
    (List.exists extents ~f:(fun extent ->
         trips ~vector_bytes:64 ~elt_bytes:4 ~extent < trips ~vector_bytes:32 ~elt_bytes:4 ~extent));

  (* Vectorizing at all is never given up by widening: whatever a 32-byte machine renders as
     vectors, a 64-byte one does too. *)
  p_all "widening never turns a vectorized extent into a serial one" extents ~f:(fun extent ->
      List.for_all [ 2; 4; 8 ] ~f:(fun elt_bytes ->
          Option.is_none (lanes_for ~vector_bytes:32 ~elt_bytes ~extent)
          || Option.is_some (lanes_for ~vector_bytes:64 ~elt_bytes ~extent)));

  (* An accumulating loop ends in a horizontal fold as long as the lane count, so the width that
     minimizes updates is not always the width that minimizes the whole rendering. *)
  p "a short reduction takes the narrower width the elementwise metric would not (64 -> 8 lanes)"
    (Option.equal Int.equal (lanes_for ~vector_bytes:64 ~elt_bytes:4 ~extent:64) (Some 16)
    && Option.equal Int.equal (reduce_lanes_for ~vector_bytes:64 ~elt_bytes:4 ~extent:64) (Some 8));
  p "a long reduction still takes the full width (4096 -> 16 lanes)"
    (Option.equal Int.equal (reduce_lanes_for ~vector_bytes:64 ~elt_bytes:4 ~extent:4096) (Some 16));
  p_all
    "the reduction width offers the same rungs, so it too never declines where 32 bytes would not"
    extents ~f:(fun extent ->
      List.for_all [ 2; 4; 8 ] ~f:(fun elt_bytes ->
          Option.is_none (reduce_lanes_for ~vector_bytes:32 ~elt_bytes ~extent)
          || Option.is_some (reduce_lanes_for ~vector_bytes:64 ~elt_bytes ~extent)))
