(** The anti-degeneracy machinery the matmul-shaped benches share (gh-ocannl-711): an aperiodic mix
    of a (row, column) pair, the operand values minted from it, and the position-weighted
    whole-output checksum that is each bench's correctness guard.

    Two benches ([bin/schedule_bench.ml], [bin/narrow_gebp_bench.ml]) time several schedules of the
    same matmul against each other and compare a checksum of the whole output, because a single spot
    cell cannot see a remainder region and a mishandled edge peel is exactly what these schedules
    risk. A guard that cannot see a row permutation is worse than none: the bench then reports a
    fast WRONG schedule as a win, and the autotuner learns from those reports. Three ways for it to
    go blind, all of which have been live in one copy or the other:

    - {b A residue of the FLATTENED offset.} [t = i * row_stride + j] loses its row dependence
      precisely when the modulus divides the row stride: [1 + (t mod 251)] gives every row the
      identical weight vector [1 + j] at n = 251, 502, 753, …, so a swapped pair of rows leaves the
      checksum unchanged — and the spot cell at [1][1] is blind to other rows at the same extent, so
      both halves of the guard fail together. Operand data collapses the same way: an mb of
      [(t mod 17) - 8] over a k x n array has every row identical whenever 17 divides n, and a
      schedule substituting the wrong row of a collapsed operand computes the right answer.
      {!mix} keys on the (row, column) PAIR, so the row index enters the value in its own right and
      no divisibility relation between a modulus and a stride can erase it. Mixing rather than a
      per-axis residue also removes the shift symmetry — any value drawn from [index mod p] repeats
      under [k -> k + p], so if both operands share that period every packed K panel is identical
      and a staging bug that repeats the wrong panel is invisible; the packing factors are user
      arguments, so no fixed period can be assumed coprime to them.

    - {b Too few weights to tell the rows apart.} A weight capped at [weight_cap] puts each row's
      weight vector in a space of size [weight_cap ^ row_stride], so at a narrow row stride two rows
      collide by birthday long before the pigeonhole bound, and swapping THOSE two is invisible to
      any weighting: at row_stride = 2, one stream gives rows 9 and 363 the same weights. No bounded
      weight escapes this, so the answer is to make the space large enough that no geometry a bench
      is run at reaches it — {!whole_output} accumulates one sum per salt in {!weight_salts},
      squaring the space per extra stream while each sum stays exactly as exact as it was.

    - {b A producer value that IS the accumulator's init.} An operand row of all zeros makes the
      corresponding output row all zeros, which is indistinguishable from a schedule that dropped
      that row entirely against a zero-initialized destination. A row of [row_stride] independent
      residues is all-zero with probability [levels ^ -row_stride] — likely at the narrow extents,
      and it is not the checksum's job to catch what the data hid. {!positive_level} mints the
      multiplicand strictly positive, so no row of it can be all-zero however narrow the row is.

    Both benches had their own copy of this, which is how the fixed version and the degenerate one
    came to sit one file apart. It lives here once instead;
    [test/operations/bench_checksum_discrimination] pins what it must discriminate, with the
    single-stream and flat-offset forms as the negative controls. *)

open Base

(** An aperiodic mix of two indices: no shift of either index is a symmetry, at any lag. Every
    intermediate is masked below 2^24, so nothing overflows a 63-bit int, the result is non-negative
    (so [%] on it is a true residue), and the sequence is reproducible by an external oracle. The
    salt lets one bench mint several independent streams — its operands and its checksum weights —
    from the same function. *)
let mix ~salt a b =
  let x = (a * 73856093) lxor (b * 19349663) lxor salt in
  let x = x lxor (x lsr 13) land 0xFFFFFF in
  let x = x * 1274126177 land 0xFFFFFF in
  x lxor (x lsr 7)

(** [residue ~salt ~row_stride ~modulus t] is [mix] of the (row, column) pair that the flat offset
    [t] denotes in a row-major array of the given row stride, reduced mod [modulus]. This is the
    call site to prefer over a hand-written [t % modulus]: same shape, no divisibility collapse.
    Non-negative, and below [modulus]. *)
let residue ~salt ~row_stride ~modulus t =
  if row_stride <= 0 then
    invalid_arg
      (Printf.sprintf "Bench_checksum.residue: row_stride = %d must be positive" row_stride);
  if modulus <= 0 then
    invalid_arg (Printf.sprintf "Bench_checksum.residue: modulus = %d must be positive" modulus);
  mix ~salt (t / row_stride) (t % row_stride) % modulus

(** [positive_level ~salt ~row_stride ~levels ~scale t] is a producer value drawn from the
    [levels] multiples of [scale] starting at [scale] — STRICTLY POSITIVE, never the zero an
    accumulator is initialized to. Use it for the multiplicand whose row spans the reduction (a
    matmul's left operand): a zero row there zeroes the whole output row, which reads exactly like a
    schedule that dropped the row. The other operand may keep zero in its set — with this one
    positive, no output row is systematically zero. *)
let positive_level ~salt ~row_stride ~levels ~scale t =
  Float.of_int (1 + residue ~salt ~row_stride ~modulus:levels t) *. scale

(** The cap on the checksum weights. Small on purpose: the benches' operands are exact in binary and
    their products are small multiples of a negative power of two, so a weight below 256 keeps the
    whole weighted reduction exact in the double accumulator — which is what lets variants that sum
    in different orders be compared for BITWISE equality. *)
let weight_cap = 251

(** The salts the checksum's weight streams are drawn from, distinct from the salts the benches mint
    operands with so that the weights are independent of the data they weigh.

    TWO of them, because one is not enough at a narrow row stride: a row's weight vector then lives
    in [weight_cap ^ row_stride] values, and two rows landing on the same vector are a pair whose
    swap NO weighting of that stream can see. A second independent stream squares the space — at the
    narrowest legal stride, 2, from 63001 to about 4e9 — which puts the birthday threshold far
    beyond any row count a bench is run at, and costs one more pass outside the timed region.
    Neither sum's exactness argument changes: each is the same bounded-weight reduction it was. *)
let weight_salts = [ 0x7E51; 0x2F3B ]

(** One weight stream of {!whole_output}. Exposed for the discrimination test's negative control —
    a bench should call {!whole_output}, which is the guard. *)
let weighted ~salt ~row_stride values =
  Array.foldi values ~init:0.0 ~f:(fun t acc v ->
      acc +. (v *. Float.of_int (1 + residue ~salt ~row_stride ~modulus:weight_cap t)))

(** [whole_output ~row_stride values] is the position-weighted checksum of a row-major
    [m x row_stride] output, one sum per stream in {!weight_salts}. Weighted rather than plain
    because a plain sum reads only the multiset, so a tail written with the right values at the
    wrong offsets — a permutation, which is what a misplaced edge peel produces — leaves it
    unchanged. Every correct variant of a computation produces the identical list.

    This is a correctness guard, not a measurement: call it OUTSIDE the timed region. *)
let whole_output ~row_stride values =
  List.map weight_salts ~f:(fun salt -> weighted ~salt ~row_stride values)

(** The weights {!whole_output} applies to one row, every stream concatenated: the vector two rows
    must differ in for a swap of them to be visible at all. What the discrimination test sweeps. *)
let row_weights ~row_stride ~row =
  Array.of_list
    (List.concat_map weight_salts ~f:(fun salt ->
         List.init row_stride ~f:(fun j ->
             1 + residue ~salt ~row_stride ~modulus:weight_cap ((row * row_stride) + j))))

(** A checksum rendered for a bench's timing line. *)
let render checksums =
  String.concat ~sep:"/" (List.map checksums ~f:(fun c -> Printf.sprintf "%.10g" c))

(** The degenerate flat-offset weight this module exists to replace, kept so the discrimination test
    can carry it as a NEGATIVE CONTROL: a check that the new form passes is evidence only once the
    old form is shown to fail the same check. Not for use in a bench. *)
let flat_offset_weighted values =
  Array.foldi values ~init:0.0 ~f:(fun t acc v -> acc +. (v *. Float.of_int (1 + (t % weight_cap))))
