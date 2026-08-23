(** The anti-degeneracy machinery the matmul-shaped benches share (gh-ocannl-711): an aperiodic mix
    of a (row, column) pair, the operand residues minted from it, and the position-weighted
    whole-output checksum that is each bench's correctness guard.

    Two benches ([bin/schedule_bench.ml], [bin/narrow_gebp_bench.ml]) time several schedules of the
    same matmul against each other and compare a checksum of the whole output, because a single spot
    cell cannot see a remainder region and a mishandled edge peel is exactly what these schedules
    risk. That guard is only as good as its position dependence, and the natural way to write it —
    a residue of the FLATTENED offset [t = i * row_stride + j] — loses its row dependence precisely
    when the modulus divides the row stride. It is not a hypothetical: [1 + (t mod 251)] gives every
    row the identical weight vector [1 + j] at n = 251, 502, 753, …, so a row permutation (a
    misplaced row-edge peel) leaves the checksum unchanged, and the spot cell at [1][1] is blind to
    corruption in other rows at the same time — both halves of the check fail together, which is the
    situation the checksum exists to prevent. The same collapse hits operand data drawn as
    [(t mod p)]: an mb of [(t mod 17) - 8] over a k x n array has every row identical whenever 17
    divides n, and a transform substituting the wrong row of a collapsed operand then computes the
    correct output, which no whole-output check can see.

    Keying on the (row, column) PAIR through [mix] removes the class rather than dodging it: the
    row index enters the value in its own right, so no divisibility relation between a modulus and
    a stride can erase it. Mixing rather than a per-axis residue also removes the shift symmetry —
    any value drawn from [index mod p] repeats under [k -> k + p], so if both operands share that
    period every packed K panel is identical and a staging bug that substitutes or repeats the wrong
    panel is invisible; the packing factors are user arguments, so no fixed period can be assumed
    coprime to them.

    Both benches called this machinery through their own copies, which is how the fixed version and
    the degenerate one came to sit one file apart. It lives here once instead;
    [test/operations/bench_checksum_discrimination] pins what it must discriminate, with the
    flat-offset forms as the negative controls. *)

open Base

(** An aperiodic mix of two indices: no shift of either index is a symmetry, at any lag. Every
    intermediate is masked below 2^24, so nothing overflows a 63-bit int, the result is
    non-negative (so [%] on it is a true residue), and the sequence is reproducible by an external
    oracle. The salt lets one bench mint several independent streams — its two operands and its
    checksum weight — from the same function. *)
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

(** The cap on the checksum weights. Small on purpose: the benches' operands are exact in binary and
    their products are small multiples of a negative power of two, so a weight below 256 keeps the
    whole weighted reduction exact in the double accumulator — which is what lets variants that sum
    in different orders be compared for BITWISE equality. *)
let weight_cap = 251

(** The default salt for a checksum weight. Distinct from the salts the benches mint operands with,
    so the weight stream is independent of the data it weighs. *)
let default_salt = 0x7E51

(** [whole_output ~row_stride values] is the position-weighted checksum of a row-major [m x
    row_stride] output: [sum_t values.(t) * (1 + residue t)]. Weighted rather than plain because a
    plain sum reads only the multiset, so a tail written with the right values at the wrong offsets
    — a permutation, which is what a misplaced edge peel produces — leaves it unchanged; and
    weighted through the (row, column) pair rather than the flat offset for the reason at the top of
    this file. Every correct variant of a computation prints the identical value.

    This is a correctness guard, not a measurement: call it OUTSIDE the timed region. *)
let whole_output ?(salt = default_salt) ~row_stride values =
  Array.foldi values ~init:0.0 ~f:(fun t acc v ->
      acc +. (v *. Float.of_int (1 + residue ~salt ~row_stride ~modulus:weight_cap t)))

(** The degenerate flat-offset weight this module exists to replace, kept so the discrimination test
    can carry it as a NEGATIVE CONTROL: a check that the new form passes is evidence only once the
    old form is shown to fail the same check. Not for use in a bench. *)
let flat_offset_weighted values =
  Array.foldi values ~init:0.0 ~f:(fun t acc v -> acc +. (v *. Float.of_int (1 + (t % weight_cap))))
