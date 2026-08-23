(* What the benches' whole-output checksum has to discriminate (gh-ocannl-711).

   [bin/schedule_bench.ml] and [bin/narrow_gebp_bench.ml] compare hand-written schedules of the same
   matmul against each other, and the only thing standing between "this schedule is fast" and "this
   schedule is fast and correct" is a checksum of the whole output plus one interior spot cell. A
   checksum that cannot see a row permutation is worse than none: the bench then reports a fast
   WRONG schedule as a win, and the autotuner learns from those reports.

   The natural way to write such a weight — a residue of the FLATTENED offset [t = i*n + j] — is
   degenerate exactly where these benches run. [1 + (t mod 251)] gives every row the identical
   weight vector whenever 251 divides n, so a swapped pair of rows leaves the checksum unchanged;
   and at the same extent the spot cell at [1][1] sees nothing outside its own row, so both halves
   of the guard fail together. The same collapse hits operand data drawn as [(t mod p)] over a
   row-major array: at [p | row_stride] every row is identical, so a schedule substituting the wrong
   row computes the right answer and no whole-output check can tell.

   [Bench_checksum] keys both on the (row, column) pair through an aperiodic mix instead. Every
   claim below is paired with the flat-offset form as a NEGATIVE CONTROL: a check the new form
   passes is evidence only once the old form is shown to fail the same check. The controls are the
   reason [Bench_checksum.flat_offset_weighted] still exists. *)

open Base
module Bc = Bench_checksum

let p = Stdio.printf

(* A synthetic m x n output whose value varies with BOTH indices, which is what a matmul result
   does. A value depending on the row alone (or an affine [i*n + j]) would make every row difference
   constant along the row, and a row swap would then be visible only through the SUM of each row's
   weights — a much weaker thing to ask of a weight, and one that collides by accident at a handful
   of extents for reasons that have nothing to do with the degeneracy under test. *)
let output ~m ~n =
  Array.init (m * n) ~f:(fun t -> Float.of_int (1 + (Bc.mix ~salt:0x00A5 (t / n) (t % n) % 97)))

let swap_rows ~n values ~r1 ~r2 =
  let v = Array.copy values in
  for j = 0 to n - 1 do
    let keep = v.((r1 * n) + j) in
    v.((r1 * n) + j) <- v.((r2 * n) + j);
    v.((r2 * n) + j) <- keep
  done;
  v

let plain_sum = Array.fold ~init:0.0 ~f:( +. )

(* Rows of an operand minted the way [schedule_bench] mints ma and mb: a residue of the flat offset
   over a row-major [rows x row_stride] array, either through the mix or the flat form. *)
let mixed_rows ~salt ~modulus ~row_stride ~rows =
  List.init rows ~f:(fun r ->
      Array.init row_stride ~f:(fun c -> Bc.residue ~salt ~row_stride ~modulus ((r * row_stride) + c)))

let flat_rows ~modulus ~row_stride ~rows =
  List.init rows ~f:(fun r ->
      Array.init row_stride ~f:(fun c -> ((r * row_stride) + c) % modulus))

let pairwise_distinct rows =
  let n = List.length rows in
  n
  = List.length
      (List.dedup_and_sort rows ~compare:(fun a b -> Array.compare Int.compare a b))

let all_identical = function
  | [] | [ _ ] -> true
  | first :: rest -> List.for_all rest ~f:(fun r -> Array.equal Int.equal first r)

(* The extents swept for the row-permutation claims: everything from the smallest two-row output up
   past the third multiple of 251, so the degenerate extents 251, 502 and 753 are reached by a sweep
   rather than named into it. *)
let extents = List.range 2 801

let row_pairs = [ (0, 1); (1, 3); (0, 3); (2, 3) ]
let rows = 4

let () =
  (* Which extents each form of the guard FAILED to notice a row swap at. *)
  let missed_mixed = ref [] and missed_flat = ref [] and missed_plain = ref [] in
  List.iter extents ~f:(fun n ->
      let v = output ~m:rows ~n in
      let mixed = Bc.whole_output ~row_stride:n v in
      let flat = Bc.flat_offset_weighted v in
      let plain = plain_sum v in
      List.iter row_pairs ~f:(fun (r1, r2) ->
          let w = swap_rows ~n v ~r1 ~r2 in
          if Float.equal (Bc.whole_output ~row_stride:n w) mixed then
            missed_mixed := n :: !missed_mixed;
          if Float.equal (Bc.flat_offset_weighted w) flat then missed_flat := n :: !missed_flat;
          if Float.equal (plain_sum w) plain then missed_plain := n :: !missed_plain));
  let uniq ns = List.dedup_and_sort ~compare:Int.compare ns in
  let flat_missed = uniq !missed_flat in
  p "extents swept: %d..%d, row pairs per extent: %d\n" (List.hd_exn extents)
    (List.last_exn extents) (List.length row_pairs);
  p "extents where the flat-offset weight missed a row swap: %s\n"
    (String.concat ~sep:", " (List.map flat_missed ~f:Int.to_string));
  Verdict.p "the shared checksum sees every row swap at every extent swept"
    (List.is_empty !missed_mixed);
  (* The multiples of 251 in range are where the flat form is degenerate BY CONSTRUCTION (rather
     than by an accidental weight collision), and it must miss the swap at every row pair there. *)
  let degenerate = List.filter extents ~f:(fun n -> n % Bc.weight_cap = 0) in
  Verdict.p "the flat-offset weight misses a row swap at every multiple of 251 (negative control)"
    (List.for_all degenerate ~f:(fun n ->
         List.length (List.filter !missed_flat ~f:(fun m -> m = n)) = List.length row_pairs));
  Verdict.p "the shared checksum sees a row swap at those same multiples of 251"
    (List.for_all degenerate ~f:(fun n -> not (List.mem !missed_mixed n ~equal:Int.equal)));
  (* Why the guard is WEIGHTED at all: a permutation preserves the multiset, and a plain sum reads
     nothing else. *)
  Verdict.p "an unweighted sum misses every row swap at every extent (negative control)"
    (List.length !missed_plain = List.length extents * List.length row_pairs);

  (* Operands. [schedule_bench] draws ma as a residue mod 13 over a row-major m x k array and mb as
     a residue mod 17 over a k x n one, so the collapse lands at 13 | k and 17 | n respectively.
     Sweep the multiples of each modulus. *)
  let operand_multiples = List.range 1 25 in
  List.iter
    [ ("ma", 0x5A17, 13); ("mb", 0x3C6E, 17) ]
    ~f:(fun (name, salt, modulus) ->
      let strides = List.map operand_multiples ~f:(fun mult -> modulus * mult) in
      Verdict.pf "mixed %s rows stay pairwise distinct at every row stride %d divides" name modulus
        (List.for_all strides ~f:(fun row_stride ->
             pairwise_distinct (mixed_rows ~salt ~modulus ~row_stride ~rows:16)));
      Verdict.pf "flat-offset %s rows are all identical at every row stride %d divides (negative \
                  control)"
        name modulus
        (List.for_all strides ~f:(fun row_stride ->
             all_identical (flat_rows ~modulus ~row_stride ~rows:16))));

  (* The arithmetic the weights' exactness argument rests on: a residue is a residue (non-negative,
     below its modulus), so a weight is in [1, 251] and products of the benches' exact-in-binary
     operands stay exact in the double accumulator. *)
  let residues_in_range =
    List.for_all [ 3; 5; 13; 17; Bc.weight_cap ] ~f:(fun modulus ->
        List.for_all (List.range 1 40) ~f:(fun row_stride ->
            List.for_all (List.range 0 (7 * row_stride)) ~f:(fun t ->
                let r = Bc.residue ~salt:0x7E51 ~row_stride ~modulus t in
                r >= 0 && r < modulus)))
  in
  Verdict.p "every residue lands in [0, modulus)" residues_in_range;
  let refuses f = match f () with exception Invalid_argument _ -> true | _ -> false in
  Verdict.p "residue refuses a non-positive row stride"
    (refuses (fun () -> Bc.residue ~salt:0 ~row_stride:0 ~modulus:13 1));
  Verdict.p "residue refuses a non-positive modulus"
    (refuses (fun () -> Bc.residue ~salt:0 ~row_stride:8 ~modulus:0 1))
