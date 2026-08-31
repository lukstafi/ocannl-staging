(** Focused checks on {!Training_golden.recent_mean_exn}, the window statistic every [train]-tier
    convergence claim rests on (gh-ocannl-854).

    The training integrations that use it log one loss per epoch newest-first and claim a bound on
    the mean of the last few, so two properties of this helper are load-bearing and neither is
    visible in those tests' goldens: the window is taken from the NEWEST end -- an
    off-by-orientation would average the epochs a run starts from, where a diverging model still
    passes -- and an undersized window is refused rather than averaged short, so a training loop
    shortened by an edit or an early exit cannot quietly weaken the claim built on it.

    The equalities below are over values exact in binary -- small integer sums whose divisor divides
    them exactly -- so they are decidable rather than tolerance-bounded. The one window mean that is
    not exact is claimed through a two-sided bound instead. *)

open Base
open Verdict.Claims

let mean = Training_golden.recent_mean_exn

(* [true] exactly when the call is refused with [Invalid_argument]. Any other exception answers
   [false] rather than escaping, so a change of failure mode is reported as the claim that broke
   rather than as an unlabelled crash. *)
let rejected ~count values =
  match mean ~count values with
  | (_ : float) -> false
  | exception Invalid_argument _ -> true
  | exception _ -> false

let () =
  (* Newest-first selection. The tail entries are large enough that every other reading of the list
     is a different number: oldest-three is 67.666…, all-five is 41.2, newest-three is 2. *)
  p "the window averages the newest entries: (1+2+3)/3"
    (Float.equal (mean ~count:3 [ 1.; 2.; 3.; 100.; 100. ]) 2.);
  (* The same data in the other orientation. A helper that took its window from the older end would
     answer 2. here too, so this is the claim that separates the two readings; the bound is
     two-sided because (100+100+3)/3 is not exact in binary and a one-sided bound would admit an
     answer drawn from the wrong end of a longer log. *)
  p "reversing the log gives the mean of the other three entries, near 67.67"
    (let reversed = mean ~count:3 [ 100.; 100.; 3.; 2.; 1. ] in
     Float.(reversed > 67.6 && reversed < 67.7));
  (* Entries outside the window do not reach the result at all -- not merely down-weighted by a
     larger divisor, which a whole-list mean over a count-sized divisor would also produce. *)
  p_all "entries older than the window do not move the mean"
    [ [ 1.; 3.; 5. ]; [ 1.; 3.; -1_000_000. ]; [ 1.; 3.; 5.; 7.; 9. ] ]
    ~min:3
    ~f:(fun values -> Float.equal (mean ~count:2 values) 2.);
  p "a window of one is the newest entry" (Float.equal (mean ~count:1 [ 7.; 0. ]) 7.);
  (* A window that exactly fits is the whole-list mean: the boundary the length guard admits. *)
  p "a window as long as the log is the whole-log mean: (1+2+3+4)/4"
    (Float.equal (mean ~count:4 [ 1.; 2.; 3.; 4. ]) 2.5);
  (* Undersized windows. The one-short case is the regression this guard exists for: a training loop
     that logged fewer epochs than the claim asks about. *)
  p "a window one entry longer than the log is refused" (rejected ~count:5 [ 1.; 2.; 3.; 4. ]);
  p "a window over an empty log is refused" (rejected ~count:1 []);
  p_all "a window longer than the log is refused, by however much"
    [ (2, [ 1. ]); (10, [ 1.; 2.; 3. ]); (1, []) ]
    ~min:3
    ~f:(fun (count, values) -> rejected ~count values);
  (* Non-positive counts, which would otherwise divide by zero or by a negative. *)
  p_all "a non-positive window is refused" [ 0; -1; -10 ] ~min:3 ~f:(fun count ->
      rejected ~count [ 1.; 2.; 3. ])
