(** Host-side statistics used by training-golden claims. *)

open Base

(** [recent_mean_exn ~count values] averages the first [count] values in a newest-first list. It
    rejects an undersized window so a shortened training loop cannot silently weaken a claim. *)
let recent_mean_exn ~count values =
  if count <= 0 then invalid_arg "Training_golden.recent_mean_exn: count must be positive";
  if List.length values < count then
    invalid_arg "Training_golden.recent_mean_exn: fewer values than the requested window";
  List.sum (module Float) (List.take values count) ~f:Fn.id /. Float.of_int count
