(* gh-ocannl-514 phase 4b: bound pruning against the measured incumbent in [Autotune.tune],
   config [autotune_bound_pruning]. The dune rule pins the cc backend and a tiny envelope
   (model_peak_* = 1e3), making the schedule-invariant roofline floor astronomically larger than
   any measured time — every enumerative sketch candidate is then provably unable to win and is
   pruned before compile, deterministically on any machine, while the baseline and presets are
   exempt and still timed. The tuned routine must still compute correct values (the winner comes
   from the never-pruned candidates). *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module Asgns = Ir.Assignments

let p name b = Stdio.printf "%s: %b\n" name b

let named name (comp : Asgns.comp) : Asgns.comp =
  { comp with asgns = Asgns.Block_comment (name, comp.asgns) }

let () =
  let nn = 64 in
  let av = Array.init (nn * nn) ~f:(fun x -> Float.of_int (x % 13) *. 0.25) in
  let bv = Array.init (nn * nn) ~f:(fun x -> (Float.of_int (x % 7) -. 3.) *. 0.5) in
  let a = TDSL.ndarray av ~label:[ "a" ] ~input_dims:[ nn ] ~output_dims:[ nn ] () in
  let b = TDSL.ndarray bv ~label:[ "b" ] ~input_dims:[ nn ] ~output_dims:[ nn ] () in
  (* Unscheduled twin for the values check. *)
  let%op c0 = a * b in
  let ctx0, r0 =
    Context.compile (Context.auto ()) (named "bp_serial" (Train.forward c0)) Ir.Indexing.Empty
  in
  let ctx0 = Context.run ctx0 r0 in
  let want = Context.get_values ctx0 c0.Tensor.value in
  (* The tuned run, bound pruning on (via the rule's command line). *)
  let%op c1 = a * b in
  let reports = ref [] in
  let ctx, routine =
    Autotune.tune ~rounds:0 ~repeats:1 ~cache_dir:"autotune_bound_pruning_cache"
      ~report:(fun r -> reports := r :: !reports)
      (Context.auto ())
      (named "bp_tuned" (Train.forward c1))
      Ir.Indexing.Empty
  in
  let ctx = Context.run ctx routine in
  let got = Context.get_values ctx c1.Tensor.value in
  p "tuned values match the serial twin"
    (Array.for_all2_exn got want ~f:(fun x y -> Float.(abs (x - y) < 1e-4)));
  match !reports with
  | [ r ] ->
      p "search completed (not partial)" (not r.Autotune.partial);
      p "bound pruning fathomed sketch candidates" (r.Autotune.bound_pruned > 0);
      p "never-pruned candidates were still timed" (r.Autotune.candidates_timed >= 1);
      p "keep-fraction pruning is counted apart" (r.Autotune.model_pruned = 0)
  | _ -> Stdio.printf "expected exactly one report\n"
