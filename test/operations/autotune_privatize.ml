(* The autotuner's privatized preset extension (Autotune.extend_with_privatize), used by the
   privatized fission-candidate flavor: detection of a materialized read-modify-write accumulator
   and executed parity of the privatized schedule.

   [mc = ma * mb] with a materialized output lowers to [Zero_out mc] plus a serial nest accumulating
   [mc[i,j] += ma[i,k] * mb[k,j]]. The extension over the backend's default preset (which may well
   be empty here — the whole-routine annotator bails on the materialized [Zero_out]; the fission
   pipeline separates zeros into their own segments) must append exactly one [Privatize] targeting
   [mc] over the serial reduction loop, and applying the extended schedule must compute the same
   values as the identity-transform twin. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module LL = Ir.Low_level
module Sched = Ir.Schedule
module Asgns = Ir.Assignments

let p name b = Stdio.printf "%s: %b\n" name b

(* Zeros compare equal to zeros. A fragment mapping that reads outside the staged block, a kernel
   that never ran, or a reference whose own setup silently collapsed all yield all-zeros, and a
   parity check between two zero arrays passes while covering nothing (gh-ocannl-481 item 3). Every
   reference array is pinned nonzero where it is produced, so the parity claims below have content.
   *)
let nonzero name (a : float array) =
  if not (Array.exists a ~f:(fun x -> Float.(x <> 0.))) then
    failwith (name ^ ": the reference is all zeros — the parity checks against it are vacuous");
  a
let approx a b = Float.(abs (a -. b) < 1e-3)
let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")

let named name (comp : Asgns.comp) : Asgns.comp =
  { comp with asgns = Asgns.Block_comment (name, comp.asgns) }

let n = 16

let () =
  let mav = Array.init (n * n) ~f:(fun i -> Float.of_int (i % 13) *. 0.25) in
  let mbv = Array.init (n * n) ~f:(fun i -> Float.of_int (i % 17) -. 8.) in
  let ma = TDSL.ndarray mav ~label:[ "ma" ] ~input_dims:[ n ] ~output_dims:[ n ] () in
  let mb = TDSL.ndarray mbv ~label:[ "mb" ] ~input_dims:[ n ] ~output_dims:[ n ] () in

  (* --- Identity-transform twin --- *)
  let%op mc0 = ma * mb in
  let ctx_s = Context.auto () in
  let ctx_s, routine_s =
    Context.compile
      ~lowered_transform:(fun opt -> opt)
      ctx_s
      (named "priv_naive" (Train.forward mc0))
      Ir.Indexing.Empty
  in
  let ctx_s = Context.run ctx_s routine_s in
  let got_naive = nonzero "apz_naive" (Context.get_values ctx_s mc0.Tensor.value) in

  (* --- The backend's default preset, extended with privatization --- *)
  let%op mc1 = ma * mb in
  let n_privatize = ref (-1) in
  let transform (opt : LL.optimized) =
    let preset =
      if Sched.backend_is_gpu backend_name then Sched.default_gpu ~min_parallel:1 opt
      else if Sched.backend_is_cpu backend_name then Sched.default_cpu ~min_parallel:1 opt
      else []
    in
    let extended = Autotune.extend_with_privatize ~static_indices:[] preset opt in
    n_privatize :=
      List.count extended ~f:(function
        | Sched.Privatize { target; _ } -> Ir.Tnode.equal target mc1.Tensor.value
        | _ -> false);
    Sched.apply extended opt
  in
  let ctx_a = Context.auto () in
  let ctx_a, routine_a =
    Context.compile ~lowered_transform:transform ctx_a
      (named "priv_tuned" (Train.forward mc1))
      Ir.Indexing.Empty
  in
  let ctx_a = Context.run ctx_a routine_a in
  let got_priv = Context.get_values ctx_a mc1.Tensor.value in
  p "extension appends exactly one Privatize targeting the accumulator" (!n_privatize = 1);
  p "privatized preset matches the identity twin" (Array.for_all2_exn got_priv got_naive ~f:approx)
