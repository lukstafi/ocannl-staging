open Base
open Stdio
open Ocannl
open Operation.DSL_modules
module IDX = Train.IDX

(* Test 1: RAW dependency — sgd_routine reads gradients written by grad_routine. Pattern from
   zero2hero_1of7_exec.ml simple_gradients_hosted. *)
let test_raw_dependency () =
  printf "=== Test 1: RAW dependency ===\n";
  Tensor.unsafe_reinitialize ();
  let ctx = Context.auto () in
  let%op e = { a = [ 2 ] } *. { b = [ -3 ] } in
  let%op d = e + { c = [ 10 ] } in
  let%op l = d *. { f = [ -2 ] } in
  let grad = Train.grad_update l in
  let%op learning_rate = 0.1 in
  Train.every_non_literal_materialized l;
  Train.every_non_literal_materialized learning_rate;
  let sgd = Train.sgd_update ~learning_rate l in
  let ctx = Train.init_params ctx IDX.empty l in
  let grad_routine = Train.to_routine ctx IDX.empty grad in
  let sgd_routine = Train.to_routine (Context.context grad_routine) IDX.empty sgd in
  let grad_id = Context.routine_id grad_routine in
  let sgd_deps = Context.execution_deps sgd_routine in
  printf "sgd depends on grad: %b\n" (List.mem sgd_deps grad_id ~equal:Int.equal);
  printf "sgd has deps: %b\n" (not (List.is_empty sgd_deps));
  (* Correct order: grad then sgd *)
  let ctx' = Context.run ctx grad_routine in
  let _ctx' = Context.run ctx' sgd_routine in
  printf "Correct order (grad then sgd): OK\n"

(* Test 2: Disjoint routines — sibling compiles from same context produce no deps *)
let test_disjoint () =
  printf "\n=== Test 2: Disjoint routines ===\n";
  Tensor.unsafe_reinitialize ();
  let ctx = Context.auto () in
  let%op loss_x = { x_param = [ 5 ] } *. { x_in = [ 3 ] } in
  let%op loss_y = { y_param = [ 7 ] } *. { y_in = [ 2 ] } in
  Train.every_non_literal_materialized loss_x;
  Train.every_non_literal_materialized loss_y;
  let grad_x = Train.grad_update loss_x in
  let grad_y = Train.grad_update loss_y in
  let ctx = Train.init_params ctx IDX.empty loss_x in
  let ctx = Train.init_params ctx IDX.empty loss_y in
  (* Compile from same context — sibling branches, should be independent *)
  let routine_x = Train.to_routine ctx IDX.empty grad_x in
  let routine_y = Train.to_routine ctx IDX.empty grad_y in
  let x_id = Context.routine_id routine_x in
  let y_id = Context.routine_id routine_y in
  let x_deps = Context.execution_deps routine_x in
  let y_deps = Context.execution_deps routine_y in
  (* Neither should depend on the other — they may have deps on init_params routines *)
  printf "x depends on y: %b\n" (List.mem x_deps y_id ~equal:Int.equal);
  printf "y depends on x: %b\n" (List.mem y_deps x_id ~equal:Int.equal);
  (* Both should be runnable since init_params already executed *)
  printf "can_run x: %b\n" (Context.can_run ctx routine_x);
  printf "can_run y: %b\n" (Context.can_run ctx routine_y);
  (* Either order should work *)
  let ctx' = Context.run ctx routine_y in
  let _ctx' = Context.run ctx' routine_x in
  printf "Reverse order (y then x): OK\n"

(* Test 3: can_run reflects execution state *)
let test_can_run () =
  printf "\n=== Test 3: can_run ===\n";
  Tensor.unsafe_reinitialize ();
  let ctx = Context.auto () in
  let%op e = { a = [ 2 ] } *. { b = [ -3 ] } in
  let%op d = e + { c = [ 10 ] } in
  let%op l = d *. { f = [ -2 ] } in
  let grad = Train.grad_update ~setup_for_parallel:true l in
  let%op learning_rate = 0.1 in
  let sgd = Train.sgd_update ~learning_rate l in
  let ctx = Train.init_params ctx IDX.empty l in
  let grad_routine = Train.to_routine ctx IDX.empty grad in
  let sgd_routine = Train.to_routine (Context.context grad_routine) IDX.empty sgd in
  printf "can_run grad (before execution): %b\n" (Context.can_run ctx grad_routine);
  printf "can_run sgd (before grad): %b\n" (Context.can_run ctx sgd_routine);
  let ctx' = Context.run ctx grad_routine in
  printf "can_run sgd (after grad): %b\n" (Context.can_run ctx' sgd_routine)

(* Test 4: Negative test — running a dependent routine out of order raises Failure *)
let test_wrong_order_raises () =
  printf "\n=== Test 4: Wrong order raises ===\n";
  Tensor.unsafe_reinitialize ();
  let ctx = Context.auto () in
  let%op e = { a = [ 2 ] } *. { b = [ -3 ] } in
  let%op d = e + { c = [ 10 ] } in
  let%op l = d *. { f = [ -2 ] } in
  let grad = Train.grad_update l in
  let%op learning_rate = 0.1 in
  Train.every_non_literal_materialized l;
  Train.every_non_literal_materialized learning_rate;
  let sgd = Train.sgd_update ~learning_rate l in
  let ctx = Train.init_params ctx IDX.empty l in
  let grad_routine = Train.to_routine ctx IDX.empty grad in
  let sgd_routine = Train.to_routine (Context.context grad_routine) IDX.empty sgd in
  (* sgd depends on grad — running sgd first must fail *)
  try
    ignore (Context.run ctx sgd_routine);
    printf "Wrong order (sgd before grad): no error (BUG)\n"
  with Failure msg ->
    let is_enforcement = String.is_substring msg ~substring:"Context.run:" in
    printf "Wrong order raises Failure from Context.run: %b\n" is_enforcement

(* Test 5: Re-execution pattern — grad -> sgd -> grad succeeds without reset *)
let test_reexecution () =
  printf "\n=== Test 5: Re-execution (grad -> sgd -> grad) ===\n";
  Tensor.unsafe_reinitialize ();
  let ctx = Context.auto () in
  let%op e = { a = [ 2 ] } *. { b = [ -3 ] } in
  let%op d = e + { c = [ 10 ] } in
  let%op l = d *. { f = [ -2 ] } in
  let grad = Train.grad_update ~setup_for_parallel:true l in
  let%op learning_rate = 0.1 in
  let sgd = Train.sgd_update ~learning_rate l in
  let ctx = Train.init_params ctx IDX.empty l in
  let grad_routine = Train.to_routine ctx IDX.empty grad in
  let sgd_routine = Train.to_routine (Context.context grad_routine) IDX.empty sgd in
  let ctx' = Context.run ctx grad_routine in
  let ctx' = Context.run ctx' sgd_routine in
  let _ctx' = Context.run ctx' grad_routine in
  printf "grad -> sgd -> grad: OK\n"

(* Test 6: rollback withdraws an optimistic execution marking (gh-ocannl-536). [Context.run] marks
   a routine executed before a later [sync] can report an asynchronous failure, so a contained
   launch/sync rejection has to undo it — otherwise the next candidate the autotuner compiles in the
   same lineage sees a dependency satisfied by a routine that never completed. *)
let test_rollback_execution () =
  printf "\n=== Test 6: rollback of an execution marking ===\n";
  Tensor.unsafe_reinitialize ();
  let ctx = Context.auto () in
  let%op e = { a = [ 2 ] } *. { b = [ -3 ] } in
  let%op d = e + { c = [ 10 ] } in
  let%op l = d *. { f = [ -2 ] } in
  let grad = Train.grad_update ~setup_for_parallel:true l in
  let%op learning_rate = 0.1 in
  let sgd = Train.sgd_update ~learning_rate l in
  let ctx = Train.init_params ctx IDX.empty l in
  let grad_routine = Train.to_routine ctx IDX.empty grad in
  let sgd_routine = Train.to_routine (Context.context grad_routine) IDX.empty sgd in
  let ctx' = Context.run ctx grad_routine in
  printf "can_run sgd (after grad): %b\n" (Context.can_run ctx' sgd_routine);
  Context.rollback_execution ctx' (Context.routine_id grad_routine);
  printf "can_run sgd (after rollback): %b\n" (Context.can_run ctx' sgd_routine);
  (* The ledger is shared by reference across the lineage, so the rollback is visible from the
     context the routine was compiled from as well. *)
  let ctx' = Context.run ctx' grad_routine in
  printf "re-running grad restores it: %b\n" (Context.can_run ctx' sgd_routine)

(* Test 7: poisoning condemns the lineage (gh-ocannl-536). A launch or sync failure that may have
   left device buffers partially written has no restore path, so continuing to time candidates on
   that lineage would score suspect data. Every entrypoint refuses, naming the routine. *)
let test_poisoned_lineage () =
  printf "\n=== Test 7: poisoned lineage ===\n";
  Tensor.unsafe_reinitialize ();
  let ctx = Context.auto () in
  let%op e = { a = [ 2 ] } *. { b = [ -3 ] } in
  let%op l = e + { c = [ 10 ] } in
  Train.every_non_literal_materialized l;
  let grad = Train.grad_update l in
  let ctx = Train.init_params ctx IDX.empty l in
  let routine = Train.to_routine ctx IDX.empty grad in
  let ctx = Context.run ctx routine in
  Context.poison_lineage ctx
    ~routine_name:(Context.routine_name routine)
    (Failure "synthetic device failure");
  let refuses what f =
    match f () with
    | _ -> printf "%s: not refused (BUG)\n" what
    | exception Failure msg ->
        printf "%s refused, names the routine: %b, names the cause: %b\n" what
          (String.is_substring msg ~substring:(Context.routine_name routine))
          (String.is_substring msg ~substring:"synthetic device failure")
  in
  refuses "run" (fun () -> ignore (Context.run ctx routine));
  refuses "sync" (fun () -> Context.sync ctx);
  refuses "get_values" (fun () -> ignore (Context.get_values ctx l.Tensor.value));
  (* Same-backend [copy] dispatches through the backend's transfer machinery rather than [to_host],
     so guarding the host round-trip alone would let a poisoned source export a suspect buffer into
     a clean lineage. *)
  let clean = Context.auto () in
  refuses "copy out" (fun () -> ignore (Context.copy ~src:ctx ~dst:clean l.Tensor.value));
  refuses "copy in" (fun () -> ignore (Context.copy ~src:clean ~dst:ctx l.Tensor.value))

let () =
  test_raw_dependency ();
  test_disjoint ();
  test_can_run ();
  test_wrong_order_raises ();
  test_reexecution ();
  test_rollback_execution ();
  test_poisoned_lineage ()
