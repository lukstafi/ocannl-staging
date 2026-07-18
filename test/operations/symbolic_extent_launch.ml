open Base
open Ocannl
open Operation.DSL_modules
open Stdio
module IDX = Train.IDX

(* gh-490 stage 2: one compiled routine serves multiple runtime extents. A symbolic-extent axis
   lowers to loops whose bodies are guarded by [iterator < extent], where the extent is a launch
   parameter (the same static symbol that shape inference carries as [Row.Sym]); binding a smaller
   value computes exactly the valid prefix, while the buffer stays sized at the declared maximum.
   The [extent=4] total of 4.0 (not 6.0) is what proves the guard executes: an unguarded (max
   extent) reduction would sum the whole buffer. *)

let () =
  Tensor.unsafe_reinitialize ();
  let seq, bindings = IDX.get_static_symbol ~static_range:6 IDX.empty in
  let%op x = { x = 0.5 } in
  let%op y = (2. *. x) ++ "s=>s" [ "s" ] in
  Shape.set_sym_dim s seq;
  let%op total = y ++ "... => 0" in
  Train.set_materialized y.value;
  Train.set_materialized total.value;
  let ctx = Context.auto () in
  let ctx = Train.init_params ctx bindings total in
  let routine = Train.to_routine ctx bindings (Train.forward total) in
  let ctx = Context.context routine in
  let seq_ref = IDX.find_exn (Context.bindings routine) seq in
  printf "y dims (buffer sized at the declared maximum): %s\n" (Ir.Tnode.dims_to_string y.value);
  let ctx =
    List.fold [ 6; 4; 1; 0 ] ~init:ctx ~f:(fun ctx extent ->
        seq_ref := extent;
        let ctx = Context.run ctx routine in
        let total_v = (Context.get_values ctx total.value).(0) in
        let ys = Context.get_values ctx y.value in
        let prefix = Array.sub ys ~pos:0 ~len:extent in
        let prefix_ok = Array.for_all prefix ~f:(fun v -> Float.(abs (v -. 1.0) < 1e-6)) in
        printf "extent=%d: total=%.1f (expected %.1f), y valid prefix all 1.0: %b\n" extent total_v
          (Float.of_int extent) prefix_ok;
        ctx)
  in
  (* Out-of-range extents are rejected at bind validation. *)
  seq_ref := 7;
  (try
     ignore (Context.run ctx routine : Context.t);
     printf "ERROR: expected a bind-validation error for extent=7\n"
   with Utils.User_error msg -> printf "extent=7 rejected: %s\n" msg)
