(* gh-494 waypoint 2 follow-up: the read-before-write decider flip. [visit_llc]'s tracer
   truncates loops at [virtualize_settings.max_tracing_dim] (default 5), so on a padded-conv
   intermediate the consumer's affine reads (e.g. [oh+kh]) reach positions past the traced write
   range and are classified [Recurrent] — spuriously: every read is in fact covered by prior
   in-routine writes. The affine containment query ([Low_level.reads_covered_query], mirroring
   the tracer's semantics exactly) now cancels the spurious flag, skipping the [On_device]
   pessimization; it never overrides in the other direction.

   Pinned here, on a chain of two padded convs whose intermediate has spatial dims 9x9 (> the
   tracing truncation):
   - the tracer still flags the intermediate recurrent (otherwise this probe went stale);
   - the override fires: [read_before_write] stays false, so the intermediate is no longer
     classified as a routine input ([input_and_output_nodes]) — pre-flip it was, forcing
     [On_device] and excluding it from buffer aliasing;
   - executed parity: the default run's values match a run with the intermediate forced
     materialized (the pre-flip placement) — the decider flip must not change computed values.

   Printed facts are booleans/PASS lines so the expected output stays backend-stable. *)

open Base
open Stdio
open Ocannl
open Ocannl.Operation.DSL_modules
module LL = Ir.Low_level
module Tn = Ir.Tnode

let p name b = printf "%s: %b\n" name b

(* Deterministic input and (via [fixed_state_for_init]) deterministic conv params, so the two
   phases compute the same function. *)
let build () =
  Utils.settings.fixed_state_for_init <- Some 42;
  Tensor.unsafe_reinitialize ();
  let x_src =
    NTDSL.init ~l:"rbwf_xsrc" ~prec:Ir.Ops.single ~o:[ 9; 9; 2 ]
      ~f:(fun idcs -> Float.of_int (((idcs.(0) * 3) + (idcs.(1) * 5) + (idcs.(2) * 7)) % 11) *. 0.125)
      ()
  in
  (* An [init] data node has its layout committed at creation; the padded conv needs a fresh
     operand to grant the halo (see padding_lifecycle.ml). *)
  let x = NTDSL.O.einsum1 "h, w, c => h, w, c" x_src in
  let conv1 =
    Nn_blocks.conv2d ~label:[ "rbwf_c1" ] ~kernel_size:3 ~use_padding:true ~out_channels:3 ()
  in
  let conv2 =
    Nn_blocks.conv2d ~label:[ "rbwf_c2" ] ~kernel_size:3 ~use_padding:true ~out_channels:2 ()
  in
  let h = conv1 x in
  let y = conv2 h in
  (h, y)

let run ~materialize_h =
  let h, y = build () in
  if materialize_h then Train.set_materialized h.Tensor.value;
  Train.set_materialized y.Tensor.value;
  Tn.set_observable y.Tensor.value;
  let ctx = Context.auto () in
  let ctx = Train.init_params ctx Ir.Indexing.Empty y in
  let captured = ref None in
  let transform (opt : LL.optimized) =
    captured := Some opt;
    opt
  in
  let ctx, routine =
    Context.compile ~lowered_transform:transform ctx (Train.forward y) Ir.Indexing.Empty
  in
  let ctx = Context.run ctx routine in
  let yv = Context.get_values ctx y.Tensor.value in
  (yv, Option.value_exn !captured, h)

let () =
  let yv_default, opt, h = run ~materialize_h:false in
  let h_tn = h.Tensor.value in
  (match Hashtbl.find opt.LL.traced_store h_tn with
  | None -> p "default: intermediate traced" false
  | Some traced ->
      p "default: tracer flags the intermediate recurrent"
        (Hashtbl.exists traced.LL.accesses ~f:LL.is_recurrent);
      p "default: read_before_write overridden to false" (not traced.LL.read_before_write));
  let (inputs, _outputs), _merge = LL.input_and_output_nodes opt in
  p "default: intermediate is not a routine input" (not (Set.mem inputs h_tn));
  let yv_mat, opt_mat, h_mat = run ~materialize_h:true in
  (match Hashtbl.find opt_mat.LL.traced_store h_mat.Tensor.value with
  | None -> p "materialized: intermediate traced" false
  | Some traced ->
      p "materialized: tracer flags the intermediate recurrent"
        (Hashtbl.exists traced.LL.accesses ~f:LL.is_recurrent));
  p "materialized: intermediate stays non-virtual"
    (not (Tn.Placements.known_virtual opt_mat.LL.optimize_ctx.LL.placements h_mat.Tensor.value));
  p "parity: same result length" (Array.length yv_default = Array.length yv_mat);
  let max_diff =
    Array.foldi yv_default ~init:0.0 ~f:(fun i acc v -> Float.max acc (Float.abs (v -. yv_mat.(i))))
  in
  if Float.(max_diff <= 1e-5) then printf "parity: PASS\n"
  else printf "parity: FAIL (max diff %.8f)\n" max_diff
