(* gh-ocannl-817: settle whether packed uniform can reach [Set_vec_unop] lowering with a symbolic
   extent, then pin the lowering boundary independently of shape inference.

   The real graph leg asks shape inference to put a gh-490 symbolic dimension on a packed [uniform].
   Its round-up counter relation is a [Total_elems] constraint, and symbolic dimensions are
   deliberately outside that arithmetic: the graph is rejected before projections can be derived.
   The hand-built leg starts from the projections of a valid concrete packed uniform, adds the
   symbolic-extent metadata that the rejected graph would have carried, and proves the lowerer also
   refuses it explicitly rather than silently emitting an unguarded vector store. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module Asgns = Ir.Assignments
module Idx = Ir.Indexing
open Verdict.Claims

let rejection f =
  match f () with
  | () -> None
  | exception Row.Shape_error (msg, _) -> Some msg
  | exception Utils.User_error msg -> Some msg

let says rejection substring =
  Option.value_map rejection ~default:false ~f:(String.is_substring ~substring)

let rec find_set_vec = function
  | Asgns.Set_vec_unop { op; lhs; rhs; projections; _ } -> Some (op, lhs, rhs, projections)
  | Seq (a, b) -> Option.first_some (find_set_vec a) (find_set_vec b)
  | Block_comment (_, body) -> find_set_vec body
  | Noop | Accum_op _ | Fetch _ -> None

let rec find_accum_projections = function
  | Asgns.Accum_op { projections; _ } -> Some projections
  | Seq (a, b) -> Option.first_some (find_accum_projections a) (find_accum_projections b)
  | Block_comment (_, body) -> find_accum_projections body
  | Noop | Set_vec_unop _ | Fetch _ -> None

let () =
  Tensor.unsafe_reinitialize ();
  let extent, bindings = Idx.get_static_symbol ~static_range:6 Idx.Empty in
  let u = TDSL.uniform () () in
  let%op symbolic_u = u ++ "s=>s" [ "s" ] in
  Shape.set_sym_dim s extent;
  let graph_rejection =
    rejection (fun () ->
        let comp = Train.forward symbolic_u in
        ignore
          (Asgns.to_low_level ~static_indices:(Idx.bound_symbols bindings) comp.Asgns.asgns
            : Ir.Low_level.t))
  in
  p "packed uniform with a symbolic extent is rejected before Set_vec_unop lowering"
    (Option.is_some graph_rejection);
  p "the reachability refusal identifies unsupported total-elements arithmetic"
    (says graph_rejection "Total_elems" && says graph_rejection "symbolic dimensions");

  Tensor.unsafe_reinitialize ();
  let concrete = TDSL.uniform () ~output_dims:[ 6 ] () in
  let comp = Train.forward concrete in
  let op, lhs, rhs, projections = Option.value_exn (find_set_vec comp.Asgns.asgns) in
  let projections = Lazy.force projections in
  let iter =
    match Array.to_list projections.components with
    | [ [ (_, iter) ] ] -> iter
    | _ -> failwith "expected concrete packed uniform to have one product iterator"
  in
  let guarded_projections = { projections with extent_syms = [ (Some iter, extent) ] } in
  let hand_built =
    Asgns.Set_vec_unop
      {
        op;
        lhs;
        rhs;
        projections = lazy guarded_projections;
        projections_debug = "gh-817 injected symbolic extent";
      }
  in
  let lowering_rejection =
    rejection (fun () ->
        ignore
          (Asgns.to_low_level ~static_indices:(Idx.bound_symbols bindings) hand_built
            : Ir.Low_level.t))
  in
  p "Set_vec_unop lowering explicitly rejects bound symbolic extents"
    (Option.is_some lowering_rejection);
  p "the lowering refusal names Set_vec_unop and the symbolic-extent hazard"
    (says lowering_rejection "Set_vec_unop" && says lowering_rejection "symbolic extent");
  let maximum_shape = Asgns.to_low_level hand_built in
  p "the same unbound extent retains maximum-shape vector stores"
    (Ll_test.count_stmt maximum_shape ~f:(function
       | Ir.Low_level.Set_from_vec _ -> true
       | _ -> false)
    > 0);

  Tensor.unsafe_reinitialize ();
  let singleton_extent, singleton_bindings = Idx.get_static_symbol ~static_range:1 Idx.Empty in
  let%op singleton_source = { singleton_source = 0.5 } in
  let%op singleton_symbolic = (2. *. singleton_source) ++ "z=>z" [ "z" ] in
  Shape.set_sym_dim z singleton_extent;
  let singleton_comp = Train.forward singleton_symbolic in
  let singleton_projections =
    Option.value_exn (find_accum_projections singleton_comp.Asgns.asgns) |> Lazy.force
  in
  p "a maximum-one symbolic axis retains extent metadata without an iterator"
    (List.exists singleton_projections.extent_syms ~f:(fun (iter, sym) ->
         Option.is_none iter && Idx.equal_static_symbol sym singleton_extent));
  p "zero is a valid runtime binding for a maximum-one extent"
    (Result.is_ok (Result.try_with (fun () -> Idx.validate_bound_value singleton_extent 0)));
  let singleton_set_vec =
    Asgns.Set_vec_unop
      {
        op;
        lhs;
        rhs;
        projections = lazy { projections with extent_syms = singleton_projections.extent_syms };
        projections_debug = "gh-817 maximum-one symbolic extent";
      }
  in
  let singleton_rejection =
    rejection (fun () ->
        ignore
          (Asgns.to_low_level
             ~static_indices:(Idx.bound_symbols singleton_bindings)
             singleton_set_vec
            : Ir.Low_level.t))
  in
  p "Set_vec_unop rejects a bound maximum-one symbolic extent" (Option.is_some singleton_rejection)
