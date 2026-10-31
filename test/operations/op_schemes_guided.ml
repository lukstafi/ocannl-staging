(* gh-ocannl-1761: guided registration, the opt-in [op_shape_schemes=guided] mode, pinned below. A
   schemed [%op] application instantiates its scheme once against the arguments, then runs its body;
   every update step the body builds that matches the next step of the trace registers without
   deriving or solving its own inequalities, its new shapes unified with the scheme's renamed rows.
   The oracle is the same construction with the schemes off: closed shapes, ids, random draws,
   values and gradients must be identical. Each claim about the mechanism comes with a control that
   comes out the other way: schemes off, checking mode, or a sabotaged guide. *)

open Base
open Ocannl.Nn_blocks.DSL_modules
open Verdict.Claims
module Row = Ocannl.Row
module Nn = Ocannl.Nn_blocks

let set_mode m = Tensor.For_testing.set_op_shape_schemes (Some m)
let () = set_mode `Guided

let with_mode m f =
  set_mode m;
  Exn.protect ~f ~finally:(fun () -> set_mode `Guided)

let outcome f =
  match f () with
  | (_ : 'a) -> Ok ()
  | exception Row.Shape_error (msg, _) -> Error msg
  | exception Tensor.Session_error (msg, _) -> Error msg

let error_message f = match outcome f with Ok () -> None | Error msg -> Some msg

let has_substring ~substring = function
  | Some msg -> String.is_substring msg ~substring
  | None -> false

let leaf ?(batch = []) name output =
  Tensor.term ~grad_spec:Prohibit_grad ~label:[ name ] ~batch_dims:batch ~output_dims:output ()

let events f =
  let log = ref [] in
  let result = Tensor.For_testing.observe_schemes f ~on_event:(fun e -> log := e :: !log) in
  (result, List.rev !log)

let guided name = function Tensor.Scheme.Guided n -> String.equal n name | _ -> false

let about name = function
  | Tensor.Scheme.Traced n | Checked n | Guided n | Unschemed (n, _) -> String.equal n name

(* The registration statistics [f] adds. *)
let counted f =
  let c0 = Tensor.Scheme.registration_counts () in
  let result = f () in
  let c1 = Tensor.Scheme.registration_counts () in
  (result, Shape.diff_registration_counts c1 c0)

let report name (c : Shape.registration_counts) =
  Stdio.eprintf "(not part of the golden) %s: %s\n%!" name
    (Sexp.to_string_hum (Shape.sexp_of_registration_counts c))

let closed_dims ts = List.map ts ~f:(fun t -> Array.to_list (Shape.to_dims t.Tensor.shape))

(* {2 The functions under test} *)

let runs = ref 0

let%op layer () x =
  Int.incr runs;
  ({ w; o = [ 4 ] } * x) + 1.0

let%op layer_plain () x =
  Int.incr runs;
  ({ w; o = [ 4 ] } * x) + 1.0
[@@ocannl.no_scheme]

(* A dimension captured inside the body, then set: its step registers normally (a fallback), and the
   steps after it are still guided. *)
let%op captures () x =
  let y = x ++ "...|...->ab => ...|...->a" [ "b" ] in
  [%oc Shape.set_dim b 3];
  (y *. 2.0) + 1.0

let%op captures_plain () x =
  let y = x ++ "...|...->ab => ...|...->a" [ "b" ] in
  [%oc Shape.set_dim b 3];
  (y *. 2.0) + 1.0
[@@ocannl.no_scheme]

(* Bodies whose graph follows a flag the trace cannot see. *)
let extra = ref false
let drop = ref false
let rewire = ref false
let concrete = ref false
let close_inside = ref false
let checkpoint_inside = ref false
let maybe_extra y = if !extra then ignore (TDSL.O.(y + y) : Tensor.t)

let maybe_drop y =
  if !drop then y
  else
    let%op r = y *. 2.0 in
    r

let wire x y z = if !rewire then TDSL.O.(z + x) else TDSL.O.(z + y)

let maybe_concrete x =
  let t =
    if !concrete then Tensor.term ~grad_spec:Prohibit_grad ~label:[ "k" ] ~output_dims:[ 3 ] ()
    else Tensor.term ~grad_spec:Prohibit_grad ~label:[ "k" ] ()
  in
  TDSL.O.(x + t)

let maybe_close y = if !close_inside then ignore (Shape.to_dims y.Tensor.shape : int array)

let maybe_checkpoint () =
  if !checkpoint_inside then ignore (Tensor.checkpoint ~label:"inside the body" () : _)

let%op flagged () x =
  let y = x + 1.0 in
  [%oc maybe_extra y];
  let z = [%oc maybe_drop y] in
  let r = [%oc wire x y z] in
  let r = [%oc maybe_concrete r] in
  [%oc maybe_close r];
  [%oc maybe_checkpoint ()];
  r *. 0.5

let%op flagged_plain () x =
  let y = x + 1.0 in
  [%oc maybe_extra y];
  let z = [%oc maybe_drop y] in
  let r = [%oc wire x y z] in
  let r = [%oc maybe_concrete r] in
  [%oc maybe_close r];
  [%oc maybe_checkpoint ()];
  r *. 0.5
[@@ocannl.no_scheme]

let reset_flags () =
  List.iter [ extra; drop; rewire; concrete; close_inside; checkpoint_inside ] ~f:(fun r ->
      r := false)

(* Gradient helpers: [sin]'s gradient code builds a [cos x] when [x] needs a gradient -- at the
   application, never at the trace -- ahead of the body's own [cos x]. *)
let%op sin_cos () x =
  let y = sin x in
  cos x + y

let%op sin_cos_plain () x =
  let y = sin x in
  cos x + y
[@@ocannl.no_scheme]

(* Nesting: a closure built inside the body, and a function of sub-layers. *)
let%op const_layer () x = { cw = 0.5; o = [ 4 ] } * x

let%op outer () x =
  let inner = const_layer () in
  inner x

let%op stack () =
  let l1 = layer () in
  let l2 = layer () in
  fun x -> l2 (l1 x)

(* {2 The mechanism} *)

let () =
  Verdict.case "applications register guided, skipping the steps' own inequalities" (fun () ->
      Tensor.unsafe_reinitialize ();
      runs := 0;
      let lay = layer () in
      let x1 = leaf ~batch:[ 2 ] "x1" [ 3 ] and x2 = leaf ~batch:[ 5 ] "x2" [ 3 ] in
      let ((_ : Tensor.t), evs), c1 = counted (fun () -> events (fun () -> lay x1)) in
      let ((_ : Tensor.t), evs2), c2 = counted (fun () -> events (fun () -> lay x2)) in
      report "layer, first application" c1;
      report "layer, second application" c2;
      p "each application registers guided"
        (List.count (evs @ evs2) ~f:(guided "layer") = 2
        && c1.guided_applications = 1 && c2.guided_applications = 1);
      p "every step of the body is guided, and the same number each time"
        (c1.guided_steps > 0 && c1.guided_steps = c2.guided_steps);
      p "no step ran its own inequalities"
        (c1.inline_registrations = 0 && c2.inline_registrations = 0);
      p "no fallback, no unmatched step" (c1.fallback_steps + c1.unmatched_steps = 0);
      p "the body still ran once at the trace and once per application" (!runs = 3);
      let (_ : Tensor.t), off =
        with_mode `Off (fun () ->
            Tensor.unsafe_reinitialize ();
            let lay = layer () in
            let x = leaf ~batch:[ 2 ] "x" [ 3 ] in
            counted (fun () -> lay x))
      in
      p "control (schemes off): the body's steps run their own inequalities"
        (off.guided_steps = 0 && off.inline_registrations = c1.guided_steps);
      let shapes_of lay =
        let a = leaf ~batch:[ 2 ] "a" [ 3 ] in
        let b = leaf ~batch:[ 5 ] "b" [ 3 ] in
        closed_dims [ lay a; lay b ]
      in
      Tensor.unsafe_reinitialize ();
      let schemed = shapes_of (layer ()) in
      Tensor.unsafe_reinitialize ();
      let plain = shapes_of (layer_plain ()) in
      p "the applications close as the opted-out twin's do"
        (List.equal (List.equal Int.equal) schemed plain));
  Verdict.case "a capture registers normally, and the steps after it are guided" (fun () ->
      Tensor.unsafe_reinitialize ();
      let f = captures () in
      let x = leaf ~batch:[ 2 ] "x" [ 4; 3 ] in
      let r, c = counted (fun () -> f x) in
      report "captures" c;
      p "the capturing step is a fallback" (c.fallback_steps = 1);
      p "the steps after it are guided" (c.guided_steps > 0 && c.unmatched_steps = 0);
      let schemed = closed_dims [ r ] in
      Tensor.unsafe_reinitialize ();
      let plain = closed_dims [ captures_plain () (leaf ~batch:[ 2 ] "x" [ 4; 3 ]) ] in
      p "and the application closes as the opted-out twin's does"
        (List.equal (List.equal Int.equal) schemed plain));
  Verdict.case "a step the trace lacked registers normally" (fun () ->
      Tensor.unsafe_reinitialize ();
      reset_flags ();
      let f = flagged () in
      extra := true;
      let r, c = counted (fun () -> f (leaf ~batch:[ 2 ] "x" [ 3 ])) in
      reset_flags ();
      report "flagged, with an extra step" c;
      p "the extra step is unmatched, the trace's steps all guided"
        (c.unmatched_steps >= 1 && c.guided_steps > 0);
      let schemed = closed_dims [ r ] in
      Tensor.unsafe_reinitialize ();
      extra := true;
      let plain = closed_dims [ flagged_plain () (leaf ~batch:[ 2 ] "x" [ 3 ]) ] in
      reset_flags ();
      p "and the application closes as the opted-out twin's does"
        (List.equal (List.equal Int.equal) schemed plain));
  Verdict.case "a body that builds a different graph raises" (fun () ->
      let diverge ~mode flag =
        with_mode mode (fun () ->
            Tensor.unsafe_reinitialize ();
            reset_flags ();
            let f = flagged () in
            flag := true;
            let msg = error_message (fun () -> f (leaf ~batch:[ 2 ] "x" [ 3 ])) in
            reset_flags ();
            msg)
      in
      List.iter
        [ ("a dropped step", drop); ("a rewired operand", rewire); ("a created shape", concrete) ]
        ~f:(fun (what, flag) ->
          let msg = diverge ~mode:`Guided flag in
          Stdio.eprintf "(not part of the golden) %s: %s\n%!" what
            (Option.value msg ~default:"no error");
          p
            (what ^ ": raises, naming the function and the divergence")
            (has_substring ~substring:"applying the shape scheme flagged" msg
            && has_substring ~substring:"built a different graph than at its trace" msg);
          p (what ^ ": and the session is tainted") (Option.is_some (Shape.taint_provenance ()));
          p
            (what ^ ": control (checking mode): no divergence is reported")
            (not (has_substring ~substring:"different graph" (diverge ~mode:`Check flag))));
      let msg =
        Tensor.For_testing.with_guided_sabotage No_creation_check (fun () ->
            diverge ~mode:`Guided concrete)
      in
      p "control (no creation check): the changed created shape goes unnoticed"
        (not (has_substring ~substring:"different graph" msg)));
  Verdict.case "a closure or a checkpoint inside a guided body is refused" (fun () ->
      List.iter
        [
          ("a closure", close_inside, "finish_inference");
          ("a checkpoint", checkpoint_inside, "checkpoint");
        ]
        ~f:(fun (what, flag, word) ->
          let run mode =
            with_mode mode (fun () ->
                Tensor.unsafe_reinitialize ();
                reset_flags ();
                let f = flagged () in
                flag := true;
                let msg = error_message (fun () -> f (leaf ~batch:[ 2 ] "x" [ 3 ])) in
                reset_flags ();
                msg)
          in
          let msg = run `Guided in
          p
            (what ^ " after the last matched step is refused before it commits")
            (has_substring ~substring:word msg
            && has_substring ~substring:"inside a guided application of the shape scheme flagged"
                 msg);
          p (what ^ ": control (checking mode): allowed") (Option.is_none (run `Check))));
  Verdict.case "an argument-dependent error raises before the body, untainted" (fun () ->
      let conflict mode =
        with_mode mode (fun () ->
            Tensor.unsafe_reinitialize ();
            runs := 0;
            let lay = layer () in
            ignore (lay (leaf ~batch:[ 2 ] "x" [ 3 ]) : Tensor.t);
            let before = !runs in
            let msg = error_message (fun () -> lay (leaf ~batch:[ 2 ] "y" [ 5 ])) in
            let ran = !runs - before in
            let usable = Result.is_ok (outcome (fun () -> lay (leaf ~batch:[ 7 ] "z" [ 3 ]))) in
            (msg, ran, usable))
      in
      let msg, ran, usable = conflict `Guided in
      p "the conflict raises, naming the function and the application"
        (has_substring ~substring:"applying the shape scheme layer to y" msg);
      p "before the body ran" (ran = 0);
      p "and the session stays usable" usable;
      let msg', ran', usable' = conflict `Check in
      p "as in checking mode"
        (has_substring ~substring:"applying the shape scheme layer to y" msg' && ran' = 0 && usable'));
  Verdict.case "gradient helpers do not take the forward steps' places" (fun () ->
      let x () = Tensor.term_init ~grad_spec:Require_grad [| 0.5; -1.0; 2.0 |] ~label:[ "x" ] () in
      Tensor.unsafe_reinitialize ();
      let f = sin_cos () in
      let r, c = counted (fun () -> f (x ())) in
      report "sin_cos" c;
      p "with a differentiable argument, every forward step is guided and nothing diverges"
        (c.guided_steps > 0 && c.unmatched_steps = 0);
      p "the gradient helpers register normally" (c.backward_steps > 0);
      let schemed = closed_dims [ r ] in
      Tensor.unsafe_reinitialize ();
      let plain = closed_dims [ sin_cos_plain () (x ()) ] in
      p "and the application closes as the opted-out twin's does"
        (List.equal (List.equal Int.equal) schemed plain));
  Verdict.case "nested applications run under the outer guide" (fun () ->
      Tensor.unsafe_reinitialize ();
      let f = stack () in
      let ((_ : Tensor.t), evs), c =
        counted (fun () -> events (fun () -> f (leaf ~batch:[ 2 ] "x" [ 3 ])))
      in
      report "stack" c;
      p "the outer application is guided" (List.exists evs ~f:(guided "stack"));
      p_none "the inner applications run plain" evs ~f:(about "layer");
      p "the inner steps are the outer guide's" (c.guided_steps > 0 && c.unmatched_steps = 0);
      Tensor.unsafe_reinitialize ();
      let f = outer () in
      let ((_ : Tensor.t), evs), c =
        counted (fun () -> events (fun () -> f (leaf ~batch:[ 2 ] "x" [ 3 ])))
      in
      report "outer" c;
      p "a closure built inside a guided body is left for its first application outside"
        (List.exists evs ~f:(guided "outer")
        && List.for_all evs ~f:(fun e -> not (about "const_layer" e)));
      p "and the outer application matches fully" (c.unmatched_steps = 0 && c.fallback_steps = 0));
  Verdict.case "a rewind after a guided application leaves nothing" (fun () ->
      Tensor.unsafe_reinitialize ();
      let lay = layer () in
      ignore (lay (leaf ~batch:[ 2 ] "x" [ 3 ]) : Tensor.t);
      let ck = Tensor.checkpoint ~label:"before the application" () in
      ignore (lay (leaf ~batch:[ 5 ] "y" [ 3 ]) : Tensor.t);
      Tensor.rewind ck;
      p_empty "the rewound session is the checkpoint's" ~over:Tensor.For_testing.restore_categories
        (Tensor.For_testing.checkpoint_differences ck);
      p "and no guide is left installed" (not (Shape.guide_active ())))

(* {2 Inline parity over the library's blocks} *)

let () =
  Verdict.case "nn_blocks models close, initialize and train as inline" (fun () ->
      let result = ref None and after = ref None in
      let run mode =
        with_mode mode (fun () ->
            Tensor.unsafe_reinitialize ();
            let first = Tensor.term ~grad_spec:Prohibit_grad ~label:[ "first" ] () in
            let mlp = Nn.mlp ~label:[ "m" ] ~hid_dims:[ 8; 6 ] () in
            let ln = Nn.layer_norm ~label:[ "ln" ] () in
            (* Distinct values per row, so a misrouted row shows in the outputs. *)
            let values =
              [|
                0.5;
                -1.25;
                0.75;
                1.5;
                -0.25;
                -1.0;
                0.25;
                1.25;
                -0.5;
                -1.5;
                1.0;
                0.0;
                0.375;
                -0.625;
                0.875;
              |]
            in
            let input batch =
              Tensor.term_init ~grad_spec:Prohibit_grad
                (Array.sub values ~pos:0 ~len:(batch * 3))
                ~label:[ "x" ] ~batch_dims:[ batch ] ~output_dims:[ 3 ] ()
            in
            let x2 = input 2 in
            let tgt =
              Tensor.term_init ~grad_spec:Prohibit_grad [| 0.5; -0.25; 1.0 |] ~label:[ "tgt" ]
                ~output_dims:[ 3 ] ()
            in
            let (), c =
              counted (fun () ->
                  let%op loss = (ln (mlp x2) *. tgt) ++ "... => 0" in
                  Set.iter loss.Tensor.params ~f:(fun t ->
                      Option.iter t.Tensor.diff ~f:(fun d ->
                          Ocannl.Train.set_materialized d.Tensor.grad));
                  let ctx = Ocannl.Train.update_once (Context.auto ()) loss in
                  let rec all acc t =
                    if Map.mem acc t.Tensor.value.Ir.Tnode.id then acc
                    else
                      List.fold t.Tensor.children
                        ~init:(Map.set acc ~key:t.Tensor.value.Ir.Tnode.id ~data:t) ~f:(fun acc s ->
                          all acc s.Tensor.subtensor)
                  in
                  let nodes = all (Map.empty (module Int)) loss in
                  let params = Set.to_list loss.Tensor.params in
                  let values t = Array.to_list (Context.get_values ctx t.Tensor.value) in
                  let grads t =
                    Option.value_map t.Tensor.diff ~default:[] ~f:(fun d ->
                        Array.to_list (Context.get_values ctx d.Tensor.grad))
                  in
                  result :=
                    Some
                      ( Map.to_alist
                          (Map.map nodes ~f:(fun t -> Array.to_list (Shape.to_dims t.Tensor.shape))),
                        values loss :: List.concat_map params ~f:(fun t -> [ values t; grads t ]),
                        ( Tensor.get_next_id (),
                          loss.Tensor.value.Ir.Tnode.uid - first.Tensor.value.Ir.Tnode.uid ) );
                  (* After the closure, another batch size. *)
                  let y = mlp (input 5) in
                  let ctx = Ocannl.Train.forward_once ctx y in
                  after :=
                    Some
                      ( Array.to_list (Shape.to_dims y.Tensor.shape),
                        Array.to_list (Context.get_values ctx y.Tensor.value) ))
            in
            report
              (match mode with
              | `Guided -> "nn_blocks, guided"
              | `Check -> "nn_blocks, check"
              | `Off -> "nn_blocks, off")
              c;
            (Option.value_exn !result, Option.value_exn !after, c))
      in
      let take mode =
        result := None;
        after := None;
        run mode
      in
      let (dims_o, vals_o, ids_o), after_o, _ = take `Off in
      let (dims_g, vals_g, ids_g), after_g, c_g = take `Guided in
      p "the guided run registered guided steps"
        (c_g.guided_steps > 0 && c_g.guided_applications > 0);
      p "every tensor of the graph closes to the same dims, intermediates and parameters included"
        (List.equal (fun (i, d) (i', d') -> i = i' && List.equal Int.equal d d') dims_g dims_o);
      p "the session hands out the same tensor ids and node uids" (Poly.equal ids_g ids_o);
      p "the loss, the parameters (random initialization included) and their gradients are the same"
        (List.equal (List.equal Float.equal) vals_g vals_o);
      p "after the closure, an application at another batch size closes and computes the same"
        (Poly.equal after_g after_o);
      let sabotaged =
        try
          let (dims_s, _, _), _, _ =
            Tensor.For_testing.with_guided_sabotage No_unify (fun () -> take `Guided)
          in
          List.equal (fun (i, d) (i', d') -> i = i' && List.equal Int.equal d d') dims_s dims_o
        with _ -> false
      in
      p "control (no unification): the graph does not close as inline" (not sabotaged))
