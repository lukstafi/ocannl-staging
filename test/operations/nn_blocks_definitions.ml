(* gh-ocannl-1751: the definition-check convention, applied to the library. A [%op] block definition
   is checked early by [Tensor.Scheme.check_definition], which runs a trial application at a default
   configuration in a window that is always rewound: the [()] traces run there, and with them any
   shape error that holds for every argument. [lib/nn_blocks.ml] does not do this at load time
   (module initialization is no place for a trace); this test does it for every schemed block. *)

open Base
open Ocannl.Nn_blocks.DSL_modules
open Verdict.Claims
module Nn = Ocannl.Nn_blocks
module Row = Ocannl.Row

let ignore_ f () = ignore (f () : _)

(* Each schemed block, by the name its scheme is traced under, with a trial configuration. *)
let blocks =
  [
    ("mlp_layer", ignore_ (fun () -> Nn.mlp_layer ~label:[ "t" ] ~hid_dim:8 ()));
    ("mlp", ignore_ (fun () -> Nn.mlp ~label:[ "t" ] ~hid_dims:[ 8; 8 ] ()));
    ("softmax", ignore_ (fun () -> Nn.softmax ~spec:"... | v" ()));
    ("cross_entropy_loss", ignore_ (fun () -> Nn.cross_entropy_loss ~spec:"... | v" ()));
    ("layer_norm", ignore_ (fun () -> Nn.layer_norm ~label:[ "t" ] ()));
    ("conv2d", ignore_ (fun () -> Nn.conv2d ~label:[ "t" ] ~out_channels:4 ()));
    ( "depthwise_separable_conv2d",
      ignore_ (fun () -> Nn.depthwise_separable_conv2d ~label:[ "t" ] ~out_channels:4 ()) );
    ("max_pool2d", ignore_ (fun () -> Nn.max_pool2d ()));
    ("max_pool2d_copy", ignore_ (fun () -> Nn.max_pool2d_copy ()));
    ("avg_pool2d", ignore_ (fun () -> Nn.avg_pool2d ()));
  ]

(* A block whose body contradicts itself whatever its argument. *)
let%op inconsistent () x = x + ({ c3 = 0.; o = [ 3 ] } + { c4 = 0.; o = [ 4 ] })

(* The blocks the [%op] expansion of [lib/nn_blocks.ml] schemes at a [()] stage, derived from the
   expansion itself: the list above is held to it, so a new schemed block needs its trial. *)
let schemed_at_unit =
  let text = Stdio.In_channel.read_all "../../lib/nn_blocks.ml" in
  Ppx_ocannl.schemed_definitions (Ppxlib.Parse.implementation (Lexing.from_string text))
  |> List.filter_map ~f:(fun (name, at_unit) -> if at_unit then Some name else None)
  |> List.dedup_and_sort ~compare:String.compare

let () =
  Verdict.case "the trials cover exactly the blocks the expansion schemes at ()" (fun () ->
      let trials = List.dedup_and_sort ~compare:String.compare (List.map blocks ~f:fst) in
      Stdio.eprintf "(not part of the golden) schemed at (): %s\n%!"
        (String.concat ~sep:" " schemed_at_unit);
      p_exists "the expansion schemes some block at its () stage" schemed_at_unit ~f:(fun _ -> true);
      p "the trial list is the expansion's list" (List.equal String.equal trials schemed_at_unit));
  Verdict.case "every schemed nn_blocks block passes its definition check" (fun () ->
      Tensor.unsafe_reinitialize ();
      let ck = Tensor.checkpoint ~label:"before the checks" () in
      let next_id = Tensor.get_next_id () in
      List.iter blocks ~f:(fun (name, trial) ->
          match Tensor.Scheme.check_definition ~name trial with
          | traced ->
              p (name ^ ": its own scheme was traced") (List.mem traced name ~equal:String.equal)
          | exception (Row.Shape_error (msg, _) | Tensor.Session_error (msg, _)) ->
              Stdio.eprintf "(not part of the golden) %s refused: %s\n%!" name msg;
              p (name ^ ": its own scheme was traced") false);
      p "the checks allocated no session id" (Tensor.get_next_id () = next_id);
      p_empty "the checks left the session as it was" ~over:Tensor.For_testing.restore_categories
        (Tensor.For_testing.checkpoint_differences ck));
  Verdict.case "control: an inconsistent definition is reported, naming it" (fun () ->
      Tensor.unsafe_reinitialize ();
      let ck = Tensor.checkpoint ~label:"before the check" () in
      (match Tensor.Scheme.check_definition ~name:"inconsistent" (ignore_ inconsistent) with
      | _ -> p "the inconsistent definition is refused" false
      | exception Row.Shape_error (msg, _) ->
          Stdio.eprintf "(not part of the golden) refused: %s\n%!" msg;
          p "the inconsistent definition is refused, naming the definition and its scheme"
            (String.is_substring msg
               ~substring:
                 "checking the definition inconsistent: tracing the shape scheme inconsistent"));
      p_empty "the failed check left the session as it was"
        ~over:Tensor.For_testing.restore_categories
        (Tensor.For_testing.checkpoint_differences ck);
      p "the session is usable after the failed check"
        (match Ocannl.Nn_blocks.mlp_layer ~label:[ "after" ] ~hid_dim:2 () with
        | (_ : Tensor.t -> Tensor.t) -> true
        | exception _ -> false))
