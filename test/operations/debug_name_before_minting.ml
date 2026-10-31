open! Base
open Verdict.Claims
module Tn = Ir.Tnode

(* gh-ocannl-1686: one scheme names a tensor node. Before any renderer mints a code name, its debug
   name is [Tnode.label_name], the identifier the heuristic style mints for a label no other node of
   the routine carries -- so minting such a node leaves its debug name where it was. (It used to
   collapse repeated components, [kernel_downsample2], where the minted name read
   [kernel_downsample_downsample].) The control: a label the routine repeats is minted in the
   [n<id>_<label>] form, which does move the debug name. *)

let next_id = ref 7000

let mk label =
  Int.incr next_id;
  Tn.create (Tn.Specified Ir.Ops.single) ~id:!next_id ~label
    ~unpadded_dims:(lazy [| 1 |])
    ~padding:(lazy None)
    ()

let labels =
  [
    [ "attention" ];
    [ "attention"; "attention" ];
    [ "kernel"; "downsample"; "downsample" ];
    [ "encoder"; "attention"; "attention"; "output" ];
    [ "foo"; "-"; "foo"; "foo" ];
    [ "*."; "w1" ];
    [ "2"; "scale" ];
    [ "2" ];
    [];
    [ "grad"; "w" ];
    [ "grad"; "grad"; "grad" ];
    [ "grad" ];
  ]

let no_repeats () = Hashtbl.create (module String)

let mint style tn =
  Tn.styled_ident ~repeating_nograd_idents:(no_repeats ()) ~repeating_grad_idents:(no_repeats ())
    (`Heuristic_ocannl style) tn

(* The display name, read as the claim under test compares it. *)
let shown tn =
  Tn.debug_name tn
  [@rendered_name "the test compares the display name itself with label_name, its subject"]

let () =
  let nodes = List.map labels ~f:mk in
  List.iter nodes ~f:(fun tn ->
      Stdio.printf "[%s] -> %s\n" (String.concat ~sep:"; " tn.Tn.label) (Tn.label_name tn));
  p_all "before minting, the debug name is label_name" nodes ~f:(fun tn ->
      String.equal (shown tn) (Tn.label_name tn));
  p_all "the label-only form (Tensor's shape name) gives the same name" nodes ~f:(fun tn ->
      String.equal (Tn.get_debug_name ~id:tn.Tn.id ~label:tn.Tn.label ()) (shown tn));
  p_all "the heuristic style mints label_name for a label the routine does not repeat" nodes
    ~f:(fun tn -> String.equal (mint `Dot_grad tn) (Tn.label_name tn));
  List.iter nodes ~f:(fun tn -> Tn.update_code_name tn (mint `Under_grad tn));
  p_all "minting such a node leaves its debug name where it was" nodes ~f:(fun tn ->
      String.equal (shown tn) (Tn.label_name tn));
  (* Control: the same label repeated in the routine mints the [n<id>_] form, and the debug name
     follows the code name, away from label_name. *)
  let tn = mk [ "kernel"; "downsample"; "downsample" ] in
  let repeating = no_repeats () in
  Hashtbl.set repeating ~key:(Tn.label_name tn) ~data:();
  Tn.update_code_name tn
    (Tn.styled_ident ~repeating_nograd_idents:repeating ~repeating_grad_idents:(no_repeats ())
       (`Heuristic_ocannl `Under_grad) tn);
  Stdio.printf "repeated in its routine: %s\n" (shown tn);
  p "control: a label the routine repeats mints a name that moves the debug name"
    (not (String.equal (shown tn) (Tn.label_name tn)))
