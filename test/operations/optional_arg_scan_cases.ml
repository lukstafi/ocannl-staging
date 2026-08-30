(** Synthetic controls for the [lib/] optional-argument honesty scan (gh-ocannl-811).

    The live inventory is normally clean, so silence cannot say whether it found no violation or
    forgot how to recognize one. These snippets put the violating discard forms and their nearest
    honest counterparts through the same parsed-source classifier the repository scan uses. *)

open Base
module Scan = Test_utils.Optional_arg_scan

let one ?argument source =
  let args = Scan.args_in_source ~source:"fixture.ml" source in
  match argument with
  | Some label -> List.find_exn args ~f:(fun arg -> String.equal arg.Scan.label label)
  | None -> (
      match args with
      | [ arg ] -> arg
      | _ -> failwith (Printf.sprintf "expected one optional argument, got %d" (List.length args)))

let case ?argument label source ~implemented ~honest =
  let arg = one ?argument source in
  Verdict.p
    (label ^ ": implementation classified")
    (match (arg.Scan.implementation, implemented) with
    | Scan.Implemented, true | Scan.Unimplemented, false -> true
    | _ -> false);
  Verdict.p (label ^ ": honesty classified") (Bool.equal (Scan.honest arg) honest)

let () =
  case "ordinary optional argument affects its result"
    {ocaml|let f ?(scale = 2) x = x * scale|ocaml} ~implemented:true ~honest:true;
  case "forwarding an optional argument is a use" {ocaml|let f ?scale x = helper ?scale x|ocaml}
    ~implemented:true ~honest:true;
  case "an op spec coefficient is a ppx-generated use"
    {ocaml|let%op f ?(stride = 1) x = x ++ "stride*o+k => o"|ocaml} ~implemented:true ~honest:true;
  case "legacy convolution padding is a ppx-generated use"
    {ocaml|let%op f ?(use_padding = true) x = x ++ "stride*o+k => o"|ocaml} ~implemented:true
    ~honest:true;
  case "a shadowed op spec coefficient does not use the outer option"
    {ocaml|let%op f ?(stride = 1) x = let stride = 2 in x ++ "stride*o+k => o"|ocaml}
    ~implemented:false ~honest:false;
  case "an unrelated string in an op body does not use the option"
    {ocaml|let%op f ?(stride = 1) x = ignore "stride*o+k => o"; x|ocaml} ~implemented:false
    ~honest:false;
  case ~argument:"scale" "a later optional default uses an earlier option"
    {ocaml|let f ?(scale = 2) ?(fallback = scale) () = fallback|ocaml} ~implemented:true
    ~honest:true;
  case ~argument:"scale" "a nested optional default sees the outer option before shadowing"
    {ocaml|let f ?(scale = 2) () = fun ?(scale = scale) () -> scale|ocaml} ~implemented:true
    ~honest:true;
  case "an expression-level op extension enables generated uses"
    {ocaml|let f ?(stride = 1) x = [%op x ++ "stride*o+k => o"]|ocaml} ~implemented:true
    ~honest:true;
  case "ordinary label discarded through let-wildcard is rejected"
    {ocaml|let f ?(feature = true) () = let _ = feature in ()|ocaml} ~implemented:false
    ~honest:false;
  case "ordinary label discarded through ignore is rejected"
    {ocaml|let f ?(feature = true) () = ignore feature|ocaml} ~implemented:false ~honest:false;
  case "ordinary label discarded through the pipe operator is rejected"
    {ocaml|let f ?(feature = true) () = feature |> ignore|ocaml} ~implemented:false ~honest:false;
  case "ordinary label discarded through the apply operator is rejected"
    {ocaml|let f ?(feature = true) () = ignore @@ feature|ocaml} ~implemented:false ~honest:false;
  case "ordinary label discarded through named throwaway is rejected"
    {ocaml|let f ?(feature = true) () = let _unused = feature in ()|ocaml} ~implemented:false
    ~honest:false;
  case "an underscore-prefixed local forwarded later is a real use"
    {ocaml|let f ?(feature = 1) () = let _value = feature in helper _value|ocaml} ~implemented:true
    ~honest:true;
  case "underscore label makes an unimplemented option caller-visible"
    {ocaml|let f ?(_feature = true) () = ()|ocaml} ~implemented:false ~honest:true;
  case "implemented underscore label is stale" {ocaml|let f ?(_feature = 2) x = x * _feature|ocaml}
    ~implemented:true ~honest:false;
  case "a nested shadow does not pretend the outer option is used"
    {ocaml|let f ?(feature = 2) () = let g feature = feature + 1 in g 3|ocaml} ~implemented:false
    ~honest:false;
  case "comments and strings do not pretend the option is used"
    {ocaml|let f ?(feature = 2) () = (* feature *) ignore "feature"|ocaml} ~implemented:false
    ~honest:false;
  case "an optional argument on a directly returned closure is inventoried"
    {ocaml|let make () = fun ?(feature = true) () -> ignore feature|ocaml} ~implemented:false
    ~honest:false
