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
  case "an op axis label is not a ppx-generated use"
    {ocaml|let%op f ?(o = 1) x = x ++ "i => o"|ocaml} ~implemented:false ~honest:false;
  case "an op dilation coefficient is a ppx-generated use"
    {ocaml|let%op f ?(dilation = 1) x = x ++ "o<+dilation*k => o"|ocaml} ~implemented:true
    ~honest:true;
  case "a concat coefficient is a ppx-generated use"
    {ocaml|let%op f ?(stride = 1) x = (x, x) ++^ "stride*o+k; i => o"|ocaml} ~implemented:true
    ~honest:true;
  case "a qualified custom operator does not generate a use"
    {ocaml|let%op f ?(stride = 1) x = let _ = stride in My.( ++ ) x "stride*o+k => o"|ocaml}
    ~implemented:false ~honest:false;
  case "an einsum-shaped argument outside a unary PPX spec position is not a generated use"
    {ocaml|let%op f ?(stride = 1) x = let _ = stride in let ( ++ ) x _ = x in x ++ spec "stride*o+k => o"|ocaml}
    ~implemented:false ~honest:false;
  case "an axis-only literal without the PPX dispatch guard generates no read"
    {ocaml|let%op f ?(stride = 1) x = let _ = stride in let ( ++ ) x _ = x in x ++ "stride*o+k"|ocaml}
    ~implemented:false ~honest:false;
  case "concat generated reads are op-only"
    {ocaml|let%cd f ?(stride = 1) x = let _ = stride in let ( ++^ ) x _ = x in x ++^ "stride*o+k; i => o"|ocaml}
    ~implemented:false ~honest:false;
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
  case "an oc anti-quotation does not generate einsum coefficient reads"
    {ocaml|let%op f ?(stride = 1) x = let _ = stride in [%oc let ( ++ ) x _ = x in x ++ "stride*o+k => o"]|ocaml}
    ~implemented:false ~honest:false;
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
  case "a locally shadowed ignore is a real use"
    {ocaml|let f ?(_feature = true) () = let ignore x = if x then enable () in ignore _feature|ocaml}
    ~implemented:true ~honest:false;
  case "ignore shadowing survives underscore-local forwarding analysis"
    {ocaml|let f ?(_feature = true) () = let ignore x = if x then enable () in let _v = _feature in ignore _v|ocaml}
    ~implemented:true ~honest:false;
  case "a later ignore parameter makes its call a real use"
    {ocaml|let f ?(_feature = true) ignore = ignore _feature|ocaml} ~implemented:true ~honest:false;
  case "an earlier ignore parameter makes its call a real use"
    {ocaml|let f ignore ?(_feature = true) () = ignore _feature|ocaml} ~implemented:true
    ~honest:false;
  case "a preceding top-level ignore binding makes its call a real use"
    {ocaml|let ignore x = if x then enable ()
let f ?(_feature = true) () = ignore _feature|ocaml}
    ~implemented:true ~honest:false;
  case "a preceding Stdlib.ignore alias remains a discard"
    {ocaml|let ignore = Stdlib.ignore
let f ?(feature = true) () = ignore feature|ocaml}
    ~implemented:false ~honest:false;
  case "a locally opened ignore can be effectful"
    {ocaml|let f ?(_feature = true) () = let open Effects in ignore _feature|ocaml}
    ~implemented:true ~honest:false;
  case "a local Base open preserves standard ignore"
    {ocaml|let f ?(feature = true) () = let open Base in ignore feature|ocaml} ~implemented:false
    ~honest:false;
  case "a local open can shadow the optional value itself"
    {ocaml|module Defaults = struct let feature = false end
let f ?(feature = true) () = let _ = feature in let open Defaults in feature|ocaml}
    ~implemented:false ~honest:false;
  case "underscore label makes an unimplemented option caller-visible"
    {ocaml|let f ?(_feature = true) () = ()|ocaml} ~implemented:false ~honest:true;
  case "implemented underscore label is stale" {ocaml|let f ?(_feature = 2) x = x * _feature|ocaml}
    ~implemented:true ~honest:false;
  case "a nested shadow does not pretend the outer option is used"
    {ocaml|let f ?(feature = 2) () = let g feature = feature + 1 in g 3|ocaml} ~implemented:false
    ~honest:false;
  case "a let-operator binder does not pretend the outer option is used"
    {ocaml|let f ?(feature = true) () = let* feature = source in if feature then 1 else 0|ocaml}
    ~implemented:false ~honest:false;
  case "local module value bindings shadow sequentially"
    {ocaml|let f ?(feature = true) () = let _ = feature in let module M = struct let feature = false let enabled = feature end in M.enabled|ocaml}
    ~implemented:false ~honest:false;
  case "local module opens affect following structure items"
    {ocaml|let f ?(_feature = true) () = let module M = struct open Effects let () = ignore _feature end in ()|ocaml}
    ~implemented:true ~honest:false;
  case "an object self pattern shadows the outer option"
    {ocaml|let f ?(feature = true) () = let _ = feature in object (feature) method enabled = feature end|ocaml}
    ~implemented:false ~honest:false;
  case "comments and strings do not pretend the option is used"
    {ocaml|let f ?(feature = 2) () = (* feature *) ignore "feature"|ocaml} ~implemented:false
    ~honest:false;
  case "an optional argument on a directly returned closure is inventoried"
    {ocaml|let make () = fun ?(feature = true) () -> ignore feature|ocaml} ~implemented:false
    ~honest:false;
  case "a constrained optional function is inventoried"
    {ocaml|let f = (fun ?(feature = true) () -> ignore feature : ?feature:bool -> unit -> unit)|ocaml}
    ~implemented:false ~honest:false;
  case ~argument:"feature" "optional closures returned through control flow are inventoried"
    {ocaml|let make flag = if flag then (fun ?(feature = true) () -> ignore feature) else (fun ?(feature = true) () -> ignore feature)|ocaml}
    ~implemented:false ~honest:false;
  case "a returned optional closure inherits its enclosing ignore binding"
    {ocaml|let make () = let ignore x = enable x in fun ?(_feature = true) () -> ignore _feature|ocaml}
    ~implemented:true ~honest:false;
  case "an optional closure returned normally from try is inventoried"
    {ocaml|let make () = try (fun ?(feature = true) () -> ignore feature) with _ -> raise Exit|ocaml}
    ~implemented:false ~honest:false;
  case ~argument:"feature" "an optional function exported through a tuple is inventoried"
    {ocaml|let f, sentinel = ((fun ?(feature = true) () -> ignore feature), 0)|ocaml}
    ~implemented:false ~honest:false;
  case "a later definition replaces the earlier inventory entry"
    {ocaml|let f ?(feature = true) () = enable feature
let f ?(feature = true) () = ignore feature|ocaml}
    ~implemented:false ~honest:false;
  case ~argument:"feature" "recursive-module definitions retain qualification"
    {ocaml|module rec A : sig val f : ?feature:bool -> unit -> unit end = struct
  let f ?(feature = true) () = ignore feature
end and B : sig val f : ?other:bool -> unit -> unit end = struct
  let f ?(other = true) () = enable other
end|ocaml}
    ~implemented:false ~honest:false
