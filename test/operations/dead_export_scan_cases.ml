open Base
module Scan = Test_utils.Dead_export_scan

let export_keys exports = List.map exports ~f:Scan.export_key |> List.sort ~compare:String.compare

let fixture_exports =
  Scan.exports_of_source ~source:"arrayjit/lib/sample.ml"
    {|
let plain = 1
let pair, alias = (2, 3)
let (!@) x = x
let%trace extended = 4
external primitive : int -> int = "fixture_primitive"
type t = Root [@@deriving sexp_of, compare, equal]
type named = Named [@@deriving sexp, compare, equal]
type poly = [ `One | `Two ] [@@deriving sexp]
type group_a = Group_a and group_b = Group_b [@@deriving equal]
let outer =
  let nested = 5 in
  nested
module Nested = struct
  let hidden = 6
  type nested = Nested [@@deriving equal]
end
|}

let refs sources = Scan.references ~exports:fixture_exports ~sources

let referenced refs value =
  List.exists refs ~f:(fun (reference : Scan.reference) -> String.equal reference.value value)

let () =
  Verdict.p "top-level lets, patterns, extensions, and externals are exports"
    (List.equal String.equal (export_keys fixture_exports)
       [
         "Sample.!@";
         "Sample.__poly_of_sexp__";
         "Sample.alias";
         "Sample.compare";
         "Sample.compare_named";
         "Sample.equal";
         "Sample.equal_group_a";
         "Sample.equal_group_b";
         "Sample.equal_named";
         "Sample.extended";
         "Sample.named_of_sexp";
         "Sample.outer";
         "Sample.pair";
         "Sample.plain";
         "Sample.poly_of_sexp";
         "Sample.primitive";
         "Sample.sexp_of_named";
         "Sample.sexp_of_poly";
         "Sample.sexp_of_t";
       ]);
  Verdict.p "nested lets and nested-module derived values are not exports"
    (not
       (List.exists fixture_exports ~f:(fun (export : Scan.export) ->
            String.equal export.value "nested"
            || String.equal export.value "hidden"
            || String.equal export.value "equal_nested")));
  Verdict.p "a deriving on the last declaration covers its whole recursive type group"
    (List.mem (export_keys fixture_exports) "Sample.equal_group_a" ~equal:String.equal
    && List.mem (export_keys fixture_exports) "Sample.equal_group_b" ~equal:String.equal);
  Verdict.p "sexp derives the polymorphic-variant parser helper"
    (List.mem (export_keys fixture_exports) "Sample.__poly_of_sexp__" ~equal:String.equal);
  Verdict.p "dune select alternatives are not mistaken for module sources"
    (Option.is_none (Scan.module_name_of_source "arrayjit/lib/sample.missing.ml"));
  let direct =
    refs
      [
        ("consumer.ml", "let a = Sample.plain\nlet b = Wrapper.Sample.pair\nlet c = Sample.(!@) 1\n");
        ("arrayjit/lib/sample.ml", "let self = Sample.outer\n");
      ]
  in
  Verdict.p "direct and nested qualified paths count outside the defining source"
    (referenced direct "plain" && referenced direct "pair" && referenced direct "!@");
  Verdict.p "references in the defining source do not count" (not (referenced direct "outer"));
  let aliased =
    refs
      [
        ( "aliases.ml",
          "module S = Sample\n\
           module T = S\n\
           let a = T.alias\n\
           let b = let module U = T in U.extended\n" );
      ]
  in
  Verdict.p "structure, chained, and local module aliases count"
    (referenced aliased "alias" && referenced aliased "extended");
  let opened =
    refs
      [
        ( "opened.ml",
          "open Sample\n\
           let a = plain\n\
           let b = let open Sample in pair\n\
           let c = let primitive = 0 in primitive\n" );
      ]
  in
  Verdict.p "unqualified identifiers in an open scope count conservatively"
    (referenced opened "plain" && referenced opened "pair" && referenced opened "primitive");
  let inert = refs [ ("inert.ml", "let s = \"Sample.outer\" (* Sample.alias *)\n") ] in
  Verdict.p "comments and strings do not count as references" (List.is_empty inert);
  let extensions =
    refs
      [
        ( "extensions.ml",
          "module S = Sample\n\
           let a = [%equal: S.named] Named Named\n\
           let b = [%compare: Wrapper.Sample.t] Root Root\n\
           let c = [%sexp_of: Sample.named list]\n\
           let d = [%of_sexp: Sample.named]\n" );
      ]
  in
  Verdict.p "equal, compare, sexp_of, and of_sexp extensions reference derived values"
    (referenced extensions "equal_named"
    && referenced extensions "compare"
    && referenced extensions "sexp_of_named"
    && referenced extensions "named_of_sexp");
  Verdict.p "an extension does not credit a different derivation from the same type"
    (not (referenced extensions "compare_named"));
  let derived =
    refs [ ("derived.ml", "type t = Sample.named [@@deriving sexp, compare, equal]\n") ]
  in
  Verdict.p "a deriving references converters of its component types"
    (referenced derived "sexp_of_named"
    && referenced derived "named_of_sexp"
    && referenced derived "compare_named"
    && referenced derived "equal_named");
  let ignored =
    refs
      [
        ( "ignored.ml",
          "type a = (Sample.named[@sexp.opaque]) [@@deriving sexp]\n\
           type b = (Sample.named[@compare.ignore]) [@@deriving compare]\n\
           type c = (Sample.named[@equal.ignore]) [@@deriving equal]\n\
           let a = [%sexp_of: (Sample.named[@sexp.opaque]) list]\n\
           let b = [%sexp_of: Sample.named sexp_opaque]\n" );
      ]
  in
  Verdict.p "opaque and ignored types do not credit unused derived values"
    (not
       (referenced ignored "sexp_of_named"
       || referenced ignored "named_of_sexp"
       || referenced ignored "compare_named"
       || referenced ignored "equal_named"));
  let functor_application = refs [ ("functor.ml", "let f = [%sexp_of: F(X).t]\n") ] in
  Verdict.p "a functor-application type path is accepted without guessed credit"
    (List.is_empty functor_application);
  let included = refs [ ("included.ml", "include Sample\n") ] in
  Verdict.p "include counts as a reference to every re-exported value"
    (List.length included = List.length fixture_exports);
  let counts = Scan.counts ~exports:fixture_exports direct in
  Verdict.p "a synthetic zero-reference export is detected"
    (Hashtbl.find_exn counts "Sample.primitive" = 0);
  Verdict.p "a synthetic zero-reference derived export is detected"
    (Hashtbl.find_exn counts "Sample.equal_named" = 0);
  let extension_counts = Scan.counts ~exports:fixture_exports extensions in
  Verdict.p "an export referenced only through an extension point is not reported dead"
    (Hashtbl.find_exn extension_counts "Sample.equal_named" > 0)
