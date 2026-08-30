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
let outer =
  let nested = 5 in
  nested
module Nested = struct let hidden = 6 end
|}

let refs sources = Scan.references ~exports:fixture_exports ~sources

let referenced refs value =
  List.exists refs ~f:(fun (reference : Scan.reference) -> String.equal reference.value value)

let () =
  Verdict.p "top-level lets, patterns, extensions, and externals are exports"
    (List.equal String.equal (export_keys fixture_exports)
       [
         "Sample.!@";
         "Sample.alias";
         "Sample.extended";
         "Sample.outer";
         "Sample.pair";
         "Sample.plain";
         "Sample.primitive";
       ]);
  Verdict.p "nested lets and nested-module values are not exports"
    (not
       (List.exists fixture_exports ~f:(fun (export : Scan.export) ->
            String.equal export.value "nested" || String.equal export.value "hidden")));
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
  let included = refs [ ("included.ml", "include Sample\n") ] in
  Verdict.p "include counts as a reference to every re-exported value"
    (List.length included = List.length fixture_exports);
  let counts = Scan.counts ~exports:fixture_exports direct in
  Verdict.p "a synthetic zero-reference export is detected"
    (Hashtbl.find_exn counts "Sample.primitive" = 0)
