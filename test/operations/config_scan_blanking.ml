(** The blanking pass that both configuration scanners rest on, exercised directly.

    [Config_key_scan.blank_bodies] decides what counts as code, so every one of its mistakes is
    silent by construction: a desynchronised walk blanks live source, and keys that vanish from a
    scan look exactly like keys that were never read. Two such bugs reached review on PR #340 --
    prose read as a call site, then a quoted string inside a comment read as a nested comment --
    which is the argument for testing the walker on hostile input rather than only on the library
    sources that happen to exist today.

    The cases below are that hostile input: OCaml the compiler accepts and the walker must agree
    with. *)

open Base
open Stdio
module Scan = Test_utils.Config_key_scan

(* A key that MUST be found is spelled with the marker; a key that must NOT be found sits in prose
   or in a blanked body. Each case pairs a snippet with the keys the scan should report. *)
let cases =
  [
    ("plain call site", {ocaml|let x = get ~arg_name:"alpha" ~default:""|ocaml}, [ "alpha" ]);
    ( "optional-parameter default",
      {ocaml|let f ?(arg_name = "beta") () = get ~arg_name ~default:""|ocaml},
      [ "beta" ] );
    ( "prose in a comment is not a call site",
      {ocaml|(* pass ~arg_name:"phantom" here *) let x = get ~arg_name:"gamma" ~default:""|ocaml},
      [ "gamma" ] );
    ( "nested comments close in the right order",
      {ocaml|(* outer (* inner ~arg_name:"phantom" *) still comment *) let x = get ~arg_name:"delta" ~default:""|ocaml},
      [ "delta" ] );
    ( "a string inside a comment does not swallow the file",
      {ocaml|(* the "(*" spelling *) let x = get ~arg_name:"epsilon" ~default:""|ocaml},
      [ "epsilon" ] );
    ( "a quoted string inside a comment is not a nested comment",
      {ocaml|(* example: {| (* text |} *) let x = get ~arg_name:"zeta" ~default:""|ocaml},
      [ "zeta" ] );
    ( "a tagged quoted string inside a comment",
      {ocaml|(* example: {sql| (* |sql} *) let x = get ~arg_name:"eta" ~default:""|ocaml},
      [ "eta" ] );
    ( "a double quote as a character literal does not open a string",
      {ocaml|let q = '"' in let x = get ~arg_name:"theta" ~default:""|ocaml},
      [ "theta" ] );
    ( "an escaped quote inside a string does not end it",
      {ocaml|let s = "a \" (* not a comment *)" in let x = get ~arg_name:"iota" ~default:""|ocaml},
      [ "iota" ] );
    ( "a backslash-backslash char literal does not eat the closing quote",
      {ocaml|let c = '\\' in let x = get ~arg_name:"kappa" ~default:""|ocaml},
      [ "kappa" ] );
    ( "a type variable is not a character literal",
      {ocaml|let f (x : 'a list) = x let y = get ~arg_name:"lambda" ~default:""|ocaml},
      [ "lambda" ] );
    ( "a record literal is not a quoted string",
      {ocaml|let r = { field } let x = get ~arg_name:"mu" ~default:""|ocaml},
      [ "mu" ] );
  ]

(* The other half of the contract: offsets survive blanking, so a scanner can report the ORIGINAL
   line for a position it found in the blanked text. *)
let offsets_preserved =
  List.for_all cases ~f:(fun (_, source, _) ->
      let blanked = Scan.blank_bodies source in
      let blanked_strings = Scan.blank_bodies ~strings:true source in
      String.length blanked = String.length source
      && String.length blanked_strings = String.length source
      && String.count blanked ~f:(Char.equal '\n') = String.count source ~f:(Char.equal '\n'))

let () =
  let ok = ref true in
  List.iter cases ~f:(fun (name, source, expected) ->
      let found = List.sort ~compare:String.compare (Scan.keys_in_source source) in
      let expected = List.sort ~compare:String.compare expected in
      if List.equal String.equal found expected then printf "ok: %s\n" name
      else (
        ok := false;
        printf "FAIL: %s -- expected [%s], found [%s]\n" name
          (String.concat ~sep:"; " expected)
          (String.concat ~sep:"; " found)));
  (* Blanking a body must not shorten the text, or every reported line number drifts. *)
  if offsets_preserved then printf "ok: offsets and line counts preserved\n"
  else (
    ok := false;
    printf "FAIL: blanking changed the length or the line count of a snippet\n");
  if not !ok then Stdlib.exit 1
