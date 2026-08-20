(** How the claim-shape reader behind [verdict_ratchet] reads a format and a source, exercised on
    input built to break it rather than on whatever the test directories happen to contain today.

    [Verdict_scan] decides what counts as a self-decided claim, and both of its mistakes are quiet.
    A shape it stops recognising leaves an empty offender list, which reads exactly like a clean
    tree -- the failure mode gh-ocannl-668 exists to close, arriving in the check meant to close it.
    A shape it recognises too eagerly turns a descriptive print into a demand for an exemption, and
    an exemption list long enough stops being read.

    So the two halves are pinned separately: which formats are the claim shape, and which literals
    of a source the walk reaches. *)

open Base
open Stdio
module Scan = Test_utils.Verdict_scan

(* Failures go through [Verdict], so that a regression exits nonzero instead of being `dune
   promote`d into the golden as the expected output (gh-ocannl-601) -- the rule this whole check is
   about, and not one to be exempt from. *)
let fail fmt = Printf.ksprintf Verdict.fail fmt

(* Planted for [verdict_ratchet], which names both in its canary list and fails if its scan of the
   test directories does not find them here.

   They are ordinary fixture inputs -- the two spellings below feed the shape cases -- and they are
   also the only thing standing between a walk that went blind and an empty offender list read as a
   clean tree. The second is written over a line continuation on purpose: its decoded value is the
   claim shape while no single line of this file contains it, so a reader that matched text would
   find the first and miss this one. *)
let planted_plain = "planted canary: %b\n"

let planted_continued = "planted canary over a \
                         continuation: %b\n"

(* Formats that ARE the claim shape, with the label each yields. *)
let claim_cases =
  [
    ("the plain form", "fused: %b\n", "fused");
    ("no trailing newline", "fused: %b", "fused");
    ("a blank line either side", "\nfused: %b\n\n", "fused");
    ("a flush directive consumes no argument", "fused: %b\n%!", "fused");
    ("nor does a separator", "fused: %b\n%,", "fused");
    ("a per cent sign in the label", "100%% covered: %b\n", "100% covered");
    ("a colon inside the label", "kernel k: fused: %b\n", "kernel k: fused");
    ("no space after the colon", "fused:%b\n", "fused");
    ("a tab before the boolean", "fused:\t%b\n", "fused");
    (* The two literals the ratchet looks for, which are fixture inputs first. The continuation is
       what says the reader works on decoded values: its source line ends mid-label. *)
    ("the planted canary", planted_plain, "planted canary");
    ("the planted canary over a continuation", planted_continued, "planted canary over a continuation");
  ]

(* Formats that are NOT, each for its own reason. The first is issue #624's population, deliberately
   out of scope: what its claim even is takes per-site judgement. The rest are descriptive prints,
   which is the escape hatch that keeps the exemption list short enough to read. *)
let non_claim_cases =
  [
    ("a computed label is #624's, not this check's", "%s fused: %b\n");
    ("a second boolean", "fused: %b %b\n");
    ("an interpolated value after the boolean", "fused: %b (%d blocks)\n");
    ("an annotation after the boolean", "fused: %b (expect false)\n");
    ("a width means the print is laying out a column", "fused: %6b\n");
    ("so does a left-justifying flag", "fused: %-6b\n");
    ("no colon, so no label", "fused? %b\n");
    ("no label before the colon", ": %b\n");
    ("a bare boolean", "%b\n");
    ("nothing before it but a blank line", "\n%b\n");
    ("not a boolean at all", "fused: %d\n");
    ("a boolean that is not the last conversion", "fused: %b, blocks: %d\n");
    ("prose after the boolean", "fused: %b -- and the reason\n");
    (* A per cent with nothing after it is not a directive, and must not run the scan off the end of
       the format either. *)
    ("a trailing per cent neither crashes the scan nor vanishes", "fused: %b\n%");
    (* Nested formats are counted as consuming, which takes them out of the shape rather than into
       it: erring that way costs a report that was never made, never a false one. *)
    ("a nested format is not read as a bare boolean", "fused: %b %{%d%}\n");
  ]

(* Whole sources, with the labels the walk must report and where it says each one sits. A source
   case exists where the answer depends on the SOURCE and not on the format: where the literal sits,
   how it is spelled, whether it is prose. *)
let source_cases =
  [
    ( "a claim applied to a printing function names it",
      {ocaml|let () = Stdio.printf "fused: %b\n" true|ocaml},
      [ ("fused", Some "Stdio.printf") ] );
    ( "the receiver is reported as written",
      {ocaml|open Stdio
let () = eprintf "fused: %b\n" true|ocaml},
      [ ("fused", Some "eprintf") ] );
    (* The wrapper is why the check is about the literal and not about the call: a
       `let p name b = printf "%s: %b\n" name b` helper is what these sites looked like before
       Verdict existed, and pointing the scan at printing functions alone would miss every claim
       routed through one. *)
    ( "a claim handed to a wrapper is still a claim",
      {ocaml|let () = report "fused: %b\n" true|ocaml},
      [ ("fused", Some "report") ] );
    ( "a claim bound to a name, applied to nothing",
      {ocaml|let message = "fused: %b\n"|ocaml},
      [ ("fused", None) ] );
    ( "a claim inside a list is reached",
      {ocaml|let messages = [ "fused: %b\n"; "tiled: %b\n" ]|ocaml},
      [ ("fused", None); ("tiled", None) ] );
    (* Delimiters and escapes are the parser's business, not the scan's: what is compared is the
       decoded value. *)
    ( "a quoted-string literal decodes the same way",
      {ocaml|let () = printf {xx|fused: %b|xx} true|ocaml},
      [ ("fused", Some "printf") ] );
    ( "a line continuation joins the label",
      "let () = printf \"fused over a \\\n   continuation: %b\\n\" true",
      [ ("fused over a continuation", Some "printf") ] );
    (* Prose is not a print. An ordinary comment the parser discards outright; a documentation
       comment survives as an attribute holding a string, which is why attribute payloads are
       skipped rather than merely hoped about. *)
    ("an ordinary comment quoting the shape", {ocaml|(* fused: %b *) let x = 1|ocaml}, []);
    ("a documentation comment quoting the shape", {ocaml|(** fused: %b *) let x = 1|ocaml}, []);
    ( "an attached documentation comment quoting the shape",
      {ocaml|let x = 1 (** fused: %b *)|ocaml},
      [] );
    (* An extension payload is code, or the golden text of some, so it is NOT skipped. *)
    ( "a claim inside an extension payload is reached",
      {ocaml|let () = [%probe let () = printf "fused: %b\n" true]|ocaml},
      [ ("fused", Some "printf") ] );
    ( "a claim quoted inside a string is not a claim",
      {ocaml|let doc = "write printf \"fused: %b\\n\" here"|ocaml},
      [] );
  ]

let show_printer = Option.value ~default:"<unapplied>"

let () =
  List.iter claim_cases ~f:(fun (name, format, expected) ->
      match Scan.claim_label format with
      | Some found when String.equal found expected -> printf "ok: claim shape -- %s\n" name
      | Some found -> fail "claim shape -- %s: expected label %S, found %S" name expected found
      | None -> fail "claim shape -- %s: expected label %S, found no claim" name expected);
  List.iter non_claim_cases ~f:(fun (name, format) ->
      match Scan.claim_label format with
      | None -> printf "ok: not a claim -- %s\n" name
      | Some found -> fail "not a claim -- %s: read it as the claim %S" name found);
  List.iter source_cases ~f:(fun (name, source, expected) ->
      let found =
        try
          List.map (Scan.scan source).Scan.sites ~f:(fun site ->
              (site.Scan.label, site.Scan.printer))
        with _ ->
          fail "source -- %s: the snippet does not parse" name;
          []
      in
      let render sites =
        List.map sites ~f:(fun (label, printer) -> label ^ " (" ^ show_printer printer ^ ")")
        |> String.concat ~sep:"; "
      in
      if
        List.equal
          (fun (a, b) (c, d) -> String.equal a c && Option.equal String.equal b d)
          found expected
      then printf "ok: source -- %s\n" name
      else fail "source -- %s: expected [%s], found [%s]" name (render expected) (render found));
  (* The census the ratchet leans on to tell "nothing to report" from "nothing read": on a source
     with literals it must count them, and it must place the ones an application receives. *)
  let counted = Scan.scan {ocaml|let unapplied = "c"
let () = printf "a" "b"|ocaml} in
  Verdict.p "the walk counts every string literal it passes" (counted.Scan.literals = 3);
  Verdict.p "and places the ones a named function receives" (counted.Scan.applied_literals = 2)
