(** Finding self-decided verdicts that a test prints itself instead of routing through {!Verdict}.

    gh-ocannl-601 established the rule and swept the sites that existed then: a claim the test
    decides — [<label>: <bool>] — must go through [Verdict], so that a failure exits the process
    nonzero. Printed and exited 0, the same line is gated only by the golden diff, and the natural
    next move on a failing diff is [dune promote], which records [<label>: false] as the expected
    output. In a golden that is nothing but verdict lines, a blessed regression and a deliberately
    recorded negative fact are the same text.

    That sweep was one-time and left no mechanical trace, so the population regrew: a test written
    afterwards ([bandwidth_calibration], gh-ocannl-578) came in with four fresh
    [Stdio.printf "…: %b\n"] claims, and nothing failed or warned. This module is the reader behind
    the ratchet that stops that (gh-ocannl-668).

    {1 What shape is recognised}

    The literal-label form, and only it: a format string whose single argument-consuming conversion
    is a bare [%b] at the end, preceded by a label ending in a colon. ["k-blocks fused: %b\n"] is
    one; ["%s fused: %b\n"] is not, because the label is computed — those are issue #624's remaining
    sites, which need per-site judgement and are deliberately out of reach here.

    Carrying any other conversion is therefore the escape hatch a descriptive print already has. The
    other is a named exemption in the test that consumes this, which is where a literal-label [%b]
    that genuinely is not an assertion goes.

    {1 Why it reads the parse tree}

    Same argument as {!Config_key_scan}, whose parse helpers this shares, and the same conclusion:
    this module parses, it does not match text. A text scan reads prose as code — this file's own
    documentation quotes the shape it hunts — and cannot tell a format written over a line
    continuation from one written on a line. On the parse tree a string literal carries its decoded
    value whatever escapes or delimiters produced it.

    Attribute payloads are the one thing deliberately skipped. A documentation comment becomes an
    [@@@ocaml.doc] attribute holding a string, so a comment quoting the shape arrives as a literal
    to flag; that is prose, and prose is not a print. Extension payloads are {e not} skipped — a
    [%cd] or [%expect] payload is code, or is the golden text of one.

    {1 Every claim-shaped literal, not only those applied to a [printf]}

    What the check fails on is the literal, wherever it sits; which function receives it is reported
    but decides nothing. The alternative — flag only what is applied to a printing function — is
    evaded by any wrapper, including the [let p name b = Stdio.printf "%s: %b\n" name b] helpers
    several tests defined before {!Verdict} existed. Those helpers are the exact population this
    ratchet exists to keep from regrowing, so a claim-shaped format string in a test source is
    either an assertion or, by exemption, a documented exception. *)

open Base
open Parsetree

(* The parse helpers are Config_key_scan's: one place in this repository decides how an OCaml source
   is read for a scan, and it carries the notes on what varies between the compiler versions the
   opam files declare a floor for (its [flatten_longident] comment). *)
module Read = Config_key_scan

type directive = { start : int; stop : int; conversion : char }
(** A conversion directive as the format spells it: where the ['%'] is, where the conversion
    character is, and what it is. A bare [%b] has [stop = start + 1]; anything wider carries flags,
    a width or a precision between the two. *)

(** Directives that consume no argument, so a format carrying one still takes however many arguments
    it took: [%%] (a literal per cent), [%!] (flush), [%,] (a separator that prints nothing).
    Everything else counts as consuming — including the nested-format spellings [%\{…%\}] and
    [%(…%)], whose presence therefore takes a format {e out} of the recognised shape rather than
    into it. Erring that way costs a report that was never made, never a false one. *)
let consumes_nothing = function '%' | '!' | ',' -> true | _ -> false

(** The conversion directives of a format string, in order. Scans for ['%'], then flags, width and
    precision, then the conversion character. A trailing ['%'] with nothing after it is not a
    directive and ends the scan. *)
let directives format =
  let length = String.length format in
  let rec skip_modifiers index =
    if index >= length then index
    else
      match format.[index] with
      | '-' | '0' | '+' | ' ' | '#' | '.' | '*' -> skip_modifiers (index + 1)
      | character when Char.is_digit character -> skip_modifiers (index + 1)
      | _ -> index
  in
  let rec scan position found =
    if position >= length then List.rev found
    else if not (Char.equal format.[position] '%') then scan (position + 1) found
    else
      let stop = skip_modifiers (position + 1) in
      if stop >= length then List.rev found
      else scan (stop + 1) ({ start = position; stop; conversion = format.[stop] } :: found)
  in
  scan 0 []

(** What a reader sees printed by a stretch of format text that consumes no arguments: [%%] becomes
    a per cent sign, [%!] and [%,] print nothing, everything else is itself. Applied to the text
    before and after the [%b], where by construction every directive left is one of those. *)
let printed_text text =
  let length = String.length text in
  let buffer = Buffer.create length in
  let rec scan index =
    if index >= length then Buffer.contents buffer
    else if Char.equal text.[index] '%' && index + 1 < length then (
      if Char.equal text.[index + 1] '%' then Buffer.add_char buffer '%';
      scan (index + 2))
    else (
      Buffer.add_char buffer text.[index];
      scan (index + 1))
  in
  scan 0

(** [claim_label format] is the label of the claim [format] prints, when it prints one.

    Three things must hold at once:

    - exactly one argument-consuming conversion in the whole format, and it is a bare [%b] — a width
      or a flag means the print is laying out a column, which is formatting rather than asserting;
    - what follows it prints as whitespace, or as nothing: a newline, a blank line, a [%!] flush;
    - what precedes it ends in a colon, with a non-empty label before that colon.

    So ["k-blocks fused: %b\n"] yields [Some "k-blocks fused"], while ["%s fused: %b\n"],
    ["fused: %b (expect false)\n"] and ["fused? %b\n"] all yield [None] — the first because the
    label is computed, the other two because the line is not the bare claim form the gh-ocannl-601
    sweep normalised to, and a reader cannot take their boolean at face value. *)
let claim_label format =
  match List.filter (directives format) ~f:(fun d -> not (consumes_nothing d.conversion)) with
  | [ { start; stop; conversion = 'b' } ] when stop = start + 1 ->
      let tail = printed_text (String.subo format ~pos:(stop + 1)) in
      if not (String.for_all tail ~f:Char.is_whitespace) then None
      else
        let head = String.rstrip (printed_text (String.sub format ~pos:0 ~len:start)) in
        Option.bind (String.chop_suffix head ~suffix:":") ~f:(fun label ->
            let label = String.strip label in
            if String.is_empty label then None else Some label)
  | _ -> None

type site = {
  label : string;  (** The claim as the format spells it, without the colon. *)
  format : string;  (** The whole format, so a report can show what is written. *)
  line : int;  (** 1-based, as the parser located the literal. *)
  printer : string option;
      (** The function the literal is an argument of, where it is one: ["Stdio.printf"],
          ["Printf.eprintf"]. [None] for a literal bound to a name, handed to a wrapper or built
          into a list — flagged just the same; this only says what the report can show. *)
}

type scan = {
  sites : site list;  (** The claim-shaped literals, in source order. *)
  literals : int;  (** Every string literal in expression position. *)
  applied_literals : int;  (** How many of those are an argument of a named function. *)
}
(** [literals] and [applied_literals] are not verdicts; they are what a blind walk cannot produce. A
    walk whose expression hook stopped firing reports zero of both over a corpus where thousands are
    expected, so the consuming test can say that its empty offender list came from a scan that read
    something — rather than leaving "no offenders" and "no reading" the same result. *)

(** [scan content] reads one source. Raises if [content] does not parse: a reader that cannot read
    its input must say so rather than report an empty census. *)
let scan content =
  let ast = Read.structure_of content in
  (* The function each literal is an argument of, noted as the application is entered — an iterator
     visits a node before its children, so the argument's own visit below finds it. *)
  let printers = Hashtbl.create (module Int) in
  let sites = ref [] and literals = ref 0 in
  let iterator =
    {
      Ast_iterator.default_iterator with
      (* Prose is not a print: see the note on attribute payloads above. *)
      attribute = (fun _ _ -> ());
      expr =
        (fun self expr ->
          (match expr.pexp_desc with
          | Pexp_apply (callee, arguments) -> (
              match Read.longident_of callee with
              | Some path ->
                  let name = String.concat ~sep:"." path in
                  List.iter arguments ~f:(fun (_, argument) ->
                      if Option.is_some (Read.string_literal argument) then
                        Hashtbl.set printers ~key:argument.pexp_loc.loc_start.pos_cnum ~data:name)
              | None -> ())
          | _ -> ());
          (match Read.string_literal expr with
          | Some value -> (
              Int.incr literals;
              match claim_label value with
              | Some label ->
                  sites :=
                    {
                      label;
                      format = value;
                      line = expr.pexp_loc.loc_start.pos_lnum;
                      printer = Hashtbl.find printers expr.pexp_loc.loc_start.pos_cnum;
                    }
                    :: !sites
              | None -> ())
          | None -> ());
          Ast_iterator.default_iterator.expr self expr);
    }
  in
  iterator.structure iterator ast;
  {
    sites = List.rev !sites;
    literals = !literals;
    applied_literals = Hashtbl.length printers;
  }
