(** gh-ocannl-807: the changelog's `## [Unreleased]` section obeys the editorial convention
    mechanically.

    `CHANGES.md` is written in editorial passes rather than in feature PRs, and the two rules a pass
    has to hold are stated in [docs/agent-notes/conventions.md]: a top-level bullet is at most three
    lines, and it cites the record the reader can follow — a `gh-ocannl-NNN` issue or a
    `lukstafi/ocannl-staging` PR (`PR #NNN`). Both were enforced by hand on the 2026-09-02 batch
    catch-up (lukstafi/ocannl-staging PR #601, review findings 3915031304 and 3915183641), in rounds
    three and four of eight. A reviewer re-deriving a decidable rule from prose costs a round per
    rule; this file costs none.

    Only the Unreleased section is read. Released sections are history — they record what a past
    release said, under whatever conventions held then, and rewriting them to satisfy today's rules
    would falsify the record rather than fix anything.

    {1 What the golden holds}

    The claims, and no tally: the number of bullets in Unreleased moves on every editorial pass and
    drops to zero at every release, so a count in the golden would make every correct change a
    promote indistinguishable from blessing a regression. The exact counts go to stderr. What a
    count was there for — the assurance that a scan reporting nothing read something — is kept as
    the population guard {!Verdict.p_all} applies, plus the synthetic controls below, which hold the
    two rules against text built to break them so a checker that stopped deciding cannot pass here
    by finding nothing. *)

open Base
open Stdio
open Verdict.Claims

type bullet = { first_line : string; lines : string list }
(** A top-level bullet: the physical lines of one `- ` item, first line first. *)

let is_top_level_bullet line = String.is_prefix line ~prefix:"- "

let is_continuation line =
  String.is_prefix line ~prefix:"  " && not (String.is_empty (String.strip line))

(** The lines of `## [Unreleased]`, from the heading to the next `## ` heading or the end of file.
    Subheadings (`### Added` and friends) stay in: they separate bullets, they never continue one.
*)
let unreleased_section lines =
  let rec find = function
    | [] -> None
    | line :: rest when String.is_prefix line ~prefix:"## [Unreleased]" ->
        Some (List.take_while rest ~f:(fun l -> not (String.is_prefix l ~prefix:"## ")))
    | _ :: rest -> find rest
  in
  find lines

(** The top-level bullets of a section. A bullet runs from its `- ` line to the first line that is
    not an indented, non-blank continuation of it. *)
let bullets_of lines =
  let rec loop acc = function
    | [] -> List.rev acc
    | line :: rest when is_top_level_bullet line ->
        let continuation, rest = List.split_while rest ~f:is_continuation in
        loop ({ first_line = line; lines = line :: continuation } :: acc) rest
    | _ :: rest -> loop acc rest
  in
  loop [] lines

let max_lines = 3
let within_line_budget bullet = List.length bullet.lines <= max_lines

(** A digit-carrying occurrence of one of the two citation forms. Bare "gh-ocannl-" or "PR #" prose
    with no number behind it is not a citation, which is why the digit is part of the question. *)
let cites_record bullet =
  let text = String.concat ~sep:" " bullet.lines in
  let followed_by_digit prefix =
    let plen = String.length prefix and tlen = String.length text in
    let rec at i =
      if i + plen >= tlen then false
      else if
        String.equal (String.sub text ~pos:i ~len:plen) prefix && Char.is_digit text.[i + plen]
      then true
      else at (i + 1)
    in
    at 0
  in
  followed_by_digit "gh-ocannl-" || followed_by_digit "PR #"

let opening bullet =
  let text = String.strip bullet.first_line in
  if String.length text <= 78 then text else String.prefix text 78 ^ "..."

let report label offenders =
  List.iter offenders ~f:(fun bullet -> eprintf "  %s: %s\n" label (opening bullet))

(* The synthetic controls: the same two predicates over text built to break them. Without these, a
   checker that stopped deciding -- a section finder that finds nothing, a citation reader that
   answers true for everything -- passes over a clean changelog and the golden says so in green. *)
let control_section =
  [
    "";
    "### Added";
    "";
    "- A compliant bullet, one line, citing (gh-ocannl-807).";
    "- A bullet whose continuation lines carry it past the budget, line two of the four";
    "  it is going to take, line three of the four it is going to take, and line four";
    "  which is where it exceeds what the convention allows for one entry, and it does";
    "  cite (gh-ocannl-807) so only the length rule can flag it.";
    "- A bullet with no record behind it at all, mentioning PR # and gh-ocannl- as prose.";
  ]

let () =
  let path =
    match Array.to_list (Sys.get_argv ()) with
    | _ :: path :: _ -> path
    | _ ->
        fail "the changelog scan is handed the path to CHANGES.md";
        Stdlib.exit 1
  in
  let contents = Stdlib.In_channel.with_open_bin path Stdlib.In_channel.input_all in
  let lines = String.split_lines contents in
  let section =
    match unreleased_section lines with
    | Some section -> section
    | None ->
        fail "CHANGES.md carries a `## [Unreleased]` section";
        Stdlib.exit 1
  in
  let bullets = bullets_of section in
  eprintf "Read %d lines of `## [Unreleased]`, %d top-level bullets.\n" (List.length section)
    (List.length bullets);
  print_endline
    "The `## [Unreleased]` section of CHANGES.md against the editorial convention stated in\n\
     docs/agent-notes/conventions.md (gh-ocannl-807). Released sections are history and are not\n\
     read; the counts scanned go to stderr, since a tally in a golden moves on every editorial\n\
     pass (gh-ocannl-665).\n";
  report "over three lines" (List.filter bullets ~f:(Fn.non within_line_budget));
  p_all "every Unreleased bullet is at most three lines" bullets ~f:within_line_budget;
  report "no gh-ocannl-NNN or PR #NNN citation" (List.filter bullets ~f:(Fn.non cites_record));
  p_all "every Unreleased bullet cites gh-ocannl-NNN or a PR #NNN" bullets ~f:cites_record;
  let control = bullets_of control_section in
  p "the section reader finds the three synthetic control bullets" (List.length control = 3);
  p_exists "the length rule flags a synthetic four-line bullet" control
    ~f:(Fn.non within_line_budget);
  p_exists "the citation rule flags a synthetic uncited bullet" control ~f:(Fn.non cites_record);
  p_exists "both rules pass a synthetic compliant bullet" control ~f:(fun bullet ->
      within_line_budget bullet && cites_record bullet)
