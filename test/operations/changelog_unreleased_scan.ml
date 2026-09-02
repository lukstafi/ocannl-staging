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

    {1 What a bullet is}

    Both rules are decided over a bullet's full extent, so the parser has to agree with Markdown
    about where an item ends, not with the common case. A list item stays open across a BLANK LINE
    when an indented block follows it — a second paragraph of the same entry — and a parser that
    ends the item at the blank counts a four-paragraph entry as three lines and cannot see a
    citation that sits in the discarded half. An UNRELEASED HEADING is likewise required to be
    unique: a merge or an editorial pass that leaves two of them puts bullets under a second anchor
    that a first-match reader never looks at. Both were review findings on this file
    (lukstafi/ocannl-staging PR #603), and both are what the synthetic controls below hold.

    The line budget counts a bullet's non-blank lines: the blank between two paragraphs of one entry
    separates prose rather than adding any.

    {1 What the golden holds}

    The claims, and no tally: the number of bullets in Unreleased moves on every editorial pass and
    drops to zero at every release, so a count in the golden would make every correct change a
    promote indistinguishable from blessing a regression. The exact counts go to stderr. What a
    count was there for — the assurance that a scan reporting nothing read something — is kept as
    the population guard {!Verdict.p_all} applies, plus the synthetic controls, which hold every
    rule against text built to break it so a checker that stopped deciding cannot pass here by
    finding nothing. *)

open Base
open Stdio
open Verdict.Claims

type bullet = { first_line : string; lines : string list }
(** A top-level bullet: the physical lines of one `- ` item, first line first, blank separators
    included where they sit between two blocks of the same item. *)

let is_blank line = String.is_empty (String.strip line)
let is_top_level_bullet line = String.is_prefix line ~prefix:"- "

(** An indented, non-blank line: the continuation of whatever item is open. *)
let is_indented line = String.is_prefix line ~prefix:"  " && not (is_blank line)

let unreleased_heading = "## [Unreleased]"
let is_unreleased_heading line = String.is_prefix line ~prefix:unreleased_heading
let unreleased_headings lines = List.filter lines ~f:is_unreleased_heading

(** The lines under `## [Unreleased]`, to the next `## ` heading or the end of file. Subheadings
    (`### Added` and friends) stay in: they separate bullets, they never continue one. Returns
    [None] unless there is exactly ONE Unreleased heading — with two, a first-match reader silently
    stops looking before the second one's bullets. *)
let unreleased_section lines =
  match unreleased_headings lines with
  | [ _ ] ->
      let rec find = function
        | [] -> None
        | line :: rest when is_unreleased_heading line ->
            Some (List.take_while rest ~f:(fun l -> not (String.is_prefix l ~prefix:"## ")))
        | _ :: rest -> find rest
      in
      find lines
  | _ -> None

(** The lines of one open item, and what is left after it. A run of blank lines belongs to the item
    only when an indented line follows it: that is the second-paragraph shape. A blank run at the
    end of a section, or one followed by a heading or the next bullet, closes the item and is not
    consumed. *)
let rec item_lines acc rest =
  match rest with
  | line :: tail when is_indented line -> item_lines (line :: acc) tail
  | line :: _ when is_blank line -> (
      let blanks, after = List.split_while rest ~f:is_blank in
      match after with
      | next :: _ when is_indented next -> item_lines (List.rev_append blanks acc) after
      | _ ->
          ignore line;
          (List.rev acc, rest))
  | _ -> (List.rev acc, rest)

(** The top-level bullets of a section. *)
let bullets_of lines =
  let rec loop acc = function
    | [] -> List.rev acc
    | line :: rest when is_top_level_bullet line ->
        let continuation, rest = item_lines [] rest in
        loop ({ first_line = line; lines = line :: continuation } :: acc) rest
    | _ :: rest -> loop acc rest
  in
  loop [] lines

let max_lines = 3
let within_line_budget bullet = List.count bullet.lines ~f:(Fn.non is_blank) <= max_lines
let development_repo = "lukstafi/ocannl-staging"

(** A digit-carrying occurrence of a citation form. Bare "gh-ocannl-" or "PR #" prose with no number
    behind it is not a citation, which is why the digit is part of the question; and a `PR #NNN` is
    a citation only alongside the development repository that numbers it, since a bare `PR #123`
    names a pull request in whatever repository the reader happens to think of. *)
let cites_record bullet =
  let text = String.concat ~sep:" " bullet.lines in
  let contains pattern = Option.is_some (String.substr_index text ~pattern) in
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
  followed_by_digit "gh-ocannl-" || (contains development_repo && followed_by_digit "PR #")

let opening bullet =
  let text = String.strip bullet.first_line in
  if String.length text <= 78 then text else String.prefix text 78 ^ "..."

let report label offenders =
  List.iter offenders ~f:(fun bullet -> eprintf "  %s: %s\n" label (opening bullet))

(* The synthetic controls: the same predicates over text built to break them. Without these, a
   checker that stopped deciding -- a section finder that finds nothing, a citation reader that
   answers true for everything, an item reader that ends at the first blank line -- passes over a
   clean changelog and the golden says so in green. Each control below is a shape a review finding
   named: the two-paragraph entry and the unqualified `PR #NNN` are findings 3916346082 and
   3916346095 on lukstafi/ocannl-staging PR #603. *)
let control_section =
  [
    "";
    "### Added";
    "";
    "- A compliant bullet, one line, citing (gh-ocannl-807).";
    "- A bullet citing the development repository rather than an issue";
    "  (`lukstafi/ocannl-staging` PR #601).";
    "- A bullet whose continuation lines carry it past the budget, line two of the four";
    "  it is going to take, line three of the four it is going to take, and line four";
    "  which is where it exceeds what the convention allows for one entry, and it does";
    "  cite (gh-ocannl-807) so only the length rule can flag it.";
    "- A bullet of three lines, the second of them here, and the third of them here,";
    "  which then continues after a blank line into a second indented paragraph";
    "";
    "  that carries the fourth and fifth lines past the budget, and where the whole";
    "  citation (gh-ocannl-807) sits, so a reader that stopped at the blank line sees";
    "  neither the length nor the citation.";
    "- A bullet naming somebody else's pull request, Foo PR #123, and no OCANNL record.";
  ]

(* Two Unreleased anchors, the shape a merge leaves behind (finding 3916346112): the second one's
   bullets are what a first-match reader never sees. *)
let control_duplicate_headings =
  [ "# Changelog"; ""; "## [Unreleased]"; ""; "- one (gh-ocannl-807)."; ""; "## [Unreleased]"; "" ]

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
  print_endline
    "The `## [Unreleased]` section of CHANGES.md against the editorial convention stated in\n\
     docs/agent-notes/conventions.md (gh-ocannl-807). Released sections are history and are not\n\
     read; the counts scanned go to stderr, since a tally in a golden moves on every editorial\n\
     pass (gh-ocannl-665).\n";
  let headings = List.length (unreleased_headings lines) in
  if headings <> 1 then
    eprintf "  CHANGES.md carries %d `%s` headings\n" headings unreleased_heading;
  p "CHANGES.md carries exactly one `## [Unreleased]` heading" (headings = 1);
  let section = Option.value (unreleased_section lines) ~default:[] in
  let bullets = bullets_of section in
  eprintf "Read %d lines of `## [Unreleased]`, %d top-level bullets.\n" (List.length section)
    (List.length bullets);
  report "over three lines" (List.filter bullets ~f:(Fn.non within_line_budget));
  p_all "every Unreleased bullet is at most three lines" bullets ~f:within_line_budget;
  report "no gh-ocannl-NNN or `lukstafi/ocannl-staging` PR #NNN citation"
    (List.filter bullets ~f:(Fn.non cites_record));
  p_all "every Unreleased bullet cites gh-ocannl-NNN or a staging PR #NNN" bullets ~f:cites_record;
  let control = bullets_of control_section in
  p "the section reader finds the five synthetic control bullets" (List.length control = 5);
  p_exists "the length rule flags a synthetic four-line bullet" control
    ~f:(Fn.non within_line_budget);
  p_exists "the length rule flags a bullet whose fourth line follows a blank one" control
    ~f:(fun bullet -> (not (within_line_budget bullet)) && List.exists bullet.lines ~f:is_blank);
  p_exists "the citation rule flags a synthetic uncited bullet" control ~f:(Fn.non cites_record);
  p_exists "the citation rule flags an unqualified `PR #NNN`" control ~f:(fun bullet ->
      (not (cites_record bullet))
      && String.is_substring (String.concat ~sep:" " bullet.lines) ~substring:"PR #");
  p_exists "both rules pass a bullet citing a staging PR" control ~f:(fun bullet ->
      within_line_budget bullet && cites_record bullet
      && String.is_substring (String.concat ~sep:" " bullet.lines) ~substring:development_repo);
  p_exists "both rules pass a synthetic compliant bullet" control ~f:(fun bullet ->
      within_line_budget bullet && cites_record bullet);
  p "the section reader refuses a changelog with two Unreleased headings"
    (Option.is_none (unreleased_section control_duplicate_headings)
    && List.length (unreleased_headings control_duplicate_headings) = 2)
