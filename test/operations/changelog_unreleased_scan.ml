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

    {1 Recognizers that decide rather than approximate}

    Everything below has one failure mode: a recognizer loose enough to let a bullet out of the
    scan's population, or an extent short enough to hide half of one, passes silently — the section
    keeps its other bullets, {!Verdict.p_all}'s population guard stays satisfied, and the golden is
    green. So each of the four is pinned, and each has a control (all four were review findings on
    lukstafi/ocannl-staging PR #603):

    - The ANCHOR is the exact line [## [Unreleased]]. A prefix match calls [## [Unreleased] (draft)]
      the anchor and a historical [## [Unreleased] old] a duplicate; an exact match refuses the
      first for lack of an anchor and ignores the second, which is what the shared anchor means.
      There has to be exactly one: two of them put bullets under a heading a first-match reader
      never reaches.
    - A BULLET is any top-level unordered item — [- ], [* ] or [+ ]. Ignoring the markers an
      editorial pass does not usually reach for would drop those entries out of the population
      rather than fail; a top-level ORDERED item is refused outright by a claim of its own, since
      nothing here knows what its extent means.
    - An ITEM'S EXTENT follows Markdown, not the common case: it takes lazy continuation lines (an
      unindented prose line inside the same paragraph) and stays open across a blank line when an
      indented block follows, closing at a blank run that ends the entry, at a heading, and at the
      next top-level item. The budget counts non-blank lines — the separator between two paragraphs
      of one entry is not a line of it.
    - A PR CITATION is one construct: the development repository immediately qualifying the number
      ([`lukstafi/ocannl-staging` PR #601], or [lukstafi/ocannl-staging#601]). Two independent
      substring hits let "development happens in `lukstafi/ocannl-staging`; dependency Foo PR #123"
      qualify another project's pull request.

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
(** A top-level bullet: the physical lines of one item, first line first, blank separators included
    where they sit between two blocks of the same item. *)

let is_blank line = String.is_empty (String.strip line)

(** The level of an ATX heading — the run of [#] at column 0, closed by whitespace or the end of the
    line — and [None] for anything else. A prefix test cannot answer this: [### Added] and
    [#### Security] are both subsections of the entry they sit under, while [## [1.0.1]] ends it. *)
let heading_level line =
  let hashes =
    let rec run i = if i < String.length line && Char.equal line.[i] '#' then run (i + 1) else i in
    run 0
  in
  if hashes > 0 && (hashes = String.length line || Char.is_whitespace line.[hashes]) then
    Some hashes
  else None

let is_heading line = Option.is_some (heading_level line)

(** A top-level list marker: [-], [*] or [+] at column 0, closed by whitespace or the end of the
    line. The separator is any whitespace, and any amount of it — [-  two spaces] and a tab-indented
    entry are ordinary Markdown items, and one omitted from the population is one whose violations
    the scan never sees. A marker alone on its line is an empty item, which is still an item. *)
let marker_chars = [ '-'; '*'; '+' ]

let opens_item line ~markers =
  (not (String.is_empty line))
  && List.mem markers line.[0] ~equal:Char.equal
  && (String.length line = 1 || Char.is_whitespace line.[1])

let is_top_level_bullet line = opens_item line ~markers:marker_chars

(** A top-level ORDERED item: digits at column 0 followed by [.] or [)], closed the same way.
    Recognized only to refuse it — an ordered changelog entry is not a shape this scan knows how to
    read, and skipping it would drop it out of the population silently. *)
let is_top_level_ordered line =
  let digits =
    let rec run i = if i < String.length line && Char.is_digit line.[i] then run (i + 1) else i in
    run 0
  in
  digits > 0 && opens_item (String.drop_prefix line digits) ~markers:[ '.'; ')' ]

(** An indented, non-blank line: the continuation of whatever item is open, in any position. Any
    leading whitespace indents, tabs included. *)
let is_indented line = (not (is_blank line)) && Char.is_whitespace line.[0]

(** A lazy continuation: an unindented prose line inside an open paragraph. Markdown keeps it in the
    item; only a structural boundary — a blank line, a heading, the next top-level item — closes it.
*)
let is_lazy_continuation line =
  (not (is_blank line))
  && (not (is_heading line))
  && (not (is_top_level_bullet line))
  && not (is_top_level_ordered line)

let unreleased_heading = "## [Unreleased]"
let is_unreleased_heading line = String.equal (String.strip line) unreleased_heading
let unreleased_headings lines = List.filter lines ~f:is_unreleased_heading
let ends_section line = match heading_level line with Some level -> level <= 2 | None -> false

(** The lines under `## [Unreleased]`, to the next heading of level 1 or 2, or the end of file.
    Subheadings (`### Added`, `#### Security`) stay in: they separate bullets, they never continue
    one, and one nested deeper than the pass happened to use is no boundary. Returns [None] unless
    there is exactly ONE Unreleased anchor — with two, a first-match reader silently stops looking
    before the second one's bullets. *)
let unreleased_section lines =
  match unreleased_headings lines with
  | [ _ ] ->
      let rec find = function
        | [] -> None
        | line :: rest when is_unreleased_heading line ->
            Some (List.take_while rest ~f:(Fn.non ends_section))
        | _ :: rest -> find rest
      in
      find lines
  | _ -> None

(** The lines of one open item, and what is left after it. [lazy_ok] says whether a paragraph is
    open, so an unindented prose line continues the item; a blank run belongs to the item only when
    an indented line follows it, which is the second-paragraph shape. A blank run at the end of a
    section, or one followed by a heading or the next item, closes the item and is not consumed. *)
let rec item_lines ~lazy_ok acc rest =
  match rest with
  | line :: tail when is_indented line -> item_lines ~lazy_ok:true (line :: acc) tail
  | line :: tail when lazy_ok && is_lazy_continuation line ->
      item_lines ~lazy_ok:true (line :: acc) tail
  | line :: _ when is_blank line -> (
      let blanks, after = List.split_while rest ~f:is_blank in
      match after with
      | next :: _ when is_indented next ->
          item_lines ~lazy_ok:true (List.rev_append blanks acc) after
      | _ -> (List.rev acc, rest))
  | _ -> (List.rev acc, rest)

(** The top-level bullets of a section. *)
let bullets_of lines =
  let rec loop acc = function
    | [] -> List.rev acc
    | line :: rest when is_top_level_bullet line ->
        let continuation, rest = item_lines ~lazy_ok:true [] rest in
        loop ({ first_line = line; lines = line :: continuation } :: acc) rest
    | _ :: rest -> loop acc rest
  in
  loop [] lines

let max_lines = 3
let within_line_budget bullet = List.count bullet.lines ~f:(Fn.non is_blank) <= max_lines
let development_repo = "lukstafi/ocannl-staging"

(** Whether [text] carries [prefix] with a digit right behind it. Bare "gh-ocannl-" prose with no
    number is not a citation, which is why the digit is part of the question. *)
let followed_by_digit text ~prefix =
  let plen = String.length prefix and tlen = String.length text in
  let rec at i =
    if i + plen >= tlen then false
    else if String.equal (String.sub text ~pos:i ~len:plen) prefix && Char.is_digit text.[i + plen]
    then true
    else at (i + 1)
  in
  at 0

(** Whether [text] cites a pull request of the development repository AS ONE CONSTRUCT: the
    repository name immediately qualifying the number, across an optional closing backtick and
    whitespace ([`lukstafi/ocannl-staging` PR #601]) or directly ([lukstafi/ocannl-staging#601]).
    Two independent substring hits would let a sentence naming the repository qualify any other
    project's `PR #NNN`. *)
let cites_staging_pr text =
  let repo = development_repo in
  let rlen = String.length repo and tlen = String.length text in
  let qualifies j =
    if j < tlen && Char.equal text.[j] '#' then j + 1 < tlen && Char.is_digit text.[j + 1]
    else
      let j = if j < tlen && Char.equal text.[j] '`' then j + 1 else j in
      let rec skip k = if k < tlen && Char.is_whitespace text.[k] then skip (k + 1) else k in
      let k = skip j in
      let pr = "PR #" in
      k + String.length pr < tlen
      && String.equal (String.sub text ~pos:k ~len:(String.length pr)) pr
      && Char.is_digit text.[k + String.length pr]
  in
  let rec at i =
    if i + rlen > tlen then false
    else if String.equal (String.sub text ~pos:i ~len:rlen) repo && qualifies (i + rlen) then true
    else at (i + 1)
  in
  at 0

let cites_record bullet =
  let text = String.concat ~sep:" " bullet.lines in
  followed_by_digit text ~prefix:"gh-ocannl-" || cites_staging_pr text

let mentions bullet substring = String.is_substring (String.concat ~sep:" " bullet.lines) ~substring

let opening bullet =
  let text = String.strip bullet.first_line in
  if String.length text <= 78 then text else String.prefix text 78 ^ "..."

let report label offenders =
  List.iter offenders ~f:(fun bullet -> eprintf "  %s: %s\n" label (opening bullet))

(* The synthetic controls: the same recognizers over text built to break them. Without these, one
   that stopped deciding -- an item reader that ends at the first blank or at the first unindented
   line, a citation reader that takes two independent substring hits, a bullet reader blind to `* `
   -- passes over a clean changelog and the golden says so in green. Every entry below is a shape a
   review finding named on lukstafi/ocannl-staging PR #603. *)
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
    "- A bullet whose continuation lines are lazy rather than indented,";
    "line two of it, unindented,";
    "line three of it, and then";
    "line four, which is where the citation (gh-ocannl-807) sits as well.";
    "- A bullet naming somebody else's pull request, Foo PR #123, and no OCANNL record.";
    "- A bullet saying development happens in `lukstafi/ocannl-staging`, and separately";
    "  mentioning that dependency Foo PR #123 changed something.";
    "* An asterisk-marked entry, with no citation of any kind behind it.";
    "-  A double-spaced marker, and no citation behind it either.";
    "*\tA tab-separated marker, with no citation behind it.";
    "-";
  ]

(* Two Unreleased anchors, the shape a merge leaves behind: the second one's bullets are what a
   first-match reader never sees. *)
let control_duplicate_anchors =
  [ "# Changelog"; ""; "## [Unreleased]"; ""; "- one (gh-ocannl-807)."; ""; "## [Unreleased]"; "" ]

(* Headings that are not the shared anchor: a decorated one is no anchor at all rather than the
   anchor, and a historical variant is not a duplicate of it. *)
let control_inexact_anchors =
  [ "## [Unreleased] (draft)"; ""; "- one (gh-ocannl-807)."; ""; "## [Unreleased] old"; "" ]

let control_ordered_item =
  [ "- one (gh-ocannl-807)."; "1. An ordered top-level item."; "2) Another." ]

(* A subsection nested deeper than the pass happened to use, and the released section behind it: the
   first is no boundary, the second is (Codex P2, round 3). *)
let control_deep_subheading =
  [
    "## [Unreleased]";
    "";
    "### Added";
    "";
    "- one (gh-ocannl-807).";
    "";
    "#### Security";
    "";
    "- two, uncited and inside Unreleased.";
    "";
    "## [1.0.1] -- 2026-08-26";
    "";
    "- released, uncited, and none of this scan's business.";
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
  print_endline
    "The `## [Unreleased]` section of CHANGES.md against the editorial convention stated in\n\
     docs/agent-notes/conventions.md (gh-ocannl-807). Released sections are history and are not\n\
     read; the counts scanned go to stderr, since a tally in a golden moves on every editorial\n\
     pass (gh-ocannl-665).\n";
  let anchors = List.length (unreleased_headings lines) in
  if anchors <> 1 then eprintf "  CHANGES.md carries %d `%s` headings\n" anchors unreleased_heading;
  p "CHANGES.md carries exactly one `## [Unreleased]` heading" (anchors = 1);
  let section = Option.value (unreleased_section lines) ~default:[] in
  let bullets = bullets_of section in
  eprintf "Read %d lines of `## [Unreleased]`, %d top-level bullets.\n" (List.length section)
    (List.length bullets);
  (* Emptiness is the passing case here: an ordered top-level item has no extent this scan knows how
     to read, so it is refused rather than skipped. *)
  let ordered = List.filter section ~f:is_top_level_ordered in
  List.iter ordered ~f:(fun line -> eprintf "  ordered top-level item: %s\n" (String.strip line));
  p "every top-level list item in Unreleased is unordered" (List.is_empty ordered);
  (* Unguarded universals, deliberately, and this is the one site in the file where emptiness is a
     PASSING case: an editorial pass at release prep moves every bullet into the new released
     section, and the Unreleased section that remains is legitimately empty until the next merge. A
     population guard here would fail the release-prep build for having nothing to complain about
     (Codex P2, round 3). What the guard is normally for -- a scan that reports nothing because it
     read nothing -- is carried instead by the anchor claim above, which fails when the section
     cannot be found at all, and by the synthetic controls below, whose population is fixed,
     non-empty, and exercises both rules in both directions on every run. *)
  report "over three lines" (List.filter bullets ~f:(Fn.non within_line_budget));
  p "every Unreleased bullet is at most three lines" (List.for_all bullets ~f:within_line_budget);
  report "no gh-ocannl-NNN or `lukstafi/ocannl-staging` PR #NNN citation"
    (List.filter bullets ~f:(Fn.non cites_record));
  p "every Unreleased bullet cites gh-ocannl-NNN or a staging PR #NNN"
    (List.for_all bullets ~f:cites_record);
  let control = bullets_of control_section in
  p "the section reader finds the eleven synthetic control bullets" (List.length control = 11);
  p_exists "the length rule flags a synthetic four-line bullet" control
    ~f:(Fn.non within_line_budget);
  p_exists "the length rule flags a bullet whose fourth line follows a blank one" control
    ~f:(fun bullet -> (not (within_line_budget bullet)) && List.exists bullet.lines ~f:is_blank);
  p_exists "the length rule flags a bullet whose fourth line is a lazy continuation" control
    ~f:(fun bullet ->
      (not (within_line_budget bullet)) && mentions bullet "line four, which is where");
  p_exists "the bullet reader takes an asterisk-marked entry" control ~f:(fun bullet ->
      String.is_prefix bullet.first_line ~prefix:"* " && not (cites_record bullet));
  p_exists "the citation rule flags a synthetic uncited bullet" control ~f:(fun bullet ->
      (not (cites_record bullet)) && mentions bullet "no OCANNL record");
  p_exists "the citation rule flags an unqualified `PR #NNN`" control ~f:(fun bullet ->
      (not (cites_record bullet)) && mentions bullet "somebody else's pull request");
  p_exists "the citation rule flags a repository name not qualifying the PR number" control
    ~f:(fun bullet ->
      (not (cites_record bullet))
      && mentions bullet development_repo && mentions bullet "Foo PR #123");
  p_exists "both rules pass a bullet citing a staging PR" control ~f:(fun bullet ->
      within_line_budget bullet && cites_record bullet && mentions bullet development_repo);
  p_exists "both rules pass a synthetic compliant bullet" control ~f:(fun bullet ->
      within_line_budget bullet && cites_record bullet);
  p "the section reader refuses a changelog with two Unreleased anchors"
    (Option.is_none (unreleased_section control_duplicate_anchors)
    && List.length (unreleased_headings control_duplicate_anchors) = 2);
  p "only the exact `## [Unreleased]` line counts as the anchor"
    (List.is_empty (unreleased_headings control_inexact_anchors)
    && Option.is_none (unreleased_section control_inexact_anchors));
  p_exists "the ordered-item refusal sees a top-level ordered item" control_ordered_item
    ~f:is_top_level_ordered;
  p_all "the bullet reader takes a marker separated by any whitespace, or none"
    [ "-  two spaces"; "*\ttab"; "+ one space"; "-" ]
    ~f:is_top_level_bullet;
  p_none "the bullet reader takes neither a rule nor emphasis for a bullet"
    [ "---"; "***"; "*emphasis* opens this line"; "-1 is a number" ]
    ~f:is_top_level_bullet;
  let deep = bullets_of (Option.value (unreleased_section control_deep_subheading) ~default:[]) in
  p "a level-4 subheading stays inside Unreleased, and the released section does not"
    (List.length deep = 2
    && List.exists deep ~f:(fun bullet -> mentions bullet "inside Unreleased")
    && not (List.exists deep ~f:(fun bullet -> mentions bullet "released, uncited")))
