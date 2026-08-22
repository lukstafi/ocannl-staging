(** What {!Test_utils.Agent_notes_scan} calls a defect, on input built to break it.

    The live-tree scan next door ([agent_notes_structure]) is green whenever the notes are intact,
    which is most days — and a check that is green because it sees nothing looks exactly like a check
    that is green because everything holds. So each of the five rules is exercised here on synthetic
    notes: a violation the rule must flag, and beside it the nearest legitimate text it must NOT,
    since a rule that fires on ordinary prose gets turned off rather than obeyed.

    The fixtures are written against the shapes the real defects took (lukstafi/ocannl-staging#406):
    a bullet cut mid-sentence by a merge with its tail stranded inside a later bullet, an index hook
    naming a file that does not carry it, a table row wrapped across two physical lines. Each case
    names the findings the scan should produce, as ["<rule> @ <where>"], in order. *)

open Base
open Stdio
module Notes = Test_utils.Agent_notes_scan

(* Failures go through [Verdict], so that a regression exits nonzero instead of being `dune promote`d
   into the golden as the expected output (gh-ocannl-601). *)
let fail fmt = Printf.ksprintf Verdict.fail fmt

let render (f : Notes.finding) = Printf.sprintf "%s @ %s" f.Notes.rule f.Notes.where

(* ------------------------------------------------------------------ *)
(* Rule 1: bullet integrity, over a single file *)
(* ------------------------------------------------------------------ *)

(* Each case is a file body and the findings expected from [check_structure] alone. Line 1 of every
   body is the heading, so the reported line numbers read off the fixture directly. *)
let structure_cases =
  [
    ( "a flat list of finished bullets",
      "# Title\n\n- One fact, stated.\n- Another, with a continuation line that\n  wraps and then \
       ends.\n",
      [] );
    (* The shape a merge leaves behind: an incoming bullet inserted after the FIRST line of its hunk
       context instead of the last, so the bullet above it stops mid-word. The tail lands inside a
       later bullet, where nothing marks it -- the truncation above is the decidable half. *)
    ( "a bullet a merge cut in half",
      "# Title\n\n- A fact whose closing sentence stops mid-senten\n- An unrelated incoming \
       bullet, complete in itself.\n  ce and the rest of it, stranded here.\n",
      [ "bullet-integrity @ f.md:3" ] );
    ("a bullet ending in a colon", "# Title\n\n- The three of them are:\n", []);
    ( "a bullet ending in a parenthesis after the period",
      "# Title\n\n- A fact with its issue named (gh-ocannl-406.)\n",
      [] );
    ("a bullet ending in bold", "# Title\n\n- A fact that ends **emphatically.**\n", []);
    ("a bullet ending in a code span", "# Title\n\n- A fact naming `Ops.promote_prec`.\n", []);
    ( "a bullet ending in an identifier and no punctuation",
      "# Title\n\n- A fact that trails off into `Ops.promote_prec`\n",
      [ "bullet-integrity @ f.md:3" ] );
    (* One nesting level is what the notes use; its continuations sit two deeper again. *)
    ( "a nested bullet with its own continuations",
      "# Title\n\n- A fact with parts, as follows:\n  - the first part, which\n    wraps across a \
       line.\n  - the second part.\n- The next fact.\n",
      [] );
    (* Findings come out in document order, parent before child, however the list nests. *)
    ( "a nested list where parent and child are both cut short",
      "# Title\n\n- A parent that stops mid-senten\n  - a child that stops too\n- A finished \
       fact.\n",
      [ "bullet-integrity @ f.md:3"; "bullet-integrity @ f.md:4" ] );
    ( "an indented line at the parent's depth while a nested bullet is open",
      "# Title\n\n- A fact with parts:\n  - the first part.\n  more of the first fact.\n",
      [ "bullet-integrity @ f.md:5" ] );
    ( "an indented line continuing nothing",
      "# Title\n\n- A finished fact.\n\n  a line indented under a blank one.\n",
      [ "bullet-integrity @ f.md:5" ] );
    ( "a continuation indented past its bullet",
      "# Title\n\n- A fact that\n    continues four deep.\n",
      [ "bullet-integrity @ f.md:3"; "bullet-integrity @ f.md:4" ] );
    ( "a nested bullet under no bullet at all",
      "# Title\n\nPreamble prose.\n\n  - a bullet indented under nothing.\n",
      [ "bullet-integrity @ f.md:5" ] );
    ( "a bullet written with a star",
      "# Title\n\n* A fact written with the other marker.\n",
      [ "bullet-integrity @ f.md:3" ] );
    ("an empty bullet", "# Title\n\n-\n- A real fact.\n", [ "bullet-integrity @ f.md:3" ]);
    ( "a tab in the indentation",
      "# Title\n\n- A fact that\n\tcontinues after a tab.\n",
      [ "bullet-integrity @ f.md:3"; "bullet-integrity @ f.md:4" ] );
    (* Prose paragraphs at column zero are the abstract and the backlink, and they close the list
       above them rather than continuing it. *)
    (* Round 1's family: four constructs that fell through to "prose", which is not checked -- so
       the text inside them got no termination, no repetition and no table rule at all, and the
       golden stayed green over it. Recognition is closed now, and the fallthrough is a finding. *)
    ( "an ordered item at column zero",
      "# Title\n\n1. A fact written as an ordered item.\n",
      [ "bullet-integrity @ f.md:3" ] );
    ( "an ordered item written with a parenthesis",
      "# Title\n\n1) A fact written as an ordered item.\n",
      [ "bullet-integrity @ f.md:3" ] );
    (* A line wholly inside a code span or a comment is an example. Parsing it invents findings about
       text nobody sees -- and can report the closing delimiter as an illegal lazy continuation. *)
    ( "a bullet-looking line inside a multiline code span",
      "# Title\n\n- A fact showing the shape `\n  - not a bullet\n  ` in passing.\n",
      [] );
    ( "a bullet-looking line inside an HTML comment",
      "# Title\n\n<!-- draft:\n- not a bullet yet\n-->\n\n- A fact.\n",
      [ "bullet-integrity @ f.md:3" ] );
    ( "a table-looking line inside a multiline code span",
      "# Title\n\n- A fact showing the shape `\n  | not | a table |\n  ` in passing.\n",
      [] );
    ( "an ordered item written with a tab",
      "# Title\n\n1.\tA fact written as an ordered item.\n",
      [ "bullet-integrity @ f.md:3" ] );
    ( "a star marker written with a tab",
      "# Title\n\n*\tA fact written with the other marker.\n",
      [ "bullet-integrity @ f.md:3" ] );
    ( "a dash marker written with a tab",
      "# Title\n\n-\tA fact written with a tab.\n",
      [ "bullet-integrity @ f.md:3" ] );
    ( "an ordered item indented under a bullet",
      "# Title\n\n- A fact with parts:\n  1. the first part.\n",
      [ "bullet-integrity @ f.md:4" ] );
    ( "a number that merely opens a sentence",
      "# Title\n\n- 296 stanzas were placed against a floor of 295.\n",
      [] );
    ( "a fenced block, whose contents are not bullets",
      "# Title\n\n- A fact.\n\n```\n- not a bullet, an example\n| not | a table |\n```\n\n-        Another fact.\n",
      [ "bullet-integrity @ f.md:5" ] );
    ( "a fence that is never closed",
      "# Title\n\n- A fact.\n\n~~~\n- not a bullet\n",
      [ "bullet-integrity @ f.md:5"; "bullet-integrity @ f.md:7" ] );
    ("a block quote", "# Title\n\n> Quoted prose.\n", [ "bullet-integrity @ f.md:3" ]);
    ( "a code span the file never closes",
      "# Title\n\n- A fact naming `Ops.promote_prec.",
      [ "bullet-integrity @ f.md:3" ] );
    ( "an HTML comment the file never closes",
      "# Title\n\nProse <!-- and then nothing.\n\n- A fact.\n",
      [ "bullet-integrity @ f.md:3"; "bullet-integrity @ f.md:6" ] );
    (* Markdown honours a quote marker at every depth, so nested under a bullet this is a quote
       inside the list item -- not part of the bullet's prose, and checked by nothing if folded in. *)
    ( "a block quote nested under a bullet",
      "# Title\n\n- A fact:\n  > Quoted guidance.\n",
      [ "bullet-integrity @ f.md:4" ] );
    ( "an index whose rows drifted four spaces right",
      "# Title\n\n    | File | Covers |\n    | --- | --- |\n    | a | b |\n",
      [ "bullet-integrity @ f.md:3"; "bullet-integrity @ f.md:4"; "bullet-integrity @ f.md:5" ] );
    ( "prose at column zero directly under a bullet",
      "# Title\n\n- A fact follows:\nthe omitted continuation.\n",
      [ "bullet-integrity @ f.md:4" ] );
    ( "a blank line makes the same prose a paragraph",
      "# Title\n\n- A fact follows:\n\nAn ordinary paragraph.\n",
      [] );
    ( "a heading written without a space after the hashes",
      "# Title\n\n##ident_blacklist\n\n- A fact.\n",
      [ "bullet-integrity @ f.md:3" ] );
    ( "a comparison at the start of a continuation is not a quote",
      "# Title\n\n- A fact about widths\n  >= 8, and about `dune build`.\n",
      [] );
    ( "an HTML block at column zero",
      "# Title\n\n<details><summary>x</summary>\n",
      [ "bullet-integrity @ f.md:3" ] );
    ( "an angle bracket inside a continuation is ordinary text",
      "# Title\n\n- A fact about widths\n  <= 8, and about `dune build <that target>`.\n",
      [] );
    ("a thematic break", "# Title\n\n- A fact.\n\n---\n\n- Another fact.\n",
      [ "bullet-integrity @ f.md:5" ] );
    ( "a third nesting level",
      "# Title\n\n- A fact with parts:\n  - the first part, itself with parts:\n    - a third        level.\n",
      [ "bullet-integrity @ f.md:5" ] );
    ( "prose at column zero between lists",
      "# Title\n\n- A fact.\n\nA paragraph of prose.\n\n- Another fact.\n",
      [] );
    ( "a heading closes the list above it",
      "# Title\n\n- A fact.\n\n## Section\n\n- Another fact.\n",
      [] );
  ]

(* ------------------------------------------------------------------ *)
(* Rule 3: table shape *)
(* ------------------------------------------------------------------ *)

let table_cases =
  [
    ( "a well-formed table",
      "# Title\n\n| File | What it covers |\n| --- | --- |\n| a | b |\n| c | d |\n",
      [] );
    (* PR #406's third finding: an edit wrapped a row, which ends the table -- the row is truncated
       and every row below it renders as pipe-delimited prose. Both halves are visible: the row does
       not close, and the line below the block carries the rest of it. *)
    ( "a row wrapped across two physical lines",
      "# Title\n\n| File | What it covers |\n| --- | --- |\n| a | a description long enough to\n  \
       have been wrapped |\n| c | d |\n",
      [ "table-shape @ f.md:5"; "table-shape @ f.md:7" ] );
    ( "the wrapped tail below a table that itself closes",
      "# Title\n\n| File | What it covers |\n| --- | --- |\n| a | b |\nthe rest of it |\n",
      [ "table-shape @ f.md:6" ] );
    ( "a row with a cell too few",
      "# Title\n\n| File | What it covers |\n| --- | --- |\n| a |\n",
      [ "table-shape @ f.md:5" ] );
    ( "a row with a cell too many",
      "# Title\n\n| File | What it covers |\n| --- | --- |\n| a | b | c |\n",
      [ "table-shape @ f.md:5" ] );
    ( "a header with no delimiter row",
      "# Title\n\n| File | What it covers |\n| a | b |\n| c | d |\n",
      [ "table-shape @ f.md:4" ] );
    ("a header and a delimiter and no rows", "# Title\n\n| File | Covers |\n| --- | --- |\n",
      [ "table-shape @ f.md:3" ] );
    (* The notes quote pipes constantly (`=:||`, `none|cc|metal`), and a quoted one is not a cell
       separator -- a rule that thought otherwise would fire on a dozen real bullets. *)
    ( "a pipe inside a code span is not a cell separator",
      "# Title\n\n- The vocabulary is closed (`none|cc|metal`).\n",
      [] );
    ( "a pipe inside a code span below a table",
      "# Title\n\n| File | Covers |\n| --- | --- |\n| a | b |\n\n- The spelling is `a|b`.\n",
      [] );
    ( "an escaped pipe is not a cell separator either",
      "# Title\n\n| File | Covers |\n| --- | --- |\n| a \\| b | c |\n",
      [] );
    (* Four spaces make an indented code block, so an index whose rows drifted right renders as a
       code sample with no navigable link in it. Three is Markdown's limit and still a table. *)
    ( "a table indented three spaces is still a table",
      "# Title\n\n   | File | Covers |\n   | --- | --- |\n   | a | b |\n",
      [] );
    ( "a table indented four spaces is a code block",
      "# Title\n\n    | File | Covers |\n    | --- | --- |\n    | a | b |\n",
      [] );
    ( "an escaped backslash leaves the pipe separating",
      "# Title\n\n| File | Covers |\n| --- | --- |\n| a \\\\| b | c |\n",
      [ "table-shape @ f.md:5" ] );
    ( "a table example inside a code span is not a table",
      "# Title\n\n- A fact showing `\n  | File | Covers |\n  | --- | --- |\n  ` in passing.\n",
      [] );
    ( "a delimiter cell of one hyphen",
      "# Title\n\n| File | Covers |\n| - | --- |\n| a | b |\n",
      [ "table-shape @ f.md:4" ] );
    ( "a delimiter cell of two hyphens",
      "# Title\n\n| File | Covers |\n| -- | -- |\n| a | b |\n",
      [ "table-shape @ f.md:4" ] );
    ("alignment colons are still a delimiter row",
      "# Title\n\n| File | Covers |\n| :--- | ---: |\n| a | b |\n", [] );
  ]

(* ------------------------------------------------------------------ *)
(* Rules 2, 4 and 5: the index against the files it indexes *)
(* ------------------------------------------------------------------ *)

(* The backlink a file at [depth] directories below docs/ has to carry. Written as a function of the
   depth rather than as the one-level constant it used to be: the constant made the nested-file
   fixture bless a target that resolves to nothing (Codex P2, round 2), which is the sharpest way a
   test can go wrong -- it locked the defect in rather than catching it. *)
let backlink_at depth =
  let up = String.concat (List.init depth ~f:(fun _ -> "../")) in
  Printf.sprintf "Part of the agent notes; the [index](%sagent-notes.md) carries the rest.\n" up

let file_at depth body = "# A file\n\n" ^ backlink_at depth ^ "\n" ^ body

(* Every unnested fixture sits at agent-notes/<name>, one directory below the index. *)
let file body = file_at 1 body

let index rows =
  "# Agent notes\n\nAn index.\n\n| File | What it covers |\n| --- | --- |\n"
  ^ String.concat ~sep:"" (List.map rows ~f:(fun r -> r ^ "\n"))

let row name hooks = Printf.sprintf "| [%s](agent-notes/%s) | %s |" name name hooks

(* Each case is an index body, the files it should be checked against, and the expected findings. *)
let index_cases =
  [
    ( "an index whose rows all resolve",
      index [ row "a.md" "the `Widget` seam"; row "b.md" "the `Gadget` seam" ],
      [ ("agent-notes/a.md", file "- A fact about `Widget`.\n");
        ("agent-notes/b.md", file "- A fact about `Gadget`.\n") ],
      [] );
    (* PR #406's second finding, from both ends: the hook sits on the wrong row, so following the
       index leads away from the trap it names. *)
    ( "a hook the file it points at does not carry",
      index [ row "a.md" "the `Widget` seam"; row "b.md" "`fast math` and the `Gadget` seam" ],
      [ ("agent-notes/a.md", file "- A fact about `Widget`, and about fast math.\n");
        ("agent-notes/b.md", file "- A fact about `Gadget`.\n") ],
      [ "index-agreement @ agent-notes.md:8" ] );
    ( "a hook whose wording drifted from the file's",
      index [ row "a.md" "the `identifier blacklist`" ],
      [ ("agent-notes/a.md", file "- The `ident_blacklist` covers generated names.\n") ],
      [ "index-agreement @ agent-notes.md:7" ] );
    ( "a link to a file that is not there",
      index [ row "a.md" "the `Widget` seam"; row "gone.md" "something" ],
      [ ("agent-notes/a.md", file "- A fact about `Widget`.\n") ],
      [ "index-agreement @ agent-notes.md:8" ] );
    ( "link text that does not name its file",
      index [ "| [shape inference](agent-notes/a.md) | the `Widget` seam |" ],
      [ ("agent-notes/a.md", file "- A fact about `Widget`.\n") ],
      [ "index-agreement @ agent-notes.md:7" ] );
    ( "a first cell that is not a link",
      index [ "| a.md | the `Widget` seam |" ],
      [ ("agent-notes/a.md", file "- A fact about `Widget`.\n") ],
      [ "index-agreement @ agent-notes.md:7"; "reachability @ agent-notes/a.md" ] );
    ( "an anchor the file has a heading for",
      index [ "| [a.md](agent-notes/a.md#the-widget-seam) | the `Widget` seam |" ],
      [ ("agent-notes/a.md", file "## The Widget seam\n\n- A fact about `Widget`.\n") ],
      [] );
    ( "an anchor the file has no heading for",
      index [ "| [a.md](agent-notes/a.md#the-gadget-seam) | the `Widget` seam |" ],
      [ ("agent-notes/a.md", file "## The Widget seam\n\n- A fact about `Widget`.\n") ],
      [ "index-agreement @ agent-notes.md:7" ] );
    (* Reachability, which is the "hook names a file carrying none of it" failure seen from the
       other end: a file nothing links is a file no lookup will reach. *)
    (* PR #406's third finding, end to end: the wrap ends the table, so most of the index is outside
       it. Asking the hook and reachability rules about that would produce one "unreachable" line per
       file and bury the one finding that says what happened -- so they decline, loudly. *)
    ( "a wrapped row takes the rest of the index out of the table",
      "# Agent notes\n\nAn index.\n\n| File | What it covers |\n| --- | --- |\n| [a.md](agent-notes/a.md) | the\n`Widget` seam |\n| [b.md](agent-notes/b.md) | the `Gadget` seam |\n",
      [ ("agent-notes/a.md", file "- A fact about `Widget`.\n");
        ("agent-notes/b.md", file "- A fact about `Gadget`.\n") ],
      [ "index-agreement @ agent-notes.md"; "table-shape @ agent-notes.md:7";
        "table-shape @ agent-notes.md:9"; "reachability @ agent-notes.md" ] );
    ( "an anchor on an identifier heading, which GitHub keeps underscored",
      index [ "| [a.md](agent-notes/a.md#ident_blacklist) | the `ident_blacklist` |" ],
      [ ("agent-notes/a.md", file "## ident_blacklist\n\n- A fact about `ident_blacklist`.\n") ],
      [] );
    ( "the same anchor with the underscore hyphenated",
      index [ "| [a.md](agent-notes/a.md#ident-blacklist) | the `ident_blacklist` |" ],
      [ ("agent-notes/a.md", file "## ident_blacklist\n\n- A fact about `ident_blacklist`.\n") ],
      [ "index-agreement @ agent-notes.md:7" ] );
    (* A backlink has to be one a reader can FOLLOW. All three of these carry the bytes and none of
       them carries a link. *)
    ( "a backlink that is only a code span",
      index [ row "a.md" "the `Widget` seam" ],
      [ ("agent-notes/a.md",
          "# A file\n\nThe index lives at `[index](../agent-notes.md)`.\n\n- A fact about            `Widget`.\n") ],
      [ "reachability @ agent-notes/a.md" ] );
    ( "a backlink commented out",
      index [ row "a.md" "the `Widget` seam" ],
      [ ("agent-notes/a.md",
          "# A file\n\n<!-- [index](../agent-notes.md) -->\n\n- A fact about `Widget`.\n") ],
      [ "bullet-integrity @ agent-notes/a.md:3"; "reachability @ agent-notes/a.md" ] );
    ( "a backlink that is an image",
      index [ row "a.md" "the `Widget` seam" ],
      [ ("agent-notes/a.md",
          "# A file\n\n![index](../agent-notes.md)\n\n- A fact about `Widget`.\n") ],
      [ "reachability @ agent-notes/a.md" ] );
    ( "a backlink hidden in the middle of a multiline code span",
      index [ row "a.md" "the `Widget` seam" ],
      [ ("agent-notes/a.md",
          "# A file\n\nPart of an example: `\n[index](../agent-notes.md)\n`.\n\n- A fact about            `Widget`.\n") ],
      [ "reachability @ agent-notes/a.md" ] );
    ( "a backlink after a code span that closed on the line above",
      index [ row "a.md" "the `Widget` seam" ],
      [ ("agent-notes/a.md",
          "# A file\n\nPart of an example: `\nexample`\n[index](../agent-notes.md)\n\n- A fact            about `Widget`.\n") ],
      [] );
    ( "a backlink inside a multiline HTML comment",
      index [ row "a.md" "the `Widget` seam" ],
      [ ("agent-notes/a.md",
          "# A file\n\nProse <!--\n[index](../agent-notes.md)\n-->\n\n- A fact about            `Widget`.\n") ],
      [ "bullet-integrity @ agent-notes/a.md:3"; "reachability @ agent-notes/a.md" ] );
    ( "a backlink whose bracket is escaped",
      index [ row "a.md" "the `Widget` seam" ],
      [ ("agent-notes/a.md",
          "# A file\n\nWrite it \\[index](../agent-notes.md) to show the syntax.\n\n- A fact            about `Widget`.\n") ],
      [ "reachability @ agent-notes/a.md" ] );
    ( "an anchor naming a heading that only appears inside a comment",
      index [ "| [a.md](agent-notes/a.md#draft-heading) | the `Widget` seam |" ],
      [ ("agent-notes/a.md",
          file "<!--\n## Draft heading\n-->\n\n- A fact about `Widget`.\n") ],
      [ "bullet-integrity @ agent-notes/a.md:5"; "index-agreement @ agent-notes.md:7" ] );
    ( "an index row with a third cell",
      "# Agent notes\n\nAn index.\n\n| File | Covers | Owner |\n| --- | --- | --- |\n|        [a.md](agent-notes/a.md) | the `Widget` seam | me |\n",
      [ ("agent-notes/a.md", file "- A fact about `Widget`.\n") ],
      [ "index-agreement @ agent-notes.md:7"; "reachability @ agent-notes/a.md" ] );
    ( "a backlink one directory too far up",
      index [ row "a.md" "the `Widget` seam" ],
      [ ("agent-notes/a.md",
          "# A file\n\nSee the [index](../../agent-notes.md).\n\n- A fact about `Widget`.\n") ],
      [ "reachability @ agent-notes/a.md" ] );
    ( "an anchor on a heading written without a space",
      index [ "| [a.md](agent-notes/a.md#ident_blacklist) | the `ident_blacklist` |" ],
      [ ("agent-notes/a.md",
          file "##ident_blacklist\n\n- A fact about `ident_blacklist`.\n") ],
      [ "bullet-integrity @ agent-notes/a.md:5"; "index-agreement @ agent-notes.md:7" ] );
    ( "a backlink carrying an anchor is still a backlink",
      index [ row "a.md" "the `Widget` seam" ],
      [ ("agent-notes/a.md",
          "# A file\n\nPart of the [index](../agent-notes.md#index).\n\n- A fact about            `Widget`.\n") ],
      [] );
    (* The directory is flat today; a note in a subdirectory of it is judged like any other, which is
       what makes recursing over the tree safe rather than merely more thorough. *)
    ( "a note in a subdirectory, linked and backlinked from its own depth",
      index [ "| [a.md](agent-notes/sub/a.md) | the `Widget` seam |" ],
      [ ("agent-notes/sub/a.md", file_at 2 "- A fact about `Widget`.\n") ],
      [] );
    ( "a nested note carrying the one-level backlink, which resolves to nothing",
      index [ "| [a.md](agent-notes/sub/a.md) | the `Widget` seam |" ],
      [ ("agent-notes/sub/a.md", file_at 1 "- A fact about `Widget`.\n") ],
      [ "reachability @ agent-notes/sub/a.md" ] );
    ( "a note in a subdirectory, unlinked",
      index [ row "b.md" "the `Gadget` seam" ],
      [ ("agent-notes/b.md", file "- A fact about `Gadget`.\n");
        ("agent-notes/sub/a.md", file_at 2 "- A fact about `Widget`.\n") ],
      [ "reachability @ agent-notes/sub/a.md" ] );
    ( "a backlink written with a redundant ./ still resolves",
      index [ row "a.md" "the `Widget` seam" ],
      [ ("agent-notes/a.md",
          "# A file\n\nSee the [index](./../agent-notes.md).\n\n- A fact about `Widget`.\n") ],
      [] );
    ( "a file no row links",
      index [ row "a.md" "the `Widget` seam" ],
      [ ("agent-notes/a.md", file "- A fact about `Widget`.\n");
        ("agent-notes/b.md", file "- A fact about `Gadget`.\n") ],
      [ "reachability @ agent-notes/b.md" ] );
    ( "two rows for one file",
      index [ row "a.md" "the `Widget` seam"; row "a.md" "the `Widget` seam again" ],
      [ ("agent-notes/a.md", file "- A fact about `Widget`.\n") ],
      [ "reachability @ agent-notes.md:8" ] );
    ( "a file that does not link back to the index",
      index [ row "a.md" "the `Widget` seam" ],
      [ ("agent-notes/a.md", "# A file\n\n- A fact about `Widget`.\n") ],
      [ "reachability @ agent-notes/a.md" ] );
    (* Rule 5. A fact promoted into two files is a fact that will be corrected in one of them. *)
    ( "the same bullet in two files",
      index [ row "a.md" "the `Widget` seam"; row "b.md" "the `Gadget` seam" ],
      [ ("agent-notes/a.md", file "- One fact, promoted twice, about `Widget` and `Gadget`.\n");
        ("agent-notes/b.md", file "- One fact, promoted twice, about `Widget` and `Gadget`.\n") ],
      [ "no-repetition @ agent-notes/b.md:5" ] );
    ( "the same bullet re-wrapped",
      index [ row "a.md" "the `Widget` seam"; row "b.md" "the `Gadget` seam" ],
      [ ("agent-notes/a.md", file "- One fact, promoted twice, about `Widget` and `Gadget`.\n");
        ("agent-notes/b.md",
          file "- One fact, promoted twice, about `Widget`\n  and `Gadget`.\n") ],
      [ "no-repetition @ agent-notes/b.md:5" ] );
    ( "two bullets opening alike and diverging",
      index [ row "a.md" "the `Widget` seam"; row "b.md" "the `Gadget` seam" ],
      [ ("agent-notes/a.md",
          file "- The analysis pass establishes what the specializer may assume of `Widget`.\n");
        ("agent-notes/b.md",
          file "- The analysis pass establishes what the specializer may assume of `Gadget`.\n") ],
      [ "no-repetition @ agent-notes/b.md:5" ] );
    ( "two bullets that merely start with the same few words",
      index [ row "a.md" "the `Widget` seam"; row "b.md" "the `Gadget` seam" ],
      [ ("agent-notes/a.md",
          file "- The `Widget` seam is where storage meets compute, and it is narrow.\n");
        ("agent-notes/b.md",
          file "- The `Gadget` seam is where the pool meets the stream, and it is wide.\n") ],
      [] );
  ]

(* The LEXICAL layer, tested directly rather than only through the rules above it.

   Round 3 was six findings and three of them were here -- code-span pairing, backslash parity, ATX
   spacing -- each an approximation of a CommonMark rule, each feeding every rule in the scan, and
   each invisible from the rule level because a rule reads "no pipe there" and "no heading there" the
   same whether the primitive was right or lazy. A layer that everything depends on and nothing
   tests directly is where this class regenerates, so it gets its own fixtures. *)
let primitive_cases =
  [
    (* A run of N backticks is closed by the next run of exactly N. *)
    ("one span", "a `b` c", [ "2-5" ]);
    ("two spans", "`a` and `b`", [ "0-3"; "8-11" ]);
    ("a double run holds a single", "``a`b`` c", [ "0-7" ]);
    ("an unpaired run opens a span to the line's end", "a `b c", [ "2-6" ]);
    ("no backticks at all", "plain text", []);
    ("a comment is inert too", "a <!-- b --> c", [ "2-12" ]);
    ("a backtick inside a comment opens nothing", "a <!-- ` --> c", [ "2-12" ]);
    ("a comment opener inside code opens nothing", "a `<!--` b", [ "2-8" ]);
  ]

(* The lexer's state across line boundaries, which is where rounds 3 and 4 both landed. Each case is
   a whole file and the lines whose text is inert, as "<line>:<start>-<stop>". *)
let carry_cases =
  [
    ( "a span closed on the next line",
      "a `b\nc` d\n",
      [ "1:2-4"; "2:0-2" ] );
    (* Round 4's finding: the closing run being LAST on its line must clear the carry, and did not --
       the whole rest of the file went on reading as code. *)
    ( "a span closed by the last run on its line",
      "a `b\nexample`\n[index](../agent-notes.md)\n",
      [ "1:2-4"; "2:0-8" ] );
    ( "a blank line ends an unterminated span",
      "a `b\n\nc | d\n",
      [ "1:2-4" ] );
    ( "a comment spanning three lines",
      "Prose <!--\n[index](../agent-notes.md)\n--> after\n",
      [ "1:6-10"; "2:0-26"; "3:0-3" ] );
    ( "a comment survives a blank line",
      "Prose <!--\n\n--> after\n",
      [ "1:6-10"; "3:0-3" ] );
  ]

(* Escaping is decided by the PARITY of the backslash run, not by the character before the pipe. *)
let pipe_cases =
  [
    ("a plain row", "| a | b |", [ 0; 4; 8 ]);
    ("an escaped pipe is not a separator", "| a \\| b |", [ 0; 9 ]);
    ("an escaped backslash leaves the pipe live", "| a \\\\| b |", [ 0; 6; 10 ]);
    ("three backslashes escape it again", "| a \\\\\\| b |", [ 0; 11 ]);
    ("a pipe inside code is not a separator", "| `a|b` | c |", [ 0; 8; 12 ]);
  ]

(* An ATX marker is one to six hashes followed by a space or the end of the line. *)
let heading_cases =
  [
    ("a heading", "## Title", Some "Title");
    ("a heading with no space", "##Title", None);
    ("a bare marker", "#", Some "");
    ("closing hashes are decoration", "## Title ##", Some "Title");
    ("seven hashes is not a heading", "####### Title", None);
    ("an identifier heading", "## ident_blacklist", Some "ident_blacklist");
  ]

(* A relative link resolves against its file's DIRECTORY, and above the root is a distinct answer. *)
let resolve_cases =
  [
    ("one level up", "agent-notes/a.md", "../agent-notes.md", Some "agent-notes.md");
    ("two levels up from a nested note", "agent-notes/sub/a.md", "../../agent-notes.md",
      Some "agent-notes.md");
    ("one too many is above the root", "agent-notes/a.md", "../../agent-notes.md", None);
    ("a redundant ./ is dropped", "agent-notes/a.md", "./../agent-notes.md", Some "agent-notes.md");
    ("a sibling needs no ..", "agent-notes/a.md", "b.md", Some "agent-notes/b.md");
    ("an anchor is not part of the path", "agent-notes/a.md", "../agent-notes.md#x",
      Some "agent-notes.md");
  ]

(* The escape hatch is only a hatch if the key a message tells you to paste is the key that matches.
   Round 1 found it could match nothing at all: the documented format named the bullet's opening
   while the message carried its tail, and the comparison ran over the message text. Both halves are
   pinned here -- the key's shape, and that the message hands it to you verbatim. *)
let exemption_cases =
  [
    ( "the key names the file and the bullet's opening",
      "agent-notes/a.md",
      "# A file\n\n- A trailing identifier is the ending here: `Ops.promote_prec`\n",
      [ "agent-notes/a.md: A trailing identifier is the ending here: `Ops.promote_prec`" ] );
    ( "a re-wrapped bullet has the same key",
      "agent-notes/a.md",
      "# A file\n\n- A trailing identifier is the ending\n  here: `Ops.promote_prec`\n",
      [ "agent-notes/a.md: A trailing identifier is the ending here: `Ops.promote_prec`" ] );
    (* The display length is 48; these two agree for well past that and diverge only at the end, so
       a prefix key would have silenced both from one exemption. *)
    ( "two bullets agreeing past the display length keep distinct keys",
      "agent-notes/a.md",
      "# A file\n\n- The analysis pass establishes what the specializer may assume of `Widget`\n-        The analysis pass establishes what the specializer may assume of `Gadget`\n",
      [ "agent-notes/a.md: The analysis pass establishes what the specializer may assume of \
         `Widget`";
        "agent-notes/a.md: The analysis pass establishes what the specializer may assume of \
         `Gadget`" ] );
    ("a finished bullet has no finding to key", "agent-notes/a.md",
      "# A file\n\n- A fact that ends properly.\n", [] );
  ]

let () =
  let check name expected found =
    if List.equal String.equal found expected then printf "ok: %s\n" name
    else
      fail "%s -- expected [%s], found [%s]" name
        (String.concat ~sep:"; " expected)
        (String.concat ~sep:"; " found)
  in
  List.iter structure_cases ~f:(fun (name, body, expected) ->
      check ("bullets -- " ^ name) expected
        (List.map (Notes.check_structure ~file:"f.md" body) ~f:render));
  List.iter table_cases ~f:(fun (name, body, expected) ->
      check ("tables -- " ^ name) expected
        (List.map (Notes.check_tables ~file:"f.md" body) ~f:render));
  List.iter primitive_cases ~f:(fun (name, line, expected) ->
      let found =
        List.map (Notes.code_spans line) ~f:(fun (a, b) -> Printf.sprintf "%d-%d" a b)
      in
      check ("code spans -- " ^ name) expected found);
  List.iter carry_cases ~f:(fun (name, contents, expected) ->
      let found =
        List.concat_map ((Notes.inert_by_line contents).Notes.ranges) ~f:(fun (lineno, ranges) ->
            List.map ranges ~f:(fun (a, b) -> Printf.sprintf "%d:%d-%d" lineno a b))
      in
      check ("inert carry -- " ^ name) expected found);
  List.iter pipe_cases ~f:(fun (name, line, expected) ->
      let found = List.map (Notes.pipes_outside_code line) ~f:Int.to_string in
      check ("cell separators -- " ^ name) (List.map expected ~f:Int.to_string) found);
  List.iter heading_cases ~f:(fun (name, line, expected) ->
      check ("atx headings -- " ^ name) (Option.to_list expected)
        (Option.to_list (Notes.atx_heading line)));
  List.iter resolve_cases ~f:(fun (name, from_file, target, expected) ->
      check ("link resolution -- " ^ name) (Option.to_list expected)
        (Option.to_list (Notes.resolve_link ~from_file target)));
  List.iter exemption_cases ~f:(fun (name, file, body, expected) ->
      let keys =
        List.filter_map (Notes.check_structure ~file body) ~f:Notes.exemption_key
      in
      let message_offers_key =
        List.for_all (Notes.check_structure ~file body) ~f:(fun f ->
            match Notes.exemption_key f with
            | None -> true
            | Some k -> String.is_substring f.Notes.message ~substring:k)
      in
      check ("exemption -- " ^ name) expected keys;
      if not message_offers_key then
        fail "exemption -- %s: the finding's message does not print the key that would silence it"
          name);
  List.iter index_cases ~f:(fun (name, index_contents, files, expected) ->
      let _, found =
        Notes.check_all ~index_file:"agent-notes.md" ~index_contents ~files
      in
      check ("index -- " ^ name) expected (List.map found ~f:render))
