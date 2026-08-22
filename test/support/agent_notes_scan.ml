(** Reading [docs/agent-notes.md] and [docs/agent-notes/] as structure rather than as prose.

    The agent notes are the project's cross-session memory: an index table of one row per subsystem
    file, and files that are a title, a two-line abstract, a backlink, and then a flat list of
    bullets. Nothing checks that shape, so it corrupts silently — and a corrupted note is worse than
    a missing one, because every later session inherits it as fact. Three of the six review findings
    on the split that created these files (lukstafi/ocannl-staging#406) were exactly that, and each
    was decidable from the text alone:

    - a merge resolution inserted an incoming bullet after the FIRST line of its hunk context instead
      of the last, cutting a bullet's closing sentence in half and stranding the tail 39 lines later;
    - the index's [fast math] hook sat on a row whose file contains no fast-math guidance, so
      following the index led away from the trap it names — and two further hooks had drifted out of
      their file's wording entirely ([identifier blacklist] against [ident_blacklist]);
    - an edit wrapped one index row across two physical lines, which ends a Markdown table: the row
      was truncated and the five rows below it rendered as pipe-delimited prose.

    This module is the reader behind the checks for those, plus the two the wave added: that no file
    is unreachable from the index, and that no bullet is repeated across files. It is pure over
    strings — a file is a [(name, contents)] pair, nothing is opened here — so the negative controls
    in [test/operations/agent_notes_scan_cases.ml] exercise the same functions the live-tree scan in
    [test/operations/agent_notes_structure.ml] runs over the repository.

    {1 The five rules}

    Each is stated as the thing that must be TRUE, and each finding names the rule that failed.

    - {b bullet-integrity}: the list structure parses under one unambiguous reading, and no bullet's
      text is cut. Markers are ["- "] at an even indentation; a continuation line sits at exactly the
      innermost open bullet's indentation plus two; a bullet's text — its start line and its
      continuation lines joined by single spaces — ends in sentence-terminating punctuation. The last
      clause is the one that catches a splice: a bullet a merge cut in half ends mid-word, and the
      tail it stranded lands inside some other bullet. See {!bullet_text_is_terminated} for the exact
      predicate and its escape.
    - {b index-agreement}: every index row is [| \[<basename>\](agent-notes/<basename>) | prose with
      `hooks` |], the target is a file this scan was handed, the link text is that file's basename,
      an anchor (if the link carries one) names a heading the file has, and every backticked hook in
      the second cell occurs verbatim in the target. The index is what a lookup greps before opening
      anything, so a hook absent from its target is a dead end even when the file is right.
    - {b table-shape}: every table block — a maximal run of lines whose trimmed form starts with a
      pipe — is a table: at least a header, a delimiter row and one data row; every line closes with
      an unescaped pipe; all lines carry the same number of cells; and the line after the block, if
      non-blank, has no pipe outside inline code, which is what a wrapped row's tail looks like. This
      is checked BEFORE index-agreement consumes the rows, because a wrapped row otherwise drops
      silently out of the set being checked — which is how the wrap survived the round that made it.
    - {b reachability}: every file handed to this scan is the target of exactly one index row, and
      every file carries the backlink to the index. An orphan is precisely the "the hook names a file
      carrying none of it" failure seen from the other end.
    - {b no-repetition}: no two bullets in the notes share their whitespace-normalized text, and no
      two share their first {!near_duplicate_prefix} characters case-insensitively. A fact promoted
      twice is a fact that will be updated once.

    {1 What it deliberately does not read}

    Prose. Whether a bullet is TRUE of the code is not decidable here and is not attempted; the
    claim is only that what someone wrote is intact and reachable. Nor is Markdown implemented: this
    reads the dialect the notes are written in — flat ["- "] lists, one nesting level, one table in
    the index — and reports anything outside it rather than guessing. A note that wants a fenced code
    block or a setext heading has to teach this module about it first, which is the intended cost.

    {1 Inline code spans}

    Pipes and backticks matter to two rules, and the notes are full of both inside code spans
    ([`=:||`], [`none|cc|…`], [`[%expect {|…|}]`]). {!pipes_outside_code} and {!code_spans} are the
    one place that distinction is decided: a backtick run opens a span that the next run of the SAME
    length closes, per CommonMark, so ["``a|b``"] is one span and the pipe inside it is not a cell
    separator. An unterminated run is not a span and its content stays outside. *)

open Base

(** A structural defect: which rule it broke, where, and what is wrong. [where] is
    ["<file>:<line>"], or ["<file>"] for a whole-file finding. *)
type finding = {
  rule : string;
  file : string;  (** The notes file, apart from the line, so a consumer never re-parses [where]. *)
  where : string;
  message : string;
  subject : string option;
      (** The thing the finding is ABOUT, kept apart from the prose that describes it: for a bullet,
          its opening. An exemption names a bullet, and a message is free to be reworded, so matching
          an exemption against the message would break the moment the wording improved -- and did:
          the key format documented one thing while the message carried the bullet's TAIL (Codex P2,
          round 1). Consumers compare against [where]'s file and this. *)
}

(** One bullet, joined: [text] is the start line's content and every continuation line, separated by
    single spaces, with the ["- "] marker removed. [line] is the start line, 1-based. *)
type bullet = { file : string; line : int; indent : int; text : string }

let rule_bullet_integrity = "bullet-integrity"
let rule_index_agreement = "index-agreement"
let rule_table_shape = "table-shape"
let rule_reachability = "reachability"
let rule_no_repetition = "no-repetition"

(** All five, in the order the live-tree scan reports them. *)
let rules =
  [
    rule_bullet_integrity;
    rule_index_agreement;
    rule_table_shape;
    rule_reachability;
    rule_no_repetition;
  ]

let finding ?subject ~file ~line ~rule message =
  { rule; file; where = Printf.sprintf "%s:%d" file line; message; subject }

let file_finding ?subject ~file ~rule message = { rule; file; where = file; message; subject }

(** How an exemption or a report names the bullet a finding is about: the file it is in and the
    bullet's opening, which is what a person can copy out of the message and paste into a list. *)
let subject_key ~file ~subject = Printf.sprintf "%s: %s" file subject

(** The exemption key a finding would be silenced by, for the findings that can be exempted at all.
    Built from the finding's own structured fields, so the key a message tells you to paste is
    exactly the key that matches — the two cannot drift apart. *)
let exemption_key f =
  Option.map f.subject ~f:(fun subject -> subject_key ~file:f.file ~subject)

(* ------------------------------------------------------------------ *)
(* Lines, indentation, inline code *)
(* ------------------------------------------------------------------ *)

(** Physical lines, 1-based, with a trailing CR dropped: a golden promoted on Windows and a file
    edited there both arrive CRLF, and an indentation or terminator rule that sees the CR as content
    would fail on the same text that passes elsewhere. *)
let lines contents =
  String.split ~on:'\n' contents
  |> List.map ~f:(fun l ->
         match String.chop_suffix l ~suffix:"\r" with Some l -> l | None -> l)
  |> List.mapi ~f:(fun i l -> (i + 1, l))

let indent_of line =
  let n = String.length line in
  let rec go i = if i < n && Char.equal line.[i] ' ' then go (i + 1) else i in
  go 0

let has_leading_tab line =
  match String.lfindi line ~f:(fun _ c -> not (Char.equal c ' ')) with
  | Some i -> Char.equal line.[i] '\t'
  | None -> false

let is_blank line = String.is_empty (String.strip line)

(** Half-open [(start, stop)] character ranges of the inline code spans of a line, CommonMark's
    rule: a run of N backticks opens a span that the next run of exactly N closes, and a run with no
    partner opens nothing. The ranges include the delimiters. *)
let code_spans line =
  let n = String.length line in
  let rec runs i acc =
    if i >= n then List.rev acc
    else if Char.equal line.[i] '`' then (
      let j = ref i in
      while !j < n && Char.equal line.[!j] '`' do
        Int.incr j
      done;
      runs !j ((i, !j - i) :: acc))
    else runs (i + 1) acc
  in
  let runs = runs 0 [] in
  let rec pair runs acc =
    match runs with
    | [] -> List.rev acc
    | (start, len) :: rest -> (
        match List.findi rest ~f:(fun _ (_, l) -> l = len) with
        | None -> pair rest acc
        | Some (idx, (close_at, _)) ->
            pair (List.drop rest (idx + 1)) ((start, close_at + len) :: acc))
  in
  pair runs []

let in_any_span spans i =
  List.exists spans ~f:(fun (start, stop) -> start <= i && i < stop)

(** Positions of the ['|'] characters that separate table cells: outside inline code, and not
    backslash-escaped. *)
let pipes_outside_code line =
  let spans = code_spans line in
  String.foldi line ~init:[] ~f:(fun i acc c ->
      if
        Char.equal c '|'
        && (not (in_any_span spans i))
        && not (i > 0 && Char.equal line.[i - 1] '\\')
      then i :: acc
      else acc)
  |> List.rev

(** The cells of a table row: the text between the separating pipes, trimmed. A well-formed row
    starts and ends with one, so the empty pieces outside them are dropped. Returns [None] for a
    line that does not both start and end with a separating pipe — the shape a wrapped row takes. *)
let row_cells line =
  let line = String.strip line in
  match pipes_outside_code line with
  | [] -> None
  | first :: _ as pipes ->
      let last = List.last_exn pipes in
      if first <> 0 || last <> String.length line - 1 || List.length pipes < 2 then None
      else
        let rec cut = function
          | a :: (b :: _ as rest) -> String.strip (String.sub line ~pos:(a + 1) ~len:(b - a - 1)) :: cut rest
          | _ -> []
        in
        Some (cut pipes)

let is_table_line line = String.is_prefix (String.strip line) ~prefix:"|"

(** GitHub wants at least THREE hyphens in a delimiter cell. A cell shortened to ["-"] or ["--"]
    stops the block rendering as a table while every other property of it still holds, so accepting
    "nonempty and all dashes" would leave the rule green over a table that is no longer one (Codex
    P2, round 1). *)
let delimiter_min_hyphens = 3

let is_delimiter_row cells =
  (not (List.is_empty cells))
  && List.for_all cells ~f:(fun c ->
         let c = String.strip c in
         let c = Option.value (String.chop_prefix c ~prefix:":") ~default:c in
         let c = Option.value (String.chop_suffix c ~suffix:":") ~default:c in
         String.length c >= delimiter_min_hyphens && String.for_all c ~f:(Char.equal '-'))

(* ------------------------------------------------------------------ *)
(* Rule 1: bullet integrity *)
(* ------------------------------------------------------------------ *)

(** Trailing characters stripped before looking for the terminator, so that ["…done.**"], ["…see
    `x`."] and ["…(gh-ocannl-665)."] all read as terminated. *)
let closing_markup = [ '`'; ')'; ']'; '}'; '"'; '\''; '*'; '_' ]

let terminators = [ '.'; '!'; '?'; ':'; ';' ]

(** Whether a bullet's joined text ends a sentence. This is the whole of the "not cut in half" rule,
    and the corpus it was written against ends 175 of 177 bullets with a period, one with a colon
    and one with [".)"], so the predicate costs nothing to satisfy on purpose and is broken only by
    accident.

    The escape hatch for a bullet that genuinely ends without punctuation is to give it some — that
    is the cheap and usually right fix. The other, for a bullet whose ending is load-bearing (a
    trailing identifier, a table, a deliberate ellipsis), is the named exemption list in
    [agent_notes_structure.ml], where an entry has to name the bullet and say why. *)
let bullet_text_is_terminated text =
  let text = String.rstrip text in
  let rec strip s =
    match String.length s with
    | 0 -> s
    | n -> if List.mem closing_markup s.[n - 1] ~equal:Char.equal then strip (String.prefix s (n - 1)) else s
  in
  let s = strip text in
  (not (String.is_empty s)) && List.mem terminators s.[String.length s - 1] ~equal:Char.equal

(** Whitespace collapsed to single spaces: the one normal form the repetition rule compares in and
    the one a bullet is named by, so a bullet re-wrapped across different lines is the same bullet to
    both. *)
let normalize text =
  String.split_on_chars text ~on:[ ' '; '\t' ]
  |> List.filter ~f:(fun s -> not (String.is_empty s))
  |> String.concat ~sep:" "

(** How many characters of a bullet name it. Long enough to be unambiguous among 179, short enough
    to paste into an exemption list by hand. *)
let subject_length = 48

(** How a bullet is NAMED, in a finding and in an exemption: its OPENING. The opening rather than
    the tail, because that is the half a person recognises and the half that stays put while the
    bullet is edited -- and because an exemption keyed on the tail of an unterminated bullet would
    stop matching the moment someone finished the sentence, which is the edit it exists to survive.
*)
let bullet_subject (b : bullet) = String.prefix (normalize b.text) subject_length

(** {2 The closed dialect}

    The module's claim is that it reads the notes' dialect and REPORTS anything outside it rather
    than guessing. Round 1 of review found four separate places where the code did the opposite and
    silently skipped instead — an ordered-list item, a fenced block's contents, an HTML block, a
    thematic break — each of which drops real text out of every rule while the golden stays green.
    They are one defect, not four: an unrecognised line fell through to "prose", and prose is not
    checked. So recognition is a closed vocabulary now, and the fallthrough is a finding.

    Adding a construct to the notes therefore means teaching this function about it first, which is
    the intended cost: a note that wants a fenced example has to say what the bullet rules mean
    inside one. *)

(** A list marker at any indentation that is not this scan's ["- "]. An ordered item is the one that
    bit: at column zero it read as prose (so its text got no termination or repetition check at all),
    and indented it folded into its parent's continuation (Codex P2, round 1). *)
let foreign_list_marker stripped =
  let ordered =
    match String.lfindi stripped ~f:(fun _ c -> not (Char.is_digit c)) with
    | Some i when i > 0 && i + 1 < String.length stripped ->
        let sep = stripped.[i] and after = stripped.[i + 1] in
        if (Char.equal sep '.' || Char.equal sep ')') && Char.equal after ' ' then
          Some (String.prefix stripped (i + 1))
        else None
    | _ -> None
  in
  match ordered with
  | Some m -> Some m
  | None ->
      List.find_map [ "* "; "+ " ] ~f:(fun m ->
          if String.is_prefix stripped ~prefix:m then Some (String.rstrip m) else None)

(** A fence opens a region whose contents are not the notes' dialect at all: a ["- "] line inside one
    is example text, and reading it as a bullet invents integrity and repetition failures out of
    someone's code sample. Reported, and its contents skipped, so one unsupported construct is one
    finding. *)
let fence_marker stripped =
  List.find [ "```"; "~~~" ] ~f:(fun m -> String.is_prefix stripped ~prefix:m)

(** Block constructs that Markdown recognises only at column zero, and that the notes do not use. At
    depth they are ordinary continuation text — [`  <= 8 and …`] and [`  <that target>`] both occur —
    so the column-zero condition is load-bearing, not incidental. *)
let unsupported_block stripped =
  let thematic =
    (String.length stripped >= 3
    && List.exists [ '-'; '*'; '_' ] ~f:(fun c ->
           String.for_all stripped ~f:(fun d -> Char.equal c d || Char.equal d ' ')))
    || (String.length stripped >= 3 && String.for_all stripped ~f:(Char.equal '='))
  in
  if String.is_prefix stripped ~prefix:">" then
    Some "a block quote, which this scan does not read"
  else if String.is_prefix stripped ~prefix:"<" then
    Some "an HTML block, whose text no rule here can see"
  else if thematic then
    Some "a thematic break or a setext heading underline, neither of which the notes use"
  else None

(** One nesting level is what the notes are written in, and what {!parse_file} documents. A third
    level was being accepted anyway — [expected] simply advanced by two more spaces — so the
    discipline held only by convention (Codex P2, round 1). *)
let max_nesting_indent = 2

type parse = { bullets : bullet list; structure : finding list }

(** The one pass that both collects bullets and reports where the structure stopped parsing.

    The open-bullet stack is closed by a blank line, a heading, a table line and any line at column
    zero that is not a bullet — the notes' abstract and backlink paragraphs are exactly that. A
    continuation line must sit at the innermost open bullet's indentation plus two, with no popping:
    an indented line at the PARENT's depth while a nested bullet is open is a lazy continuation of
    the nested one to Markdown and of the parent to a naive reader, and reporting that ambiguity is
    cheaper than picking a side of it. *)
let parse_file ~file contents =
  let findings = ref [] in
  let report line rule msg = findings := finding ~file ~line ~rule msg :: !findings in
  let bad line msg = report line rule_bullet_integrity msg in
  let bullets = ref [] in
  (* Innermost FIRST: each entry is a bullet being accumulated, with its text lines reversed.
     Flushing always goes outermost first, so [bullets] comes out in document order -- which is what
     the repetition rule reports against (the SECOND occurrence is the finding) and what orders the
     findings of a file. *)
  let stack : (bullet * string list ref) list ref = ref [] in
  let close_all () =
    List.iter (List.rev !stack) ~f:(fun (b, texts) ->
        let text = String.concat ~sep:" " (List.rev !texts) in
        bullets := { b with text } :: !bullets);
    stack := []
  in
  (* Set while inside a fenced region, holding the fence that opened it: its contents are somebody's
     example, not the notes' dialect, and reading a `- ` line in one as a bullet invents failures. *)
  let fence_open = ref None in
  List.iter (lines contents) ~f:(fun (lineno, line) ->
      let stripped = String.strip line in
      match !fence_open with
      | Some opener ->
          if String.is_prefix stripped ~prefix:opener then fence_open := None
          else ()
      | None ->
          if is_blank line then close_all ()
          else if has_leading_tab line then (
            bad lineno
              "a tab in the indentation: indentation here is spaces, two per nesting level";
            close_all ())
          else
            let indent = indent_of line in
            match fence_marker stripped with
            | Some m ->
                bad lineno
                  (Printf.sprintf
                     "a %s fenced block: the notes have no fenced code, and the lines inside one \
                      are not bullets, table rows or prose -- teach this scan what they are before \
                      writing one"
                     m);
                fence_open := Some m;
                close_all ()
            | None ->
                if String.is_prefix stripped ~prefix:"#" then close_all ()
                else if is_table_line line then close_all ()
                else
                  match foreign_list_marker stripped with
                  | Some m ->
                      bad lineno
                        (Printf.sprintf
                           "the list marker %S: bullets here are written \"- \", and an item this \
                            scan does not recognise is an item no rule checks"
                           m);
                      close_all ()
                  | None -> (
                      if String.equal stripped "-" then (
                        bad lineno "an empty bullet";
                        close_all ())
                      else if String.is_prefix stripped ~prefix:"- " then (
                        (* A bullet start closes every open bullet at or inside its own depth. *)
                        let rec pop acc =
                          match !stack with
                          | (b, texts) :: rest when b.indent >= indent ->
                              stack := rest;
                              pop ((b, texts) :: acc)
                          | _ -> acc
                        in
                        (* [pop] prepends as it unwinds inwards-out, so [popped] is outermost first. *)
                        List.iter (pop []) ~f:(fun (b, texts) ->
                            let text = String.concat ~sep:" " (List.rev !texts) in
                            bullets := { b with text } :: !bullets);
                        let expected = match !stack with [] -> 0 | (b, _) :: _ -> b.indent + 2 in
                        if indent > max_nesting_indent then
                          bad lineno
                            (Printf.sprintf
                               "a bullet nested %d deep: the notes go one level down and this scan \
                                reads no further, so anything below %d is unchecked"
                               ((indent / 2) + 1) max_nesting_indent
                            )
                        else if indent <> expected then
                          bad lineno
                            (Printf.sprintf
                               "a bullet indented %d, where the open list puts the next one at %d"
                               indent expected);
                        let b = { file; line = lineno; indent; text = "" } in
                        stack := (b, ref [ String.drop_prefix stripped 2 ]) :: !stack)
                      else if indent = 0 then (
                        (match unsupported_block stripped with
                        | Some what -> bad lineno what
                        | None -> ());
                        close_all ())
                      else
                        match !stack with
                        | [] ->
                            bad lineno
                              "an indented line continuing no bullet: nothing above it is an open \
                               list item"
                        | (b, texts) :: _ ->
                            if indent <> b.indent + 2 then
                              bad lineno
                                (Printf.sprintf
                                   "an indented line at %d continuing the bullet at line %d, whose \
                                    continuations sit at %d"
                                   indent b.line (b.indent + 2))
                            else texts := stripped :: !texts));
  (match !fence_open with
  | Some m ->
      findings :=
        finding ~file ~line:(List.length (lines contents)) ~rule:rule_bullet_integrity
          (Printf.sprintf "a %s fence that is never closed, so the rest of the file is unread" m)
        :: !findings
  | None -> ());
  close_all ();
  let bullets = List.rev !bullets in
  let terminator_findings =
    List.filter_map bullets ~f:(fun b ->
        if bullet_text_is_terminated b.text then None
        else
          let subject = bullet_subject b in
          Some
            (finding ~subject ~file ~line:b.line ~rule:rule_bullet_integrity
               (Printf.sprintf
                  "a bullet that does not end a sentence, so its tail may be elsewhere: %S ends \
                   \"…%s\" -- exempt it as %S if that ending is deliberate"
                  subject (String.suffix b.text 40)
                  (subject_key ~file ~subject))))
  in
  { bullets; structure = List.rev !findings @ terminator_findings }

(** The bullets of a file, for callers that want only those. *)
let bullets ~file contents = (parse_file ~file contents).bullets

(** Rule 1 over one file. *)
let check_structure ~file contents = (parse_file ~file contents).structure

(* ------------------------------------------------------------------ *)
(* Rule 3: table shape *)
(* ------------------------------------------------------------------ *)

type table = { start_line : int; rows : (int * string) list }

(** The table blocks of a file: maximal runs of lines whose trimmed form starts with a pipe. *)
let tables contents =
  let rec go acc current = function
    | [] -> List.rev (match current with None -> acc | Some t -> t :: acc)
    | (lineno, line) :: rest ->
        if is_table_line line then
          let current =
            match current with
            | None -> Some { start_line = lineno; rows = [ (lineno, line) ] }
            | Some t -> Some { t with rows = t.rows @ [ (lineno, line) ] }
          in
          go acc current rest
        else
          let acc = match current with None -> acc | Some t -> t :: acc in
          go acc None rest
  in
  go [] None (lines contents)

(** Rule 3 over one file. *)
let check_tables ~file contents =
  let all = lines contents in
  let line_at n = List.Assoc.find all n ~equal:Int.equal in
  List.concat_map (tables contents) ~f:(fun t ->
      let report line msg = finding ~file ~line ~rule:rule_table_shape msg in
      let closed =
        List.filter_map t.rows ~f:(fun (lineno, line) ->
            match row_cells line with
            | None ->
                Some
                  (report lineno
                     "a table row that does not close with a cell separator: a row cannot span \
                      physical lines, so this ends the table and truncates the row")
            | Some _ -> None)
      in
      if not (List.is_empty closed) then closed
      else
        let cells = List.map t.rows ~f:(fun (lineno, line) -> (lineno, Option.value_exn (row_cells line))) in
        let shape =
          match cells with
          | [] | [ _ ] | [ _; _ ] ->
              [
                report t.start_line
                  (Printf.sprintf
                     "a table of %d line(s): a table is a header, a delimiter row and at least one \
                      data row"
                     (List.length cells));
              ]
          | (_, header) :: (delim_line, delim) :: _ ->
              let width = List.length header in
              let delim_finding =
                if is_delimiter_row delim then []
                else
                  [
                    report delim_line
                      "the line below a table's header is not a delimiter row, so the block above \
                       it is not a table";
                  ]
              in
              let width_findings =
                List.filter_map cells ~f:(fun (lineno, c) ->
                    if List.length c = width then None
                    else
                      Some
                        (report lineno
                           (Printf.sprintf "a row of %d cells in a table whose header has %d"
                              (List.length c) width)))
              in
              delim_finding @ width_findings
        in
        let after =
          let next = t.start_line + List.length t.rows in
          match line_at next with
          | Some l when (not (is_blank l)) && not (List.is_empty (pipes_outside_code l)) ->
              [
                report next
                  "a cell separator on the line below a table: this reads as the tail of a row that \
                   was wrapped, which ends the table above it";
              ]
          | _ -> []
        in
        shape @ after)

(* ------------------------------------------------------------------ *)
(* Rules 2 and 4: the index against the files *)
(* ------------------------------------------------------------------ *)

(** One parsed index row. [target] is as written, relative to the index's own directory; [anchor] is
    the fragment after ['#'], if any. *)
type index_row = {
  row_line : int;
  link_text : string;
  target : string;
  anchor : string option;
  hooks : string list;
}

let backticked cell =
  List.filter_map (code_spans cell) ~f:(fun (start, stop) ->
      let s = String.sub cell ~pos:start ~len:(stop - start) in
      let s = String.strip s ~drop:(Char.equal '`') in
      if String.is_empty (String.strip s) then None else Some s)

(** GitHub's heading slug: lowercased, spaces to hyphens, other punctuation dropped — and
    UNDERSCORES KEPT, which is the half that matters here. The headings a note would anchor are
    identifiers (`ident_blacklist`, `promote_prec`), and GitHub anchors those as `#ident_blacklist`;
    rewriting the underscore rejected the correct anchor and accepted the wrong one (Codex P2,
    round 1). Hyphens are likewise kept as themselves rather than re-derived. *)
let slug heading =
  String.lowercase heading
  |> String.to_list
  |> List.filter_map ~f:(fun c ->
         if Char.is_alphanum c || Char.equal c '_' || Char.equal c '-' then Some c
         else if Char.equal c ' ' then Some '-'
         else None)
  |> String.of_list

let headings contents =
  List.filter_map (lines contents) ~f:(fun (_, line) ->
      let s = String.strip line in
      if String.is_prefix s ~prefix:"#" then
        Some (String.strip (String.lstrip s ~drop:(Char.equal '#')))
      else None)

(** HTML comments removed, so that a link inside one is not a link. Unterminated, the comment runs
    to the end of the text — which is what a browser does with it too. *)
let strip_html_comments text =
  let rec go acc rest =
    match String.substr_index rest ~pattern:"<!--" with
    | None -> acc ^ rest
    | Some i -> (
        let after = String.drop_prefix rest (i + 4) in
        match String.substr_index after ~pattern:"-->" with
        | None -> acc ^ String.prefix rest i
        | Some j -> go (acc ^ String.prefix rest i) (String.drop_prefix after (j + 3)))
  in
  go "" text

(** The NAVIGABLE links of a line: [\[text\](target)] outside inline code, outside HTML comments, and
    not an image ([!\[alt\](src)] renders a picture, not a way back). A substring test for
    ["](../agent-notes.md)"] called a file reachable when those same bytes sat in a code span, in a
    comment, or behind an image (Codex P2, round 1) — all three of which a reader cannot follow. *)
let markdown_links line =
  let line = strip_html_comments line in
  let spans = code_spans line in
  let n = String.length line in
  let rec go i acc =
    if i >= n then List.rev acc
    else if Char.equal line.[i] '[' && not (in_any_span spans i) then
      let image = i > 0 && Char.equal line.[i - 1] '!' in
      match String.index_from line i ']' with
      | Some close
        when close + 1 < n
             && Char.equal line.[close + 1] '('
             && (not (in_any_span spans close))
             && not (in_any_span spans (close + 1)) -> (
          match String.index_from line (close + 1) ')' with
          | Some rparen when not (in_any_span spans rparen) ->
              let text = String.sub line ~pos:(i + 1) ~len:(close - i - 1) in
              let target =
                String.sub line ~pos:(close + 2) ~len:(rparen - close - 2)
              in
              let acc = if image then acc else (text, target) :: acc in
              go (rparen + 1) acc
          | _ -> go (i + 1) acc)
      | _ -> go (i + 1) acc
    else go (i + 1) acc
  in
  go 0 []

(** Every navigable link of a whole file. *)
let links_of contents = List.concat_map (lines contents) ~f:(fun (_, l) -> markdown_links l)

(** [\[text\](target)] filling the whole cell, and nothing else. *)
let parse_link cell =
  let cell = String.strip cell in
  match (String.chop_prefix cell ~prefix:"[", String.chop_suffix cell ~suffix:")") with
  | Some _, Some _ -> (
      let body = String.sub cell ~pos:1 ~len:(String.length cell - 2) in
      match String.substr_index body ~pattern:"](" with
      | None -> None
      | Some i ->
          let text = String.prefix body i in
          let target = String.drop_prefix body (i + 2) in
          if String.contains text ']' || String.contains target ')' then None
          else Some (text, target))
  | _ -> None

(** The index's rows, or the one finding that says its table did not parse at all.

    The distinction is the point. A wrapped row ends the table, which leaves the index looking like
    TWO tables with most of its rows outside both -- and asking the hook and reachability rules about
    that produces one spurious "unreachable" line per notes file, burying the single finding that
    says what actually happened. So a refusal comes back as [Error] and stops those rules, which then
    report that they could not be evaluated rather than reporting twelve falsehoods. It cannot hide a
    real defect: the refusal is itself a failure, and the table rule has already named the line. *)
let index_rows ~file contents =
  match tables contents with
  | [] -> Error (file_finding ~file ~rule:rule_index_agreement "no table: the index is a table")
  | _ :: _ :: _ ->
      Error
        (file_finding ~file ~rule:rule_index_agreement
           "more than one table: the index is one table, so a row that ends it puts most of the \
            index outside both")
  | [ t ] ->
      let data = match t.rows with _ :: _ :: rest -> rest | _ -> [] in
      let rows, findings =
        List.partition_map data ~f:(fun (lineno, line) ->
            match row_cells line with
            | Some (link :: rest) -> (
                let hooks = String.concat ~sep:" " rest in
                match parse_link link with
                | None ->
                    Either.Second
                      (finding ~file ~line:lineno ~rule:rule_index_agreement
                         (Printf.sprintf
                            "a row whose first cell is not a link to a notes file: %S" link))
                | Some (link_text, target) ->
                    let target, anchor =
                      match String.lsplit2 target ~on:'#' with
                      | Some (t, a) -> (t, Some a)
                      | None -> (target, None)
                    in
                    Either.First { row_line = lineno; link_text; target; anchor; hooks = backticked hooks })
            | _ ->
                Either.Second
                  (finding ~file ~line:lineno ~rule:rule_index_agreement
                     "a row with no cells"))
      in
      Ok (rows, findings)

(** Rules 2 and 4. [files] is every notes file, keyed by its path relative to the index's directory
    — ["agent-notes/build-and-test.md"] — which is what an index link spells. [index_file] is the
    index's own path relative to the same place, as it appears in the files' backlinks. *)
let check_index ~index_file ~index_contents ~(files : (string * string) list) =
  match index_rows ~file:index_file index_contents with
  | Error refusal ->
      [
        refusal;
        file_finding ~file:index_file ~rule:rule_reachability
          "not evaluated: the index's table does not parse, so which files it reaches is unknown \
           -- fix the table first";
      ]
  | Ok (rows, row_findings) ->
  let per_row =
    List.concat_map rows ~f:(fun row ->
        let report msg = finding ~file:index_file ~line:row.row_line ~rule:rule_index_agreement msg in
        let basename = List.last_exn (String.split row.target ~on:'/') in
        let text_finding =
          if String.equal row.link_text basename then []
          else
            [
              report
                (Printf.sprintf "link text %S naming the file %S: the index reads as a file list"
                   row.link_text basename);
            ]
        in
        match List.Assoc.find files row.target ~equal:String.equal with
        | None ->
            text_finding
            @ [ report (Printf.sprintf "a link to %S, which is not a notes file" row.target) ]
        | Some contents ->
            let anchor_finding =
              match row.anchor with
              | None -> []
              | Some a ->
                  if List.exists (headings contents) ~f:(fun h -> String.equal (slug h) a) then []
                  else
                    [
                      report
                        (Printf.sprintf "an anchor #%s that %s has no heading for" a row.target);
                    ]
            in
            let hook_findings =
              List.filter_map row.hooks ~f:(fun hook ->
                  if String.is_substring contents ~substring:hook then None
                  else
                    Some
                      (report
                         (Printf.sprintf
                            "the hook `%s`, which does not occur in %s: the index is what a lookup \
                             greps first, so a hook absent from its target is a dead end"
                            hook row.target)))
            in
            text_finding @ anchor_finding @ hook_findings)
  in
  let targets = List.map rows ~f:(fun r -> r.target) in
  let duplicates =
    List.filter_map rows ~f:(fun r ->
        let earlier = List.filter rows ~f:(fun o -> String.equal o.target r.target && o.row_line < r.row_line) in
        if List.is_empty earlier then None
        else
          Some
            (finding ~file:index_file ~line:r.row_line ~rule:rule_reachability
               (Printf.sprintf "a second row for %s: one row per file, or a lookup has two answers"
                  r.target)))
  in
  let orphans =
    List.filter_map files ~f:(fun (name, _) ->
        if List.mem targets name ~equal:String.equal then None
        else
          Some
            (file_finding ~file:name ~rule:rule_reachability
               "a notes file no index row links to: unreachable from the one place a lookup starts"))
  in
  let backlinks =
    let index_base = List.last_exn (String.split index_file ~on:'/') in
    List.filter_map files ~f:(fun (name, contents) ->
        (* A link, not a byte sequence: the target has to be one a reader can follow back. *)
        let target = "../" ^ index_base in
        if
          List.exists (links_of contents) ~f:(fun (_, t) ->
              String.equal (List.hd_exn (String.split t ~on:'#')) target)
        then None
        else
          Some
            (file_finding ~file:name ~rule:rule_reachability
               (Printf.sprintf
                  "no navigable link back to the index (a link to %S, outside code spans, HTML \
                   comments and images): a file reached on its own leaves the reader without the \
                   scope discipline the index carries"
                  target)))
  in
  row_findings @ per_row @ duplicates @ orphans @ backlinks

(* ------------------------------------------------------------------ *)
(* Rule 5: no bullet repeated *)
(* ------------------------------------------------------------------ *)

(** Two bullets whose normalized texts agree on this many leading characters are reported as near
    duplicates. Long enough that no two of the corpus's 177 bullets collide, short enough that a
    fact re-promoted with its tail reworded is still caught. *)
let near_duplicate_prefix = 60

(** Rule 5 over every bullet of the notes, the index's own included. Exact repetition is reported
    first and suppresses the near-duplicate report for the same pair, so one fact promoted twice is
    one finding. *)
let check_repetition (bullets : bullet list) =
  let exact = Hashtbl.create (module String) in
  let prefix = Hashtbl.create (module String) in
  List.concat_map bullets ~f:(fun b ->
      let key = normalize b.text in
      let report rule msg = finding ~file:b.file ~line:b.line ~rule msg in
      let where (o : bullet) = Printf.sprintf "%s:%d" o.file o.line in
      match Hashtbl.find exact key with
      | Some (o : bullet) ->
          [
            report rule_no_repetition
              (Printf.sprintf "a bullet repeating %s verbatim: \"%s…\"" (where o)
                 (String.prefix key 60));
          ]
      | None ->
          Hashtbl.set exact ~key ~data:b;
          let pkey = String.lowercase (String.prefix key near_duplicate_prefix) in
          if String.length pkey < near_duplicate_prefix then []
          else (
            match Hashtbl.find prefix pkey with
            | Some (o : bullet) ->
                [
                  report rule_no_repetition
                    (Printf.sprintf
                       "a bullet opening exactly as %s does: \"%s…\" — one fact promoted twice is a \
                        fact that will be updated once"
                       (where o) (String.prefix key near_duplicate_prefix));
                ]
            | None ->
                Hashtbl.set prefix ~key:pkey ~data:b;
                []))

(* ------------------------------------------------------------------ *)
(* The whole scan *)
(* ------------------------------------------------------------------ *)

(** Every rule, over an index and the files it indexes. Findings come back grouped by rule in
    {!rules} order, and within a rule in file and line order. [files] is keyed as {!check_index}
    describes. *)
let check_all ~index_file ~index_contents ~(files : (string * string) list) =
  let all = (index_file, index_contents) :: files in
  let structure = List.concat_map all ~f:(fun (file, c) -> check_structure ~file c) in
  let table = List.concat_map all ~f:(fun (file, c) -> check_tables ~file c) in
  let index = check_index ~index_file ~index_contents ~files in
  let bullets = List.concat_map all ~f:(fun (file, c) -> bullets ~file c) in
  let repetition = check_repetition bullets in
  let found = structure @ table @ index @ repetition in
  (bullets, List.concat_map rules ~f:(fun r -> List.filter found ~f:(fun f -> String.equal f.rule r)))
