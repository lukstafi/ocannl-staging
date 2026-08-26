(** Floors over a scanned census: the tripwire that keeps a scanning test from passing vacuously,
    without the tally that made it churn.

    A scanning test cannot report its own blindness. Handed nothing -- a glob that stopped matching,
    a directory that moved, a dependency the rule forgot to declare -- it finds no defects and says
    so, and "no defects" is exactly what a healthy run says too. An exact count of what was scanned
    used to close that hole, and closed it at the price of a tally: the number moved on every
    correct addition anywhere under the globs, so every contributor promoted a line they did not
    touch, and two branches adding a file to the same directory wrote identical text on that line
    and merged to a total wrong by one (gh-ocannl-701, gh-ocannl-665).

    A floor keeps the tripwire and drops the tally. A glob that breaks goes to zero, which any floor
    catches; a file added moves nothing. Floors sit well below the count of the day they are
    written, so ordinary deletions do not fail either -- the number to raise one to is never
    "today's count", which would restore the tally.

    The machinery is here rather than in any one scanner because three scans now need it and each
    would otherwise reinvent the census-by-root, the itemised diagnostic and the stderr report
    (gh-ocannl-712). What stays with each scanner is its own floor table: which roots it globs, and
    how far each may fall, are facts about that scan. *)

open Base

(** The directory [path] sits in, with the [../] prefixes dune's globs arrive with removed. *)
let directory_of path =
  let rec go p = match String.chop_prefix p ~prefix:"../" with Some p -> go p | None -> p in
  go (Stdlib.Filename.dirname path)

(** The configured root [path] sits under, or its own directory if no root claims it.

    A path outside every root is left as itself deliberately: that is an item arriving from
    somewhere the globs are not written for, and it should show up in the census as a directory
    nobody recognises rather than be filed under one that happens to be a prefix of it. The longest
    matching root wins, so a root nested inside another files its files under the more specific one.

    Roots rather than directories, because the rules glob with [glob_files_rec]: an item in a
    subdirectory of a root belongs to that root's census and that root's floor, not to a bucket of
    its own -- a bucket of its own would be a new line in the golden under no floor at all. *)
let root_of ~floors path =
  let directory = directory_of path in
  let under (root, _) =
    String.equal directory root || String.is_prefix directory ~prefix:(root ^ "/")
  in
  List.filter floors ~f:under
  |> List.max_elt ~compare:(fun (a, _) (b, _) -> Int.compare (String.length a) (String.length b))
  |> Option.value_map ~default:directory ~f:fst

(** [paths] counted per root, sorted by root. *)
let counts_by_root ~floors paths =
  List.map paths ~f:(root_of ~floors)
  |> List.sort_and_group ~compare:String.compare
  |> List.map ~f:(fun group -> (List.hd_exn group, List.length group))

(** Which roots the census actually came from, by name and not by count: the scope of the globs, for
    a golden.

    The alternative to printing it would be a list of permitted roots inside the test -- a second
    copy of the globs in the dune file, which is the hand-maintained list gh-ocannl-592 removed.
    Printing makes the scope reviewable instead: an item arriving from somewhere the globs do not
    name shows up as a new directory in the diff, and a root that stops being scanned shows up by
    leaving the line. How MANY items each root contributed is a tally, so it goes to stderr instead
    ({!report}). *)
let roots ~floors paths = List.map (counts_by_root ~floors paths) ~f:fst

(** The floors [paths] fails, itemised: a root below its bound, or absent from the census altogether
    -- which is what a glob matching nothing looks like from here.

    Itemised rather than summed, for the reason gh-ocannl-665 recorded about a floor's diagnostic:
    "one short" does not say which root is standing on nothing. [noun] names what was counted
    ("source", "golden") and [floors_name] the constant to lower when the items really went away. *)
let violations ~floors ~noun ~floors_name paths =
  let counts = Map.of_alist_exn (module String) (counts_by_root ~floors paths) in
  List.filter_map floors ~f:(fun (root, floor) ->
      let count = Option.value (Map.find counts root) ~default:0 in
      if count >= floor then None
      else
        Some
          (Printf.sprintf
             "%s contributed %d %s%s, below its floor of %d -- either the rule's glob over that \
              directory has stopped matching, or the %ss really went away and the floor in %s \
              should come down with them"
             root count noun
             (if count = 1 then "" else "s")
             floor noun floors_name))

(** The per-root counts and the total, on stderr -- where a golden does not see them, so the numbers
    stay readable in a run's output without a tally in a diffed file (gh-ocannl-665, gh-ocannl-701).
    [what] heads the report ("Sources scanned"). *)
let report ~floors ~noun ~what paths =
  Stdio.eprintf "%s per scan root (not diffed -- see gh-ocannl-701):\n" what;
  List.iter (counts_by_root ~floors paths) ~f:(fun (root, count) ->
      let floor =
        match List.Assoc.find floors root ~equal:String.equal with
        | Some floor -> Printf.sprintf "floor %d" floor
        | None -> "no floor -- not a root the globs are written for"
      in
      Stdio.eprintf "  %s: %d (%s)\n" root count floor);
  Stdio.eprintf "Total: %d %ss.\n" (List.length paths) noun
