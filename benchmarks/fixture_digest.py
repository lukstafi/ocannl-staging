#!/usr/bin/env python3
"""The recorded identity of a benchmark fixture (gh-ocannl-645, gh-ocannl-759).

`benchmarks/fixtures/` is gitignored (the fixtures are large, regenerable artifacts), so no
checkout establishes a fixture's bytes. Without a recorded digest nothing in the repository says
which bytes a published report was measured on, and nothing can catch a fixture regenerated at a
different spec revision, or by a different numpy: the difference applies *uniformly* to every cell,
so the cross-cell parity gate (which compares cells with each other, not with the workload the
report names) certifies it exactly as it certifies the intended workload. Cross-session comparisons
(`report-gh569-hip.md`'s 46.65 ms denominator against `report-gh612-hip.md`'s 32.33 ms) are only
meaningful if both ran the same bytes, and that is the whole point of such a measurement.

So `gen_fixtures.py` records `fixtures/DIGESTS.txt` (checked in, unlike the fixtures themselves) as
it generates, `orchestrate.py` refuses to measure a fixture that does not match it, and every
result row and report states the digest its numbers are on. A deliberate regeneration rewrites the
file, which shows up as a reviewable diff rather than as silence.

**A fixture is recorded per origin** (gh-ocannl-759). The measuring boxes generate their own
fixtures from their own venvs, and numpy promises no `Generator` stream stability across releases,
so the same workload spec legitimately has different bytes on different boxes -- and it does today:
`mlp_small` and `gpt2_mini` hash differently on minix and rog-nv, at identical sizes, which is why
`report-hip.md` and `report-gh675-cuda.md` are not cross-box comparable for those two. A file with
one entry per name cannot say that. It also sets a trap: whichever box regenerates first overwrites
the other's entry, and the other box's untouched fixtures then read as MISMATCH -- a changed
workload announced where nothing changed. Keying entries by `(name, origin)` removes both problems:
every box's bytes are recorded and attributed, a fixture matches if it is *some* recorded box's
bytes, and the report says whose.

This module is the one implementation of the file's format, shared by the generator, the
orchestrator and their tests; it deliberately imports nothing outside the standard library, so a
checkout without the benchmark venv can still read, check and record digests. That is also why
`--record` lives here rather than in `gen_fixtures.py`: pinning bytes that already exist needs no
numpy and no safetensors, and the boxes whose published numbers most need pinning are exactly the
ones whose fixtures predate any venv you could reconstruct.

    python3 benchmarks/fixture_digest.py --record   # pin fixtures/*.safetensors as this box's
    python3 benchmarks/fixture_digest.py --check    # what is on disk, against what is recorded
    python3 benchmarks/fixture_digest.py --list-declared-measurement-boxes
                                                    # the fleet matrix, one box per line
"""

import argparse
import hashlib
import platform
import sys
from collections import namedtuple
from pathlib import Path

DIGEST_FILE = "DIGESTS.txt"
MEASUREMENT_BOXES_FIELD = "# measurement-boxes:"

#: One recorded identity: `sha256` and `size` are the bytes, `origin` is the box that has them.
Entry = namedtuple("Entry", "sha256 size origin")

HEADER = """\
# Fixture digests: the bytes each published measurement is on (gh-ocannl-645, gh-ocannl-759).
#
# The fixtures themselves are gitignored, so this file is the only checked-in statement of
# what one contains. gen_fixtures.py rewrites the entries it regenerates; orchestrate.py
# refuses to measure a fixture that matches none of them (--no-fixture-digest-check opts out),
# and stamps every result row and report section with the digest -- and the origin -- it ran on.
#
# A changed digest here is a changed workload: numbers measured before it are not comparable
# with numbers measured after it, whatever the report calls the workload. Fixture bytes depend
# on the workload spec, on gen_fixtures.py, and on the numpy version that drew the random
# streams (numpy does not promise Generator stream stability across releases), so a mismatch
# names a real difference even when the spec is untouched.
#
# The measurement-boxes header field declares every box that publishes benchmark measurements.
# It is independent of the entry rows: that is how a missing entry can mean "this measuring box
# has unrecorded bytes" instead of being indistinguishable from "this box never measures here".
# Writers preserve the declared set and add a newly recording origin to it.
#
# The <origin> field names the box whose fixtures these bytes are, because the same workload
# has different bytes on different boxes and the reports have to say which. Two entries under
# one name are two boxes' bytes, NOT a history: numbers measured on one are not comparable
# with numbers measured on the other. Regenerating replaces only the recording box's entry --
# regeneration is a cross-box event and has to be coordinated across every origin listed here,
# or the boxes silently diverge again (see benchmarks/README.md).
#
# <sha256>  <bytes>  <name>  <origin>
"""


def header(measurement_boxes):
    """The explanatory header plus its one machine-readable measurement-box declaration."""
    boxes = sorted(check_measurement_boxes(measurement_boxes))
    marker = "# <sha256>  <bytes>  <name>  <origin>\n"
    declaration = f"{MEASUREMENT_BOXES_FIELD} {' '.join(boxes)}\n#\n"
    return HEADER.replace(marker, declaration + marker)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def this_origin():
    """The box doing the recording, or None when it cannot name itself.

    None rather than a placeholder: a literal `unknown-host` is not an origin, it is every
    nameless box sharing one. Two of them recording different bytes for one fixture would see the
    second replace the first under that one name -- exactly the "whichever box records last wins"
    loss that keying entries by origin exists to prevent, only now with a name that reads like an
    answer. `resolve_origin` turns it into a demand for an explicit --origin instead.
    """
    return platform.node() or None


def origin_default_help():
    """How the CLIs describe the default origin, on a host that may not be able to name itself."""
    node = this_origin()
    return f"default: this host, {node!r}" if node else "no default: this host reports no name"


def cli_command():
    """`fixture_digest.py` spelled so it runs from the CALLER's cwd, whatever that is.

    Remediation text is read by an operator who will paste it, and the canonical sweep command
    (`benchmarks/.venv/bin/python benchmarks/orchestrate.py`) runs from the repository root, where
    a bare `fixture_digest.py` names nothing.

    Forward slashes, on every platform. The relative form is a path ARGUMENT, and cmd, PowerShell
    and every shell a Windows operator might paste it into take `benchmarks/fixture_digest.py`
    just as happily as the backslashed spelling `str()` would give them there -- while the
    advertised command being the same string everywhere is what lets one sentence of remediation
    be pinned by one test. The absolute fallback keeps the platform's own spelling: it is not a
    relative path anyone reassembles, and on Windows a drive letter belongs with backslashes.
    """
    here = Path(__file__).resolve()
    try:
        return here.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return str(here)


def check_origin(origin):
    """Origins are non-empty, whitespace-free and comma-free.

    Whitespace because the digest file is whitespace-split, so a spaced origin writes a line that
    reads back as a different (or malformed) entry. Commas because AGREEING origins serialize as
    `a,b` into the single `fixture_origin` field every result row and report section carries
    (`status`): an origin literally named `minix,rocm` produces a field byte-identical to the one
    two boxes named `minix` and `rocm` recording the same bytes produce, so no consumer of a
    published row can tell one box from two. An origin has to name exactly one box.
    """
    if not origin or origin.split() != [origin] or "," in origin:
        raise ValueError(
            "origin must be a single non-empty word without whitespace or commas (a comma is how "
            f"the origins of agreeing boxes are joined, so it cannot be inside one), got {origin!r}"
        )
    return origin


def check_measurement_boxes(boxes):
    """A non-empty, duplicate-free list of origins, suitable for the header field."""
    boxes = [check_origin(box) for box in boxes]
    if not boxes:
        raise ValueError(
            f"{MEASUREMENT_BOXES_FIELD} must name at least one box; an empty declaration cannot "
            "distinguish a missing fixture entry from a workload no box measures"
        )
    if len(set(boxes)) != len(boxes):
        repeated = next(box for box in boxes if boxes.count(box) > 1)
        raise ValueError(
            f"{MEASUREMENT_BOXES_FIELD} names {repeated!r} more than once; it declares a set"
        )
    return boxes


def check_fixture_name(name):
    """Fixture names are single whitespace-free words, for the same reason origins are.

    `write_digests` emits them unescaped into the whitespace-split format, so a name containing
    whitespace records fine and breaks every later read of the whole rewritten file. Checked
    wherever a name is about to be committed to: recording (`one_path_per_name`) and generation
    (`gen_fixtures.py`, BEFORE any fixture is built, since building overwrites the bytes the
    published numbers rest on).
    """
    if not name or name.split() != [name]:
        raise ValueError(
            f"{name!r} cannot be recorded: the digest file is whitespace-split, so this name "
            "would write a line every later read refuses, breaking the whole rewritten file; "
            "use a single whitespace-free word"
        )
    return name


def resolve_origin(origin, adopting=None):
    """The ONE place a missing origin becomes this host's -- and the only one that may.

    Every caller routes through here rather than writing `origin or this_origin()`, because that
    idiom cannot tell "not given" from "given as empty", and an empty one is how automation fails
    (`--origin "$BOX"` with $BOX unset). Silently substituting the hostname there attributes a
    fixture to the wrong box and persists it, which is the exact error this file exists to
    prevent -- so absence defaults, emptiness is refused, and a host that cannot name itself is
    refused rather than sharing a placeholder with every other nameless box.

    `adopting` is the box a `--adopt-legacy` migration attributes the old rows to. A DEFAULTED
    origin must agree with it: the migration names one box, so resolving the two facts it writes
    ("whose were the old rows", "whose are the bytes on disk") from two independent sources can
    split one box across two origin names -- `--adopt-legacy rog-nv` on a host that calls itself
    `rog-nv-wsl` writes both names, which reads downstream as two boxes agreeing, or, for a
    fixture whose legacy entry it adopted, as the box having diverged from itself. Stating both
    explicitly is still allowed, and is how one box migrates ANOTHER box's legacy rows.
    """
    if origin is None:
        origin = this_origin()
        if origin is None:
            raise ValueError(
                "this host reports no name (platform.node() is empty), so nothing here can say "
                "whose bytes these are; pass --origin <box> explicitly. Recording them under a "
                "shared placeholder would let the next nameless box overwrite this entry under "
                "that same name, which is the provenance loss this file exists to prevent"
            )
        if adopting is not None and origin != adopting:
            raise ValueError(
                f"the legacy rows are being adopted as {adopting!r}, but this host names itself "
                f"{origin!r}, so the fixtures on disk would be recorded under a second origin: "
                "one box wearing two names is indistinguishable here from two boxes. Say which "
                f"you mean: --origin {adopting} if {adopting!r} is this box under the name the "
                f"reports use, or --origin {origin} if you are migrating another box's rows"
            )
    return check_origin(origin)


def _read_document(path, legacy_origin=None):
    """`(entries, declared_boxes)` from `path`; an absent file records neither.

    Entries under one name are origin-sorted, and one origin appears at most once per name.

    `declared_boxes` is None for a pre-gh-ocannl-850 file. That compatibility is needed so
    `--record --adopt-legacy` can migrate an old local file; the checked-in file is separately
    required by the tests to carry exactly one declaration.

    `legacy_origin` attributes pre-gh-ocannl-759 three-field lines to that box instead of
    refusing them. It exists only for the one-shot migration behind `--record --adopt-legacy`,
    where the operator is ASSERTING whose those bytes were; nothing infers it.
    """
    path = Path(path)
    entries = {}
    if not path.exists():
        return entries, None

    declared_boxes = None

    def add(lineno, sha, size, name, origin):
        """The ONE insertion point, so every parse path pays the duplicate check.

        A four-field entry and an adopted legacy line can land on the same (name, origin), and
        with two insertion sites whichever came last silently won -- making `--adopt-legacy`
        order-dependent, and able to persist the wrong historical digest for a fixture that is not
        even among the files being recorded.
        """
        # Reader inputs are held to the same identity rules as writer inputs: `check_origin`
        # guards every CLI, but a checked-in or hand-merged row bypasses them, and a recorded
        # `minix,rocm` would flow through `status` into a `fixture_origin` field byte-identical
        # to two agreeing boxes'.
        try:
            check_origin(origin)
        except ValueError as e:
            raise ValueError(f"{path}:{lineno}: {e}") from None
        by_origin = entries.setdefault(name, {})
        if origin in by_origin:
            raise ValueError(
                f"{path}:{lineno}: {name} is recorded twice for origin {origin!r}; one box has "
                "one set of bytes per fixture, so this file cannot say which"
            )
        by_origin[origin] = Entry(sha, int(size), origin)

    for lineno, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.strip()
        if line.startswith(MEASUREMENT_BOXES_FIELD):
            if declared_boxes is not None:
                raise ValueError(
                    f"{path}:{lineno}: {MEASUREMENT_BOXES_FIELD} appears more than once; "
                    "there must be one authoritative box set"
                )
            try:
                declared_boxes = check_measurement_boxes(
                    line[len(MEASUREMENT_BOXES_FIELD) :].split()
                )
            except ValueError as e:
                raise ValueError(f"{path}:{lineno}: {e}") from None
            continue
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) == 3:
            # The pre-gh-ocannl-759 three-field format. Never adopted under a GUESSED origin: an
            # unattributed digest is precisely the "which box is this?" silence this file exists
            # to break, and the boxes' bytes really do differ, so a guess would be a coin flip
            # recorded as a fact. An operator who knows whose they are says so with
            # --adopt-legacy, which is what makes this refusal a migration rather than a wall.
            if legacy_origin is not None:
                sha, size, name = fields
                add(lineno, sha, size, name, legacy_origin)
                continue
            raise ValueError(
                f"{path}:{lineno}: {line!r} is the old unattributed format; attribute it with "
                f"`python3 {cli_command()} --record --adopt-legacy <box>` (which rewrites these "
                "lines under that box and leaves their bytes untouched) so the entry says whose "
                "bytes it is (gh-ocannl-759)"
            )
        if len(fields) != 4:
            raise ValueError(
                f"{path}:{lineno}: expected '<sha256>  <bytes>  <name>  <origin>', got {line!r}"
            )
        add(lineno, *fields)
    entries = {
        name: [by_origin[o] for o in sorted(by_origin)] for name, by_origin in entries.items()
    }
    if declared_boxes is not None:
        undeclared = sorted(
            {e.origin for recorded in entries.values() for e in recorded} - set(declared_boxes)
        )
        if undeclared:
            raise ValueError(
                f"{path}: digest entries name origin(s) absent from {MEASUREMENT_BOXES_FIELD} "
                f"{', '.join(undeclared)}; add every measuring box to the one declaration"
            )
    return entries, declared_boxes


def read_digests(path, legacy_origin=None):
    """`{name: [Entry, ...]}` recorded in `path`; an absent file records nothing."""
    return _read_document(path, legacy_origin=legacy_origin)[0]


def measurement_boxes(path):
    """The declared measurement boxes, or recorded origins for a legacy undeclared file."""
    entries, declared = _read_document(path)
    if declared is not None:
        return sorted(declared)
    return sorted({e.origin for recorded in entries.values() for e in recorded})


def declared_measurement_boxes(path):
    """The explicit header declaration, or None for a legacy undeclared file."""
    return _read_document(path)[1]


def write_digests(path, entries, measurement_boxes=None):
    """Rewrite `path` with its box declaration and entries, all sorted."""
    if measurement_boxes is None:
        measurement_boxes = {
            e.origin for recorded in entries.values() for e in recorded
        }
    body = "".join(
        f"{e.sha256}  {e.size}  {name}  {e.origin}\n"
        for name in sorted(entries)
        for e in sorted(entries[name], key=lambda e: e.origin)
    )
    rendered_header = header(measurement_boxes) if measurement_boxes else HEADER
    Path(path).write_text(rendered_header + body)


def one_path_per_name(fixtures):
    """`fixtures` as paths, refusing two different files that would be recorded under one name.

    Entries are keyed by `(name, origin)`, and `read_digests` refuses that pair appearing twice
    because one box has one set of bytes per fixture -- so recording must not manufacture from the
    other side the very ambiguity reading rejects. Two paths with one basename
    (`box-a/mlp_small.safetensors` and `box-b/mlp_small.safetensors`) are two boxes' copies of one
    workload: recorded in one command under one origin, the later would replace the earlier, and
    the replacement would be ANNOUNCED as a changed workload -- a regeneration event that never
    happened, with the other box's bytes gone from the file. Repeating one path is not ambiguous,
    so it is kept.

    Names must also survive the whitespace-split format they are written into: `write_digests`
    emits them unescaped, so a name containing whitespace would produce a line every later
    `read_digests` refuses -- and since recording rewrites the whole file, one such recording
    leaves the checked-in record unreadable. Refused here, before the file is touched (the
    origin-side twin of this check is `check_origin`).
    """
    seen = {}
    for fixture in fixtures:
        fixture = Path(fixture)
        check_fixture_name(fixture.name)
        first = seen.setdefault(fixture.name, fixture)
        if first.resolve() != fixture.resolve():
            raise ValueError(
                f"{first} and {fixture} would both be recorded as {fixture.name!r} for one origin, "
                "and this file has one set of bytes per (fixture, box), so it could not say which; "
                "record them under their own origins, in their own commands"
            )
    return list(seen.values())


def record(path, fixtures, origin=None, legacy_origin=None):
    """Record `fixtures` (paths) as `origin`'s bytes in the digest file at `path`.

    Every other entry is kept -- including other origins' entries for the same fixture, which is
    the point: a box regenerating its own copy must not evict the recording that another box's
    published numbers rest on.

    Returns the list of `(name, origin, was, now)` changes, where `was` is None for a fixture this
    origin had not recorded. The caller says so out loud: a silently rewritten digest is the
    failure this file exists to prevent, and a regeneration is where it would happen.
    """
    origin = resolve_origin(origin, adopting=legacy_origin)
    fixtures = one_path_per_name(fixtures)
    entries, declared_boxes = _read_document(path, legacy_origin=legacy_origin)
    changes = []
    for fixture in fixtures:
        fixture = Path(fixture)
        now = Entry(sha256_file(fixture), fixture.stat().st_size, origin)
        others = [e for e in entries.get(fixture.name, []) if e.origin != origin]
        was = next((e for e in entries.get(fixture.name, []) if e.origin == origin), None)
        if was != now:
            changes.append((fixture.name, origin, was, now))
        entries[fixture.name] = others + [now]
    boxes = set(declared_boxes or ())
    boxes.update(e.origin for recorded in entries.values() for e in recorded)
    boxes.add(origin)
    write_digests(path, entries, boxes)
    return changes


def status(fixture, entries):
    """`(verdict, sha256, size, origins)` of `fixture` against recorded `entries`.

    Verdict is MATCH, MISMATCH (the name is recorded, but for nobody's bytes these) or UNRECORDED
    (no entry at all: a fixture nothing in the repository describes, which is what a report must
    not quietly be measured on). `origins` names the boxes whose recorded bytes these are -- the
    answer to "which box's workload is this number on" -- and is None unless the verdict is MATCH.
    """
    fixture = Path(fixture)
    sha, size = sha256_file(fixture), fixture.stat().st_size
    recorded = entries.get(fixture.name)
    if not recorded:
        return "UNRECORDED", sha, size, None
    matching = [e.origin for e in recorded if (e.sha256, e.size) == (sha, size)]
    if not matching:
        return "MISMATCH", sha, size, None
    # More than one origin here is the boxes agreeing, which is worth seeing as plainly as their
    # disagreeing is.
    return "MATCH", sha, size, ",".join(sorted(matching))


def divergent_origins(path, names, origin):
    """Declared boxes absent or on DIFFERENT bytes for some of `names`, versus `origin`.

    What a regenerating box has to be told: their entries survive (so their fixtures still pass
    the gate), but their bytes are now a different workload from the one just generated, and
    nothing else will say so until someone compares two reports.

    Divergence is judged on the bytes, never on the name being different. A coordinated
    regeneration that lands the SAME bytes on both boxes is the outcome this whole mechanism is
    steering towards, and reporting it as "the other box is still on a different workload" would
    tell the operator to redo the thing they just succeeded at.
    """
    entries, declared_boxes = _read_document(path)
    boxes = set(declared_boxes or ())
    boxes.update(e.origin for recorded in entries.values() for e in recorded)
    divergent = set()
    for name in names:
        recorded = entries.get(name, [])
        mine = next((e for e in recorded if e.origin == origin), None)
        if mine is None:
            continue
        by_origin = {e.origin: e for e in recorded}
        divergent |= {
            box
            for box in boxes
            if box != origin
            and (
                box not in by_origin
                or (by_origin[box].sha256, by_origin[box].size) != (mine.sha256, mine.size)
            )
        }
    return sorted(divergent)


def describe(name, entries):
    """`'<origin> <short sha>, ...'` for every recorded entry of `name` -- diagnostics for a
    fixture that matched nothing, so the operator sees which boxes are on record and can tell a
    regenerated copy from a copy of the other box's."""
    recorded = entries.get(name) or []
    return ", ".join(f"{e.origin} {e.sha256[:12]}…" for e in recorded) or "nothing"


def missing_origins(name, entries, measurement_boxes):
    """Declared measurement boxes with no digest entry for `name`."""
    recorded = {e.origin for e in entries.get(name, [])}
    return sorted(set(measurement_boxes) - recorded)


def _main(argv=None):
    here = Path(__file__).parent
    ap = argparse.ArgumentParser(
        description="Record or check benchmark fixture digests (gh-ocannl-645, gh-ocannl-759).",
        epilog="--record pins fixtures that already exist, WITHOUT regenerating them: it is how "
        "a box states which bytes its published numbers were measured on when those bytes "
        "predate the digest file. Regenerating instead (gen_fixtures.py) changes the workload.",
    )
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--record",
        action="store_true",
        help="record the given fixtures (default: fixtures/*.safetensors) as this origin's bytes, "
        "without regenerating anything",
    )
    mode.add_argument(
        "--check", action="store_true", help="report each fixture's status against the record"
    )
    mode.add_argument(
        "--list-declared-measurement-boxes",
        action="store_true",
        help="print the explicit measurement-boxes header, one box per line; print nothing for "
        "a legacy file with no declaration",
    )
    ap.add_argument("fixtures", nargs="*", type=Path, help="fixture paths (default: all of them)")
    ap.add_argument(
        "--origin",
        default=None,
        help=f"the box these bytes are ({origin_default_help()}). Reports name it, "
        "so make it the name the reports use.",
    )
    ap.add_argument(
        "--adopt-legacy",
        metavar="BOX",
        default=None,
        help="with --record: attribute any pre-gh-ocannl-759 unattributed (three-field) lines to "
        "BOX, which you are asserting they belong to. One-shot migration; without it such a line "
        "is an error, since nothing can infer whose bytes it recorded. The fixtures on disk are "
        "recorded under --origin, which defaults to this host only when this host names itself "
        "BOX -- otherwise say both, so one box does not end up under two origin names.",
    )
    ap.add_argument("--digests", type=Path, default=None, help=f"path to {DIGEST_FILE}")
    ap.add_argument("--fixture-dir", type=Path, default=here / "fixtures")
    args = ap.parse_args(argv)
    if args.list_declared_measurement_boxes:
        if args.fixtures:
            ap.error("--list-declared-measurement-boxes reads only --digests; give no fixtures")
        if args.origin is not None or args.adopt_legacy is not None:
            ap.error("--list-declared-measurement-boxes does not record an origin")
        digests = args.digests or args.fixture_dir / DIGEST_FILE
        boxes = declared_measurement_boxes(digests) or []
        for box in sorted(boxes):
            print(box)
        return 0
    if args.adopt_legacy is not None:
        if args.check:
            ap.error("--adopt-legacy rewrites the file, so it belongs with --record, not --check")
    if args.check and args.origin is not None:
        # Silently accepting it would let automation believe `--check --origin "$BOX"` verified
        # that box's bytes, when the verdict is against every recorded origin.
        ap.error(
            "--check reports each fixture against EVERY recorded origin, so --origin has no "
            "effect there; drop it, or use --record to write bytes under an origin"
        )
        check_origin(args.adopt_legacy)

    digests = args.digests or args.fixture_dir / DIGEST_FILE
    fixtures = args.fixtures or sorted(args.fixture_dir.glob("*.safetensors"))
    if not fixtures:
        sys.exit(f"no fixtures given and none in {args.fixture_dir}; nothing to do")
    missing = [f for f in fixtures if not f.exists()]
    if missing:
        sys.exit("no such fixture: " + ", ".join(str(f) for f in missing))

    if args.check:
        entries = read_digests(digests)
        bad = 0
        for fx in fixtures:
            verdict, sha, size, origins = status(fx, entries)
            where = f" — {origins}'s bytes" if verdict == "MATCH" else ""
            print(f"{fx.name}: sha256 {sha} ({size} bytes) — {verdict}{where}")
            if verdict != "MATCH":
                print(f"    recorded: {describe(fx.name, entries)}")
                bad += 1
        return 1 if bad else 0

    origin = resolve_origin(args.origin, adopting=args.adopt_legacy)
    changes = record(digests, fixtures, origin, legacy_origin=args.adopt_legacy)
    print(f"recorded {len(fixtures)} fixture(s) in {digests} as origin {origin!r}")
    for name, org, was, now in changes:
        if was is None:
            print(f"  new: {name} sha256 {now.sha256} ({now.size} bytes) [{org}]")
        else:
            print(f"  CHANGED for {org}: {name}")
            print(f"    was  sha256 {was.sha256} ({was.size} bytes)")
            print(f"    now  sha256 {now.sha256} ({now.size} bytes)")
    if not changes:
        print("  no change: every fixture already matched its recorded entry for this origin")
    if any(was is not None for _, _, was, _ in changes):
        print(
            "  a changed fixture is a changed workload: reports measured on the previous digest "
            "are not comparable with reports measured on this one."
        )
    return 0


if __name__ == "__main__":
    sys.exit(_main())
