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

    python3 fixture_digest.py --record            # pin fixtures/*.safetensors as this box's
    python3 fixture_digest.py --check             # what is on disk, against what is recorded
"""

import argparse
import hashlib
import platform
import sys
from collections import namedtuple
from pathlib import Path

DIGEST_FILE = "DIGESTS.txt"

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
# The <origin> field names the box whose fixtures these bytes are, because the same workload
# has different bytes on different boxes and the reports have to say which. Two entries under
# one name are two boxes' bytes, NOT a history: numbers measured on one are not comparable
# with numbers measured on the other. Regenerating replaces only the recording box's entry --
# regeneration is a cross-box event and has to be coordinated across every origin listed here,
# or the boxes silently diverge again (see benchmarks/README.md).
#
# <sha256>  <bytes>  <name>  <origin>
"""


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def this_origin():
    """Default origin: the box doing the recording, which is the fact being recorded."""
    return platform.node() or "unknown-host"


def check_origin(origin):
    """Origins are whitespace-free (the format is whitespace-split) and non-empty."""
    if not origin or origin.split() != [origin]:
        raise ValueError(f"origin must be a single non-empty word without whitespace, got {origin!r}")
    return origin


def read_digests(path):
    """`{name: [Entry, ...]}` recorded in `path`; an absent file records nothing.

    Entries under one name are origin-sorted, and one origin appears at most once per name.
    """
    path = Path(path)
    entries = {}
    if not path.exists():
        return entries
    for lineno, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) == 3:
            # The pre-gh-ocannl-759 three-field format. Not silently adopted under some guessed
            # origin: an unattributed digest is precisely the "which box is this?" silence this
            # file exists to break, and the two boxes' bytes really do differ, so a guess would
            # be a coin flip recorded as a fact.
            raise ValueError(
                f"{path}:{lineno}: {line!r} is the old unattributed format; re-record it with "
                "`python3 fixture_digest.py --record --origin <box>` so the entry says whose "
                "bytes it is (gh-ocannl-759)"
            )
        if len(fields) != 4:
            raise ValueError(
                f"{path}:{lineno}: expected '<sha256>  <bytes>  <name>  <origin>', got {line!r}"
            )
        sha, size, name, origin = fields
        by_origin = entries.setdefault(name, {})
        if origin in by_origin:
            raise ValueError(
                f"{path}:{lineno}: {name} is recorded twice for origin {origin!r}; one box has "
                "one set of bytes per fixture, so this file cannot say which"
            )
        by_origin[origin] = Entry(sha, int(size), origin)
    return {name: [by_origin[o] for o in sorted(by_origin)] for name, by_origin in entries.items()}


def write_digests(path, entries):
    """Rewrite `path` with `entries` ({name: [Entry]}), sorted by name then origin."""
    body = "".join(
        f"{e.sha256}  {e.size}  {name}  {e.origin}\n"
        for name in sorted(entries)
        for e in sorted(entries[name], key=lambda e: e.origin)
    )
    Path(path).write_text(HEADER + body)


def record(path, fixtures, origin=None):
    """Record `fixtures` (paths) as `origin`'s bytes in the digest file at `path`.

    Every other entry is kept -- including other origins' entries for the same fixture, which is
    the point: a box regenerating its own copy must not evict the recording that another box's
    published numbers rest on.

    Returns the list of `(name, origin, was, now)` changes, where `was` is None for a fixture this
    origin had not recorded. The caller says so out loud: a silently rewritten digest is the
    failure this file exists to prevent, and a regeneration is where it would happen.
    """
    origin = check_origin(origin or this_origin())
    entries = read_digests(path)
    changes = []
    for fixture in fixtures:
        fixture = Path(fixture)
        now = Entry(sha256_file(fixture), fixture.stat().st_size, origin)
        others = [e for e in entries.get(fixture.name, []) if e.origin != origin]
        was = next((e for e in entries.get(fixture.name, []) if e.origin == origin), None)
        if was != now:
            changes.append((fixture.name, origin, was, now))
        entries[fixture.name] = others + [now]
    write_digests(path, entries)
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


def other_origins(path, names, origin):
    """Origins other than `origin` recorded in `path` for any of `names`, sorted.

    What a regenerating box has to be told: their entries survive (so their fixtures still pass
    the gate), but their bytes are now a different workload from the one just generated, and
    nothing else will say so until someone compares two reports.
    """
    entries = read_digests(path)
    return sorted(
        {e.origin for name in names for e in entries.get(name, []) if e.origin != origin}
    )


def describe(name, entries):
    """`'<origin> <short sha>, ...'` for every recorded entry of `name` -- diagnostics for a
    fixture that matched nothing, so the operator sees which boxes are on record and can tell a
    regenerated copy from a copy of the other box's."""
    recorded = entries.get(name) or []
    return ", ".join(f"{e.origin} {e.sha256[:12]}…" for e in recorded) or "nothing"


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
    ap.add_argument("fixtures", nargs="*", type=Path, help="fixture paths (default: all of them)")
    ap.add_argument(
        "--origin",
        default=None,
        help=f"the box these bytes are (default: this host, {this_origin()!r}). Reports name it, "
        "so make it the name the reports use.",
    )
    ap.add_argument("--digests", type=Path, default=None, help=f"path to {DIGEST_FILE}")
    ap.add_argument("--fixture-dir", type=Path, default=here / "fixtures")
    args = ap.parse_args(argv)

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

    changes = record(digests, fixtures, args.origin)
    origin = args.origin or this_origin()
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
