#!/usr/bin/env python3
"""The recorded identity of a benchmark fixture (gh-ocannl-645).

`benchmarks/fixtures/` is gitignored — the fixtures are large, regenerable artifacts — so no
checkout establishes a fixture's bytes. Without a recorded digest nothing in the repository says
which bytes a published report was measured on, and nothing can catch a fixture regenerated at a
different spec revision, or by a different numpy: the difference applies *uniformly* to every cell,
so the cross-cell parity gate (which compares cells with each other, not with the workload the
report names) certifies it exactly as it certifies the intended workload. Cross-session comparisons
— `report-gh569-hip.md`'s 46.65 ms denominator against `report-gh612-hip.md`'s 32.33 ms — are only
meaningful if both ran the same bytes, and that is the whole point of such a measurement.

So `gen_fixtures.py` records `fixtures/DIGESTS.txt` (checked in, unlike the fixtures themselves) as
it generates, `orchestrate.py` refuses to measure a fixture that does not match it, and every
result row and report states the digest its numbers are on. A deliberate regeneration rewrites the
file, which shows up as a reviewable diff rather than as silence.

This module is the one implementation of the file's format, shared by the generator, the
orchestrator and their tests; it deliberately imports nothing outside the standard library, so a
checkout without the benchmark venv can still read and check digests.
"""

import hashlib
from pathlib import Path

DIGEST_FILE = "DIGESTS.txt"

HEADER = """\
# Fixture digests: the bytes each published measurement is on (gh-ocannl-645).
#
# The fixtures themselves are gitignored, so this file is the only checked-in statement of
# what one contains. gen_fixtures.py rewrites the entries it regenerates; orchestrate.py
# refuses to measure a fixture that does not match one (--no-fixture-digest-check opts out),
# and stamps every result row and report section with the digest it ran on.
#
# A changed digest here is a changed workload: numbers measured before it are not comparable
# with numbers measured after it, whatever the report calls the workload. Fixture bytes depend
# on the workload spec, on gen_fixtures.py, and on the numpy version that drew the random
# streams (numpy does not promise Generator stream stability across releases) — so a mismatch
# names a real difference even when the spec is untouched.
#
# <sha256>  <bytes>  <name>
"""


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_digests(path):
    """`{name: (sha256, size)}` recorded in `path`; an absent file records nothing."""
    path = Path(path)
    entries = {}
    if not path.exists():
        return entries
    for lineno, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 3:
            raise ValueError(f"{path}:{lineno}: expected '<sha256>  <bytes>  <name>', got {line!r}")
        sha, size, name = fields
        entries[name] = (sha, int(size))
    return entries


def write_digests(path, entries):
    """Rewrite `path` with `entries` ({name: (sha256, size)}), name-sorted under the header."""
    body = "".join(f"{sha}  {size}  {name}\n" for name, (sha, size) in sorted(entries.items()))
    Path(path).write_text(HEADER + body)


def record(path, fixtures):
    """Record `fixtures` (paths) into the digest file at `path`, keeping every other entry.

    Returns the list of `(name, was, now)` changes, where `was` is None for a newly recorded
    fixture — the caller says so out loud: a silently rewritten digest is the failure this file
    exists to prevent, and a regeneration is where it would happen.
    """
    entries = read_digests(path)
    changes = []
    for fixture in fixtures:
        fixture = Path(fixture)
        now = (sha256_file(fixture), fixture.stat().st_size)
        was = entries.get(fixture.name)
        if was != now:
            changes.append((fixture.name, was, now))
        entries[fixture.name] = now
    write_digests(path, entries)
    return changes


def status(fixture, entries):
    """`(verdict, sha256, size)` of `fixture` against recorded `entries`.

    Verdict is MATCH, MISMATCH (recorded, different bytes) or UNRECORDED (no entry — a fixture
    nothing in the repository describes, which is what a report must not quietly be measured on).
    """
    fixture = Path(fixture)
    sha, size = sha256_file(fixture), fixture.stat().st_size
    recorded = entries.get(fixture.name)
    if recorded is None:
        return "UNRECORDED", sha, size
    return ("MATCH" if recorded == (sha, size) else "MISMATCH"), sha, size
