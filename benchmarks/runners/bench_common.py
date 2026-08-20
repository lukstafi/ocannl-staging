"""Shared helpers for the Python benchmark runners."""

import json
import struct


def read_st_metadata(path):
    """Read the __metadata__ string map from a safetensors file header."""
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
    return header.get("__metadata__", {})


def percentiles(xs):
    s = sorted(xs)

    def p(q):
        return s[round(q / 100 * (len(s) - 1))]

    return {"p10": p(10), "p50": p(50), "p90": p(90)}


def emit(result):
    print(json.dumps(result), flush=True)


# --- Search provenance (gh-ocannl-644) ---------------------------------------------------
#
# Whether THIS process ran its own kernel search / codegen or replayed a cache. OCANNL's tuned
# cell splits the two into separate processes, because a searching process is measurably slower
# per launch; tinygrad's BEAM cell and torch.compile search in the process that then times
# steps. Whether that costs them anything is unmeasured (gh-ocannl-675) — until it is, every
# runner at least SAYS which it did, so the report does not imply the question applies to
# OCANNL alone.
#
# Both probes read framework internals, so they answer None ("cannot tell", reported as UNKNOWN)
# rather than guess. A wrong False is exactly the silent claim the field exists to prevent, so
# a probe that saw nothing at all when it should have seen something says so.


def instrument_tinygrad_beam():
    """Start counting tinygrad's beam-search disk-cache reads and writes.

    Returns the counts dict (live, read it after the steps) or None if this tinygrad cannot be
    instrumented. Call it before the first step: the search happens on the first kernel launch,
    and `tinygrad.engine.search` binds the cache helpers by from-import, so the wrappers have to
    go on *that* module's attributes rather than on `tinygrad.helpers`.
    """
    try:
        from tinygrad.engine import search
    except Exception:
        return None
    if not (hasattr(search, "diskcache_get") and hasattr(search, "diskcache_put")):
        return None
    counts = {"hit": 0, "put": 0}
    get, put = search.diskcache_get, search.diskcache_put

    def get_counting(table, key, *args, **kwargs):
        val = get(table, key, *args, **kwargs)
        if table == "beam_search" and val is not None:
            counts["hit"] += 1
        return val

    def put_counting(table, key, val, *args, **kwargs):
        if table == "beam_search":
            counts["put"] += 1
        return put(table, key, val, *args, **kwargs)

    search.diskcache_get, search.diskcache_put = get_counting, put_counting
    return counts


def tinygrad_searched(counts, beam):
    """The `searched` field for a tinygrad cell: did this process run a beam search?

    A cache write means it did; only reads mean every beam result was replayed from
    `~/.cache/tinygrad`. Beam requested but nothing observed means the internals moved under the
    probe — that is UNKNOWN, never False.
    """
    if not beam:
        return False
    if not counts or (counts["hit"] == 0 and counts["put"] == 0):
        return None
    return counts["put"] > 0


def torch_searched(torch, compiled):
    """The `searched` field for a pytorch cell: did this process run inductor's codegen?

    An eager cell compiles nothing. For a compiled one the FX-graph cache counters say whether
    codegen actually ran or was served from `~/.cache/torch/inductor`; a torch that reports
    neither cannot answer.
    """
    if not compiled:
        return False
    try:
        counters = torch._dynamo.utils.counters["inductor"]
        hits = counters.get("fxgraph_cache_hit", 0)
        misses = counters.get("fxgraph_cache_miss", 0)
    except Exception:
        return None
    if not hits and not misses:
        return None
    return misses > 0
