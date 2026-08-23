"""Shared helpers for the Python benchmark runners."""

import json
import math
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


def json_safe(obj):
    """`obj` with every non-finite float replaced by None, recursively.

    A diverged training run is exactly the run whose loss trajectory the report needs, and
    `json.dumps` writes it as `NaN` / `Infinity` -- tokens JSON does not have. Python's own loader
    accepts them, so the sweep survives, but `results.jsonl` then holds a line no other JSON reader
    will take, and the OCANNL runner emitting the same fact as `nan` had its whole cell dropped
    (gh-ocannl-676). `null` is what all three runners emit for a number they have and JSON cannot
    express; `orchestrate.py` reads it as "ran, and this number is not a number".
    """
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    return obj


def emit(result):
    # allow_nan=False so that a non-finite value json_safe did not reach -- one behind a type it
    # does not walk -- raises here rather than being written as an unparseable token.
    print(json.dumps(json_safe(result), allow_nan=False), flush=True)


# --- Search provenance (gh-ocannl-644) ---------------------------------------------------
#
# Whether THIS process ran its own kernel search / codegen or replayed a cache. OCANNL's tuned
# cell splits the two into separate processes, because a searching process is measurably slower
# per launch; tinygrad's BEAM cell and torch.compile search in the process that then times
# steps. Whether that costs them anything is unmeasured (gh-ocannl-675); until it is, every
# runner at least SAYS which it did, so the report does not imply the question applies to
# OCANNL alone.
#
# Both probes read framework internals, so they answer None ("cannot tell", reported as UNKNOWN)
# rather than guess. A wrong False is exactly the silent claim the field exists to prevent, so
# a probe that saw nothing at all when it should have seen something says so.


def instrument_tinygrad_beam():
    """Start counting tinygrad's beam searches and their disk-cache reads and writes.

    Returns the counts dict (live, read it after the steps) or None if this tinygrad cannot be
    instrumented. Call it before the first step: the search happens on the first kernel launch,
    and `tinygrad.engine.search` binds the cache helpers by from-import, so the wrappers have to
    go on *that* module's attributes rather than on `tinygrad.helpers`.

    Calls are counted as well as cache traffic because the cache is not the only regime: under
    CACHELEVEL=0 or IGNORE_BEAM_CACHE a search runs and neither reads nor writes, which cache
    counting alone reports as "cannot tell".
    """
    # tinygrad moved the beam search out of `tinygrad.engine.search` into
    # `tinygrad.codegen.opt.search` (0.13); try both, newest first, so the probe answers on
    # either. A layout it does not know still answers None rather than guessing.
    search = None
    for mod in ("tinygrad.codegen.opt.search", "tinygrad.engine.search"):
        try:
            search = __import__(mod, fromlist=["_"])
        except Exception:
            continue
        if hasattr(search, "diskcache_get") and hasattr(search, "diskcache_put"):
            break
        search = None
    if search is None:
        return None
    counts = {"call": 0, "hit": 0, "put": 0}
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

    beam = getattr(search, "beam_search", None)
    if callable(beam):

        def beam_counting(*args, **kwargs):
            counts["call"] += 1
            return beam(*args, **kwargs)

        search.beam_search = beam_counting
    return counts


def tinygrad_searched(counts, beam):
    """The `searched` field for a tinygrad cell: did this process run a beam search?

    Three outcomes, not two. A cache write means it searched, and so does a search call that read
    no cached entry (the uncached regimes above). Calls that all came back from
    `~/.cache/tinygrad`, or reads with nothing else observed, mean every beam result was replayed.
    Beam requested and nothing at all observed means the internals moved under the probe: that is
    UNKNOWN, never False.
    """
    if not beam:
        return False
    if not counts:
        return None
    if counts["put"] or counts["call"] > counts["hit"]:
        return True
    if counts["hit"] or counts["call"]:
        return False
    return None


# Inductor's FX-graph cache outcomes, by what they say about THIS process. A bypass is a graph
# the cache refused to serve, so it was generated here exactly as a miss was; reading only
# hit-vs-miss calls a run with one hit and one bypass a replay, which is a compiled graph this
# process paid for and the report would not show.
TORCH_CODEGEN_COUNTERS = ("fxgraph_cache_miss", "fxgraph_cache_bypass")
TORCH_REPLAY_COUNTERS = ("fxgraph_cache_hit",)


def torch_searched(torch, compiled):
    """The `searched` field for a pytorch cell: did this process run inductor's codegen?

    An eager cell compiles nothing. For a compiled one the FX-graph cache counters say whether any
    graph was generated here or all of them came from `~/.cache/torch/inductor`; a torch that
    reports none of them cannot answer. A run is mixed as soon as it has more than one graph, so
    the question is whether ANY graph was generated here, not whether the last one was.
    """
    if not compiled:
        return False
    try:
        counters = torch._dynamo.utils.counters["inductor"]
        codegen = sum(counters.get(name, 0) for name in TORCH_CODEGEN_COUNTERS)
        replayed = sum(counters.get(name, 0) for name in TORCH_REPLAY_COUNTERS)
    except Exception:
        return None
    if codegen:
        return True
    return False if replayed else None
