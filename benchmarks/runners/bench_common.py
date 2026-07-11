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
