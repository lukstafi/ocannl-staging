#!/usr/bin/env python3
"""Run the cross-framework benchmark matrix (OCANNL / PyTorch / tinygrad) and report.

For each fixture in fixtures/, runs every (framework, backend, variant, precision) cell,
collects the JSON result lines, enforces the loss-trajectory parity gate against the PyTorch
CPU reference, and writes results/results.jsonl plus a markdown report.

Scheduling (`default` / `materialized` / `tuned`) and storage precision (`f32` / `bf16` /
`f16`) are INDEPENDENT axes of an OCANNL cell, and the matrix is their product
(gh-ocannl-539). Overloading one variant string made tuned reduced-precision cells
inexpressible, which is where tensor cores would show at all on backends whose only mma route
is a reduced input format (RDNA3/3.5 WMMA has no f32-input shape).

Timing results are only comparable when the parity gate passes: a FAIL means that cell was
not computing the same training trajectory, so its step times are flagged, not compared.

A requested cell a workload cannot express — a reduced precision on the conv runner, an f16
gate-cost leg on a forward-only workload — is reported as NOT APPLICABLE with its reason, in
the run log and in a report section, rather than quietly not being run (gh-ocannl-551).
"""

import argparse
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

import fixture_digest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
# BENCH_VENV_PY overrides the venv interpreter — for environments where benchmarks/.venv
# is unusable (e.g. deep worktree paths hitting Windows MAX_PATH during torch install).
VENV_PY = Path(
    os.environ.get(
        "BENCH_VENV_PY",
        HERE / ".venv/Scripts/python.exe"
        if (HERE / ".venv/Scripts/python.exe").exists()
        else HERE / ".venv/bin/python",
    )
)
# BENCH_CELL_LOG_DIR: keep every cell's raw combined output under this directory, one file per
# cell label. Unset (the default) discards a successful cell's output as before.
CELL_LOG_DIR = (
    Path(os.environ["BENCH_CELL_LOG_DIR"]) if os.environ.get("BENCH_CELL_LOG_DIR") else None
)
PARITY_TOL = 2e-3
# Accuracy-parity gates for the OCANNL mixed-precision legs (gh-ocannl-492 task 4), with roughly
# 10x headroom over the largest drift measured by the macOS cc/Metal sweep.
PARITY_TOL_PRECISION = {"bf16": 4e-3, "f16": 2e-3}
# A parity tolerance cannot reject an input-independent forward when the reference itself moves
# slowly. Require at least one part per million of relative loss variation over the parity window.
LOSS_MOVE_MIN_REL = 1e-6
REFERENCE = ("pytorch", "cpu", "eager")
# Cells whose protocol splits the search and the timing into two processes (gh-ocannl-644): the
# search leaves an OCANNL process measurably slower per launch, so step times from it are not the
# artifact's. These are the only cells a searching process is a violation for.
TWO_PASS_CELLS = {("ocannl", "tuned")}
# Cells that search or compile in the process that then times steps — by protocol, not by
# mistake. Whether that costs them anything is unmeasured (gh-ocannl-675), so they are stated in
# the report and never gated; the alternative, saying nothing, reads as though the question
# applied to OCANNL alone.
SAME_PROCESS_CELLS = {("tinygrad", "beam"), ("pytorch", "compiled")}
# How a cell's measurement pass is rendered in the report's `pass` column (gh-ocannl-644). A
# runner predating the `searched` field, or one whose probe could not read its framework's
# internals, cannot answer — which is not the same as a violation.
PROVENANCE_MARK = {
    "REPLAY": "replay",
    "SEARCH-PASS": "**SEARCH PASS**",
    "NO-SEARCH": "no search",
    "SAME-PROCESS": "same-process",
    "CACHED": "cached",
    "UNKNOWN": "?",
}
# Report row order within a workload: precision-major, f32 first (it is the reference's precision
# and every non-OCANNL cell's), then the reduced precisions; a gate-cost leg sorts directly after
# the storage precision it varies; p50-ascending within each group.
PRECISION_ORDER = ["f32", "bf16", "f16"]
# The precision axis is a property of the runner, not of OCANNL: only bench_mlp and bench_gpt
# implement BENCH_PRECISION, so reduced-precision cells are generated for those models only.
PRECISION_MODELS = ("mlp", "gpt")
# The gh-ocannl-492 task-5 gate-cost legs: f16 with the dynamic loss-scaling gate replaced by a
# fixed scale (no gate, no host read) or by the fused on-device gate sampled every N steps. They
# vary how the optimizer's inf/nan gate is paid for, so they need a workload that HAS an optimizer
# — on a forward-only fixture they are reported as not applicable rather than silently dropped
# (gh-ocannl-551).
GATE_LEG_ENV = {"f16-static": {"BENCH_STATIC_SCALE": "1"}}
GATED_RE = re.compile(r"^f16-gated([1-9][0-9]*)$")

# The GPU column of the matrix, per --gpu choice: OCANNL backend, PyTorch device,
# tinygrad device. The CPU column (cc / cpu / CPU) is always run.
GPU_DEVICES = {
    "metal": ("metal", "mps", "METAL"),
    "cuda": ("cuda", "cuda", "CUDA"),
    # OCANNL's hip backend (AMD ROCm/HIP). PyTorch exposes HIP as its "cuda" device — on
    # Linux ROCm and, since ROCm 7.2, on Windows via AMD's official wheels (main() probes
    # torch.version.hip and drops the column when the venv's torch is not a HIP build).
    # tinygrad uses AMD on Linux ROCm and its OpenCL device (CL) on Windows.
    "hip": ("hip", "cuda", "AMD" if platform.system() == "Linux" else "CL"),
    "none": (None, None, None),
}

# Known-pathological cells excluded from the default matrix, as
# (workload, backend, variant, precision), where precision None means "at every precision" — a
# scheduling pathology has no dependence on the storage precision, so adding a precision axis must
# not let one back in through the bf16/f16 columns. Written as set(), not {}, so that emptying it
# does not silently turn it into a dict.
# Currently empty: the metal-default-schedule pathologies (gpt2_mini 81 s/step -> ~0.3 s,
# lenet 3.2 s/step + parity FAIL -> 0.22 s exact, mlp_wide >10 s/step -> 6 ms) were fixed by
# lowering the default GPU schedule's serial-fallback threshold, promoting statement-crossing
# Local scratch at fission, and working around a Metal compiler miscompilation of scalar
# read-modify-write accumulation (see arrayjit/lib/c_syntax.ml volatile_scalar_rmw and
# benchmarks/runners/ocannl/bench_metal_bug.ml).
# cifar_conv metal/tuned USED to belong here: the search completed but the post-tune re-init hung
# the process (Metal reinit-after-tune race, PR #109/#174). Retested at gh-ocannl-538 with the
# byte-for-byte command orchestrate issues for that cell — completed in ~4 min wall, no hang,
# search 230.3 s against the 2069 s the same standalone search took in the gh-476 sweep, and the
# full sweep reproduced its p50 to 0.005% (277.915 vs 277.928 ms). The searches got cheap enough to
# stop provoking it somewhere between those sweeps; gh-ocannl-532 (an unparallelized candidate is no
# longer dispatched on a GPU backend) is the likeliest reason.
SKIP_CELLS = set(
    # mlp_wide/cifar_conv/cifar_stride hip/tuned USED to belong here: the search appeared to wedge,
    # but perf showed 100% of the time in libhsa-runtime64 busy-waiting on a dispatch — the
    # autotuner was timing the unparallelized serial baseline, i.e. the whole training step in one
    # work-item, four times (warmup plus autotune_repeats). Hours per run on gfx1151, with
    # Windows-side driver timeouts and a lost display (gh-ocannl-532). Fixed in the tuner: an
    # unparallelized candidate is no longer dispatched on a GPU backend. Confirmed on the machine
    # that produced the symptom (gfx1151 / WSL, ROCm), all three cells completing with the display
    # intact: mlp_wide 34 s wall (search 32.1 s, 1.70 ms/step), cifar_stride 132 s (125.3 s,
    # 18.06 ms), cifar_conv 181 s (164.6 s, 61.09 ms).
)

sys.path.insert(0, str(HERE / "runners"))
from bench_common import read_st_metadata  # noqa: E402


def cell_skipped(workload, backend, variant, precision):
    """Whether a cell is in SKIP_CELLS, honouring the None-precision wildcard."""
    return (workload, backend, variant, precision) in SKIP_CELLS or (
        workload,
        backend,
        variant,
        None,
    ) in SKIP_CELLS


def cell_name(variant, precision):
    """Display name of an OCANNL cell's (scheduling, precision) pair.

    f32 cells keep their bare variant name, so the labels and reports of an f32-only matrix are
    unchanged; a reduced-precision cell is named by the product, e.g. `tuned/bf16`.
    """
    return variant if precision == "f32" else f"{variant}/{precision}"


def precision_base(precision):
    """The storage precision a cell computes in, without a gate-leg suffix."""
    return precision.split("-", 1)[0]


def precision_rank(precision):
    base = precision_base(precision)
    rank = (
        PRECISION_ORDER.index(base) if base in PRECISION_ORDER else len(PRECISION_ORDER)
    )
    # A gate leg sorts right after the plain cell of the same storage precision: it is a variant
    # of how that precision's optimizer step is gated, not a precision of its own.
    return rank * 10 + (0 if precision == base else 1)


def parity_tol(precision):
    return PARITY_TOL_PRECISION.get(precision_base(precision), PARITY_TOL)


def precision_spec(spec):
    """--precision argument: a storage precision, or one of the f16 gate-cost legs."""
    if spec in ("bf16", "f16") or spec in GATE_LEG_ENV or GATED_RE.match(spec):
        return spec
    raise argparse.ArgumentTypeError(
        f"unknown precision {spec!r}; expected bf16, f16, f16-static, or f16-gatedN"
    )


def precision_env(precision):
    """The BENCH_* environment a precision cell is dispatched with."""
    env = {"BENCH_PRECISION": precision_base(precision)}
    env.update(GATE_LEG_ENV.get(precision, {}))
    gated = GATED_RE.match(precision)
    if gated:
        env["BENCH_GATE_INTERVAL"] = gated.group(1)
    return env


def cell_env(base, fixture, variant, precision):
    """The environment an OCANNL cell is dispatched with."""
    env = dict(
        base,
        BENCH_FIXTURE=str(fixture),
        BENCH_TUNE="1" if variant == "tuned" else "0",
        BENCH_MATERIALIZE="1" if variant == "materialized" else "0",
        # A gate leg carries its own BENCH_* flags; clear the others so a stray value from the
        # caller's environment cannot leak into a cell.
        BENCH_STATIC_SCALE="0",
        BENCH_GATE_INTERVAL="0",
    )
    # Applied after, not as keywords: a gate leg's flags collide with the cleared defaults above
    # and dict() rejects duplicate keywords.
    env.update(precision_env(precision))
    return env


def precision_unavailable(model, mode, precision):
    """Why this workload cannot express this precision cell, or None if it can.

    An unavailable cell is reported (in the run log and the report) instead of quietly not being
    run: an empty row and an unrun row are otherwise indistinguishable (gh-ocannl-551).
    """
    if model not in PRECISION_MODELS:
        return f"the {model} runner has no BENCH_PRECISION support"
    if (precision in GATE_LEG_ENV or GATED_RE.match(precision)) and mode != "train":
        return (
            "gate-cost legs measure the optimizer's loss-scaling gate; this workload is "
            "forward-only (mode: infer)"
        )
    return None


def ocannl_exe(model):
    return ROOT / f"_build/default/benchmarks/runners/ocannl/bench_{model}.exe"


def run_cell(label, cmd, env=None, cwd=None):
    print(f"--- {label}", flush=True)
    proc = subprocess.run(
        cmd, env=env, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    if CELL_LOG_DIR:
        # A cell's own output is otherwise discarded on success, which throws away exactly the
        # evidence a measurement sweep is asked to report: with autotune_log=true the search
        # pass's candidate lines (seeded vs timed, FAILED, dedup, split-reduce evictions) live
        # here and nowhere else, and re-running the searches to recover them costs as much as the
        # sweep. Off unless BENCH_CELL_LOG_DIR is set, so the default run is unchanged.
        CELL_LOG_DIR.mkdir(parents=True, exist_ok=True)
        safe = "".join(c if c.isalnum() or c in "-._" else "_" for c in label)
        (CELL_LOG_DIR / f"{safe}.log").write_text(proc.stdout)
    line = next(
        (l for l in reversed(proc.stdout.splitlines()) if l.startswith("{")), None
    )
    if proc.returncode != 0 or line is None:
        print(proc.stdout[-4000:])
        print(f"!!! {label} failed (exit {proc.returncode})", flush=True)
        return None
    result = json.loads(line)
    print(
        f"    p50 {num(result['step_ms']['p50'], '.3f')} ms, "
        f"compile {num(result['compile_s'], '.2f')} s",
        flush=True,
    )
    return result


def finite(x):
    """Whether a JSON number from a result line is a real number this report can compare.

    A runner emits `null` for a value it has but cannot express in JSON -- a diverged loss, a time
    it never measured (gh-ocannl-676) -- so `None` reaching here means "ran, and this number is not
    a number", not "runner failed". `NaN` is accepted by Python's `json.loads` (its non-standard
    extension) and so can still arrive from an older result file, and is the same fact.
    """
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def diverged_at(losses):
    """The index of the first non-finite loss, or None for a trajectory that stayed finite."""
    return next((i for i, loss in enumerate(losses) if not finite(loss)), None)


def finite_prefix(losses):
    """The steps before the trajectory left the finite numbers.

    Nothing after the first non-finite loss is evidence about the run: a finite value FOLLOWING a
    NaN is whatever the arithmetic settled on afterwards, not drift the reference can be compared
    against, and not movement. Both readers of a trajectory take this prefix, so neither can report
    a number the DIVERGED verdict says is meaningless (gh-ocannl-676).
    """
    cut = diverged_at(losses)
    return losses if cut is None else losses[:cut]


def json_safe(obj):
    """`obj` with every non-finite float replaced by None, recursively.

    `json.dumps` writes `NaN` / `Infinity` for those, which its own loader accepts and no other
    JSON reader does -- results.jsonl is read by jq, by pandas and by the next session's scripts,
    so what this sweep writes has to be JSON.
    """
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    return obj


def num(x, spec):
    """A number for the report table, or `n/a` when the runner had none to give."""
    return format(x, spec) if finite(x) else "n/a"


def loss_moved(losses):
    """Whether a loss trajectory has more than floating-point-noise-level variation.

    Over the prefix before the first non-finite step: a diverged trajectory is not a stationary
    one, and reporting it as stationary names the wrong defect. A prefix shorter than two steps has
    nothing to say about movement, and the cell is reported as diverged instead.
    """
    prefix = finite_prefix(losses)
    if len(prefix) < 2:
        return False
    scale = max(max(abs(loss) for loss in prefix), 1e-6)
    return max(prefix) - min(prefix) > LOSS_MOVE_MIN_REL * scale


def parity_check(results):
    """Annotate each result with parity vs the reference run of the same workload."""
    by_workload = {}
    for r in results:
        by_workload.setdefault(r["workload"], []).append(r)
    for workload, rs in by_workload.items():
        ref = next(
            (r for r in rs if (r["framework"], r["backend"], r["variant"]) == REFERENCE),
            None,
        )
        ref_prefix = finite_prefix(ref["losses"]) if ref is not None else []
        ref_diverged = ref is not None and len(ref_prefix) < len(ref["losses"])
        for r in rs:
            r["parity_loss_moved"] = loss_moved(r["losses"])
            r["diverged_at"] = diverged_at(r["losses"])
            if ref is not None:
                # Compared over the prefix both trajectories reached while still finite, so a
                # DIVERGED row's parity_max_rel is drift measured BEFORE it went and nothing else.
                n = min(len(finite_prefix(r["losses"])), len(ref_prefix))
                if n:
                    r["parity_max_rel"] = max(
                        abs(a - b) / max(abs(b), 1e-6)
                        for a, b in zip(r["losses"][:n], ref["losses"][:n])
                    )
            if r["diverged_at"] is not None:
                # The cell ran and its training blew up: a gate failure naming its cause, not a
                # missing cell and not a stationary one (gh-ocannl-676). Parity is meaningless past
                # the divergence, so what is reported is where it happened -- and, when there is a
                # finite prefix shared with the reference, how far it had drifted before it did.
                r["parity"] = "DIVERGED"
            elif ref is None or ref_diverged:
                # A diverged reference is no reference: comparing against a trajectory that blew up
                # would report the reference's defect as every other cell's.
                r["parity"] = "NO-REF"
            elif r is ref:
                r["parity"] = "REF"
                r["parity_max_rel"] = 0.0
            else:
                max_rel = r.get("parity_max_rel")
                tol = parity_tol(r.get("precision", "f32"))
                r["parity"] = (
                    "PASS"
                    if max_rel is not None and max_rel < tol and r["parity_loss_moved"]
                    else "FAIL"
                )


def check_fixture_digests(fixtures, digests_path=None, allow_unpinned=False):
    """What the fixtures ARE, checked before anything is measured on them (gh-ocannl-645).

    Returns `{fixture path: sha256}` for stamping onto the results. Exits on a fixture whose
    bytes are not the ones `fixtures/DIGESTS.txt` records unless `allow_unpinned` — a
    differently generated fixture is consumed uniformly by every cell, so the cross-cell parity
    gate certifies it exactly as it certifies the intended workload, and the digest is the only
    thing that ties a published number to the workload the report names.
    """
    entries = fixture_digest.read_digests(
        digests_path or HERE / "fixtures" / fixture_digest.DIGEST_FILE
    )
    shas = {}
    unpinned = []
    for fx in fixtures:
        verdict, sha, size = fixture_digest.status(fx, entries)
        shas[fx] = sha
        print(f"fixture {fx.name}: sha256 {sha} ({size} bytes) — {verdict}", flush=True)
        if verdict != "MATCH":
            unpinned.append((fx, verdict))
    named = ", ".join(f"{fx.name} ({verdict})" for fx, verdict in unpinned)
    if unpinned and not allow_unpinned:
        sys.exit(
            f"refusing to measure {named}: these bytes are not the ones "
            "fixtures/DIGESTS.txt records, so a report of them would name a workload nothing "
            "pins. Re-run gen_fixtures.py to regenerate and re-record (the digest diff is the "
            "review), or pass --no-fixture-digest-check to measure them as they are."
        )
    if unpinned:
        print(
            f"MEASURING UNPINNED FIXTURES: {named} — results carry their measured digest, but "
            "nothing checked in describes them",
            flush=True,
        )
    return shas


def search_provenance(result):
    """What one OCANNL process did about searching: SEARCHED, REPLAY, NO-SEARCH or UNKNOWN.

    The single reading of `searched`, shared by every consumer, because the fact is three-valued
    and each boolean spelling of it has been wrong in a different place (gh-ocannl-644 review):
    a process can run a search, replay cached winners, or — under `autotune_search=false`, the
    reproducible profile — do neither and ship the untuned default. `not searched` therefore does
    not mean "replayed", which is what a `(cached)` label on its compile cost would claim.

    The `tune` object's totals are the evidence separating the second case from the third: an
    OCANNL cell that tuned anything reports them, and without them there is nothing to tell those
    two apart with, so the answer is UNKNOWN rather than a guess.
    """
    searched = result.get("searched")
    if searched is None:
        return "UNKNOWN"
    if searched:
        return "SEARCHED"
    tune = result.get("tune")
    if not tune:
        return "UNKNOWN"
    return "NO-SEARCH" if not tune.get("searches") and not tune.get("replays") else "REPLAY"


# How a two-pass cell's own verdict maps to the report's `pass` column: only a searching process
# is a protocol violation, and only a genuine replay may call the carried-over compile cost cached.
TWO_PASS_VERDICT = {
    "SEARCHED": "SEARCH-PASS",
    "REPLAY": "REPLAY",
    "NO-SEARCH": "NO-SEARCH",
    "UNKNOWN": "UNKNOWN",
}
# What a search pass's verdict says about the `compile s` carried over from it.
COMPILE_S_NOTE = {"REPLAY": " (cached)", "NO-SEARCH": " (no search)"}


def provenance_check(results):
    """Annotate each searching cell with which process produced its step times (gh-ocannl-644).

    OCANNL's two-pass protocol exists because a searching process is measurably slower per launch,
    so pass-1 step times understate the tuned artifact. Both passes emit the same
    framework/backend/variant/precision, and until the runner carried `searched` nothing in the
    artifact distinguished them: `report-gh612-hip.md` quoted pass-1 timings for fifteen revisions
    and only a reviewer reading the driver caught it.

    A two-pass cell is REPLAY (compliant) or SEARCH-PASS (its timings came from a process that
    searched — returned as a violation). NO-SEARCH is the third case: with `autotune_search=false`
    (the reproducible profile) a tuned cell neither searches nor replays, it ships the untuned
    default — nothing to gate, but calling that a replay would credit the row with a tuned artifact
    it does not have. A cell whose protocol searches in the timing process is
    SAME-PROCESS or, when it replayed its framework's own cache instead, CACHED; neither is a
    violation, because nothing yet says those frameworks pay for it (gh-ocannl-675) — the point of
    annotating them is that the report stops implying the question is OCANNL's alone. UNKNOWN is a
    runner predating the field, or a probe that could not read its framework's internals.

    Cells that neither search nor compile — eager, jit, an untuned OCANNL default — get no
    annotation: there is no process distinction for them to be on the wrong side of.
    """
    violations = []
    for r in results:
        cell = (r.get("framework"), r.get("variant"))
        two_pass = cell in TWO_PASS_CELLS
        if not two_pass and cell not in SAME_PROCESS_CELLS:
            continue
        searched = r.get("searched")
        if searched is None:
            r["provenance"] = "UNKNOWN"
        elif two_pass:
            verdict = search_provenance(r)
            r["provenance"] = TWO_PASS_VERDICT[verdict]
            if verdict == "SEARCHED":
                violations.append(r)
        else:
            r["provenance"] = "SAME-PROCESS" if searched else "CACHED"
    return violations


def report(results, out_dir, unavailable=()):
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.jsonl", "w") as f:
        for r in results:
            # allow_nan=False so a non-finite value this sweep computed itself cannot slip out as
            # `NaN`: json_safe has already mapped the ones it expects, and anything left is a bug
            # worth raising over rather than writing an unreadable file (gh-ocannl-676).
            f.write(json.dumps(json_safe(r), allow_nan=False) + "\n")
    lines = []
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True
    ).stdout.strip()
    lines.append(f"# Benchmark results\n")
    lines.append(
        f"platform: {platform.platform()} {platform.machine()} | "
        f"ocannl commit: {commit} | parity tol: {PARITY_TOL:g} (max rel diff over "
        f"first parity steps vs pytorch/cpu/eager; reduced precisions get their own envelope: "
        + ", ".join(f"{p} {t:g}" for p, t in sorted(PARITY_TOL_PRECISION.items()))
        + ")\n"
    )
    for workload in sorted({r["workload"] for r in results}):
        lines.append(f"\n## {workload}\n")
        rows = [r for r in results if r["workload"] == workload]
        # Which bytes these numbers are on (gh-ocannl-645). A report is compared across sessions
        # and machines; without this the comparison rests on the assumption that everyone
        # regenerated the same fixture. More than one line here means the section mixes fixtures.
        measured_on = sorted(
            {(r.get("fixture"), r.get("fixture_sha256")) for r in rows if r.get("fixture_sha256")}
        )
        for fixture, sha in measured_on:
            lines.append(f"measured on `{fixture}`, sha256 `{sha}`\n")
        # Precision-major, p50-ascending within a precision: scheduling variants are ranked
        # against the others computing in the same format, and a reduced-precision block reads as
        # its own group rather than being interleaved by a speed it owes to its storage format.
        rows.sort(
            key=lambda r: (
                precision_rank(r.get("precision", "f32")),
                r["step_ms"]["p50"] if finite(r["step_ms"]["p50"]) else math.inf,
            )
        )
        precisions = {r.get("precision", "f32") for r in rows}
        if len(precisions) > 1:
            lines.append(
                "Rows are grouped by precision (f32 first), p50-ascending within each group.\n"
            )
        with_tokens = any(r.get("tokens_per_step") for r in rows)
        # gh-ocannl-644: which process produced the step times. Only a cell that searches or
        # compiles has the distinction, so the column appears only where one was measured.
        with_provenance = any(r.get("provenance") for r in rows)
        if with_provenance:
            lines.append(
                "`pass` says which process produced a searching cell's step times. For the OCANNL "
                "tuned cell, whose protocol splits them: `replay` is the fresh pass-2 process "
                "replaying the cached winner, **`SEARCH PASS`** is the searching process itself — "
                "whose accumulated modules and buffers inflate every launch, so those numbers are "
                "not comparable with the others, and `no search` is a tuned cell that searched "
                "nothing and replayed nothing (autotune_search=false), so it shipped the untuned "
                "default. For a tinygrad `beam` or a `torch.compile` cell, "
                "which search in the timing process by protocol: `same-process` searched here, "
                "`cached` replayed its framework's own cache (so its `compile s` is a replay cost). "
                "A tuned cell's `compile s` carries the same statement about the search pass it "
                "came from: `(cached)` for one that replayed the schedule cache, `(no search)` "
                "for one that searched nothing at all.\n"
            )
        header = "| framework | backend | variant | precision | step p50 ms | p10 | p90 | queued ms | compile s |"
        rule = "|---|---|---|---|---|---|---|---|---|"
        if with_provenance:
            header += " pass |"
            rule += "---|"
        header += " parity |"
        rule += "---|"
        if with_tokens:
            header += " tok/s |"
            rule += "---|"
        lines.append(header)
        lines.append(rule)
        for r in rows:
            s = r["step_ms"]
            parity = r["parity"]
            if parity not in ("REF", "NO-REF") and finite(r.get("parity_max_rel")):
                parity += f" ({r['parity_max_rel']:.1e})"
            if r.get("diverged_at") is not None:
                parity += f" (loss non-finite from step {r['diverged_at']})"
            elif not r["parity_loss_moved"]:
                parity += " (loss stationary)"
            tokens = ""
            if with_tokens:
                tps = r.get("tokens_per_step")
                tokens = (
                    f" {tps * 1000 / s['p50']:,.0f} |" if tps and finite(s["p50"]) else " |"
                )
            compile_s = num(r["compile_s"], ".2f")
            compile_s += COMPILE_S_NOTE.get(r.get("search_pass"), "")
            provenance = ""
            if with_provenance:
                provenance = " %s |" % PROVENANCE_MARK.get(r.get("provenance"), "—")
            lines.append(
                f"| {r['framework']} | {r['backend']} | {r['variant']} "
                f"| {r.get('precision', 'f32')} "
                f"| {num(s['p50'], '.3f')} | {num(s['p10'], '.3f')} | {num(s['p90'], '.3f')} "
                f"| {num(r['queued_step_ms'], '.3f')} | {compile_s} |{provenance} {parity} |{tokens}"
            )
    if unavailable:
        # Stated where the matrix is read: a cell missing because the workload cannot express it
        # is not a cell someone forgot to run (gh-ocannl-551).
        lines.append("\n## Cells not applicable\n")
        lines.append(
            "Requested cells this matrix cannot contain — the workload cannot express them, so "
            "their absence above is structural, not an unrun measurement.\n"
        )
        lines.append("| workload | precision | why |")
        lines.append("|---|---|---|")
        for workload, precision, reason in unavailable:
            lines.append(f"| {workload} | {precision} | {reason} |")
    text = "\n".join(lines) + "\n"
    (out_dir / "report.md").write_text(text)
    print("\n" + text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workloads", nargs="*", help="workload names (default: all fixtures)")
    ap.add_argument("--tuned", action="store_true", help="add the OCANNL autotuned variant")
    ap.add_argument(
        "--precision",
        nargs="*",
        default=[],
        type=precision_spec,
        metavar="bf16|f16|f16-static|f16-gatedN",
        help="add OCANNL reduced storage precisions (training workloads: master weights + "
        "storage policy, f16 with dynamic loss scaling; the forward-only gpt workload: "
        "load-time weight conversion; gh-ocannl-492). Precision is an axis independent of "
        "the scheduling variant, so each one requested here is run against every selected "
        "variant — `--tuned --precision bf16` measures tuned bf16, which is the only "
        "tensor-core route on RDNA3/3.5 (gh-ocannl-539). f16-static (fixed scale, no gate) "
        "and f16-gatedN (fused on-device gate sampled every N steps) are the task-5 "
        "gate-cost legs; they need a training workload and are reported as not applicable "
        "on a forward-only one (gh-ocannl-551)",
    )
    ap.add_argument(
        "--materialized",
        action="store_true",
        help="add the OCANNL materialized-activations variant",
    )
    ap.add_argument("--nojit", action="store_true", help="add the tinygrad nojit variant")
    ap.add_argument(
        "--torch-compile",
        action="store_true",
        help="add the pytorch torch.compile variant",
    )
    ap.add_argument(
        "--beam",
        type=int,
        default=0,
        metavar="N",
        help="add the tinygrad BEAM=N search variant (0 = off)",
    )
    ap.add_argument(
        "--gpu",
        choices=sorted(GPU_DEVICES),
        default="metal" if platform.system() == "Darwin" else "cuda",
        help="GPU backend for the non-CPU column of the matrix (default: metal on macOS, "
        "cuda elsewhere; none = CPU-only matrix)",
    )
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument(
        "--no-fixture-digest-check",
        action="store_true",
        help="measure fixtures whose bytes do not match fixtures/DIGESTS.txt. The digest is "
        "what says which workload a published number is on (gh-ocannl-645); use this only "
        "for a deliberately regenerated fixture you are about to re-record",
    )
    ap.add_argument(
        "--no-skip-cells",
        action="store_true",
        help="run the SKIP_CELLS entries too. Each was observed pathological on one "
        "machine/backend/OS; use this to retest whether the entry still applies here",
    )
    ap.add_argument(
        "--only",
        nargs="*",
        default=["ocannl", "pytorch", "tinygrad"],
        help="frameworks to run",
    )
    args = ap.parse_args()
    gpu_ocannl, gpu_torch, gpu_tiny = GPU_DEVICES[args.gpu]
    if args.gpu == "hip" and gpu_torch and "pytorch" in args.only:
        # "cuda" only reaches the AMD GPU when torch is a ROCm/HIP build (a stock CPU or
        # CUDA wheel isn't); probe the bench venv and fall back to the CPU-only column.
        probe = subprocess.run(
            [str(VENV_PY), "-c", "import sys, torch; sys.exit(0 if torch.version.hip else 1)"],
            capture_output=True,
        )
        if probe.returncode != 0:
            print("torch in the bench venv is not a ROCm/HIP build; "
                  "skipping the PyTorch GPU column", flush=True)
            gpu_torch = None
    if args.gpu == "hip" and gpu_tiny == "AMD" and not os.path.exists("/dev/kfd"):
        # tinygrad's AMD device drives the KFD driver directly; under WSL there is no
        # /dev/kfd (the GPU is reached through /dev/dxg and the WSL HSA runtime), but
        # tinygrad's HIP device goes through the HIP runtime and does reach the GPU.
        print("no /dev/kfd (WSL?); using tinygrad's HIP device for the GPU column", flush=True)
        gpu_tiny = "HIP"

    fixtures = sorted((HERE / "fixtures").glob("*.safetensors"))
    if args.workloads:
        fixtures = [f for f in fixtures if f.stem in args.workloads]
    if not fixtures:
        sys.exit("no fixtures found — run gen_fixtures.py first")

    fixture_shas = check_fixture_digests(fixtures, allow_unpinned=args.no_fixture_digest_check)

    metas = {fx: read_st_metadata(fx) for fx in fixtures}
    models = {fx: metas[fx].get("model", "mlp") for fx in fixtures}
    if "ocannl" in args.only and not args.skip_build:
        targets = sorted(
            {f"benchmarks/runners/ocannl/bench_{m}.exe" for m in models.values()}
        )
        subprocess.run(["dune", "build", "--root", ".", *targets], cwd=ROOT, check=True)

    results = []
    failures = []
    unavailable = []
    partial = HERE / "results" / "partial.jsonl"
    partial.parent.mkdir(parents=True, exist_ok=True)
    partial.write_text("")  # fresh run

    # The fixture the cells currently being dispatched are measuring — stamped onto every result
    # so a row, and the report built from it, states its own workload identity (gh-ocannl-645)
    # rather than leaving it to how the operator ran the sweep.
    stamp = {}

    def collect(label, cmd, override=None, **kwargs):
        t0 = time.monotonic()
        r = run_cell(label, cmd, **kwargs)
        if r:
            r.update(stamp)
            if override:
                r.update(override)
            results.append(r)
            # Stream each cell as it lands so an interrupted run keeps its results. Through
            # json_safe like the final file: an interrupted run's rows are read by the same
            # readers, and a diverged cell is exactly the one an operator interrupts around.
            with open(partial, "a") as f:
                f.write(json.dumps(json_safe(r), allow_nan=False) + "\n")
        else:
            failures.append(label)
        print(f"    cell took {time.monotonic() - t0:.0f}s", flush=True)

    for fx in fixtures:
        name = fx.stem
        model = models[fx]
        stamp.clear()
        stamp.update(fixture=fx.name, fixture_sha256=fixture_shas[fx])
        if "ocannl" in args.only:
            variants = ["default"]
            if args.materialized:
                variants.append("materialized")
            if args.tuned:
                variants.append("tuned")
            # Both training runners implement the mixed-precision recipe (master weights, storage
            # policy, f16 loss scaling and its gate-cost legs); the forward-only gpt fixture takes
            # BENCH_PRECISION as load-time weight conversion instead (gh-ocannl-492 task 4). Every
            # scheduling variant composes with every precision (gh-ocannl-529 lifted the
            # runner-side guard), so the OCANNL cells of a backend are the product of the two axes.
            mode = metas[fx].get("mode", "train")
            precisions = ["f32"]
            for precision in args.precision:
                reason = precision_unavailable(model, mode, precision)
                if reason:
                    unavailable.append((name, precision, reason))
                    print(f"--- {name} ocannl/*/{precision}: NOT APPLICABLE ({reason})")
                else:
                    precisions.append(precision)
            for backend in ["cc"] + ([gpu_ocannl] if gpu_ocannl else []):
                for precision in precisions:
                    for variant in variants:
                        cell = cell_name(variant, precision)
                        if cell_skipped(name, backend, variant, precision) and not args.no_skip_cells:
                            print(f"--- {name} ocannl/{backend}/{cell}: SKIPPED (SKIP_CELLS; "
                                  "--no-skip-cells to run it anyway)")
                            continue
                        env = cell_env(os.environ, fx, variant, precision)
                        cmd = [str(ocannl_exe(model)), f"--ocannl_backend={backend}"]
                        label = f"{name} ocannl/{backend}/{cell}"
                        # The cell's identity is what was dispatched, not what the runner chose to
                        # call itself: stamp both axes so a report row cannot silently collapse
                        # two cells (a runner predating gh-ocannl-539 reports a reduced-precision
                        # cell's variant as its precision).
                        ident = {"variant": variant, "precision": precision}
                        if variant == "tuned":
                            # Two-pass protocol: the search leaves the process slower (extra
                            # per-launch overhead from accumulated modules/buffers — measured
                            # 2.5-3.5x on small CUDA kernels), so pass 1 runs the search and
                            # populates autotune_cache (its compile_s is the search cost), and a
                            # fresh pass-2 process replays the cached winner for the step timings.
                            pass1 = run_cell(f"{label} (search pass)", cmd, env=env, cwd=HERE)
                            if pass1 is None:
                                failures.append(f"{label} (search pass)")
                                continue
                            # What the search pass actually did, which is not derivable from the
                            # compile_s it hands over: a warm autotune_cache makes it a replay, and
                            # autotune_search=false makes it neither a search nor a replay. Both
                            # are legitimate, and both would otherwise be published as a search
                            # cost. Stamped so the report can say which.
                            pass1_verdict = search_provenance(pass1)
                            if pass1_verdict != "SEARCHED":
                                print(f"    search pass verdict {pass1_verdict}: its compile_s is "
                                      "not a from-scratch search cost", flush=True)
                            collect(label, cmd, env=env, cwd=HERE,
                                    override={**ident, "compile_s": pass1["compile_s"],
                                              "search_pass": pass1_verdict})
                        else:
                            collect(label, cmd, env=env, cwd=HERE, override=ident)
        if "pytorch" in args.only:
            for device in ["cpu"] + ([gpu_torch] if gpu_torch else []):
                for compiled in [False] + ([True] if args.torch_compile else []):
                    collect(
                        f"{name} pytorch/{device}/{'compiled' if compiled else 'eager'}",
                        [str(VENV_PY), str(HERE / "runners/pytorch/run.py"), "--fixture", str(fx), "--device", device]
                        + (["--compile"] if compiled else []),
                    )
        if "tinygrad" in args.only:
            for device in ["CPU"] + ([gpu_tiny] if gpu_tiny else []):
                for jit in [1] + ([0] if args.nojit else []):
                    collect(
                        f"{name} tinygrad/{device}/{'jit' if jit else 'nojit'}",
                        [str(VENV_PY), str(HERE / "runners/tinygrad/run.py"), "--fixture", str(fx), "--device", device, "--jit", str(jit)],
                    )
                if args.beam:
                    collect(
                        f"{name} tinygrad/{device}/beam",
                        [str(VENV_PY), str(HERE / "runners/tinygrad/run.py"), "--fixture", str(fx), "--device", device, "--beam", str(args.beam)],
                    )

    parity_check(results)
    provenance_violations = provenance_check(results)
    report(results, HERE / "results", unavailable)
    ok = True
    if unavailable:
        # Not a failure: these cells were requested but the workload cannot express them. Saying so
        # is the point — an unavailable cell and an unrun one otherwise look identical.
        print(
            f"NOT APPLICABLE: {len(unavailable)} requested cell(s) the workload cannot express: "
            + ", ".join(f"{w}/{p}" for w, p, _ in unavailable),
            flush=True,
        )
    if failures:
        ok = False
        print(
            f"RUNNER FAILURES: {len(failures)} cell(s) produced no result: "
            + ", ".join(failures),
            flush=True,
        )
    no_ref = [r for r in results if r["parity"] == "NO-REF"]
    if no_ref and "pytorch" in args.only:
        # The reference cell was requested but is missing — the gate would be vacuous.
        ok = False
        print(f"PARITY GATE: {len(no_ref)} cell(s) have no reference to compare against", flush=True)
    if provenance_violations:
        # Not a parity problem: these cells computed the right thing. Their step times are the
        # search process's, which the two-pass protocol exists to keep out of a report — and which
        # nothing downstream can detect once the numbers are quoted (gh-ocannl-644).
        ok = False
        labels = ", ".join(
            f"{r['workload']} {r['backend']}/"
            f"{cell_name(r['variant'], r.get('precision', 'f32'))}"
            for r in provenance_violations
        )
        print(
            f"PROVENANCE GATE: {len(provenance_violations)} tuned cell(s) reported step times "
            f"from a process that searched: {labels}",
            flush=True,
        )
    failed = [r for r in results if r["parity"] == "FAIL"]
    if failed:
        ok = False
        print(f"PARITY GATE: {len(failed)} cell(s) FAILED", flush=True)
    diverged = [r for r in results if r["parity"] == "DIVERGED"]
    if diverged:
        # A cell that ran and blew up, reported as such (gh-ocannl-676). It is a gate failure like
        # any other parity failure, named separately because the finding is different: nothing is
        # wrong with the runner, and the trajectory that shows it is in results.jsonl.
        ok = False
        labels = ", ".join(
            f"{r['workload']} {r['framework']}/{r['backend']}/"
            f"{cell_name(r['variant'], r.get('precision', 'f32'))}"
            f" (from step {r['diverged_at']})"
            for r in diverged
        )
        print(
            f"PARITY GATE: {len(diverged)} cell(s) DIVERGED (non-finite loss): {labels}",
            flush=True,
        )
    stationary = [
        r for r in results if not r["parity_loss_moved"] and r["diverged_at"] is None
    ]
    if stationary:
        ok = False
        labels = ", ".join(
            f"{r['workload']} {r['framework']}/{r['backend']}/"
            f"{cell_name(r['variant'], r.get('precision', 'f32'))}"
            for r in stationary
        )
        print(
            f"LOSS-MOVEMENT GATE: {len(stationary)} stationary cell(s): {labels}",
            flush=True,
        )
    if not ok:
        sys.exit(1)
    print("PARITY GATE: all cells passed", flush=True)


if __name__ == "__main__":
    main()
