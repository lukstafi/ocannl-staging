#!/usr/bin/env python3
"""gh-ocannl-675, CUDA leg: pass-1 (searched here) vs pass-2 (fresh process, replay) step times.

One arm = one pair of processes: pass 1 searches (cold cache) and times, pass 2 replays that
cache in a fresh process and times again. Arms are run one at a time, alternating across repeats
so a drifting box moves both halves of a pair together, behind an idle gate and pinned with
taskset. Every record goes to $GH675_OUT/records.jsonl verbatim; the analysis pairs pass 1 with
pass 2 WITHIN a repeat and reports the median of the paired ratios.

    python3 gh675_cells.py --arms A_beam2 C_jit_cold C_eager D_ocannl --workloads mlp_small \
        --repeats 3 --start-repeat 1

Arms: A_beam* tinygrad BEAM=N, B_compiled/B_maxauto torch.compile, C_* the non-searching
controls (their X must be ~0 or the spread is process noise, not search residue), D_ocannl the
tuned OCANNL cell the two-pass protocol exists for, REF_cpu the parity reference.

Four things a ratio needs in order to mean anything, which this driver establishes rather than
assumes -- each of them a way a pair can look valid and not be:
  * the TREATMENT is the arm's, not the shell's: every framework knob is stripped from what a
    child inherits and set explicitly per arm (an exported `BEAM` otherwise makes the controls
    search while they still report `searched: false`);
  * the PROVENANCE is checked per pair against what the arm claims, and a pair that searched in
    pass 2 (or failed to in pass 1) is stamped `provenance_ok: false` and announced;
  * the FIXTURE is identified by sha256 on every record, and `--expect-digest` refuses a run whose
    bytes are not the ones being resumed;
  * a WEDGE is recorded against the pass that wedged, so a searching-process hang and a replaying
    -process hang stay distinguishable.
Nothing this driver writes touches the checkout: caches, records and the tuned cell's
`autotune_cache_dir` all live under $GH675_OUT.

See benchmarks/report-gh675-cuda.md for what it measured.
"""
import argparse, hashlib, json, os, shutil, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HERE = ROOT / "benchmarks"
VENV = HERE / ".venv/bin/python"
# Everything this driver writes -- the per-run records and the per-arm caches whose warmth IS the
# experiment -- goes outside the checkout, under $GH675_OUT (default: a tmp dir).
SP = Path(os.environ.get("GH675_OUT", "/tmp/gh675"))
RECORDS = SP / "records.jsonl"
CACHES = SP / "caches"
NVSMI = "/usr/lib/wsl/lib/nvidia-smi"
TASKSET = ["taskset", "-c", "0-15"]

FIXTURE = {w: str(HERE / f"fixtures/{w}.safetensors") for w in ("mlp_small", "gpt2_mini")}
DIGESTS = {}  # filled at startup; stamped on every record

# Every variable that is a TREATMENT in this experiment rather than part of the box. A child
# inherits the driver's environment, so one of these exported in the operator's shell silently
# rewrites an arm: a stray `BEAM=2` makes the non-beam CONTROLS beam-search, and the runner --
# which reports `searched` from its own command line, not from the env -- then labels them
# `searched: false`, which is the one reading that cannot be caught downstream. Each arm states
# its treatment explicitly below and everything here is stripped from what it inherits, so an arm
# is what the driver says it is rather than what the shell happened to hold (gh-ocannl-675).
TREATMENT_VARS = (
    "BEAM", "CACHEDB", "CACHELEVEL", "IGNORE_BEAM_CACHE", "JIT", "DEV", "PARALLEL",
    "TORCHINDUCTOR_CACHE_DIR", "TORCHINDUCTOR_FX_GRAPH_CACHE", "TORCHINDUCTOR_MAX_AUTOTUNE",
    "TRITON_CACHE_DIR", "CUDA_CACHE_PATH",
)
TREATMENT_PREFIXES = ("BENCH_", "OCANNL_")


def base_env():
    """The inherited environment with every treatment variable removed."""
    return {
        k: v
        for k, v in os.environ.items()
        if k not in TREATMENT_VARS and not k.startswith(TREATMENT_PREFIXES)
    }


def digests():
    """sha256 of each fixture, stamped onto every record.

    The fixtures are gitignored and regenerable, and `gen_fixtures.py` does not promise stable
    bytes across numpy releases, so a filename does not identify a workload. Without this a run
    resumed after a regeneration mixes two workloads under one `(workload, repeat)` label
    (gh-ocannl-645 is the same rule for published reports).
    """
    out = {}
    for w, path in FIXTURE.items():
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        out[w] = h.hexdigest()
    return out


def gate():
    """Wait until the box is quiet enough to time on."""
    for _ in range(24):
        load = float(open("/proc/loadavg").read().split()[0])
        try:
            util = subprocess.run([NVSMI, "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                                  capture_output=True, text=True, timeout=30).stdout.strip().splitlines()[0]
            util = int(util)
        except Exception:
            util = 0
        # WSL's load average decays slowly and counts uninterruptible tasks, so it is a poor
        # instantaneous gate; the GPU is the resource under measurement. Wait for both, but do
        # not stall the sweep on a stale load figure.
        if load < 8.0 and util < 10:
            return
        time.sleep(5)
    print("gate: giving up waiting for idle", file=sys.stderr)


def run(cmd, env=None, cwd=None, timeout=None):
    e = base_env()
    if env:
        e.update(env)
    gate()
    t0 = time.monotonic()
    p = subprocess.run(TASKSET + cmd, env=e, cwd=cwd or str(HERE), capture_output=True, text=True,
                       timeout=timeout)
    wall = time.monotonic() - t0
    line = None
    for ln in reversed(p.stdout.strip().splitlines()):
        ln = ln.strip()
        if ln.startswith("{") and ln.endswith("}"):
            try:
                line = json.loads(ln)
                break
            except Exception:
                pass
    if line is None:
        return {"__failed__": True, "rc": p.returncode, "wall_s": wall,
                "stdout_tail": p.stdout[-3000:], "stderr_tail": p.stderr[-3000:]}
    line["wall_s"] = round(wall, 2)
    line["rc"] = p.returncode
    return line


def tiny(workload, beam=0, cachedb=None, retime=True):
    cmd = [str(VENV), str(HERE / "runners/tinygrad/run.py"), "--fixture", FIXTURE[workload],
           "--device", "CUDA", "--jit", "1"]
    if beam:
        cmd += ["--beam", str(beam)]
    if retime:
        cmd += ["--retime"]
    # BEAM is stated for every tinygrad arm, `0` included: the controls' whole claim is that they
    # do not search, and inheriting an exported BEAM would make that claim false while the runner
    # kept reporting `searched: false` from its own argv.
    return cmd, {"CACHEDB": str(cachedb), "BEAM": str(beam)}


def torchcmd(workload, compiled, mode=None, cachedir=None, retime=True):
    cmd = [str(VENV), str(HERE / "runners/pytorch/run.py"), "--fixture", FIXTURE[workload],
           "--device", "cuda"]
    if compiled:
        cmd += ["--compile"]
        if mode:
            cmd += ["--compile-mode", mode]
    if retime:
        cmd += ["--retime"]
    env = {}
    if cachedir:
        env["TORCHINDUCTOR_CACHE_DIR"] = str(cachedir)
        env["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"
        env["TRITON_CACHE_DIR"] = str(Path(cachedir) / "triton")
    return cmd, env


def ocannl_cache(workload):
    """The tuned cell's autotune cache -- under GH675_OUT like every other arm's.

    The default is `./autotune_cache` relative to the runner's cwd, i.e. inside the checkout: a
    driver that wiped THAT before each pass 1 would destroy whatever tuning results the operator
    had from other benchmark work, and would couple the experiment to checkout state.
    """
    return CACHES / f"ocannl_{workload}"


def ocannl(workload):
    exe = {"mlp_small": "bench_mlp", "gpt2_mini": "bench_gpt"}[workload]
    cmd = [str(ROOT / f"_build/default/benchmarks/runners/ocannl/{exe}.exe"),
           "--ocannl_backend=cuda",
           f"--ocannl_autotune_cache_dir={ocannl_cache(workload)}"]
    env = {"BENCH_FIXTURE": FIXTURE[workload], "BENCH_TUNE": "1", "BENCH_MATERIALIZE": "0",
           "BENCH_PRECISION": "f32", "BENCH_STATIC_SCALE": "0", "BENCH_GATE_INTERVAL": "0"}
    return cmd, env


def wipe(p):
    p = Path(p)
    if p.exists():
        shutil.rmtree(p, ignore_errors=True)
    p.mkdir(parents=True, exist_ok=True)


# What each arm's two passes MUST report for `searched`. A searching arm whose pass 1 comes back
# `false`/`null` did not search (a cache wipe that failed, a probe that could not bind), and one
# whose pass 2 comes back `true` did not replay (a partially written cache is the observed case) --
# in both the pair's ratio is not the quantity this experiment is about, and nothing downstream can
# tell. So the expectation is stated per arm and checked per pair, and a record that violates it is
# stamped `provenance_ok: false` with the reason (gh-ocannl-644 is the field this reads).
EXPECTED_SEARCHED = {
    "A_beam2": (True, False),
    "A_beam8": (True, False),
    "B_compiled": (True, False),
    "B_maxauto": (True, False),
    "D_ocannl": (True, False),
    "C_jit_warm": (False, False),
    "C_jit_cold": (False, False),
    "C_eager": (False, False),
}


def check_provenance(arm, p1, p2):
    """[] if the pair's `searched` fields are what this arm's design says, else the complaints."""
    want = EXPECTED_SEARCHED.get(arm)
    if want is None:
        return []
    bad = []
    for want_v, got, which in ((want[0], p1.get("searched"), "pass1"),
                               (want[1], p2.get("searched"), "pass2")):
        if got is not want_v:
            bad.append(f"{which} searched={got!r}, expected {want_v!r}")
    return bad


def emit(rec):
    with open(RECORDS, "a") as f:
        f.write(json.dumps(rec) + "\n")
    r1, r2 = rec.get("pass1", {}), rec.get("pass2", {})
    def p50(r):
        return (r.get("step_ms") or {}).get("p50")
    print(f"  {rec['arm']:28s} {rec['workload']:10s} rep{rec['repeat']}  "
          f"p1={p50(r1)} p2={p50(r2)}  searched={r1.get('searched')}/{r2.get('searched')} "
          f"compile_s={r1.get('compile_s')}/{r2.get('compile_s')}", flush=True)


def timed_run(cmd, env, timeout):
    """`run`, with a timeout recorded as a failure of THIS invocation rather than of the pair.

    Letting `TimeoutExpired` escape to the caller loses which half wedged -- and a pass-1 wedge
    (the searching process) and a pass-2 wedge (the replaying one) are different findings, which
    is exactly the distinction gh-ocannl-760 rests on.
    """
    try:
        return run(cmd, env, timeout=timeout)
    except subprocess.TimeoutExpired as ex:
        return {"__failed__": True, "timeout_s": timeout, "timeout": str(ex)}


def do_pair(arm, workload, repeat, mk1, mk2, pre1=None, pre2=None, timeout=None):
    if pre1:
        pre1()
    c, e = mk1()
    p1 = timed_run(c, e, timeout)
    if pre2:
        pre2()
    c, e = mk2()
    p2 = timed_run(c, e, timeout)
    rec = {"arm": arm, "workload": workload, "repeat": repeat, "pass1": p1, "pass2": p2,
           "fixture_sha256": DIGESTS.get(workload), "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
    if not (p1.get("__failed__") or p2.get("__failed__")):
        bad = check_provenance(arm, p1, p2)
        rec["provenance_ok"] = not bad
        if bad:
            rec["provenance"] = bad
            print(f"  !! {arm} {workload} rep{repeat}: PROVENANCE {'; '.join(bad)} "
                  f"-- pair recorded as invalid, exclude it", file=sys.stderr, flush=True)
    emit(rec)


ARMS = {}


def arm(name):
    def deco(fn):
        ARMS[name] = fn
        return fn
    return deco


@arm("A_beam2")
def a_beam2(w, rep):
    db = CACHES / f"tiny_beam2_{w}"
    do_pair("A_beam2", w, rep,
            lambda: tiny(w, beam=2, cachedb=db / "cache.db"),
            lambda: tiny(w, beam=2, cachedb=db / "cache.db"),
            pre1=lambda: wipe(db), timeout=2400)


@arm("A_beam8")
def a_beam8(w, rep):
    db = CACHES / f"tiny_beam8_{w}"
    do_pair("A_beam8", w, rep,
            lambda: tiny(w, beam=8, cachedb=db / "cache.db"),
            lambda: tiny(w, beam=8, cachedb=db / "cache.db"),
            pre1=lambda: wipe(db), timeout=3600)


@arm("C_jit_cold")
def c_jit_cold(w, rep):
    db = CACHES / f"tiny_jitcold_{w}"
    do_pair("C_jit_cold", w, rep,
            lambda: tiny(w, cachedb=db / "cache.db"),
            lambda: tiny(w, cachedb=db / "cache.db"),
            pre1=lambda: wipe(db), timeout=1200)


@arm("C_jit_warm")
def c_jit_warm(w, rep):
    db = CACHES / f"tiny_jitwarm_{w}"
    db.mkdir(parents=True, exist_ok=True)
    if not (db / "cache.db").exists():
        # This arm's claim is warm-vs-warm: it is the control that says a pair of processes doing
        # no compilation and no search differ by ~nothing. Against an empty cache its first pair
        # would be cold-vs-warm instead -- a compile control, and one whose X would be read as
        # process spread. Prewarm once, unrecorded, rather than depending on the operator having
        # run (and discarded) a repeat 0.
        print(f"  prewarming {db} (unrecorded)", flush=True)
        c, e = tiny(w, cachedb=db / "cache.db", retime=False)
        timed_run(c, e, 1200)
    do_pair("C_jit_warm", w, rep,
            lambda: tiny(w, cachedb=db / "cache.db"),
            lambda: tiny(w, cachedb=db / "cache.db"), timeout=1200)


@arm("B_compiled")
def b_compiled(w, rep):
    cd = CACHES / f"inductor_compiled_{w}_{rep}"
    do_pair("B_compiled", w, rep,
            lambda: torchcmd(w, True, cachedir=cd),
            lambda: torchcmd(w, True, cachedir=cd),
            pre1=lambda: wipe(cd), timeout=2400)


@arm("B_maxauto")
def b_maxauto(w, rep):
    cd = CACHES / f"inductor_maxauto_{w}_{rep}"
    do_pair("B_maxauto", w, rep,
            lambda: torchcmd(w, True, mode="max-autotune", cachedir=cd),
            lambda: torchcmd(w, True, mode="max-autotune", cachedir=cd),
            pre1=lambda: wipe(cd), timeout=3600)


@arm("C_eager")
def c_eager(w, rep):
    do_pair("C_eager", w, rep,
            lambda: torchcmd(w, False),
            lambda: torchcmd(w, False), timeout=1200)


@arm("D_ocannl")
def d_ocannl(w, rep):
    ac = ocannl_cache(w)
    do_pair("D_ocannl", w, rep,
            lambda: ocannl(w), lambda: ocannl(w),
            pre1=lambda: wipe(ac), timeout=3600)


@arm("REF_cpu")
def ref_cpu(w, rep):
    cmd = [str(VENV), str(HERE / "runners/pytorch/run.py"), "--fixture", FIXTURE[w], "--device", "cpu"]
    r = run(cmd, {}, timeout=3600)
    emit({"arm": "REF_cpu", "workload": w, "repeat": rep, "pass1": r, "pass2": {},
          "ts": time.strftime("%Y-%m-%dT%H:%M:%S")})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--workloads", nargs="+", default=["mlp_small"])
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--start-repeat", type=int, default=1)
    ap.add_argument("--expect-digest", nargs="*", metavar="WORKLOAD=SHA256",
                    help="refuse to run unless the fixture's bytes hash to this -- what makes a "
                         "resumed run the same experiment as the one it resumes")
    args = ap.parse_args()
    CACHES.mkdir(parents=True, exist_ok=True)
    DIGESTS.update(digests())
    print(f"records: {RECORDS}", flush=True)
    for w, d in sorted(DIGESTS.items()):
        print(f"fixture {w}: sha256 {d}", flush=True)
    if args.expect_digest:
        pinned = dict(x.split("=", 1) for x in args.expect_digest)
        wrong = {w: DIGESTS[w] for w, d in pinned.items() if DIGESTS.get(w) != d}
        if wrong:
            sys.exit(f"fixture digest mismatch (pinned vs actual): {pinned} vs {wrong} -- these "
                     "are different workload bytes under the same names; regenerate or re-pin")
    for rep in range(args.start_repeat, args.start_repeat + args.repeats):
        for w in args.workloads:
            for a in args.arms:
                print(f"== rep{rep} {w} {a}", flush=True)
                try:
                    ARMS[a](w, rep)
                except subprocess.TimeoutExpired as ex:
                    emit({"arm": a, "workload": w, "repeat": rep,
                          "pass1": {"__failed__": True, "timeout": str(ex)}, "pass2": {},
                          "ts": time.strftime("%Y-%m-%dT%H:%M:%S")})
