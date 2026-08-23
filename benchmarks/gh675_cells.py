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
See benchmarks/report-gh675-cuda.md for what it measured.
"""
import argparse, json, os, shutil, subprocess, sys, time
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
    e = dict(os.environ)
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
    return cmd, {"CACHEDB": str(cachedb)}


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


def ocannl(workload):
    exe = {"mlp_small": "bench_mlp", "gpt2_mini": "bench_gpt"}[workload]
    cmd = [str(ROOT / f"_build/default/benchmarks/runners/ocannl/{exe}.exe"), "--ocannl_backend=cuda"]
    env = {"BENCH_FIXTURE": FIXTURE[workload], "BENCH_TUNE": "1", "BENCH_MATERIALIZE": "0",
           "BENCH_PRECISION": "f32", "BENCH_STATIC_SCALE": "0", "BENCH_GATE_INTERVAL": "0"}
    return cmd, env


def wipe(p):
    p = Path(p)
    if p.exists():
        shutil.rmtree(p, ignore_errors=True)
    p.mkdir(parents=True, exist_ok=True)


def emit(rec):
    with open(RECORDS, "a") as f:
        f.write(json.dumps(rec) + "\n")
    r1, r2 = rec.get("pass1", {}), rec.get("pass2", {})
    def p50(r):
        return (r.get("step_ms") or {}).get("p50")
    print(f"  {rec['arm']:28s} {rec['workload']:10s} rep{rec['repeat']}  "
          f"p1={p50(r1)} p2={p50(r2)}  searched={r1.get('searched')}/{r2.get('searched')} "
          f"compile_s={r1.get('compile_s')}/{r2.get('compile_s')}", flush=True)


def do_pair(arm, workload, repeat, mk1, mk2, pre1=None, pre2=None, timeout=None):
    if pre1:
        pre1()
    c, e = mk1()
    p1 = run(c, e, timeout=timeout)
    if pre2:
        pre2()
    c, e = mk2()
    p2 = run(c, e, timeout=timeout)
    emit({"arm": arm, "workload": workload, "repeat": repeat, "pass1": p1, "pass2": p2,
          "ts": time.strftime("%Y-%m-%dT%H:%M:%S")})


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
    ac = HERE / "autotune_cache"
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
    args = ap.parse_args()
    CACHES.mkdir(parents=True, exist_ok=True)
    print(f"records: {RECORDS}", flush=True)
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
