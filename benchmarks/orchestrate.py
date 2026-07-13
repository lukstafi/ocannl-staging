#!/usr/bin/env python3
"""Run the cross-framework benchmark matrix (OCANNL / PyTorch / tinygrad) and report.

For each fixture in fixtures/, runs every (framework, backend, variant) cell, collects the
JSON result lines, enforces the loss-trajectory parity gate against the PyTorch CPU
reference, and writes results/results.jsonl plus a markdown report.

Timing results are only comparable when the parity gate passes: a FAIL means that cell was
not computing the same training trajectory, so its step times are flagged, not compared.
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
VENV_PY = HERE / ".venv/bin/python"
PARITY_TOL = 2e-3
REFERENCE = ("pytorch", "cpu", "eager")

# The GPU column of the matrix, per --gpu choice: OCANNL backend, PyTorch device,
# tinygrad device. The CPU column (cc / cpu / CPU) is always run.
GPU_DEVICES = {
    "metal": ("metal", "mps", "METAL"),
    "cuda": ("cuda", "cuda", "CUDA"),
    "none": (None, None, None),
}

# Known-pathological cells excluded from the default matrix, as (workload, backend, variant).
# Currently empty: the metal-default-schedule pathologies (gpt2_mini 81 s/step -> ~0.3 s,
# lenet 3.2 s/step + parity FAIL -> 0.22 s exact, mlp_wide >10 s/step -> 6 ms) were fixed by
# lowering the default GPU schedule's serial-fallback threshold, promoting statement-crossing
# Local scratch at fission, and working around a Metal compiler miscompilation of scalar
# read-modify-write accumulation (see arrayjit/lib/c_syntax.ml volatile_scalar_rmw and
# benchmarks/runners/ocannl/bench_metal_bug.ml).
SKIP_CELLS = set()

sys.path.insert(0, str(HERE / "runners"))
from bench_common import read_st_metadata  # noqa: E402


def ocannl_exe(model):
    return ROOT / f"_build/default/benchmarks/runners/ocannl/bench_{model}.exe"


def run_cell(label, cmd, env=None, cwd=None):
    print(f"--- {label}", flush=True)
    proc = subprocess.run(
        cmd, env=env, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    line = next(
        (l for l in reversed(proc.stdout.splitlines()) if l.startswith("{")), None
    )
    if proc.returncode != 0 or line is None:
        print(proc.stdout[-4000:])
        print(f"!!! {label} failed (exit {proc.returncode})", flush=True)
        return None
    result = json.loads(line)
    p50 = result["step_ms"]["p50"]
    print(f"    p50 {p50:.3f} ms, compile {result['compile_s']:.2f} s", flush=True)
    return result


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
        for r in rs:
            if ref is None:
                r["parity"] = "NO-REF"
                continue
            if r is ref:
                r["parity"] = "REF"
                r["parity_max_rel"] = 0.0
                continue
            n = min(len(r["losses"]), len(ref["losses"]))
            max_rel = max(
                abs(a - b) / max(abs(b), 1e-6)
                for a, b in zip(r["losses"][:n], ref["losses"][:n])
            )
            r["parity_max_rel"] = max_rel
            r["parity"] = "PASS" if max_rel < PARITY_TOL else "FAIL"


def report(results, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    lines = []
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True
    ).stdout.strip()
    lines.append(f"# Benchmark results\n")
    lines.append(
        f"platform: {platform.platform()} {platform.machine()} | "
        f"ocannl commit: {commit} | parity tol: {PARITY_TOL:g} (max rel diff over "
        f"first parity steps vs pytorch/cpu/eager)\n"
    )
    for workload in sorted({r["workload"] for r in results}):
        lines.append(f"\n## {workload}\n")
        rows = [r for r in results if r["workload"] == workload]
        rows.sort(key=lambda r: r["step_ms"]["p50"])
        with_tokens = any(r.get("tokens_per_step") for r in rows)
        header = "| framework | backend | variant | step p50 ms | p10 | p90 | queued ms | compile s | parity |"
        rule = "|---|---|---|---|---|---|---|---|---|"
        if with_tokens:
            header += " tok/s |"
            rule += "---|"
        lines.append(header)
        lines.append(rule)
        for r in rows:
            s = r["step_ms"]
            parity = r["parity"]
            if parity not in ("REF", "NO-REF"):
                parity += f" ({r['parity_max_rel']:.1e})"
            tokens = ""
            if with_tokens:
                tps = r.get("tokens_per_step")
                tokens = f" {tps * 1000 / s['p50']:,.0f} |" if tps else " |"
            lines.append(
                f"| {r['framework']} | {r['backend']} | {r['variant']} "
                f"| {s['p50']:.3f} | {s['p10']:.3f} | {s['p90']:.3f} "
                f"| {r['queued_step_ms']:.3f} | {r['compile_s']:.2f} | {parity} |{tokens}"
            )
    text = "\n".join(lines) + "\n"
    (out_dir / "report.md").write_text(text)
    print("\n" + text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workloads", nargs="*", help="workload names (default: all fixtures)")
    ap.add_argument("--tuned", action="store_true", help="add the OCANNL autotuned variant")
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
        "--gpu",
        choices=sorted(GPU_DEVICES),
        default="metal" if platform.system() == "Darwin" else "cuda",
        help="GPU backend for the non-CPU column of the matrix (default: metal on macOS, "
        "cuda elsewhere; none = CPU-only matrix)",
    )
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument(
        "--only",
        nargs="*",
        default=["ocannl", "pytorch", "tinygrad"],
        help="frameworks to run",
    )
    args = ap.parse_args()
    gpu_ocannl, gpu_torch, gpu_tiny = GPU_DEVICES[args.gpu]

    fixtures = sorted((HERE / "fixtures").glob("*.safetensors"))
    if args.workloads:
        fixtures = [f for f in fixtures if f.stem in args.workloads]
    if not fixtures:
        sys.exit("no fixtures found — run gen_fixtures.py first")

    models = {fx: read_st_metadata(fx).get("model", "mlp") for fx in fixtures}
    if "ocannl" in args.only and not args.skip_build:
        targets = sorted(
            {f"benchmarks/runners/ocannl/bench_{m}.exe" for m in models.values()}
        )
        subprocess.run(["dune", "build", "--root", ".", *targets], cwd=ROOT, check=True)

    results = []
    failures = []
    partial = HERE / "results" / "partial.jsonl"
    partial.parent.mkdir(parents=True, exist_ok=True)
    partial.write_text("")  # fresh run

    def collect(label, cmd, override=None, **kwargs):
        t0 = time.monotonic()
        r = run_cell(label, cmd, **kwargs)
        if r:
            if override:
                r.update(override)
            results.append(r)
            # Stream each cell as it lands so an interrupted run keeps its results.
            with open(partial, "a") as f:
                f.write(json.dumps(r) + "\n")
        else:
            failures.append(label)
        print(f"    cell took {time.monotonic() - t0:.0f}s", flush=True)

    for fx in fixtures:
        name = fx.stem
        model = models[fx]
        if "ocannl" in args.only:
            variants = ["default"]
            if args.materialized:
                variants.append("materialized")
            if args.tuned:
                variants.append("tuned")
            for backend in ["cc"] + ([gpu_ocannl] if gpu_ocannl else []):
                for variant in variants:
                    if (name, backend, variant) in SKIP_CELLS:
                        print(f"--- {name} ocannl/{backend}/{variant}: SKIPPED (SKIP_CELLS)")
                        continue
                    env = dict(
                        os.environ,
                        BENCH_FIXTURE=str(fx),
                        BENCH_TUNE="1" if variant == "tuned" else "0",
                        BENCH_MATERIALIZE="1" if variant == "materialized" else "0",
                    )
                    cmd = [str(ocannl_exe(model)), f"--ocannl_backend={backend}"]
                    label = f"{name} ocannl/{backend}/{variant}"
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
                        collect(label, cmd, env=env, cwd=HERE,
                                override={"compile_s": pass1["compile_s"]})
                    else:
                        collect(label, cmd, env=env, cwd=HERE)
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

    parity_check(results)
    report(results, HERE / "results")
    ok = True
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
    failed = [r for r in results if r["parity"] == "FAIL"]
    if failed:
        ok = False
        print(f"PARITY GATE: {len(failed)} cell(s) FAILED", flush=True)
    if not ok:
        sys.exit(1)
    print("PARITY GATE: all cells passed", flush=True)


if __name__ == "__main__":
    main()
