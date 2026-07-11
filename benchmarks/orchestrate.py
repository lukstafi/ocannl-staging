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
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
VENV_PY = HERE / ".venv/bin/python"
OCANNL_EXE = ROOT / "_build/default/benchmarks/runners/ocannl/bench_mlp.exe"
PARITY_TOL = 2e-3
REFERENCE = ("pytorch", "cpu", "eager")


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
        lines.append(
            "| framework | backend | variant | step p50 ms | p10 | p90 | queued ms | compile s | parity |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|")
        rows = [r for r in results if r["workload"] == workload]
        rows.sort(key=lambda r: r["step_ms"]["p50"])
        for r in rows:
            s = r["step_ms"]
            parity = r["parity"]
            if parity not in ("REF", "NO-REF"):
                parity += f" ({r['parity_max_rel']:.1e})"
            lines.append(
                f"| {r['framework']} | {r['backend']} | {r['variant']} "
                f"| {s['p50']:.3f} | {s['p10']:.3f} | {s['p90']:.3f} "
                f"| {r['queued_step_ms']:.3f} | {r['compile_s']:.2f} | {parity} |"
            )
    text = "\n".join(lines) + "\n"
    (out_dir / "report.md").write_text(text)
    print("\n" + text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workloads", nargs="*", help="workload names (default: all fixtures)")
    ap.add_argument("--tuned", action="store_true", help="add the OCANNL autotuned variant")
    ap.add_argument("--nojit", action="store_true", help="add the tinygrad nojit variant")
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument(
        "--only",
        nargs="*",
        default=["ocannl", "pytorch", "tinygrad"],
        help="frameworks to run",
    )
    args = ap.parse_args()

    fixtures = sorted((HERE / "fixtures").glob("*.safetensors"))
    if args.workloads:
        fixtures = [f for f in fixtures if f.stem in args.workloads]
    if not fixtures:
        sys.exit("no fixtures found — run gen_fixtures.py first")

    if "ocannl" in args.only and not args.skip_build:
        subprocess.run(
            ["dune", "build", "--root", ".", "benchmarks/runners/ocannl/bench_mlp.exe"],
            cwd=ROOT,
            check=True,
        )

    results = []
    failures = []

    def collect(label, cmd, **kwargs):
        r = run_cell(label, cmd, **kwargs)
        if r:
            results.append(r)
        else:
            failures.append(label)

    for fx in fixtures:
        name = fx.stem
        if "ocannl" in args.only:
            for backend in ["cc", "metal"]:
                for tuned in [False] + ([True] if args.tuned else []):
                    env = dict(
                        os.environ, BENCH_FIXTURE=str(fx), BENCH_TUNE="1" if tuned else "0"
                    )
                    variant = "tuned" if tuned else "default"
                    collect(
                        f"{name} ocannl/{backend}/{variant}",
                        [str(OCANNL_EXE), f"--ocannl_backend={backend}"],
                        env=env,
                        cwd=HERE,  # picks up benchmarks/ocannl_config
                    )
        if "pytorch" in args.only:
            for device in ["cpu", "mps"]:
                collect(
                    f"{name} pytorch/{device}/eager",
                    [str(VENV_PY), str(HERE / "runners/pytorch/run.py"), "--fixture", str(fx), "--device", device],
                )
        if "tinygrad" in args.only:
            for device in ["CPU", "METAL"]:
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
