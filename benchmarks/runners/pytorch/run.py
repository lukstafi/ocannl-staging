#!/usr/bin/env python3
"""PyTorch runner for the cross-framework benchmark suite (see benchmarks/README.md).

Same protocol as the other runners: load the self-describing fixture, train an n-layer relu
MLP with softmax cross-entropy and plain SGD, emit one JSON line with parity losses and
steady-state step times.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bench_common import emit, percentiles, read_st_metadata

import torch
from safetensors.torch import load_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", required=True)
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps"])
    ap.add_argument("--compile", action="store_true")
    args = ap.parse_args()

    meta = read_st_metadata(args.fixture)
    n_layers = int(meta["n_layers"])
    batch_size = int(meta["batch_size"])
    lr = float(meta["lr"])
    parity_steps = int(meta["parity_steps"])
    warmup_steps = int(meta["warmup_steps"])
    timed_steps = int(meta["timed_steps"])

    torch.set_float32_matmul_precision("highest")
    dev = torch.device(args.device)
    data = load_file(args.fixture)
    params = []
    for i in range(1, n_layers + 1):
        w = data[f"w{i}"].to(dev).requires_grad_()
        b = data[f"b{i}"].to(dev).requires_grad_()
        params.append((w, b))
    flat = [p for wb in params for p in wb]
    x = data["x"].to(dev)
    y = data["y"].to(dev)
    n_batches = x.shape[0] // batch_size
    batches = [
        (x[i * batch_size : (i + 1) * batch_size], y[i * batch_size : (i + 1) * batch_size])
        for i in range(n_batches)
    ]

    def loss_fn(xb, yb):
        h = xb
        for i, (w, b) in enumerate(params):
            h = h @ w.T + b
            if i < n_layers - 1:
                h = torch.relu(h)
        probs = torch.softmax(h, dim=-1)
        correct = (probs * yb).sum(dim=-1)
        return (-correct.log()).mean()

    if args.compile:
        loss_fn = torch.compile(loss_fn)

    def step(k):
        xb, yb = batches[k % n_batches]
        loss = loss_fn(xb, yb)
        loss.backward()
        with torch.no_grad():
            for p in flat:
                p -= lr * p.grad
        for p in flat:
            p.grad = None
        return loss

    def sync():
        if args.device == "mps":
            torch.mps.synchronize()

    # The first step doubles as the compile probe (graph build; with --compile, compilation).
    # Its loss value is unaffected by the timing, so it is also parity step 0.
    k = 0
    losses = []
    t0 = time.perf_counter()
    losses.append(step(k).item())
    compile_s = time.perf_counter() - t0
    k += 1
    for _ in range(parity_steps - 1):
        losses.append(step(k).item())
        k += 1
    for _ in range(warmup_steps):
        step(k)
        k += 1
    sync()
    synced = []
    for _ in range(timed_steps):
        t0 = time.perf_counter()
        step(k)
        k += 1
        sync()
        synced.append((time.perf_counter() - t0) * 1e3)
    t0 = time.perf_counter()
    for _ in range(timed_steps):
        step(k)
        k += 1
    sync()
    queued = (time.perf_counter() - t0) / timed_steps * 1e3

    emit(
        {
            "framework": "pytorch",
            "backend": args.device,
            "variant": "compiled" if args.compile else "eager",
            "workload": meta["name"],
            "compile_s": round(compile_s, 3),
            "step_ms": percentiles(synced),
            "queued_step_ms": queued,
            "timed_steps": timed_steps,
            "losses": losses,
            "version": torch.__version__,
        }
    )


if __name__ == "__main__":
    main()
