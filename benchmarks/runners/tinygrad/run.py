#!/usr/bin/env python3
"""tinygrad runner for the cross-framework benchmark suite (see benchmarks/README.md).

Same protocol as the other runners: load the self-describing fixture, train an n-layer relu
MLP with softmax cross-entropy and plain SGD (tinygrad's nn.optim.SGD), emit one JSON line
with parity losses and steady-state step times.
"""

import argparse
import os
import sys
import time
from importlib.metadata import version as pkg_version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bench_common import emit, percentiles, read_st_metadata

ap = argparse.ArgumentParser()
ap.add_argument("--fixture", required=True)
ap.add_argument("--device", default="CPU", choices=["CPU", "METAL"])
ap.add_argument("--jit", type=int, default=1)
args = ap.parse_args()
os.environ["DEV"] = args.device

import numpy as np
from safetensors.numpy import load_file
from tinygrad import Device, Tensor, TinyJit
from tinygrad.nn.optim import SGD


def main():
    meta = read_st_metadata(args.fixture)
    n_layers = int(meta["n_layers"])
    batch_size = int(meta["batch_size"])
    lr = float(meta["lr"])
    parity_steps = int(meta["parity_steps"])
    warmup_steps = int(meta["warmup_steps"])
    timed_steps = int(meta["timed_steps"])

    data = load_file(args.fixture)
    params = []
    for i in range(1, n_layers + 1):
        w = Tensor(data[f"w{i}"])
        b = Tensor(data[f"b{i}"])
        w.requires_grad = True
        b.requires_grad = True
        params.append((w, b))
    flat = [p for wb in params for p in wb]
    opt = SGD(flat, lr=lr)
    n_batches = data["x"].shape[0] // batch_size
    batches = [
        (
            Tensor(data["x"][i * batch_size : (i + 1) * batch_size]).realize(),
            Tensor(data["y"][i * batch_size : (i + 1) * batch_size]).realize(),
        )
        for i in range(n_batches)
    ]

    def step_inner(xb, yb):
        h = xb
        for i, (w, b) in enumerate(params):
            h = h.linear(w.T, b)
            if i < n_layers - 1:
                h = h.relu()
        probs = h.softmax(-1)
        correct = (probs * yb).sum(-1)
        loss = -(correct.log()).mean()
        opt.zero_grad()
        loss.backward()
        # Realize the loss value before opt.step(): the step assigns params in place, and a
        # later realize would recompute the (fused-away) loss from the updated weights.
        loss.realize()
        opt.step()
        return loss

    if args.jit:
        step_inner = TinyJit(step_inner)

    def step(k):
        xb, yb = batches[k % n_batches]
        return step_inner(xb, yb)

    def sync():
        Device[Device.DEFAULT].synchronize()

    with Tensor.train():
        # First step doubles as the compile probe (kernel compilation; with jit, part of
        # capture). Its loss value is unaffected by the timing, so it is parity step 0.
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
            "framework": "tinygrad",
            "backend": args.device,
            "variant": "jit" if args.jit else "nojit",
            "workload": meta["name"],
            "compile_s": round(compile_s, 3),
            "step_ms": percentiles(synced),
            "queued_step_ms": queued,
            "timed_steps": timed_steps,
            "losses": losses,
            "version": pkg_version("tinygrad"),
        }
    )


if __name__ == "__main__":
    main()
