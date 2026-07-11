#!/usr/bin/env python3
"""Generate self-describing safetensors fixtures for the cross-framework benchmark suite.

Each fixture holds the initial weights, the full dataset (inputs and one-hot labels), and
the workload hyperparameters in the safetensors __metadata__ map, so every runner needs
only the fixture path. All payloads are float32; weights are [fan_out, fan_in] row-major
(the shared convention: PyTorch nn.Linear layout, OCANNL output-axes-then-input-axes).
"""

import json
import sys
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file


def gen_moons(rng, n, noise=0.1):
    """Two interleaved half-moons, n samples, inputs [n, 2], labels [n] in {0, 1}."""
    n0 = n // 2
    n1 = n - n0
    t0 = rng.uniform(0.0, np.pi, n0)
    t1 = rng.uniform(0.0, np.pi, n1)
    x0 = np.stack([np.cos(t0), np.sin(t0)], axis=1)
    x1 = np.stack([1.0 - np.cos(t1), 0.5 - np.sin(t1)], axis=1)
    x = np.concatenate([x0, x1]).astype(np.float32)
    x += rng.normal(0.0, noise, x.shape).astype(np.float32)
    y = np.concatenate([np.zeros(n0, np.int64), np.ones(n1, np.int64)])
    perm = rng.permutation(n)
    return x[perm], y[perm]


def gen_gaussian(rng, n, din, num_classes):
    x = rng.normal(0.0, 1.0, (n, din)).astype(np.float32)
    y = rng.integers(0, num_classes, n)
    return x, y


def one_hot(y, num_classes):
    out = np.zeros((y.shape[0], num_classes), np.float32)
    out[np.arange(y.shape[0]), y] = 1.0
    return out


def build(spec_path: Path, out_dir: Path):
    spec = json.loads(spec_path.read_text())
    rng = np.random.default_rng(spec["seed"])
    dims = spec["dims"]
    total = spec["batch_size"] * spec["n_batches"]

    tensors = {}
    for i, (din, dout) in enumerate(zip(dims, dims[1:]), start=1):
        scale = np.sqrt(1.0 / din)
        tensors[f"w{i}"] = rng.uniform(-scale, scale, (dout, din)).astype(np.float32)
        tensors[f"b{i}"] = np.zeros(dout, np.float32)

    if spec["data"] == "moons":
        assert dims[0] == 2 and dims[-1] == 2, "moons data is 2-D input, 2 classes"
        x, y = gen_moons(rng, total)
    elif spec["data"] == "gaussian":
        x, y = gen_gaussian(rng, total, dims[0], dims[-1])
    else:
        raise ValueError(f"unknown data kind {spec['data']!r}")
    tensors["x"] = x
    tensors["y"] = one_hot(y, dims[-1])

    meta = {
        "name": spec["name"],
        "n_layers": str(len(dims) - 1),
        "batch_size": str(spec["batch_size"]),
        "lr": repr(spec["lr"]),
        "seed": str(spec["seed"]),
        "parity_steps": str(spec["parity_steps"]),
        "warmup_steps": str(spec["warmup_steps"]),
        "timed_steps": str(spec["timed_steps"]),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{spec['name']}.safetensors"
    save_file(tensors, str(out_path), metadata=meta)
    print(f"wrote {out_path} ({out_path.stat().st_size} bytes)")


if __name__ == "__main__":
    here = Path(__file__).parent
    specs = [Path(a) for a in sys.argv[1:]] or sorted((here / "workloads").glob("*.json"))
    for spec in specs:
        build(spec, here / "fixtures")
