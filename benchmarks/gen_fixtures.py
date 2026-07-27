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


def uniform(rng, fan_in, shape):
    scale = np.sqrt(1.0 / fan_in)
    return rng.uniform(-scale, scale, shape).astype(np.float32)


def build_mlp(spec, rng, tensors, meta):
    dims = spec["dims"]
    total = spec["batch_size"] * spec["n_batches"]
    for i, (din, dout) in enumerate(zip(dims, dims[1:]), start=1):
        tensors[f"w{i}"] = uniform(rng, din, (dout, din))
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
    meta["n_layers"] = str(len(dims) - 1)


def build_conv(spec, rng, tensors, meta):
    """A two-conv classifier. Weight layouts follow OCANNL's axis order (output axes then
    input axes, channels-last images): conv kernel [oc, kh, kw, ic]; fc1 weight
    [hid, oh, ow, oc] (input axes = the conv feature map); images [total, h, w, c]. The
    Python runners permute to NCHW.

    Three shapes drive the same builder: LeNet-5 (1-channel, valid convs, small channels),
    the cifar-scale variant (gh-ocannl-500: 3-channel 44x44, valid 5x5 convs so both conv
    GEMM rows land on multiples of 8 — 40 and 16 — and channels multiples of 8, so the
    blocked conv sketch legs' per-block tile gates fire), and the stride-2-stem variant
    (gh-ocannl-502: 3-channel 51x51, conv1 at stride 2 — the strided downsampling site the
    compacting Stage targets — with rows 24 and 8 still multiples of 8). [in_channels]
    defaults to 1, [use_padding] to false, and [stride1]/[stride2] to 1, keeping the LeNet
    fixture bit-identical. Strides are only supported with valid convs: same-padding
    output-size conventions differ across frameworks once stride > 1."""
    total = spec["batch_size"] * spec["n_batches"]
    img, classes = spec["image_size"], spec["classes"]
    c1, c2, k = spec["channels1"], spec["channels2"], spec["kernel_size"]
    ic = spec.get("in_channels", 1)
    use_padding = spec.get("use_padding", False)
    s1, s2 = spec.get("stride1", 1), spec.get("stride2", 1)
    if use_padding:
        assert s1 == 1 and s2 == 1, "strides require use_padding: false"
        # Same-padding keeps the spatial extent; two 2x pools halve it twice.
        fm = img // 2 // 2
    else:
        # conv-valid at stride, pool2, conv-valid at stride, pool2. OCANNL's valid conv
        # (and pool) require exact divisibility — assert instead of silently flooring.
        assert (img - k) % s1 == 0, "conv1: (image_size - kernel_size) mod stride1 != 0"
        c1_out = (img - k) // s1 + 1
        assert c1_out % 2 == 0, "pool1: conv1 output extent must be even"
        p1 = c1_out // 2
        assert (p1 - k) % s2 == 0, "conv2: (input - kernel_size) mod stride2 != 0"
        c2_out = (p1 - k) // s2 + 1
        assert c2_out % 2 == 0, "pool2: conv2 output extent must be even"
        fm = c2_out // 2
    fc1, fc2 = spec["fc1"], spec["fc2"]
    tensors["conv1_kernel"] = uniform(rng, k * k * ic, (c1, k, k, ic))
    tensors["conv1_bias"] = np.zeros(c1, np.float32)
    tensors["conv2_kernel"] = uniform(rng, k * k * c1, (c2, k, k, c1))
    tensors["conv2_bias"] = np.zeros(c2, np.float32)
    tensors["fc1_w"] = uniform(rng, fm * fm * c2, (fc1, fm, fm, c2))
    tensors["fc1_b"] = np.zeros(fc1, np.float32)
    tensors["fc2_w"] = uniform(rng, fc1, (fc2, fc1))
    tensors["fc2_b"] = np.zeros(fc2, np.float32)
    tensors["w_logits"] = uniform(rng, fc2, (classes, fc2))
    tensors["b_logits"] = np.zeros(classes, np.float32)
    tensors["x"] = rng.normal(0.0, 1.0, (total, img, img, ic)).astype(np.float32)
    tensors["y"] = one_hot(rng.integers(0, classes, total), classes)
    for key in ("image_size", "classes", "channels1", "channels2", "kernel_size", "fc1", "fc2"):
        meta[key] = str(spec[key])
    meta["in_channels"] = str(ic)
    meta["use_padding"] = "true" if use_padding else "false"
    meta["stride1"] = str(s1)
    meta["stride2"] = str(s2)


def build_gpt(spec, rng, tensors, meta):
    """GPT-2-style decoder, inference-only. OCANNL layouts:
    wte [d_model, vocab] (output d, input v — used transposed as the tied lm_head);
    wq/wk/wv [heads, d_head, d_model]; wo [d_model, heads, d_head];
    ffn w1 [d_ff, d_model], w2 [d_model, d_ff]; ln gammas/betas [d_model];
    wpe [seq, d_model]. Token ids and CE-target ids as integral float32 [total, seq]."""
    total = spec["batch_size"] * spec["n_batches"]
    d, v, seq = spec["d_model"], spec["vocab"], spec["seq_len"]
    nh, dh, dff = spec["n_head"], spec["d_model"] // spec["n_head"], spec["d_ff"]
    tensors["wte"] = uniform(rng, d, (d, v))
    tensors["wpe"] = (0.01 * rng.normal(0.0, 1.0, (seq, d))).astype(np.float32)
    for i in range(spec["n_layer"]):
        for w in ("wq", "wk", "wv"):
            tensors[f"l{i}_{w}"] = uniform(rng, d, (nh, dh, d))
        tensors[f"l{i}_wo"] = uniform(rng, nh * dh, (d, nh, dh))
        tensors[f"l{i}_ln1_g"] = np.ones(d, np.float32)
        tensors[f"l{i}_ln1_b"] = np.zeros(d, np.float32)
        tensors[f"l{i}_ln2_g"] = np.ones(d, np.float32)
        tensors[f"l{i}_ln2_b"] = np.zeros(d, np.float32)
        tensors[f"l{i}_ffn_w1"] = uniform(rng, d, (dff, d))
        tensors[f"l{i}_ffn_b1"] = np.zeros(dff, np.float32)
        tensors[f"l{i}_ffn_w2"] = uniform(rng, dff, (d, dff))
        tensors[f"l{i}_ffn_b2"] = np.zeros(d, np.float32)
    tensors["lnf_g"] = np.ones(d, np.float32)
    tensors["lnf_b"] = np.zeros(d, np.float32)
    tensors["ids"] = rng.integers(0, v, (total, seq)).astype(np.float32)
    tensors["tgt"] = rng.integers(0, v, (total, seq)).astype(np.float32)
    for key in ("n_layer", "n_head", "d_model", "d_ff", "vocab", "seq_len"):
        meta[key] = str(spec[key])


def build(spec_path: Path, out_dir: Path):
    spec = json.loads(spec_path.read_text())
    rng = np.random.default_rng(spec["seed"])
    model = spec.get("model", "mlp")
    tensors = {}
    meta = {
        "name": spec["name"],
        "model": model,
        "mode": spec.get("mode", "train"),
        "batch_size": str(spec["batch_size"]),
        "lr": repr(spec.get("lr", 0.0)),
        "seed": str(spec["seed"]),
        "parity_steps": str(spec["parity_steps"]),
        "warmup_steps": str(spec["warmup_steps"]),
        "timed_steps": str(spec["timed_steps"]),
    }
    {"mlp": build_mlp, "conv": build_conv, "gpt": build_gpt}[model](spec, rng, tensors, meta)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{spec['name']}.safetensors"
    save_file(tensors, str(out_path), metadata=meta)
    print(f"wrote {out_path} ({out_path.stat().st_size} bytes)")


if __name__ == "__main__":
    here = Path(__file__).parent
    specs = [Path(a) for a in sys.argv[1:]] or sorted((here / "workloads").glob("*.json"))
    for spec in specs:
        build(spec, here / "fixtures")
