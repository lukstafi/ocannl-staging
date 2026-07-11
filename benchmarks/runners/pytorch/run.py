#!/usr/bin/env python3
"""PyTorch runner for the cross-framework benchmark suite (see benchmarks/README.md).

Model-dispatched on the fixture metadata: mlp / conv (LeNet-5, valid convs) / gpt
(pre-LN GPT-2-style decoder, inference-only). Math mirrors the OCANNL runners exactly:
same CE formulation (softmax -> one-hot dot -> log), same LayerNorm (biased variance,
eps inside sqrt), same tanh-gelu constant, same -1e9 causal-mask fill, same conv layouts
(fixture is channels-last / OCANNL axis order; this runner permutes to NCHW once).
"""

import argparse
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bench_common import emit, percentiles, read_st_metadata

import torch
import torch.nn.functional as F
from safetensors.torch import load_file


def ce_onehot(logits, y_onehot):
    probs = torch.softmax(logits, dim=-1)
    correct = (probs * y_onehot).sum(dim=-1)
    return (-correct.log()).mean()


def build_mlp(meta, data, dev):
    n_layers = int(meta["n_layers"])
    params = []
    for i in range(1, n_layers + 1):
        params.append(
            (data[f"w{i}"].to(dev).requires_grad_(), data[f"b{i}"].to(dev).requires_grad_())
        )
    flat = [p for wb in params for p in wb]

    def forward(xb):
        h = xb
        for i, (w, b) in enumerate(params):
            h = h @ w.T + b
            if i < n_layers - 1:
                h = torch.relu(h)
        return h

    def loss_fn(xb, yb):
        return ce_onehot(forward(xb), yb)

    x, y = data["x"].to(dev), data["y"].to(dev)
    return loss_fn, flat, (x, y), None


def build_conv(meta, data, dev):
    def leaf(t):
        return t.contiguous().to(dev).requires_grad_()

    w1 = leaf(data["conv1_kernel"].permute(0, 3, 1, 2))  # [oc,ic,kh,kw]
    b1 = leaf(data["conv1_bias"])  # per-channel bias [oc]
    w2 = leaf(data["conv2_kernel"].permute(0, 3, 1, 2))
    b2 = leaf(data["conv2_bias"])
    wf1 = leaf(data["fc1_w"].reshape(data["fc1_w"].shape[0], -1))  # [hid, oh*ow*oc]
    bf1 = leaf(data["fc1_b"])
    wf2 = leaf(data["fc2_w"])
    bf2 = leaf(data["fc2_b"])
    wl = leaf(data["w_logits"])
    bl = leaf(data["b_logits"])
    flat = [w1, b1, w2, b2, wf1, bf1, wf2, bf2, wl, bl]

    def forward(xb):
        z = F.conv2d(xb, w1, b1)  # valid convolution
        z = F.max_pool2d(torch.relu(z), 2)
        z = F.conv2d(z, w2, b2)
        z = F.max_pool2d(torch.relu(z), 2)
        h = z.permute(0, 2, 3, 1).reshape(z.shape[0], -1)  # match OCANNL [oh,ow,oc] order
        h = torch.relu(h @ wf1.T + bf1)
        h = torch.relu(h @ wf2.T + bf2)
        return h @ wl.T + bl

    def loss_fn(xb, yb):
        return ce_onehot(forward(xb), yb)

    x = data["x"].permute(0, 3, 1, 2).contiguous().to(dev)  # NCHW
    y = data["y"].to(dev)
    return loss_fn, flat, (x, y), None


def gelu_tanh(x):
    return 0.5 * x * (1.0 + torch.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))


def layernorm(x, g, b):
    mu = x.mean(-1, keepdim=True)
    c = x - mu
    var = (c * c).mean(-1, keepdim=True)
    return c / torch.sqrt(var + 1e-5) * g + b


def build_gpt(meta, data, dev):
    n_layer, nh = int(meta["n_layer"]), int(meta["n_head"])
    d, v, seq = int(meta["d_model"]), int(meta["vocab"]), int(meta["seq_len"])
    dh = d // nh
    wte = data["wte"].to(dev)  # [d, v]
    emb = wte.T.contiguous()  # [v, d]
    wpe = data["wpe"].to(dev)
    layers = []
    for i in range(n_layer):
        layers.append(
            {
                "wq": data[f"l{i}_wq"].reshape(nh * dh, d).to(dev),
                "wk": data[f"l{i}_wk"].reshape(nh * dh, d).to(dev),
                "wv": data[f"l{i}_wv"].reshape(nh * dh, d).to(dev),
                "wo": data[f"l{i}_wo"].reshape(d, nh * dh).to(dev),
                "g1": data[f"l{i}_ln1_g"].to(dev),
                "b1": data[f"l{i}_ln1_b"].to(dev),
                "g2": data[f"l{i}_ln2_g"].to(dev),
                "b2": data[f"l{i}_ln2_b"].to(dev),
                "fw1": data[f"l{i}_ffn_w1"].to(dev),
                "fb1": data[f"l{i}_ffn_b1"].to(dev),
                "fw2": data[f"l{i}_ffn_w2"].to(dev),
                "fb2": data[f"l{i}_ffn_b2"].to(dev),
            }
        )
    gf, bf = data["lnf_g"].to(dev), data["lnf_b"].to(dev)
    mask = torch.tril(torch.ones(seq, seq, dtype=torch.bool, device=dev))  # [s, t]

    def forward(ids):
        b = ids.shape[0]
        x = emb[ids] + wpe
        for p in layers:
            h = layernorm(x, p["g1"], p["b1"])
            q = (h @ p["wq"].T).view(b, seq, nh, dh)
            k = (h @ p["wk"].T).view(b, seq, nh, dh)
            vv = (h @ p["wv"].T).view(b, seq, nh, dh)
            att = torch.einsum("bshd,bthd->bsth", q, k) / math.sqrt(dh)
            att = torch.where(mask[None, :, :, None], att, torch.tensor(-1e9, device=dev))
            att = torch.softmax(att, dim=2)
            out = torch.einsum("bsth,bthe->bshe", att, vv).reshape(b, seq, nh * dh)
            x = x + out @ p["wo"].T
            h2 = layernorm(x, p["g2"], p["b2"])
            x = x + (gelu_tanh(h2 @ p["fw1"].T + p["fb1"]) @ p["fw2"].T + p["fb2"])
        x = layernorm(x, gf, bf)
        return x @ wte  # tied lm_head

    def loss_fn(ids, tgt_onehot):
        logp = torch.log_softmax(forward(ids), dim=-1)
        return -((logp * tgt_onehot).sum(-1)).mean()

    ids = data["ids"].long().to(dev)
    tgt = F.one_hot(data["tgt"].long(), v).float().to(dev)
    return loss_fn, [], (ids, tgt), int(meta["batch_size"]) * seq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", required=True)
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps"])
    ap.add_argument("--compile", action="store_true")
    args = ap.parse_args()

    meta = read_st_metadata(args.fixture)
    model = meta.get("model", "mlp")
    mode = meta.get("mode", "train")
    batch_size = int(meta["batch_size"])
    lr = float(meta["lr"])
    parity_steps = int(meta["parity_steps"])
    warmup_steps = int(meta["warmup_steps"])
    timed_steps = int(meta["timed_steps"])

    torch.set_float32_matmul_precision("highest")
    dev = torch.device(args.device)
    data = load_file(args.fixture)
    build = {"mlp": build_mlp, "conv": build_conv, "gpt": build_gpt}[model]
    loss_fn, flat, (x, y), tokens_per_step = build(meta, data, dev)
    n_batches = x.shape[0] // batch_size
    batches = [
        (x[i * batch_size : (i + 1) * batch_size], y[i * batch_size : (i + 1) * batch_size])
        for i in range(n_batches)
    ]

    if args.compile:
        loss_fn = torch.compile(loss_fn)

    if mode == "train":

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

    else:

        def step(k):
            xb, yb = batches[k % n_batches]
            with torch.no_grad():
                return loss_fn(xb, yb)

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

    result = {
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
    if tokens_per_step:
        result["tokens_per_step"] = tokens_per_step
    emit(result)


if __name__ == "__main__":
    main()
