#!/usr/bin/env python3
"""gh-ocannl-531: generate the kernel variants that ffn1_nvrtc_harness.c times.

Input is arm A's emitted CUDA source -- the 117-kernel, 20-`mma_sync` snapshot described in
benchmarks/report-gh531-profile.md. Output is that same file with one kernel rewritten, so the
harness always compiles a real emitted translation unit rather than a hand-written stand-in.

    python3 benchmarks/ffn1_make_variants.py armA-117.cu outdir/

Produces, for each chunk count C in 1,2,4,8,16,32,64,128:

  seg25_chunk<C>.cu    `seg25`'s two output loops (the GEMM's `j` and the gelu's `j`) rebased onto
                       blockIdx.y, so block (x,y) owns the same token as before and a 1024/C slice
                       of the output axis. Launch with grid=(8,C). Legal as-is: `j` carries no
                       reduction here.

  seg111_split<C>.cu   `seg111` FISSIONED into two kernels, because its output axis is the
                       vocabulary and that axis carries the `max_logits` reduction -- chunking it
                       in place would have every block write a partial maximum into one cell.
                         cross_entropy_loss_fwd__seg111_gemm    the logits GEMM, `v` chunked, grid=(8,C)
                         cross_entropy_loss_fwd__seg111_reduce  the max_logits init + row max, grid=(8,1)
                       Time the pair and add them; the reduce half is identical for every C, so
                       measure it from seg111_split1.cu (the harness runs the GEMM half first to
                       fill the logits, at grid=(8,1), which only covers the whole vocabulary when
                       the GEMM is unchunked).

The rewrite is textual and asserted at every step: if the emitted source ever stops matching these
loop shapes, this script fails rather than silently producing something that is not the shipped
kernel.
"""

import os
import re
import sys

CHUNKS = [1, 2, 4, 8, 16, 32, 64, 128]
VOCAB = 1024
DFF = 1024


def rebase_loop(src, var, per, count=1):
    """Rewrite `for (int v = 0; v <= 1023; ++v)` onto a blockIdx.y-selected slice."""
    old = "for (int %s = 0; %s <= 1023; ++%s)" % (var, var, var)
    if src.count(old) < count:
        raise SystemExit("expected loop over %s not found: %r" % (var, old))
    new = ("for (int {v} = (int)blockIdx.y * {p}; "
           "{v} < ((int)blockIdx.y + 1) * {p}; ++{v})").format(v=var, p=per)
    return src.replace(old, new, count)


def kernel_span(src, name):
    start = src.index('extern "C" __global__ void %s(' % name)
    nxt = src.index('extern "C" __global__ void ', start + 1)
    return start, nxt


def top_level_blocks(body):
    """The `{ ... }` statement blocks of a kernel body, in order."""
    inner = body[body.index("/* Main logic. */") + len("/* Main logic. */"):body.rindex("}")]
    blocks, depth, cur = [], 0, ""
    for ch in inner:
        if ch == "{":
            depth += 1
            if depth == 1:
                cur = ""
                continue
        if ch == "}":
            depth -= 1
            if depth == 0:
                blocks.append(cur)
                continue
        if depth >= 1:
            cur += ch
    return blocks


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    src = open(sys.argv[1]).read()
    outdir = sys.argv[2]
    os.makedirs(outdir, exist_ok=True)

    # Sanity: this must be arm A, the shipping artifact, not arm B's discarded emission.
    n_glob = src.count("__global__")
    n_mma = src.count("mma_sync")
    if n_glob != 117 or n_mma != 20:
        raise SystemExit(
            "input does not look like arm A (%d kernels, %d mma_sync; expected 117 and 20). "
            "See report-gh531-profile.md: the .cu left on disk after a tuned run is arm B."
            % (n_glob, n_mma))

    # --- seg25: rebase both output loops in place ---------------------------
    s25, e25 = kernel_span(src, "cross_entropy_loss_fwd__seg25")
    body25 = src[s25:e25]
    gemm_var = re.search(r"for \(int (i\d+) = 0; \1 <= 1023; \+\+\1\) \{\s*\n\s*for \(int i\d+ = 0;"
                         r" i\d+ <= 255", body25).group(1)
    gelu_var = [v for v in re.findall(r"for \(int (i\d+) = 0; \1 <= 1023", body25) if v != gemm_var][0]
    for c in CHUNKS:
        per = DFF // c
        b = rebase_loop(body25, gemm_var, per)
        b = rebase_loop(b, gelu_var, per)
        open(os.path.join(outdir, "seg25_chunk%d.cu" % c), "w").write(src[:s25] + b + src[e25:])

    # --- seg111: fission, then chunk the GEMM half --------------------------
    s111, e111 = kernel_span(src, "cross_entropy_loss_fwd__seg111")
    body111 = src[s111:e111]
    blocks = top_level_blocks(body111)
    if len(blocks) != 3:
        raise SystemExit("seg111 was expected to be GEMM + max init + max reduce, got %d blocks"
                         % len(blocks))
    sig = ('extern "C" __global__ void cross_entropy_loss_fwd__seg111%s(\n'
           '    const int i1,\n'
           '    float *__restrict__ logits,\n'
           '    float *__restrict__ max_logits,\n'
           '    float *__restrict__ n794_layer_norm,\n'
           '    float *__restrict__ wte) {\n')
    v_var = re.search(r"for \(int (i\d+) = 0; \1 <= 1023", blocks[0]).group(1)
    for c in CHUNKS:
        gemm = rebase_loop(blocks[0], v_var, VOCAB // c)
        out = src[:s111]
        out += (sig % "_gemm") + "{" + gemm + "}\n}\n"
        out += (sig % "_reduce") + "{" + blocks[1] + "}\n{" + blocks[2] + "}\n}\n"
        out += src[e111:]
        open(os.path.join(outdir, "seg111_split%d.cu" % c), "w").write(out)

    print("wrote %d seg25 and %d seg111 variants to %s" % (len(CHUNKS), len(CHUNKS), outdir))


if __name__ == "__main__":
    main()
