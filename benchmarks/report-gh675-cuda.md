# gh-ocannl-675, CUDA leg: does a process that searched time its steps more slowly?

**Box `rog-nv-wsl` (RTX 5070 Ti Laptop, WSL2), 2026-08-23/24.** The sibling ROCm leg
(gfx1151) is the comment on gh-ocannl-675 dated 2026-08-23; this file is the NVIDIA half and the
one that settles the split/no-split call, because it ran second.

**Headline.** On this GPU the *only* searching cell whose timing process is measurably slower is
tinygrad's `--beam N` — by **+6.4%** on `mlp_small` and **+2.3%** on `gpt2_mini`, small but
carried by *every* paired repeat (9/9 and 4/4). `torch.compile` goes the other way: its
searching process is **faster** than the fresh process that replays inductor's cache
(−12.0% / −4.0%), so splitting its passes would penalise it, not credit it. OCANNL's own tuned
cell — the one the two-pass protocol exists for — shows a residue only after a *long* search
(+10.3% / +8.4% behind a 16 s search) and none at all behind a 4 s one. **The 2.5–3.5x that
`README.md` attributes to small CUDA kernels does not reproduce here, on the hardware class it
names.**

## Environment

| | |
|---|---|
| box | ASUS ROG, NVIDIA GeForce RTX 5070 Ti Laptop GPU (compute capability 12.0, 12227 MiB), WSL2, kernel 6.18.33.2-microsoft-standard-WSL2, 24 logical cores |
| driver / toolkit | driver 610.62 (CUDA API 13.3), CUDA toolkit 13.3 (`nvcc` V13.3.73) at `/usr/local/cuda` |
| PyTorch | `2.13.0+cu130` (stock Linux wheel), device `cuda` |
| tinygrad | 0.13.0, editable install from `~/tinygrad` at `62273d50f`, device **`CUDA`** (the system nvrtc works here; the README's `LD_LIBRARY_PATH` caveat applies only to a driver older than the toolkit, which this box no longer has) |
| other | python 3.12.13, numpy 2.5.1, safetensors 0.8.0 |
| ocannl | worktree of `origin/master` `7014dc44`, backend `cuda`, `BENCH_PRECISION=f32` |
| fixtures | `mlp_small.safetensors` sha256 `f09de950…298c44ca` (51384 B), `gpt2_mini.safetensors` sha256 `043c1ea8…7ca2009e` (13871360 B) |

The fixture digests differ from the ROCm leg's for the same workload names (same sizes, different
bytes: `gen_fixtures.py` draws from a numpy version that does not promise stream stability).
Absolute step times are therefore **not** comparable across the two legs — the *ratios* are what
both legs report, and a ratio is taken within one box, one fixture, one pair of processes.
`fixtures/DIGESTS.txt` still carries no entries, so these are recorded here rather than checked
against it.

## Method

- Driver: `benchmarks/gh675_cells.py` (checked in with this report). One arm = one pair of
  processes; arms alternate across repeats rather than running an arm to completion, so a drifting
  box moves both halves of a pair together. Each run is pinned with `taskset -c 0-15` behind an
  idle gate (load average and GPU utilisation).
- **X is the median of the paired per-repeat ratios** (pass-1 `step_ms` p50 over pass-2 `step_ms`
  p50, minus 1), not a ratio of arm-level medians. Repeat 0 of every arm is a discarded warm-up.
- Caches whose warmth *is* the experiment: tinygrad's kernel cache is redirected per arm with
  `CACHEDB=` and deleted before each pass 1; inductor gets a fresh `TORCHINDUCTOR_CACHE_DIR` per
  repeat with `TORCHINDUCTOR_FX_GRAPH_CACHE=1`; OCANNL's `autotune_cache/` is wiped before each
  pass 1.
- Provenance is verified, not assumed: every pass-1 row reports `searched: true` and every pass-2
  row `searched: false`. That needed the gh-ocannl-751 patch (see "Two things found on the way").
- Parity is the orchestrator's gate recomputed offline against `pytorch/cpu/eager`, tolerance
  2e-3: **every cell passes**, worst 8.7e-07, so all rows are comparable.

## Results

`X` = (pass-1 step p50) / (pass-2 step p50) − 1, median over paired repeats. `pos/n` is how many
of those paired ratios were positive — the statistic that separates a small real effect from
noise. `blk2/p2` is the retime control (see below). Step times are the sync-after-every-step
`step_ms` p50, i.e. the number the report publishes.

| workload | cell | pass-1 p50 (ms) | pass-2 p50 (ms) | n | **X** | pos/n | X IQR | `blk2/p2` | pass-1 `compile_s` |
|---|---|---|---|---|---|---|---|---|---|
| mlp_small | tinygrad/CUDA/beam, BEAM=2 | 0.1138 | 0.1047 | 9 | **+6.4%** | 9/9 | +4.4 .. +8.8 | 1.06 | 13.5–20.8 |
| mlp_small | tinygrad/CUDA/beam, BEAM=8 | 0.1116 | 0.1030 | 2 | **+8.3%** | 2/2 | +7.8 .. +8.9 | 1.07 | 53.0–57.1 |
| mlp_small | pytorch/cuda/compiled (default) | 0.5455 | 0.6154 | 9 | **−12.0%** | 1/9 | −17.6 .. −5.1 | 0.91 | 1.9–2.6 |
| mlp_small | pytorch/cuda/compiled, `max-autotune` | 0.4398 | 0.4657 | 9 | **−2.0%** | 1/9 | −9.4 .. −0.6 | 0.96 | 4.0–4.4 |
| mlp_small | *control* tinygrad/CUDA/jit, warm both passes | 0.1217 | 0.1217 | 9 | **−1.5%** | 3/9 | −6.9 .. +0.8 | 0.98 | 0.5 |
| mlp_small | *control* tinygrad/CUDA/jit, **cold compile, no search** | 0.1206 | 0.1208 | 9 | **−0.1%** | 4/9 | −0.8 .. +2.3 | 0.98 | 0.6–1.2 |
| mlp_small | *control* pytorch/cuda/eager | 0.5811 | 0.6030 | 9 | **−1.6%** | 3/9 | −3.9 .. +3.2 | 0.97 | 0.2 |
| mlp_small | *anchor* ocannl/cuda/tuned | 0.0501 | 0.0501 | 9 | **−0.0%** | 4/9 | −3.9 .. +2.3 | — | 3.6–16.4 |
| gpt2_mini | tinygrad/CUDA/beam, BEAM=2 | 1.5989 | 1.5446 | 4 | **+2.3%** | 4/4 | +2.0 .. +5.1 | 1.03 | 58.6–68.4 |
| gpt2_mini | pytorch/cuda/compiled (default) | 1.1461 | 1.1855 | 5 | **−4.0%** | 0/5 | −4.0 .. −0.8 | 1.00 | 3.7–4.6 |
| gpt2_mini | pytorch/cuda/compiled, `max-autotune` | 1.4710 | 1.4873 | 3 | **−1.1%** | 1/3 | −5.6 .. +2.2 | 1.00 | 10.0–11.4 |
| gpt2_mini | *control* tinygrad/CUDA/jit, warm both passes | 5.3569 | 5.4764 | 3 | **−2.5%** | 1/3 | −2.7 .. +0.5 | 1.00 | 1.0–1.1 |
| gpt2_mini | *control* tinygrad/CUDA/jit, **cold compile, no search** | 5.4763 | 5.3677 | 3 | **+0.1%** | 2/3 | −0.2 .. +2.6 | 1.02 | 1.1–1.8 |
| gpt2_mini | *control* pytorch/cuda/eager | 2.3969 | 2.3914 | 5 | **+0.6%** | 3/5 | −0.0 .. +1.9 | 1.02 | 0.2 |
| gpt2_mini | *anchor* ocannl/cuda/tuned | 9.6509 | 7.4012 | 3 | **+0.5%** | 2/3 | −0.4 .. +47.7 | — | 206–613 |

**Per-cell X on this hardware:** tinygrad `beam` **+6.4%** (mlp_small) / **+2.3%** (gpt2_mini);
`torch.compile` default **−12.0%** / **−4.0%**; `torch.compile max-autotune` **−2.0%** / **−1.1%**;
OCANNL `tuned` **−0.0%** / **+0.5%** pooled (but see the anchor section — pooling hides a split).

## What the controls say

1. **Process-to-process spread.** The three non-searching, non-compiling controls sit at |X| ≤ 2.5%
   at the median, and their paired ratios are a coin flip (3/9, 4/9, 3/9 positive on `mlp_small`).
   A single repeat of the 0.1 ms cell still swings ±10%, which is why the beam row is n=9 and why
   the sign count, not the range, is the statistic.
2. **Compilation is not the mechanism.** A tinygrad `--jit 1` cell run with a *deleted* kernel
   cache compiles every kernel in the timing process and comes out at −0.1% / +0.1% against its own
   warm replay, 4/9 and 2/3 positive. The only difference between that control and the beam arm is
   `BEAM=N`.
3. **It is not first-block warmup.** A `--retime` flag (added for this probe) times a *second*
   block of `timed_steps` inside the same process, ~400 steps in. The searching process stays slow:
   block-2 / pass-2 = **1.06** (mlp_small BEAM=2), **1.07** (BEAM=8), **1.03** (gpt2_mini), against
   0.97–0.98 for the controls and 0.91 for `torch.compile`, whose block 2 confirms it is genuinely
   the *faster* process.
4. **Beam width is a weak dose-response knob.** BEAM=8 costs 3.4x the search (55 s vs 16 s) and
   moves X from +6.4% to +8.3% — present, but far from proportional.
5. **The effect is small and one-sided, not large and noisy.** 9/9 + 4/4 + 2/2 positive paired
   ratios across the three beam rows is what makes a ~6% median statable at all; by contrast
   `torch.compile`'s 1/9 and 0/5 make its *negative* sign equally real.

## The OCANNL anchor, and why the 2.5–3.5x is not reproduced

`ocannl/cuda/tuned` on `mlp_small` splits cleanly in two by how long its own search took:

| regime | repeats | pass-1 `compile_s` | X |
|---|---|---|---|
| long search | 1, 2 (and the discarded warm-up 0) | 16.4 s | **+10.3%**, **+8.4%** (warm-up: +34.6%) |
| short search | 3–9 | 3.6–5.2 s | **−0.2%** median, 3/7 positive |

Every one of those ten passes genuinely searched (`searched: true`, `searches: 2`, `replays: 0`
in the `tune` object) — the provenance field rules out the obvious explanation that the later
runs replayed a cache. What changed is the *cost* of the search: the CUDA driver's JIT cache
(`~/.nv/ComputeCache`, 323 MB with 17,306 files written during this session) is warmed by every
preceding cell, so a "from-scratch" OCANNL search stops being from-scratch after the first few
repeats even with `autotune_cache/` wiped. The residue tracks the length of the search that
preceded the timing, which is consistent with the README's stated mechanism (a searching process
accumulates modules and buffers) — and it means **`compile_s` in any published CUDA report is a
warm-JIT number unless `~/.nv/ComputeCache` was cleared too.**

Either way the headline number in `README.md` does not survive: the largest OCANNL residue
measured here is **+10.3%** on the smallest kernels, against a documented **2.5–3.5x**; on
`gpt2_mini` it is +0.5%; and the ROCm leg measured −0.2% / −0.1%. Two boxes, two backends, no
reproduction.

## The call: keep every cell single-pass

Reading both legs against the brief's rule (split a searching cell whose `mlp_small` X exceeds
~10%):

| cell | CUDA (this leg) | ROCm (gfx1151) | verdict |
|---|---|---|---|
| tinygrad `--beam N` | +6.4% / +2.3% | +14.9% / +9.0% | **no split** — over the line on one box, well under on the other |
| `torch.compile` default | −12.0% / −4.0% | +7.2% / +21.2% | **no split** — the sign itself is box-dependent |
| `torch.compile max-autotune` | −2.0% / −1.1% | +12.3% / +33.9% | **no split**; add the variant single-pass if it is added |
| OCANNL `tuned` (anchor) | +10.3% behind a 16 s search, ≈0 behind a 4 s one | −0.2% / −0.1% | protocol kept, rationale re-stated |

No cell clears the line on both boxes, and no cell clears it on the box whose hardware the
README's own rationale names. What the numbers do establish is that the *residue is a property of
the search's cost on that device*, not of the framework: it appears wherever a long search
precedes the timing (OCANNL behind 16 s of search on CUDA, tinygrad's beam on both boxes, torch's
max-autotune on ROCm) and vanishes when the search is cheap. So the honest matrix rule is
"single pass, measured at ≤X% per cell per box", with X recorded — which is a result, and it is
what makes the asymmetry defensible instead of merely documented.

The two-pass protocol stays on the OCANNL tuned cell: it is cheap there (the cell is already
two processes), it is the only cell whose `compile_s` is *defined* as a from-scratch search cost,
and it is the only one gated on provenance. It is now justified by a measured ≤10.3% rather than
by an unreproduced 2.5–3.5x.

## Two things found on the way (not part of the measurement)

1. **The runner leaves `runners/` on `sys.path`, which wedges every parallel beam search.**
   `runners/tinygrad/run.py` inserts its parent directory on `sys.path` to import `bench_common`
   — after importing tinygrad, precisely so the `runners/tinygrad/` directory cannot shadow the
   real package. But tinygrad's beam search compiles its candidates in a **`spawn` pool**, and a
   spawned worker re-executes the module top-level with the *parent's* `sys.path`, where the
   `import tinygrad` at line 39 now runs with `runners/` still in front. The path scan finds
   `runners/tinygrad/` as a namespace portion, the editable install's meta-path finder (appended
   *after* `PathFinder`) never gets a look, and every worker dies with
   `ImportError: cannot import name 'Tensor' from 'tinygrad' (unknown location)`. The pool
   respawns them forever, so the search **hangs instead of failing**: the first attempt here sat
   35 minutes at 3% CPU with the GPU idle and had to be killed. Fixed in this branch by dropping
   the path entry again after the import; a cold BEAM=2 search then finishes in 28 s. This affects
   any device whose beam search parallelises (`CUDA`, `AMD`, `NV`, `METAL`, `HIP`) with an editable
   tinygrad — i.e. exactly the canonical `orchestrate.py --beam N` run on this box.
2. **The `searched` provenance probe is stale for tinygrad ≥ 0.13** (gh-ocannl-751, filed by the
   ROCm leg with the patch): `instrument_tinygrad_beam` binds `tinygrad.engine.search`, which
   0.13 replaced with `tinygrad.codegen.opt.search`, so every beam cell reports `searched: null`
   and the report's `pass` column prints `?`. The patch from that issue is applied in this branch;
   without it none of the beam rows above could have stated which pass produced them.

Also observed, and **not** explained: one `mlp_small` BEAM=8 pass-1 and one `gpt2_mini` BEAM=2
pass-1 wedged the same way as the `sys.path` bug did (minutes at ~1% CPU, GPU idle) *after* that
bug was fixed, against 55 s and 63 s for the same searches in their other repeats. Both were
killed by the driver's cap and their pairs dropped (hence n=2 and n=4 in those rows). The ROCm
leg reports one wedge of the same shape. Something in tinygrad's parallel beam pool deadlocks
intermittently; it is worth its own issue (gh-ocannl-760).

Raw per-run records (every arm, every repeat, losses, `queued_step_ms`, `retime_step_ms`,
`searched`, the OCANNL `tune` objects) are kept on the box at `$GH675_OUT/records.jsonl`
(99 pairs); say the word and I will attach them.
