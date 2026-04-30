# Lean Attention (Flash Attention + softmax-as-reduce): Feasibility Study

**Task**: gh-ocannl-263
**Issue**: https://github.com/ahrefs/ocannl/issues/263
**Milestone**: v0.9 (program search / performance explorations)
**Status**: Research / feasibility — write-up deliverable

## Goal

Assess whether and how OCANNL should support [Lean Attention](https://arxiv.org/abs/2405.10480)
(Flash Attention extended with softmax-as-reduce and stream-K tile distribution
for the decoder-only decode phase). The deliverable is a written feasibility
assessment that determines the prerequisites OCANNL needs before any concrete
implementation can land, picks an implementation strategy from the available
options (external fused kernel vs. IR-level expression vs. program-search
discovery), and produces actionable follow-up tasks. This is an `explore` task:
no production code changes are required by this proposal.

## Acceptance Criteria

The deliverable is a write-up (extending or replacing this proposal) plus a
summarizing comment on issue #263. The write-up must cover:

- [ ] **Algorithm summary**: a precise description of online softmax, the
      Flash Attention forward pass, and the additions Lean Attention makes
      (stream-K-style work distribution across attention tiles, decode-phase
      specialization where Q has one row but K/V span the full context).
- [ ] **Feasibility assessment for OCANNL**: identify which of OCANNL's existing
      mechanisms can already express each piece (e.g. axis-aligned reductions,
      fused composition of primitives, einsum-style projections) and which
      cannot.
- [ ] **Prerequisites enumeration**: a concrete, ordered list of OCANNL
      capabilities that must exist before Lean Attention can be implemented in
      any meaningful form. At minimum the assessment must cover: tiling
      (v0.8 / #412), kernel fusion / megakernels (#318), shared memory and
      synchronization in generated CUDA, and multi-value reductions (online
      softmax keeps `(m, l, o)` — running max, running sum-of-exp, running
      output — and merges them with a non-trivial associative combiner).
- [ ] **Strategy recommendation**: an explicit recommendation among
      Option A (external handcrafted / cuDNN-fused kernel bound as a custom op),
      Option B (extend the low-level IR with the missing primitives so OCANNL
      generates the kernel), and Option C (define attention naively and rely on
      v0.9 program search to discover an efficient schedule). The recommendation
      must justify the choice in light of OCANNL's stated mission (a
      compiler / DSL, not a hand-tuned kernel library) and the timeline
      (v0.9 = Aug 2026, ICFP week).
- [ ] **Follow-up tasks**: a list of harness/GitHub issues that should be
      filed to track the prerequisites and the eventual implementation, with
      explicit dependencies (e.g. "blocked by #412 tiling", "depends on
      decision in this write-up").
- [ ] **Decode-phase scope**: the write-up notes whether the recommendation
      addresses decode-only (Lean Attention's target regime, batch=1,
      `seq_q=1`, large KV cache) or also prefill / training, and whether
      prefill or training Flash Attention is in scope as a stepping stone.
- [ ] **Out-of-scope clarity**: the write-up explicitly states what is
      *not* answered (e.g. backwards pass, multi-query / grouped-query
      attention specifics, paged KV cache integration, cross-backend story
      for Metal).
- [ ] A comment summarizing the findings is posted on
      [issue #263](https://github.com/ahrefs/ocannl/issues/263).

## Context

### How attention currently works in OCANNL

Multi-head attention is composed from primitives in
`lib/nn_blocks.ml`:

- `softmax ~spec` (around the `let%op softmax` definition) is built from
  `max-subtraction (@^^)`, `exp`, and `sum-reduce (++)` — i.e. the
  numerically-stable but unfused three-pass softmax.
- `multi_head_attention` (same file) projects Q/K/V, computes scores via
  einsum, applies `softmax` along the key axis, then einsums with V. There is
  no fused attention kernel and no online softmax.
- `cross_attention` and the masked variants reuse the same `softmax` helper.

Reductions and einsum are expressed via `Tensor.operation` and lowered to
`arrayjit/lib/low_level.ml`'s `For_loop` / `Set` / `Get` IR. Code generation
flows through `arrayjit/lib/c_syntax.ml` and the backends
(`arrayjit/lib/cuda_backend.ml`, `arrayjit/lib/metal_backend.ml`,
`arrayjit/lib/cc_backend.ml`).

### What's missing for Lean Attention

A single kernel that streams a Q tile, scans K/V tiles, and updates running
`(m, l, o)` triples in registers / shared memory requires:

1. **Loop tiling as an IR pass** — splitting an iteration into outer (tile)
   and inner (element) loops with controllable tile sizes. Tracked by #412.
   Today every CUDA kernel runs with `grid_dim=1, block_dim=1` (the
   `kernel_prep_line` guard in `cuda_backend.ml` returns early on every
   thread except `(0,0)`). Lean Attention is meaningless without removing
   this guard and mapping outer tile loops to `blockIdx` / inner loops to
   `threadIdx`.
2. **Kernel fusion / megakernels** — combining the three softmax passes plus
   the `S = QK^T` and `O = SV` einsums into one kernel. Tracked by #318.
3. **Shared memory and barriers** — the IR/codegen need a way to declare
   `__shared__` buffers and emit `__syncthreads()` (CUDA) /
   `threadgroup` + `threadgroup_barrier` (Metal). Today neither is emitted.
4. **Multi-value reductions** — online softmax's reduction operator merges
   triples `(m_a, l_a, o_a) ⊕ (m_b, l_b, o_b)` with a numerically-stable
   correction. OCANNL's reductions today are scalar-valued
   (`Accum_op` with a single accumulator over a single LHS). Either the IR
   gains tuple-valued reductions, or the writer manually splits the state
   across three tensors with hand-written cross-tile bookkeeping.
5. **Stream-K work distribution** — Lean Attention's distinguishing feature
   (vs. plain Flash Attention) is partitioning attention tiles across
   streaming multiprocessors so all SMs stay busy when Q has one row.
   Expressing this requires either explicit grid-index control surfaced
   to the frontend, or a scheduler smart enough to discover it. Neither
   exists today.
6. **A way to bind external code (Option A)** — if OCANNL chooses to call
   cuDNN's fused attention or a hand-written CUDA kernel, the compiler
   needs an FFI for "this assignment's RHS is implemented by this opaque
   kernel". That mechanism does not currently exist as a first-class
   construct; backends emit C/CUDA from `low_level.ml` and link the
   resulting `.so`/`.metallib` — there is no path for a precompiled blob.

### Related work in the repository

- **#412 — Tiling and multi-threaded GPU kernels**: the v0.8 milestone. A
  proposal already exists (`docs/proposals/gh-ocannl-412.md`). Lean
  Attention is downstream of this.
- **#318 — Megakernels / kernel fusion**: prerequisite for any fused
  attention kernel.
- **#242 — TVM deep dive** (`docs/proposals/gh-ocannl-242.md`): TVM's
  schedule primitives and tensorization story are directly relevant; in
  particular TVM's tensorize and split / reorder / cache_read primitives
  are what an Option B IR would need to subsume.
- **#261 — Superoptimizers / program search for tensor programs**: the
  v0.9 vehicle. Option C lives here: define attention naively and let
  search discover Flash/Lean.
- **OCANNL mission alignment**: the README and ROADMAP frame OCANNL as a
  compiler / DSL inspired by tinygrad, Luminal, TVM — not as a curated
  library of hand-tuned kernels. Option A (external fused kernel) is
  therefore in tension with the project's stated direction; Option B and
  Option C are more aligned but more expensive to deliver.

### Decode-phase specifics

Lean Attention's gains come from the decode regime: `seq_q = 1`, `seq_kv`
in the tens to hundreds of thousands, batch typically small. This is the
inference / autoregressive-generation case, not training. For OCANNL,
this matters because:

- Training-time Flash Attention has different tile shapes and a backward
  pass; conflating the two scopes makes the write-up unfocused.
- OCANNL's transformer examples (#65, #66, #74) are training/inference
  toys; a real LLM-inference integration (paged KV cache, GQA, etc.) is
  not on the roadmap until after v1.0.
- Whether this task should produce a decode-only Lean Attention
  implementation, a training-time Flash Attention implementation, or
  both, is one of the strategic questions the write-up must answer.

## Approach

*Suggested approach — agents may deviate if they find a better path.*

The deliverable is a research write-up, not a code change. A reasonable
sequence:

1. **Read and summarize the paper** — Lean Attention (arxiv 2405.10480),
   plus Flash Attention 1 and 2 (arxiv 2205.14135, 2307.08691) for
   background, plus the online softmax derivation.
2. **Map each algorithmic primitive to OCANNL** — which can be expressed
   today (e.g. einsum projections, axis-aligned reductions); which need
   IR extensions (multi-value reductions, shared memory); which require
   scheduler / search support (stream-K tile distribution).
3. **Cross-reference the existing proposals** for #412, #318, #261, #242
   so that the prerequisite list reuses already-tracked work rather
   than inventing parallel issues.
4. **Compare strategies (A/B/C)** along axes: time-to-first-result,
   alignment with OCANNL's compiler-not-library mission, maintenance
   cost, generality (does the chosen path also help non-attention
   workloads?), and dependence on third-party kernels.
5. **Write the assessment** by extending this file in place. The final
   document replaces the "this is a study" framing with a concrete
   recommendation and a list of follow-up tasks.
6. **Post a summary comment** on issue #263 linking back to the
   committed write-up.
7. **Optionally file follow-up GitHub issues** for any prerequisites that
   are not already tracked, with `blocked_by` set to the relevant
   existing issues.

The agent should not attempt to implement Flash or Lean Attention as
part of this task; doing so prematurely (before tiling, megakernels,
and shared-memory codegen exist) would necessarily fall back to
Option A (an external kernel), which preempts the strategic decision
this study is meant to inform.

## Scope

**In scope**:
- A written feasibility assessment in this file.
- Strategy recommendation (A vs. B vs. C).
- Enumeration of prerequisites, cross-referenced against existing
  ROADMAP items and GitHub issues.
- A summarizing comment on issue #263.
- Optional: filing follow-up GitHub issues for any new prerequisites
  not already tracked.

**Out of scope**:
- Any implementation of Flash Attention, Lean Attention, or online
  softmax. These wait until the prerequisites land (v0.8 tiling /
  megakernels at the earliest).
- Backwards pass / training-time Flash Attention details beyond noting
  whether they are in or out of scope for the eventual implementation.
- Multi-query / grouped-query attention, paged KV cache, speculative
  decoding integration. These are layered concerns that can be added
  after a baseline Flash/Lean Attention exists.
- Metal-specific or CPU-backend implementation details (the assessment
  may note these as future work but is not required to design them).
- Changes to `lib/nn_blocks.ml`, `arrayjit/lib/low_level.ml`,
  `arrayjit/lib/cuda_backend.ml`, or any other source file. This task
  produces documentation only.

**Dependencies**:
- Logical dependency on #412 (tiling) and #318 (megakernels) for any
  eventual implementation; the present *study* does not block on them.
- Related to #242 (TVM deep dive) and #261 (superoptimizers) — the
  study's strategy comparison should reference whatever conclusions
  those tasks reach if they have landed first.
