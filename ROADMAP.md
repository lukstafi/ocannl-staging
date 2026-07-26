# OCANNL Roadmap to v1.0

**Headline target: ICFP 2026 week (August 24, 2026)**

This roadmap outlines the development plan for OCANNL from the current state to version 1.0, incorporating academic paper milestones for workshops collocated with ICFP 2026 (OCaml Workshop, FProPer). Dates indicate **end of period** targets.

> **Schedule note (July 2026):** the roadmap drifted from its original dating because of a slowdown between January and May 2026. v0.7 is now the catch-up release. Three structural changes follow from that:
>
> - **v0.6.4 is skipped as a release.** Its scope — axis concatenation/block tensors (#49), RoPE and non-learned position embeddings (#398), the decoder-only transformer toy (#57) — is complete (the GitHub milestone is closed), but it ships inside **v0.7** rather than as a separate tagged release. The last tagged release before v0.7 was **0.6.3**.
> - **v0.7.2 is consolidated into v0.7.** The compiler-optimization and memory-management work that was scheduled separately (loop hoisting, CSE, the universal pool allocator) is part of the single **v0.7** milestone.
> - **v0.7.1 was dissolved.** Its two tracks were redistributed: the **AMD HIP backend (#411)** shipped in **v0.8**, and the **real-world examples** (makemore #59, CNN/CIFAR #54, LSTM #60, transformer inference #377) and **tokenizer bindings** moved to **v0.9**. The GitHub milestone has been deleted.
>
> The version sequence is now: `0.7 → 0.8 → 0.9 → 1.0 → 1.1`. Milestone *scope* below tracks the GitHub milestones, which are the source of truth.

---

## Released: Foundation (through 0.6.3)

The 0.6.x line stabilized the frontend: the Menhir einsum parser and "missing hidden dimensions" error detection (0.6.2), then padding inference for convolutions with a toy CNN (0.6.3). See [CHANGES.md](CHANGES.md) for details.

---

## v0.7 — July 3, 2026
**Theme: Frontend finalization and compiler optimizations (paper-ready)**

This is the consolidated "paper-ready" release. It absorbs the frontend-finalization work originally split across v0.6.4/v0.6.5/v0.7.0 and the compiler-optimization work originally planned as v0.7.2. GitHub milestone scope: *"inlining- and simplification-related optimizations, memory management, session management."*

**Frontend finalization (done):**
- **Remove the hosted tensor mode** (#333) — got rid of the `array` field of `Tnode.t` and the "hosted" memory mode; value access and printing are now context-mediated.
- **Tensor persistence** (#373) — tensor saving, loading, and restoring.
- **Axis concatenation / block tensors** (#49) — `a^b` einsum syntax for stacking/concatenation, with shifting (`1^i=>i`) and padding (`i=>1^i`) as fixed-index special cases; n-ary block-tensor specs.
- **RoPE and non-learned position embeddings** (#398).
- **Decoder-only autoregressive transformer toy example** (#57).
- **Ternary einsum notation** (#305) and ternary projection inference.
- **Sasha Rush Tensor Puzzles** (#308) in extended einsum notation.
- **uint32/uint64 indexing precisions** (#349, #177) driven by the `big_models` setting.
- Identifier hygiene: blacklist primitive-operator/reserved names (#383); collapse repeated label components in `debug_name` (#281).
- Configuration: relax the required `ocannl_` CLI prefix and validate config keys (#409).
- `-march=native` C-compiler flag (#311); restore CUDA pre-loaded builtins via a cudajit helper (#353); remove remaining unnecessary buffer zeroing (#382); rename routine/kernel params to `kparam`/`kparams` (#356).

**Compiler optimizations (done):**
- **Loop-invariant code motion** (#350), prior to visit counting.
- **Common subexpression elimination** after inlining (#351).
- Extend virtual-node inlining to non-scalar constants and ranges (#142).
- **Universal pool allocator across backends** (#344) — tensors are addressed through pooled locations; working tensors are bump-packed per context delta, constants live in per-device pools, merge buffers stay reserved, and Metal uses pool slabs plus a slot table to avoid binding-limit pressure.
- **Sharding and minimal-copy slicing foundation** (#293) — `shard_along` / `gather`, data-parallel training with merge-buffer all-reduce, and zero-copy leading-axis slice views.

**Documentation and paper artifacts (done):**
- **`lowering_and_inlining.md` audit** (#296) — the lowering/optimization docs were fleshed out alongside a `low_level.ml` audit.
- **Workshop article:** `docs/ocannl_workshop_article_human.md`, LaTeX source, and rendered PDF.
- **Formal core technical report:** `docs/ocannl-formal-core-technical-report.md` and LaTeX source, covering the core shape/projection inference proof effort.
- **Shape constraint generation notes:** `docs/shape-constraint-generation.md`, documenting the front-end elaboration boundary from `shape.ml` into core constraints.

**Deferred after v0.7:**
- **Tensor-node ID namespaces** (#372).
- **`Local_scope` initialization tracking** (#340).
- Remaining sharding/slicing extensions beyond the v0.7 data-parallel and zero-copy leading-axis foundation (#293 follow-ups).
- Inlining stretch goals: share one `for` loop across virtual tensors (#134); inline virtual nodes with non-linear index symbols (#133).

This release is the basis for the workshop paper examples: a clean context-based API (no hosted tensors), shape concatenation, a complete transformer with RoPE, and a written formal account of the core shape/projection inference machinery.

---

## v0.8 — July 13, 2026
**Theme: Parallel schedules and autotuning; AMD HIP backend**

GitHub milestone scope: *"GPU tiling and related optimizations in the polyhedral style, with heuristic syntactic metrics for now. HIP backend (AMD hardware)."*

> **Outcome vs. the original plan:** measured schedule search was pulled in one release early, and the matmul track continued through generated tensor-core instructions (#412). Full beam search over schedule compositions remains v0.9.

**Parallel schedules (done):**
- **Automatic GPU schedules** — CUDA and Metal kernels now parallelize by default (`automatic_gpu_schedule`): hardware axis types render to grid/block/thread loops with launch dimensions, barriers, and shared-memory tiles; per-backend `hardware_limits` validate block sizes, thread counts, and shared-memory use of every kernel.
- **Kernel fission** — routines split into multiple kernels at materialized cross-nest edges, with aligned cross-nest parallelism merging equal-geometry nests losslessly; Metal encodes fissioned steps as fused command-buffer segments.
- **CPU kernel-level parallelism** — the `cc` backend renders parallel loops through a thread pool by default; backends renamed to `cc` / `multidev_cc` (`sync_cc` / `multicore_cc` remain as deprecated aliases).
- **Matmul tiling and tensor cores** (#412, done) — register-tiled `Tile_mma` microkernels, SIMD vector-extension codegen with reduction chains (#468, #469), shared/packed staging, warp shuffles, CUDA WMMA/inline PTX, Metal simdgroup matrices, and HIP rocWMMA.

**Autotuning (done — pulled into v0.8):**
- **Measured schedule search** — `Autotune.tune` searches canonical schedule candidates with execution-based timing: a digest-guarded schedule cache, per-segment schedules for fissioned routines, sketch seeding (e.g. matmul), and placement A/B tuning. Full beam search over schedules remains v0.9.

**AMD HIP backend (done)** (#411) — implemented via the standalone `hipjit` bindings (independent GitHub project and opam package, following the `cudajit`/`metal` pattern), mirroring the CUDA backend's code generation, memory management, and synchronization.

**Benchmarks and platforms (done):**
- **Cross-framework benchmark suite** — `benchmarks/` compares OCANNL against PyTorch (including `torch.compile`) and tinygrad (including BEAM search), gated on loss-parity; checked-in example reports for Metal, CUDA, and Windows/HIP feed the workshop article's benchmark appendix.
- **Windows** — the full test suite is green on the `cc` and `hip` backends; the CUDA backend was restored on Windows (NVRTC arch floors for half/bf16 intrinsics).
- **Megakernel exploration** (#318, done as a study); **Metal private mode** (#320, done).

**Deferred after v0.8:**
- **MSVC on the native-Windows C backend** (#313, closed as not planned) — Windows remains supported through mingw-w64.
- **AVX/AVX2 intrinsics** (#164, done) — the delivered CPU bundle uses pool-backed `Grid` rendering, probed SIMD compiler flags, and portable compiler vector extensions rather than architecture-specific intrinsic calls.
- Stretch / study items not taken up: `ggml` efficiency lessons (#163); restore CUDA `__constant__` arrays (#195); small-Transformer digit-addition reproduction (#427).

---

## v0.9 — August 24, 2026 (ICFP week)
**Theme: Program search and optimization**

A research-heavy milestone. GitHub milestone scope: *"Program search with execution-based per-backend or aggregate-of-backends cost functions; broadening code-graph rewriting rules."*

- **Full beam search over schedules** — extend v0.8's measured autotuner (single-step candidate menus) to a Halide-/tinygrad-BEAM-style beam search over schedule compositions.
- **Cost functions** — per-backend execution-based metrics and aggregate cost functions across backends.
- **Code-graph rewriting** — a broader range of rewriting rules, augmenting the v0.8 tiling/layout mechanisms.
- **Tiramisu/Telamon study** (#267, done) — see [the optimization-space and pruning notes](docs/blog/tiramisu-telamon-optimization-space-pruning.md).
- **Superoptimizers for tensor programs** (#261, done) — see [the research notes](docs/research/superoptimizers.md).
- **Candle study** (#265, closed as not planned).

**Real-world examples and tokenization** (redistributed here from the dissolved v0.7.1):
- **makemore progression** (#59, done) — the character-level language-model series mirroring Karpathy's *Neural Networks: Zero to Hero* (see [docs/makemore_tutorial.md](docs/makemore_tutorial.md)); includes the Bengio-style MLP and BatchNorm variants.
- **CNN classifiers** (#54, done) — MNIST and CIFAR-10 training examples.
- **LSTM example** (#60).
- **Transformer inference demo** (#377, done) — GPT-2 124M inference with pretrained HuggingFace weights (safetensors reader, BPE tokenizer, greedy decoding), exact against a NumPy reference; `test/gpt2/gpt2_generate.ml` is the tutorial executable and `test/gpt2/gpt2_dry_run.ml` the full-scale dry run.
- **Tokenizer bindings** (done) — developed in the spin-off [ocaml-dataprep](https://github.com/ahrefs/ocaml-dataprep) project (opam package `dataprep`); the `Dataprep.Bpe` HuggingFace-compatible BPE tokenizer is bridged to OCANNL tensors via `Nn_blocks.token_ids_of_array` / `token_ids_of_batch`, integration-tested in `test/training/tokenizer_roundtrip.ml`.

> **Date note:** the GitHub milestone carries a stale 2026-05-30 due date; the ICFP-week anchor above is authoritative.

---

## v1.0 — Q4 2026
**Theme: Documentation, completeness, ergonomics, safety**

GitHub milestone scope: *"Few documentation gaps, some degree of feature completeness, ergonomics, safety."* Already largely de-risked — key items below are done.

- **Safety (done):** static verification of merge-buffer nodes "in the right direction" (#288); rank-cycle detection for row variables (#247).
- **Determinism (done):** resolve `multicore_cc` non-determinism and restore it as the primary testing target (#341).
- **`%cd` ergonomics (done):** simplify translations from `%cd` (#348); accept `:=` for the `Fetch` constructor (#209).
- **Open — ergonomics:** concise syntax for merge-buffer transfers; execution dependency tracking (mirroring compilation); local let-bindings in `%cd` (#80).
- **Open — completeness:** demonstrate model surgery (#33); training checkpointing (#96); inference plugin/binary generation (#97); experiment tracking and plot improvements (#122, #103); `polars-ocaml` integration (#219).

---

## v1.1 and beyond
**Theme: Shape-inference and safety enhancements; advanced examples**

GitHub milestone scope: *"Consider introducing axis labels. Consider introducing shape schemes."*

- **Shape schemes for tensor functions** (#404).
- **Axis labels** (vs. the dimension basis) — design exploration.
- **Advanced examples:** BERT / ModernBERT (#297); DisTrO low-communication distributed data parallelism (#278).

---

## Key Milestones Summary

| Version | Target | Status | Key Deliverables |
|---------|--------|--------|------------------|
| 0.6.2  | Nov 2025 | released | Menhir parser, hidden-dimension errors |
| 0.6.3  | Dec 2025 | released | Padding inference, toy CNN |
| ~~0.6.4~~ | — | **skipped** (folds into 0.7) | Concatenation, RoPE, transformer toy |
| **0.7** | Jul 3, 2026 | **released** | **Frontend finalization + compiler optimizations** (consolidates 0.7.2) |
| ~~0.7.1~~ | — | **dissolved** | AMD HIP backend → 0.8; examples + tokenizers → 0.9 |
| **0.8** | Jul 13, 2026 | **released** | **Parallel schedules (GPU + CPU), autotuning, SIMD/`Tile_mma`, AMD HIP backend, benchmark suite** |
| 0.9    | Aug 24, 2026 | planned | Program search: full beam search and cost functions **(ICFP week)**; examples: makemore, MNIST/CIFAR, LSTM, transformer inference, tokenizers |
| 1.0    | Q4 2026 | mostly de-risked | Docs, completeness, ergonomics, safety |
| 1.1+   | post-1.0 | backlog | Shape schemes, axis labels, BERT, DisTrO |

---

## Workshop Paper Artifacts (OCaml Workshop / FProPer at ICFP 2026)

The v0.7 release includes the paper-facing material needed for workshop submission and follow-up discussion:

- Workshop article: [docs/ocannl_workshop_article_human.md](docs/ocannl_workshop_article_human.md).
- Workshop article PDF: [docs/html/pdfs/ocannl_workshop_article_human.pdf](docs/html/pdfs/ocannl_workshop_article_human.pdf).
- Formal core technical report: [docs/ocannl-formal-core-technical-report.md](docs/ocannl-formal-core-technical-report.md).
- Shape constraint generation notes: [docs/shape-constraint-generation.md](docs/shape-constraint-generation.md).

### Proposed Title
*"Generalized Einsum with Row Variables: Shape Inference for Deep Learning in OCaml"*

### Key Contributions
1. **Generalized einsum notation** with convolutions, strided iteration, and concatenation
2. **Row variables** for flexible axis handling ("principle of least commitment")
3. **Constraint-based shape inference** with provenance tracking for error messages
4. **Dimension basis** design rationale (vs. axis labels)
5. **Integration with OCaml's type system** via syntax extensions

### Related Work to Address
- einops (#413)
- torchdim / DumPy (#316)
- Named tensors in PyTorch/JAX
- Dependent types for tensor shapes

### Why v0.7 Before the Paper
The paper needs working examples on OCANNL's mature frontend, all delivered by v0.7:
- Clean context-based API (no hosted tensors)
- Shape concatenation syntax (`^`)
- Complete transformer example with RoPE
- Consistent, documented API surface

The deep semantic groundwork for the paper (the two-sorted ground algebra, the rank-fact graph and rank-cycle check, ≈-semantics for row equality) has been developing alongside v0.7 in the in-progress proposals.
