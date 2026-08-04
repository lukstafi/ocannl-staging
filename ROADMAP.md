# OCANNL Roadmap to v1.0

**Headline target: v1.0 at the IFL 2026 draft-paper deadline (September 4, 2026)**

This roadmap outlines the development plan for OCANNL from the current state to version 1.0 and beyond, pinned to the deadlines of the paper track — now [IFL 2026](https://ifl26.cse.chalmers.se/), the 38th Symposium on Implementation and Application of Functional Languages (Gothenburg, October 28–30, 2026; draft papers due September 4, 2026). Dates indicate **end of period** targets.

> **Schedule note (July 2026):** the roadmap drifted from its original dating because of a slowdown between January and May 2026. v0.7 is the catch-up release. Three structural changes follow from that:
>
> - **v0.6.4 is skipped as a release.** Its scope — axis concatenation/block tensors (#49), RoPE and non-learned position embeddings (#398), the decoder-only transformer toy (#57) — is complete (the GitHub milestone is closed), but it ships inside **v0.7** rather than as a separate tagged release. The last tagged release before v0.7 was **0.6.3**.
> - **v0.7.2 is consolidated into v0.7.** The compiler-optimization and memory-management work that was scheduled separately (loop hoisting, CSE, the universal pool allocator) is part of the single **v0.7** milestone.
> - **v0.7.1 was dissolved.** Its two tracks were redistributed: the **AMD HIP backend (#411)** shipped in **v0.8**; completed examples and tokenizer work landed subsequently, while remaining examples now follow their current GitHub milestone assignments. The GitHub milestone has been deleted.
>
> **Update (August 2026):** the v0.9 milestone closed on schedule, so **v1.0 becomes the next paper-deadline release** and v1.1 follows it. Two rebalances came with that: CUDA/HIP graph capture (#488) moved from v0.9 to v1.0, and the training/deployment utilities (#96, #97, #122, #465, #467) plus the `lib/` design study (#435) moved from v1.0 to v1.1, in favor of the compiler-tier and diagnostics work the v0.9 sweep exposed.
>
> **Venue change (August 2026):** the OCaml Workshop submission was not accepted — the article was written as a research report rather than as an introductory demonstration, which put it outside that audience's scope. The paper track now targets **IFL 2026** (Gothenburg, October 28–30, 2026), whose draft-paper deadline is **September 4, 2026** (post-symposium papers for the formal proceedings are due November 25, 2026). The release schedule follows the new venue: **v1.0 lands on the September 4 draft-paper deadline** and **v1.1 on October 28, the symposium's opening day**. The existing workshop article and its PDF stay in the repository unchanged, as a historical artifact capturing the state of the project at v0.8.
>
> The version sequence is: `0.7 → 0.8 → 0.9 → 1.0 → 1.1`. Milestone *scope* below tracks the GitHub milestones, which are the source of truth.

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
- **Formal core technical report:** `docs/ocannl-formal-core-technical-report.latex` (the authoritative source) and its rendered PDF, covering the core shape/projection inference proof effort.
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

> **Outcome vs. the original plan:** measured schedule search was pulled in one release early, and the matmul track continued through generated tensor-core instructions (#412). Follow-up schedule-quality work is assigned according to the current GitHub milestones below.

**Parallel schedules (done):**
- **Automatic GPU schedules** — CUDA and Metal kernels now parallelize by default (`automatic_gpu_schedule`): hardware axis types render to grid/block/thread loops with launch dimensions, barriers, and shared-memory tiles; per-backend `hardware_limits` validate block sizes, thread counts, and shared-memory use of every kernel.
- **Kernel fission** — routines split into multiple kernels at materialized cross-nest edges, with aligned cross-nest parallelism merging equal-geometry nests losslessly; Metal encodes fissioned steps as fused command-buffer segments.
- **CPU kernel-level parallelism** — the `cc` backend renders parallel loops through a thread pool by default; backends renamed to `cc` / `multidev_cc` (`sync_cc` / `multicore_cc` remain as deprecated aliases).
- **Matmul tiling and tensor cores** (#412) — register-tiled `Tile_mma` microkernels, SIMD vector-extension codegen with reduction chains (#468, #469), shared/packed staging, warp shuffles, CUDA WMMA/inline PTX, Metal simdgroup matrices, and HIP rocWMMA. The first tranche shipped here; the issue itself continued into v0.9 and was closed there.

**Autotuning (done — pulled into v0.8):**
- **Measured schedule search** — `Autotune.tune` searches canonical schedule candidates with execution-based timing: a digest-guarded schedule cache, per-segment schedules for fissioned routines, sketch seeding (e.g. matmul), and placement A/B tuning.

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

## v0.9 — August 3, 2026
**Theme: Schedule quality, deterministic parallelism, and convolution performance**

A research-heavy milestone, closed with all 44 assigned issues resolved. GitHub milestone scope: *"Program search with execution-based per-backend or aggregate-of-backends cost functions; broadening code-graph rewriting rules."*

**Search quality and schedule legality (done):**
- **Constraint-based schedule legality** (#494) — `Ir.Affine` and `Low_level.affine_accesses` expose loop boxes and access relations; conflict, coverage, fiber-cardinality and read-before-write queries drive shared-memory safety, fission, scratch validation, and a `Schedule.op_legality` oracle that prunes proven-illegal proposals.
- **Analytic cost model** (#491) — footprint/FLOP extraction, roofline lower bounds, per-backend envelopes, model-picked untuned defaults (`model_default_schedule`) and a keep-fraction pre-filter over sketch seeds, with calibration logging. Advisory throughout: candidates without model coverage are never dropped.
- **Cross-machine benchmark and tuning sweep** (#476), re-measured under #538 after the search changes — full Metal, CUDA and HIP columns from a wiped autotune cache, plus paired A/B legs. The checked-in reports under `benchmarks/` are the record.
- **Pad-to-tile scheduling** (#485), **static partitioning** (#508), and the **tf32 numerics policy** (#478); CUDA 13 / cudajit fixes (#482).

**Parallel execution and numerics (done):**
- **Deterministic split reductions** (#484) — two-pass tree combines with autotune seeding, extended by `Swap`-hoist composition (#537) so conv-gradient accumulations became reachable.
- **Mixed-precision training recipe** (#492) — precision-assignment policy, master weights with cast twins, dynamic loss scaling with a fused on-device gate, and forward-only reduced precision by load-time conversion.
- **Packed-uniform retirement** (#509) — the packed `uniform` became total over shapes, took over default parameter initialization, and gained lane-extract virtualization.
- Correct overlapping-window gradients for tropical/einmax1 reductions (#512), with the non-overlapping fast path restored (#527).

**Convolution performance (done):**
- Implicit-GEMM sketch families (#493), blocked tile flavors (#500), epilogue twins (#501), compact strided-row staging (#502), and clamped-window lowering for padded max-family pooling (#504).

**Search survivability (done — scope that emerged from the sweep):**
- **Typed candidate-failure containment** (#536) with the Metal, CUDA and HIP arms; **HIP scratch pre-validation** (#533); unparallelized GPU candidates refused rather than dispatched (#532); the decline census made complete (#541, #543); advisory-fallback and model-ranking fixes (#519, #522); GPU mma candidates made reachable (#521); benchmark matrix and parity-gate corrections (#523, #529, #538, #539); HIP tensor-core and memory-accounting fixes (#540, #542).
- Frontend and backend defects surfaced along the way: unary einsum specs with convolution indices (#515), CUDA's clang-only `0.0h` half literal (#518), and Metal bf16 uniform builtins writing raw bit patterns (#520).

**Examples, research, and dispositions:**
- CNN classifiers (#54), GPT-2 inference (#377), and the matmul-to-tensor-cores track (#412) closed here.
- Research: TVM (#242), [Tiramisu/Telamon](docs/blog/tiramisu-telamon-optimization-space-pruning.md) (#267), [superoptimizers](docs/research/superoptimizers.md) (#261), and Lean Attention (#263).
- Candle (#265), Petalisp/Caten (#306), and MSVC (#313) were evaluated and closed as not planned.

**Deferred out of v0.9:**
- CUDA/HIP graph capture (#488) moved to v1.0.

---

## v1.0 — September 4, 2026 (IFL draft-paper deadline)
**Theme: Advanced compiler tiers and schedule-quality follow-through**

GitHub milestone scope: *"Branch-and-bound on the analytic cost model. Better performance: better tensor cores, algebraic rewrites (non-numeric-preserving), better beam search."* The completeness, ergonomics and safety goals originally under v1.0 moved to v1.1: v1.0 marks the compilation side reaching the shape argued for in [the compilation manifesto](docs/compilation_manifesto.md).

**Advanced compiler tiers:**
- CUDA tensor-core profile completeness (#481), fused attention via online softmax (#483), software-pipelined double-buffered staging (#487), CUDA/HIP graph capture of the fissioned step (#488), and rematerialization on top of the liveness planner (#498).
- Remaining convolution tiers: zero-nest workgroup geometry (#503) and Winograd (#505).
- Branch-and-bound schedule inference (#514).

**Schedule-quality follow-through (from the v0.9 sweep):**
- Rank-3 (batched/attention) matmul sites are never seeded, so `gpt2_mini` cannot reach tensor cores (#528); CUDA bf16 mma times scalar-fallback code under an mma label (#545). Metal's placement A/B was measured and found sound (#546, [report](benchmarks/report-gh546-metal.md)): the arms are separated far outside the noise, and what the discarded arm actually holds is the *only* tensorized candidate that exists at reduced precision, since virtual cast twins make the matmul site's precision triple unadvertisable. The open follow-up is a placement targeted at the site rather than a precision-aware arm choice (#558).
- The tuner's baseline is the unscheduled serial form rather than the default schedule — the shared cause behind #532 and #533 (#552).
- Conv-sketch tuning wins do not port across CPUs (#530); `gpt2_mini` inference is far off torch CUDA and moves under neither tuning nor materialization (#531); intermittent OOM during CUDA tuning (#550); `bench_gpt` gate-cost legs (#551).
- CPU reduced precision: native fp16 arithmetic (#516) and 16-bit storage with f32 compute (#517).
- Retire the concrete-index tracer in favor of affine access relations (#554).

**Frontend and diagnostics:**
- Establish a routine-name collision policy (#513).
- Shape inference: "close down when known" is a leaf-tensor rule applied too widely (#544).

**Roadmap-only ergonomics:**
- Concise merge-buffer transfer composition.
- Execution-dependency tracking analogous to initialization dependencies.

Work already completed in this milestone: safety, determinism and `%cd` simplification (#288, #247, #341, #348, #209), the tracing design (#160), `%op` inline-initializer scoping (#511), the mixed-precision cost diagnosis (#535), and the f16/bf16 defects that shipped inside v0.9 — the reduction-identity cutoff (#547), the causal mask's sentinel (#548), and the GPU bfloat16 math builtins (#549).

---

## v1.1 — October 28, 2026 (IFL symposium week)
**Theme: Training/deployment utilities, shape design, model examples, and integrations**

GitHub milestone scope: *"Completeness, more examples, potentially cloud-tested (datacenter) GPU targets for Nvidia and AMD."* This milestone inherits the documentation, feature-completeness, ergonomics and safety goals originally scoped for v1.0, alongside the axis-label and shape-scheme design directions.

**Training and deployment** (moved here from v1.0):
- Resumable checkpoints (#96), inference binaries/plugins (#97), experiment tracking (#122), training-loop utilities such as LR schedules and gradient accumulation (#465), and mmap-backed checkpoint loading (#467).

**Shape and frontend design:**
- Shape schemes for tensor functions (#404) and the axis-label design direction.
- Local let-bindings in `%cd` (#80) and plot legends/ticks (#103).

**User-facing library:**
- Apply lessons from Simply/NanoDO to `lib/` (#435).

**Model examples and research:**
- Model surgery (#33), LSTM (#60), Bonsai RNN (#182), digit addition (#427), BERT/ModernBERT (#297), and DisTrO (#278).

**Integrations and external-framework study:**
- Polars integration (#219) and a krnl/autograph study (#277).

**Deferred backend experiments:**
- CUDA pinned host buffers (#170), CUDA constant memory (#195), PoPE (#444), and HIP CDNA tensor cores via MFMA (#477).

Quantization (#137, #271), the WebGPU/WASM target (#123), the LLVM backend (#200), the fork-based backend (#161), `strict` axis naming (#190), the DumPy/torchdim deep dive (#316), and the Fleuret lecture examples (#216) were completed or dispositioned in this milestone.

---

## Key Milestones Summary

| Version | Target | Status | Key Deliverables |
|---------|--------|--------|------------------|
| 0.6.2  | Nov 2025 | released | Menhir parser, hidden-dimension errors |
| 0.6.3  | Dec 2025 | released | Padding inference, toy CNN |
| ~~0.6.4~~ | — | **skipped** (folds into 0.7) | Concatenation, RoPE, transformer toy |
| **0.7** | Jul 3, 2026 | **released** | **Frontend finalization + compiler optimizations** (consolidates 0.7.2) |
| ~~0.7.1~~ | — | **dissolved** | AMD HIP backend → 0.8; completed examples and tokenizers landed subsequently |
| **0.8** | Jul 13, 2026 | **released** | **Parallel schedules (GPU + CPU), autotuning, SIMD/`Tile_mma`, AMD HIP backend, benchmark suite** |
| **0.9** | Aug 3, 2026 | **released** | **Schedule quality, deterministic parallelism, mixed precision, convolution performance, and search survivability** |
| 1.0    | Sep 4, 2026 | planned | Advanced compiler tiers and schedule-quality follow-through **(IFL draft-paper deadline)** |
| 1.1    | Oct 28, 2026 | planned | Release completeness: training/deployment utilities, shape design, model examples, and integrations **(IFL symposium week)** |

---

## Paper Artifacts (IFL 2026; formerly OCaml Workshop / FProPer at ICFP 2026)

The paper track now targets [IFL 2026](https://ifl26.cse.chalmers.se/): draft papers are due **September 4, 2026** (the v1.0 date), the symposium runs **October 28–30, 2026** in Gothenburg (v1.1 opens on its first day), and post-symposium papers for the formal proceedings are due **November 25, 2026**.

The paper-facing material, first assembled for the v0.7/v0.8 workshop submission, carries over:

- Workshop article: [docs/ocannl_workshop_article_human.md](docs/ocannl_workshop_article_human.md) — **historical artifact**, kept as-is; it describes the project as of the [0.8 release](https://github.com/ahrefs/ocannl/releases/tag/0.8) and was written for the OCaml Workshop / FProPer submission, which was not accepted. It is the starting point for the IFL draft, not the draft itself.
- Workshop article PDF: [docs/html/pdfs/ocannl_workshop_article_human.pdf](docs/html/pdfs/ocannl_workshop_article_human.pdf) — likewise archival, rendered from the v0.8-era source.
- Formal core technical report: [rendered PDF](https://ahrefs.github.io/ocannl/docs/pdfs/ocannl-formal-core-technical-report.pdf), LaTeX source in [docs/](docs/ocannl-formal-core-technical-report.latex) — live, still developing.
- Shape constraint generation notes: [docs/shape-constraint-generation.md](docs/shape-constraint-generation.md) — live.

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
