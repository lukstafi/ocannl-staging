# OCANNL Roadmap

**v1.0 released August 13, 2026. Next: v1.1, soft target August 24, 2026.**

This roadmap outlines the development plan for OCANNL through version 1.0 and beyond. Dates indicate **end of period** targets. Through v1.0 the schedule was pinned to conference deadlines; it is now project-internal, and the dates below are aspirational rather than external commitments.

> **Schedule note (July 2026):** the roadmap drifted from its original dating because of a slowdown between January and May 2026. v0.7 is the catch-up release. Three structural changes follow from that:
>
> - **v0.6.4 is skipped as a release.** Its scope — axis concatenation/block tensors (#49), RoPE and non-learned position embeddings (#398), the decoder-only transformer toy (#57) — is complete (the GitHub milestone is closed), but it ships inside **v0.7** rather than as a separate tagged release. The last tagged release before v0.7 was **0.6.3**.
> - **v0.7.2 is consolidated into v0.7.** The compiler-optimization and memory-management work that was scheduled separately (loop hoisting, CSE, the universal pool allocator) is part of the single **v0.7** milestone.
> - **v0.7.1 was dissolved.** Its two tracks were redistributed: the **AMD HIP backend (#411)** shipped in **v0.8**; completed examples and tokenizer work landed subsequently, while remaining examples now follow their current GitHub milestone assignments. The GitHub milestone has been deleted.
>
> **Update (August 2026):** the v0.9 milestone closed on schedule, and two rebalances came with that: CUDA/HIP graph capture (#488) moved from v0.9 to v1.0, and the training/deployment utilities plus the `lib/` design study moved out of v1.0, in favor of the compiler-tier and diagnostics work the v0.9 sweep exposed. (Their current homes are v1.1 and v1.2 — see the rebalance note below.)
>
> **Venue history (August 2026):** the OCaml Workshop submission was not accepted — the article was written as a research report rather than as an introductory demonstration, which put it outside that audience's scope. IFL 2026 was then considered as the next target and **decided against as a poor fit**. No conference submission is currently scheduled; the paper-facing artifacts below stay in the repository and the formal core technical report continues as live work. The workshop article and its PDF are kept unchanged, as a historical artifact capturing the state of the project at v0.8.
>
> **v1.0 shipped August 13, 2026**, with its milestone fully closed (49 issues). Three consequences for what follows: the release dates are no longer pinned to paper deadlines, **v1.1's soft target moves to August 24, 2026** (the OCaml Workshop date, used as an anchor rather than as a submission), and v1.1/v1.2 were rebalanced along a different seam than the original split — **v1.1 is the compiler work plus the training-loop mechanics it needs**, and **v1.2 is the consumers and explorations**: models, reproductions, demos, integrations, the training experience a user sees, and performance items gated on hardware or on evidence not yet in hand.
>
> **Update (late August 2026):** v1.2 was split along the performance seam. **v1.2 is now performance-chasing in the `approximate` profile, demonstrated on benchmarks** — the numerics-changing tier (fused attention, Winograd, tf32/fp16 arithmetic) behind a third preset, the exact-numerics performance residue, and the benchmark legs that expose where OCANNL wins and loses; the Winograd and zero-nest conv tiers (#505, #503) moved into it from v1.1. Everything else that was in v1.2 — the consumers, explorations, training experience, engineering hygiene, and hardware-gated items — is **v1.3**.
>
> The version sequence is: `0.7 → 0.8 → 0.9 → 1.0 → 1.1 → 1.2 → 1.3`. Milestone *scope* below tracks the GitHub milestones, which are the source of truth.

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

## v1.0 — August 13, 2026 (released)
**Theme: Advanced compiler tiers and schedule-quality follow-through**

Closed with all 49 assigned issues resolved. GitHub milestone scope: *"Branch-and-bound on the analytic cost model. Better performance: better tensor cores, algebraic rewrites (non-numeric-preserving), better beam search."* The completeness, ergonomics and safety goals originally under v1.0 moved to v1.1 and v1.2: v1.0 marks the compilation side reaching the shape argued for in [the compilation manifesto](docs/compilation_manifesto.md).

**Advanced compiler tiers (done):**
- **Branch-and-bound schedule inference** (#514) — `Ir.Schedule_space` as a refinement tree over partial schedules, legality verdicts with witnesses deciding subtrees before any member is built, `Cost_model.completion_floor` as an admissible downward bound, the placement-space search with its enablement prior, and the staged tile lattice as corner-judged interval boxes. Delivered in phases 0–6, with a three-machine evaluation as its research output ([report](benchmarks/report-gh514-eval.md)).
- **Inlining as a first-class, searchable schedule decision** (#555), on top of retiring the concrete-index tracer in favor of the affine access relations (#554); the analysis is shared across sibling candidate compiles (#560).
- CUDA tensor-core profile completeness (#481), software-pipelined double-buffered staging (#487), CUDA/HIP graph capture of the fissioned step (#488), and budget-driven rematerialization on top of the liveness planner (#498).
- CPU reduced precision: 16-bit storage with f32 compute (#517) and native fp16 arithmetic (#516).

**Schedule quality — the `gpt2_mini` arc (done):**
- The v0.9 sweep left `gpt2_mini` 72x off torch CUDA and unmoved by tuning or materialization (#531). Attributing the step (#531, [report](benchmarks/report-gh531-profile.md)) put 70.2% of it in five kernels declined by one companion-coverage rule; judging that rule at the site's arity (#569) took the tuned step 107.4 → 52.4 ms on CUDA and 45.6 → 25.4 ms on HIP. Batched/rank-3 matmul sites are seeded (#528), so tensor cores are reachable on transformer workloads at all.
- CUDA bf16 mma timed scalar-fallback code under an mma label, because capability was keyed on the multiplicand pair rather than the accumulator format (#545); the tuner reports the untuned default's measured time as its honest reference point (#552); the search report names the winning candidate, and Metal's placement A/B was measured and found sound (#546, [report](benchmarks/report-gh546-metal.md)).
- Conv-sketch tuning wins did not port across CPUs; the cause was pool heterogeneity rather than the seeds, and `cc` now restricts its worker pool to one core class on hybrid machines (#530, [proposal](docs/proposals/gh-ocannl-530-pool-uniformity.md)). Device-memory accumulation during tuning and the placement-arm containment around it (#550), `bench_gpt` gate-cost legs (#551).
- Search-plumbing consolidation: intra-statement order in affine access paths (#561), a shared canonical-llc emission core (#563), pre-dispatch validation as its own contained phase (#564), interval narrowing from `If` conditions in `simplify_llc` (#566), barrier elision for same-anchor shared stages (#567), and the numerics policy entering the schedule cache key (#568).

**Frontend, configuration and diagnostics (done):**
- Shape inference: "close down when known" narrowed to the leaf-tensor rule it always was, with `stretch` requesting use-site resolution by name (#544).
- Config profiles `reproducible` / `performance` with picker-inherited precedence (#559); config startup chatter moved off stdout (#581).
- Routine name-clash policy established — status quo adequate, the correctness hazard being structurally solved and the residue debugging-quality only (#513).
- Site-targeted materialization: the capability exists twice over and ships nowhere measured (#558).

**Infrastructure:**
- CI moved to OCaml 5.5, Windows off the per-PR path onto a twice-weekly schedule, and the GPU backends onto a daily cross-machine sweep (`tools/sweep.sh`) — CI's runners have no GPU, so Metal, CUDA and HIP had never been covered there at all.

**Not taken up in v1.0, and where it went:**
- Fused attention via online softmax (#483) and the remaining convolution tiers — zero-nest workgroup geometry (#503) and Winograd (#505) — move to v1.1. The `gpt2_mini` attribution retargeted #483 at 5.4% of the step at seq 128 (a plurality at native context), which is the trigger it now waits on.
- Roadmap-only ergonomics — concise merge-buffer transfer composition, execution-dependency tracking — remain unscheduled proposals.

Work that landed in this milestone before the v0.9 cutoff and shipped inside v0.9: safety, determinism and `%cd` simplification (#288, #247, #341, #348, #209), the tracing design (#160), `%op` inline-initializer scoping (#511), the mixed-precision cost diagnosis (#535), and the f16/bf16 defects — the reduction-identity cutoff (#547), the causal mask's sentinel (#548), and the GPU bfloat16 math builtins (#549).

---

## v1.1 — August 24, 2026 (soft target)
**Theme: Performance carry-overs, algebraic rewrites, and training-loop utilities**

GitHub milestone scope: *"Performance carry-overs, algebraic rewrites, training utilities."* This is the compiler milestone: the tiers v1.0 did not reach, the follow-ups v1.0's own evaluation filed, and the training-loop mechanics that belong next to them. The user-facing training *experience* — checkpointing, tracking, plots — moved to v1.2, so that this milestone stays one kind of work. The date is an anchor (the OCaml Workshop date) used for pacing, not a commitment: scope decides when this closes, and the date is re-set when it arrives rather than treated as missed.

**Training-loop utilities:**
- LR schedules, global-norm clipping and gradient accumulation (#465), and mmap-backed checkpoint loading with aligned payloads and zero-copy hosted arrays (#467) — the mechanics a training loop runs on, as against the experience built on top of them.

**Algebraic and convolution tiers** (carried over from v1.0, then moved on to v1.2 in the late-August split): fused attention via online softmax (#483), zero-nest workgroup geometry for whole-routine zeroed GPU sites (#503), and the Winograd conv rewrite (#505) — they are numerics-changing or benchmark-demonstrated work, which is v1.2's definition.

**Search follow-ups from the v1.0 evaluation** — the deliberate consequence of v1.0 recording its nulls rather than shipping them as wins, each filed with its evidence attached:
- Lift statically-decidable builder preconditions into tree verdicts, so the tile lattice stops pricing spaces whose members die at candidate build (#577); make the envelope's memory leg fittable (#578); give enablement promotion a profitability term (#579); extract the sketch-family trees from `autotune.ml` before either grows them (#580).
- `Tile_mma` register tiling for narrow 16-bit operands (#575).
- The `gpt2_mini` residue: fission the `lm_head` segment apart from its `max_logits` reduction (#574), and the Virtual residual stream's quadratic-in-depth re-summation (#573).

**Robustness and test seams:**
- A digest-completeness consistency test, so every codegen-affecting knob is either in cache identity or explicitly exempt (#572); a fault-injection inventory for resource-owning seams (#571); a seam for compiling and linking a hand-built `Low_level.optimized`, giving analysis-level passes executed parity coverage (#562); operand-evaluation conditionality encoded once instead of three times (#582).

Quantization (#137, #271), the WebGPU/WASM target (#123), the LLVM backend (#200), the fork-based backend (#161), `strict` axis naming (#190), local let-bindings in `%cd` (#80), the DumPy/torchdim deep dive (#316), the Imbue training-in-the-large study (#270), and the Fleuret lecture examples (#216) were completed or dispositioned in this milestone.

---

## v1.2 — undated
**Theme: Performance-chasing in the approximate profile, demonstrated on benchmarks**

GitHub milestone scope: *"Performance-chasing in the approximate profile, demonstrated on benchmarks."* The `performance` profile is defined as the fastest configuration *at unchanged semantics*; this milestone chases performance past that line, under a third preset whose results differ from the exact profiles by a tolerance the benchmark parity envelope names. An issue here closes with a before/after benchmark cell in a report. Undated: paced by what the measurements say.

**The regime and its evidence:**
- The `approximate` profile (#719): a third built-in preset bundling tf32 matmuls, fp16 arithmetic, fast-math, and every algebraic-rewrite gate as it lands, with a benchmark regime column whose torch counterpart runs torch's own defaults (tf32, SDPA, cudnn autotuning) rather than the pinned-exact settings the parity oracle needs.
- The benchmark expansion (#720): sequence-length and batch-size scaling curves, a 3×3 padded conv workload, roofline-attainment and device-memory columns, thread-count parity on the CPU column — each leg the demonstration for named issues below. Gemma 3 (270M/1B) as the real-weights long-context target (#570).

**Algebraic rewrites — the numerics-changing tier:**
- Fused attention via online softmax (#483), on the minimal loop-carried-recurrence construct the IR needs for it (#696) — the `seq²` materialization is what the scaling curve exposes.
- Winograd F(2×2, 3×3) (#505), and the zero-nest workgroup geometry that lets whole-routine GPU conv candidates be proposed (#503).
- fp16 accumulator width aligned with `fp16_arithmetic` (#680); warp-shuffle reductions at the named accumulator residency for narrow storage (#682).

**Exact-numerics performance residue:**
- Footprint-scoped materialization — a middle ground between inlining and a full buffer (#616); register-tile geometry as a schedule decision the tuner can time (#619), the column-remainder peel cliff (#620), and packed GEBP schedules at non-dividing extents so the cost model can be measured where it is questioned (#627); vector `Max`/`Min` reductions off the per-lane loop (#649).
- Cost-model fidelity: the advisory envelope's two consumers with opposite biases (#636) and hoisted scope bodies in `sc_flops` (#637).
- Async-copy staging refinements — Metal `simdgroup_async_copy`, HIP direct-to-LDS, pipeline depths > 2 (#576); device memory management under pressure (#565), whose first consumer the long-context legs are expected to be.

---

## v1.3 — undated
**Theme: Consumers, explorations, and the hygiene the reviews filed**

GitHub milestone scope: *"Consumers and explorations: models, reproductions, demos, integrations, and the training experience; plus the engineering-hygiene and test-seam follow-ups the v1.0–v1.1 review cycles filed, and performance items gated on hardware not in the fleet."* Everything that consumes the compiler rather than building it, and the follow-ups that make the repository easier to work in. Paced by interest and by what the hardware allows, not by a date.

**Training experience:**
- Resumable checkpoints (#96), experiment tracking — graphs of observables such as loss and device health (#122), plot legends and axis ticks (#103), and backend zero-copy from mmap-backed checkpoints (#585).

**Models, reproductions and demos:**
- Model surgery (#33), LSTM (#60), Bonsai RNN (#182), digit addition (#427), BERT/ModernBERT (#297), DisTrO (#278).

**Frontend design, library and deployment:**
- Shape schemes for tensor functions (#404), the Simply/NanoDO study for `lib/` (#435), PoPE (#444), inference plugins/binaries (#97), and the ppxlib ceiling migration (#695).

**Integrations and external-framework study:**
- Polars integration (#219) and a krnl/autograph study (#277).

**Hardware-gated performance items:**
- HIP CDNA tensor cores via MFMA (#477) — no CDNA box in the fleet; CUDA pinned host buffers (#170) and CUDA constant memory (#195).

**Engineering hygiene and test seams** (filed by the v1.0–v1.1 review cycles, each grounded in a PR's review history):
- Configuration and caching: per-device schedule-cache identity (#594), the digest-completeness test's classification-vs-tag blindness (#596), the bootstrap keys' precedence walk written three times (#604), placement provenance without a decoder (#609), one-element constant literals with no host-backed form (#641).
- IR and codegen: algebraically dead reads at store time (#625), hand-rolled `Low_level` walkers re-deriving conventions (#630), the cc/`builtins.c` duplication (#656), the fp8 codecs' lost exhaustive verification (#657), the HIP fp8 ROCm bug record (#647), cross-target compile checks for emitted kernels (#650), cc rendering for workgroup-shared staging so GPU-sketch parity legs run off-GPU (#678).
- Test infrastructure: `config_dep_completeness`'s untested resolution half (#603), `test-run.sh`'s launch protocol (#606), `last` pointer (#607) and lock entry point (#671), the remaining hand-built IR traversals (#608), computed-label boolean claims (#624), operands that stop discriminating at dividing sizes (#640), goldens pinning node ids (#642) and `~here` line numbers (#672), the schedule-composition property test the reduction forms need (#664).
- Documentation: AGENTS.md/CLAUDE.md drift (#653), the dune 3.20 recipe fallback (#654), uncompiled doc examples (#660).

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
| **1.0** | Aug 13, 2026 | **released** | **Branch-and-bound schedule inference, inlining as a searchable decision, graph capture, software pipelining, rematerialization, CPU reduced precision, and the 2x `gpt2_mini` step** |
| 1.1    | Aug 24, 2026 (soft) | planned | Performance carry-overs, algebraic rewrites, training-loop utilities: fused attention and the remaining conv tiers, the v1.0 search follow-ups, LR schedules and mmap checkpoint loading |
| 1.2    | undated | planned | Performance-chasing in the approximate profile, demonstrated on benchmarks: the `approximate` preset, fused attention and Winograd, the exact-numerics residue, and the benchmark legs that expose wins and losses |
| 1.3    | undated | planned | Consumers and explorations: models, reproductions, demos, integrations, the training experience, review-filed hygiene, and hardware-gated performance items |

---

## Paper Artifacts (no venue currently targeted)

**No conference submission is scheduled.** The OCaml Workshop / FProPer submission was not accepted, and IFL 2026 was considered and decided against as a poor fit. The written material below is therefore maintained for its own sake — as the project's technical account, and as the starting point should a venue be chosen later. Only the formal core technical report and the constraint-generation notes are live work; the rest is archival.

The paper-facing material, first assembled for the v0.7/v0.8 workshop submission:

- Workshop article: [docs/ocannl_workshop_article_human.md](docs/ocannl_workshop_article_human.md) — **historical artifact**, kept as-is; it describes the project as of the [0.8 release](https://github.com/ahrefs/ocannl/releases/tag/0.8) and was written for the OCaml Workshop / FProPer submission, which was not accepted.
- Workshop article PDF: [docs/html/pdfs/ocannl_workshop_article_human.pdf](docs/html/pdfs/ocannl_workshop_article_human.pdf) — likewise archival, rendered from the v0.8-era source.
- Formal core technical report: [rendered PDF](https://ahrefs.github.io/ocannl/docs/pdfs/ocannl-formal-core-technical-report.pdf), LaTeX source in [docs/](docs/ocannl-formal-core-technical-report.latex) — live, still developing.
- Shape constraint generation notes: [docs/shape-constraint-generation.md](docs/shape-constraint-generation.md) — live.

The intended shape of a paper, should one be written, is recorded below.

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

### Why v0.7 Was the Prerequisite
Any such paper needs working examples on OCANNL's mature frontend, all delivered by v0.7:
- Clean context-based API (no hosted tensors)
- Shape concatenation syntax (`^`)
- Complete transformer example with RoPE
- Consistent, documented API surface

The deep semantic groundwork (the two-sorted ground algebra, the rank-fact graph and rank-cycle check, ≈-semantics for row equality) now lives in the formal core technical report and its appendix.
