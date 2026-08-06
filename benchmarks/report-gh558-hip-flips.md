# gh-ocannl-558 on HIP: the flip chain reaches the tensor cores, and materialize-all is still faster

Measurement-only report. No library code changed; the two instruments added to `bench_mlp` are named
in [Instruments](#instruments).

The reduced scope [#558 asks for](https://github.com/ahrefs/ocannl/issues/558#issuecomment-5204252583) is
whether gh-555's landing already subsumes this issue's mechanism: does `tune_inline_flips > 0` find
the weight cast twins' materialization unaided, or are the twins caller-seeded and therefore absent
from the decision surface, leaving a seeding hook to add? On HIP the answer is unambiguous and it is
neither of the two the issue anticipated.

**Verdict: chain-finds-but-doesn't-ship.**

- The **premise holds**: under default placements the reduced-precision cast twins are virtual, the
  matmul sites read f32 masters, and arm A seeds **zero** tensorized candidates — 0 in 10 of 10
  default-placement searches, on both cells.
- The twins **are on the decision surface**, at both cells, as `Materialize` flip candidates. There
  is no seeding hook missing. What #558's comment guessed (caller-seeded rather than
  policy-decided) is not what happens.
- The chain **finds them and it works**: materializing the twins from arm A's context converts the
  tensorized family from unreachable to crowned, and on the GEMM-dominated cell that is worth
  **−37.0%** inside arm A for the flip chain (**−33.7%** for the `decide_materialized` control,
  which materializes the twins and nothing else). The crowned arm-A artifacts are tensorized by the
  `mma_statements` / `mma_scalar_fallbacks` counters; the direct rocWMMA source read is of arm B's
  shipping artifact — see [the emission section](#the-emission-read-off-the-source) for which claim
  rests on which.
- It **still does not ship**, and not because tensorization is uncompetitive. On `mlp_wide`
  materialize-all is *better still*: arm A with the twins materialized is **+11.5%** behind arm B
  (paired, in-process, 3 of 3), and the budget-18 flip chain lands **+5.2%** behind it (3 of 3) for
  **~11x** the search cost. This inverts gh-546's Metal picture, where arm B was the expensive arm.

So #558's mechanism is real, reachable today with no new code, and — on this backend — reaches an
optimum that the plain A/B already reaches more cheaply.

## Provenance

minix-pc: AMD Ryzen AI MAX+ 395 w/ Radeon 8060S iGPU (gfx1151, RDNA3.5), ROCm 7.14.60850, **WSL2**
(kernel 6.18.33.2-microsoft-standard-WSL2) — not bare metal; the CPU and the iGPU share the LPDDR5X
controller, so all cells below ran serially with the box otherwise idle and every process capped at
`taskset -c 0-15`.

Tree: staging master `3e7db701` (contains gh-555's landing — `tune_inline_flips`,
`Context.decide_materialized`, `optimized.flip_candidates` — and gh-492's master-weight/cast-twin
machinery). Backend `hip`, `BENCH_PRECISION=bf16` (rocWMMA's only route on RDNA3.5: its WMMA has no
f32-input shape). One process per run, **a fresh `autotune_cache_dir` for every search**,
`OCANNL_AUTOTUNE_LOG=true`, `~rounds:0` as `bench_mlp` always passes.

Cells: `mlp_small` (2→64→64→2, batch 64, 32 batches) — the cell #558's reduced scope names — and
`mlp_wide` (256→1024→1024→10, batch 256, 16 batches), added because that is where the margin the
scope note quotes actually lives (see [A correction](#a-correction-to-the-scopes-premise)).

Per gfx1151's WMMA not being exactly rounded in any format combination, nothing here asserts
bitwise parity; all cells passed the suite's own bf16 parity gate.

## A correction to the scope's premise

The reduced scope carries the −12.8% bf16 mma margin from the gh-538 HIP leg into `mlp_small`. That
number is [report-hip.md](report-hip.md)'s and it is **`mlp_wide`'s, not `mlp_small`'s** — and it is
a *within-arm-B* comparison (the crowned `F_sketch[mma-gpu 16x32x32 ep]` against the best scalar
candidate the same search timed), not a shipping delta. On `mlp_small` there is no such margin to be
had: the cell is launch-bound at ~1.8 MFLOP/step, and the tensorized candidates lose there even in
the arm where they have always been expressible (below). That is why `mlp_wide` is measured here
too; without it the session would answer "does twin materialization ship the mma win" on a cell that
has no mma win to ship.

## Instruments

`Train.tune_placements` already reports everything needed; `bench_mlp` was not wiring two of it up.
Both additions are benchmark-side and measurement-only:

- **`flip_report` is now passed through**, printed under a `tune flip:` tag (the positional
  arm-A-then-arm-B contract of `report` is preserved — flip arms deliberately do **not** enter the
  result line's `arms` array). The dump also gained `mma_statements` / `mma_scalar_fallbacks`, which
  is what separates "a schedule carrying a `Tensorize`" from "a schedule that emitted one".
- **`BENCH_FLIP_DUMP=1`** prints the whole `flip_candidates` list of the default-placement compile,
  not just the `tune_inline_flips` prefix the chain can afford. Without it, "the twins are not on
  the surface" and "the twins ranked below the budget" are indistinguishable — and they are the two
  outcomes #558's comment asks to be told apart.
- **`BENCH_PRESEED_TWINS=1`** hands the twins to `Context.decide_materialized` before tuning: part
  3's control, and the other route #558's comment names. Note this is *not* the pre-existing
  `BENCH_TWIN_PLACEMENT=materialized`, which declares tnode-level intent and is therefore invisible
  to both the placement A/B and the flip chain. The runner refuses the two flags together: with
  `=virtual` the pre-seed would be a silent no-op (`decide_materialized` skips declared-virtual
  nodes by contract) and with `=materialized` the intent pins the twins in both arms, so the run
  would no longer be the context-level-decision-only experiment. No run in this report set
  `BENCH_TWIN_PLACEMENT`; the twins are left at `Twin_auto` throughout.

## Part 1 — the premise, on both cells

Arm A seeds zero tensorized candidates; arm B seeds and times them. Per-arm fields, one row per
search, fresh cache each:

| cell | arm | best ms (3 runs) | mma seeded → timed | mma_best ms | crowns a `Tensorize` |
|---|---|---|---|---|---|
| `mlp_small` | A (default) | 0.0996 / 0.1053 / 0.1042 | **0 → 0** | none | impossible (none seeded) |
| `mlp_small` | B (materialize-all) | 0.1018 / 0.1135 / 0.1122 | 29 → 17 | 0.1121 / 0.1149 / 0.1122 | 1 of 3 |
| `mlp_wide` | A (default) | 2.2213 / 2.2778 / 2.2045 | **0 → 0** | none | impossible (none seeded) |
| `mlp_wide` | B (materialize-all) | 1.2880 / 1.2602 / 1.2980 | 37 → 21 | 1.2880 / 1.2602 / 1.2980 | **3 of 3** |

The premise holds on HIP exactly as gh-546 found it on Metal, and it is deterministic: 0 → 0 in
every default-placement arm-A search in this report — the six tabulated above plus the four more
inside the `mlp_wide` flip runs.

The two cells then diverge in which arm ships:

- `mlp_small`: **arm A ships, 6 of 6.** Arm B's tensorized candidates are expressible and lose —
  `mma_best` 0.1121–0.1149 against the shipping 0.0996–0.1053. Same conclusion as gh-546's Metal
  leg, reached by a different route.
- `mlp_wide`: **arm B ships in all 11 searches, and its winner is tensorized in all 11** — a −42.6%
  margin over arm A on the three-run means. Materialize-all is not the expensive arm here; it is the
  arm that pays for its extra kernels with tensor cores.

## Part 2.1 — the twins are on the decision surface

`BENCH_FLIP_DUMP=1`, default-placement compile. Both cells expose 66 flip candidates, of which
**six are the cast twins** (`materialize`, bf16 — the three weight twins and the three bias twins;
`mlp_small`'s and `mlp_wide`'s graphs are the same shape):

| cell | twin ranks in `flip_candidates` | their `fc_recompute_cost` | top-ranked candidate's cost |
|---|---|---|---|
| `mlp_small` | **1, 2**, 7, 10, 12, 14 | 128, 128, 64, 64, 64, 64 | 128 (`batch_x`) |
| `mlp_wide` | **7, 8**, 10, 13, 15, 17 | 512, 512, 256, 256, 256, 256 | 2048 (`batch_x`) |

This answers the reduced scope's question 1 and reverses its suspicion. The twins are ordinary
policy-decided virtual nodes, `Tn.Placements.raw_entry` reports them as `Virtual` without a declared
intent, and `low_level.ml`'s candidate filter admits them. **No seeding hook is missing.**

The control confirms the identification exactly: with `BENCH_PRESEED_TWINS=1` the surface drops from
66 candidates to 60 and **the six `materialize`-cast entries are precisely the ones that disappear**
— the pre-seed decided exactly those six nodes and nothing else.

But the ranking is the whole story on the cell that matters. On `mlp_small` the recompute-cost bound
puts two twins at ranks 1–2, so a budget-5 chain reaches them. On `mlp_wide` the four relu/backward
`Inline` candidates outrank every twin (cost 2048/1024 against 512/256), so **a budget-5 chain never
tries a twin at all** — it spends its entire budget on candidates that seed nothing:

```
mlp_wide, tune_inline_flips=5:
  flip materialize batch_x   (cost 2048)  2.1245 ms   0 seeded
  flip inline n32_relu       (cost 2048)  4.4236 ms   0 seeded
  flip inline n34.grad       (cost 2048)  3.1825 ms   0 seeded
  flip inline n40.grad       (cost 2048)  2.2989 ms   0 seeded
  flip inline n32_relu.grad  (cost 1024)  5.4305 ms   0 seeded
  -> did not improve on the A/B winner (2.1245 ms vs 1.3328 ms)
```

The gap #558 would reduce to is therefore **not** a seeding hook. It is that the recompute-cost
bound is not a proxy for "unlocks a candidate family", and on a GEMM-dominated graph it ranks the
twins below a budget anyone would set by default.

## Part 2.2 — a materialized twin does seed and time mma, from arm A's context

It does, at both cells, and one twin is enough. Per-flip report fields (the `tune flip:` tag),
`mlp_small`, budget 5, the three replicates:

| flip tried | r1 | r2 | r3 |
|---|---|---|---|
| `materialize batch_x` | 0.1163, 0 → 0 | 0.1195, 0 → 0 | 0.1070, 0 → 0 |
| `materialize n6_cast` | 0.1099, **3 → 0** | **0.0982**, 3 → 0 | 0.1127, 3 → 0 |
| `materialize n14_cast` | 0.1024, **15 → 10** | 0.1134, **18 → 10** | **0.0982**, **15 → 10** |
| `inline n32_relu` | 0.1147, 0 → 0 | 0.1206, 6 → 0 | 0.1096, 0 → 0 |
| `inline n34.grad` | 0.1111, 0 → 0 | 0.1025, 3 → 0 | 0.1051, 15 → 10 |

and the `decide_materialized` control (part 3), which materializes all six twins at once:

| `mlp_small` arm A | best ms | mma seeded → timed | mma_best ms |
|---|---|---|---|
| twins virtual (default) | 0.0996 / 0.1053 / 0.1042 | 0 → 0 | none |
| twins pre-materialized | 0.1018 / 0.1005 / 0.0947 | **18 → 10** | 0.1156 / 0.1170 / 0.1171 |

Three facts follow, all of them what #558 predicted:

1. **Unreachable becomes expressible at arm A's cost.** Arm A's best is unchanged by materializing
   the twins — 0.1030 ms mean against 0.0990 ms mean, a 4% *improvement* that is inside the ±5%
   run-to-run spread of arm A itself. The three (six) small casts are free.
2. **It is a seeding gate, not a ranking outcome, and it lifts deterministically** — 18 → 10 in 3 of
   3 preseed runs, from 0 → 0 in 3 of 3 defaults.
3. **On `mlp_small` the newly expressible candidates then lose**, by 13–24% (`mma_best`
   0.1156–0.1171 against a crowned 0.0947–0.1018). Exactly gh-546's Metal outcome, with a wider
   margin.

## Part 2.3 — does the refined result ship?

### `mlp_small`: yes sometimes, and it means nothing

Budget 5, three replicates:

| run | A/B winner | chain result | accepted flips | ships? | tensorized? |
|---|---|---|---|---|---|
| r1 | 0.0996 (A) | 0.0996 | none | no | — |
| r2 | 0.1053 (A) | 0.0982 (−6.7%) | `materialize n6_cast` | yes | **no** (`F_split[...]`) |
| r3 | 0.1042 (A) | 0.0982 (−5.8%) | `materialize n14_cast` | yes | **yes** — `F_sketch[mma-gpu 16x32x32]`, `mma_statements=1`, `mma_scalar_fallbacks=0` |

So on this cell the chain does ship a genuinely tensorized artifact — once in three runs. But arm A
alone spans 0.0996–0.1053 across those same three runs (±5%), and the same `n6_cast` flip measured
0.1099 in r1 and 0.0982 in r2. **The −5.8% and −6.7% "wins" are inside the spread of the thing they
are wins over.** The accept-on-improvement rule is latching onto timing noise here, in both
directions: nothing on this cell is decided by the mechanism under test.

### `mlp_wide`: no — and this is the real answer

Budget 18, enough to reach every twin. Three replicates try the same 18 candidates in the same
order — the surface and its ranking are deterministic — and differ only in which flips the timings
accept. One run traced in full (r1):

```
arm A                              2.2096         0 seeded
 flip materialize batch_x          2.1081  ACCEPT   0 seeded
 flip inline n32_relu              4.4086  reject   0 seeded
 flip inline n34.grad              3.1770  reject   0 seeded
 flip inline n40.grad              2.3085  reject   0 seeded
 flip inline n32_relu.grad         5.2647  reject   0 seeded
 flip inline n34                   5.2457  reject   0 seeded
 flip inline n40                   4.3493  reject   0 seeded
 flip materialize n6_cast          2.1810  reject   3 seeded ->  0 timed
 flip materialize n14_cast         1.5223  ACCEPT  11 seeded ->  6 timed   <- first tensorized crowning
 flip inline n6_cast.grad          1.5223  reject
 flip materialize n10_cast         1.5028  ACCEPT  11 seeded ->  6 timed
 flip inline n10_cast.grad         1.5028  reject
 flip inline n14_cast.grad         1.5028  reject
 flip materialize n18_cast         1.5381  reject  11 seeded ->  6 timed
 flip inline n18_cast.grad         1.5028  reject
 flip materialize n22_cast         1.4968  ACCEPT  26 seeded -> 11 timed
 flip inline n22_cast.grad         1.4968  reject
 flip materialize n26_cast         1.4110  ACCEPT  26 seeded -> 11 timed
-> did not improve on the A/B winner (1.4110 ms vs 1.2842 ms)
```

The mechanism works, and works well: the accepted flips take arm A from 2.2096 to 1.4110 ms,
crowning schedules whose `mma_statements` climbs 1 → 2 with `mma_scalar_fallbacks = 0` throughout.
Discovered unaided, no new code, exactly as #558's comment hoped. Across the three replicates:

| | r1 | r2 | r3 | mean |
|---|---|---|---|---|
| arm A | 2.2096 | 2.1433 | 2.1619 | 2.1716 |
| chain result | 1.4110 | **1.3256** | 1.3704 | 1.3690 |
| chain vs arm A | −36.1% | −38.2% | −36.6% | **−37.0%** |
| arm B (the A/B winner) | 1.2842 | 1.3108 | 1.3102 | 1.3017 |
| chain vs arm B | +9.9% | +1.1% | +4.6% | **+5.2%** |
| ships? | no | no | no | **0 of 3** |
| search cost (`compile_s`) | 542.0 s | 550.9 s | 540.8 s | 544.6 s |

It loses every time. Materialize-all is ahead by 1.1–9.9% (mean 5.2%, 3 of 3), and the plain A/B
found it in 35–82 s of search against the chain's ~545 s — **roughly 11x the search cost to finish
behind**. Its best draw (r2, 1.3256 ms, `mma_statements = 2`) essentially ties arm B, which is the
most that can be said for it: at its best the chain reaches materialize-all's optimum, and it never
beats it.

The `decide_materialized` control (part 3) says the same thing without the search, three replicates:

| `mlp_wide` arm A | best ms | mma seeded → timed | crowns a `Tensorize` |
|---|---|---|---|
| twins virtual (default) | 2.2213 / 2.2778 / 2.2045 | 0 → 0 | 0 of 3 |
| twins pre-materialized | **1.4940 / 1.4810 / 1.4705** | **14 → 6** | **3 of 3**, `mma_statements=1`, `mma_scalar_fallbacks=0` |
| arm B, same processes | 1.3443 / 1.3027 / 1.3419 | 37 → 21 | 3 of 3, `mma_statements=2` |

**−33.7% inside arm A** from three weight casts and three bias casts, ratios over the means, and the
per-arm spread is ±1.6% (default) and ±0.8% (pre-seeded) — the effect is ~20x the spread. And
paired in-process against arm B in the same run: **1.111, 1.137, 1.096 — a mean +11.5%, 3 of 3, no
overlap.** Site-targeted materialization reaches the tensorized family; it does not reach
materialize-all's optimum, which on this cell is a *two*-site tensorization (`mma_statements=2`)
that only whole-graph materialization exposes.

Part 3's control therefore **reproduces the flip-chain result**: 1.4705–1.4940 ms pre-seeded against
the chain's 1.3256–1.4110 ms (the chain also materializes `batch_x` and can stop short of all six
twins), both crowning tensorized artifacts with no scalar fallback, both losing to arm B — the
control by 11.5%, the chain by 5.2%. The seeding hook and the search find the same answer; the
search costs ~11x more and pays for it with a slightly better draw.

Whole-cell, the deliverable is a null: both configurations ship arm B, so the replayed step times
are the same artifact. Replay pass, `step_ms` p50 — 1.5079 / 1.5151 / 1.6245 (default) against
1.5112 / 1.5217 / 1.5138 (pre-seeded), a −2.2% mean difference well inside the spread of either.
**Nothing measured in this report changes what `mlp_wide`/hip/bf16 ships.**

## The emission, read off the source

Per the gh-538 contract, not from the label. `output_debug_files_in_build_directory=true` on the
pre-seeded `mlp_wide` run; the shipping artifact's generated HIP:

```
build_files/bench_mlp/cross_entropy_loss_forward_and_gradient_then_sgd_update__seg.hip
  /* tile_mma 16x32x1024 (rocwmma) */
  /* tile_mma fragment update 32x32x32 (rocwmma) */
  rocwmma::mma_sync            2      rocwmma::fragment           6
  rocwmma::load_matrix_sync    6      rocwmma::bfloat16_t        18
  rocwmma::store_matrix_sync   2      rocwmma::accumulator        2
```

Two `tile_mma` sites, matching the artifact's `mma_statements = 2`, with no scalar fallback. This is
real rocWMMA on gfx1151, and it grounds the counter for the arm-A artifacts too.

**A limitation to state plainly.** That source is arm B's, because arm B is what ships on
`mlp_wide`. The arm-A tensorized crownings reported above (pre-seeded arm A, and the flip chain)
are evidenced at the counter level — `mma_statements ≥ 1` with `mma_scalar_fallbacks = 0`, which
gh-545 defined as an emission-level fact, not a label — but their generated source is overwritten by
the later arm's compile before the process exits, so it was not read directly. The claim "arm A
crowns a genuinely tensorized artifact" rests on the counters; the claim "HIP bf16 mma sites emit
rocWMMA on this box" rests on the source above.

## What this means for #558

The reduced scope offered two possible shapes. Neither is what happened:

- Not "the chain finds it unaided, done" — it does find it, but only with a budget large enough to
  outrank four `Inline` candidates that seed nothing, and the result loses to the arm the plain A/B
  already ships.
- Not "the twins are caller-seeded, add the seeding hook" — they are policy-decided and already on
  the decision surface at both cells.

What is left is narrower and different from either. Two things are worth saying to the issue:

1. **The ranking, not the surface, is what keeps the mechanism out of reach at a sane budget.**
   `fc_recompute_cost` is a recompute-cost bound; it has no term for "materializing this node makes
   a candidate family expressible". A cheap fix in that direction would be to let a node whose
   materialization changes a detected matmul site's operand precision triple sort ahead of its
   cost — this is #558's "candidate-level decision" option, reduced to a comparator rather than a
   new arm. Ranking changes were out of scope for this session and none were made.
2. **On HIP the payoff question is already answered, negatively, and for a reason worth recording.**
   #558's item 1 asks whether site-targeted materialization pays on the GEMM-dominated cells. On
   `mlp_wide`/hip/bf16 it does not — not because tensorization is uncompetitive (reaching it inside
   arm A is worth −33.7%) but because whole-graph materialization is *better* at exposing it: two
   tensorized sites against one, which is where arm B's remaining margin comes from. `mlp_small` is the launch-bound cell where the tensorized candidates lose whichever
   way they are reached. So on this backend the thing #558 would build reaches a strictly worse
   optimum than the thing that exists, on both cells measured.

That is a capability confirmed and a payoff refuted, on the backend where bf16 is the only
tensor-core route. Whether it holds on Metal — where arm B is the expensive arm and this
calculus could come out the other way — is #546's cell and not measured here.

## Runs behind this report

17 tuning runs with a fresh cache each — 108 individual `Autotune.tune` searches — plus 9
cache-replay passes, all serial on an otherwise idle box: `mlp_small` budget-5 chains ×3, `mlp_small` pre-seeded A/B ×3, `mlp_wide` A/B ×3, `mlp_wide`
pre-seeded A/B ×3, `mlp_wide` budget-5 chain ×1, `mlp_wide` budget-18 chain ×3, plus the
debug-emission run. Reproduce any of them with:

```bash
cd benchmarks && BENCH_FIXTURE=fixtures/mlp_wide.safetensors BENCH_TUNE=1 BENCH_PRECISION=bf16 BENCH_TUNE_REPORT=1 BENCH_FLIP_DUMP=1 taskset -c 0-15 ../_build/default/benchmarks/runners/ocannl/bench_mlp.exe --ocannl_backend=hip --ocannl_autotune_log=true --ocannl_tune_inline_flips=18 --ocannl_autotune_cache_dir=$(mktemp -d)
```
