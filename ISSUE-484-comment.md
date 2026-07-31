## HIP leg: task 3 is inert here too, and here is *why* the conv gradients are rejected

Measured on gfx1151 (minix, WSL2), tree `d71efc99` — i.e. **post gh-527 and post gh-521**, so the
two known confounders are out of the way. HIP is the most sensitive detector for this: the gh-476
sweep put the target segment at 97.5% of the lenet step there.

This confirms the CUDA leg's headline independently on a second backend (the seeding is inert on
the conv benchmarks; the detector proposes only the classifier head), and **answers that leg's #1
open question** — *"Find out why the conv-gradient accumulations are rejected by
`split_reduce_sites`… they pass the extent floor, so the rejection is in `Sched.op_legality` or in
the dedup-by-axis-symbol. Pinning down which is the next step."* It is `op_legality`, and the exact
rule is below.

### First, the confounder: gh-527 was most of it, but not all

The gh-476 report deferred this decomposition until gh-527 was fixed. Re-running its own instrument
(`BENCH_SEG_TIMES=1`, min of 20):

| workload | seg22 (`bias_conv1.grad`/`n65.grad`) | before | after | share before → after |
|---|---|---|---|---|
| lenet/hip | | 350.712 ms | **72.76 ms** (4.8×) | 97.5% → **89.3%** |
| cifar_stride/hip | | 201.416 ms | **50.03 ms** (4.0×) | 93.3% → **78.7%** |
| lenet step total | | 359.743 ms | **81.49 ms** (4.4×) | |
| cifar_stride step total | | 215.778 ms | **63.56 ms** (3.4×) | |

lenet seg22 over three consecutive runs: 73.132 / 72.773 / 72.756 ms (<0.5% spread). Seg-times sums
agree with independently measured step times to within 0.25%. Materialized cells barely moved
(lenet 7.165 → 6.603, cifar_stride 21.305 → 21.064), the expected control since gh-527's regression
lived in the recompute path only.

**So of lenet's original 350.7 ms, ~278 ms was gh-527 and ~73 ms is the inherent serial reduction —
this issue's actual subject.** It is still the single dominant cost of the default-placement step
at 89%; every other segment is ≤ 3.6 ms. The "highest-value target in the GPU pipeline" framing
survives, at a quarter of the previously quoted magnitude.

### Task 3 has no effect on it, for a structural reason

Task 3 landed before this measurement, so the 73 ms residue is measured *with* the split-reduce
seeding active. `BENCH_SR_SITES=1` — extended on this branch to print, beyond the detected sites,
the `op_legality` verdict for splitting each enclosing serial loop:

```
split-reduce sites detected: 2
  cross_entropy: reduction extent 64, target cells 1
  n105: reduction extent 84, target cells 640

w:bias_conv1.grad(6) loops[i527=64s,i528=28s,i529=28s,i530=6s]
    axis i527 extent 64 -> illegal: the accumulation cell mentions i530, which is not bound
                                    by a loop enclosing the reduction loop in this statement
    axis i528 extent 28 -> illegal: (same)
    axis i529 extent 28 -> illegal: (same)
    axis i530 extent 6  -> illegal: the accumulation cell mentions the reduction loop i530
                                    — not a reduction over it
```

**All four axes are rejected**, so the `sr_red_min = 64` extent floor is not even the operative
filter. The same verdict hits `kernel_conv1.grad`, `bias_conv2.grad`, `b_fc1.grad`, `b_fc2.grad`,
`w_logits.grad`, `b_logits.grad` — **every parameter-gradient accumulation in the network**. Only
two sites in the entire lenet graph seed at all, and neither is a conv gradient.

Root cause, `schedule.ml` (`Split_reduce` v1):

```ocaml
List.iter all_syms ~f:(fun s ->
    if not (List.mem enclosing_syms s ~equal:Indexing.equal_symbol) then
      invalid_arg ("Schedule.Split_reduce: the accumulation cell mentions " ^ ...
                   ", which is not bound by a loop enclosing the reduction loop"))
```

Every accumulation-cell symbol must be bound **outside** the reduction loop. OCANNL lowers a conv
bias gradient the other way round — output channel innermost (`i530`, extent 6), reduction loops
(batch 64, y 28, x 28) outside it. The recognizer's own error message names the missing
prerequisite: *"Swap it inside `<axis>` (or split-reduce it) first"* — but seeding never proposes
that composition.

Note this is the shape the issue text itself calls out (embedding backward, split-K GEMMs, conv
weight/bias gradients). The one shape v1 *can* take — output loops outside, reduction innermost —
is the MLP matmul-K shape, which is where the two seeded sites come from.

### End-to-end: no measurable win on any completed cell

| workload | default | materialized | tuned | tuned winner |
|---|---|---|---|---|
| mlp_small | 0.335 | 0.322 | **0.333** | `F_preset[bs=cfg priv cfg-thresh]` — not split |
| lenet | 81.56 | 6.603 | **6.622** | `F_split[n105 red84 out640 b32]`, arm B |
| cifar_stride | 63.72 | 21.064 | — | did not complete (see gh-532) |

step p50 ms, two-pass protocol, from-scratch cache.

- **mlp_small**: split places 4th in the search (0.2772) behind two presets and a GPU sketch
  (0.2674). Tuned ≈ default ≈ materialized.
- **lenet**: a split candidate *does* take arm B in-search, 6.2109 vs best preset 6.3941 (+2.9%) —
  but on site `n105` (the fc2 activation), **not** `bias_conv1.grad`. The margin does not survive
  replay in a fresh process: tuned 6.622 vs materialized 6.603, i.e. 0.3% slower.

The tuner reaches lenet's good schedule by *materializing* the serial segment away (arm B, 12.4×
better than default), not by splitting it — which is why splitting it buys nothing on top.

### One caveat on earlier data

Any pre-gh-521 claim that the split family "won" a cell should be discarded. Before gh-521 landed,
every `F_sketch[gpu …]` candidate failed to compile on HIP and split-reduce won mlp_small by
default; with 35 GPU sketch candidates now reaching timing, it wins nothing.

### Suggested next step

Task 3 is sound machinery pointed at sites that do not include the motivating one. Closing the gap
needs either (a) seeding `Swap` (or a loop-interchange) ahead of `Split_reduce` so the output-index
loop can be hoisted outside the reduction nest, or (b) relaxing the v1 enclosing-loop precondition
to allow accumulation-cell symbols bound *inside* the split axis, with the partials node gaining
those axes. Until one of them exists, the 73 ms is untouched and the checked-in tests pass while
the benchmark shows nothing — worth an explicit note in the task-3 docs.
