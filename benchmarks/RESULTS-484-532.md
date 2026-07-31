# gh-484 measurement + gh-532 diagnosis — HIP / gfx1151 / WSL2 (minix)

Tree: `e938d504` (= origin/master `d71efc99` + the BENCH_SR_SITES probe), i.e. **post gh-527
and post gh-521**. All cells HIP, `taskset -c 0-15`, run serially on an otherwise idle machine.
Raw logs were lost to a reboot; these are the extracted results.

## gh-484: per-segment attribution (BENCH_SEG_TIMES, min of 20)

The measurement the gh-476 report deferred until gh-527 was fixed.

| workload | dominant segment | report (bd075cd5) | now | factor | share before -> after |
|---|---|---|---|---|---|
| lenet/hip | seg22 `bias_conv1.grad`/`n65.grad` | 350.712 ms | **72.76 ms** | 4.8x | 97.5% -> **89.3%** |
| cifar_stride/hip | seg22 `bias_conv1.grad`/`n65.grad` | 201.416 ms | **50.03 ms** | 4.0x | 93.3% -> **78.7%** |
| lenet step total | | 359.743 ms | **81.49 ms** | 4.4x | |
| cifar_stride step total | | 215.778 ms | **63.56 ms** | 3.4x | |

lenet seg22 repeatability: 73.132 / 72.773 / 72.756 ms across three runs (<0.5% spread).
Seg-times sums cross-validate against measured step times to within 0.25% (lenet 81.49 vs
81.56; cifar_stride 63.56 vs 63.72).

**Decomposition**: of lenet's original 350.7 ms segment, ~278 ms was gh-527's pooling-gate
regression and **~73 ms is the inherent serial reduction** — gh-484's actual subject. It
remains the single dominant cost of the default-placement step at 89%.

## gh-484: does task 3 reach it? No.

(Independently confirms the CUDA leg's headline — `benchmarks/report-gh484-cuda.md`, merged in
parallel — on a second backend, and answers that leg's open question #1, "why are the conv-gradient
accumulations rejected": it is `Sched.op_legality`, not the dedup-by-axis-symbol.)

`BENCH_SR_SITES=1` (this branch's new probe) on lenet, current tree:

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

All four axes rejected — so the `sr_red_min = 64` extent floor is not even the operative
filter. The same verdict hits `kernel_conv1.grad`, `bias_conv2.grad`, `b_fc1.grad`,
`b_fc2.grad`, `w_logits.grad`, `b_logits.grad`: **every parameter-gradient accumulation in
the network**.

Root cause: `Split_reduce` v1 (schedule.ml:2118) requires every accumulation-cell symbol to
be bound by a loop *enclosing* the reduction loop. OCANNL lowers a conv bias gradient with
the output-channel loop **innermost** and the reduction loops (batch, y, x) outside it —
the exact inverse. The error message names the missing prerequisite itself: "Swap it inside
<axis> (or split-reduce it) first". Seeding never proposes that composition.

## gh-484: end-to-end cells

| workload | default | materialized | tuned | tuned winner |
|---|---|---|---|---|
| mlp_small | 0.335 | 0.322 | **0.333** | `F_preset[bs=cfg priv cfg-thresh]` (not split) |
| lenet | 81.56 | 6.603 | **6.622** | `F_split[n105 red84 out640 b32]`, arm B |
| cifar_stride | 63.72 | 21.064 | INCOMPLETE | — |

(step p50 ms, two-pass protocol, from-scratch cache)

- **mlp_small**: split family places 4th (0.2772) behind two presets and a GPU sketch
  (0.2674). Tuned == default == materialized: no win.
- **lenet**: a split candidate does win arm B in-search (6.2109 vs best preset 6.3941,
  +2.9%) — but on site `n105` (fc2 activation), **not** `bias_conv1.grad`. The margin does
  not survive replay: tuned 6.622 vs materialized 6.603, i.e. 0.3% slower.
- Net: **gh-484 task 3 delivers no measurable end-to-end benefit on any completed cell**,
  and cannot reach the site it was filed for.

## gh-521 changed the competitive picture

Yesterday (pre-rebase) every `F_sketch[gpu …]` candidate failed to compile on HIP and the
split family won mlp_small by default. Post-gh-521, 35 GPU sketch candidates reach timing
and split-reduce wins nothing. Any pre-gh-521 claim that the split family "won" a cell is an
artifact of its competition being broken.

## gh-532: the diagnosis in the issue is wrong

Re-run of the issue's own repro at `8436e362` under WSL, idle machine.

1. **Not a middle-end wedge.** `perf record`, 11,890 samples: 66.63%
   `rocr::core::Runtime::AsyncEventsLoop`, 33.37% `rocr::core::BusyWaitSignal::WaitRelaxed`,
   both in `libhsa-runtime64`. **Zero OCaml frames.** Re-sampled at 85 min: identical.
2. **"Zero debug artifacts" is false.** 12 artifacts including a linked `.hsaco` appear
   within 3 s. What is zero is artifacts *after* the search banner.
3. **Mechanism: the serial baseline candidate.** The dispatched kernel
   `cross_entropy_loss_forward_and_gradient_then_sgd_update.hip` contains **no `threadIdx`
   or `blockIdx` reference at all** — every loop is a plain serial `for`, e.g.
   `for(i102<=255) for(i103<=1023) for(i104<=1023)` = 268 M iterations. Order 1e9 dependent
   scalar iterations in ONE work-item, executed **four** times (`time_routine` does an
   untimed warmup plus `autotune_repeats=3`). Capped at 2 h with no `baseline:` line.

The affected/unaffected split in the issue's table is just a per-step-compute ranking:
mlp_small (~1.7 MFLOP) completes in 40 s; lenet (~30 MFLOP) completes; mlp_wide (~2 GFLOP)
and cifar_conv do not. WSL-vs-Windows was never the right axis — and the native-Windows
111.78 s figure is more likely a search against a warm `autotune_cache` (the README warns
it must be wiped) than an environment difference. **Do not spend a bisect on this.**

## gh-532 also destabilizes the display driver

Across this session: two "AMD Bug Report Tool" driver-timeout notifications, one transient
GPU fault that made a 2-minute job hang for 30 minutes (it ran in <120 s afterwards), a
silent death of `cifar_stride hip/tuned` at ~11.5 min (exit 15, only the arm A banner
written), and finally **loss of display output requiring a reboot**. The WSL guest's
`dmesg` is clean of GPU faults throughout — this is only visible from the Windows host.

Common factor: multi-hour uninterruptible single-work-item dispatches on the same device
that drives the display. This raises bounding the serial baseline from a performance
nicety to a stability requirement.

## Open

- `cifar_stride hip/tuned` has no step-time number (three attempts), but the cause is now
  **settled rather than assumed**. A fourth attempt on the rebased tree was perf-sampled 45 s
  in, at which point a Windows driver-timeout notification had already fired: 66.63%
  `AsyncEventsLoop` / 33.35% `BusyWaitSignal::WaitRelaxed`, **zero OCaml frames** — identical
  to mlp_wide to two decimal places. Same slow serial baseline, not a distinct bug. The run
  was stopped deliberately: the diagnostic question was answered and the remaining value (one
  step-time number) did not justify another display-driver crash. This positively refutes the
  earlier suspicion that gh-484 task 3 introduced a new wedge here.
- Suggested fixes for gh-532: bound or skip the baseline candidate by predicted cost (the
  roofline model already prices it); or seed the baseline from a cheap parallel preset
  rather than the unscheduled form. Both also fix the driver-stability problem.
- Worth checking whether CUDA/Metal legs show the same signature on heavy workloads.
