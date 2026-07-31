## The diagnosis in this issue is wrong — it is not a middle-end wedge

Re-ran this issue's own repro at `8436e362` under WSL2 on minix (gfx1151, ROCm 7.14), on an idle
machine, with `perf`. The *symptom* reported here is real and reproduces exactly. The diagnosis
does not hold, and two of the three evidence claims are inaccurate.

### 1. 100% of the CPU time is in the HSA runtime, not the compiler

`perf record -F 199 -g` over 20 s of the spinning process, 11,890 samples:

```
66.63%  bench_mlp.exe  libhsa-runtime64.so.1.21.0  rocr::core::Runtime::AsyncEventsLoop(void*)
33.37%  bench_mlp.exe  libhsa-runtime64.so.1.21.0  rocr::core::BusyWaitSignal::WaitRelaxed(...)
```

**Zero OCaml frames.** Re-sampled 85 minutes later: byte-for-byte the same two symbols in the same
2:1 ratio. The process is not lowering, scheduling or digesting anything — it is busy-waiting on an
HSA completion signal for a kernel that has already been compiled and dispatched. The "constant
~3-thread CPU spin" this issue reports is ROCr's busy-wait, not compiler work.

### 2. "Zero debug artifacts written" is not correct

With `--ocannl_output_debug_files_in_build_directory=true`:

```
09:12:53                 process start
09:12:55.5 – 09:12:55.8  12 artifacts, incl. linked .hsaco, for init_params_for_cross_entropy_loss
                         and cross_entropy_loss_forward_and_gradient_then_sgd_update
09:12:55.8 → +2 h        nothing further
```

Codegen and hiprtc are reached and succeed in about 3 seconds. What is zero is artifacts written
*after* the search banner. As phrased ("never reaches rendering or hiprtc, not even for the serial
baseline") the issue sends the reader to the wrong subsystem.

### 3. The mechanism: the serial baseline candidate is unbounded in cost

`Autotune.tune` times the unfissioned serial baseline before any candidate (`time_routine`). The
dispatched kernel `cross_entropy_loss_forward_and_gradient_then_sgd_update.hip` contains **no
`threadIdx` or `blockIdx` reference anywhere** — every loop is a plain serial `for`:

```
for (i219 <= 255) for (i220 <= 1023)                     ->   262 K iterations
for (i95  <= 255) for (i96  <= 1023) for (i97  <= 255)   ->  67.1 M
for (i102 <= 255) for (i103 <= 1023) for (i104 <= 1023)  -> 268.4 M
... plus the backward nests
```

So the entire mlp_wide training step executes in **one work-item** — order 1e9 dependent scalar
iterations — and it executes **four** times, not three: `time_routine` does an untimed warmup run
before the `autotune_repeats=3` loop.

The affected/unaffected table in this issue is then just a per-step-compute ranking:

| cell | step FLOPs (approx) | outcome |
|---|---|---|
| `mlp_small` dims [2,64,64,2] b64 | ~1.7 MFLOP | completes, ~40 s |
| `lenet` | ~30 MFLOP | completes |
| `mlp_wide` dims [256,1024,1024,10] b256 | ~2 GFLOP | "wedges" |
| `cifar_conv` | larger | "wedges" |

Single-lane throughput calibrated from the cells that do complete is 7–46 MFLOP/s; mlp_wide's
1024×1024 weights are cache-resident for nobody on one lane, so it should be lower still. A clean
re-run was **capped at 2 h with no `baseline:` line**, having spent the whole time in
`BusyWaitSignal::WaitRelaxed`. That is the predicted duration for four runs, not a contradiction of
it. The 50-minute watchdog was truncating a search that was progressing.

### 4. It also takes the display driver down

These are multi-hour uninterruptible dispatches on the same device that drives the display. In one
session this produced two Windows-side "AMD Bug Report Tool" driver-timeout notifications, a
transient GPU fault that made a sub-2-minute job hang for 30 minutes (the same job ran in <120 s
afterwards), a silent death of `cifar_stride hip/tuned` at ~11.5 min, and finally **loss of display
output requiring a reboot**. The WSL guest's `dmesg` was clean of GPU faults throughout, so none of
this is visible from inside Linux — which is part of why it kept looking like a hang.

That makes bounding the baseline a stability requirement, not a performance nicety.

### Consequences

- **Retitle**: not "wedges in the middle-end before its first candidate" but "the serial baseline
  candidate is unbounded in cost relative to the workload".
- **The bisect this issue asks for would have chased nothing.** WSL-vs-Windows was never the right
  axis — the symptom reproduces at `8436e362`, and the native-Windows 111.78 s figure is more
  likely a search against a warm `autotune_cache` (the benchmarks README warns it must be wiped for
  `compile_s` to mean anything) than an environment difference.
- **Fix directions**: bound or skip the baseline candidate by predicted cost — the roofline model
  already prices it, and a baseline it prices orders of magnitude off the parallel presets need not
  be measured at all; or seed the baseline from a cheap parallel preset rather than the unscheduled
  form. Either also fixes the driver-stability problem.
- `SKIP_CELLS` entries for `mlp_wide hip/tuned` and the `cifar_conv` watchdog kill are masking a
  cost problem, not a hang.
- **Probably not HIP-specific.** `time_routine` times the serial baseline on every backend, so any
  sufficiently heavy workload should show it. Worth checking the CUDA and Metal legs — the tell is
  a tuned cell that "hangs" where the untuned one is fine.

`benchmarks/report-hip.md` has been corrected accordingly (it carried the same wrong diagnosis).
