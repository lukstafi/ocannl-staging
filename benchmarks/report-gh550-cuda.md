# gh-ocannl-550, accumulation half: attribution and eager candidate release

**Box**: rog — RTX 5070 Ti Laptop GPU (12,227 MiB, compute capability 12.0), CUDA 13.3, **WSL2**
(not native Linux). Whole-cell numbers are not comparable across machines or sessions.

**Workload**: `gpt2_mini` (forward-only), `BENCH_TUNE=1`, i.e. `Train.tune_placements ~rounds:0` with
a scratch `timing_ctx` — two placement arms, arm A default, arm B materialize-all.

**Invocation**, one process per replicate, box otherwise idle, **fresh `autotune_cache_dir` per
run** (a warm cache replays a winner and searches nothing — and per gh-ocannl-568 the cache key does
not cover the Numerics policy, so a tf32-tuned entry would silently replay into a default-flags run):

```
BENCH_FIXTURE=.../gpt2_mini.safetensors BENCH_TUNE=1 bench_gpt.exe \
  --ocannl_backend=cuda --ocannl_tf32_matmuls=true --ocannl_autotune_log=true \
  --ocannl_autotune_cache_dir=<fresh>
```

Exit codes are recorded unpiped (`printf '%s\n' "$?" > <tag>.exit`); device memory is sampled by
`nvidia-smi` every 2 s alongside the run, and the per-candidate census comes from inside the process
(`autotune_log=true`, gh-550's new `census` line), so it is pinned to a candidate rather than to a
wall-clock instant.

**Pairing.** Both arms of the comparison come from ONE build, selected by a temporary
`OCANNL_550_NO_RELEASE` kill switch on `Context.release` that was removed before commit. So the
before/after is not a cross-build comparison, and the "before" arm is behaviourally identical to
master (the census counters themselves are pure).

## 1. Attribution: it is the pool table, and it is not a GC-pressure problem

One instrumented run with release disabled. The census separates four classes; the growth is in
exactly one of them.

| candidate # | label | live pools | working-pool MiB | live contexts | contexts released | live modules | modules loaded / unloaded | device MiB |
|---|---|---|---|---|---|---|---|---|
| 1 | `W_preset[bs=cfg]` | 5 | 220.5 | 6 | 0 | 3 | 4 / 1 | 229.8 |
| 10 | `F_preset[bs=256]` | 14 | 1,141.4 | 15 | 0 | 3 | 13 / 10 | 1,150.7 |
| 35 | `F_sketch[mma-gpu 32x32x16]` | 30 | 2,779.3 | 31 | 0 | 3 | 29 / 26 | 2,788.6 |
| 60 | `F_sketch[mma-gpu 16x32x0]` | 39 | 3,700.7 | 40 | 0 | 2 | 38 / 36 | 3,709.9 |
| 85 | `F_sketch[gpu 64x64x8/4x4]` | 49 | 4,724.4 | 50 | 0 | 3 | 48 / 45 | 4,733.6 |
| 110 | `F_split[n289 red256 out1024 b32]` | 65 | 6,372.2 | 66 | 0 | 4 | 64 / 60 | 6,381.5 |
| 125 | `F_preset[bs=256]` | 83 | 11,354.8 | 86 | 0 | 5 | 82 / 77 | 11,364.1 |
| 261 (last) | `F_split[8 sites]` | 128 | 28,802.8 | 131 | 0 | 3 | 127 / 124 | 28,812.1 |

Read the columns against each other:

- **Working pools grow one per candidate, ~102 MiB each, and `pools_freed` stays 0 for the whole
  search.** That is the entire curve: `device MiB` (the backend's own `Context.get_used_memory`)
  equals live working bytes plus the single 9.3 MiB constant pool, to the decimal, at every sample.
- **Contexts grow one per candidate and `contexts_released` stays 0.** Same shape, same cause.
- **Modules do NOT accumulate.** 82 loaded, 77 unloaded, live count flat at 2–5 across the whole
  search. cudajit unloads a module from its own GC finalizer and that finalizer fires perfectly
  well.
- **Constant pools do not accumulate** (one, 9.3 MiB, flat): they are deduped per device by
  `constant_buffer_cache`, so a second candidate reuses the first's.
- **The merge pool cannot accumulate** by construction — one reserved entry per device, grown in
  place.

### Where the failure lands on THIS box, and why the landmark moved

The reference failure (benchmarks/report-gh528-gpt2-cuda.md §3, five of five replicates) hit absorbed
`Backend_link` out-of-memory declines from **arm-B candidate 47** onwards. Here it did not: this run
took **no** per-candidate OOM decline at all, ran all 261 candidates, and met
`CUDA_ERROR_OUT_OF_MEMORY` only in the aftermath — the winner replay and the untuned-default fallback
behind it (`autotune: winner replay FAILED (F_saved[66 segs]) … CUDA_ERROR_OUT_OF_MEMORY`). Contained
by the gh-550 robustness half (lukstafi/ocannl-staging#295), so arm B ranks at `infinity`, arm A ships
at 105.008 ms, step p50 102.636 ms, and the process exits 0.

The reason the landmark moved is the box, not the bug: **on WSL2 the CUDA driver backs allocations
past VRAM with host memory**, so the run reached **28.8 GB requested** on a 12,227 MiB card while
`nvidia-smi` sat pinned at 11,879 MiB. The runway is longer and the symptom is different — candidate
times degraded from ~105 ms to **3,563 ms** as the search began thrashing, and total compile time was
**767 s** for what the fixed runs do in a fraction of that. So the position of the first OOM is a
property of how much host memory the driver can borrow, and "candidate 47" should be read as a
landmark rather than a specification. The invariant across boxes is the curve: +1 pool and ~102 MiB
per candidate attempted, `pools_freed = 0`.

### The mechanism, which is not the one the issue expected

The expectation on file was "artifacts are released by GC finalizers, and the OCaml GC feels no
pressure from device memory, so finalizers never run." The module column refutes that: on this very
run, in this very process, 77 of 82 module finalizers ran. Nothing is wrong with GC pressure.

What is wrong is that a pool can never be finalized **at all**. Each backend keeps a private
`Slab.pools : (device_id, pool_id) -> base` table (`Backend_impl.Make_slab` for the C backends,
`Cuda_backend.Slab` for CUDA), and that table is a module-level value — a strong GC root holding
every slab it ever allocated. The comment at `Make_slab.free_pool` says so explicitly ("otherwise
`pools` keeps a strong reference to every tnode buffer for the lifetime of the backend module and
the GC finalizer never runs"), and the one function that drops an entry, `Backends.finalize`, **had
no caller anywhere in the repo**: it was reachable only through `Backends.finalize` itself, exported
from `backends.mli` with a doc comment saying it is "not obligatory because all pools are freed when
their backend buffers are garbage-collected" — which is exactly what the table prevents.

So this is closer to a true leak than to a missed collection, and it changes the fix: no amount of
`Gc.full_major` between candidates could have helped, and none is used.

### It scales with candidates *attempted* — not with kept, and not even with ranked

The issue's first open question was whether peak memory scales with the kept candidates or with the
whole ranked set. Neither. The pool is allocated at **link**, before the search decides anything
about the candidate, so:

```
autotune: W_preset[bs=cfg]: dedup (digest 6bf339c4/84848ff5)
autotune: census after W_preset[bs=cfg]: pools 5 live = 4 working (220.5 MiB) ... 0 freed
autotune: W_preset[bs=64]: dedup (digest 6bf339c4/84848ff5)
autotune: census after W_preset[bs=64]: pools 6 live = 5 working (322.8 MiB) ... 0 freed
```

Four *deduplicated* `W_preset` candidates — candidates the search discards as identical to one it
already has — each pay a full 102 MiB working pool. `Backend_link` declines pay the same way, which
is why an out-of-memory decline used to make the next one likelier rather than less.

## 2. The fix: eager release at the candidate boundary

`Context.release` (idempotent, eager, finalizer-independent) frees the pools a context owns and its
parent does not, and drops the table entries — the existing `Backends.finalize` seam, now with a
caller. `Autotune.tune` uses it at the one place that needs no allocator to know a lifetime:

- the candidate pool IS the beam, bounded at `beam_width` as candidates arrive (keeping the k
  smallest incrementally keeps the k smallest overall), and whatever falls out is released;
- a dedup, a non-dispatchable degeneration and a contained run failure release immediately;
- a beam round accumulates into a second bounded list, so a round's also-rans do not survive it
  either (16 compiles per round on the gh-ocannl-543 chain);
- the exit sweep runs after the report is built and **before** the winner replay and the
  untuned-default fallback — the two compiles the exhausted device used to defeat;
- survivors are excluded by physical identity: the beam, and the running best (which can lag the
  beam by one sub-threshold round). The digest-dedup tables hold strings and floats, never a
  compiled artifact, so nothing can resurrect a released candidate.

## 3. Result

Three cold tf32 replicates with release on, fresh cache dir each, same binary as §1.

| run | exit | OOM occurrences | candidates | peak `nvidia-smi` MiB | max working-pool MiB | pools alloc / freed | arm A best (ms) | arm B | shipped | step p50 (ms) | compile s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| r1 | 0 | **0** | 261 | 1,935 | 1,286.0 | 129 / 122 | 102.091 | **190.731 (completed)** | A | 102.134 | 105.2 |
| r2 | 0 | **0** | 261 | 1,924 | 1,286.4 | 129 / 122 | 101.911 | **186.255 (completed)** | A | 102.343 | 104.8 |
| r3 | 0 | **0** | 261 | 1,930 | 1,287.1 | 129 / 122 | 103.216 | **190.775 (completed)** | A | 103.189 | 145.0 |
| *(§1, no release)* | 0 | 5 | 261 | 11,879 | **28,802.8** | 127 / 0 | 105.008 | FAILED (OOM) | A | 102.636 | 767.1 |

The curve flattens, and it flattens at the beam rather than at any function of the search's size. In
every replicate the live working-pool count oscillates between **4 and 7** for all 261 candidates; the
byte figure tracks what the beam happens to hold (~118–323 MiB through arm A, up to 1,287 MiB in arm
B, whose materialize-all candidates are each ~1 GB) and never trends.

The paired point, candidate 125 of the same 261-candidate stream:

```
release ON : pools  6 live =  5 working (  893.4 MiB) + 1 constant; 83 allocated, 77 freed | contexts  9 live
release OFF: pools 83 live = 82 working (11354.8 MiB) + 1 constant; 83 allocated,  0 freed | contexts 86 live
```

Same 83 pools allocated — the search is unchanged, candidate for candidate — and the difference is
entirely in what was given back.

Three things beyond the memory number:

1. **The OOM does not occur at all** — not absorbed, not contained: absent. `grep -c OUT_OF_MEMORY`
   is 0 in each replicate.
2. **Arm B now completes.** In §1 it died at its winner replay and ranked at `infinity`; here it
   returns 190.731 ms and loses the A/B honestly. The containment fix (#295) made the failure
   survivable; this makes it not happen, so arm B's result is available for comparison again.
3. **The search got 7.3x faster in wall clock** (767.1 s → 105.2 s). That is not allocator overhead:
   it is the unfixed run thrashing once WSL2 starts backing allocations with host memory (its
   candidate times degraded from ~105 ms to 3,563 ms). Expect the speedup to be smaller on a box
   that OOMs promptly instead of spilling — the shipped step time is unchanged either way (102.134
   vs 102.636 ms, same crowned label).

## 4. Rider: `test_cuda_pool_offset` is not stale

This CUDA-only golden pins arena buffer offsets and had not run since gh-498's canonical-ordering
change (`ecf299a7`), so it was presumed stale. It is not: run here it matches byte-for-byte, so there
is nothing to promote.

Checked before believing that, because this test's off-platform stub *echoes the golden file* and would
pass vacuously if `select` had picked it: `_build/default/test/operations/test_cuda_pool_offset.ml`
resolves to `test_cuda_pool_offset.real.ml`, and replacing the golden with a sentinel left the output
unchanged (the stub would have echoed the sentinel).

Why gh-498 did not move these offsets: canonical order is by `Tnode.uid`, `p` is created before `q`,
so uid order coincides with the `traced_store` order the golden was recorded under. The invariants
hold on the printed values — `p` at offset 0 and `q` at offset 32, 8 bytes each, so no overlap, and
both offsets are multiples of `Ops.buffer_alignment` (32), the spacing being exactly that alignment's
padding.

One gap worth naming rather than fixing here: this test prints offsets but not the pool's total size,
so a *packing* regression (the same non-overlapping offsets in a needlessly larger arena) would not
show up in it. Answering that question needs a test that asserts on pool bytes;
`Ir.Alloc_census.live_pool_bytes` now makes that cheap to write.

## Provenance and hygiene

- **WSL2**, RTX 5070 Ti Laptop, CUDA 13.3. Not comparable across machines.
- Fresh `--ocannl_autotune_cache_dir` per replicate; no two GPU processes ran concurrently.
- Exit codes read from a file written by the runner, never from a pipeline's status.
- The debug `.cu` left on disk after a `tune_placements` run belongs to **arm B** (the last search),
  not to the shipped artifact; no claim here is made from it.
- The census is exact for what it counts and is not a device-memory total: it excludes the reserved
  merge pool (one entry per device, grown in place) and, on `cc`, counts host allocations whose GC
  finalizer has not yet run. `hip` and `metal` leave the module counters at zero rather than
  reporting a wrong number.
