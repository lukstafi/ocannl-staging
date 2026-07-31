# Candidate-failure containment (gh-ocannl-536)

## The problem, restated

The compile path has one channel for "this did not work" — `exn`, overwhelmingly
`Invalid_argument` — and every site that observes it holds a private theory of what it means. The
inventory is larger than the three sites the issue names:

| layer | site | today |
|---|---|---|
| candidate loop | `compile_candidate`'s `with exn -> Error (Exn.to_string exn)` (autotune.ml:2719) | catches everything, collapses to a string |
| candidate loop | `try_spec` (autotune.ml:3367): the `FAILED` arm at :3370 and the `RUN FAILED` arm at :3406, the latter wrapping `time_routine` — launch *and* sync | catch everything |
| cache replay | the same `compile_spec_real` `Result` consumed at autotune.ml:3325, `cache entry replay FAILED, re-searching` | catches everything (and should) |
| advisory pick | `compile_advisory` (autotune.ml:2966), `model_default`'s counting arm at :3033, and four bare `exception _ -> None` arms (autotune.ml:3065, 3082, 3097, 3118) | catch everything, then recompile |
| tuner internals | `exception _` fallbacks at autotune.ml:2054 (→ `256 * 1024`), 2166, 2172, 2282, 2316, 3471 | catch everything, substitute a default |
| backend selection | `Context.auto`'s `try … with _ -> try_backends rest` (context.ml:103, :109) | catches everything, silently picks another backend |
| hand schedules | a user's own `?lowered_transform` through `Context.compile` (context.ml:111) | catches nothing |

Two corrections to the issue's reading of its own three instances, because they change what the
design has to do:

- **#519 is already fixed, and its fix is a hand-rolled instance of what this note proposes.** The
  landed fix has two halves: `validate_segments` (autotune.ml:2963) runs `validate_parallel`
  *eagerly* at the transform seam so the seam's handler fires (`validate_parallel` signals purely
  through `invalid_arg`, low_level.ml:3093-3190), and `compile_advisory` adds a
  backstop catch around `Context.compile` with a `fallback_if` guard against recompiling a failure
  the default pipeline shares. That is a phase boundary, a classification, and a fallback policy —
  written once, by hand, for one caller. The design's job is to make it the general case, not to
  fix #519 again.
- **#533 is not a "too narrow guard".** `try_spec`'s arm is `| exception exn ->` and wraps
  `time_routine`, which contains both `Context.run` and `Context.sync` (autotune.ml:68-95). A HIP
  driver abort raised out of `H.Stream.synchronize` (hip_backend.ml:349) *is* that shape and *is*
  caught. So either the fatal escape came from one of the two timing sites that are deliberately
  uncaught — the baseline at autotune.ml:3340, documented as the user's bug — or from the
  benchmark harness above `tune`, in both cases *after* an earlier candidate's abort had already
  killed the queue. Either way the catch was not the problem: the search continued on dead state.

That second correction is the whole reason this design needs two axes rather than one. It also
means **the escape site in #533 must be established before implementation** — the traceback in the
issue is truncated above `time_routine`, and the two readings imply different fixes. This is a
one-line instrumentation ask on the HIP machine (print the caller, or bracket the baseline timing),
not a design question.

So the three instances sort as: #519 = attribution, scope of the guard (fixed); the too-wide
catches above = attribution, wrong direction, unfiled; #533 = **damage**, orthogonal to how wide
any catch is.

## Where the current behaviour is actually wrong

Stripped of the "three sites guessing" narrative — post-#519 they all guess the same way, *catch
everything* — three defects remain, and they are the deliverables:

1. **Over-containment.** `with exn -> Error …` also absorbs `Out_of_memory`, `Assert_failure` and
   `Stack_overflow`. There are 20 `assert`s in low_level.ml and 13 in schedule.ml; a middle end that
   trips one during a tuning run reports a `FAILED` line and crowns some other candidate. That is
   the failure mode the issue names as worse than the crash. The current code has it.
2. **No recovery.** Nothing in the search asks whether the device still works after a candidate
   failed. #533 is this.
3. **No census.** `Exn.to_string exn` is where the structured decline data is thrown away.
   `Autotune.report` carries `candidates_failed : int` (autotune.ml:10) and nothing else — a single
   counter incremented at :3371 and :3407, with the message only logged, and hardcoded to `0` on the
   cache-hit path (:3308). #521's blocker table was assembled by hand-grepping `autotune_log`, and
   gh-479's "did the tensorized candidate lose, or never run?" is unanswerable from the returned
   value. `model_default` keeps its own parallel counter (`n_rejected`, autotune.ml:3033) — a second
   hand-rolled census of the same event.

## Exhibit: the same constructor, opposite meanings

`invalid_arg` is called 108 times in `schedule.ml`, all op preconditions — "this candidate is not
applicable", expected, skippable. (The `grep -c` figure of 113 that circulates includes one doc
comment and four *catch* sites at schedule.ml:3121, 3515, 3617, 3643; `raise (Invalid_argument …)`
appears zero times.) It is also what the cc backend raises when the generated C fails to compile
(cc_backend.ml:228), where the message itself reads *"This is a bug in OCANNL.
Please file an issue with the generated .c file"*. One constructor, two meanings at opposite ends
of the containment decision.

That second one is a trap this design must not walk into. Containing a cc codegen failure is
*correct* for the search — the candidate is candidate-specific and the run is better off
continuing — and it institutionalizes hiding an OCANNL bug behind a decline. The taxonomy below
therefore carries a severity distinction: `Backend_rejected` from a C backend is always a bug, and
its count is reported and logged even when the search survives it. Contained must not mean quiet.

`Utils.User_error` (utils.ml:935) is the closest thing to an existing attribution channel, and it
is ambiguous at exactly the interesting place: it carries the pool-capacity errors that are genuine
user misuse ("set `large_models=true`", backends.ml:33-38) *and* the hardware-limit rejections that
are ordinary declines (`Schedule.check_hardware_limits`, schedule.ml:5120, raising at 5126 and
5137; metal_backend.ml:1250). It is also raised across some fifteen modules for unrelated reasons,
so "is a `User_error`" carries no containment information on its own.

And on CUDA the question cannot be asked at all: `Nvrtc.compile_to_ptx` (cuda_backend.ml:292) is
called bare — the file contains no `try` at all — and no site in the tree names a cudajit exception
constructor. Whatever those bindings raise, we classify it by not looking.

## The conceptual core: two axes, not one

The issue's `Valid | Declined of reason | Fatal` is right about the shape and one axis short.

1. **Attribution** — *this schedule cannot be built or run here* versus *this process is in
   trouble*. Decides whether the search may continue at all.
2. **Damage** — what state the failure destroyed: nothing (a compile-time refusal), the
   stream/queue, or the device context. Decides what must be rebuilt before the search *can*
   continue.

#533 is attribution=candidate, damage≠nothing, which is exactly the cell no amount of widening or
narrowing a catch addresses.

**Damage is measured, not declared.** The note's earlier form had the classifier report a `damage`
field, which no classifier can honestly fill: the exception that surfaces from
`H.Stream.synchronize` is `HIP_ERROR_INVALID_VALUE`, indistinguishable from a dozen benign launch
errors, and the queue's state is not in it. Replace the declared field with a probe:

```ocaml
val device_health : device -> [ `Healthy | `Recovered | `Lost ]
```

which the search calls after any non-`Valid` outcome at the launch phase. A backend implements it
as "run a trivial no-op routine and synchronize; on failure, try re-creating `device.runner`; on
failure again, `Lost`". That answers the question the exception cannot, costs one launch per
failure (rare by construction), and needs no per-driver error-code table.

**The probe answers the damage question and only the damage question.** `Recovered` establishes
that the state was rebuildable; it says nothing about whose fault the failure was, and the two axes
must not be allowed to leak into each other here of all places. So `device_health` never upgrades a
verdict: a `Fatal` from rule 3's launch arm stays `Fatal` whatever the probe reports, and the probe
only decides whether an already-`Rejected` outcome (one rule 2 attributed) may continue on this
device or must escalate for lack of a device to continue on. Letting `Recovered` turn an
unattributed launch failure into a decline would reintroduce exactly the silent degradation the
whole design exists to remove — and would be indistinguishable, from the report's point of view,
from today's blanket `RUN FAILED`.

Re-creating the runner is a plausible `Recovered` on CUDA — `Cu.Stream.create` is called exactly
once in the tree (cuda_backend.ml:331, one compute stream per device), so the handle has one owner.
Whether an HSA-aborted HIP queue survives it is an empirical question for the HIP machine, and the
probe is how we find out rather than something to assume.

## Provenance: the verdict is not a property of the failure

`Schedule.check_hardware_limits` raising on an oversized workgroup is the right answer for a
hand-written schedule — the user asked for something the device cannot do — and the wrong answer
for a tuner candidate, where it is an ordinary decline. Nothing about the exception distinguishes
them; the difference is entirely *who proposed the schedule*.

This rules out the tempting design where the exception constructor carries the verdict (a
`Candidate_declined` exception that catch sites match on), and with it the reflex of rewriting 108
`invalid_arg` sites: those sites do not know their verdict. The vocabulary describes the *failure*;
the *caller* supplies the policy:

- the candidate loop: rejection ⇒ decline, continue;
- cache replay: any failure ⇒ discard the entry, re-search (the entry is untrusted persisted data,
  possibly from another machine or an older tree);
- `model_default` (documented advisory): rejection ⇒ fall back to the default pipeline;
- a user's own `lowered_transform`: rejection ⇒ raise, reason rendered — today's behaviour,
  unchanged, better worded.

## Vocabulary

Two thirds of this vocabulary already exist, one layer down and one layer up:

- `Schedule.op_verdict = Op_legal | Op_illegal of string | Op_unknown of string` (schedule.ml:3170)
  is the *pre-compile* three-valued oracle. Its `Op_illegal` is built by catching `Invalid_argument`
  out of a hermetic apply (schedule.ml:3515 and 3643 out of `apply_opt_op`, 3617 out of
  `tensorize_llc`) — the legality layer already treats
  "`Invalid_argument` escaping `apply`" as a decline verdict. The new type is the *post-hoc*
  counterpart of that judgement and should read like it.
- `C_syntax.mma_rendering = Mma_intrinsics | Mma_register_tiled | Mma_scalar_fallback`
  (c_syntax.ml:41) with its census is the *rendering-level* outcome record; declines below it are
  log-only strings (`declinef`, c_syntax.ml:845).

What is missing is the boundary in between. Proposed:

```ocaml
(* arrayjit/lib/schedule_outcome.ml — depends on Base only; below Backend_intf and Schedule. *)

type resource =
  | Workgroup_threads
  | Workgroup_memory        (* threadgroup / __shared__ *)
  | Thread_scratch          (* private segment, local memory, spills — #533 *)

type rejection =
  | Illegal_schedule of { check : string; detail : string }
      (* op preconditions in [Schedule.apply]; [Low_level.validate_parallel]. *)
  | Unsupported of { feature : string; detail : string }
      (* no mma for this precision/extent/arch; thread-space operands. *)
  | Resource_exceeded of {
      resource : resource; requested : int; limit : int option; detail : string }
  | Backend_rejected of { stage : string; detail : string }
      (* nvrtc / hiprtc / MSL / cc compiler diagnostics. Always also a bug report on cc. *)
  | Unclassified of string  (* migration bucket; see the ratchet below. *)

type outcome = Valid | Rejected of rejection | Fatal of exn
```

`rejection` is deliberately coarse: it is a *census key*, not a diagnosis. The free-text `detail`
keeps the message that exists today, so no diagnostic power is lost while the taxonomy stays small
enough that #521's blocker table becomes a fold over the report.

Two variants from the earlier draft are deliberately absent. `Registers` has no reporting site
anywhere. `Grid_private_bytes` — the cc pool-worker stack cap — looked like a resource rejection
and is not one: c_syntax.ml:1185 does not raise, it *declines the parallel rendering and keeps the
loop serial*, logging through `declinef`. Folding it in would merge the renderer census with the
candidate census, which is precisely the distinction gh-479 exists to preserve. Renderer-level
degradations stay in `declinef` and `mma_census`.

Also out of scope, and worth naming so they are not swept in: the internal bail-outs
(`exception Bail`, `Opaque_stmt`, `Unfissionable` in schedule.ml, `Fail` in affine.ml) are analyses
declining inside a phase, never crossing a boundary. They stay local.

## Policy: classify at the phase boundary, not at the raise site

The phase fixes the meaning of anything that escapes it, which is what makes the change small.
Three rules, applied in order:

1. **Fatal deny-list, unconditional.** `Out_of_memory`, `Stack_overflow`, `Sys.Break`,
   `Assert_failure`. Host exhaustion, user interrupt, and "the compiler is confused" are never a
   candidate's fault, at any boundary. This is the part of the change that *removes* containment
   rather than adding it. (`Utils.User_error` deliberately does **not** join the list: as above, it
   is what the hardware-limit declines are made of.)
   *Exception, and it is not cosmetic:* cache replay (autotune.ml:3325) opts out of the deny-list
   for everything except `Out_of_memory` and `Sys.Break`. A cache entry is untrusted data; an
   `Assert_failure` while replaying one means the entry is incompatible with this tree, not that
   the compiler is confused, and the correct response is to discard it and re-search.
2. **Backend override.** A new `Backend_intf` hook, `classify_failure : exn -> outcome option`,
   lets a backend recognize its own driver errors; `None` means "not mine". `backend_intf.ml` has no
   failure vocabulary at all today, so this is genuinely new surface: it belongs in
   `Backend_device_common` (backend_intf.ml:269), which `Backend` includes, and every implementor
   must supply it — the four real backends, `backend_impl.ml`'s functors, and the `*_missing.ml`
   stubs, with `backends.ml`/`context.ml` doing the first-class-module dispatch. Budget for that
   breadth; the default (`fun _ -> None`) keeps each one a one-liner until it has something to say.
3. **Phase default.**
   - `Schedule.apply` / `check_hardware_limits` → `Rejected` (`Illegal_schedule`,
     `Resource_exceeded`). The boundary knows these are op preconditions even though the 108 raise
     sites do not.
   - backend codegen, compile **and link** → `Rejected`. This covers `validate_parallel` (inside
     codegen, past the transform seam — the geometry #519 was about), the backend compilers, and
     Metal's post-link threadgroup check at metal_backend.ml:1243-1253, which is where the analogous
     scratch check below will live. `Context.compile` calls `Backend.compile` then `Backend.link`
     (context.ml:111-130); both are one phase for this purpose.
   - launch / sync (`Context.run`, `Context.sync`) → **`Fatal`**. A run failure is normally the
     program's bug (uninitialized inputs — the property `tune` already documents for its uncaught
     baseline timing, autotune.ml:3338), and rule 2 is the only way a launch failure becomes
     attributable.

Rule 3's compile arm is safe for a reason already load-bearing in the architecture: `tune`'s *base*
compile is uncaught (autotune.ml:3253). A compile failure the computation has regardless of
schedule — a genuine user error such as the pool-capacity one — surfaces there, before any
candidate is proposed. What remains, by construction, is failures only some candidates have.

**Two behavioural tightenings, not one.** Rule 1 is the first. Rule 3's launch arm is the second:
`try_spec`'s blanket `RUN FAILED` catch becomes "decline only what the backend claims, else fatal".
That is deliberate — a candidate that produces garbage should not be quietly scored — and it is the
arm #533 populates.

## This design does not, by itself, fix #533

Worth stating plainly, because the earlier draft let the reader assume otherwise. Under rule 3 a
launch/sync failure is `Fatal` unless HIP's `classify_failure` attributes it, and what HIP delivers
at that point is `HIP_ERROR_INVALID_VALUE` out of `synchronize` — the same code an uninitialized
input yields. There is no signal to classify on. So after this design lands, #533's abort is still
fatal; what changes is that it is *attributed*: the crash names the candidate and its rejection
reason instead of surfacing as a bare `hip_stream_synchronize` backtrace. That is a real
improvement over today and it is not containment.

Containment of #533 therefore comes from prediction, not from the probe. `device_health` only
supplies the device to continue *on* once something else has established that the candidate was at
fault; on its own it cannot make an unattributed HIP abort a decline, for the reason given above.
Attribution at launch has exactly two possible sources: a HIP `classify_failure` that finds a
narrower signal than `HIP_ERROR_INVALID_VALUE` — worth one look at whether the runtime exposes the
aborted-queue status distinctly, but do not plan on it — or moving the failure to compile time,
where the phase already supplies attribution for free. The precedent for the latter
exists: Metal already re-checks the workgroup size against the compiled pipeline state at link time
(metal_backend.ml:1243-1253) on top of the pre-compile check. The analogue:

- add `max_thread_scratch_bytes : int option` to `Backend_intf.hardware_limits` (backend_intf.ml:58),
  same `None = no limit` convention as its siblings;
- at link, query the compiled kernel's static scratch usage (`cuFuncGetAttribute` local-size bytes,
  the HIP equivalent) and raise on the same footing as the threads-per-workgroup check, so rule 3's
  compile arm turns it into `Resource_exceeded Thread_scratch` with damage nothing.

If the OCaml bindings do not expose those queries, extending the bindings is the work. This is the
design's centre of gravity: **turn failures whose damage exceeds nothing into failures that happen
before launch.** It is also separable, and should probably land first — it fixes #533 on its own,
whereas the classification without it only improves the message.

Two notes on how far the precedent actually reaches. Metal's link-time check is the *only* pipeline
query the backend makes: shared memory is checked device-wide and ahead of time, from
`limits.max_workgroup_memory_bytes` (metal_backend.ml:347) inside `check_hardware_limits`, against
the schedule's declared usage rather than the compiled kernel's. The binding for the per-kernel
answer already exists and is simply unused (`Metal.ComputePipelineState.get_static_threadgroup_memory_length`),
so the same post-link pattern closes a second gap for free. And on this machine cudajit is not
installed — `cuda_backend_impl.missing.ml` is selected — so the CUDA arm of both the query and
`classify_failure` can be written but not exercised here; it needs a session on CUDA hardware, as
the HIP arm needs one on minix.

## Fatal must still emit the report

The issue's complaint is that uncontained failures *delete measurements*. A `Fatal` that unwinds
out of `tune` today takes the whole report with it, including the candidates that did time
successfully. Since `report` is already an optional callback (autotune.ml:3277), the fix is one
`Exn.protect`: emit the partial report — timed count, decline census, best-so-far — before
re-raising. A run that dies at candidate 40 of 60 should still tell you what the first 39 found.

## What the sites become

- **`compile_candidate`** (autotune.ml:2719): `with exn -> Error (Exn.to_string exn)` becomes
  `contain ~phase:Compile`, returning `(compiled, rejection) Result.t` — the lossy collapse to one
  string is where the census data is thrown away today. Behaviour otherwise unchanged, except that
  deny-list exceptions now propagate.
- **`try_spec`'s timing arm** (autotune.ml:3406): `contain ~phase:Launch`. On a `Rejected` outcome,
  `device_health` before the next candidate decides whether the search has a device left; on a
  `Fatal` one it propagates regardless of what the probe would have said.
- **Cache replay** (autotune.ml:3325): `contain ~phase:Compile ~untrusted:true`, i.e. rule 1's
  narrowed deny-list; behaviour unchanged, policy now explicit.
- **`compile_advisory`** (autotune.ml:2966) and `model_default`'s bare `exception _` arms: keep the
  fallback for `Rejected`, propagate `Fatal` immediately instead of paying for a recompile that will
  fail the same way. `fallback_if` narrows to its real job.
- **`validate_segments`** (autotune.ml:2963) can be **retired**. It exists because `validate_parallel`
  runs past the seam and the seam's handler could not see it; with the compile phase classifying
  that failure as `Rejected`, `compile_advisory`'s backstop reaches the same fallback without
  duplicating the check. Deleting it is the concrete simplification that shows the abstraction pays
  for itself. (Keep it if profiling shows the eager check meaningfully cheaper than a failed
  compile — but say so, rather than leaving both.)
- **`Context.auto`** (context.ml:103): `try … with _ -> try_backends rest` picks the next backend on
  *any* failure, including `Out_of_memory` and a user's Ctrl-C. Same deny-list applies; unrelated to
  tuning, same bug.

One `contain` with a phase parameter, one classifier, two backend hooks.

## Consumers

`Autotune.report` gains `declines : (rejection_key * int) list` alongside `candidates_failed` —
#521's hand-assembled census becomes a returned value, and gh-479's "did the tensorized candidate
lose, or never run?" is answered from the same fold. Exactly one consumer needs updating today
(benchmarks/runners/ocannl/bench_mlp.ml:225). `C_syntax.mma_census` stays: *declined rendering* and
*declined candidate* are different events, and telling them apart is the point of that census.
`autotune_log` keeps its per-candidate line; the census is what a benchmark runner tabulates
without parsing prose.

## Migration and compatibility

- **Unclassified failures stay containable by default.** Rule 3 makes an unrecognized exception at a
  compile boundary an `Unclassified` rejection rather than a fatal, so no currently-surviving search
  starts crashing on that account. The tightenings are rules 1 and 3-launch, both deliberate.
- **A ratchet, not a flag day.** `Unclassified` counts show up in the census; a config key
  `strict_failure_classification=false` flips them to `Fatal` for CI and development, so the
  taxonomy's gaps stay visible and get closed one at a time. (Named for the classifier rather than
  `autotune_*`, because `Context.compile` and `model_default` share it.)
- **User-facing behaviour is unchanged**: a hand-written schedule that violates a precondition still
  raises out of `Context.compile`, with a message built from the same `detail`. Nine test files
  pattern-match or print `Utils.User_error` as the public contract (test/operations/schedule_ops.ml,
  buffer_aliasing.ml, test_slice_alias.ml, context_copy.ml, merge_buffer_static_verification.ml,
  test_threefry_precision.ml, test_bounds_folded_gather.ml, symbolic_extent_launch.ml,
  test/einsum/test_symbolic_extents.ml); they keep passing.
- One new config key ⇒ the three-place update (`ocannl_config.reference`, `Utils.known_config_keys`,
  the consistency test's scan list). Adding a `hardware_limits` field touches its `deriving
  sexp, compare, equal` users — check whether it reaches the schedule-cache key before assuming
  existing caches survive.

### Test plan

Tests pinning the boundary in *both* directions — the second matters as much as the first, since
the failure this design most wants to prevent is a silently degraded tuning result.

Two run on `cc`, so CI enforces the boundary itself on every machine:

- **Contained**: seed a candidate through `tune`'s transform seam that violates an op precondition
  (`Schedule.apply` raises `Invalid_argument`, device-independent). Assert the search completes, the
  report counts one decline, and its key is `Illegal_schedule`.
- **Not contained**: a transform raising `Assert_failure` makes `tune` propagate rather than log a
  `FAILED` line and crown a different candidate — *and* the partial report is emitted before it
  propagates.

Plus a unit test of the classifier over synthesized exceptions at each phase, including the cache
replay opt-out and each `rejection` constructor. This is where `Resource_exceeded` gets its
portable coverage: the classifier is a pure function, so the phase-to-verdict mapping is testable
on `cc` with a synthesized `check_hardware_limits` failure, no device limit required.

One test is genuinely GPU-gated and must be marked as such rather than counted as CI coverage:

- **Contained, resource, end to end**: on a backend actually reporting `max_threads_per_workgroup`,
  an over-budget workgroup declines with key `Resource_exceeded Workgroup_threads`. `cc` cannot
  stand in — `cpu_mma_limits` builds on `no_hardware_limits` (schedulers.ml:20-27), so
  `max_threads_per_workgroup` is `None` and `check_hardware_limits` has nothing to reject on;
  hardware-axis annotations render serially there by design. Metal locally; the other GPU backends
  in later sessions on their machines.

Injecting synthetic limits to make that last one portable is possible but not worth it: it would
need a seam into `hardware_limits`, which backends supply, and what it would then test is the
classifier — already covered above, without the seam. The GPU test's value is confirming that the
*real* rejection path lands in the right bucket, and only real hardware limits do that.

Placement: `test/operations/autotune_containment.ml` with an `.expected` file, alongside the
existing `autotune_smoke` family.

## Non-goals

- Making every failure recoverable. Device-scope damage ends the run; the deliverable there is
  attribution in the error message, not survival.
- Retrying candidates. A rejection is final for that candidate on that device.
- **Hangs.** #532 — the middle-end wedge that cost the same HIP cells — raises nothing, so no
  classification reaches it. Containing a hang needs a timeout, a different mechanism, and its own
  decision about what a partially-timed candidate means. Explicitly out of scope here.
- Classifying *wrong values*. A candidate that compiles, runs, and computes garbage belongs to
  gh-ocannl-484's numerics pinning, not to containment.

## Suggested landing order

0. Establish where #533's exception actually escaped (HIP machine, one instrumented run). If it was
   the baseline timing, step 1 is the entire fix and steps 2-4 are about the census and the deny-list.
1. `max_thread_scratch_bytes` + the link-time query (fixes #533 on its own, no new vocabulary), plus
   Metal's unused static-threadgroup-memory query while the pattern is open.
2. `schedule_outcome.ml`, `contain`, rule 1, the partial report on `Fatal`.
3. The census in `Autotune.report`, `bench_mlp` consumer, retire `validate_segments`.
4. `classify_failure` / `device_health` per backend, rule 3's launch arm, the `Unclassified` ratchet.

Steps 1 and 4 are the only ones needing hardware this machine does not have (CUDA and HIP arms
unexercisable here; Metal and cc cover the rest). Steps 2 and 3 are portable and carry the tests.
