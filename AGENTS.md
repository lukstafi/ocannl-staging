# OCANNL Agent Guide

This file guides coding agents working in this repository. It is the single authoritative copy:
Claude Code loads it through CLAUDE.md's `@AGENTS.md` include, other agents read it directly, and
edits go here — CLAUDE.md carries only the include and notes that apply to Claude Code alone
(gh-ocannl-653).

## Project Overview

OCANNL (OCaml Compiles Algorithms for Neural Networks Learning) is a from-scratch compiled Deep Learning framework with an optimizing compiler. The project consists of two main packages:

- `arrayjit`: The low-level optimizing compiler with multiple backends (CPU, CUDA, Metal)
- `neural_nets_lib`: The high-level deep learning framework with syntax extensions, shape inference, and backpropagation

## Structure and Ownership

- `lib/`: user-facing recipes (training utilities, nn blocks, re-exports).
- `tensor/`: core framework internals (Tensor, Shape, Operation, ppx_%op/%cd).
- `arrayjit/`: compiler + backends (`assignments.ml`, `low_level.ml`, `indexing.ml`, `schedule.ml`, `context.ml`, and the `*_backend.ml` implementations).
- `bin/`: runnable benchmarks and demos.
- `test/`: tutorials and tests (ppx_expect and standalone `.expected` tests).
- `docs/`: slides and reference docs.
- `build_files/` and `log_files/`: generated artifacts when debug settings are enabled.

Key reference files:

- `docs/syntax_extensions.md` (authoritative for %op/%cd)
- `docs/shape_inference.md` (shape/projection inference pipeline)
- `arrayjit/lib/context.mli` (context-based runtime API)
- `ocannl_config.reference` (all configuration keys and defaults)
- `docs/agent-notes.md` (index) and `docs/agent-notes/` (distilled cross-session agent knowledge)

## Build Commands

The project uses Dune for building and testing:

```bash
# Build all packages; this triggers running executables for cram-style tests
dune build

# Only compile -- do not run any executable
dune build @check

# Build specific package
dune build -p neural_nets_lib
dune build -p arrayjit

# Run tests
dune runtest

# Run tests for a specific backend (bash syntax)
OCANNL_BACKEND=cuda dune runtest

# Install dependencies (OCaml >= 5.3)
opam install . --deps-only

# Install with optional backends  
opam install cudajit  # for CUDA backend
opam install hipjit   # for AMD HIP backend
```

**Worktrees**: nested ones (`.claude/worktrees/`, the Claude Code default) need a `dune-workspace` at their root, or dune builds the PARENT checkout instead; the SessionStart hook writes it, so after a mid-session worktree switch run `scripts/setup-ocaml-env.sh` by hand. See docs/agent-notes/build-and-test.md for why.

**Windows shells**: `opam env --shell=sh` emits cygwin-style paths that break under Git Bash (MSYS), so a Git Bash session without a primed environment gets a half-working toolchain (dune found but linking fails with `cygpath: error converting ... -lpthread`). Source `tools/opam-env.sh` first — it rewrites the paths for MSYS and works from any POSIX shell. On Windows, link steps also flood stderr with benign binutils warnings (`Warning: corrupt .drectve at end of def file`, from MSVC-produced import libraries like ROCm's/CUDA's); `tools/dune-quiet.sh <dune args>` runs dune with exactly those lines filtered, preserving the exit status. Use **Git Bash** specifically, not a Cygwin bash (opam's, or whatever `bash` resolves to once opam's cygwin is on PATH): the two are told apart by `uname -o` (`Msys` vs `Cygwin`, since both bashes report `OSTYPE=cygwin`), only the MSYS one gets `opam-env.sh`'s path rewrite, and Cygwin ships no `perl` — which `tools/test-run.sh` needs for its lock, cap and `last` pointer, and refuses without (gh-ocannl-662).

**New ppx-expectation files** (`test/ppx/*_expected.ml`, compared against pretty-printed ppx output) must stay unformatted — add them to `.ocamlformat-ignore`, or the unattended formatting sweep fails to converge and aborts.

## Architecture Overview

**Before working on a subsystem, read the matching file under `docs/agent-notes/`** (the index is
`docs/agent-notes.md`) — distilled
cross-session agent knowledge (solver/backend traps, known bugs with workarounds, debug recipes,
design history) that is not derivable from the code alone.

### Key Concepts

1. **Dual Syntax Extensions**:
   - `%cd` ("code"): For assignment computations (`Assignments.comp`)
   - `%op` ("operation"): For tensor expressions (`Tensor.t`)
   - Inline declarations lift to unit parameter `()` scope, enabling parameter reuse

2. **Shape Inference**: 
   - Three axis kinds: batch | input -> output (matrix convention: input rightmost)
   - Row variables (`..d..`) enable flexible axis handling and broadcasting
   - Einsum notation supports convolutions, reductions, and arbitrary permutations
   - "Principle of least commitment": use row variables where axis count doesn't matter
   - Shape inference completion is forced by lowering: via `Context.compile`, or wrappers such as `Train.to_routine`, `Train.run_once` or `Train.forward_once`; `finish_inference` closes still-unsolved dims (GLB where known, otherwise 1/broadcast)
   - Projection inference is re-derived per operation (`derive_projections`, fresh projection ids) to avoid cross-op contamination
   - Operations in `Operation`, `TDSL`, `NTDSL` return functions with `Tensor.op_fun` type, so that shapes can be specified at call sites if needed
   -  Operations in `TDSL.O` (opened for `%op`), `NTDSL.O` (opened for `%cd`) hide this so that shapes have to be inferred
   
3. **Backend Architecture**: Unified interface supporting CPU (multicore), CUDA, HIP, and Metal backends

4. **Memory Management**: Tensor node memory modes are `Virtual`, `Local`, and `On_device`.
   CPU-side reads and writes are explicit, context-mediated operations
   (`Context.to_host`/`from_host`, `get_values`/`set_values`).

## Development Workflow

### Testing

- Tests are implemented either as inline expectations using `ppx_expect`; or as cram-style tests using Dune's `test` stanza where an `.ml` file is compiled, executed, and its output compared against an `.expected` file
- The two approaches are exclusive: a test using using `.expected` file target cannot also use `%expect` inline expectations
- `.expected` tests, i.e. using the `test` stanza, are easier to debug, use them for testing new features
- Tutorial files, i.e. `%expect` tests, in `test/` serve as both documentation and integration tests, should only be used when the outputs are illustrative

**Running Tests**:
- `dune runtest` - runs all tests including inline tests and cram-style tests, EXCEPT the training integration runs (`@train`) and the slow training runs (`@slow`), which have their own aliases (see below)
- `dune runtest test/operations/` - runs all tests in operations directory
- Avoid `dune exec test/.../test_name.exe` for standalone tests: `dune exec` keeps the invocation cwd, the config search walks UP from cwd (`Utils.config_file_args`), and the root `ocannl_config` is deliberately gitignored (personal dev settings) — so fresh clones, CI, and worktrees find no config and the test dies partway, or `Context.auto` silently picks metal→cuda→cc (a "cc" run quietly executes on the GPU). Tests only find their config under `dune runtest`/build rules because dune sets their cwd to `_build/default/<test dir>` where `(copy_files ../config/ocannl_config)` materialized it
- Instead, run the one test by its own alias: `dune build @test/operations/runtest-<name>`. It runs the exe with dune's cwd AND applies the `.expected` diff, so a changed output fails the build instead of sitting unread in a file, and `dune promote` accepts it; a misspelled name exits 1 ("Alias ... is empty"), not a silent no-op. The run still leaves `_build/default/<dir>/<name>.exe.output` to inspect
- A test written as an `(executable)` plus a `(rule)` that diffs a golden — the scanning tests, the codegen snapshot rules, the config-precedence rules, the ppx-output diffs — gets no alias from dune, so each such rule carries `(alias runtest-<name>)` of its own (gh-ocannl-726) and `dune build @test/operations/runtest-verdict_ratchet` runs that one test and applies its diff, promotable, with a misspelling exiting 1. Three rules govern writing one, all three checked by `test/operations/env_var_deps`: the rule names that alias and NOT `runtest` (a rule on two aliases makes building either build both, so naming both would put the whole directory behind the per-test alias); an `(alias (name runtest) (deps (alias runtest-<name>) …))` stanza per directory aggregates them back, or plain `dune runtest` skips them silently; and `<name>` begins with the golden the rule checks — checked literally, since a reader reaches for the alias with the failing golden in hand — and is never a `(test)` stanza's name in that directory, since dune generates `runtest-<name>` for those; where one run writes several goldens the alias says which (`runtest-top_down_prec-extension`, `runtest-top_down_prec-unoptimized`). The rule also depends on `(alias runtest-env_spelling_gate)`, which is how the ambient gate reaches a per-test alias
- The repo-wide scans — the rules that parse the repository itself: config consistency and classification, env-var deps, verdict ratchet, codegen-text inventory, agent-note structure, shell-script parsing, and their kin (the authoritative list is the `(name scans)` stanza in `test/operations/dune`) — share the family alias `dune build @test/operations/scans`, which runs them all in seconds without the directory suite (gh-ocannl-703). They are the checks a small targeted change trips, so run the family before pushing a change to a config key, a dune stanza, a printed claim, an agent note, or a new script or source file a scan's glob would pick up (a new shell script, say, joins the `shell_scripts_parse` golden). A scan the family stanza omits fails `env_var_deps`, which derives the family from the rules that glob the repository rather than from a second copy of the list — which is also why this bullet describes the family instead of enumerating it
- Two focused aggregates sit beside `scans`, built the same way (gh-ocannl-783): `dune build @metal-codegen` runs the Metal-pinned tests — the executed Metal-only guards and the emitted-MSL structural ones — and `dune build @lifecycle` runs the resource-lifecycle probes, the tests that drive `Ir.Resource_fault_injection` or read `Ir.Alloc_census`. Each is spelled identically in `test/operations/dune` and `arrayjit/test/dune`, so the root-level `@<family>` runs both halves. Membership is derived and not written down twice — the backend marker for Metal, the modules' use of the instrumentation for lifecycle — so a member the family stanza omits fails `env_var_deps`; the derivation is a floor, and a family may list more (`arrayjit/test`'s `test_slab_free_on_grow`). A new family owes the same: a derivation from something the member stanza declares for an independent reason, a dependency on `(alias runtest-env_spelling_gate)` since a family alias is a build entry point, and an arm in the `env_var_deps --control` tree. `docs/agent-notes/build-and-test.md` has the shapes
- Dune's OWN `runtest-<name>` (per `(test)`/`(tests)` name and per inline-test library) needs **dune >= 3.20**, which is the project's declared floor (`dune-project`, so the generated opam files too): the targeted-test workflow and the focused aggregate families are built on those aliases, so promising 3.18 promised a toolchain on which they reach nothing. The hand-written aliases above are ordinary rule fields and work on any dune. For a target with no alias at all, build it instead: `dune build test/operations/<name>.exe.output` (or `<name>.actual`), inspect it under `_build/default/<dir>/`, then `dune runtest <dir>` to register the diff and promote. For `bin/` executables (same cwd trap), pin `OCANNL_BACKEND=...` explicitly
- Pinning a variable on such a probe (`OCANNL_BACKEND=cuda dune build @test/operations/runtest-<name>`) reaches the run only for variables the stanza DECLARES: dune tracks no environment variable it was not told about, so an undeclared one leaves the target up to date and the previous run's result in place. Every test stanza that can select a backend declares `(env_var OCANNL_BACKEND)` — one spelling, uppercase, since gh-ocannl-652 dropped the lowercase `ocannl_<key>` the environment used to be read under. A stanza that names its backend, or links none, declares nothing and instead carries a marker comment INSIDE its own parentheses saying which and why: `; ocannl-backend: none -- links arrayjit.ir alone and calls only pure arithmetic`, `; ocannl-backend: metal -- pins MSL emission`. Exactly one of the two, over every stanza that runs an executable — neither is the hole gh-ocannl-659 closed (dune then serves the previous backend's output as a pass), and both is contradictory intent. The words are `none`, `cc`, `multidev_cc`, `cuda`, `hip`, `metal`, comma-separated where a stanza honestly names two; the reason is required and must be more than one word. For an `(executable)` plus a `(rule)`, the marker goes on the rule — same placement as the `ocannl_config` dep. For any other key add `OCANNL_<KEY>` to that stanza's `(deps ...)` first — `test/operations/env_var_deps` checks all of this, that every `(env_var ...)` addressed to OCANNL names a spelling a run reads, and that a stanza whose modules call `Test_utils.Generated.init` declares `(env_var OCANNL_BUILD_FILES_PREFIX)` where dune runs it — in its own `(deps ...)`, or, for an `(executable)`, in the rule that runs it — with a declaration nothing calls for reported too (gh-ocannl-628, gh-ocannl-659, gh-ocannl-723). A test that GUARDS on the ambient environment — refusing to run when a variable that would rewrite its golden is set, which is how `startup_streams`, `profile_precedence` and `config_profiles` protect theirs — owes the same declaration for every key its guard names: the guard only runs when dune reruns the rule, so an undeclared key is one it never sees, and `env_var_deps` derives the keys from the source and pairs them with the deps rather than trusting the two lists to agree (gh-ocannl-749; a variable the rule pins with `(setenv …)` is exempt, and pinning is the better option where it is available). Setting the dropped lowercase form is a fatal startup error, not a silent no-op
- `env_var_deps` also checks that each library declares the `OCANNL_LOG_LEVEL_<MODULE>` tracing gates its modules read (gh-ocannl-628), and that every alias with test actions in a directory carries that directory's `env_spelling_gate` — an action depending on `(universe)`, so it reruns every invocation and no suite can come back green with a rejected lowercase spelling (`ocannl_backend=…`) ambient. `runtest` and `slow` are gated separately, and a gate in a file that serializes on `ocannl_training_test` must take the lock (the gate starts no pool; an unlocked action in a locked file is what the next training test gets copied from). Hand-written per-test aliases depend on the gate explicitly — `(deps (alias runtest-env_spelling_gate) …)` — while dune's GENERATED `runtest-<name>` alias for a `(test)` stanza does not, so such a targeted run can still be served stale under a rejected spelling
- Config startup chatter (the welcome message, the `log_config_sourcing` trace, the profile banner) goes to stderr, so an OCANNL-linked executable's stdout stays a clean data channel and `.expected` goldens (dune captures stdout) never see it. The sourcing trace is off by default; pass `--ocannl_log_config_sourcing=true` (or set it in the config file) to see where each setting a run reads comes from
- Confirming WHICH backend a probe ran on cannot come from a backend-uniform golden: a test that announces `SKIPPED on <backend>` writes that to stderr and still prints `<claim>: true` on stdout, precisely so the golden stays backend-uniform — so a GPU run's `.exe.output` can be byte-identical to the cc one. Read the run's stderr, or have the test print the backend (gh-ocannl-622)
- When probing that way, do NOT discard the build's stderr (`2>/dev/null`) without checking its exit status: a FAILED build (e.g. a warning-as-error from a temporary edit) leaves the previous `_build/default/<dir>/<name>.exe.output` untouched, so the stale file reads as a green probe — this turned a negative control into a false positive during gh-554
- `(copy_files ../config/ocannl_config)` only materializes the config for rules that DEPEND on it: every `(test)`/`(tests)` stanza and every `(library)` with `(inline_tests)` must list `ocannl_config` in its `(deps ...)`, and so must every `(rule ...)` that runs a test executable (an `(executable)` stanza has no `deps` field, so its companion rule is where the dep goes). Nothing is sandboxed, so a stanza missing it does not fail on its own — it reads whatever is in `_build/default/<test dir>` when it happens to run, which makes both the `.exe.output` probe and the test itself order-dependent (gh-ocannl-586). "Pure" tests are no exception: without the dep they also get default `print_decimals_precision`/`fixed_state_for_init` and a config-sourcing trace on stderr. `test/operations/config_dep_completeness` checks this over every `dune` file in the repository, including new directories, so an omission fails the suite rather than drifting (gh-ocannl-597); an executable that no rule runs is not a site, and a rule running something that reads no configuration goes on that test's named exemption list
- A `(test)` stanza automatically diffs a `<name>.expected` sitting next to it against the test's output on `dune runtest`; no extra stanza is needed to "assert" the golden. The explicit `rule` + `diff` pattern (as in `test_config_consistency`) is only for tests not run by a `(test)` stanza, such as the `@slow` rules
- Run suites through `tools/test-run.sh` rather than hand-rolling shell around dune: `tools/test-run.sh run runtest test/operations`, `tools/test-run.sh run build @slow` (arguments after the options are dune's argv; default `runtest`). It runs dune unpiped (piping dune to anything reports the pipe's status, so a promotion diff reads as green), caps wall-clock time (default 3600s, `--cap N`, raise for `@slow`), logs to a file ending in an `exit: N` sentinel, prints a compact digest (verdict, promotion diffs, error fingerprint), and exits with dune's status. The script header documents the other shell traps it absorbs
- From PowerShell, QUOTE dune alias targets: PowerShell parses an unquoted `@word` argument as a splatting variable (undefined here → expands to nothing), so `dune build @runtest @slow` silently degrades to a plain `dune build` that exits 0 having run no tests — a false green. Use `dune build "@runtest" "@slow"` (bash/cmd are unaffected)
- Inline tests (like those in `test_threefry4x32.ml`) are part of library modules and run via `dune runtest`, not `dune exec`
- Scope test runs to the code and configuration paths a change can reach. Start with affected directory aliases such as `dune build @test/operations/runtest`, `@test/einsum/runtest`, `@test/ppx/runtest`, `@arrayjit/runtest`, or `@test/training/runtest`, then run `dune build @check`. Before broad testing of a config-gated path, grep whether any test or test config enables the gate; reserve the full regular/slow suites for cross-cutting changes
- Never write a sleep/`pgrep` waiter loop around a test run: `pgrep` waiters match the editor's immortal `dune ocaml-merlin` daemons (or the waiter itself) and spin forever, stranding shells. For a long run, launch `tools/test-run.sh run ...` through your harness's background execution (e.g. Claude Code's Bash background mode) and act on its completion notification — no polling needed. Only for a run that must outlive the session, use `tools/test-run.sh start ...`, then the non-blocking `status last` or the hard-bounded `wait last`; both read the run's recorded verdict file rather than probing processes
- Keep library sources unchanged while Dune is running; edits invalidate in-flight rules and can repeat expensive work

**Training integration runs (the `train` alias)**:
- The toy training integrations (`mlp_names`, `mlp_bn_names`, `circles_conv`, `fsm_transformer`, `transformer_names`) live on the `train` alias, off the `runtest` path: they serialize on the training lock, so in the default suite they set every full run's wall-clock tail (tens to hundreds of seconds each). They are NOT slow tests — small by intent, unlike `@slow`'s real-dataset runs — just a separate entry point. Run them with `dune build @train`, one at a time via `@test/training/train-<name>`, or everything via `dune build @runtest @train`; the daily sweep's full-suite units cover them on every backend. The alias follows the same shape as `slow` below (per-test `train-<name>` aliases, a `train-env_spelling_gate`, an aggregate stanza, all checked by `env_var_deps`)
- Per-PR CI executes `@train` as a macOS-only shard (ubuntu would run the serialized chain slower than its whole main job, so it covers the tier through the daily sweep instead — the reasoning and measurements live in `ci.yml`); after touching training dynamics, `Train.*` plumbing, or the autotuner's fission path, still run the affected members locally before pushing

**Slow training tests (the `slow` alias)**:
- Explicit `@slow` training tests are excluded from `dune runtest`; run them only when relevant.
- Regular and `@slow` training actions share the `ocannl_training_test` Dune lock, so they do not run their process-local OpenMP pools concurrently with one another; compilation remains parallel. Preserve the lock on new regular training tests and `@slow` rules. Other pool-using tests, notably `test/operations/cpu_parallel`, intentionally remain unlocked as concurrency stress.
- They are still ordinary executables: `dune build @check` compiles them, so they cannot bit-rot.
- Run them on demand with `dune build @slow` (uses their `.expected` files; `dune promote` to accept changes). Use `dune build @runtest @slow` to run the entire suite.
- Each slow rule also sits on its own `slow-<name>` alias, so ONE slow test reruns alone after a change: `OCANNL_BACKEND=cc dune build @test/training/slow-cifar_conv` runs that directory's `env_spelling_gate` and then just `cifar_conv`, applying its `.expected` diff (promotable with `dune promote`); a misspelled name exits 1 (`Alias ... is empty`) rather than silently running nothing. `@slow` is per directory an `(alias (name slow) (deps (alias slow-env_spelling_gate) (alias slow-<name>) ...))` stanza aggregating them, and `test/operations/env_var_deps` fails on a `slow-<name>` rule that stanza does not list, since `dune build @slow` would otherwise skip it silently
- To gate a new slow test, in its `test/.../dune` replace its `(test ...)` stanza with an `(executable ...)` plus a `(rule (alias slow-<name>) ...)` that runs the exe and diffs against `<name>.expected`, with `(alias slow-env_spelling_gate)` first in its `(deps ...)` so the ambient gate runs before the test, and add `(alias slow-<name>)` to the directory's `(alias (name slow) ...)` stanza (see `test/training/dune` for the pattern). Wrap the rule's action in `(no-infer ...)` so the `<name>.actual` output is NOT registered as a build target — otherwise a plain `dune build` (the `@all` alias builds every file target) would run the slow exe anyway.

**Test Types**:
- **Inline tests**: Files included in library `modules` field with `inline_tests` stanza (e.g., `test_threefry4x32.ml` in `operations_tutorials` library)
- **Standalone tests**: Files with dedicated `test` stanza and corresponding `.expected` files (e.g., `threefry4x32_demo`)
- Use `dune promote` to accept test output changes (on Windows prefer `tools/promote.sh` — it strips CRLF from promoted goldens). **During a merge use `tools/promote.sh` on every platform**: promotion writes the working tree while `git commit` mid-merge takes the INDEX, so a golden promoted after its `git add` is committed with the pre-promotion content — every local run reads the working tree and passes, and CI builds the committed tree and fails on the golden diff. The script stages what it promoted and says so (warning, rather than staging, for a golden still unmerged); outside a merge it does nothing extra. `tools/test-promote.sh` is its hand-run harness
- **A test that decides its own verdict must report it through `Verdict`** (`test/support/verdict.ml`; add `verdict` to the stanza's `(libraries ...)`): `Verdict.p "claim" b` prints `claim: b`, `Verdict.pass_fail "label" b ~detail` prints a `label: PASS`/`FAIL` column with the machine-specific number only on failure, `Verdict.fail "what is wrong"` reports a one-off, `Verdict.skipped ~backend "claim"` prints the passing golden line for a leg this backend cannot evaluate and announces it on stderr. **A claim quantified over a collection goes through `Verdict.p_all "every seed spreads j" seeds ~f` (or `p_none`, `p_exists`, `p_empty "claim" ~over:population derived`), never through `p` applied to a `List.for_all`**: "every X …" is true of an empty X, and its golden line is byte-identical to one a real population passed (gh-ocannl-729). These print exactly what `p` prints on a non-empty collection and a distinct `<claim> (empty): false` on an empty one; `?min:n` raises the floor and reports `<claim> (only 1 of 4): false`. Arrays reach them through `Array.to_list`. Leave the unguarded spelling only where emptiness is the passing case ("no candidate declines"), and say so at the site. A failed check exits the process 1 from a shared teardown and echoes to stderr (dune discards the redirected stdout of a process that exits nonzero). Without that exit status the golden diff is the only gate, and it is promotable: a test that prints `FAIL: …` or `<claim>: false` and exits 0 lets the natural next move — `dune promote` — record the failure as the expected output, and nothing fails again until someone reads the golden (gh-ocannl-601). Two consequences: **phrase every claim so `true` is the passing reading** (rename the fact, `not (…)`, rather than recording a designed `false` — in a golden a blessed regression and a designed negative are the same line), and keep purely descriptive output (values and tables the golden pins) on plain `printf`, which is not an assertion channel. `Ll_test.p` is `Verdict.p`, so hand-built-IR tests get this for free. **The literal-label half of this is enforced**, not merely documented: `test/operations/verdict_ratchet` parses every test source and fails on a format whose one argument-consuming conversion is a bare `%b` at the end behind a literal label (`"fused: %b\n"`), which is the shape a self-decided claim takes (gh-ocannl-668, after a post-sweep test reintroduced four such sites unnoticed). A descriptive `%b` print escapes by carrying any other conversion — which one interpolating what it describes does anyway — or by a named exemption there. Computed labels (`"%s fused: %b\n"`) are out of its reach by design and remain issue #624's manual work
- **A float that a device reduction produced does not belong in a stdout golden at a fixed precision.** Lowering `%.4f` to `%.1f` moves the rounding tie, it does not remove it: `cifar_conv`'s epoch-30 mean read `1.0` on cc and `1.1` on cuda at the same commit, and no promotion served both. Print the exact digits to stderr, tagged `(not part of the golden)`, and put a `Verdict` claim about what the number was showing on stdout — a threshold the trained value clears and an untrained one does not, a fall between the first and last logged epoch, an argmax or a ranking, a closed-form value within a tolerance. The claim must fail on a wrong number, so "is finite" alone is not one — and it must be TWO-SIDED wherever the digits were: an upper bound on a loss admits the finite negative value a dropped negation produces, and a claim about ranking or ordering admits a badly distorted magnitude, so pair it with a validity range. Mark every relocated line `(not part of the golden)`. Floats that are exact by construction (threshold constants, power-of-two loss scales, host-side closed-form schedules, small dyadic sums) stay on stdout. Nothing lints this; the audited `test/training`/`test/gpt2` sites and the classification rule are in `docs/agent-notes/build-and-test.md` (gh-ocannl-725)
- A few `%expect_test` blocks capture exception backtraces that hard-code `file:line` (e.g. `test/operations/primitive_ops.ml` embeds `context.ml` line numbers). Any edit that shifts lines in such a file forces a line-number-only re-promote — benign noise, not a behavior change; promote it in the same shell as the failing run (the `.ml.corrected` can vanish between runs)
- For optimizer passes that change *what value a cell holds* (virtualization guards, index solving, accumulation/init elision), a structural test on the emitted op tree is necessary but NOT sufficient — also assert on executed output vs. a materialized/reference run (`Context.compile`/`run`/`get_values`); a pass has shipped green structural checks while computing all zeros
- The executed reference must DISCRIMINATE, not merely exist: every producer should write a value that varies with *every* symbol of its iteration and stays clear of the init/sentinel value (`1 + i`, `1 + 10*outer + inner` — see the `tick`/`tag` helpers in `test/operations/virtual_diagonal.ml`). A constant producer replays an identical assignment under a too-wide range guard, a value omitting one symbol is constant along that axis under a wrong substitution, and a value colliding with the zero-init hides a dropped first iteration — the leg passes for the wrong reason
- **A test that asserts on generated code reads it through `Test_utils.Generated`** (`test/support/generated.ml`; add `test_utils` to the stanza's `(libraries ...)`), never by opening `build_files/<routine>.<ext>` itself. Artifacts outlive the run that wrote them (`clean_up_build_files_on_startup=false` in the test config) and a second compile under the same routine name overwrites the first's, so a read that does not establish provenance can keep asserting on a kernel that is no longer emitted. Call `Generated.init ~backend_name` once, before the first compile (such a test must leave `build_files_prefix` at its default: a configured prefix names a directory the process does not own, so provenance cannot be established there and `init` refuses it), and declare `(env_var OCANNL_BUILD_FILES_PREFIX)` on the stanza dune runs it under, since `init` reads that key to decide whether the directory is this run's to empty — `env_var_deps` requires the declaration of every stanza whose modules call it (gh-ocannl-723); then `Generated.assert_emits ~routine ~contains "claim"`, or `Generated.read routine` for a check that needs the source. A missing artifact fails the run rather than answering `None`, so a leg this backend cannot evaluate must be gated and reported with `Verdict.skipped` instead of reaching the read; a loop compiling several candidates under one routine name calls `Generated.arm routine` before each compile, which is what makes each candidate judged by its own kernel
- **Pin the relationship, not the restatement**: where a check needs a set another part of the system owns (the backends, the config keys, a scan's own rule names), derive it or assert the two equal from where the link cost is already paid — never write the set down again and assert that the copy still says what it says, which is a test that cannot fail. Compare as sorted lists of identities, claim one bare boolean through `Verdict` with both lists on stderr, and control it on a synthesized violation. Judgment lists (`digest_completeness`'s `codegen_stage_modules`) and deliberately independent constants (`Config_key_scan.scan_root_floors`) are not members; leave them written down with their reason. `docs/agent-notes/build-and-test.md` has the shapes and the exemplars
- Backend codegen snapshots (e.g. `.cu.expected` files, `test_cuda_pool_offset.expected`) go stale when codegen changes land without that backend's hardware available to re-record them — expect to re-promote such snapshots when the hardware next runs the suite
- Before changing code generation, run `dune build @test/operations/runtest-codegen_text_inventory`: its golden enumerates every file that pins the TEXT of emitted code — codegen goldens in BOTH `test/` and `arrayjit/test/`, and the test sources that assert on emitted text from a string literal, which no `.expected` scan can see and which fail a plain `dune build` because they are `Verdict` claims (gh-ocannl-712; see docs/agent-notes/build-and-test.md). The set of values that hand a test generated text in memory is DERIVED from the compiler libraries' compiled interfaces (gh-ocannl-748), so a renderer added to a library needs no list updating — but reach one through an `open` rather than a qualifier and the scan refuses the file by name, since an unqualified call is invisible to it
- **Test Placement Guidelines**:
  * Always add tests under one of the test subdirectories
  * Default location is `test/operations`
  * Use `test/einsum` for tests involving complex einsum specifications
  * Use `test/training` for tests involving training loops
  * When adding a test, update the corresponding test stanza — including `ocannl_config` in its `(deps ...)`, plus `(env_var OCANNL_BACKEND)` if the test picks a backend rather than naming one, or a `; ocannl-backend: <word> -- <reason>` marker comment inside the stanza if it does not (gh-ocannl-659; one or the other, never neither)
  * For standalone tests, add an `.expected` file for test results (can initially be empty)
  * Tests that build `Ir.Low_level.t` by hand share the `ll_test` library (`test/support/ll_test.ml`): node/index/statement builders, one exhaustive IR traversal with the structural counters derived from it, and the `?prelowered` optimize/run/execute harness. Add `ll_test` to `(libraries ...)` instead of re-deriving the machinery. It links `ocannl`, which is why it sits beside `test_utils` rather than inside it — `test_utils` stays on `arrayjit.ir` alone, for tests that link no more than that
  * Tests that enable `output_debug_files_in_build_directory` in one directory all execute from the same `_build` directory, and dune runs them concurrently — but each process writes to its own `build_files/<exe-name>/` subdirectory (and cc compiles from a private temp copy), so cross-test clashes are structurally prevented; only `build_files_prefix=.` (flat legacy layout) shared across processes can still race. Routine-name uniqueness matters WITHIN one executable: a same-named routine silently overwrites the earlier one's debug artifacts (`.cd`/`.ll`/`.c`), so a later reader or a dune snapshot-copy rule sees the survivor. Test-unique name prefixes (the existing `af_`, `ops_`, `smem_` conventions) remain useful for grepping and for keeping within-exe names distinct

**Windows portability for `.expected` tests**:
- `dune promote` on Windows writes CRLF line endings into `.expected` files (the test exe's stdout is text-mode). `.gitattributes` pins `*.expected` (and `test/ppx/*_expected.ml`) to LF, so git normalizes them on commit and promote-introduced CRs stay out of diffs; promoting via `tools/promote.sh` strips them from the working tree too. PowerShell `Set-Content`/`Out-File` also write CRLF — edit `.expected` files with bash tools instead
- The Windows C runtime prints 3-digit float exponents (`e+018` where Linux prints `e+18`) — format floats destined for `.expected` files with `Ir.Ndarray.concise_float ~prec` (normalizes exponents portably) instead of `%g`/`%e`
- The Windows C runtime rounds representable decimal ties away from zero while glibc rounds to even (`%.1f` of `2.25` prints `2.3` on Windows, `2.2` on Linux) — avoid tie values in test data, or print with OCaml's `%h` hex-float format, which sidesteps decimal rounding entirely
- The `test_utils` library (`test/support/test_utils.ml`) packages these rules as portable-by-construction printers (`print_float`/`print_floats`/`hex_float`, plus `set_binary_stdout` for stubs that echo a golden file byte-for-byte) — add `test_utils` to a new test's `(libraries ...)` instead of re-deriving them

**Module Paths and Common APIs**:

- **For files outside OCANNL implementation (tests, examples, user code), always start with `open Ocannl.Operation.DSL_modules`** - this brings all DSL modules into scope (defined near the end of `tensor/operation.ml`)
- Available modules after `open Ocannl.Operation.DSL_modules`:
  - `Ir` - Low-level IR types and operations (Ndarray, Ops, Tnode, etc.)
  - `Row` - Row variables for shape inference
  - `Shape` - Shape inference and einsum notation
  - `Tensor` - Core tensor type and operations
  - `TDSL` - Tensor DSL with automatic differentiation (grad_spec: If_needed)
  - `NTDSL` - No-gradient tensor DSL (grad_spec: Prohibit_grad)
- There is no `PDSL` (Require_grad DSL). To build a differentiable leaf tensor with concrete values (e.g. in tests), pass the grad spec explicitly: `Operation.init ~l ~prec ~b ~o ~f ~grad_spec:Tensor.Require_grad ()` or `Tensor.term_init values ~grad_spec:Require_grad ()` (1-D); see `test/training/fused_classifier.ml`, `test/operations/primitive_ops.ml`
- Precision values: `Ir.Ops.single`, `Ir.Ops.double`, `Ir.Ops.half` (lowercase)
- Tensor printing in expect tests: `Tensor.print ~here:[%here] ~force:false ~with_code:false ~with_grad:false \`Inline tensor`
- For simple test executables, use `(libraries base ocannl stdio)` in dune file

### Pull Requests

- A **PR accomplishes a goal**: one thing that is true about the system afterwards and was not before, stated in its title. Scope it generously — carry the goal to its natural completion (the change, the tests that pin it, the docs it justifies, the follow-on cleanups it exposes) rather than stopping at the smallest reviewable increment and opening three PRs. Velocity and clarity comes from finishing a goal in one pass
- A **commit is one move toward the goal**, not one slice of the work by artifact type. Whatever a move needs — the logic change, the tests that pin it, its `.expected` goldens, the doc or agent-note it justifies — belongs in that one commit. Splitting a change away from its own tests or documentation makes the series harder to read, not easier
- A goal usually takes several moves, so a series of topical commits is the norm — merged with a merge commit that preserves the series
- **Do NOT touch `CHANGES.md` in feature work** (gh-ocannl-807). The changelog is written in editorial passes — at release prep, or an occasional explicitly-requested batch catch-up — from the durable records the work already leaves: merge commits (`git log --first-parent`), PR bodies, and issue closing comments. A per-PR entry duplicates the PR body you just wrote, and the shared `## [Unreleased]` anchor made every concurrent PR conflict there, each collision costing a full CI cycle. When the editorial pass runs: bullets are user-facing (what changed for someone using the library — internal test/tooling plumbing usually earns no bullet), one to three lines each, citing `gh-ocannl-NNN`; the mechanism, rationale, and measured numbers stay in the PR, the issue, and `docs/agent-notes/`
- When you notice unrelated code smells or design problems, file separate issues
- Follow-up fixing commits are fine, and test-expectation promotions that span several topics can land in a final tests/promotions commit
- When creating commits, include the work summary in the commit message and credit yourself as a co-author
- Each commit should at least compile: loop `git checkout <rev> && dune build @check` over `git rev-list --reverse master..HEAD` (interactive rebase is typically unavailable in agent harnesses)
- **Bring the base in before opening the PR and again before merging**: GitHub builds the pull request's MERGE commit, so a check that scans the whole repository can be red on the tree CI builds while green on your branch. Fetch the STAGING remote (resolve which name points there per the next bullet — it need not be `origin`) and rebase onto its `master`, or merge it in where the branch is shared and rewriting is not yours to do — `docs/agent-notes/build-and-test.md` explains the mechanism and how to validate against the merged tree
- **Two repositories, and remote names are not the contract**: development — branches, PRs, and the `master` they land on — happens in `lukstafi/ocannl-staging`, while `ahrefs/ocannl` is the public repo that owns the ISSUES this codebase cites as `gh-ocannl-NNN`, the milestones and the GitHub releases, and receives release-relevant changes. Which remote name points where is local: a clone has whatever names it was given, a clone of the public repo calls IT `origin`, and a fresh clone has no second remote at all. So check `git remote -v` before trusting a name, add the other repo explicitly when you need it (`git remote add upstream https://github.com/ahrefs/ocannl.git`), and pass `--repo <owner>/<name>` to every `gh` command rather than letting it infer: issues to `ahrefs/ocannl`, PRs to `lukstafi/ocannl-staging`

### Configuration

- See `ocannl_config.reference` for documentation of all settings. It ships with every setting COMMENTED OUT (`#key=…`, no space; prose comments use `# `), so copying it verbatim states nothing
- Key configs: backend selection, debug logging, optimization levels
- **Adding a config key touches two places**, enforced by `test/operations/test_config_consistency`: document it in `ocannl_config.reference` and register it in `Utils.known_config_keys`. A new source file needs no registration — the consistency tests glob every directory that can read configuration: `arrayjit/lib/*.ml`, `tensor/*.ml`, `lib/*.ml`, `bin/*.ml`, `tools/*.ml`, `benchmarks/runners/ocannl/*.ml` (gh-ocannl-592). A new source file also owes those tests no promote round: their goldens name the scanned roots and hold a per-root floor under each, with the exact counts on stderr (gh-ocannl-701). A key read only from a test is out of scope, deliberately: tests are not user-facing configuration. What the scan does need is the key spelled as a string literal at the call site (`~arg_name:"the_key"`): a helper taking the key as a parameter hides every key routed through it, so the same test fails any non-literal use outside the named lookup functions
- **Classify the key too**: `test/operations/digest_completeness` fails on a key with no entry in `Utils.config_key_classification` (gh-ocannl-572) — say whether it reaches the schedule cache's identity, and which component
- **A new post-lowering module still needs registering** in that test's `codegen_stage_modules` list — a new backend, or anything else reading configuration after lowering. The globs scan it either way, but that list is what marks its reads as happening downstream of the canonical digest, so a `Code_borne` misclassification of one of its keys goes unnoticed without it. It is a judgment call about the module, which is why it stays hand-written where the file list no longer is

**Configuration Methods** (in order of precedence):
1. Command-line flags: `--ocannl_<option>=<value>` (e.g., `--ocannl_backend=cuda`)
2. Environment variables: `OCANNL_<OPTION>=<value>` (e.g., `OCANNL_BACKEND=cuda`)
3. Config file: `ocannl_config` in current or ancestor directories

**Config profiles** (gh-ocannl-559): `profile=reproducible|performance` picks a preset bundle
whose payload (an embedded partial config file in `arrayjit/lib/utils.ml`) applies at the sublevel
just below the explicit keys of whichever source picked it — so explicit keys beat a profile of
equal immediacy, and a CLI-picked profile beats an exhaustive config file. Payload keys must be
known and documented, and the reference file's verbatim quote of each payload is checked, both by
`test/operations/test_config_consistency`.

**Testing with Different Configurations**:

- Dune re-runs a test for an environment variable only where the stanza declares it as an `(env_var OCANNL_<KEY>)` dependency — so for a key a test should react to, add the declaration (see the testing section above), rather than working around the stale run
- **Warning**: `dune test --force` does NOT re-run expect tests (only rules with alias fields)
- For a one-off configuration change no stanza declares (notably for inline `%expect` tests): modify `test/config/ocannl_config` directly, run `dune clean`, or touch the affected test sources

**Important Debug Settings**:
- `output_debug_files_in_build_directory=true` - enables `build_files/` generation; files go to `build_files/<exe-name>/` (per-executable subdirectory, override with `build_files_prefix`; `build_files_prefix=.` for a flat layout)
- `debug_log_from_routines=true` - enables runtime logging from kernels aka. routines
- `debug_log_to_stream_files=true` - writes logs from kernels/routines to `log_files/<exe-name>/<backend>-<device>-<stream>.log`
- `clean_up_build_files_on_startup=false` and `clean_up_log_files_on_startup=false` - preserve debug files between runs
- CUDA routine logs may require `Utils.capture_stdout_logs` (see README)

**Available Backends**:
- `cc` (the default) combines the implementation cc_backend.ml with the scheduler `Sync` in schedulers.ml; kernel-level CPU parallelism is automatic (pool-rendered Grid loops)
- `multidev_cc` combines cc_backend.ml with the scheduler `Multidev`: multiple worker-domain CPU devices, for debugging multi-device parallel workflows ("sync_cc"/"multicore_cc" are accepted as deprecated aliases of cc/multidev_cc)
- `cuda` with implementation in cuda_backend.ml
- `hip` (AMD ROCm/HIP) with implementation in hip_backend.ml, mirroring the CUDA backend
- `metal` with implementation in metal_backend.ml

Backends are process-wide singletons: use `Backends.get_backend ()` or the Context API (`arrayjit/lib/context.mli`); `fresh_backend` is retired. Merge buffers (`.merge`) support stream-to-stream reductions in `%cd`.

### Backend Development

- Backends must implement stream-based execution with FIFO queuing
- Support for events and synchronization between streams/devices  
- Code generation through `Low_level.t` to backend-specific representations

**Backend Code Generation Architecture**:
- `c_syntax.ml` provides a functor with default C code generation patterns
- `cc_backend.ml` uses defaults from `c_syntax.ml` with minimal overrides
- `cuda_backend.ml` overrides more functions for CUDA-specific syntax (e.g., `__float2half`)
- `metal_backend.ml` overrides using MSL-specific syntax
- Backends must provide `convert_precision` for type conversions
- Builtin functions (e.g., type conversions) must be implemented in the per-backend builtin modules prepended to generated code: `builtins_cc.ml` for the C backends, `builtins_cuda.ml` (CUDA), `builtins_hip.ml` (HIP), `builtins_metal.ml` (Metal). `builtins.c` provides the host-side FFI stubs compiled into the library
- When adding new precision types, ensure conversion functions exist in all backend builtins

### Syntax Extensions

- `%cd` requires `NTDSL` module in scope (from `Operation.NTDSL`)
- `%op` requires `TDSL` module in scope (from `Operation.TDSL`)
- Record syntax for inline tensor declarations: `{ tensor_name }` or `{ tensor_name = init_expr }`
- Generalized einsum notation for complex tensor operations

**Key differences between %op and %cd**:
- `%op` allows initialization expressions (`{ x = uniform () }`), used for model parameters
- `%cd` is self-referential only (`{ x }`), used in computation graphs where tensors are defined by operations
- Inline parameter init in `%op` is forward-only and uses NTDSL internally; `TDSL.param` adds the final parameter gradient
- See `docs/syntax_extensions.md` for comprehensive documentation

**Record syntax features**:
- OCaml punning: `{ x }` expands to default initialization (centered scaled packed `uniform()` over `[-0.25, 0.25)` for parameters in %op, but configurable via `TDSL.default_param_init`)
- Shorthand field names: `o` → `output_dims`, `i` → `input_dims`, `b` → `batch_dims`
- Additional fields map to labeled arguments of tensor creation functions `Tensor.op_fun`
- Dimension specification for tensor literals: lists `[...]` for output, tuples `(...)` for input, arrays `[|...|]` for batch

**Einsum notation**:
- Binary form: `tensor1 +* "spec1; spec2 => result_spec" tensor2`
- Unary form: `tensor ++ "spec => result_spec"`
- Capture dimensions: `+* "spec" ["var1"; "var2"]` binds dimension variables
- Use `Shape.set_dim var value` to constrain captured dimensions
- Special operators -- binary: `+*` (`einsum`, add-reduce with multiply), `@^+` (`tropical`, max-reduce with add), `+++` (`outer_sum`, add-reduce with add); unary: `++` (`einsum1`, add-reduce), `@^^` (`einmax1`, max-reduce)
- Concatenation: `a^b` in specs creates concatenated axis (for slicing, block tensors)

**Common gotchas and idioms**:
- `*` is tensor/matrix multiply, `*.` is pointwise multiply (no `/`, use `/.` for pointwise division)
- `**.` is pointwise power with a numeric exponent (specialized gradients)
- Use `_rhs1`/`_rhs2`/`_lhs` suffixes in `%cd` for intermediate tensors when projection slots matter
- `stretch 1.0` creates a shape-inferred constant 1 whose shape resolves at the use site; `1.0` alone is a fixed scalar. Operation results otherwise close down to their arguments' shapes — a use site broadcasts them in but cannot widen them (gh-544; the old `0.5 + 0.5` idiom relied on the pre-544 widening default)
- Einsum spec must be a literal string when capturing dimensions: `x ++ "a,b" ["a"]` works, `let s = "a,b" in x ++ s ["a"]` fails
- Single-char vs multi-char mode: `"abc"` = 3 axes; `"abc,"` = 1 axis named `abc` (comma triggers multi-char)
- `{ param }` in `%op` creates learnable parameters; same syntax in `%cd` creates non-differentiable tensors
- Sub-modules with `()` must be bound before input: `let layer = make_layer () in fun x -> layer x`
- No reshape/flatten—use multi-axis operations or row variables instead

## Common Development Tasks

Touch-lists for adding a primitive operation, extending a backend, extending shape
inference, and diagnosing backend output discrepancies live in the `extending-ocannl`
skill (`.claude/skills/extending-ocannl/SKILL.md`). Debug-artifact and ppx_minidebug
tracing recipes live in the `ocannl-debug-tracing` skill
(`.claude/skills/ocannl-debug-tracing/SKILL.md`). Agents without skill support read
the files directly. In brief:

- New primitive ops: `arrayjit/lib/ops.ml` (+ `Ir.Ops`), wired into `tensor/operation.ml`
- New tensor convenience functions: `tensor/operation.ml` (use `%cd` for forward/backprop)
- Shape/projection changes: `tensor/shape.ml`, `tensor/row.ml`, `arrayjit/lib/indexing.ml`

## Performance Considerations

- Virtual nodes are inlined automatically (controlled by `virtualize_max_visits`)
- Scalar constants can be inlined via `inline_scalar_constexprs=true`
- Memory sharing optimizations through cross-stream tensor nodes
