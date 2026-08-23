# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OCANNL (OCaml Compiles Algorithms for Neural Networks Learning) is a from-scratch compiled Deep Learning framework with an optimizing compiler. The project consists of two main packages:

- `arrayjit`: The low-level optimizing compiler with multiple backends (CPU, CUDA, Metal)
- `neural_nets_lib`: The high-level deep learning framework with syntax extensions, shape inference, and backpropagation

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

# Install dependencies
opam install . --deps-only

# Install with optional backends  
opam install cudajit  # for CUDA backend
opam install hipjit   # for AMD HIP backend
```

**Worktrees**: nested ones (`.claude/worktrees/`, the Claude Code default) need a `dune-workspace` at their root, or dune builds the PARENT checkout instead; the SessionStart hook writes it, so after a mid-session worktree switch run `scripts/setup-ocaml-env.sh` by hand. See docs/agent-notes/build-and-test.md for why.

**Windows shells**: `opam env --shell=sh` emits cygwin-style paths that break under Git Bash (MSYS), so a Git Bash session without a primed environment gets a half-working toolchain (dune found but linking fails with `cygpath: error converting ... -lpthread`). Source `tools/opam-env.sh` first — it rewrites the paths for MSYS and works from any POSIX shell. On Windows, link steps also flood stderr with benign binutils warnings (`Warning: corrupt .drectve at end of def file`, from MSVC-produced import libraries like ROCm's/CUDA's); `tools/dune-quiet.sh <dune args>` runs dune with exactly those lines filtered, preserving the exit status. Use **Git Bash** specifically, not a Cygwin bash (opam's, or whatever `bash` resolves to once opam's cygwin is on PATH): the two are told apart by `uname -o` (`Msys` vs `Cygwin`, since both bashes report `OSTYPE=cygwin`), only the MSYS one gets `opam-env.sh`'s path rewrite, and Cygwin ships no `perl` — which `tools/test-run.sh` needs for its lock, cap and `last` pointer, and refuses without (gh-ocannl-662).

**Formatting**: type in your own style and do NOT run `dune fmt` in feature work — the repo is reformatted by the scheduled sweep `tools/format-sweep.sh`, which runs only at quiet periods (no open PRs, no active worktrees), iterates format → `@check` → `runtest` → promote to a fixed point (formatting shifts the line numbers some `%expect` goldens embed, so their re-promotions are part of the sweep), and lands standalone commits recorded in `.git-blame-ignore-revs`. A mid-cycle `dune fmt` picks up other changes' accumulated drift and pollutes the diff; recover with `git restore .`. The one formatting duty left in feature work: new ppx-expectation files (`test/ppx/*_expected.ml`, compared against pretty-printed ppx output) must stay unformatted — add them to `.ocamlformat-ignore`, or the unattended sweep will fail to converge and abort.

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
   - Shape inference completion is forced by lowering: via `Context.compile`, or wrappers such as `Train.to_routine`, `Train.run_once` or `Train.forward_once`
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
- `dune runtest` - runs all tests including inline tests and cram-style tests, EXCEPT the slow training runs (see below)
- `dune runtest test/operations/` - runs all tests in operations directory
- Avoid `dune exec test/.../test_name.exe` for standalone tests: `dune exec` keeps the invocation cwd, the config search walks UP from cwd (`Utils.config_file_args`), and the root `ocannl_config` is deliberately gitignored (personal dev settings) — so fresh clones, CI, and worktrees find no config and the test dies partway, or `Context.auto` silently picks metal→cuda→cc (a "cc" run quietly executes on the GPU). Tests only find their config under `dune runtest`/build rules because dune sets their cwd to `_build/default/<test dir>` where `(copy_files ../config/ocannl_config)` materialized it
- Instead, run the one test by its own alias: `dune build @test/operations/runtest-<name>`. It runs the exe with dune's cwd AND applies the `.expected` diff, so a changed output fails the build instead of sitting unread in a file, and `dune promote` accepts it; a misspelled name exits 1 ("Alias ... is empty"), not a silent no-op. The run still leaves `_build/default/<dir>/<name>.exe.output` to inspect
- A test written as an `(executable)` plus a `(rule)` that diffs a golden — the scanning tests, the codegen snapshot rules, the config-precedence rules, the ppx-output diffs — gets no alias from dune, so each such rule carries `(alias runtest-<name>)` of its own (gh-ocannl-726) and `dune build @test/operations/runtest-verdict_ratchet` runs that one test and applies its diff, promotable, with a misspelling exiting 1. Three rules govern writing one, all three checked by `test/operations/env_var_deps`: the rule names that alias and NOT `runtest` (a rule on two aliases makes building either build both, so naming both would put the whole directory behind the per-test alias); an `(alias (name runtest) (deps (alias runtest-<name>) …))` stanza per directory aggregates them back, or plain `dune runtest` skips them silently; and `<name>` is the golden the rule checks, never a `(test)` stanza's name in that directory, since dune generates `runtest-<name>` for those. The rule also depends on `(alias runtest-env_spelling_gate)`, which is how the ambient gate reaches a per-test alias
- The repo-wide scans — `test_config_consistency`, `digest_completeness`, `config_dep_completeness`, `env_var_deps`, `cache_dir_ignores`, `verdict_ratchet`, `agent_notes_structure` — also share the family alias `dune build @test/operations/scans`, which runs all seven in seconds without the directory suite (gh-ocannl-703). They are the checks a small targeted change trips, so run them before pushing a change to a config key, a dune stanza, a printed claim, or an agent note. A scan the family stanza omits fails `env_var_deps`, which derives the family from the rules that glob the repository rather than from a second copy of the list
- Dune's own `runtest-<name>` (per `(test)`/`(tests)` name and per inline-test library) needs **dune >= 3.20** while the packages' floor is 3.18; on an older dune, and for a target with no alias at all, build it instead: `dune build test/operations/<name>.exe.output` (or `<name>.actual`), inspect it under `_build/default/<dir>/`, then `dune runtest <dir>` to register the diff and promote. For `bin/` executables (same cwd trap), pin `OCANNL_BACKEND=...` explicitly
- Pinning a variable on such a probe (`OCANNL_BACKEND=cuda dune build @test/operations/runtest-<name>`) reaches the run only for variables the stanza DECLARES: dune tracks no environment variable it was not told about, so an undeclared one leaves the target up to date and the previous run's result in place. Every test stanza that can select a backend declares `(env_var OCANNL_BACKEND)` — one spelling, uppercase, since gh-ocannl-652 dropped the lowercase `ocannl_<key>` the environment used to be read under. A stanza that names its backend, or links none, declares nothing and instead carries a marker comment INSIDE its own parentheses saying which and why: `; ocannl-backend: none -- links arrayjit.ir alone and calls only pure arithmetic`, `; ocannl-backend: metal -- pins MSL emission`. Exactly one of the two, over every stanza that runs an executable — neither is the hole gh-ocannl-659 closed (dune then serves the previous backend's output as a pass), and both is contradictory intent. The words are `none`, `cc`, `multidev_cc`, `cuda`, `hip`, `metal`, comma-separated where a stanza honestly names two; the reason is required and must be more than one word. For an `(executable)` plus a `(rule)`, the marker goes on the rule — same placement as the `ocannl_config` dep. For any other key add `OCANNL_<KEY>` to that stanza's `(deps ...)` first — `test/operations/env_var_deps` checks all of this, and that every `(env_var ...)` addressed to OCANNL names a spelling a run reads (gh-ocannl-628, gh-ocannl-659). Setting the dropped lowercase form is a fatal startup error, not a silent no-op
- Confirming WHICH backend a probe ran on cannot come from a backend-uniform golden: a test that announces `SKIPPED on <backend>` writes that to stderr and still prints `<claim>: true` on stdout, precisely so the golden stays backend-uniform — so a GPU run's `.exe.output` can be byte-identical to the cc one. Read the run's stderr, or have the test print the backend (gh-ocannl-622)
- When probing that way, do NOT discard the build's stderr (`2>/dev/null`) without checking its exit status: a FAILED build (e.g. a warning-as-error from a temporary edit) leaves the previous `_build/default/<dir>/<name>.exe.output` untouched, so the stale file reads as a green probe — this turned a negative control into a false positive during gh-554
- `(copy_files ../config/ocannl_config)` only materializes the config for rules that DEPEND on it: every `(test)`/`(tests)` stanza must list `ocannl_config` in its `(deps ...)`, and so must every `(rule ...)` that runs a test executable (an `(executable)` stanza has no `deps` field, so its companion rule is where the dep goes). Nothing is sandboxed, so a stanza missing it does not fail on its own — it reads whatever is in `_build/default/<test dir>` when it happens to run, which makes both the `.exe.output` probe and the test itself order-dependent (gh-ocannl-586). "Pure" tests are no exception: without the dep they also get default `print_decimals_precision`/`fixed_state_for_init` and a config-sourcing trace on stderr. `test/operations/config_dep_completeness` checks this over every `dune` file in the repository, including new directories, so an omission fails the suite rather than drifting (gh-ocannl-597); an executable that no rule runs is not a site, and a rule running something that reads no configuration goes on that test's named exemption list
- A `(test)` stanza automatically diffs a `<name>.expected` sitting next to it against the test's output on `dune runtest`; no extra stanza is needed to "assert" the golden. The explicit `rule` + `diff` pattern (as in `test_config_consistency`) is only for tests not run by a `(test)` stanza, such as the `@slow` rules
- Run suites through `tools/test-run.sh` rather than hand-rolling shell around dune: `tools/test-run.sh run runtest test/operations`, `tools/test-run.sh run build @slow` (arguments after the options are dune's argv; default `runtest`). It runs dune unpiped (piping dune to anything reports the pipe's status, so a promotion diff reads as green), caps wall-clock time (default 3600s, `--cap N`, raise for `@slow`), logs to a file ending in an `exit: N` sentinel, prints a compact digest (verdict, promotion diffs, error fingerprint), and exits with dune's status. The script header documents the other shell traps it absorbs
- From PowerShell, QUOTE dune alias targets: PowerShell parses an unquoted `@word` argument as a splatting variable (undefined here → expands to nothing), so `dune build @runtest @slow` silently degrades to a plain `dune build` that exits 0 having run no tests — a false green. Use `dune build "@runtest" "@slow"` (bash/cmd are unaffected)
- Inline tests (like those in `test_threefry4x32.ml`) are part of library modules and run via `dune runtest`, not `dune exec`
- Scope test runs to the code and configuration paths a change can reach. Start with affected directory aliases such as `dune build @test/operations/runtest`, `@test/einsum/runtest`, `@test/ppx/runtest`, `@arrayjit/runtest`, or `@test/training/runtest`, then run `dune build @check`. Before broad testing of a config-gated path, grep whether any test or test config enables the gate; reserve the full regular/slow suites for cross-cutting changes
- Never write a sleep/`pgrep` waiter loop around a test run: `pgrep` waiters match the editor's immortal `dune ocaml-merlin` daemons (or the waiter itself) and spin forever, stranding shells. For a long run, launch `tools/test-run.sh run ...` through the Bash tool's background mode and act on its completion notification — no polling needed. Only for a run that must outlive the session, use `tools/test-run.sh start ...`, then the non-blocking `status last` or the hard-bounded `wait last`; both read the run's recorded verdict file rather than probing processes
- Keep library sources unchanged while Dune is running; edits invalidate in-flight rules and can repeat expensive work

**Slow training tests (the `slow` alias)**:
- Explicit `@slow` training tests are excluded from `dune runtest`; run them only when relevant.
- Regular and `@slow` training actions share the `ocannl_training_test` Dune lock, so they do not run their process-local OpenMP pools concurrently with one another; compilation remains parallel. Preserve the lock on new regular training tests and `@slow` rules. Other pool-using tests, notably `test/operations/cpu_parallel`, intentionally remain unlocked as concurrency stress.
- They are still ordinary executables: `dune build @check` compiles them, so they cannot bit-rot.
- Run them on demand with `dune build @slow` (uses their `.expected` files; `dune promote` to accept changes). Use `dune build @runtest @slow` to run the entire suite.
- Each slow rule also sits on its own `slow-<name>` alias, so ONE slow test reruns alone after a change: `OCANNL_BACKEND=cc dune build @test/training/slow-mlp_names` runs that directory's `env_spelling_gate` and then just `mlp_names`, applying its `.expected` diff (promotable with `dune promote`); a misspelled name exits 1 (`Alias ... is empty`) rather than silently running nothing. `@slow` is per directory an `(alias (name slow) (deps (alias slow-env_spelling_gate) (alias slow-<name>) ...))` stanza aggregating them, and `test/operations/env_var_deps` fails on a `slow-<name>` rule that stanza does not list, since `dune build @slow` would otherwise skip it silently
- To gate a new slow test, in its `test/.../dune` replace its `(test ...)` stanza with an `(executable ...)` plus a `(rule (alias slow-<name>) ...)` that runs the exe and diffs against `<name>.expected`, with `(alias slow-env_spelling_gate)` first in its `(deps ...)` so the ambient gate runs before the test, and add `(alias slow-<name>)` to the directory's `(alias (name slow) ...)` stanza (see `test/training/dune` for the pattern). Wrap the rule's action in `(no-infer ...)` so the `<name>.actual` output is NOT registered as a build target — otherwise a plain `dune build` (the `@all` alias builds every file target) would run the slow exe anyway.

**Test Types**:
- **Inline tests**: Files included in library `modules` field with `inline_tests` stanza (e.g., `test_threefry4x32.ml` in `operations_tutorials` library)
- **Standalone tests**: Files with dedicated `test` stanza and corresponding `.expected` files (e.g., `threefry4x32_demo`)
- Use `dune promote` to accept test output changes (on Windows prefer `tools/promote.sh` — it strips CRLF from promoted goldens)
- **A test that decides its own verdict must report it through `Verdict`** (`test/support/verdict.ml`; add `verdict` to the stanza's `(libraries ...)`): `Verdict.p "claim" b` prints `claim: b`, `Verdict.pass_fail "label" b ~detail` prints a `label: PASS`/`FAIL` column with the machine-specific number only on failure, `Verdict.fail "what is wrong"` reports a one-off, `Verdict.skipped ~backend "claim"` prints the passing golden line for a leg this backend cannot evaluate and announces it on stderr. A failed check exits the process 1 from a shared teardown and echoes to stderr (dune discards the redirected stdout of a process that exits nonzero). Without that exit status the golden diff is the only gate, and it is promotable: a test that prints `FAIL: …` or `<claim>: false` and exits 0 lets the natural next move — `dune promote` — record the failure as the expected output, and nothing fails again until someone reads the golden (gh-ocannl-601). Two consequences: **phrase every claim so `true` is the passing reading** (rename the fact, `not (…)`, rather than recording a designed `false` — in a golden a blessed regression and a designed negative are the same line), and keep purely descriptive output (values and tables the golden pins) on plain `printf`, which is not an assertion channel. `Ll_test.p` is `Verdict.p`, so hand-built-IR tests get this for free. **The literal-label half of this is enforced**, not merely documented: `test/operations/verdict_ratchet` parses every test source and fails on a format whose one argument-consuming conversion is a bare `%b` at the end behind a literal label (`"fused: %b\n"`), which is the shape a self-decided claim takes (gh-ocannl-668, after a post-sweep test reintroduced four such sites unnoticed). A descriptive `%b` print escapes by carrying any other conversion — which one interpolating what it describes does anyway — or by a named exemption there. Computed labels (`"%s fused: %b\n"`) are out of its reach by design and remain issue #624's manual work
- **A float that a device reduction produced does not belong in a stdout golden at a fixed precision.** Lowering `%.4f` to `%.1f` moves the rounding tie, it does not remove it: `cifar_conv`'s epoch-30 mean read `1.0` on cc and `1.1` on cuda at the same commit, and no promotion served both. Print the exact digits to stderr, tagged `(not part of the golden)`, and put a `Verdict` claim about what the number was showing on stdout — a threshold the trained value clears and an untrained one does not, a fall between the first and last logged epoch, an argmax or a ranking, a closed-form value within a tolerance. The claim must fail on a wrong number, so "is finite" alone is not one — and it must be TWO-SIDED wherever the digits were: an upper bound on a loss admits the finite negative value a dropped negation produces, and a claim about ranking or ordering admits a badly distorted magnitude, so pair it with a validity range. Mark every relocated line `(not part of the golden)`. Floats that are exact by construction (threshold constants, power-of-two loss scales, host-side closed-form schedules, small dyadic sums) stay on stdout. Nothing lints this; the audited `test/training`/`test/gpt2` sites and the classification rule are in `docs/agent-notes/build-and-test.md` (gh-ocannl-725)
- A few `%expect_test` blocks capture exception backtraces that hard-code `file:line` (e.g. `test/operations/primitive_ops.ml` embeds `context.ml` line numbers). Any edit that shifts lines in such a file forces a line-number-only re-promote — benign noise, not a behavior change; promote it in the same shell as the failing run (the `.ml.corrected` can vanish between runs)
- For optimizer passes that change *what value a cell holds* (virtualization guards, index solving, accumulation/init elision), a structural test on the emitted op tree is necessary but NOT sufficient — also assert on executed output vs. a materialized/reference run (`Context.compile`/`run`/`get_values`); a pass has shipped green structural checks while computing all zeros
- The executed reference must DISCRIMINATE, not merely exist: every producer should write a value that varies with *every* symbol of its iteration and stays clear of the init/sentinel value (`1 + i`, `1 + 10*outer + inner` — see the `tick`/`tag` helpers in `test/operations/virtual_diagonal.ml`). A constant producer replays an identical assignment under a too-wide range guard, a value omitting one symbol is constant along that axis under a wrong substitution, and a value colliding with the zero-init hides a dropped first iteration — the leg passes for the wrong reason
- **A test that asserts on generated code reads it through `Test_utils.Generated`** (`test/support/generated.ml`; add `test_utils` to the stanza's `(libraries ...)`), never by opening `build_files/<routine>.<ext>` itself. Artifacts outlive the run that wrote them (`clean_up_build_files_on_startup=false` in the test config) and a second compile under the same routine name overwrites the first's, so a read that does not establish provenance can keep asserting on a kernel that is no longer emitted. Call `Generated.init ~backend_name` once, before the first compile (such a test must leave `build_files_prefix` at its default: a configured prefix names a directory the process does not own, so provenance cannot be established there and `init` refuses it); then `Generated.assert_emits ~routine ~contains "claim"`, or `Generated.read routine` for a check that needs the source. A missing artifact fails the run rather than answering `None`, so a leg this backend cannot evaluate must be gated and reported with `Verdict.skipped` instead of reaching the read; a loop compiling several candidates under one routine name calls `Generated.arm routine` before each compile, which is what makes each candidate judged by its own kernel
- Backend codegen snapshots (e.g. `.cu.expected` files, `test_cuda_pool_offset.expected`) go stale when codegen changes land without that backend's hardware available to re-record them — expect to re-promote such snapshots when the hardware next runs the suite
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
- When you notice unrelated code smells or design problems, file separate issues
- Follow-up fixing commits are fine, and test-expectation promotions that span several topics can land in a final tests/promotions commit
- Each commit should at least compile: loop `git checkout <rev> && dune build @check` over `git rev-list --reverse master..HEAD` (interactive rebase is unavailable in this harness)
- **Bring the base in before opening the PR and again before merging**: GitHub builds the pull request's MERGE commit, so a check that scans the whole repository can be red on the tree CI builds while green on your branch. Fetch the STAGING remote (resolve which name points there per the next bullet — it need not be `origin`) and rebase onto its `master`, or merge it in where the branch is shared and rewriting is not yours to do — `docs/agent-notes/build-and-test.md` explains the mechanism and how to validate against the merged tree
- **Two repositories, and remote names are not the contract**: development — branches, PRs, and the `master` they land on — happens in `lukstafi/ocannl-staging`, while `ahrefs/ocannl` is the public repo that owns the ISSUES this codebase cites as `gh-ocannl-NNN`, the milestones and the GitHub releases, and receives release-relevant changes. Which remote name points where is local: a clone has whatever names it was given, a clone of the public repo calls IT `origin`, and a fresh clone has no second remote at all. So check `git remote -v` before trusting a name, add the other repo explicitly when you need it (`git remote add upstream https://github.com/ahrefs/ocannl.git`), and pass `--repo <owner>/<name>` to every `gh` command rather than letting it infer: issues to `ahrefs/ocannl`, PRs to `lukstafi/ocannl-staging`

### Configuration

- See `ocannl_config.reference` for documentation of all settings
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

- When using environment variables for test configuration other than OCANNL_BACKEND, Dune won't detect changes and may skip tests
- **Warning**: `dune test --force` does NOT re-run expect tests (only rules with alias fields)
- Reliable ways to ensure tests run with new configuration:
  1. Modify `test/config/ocannl_config` directly
  2. Run `dune clean` before testing
  3. Touch/modify test source files
  4. OCANNL_BACKEND environment variable is an exception (explicit dependency)

**Important Debug Settings**:
- `output_debug_files_in_build_directory=true` - enables `build_files/` generation; files go to `build_files/<exe-name>/` (per-executable subdirectory, override with `build_files_prefix`; `build_files_prefix=.` for a flat layout)
- `debug_log_from_routines=true` - enables runtime logging from kernels aka. routines
- `debug_log_to_stream_files=true` - writes logs from kernels/routines to `log_files/<exe-name>/<backend>-<device>-<stream>.log`
- `clean_up_artifacts_on_startup=false` - preserves debug files between runs

**Available Backends**:
- `cc` (the default) combines the implementation cc_backend.ml with the scheduler `Sync` in schedulers.ml; kernel-level CPU parallelism is automatic (pool-rendered Grid loops)
- `multidev_cc` combines cc_backend.ml with the scheduler `Multidev`: multiple worker-domain CPU devices, for debugging multi-device parallel workflows ("sync_cc"/"multicore_cc" are accepted as deprecated aliases of cc/multidev_cc)
- `cuda` with implementation in cuda_backend.ml
- `hip` (AMD ROCm/HIP) with implementation in hip_backend.ml, mirroring the CUDA backend
- `metal` with implementation in metal_backend.ml

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
- Builtin functions (e.g., type conversions) must be implemented in:
  - `builtins.c` for C backends
  - `builtins_cuda.ml` for CUDA backend, `builtins_metal.ml` form Metal backend
- When adding new precision types, ensure conversion functions exist in all backend builtins

### Syntax Extensions

- `%cd` requires `NTDSL` module in scope (from `Operation.NTDSL`)
- `%op` requires `TDSL` module in scope (from `Operation.TDSL`)
- Record syntax for inline tensor declarations: `{ tensor_name }` or `{ tensor_name = init_expr }`
- Generalized einsum notation for complex tensor operations

**Key differences between %op and %cd**:
- `%op` allows initialization expressions (`{ x = uniform () }`), used for model parameters
- `%cd` is self-referential only (`{ x }`), used in computation graphs where tensors are defined by operations
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
- `stretch 1.0` creates a shape-inferred constant 1 whose shape resolves at the use site; `1.0` alone is a fixed scalar. Operation results otherwise close down to their arguments' shapes — a use site broadcasts them in but cannot widen them (gh-544; the old `0.5 + 0.5` idiom relied on the pre-544 widening default)
- Einsum spec must be a literal string when capturing dimensions: `x ++ "a,b" ["a"]` works, `let s = "a,b" in x ++ s ["a"]` fails
- Single-char vs multi-char mode: `"abc"` = 3 axes; `"abc,"` = 1 axis named `abc` (comma triggers multi-char)
- `{ param }` in `%op` creates learnable parameters; same syntax in `%cd` creates non-differentiable tensors
- Sub-modules with `()` must be bound before input: `let layer = make_layer () in fun x -> layer x`
- No reshape/flatten—use multi-axis operations or row variables instead

## Common Development Tasks

Touch-lists for adding a primitive operation, extending a backend, extending shape
inference, and diagnosing backend output discrepancies live in the `extending-ocannl`
skill. Debug-artifact and ppx_minidebug tracing recipes live in the
`ocannl-debug-tracing` skill.

## Performance Considerations

- Virtual nodes are inlined automatically (controlled by `virtualize_max_visits`)
- Scalar constants can be inlined via `inline_scalar_constexprs=true`
- Memory sharing optimizations through cross-stream tensor nodes
