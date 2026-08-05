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

**Worktrees**: nested ones (`.claude/worktrees/`, the Claude Code default) need a `dune-workspace` at their root, or dune builds the PARENT checkout instead; the SessionStart hook writes it, so after a mid-session worktree switch run `scripts/setup-ocaml-env.sh` by hand. See docs/agent-notes.md for why.

**Windows shells**: `opam env --shell=sh` emits cygwin-style paths that break under Git Bash (MSYS), so a Git Bash session without a primed environment gets a half-working toolchain (dune found but linking fails with `cygpath: error converting ... -lpthread`). Source `tools/opam-env.sh` first — it rewrites the paths for MSYS and works from any POSIX shell. On Windows, link steps also flood stderr with benign binutils warnings (`Warning: corrupt .drectve at end of def file`, from MSVC-produced import libraries like ROCm's/CUDA's); `tools/dune-quiet.sh <dune args>` runs dune with exactly those lines filtered, preserving the exit status.

**Formatting**: the repo is not fully ocamlformat-clean, and CI does not enforce formatting — do NOT run `dune fmt` as part of feature work (it reformats the entire repo and pollutes the diff; recover from an accidental sweep with `git restore .`). Match the surrounding style by hand; to check just your own lines, diff against `_build/default/<dir>/.formatted/<file>`. Formatting-state updates land as standalone formatting commits, paired with updating `.ocamlformat-ignore` — ppx-expectation files (`test/ppx/*_expected.ml`, compared against pretty-printed ppx output) must stay unformatted, so add new ones to that list.

## Architecture Overview

**Before working on a subsystem, skim the matching section of `docs/agent-notes.md`** — distilled
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
- Instead, build the output-capturing rule: `dune build test/operations/<name>.exe.output`, inspect `_build/default/<dir>/<name>.exe.output`, then `dune runtest <dir>` to register the `.expected` diff and promote. For `bin/` executables (same trap), pin `OCANNL_BACKEND=...` explicitly
- When probing that way, do NOT discard the build's stderr (`2>/dev/null`) without checking its exit status: a FAILED build (e.g. a warning-as-error from a temporary edit) leaves the previous `_build/default/<dir>/<name>.exe.output` untouched, so the stale file reads as a green probe — this turned a negative control into a false positive during gh-554
- A `(test)` stanza automatically diffs a `<name>.expected` sitting next to it against the test's output on `dune runtest`; no extra stanza is needed to "assert" the golden. The explicit `rule` + `diff` pattern (as in `test_config_consistency`) is only for tests not run by a `(test)` stanza, such as the `@slow` rules
- Never judge `dune runtest` through a pipe: `... 2>&1 | tail -3 && echo OK` reports the status of `tail`, not dune (no pipefail), and promotion-diff failures look like normal chatter. To capture output while preserving Dune's status, run `( dune runtest; st=$?; printf 'exit: %s\n' "$st"; exit "$st" ) > /tmp/suite.log 2>&1`; append `& pid=$!` for a background run. Do not name that variable `status`: in zsh it is a read-only alias for `$?`, so the assignment errors out, no sentinel is printed, and the subshell exits 1 — a green suite looks failed. Both `wait "$pid"` and the logged `exit: N` sentinel then report Dune's status; grep the log for `^File .*expected` to list promotion diffs
- From PowerShell, QUOTE dune alias targets: PowerShell parses an unquoted `@word` argument as a splatting variable (undefined here → expands to nothing), so `dune build @runtest @slow` silently degrades to a plain `dune build` that exits 0 having run no tests — a false green. Use `dune build "@runtest" "@slow"` (bash/cmd are unaffected)
- Inline tests (like those in `test_threefry4x32.ml`) are part of library modules and run via `dune runtest`, not `dune exec`
- Scope test runs to the code and configuration paths a change can reach. Start with affected directory aliases such as `dune build @test/operations/runtest`, `@test/einsum/runtest`, `@test/ppx/runtest`, `@arrayjit/runtest`, or `@test/training/runtest`, then run `dune build @check`. Before broad testing of a config-gated path, grep whether any test or test config enables the gate; reserve the full regular/slow suites for cross-cutting changes
- Keep library sources unchanged while Dune is running; edits invalidate in-flight rules and can repeat expensive work. For a background run, record its exact PID and exit status; do not poll with `pgrep -f`, which can match the waiter itself

**Slow training tests (the `slow` alias)**:
- Explicit `@slow` training tests are excluded from `dune runtest`; run them only when relevant.
- Regular and `@slow` training actions share the `ocannl_training_test` Dune lock, so they do not run their process-local OpenMP pools concurrently with one another; compilation remains parallel. Preserve the lock on new regular training tests and `@slow` rules. Other pool-using tests, notably `test/operations/cpu_parallel`, intentionally remain unlocked as concurrency stress.
- They are still ordinary executables: `dune build @check` compiles them, so they cannot bit-rot.
- Run them on demand with `dune build @slow` (uses their `.expected` files; `dune promote` to accept changes). Use `dune build @runtest @slow` to run the entire suite.
- To gate a new slow test, in its `test/.../dune` replace its `(test ...)` stanza with an `(executable ...)` plus a `(rule (alias slow) ...)` that runs the exe and diffs against `<name>.expected` (see `test/training/dune` for the pattern). Wrap the rule's action in `(no-infer ...)` so the `<name>.actual` output is NOT registered as a build target — otherwise a plain `dune build` (the `@all` alias builds every file target) would run the slow exe anyway.

**Test Types**:
- **Inline tests**: Files included in library `modules` field with `inline_tests` stanza (e.g., `test_threefry4x32.ml` in `operations_tutorials` library)
- **Standalone tests**: Files with dedicated `test` stanza and corresponding `.expected` files (e.g., `threefry4x32_demo`)
- Use `dune promote` to accept test output changes (on Windows prefer `tools/promote.sh` — it strips CRLF from promoted goldens)
- A few `%expect_test` blocks capture exception backtraces that hard-code `file:line` (e.g. `test/operations/primitive_ops.ml` embeds `context.ml` line numbers). Any edit that shifts lines in such a file forces a line-number-only re-promote — benign noise, not a behavior change; promote it in the same shell as the failing run (the `.ml.corrected` can vanish between runs)
- For optimizer passes that change *what value a cell holds* (virtualization guards, index solving, accumulation/init elision), a structural test on the emitted op tree is necessary but NOT sufficient — also assert on executed output vs. a materialized/reference run (`Context.compile`/`run`/`get_values`); a pass has shipped green structural checks while computing all zeros
- Backend codegen snapshots (e.g. `.cu.expected` files, `test_cuda_pool_offset.expected`) go stale when codegen changes land without that backend's hardware available to re-record them — expect to re-promote such snapshots when the hardware next runs the suite
- **Test Placement Guidelines**:
  * Always add tests under one of the test subdirectories
  * Default location is `test/operations`
  * Use `test/einsum` for tests involving complex einsum specifications
  * Use `test/training` for tests involving training loops
  * When adding a test, update the corresponding test stanza
  * For standalone tests, add an `.expected` file for test results (can initially be empty)
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

- Prefer a series of 3-4 topical commits (one per coherent sub-feature) plus follow-up fixing commits, merged with a merge commit that preserves the series — not one big squashed commit. The series documents the design decomposition and keeps each piece independently reviewable and bisectable
- Each commit should at least compile: loop `git checkout <rev> && dune build @check` over `git rev-list --reverse master..HEAD` (interactive rebase is unavailable in this harness)
- Test-expectation promotions that mix topics can land in a final tests/promotions commit

### Configuration

- See `ocannl_config.reference` for documentation of all settings
- Key configs: backend selection, debug logging, optimization levels
- **Adding a config key touches three places**, enforced by `test/operations/test_config_consistency`: document it in `ocannl_config.reference`, register it in `Utils.known_config_keys`, and if a new `.ml` file gains `get_global_arg` call sites, add that file to the consistency test's source-scan list

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
