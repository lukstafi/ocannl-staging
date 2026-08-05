# OCANNL Agent Guide

OCANNL (OCaml Compiles Algorithms for Neural Networks Learning) is a from-scratch compiled deep
learning framework with an optimizing compiler. The repo contains two main packages:
- arrayjit: low-level IR, lowering, scheduling, and backend codegen (CPU/CUDA/HIP/Metal).
- neural_nets_lib: high-level tensor DSL, shape inference, backprop, and user-facing blocks.

## Structure and Ownership
- lib/: user-facing recipes (training utilities, nn blocks, re-exports).
- tensor/: core framework internals (Tensor, Shape, Operation, ppx_%op/%cd).
- arrayjit/: compiler + backends (assignments.ml, low_level.ml, indexing.ml, schedule.ml,
  context.ml, and the *_backend.ml implementations).
- bin/: runnable examples and demos.
- test/: tutorials and tests (ppx_expect and standalone .expected tests).
- docs/: slides and reference docs; ocannl_config.reference is the configuration source of truth.
- build_files/ and log_files/: generated artifacts when debug settings are enabled.

Key reference files:
- docs/syntax_extensions.md (authoritative for %op/%cd)
- docs/shape_inference.md (shape/projection inference pipeline)
- arrayjit/lib/context.mli (context-based runtime API)
- ocannl_config.reference (all configuration keys and defaults)
- docs/agent-notes.md (distilled cross-session agent knowledge: subsystem traps, known bugs with
  workarounds, debug recipes — skim the matching section before working on a subsystem)

## Conceptual Map (How It Fits Together)
- Tensor expressions (%op, Tensor.t) build a graph with shape inference and backprop rules.
- Assignments (%cd, Assignments.comp) express low-level compute and are compiled by arrayjit.
- Shape inference runs during construction and is finalized by finish_inference before lowering;
  Context.compile and Train.to_routine/run_once/forward_once force completion.
- Projection inference is re-derived per operation to avoid cross-op contamination.

## Build, Run, Test
- Install deps: `opam install . --deps-only` (OCaml >= 5.3).
- Optional GPU packages: `opam install cudajit` (CUDA), `opam install hipjit` (HIP).
- Build: `dune build` (runs cram-style tests) or `dune build @check` (compile only).
- Run an example: `dune exec bin/hello_world.exe`.
- Run the regular suite: `dune runtest` (`cc` is the default backend).
- Run regular + slow training tests: `dune build @runtest @slow`.
- Do not run `dune fmt` during feature work: the repository is not fully ocamlformat-clean and a
  formatting sweep pollutes the diff. Match surrounding style; formatting-state changes are
  standalone commits paired with `.ocamlformat-ignore` updates.

Worktree and shell notes:
- Put worktrees outside the repository when possible. For a nested worktree, run
  `scripts/setup-ocaml-env.sh` from its root to create a worktree-local `dune-workspace`;
  otherwise Dune can silently build the parent checkout. Normal Dune commands and
  `dune promote` then work without `--root .`; see `docs/agent-notes.md` for the mechanics.
- In Windows Git Bash, source `tools/opam-env.sh` before building. Use
  `tools/dune-quiet.sh <dune args>` to filter only the known benign linker warnings.
- In PowerShell, quote aliases (`dune build "@runtest" "@slow"`); unquoted `@name` is splatted
  and can turn a test run into a false-green plain build.

Testing notes:
- Scope test runs to the code and configuration paths a change can reach. Start with affected
  directory aliases such as `dune build @test/operations/runtest`, `@test/einsum/runtest`,
  `@test/ppx/runtest`, `@arrayjit/runtest`, or `@test/training/runtest`, then run
  `dune build @check`. Before broad testing of a config-gated path, use `rg` to verify which tests
  or test configs enable the gate; reserve the full regular/slow suites for cross-cutting changes.
- Inline ppx_expect tests and standalone Dune `test` stanzas with `.expected` files are exclusive.
  Prefer standalone tests for new compiler features; reserve tutorial `%expect` tests for
  illustrative output.
- Avoid `dune exec test/.../<name>.exe` for standalone tests: its working directory bypasses the
  copied `test/config/ocannl_config`, so a test may fail or silently select another backend.
  Build `test/<dir>/<name>.exe.output`, inspect `_build/default/test/<dir>/<name>.exe.output`,
  then run `dune runtest test/<dir>/` and promote. Pin `OCANNL_BACKEND` for bin executables.
  Check that build's exit status (or let its stderr through): if it fails, the previous
  `.exe.output` stays in place and the stale content reads as a green probe.
- A `(test)` stanza already diffs an adjacent `<name>.expected` against the test's output during
  `dune runtest`; the explicit `rule` + `diff` pattern (as in `test_config_consistency`) is only
  needed for tests not run by a `(test)` stanza, such as the `@slow` rules.
- Explicit `@slow` training tests are excluded from `dune runtest`; run `dune build @slow` when
  relevant. Both regular and slow executables are compiled by `dune build @check`.
- Training actions share the `ocannl_training_test` Dune lock, so they do not run their
  process-local OpenMP pools concurrently with one another; compilation remains parallel.
  Preserve this lock on new regular training tests and `@slow` rules. Other pool-using tests,
  notably `test/operations/cpu_parallel`, intentionally remain unlocked as concurrency stress.
- Keep library sources unchanged while Dune is running; edits invalidate in-flight rules and can
  repeat expensive work. For a background run, record its exact PID and exit status; do not poll
  with `pgrep -f`, which can match the waiter itself.
- Do not judge a Dune test through a pipe unless `pipefail` is set; otherwise the consumer's exit
  status can hide expectation diffs.
- `OCANNL_BACKEND` is special-cased by tests; other env vars may not retrigger tests without
  touching sources or cleaning.
- Tests read `test/config/ocannl_config` and can emit .ll/.c/.cu/.metal into build_files/.
- The test configs set `log_config_sourcing=false`, so `.expected` files contain no config-lookup
  banner. Pass `--ocannl_log_config_sourcing=true` (or set it in the config file) to see where
  each setting a run reads comes from; that output then has to be kept out of `.expected` files.
  Canonical workflow:
  write the test, run `dune build test/<...>.exe.output`, then either
  `cp _build/default/test/<...>.exe.output test/<name>.expected` or `dune promote`. Both
  capture the banner correctly.
- On Windows, use `tools/promote.sh` to strip CRLF from promoted goldens. Use
  `test/support/test_utils.ml` printers for portable float output.
- Debug artifacts are isolated under `build_files/<exe-name>/`, so cross-test clashes are
  prevented unless processes explicitly share the flat legacy `build_files_prefix=.` layout.
  Within one executable, duplicate routine names silently overwrite earlier `.cd`/`.ll`/`.c`
  artifacts; keep routine names unique (test-specific prefixes remain useful).
- For optimizer passes that change cell values, emitted-IR structure is not sufficient: also
  assert executed output against a materialized or otherwise independent reference run.

## Coding Conventions
- Prefer small, composable functions; avoid unneeded global state.
- snake_case for files and functions; modules and constructors are capitalized by OCaml.
- Default to ASCII; don’t introduce Unicode unless file already uses it.

## DSL Usage (%op and %cd)
For code outside the core implementation (tests/examples/user code), start with:
`open Ocannl.Operation.DSL_modules`
This brings in Tensor, Shape, TDSL/NTDSL, and Ir.

Key points:
- %op builds Tensor.t; %cd builds Assignments.comp.
- %op requires TDSL in scope; %cd requires NTDSL in scope. Inline parameter init in %op is
  forward-only and uses NTDSL internally; TDSL.param adds the final parameter gradient.
- There is no PDSL. For a differentiable concrete leaf, pass
  `~grad_spec:Tensor.Require_grad` explicitly to Operation.init/Tensor.term_init.
- `%op` inline params allow `{ w }` or `{ w = init }`; `%cd` declarations are self-referential
  `{ w }` only. Dimension shorthands are `o`/`i`/`b`.
- `%op` uses a unit-parameter `()` boundary to lift parameter creation; bind layers at `()`
  before applying to inputs to avoid mis-scoped parameters.
- `**.` is pointwise power with numeric exponent (specialized gradients).

## Idioms & Gotchas
- `*` is matrix/compose; `*.` is pointwise. Use `/.` for pointwise division.
- `%op` inline params without brackets use shape inference; brackets `[...]` fix shape and values.
- Einsum capture requires a literal string: `x ++ "a,b" ["a"]` works; `let s = ... in x ++ s ["a"]` does not.
- Einsum labels: `"abc"` means 3 axes; `"abc,"` means a single axis named `abc` (comma = multi-char mode).
- `0.5 + 0.5` creates an inferred-shape constant that adapts to usage (GLB when known, otherwise
  guessed to the broadcast unit); a lone `1.0` is a fixed scalar dimension and won’t grow with context.
- Use `_rhs1/_rhs2/_lhs` suffixes in %cd for intermediate tensors when projection slots matter.

## Shape & Projection Inference
- Shapes have three rows: batch | input -> output (input is rightmost in underlying arrays).
- Broadcasting can occur with fixed head/tail axes (row variables).
- finish_inference closes unsolved dims (GLB or 1/broadcast); derive_projections re-solves with
  fresh projection ids per op to avoid contamination.
- Generalized einsum `~logic:"...=>..."` supports convolutions, striding, and concatenation `^`.

## Backends, Contexts, and Host Access
- Backends: `cc` (default, automatically pool-parallel within kernels), `multidev_cc`
  (multi-device debugging), `cuda`, `hip`, and `metal`. `sync_cc`/`multicore_cc` are deprecated
  aliases of `cc`/`multidev_cc`.
- Backends are process-wide singletons. Use `Backends.get_backend ()` or the Context API
  (`arrayjit/lib/context.mli`); `fresh_backend` is retired.
- CPU-side value access is explicit and context-mediated (`Context.to_host`/`from_host`/
  `get_values`/`set_values`).
- Merge buffers (`.merge`) support stream-to-stream reductions in %cd.

## Configuration
- Precedence is command-line `--ocannl_<key>`, environment `OCANNL_<KEY>`, then the nearest
  `ocannl_config` in the current/ancestor directories.
- `ocannl_config.reference` is authoritative. Adding a key requires three updates, enforced by
  `test/operations/test_config_consistency`: document it there, register it in
  `Utils.known_config_keys`, and add a newly participating source file to the test's scan list.
- `dune test --force` does not reliably rerun inline expectations. For non-backend config changes,
  edit `test/config/ocannl_config`, clean, or touch the affected test/module.

## Adding Features
- New primitive ops: arrayjit/lib/ops.ml (+ Ir.Ops) and wire into tensor/operation.ml.
- New tensor convenience functions: tensor/operation.ml (use %cd for forward/backprop).
- Shape/projection changes: tensor/shape.ml, tensor/row.ml, arrayjit/lib/indexing.ml.
- Add tests under test/ (einsum/operations/training/ppx as appropriate).
- Prefer 3-4 coherent, independently compiling commits over one large squash; expectation-only
  changes may be grouped in a final test/promotions commit.
- When creating commits, include the work summary in the commit message and credit yourself as a co-author.

## Debugging & Logs
- Enable `output_debug_files_in_build_directory=true` to emit .ll/.c/.cu/.metal.
- Generated artifacts live under per-executable `build_files/<exe-name>/` and
  `log_files/<exe-name>/` directories unless `build_files_prefix` overrides this.
- Enable `debug_log_from_routines=true` for kernel logging; use
  `debug_log_to_stream_files=true` for per-backend/device/stream log files.
- CUDA routine logs may require `Utils.capture_stdout_logs` (see README).
