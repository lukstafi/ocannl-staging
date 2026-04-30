# Ternary einsum notation / projection inference

## Goal

Support 3-operand einsum notation `"rhs1 ; rhs2 ; rhs3 => lhs"` in OCANNL's
`%cd` PPX, shape/projection inference, and a corresponding tensor-level entry
point, with working gradients. Removes the explicit "not supported yet"
blocker in `ppx_cd.ml` that references this issue.

Tracks [issue #305](https://github.com/ahrefs/ocannl/issues/305): "It's not
needed, but cool and not much work."

## Acceptance Criteria

- [ ] In `%cd` blocks, a ternary operator with `~logic:"<einsum-spec>"` no
  longer raises the
  `"einsum notation for ternary operators not supported yet, see issue #305"`
  error. The spec routes through ternary einsum shape inference and
  projection derivation.
- [ ] `Shape.ternary_type` gains a constructor representing ternary einsum
  (e.g. `Einsum_tern of string * delayed_var_ref list`), and
  `Shape.Broadcast_tern (Einsum_tern ..., sh1, sh2, sh3)` participates in
  constraint generation in `shape.ml` analogously to the existing binary
  `Broadcast (Einsum ...)` branch.
- [ ] `derive_projections` produces correct iteration product space for
  3-operand einsum: shared label appears once in the product, contraction
  labels (present in RHS{1,2,3} but absent in LHS) become reduction axes.
- [ ] Pure-contraction chain: `"ij ; jk ; km => im"` produces the same numeric
  result (within `Float.equal` after a deterministic backend run) as the
  binary chain `(a *+ "ij;jk=>ik" b) *+ "ik;km=>im" c` for randomly-shaped
  inputs of compatible dims. This is the **falsifier** test for projection
  inference.
- [ ] A non-trivial mixed-axis test: e.g. `"bij ; bjk ; bkm => bim"` (batched
  triple matmul) matches the equivalent binary chain.
- [ ] Gradient correctness: for `loss = sum(einsum3 spec a b c)` and the
  binary-chain equivalent, gradients on `a`, `b`, `c` agree (within
  numerical tolerance) on a small randomized test.
- [ ] An `operation.ml` entry point (e.g. `einsum3 spec t1 t2 t3` or extending
  `*+` / `+*` to accept three operands when the spec has three RHS) exposes
  ternary einsum from OCaml code, not only from `%cd` blocks. The exact
  surface API is left to the implementer.
- [ ] No regression in existing binary-einsum and unary-permute tests
  (`test/ppx/test_ppx_op.ml`, `test/training/mlp_names.ml`,
  `test/ppx/test_ppx_name_conflict.ml`).
- [ ] The `[%expr ...]` error builder at the ternary-spec branch in
  `ppx_cd.ml` no longer references `#305`.

## Context

### What is already in place (verified)

- **Parser**: `tensor/parser.mly` rule `rhs_specs` already recurses on
  `SEMICOLON`, and `einsum_spec` returns
  `parsed_axis_labels list * parsed_axis_labels`. The list is unbounded —
  no parser change is required to accept 3+ RHS specs.
- **PPX RHS3 plumbing**: `tensor/ppx_cd.ml` has the full `RHS3` slot
  machinery: `projections_slot` includes `RHS3`, `ternary_op` lookup,
  `process_assign_ternop` and `process_raw_ternop` set up `setup_r3`,
  build `rhs_dims = [| rhs1_dims; rhs2_dims; rhs3_dims |]` and
  `project_rhs = [| project_rhs1; project_rhs2; project_rhs3 |]`, and emit
  `Ternop { op; rhs1; rhs2; rhs3 }` into `Accum_op`.
- **Assignments IR**: `arrayjit/lib/assignments.ml` `Ternop` is a buffer
  triple. Iteration order is determined by the `projections` field of
  `Accum_op`, *not* by the op constructor. So the IR is agnostic to whether
  the projections are pointwise or contraction-style — it will iterate
  whatever projection structure `derive_projections` produces.
- **Existing ternary ops**: `operation.ml` defines `fma` and `where` via
  `Tensor.ternop ~ternary_op:Pointwise_tern`. These are pointwise.
- **Ternary shape branches**: `Shape.Broadcast_tern` exists with
  `Pointwise_tern | Compose_accumulate | Defined_by_cd_logic` variants; the
  first two have constraint-generation branches in `shape.ml`.

### The actual blocker

The error message at the spec dispatch site for ternary ops in `ppx_cd.ml`
(in the branch handling `accu_op lhs (tern_op (rhs1, rhs2, rhs3) ~logic:"...")`):

```
"ppx_ocannl %%cd: expected <.> or <@>, found <%s> -- einsum notation for ternary
 operators not supported yet, see issue #305"
```

The branch only accepts `"."` (mapped to `Shape.Pointwise_bin`) and `"@"`
(mapped to `Shape.Compose`) as the `logic` payload — it has no `Shape.Einsum`
analog because `Shape.ternary_type` has no `Einsum` constructor. Compare the
binary case in the same file (a few lines earlier), which falls through to
`Shape.Einsum (logic, [])` for any other spec.

### What is missing (the real work)

1. **`Shape.ternary_type`** (`tensor/shape.mli`, `tensor/shape.ml`) needs a
   new constructor for ternary einsum, carrying the spec string and
   `delayed_var_ref list` (mirroring binary `Einsum`).
2. **Shape constraint generation** (`shape.ml`): a new branch in the big
   `match` that handles `Broadcast_tern (Einsum_tern (spec, dim_refs), sh1,
   sh2, sh3)`. The binary branch (`Broadcast (Einsum ...)`) is ~140 lines
   and already carries a `TODO: refactor to avoid duplication with the one
   for unary einsum`. A ternary copy is feasible but a refactor that
   parameterizes over arity (1/2/3 RHS via `parsed_axis_labels list`) would
   be cleaner. Choice left to implementer; refactor is preferred.
3. **`derive_projections`** for the ternary einsum branch — produce the
   union product space, with reduction axes for any label that is in some
   RHS but not in LHS. The binary case in the same function is the
   reference.
4. **Logic dispatch in `ppx_cd.ml`**: replace the error builder so that any
   spec other than `"."`/`"@"` is wrapped as
   `Shape.Einsum_tern (logic, [])` (mirroring the binary case at the
   sibling branch).
5. **A pointwise primitive for the underlying scalar op.** This is the one
   subtlety. The existing `Ternop` primitives are `fma(a,b,c)=a*b+c` and
   `where(c,a,b)`. Neither cleanly expresses pure ternary contraction
   (which wants `a*b*c` accumulated via `+=`). Two options:

   - **(a)** Add a new ternary scalar op `Mul3` in `arrayjit/lib/ops.ml`
     and emit it from the new `einsum3` operation builder. This is the
     orthogonal, principled choice.
   - **(b)** Decompose ternary einsum at `operation.ml` into a chained
     binary einsum `(a *+ spec12 b) *+ spec23 c`, where the splits are
     derived from the parsed spec. The PPX still routes through the new
     ternary einsum shape-inference path (so projections are unified
     across all three operands at the shape level), but execution uses
     two binary kernels. Skips the scalar-op work but introduces an
     intermediate buffer.

   The proposal accepts either option. Recommendation: start with (a) for
   single-kernel execution and to keep the operation orthogonal; fall back
   to (b) only if backend codegen for `Mul3` proves disproportionate.

6. **Gradient**: each gradient of a ternary einsum is itself a ternary or
   binary einsum (e.g. `d/dA(A*B*C contracted)` is `B*C` contracted with the
   incoming gradient). Implement in `operation.ml` alongside `einsum`.

### Code pointers

- Blocker dispatch: `tensor/ppx_cd.ml`, function processing
  `accu_op lhs (tern_op (rhs1,rhs2,rhs3) ~logic)` — search for the
  `"einsum notation for ternary operators not supported yet"` string.
- Binary einsum logic-dispatch reference: same file, the analogous
  `process_raw_binop` dispatch ~30 lines above.
- Binary einsum shape constraints: `tensor/shape.ml`, branch
  `Broadcast (Einsum (spec, dim_refs), sh1, sh2)` — ~140 lines, marked
  with refactor TODO.
- Ternary shape constraints (existing): `tensor/shape.ml`, branches
  `Broadcast_tern (Compose_accumulate, ...)` and
  `Broadcast_tern (Pointwise_tern, ...)`.
- `compose_type` / `ternary_type` definitions: `tensor/shape.mli`.
- Ternary scalar ops registry: `tensor/ppx_cd.ml` `ternary_ops` Hashtbl
  (currently `where`, `fma`).
- Existing ternary tensor ops: `tensor/operation.ml`, `fma`, `where`.
- Binary einsum API: `tensor/operation.ml`, `einsum`, `pointmul`,
  `outer_sum` (search for `Shape.Einsum (spec, capture_dims)`).
- Tests to consult for the testing pattern: `test/ppx/test_ppx_op.ml`
  (uses `+*` infix einsum), `test/training/mlp_names.ml`.

## Approach

*Suggested approach — agents may deviate if they find a better path.*

1. Add `Einsum_tern of string * delayed_var_ref list` to `Shape.ternary_type`
   in `shape.mli` / `shape.ml`, with a string-form representative for
   pretty-printing similar to existing variants.
2. In `shape.ml`, add a `Broadcast_tern (Einsum_tern (spec, dim_refs), sh1,
   sh2, sh3)` branch to constraint generation. Implementer's discretion
   whether to copy-adapt the binary branch or refactor binary+ternary onto
   a shared helper that takes a `parsed_axis_labels list`. Refactor is
   preferred because the binary branch already carries a TODO for it; the
   refactor delivers ternary "for free" and pays down debt.
3. Add a `Broadcast_tern (Einsum_tern ..., ...)` branch to
   `derive_projections` (same file). Reuse the binary projection-derivation
   logic generalized to N RHS shapes.
4. In `ppx_cd.ml`, replace the error in the ternary-spec dispatch with the
   binary-style `Shape.Einsum_tern (logic, [])` wrapping (drop the `#305`
   reference from any remaining diagnostic).
5. Pick option (a) or (b) from Context §"What is missing" item 5. If (a):
   add `Mul3` (or similar) to `Ir.Ops.ternop` and supply the codegen for
   relevant backends (CPU C and CUDA at minimum; backends are listed in
   `arrayjit/lib/`).
6. Add `einsum3 spec t1 t2 t3` (or three-operand `*+`) in `operation.ml`,
   returning a tensor with the right `op_asn` and `grad_asn` per option (a)
   or (b).
7. Tests in `test/ppx/`:
   - `"ij;jk;km=>im"` matches the binary chain numerically.
   - `"bij;bjk;bkm=>bim"` batched.
   - Gradient agreement test against the binary chain.
   - At least one test that exercises `%cd` with a ternary einsum spec
     directly (verifying the PPX dispatch fix).
8. Run the existing test suite to confirm no regressions in binary einsum
   or `fma`/`where`.

## Scope

**In scope**:
- Single new ternary einsum variant in `Shape.ternary_type`.
- Shape inference, projection inference for 3-RHS einsum.
- PPX dispatch fix.
- One operation-level entry point with gradient.
- A minimal scalar primitive (new `Mul3` or chained-binary fallback).
- Tests covering the falsifier scenario (chain contraction equality) and
  gradient agreement.

**Out of scope**:
- N-ary (4+) einsum. The codepaths added should not preclude future
  generalization, but no N-ary work is performed here.
- New surface syntax beyond the existing `;`-separator convention.
- Refactoring binary/unary einsum infrastructure beyond what is needed to
  avoid copy-pasting the constraint branch a third time. (If the
  refactor becomes large, ship the duplication and file a follow-up.)
- Backend optimization of the new scalar op beyond functional correctness.

**Dependencies**: none.

**Effort note**: The original "small (1-2 days)" estimate assumed only the
PPX error needed removal. Verified investigation shows `Shape.ternary_type`
needs a new variant plus a new ~100-line shape-constraint branch (or a
refactor of the binary branch), `derive_projections` needs a new branch, and
a new scalar primitive plus its codegen are required for option (a). A more
realistic estimate is **medium (3-5 days)**, dominated by the shape.ml work
and tests. If the refactor is taken, the ceiling rises but unary/binary
einsum benefit. If the implementer prefers option (b) (decomposition into
chained binary einsums) they may be able to keep this closer to small, at
the cost of an intermediate buffer per call.
