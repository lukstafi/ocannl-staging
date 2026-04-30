# Prohibit `Compose` shape logic for `/` and `**` (and other non-multiply binary ops)

## Goal

Resolve [issue #192](https://github.com/ahrefs/ocannl/issues/192): division
(`/`) and to-power-of (`**`) combined with the `Compose` composition type
(`~logic:"@"`) currently compile successfully via the `%cd` ppx but produce
mathematically meaningless results — silently. Either the semantics is
broken (e.g., `=:+ a / b ~logic:"@"` computes `lhs[i,j] := sum_k a[i,k] / b[k,j]`,
not matrix inverse) or the combination should be prohibited. Per the issue
body, expanding as matrix pseudo-inverse is overkill, so we prohibit.

## Acceptance Criteria

- [ ] Using `~logic:"@"` (i.e. `Compose`) with any binary primitive op other
      than `Mul` produces a clear compile-time error from the `%cd` ppx.
      The error names the offending op and points the user at pointwise
      (`~logic:"."`) or einsum notation as alternatives.
- [ ] Using `~logic:"@"` with `*` / `mul` continues to work as before
      (matrix multiply / tensor contraction). No regression in existing
      tests, examples, or `docs/slides-*.md` snippets that use
      `=:+ a * b ~logic:"@"`.
- [ ] At least one positive test (or expect-test) demonstrates that
      `=:+ a * b ~logic:"@"` still compiles.
- [ ] At least one negative test demonstrates that
      `=:+ a / b ~logic:"@"` and `=:+ a ** b ~logic:"@"` produce a
      compile-time error with a useful message. (Negative tests can use
      a `dune` `cram` test, an expect-test capturing a ppx error, or a
      manually-checked `.expected` showing the error — pick whichever
      style the project already uses for ppx error tests.)
- [ ] The ternary `Compose_accumulate` path (used by `fma`) is left
      untouched — it is a real, intentional use of compose-style
      contraction.
- [ ] If existing code in `tensor/`, `lib/`, `bin/`, `test/`, or the
      example slides uses `~logic:"@"` with `/`, `**`, or another
      non-multiply binary op, it is either fixed (switching to `~logic:"."`
      or einsum) or, if intentional and meaningful, surfaced for user
      decision before the prohibition lands.

## Context

### Where `Compose` enters from user code

The `%cd` ppx (`tensor/ppx_cd.ml`) parses an expression of the form
`accu_op lhs (bin_op rhs1 rhs2 ~logic:"<spec>")` and turns the spec into
a `Shape.compose_type`:

```ocaml
if String.equal spec "." then [%expr Shape.Pointwise_bin]
else if String.equal spec "@" then [%expr Shape.Compose]
else [%expr Shape.Einsum ([%e logic], [])]
```

This branch is reached from two duplicated patterns (curried and tupled
`bin_op` application forms) — both should get the new check. The lookup
`binary_op bin_op` resolves `bin_op` to a primitive `Ir.Ops.*` constructor
via the `binary_ops` table in `tensor/ppx_shared.ml`. We need to inspect
which primitive op is being used and reject `Compose` for everything that
is not `Mul`.

### Why the bug is silent today

`tensor/operation.ml`'s top-level helpers `pointpow` (`**.`) and `pointdiv`
(`/.`) are hardwired to `compose_op_of_spec`, which only produces
`Pointwise_bin` or `Einsum` — so the public OCaml-level operators cannot
reach `Compose`. The hole is the `%cd` DSL, where `~logic:"@"` is accepted
for *any* entry in `binary_ops`, including `/`, `**`, `<`, `||`, `max`,
`min`, etc.

Once the `Compose` shape is constructed, `Shape.get_inequalities` (the
`Broadcast (Compose, sh1, sh2)` branch in `tensor/shape.ml`) emits the
matrix-multiply-style row inequalities — pairing `sh1.input` with
`sh2.output` for contraction — and `arrayjit/lib/assignments.ml` lowers
the resulting projections by iterating over `product_space` and applying
the chosen op. With `=:+` (initialize-neutral, `Add` accumulator) and
`Div`, the lowering computes `sum_k a[i,k] / b[k,j]`. There is no
warning, no error, no documentation; it just runs.

There is precedent for this kind of compile-time guard: the `*` entry in
`binary_ops` already raises a ppx error if no `~logic` is given
("No default compose type for binary `*`, try ..."), and the `/` entry
does the same. The proposed check extends the same style of guard to the
case where `~logic:"@"` is supplied for an op that has no `Compose`
meaning.

### Key code pointers

- `tensor/ppx_cd.ml` — the two `Pexp_constant (Pconst_string (spec, ...))`
  branches that translate `~logic:"@"` to `Shape.Compose` for binary ops
  (search for `Shape.Compose`). The ternary branch right below them
  handles `fma` / `Compose_accumulate` and should be left alone.
- `tensor/ppx_shared.ml` — `binary_ops` table; the `Ir.Ops.*` constructor
  associated with each ident is what the check needs to inspect.
- `tensor/shape.ml` / `tensor/shape.mli` — `compose_type` definition and
  `Broadcast (Compose, ...)` shape-inference branch.
- `tensor/operation.ml` — `pointpow`, `pointdiv`, `pointmul`, `matmul`;
  shows that the user-facing OCaml operators already restrict themselves
  to pointwise/einsum and never reach `Compose` from this path.
- `arrayjit/lib/assignments.ml` — `loop_accum` etc.; consumes the
  projections produced by shape inference and applies the op blindly.

### Related

- Issue #305 (tracked in the existing ppx error message about ternary
  einsum notation) is unrelated but lives next to the code we touch.
- The existing `*` and `/` "no default compose" errors in
  `tensor/ppx_shared.ml` are the model for the new error wording.

## Approach (optional)

*Suggested approach — agents may deviate if they find a better path.*

In `tensor/ppx_cd.ml`, in the binary branch that maps `~logic:"@"` to
`Shape.Compose`, check the resolved primitive op (returned by
`binary_op bin_op`) and emit a compile-time error when the op is anything
other than `Ir.Ops.Mul`. The simplest implementation is a small allow-list
match on the `bin_op` *string* (before calling `binary_op`), since at ppx
time we have the source identifier (`*`, `mul`, `/`, `**`, etc.) directly
and don't need to inspect the AST of the resolved expression. Allowed:
`*` and `mul`. Everything else with `~logic:"@"` → ppx error like:

> ppx_ocannl %cd: `~logic:"@"` (Compose) is only meaningful for tensor
> multiplication; use `~logic:"."` for pointwise `<op>`, or einsum
> notation for a custom contraction.

Apply the same check in both duplicated patterns (curried and tupled
forms). Leave the einsum (`else`) branch alone — users who know what
they want via einsum can express it.

A defensive, lower-priority improvement: also raise from
`Shape.get_inequalities` if a `Broadcast (Compose, _, _)` is paired with
a non-`Mul` accumulator op at the assignments level. This catches code
paths that bypass the ppx (none currently exist in-tree, but it
future-proofs library users who construct shapes by hand). This is
optional and can be deferred.

## Scope

In scope:
- Compile-time guard in `tensor/ppx_cd.ml` for binary `~logic:"@"`.
- Tests covering the positive (multiply) and negative (div, pow, and at
  least one other non-multiply op) cases.
- A short note in `docs/syntax_extensions.md` clarifying that `~logic:"@"`
  is reserved for multiplication.
- Audit of in-tree uses of `~logic:"@"` (a single `grep` will do) to
  confirm no fix-ups are needed elsewhere.

Out of scope:
- Implementing matrix inverse / pseudo-inverse for `/` with `Compose`.
- Reworking the `compose_type` algebra or adding new shape logics.
- Touching the ternary `Compose_accumulate` path.
- Changing the public `pointdiv` / `pointpow` operators in
  `tensor/operation.ml` — they already correctly avoid `Compose`.

Dependencies: none. This is a self-contained ppx + tests change.
