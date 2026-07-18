# gh-490: Symbolic extents — v1: symbolic dimensions in shape inference

Issue gh-490 proposes launch-time-bound extents: specific axes (batch, sequence) stay symbolic
through lowering and codegen, becoming runtime kernel parameters, so one compile + one tune-cache
entry serves every extent up to a maximum. This document covers the **first increment**: plugging
static-indexing symbols into shape representations and inference, and reconciling symbolic shapes
with Tnode dims. Lowering, codegen, launch binding, and autotune identity are follow-up tasks (see
the issue's task list).

## Design

### Representation: a new `Row.dim` constructor

```ocaml
| Sym of sym_dim
and sym_dim =
  { sym : Ir.Indexing.static_symbol; sym_basis : string; sym_proj_id : proj_id option }
```

A symbolic dimension reuses the **existing static-indexing binding mechanism**: the symbol is an
ordinary `Ir.Indexing.static_symbol` (as threaded through `bindings` and bound per launch). Its
declared `static_range` is mandatory (and must be positive): it is the **maximum extent**, used to
size allocations and, until launch-time binding lands, to materialize the axis everywhere
downstream of inference. `sym_basis` and `sym_proj_id` mirror `solved_dim`'s fields so the basis
discipline and projection inference extend mechanically.

Note the range convention: for slicing (`@|`) the bound value is an index in `[0, range)`; for an
extent the bound value will be a size in `[0, range]` (buffers are sized `range`). Bind-time
validation of extent parameters is part of the backend-plumbing follow-up.

### Inference semantics: a rigid constant (universal variable)

`Sym s` behaves as a minted constant:

- **Equality** (`unify_dim`): `Sym s = Sym s` (same symbol; bases must agree, mirroring `Dim`).
  `Sym` vs. a different `Sym`, or vs. any concrete `Dim` — even one equal to its range — is a
  `Shape_error` mismatch. `Var v = Sym s` solves `v := Sym s`.
- **Broadcasting** (`solve_dim_ineq`, row GLB merge): the broadcast top `1_(bcast_if_1)` broadcasts
  *to* a symbolic axis (scalars compose with symbolic-extent tensors); a `Sym` never broadcasts
  itself. Two incompatible rigid bounds flowing into the same variable demote the bound to the
  broadcast top, exactly as differing concrete `Dim`s do.
- **Arithmetic is off-limits (v1)**: any `Total_elems` constraint that would need the symbolic
  axis's size — reshape-style totals, `set_dim` on rows containing it, strided-var substitutions,
  divided-by factors — fails fast with "Total_elems constraints are not supported with symbolic
  dimensions" (`Row.fail_if_total_elems_over_sym` at the reduction/substitution choke points). We
  deliberately refrain from a `Sym_denom`-style solver extension in v1. Similarly, convolutions /
  affine transforms over a symbolic axis, and `Concat` axes equated with a symbolic extent, are
  rejected (structural cancellation of equal `Sym` components inside `Concat`-vs-`Concat` still
  works).
  - Consequence: **random-init parameters cannot have symbolic axes in v1** — `uniform ()` etc.
    route the threefry counter mapping through a `Total_elems` (uint4x32-blocks = numel/4)
    constraint. Constant-init and data terminals are fine. Lifting this (the RNG constraint only
    needs the *materialized* numel, which is well-defined) is a natural follow-up.

### Reconciliation with Tnode dims: materialize at the maximum extent

`Shape.to_dims` (→ `Tnode.dims`) maps `Sym s` to its range: allocation, host transfers, and numel
contracts all see the max extent. Projection inference (`Row.solved_dim_of_sym`) does the same, so
product spaces, iterators, and generated loops run over the full range. In this increment a
symbolic-dim program is therefore **semantically identical to writing the max extent**, with the
symbol tracked end-to-end through inference; the follow-up tasks swap the materialization seam for
a launch-time parameter without touching the solver again.

### API: `Shape.set_sym_dim`

Symbols plug in through the existing einsum capture mechanism, mirroring `Shape.set_dim`:

```ocaml
let seq, bindings = Idx.get_static_symbol ~static_range:512 bindings in
let%op y = x ++ "b s d => b s d" [ "s" ] in
Shape.set_sym_dim s seq
```

`delayed_var_ref` gains `mutable solved_sym : static_symbol option`; the einsum re-binding path
(`bind_delayed_vars_to_envs`) re-emits `Var v = Sym sym` equations for symbolic refs the same way
it re-emits concrete `set_dim` values, so a captured ref can be reused across specs. `set_equal`
propagates symbolic settings; `set_scale` and row-variable refs reject them. Conflicts (concrete
then symbolic, symbolic then different symbol) raise `Shape_error`.

Solutions are *not* captured back into `var_ref.solved_dim` (it is an `int option`); a var solved
to a symbolic dim by inference leaves the ref unresolved. Capturing into `solved_sym` is a possible
refinement.

## Testing

`test/einsum/test_symbolic_extents.ml`: happy paths (capture + solve + materialize at range,
whole-buffer forward and reduction executed on the default backend, same-symbol unification across
tensors, scalar broadcast) and error paths (missing range, set_dim/set_sym_dim conflicts,
symbolic-vs-concrete solver mismatch, distinct symbols, row-var rejection, Total_elems fail-fast).

## Follow-ups (rest of gh-490)

1. Keep `Sym` through `Low_level`: bounded-symbol loop extents, simplifier guards.
2. Backend param plumbing + launch binding (cc/Metal first), extent-range validation (`v <= range`).
3. Autotune cache identity `(digest, tuning-size)` with bucketing.
4. RNG init over symbolic axes via materialized numel; capturing `solved_sym`; possibly `Sym`
   support in `Total_elems` (`Sym_denom`).
