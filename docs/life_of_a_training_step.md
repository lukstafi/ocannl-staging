# Life of a Training Step: OCANNL's Compilation Pipeline End to End

This tutorial walks one small program through the whole compilation stack — from a `%op`
expression to machine code executing on a device — naming every stage, the source files and
functions that implement it, and the debug artifact it leaves behind. It is the connective
tissue between the deep-dive documents, each of which covers one segment:

- [Tensors and Contexts](tensors_and_contexts.md) — the user-facing objects (stage 0 and stage 7 here).
- [Syntax extensions](syntax_extensions.md) — what `%op`/`%cd` expand to.
- [Shape inference](shape_inference.md) — the constraint system behind projections (stage 2).
- [Lowering and inlining](lowering_and_inlining.md) — the `Low_level` optimization pipeline (stages 3–4).
- [Schedules and autotuning](schedules_and_autotuning.md) — the loop-transform layer and its search (stage 5).
- [The compilation manifesto](compilation_manifesto.md) — why the pipeline is shaped this way.

Like those documents, this one references code by module, function, and constructor name
rather than line number, so it stays valid as files evolve. Quoted artifacts were captured in
August 2026; regenerate them with the companion program below if they drift.

## The pipeline at a glance

```
      %op / %cd user code                                  tensor/operation.ml, ppx
              │
              ▼
 0.  Tensor.t construction: each tensor carries its        tensor/tensor.ml (Tensor.op)
     forward and backprop code as Assignments.comp
              │
              ▼
 1.  A whole-step comp: Train.grad_update sequences        lib/train.ml
     forward + zero-grads + backprop (+ optimizer)
              │
              ▼          Context.compile ctx comp bindings
              ▼
 2.  Shape inference completes; projections derived        tensor/shape.ml, row.ml
     (forced by lowering through the lazy in Accum_op)
              │
              ▼
 3.  Lowering: Assignments.to_low_level builds loop        arrayjit/lib/assignments.ml
     nests from projections → Low_level.t          [.cd, -unoptimized.ll artifacts]
              │
              ▼
 4.  Backend-independent optimization: analysis,           arrayjit/lib/low_level.ml
     placement, virtualization/inlining, simplify,
     CSE → Low_level.optimized                     [.ll artifact]
              │
              ▼
 5.  Schedules: default parallelization annotations        arrayjit/lib/schedule.ml
     and kernel fission, or explicit/autotuned
     transforms at the lowered_transform seam
              │
              ▼
 6.  Codegen: C_syntax renders each kernel as C /          arrayjit/lib/c_syntax.ml,
     CUDA C++ / MSL; backend compiles it           cc_backend.ml, cuda_backend.ml,
     (cc → dlopen, nvrtc → PTX, MSL → metallib)    metal_backend.ml  [.c/.cu/.metal]
              │
              ▼
 7.  Link and run: buffers allocated into a child          arrayjit/lib/backends.ml,
     context, kernels become a Task.t; Context.run         context.ml, task.ml
     dispatches it on the device's stream
```

Stages 2–7 all happen inside one call to `Context.compile` (+ `Context.run`); stages 0–1 are
pure code construction that happens as your OCaml program builds tensor expressions.

## The companion program

Everything below refers to this program — a dense layer with a relu and a sum-everything
loss, deliberately tiny so every artifact fits on a screen:

```ocaml
(* tutorial_example.ml *)
open Base
open Stdio
open Ocannl
module IDX = Train.IDX
open Ocannl.Operation.DSL_modules

let () =
  let ctx = Context.auto () in
  (* A fixed input: batch of 2, 3 features. *)
  let x =
    TDSL.ndarray
      [| 1.; 2.; 3.; 4.; 5.; 6. |]
      ~label:[ "x" ] ~batch_dims:[ 2 ] ~output_dims:[ 3 ] ()
  in
  (* One dense layer with a relu, and a sum-everything loss. *)
  let%op y = relu (({ w; o = [ 2 ] }) * x + { b }) in
  let%op loss = y ++ "...|... => 0" in
  (* Build the forward + backprop assignments, then compile them as one routine. *)
  let update = Train.grad_update loss in
  let ctx = Train.init_params ctx IDX.empty loss in
  let ctx, routine = Context.compile ctx update IDX.empty in
  let ctx = Context.run ctx routine in
  printf "loss = %.4f\n" (Context.get_values ctx loss.Tensor.value).(0);
  Train.printf_tree ctx ~with_grad:true loss
```

Build it against the `ocannl` library and run it with debug artifacts enabled:

```bash
OCANNL_BACKEND=cc OCANNL_OUTPUT_DEBUG_FILES_IN_BUILD_DIRECTORY=true ./tutorial_example.exe
```

The run prints `loss = 0.6590` and leaves, under `build_files/tutorial_example/`, four files
per compiled routine — the artifacts quoted throughout this tutorial:

| file | stage | contents |
|---|---|---|
| `<routine>.cd` | 1 | the `Assignments.comp`, printed in `%cd` syntax |
| `<routine>-unoptimized.ll` | 3 | the freshly lowered `Low_level.t` |
| `<routine>.ll` | 4 | the optimized `Low_level.t` |
| `<routine>.c` / `.cu` / `.metal` | 6 | the generated backend source |

Two routines get compiled: `init_params_for_loss` (parameter initialization; see
`Train.init_params` below) and `loss_forward_and_gradient_update` — the training step, our
protagonist. One detail worth savoring before we start: the same program run with
`OCANNL_BACKEND=metal` prints the same `loss = 0.6590` — bitwise determinism across backends
is a design invariant (see the [manifesto](compilation_manifesto.md), §4), not luck.

## Stage 0: tensors carry their code

*Code: `tensor/tensor.ml` (`Tensor.op`, `binop`/`unop`/`term`), `tensor/operation.ml`.*

OCANNL has no separate "tracing" or "graph capture" step: a `Tensor.t` *is* the graph. Each
tensor records, alongside its `value : Tnode.t` and `shape : Shape.t`:

- `forward : Assignments.comp` — the code that computes this tensor *and all subtensors it
  consumed*;
- `diff : diff option`, where `diff` holds `grad : Tnode.t`, `zero_grads : Assignments.t`,
  and `backprop : Assignments.comp`.

Every operation — `+`, `*`, `relu`, einsum — funnels into the internal constructor
`Tensor.op`, parameterized by two code-building closures written in `%cd` syntax in
`tensor/operation.ml`. For addition:

```ocaml
let%cd op_asn ~t ~t1 ~t2 ~projections = v =:+ v1 + v2 in
let%cd grad_asn ~t:_ ~g ~t1 ~t2 ~projections = g1 =+ g; g2 =+ g in
...
```

`op_asn` emits the forward assignment, `grad_asn` the gradient accumulations. `Tensor.op`
sequences the operands' `forward` comps followed by this operation's `op_asn` (an
`Assignments.sequence`), and prepends this operation's `grad_asn` to the operands' backprop
comps — backprop code accumulates in reverse construction order, with a topological
re-ordering pass (gh-461) making sure a fragment that accumulates into a gradient runs before
the fragment that embeds that node's backprop code.

Two bookkeeping devices matter downstream:

- **Forward roots.** `session_state.forward_roots` tracks tensors whose forward code has not
  yet been consumed by a consumer expression. When `y`'s construction consumes `x`, `x` stops
  being a root and its code is spliced into `y.forward`. `Tensor.consume_forward_code` /
  `consume_backprop_code` are the explicit consumption points used by `Train`.
- **Embedded nodes.** `Assignments.comp` is not bare code: it pairs `asgns : Assignments.t`
  with `embedded_nodes : Set.M(Tnode).t` — the nodes this comp *owns* (creates and computes).
  Any node mentioned by the code but not embedded must already exist in the context the comp
  is compiled against. This ownership set is what later lets the compile step distinguish
  "allocate this here" from "expect this from a parent context" (`from_prior_context` in
  `arrayjit/lib/backends.ml`).

The `Assignments.t` type itself (`arrayjit/lib/assignments.ml`) is a short imperative
language of *accumulating assignments*:

```ocaml
type t =
  | Noop
  | Seq of t * t
  | Block_comment of string * t
  | Accum_op of { initialize_neutral : bool; accum : Ops.binop; lhs : Tn.t;
                  rhs : accum_rhs;                    (* Ternop/Binop/Unop/Block/Rev_sides *)
                  projections : Indexing.projections Lazy.t;
                  projections_debug : string }
  | Set_vec_unop of { ... }                           (* vectorized fills, e.g. threefry *)
  | Fetch of { array : Tn.t; fetch_op : fetch_op; dims : int array Lazy.t }
```

An `Accum_op` says: accumulate (`accum` ∈ `Add`, `Max`, ... — `=:+` means "initialize
neutral, then add-accumulate") the right-hand-side expression into `lhs`, iterating over the
loop structure described by `projections` — a *lazy* value, and that laziness is the hinge of
stage 2. `Fetch` initializes a node from a constant, a fill array, a random-seed embed, etc.
`Block_comment` is not cosmetic: `Assignments.get_name_exn` derives the routine's name (and
hence its artifact file names and kernel names) from the outermost block comment.

## Stage 1: one comp for the whole step

*Code: `lib/train.ml` (`grad_update`, `sgd_update`, `init_params`, `to_routine`).*

`Train.grad_update loss` builds the entire training step as a single comp — this is OCANNL's
"schedule the step, not the kernel" decision (manifesto §1). It is itself just a `%cd`
quotation: consume `loss`'s forward code, sequence the gradient zeroing, seed
`loss.grad =: 1`, and splice `loss`'s backprop code, all wrapped in named block comments. An
optimizer step (`Train.sgd_update`, also plain `%cd` code built per parameter) can be
sequenced into the same comp — forward, backward, and update then compile as one program, and
the optimizer's reads of gradients are ordinary dataflow visible to every later optimization.

The `.cd` artifact shows the result for our example — this is the *entire* high-level
program, pretty-printed by `Assignments.to_doc`:

```
loss_forward_and_gradient_update (): # "loss forward and gradient update";
  x =: constant_fill([1., 2., 3., 4., 5., 6.]);
  n22 =:+ w * x ~logic:"@";
  n24 =:+ n22 + b;
  relu_y =:+ relu n24;
  loss =:+ relu_y ~logic:"...|... => 0";
  # "loss zero grads and backprop";
  b.grad =: 0.;
  w.grad =: 0.;
  n22.grad =: 0.;
  n24.grad =: 0.;
  relu_y.grad =: 0.;
  loss.grad =: 0.;
  _1 =: 1.;
  loss.grad =: _1;
  relu_y.grad =+ loss.grad ~logic:"...|... => 0 [lhs←rhs1, rhs1←lhs]";
  n24.grad =+ n24 -?/ relu_y.grad ~logic:". [lhs←rhs1, rhs2←lhs]";
  n22.grad =+ n24.grad ~logic:". [lhs←rhs1, rhs1←lhs]";
  b.grad =+ n24.grad ~logic:". [lhs←rhs2, rhs1←lhs]";
  w.grad =+ n22.grad * x ~logic:"@ [lhs←rhs1, rhs1←lhs]";
```

Reading it: `n22`, `n24`, `relu_y` are the anonymous intermediates (`w * x`, `+ b`, the
relu); `~logic` records each assignment's projection spec — `"@"` is matrix multiply,
`"...|... => 0"` the sum-reduce einsum from the source, `.` pointwise; the bracketed suffixes
on backprop assignments record how the gradient projection was derived from the forward one
(swap lhs with the rhs being differentiated). `-?/` is `relu_gate`. Note what is *absent*:
shapes. Nothing here has committed to dimensions yet.

Parameter initialization is a separate comp compiled the same way: `Train.init_params`
gathers the `forward` code of `loss.params` (the transitively-reachable parameters — a set
each tensor also carries) into an `init_params_for_loss` routine. Its `.cd` artifact is a
nice bonus read: OCANNL's counter-based PRNG means `w`'s "uniform init" is itself ordinary
compiled code (`threefry4x32` applied to `range_over_offsets`, then bit-converted and
affine-mapped), not a host-side RNG call.

`Train`'s convenience wrappers (`to_routine`, `forward_once`, `update_once`, `run_once`) all
reduce to: build a comp as above, then call `Context.compile` — the door to everything below.

## Stage 2: shape inference completes, projections appear

*Code: `tensor/shape.ml` (`propagate_shapes`, `finish_inference`, `derive_projections`),
`tensor/row.ml` (the solver); consumed via `arrayjit/lib/indexing.ml`.*

While tensors were being constructed, each operation eagerly registered its shape constraints
(`Shape.propagate_shapes`, called from `Tensor.op` and `Tensor.raw_binop`/`raw_unop`) and ran
the cheap first solver stage. But nothing forced final answers: the `projections` field of
every `Accum_op` is a `lazy` closing over that operation's `Shape.update_step`.

The forcing happens when compilation begins. `Assignments.to_low_level` does
`Lazy.force projections` on the first `Accum_op` it lowers; the lazy calls
`Shape.get_projections`, which calls `Shape.finish_inference` — and *that* runs the
remaining solver stages over all constraints registered so far, resolves every remaining row
and dimension variable (unsolved ones become an error or default per the solver's rules),
then calls `Shape.derive_projections` for each pending update step. This is what "shape
inference completion is forced by lowering" means concretely; you can watch it in any
backtrace of a shape error raised from inside `Context.compile` — the frames run from
`Assignments.to_low_level` through `CamlinternalLazy.force` into `Shape.finish_inference`
and `Row.solve_inequalities`.

What pops out, per assignment, is an `Indexing.projections` record
(`arrayjit/lib/indexing.ml`) — the sole interface between shape inference and code
generation. Conceptually it answers: *what is the iteration space of this assignment, and how
does each tensor index into it?*

```ocaml
type projections = {
  product_space : int list array;        (* the iteration space, one entry per loop *)
  product_iterators : symbol list array;  (* its iterator symbols (lists ≠ singleton
                                             only for concatenation components) *)
  lhs_dims : int array;
  rhs_dims : int array array;
  project_lhs : axis_index array;        (* product-space point → LHS index, per axis *)
  project_rhs : axis_index array array;  (* same per RHS argument *)
  extent_syms : ...; debug_info : ...;
}
```

where `axis_index` is `Fixed_idx of int` (pinned positions, and any non-iterated dim-1
axis), `Iterator of symbol` (the common case; also how a static-symbol batch slice reads),
`Affine of { symbols : (int * symbol) list; offset : int }` (convolutions:
`stride·i + dilation·k − padding`), `Sub_axis` (member of a multi-axis flattened access),
and `Concat of symbol list` (concatenated axes; eliminated during lowering).

For our matmul assignment `n22 =:+ w * x ~logic:"@"`, with `n22 : 2×2`, `w : 2×3`,
`x : 2×3`, the derived record is (symbols renamed for readability):

```
product_space     = [| [2]; [2]; [3] |]        (* batch row i, output col j, input k *)
product_iterators = [| [i]; [j]; [k] |]
project_lhs       = [| Iterator i; Iterator j |]                        (* n22[i, j] *)
project_rhs       = [| [| Iterator j; Iterator k |];                    (* w[j, k]   *)
                       [| Iterator i; Iterator k |] |]                  (* x[i, k]   *)
```

Reductions are implicit: an axis that appears in the product space but not in `project_lhs`
(here `k`) is a reduction axis — that is all a reduction is, which is why
`initialize_neutral` plus the accumulation operator fully determine the loop's semantics.
This representation is also why OCANNL has no layout/reshape subsystem: every access
pattern — transposes, broadcasts, convolutions, slicing — is index arithmetic in
`project_*`, and einsum projections are the sole loop-nest generator.

Because the whole training step lowers at once, one `finish_inference` serves all
assignments; einsum specs, convolution strides (`Affine` indices with
`stride·i + dilation·j − padding` symbol combinations), and broadcast row variables have all
been settled by the same constraint solve. See [shape_inference.md](shape_inference.md) for
the solver itself and [syntax_extensions.md](syntax_extensions.md) for the einsum notation
that generates the constraints.

## Stage 3: lowering to Low_level

*Code: `arrayjit/lib/assignments.ml` (`lower`, `to_low_level`).*

`Assignments.lower` (called from `Backends.lower_assignments`,
`arrayjit/lib/backends.ml`) translates the comp into `Low_level.t` — a C-like mini-language
of nested `For_loop`s over scalar `Set`/`Get` operations (the full grammar is at the top of
[lowering_and_inlining.md](lowering_and_inlining.md)). The recipe for each `Accum_op`:

1. every entry of `projections.product_space` becomes a `For_loop` with a **fresh** symbol
   (product iterators may be shared between operations, so lowering α-renames; the
   substitution also rewrites symbols inside `Affine` indices);
2. the loop body is a `Set` of `lhs` at `project_lhs`, whose right-hand side combines
   `accum` with the `Get`s of the rhs buffers at their `project_rhs` indices — unless the
   projection is provably injective (`Affine.is_injective`), in which case the
   read-modify-write collapses to a plain store;
3. if `initialize_neutral` is set and the projection isn't surjective-and-injective, a
   `Zero_out` (or neutral-element fill) precedes the nest;
4. concatenation axes (`Concat` indices, from `Block`/`Rev_sides` assignments) are
   eliminated here, by emitting one loop nest per component in sequence.

The `-unoptimized.ll` artifact for our example (excerpt):

```
  x[0, 0] := 1.0;
  ...
  x[1, 2] := 6.0;
  zero_out n22;
  for i32 = 0 to 1 {
    for i33 = 0 to 1 {
      for i34 = 0 to 2 {
        n22[i32, i33] := (n22[i32, i33] + (w[i33, i34] * x[i32, i34]));
      }
    }
  }
  for i35 = 0 to 1 {
    for i36 = 0 to 1 { n24[i35, i36] := (n22[i35, i36] + b[i36]); }
  }
  ...
  for i50 = 0 to 1 {
    for i51 = 0 to 1 {
      for i52 = 0 to 2 {
        w.grad[i51, i52] := (w.grad[i51, i52] + (n22.grad[i50, i51] * x[i50, i52]));
      }
    }
  }
```

Everything is now concrete: shapes are integers, the matmul is three nested loops, every
intermediate is written to what is still nominally its own array. The `Constant_fill` fetch
of `x` unrolled into six scalar stores. This is the maximally naive program — stage 4 exists
to un-naive it.

## Stage 4: backend-independent optimization

*Code: `arrayjit/lib/low_level.ml` (`optimize` = `analyze_proc` + `specialize_proc`); the
per-pass detail is [lowering_and_inlining.md](lowering_and_inlining.md).*

`Low_level.optimize` runs the pipeline summarized as:

```
analyze_proc:      trace_node_facts   (structural facts per tensor node)
                   affine access metrics (exact read-multiplicity/coverage queries)
specialize_proc:   decide_placements  (virtual? local? on-device? — per compile lineage)
                   virtual_llc + inline_computation   (inlining, with legality checks)
                   cleanup_virtual_llc  (drop dead materialized writes)
                   simplify_llc       (constant folding, algebra, FMA)
                   rewrite_one_hot_reductions
                   eliminate_common_subexpressions + hoist_cross_statement_cse
```

The analysis/specialization split (gh-555) is what makes search affordable: the expensive
decision-independent analysis is computed once per routine (and memoized in a process-global
cache keyed by a structural digest), while placement decisions replay cheaply per candidate.
The **default state of every tensor node is `Virtual`** — no bytes anywhere, consumers
recompute the defining expression inline — and materialization must be earned (visit-count
and recompute-cost caps) or forced (user intent, observability). Decisions land in
`Placements` on the `optimize_ctx`, which is *forked per compile* (`copy_optimize_ctx` in
`Backends.lower_assignments`): the same tensor can be virtual in one routine and on-device in
another.

The optimized `.ll` for our example repays close reading:

```
  zero_out loss;
  for i39 = 0 to 1 {
    for i40 = 0 to 1 {
      loss[i39, 0] := (loss[i39, 0] + relu((v27_n22 {
        v27_n22 := 0.0;
        for i57 = 0 to 2 {
          v27_n22 := fma(w[i40, i57], x[i39, i57], v27_n22);
        }
      } + b[i40])));
    }
  }
  /* loss zero grads and backprop */
  zero_out b.grad;
  zero_out w.grad;
  zero_out n22.grad;
  for i46 = 0 to 1 {
    for i47 = 0 to 1 {
      n22.grad[i46, i47] := (n22.grad[i46, i47] + relu_gate((v27_n22 { ... }
        + b[i47]), 1.0));
    }
  }
  ...
```

What happened, pass by pass:

- **`n22`, `n24`, `relu_y` are gone as arrays.** All three stayed `Virtual`; the matmul
  re-appears as a `Local_scope` (`v27_n22 { ... }`) *inline at each consumption site* — the
  forward-pass loop nest and both backprop nests each recompute the row-dot-product they
  need. Recompute-over-materialize is the default trade; the caps in `virtualize_settings`
  and the transitive fan-in guard bound how far it goes.
- **Their gradients mostly vanished too.** `relu_y.grad` and `loss.grad` were inlined down to
  the constant `1.0` inside `relu_gate(..., 1.0)` — constant propagation through the
  virtualized seed (`loss.grad =: _1; _1 =: 1.`). Only `n22.grad` survives as storage. Why it
  and nothing else: the visit cap (`virtualize_max_visits`, default 1) counts *per-cell read
  multiplicity*, computed exactly from the affine access relations, and a read at its own
  statement's write position (accumulation-style read-modify-write) is exempt. Every other
  intermediate is read once per cell, or only at its own write position, or is a scalar
  constant expression (the seed `_1`, `loss.grad`), which inlines regardless; but the `w.grad`
  nest reads each `n22.grad[i, j]` cell once per element of the `k` loop — multiplicity 3 —
  so recomputing it inline would triple its cost, and the policy materializes it
  (provenance `Never_virtual 1`).
- **`x`'s six stores vanished** — but differently: `x` is materialized (it's read
  everywhere), and a materialized node whose only writes are literal constants registered
  with `Host_inits` has its in-kernel initialization *moved to link time*
  (`hosted_constant_inits_to_link_time`, gh-633): the values upload once into each context's
  buffer instead of re-executing on every step.
- **FMA formation** (`simplify_llc`): the multiply-accumulate became `fma(...)`.
- **Zeroing survives only where storage survives**: `zero_out` of `loss`, `b.grad`,
  `w.grad`, `n22.grad` — the other five zero-grad statements from the `.cd` died with their
  arrays.

The result record, `Low_level.optimized`, carries more than the code `llc`: the
`traced_store` of per-node facts (which stage 6 uses to derive kernel parameters), the
`optimize_ctx` to thread into the produced context, and schedule-related sets
(`workgroup_shared`, `zero_fringe`, ...) that are empty until stage 5 populates them. It also
reports `flip_candidates` — the inlining decisions a search may flip, which is how
placement became a searchable schedule dimension rather than a fixed heuristic (gh-555).

## Stage 5: schedules — parallelism as a separate concern

*Code: `arrayjit/lib/schedule.ml`; the full story is
[schedules_and_autotuning.md](schedules_and_autotuning.md).*

Everything so far decided *what* to compute and *where values live*; nothing yet decided how
loops map onto hardware. That is the schedule layer, a pure
`Low_level.optimized -> Low_level.optimized` transformation applied at the
`?lowered_transform` seam of backend `compile` (reachable from user code via
`Context.compile ~lowered_transform`). When no explicit transform is given —
our example's case — `Schedule.maybe_default_schedules` applies the default annotators plus
kernel fission:

- **Default GPU/CPU presets**: for each loop nest whose parallelism is provable from the code
  (every materialized write covered by the loop index; conservative race analysis, backed by
  the `Ir.Affine` relational queries), retype loops to hardware axes — `Grid` and
  `Workgroup` on GPU, an outer `Grid` loop rendered onto a thread pool on CPU. Reduction
  loops stay serial: OCANNL's determinism contract forbids atomics, so parallel reductions
  happen only via the explicit `Split_reduce` two-pass transform.
- **Kernel fission** (`Schedule.fission_scheduled`): the whole-step program is cut into
  segments at materialized producer→consumer edges that cannot share a grid launch; each
  segment compiles as its own kernel, chained by device events. This is the fission-not-fusion
  inversion (manifesto §1): kernel boundaries are *discovered* in the fused program, not
  assembled from an operator graph.

In our toy example every extent is below `gpu_schedule_min_parallel`/
`cpu_schedule_min_parallel`, so the nests stay serial and no fission cut is needed — the
whole step ships as one kernel even on Metal. On real models this stage is where the
matmul-tiling, staging, tensorization, and split-reduction transforms apply, either as
autotuner-searched compositions (`Autotune.tune`, with winners persisted in a schedule cache
keyed by a structural digest of the optimized code) or as the model-ranked default. The
transforms are *total*: any schedule on any backend renders correctly, with graceful decline
to scalar fallbacks — performance, never correctness, is what's being searched.

## Stage 6: code generation

*Code: `arrayjit/lib/c_syntax.ml` (the shared functor), `cc_backend.ml`,
`cuda_backend.ml`, `hip_backend.ml`, `metal_backend.ml`, `builtins.c`/`builtins_*.ml`.*

All current backends emit C-family text through the `C_syntax` functor, which turns
`Low_level.t` into a PPrint document given a small per-backend vocabulary (type names,
builtin spellings, how to render precision conversions, how a `Grid` loop binds to the
hardware index, ...). The cc backend takes the defaults nearly verbatim; CUDA/HIP override
what NVIDIA/AMD dialects need (e.g. `__float2half`, wmma intrinsics, `cp.async` staging);
Metal overrides for MSL (`simdgroup` matrices, threadgroup memory).

The functor's entry point is `C_syntax.compile_proc`, called from each backend's
`compile`/`compile_batch`. It derives the kernel's parameter list, then renders the body
(`pp_ll`, a structural recursion over `Low_level.t`):

- `For_loop { axis = Serial }` → a plain `for`; `Grid`/`Workgroup` axes → the backend's
  hardware index binding (`blockIdx`/`threadIdx` on CUDA, `gid`/`lid` on Metal) — and on
  cc, which has no hardware indices, an eligible `Grid` loop renders as a chunked
  `dispatch_apply`/OpenMP thread-pool loop and everything else stays serial;
  `Vectorized`/`Unrolled` axes try SIMD/unrolled renderings with serial fallbacks.
- `Set`/`Get` → array stores/loads with Horner-form row-major offsets; `Local_scope` → a
  scoped local variable; `Zero_out` → a zeroing nest (or elided into the local array's
  `= {0}` initializer); `Tile_mma` → the decline ladder from intrinsics (wmma /
  `simdgroup_matrix` / rocWMMA) through register-tiled C down to the scalar fallback, with
  every outcome recorded in the `mma_census`.

The generated `.c` for our example (excerpt):

```c
void loss_forward_and_gradient_update(
    float *restrict b,
    float *restrict b_grad,
    float *restrict loss,
    float *restrict w,
    float *restrict w_grad,
    float *restrict x) {

  /* Local declarations and initialization. */
  float n22_grad[4] __attribute__((aligned(32))) = {0};

  for (int32_t i39 = 0; i39 <= 1; ++i39) {
    {
      float v43_loss;
      v43_loss = loss[(i39) * 1 + 0];
      for (int32_t i40 = 0; i40 <= 1; ++i40) {
        {
          float v27_n22;
          v27_n22 = (float)(0.0);
          for (int32_t i57 = 0; i57 <= 2; ++i57) {
            v27_n22 = fmaf(w[(i40) * 3 + i57], x[(i39) * 3 + i57], v27_n22);
          }
          v43_loss = (v43_loss + fmaxf(0.0, (v27_n22 + b[i40])));
        }
      }
      loss[(i39) * 1 + 0] = v43_loss;
    }
  }
  ...
}
```

Observations that generalize:

- **Parameters are exactly the materialized context nodes** (derived from the
  `traced_store`): `b`, `w`, `x`, `loss`, and the two surviving gradients. Virtual nodes
  never appear; `n22_grad` resolved to `Local` — routine-scoped scratch, here a stack array
  inside the function (a node over `stack_threshold_in_bytes` resolves `On_device`
  instead). Kernels
  are context-independent: the same compiled function can link against different contexts'
  buffers.
- **`Local_scope`s became scoped local variables** (`v27_n22`, `v43_loss` — the latter a
  read-modify-write contraction of the `loss` accumulation).
- **Multi-dimensional indices flattened to row-major arithmetic** (`(i40) * 3 + i57`).
- On Metal the same body appears inside a `kernel void ... [[buffer(n)]]` function whose
  parameters are *memory pools plus a slot table* rather than one buffer per node — GPU
  backends pack node buffers into pooled device allocations and pass base+offset pairs.
- Numeric builtins that C lacks (`relu_gate` variants, precision conversions, the
  `threefry4x32` PRNG, bf16/fp8 arithmetic) come from `builtins.c` (textually prepended for
  C backends) or `builtins_cuda.ml`/`builtins_metal.ml` (per-dialect strings).

Then each backend turns text into an executable object, in its `compile`/`compile_batch`:
`cc_backend` writes the `.c` file, invokes the configured C compiler into a shared object,
and `dlopen`s it (CPU code can load at compile time — there is one program memory);
`cuda_backend` compiles through NVRTC to PTX but must defer module loading to link time
(`cuModuleLoadDataEx` loads into a device-specific context); `metal_backend` compiles MSL to
a `metallib` similarly. This is why the backend interface distinguishes `compile` (make the
artifact) from `link` (bind it to a device context) — and batches both
(`compile_batch`/`link_batch`) so many routines share one compiler invocation and one loaded
module.

## Stage 7: link, contexts, and running

*Code: `arrayjit/lib/backends.ml` (`Raise_backend.link`, `allocate_delta`),
`arrayjit/lib/context.ml`, `arrayjit/lib/task.ml`; user-facing story in
[tensors_and_contexts.md](tensors_and_contexts.md).*

`Context.compile` returns a `routine` — and behind that word sits the last stretch of the
pipeline, `Raise_backend.link`:

1. **Verify** that every node the code expects from a prior context is actually there
   (`from_prior_context` vs. the context's buffers), and that a merge-buffer consumer is
   linked against the context of the transfer that produced it (static check, gh-288).
2. **Allocate the buffer delta**: materialized nodes not yet present in the parent context
   get device memory now — packed into pools (with liveness-based aliasing of disjoint-lived
   buffers when `buffer_aliasing` is on), constants deduplicated through a per-device cache.
   Nodes with registered host-init data (our `x`) are uploaded here — the link-time half of
   the gh-633 constant-init move from stage 4. (`w` and `b` need no upload: the
   `init_params_for_loss` routine already computed them on-device, in this same lineage.)
3. **Bind the kernel** to the buffers and to the static-index bindings, producing a
   `Task.t` — a plain thunk wrapper. Fissioned segments become one task chained by device
   events (or a CUDA graph / single Metal command buffer, per backend).
4. **Make a child context**: contexts form a lineage; the routine's context extends its
   parent with the new buffers and the forked `optimize_ctx` (so a node this compile decided
   to keep virtual stays virtual for every later compile in this lineage, and its stored
   computation can be inlined into later routines).

Back in `context.ml`, the routine is registered in the lineage's execution ledger, and its
`execution_deps` are derived from read/write hazards against previously compiled routines.
`Context.run` then validates (dependencies satisfied, inputs initialized, bindings in range)
and dispatches the task onto the device's stream — a FIFO queue with events; on the default
`cc` backend the "stream" degenerates to synchronous execution, on `multidev_cc` it is a
worker domain, on GPUs a real device stream. Dynamic loop bounds — minibatch position, etc. —
enter through the `bindings` (`Train.IDX.get_static_symbol`): they lower to kernel
parameters, so re-running with a new index value is just writing an `int ref`, no
recompilation.

Reading results back is explicit and context-mediated: `Context.get_values ctx tn` performs
an on-demand device-to-host transfer (there are no host-resident tensor copies at all,
gh-333). Which is where our program prints its `loss = 0.6590` — the end of the line.

## Recap: where to look when

| you want to... | look at |
|---|---|
| see the high-level code of a routine | `.cd` artifact; `Assignments.to_doc`; `Train.to_routine ~output_cd_file:true` |
| see loop nests before/after optimization | `-unoptimized.ll` / `.ll` artifacts (`Low_level.to_doc`) |
| understand why a node materialized | `Tnode.debug_memory_mode`, provenance codes in [lowering_and_inlining.md](lowering_and_inlining.md) |
| see the kernel source | `.c`/`.cu`/`.metal` artifacts (config `output_debug_files_in_build_directory=true`) |
| see what a kernel computed at runtime | config `debug_log_from_routines=true` (see the debug-tracing skill / docs) |
| trace scheduling decisions | `schedule_log_declines`, `schedule_log_launches`; `routine.mma` |
| understand compile failures by phase | `Schedule_outcome` classification (`Transform` / `Backend_compile` / `Backend_link`) |

And the one-sentence summary of the whole pipeline: **tensors carry `%cd` code; `Train`
sequences a whole step of it; lowering forces shape inference and mints loop nests from
projections; `Low_level.optimize` decides what deserves memory and inlines the rest;
schedules retype loops onto hardware and cut kernels at materialization edges; `C_syntax`
renders each kernel to a C dialect the backend compiles; and linking binds kernels to
per-context device buffers that `Context.run` dispatches on a stream.**
