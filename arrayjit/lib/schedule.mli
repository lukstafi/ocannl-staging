(** {1 Schedule IR: loop-nest transforms as values}

    Halide-style schedules over the lowered, optimized IR: a list of {!optop}s applied as a pure
    [Low_level.optimized -> Low_level.optimized] pass at the [?lowered_transform] seam of backend
    [compile]. See docs/proposals/schedule-ir-optops.md for the design, including the normative
    pass-ordering contract (§2): schedules run after the whole [optimize_proc] pipeline (so they see
    fused code), {!apply} folds freshly constructed guards by re-running [simplify_llc] (plus CSE
    and hoisting when a transform duplicated code), and there is no re-virtualization. *)

open Base
module Tn = Tnode

type optop =
  | Split of {
      axis : Indexing.symbol;  (** The loop to split, identified by its index symbol. *)
      factor : int;  (** Extent of the new inner loop. *)
      outer : Low_level.axis_type;  (** Axis type of the new outer loop ([Serial] = no retype). *)
      inner : Low_level.axis_type;  (** Axis type of the new inner loop. *)
      outer_index : Indexing.symbol;  (** Fresh symbol for the outer loop; see {!split}. *)
      inner_index : Indexing.symbol;  (** Fresh symbol for the inner loop; see {!split}. *)
    }
      (** [For_loop i in [0, N)] becomes [i_o in [0, ceil(N/factor)) { i_i in [0, factor) }] with
          [i := factor*i_o + i_i] substituted throughout the body (index vectors and
          [Embed_index]), and — when [factor] does not divide [N] — the body wrapped in an
          [If (factor*i_o + i_i < N)] remainder guard (construct-then-fold: {!apply}'s trailing
          simplify erases it whenever the loop extents prove it). The split loop must start at 0,
          which lowering guarantees. Splitting a serial loop preserves iteration order; retyping
          the results to hardware axes carries the iteration-independence obligation exactly as
          for [Low_level.validate_parallel]. *)
  | Swap of { outer : Indexing.symbol; inner : Indexing.symbol }
      (** Interchange two perfectly nested loops (the outer loop's body must be exactly the inner
          loop; fails loudly otherwise). Reorders iterations — legal for the associative-commutative
          accumulation patterns lowering emits; bitwise reproducibility is the caller's concern. *)
  | Retype of { axis : Indexing.symbol; ty : Low_level.axis_type }
      (** Change a loop's axis type in place. Retyping to a hardware kind requires [from_ = 0] and
          iteration independence (the caller's obligation; structure is checked downstream by
          [Low_level.validate_parallel]). Retyping to [Vectorized] likewise asserts iteration
          independence: the C backends render the loop under vectorization pragmas (gh-ocannl-164),
          backends without pragmas render it as a plain serial loop. *)
  | Unroll of { axis : Indexing.symbol; materialize : bool }
      (** [materialize = false]: set the axis type to [Unrolled] — codegen repeats the body with the
          index bound as a per-block constant (after simplify/CSE have run, so the copies are opaque
          to the optimizer). [materialize = true]: unroll in the IR by substituting index constants,
          so that {!apply}'s trailing simplify + CSE see the copies — constant-folding [Affine]
          indices and deduplicating repeated loads. Register blocktiling is [Split] + materializing
          [Unroll] + the existing CSE (schedule-ir-optops §4). *)
  | Partition of {
      axis : Indexing.symbol;  (** The loop to partition, identified by its index symbol. *)
      breakpoints : int list;
          (** Segment starts: strictly increasing, each strictly inside the loop range
              ([from_ < b <= to_], so every segment is non-empty). *)
      segment_indices : Indexing.symbol list;
          (** Fresh symbols, one per segment ([length breakpoints + 1]); see {!partition}. *)
    }
      (** Index-set splitting (gh-ocannl-508): replace the [Serial] loop
          [For_loop axis in [from_, to_]] by consecutive segment loops
          [s_0 in [from_, b_1) ; s_1 in [b_1, b_2) ; ... ; s_m in [b_m, to_]], each binding a fresh
          symbol substituted for [axis] in a copy of the body (scalar-local scope ids refreshed per
          copy, as for materializing [Unroll]). Segment ranges stay absolute — no index arithmetic
          changes, iteration order is preserved exactly — and each segment's narrowed range lets
          {!apply}'s trailing simplify interval-fold the guards it decides (statement [If]s and
          scalar [Where] range guards alike). This is the segmented-rendering replacement for
          in-loop range guards: an inlined concatenation's per-component guards (partition the
          consumer at the component boundaries, e.g. from {!partition_breakpoints}) and [Split]'s
          construct-then-fold remainder guard (partition at the last tile-multiple first, then
          [Split] the dividing main segment — clean main nest plus epilogue) both specialize into
          guard-free segment nests, and the fresh symbols make each segment individually
          addressable by subsequent ops (per-segment scheduling). *)
  | Stage of {
      source : Tn.t;
      tile_loops : Indexing.symbol list;
      shared : bool;
      cooperative : int option;
      hoisted : bool;
      swizzle : bool;
    }
      (** Stage reads of [source] through a tile: a fresh [Local]-mode node registered in the traced
          store, its dims derived per source axis from the range of the index terms over
          [tile_loops] (schedule-ir-optops §5). All reads of [source] must use one index vector (v1)
          whose per-axis terms split cleanly into tile-loop terms (positive coefficients) and outer
          terms. With [shared = true] the tile is added to [workgroup_shared] and a cooperative-load
          nest plus barriers are inserted at the deepest loop carrying an outer-part symbol or a
          reused [Workgroup] tile axis: [Workgroup]-typed tile loops are reused as the cooperating
          thread indices, [Serial] tile loops are iterated under fresh symbols, per-axis edge guards
          are constructed then folded, and redundant loading along non-participating workgroup axes
          is restricted to one representative thread. A shared stage with no anchor (no outer-part
          symbol, no reused workgroup axis — e.g. staging a broadcast vector) wraps the outermost
          tile loop instead of the routine root, so enclosing workgroup axes can guard the loads;
          every workgroup slot active in the kernel's launch must be reused or bound by a loop
          enclosing the staging point, otherwise threads differing in the uncovered slot would race
          on the shared tile and the op raises. Note that [Split]'s whole-body remainder guards
          would place the inserted barriers under divergent control flow (rejected by
          [validate_parallel]), so v1 shared staging requires tile sizes dividing the extents. With
          [shared = false] (CPU operand packing) all tile loops must be [Serial] and a plain serial
          copy nest is inserted, no barriers. The source must not be written in the routine.

          [cooperative = Some w] is the lane-aware mode for composing with {!constructor-Tensorize}
          (docs/proposals/tensorize-mma.md, "Lane-aware Stage"): shared staging with all-[Serial]
          tile loops whose load nest is wrapped in a {e fresh} extent-[w] [Workgroup] lane loop —
          positionally the same slot 0 the tensorized micro-kernel's lane loop binds, with extent
          agreement enforced downstream by [Low_level.validate_parallel]'s barrier-strength
          uniformity (pass the backend's {!field:Backend_intf.mma_simd_width} to both ops). The lane
          is folded linearly into the innermost fresh copy loop: extent [<= w] replaces the loop by
          the lane index under a [lane < extent] guard (folds when equal); an extent divisible by
          [w] iterates [extent / w] chunks at [w*step + lane]; otherwise the whole nest is
          restricted to lane 0 (division is not expressible in affine indices). The lane loop covers
          workgroup slot 0 for the staging-point coverage rule by construction.

          [swizzle = true] stores the tile in an XOR-swizzled layout (the bank-conflict-avoidance
          follow-up of docs/proposals/tensorize-mma.md): the tile node is added to the [optimized]
          record's {!field:Low_level.swizzled} set and codegen remaps every element access
          [P*C + col] to [P*C + (col lxor (P land (C-1)))] — a per-row bijection of the minor axis,
          so semantics are unchanged while same-column accesses (the classic strided read of a
          staged tile) spread across shared-memory banks. Requires [shared = true], a tile with at
          least two axes, and a power-of-two minor tile dim [> 1]. Renderings that assume row-major
          storage decline swizzled operands: [Tile_mma] intrinsic/register-tiled paths fall back to
          the scalar micro-kernel (which reads elementwise and stays correct), so do not combine
          [swizzle] with {!constructor-Tensorize} when the intrinsics are the goal — the swizzled
          tile is for scalar/register-blocktiled staged kernels until [ldmatrix]-style
          swizzle-aware fragment loads exist.

          [hoisted = true] packs a compile-time-constant operand once, out of the routine
          (gh-ocannl-470, the compiler-native analog of ggml's [CPU_REPACK] [set_tensor] hook):
          instead of a per-invocation scratch tile refilled by an in-kernel load nest, the packed
          layout covers the {e whole} source — one packed-buffer axis per outer coordinate
          ([outer part / tile dim], requiring outer-part coefficients and offsets divisible by the
          tile dim on tiled axes) followed by the tile axes — and the reads are remapped to it
          directly; no load nest, no barriers. The packed node is minted as a host-initialized
          constant: its buffer is computed on the host when first forced (a [Host_inits] lazy driven
          by the same affine index maps; pad slots of edge tiles are zero-filled) and uploaded once
          per device into the constant pool ([constant_buffer_cache]), so re-linking into sibling
          contexts reuses the same packed buffer. Requires [shared = false] (and hence all-[Serial]
          tile loops), a source with registered host-init data that is known constant (declared
          [Effectively_constant] intent or a constant placement), no padding on the source, and
          every outer-part symbol bound by an enclosing loop (static/dynamic indices are rejected —
          packing runs at link time with no bindings). Note: later host-side writes to the source
          (e.g. [set_values]) do NOT refresh the packed copy. *)
  | Privatize of { target : Tn.t; over : Indexing.symbol }
      (** Accumulator privatization: contract the read-modify-write accumulation of the materialized
          [target] across the (Serial) [over] loop's whole subtree into a per-thread [Local]
          accumulator tile — initialized from [target] before the loop, accumulated in place, stored
          back after, one final write per element. This recovers, for materialized nodes, the
          scope-local form virtualization gives virtual accumulators ([Local_scope]) — and because a
          routine-local tile cannot alias the kernel's device pointers, downstream compilers
          register-allocate it without waiting for [restrict] (gh-ocannl-164). Tile shape: per
          target axis, the index terms over loops nested inside [over] (required [Serial] with
          [from_ = 0]); no such terms yields a scalar accumulator. All accesses of [target] under
          [over] must use one index vector, which must not mention [over] itself, and must sit under
          one [If] guard chain, which must be iteration-invariant (no memory reads, no symbols bound
          inside [over]'s subtree) — the same predicate then gates the init-load and store-back, so
          lane-restricted accumulations (e.g. [w == 0]) privatize correctly; per-iteration guards
          are rejected. A [Zero_out] of [target] elsewhere is left in place — the init-load observes
          it, so semantics are preserved without a surjectivity analysis. Compose as: [Split]s →
          [Stage]s → [Privatize] → materializing [Unroll]s (the unrolls then turn the tile accesses
          into constant-indexed, register-allocatable form). *)
  | Expand_zero of { tn : Tn.t; indices : Indexing.symbol list }
      (** Expand the unique [Zero_out tn] statement into an ordinary loop nest over the supplied
          symbols (one per axis of [tn]'s padded dims; see {!expand_zero}). Whole-node [Zero_out] of
          a materialized node is rejected by [Low_level.validate_parallel] in multi-threaded kernels
          — expanding it first lets the schedule split and annotate the zeroing with the same
          hardware geometry as the computation that follows. *)
  | Tensorize of {
      i : Indexing.symbol;
      j : Indexing.symbol;
      k : Indexing.symbol;
      lane : Indexing.symbol;  (** Fresh symbol for the cooperating lane loop; see {!tensorize}. *)
      simd_width : int;
          (** Extent of the lane loop (the backend's {!field:Backend_intf.mma_simd_width}; 32 on
              Metal and CUDA). *)
    }
      (** Tensor-core emission (docs/proposals/tensorize-mma.md §3): replace the perfectly nested
          serial [i × j × k] matmul micro-kernel — whose body is the single accumulation
          [d[..., i, j] += a[..., i, k] * b[..., k, j]] (plain-add or FMA form; each operand's tile
          spans its last two axes with unit coefficients, transposed layouts rejected in v1) — with
          a [Low_level.Tile_mma] block statement covering the full [m×n×k] extents, wrapped in a
          fresh [Workgroup]-typed lane loop of extent [simd_width]. The original nest is kept as the
          statement's scalar [fallback]: backends without an MMA hook (or declining a particular
          precision/shape) render it once per simdgroup under an [if (lane == 0)] guard, so the op
          is always semantics-preserving. Apply after [Split]s and [Stage]s; [Stage]/[Privatize]
          must come before it. Divisibility by the intrinsic tile (8 on Metal) is a per-call
          emission concern, not checked here. *)
  | Fuse_epilogue of { target : Tn.t; shared : bool }
      (** Epilogue fusion (gh-ocannl-486): fold the sole-consumer, index-space-compatible
          elementwise tail that re-reads [target] — the typical bias add / activation / residual
          after a reduction — into [target]'s store-back site, eliminating the tail's separate
          memory pass and keeping the fused routine a single kernel/segment. Recognized sites: the
          lane-0 fragment store-back synthesized by [Tensorize]'s accumulator contraction (the tail
          becomes a fourth, lane-0-guarded statement of the marked region, rendered after the
          backend's intrinsic block), the [Privatize] tile store-back (per-element), and the plain
          accumulation nest (the tail slides inside the output loops after the serial reduction
          loop). The tail must be the first real statement after the last statement writing
          [target]: a perfect Serial nest over exactly [target]'s dims, assigning a different node
          at the identity index tuple, elementwise, with every read of [target] at that tuple, and
          no later statement mentioning [target]. The store-back of [target] itself is kept.
          Elementwise tails never reorder the reduction, so on the C backends the fused values are
          bitwise equal to the two-kernel form. Apply after [Tensorize]/[Privatize].

          [shared] (GPU only, requires the fragment site): the fused tail is often [target]'s last
          consumer, so placement makes [target] routine-local — a per-thread array the fragment
          hooks cannot [simdgroup_load] from. With [shared], [target] is placed in workgroup-shared
          memory (like [Stage]'s shared tiles) unless already settled on-device, so the intrinsic
          fragment path fires against threadgroup memory. CPU backends reject shared placement. *)
[@@deriving sexp_of]

type schedule = optop list [@@deriving sexp_of]
(** Applied left to right by {!apply}. *)

val split :
  axis:Indexing.symbol ->
  factor:int ->
  outer:Low_level.axis_type ->
  inner:Low_level.axis_type ->
  optop * Indexing.symbol * Indexing.symbol
(** Builds a {!constructor-Split} with fresh outer and inner index symbols (via
    [Indexing.get_symbol]) and returns them, so subsequent ops in a programmatically built schedule
    can reference the new loops. *)

val partition : axis:Indexing.symbol -> breakpoints:int list -> optop * Indexing.symbol list
(** Builds a {!constructor-Partition} with one fresh index symbol per segment (via
    [Indexing.get_symbol]) and returns them in segment order, so subsequent ops in a
    programmatically built schedule can address individual segments. *)

val partition_breakpoints : axis:Indexing.symbol -> Low_level.t -> int list
(** Derives candidate {!constructor-Partition} breakpoints of the [axis] loop from the guards
    already present in its body: statement [If] conditions and scalar [Where] conditions (e.g. the
    virtualizer's per-component range guards of an inlined concatenation, or [Split]'s remainder
    guard) are scanned for comparisons whose two sides differ by [k*axis + off] with everything else
    constant — each flips truth value at one point of the axis range. Returns the collected flip
    points that fall strictly inside the loop range, sorted and deduplicated (possibly empty — e.g.
    when every guard is already interval-decided); partitioning at them makes every such guard
    interval-decided within each segment, so {!apply}'s trailing simplify erases them. Raises
    [Invalid_argument] when no statement-level loop binds [axis]. *)

val expand_zero : tn:Tn.t -> optop * Indexing.symbol list
(** Builds an {!constructor-Expand_zero} with one fresh symbol per axis of [tn] (forcing [tn]'s
    dims) and returns the symbols for subsequent [Split]/[Retype] ops. *)

val tensorize :
  i:Indexing.symbol ->
  j:Indexing.symbol ->
  k:Indexing.symbol ->
  simd_width:int ->
  optop * Indexing.symbol
(** Builds a {!constructor-Tensorize} with a fresh lane symbol (via [Indexing.get_symbol]) and
    returns it. *)

val can_fuse_epilogue : target:Tn.t -> Low_level.optimized -> bool
(** Whether {!constructor-Fuse_epilogue} on [target] would succeed on this code — i.e. an eligible
    elementwise tail immediately follows the reduction over [target]. Used by the autotune seeding
    to propose fused-epilogue sketch variants (the check runs on the pre-schedule code, where the
    plain accumulation-nest site applies). *)

val hoistable_constant : Tn.t -> bool
(** Whether the node is eligible as a [hoisted] {!constructor-Stage} source: declared value-constant
    ([Tnode.known_host_constant]) with registered host-init data to pack from. Shared by the
    autotune sketch and by [Schedule_cache.canonicalize], which renders it per tensor node so that
    same-shape programs differing in operand constancy do not share cached schedules. *)

val apply :
  ?static_indices:Indexing.static_symbol list ->
  schedule ->
  Low_level.optimized ->
  Low_level.optimized
(** Applies the ops left to right to the optimized code, then re-runs [Low_level.simplify_llc]
    (which folds remainder and edge guards the loop extents prove) and, when a materializing
    [Unroll] duplicated code, CSE + cross-statement hoisting. [Stage] registers its tile in the
    traced store (and [workgroup_shared] when shared); the optimization context and merge node are
    never changed. Raises [Invalid_argument] when an op references a loop that does not exist at its
    point in the schedule, or violates an op precondition (see {!optop}). An empty schedule is the
    identity. *)

type op_verdict = Op_legal | Op_illegal of string | Op_unknown of string
[@@deriving sexp_of, equal]

val op_legality : Low_level.optimized -> optop -> op_verdict
(** gh-494 waypoint 3: the op-legality oracle. A schedule op is a thread-pairing transform over
    the routine's access relations; its obligation is decided by the {!Affine} queries — pairing
    the op's loop symbol(s) as thread identity must leave every write-involving access pair of
    every written node [Disjoint] or [Same_thread] (with the reduction reassociation license for
    [Vectorized] retypes and [Swap]s of accumulations). Three-valued and sound in both proven
    directions: [Op_legal] proves the annotation race-free; [Op_illegal] proves a violation (e.g.
    a materialized node's unguarded write provably independent of the retyped axis); [Op_unknown]
    means "compile and see" — the op may be valid under semantics the queries do not model
    (per-thread scratch copies, renderer fallbacks) — and must never be treated as a rejection.
    Hardware annotations interleave sibling statements' threads with no grid-wide
    synchronization, so a hardware retype/split of a loop sharing an accessed node across its
    statement boundary (with a write on either side) reports [Op_unknown]: such pairs need the
    default annotator's aligned-mapping analysis, which the single-op oracle does not model.
    [Vectorized] retypes and [Swap]s only reorder within the nest and keep the intra-nest scope.

    [Tensorize]'s role assignment is decided: the micro-kernel recognition of apply (a pure
    function of the code) is probed, so an invalid role permutation — e.g. the reduction loop
    assigned to [i]/[j], violating the discipline that the [k] role is the only rmw-carrying loop
    of the accumulation — is a proven [Op_illegal]; on structural success the affine queries
    decide [i]/[j] iteration independence under the reduction-reassociation license (the intrinsic
    reassociates [k]), with the hardware-lane cross-statement downgrade to [Op_unknown] as above.
    [Stage] is likewise probed hermetically (a raising apply is [Op_illegal]), then its implicit
    contract — the staged tile covers the reads it replaces within the staging scope — is the
    containment query [Affine.read_covered_before] over the fresh tile's accesses: a covered
    non-shared (packing) stage is [Op_legal]; shared staging (barrier placement and launch
    geometry are validated downstream) and hoisted staging (link-time host-side packing) report
    [Op_unknown]. [Privatize]/[Expand_zero]/[Fuse_epilogue] report [Op_unknown]; their own
    apply-time preconditions remain in force. *)

val schedule_legality : Low_level.optimized -> schedule -> (optop * op_verdict) list
(** Per-op verdicts, each against the code with the preceding ops applied (on a hermetic copy —
    checking never mutates the argument). Stops after a proven-illegal op or a failing
    application; an op that fails to apply reports [Op_illegal] with the exception. *)

val default_gpu :
  ?block_size:int ->
  ?min_parallel:int ->
  ?limits:Backend_intf.hardware_limits ->
  Low_level.optimized ->
  schedule
(** The default GPU annotator preset (schedule-ir-optops §6): for each top-level loop nest whose
    parallelism is provable from the lowered code alone, produce ops annotating exactly one [Grid]
    and one [Workgroup] loop (splitting single parallel loops by [block_size], default from config
    [gpu_schedule_block_size] = 256, clamped to [limits]'
    {!field:Backend_intf.max_threads_per_workgroup} when given — the configured block size is a
    target, the device's workgroup capacity a hard cap). A loop is parallelizable when its index
    occurs as a plain [Iterator] component in every materialized write vector beneath it — the same
    coverage property [Low_level.validate_parallel] enforces, used generatively — and the kernel
    passes a conservative race analysis (all accesses to written nodes agree on parallel-index
    components, no [Zero_out] of materialized nodes, no barriers or opaque statements; reduction
    loops stay serial). Cross-nest producer/consumer (or WAW/WAR) pairs over a written node are
    allowed only when {e aligned}: the linked nests' chains are trimmed to a common equal-extent
    prefix — identical annotation geometry, so each hardware thread covers the same index slice in
    every linked nest — and per axis position the paired accesses either both use plain [Iterator]s
    of same-chain-position parallel symbols or neither mentions one; otherwise the analysis bails.
    For non-materialized (per-thread copy) scratch the edge additionally requires value
    thread-invariance at the chosen trim: a chain symbol feeding a scratch write's value without
    pinning the written cell would leave each consumer thread's copy holding its own chunk's last
    value where the serial reference holds the last chunk's, so the trim search serializes that
    loop (gh-494; direct syntactic dependence only).
    Returns the empty schedule when any check fails or when the largest parallelizable nest has
    fewer than [min_parallel] iterations (default from config [gpu_schedule_min_parallel] = 64: a
    kernel launches either way, so any real parallelism beats the serial 1x1 fallback — a single GPU
    thread is 1-2 orders of magnitude slower than a CPU core; the remaining small threshold keeps
    sub-simdgroup-scale programs fully serial so their segments coalesce and placements stay
    unchanged). *)

val default_cpu : ?min_parallel:int -> Low_level.optimized -> schedule
(** The default CPU annotator preset: the same conservative analysis as {!default_gpu}, but each
    nest's outermost parallelizable loop is merely retyped to [Grid] — pool-backed Grid rendering in
    the C backend (docs/proposals/gh-ocannl-164.md) partitions that loop into contiguous chunks
    executing on a process-global native thread pool, and a [Workgroup] split would only add loop
    structure that runs serially inside a chunk. Returns the empty schedule below [min_parallel]
    (default from config [cpu_schedule_min_parallel] = 16384; task fan-out costs more than a GPU
    launch is worth on small kernels). *)

val log_launches : bool Lazy.t
(** Config [schedule_log_launches]: backend [compile] logs one stderr line per compiled segment with
    its launch grid/block dims and statement count — for diffing what two compiles of nominally
    identical code actually emit. *)

val backend_is_gpu : string -> bool
(** Whether the named backend binds hardware indices (currently: name contains ["cuda"] or
    ["metal"]). *)

val backend_is_cpu : string -> bool
(** Whether the named backend renders [Grid] loops on the CPU pool (currently: name contains
    ["cc"]). *)

val automatic_schedule_active : backend_name:string -> bool
(** Whether {!maybe_default_schedule} (and {!maybe_default_schedules}) would apply a default preset
    on this backend rather than the identity: the backend binds Grid loops, the corresponding config
    gate ([automatic_gpu_schedule] / [automatic_cpu_schedule]) is on, and runtime kernel logging
    ([debug_log_from_routines]) is off. Callers that substitute their own default-schedule choice
    (e.g. the cost-model default selection of gh-ocannl-491) must respect this gate. *)

val maybe_default_schedule :
  backend_name:string ->
  ?limits:Backend_intf.hardware_limits ->
  static_indices:Indexing.static_symbol list ->
  Low_level.optimized ->
  Low_level.optimized
(** The implicit transform applied by backend [compile] when the caller passes no
    [?lowered_transform]: {!apply} of {!default_gpu} on GPU backends, of {!default_cpu} on CPU
    backends, the identity otherwise. [limits] (default {!Backend_intf.no_hardware_limits}) should
    be the compiling backend's {!Backend_intf.Backend_device_common.hardware_limits}. Disabled by
    config [automatic_gpu_schedule=false] / [automatic_cpu_schedule=false] respectively, and skipped
    when runtime kernel logging ([debug_log_from_routines]) is active, to keep logs serial and
    deterministic. *)

val zero_expansion :
  ?block_size:int ->
  ?min_parallel:int ->
  limits:Backend_intf.hardware_limits ->
  Tnode.t list ->
  schedule
(** The expand-and-annotate schedule {!maybe_default_schedules} applies to a fission segment of
    materialized whole-node [Zero_out]s on GPU backends: {!expand_zero} plus the same Grid/Workgroup
    geometry policy as {!default_gpu}. Below [min_parallel] (largest node) the zeros stay whole-node
    (a serial kernel renders them as [memset]). Exposed for callers (e.g. the autotuner) that
    replicate the default fission pipeline with custom per-segment schedules. *)

val fission_scheduled :
  ?promote_locals:bool ->
  preset:(Low_level.optimized -> schedule) ->
  zero_sched:(Tnode.t list -> schedule) ->
  static_indices:Indexing.static_symbol list ->
  Low_level.optimized ->
  ([ `Normal | `Zeros | `Solo ] * Low_level.optimized * schedule * Low_level.optimized) list
(** The kernel-fission pipeline underlying {!maybe_default_schedules}, with caller-supplied
    per-segment schedules and a per-segment result: the routine's top-level statements are
    partitioned at cross-workgroup dependency edges exactly as described there (edges the aligned
    cross-nest rule of {!default_gpu} proves race-free without losing any nest's standalone
    parallelism do not cut), [preset] is called on each [`Normal] segment's (pre-schedule)
    [optimized] slice and [zero_sched] on each [`Zeros] segment's nodes, and each result tuple
    carries the segment kind, the pre-schedule segment, the schedule chosen for it, and the
    scheduled segment ({!apply} of the schedule). [`Solo] segments (opaque to the analysis, or
    coalesced runs of unannotated segments) get the empty schedule. When fission does not apply
    (single segment, unfissionable crossings, or everything coalesces back) the result is a single
    [`Normal] tuple over the whole routine with [preset]'s schedule. Callers compile each scheduled
    segment as its own kernel in order (the plural transform seam of backend [compile]); see
    {!maybe_default_schedules} for the synchronization contract.

    [promote_locals] (default [false]): promote statement-crossing [Local] scratch to [On_device]
    before segmentation. A nest whose only writes land in [Local] scratch gets no parallel chain
    (the annotator's coverage property quantifies over materialized writes), and Local
    producer/consumer edges do not cut — so such producers either drag their segment down to a
    serial 1x1 launch or are redundantly re-executed by every hardware thread; small reduction
    intermediates (layer-norm statistics, softmax max/denominator) land exactly here via the [Local]
    stack threshold. Promotion lets the ordinary materialized machinery apply; promotions fission
    does not end up needing (single-kernel fallbacks, or all accesses within one segment after
    coalescing) are restored. {!maybe_default_schedules} passes [true] on GPU backends, where a
    serial nest costs orders of magnitude more than on CPU. *)

val maybe_default_schedules :
  backend_name:string ->
  ?limits:Backend_intf.hardware_limits ->
  static_indices:Indexing.static_symbol list ->
  Low_level.optimized ->
  Low_level.optimized list
(** Like {!maybe_default_schedule}, but with kernel fission: the routine's top-level statements are
    partitioned into segments at the cross-workgroup dependency edges the default annotator's
    interference analysis would otherwise reject wholesale — materialized producer/consumer (and
    WAW/WAR) pairs of sibling statements, bare materialized writes, materialized whole-node
    [Zero_out]s, statements opaque to the analysis — and each segment receives the default schedule
    independently, on its own launch geometry. The caller compiles each returned [optimized] as its
    own kernel and runs them in order on the routine's stream, chaining a device-side event at each
    boundary — that event supplies the grid-wide synchronization at each cut (queue FIFO alone does
    not order overlapping command buffers over Metal's untracked resources). Scalar scope-locals
    hoisted across statements are replicated into consuming segments when provably value-preserving
    (segments merge back and run serially otherwise); [Local]-placed scratch whose live range
    crosses a cut is promoted to [On_device] in the compile's placement fork. On GPU, segments
    consisting of materialized [Zero_out]s are expanded ({!expand_zero}) and annotated like ordinary
    nests. Adjacent segments that end up unannotated are coalesced, so an all-serial routine stays a
    single kernel. Returns a singleton equal to {!maybe_default_schedule}'s result when fission does
    not apply (non-GPU/CPU backend, automatic scheduling disabled, config [schedule_fission=false],
    kernel logging active, or nothing to split). *)

val check_hardware_limits :
  name:string -> limits:Backend_intf.hardware_limits -> Low_level.optimized -> unit
(** Validates the scheduled kernel [name] against the device limits, raising [Utils.User_error] on
    violation: the launch's workgroup size (the product of {!Low_level.launch_dims}' block
    dimensions) against {!field:Backend_intf.max_threads_per_workgroup}, and the total bytes of
    [workgroup_shared] tiles against {!field:Backend_intf.max_workgroup_memory_bytes}. Backend
    [compile] calls this after any [?lowered_transform] (or the default annotator, which already
    respects the limits), turning driver-level launch failures into early, named errors. A no-op for
    all-[None] limits. *)
