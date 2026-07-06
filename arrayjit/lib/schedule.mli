(** {1 Schedule IR: loop-nest transforms as values}

    Halide-style schedules over the lowered, optimized IR: a list of {!optop}s applied as a pure
    [Low_level.optimized -> Low_level.optimized] pass at the [?lowered_transform] seam of backend
    [compile]. See docs/proposals/schedule-ir-optops.md for the design, including the normative
    pass-ordering contract (§2): schedules run after the whole [optimize_proc] pipeline (so they
    see fused code), {!apply} folds freshly constructed guards by re-running [simplify_llc] (plus
    CSE and hoisting when a transform duplicated code), and there is no re-virtualization. *)

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
          loop; fails loudly otherwise). Reorders iterations — legal for the
          associative-commutative accumulation patterns lowering emits; bitwise reproducibility is
          the caller's concern. *)
  | Retype of { axis : Indexing.symbol; ty : Low_level.axis_type }
      (** Change a loop's axis type in place. Retyping to a hardware kind requires [from_ = 0] and
          iteration independence (the caller's obligation; structure is checked downstream by
          [Low_level.validate_parallel]). Retyping to [Vectorized] likewise asserts iteration
          independence: the C backends render the loop under vectorization pragmas (gh-ocannl-164),
          backends without pragmas render it as a plain serial loop. *)
  | Unroll of { axis : Indexing.symbol; materialize : bool }
      (** [materialize = false]: set the axis type to [Unrolled] — codegen repeats the body with
          the index bound as a per-block constant (after simplify/CSE have run, so the copies are
          opaque to the optimizer). [materialize = true]: unroll in the IR by substituting index
          constants, so that {!apply}'s trailing simplify + CSE see the copies — constant-folding
          [Affine] indices and deduplicating repeated loads. Register blocktiling is [Split] +
          materializing [Unroll] + the existing CSE (schedule-ir-optops §4). *)
  | Stage of { source : Tn.t; tile_loops : Indexing.symbol list; shared : bool }
      (** Stage reads of [source] through a tile: a fresh [Local]-mode node registered in the
          traced store, its dims derived per source axis from the range of the index terms over
          [tile_loops] (schedule-ir-optops §5). All reads of [source] must use one index vector
          (v1) whose per-axis terms split cleanly into tile-loop terms (positive coefficients) and
          outer terms. With [shared = true] the tile is added to [workgroup_shared] and a
          cooperative-load nest plus barriers are inserted at the deepest loop carrying an
          outer-part symbol or a reused [Workgroup] tile axis: [Workgroup]-typed tile loops are
          reused as the cooperating thread indices, [Serial] tile loops are iterated under fresh
          symbols, per-axis edge guards are constructed then folded, and redundant loading along
          non-participating workgroup axes is restricted to one representative thread. A shared
          stage with no anchor (no outer-part symbol, no reused workgroup axis — e.g. staging a
          broadcast vector) wraps the outermost tile loop instead of the routine root, so
          enclosing workgroup axes can guard the loads; every workgroup slot active in the
          kernel's launch must be reused or bound by a loop enclosing the staging point,
          otherwise threads differing in the uncovered slot would race on the shared tile and
          the op raises. Note that
          [Split]'s whole-body remainder guards would place the inserted barriers under divergent
          control flow (rejected by [validate_parallel]), so v1 shared staging requires tile sizes
          dividing the extents. With [shared = false] (CPU operand packing) all tile loops must be
          [Serial] and a plain serial copy nest is inserted, no barriers. The source must not be
          written in the routine. *)
  | Privatize of { target : Tn.t; over : Indexing.symbol }
      (** Accumulator privatization: contract the read-modify-write accumulation of the
          materialized [target] across the (Serial) [over] loop's whole subtree into a per-thread
          [Local] accumulator tile — initialized from [target] before the loop, accumulated in
          place, stored back after, one final write per element. This recovers, for materialized
          nodes, the scope-local form virtualization gives virtual accumulators ([Local_scope]) —
          and because a routine-local tile cannot alias the kernel's device pointers, downstream
          compilers register-allocate it without waiting for [restrict] (gh-ocannl-164). Tile
          shape: per target axis, the index terms over loops nested inside [over] (required
          [Serial] with [from_ = 0]); no such terms yields a scalar accumulator. All accesses of
          [target] under [over] must use one index vector, which must not mention [over] itself,
          and must sit under one [If] guard chain, which must be iteration-invariant (no memory
          reads, no symbols bound inside [over]'s subtree) — the same predicate then gates the
          init-load and store-back, so lane-restricted accumulations (e.g. [w == 0]) privatize
          correctly; per-iteration guards are rejected.
          A [Zero_out] of [target] elsewhere is left in place — the init-load observes it, so
          semantics are preserved without a surjectivity analysis. Compose as: [Split]s →
          [Stage]s → [Privatize] → materializing [Unroll]s (the unrolls then turn the tile
          accesses into constant-indexed, register-allocatable form). *)
  | Expand_zero of { tn : Tn.t; indices : Indexing.symbol list }
      (** Expand the unique [Zero_out tn] statement into an ordinary loop nest over the supplied
          symbols (one per axis of [tn]'s padded dims; see {!expand_zero}). Whole-node [Zero_out]
          of a materialized node is rejected by [Low_level.validate_parallel] in multi-threaded
          kernels — expanding it first lets the schedule split and annotate the zeroing with the
          same hardware geometry as the computation that follows. *)
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
    [Indexing.get_symbol]) and returns them, so subsequent ops in a programmatically built
    schedule can reference the new loops. *)

val expand_zero : tn:Tn.t -> optop * Indexing.symbol list
(** Builds an {!constructor-Expand_zero} with one fresh symbol per axis of [tn] (forcing [tn]'s
    dims) and returns the symbols for subsequent [Split]/[Retype] ops. *)

val apply :
  ?static_indices:Indexing.static_symbol list ->
  schedule ->
  Low_level.optimized ->
  Low_level.optimized
(** Applies the ops left to right to the optimized code, then re-runs [Low_level.simplify_llc]
    (which folds remainder and edge guards the loop extents prove) and, when a materializing
    [Unroll] duplicated code, CSE + cross-statement hoisting. [Stage] registers its tile in the
    traced store (and [workgroup_shared] when shared); the optimization context and merge node are
    never changed. Raises [Invalid_argument] when an op references a loop that does not exist at
    its point in the schedule, or violates an op precondition (see {!optop}). An empty schedule is
    the identity. *)

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
    occurs as a plain
    [Iterator] component in every materialized write vector beneath it — the same coverage
    property [Low_level.validate_parallel] enforces, used generatively — and the kernel passes a
    conservative race analysis (no cross-nest producer/consumer pairs, all accesses to written
    nodes agree on parallel-index components, no [Zero_out] of materialized nodes, no barriers or
    opaque statements; reduction loops stay serial). Returns the empty schedule when any check
    fails or when the largest parallelizable nest has fewer than [min_parallel] iterations
    (default from config [gpu_schedule_min_parallel] = 1024). *)

val default_cpu : ?min_parallel:int -> Low_level.optimized -> schedule
(** The default CPU annotator preset: the same conservative analysis as {!default_gpu}, but each
    nest's outermost parallelizable loop is merely retyped to [Grid] — pool-backed Grid rendering
    in the C backend (docs/proposals/gh-ocannl-164.md) partitions that loop into contiguous chunks
    executing on a process-global native thread pool, and a [Workgroup] split would only add loop
    structure that runs serially inside a chunk. Returns the empty schedule below [min_parallel]
    (default from config [cpu_schedule_min_parallel] = 16384; task fan-out costs more than a GPU
    launch is worth on small kernels). *)

val backend_is_gpu : string -> bool
(** Whether the named backend binds hardware indices (currently: name contains ["cuda"] or
    ["metal"]). *)

val backend_is_cpu : string -> bool
(** Whether the named backend renders [Grid] loops on the CPU pool (currently: name contains
    ["cc"]). *)

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
    config [automatic_gpu_schedule=false] / [automatic_cpu_schedule=false] respectively, and
    skipped when runtime kernel logging ([debug_log_from_routines]) is active, to keep logs serial
    and deterministic. *)

val maybe_default_schedules :
  backend_name:string ->
  ?limits:Backend_intf.hardware_limits ->
  static_indices:Indexing.static_symbol list ->
  Low_level.optimized ->
  Low_level.optimized list
(** Like {!maybe_default_schedule}, but with kernel fission: the routine's top-level statements
    are partitioned into segments at the cross-workgroup dependency edges the default annotator's
    interference analysis would otherwise reject wholesale — materialized producer/consumer (and
    WAW/WAR) pairs of sibling statements, bare materialized writes, materialized whole-node
    [Zero_out]s, statements opaque to the analysis — and each segment receives the default
    schedule independently, on its own launch geometry. The caller compiles each returned
    [optimized] as its own kernel and runs them in order on the routine's stream, chaining a
    device-side event at each boundary — that event supplies the grid-wide synchronization at
    each cut (queue FIFO alone does not order overlapping command buffers over Metal's untracked
    resources). Scalar scope-locals hoisted across
    statements are replicated into consuming segments when provably value-preserving (segments
    merge back and run serially otherwise); [Local]-placed scratch whose live range crosses a cut
    is promoted to [On_device] in the compile's placement fork. On GPU, segments consisting of
    materialized [Zero_out]s are expanded ({!expand_zero}) and annotated like ordinary nests.
    Adjacent segments that end up unannotated are coalesced, so an all-serial routine stays a
    single kernel. Returns a singleton equal to {!maybe_default_schedule}'s result when fission
    does not apply (non-GPU/CPU backend, automatic scheduling disabled, config
    [schedule_fission=false], kernel logging active, or nothing to split). *)

val check_hardware_limits :
  name:string -> limits:Backend_intf.hardware_limits -> Low_level.optimized -> unit
(** Validates the scheduled kernel [name] against the device limits, raising [Utils.User_error] on
    violation: the launch's workgroup size (the product of {!Low_level.launch_dims}' block
    dimensions) against {!field:Backend_intf.max_threads_per_workgroup}, and the total bytes of
    [workgroup_shared] tiles against {!field:Backend_intf.max_workgroup_memory_bytes}. Backend
    [compile] calls this after any [?lowered_transform] (or the default annotator, which already
    respects the limits), turning driver-level launch failures into early, named errors. A no-op
    for all-[None] limits. *)
