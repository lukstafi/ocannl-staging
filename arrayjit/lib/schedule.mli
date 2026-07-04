(** {1 Schedule IR: loop-nest transforms as values}

    Halide-style schedules over the lowered, optimized IR: a list of {!optop}s applied as a pure
    [Low_level.optimized -> Low_level.optimized] pass at the [?lowered_transform] seam of backend
    [compile]. See docs/proposals/schedule-ir-optops.md for the design, including the normative
    pass-ordering contract (§2): schedules run after the whole [optimize_proc] pipeline (so they
    see fused code), {!apply} folds freshly constructed guards by re-running [simplify_llc] (plus
    CSE and hoisting when a transform duplicated code), and there is no re-virtualization. *)

open Base

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
          [Low_level.validate_parallel]). *)
  | Unroll of { axis : Indexing.symbol; materialize : bool }
      (** [materialize = false]: set the axis type to [Unrolled] — codegen repeats the body with
          the index bound as a per-block constant (after simplify/CSE have run, so the copies are
          opaque to the optimizer). [materialize = true]: unroll in the IR by substituting index
          constants, so that {!apply}'s trailing simplify + CSE see the copies — constant-folding
          [Affine] indices and deduplicating repeated loads. Register blocktiling is [Split] +
          materializing [Unroll] + the existing CSE (schedule-ir-optops §4). *)
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

val apply :
  ?static_indices:Indexing.static_symbol list ->
  schedule ->
  Low_level.optimized ->
  Low_level.optimized
(** Applies the ops left to right to [optimized.llc], then re-runs [Low_level.simplify_llc] (which
    folds remainder guards the loop extents prove) and, when a materializing [Unroll] duplicated
    code, CSE + cross-statement hoisting. The traced store, optimization context and merge node
    are unchanged — v1 ops are structural rewrites that create no tensor nodes. Raises
    [Invalid_argument] when an op references a loop that does not exist at its point in the
    schedule, or violates an op precondition (see {!optop}). An empty schedule is the identity. *)

val default_gpu : ?block_size:int -> ?min_parallel:int -> Low_level.optimized -> schedule
(** The default GPU annotator preset (schedule-ir-optops §6): for each top-level loop nest whose
    parallelism is provable from the lowered code alone, produce ops annotating exactly one [Grid]
    and one [Workgroup] loop (splitting single parallel loops by [block_size], default from config
    [gpu_schedule_block_size] = 256). A loop is parallelizable when its index occurs as a plain
    [Iterator] component in every materialized write vector beneath it — the same coverage
    property [Low_level.validate_parallel] enforces, used generatively — and the kernel passes a
    conservative race analysis (no cross-nest producer/consumer pairs, all accesses to written
    nodes agree on parallel-index components, no [Zero_out] of materialized nodes, no barriers or
    opaque statements; reduction loops stay serial). Returns the empty schedule when any check
    fails or when the largest parallelizable nest has fewer than [min_parallel] iterations
    (default from config [gpu_schedule_min_parallel] = 1024). *)

val backend_is_gpu : string -> bool
(** Whether the named backend binds hardware indices (currently: name contains ["cuda"] or
    ["metal"]). *)

val maybe_default_gpu :
  backend_name:string ->
  static_indices:Indexing.static_symbol list ->
  Low_level.optimized ->
  Low_level.optimized
(** The implicit transform applied by backend [compile] when the caller passes no
    [?lowered_transform]: {!apply} of {!default_gpu} on GPU backends, the identity otherwise.
    Disabled by config [automatic_gpu_schedule=false], and skipped when runtime kernel logging
    ([debug_log_from_routines]) is active, to keep logs serial and deterministic. *)
