(** Analytic cost model, the extraction half (gh-ocannl-491 task 1): per-kernel compulsory memory
    footprints, arithmetic-op counts, arithmetic intensity, and the roofline lower-bound time under
    advisory envelope constants ({!Backend_intf.hardware_limits}'s [peak_flops] /
    [peak_memory_bandwidth]).

    The analysis is pure and backend-free: it reads a (typically optimized) {!Low_level.t} through
    the affine-access artifacts of gh-ocannl-494 ({!Low_level.affine_accesses},
    {!Affine.fiber_cardinality}) and returns numbers; envelope constants are plain floats supplied
    by the caller. The model ranks candidate schedules — it does not predict runtimes.

    Approximation contract (each direction is explicit, all biases point the same way):

    - Byte counts are upper bounds on compulsory traffic (distinct cells touched, perfect-cache
      assumption): per-access image cardinalities are exact for injective interpretable maps;
      non-injective maps, guarded ([If]) accesses (counted guards-taken), vectorized-write runs, and
      uninterpretable components ([Sub_axis]/[Concat]/dynamic indices — whole-node fallback) only
      over-count, as does summing multiple same-direction accesses of one node (a union bound,
      capped by the node's size). [fp_approx] is [false] only when the count is exact.
    - The op count is an upper bound in the same guards-taken sense, and counts every scalar
      [Unop]/[Binop]/[Ternop] evaluation as one "FLOP" regardless of precision or integerness —
      except the two-operation ternaries [FMA]/[Mul3], which count two (matching [peak_flops]'
      FMA-counted-as-two convention, so an FMA-form kernel scores the same as its mul+add form);
      selection ops [Arg1]/[Arg2]/[Identity] count zero; [Tile_mma] counts its [2*m*n*k]
      multiply-adds once per tile, not per cooperating lane.
    - The one under-counting escape hatch is opaque code — [Staged_compilation] statements and
      merge-buffer reads are invisible to the analysis — and it is flagged: when [opaque] is set,
      the upper-bound reading of the byte and op counts no longer holds. *)

type node_footprint = {
  fp_read_bytes : int;  (** Distinct cells read (including accumulation reads) times byte width. *)
  fp_write_bytes : int;  (** Distinct cells written times byte width. *)
  fp_rmw_bytes : int;
      (** The subset of [fp_write_bytes] written by read-modify-write accumulations ([a_rmw]). *)
  fp_approx : bool;
      (** [true] when any contributing count is an upper bound rather than exact (see the module
          contract for the causes). *)
}
[@@deriving sexp_of]

type summary = {
  per_node : (Tnode.t * node_footprint) list;  (** In first-access program order. *)
  read_bytes : int;  (** Sum of [fp_read_bytes] over [per_node]. *)
  write_bytes : int;  (** Sum of [fp_write_bytes] over [per_node]. *)
  flops : int;  (** Whole-kernel arithmetic-op count (loop extents times per-statement ops). *)
  flops_approx : bool;  (** [true] when guarded ([If]) code contributed (guards-taken bound). *)
  opaque : bool;
      (** [true] when the code contains [Staged_compilation] or merge-buffer reads: some traffic and
          ops are invisible to the analysis, so counts may UNDER-estimate. *)
}
[@@deriving sexp_of]

val analyze : Low_level.t -> summary
(** The whole-kernel extraction: footprints from {!Low_level.affine_accesses} image cardinalities,
    op counts from a loop-nest walk. Input is typically post-[optimize] code — the analysis charges
    whatever materialization the code actually performs, so virtual/inlined nodes never appear and
    recomputation is charged as ops, not bytes. *)

val total_bytes : summary -> int
(** [read_bytes + write_bytes]: the "bytes moved" denominator of arithmetic intensity — an
    accumulation's cells are charged once in each direction. *)

val arithmetic_intensity : summary -> float
(** FLOPs per byte moved, [flops / max 1 (total_bytes)]. *)

val roofline_seconds :
  ?peak_flops:float ->
  ?peak_memory_bandwidth:float ->
  flops:int ->
  bytes:int ->
  unit ->
  float option
(** The roofline lower-bound time: [max (flops/peak_flops) (bytes/peak_memory_bandwidth)] over the
    envelope constants present ([peak_flops] in FLOP/s, [peak_memory_bandwidth] in bytes/s — the
    advisory {!Backend_intf.hardware_limits} fields); [None] when neither is given. Monotone:
    raising either constant never increases the bound. A lower bound only up to the model's
    upper-bound byte/op counts — rank with it, do not predict. *)
