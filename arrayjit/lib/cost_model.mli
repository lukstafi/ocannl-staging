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

module Calibration : sig
  (** The calibration TSV schema (config [autotune_calibration_file], gh-ocannl-491 task 4) and
      the envelope fitter over it (gh-ocannl-514 phase 0): one row per timed candidate, the
      model's inputs and roofline score next to the measured time. This module is the schema's
      single owner — rows are emitted through {!to_line} (by [Autotune]) and read back through
      {!of_line} (by [tools/fit_envelope.exe]), so writer and reader cannot drift apart. *)

  type row = {
    backend : string;
    digest : string;  (** The candidate's digest tag, already shortened at emission. *)
    label : string;
    measured_ms : float;
    model_ms : float option;
        (** The roofline bound under the envelope constants in force at recording time; [None]
            when the model had no coverage (opaque code or no envelope constants). *)
    kernels : int;
    flops : int;  (** Aggregated over the candidate's kernels, like [bytes]. *)
    bytes : int;
    opaque : bool;
  }
  [@@deriving sexp_of]

  val to_line : row -> string
  (** Tab-separated, no trailing newline. [of_line (to_line r)] recovers [r] up to float
      formatting ([measured_ms] and [model_ms] record 6 decimals). *)

  val of_line : string -> row option
  (** [None] on malformed lines (wrong column count, unparseable numbers). *)

  type fit = {
    fit_backend : string;
    fit_rows : int;  (** Scoreable rows: non-opaque with a positive measured time. *)
    fit_opaque : int;  (** Opaque rows, excluded — the model never scores them. *)
    fit_multi_kernel : int;  (** Among scoreable rows; see {!fit} for the aggregate caveat. *)
    fit_violations : int;
        (** Rows whose recorded [model_ms] exceeds [measured_ms]: the envelope in force when
            they were recorded understated this machine's peaks. *)
    fit_peak_flops : (float * string) option;
        (** The implied minimal sound constant and the binding row's [label (digest)]; [None]
            when no scoreable row has a positive count for the leg. *)
    fit_peak_memory_bandwidth : (float * string) option;
  }
  [@@deriving sexp_of]

  val fit : row list -> fit list
  (** Grouped by backend in order of first appearance. The fitted constants are the tightest
      envelope under which the roofline bound is sound on the given data: per row,
      [bound <= measured] requires [peak >= counts/measured] on each leg, so each leg's fit is
      the maximum achieved [counts/time] over the scoreable rows — where "achieved" is by the
      model's own upper-bound counts, exactly the direction bound-soundness needs. Overstating a
      peak only weakens pruning (the bound stays a lower bound); understating one breaks
      fathoming. Multi-kernel rows aggregate per-kernel counts, so their constraints are
      necessary but not sufficient (a sum of per-kernel max-of-legs can exceed the aggregate
      legs); residual violations surface through the continuous agreement check in
      [Autotune.tune] and prompt a refit. *)

  val report : fit -> string
  (** Config-pasteable: [model_peak_*=...] lines under [#] comment lines naming the binding rows
      (config comments must be whole lines). Printed constants carry a ~2e-6 relative bump so
      that 7-significant-digit truncation cannot land below the implied minimum. *)
end
