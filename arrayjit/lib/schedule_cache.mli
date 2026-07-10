(** {1 Process-independent schedule identities and the schedule disk cache}

    Support for persisting and replaying {!Schedule.schedule} values (docs/proposals: the autotune
    companion of schedule-ir-optops.md). Schedules embed {!Indexing.symbol}s and {!Tnode.t}s whose
    identities are process-local (global counters), so a schedule value is only meaningful against
    the one lowering it was built for — and every backend [compile] lowers afresh. This module
    gives both a {e canonical}, structural identity:

    - loop symbols are numbered by the preorder position of their binding [For_loop] in the
      optimized code ([Base]),
    - static-index symbols by their position in the [static_indices] list ([Static]) — they occur
      free in the code, so the traversal cannot discover them and they are pre-seeded,
    - symbols minted by schedule ops themselves (e.g. [Split]'s fresh loops) by the index of the
      op that mints them ([Minted]) — replay re-mints them through the {!Schedule} builders,
    - tensor nodes by first occurrence in the same traversal.

    The same traversal renders the code into a canonical string and digests it. The digest is the
    safety guarantee: a schedule saved against a digest is only ever replayed onto code with an
    equal digest, which makes the canonical numbering total and unambiguous by construction —
    nondeterministic lowering degrades to cache misses, never to a schedule applied to the wrong
    loop. *)

open Base

(** Which fresh symbol of a schedule op a [Minted] reference names. *)
type mint_role =
  | Split_outer
  | Split_inner
  | Expand_axis of int  (** The [i]-th fresh symbol of an [Expand_zero]. *)
  | Tensorize_lane
[@@deriving sexp, compare, equal]

(** A process-independent name for a symbol occurring in a schedule. [Base i] is the [i]-th
    [For_loop] binder in preorder of the optimized code the schedule applies to; [Static k] the
    [k]-th static index; [Minted (op, role)] the fresh symbol in [role] of the [op]-th (0-based)
    op of the same saved schedule. *)
type sym_ref = Base of int | Static of int | Minted of int * mint_role
[@@deriving sexp, compare, equal]

(** {!Schedule.optop} with symbols replaced by {!sym_ref}s and tensor nodes by their canonical
    first-occurrence index. *)
type saved_optop =
  | Split of { axis : sym_ref; factor : int; outer : Low_level.axis_type; inner : Low_level.axis_type }
  | Swap of { outer : sym_ref; inner : sym_ref }
  | Retype of { axis : sym_ref; ty : Low_level.axis_type }
  | Unroll of { axis : sym_ref; materialize : bool }
  | Stage of {
      source : int;
      tile_loops : sym_ref list;
      shared : bool;
      cooperative : int option;
      hoisted : bool;
    }
  | Privatize of { target : int; over : sym_ref }
  | Expand_zero of { tn : int }
  | Tensorize of { i : sym_ref; j : sym_ref; k : sym_ref; simd_width : int }
[@@deriving sexp, compare, equal]

type saved_schedule = saved_optop list [@@deriving sexp, compare, equal]

type canonical
(** The canonical identity of one [Low_level.optimized] value: the digest, the loop-binder and
    static-symbol numbering, and the tensor-node numbering. *)

val canonicalize : ?static_indices:Indexing.static_symbol list -> Low_level.optimized -> canonical
(** Walks the optimized code once in preorder, numbering [For_loop] binders, first-occurrence
    tensor nodes (their dims, precision, and hoisted-packing eligibility
    [Schedule.hoistable_constant] enter the digest — schedule validity depends on operand
    constancy, gh-ocannl-470), and rendering the canonical form. [static_indices] must be the
    same list the code was lowered with ({!Indexing.bound_symbols} of the compile's bindings). *)

val digest : canonical -> string
(** Hex digest of the canonical rendering. Equal digests mean structurally identical code, hence
    interchangeable canonical numberings. *)

val complete : canonical -> bool
(** [false] when the code contains constructs the canonical rendering cannot capture
    ([Staged_compilation] closures, unbound or shadowed loop symbols). Incomplete canonical forms
    must not be used as {b disk}-cache keys (distinct programs could collide); within one process
    they still support {!to_saved}/{!of_saved} round-trips. *)

val tn_of_ref : canonical -> int -> Tnode.t
(** The tensor node at a canonical index. Raises [Invalid_argument] on out-of-range. *)

(** {2 Symbol resolution registries}

    A [registry] resolves the symbols of a particular compile's code to [sym_ref]s: base and
    static symbols through its {!canonical}, schedule-minted symbols through entries recorded by
    {!to_saved} / {!of_saved}. Use it to translate loops of {e transformed} code (base code with a
    schedule prefix applied) into references a schedule extension can persist. *)

type registry

val base_registry : canonical -> registry
val resolve : registry -> Indexing.symbol -> sym_ref option
val resolve_tn : registry -> Tnode.t -> int option

val to_saved : registry -> Schedule.schedule -> saved_schedule * registry
(** Serializes a schedule built against the registry's compile (e.g. by {!Schedule.default_gpu}),
    recording each op's minted symbols in the returned registry (op indices continue from the
    number of ops already recorded in the input registry, so extensions of replayed prefixes stay
    consistent). Raises [Invalid_argument] when an op references a symbol or tensor node the
    registry cannot resolve. *)

val of_saved : canonical -> saved_schedule -> Schedule.schedule * registry
(** Replays a saved schedule against a (fresh) compile's canonical form: base and static
    references resolve through [canonical], minting ops go through the {!Schedule} builders and
    their fresh symbols are recorded for later references. Raises [Invalid_argument] on dangling
    references (canonical mismatch — always guard with {!digest} equality first). *)

(** {2 The disk cache} *)

type entry = {
  version : int;
  backend : string;
  source_digest : string;
  saved : saved_schedule;
  segments : (string * saved_schedule) list option; [@sexp.option]
      (** A fissioned winner (docs: per-fission-segment tuning): per-segment schedules keyed by
          the {e pre-schedule} segment's canonical digest — replay routes each of
          {!Schedule.fission_scheduled}'s [`Normal] segments through this association ([saved] is
          then empty; unmatched segments degrade to the empty schedule). [None] for whole-routine
          schedules. *)
  best_ms : float;  (** The winning candidate's measured time, for diagnostics. *)
  baseline_ms : float;  (** The unscheduled baseline's measured time, for diagnostics. *)
}
[@@deriving sexp]

val entry_version : int
(** Bumped when the canonical rendering or the saved-schedule format changes; stale entries are
    ignored by {!lookup}. *)

val cache_key : canonical -> backend:string -> string
(** Filename-safe cache key: the digest plus the backend name. Callers time kernels on a concrete
    device, so include anything else that distinguishes performance environments in [backend]
    (e.g. a device id) if needed. *)

val store : dir:string -> key:string -> entry -> unit
(** Writes the entry to [dir]/[key].sexp, creating [dir] (and parents) if missing. Tolerates
    concurrent writers (last write wins; writes go through a temp file + rename). *)

val lookup : dir:string -> key:string -> entry option
(** [None] on missing file, unparsable content, or version/digest mismatch. *)
