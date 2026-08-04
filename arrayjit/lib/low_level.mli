(** {1 A for-loop-based array language and backend-agnostic optimization} *)

open Base

(** {2 Global references} *)

module Scope_id : sig
  type t = { tn : Tnode.t; scope_id : int } [@@deriving sexp_of, equal, hash, compare]
  type comparator_witness

  val comparator : (t, comparator_witness) Base.Comparator.t
end

type scope_id = Scope_id.t = { tn : Tnode.t; scope_id : int }
[@@deriving sexp_of, equal, hash, compare]

val get_scope : Tnode.t -> scope_id

(** {2 Low-level representation} *)

(** How a loop's iterations map to hardware; see docs/proposals/axis-types-for-loops.md. [Serial] is
    an ordinary for-loop. [Grid] / [Workgroup] bind the loop index to a GPU grid / workgroup (block,
    threadgroup) hardware index instead of looping; [Workgroup_reduce] is a [Workgroup] axis
    participating in a workgroup-cooperative reduction (see the contract below). [Unrolled] is
    emitted as the repeated body with substituted constants. [Vectorized] renders eligible bodies as
    explicit SIMD code — elementwise statements via vector extensions / packed loads (gh-ocannl-164
    / gh-ocannl-463), a single recognized accumulation as independent accumulator chains with a
    horizontal reduce at exit on CPU backends (gh-ocannl-468) — and everything else as a serial loop
    annotated with the backend's vectorization pragmas when it provides them (a plain, un-annotated
    serial loop for accumulating bodies, whose loop-carried dependency the pragmas would deny) —
    like the hardware kinds, the annotating pass asserts iteration independence or, for a recognized
    accumulation, licenses reassociating it. Hardware slots are positional: among a kernel's loops
    of one kind, the innermost binds [.x], then [.y], [.z]. Annotated loops must have [from_ = 0]
    and iterations with no cross-iteration dependencies ([Vectorized] accumulations again excepted).
    [Workgroup_reduce] is the labelled exception; its body must either stage its communication
    explicitly through workgroup-shared nodes and barriers (rendered by binding the index like
    [Workgroup]), or be a single accumulation statement [acc = op(acc, contrib)] over an
    associative-commutative [op] with the accumulator's indices free of the loop index — the
    renderer then owns the communication: warp/simdgroup shuffles on GPU backends (gh-ocannl-462),
    the plain serial loop on CPU backends. Like [Vectorized], the annotation licenses reassociating
    the (floating-point) reduction. *)
type axis_type = Serial | Grid | Workgroup | Workgroup_reduce | Unrolled | Vectorized
[@@deriving sexp, compare, equal]

val axis_type_label : axis_type -> string
(** Loop keyword used by the human-readable printers: plain ["for"] for [Serial], ["for@<axis>"]
    otherwise. *)

(** Cases: [t] -- code, [scalar_t] -- single number at some precision. *)
type t =
  | Noop
  | Comment of string
  | Staged_compilation of (unit -> PPrint.document)
  | Seq of t * t
  | For_loop of {
      index : Indexing.symbol;
      from_ : int;
      to_ : int;
      body : t;
      trace_it : bool;
      axis : axis_type;
    }
  | Zero_out of Tnode.t
  | Set of {
      tn : Tnode.t;
      idcs : Indexing.axis_index array;
      llsc : scalar_t;
      mutable debug : string;
    }
  | Set_dynamic of {
      tn : Tnode.t;
      idcs : Indexing.axis_index array;
          (** Static everywhere except [dyn_axis] (a [Fixed_idx 0] placeholder there). *)
      dyn_axis : int;  (** Which [idcs] slot is replaced by [dyn_value] at codegen time. *)
      dyn_value : scalar_arg;
          (** Integer-valued index spliced into the row-major offset at [dyn_axis]. *)
      llsc : scalar_t;
      mutable debug : string;
    }
      (** A scatter: like [Set] but the write lands at a runtime row of axis [dyn_axis] — the write
          counterpart of {!scalar_t.Get_dynamic}. gh-466: produced only by
          {!rewrite_one_hot_reductions} (transposed one-hot pattern, the embedding-table gradient);
          never constructed by [Assignments] lowering. Schedule analyses must treat this write as
          statically unknown: loops whose index reaches [dyn_value] carry a cross-iteration write
          dependency and must stay serial (the deterministic no-atomics invariant). *)
  | Set_from_vec of {
      tn : Tnode.t;
      idcs : Indexing.axis_index array;
      length : int;
      vec_unop : Ops.vec_unop;
      arg : scalar_arg;
      mutable debug : string;
    }
  | Set_local of scope_id * scalar_t
  | Declare_local of { id : scope_id; needs_init : bool }
  | Workgroup_barrier
      (** Workgroup-scoped synchronization ([__syncthreads()] / [threadgroup_barrier]). An opaque
          effectful statement: no CSE, hoisting, or code motion across it. Grid-scoped
          synchronization is deliberately not representable. *)
  | If of { cond : scalar_arg; body : t }
      (** Guarded statement: [body] executes iff [cond] is nonzero (renders as
          [if (cond != 0) { body }]). Introduced by launch-extent guards on hardware-annotated loops
          (docs/proposals/axis-types-for-loops.md §2); [simplify_llc] erases a guard whose condition
          an interval proves. A conditional write is never a definite write; virtualization treats
          guarded computations as non-inlineable in v1. *)
  | Tile_mma of {
      d : Tnode.t * Indexing.axis_index array;  (** Accumulator block base. *)
      a : Tnode.t * Indexing.axis_index array;
      b : Tnode.t * Indexing.axis_index array;
      ta : bool;  (** [a] is stored transposed: its tile axes are [k, i]-major rather than [i, k]. *)
      tb : bool;  (** [b] is stored transposed: its tile axes are [j, k]-major rather than [k, j]. *)
      m : int;
      n : int;
      k : int;  (** Covered block extents (multiples of the backend's intrinsic tile). *)
      ldd : int;
      lda : int;
      ldb : int;
          (** Leading-dimension strides in elements, recorded by {!Schedule.optop.Tensorize}: the
              tnode's minor dim in the plain last-two-axes case, larger when interior batch axes
              sit between the tile roles (gh-ocannl-528). *)
      lane : Indexing.symbol;  (** The cooperating [Workgroup] axis (extent = SIMD width). *)
      fallback : t;  (** Semantically equivalent scalar micro-kernel over fresh serial symbols. *)
    }
      (** Cooperative tile multiply-accumulate (docs/proposals/tensorize-mma.md):
          [d[i,j] += Σ_{l<k} a[i,l] * b[l,j]] for [i < m], [j < n], relative to the operands' base
          index vectors, executed jointly by the threads of the [lane] axis (tensor cores /
          [simdgroup_matrix]). Each operand's tile is a 2-D slice: minor tile axis on the tnode's
          last axis (stride 1), major tile axis at the recorded leading-dimension stride (with
          [ta]/[tb] the stored layout is the role's transpose and emissions use the hardware
          transpose flag); the base indices must not mention [lane]. Constructed by schedule
          transforms only ({!Schedule.optop.Tensorize}), after the optimization pipeline. Backends
          without an MMA hook render [fallback] under an [if (lane == 0)] guard. Validates like
          {!Workgroup_barrier} plus a write of [d] for the coverage rule; see {!validate_parallel}.
      *)
[@@deriving sexp_of, equal]

and scalar_t =
  | Local_scope of { id : scope_id; body : t; orig_indices : Indexing.axis_index array }
  | Get_local of scope_id
  | Get of Tnode.t * Indexing.axis_index array
  | Get_dynamic of {
      tn : Tnode.t;  (** The gathered table; treated as a read of [tn], like [Get]. *)
      idcs : Indexing.axis_index array;  (** Static everywhere except [dyn_axis]. *)
      dyn_axis : int;  (** Which [idcs] slot is replaced by [dyn_value] at codegen time. *)
      dyn_value : scalar_arg;
          (** Integer-valued index spliced into the row-major offset at [dyn_axis]. gh-343: produced
              only by {!rewrite_one_hot_reductions}; never escapes low-level / backend codegen. *)
    }
  | Get_merge_buffer of Tnode.t * Indexing.axis_index array
  | Ternop of Ops.ternop * scalar_arg * scalar_arg * scalar_arg
  | Binop of Ops.binop * scalar_arg * scalar_arg
  | Unop of Ops.unop * scalar_arg
  | Constant of float
  | Constant_bits of int64  (** Direct bit representation, primarily for uint4x32 *)
  | Embed_index of Indexing.axis_index
[@@deriving sexp_of, equal, compare]

and scalar_arg = scalar_t * Ops.prec [@@deriving sexp_of, equal, compare]
(** The argument precision is preserved in heterogeneous precision operation arguments, and is
    ignored (overridden) in homogeneous precision operations. *)

val scalar_precision : scalar_t -> Ops.prec
val apply_op : Ops.op -> scalar_t array -> scalar_t
val flat_lines : t list -> t list
val unflat_lines : t list -> t
val loop_over_dims : int array -> body:(Indexing.axis_index array -> t) -> t
val unroll_dims : int array -> body:(Indexing.axis_index array -> offset:int -> t) -> t

val loop_over_padding_region :
  dims:int array -> padding:Ops.axis_padding array -> body:(Indexing.axis_index array -> t) -> t
(** Generate loops that iterate only over the padding margins of a tensor. For dimensions with
    padding, generates separate loops for left margin, middle (recursing), and right margin. The
    middle region continues recursing to find padding in other dimensions. *)

val has_accumulation : t -> bool
(** Whether the tree carries a read-modify-write accumulation: some [Set] (resp. [Set_local]) reads
    its own target — a loop-carried dependency through memory when the written cell does not vary
    with an enclosing loop. Conservative: [Local_scope] contents count as reading anything, and
    [Tile_mma] and (gh-466) [Set_dynamic] accumulate by construction. Used by the autotune menu and
    by codegen fallbacks that must not assert iteration independence (e.g. vectorization pragmas)
    over an accumulating body (gh-ocannl-468). *)

(** {2 Hardware axis analyses}

    Phase B of docs/proposals/axis-types-for-loops.md. Hardware slot assignment is positional, not
    stored in the IR: among a kernel's annotated loops of one kind, the innermost binds [.x] (slot
    0), the next [.y], then [.z]; [Workgroup] and [Workgroup_reduce] share the block/threadgroup
    slot space. *)

type launch_dims = { grid : int array; block : int array } [@@deriving sexp_of, equal]
(** Arrays of length 3 ([.x], [.y], [.z]); all-1s for all-[Serial] code. *)

type hardware_axis_info = {
  ha_index : Indexing.symbol;
  ha_kind : [ `Grid | `Workgroup ];
  ha_slot : int;  (** Positional: the innermost same-kind loop binds [.x] = slot 0. *)
  ha_from_ : int;
  ha_extent : int;  (** [to_ - from_ + 1]. *)
}

val hardware_axes : t -> hardware_axis_info list
(** All hardware-annotated loops in pre-order, with their positional slots. *)

val launch_dims : t -> launch_dims
(** Per-slot maximum extents over the kernel's annotated loops. *)

val validate_parallel : Tnode.Placements.t -> t -> unit
(** Backend-independent well-formedness of hardware annotations (axis-types proposal §2); a no-op
    for all-[Serial] code. Raises [Invalid_argument] on structural violations: nonzero [from_], more
    than 3 slots per kind, annotated loops inside [Local_scope] bodies, barriers under divergent
    extents or [If] guards, writes to materialized nodes not nested under annotated loops covering
    {e every} active (non-unit) hardware dimension — launch dimensions are global to the kernel, so
    an uncovered dimension executes the write once per hardware index — and whole-node [Zero_out] of
    materialized nodes in multi-threaded kernels (nesting never distributes it). Cannot prove
    iteration independence — that is the annotating pass's obligation. *)

val validate_parallel_classified : Tnode.Placements.t -> t -> unit
(** Internal backend-facing variant of {!validate_parallel}; transports a validation
    [Invalid_argument] as a typed {!Schedule_outcome.Illegal_schedule}. *)

val guard_annotated_extents : should_guard:([ `Grid | `Workgroup ] -> bool) -> t -> t
(** Wraps bodies of annotated loops whose extent is below their slot's launch dimension in
    [If (index < extent)] guards, for the kinds the backend binds in hardware. *)

(** {2 Optimization} *)

type virtualize_settings = {
  mutable enable_device_only : bool;
  mutable max_visits : int;
  mutable max_inline_reduction : int;
      (** Recompute-cost cap for inlining: a node whose setters have enclosing reduction loops
          (loops not appearing in the setter's indices) with a trip-count product exceeding this
          value is never virtualized. Negative values disable the cap. *)
  mutable max_tracing_dim : int;
  mutable inline_scalar_constexprs : bool;
  mutable inline_simple_computations : bool;
  mutable inline_complex_computations : bool;
}

val virtualize_settings : virtualize_settings

type visits =
  | Visits of int
  | Recurrent
      (** A [Recurrent] visit is when there is an access prior to any assignment in an update. *)
[@@deriving sexp, equal, variants]

type traced_array = {
  tn : Tnode.t;
  assignments : int array Base.Hash_set.t;
  accesses : (int array, visits) Base.Hashtbl.t;
  mutable zero_initialized_by_code : bool;
  mutable zeroed_out : bool;
  mutable read_before_write : bool;
      (** The node is read before it is written (i.e. it is recurrent). *)
  mutable read_only : bool;
      (** Surprisingly, the notions of read-only and of constant memory mode come apart: small
          hosted constants are not read-only because they are initialized on devices by being
          assigned to; and a volatile memory mode is read-only from the devices' perspective. *)
  mutable is_scalar_constexpr : bool;
      (** True only if the tensor node has all axes of dimension 1, is either zeroed-out or assigned
          before accessed, is assigned at most once, and from an expression involving only constants
          or tensor nodes that were at the time is_scalar_constexpr. *)
  mutable is_accessing : bool;
      (** False only if the tensor node is built from index embeddings and scalar constant
          expressions. *)
  mutable is_complex : bool;
      (** True only if the tensor node is built from a genuinely complex scalar computation (one
          that accesses other non-constexpr computations). Sharing a loop symbol with another tensor
          does not, by itself, make a node complex (see #134). *)
  mutable prefers_virtual_one_hot : bool;
      (** True when at least one setter for this tensor is a one-hot selector assignment, i.e. a
          [Cmpeq] between the embedded range iterator and a loop-variable-free expression. When
          [has_non_one_hot_setter] is false this tensor is exempt from the visit-count
          [Never_virtual] rule (task-73617488). *)
  mutable has_non_one_hot_setter : bool;
      (** True when at least one setter is NOT a one-hot selector (including [Set_from_vec]). A
          tensor with [prefers_virtual_one_hot && not has_non_one_hot_setter] is the candidate for
          the one-hot virtualizer exemption. *)
  mutable is_range_producer : bool;
      (** True when at least one [Set] assigns this tensor from a bare [Embed_index] scalar, i.e.
          the tensor is a [Range_over_offsets] producer. Used by the indirect arm of
          [is_one_hot_selector_assignment] to prove that a [Get(rtn, [k])] will inline to
          [Embed_index k] rather than arbitrary values (task-73617488). *)
  mutable inline_reduction_extent : int;
      (** The largest product of trip counts of loops that enclose one of the node's setters without
          appearing in its indices (i.e. reduction loops). Inlining the computation replays these
          loops at every read site; compared against [virtualize_settings.max_inline_reduction]. *)
  mutable read_by_other : bool;
      (** True when some statement other than the node's own setters reads the node. Unlike
          [accesses], same-cell reads count, while a setter's own read-modify-write does not. Gates
          the recompute-cost guard: a node never read in the routine has no inlining cost, so it
          must stay eligible for virtual dead-code elimination. *)
}
[@@deriving sexp_of]

val get_node : (Tnode.t, traced_array) Base.Hashtbl.t -> Tnode.t -> traced_array
val optimize_integer_pow : bool ref

type traced_store = (Tnode.t, traced_array) Base.Hashtbl.t [@@deriving sexp_of]

type optimize_ctx = {
  computations : (Tnode.t, (Indexing.axis_index array option * t) list) Base.Hashtbl.t;
      (** The computations (of the tensor node) are retrieved for optimization just as they are
          populated, so that the inlined code corresponds precisely to the changes to the arrays
          that would happen up till that point. Within the code blocks paired with an index tuple,
          all assignments and accesses must happen via the index tuple; if this is not the case for
          some assignment, the node cannot be virtual. Currently, we only allow for-loop symbols in
          assignment indices of virtual nodes. *)
  placements : Tnode.Placements.t;
      (** Per-compilation-lineage memory-mode resolution
          (docs/proposals/context-scoped-memory-modes.md): the pipeline's placement decisions
          (Virtual / Local / On_device) land here, seeded by and never written back to the tnodes'
          declared intent ({!Tnode.field-memory_mode}). *)
  alias_candidates : Hash_set.M(Tnode).t;
      (** gh-ocannl-489 liveness-based buffer aliasing: nodes the memory planner may place at
          overlapping byte ranges within the routine's working pool (decided per compile, before
          codegen). Codegen must not emit the [restrict] qualifier for these parameters — whether a
          candidate pair actually shares bytes is settled only at link time, and an aliased
          [restrict] pair is a miscompile. *)
}
[@@deriving sexp_of]

val empty_optimize_ctx : unit -> optimize_ctx

val copy_optimize_ctx : optimize_ctx -> optimize_ctx
(** A shallow-copy fork of the lineage state ([computations] and [placements] tables): the copy sees
    everything decided so far; its later mutations are invisible to the original and to sibling
    copies. Backend [compile] forks the incoming context's [optimize_ctx] through this, so sibling
    candidate compiles from one frontier are hermetic. *)

type optimized = {
  traced_store : traced_store;
  optimize_ctx : optimize_ctx;
  llc : t;
  merge_node : Tnode.t option;
  workgroup_shared : Set.M(Tnode).t;
      (** [Local]-memory-mode nodes to be placed in workgroup-shared memory ([__shared__] /
          [threadgroup]) instead of kernel-local arrays. Populated by schedule transforms; empty for
          unscheduled code. See docs/proposals/axis-types-for-loops.md. *)
  simdgroup_fragments : Set.M(Tnode).t;
      (** [Local]-memory-mode accumulator tiles whose init-load, serial reduction and store-back
          form one per-simdgroup fragment lifetime. Backends without a fragment rendering ignore the
          marking and use the ordinary local-array code; Metal maps the marked region to a
          persistent [simdgroup_matrix] array. *)
  swizzled : Set.M(Tnode).t;
      (** Nodes stored in an XOR-swizzled layout (docs/proposals/tensorize-mma.md, "Swizzled
          staging"): codegen remaps every element access [flat = P*C + col] (with [C] the minor dim,
          [P] the linearized prefix) to [P*C + (col lxor (P land (C-1)))] — a bijection on the
          buffer, so the IR-level semantics are unchanged; only the physical layout differs,
          spreading same-column accesses across shared-memory banks. Populated by
          [Schedule.Stage ~swizzle:true]. Renderings that assume a row-major layout must decline
          swizzled nodes. *)
  zero_fringe : Set.M(Tnode).t;
      (** Schedule-minted staged tiles whose whole index space is safe to read: slots outside the
          staged source region (edge tiles of a non-dividing or padded staging, gh-ocannl-485) hold
          0 — the add-reduce accumulation identity — written by the load nest's [Where]-form edge
          guards or by the host-side constant packing. [Schedule.Tensorize] consults this to
          discharge pad guards on the intrinsic path. *)
}
[@@deriving sexp_of]

val optimize :
  optimize_ctx ->
  unoptim_ll_source:(PPrint.document -> unit) option ->
  ll_source:(PPrint.document -> unit) option ->
  name:string ->
  Indexing.static_symbol list ->
  t ->
  optimized

val reads_scope_before_set : scope_id -> t -> bool
(** [reads_scope_before_set id body] returns [true] if [id] is read (via [Get_local]) before the
    first definitely-executed [Set_local id] in [body]. Use this at code-generation time to decide
    whether a [Local_scope] or [Declare_local] declaration needs a zero initializer. *)

val simplify_llc : Indexing.static_symbol list -> t -> t
(** Top-down algebraic simplification with interval-driven comparison folding (in particular, it
    erases [If] guards whose conditions the loop extents prove). Called internally by [optimize];
    exposed for [Schedule.apply], whose transforms construct guards after the pipeline's simplify
    already ran (docs/proposals/schedule-ir-optops.md §2), and for testing. Pure and idempotent. *)

val rewrite_one_hot_reductions : ?static_indices:Indexing.static_symbol list -> t -> t
(** gh-343: rewrites the narrow one-hot embedding pattern -- an [Add] reduction over a loop variable
    [k] whose body selects an embedding-table row via [k == index_expr] (a logical one-hot) -- into
    a guarded dynamic gather ({!Get_dynamic}) that reads the table row at [index_expr] directly,
    with an in-range guard returning 0 out of [\[0, vocab_size)] to preserve the one-hot semantics.
    The guard is constructed generically and interval analysis
    (docs/proposals/interval-analysis-scalar-t.md) erases the conjuncts it can prove -- from the
    index precision's machine range, loop extents seeded from [static_indices], and settled
    per-tensor bounds ({!Tnode.bounds_state}).

    gh-466: also rewrites the {e transposed} one-hot pattern -- the embedding-table gradient
    [for k in \[0, V): tn[.., k, ..] += (k == index_expr) * g] where the loop variable indexes the
    written tensor itself -- into a guarded dynamic scatter-accumulate ({!Set_dynamic}):
    [if in_range(index_expr): tn[.., index_expr, ..] += g], dropping the O(V) per-position work
    (llm.c's deterministic encoder backward, docs/research/llmc-lessons.md B5). The enclosing
    position loops keep their original serial order and the schedule analyses never parallelize over
    a dynamically-written node, preserving determinism without atomics.

    Unmatched or unsupported reductions are left unchanged. Called internally by [optimize] between
    [simplify_llc] and [eliminate_common_subexpressions]; exposed for testing. *)

val eliminate_common_subexpressions : t -> t
(** Eliminates common subexpressions within each statement's scalar expression tree. Replaces
    duplicate [Local_scope] nodes (structurally identical modulo [scope_id]) with [Get_local]
    references to the first occurrence. Called internally by [optimize]; exposed for testing. *)

val hoist_cross_statement_cse : t -> t
(** Hoists shared [Local_scope] computations from sibling statements to the enclosing scope. When
    two or more sibling statements share an alpha-equivalent [Local_scope] node, the computation is
    extracted as a [Declare_local] + body preceding the first user, and all occurrences are replaced
    with [Get_local]. *)

val input_and_output_nodes : optimized -> (Set.M(Tnode).t * Set.M(Tnode).t) * Tnode.t option
(** Inputs are the materialized read-only and read-before-write (within the code) non-constant
    non-merge nodes. They are inputs in a broad sense, as they could be recurrent nodes or
    parameters. Outputs are all the materialized nodes written-to by the code. The last returned
    component is the input merge node, if used in the code. *)

val loop_bounds : t -> (Indexing.symbol * (int * int)) list
(** All [For_loop] bindings within the code (loop symbols are unique within a routine), with
    inclusive iteration bounds — the box environment for {!Affine} queries. *)

val scope_value_syms : t -> (int, Indexing.symbol list) Base.Hashtbl.t
(** The value-dependence symbols of statement-level scalar scope-locals, whole-code: per scope id,
    the union of the symbols its statement-level assignments depend on, transitively through
    [Get_local] references (such a local may be assigned in one statement and read in another).
    Assignments inside [Local_scope] bodies are not recorded — a scope id is re-instantiated at
    every use site with per-site loop symbols, and scope-internal flow is covered lexically by the
    value scans. Consumed by the setter value scans so a value routed through a scope-local is not
    laundered of its symbols (gh-494 per-thread value-variance). *)

val scalar_value_syms :
  locals:(int, Indexing.symbol list) Base.Hashtbl.t -> scalar_t -> Indexing.symbol list
(** Loop symbols a scalar expression's value depends on syntactically — index symbols of reads,
    embedded indices, dynamic-index sub-expressions — resolving scope-locals through [locals] (from
    {!scope_value_syms}). *)

val affine_accesses : t -> Tnode.t Affine.access list
(** gh-494 waypoint 1: the routine's tensor-node accesses as explicit affine relations
    ({!Affine.access}), extracted from (typically optimized) code, in program order (a statement's
    right-hand-side reads precede its write; [Local_scope] bodies are descended into at their use
    site; [Tile_mma] is traversed through its scalar [fallback]). Not represented: scope-locals,
    merge-buffer reads, and opaque [Staged_compilation] — callers needing exhaustiveness must check
    for the latter separately. *)

val buffer_access_spans : stmt_serial:bool -> t list -> (Tnode.t, int * int) Base.Hashtbl.t option
(** gh-ocannl-489 liveness-based buffer aliasing: per-tnode access span over the final
    (post-schedule, post-fission) code of a routine, as a closed interval of positions; the input is
    the routine's kernels in execution order (a singleton when not fissioned). With
    [stmt_serial:true] every top-level statement gets its own position — sound only for backends
    where consecutive top-level statements of one compiled procedure are fully synchronized (the C
    backends); with [stmt_serial:false] all statements of a segment share one position, since GPU
    kernels have no grid-wide synchronization between top-level statements. Returns [None] when the
    code contains [Staged_compilation] (opaque accesses: no aliasing plan can be trusted). *)

val sink_zero_outs : t -> t
(** gh-ocannl-489 follow-up: sinks each top-level [Zero_out] to just before the first later
    top-level statement accessing the zeroed node ([Train.grad_update]'s up-front [zero_grads] block
    otherwise starts every gradient's live span at that block, nesting the backprop chain's
    intervals and defeating the arena planner). Sound: a [Zero_out] commutes with statements not
    accessing the node; it never crosses such an access, a [Staged_compilation], or a
    [Workgroup_barrier]. Apply to whole-routine code BEFORE scheduling/fission. *)

(** {2 Printing} *)

val code_hum_margin : int ref

val function_header_doc :
  ?name:string -> ?static_indices:Indexing.static_symbol list -> unit -> PPrint.document

val get_ident_within_code : ?no_dots:bool -> ?blacklist:string list -> t array -> Tnode.t -> string

val to_doc_cstyle :
  ?name:string -> ?static_indices:Indexing.static_symbol list -> unit -> t -> PPrint.document
(** Adheres more to the C syntax, outputs implicit type casts. *)

val to_doc :
  ?name:string -> ?static_indices:Indexing.static_symbol list -> unit -> t -> PPrint.document
(** Adheres to the %cd syntax. *)
