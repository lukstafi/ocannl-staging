(** The partial-schedule decision space (gh-ocannl-514 phases 1-2): the representation
    branch-and-bound schedule search operates on, per docs/proposals/gh-ocannl-514.md.

    A search node is a {e partial schedule shape} — some decisions committed, the rest open. Two
    kinds of decision level make up the space:

    - {b Placement levels}: per policy-decided tensor node, where its value lives on the
      inline/compute-at/materialize spectrum ({!placement}). The Inline/Materialize poles are the
      landed decision vector ([Context.decide_inline], [optimized.flip_candidates]);
      [Pl_stage_at] is the compute-at middle, represented here from the start so it is a decision
      level rather than a bolt-on — its instantiation (a [Schedule.Stage] at the given loop) lands
      with the later phases.
    - {b Family levels}: which sketch pipeline, and its parameters — represented as a refinement
      {!tree} whose choices may depend on earlier commitments (a staging shape constrains which
      geometries remain; a twin exists only for staged geometries). The sketch families factor
      into such trees instead of flat enumerations ([Autotune.matmul_sketch_tree] is the first);
      today's seed lists are exactly the trees' {!leaves}.

    Phase 2 makes the tree carry {e legality over partial shapes}: each choice's children are
    judged with the same three-valued logic as [Schedule.op_legality], quantified over
    completions — {!child.Refuted} asserts no completion below the child is valid (with the
    violated constraint as witness, a decline explanation produced before any compilation,
    gh-ocannl-479), {!child.Unknown} never fathoms, and construction commits to a verdict at the
    moment the domain is built rather than leaving branches to fail at candidate compile. A
    fourth verdict, {!child.Excluded}, keeps the landed policy/legality separation (gh-ocannl-555:
    preferences suppress heuristic caps only, legality is non-negotiable) visible in the space
    itself: an [Excluded] branch is valid but suppressed by default-policy search economy — a
    driver may re-propose it by lifting the policy; it must never re-propose a [Refuted] one. *)

type placement = Pl_inline | Pl_stage_at of Indexing.symbol | Pl_materialize
[@@deriving sexp_of, compare, equal]
(** One tensor node's placement level. [Pl_stage_at s] materializes the node at loop [s]'s scope —
    recomputed per iteration of the loops outside [s], shared by the loops inside. [Pl_inline] and
    [Pl_materialize] are the poles the greedy flip chain already searches. Loop identity is by
    lowering-local symbol: persistence across fresh lowerings (schedule-cache rebinding) is the
    later phases' concern, like instantiation. *)

type 'a child =
  | Child of 'a tree Lazy.t  (** Feasible as far as construction can decide. *)
  | Unknown of string * 'a tree Lazy.t
      (** The [Op_unknown] analogue: construction cannot decide (the witness says why — e.g. a
          permissively detected site whose mis-detections only candidate compilation rejects), so
          the subtree stays enumerable and a fathom must never prune it; its completions settle at
          candidate compile. *)
  | Excluded of string * 'a child Lazy.t
      (** Valid but not proposed: default-policy search economy (a dominated or degenerate shape,
          a twin whose measured cost exceeds its possible win). The witness states the policy;
          the payload is the {e same judgment with only this policy lifted} — forcing it is the
          driver's lift operation ({!lift_excluded}), and the result remains subject to legality
          (a lifted branch may still be [Refuted] on other grounds). {!leaves}, {!enumerate} and
          the collectors treat the exclusion as opaque: they neither enumerate nor descend into
          the payload. *)
  | Refuted of string
      (** The [Op_illegal] analogue, quantified over completions: {e no} completion below this
          child satisfies the family's contract — an op precondition, a resource cap, or a
          statically-decided emission constraint refutes them all. The witness is the violated
          constraint: a decline explanation produced before any compilation. *)

and 'a tree =
  | Leaf of 'a  (** A fully committed schedule shape — today, one sketch-seed parameter set. *)
  | Choice of { level : string; children : (string * 'a child) list }
      (** One open decision: [level] names it, each child one way to commit it (labelled, with
          its verdict). A [Choice] whose children are all [Refuted]/[Excluded] is an infeasible
          node ({!leaves} is empty). Levels appear in {e emission order} (the order candidates
          reach timing), not necessarily dependency order. Verdicts are decided when the parent
          is constructed — fathoming needs them without expansion — while feasible children's
          subtrees stay lazy: expanding one is a decision, not a side effect of construction. *)

val leaves : 'a tree -> 'a list
(** All completions of [Child] and [Unknown] branches, in tree traversal order — the order the
    flat enumerations produced, so a factored family's [leaves] replaces its seed list drop-in.
    Forces the reachable subtrees. *)

val enumerate : 'a tree -> ((string * string) list * 'a) list
(** {!leaves} with each completion's decision path — the committed [(level, label)] vector that
    identifies the leaf; prefixes of these paths are the partial vectors interior nodes stand
    for. *)

val refutations : 'a tree -> ((string * string) list * string) list
(** Every [Refuted] child with its decision path (ending at the refuted commitment) and witness:
    the shape's pre-compilation decline explanations (gh-ocannl-479), in traversal order. *)

val exclusions : 'a tree -> ((string * string) list * string) list
(** Every [Excluded] child with path and policy witness — the branches a driver could re-propose
    by lifting default-policy economy. *)

val unknowns : 'a tree -> ((string * string) list * string) list
(** Every [Unknown] child with path and witness — the branches whose verdicts only candidate
    compilation settles; a fathom must treat them as feasible. *)

val lift_excluded : 'a child -> 'a child
(** The policy-lift operation: [Excluded (_, payload)] becomes the forced payload — the same
    judgment with only that one policy lifted, still subject to legality; any other child is
    returned unchanged. Lifting is per exclusion: a payload may itself contain further
    [Excluded] children deeper in its subtree. *)

val count_choices : 'a tree -> int
(** Interior (decision) nodes reachable through [Child]/[Unknown]; forces them. *)

val depth : 'a tree -> int
(** Longest reachable decision chain root-to-leaf; forces reachable subtrees. *)
