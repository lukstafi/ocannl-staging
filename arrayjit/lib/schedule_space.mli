(** The partial-schedule decision space (gh-ocannl-514 phase 1): the representation branch-and-bound
    schedule search operates on, per docs/proposals/gh-ocannl-514.md.

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

    Fathoming (phases 2-4) prunes a [Choice] child without forcing its subtree: legality refutes
    every completion below it, or the optimistic bound meets the incumbent. The lazy children are
    that contract's representation — expanding a node must be a decision, not a side effect of
    construction. *)

type placement = Pl_inline | Pl_stage_at of Indexing.symbol | Pl_materialize
[@@deriving sexp_of, compare, equal]
(** One tensor node's placement level. [Pl_stage_at s] materializes the node at loop [s]'s scope —
    recomputed per iteration of the loops outside [s], shared by the loops inside. [Pl_inline] and
    [Pl_materialize] are the poles the greedy flip chain already searches. Loop identity is by
    lowering-local symbol: persistence across fresh lowerings (schedule-cache rebinding) is the
    later phases' concern, like instantiation. *)

type 'a tree =
  | Leaf of 'a  (** A fully committed schedule shape — today, one sketch-seed parameter set. *)
  | Choice of { level : string; children : (string * 'a tree Lazy.t) list }
      (** One open decision: [level] names it, each child is one way to commit it (labelled for
          witnesses and reports), lazily refined. A [Choice] with no children is an infeasible
          node — every completion was filtered out ({!leaves} is empty). Levels appear in
          {e emission order} (the order candidates reach timing), which is not necessarily
          dependency order. *)

val leaves : 'a tree -> 'a list
(** All completions, in tree traversal order — the order the flat enumerations produced, so a
    factored family's [leaves] replaces its seed list drop-in. Forces the whole tree. *)

val enumerate : 'a tree -> ((string * string) list * 'a) list
(** [leaves] with each completion's decision path — the committed [(level, label)] vector that
    identifies the leaf; prefixes of these paths are the partial vectors interior nodes stand
    for. *)

val count_choices : 'a tree -> int
(** Interior (decision) nodes; forces the whole tree. *)

val depth : 'a tree -> int
(** Longest decision chain root-to-leaf; forces the whole tree. *)
