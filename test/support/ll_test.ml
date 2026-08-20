(** Hand-built {!Ir.Low_level} test harness: builders, exhaustive traversals, and the executed-leg
    machinery, shared by the tests that construct IR the [Assignments] pipeline never emits
    (gh-ocannl-600).

    Why hand-built IR at all: high-level lowering gives each assignment its own loop nest, so shapes
    like two tensors sharing one for-loop, a [Get] of a diagonal at two distinct call-site symbols,
    or sibling [Local_scope]s reading a node a later statement overwrites are unreachable through
    [Assignments] — yet they are exactly the shapes the virtualizer's guards exist for. Such a case
    is built as an {!Ir.Low_level.t}, run through {!optimize} (the same
    [analyze_proc]/[specialize_proc] pipeline the backends use), asserted on structurally, and then
    EXECUTED through the [?prelowered] seam (gh-ocannl-562) — because virtualization rewrites what
    value a cell holds, which no structural pin can reach.

    Four doctrines are built into the helpers here, each learned from a leg that passed for the
    wrong reason:

    - The oracle must DISCRIMINATE. A producer value has to vary with every symbol of its iteration
      and stay off the init value, or a too-wide range guard replays an identical assignment, a
      wrong substitution stays invisible along the axis whose symbol the value omits, and a dropped
      first iteration hides in the zero-init. Hence {!tick} / {!tag} / {!ramp} rather than
      constants, and {!drift} where what has to discriminate is numeric (a narrow storage precision
      the running sums must leave behind) rather than which iteration wrote the cell.
    - Cells no writer covers must carry a {!sentinel}, so "wrote the wrong cells" fails the value
      check instead of reading whatever the buffer happened to hold.
    - A claim must be able to FAIL. Every {!p} is an assertion, not a recorded observation: a run
      with a false claim exits nonzero, so a regression cannot be [dune promote]d into the golden
      (gh-ocannl-601). Claims are therefore phrased so [true] is the passing reading.
    - A node to be read back must be declared {!materialize}d: [known_non_virtual] does NOT mean
      "has a context buffer" — a node written and read within one routine and never observed is
      placed [Local], routine-scoped scratch, and host access to it raises (gh-ocannl-599).

    This library links [ocannl], which is why it lives beside {!Test_utils} rather than inside it:
    [test_utils] deliberately depends on [arrayjit.ir] alone, for tests that link no more than that.
*)

open Base
module LL = Ir.Low_level
module Tn = Ir.Tnode
module Ops = Ir.Ops
module Idx = Ir.Indexing

let single = Ops.single

(** {1 Tensor nodes} *)

(** [node_factory ~first_id ~dims ()] returns a maker of fresh single-precision tensor nodes with
    consecutive ids above [first_id] and default dimensions [dims] (overridable per node). Each test
    executable picks an id range of its own, so nodes stay distinguishable in debug output. *)
let node_factory ?(prec = single) ~first_id ~dims () =
  let next_id = ref first_id in
  fun ?(dims = dims) label ->
    Int.incr next_id;
    Tn.create (Tn.Specified prec) ~id:!next_id ~label:[ label ]
      ~unpadded_dims:(lazy dims)
      ~padding:(lazy None)
      ()

(** Declares [tn] materialized and observable: the executed legs seed and read back exactly these
    nodes, and observability is also what forbids the buffer-aliasing planner from handing their
    bytes to another node. Both are declared intent, settled before optimization, so neither
    perturbs a structural pin. *)
let materialize tn =
  Tn.update_memory_mode tn Tn.On_device 99;
  Tn.set_observable tn

(** Declares [tn] virtual — the standing of the scope-local scalars a virtualizer-emitted
    [Local_scope] owns. *)
let virtualize tn = Tn.update_memory_mode tn Tn.Virtual 99

(** {1 Index and statement builders} *)

let sym () = Idx.get_symbol ()
let iter s : Idx.axis_index = Idx.Iterator s
let fixed n : Idx.axis_index = Idx.Fixed_idx n

(** [aff terms offset] is the affine index [sum (coeff * symbol) + offset]. *)
let aff terms offset : Idx.axis_index = Idx.Affine { symbols = terms; offset }

let set tn idcs llsc : LL.t = LL.Set { tn; idcs; llsc; debug = "" }
let get tn idcs : LL.scalar_t = LL.Get (tn, idcs)
let zero tn : LL.t = LL.Zero_out tn
let seq a b : LL.t = LL.Seq (a, b)

(** [loop ~upto s body] iterates [s] over [0 .. upto] INCLUSIVE, mirroring
    {!Ir.Low_level.For_loop}'s own bounds; [upto < 0] is a dead loop, which is a case worth
    building. *)
let loop ~upto s body : LL.t = LL.For_loop { index = s; from_ = 0; to_ = upto; body; axis = Serial }

(** [loop_n s n body] iterates [s] over a range of WIDTH [n], i.e. [0 .. n-1]. *)
let loop_n s n body : LL.t = loop ~upto:(n - 1) s body

(** {1 Scalar builders} *)

let c x : LL.scalar_t = LL.Constant x
let embed s : LL.scalar_t = LL.Embed_index (iter s)
let binop op a b : LL.scalar_t = LL.Binop (op, (a, single), (b, single))
let add a b = binop Ops.Add a b
let mul a b = binop Ops.Mul a b

(** {1 Discriminating producer values}

    A producer that writes a constant makes an executed leg blind to WHICH iteration supplied a cell
    — and that is precisely what the guards under test decide. These three write values that
    identify the iteration and stay clear of the zero-init. *)

(** [tick s] is [1 + s]: identifies a one-symbol producer's iteration, and the [1 +] keeps even the
    [s = 0] write distinct from the zero-init, so a dropped first iteration is visible. *)
let tick s = add (c 1.) (embed s)

(** [tag outer inner] is [1 + 10*outer + inner]: identifies both symbols of a two-symbol producer,
    so a substitution wrong on either axis shows up. *)
let tag outer inner = add (c 1.) (add (mul (c 10.) (embed outer)) (embed inner))

(** [ramp base s] is [base + s]: a per-cell value for a sibling provider, distinguished from other
    providers by [base]. [Embed_index] is not an array access, so this does not make the provider
    [is_complex]. *)
let ramp base s = add (c base) (embed s)

(** {1 Discriminating storage values}

    The counterpart of {!tick} / {!tag} / {!ramp} for tests that seed real tensor nodes rather than
    build producers: host-side [~f] initializers for [NTDSL.init], indexed by the cell's
    multi-index.

    Narrow-storage accumulator tests (gh-ocannl-639) need a sharper property than "varies with every
    symbol": the cells must be EXACT in the storage precision while their running sums are not, or
    the leg passes for the wrong reason. A zero-mean operand random-walks small enough that every
    bf16 partial sum stays bf16-exact, so per-step narrowing is invisible and a schedule-dependent
    accumulator width reads as parity — the zero-mean trap that docs/agent-notes.md's gh-ocannl-639
    entry records (trap 2), found the hard way by this test's first draft. Hence a cycle whose cells
    are exact and whose running sums DRIFT out of the storage format's exactness range.

    Both halves of that are arithmetic, not rules of thumb, and {!cycle} states each as a condition
    a caller has to check rather than assume. *)

(** [flat ~dims idcs] is the row-major flat index of [idcs] in a [dims]-shaped array. The leading
    dimension does not enter, so [dims.(0)] may be any extent the caller finds convenient. *)
let flat ~dims idcs =
  Array.foldi idcs ~init:0 ~f:(fun ax acc i -> if ax = 0 then i else (acc * dims.(ax)) + i)

(** [blind_axis ~dims ~modulus] is the outermost axis whose index {!cycle} would ignore, if any:
    stepping along axis [ax] moves {!flat} by that axis's row-major stride, so the cycle is constant
    along [ax] exactly when [modulus] divides that stride. *)
let blind_axis ~dims ~modulus =
  let found = ref None and stride = ref 1 in
  for ax = Array.length dims - 1 downto 0 do
    if !stride % modulus = 0 then found := Some (ax, !stride);
    stride := !stride * dims.(ax)
  done;
  !found

(** [cycle ~dims ~modulus ~offset ~stride idcs] is [(flat idcs mod modulus + offset) * stride]: the
    values [k * stride] for [k] cycling through [offset .. offset + modulus - 1] with period
    [modulus]. Reusing it means checking two conditions, neither of which the obvious phrasings
    imply:

    - {b It must vary with every index.} Coprimality with the reduction EXTENT is not the condition
      — [dims = [|2; 4; 3|]] with [modulus = 3] has row-major strides [12; 3; 1], so despite 3 and 4
      being coprime the value depends on the innermost index alone and a wrong substitution on
      either other axis stays invisible. The condition is on the STRIDES: no axis's stride may be a
      multiple of [modulus] ({!blind_axis}, which this function raises on — a [modulus] coprime to
      every [dims.(1 ..)] satisfies it, since the strides are their products).
    - {b The partial sums must actually leave exactness.} Count in units of [stride], which makes
      every cell and every partial sum an integer [k]: with [stride] a negative power of two, a
      format with [p] significand bits holds [k * stride] exactly for every [|k| <= 2^p], and above
      that only for the [k] whose trailing zeros make up the difference. So cells are exact when the
      whole range [offset .. offset + modulus - 1] fits within [2^p], and a reduction of [n] terms
      leaves exactness only once its running [k] clears [2^p] (and lands off the sparser multiples
      still representable above it) — a statement about the extent, not just about the mean being
      nonzero. A test relying on inexact partials has to EXHIBIT that crossing rather than infer it;
      [test/operations/discriminating_values] does so for {!drift}, and a nonzero mean over too few
      terms is the zero-mean trap wearing a different hat. *)
let cycle ~dims ~modulus ~offset ~stride idcs =
  (match blind_axis ~dims ~modulus with
  | Some (ax, s) ->
      raise
        (Invalid_argument
           (Printf.sprintf
              "Ll_test.cycle: modulus %d is blind to axis %d of %s (row-major stride %d is a \
               multiple of it), so the value would not vary with that index"
              modulus ax
              (Sexp.to_string (Array.sexp_of_t Int.sexp_of_t dims))
              s))
  | None -> ());
  (Float.of_int (flat ~dims idcs % modulus) +. offset) *. stride

(** [drift ~dims idcs] is {!cycle} at [13/20/(1/64)], the accumulator-width tests' operand: cells
    are the multiples of 1/64 between 0.3125 and 0.5, i.e. [k * (1/64)] for [k] in [20 .. 32], with
    mean [k = 26]. In {!cycle}'s units: cells are exact in bf16 ([p = 8], [32 < 256]) and stay exact
    in f32 ([p = 24]) along with every partial sum a test of this size can build, which is what lets
    an f64 host-side reference reproduce the widened kernel bitwise; while the running sum passes
    [2^8 = 256] units — the value 4, bf16's last guaranteed-exact multiple of 1/64 — at the eleventh
    term, landing on the bf16-unrepresentable [275/64], so a reduction longer than that visibly
    diverges if the accumulator narrows per step instead of once at the store (the legs using this
    reduce 16 and 36 terms). 13 is prime and divides none of those shapes' strides, so every index
    discriminates — and {!cycle} raises rather than let a later [~dims] silently break that.
    [test/operations/discriminating_values] pins each of those numbers. *)
let drift ~dims = cycle ~dims ~modulus:13 ~offset:20. ~stride:0.015625

(** {1 Optimization} *)

(** [optimize ~name llc] runs the backends' own pipeline ([analyze_proc] -> [specialize_proc]:
    structural facts -> placements -> [virtual_llc] -> cleanup -> simplify -> CSE -> hoist) over
    hand-built code.

    [~materialized] pre-decides those nodes' placement in the lineage state the optimization reads,
    which is what {!Context.decide_materialized} does for the [Assignments] pipeline — and the only
    way to do it for this path, since the [?prelowered] seam replaces the context's lineage with the
    optimized record's own [optimize_ctx], so a decision recorded on a context never reaches the
    optimization. Re-specializing the SAME [LL.t] with a virtualization candidate materialized gives
    a case its differential arm: the inlined and materialized readings of one program must agree
    cell for cell, which is what pins a guard. *)
let optimize ?(materialized = []) ~name llc : LL.optimized =
  let ctx : LL.optimize_ctx = LL.empty_optimize_ctx () in
  LL.decide_materialized ~provenance:589 ctx materialized;
  LL.optimize ctx ~unoptim_ll_source:None ~ll_source:None ~name [] llc

(** Post-optimization placement probes. Decisions live on the [optimize_ctx]'s placements
    (context-scoped memory modes), not on the tnode, which holds only declared intent. *)
let known_virtual (o : LL.optimized) tn =
  Tn.Placements.known_virtual o.LL.optimize_ctx.placements tn

let known_non_virtual (o : LL.optimized) tn =
  Tn.Placements.known_non_virtual o.LL.optimize_ctx.placements tn

(** {1 The executed leg} *)

(* One root context per executable: [Context.compile] forks the lineage for each compile, so sibling
   executions do not observe each other's decisions. *)
let base_ctx = lazy (Context.auto ())

(** [link ~name o] compiles the optimized record AS WRITTEN through the [?prelowered] seam,
    returning the advanced context and the compiled routine — so a test can assert on the
    [inputs]/[outputs] the link actually computed, read straight off the routine record, instead of
    re-deriving them via [input_and_output_nodes] (gh-ocannl-590).

    The identity [lowered_transform] takes the place of the default schedule annotator, which would
    otherwise parallelize or fission the hand-built loop nest — the point of the case is usually the
    nest's exact shape. *)
let link ?ctx ~name (o : LL.optimized) =
  let ctx = match ctx with Some ctx -> ctx | None -> Lazy.force base_ctx in
  Context.compile ~name ~prelowered:o
    ~lowered_transform:(fun x -> x)
    ctx Ir.Assignments.empty_comp Idx.Empty

(** [run_linked (ctx, routine) ~seed] drives the executed leg of an ALREADY-LINKED pair: uploads
    [seed], runs, and returns the context the values can be read from. It is the half of {!run} that
    survives keeping the routine — a test asserting on the [inputs]/[outputs] the link actually
    computed (gh-ocannl-590) holds the pair {!link} returned, so it needs the execution without
    compiling a second time. Every node in [seed] must have been {!materialize}d: host access to a
    node the pipeline placed [Local] raises (gh-ocannl-599). *)
let run_linked (ctx, routine) ~(seed : (Tn.t * float array) list) =
  let ctx = List.fold seed ~init:ctx ~f:(fun ctx (tn, vs) -> Context.set_values ctx tn vs) in
  Context.run ctx routine

(** [run ~name o ~seed] is {!link} followed by {!run_linked}, for the tests that have no use for the
    routine itself. Same materialization requirement on [seed]. *)
let run ?ctx ~name (o : LL.optimized) ~seed = run_linked (link ?ctx ~name o) ~seed

(** {!run}, then read back [read] in order. Same materialization requirement, on both lists. *)
let execute ?ctx ~name (o : LL.optimized) ~seed ~(read : Tn.t list) =
  let ctx = run ?ctx ~name o ~seed in
  List.map read ~f:(Context.get_values ctx)

(** Whether [f] was refused because the node it touched is placed [Local] — routine-scoped scratch
    whose values never reach a context buffer (gh-ocannl-599). *)
let refused_as_local f =
  try
    ignore (f ());
    false
  with Utils.User_error msg -> String.is_substring msg ~substring:"placed Local"

(** [optimize_and_execute] is {!optimize} followed by {!execute} under the same name, returning both
    the optimized record (for structural probes) and the values read back. *)
let optimize_and_execute ?ctx ?materialized ~name llc ~seed ~read =
  let o = optimize ?materialized ~name llc in
  (o, execute ?ctx ~name o ~seed ~read)

(** The value seeded into cells no writer is supposed to cover: distinct from every producer value
    the builders above generate, and from the zero-init. *)
let sentinel = -1.

(** [blank n] is [n] cells of {!sentinel}. *)
let blank n = Array.create ~len:n sentinel

(** Whether two value arrays match elementwise within [tol] (and have the same length). *)
let close ?(tol = 1e-5) values expected =
  Array.length values = Array.length expected
  && Array.for_alli values ~f:(fun i v -> Float.(abs (v -. expected.(i)) <= tol))

(** [same got expected] is {!close} over the list {!execute} returns. Raises if the lists differ in
    length — a mismatched [~read] is a test bug, not a failed assertion. *)
let same ?tol got expected = List.for_all2_exn got expected ~f:(close ?tol)

(** Asserts a named boolean claim, printing [name: b]. Booleans keep [.expected] files
    backend-stable; {!Verdict.p} is what keeps a [false] from being [dune promote]d into the golden
    as the expected output, so every claim has to be phrased so that [true] is the passing reading —
    a fact whose desired value is [false] gets renamed, not recorded (gh-ocannl-601). *)
let p = Verdict.p

(** {1 Structural probes}

    One exhaustive pair of traversals over [Low_level.t] / [scalar_t], from which every counter
    below is derived. Every new IR constructor is handled HERE, once — three hand-maintained copies
    of this pair is what gh-ocannl-600 retired. *)

let rec walk_t ~on_set ~on_get ~on_binop ~on_ternop ~on_scope (llc : LL.t) =
  let recur_t = walk_t ~on_set ~on_get ~on_binop ~on_ternop ~on_scope in
  let recur_s = walk_s ~on_set ~on_get ~on_binop ~on_ternop ~on_scope in
  match llc with
  | LL.Noop | LL.Declare_local _ | LL.Comment _ | LL.Staged_compilation _ | LL.Workgroup_barrier
  | LL.Tile_mma _ ->
      ()
  | LL.Seq (a, b) ->
      recur_t a;
      recur_t b
  | LL.For_loop { body; _ } -> recur_t body
  | LL.Zero_out tn -> on_set tn
  | LL.Set { tn; llsc; _ } ->
      on_set tn;
      recur_s llsc
  | LL.Set_dynamic { tn; dyn_value = v, _; llsc; _ } ->
      on_set tn;
      recur_s v;
      recur_s llsc
  | LL.Set_from_vec { tn; arg = s, _; _ } ->
      on_set tn;
      recur_s s
  | LL.Set_local (_, s) -> recur_s s
  | LL.If { cond = c, _; body } ->
      recur_s c;
      recur_t body

and walk_s ~on_set ~on_get ~on_binop ~on_ternop ~on_scope (s : LL.scalar_t) =
  let recur_t = walk_t ~on_set ~on_get ~on_binop ~on_ternop ~on_scope in
  let recur_s = walk_s ~on_set ~on_get ~on_binop ~on_ternop ~on_scope in
  match s with
  | LL.Constant _ | LL.Constant_bits _ | LL.Get_local _ | LL.Embed_index _ | LL.Get_merge_buffer _
    ->
      ()
  | LL.Get (tn, _) -> on_get tn
  | LL.Get_dynamic { tn; dyn_value = v, _; _ } ->
      on_get tn;
      recur_s v
  | LL.Local_scope { id; body; _ } ->
      on_scope id;
      recur_t body
  | LL.Ternop (op, (a, _), (b, _), (d, _)) ->
      on_ternop op;
      recur_s a;
      recur_s b;
      recur_s d
  | LL.Binop (op, (a, _), (b, _)) ->
      on_binop op;
      recur_s a;
      recur_s b
  | LL.Unop (_, (a, _)) -> recur_s a

let ignore1 _ = ()

(** [walk] over a statement, with only the callbacks the caller cares about. *)
let walk ?(on_set = ignore1) ?(on_get = ignore1) ?(on_binop = ignore1) ?(on_ternop = ignore1)
    ?(on_scope = ignore1) llc =
  walk_t ~on_set ~on_get ~on_binop ~on_ternop ~on_scope llc

(** [walk_scalar] over a scalar expression, likewise. *)
let walk_scalar ?(on_set = ignore1) ?(on_get = ignore1) ?(on_binop = ignore1) ?(on_ternop = ignore1)
    ?(on_scope = ignore1) s =
  walk_s ~on_set ~on_get ~on_binop ~on_ternop ~on_scope s

let count f =
  let n = ref 0 in
  f (fun () -> Int.incr n);
  !n

(** How many setters (including [Zero_out]) of [tn] survive in the optimized form. *)
let count_set (o : LL.optimized) tn =
  count (fun bump -> walk o.LL.llc ~on_set:(fun t -> if Tn.equal t tn then bump ()))

(** How many array READS of [tn] survive — [0] is what "the producer was inlined" means. *)
let count_get (o : LL.optimized) tn =
  count (fun bump -> walk o.LL.llc ~on_get:(fun t -> if Tn.equal t tn then bump ()))

(** How many [Where] ternops survive: the shape a virtualization equality/range guard renders as. *)
let count_where (o : LL.optimized) =
  count (fun bump -> walk o.LL.llc ~on_ternop:(function Ops.Where -> bump () | _ -> ()))

(** [(wheres, cmples, cmplts)] in the optimized form. A range guard emitted by unit-coefficient
    solving renders as [Where (And (Cmple _, Cmplt _), value, Get_local)] — one comparison shape per
    role, a non-strict lower bound and a strict upper bound — whereas a pure structural affine match
    introduces none of the three. *)
let count_guard_ops (o : LL.optimized) =
  let wh = ref 0 and le = ref 0 and lt = ref 0 in
  walk o.LL.llc
    ~on_ternop:(function Ops.Where -> Int.incr wh | _ -> ())
    ~on_binop:(function Ops.Cmple -> Int.incr le | Ops.Cmplt -> Int.incr lt | _ -> ());
  (!wh, !le, !lt)

(** How many [Local_scope]s a RAW statement carries, nested ones included — [0] after
    [hoist_cross_statement_cse] lifted a shared body out of its users. Takes a statement rather than
    an [optimized] record, because the passes it observes are run directly. *)
let count_scopes (llc : LL.t) = count (fun bump -> walk llc ~on_scope:(fun _ -> bump ()))

(** The [is_complex] fact recorded for [tn] by the structural facts pass. *)
let is_complex (o : LL.optimized) tn = (Hashtbl.find_exn o.LL.traced_store tn).LL.is_complex

(** Whether the facts pass classified [tn] as read before written within the routine — hence a
    routine input whose incoming buffer contents must be preserved. *)
let read_before_write (o : LL.optimized) tn =
  (Hashtbl.find_exn o.LL.traced_store tn).LL.read_before_write

(** Whether [tn] carries no assignment in this routine. *)
let read_only (o : LL.optimized) tn = (Hashtbl.find_exn o.LL.traced_store tn).LL.read_only
