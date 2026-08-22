(* The serial-rendered forms of ONE reduction, enumerated mechanically (gh-ocannl-664).

   The set of forms a single reduction loop can reach at codegen is defined implicitly, by the
   union of the schedule ops that can rewrite it and the codegen arms that can render the result.
   Three optimizer arcs enumerated its members one review round at a time — gh-ocannl-639 took six
   rounds over [Unroll] (both representations), sequential double materialization, [Pad]-guarded
   copies, [Partition] segments, Retype-[Vectorized] and Retype-[Workgroup_reduce]; gh-ocannl-663
   added schedule-minted vs codegen-minted scopes, [Tile_mma] fallbacks, the rng-exclusion
   granularity, virtualization's [Where]-guarded update spelling and init-vs-update rendering;
   gh-ocannl-693 added the localized-at-identity-precision form and the peeled-guard shapes. Every
   one of those was a point in the same product, found by a reviewer rather than by a test.

   This test defines the set EXTENSIONALLY: a table of (composition x storage precision), each
   member naming the form it claims to reach, executed and read back. Two claims per member, and
   the second is what makes the first trustworthy:

   - VALUE. A localizing member must agree BITWISE with the serial baseline's execution; a
     read-modify-write member must agree bitwise with the host's per-step-narrowed reference. Both
     references are exact: the operands are storage-exact multiples of 1/8 whose PARTIAL SUMS leave
     the storage format's exactness range (see {!cells}), so the two references differ at bf16 and
     f16 — proven here rather than assumed, by the "the whole-nest and per-step references differ"
     claims. At f32 they coincide, which is
     the identity-precision leg gh-ocannl-693 added: there the value claim pins substitution and
     iteration coverage rather than accumulator width.
   - FORM. The emitted kernel is classified — localized scope, per-step read-modify-write, SIMD
     accumulator grid, warp-shuffle tree, [Tile_mma] scalar fallback — and must be the form the
     member claims. Without this a composition that silently stopped reaching its form would keep
     passing its value claim by falling back to another one, which is exactly the false green the
     issue is about: agreement between two renderings is worthless if they are the same rendering.

   The member list itself is printed, so a form added to codegen without a table entry shows up as
   a golden diff rather than as silence. Members a backend cannot evaluate (SIMD grids off the C
   backends, warp shuffles off the GPUs) stay in the table and report {!Verdict.skipped}.

   The reduction is hand-built {!Ir.Low_level} (via [ll_test]) so that the nest shape is the
   test's, not shape inference's, and so that the loop symbols are in hand: every composition names
   its axes directly instead of re-deriving them from the lowered nest.

   Three points of the space are deliberately absent, and saying so is part of defining the set:

   - The [Grid] arm, whose fallback is the plain serial loop and NOT the localizing one — the one
     arm that can serialize without localizing. No schedule op in the tree produces an unbindable,
     non-parallel-eligible [Grid] level over a reduction axis, and one that existed would be a
     cross-thread race rather than a width question, so a member here could only be built by hand
     out of a shape the pipeline cannot reach.
   - The rng-mentioning update, one of the two configuration-independent declines. Its source is a
     uint4x32 key whose host representation is not a float array, so building it costs more than it
     would pin here; [test/operations/narrow_rng_nesting.ml]'s reduced-uniform leg is where that
     decline lives.
   - The HONOURED register-tile and tensor-core renderings. Those hold their accumulator in a
     C-tile rather than in a serial nest, so they are not serial forms of this reduction;
     [tile_mma_narrow] pins their width. What is a member is the [Tile_mma] SCALAR FALLBACK, whose
     reduction is a serial nest like any other. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module Cs = Ir.C_syntax
module Tn = Ir.Tnode
module LL = Ir.Low_level
module Idx = Ir.Indexing
module Ops = Ir.Ops
module Sched = Ir.Schedule
module Generated = Test_utils.Generated

let () = Utils.settings.output_debug_files_in_build_directory <- true
let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")
let () = Generated.init ~backend_name
let on_cpu = String.is_substring backend_name ~substring:"cc"
let p = Verdict.p
let skipped = Verdict.skipped ~backend:backend_name

(* {1 The reduction}

   [out[r] = sum_k x[r, k]] over [rows] x [cols]. Small on purpose: every member compiles its own
   kernel, and the table is a cross product. *)

let rows = 3

(* A multiple of the GPU backends' warp size, so that the [Workgroup_reduce] member can reach the
   shuffle-tree rendering where the backend has one: that rendering REFUSES (loudly, by design — a
   plain hardware binding would race the accumulator update) an extent it cannot cover a warp at a
   time. *)
let cols = 32

(* The operand cells: [(flat mod 13 + 200) / 8], i.e. the multiples of 1/8 between 25 and 26.5.

   Both halves of the discrimination are arithmetic (see {!Ll_test.cycle}). Exactness: in units of
   1/8 every cell is an integer in [200, 212], and a format with [p] significand bits holds every
   integer up to [2^p] — 212 < 256 = 2^8, so the cells are exact in bf16, in f16 and in f32. Drift:
   the running sum passes 2^8 units after the second term and 2^11 units after the tenth, so over
   16 terms a per-step narrowing to bf16 or to f16 visibly diverges from a whole-nest accumulation,
   while the total (3296 units, 412.0) stays exact in f32 — which is what lets the host reference
   reproduce the kernel bitwise. 13 is coprime to the row-major strides 16 and 1, so the value
   varies with BOTH loop symbols; {!Ll_test.cycle} raises rather than let that lapse. *)
let cells = Ll_test.cycle ~dims:[| rows; cols |] ~modulus:13 ~offset:200. ~stride:0.125
let cell r k = cells [| r; k |]
let x_values = Array.init (rows * cols) ~f:(fun n -> cell (n / cols) (n % cols))

(* Host references, per storage precision. [round] is the library's own conversion, so these are
   the kernel's roundings and not an approximation of them. *)
let round (prec : Ops.prec) v =
  match prec with
  | Ops.Bfloat16_prec _ -> Ops.bfloat16_to_single (Ops.single_to_bfloat16 v)
  | Ops.Half_prec _ -> Ops.half_to_single (Ops.single_to_half v)
  | Ops.Fp8_prec _ -> Ops.fp8_to_single (Ops.single_to_fp8 v)
  | _ -> v

(* The whole-nest reference: accumulate wide, narrow once at the store. *)
let whole_nest_ref prec =
  Array.init rows ~f:(fun r ->
      let acc = ref 0.0 in
      for k = 0 to cols - 1 do
        acc := !acc +. cell r k
      done;
      round prec !acc)

(* The per-step reference: narrow to storage at every accumulation step, which is what a
   read-modify-write rendering does. The addition itself happens at compute precision, and the
   operands are storage-exact, so one rounding per step is the whole of it. *)
let per_step_ref prec =
  Array.init rows ~f:(fun r ->
      let acc = ref 0.0 in
      for k = 0 to cols - 1 do
        acc := round prec (!acc +. cell r k)
      done;
      !acc)

let precs = [ ("f32", Ops.single); ("bf16", Ops.bfloat16); ("f16", Ops.half) ]

(* {1 Program shapes}

   Five spellings of the same mathematical reduction. All the guards are VACUOUS — the mask is
   all-zero, the runtime extent is bound to [cols] at launch — so every shape computes the full sum
   and one baseline serves them all; what the guards change is which peel decision codegen and the
   schedule mints reach, which is the point. *)

type shape =
  | Plain  (** [for r: for k: out[r] += x[r,k]] — the recognizer's canonical nest. *)
  | Runtime_guard
      (** The gh-490 symbolic-extent shape: [If (k < s)] with [s] a STATIC symbol, a kernel
          parameter bound at launch. Its bound is outside every loop, so the peel must take it
          (gh-ocannl-693); the mints' agreement with the serial baseline under it is gh-ocannl-715. *)
  | Data_guard
      (** [If (mask[k] < 1)] — a guard that is not [pure_index_guard], so the peel refuses and the
          nest renders as per-step read-modify-writes. Seeded all-zero, hence always true. *)
  | Side_write
      (** The reduction level holds a SECOND statement (a write to another node), one of
          [peel_accum_nest]'s structural refusals. The sibling does not feed [out], so the value is
          the same and the form is not. *)
  | Virtual_acc
      (** The accumulator is a VIRTUAL node the virtualizer inlines at its read site: the
          [Local_scope] with [mint = Inlined_computation], the other producer of the scope form. *)
  | Where_scope
      (** Virtualization's guarded-update spelling: an already-scoped nest whose update is
          [Set_local (id, Where (cond, acc + x, Get_local id))] — an expression-spelled guard whose
          else-arm carries the accumulator through. Post-optimize IR, so it reaches the backend
          through {!Ll_test.optimize_scoped} rather than through [LL.optimize]. *)

let shape_name = function
  | Plain -> "plain"
  | Runtime_guard -> "runtime-extent guard"
  | Data_guard -> "data-dependent guard"
  | Side_write -> "sibling statement"
  | Virtual_acc -> "virtual accumulator"
  | Where_scope -> "Where-guarded scope"

type prog = {
  llc : LL.t;
  raw : LL.t option;  (** The unscoped twin {!Ll_test.optimize_scoped} needs, for [Where_scope]. *)
  r : Idx.symbol;
  k : Idx.symbol;
  out : Tn.t;
  materialized : Tn.t list;
  seed : (Tn.t * float array) list;
  bindings : Idx.unit_bindings;
  bind : Idx.lowered_bindings -> unit;
}

(* Ids are the test's own range, bumped per program so that nodes stay distinguishable in debug
   output and no two compiles share a tensor node. *)
let next_id = ref 964_000_000

(* A launch parameter standing for a symbolic DIMENSION rather than an index: its bound value is a
   size in [0, cols] INCLUSIVE, the standing [Row.get_sym_dim] gives a gh-490 extent symbol. Without
   [used_as_extent] the bind-time validation rejects the one binding these members want — the extent
   that covers the whole axis, which is what makes the guard vacuous and the value comparable to the
   unguarded baseline. *)
let extent_symbol () =
  let s, bindings = Idx.get_static_symbol ~static_range:cols Idx.Empty in
  s.Idx.used_as_extent <- true;
  (s, bindings)

let make ~(prec : Ops.prec) ~(shape : shape) () : prog =
  let node ?(dims = [| rows; cols |]) label =
    next_id := !next_id + 1;
    Tn.create (Tn.Specified prec) ~id:!next_id ~label:[ label ]
      ~unpadded_dims:(lazy dims)
      ~padding:(lazy None)
      ()
  in
  let out = node ~dims:[| rows |] "rfout" in
  let x = node "rfxs" in
  Ll_test.materialize out;
  Ll_test.materialize x;
  let r = Ll_test.sym () and k = Ll_test.sym () in
  let iprec = Ops.index_prec () in
  let ri = Idx.Iterator r and ki = Idx.Iterator k in
  let acc_cell = [| ri |] in
  let get_acc = LL.Get (out, acc_cell) in
  let get_x = LL.Get (x, [| ri; ki |]) in
  let update = LL.Binop (Ops.Add, (get_acc, prec), (get_x, prec)) in
  let plain_body = LL.Set { tn = out; idcs = acc_cell; llsc = update; debug = "" } in
  let seed = [ (out, Array.create ~len:rows 0.0); (x, x_values) ] in
  let base =
    {
      llc = LL.Noop;
      raw = None;
      r;
      k;
      out;
      materialized = [ out; x ];
      seed;
      bindings = Idx.Empty;
      bind = (fun _ -> ());
    }
  in
  let nest ?(body = plain_body) () =
    LL.For_loop
      {
        index = r;
        from_ = 0;
        to_ = rows - 1;
        axis = LL.Serial;
        body = LL.For_loop { index = k; from_ = 0; to_ = cols - 1; axis = LL.Serial; body };
      }
  in
  match shape with
  | Plain -> { base with llc = nest () }
  | Runtime_guard ->
      let s, bindings = extent_symbol () in
      let cond =
        LL.Binop
          ( Ops.Cmplt,
            (LL.Embed_index ki, iprec),
            (LL.Embed_index (Idx.Iterator s.Idx.static_symbol), iprec) )
      in
      {
        base with
        llc = nest ~body:(LL.If { cond = (cond, iprec); body = plain_body }) ();
        bindings;
        bind = (fun lowered -> Idx.find_exn lowered s := cols);
      }
  | Data_guard ->
      let mask = node ~dims:[| cols |] "rfmask" in
      Ll_test.materialize mask;
      let cond =
        LL.Binop (Ops.Cmplt, (LL.Get (mask, [| ki |]), prec), (LL.Constant 1.0, prec))
      in
      {
        base with
        llc = nest ~body:(LL.If { cond = (cond, prec); body = plain_body }) ();
        materialized = [ out; x; mask ];
        seed = (mask, Array.create ~len:cols 0.0) :: seed;
      }
  | Side_write ->
      (* Indexed by BOTH symbols, so the sibling is not loop-invariant: a hoistable one would leave
         the level holding a single statement again and the member would silently become a
         localizing one. *)
      let side = node "rfside" in
      Ll_test.materialize side;
      let sibling =
        LL.Set
          {
            tn = side;
            idcs = [| ri; ki |];
            llsc = LL.Binop (Ops.Add, (get_x, prec), (LL.Constant 1.0, prec));
            debug = "";
          }
      in
      {
        base with
        llc = nest ~body:(LL.Seq (plain_body, sibling)) ();
        materialized = [ out; x; side ];
        seed = (side, Array.create ~len:(rows * cols) Ll_test.sentinel) :: seed;
      }
  | Virtual_acc ->
      let tmp = node ~dims:[| rows |] "rftmp" in
      Ll_test.virtualize tmp;
      let tmp_update =
        LL.Set
          {
            tn = tmp;
            idcs = acc_cell;
            llsc = LL.Binop (Ops.Add, (LL.Get (tmp, acc_cell), prec), (get_x, prec));
            debug = "";
          }
      in
      let accumulate =
        LL.For_loop
          {
            index = r;
            from_ = 0;
            to_ = rows - 1;
            axis = LL.Serial;
            body =
              LL.For_loop
                { index = k; from_ = 0; to_ = cols - 1; axis = LL.Serial; body = tmp_update };
          }
      in
      let copy_sym = Ll_test.sym () in
      let copy =
        LL.For_loop
          {
            index = copy_sym;
            from_ = 0;
            to_ = rows - 1;
            axis = LL.Serial;
            body =
              LL.Set
                {
                  tn = out;
                  idcs = [| Idx.Iterator copy_sym |];
                  llsc = LL.Get (tmp, [| Idx.Iterator copy_sym |]);
                  debug = "";
                };
          }
      in
      {
        base with
        llc = LL.Seq (LL.Zero_out tmp, LL.Seq (accumulate, copy));
        seed = [ (x, x_values) ];
      }
  | Where_scope ->
      let s, bindings = extent_symbol () in
      let cond =
        LL.Binop
          ( Ops.Cmplt,
            (LL.Embed_index ki, iprec),
            (LL.Embed_index (Idx.Iterator s.Idx.static_symbol), iprec) )
      in
      let id = LL.get_scope out in
      let body =
        LL.Seq
          ( LL.Set_local (id, get_acc),
            LL.For_loop
              {
                index = k;
                from_ = 0;
                to_ = cols - 1;
                axis = LL.Serial;
                body =
                  LL.Set_local
                    ( id,
                      LL.Ternop
                        ( Ops.Where,
                          (cond, iprec),
                          (LL.Binop (Ops.Add, (LL.Get_local id, prec), (get_x, prec)), prec),
                          (LL.Get_local id, prec) ) );
              } )
      in
      let scoped =
        LL.For_loop
          {
            index = r;
            from_ = 0;
            to_ = rows - 1;
            axis = LL.Serial;
            body =
              LL.Set
                {
                  tn = out;
                  idcs = acc_cell;
                  llsc =
                    (* [Schedule_minted], not [Inlined_computation], even though the UPDATE is
                       spelled the way [inline_computation] spells a guarded one: the scope is over
                       a MATERIALIZED node, which the virtualizer never produces (gh-ocannl-681) —
                       this is post-optimize IR handed to the backend the way the schedule mints
                       hand theirs. The mint records which side built the scope; the borrowed
                       spelling is what the member is about. *)
                    LL.Local_scope
                      { id; body; orig_indices = acc_cell; mint = LL.Schedule_minted };
                  debug = "";
                };
          }
      in
      let raw =
        nest
          ~body:
            (LL.Set
               {
                 tn = out;
                 idcs = acc_cell;
                 llsc =
                   LL.Ternop
                     (Ops.Where, (cond, iprec), (update, prec), (get_acc, prec));
                 debug = "";
               })
          ()
      in
      {
        base with
        llc = scoped;
        raw = Some raw;
        bindings;
        bind = (fun lowered -> Idx.find_exn lowered s := cols);
      }

(* {1 Reading the rendered form off the emitted kernel}

   Classification is textual, over the artifact {!Test_utils.Generated} guarantees this run emitted.
   It is deliberately phrased in terms of the STORED node rather than of any particular local's
   name: the localizing forms differ in where the accumulator lives (a codegen-minted scope, a
   schedule-minted one, a SIMD register grid, a warp's registers) but agree on what they do to the
   node — read it at most once, write it at most once, never both in one statement. The
   read-modify-write forms are exactly the ones with a statement doing both.

   Statements are split on [;], not on newlines: the pretty-printer breaks a long value expression
   across lines, so a line-based reading of "does this statement touch the node twice" answers no
   for precisely the wide read-modify-write it is meant to catch. *)

type form = Localized | Partials_combine | Rmw | Simd | Warp | Mma_fallback

let form_name = function
  | Localized -> "localized scope"
  | Partials_combine -> "scoped block partials + combine"
  | Rmw -> "per-step read-modify-write"
  | Simd -> "SIMD accumulator grid"
  | Warp -> "warp-shuffle tree"
  | Mma_fallback -> "Tile_mma scalar fallback"

let is_ident_char c = Char.is_alphanum c || Char.equal c '_'

(* The code name codegen derived for a node is not predictable from its label (it goes through a
   blacklist and a dot-stripping pass), so it is read off the source: an identifier CONTAINING the
   label and followed by a subscript. Several can match — [Split_reduce] mints a [partials_<parent>]
   node, whose name contains the parent's — so the SHORTEST is taken, which is the label itself
   wherever codegen kept it. *)
let ident_for src ~label =
  let n = String.length src in
  String.substr_index_all src ~may_overlap:false ~pattern:label
  |> List.filter_map ~f:(fun at ->
         let b = ref at and e = ref (at + String.length label) in
         while !b > 0 && is_ident_char src.[!b - 1] do
           Int.decr b
         done;
         while !e < n && is_ident_char src.[!e] do
           Int.incr e
         done;
         if !e < n && Char.equal src.[!e] '[' then Some (String.sub src ~pos:!b ~len:(!e - !b))
         else None)
  |> List.min_elt ~compare:(fun a b -> Int.compare (String.length a) (String.length b))

let statements src =
  List.map (String.split src ~on:';') ~f:(fun st ->
      String.concat ~sep:" " (String.split_on_chars st ~on:[ '\n'; '\t' ]))

(* Whether [tok] is a scope-local name: [v<digits>_<node ident>], {!C_syntax.pp_scope_id}'s
   spelling. Which node it belongs to is not constrained — the accumulator of a [Virtual_acc]
   member is a different node from the one stored. *)
let is_scope_local tok =
  String.length tok > 2
  && Char.equal tok.[0] 'v'
  &&
  match String.index tok '_' with
  | None | Some 1 -> false
  | Some us -> String.for_all (String.sub tok ~pos:1 ~len:(us - 1)) ~f:Char.is_digit

type reading = {
  rmw_statements : int;  (** Statements reading AND writing the stored node: the RMW fingerprint. *)
  stores_from_local : int;
      (** [node[...] = <conversion>(v<n>_...)]: the once-per-cell narrowing store that ends a
          localized nest. The conversion is why the right-hand side is searched for a scope-local
          TOKEN rather than compared to one: at bf16 the store reads
          [rfout[i] = single_to_bfloat16(v5_rfout)], and a test looking for a bare local would
          report every narrow-storage kernel as unlocalized — which is the reading the whole
          accumulator-width policy is about. *)
  foreign_local_stores : int;
      (** The same shape into some OTHER node: [Split_reduce]'s block partials, each of which
          localizes while the combine that folds them into the target does not. *)
  node_accesses : int;
  has_simd : bool;
  has_warp : bool;
}

(* Whether [st] mentions a scope-local name anywhere. *)
let mentions_scope_local st =
  let n = String.length st in
  let rec go i =
    if i >= n then false
    else if is_ident_char st.[i] then begin
      let j = ref i in
      while !j < n && is_ident_char st.[!j] do
        Int.incr j
      done;
      if is_scope_local (String.sub st ~pos:i ~len:(!j - i)) then true else go !j
    end
    else go (i + 1)
  in
  go 0

(* Occurrences of [ident] as a WHOLE identifier followed by a subscript. Substring counting is
   wrong here: [partials_rfout[..]] contains [rfout[], so a reduction whose partials node is named
   after its target would read as touching the target twice. *)
let subscripts st ~ident =
  List.count
    (String.substr_index_all st ~may_overlap:false ~pattern:(ident ^ "["))
    ~f:(fun at -> at = 0 || not (is_ident_char st.[at - 1]))

let read_form src ~label =
  match ident_for src ~label with
  | None -> None
  | Some ident ->
      let sts = statements src in
      let accesses st = subscripts st ~ident in
      (* A store of a scope local: the assignment's left-hand side is an array subscript, and its
         right-hand side mentions a local. *)
      let stores_a_local st =
        match String.substr_index st ~pattern:"] = " with
        | None -> false
        | Some at -> mentions_scope_local (String.drop_prefix st (at + 4))
      in
      let assigns_node st =
        match (String.substr_index st ~pattern:"] = ", String.substr_index st ~pattern:(ident ^ "[")) with
        | Some at, Some lhs -> lhs < at && (lhs = 0 || not (is_ident_char st.[lhs - 1]))
        | _ -> false
      in
      Some
        {
          rmw_statements = List.count sts ~f:(fun st -> accesses st >= 2);
          stores_from_local =
            List.count sts ~f:(fun st -> accesses st = 1 && assigns_node st && stores_a_local st);
          foreign_local_stores =
            List.count sts ~f:(fun st -> accesses st = 0 && stores_a_local st);
          node_accesses = List.sum (module Int) sts ~f:accesses;
          has_simd = String.is_substring src ~substring:"Vectorized reduction rendering";
          has_warp = String.is_substring src ~substring:"ocannl_shfl_xor";
        }

(* What the reading says the kernel rendered. The order matters, and not merely for tidiness: a
   SIMD grid's epilogue is TEXTUALLY a read-modify-write ([out[i] = out[i] + vred_total]) — one per
   nest rather than one per step, which is the entire difference — so testing the RMW fingerprint
   first would classify the vector rendering as the form it exists to avoid. The more specific form
   wins, and the per-nest count is checked separately against the member's declared sites. *)
let form_of reading =
  if reading.has_warp then Warp
  else if reading.has_simd then Simd
  else if reading.rmw_statements = 0 && reading.stores_from_local > 0 then Localized
  else if reading.rmw_statements > 0 && reading.stores_from_local = 0
          && reading.foreign_local_stores > 0
  then Partials_combine
  else Rmw


(* {1 Executing a member} *)

(* One root context for the whole run: [Context.compile] forks the lineage per compile, so members
   do not observe each other's placement decisions. *)
let base_ctx = lazy (Context.auto ())

let execute ~name ~(prog : prog) ~(sched : Sched.schedule) =
  (* The launch parameters have to reach BOTH halves: [LL.optimize]'s walk asserts that every
     symbol it meets is in scope (a runtime-extent guard mentions one that no loop binds), and
     [Sched.apply] folds guards against their declared ranges. *)
  let static_indices = Idx.bound_symbols prog.bindings in
  let o =
    match prog.raw with
    | None -> Ll_test.optimize ~materialized:prog.materialized ~static_indices ~name prog.llc
    | Some raw ->
        Ll_test.optimize_scoped ~materialized:prog.materialized ~static_indices ~name ~raw prog.llc
  in
  let ctx, routine =
    Context.compile ~name ~prelowered:o
      ~lowered_transform:(fun opt -> Sched.apply ~static_indices sched opt)
      (Lazy.force base_ctx) Ir.Assignments.empty_comp prog.bindings
  in
  prog.bind routine.Context.bindings;
  let ctx =
    List.fold prog.seed ~init:ctx ~f:(fun ctx (tn, vs) -> Context.set_values ctx tn vs)
  in
  let ctx = Context.run ctx routine in
  Context.get_values ctx prog.out

(* {1 The member table}

   Each member is one point of the product the issue asks to sweep: a schedule composition over one
   of the {!shape}s, the form it claims to reach, and the reference its value must match. *)

(* Every value claim is BITWISE, the reassociating compositions included. A swapped split, a
   block-partial reduction and a shuffle tree all change the order of the additions, so they would
   normally earn a tolerance — but each of them is declared over f32 only (or, for the shuffle,
   available at f32 only), and there these operands' partial sums are exact, so every association
   gives the same float. That is not an assumption: the "at f32 the whole-nest and per-step
   references coincide" claim asserts exactly the exactness the argument rests on. A tolerance would have hidden a real narrowing at a seam behind
   an allowance for a reassociation that cannot cost anything here. *)

type reference =
  | Baseline  (** The serial rendering of the plain nest, executed at this precision. *)
  | Per_step  (** The host's per-step-narrowed reference: what a read-modify-write form computes. *)

type member = {
  slug : string;  (** Short name; also the routine name's stem, so artifacts are per member. *)
  what : string;  (** What the composition is, in words. *)
  shape : shape;
  sched : prog -> Sched.schedule;
  expect : form;  (** The form this backend must reach. *)
  claimed : string;
      (** How the form is NAMED in the printed table. Backend-uniform, unlike [expect]: the
          [.expected] golden is one file for every backend, so a member whose form is
          backend-dependent names both readings here. *)
  reference : reference;
  store_sites : int;
      (** How many CLOSING STORES of the accumulated node the localized form may emit: one per
          textual copy of the surviving output loop's body, which is one unless the composition
          duplicated that loop ([Unroll ~materialize] or [Partition] of the OUTPUT axis). Pinned
          exactly, not as "at least one" (Codex P2, round 1): a regression that gave each
          [Partition] segment or each unrolled copy its OWN scope and closing store would still
          read as localized under a positive-count test, and on a backend whose accumulator already
          resides at storage width the extra store/reload seams need not change the executed value
          either — a false green in both claims at once. The node's total subscript count is bounded
          by [2 * store_sites] for the same reason: each site may open the cell and close it, and
          nothing else may touch it. *)
  extra : string list;
      (** Substrings the emitted kernel must also contain, ANDed into the form claim. For the one
          distinguishing feature the node-access classifier cannot see: WHICH spelling the update
          took inside the scope. *)
  precisions : string list;
  available : string -> bool;
      (** Whether this backend can evaluate the member at the given storage precision. Per
          PRECISION, not merely per backend: the warp-shuffle rendering exists only for single- and
          double-precision accumulators, so the same member is evaluable at f32 and vacuous at bf16
          on the very same GPU. Legs it excludes stay in the printed table and report
          {!Verdict.skipped}, so the golden is backend-uniform and [grep SKIPPED] enumerates what
          this hardware did not check. *)
}

let no_ops _ = []

let member ?(shape = Plain) ?(sched = no_ops) ?(expect = Localized) ?claimed ?(reference = Baseline)
    ?(store_sites = 1) ?(extra = []) ?(precisions = [ "f32"; "bf16"; "f16" ])
    ?(available = fun _ -> true) slug what =
  let claimed = Option.value claimed ~default:(form_name expect) in
  {
    slug;
    what;
    shape;
    sched;
    expect;
    claimed;
    reference;
    store_sites;
    extra;
    precisions;
    available;
  }

let members =
  [
    (* --- the baseline itself: gh-ocannl-693's identity-precision localization, which before it
       left every f32 reduction doing one global read-modify-write per step. --- *)
    member "serial" "no schedule ops (the reference rendering)";
    (* --- the two [Unroll] representations autotune proposes over small reduction loops --- *)
    member "unroll-annot" "Unroll (annotated: codegen repeats the body)" ~sched:(fun g ->
        [ Sched.Unroll { axis = g.k; materialize = false } ]);
    member "unroll-mat" "Unroll ~materialize (the schedule mints the scope)" ~sched:(fun g ->
        [ Sched.Unroll { axis = g.k; materialize = true } ]);
    (* The output loop is gone, so each of its [rows] copies closes its own cell -- which is not a
       seam but the whole of that cell's reduction. *)
    member "unroll-outer-mat" "Unroll ~materialize of the OUTPUT axis" ~store_sites:rows
      ~sched:(fun g -> [ Sched.Unroll { axis = g.r; materialize = true } ]);
    (* --- [Partition]: an index-set specialization of one reduction, so its segment seams must not
       become narrowing points — one scope spans every segment. --- *)
    member "partition" "Partition of the reduction axis into three segments" ~sched:(fun g ->
        let pt, _ = Sched.partition ~axis:g.k ~breakpoints:[ 4; 12 ] in
        [ pt ]);
    member "partition-then-unroll" "Partition, then Unroll one segment (the seam stays addressable)"
      ~sched:(fun g ->
        let pt, segs = Sched.partition ~axis:g.k ~breakpoints:[ 4; 12 ] in
        [ pt; Sched.Unroll { axis = List.hd_exn segs; materialize = false } ]);
    (* Two segment loops over the output axis, each carrying a whole reduction: two sites. *)
    member "partition-outer" "Partition of the OUTPUT axis (the reduction is inside a segment)"
      ~store_sites:2
      ~sched:(fun g ->
        let pt, _ = Sched.partition ~axis:g.r ~breakpoints:[ 1 ] in
        [ pt ]);
    (* --- [Split], with and without a reordering --- *)
    member "split-then-unroll" "Split the reduction axis, then Unroll ~materialize the inner half"
      ~sched:(fun g ->
        let sp, _outer, inner = Sched.split ~axis:g.k ~factor:4 ~outer:LL.Serial ~inner:LL.Serial in
        [ sp; Sched.Unroll { axis = inner; materialize = true } ]);
    member "split-then-swap" "Split the reduction axis, then Swap the halves"
      ~precisions:[ "f32" ] ~sched:(fun g ->
        let sp, outer, inner = Sched.split ~axis:g.k ~factor:4 ~outer:LL.Serial ~inner:LL.Serial in
        [ sp; Sched.Swap { outer; inner } ]);
    (* --- [Pad]: guarded copies, whose constant-bounded guard the mint must peel into the scope --- *)
    member "pad-then-unroll-mat" "Pad the reduction axis, then Unroll ~materialize (guarded copies)"
      ~sched:(fun g ->
        [
          Sched.Pad { axis = g.k; to_multiple_of = 6 };
          Sched.Unroll { axis = g.k; materialize = true };
        ]);
    (* --- [Split_reduce]: block partials plus a combine nest. The reduction scopes do NOT coincide
       (each partial narrows at its own store), so this member claims f32 only, where the sums are
       exact. --- *)
    member "split-reduce" "Split_reduce into four block partials plus a combine nest"
      ~expect:Partials_combine ~precisions:[ "f32" ] ~sched:(fun g ->
        let op, _b, _i, _c = Sched.split_reduce ~axis:g.k ~target:g.out ~num_blocks:4 in
        [ op ]);
    (* --- the [Vectorized] arm: a SIMD accumulator grid plus its scalar tail, and the same
       rendering NESTED inside a surviving serial reduction level. --- *)
    member "retype-vectorized" "Retype the reduction axis to Vectorized" ~expect:Simd
      ~available:(fun _ -> on_cpu)
      ~sched:(fun g -> [ Sched.Retype { axis = g.k; ty = LL.Vectorized } ]);
    member "split-then-vectorize-inner" "Split, then Retype the INNER half to Vectorized"
      ~expect:Simd
      ~available:(fun _ -> on_cpu)
      ~sched:(fun g ->
        let sp, _outer, inner = Sched.split ~axis:g.k ~factor:4 ~outer:LL.Serial ~inner:LL.Serial in
        [ sp; Sched.Retype { axis = inner; ty = LL.Vectorized } ]);
    (* The Vectorized arm's OTHER exit: a loop-carried accumulation that no vector rendering
       accepted falls back to the localizing peel, never to the pragma'd loop — the pragmas assert
       iteration independence, which an accumulation does not satisfy. A two-wide inner half is
       below the profitability gate, so the SIMD grid declines and the peel takes it. *)
    member "split-then-vectorize-narrow" "Split into a two-wide inner half, then Retype it Vectorized"
      ~available:(fun _ -> on_cpu)
      ~sched:(fun g ->
        let sp, _outer, inner = Sched.split ~axis:g.k ~factor:2 ~outer:LL.Serial ~inner:LL.Serial in
        [ sp; Sched.Retype { axis = inner; ty = LL.Vectorized } ]);
    (* --- a hardware reduction kind: the warp-shuffle tree where the backend binds one, and the
       serialized fallback (which is a serial level to the peel) where it does not. --- *)
    member "retype-workgroup-reduce" "Retype the reduction axis to Workgroup_reduce"
      ~expect:(if on_cpu then Localized else Warp)
      ~claimed:"warp-shuffle tree, or the localized scope where no lane index is bound"
        (* The shuffle tree takes single- or double-precision accumulators only, and refuses
           anything else by raising rather than by falling back — so off the C backends this member
           is evaluable at f32 alone. *)
      ~available:(fun prec_name -> on_cpu || String.equal prec_name "f32")
      ~sched:(fun g -> [ Sched.Retype { axis = g.k; ty = LL.Workgroup_reduce } ]);
    (* The plain [Workgroup] arm: a hardware binding where the backend has an index for the slot,
       and otherwise the localizing peel. Only the second half is a member here — binding a
       REDUCTION axis to a workgroup dimension is a cross-lane race wherever the binding exists, so
       the leg runs on the backends that serialize it and is skipped where it would not be. *)
    member "retype-workgroup" "Retype the reduction axis to Workgroup (no index bound: serialized)"
      ~available:(fun _ -> on_cpu)
      ~sched:(fun g -> [ Sched.Retype { axis = g.k; ty = LL.Workgroup } ]);
    (* --- the gh-490 runtime-extent guard, whose bound is a kernel parameter rather than a
       constant. The peel must see through it (gh-ocannl-693) and the mints must agree with the
       serial baseline under it (gh-ocannl-715): a refused mint round-trips the accumulator per
       copy, which on narrow storage is a different number. --- *)
    member "runtime-guard" "a runtime-extent guard, unscheduled" ~shape:Runtime_guard;
    member "runtime-guard-unroll-mat" "a runtime-extent guard under Unroll ~materialize"
      ~shape:Runtime_guard ~sched:(fun g -> [ Sched.Unroll { axis = g.k; materialize = true } ]);
    member "runtime-guard-partition" "a runtime-extent guard under Partition of the reduction axis"
      ~shape:Runtime_guard ~sched:(fun g ->
        let pt, _ = Sched.partition ~axis:g.k ~breakpoints:[ 4; 12 ] in
        [ pt ]);
    (* --- the OTHER producer of the scope form: virtualization's inline at a read site --- *)
    member "virtual-accumulator" "a virtual accumulator inlined at its read site"
      ~shape:Virtual_acc;
    member "where-guarded-update" "virtualization's Where-guarded update spelling"
      ~shape:Where_scope ~extra:[ " ? " ];
    (* --- the two read-modify-write forms, i.e. the declines. Without these the localized claims
       could not fail: a classifier that answered "localized" for everything would pass every other
       member, and the whole table would be measuring nothing. --- *)
    member "decline-data-guard" "a data-dependent guard (not a pure index guard)" ~shape:Data_guard
      ~expect:Rmw ~reference:Per_step;
    member "decline-sibling-statement" "a second statement in the reduction level"
      ~shape:Side_write ~expect:Rmw ~reference:Per_step;
    member "decline-sibling-unrolled" "the same level Unrolled: one read-modify-write per copy"
      ~shape:Side_write ~expect:Rmw ~reference:Per_step ~sched:(fun g ->
        [ Sched.Unroll { axis = g.k; materialize = false } ]);
  ]

(* {1 Coverage ratchets}

   Printing the member list says what the table SWEEPS; on its own it cannot say what the table
   MISSES. A schedule op or an axis kind that no member reaches changes neither the list nor the
   output, which leaves exhaustiveness as an unenforced advertisement and puts the next form back on
   the review-only discovery path this test exists to replace (Codex P2, round 1).

   So the two variant types that DEFINE the space are matched exhaustively below, and the compiler
   is the ratchet: adding a constructor to [Schedule.optop] or to [Low_level.axis_type] fails to
   build this file until someone says which member reaches it, or why it is out of scope. The
   classification is printed, and every member name it cites is checked to exist — so the coverage
   claim cannot drift from the table by renaming either side.

   The limit is worth stating rather than glossing: this catches a new schedule OP and a new axis
   KIND. It does not catch a new rendering ARM inside an axis kind the table already reaches — if
   such an arm fires for a member the form claim moves and the golden diffs, but an arm reachable
   only through a shape no member builds is invisible here. Closing that needs a census emitted by
   codegen itself, in the shape of [C_syntax.mma_census]. *)

type coverage =
  | Covered of string list  (** The members that reach it, by slug. *)
  | Out_of_scope of string  (** Why it is not a serial-rendered form of a reduction. *)

let optop_coverage (op : Sched.optop) : coverage =
  match op with
  | Sched.Split _ ->
      Covered
        [
          "split-then-unroll"; "split-then-swap"; "split-then-vectorize-inner";
          "split-then-vectorize-narrow";
        ]
  | Sched.Swap _ -> Covered [ "split-then-swap" ]
  | Sched.Retype _ ->
      Covered
        [
          "retype-vectorized"; "split-then-vectorize-inner"; "split-then-vectorize-narrow";
          "retype-workgroup-reduce"; "retype-workgroup";
        ]
  | Sched.Unroll _ -> Covered [ "unroll-annot"; "unroll-mat"; "unroll-outer-mat" ]
  | Sched.Partition _ -> Covered [ "partition"; "partition-then-unroll"; "partition-outer" ]
  | Sched.Pad _ -> Covered [ "pad-then-unroll-mat" ]
  | Sched.Split_reduce _ -> Covered [ "split-reduce" ]
  | Sched.Tensorize _ -> Covered [ "tile-mma-fallback" ]
  | Sched.Expand_zero _ -> Covered [ "tile-mma-fallback" ]
  | Sched.Stage _ ->
      Out_of_scope
        "stages an OPERAND into shared memory: it changes where the reduction reads from, never \
         where its accumulator lives"
  | Sched.Privatize _ ->
      Out_of_scope
        "gives a scratch node a per-thread copy, which is a parallelism decision about a different \
         node than the one being accumulated"
  | Sched.Fuse_epilogue _ ->
      Out_of_scope
        "splices a consumer AFTER the reduction's closing store, so the form it follows is \
         whichever one this table already pins"

let axis_coverage (ty : LL.axis_type) : coverage =
  match ty with
  | LL.Serial -> Covered [ "serial"; "decline-data-guard"; "decline-sibling-statement" ]
  | LL.Unrolled -> Covered [ "unroll-annot"; "decline-sibling-unrolled" ]
  | LL.Vectorized ->
      Covered [ "retype-vectorized"; "split-then-vectorize-inner"; "split-then-vectorize-narrow" ]
  | LL.Workgroup_reduce -> Covered [ "retype-workgroup-reduce" ]
  | LL.Workgroup -> Covered [ "retype-workgroup" ]
  | LL.Grid ->
      Out_of_scope
        "its fallback is the plain serial loop and NOT the localizing one — the one arm that \
         serializes without localizing. No schedule op in the tree produces an unbindable, \
         non-parallel-eligible Grid level over a reduction axis, and one that existed would be a \
         cross-thread race rather than a width question"

(* Sample values, one per constructor, purely to drive the printed table: the classification itself
   is the exhaustive [match] above, which is what the compiler checks. A constructor added without a
   sample here still fails the build at that match — add both. *)
let coverage_sym = Ll_test.sym ()

let coverage_node =
  Int.incr next_id;
  Tn.create (Tn.Specified Ops.single) ~id:!next_id ~label:[ "rfcov" ]
    ~unpadded_dims:(lazy [| 1 |])
    ~padding:(lazy None)
    ()

let optop_samples : Sched.optop list =
  let s = coverage_sym in
  [
    Sched.Split
      { axis = s; factor = 1; outer = LL.Serial; inner = LL.Serial; outer_index = s; inner_index = s };
    Sched.Swap { outer = s; inner = s };
    Sched.Retype { axis = s; ty = LL.Serial };
    Sched.Unroll { axis = s; materialize = false };
    Sched.Partition { axis = s; breakpoints = []; segment_indices = [] };
    Sched.Pad { axis = s; to_multiple_of = 1 };
    Sched.Stage
      {
        source = coverage_node;
        tile_loops = [];
        shared = false;
        cooperative = None;
        hoisted = false;
        swizzle = None;
        pad_stride = None;
        pipeline_depth = 0;
        tile_prec = None;
      };
    Sched.Privatize { target = coverage_node; over = s };
    Sched.Expand_zero { tn = coverage_node; indices = [] };
    Sched.Tensorize { i = s; j = s; k = s; lane = s; simd_width = 1 };
    Sched.Fuse_epilogue { target = coverage_node; shared = false };
    Sched.Split_reduce
      {
        axis = s;
        target = coverage_node;
        num_blocks = 1;
        block_index = s;
        inner_index = s;
        combine_indices = [];
      };
  ]

let axis_samples : LL.axis_type list =
  [ LL.Serial; LL.Unrolled; LL.Vectorized; LL.Workgroup; LL.Workgroup_reduce; LL.Grid ]

(* The constructor's own name, off its sexp — so the printed table cannot disagree with the value
   it describes. *)
let constructor_name sexp =
  match sexp with Sexp.List (Sexp.Atom name :: _) -> name | Sexp.Atom name -> name | _ -> "?"

(* {1 Running the table} *)

let same_form a b = String.equal (form_name a) (form_name b)

let agrees got want =
  Array.length got = Array.length want && Array.for_all2_exn got want ~f:Float.equal

let show vs = String.concat ~sep:" " (Array.to_list (Array.map vs ~f:(Printf.sprintf "%h")))
let routine_stem slug = String.tr slug ~target:'-' ~replacement:'_'

(* The coverage tables, printed and asserted: every constructor is reached by a member that
   EXISTS, or exempted with a reason. A slug naming no member fails the claim, so renaming a member
   without updating its coverage entry is a failure rather than a quiet gap. *)
let () =
  let slugs =
    "tile-mma-fallback" :: List.map members ~f:(fun m -> m.slug) |> Set.of_list (module String)
  in
  let report ~what ~name_of ~classify samples =
    let ok = ref true in
    List.iter samples ~f:(fun sample ->
        let name = name_of sample in
        match classify sample with
        | Covered [] ->
            ok := false;
            Stdio.printf "  %-18s covered by nothing\n" name
        | Covered members_of ->
            let missing =
              List.filter members_of ~f:(fun slug -> not (Set.mem slugs slug))
            in
            if not (List.is_empty missing) then begin
              ok := false;
              Stdio.eprintf "  %s cites members that do not exist: %s\n" name
                (String.concat ~sep:", " missing)
            end;
            Stdio.printf "  %-18s covered by %s\n" name (String.concat ~sep:", " members_of)
        | Out_of_scope why -> Stdio.printf "  %-18s out of scope: %s\n" name why);
    p
      (Printf.sprintf
         "every %s constructor is reached by a member that exists, or exempted with a reason" what)
      !ok
  in
  Stdio.printf "Schedule.optop coverage:\n";
  report ~what:"Schedule.optop"
    ~name_of:(fun op -> constructor_name (Sched.sexp_of_optop op))
    ~classify:optop_coverage optop_samples;
  Stdio.printf "Low_level.axis_type coverage:\n";
  report ~what:"Low_level.axis_type"
    ~name_of:(fun ty -> constructor_name (LL.sexp_of_axis_type ty))
    ~classify:axis_coverage axis_samples;
  (* A duplicated sample would print one constructor twice and leave another unprinted, while the
     exhaustive matches above stayed satisfied. *)
  let distinct l =
    List.length (List.dedup_and_sort l ~compare:String.compare) = List.length l
  in
  p "the coverage samples name distinct constructors"
    (distinct (List.map optop_samples ~f:(fun op -> constructor_name (Sched.sexp_of_optop op)))
    && distinct (List.map axis_samples ~f:(fun ty -> constructor_name (LL.sexp_of_axis_type ty))))

let () =
  Stdio.printf "reduction: out[%d] = sum of %d terms, hand-built Low_level, %d members\n" rows cols
    (List.length members + 1);
  List.iteri members ~f:(fun i m ->
      Stdio.printf "  %02d %-27s [%-12s] over %-21s %-70s -> %s\n" (i + 1) m.slug
        (String.concat ~sep:" " m.precisions)
        (shape_name m.shape) m.what m.claimed);
  Stdio.printf "  %02d %-27s [%-12s] over %-21s %-70s -> %s\n"
    (List.length members + 1)
    "tile-mma-fallback" "f32 bf16" "small contraction"
    "Tensorize whose emission preconditions fail (transposed-B operands)"
    (form_name Mma_fallback)

(* The baselines: the plain nest with no schedule ops, one per precision. Every localizing member
   is compared against these, so a regression that moved the BASELINE would fail its own host-
   reference claim below rather than silently shifting the whole table. *)
let baselines =
  List.map precs ~f:(fun (prec_name, prec) ->
      let prog = make ~prec ~shape:Plain () in
      (prec_name, execute ~name:("rf_baseline_" ^ prec_name) ~prog ~sched:[]))

let baseline prec_name = List.Assoc.find_exn baselines ~equal:String.equal prec_name

let () =
  List.iter precs ~f:(fun (prec_name, prec) ->
      let wide = whole_nest_ref prec and stepped = per_step_ref prec in
      let got = baseline prec_name in
      let differ = not (Array.for_all2_exn wide stepped ~f:Float.equal) in
      (* The discrimination control. Without it every value claim in the table could hold because
         the operands never leave the storage format's exactness range, in which case a form that
         narrowed at every step would be indistinguishable from one that narrows once. *)
      if String.equal prec_name "f32" then
        p "at f32 the whole-nest and per-step references coincide (the identity-precision leg)"
          (not differ)
      else
        p
          (Printf.sprintf
             "at %s the whole-nest and per-step references differ (the operands discriminate \
              accumulator width)"
             prec_name)
          differ;
      let matches_wide = Array.for_all2_exn got wide ~f:Float.equal in
      let matches_stepped = Array.for_all2_exn got stepped ~f:Float.equal in
      if not (matches_wide || matches_stepped) then
        Stdio.eprintf "  baseline %s: got [%s] wide [%s] per-step [%s]\n" prec_name (show got)
          (show wide) (show stepped);
      p
        (Printf.sprintf "the %s serial baseline is one of the two host references" prec_name)
        (matches_wide || matches_stepped);
      (* Which regime this backend is in, on stderr so the golden stays backend-uniform: whether
         the accumulator resolves wider than storage decides whether the localized and
         read-modify-write forms are also distinguishable BY VALUE here, or only structurally. *)
      Stdio.eprintf "accumulator residency at %s: %s\n%!" prec_name
        (match (matches_wide, matches_stepped) with
        | true, false -> "wider than storage (whole-nest)"
        | false, true -> "storage width (per-step)"
        | true, true -> "indistinguishable by value at this precision"
        | false, false -> "neither host reference"))

let () =
  List.iter members ~f:(fun m ->
      List.iter precs ~f:(fun (prec_name, prec) ->
          if List.mem m.precisions prec_name ~equal:String.equal then begin
            let form_claim =
              Printf.sprintf "%s @ %s renders the form its composition claims" m.slug prec_name
            in
            let value_claim =
              Printf.sprintf "%s @ %s agrees with its reference value" m.slug prec_name
            in
            if not (m.available prec_name) then begin
              skipped form_claim;
              skipped value_claim
            end
            else begin
              let name = Printf.sprintf "rf_%s_%s" (routine_stem m.slug) prec_name in
              let prog = make ~prec ~shape:m.shape () in
              let got = execute ~name ~prog ~sched:(m.sched prog) in
              let src = Generated.read name in
              let extra_ok =
                List.for_all m.extra ~f:(fun sub -> String.is_substring src ~substring:sub)
              in
              if not extra_ok then
                Stdio.eprintf "  %s: the kernel lacks one of [%s]\n" name
                  (String.concat ~sep:"; " m.extra);
              (match read_form src ~label:"rfout" with
              | None ->
                  Stdio.eprintf "  %s: no accumulator identifier in the emitted kernel\n" name;
                  p form_claim false
              | Some reading ->
                  let rendered = form_of reading in
                  (* The coarse form is not the whole claim: a localizing composition must close the
                     cell exactly as often as it opens the output loop's body, and touch the node no
                     more than that (Codex P2, round 1). The declining and partial forms are
                     identified by what they do to the node in the first place, so the count adds
                     nothing there. *)
                  let sites_ok =
                    match m.expect with
                    | Localized ->
                        reading.stores_from_local = m.store_sites
                        && reading.node_accesses <= 2 * m.store_sites
                    | Simd | Warp ->
                        (* A vector or shuffle epilogue closes the cell ONCE per nest, and may
                           spell that as [out[i] = out[i] + vred_total] rather than as a store
                           from a scope local — textually a read-modify-write, but one per nest
                           instead of one per step, which is the whole distinction. Either
                           spelling counts, and the access bound is what refuses a regression
                           that closed once per chain. *)
                        reading.stores_from_local + reading.rmw_statements = m.store_sites
                        && reading.node_accesses <= 2 * m.store_sites
                    | Partials_combine | Rmw | Mma_fallback -> true
                  in
                  if not sites_ok then
                    Stdio.eprintf
                      "  %s: %d closing store(s) and %d node subscript(s), expected %d and at most \
                       %d\n"
                      name reading.stores_from_local reading.node_accesses m.store_sites
                      (2 * m.store_sites);
                  if not (same_form rendered m.expect) then
                    Stdio.eprintf
                      "  %s: rendered %s, claimed %s (node subscripts %d, rmw statements %d, \
                       stores from a local %d, foreign local stores %d)\n"
                      name (form_name rendered) (form_name m.expect) reading.node_accesses
                      reading.rmw_statements reading.stores_from_local reading.foreign_local_stores;
                  p form_claim (same_form rendered m.expect && sites_ok && extra_ok));
              let want =
                match m.reference with
                | Baseline -> baseline prec_name
                | Per_step -> per_step_ref prec
              in
              let ok = agrees got want in
              if not ok then
                Stdio.eprintf "  %s: got [%s] want [%s]\n" name (show got) (show want);
              p value_claim ok
            end
          end))

(* {1 The [Tile_mma] fallback}

   The register-tiled micro-kernel holds its C-tile across the whole k extent, and when its emission
   preconditions fail it renders the lane-0 SCALAR fallback — whose reduction is a serial nest like
   any other, and therefore another member of this set (gh-ocannl-663 review round 2, "scopes inside
   [Tile_mma] fallbacks"). It needs an i/j/k triple, so it rides a small contraction rather than the
   row sum, and it is read through the routine's own [Tile_mma] census rather than through the
   textual classifier: [Scalar_fallback] is exactly "every [Tile_mma] statement declined", which is
   the fact the member claims.

   The transposed-B operand layout is the decline the C backends actually have (the gradient-GEMM
   shape); on a backend whose renderer does not decline there, the member reports skipped. *)

let mma_n = 16

let mma_matmul ~tag ~prec =
  let av = Array.init (mma_n * mma_n) ~f:(fun t -> Float.of_int (t % 7) *. 0.25) in
  let bv = Array.init (mma_n * mma_n) ~f:(fun t -> Float.of_int (t % 5) *. 0.5) in
  let ma = TDSL.ndarray av ~label:[ tag ^ "_a" ] ~output_dims:[ mma_n; mma_n ] () in
  let mb = TDSL.ndarray bv ~label:[ tag ^ "_b" ] ~output_dims:[ mma_n; mma_n ] () in
  let%op mc = ma +* "ik;jk=>ij" mb in
  Tn.update_prec mc.Tensor.value prec;
  mc

let mma_nest_syms (opt : LL.optimized) =
  let strip stmts = List.filter stmts ~f:(function LL.Noop | LL.Comment _ -> false | _ -> true) in
  let rec path (llc : LL.t) : Idx.symbol list =
    match llc with
    | LL.For_loop { index; body; _ } ->
        index :: (match strip (LL.flat_lines [ body ]) with [ single ] -> path single | _ -> [])
    | LL.If { body; _ } -> path body
    | _ -> []
  in
  let paths =
    List.filter_map (LL.flat_lines [ opt.LL.llc ]) ~f:(fun stmt ->
        match path stmt with [] -> None | pth -> Some pth)
  in
  match List.find_exn paths ~f:(fun pth -> List.length pth = 3) with
  | [ i; j; k ] -> (i, j, k)
  | _ -> assert false

let mma_run ~name ~out ~tensorize comp =
  let transform (opt : LL.optimized) =
    if not tensorize then opt
    else
      let i, j, k = mma_nest_syms opt in
      let ez, zsyms = Sched.expand_zero ~tn:out in
      let zj = match zsyms with [ _; zj ] -> zj | _ -> assert false in
      let tz, _lane = Sched.tensorize ~i ~j ~k ~simd_width:mma_n in
      Sched.apply [ ez; Sched.Retype { axis = zj; ty = LL.Workgroup }; tz ] opt
  in
  let ctx, routine =
    Context.compile ~name ~lowered_transform:transform (Lazy.force base_ctx) comp Idx.Empty
  in
  let ctx = Context.run ctx routine in
  (Context.get_values ctx out, routine.Context.mma)

let () =
  List.iter [ ("f32", Ops.single); ("bf16", Ops.bfloat16) ] ~f:(fun (prec_name, prec) ->
      let form_claim =
        Printf.sprintf "tile-mma-fallback @ %s renders the form its composition claims" prec_name
      in
      let value_claim =
        Printf.sprintf "tile-mma-fallback @ %s agrees with its reference value" prec_name
      in
      if not on_cpu then begin
        skipped form_claim;
        skipped value_claim
      end
      else begin
        let plain = mma_matmul ~tag:("rfmma_p_" ^ prec_name) ~prec in
        let want, _ =
          mma_run
            ~name:("rf_mma_plain_" ^ prec_name)
            ~out:plain.Tensor.value ~tensorize:false
            (Train.forward plain)
        in
        let tiled = mma_matmul ~tag:("rfmma_t_" ^ prec_name) ~prec in
        let got, census =
          mma_run
            ~name:("rf_mma_fallback_" ^ prec_name)
            ~out:tiled.Tensor.value ~tensorize:true
            (Train.forward tiled)
        in
        p form_claim
          (Cs.equal_tensorization census.Cs.tensorization Cs.Scalar_fallback
          && census.Cs.statements > 0
          && census.Cs.scalar_fallbacks = census.Cs.statements);
        let ok = agrees got want in
        if not ok then
          Stdio.eprintf "  rf_mma_fallback_%s: got [%s] want [%s]\n" prec_name (show got)
            (show want);
        p value_claim ok
      end)
