(* Precision-neutral accumulator localization (gh-ocannl-693).

   A recognized serial reduction nest holds its accumulator in a scope LOCAL across the whole nest
   and stores once, after the nest — at EVERY precision, not only where the numerics policy asks for
   a widening (gh-ocannl-639). Before this, [C_syntax.try_localize_serial_reduce] declined whenever
   [acc_prec] was the identity on the storage precision, so an f32 reduction that no schedule op
   reached did one global read-modify-write per inner iteration: a scalar loss reduction, a gradient
   accumulated over a batch loop, an unmatched contraction. On Metal that residency was not merely
   likely but guaranteed — [volatile_serial_accumulation] shadows exactly that statement shape with
   a volatile-qualified alias, to stop the shader compiler touching it.

   The claims are therefore both structural and executed:

   - the emitted f32 kernel opens the accumulator into a local, updates the local inside the
   reduction loop, and writes the node once — no [acc[k] = ... acc[k] ...] inside the loop; - the
   workaround decision agrees with what the compile REPORTED (gh-ocannl-782): the routine's
   volatility census names the local, says whether its accumulating device reads use the workaround,
   and says whether this backend asked for the workaround at all — so the expectation is read off
   the compile instead of off the backend's name, and the emitted text is checked against it. On a
   backend that requests the workaround the accumulator stays plain and register-resident while the
   accumulating loop's device reads use expression-level [volatile] pointer casts (gh-ocannl-820);
   on one that does not, both stay plain. Either way the kernel carries no RMW shadow declaration:
   localization lifted that device-memory RMW, and the census's RMW-read count says so independently
   of the text; - the values match a host reference computed in the same summation order. The
   producers discriminate: every element is [1 + 10*i + j], so it varies with BOTH loop symbols and
   is clear of the zero the accumulator is initialized to — a constant producer would survive a
   dropped or replayed iteration, and a value omitting a symbol would survive a wrong substitution.

   Two nest shapes, because they exercise different placements of the localized store: a scalar
   reduction over both axes (the loss shape — the store lands above every loop, at function scope)
   and a row reduction (the batch-gradient / contraction shape — the store lands inside the
   surviving output loop, whose symbol the cell DOES mention, so the volatile predicate is false
   there too). *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module Tn = Ir.Tnode

let () = Utils.settings.output_debug_files_in_build_directory <- true
let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")
let () = Test_utils.Generated.init ~backend_name
let rows = 6
let cols = 7

(* Discriminating producer: varies with both symbols, never zero. *)
let elem i j = 1.0 +. (10.0 *. Float.of_int i) +. Float.of_int j
let data = Array.init (rows * cols) ~f:(fun n -> elem (n / cols) (n % cols))

let x =
  TDSL.ndarray data ~label:[ "resx" ] ~batch_dims:[] ~input_dims:[] ~output_dims:[ rows; cols ] ()

(* Leg 1: scalar reduction over both axes — the loss shape. *)
let%op total = x ++ "ij => 0"

(* Leg 2: reduction over the row axis only — the batch-gradient / contraction shape. *)
let%op per_col = x ++ "ij => j"

let () =
  Train.set_materialized total.Tensor.value;
  Train.set_materialized per_col.Tensor.value;
  let ctx = Context.auto () in
  let ctx, r_total =
    Context.compile ~name:"res_total" ctx (Train.forward total) Ir.Indexing.Empty
  in
  let ctx = Context.run ctx r_total in
  let ctx, r_per_col =
    Context.compile ~name:"res_per_col" ctx (Train.forward per_col) Ir.Indexing.Empty
  in
  let ctx = Context.run ctx r_per_col in
  let got_total = (Context.get_values ctx total.Tensor.value).(0) in
  let got_per_col = Context.get_values ctx per_col.Tensor.value in

  (* Host reference, same summation order as the emitted nest. *)
  let ref_total =
    let acc = ref 0.0 in
    for i = 0 to rows - 1 do
      for j = 0 to cols - 1 do
        acc := !acc +. elem i j
      done
    done;
    !acc
  in
  let ref_per_col =
    Array.init cols ~f:(fun j ->
        let acc = ref 0.0 in
        for i = 0 to rows - 1 do
          acc := !acc +. elem i j
        done;
        !acc)
  in
  Verdict.p "scalar reduction matches host reference" Float.(abs (got_total - ref_total) < 1e-3);
  Verdict.p "row reduction matches host reference"
    (Array.length got_per_col = cols
    && Array.for_all2_exn got_per_col ref_per_col ~f:(fun a b -> Float.(abs (a - b) < 1e-3)));

  (* The reference must be able to tell a dropped-iteration kernel apart from a correct one: with
     [rows * cols] terms all distinct and nonzero, a last-iteration-only result is far away. *)
  let last_only = elem (rows - 1) (cols - 1) in
  Verdict.p "reference discriminates a last-iteration-only result"
    Float.(abs (ref_total - last_only) > 1.0);

  (* Structural: the localized form. The scope local is [v<scope_id>_<ident>], where [<ident>] is
     the code name codegen derived for the node -- not predictable from the tensor's label (it goes
     through a blacklist and a dot-stripping pass), so it is READ OFF the opening init rather than
     guessed: the only statement of the shape [v<n>_<ident> = <ident>[...]] is the scope's opening
     read of the accumulator cell. If no such statement exists the accumulator was not localized,
     which is the failure this test exists to catch.

     Split on [;] rather than on newlines: the pretty-printer breaks a long value expression across
     lines, so a line-based read of "does this statement touch the node twice" would answer no for
     exactly the wide read-modify-write it is meant to catch. *)
  let normalize st = String.concat ~sep:" " (String.split_on_chars st ~on:[ '\n'; '\t' ]) in
  let scope_init st =
    match String.substr_index st ~pattern:" = " with
    | None -> None
    | Some at ->
        (* Splitting on [;] carries a [for] header's tail into the next statement, so the assigned
           name is the LAST token before the [=], not the whole prefix. *)
        let lhs =
          match List.rev (String.split_on_chars (String.prefix st at) ~on:[ ' '; '\n'; '\t' ]) with
          | last :: _ -> last
          | [] -> ""
        in
        let rhs = String.strip (String.drop_prefix st (at + 3)) in
        Option.bind (String.index lhs '_') ~f:(fun us ->
            let ident = String.drop_prefix lhs (us + 1) in
            let is_scope_local =
              String.is_prefix lhs ~prefix:"v" && us > 1
              && String.for_all (String.sub lhs ~pos:1 ~len:(us - 1)) ~f:Char.is_digit
              && not (String.is_empty ident)
            in
            if is_scope_local && String.is_prefix rhs ~prefix:(ident ^ "[") then
              Some (lhs, ident, st)
            else None)
  in
  (* The census names the local by the identifier the declaration carries ([scope_local_ident] is
     the one definition of that convention), so "is this one volatile" can be answered off the
     emitted text without guessing where the qualifier would sit. *)
  let declared_volatile source local =
    match String.substr_index source ~pattern:(" " ^ local) with
    | None -> None
    | Some at ->
        let before = String.prefix source at in
        let line = List.last_exn (String.split_lines before) in
        Some (String.is_substring line ~substring:"volatile")
  in
  let check_localized (compiled : Context.routine) routine label =
    let source = Test_utils.Generated.read routine in
    let statements = List.map (String.split source ~on:';') ~f:normalize in
    let fail_all () =
      List.iter
        [
          "accumulator opens into a scope local";
          "the reduction updates the local, not the node";
          "no statement both reads and writes the node";
          "the node is stored exactly once, from the local";
          "the census names the accumulator the kernel declares";
          "the accumulator declaration stays plain";
          "the accumulating loop's volatile-read form is the one the census reports";
          "the opening read stays plain";
          "no rmw shadow declaration, by the census and by the text";
        ] ~f:(fun c -> Verdict.p (label ^ ": " ^ c) false)
    in
    match List.find_map statements ~f:scope_init with
    | None -> fail_all ()
    | Some (local, ident, opening) ->
        let count st pattern =
          List.length (String.substr_index_all st ~may_overlap:false ~pattern)
        in
        let node_accesses st = count st (ident ^ "[") in
        let accumulation_updates =
          List.filter statements ~f:(fun st ->
              node_accesses st = 0
              && count st local >= 2
              && String.is_substring st ~substring:(local ^ " = "))
        in
        Verdict.p (label ^ ": accumulator opens into a scope local") true;
        (* The accumulation: assigns the local from itself, and never touches the node. *)
        Verdict.p
          (label ^ ": the reduction updates the local, not the node")
          (not (List.is_empty accumulation_updates));
        (* The read-modify-write shape is one statement reading and writing the node. Its absence is
           what localization buys; the [Zero_out] statement reaches the node once and is not it. *)
        Verdict.p_all (label ^ ": no statement both reads and writes the node") statements
          ~f:(fun st -> node_accesses st <= 1);
        Verdict.p
          (label ^ ": the node is stored exactly once, from the local")
          (1
          = List.count statements ~f:(fun st ->
              node_accesses st = 1 && String.is_substring st ~substring:("] = " ^ local)));
        (* What the compile REPORTED about this accumulator (gh-ocannl-782), and the emitted text
           checked against it. The expectation is the census's [requested] flag — the capability as
           the backend that ran this compile stated it — not a second reading of the backend name.
           These claims are therefore backend-uniform: they say the same thing on Metal, where the
           workaround is requested, and on the C backends, where it is not. *)
        let v = compiled.Context.volatility in
        let accumulations =
          List.filter_map v.Ir.C_syntax.entries ~f:(fun (_, site) ->
              match site with
              | Ir.C_syntax.Volatile_accumulation_reads name -> Some (name, true)
              | Ir.C_syntax.Plain_accumulator name -> Some (name, false)
              | Ir.C_syntax.Volatile_rmw_reads _ -> None)
        in
        Verdict.p
          (label ^ ": the census names the accumulator the kernel declares")
          (List.mem (List.map accumulations ~f:fst) local ~equal:String.equal);
        Verdict.p_all (label ^ ": the accumulator declaration stays plain") accumulations
          ~f:(fun (name, _) ->
            match declared_volatile source name with Some declared -> not declared | None -> false);
        Verdict.p_all
          (label ^ ": the accumulating loop's volatile-read form is the one the census reports")
          accumulations ~f:(fun (_, volatile_reads) ->
            Bool.equal volatile_reads v.Ir.C_syntax.requested
            && (not (List.is_empty accumulation_updates))
            && List.for_all accumulation_updates ~f:(fun st ->
                Bool.equal
                  (String.is_substring st ~substring:"device volatile float*")
                  volatile_reads));
        Verdict.p
          (label ^ ": the opening read stays plain")
          (not (String.is_substring opening ~substring:"volatile"));
        (* Localization lifted the device-memory read-modify-write the pointer shadow pins, so this
           routine has none — asserted twice over, from the census and from the emitted text, which
           is what makes either one a check rather than a restatement. *)
        Verdict.p
          (label ^ ": no rmw shadow declaration, by the census and by the text")
          (v.Ir.C_syntax.volatile_rmw_reads = 0
          && not (String.is_substring source ~substring:"__rmw_"))
  in
  check_localized r_total "res_total" "scalar reduction";
  check_localized r_per_col "res_per_col" "row reduction"

(* {1 The volatility census's own bracket}

   [with_volatility_census] is what makes the census a property of the compiled routine rather than
   of whichever caller remembered to collect it (gh-ocannl-782), so its nesting and restoration are
   pinned directly — with hand-written entries, so these claims do not depend on what any backend
   renders. [requested] is part of the state it restores: a nested compile on a backend that asks
   for the workaround must not leave an enclosing collection claiming the flag of the inner one. *)
let () =
  let module Cs = Ir.C_syntax in
  let entries_equal =
    List.equal (fun (n1, s1) (n2, s2) -> String.equal n1 n2 && Cs.equal_volatility_site s1 s2)
  in
  let outer = Cs.Volatile_accumulation_reads "v1_outer"
  and inner = Cs.Plain_accumulator "v2_inner" in
  let inner_summary, outer_summary =
    Cs.with_volatility_census (fun () ->
        Cs.volatility_census := ("outer_kernel", outer) :: !Cs.volatility_census;
        Cs.volatility_requested := true;
        let (), inner_summary =
          Cs.with_volatility_census (fun () ->
              Cs.volatility_census := ("inner_kernel", inner) :: !Cs.volatility_census)
        in
        inner_summary)
  in
  Verdict.p "an inner bracket summarizes only its own sites"
    (entries_equal inner_summary.Cs.entries [ ("inner_kernel", inner) ]);
  (* Additively, not shadowing: an enclosing collection still sees what an inner compile censused,
     or wrapping the compile path in the bracket would silently empty every outer one. *)
  Verdict.p "the enclosing bracket sees both, in emission order"
    (entries_equal outer_summary.Cs.entries [ ("outer_kernel", outer); ("inner_kernel", inner) ]);
  Verdict.p "an inner bracket reports the capability of the compile it bracketed"
    (not inner_summary.Cs.requested);
  Verdict.p "the enclosing bracket keeps its own capability across the nested one"
    outer_summary.Cs.requested;
  Verdict.p "the counts classify what was collected"
    (outer_summary.Cs.volatile_accumulations = 1
    && outer_summary.Cs.plain_accumulations = 1
    && outer_summary.Cs.volatile_rmw_reads = 0);
  Verdict.p "a completed bracket leaves the census global as it found it"
    (List.is_empty !Cs.volatility_census);
  Verdict.p "collection is off outside every bracket" (not !Cs.volatility_census_enabled)
