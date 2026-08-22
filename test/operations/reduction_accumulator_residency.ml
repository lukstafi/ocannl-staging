(* Precision-neutral accumulator localization (gh-ocannl-693).

   A recognized serial reduction nest holds its accumulator in a scope LOCAL across the whole nest
   and stores once, after the nest — at EVERY precision, not only where the numerics policy asks for
   a widening (gh-ocannl-639). Before this, [C_syntax.try_localize_serial_reduce] declined whenever
   [acc_prec] was the identity on the storage precision, so an f32 reduction that no schedule op
   reached did one global read-modify-write per inner iteration: a scalar loss reduction, a gradient
   accumulated over a batch loop, an unmatched contraction. On Metal that residency was not merely
   likely but guaranteed — [volatile_scalar_rmw] shadows exactly that statement shape with a
   volatile-qualified alias, to stop the shader compiler touching it.

   The claims are therefore both structural and executed:

   - the emitted f32 kernel opens the accumulator into a local, updates the local inside the
     reduction loop, and writes the node once — no [acc[k] = ... acc[k] ...] inside the loop;
   - on Metal the localized kernel carries no [volatile] alias, because the read-modify-write the
     shadow pins is gone. That is a property of WHERE the rewrite puts the store, not of the
     shadow's predicate: localization lifts the [Set] out of the very loops across which the cell
     was invariant, so the predicate is already false. (Nothing to evaluate on a backend whose
     [volatile_scalar_rmw] is [false] — reported as skipped rather than as a vacuous pass.)
   - the values match a host reference computed in the same summation order. The producers
     discriminate: every element is [1 + 10*i + j], so it varies with BOTH loop symbols and is
     clear of the zero the accumulator is initialized to — a constant producer would survive a
     dropped or replayed iteration, and a value omitting a symbol would survive a wrong
     substitution.

   Two nest shapes, because they exercise different placements of the localized store: a scalar
   reduction over both axes (the loss shape — the store lands above every loop, at function scope)
   and a row reduction (the batch-gradient / contraction shape — the store lands inside the surviving
   output loop, whose symbol the cell DOES mention, so the volatile predicate is false there too). *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module Tn = Ir.Tnode

let () = Utils.settings.output_debug_files_in_build_directory <- true

let backend_name =
  String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")

(* [volatile_scalar_rmw] is a backend source constant, not a config key; Metal is the only backend
   that sets it. On every other backend "the kernel carries no volatile alias" is vacuously true and
   is reported as skipped instead. *)
let has_volatile_shadow = String.is_substring backend_name ~substring:"metal"
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
        (* Splitting on [;] carries a [for] header's tail into the next statement, so the
           assigned name is the LAST token before the [=], not the whole prefix. *)
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
            if is_scope_local && String.is_prefix rhs ~prefix:(ident ^ "[") then Some (lhs, ident)
            else None)
  in
  let check_localized routine label =
    let statements =
      List.map (String.split (Test_utils.Generated.read routine) ~on:';') ~f:normalize
    in
    let fail_all () =
      List.iter
        [
          "accumulator opens into a scope local";
          "the reduction updates the local, not the node";
          "no statement both reads and writes the node";
          "the node is stored exactly once, from the local";
          "no volatile shadow on the localized store";
        ]
        ~f:(fun c -> Verdict.p (label ^ ": " ^ c) false)
    in
    match List.find_map statements ~f:scope_init with
    | None -> fail_all ()
    | Some (local, ident) ->
        let count st pattern =
          List.length (String.substr_index_all st ~may_overlap:false ~pattern)
        in
        let node_accesses st = count st (ident ^ "[") in
        Verdict.p (label ^ ": accumulator opens into a scope local") true;
        (* The accumulation: assigns the local from itself, and never touches the node. *)
        Verdict.p
          (label ^ ": the reduction updates the local, not the node")
          (List.exists statements ~f:(fun st ->
               node_accesses st = 0 && count st local >= 2
               && String.is_substring st ~substring:(local ^ " = ")));
        (* The read-modify-write shape is one statement reading and writing the node. Its absence
           is what localization buys; the [Zero_out] statement reaches the node once and is not
           it. *)
        Verdict.p
          (label ^ ": no statement both reads and writes the node")
          (List.for_all statements ~f:(fun st -> node_accesses st <= 1));
        Verdict.p
          (label ^ ": the node is stored exactly once, from the local")
          (1
          = List.count statements ~f:(fun st ->
                node_accesses st = 1 && String.is_substring st ~substring:("] = " ^ local)));
        if has_volatile_shadow then
          Test_utils.Generated.assert_omits ~routine ~contains:"volatile"
            (label ^ ": no volatile shadow on the localized store")
        else
          Verdict.skipped ~backend:backend_name
            (label ^ ": no volatile shadow on the localized store")
  in
  check_localized "res_total" "scalar reduction";
  check_localized "res_per_col" "row reduction"
