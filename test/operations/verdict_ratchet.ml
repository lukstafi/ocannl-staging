(* gh-ocannl-668: no test source prints a self-decided claim outside `Verdict`.

   gh-ocannl-601 settled the rule. A `(test)` stanza gates a run on two things — the exit status,
   and the diff against the golden — and a test that prints `<claim>: false` and exits 0 has only
   the second. That gate is promotable: the diff fails, the natural next move is `dune promote`, and
   the failure becomes the expected output. In a golden made of verdict lines a blessed regression
   and a deliberately recorded negative fact are the same text, so nothing fails again until someone
   reads the file. Routing the claim through `Verdict` adds the first gate by construction.

   The sweep that converted the 125 literal-label sites then in tree was one-time, and it left no
   mechanical trace. `test/operations/bandwidth_calibration.ml`, written afterwards for
   gh-ocannl-578, arrived with four fresh `Stdio.printf "…: %b\n"` claims and nothing failed,
   warned, or so much as remarked on it -- they were converted much later, in passing, by work whose
   subject was something else. The convention lived in prose, and a new test is written by matching
   a neighbour: in `test/operations` the neighbours are a mixture, legitimate descriptive `%b`
   prints sitting next to converted assertions, so the local example does not teach the rule. This
   is the mechanical trace.

   What it flags is a format whose LAST argument-consuming conversion is a bare `%b` at the end,
   behind a label ending in a colon, an equals sign or an arrow -- in either of two kinds. A LITERAL
   label is written out (`"k-blocks fused: %b\n"`); a COMPUTED one is built from arguments (`"%s
   aligned: %b\n"`, `"Epoch %d, loss below threshold=%b\n"`).

   Both were once out of reach, for different reasons, and gh-ocannl-624 closed both. The computed
   form had nowhere to go: `Verdict.p` takes a label, not a format, so converting a computed claim
   meant splitting the line by hand and every site was a judgement call. `Verdict.pf` and
   `Verdict.claimf` are that missing entry point, and with the sweep done the shape can be held. The
   separator was the quieter half: reading only a colon, this check could not see `"round-trip
   identity = %b\n"` -- the spelling most of the sites it was written for actually used, in
   `data_parallel`, `shard_transfer`, `test_buffer_loc` and a dozen more. Neither hole showed up as
   a failure. Both showed up as a clean report.

   A descriptive `%b` print therefore has one escape hatch left, not two. Carrying a second
   conversion no longer works, because a computed label carries one by construction; what remains is
   a named exemption below with the reason the line is not an assertion. The list is short, and the
   reason it stays short is structural: a print whose boolean is not a verdict is a census row or a
   table, and those rarely END on the boolean.

   gh-ocannl-801 closes the same escape hatch one level deeper. A parity or collection quantifier
   can sit in a file-local helper, with only [p "claim" (close got want)] left at the claim site.
   The gh-ocannl-729/746 sweeps could not find that shape by looking for the quantifier beside [p],
   and ten helpers carried the empty-population hole until a manual read found them. The second
   reader below follows local bindings from a Verdict boolean back to [for_all], [for_all2_exn],
   [is_empty], or a negated [exists], and requires the helper to make non-emptiness part of its
   passing result.

   Every synthetic control below earned its place by a mutation run -- the scanner mechanism it pins
   disabled, this alias re-run, exactly that control failing. The manifest of those runs, one row
   per mechanism with the control labels and the retained `tools/test-run.sh` run ids, is
   `test/operations/verdict_ratchet_controls.md`; renaming or adding a control updates both. *)

open Base
open Stdio

let printf = Test_utils.Refusal_control_manifest.printf

module Scan = Test_utils.Verdict_scan
module Dune = Test_utils.Dune_stanza_scan
module Sources = Test_utils.Config_key_scan
module Ast_traverse = Ppxlib.Ast_traverse
module Asttypes = Ppxlib.Asttypes
open Ppxlib.Parsetree

type quantifier_kind = For_all | For_all2 | Is_empty | Not_exists

let quantifier_name = function
  | For_all -> "for_all"
  | For_all2 -> "for_all2_exn"
  | Is_empty -> "is_empty"
  | Not_exists -> "not exists"

type quantifier = { kind : quantifier_kind; populations : Set.M(String).t; sealed : bool }
type claim_kind = P | Pf | Pass_fail | Claim | Claimf

(* A definition's [position] is its absolute character offset, which no two bindings share; [line]
   and [column] are for saying where it is. Identity has to be the offset: `let refused = … in`
   twice in one expression, or two local scopes written on one line, are two bodies a line number
   cannot tell apart -- and telling two bodies apart is the whole of what the exemption key rests
   on. *)
type definition_site = { line : int; column : int; position : int }

type helper_binding = {
  name : string;
  site : definition_site;
  optional_label : string option;
  dependencies : helper_dependency list;
  guards : Set.M(String).t;
  unguarded : quantifier list;
  negated_unguarded : quantifier list;
  constant_bool : bool option;
  quantifier_alias : (quantifier_kind * int) option;
  claim_kind : claim_kind option;
  claim_wrapper : wrapper_slot list option;
}

and helper_dependency = {
  binding : helper_binding;
  positive : bool;
  forwards_guards : bool;
  supplied_optional : Set.M(String).t option;
}

and wrapper_slot = {
  label : string option;
  optional : bool;
  unlabelled_index : int option;
  positive : bool;
  default_binding : helper_binding option;
}

type quantified_claim = {
  helper : string;
  helper_site : definition_site;
  claim_line : int;
  quantifiers : quantifier_kind list;
}

let describe_site site = Printf.sprintf "%d:%d" site.line site.column

let site_of_location location =
  let start = location.Ppxlib.Location.loc_start in
  {
    line = start.Stdlib.Lexing.pos_lnum;
    column = start.Stdlib.Lexing.pos_cnum - start.Stdlib.Lexing.pos_bol;
    position = start.Stdlib.Lexing.pos_cnum;
  }

let path_ends path ~container ~member =
  match List.rev path with
  | found_member :: found_container :: _ ->
      String.equal found_member member && String.equal found_container container
  | _ -> false

let is_collection_call expr ~member =
  match Sources.longident_of expr with
  | Some path ->
      path_ends path ~container:"Array" ~member || path_ends path ~container:"List" ~member
  | None -> false

let is_name expr name =
  match Sources.longident_of expr with
  | Some path -> Option.value_map (List.last path) ~default:false ~f:(String.equal name)
  | None -> false

let rec function_body expr =
  match expr.pexp_desc with
  | Pexp_function (_, _, Pfunction_body body) -> function_body body
  | _ -> expr

let unlabelled arguments =
  List.filter_map arguments ~f:(function Asttypes.Nolabel, argument -> Some argument | _ -> None)

type binding_part = {
  name : string;
  expression : expression;
  location : Ppxlib.Location.t;
  exact : bool;
}

let pattern_names pattern =
  let names = ref [] in
  let iterator =
    object
      inherit Ast_traverse.iter as super

      method! pattern pattern =
        (match pattern.ppat_desc with
        | Ppat_var { txt; _ } | Ppat_alias (_, { txt; _ }) ->
            names := (txt, pattern.ppat_loc) :: !names
        | _ -> ());
        super#pattern pattern
    end
  in
  iterator#pattern pattern;
  List.rev !names
  |> List.dedup_and_sort ~compare:(fun (left, _) (right, _) -> String.compare left right)

let conservative_binding_parts pattern expression =
  pattern_names pattern
  |> List.map ~f:(fun (name, location) -> { name; expression; location; exact = false })

(* A destructuring pattern is not permission to lose the binding. Literal tuples and records give
   each name its exact producer; for a shape we cannot align, every bound name retains the whole
   expression as a conservative producer. [returned_quantifiers] uses [exact] to decide whether it
   may restrict itself to the returned path or must inspect that producer broadly. *)
let rec binding_parts pattern expression =
  match (pattern.ppat_desc, expression.pexp_desc) with
  | Ppat_var { txt; _ }, _ ->
      [ { name = txt; expression; location = pattern.ppat_loc; exact = true } ]
  | Ppat_alias (inner, { txt; _ }), _ ->
      { name = txt; expression; location = pattern.ppat_loc; exact = true }
      :: binding_parts inner expression
  | Ppat_constraint (inner, _), _ -> binding_parts inner expression
  | Ppat_tuple patterns, Pexp_tuple expressions when List.length patterns = List.length expressions
    ->
      List.map2_exn patterns expressions ~f:binding_parts |> List.concat
  | Ppat_record (patterns, _), Pexp_record (expressions, None) ->
      List.map patterns ~f:(fun (pattern_label, pattern) ->
          List.find_map expressions ~f:(fun (expression_label, expression) ->
              if Poly.equal pattern_label.txt expression_label.txt then
                Some (binding_parts pattern expression)
              else None))
      |> Option.all
      |> Option.value_map ~default:(conservative_binding_parts pattern expression) ~f:List.concat
  | _ -> conservative_binding_parts pattern expression

let rec population_name expr =
  match Sources.longident_of expr with
  | Some [ name ] -> Some name
  | _ -> (
      match expr.pexp_desc with
      | Pexp_constraint (inner, _) | Pexp_coerce (inner, _, _) -> population_name inner
      | Pexp_apply (callee, _)
        when is_collection_call callee ~member:"filter"
             || is_collection_call callee ~member:"filter_map" ->
          (* A filtered view is its own population. Collapsing it to the source lets a non-empty
             [filter rows ~f:p1] guard a distinct, empty [filter rows ~f:p2]. Pretty-printing the
             location-free AST gives repeated spellings the same lexical identity while retaining
             the predicate that distinguishes derived populations. *)
          Some (Stdlib.Format.asprintf "%a" Ppxlib.Pprintast.expression expr)
      | _ -> None)

let populations arguments count =
  unlabelled arguments |> Fn.flip List.take count
  |> List.filter_map ~f:population_name
  |> Set.of_list (module String)

let collection_quantifier callee =
  if is_collection_call callee ~member:"for_all" then Some (For_all, 1)
  else if is_collection_call callee ~member:"for_all2_exn" then Some (For_all2, 2)
  else if is_collection_call callee ~member:"is_empty" then Some (Is_empty, 1)
  else if is_collection_call callee ~member:"exists" then Some (Not_exists, 1)
  else None

let bool_literal expr value =
  match expr.pexp_desc with
  | Pexp_construct ({ txt = Ppxlib.Longident.Lident found; _ }, None) ->
      String.equal found (Bool.to_string value)
  | _ -> false

let literal_bool expr =
  if bool_literal expr true then Some true else if bool_literal expr false then Some false else None

let rec bool_pattern_matches pattern value =
  match pattern.ppat_desc with
  | Ppat_construct ({ txt = Ppxlib.Longident.Lident found; _ }, None) ->
      String.equal found (Bool.to_string value)
  | Ppat_any | Ppat_var _ -> true
  | Ppat_alias (inner, _) | Ppat_constraint (inner, _) | Ppat_open (_, inner) ->
      bool_pattern_matches inner value
  | Ppat_or (left, right) -> bool_pattern_matches left value || bool_pattern_matches right value
  | _ -> false

let boolean_match_polarity cases =
  let output_for input =
    List.find cases ~f:(fun case -> bool_pattern_matches case.pc_lhs input)
    |> Option.bind ~f:(fun case ->
        if Option.is_none case.pc_guard then literal_bool case.pc_rhs else None)
  in
  match (output_for true, output_for false) with
  | Some true, Some false -> Some true
  | Some false, Some true -> Some false
  | _ -> None

let result_polarities positive result =
  if bool_literal result true then [ positive ]
  else if bool_literal result false then [ not positive ]
  else [ positive; not positive ]

let condition_polarities positive yes no =
  match no with
  | None -> []
  | Some no when bool_literal yes true && bool_literal no false -> [ positive ]
  | Some no when bool_literal yes false && bool_literal no true -> [ not positive ]
  | Some no
    when (bool_literal yes true && bool_literal no true)
         || (bool_literal yes false && bool_literal no false) ->
      []
  | Some _ -> []

let is_boolean_comparison callee =
  is_name callee "=" || is_name callee "<>" || is_name callee "equal"

let is_transparent_boolean_wrapper callee =
  match Sources.longident_of callee with
  | Some ([ "Fn"; "id" ] | [ "Fun"; "id" ] | [ "Stdlib"; "Fun"; "id" ]) -> true
  | _ -> false

let compared_argument callee left right ~positive =
  if is_boolean_comparison callee then
    let equal = not (is_name callee "<>") in
    match
      ( bool_literal left true,
        bool_literal left false,
        bool_literal right true,
        bool_literal right false )
    with
    | true, _, _, _ -> Some (right, Bool.equal positive equal)
    | _, true, _, _ -> Some (right, Bool.equal positive (not equal))
    | _, _, true, _ -> Some (left, Bool.equal positive equal)
    | _, _, _, true -> Some (left, Bool.equal positive (not equal))
    | _ -> None
  else None

let quantifiers_in ?(positive = true) expr =
  let found = ref [] in
  let positive = ref positive in
  let iterator =
    object (self)
      inherit Ast_traverse.iter as super
      method! attribute _ = ()

      method! expression expr =
        let visit_with_polarity polarity argument =
          let previous = !positive in
          positive := polarity;
          self#expression argument;
          positive := previous
        in
        match expr.pexp_desc with
        | Pexp_apply (callee, [ (Asttypes.Nolabel, argument) ]) when is_name callee "not" ->
            visit_with_polarity (not !positive) argument
        | Pexp_apply (apply, [ (Asttypes.Nolabel, function_); (Asttypes.Nolabel, argument) ])
          when is_name apply "@@" && is_name function_ "not" ->
            visit_with_polarity (not !positive) argument
        | Pexp_apply (pipe, [ (Asttypes.Nolabel, population); (Asttypes.Nolabel, piped_call) ])
          when is_name pipe "|>" -> (
            match piped_call.pexp_desc with
            | Pexp_ident _ when is_name piped_call "not" ->
                visit_with_polarity (not !positive) population
            | Pexp_apply (callee, arguments) -> (
                match collection_quantifier callee with
                | Some (kind, count)
                  when (!positive && not (Poly.equal kind Not_exists))
                       || ((not !positive) && Poly.equal kind Not_exists) ->
                    found :=
                      {
                        kind;
                        populations =
                          populations ((Asttypes.Nolabel, population) :: arguments) count;
                        sealed = false;
                      }
                      :: !found;
                    self#expression population;
                    List.iter arguments ~f:(fun (_, argument) -> self#expression argument)
                | _ -> (
                    match unlabelled arguments with
                    | [ literal ] -> (
                        match compared_argument callee literal population ~positive:!positive with
                        | Some (argument, polarity) -> visit_with_polarity polarity argument
                        | None -> super#expression expr)
                    | _ -> super#expression expr))
            | _ -> super#expression expr)
        | Pexp_apply (callee, [ (Asttypes.Nolabel, left); (Asttypes.Nolabel, right) ])
          when is_boolean_comparison callee -> (
            match compared_argument callee left right ~positive:!positive with
            | Some (argument, polarity) -> visit_with_polarity polarity argument
            | None -> super#expression expr)
        | Pexp_apply (callee, arguments) ->
            let add kind count =
              found := { kind; populations = populations arguments count; sealed = false } :: !found
            in
            if !positive && is_collection_call callee ~member:"for_all" then add For_all 1
            else if !positive && is_collection_call callee ~member:"for_all2_exn" then
              add For_all2 2
            else if !positive && is_collection_call callee ~member:"is_empty" then add Is_empty 1
            else if (not !positive) && is_collection_call callee ~member:"exists" then
              add Not_exists 1;
            super#expression expr
        | _ -> super#expression expr
    end
  in
  iterator#expression (function_body expr);
  List.rev !found

let int_literal expr =
  match expr.pexp_desc with
  | Pexp_constant (Pconst_integer (value, _)) -> Option.try_with (fun () -> Int.of_string value)
  | _ -> None

let length_population expr =
  match expr.pexp_desc with
  | Pexp_apply (callee, arguments) when is_collection_call callee ~member:"length" ->
      List.hd (unlabelled arguments) |> Option.bind ~f:population_name
  | _ -> None

(* Populations that HAVE to be non-empty for this expression to be true. This is intentionally a
   small boolean grammar: conjunction composes requirements, [not (X.is_empty xs)] is the spelling
   the gh-ocannl-746 helper sweep installed, and a positive literal length pins the same fact. A
   construct the reader cannot prove contributes no guard, so it produces a loud finding rather than
   silently licensing a vacuous helper. *)
let rec required_nonempty expr =
  let none () = Set.empty (module String) in
  match expr.pexp_desc with
  | Pexp_function (_, _, Pfunction_body body) -> required_nonempty body
  | Pexp_function (_, _, Pfunction_cases ([ case ], _, _)) -> required_nonempty case.pc_rhs
  | Pexp_function (_, _, Pfunction_cases (_, _, _)) -> Set.empty (module String)
  | Pexp_let (_, _, body) -> required_nonempty body
  | Pexp_letmodule (_, _, body) | Pexp_letexception (_, body) -> required_nonempty body
  | Pexp_open (_, body) -> required_nonempty body
  | Pexp_constraint (inner, _) | Pexp_coerce (inner, _, _) -> required_nonempty inner
  | Pexp_apply (callee, [ (Asttypes.Nolabel, argument) ]) when is_name callee "not" -> (
      match argument.pexp_desc with
      | Pexp_apply (empty, arguments) when is_collection_call empty ~member:"is_empty" ->
          populations arguments 1
      | _ -> none ())
  | Pexp_apply (op, [ (Asttypes.Nolabel, left); (Asttypes.Nolabel, right) ]) when is_name op "&&" ->
      Set.union (required_nonempty left) (required_nonempty right)
  | Pexp_apply (op, [ (Asttypes.Nolabel, left); (Asttypes.Nolabel, right) ]) when is_name op "="
    -> (
      match
        (length_population left, int_literal right, int_literal left, length_population right)
      with
      | Some population, Some n, _, _ when n > 0 -> Set.singleton (module String) population
      | _, _, Some n, Some population when n > 0 -> Set.singleton (module String) population
      | _ -> none ())
  | Pexp_apply (op, [ (Asttypes.Nolabel, left); (Asttypes.Nolabel, right) ]) -> (
      match
        ( Sources.longident_of op,
          length_population left,
          int_literal right,
          int_literal left,
          length_population right )
      with
      | Some _, Some population, Some n, _, _
        when (is_name op ">" && n >= 0) || (is_name op ">=" && n > 0) ->
          Set.singleton (module String) population
      | Some _, _, _, Some n, Some population
        when (is_name op "<" && n >= 0) || (is_name op "<=" && n > 0) ->
          Set.singleton (module String) population
      | _ -> none ())
  | Pexp_ifthenelse (condition, yes, Some no) when bool_literal no false ->
      Set.union (required_nonempty condition) (required_nonempty yes)
  | _ -> none ()

(* Quantifiers that contribute to the value a helper RETURNS, rather than to setup or validation it
   performs on the way. [nonzero] helpers are the important near miss: they use [not (exists ...)]
   only to decide whether to raise, then return the input array. Treating every expression in their
   body as the helper's boolean made every later parity claim look helper-wrapped. *)
let rec returned_quantifiers ?(positive = true) expr =
  match expr.pexp_desc with
  | Pexp_function (_, _, Pfunction_body body) -> returned_quantifiers ~positive body
  | Pexp_function (_, _, Pfunction_cases (cases, _, _)) ->
      List.concat_map cases ~f:(fun case ->
          let guard_quantifiers =
            Option.value_map case.pc_guard ~default:[] ~f:(fun guard ->
                result_polarities positive case.pc_rhs
                |> List.concat_map ~f:(fun guard_positive ->
                    quantifiers_in ~positive:guard_positive guard))
          in
          guard_quantifiers @ returned_quantifiers ~positive case.pc_rhs)
  | Pexp_let (_, bindings, body) ->
      let returned_bindings = returned_binding_polarities positive body in
      (unguarded_component ~positive body
      |> List.map ~f:(fun quantifier -> { quantifier with sealed = true }))
      @ List.concat_map bindings ~f:(fun binding ->
          binding_parts binding.pvb_pat binding.pvb_expr
          |> List.concat_map ~f:(fun part ->
              List.filter_map returned_bindings ~f:(fun (name, returned_positive) ->
                  if String.equal name part.name then
                    Some
                      ((if part.exact then
                          unguarded_component ~positive:returned_positive part.expression
                        else
                          let guards = required_nonempty part.expression in
                          quantifiers_in ~positive:returned_positive part.expression
                          |> List.filter ~f:(fun quantifier ->
                              Set.is_empty quantifier.populations
                              || Set.is_empty (Set.inter guards quantifier.populations)))
                      |> List.map ~f:(fun quantifier -> { quantifier with sealed = true }))
                  else None)
              |> List.concat))
  | Pexp_sequence (_, result) -> returned_quantifiers ~positive result
  | Pexp_letmodule (_, _, body) | Pexp_letexception (_, body) -> returned_quantifiers ~positive body
  | Pexp_open (_, body) -> returned_quantifiers ~positive body
  | Pexp_constraint (inner, _) | Pexp_coerce (inner, _, _) -> returned_quantifiers ~positive inner
  | Pexp_tuple expressions -> List.concat_map expressions ~f:(unguarded_component ~positive)
  | Pexp_record (fields, base) ->
      List.concat_map fields ~f:(fun (_, expression) -> unguarded_component ~positive expression)
      @ Option.value_map base ~default:[] ~f:(unguarded_component ~positive)
  | Pexp_ifthenelse (condition, yes, no) ->
      let condition_quantifiers =
        condition_polarities positive yes no
        |> List.concat_map ~f:(fun condition_positive ->
            quantifiers_in ~positive:condition_positive condition)
      in
      condition_quantifiers
      @ returned_quantifiers ~positive yes
      @ Option.value_map no ~default:[] ~f:(returned_quantifiers ~positive)
  | Pexp_match (scrutinee, cases) ->
      let scrutinee_quantifiers =
        Option.value_map (boolean_match_polarity cases) ~default:[] ~f:(fun same_polarity ->
            quantifiers_in ~positive:(Bool.equal positive same_polarity) scrutinee)
      in
      scrutinee_quantifiers
      @ List.concat_map cases ~f:(fun case ->
          let returned_bindings = returned_binding_polarities positive case.pc_rhs in
          let guard_quantifiers =
            Option.value_map case.pc_guard ~default:[] ~f:(fun guard ->
                result_polarities positive case.pc_rhs
                |> List.concat_map ~f:(fun guard_positive ->
                    quantifiers_in ~positive:guard_positive guard))
          in
          guard_quantifiers
          @ (returned_quantifiers ~positive case.pc_rhs
            |> List.map ~f:(fun quantifier -> { quantifier with sealed = true }))
          @ (binding_parts case.pc_lhs scrutinee
            |> List.concat_map ~f:(fun part ->
                List.filter_map returned_bindings ~f:(fun (name, returned_positive) ->
                    if String.equal name part.name then
                      Some
                        (if part.exact then
                           returned_quantifiers ~positive:returned_positive part.expression
                         else quantifiers_in ~positive:returned_positive part.expression)
                    else None)
                |> List.concat)))
  | Pexp_try (body, cases) ->
      returned_quantifiers ~positive body
      @ List.concat_map cases ~f:(fun case ->
          let guard_quantifiers =
            Option.value_map case.pc_guard ~default:[] ~f:(fun guard ->
                result_polarities positive case.pc_rhs
                |> List.concat_map ~f:(fun guard_positive ->
                    quantifiers_in ~positive:guard_positive guard))
          in
          guard_quantifiers @ returned_quantifiers ~positive case.pc_rhs)
  | Pexp_apply (callee, [ (Asttypes.Nolabel, left); (Asttypes.Nolabel, right) ])
    when is_name callee "&&" ->
      returned_quantifiers ~positive left @ returned_quantifiers ~positive right
  | Pexp_construct ({ txt = Ppxlib.Longident.Lident "Some"; _ }, Some payload) ->
      returned_quantifiers ~positive payload
  | Pexp_apply (callee, _)
    when is_collection_call callee ~member:"for_all"
         || is_collection_call callee ~member:"for_all2_exn"
         || is_collection_call callee ~member:"is_empty"
         || is_collection_call callee ~member:"exists"
         || is_name callee "not" || is_name callee "&&" || is_name callee "||" || is_name callee "="
         || is_name callee "<>" || is_name callee "equal" || is_name callee "|>"
         || is_name callee "@@" ->
      quantifiers_in ~positive expr
  | Pexp_apply (callee, arguments) when is_transparent_boolean_wrapper callee -> (
      match unlabelled arguments with
      | [ argument ] -> returned_quantifiers ~positive argument
      | _ -> [])
  | _ -> []

and unguarded_component ~positive expression =
  let guards = required_nonempty expression in
  returned_quantifiers ~positive expression
  |> List.filter ~f:(fun quantifier ->
      quantifier.sealed
      || Set.is_empty quantifier.populations
      || Set.is_empty (Set.inter guards quantifier.populations))

and returned_binding_polarities positive expr =
  match expr.pexp_desc with
  | Pexp_ident { txt = Ppxlib.Longident.Lident name; _ } -> [ (name, positive) ]
  | Pexp_function (_, _, Pfunction_body body) -> returned_binding_polarities positive body
  | Pexp_function (_, _, Pfunction_cases (cases, _, _)) ->
      List.concat_map cases ~f:(fun case ->
          let shadowed =
            pattern_names case.pc_lhs |> List.map ~f:fst |> Set.of_list (module String)
          in
          returned_binding_polarities positive case.pc_rhs
          |> List.filter ~f:(fun (name, _) -> not (Set.mem shadowed name)))
  | Pexp_let (_, bindings, body) ->
      let shadowed =
        List.concat_map bindings ~f:(fun binding -> List.map (pattern_names binding.pvb_pat) ~f:fst)
        |> Set.of_list (module String)
      in
      returned_binding_polarities positive body
      |> List.filter ~f:(fun (name, _) -> not (Set.mem shadowed name))
  | Pexp_sequence (_, result) -> returned_binding_polarities positive result
  | Pexp_letmodule (_, _, body) | Pexp_letexception (_, body) ->
      returned_binding_polarities positive body
  | Pexp_open (_, body) -> returned_binding_polarities positive body
  | Pexp_constraint (inner, _) | Pexp_coerce (inner, _, _) ->
      returned_binding_polarities positive inner
  | Pexp_ifthenelse (condition, yes, no) ->
      (condition_polarities positive yes no
      |> List.concat_map ~f:(fun condition_positive ->
          returned_binding_polarities condition_positive condition))
      @ returned_binding_polarities positive yes
      @ Option.value_map no ~default:[] ~f:(returned_binding_polarities positive)
  | Pexp_match (scrutinee, cases) ->
      Option.value_map (boolean_match_polarity cases) ~default:[] ~f:(fun same_polarity ->
          returned_binding_polarities (Bool.equal positive same_polarity) scrutinee)
      @ List.concat_map cases ~f:(fun case ->
          let returned = returned_binding_polarities positive case.pc_rhs in
          let shadowed =
            pattern_names case.pc_lhs |> List.map ~f:fst |> Set.of_list (module String)
          in
          (returned |> List.filter ~f:(fun (name, _) -> not (Set.mem shadowed name)))
          @ (binding_parts case.pc_lhs scrutinee
            |> List.concat_map ~f:(fun part ->
                List.filter_map returned ~f:(fun (name, returned_positive) ->
                    if String.equal name part.name then
                      Some (returned_binding_polarities returned_positive part.expression)
                    else None)
                |> List.concat)))
  | Pexp_try (body, cases) ->
      returned_binding_polarities positive body
      @ List.concat_map cases ~f:(fun case ->
          let shadowed =
            pattern_names case.pc_lhs |> List.map ~f:fst |> Set.of_list (module String)
          in
          returned_binding_polarities positive case.pc_rhs
          |> List.filter ~f:(fun (name, _) -> not (Set.mem shadowed name)))
  | Pexp_apply (callee, [ (Asttypes.Nolabel, argument) ]) when is_name callee "not" ->
      returned_binding_polarities (not positive) argument
  | Pexp_apply (apply, [ (Asttypes.Nolabel, function_); (Asttypes.Nolabel, argument) ])
    when is_name apply "@@" && is_name function_ "not" ->
      returned_binding_polarities (not positive) argument
  | Pexp_apply (pipe, [ (Asttypes.Nolabel, value); (Asttypes.Nolabel, piped_call) ])
    when is_name pipe "|>" -> (
      match piped_call.pexp_desc with
      | Pexp_ident _ when is_name piped_call "not" ->
          returned_binding_polarities (not positive) value
      | Pexp_apply (callee, arguments) -> (
          match unlabelled arguments with
          | [ literal ] -> (
              match compared_argument callee literal value ~positive with
              | Some (argument, polarity) -> returned_binding_polarities polarity argument
              | None -> returned_binding_polarities positive piped_call)
          | _ -> returned_binding_polarities positive piped_call)
      | _ -> returned_binding_polarities positive piped_call)
  | Pexp_apply (callee, [ (Asttypes.Nolabel, left); (Asttypes.Nolabel, right) ])
    when is_boolean_comparison callee -> (
      match compared_argument callee left right ~positive with
      | Some (argument, polarity) -> returned_binding_polarities polarity argument
      | None ->
          returned_binding_polarities positive left @ returned_binding_polarities positive right)
  | Pexp_apply (callee, arguments) when is_name callee "&&" || is_name callee "||" ->
      List.concat_map arguments ~f:(fun (_, argument) ->
          returned_binding_polarities positive argument)
  | Pexp_apply (callee, _) -> (
      match Sources.longident_of callee with Some [ name ] -> [ (name, positive) ] | _ -> [])
  | _ -> []

(* [Verdict] is the only module a claim is reached through: [Ll_test] re-exported six of these names
   while [Verdict] had no open-able surface, and gh-ocannl-815 retired that copy in favour of [open
   Verdict.Claims], which the environment below models. *)
let claim_kind_of_path path =
  match path with
  | "Verdict" :: _ -> (
      match List.last path with
      | Some "p" -> Some P
      | Some "pf" -> Some Pf
      | Some "pass_fail" -> Some Pass_fail
      | Some "claim" -> Some Claim
      | Some "claimf" -> Some Claimf
      | _ -> None)
  | _ -> None

let module_path module_expr =
  match module_expr.pmod_desc with
  | Pmod_ident { txt; _ } -> ( try Some (Ppxlib.Longident.flatten_exn txt) with _ -> None)
  | _ -> None

let opens_verdict_claims module_expr =
  Option.value_map (module_path module_expr) ~default:false
    ~f:(List.equal String.equal [ "Verdict"; "Claims" ])

(* [open Verdict.Claims] is the migration target of gh-ocannl-815. Model its bindings explicitly so
   helper-following does not lose sight of an unqualified [p]/[claim]/[pass_fail] call when the file
   deletes its local aliases. Ordinary value bindings are prepended later and therefore shadow these
   exactly as they do in OCaml. *)
let opened_claim_bindings site =
  [ ("p", P); ("pf", Pf); ("pass_fail", Pass_fail); ("claim", Claim); ("claimf", Claimf) ]
  |> List.map ~f:(fun (name, claim_kind) ->
      {
        name;
        site;
        optional_label = None;
        dependencies = [];
        guards = Set.empty (module String);
        unguarded = [];
        negated_unguarded = [];
        constant_bool = None;
        quantifier_alias = None;
        claim_kind = Some claim_kind;
        claim_wrapper = None;
      })

let open_claims environment declaration =
  let verdict_environment =
    if opens_verdict_claims declaration.popen_expr then
      let start = declaration.popen_loc.loc_start in
      List.rev_append
        (opened_claim_bindings
           {
             line = start.Stdlib.Lexing.pos_lnum;
             column = start.Stdlib.Lexing.pos_cnum - start.Stdlib.Lexing.pos_bol;
             position = start.Stdlib.Lexing.pos_cnum;
           })
        environment
    else environment
  in
  match module_path declaration.popen_expr with
  | None -> verdict_environment
  | Some path ->
      let prefix = String.concat ~sep:"." path ^ "." in
      let opened =
        List.filter_map environment ~f:(fun (binding : helper_binding) ->
            String.chop_prefix binding.name ~prefix
            |> Option.map ~f:(fun name -> { binding with name }))
      in
      List.rev_append opened verdict_environment

let lookup (environment : helper_binding list) name =
  List.find environment ~f:(fun binding -> String.equal binding.name name)

let claim_target environment callee =
  match Sources.longident_of callee with
  | Some path -> (
      match lookup environment (String.concat ~sep:"." path) with
      | Some binding -> Option.map binding.claim_kind ~f:(fun kind -> (kind, Some binding))
      | None -> Option.map (claim_kind_of_path path) ~f:(fun kind -> (kind, None)))
  | None -> None

let constant_bool_of environment expr =
  match literal_bool expr with
  | Some _ as value -> value
  | None -> (
      match Sources.longident_of expr with
      | Some [ name ] ->
          lookup environment name |> Option.bind ~f:(fun binding -> binding.constant_bool)
      | _ -> None)

let resolved_bool = constant_bool_of

let quantifier_alias_of environment expr =
  match collection_quantifier expr with
  | Some _ as quantifier -> quantifier
  | None -> (
      match Sources.longident_of expr with
      | Some path ->
          lookup environment (String.concat ~sep:"." path)
          |> Option.bind ~f:(fun binding -> binding.quantifier_alias)
      | None -> None)

let resolved_compared_argument environment callee left right ~positive =
  match compared_argument callee left right ~positive with
  | Some _ as found -> found
  | None when is_boolean_comparison callee -> (
      let equal = not (is_name callee "<>") in
      let with_constant constant argument =
        Some (argument, Bool.equal positive (Bool.equal constant equal))
      in
      match (resolved_bool environment left, resolved_bool environment right) with
      | Some constant, None -> with_constant constant right
      | None, Some constant -> with_constant constant left
      | Some _, Some _ | None, None -> None)
  | None -> None

let resolved_boolean_match_polarity environment cases =
  let output_for input =
    List.find cases ~f:(fun case -> bool_pattern_matches case.pc_lhs input)
    |> Option.bind ~f:(fun case ->
        if Option.is_none case.pc_guard then resolved_bool environment case.pc_rhs else None)
  in
  match (output_for true, output_for false) with
  | Some true, Some false -> Some true
  | Some false, Some true -> Some false
  | _ -> None

let resolved_result_polarities environment positive result =
  match resolved_bool environment result with
  | Some true -> [ positive ]
  | Some false -> [ not positive ]
  | None -> []

let dependency_result_polarities environment positive result =
  match resolved_bool environment result with
  | Some true -> [ positive ]
  | Some false -> [ not positive ]
  | None -> result_polarities positive result

let resolved_condition_polarities environment positive yes no =
  match
    Option.bind no ~f:(fun no ->
        Option.both (resolved_bool environment yes) (resolved_bool environment no))
  with
  | Some (true, false) -> [ positive ]
  | Some (false, true) -> [ not positive ]
  | Some (true, true | false, false) | None -> []

let dependency_condition_polarities environment positive yes no =
  match
    Option.bind no ~f:(fun no ->
        Option.both (resolved_bool environment yes) (resolved_bool environment no))
  with
  | Some (true, false) -> [ positive ]
  | Some (false, true) -> [ not positive ]
  | Some (true, true | false, false) -> []
  | None -> [ positive ]

let shadow_parameters environment parameters =
  List.fold parameters ~init:environment ~f:(fun environment parameter ->
      match parameter.pparam_desc with
      | Pparam_val (_, _, pattern) ->
          let shadowed = pattern_names pattern |> List.map ~f:fst |> Set.of_list (module String) in
          List.filter environment ~f:(fun (binding : helper_binding) ->
              not (Set.mem shadowed binding.name))
      | Pparam_newtype _ -> environment)

let constant_binding environment part =
  {
    name = part.name;
    site = site_of_location part.location;
    optional_label = None;
    dependencies = [];
    guards = Set.empty (module String);
    unguarded = [];
    negated_unguarded = [];
    constant_bool = constant_bool_of environment part.expression;
    quantifier_alias = quantifier_alias_of environment part.expression;
    claim_kind = None;
    claim_wrapper = None;
  }

let rec alias_quantifiers environment ?(positive = true) expr =
  match expr.pexp_desc with
  | Pexp_function (parameters, _, Pfunction_body body) ->
      alias_quantifiers (shadow_parameters environment parameters) ~positive body
  | Pexp_function (parameters, _, Pfunction_cases (cases, _, _)) ->
      let environment = shadow_parameters environment parameters in
      List.concat_map cases ~f:(fun case ->
          let shadowed =
            pattern_names case.pc_lhs |> List.map ~f:fst |> Set.of_list (module String)
          in
          let case_environment =
            List.filter environment ~f:(fun (binding : helper_binding) ->
                not (Set.mem shadowed binding.name))
          in
          let guards =
            Option.value_map case.pc_guard ~default:[] ~f:(fun guard ->
                resolved_result_polarities case_environment positive case.pc_rhs
                |> List.concat_map ~f:(fun guard_positive ->
                    quantifiers_in ~positive:guard_positive guard))
          in
          guards @ alias_quantifiers case_environment ~positive case.pc_rhs)
  | Pexp_let (_, bindings, body) ->
      let local =
        List.concat_map bindings ~f:(fun binding ->
            binding_parts binding.pvb_pat binding.pvb_expr
            |> List.map ~f:(constant_binding environment))
      in
      let body_environment = List.rev_append local environment in
      let returned = returned_binding_polarities positive body in
      alias_quantifiers body_environment ~positive body
      @ List.concat_map bindings ~f:(fun binding ->
          binding_parts binding.pvb_pat binding.pvb_expr
          |> List.concat_map ~f:(fun part ->
              List.filter_map returned ~f:(fun (name, returned_positive) ->
                  if String.equal name part.name then
                    Some (alias_quantifiers environment ~positive:returned_positive part.expression)
                  else None)
              |> List.concat))
  | Pexp_sequence (_, result) | Pexp_open (_, result) ->
      alias_quantifiers environment ~positive result
  | Pexp_letmodule (_, _, body) | Pexp_letexception (_, body) ->
      alias_quantifiers environment ~positive body
  | Pexp_constraint (inner, _) | Pexp_coerce (inner, _, _) ->
      alias_quantifiers environment ~positive inner
  | Pexp_tuple expressions ->
      List.concat_map expressions ~f:(alias_quantifiers environment ~positive)
  | Pexp_record (fields, base) ->
      List.concat_map fields ~f:(fun (_, field) -> alias_quantifiers environment ~positive field)
      @ Option.value_map base ~default:[] ~f:(alias_quantifiers environment ~positive)
  | Pexp_ifthenelse (condition, yes, no) ->
      let from_condition =
        resolved_condition_polarities environment positive yes no
        |> List.concat_map ~f:(fun condition_positive ->
            quantifiers_in ~positive:condition_positive condition)
      in
      from_condition
      @ alias_quantifiers environment ~positive yes
      @ Option.value_map no ~default:[] ~f:(alias_quantifiers environment ~positive)
  | Pexp_match (scrutinee, cases) ->
      let scrutinee_quantifiers =
        Option.value_map (resolved_boolean_match_polarity environment cases) ~default:[]
          ~f:(fun same_polarity ->
            quantifiers_in ~positive:(Bool.equal positive same_polarity) scrutinee)
      in
      scrutinee_quantifiers
      @ List.concat_map cases ~f:(fun case ->
          let shadowed =
            pattern_names case.pc_lhs |> List.map ~f:fst |> Set.of_list (module String)
          in
          let case_environment =
            List.filter environment ~f:(fun (binding : helper_binding) ->
                not (Set.mem shadowed binding.name))
          in
          let guards =
            Option.value_map case.pc_guard ~default:[] ~f:(fun guard ->
                resolved_result_polarities case_environment positive case.pc_rhs
                |> List.concat_map ~f:(fun guard_positive ->
                    quantifiers_in ~positive:guard_positive guard))
          in
          guards @ alias_quantifiers case_environment ~positive case.pc_rhs)
  | Pexp_try (body, cases) ->
      alias_quantifiers environment ~positive body
      @ List.concat_map cases ~f:(fun case ->
          let shadowed =
            pattern_names case.pc_lhs |> List.map ~f:fst |> Set.of_list (module String)
          in
          let case_environment =
            List.filter environment ~f:(fun (binding : helper_binding) ->
                not (Set.mem shadowed binding.name))
          in
          let guards =
            Option.value_map case.pc_guard ~default:[] ~f:(fun guard ->
                resolved_result_polarities case_environment positive case.pc_rhs
                |> List.concat_map ~f:(fun guard_positive ->
                    quantifiers_in ~positive:guard_positive guard))
          in
          guards @ alias_quantifiers case_environment ~positive case.pc_rhs)
  | Pexp_apply (callee, [ (Asttypes.Nolabel, left); (Asttypes.Nolabel, right) ])
    when is_boolean_comparison callee -> (
      match resolved_compared_argument environment callee left right ~positive with
      | Some (argument, argument_positive) -> quantifiers_in ~positive:argument_positive argument
      | None -> [])
  | Pexp_apply (({ pexp_desc = Pexp_function _; _ } as function_), _) ->
      alias_quantifiers environment ~positive function_
  | Pexp_apply (callee, arguments) -> (
      match quantifier_alias_of environment callee with
      | Some (kind, count)
        when (positive && not (Poly.equal kind Not_exists))
             || ((not positive) && Poly.equal kind Not_exists) ->
          [ { kind; populations = populations arguments count; sealed = false } ]
      | Some _ | None -> [])
  | _ -> []

let argument_at_slot arguments slot =
  match (slot.label, slot.unlabelled_index) with
  | Some name, _ ->
      List.find_map arguments ~f:(fun (label, argument) ->
          match label with
          | Asttypes.Labelled found when String.equal found name -> Some argument
          | Optional found when slot.optional && String.equal found name -> (
              match argument.pexp_desc with
              | Pexp_construct ({ txt = Ppxlib.Longident.Lident "Some"; _ }, Some payload) ->
                  Some payload
              | Pexp_construct ({ txt = Ppxlib.Longident.Lident "None"; _ }, None) -> None
              | _ -> Some argument)
          | _ -> None)
  | None, Some index -> List.nth (unlabelled arguments) index
  | None, None -> None

let slot_definitely_supplied arguments slot =
  match slot.label with
  | None -> Option.is_some (argument_at_slot arguments slot)
  | Some name ->
      List.exists arguments ~f:(fun (label, argument) ->
          match label with
          | Asttypes.Labelled found -> String.equal found name
          | Optional found when slot.optional && String.equal found name -> (
              match argument.pexp_desc with
              | Pexp_construct ({ txt = Ppxlib.Longident.Lident "Some"; _ }, Some _) -> true
              | _ -> false)
          | _ -> false)

let claim_arguments target arguments =
  match target with
  | _, Some { claim_wrapper = Some slots; _ } ->
      List.filter_map slots ~f:(fun slot ->
          Option.map (argument_at_slot arguments slot) ~f:(fun argument ->
              (argument, slot.positive)))
  | _ ->
      Option.to_list (List.last (unlabelled arguments))
      |> List.map ~f:(fun argument -> (argument, true))

let claim_default_bindings target arguments =
  match target with
  | _, Some { claim_wrapper = Some slots; _ } ->
      List.filter_map slots ~f:(fun slot ->
          if slot_definitely_supplied arguments slot then None
          else Option.map slot.default_binding ~f:(fun binding -> (binding, slot.positive)))
  | _ -> []

let rec make_bindings environment value =
  binding_parts value.pvb_pat value.pvb_expr |> List.map ~f:(make_binding_part environment)

and make_binding_part ?optional_label environment part =
  let guards = required_nonempty part.expression in
  let returned =
    returned_quantifiers part.expression @ alias_quantifiers environment part.expression
  in
  let unguarded =
    List.filter returned ~f:(fun quantifier ->
        quantifier.sealed
        || Set.is_empty quantifier.populations
        || Set.is_empty (Set.inter guards quantifier.populations))
  in
  let negated_unguarded =
    returned_quantifiers ~positive:false part.expression
    @ alias_quantifiers environment ~positive:false part.expression
  in
  let constant_bool = constant_bool_of environment part.expression in
  let dependencies =
    function_dependencies environment part.expression
    |> List.filter ~f:(fun dependency ->
        (* A returned local quantifier is already attributed to this binding by
           [returned_quantifiers]. Keep outer dependencies beside it, but not the local definition
           of the same quantifier: reporting both would make one semantic hole need two exemptions.
           With no direct hole, local dependencies remain the path that carries intermediate
           negation and guards to an outer binding. *)
        (List.is_empty unguarded && List.is_empty negated_unguarded)
        || List.exists environment ~f:(fun (outer : helper_binding) ->
            outer.site.position = dependency.binding.site.position))
  in
  let claim_kind, claim_wrapper =
    match Sources.longident_of part.expression with
    | Some [ alias ] ->
        lookup environment alias
        |> Option.value_map ~default:(None, None) ~f:(fun binding ->
            (binding.claim_kind, binding.claim_wrapper))
    | Some path -> (claim_kind_of_path path, None)
    | None -> Option.value (wrapper_signature environment part.expression) ~default:(None, None)
  in
  let start = part.location.loc_start in
  let site =
    {
      line = start.Stdlib.Lexing.pos_lnum;
      column = start.Stdlib.Lexing.pos_cnum - start.Stdlib.Lexing.pos_bol;
      position = start.Stdlib.Lexing.pos_cnum;
    }
  in
  {
    name = part.name;
    site;
    optional_label;
    dependencies;
    guards;
    unguarded;
    negated_unguarded;
    constant_bool;
    quantifier_alias = quantifier_alias_of environment part.expression;
    claim_kind;
    claim_wrapper;
  }

and wrapper_signature environment expression =
  let rec parameters_and_body parameters expression =
    match expression.pexp_desc with
    | Pexp_function (more, _, Pfunction_body body) -> parameters_and_body (parameters @ more) body
    | _ -> (parameters, expression)
  in
  let resolve_alias aliases (name, positive) =
    match List.Assoc.find aliases name ~equal:String.equal with
    | None -> [ (name, positive) ]
    | Some targets ->
        List.map targets ~f:(fun (target, target_positive) ->
            (target, Bool.equal positive target_positive))
  in
  let extend_aliases aliases recursive bindings =
    let parts =
      List.concat_map bindings ~f:(fun binding -> binding_parts binding.pvb_pat binding.pvb_expr)
    in
    let shadowed = List.map parts ~f:(fun part -> part.name) |> Set.of_list (module String) in
    let outer_aliases = List.filter aliases ~f:(fun (name, _) -> not (Set.mem shadowed name)) in
    let local_aliases =
      match recursive with
      | Asttypes.Nonrecursive ->
          List.map parts ~f:(fun part ->
              let targets =
                returned_binding_polarities true part.expression
                |> List.concat_map ~f:(resolve_alias aliases)
              in
              (part.name, targets))
      | Recursive -> List.map parts ~f:(fun part -> (part.name, []))
    in
    List.rev_append local_aliases outer_aliases
  in
  let shadow_aliases aliases pattern =
    let shadowed = pattern_names pattern |> List.map ~f:fst |> Set.of_list (module String) in
    let outer = List.filter aliases ~f:(fun (name, _) -> not (Set.mem shadowed name)) in
    Set.to_list shadowed |> List.map ~f:(fun name -> (name, [])) |> Fn.flip List.rev_append outer
  in
  let rec claim_calls claim_environment aliases expression =
    match expression.pexp_desc with
    | Pexp_let (recursive, bindings, body) ->
        let binding_aliases =
          match recursive with
          | Asttypes.Nonrecursive -> aliases
          | Recursive -> extend_aliases aliases recursive bindings
        in
        List.concat_map bindings ~f:(fun binding ->
            claim_calls claim_environment binding_aliases binding.pvb_expr)
        @ claim_calls claim_environment (extend_aliases aliases recursive bindings) body
    | Pexp_sequence (left, right) ->
        claim_calls claim_environment aliases left @ claim_calls claim_environment aliases right
    | Pexp_letmodule (_, _, body) | Pexp_letexception (_, body) ->
        claim_calls claim_environment aliases body
    | Pexp_open (declaration, body) ->
        claim_calls (open_claims claim_environment declaration) aliases body
    | Pexp_constraint (inner, _) | Pexp_coerce (inner, _, _) ->
        claim_calls claim_environment aliases inner
    | Pexp_ifthenelse (_, yes, no) ->
        claim_calls claim_environment aliases yes
        @ Option.value_map no ~default:[] ~f:(claim_calls claim_environment aliases)
    | Pexp_match (_, cases) ->
        List.concat_map cases ~f:(fun case ->
            claim_calls claim_environment (shadow_aliases aliases case.pc_lhs) case.pc_rhs)
    | Pexp_try (body, cases) ->
        claim_calls claim_environment aliases body
        @ List.concat_map cases ~f:(fun case ->
            claim_calls claim_environment (shadow_aliases aliases case.pc_lhs) case.pc_rhs)
    | Pexp_function (parameters, _, Pfunction_body body) ->
        let aliases =
          List.fold parameters ~init:aliases ~f:(fun aliases parameter ->
              match parameter.pparam_desc with
              | Pparam_val (_, _, pattern) -> shadow_aliases aliases pattern
              | Pparam_newtype _ -> aliases)
        in
        claim_calls claim_environment aliases body
    | Pexp_function (parameters, _, Pfunction_cases (cases, _, _)) ->
        let aliases =
          List.fold parameters ~init:aliases ~f:(fun aliases parameter ->
              match parameter.pparam_desc with
              | Pparam_val (_, _, pattern) -> shadow_aliases aliases pattern
              | Pparam_newtype _ -> aliases)
        in
        List.concat_map cases ~f:(fun case ->
            claim_calls claim_environment (shadow_aliases aliases case.pc_lhs) case.pc_rhs)
    | Pexp_apply (callee, arguments) -> (
        match claim_target claim_environment callee with
        | Some target -> [ (target, arguments, aliases) ]
        | None ->
            claim_calls claim_environment aliases callee
            @ List.concat_map arguments ~f:(fun (_, argument) ->
                claim_calls claim_environment aliases argument))
    | _ -> []
  in
  let parameters, body = parameters_and_body [] expression in
  let function_cases =
    match body.pexp_desc with
    | Pexp_function (_, _, Pfunction_cases (cases, _, _)) -> cases
    | _ -> []
  in
  let calls =
    if List.is_empty function_cases then claim_calls environment [] body
    else List.concat_map function_cases ~f:(fun case -> claim_calls environment [] case.pc_rhs)
  in
  let unlabelled_parameters =
    List.count parameters ~f:(fun parameter ->
        match parameter.pparam_desc with
        | Pparam_val (Asttypes.Nolabel, _, _) -> true
        | Pparam_val ((Labelled _ | Optional _), _, _) | Pparam_newtype _ -> false)
  in
  let partial_claim_slot ((kind, wrapper), arguments, _) =
    let unlabelled_arguments = unlabelled arguments in
    let pending_format_arguments =
      match (kind, unlabelled_arguments) with
      | (Pf | Claimf), format :: supplied ->
          Option.map (Sources.string_literal format) ~f:(fun format ->
              let expected =
                Scan.directives format
                |> List.sum
                     (module Int)
                     ~f:(fun (directive : Scan.directive) ->
                       if Scan.consumes_nothing directive.conversion then 0
                       else
                         let modifiers =
                           String.sub format ~pos:(directive.start + 1)
                             ~len:(directive.stop - directive.start - 1)
                         in
                         1
                         + String.count modifiers ~f:(Char.equal '*')
                         + if Char.equal directive.conversion 'a' then 1 else 0)
              in
              if List.length supplied <= expected then Some (expected - List.length supplied)
              else None)
          |> Option.join
      | (P | Pass_fail | Claim), [ _ ] -> Some 0
      | (P | Pass_fail | Claim), _ -> None
      | (Pf | Claimf), [] -> None
    in
    match (wrapper, pending_format_arguments) with
    | Some { claim_wrapper = Some _; _ }, _ | _, None -> None
    | _, Some missing ->
        Some
          {
            label = None;
            optional = false;
            unlabelled_index = Some (unlabelled_parameters + missing);
            positive = true;
            default_binding = None;
          }
  in
  let forwarded_wrapper_slots ((_, wrapper), arguments, _) =
    match wrapper with
    | Some { claim_wrapper = Some slots; _ } ->
        let supplied_unlabelled = List.length (unlabelled arguments) in
        List.filter_map slots ~f:(fun slot ->
            match (slot.label, slot.unlabelled_index) with
            | None, Some index when index >= supplied_unlabelled ->
                Some
                  {
                    slot with
                    unlabelled_index = Some (unlabelled_parameters + index - supplied_unlabelled);
                  }
            | Some label, _ ->
                let supplied =
                  List.exists arguments ~f:(fun (argument_label, _) ->
                      match argument_label with
                      | Asttypes.Labelled found | Optional found -> String.equal found label
                      | Nolabel -> false)
                in
                if supplied then None else Some slot
            | None, Some _ | None, None -> None)
    | Some { claim_wrapper = None; _ } | None -> []
  in
  match calls with
  | [] -> None
  | ((kind, _), _, _) :: _ as calls ->
      let partial_claim_slots =
        List.filter_map calls ~f:partial_claim_slot
        @ List.concat_map calls ~f:forwarded_wrapper_slots
      in
      let claimed_names =
        List.concat_map calls ~f:(fun (target, arguments, aliases) ->
            claim_arguments target arguments
            |> List.concat_map ~f:(fun (argument, positive) ->
                returned_binding_polarities positive argument
                |> List.concat_map ~f:(resolve_alias aliases)))
      in
      let _, slots, _ =
        List.fold parameters ~init:(0, [], environment)
          ~f:(fun (unlabelled_index, slots, parameter_environment) parameter ->
            match parameter.pparam_desc with
            | Pparam_newtype _ -> (unlabelled_index, slots, parameter_environment)
            | Pparam_val (label, default, pattern) ->
                let names = pattern_names pattern |> List.map ~f:fst in
                let shadowed = Set.of_list (module String) names in
                let outer_environment =
                  List.filter parameter_environment ~f:(fun (binding : helper_binding) ->
                      not (Set.mem shadowed binding.name))
                in
                let optional_label =
                  match label with
                  | Asttypes.Optional name | Labelled name -> Some name
                  | Nolabel -> None
                in
                let default_bindings =
                  Option.value_map default ~default:[] ~f:(fun default ->
                      binding_parts pattern default
                      |> List.map ~f:(make_binding_part ?optional_label parameter_environment))
                in
                let matches =
                  List.filter_map claimed_names ~f:(fun (name, positive) ->
                      if List.mem names name ~equal:String.equal then Some (name, positive)
                      else None)
                in
                let slot =
                  match label with
                  | Asttypes.Nolabel ->
                      {
                        label = None;
                        optional = false;
                        unlabelled_index = Some unlabelled_index;
                        positive = true;
                        default_binding = None;
                      }
                  | Labelled name ->
                      {
                        label = Some name;
                        optional = false;
                        unlabelled_index = None;
                        positive = true;
                        default_binding = None;
                      }
                  | Optional name ->
                      {
                        label = Some name;
                        optional = true;
                        unlabelled_index = None;
                        positive = true;
                        default_binding = None;
                      }
                in
                let slots =
                  List.rev_append
                    (List.map matches ~f:(fun (name, positive) ->
                         {
                           slot with
                           positive;
                           default_binding =
                             List.find default_bindings ~f:(fun binding ->
                                 String.equal binding.name name);
                         }))
                    slots
                in
                let unlabelled_index =
                  match label with Nolabel -> unlabelled_index + 1 | _ -> unlabelled_index
                in
                (unlabelled_index, slots, List.rev_append default_bindings outer_environment))
      in
      let case_slots =
        List.filter_map function_cases ~f:(fun case ->
            let names = pattern_names case.pc_lhs |> List.map ~f:fst in
            List.find_map claimed_names ~f:(fun (name, positive) ->
                if List.mem names name ~equal:String.equal then
                  Some
                    {
                      label = None;
                      optional = false;
                      unlabelled_index = Some unlabelled_parameters;
                      positive;
                      default_binding = None;
                    }
                else None))
      in
      Some (Some kind, Some (List.rev_append slots (case_slots @ partial_claim_slots)))

and make_binding_group environment recursive values =
  match recursive with
  | Asttypes.Nonrecursive -> List.concat_map values ~f:(make_bindings environment)
  | Recursive ->
      (* Recursive siblings are simultaneously in scope. Recompute the finite group once per bound
         name so a dependency can cross the longest possible sibling chain without making the
         surrounding lexical environment recursive too. *)
      let iterations =
        List.sum (module Int) values ~f:(fun value -> List.length (pattern_names value.pvb_pat))
        |> Int.max 1
      in
      let rec close remaining siblings =
        if remaining = 0 then siblings
        else
          let recursive_environment = List.rev_append siblings environment in
          close (remaining - 1) (List.concat_map values ~f:(make_bindings recursive_environment))
      in
      close iterations []

and function_dependencies environment expr =
  match expr.pexp_desc with
  | Pexp_function (parameters, _, Pfunction_body body) ->
      let body_environment = function_parameter_environment environment parameters in
      function_dependencies body_environment body
  | Pexp_function (parameters, _, Pfunction_cases (cases, _, _)) ->
      let body_environment = function_parameter_environment environment parameters in
      List.concat_map cases ~f:(fun case ->
          let shadowed =
            pattern_names case.pc_lhs |> List.map ~f:fst |> Set.of_list (module String)
          in
          let case_environment =
            List.filter body_environment ~f:(fun (binding : helper_binding) ->
                not (Set.mem shadowed binding.name))
          in
          let guard_dependencies =
            Option.value_map case.pc_guard ~default:[] ~f:(fun guard ->
                dependency_result_polarities case_environment true case.pc_rhs
                |> List.concat_map ~f:(fun positive ->
                    binding_dependencies ~positive case_environment guard))
          in
          guard_dependencies @ function_dependencies case_environment case.pc_rhs)
  | _ -> binding_dependencies environment expr

and function_parameter_environment environment parameters =
  List.fold parameters ~init:environment ~f:(fun parameter_environment parameter ->
      match parameter.pparam_desc with
      | Pparam_val (label, default, pattern) ->
          let names = pattern_names pattern |> List.map ~f:fst in
          let shadowed = Set.of_list (module String) names in
          let outer_environment =
            List.filter parameter_environment ~f:(fun (binding : helper_binding) ->
                not (Set.mem shadowed binding.name))
          in
          Option.value_map default ~default:outer_environment ~f:(fun default ->
              let optional_label =
                match label with
                | Asttypes.Optional name -> Some name
                | Labelled name -> Some name
                | Nolabel -> None
              in
              let defaults =
                binding_parts pattern default
                |> List.map ~f:(make_binding_part ?optional_label parameter_environment)
              in
              List.rev_append defaults outer_environment)
      | Pparam_newtype _ -> parameter_environment)

and binding_dependencies ?(positive = true) environment expr =
  let bindings = ref [] in
  let rec visit environment positive forwards_guards expr =
    let visit_arguments environment positive forwards_guards arguments =
      List.iter arguments ~f:(fun (_, argument) ->
          visit environment positive forwards_guards argument)
    in
    match expr.pexp_desc with
    | Pexp_let (recursive, values, body) ->
        let local = make_binding_group environment recursive values in
        visit (List.rev_append local environment) positive false body
    | Pexp_letmodule (_, _, body) | Pexp_letexception (_, body) ->
        visit environment positive forwards_guards body
    | Pexp_sequence (_, result) -> visit environment positive forwards_guards result
    | Pexp_ifthenelse (condition, yes, no) ->
        dependency_condition_polarities environment positive yes no
        |> List.iter ~f:(fun condition_positive ->
            visit environment condition_positive false condition);
        visit environment positive forwards_guards yes;
        Option.iter no ~f:(visit environment positive forwards_guards)
    | Pexp_match (scrutinee, cases) ->
        let scrutinee_positive =
          Option.value_map (resolved_boolean_match_polarity environment cases) ~default:positive
            ~f:(fun same_polarity -> Bool.equal positive same_polarity)
        in
        visit environment scrutinee_positive false scrutinee;
        List.iter cases ~f:(fun case ->
            let shadowed =
              pattern_names case.pc_lhs |> List.map ~f:fst |> Set.of_list (module String)
            in
            let case_environment =
              List.filter environment ~f:(fun (binding : helper_binding) ->
                  not (Set.mem shadowed binding.name))
            in
            Option.iter case.pc_guard ~f:(fun guard ->
                dependency_result_polarities case_environment positive case.pc_rhs
                |> List.iter ~f:(fun guard_positive ->
                    visit case_environment guard_positive false guard));
            visit case_environment positive forwards_guards case.pc_rhs)
    | Pexp_try (body, cases) ->
        visit environment positive forwards_guards body;
        List.iter cases ~f:(fun case ->
            let shadowed =
              pattern_names case.pc_lhs |> List.map ~f:fst |> Set.of_list (module String)
            in
            let case_environment =
              List.filter environment ~f:(fun (binding : helper_binding) ->
                  not (Set.mem shadowed binding.name))
            in
            Option.iter case.pc_guard ~f:(fun guard ->
                dependency_result_polarities case_environment positive case.pc_rhs
                |> List.iter ~f:(fun guard_positive ->
                    visit case_environment guard_positive false guard));
            visit case_environment positive forwards_guards case.pc_rhs)
    | Pexp_apply (callee, [ (Asttypes.Nolabel, argument) ]) when is_name callee "not" ->
        visit environment (not positive) forwards_guards argument
    | Pexp_apply (apply, [ (Asttypes.Nolabel, function_); (Asttypes.Nolabel, argument) ])
      when is_name apply "@@" && is_name function_ "not" ->
        visit environment (not positive) forwards_guards argument
    | Pexp_apply (pipe, [ (Asttypes.Nolabel, value); (Asttypes.Nolabel, piped_call) ])
      when is_name pipe "|>" -> (
        match piped_call.pexp_desc with
        | Pexp_ident _ when is_name piped_call "not" ->
            visit environment (not positive) forwards_guards value
        | Pexp_apply (callee, arguments) -> (
            match unlabelled arguments with
            | [ literal ] -> (
                match compared_argument callee literal value ~positive with
                | Some (argument, polarity) -> visit environment polarity forwards_guards argument
                | None ->
                    visit_arguments environment positive false
                      [ (Asttypes.Nolabel, value); (Asttypes.Nolabel, piped_call) ])
            | _ ->
                visit_arguments environment positive false
                  [ (Asttypes.Nolabel, value); (Asttypes.Nolabel, piped_call) ])
        | _ ->
            visit_arguments environment positive false
              [ (Asttypes.Nolabel, value); (Asttypes.Nolabel, piped_call) ])
    | Pexp_apply (callee, [ (Asttypes.Nolabel, left); (Asttypes.Nolabel, right) ])
      when is_boolean_comparison callee -> (
        match resolved_compared_argument environment callee left right ~positive with
        | Some (argument, polarity) -> visit environment polarity forwards_guards argument
        | None ->
            visit_arguments environment positive forwards_guards
              [ (Asttypes.Nolabel, left); (Asttypes.Nolabel, right) ])
    | Pexp_apply (callee, arguments) when is_name callee "&&" || is_name callee "||" ->
        visit_arguments environment positive forwards_guards arguments
    | Pexp_apply
        ( ({ pexp_desc = Pexp_function (parameters, _, Pfunction_body body); _ } as _callee),
          arguments ) ->
        visit (shadow_parameters environment parameters) positive forwards_guards body;
        visit_arguments environment positive false arguments
    | Pexp_apply
        ( ({ pexp_desc = Pexp_function (parameters, _, Pfunction_cases (cases, _, _)); _ } as _callee),
          arguments ) ->
        let function_environment = shadow_parameters environment parameters in
        List.iter cases ~f:(fun case ->
            let shadowed =
              pattern_names case.pc_lhs |> List.map ~f:fst |> Set.of_list (module String)
            in
            let case_environment =
              List.filter function_environment ~f:(fun (binding : helper_binding) ->
                  not (Set.mem shadowed binding.name))
            in
            Option.iter case.pc_guard ~f:(visit case_environment positive false);
            visit case_environment positive forwards_guards case.pc_rhs);
        visit_arguments environment positive false arguments
    | Pexp_apply (callee, arguments) -> (
        match Sources.longident_of callee with
        | Some path -> (
            match lookup environment (String.concat ~sep:"." path) with
            | Some binding ->
                let supplied_optional =
                  List.filter_map arguments ~f:(fun (label, argument) ->
                      match label with
                      | Asttypes.Labelled name -> Some name
                      | Optional name -> (
                          match argument.pexp_desc with
                          | Pexp_construct ({ txt = Ppxlib.Longident.Lident "Some"; _ }, Some _) ->
                              Some name
                          | _ -> None)
                      | Nolabel -> None)
                  |> Set.of_list (module String)
                in
                bindings :=
                  {
                    binding;
                    positive;
                    forwards_guards = false;
                    supplied_optional = Some supplied_optional;
                  }
                  :: !bindings;
                visit_arguments environment positive false arguments
            | None ->
                visit environment positive false callee;
                visit_arguments environment positive false arguments)
        | _ ->
            visit environment positive false callee;
            visit_arguments environment positive false arguments)
    | Pexp_function _ -> ()
    | _ ->
        (match Sources.longident_of expr with
        | Some path ->
            Option.iter
              (lookup environment (String.concat ~sep:"." path))
              ~f:(fun binding ->
                bindings :=
                  { binding; positive; forwards_guards; supplied_optional = None } :: !bindings)
        | _ -> ());
        let iterator =
          object
            inherit Ast_traverse.iter as super
            method! attribute _ = ()
            method! expression child = visit environment positive forwards_guards child
            method children child = super#expression child
          end
        in
        iterator#children expr
  in
  visit environment positive true expr;
  !bindings

let quantified_claims structure =
  let origins (dependencies : helper_dependency list) =
    let rec visit seen inherited_guards positive supplied_optional (binding : helper_binding) =
      let key =
        binding.name ^ ":"
        ^ Int.to_string binding.site.position
        ^ ":" ^ Bool.to_string positive ^ ":"
        ^ String.concat ~sep:"," (Set.to_list supplied_optional)
      in
      if Set.mem seen key then []
      else if Option.value_map binding.optional_label ~default:false ~f:(Set.mem supplied_optional)
      then []
      else
        let seen = Set.add seen key in
        let guards =
          if positive then Set.union inherited_guards binding.guards else inherited_guards
        in
        let candidates = if positive then binding.unguarded else binding.negated_unguarded in
        let uncovered =
          List.filter candidates ~f:(fun quantifier ->
              quantifier.sealed
              || Set.is_empty quantifier.populations
              || Set.is_empty (Set.inter guards quantifier.populations))
        in
        let direct = if List.is_empty uncovered then [] else [ (binding, uncovered) ] in
        direct
        @ List.concat_map binding.dependencies ~f:(fun dependency ->
            let inherited_guards =
              if positive && dependency.forwards_guards then guards else Set.empty (module String)
            in
            let supplied_optional =
              Option.value dependency.supplied_optional ~default:supplied_optional
            in
            visit seen inherited_guards
              (Bool.equal positive dependency.positive)
              supplied_optional dependency.binding)
    in
    List.concat_map dependencies ~f:(fun dependency ->
        let supplied_optional =
          Option.value dependency.supplied_optional ~default:(Set.empty (module String))
        in
        visit
          (Set.empty (module String))
          (Set.empty (module String))
          dependency.positive supplied_optional dependency.binding)
    |> List.dedup_and_sort ~compare:(fun (left, _) (right, _) ->
        Int.compare left.site.position right.site.position)
  in
  let found = ref [] in
  let record_dependencies ~claim_line dependencies =
    dependencies |> origins
    |> List.iter ~f:(fun ((binding : helper_binding), quantifiers) ->
        found :=
          {
            helper = binding.name;
            helper_site = binding.site;
            claim_line;
            quantifiers =
              List.map quantifiers ~f:(fun quantifier -> quantifier.kind)
              |> List.dedup_and_sort ~compare:Poly.compare;
          }
          :: !found)
  in
  let record_origins ~claim_line ~positive environment boolean =
    binding_dependencies ~positive environment boolean |> record_dependencies ~claim_line
  in
  let record_direct_quantifiers ~claim_site ~positive ~environment (wrapper : helper_binding)
      boolean =
    let guards = required_nonempty boolean in
    let quantifiers =
      returned_quantifiers ~positive boolean @ alias_quantifiers environment ~positive boolean
      |> List.filter ~f:(fun quantifier ->
          quantifier.sealed
          || Set.is_empty quantifier.populations
          || Set.is_empty (Set.inter guards quantifier.populations))
    in
    if not (List.is_empty quantifiers) then
      let argument_site = site_of_location boolean.pexp_loc in
      found :=
        {
          helper = wrapper.name;
          helper_site = argument_site;
          claim_line = claim_site.line;
          quantifiers =
            List.map quantifiers ~f:(fun quantifier -> quantifier.kind)
            |> List.dedup_and_sort ~compare:Poly.compare;
        }
        :: !found
  in
  let record_claim environment expr =
    match expr.pexp_desc with
    | Pexp_apply (callee, arguments) -> (
        match claim_target environment callee with
        | None -> ()
        | Some ((_, wrapper) as target) ->
            let claim_site = site_of_location expr.pexp_loc in
            let claim_line = claim_site.line in
            claim_arguments target arguments
            |> List.iter ~f:(fun (boolean, positive) ->
                record_origins ~claim_line ~positive environment boolean;
                Option.iter wrapper ~f:(fun binding ->
                    if Option.is_some binding.claim_wrapper then
                      record_direct_quantifiers ~claim_site ~positive ~environment binding boolean));
            claim_default_bindings target arguments
            |> List.iter ~f:(fun (binding, positive) ->
                record_dependencies ~claim_line
                  [
                    {
                      binding;
                      positive;
                      forwards_guards = false;
                      supplied_optional = Some (Set.empty (module String));
                    };
                  ]))
    | _ -> ()
  in
  let prefix_bindings prefix bindings =
    List.map bindings ~f:(fun (binding : helper_binding) ->
        { binding with name = prefix ^ "." ^ binding.name })
  in
  let rec module_claim_bindings environment module_expr =
    match module_expr.pmod_desc with
    | Pmod_structure items ->
        let _, exports =
          List.fold items ~init:(environment, []) ~f:(fun (environment, exports) item ->
              match item.pstr_desc with
              | Pstr_value (recursive, bindings) ->
                  let local = make_binding_group environment recursive bindings in
                  (List.rev_append local environment, List.rev_append local exports)
              | Pstr_open declaration -> (open_claims environment declaration, exports)
              | Pstr_module binding -> (
                  match binding.pmb_name.txt with
                  | Some name ->
                      let nested =
                        module_claim_bindings environment binding.pmb_expr |> prefix_bindings name
                      in
                      (List.rev_append nested environment, List.rev_append nested exports)
                  | None -> (environment, exports))
              | _ -> (environment, exports))
        in
        exports
    | Pmod_constraint (inner, _) -> module_claim_bindings environment inner
    | _ -> []
  in
  let rec scan_expression environment expr =
    record_claim environment expr;
    match expr.pexp_desc with
    | Pexp_let (recursive, bindings, body) ->
        let local = make_binding_group environment recursive bindings in
        let binding_environment =
          match recursive with
          | Asttypes.Nonrecursive -> environment
          | Recursive -> List.rev_append local environment
        in
        List.iter bindings ~f:(fun binding -> scan_expression binding_environment binding.pvb_expr);
        scan_expression (List.rev_append local environment) body
    | Pexp_open (declaration, body) ->
        scan_module environment declaration.popen_expr;
        scan_expression (open_claims environment declaration) body
    | _ ->
        let iterator =
          object
            inherit Ast_traverse.iter as super
            method! attribute _ = ()
            method! expression child = scan_expression environment child
            method! structure nested = scan_structure environment nested
            method children child = super#expression child
          end
        in
        iterator#children expr
  and scan_structure environment items =
    ignore
      (List.fold items ~init:environment ~f:(fun environment item ->
           match item.pstr_desc with
           | Pstr_value (recursive, bindings) ->
               let local = make_binding_group environment recursive bindings in
               let binding_environment =
                 match recursive with
                 | Asttypes.Nonrecursive -> environment
                 | Recursive -> List.rev_append local environment
               in
               List.iter bindings ~f:(fun binding ->
                   scan_expression binding_environment binding.pvb_expr);
               List.rev_append local environment
           | Pstr_eval (expr, _) ->
               scan_expression environment expr;
               environment
           | Pstr_module binding -> (
               scan_module environment binding.pmb_expr;
               match binding.pmb_name.txt with
               | Some name ->
                   module_claim_bindings environment binding.pmb_expr
                   |> prefix_bindings name
                   |> Fn.flip List.rev_append environment
               | None -> environment)
           | Pstr_recmodule bindings ->
               List.iter bindings ~f:(fun binding -> scan_module environment binding.pmb_expr);
               environment
           | Pstr_open declaration ->
               scan_module environment declaration.popen_expr;
               open_claims environment declaration
           | _ ->
               let iterator =
                 object
                   inherit Ast_traverse.iter as super
                   method! attribute _ = ()
                   method! expression expr = scan_expression environment expr
                   method! structure nested = scan_structure environment nested
                   method! structure_item item = super#structure_item item
                 end
               in
               iterator#structure_item item;
               environment))
  and scan_module environment module_expr =
    match module_expr.pmod_desc with
    | Pmod_structure nested -> scan_structure environment nested
    | _ ->
        let iterator =
          object
            inherit Ast_traverse.iter as super
            method! attribute _ = ()
            method! expression expr = scan_expression environment expr
            method! structure nested = scan_structure environment nested
            method! module_expr module_expr = super#module_expr module_expr
          end
        in
        iterator#module_expr module_expr
  in
  scan_structure [] structure;
  List.rev !found
  |> List.dedup_and_sort ~compare:(fun a b ->
      match Int.compare a.claim_line b.claim_line with
      | 0 -> (
          match Int.compare a.helper_site.position b.helper_site.position with
          | 0 -> String.compare a.helper b.helper
          | order -> order)
      | order -> order)

(* Sources whose claim-shaped literals are this check's own input rather than anything printed: the
   table that pins the shape reader on hostile formats, which has to spell the shapes out to pin
   them.

   The file is the honest unit here, not its labels one at a time. That table grows a case whenever
   the reader learns a distinction, so a list of its labels would be a second copy of it, maintained
   by whoever adds the case and read by nobody -- and the labels are fixture words like "fused",
   which say nothing about whether a literal is a print. What keeps the exemption from being a hole
   is the canary list below: two of those literals are named there, and this check fails if its scan
   of this file stops finding them. *)
let data_sources =
  [
    ( "test/operations/verdict_scan_cases.ml",
      "the fixture table for the shape reader: its claim-shaped formats are inputs, compared \
       against the labels they should yield, and never printed" );
  ]

(* Individual claim-shaped literals that are not assertions. Each has to earn its place on every run
   (see the staleness check at the end): an exemption is a claim about a line of code, and a claim
   that stops being true is not a free pass.

   LITERAL-label sites, keyed by "<repository-relative path>:<label>". Empty, and that is the state
   of the tree rather than an oversight: a bare `"<label>: %b"` line with nothing else on it is an
   assertion in every case the sweeps found. *)
let exempt_sites : (string * string) list = []

(* COMPUTED-label sites, keyed by "<repository-relative path>:<the format up to the boolean>". The
   head rather than the label, because a computed label is only what survives rendering a head whose
   arguments this reader cannot fill in -- a hint for a report, not an identity. And the head rather
   than the whole format, because the whole format IS the claim shape: a list of them written out
   here would be a list of claims in a test source, and this check would have to exempt its own file
   to hold everyone else to the rule. A head stops before the boolean, so it is not one.

   Every entry here but the first is a row of a table or a census, where the boolean records what
   happened rather than deciding whether it was right. Each also carries its assertion separately,
   through `Verdict.claim`/`claimf` on the same bound boolean, which is the pattern that lets a row
   keep its shape without losing its gate -- so what is exempted is the PRINT, never the check. An
   entry whose test has no such claim beside it is an exemption that should not have been granted,
   and that is not a hypothetical: `affine_extraction`'s parallelizability table was exempted here
   while nothing claimed it, so a conflict analysis that stopped seeing the reduction's cross-thread
   dependence would have flipped a row to `true`, exited zero, and been promotable (Codex P2, round
   2). All seven entries were audited against the invariant when that one was found.

   The first entry is the structural exception, and it is the only kind there can be: the body of
   the claim printer itself, which is not a row and has no claim beside it because it IS the claim.
   A second entry of that kind would mean a second gate. *)
let exempt_computed_sites =
  [
    ( "test/support/verdict.ml:%s: ",
      "the body of `Verdict.p` itself -- the claim printer every converted site routes THROUGH, \
       which is the one place in the tree where printing `<label>: <bool>` is the gate rather than \
       a way around it" );
    ( "test/operations/affine_extraction.ml:%s %s parallelizable: ",
      "the per-symbol parallelizability table: a reduced axis is legitimately not parallelizable, \
       so `false` is a fact the golden pins rather than a defect" );
    ( "test/operations/bench_args_parsing.ml:%-22s option: ",
      "the argument-classification census, whose whole point is that some strings are options and \
       others are not; the assertion sits beside it as `Verdict.claim (s ^ \" classified as \
       documented\")`" );
    ( "test/operations/reduction_inline_guard.ml:small reduction (K=4): virtual=%b non-virtual=",
      "a tri-state placement row: the pair of booleans is the reading, and each is claimed \
       separately beside it" );
    ( "test/operations/reduction_inline_guard.ml:large reduction (K=64): virtual=%b non-virtual=",
      "the same row for the large reduction" );
    ( "test/operations/reduction_inline_guard.ml:dead large reduction (K=64): virtual=%b \
       non-virtual=",
      "the same row for the dead large reduction" );
    ( "test/operations/test_execution_deps.ml:%s refused, names the routine: %b, names the cause: ",
      "a two-property row about one refusal; both properties are claimed beside it" );
    ( "test/operations/observable_grads.ml:%s placement: %s; in context: %b; observable intent: ",
      "`in context` is legitimately false for a virtualized leg, so the row describes; the \
       assertion is `observable intent`, claimed beside it" );
  ]

(* Literals planted in the fixture file so that this scan has something it MUST find. They are its
   inputs there -- the two spellings whose decoded value is the claim shape -- and they are what
   says the corpus walk is still walking: an empty offender list means "no test prints a bare claim"
   only if the reader that produced it can still see one. A walk that went blind reports the same
   empty list as a clean tree, and these are the difference.

   The second is deliberately spelled over a line continuation. A reader matching text would find
   the first and miss it, which is the failure mode that argues for parsing rather than the one that
   argues for a canary -- both are worth pinning. *)
let canary_sites =
  [
    ( "test/operations/verdict_scan_cases.ml:planted canary",
      "the plain spelling, a fixture input for the shape reader" );
    ( "test/operations/verdict_scan_cases.ml:planted canary over a continuation",
      "the same literal written over a line continuation, which only a reader of decoded values \
       finds" );
  ]

(* Quantified bindings whose passing meaning genuinely ALLOWS an empty population. Keyed by
   [<repository-relative path>:<binding name>], and stale-checked below. The exemption is on the
   binding rather than every claim that uses it: the binding is the unit whose boolean semantics
   decide what empty means, and every use reaches the same decision.

   That last sentence is a precondition, not a fact about names, so it is checked below rather than
   assumed. A name is the unit only while it denotes ONE definition: a file that shadows `refused`
   with a second `refused`, or defines one per local scope, hands both to the same key, and the
   exemption -- granted after reading one body -- would license the other silently, which is exactly
   the shape the reader was widened to catch. Two definitions under one exempted key therefore
   REFUSE the run; the fix is to give them separate names, or to hoist them into one. *)
let exempt_quantified_helpers =
  [
    ( "test/operations/backend_golden_family_scan.ml:complete",
      "empty incomplete/error lists are the passing evidence; the non-empty synthetic-control \
       population is guarded in the same binding" );
    ( "test/operations/autotune_routine_name.ml:contributed",
      "a contended search may legitimately contribute no rows; its report counters separately \
       prove whether that absence came from refused timings rather than a lost result" );
    ( "test/operations/env_var_deps.ml:family_floor_met",
      "a non-repository synthetic run deliberately skips repository-only family floors; the \
       repository path checks every family and reports each shortfall separately" );
    ( "test/operations/epilogue_fusion_mma_seeds.ml:vacuous",
      "an empty GPU mma family deliberately selects the environment-gated vacuity path; the \
       non-vacuous path separately requires and executes the epilogue twins" );
    ( "test/operations/fission_schedule.ml:annotated",
      "the merge-back case deliberately requires the consumer segment to have no hardware axes; \
       the same claim also requires the producer segment to be annotated" );
    ( "test/operations/ocamlformat_ignore_scan.ml:refused",
      "the message list is an optional strengthening of the child-exit refusal: an empty list \
       deliberately means that the nonzero status alone is the passing evidence" );
    ( "test/operations/reduction_forms.ml:changed",
      "an empty schedule deliberately needs no IR change, and an empty per-op result means there \
       were no per-op transformations to validate" );
    ( "test/operations/reduction_forms.ml:extra_ok",
      "an empty extra-fragment list deliberately means the member requires no additional emitted \
       assignment fragments" );
    ( "test/operations/schedule_batched_mma.ml:has_uniform_bf16_tile",
      "an absent or empty hardware mma capability list is the environment gate that makes the \
       backend-uniform non-support legs pass without attempting a tensor-core candidate" );
    ( "test/operations/shell_scripts_parse.ml:line_enables_errexit",
      "a line may legitimately parse to no parent-affecting command fragment; this exists result \
       is an internal classification input, while the fixed non-empty case table is the test \
       population" );
  ]

(* Synthetic inputs state the helper rule independently of whatever helpers happen to be in the
   repository today. The first four are negative controls: the rule must return an offender for
   each, which is the same list the corpus loop below turns into a [Verdict.fail]. The rest are the
   nearest accepted forms, so widening the ratchet until ordinary boolean helpers need exemptions
   also fails here rather than growing a noisy central list. Which scanner mechanism each label
   guards, and the mutation run that proved it, is the table in verdict_ratchet_controls.md. *)
let quantified_helper_controls =
  [
    ( "refuses an unguarded for_all2_exn helper behind a local Verdict alias",
      {ocaml|let p = Verdict.p
let close got want = Array.for_all2_exn got want ~f:Float.equal
let () = p "the values agree" (close got want)|ocaml},
      [ "close" ] );
    ( "refuses an unguarded helper behind an open of Verdict.Claims",
      {ocaml|open Verdict.Claims
let close got want = Array.for_all2_exn got want ~f:Float.equal
let () = p "the values agree" (close got want)|ocaml},
      [ "close" ] );
    ( "keeps an open of Verdict.Claims inside its local scope",
      {ocaml|let close got want = Array.for_all2_exn got want ~f:Float.equal
let guarded () =
  let open Verdict.Claims in
  p "the values agree" (close got want)
let () = p "unrelated local function" true|ocaml},
      [ "close" ] );
    ( "refuses a sibling for_all helper through an intermediate result binding",
      {ocaml|let agrees xs = List.for_all xs ~f:Fn.id
let ok = agrees samples
let () = Verdict.claim "every sample agrees" ok|ocaml},
      [ "agrees" ] );
    ( "refuses a fully applied quantifier bound before the claim",
      {ocaml|let close = Array.for_all2_exn got want ~f:Float.equal
let () = Verdict.p "the values agree" close|ocaml},
      [ "close" ] );
    ( "refuses a quantified binding passed through a Verdict wrapper",
      {ocaml|let print_check name passed = Verdict.pass_fail ("  " ^ name) passed
let all_pass = List.for_all rows ~f:Fn.id
let () = print_check "all rows pass" all_pass|ocaml},
      [ "all_pass" ] );
    ( "refuses a direct quantifier passed through a Verdict wrapper",
      {ocaml|let print_check name passed = Verdict.pass_fail ("  " ^ name) passed
let () = print_check "all rows pass" (List.for_all rows ~f:Fn.id)|ocaml},
      [ "print_check" ] );
    ( "refuses a direct quantifier returned by an immediately invoked function",
      {ocaml|let check ok = Verdict.p "all rows pass" ok
let () = check ((fun () -> List.for_all rows ~f:Fn.id) ())|ocaml},
      [ "check" ] );
    ( "accepts a negated quantifier returned by an immediately invoked function",
      {ocaml|let check ok = Verdict.p "some row fails" ok
let () = check ((fun () -> not (List.for_all rows ~f:Fn.id)) ())|ocaml},
      [] );
    ( "refuses a quantified binding returned by an immediately invoked function",
      {ocaml|let all = List.for_all rows ~f:Fn.id
let check ok = Verdict.p "all rows pass" ok
let () = check ((fun () -> all) ())|ocaml},
      [ "all" ] );
    ( "refuses a direct quantifier called through a function alias",
      {ocaml|let every = List.for_all
let check ok = Verdict.p "all rows pass" ok
let () = check (every rows ~f:Fn.id)|ocaml},
      [ "check" ] );
    ( "accepts a negated quantifier called through a function alias",
      {ocaml|let every = List.for_all
let check ok = Verdict.p "some row fails" ok
let () = check (not (every rows ~f:Fn.id))|ocaml},
      [] );
    ( "accepts a guarded direct quantifier passed through a Verdict wrapper",
      {ocaml|let print_check name passed = Verdict.pass_fail ("  " ^ name) passed
let () =
  print_check "all rows pass"
    ((not (List.is_empty rows)) && List.for_all rows ~f:Fn.id)|ocaml},
      [] );
    ( "accepts a negated direct quantifier passed through a Verdict wrapper",
      {ocaml|let print_check name passed = Verdict.pass_fail ("  " ^ name) passed
let () = print_check "some row fails" (not (List.for_all rows ~f:Fn.id))|ocaml},
      [] );
    ( "refuses a direct exists negated by a labeled Verdict wrapper parameter",
      {ocaml|let check ~ok = Verdict.p "no rows match" (not ok)
let () = check ~ok:(List.exists rows ~f:Fn.id)|ocaml},
      [ "check" ] );
    ( "refuses a bound exists negated by a labeled Verdict wrapper parameter",
      {ocaml|let check ~ok = Verdict.p "no rows match" (not ok)
let some_match = List.exists rows ~f:Fn.id
let () = check ~ok:some_match|ocaml},
      [ "some_match" ] );
    ( "accepts a positive exists passed through a labeled Verdict wrapper parameter",
      {ocaml|let check ~ok = Verdict.p "some row matches" ok
let () = check ~ok:(List.exists rows ~f:Fn.id)|ocaml},
      [] );
    ( "uses an omitted optional default that feeds a Verdict wrapper claim",
      {ocaml|let all = List.for_all rows ~f:Fn.id
let check ?(ok = all) () = Verdict.p "all rows pass" ok
let () = check ()|ocaml},
      [ "all" ] );
    ( "does not use a Verdict wrapper default when its argument is supplied",
      {ocaml|let all = List.for_all rows ~f:Fn.id
let check ?(ok = all) () = Verdict.p "the supplied constant passes" ok
let () = check ~ok:true ()|ocaml},
      [] );
    ( "uses a Verdict wrapper default preserved through partial optional None",
      {ocaml|let check ?(ok = List.for_all rows ~f:Fn.id) () = Verdict.p "all rows pass" ok
let use = check ?ok:None
let () = use ()|ocaml},
      [ "ok" ] );
    ( "inspects the possible payload of an unknown forwarded wrapper option",
      {ocaml|let forwarded = Some (List.for_all rows ~f:Fn.id)
let check ?(ok = false) () = Verdict.p "all rows pass" ok
let () = check ?ok:forwarded ()|ocaml},
      [ "forwarded" ] );
    ( "refuses a direct quantifier passed to a partially applied Verdict claim",
      {ocaml|let check = Verdict.p "all rows pass"
let () = check (List.for_all rows ~f:Fn.id)|ocaml},
      [ "check" ] );
    ( "refuses a direct quantifier passed to a curried partial Verdict wrapper",
      {ocaml|let check label = Verdict.p label
let () = check "all rows pass" (List.for_all rows ~f:Fn.id)|ocaml},
      [ "check" ] );
    ( "refuses a direct quantifier passed to a formatted partial Verdict wrapper",
      {ocaml|let check label = Verdict.pf "%s rows pass" label
let () = check "all" (List.for_all rows ~f:Fn.id)|ocaml},
      [ "check" ] );
    ( "refuses a direct quantifier passed to a partially applied local wrapper",
      {ocaml|let check label ok = Verdict.p label ok
let use = check "all rows pass"
let () = use (List.for_all rows ~f:Fn.id)|ocaml},
      [ "use" ] );
    ( "refuses a direct quantifier passed through a wrapper with tail setup",
      {ocaml|let check ok =
  let label = "all rows pass" in
  Verdict.p label ok
let () = check (List.for_all rows ~f:Fn.id)|ocaml},
      [ "check" ] );
    ( "refuses a direct quantifier passed through a wrapper setup alias",
      {ocaml|let check ok =
  let result = ok in
  Verdict.p "all rows pass" result
let () = check (List.for_all rows ~f:Fn.id)|ocaml},
      [ "check" ] );
    ( "does not connect a wrapper parameter hidden by a setup constant",
      {ocaml|let check ok =
  let result = true in
  Verdict.p "the constant passes" result
let () = check (List.for_all rows ~f:Fn.id)|ocaml},
      [] );
    ( "refuses every quantified argument passed through a sequential wrapper",
      {ocaml|let check first second =
  Verdict.p "first rows pass" first;
  Verdict.p "second rows pass" second
let () = check (List.for_all first_rows ~f:Fn.id) true|ocaml},
      [ "check" ] );
    ( "refuses a quantified argument claimed inside wrapper control flow",
      {ocaml|let enabled = true
let check ok = if enabled then Verdict.p "all rows pass" ok else ()
let () = check (List.for_all rows ~f:Fn.id)|ocaml},
      [ "check" ] );
    ( "refuses a quantified condition used as a wrapper claim value",
      {ocaml|let check ok = Verdict.p "all rows pass" (if ok then true else false)
let () = check (List.for_all rows ~f:Fn.id)|ocaml},
      [ "check" ] );
    ( "accepts an inverted quantified condition used as a wrapper claim value",
      {ocaml|let check ok = Verdict.p "some row fails" (if ok then false else true)
let () = check (List.for_all rows ~f:Fn.id)|ocaml},
      [] );
    ( "refuses a quantified argument claimed inside an eager wrapper call",
      {ocaml|let check ok = ignore (Verdict.p "all rows pass" ok)
let () = check (List.for_all rows ~f:Fn.id)|ocaml},
      [ "check" ] );
    ( "refuses a quantified argument claimed under a local Verdict open",
      {ocaml|let check ok =
  let open Verdict.Claims in
  p "all rows pass" ok
let () = check (List.for_all rows ~f:Fn.id)|ocaml},
      [ "check" ] );
    ( "refuses a quantified argument claimed by a function-case wrapper",
      {ocaml|let check = function ok -> Verdict.p "all rows pass" ok
let () = check (List.for_all rows ~f:Fn.id)|ocaml},
      [ "check" ] );
    ( "refuses a quantified argument forwarded through a match wrapper",
      {ocaml|let check ok = Verdict.p "all rows pass" (match ok with value -> value)
let () = check (List.for_all rows ~f:Fn.id)|ocaml},
      [ "check" ] );
    ( "refuses a quantified argument forwarded by a Boolean match wrapper",
      {ocaml|let check ok =
  Verdict.p "all rows pass" (match ok with true -> true | false -> false)
let () = check (List.for_all rows ~f:Fn.id)|ocaml},
      [ "check" ] );
    ( "accepts a quantified argument inverted by a Boolean match wrapper",
      {ocaml|let check ok =
  Verdict.p "some row fails" (match ok with true -> false | false -> true)
let () = check (List.for_all rows ~f:Fn.id)|ocaml},
      [] );
    ( "refuses a quantified argument claimed inside a callback",
      {ocaml|let check ok =
  List.iter [ () ] ~f:(fun () -> Verdict.p "all rows pass" ok)
let () = check (List.for_all rows ~f:Fn.id)|ocaml},
      [ "check" ] );
    ( "does not connect a callback-shadowed parameter to its wrapper",
      {ocaml|let check ok =
  List.iter [ true ] ~f:(fun ok -> Verdict.p "the constant passes" ok)
let () = check (List.for_all rows ~f:Fn.id)|ocaml},
      [] );
    ( "refuses a quantified argument passed to a qualified local-module wrapper",
      {ocaml|module Checks = struct
  let check ok = Verdict.p "all rows pass" ok
end
let () = Checks.check (List.for_all rows ~f:Fn.id)|ocaml},
      [ "Checks.check" ] );
    ( "refuses a quantified argument passed through an opened local module",
      {ocaml|module Checks = struct
  let check ok = Verdict.p "all rows pass" ok
end
open Checks
let () = check (List.for_all rows ~f:Fn.id)|ocaml},
      [ "check" ] );
    ( "refuses a quantified helper called through a local module",
      {ocaml|module Checks = struct
  let all rows = List.for_all rows ~f:Fn.id
end
let () = Verdict.p "all rows pass" (Checks.all rows)|ocaml},
      [ "Checks.all" ] );
    ( "refuses a quantified helper called through an opened local module",
      {ocaml|module Checks = struct
  let all rows = List.for_all rows ~f:Fn.id
end
open Checks
let () = Verdict.p "all rows pass" (all rows)|ocaml},
      [ "all" ] );
    ( "accepts a fully applied quantified binding with a non-empty witness",
      {ocaml|let close =
  (not (Array.is_empty got)) && Array.for_all2_exn got want ~f:Float.equal
let () = Verdict.p "the values agree" close|ocaml},
      [] );
    ( "accepts a negated fully applied quantified binding",
      {ocaml|let differs = not (Array.for_all2_exn got want ~f:Float.equal)
let () = Verdict.p "some value differs" differs|ocaml},
      [] );
    ( "refuses a fully applied quantifier compared with true",
      {ocaml|let close = Array.for_all2_exn got want ~f:Float.equal |> Bool.equal true
let () = Verdict.p "the values agree" close|ocaml},
      [ "close" ] );
    ( "accepts a fully applied quantifier compared with false",
      {ocaml|let differs = Array.for_all2_exn got want ~f:Float.equal |> Bool.equal false
let () = Verdict.p "some value differs" differs|ocaml},
      [] );
    ( "refuses a direct Bool.equal true around a fully applied quantifier",
      {ocaml|let close = Bool.equal (List.for_all rows ~f:Fn.id) true
let () = Verdict.p "all rows pass" close|ocaml},
      [ "close" ] );
    ( "accepts a direct Bool.equal false around a fully applied quantifier",
      {ocaml|let differs = Bool.equal (List.for_all rows ~f:Fn.id) false
let () = Verdict.p "some row fails" differs|ocaml},
      [] );
    ( "refuses a bound exists compared with a false Boolean alias",
      {ocaml|let some = List.exists rows ~f:Fn.id
let no = false
let result = Bool.equal some no
let () = Verdict.p "no rows match" result|ocaml},
      [ "some" ] );
    ( "refuses a direct exists compared with a false Boolean alias",
      {ocaml|let no = false
let result = Bool.equal (List.exists rows ~f:Fn.id) no
let () = Verdict.p "no rows match" result|ocaml},
      [ "result" ] );
    ( "refuses a fully applied quantifier through a transparent Boolean wrapper",
      {ocaml|let ok = Fn.id (List.for_all rows ~f:Fn.id)
let () = Verdict.p "all rows pass" ok|ocaml},
      [ "ok" ] );
    ( "refuses a returned quantifier behind a local open",
      {ocaml|let result = let open Base in List.for_all rows ~f:Fn.id
let () = Verdict.p "all rows pass" result|ocaml},
      [ "result" ] );
    ( "refuses a returned quantifier behind local module setup",
      {ocaml|let result = let module M = struct end in List.for_all rows ~f:Fn.id
let () = Verdict.p "all rows pass" result|ocaml},
      [ "result" ] );
    ( "refuses a positive intermediate binding",
      {ocaml|let close = List.for_all rows ~f:Fn.id
let still_close = close
let () = Verdict.p "all rows pass" still_close|ocaml},
      [ "close" ] );
    ( "accepts a negated intermediate binding",
      {ocaml|let close = List.for_all rows ~f:Fn.id
let differs = not close
let () = Verdict.p "some row fails" differs|ocaml},
      [] );
    ( "accepts a piped negated intermediate binding",
      {ocaml|let close = List.for_all rows ~f:Fn.id
let differs = close |> not
let () = Verdict.p "some row fails" differs|ocaml},
      [] );
    ( "accepts a directly quantified value piped through not",
      {ocaml|let differs = List.for_all rows ~f:Fn.id |> not
let () = Verdict.p "some row fails" differs|ocaml},
      [] );
    ( "refuses a negated exists written with the application operator",
      {ocaml|let none = not @@ List.exists rows ~f:bad
let () = Verdict.p "no row is bad" none|ocaml},
      [ "none" ] );
    ( "accepts a positive bound exists",
      {ocaml|let some_bad = List.exists rows ~f:bad
let () = Verdict.p "some row is bad" some_bad|ocaml},
      [] );
    ( "refuses a negated bound exists",
      {ocaml|let some_bad = List.exists rows ~f:bad
let () = Verdict.p "no row is bad" (not some_bad)|ocaml},
      [ "some_bad" ] );
    ( "accepts a guarded intermediate binding",
      {ocaml|let close = List.for_all rows ~f:Fn.id
let guarded = (not (List.is_empty rows)) && close
let () = Verdict.p "all rows pass" guarded|ocaml},
      [] );
    ( "accepts a quantifier guarded by the same filtered population",
      {ocaml|let selected = List.filter rows ~f:eligible
let close = (not (List.is_empty selected)) && List.for_all selected ~f:Fn.id
let () = Verdict.p "all selected rows pass" close|ocaml},
      [] );
    ( "refuses a guard on a differently filtered population",
      {ocaml|let close =
  (not (List.is_empty (List.filter rows ~f:p1)))
  && List.for_all (List.filter rows ~f:p2) ~f:q
let () = Verdict.p "all selected rows pass" close|ocaml},
      [ "close" ] );
    ( "conservatively refuses an outer guard across a helper call",
      {ocaml|let all xs = List.for_all xs ~f:Fn.id
let checked xs = (not (List.is_empty xs)) && all xs
let () = Verdict.p "all rows pass" (checked rows)|ocaml},
      [ "all" ] );
    ( "refuses a mismatched actual hidden by equal formal names",
      {ocaml|let all xs = List.for_all xs ~f:Fn.id
let checked xs other = (not (List.is_empty xs)) && all other
let () = Verdict.p "all rows pass" (checked rows other_rows)|ocaml},
      [ "all" ] );
    ( "refuses a shadowed guard identity across a nested alias",
      {ocaml|let close = List.for_all rows ~f:Fn.id
let guarded =
  let rows = [ true ] in
  (not (List.is_empty rows)) && close
let () = Verdict.p "all outer rows pass" guarded|ocaml},
      [ "close" ] );
    ( "refuses a binding nested directly inside a claim argument",
      {ocaml|let () =
  Verdict.p "all rows pass" (let ok = List.for_all rows ~f:Fn.id in ok)|ocaml},
      [ "ok" ] );
    ( "accepts a guarded binding nested directly inside a claim argument",
      {ocaml|let () =
  Verdict.p "all rows pass"
    (let ok = (not (List.is_empty rows)) && List.for_all rows ~f:Fn.id in ok)|ocaml},
      [] );
    ( "keeps an outer witness from guarding a shadowed nested population",
      {ocaml|let checked =
  (not (List.is_empty rows))
  && (let rows = [] in List.for_all rows ~f:Fn.id)
let () = Verdict.p "all rows pass" checked|ocaml},
      [ "checked" ] );
    ( "accepts a nested quantified population with its own witness",
      {ocaml|let checked =
  let rows = [ true ] in
  (not (List.is_empty rows)) && List.for_all rows ~f:Fn.id
let () = Verdict.p "all rows pass" checked|ocaml},
      [] );
    ( "accepts a negated binding nested directly inside a claim argument",
      {ocaml|let () =
  Verdict.p "some row fails" (let ok = List.for_all rows ~f:Fn.id in not ok)|ocaml},
      [] );
    ( "does not return a quantified binding shadowed by a later local",
      {ocaml|let result =
  let ok = List.for_all rows ~f:Fn.id in
  let ok = true in
  ok
let () = Verdict.p "the constant passes" result|ocaml},
      [] );
    ( "does not let an outer guard witness a match-bound population",
      {ocaml|let result =
  (not (List.is_empty rows))
  && match [] with rows -> List.for_all rows ~f:Fn.id
let () = Verdict.p "all inner rows pass" result|ocaml},
      [ "result" ] );
    ( "refuses a quantified component of a destructured tuple binding",
      {ocaml|let () =
  Verdict.p "all rows pass"
    (let ok, detail = List.for_all rows ~f:Fn.id, info in
     ok)|ocaml},
      [ "ok" ] );
    ( "conservatively refuses a quantified component of a record binding",
      {ocaml|let () =
  Verdict.p "all rows pass"
    (let { ok; detail } = { ok = List.for_all rows ~f:Fn.id; detail = info } in
     ok)|ocaml},
      [ "ok" ] );
    ( "refuses a quantified component destructured from an intermediate aggregate",
      {ocaml|let packed = List.for_all rows ~f:Fn.id, true
let ok, _ = packed
let () = Verdict.p "all rows pass" ok|ocaml},
      [ "packed" ] );
    ( "refuses a helper that returns a fully applied quantified local binding",
      {ocaml|let close xs =
  let ok = List.for_all xs ~f:Fn.id in
  ok
let () = Verdict.p "every sample agrees" (close samples)|ocaml},
      [ "close" ] );
    ( "refuses a helper that returns a match-bound quantified value",
      {ocaml|let close xs =
  match List.for_all xs ~f:Fn.id with ok -> ok
let () = Verdict.p "every sample agrees" (close samples)|ocaml},
      [ "close" ] );
    ( "refuses a direct quantifier forwarded by a Boolean constructor match",
      {ocaml|let close =
  match List.for_all rows ~f:Fn.id with true -> true | false -> false
let () = Verdict.p "all rows pass" close|ocaml},
      [ "close" ] );
    ( "refuses a direct quantifier forwarded by a wildcard Boolean match",
      {ocaml|let close =
  match List.for_all rows ~f:Fn.id with false -> false | _ -> true
let () = Verdict.p "all rows pass" close|ocaml},
      [ "close" ] );
    ( "accepts a direct quantifier inverted by a wildcard Boolean match",
      {ocaml|let differs =
  match List.for_all rows ~f:Fn.id with false -> true | _ -> false
let () = Verdict.p "some row fails" differs|ocaml},
      [] );
    ( "accepts a direct quantifier inverted by a Boolean constructor match",
      {ocaml|let differs =
  match List.for_all rows ~f:Fn.id with true -> false | false -> true
let () = Verdict.p "some row fails" differs|ocaml},
      [] );
    ( "refuses a bound exists inverted by a Boolean constructor match",
      {ocaml|let some = List.exists rows ~f:Fn.id
let none = match some with true -> false | false -> true
let () = Verdict.p "no rows match" none|ocaml},
      [ "some" ] );
    ( "refuses a bound exists inverted by aliased Boolean match outcomes",
      {ocaml|let some = List.exists rows ~f:Fn.id
let yes = true
let no = false
let result = match some with true -> no | false -> yes
let () = Verdict.p "no rows match" result|ocaml},
      [ "some" ] );
    ( "refuses a direct exists inverted by aliased Boolean match outcomes",
      {ocaml|let yes = true
let no = false
let result = match List.exists rows ~f:Fn.id with true -> no | false -> yes
let () = Verdict.p "no rows match" result|ocaml},
      [ "result" ] );
    ( "refuses a bound quantifier returned through an if condition",
      {ocaml|let all = List.for_all rows ~f:Fn.id
let close = if all then true else false
let () = Verdict.p "all rows pass" close|ocaml},
      [ "all" ] );
    ( "accepts an inverted bound quantifier returned through an if condition",
      {ocaml|let all = List.for_all rows ~f:Fn.id
let differs = if all then false else true
let () = Verdict.p "some row fails" differs|ocaml},
      [] );
    ( "refuses a direct quantifier returned through an if condition",
      {ocaml|let close = if List.for_all rows ~f:Fn.id then true else false
let () = Verdict.p "all rows pass" close|ocaml},
      [ "close" ] );
    ( "accepts an inverted direct quantifier returned through an if condition",
      {ocaml|let differs = if List.for_all rows ~f:Fn.id then false else true
let () = Verdict.p "some row fails" differs|ocaml},
      [] );
    ( "refuses a direct if condition whose false outcome is a Boolean alias",
      {ocaml|let no = false
let result = if List.exists rows ~f:Fn.id then no else true
let () = Verdict.p "no rows match" result|ocaml},
      [ "result" ] );
    ( "refuses a bound if condition whose false outcome is a Boolean alias",
      {ocaml|let some = List.exists rows ~f:Fn.id
let no = false
let result = if some then no else true
let () = Verdict.p "no rows match" result|ocaml},
      [ "some" ] );
    ( "refuses a direct if condition whose local false outcome is a Boolean alias",
      {ocaml|let result =
  let no = false in
  if List.exists rows ~f:Fn.id then no else true
let () = Verdict.p "no rows match" result|ocaml},
      [ "result" ] );
    ( "refuses an aliased quantified condition in a protected try body",
      {ocaml|let no = false
let result = try if List.exists rows ~f:Fn.id then no else true with _ -> false
let () = Verdict.p "no rows match" result|ocaml},
      [ "result" ] );
    ( "does not attribute a condition whose branches return the same literal",
      {ocaml|let all = List.for_all rows ~f:Fn.id
let result = if all then true else true
let () = Verdict.p "the constant passes" result|ocaml},
      [] );
    ( "refuses a bound quantifier returned through a match guard",
      {ocaml|let all = List.for_all rows ~f:Fn.id
let result = match () with () when all -> true | () -> false
let () = Verdict.p "all rows pass" result|ocaml},
      [ "all" ] );
    ( "accepts an inverted bound quantifier returned through a match guard",
      {ocaml|let all = List.for_all rows ~f:Fn.id
let result = match () with () when all -> false | () -> true
let () = Verdict.p "some row fails" result|ocaml},
      [] );
    ( "refuses a direct quantifier returned through a match guard",
      {ocaml|let result =
  match () with () when List.for_all rows ~f:Fn.id -> true | () -> false
let () = Verdict.p "all rows pass" result|ocaml},
      [ "result" ] );
    ( "accepts an inverted direct quantifier returned through a match guard",
      {ocaml|let result =
  match () with () when List.for_all rows ~f:Fn.id -> false | () -> true
let () = Verdict.p "some row fails" result|ocaml},
      [] );
    ( "refuses a direct match guard whose false result is a Boolean alias",
      {ocaml|let no = false
let result = match () with () when List.exists rows ~f:Fn.id -> no | () -> true
let () = Verdict.p "no rows match" result|ocaml},
      [ "result" ] );
    ( "refuses a bound match guard whose false result is a Boolean alias",
      {ocaml|let some = List.exists rows ~f:Fn.id
let no = false
let result = match () with () when some -> no | () -> true
let () = Verdict.p "no rows match" result|ocaml},
      [ "some" ] );
    ( "refuses a bound quantifier returned from a protected try body",
      {ocaml|let all = List.for_all rows ~f:Fn.id
let close = try all with _ -> false
let () = Verdict.p "all rows pass" close|ocaml},
      [ "all" ] );
    ( "refuses a direct quantifier returned through a try-case guard",
      {ocaml|let close =
  try raise Exit with
  | Exit when List.for_all rows ~f:Fn.id -> true
  | Exit -> false
let () = Verdict.p "all rows pass" close|ocaml},
      [ "close" ] );
    ( "accepts an inverted direct quantifier returned through a try-case guard",
      {ocaml|let differs =
  try raise Exit with
  | Exit when List.for_all rows ~f:Fn.id -> false
  | Exit -> true
let () = Verdict.p "some row fails" differs|ocaml},
      [] );
    ( "refuses a bound exists inverted through a try-case guard",
      {ocaml|let some = List.exists rows ~f:Fn.id
let none =
  try raise Exit with
  | Exit when some -> false
  | Exit -> true
let () = Verdict.p "no rows match" none|ocaml},
      [ "some" ] );
    ( "refuses a quantified helper reached through a mutually recursive sibling",
      {ocaml|let rec close xs = all xs
and all xs = List.for_all xs ~f:Fn.id
let () = Verdict.p "every sample agrees" (close samples)|ocaml},
      [ "all" ] );
    ( "does not resolve an outer quantified binding shadowed by a function parameter",
      {ocaml|let ok = List.for_all rows ~f:Fn.id
let identity ok = ok
let () = Verdict.p "constant identity passes" (identity true)|ocaml},
      [] );
    ( "does not resolve an outer binding shadowed by a match pattern",
      {ocaml|let ok = List.for_all rows ~f:Fn.id
let identity x = match x with ok -> ok
let () = Verdict.p "constant identity passes" (identity true)|ocaml},
      [] );
    ( "does not return an outer quantified local shadowed by a match pattern",
      {ocaml|let result =
  let ok = List.for_all rows ~f:Fn.id in
  ignore ok;
  match true with ok -> ok
let () = Verdict.p "the matched constant passes" result|ocaml},
      [] );
    ( "still resolves a non-shadowed quantified binding returned by a function",
      {ocaml|let ok = List.for_all rows ~f:Fn.id
let return_ok value = ok
let () = Verdict.p "all rows pass" (return_ok true)|ocaml},
      [ "ok" ] );
    ( "resolves an outer quantified binding used by an optional default",
      {ocaml|let ok = List.for_all rows ~f:Fn.id
let use ?(ok = ok) () = ok
let () = Verdict.p "all rows pass" (use ())|ocaml},
      [ "ok" ] );
    ( "does not use an optional default dependency when the caller supplies the argument",
      {ocaml|let outer_ok = List.for_all rows ~f:Fn.id
let use ?(ok = outer_ok) () = ok
let () = Verdict.p "the supplied constant passes" (use ~ok:true ())|ocaml},
      [] );
    ( "uses an optional default when a forwarded argument is None",
      {ocaml|let all = List.for_all rows ~f:Fn.id
let use ?(ok = all) () = ok
let () = Verdict.p "all rows pass" (use ?ok:None ())|ocaml},
      [ "all" ] );
    ( "does not use an optional default when a forwarded argument is definitely Some",
      {ocaml|let all = List.for_all rows ~f:Fn.id
let use ?(ok = all) () = ok
let () = Verdict.p "the forwarded constant passes" (use ?ok:(Some true) ())|ocaml},
      [] );
    ( "resolves a quantified binding through chained optional defaults",
      {ocaml|let all = List.for_all rows ~f:Fn.id
let use ?(x = all) ?(result = x) () = result
let () = Verdict.p "all rows pass" (use ())|ocaml},
      [ "all" ] );
    ( "suppresses an earlier default inside a later default when the caller supplies it",
      {ocaml|let all = List.for_all rows ~f:Fn.id
let use ?(x = all) ?(result = x) () = result
let () = Verdict.p "the supplied constant passes" (use ~x:true ())|ocaml},
      [] );
    ( "preserves polarity through an optional default",
      {ocaml|let ok = List.for_all rows ~f:Fn.id
let use ?(ok = not ok) () = ok
let () = Verdict.p "some row fails" (use ())|ocaml},
      [] );
    ( "refuses a quantified helper written with function-case syntax",
      {ocaml|let close = function xs -> List.for_all xs ~f:Fn.id
let () = Verdict.p "all rows pass" (close rows)|ocaml},
      [ "close" ] );
    ( "accepts a guarded quantified helper written with function-case syntax",
      {ocaml|let close = function
  | xs -> (not (List.is_empty xs)) && List.for_all xs ~f:Fn.id
let () = Verdict.p "all rows pass" (close rows)|ocaml},
      [] );
    ( "does not share a function-case guard with another case",
      {ocaml|let close = function
  | true, xs -> (not (List.is_empty xs)) && List.for_all xs ~f:Fn.id
  | false, xs -> List.for_all xs ~f:Fn.id
let () = Verdict.p "all rows pass" (close (false, rows))|ocaml},
      [ "close" ] );
    ( "accepts a helper that negates a quantified local binding",
      {ocaml|let differs xs =
  let close = List.for_all xs ~f:Fn.id in
  not close
let () = Verdict.p "some sample differs" (differs samples)|ocaml},
      [] );
    ( "refuses a double negation around a quantified local binding",
      {ocaml|let differs xs =
  let close = List.for_all xs ~f:Fn.id in
  not close
let () = Verdict.p "every sample agrees" (not (differs samples))|ocaml},
      [ "differs" ] );
    ( "refuses a quantifier written in pipeline style",
      {ocaml|let close xs = xs |> List.for_all ~f:Fn.id
let () = Verdict.p "every sample agrees" (close samples)|ocaml},
      [ "close" ] );
    ( "keeps helper resolution inside its lexical scope",
      {ocaml|let close xs = List.for_all xs ~f:Fn.id
let unrelated () =
  let close xs = (not (List.is_empty xs)) && List.for_all xs ~f:Fn.id in
  close samples
let () = Verdict.p "every sample agrees" (close samples)|ocaml},
      [ "close" ] );
    ( "refuses a reversed length upper bound masquerading as a witness",
      {ocaml|let close xs = 4 > List.length xs && List.for_all xs ~f:Fn.id
let () = Verdict.p "every sample agrees" (close samples)|ocaml},
      [ "close" ] );
    ( "preserves positive polarity through comparison with false",
      {ocaml|let close xs = List.for_all xs ~f:Fn.id
let () = Verdict.p "every sample agrees" (not (Bool.equal (close samples) false))|ocaml},
      [ "close" ] );
    ( "refuses an is_empty helper whose claim can pass on an empty source",
      {ocaml|let no_bad xs = List.is_empty (List.filter xs ~f:bad)
let () = Verdict.p "no sample is bad" (no_bad samples)|ocaml},
      [ "no_bad" ] );
    ( "refuses a negated exists helper with the same empty-population hole",
      {ocaml|let none_bad xs = not (Array.exists xs ~f:bad)
let () = Verdict.pass_fail "no sample is bad" (none_bad samples)|ocaml},
      [ "none_bad" ] );
    ( "accepts the explicit non-empty guard installed by the parity sweep",
      {ocaml|let close got want =
  (not (Array.is_empty got)) && Array.for_all2_exn got want ~f:Float.equal
let () = Verdict.p "the values agree" (close got want)|ocaml},
      [] );
    ( "does not let a guard on somebody else's population answer for the helper",
      {ocaml|let close got want other =
  (not (Array.is_empty other)) && Array.for_all2_exn got want ~f:Float.equal
let () = Verdict.p "the values agree" (close got want other)|ocaml},
      [ "close" ] );
    ( "accepts a positive literal length as the non-empty witness",
      {ocaml|let close got want =
  Array.length got = 4 && Array.for_all2_exn got want ~f:Float.equal
let () = Verdict.p "the values agree" (close got want)|ocaml},
      [] );
    ( "accepts a negated for_all2_exn discrimination helper",
      {ocaml|let differs got want = not (Array.for_all2_exn got want ~f:Float.equal)
let () = Verdict.p "some value differs" (differs got want)|ocaml},
      [] );
    ( "accepts a positive exists helper, which is false on an empty population",
      {ocaml|let some_bad xs = List.exists xs ~f:bad
let () = Verdict.p "some sample is bad" (some_bad samples)|ocaml},
      [] );
    ( "ignores a quantified helper that reaches no Verdict claim",
      {ocaml|let close got want = Array.for_all2_exn got want ~f:Float.equal
let () = if close got want then Stdio.printf "same\n"|ocaml},
      [] );
  ]

(* The shadowing fixtures, which the control list above cannot state: those cases compare the helper
   NAMES a source yields, and a name shadowed by a second definition of itself appears once in that
   comparison however many bodies carry it. What has to be pinned here is the opposite -- that one
   name comes back as two definitions -- so these are read for their definition SITES, and each
   collision is then handed to the refusal it must produce. Both claims of the same name are
   unguarded in each, so the reader would find every body on its own; the point is that a single
   exemption key would cover them all.

   The second fixture is the one a line number cannot serve. Its two `close` bindings are local to
   separate expressions written on ONE line -- the shape of a scanner test that spells a small
   helper inline in each of its cases -- so a definition identified by its line is one definition,
   and the exemption covers a body nobody read while this check reports green. *)
let shadowed_helper_fixture =
  {ocaml|let close got want = Array.for_all2_exn got want ~f:Float.equal
let () = Verdict.p "the first pair agrees" (close got want)
let close got want = Array.for_all2_exn got want ~f:Float.equal
let () = Verdict.p "the second pair agrees" (close got want)|ocaml}

let same_line_shadowed_helper_fixture =
  {ocaml|let () = (let close got want = Array.for_all2_exn got want ~f:Float.equal in Verdict.p "the first pair agrees" (close got want)); (let close got want = Array.for_all2_exn got want ~f:Float.equal in Verdict.p "the second pair agrees" (close got want))|ocaml}

let repeated_wrapper_call_fixture =
  {ocaml|let check value = Verdict.p "the collection property holds" value
let () = check (List.is_empty optional_rows)
let () = check (List.for_all rows ~f:Fn.id)|ocaml}

let multi_slot_wrapper_call_fixture =
  {ocaml|let check first second =
  Verdict.p "the optional rows are absent" first;
  Verdict.p "all rows pass" second
let () = check (List.is_empty optional_rows) (List.for_all rows ~f:Fn.id)|ocaml}

(* The definitions each exemption key resolves to, so that "one key, one helper" is something this
   check reads off the corpus rather than a property of names it hopes holds. Keyed by offset and
   carrying the printable site, so the report says where each body is and the identity does not
   depend on the report's precision.

   [record_definition] is one function rather than two spellings of an update because the corpus and
   the control below must agree on both halves of it -- the key, and what counts as a definition. A
   control that reproduced the aggregation instead of calling it would pass while the corpus stopped
   recording, and with one exempted helper in the tree nothing else would notice. *)
let quantified_exemption_key ~source claim = source ^ ":" ^ claim.helper

let scan_exemption_key ~source site =
  let identity =
    Scan.(match site.kind with Literal_label -> site.label | Computed_label -> site.head)
  in
  source ^ ":" ^ identity

let record_definition definitions ~key ~position ~description =
  Map.update definitions key ~f:(fun previous ->
      Map.set
        (Option.value previous ~default:(Map.empty (module Int)))
        ~key:position ~data:description)

let record_quantified_definition ~source definitions claim =
  record_definition definitions
    ~key:(quantified_exemption_key ~source claim)
    ~position:claim.helper_site.position ~description:(describe_site claim.helper_site)

let record_scan_definition ~source definitions site =
  record_definition definitions ~key:(scan_exemption_key ~source site) ~position:site.Scan.position
    ~description:(Printf.sprintf "%d:%d" site.Scan.line site.Scan.column)

let definition_sites ~source claims =
  List.fold claims ~init:(Map.empty (module String)) ~f:(record_quantified_definition ~source)

let colliding_exemptions definitions =
  Map.to_alist definitions
  |> List.filter_map ~f:(fun (key, sites) ->
      if Map.length sites > 1 then Some (key, Map.data sites) else None)

let run_quantified_helper_controls () =
  List.map quantified_helper_controls ~f:(fun (label, source, expected) ->
      let found =
        quantified_claims (Sources.structure_of source)
        |> List.map ~f:(fun claim -> claim.helper)
        |> List.dedup_and_sort ~compare:String.compare
      in
      let ok = List.equal String.equal found expected in
      if not ok then
        eprintf "quantified-helper control %S expected [%s], found [%s]\n" label
          (String.concat ~sep:", " expected)
          (String.concat ~sep:", " found);
      (label, ok))

(* The manifest's pin. [verdict_ratchet_controls.md] is prose, and prose drifts: a control renamed
   here and not there leaves a row nobody can find, and a row whose phrase names nothing leaves an
   inventory that reads complete. So the two are held equal from where the labels already are, in
   both directions -- every control label printed under "Synthetic helper-rule controls:", the
   run_*_control families included, appears in the manifest, and every phrase the manifest sets in
   backticks with a space in it (its convention for naming a control; a phrase starting with [dune]
   is a command) is such a label. The manifest is handed over by the rule's [(deps ...)], which is
   what makes a change to it re-run this. *)
let manifest_file = "verdict_ratchet_controls.md"

let manifest_control_phrases text =
  let rec collect acc from =
    match String.index_from text from '`' with
    | None -> acc
    | Some start -> (
        match String.index_from text (start + 1) '`' with
        | None -> acc
        | Some stop ->
            let span = String.sub text ~pos:(start + 1) ~len:(stop - start - 1) in
            let names_a_control =
              String.contains span ' '
              && (not (String.contains span '\n'))
              && not (String.is_prefix span ~prefix:"dune")
            in
            collect (if names_a_control then span :: acc else acc) (stop + 1))
  in
  collect [] 0 |> List.dedup_and_sort ~compare:String.compare

let manifest_row_label = "every synthetic control has a row in the mutation-run manifest"
let manifest_phrase_label = "every control phrase in the mutation-run manifest names a live control"

(* [controls] is every control result printed under "Synthetic helper-rule controls:" before these
   two -- the quantified list AND the run_*_control families, since the manifest promises them all
   -- so a case added to any family without a row fails here, not only one added to the list. *)
let run_manifest_controls ~manifest ~controls =
  let labels = List.map controls ~f:fst @ [ manifest_row_label; manifest_phrase_label ] in
  let phrases =
    match manifest with
    | Some text -> manifest_control_phrases text
    | None ->
        eprintf "%s is not among the arguments -- the rule's deps no longer hand it over\n"
          manifest_file;
        []
  in
  let label_set = Set.of_list (module String) labels in
  let phrase_set = Set.of_list (module String) phrases in
  List.iter labels ~f:(fun label ->
      if not (Set.mem phrase_set label) then
        eprintf "synthetic control without a row in %s: %S\n" manifest_file label);
  List.iter phrases ~f:(fun phrase ->
      if not (Set.mem label_set phrase) then
        eprintf "phrase in %s names no synthetic control: %S\n" manifest_file phrase);
  [
    (manifest_row_label, (not (List.is_empty labels)) && List.for_all labels ~f:(Set.mem phrase_set));
    ( manifest_phrase_label,
      (not (List.is_empty phrases)) && List.for_all phrases ~f:(Set.mem label_set) );
  ]

let quantified_failure source claim =
  let key = quantified_exemption_key ~source claim in
  Printf.sprintf
    "%s:%d sends `%s` from line %d into a Verdict claim, but that binding's `%s` can pass on an \
     empty population -- use the matching `Verdict.p_*` combinator, or make non-emptiness part of \
     the binding's passing result. If emptiness is the intended passing case, exempt `%s` by name \
     in verdict_ratchet.ml and say why"
    source claim.claim_line claim.helper claim.helper_site.line
    (List.map claim.quantifiers ~f:quantifier_name |> String.concat ~sep:", ")
    key

let refusal_mode = "--quantified-helper-refusal-control"

let run_refusal_control () =
  let exe = Stdlib.Sys.executable_name in
  let capture suffix = Stdlib.Filename.temp_file "verdict_ratchet_control" suffix in
  let out_path = capture ".out" and err_path = capture ".err" in
  let open_capture path = Unix.openfile path [ Unix.O_WRONLY; Unix.O_TRUNC ] 0o600 in
  let out = open_capture out_path and err = open_capture err_path in
  let pid = Unix.create_process exe [| exe; refusal_mode |] Unix.stdin out err in
  let _, status = Unix.waitpid [] pid in
  Unix.close out;
  Unix.close err;
  let output = In_channel.read_all out_path ^ In_channel.read_all err_path in
  let unlink path = try Unix.unlink path with Unix.Unix_error _ -> () in
  unlink out_path;
  unlink err_path;
  let ok =
    (match status with Unix.WEXITED 1 -> true | _ -> false)
    && String.is_substring output ~substring:"control_fixture.ml:3 sends `close`"
    && String.is_substring output ~substring:"can pass on an empty population"
  in
  if not ok then
    eprintf "the helper-refusal child did not reject its planted fixture as designed:\n%s\n" output;
  ("the shipping ratchet process refuses the planted helper fixture", ok)

let refuse_stale_quantified ~fail stale_quantified =
  if not (Set.is_empty stale_quantified) then
    fail
      (Printf.sprintf
         "exempted quantified bindings that no Verdict claim reaches any more -- drop them from \
          the exemption list: %s"
         (String.concat ~sep:", " (Set.to_list stale_quantified)))

let refuse_colliding_quantified ~fail colliding =
  if not (List.is_empty colliding) then
    fail
      (Printf.sprintf
         "exempted quantified bindings whose key names more than one definition, so one granted \
          exemption is silently covering helpers nobody read -- give the shadowing definitions \
          separate names, or hoist them into one: %s"
         (String.concat ~sep:", "
            (List.map colliding ~f:(fun (key, sites) ->
                 Printf.sprintf "%s (definitions at %s)" key (String.concat ~sep:", " sites)))))

let run_shadowed_quantified_control ?(helper = "close") ?(site_kind = "definitions") label fixture =
  let colliding =
    quantified_claims (Sources.structure_of fixture)
    |> definition_sites ~source:"fixture"
    |> colliding_exemptions
  in
  let two_definitions =
    match colliding with
    | [ (key, [ first; second ]) ] ->
        String.equal key ("fixture:" ^ helper) && not (String.equal first second)
    | _ -> false
  in
  if not two_definitions then
    eprintf "the %s fixture resolved to %s, not to one name with two %s\n" label
      (String.concat ~sep:", "
         (List.map colliding ~f:(fun (key, sites) ->
              Printf.sprintf "%s (%s)" key (String.concat ~sep:", " sites))))
      site_kind;
  let source = "test/operations/verdict_ratchet.ml" in
  let format =
    "exempted quantified bindings whose key names more than one definition, so one granted \
     exemption is silently covering helpers nobody read -- give the shadowing definitions separate \
     names, or hoist them into one: %s"
  in
  let refused = ref false in
  let fail _message =
    refused := true;
    Test_utils.Refusal_control_manifest.observe_failure ~source ~format
  in
  refuse_colliding_quantified ~fail colliding;
  [
    (Printf.sprintf "a %s helper name resolves to two %s, not one" label site_kind, two_definitions);
    (Printf.sprintf "refuses an exemption key that names both %s %s" label site_kind, !refused);
  ]

let run_shadowed_quantified_controls () =
  run_shadowed_quantified_control "shadowed" shadowed_helper_fixture
  @ run_shadowed_quantified_control "same-line shadowed" same_line_shadowed_helper_fixture
  @ run_shadowed_quantified_control ~helper:"check" ~site_kind:"call sites" "reused wrapper call"
      repeated_wrapper_call_fixture
  @ run_shadowed_quantified_control ~helper:"check" ~site_kind:"call slots"
      "multi-slot wrapper call" multi_slot_wrapper_call_fixture

let run_stale_quantified_control () =
  let source = "test/operations/verdict_ratchet.ml" in
  let format =
    "exempted quantified bindings that no Verdict claim reaches any more -- drop them from the \
     exemption list: %s"
  in
  let refused = ref false in
  let fail _message =
    refused := true;
    Test_utils.Refusal_control_manifest.observe_failure ~source ~format
  in
  refuse_stale_quantified ~fail (Set.singleton (module String) "fixture:stale");
  ("refuses a stale quantified-helper exemption", !refused)

let repeated_literal_site_fixture =
  {ocaml|let () = Stdio.printf "repeated label: %b\n" first
let () = Stdio.printf "repeated label: %b\n" second|ocaml}

let same_line_repeated_literal_site_fixture =
  {ocaml|let () = Stdio.printf "repeated label: %b\n" first; Stdio.printf "repeated label: %b\n" second|ocaml}

let repeated_computed_site_fixture =
  {ocaml|let () = Stdio.printf "%s repeated row: %b\n" first_name first
let () = Stdio.printf "%s repeated row: %b\n" second_name second|ocaml}

let same_line_repeated_computed_site_fixture =
  {ocaml|let () = Stdio.printf "%s repeated row: %b\n" first_name first; Stdio.printf "%s repeated row: %b\n" second_name second|ocaml}

let refuse_colliding_sites ~fail colliding =
  if not (List.is_empty colliding) then
    fail
      (Printf.sprintf
         "exempted claim-shaped literal keys that name more than one source site, so one granted \
          exemption is silently covering prints nobody read -- give the sites distinct labels or \
          formats, or route them through one shared printer: %s"
         (String.concat ~sep:", "
            (List.map colliding ~f:(fun (key, sites) ->
                 Printf.sprintf "%s (sites at %s)" key (String.concat ~sep:", " sites)))))

let site_definitions ~source ~kind fixture =
  (Scan.scan fixture).Scan.sites
  |> List.filter ~f:(fun site -> Poly.equal site.Scan.kind kind)
  |> List.fold ~init:(Map.empty (module String)) ~f:(record_scan_definition ~source)

let run_colliding_site_control label kind expected_key fixture =
  let colliding = site_definitions ~source:"fixture" ~kind fixture |> colliding_exemptions in
  let two_sites =
    match colliding with
    | [ (key, [ first; second ]) ] ->
        String.equal key expected_key && not (String.equal first second)
    | _ -> false
  in
  if not two_sites then
    eprintf "the %s fixture resolved to %s, not to one key with two source sites\n" label
      (String.concat ~sep:", "
         (List.map colliding ~f:(fun (key, sites) ->
              Printf.sprintf "%s (%s)" key (String.concat ~sep:", " sites))));
  let source = "test/operations/verdict_ratchet.ml" in
  let format =
    "exempted claim-shaped literal keys that name more than one source site, so one granted \
     exemption is silently covering prints nobody read -- give the sites distinct labels or \
     formats, or route them through one shared printer: %s"
  in
  let refused = ref false in
  let fail _message =
    refused := true;
    Test_utils.Refusal_control_manifest.observe_failure ~source ~format
  in
  refuse_colliding_sites ~fail colliding;
  [
    (Printf.sprintf "a %s exemption key resolves to two source sites, not one" label, two_sites);
    (Printf.sprintf "refuses an exemption key that names both %s source sites" label, !refused);
  ]

let run_colliding_site_controls () =
  run_colliding_site_control "repeated literal-label" Scan.Literal_label "fixture:repeated label"
    repeated_literal_site_fixture
  @ run_colliding_site_control "same-line repeated literal-label" Scan.Literal_label
      "fixture:repeated label" same_line_repeated_literal_site_fixture
  @ run_colliding_site_control "repeated computed-label" Scan.Computed_label
      "fixture:%s repeated row: " repeated_computed_site_fixture
  @ run_colliding_site_control "same-line repeated computed-label" Scan.Computed_label
      "fixture:%s repeated row: " same_line_repeated_computed_site_fixture

let base_dir = Dune.base_dir
let repo_relative = Dune.repo_relative

let () =
  if Array.length Stdlib.Sys.argv >= 2 && String.equal Stdlib.Sys.argv.(1) refusal_mode then (
    let _, source, _ = List.hd_exn quantified_helper_controls in
    let claims = quantified_claims (Sources.structure_of source) in
    if List.is_empty claims then (
      eprintf "the planted helper fixture produced no finding\n";
      Stdlib.exit 2);
    List.iter claims ~f:(fun claim -> Verdict.fail (quantified_failure "control_fixture.ml" claim));
    Stdlib.exit 1);
  if Array.length Stdlib.Sys.argv < 2 then (
    eprintf "Usage: %s <workspace_root> <source...>\n" Stdlib.Sys.argv.(0);
    Stdlib.exit 1);
  let base = base_dir Stdlib.Sys.argv.(1) in
  (* Reported repository-relative, opened as dune handed them over: the working directory is the
     rule's own, deep in the build tree. *)
  let arguments =
    Array.to_list (Array.subo Stdlib.Sys.argv ~pos:2)
    |> List.map ~f:(fun path -> (repo_relative base path, path))
  in
  let on_disk = Map.of_alist_reduce (module String) arguments ~f:(fun first _ -> first) in
  (* `.ml` files, minus dune's preprocessed twin of one already in the list: the twin is the ppx
     expansion of a file scanned anyway, and it exists only where the library that owns it is built.
     Shared with the configuration scans, which need the same thing of the same `%{deps}`. *)
  let sources = Sources.sources_among (List.map arguments ~f:fst) in
  if List.is_empty sources then (
    Verdict.fail "no OCaml sources among the arguments -- the rule's globs match nothing";
    Stdlib.exit 1);
  (* Failures go through [Verdict]: the module whose absence at these sites is the whole subject.
     Reported on both channels, and the run exits nonzero from its teardown, so the exit status
     rather than a promotable golden diff carries the verdict (gh-ocannl-601). *)
  let fail message = Verdict.fail message in
  let exemptions = Map.of_alist_exn (module String) exempt_sites in
  let computed_exemptions = Map.of_alist_exn (module String) exempt_computed_sites in
  let quantified_exemptions = Map.of_alist_exn (module String) exempt_quantified_helpers in
  let computed_used = ref (Set.empty (module String)) in
  let quantified_used = ref (Set.empty (module String)) in
  let literal_definitions = ref (Map.empty (module String)) in
  let computed_definitions = ref (Map.empty (module String)) in
  let quantified_definitions = ref (Map.empty (module String)) in
  let canaries = Map.of_alist_exn (module String) canary_sites in
  let data = Map.of_alist_exn (module String) data_sources in
  let exemptions_used = ref (Set.empty (module String)) in
  let canaries_found = ref (Set.empty (module String)) in
  let data_used = ref (Set.empty (module String)) in
  let literals = ref 0 and applied = ref 0 and offenders = ref 0 in
  let quantified_offenders = ref 0 in
  let manifest =
    List.find_map arguments ~f:(fun (relative, path) ->
        if String.is_suffix relative ~suffix:("/" ^ manifest_file) then
          Some (Stdio.In_channel.read_all path)
        else None)
  in
  let control_results =
    run_quantified_helper_controls ()
    @ [ run_refusal_control (); run_stale_quantified_control () ]
    @ run_shadowed_quantified_controls ()
    @ run_colliding_site_controls ()
  in
  let control_results =
    control_results @ run_manifest_controls ~manifest ~controls:control_results
  in
  let per_directory = Hashtbl.create (module String) in
  printf
    "Test sources that print a claim they decided themselves, outside `Verdict`: a format whose\n\
     last argument-consuming conversion is a bare `%%b` at the end, behind a label ending in `:`,\n\
     `=` or `->` -- written out (gh-ocannl-668) or computed from arguments (gh-ocannl-624). Such\n\
     a line is gated only by the golden diff, and a golden diff is `dune promote`-able -- which\n\
     is how a failure gets recorded as the expected output.\n\n";
  List.iter sources ~f:(fun source ->
      let path = Map.find_exn on_disk source in
      let content = In_channel.read_all path in
      (* A source this reader cannot read is reported by NAME and the scan carries on, rather than
         taking the run down with a syntax error naming no file: the corpus is globbed, so what
         arrives is whatever the test directories hold -- including whatever a `(select …)` or a ppx
         put there -- and the one thing worse than a parse failure here is one that leaves nobody
         knowing which of three hundred files it was about. *)
      let scanned, helper_claims =
        try (Scan.scan content, quantified_claims (Sources.structure_of content))
        with exception_ ->
          fail
            (Printf.sprintf "%s does not parse as OCaml, so this check cannot vouch for it: %s"
               source (Exn.to_string exception_));
          ({ Scan.sites = []; literals = 0; applied_literals = 0 }, [])
      in
      literals := !literals + scanned.Scan.literals;
      applied := !applied + scanned.Scan.applied_literals;
      Hashtbl.update per_directory (Stdlib.Filename.dirname source) ~f:(fun previous ->
          let files, found = Option.value previous ~default:(0, 0) in
          (files + 1, found + List.length scanned.Scan.sites));
      List.iter scanned.Scan.sites ~f:(fun site ->
          (* A literal-label site is named by its label, which IS what the format says; a computed
             one by the whole format, because its label is only what survived rendering a head this
             reader cannot fill in. *)
          let computed =
            Scan.(match site.kind with Computed_label -> true | Literal_label -> false)
          in
          let key = scan_exemption_key ~source site in
          let where = Printf.sprintf "%s:%d:%d" source site.Scan.line site.Scan.column in
          let how =
            match site.Scan.printer with
            | Some printer -> Printf.sprintf " through `%s`" printer
            | None -> ""
          in
          let canary_key = source ^ ":" ^ site.Scan.label in
          if Map.mem canaries canary_key then (
            canaries_found := Set.add !canaries_found canary_key;
            data_used := Set.add !data_used source)
          else if Map.mem data source then data_used := Set.add !data_used source
          else if (not computed) && Map.mem exemptions key then (
            exemptions_used := Set.add !exemptions_used key;
            literal_definitions := record_scan_definition ~source !literal_definitions site)
          else if computed && Map.mem computed_exemptions key then (
            computed_used := Set.add !computed_used key;
            computed_definitions := record_scan_definition ~source !computed_definitions site)
          else (
            Int.incr offenders;
            let remedy =
              if computed then
                Printf.sprintf
                  "write it as `Verdict.pf \"%s\" <args> <bool>` (or `Verdict.claimf`, if the \
                   surrounding row must keep its shape)"
                  (String.substr_replace_all
                     (String.chop_suffix_if_exists site.Scan.format ~suffix:"\n")
                     ~pattern:": %b" ~with_:"")
              else Printf.sprintf "write it as `Verdict.p \"%s\" <bool>`" site.Scan.label
            in
            fail
              (Printf.sprintf
                 "%s prints the claim `%s`%s, deciding its own verdict outside `Verdict` -- %s, so \
                  that a false exits the run instead of being `dune promote`d into %s. If the line \
                  describes rather than asserts, exempt it by name in verdict_ratchet.ml with the \
                  reason it is not an assertion"
                 where site.Scan.label how remedy
                 (Stdlib.Filename.remove_extension (Stdlib.Filename.basename source) ^ ".expected"))));
      List.iter helper_claims ~f:(fun claim ->
          let key = quantified_exemption_key ~source claim in
          if Map.mem quantified_exemptions key then (
            quantified_used := Set.add !quantified_used key;
            quantified_definitions :=
              record_quantified_definition ~source !quantified_definitions claim)
          else (
            Int.incr quantified_offenders;
            fail (quantified_failure source claim))));
  (* Which directories the corpus came from, by name and not by count: a file added anywhere under
     `test/` moved a tally here, so every contributor would promote this file over a change that
     never touched it -- a promote indistinguishable from blessing a real regression (the lesson of
     gh-ocannl-665). The counts go to stderr, which a `(test)` stanza does not diff. A directory
     that stops being scanned still shows up, by leaving this line. *)
  let directories = Hashtbl.keys per_directory |> List.sort ~compare:String.compare in
  printf "Directories scanned: %s\n\n" (String.concat ~sep:", " directories);
  printf "Sources whose claim-shaped literals are this check's own input, not prints:\n";
  List.iter data_sources ~f:(fun (path, why) -> printf "  %s -- %s\n" path why);
  printf "\nPlanted in that fixture so that a scan which went blind cannot report a clean tree:\n";
  List.iter canary_sites ~f:(fun (key, why) -> printf "  %s -- %s\n" key why);
  printf "\nLiteral-label claims exempted, with the reason each is not an assertion:\n";
  if List.is_empty exempt_sites then
    printf
      "  (none: a bare `<label>: %%b` line with nothing else on it has always been a verdict)\n"
  else List.iter exempt_sites ~f:(fun (key, why) -> printf "  %s -- %s\n" key why);
  printf
    "\n\
     Computed-label claims exempted -- rows and tables that describe rather than decide,\n\
     each carrying its assertion separately through `Verdict.claim`/`claimf`:\n";
  List.iter exempt_computed_sites ~f:(fun (key, why) -> printf "  %s -- %s\n" key why);
  printf "\nQuantified bindings exempted because emptiness is their passing meaning:\n";
  if List.is_empty exempt_quantified_helpers then
    printf "  (none: every quantified binding in a claim must witness a population)\n"
  else List.iter exempt_quantified_helpers ~f:(fun (key, why) -> printf "  %s -- %s\n" key why);
  printf "\nSynthetic helper-rule controls:\n";
  List.iter control_results ~f:(fun (label, ok) -> Verdict.pf "%s" label ok);
  let stale =
    Set.union
      (Set.diff (Set.of_list (module String) (List.map exempt_sites ~f:fst)) !exemptions_used)
      (Set.diff
         (Set.of_list (module String) (List.map exempt_computed_sites ~f:fst))
         !computed_used)
  in
  let stale_quantified =
    Set.diff
      (Set.of_list (module String) (List.map exempt_quantified_helpers ~f:fst))
      !quantified_used
  in
  let colliding_quantified = colliding_exemptions !quantified_definitions in
  let colliding_sites =
    colliding_exemptions !literal_definitions @ colliding_exemptions !computed_definitions
  in
  if not (Set.is_empty stale) then
    fail
      (Printf.sprintf
         "exempted literals that no source carries any more -- drop them from the exemption list: \
          %s"
         (String.concat ~sep:", " (Set.to_list stale)));
  refuse_stale_quantified ~fail stale_quantified;
  refuse_colliding_sites ~fail colliding_sites;
  refuse_colliding_quantified ~fail colliding_quantified;
  (* An exempted source that carries no claim-shaped literal is either a file that stopped being a
     fixture, or one this scan stopped reading -- and the second is what a blanket exemption is
     capable of hiding, so it is checked rather than trusted. *)
  let unread = Set.diff (Set.of_list (module String) (List.map data_sources ~f:fst)) !data_used in
  if not (Set.is_empty unread) then
    fail
      (Printf.sprintf
         "sources exempted as this check's own input that carry no claim-shaped literal any more \
          -- either they are no longer fixtures, or the scan is no longer reading them: %s"
         (String.concat ~sep:", " (Set.to_list unread)));
  let missing =
    Set.diff (Set.of_list (module String) (List.map canary_sites ~f:fst)) !canaries_found
  in
  if not (Set.is_empty missing) then
    fail
      (Printf.sprintf
         "planted canaries the scan did not find: %s -- either the fixture no longer carries them, \
          or this scan has stopped reading the corpus and its empty offender list means nothing"
         (String.concat ~sep:", " (Set.to_list missing)));
  eprintf "Sources scanned per directory (not diffed -- see gh-ocannl-665):\n";
  List.iter directories ~f:(fun directory ->
      let files, found = Hashtbl.find_exn per_directory directory in
      eprintf "  %s: %d source%s, %d claim-shaped literal%s\n" directory files
        (if files = 1 then "" else "s")
        found
        (if found = 1 then "" else "s"));
  eprintf "Totals: %d sources, %d string literals (%d of them an argument of a named function).\n"
    (List.length sources) !literals !applied;
  printf "\n";
  (* Stated so that `true` is the passing reading, as every line of a golden should be. *)
  Verdict.p "every test source decides its claims through Verdict" (!offenders = 0);
  Verdict.p "every quantified binding used by a claim witnesses a non-empty population"
    (!quantified_offenders = 0);
  Verdict.p "the scan found every literal planted for it" (Set.is_empty missing);
  Verdict.p "every exemption on this check's lists is still earned"
    (Set.is_empty unread && Set.is_empty stale && Set.is_empty stale_quantified);
  Verdict.p "every exempted claim-shaped literal is one source site, not a shared key"
    (List.is_empty colliding_sites);
  Verdict.p "every exempted quantified binding is one definition, not a shared name"
    (List.is_empty colliding_quantified);
  (* What a blind walk cannot produce. Without these, "no offenders" and "read nothing" are the same
     result -- and the second is the one that arrives silently. *)
  Verdict.p "the walk read string literals out of these sources" (!literals > 0);
  Verdict.p "and placed some of them as arguments of a named function" (!applied > 0);
  Verdict.p "over more than one test directory" (List.length directories > 1);
  if not (Verdict.any_failed ()) then
    printf
      "\nOK: test claims route through `Verdict`, and quantified bindings cannot pass on nothing.\n";
  Test_utils.Refusal_control_manifest.print "verdict_ratchet.ml"
