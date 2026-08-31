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
   passing result. *)

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

type quantifier = { kind : quantifier_kind; populations : Set.M(String).t }
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
  dependencies : helper_binding list;
  unguarded : quantifier list;
  claim_kind : claim_kind option;
}

type quantified_claim = {
  helper : string;
  helper_site : definition_site;
  claim_line : int;
  quantifiers : quantifier_kind list;
}

let describe_site site = Printf.sprintf "%d:%d" site.line site.column

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

let rec helper_name pattern =
  match pattern.ppat_desc with
  | Ppat_var { txt; _ } -> Some txt
  | Ppat_alias (_, { txt; _ }) -> Some txt
  | Ppat_constraint (inner, _) -> helper_name inner
  | _ -> None

let rec function_body expr =
  match expr.pexp_desc with
  | Pexp_function (_, _, Pfunction_body body) -> function_body body
  | _ -> expr

let is_function expr = match expr.pexp_desc with Pexp_function _ -> true | _ -> false

let unlabelled arguments =
  List.filter_map arguments ~f:(function Asttypes.Nolabel, argument -> Some argument | _ -> None)

let rec population_name expr =
  match Sources.longident_of expr with
  | Some [ name ] -> Some name
  | _ -> (
      match expr.pexp_desc with
      | Pexp_constraint (inner, _) | Pexp_coerce (inner, _, _) -> population_name inner
      | Pexp_apply (callee, arguments)
        when is_collection_call callee ~member:"filter"
             || is_collection_call callee ~member:"filter_map" ->
          List.hd (unlabelled arguments) |> Option.bind ~f:population_name
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

let quantifiers_in expr =
  let found = ref [] in
  let positive = ref true in
  let iterator =
    object (self)
      inherit Ast_traverse.iter as super
      method! attribute _ = ()

      method! expression expr =
        match expr.pexp_desc with
        | Pexp_apply (callee, [ (Asttypes.Nolabel, argument) ]) when is_name callee "not" ->
            let previous = !positive in
            positive := not previous;
            self#expression argument;
            positive := previous
        | Pexp_apply (pipe, [ (Asttypes.Nolabel, population); (Asttypes.Nolabel, piped_call) ])
          when is_name pipe "|>" -> (
            match piped_call.pexp_desc with
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
                      }
                      :: !found;
                    self#expression population;
                    List.iter arguments ~f:(fun (_, argument) -> self#expression argument)
                | _ -> super#expression expr)
            | _ -> super#expression expr)
        | Pexp_apply (callee, arguments) ->
            let add kind count =
              found := { kind; populations = populations arguments count } :: !found
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

(* Quantifiers that contribute to the value a helper RETURNS, rather than to setup or validation it
   performs on the way. [nonzero] helpers are the important near miss: they use [not (exists ...)]
   only to decide whether to raise, then return the input array. Treating every expression in their
   body as the helper's boolean made every later parity claim look helper-wrapped. *)
let rec returned_quantifiers expr =
  match expr.pexp_desc with
  | Pexp_function (_, _, Pfunction_body body) -> returned_quantifiers body
  | Pexp_let (_, bindings, body) ->
      let returned_names = returned_value_names body in
      returned_quantifiers body
      @ List.concat_map bindings ~f:(fun binding ->
          match helper_name binding.pvb_pat with
          | Some name when Set.mem returned_names name -> returned_quantifiers binding.pvb_expr
          | _ -> [])
  | Pexp_sequence (_, result) -> returned_quantifiers result
  | Pexp_constraint (inner, _) | Pexp_coerce (inner, _, _) -> returned_quantifiers inner
  | Pexp_ifthenelse (_, yes, no) ->
      returned_quantifiers yes @ Option.value_map no ~default:[] ~f:returned_quantifiers
  | Pexp_match (_, cases) | Pexp_try (_, cases) ->
      List.concat_map cases ~f:(fun case -> returned_quantifiers case.pc_rhs)
  | Pexp_apply (callee, _)
    when is_collection_call callee ~member:"for_all"
         || is_collection_call callee ~member:"for_all2_exn"
         || is_collection_call callee ~member:"is_empty"
         || is_name callee "not" || is_name callee "&&" || is_name callee "||" || is_name callee "="
         || is_name callee "<>" || is_name callee "|>" ->
      quantifiers_in expr
  | _ -> []

and returned_value_names expr =
  match expr.pexp_desc with
  | Pexp_ident { txt = Ppxlib.Longident.Lident name; _ } -> Set.singleton (module String) name
  | Pexp_function (_, _, Pfunction_body body) -> returned_value_names body
  | Pexp_let (_, _, body) -> returned_value_names body
  | Pexp_sequence (_, result) -> returned_value_names result
  | Pexp_constraint (inner, _) | Pexp_coerce (inner, _, _) -> returned_value_names inner
  | Pexp_ifthenelse (_, yes, no) ->
      Set.union (returned_value_names yes)
        (Option.value_map no ~default:(Set.empty (module String)) ~f:returned_value_names)
  | Pexp_match (_, cases) | Pexp_try (_, cases) ->
      List.fold cases
        ~init:(Set.empty (module String))
        ~f:(fun names case -> Set.union names (returned_value_names case.pc_rhs))
  | Pexp_apply (callee, arguments)
    when is_name callee "not" || is_name callee "&&" || is_name callee "||" || is_name callee "="
         || is_name callee "<>" ->
      List.fold arguments
        ~init:(Set.empty (module String))
        ~f:(fun names (_, argument) -> Set.union names (returned_value_names argument))
  | Pexp_apply (callee, _) -> (
      match Sources.longident_of callee with
      | Some [ name ] -> Set.singleton (module String) name
      | _ -> Set.empty (module String))
  | _ -> Set.empty (module String)

let is_partial_quantifier expr =
  match expr.pexp_desc with
  | Pexp_apply (callee, arguments) when is_collection_call callee ~member:"for_all" ->
      List.length (unlabelled arguments) < 1
  | Pexp_apply (callee, arguments) when is_collection_call callee ~member:"for_all2_exn" ->
      List.length (unlabelled arguments) < 2
  | Pexp_apply (callee, arguments) when is_collection_call callee ~member:"is_empty" ->
      List.length (unlabelled arguments) < 1
  | _ -> false

let int_literal expr =
  match expr.pexp_desc with
  | Pexp_constant (Pconst_integer (value, _)) -> Option.try_with (fun () -> Int.of_string value)
  | _ -> None

let length_population expr =
  match expr.pexp_desc with
  | Pexp_apply (callee, arguments) when is_collection_call callee ~member:"length" ->
      List.hd (unlabelled arguments) |> Option.bind ~f:population_name
  | _ -> None

let bool_literal expr value =
  match expr.pexp_desc with
  | Pexp_construct ({ txt = Ppxlib.Longident.Lident found; _ }, None) ->
      String.equal found (Bool.to_string value)
  | _ -> false

(* Populations that HAVE to be non-empty for this expression to be true. This is intentionally a
   small boolean grammar: conjunction composes requirements, [not (X.is_empty xs)] is the spelling
   the gh-ocannl-746 helper sweep installed, and a positive literal length pins the same fact. A
   construct the reader cannot prove contributes no guard, so it produces a loud finding rather than
   silently licensing a vacuous helper. *)
let rec required_nonempty expr =
  let none () = Set.empty (module String) in
  match expr.pexp_desc with
  | Pexp_function (_, _, Pfunction_body body) -> required_nonempty body
  | Pexp_let (_, _, body) -> required_nonempty body
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

let names_in expr =
  let names = ref (Set.empty (module String)) in
  let iterator =
    object
      inherit Ast_traverse.iter as super
      method! attribute _ = ()
      method! value_binding _ = ()

      method! expression expr =
        (match expr.pexp_desc with Pexp_function _ -> () | _ -> super#expression expr);
        match Sources.longident_of expr with
        | Some [ name ] -> names := Set.add !names name
        | _ -> ()
    end
  in
  iterator#expression expr;
  !names

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

let opens_verdict_claims module_expr =
  match module_expr.pmod_desc with
  | Pmod_ident { txt; _ } -> (
      try List.equal String.equal (Ppxlib.Longident.flatten_exn txt) [ "Verdict"; "Claims" ]
      with _ -> false)
  | _ -> false

(* [open Verdict.Claims] is the migration target of gh-ocannl-815. Model its bindings explicitly so
   helper-following does not lose sight of an unqualified [p]/[claim]/[pass_fail] call when the file
   deletes its local aliases. Ordinary value bindings are prepended later and therefore shadow these
   exactly as they do in OCaml. *)
let opened_claim_bindings site =
  [ ("p", P); ("pf", Pf); ("pass_fail", Pass_fail); ("claim", Claim); ("claimf", Claimf) ]
  |> List.map ~f:(fun (name, claim_kind) ->
      { name; site; dependencies = []; unguarded = []; claim_kind = Some claim_kind })

let open_claims environment declaration =
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

let lookup environment name =
  List.find environment ~f:(fun binding -> String.equal binding.name name)

let positive_names expr =
  let names = ref (Set.empty (module String)) in
  let rec visit positive expr =
    let visit_arguments positive arguments =
      List.iter arguments ~f:(fun (_, argument) -> visit positive argument)
    in
    match expr.pexp_desc with
    | Pexp_apply (callee, [ (Asttypes.Nolabel, argument) ]) when is_name callee "not" ->
        visit (not positive) argument
    | Pexp_apply (callee, [ (Asttypes.Nolabel, left); (Asttypes.Nolabel, right) ])
      when is_name callee "=" || is_name callee "<>" || is_name callee "equal" -> (
        let equal = not (is_name callee "<>") in
        match
          ( bool_literal left true,
            bool_literal left false,
            bool_literal right true,
            bool_literal right false )
        with
        | true, _, _, _ -> visit (Bool.equal positive equal) right
        | _, true, _, _ -> visit (Bool.equal positive (not equal)) right
        | _, _, true, _ -> visit (Bool.equal positive equal) left
        | _, _, _, true -> visit (Bool.equal positive (not equal)) left
        | _ -> visit_arguments positive [ (Asttypes.Nolabel, left); (Asttypes.Nolabel, right) ])
    | Pexp_function _ -> ()
    | _ ->
        (match Sources.longident_of expr with
        | Some [ name ] when positive -> names := Set.add !names name
        | _ -> ());
        let iterator =
          object
            inherit Ast_traverse.iter as super
            method! attribute _ = ()
            method! expression child = visit positive child
            method children child = super#expression child
          end
        in
        iterator#children expr
  in
  visit true expr;
  !names

let claim_kind environment callee =
  match Sources.longident_of callee with
  | Some [ alias ] -> lookup environment alias |> Option.bind ~f:(fun binding -> binding.claim_kind)
  | Some path -> claim_kind_of_path path
  | None -> None

let make_binding environment value =
  helper_name value.pvb_pat
  |> Option.map ~f:(fun name ->
      let guards = required_nonempty value.pvb_expr in
      let unguarded =
        if is_function value.pvb_expr || is_partial_quantifier value.pvb_expr then
          List.filter (returned_quantifiers value.pvb_expr) ~f:(fun quantifier ->
              Set.is_empty quantifier.populations
              || Set.is_empty (Set.inter guards quantifier.populations))
        else []
      in
      let dependencies =
        names_in (function_body value.pvb_expr)
        |> Set.to_list
        |> List.filter_map ~f:(lookup environment)
      in
      let claim_kind =
        match Sources.longident_of value.pvb_expr with
        | Some [ alias ] ->
            lookup environment alias |> Option.bind ~f:(fun binding -> binding.claim_kind)
        | Some path -> claim_kind_of_path path
        | None -> None
      in
      let start = value.pvb_loc.loc_start in
      let site =
        {
          line = start.Stdlib.Lexing.pos_lnum;
          column = start.Stdlib.Lexing.pos_cnum - start.Stdlib.Lexing.pos_bol;
          position = start.Stdlib.Lexing.pos_cnum;
        }
      in
      { name; site; dependencies; unguarded; claim_kind })

let quantified_claims structure =
  let origins bindings =
    let rec visit seen binding =
      let key = binding.name ^ ":" ^ Int.to_string binding.site.position in
      if Set.mem seen key then []
      else
        let seen = Set.add seen key in
        let direct = if List.is_empty binding.unguarded then [] else [ binding ] in
        direct @ List.concat_map binding.dependencies ~f:(visit seen)
    in
    List.concat_map bindings ~f:(visit (Set.empty (module String)))
    |> List.dedup_and_sort ~compare:(fun a b -> Int.compare a.site.position b.site.position)
  in
  let found = ref [] in
  let record_claim environment expr =
    match expr.pexp_desc with
    | Pexp_apply (callee, arguments) ->
        Option.iter (claim_kind environment callee) ~f:(fun _ ->
            match List.last (unlabelled arguments) with
            | None -> ()
            | Some boolean ->
                positive_names boolean |> Set.to_list
                |> List.filter_map ~f:(lookup environment)
                |> origins
                |> List.iter ~f:(fun binding ->
                    found :=
                      {
                        helper = binding.name;
                        helper_site = binding.site;
                        claim_line = expr.pexp_loc.loc_start.pos_lnum;
                        quantifiers =
                          List.map binding.unguarded ~f:(fun quantifier -> quantifier.kind)
                          |> List.dedup_and_sort ~compare:Poly.compare;
                      }
                      :: !found))
    | _ -> ()
  in
  let rec scan_expression environment expr =
    record_claim environment expr;
    match expr.pexp_desc with
    | Pexp_let (_, bindings, body) ->
        List.iter bindings ~f:(fun binding -> scan_expression environment binding.pvb_expr);
        let local = List.filter_map bindings ~f:(make_binding environment) in
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
           | Pstr_value (_, bindings) ->
               List.iter bindings ~f:(fun binding -> scan_expression environment binding.pvb_expr);
               List.rev_append (List.filter_map bindings ~f:(make_binding environment)) environment
           | Pstr_eval (expr, _) ->
               scan_expression environment expr;
               environment
           | Pstr_module binding ->
               scan_module environment binding.pmb_expr;
               environment
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

(* Helper-wrapped quantified claims whose passing meaning genuinely ALLOWS an empty population.
   Keyed by [<repository-relative path>:<helper name>], and stale-checked below. The exemption is on
   the helper rather than every claim that calls it: the helper is the unit whose boolean semantics
   decide what empty means, and every call reaches the same decision.

   That last sentence is a precondition, not a fact about names, so it is checked below rather than
   assumed. A name is the unit only while it denotes ONE definition: a file that shadows `refused`
   with a second `refused`, or defines one per local scope, hands both to the same key, and the
   exemption -- granted after reading one body -- would license the other silently, which is exactly
   the shape the reader was widened to catch. Two definitions under one exempted key therefore
   REFUSE the run; the fix is to give them separate names, or to hoist them into one. *)
let exempt_quantified_helpers =
  [
    ( "test/operations/ocamlformat_ignore_scan.ml:refused",
      "the message list is an optional strengthening of the child-exit refusal: an empty list \
       deliberately means that the nonzero status alone is the passing evidence" );
  ]

(* Synthetic inputs state the helper rule independently of whatever helpers happen to be in the
   repository today. The first four are negative controls: the rule must return an offender for
   each, which is the same list the corpus loop below turns into a [Verdict.fail]. The rest are the
   nearest accepted forms, so widening the ratchet until ordinary boolean helpers need exemptions
   also fails here rather than growing a noisy central list. *)
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
    ( "refuses a helper that returns a fully applied quantified local binding",
      {ocaml|let close xs =
  let ok = List.for_all xs ~f:Fn.id in
  ok
let () = Verdict.p "every sample agrees" (close samples)|ocaml},
      [ "close" ] );
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

(* The definitions each exemption key resolves to, so that "one key, one helper" is something this
   check reads off the corpus rather than a property of names it hopes holds. Keyed by offset and
   carrying the printable site, so the report says where each body is and the identity does not
   depend on the report's precision.

   [record_definition] is one function rather than two spellings of an update because the corpus and
   the control below must agree on both halves of it -- the key, and what counts as a definition. A
   control that reproduced the aggregation instead of calling it would pass while the corpus stopped
   recording, and with one exempted helper in the tree nothing else would notice. *)
let exemption_key ~source claim = source ^ ":" ^ claim.helper

let record_definition ~source definitions claim =
  Map.update definitions (exemption_key ~source claim) ~f:(fun previous ->
      Map.set
        (Option.value previous ~default:(Map.empty (module Int)))
        ~key:claim.helper_site.position ~data:(describe_site claim.helper_site))

let definition_sites ~source claims =
  List.fold claims ~init:(Map.empty (module String)) ~f:(record_definition ~source)

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

let quantified_failure source claim =
  let key = exemption_key ~source claim in
  Printf.sprintf
    "%s:%d sends `%s` from line %d into a Verdict claim, but that helper's `%s` can pass on an \
     empty population -- use the matching `Verdict.p_*` combinator, or make non-emptiness part of \
     the helper's passing result. If emptiness is the intended passing case, exempt `%s` by name \
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
         "exempted quantified helpers that no Verdict claim reaches any more -- drop them from the \
          exemption list: %s"
         (String.concat ~sep:", " (Set.to_list stale_quantified)))

let refuse_colliding_quantified ~fail colliding =
  if not (List.is_empty colliding) then
    fail
      (Printf.sprintf
         "exempted quantified helpers whose key names more than one definition, so one granted \
          exemption is silently covering helpers nobody read -- give the shadowing definitions \
          separate names, or hoist them into one: %s"
         (String.concat ~sep:", "
            (List.map colliding ~f:(fun (key, sites) ->
                 Printf.sprintf "%s (definitions at %s)" key (String.concat ~sep:", " sites)))))

let run_shadowed_quantified_control label fixture =
  let colliding =
    quantified_claims (Sources.structure_of fixture)
    |> definition_sites ~source:"fixture"
    |> colliding_exemptions
  in
  let two_definitions =
    match colliding with
    | [ ("fixture:close", [ first; second ]) ] -> not (String.equal first second)
    | _ -> false
  in
  if not two_definitions then
    eprintf "the %s fixture resolved to %s, not to one name with two definitions\n" label
      (String.concat ~sep:", "
         (List.map colliding ~f:(fun (key, sites) ->
              Printf.sprintf "%s (%s)" key (String.concat ~sep:", " sites))));
  let source = "test/operations/verdict_ratchet.ml" in
  let format =
    "exempted quantified helpers whose key names more than one definition, so one granted \
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
    (Printf.sprintf "a %s helper name resolves to two definitions, not one" label, two_definitions);
    (Printf.sprintf "refuses an exemption key that names both %s definitions" label, !refused);
  ]

let run_shadowed_quantified_controls () =
  run_shadowed_quantified_control "shadowed" shadowed_helper_fixture
  @ run_shadowed_quantified_control "same-line shadowed" same_line_shadowed_helper_fixture

let run_stale_quantified_control () =
  let source = "test/operations/verdict_ratchet.ml" in
  let format =
    "exempted quantified helpers that no Verdict claim reaches any more -- drop them from the \
     exemption list: %s"
  in
  let refused = ref false in
  let fail _message =
    refused := true;
    Test_utils.Refusal_control_manifest.observe_failure ~source ~format
  in
  refuse_stale_quantified ~fail (Set.singleton (module String) "fixture:stale");
  ("refuses a stale quantified-helper exemption", !refused)

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
  let quantified_definitions = ref (Map.empty (module String)) in
  let canaries = Map.of_alist_exn (module String) canary_sites in
  let data = Map.of_alist_exn (module String) data_sources in
  let exemptions_used = ref (Set.empty (module String)) in
  let canaries_found = ref (Set.empty (module String)) in
  let data_used = ref (Set.empty (module String)) in
  let literals = ref 0 and applied = ref 0 and offenders = ref 0 in
  let quantified_offenders = ref 0 in
  let control_results =
    run_quantified_helper_controls ()
    @ [ run_refusal_control (); run_stale_quantified_control () ]
    @ run_shadowed_quantified_controls ()
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
          let key = source ^ ":" ^ if computed then site.Scan.head else site.Scan.label in
          let where = Printf.sprintf "%s:%d" source site.Scan.line in
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
          else if (not computed) && Map.mem exemptions key then
            exemptions_used := Set.add !exemptions_used key
          else if computed && Map.mem computed_exemptions key then
            computed_used := Set.add !computed_used key
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
          let key = exemption_key ~source claim in
          if Map.mem quantified_exemptions key then (
            quantified_used := Set.add !quantified_used key;
            quantified_definitions := record_definition ~source !quantified_definitions claim)
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
  printf "\nHelper-wrapped quantified claims exempted because emptiness is their passing meaning:\n";
  if List.is_empty exempt_quantified_helpers then
    printf "  (none: every helper-wrapped quantifier in a claim must witness a population)\n"
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
  if not (Set.is_empty stale) then
    fail
      (Printf.sprintf
         "exempted literals that no source carries any more -- drop them from the exemption list: \
          %s"
         (String.concat ~sep:", " (Set.to_list stale)));
  refuse_stale_quantified ~fail stale_quantified;
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
  Verdict.p "every helper-wrapped quantified claim witnesses a non-empty population"
    (!quantified_offenders = 0);
  Verdict.p "the scan found every literal planted for it" (Set.is_empty missing);
  Verdict.p "every exemption on this check's lists is still earned"
    (Set.is_empty unread && Set.is_empty stale && Set.is_empty stale_quantified);
  Verdict.p "every exempted quantified helper is one definition, not a shared name"
    (List.is_empty colliding_quantified);
  (* What a blind walk cannot produce. Without these, "no offenders" and "read nothing" are the same
     result -- and the second is the one that arrives silently. *)
  Verdict.p "the walk read string literals out of these sources" (!literals > 0);
  Verdict.p "and placed some of them as arguments of a named function" (!applied > 0);
  Verdict.p "over more than one test directory" (List.length directories > 1);
  if not (Verdict.any_failed ()) then
    printf
      "\n\
       OK: test claims route through `Verdict`, and helper-wrapped quantifiers cannot pass on \
       nothing.\n";
  Test_utils.Refusal_control_manifest.print "verdict_ratchet.ml"
