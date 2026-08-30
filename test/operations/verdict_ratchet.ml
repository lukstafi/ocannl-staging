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

type helper_binding = {
  name : string;
  line : int;
  expression : expression;
  dependencies : Set.M(String).t;
  unguarded : quantifier list;
}

type quantified_claim = {
  helper : string;
  helper_line : int;
  claim_line : int;
  quantifiers : quantifier_kind list;
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
  | Pexp_let (_, _, body) -> returned_quantifiers body
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
         || is_name callee "<>" ->
      quantifiers_in expr
  | _ -> []

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
  | Pexp_apply (op, [ (Asttypes.Nolabel, left); (Asttypes.Nolabel, right) ])
    when is_name op "=" || is_name op ">" || is_name op ">=" -> (
      match
        (length_population left, int_literal right, int_literal left, length_population right)
      with
      | Some population, Some n, _, _ when n > 0 -> Set.singleton (module String) population
      | _, _, Some n, Some population when n > 0 -> Set.singleton (module String) population
      | _ -> none ())
  | Pexp_ifthenelse (condition, yes, Some no) when bool_literal no false ->
      Set.union (required_nonempty condition) (required_nonempty yes)
  | _ -> none ()

let names_in ?(positive_only = false) expr =
  let names = ref (Set.empty (module String)) in
  let positive = ref true in
  let iterator =
    object (self)
      inherit Ast_traverse.iter as super
      method! attribute _ = ()
      method! value_binding _ = ()

      method! expression expr =
        match expr.pexp_desc with
        | Pexp_function _ -> ()
        | Pexp_apply (callee, [ (Asttypes.Nolabel, argument) ]) when is_name callee "not" ->
            let previous = !positive in
            positive := not previous;
            self#expression argument;
            positive := previous
        | _ ->
            (match Sources.longident_of expr with
            | Some [ name ] when (not positive_only) || !positive -> names := Set.add !names name
            | _ -> ());
            super#expression expr
    end
  in
  iterator#expression expr;
  !names

let bindings_of structure =
  let found = ref [] in
  let iterator =
    object
      inherit Ast_traverse.iter as super
      method! attribute _ = ()

      method! value_binding value =
        (match helper_name value.pvb_pat with
        | None -> ()
        | Some name ->
            let guards = required_nonempty value.pvb_expr in
            let unguarded =
              if is_function value.pvb_expr || is_partial_quantifier value.pvb_expr then
                List.filter (returned_quantifiers value.pvb_expr) ~f:(fun quantifier ->
                    Set.is_empty quantifier.populations
                    || Set.is_empty (Set.inter guards quantifier.populations))
              else []
            in
            found :=
              {
                name;
                line = value.pvb_loc.loc_start.pos_lnum;
                expression = value.pvb_expr;
                dependencies = names_in (function_body value.pvb_expr);
                unguarded;
              }
              :: !found);
        super#value_binding value
    end
  in
  iterator#structure structure;
  List.rev !found

type claim_kind = P | Pf | Pass_fail | Claim | Claimf

let claim_kind_of_path path =
  match path with
  | ("Verdict" | "Ll_test") :: _ -> (
      match List.last path with
      | Some "p" -> Some P
      | Some "pf" -> Some Pf
      | Some "pass_fail" -> Some Pass_fail
      | Some "claim" -> Some Claim
      | Some "claimf" -> Some Claimf
      | _ -> None)
  | _ -> None

let claim_aliases bindings =
  let aliases = Hashtbl.create (module String) in
  let changed = ref true in
  while !changed do
    changed := false;
    List.iter bindings ~f:(fun binding ->
        if not (Hashtbl.mem aliases binding.name) then
          let kind =
            match Sources.longident_of binding.expression with
            | Some [ alias ] -> Hashtbl.find aliases alias
            | Some path -> claim_kind_of_path path
            | None -> None
          in
          Option.iter kind ~f:(fun kind ->
              changed := true;
              Hashtbl.set aliases ~key:binding.name ~data:kind))
  done;
  aliases

let quantified_claims structure =
  let bindings = bindings_of structure in
  let by_name = Hashtbl.create (module String) in
  List.iter bindings ~f:(fun binding -> Hashtbl.add_multi by_name ~key:binding.name ~data:binding);
  let aliases = claim_aliases bindings in
  let origins ~before names =
    let rec visit seen ~before name =
      if Set.mem seen name then []
      else
        let seen = Set.add seen name in
        let visible =
          Hashtbl.find_multi by_name name |> List.filter ~f:(fun binding -> binding.line <= before)
        in
        let visible =
          match List.max_elt visible ~compare:(fun a b -> Int.compare a.line b.line) with
          | None -> []
          | Some latest -> List.filter visible ~f:(fun binding -> binding.line = latest.line)
        in
        visible
        |> List.concat_map ~f:(fun binding ->
            let direct = if List.is_empty binding.unguarded then [] else [ binding ] in
            direct
            @ (Set.to_list binding.dependencies
              |> List.concat_map ~f:(visit seen ~before:binding.line)))
    in
    Set.to_list names
    |> List.concat_map ~f:(visit (Set.empty (module String)) ~before)
    |> List.dedup_and_sort ~compare:(fun a b -> Int.compare a.line b.line)
  in
  let found = ref [] in
  let iterator =
    object
      inherit Ast_traverse.iter as super
      method! attribute _ = ()

      method! expression expr =
        (match expr.pexp_desc with
        | Pexp_apply (callee, arguments) ->
            let kind =
              match Sources.longident_of callee with
              | Some [ alias ] -> Hashtbl.find aliases alias
              | Some path -> claim_kind_of_path path
              | None -> None
            in
            Option.iter kind ~f:(fun _ ->
                match List.last (unlabelled arguments) with
                | None -> ()
                | Some boolean ->
                    List.iter
                      (origins ~before:expr.pexp_loc.loc_start.pos_lnum
                         (names_in ~positive_only:true boolean))
                      ~f:(fun binding ->
                        found :=
                          {
                            helper = binding.name;
                            helper_line = binding.line;
                            claim_line = expr.pexp_loc.loc_start.pos_lnum;
                            quantifiers =
                              List.map binding.unguarded ~f:(fun quantifier -> quantifier.kind)
                              |> List.dedup_and_sort ~compare:Poly.compare;
                          }
                          :: !found))
        | _ -> ());
        super#expression expr
    end
  in
  iterator#structure structure;
  List.rev !found
  |> List.dedup_and_sort ~compare:(fun a b ->
      match Int.compare a.claim_line b.claim_line with
      | 0 -> (
          match Int.compare a.helper_line b.helper_line with
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
   decide what empty means, and every call reaches the same decision. *)
let exempt_quantified_helpers : (string * string) list = []

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
    ( "refuses a sibling for_all helper through an intermediate result binding",
      {ocaml|let agrees xs = List.for_all xs ~f:Fn.id
let ok = agrees samples
let () = Verdict.claim "every sample agrees" ok|ocaml},
      [ "agrees" ] );
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
  let key = source ^ ":" ^ claim.helper in
  Printf.sprintf
    "%s:%d sends `%s` from line %d into a Verdict claim, but that helper's `%s` can pass on an \
     empty population -- use the matching `Verdict.p_*` combinator, or make non-emptiness part of \
     the helper's passing result. If emptiness is the intended passing case, exempt `%s` by name \
     in verdict_ratchet.ml and say why"
    source claim.claim_line claim.helper claim.helper_line
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
  let canaries = Map.of_alist_exn (module String) canary_sites in
  let data = Map.of_alist_exn (module String) data_sources in
  let exemptions_used = ref (Set.empty (module String)) in
  let canaries_found = ref (Set.empty (module String)) in
  let data_used = ref (Set.empty (module String)) in
  let literals = ref 0 and applied = ref 0 and offenders = ref 0 in
  let quantified_offenders = ref 0 in
  let control_results = run_quantified_helper_controls () @ [ run_refusal_control () ] in
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
          let key = source ^ ":" ^ claim.helper in
          if Map.mem quantified_exemptions key then quantified_used := Set.add !quantified_used key
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
  if not (Set.is_empty stale) then
    fail
      (Printf.sprintf
         "exempted literals that no source carries any more -- drop them from the exemption list: \
          %s"
         (String.concat ~sep:", " (Set.to_list stale)));
  if not (Set.is_empty stale_quantified) then
    fail
      (Printf.sprintf
         "exempted quantified helpers that no Verdict claim reaches any more -- drop them from the \
          exemption list: %s"
         (String.concat ~sep:", " (Set.to_list stale_quantified)));
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
  (* What a blind walk cannot produce. Without these, "no offenders" and "read nothing" are the same
     result -- and the second is the one that arrives silently. *)
  Verdict.p "the walk read string literals out of these sources" (!literals > 0);
  Verdict.p "and placed some of them as arguments of a named function" (!applied > 0);
  Verdict.p "over more than one test directory" (List.length directories > 1);
  if not (Verdict.any_failed ()) then
    printf
      "\n\
       OK: test claims route through `Verdict`, and helper-wrapped quantifiers cannot pass on \
       nothing.\n"
