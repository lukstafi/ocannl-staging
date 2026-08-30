(** The honesty rule for optional arguments accepted by [lib/] entry points (gh-ocannl-811).

    An optional argument that is deliberately accepted before it is implemented must say so in the
    caller-visible label: [?_feature], not [?feature]. OCaml's unused-variable warning catches an
    ordinary argument that is never mentioned, but it cannot distinguish a real use from the two
    conventional ways of silencing that warning -- [let _ = feature] and [ignore feature]. This
    reader makes that distinction over the parsed source.

    The reverse direction matters too. Once [_feature] starts affecting the result, the underscore
    has gone stale and the public type still says "not implemented". Such an argument must be
    renamed back in the same change that implements it.

    This is deliberately a source rule, not an attempt to prove semantic dependence. An arbitrary
    expression can consume a value and still arrange to have no effect. The rule closes the silent
    accepted-and-discarded shape that shipped in [Nn_blocks], while executed tests remain the oracle
    for optimizer plumbing whose argument can be forwarded incorrectly without any discard in the
    source. *)

open Base
open Ppxlib.Parsetree
module Ast_traverse = Ppxlib.Ast_traverse
module Asttypes = Ppxlib.Asttypes
module Longident = Ppxlib.Longident
module Parse = Ppxlib.Parse

type implementation = Implemented | Unimplemented

type optional_arg = {
  source : string;
  definition : string;
  label : string;
  implementation : implementation;
}

let implementation_name = function Implemented -> "implemented" | Unimplemented -> "unimplemented"
let caller_visible_unimplemented label = String.is_prefix label ~prefix:"_"

let honest arg =
  Bool.equal
    (caller_visible_unimplemented arg.label)
    (match arg.implementation with Implemented -> false | Unimplemented -> true)

let render arg =
  Printf.sprintf "%s:%s ?%s -- %s" arg.source arg.definition arg.label
    (implementation_name arg.implementation)

let structure_of content = Parse.implementation (Lexing.from_string content)

let rec pattern_name pattern =
  match pattern.ppat_desc with
  | Ppat_var { txt; _ } -> Some txt
  | Ppat_constraint (inner, _) | Ppat_alias (inner, _) -> pattern_name inner
  | _ -> None

let pattern_binds pattern name =
  let found = ref false in
  let iterator =
    object
      inherit Ast_traverse.iter as super

      method! pattern pattern =
        (match pattern.ppat_desc with
        | Ppat_var { txt; _ } when String.equal txt name -> found := true
        | _ -> ());
        super#pattern pattern
    end
  in
  iterator#pattern pattern;
  !found

let rec wildcard_pattern pattern =
  match pattern.ppat_desc with
  | Ppat_any -> true
  | Ppat_constraint (inner, _) | Ppat_alias (inner, _) -> wildcard_pattern inner
  | _ -> false

let flatten_ident expression =
  match expression.pexp_desc with
  | Pexp_ident { txt; _ } -> ( try Some (Longident.flatten_exn txt) with _ -> None)
  | _ -> None

let rec direct_name expression =
  match expression.pexp_desc with
  | Pexp_constraint (inner, _) | Pexp_coerce (inner, _, _) -> direct_name inner
  | _ -> ( match flatten_ident expression with Some [ name ] -> Some name | _ -> None)

type function_tail = Expression of expression | Cases of case list

let function_params expression =
  let rec peel params expression =
    match expression.pexp_desc with
    | Pexp_function (more, _, body) -> (
        let params = params @ more in
        match body with
        | Pfunction_body inner -> peel params inner
        | Pfunction_cases (cases, _, _) -> (params, Cases cases))
    | _ -> (params, Expression expression)
  in
  peel [] expression

(** Whether [name] is used for anything except a direct discard in [tail]. The small lexical-scope
    walk is what keeps a nested [let name = ...] or [fun name -> ...] from lending a false use to
    the outer optional argument. *)
let is_word_char = function 'a' .. 'z' | 'A' .. 'Z' | '0' .. '9' | '_' -> true | _ -> false

let mentions_word text word =
  String.substr_index_all text ~may_overlap:false ~pattern:word
  |> List.exists ~f:(fun index ->
      let before = index = 0 || not (is_word_char text.[index - 1]) in
      let after_index = index + String.length word in
      let after = after_index = String.length text || not (is_word_char text.[after_index]) in
      before && after)

(* The legacy convolution spelling omits the padding marker: [stride*o+k], against explicit
   [stride*o<+k] / [stride*o=+k]. The ppx turns that omission into a runtime read of [use_padding]
   (tensor/ppx_shared.ml), so it is a source-level use even though the identifier is generated only
   after this scan runs. *)
let has_unmarked_convolution text =
  String.to_list text
  |> List.fold ~init:(None, false) ~f:(fun (previous_non_space, found) char ->
      if found then (previous_non_space, true)
      else if Char.is_whitespace char then (previous_non_space, false)
      else if Char.equal char '+' then
        ( Some char,
          not
            (Option.value_map previous_non_space ~default:false ~f:(fun previous ->
                 Char.equal previous '<' || Char.equal previous '=')) )
      else (Some char, false))
  |> snd

let dsl_generated_use name tail =
  let found = ref false in
  let iterator =
    object
      inherit Ast_traverse.iter as super

      method! expression expression =
        (match expression.pexp_desc with
        | Pexp_constant (Pconst_string (text, _, _))
          when mentions_word text name
               || (String.equal name "use_padding" && has_unmarked_convolution text) ->
            found := true
        | _ -> ());
        super#expression expression
    end
  in
  (match tail with
  | Expression expression -> iterator#expression expression
  | Cases cases -> iterator#cases cases);
  !found

let meaningfully_used ~dsl name tail =
  let meaningful = ref false in
  let shadowed = ref false in
  let within_shadow shadows f =
    let saved = !shadowed in
    shadowed := saved || shadows;
    f ();
    shadowed := saved
  in
  let iterator =
    object (self)
      inherit Ast_traverse.iter as super

      method! case case =
        within_shadow (pattern_binds case.pc_lhs name) (fun () ->
            Option.iter case.pc_guard ~f:self#expression;
            self#expression case.pc_rhs)

      method! expression expression =
        if not !shadowed then
          match expression.pexp_desc with
          | Pexp_ident _ -> (
              match direct_name expression with
              | Some found when String.equal found name -> meaningful := true
              | _ -> ())
          | Pexp_apply (callee, [ (Asttypes.Nolabel, argument) ]) -> (
              match (flatten_ident callee, direct_name argument) with
              | Some path, Some found
                when String.equal found name
                     && Option.value_map (List.last path) ~default:false ~f:(String.equal "ignore")
                ->
                  ()
              | _ -> super#expression expression)
          | Pexp_let (recursive, bindings, body) ->
              let binds =
                List.exists bindings ~f:(fun binding -> pattern_binds binding.pvb_pat name)
              in
              List.iter bindings ~f:(fun binding ->
                  let discarded =
                    wildcard_pattern binding.pvb_pat
                    && Option.value_map (direct_name binding.pvb_expr) ~default:false
                         ~f:(String.equal name)
                  in
                  if not discarded then
                    within_shadow
                      (match recursive with Recursive -> binds | Nonrecursive -> false)
                      (fun () -> self#expression binding.pvb_expr));
              within_shadow binds (fun () -> self#expression body)
          | Pexp_function (params, _, _) ->
              let binds =
                List.exists params ~f:(fun param ->
                    match param.pparam_desc with
                    | Pparam_val (_, _, pattern) -> pattern_binds pattern name
                    | Pparam_newtype _ -> false)
              in
              within_shadow binds (fun () -> super#expression expression)
          | Pexp_for (pattern, from_, to_, _, body) ->
              self#expression from_;
              self#expression to_;
              within_shadow (pattern_binds pattern name) (fun () -> self#expression body)
          | _ -> super#expression expression
    end
  in
  (match tail with
  | Expression expression -> iterator#expression expression
  | Cases cases -> iterator#cases cases);
  !meaningful || (dsl && dsl_generated_use name tail)

let args_in_source ~source content =
  let found = ref [] in
  let local_expression_depth = ref 0 in
  let module_path = ref [] in
  let dsl_extension_depth = ref 0 in
  let within_module name f =
    let saved = !module_path in
    module_path := name :: saved;
    f ();
    module_path := saved
  in
  let qualify name = String.concat ~sep:"." (List.rev (name :: !module_path)) in
  let iterator =
    object
      inherit Ast_traverse.iter as super

      method! structure_item item =
        match item.pstr_desc with
        | Pstr_module { pmb_name = { txt = Some name; _ }; _ } ->
            within_module name (fun () -> super#structure_item item)
        | Pstr_extension (({ txt = "op" | "cd"; _ }, _), _) ->
            Int.incr dsl_extension_depth;
            Exn.protect
              ~f:(fun () -> super#structure_item item)
              ~finally:(fun () -> Int.decr dsl_extension_depth)
        | _ -> super#structure_item item

      method! expression expression =
        Int.incr local_expression_depth;
        Exn.protect
          ~f:(fun () -> super#expression expression)
          ~finally:(fun () -> Int.decr local_expression_depth)

      method! value_binding binding =
        if !local_expression_depth = 0 then
          match pattern_name binding.pvb_pat with
          | None -> ()
          | Some definition ->
              let params, tail = function_params binding.pvb_expr in
              List.iter params ~f:(fun param ->
                  match param.pparam_desc with
                  | Pparam_val (Asttypes.Optional label, _, pattern) ->
                      let implementation =
                        match pattern_name pattern with
                        | Some name when meaningfully_used ~dsl:(!dsl_extension_depth > 0) name tail
                          ->
                            Implemented
                        | _ -> Unimplemented
                      in
                      found :=
                        { source; definition = qualify definition; label; implementation } :: !found
                  | _ -> ());
              super#value_binding binding
    end
  in
  iterator#structure (structure_of content);
  List.rev !found
