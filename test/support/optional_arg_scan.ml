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

let pattern_names pattern =
  let found = ref [] in
  let iterator =
    object
      inherit Ast_traverse.iter as super

      method! pattern pattern =
        (match pattern.ppat_desc with Ppat_var { txt; _ } -> found := txt :: !found | _ -> ());
        super#pattern pattern
    end
  in
  iterator#pattern pattern;
  List.dedup_and_sort !found ~compare:String.compare

type throwaway = Not_throwaway | Wildcard | Named_throwaway of string list

let rec wildcard_pattern pattern =
  match pattern.ppat_desc with
  | Ppat_any -> true
  | Ppat_constraint (inner, _) -> wildcard_pattern inner
  | _ -> false

let throwaway_pattern pattern =
  if wildcard_pattern pattern then Wildcard
  else
    let names = pattern_names pattern in
    if
      (not (List.is_empty names))
      && List.for_all names ~f:(fun name -> String.is_prefix name ~prefix:"_")
    then Named_throwaway names
    else Not_throwaway

let flatten_ident expression =
  match expression.pexp_desc with
  | Pexp_ident { txt; _ } -> ( try Some (Longident.flatten_exn txt) with _ -> None)
  | _ -> None

let rec direct_name expression =
  match expression.pexp_desc with
  | Pexp_constraint (inner, _) | Pexp_coerce (inner, _, _) -> direct_name inner
  | _ -> ( match flatten_ident expression with Some [ name ] -> Some name | _ -> None)

let path_ends_in expression wanted =
  match flatten_ident expression with
  | Some path -> Option.value_map (List.last path) ~default:false ~f:(String.equal wanted)
  | None -> false

let direct_discard name expression =
  match expression.pexp_desc with
  | Pexp_apply (callee, [ (Asttypes.Nolabel, argument) ]) ->
      path_ends_in callee "ignore"
      && Option.value_map (direct_name argument) ~default:false ~f:(String.equal name)
  | Pexp_apply (operator, [ (_, left); (_, right) ]) when path_ends_in operator "|>" ->
      Option.value_map (direct_name left) ~default:false ~f:(String.equal name)
      && path_ends_in right "ignore"
  | Pexp_apply (operator, [ (_, left); (_, right) ]) when path_ends_in operator "@@" ->
      path_ends_in left "ignore"
      && Option.value_map (direct_name right) ~default:false ~f:(String.equal name)
  | _ -> false

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
let generated_reads_in_einsum spec =
  let found = ref [] in
  let add_nonliteral value =
    match Int.of_string value with _ -> () | exception _ -> found := value :: !found
  in
  let add_axis = function
    | Einsum_parser.Affine_spec { stride; conv; _ } ->
        add_nonliteral stride;
        Option.iter conv ~f:(fun conv ->
            add_nonliteral conv.dilation;
            match conv.use_padding with `Unspecified -> found := "use_padding" :: !found | _ -> ())
    | Label _ | Fixed_index _ | Concat_spec _ -> ()
  in
  let add_parsed parsed =
    List.iter
      (parsed.Einsum_parser.given_batch @ parsed.given_input @ parsed.given_output
     @ parsed.given_beg_batch @ parsed.given_beg_input @ parsed.given_beg_output)
      ~f:add_axis
  in
  (try
     let rhs, lhs = Einsum_parser.einsum_of_spec spec in
     List.iter rhs ~f:add_parsed;
     add_parsed lhs
   with Einsum_parser.Parse_error _ -> (
     (* This mirrors the PPX fallback for a one-sided axis-label specification. *)
     try add_parsed (Einsum_parser.axis_labels_of_spec spec)
     with Einsum_parser.Parse_error _ -> ()));
  List.dedup_and_sort !found ~compare:String.compare

let string_constant expression =
  match expression.pexp_desc with
  | Pexp_constant (Pconst_string (text, _, _)) -> Some text
  | _ -> None

let einsum_operators = [ "+*"; "@^+"; "+++"; "++"; "@^^" ]

(* The spec is the right operand's callee in [x ++ "spec" dims] / [x +* "spec" y], and its first
   direct argument in the inline-tensor form [x +* kernel "spec" dims]. Restricting generated reads
   to this exact operator position keeps a diagnostic string from lending an option a use. *)
let einsum_spec expression =
  match expression.pexp_desc with
  | Pexp_apply (operator, [ (_, _left); (_, right) ]) -> (
      let is_einsum =
        match flatten_ident operator with
        | Some path ->
            Option.value_map (List.last path) ~default:false ~f:(fun name ->
                List.mem einsum_operators name ~equal:String.equal)
        | None -> false
      in
      if not is_einsum then None
      else
        match string_constant right with
        | Some _ as spec -> spec
        | None -> (
            match right.pexp_desc with
            | Pexp_apply (callee, arguments) -> (
                match string_constant callee with
                | Some _ as spec -> spec
                | None -> List.find_map arguments ~f:(fun (_, arg) -> string_constant arg))
            | _ -> None))
  | _ -> None

let rec meaningfully_used ~dsl name tail =
  let meaningful = ref false in
  let shadowed = ref false in
  let dsl_mode = ref dsl in
  let within_shadow shadows f =
    let saved = !shadowed in
    shadowed := saved || shadows;
    f ();
    shadowed := saved
  in
  let within_dsl enabled f =
    let saved = !dsl_mode in
    dsl_mode := saved || enabled;
    f ();
    dsl_mode := saved
  in
  let iterator =
    object (self)
      inherit Ast_traverse.iter as super

      method! case case =
        within_shadow (pattern_binds case.pc_lhs name) (fun () ->
            Option.iter case.pc_guard ~f:self#expression;
            self#expression case.pc_rhs)

      method! expression expression =
        if not !shadowed then (
          if !dsl_mode then
            Option.iter (einsum_spec expression) ~f:(fun spec ->
                if List.mem (generated_reads_in_einsum spec) name ~equal:String.equal then
                  meaningful := true);
          match expression.pexp_desc with
          | Pexp_extension ({ txt = "op" | "cd"; _ }, _) ->
              within_dsl true (fun () -> super#expression expression)
          | Pexp_ident _ -> (
              match direct_name expression with
              | Some found when String.equal found name -> meaningful := true
              | _ -> ())
          | Pexp_apply _ when direct_discard name expression -> ()
          | Pexp_let (recursive, bindings, body) ->
              let binds =
                List.exists bindings ~f:(fun binding -> pattern_binds binding.pvb_pat name)
              in
              List.iter bindings ~f:(fun binding ->
                  let discarded =
                    if
                      not
                        (Option.value_map (direct_name binding.pvb_expr) ~default:false
                           ~f:(String.equal name))
                    then false
                    else
                      match throwaway_pattern binding.pvb_pat with
                      | Not_throwaway -> false
                      | Wildcard -> true
                      | Named_throwaway bound_names ->
                          let used_in_body =
                            List.exists bound_names ~f:(fun bound ->
                                meaningfully_used ~dsl:!dsl_mode bound (Expression body))
                          in
                          let used_in_recursive_group =
                            match recursive with
                            | Nonrecursive -> false
                            | Recursive ->
                                List.exists bindings ~f:(fun other ->
                                    List.exists bound_names ~f:(fun bound ->
                                        meaningfully_used ~dsl:!dsl_mode bound
                                          (Expression other.pvb_expr)))
                          in
                          not (used_in_body || used_in_recursive_group)
                  in
                  if not discarded then
                    within_shadow
                      (match recursive with Recursive -> binds | Nonrecursive -> false)
                      (fun () -> self#expression binding.pvb_expr));
              within_shadow binds (fun () -> self#expression body)
          | Pexp_function (params, _, body) ->
              let visible = ref true in
              List.iter params ~f:(fun param ->
                  match param.pparam_desc with
                  | Pparam_newtype _ -> ()
                  | Pparam_val (_, default, pattern) ->
                      if !visible then Option.iter default ~f:self#expression;
                      if !visible && pattern_binds pattern name then visible := false);
              if !visible then self#function_body body
          | Pexp_letop { let_; ands; body } ->
              self#expression let_.pbop_exp;
              List.iter ands ~f:(fun binding -> self#expression binding.pbop_exp);
              let binds =
                pattern_binds let_.pbop_pat name
                || List.exists ands ~f:(fun binding -> pattern_binds binding.pbop_pat name)
              in
              within_shadow binds (fun () -> self#expression body)
          | Pexp_for (pattern, from_, to_, _, body) ->
              self#expression from_;
              self#expression to_;
              within_shadow (pattern_binds pattern name) (fun () -> self#expression body)
          | _ -> super#expression expression)
    end
  in
  (match tail with
  | Expression expression -> iterator#expression expression
  | Cases cases -> iterator#cases cases);
  !meaningful

let implementation_of ~dsl ~params ~position ~tail pattern =
  let names = pattern_names pattern in
  let name_used name =
    (* An earlier parameter is in scope in every later default until a later parameter shadows it;
       that shadowing parameter's own default is still evaluated before its pattern binds. *)
    let rec used_after = function
      | [] -> meaningfully_used ~dsl name tail
      | param :: later -> (
          match param.pparam_desc with
          | Pparam_newtype _ -> used_after later
          | Pparam_val (_, default, pattern) ->
              let used_in_default =
                Option.value_map default ~default:false ~f:(fun expression ->
                    meaningfully_used ~dsl name (Expression expression))
              in
              used_in_default || if pattern_binds pattern name then false else used_after later)
    in
    used_after (List.drop params (position + 1))
  in
  if List.exists names ~f:name_used then Implemented else Unimplemented

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
              List.iteri params ~f:(fun position param ->
                  match param.pparam_desc with
                  | Pparam_val (Asttypes.Optional label, _, pattern) ->
                      let implementation =
                        implementation_of ~dsl:(!dsl_extension_depth > 0) ~params ~position ~tail
                          pattern
                      in
                      found :=
                        { source; definition = qualify definition; label; implementation } :: !found
                  | _ -> ());
              super#value_binding binding
    end
  in
  iterator#structure (structure_of content);
  List.rev !found
