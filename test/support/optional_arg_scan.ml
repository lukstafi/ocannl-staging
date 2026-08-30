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
type dsl_context = Outside_dsl | Op_dsl | Cd_dsl

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

let is_ignore ~unqualified_visible expression =
  match flatten_ident expression with
  | Some [ "ignore" ] -> unqualified_visible
  | Some [ "Stdlib"; "ignore" ] -> true
  | _ -> false

let binding_defines_standard_ignore binding =
  pattern_binds binding.pvb_pat "ignore"
  && List.equal String.equal (pattern_names binding.pvb_pat) [ "ignore" ]
  && match flatten_ident binding.pvb_expr with Some [ "Stdlib"; "ignore" ] -> true | _ -> false

let ignore_binding_state bindings =
  let bindings = List.filter bindings ~f:(fun binding -> pattern_binds binding.pvb_pat "ignore") in
  match bindings with
  | [] -> None
  | _ -> Some (not (List.for_all bindings ~f:binding_defines_standard_ignore))

let structure_item_may_shadow_ignore item =
  match item.pstr_desc with
  | Pstr_open { popen_expr = { pmod_desc = Pmod_ident { txt = Lident name; _ }; _ }; _ } ->
      not (String.equal name "Base" || String.equal name "Stdlib")
  | Pstr_open _ | Pstr_include _ -> true
  | Pstr_primitive { pval_name = { txt = "ignore"; _ }; _ } -> true
  | _ -> false

let values_bound_by_structure items =
  List.concat_map items ~f:(fun item ->
      match item.pstr_desc with
      | Pstr_value (_, bindings) ->
          List.concat_map bindings ~f:(fun binding -> pattern_names binding.pvb_pat)
      | Pstr_primitive value_description -> [ value_description.pval_name.txt ]
      | _ -> [])
  |> List.dedup_and_sort ~compare:String.compare

let module_values structure =
  let found = ref [] in
  let iterator =
    object
      inherit Ast_traverse.iter as super

      method! structure_item item =
        (match item.pstr_desc with
        | Pstr_module
            {
              pmb_name = { txt = Some module_name; _ };
              pmb_expr = { pmod_desc = Pmod_structure items; _ };
              _;
            } ->
            found := (module_name, values_bound_by_structure items) :: !found
        | _ -> ());
        super#structure_item item
    end
  in
  iterator#structure structure;
  !found

let module_expr_binds module_values module_expr name =
  match module_expr.pmod_desc with
  | Pmod_structure items -> List.mem (values_bound_by_structure items) name ~equal:String.equal
  | Pmod_ident { txt = Lident module_name; _ } ->
      List.Assoc.find module_values module_name ~equal:String.equal
      |> Option.value_map ~default:false ~f:(fun values -> List.mem values name ~equal:String.equal)
  | _ -> false

let direct_discard ~unqualified_ignore_visible name expression =
  match expression.pexp_desc with
  | Pexp_apply (callee, [ (Asttypes.Nolabel, argument) ]) ->
      is_ignore ~unqualified_visible:unqualified_ignore_visible callee
      && Option.value_map (direct_name argument) ~default:false ~f:(String.equal name)
  | Pexp_apply ({ pexp_desc = Pexp_ident { txt = Lident "|>"; _ }; _ }, [ (_, left); (_, right) ])
    ->
      Option.value_map (direct_name left) ~default:false ~f:(String.equal name)
      && is_ignore ~unqualified_visible:unqualified_ignore_visible right
  | Pexp_apply ({ pexp_desc = Pexp_ident { txt = Lident "@@"; _ }; _ }, [ (_, left); (_, right) ])
    ->
      is_ignore ~unqualified_visible:unqualified_ignore_visible left
      && Option.value_map (direct_name right) ~default:false ~f:(String.equal name)
  | _ -> false

type function_tail = Expression of expression | Cases of case list

let function_params expression =
  let rec peel params expression =
    match expression.pexp_desc with
    | Pexp_constraint (inner, _) | Pexp_coerce (inner, _, _) -> peel params inner
    | Pexp_function (more, _, body) -> (
        let params = params @ more in
        match body with
        | Pfunction_body inner -> peel params inner
        | Pfunction_cases (cases, _, _) -> (params, Cases cases))
    | _ -> (params, Expression expression)
  in
  peel [] expression

let rec function_paths ~ignore_is_shadowed expression =
  let params, tail = function_params expression in
  let rec returned_paths ~ignore_is_shadowed expression =
    match expression.pexp_desc with
    | Pexp_function _ -> function_paths ~ignore_is_shadowed expression
    | Pexp_constraint (inner, _) | Pexp_coerce (inner, _, _) ->
        returned_paths ~ignore_is_shadowed inner
    | Pexp_ifthenelse (_, then_, else_) ->
        returned_paths ~ignore_is_shadowed then_
        @ Option.value_map else_ ~default:[] ~f:(returned_paths ~ignore_is_shadowed)
    | Pexp_match (_, cases) ->
        List.concat_map cases ~f:(fun case ->
            returned_paths
              ~ignore_is_shadowed:(ignore_is_shadowed || pattern_binds case.pc_lhs "ignore")
              case.pc_rhs)
    | Pexp_try (body, cases) ->
        returned_paths ~ignore_is_shadowed body
        @ List.concat_map cases ~f:(fun case ->
            returned_paths
              ~ignore_is_shadowed:(ignore_is_shadowed || pattern_binds case.pc_lhs "ignore")
              case.pc_rhs)
    | Pexp_let (_, bindings, body) ->
        returned_paths
          ~ignore_is_shadowed:
            (Option.value (ignore_binding_state bindings) ~default:ignore_is_shadowed)
          body
    | Pexp_open (open_declaration, body) ->
        let opened_may_shadow =
          match open_declaration.popen_expr.pmod_desc with
          | Pmod_ident { txt = Lident name; _ } ->
              not (String.equal name "Base" || String.equal name "Stdlib")
          | _ -> true
        in
        returned_paths ~ignore_is_shadowed:(ignore_is_shadowed || opened_may_shadow) body
    | Pexp_letop { let_; ands; body } ->
        let binds_ignore =
          pattern_binds let_.pbop_pat "ignore"
          || List.exists ands ~f:(fun binding -> pattern_binds binding.pbop_pat "ignore")
        in
        returned_paths ~ignore_is_shadowed:(ignore_is_shadowed || binds_ignore) body
    | Pexp_letmodule (_, _, body) | Pexp_letexception (_, body) | Pexp_sequence (_, body) ->
        returned_paths ~ignore_is_shadowed body
    | _ -> []
  in
  let nested =
    match tail with
    | Expression body -> returned_paths ~ignore_is_shadowed body
    | Cases cases ->
        List.concat_map cases ~f:(fun case ->
            returned_paths
              ~ignore_is_shadowed:(ignore_is_shadowed || pattern_binds case.pc_lhs "ignore")
              case.pc_rhs)
  in
  (params, tail, ignore_is_shadowed) :: nested

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

(* The spec is the right operand's callee in [x ++ "spec" dims] / [x +* "spec" y], and its first
   direct argument in the inline-tensor form [x +* kernel "spec" dims]. Restricting generated reads
   to this exact operator position keeps a diagnostic string from lending an option a use. *)
let einsum_spec ~dsl expression =
  match expression.pexp_desc with
  | Pexp_apply (operator, [ (_, _left); (_, right) ]) ->
      let operator_kind =
        match operator.pexp_desc with
        | Pexp_ident { txt = Lident name; _ }
          when List.mem Einsum_parser.binary_operators_with_generated_specs name ~equal:String.equal
               && not (Poly.equal dsl Outside_dsl) ->
            `Binary
        | Pexp_ident { txt = Lident name; _ }
          when List.mem Einsum_parser.unary_operators_with_generated_specs name ~equal:String.equal
               && not (Poly.equal dsl Outside_dsl)
               || Poly.equal dsl Op_dsl
                  && String.equal name Einsum_parser.concat_operator_with_generated_specs ->
            `Unary
        | _ -> `Other
      in
      let spec =
        match (operator_kind, string_constant right, right.pexp_desc) with
        | `Unary, (Some _ as spec), _ -> spec
        | `Unary, None, Pexp_apply (callee, _) -> string_constant callee
        | `Unary, None, _ -> None
        | `Binary, None, Pexp_apply (callee, arguments) -> (
            match string_constant callee with
            | Some _ as spec -> spec
            | None -> (
                match arguments with (_, first) :: _ -> string_constant first | [] -> None))
        | (`Binary | `Other), _, _ -> None
      in
      Option.filter spec ~f:(fun spec -> String.contains spec '>')
  | _ -> None

let rec meaningfully_used ?(ignore_is_shadowed = false) ~module_values ~dsl name tail =
  let meaningful = ref false in
  let shadowed = ref false in
  let ignore_shadowed = ref ignore_is_shadowed in
  let dsl_mode = ref dsl in
  let within_shadow shadows f =
    let saved = !shadowed in
    shadowed := saved || shadows;
    f ();
    shadowed := saved
  in
  let within_dsl context f =
    let saved = !dsl_mode in
    dsl_mode := context;
    f ();
    dsl_mode := saved
  in
  let without_dsl f =
    let saved = !dsl_mode in
    dsl_mode := Outside_dsl;
    f ();
    dsl_mode := saved
  in
  let within_ignore_shadow shadows f =
    let saved = !ignore_shadowed in
    ignore_shadowed := saved || shadows;
    f ();
    ignore_shadowed := saved
  in
  let within_ignore_state state f =
    let saved = !ignore_shadowed in
    ignore_shadowed := state;
    f ();
    ignore_shadowed := saved
  in
  let iterator =
    object (self)
      inherit Ast_traverse.iter as super

      method! structure items =
        let rec walk = function
          | [] -> ()
          | { pstr_desc = Pstr_value (recursive, bindings); _ } :: rest ->
              let binds_name =
                List.exists bindings ~f:(fun binding -> pattern_binds binding.pvb_pat name)
              in
              let next_ignore = ignore_binding_state bindings in
              within_ignore_state
                (match (recursive, next_ignore) with
                | Recursive, Some state -> state
                | _ -> !ignore_shadowed)
                (fun () ->
                  within_shadow
                    (match recursive with Recursive -> binds_name | Nonrecursive -> false)
                    (fun () ->
                      List.iter bindings ~f:(fun binding -> self#expression binding.pvb_expr)));
              within_ignore_state (Option.value next_ignore ~default:!ignore_shadowed) (fun () ->
                  within_shadow binds_name (fun () -> walk rest))
          | item :: rest ->
              self#structure_item item;
              within_ignore_shadow (structure_item_may_shadow_ignore item) (fun () -> walk rest)
        in
        walk items

      method! case case =
        within_ignore_shadow (pattern_binds case.pc_lhs "ignore") (fun () ->
            within_shadow (pattern_binds case.pc_lhs name) (fun () ->
                Option.iter case.pc_guard ~f:self#expression;
                self#expression case.pc_rhs))

      method! expression expression =
        if not !shadowed then (
          if not (Poly.equal !dsl_mode Outside_dsl) then
            Option.iter (einsum_spec ~dsl:!dsl_mode expression) ~f:(fun spec ->
                if List.mem (generated_reads_in_einsum spec) name ~equal:String.equal then
                  meaningful := true);
          match expression.pexp_desc with
          | Pexp_extension ({ txt = "oc"; _ }, _) ->
              (* [%oc] is an anti-quotation boundary: the OCANNL PPXs preserve its payload. *)
              without_dsl (fun () -> super#expression expression)
          | Pexp_extension ({ txt = "op"; _ }, _) ->
              within_dsl Op_dsl (fun () -> super#expression expression)
          | Pexp_extension ({ txt = "cd"; _ }, _) ->
              within_dsl Cd_dsl (fun () -> super#expression expression)
          | Pexp_ident _ -> (
              match direct_name expression with
              | Some found when String.equal found name -> meaningful := true
              | _ -> ())
          | Pexp_apply _
            when direct_discard ~unqualified_ignore_visible:(not !ignore_shadowed) name expression
            ->
              ()
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
                                meaningfully_used ~ignore_is_shadowed:!ignore_shadowed
                                  ~module_values ~dsl:!dsl_mode bound (Expression body))
                          in
                          let used_in_recursive_group =
                            match recursive with
                            | Nonrecursive -> false
                            | Recursive ->
                                List.exists bindings ~f:(fun other ->
                                    List.exists bound_names ~f:(fun bound ->
                                        meaningfully_used ~ignore_is_shadowed:!ignore_shadowed
                                          ~module_values ~dsl:!dsl_mode bound
                                          (Expression other.pvb_expr)))
                          in
                          not (used_in_body || used_in_recursive_group)
                  in
                  if not discarded then
                    within_ignore_shadow
                      (match recursive with
                      | Recursive ->
                          List.exists bindings ~f:(fun binding ->
                              pattern_binds binding.pvb_pat "ignore")
                      | Nonrecursive -> false)
                      (fun () ->
                        within_shadow
                          (match recursive with Recursive -> binds | Nonrecursive -> false)
                          (fun () -> self#expression binding.pvb_expr)));
              within_ignore_shadow
                (List.exists bindings ~f:(fun binding -> pattern_binds binding.pvb_pat "ignore"))
                (fun () -> within_shadow binds (fun () -> self#expression body))
          | Pexp_function (params, _, body) ->
              let visible = ref true in
              let ignore_visible = ref (not !ignore_shadowed) in
              List.iter params ~f:(fun param ->
                  match param.pparam_desc with
                  | Pparam_newtype _ -> ()
                  | Pparam_val (_, default, pattern) ->
                      if !visible then
                        within_ignore_shadow (not !ignore_visible) (fun () ->
                            Option.iter default ~f:self#expression);
                      if !visible && pattern_binds pattern name then visible := false;
                      if pattern_binds pattern "ignore" then ignore_visible := false);
              if !visible then
                within_ignore_shadow (not !ignore_visible) (fun () -> self#function_body body)
          | Pexp_letop { let_; ands; body } ->
              self#expression let_.pbop_exp;
              List.iter ands ~f:(fun binding -> self#expression binding.pbop_exp);
              let binds =
                pattern_binds let_.pbop_pat name
                || List.exists ands ~f:(fun binding -> pattern_binds binding.pbop_pat name)
              in
              let binds_ignore =
                pattern_binds let_.pbop_pat "ignore"
                || List.exists ands ~f:(fun binding -> pattern_binds binding.pbop_pat "ignore")
              in
              within_ignore_shadow binds_ignore (fun () ->
                  within_shadow binds (fun () -> self#expression body))
          | Pexp_open (open_declaration, body) ->
              (* A local open can replace both unqualified [ignore] and the target name. *)
              self#module_expr open_declaration.popen_expr;
              within_ignore_shadow true (fun () ->
                  within_shadow (module_expr_binds module_values open_declaration.popen_expr name)
                    (fun () -> self#expression body))
          | Pexp_object class_structure ->
              within_ignore_shadow (pattern_binds class_structure.pcstr_self "ignore") (fun () ->
                  within_shadow (pattern_binds class_structure.pcstr_self name) (fun () ->
                      List.iter class_structure.pcstr_fields ~f:self#class_field))
          | Pexp_for (pattern, from_, to_, _, body) ->
              self#expression from_;
              self#expression to_;
              within_ignore_shadow (pattern_binds pattern "ignore") (fun () ->
                  within_shadow (pattern_binds pattern name) (fun () -> self#expression body))
          | _ -> super#expression expression)
    end
  in
  (match tail with
  | Expression expression -> iterator#expression expression
  | Cases cases -> iterator#cases cases);
  !meaningful

let implementation_of ~dsl ~module_values ~ignore_is_shadowed ~params ~position ~tail pattern =
  let names = pattern_names pattern in
  let name_used name =
    (* An earlier parameter is in scope in every later default until a later parameter shadows it;
       that shadowing parameter's own default is still evaluated before its pattern binds. *)
    let rec used_after ignore_is_shadowed = function
      | [] -> meaningfully_used ~ignore_is_shadowed ~module_values ~dsl name tail
      | param :: later -> (
          match param.pparam_desc with
          | Pparam_newtype _ -> used_after ignore_is_shadowed later
          | Pparam_val (_, default, pattern) ->
              let used_in_default =
                Option.value_map default ~default:false ~f:(fun expression ->
                    meaningfully_used ~ignore_is_shadowed ~module_values ~dsl name
                      (Expression expression))
              in
              used_in_default
              ||
              if pattern_binds pattern name then false
              else used_after (ignore_is_shadowed || pattern_binds pattern "ignore") later)
    in
    let ignore_is_shadowed =
      ignore_is_shadowed
      || List.take params (position + 1)
         |> List.exists ~f:(fun param ->
             match param.pparam_desc with
             | Pparam_val (_, _, pattern) -> pattern_binds pattern "ignore"
             | Pparam_newtype _ -> false)
    in
    used_after ignore_is_shadowed (List.drop params (position + 1))
  in
  if List.exists names ~f:name_used then Implemented else Unimplemented

let rec named_expressions pattern expression =
  match (pattern.ppat_desc, expression.pexp_desc) with
  | Ppat_var { txt; _ }, _ -> [ (txt, expression) ]
  | Ppat_constraint (inner, _), _ -> named_expressions inner expression
  | Ppat_alias (inner, { txt; _ }), _ -> (txt, expression) :: named_expressions inner expression
  | Ppat_tuple patterns, Pexp_tuple expressions when List.length patterns = List.length expressions
    ->
      List.map2_exn patterns expressions ~f:named_expressions |> List.concat
  | _ when List.is_empty (pattern_names pattern) -> []
  | _ ->
      failwith
        (Printf.sprintf
           "optional_arg_inventory: unsupported top-level binding pattern at line %d; split the \
            exported values into named bindings so optional arguments cannot be omitted"
           pattern.ppat_loc.loc_start.pos_lnum)

let args_in_source ~source content =
  let structure = structure_of content in
  let module_values = module_values structure in
  let found = ref [] in
  let local_expression_depth = ref 0 in
  let module_path = ref [] in
  let dsl_context = ref Outside_dsl in
  let top_ignore_shadowed = ref false in
  let within_module name f =
    let saved = !module_path in
    let saved_ignore = !top_ignore_shadowed in
    module_path := name :: saved;
    f ();
    module_path := saved;
    top_ignore_shadowed := saved_ignore
  in
  let qualify name = String.concat ~sep:"." (List.rev (name :: !module_path)) in
  let same_optional left right =
    String.equal left.source right.source
    && String.equal left.definition right.definition
    && String.equal left.label right.label
  in
  let add_optional into arg =
    match List.find !into ~f:(fun previous -> same_optional previous arg) with
    | None -> into := arg :: !into
    | Some previous ->
        let implementation =
          match (previous.implementation, arg.implementation) with
          | Implemented, _ | _, Implemented -> Implemented
          | Unimplemented, Unimplemented -> Unimplemented
        in
        into :=
          { arg with implementation }
          :: List.filter !into ~f:(fun candidate -> not (same_optional candidate arg))
  in
  let iterator =
    object (self)
      inherit Ast_traverse.iter as super

      method! structure items =
        if !local_expression_depth > 0 then super#structure items
        else
          List.iter items ~f:(fun item ->
              self#structure_item item;
              match item.pstr_desc with
              | Pstr_value (_, bindings) ->
                  Option.iter (ignore_binding_state bindings) ~f:(fun state ->
                      top_ignore_shadowed := state)
              | _ -> if structure_item_may_shadow_ignore item then top_ignore_shadowed := true)

      method! structure_item item =
        match item.pstr_desc with
        | Pstr_module { pmb_name = { txt = Some name; _ }; _ } ->
            within_module name (fun () -> super#structure_item item)
        | Pstr_extension (({ txt = ("op" | "cd") as extension; _ }, _), _) ->
            let saved = !dsl_context in
            dsl_context := if String.equal extension "op" then Op_dsl else Cd_dsl;
            Exn.protect
              ~f:(fun () -> super#structure_item item)
              ~finally:(fun () -> dsl_context := saved)
        | _ -> super#structure_item item

      method! expression expression =
        Int.incr local_expression_depth;
        Exn.protect
          ~f:(fun () -> super#expression expression)
          ~finally:(fun () -> Int.decr local_expression_depth)

      method! value_binding binding =
        if !local_expression_depth = 0 then (
          let named = named_expressions binding.pvb_pat binding.pvb_expr in
          let definitions =
            List.map named ~f:(fun (definition, _) -> qualify definition)
            |> List.dedup_and_sort ~compare:String.compare
          in
          (* A later source binding replaces the earlier exported value. Branches of this one
             binding still merge below because they describe result paths of the same value. *)
          found :=
            List.filter !found ~f:(fun arg ->
                not (List.mem definitions arg.definition ~equal:String.equal));
          let binding_found = ref [] in
          List.iter named ~f:(fun (definition, expression) ->
              List.iter (function_paths ~ignore_is_shadowed:!top_ignore_shadowed expression)
                ~f:(fun (params, tail, path_ignore_is_shadowed) ->
                  List.iteri params ~f:(fun position param ->
                      match param.pparam_desc with
                      | Pparam_val (Asttypes.Optional label, _, pattern) ->
                          let implementation =
                            implementation_of ~dsl:!dsl_context ~module_values
                              ~ignore_is_shadowed:path_ignore_is_shadowed ~params ~position ~tail
                              pattern
                          in
                          add_optional binding_found
                            { source; definition = qualify definition; label; implementation }
                      | _ -> ())));
          found := !binding_found @ !found);
        super#value_binding binding
    end
  in
  iterator#structure structure;
  List.rev !found
