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
type namespace = Value | Class

type optional_arg = {
  source : string;
  definition : string;
  label : string;
  implementation : implementation;
  namespace : namespace;
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
let signature_of content = Parse.interface (Lexing.from_string content)

let definition_owned_by owner definition =
  String.equal owner definition
  || String.is_prefix definition ~prefix:(owner ^ ".")
  || String.is_prefix definition ~prefix:(owner ^ "#")

type public_surface = {
  public_values : string list;
  public_classes : string list;
  opaque_modules : string list;
  include_all : bool;
}

let public_surface_of_interface content =
  let values = ref [] in
  let classes = ref [] in
  let opaque_modules = ref [] in
  let include_all = ref false in
  let qualify path name = String.concat ~sep:"." (path @ [ name ]) in
  let rec module_type path module_type_expr =
    match module_type_expr.pmty_desc with
    | Pmty_signature items -> signature path items
    | Pmty_with (inner, _) -> module_type path inner
    | _ -> opaque_modules := String.concat ~sep:"." path :: !opaque_modules
  and signature path items =
    List.iter items ~f:(fun item ->
        match item.psig_desc with
        | Psig_value value -> values := qualify path value.pval_name.txt :: !values
        | Psig_class declarations ->
            List.iter declarations ~f:(fun declaration ->
                classes := qualify path declaration.pci_name.txt :: !classes)
        | Psig_module declaration ->
            Option.iter declaration.pmd_name.txt ~f:(fun name ->
                module_type (path @ [ name ]) declaration.pmd_type)
        | Psig_recmodule declarations ->
            List.iter declarations ~f:(fun declaration ->
                Option.iter declaration.pmd_name.txt ~f:(fun name ->
                    module_type (path @ [ name ]) declaration.pmd_type))
        | Psig_include declaration -> (
            match declaration.pincl_mod.pmty_desc with
            | Pmty_signature _ | Pmty_with _ -> module_type path declaration.pincl_mod
            | _ -> include_all := true)
        | _ -> ())
  in
  signature [] (signature_of content);
  {
    public_values = List.dedup_and_sort !values ~compare:String.compare;
    public_classes = List.dedup_and_sort !classes ~compare:String.compare;
    opaque_modules = List.dedup_and_sort !opaque_modules ~compare:String.compare;
    include_all = !include_all;
  }

let exposed_by surface arg =
  surface.include_all
  || List.exists surface.opaque_modules ~f:(fun owner -> definition_owned_by owner arg.definition)
  ||
  let owners =
    match arg.namespace with Value -> surface.public_values | Class -> surface.public_classes
  in
  List.exists owners ~f:(fun owner -> definition_owned_by owner arg.definition)

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

let is_ignore ~unqualified_visible ~qualified_standard_visible expression =
  match flatten_ident expression with
  | Some [ "ignore" ] -> unqualified_visible
  | Some [ (("Base" | "Stdlib") as qualifier); "ignore" ] ->
      qualified_standard_visible qualifier expression.pexp_loc.loc_start.pos_cnum
  | _ -> false

let binding_defines_standard_ignore ?(qualified_standard_visible = fun _qualifier _position -> true)
    binding =
  pattern_binds binding.pvb_pat "ignore"
  && List.equal String.equal (pattern_names binding.pvb_pat) [ "ignore" ]
  && is_ignore ~unqualified_visible:false ~qualified_standard_visible binding.pvb_expr

let ignore_binding_state ?qualified_standard_visible bindings =
  let bindings = List.filter bindings ~f:(fun binding -> pattern_binds binding.pvb_pat "ignore") in
  match bindings with
  | [] -> None
  | _ ->
      Some
        (not
           (List.for_all bindings ~f:(fun binding ->
                binding_defines_standard_ignore ?qualified_standard_visible binding)))

let direct_values_bound_by_structure items =
  List.concat_map items ~f:(fun item ->
      match item.pstr_desc with
      | Pstr_value (_, bindings) ->
          List.concat_map bindings ~f:(fun binding -> pattern_names binding.pvb_pat)
      | Pstr_primitive value_description -> [ value_description.pval_name.txt ]
      | _ -> [])
  |> List.dedup_and_sort ~compare:String.compare

type module_reference = { path : string list; scope : string list; position : int }

type module_export = {
  path : string list;
  position : int;
  values : string list;
  references : module_reference list;
}

let ident_path ident = try Some (Longident.flatten_exn ident) with _ -> None

let rec module_expr_exports ~scope module_expr =
  match module_expr.pmod_desc with
  | Pmod_structure items -> structure_exports ~scope items
  | Pmod_ident { txt; _ } ->
      ( [],
        Option.value_map (ident_path txt) ~default:[] ~f:(fun path ->
            [ { path; scope; position = module_expr.pmod_loc.loc_start.pos_cnum } ]) )
  | Pmod_constraint (inner, _) -> module_expr_exports ~scope inner
  | _ -> ([], [])

and structure_exports ~scope items =
  let included_values, references =
    List.fold items ~init:([], []) ~f:(fun (values, references) item ->
        match item.pstr_desc with
        | Pstr_include include_declaration ->
            let more_values, more_references =
              module_expr_exports ~scope include_declaration.pincl_mod
            in
            (more_values @ values, more_references @ references)
        | _ -> (values, references))
  in
  ( direct_values_bound_by_structure items @ included_values
    |> List.dedup_and_sort ~compare:String.compare,
    references )

let module_values structure =
  let found = ref [] in
  let module_path = ref [] in
  let within_module name f =
    let saved = !module_path in
    module_path := saved @ [ name ];
    Exn.protect ~f ~finally:(fun () -> module_path := saved)
  in
  let record name binding =
    let path = !module_path @ [ name ] in
    let values, references =
      match binding.pmb_expr.pmod_desc with
      | Pmod_structure _ | Pmod_constraint _ -> module_expr_exports ~scope:path binding.pmb_expr
      | Pmod_ident _ -> module_expr_exports ~scope:!module_path binding.pmb_expr
      | _ -> ([], [])
    in
    found := { path; position = binding.pmb_loc.loc_start.pos_cnum; values; references } :: !found
  in
  let iterator =
    object (self)
      inherit Ast_traverse.iter as super

      method! structure_item item =
        match item.pstr_desc with
        | Pstr_module ({ pmb_name = { txt = Some module_name; _ }; _ } as binding) ->
            record module_name binding;
            within_module module_name (fun () -> self#module_expr binding.pmb_expr)
        | Pstr_recmodule bindings ->
            List.iter bindings ~f:(fun binding ->
                match binding.pmb_name.txt with
                | Some module_name ->
                    record module_name binding;
                    within_module module_name (fun () -> self#module_expr binding.pmb_expr)
                | None -> self#module_expr binding.pmb_expr)
        | _ -> super#structure_item item
    end
  in
  iterator#structure structure;
  !found

let module_candidates ~module_path ~opened_paths path =
  let rec ancestor_candidates prefix =
    match List.drop_last prefix with
    | None -> []
    | Some parent -> (parent @ path) :: ancestor_candidates parent
  in
  let opened_candidates = List.map opened_paths ~f:(fun opened -> opened @ path) in
  match module_path with
  | [] -> opened_candidates @ [ path ]
  | _ -> ((module_path @ path) :: opened_candidates) @ ancestor_candidates module_path

let lookup_module_exports module_values ~module_path ~opened_paths ~position path =
  module_candidates ~module_path ~opened_paths path
  |> List.find_map ~f:(fun candidate ->
      List.filter module_values ~f:(fun export ->
          List.equal String.equal export.path candidate && export.position <= position)
      |> List.max_elt ~compare:(fun left right -> Int.compare left.position right.position))
  |> Option.to_list

let lookup_module_values module_values ~module_path ~opened_paths ~position path =
  let rec values_of_export visited export =
    let key = (export.path, export.position) in
    if
      List.mem visited key ~equal:(fun (left_path, left_position) (right_path, right_position) ->
          List.equal String.equal left_path right_path && Int.equal left_position right_position)
    then []
    else
      export.values
      @ List.concat_map export.references ~f:(fun reference ->
          lookup_module_exports module_values ~module_path:reference.scope ~opened_paths:[]
            ~position:reference.position reference.path
          |> List.concat_map ~f:(values_of_export (key :: visited)))
  in
  let exports = lookup_module_exports module_values ~module_path ~opened_paths ~position path in
  match exports with
  | [] -> None
  | _ ->
      exports
      |> List.concat_map ~f:(values_of_export [])
      |> List.dedup_and_sort ~compare:String.compare
      |> Option.some

let opened_module_paths module_values ~module_path ~opened_paths ~position path =
  lookup_module_exports module_values ~module_path ~opened_paths ~position path
  |> List.map ~f:(fun export -> export.path)
  |> List.dedup_and_sort ~compare:(List.compare String.compare)

let module_expr_opened_paths module_values ~module_path ~opened_paths module_expr =
  match module_expr.pmod_desc with
  | Pmod_ident { txt; _ } ->
      Option.value_map (ident_path txt) ~default:[] ~f:(fun path ->
          opened_module_paths module_values ~module_path ~opened_paths
            ~position:module_expr.pmod_loc.loc_start.pos_cnum path)
  | _ -> []

let module_expr_binds module_values ~module_path ~opened_paths module_expr name =
  match module_expr.pmod_desc with
  | Pmod_structure items ->
      let values, references = structure_exports ~scope:module_path items in
      let included_values =
        List.concat_map references ~f:(fun reference ->
            lookup_module_values module_values ~module_path:reference.scope ~opened_paths
              ~position:reference.position reference.path
            |> Option.value ~default:[])
      in
      List.mem (values @ included_values) name ~equal:String.equal
  | Pmod_ident { txt; _ } -> (
      match ident_path txt with
      | Some path ->
          lookup_module_values module_values ~module_path ~opened_paths
            ~position:module_expr.pmod_loc.loc_start.pos_cnum path
          |> Option.value_map ~default:false ~f:(fun values ->
              List.mem values name ~equal:String.equal)
      | None -> false)
  | _ -> false

let module_expr_ignore_binding_state module_values ~module_path ~opened_paths module_expr =
  match module_expr.pmod_desc with
  | Pmod_ident { txt; _ } -> (
      match ident_path txt with
      | Some path -> (
          match
            lookup_module_values module_values ~module_path ~opened_paths
              ~position:module_expr.pmod_loc.loc_start.pos_cnum path
          with
          | Some values -> if List.mem values "ignore" ~equal:String.equal then Some true else None
          | None
            when List.equal String.equal path [ "Base" ]
                 || List.equal String.equal path [ "Stdlib" ] ->
              Some false
          | None -> Some true)
      | None -> Some true)
  | Pmod_structure _ ->
      if module_expr_binds module_values ~module_path ~opened_paths module_expr "ignore" then
        Some true
      else None
  | _ -> Some true

let module_expr_may_shadow_ignore module_values ~module_path ~opened_paths module_expr =
  Option.value
    (module_expr_ignore_binding_state module_values ~module_path ~opened_paths module_expr)
    ~default:false

let structure_item_ignore_binding_state module_values ~module_path ~opened_paths item =
  match item.pstr_desc with
  | Pstr_open open_declaration ->
      module_expr_ignore_binding_state module_values ~module_path ~opened_paths
        open_declaration.popen_expr
  | Pstr_include include_declaration ->
      module_expr_ignore_binding_state module_values ~module_path ~opened_paths
        include_declaration.pincl_mod
  | Pstr_primitive { pval_name = { txt = "ignore"; _ }; _ } -> Some true
  | _ -> None

let open_description_binds module_values ~module_path ~opened_paths
    (open_declaration : open_description) name =
  match ident_path open_declaration.popen_expr.txt with
  | Some path ->
      lookup_module_values module_values ~module_path ~opened_paths
        ~position:open_declaration.popen_loc.loc_start.pos_cnum path
      |> Option.value_map ~default:false ~f:(fun values -> List.mem values name ~equal:String.equal)
  | None -> false

let open_description_ignore_binding_state module_values ~module_path ~opened_paths
    (open_declaration : open_description) =
  match ident_path open_declaration.popen_expr.txt with
  | Some path -> (
      match
        lookup_module_values module_values ~module_path ~opened_paths
          ~position:open_declaration.popen_loc.loc_start.pos_cnum path
      with
      | Some values -> if List.mem values "ignore" ~equal:String.equal then Some true else None
      | None
        when List.equal String.equal path [ "Base" ] || List.equal String.equal path [ "Stdlib" ] ->
          Some false
      | None -> Some true)
  | None -> Some true

let open_description_may_shadow_ignore module_values ~module_path ~opened_paths open_declaration =
  Option.value
    (open_description_ignore_binding_state module_values ~module_path ~opened_paths open_declaration)
    ~default:false

let open_description_opened_paths module_values ~module_path ~opened_paths
    (open_declaration : open_description) =
  Option.value_map (ident_path open_declaration.popen_expr.txt) ~default:[] ~f:(fun path ->
      opened_module_paths module_values ~module_path ~opened_paths
        ~position:open_declaration.popen_loc.loc_start.pos_cnum path)

let structure_item_may_shadow_name module_values ~module_path ~opened_paths item name =
  match item.pstr_desc with
  | Pstr_open open_declaration ->
      module_expr_binds module_values ~module_path ~opened_paths open_declaration.popen_expr name
  | Pstr_include include_declaration ->
      module_expr_binds module_values ~module_path ~opened_paths include_declaration.pincl_mod name
  | Pstr_primitive value_description -> String.equal value_description.pval_name.txt name
  | _ -> false

let direct_discard ~unqualified_ignore_visible ~qualified_standard_visible name expression =
  match expression.pexp_desc with
  | Pexp_apply (callee, [ (Asttypes.Nolabel, argument) ]) ->
      is_ignore ~unqualified_visible:unqualified_ignore_visible ~qualified_standard_visible callee
      && Option.value_map (direct_name argument) ~default:false ~f:(String.equal name)
  | Pexp_apply ({ pexp_desc = Pexp_ident { txt = Lident "|>"; _ }; _ }, [ (_, left); (_, right) ])
    ->
      Option.value_map (direct_name left) ~default:false ~f:(String.equal name)
      && is_ignore ~unqualified_visible:unqualified_ignore_visible ~qualified_standard_visible right
  | Pexp_apply ({ pexp_desc = Pexp_ident { txt = Lident "@@"; _ }; _ }, [ (_, left); (_, right) ])
    ->
      is_ignore ~unqualified_visible:unqualified_ignore_visible ~qualified_standard_visible left
      && Option.value_map (direct_name right) ~default:false ~f:(String.equal name)
  | _ -> false

type function_tail = Expression of expression | Cases of case list | Class of class_expr
type local_result = Resolved of expression | Unresolved

type local_binding = {
  local_name : string;
  local_result : local_result;
  local_module_path : string list;
  local_ignore_is_shadowed : bool;
  local_opened_paths : string list list;
  rhs_locals : local_binding list option;
}

let rec local_named_expressions pattern expression =
  match (pattern.ppat_desc, expression.pexp_desc) with
  | Ppat_var { txt; _ }, _ -> [ (txt, Resolved expression) ]
  | Ppat_constraint (inner, _), _ -> local_named_expressions inner expression
  | Ppat_alias (inner, { txt; _ }), _ ->
      (txt, Resolved expression) :: local_named_expressions inner expression
  | Ppat_tuple patterns, Pexp_tuple expressions when List.length patterns = List.length expressions
    ->
      List.map2_exn patterns expressions ~f:local_named_expressions |> List.concat
  | _ -> List.map (pattern_names pattern) ~f:(fun name -> (name, Unresolved))

let function_params expression =
  let rec peel params expression =
    match expression.pexp_desc with
    | Pexp_constraint (inner, _) | Pexp_coerce (inner, _, _) | Pexp_poly (inner, _) ->
        peel params inner
    | Pexp_function (more, _, body) -> (
        let params = params @ more in
        match body with
        | Pfunction_body inner -> peel params inner
        | Pfunction_cases (cases, _, _) -> (params, Cases cases))
    | _ -> (params, Expression expression)
  in
  peel [] expression

let rec function_paths ?(locals = []) ~module_values ~module_path ~opened_paths ~ignore_is_shadowed
    expression =
  let params, tail = function_params expression in
  let shadow_locals pattern ~ignore_is_shadowed ~opened_paths locals =
    List.map (pattern_names pattern) ~f:(fun name ->
        {
          local_name = name;
          local_result = Unresolved;
          local_module_path = module_path;
          local_ignore_is_shadowed = ignore_is_shadowed;
          local_opened_paths = opened_paths;
          rhs_locals = None;
        })
    @ locals
  in
  let rec returned_paths ~locals ~opened_paths ~ignore_is_shadowed expression =
    match expression.pexp_desc with
    | Pexp_function _ ->
        function_paths ~locals ~module_values ~module_path ~opened_paths ~ignore_is_shadowed
          expression
    | Pexp_constraint (inner, _) | Pexp_coerce (inner, _, _) ->
        returned_paths ~locals ~opened_paths ~ignore_is_shadowed inner
    | Pexp_ident { txt = Lident name; _ } -> (
        match List.findi locals ~f:(fun _ binding -> String.equal binding.local_name name) with
        | Some (index, ({ local_result = Resolved bound_expression; _ } as binding)) ->
            let locals =
              Option.value binding.rhs_locals
                ~default:(List.filteri locals ~f:(fun i _ -> i <> index))
            in
            function_paths ~locals ~module_values ~module_path:binding.local_module_path
              ~opened_paths:binding.local_opened_paths
              ~ignore_is_shadowed:binding.local_ignore_is_shadowed bound_expression
        | Some (_, { local_result = Unresolved; _ }) -> []
        | None -> [])
    | Pexp_ifthenelse (_, then_, else_) ->
        returned_paths ~locals ~opened_paths ~ignore_is_shadowed then_
        @ Option.value_map else_ ~default:[]
            ~f:(returned_paths ~locals ~opened_paths ~ignore_is_shadowed)
    | Pexp_match (_, cases) ->
        List.concat_map cases ~f:(fun case ->
            let ignore_is_shadowed = ignore_is_shadowed || pattern_binds case.pc_lhs "ignore" in
            returned_paths
              ~locals:(shadow_locals case.pc_lhs ~ignore_is_shadowed ~opened_paths locals)
              ~opened_paths ~ignore_is_shadowed case.pc_rhs)
    | Pexp_try (body, cases) ->
        returned_paths ~locals ~opened_paths ~ignore_is_shadowed body
        @ List.concat_map cases ~f:(fun case ->
            let ignore_is_shadowed = ignore_is_shadowed || pattern_binds case.pc_lhs "ignore" in
            returned_paths
              ~locals:(shadow_locals case.pc_lhs ~ignore_is_shadowed ~opened_paths locals)
              ~opened_paths ~ignore_is_shadowed case.pc_rhs)
    | Pexp_let (recursive, bindings, body) ->
        let binding_ignore_is_shadowed =
          match recursive with
          | Nonrecursive -> ignore_is_shadowed
          | Recursive -> Option.value (ignore_binding_state bindings) ~default:ignore_is_shadowed
        in
        let new_locals =
          List.concat_map bindings ~f:(fun binding ->
              List.map (local_named_expressions binding.pvb_pat binding.pvb_expr)
                ~f:(fun (name, result) ->
                  {
                    local_name = name;
                    local_result = result;
                    local_module_path = module_path;
                    local_ignore_is_shadowed = binding_ignore_is_shadowed;
                    local_opened_paths = opened_paths;
                    rhs_locals =
                      (match recursive with Nonrecursive -> Some locals | Recursive -> None);
                  }))
        in
        returned_paths ~locals:(new_locals @ locals) ~opened_paths
          ~ignore_is_shadowed:
            (Option.value (ignore_binding_state bindings) ~default:ignore_is_shadowed)
          body
    | Pexp_open (open_declaration, body) ->
        let opened_ignore_state =
          module_expr_ignore_binding_state module_values ~module_path ~opened_paths
            open_declaration.popen_expr
        in
        let opened_paths =
          module_expr_opened_paths module_values ~module_path ~opened_paths
            open_declaration.popen_expr
          @ opened_paths
        in
        returned_paths ~locals ~opened_paths
          ~ignore_is_shadowed:(Option.value opened_ignore_state ~default:ignore_is_shadowed)
          body
    | Pexp_letop { let_; ands; body } ->
        let binds_ignore =
          pattern_binds let_.pbop_pat "ignore"
          || List.exists ands ~f:(fun binding -> pattern_binds binding.pbop_pat "ignore")
        in
        returned_paths ~locals ~opened_paths
          ~ignore_is_shadowed:(ignore_is_shadowed || binds_ignore)
          body
    | Pexp_letmodule (_, _, body) | Pexp_letexception (_, body) | Pexp_sequence (_, body) ->
        returned_paths ~locals ~opened_paths ~ignore_is_shadowed body
    | _ -> []
  in
  let nested =
    match tail with
    | Expression body -> returned_paths ~locals ~opened_paths ~ignore_is_shadowed body
    | Cases cases ->
        List.concat_map cases ~f:(fun case ->
            let ignore_is_shadowed = ignore_is_shadowed || pattern_binds case.pc_lhs "ignore" in
            returned_paths
              ~locals:(shadow_locals case.pc_lhs ~ignore_is_shadowed ~opened_paths locals)
              ~opened_paths ~ignore_is_shadowed case.pc_rhs)
    | Class _ -> []
  in
  (params, tail, module_path, ignore_is_shadowed, opened_paths) :: nested

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

let nonempty_capture_list expression =
  match expression.pexp_desc with
  | Pexp_construct ({ txt = Lident "::"; _ }, Some { pexp_desc = Pexp_tuple [ head; _tail ]; _ }) ->
      Option.is_some (string_constant head)
  | _ -> false

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
        | `Unary, None, Pexp_apply (callee, [ (Nolabel, captures) ])
          when nonempty_capture_list captures ->
            string_constant callee
        | `Unary, None, _ -> None
        | `Binary, None, Pexp_apply (callee, arguments) -> (
            match arguments with
            | [ (Nolabel, argument) ] -> (
                match string_constant callee with
                | Some _ as spec -> spec
                | None -> string_constant argument)
            | [ (Nolabel, first); (Nolabel, second) ] -> (
                match string_constant callee with
                | Some _ as spec when nonempty_capture_list first -> spec
                | _ when nonempty_capture_list second -> string_constant first
                | _ -> None)
            | _ -> None)
        | (`Binary | `Other), _, _ -> None
      in
      Option.filter spec ~f:(fun spec -> String.contains spec '>')
  | _ -> None

let rec meaningfully_used ?(ignore_is_shadowed = false) ~module_values ~module_path ~opened_paths
    ~dsl name tail =
  let meaningful = ref false in
  let shadowed = ref false in
  let ignore_shadowed = ref ignore_is_shadowed in
  let opened_modules = ref opened_paths in
  let local_module_exports = ref [] in
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
  let within_opened_state state f =
    let saved = !opened_modules in
    opened_modules := state;
    f ();
    opened_modules := saved
  in
  let within_local_module export f =
    let saved = !local_module_exports in
    local_module_exports := export :: saved;
    f ();
    local_module_exports := saved
  in
  let effective_module_values () = !local_module_exports @ module_values in
  let qualified_standard_visible qualifier position =
    List.is_empty
      (lookup_module_exports (effective_module_values ()) ~module_path ~opened_paths:!opened_modules
         ~position [ qualifier ])
  in
  let ignore_state bindings = ignore_binding_state ~qualified_standard_visible bindings in
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
              let next_ignore = ignore_state bindings in
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
              let newly_opened =
                match item.pstr_desc with
                | Pstr_open open_declaration ->
                    module_expr_opened_paths (effective_module_values ()) ~module_path
                      ~opened_paths:!opened_modules open_declaration.popen_expr
                | _ -> []
              in
              within_opened_state (newly_opened @ !opened_modules) (fun () ->
                  within_ignore_state
                    (Option.value
                       (structure_item_ignore_binding_state (effective_module_values ())
                          ~module_path ~opened_paths:!opened_modules item)
                       ~default:!ignore_shadowed)
                    (fun () ->
                      within_shadow
                        (structure_item_may_shadow_name (effective_module_values ()) ~module_path
                           ~opened_paths:!opened_modules item name)
                        (fun () -> walk rest)))
        in
        walk items

      method! case case =
        within_ignore_shadow (pattern_binds case.pc_lhs "ignore") (fun () ->
            within_shadow (pattern_binds case.pc_lhs name) (fun () ->
                Option.iter case.pc_guard ~f:self#expression;
                self#expression case.pc_rhs))

      method! class_structure class_structure =
        within_ignore_shadow (pattern_binds class_structure.pcstr_self "ignore") (fun () ->
            within_shadow (pattern_binds class_structure.pcstr_self name) (fun () ->
                let instance_names =
                  List.filter_map class_structure.pcstr_fields ~f:(fun field ->
                      match field.pcf_desc with Pcf_val ({ txt; _ }, _, _) -> Some txt | _ -> None)
                in
                List.iter class_structure.pcstr_fields ~f:(fun field ->
                    match field.pcf_desc with
                    | Pcf_val (_, _, Cfk_concrete (_, initial_value)) ->
                        self#expression initial_value
                    | Pcf_val (_, _, Cfk_virtual _) -> ()
                    | _ ->
                        within_ignore_shadow (List.mem instance_names "ignore" ~equal:String.equal)
                          (fun () ->
                            within_shadow (List.mem instance_names name ~equal:String.equal)
                              (fun () -> self#class_field field)))))

      method! class_expr class_expression =
        if not !shadowed then
          match class_expression.pcl_desc with
          | Pcl_fun (_, default, pattern, body) ->
              Option.iter default ~f:self#expression;
              within_ignore_shadow (pattern_binds pattern "ignore") (fun () ->
                  within_shadow (pattern_binds pattern name) (fun () -> self#class_expr body))
          | Pcl_open (open_declaration, body) ->
              self#open_description open_declaration;
              within_ignore_state
                (Option.value
                   (open_description_ignore_binding_state (effective_module_values ()) ~module_path
                      ~opened_paths:!opened_modules open_declaration)
                   ~default:!ignore_shadowed)
                (fun () ->
                  let opened_paths =
                    open_description_opened_paths (effective_module_values ()) ~module_path
                      ~opened_paths:!opened_modules open_declaration
                    @ !opened_modules
                  in
                  within_opened_state opened_paths (fun () ->
                      within_shadow
                        (open_description_binds (effective_module_values ()) ~module_path
                           ~opened_paths:!opened_modules open_declaration name)
                        (fun () -> self#class_expr body)))
          | Pcl_let (recursive, bindings, body) ->
              let binds =
                List.exists bindings ~f:(fun binding -> pattern_binds binding.pvb_pat name)
              in
              List.iter bindings ~f:(fun binding ->
                  let discarded =
                    Option.value_map (direct_name binding.pvb_expr) ~default:false
                      ~f:(String.equal name)
                    &&
                    match throwaway_pattern binding.pvb_pat with
                    | Not_throwaway -> false
                    | Wildcard -> true
                    | Named_throwaway bound_names ->
                        let used_in_body =
                          List.exists bound_names ~f:(fun bound ->
                              meaningfully_used ~ignore_is_shadowed:!ignore_shadowed
                                ~module_values:(effective_module_values ()) ~module_path
                                ~opened_paths:!opened_modules ~dsl:!dsl_mode bound (Class body))
                        in
                        let used_in_recursive_group =
                          match recursive with
                          | Nonrecursive -> false
                          | Recursive ->
                              List.exists bindings ~f:(fun other ->
                                  List.exists bound_names ~f:(fun bound ->
                                      meaningfully_used ~ignore_is_shadowed:!ignore_shadowed
                                        ~module_values:(effective_module_values ()) ~module_path
                                        ~opened_paths:!opened_modules ~dsl:!dsl_mode bound
                                        (Expression other.pvb_expr)))
                        in
                        not (used_in_body || used_in_recursive_group)
                  in
                  if not discarded then
                    within_ignore_state
                      (match (recursive, ignore_state bindings) with
                      | Recursive, Some state -> state
                      | _ -> !ignore_shadowed)
                      (fun () ->
                        within_shadow
                          (match recursive with Recursive -> binds | Nonrecursive -> false)
                          (fun () -> self#expression binding.pvb_expr)));
              within_ignore_state
                (Option.value (ignore_state bindings) ~default:!ignore_shadowed)
                (fun () -> within_shadow binds (fun () -> self#class_expr body))
          | _ -> super#class_expr class_expression

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
            when direct_discard ~unqualified_ignore_visible:(not !ignore_shadowed)
                   ~qualified_standard_visible name expression ->
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
                                  ~module_values:(effective_module_values ()) ~module_path
                                  ~opened_paths:!opened_modules ~dsl:!dsl_mode bound
                                  (Expression body))
                          in
                          let used_in_recursive_group =
                            match recursive with
                            | Nonrecursive -> false
                            | Recursive ->
                                List.exists bindings ~f:(fun other ->
                                    List.exists bound_names ~f:(fun bound ->
                                        meaningfully_used ~ignore_is_shadowed:!ignore_shadowed
                                          ~module_values:(effective_module_values ()) ~module_path
                                          ~opened_paths:!opened_modules ~dsl:!dsl_mode bound
                                          (Expression other.pvb_expr)))
                          in
                          not (used_in_body || used_in_recursive_group)
                  in
                  if not discarded then
                    within_ignore_state
                      (match (recursive, ignore_state bindings) with
                      | Recursive, Some state -> state
                      | _ -> !ignore_shadowed)
                      (fun () ->
                        within_shadow
                          (match recursive with Recursive -> binds | Nonrecursive -> false)
                          (fun () -> self#expression binding.pvb_expr)));
              within_ignore_state
                (Option.value (ignore_state bindings) ~default:!ignore_shadowed)
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
          | Pexp_letmodule ({ txt = Some module_name; _ }, module_expression, body) ->
              self#module_expr module_expression;
              let path = module_path @ [ module_name ] in
              let values, references =
                match module_expression.pmod_desc with
                | Pmod_structure _ | Pmod_constraint _ ->
                    module_expr_exports ~scope:path module_expression
                | Pmod_ident _ -> module_expr_exports ~scope:module_path module_expression
                | _ -> ([], [])
              in
              within_local_module
                {
                  path;
                  position = module_expression.pmod_loc.loc_start.pos_cnum;
                  values;
                  references;
                } (fun () -> self#expression body)
          | Pexp_open (open_declaration, body) ->
              (* A local open can replace both unqualified [ignore] and the target name. *)
              self#module_expr open_declaration.popen_expr;
              within_ignore_state
                (Option.value
                   (module_expr_ignore_binding_state (effective_module_values ()) ~module_path
                      ~opened_paths:!opened_modules open_declaration.popen_expr)
                   ~default:!ignore_shadowed)
                (fun () ->
                  let opened_paths =
                    module_expr_opened_paths (effective_module_values ()) ~module_path
                      ~opened_paths:!opened_modules open_declaration.popen_expr
                    @ !opened_modules
                  in
                  within_opened_state opened_paths (fun () ->
                      within_shadow
                        (module_expr_binds (effective_module_values ()) ~module_path
                           ~opened_paths:!opened_modules open_declaration.popen_expr name)
                        (fun () -> self#expression body)))
          | Pexp_object class_structure -> self#class_structure class_structure
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
  | Cases cases -> iterator#cases cases
  | Class class_expression -> iterator#class_expr class_expression);
  !meaningful

let implementation_of ~dsl ~module_values ~module_path ~opened_paths ~ignore_is_shadowed ~params
    ~position ~tail pattern =
  let names = pattern_names pattern in
  let name_used name =
    (* An earlier parameter is in scope in every later default until a later parameter shadows it;
       that shadowing parameter's own default is still evaluated before its pattern binds. *)
    let rec used_after ignore_is_shadowed = function
      | [] ->
          meaningfully_used ~ignore_is_shadowed ~module_values ~module_path ~opened_paths ~dsl name
            tail
      | param :: later -> (
          match param.pparam_desc with
          | Pparam_newtype _ -> used_after ignore_is_shadowed later
          | Pparam_val (_, default, pattern) ->
              let used_in_default =
                Option.value_map default ~default:false ~f:(fun expression ->
                    meaningfully_used ~ignore_is_shadowed ~module_values ~module_path ~opened_paths
                      ~dsl name (Expression expression))
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

let method_optional_args ~source ~definition ~namespace ~dsl ~module_values ~module_path
    ~opened_paths ~ignore_is_shadowed class_structure =
  let instance_names =
    List.filter_map class_structure.pcstr_fields ~f:(fun field ->
        match field.pcf_desc with Pcf_val ({ txt; _ }, _, _) -> Some txt | _ -> None)
  in
  let ignore_is_shadowed =
    ignore_is_shadowed
    || pattern_binds class_structure.pcstr_self "ignore"
    || List.mem instance_names "ignore" ~equal:String.equal
  in
  List.concat_map class_structure.pcstr_fields ~f:(fun field ->
      match field.pcf_desc with
      | Pcf_method ({ txt = method_name; _ }, _, Cfk_concrete (_, expression)) ->
          let definition = definition ^ "#" ^ method_name in
          List.concat_map
            (function_paths ~module_values ~module_path ~opened_paths ~ignore_is_shadowed expression)
            ~f:(fun (params, tail, path_module_path, path_ignore_is_shadowed, path_opened_paths) ->
              List.filter_mapi params ~f:(fun position param ->
                  match param.pparam_desc with
                  | Pparam_val (Optional label, _, pattern) ->
                      let implementation =
                        implementation_of ~dsl ~module_values ~module_path:path_module_path
                          ~opened_paths:path_opened_paths
                          ~ignore_is_shadowed:path_ignore_is_shadowed ~params ~position ~tail
                          pattern
                      in
                      Some { source; definition; label; implementation; namespace }
                  | _ -> None))
      | _ -> [])

let class_optional_args ~source ~definition ~dsl ~module_values ~module_path ~opened_paths
    ~ignore_is_shadowed class_expression =
  let found = ref [] in
  let rec walk ~ignore_is_shadowed ~opened_paths class_expression =
    match class_expression.pcl_desc with
    | Pcl_fun (label, _, pattern, body) ->
        (match label with
        | Optional label ->
            let implementation =
              if
                List.exists (pattern_names pattern) ~f:(fun name ->
                    meaningfully_used
                      ~ignore_is_shadowed:(ignore_is_shadowed || pattern_binds pattern "ignore")
                      ~module_values ~module_path ~opened_paths ~dsl name (Class body))
              then Implemented
              else Unimplemented
            in
            found := { source; definition; label; implementation; namespace = Class } :: !found
        | Nolabel | Labelled _ -> ());
        walk
          ~ignore_is_shadowed:(ignore_is_shadowed || pattern_binds pattern "ignore")
          ~opened_paths body
    | Pcl_let (_, bindings, body) ->
        walk ~opened_paths
          ~ignore_is_shadowed:
            (Option.value (ignore_binding_state bindings) ~default:ignore_is_shadowed)
          body
    | Pcl_constraint (inner, _) -> walk ~ignore_is_shadowed ~opened_paths inner
    | Pcl_open (open_declaration, body) ->
        let newly_opened =
          open_description_opened_paths module_values ~module_path ~opened_paths open_declaration
        in
        walk ~opened_paths:(newly_opened @ opened_paths)
          ~ignore_is_shadowed:
            (Option.value
               (open_description_ignore_binding_state module_values ~module_path ~opened_paths
                  open_declaration)
               ~default:ignore_is_shadowed)
          body
    | Pcl_structure class_structure ->
        found :=
          method_optional_args ~source ~definition ~namespace:Class ~dsl ~module_values ~module_path
            ~opened_paths ~ignore_is_shadowed class_structure
          @ !found
    | Pcl_constr _ | Pcl_apply _ | Pcl_extension _ -> ()
  in
  walk ~ignore_is_shadowed ~opened_paths class_expression;
  List.rev !found

let object_method_optional_args ~source ~definition ~dsl ~module_values ~module_path ~opened_paths
    ~ignore_is_shadowed expression =
  let rec walk ~opened_paths ~ignore_is_shadowed expression =
    match expression.pexp_desc with
    | Pexp_object class_structure ->
        method_optional_args ~source ~definition ~namespace:Value ~dsl ~module_values ~module_path
          ~opened_paths ~ignore_is_shadowed class_structure
    | Pexp_constraint (inner, _) | Pexp_coerce (inner, _, _) | Pexp_poly (inner, _) ->
        walk ~opened_paths ~ignore_is_shadowed inner
    | Pexp_ifthenelse (_, then_, else_) ->
        walk ~opened_paths ~ignore_is_shadowed then_
        @ Option.value_map else_ ~default:[] ~f:(walk ~opened_paths ~ignore_is_shadowed)
    | Pexp_match (_, cases) ->
        List.concat_map cases ~f:(fun case ->
            walk ~opened_paths
              ~ignore_is_shadowed:(ignore_is_shadowed || pattern_binds case.pc_lhs "ignore")
              case.pc_rhs)
    | Pexp_try (body, cases) ->
        walk ~opened_paths ~ignore_is_shadowed body
        @ List.concat_map cases ~f:(fun case ->
            walk ~opened_paths
              ~ignore_is_shadowed:(ignore_is_shadowed || pattern_binds case.pc_lhs "ignore")
              case.pc_rhs)
    | Pexp_let (_, bindings, body) ->
        walk ~opened_paths
          ~ignore_is_shadowed:
            (Option.value (ignore_binding_state bindings) ~default:ignore_is_shadowed)
          body
    | Pexp_open (open_declaration, body) ->
        let newly_opened =
          module_expr_opened_paths module_values ~module_path ~opened_paths
            open_declaration.popen_expr
        in
        walk ~opened_paths:(newly_opened @ opened_paths)
          ~ignore_is_shadowed:
            (Option.value
               (module_expr_ignore_binding_state module_values ~module_path ~opened_paths
                  open_declaration.popen_expr)
               ~default:ignore_is_shadowed)
          body
    | Pexp_sequence (_, body) -> walk ~opened_paths ~ignore_is_shadowed body
    | Pexp_function _ -> (
        let params, tail = function_params expression in
        let ignore_is_shadowed =
          ignore_is_shadowed
          || List.exists params ~f:(fun param ->
              match param.pparam_desc with
              | Pparam_val (_, _, pattern) -> pattern_binds pattern "ignore"
              | Pparam_newtype _ -> false)
        in
        match tail with
        | Expression body -> walk ~opened_paths ~ignore_is_shadowed body
        | Cases cases ->
            List.concat_map cases ~f:(fun case ->
                walk ~opened_paths
                  ~ignore_is_shadowed:(ignore_is_shadowed || pattern_binds case.pc_lhs "ignore")
                  case.pc_rhs)
        | Class _ -> [])
    | _ -> []
  in
  walk ~opened_paths ~ignore_is_shadowed expression

let record_field_optional_args ~source ~definition ~dsl ~module_values ~module_path ~opened_paths
    ~ignore_is_shadowed ~locals expression =
  let shadow_locals pattern ~ignore_is_shadowed ~opened_paths locals =
    List.map (pattern_names pattern) ~f:(fun name ->
        {
          local_name = name;
          local_result = Unresolved;
          local_module_path = module_path;
          local_ignore_is_shadowed = ignore_is_shadowed;
          local_opened_paths = opened_paths;
          rhs_locals = None;
        })
    @ locals
  in
  let optional_args ~definition ~locals ~opened_paths ~ignore_is_shadowed expression =
    List.concat_map
      (function_paths ~locals ~module_values ~module_path ~opened_paths ~ignore_is_shadowed
         expression)
      ~f:(fun (params, tail, path_module_path, path_ignore_is_shadowed, path_opened_paths) ->
        List.filter_mapi params ~f:(fun position param ->
            match param.pparam_desc with
            | Pparam_val (Optional label, _, pattern) ->
                let implementation =
                  implementation_of ~dsl ~module_values ~module_path:path_module_path
                    ~opened_paths:path_opened_paths ~ignore_is_shadowed:path_ignore_is_shadowed
                    ~params ~position ~tail pattern
                in
                Some { source; definition; label; implementation; namespace = Value }
            | _ -> None))
  in
  let rec walk ~definition ~locals ~opened_paths ~ignore_is_shadowed expression =
    match expression.pexp_desc with
    | Pexp_record (fields, _) ->
        List.concat_map fields ~f:(fun (field, expression) ->
            let field_name =
              match ident_path field.txt with
              | Some path -> String.concat ~sep:"." path
              | None -> "<field>"
            in
            let definition = definition ^ "." ^ field_name in
            optional_args ~definition ~locals ~opened_paths ~ignore_is_shadowed expression
            @ walk ~definition ~locals ~opened_paths ~ignore_is_shadowed expression)
    | Pexp_tuple expressions | Pexp_array expressions ->
        List.concat_mapi expressions ~f:(fun index expression ->
            let definition = definition ^ "." ^ Int.to_string index in
            optional_args ~definition ~locals ~opened_paths ~ignore_is_shadowed expression
            @ walk ~definition ~locals ~opened_paths ~ignore_is_shadowed expression)
    | Pexp_construct (_, Some payload) | Pexp_variant (_, Some payload) ->
        let definition = definition ^ ".payload" in
        optional_args ~definition ~locals ~opened_paths ~ignore_is_shadowed payload
        @ walk ~definition ~locals ~opened_paths ~ignore_is_shadowed payload
    | Pexp_constraint (inner, _) | Pexp_coerce (inner, _, _) ->
        walk ~definition ~locals ~opened_paths ~ignore_is_shadowed inner
    | Pexp_ifthenelse (_, then_, else_) ->
        walk ~definition ~locals ~opened_paths ~ignore_is_shadowed then_
        @ Option.value_map else_ ~default:[]
            ~f:(walk ~definition ~locals ~opened_paths ~ignore_is_shadowed)
    | Pexp_match (_, cases) ->
        List.concat_map cases ~f:(fun case ->
            let ignore_is_shadowed = ignore_is_shadowed || pattern_binds case.pc_lhs "ignore" in
            walk ~definition
              ~locals:(shadow_locals case.pc_lhs ~ignore_is_shadowed ~opened_paths locals)
              ~opened_paths ~ignore_is_shadowed case.pc_rhs)
    | Pexp_try (body, cases) ->
        walk ~definition ~locals ~opened_paths ~ignore_is_shadowed body
        @ List.concat_map cases ~f:(fun case ->
            let ignore_is_shadowed = ignore_is_shadowed || pattern_binds case.pc_lhs "ignore" in
            walk ~definition
              ~locals:(shadow_locals case.pc_lhs ~ignore_is_shadowed ~opened_paths locals)
              ~opened_paths ~ignore_is_shadowed case.pc_rhs)
    | Pexp_let (recursive, bindings, body) ->
        let binding_ignore_is_shadowed =
          match recursive with
          | Nonrecursive -> ignore_is_shadowed
          | Recursive -> Option.value (ignore_binding_state bindings) ~default:ignore_is_shadowed
        in
        let new_locals =
          List.concat_map bindings ~f:(fun binding ->
              List.map (local_named_expressions binding.pvb_pat binding.pvb_expr)
                ~f:(fun (name, result) ->
                  {
                    local_name = name;
                    local_result = result;
                    local_module_path = module_path;
                    local_ignore_is_shadowed = binding_ignore_is_shadowed;
                    local_opened_paths = opened_paths;
                    rhs_locals =
                      (match recursive with Nonrecursive -> Some locals | Recursive -> None);
                  }))
        in
        walk ~definition ~locals:(new_locals @ locals) ~opened_paths
          ~ignore_is_shadowed:
            (Option.value (ignore_binding_state bindings) ~default:ignore_is_shadowed)
          body
    | Pexp_open (open_declaration, body) ->
        let opened_ignore_state =
          module_expr_ignore_binding_state module_values ~module_path ~opened_paths
            open_declaration.popen_expr
        in
        let opened_paths =
          module_expr_opened_paths module_values ~module_path ~opened_paths
            open_declaration.popen_expr
          @ opened_paths
        in
        walk ~definition ~locals ~opened_paths
          ~ignore_is_shadowed:(Option.value opened_ignore_state ~default:ignore_is_shadowed)
          body
    | Pexp_sequence (_, body) -> walk ~definition ~locals ~opened_paths ~ignore_is_shadowed body
    | _ -> []
  in
  walk ~definition ~locals ~opened_paths ~ignore_is_shadowed expression

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

let packed_module_optional_args ~source ~definition ~dsl ~module_values ~module_path ~opened_paths
    ~ignore_is_shadowed expression =
  let rec module_structure module_expression =
    match module_expression.pmod_desc with
    | Pmod_structure items -> Some items
    | Pmod_constraint (inner, _) -> module_structure inner
    | _ -> None
  in
  let rec packed expression =
    match expression.pexp_desc with
    | Pexp_pack module_expression -> module_structure module_expression
    | Pexp_constraint (inner, _) | Pexp_coerce (inner, _, _) -> packed inner
    | _ -> None
  in
  match packed expression with
  | None -> []
  | Some items ->
      List.concat_map items ~f:(fun item ->
          match item.pstr_desc with
          | Pstr_value (_, bindings) ->
              List.concat_map bindings ~f:(fun binding ->
                  List.concat_map (named_expressions binding.pvb_pat binding.pvb_expr)
                    ~f:(fun (field_name, expression) ->
                      let definition = definition ^ "." ^ field_name in
                      List.concat_map
                        (function_paths ~module_values ~module_path ~opened_paths
                           ~ignore_is_shadowed expression)
                        ~f:(fun
                            ( params,
                              tail,
                              path_module_path,
                              path_ignore_is_shadowed,
                              path_opened_paths )
                          ->
                          List.filter_mapi params ~f:(fun position param ->
                              match param.pparam_desc with
                              | Pparam_val (Optional label, _, pattern) ->
                                  let implementation =
                                    implementation_of ~dsl ~module_values
                                      ~module_path:path_module_path ~opened_paths:path_opened_paths
                                      ~ignore_is_shadowed:path_ignore_is_shadowed ~params ~position
                                      ~tail pattern
                                  in
                                  Some
                                    { source; definition; label; implementation; namespace = Value }
                              | _ -> None))))
          | _ -> [])

let args_in_source ?interface ~source content =
  let structure = structure_of content in
  let module_values = module_values structure in
  let found = ref [] in
  let local_expression_depth = ref 0 in
  let module_path = ref [] in
  let dsl_context = ref Outside_dsl in
  let top_ignore_shadowed = ref false in
  let top_opened_paths = ref [] in
  let top_function_bindings = ref [] in
  let defer_top_bindings = ref false in
  let qualified_standard_visible qualifier position =
    List.is_empty
      (lookup_module_exports module_values ~module_path:(List.rev !module_path)
         ~opened_paths:!top_opened_paths ~position [ qualifier ])
  in
  let top_ignore_binding_state bindings =
    ignore_binding_state ~qualified_standard_visible bindings
  in
  let within_module name f =
    let saved = !module_path in
    let saved_ignore = !top_ignore_shadowed in
    let saved_opened = !top_opened_paths in
    let saved_functions = !top_function_bindings in
    module_path := name :: saved;
    f ();
    module_path := saved;
    top_ignore_shadowed := saved_ignore;
    top_opened_paths := saved_opened;
    top_function_bindings := saved_functions
  in
  let qualify name = String.concat ~sep:"." (List.rev (name :: !module_path)) in
  let same_optional left right =
    String.equal left.source right.source
    && String.equal left.definition right.definition
    && String.equal left.label right.label
    && Poly.equal left.namespace right.namespace
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
  let remove_owned_definitions ?namespace owners =
    found :=
      List.filter !found ~f:(fun arg ->
          not
            (Option.value_map namespace ~default:true ~f:(Poly.equal arg.namespace)
            && List.exists owners ~f:(fun owner -> definition_owned_by owner arg.definition)))
  in
  let install_top_bindings recursive bindings =
    let previous_locals = !top_function_bindings in
    let bound_names = List.concat_map bindings ~f:(fun binding -> pattern_names binding.pvb_pat) in
    top_function_bindings :=
      List.filter previous_locals ~f:(fun binding ->
          not (List.mem bound_names binding.local_name ~equal:String.equal));
    top_function_bindings :=
      List.concat_map bindings ~f:(fun binding ->
          List.map (local_named_expressions binding.pvb_pat binding.pvb_expr)
            ~f:(fun (name, result) ->
              {
                local_name = name;
                local_result = result;
                local_module_path = List.rev !module_path;
                local_ignore_is_shadowed = !top_ignore_shadowed;
                local_opened_paths = !top_opened_paths;
                rhs_locals =
                  (match recursive with
                  | Asttypes.Nonrecursive -> Some previous_locals
                  | Asttypes.Recursive -> None);
              }))
      @ !top_function_bindings
  in
  let copy_module_exports ~destination_prefix module_expression =
    match module_expression.pmod_desc with
    | Pmod_ident { txt; _ } ->
        Option.iter (ident_path txt) ~f:(fun path ->
            let targets =
              opened_module_paths module_values ~module_path:(List.rev !module_path)
                ~opened_paths:!top_opened_paths
                ~position:module_expression.pmod_loc.loc_start.pos_cnum path
            in
            List.iter targets ~f:(fun target ->
                let target_prefix = String.concat ~sep:"." target in
                List.filter !found ~f:(fun arg ->
                    String.is_prefix arg.definition ~prefix:(target_prefix ^ "."))
                |> List.iter ~f:(fun arg ->
                    let suffix = String.drop_prefix arg.definition (String.length target_prefix) in
                    let definition =
                      if String.is_empty destination_prefix then String.drop_prefix suffix 1
                      else destination_prefix ^ suffix
                    in
                    add_optional found { arg with definition })))
    | _ -> ()
  in
  let iterator =
    object (self)
      inherit Ast_traverse.iter as super

      method! structure items =
        if !local_expression_depth > 0 then super#structure items
        else
          let rec walk = function
            | [] -> ()
            | item :: rest -> (
                (match item.pstr_desc with
                | Pstr_value (recursive, bindings) ->
                    let saved_defer = !defer_top_bindings in
                    defer_top_bindings := true;
                    Exn.protect
                      ~f:(fun () -> self#structure_item item)
                      ~finally:(fun () -> defer_top_bindings := saved_defer);
                    install_top_bindings recursive bindings
                | _ -> self#structure_item item);
                match item.pstr_desc with
                | Pstr_value (_, bindings) ->
                    Option.iter (top_ignore_binding_state bindings) ~f:(fun state ->
                        top_ignore_shadowed := state);
                    walk rest
                | Pstr_open open_declaration ->
                    let newly_opened =
                      module_expr_opened_paths module_values ~module_path:(List.rev !module_path)
                        ~opened_paths:!top_opened_paths open_declaration.popen_expr
                    in
                    top_opened_paths := newly_opened @ !top_opened_paths;
                    Option.iter
                      (structure_item_ignore_binding_state module_values
                         ~module_path:(List.rev !module_path) ~opened_paths:!top_opened_paths item)
                      ~f:(fun state -> top_ignore_shadowed := state);
                    walk rest
                | _ ->
                    Option.iter
                      (structure_item_ignore_binding_state module_values
                         ~module_path:(List.rev !module_path) ~opened_paths:!top_opened_paths item)
                      ~f:(fun state -> top_ignore_shadowed := state);
                    walk rest)
          in
          walk items

      method! structure_item item =
        match item.pstr_desc with
        | Pstr_module { pmb_name = { txt = Some name; _ }; pmb_expr; _ } ->
            let destination_prefix = qualify name in
            remove_owned_definitions [ destination_prefix ];
            copy_module_exports ~destination_prefix pmb_expr;
            within_module name (fun () -> super#structure_item item)
        | Pstr_recmodule bindings ->
            List.filter_map bindings ~f:(fun binding -> Option.map binding.pmb_name.txt ~f:qualify)
            |> remove_owned_definitions;
            List.iter bindings ~f:(fun binding ->
                match binding.pmb_name.txt with
                | Some name ->
                    copy_module_exports ~destination_prefix:(qualify name) binding.pmb_expr;
                    within_module name (fun () -> self#module_expr binding.pmb_expr)
                | None -> self#module_expr binding.pmb_expr)
        | Pstr_include include_declaration ->
            copy_module_exports
              ~destination_prefix:(String.concat ~sep:"." (List.rev !module_path))
              include_declaration.pincl_mod;
            super#structure_item item
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

      method! class_expr class_expression =
        Int.incr local_expression_depth;
        Exn.protect
          ~f:(fun () -> super#class_expr class_expression)
          ~finally:(fun () -> Int.decr local_expression_depth)

      method! class_declaration declaration =
        if !local_expression_depth = 0 then (
          let definition = qualify declaration.pci_name.txt in
          remove_owned_definitions ~namespace:Class [ definition ];
          found :=
            class_optional_args ~source ~definition ~dsl:!dsl_context ~module_values
              ~module_path:(List.rev !module_path) ~opened_paths:!top_opened_paths
              ~ignore_is_shadowed:!top_ignore_shadowed declaration.pci_expr
            @ !found);
        super#class_declaration declaration

      method! value_binding binding =
        if !local_expression_depth = 0 then (
          let named = named_expressions binding.pvb_pat binding.pvb_expr in
          let definitions =
            List.map named ~f:(fun (definition, _) -> qualify definition)
            |> List.dedup_and_sort ~compare:String.compare
          in
          (* A later source binding replaces the earlier exported value. Branches of this one
             binding still merge below because they describe result paths of the same value. *)
          remove_owned_definitions ~namespace:Value definitions;
          let binding_found = ref [] in
          List.iter named ~f:(fun (definition, expression) ->
              List.iter
                (function_paths ~module_values ~module_path:(List.rev !module_path)
                   ~locals:!top_function_bindings ~opened_paths:!top_opened_paths
                   ~ignore_is_shadowed:!top_ignore_shadowed expression)
                ~f:(fun
                    (params, tail, path_module_path, path_ignore_is_shadowed, path_opened_paths) ->
                  List.iteri params ~f:(fun position param ->
                      match param.pparam_desc with
                      | Pparam_val (Asttypes.Optional label, _, pattern) ->
                          let implementation =
                            implementation_of ~dsl:!dsl_context ~module_values
                              ~module_path:path_module_path ~opened_paths:path_opened_paths
                              ~ignore_is_shadowed:path_ignore_is_shadowed ~params ~position ~tail
                              pattern
                          in
                          add_optional binding_found
                            {
                              source;
                              definition = qualify definition;
                              label;
                              implementation;
                              namespace = Value;
                            }
                      | _ -> ()));
              List.iter
                (object_method_optional_args ~source ~definition:(qualify definition)
                   ~dsl:!dsl_context ~module_values ~module_path:(List.rev !module_path)
                   ~opened_paths:!top_opened_paths ~ignore_is_shadowed:!top_ignore_shadowed
                   expression)
                ~f:(add_optional binding_found);
              List.iter
                (record_field_optional_args ~source ~definition:(qualify definition)
                   ~dsl:!dsl_context ~module_values ~module_path:(List.rev !module_path)
                   ~opened_paths:!top_opened_paths ~ignore_is_shadowed:!top_ignore_shadowed
                   ~locals:!top_function_bindings expression)
                ~f:(add_optional binding_found);
              List.iter
                (packed_module_optional_args ~source ~definition:(qualify definition)
                   ~dsl:!dsl_context ~module_values ~module_path:(List.rev !module_path)
                   ~opened_paths:!top_opened_paths ~ignore_is_shadowed:!top_ignore_shadowed
                   expression)
                ~f:(add_optional binding_found));
          found := !binding_found @ !found);
        if !local_expression_depth = 0 && not !defer_top_bindings then
          install_top_bindings Asttypes.Nonrecursive [ binding ];
        super#value_binding binding
    end
  in
  iterator#structure structure;
  let args = List.rev !found in
  Option.value_map interface ~default:args ~f:(fun interface ->
      let surface = public_surface_of_interface interface in
      List.filter args ~f:(exposed_by surface))
