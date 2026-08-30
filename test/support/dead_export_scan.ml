(** The source reader behind the dead implicit-export ratchet.

    This is deliberately a first cut. An implementation without an [.mli] exports every
    source-declared top-level [let] and [external], including helpers intended only for the
    implementation. We enumerate those declarations in [arrayjit/lib/] and [tensor/], then count
    references from every other OCaml source.

    A reference is conservative: a direct qualified path ([M.v]), a path through a module alias, or
    an unqualified identifier inside the lexical range of [open M]. Alias scopes are deliberately
    over-approximated to the whole source, and an [include M] counts as a reference to every value
    because it re-exports the whole interface. Both choices can hide a dead export through a false
    positive, but cannot falsely reject an ordinary use. Values introduced only by a PPX expansion
    or an [include] inside the defining module are outside this source-level first cut. *)

open Base
open Ppxlib.Parsetree
module Ast_traverse = Ppxlib.Ast_traverse
module Read = Config_key_scan

type export = { module_name : string; value : string; source : string; line : int }

type reference = {
  module_name : string;
  value : string;
  source : string;
  line : int;
  spelling : string;
}

let export_key ({ module_name; value; _ } : export) = module_name ^ "." ^ value

let valid_module_stem stem =
  (not (String.is_empty stem))
  && Char.is_alpha stem.[0]
  && String.for_all stem ~f:(fun c -> Char.is_alphanum c || Char.equal c '_' || Char.equal c '\'')

(** The OCaml module name of a direct [*.ml] source. Dune select alternatives such as
    [cuda_backend_impl.cudajit.ml] are not modules under that basename and return [None]. *)
let module_name_of_source source =
  match String.chop_suffix (Stdlib.Filename.basename source) ~suffix:".ml" with
  | Some stem when valid_module_stem stem -> Some (String.capitalize stem)
  | Some _ | None -> None

let pattern_names pattern =
  let names = ref [] in
  let iterator =
    object
      inherit Ast_traverse.iter as super

      method! pattern pattern =
        (match pattern.ppat_desc with Ppat_var { txt; _ } -> names := txt :: !names | _ -> ());
        super#pattern pattern
    end
  in
  iterator#pattern pattern;
  List.dedup_and_sort !names ~compare:String.compare

let exports_of_source ~source contents =
  match module_name_of_source source with
  | None -> []
  | Some module_name ->
      let rec items acc structure = List.fold structure ~init:acc ~f:item
      and item acc structure_item =
        match structure_item.pstr_desc with
        | Pstr_value (_, bindings) ->
            List.fold bindings ~init:acc ~f:(fun acc binding ->
                List.fold (pattern_names binding.pvb_pat) ~init:acc ~f:(fun acc value ->
                    { module_name; value; source; line = binding.pvb_loc.loc_start.pos_lnum } :: acc))
        | Pstr_primitive description ->
            {
              module_name;
              value = description.pval_name.txt;
              source;
              line = description.pval_loc.loc_start.pos_lnum;
            }
            :: acc
        (* [let%foo] is an extension carrying a structure payload before the PPX rewrites it. It is
           still a source-declared top-level value, so unwrap structure payloads at this level but
           never descend into a nested module. *)
        | Pstr_extension ((_, PStr nested), _) -> items acc nested
        | _ -> acc
      in
      items [] (Read.structure_of contents)
      |> List.dedup_and_sort ~compare:(fun (a : export) (b : export) ->
          match String.compare a.value b.value with
          | 0 -> String.compare a.source b.source
          | ordering -> ordering)

let path_last path = List.last path

let path_qualifier path =
  match List.rev path with _value :: qualifier :: _ -> Some qualifier | _ -> None

let module_expr_name module_expr =
  let rec unwrap module_expr =
    match module_expr.pmod_desc with Pmod_constraint (inner, _) -> unwrap inner | _ -> module_expr
  in
  match (unwrap module_expr).pmod_desc with
  | Pmod_ident { txt; _ } -> ( try Read.flatten_longident txt |> List.last with _ -> None)
  | _ -> None

(** References to [exports] from [sources]. Sources are [(repository-relative path, contents)]. *)
let references ~(exports : export list) ~sources =
  let modules =
    List.map exports ~f:(fun export -> export.module_name)
    |> List.dedup_and_sort ~compare:String.compare
  in
  let export_names =
    List.fold exports
      ~init:(Map.empty (module String))
      ~f:(fun map export ->
        Map.update map export.module_name ~f:(function
          | None -> Set.singleton (module String) export.value
          | Some values -> Set.add values export.value))
  in
  List.concat_map sources ~f:(fun (source, contents) ->
      let structure = Read.structure_of contents in
      List.concat_map modules ~f:(fun module_name ->
          let aliases, _opened, open_ranges =
            Read.module_bindings_of structure ~wanted:module_name
          in
          let receivers = Set.of_list (module String) (module_name :: aliases) in
          let names = Map.find_exn export_names module_name in
          let found = ref [] in
          let add ~value ~line ~spelling =
            if Set.mem names value then
              found := { module_name; value; source; line; spelling } :: !found
          in
          let iterator =
            object
              inherit Ast_traverse.iter as super

              method! structure_item item =
                (* [include M] consumes and re-exports M's entire interface. Count every value; an
                   external caller may subsequently reach it only through the including module. *)
                (match item.pstr_desc with
                | Pstr_include { pincl_mod; _ } -> (
                    match module_expr_name pincl_mod with
                    | Some receiver when Set.mem receivers receiver ->
                        Set.iter names ~f:(fun value ->
                            add ~value ~line:item.pstr_loc.loc_start.pos_lnum
                              ~spelling:("include " ^ receiver))
                    | Some _ | None -> ())
                | _ -> ());
                super#structure_item item

              method! expression expression =
                (match Read.longident_of expression with
                | Some path -> (
                    match (path_last path, path_qualifier path) with
                    | Some value, Some receiver when Set.mem receivers receiver ->
                        add ~value ~line:expression.pexp_loc.loc_start.pos_lnum
                          ~spelling:(String.concat ~sep:"." path)
                    | Some value, None
                      when Read.within open_ranges expression.pexp_loc.loc_start.pos_cnum ->
                        add ~value ~line:expression.pexp_loc.loc_start.pos_lnum ~spelling:value
                    | _ -> ())
                | None -> ());
                super#expression expression
            end
          in
          iterator#structure structure;
          List.rev !found))
  |> List.filter ~f:(fun reference ->
      not
        (List.exists exports ~f:(fun export ->
             String.equal export.module_name reference.module_name
             && String.equal export.source reference.source)))

let counts ~(exports : export list) references =
  let table = Hashtbl.create (module String) in
  List.iter exports ~f:(fun export -> Hashtbl.set table ~key:(export_key export) ~data:0);
  List.iter references ~f:(fun reference ->
      Hashtbl.update table
        (reference.module_name ^ "." ^ reference.value)
        ~f:(fun count -> 1 + Option.value count ~default:0));
  table
