(** Scanning OCaml sources for the configuration keys they read.

    Shared by the two consistency tests that hold the configuration honest:
    [test_config_consistency] (every key documented and registered) and [digest_completeness] (every
    key classified against the cache identity, gh-ocannl-572). Both need the same fact — which keys
    a source file reads — so they read it the same way.

    {1 Reading these sources as OCaml}

    This module parses. It does not match text, and it does not match token shapes.

    That is the conclusion of five review rounds on PR #340, each finding the same kind of defect
    one level down: prose read as a call site, a quoted string inside a comment read as a nested
    comment, a character literal opening a string, an escaped literal losing its value, and finally
    [let () = …] not counting as a definition while [?(arg_name : string = "k")] did not count as a
    literal. Every one was silent by construction — keys that vanish from a scan look exactly like
    keys that were never read — and every fix left the next unhandled spelling waiting.

    An approximation of OCaml has no natural stopping point; the grammar does. On the parse tree a
    labelled argument is one node however it was spelled, a string literal carries its decoded value
    whatever escapes produced it, and a structure item covers exactly the source it covers. *)

open Base
open Parsetree

(** The label whose argument names a configuration key. *)
let label = "arg_name"

let is_our_label = function
  | Asttypes.Labelled name | Asttypes.Optional name -> String.equal name label
  | Asttypes.Nolabel -> false

(** The value of a string literal as the parser resolved it: [{|k|}], ["k"] and a continuation
    spelled over two lines all arrive here decoded. *)
let string_literal expr =
  match expr.pexp_desc with
  | Pexp_constant { pconst_desc = Pconst_string (value, _, _); _ } -> Some value
  | _ -> None

(* [Ldot] carries a located prefix in this compiler; [.txt] drops the location. *)
let rec flatten_longident = function
  | Longident.Lident name -> [ name ]
  | Longident.Ldot (prefix, name) -> flatten_longident prefix.Location.txt @ [ name.Location.txt ]
  | Longident.Lapply (f, x) -> flatten_longident f.Location.txt @ flatten_longident x.Location.txt

let longident_of expr =
  match expr.pexp_desc with Pexp_ident { txt; _ } -> Some (flatten_longident txt) | _ -> None

(** Raises if [content] does not parse: a scan that cannot read its input must say so rather than
    report an empty census. *)
let structure_of content = Parse.implementation (Lexing.from_string content)

(** Every place the [arg_name] label names a configuration key, and what it names it with. [key] is
    [Some k] when the argument is a string literal — the convention both consistency tests rely on
    to find a read — and [None] when it is anything else: a variable, a punned parameter, an
    expression. [offset] is where the argument sits, for a caller that wants to report the line.

    Both positions count, because both name keys: an argument at an application site
    ([~arg_name:"key"], [?arg_name:"key"]) and a parameter's default ([?(arg_name = "key")], with or
    without a type annotation). A label in a {e type} is not a use of anything and does not
    appear. *)
type label_use = { key : string option; offset : int }

let label_uses content =
  let uses = ref [] in
  let record key (loc : Location.t) = uses := { key; offset = loc.loc_start.pos_cnum } :: !uses in
  let iterator =
    {
      Ast_iterator.default_iterator with
      expr =
        (fun self expr ->
          (match expr.pexp_desc with
          | Pexp_apply (_, args) ->
              List.iter args ~f:(fun (lbl, arg) ->
                  if is_our_label lbl then record (string_literal arg) arg.pexp_loc)
          | Pexp_function (params, _, _) ->
              List.iter params ~f:(fun param ->
                  match param.pparam_desc with
                  | Pparam_val (lbl, default, _) when is_our_label lbl ->
                      record (Option.bind default ~f:string_literal) param.pparam_loc
                  | _ -> ())
          | _ -> ());
          Ast_iterator.default_iterator.expr self expr);
    }
  in
  iterator.structure iterator (structure_of content);
  List.rev !uses

(** Every value binding, as (where it starts, where it ends, its name if it has a simple one).

    Two things this has to get right for exemptions to mean anything (Codex P2, round 6):

    - each binding carries its OWN range, not its structure item's, or the siblings of a
      [let … and …] group become indistinguishable and a use in one is read as a use in another;
    - a binding nested in a module carries its qualified name, so that a [Sneaky.get_global_arg]
      cannot collect the exemption written for the top-level [get_global_arg]. Bare names are
      therefore exactly the top-level ones, which is what an exemption list means to name.

    Pattern bindings ([let () = …]) are included with no name: a use inside one belongs to it, and
    it has nothing an exemption could match. *)
let definitions content =
  let items = ref [] in
  let path = ref [] in
  let qualify name = String.concat ~sep:"." (List.rev (name :: !path)) in
  let within name f =
    let saved = !path in
    path := name :: !path;
    f ();
    path := saved
  in
  let iterator =
    {
      Ast_iterator.default_iterator with
      structure_item =
        (fun self item ->
          match item.pstr_desc with
          | Pstr_value (_, bindings) ->
              List.iter bindings ~f:(fun binding ->
                  let name =
                    match binding.pvb_pat.ppat_desc with
                    | Ppat_var { txt; _ } -> Some (qualify txt)
                    | Ppat_constraint ({ ppat_desc = Ppat_var { txt; _ }; _ }, _) ->
                        Some (qualify txt)
                    | _ -> None
                  in
                  items :=
                    (binding.pvb_loc.loc_start.pos_cnum, binding.pvb_loc.loc_end.pos_cnum, name)
                    :: !items);
              Ast_iterator.default_iterator.structure_item self item
          | Pstr_module { pmb_name = { txt = name; _ }; _ } ->
              within
                (Option.value name ~default:"_")
                (fun () -> Ast_iterator.default_iterator.structure_item self item)
          | Pstr_recmodule _ ->
              (* One prefix for the group: the point is only that nothing inside is bare. *)
              within "_" (fun () -> Ast_iterator.default_iterator.structure_item self item)
          (* A local [let module M = … in …] needs no case of its own: this compiler represents it
             as a structure item inside the expression (Pexp_struct_item), so it arrives here and
             takes its prefix like any other module. *)
          | _ -> Ast_iterator.default_iterator.structure_item self item);
    }
  in
  iterator.structure iterator (structure_of content);
  List.rev !items

(** The definition an offset sits in: the smallest one containing it, so an inner binding wins over
    an outer one. *)
let definition_at definitions offset =
  List.filter definitions ~f:(fun (start, stop, _) -> start <= offset && offset < stop)
  |> List.min_elt ~compare:(fun (a_start, a_stop, _) (b_start, b_stop, _) ->
         Int.compare (a_stop - a_start) (b_stop - b_start))
  |> Option.bind ~f:(fun (_, _, name) -> name)

(** The keys [content] reads through the label.

    A key that reaches the lookup any other way — through a helper taking the name as a
    parameter — is invisible to this scan, and hence to both tests built on it. That is why
    [test_config_consistency] separately fails any non-literal use of the label outside the handful
    of named functions that implement the lookup. *)
let keys_in_source content =
  List.filter_map (label_uses content) ~f:(fun use -> use.key)
  |> List.filter ~f:(fun key -> not (String.is_empty key))

(** [keys_in_source] over each file, as a set. Call sites only — this is what
    [test_config_consistency] means by "every key a source file asks for is documented and
    registered". *)
let keys_in_files files =
  List.concat_map files ~f:(fun fname -> keys_in_source (Stdio.In_channel.read_all fname))
  |> Set.of_list (module String)

(** The other spelling of a configuration read: a field of the startup-resolved [Utils.settings]
    record, whose field names {e are} the config keys, and the two predicates over it that fold in
    the [log_level > 1] threshold ([debug_log_from_routines], [with_runtime_debug]). A census built
    from [arg_name] literals alone would miss every one of these — [large_models] is read as
    [Utils.settings.large_models] in the codegen, so a future misclassification of it could pass
    unchallenged (Codex P2 on PR #337).

    A field {e read} and a predicate {e call}: naming either in prose is not a use of it. *)
let settings_keys_in_source content =
  let keys = ref [] in
  let ends_with path suffix =
    let extra = List.length path - List.length suffix in
    extra >= 0 && List.equal String.equal (List.drop path extra) suffix
  in
  let is_unit expr =
    match expr.pexp_desc with
    | Pexp_construct ({ txt = Longident.Lident "()"; _ }, None) -> true
    | _ -> false
  in
  let iterator =
    {
      Ast_iterator.default_iterator with
      expr =
        (fun self expr ->
          (match expr.pexp_desc with
          (* Qualified: [Low_level.virtualize_settings] and friends are records of the same shape
             whose field names are NOT config keys ([max_visits] against [virtualize_max_visits]),
             so an unqualified match would attribute reads to keys that do not exist. *)
          | Pexp_field (record, { txt = field; _ }) -> (
              match (longident_of record, flatten_longident field) with
              | Some path, [ field ] when ends_with path [ "Utils"; "settings" ] ->
                  keys := field :: !keys
              | _ -> ())
          | Pexp_apply (f, [ (Asttypes.Nolabel, arg) ]) when is_unit arg -> (
              match longident_of f with
              | Some path when ends_with path [ "debug_log_from_routines" ] ->
                  keys := "log_level" :: "debug_log_from_routines" :: !keys
              | Some path when ends_with path [ "with_runtime_debug" ] ->
                  keys := "log_level" :: "output_debug_files_in_build_directory" :: !keys
              | _ -> ())
          | _ -> ());
          Ast_iterator.default_iterator.expr self expr);
    }
  in
  iterator.structure iterator (structure_of content);
  List.rev !keys

(** Every configuration read of a file — [arg_name] call sites and {!settings_keys_in_source} —
    keyed by file basename, for tests that care {e where} a key is read. Field names that are not
    config keys come along; callers intersect with the registry. *)
let keys_by_file files =
  List.map files ~f:(fun fname ->
      let content = Stdio.In_channel.read_all fname in
      ( Stdlib.Filename.basename fname,
        Set.of_list (module String) (keys_in_source content @ settings_keys_in_source content) ))
