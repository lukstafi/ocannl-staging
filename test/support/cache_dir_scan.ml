(** Scanning OCaml sources for the schedule-cache directories they name.

    An autotune run writes its saved schedules into a directory named relative to the working
    directory, so a test that names one leaves that directory wherever the executable ran. Under
    [dune runtest] that is inside [_build/], which is why a missing root [.gitignore] entry used to
    stay missing: the directory materialises in the repository root only when someone runs the test
    executable there by hand, and then it is untracked on that one machine.

    The ignore list answers that with a single glob over a name prefix rather than an entry per
    directory, so it cannot fall behind. What has to hold instead is that every directory a source
    names carries the prefix, which is what this module reads sources to establish.

    {1 Reading these sources as OCaml}

    Parsed, not grepped, for the reasons {!Config_key_scan} sets out at length — and this scan needs
    the parse tree more than that one does, because the name is usually not written at the call
    site. Of the two dozen [~cache_dir] arguments that name a directory, most arrive through a
    binding ([let cache_dir = "…" in … Autotune.tune ~cache_dir]) rather than as a literal, so a
    scan matching [~cache_dir:"…"] would see a minority of them and vouch for nothing about the
    rest.

    {1 What is resolved, and how far}

    Resolution is per FILE and flat: a binding's literal is available to every use in the same
    source, regardless of scope. A use that resolves to no literal is reported as such rather than
    dropped, so a spelling this module cannot follow fails the check that uses it instead of quietly
    shrinking the census. The one such spelling in the tree is a pass-through: a function taking
    [~cache_dir] and forwarding it, whose directory is named at ITS call sites and checked there. *)

open Base
open Parsetree

(** The label whose argument names a schedule cache directory. *)
let label = "cache_dir"

(** The prefix every such directory carries, so that one [.gitignore] glob covers all of them. It is
    the built-in default's own name, which is why the default needs no special case: [autotune_cache]
    starts with [autotune_cache]. *)
let required_prefix = "autotune_cache"

let string_literal = Config_key_scan.string_literal
let structure_of = Config_key_scan.structure_of
let longident_of = Config_key_scan.longident_of
let pattern_name = Config_key_scan.pattern_name

let is_our_label = function
  | Asttypes.Labelled name | Asttypes.Optional name -> String.equal name label
  | Asttypes.Nolabel -> false

let mentions_label name = String.is_substring name ~substring:label

(** How a [~cache_dir] argument names its directory. [Names] carries a directory the prefix rule
    applies to; [Disabled] is the empty string, which turns the disk cache off and creates nothing;
    [Forwarded] is an identifier some function in this file takes as a parameter, whose value
    arrives from a call site scanned in its own right; [Unresolved] is everything else, which the
    check reports rather than assumes. *)
type resolution = Names of string | Disabled | Forwarded of string | Unresolved of string

type use = { resolution : resolution; line : int }

let describe = function
  | Names name -> "names " ^ name
  | Disabled -> "disables the cache"
  | Forwarded name -> "forwards the parameter " ^ name
  | Unresolved how -> "names " ^ how

(* Every binding whose name mentions [cache_dir] and whose right-hand side is a string literal, and
   every name a function takes as a parameter that mentions it. The first table resolves a use; the
   second tells a use that resolves to nothing whether it is a pass-through. *)
let scope ast =
  let literals = Hashtbl.create (module String) in
  let parameters = Hash_set.create (module String) in
  let iterator =
    {
      Ast_iterator.default_iterator with
      value_binding =
        (fun self binding ->
          (match (pattern_name binding.pvb_pat, string_literal binding.pvb_expr) with
          | Some name, Some value when mentions_label name ->
              Hashtbl.set literals ~key:name ~data:value
          | _ -> ());
          Ast_iterator.default_iterator.value_binding self binding);
      expr =
        (fun self expr ->
          (match expr.pexp_desc with
          | Pexp_function (params, _, _) ->
              List.iter params ~f:(fun param ->
                  match param.pparam_desc with
                  | Pparam_val (lbl, _, pat) ->
                      let named =
                        match (lbl, pattern_name pat) with
                        | (Asttypes.Labelled name | Asttypes.Optional name), _
                          when mentions_label name ->
                            Some name
                        | _, Some name when mentions_label name -> Some name
                        | _ -> None
                      in
                      Option.iter named ~f:(Hash_set.add parameters)
                  | _ -> ())
          | _ -> ());
          Ast_iterator.default_iterator.expr self expr);
    }
  in
  iterator.structure iterator ast;
  (literals, parameters)

(** Every [~cache_dir] / [?cache_dir] argument in [content], and what each resolves to. *)
let uses content =
  let ast = structure_of content in
  let literals, parameters = scope ast in
  let resolve argument =
    match string_literal argument with
    | Some "" -> Disabled
    | Some value -> Names value
    | None -> (
        match Option.bind (longident_of argument) ~f:List.last with
        | None -> Unresolved "an expression"
        | Some name -> (
            match Hashtbl.find literals name with
            | Some value -> Names value
            | None ->
                if Hash_set.mem parameters name then Forwarded name
                else Unresolved ("`" ^ name ^ "`")))
  in
  let found = ref [] in
  let iterator =
    {
      Ast_iterator.default_iterator with
      expr =
        (fun self expr ->
          (match expr.pexp_desc with
          | Pexp_apply (_, args) ->
              List.iter args ~f:(fun (lbl, argument) ->
                  if is_our_label lbl then
                    found :=
                      { resolution = resolve argument; line = argument.pexp_loc.loc_start.pos_lnum }
                      :: !found)
          | _ -> ());
          Ast_iterator.default_iterator.expr self expr);
    }
  in
  iterator.structure iterator ast;
  List.rev !found

(** The glob that has to be in the root [.gitignore] for the prefix rule to ignore anything.
    Root-anchored and directory-only: a cache directory is only ever created in the working
    directory, and an unanchored pattern would hide a stray copy anywhere in the tree — which is how
    three of them once reached [test/config/] and were committed. *)
let required_glob = "/" ^ required_prefix ^ "*/"

(** Whether an ignore file carries [required_glob] as an ignore rather than a negation. Read line by
    line rather than as a glob engine: what is asked is whether this exact rule is present. *)
let declares_required_glob content =
  String.split_lines content
  |> List.exists ~f:(fun line -> String.equal (String.strip line) required_glob)
