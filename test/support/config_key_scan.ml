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

(* The library's own function rather than a match on the constructors: [Ldot] carries located
   components in 5.5 and bare ones in 5.3, and this scanner has to build across the whole floor the
   opam files declare -- the scheduled 5.3 job caught exactly that.

   [Longident.flatten] fatal-errors on a functor application, and it is worth recording why that
   cannot arise here rather than guarding against it (Codex P2 on PR #342, whose premise this is):
   in EXPRESSION position OCaml gives no [Pexp_ident] an applied path. [Set.Make(String).empty] and
   [F(X).Utils.settings.large_models] both parse as [Pexp_field] over a constructor application,
   with a qualified label -- checked against the parser, not assumed. Since {!longident_of} runs on
   [Pexp_ident] alone, and a record-field label is never an application, every path reaching here is
   [Lident] or [Ldot]. A future parser that changed this would take the test down loudly, which
   beats a fallback silently dropping a read. *)
let flatten_longident = Longident.flatten

let longident_of expr =
  match expr.pexp_desc with Pexp_ident { txt; _ } -> Some (flatten_longident txt) | _ -> None

(** Raises if [content] does not parse: a scan that cannot read its input must say so rather than
    report an empty census. *)
let structure_of content = Parse.implementation (Lexing.from_string content)

(** The ppx_minidebug extension whose argument names a module's compile-time tracing gate. *)
let tracing_gate_extension = "global_debug_log_level_from_env_var"

(** The environment variables [content] reads at RUN time by name: the string literal argument of a
    [Sys.getenv] / [Sys.getenv_opt], under any receiver path ([Stdlib.Sys.getenv_opt], [Sys.getenv]
    with [Base] in scope). A dynamic argument is deliberately not reported -- the reader that takes
    the name as a parameter is {!Utils.read_env_var} itself, whose keys are a different question
    (gh-ocannl-628, Codex P2 round 2 on PR #371).

    Walks expressions, not just structure items, because these reads sit anywhere in a function
    body. *)
let env_var_reads_in_source content =
  let found = ref [] in
  let iterator =
    {
      Ast_iterator.default_iterator with
      expr =
        (fun self expr ->
          (match expr.pexp_desc with
          | Pexp_apply (callee, [ (Asttypes.Nolabel, argument) ]) -> (
              match (longident_of callee, string_literal argument) with
              | Some path, Some name -> (
                  match List.last path with
                  | Some ("getenv" | "getenv_opt") -> found := name :: !found
                  | _ -> ())
              | _ -> ())
          | _ -> ());
          Ast_iterator.default_iterator.expr self expr);
    }
  in
  iterator.structure iterator (structure_of content);
  List.rev !found

(** The environment variables ppx_minidebug reads while preprocessing [content]: the argument of
    each [%%global_debug_log_level_from_env_var] at the top of the file. What a library must declare
    in its [preprocessor_deps] for the gate to invalidate anything (gh-ocannl-628).

    Parsed like everything else here rather than grepped: this file's own doc comments name the
    extension, and a text scan would read a mention as a declaration -- exactly the class of defect
    that made this module parse in the first place. *)
let tracing_gates_in_source content =
  structure_of content
  |> List.filter_map ~f:(fun item ->
      match item.pstr_desc with
      | Pstr_extension (({ txt; _ }, payload), _) when String.equal txt tracing_gate_extension -> (
          match payload with
          | PStr [ { pstr_desc = Pstr_eval (expr, _); _ } ] -> string_literal expr
          | _ -> None)
      | _ -> None)

type label_use = { key : string option; offset : int }
(** Every place the [arg_name] label names a configuration key, and what it names it with. [key] is
    [Some k] when the argument is a string literal — the convention both consistency tests rely on
    to find a read — and [None] when it is anything else: a variable, a punned parameter, an
    expression. [offset] is where the argument sits, for a caller that wants to report the line.

    Both positions count, because both name keys: an argument at an application site
    ([~arg_name:"key"], [?arg_name:"key"]) and a parameter's default ([?(arg_name = "key")], with or
    without a type annotation). A label in a {e type} is not a use of anything and does not appear.
*)

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

type definition = { start : int; stop : int; name : string option; top_level : bool }
(** Every function in the file: the source range it covers, the name of the binding it is the body
    of (if any), and whether that binding is top-level.

    {2 Why functions and not bindings}

    An exemption says a named top-level function is trusted plumbing. The question is therefore
    which function a use sits in, and the answer is the innermost enclosing {e lambda} — not the
    innermost enclosing binding, which is a different thing that merely correlates with it.

    Three rounds of review found the correlation breaking, each in a new spelling: a local
    [let hidden name = …], then a nameless [let result, source = …] that had to stay transparent
    because the real plumbing forwards its key from inside one, then
    [let (hidden as alias) = fun name -> …], whose pattern yields no simple name. Keying on the
    lambda answers all three at once and does not care how the helper was introduced:

    - the plumbing's own [let result, source = resolve_config_value … ~arg_name:n in …] is no lambda
      at all, so the use stays with the host function, as it must;
    - a local helper is a lambda, so the use is its own, whatever pattern binds it;
    - an inline [(fun name -> … ~arg_name:name)] is a lambda with no binding at all, so it has no
      name and can be exempt nowhere — a hole that was open until this round and that no list of
      binding forms would have closed.

    [name] is best-effort and fails safe: a pattern this does not recognise yields no name, and a
    nameless function is refused rather than exempted. The module path is kept for the reader, as
    [top_level] is what an exemption may rely on. *)

(* The simple name a pattern binds, where it has one. Best-effort by design: see above. *)
let rec pattern_name pattern =
  match pattern.ppat_desc with
  | Ppat_var { txt; _ } -> Some txt
  | Ppat_constraint (inner, _) | Ppat_alias (inner, _) -> pattern_name inner
  | _ -> None

let definitions content =
  let ast = structure_of content in
  let root_bindings =
    List.concat_map ast ~f:(fun item ->
        match item.pstr_desc with
        | Pstr_value (_, bindings) ->
            List.map bindings ~f:(fun binding -> binding.pvb_loc.loc_start.pos_cnum)
        | _ -> [])
    |> Set.of_list (module Int)
  in
  (* Which functions are the body of a named binding, and whether that binding is top-level. *)
  let named = Hashtbl.create (module Int) in
  let path = ref [] in
  let qualify name = String.concat ~sep:"." (List.rev (name :: !path)) in
  let within name f =
    let saved = !path in
    path := name :: !path;
    f ();
    path := saved
  in
  let naming =
    {
      Ast_iterator.default_iterator with
      structure_item =
        (fun self item ->
          match item.pstr_desc with
          | Pstr_module { pmb_name = { txt = name; _ }; _ } ->
              within (Option.value name ~default:"_") (fun () ->
                  Ast_iterator.default_iterator.structure_item self item)
          (* Anonymous nesting has no name to borrow; the reader gets a placeholder. Correctness
             does not ride on this list being complete -- [top_level] does that. *)
          | Pstr_recmodule _ | Pstr_open _ | Pstr_include _ | Pstr_extension _ ->
              within "_" (fun () -> Ast_iterator.default_iterator.structure_item self item)
          | _ -> Ast_iterator.default_iterator.structure_item self item);
      value_binding =
        (fun self binding ->
          (match (pattern_name binding.pvb_pat, binding.pvb_expr.pexp_desc) with
          | Some name, Pexp_function _ ->
              Hashtbl.set named ~key:binding.pvb_expr.pexp_loc.loc_start.pos_cnum
                ~data:(qualify name, Set.mem root_bindings binding.pvb_loc.loc_start.pos_cnum)
          | _ -> ());
          Ast_iterator.default_iterator.value_binding self binding);
    }
  in
  naming.structure naming ast;
  let items = ref [] in
  let collecting =
    {
      Ast_iterator.default_iterator with
      expr =
        (fun self expr ->
          (match expr.pexp_desc with
          | Pexp_function _ ->
              let start = expr.pexp_loc.loc_start.pos_cnum in
              let name, top_level =
                match Hashtbl.find named start with
                | Some (name, top_level) -> (Some name, top_level)
                | None -> (None, false)
              in
              items := { start; stop = expr.pexp_loc.loc_end.pos_cnum; name; top_level } :: !items
          | _ -> ());
          Ast_iterator.default_iterator.expr self expr);
    }
  in
  collecting.structure collecting ast;
  List.rev !items

(** The function an offset sits in: the smallest one containing it, so a local helper wins over the
    function that defines it. *)
let definition_at definitions offset =
  List.filter definitions ~f:(fun d -> d.start <= offset && offset < d.stop)
  |> List.min_elt ~compare:(fun a b -> Int.compare (a.stop - a.start) (b.stop - b.start))

(** The keys [content] reads through the label.

    A key that reaches the lookup any other way — through a helper taking the name as a parameter —
    is invisible to this scan, and hence to both tests built on it. That is why
    [test_config_consistency] separately fails any non-literal use of the label outside the handful
    of named functions that implement the lookup. *)
let keys_in_source content =
  List.filter_map (label_uses content) ~f:(fun use -> use.key)
  (* The empty string names no setting, so it is not a key and does not enter a census. It does not
     get to pass unnoticed either: [test_config_consistency] reports it from {!label_uses}, where a
     literal is still visible as a literal. *)
  |> List.filter ~f:(fun key -> not (String.is_empty key))

(** The library sources among a rule's dependencies, sorted and deduplicated.

    Both consistency tests are handed [%{deps}] — the whole dependency set of the rule that runs
    them, flattened into one argument list — because the files they scan are globbed rather than
    enumerated (gh-ocannl-592: a hand-written list says nothing about the file that falls off it,
    and three files did, each staying green because their keys were read from somewhere else too).

    Two kinds of argument are therefore not sources. Anything that is not a [.ml] file: the config
    file the rule depends on, the reference file, the executable itself. And dune's preprocessed
    twin [x.pp.ml] of an [x.ml] that is in the list — those are the ppx expansion of a file already
    scanned, so they would double every census, and they exist only where the library that owns them
    is built, which is what would make the census differ between a machine with the CUDA toolchain
    and one without. A twin is dropped only when its original is present, so a source genuinely
    named [x.pp.ml] is not silently lost.

    Nothing else is filtered, and both tests print how many files they scanned: a glob that stops
    matching shows up as a diff rather than as a quietly smaller census. *)
let sources_among args =
  let sources =
    List.filter args ~f:(String.is_suffix ~suffix:".ml")
    |> List.dedup_and_sort ~compare:String.compare
  in
  let present = Set.of_list (module String) sources in
  List.filter sources ~f:(fun path ->
      match String.chop_suffix path ~suffix:".pp.ml" with
      | Some stem -> not (Set.mem present (stem ^ ".ml"))
      | None -> true)

(** [files] counted per directory, as a line for a golden: which directories the census actually
    came from, and how many files each contributed.

    The alternative to printing it would be a list of permitted roots inside the test — a second
    copy of the globs in [test/operations/dune], which is the hand-maintained list gh-ocannl-592
    removed. Printing makes the scope reviewable instead: a file arriving from somewhere the globs
    do not name shows up as a new directory in the diff, and the per-directory counts move when a
    glob starts or stops matching. The generated sources dune produces in these directories
    ([tensor/parser.ml] from menhir, the [select]ed backend implementations) are part of the count,
    as they are part of the library (Codex P2, round 9 of PR #343). *)
let by_directory files =
  let strip path =
    let rec go p = match String.chop_prefix p ~prefix:"../" with Some p -> go p | None -> p in
    go (Stdlib.Filename.dirname path)
  in
  List.map files ~f:strip
  |> List.sort_and_group ~compare:String.compare
  |> List.map ~f:(fun group -> Printf.sprintf "%s %d" (List.hd_exn group) (List.length group))
  |> String.concat ~sep:", "

(** Basenames that more than one of [files] carries.

    Both tests key something by basename — the forwarding exemptions in [test_config_consistency],
    the codegen-stage module list in [digest_completeness] — which is unambiguous only while the
    files scanned have distinct names. With the scan list enumerated by hand that could not happen
    by accident; with globs over whole directories it can, the day someone adds a [tensor/utils.ml]
    (Codex P2, PR #340 round 10). So it fails where the fix is, rather than silently lending one
    file's exemptions to another. *)
let duplicate_basenames files =
  List.map files ~f:Stdlib.Filename.basename
  |> List.sort_and_group ~compare:String.compare
  |> List.filter_map ~f:(function name :: _ :: _ -> Some name | _ -> None)

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
          (* The field label may itself be module-qualified -- [r.Utils.large_models] is how one
             disambiguates a field name -- so the read is named by its LAST component, not by the
             whole label (Codex P2, round 7). The receiver check stays as it is. *)
          | Pexp_field (record, { txt = field; _ }) -> (
              match longident_of record with
              | Some path when ends_with path [ "Utils"; "settings" ] -> (
                  match List.last (flatten_longident field) with
                  | Some field -> keys := field :: !keys
                  | None -> ())
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

(** Where a file reads a field of a record called [settings] without the [Utils.] receiver that
    {!settings_keys_in_source} recognises — a bare [settings.large_models], or one through an alias
    ([module U = Utils], [let open Utils in …], [include Utils], at any depth).

    That receiver is the convention the settings census rests on, and a read spelled any other way
    vanishes from it: a key read only that way at codegen could then be classified code-borne and
    pass [digest_completeness] unchallenged, which is the misclassification that test exists to
    catch (Codex P2, rounds 14 and 15 of PR #343).

    Resolving aliases and opens is the other way to close this, and a bigger machine than the
    convention needs — no source spells a read that way today, so the convention is checked instead,
    exactly as the string-literal one for [arg_name] is. Policing the READ rather than the scope is
    what makes it precise: [module Lazy = Utils.Lazy] and a qualified record expression such as
    [Utils.{ value; unique_id }] introduce no such read and do not appear here, while
    [Low_level.virtualize_settings.max_visits] is a different record and never did.

    [offset] is where the read sits, for the caller's report. *)
let unqualified_settings_reads content =
  let found = ref [] in
  let is_utils_settings path =
    List.length path >= 2
    && List.equal String.equal (List.drop path (List.length path - 2)) [ "Utils"; "settings" ]
  in
  (* Where `Utils.settings` is used AS a record rather than read through: the receiver of a field
     access is the one place it may appear. Everything else -- `let s = Utils.settings`, `read
     Utils.settings`, storing it, returning it -- puts the reads out of the census's reach, since
     they are then spelled against a name this scan has no reason to know (Codex P2, rounds 16 and
     17 of PR #343). Blessing that one position covers the whole class at once, where naming the
     ways to alias it did not. *)
  let blessed = Hash_set.create (module Int) in
  (* The predicates {!settings_keys_in_source} folds thresholds into are recognised as CALLS, so
     handing one around as a value loses its keys the same way handing the record around does (Codex
     P2, round 20). Their one visible position is the function of an application. *)
  let predicates = [ "debug_log_from_routines"; "with_runtime_debug" ] in
  let ends_in names path =
    List.last path |> Option.value_map ~default:false ~f:(List.mem names ~equal:String.equal)
  in
  let iterator =
    {
      Ast_iterator.default_iterator with
      expr =
        (fun self expr ->
          (match expr.pexp_desc with
          | Pexp_apply (f, _) -> (
              match longident_of f with
              | Some path when ends_in predicates path ->
                  Hash_set.add blessed f.pexp_loc.loc_start.pos_cnum
              | _ -> ())
          | _ -> ());
          (match expr.pexp_desc with
          (* A write is not a read -- the census counts none of these -- but it is the same
             qualified use of the record, and `Utils.settings.k <- v` is how train.ml sets a few. *)
          | Pexp_field (receiver, _) | Pexp_setfield (receiver, _, _) -> (
              match longident_of receiver with
              | Some path when is_utils_settings path ->
                  Hash_set.add blessed receiver.pexp_loc.loc_start.pos_cnum
              | Some path
                when List.last path |> Option.value_map ~default:false ~f:(String.equal "settings")
                ->
                  (* A field read of something CALLED settings but not `Utils.settings`: a bare
                     `settings.k` under an open, or `U.settings.k` under an alias. *)
                  found := expr.pexp_loc.loc_start.pos_cnum :: !found
              | _ -> ())
          | Pexp_ident { txt; _ } when ends_in predicates (flatten_longident txt) ->
              (* Blessed above if this is the function of an application. *)
              found := expr.pexp_loc.loc_start.pos_cnum :: !found
          | Pexp_ident { txt; _ }
            when List.last (flatten_longident txt)
                 |> Option.value_map ~default:false ~f:(String.equal "settings") ->
              (* Any identifier ending in `settings`, not only the qualified one: under a local open
                 the record is spelled bare, and `read settings` hands it on just as `read
                 Utils.settings` does (Codex P2, round 18). Recorded now and discarded below if it
                 turns out to be a blessed receiver -- a field access is visited before its
                 receiver. *)
              found := expr.pexp_loc.loc_start.pos_cnum :: !found
          | _ -> ());
          Ast_iterator.default_iterator.expr self expr);
    }
  in
  iterator.structure iterator (structure_of content);
  List.rev !found |> List.filter ~f:(fun offset -> not (Hash_set.mem blessed offset))

(** Every configuration read of a file — [arg_name] call sites and {!settings_keys_in_source} —
    keyed by file basename, for tests that care {e where} a key is read. Field names that are not
    config keys come along; callers intersect with the registry. *)
let keys_by_file files =
  List.map files ~f:(fun fname ->
      let content = Stdio.In_channel.read_all fname in
      ( Stdlib.Filename.basename fname,
        Set.of_list (module String) (keys_in_source content @ settings_keys_in_source content) ))
