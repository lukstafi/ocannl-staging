(** The values that hand a caller the TEXT of generated code, read off the compiler libraries'
    interfaces (gh-ocannl-748).

    {!Test_utils.Codegen_text_scan} recognises a test as reaching generated text in memory when it
    calls one of these. That set used to be written down in the scan -- five names -- and it was the
    one hand-maintained frontier left in the codegen-text inventory: a way of getting emitted text
    into a test that the list does not name does not shrink the inventory {e visibly}, it just
    leaves files off it, and three of the four review rounds on gh-ocannl-712 found a member of
    exactly that shape ([compile_proc], then [compile_main], then [Canonical_render.emit], which
    writes into a buffer instead of returning a document and so was invisible to a list written from
    the first two).

    So the set is derived from what the libraries export, and derived from their {b types} rather
    than from their sources. That distinction is the whole reason this module reads compiled
    interfaces: [C_syntax.compile_proc] carries no return annotation and its module has no [.mli],
    so its result -- a tuple whose second component is the kernel's document -- exists only as an
    inferred type. A source-level scan would have to be told about it, which is the frontier again.
    A [.cmi] holds the type the compiler inferred, whether or not anyone wrote it down.

    {1 What counts}

    An emitter is an exported value whose {b result} mentions [PPrint.document] -- the type every
    renderer in this tree produces -- or which takes a [Buffer.t] {b argument}, the shape
    [Canonical_render.emit] has: the text lands in the caller's buffer instead of coming back. The
    result is what is left of the type after its arrows, tuples and type arguments included, so
    [compile_proc]'s triple counts and [Low_level.lower], which merely {e accepts}
    [ll_source:(PPrint.document -> unit)] callbacks, does not.

    The buffer labels come back with the emitter, because the scan needs them: a call
    [CR.emit ~buf policy llc] deposits generated text in [buf], so [buf] carries the same taint a
    returned document would.

    Over-inclusion is deliberate here, and it is the direction this whole scan errs in: a fragment
    renderer like [C_syntax.pp_scalar] is an emitter by this rule, and if a test calls it, it is
    pinning emitted text. The cost of a name too many is an inventory line; the cost of a name too
    few is a file silently missing from it.

    {1 What is read}

    The caller hands over [.cmi] files. A library's wrapper interface ([ir.cmi]) is a list of module
    ALIASES, which is how this module knows which modules the library has without being told: the
    alias list is the declaration, the member interfaces are what gets walked, and
    [codegen_text_inventory] fails when the two disagree. That is the guard against the silent
    failure a derivation has instead of a stale list -- handed nothing, a scan finds nothing and
    says so cheerfully. *)

open Base

type emitter = {
  name : string;  (** The value's name, which is what a call site spells behind its qualifier. *)
  origins : string list;  (** Every qualified path defining it, sorted. *)
  buffer_labels : string list;
      (** Labels of its [Buffer.t] parameters: the arguments generated text lands in. Sorted;
          [""] stands for an unlabelled one. *)
}

type interface = {
  library : string;  (** The wrapper module's name, [Ir] for [arrayjit.ir]. *)
  declared : string list;  (** The modules its wrapper interface aliases, sorted. *)
  read : string list;  (** Those whose own interface was found and walked, sorted. *)
  missing : string list;  (** Declared but not found: the shape a broken hand-over takes. *)
}

type t = {
  emitters : emitter list;  (** The frontier: what the scan matches call sites against. *)
  combinators : emitter list;
      (** Values that produce a document out of nothing the libraries define -- strings, numbers,
          other documents. Not part of the frontier, and reported rather than dropped: see
          {!classify_value}, and the inventory's golden, which lists them so that a renderer landing
          in this bucket shows up as a line in a diff instead of as an absence. *)
  interfaces : interface list;
}

(* ------------------------------------------------------------------- types *)

let path_components path = String.split (Path.name path) ~on:'.'

(** The type every renderer in this tree produces. Matched on the path's shape rather than on one
    spelling of it: the same type prints as [PPrint.document] and as [PPrint.ToBuffer.document]
    depending on which alias the interface reached for. *)
let is_document path =
  match List.rev (path_components path) with
  | "document" :: _ -> List.exists (path_components path) ~f:(String.equal "PPrint")
  | _ -> false

(** A buffer the caller supplies for the text to land in. [Buffer.t] arrives under whichever module
    the defining file had open -- [Stdlib.Buffer.t], [Base.Buffer.t], bare [Buffer.t] -- so the
    last two components are what identifies it. *)
let is_buffer path =
  match List.rev (path_components path) with "t" :: "Buffer" :: _ -> true | _ -> false

(** The arrow spine of a type: what the value is applied to, and what is left when it is. *)
let rec spine ty =
  match Types.get_desc ty with
  | Types.Tarrow (label, dom, cod, _) ->
      let doms, res = spine cod in
      ((label, dom) :: doms, res)
  | Types.Tlink t | Types.Tsubst (t, _) | Types.Tpoly (t, _) -> spine t
  | _ -> ([], ty)

(** Whether [want] occurs anywhere in [ty], through tuples, type arguments and arrows alike. A
    document reached through a tuple ([compile_proc]) or an option ([render_mma_fragment_scope]) is
    still a document the caller receives. *)
let rec mentions ~want ty =
  match Types.get_desc ty with
  | Types.Tconstr (path, args, _) -> want path || List.exists args ~f:(mentions ~want)
  | Types.Tlink t | Types.Tsubst (t, _) | Types.Tpoly (t, _) -> mentions ~want t
  | Types.Tarrow (_, a, b, _) -> mentions ~want a || mentions ~want b
  | _ -> children ~want ty

(** Every immediate component of a type this module does not name a constructor for -- a tuple's
    parts above all.

    Descending generically rather than by pattern is what makes this compile on every OCaml the opam
    files claim (5.3 up) while missing nothing: [Types.type_desc] moves between releases, and
    [Ttuple] is the one that moved under this scan -- 5.4 gave its components optional labels, so a
    pattern written for either compiler fails to build on the other. [Btype]'s iterator is the
    compiler's own traversal and stays right through such a change; the alternative, a wildcard
    answering [false], would have gone on compiling and quietly stopped seeing [compile_proc]'s
    document, which arrives inside a tuple. *)
and children ~want ty =
  let found = ref false in
  Btype.iter_type_expr (fun component -> if mentions ~want component then found := true) ty;
  !found

(** Whether the value's result HANDS BACK a document, as opposed to accepting one. Through tuples
    and type arguments -- [compile_proc]'s triple, [render_mma_fragment_scope]'s option -- but into
    an arrow only on the way out: [Utils.output_to_build_file], whose result is
    [(PPrint.document -> unit) option], takes documents from its caller and prints them; it renders
    nothing. *)
let rec produces_document ty =
  match Types.get_desc ty with
  | Types.Tconstr (path, args, _) -> is_document path || List.exists args ~f:produces_document
  | Types.Tlink t | Types.Tsubst (t, _) | Types.Tpoly (t, _) -> produces_document t
  | Types.Tarrow (_, _, cod, _) -> produces_document cod
  | _ ->
      (* A tuple's components and anything else composite, through the compiler's own traversal --
         see {!children}, whose reasoning this shares. The arrow is the one case that must NOT go
         through it: what a result accepts is not what it produces. *)
      let found = ref false in
      Btype.iter_type_expr (fun component -> if produces_document component then found := true) ty;
      !found

let label_name = function
  | Asttypes.Nolabel -> ""
  | Asttypes.Labelled l | Asttypes.Optional l -> l

(** Where a type comes from when it comes from outside the code being rendered: the compiler's
    predefined types and the general-purpose libraries every module here has open. Written down
    because it is a fact about the OCaml ecosystem rather than about this repository, and because
    every mistake in it errs toward including one name too many -- a type this list forgets makes
    its function look like a renderer, which costs an inventory line, where naming the libraries'
    OWN modules instead would silently miss any type reached through a local alias ([Tn.t] for
    [Tnode.t], as [C_syntax] spells it throughout). *)
let foreign_roots = [ "Stdlib"; "Base"; "Caml"; "PPrint"; "Sexplib"; "Sexplib0"; "Ppx_sexp_conv_lib" ]

(** Whether the type is one of the values being RENDERED rather than a general-purpose one: an IR
    node, a precision, an operator. Anything not predefined and not from {!foreign_roots}, including
    a type an interface spells unqualified -- [Canonical_render.emit] takes a [t] and a [policy],
    both bare identifiers inside [Low_level]'s own interface. *)
let is_rendered_type path =
  match path with
  | Path.Pident id -> not (Ident.is_predef id)
  | _ -> (
      match path_components path with
      | head :: _ ->
          not
            (List.exists foreign_roots ~f:(fun root ->
                 String.equal head root || String.is_prefix head ~prefix:(root ^ "__")))
      | [] -> false)

type kind = Renders | Combines

(** What [name] is, or [None] when its type produces no text at all.

    {!Renders} is the frontier; {!Combines} is a document built out of nothing the libraries define.

    Two conditions, and the second is what keeps the set to renderers. An emitter hands back a
    document (or takes the buffer to write it into), AND it is given something to render -- a type
    that is neither predefined nor from a general-purpose library. Without that second condition
    every document COMBINATOR is a member: [Indexing.Doc_helpers.int : int -> PPrint.document]
    builds a document out of an integer and nothing else, and since the scan matches an emitter by
    NAME behind any qualifier, admitting it made members of every test calling [Bench_args.int] or
    [Random.State.int]. A combinator renders no program, so nothing about it moves when code
    generation does.

    The limit that leaves: a renderer of something CONSTANT -- [unit -> PPrint.document] for a fixed
    preamble -- is given nothing and would not be derived. None exists today, and the shape is
    visible in this rule rather than hidden in a list. *)
let classify_value ~origin ~name val_type =
  let doms, result = spine val_type in
  let buffer_labels =
    List.filter_map doms ~f:(fun (label, dom) ->
        if mentions ~want:is_buffer dom then Some (label_name label) else None)
    |> List.dedup_and_sort ~compare:String.compare
  in
  let given_something_to_render =
    List.exists doms ~f:(fun (_, dom) -> mentions ~want:is_rendered_type dom)
  in
  if produces_document result || not (List.is_empty buffer_labels) then
    Some
      ((if given_something_to_render then Renders else Combines), { name; origins = [ origin ]; buffer_labels })
  else None

(* -------------------------------------------------------------- signatures *)

type tables = {
  renders : (string, emitter) Hashtbl.t;
  combines : (string, emitter) Hashtbl.t;
}
(** The two buckets, filled by one walk. *)

(** The values an interface exports, with the modules and module types it nests.

    Module types are walked as well as modules: a signature is what some module implements, and the
    backends implement [C_syntax_config] -- so a renderer declared only there is still one a call
    site can reach. *)
let rec walk_signature ~prefix ~found items =
  List.iter items ~f:(fun item ->
      match item with
      | Types.Sig_value (id, vd, _) -> (
          let name = Ident.name id in
          match classify_value ~origin:(prefix ^ name) ~name vd.Types.val_type with
          | None -> ()
          | Some (kind, value) ->
              (* One name, one entry per bucket: [to_doc] is three modules' renderer, and a name that
                 both renders and combines (through different modules) belongs on the frontier, so
                 the buckets are kept apart and the frontier wins where they overlap. *)
              let table = match kind with Renders -> found.renders | Combines -> found.combines in
              let key = name in
              let merged =
                match Hashtbl.find table key with
                | None -> value
                | Some previous ->
                    {
                      value with
                      origins = previous.origins @ value.origins;
                      buffer_labels = previous.buffer_labels @ value.buffer_labels;
                    }
              in
              Hashtbl.set table ~key ~data:merged)
      | Types.Sig_module (id, _, md, _, _) ->
          walk_module_type ~prefix:(prefix ^ Ident.name id ^ ".") ~found md.Types.md_type
      | Types.Sig_modtype (id, md, _) ->
          Option.iter md.Types.mtd_type
            ~f:(walk_module_type ~prefix:(prefix ^ Ident.name id ^ ".") ~found)
      | Types.Sig_type _ | Types.Sig_typext _ | Types.Sig_class _ | Types.Sig_class_type _ -> ())

and walk_module_type ~prefix ~found = function
  | Types.Mty_signature items -> walk_signature ~prefix ~found items
  (* A functor's parameter is what the caller supplies; its BODY is what the application exports,
     and [C_syntax.C_syntax] -- where [compile_proc] lives -- is exactly such a body. *)
  | Types.Mty_functor (_, body) -> walk_module_type ~prefix ~found body
  (* A named signature ([Mty_ident]) and an alias ([Mty_alias]) both point elsewhere; the elsewhere
     is a module of the same library, walked in its own right. Open for the same reason the type
     match above is: [Types.module_type] differs between the compilers this builds on. *)
  | _ -> ()

(** The modules a wrapper interface aliases: [module C_syntax = Ir__C_syntax], one per module of the
    library. Top-level aliases only -- an alias inside a member module is that module's local
    shorthand ([module Tn = Tnode]), and following those would demand the interface of every library
    this one depends on. *)
let top_level_aliases items =
  List.filter_map items ~f:(function
    | Types.Sig_module (_, _, { Types.md_type = Types.Mty_alias path; _ }, _, _) ->
        Some (Path.name path)
    | _ -> None)

(** The member module an alias target names, or [None] for an alias that leaves the library.

    Dune prefixes a wrapped library's modules with the library's own name --
    [Ir__C_syntax], [Utils__Datatypes] -- and an alias may reach past the module to something inside
    it ([module Tree_map = Utils__Datatypes.Tree_map], [module Cpu_topology = Utils__.Cpu_topology],
    where [Utils__] is dune's own alias module). What identifies the member is the head of the path
    under the library's prefix, so both spellings answer the same module, and an alias to another
    library answers nothing. *)
let member_of ~library target =
  let prefix = library ^ "__" in
  match String.split target ~on:'.' with
  | head :: rest when String.equal head prefix -> (
      match rest with first :: _ -> Some (prefix ^ first) | [] -> None)
  | head :: _ when String.is_prefix head ~prefix && String.length head > String.length prefix ->
      Some head
  | _ -> None

(* ------------------------------------------------------------------ inputs *)

(** The module a [.cmi] holds, by the file's name: dune writes [Ir__C_syntax] to [ir__C_syntax.cmi].
*)
let module_of_path path =
  String.capitalize (Stdlib.Filename.remove_extension (Stdlib.Filename.basename path))

(** The library a module belongs to: itself, with dune's trailing [__] taken off. A library whose
    modules include one of its own name ([utils.ml] in [arrayjit.utils]) keeps that module as
    [Utils] and puts the aliases in [Utils__]; one that does not ([arrayjit.ir]) puts them in [Ir].
    Stripping the underscores files both under one library. *)
let library_of_module m = Option.value (String.chop_suffix m ~suffix:"__") ~default:m

(** A wrapper interface is a library's own top-level module rather than a member: [ir.cmi] and
    [utils.cmi] / [utils__.cmi], beside [ir__Low_level.cmi]. Dune's naming is what tells them apart,
    and a change to it does not pass silently -- the wrappers are where the module list comes from,
    so no wrapper means no declared modules, which [codegen_text_inventory] reports as a failure
    rather than as an empty census. *)
let is_wrapper path =
  not (String.is_substring (library_of_module (module_of_path path)) ~substring:"__")

let read_signature path = (Cmi_format.read_cmi path).Cmi_format.cmi_sign

(** [derive paths] is the frontier the handed-over interfaces declare.

    [paths] are [.cmi] files: the wrappers of the libraries to scan, and the member interfaces. A
    member the wrappers declare but the hand-over omits is looked for beside its wrapper, which is
    what makes the package-filtered build work: there the libraries resolve from the switch, where
    the whole installed directory is on disk, while an in-workspace build hands over the object
    directory. Whatever is still missing is reported, never assumed absent. *)
let derive paths =
  let found =
    { renders = Hashtbl.create (module String); combines = Hashtbl.create (module String) }
  in
  let available =
    List.map paths ~f:(fun path -> (module_of_path path, path))
    |> Map.of_alist_reduce (module String) ~f:(fun first _ -> first)
  in
  let prefix_of target =
    String.substr_replace_all (library_of_module target) ~pattern:"__" ~with_:"." ^ "."
  in
  let wrappers =
    (* Deduplicated by module: the same interface can arrive twice, once from the object directory
       and once from the installed one. Which copy wins decides only where members are looked for
       beside it, and either directory answers with the same library. *)
    List.filter paths ~f:is_wrapper
    |> List.map ~f:(fun path -> (module_of_path path, path))
    |> Map.of_alist_reduce (module String) ~f:(fun first _ -> first)
    |> Map.to_alist
    |> List.sort_and_group ~compare:(fun (a, _) (b, _) ->
           String.compare (library_of_module a) (library_of_module b))
  in
  let interfaces =
    List.map wrappers ~f:(fun group ->
        let library = library_of_module (fst (List.hd_exn group)) in
        let declared =
          List.concat_map group ~f:(fun (wrapper_module, wrapper) ->
              let items = read_signature wrapper in
              walk_signature ~prefix:(prefix_of wrapper_module) ~found items;
              List.filter_map (top_level_aliases items) ~f:(member_of ~library))
          |> List.dedup_and_sort ~compare:String.compare
        in
        let directories = List.map group ~f:(fun (_, wrapper) -> Stdlib.Filename.dirname wrapper) in
        let read, missing =
          List.partition_map declared ~f:(fun target ->
              let beside =
                List.find_map directories ~f:(fun directory ->
                    let candidate =
                      Stdlib.Filename.concat directory (String.uncapitalize target ^ ".cmi")
                    in
                    if Stdlib.Sys.file_exists candidate then Some candidate else None)
              in
              match (Map.find available target, beside) with
              | Some path, _ -> First (target, path)
              | None, Some path -> First (target, path)
              | None, None -> Second target)
        in
        List.iter read ~f:(fun (target, path) ->
            walk_signature ~prefix:(prefix_of target) ~found (read_signature path));
        { library; declared; read = List.map read ~f:fst; missing })
  in
  let collect table =
    Hashtbl.data table
    |> List.map ~f:(fun e ->
           {
             e with
             origins = List.dedup_and_sort e.origins ~compare:String.compare;
             buffer_labels = List.dedup_and_sort e.buffer_labels ~compare:String.compare;
           })
    |> List.sort ~compare:(fun a b -> String.compare a.name b.name)
  in
  let emitters = collect found.renders in
  (* A name that renders under one module and merely combines under another is on the frontier: the
     scan matches names, and the rendering spelling is the one that has to be caught. *)
  let combinators =
    List.filter (collect found.combines) ~f:(fun c ->
        not (List.exists emitters ~f:(fun e -> String.equal e.name c.name)))
  in
  { emitters; combinators; interfaces }
