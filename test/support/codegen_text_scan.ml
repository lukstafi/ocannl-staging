(** What in this repository pins the TEXT of generated code (gh-ocannl-712).

    A change to code generation -- how a float constant is spelled, how a loop is opened, which
    intrinsic a reduction renders as -- has a blast radius made of two populations that no single
    search finds:

    - {b goldens}, [.expected] files holding emitted kernel or IR source, in [test/] and in
      [arrayjit/test/] both. Scanning one tree and concluding is how gh-ocannl-623's first CI run
      went red: three [arrayjit/test] goldens quote emitted constants and no [test/*] glob sees
      them.
    - {b test sources}, which assert on emitted text from a string literal in the [.ml] rather than
      from a golden -- [Generated.assert_emits ~contains:"…"], or [Generated.read] followed by a
      substring test. No [.expected] scan of any thoroughness finds these, and because they are
      {!Verdict} claims they exit nonzero, so they fail a plain [dune build] rather than only
      [dune runtest]. That is what CI actually tripped on.

    This module decides both, as pure functions over a path and its contents, so that
    [codegen_text_inventory] can run them over the live tree and [codegen_text_scan_cases] can run
    the same code over input built to break it.

    {1 How a golden is recognised}

    Two independent routes, either sufficient.

    By {b extension}: [.c.expected], [.cu.expected], [.hip.expected], [.metal.expected],
    [.ll.expected], [.cd.expected]. These name the artifact they snapshot, so they are members
    whatever they contain -- on a machine without the toolchain such a file can hold a skip notice
    and it is still that backend's snapshot, to be re-recorded when the hardware next runs.

    By {b content markers}: a file whose text carries the syntax of an emitted kernel or of the
    low-level IR dump. Markers are grouped into families (c, cuda, hip, metal, ll, routine-log) so
    that the inventory says which substrate a member needs re-running on, and each is a string that
    only emitted text spells -- [for (int32_t ], [*restrict ], [(float)(], [__global__],
    [threadgroup float], [] := ], [/* end */].

    A marker only counts on a line that is not a {!Verdict} claim. Claim labels are prose ABOUT a
    kernel and routinely quote its vocabulary -- ["padded GPU intrinsics fire against the threadgroup
    fragment: true"] is a verdict, not Metal source -- and a golden made of such lines moves when
    the claim is reworded, never when codegen changes.

    {1 How a source site is recognised}

    A file is a member when it reads generated source at all: through {!Test_utils.Generated}, or by
    opening [build_files/] itself (which two tests predating that module still do, and which the
    inventory tags, since such a read is unchecked for freshness).

    Under each member the inventory itemises the text it pins, found by the same literal-spelling
    discipline the configuration scan uses: the argument of a substring test, {e at the call site}.
    A [Printf.sprintf] format is itemised as well, because a pinned fragment with a hole in it --
    ["< (int)(%d.0))) {"] -- is exactly the context gh-ocannl-623 was found in only by reading a
    failing kernel.

    Anything else the scan reports rather than assumes: a site whose text is computed, or reached
    through a helper that takes it as a parameter, marks the file's itemisation partial. That is the
    documented limit of the discipline, and the reason it is a limit rather than a hole: the FILE is
    still listed, so a codegen change is still told to re-run it -- only the fragment cannot be
    named.

    {1 Reading these sources as OCaml}

    The source half parses, for the reasons {!Config_key_scan}'s header sets out at length: an
    approximation of OCaml has no natural stopping point, the grammar does. The golden half matches
    text, because a golden is not a language -- it is whatever the backend printed. *)

open Base
open Ppxlib.Parsetree
module Ast_traverse = Ppxlib.Ast_traverse
module Asttypes = Ppxlib.Asttypes
module Longident = Ppxlib.Longident
module Parse = Ppxlib.Parse

(* ------------------------------------------------------------------ goldens *)

(** The families a member belongs to. The name is what the inventory prints, and what a reader greps
    for when asking "which backends must re-record". *)
type family = C | Cuda | Hip | Metal | Ll | Routine_log

let family_name = function
  | C -> "c"
  | Cuda -> "cuda"
  | Hip -> "hip"
  | Metal -> "metal"
  | Ll -> "ll"
  | Routine_log -> "routine-log"

let family_rank = function
  | C -> 0
  | Cuda -> 1
  | Hip -> 2
  | Metal -> 3
  | Ll -> 4
  | Routine_log -> 5

(** Extensions that DECLARE the artifact snapshotted, ordered longest-first so that [.cu.expected]
    is tried before a hypothetical [.expected]. A file matching one is a member whatever it
    contains. *)
let declared_extensions =
  [
    (".c.expected", C);
    (".cu.expected", Cuda);
    (".hip.expected", Hip);
    (".metal.expected", Metal);
    (".msl.expected", Metal);
    (".ll.expected", Ll);
    (".cd.expected", Ll);
  ]

(** A line the scan must not read as emitted text: the output of {!Verdict}, whose labels are prose
    about a kernel and freely quote its vocabulary. *)
let is_claim_line line =
  let line = String.rstrip line in
  String.is_prefix line ~prefix:"FAIL: "
  || List.exists [ ": true"; ": false"; ": PASS"; ": FAIL" ] ~f:(fun suffix ->
         String.is_suffix line ~suffix)

(** A low-level-IR loop header, [for i12 = 0 to 4 {]. Recognised by shape rather than by a needle
    because the induction variable carries a number: ["for i"], digits, [" = "]. *)
let is_ll_loop_line line =
  let n = String.length line in
  let rec from i =
    match String.substr_index ~pos:i line ~pattern:"for i" with
    | None -> false
    | Some start ->
        let j = ref (start + String.length "for i") in
        while !j < n && Char.is_digit line.[!j] do
          Int.incr j
        done;
        (!j > start + String.length "for i" && String.is_prefix (String.drop_prefix line !j) ~prefix:" = ")
        || from (start + 1)
  in
  from 0

type test = Needles of string list | Shape of (string -> bool)

type marker = { tag : string; family : family; test : test }

(** The content markers, by family.

    Each needle is a string only emitted text spells. Two temptations were measured and rejected: a
    bare ["threadgroup "] (matches the claim label "against the threadgroup fragment"), and
    ["device "] (matches every [On_device] table and every "device footprint" verdict in the tree).
    The survivors produce no false member over the repository as it stands, and the cases test pins
    the near misses. *)
let markers =
  [
    { tag = "c-for"; family = C; test = Needles [ "for (int32_t " ] };
    { tag = "c-restrict"; family = C; test = Needles [ "*restrict " ] };
    { tag = "c-prec-cast"; family = C; test = Needles [ "(float)("; "(double)("; "(int)(" ] };
    { tag = "c-decl-banner"; family = C; test = Needles [ "/* Local declarations" ] };
    { tag = "c-logic-banner"; family = C; test = Needles [ "/* Main logic. */" ] };
    { tag = "c-align-attr"; family = C; test = Needles [ "__attribute__((aligned" ] };
    {
      tag = "cuda-kernel";
      family = Cuda;
      test = Needles [ "__global__"; "threadIdx."; "blockIdx."; "__shared__"; "__syncthreads(" ];
    };
    {
      tag = "metal-kernel";
      family = Metal;
      test =
        Needles
          [
            "kernel void ";
            "[[kernel]]";
            "threadgroup float";
            "threadgroup half";
            "threadgroup_barrier";
            "simdgroup_";
            "[[thread_position_in_";
          ];
    };
    { tag = "ll-loop"; family = Ll; test = Shape is_ll_loop_line };
    { tag = "ll-assign"; family = Ll; test = Needles [ "] := " ] };
    { tag = "ll-end"; family = Ll; test = Needles [ "/* end */" ] };
    {
      tag = "routine-log";
      family = Routine_log;
      test = Needles [ "{=MAYBE UNINITIALIZED}"; "COMMENT: " ];
    };
  ]

let marker_matches marker line =
  match marker.test with
  | Needles needles -> List.exists needles ~f:(fun n -> String.is_substring line ~substring:n)
  | Shape f -> f line

type golden = {
  path : string;
  by_extension : string option;
      (** The declaring extension, when the file has one: a member whatever it contains. *)
  families : string list;  (** Sorted family names, for the inventory line. *)
  tags : string list;  (** Sorted content-marker tags; empty for an extension-only member. *)
}

(** [classify_golden ~path ~contents] is [Some] when the file pins emitted text. [path] is used for
    its extension only, so it may carry any prefix dune's globs put on it. *)
let classify_golden ~path ~contents =
  let basename = Stdlib.Filename.basename path in
  let by_extension =
    List.find declared_extensions ~f:(fun (ext, _) -> String.is_suffix basename ~suffix:ext)
  in
  let lines = String.split_lines contents |> List.filter ~f:(fun l -> not (is_claim_line l)) in
  let hit marker = List.exists lines ~f:(marker_matches marker) in
  let matched = List.filter markers ~f:hit in
  match (by_extension, matched) with
  | None, [] -> None
  | _ ->
      (* The DECLARING extension is authoritative about which substrate produced the file, and the
         markers are evidence only where there is none. CUDA and HIP spell the same launch
         vocabulary, so a [.hip.expected] matching [cuda-kernel] is HIP text, not CUDA text -- and
         a snapshot named for its backend needs re-recording on that backend whatever else its
         markers say. Where nothing declares, the markers are all there is, and a file carrying
         several dialects (a routine log holds both the IR line and the C rendering) names them all.
      *)
      let families =
        (match by_extension with
        | Some (_, family) -> [ family ]
        | None -> List.map matched ~f:(fun m -> m.family))
        |> List.dedup_and_sort ~compare:(fun a b -> Int.compare (family_rank a) (family_rank b))
        |> List.map ~f:family_name
      in
      Some
        {
          path;
          by_extension = Option.map by_extension ~f:fst;
          families;
          tags = List.map matched ~f:(fun m -> m.tag) |> List.dedup_and_sort ~compare:String.compare;
        }

(* ------------------------------------------------------------- source sites *)

let string_literal expr =
  match expr.pexp_desc with Pexp_constant (Pconst_string (value, _, _)) -> Some value | _ -> None

let longident_of expr =
  match expr.pexp_desc with
  | Pexp_ident { txt; _ } -> Some (Longident.flatten_exn txt)
  | _ -> None

(** Raises if [content] does not parse: a scan that cannot read its input must say so rather than
    report an empty census. *)
let structure_of content = Parse.implementation (Lexing.from_string content)

let path_ends path ~name = match List.last path with Some n -> String.equal n name | None -> false

(** Every unqualified identifier occurring in [expr]: the names taint can travel along. *)
let idents_in expr =
  let found = ref [] in
  let iterator =
    object
      inherit Ast_traverse.iter as super

      method! expression e =
        (match e.pexp_desc with
        | Pexp_ident { txt = Longident.Lident name; _ } -> found := name :: !found
        | _ -> ());
        super#expression e
    end
  in
  iterator#expression expr;
  !found

(** Whether [expr] names any of [names] as the last component of a path qualified by [qualifier].

    The qualifier is required, and that is the whole point: [build_file] is an ordinary name a test
    may bind for itself ([test_safetensors] writes safetensors fixtures through one), while
    [Utils.build_file] is a read of the artifact directory. Keying on the last component alone read
    the first as the second. *)
let names_qualified expr ~qualifier ~names =
  let found = ref false in
  let iterator =
    object
      inherit Ast_traverse.iter as super

      method! expression e =
        (match longident_of e with
        | Some path
          when List.exists names ~f:(fun n -> path_ends path ~name:n)
               && List.exists path ~f:(String.equal qualifier) ->
            found := true
        | _ -> ());
        super#expression e
    end
  in
  iterator#expression expr;
  !found

(** The reader that hands a test its generated source. [Generated.read] under any prefix
    ([Test_utils.Generated.read] included); the module qualifier is required, so a local [read] is
    not mistaken for it. *)
let mentions_generated_read expr =
  let found = ref false in
  let iterator =
    object
      inherit Ast_traverse.iter as super

      method! expression e =
        (match longident_of e with
        | Some path
          when path_ends path ~name:"read"
               && List.exists path ~f:(String.equal "Generated") ->
            found := true
        | _ -> ());
        super#expression e
    end
  in
  iterator#expression expr;
  !found

(** Reads of [build_files/] that do not go through {!Test_utils.Generated}, and so are unchecked for
    freshness: [Utils.build_files_dir], [Utils.build_file]. *)
let direct_artifact_names = [ "build_files_dir"; "build_file" ]

let direct_artifact_qualifier = "Utils"

let reads_artifacts_directly expr =
  names_qualified expr ~qualifier:direct_artifact_qualifier ~names:direct_artifact_names

(** The unlabelled arguments of an application, in order. *)
let positional args =
  List.filter_map args ~f:(fun (label, arg) ->
      match label with Asttypes.Nolabel -> Some arg | _ -> None)

(** The labelled arguments that name a fragment of text to look for. [~substring] and [~contains]
    are the assertion spellings; [~pattern] is [String.substr_index]/[substr_index_all], which the
    tests that COUNT occurrences of a fragment use, and which pins text exactly as an assertion
    does. *)
let text_test_labels = [ "substring"; "contains"; "pattern" ]

type text_test = {
  text : expression;  (** The fragment argument. *)
  tested : expression option;  (** The haystack, when the call takes one positionally. *)
  inherent : bool;
      (** Whether the call is generated-source-testing by construction: {!Generated.assert_emits}
          and {!Generated.assert_omits} name the routine rather than passing the source, and their
          remaining positional argument is the claim, not the haystack. *)
}

let text_test expr =
  match expr.pexp_desc with
  | Pexp_apply (callee, args) -> (
      match
        List.find_map args ~f:(fun (label, arg) ->
            match label with
            | Asttypes.Labelled l when List.exists text_test_labels ~f:(String.equal l) -> Some arg
            | _ -> None)
      with
      | None -> None
      | Some text ->
          let inherent =
            match longident_of callee with
            | Some path ->
                path_ends path ~name:"assert_emits" || path_ends path ~name:"assert_omits"
            | None -> false
          in
          Some { text; tested = (if inherent then None else List.hd (positional args)); inherent })
  | _ -> None

(** The name a simple [let] binds, or [None] for a pattern this scan does not follow. Used for the
    parameter peel, where a position has to be exact. *)
let bound_name pattern =
  match pattern.ppat_desc with Ppat_var { txt; _ } -> Some txt | _ -> None

(** Every variable a pattern binds, tuple and record patterns included. Used for taint, where
    [let values, src = run () in] must taint [src] -- a source reached through a tuple is a source.
*)
let pattern_names pattern =
  let found = ref [] in
  let iterator =
    object
      inherit Ast_traverse.iter as super

      method! pattern p =
        (match p.ppat_desc with Ppat_var { txt; _ } -> found := txt :: !found | _ -> ());
        super#pattern p
    end
  in
  iterator#pattern pattern;
  !found

(* Positional parameters, in order, stopping at the first one this scan does not follow: a labelled
   or optional parameter, or a pattern that is not a plain name. Stopping keeps the positions
   honest -- a call site is matched by argument POSITION, and a parameter skipped rather than
   stopped at would shift every position after it. *)
let peel_params expr =
  let rec go acc expr =
    match expr.pexp_desc with
    | Pexp_function (params, _, body) ->
        let rec take = function
          | [] -> []
          | param :: rest -> (
              match param.pparam_desc with
              | Pparam_val (Asttypes.Nolabel, None, pat) -> (
                  match bound_name pat with Some p -> p :: take rest | None -> [])
              | _ -> [])
        in
        let taken = take params in
        let acc = acc @ taken in
        let complete = List.length taken = List.length params in
        (match body with
        | Pfunction_body inner when complete -> go acc inner
        | Pfunction_body inner -> (acc, inner)
        | Pfunction_cases _ -> (acc, expr))
    | _ -> (acc, expr)
  in
  go [] expr

type binding = { names : string list; params : string list; body : expression }

(** Every [let]-bound value in [expr] or [structure], nested ones included. *)
let bindings_of collect =
  let found = ref [] in
  let iterator =
    object
      inherit Ast_traverse.iter as super

      method! value_binding vb =
        let params, body = peel_params vb.pvb_expr in
        found := { names = pattern_names vb.pvb_pat; params; body } :: !found;
        super#value_binding vb
    end
  in
  collect iterator;
  List.rev !found

let bindings_in_structure structure = bindings_of (fun it -> it#structure structure)
let bindings_in_expression expr = bindings_of (fun it -> it#expression expr)

(** Names carrying generated source, to a fixed point: seeded by [Generated.read] and by a direct
    [build_files/] read, and spreading along [let] bindings whose right-hand side mentions one.

    Scope is deliberately ignored -- a name is tainted for the whole file. A scan that itemises what
    a file pins does not need to know which [let] shadowed which; over-reach costs an extra
    inventory line, under-reach costs a missed pin. *)
let tainted_names bindings =
  let tainted = ref (Set.empty (module String)) in
  let seeded body = mentions_generated_read body || reads_artifacts_directly body in
  let changed = ref true in
  while !changed do
    changed := false;
    List.iter bindings ~f:(fun { names; params = _; body } ->
        if not (List.for_all names ~f:(Set.mem !tainted)) then
          if seeded body || List.exists (idents_in body) ~f:(fun i -> Set.mem !tainted i) then (
            tainted := List.fold names ~init:!tainted ~f:Set.add;
            changed := true))
  done;
  !tainted

type predicate = {
  pred_name : string;
  text_at : int;  (** Position of the pinned-text parameter among the positional arguments. *)
  source_param : string option;
      (** The parameter that carries the generated source, by name, when the predicate takes it
          rather than closing over it. Such a parameter IS generated source inside the predicate's
          body, so a literal the body tests against it -- the ["Main logic"] banner a helper slices
          on -- is a pin like any other, and the name joins the tainted set for the pin walk. *)
  source_at : int option;
      (** Position of the parameter that carries the generated source, when the predicate takes it
          rather than closing over it. Checked at each call site: a predicate is only pinning where
          the haystack it is handed really is generated source. *)
}

(** Which of [params] a name inside [body] derives from, to a fixed point.

    The haystack a predicate tests is not always a parameter spelled at the test: [let has sub s =
    let body = strip s in String.is_substring body ~substring:sub] reaches it through a local
    binding. Following that is what tells this predicate -- which pins -- apart from [let has s =
    String.is_substring backend_name ~substring:s], which tests the backend's NAME and pins nothing.
    A name deriving from exactly one parameter identifies it; from several, the predicate falls back
    to the closing-over-a-tainted-name route. *)
let params_derived_in ~params body =
  let table = Hashtbl.create (module String) in
  let of_expr e =
    List.fold (idents_in e) ~init:(Set.empty (module String)) ~f:(fun acc name ->
        let acc = if List.exists params ~f:(String.equal name) then Set.add acc name else acc in
        match Hashtbl.find table name with Some s -> Set.union acc s | None -> acc)
  in
  let inner = bindings_in_expression body in
  let changed = ref true in
  while !changed do
    changed := false;
    List.iter inner ~f:(fun { names; params = _; body } ->
        let from = of_expr body in
        if not (Set.is_empty from) then
          List.iter names ~f:(fun name ->
              let previous =
                Option.value (Hashtbl.find table name) ~default:(Set.empty (module String))
              in
              if not (Set.equal previous (Set.union previous from)) then (
                Hashtbl.set table ~key:name ~data:(Set.union previous from);
                changed := true)))
  done;
  of_expr

(** Predicates whose literal argument IS a pinned fragment: the [let has s = String.is_substring src
    ~substring:s] idiom and its variants -- one that takes the source as a parameter ([let src_has
    src s = …]), one that reaches it through a local binding, one that counts occurrences with
    [~pattern].

    Also returns the source ranges of the text arguments inside those definitions. Those sites test
    a PARAMETER, not a fragment, and reading them as pins would mark every file using the idiom as
    pinning text the scan cannot name. Skipped by range at the pin walk rather than by skipping the
    whole binding, so a literal a predicate's body pins alongside its parameter still counts. *)
let predicates ~tainted bindings =
  let consumed = ref [] in
  let predicates =
    List.filter_map bindings ~f:(fun { names; params; body } ->
        match (names, params) with
        | [ name ], _ :: _ ->
            let index_of p =
              List.findi params ~f:(fun _ q -> String.equal p q) |> Option.map ~f:fst
            in
            let closes_over_source =
              List.exists (idents_in body) ~f:(fun i -> Set.mem tainted i)
            in
            let derived_params = params_derived_in ~params body in
            let result = ref None in
            let consider { text; tested; inherent } =
              let text_param =
                List.find params ~f:(fun p -> List.exists (idents_in text) ~f:(String.equal p))
              in
              match text_param with
              | None -> ()
              | Some text_param ->
                  consumed :=
                    (text.pexp_loc.loc_start.pos_cnum, text.pexp_loc.loc_end.pos_cnum) :: !consumed;
                  if Option.is_none !result then
                    let source_param =
                      Option.bind tested ~f:(fun tested ->
                          match Set.to_list (derived_params tested) with
                          | [ p ] when not (String.equal p text_param) -> Some p
                          | _ -> None)
                    in
                    let record source =
                      result :=
                        Some
                          {
                            pred_name = name;
                            text_at = Option.value_exn (index_of text_param);
                            source_param = source;
                            source_at = Option.bind source ~f:index_of;
                          }
                    in
                    if Option.is_some source_param then record source_param
                    else if closes_over_source || inherent then record None
            in
            let iterator =
              object
                inherit Ast_traverse.iter as super

                method! expression e =
                  (match text_test e with Some t -> consider t | None -> ());
                  super#expression e
              end
            in
            iterator#expression body;
            !result
        | _ -> None)
  in
  (predicates, !consumed)

type pin = Literal of string | Format of string | Interpolated of string | Computed

(** How a pin argument reads.

    A literal is itself. A [Printf.sprintf] format, and a concatenation with a literal part, are
    fragments with a HOLE in them -- ["(float)(" ^ spelling ^ ")"], ["< (int)(%d.0))) {"] -- and
    both are itemised with the hole shown, because that is the context gh-ocannl-623 was found in
    only by reading a failing kernel. Anything with no literal part at all is [Computed], which
    marks the file's itemisation partial rather than being dropped silently. *)
let rec pin_of_expr expr =
  match string_literal expr with
  | Some text -> Literal text
  | None -> (
      match expr.pexp_desc with
      | Pexp_apply (callee, args) -> (
          match longident_of callee with
          | Some path when path_ends path ~name:"sprintf" || path_ends path ~name:"ksprintf" -> (
              match List.filter_map (positional args) ~f:string_literal with
              | fmt :: _ -> Format fmt
              | [] -> Computed)
          | Some [ "^" ] -> (
              let parts =
                List.map (positional args) ~f:(fun a ->
                    match pin_of_expr a with
                    | Literal text -> Printf.sprintf "%S" text
                    | Interpolated text -> text
                    | Format _ | Computed -> "...")
              in
              let rendered = String.concat ~sep:" ^ " parts in
              if String.is_substring rendered ~substring:"\"" then Interpolated rendered
              else Computed)
          | _ -> Computed)
      | _ -> Computed)

type site = {
  site_path : string;
  pins : string list;  (** Sorted, deduplicated, each rendered ready for the inventory. *)
  partial : bool;  (** Some pinned fragment could not be named at its call site. *)
  direct : bool;  (** Reads [build_files/] without going through {!Test_utils.Generated}. *)
}

let render_pin = function
  | Literal text -> Some (Printf.sprintf "%S" text)
  | Format fmt -> Some (Printf.sprintf "sprintf %S" fmt)
  | Interpolated rendered -> Some rendered
  | Computed -> None

(** [classify_source ~path ~contents] is [Some] when the file reads generated source at all.

    Raises if [contents] does not parse. *)
let classify_source ~path ~contents =
  let structure = structure_of contents in
  let reads_generated = ref false in
  let reads_direct = ref false in
  let scan_reads =
    object
      inherit Ast_traverse.iter as super

      method! expression e =
        (match longident_of e with
        | Some p
          when (path_ends p ~name:"read" || path_ends p ~name:"assert_emits"
              || path_ends p ~name:"assert_omits")
               && List.exists p ~f:(String.equal "Generated") ->
            reads_generated := true
        | Some p
          when List.exists direct_artifact_names ~f:(fun n -> path_ends p ~name:n)
               && List.exists p ~f:(String.equal direct_artifact_qualifier) ->
            reads_direct := true
        | _ -> ());
        super#expression e
    end
  in
  scan_reads#structure structure;
  if not (!reads_generated || !reads_direct) then None
  else
    let bindings = bindings_in_structure structure in
    let tainted = tainted_names bindings in
    let predicates, consumed = predicates ~tainted bindings in
    (* A predicate's source parameter IS generated source, inside that predicate's body. Adding the
       name to the tainted set is how the literals a helper tests against it -- the banner it slices
       on, a second fragment it checks alongside its own argument -- become pins rather than being
       lost with the helper. Names are file-global here, as everywhere in this scan. *)
    let tainted =
      List.fold predicates ~init:tainted ~f:(fun acc p ->
          match p.source_param with Some name -> Set.add acc name | None -> acc)
    in
    let pins = ref [] in
    let is_consumed (e : expression) =
      List.mem consumed
        (e.pexp_loc.loc_start.pos_cnum, e.pexp_loc.loc_end.pos_cnum)
        ~equal:(fun (a, b) (c, d) -> a = c && b = d)
    in
    let record text = if not (is_consumed text) then pins := pin_of_expr text :: !pins in
    let mentions_tainted e =
      List.exists (idents_in e) ~f:(fun i -> Set.mem tainted i) || mentions_generated_read e
    in
    let iterator =
      object
        inherit Ast_traverse.iter as super

        method! expression e =
          (match text_test e with
          | Some { text; tested; inherent } ->
              if inherent || Option.value_map tested ~default:false ~f:mentions_tainted then
                record text
          | None -> (
              match e.pexp_desc with
              | Pexp_apply (callee, args) -> (
                  match longident_of callee with
                  | Some [ name ] -> (
                      match List.find predicates ~f:(fun p -> String.equal p.pred_name name) with
                      | None -> ()
                      | Some predicate -> (
                          let args = positional args in
                          let source_ok =
                            match predicate.source_at with
                            | None -> true
                            | Some j -> (
                                match List.nth args j with
                                | Some a -> mentions_tainted a
                                | None -> false)
                          in
                          match (source_ok, List.nth args predicate.text_at) with
                          | true, Some text -> record text
                          | _ -> ()))
                  | _ -> ())
              | _ -> ()));
          super#expression e
      end
    in
    iterator#structure structure;
    let all = !pins in
    Some
      {
        site_path = path;
        pins = List.filter_map all ~f:render_pin |> List.dedup_and_sort ~compare:String.compare;
        partial = List.exists all ~f:(function Computed -> true | _ -> false);
        direct = !reads_direct;
      }
