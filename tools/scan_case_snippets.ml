(* The engine behind tools/typecheck-scan-cases.sh: type-check every OCaml snippet a scan's cases
   module feeds its scanner, the way docs/agent-notes/build-and-test.md (Repository scans) asks
   before an arm is trusted. The parser accepts programs the type checker rejects -- a duplicate
   top-level [module] parses cleanly -- and an arm over such a snippet pins the scanner's reading of
   code no compiler would build.

   WHICH SNIPPETS. A point-in-time reading of the cases module's parse tree, not of its text: a
   snippet is a string constant spelled as a quoted string with the [ocaml] delimiter
   ({ocaml|...|ocaml}), wherever it sits in the module. That delimiter is the shared case-table
   shape: the [*_scan_cases] tables spell their OCaml inputs that way and nothing else. A snippet
   spelled as an ordinary "..." literal (as the arms whose case IS an escape, a line continuation,
   must be), or with another delimiter, is not read. Text in comments is never read.

   ITS NAME. The first ordinary string constant (any delimiter but [ocaml]) among the elements of
   the tuple the snippet is an element of, or among the unlabelled arguments of the application it
   is an argument of, that precedes the snippet -- the two case-table shapes the tables use,
   [("name", {ocaml|...|ocaml}, expected)] and [case "name" {ocaml|...|ocaml} ~flag]. Otherwise the
   snippet is named by its line alone (and cannot be listed below). Every report carries the line. A
   snippet passed as an argument labelled [~interface] is a signature (the [.mli] a case pairs with
   its implementation): it is compiled as an interface, and named after its application's case with
   " (interface)" appended.

   HOW IT IS CHECKED. Each snippet becomes its own compilation unit in a scratch directory, preceded
   by [open! <stub>] when a stub module is given (compiled first, never pasted in: an arm may rebind
   a stub's name, legal over an open and a duplicate definition beside a paste) and by a line
   directive, so a rejection points at the snippet's own line in the cases module. It is compiled
   with [ocamlc -stop-after typing -w -a -c]: warnings are not rejections. A library a snippet or
   the stub opens is reached with [-I] its directory of compiled interfaces ([-I "$(opam var
   lib)/base"]), which is all typing reads.

   KNOWN REJECTIONS. An arm may feed its scanner a parse shape no compiler accepts -- on purpose, to
   pin what the parser makes of a spelling the language does not produce, or as a finding not yet
   settled. The stub lists each such case by name in a floating attribute, [[@@@known_rejection
   ("case name", "text of the error")]], with its reason in a comment beside it, and every run
   prints it. Those snippets are compiled like the rest and must be rejected with output containing
   that text, which must not be empty (every rejection contains the empty text): rejected for
   another reason is a rejection like any other, and accepted, or a listed name that matches no
   checked snippet, is a stale listing; both fail the run. A listed name that matches a REFUSED
   snippet (below) is neither: nothing was compiled to judge it.

   PROJECT SYNTAX IS CHECKED IN ITS OWNING MODULE'S CONTEXT. A snippet carrying an extension node
   ([let%op], [[%cd ...]], any [%name]) or a [[@@deriving ...]] attribute needs preprocessors,
   libraries and DSL opens that bare ocamlc lacks. The stub names, in a floating [[@@@owning_module
   ("lib/nn_blocks.ml", "open Ocannl_tensor.Operation.DSL_modules")]], a module of the repository
   (relative to [--root]) whose stanza such code would live in, and the prelude (its DSL opens, any
   text a top-level item there sees) each such arm starts with; the string alone is a module with an
   empty prelude. That module's directory is built ([dune build @@DIR/check]: its ppx driver, its
   libraries' interfaces, its merlin configuration), and dune's own answer to what compiling that
   module takes ([dune ocaml-merlin]: the [-ppx] driver, the [-open]s and flags, every library's
   [-I]) is the context: the stub is compiled a second time under it, the prelude once alone (a
   prelude that does not compile fails the run), and each project-syntax arm as [PRELUDE; open!
   <stub>; ARM] with the context's flags and [-w -a] after them -- the owning stanza's reading of
   the arm, not a copy of its fields. An extension no preprocessor of that stanza expands ([[%%ext
   ...]]) is then rejected by the compiler like any other ill-typed arm, so an arm meant as such a
   parse shape is a known rejection. Plain arms are compiled bare as before, so a stub must compile
   both ways. Without a declared owning module, or under [--no-project-context], a project-syntax
   arm is REFUSED by name and not compiled. A snippet the parser rejects goes to the compiler
   anyway, whose syntax error is then the rejection.

   Exit status: 0 when every checked snippet was accepted (or rejected as the stub lists); 1 when
   any was rejected unlisted, or a listing is stale; 2 on a usage error, a stub that does not
   compile, an owning module whose context does not build or whose prelude does not compile, or a
   cases module in which no snippet could be checked (no snippet at all, or every one refused) -- a
   run that checked nothing is not a clean verdict. *)

let usage =
  "usage: scan_case_snippets.exe [--stub STUB.ml|STUB.mli] [-I DIR]... [--root DIR] \
   [--no-project-context] [--verbose] CASES.ml..."

let die fmt =
  Printf.ksprintf
    (fun s ->
      prerr_endline ("typecheck-scan-cases: " ^ s);
      exit 2)
    fmt

type snippet = {
  line : int;
  column : int;
  text : string;
  mutable name : string option;
  mutable interface : bool;
}

let read_file path =
  try In_channel.with_open_bin path In_channel.input_all
  with Sys_error msg -> die "cannot read %s" msg

let write_file path contents = Out_channel.with_open_bin path (fun oc -> output_string oc contents)

(* The snippets of one cases module, in source order. *)
let snippets_of ~path source =
  let open Ppxlib in
  let structure =
    try Parse.implementation (Lexing.from_string source)
    with exn -> die "%s does not parse: %s" path (Printexc.to_string exn)
  in
  let found = Hashtbl.create 64 and interfaces = Hashtbl.create 8 in
  let order = ref [] in
  let snippet_key (e : expression) =
    match e.pexp_desc with
    | Pexp_constant (Pconst_string (_, loc, Some "ocaml")) -> Some loc.loc_start.pos_cnum
    | _ -> None
  in
  let plain_string (e : expression) =
    match e.pexp_desc with
    | Pexp_constant (Pconst_string (s, _, delimiter)) when delimiter <> Some "ocaml" -> Some s
    | _ -> None
  in
  (* Name every snippet among [children] by the first ordinary string before it. *)
  let name_siblings children =
    ignore
      (List.fold_left
         (fun name child ->
           match (name, plain_string child) with
           | None, Some s -> Some s
           | _ ->
               (match (name, snippet_key child) with
               | Some n, Some key -> Hashtbl.replace found key (`Named n)
               | _ -> ());
               name)
         None children)
  in
  let collector =
    object
      inherit Ast_traverse.iter as super

      method! expression e =
        (match e.pexp_desc with
        | Pexp_constant (Pconst_string (text, loc, Some "ocaml")) ->
            let key = loc.loc_start.pos_cnum in
            (* The constant's location spans its content, without the delimiters. *)
            let column = loc.loc_start.pos_cnum - loc.loc_start.pos_bol in
            order :=
              (key, { line = loc.loc_start.pos_lnum; column; text; name = None; interface = false })
              :: !order
        | Pexp_tuple elements -> name_siblings elements
        | Pexp_apply (_, args) ->
            let unlabelled =
              List.filter_map (function Nolabel, arg -> Some arg | _ -> None) args
            in
            name_siblings unlabelled;
            List.iter
              (function
                | Labelled "interface", arg -> (
                    match snippet_key arg with
                    | Some key -> (
                        Hashtbl.replace interfaces key ();
                        match List.find_map plain_string unlabelled with
                        | Some n -> Hashtbl.replace found key (`Named (n ^ " (interface)"))
                        | None -> ())
                    | None -> ())
                | _ -> ())
              args
        | _ -> ());
        super#expression e
    end
  in
  collector#structure structure;
  List.rev_map
    (fun (key, snippet) ->
      (match Hashtbl.find_opt found key with
      | Some (`Named n) -> snippet.name <- Some n
      | None -> ());
      snippet.interface <- Hashtbl.mem interfaces key;
      snippet)
    !order

(* The project syntax a snippet carries, if any: what bare ocamlc cannot expand. *)
let project_syntax ~interface text =
  let open Ppxlib in
  let parsed =
    try
      let lexbuf = Lexing.from_string text in
      Some
        (if interface then `Signature (Parse.interface lexbuf)
         else `Structure (Parse.implementation lexbuf))
    with _ -> None
  in
  match parsed with
  | None -> None
  | Some parsed ->
      let seen = ref None in
      let note s = if Option.is_none !seen then seen := Some s in
      let finder =
        object
          inherit Ast_traverse.iter as super

          method! extension ((name, _) as ext) =
            note ("%" ^ name.txt);
            super#extension ext

          method! attribute attr =
            (match attr.attr_name.txt with
            | "deriving" | "deriving_inline" -> note ("[@@" ^ attr.attr_name.txt ^ "]")
            | _ -> ());
            super#attribute attr
        end
      in
      (match parsed with `Signature s -> finder#signature s | `Structure s -> finder#structure s);
      !seen

(* What a stub lists in floating attributes: the cases no compiler is meant to accept,
   [[@@@known_rejection ("case name", "error text")]], and the module whose context its
   project-syntax arms are checked in, [[@@@owning_module ("path", "prelude")]]. *)
type owning = { module_path : string; prelude : string; line : int }

let stub_attributes path =
  let open Ppxlib in
  let source = read_file path in
  let lexbuf = Lexing.from_string source in
  let floating =
    try
      if Filename.check_suffix path ".mli" then
        List.filter_map
          (fun item -> match item.psig_desc with Psig_attribute a -> Some a | _ -> None)
          (Parse.interface lexbuf)
      else
        List.filter_map
          (fun item -> match item.pstr_desc with Pstr_attribute a -> Some a | _ -> None)
          (Parse.implementation lexbuf)
    with exn -> die "the stub %s does not parse: %s" path (Printexc.to_string exn)
  in
  let string e =
    match e.pexp_desc with Pexp_constant (Pconst_string (s, _, _)) -> Some s | _ -> None
  in
  let payload attr =
    match attr.attr_payload with
    | PStr [ { pstr_desc = Pstr_eval (e, _); _ } ] -> Some e
    | _ -> None
  in
  let line attr = attr.attr_loc.loc_start.pos_lnum in
  let known =
    List.filter_map
      (fun attr ->
        if attr.attr_name.txt <> "known_rejection" then None
        else
          match Option.map (fun e -> e.pexp_desc) (payload attr) with
          | Some (Pexp_tuple [ name; error ])
            when Option.is_some (string name) && not (List.mem (string error) [ None; Some "" ]) ->
              Some (Option.get (string name), Option.get (string error))
          | _ ->
              die
                "%s, line %d: [@@@known_rejection] takes a pair of strings: the case's name, and \
                 the non-empty text the compiler's rejection must contain"
                path (line attr))
      floating
  in
  List.iter
    (fun (name, _) ->
      if List.length (List.filter (fun (n, _) -> n = name) known) > 1 then
        die "%s: [@@@known_rejection] lists %S more than once" path name)
    known;
  let owning =
    List.filter_map
      (fun attr ->
        if attr.attr_name.txt <> "owning_module" then None
        else
          let declared module_path prelude =
            match (string module_path, string prelude) with
            | Some m, Some p when m <> "" -> Some { module_path = m; prelude = p; line = line attr }
            | _ -> None
          in
          let read =
            match payload attr with
            | Some { pexp_desc = Pexp_tuple [ module_path; prelude ]; _ } ->
                declared module_path prelude
            | Some e -> declared e (Ast_builder.Default.estring ~loc:e.pexp_loc "")
            | None -> None
          in
          match read with
          | Some o -> Some o
          | None ->
              die
                "%s, line %d: [@@@owning_module] takes a module of the repository, as a non-empty \
                 string, or a pair of it and the prelude the project-syntax arms start with"
                path (line attr))
      floating
  in
  let owning =
    match owning with
    | [] -> None
    | [ o ] -> Some o
    | _ :: o :: _ -> die "%s, line %d: a second [@@@owning_module]" path o.line
  in
  (known, owning)

let run_quiet ~log command = Sys.command (command ^ " >" ^ Filename.quote log ^ " 2>&1") = 0

(* A canonical S-expression, the form [dune ocaml-merlin] replies in. *)
type sexp = Atom of string | List of sexp list

let csexp ~what s =
  let n = String.length s in
  let malformed () = die "%s: not a canonical S-expression: %S" what s in
  let rec value i =
    if i >= n then malformed ()
    else if s.[i] = '(' then items (i + 1) []
    else
      match String.index_from_opt s i ':' with
      | None -> malformed ()
      | Some colon -> (
          match int_of_string_opt (String.sub s i (colon - i)) with
          | Some len when len >= 0 && colon + 1 + len <= n ->
              (Atom (String.sub s (colon + 1) len), colon + 1 + len)
          | _ -> malformed ())
  and items i acc =
    if i >= n then malformed ()
    else if s.[i] = ')' then (List (List.rev acc), i + 1)
    else
      let v, j = value i in
      items j (v :: acc)
  in
  fst (value 0)

(* The flags compiling [owning.module_path] takes, as dune answers its editors: the [-ppx] driver,
   the stanza's [-open]s and flags, and an [-I] per library interface directory -- after building
   that module's directory, which puts all of them in place. *)
let owning_context ~root ~dir ~stub_path owning =
  let source = Filename.concat root owning.module_path in
  if not (Sys.file_exists source) then
    die "%s, line %d: [@@@owning_module] %S: no such file under %s" stub_path owning.line
      owning.module_path root;
  let owning_dir = Filename.dirname owning.module_path in
  let alias = if owning_dir = "." then "@@check" else "@@" ^ owning_dir ^ "/check" in
  let log = Filename.concat dir "owning-build.log" in
  if
    not
      (run_quiet ~log
         (Printf.sprintf "dune build --root %s %s" (Filename.quote root) (Filename.quote alias)))
  then
    die "the owning module %s's context does not build (dune build %s):\n%s" owning.module_path
      alias (read_file log);
  let request = Filename.concat dir "merlin-request"
  and reply = Filename.concat dir "merlin-reply" in
  write_file request (Printf.sprintf "(4:File%d:%s)" (String.length source) source);
  let log = Filename.concat dir "merlin.log" in
  if
    Sys.command
      (Printf.sprintf "dune ocaml-merlin --root %s <%s >%s 2>%s" (Filename.quote root)
         (Filename.quote request) (Filename.quote reply) (Filename.quote log))
    <> 0
  then die "dune ocaml-merlin failed for %s:\n%s" owning.module_path (read_file log);
  let what = "dune ocaml-merlin's configuration of " ^ owning.module_path in
  let entries = match csexp ~what (read_file reply) with List l -> l | Atom _ -> die "%s" what in
  let includes = ref [] and flags = ref [] in
  List.iter
    (function
      | List [ Atom "ERROR"; Atom msg ] -> die "%s: %s" what msg
      | List [ Atom "B"; Atom d ] -> includes := d :: !includes
      | List [ Atom "FLG"; List args ] ->
          flags :=
            List.rev_append
              (List.map (function Atom a -> a | List _ -> die "%s: a nested flag" what) args)
              !flags
      | _ -> ())
    entries;
  if not (List.exists (fun f -> f = "-ppx") !flags) then
    die "%s names no -ppx driver: its stanza has no preprocessor to check project syntax with" what;
  List.map Filename.quote (List.rev !flags)
  @ List.concat_map (fun d -> [ "-I"; Filename.quote d ]) (List.rev !includes)

let () =
  let stub = ref None and includes = ref [] and verbose = ref false in
  let root = ref None and project_context = ref true in
  let cases = ref [] in
  let absolute d = if Filename.is_relative d then Filename.concat (Sys.getcwd ()) d else d in
  let specs =
    [
      ("--stub", Arg.String (fun s -> stub := Some s), "STUB  a module compiled first and opened");
      ( "-I",
        Arg.String (fun d -> includes := absolute d :: !includes),
        "DIR  a directory of compiled interfaces (.cmi) a snippet or the stub opens" );
      ( "--root",
        Arg.String (fun d -> root := Some (absolute d)),
        "DIR  the dune workspace an [@@@owning_module] is read in and built" );
      ( "--no-project-context",
        Arg.Clear project_context,
        " refuse project-syntax arms rather than build their owning module's context" );
      ("--verbose", Arg.Set verbose, " also print every accepted snippet");
    ]
  in
  (try Arg.parse_argv Sys.argv specs (fun f -> cases := f :: !cases) usage with
  | Arg.Bad msg -> die "%s" (String.trim msg)
  | Arg.Help msg ->
      print_string msg;
      exit 0);
  let cases = List.rev !cases in
  if cases = [] then die "no cases module given\n%s" usage;
  let dir = Filename.temp_dir "typecheck-scan-cases" "" in
  at_exit (fun () -> ignore (Sys.command ("rm -rf " ^ Filename.quote dir)));
  (* The stub is compiled in a directory of its own, and each arm in another, so an arm sees the
     stub's interface and nothing a neighbouring arm left behind. *)
  let stub_dir = Filename.concat dir "stub" in
  Sys.mkdir stub_dir 0o755;
  let bare_flags = List.concat_map (fun d -> [ "-I"; Filename.quote d ]) (List.rev !includes) in
  (* Warnings are not rejections: [-w -a] comes after a context's own [-w] flags, so it wins. *)
  let ocamlc ~flags ~cwd ~log args =
    run_quiet ~log
      (Printf.sprintf "cd %s && ocamlc %s -w -a %s" (Filename.quote cwd) (String.concat " " flags)
         args)
  in
  let stub_file =
    match !stub with
    | None -> None
    | Some path ->
        let ext = Filename.extension path in
        if ext <> ".ml" && ext <> ".mli" then die "--stub %s: expected a .ml or .mli file" path;
        Some (path, "scan_case_stub" ^ ext)
  in
  (* Compile the stub into [into] under [flags]; the line that opens it at an arm's top. *)
  let compile_stub ~flags ~into ~context =
    match stub_file with
    | None -> ""
    | Some (path, target) ->
        write_file (Filename.concat into target) (read_file path);
        let log = Filename.concat dir "stub.log" in
        if not (ocamlc ~flags ~cwd:into ~log ("-c " ^ target)) then
          die "the stub %s does not compile%s:\n%s" path context (read_file log);
        "open! Scan_case_stub\n"
  in
  let opener =
    compile_stub ~flags:(("-I " ^ Filename.quote stub_dir) :: bare_flags) ~into:stub_dir ~context:""
  in
  let known, owning = match !stub with Some path -> stub_attributes path | None -> ([], None) in
  (* The owning module's context, built on the first project-syntax arm that needs it: its flags,
     the prelude, and the opener of the stub compiled under it. *)
  let context =
    lazy
      (match (owning, !stub) with
      | Some o, Some stub_path when !project_context ->
          let root =
            match !root with
            | Some r -> r
            | None ->
                die "%s declares [@@@owning_module]: --root names the workspace to build it in"
                  stub_path
          in
          let flags = owning_context ~root ~dir ~stub_path o in
          let ctx_stub_dir = Filename.concat dir "stub-in-context" in
          Sys.mkdir ctx_stub_dir 0o755;
          let flags = flags @ [ "-I"; Filename.quote ctx_stub_dir ] in
          let opener =
            compile_stub ~flags ~into:ctx_stub_dir ~context:(" in the context of " ^ o.module_path)
          in
          let prelude_dir = Filename.concat dir "prelude" in
          Sys.mkdir prelude_dir 0o755;
          write_file
            (Filename.concat prelude_dir "prelude.ml")
            (Printf.sprintf "# 1 %S\n%s\n" (stub_path ^ ", [@@@owning_module]'s prelude") o.prelude);
          let log = Filename.concat dir "prelude.log" in
          if not (ocamlc ~flags ~cwd:prelude_dir ~log "-stop-after typing -c prelude.ml") then
            die "the [@@@owning_module] prelude of %s does not compile in the context of %s:\n%s"
              stub_path o.module_path (read_file log);
          Ok (o, flags, o.prelude ^ "\n", opener)
      | Some _, Some _ -> Error "--no-project-context: not built in its owning module's context"
      | _, Some _ -> Error "the stub declares no [@@@owning_module] to check it in"
      | _, None -> Error "no stub declares an [@@@owning_module] to check it in")
  in
  let contains ~sub s =
    let n = String.length sub and m = String.length s in
    let rec at i = i + n <= m && (String.sub s i n = sub || at (i + 1)) in
    at 0
  in
  let rejected = ref 0 and refused = ref 0 and accepted = ref 0 and known_rejected = ref 0 in
  let stale = ref 0 and matched = ref [] and nothing_checked = ref [] in
  let counter = ref 0 and in_context = ref 0 in
  let print_text text =
    String.split_on_char '\n' (String.trim text)
    |> List.iter (fun l -> Printf.printf "          %s\n" l)
  in
  List.iter
    (fun path ->
      let snippets = snippets_of ~path (read_file path) in
      let checked_here = ref 0 in
      List.iter
        (fun s ->
          let name = match s.name with Some n -> n | None -> "(unnamed snippet)" in
          let label = Printf.sprintf "%s:%d: %s" path s.line name in
          let setting =
            match project_syntax ~interface:s.interface s.text with
            | None -> Ok (bare_flags @ [ "-I"; Filename.quote stub_dir ], opener)
            | Some form -> (
                match Lazy.force context with
                | Ok (_, flags, prelude, opener) ->
                    incr in_context;
                    (* The prelude is top-level code: an interface starts at the stub's open. *)
                    Ok (flags, if s.interface then opener else prelude ^ opener)
                | Error why -> Error (form, why))
          in
          match setting with
          | Error (form, why) ->
              (* A listing of a refused arm is unread, not stale: nothing was compiled to judge
                 it. *)
              if List.mem_assoc name known then matched := name :: !matched;
              incr refused;
              Printf.printf "REFUSED   %s\n          project syntax (%s): %s\n" label form why
          | Ok (flags, header) -> (
              incr counter;
              incr checked_here;
              let unit = Printf.sprintf "arm_%d" !counter in
              let arm_dir = Filename.concat dir unit in
              Sys.mkdir arm_dir 0o755;
              let source = if s.interface then "arm.mli" else "arm.ml" in
              (* The directive puts the snippet at its own line of the cases module, and the padding
                 at its own column, so a rejection points at the arm's text. *)
              write_file (Filename.concat arm_dir source)
                (Printf.sprintf "%s# %d %S\n%s%s\n" header s.line path (String.make s.column ' ')
                   s.text);
              let log = Filename.concat dir (unit ^ ".log") in
              let ok = ocamlc ~flags ~cwd:arm_dir ~log ("-stop-after typing -c " ^ source) in
              let output = read_file log in
              let listing = match s.name with Some n -> List.assoc_opt n known | None -> None in
              if Option.is_some listing then matched := name :: !matched;
              match (ok, listing) with
              | true, None ->
                  incr accepted;
                  if !verbose then Printf.printf "ok        %s\n" label
              | false, None ->
                  incr rejected;
                  Printf.printf "REJECTED  %s\n" label;
                  print_text output
              | false, Some error when contains ~sub:error output ->
                  incr known_rejected;
                  Printf.printf "REJECTED (known: the stub lists it)  %s\n" label;
                  if !verbose then print_text output
              | false, Some error ->
                  incr rejected;
                  Printf.printf "REJECTED  %s\n          not with the stub's listed %S:\n" label
                    error;
                  print_text output
              | true, Some _ ->
                  incr stale;
                  Printf.printf
                    "STALE     %s\n          accepted, though the stub lists it known_rejection\n"
                    label))
        snippets;
      if !checked_here = 0 then nothing_checked := path :: !nothing_checked)
    cases;
  List.iter
    (fun (name, _) ->
      if not (List.mem name !matched) then (
        incr stale;
        Printf.printf
          "STALE     the stub lists known_rejection %S, which names no checked snippet\n" name))
    known;
  (match (owning, !in_context) with
  | Some o, n when n > 0 ->
      Printf.printf "CONTEXT   %d project-syntax snippet(s) checked in the context of %s\n" n
        o.module_path
  | _ -> ());
  Printf.printf
    "typecheck-scan-cases: %d snippet(s) checked: %d accepted, %d rejected, %d known rejection(s); \
     %d refused; %d stale listing(s)\n"
    !counter !accepted !rejected !known_rejected !refused !stale;
  if !nothing_checked <> [] then (
    List.iter
      (fun p -> Printf.eprintf "typecheck-scan-cases: %s: no snippet was checked\n" p)
      (List.rev !nothing_checked);
    exit 2);
  exit (if !rejected + !stale > 0 then 1 else 0)
