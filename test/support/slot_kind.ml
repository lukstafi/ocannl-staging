(** Whether a dune invocation can run a stanza that holds a GPU whatever the configuration says
    (gh-ocannl-1004).

    tools/fleet-slot-run.sh declares a batch's fleet slot [--cpu] only when the batch cannot hold a
    GPU. The resolved configuration answers that for every stanza that selects its backend from it
    -- the ones declaring [(env_var OCANNL_BACKEND)] -- but not for the ones that NAME theirs: a
    stanza carrying [; ocannl-backend: cuda -- …] goes at the CUDA backend by name, and its [select]
    arm follows the library's availability, not the environment. So [dune runtest test/operations]
    on a box with cudajit runs [test_cuda_pool_offset] on the GPU while both configurations say cc.
    Those markers are complete by construction -- [env_var_deps] fails a stanza that runs an
    executable and carries neither the variable nor a marker (gh-ocannl-659) -- so the stanzas a run
    can reach, and their markers, are the whole answer.

    What a run reaches is read from the argv the way dune reads it, conservatively:
    - [@dir/alias] builds [alias] in [dir] and every directory below it, [@@dir/alias] in [dir]
      alone; an alias reaches what its stanzas attach to it, closed under the [(alias …)] its [deps]
      name ({!Dune_stanza_scan.aliases_reached_from}), plus the [runtest-<name>] dune generates for
      a [(test)]/[(tests)] stanza and an inline-test library;
    - [runtest]/[test] with directories runs [@runtest] under each, and with none under the root;
    - [build] with no target builds the default alias everywhere, and a path target is taken as
      reaching anything in its directory -- neither is worth modelling finely, and both are answered
      on the side that costs only a wait;
    - any other subcommand runs no test and reaches nothing. [exec] is refused before this is asked
      (its program may pick any backend), and so is a command-line backend flag.

    A dune file this cannot read is itself a GPU answer, for the same reason. *)

open Base
module Scan = Dune_stanza_scan

let gpu_backends = [ "cuda"; "hip"; "metal" ]

type stanza = {
  dir : string;  (** the directory dune applies it in, repository-relative, [""] for the root *)
  attached : string list;  (** the aliases it attaches to or defines *)
  sexp : Sexplib.Sexp.t;
  gpu : string option;  (** the GPU backends its marker names, if any *)
}

(** The aliases a stanza sits on, including the per-stanza one dune generates for a test and an
    inline-test library, which {!Scan.aliases_of} leaves to its callers. *)
let attached_aliases sexp =
  let generated =
    match Scan.head sexp with
    | Some ("test" | "tests") -> List.map (Scan.names_of sexp) ~f:(fun n -> "runtest-" ^ n)
    | Some "library" when Option.is_some (Scan.field sexp "inline_tests") ->
        "runtest" :: List.map (Scan.names_of sexp) ~f:(fun n -> "runtest-" ^ n)
    | _ -> []
  in
  Scan.aliases_of sexp @ Option.to_list (Scan.alias_stanza_name sexp) @ generated

let gpu_of_marker = function
  | Scan.Names_backend (_, { backend; _ }) | Scan.Declares_and_names (_, { backend; _ }) ->
      let named = String.split backend ~on:',' in
      let gpus = List.filter named ~f:(List.mem gpu_backends ~equal:String.equal) in
      if List.is_empty gpus then None else Some (String.concat ~sep:"," gpus)
  | _ -> None

let join dir sub = match (dir, sub) with "", s -> s | d, "" -> d | d, s -> d ^ "/" ^ s

(** The stanzas of one dune file, in [dir]. A marker the contract refuses is not read as absent: it
    raises, and the caller takes the unreadable file as a GPU answer. *)
let stanzas_of ~dir content =
  let contract = Scan.backend_marker_contract content in
  if not (List.is_empty contract.Scan.contract_issues) then
    failwith "a backend marker the env_var_deps contract refuses";
  List.map contract.Scan.contract_stanzas ~f:(fun marked ->
      let st = marked.Scan.marker_stanza in
      let sexp = st.Scan.marked_sexp in
      {
        dir = join dir st.Scan.marked_subdir;
        attached = attached_aliases sexp;
        sexp;
        gpu = gpu_of_marker (Scan.backend_rule_of marked);
      })

type target =
  | Alias of { dir : string; alias : string; recursive : bool }
  | Directory of string  (** anything built in this directory, non-recursively *)

let strip_dot_slash p =
  let p = Option.value (String.chop_prefix p ~prefix:"./") ~default:p in
  Option.value (String.chop_suffix p ~suffix:"/") ~default:p

let alias_target ~recursive spec =
  let spec = strip_dot_slash spec in
  match String.rsplit2 spec ~on:'/' with
  | Some (dir, alias) -> Alias { dir = strip_dot_slash dir; alias; recursive }
  | None -> Alias { dir = ""; alias = spec; recursive }

(* dune options that take a value as the next word, so it is not read as a target. An option this
   does not know only risks its value being read as a directory target -- one with no stanzas, or,
   if it names a real one, a wider answer. *)
let valued_options =
  [
    "-j";
    "--jobs";
    "--root";
    "--profile";
    "--build-dir";
    "-p";
    "--only-packages";
    "--display";
    "--cache";
    "--workspace";
    "--config-file";
    "--instrument-with";
    "--x";
    "--sandbox";
    "--promote";
    "--diff-command";
    "--error-reporting";
    "--action-stdout-on-success";
    "--action-stderr-on-success";
    "--trace-file";
    "--wait-for-filesystem-clock";
  ]

(** The targets a dune argv builds, or [None] for a subcommand that runs no test. *)
let targets argv =
  let rec positional acc = function
    | [] -> List.rev acc
    | "--" :: _ -> List.rev acc
    | opt :: _ :: rest when List.mem valued_options opt ~equal:String.equal -> positional acc rest
    | word :: rest when String.is_prefix word ~prefix:"-" -> positional acc rest
    | word :: rest -> positional (word :: acc) rest
  in
  match argv with
  | ("runtest" | "test") :: rest -> (
      match positional [] rest with
      | [] -> Some [ Alias { dir = ""; alias = "runtest"; recursive = true } ]
      | dirs ->
          Some
            (List.map dirs ~f:(fun d ->
                 Alias { dir = strip_dot_slash d; alias = "runtest"; recursive = true })))
  | "build" :: rest -> (
      match positional [] rest with
      | [] -> Some [ Alias { dir = ""; alias = "default"; recursive = true } ]
      | words ->
          Some
            (List.map words ~f:(fun w ->
                 match String.chop_prefix w ~prefix:"@@" with
                 | Some spec -> alias_target ~recursive:false spec
                 | None -> (
                     match String.chop_prefix w ~prefix:"@" with
                     | Some spec -> alias_target ~recursive:true spec
                     | None ->
                         let w = strip_dot_slash w in
                         let w =
                           match String.chop_prefix w ~prefix:"_build/default/" with
                           | Some rest -> rest
                           | None -> w
                         in
                         Directory
                           (match String.rsplit2 w ~on:'/' with Some (d, _) -> d | None -> "")))))
  | _ -> None

let in_scope ~recursive ~root dir =
  String.equal root dir
  || (recursive && (String.is_empty root || String.is_prefix dir ~prefix:(root ^ "/")))

(** The first GPU stanza [target] reaches, as [(dir, names, backends)]. *)
let reached_gpu stanzas target =
  let gpu_in stanzas_here reached =
    List.find_map stanzas_here ~f:(fun s ->
        match s.gpu with
        | Some backends when List.exists s.attached ~f:(Set.mem reached) ->
            let what =
              match Scan.names_of s.sexp with
              | [] -> "the rule on " ^ String.concat ~sep:"," (Scan.aliases_of s.sexp)
              | names -> String.concat ~sep:"," names
            in
            Some (s.dir, what, backends)
        | _ -> None)
  in
  let by_dir = List.sort_and_group stanzas ~compare:(fun a b -> String.compare a.dir b.dir) in
  List.find_map by_dir ~f:(fun group ->
      let dir = (List.hd_exn group).dir in
      match target with
      | Directory d when String.equal d dir ->
          gpu_in group
            (Set.of_list (module String) (List.concat_map group ~f:(fun s -> s.attached)))
      | Directory _ -> None
      | Alias { dir = root; alias; recursive } ->
          if not (in_scope ~recursive ~root dir) then None
          else
            let sexps = List.map group ~f:(fun s -> s.sexp) in
            (* `default` builds every target in the directory rather than an alias's members, so it
               reaches every stanza there; any other alias reaches its closure. *)
            let reached =
              if String.equal alias "default" then
                Set.of_list (module String) (List.concat_map group ~f:(fun s -> s.attached))
              else Scan.aliases_reached_from sexps alias
            in
            gpu_in group reached)

(** The verdict for [argv], given every dune file of the tree as [(dir, content)]: [Ok None] when
    the run cannot reach a stanza that names a GPU backend, [Ok (Some why)] when it can, and
    [Error why] when a dune file could not be read -- which the caller also treats as a GPU. *)
let verdict ~dune_files argv =
  match targets argv with
  | None -> Ok None
  | Some targets -> (
      let read =
        List.fold_result dune_files ~init:[] ~f:(fun acc (dir, content) ->
            match stanzas_of ~dir content with
            | stanzas -> Ok (stanzas :: acc)
            | exception exn ->
                Error
                  (Printf.sprintf "the dune file in %s is unreadable here (%s)"
                     (if String.is_empty dir then "." else dir)
                     (Exn.to_string exn)))
      in
      match read with
      | Error why -> Error why
      | Ok stanzas ->
          let stanzas = List.concat stanzas in
          Ok
            (List.find_map targets ~f:(fun t ->
                 Option.map (reached_gpu stanzas t) ~f:(fun (dir, names, backends) ->
                     Printf.sprintf "it reaches %s in %s, which names %s" names
                       (if String.is_empty dir then "." else dir)
                       backends))))
