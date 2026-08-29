(* gh-ocannl-803: raw rename publication belongs in [Utils.Atomic_file].

   The fourth hand-rolled temp-then-rename in the repository used a fixed [<tar>.part] staging path,
   so concurrent dune runs streamed into one file. The first three copies had already moved behind
   [Atomic_file], but nothing kept a fifth from appearing. This source scan is that ratchet: every
   reference whose terminal identifier is [rename] must live in a named exemption carrying the
   reason it cannot use the helper. Matching the terminal name rather than four qualified paths is
   deliberate: [module FS = Sys; FS.rename] and [let open Sys in rename] are ordinary OCaml
   spellings of the same primitive. An unrelated API with that terminal name earns a reasoned
   exemption instead of becoming a silent escape hatch.

   This is deliberately an OCaml scan. [tools/test-run.sh] has the shell twin (the publication of
   its [last] pointer, near line 710 when this check was written); an OCaml parse cannot reach it,
   and the shell-side discipline remains tracked through gh-ocannl-607 rather than being silently
   claimed here.

   The reader uses ppxlib's parse tree, not text: comments and strings describing [Sys.rename] are
   not calls, while a source spelling the identifier over ordinary whitespace still is. *)

open Base
open Stdio
module Read = Test_utils.Config_key_scan
module Ast_traverse = Ppxlib.Ast_traverse

type reference = { source : string; line : int }

(* A source-level exemption is intentionally conspicuous and carries its reviewable reason. The
   staleness claim below requires every row to cover a reference still present in that source, so a
   migrated or removed site cannot leave a permanent hole. *)
let exempt_sources =
  [
    ( "arrayjit/lib/atomic_file.ml",
      "implements Atomic_file's single commit primitive; every other OCaml publisher routes \
       through this module" );
  ]

let rename_references ~source content =
  let found = ref [] in
  let iterator =
    object
      inherit Ast_traverse.iter as super

      method! expression expression =
        (match Option.bind (Read.longident_of expression) ~f:List.last with
        | Some "rename" ->
            found := { source; line = expression.pexp_loc.loc_start.pos_lnum } :: !found
        | _ -> ());
        super#expression expression
    end
  in
  iterator#structure (Read.structure_of content);
  List.rev !found

let () =
  if Array.length Stdlib.Sys.argv < 2 then (
    eprintf "Usage: %s <workspace_root> <source...>\n" Stdlib.Sys.argv.(0);
    Stdlib.exit 1);
  let base = Test_utils.Dune_stanza_scan.base_dir Stdlib.Sys.argv.(1) in
  let arguments =
    Array.to_list (Array.subo Stdlib.Sys.argv ~pos:2)
    |> List.map ~f:(fun path -> (Test_utils.Dune_stanza_scan.repo_relative base path, path))
  in
  let on_disk = Map.of_alist_reduce (module String) arguments ~f:(fun first _ -> first) in
  let sources = Read.sources_among (List.map arguments ~f:fst) in
  if List.is_empty sources then (
    Verdict.fail "no OCaml sources among the arguments -- the rule's glob matches nothing";
    Stdlib.exit 1);
  let references =
    List.concat_map sources ~f:(fun source ->
        let path = Map.find_exn on_disk source in
        match rename_references ~source (In_channel.read_all path) with
        | references -> references
        | exception exn ->
            Verdict.fail
              (Printf.sprintf "%s does not parse as OCaml, so this scan cannot vouch for it: %s"
                 source (Exn.to_string exn));
            [])
  in
  let exemptions = Map.of_alist_exn (module String) exempt_sources in
  let exercised = ref (Set.empty (module String)) in
  let offenders =
    List.filter references ~f:(fun reference ->
        match Map.find exemptions reference.source with
        | Some _ ->
            exercised := Set.add !exercised reference.source;
            false
        | None -> true)
  in
  List.iter offenders ~f:(fun { source; line } ->
      eprintf
        "%s:%d: raw rename reference bypasses Utils.Atomic_file -- route publication through \
         Atomic_file, or add a named exemption with the reason this rename is unrelated\n"
        source line);
  eprintf "Scanned %d OCaml sources; found %d raw rename reference(s).\n" (List.length sources)
    (List.length references);
  printf "Named raw rename exemptions:\n";
  Map.iteri exemptions ~f:(fun ~key:source ~data:reason -> printf "  %s -- %s\n" source reason);
  printf "\n";
  Verdict.p_empty "no raw rename reference exists outside Atomic_file" ~over:references offenders;
  Verdict.p_all "every named raw rename exemption has a reason" exempt_sources
    ~f:(fun (_, reason) -> not (String.is_empty (String.strip reason)));
  Verdict.p_all "every named raw rename exemption is exercised" exempt_sources
    ~f:(fun (source, _) -> Set.mem !exercised source)
