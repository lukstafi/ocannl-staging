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

type reference = { source : string; line : int; identifier : string list }
type exemption = { source : string; identifier : string list; reason : string }

(* Each row exempts one occurrence with one exact identifier, not its whole source. The staleness
   claim below requires every row to be consumed exactly once, so another [rename] in the same file
   is an offender even when it uses the commit primitive's spelling (Codex P2, round 3). *)
let exempt_references =
  [
    {
      source = "arrayjit/lib/atomic_file.ml";
      identifier = [ "Stdlib"; "Sys"; "rename" ];
      reason =
        "implements Atomic_file's single commit primitive; every other OCaml publisher routes \
         through this module";
    };
  ]

let rename_references ~source content =
  let found = ref [] in
  let iterator =
    object
      inherit Ast_traverse.iter as super

      method! expression expression =
        (match Read.longident_of expression with
        | Some identifier
          when Option.value_map (List.last identifier) ~default:false ~f:(String.equal "rename") ->
            found := { source; line = expression.pexp_loc.loc_start.pos_lnum; identifier } :: !found
        | _ -> ());
        super#expression expression
    end
  in
  iterator#structure (Read.structure_of content);
  List.rev !found

(* Permanent controls for every spelling the terminal-name policy deliberately catches. The live
   repository currently exercises only the exempt [Stdlib.Sys.rename]; without these snippets the
   alias/open arms could disappear and shrink the census to a still-green singleton. The unrelated
   module is intentional too: terminal [rename] is conservative, whatever module owns it. *)
let matcher_cases =
  [
    ("Sys", "let publish a b = Sys.rename a b", [ "Sys"; "rename" ]);
    ("Stdlib.Sys", "let publish a b = Stdlib.Sys.rename a b", [ "Stdlib"; "Sys"; "rename" ]);
    ("Unix", "let publish a b = Unix.rename a b", [ "Unix"; "rename" ]);
    ("Stdlib.Unix", "let publish a b = Stdlib.Unix.rename a b", [ "Stdlib"; "Unix"; "rename" ]);
    ("module alias", "module FS = Sys\nlet publish a b = FS.rename a b", [ "FS"; "rename" ]);
    ("opened module", "open Sys\nlet publish a b = rename a b", [ "rename" ]);
    ("local open", "let publish a b = let open Sys in rename a b", [ "rename" ]);
    ("unrelated API", "let publish a b = Other.rename a b", [ "Other"; "rename" ]);
  ]

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
  let exemption_key ({ source; identifier; _ } : exemption) = (source, identifier) in
  let reference_key ({ source; identifier; _ } : reference) = (source, identifier) in
  let remaining =
    ref
      (List.fold exempt_references ~init:Map.Poly.empty ~f:(fun counts exemption ->
           Map.Poly.update counts (exemption_key exemption) ~f:(function
             | None -> 1
             | Some count -> count + 1)))
  in
  let offenders =
    List.filter references ~f:(fun reference ->
        let key = reference_key reference in
        match Map.Poly.find !remaining key with
        | Some count when count > 0 ->
            remaining := Map.Poly.set !remaining ~key ~data:(count - 1);
            false
        | _ -> true)
  in
  List.iter offenders ~f:(fun { source; line; _ } ->
      eprintf
        "%s:%d: raw rename reference bypasses Utils.Atomic_file -- route publication through \
         Atomic_file, or add a named exemption with the reason this rename is unrelated\n"
        source line);
  eprintf "Scanned %d OCaml sources; found %d raw rename reference(s).\n" (List.length sources)
    (List.length references);
  printf "Named raw rename exemptions:\n";
  List.iter exempt_references ~f:(fun { source; identifier; reason } ->
      printf "  %s: %s -- %s\n" source (String.concat ~sep:"." identifier) reason);
  printf "\n";
  Verdict.p_all ~min:8 "every qualified, aliased, opened, and unrelated rename spelling is detected"
    matcher_cases ~f:(fun (label, content, expected) ->
      match rename_references ~source:("synthetic " ^ label) content with
      | [ { identifier; _ } ] -> List.equal String.equal identifier expected
      | _ -> false);
  Verdict.p_empty "no raw rename reference exists outside Atomic_file" ~over:references offenders;
  Verdict.p_all "every named raw rename exemption has a reason" exempt_references
    ~f:(fun { reason; _ } -> not (String.is_empty (String.strip reason)));
  Verdict.p_all "every named raw rename exemption matches exactly one live reference"
    exempt_references ~f:(fun exemption ->
      Map.Poly.find !remaining (exemption_key exemption) |> Option.value ~default:0 |> Int.equal 0)
