(** The inventory of everything in this tree that pins the TEXT of generated code (gh-ocannl-712).

    A codegen change -- how a float constant is spelled, how a loop is opened, which intrinsic a
    reduction renders as -- has to re-run every golden that snapshots emitted source and every test
    that asserts on it from a string literal. Before this, working that out was a search, and the
    search was incomplete twice over on gh-ocannl-623: [arrayjit/test/] carries codegen goldens that
    no [test/*.expected] glob sees, and a substring assertion inside a [.ml] is invisible to any
    [.expected] scan however thorough. The second miss is the expensive one -- those assertions are
    {!Verdict} claims, so they exit nonzero and fail a plain [dune build], not merely [dune runtest].

    So the population is enumerated here, and the enumeration is a golden. Adding a pin promotes a
    line; a codegen change reads the list and knows what it must re-run, before pushing rather than
    after a GPU box reports red days later.

    The rules are decided in {!Test_utils.Codegen_text_scan} as pure functions over a path and its
    contents, so [codegen_text_scan_cases] exercises the same code on input built to break it.

    {1 What the golden holds}

    The inventory itself: every member, and under each source site the fragments it pins. That is a
    list that moves when someone adds a pin -- which is the point, and is not the tally gh-ocannl-665
    warns about, because the line that appears names the pin that appeared. The COUNTS go to stderr,
    and what the counts were there for -- the assurance that a scan reporting nothing scanned
    something -- is kept as floors, which fail the run rather than printing (gh-ocannl-701). *)

open Base
open Stdio
module Scan = Test_utils.Codegen_text_scan
module Floors = Test_utils.Scan_floors

(** Lower bounds per scanned root, well below the census of the day they were written. See
    {!Test_utils.Scan_floors}: a glob that breaks goes to zero, a member added moves nothing. *)
let golden_floors = [ ("test", 12); ("arrayjit/test", 2) ]

(** [arrayjit/test] has its own floor, and the reason it once looked as though it could not have
    sites is worth keeping: [test_utils] is a [neural_nets_lib] library, so the arrayjit tests
    cannot link {!Test_utils.Generated} at all. They reach generated text the third way instead --
    calling [C_syntax.compile_proc] and rendering the document -- which is precisely the route the
    first version of this scan did not model, and which made a whole root look empty (Codex P2,
    round 2). *)
let source_floors = [ ("test", 20); ("arrayjit/test", 2) ]

(** Files the globs hand over that are not members of either population, each for a reason that is
    about this scan rather than about them.

    Both are checked to be present below: an exclusion naming a file the globs no longer produce has
    stopped excluding anything and is hiding the next file that takes its name. *)
let excluded =
  [
    ( "test/support/generated.ml",
      "the freshness-checked reader itself -- it opens build_files/ because that is its job, and a \
       codegen change does not re-run it" );
    ( "test/operations/codegen_text_inventory.expected",
      "this scan's own output, which quotes the fragments the sources pin: including it would make \
       the scan's result depend on its own golden, so a promote would take two rounds to converge" );
    ( "test/operations/generated_provenance.ml",
      "the test OF the freshness-checked reader: it writes its artifact contents by hand and checks \
       that a stale or overwritten one is refused, so the strings under it are fixtures no backend \
       emits and a codegen change does not re-run it" );
  ]

let read path = Stdlib.In_channel.with_open_bin path Stdlib.In_channel.input_all

module Dune = Test_utils.Dune_stanza_scan

let () =
  if Array.length Stdlib.Sys.argv < 2 then (
    eprintf "Usage: %s <workspace_root> <file...>\n" Stdlib.Sys.argv.(0);
    Stdlib.exit 1);
  (* Reported repository-relative, opened as dune handed them over: the working directory is the
     rule's own, deep in the build tree. The translation is [Dune_stanza_scan]'s, shared with the
     other scans so that two of them cannot disagree about what a path names. *)
  let base = Dune.base_dir Stdlib.Sys.argv.(1) in
  let arguments =
    Array.to_list (Array.subo Stdlib.Sys.argv ~pos:2)
    |> List.map ~f:(fun path -> (Dune.repo_relative base path, path))
    |> List.dedup_and_sort ~compare:(fun (a, _) (b, _) -> String.compare a b)
  in
  let is_excluded path = List.Assoc.mem excluded path ~equal:String.equal in
  let of_suffix suffix = List.filter arguments ~f:(fun (name, _) -> String.is_suffix name ~suffix) in
  let golden_files = of_suffix ".expected" in
  let all_ml = of_suffix ".ml" in
  let present name = List.exists all_ml ~f:(fun (n, _) -> String.equal n name) in
  let source_files =
    (* Two kinds of argument are a second copy of a source already in the list, and both would make
       the inventory differ between machines rather than between trees.

       dune's preprocessed twin [x.pp.ml] is the ppx expansion of [x.ml], and exists only where the
       library that owns it is built. And a [(select)] target [x.ml] is a COPY of whichever of
       [x.real.ml] / [x.missing.ml] this machine's toolchain selected -- so on a box with Metal it
       holds the real test and on a box without it holds the stub, and the inventory would move with
       the hardware. Both are dropped only when the source they copy is present, so a file genuinely
       named that way is not lost; the [.real.ml] and [.missing.ml] originals are inventoried on
       every machine alike. *)
    List.filter all_ml ~f:(fun (name, _) ->
        match String.chop_suffix name ~suffix:".pp.ml" with
        | Some stem -> not (present (stem ^ ".ml"))
        | None -> (
            match String.chop_suffix name ~suffix:".ml" with
            | Some stem -> not (present (stem ^ ".real.ml") || present (stem ^ ".missing.ml"))
            | None -> true))
  in
  let handed_over = List.map (golden_files @ source_files) ~f:fst in
  let stale =
    List.filter excluded ~f:(fun (path, _) ->
        not (List.mem handed_over path ~equal:String.equal))
  in
  List.iter stale ~f:(fun (path, reason) ->
      Verdict.fail
        (Printf.sprintf
           "the exclusion for %s (%s) names a file the globs no longer hand over -- drop it, or fix \
            the path it was meant to name"
           path reason));
  let by_itself =
    List.filter_map golden_files ~f:(fun (name, on_disk) ->
        if is_excluded name then None
        else Scan.classify_golden ~path:name ~contents:(read on_disk))
  in
  let unparsed = ref [] in
  let sites =
    List.filter_map source_files ~f:(fun (name, on_disk) ->
        if is_excluded name then None
        else
          try Scan.classify_source ~path:name ~contents:(read on_disk)
          with _ ->
            unparsed := name :: !unparsed;
            None)
  in
  (* Goldens that nothing about the file itself made members, paired with the test beside them. See
     Codegen_text_scan.classify_associated: the markers describe whole dumps, and a golden can hold
     emitted text in fragments instead. *)
  let stems =
    List.map sites ~f:(fun s -> (Scan.source_stem s.Scan.site_path, s.Scan.site_path))
    |> Map.of_alist_reduce (module String) ~f:(fun first _ -> first)
  in
  let already = Set.of_list (module String) (List.map by_itself ~f:(fun g -> g.Scan.path)) in
  let associated =
    List.filter_map golden_files ~f:(fun (name, on_disk) ->
        if is_excluded name || Set.mem already name then None
        else
          match String.chop_suffix name ~suffix:".expected" with
          | None -> None
          | Some stem ->
              Option.bind (Map.find stems stem) ~f:(fun source ->
                  Scan.classify_associated ~path:name ~contents:(read on_disk) ~source))
  in
  let goldens = by_itself @ associated in
  List.iter !unparsed ~f:(fun path ->
      Verdict.fail
        (Printf.sprintf
           "%s does not parse as OCaml, so the scan cannot say whether it pins emitted text" path));
  let golden_paths = List.map goldens ~f:(fun g -> g.Scan.path) in
  let site_paths = List.map sites ~f:(fun s -> s.Scan.site_path) in
  Floors.report ~floors:golden_floors ~noun:"golden" ~what:"Goldens holding emitted text" golden_paths;
  Floors.report ~floors:source_floors ~noun:"site" ~what:"Sources pinning emitted text" site_paths;
  eprintf "Files handed over: %d goldens, %d sources.\n" (List.length golden_files)
    (List.length source_files);
  print_endline
    "What in this tree pins the TEXT of generated code (gh-ocannl-712). A codegen change re-runs\n\
     every golden listed here, on the backend whose family it names, and every test source under\n\
     it. The rules are stated in test/support/codegen_text_scan.ml; the counts scanned go to\n\
     stderr, since a tally in a golden moves on every correct addition anywhere (gh-ocannl-665).\n";
  printf "== goldens holding emitted kernel or IR text ==\n";
  printf "roots scanned: %s\n"
    (String.concat ~sep:", " (Floors.roots ~floors:golden_floors golden_paths));
  List.iter
    (List.sort goldens ~compare:(fun a b -> String.compare a.Scan.path b.Scan.path))
    ~f:(fun g ->
      let origin =
        (match g.Scan.by_extension with Some ext -> [ "extension " ^ ext ] | None -> [])
        @ (match g.Scan.tags with
          | [] -> []
          | tags -> [ "markers " ^ String.concat ~sep:" " tags ])
        @ match g.Scan.beside with Some source -> [ "beside " ^ source ] | None -> []
      in
      printf "%s [%s] %s\n" g.Scan.path
        (String.concat ~sep:" " g.Scan.families)
        (String.concat ~sep:"; " origin));
  printf "\n== test sources pinning emitted text ==\n";
  printf "roots scanned: %s\n"
    (String.concat ~sep:", " (Floors.roots ~floors:source_floors site_paths));
  List.iter
    (List.sort sites ~compare:(fun a b -> String.compare a.Scan.site_path b.Scan.site_path))
    ~f:(fun s ->
      let flags =
        (if s.Scan.direct then [ "reads build_files/ directly" ] else [])
        @ (if s.Scan.rendered then [ "renders generated text in memory" ] else [])
        @ if s.Scan.partial then [ "also pins text this scan cannot name" ] else []
      in
      printf "%s%s\n" s.Scan.site_path
        (match flags with [] -> "" | flags -> " [" ^ String.concat ~sep:"; " flags ^ "]");
      List.iter s.Scan.pins ~f:(fun pin -> printf "    %s\n" pin));
  printf "\n";
  let golden_violations =
    Floors.violations ~floors:golden_floors ~noun:"golden"
      ~floors_name:"codegen_text_inventory.golden_floors" golden_paths
  in
  let source_violations =
    Floors.violations ~floors:source_floors ~noun:"site"
      ~floors_name:"codegen_text_inventory.source_floors" site_paths
  in
  List.iter (golden_violations @ source_violations) ~f:Verdict.fail;
  Verdict.p "every scanned root meets its golden floor" (List.is_empty golden_violations);
  Verdict.p "every scanned root meets its source-site floor" (List.is_empty source_violations);
  Verdict.p "every source handed over parsed as OCaml" (List.is_empty !unparsed);
  Verdict.p "every exclusion still names a file the globs hand over" (List.is_empty stale)
