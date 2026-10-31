(* The one relationship the dune-marker grammar rests on and nothing used to state (gh-ocannl-689).

   [Dune_stanza_scan.marker_backends] is the closed vocabulary of the [; ocannl-backend: <word> --
   <reason>] stanza marker, and its closedness is load-bearing: it is what makes [; ocannl-backend:
   metl -- ...] fail instead of reading as a truthful exemption. But the list is written out as
   text, deliberately -- the scanning tests link [arrayjit.utils] and the source scanners, and
   dragging the backend closure into a check that reads dune files would be a bad trade -- so
   nothing in the scanner relates it to the backends OCANNL actually has.

   Both drifts are silent, and the one that matters is not the one it looks like. Remove or rename a
   backend and markers naming the dead one keep passing. Add one, and the CORRECT marker for it is
   rejected as malformed; the author's fix is then to reach for [none], which is a lie the grammar
   accepts. So the failure mode is not a red build someone fixes by editing a list, it is pressure
   toward the least honest classification available.

   This test is where the two lists meet, and it exists as its own executable because that is the
   whole cost being managed: it links the backend closure so that no scanning test has to. It starts
   no context, opens no device and compiles nothing -- [Backends.all_of_backend] is a derived
   constant.

   It has since become the place for every relationship that needs the closure, for the same reason:
   the second one below relates the backends to the capability facts their modules state, the third
   to what the configuration reader writes for each of them. *)

open Base
module Backends = Context.Backends
module Scan = Test_utils.Dune_stanza_scan

let () =
  let sorted l = List.sort l ~compare:String.compare in
  (* Plus [none], which is not a backend: it says the run does not depend on the configured backend
     at all. Every other word must name one. *)
  let expected = sorted ("none" :: List.map Backends.all_of_backend ~f:Backends.backend_name) in
  let admitted = sorted Scan.marker_backends in
  (* Compared as sorted lists rather than as sets, so a word repeated on either side is a mismatch
     too: a vocabulary with a duplicate in it is not the set it is standing in for. *)
  let agree = List.equal String.equal admitted expected in
  if not agree then
    (* The claim is a bare boolean so that the golden stays fixed as backends come and go; the two
       lists go to stderr, where a failing run shows which way they parted. *)
    Stdio.eprintf "the marker admits [%s]; the backends OCANNL has, plus none, are [%s]\n"
      (String.concat ~sep:"; " admitted)
      (String.concat ~sep:"; " expected);
  Verdict.p "the ocannl-backend marker vocabulary is exactly OCANNL's backends plus none" agree

(* The second relationship, gh-ocannl-706 and gh-ocannl-1723. What a compiler decision asks of a
   backend -- whether its [Grid] loops bind a hardware grid or render on the host pool, whether it
   seeds the untuned plan, how it times -- is the backend's [Ir.Backend_intf.backend_facts], stated
   by its module. It used to be read off the backend's NAME by substring ([backend_is_cpu] was
   "contains cc"), so a backend the substrings were never taught read as neither CPU nor GPU, and
   every caller phrased [if on_gpu then ... else ...] took the CPU branch for it; [hipcc] would have
   read as both. Exactly-one is now the type's ([grid_execution] has two constructors), so what is
   left to hold is that every backend's facts are reachable from its name, consistent with one
   another, and that a name is parsed, never classified. *)
let () =
  let module BI = Ir.Backend_intf in
  (* Every name [get_backend] accepts -- the canonical ones and the deprecated aliases -- reaches
     exactly the facts the module of the backend it parses to states. *)
  Verdict.p_all "every accepted backend name reaches the facts its backend's module states"
    Backends.accepted_names ~f:(fun (name, backend) ->
      BI.equal_backend_facts (Context.backend_facts_of_name name) (Backends.backend_facts backend));
  let inconsistent =
    List.filter_map Backends.all_of_backend ~f:(fun b ->
        match BI.facts_inconsistencies (Backends.backend_facts b) with
        | [] -> None
        | broken -> Some (Backends.backend_name b, broken))
  in
  List.iter inconsistent ~f:(fun (name, broken) ->
      Stdio.eprintf "%s states inconsistent facts: %s\n" name (String.concat ~sep:"; " broken));
  Verdict.p_empty "every backend OCANNL has states facts consistent with one another"
    ~over:Backends.all_of_backend inconsistent;
  (* Names the substring predicates classified: [hipcc] as CPU and GPU at once, [cuda_cc] and
     [metal_cc] likewise, [vulkan] as neither. Drawn from outside the corpus, because every backend
     name today parses: a control from the corpus would encode the absence of the violating
     shape. *)
  let refused name =
    match Context.backend_facts_of_name name with
    | (_ : BI.backend_facts) -> false
    | exception Invalid_argument _ -> true
  in
  Verdict.p_all "a name that is not a backend's is refused, never classified by what it contains"
    [ "hipcc"; "cuda_cc"; "metal_cc"; "vulkan" ]
    ~f:refused;
  (* The consistency check's own control: each relation, broken alone on a backend's facts, is
     reported. *)
  let breaks =
    [
      { BI.cc_facts with workgroup_shared_placement = true };
      { BI.cc_facts with seeds_untuned_default = true };
      { BI.cc_facts with queued_timing = Queued_deep };
      { BI.cc_facts with retries_contended_window = true };
      { BI.cc_facts with node_ident_prefix = "t_" };
      { BI.metal_facts with node_ident_prefix = "" };
    ]
  in
  Verdict.p_none "and a record breaking any one relation is reported" breaks ~f:(fun facts ->
      List.is_empty (BI.facts_inconsistencies facts))

(* The third relationship, gh-ocannl-1581. [Config_reads] is what [ocannl_read_config] writes for a
   backend setting -- the canonical name a per-backend golden is named after, the extension an
   emitted-code artifact carries -- and [codegen_text_inventory] names the backend a golden's
   condition means by running it over every backend. Two drifts would be silent. A backend the
   reader was never taught an extension for writes C's, so its kernel snapshots are diffed against
   the C ones and the inventory files a CUDA-shaped condition under [cc]. And a deprecated alias
   resolving otherwise than [Backends.get_backend] resolves it names a log golden after a backend
   the run did not use. *)
let () =
  let names = List.map Backends.all_of_backend ~f:Backends.backend_name in
  Verdict.p_none "every backend OCANNL has has an emitted-code extension the reader writes for it"
    names ~f:(fun name -> Option.is_none (Config_reads.code_extension name));
  Verdict.p_all "the reader writes every backend OCANNL has under its own name" names
    ~f:(fun name ->
      Option.equal String.equal (Config_reads.written ~key:"backend" name) (Some name));
  Verdict.p_all "every deprecated alias the reader resolves names the backend get_backend selects"
    Config_reads.deprecated_aliases ~f:(fun (alias, canonical) ->
      String.equal canonical (Backends.backend_name (Backends.get_backend ~backend_name:alias ())));
  (* The control: a name no backend has is one the reader has no extension for, so the claim above
     is not passing on a function that answers everything. *)
  Verdict.p "a name that is no backend has no extension"
    (Option.is_none (Config_reads.code_extension "vulkan"))
