(* The one relationship the dune-marker grammar rests on and nothing used to state (gh-ocannl-689).

   [Dune_stanza_scan.marker_backends] is the closed vocabulary of the [; ocannl-backend: <word> --
   <reason>] stanza marker, and its closedness is load-bearing: it is what makes
   [; ocannl-backend: metl -- ...] fail instead of reading as a truthful exemption. But the list is
   written out as text, deliberately -- the scanning tests link [arrayjit.utils] and the source
   scanners, and dragging the backend closure into a check that reads dune files would be a bad
   trade -- so nothing in the scanner relates it to the backends OCANNL actually has.

   Both drifts are silent, and the one that matters is not the one it looks like. Remove or rename a
   backend and markers naming the dead one keep passing. Add one, and the CORRECT marker for it is
   rejected as malformed; the author's fix is then to reach for [none], which is a lie the grammar
   accepts. So the failure mode is not a red build someone fixes by editing a list, it is pressure
   toward the least honest classification available.

   This test is where the two lists meet, and it exists as its own executable because that is the
   whole cost being managed: it links the backend closure so that no scanning test has to. It starts
   no context, opens no device and compiles nothing -- [Backends.all_of_backend] is a derived constant. *)

open Base
module Backends = Context.Backends_deprecated
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
