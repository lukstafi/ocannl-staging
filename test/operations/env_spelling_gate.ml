(* gh-ocannl-652, Codex P1 on PR #389: the ambient check that a cached test cannot hide.

   Setting a spelling OCANNL does not read aborts the run -- but only a run that HAPPENS. A dune
   rule reruns for an environment variable solely where the stanza declares it, and the rejected
   spellings are declared nowhere any more (that was the point: 227 declarations, growing with
   every test). So `ocannl_backend=cuda dune runtest` would rerun nothing, serve the previous
   run's outputs, and report a green suite -- the fatal check never reached, the user still
   believing the variable decided something.

   This rule closes that for the suite as a whole. Its `(universe)` dependency says "the state of
   the world decides this action", so dune reruns it on every invocation, and running it is enough:
   the check lives in `Utils`' own initialization, so a rejected spelling kills this process before
   `main`, and `dune runtest` goes red naming the variable.

   `(universe)` rather than redeclaring the spellings: the honest dependency is the environment
   itself, and enumerating it would be 112 keys here, a consistency check to keep the list whole,
   and an exemption in `env_var_deps` for declaring names nothing reads -- to buy a narrower
   guarantee than the one line gives. It also covers what the old per-stanza declarations never
   did: those named `backend` (218 stanzas) and eight other keys once each, so a lowercase
   `ocannl_virtualize_max_visits` was served stale everywhere even before this PR.

   What it does not cover, deliberately: a single test run by its own alias
   (`dune build @test/operations/runtest-<name>`) does not build this gate, so a probe pinned that
   way can still be served stale. Running the directory's suite reaches it. *)

open Base
open Stdio

let () =
  (* Unreachable with a rejected spelling set -- `Utils`' initializer exits first -- which is the
     claim: the two must not be able to disagree. It is stated rather than assumed so that making
     the abort conditional some day fails here instead of silently reopening the hole. *)
  let fatal = List.filter (Utils.unread_env_vars ()) ~f:(fun (_, is_fatal, _) -> is_fatal) in
  Verdict.p "no ambient environment variable is a rejected spelling of a known key"
    (List.is_empty fatal);
  List.iter fatal ~f:(fun (name, _, reason) -> printf "  %s %s\n" name reason)
