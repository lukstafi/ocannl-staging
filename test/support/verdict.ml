(** Test verdicts that cannot be [dune promote]d into passing (gh-ocannl-601).

    A [(test)] stanza gates a run on two things: the executable's exit status, and the diff between
    its stdout and the [.expected] golden. A test that decides its own verdict — printing ["PASS"],
    ["FAIL: …"] or ["<claim>: false"] — and exits 0 regardless has only the second gate, and the
    second gate is promotable. A genuine failure changes stdout, the diff fails, and the natural
    next move — [dune promote] — records the failure text as the expected output. Nothing fails
    again until someone reads the golden, and in a golden that is nothing but verdict lines a
    blessed failure is indistinguishable from a designed one: the boolean form is the worst of it,
    since a promoted [false] reads exactly like an intentionally recorded negative fact.

    So verdicts go through here. Each check records its outcome; a run with any failure exits 1 from
    a teardown registered once, whatever the checks are and however many of them there are — so a
    test built on this module gets the exit-status gate by construction, without an end-of-file call
    it could forget. The failures are also echoed to stderr, because dune never writes the
    redirected stdout of a process that exits nonzero: stderr is the channel on which the message
    survives to be read.

    Assertions belong here; descriptive output does not. Printing a fact the golden pins ("losses:
    [1.0; 0.5]", "producer inlined: true" where the point is to record what happened) stays a plain
    [printf]. What comes here is the line whose content IS the pass/fail decision. *)

open Base

let failures = ref 0

(** Records a failure, on stdout (where the golden sees it) and stderr (where it survives the
    nonzero exit). [msg] should read as the thing that is wrong, without a "FAIL" prefix of its
    own. *)
let fail msg =
  Int.incr failures;
  Stdio.printf "FAIL: %s\n" msg;
  Stdio.eprintf "FAIL: %s\n" msg

(** [claim name b] fails the run when [b] is false, printing nothing on stdout: for a test that
    renders the boolean itself, in a column layout of its own. Prefer {!p}, which does both. *)
let claim name b =
  if not b then (
    Int.incr failures;
    Stdio.eprintf "FAIL: %s: false\n" name)

(** [p name b] prints the named boolean fact [name: b] and fails the run when [b] is false. Booleans
    keep [.expected] files backend-stable, which is why so many tests report this way; a fact whose
    expected value is [false] should be renamed to the claim that holds, so that every line of a
    golden reads as an assertion that passed. *)
let p name b =
  (* Flushed, so that a run cut short by a crash still shows every check that had passed -- the
     tests that spelled this helper themselves were split on whether to add [%!], and the flush is
     free at test scale. It cannot reorder anything: flushing empties a buffer, it does not move
     writes past one another. *)
  Stdio.printf "%s: %b\n%!" name b;
  claim name b

(** [skipped ~backend name] reports a leg the run's backend cannot evaluate: a GPU intrinsic on a
    CPU backend, a tf32 policy outside CUDA. It prints the same stdout line {!p} would — the
    [.expected] goldens are backend-uniform, and a [(test)] stanza diffs stdout ONLY, so stderr is
    free — and announces the skip on stderr, naming the claim. [grep SKIPPED] over a run then
    enumerates exactly what that hardware did not verify.

    Use it in place of a bare [p name true]: that line is byte-identical to a verified run's, so
    neither the transcript nor a reviewer can tell the claim was never evaluated — which is how a
    [Tensorize] leg came to "cover" the gh-ocannl-528 interior-batch bug without ever checking it.
    The other honest form is putting the condition into the label itself ("… (skipped: non-C
    backend)"), which distinguishes the golden line; only the indistinguishable bare [true] is the
    one to reject.

    [~backend] is the run's backend name, as each test already derives it for its own gating
    ([String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")]). It is passed in
    rather than read here so that this library keeps depending on nothing but [base] and [stdio]:
    reporting a verdict is not a reason to link OCANNL's configuration machinery. *)
let skipped ~backend name =
  Stdio.eprintf "SKIPPED on %s (vacuous): %s\n%!" backend name;
  p name true

(** [pass_fail label b] prints [label: PASS] or [label: FAIL], and fails the run in the latter case.
    [?detail] is evaluated only on failure and appended in parentheses — the place for a machine-
    specific number (a measured value, a difference) that must stay out of a passing golden. *)
let pass_fail ?detail label b =
  if b then Stdio.printf "%s: PASS\n" label
  else (
    let detail = match detail with None -> "" | Some f -> " (" ^ f () ^ ")" in
    Int.incr failures;
    Stdio.printf "%s: FAIL%s\n" label detail;
    Stdio.eprintf "FAIL: %s%s\n" label detail)

(** Whether any check has failed so far. For a test that wants to say something extra about a bad
    run; the exit status is taken care of without it. *)
let any_failed () = !failures > 0

(* Registered at module initialization, so it covers every test that links this module — including
   one whose checks are all in the middle of the file, or that ends by raising. Calling [exit] from
   an [at_exit] handler is defined: each registered function runs at most once, so the nested
   [do_at_exit] skips this one and proceeds to the rest (stdout's flush among them). *)
let () =
  Stdlib.at_exit (fun () ->
      if !failures > 0 then (
        Stdio.Out_channel.flush Stdio.stdout;
        Stdio.eprintf "FAILED: %d check%s did not hold.\n" !failures
          (if !failures = 1 then "" else "s");
        Stdio.Out_channel.flush Stdio.stderr;
        Stdlib.exit 1))
