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
    nonzero exit). [msg] should read as the thing that is wrong, without a "FAIL" prefix of its own.
*)
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

(** [pf fmt … b] is {!p} with a COMPUTED label: the format and its arguments render the label, and
    the boolean follows, so the call reads in the same order as [p name b] —
    [Verdict.pf "%s gradients match the oracle" label ok] prints
    ["… gradients match the oracle: true"] and fails the run on [false].

    This is the entry point for the claims gh-ocannl-601 could not convert (gh-ocannl-624): a claim
    whose label names which leg, which epoch or which measured quantity it is about has to build
    that label from an argument, and before this it had no choice but a bare [printf] — which exits
    0 on [false] and lets the failure be [dune promote]d into the golden, the exact hazard the
    literal-label sites were swept for. A computed label is not a weaker claim than a literal one,
    so it must not have a weaker gate.

    The rendered label is what {!p} prints and what a failure names on stderr, so it should read as
    the fact that holds — the same rule as {!p}: phrase it so [true] is the passing reading. *)
let pf fmt = Printf.ksprintf (fun label b -> p label b) fmt

(** [claimf fmt … b] is {!claim} with a computed label: it fails the run without printing anything
    on stdout, for a test that renders the boolean itself in a column layout of its own. Prefer
    {!pf}, which does both; reach for this only where the surrounding line is a table whose shape is
    the point, and bind the boolean to a name used by both the print and the claim, so the two
    cannot drift apart. *)
let claimf fmt = Printf.ksprintf (fun label b -> claim label b) fmt

(** {1 Quantified claims}

    ["every X …"] is TRUE of an empty X, and in a golden that line is byte-identical to one a real
    population passed (gh-ocannl-729). The hole is invisible by construction, and it opens exactly
    where the claim is most worth making: quantified over a DERIVED collection — the seeds a family
    tree yields, the refutations a gate raises, the statements a kernel emits — where the collection
    being empty is a plausible regression rather than an impossible one. A seeding change that
    empties a family silently converts several claims from checks into decoration.

    So a quantified claim goes through one of the combinators below rather than through {!p} applied
    to a [List.for_all]: they carry the non-emptiness guard, which makes the guarded form the
    shortest one to write. A non-empty collection prints exactly what {!p} prints, so goldens keep
    their shape; an empty one prints a DISTINCT [<claim> (empty): false], so a reader sees why the
    line failed without opening the source.

    The single-collection combinators take lists; an array reaches them through [Array.to_list],
    which is free at test scale and keeps this library's surface small. The PAIRWISE one, {!p_all2},
    is the exception and takes arrays — the executed-parity genre it serves reads both sides out of
    [Context.get_values], and it has to compare their LENGTHS, which is the one question a list
    cannot answer without walking it.

    [?min] raises the floor for a site that knows one — ["every one of the four curated tiles …"] is
    a different claim from ["every tile …"], and a menu that silently shrank to one member should
    fail it. In {!p_all}, {!p_none} and {!p_empty} it bounds the COLLECTION, which is where those
    claims can go vacuous; in {!p_exists} it counts WITNESSES, for the reason given there. Below the
    floor the line names the shortfall: [<claim> (only 1 of 4): false]. *)

(* A failure the claim itself could not express: the collection was too small for the quantifier to
   mean anything, so the line says so where a reader will find it rather than printing the bare
   [false] a satisfied-looking claim shares. *)
let short_fail name detail =
  Int.incr failures;
  Stdio.printf "%s (%s): false\n%!" name detail;
  Stdio.eprintf "FAIL: %s (%s): false\n" name detail

(* The one place the POPULATION floor is checked, so that every combinator resting on it reports a
   short collection the same way. [holds] is not evaluated when the floor fails: on an empty
   collection a quantifier answers without looking, and answering is what we are refusing to accept.
   The default floor asks [is_empty] rather than [length], so the common path stays O(1) over a
   collection that can be a whole tensor readback; the length is computed only to report a
   failure. *)
let with_population ?(min = 1) name ~is_empty ~length holds =
  if if min <= 1 then not (is_empty ()) else length () >= min then p name (holds ())
  else
    let length = length () in
    short_fail name (if length = 0 then "empty" else Printf.sprintf "only %d of %d" length min)

let quantified ?min name xs holds =
  with_population ?min name
    ~is_empty:(fun () -> List.is_empty xs)
    ~length:(fun () -> List.length xs)
    holds

(** [p_all name xs ~f] claims that every element of [xs] satisfies [f], and that there is an element
    — the guarded form of [p name (List.for_all xs ~f)]. Reach for it wherever the claim reads
    ["every …"]. *)
let p_all ?min name xs ~f = quantified ?min name xs (fun () -> List.for_all xs ~f)

(** [p_all2 name got want ~f] claims that [got] and [want] agree cell for cell under [f], that they
    have the same length, and that there is a cell — the guarded form of
    [p name (Array.for_all2_exn got want ~f)], and the executed-parity genre's entry point
    (gh-ocannl-746). Reach for it wherever a claim compares a readback against a reference.

    Emptiness is not far-fetched here, which is why the guard matters more than it looks. Both sides
    usually come from [Context.get_values], and a node that stopped being materialized, a reference
    run whose own setup collapsed, or a readback of a virtualized node answers with NOTHING ON BOTH
    SIDES AT ONCE — the reference does not discriminate, because it went through the same path.
    [Array.for_all2_exn [||] [||]] is [true], so the line that reports the strongest kind of check
    AGENTS.md asks for is byte-identical to one a real readback passed.

    A LENGTH MISMATCH is reported as a failed claim, [<claim> (length 12 vs 16): false], rather than
    as the [Invalid_argument] [Array.for_all2_exn] raises: an exception fails the run too, but it
    fails it without a claim line and without naming which claim, and the mismatch is a finding
    about the run (one side was reshaped, or one readback was truncated) that deserves to be read
    off the transcript. Reported before emptiness, so a one-sided empty says which side it was.

    [?min] bounds the COMMON length, as in {!p_all}. *)
let p_all2 ?min name got want ~f =
  let n_got = Array.length got and n_want = Array.length want in
  if n_got <> n_want then short_fail name (Printf.sprintf "length %d vs %d" n_got n_want)
  else
    with_population ?min name
      ~is_empty:(fun () -> n_got = 0)
      ~length:(fun () -> n_got)
      (fun () -> Array.for_all2_exn got want ~f)

(** [p_none name xs ~f] claims that no element of [xs] satisfies [f], and that there is an element —
    the guarded form of [p name (not (List.exists xs ~f))] and of
    [p name (List.is_empty (List.filter xs ~f))]. The mirror of {!p_all}, and the one the
    ["no X is …"] claims want: filtering an empty collection also yields nothing, so the unguarded
    spelling passes on an empty input just as [List.for_all] does. *)
let p_none ?min name xs ~f = quantified ?min name xs (fun () -> not (List.exists xs ~f))

(** [p_empty name ~over xs] claims that the derived collection [xs] is empty, and that the
    collection it was derived from, [over], is not — the guarded form of [p name (List.is_empty xs)]
    where [xs] is a precomputed subset (the invalid seeds, the declined candidates, the offending
    rows) of a population that must itself exist. Prefer {!p_none} where the predicate can simply be
    passed; this is for the sites that keep the derived list around to report it. *)
let p_empty ?min name ~over xs = quantified ?min name over (fun () -> List.is_empty xs)

(** [p_exists name xs ~f] claims that some element of [xs] satisfies [f], and [p_exists ~min:n] that
    at least [n] do. Its [?min] counts WITNESSES where its siblings' counts the population, and the
    asymmetry is the point rather than an inconsistency: for an existential a population floor buys
    nothing, since [List.exists] already answers [false] on an empty collection, so the only reading
    of [~min:2] that adds a constraint is "two of them". Reading it as a population floor is what
    would produce the false green this module exists to prevent — a two-element list with one
    witness passing ["at least two of the seeds pipeline"] (Codex P2, round 1).

    An empty collection still says so: [<claim> (empty): false], because "nothing satisfied [f]" and
    "there was nothing to satisfy it" are different findings. Too few witnesses under an explicit
    floor names the shortfall as [<claim> (only 1 of 2 match): false] — "match", so that the line
    cannot be read as the population shortfall its siblings print. At the default floor of one the
    line is the bare [<claim>: false] {!p} would print: there is no shortfall to name. *)
let p_exists ?(min = 1) name xs ~f =
  if List.is_empty xs then short_fail name "empty"
  else
    let witnesses = List.count xs ~f in
    if witnesses >= min then p name true
    else if min <= 1 then p name false
    else short_fail name (Printf.sprintf "only %d of %d match" witnesses min)

(** [skipped ~backend name] reports a leg the run's backend cannot evaluate: a GPU intrinsic on a
    CPU backend, a tf32 policy outside CUDA. It prints the same stdout line {!p} would — the
    [.expected] goldens are backend-uniform, and a [(test)] stanza diffs stdout ONLY, so stderr is
    free — and announces the skip on stderr, naming the claim. [grep SKIPPED] over a run then
    enumerates exactly what that hardware did not verify. A second
    [OCANNL_VERDICT_SKIP<TAB>executable<TAB>claim] record on stderr is the machine-readable form;
    the executable identity keeps equal labels in different test legs distinct when sweep logs are
    intersected across backends.

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
  (* A second, machine-oriented record gives cross-run consumers a stable test-leg identity without
     changing the human line above (the documented [grep SKIPPED] convention). The executable
     basename is stable across worktrees and machines; pairing it with the claim prevents equal
     labels in different tests from being conflated. [String.escaped] keeps each field on one TSV
     line even if a computed label contains a control character. *)
  Stdio.eprintf "OCANNL_VERDICT_SKIP\t%s\t%s\n%!"
    (Stdlib.String.escaped (Stdlib.Filename.basename Stdlib.Sys.executable_name))
    (Stdlib.String.escaped name);
  p name true

(** [pass_fail label b] prints [label: PASS] or [label: FAIL], and fails the run in the latter case.
    [?detail] is evaluated only on failure and appended in parentheses — the place for a machine-
    specific number (a measured value, a difference) that must stay out of a passing golden. *)
let pass_fail ?detail label b =
  if b then Stdio.printf "%s: PASS\n" label
  else
    let detail = match detail with None -> "" | Some f -> " (" ^ f () ^ ")" in
    Int.incr failures;
    Stdio.printf "%s: FAIL%s\n" label detail;
    Stdio.eprintf "FAIL: %s%s\n" label detail

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
