(** Freshness-checked reads of the generated-kernel artifacts under [build_files/] (gh-ocannl-655).

    Tests that assert on emitted code read [build_files/<exe>/<routine>.<ext>]. The artifact is a
    side effect of a compile, not a value the test holds, and nothing in the path ties it to the
    compile it is supposed to describe:

    - [test/config/ocannl_config] sets [clean_up_build_files_on_startup=false], so an artifact
      outlives the run that produced it, indefinitely. A test asserting "the kernel contains
      [single_to_fp8]" keeps passing after the kernel stops being emitted at all — folded to a
      constant, erased by precision inference, fissioned into a differently-named routine — because
      it is reading the previous run's file. A structural check that has quietly stopped
      structurally checking is worth less than no check, because it is counted as coverage.
    - Within one run, two compiles under the same routine name overwrite each other's artifact. A
      loop that compiles several candidates under one name and credits each with what it finds on
      disk afterwards is crediting candidate N with candidate N-1's source whenever a compile skips
      emission (a schedule-cache hit, a reused artifact, a conditional write).

    So this module owns the read, and makes both cases fail rather than pass:

    - {!init} — called once, before any compile — empties this process's own [build_files/<exe>/]
      subdirectory. Every artifact present afterwards was written by this run, so existence *is*
      freshness and no clock is involved. The deletion is scoped to the per-executable subdirectory
      [Utils.build_files_dir] resolves to, which no other process writes to; a run configured for
      the flat legacy layout ([build_files_prefix=.], shared between concurrent tests) is rejected
      instead of being cleaned.
    - {!read} fails the run (through {!Verdict}, so the exit status carries it and no [dune promote]
      can bless it) when the artifact is missing. Silence must not be indistinguishable from a pass,
      which is what a [string option] returning [None] invited: some call sites recorded that as
      [false], others forgot it.
    - Reading one routine twice, with the artifact's contents having changed in between, is reported
      as an overwrite: within a run that is a same-named second routine, or an unarmed second
      candidate. {!arm} — delete this routine's artifact, right before compiling the candidate that
      should produce it — is how a loop says the overwrite is intended, and turns "was it really
      candidate N's kernel?" into a checked fact for each candidate in turn.

    A test whose leg the backend cannot evaluate (a GPU intrinsic on a CPU backend) must not reach
    {!read} at all: gate the leg and report it with [Verdict.skipped]. An absent artifact is a
    failure here, deliberately. *)

open Base

(* The run's backend name, lowercased, as passed to [init]. [None] until then, so that a read
   preceding initialization is caught rather than silently answered from a stale directory. *)
let backend : string option ref = ref None

(* Contents digest of each artifact path already read, so that a second read of a path whose
   contents changed is reported instead of silently attributed to the wrong compile. *)
let read_digests : (string, string) Hashtbl.t = Hashtbl.create (module String)

let uninitialized where =
  Verdict.fail
    (where ^ ": Test_utils.Generated.init ~backend_name must be called before any compile, so that "
   ^ "artifacts left by earlier runs cannot be mistaken for this run's")

(** The source extension the run's backend emits: [.metal], [.cu], [.hip], or [.c] for the C
    backends. *)
let extension_of_backend backend_name =
  let has s = String.is_substring (String.lowercase backend_name) ~substring:s in
  if has "metal" then ".metal" else if has "cuda" then ".cu" else if has "hip" then ".hip" else ".c"

let extension () =
  match !backend with
  | Some b -> extension_of_backend b
  | None ->
      uninitialized "Generated.extension";
      ".c"

(** [path ?ext routine] is where the backend writes [routine]'s generated source. [?ext] overrides
    the backend-derived extension, for a test pinning one backend's artifact specifically. *)
let path ?ext routine =
  let ext = match ext with Some e -> e | None -> extension () in
  Utils.build_file (routine ^ ext)

(* Deletion IS the freshness guarantee here, so a delete that does not happen cannot be swallowed: a
   surviving file is one a later {!read} would accept as this run's, which is the whole failure this
   module exists to prevent. Absence is therefore verified rather than inferred from the call not
   raising — a Windows sharing violation, a read-only directory, or a file another process holds
   open all leave the path in place. Deleting something that was not there is not a failure, which
   is why the verdict is on [file_exists] afterwards rather than on the exception. *)
let remove_or_fail ~context p =
  let err =
    try
      Stdlib.Sys.remove p;
      None
    with Stdlib.Sys_error msg -> Some msg
  in
  if Stdlib.Sys.file_exists p then
    Verdict.fail
      (Printf.sprintf
         "%s: could not delete %s%s — a stale artifact left in place would be read as this run's"
         context p
         (match err with Some msg -> " (" ^ msg ^ ")" | None -> ""))

(** [init ~backend_name] empties this executable's [build_files/] subdirectory, so that every
    artifact found later was emitted by this run. Call it once at the top of the test, before any
    compile — [Verdict.fail]s if a read happens without it.

    [~backend_name] selects the source extension; tests already derive it as
    [String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")]. It is passed in
    rather than read here to keep the dependency at [arrayjit.utils]: this module resolves paths, it
    does not choose backends. *)
let init ~backend_name =
  backend := Some (String.lowercase backend_name);
  Hashtbl.clear read_digests;
  let dir = Utils.build_files_dir () in
  (* Whether the layout is flat is asked of the resolution itself — [artifacts_subdir] answering
     [None] — not inferred from the directory's name: [build_files_prefix=build_files] is a
     perfectly ordinary scoped prefix that resolves to [build_files/build_files], whose basename is
     indistinguishable from the flat root's. *)
  if Option.is_none (Utils.artifacts_subdir ()) then
    (* The flat legacy layout is shared by every concurrently running test, so emptying it would
       delete artifacts belonging to other processes. Fail loudly: freshness cannot be established
       here, and a test that silently gave it up would be back where this module started. *)
    Verdict.fail
      "Generated.init: build_files is flat (build_files_prefix=.), shared with concurrently \
       running tests; artifact freshness cannot be scoped to this process"
  else
    Array.iter (Stdlib.Sys.readdir dir) ~f:(fun entry ->
        let p = Stdlib.Filename.concat dir entry in
        let is_dir = try Stdlib.Sys.is_directory p with Stdlib.Sys_error _ -> true in
        if not is_dir then remove_or_fail ~context:"Generated.init" p)

(** [arm ?ext routine] deletes [routine]'s artifact, so that the next {!read} of it sees only what
    the *next* compile emits. Use it in a loop that compiles several candidates under one routine
    name: without it, the second read of a changed artifact is reported as an unattributed
    overwrite. *)
let arm ?ext routine =
  let p = path ?ext routine in
  Hashtbl.remove read_digests p;
  remove_or_fail ~context:("Generated.arm " ^ routine) p

(** [read ?ext routine] is the generated source [routine]'s compile emitted during this run.

    Fails the run, and answers [""], when the artifact is missing: the kernel was not emitted, the
    routine is named something else now, or the test forgot
    [Utils.settings.output_debug_files_in_build_directory <- true]. Answering rather than raising
    keeps the run's remaining checks reporting; the failure is already recorded, so the exit status
    holds whatever the caller concludes from the empty source. *)
let read ?ext routine =
  if Option.is_none !backend then (
    uninitialized ("Generated.read " ^ routine);
    "")
  else
    let p = path ?ext routine in
    if not (Stdlib.Sys.file_exists p) then (
      Verdict.fail
        (Printf.sprintf
           "no generated source for routine %s: %s was not emitted by this run (kernel folded \
            away, renamed, fissioned, or debug artifacts are off)"
           routine p);
      "")
    else
      let src = Stdio.In_channel.read_all p in
      let digest = Stdlib.Digest.string src in
      (match Hashtbl.find read_digests p with
      | Some previous when not (String.equal previous digest) ->
          Verdict.fail
            (Printf.sprintf
               "generated source for routine %s changed between reads: %s was overwritten by \
                another compile under the same name, so the earlier reading described a different \
                kernel (name the routines apart, or Generated.arm before each compile)"
               routine p)
      | Some _ | None -> ());
      Hashtbl.set read_digests ~key:p ~data:digest;
      src

(** [assert_emits ~routine ~contains claim] records [claim] as holding exactly when [routine]'s
    generated source contains [contains] — the common case, as an assertion rather than a [match]
    whose [None] arm has to be remembered. *)
let assert_emits ?ext ~routine ~contains claim =
  Verdict.p claim (String.is_substring (read ?ext routine) ~substring:contains)

(** [assert_omits ~routine ~contains claim] is {!assert_emits} for a substring that must NOT appear.
    Note that it still requires the artifact to exist: "the kernel does not contain [x]" is not a
    fact a missing kernel establishes. *)
let assert_omits ?ext ~routine ~contains claim =
  Verdict.p claim (not (String.is_substring (read ?ext routine) ~substring:contains))
