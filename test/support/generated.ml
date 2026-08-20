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
      freshness and no clock is involved. What makes that safe is that the DEFAULT subdirectory is
      derived from the executable's own name and dune runs one process per executable, so nothing
      else writes there. The other two resolutions are not this process's to empty and get their own
      treatment: the flat legacy layout ([build_files_prefix=.]) is rejected outright, and an
      explicit [build_files_prefix] — which two executables can be given alike — deletes nothing at
      all, recording instead how each file in the directory stood when the run began, so that a read
      can ask whether that particular artifact has changed since.
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

(* Artifact paths deleted by {!arm} since {!init}, i.e. those whose freshness has been established
   one path at a time rather than by emptying the directory. Only consulted where the directory is
   not this process's own. *)
let armed : (string, unit) Hashtbl.t = Hashtbl.create (module String)

(* Whether wholesale cleanup of the artifact directory is this process's to do.

   Only the DEFAULT, executable-derived subdirectory is inherently process-private: dune runs one
   process per executable, so nothing else writes there. Every other resolution can be shared — the
   flat layout by definition, and an explicit [build_files_prefix] because two executables can be
   given the same one — and emptying a shared directory would delete a concurrently running test's
   kernel between its compile and its read. So the three cases get three behaviours rather than one:
   sweep, refuse, or establish freshness per routine through {!arm}. *)
type scoping = Private_to_this_exe | Shared_prefix of string | Flat

(* Read from the CONFIGURED VALUE, not from what the resolved directory happens to look like. The
   property is "nobody configured this directory, so it is mine by construction" — and no comparison
   of the result can establish that: [build_files_prefix=<this exe's own basename>] resolves exactly
   where the default would, while remaining a prefix a second executable can be given too. The empty
   string is what [Utils.artifacts_subdir] itself treats as "unset". *)
let scoping () =
  match Utils.get_global_arg ~default:"" ~arg_name:"build_files_prefix" with
  | "" -> Private_to_this_exe
  | "." -> Flat
  | prefix -> Shared_prefix prefix

(* [Flat] never reaches a read (init aborts), so the remaining question at read time is whether the
   directory was swept — in which case existence is provenance — or whether the artifact has to be
   newer than the run. *)
let sweeps_directory = ref true

(* When the directory was not swept: what each file in it looked like when the run began, as (mtime,
   contents digest). An artifact is then judged against ITSELF as it stood at that moment, rather
   than against any clock reading or marker.

   This replaces a marker file whose mtime served as a floor, and with it three hazards that were
   all one hazard: a stale artifact could TIE with the marker's timestamp on a coarse filesystem, a
   marker that could not be rewritten silently left the previous run's floor standing, and the
   marker path itself could be a symlink whose target got truncated. None of them has anywhere to
   bite here — there is no marker to write, to fail to write, or to follow — and a tie stops being
   ambiguous: a file whose mtime AND contents are both unchanged since the run began is exactly a
   file this run did not write. *)
let snapshot : (string, float * string) Hashtbl.t = Hashtbl.create (module String)

(* Whether [p] has been written since the snapshot: absent from it (created during the run), or
   changed in contents, or carrying a strictly newer mtime (an identical kernel re-emitted). *)
let written_since_snapshot p ~src =
  match Hashtbl.find snapshot p with
  | None -> true
  | Some (mtime0, digest0) -> (
      (not (String.equal (Stdlib.Digest.string src) digest0))
      ||
      match Unix.stat p with
      | { Unix.st_mtime; _ } -> Float.(st_mtime > mtime0)
      | exception Unix.Unix_error _ -> false)

(* A refusal that returns is not a refusal. [Verdict.fail] only increments a counter, so a caller
   that reports an unusable artifact directory and carries on hands the rest of the test exactly the
   directory it just refused — and {!arm}, which deletes, would then run against it. The exit status
   arriving eventually at teardown does not help: by then the deletions have happened. So refusing
   the DIRECTORY aborts the process, at the point of refusal, before any compile or cleanup can go
   through it. Exit 1 rather than an exception, matching what [Verdict]'s own teardown would
   produce: [Stdlib.exit] runs that teardown on the way out, so the failure is still reported
   normally. *)
let abort msg =
  Verdict.fail msg;
  Stdio.Out_channel.flush Stdio.stdout;
  Stdlib.exit 1

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

(* [Stdlib.Sys.is_directory] follows symlinks, so a symlinked [build_files/] or [build_files/<exe>/]
   is accepted as an ordinary directory — and a sweep would then [readdir] the LINK TARGET and
   unlink real files outside the artifact tree. [arrayjit/lib/utils.ml]'s startup cleanup uses
   [Unix.lstat] for exactly this reason; the same question has to be asked here, of the destructive
   operation this module performs. Unlinking an individual symlinked ENTRY is not the hazard (that
   removes the link, not its target, which is what the startup cleanup relies on too) — the
   containing directory is. *)

(** [init ~backend_name] empties this executable's [build_files/] subdirectory, so that every
    artifact found later was emitted by this run. Call it once at the top of the test, before any
    compile — [Verdict.fail]s if a read happens without it.

    [~backend_name] selects the source extension; tests already derive it as
    [String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")]. It is passed in
    rather than read here to keep the dependency at [arrayjit.utils]: this module resolves paths, it
    does not choose backends. *)
let is_symlink p =
  match Unix.lstat p with
  | { Unix.st_kind = Unix.S_LNK; _ } -> true
  | _ -> false
  | exception Unix.Unix_error _ -> false

let init ~backend_name =
  backend := Some (String.lowercase backend_name);
  Hashtbl.clear read_digests;
  Hashtbl.clear armed;
  let dir = Utils.build_files_dir () in
  if is_symlink dir || is_symlink (Stdlib.Filename.dirname dir) then
    (* Refused rather than skipped, and aborted rather than merely recorded: following the link
       would delete files that are not this test's — strictly worse than any stale artifact — and a
       refusal the run continues past would do exactly that at the next {!arm}. *)
    abort
      (Printf.sprintf
         "Generated.init: %s (or its parent) is a symbolic link; refusing to use it, since \
          following it would delete files outside the artifact tree"
         dir)
  else
    match scoping () with
    | Flat ->
        (* Shared with every concurrently running test by definition, so there is no cleanup this
           process may perform and no per-routine deletion that would be safe either: another test
           can be writing the very name this one is about to read. Nothing this module offers works
           here, so the run stops rather than going on to delete and read another test's
           artifacts. *)
        abort
          "Generated.init: build_files is flat (build_files_prefix=.), shared with concurrently \
           running tests; artifact freshness cannot be established here"
    | Private_to_this_exe ->
        (* Derived from this executable's own name, so nothing else writes here: emptying it makes
           existence mean "written by this run" for every routine at once. *)
        sweeps_directory := true;
        Array.iter (Stdlib.Sys.readdir dir) ~f:(fun entry ->
            let p = Stdlib.Filename.concat dir entry in
            let is_dir = try Stdlib.Sys.is_directory p with Stdlib.Sys_error _ -> true in
            if not is_dir then remove_or_fail ~context:"Generated.init" p)
    | Shared_prefix _ ->
        (* An explicit prefix is not this executable's to empty — two tests can be configured with
           the same one, and a wholesale sweep would delete a concurrent test's kernel between its
           compile and its read. So freshness is established WITHOUT deleting anything: a marker
           read accepts only an artifact that has changed since. That is weaker than the sweep (it
           cannot tell this run's kernel from a concurrent test's under the same name, which is the
           same-name hazard that exists anyway) but it is the guarantee that matters — an artifact
           outliving the run that produced it is what this module exists to catch — and it costs no
           deletion at all.

           Nothing is written to establish it either: the directory is recorded as it stands, and a
           read later asks whether that particular file has changed since. *)
        sweeps_directory := false;
        Hashtbl.clear snapshot;
        Array.iter (Stdlib.Sys.readdir dir) ~f:(fun entry ->
            let p = Stdlib.Filename.concat dir entry in
            (* [lstat], so a symlinked entry is recorded as what it is rather than followed. *)
            match Unix.lstat p with
            | { Unix.st_kind = Unix.S_REG; st_mtime; _ } -> (
                match Stdlib.Digest.file p with
                | digest -> Hashtbl.set snapshot ~key:p ~data:(st_mtime, digest)
                | exception Stdlib.Sys_error _ -> ())
            | _ -> ()
            | exception Unix.Unix_error _ -> ())

(** [arm ?ext routine] deletes [routine]'s artifact, so that the next {!read} of it sees only what
    the *next* compile emits. Use it in a loop that compiles several candidates under one routine
    name: without it, the second read of a changed artifact is reported as an unattributed
    overwrite. *)
let arm ?ext routine =
  let p = path ?ext routine in
  Hashtbl.remove read_digests p;
  Hashtbl.set armed ~key:p ~data:();
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
      if
        (not !sweeps_directory)
        && (not (Hashtbl.mem armed p))
        && not (written_since_snapshot p ~src)
      then (
        (* Nothing was swept here, so existence alone does not establish provenance: this file is
           byte-for-byte what stood in the directory before the run started, and no later write
           touched it — exactly the stale artifact the old readers accepted. An ARMED path is
           exempt: it was deleted outright, so its existence is provenance in its own right. *)
        Verdict.fail
          (Printf.sprintf
             "stale generated source for routine %s: %s is unchanged from before this run began \
              (build_files_prefix is set explicitly, so the directory is not this executable's to \
              empty; arm the routine if its kernel is legitimately identical)"
             routine p);
        "")
      else (
        (match Hashtbl.find read_digests p with
        | Some previous when not (String.equal previous digest) ->
            Verdict.fail
              (Printf.sprintf
                 "generated source for routine %s changed between reads: %s was overwritten by \
                  another compile under the same name, so the earlier reading described a \
                  different kernel (name the routines apart, or Generated.arm before each compile)"
                 routine p)
        | Some _ | None -> ());
        Hashtbl.set read_digests ~key:p ~data:digest;
        src)

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
