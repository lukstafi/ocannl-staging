(* The provenance guarantees of [Test_utils.Generated] (gh-ocannl-655), pinned executably.

   That module is what ~30 tests' structural assertions now rest on, so its preconditions are
   load-bearing in a way that is easy to break silently: every one of them fires only when something
   has ALREADY gone wrong, which is precisely the kind of code that rots unobserved. Two review
   rounds on the introducing PR found three such defects by inspection alone — a swallowed deletion
   failure, a directory-scoping test that misread a valid configuration, and a sweep that was not
   this process's to perform. This probe exists so that the next one fails a test instead.

   Each mode is one guarantee, and the dune rules assert the EXIT STATUS rather than the output.
   [Verdict] failures exit 1 from a shared teardown, so "this call must be refused" is exactly a
   nonzero exit, and [with-accepted-exit-codes] states which way each mode must go. The modes that
   must be refused therefore make no claim of their own — the refusal IS the assertion, and a
   designed-false [Verdict.p] would only add a promotable line saying so. The modes that must
   succeed claim normally, so that a module which refused everything could not pass them.

   The artifacts are written by hand rather than compiled: what is under test is how this module
   decides whether a file on disk belongs to this run, and a real backend would only make the setup
   slower and the failure modes harder to stage. *)

open Base
module Generated = Test_utils.Generated

let backend_name = "cc"
let dir () = Utils.build_files_dir ()
let path routine = Stdlib.Filename.concat (dir ()) (routine ^ ".c")

(* The symlink modes below must NOT call [dir ()] first: it creates the directory whose place the
   link has to take. They resolve the same path the way Generated does instead. *)
let artifacts_root = "build_files"

let scoped_dir () =
  Stdlib.Filename.concat artifacts_root
    (Stdlib.Filename.remove_extension (Stdlib.Filename.basename Stdlib.Sys.executable_name))

let link_target () = Stdlib.Filename.concat artifacts_root "gp_symlink_target"
let precious () = Stdlib.Filename.concat (link_target ()) "gp_precious.c"
let ignore_unix f x = try f x with Unix.Unix_error _ -> ()

(* Stands in for the backend emitting a kernel. *)
let emit routine contents = Stdio.Out_channel.write_all (path routine) ~data:contents

let () =
  let mode =
    match Array.to_list Stdlib.Sys.argv with
    | _ :: m :: _ when not (String.is_prefix m ~prefix:"--") -> m
    | _ -> failwith "generated_provenance: expected a mode argument"
  in
  match mode with
  (* === Must succeed === *)

  (* The happy path. Pins that the refusals below are not vacuous: a module that refused every read
     would satisfy all of them. *)
  | "fresh" ->
      Generated.init ~backend_name;
      emit "gp_fresh" "kernel body: fresh\n";
      Verdict.p "a freshly emitted artifact reads back"
        (String.is_substring (Generated.read "gp_fresh") ~substring:"fresh")
  (* Arming is what turns per-candidate attribution into a checked fact: the same two-kernel
     sequence that is refused unarmed below must go through when each compile is armed. *)
  | "armed_overwrite" ->
      Generated.init ~backend_name;
      emit "gp_twice" "kernel body: candidate 1\n";
      ignore (Generated.read "gp_twice" : string);
      Generated.arm "gp_twice";
      emit "gp_twice" "kernel body: candidate 2\n";
      Verdict.p "an armed second reading is attributed to its own compile"
        (String.is_substring (Generated.read "gp_twice") ~substring:"candidate 2")
  (* An explicit build_files_prefix is not this executable's to empty — two tests can be given the
     same one — but it is not fatal either: the per-routine route stays open. Both halves are pinned
     here, and the surviving decoy is what shows the sweep did NOT run. *)
  (* The other half: an explicitly prefixed directory must keep working for the ordinary caller, who
     calls init once and then reads — no arming — and a concurrent test's artifact in the same
     directory must survive, since nothing may be swept here. *)
  | "shared_prefix_fresh" ->
      let decoy = path "gp_other_tests_kernel" in
      Stdio.Out_channel.write_all decoy ~data:"another concurrent test's kernel\n";
      Generated.init ~backend_name;
      Verdict.p "an explicitly prefixed directory is left alone" (Stdlib.Sys.file_exists decoy);
      emit "gp_shared" "kernel body: mine\n";
      Verdict.p "an unarmed read of a freshly emitted artifact succeeds under an explicit prefix"
        (String.is_substring (Generated.read "gp_shared") ~substring:"mine");
      Generated.arm "gp_shared";
      emit "gp_shared" "kernel body: armed\n";
      Verdict.p "an armed read under an explicit prefix succeeds"
        (String.is_substring (Generated.read "gp_shared") ~substring:"armed")
  (* === Must be refused (the refusal is the assertion; see the header) === *)
  (* The issue's own failure: an artifact left by an EARLIER run, which the readers this replaced
     accepted because it happened to exist. Written before init, so init's sweep must remove it. *)
  | "stale" ->
      let stale = path "gp_stale" in
      Stdio.Out_channel.write_all stale ~data:"kernel body: from a previous run\n";
      Generated.init ~backend_name;
      Stdio.printf "read after sweep: %S\n" (Generated.read "gp_stale")
  (* A kernel that stopped being emitted at all: folded to a constant, renamed, fissioned. *)
  | "missing" ->
      Generated.init ~backend_name;
      Stdio.printf "read of a never-emitted routine: %S\n" (Generated.read "gp_never_emitted")
  (* Two kernels read under one routine name, unarmed: the same-process overwrite. *)
  | "overwrite" ->
      Generated.init ~backend_name;
      emit "gp_twice" "kernel body: candidate 1\n";
      ignore (Generated.read "gp_twice" : string);
      emit "gp_twice" "kernel body: candidate 2\n";
      ignore (Generated.read "gp_twice" : string);
      Stdio.printf "second unarmed reading completed\n"
  (* Armed, but the compile emitted nothing — the case [arm] exists for. The previous candidate's
     kernel is gone, so nothing can be credited to this one. *)
  | "armed_no_emission" ->
      Generated.init ~backend_name;
      emit "gp_twice" "kernel body: candidate 1\n";
      ignore (Generated.read "gp_twice" : string);
      Generated.arm "gp_twice";
      Stdio.printf "read after an armed compile emitted nothing: %S\n" (Generated.read "gp_twice")
  (* The flat layout is shared with every concurrently running test, so init must refuse it outright
     rather than empty it. *)
  | "flat" ->
      Generated.init ~backend_name;
      Stdio.printf "init returned under the flat layout\n"
  (* Under an explicit prefix nothing is swept, so provenance comes from the timestamp floor: an
     artifact predating the run is exactly the stale file the old readers accepted, and must be
     refused even though it exists. *)
  | "shared_prefix_stale" ->
      emit "gp_shared" "kernel body: from a previous run\n";
      Generated.init ~backend_name;
      Stdio.printf "stale read under an explicit prefix: %S\n" (Generated.read "gp_shared")
  (* Before init the directory may hold anything, so no read may be answered. *)
  | "uninitialized" -> Stdio.printf "read before init: %S\n" (Generated.read "gp_anything")
  (* === The symlinked artifact directory ===

     [Sys.is_directory] follows symlinks, so a symlinked build_files/<exe>/ reads as an ordinary
     directory and a sweep would unlink real files in the link's TARGET — outside the artifact tree
     entirely. Two things make this mode's shape what it is.

     Refusal now ABORTS the process, so [init] cannot be called in-process and observed; it is run
     in a child, whose exit status carries the refusal. And symlink creation is a privilege on
     Windows (Developer Mode, or SeCreateSymbolicLink) that the CI matrix does not necessarily have,
     so the staging is attempted rather than assumed: where links cannot be made the leg is reported
     as skipped instead of failing for the host's permissions. Both halves are then checked here —
     the child refused, AND the planted file in the link target is still there — so this stays a
     single mode that must SUCCEED. *)
  | "symlink" -> (
      ignore_unix (fun d -> Unix.mkdir d 0o777) artifacts_root;
      ignore_unix (fun d -> Unix.mkdir d 0o777) (link_target ());
      Stdio.Out_channel.write_all (precious ()) ~data:"a file that is not this test's to delete\n";
      let scoped = scoped_dir () in
      (* Whatever an earlier mode left here has to go, so the link can take its place. *)
      ignore_unix Unix.unlink scoped;
      (try
         Array.iter (Stdlib.Sys.readdir scoped) ~f:(fun e ->
             ignore_unix Unix.unlink (Stdlib.Filename.concat scoped e));
         ignore_unix Unix.rmdir scoped
       with Stdlib.Sys_error _ -> ());
      (* Relative, so it resolves inside build_files/ wherever the tree is checked out.
         [~to_dir:true] is ignored on Unix and load-bearing on Windows, where a link created without
         it is a FILE symlink that would not stand in for a directory at all. *)
      match Unix.symlink ~to_dir:true "gp_symlink_target" scoped with
      | exception Unix.Unix_error (err, _, _) ->
          Stdio.eprintf "symbolic links unavailable here (%s)\n" (Unix.error_message err);
          Verdict.skipped ~backend:backend_name
            "a symlinked artifact directory is refused rather than followed into"
      | () ->
          let exe = Stdlib.Sys.executable_name in
          let pid =
            Unix.create_process exe [| exe; "symlink_child" |] Unix.stdin Unix.stdout Unix.stderr
          in
          let refused = match Unix.waitpid [] pid with _, Unix.WEXITED 1 -> true | _ -> false in
          let survived = Stdlib.Sys.file_exists (precious ()) in
          (* Restore a real directory for whatever runs next. *)
          ignore_unix Unix.unlink scoped;
          Verdict.p "a symlinked artifact directory is refused rather than followed into"
            (refused && survived))
  (* Run by the mode above, in a child process, because refusing a directory aborts.

     The arm attempt after init is the point: an exit status alone cannot tell a refusal that
     STOPPED the run from one that merely recorded a failure and returned, and the difference is
     whether the deletions that follow go through the link. Under a terminating refusal this line is
     never reached; under a non-terminating one it unlinks the planted file in the link target, and
     the parent's survival check fails. That regression shipped once — a lost edit left this branch
     calling Verdict.fail — so the probe now discriminates it rather than trusting the exit code. *)
  | "symlink_child" ->
      Generated.init ~backend_name;
      Generated.arm "gp_precious";
      Stdio.printf "init returned on a symlinked artifact directory\n"
  | m -> failwith ("generated_provenance: unknown mode " ^ m)
