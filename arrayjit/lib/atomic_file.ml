open Base

let staging_infix = ".ocannl-stage."

(* A staging name has to satisfy three things at once, and the first two pull against each other:
   it must be recognizable (the sweep DELETES what it recognizes), it must stay inside the
   filesystem's per-component limit however long the target's name is, and it must be unique among
   every writer that can reach this directory. So the target's contribution is a bounded STEM
   derived from its basename, and everything after the infix is the writer's identity. *)
let max_component_bytes = 255

(* Everything after the infix is FIXED-WIDTH lowercase hex: the pid, the counter, the nonce. Named
   here because three separate things have to describe the same set of names -- the generator, the
   recognizer the destructive sweep consults, and the `.gitignore` rule, which is a glob and so can
   only be exact over fixed widths. A variable-width field forces the glob to spell `[0-9]*`, which
   git reads as "a digit then anything", and an ordinary file with a non-numeric field is hidden
   from its author's own `git status` (Codex P2, round 6). A recognizer looser than the generator is
   the same defect pointed the other way: a licence to delete somebody else's file. *)
let field_width = 8
let nonce_width = 16

(* What the target may contribute. The rest is the infix, a pid, a counter and a nonce; 64 bytes
   covers a 64-bit pid printed in full. *)
let stem_budget = max_component_bytes - 64
let short_digest s = String.prefix (Stdlib.Digest.to_hex (Stdlib.Digest.string s)) 8

(* Truncation is by BYTES, because the limit is in bytes — but a cut in the middle of a multibyte
   character produces a name that is not valid UTF-8, which Windows refuses to open even though the
   target's own name was fine (Codex P2, round 4). So back off to a byte that does not continue a
   sequence. A name that was not valid UTF-8 to begin with is not made worse: the cut lands where it
   would have anyway, and the digest keeps the stem unique regardless. *)
let utf8_prefix s at =
  let continues i = Char.to_int s.[i] land 0xC0 = 0x80 in
  let rec back k = if k <= 0 then 0 else if continues k then back (k - 1) else k in
  String.prefix s (back (Int.min at (String.length s)))

(* Bounded, and a function of the basename alone — which is what lets the recognizer below rebuild
   it from the target instead of storing it. A long checkpoint name used to fit only because the
   old suffix was four characters (Codex P2, round 2); now it is truncated and disambiguated by a
   digest of the whole name, so two long names that share a prefix still get distinct stems. *)
let staging_stem basename =
  if String.length basename <= stem_budget then basename
  else
    (* The digest is taken over the LOWERCASED name so that the stem stays caseless too: on a
       case-insensitive volume [Model.bin] and [model.bin] are one file, and a case-sensitive digest
       would give their stems different tails that no caseless comparison could reconcile. *)
    utf8_prefix basename (stem_budget - 9) ^ "~" ^ short_digest (String.lowercase basename)

(* [<stem>] ^ [staging_infix] ^ [<pid>] ^ "." ^ [<counter>] ^ "." ^ [<nonce>]. Recognition is the
   whole shape rather than a search for the infix: `report.ocannl-stage.backup` is somebody's file,
   and answering "staging" on it would make the sweep destructive over names it was never promised
   (Codex P2, round 1). Returns the stem exactly when [name] is a staging name — which is also how
   a caller asks about ONE published file rather than about a whole directory. *)
let staging_stem_of name =
  (* Every component is recognized at its GENERATED width and alphabet, not as "some hexadecimal": a
     sweep that accepts `report.ocannl-stage.1.2.a` deletes a file this module never wrote. *)
  let hex_field width part =
    String.length part = width
    && String.for_all part ~f:(fun c -> Char.is_digit c || Char.between c ~low:'a' ~high:'f')
  in
  match List.last (String.substr_index_all name ~may_overlap:false ~pattern:staging_infix) with
  | None -> None
  | Some at ->
      let stem = String.prefix name at in
      let stamp = String.drop_prefix name (at + String.length staging_infix) in
      (* Bounded as well as non-empty. [staging_stem] never emits more than [stem_budget] bytes, so a
         longer stem is a name this module cannot have written — and the sweep deletes by this
         predicate (Codex P2, round 8). The bound is the whole of what is checkable: a stem at or
         under the budget is emitted verbatim, and a truncated one is a prefix plus [~] and a digest,
         which a short basename could also happen to spell. *)
      if String.is_empty stem || String.length stem > stem_budget then None
      else (
        match String.split stamp ~on:'.' with
        | [ pid; counter; nonce ]
          when hex_field field_width pid && hex_field field_width counter
               && hex_field nonce_width nonce ->
            Some stem
        | _ -> None)

let is_staging_file name = Option.is_some (staging_stem_of name)

let is_staging_file_for ~path name =
  (* Caseless, unconditionally. On Windows and on a default macOS volume `Model.bin` and `model.bin`
     ARE the same target, so a case-sensitive comparison would leave a model-sized artifact
     unreclaimed there (Codex P2, round 2). Where paths really are case-sensitive the only effect is
     that one target's save also reclaims a case-twin's staging file — which is an OCANNL staging
     file, abandoned for over an hour, and something the directory-wide sweep would remove anyway. *)
  Option.exists (staging_stem_of name)
    ~f:(String.Caseless.equal (staging_stem (Stdlib.Filename.basename path)))

let rec ensure_dir dir =
  if String.is_empty dir || String.equal dir "." || String.equal dir "/" then ()
  else if Stdlib.Sys.file_exists dir then ()
  else (
    ensure_dir (Stdlib.Filename.dirname dir);
    (* Concurrent creators race benignly. *)
    try Stdlib.Sys.mkdir dir 0o777 with Stdlib.Sys_error _ -> ())

(* The counter distinguishes the staging files of two writers inside ONE process; the pid
   distinguishes processes on one host. Neither is enough on a filesystem shared between hosts or
   pid namespaces, where two writers can hold the same pid and both counters start at zero (Codex
   P1, round 2) — so a nonce joins them, and, because a nonce only makes a collision unlikely,
   the file is CREATED EXCLUSIVELY: a name already taken is retried rather than opened. *)
let next_staging_id : int Atomic.t = Atomic.make 0

(* One 64-bit draw, so every value the recognizer accepts is one the generator can produce. Two
   28-bit draws rendered as eight hex digits each left characters 1 and 9 of every nonce at zero,
   which made the recognizer strictly looser than generation — and it is the recognizer that decides
   what the sweep deletes (Codex P2, round 6). *)
let fresh_nonce () =
  let state = Stdlib.Random.State.make_self_init () in
  let nonce = Printf.sprintf "%016Lx" (Stdlib.Random.State.bits64 state) in
  assert (String.length nonce = nonce_width);
  nonce

let staging_path path =
  (* Masked to the field's width rather than trusted to fit: a pid or a counter that overflowed it
     would render wider and produce a name the recognizer -- and the ignore glob -- would not accept.
     Wrapping costs nothing, since it is the nonce and the exclusive creation that carry uniqueness.
  *)
  let field value = Printf.sprintf "%0*x" field_width (value land 0xFFFFFFFF) in
  let name =
    Printf.sprintf "%s%s%s.%s.%s"
      (staging_stem (Stdlib.Filename.basename path))
      staging_infix
      (field (Unix.getpid ()))
      (field (Atomic.fetch_and_add next_staging_id 1))
      (fresh_nonce ())
  in
  Stdlib.Filename.concat (Stdlib.Filename.dirname path) name

let staging_attempts = 8

(* Exclusive creation is what turns "no two writers pick the same name" from a probability into a
   guarantee: a collision — same host, same pid namespace, or an old artifact that happens to
   collide — fails the open rather than sharing the file. *)
let open_staging path ~binary =
  let rec attempt n =
    let staging = staging_path path in
    match Unix.openfile staging [ Unix.O_WRONLY; Unix.O_CREAT; Unix.O_EXCL ] 0o666 with
    | fd ->
        let oc = Unix.out_channel_of_descr fd in
        Stdlib.set_binary_mode_out oc binary;
        (staging, oc)
    | exception Unix.Unix_error (Unix.EEXIST, _, _) when n < staging_attempts -> attempt (n + 1)
    (* A filesystem refusal reaches the caller as [Sys_error], whichever operation refused. Opening
       through [Unix] is an implementation choice — it is what makes the creation exclusive — and it
       must not change what a caller catches: [Schedule_cache.store] treats a refusal as a future
       cache miss, and an unconverted [Unix_error] from this open would escape that handler and
       abort a tuning run (Codex P2, round 3). Only the filesystem's own errors are converted;
       whatever [f] or [before_commit] raise passes through untouched. *)
    | exception Unix.Unix_error (error, fn, arg) ->
        raise
          (Stdlib.Sys_error
             (Printf.sprintf "%s: %s %s (%s)" staging fn arg (Unix.error_message error)))
  in
  attempt 1

let remove_quietly path =
  if Stdlib.Sys.file_exists path then try Stdlib.Sys.remove path with _ -> ()

(* The commit is one [rename], which is atomic on POSIX and on NTFS. It can still FAIL on Windows:
   the C runtime opens files without [FILE_SHARE_DELETE], so while another handle to the target is
   open — a concurrent reader of the same cache entry — replacing it is refused with a sharing
   violation. That refusal is transient by nature (it lasts as long as the other reader's handle),
   so retry a bounded number of times before reporting it. Nothing here retries on POSIX: the first
   attempt succeeds, and the loop costs one call. *)
let commit_attempts = 8
let commit_backoff_seconds = 0.002

let commit ~staging ~path =
  let rec attempt n =
    match Stdlib.Sys.rename staging path with
    | () -> ()
    | exception (Stdlib.Sys_error _ as exn) ->
        if n >= commit_attempts then raise exn
        else (
          Unix.sleepf (commit_backoff_seconds *. Float.of_int n);
          attempt (n + 1))
  in
  attempt 1

(* No [ensure_dir] here: publishing is not directory management, and creating one silently would
   turn a save to a mistyped path into a save that succeeds somewhere nobody looks. A caller whose
   directory may be missing calls [ensure_dir] first, as [Schedule_cache.store] does. *)
let with_channel ?before_commit ?(binary = true) ~path ~f () =
  let staging, oc = open_staging path ~binary in
  match
    let result = f oc in
    (* Closed BEFORE the commit, not after: a still-open staging file cannot be renamed on Windows,
       and an unflushed buffer would commit a truncated payload on every platform. *)
    Stdlib.close_out oc;
    Option.iter before_commit ~f:(fun hook -> hook ());
    commit ~staging ~path;
    result
  with
  | result -> result
  | exception exn ->
      let backtrace = Stdlib.Printexc.get_raw_backtrace () in
      (* Closed before it is removed, for the same reason: Windows refuses to delete a file this
         process still holds open. Closing twice is why this is the [_noerr] form. *)
      Stdlib.close_out_noerr oc;
      remove_quietly staging;
      Stdlib.Printexc.raise_with_backtrace exn backtrace

let write_all ?before_commit ?binary ~path ~data () =
  with_channel ?before_commit ?binary ~path () ~f:(fun oc -> Stdlib.output_string oc data)

let default_max_age_seconds = 3600.

(* One sweep, two scopes. Which names it considers is the caller's only choice: everything else --
   the age gate, the best-effort handling, the refusal to look at anything that is not a staging
   name -- is the same question however wide the scope. *)
let sweep ~max_age_seconds ~dir ~selects =
  match Stdlib.Sys.readdir dir with
  | exception _ -> ()
  | entries ->
      let now = Unix.time () in
      let inactive_since path =
        match (Unix.stat path).Unix.st_mtime with
        | exception _ -> None
        | mtime -> Some (now -. mtime)
      in
      Array.iter entries ~f:(fun name ->
          if selects name then
            let path = Stdlib.Filename.concat dir name in
            match inactive_since path with
            | Some age when Float.(age > max_age_seconds) ->
                (* Read the clock again at the moment of removal. A writer's own writes advance the
                   mtime, so this threshold is on INACTIVITY, not on age since creation — and a
                   publication that resumed between the scan above and this line has just proved
                   itself live. *)
                if
                  Option.value_map (inactive_since path) ~default:false ~f:(fun age ->
                      Float.(age > max_age_seconds))
                then remove_quietly path
            | _ -> ())

let cleanup_stale ?(max_age_seconds = default_max_age_seconds) dir =
  sweep ~max_age_seconds ~dir ~selects:is_staging_file

let cleanup_stale_for ?(max_age_seconds = default_max_age_seconds) path =
  sweep ~max_age_seconds
    ~dir:(Stdlib.Filename.dirname path)
    ~selects:(is_staging_file_for ~path)

let swept : (string, unit) Hashtbl.t = Hashtbl.create (module String)
let swept_mutex = Stdlib.Mutex.create ()

let cleanup_stale_once ?max_age_seconds dir =
  (* A directory that does not exist yet has nothing to sweep and must not be RECORDED as swept: the
     cache's [lookup] reaches here before the first [store] creates the directory, and marking it
     then would spend the process's one sweep on nothing. *)
  if not (Stdlib.Sys.file_exists dir) then ()
  else
    let first =
      Stdlib.Mutex.lock swept_mutex;
      let first = not (Hashtbl.mem swept dir) in
      if first then Hashtbl.set swept ~key:dir ~data:();
      Stdlib.Mutex.unlock swept_mutex;
      first
    in
    if first then cleanup_stale ?max_age_seconds dir
