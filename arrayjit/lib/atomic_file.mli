(** Publishing a file so that no reader ever observes a half-written one (gh-ocannl-780).

    Every writer here stages its bytes into a uniquely named sibling of the target and then renames
    that sibling over it. A reader of the target therefore sees the complete previous content or the
    complete new content, never an intention in progress, and never a missing file — the target is
    never unlinked, truncated or opened for writing.

    The dance has three parts a hand-written copy tends to get only two of:

    - {b Uniqueness.} A fixed [<path>.tmp] is shared state: two concurrent writers to the same
      target write into one another's staging file and commit a mixture. The staging name here
      carries the writer's pid and a per-process counter, so no two attempts — across domains of one
      process or across processes — ever address the same artifact.
    - {b Failure cleanup.} A staging file is removed on every path out of {!publish} that is not a
      successful commit, so a failed write leaves neither a torn target nor an accumulating
      intention.
    - {b Crash-stale cleanup.} A writer killed between staging and commit cannot clean up after
      itself, and its staging file would otherwise sit in the directory forever. {!cleanup_stale}
      removes staging files older than an age no live attempt can plausibly reach;
      {!cleanup_stale_once} is the once-per-process-per-directory form the cache writers call.

    {2 Windows}

    The portable dance is not the POSIX one, and the differences are measured rather than inferred
    (gh-ocannl-588):

    - Replacing the target by [rename] is right, and truncating it in place is wrong, on Windows
      most of all: a live memory mapping of the target blocks neither the rename nor a delete
      (modern NTFS has POSIX delete semantics), but it does pin the file's SIZE, so reopening the
      path for writing — the obvious "just overwrite it" alternative — is the one operation Windows
      refuses outright ([ERROR_USER_MAPPED_FILE]). Nothing here ever opens the target.
    - The staging file is closed before it is renamed and before it is removed. The C runtime opens
      files without [FILE_SHARE_DELETE], so on Windows both operations fail while this process still
      holds the handle.
    - For the same sharing reason a rename can fail transiently while ANOTHER process has the target
      open for reading. {!publish} retries the rename a bounded number of times with a short backoff
      before propagating the failure; on POSIX the first attempt always succeeds and the retry costs
      nothing. *)

val staging_infix : string
(** The infix marking a staging file: [<target>.ocannl-stage.<pid>.<counter>]. Distinctive rather
    than generic, so a sweep can attribute leftovers and a [.gitignore] can name them. *)

val is_staging_file : string -> bool
(** Whether a file NAME (not necessarily a path) is one of this module's staging artifacts. The
    predicate tests to be checked against, rather than a second spelling of the naming scheme. *)

val ensure_dir : string -> unit
(** Creates the directory and its missing parents, tolerating concurrent creators. A no-op for
    ["."], ["/"] and the empty string, so a bare filename's [Filename.dirname] is safe to pass. *)

val publish : ?before_commit:(unit -> unit) -> path:string -> f:(string -> 'a) -> unit -> 'a
(** [publish ~path ~f ()] calls [f staging] with the path of a fresh staging file next to [path],
    then renames it over [path] and returns [f]'s result. [f] must leave no handle open on the
    staging file when it returns.

    [path]'s directory must exist — publishing is not directory management, and creating one here
    would turn a write to a mistyped path into a success nobody looks at. Call {!ensure_dir} first
    where the directory is the caller's to create.

    [?before_commit] runs after [f] and before the rename. It is the seam a caller uses to observe
    or to fail the window in which the payload is staged but not yet committed — the resource
    fault-injection points do exactly that.

    Any exception from [f], from [before_commit] or from the rename removes the staging file and is
    re-raised with its original backtrace. The target is left exactly as it was. *)

val write_all : ?before_commit:(unit -> unit) -> path:string -> data:string -> unit -> unit
(** {!publish} for a payload already in memory. Every argument is labeled and the call is closed by
    [()], as in {!publish} and {!with_channel}, so [?before_commit] can be passed in whichever
    position reads best at the call site. *)

val with_channel :
  ?before_commit:(unit -> unit) ->
  ?binary:bool ->
  path:string ->
  f:(Stdlib.out_channel -> 'a) ->
  unit ->
  'a
(** {!publish} for a payload written incrementally — a large checkpoint that is streamed rather than
    concatenated. The channel is opened in binary mode unless [~binary:false] is passed, and is
    closed before the commit whether [f] returns or raises. *)

val default_max_age_seconds : float
(** One hour: orders of magnitude above the milliseconds a staging window lasts, so a file this old
    belongs to a writer that died, not to one still running. *)

val cleanup_stale : ?max_age_seconds:float -> string -> unit
(** [cleanup_stale dir] removes the staging files in [dir] whose modification time is older than
    [?max_age_seconds] (default {!default_max_age_seconds}). Everything is best-effort: an
    unreadable directory, a file that vanished under the sweep, or a file another user owns is
    skipped rather than reported. Files that are not staging artifacts are never touched. *)

val cleanup_stale_once : ?max_age_seconds:float -> string -> unit
(** {!cleanup_stale} at most once per directory per process, for callers that would otherwise sweep
    on every write. Thread-safe across domains. *)
