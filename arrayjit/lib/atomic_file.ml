open Base

let staging_infix = ".ocannl-stage."
let is_staging_file name = String.is_substring name ~substring:staging_infix

let rec ensure_dir dir =
  if String.is_empty dir || String.equal dir "." || String.equal dir "/" then ()
  else if Stdlib.Sys.file_exists dir then ()
  else (
    ensure_dir (Stdlib.Filename.dirname dir);
    (* Concurrent creators race benignly. *)
    try Stdlib.Sys.mkdir dir 0o777 with Stdlib.Sys_error _ -> ())

(* Distinguishes the staging files of two writers inside ONE process; the pid distinguishes them
   across processes. Both halves are needed: the autotuner writes cache entries from several domains
   of one process, and several tuning processes share a cache directory. *)
let next_staging_id : int Atomic.t = Atomic.make 0

let staging_path path =
  Printf.sprintf "%s%s%d.%d" path staging_infix (Unix.getpid ())
    (Atomic.fetch_and_add next_staging_id 1)

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

let publish ?before_commit ~path ~f () =
  ensure_dir (Stdlib.Filename.dirname path);
  let staging = staging_path path in
  match
    let result = f staging in
    Option.iter before_commit ~f:(fun hook -> hook ());
    commit ~staging ~path;
    result
  with
  | result -> result
  | exception exn ->
      let backtrace = Stdlib.Printexc.get_raw_backtrace () in
      remove_quietly staging;
      Stdlib.Printexc.raise_with_backtrace exn backtrace

let write_all ?before_commit ~path ~data () =
  publish ?before_commit ~path () ~f:(fun staging -> Stdio.Out_channel.write_all staging ~data)

let with_channel ?before_commit ?(binary = true) ~path ~f () =
  publish ?before_commit ~path () ~f:(fun staging ->
      let oc = if binary then Stdlib.open_out_bin staging else Stdlib.open_out staging in
      match f oc with
      | result ->
          (* Closed BEFORE the commit, not after: a still-open staging file cannot be renamed on
             Windows, and an unflushed buffer would commit a truncated payload on every platform. *)
          Stdlib.close_out oc;
          result
      | exception exn ->
          let backtrace = Stdlib.Printexc.get_raw_backtrace () in
          (* Likewise before [publish]'s handler removes it: Windows refuses to delete a file this
             process still holds open. *)
          Stdlib.close_out_noerr oc;
          Stdlib.Printexc.raise_with_backtrace exn backtrace)

let default_max_age_seconds = 3600.

let cleanup_stale ?(max_age_seconds = default_max_age_seconds) dir =
  match Stdlib.Sys.readdir dir with
  | exception _ -> ()
  | entries ->
      let now = Unix.time () in
      Array.iter entries ~f:(fun name ->
          if is_staging_file name then
            let path = Stdlib.Filename.concat dir name in
            match (Unix.stat path).Unix.st_mtime with
            | exception _ -> ()
            | mtime -> if Float.(now -. mtime > max_age_seconds) then remove_quietly path)

let swept : (string, unit) Hashtbl.t = Hashtbl.create (module String)
let swept_mutex = Stdlib.Mutex.create ()

let cleanup_stale_once ?max_age_seconds dir =
  let first =
    Stdlib.Mutex.lock swept_mutex;
    let first = not (Hashtbl.mem swept dir) in
    if first then Hashtbl.set swept ~key:dir ~data:();
    Stdlib.Mutex.unlock swept_mutex;
    first
  in
  if first then cleanup_stale ?max_age_seconds dir
