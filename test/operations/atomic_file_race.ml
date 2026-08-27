(* Concurrent writers against {!Utils.Atomic_file} (gh-ocannl-780).

   The property under test is the one the schedule cache and the checkpoint writer depend on: a
   reader of a published path observes a COMPLETE payload — the previous one or a new one — never a
   torn one, and never a missing file. The hazards are three, and there is a leg for each: two
   writers sharing one staging artifact (uniqueness), a writer that fails between staging and commit
   (failure cleanup), and a writer killed in that window (crash-stale cleanup).

   Everything here is bounded by fixed iteration counts rather than by wall-clock time, and every
   place a domain waits for another spins under an explicit cap that FAILS the run rather than
   hanging it. Two things make the stress leg able to fail rather than merely pass:

   - Its readers are compared against the EXACT set of payloads the writers published, not against a
     shape. A shape check accepted a torn read whose truncation happened to land on the length
     lattice (Codex P2, round 2); the payloads are self-describing now, and the shape check is kept
     only to control that counterexample explicitly.
   - Readers and writers are made to overlap by a rendezvous rather than by hoping the scheduler
     interleaves them. On one core the readers could otherwise finish every read against the seeded
     file before a writer ran at all, and pass while publication truncated the target in place
     (Codex P2, round 2). Each reader now reads once before any writer proceeds past its first
     round, and keeps reading until the writers are done; every reader is claimed to have observed
     the file mid-run.

   The stress leg was controlled by hand in the other direction too: replacing its writers'
   [AF.write_all] with a direct [Out_channel.write_all] onto the published path makes "every read
   under concurrent publication observes a payload some writer published" report false. *)

open Base
module AF = Utils.Atomic_file

let dir = "atomic_file_race_dir"
let target = Stdlib.Filename.concat dir "published.bin"

let listing () =
  match Stdlib.Sys.readdir dir with
  | exception _ -> []
  | entries -> List.sort ~compare:String.compare (Array.to_list entries)

let staging_leftovers () = List.filter (listing ()) ~f:AF.is_staging_file

let reset_dir () =
  AF.ensure_dir dir;
  List.iter (listing ()) ~f:(fun name ->
      try Stdlib.Sys.remove (Stdlib.Filename.concat dir name) with _ -> ())

(* A payload is SELF-DESCRIBING: it names its writer, its round and its own body length, and ends
   with a terminator. Length alone is not enough — a run of one character truncated by a multiple of
   the step is a shorter round's payload exactly (Codex P2, round 2) — but a body that disagrees
   with its declared length, or a payload missing its terminator, is torn however it was cut. Mixing
   two writers' bytes shows up in the body's characters. *)
let base_length = 997
let length_step = 41

let payload ~writer ~round =
  let tag = Char.of_int_exn writer in
  let body_length = base_length + (length_step * round) in
  Printf.sprintf "%c|%d|%d|%s|END" tag round body_length (String.make body_length tag)

let is_well_formed data =
  match String.split data ~on:'|' with
  | [ tag; round; declared; body; "END" ] -> (
      String.length tag = 1
      &&
      match (Stdlib.int_of_string_opt round, Stdlib.int_of_string_opt declared) with
      | Some round, Some declared ->
          round >= 0
          && declared = base_length + (length_step * round)
          && String.length body = declared
          && String.for_all body ~f:(Char.equal tag.[0])
      | _ -> false)
  | _ -> false

let read_published () = try Some (Stdio.In_channel.read_all target) with _ -> None

(* Bounded spin: the deterministic-interleaving leg needs one domain to reach a known state before
   the other proceeds, and a wait that cannot end must fail rather than hang the suite. *)
let spin_limit = 20_000
let spin_pause = 0.0005

let await_flag flag =
  let rec loop n =
    if Atomic.get flag then true
    else if n >= spin_limit then false
    else (
      Unix.sleepf spin_pause;
      loop (n + 1))
  in
  loop 0

(* Control: the reader's verdict discriminates. Without this, "every read was complete" is a claim
   about a predicate nobody checked. *)
let () =
  let good = payload ~writer:(Char.to_int 'a') ~round:3 in
  Verdict.p "the completeness test accepts a whole payload" (is_well_formed good);
  Verdict.p "the completeness test rejects a truncated payload"
    (not (is_well_formed (String.prefix good 512)));
  (* The counterexample the length-lattice shape check accepted: round 3's payload cut to exactly
     round 1's length. Under a shape check keyed on length alone the two are indistinguishable. *)
  Verdict.p "the completeness test rejects a truncation aligned to the length lattice"
    (not
       (is_well_formed
          (String.prefix good (String.length (payload ~writer:(Char.to_int 'a') ~round:1)))));
  Verdict.p "the completeness test rejects a payload missing its terminator"
    (not (is_well_formed (String.drop_suffix good 4)));
  Verdict.p "the completeness test rejects a payload whose body mixes two writers"
    (not
       (is_well_formed
          (String.prefix good (String.length good - 8) ^ String.make 4 'b' ^ "|END")));
  Verdict.p "the completeness test rejects an empty file" (not (is_well_formed ""))

(* A staged-but-uncommitted writer is invisible, and it obstructs nobody: while one publish sits in
   its commit window, a whole second publish to the same path succeeds, and the first still commits
   its own complete payload afterwards. This is the deterministic form of the race the stress leg
   below runs at volume. *)
let () =
  reset_dir ();
  let seed = payload ~writer:(Char.to_int 'a') ~round:0 in
  AF.write_all ~path:target ~data:seed ();
  let staged = Atomic.make false in
  let release = Atomic.make false in
  let blocked = payload ~writer:(Char.to_int 'b') ~round:1 in
  let waited = ref false in
  let writer =
    Domain.spawn (fun () ->
        AF.write_all ~path:target ~data:blocked
          ~before_commit:(fun () ->
            Atomic.set staged true;
            waited := await_flag release)
          ())
  in
  Verdict.p "the blocked writer reaches its commit window" (await_flag staged);
  let during_stage = read_published () in
  Verdict.p "a staged payload is invisible until it commits"
    (Option.equal String.equal during_stage (Some seed));
  Verdict.p "exactly the blocked writer's staging file is present"
    (List.length (staging_leftovers ()) = 1);
  let overtaking = payload ~writer:(Char.to_int 'c') ~round:2 in
  AF.write_all ~path:target ~data:overtaking ();
  Verdict.p "a competing publish commits while another writer holds its staging file"
    (Option.equal String.equal (read_published ()) (Some overtaking));
  Verdict.p "the competing publish did not disturb the blocked writer's staging file"
    (List.length (staging_leftovers ()) = 1);
  Atomic.set release true;
  Domain.join writer;
  Verdict.p "the released writer commits its own complete payload"
    (Option.equal String.equal (read_published ()) (Some blocked));
  Verdict.p "the blocked writer waited rather than timing out" !waited;
  Verdict.p "no staging file survives the interleaving" (List.is_empty (staging_leftovers ()))

(* The stress leg. Fixed counts and a rendezvous, so the interleaving is arranged rather than hoped
   for: every writer publishes its first round and then WAITS for every reader to have read, and
   every reader keeps reading until the writers are done. Both halves are needed — the first makes
   readers observe a directory under publication, the second keeps them reading through it. *)
let writers = 4
let rounds = 150
let readers = 2
let reads_per_reader = 400

(* A ceiling on the reader loop, so a starved writer domain cannot turn "read until the writers
   finish" into an unbounded run. Far above [reads_per_reader]: reaching it is a pathology, and the
   overlap claim below is what would then fail. *)
let max_reads = 20_000

let total_publications = writers * rounds

let await ?(limit = spin_limit) predicate =
  let rec loop n =
    if predicate () then true
    else if n >= limit then false
    else (
      Unix.sleepf spin_pause;
      loop (n + 1))
  in
  loop 0

let () =
  reset_dir ();
  let seed = payload ~writer:(Char.to_int 'a') ~round:0 in
  AF.write_all ~path:target ~data:seed ();
  (* Every payload any writer will publish, plus the seed: what a reader observes must be one of
     these EXACTLY, not merely a string shaped like one. *)
  let published = Hash_set.create (module String) in
  Hash_set.add published seed;
  for i = 0 to writers - 1 do
    for round = 0 to rounds - 1 do
      Hash_set.add published (payload ~writer:(Char.to_int 'a' + i) ~round)
    done
  done;
  let refusals = Array.create ~len:writers 0 in
  (* Each observation carries the number of publications completed when it was taken, which is how a
     reader reports where in the run it read. *)
  let no_reads = ((None, 0), []) in
  let observations = Array.create ~len:readers no_reads in
  let committed = Atomic.make 0 in
  let readers_read = Atomic.make 0 in
  let writers_finished = Atomic.make 0 in
  let half_written = Atomic.make false in
  let writers_met = Array.create ~len:writers false in
  let readers_met = Array.create ~len:readers false in
  let publish_round i writer round =
    (try
       AF.write_all ~path:target ~data:(payload ~writer ~round) ();
       Atomic.incr committed
     with _ -> refusals.(i) <- refusals.(i) + 1);
    ()
  in
  (* The gate. One writer stops HALFWAY THROUGH ITS PAYLOAD -- not between two publications -- and
     every reader takes a read before it continues. Pausing between publications was not enough
     (Codex P2, round 5): on one core the readers could take every read while the writers waited,
     so nothing was read during a mutation and a truncate-in-place implementation would have passed.
     Stopped here, the bytes of a partial payload exist on disk; an implementation that wrote them
     into the target would be caught by the exact-set claim below, on any number of cores. *)
  let publish_gated i writer round =
    let whole = payload ~writer ~round in
    let half = String.length whole / 2 in
    (try
       AF.with_channel ~path:target () ~f:(fun oc ->
           Stdlib.output_string oc (String.prefix whole half);
           Stdlib.flush oc;
           Atomic.set half_written true;
           writers_met.(i) <- await (fun () -> Atomic.get readers_read >= readers);
           Stdlib.output_string oc (String.drop_prefix whole half));
       Atomic.incr committed
     with _ -> refusals.(i) <- refusals.(i) + 1);
    ()
  in
  let writer_domains =
    Array.init writers ~f:(fun i ->
        Domain.spawn (fun () ->
            let writer = Char.to_int 'a' + i in
            if i = 0 then publish_gated i writer 0
            else (
              publish_round i writer 0;
              writers_met.(i) <- await (fun () -> Atomic.get readers_read >= readers));
            for round = 1 to rounds - 1 do
              publish_round i writer round
            done;
            Atomic.incr writers_finished))
  in
  let reader_domains =
    Array.init readers ~f:(fun i ->
        Domain.spawn (fun () ->
            readers_met.(i) <- await (fun () -> Atomic.get half_written);
            (* Taken with a payload half-written on disk. *)
            let seen = ref [ (read_published (), Atomic.get committed) ] in
            let during_write = List.hd_exn !seen in
            Atomic.incr readers_read;
            let rec loop n =
              if n >= max_reads then ()
              else if n >= reads_per_reader && Atomic.get writers_finished >= writers then ()
              else (
                seen := (read_published (), Atomic.get committed) :: !seen;
                loop (n + 1))
            in
            loop 1;
            (* One last read after the loop's exit condition held, so the final observation is taken
               with every writer finished: that is what makes "kept reading until the writers were
               done" a fact about this reader rather than about the loop's shape. *)
            seen := (read_published (), Atomic.get committed) :: !seen;
            observations.(i) <- (during_write, !seen)))
  in
  Array.iter writer_domains ~f:Domain.join;
  Array.iter reader_domains ~f:Domain.join;
  let per_reader = Array.to_list observations in
  let seen = List.concat_map per_reader ~f:snd in
  Verdict.p_all ~min:writers "every writer met the readers at the rendezvous"
    (Array.to_list writers_met) ~f:Fn.id;
  Verdict.p_all ~min:readers "every reader saw a payload go half-written"
    (Array.to_list readers_met) ~f:Fn.id;
  (* The overlap claim, and the one that makes this leg independent of how many cores run it: the
     read below was taken while a writer sat inside its payload with half of it already on disk. *)
  Verdict.p_all ~min:readers "every reader's gated read saw a whole payload while one was half-written"
    per_reader
    ~f:(fun ((data, _), _) -> Option.value_map data ~default:false ~f:(Hash_set.mem published));
  Verdict.p_all ~min:readers "every reader kept reading until the writers were done" per_reader
    ~f:(fun (_, obs) ->
      List.exists obs ~f:(fun (_, at) -> at = total_publications));
  Verdict.p_all ~min:(readers * reads_per_reader)
    "every read under concurrent publication observes a payload some writer published" seen
    ~f:(function
    | None, _ -> false
    | Some data, _ -> Hash_set.mem published data);
  Verdict.p_none ~min:(readers * reads_per_reader)
    "no read under concurrent publication finds the path missing" seen ~f:(fun (data, _) ->
      Option.is_none data);
  Verdict.p_all ~min:writers "every concurrent writer committed every round"
    (Array.to_list refusals) ~f:(fun n -> n = 0);
  Verdict.p "every publication was accounted for" (Atomic.get committed = total_publications);
  Verdict.p "the file left by the race is a complete payload"
    (Option.value_map (read_published ()) ~default:false ~f:is_well_formed);
  Verdict.p "the race leaves no staging file behind" (List.is_empty (staging_leftovers ()));
  Verdict.p "the race leaves only the published file"
    (List.equal String.equal (listing ()) [ "published.bin" ])

(* A staging name is a DERIVED name, and the two things derivation must guarantee are that it fits
   where the target fits and that no two attempts pick the same one. Both are observed through the
   commit window, which is the only moment a staging file exists. *)
let () =
  reset_dir ();
  (* A basename close to the per-component limit: with the old four-character suffix it fit, and
     with a naive suffix it would not (Codex P2, round 2). *)
  let long_name = String.make 240 'n' ^ ".bin" in
  let long_target = Stdlib.Filename.concat dir long_name in
  let staged = ref [] in
  let capture () = staged := staging_leftovers () @ !staged in
  AF.write_all ~path:long_target ~data:"payload" ~before_commit:capture ();
  Verdict.p "a target named to the filesystem's limit publishes"
    (Stdlib.Sys.file_exists long_target);
  Verdict.p_all "every staging name fits one filesystem component" !staged ~f:(fun name ->
      String.length name <= 255);
  Verdict.p_all "a long target's staging file is recognized as that target's" !staged
    ~f:(AF.is_staging_file_for ~path:long_target);
  (* A long name whose characters are multibyte: a byte-wise cut can land inside one, and the
     malformed name that results is refused by Windows and by APFS alike (Codex P2, round 4). The
     3-byte character makes the budget's cut fall mid-character.
     70 of them, not 90: a component limit is 255 BYTES on ext4 while macOS counts UTF-16 units, so
     90 snowmen are a legal target name here and an illegal one on Linux, where this leg would have
     failed before reaching an assertion (Codex P1, round 7). The claim below is what pins that -- a
     fixture has to be a name every filesystem accepts, or it tests the filesystem instead. *)
  let utf8_name = String.concat (List.init 70 ~f:(fun _ -> "\xe2\x98\x83")) ^ ".bin" in
  let utf8_target = Stdlib.Filename.concat dir utf8_name in
  let utf8_staged = ref [] in
  AF.write_all ~path:utf8_target ~data:"payload"
    ~before_commit:(fun () -> utf8_staged := staging_leftovers () @ !utf8_staged)
    ();
  Verdict.p_all "every fixture name is a valid component on a 255-byte filesystem"
    [ long_name; utf8_name ]
    ~f:(fun name -> String.length name <= 255);
  (* Both fixtures must actually reach the truncating path, or they test the short-name branch under
     a long-looking name: a stem that fit would appear in the staging name verbatim. *)
  Verdict.p_all "the long fixtures exercise the truncating stem" (!staged @ !utf8_staged)
    ~f:(fun name ->
      not (String.is_prefix name ~prefix:long_name || String.is_prefix name ~prefix:utf8_name));
  Verdict.p "a multibyte target name publishes" (Stdlib.Sys.file_exists utf8_target);
  Verdict.p_all "every staging name of a multibyte target is itself valid UTF-8" !utf8_staged
    ~f:Stdlib.String.is_valid_utf_8;
  Verdict.p_all "a multibyte target's staging file fits one filesystem component" !utf8_staged
    ~f:(fun name -> String.length name <= 255);
  Verdict.p_all "a multibyte target's staging file is recognized as that target's" !utf8_staged
    ~f:(AF.is_staging_file_for ~path:utf8_target);
  (* Case: on Windows and on a default macOS volume these two spellings are one file. *)
  let shouting = Stdlib.Filename.concat dir (String.uppercase long_name) in
  Verdict.p_all "a differently-cased spelling of the target claims its staging files" !staged
    ~f:(AF.is_staging_file_for ~path:shouting);
  (* Uniqueness within a process, observed rather than assumed: each attempt's staging file is
     captured in its own commit window, and no name repeats. Uniqueness ACROSS processes rests on
     exclusive creation, which no single-process test can exercise. *)
  reset_dir ();
  let names = ref [] in
  for round = 0 to 49 do
    AF.write_all ~path:target
      ~data:(payload ~writer:(Char.to_int 'a') ~round)
      ~before_commit:(fun () -> names := staging_leftovers () @ !names)
      ()
  done;
  Verdict.p_all ~min:50 "every attempt stages exactly one file" !names ~f:AF.is_staging_file;
  Verdict.p "no two staging names repeat"
    (List.length (List.dedup_and_sort !names ~compare:String.compare) = List.length !names);
  Verdict.p_all ~min:50 "a differently-cased short target claims its staging files too" !names
    ~f:(AF.is_staging_file_for ~path:(Stdlib.Filename.concat dir "PUBLISHED.BIN"));
  Verdict.p "the repeated publishing left only the published file"
    (List.equal String.equal (listing ()) [ "published.bin" ])

(* A writer that fails in its commit window leaves the previous entry and no artifact — for a
   payload held in memory and for one streamed through a channel alike. *)
let () =
  reset_dir ();
  let seed = payload ~writer:(Char.to_int 'a') ~round:0 in
  AF.write_all ~path:target ~data:seed ();
  let failed =
    try
      AF.write_all ~path:target
        ~data:(payload ~writer:(Char.to_int 'b') ~round:1)
        ~before_commit:(fun () -> failwith "gh780 injected commit failure")
        ();
      false
    with _ -> true
  in
  Verdict.p "a publish that fails in its commit window raises" failed;
  Verdict.p "a failed publish leaves the previous payload"
    (Option.equal String.equal (read_published ()) (Some seed));
  Verdict.p "a failed publish removes its own staging file" (List.is_empty (staging_leftovers ()));
  let streamed = payload ~writer:(Char.to_int 'c') ~round:2 in
  AF.with_channel ~path:target () ~f:(fun oc -> Stdlib.output_string oc streamed);
  Verdict.p "a streamed publish commits its whole payload"
    (Option.equal String.equal (read_published ()) (Some streamed));
  let stream_failed =
    try
      let () =
        AF.with_channel ~path:target () ~f:(fun oc ->
            Stdlib.output_string oc "half";
            (failwith "gh780 injected stream failure" : unit))
      in
      false
    with _ -> true
  in
  Verdict.p "a streamed publish that raises mid-write raises" stream_failed;
  Verdict.p "a failed streamed publish leaves the previous payload"
    (Option.equal String.equal (read_published ()) (Some streamed));
  Verdict.p "a failed streamed publish removes its own staging file"
    (List.is_empty (staging_leftovers ()))

(* One exception type for filesystem refusals, whatever refused. A best-effort writer -- the schedule
   cache treats a refusal as a future miss rather than a failed tuning run -- needs one handler, not
   a taxonomy of the operations the helper happens to use internally; the exclusive open raises
   [Unix_error], and letting that escape would have walked straight past such a handler (Codex P2,
   round 3). What [f] and [before_commit] raise is the caller's own and must NOT be converted. *)
let classify f =
  match f () with
  | () -> `Returned
  | exception Stdlib.Sys_error _ -> `Sys_error
  | exception Failure msg when String.is_substring msg ~substring:"gh780" -> `Caller
  | exception _ -> `Other

let () =
  reset_dir ();
  let refusals =
    [
      (* The directory does not exist. *)
      Stdlib.Filename.concat (Stdlib.Filename.concat dir "no_such_dir") "x.bin";
      (* A path component that is a file rather than a directory. *)
      Stdlib.Filename.concat target "x.bin";
    ]
  in
  AF.write_all ~path:target ~data:"seed" ();
  Verdict.p_all "every filesystem refusal reaches the caller as Sys_error" refusals ~f:(fun path ->
      Poly.equal `Sys_error (classify (fun () -> AF.write_all ~path ~data:"payload" ())));
  Verdict.p "an exception from the caller's own writer is not converted"
    (Poly.equal `Caller
       (classify (fun () ->
            AF.with_channel ~path:target () ~f:(fun _oc ->
                (failwith "gh780 caller failure" : unit)))));
  Verdict.p "an exception from the commit hook is not converted"
    (Poly.equal `Caller
       (classify (fun () ->
            AF.write_all ~path:target ~data:"payload"
              ~before_commit:(fun () -> failwith "gh780 hook failure")
              ())));
  Verdict.p "a refused publish leaves no staging file" (List.is_empty (staging_leftovers ()))

(* Crash-stale cleanup: the writer that dies in its commit window cannot clean up after itself, so
   the sweep must — by age, and only over this module's own artifacts. *)
let age_seconds = 3600.

(* A staging name a writer could have produced: the generated shape is the stem, the infix, and then
   three FIXED-WIDTH hex fields -- pid, counter, nonce. Only names of that shape are the sweep's to
   delete, and only they are the ones `.gitignore` hides. *)
let nonce = "00c0ffee00c0ffee"

let staged_name ~target ~counter =
  Printf.sprintf "%s%s%08x.%08x.%s" target AF.staging_infix 4242 counter nonce

let plant_staging ~name ~age =
  let path = Stdlib.Filename.concat dir name in
  Stdio.Out_channel.write_all path ~data:"abandoned";
  let stamp = Unix.time () -. age in
  Unix.utimes path stamp stamp;
  path

let () =
  reset_dir ();
  let seed = payload ~writer:(Char.to_int 'a') ~round:0 in
  AF.write_all ~path:target ~data:seed ();
  let stale = plant_staging ~name:(staged_name ~target:"published.bin" ~counter:0) ~age:7200. in
  let fresh = plant_staging ~name:(staged_name ~target:"published.bin" ~counter:1) ~age:0. in
  let bystander = plant_staging ~name:"unrelated.bin" ~age:7200. in
  (* Names carrying the infix that this module did NOT generate. The sweep deletes what the
     predicate accepts, so each of these is a file somebody else owns (Codex P2, round 1): a
     descriptive suffix instead of the counter, a missing counter, a non-numeric pid, and a staging
     name with no target in front of it. *)
  let impostors =
    [
      "report" ^ AF.staging_infix ^ "backup";
      "report" ^ AF.staging_infix ^ "4242";
      "report" ^ AF.staging_infix ^ "4242.0";
      "report" ^ AF.staging_infix ^ "host7.0." ^ nonce;
      "report" ^ AF.staging_infix ^ "00001092.00000000.nonsense0nonsense";
      (* Hex, but not at the widths this module generates -- in the nonce and in the fields both. *)
      "report" ^ AF.staging_infix ^ "1.2.a";
      "report" ^ AF.staging_infix ^ "00001092.00000000." ^ nonce ^ "0";
      "report" ^ AF.staging_infix ^ "1092.0." ^ nonce;
      (* The shape a `[0-9]*` glob would have swallowed: fields that start numeric and go on. *)
      "report" ^ AF.staging_infix ^ "1abc.2bar." ^ nonce;
      (* Every field right, but a stem longer than generation can emit: it always truncates. *)
      String.make 192 's' ^ AF.staging_infix ^ "00001092.00000000." ^ nonce;
      AF.staging_infix ^ "4242.0." ^ nonce;
    ]
  in
  let planted_impostors = List.map impostors ~f:(fun name -> plant_staging ~name ~age:7200.) in
  Verdict.p_all "every planted staging file is recognized as one"
    [ stale; fresh ]
    ~f:(fun path -> AF.is_staging_file (Stdlib.Filename.basename path));
  Verdict.p "the bystander is not recognized as a staging file"
    (not (AF.is_staging_file (Stdlib.Filename.basename bystander)));
  Verdict.p_none "no name that merely contains the infix is recognized as a staging file" impostors
    ~f:AF.is_staging_file;
  (* The narrow scope: whose staging file it is, not merely that it is one. *)
  Verdict.p_all "the published file's own staging files are recognized as its"
    [ stale; fresh ]
    ~f:(fun path -> AF.is_staging_file_for ~path:target (Stdlib.Filename.basename path));
  let other_target = plant_staging ~name:(staged_name ~target:"other.bin" ~counter:9) ~age:7200. in
  Verdict.p "another target's staging file is not recognized as this one's"
    (not (AF.is_staging_file_for ~path:target (Stdlib.Filename.basename other_target)));
  AF.cleanup_stale_for ~max_age_seconds:age_seconds target;
  Verdict.p "the narrow sweep removes this target's abandoned staging file"
    (not (Stdlib.Sys.file_exists stale));
  Verdict.p "the narrow sweep spares another target's staging file"
    (Stdlib.Sys.file_exists other_target);
  let stale = plant_staging ~name:(staged_name ~target:"published.bin" ~counter:0) ~age:7200. in
  AF.cleanup_stale ~max_age_seconds:age_seconds dir;
  Verdict.p "the sweep removes an abandoned staging file" (not (Stdlib.Sys.file_exists stale));
  Verdict.p "the sweep removes another target's abandoned staging file too"
    (not (Stdlib.Sys.file_exists other_target));
  Verdict.p "the sweep spares a staging file young enough to be in flight"
    (Stdlib.Sys.file_exists fresh);
  Verdict.p "the sweep spares the published file" (Stdlib.Sys.file_exists target);
  Verdict.p "the sweep spares an aged file that is not a staging artifact"
    (Stdlib.Sys.file_exists bystander);
  Verdict.p_all "the sweep spares every aged file that merely contains the infix" planted_impostors
    ~f:Stdlib.Sys.file_exists;
  Verdict.p "the published file still reads as it was written"
    (Option.equal String.equal (read_published ()) (Some seed));
  (* A cache's reader calls the once-per-process sweep before its first writer has created the
     directory, so a missing directory must not consume the one sweep. *)
  let absent = Stdlib.Filename.concat dir "not_created_yet" in
  AF.cleanup_stale_once ~max_age_seconds:age_seconds absent;
  AF.ensure_dir absent;
  let planted_after_creation =
    let path = Stdlib.Filename.concat absent (staged_name ~target:"late.bin" ~counter:0) in
    Stdio.Out_channel.write_all path ~data:"abandoned";
    let stamp = Unix.time () -. 7200. in
    Unix.utimes path stamp stamp;
    path
  in
  AF.cleanup_stale_once ~max_age_seconds:age_seconds absent;
  Verdict.p "a directory that did not exist yet does not consume its one sweep"
    (not (Stdlib.Sys.file_exists planted_after_creation));
  (try Stdlib.Sys.rmdir absent with _ -> ());
  (* The once-per-process form sweeps this directory exactly once, however many writers call it: a
     staging file abandoned after that first sweep survives until the next process. *)
  AF.cleanup_stale_once ~max_age_seconds:age_seconds dir;
  let after_once = plant_staging ~name:(staged_name ~target:"published.bin" ~counter:2) ~age:7200. in
  AF.cleanup_stale_once ~max_age_seconds:age_seconds dir;
  Verdict.p "the once-per-process sweep does not run a second time"
    (Stdlib.Sys.file_exists after_once);
  AF.cleanup_stale ~max_age_seconds:age_seconds dir;
  Verdict.p "the unconditional sweep still removes it" (not (Stdlib.Sys.file_exists after_once));
  reset_dir ()
