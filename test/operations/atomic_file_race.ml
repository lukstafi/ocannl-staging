(* Concurrent writers against {!Utils.Atomic_file} (gh-ocannl-780).

   The property under test is the one the schedule cache and the checkpoint writer depend on: a
   reader of a published path observes a COMPLETE payload — the previous one or a new one — never a
   torn one, and never a missing file. The hazards are three, and there is a leg for each: two
   writers sharing one staging artifact (uniqueness), a writer that fails between staging and commit
   (failure cleanup), and a writer killed in that window (crash-stale cleanup).

   Everything here is bounded by fixed iteration counts rather than by wall-clock time, and the one
   place a domain waits for another spins under an explicit cap that FAILS the run rather than
   hanging it. The reader's notion of a well-formed payload is itself controlled, on hand-built torn
   and mixed payloads, so a leg that reports "every read was complete" is reporting a check that can
   say otherwise.

   The stress leg was controlled the other way too, by hand: replacing its writers' [AF.write_all]
   with a direct [Out_channel.write_all] onto the published path makes "every read under concurrent
   publication observes a complete payload" report false. The volume below is chosen to reproduce
   that reliably, so a regression in the helper is caught rather than raced past. *)

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

(* A payload is a run of one character whose length encodes the round. Torn reads are visible from
   both sides: a prefix or a suffix of a longer payload has a length off the lattice, and a payload
   two writers interleaved into one file has more than one character. *)
let base_length = 997
let length_step = 41
let payload ~writer ~round =
  String.make (base_length + (length_step * round)) (Char.of_int_exn writer)

let is_well_formed data =
  (not (String.is_empty data))
  && String.for_all data ~f:(Char.equal data.[0])
  && String.length data >= base_length
  && Int.rem (String.length data - base_length) length_step = 0

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
  Verdict.p "the completeness test rejects a payload cut on the lattice but mixed"
    (not (is_well_formed (String.prefix good base_length ^ String.make length_step 'b')));
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

(* The stress leg. Fixed counts, no timing dependence: whatever order the domains happen to run in,
   every read must land on a complete payload and every publish must commit. *)
let writers = 4
let rounds = 150
let readers = 2
let reads_per_reader = 400

let () =
  reset_dir ();
  AF.write_all ~path:target ~data:(payload ~writer:(Char.to_int 'a') ~round:0) ();
  let refusals = Array.create ~len:writers 0 in
  let observations = Array.create ~len:readers [] in
  let go = Atomic.make false in
  let writer_domains =
    Array.init writers ~f:(fun i ->
        Domain.spawn (fun () ->
            let writer = Char.to_int 'a' + i in
            while not (Atomic.get go) do
              Domain.cpu_relax ()
            done;
            for round = 0 to rounds - 1 do
              try AF.write_all ~path:target ~data:(payload ~writer ~round) ()
              with _ -> refusals.(i) <- refusals.(i) + 1
            done))
  in
  let reader_domains =
    Array.init readers ~f:(fun i ->
        Domain.spawn (fun () ->
            while not (Atomic.get go) do
              Domain.cpu_relax ()
            done;
            let seen = ref [] in
            for _ = 1 to reads_per_reader do
              seen := read_published () :: !seen
            done;
            observations.(i) <- !seen))
  in
  Atomic.set go true;
  Array.iter writer_domains ~f:Domain.join;
  Array.iter reader_domains ~f:Domain.join;
  let seen = List.concat (Array.to_list observations) in
  Verdict.p_all ~min:(readers * reads_per_reader)
    "every read under concurrent publication observes a complete payload" seen ~f:(function
    | None -> false
    | Some data -> is_well_formed data);
  Verdict.p_none ~min:(readers * reads_per_reader)
    "no read under concurrent publication finds the path missing" seen ~f:Option.is_none;
  Verdict.p_all ~min:writers "every concurrent writer committed every round"
    (Array.to_list refusals) ~f:(fun n -> n = 0);
  Verdict.p "the file left by the race is a complete payload"
    (Option.value_map (read_published ()) ~default:false ~f:is_well_formed);
  Verdict.p "the race leaves no staging file behind" (List.is_empty (staging_leftovers ()));
  Verdict.p "the race leaves only the published file"
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

(* Crash-stale cleanup: the writer that dies in its commit window cannot clean up after itself, so
   the sweep must — by age, and only over this module's own artifacts. *)
let age_seconds = 3600.

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
  let stale = plant_staging ~name:("published.bin" ^ AF.staging_infix ^ "4242.0") ~age:7200. in
  let fresh = plant_staging ~name:("published.bin" ^ AF.staging_infix ^ "4242.1") ~age:0. in
  let bystander = plant_staging ~name:"unrelated.bin" ~age:7200. in
  (* Names carrying the infix that this module did NOT generate. The sweep deletes what the
     predicate accepts, so each of these is a file somebody else owns (Codex P2, round 1): a
     descriptive suffix instead of the counter, a missing counter, a non-numeric pid, and a staging
     name with no target in front of it. *)
  let impostors =
    [
      "report" ^ AF.staging_infix ^ "backup";
      "report" ^ AF.staging_infix ^ "4242";
      "report" ^ AF.staging_infix ^ "host7.0";
      AF.staging_infix ^ "4242.0";
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
  let other_target = plant_staging ~name:("other.bin" ^ AF.staging_infix ^ "4242.9") ~age:7200. in
  Verdict.p "another target's staging file is not recognized as this one's"
    (not (AF.is_staging_file_for ~path:target (Stdlib.Filename.basename other_target)));
  AF.cleanup_stale_for ~max_age_seconds:age_seconds target;
  Verdict.p "the narrow sweep removes this target's abandoned staging file"
    (not (Stdlib.Sys.file_exists stale));
  Verdict.p "the narrow sweep spares another target's staging file"
    (Stdlib.Sys.file_exists other_target);
  let stale = plant_staging ~name:("published.bin" ^ AF.staging_infix ^ "4242.0") ~age:7200. in
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
    let path = Stdlib.Filename.concat absent ("late.bin" ^ AF.staging_infix ^ "4242.0") in
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
  let after_once = plant_staging ~name:("published.bin" ^ AF.staging_infix ^ "4242.2") ~age:7200. in
  AF.cleanup_stale_once ~max_age_seconds:age_seconds dir;
  Verdict.p "the once-per-process sweep does not run a second time"
    (Stdlib.Sys.file_exists after_once);
  AF.cleanup_stale ~max_age_seconds:age_seconds dir;
  Verdict.p "the unconditional sweep still removes it" (not (Stdlib.Sys.file_exists after_once));
  reset_dir ()
