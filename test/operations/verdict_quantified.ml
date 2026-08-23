(* The non-emptiness guarantee of Verdict's quantified claims (gh-ocannl-729).

   "every X holds" is TRUE of an empty X. Written as `p "every seed spreads j" (List.for_all seeds
   ~f:...)` it prints the same passing line whether the property was checked on a hundred seeds or
   on none, and the golden records it as verified either way -- the gh-ocannl-601 hazard one level
   up, arriving through `Verdict.p` itself. `p_all`, `p_none`, `p_empty` and `p_exists` carry the
   guard, so the claim is the shortest thing to write AND cannot pass on nothing.

   Every guarantee here fires only when a collection is empty, which in a green suite is never --
   the same shape as `generated_provenance`, and the same construction: the passing forms run
   directly, so a module that refused everything could not pass them, and the refusals run as CHILD
   processes whose streams this one captures. Capturing matters for the same reason it does there:
   a refusal prints this repository's failure marker (`FAIL:`, `FAILED:`), and a green run's log
   must not carry those words.

   Two properties, not one. The claim must FAIL on an empty collection -- exit status and a line
   naming emptiness -- and it must print BYTE-IDENTICALLY to `Verdict.p` when the collection is not
   empty, which is what lets ~44 test files convert without their goldens moving. The second is
   checked by running `p` and `p_all` in two children and comparing what each wrote. *)

open Base

let describe_status = function
  | Unix.WEXITED n -> Printf.sprintf "exited %d" n
  | Unix.WSIGNALED n -> Printf.sprintf "was killed by signal %d" n
  | Unix.WSTOPPED n -> Printf.sprintf "was stopped by signal %d" n

let ignore_unix f x = try f x with Unix.Unix_error _ -> ()

(* Runs one mode in a child and answers its status with its two streams kept APART: the shape check
   compares what a child wrote on stdout, which is the stream a `(test)` stanza diffs, and a
   refusal's stderr echo would drown that comparison. Through temporary files rather than pipes,
   for the reason `generated_provenance.run_child` gives: reading two pipes in sequence deadlocks
   once the unread one fills. *)
let run_child mode =
  let exe = Stdlib.Sys.executable_name in
  let capture suffix = Stdlib.Filename.temp_file "vq_child" suffix in
  let out_path = capture ".out" and err_path = capture ".err" in
  let open_capture p = Unix.openfile p [ Unix.O_WRONLY; Unix.O_TRUNC ] 0o600 in
  let out = open_capture out_path and err = open_capture err_path in
  let pid = Unix.create_process exe [| exe; mode |] Unix.stdin out err in
  let _, status = Unix.waitpid [] pid in
  Unix.close out;
  Unix.close err;
  let stdout_text = Stdio.In_channel.read_all out_path in
  let stderr_text = Stdio.In_channel.read_all err_path in
  ignore_unix Unix.unlink out_path;
  ignore_unix Unix.unlink err_path;
  (status, stdout_text, stderr_text)

(* Whether a child refused the way the mode under test is about: exit 1, and [line] among what it
   printed on stdout -- the stream the golden of a converted test is made of, so it is where the
   distinct wording has to appear. A failing check prints the whole capture to stderr, because the
   child's own account is what a reader needs and it is only withheld from PASSING runs. *)
let refused claim ~line (status, stdout_text, stderr_text) =
  let ok =
    (match status with Unix.WEXITED 1 -> true | _ -> false)
    && String.is_substring stdout_text ~substring:line
  in
  if not ok then
    Stdio.eprintf "the child %s without printing %S on stdout. Its capture:\n%s%s\n"
      (describe_status status) line stdout_text stderr_text;
  Verdict.p claim ok

let seeds = [ 2; 4; 6 ]
let even n = n % 2 = 0
let odd n = n % 2 = 1

let () =
  let mode =
    match Array.to_list Stdlib.Sys.argv with
    | _ :: m :: _ when not (String.is_prefix m ~prefix:"--") -> m
    | _ -> "holds"
  in
  match mode with
  (* === Must succeed: run directly by the dune rule. === *)
  | "holds" ->
      Verdict.p_all "every seed is even" seeds ~f:even;
      Verdict.p_all ~min:3 "every one of the three seeds is even" seeds ~f:even;
      Verdict.p_none "no seed is odd" seeds ~f:odd;
      Verdict.p_exists "some seed exceeds four" seeds ~f:(fun n -> n > 4);
      Verdict.p_empty "every seed validates" ~over:seeds (List.filter seeds ~f:odd);
      (* An array reaches the combinators through [Array.to_list]; the eight array sites in the
         sweep spell it that way rather than growing a second family of entry points. *)
      Verdict.p_all "every sampled value is finite" (Array.to_list [| 1.0; 2.0 |])
        ~f:Float.is_finite
  (* === Shape: what a non-empty collection prints, compared against [p]'s own line. === *)
  | "shape_p" -> Verdict.p "the claim" true
  | "shape_p_all" -> Verdict.p_all "the claim" seeds ~f:even
  | "shape_p_false" -> Verdict.p "the claim" false
  | "shape_p_all_false" -> Verdict.p_all "the claim" seeds ~f:odd
  (* === Must be refused: run as children by [refusals]. === *)
  | "all_empty" -> Verdict.p_all "every seed is even" [] ~f:even
  | "all_short" -> Verdict.p_all ~min:3 "every one of the three seeds is even" [ 2 ] ~f:even
  | "none_empty" -> Verdict.p_none "no seed is odd" [] ~f:odd
  | "exists_empty" -> Verdict.p_exists "some seed exceeds four" [] ~f:(fun n -> n > 4)
  | "empty_over_empty" -> Verdict.p_empty "every seed validates" ~over:[] []
  | "refusals" ->
      refused "an `every` claim over an empty collection fails rather than passing vacuously"
        ~line:"every seed is even (empty): false" (run_child "all_empty");
      refused "a collection below its stated floor fails, naming the shortfall"
        ~line:"every one of the three seeds is even (only 1 of 3): false" (run_child "all_short");
      refused "a `no X is` claim over an empty collection fails rather than passing vacuously"
        ~line:"no seed is odd (empty): false" (run_child "none_empty");
      refused "a `some X` claim over an empty collection names emptiness rather than the property"
        ~line:"some seed exceeds four (empty): false" (run_child "exists_empty");
      refused "an emptiness claim about a derived subset fails when the population is empty too"
        ~line:"every seed validates (empty): false" (run_child "empty_over_empty");
      (* The conversion is golden-neutral exactly to the extent that this holds. *)
      let _, plain_true, _ = run_child "shape_p" in
      let _, all_true, _ = run_child "shape_p_all" in
      Verdict.p "a satisfied quantified claim prints what `Verdict.p` prints"
        (String.equal plain_true all_true && String.equal plain_true "the claim: true\n");
      let _, plain_false, _ = run_child "shape_p_false" in
      let _, all_false, _ = run_child "shape_p_all_false" in
      Verdict.p "a non-empty collection that refutes the claim prints what `Verdict.p` prints"
        (String.equal plain_false all_false && String.equal plain_false "the claim: false\n")
  | other -> failwith ("verdict_quantified: unknown mode " ^ other)
