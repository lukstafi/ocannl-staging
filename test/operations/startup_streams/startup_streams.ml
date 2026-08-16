(* gh-ocannl-593 and gh-ocannl-595: what an OCANNL-linked executable puts on each stream before it
   reaches its own [main].

   The convention (gh-ocannl-581) is that stdout belongs to the program and every library
   diagnostic goes to stderr -- which is what lets `tools/fit_envelope.exe` and the benchmark
   runners make stdout a data channel. Nothing tested it: every other test config sets
   `suppress_welcome_message=true` and `log_config_sourcing=false`, so the whole suite would stay
   green with the [Stdio.eprintf] calls in `arrayjit/lib/utils.ml` flipped back to [Stdio.printf].

   This executable has no output of its own, which is what makes it usable as a subject: whatever a
   rule captures came from the library. The two rules next to it capture the two halves of the
   contract.

   - The stdout half: with the chatter turned all the way UP, stdout is still empty.
   - The stderr half: on a DEFAULT run, stderr is short enough that a warning on it is legible.
     That is the acceptance test of gh-ocannl-595, which is why `log_config_sourcing` defaults to
     false and `log_level` to 0. The golden is the welcome banner plus the unknown-config-key
     warning that this directory's `ocannl_config` provokes on purpose -- three lines, not
     eighty-four.

   The modes are argument-selected rather than three executables because the stdout half wants a
   subject that links the library and does nothing else, and the other two are a few lines each. *)

open Base
open Stdio

(* Keys that would rewrite the stderr golden if they arrived from the ambient environment, which
   outranks this directory's config file -- and dune tracks no environment variable but
   OCANNL_BACKEND, so a stale golden could be reused besides. As in `profiles/`, the executable
   cannot PREVENT this (the settings are read during [Utils]'s initialization) but detecting it is
   enough: a rule that exits nonzero writes no golden, so a mystifying diff becomes a named
   failure. Only the stderr rule guards; the stdout rule's claim is stream-level and holds under
   any of these. *)
let stderr_shaping_keys =
  [
    "suppress_welcome_message";
    "log_config_sourcing";
    "log_level";
    "no_config_file";
    "profile";
    "clean_up_build_files_on_startup";
    "clean_up_log_files_on_startup";
  ]

let guard () =
  List.iter stderr_shaping_keys ~f:(fun arg_name ->
      Option.iter (Utils.read_env_var arg_name) ~f:(fun (value, var) ->
          eprintf
            "startup_streams: %s=%s is set in the environment and would rewrite this test's \
             expected stderr; unset it to run the test.\n"
            var value;
          Stdlib.exit 1))

(* The captured stderr names the config file by the absolute path the walk-up search built, so the
   golden gets the basename instead. Also strips a trailing CR: on Windows the capture is
   text-mode, and the golden is pinned to LF by .gitattributes. *)
let normalize () =
  Out_channel.set_binary_mode stdout true;
  In_channel.iter_lines In_channel.stdin ~f:(fun line ->
      let line = String.rstrip ~drop:(Char.equal '\r') line in
      String.split line ~on:' '
      |> List.map ~f:(fun token ->
             if String.is_substring token ~substring:"ocannl_config" then
               Stdlib.Filename.basename token
             else token)
      |> String.concat ~sep:" "
      |> printf "%s\n")

let () =
  (* The configuration flags a rule passes are addressed to the library, which read them during
     initialization; the mode is whatever else is on the commandline. *)
  match
    Array.to_list (Array.subo Stdlib.Sys.argv ~pos:1)
    |> List.filter ~f:(Fn.non (String.is_prefix ~prefix:"-"))
  with
  (* The stdout half: the library has already spoken by the time this runs. *)
  | [] -> ()
  | "guard" :: _ -> guard ()
  | "normalize" :: _ -> normalize ()
  | arg :: _ ->
      eprintf "startup_streams: unknown mode %S (expected none, \"guard\" or \"normalize\")\n" arg;
      Stdlib.exit 1
