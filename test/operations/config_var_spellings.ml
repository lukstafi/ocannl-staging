(* gh-ocannl-605: the spellings a configuration setting has outside an ocannl_config file.

   Two lists, and then a live check that they are the truth rather than a wish. The environment
   list is the shorter one on purpose: a dune rule that has to declare the ambient variables it is
   invalidated by must enumerate it by hand, and while it held four spellings per key the
   natural-looking all-dashed `ocannl-log-level` was not among them -- a dep on it declared a
   variable nothing reads while leaving the one OCANNL does read untracked. Dashes are idiomatic
   on the commandline and stay there, where they are also normalized: the prefix separator and the
   key's own separators dash independently, so every spelling one would guess is accepted.

   The live section is what makes this a test rather than a restatement of the source. The rule
   next to it sets a synthetic key in every environment spelling that was dropped and in the one
   that was kept, and a second key in the dropped spellings only. Synthetic keys, deliberately:
   nothing there is a real setting, so the library reads no differently for having been asked.

   The commandline lookup names a REAL key, because the unknown-argument warning at the foot of
   `arrayjit/lib/utils.ml` validates the argv it did not consume -- and it normalizes every dash
   to an underscore, so `--ocannl-print-decimals-precision=7` was accepted as a spelling of a
   known key while `read_cmdline_var` ignored it, silently. Making the two agree is the other half
   of gh-ocannl-605, and this line is what pins it. *)

open Base
open Stdio

let show_names title names =
  printf "%s\n" title;
  List.iter names ~f:(printf "  %s\n")

let () =
  show_names "Environment variables for `log_level`:" (Utils.env_var_names "log_level");
  show_names "Environment variables for `profile`:" (Utils.env_var_names "profile");
  printf "\n";
  (* Each is followed by the value separator, one of `_`, `-`, `=`, or nothing. *)
  show_names "Commandline arguments for `log_level`:" (Utils.cmdline_var_names "log_level");
  show_names "Commandline arguments for `profile`, qualified_only (see `Utils.cmdline_var_names`):"
    (Utils.cmdline_var_names ~qualified_only:true "profile");
  printf "\n"

let show_lookup name key found =
  printf "  %s %S:%s%s\n" name key
    (String.make (max 1 (34 - String.length name - String.length key)) ' ')
    (match found with
    | None -> "None"
    | Some (value, source) -> Printf.sprintf "%s, from %s" value source)

let () =
  printf "Live lookups, against the environment and commandline the dune rule builds:\n";
  (* Set as `ocannl_demo_key`, `ocannl-demo_key`, `OCANNL-DEMO_KEY` and `ocannl-demo-key`: only
     the first is a spelling, and it is what the lookup must report. *)
  show_lookup "read_env_var" "demo_key" (Utils.read_env_var "demo_key");
  (* Set in the dropped spellings only -- so an unset key is what a caller of the dashed forms
     gets. *)
  show_lookup "read_env_var" "dashed_only_key" (Utils.read_env_var "dashed_only_key");
  (* The control the negative leg needs: without it that None would read the same way if the rule
     had set nothing at all, and the test would pass for the wrong reason. *)
  List.iter [ "ocannl-dashed_only_key"; "OCANNL-DASHED_ONLY_KEY" ] ~f:(fun var ->
      printf "    (control) %s is %s\n" var
        (match Stdlib.Sys.getenv_opt var with Some v -> "set to " ^ v | None -> "UNSET"));
  show_lookup "read_cmdline_var" "print_decimals_precision"
    (Utils.read_cmdline_var "print_decimals_precision")
