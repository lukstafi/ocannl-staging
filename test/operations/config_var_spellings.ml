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

   The commandline lookups name REAL keys, because the unknown-argument warning at the foot of
   `arrayjit/lib/utils.ml` inspects every argument addressed to OCANNL. It used to do so with a
   parser of its own -- split on `=`, dash to underscore, look up -- which disagreed with the
   reader in both directions: `--ocannl-print-decimals-precision=7` passed validation and was then
   ignored, while `--ocannl_log_level_0` was applied and warned about, its separator not being an
   `=`. Both now go through `Utils.cmdline_var_prefixes`, so these two lines and the sibling
   `config_var_warnings` golden are the two halves of one claim. *)

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
     had set nothing at all, and the test would pass for the wrong reason. Only the lowercase
     spelling is asked about, and it is the spelling the rule states: a lookup of the uppercase one
     answers differently on Windows, where environment names are case-insensitive and the two are
     one variable, and a control that reads differently per platform is not a control. *)
  let dropped = "ocannl-dashed_only_key" in
  printf "    (control) %s is %s\n" dropped
    (match Stdlib.Sys.getenv_opt dropped with Some v -> "set to " ^ v | None -> "UNSET");
  show_lookup "read_cmdline_var" "print_decimals_precision"
    (Utils.read_cmdline_var "print_decimals_precision");
  (* The `_` value separator, on the spelling the old validator called unknown while the reader
     applied it. *)
  show_lookup "read_cmdline_var" "log_level" (Utils.read_cmdline_var "log_level")

(* Which names the library considers addressed to its configuration (gh-ocannl-629). The two halves
   of the answer are here rather than only in the sibling `config_var_warnings` golden, where the
   reserved namespaces could only appear as an ABSENCE of warnings -- and an absence reads the same
   whether the namespace is honoured or the walk never ran.

   No case-only variant is listed: `Ocannl_Backend` is one variable with `OCANNL_BACKEND` on
   Windows and a different one everywhere else, so its classification is correct and different per
   platform, which is not something a golden can hold. The dashed spellings, which are unread on
   every platform, carry that leg instead. *)
let describe = function
  | Utils.Env_not_addressed -> "not addressed to OCANNL"
  | Utils.Env_reserved prefix -> "reserved namespace " ^ prefix
  | Utils.Env_config_key key -> "configuration key " ^ key
  | Utils.Env_unread_spelling key -> "unread spelling of " ^ key
  | Utils.Env_unknown_key key -> "addressed to OCANNL, unknown key " ^ key

let () =
  printf "\nEnvironment variable names, as classified by `Utils.classify_env_var`:\n";
  List.iter
    [
      "ocannl_backend";
      "OCANNL_BACKEND";
      "ocannl-backend";
      "OCANNL-BACKEND";
      "OCANNL_BACKEDN";
      "OCANNL_TOOL_SWEEP_STATE";
      "OCANNL_LOG_LEVEL";
      "OCANNL_LOG_LEVEL_ROW";
      "PATH";
    ]
    ~f:(fun name ->
      printf "  %-26s %s\n" ("\"" ^ name ^ "\"") (describe (Utils.classify_env_var name)))
