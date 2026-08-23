(* gh-ocannl-605: the spellings a configuration setting has outside an ocannl_config file.

   Two lists, and then a live check that they are the truth rather than a wish. The environment list
   is the shorter one on purpose, and since gh-ocannl-652 it holds ONE name: a dune rule that has to
   declare the ambient variables it is invalidated by must enumerate it by hand, and while it held
   four spellings per key the natural-looking all-dashed `ocannl-log-level` was not among them -- a
   dep on it declared a variable nothing reads while leaving the one OCANNL does read untracked.
   Dashes are idiomatic on the commandline and stay there, where they are also normalized: the
   prefix separator and the key's own separators dash independently, so every spelling one would
   guess is accepted.

   The live section is what makes this a test rather than a restatement of the source. The rule next
   to it sets a synthetic key in every environment spelling that was dropped and in the one that was
   kept, and a second key in the dropped spellings only. Synthetic keys, deliberately: nothing there
   is a real setting, so the library reads no differently for having been asked.

   The commandline lookups name REAL keys, because the unknown-argument warning at the foot of
   `arrayjit/lib/utils.ml` inspects every argument addressed to OCANNL. It used to do so with a
   parser of its own -- split on `=`, dash to underscore, look up -- which disagreed with the reader
   in both directions: `--ocannl-print-decimals-precision=7` passed validation and was then ignored,
   while `--ocannl_log_level_0` was applied and warned about, its separator not being an `=`. Both
   now go through `Utils.cmdline_var_prefixes`, so these two lines and the sibling
   `config_var_warnings` golden are the two halves of one claim. *)

open Base
open Stdio

let show_names title names =
  printf "%s\n" title;
  List.iter names ~f:(printf "  %s\n")

let () =
  show_names "Environment variable for `log_level`:" [ Utils.env_var_name "log_level" ];
  show_names "Environment variable for `profile`:" [ Utils.env_var_name "profile" ];
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
  (* Set as `OCANNL_DEMO_KEY`, `ocannl_demo_key`, `ocannl-demo_key`, `OCANNL-DEMO_KEY` and
     `ocannl-demo-key`: only the first is a spelling, and it is what the lookup must report --
     including over the lowercase one, which used to win (gh-ocannl-652). *)
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

   No case-only variant is listed: `ocannl_backend` is one variable with `OCANNL_BACKEND` on Windows
   and a different one everywhere else, so its classification is correct and different per platform,
   which is not something a golden can hold. The dashed spellings carry that leg instead: they are
   unread on every platform, punctuation being the one thing a case-insensitive environment does NOT
   fold -- which is a property of `Utils.same_env_name` rather than a wish, since the predicate it
   replaced answered "read" to every candidate on Windows (Codex P1 on PR #389). The case-only
   variants are claimed below, and pinned outright by the two-reading table after that -- which can
   hold what THIS table cannot because it names the sensitivity instead of inheriting it. *)
let describe = function
  | Utils.Env_not_addressed -> "not addressed to OCANNL"
  | Utils.Env_reserved prefix -> "reserved namespace " ^ prefix
  | Utils.Env_config_key key -> "configuration key " ^ key
  | Utils.Env_unread_spelling key -> "unread spelling of " ^ key
  | Utils.Env_unread_reserved prefix -> "unread casing in the " ^ prefix ^ " namespace"
  | Utils.Env_unknown_key key -> "addressed to OCANNL, unknown key " ^ key

let () =
  printf "\nEnvironment variable names, as classified by `Utils.classify_env_var`:\n";
  List.iter
    [
      "OCANNL_BACKEND";
      "ocannl-backend";
      "OCANNL-BACKEND";
      (* A key whose own separators are dashed, which is a spelling the COMMANDLINE reads -- so it
         is written here by someone carrying that habit over. Recognized as the key it names, and
         therefore fatal, rather than warned about as an unknown one (Codex P2 on PR #389). *)
      "OCANNL_PRINT-DECIMALS-PRECISION";
      "OCANNL_BACKEDN";
      "OCANNL_TOOL_SWEEP_STATE";
      "OCANNL_LOG_LEVEL";
      "OCANNL_LOG_LEVEL_ROW";
      "PATH";
    ] ~f:(fun name ->
      printf "  %-26s %s\n" ("\"" ^ name ^ "\"") (describe (Utils.classify_env_var name)))

(* The casing leg, as a CLAIM rather than as three more lines above, for the reason that list
   states: `ocannl_backend` and `OCANNL_BACKEND` are one variable on Windows and two here, so the
   classification is correctly different per platform and a golden cannot hold it. The equivalence
   holds everywhere -- a lowercase name under the prefix is read exactly where the environment is
   case-insensitive -- so that is what is pinned.

   All three families answer the same way since gh-ocannl-652, which is the point of the change:
   `ocannl_backend` is now no more read than `ocannl_tool_test_cap` is, where before it was read
   FIRST. The reserved half of the leg exists because waving reserved names through on the strength
   of their prefix alone suppressed the warning where the mistake looks most like a success:
   `ocannl_tool_test_cap=10` and `ocannl_log_level_row=9` are read by nobody -- the shell scripts
   and the ppx gates spell their names in uppercase -- while looking exactly like the settings they
   are not (Codex P2 on PR #371). *)
let read_under_the_prefix name =
  match Utils.classify_env_var name with
  | Utils.Env_reserved _ | Utils.Env_config_key _ -> Some true
  | Utils.Env_unread_reserved _ | Utils.Env_unread_spelling _ -> Some false
  | _ -> None

let () =
  printf "\n";
  List.iter [ "ocannl_backend"; "ocannl_log_level_row"; "ocannl_tool_test_cap" ] ~f:(fun name ->
      Verdict.p
        (Printf.sprintf "%S is read exactly where the environment is case-insensitive" name)
        (Option.equal Bool.equal (read_under_the_prefix name)
           (Some Utils.env_names_case_insensitive)))

(* BOTH platform readings, on every host (gh-ocannl-661).

   Everything above asks the classifier the platform's own question, which is the right question for
   a run and the wrong one for CI: `Utils.env_names_case_insensitive` is `Sys.win32`, so the
   case-insensitive branch was unreachable on Linux and macOS, and a change that collapsed it stayed
   green on the ordinary PR path until a scheduled Windows run went red days later. That is the
   history of this file -- the dashed spellings were classified as READ on Windows, and the golden
   that would have said so could not run anywhere it mattered.

   Naming the sensitivity makes both branches reachable everywhere, so this table is
   platform-invariant by construction: the same bytes on Windows and Linux, and the Windows column
   is checked by every PR. The list is chosen for the distinction it draws -- case-only variants,
   where the two columns MUST differ, against dashed ones, where they must NOT, punctuation being
   the thing a case-insensitive environment does not fold. *)
let () =
  printf "\nAs classified under each platform's environment semantics, on every host:\n";
  printf "  %-26s %-48s %s\n" "" "case-sensitive (POSIX)" "case-insensitive (Windows)";
  List.iter
    [
      "OCANNL_BACKEND";
      "ocannl_backend";
      "OCANNL-BACKEND";
      "ocannl-backend";
      "OCANNL_LOG_LEVEL_ROW";
      "ocannl_log_level_row";
      "OCANNL_TOOL_SWEEP_STATE";
      "ocannl_tool_sweep_state";
    ] ~f:(fun name ->
      printf "  %-26s %-48s %s\n"
        ("\"" ^ name ^ "\"")
        (describe (Utils.classify_env_var ~case_insensitive:false name))
        (describe (Utils.classify_env_var ~case_insensitive:true name)))

(* The regression itself, as a claim rather than only as a table someone must read: a dashed
   spelling is unread under BOTH readings. This is what went wrong in gh-ocannl-661, and it now
   fails on any host rather than on Windows alone. *)
let () =
  printf "\n";
  List.iter [ "OCANNL-BACKEND"; "ocannl-backend"; "OCANNL-LOG_LEVEL" ] ~f:(fun name ->
      Verdict.p_all (Printf.sprintf "%S is an unread spelling under both case sensitivities" name)
        [ false; true ] ~f:(fun case_insensitive ->
             match Utils.classify_env_var ~case_insensitive name with
             | Utils.Env_unread_spelling _ -> true
             | _ -> false))
