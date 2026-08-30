(** The source reader and relationship behind gh-ocannl-800, on diagnostics and goldens the
    repository does not have to contain. *)

open Base
open Stdio
module Scan = Test_utils.Refusal_control_scan

let () =
  let source =
    {ocaml|
let fail = Verdict.fail

let direct () =
  Verdict.fail "a direct scanner refusal has a permanent diagnostic"

let formatted name =
  fail
    (Printf.sprintf
       "%s: the formatted refusal names the `formatted_relationship` its control exercises"
       name)

let applied name =
  fail @@ Printf.sprintf "%s is stale -- drop it from the exemption list" name

let quantified xs =
  Verdict.p_all "every refusal control remains related to its diagnostic" xs ~f:Fn.id

let prose = "Verdict.fail \"quoted code is not an application\""
let dynamic reason = Verdict.fail reason
|ocaml}
  in
  let diagnostics = Scan.diagnostics source in
  let fragments = List.map diagnostics ~f:(fun diagnostic -> diagnostic.Scan.fragment) in
  Verdict.p_all ~min:4 "direct, formatted, `@@`, and quantified refusal formats are extracted"
    diagnostics ~f:(fun diagnostic -> not (String.is_empty diagnostic.Scan.fragment));
  Verdict.p "comments, quoted code, and a dynamic value contribute no diagnostic string constant"
    (List.length diagnostics = 4);
  Verdict.p "Printf substitutions are holes and a stable literal fragment becomes the fragment"
    (List.mem fragments "formatted_relationship" ~equal:String.equal);
  let one = List.hd_exn diagnostics in
  Verdict.p "an absent fragment is an orphan"
    (List.length (Scan.orphans ~control_text:"" [ one ]) = 1);
  Verdict.p "the same fragment in a control golden covers it"
    (List.is_empty (Scan.orphans ~control_text:one.fragment [ one ]));
  Verdict.p "control matching normalizes line wrapping and repeated whitespace"
    (List.is_empty
       (Scan.orphans ~control_text:"a direct scanner refusal\n    has a permanent diagnostic"
          [ one ]));
  let arguments = Array.to_list (Array.subo Stdlib.Sys.argv ~pos:1) in
  let rec pairs = function
    | source :: control :: rest -> (source, control) :: pairs rest
    | [] -> []
    | [ dangling ] ->
        Verdict.fail (Printf.sprintf "scanner source %s has no assigned control golden" dangling);
        []
  in
  printf "\nScanner refusal formats and the permanent control suite assigned to their source:\n";
  pairs arguments
  |> List.iter ~f:(fun (source, control) ->
      let control_text = In_channel.read_all control in
      let has_gated_result =
        String.split_lines control_text
        |> List.exists ~f:(fun line ->
            String.is_prefix line ~prefix:"ok:" || String.is_suffix line ~suffix:": true")
      in
      Scan.diagnostics (In_channel.read_all source)
      |> List.iter ~f:(fun diagnostic ->
          Verdict.p
            (Printf.sprintf "%s: `%s` is catalogued beside %s" source diagnostic.Scan.fragment
               control)
            has_gated_result))
