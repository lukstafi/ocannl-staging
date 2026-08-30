(** Relating repository-scanner refusal diagnostics to permanent control goldens.

    A scanner refusal is mechanically visible when its source hands a string literal, directly or as
    a [Printf.sprintf] format, to [Verdict.fail], to the local [fail] alias scanners use, or to one
    of Verdict's claim forms: a false claim emits its label as the refusal. The dynamic values a
    helper returns are deliberately outside this reader: there is no diagnostic string constant in
    that scanner source to relate. The live check names those limits in its report rather than
    pretending to infer a value through arbitrary OCaml.

    Formats are reduced to a stable literal fragment. Code-shaped tokens (underscores or qualified
    names) win; otherwise the longest three-word run does. Values substituted by the failing run
    decide nothing. Thus an [(include_subdirs %s)] refusal contributes [include_subdirs], while a
    malformed marker refusal contributes a phrase naming that condition rather than a generic word.
    A control golden is normalized the same way before matching. *)

open Base
open Ppxlib.Parsetree
module Ast_traverse = Ppxlib.Ast_traverse
module Read = Config_key_scan

type diagnostic = { line : int; fragment : string; format : string }

let normalize text =
  String.split_on_chars text ~on:[ ' '; '\t'; '\r'; '\n' ]
  |> List.filter ~f:(Fn.non String.is_empty)
  |> String.concat ~sep:" "

let trim_static_run text =
  String.strip text ~drop:(fun character ->
      Char.is_whitespace character
      || List.mem [ ':'; ';'; ','; '.'; '-'; '`'; '('; ')'; '['; ']' ] character ~equal:Char.equal)

let directive_stop format start =
  let length = String.length format in
  let rec modifiers index =
    if index >= length then index
    else
      match format.[index] with
      | '-' | '0' | '+' | ' ' | '#' | '.' | '*' -> modifiers (index + 1)
      | character when Char.is_digit character -> modifiers (index + 1)
      | _ -> index
  in
  let index = modifiers (start + 1) in
  if index >= length then length
  else
    match format.[index] with
    | ('l' | 'L' | 'n') when index + 1 < length -> index + 2
    | _ -> index + 1

let static_runs format =
  let length = String.length format in
  let buffer = Buffer.create length in
  let found = ref [] in
  let flush () =
    let run = Buffer.contents buffer |> normalize |> trim_static_run in
    Buffer.clear buffer;
    if String.count run ~f:Char.is_alpha >= 8 then found := run :: !found
  in
  let rec loop index =
    if index >= length then flush ()
    else if not (Char.equal format.[index] '%') then (
      Buffer.add_char buffer format.[index];
      loop (index + 1))
    else if index + 1 < length && Char.equal format.[index + 1] '%' then (
      Buffer.add_char buffer '%';
      loop (index + 2))
    else (
      flush ();
      loop (directive_stop format index))
  in
  loop 0;
  List.rev !found

let words text =
  String.split text ~on:' '
  |> List.map
       ~f:
         (String.strip ~drop:(fun c ->
              List.mem [ ':'; ';'; ','; '.'; '-'; '`'; '('; ')'; '['; ']' ] c ~equal:Char.equal))
  |> List.filter ~f:(Fn.non String.is_empty)

let trigrams words =
  let rec loop found = function
    | a :: (b :: c :: _ as tail) -> loop (String.concat ~sep:" " [ a; b; c ] :: found) tail
    | _ -> List.rev found
  in
  loop [] words

let fragment_of_format format =
  Option.bind
    (List.hd (static_runs format))
    ~f:(fun run ->
      let words = words run in
      let code_tokens =
        List.filter words ~f:(fun token ->
            String.length token >= 4
            && String.exists token ~f:(fun c -> Char.equal c '_' || Char.equal c '.'))
      in
      match
        List.max_elt code_tokens ~compare:(fun a b ->
            Int.compare (String.length a) (String.length b))
      with
      | Some token -> Some token
      | None -> (
          match
            List.max_elt (trigrams words) ~compare:(fun a b ->
                Int.compare (String.length a) (String.length b))
          with
          | Some phrase -> Some phrase
          | None when String.length run >= 12 -> Some run
          | None -> None))

let last_name expression = Option.bind (Read.longident_of expression) ~f:List.last

let refusal_callees =
  [ "fail"; "p"; "p_all"; "p_none"; "p_exists"; "p_empty"; "claim"; "claimf"; "pass_fail" ]

let is_refusal expression =
  Option.value_map (last_name expression) ~default:false ~f:(fun name ->
      List.mem refusal_callees name ~equal:String.equal)

let rec format_of expression =
  match Read.string_literal expression with
  | Some format -> Some format
  | None -> (
      match expression.pexp_desc with
      | Pexp_apply (operator, [ (Nolabel, _left); (Nolabel, right) ])
        when Option.value_map (last_name operator) ~default:false ~f:(String.equal "@@") ->
          format_of right
      | Pexp_apply (callee, arguments)
        when Option.value_map (Read.longident_of callee) ~default:false ~f:(function
               | [ "Printf"; ("sprintf" | "ksprintf") ] -> true
               | _ -> false) ->
          List.find_map arguments ~f:(fun (label, argument) ->
              match label with
              | Nolabel -> Read.string_literal argument
              | Labelled _ | Optional _ -> None)
      | _ -> None)

let diagnostic_argument expression =
  match expression.pexp_desc with
  | Pexp_apply (callee, arguments) when is_refusal callee ->
      List.find_map arguments ~f:(fun (label, argument) ->
          match label with Nolabel -> Some argument | Labelled _ | Optional _ -> None)
  | Pexp_apply (operator, [ (Nolabel, callee); (Nolabel, argument) ])
    when Option.value_map (last_name operator) ~default:false ~f:(String.equal "@@")
         && is_refusal callee ->
      Some argument
  | _ -> None

let diagnostics content =
  let found = ref [] in
  let iterator =
    object
      inherit Ast_traverse.iter as super

      (* A documentation comment is an attribute carrying a string. It is prose about a refusal,
         never the application that emits one. *)
      method! attribute _ = ()

      method! expression expression =
        (match Option.bind (diagnostic_argument expression) ~f:format_of with
        | Some format -> (
            match fragment_of_format format with
            | Some fragment ->
                found :=
                  { line = expression.pexp_loc.loc_start.pos_lnum; fragment; format } :: !found
            | None -> ())
        | None -> ());
        super#expression expression
    end
  in
  iterator#structure (Read.structure_of content);
  List.rev !found

let covered ~control_text diagnostic =
  String.is_substring (normalize control_text) ~substring:diagnostic.fragment

let orphans ~control_text diagnostics =
  List.filter diagnostics ~f:(fun diagnostic -> not (covered ~control_text diagnostic))
