(** Positional arguments beside the [--ocannl_*] configuration flags (gh-ocannl-634).

    Every tool here takes its geometry positionally — [schedule_bench 512 20 256] — while the
    library's own settings arrive on the same commandline, as [--ocannl_*] flags. Splitting the two
    is a three-line idiom, and it had been written five times, twice wrongly: filtering out
    everything that starts with a [-] drops a negative extent along with the flags, and where there
    are several positionals it does not merely lose that value, it shifts every later argument one
    slot left. The bench then runs at a geometry nobody asked for and reports a plausible number for
    it, which is the failure mode worth spending a module on.

    So the predicate lives here once: an option is [--]-prefixed, or a [-] followed by a non-digit;
    a bare [-64] is a positional and reaches the range check below, which names it. A lone [--] ends
    the options, so an argument that must be taken literally has an escape (a [gpt2_generate] prompt
    beginning with a dash, say) — the standard convention, and the only way to pass such a string,
    since the filter is deliberately blind to what a tool expects in each slot: it cannot tell a
    flag from a prompt.

    Parsing and validation are one call, not two: an extent that arrives as [0] or [-64] is rejected
    where it is read, by name, rather than reaching an [Array.init] or a schedule somewhere below.
*)

open Base

type t = {
  tool : string;  (** Prefixes every diagnostic, so a failure names the tool that failed. *)
  positional : string list;
  options : string list;  (** Kept for [int]'s [?flag]: the [--name=value] arguments, in order. *)
}

(** An option is [--]-prefixed (the library's [--ocannl_*] settings, and any of a tool's own), or a
    [-] followed by a non-digit. Everything else — including [-64] and a bare [-] — is positional.
*)
let is_option s =
  String.is_prefix s ~prefix:"--"
  || (String.length s > 1 && Char.equal s.[0] '-' && not (Char.is_digit s.[1]))

(** [create tool] splits [Sys.get_argv ()] — or [?argv], which is how the test drives it — into
    positionals and options. [argv.(0)] is the program name and is neither. *)
let create ?argv tool =
  let argv = match argv with Some a -> a | None -> Sys.get_argv () in
  let rest = match Array.to_list argv with [] -> [] | _ :: rest -> rest in
  (* Before a lone [--] the predicate decides; after it, every argument is positional verbatim. *)
  let before, after =
    match List.split_while rest ~f:(fun s -> not (String.equal s "--")) with
    | before, _ :: after -> (before, after)
    | before, [] -> (before, [])
  in
  {
    tool;
    positional = List.filter before ~f:(Fn.non is_option) @ after;
    options = List.filter before ~f:is_option;
  }

let bad t fmt = Printf.ksprintf (fun msg -> invalid_arg (t.tool ^ ": " ^ msg)) fmt

(** The positional arguments, in order: for a tool whose arguments are words rather than extents (a
    mode, a subcommand). *)
let positional t = t.positional

(** [string t i ~default] is positional [i], or [default] when it was not given. *)
let string t i ~default = Option.value (List.nth t.positional i) ~default

let check t ~name ~least ~where v =
  if v >= least then v
  else
    bad t "%s must be %s, got %d%s" name
      (if least = 1 then "positive"
       else if least = 0 then "nonnegative"
       else Printf.sprintf "at least %d" least)
      v where

let parse t ~name ~least ~where s =
  match Option.try_with (fun () -> Int.of_string s) with
  | None -> bad t "%s must be an integer, got %S%s" name s where
  | Some v -> check t ~name ~least ~where v

(** [int t i ~name ~default] is positional [i] as an integer, validated on the spot: each of these
    arguments is an extent or a repeat count, so the domain is [>= least] — 1 by default, and
    [~least:0] for the counts whose zero is documented, such as a leg the tool may skip.

    [?flag] names a [--flag=value] spelling that takes precedence over the positional, for an
    argument a tool wants reachable without counting slots. The default is checked too, so a tool
    cannot quietly default outside its own domain. *)
let int ?(least = 1) ?flag t i ~name ~default =
  let by_flag =
    Option.bind flag ~f:(fun flag ->
        let prefix = "--" ^ flag ^ "=" in
        let where = Printf.sprintf " (from %s)" prefix in
        (* Last wins, as with the library's own settings: [--bm=16 --bm=64] means 64. *)
        List.rev t.options
        |> List.find_map ~f:(fun s ->
            Option.map (String.chop_prefix s ~prefix) ~f:(parse t ~name ~least ~where)))
  in
  match by_flag with
  | Some v -> v
  | None -> (
      match List.nth t.positional i with
      | Some s -> parse t ~name ~least ~where:"" s
      | None -> check t ~name ~least ~where:" (the tool's default)" default)
