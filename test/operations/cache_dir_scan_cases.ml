(** How the directory reader behind [cache_dir_ignores] resolves a name, exercised on input built to
    break it rather than on whatever the repository happens to spell today.

    [Cache_dir_ignores] scans every source in the tree, so what it exercises is the set of spellings
    that are currently in use — and the reader deliberately covers more than that, because the point
    of the prefix rule is that a tuning test written tomorrow is ignored the day it is written. An
    arm covering a spelling no source uses yet is invisible to that census: it can stop working, or
    stop compiling, without a single check going red.

    That is not hypothetical for the one arm below that resolves a module alias bound in EXPRESSION
    position. The two ways the reader can be wrong are both quiet: a use it fails to recognise
    shrinks the census rather than failing it, and an alias it fails to resolve turns a cache write
    into an ordinary [~dir] it never looks at.

    So each spelling is pinned here on a snippet of its own, and the resolutions the census cannot
    reach — the empty string, a forwarded parameter, an expression — are pinned beside the ones it
    can. *)

open Base
open Stdio
module Scan = Test_utils.Cache_dir_scan

(* Failures go through [Verdict], so that a regression exits nonzero instead of being `dune
   promote`d into the golden as the expected output (gh-ocannl-601). *)
let fail fmt = Printf.ksprintf Verdict.fail fmt

(* Each case is a source and the uses the reader should find in it, rendered as the argument was
   spelled followed by what it resolves to. *)
let cases =
  [
    ( "a literal at the call site",
      {ocaml|let () = Autotune.tune ~cache_dir:"autotune_cache_x" f|ocaml},
      [ "~cache_dir names autotune_cache_x" ] );
    ( "a literal reached through a binding",
      {ocaml|let go () =
  let cache_dir = "autotune_cache_y" in
  Autotune.tune ~cache_dir f|ocaml},
      [ "~cache_dir names autotune_cache_y" ] );
    ( "the empty string turns the cache off",
      {ocaml|let () = Autotune.tune ~cache_dir:"" f|ocaml},
      [ "~cache_dir disables the cache" ] );
    ( "a parameter is forwarded, and named at its own call sites",
      {ocaml|let run ~cache_dir () = Autotune.tune ~cache_dir f|ocaml},
      [ "~cache_dir forwards the parameter cache_dir" ] );
    ( "anything else is reported rather than assumed harmless",
      {ocaml|let () = Autotune.tune ~cache_dir:(prefix ^ suffix) f|ocaml},
      [ "~cache_dir names an expression" ] );
    ( "an unresolved name is reported by name",
      {ocaml|let () = Autotune.tune ~cache_dir:elsewhere f|ocaml},
      [ "~cache_dir names `elsewhere`" ] );
    (* The direct-store spelling, whose `~dir` is told from every other `~dir` in the repository
       only by the module it is called through. *)
    ( "a direct store through a structure-level alias",
      {ocaml|module SC = Ir.Schedule_cache
let () = SC.store ~dir:"autotune_cache_z" key value|ocaml},
      [ "~dir names autotune_cache_z" ] );
    ( "a direct store through the qualified path",
      {ocaml|let () = Ir.Schedule_cache.store ~dir:"autotune_cache_q" key value|ocaml},
      [ "~dir names autotune_cache_q" ] );
    (* The arm the census cannot reach: no source in the tree binds the module this way today. It
       is also the arm whose spelling the compiler moved under it -- 5.5 represents
       `let module M = … in …` as a structure item inside the expression rather than as
       `Pexp_letmodule` -- which is why the reader works on ppxlib's parse tree, where the
       construct has one spelling on every compiler the opam files admit. *)
    ( "a direct store through an alias bound in expression position",
      {ocaml|let go () =
  let module Cache = Ir.Schedule_cache in
  Cache.store ~dir:"autotune_cache_e" key value|ocaml},
      [ "~dir names autotune_cache_e" ] );
    ( "an alias of an alias is an alias",
      {ocaml|module SC = Ir.Schedule_cache
module Cache = SC
let () = Cache.store ~dir:"autotune_cache_c" key value|ocaml},
      [ "~dir names autotune_cache_c" ] );
    (* Not every `~dir` is a cache write: inside `schedule_cache.ml` itself the directory is a
       parameter, named by whoever called in. *)
    ( "a bare store is not a call into the cache module",
      {ocaml|let () = store ~dir:"scratch" key value|ocaml},
      [] );
    ( "nor is a same-named operation on another module",
      {ocaml|let () = Other.store ~dir:"scratch" key value|ocaml},
      [] );
    (* `Schedule_cache.ensure_dir` walks the path it is handed, so a name is not a directory name
       merely by starting with the prefix. The reader reports it; the check applies the rule. *)
    ( "a traversal keeps its separators, for the prefix rule to reject",
      {ocaml|let () = Autotune.tune ~cache_dir:"autotune_cache/../leaked_cache" f|ocaml},
      [ "~cache_dir names autotune_cache/../leaked_cache" ] );
  ]

(* The other half of what the check reads: the built-in default, which no source names and which is
   therefore read out of the library that defines it. Its second read asks merely whether the key
   was set and defaults to the empty string, so an empty default is not a directory. *)
let default_cases =
  [
    ( "the default a search falls back to",
      {ocaml|let d = get ~arg_name:"autotune_cache_dir" ~default:"autotune_cache"|ocaml},
      [ "autotune_cache" ] );
    ( "an empty default names no directory",
      {ocaml|let set = get ~arg_name:"autotune_cache_dir" ~default:""|ocaml},
      [] );
    ( "another key's default is not this one's",
      {ocaml|let d = get ~arg_name:"backend" ~default:"autotune_cache"|ocaml},
      [] );
  ]

let render uses = String.concat ~sep:"; " uses

let () =
  List.iter cases ~f:(fun (name, source, expected) ->
      let found =
        try
          List.map (Scan.read source).Scan.uses ~f:(fun use ->
              use.Scan.spelling ^ " " ^ Scan.describe use.Scan.resolution)
        with _ ->
          fail "use -- %s: the snippet does not parse" name;
          []
      in
      if List.equal String.equal found expected then printf "ok: use -- %s\n" name
      else fail "use -- %s: expected [%s], found [%s]" name (render expected) (render found));
  List.iter default_cases ~f:(fun (name, source, expected) ->
      let found = (Scan.read source).Scan.builtin_defaults in
      if List.equal String.equal found expected then printf "ok: built-in default -- %s\n" name
      else
        fail "built-in default -- %s: expected [%s], found [%s]" name (render expected)
          (render found))
