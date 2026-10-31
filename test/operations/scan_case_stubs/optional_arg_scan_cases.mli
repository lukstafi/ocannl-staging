(** The stand-in names the [{ocaml|...|ocaml}] snippets of [optional_arg_scan_cases.ml] call, for
    [tools/typecheck-scan-cases.sh], which compiles this interface first and opens it at the top of
    every snippet -- so an arm may rebind any of these names ([module Base = struct ... end]), as it
    may over the real library. *)

val helper : ?scale:'a -> 'b -> unit
(** The function an option is forwarded to, and the effectful one a shadowing [ignore] calls. *)

val enable : 'a -> unit

(** [Base], as far as the arms reach into it: its standard [ignore]. *)
module Base : sig
  val ignore : 'a -> unit
end

(** A module whose [ignore] a local open brings in, effectful as far as the scanner knows. *)
module Effects : sig
  val ignore : 'a -> unit
end

val ( let* ) : 'a -> ('a -> 'b) -> 'b
val source : bool

(** A module of the arms' own whose [( ++ )] is not the DSL's. *)
module My : sig
  val ( ++ ) : 'a -> string -> 'a
end

val spec : string -> string
(** A function applied to an einsum-shaped string outside a spec position. *)

val stride : int
(** The legacy convolution spec's coefficient, where an arm binds [use_padding] but not [stride],
    and the padding flag a spec without [=] or [<] makes the [%op] expansion read. *)

val use_padding : bool

(* The project-syntax arms ([let%op], [[%op ...]], [let%cd]) are checked as top-level items of
   [lib/nn_blocks.ml], the module whose [%op] functions the scan's optional arguments come from,
   under its stanza's preprocessors and libraries; the prelude is that module's own opens. *)
[@@@owning_module "lib/nn_blocks.ml", "open! Base\nopen Ocannl_tensor.Operation.DSL_modules"]

(* A finding: OCaml forbids a second top-level module of one name ("Multiple definition of the
   module name B"), so the replacement of [B] these two arms pin -- which leaves [A]'s alias or
   include the only exporter of the optional function -- is a parse shape no source can contain. The
   reader's handling of it stays pinned as a defence until a compilable spelling of what the arms
   mean is settled. *)
[@@@known_rejection
"an optional function exported through a module alias is inventoried", "Multiple definition"]

[@@@known_rejection
"an optional function exported through a module include is inventoried", "Multiple definition"]
