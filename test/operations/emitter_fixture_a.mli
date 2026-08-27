(** Interfaces built to break the emitter derivation, rather than whatever the compiler libraries
    happen to export today (gh-ocannl-748).

    Every value here is a shape {!Emitter_frontier} has to answer for, and half of them are the
    NEAR MISSES: a document combinator, a sink that takes documents, a function of the same shape
    returning something else. Those decide the rule as much as the renderers do -- the frontier is
    matched by name behind any qualifier, so a rule that admits [combines_documents] admits every
    test calling something of that name.

    [emitter_frontier_cases] reads this library's own compiled interfaces and compares what it
    derives against what is declared here. *)

type ir
(** Stands for the code being rendered: a type this library defines, which is what tells a renderer
    from a combinator. *)

val renders_a_document : ir -> PPrint.document
(** The plain shape: given the code, hands back its text. *)

val renders_a_triple : name:string -> ir -> string list * PPrint.document * int
(** [C_syntax.compile_proc]'s shape: the document comes back inside a tuple. *)

val renders_through_an_option : ir -> PPrint.document option
(** [render_mma_fragment_scope]'s shape: through a type argument. *)

val writes_into_a_buffer : buf:Buffer.t -> ir -> unit
(** [Canonical_render.emit]'s shape: the text lands in the caller's buffer, and the label is what
    the scan taints. *)

type rendered = PPrint.document
(** A transparent alias of the document type. An interface records the path the declaration spells,
    not what it abbreviates, so a rule reading paths alone sees no document here. *)

type rendered_again = rendered
(** And an alias of the alias, since one abbreviation resolves nothing if the next does not. *)

type destination = Buffer.t
(** The same, for the buffer a serializer writes into. *)

type described = string
(** The control for all three: an alias that is not a document, so following abbreviations must not
    turn its function into a renderer. *)

val renders_through_an_alias : ir -> rendered
val renders_through_a_chain : ir -> rendered_again
val writes_into_an_aliased_buffer : buf:destination -> ir -> unit
val describes_through_an_alias : ir -> described
val combines_documents : int -> PPrint.document
(** [Doc_helpers.int]'s shape: a document out of a number. Given nothing of the library to render,
    so not on the frontier -- reported as a combinator instead of being dropped. *)

val joins_documents : PPrint.document -> PPrint.document -> PPrint.document
(** [(^^)]'s shape: documents in, document out, and no program anywhere. *)

val consumes_documents : ir -> (PPrint.document -> unit) option
(** [Utils.output_to_build_file]'s shape: it PRINTS documents its caller supplies. The document is
    in the result, but only as something the result accepts. *)

val describes_the_code : ir -> string
(** Given the code and hands back text that is not a document: not this scan's business, since
    nothing it can recognise pins it. *)
