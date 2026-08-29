(* The second half of the fixture, and the half that decides why the derivation reads COMPILED
   interfaces rather than sources: this module has no .mli and writes no return annotation, so
   nothing in its text says it produces a document. That is [C_syntax.compile_proc]'s situation
   exactly, and it is the member three review rounds of gh-ocannl-712 had to be told about.

   A second module also makes this library WRAPPED, so its interface carries the module aliases the
   derivation takes its module list from. *)

let renders_without_an_annotation code = ([ "kparam" ], Emitter_fixture_a.renders_a_document code, 0)
let names_the_routine code = Emitter_fixture_a.describes_the_code code

(* The control for how transparent aliases are keyed. [rendered] is [Emitter_fixture_a]'s name for
   the document type and this module's name for an integer, which is the ordinary situation -- [t]
   is declared in every module of every library. An alias table keyed by the bare name would read
   this as a renderer; keyed by the module path it declares in, it reads it as what it is. *)
type rendered = int

let counts_the_code (code : Emitter_fixture_a.ir) : rendered =
  String.length (Emitter_fixture_a.describes_the_code code)
