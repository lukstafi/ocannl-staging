(* The implementations are beside the point -- what this fixture library exists for is its
   INTERFACES, which emitter_frontier_cases reads back through the compiler. See the .mli. *)

type ir = string

let renders_a_document code = PPrint.string code
let renders_a_triple ~name code = ([ name ], PPrint.string code, 0)
let renders_through_an_option code = Some (PPrint.string code)
let writes_into_a_buffer ~buf code = Buffer.add_string buf code
type rendered = PPrint.document
type rendered_again = rendered
type destination = Buffer.t
type described = string

let renders_through_an_alias code = PPrint.string code
let renders_through_a_chain code = PPrint.string code
let writes_into_an_aliased_buffer ~buf code = Buffer.add_string buf code
let describes_through_an_alias code = code
let writes_into_an_unlabelled_buffer code buf = Buffer.add_string buf code
module type RENDERER = sig
  val renders_under_a_module_type : ir -> PPrint.document
end

module Named = struct
  let renders_under_a_module_type code = PPrint.string code
end

module type OUTER = sig
  module type NESTED = sig
    val renders_from_a_nested_module_type : ir -> PPrint.document
  end
end

module Outer = struct
  module type NESTED = sig
    val renders_from_a_nested_module_type : ir -> PPrint.document
  end
end

module Uses_nested = struct
  let renders_from_a_nested_module_type code = PPrint.string code
end

let combines_documents lanes = PPrint.string (Int.to_string lanes)
let joins_documents left right = PPrint.(left ^^ right)
let consumes_documents _code = None
let describes_the_code code = code
