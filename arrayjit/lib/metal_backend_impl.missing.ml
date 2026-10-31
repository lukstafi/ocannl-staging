module Impl = Lowered_backend_missing.Missing (struct
  let name = "metal"
  let backend_facts = Ir.Backend_intf.metal_facts
end)

let host_capabilities () = []
