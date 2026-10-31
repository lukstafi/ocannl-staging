module Impl = Lowered_backend_missing.Missing (struct
  let name = "hip"
  let backend_facts = Ir.Backend_intf.hip_facts
end)

let host_capabilities () = []
