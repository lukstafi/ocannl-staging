module Impl = Lowered_backend_missing.Missing (struct
  let name = "cuda"
  let backend_facts = Ir.Backend_intf.cuda_facts
end)

let host_capabilities () = []
