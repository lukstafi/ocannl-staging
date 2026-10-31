module Missing : functor
  (_ : sig
     val name : string
     val backend_facts : Ir.Backend_intf.backend_facts
   end)
  -> Ir.Backend_impl.Lowered_backend
