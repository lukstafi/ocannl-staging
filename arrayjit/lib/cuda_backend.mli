(** [convert_precision ~from ~to_] returns the [(prefix, postfix)] C/CUDA strings that wrap an
    expression to convert it from precision [from] to precision [to_]. Exposed (independently of a
    CUDA device) so the Uint4x32 bit-spreading behaviour can be unit-tested; see
    test/operations/test_cuda_prng_convert_precision.ml. *)
val convert_precision : from:Ir.Ops.prec -> to_:Ir.Ops.prec -> string * string

module Fresh () : Ir.Backend_impl.Lowered_backend
