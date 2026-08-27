(* The stand-in for the HIP arm on a box without hipjit (gh-ocannl-657); see the CUDA stub beside it
   for why it holds no vendor knowledge of its own (gh-ocannl-758). *)

type bytes_buf =
  (int, Stdlib.Bigarray.int8_unsigned_elt, Stdlib.Bigarray.c_layout) Stdlib.Bigarray.Array1.t

let last_compiled = "never: this box builds the stub"
let built = false
let unavailable () = failwith "fp8_soak: this arm is not built on this box"
let set_arch_policy (_ : [ `Device | `Backend ]) = ()
let device_count () : (int, string) Result.t = unavailable ()
let device_report () : (string * string) list = unavailable ()
let compile_options () : string list = unavailable ()
let kernel_macros () : (string * int) list = unavailable ()

let narrow_f32 ~spelling:(_ : [ `Raw | `Guarded ]) ~base:_ ~count:_ (_ : bytes_buf) : unit =
  unavailable ()

let narrow_f64 ~spelling:(_ : [ `Raw | `Guarded ]) ~base:_ ~count:_ ~lows:_ (_ : bytes_buf) : unit =
  unavailable ()
