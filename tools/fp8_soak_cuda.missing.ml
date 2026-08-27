(* The stand-in for the CUDA arm on a box without cudajit: the soak still builds and still runs its
   other arms, and asking for this one says so rather than failing to link (gh-ocannl-657).

   It carries no vendor knowledge to keep in step with the real arm -- the name, the library, the
   spellings and every message about them are fp8_soak.ml's [cuda_vendor] record, which is compiled
   on every box (gh-ocannl-758). [built = false] is the whole of what this file tells the selection,
   and nothing below it is ever reached: an unbuilt arm is refused before it is asked anything. *)

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
