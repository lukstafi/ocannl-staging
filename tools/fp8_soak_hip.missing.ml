(* The stand-in for the HIP arm on a box without hipjit (gh-ocannl-657). *)

type bytes_buf =
  (int, Stdlib.Bigarray.int8_unsigned_elt, Stdlib.Bigarray.c_layout) Stdlib.Bigarray.Array1.t

let name = "hip"
let vendor_type = "__hip_fp8_e5m2"
type arch_policy = [ `Device | `Backend ]

let set_arch_policy (_ : arch_policy) = ()

(* The real arm's spellings, so that a run on a box without the library reports the same menu rather
   than an empty one -- every entry point below fails with the same reason anyway. *)
let spellings () = [ `Guarded; `Raw ]

let spelling_label (_ : [ `Raw | `Guarded ]) = "unavailable"
let probe () = Error "not built: the hip arm needs the hipjit library"
let unavailable () = failwith "fp8_soak: the hip arm needs the hipjit library"
let describe () : string = unavailable ()
let conversion_path () : string = unavailable ()
let narrow_f32 ~spelling:_ ~base:_ ~count:_ (_ : bytes_buf) : unit = unavailable ()
let narrow_f64 ~spelling:_ ~base:_ ~count:_ ~lows:_ (_ : bytes_buf) : unit = unavailable ()
