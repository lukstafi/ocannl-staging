(* The stand-in for the HIP arm on a box without hipjit (gh-ocannl-657). *)

type bytes_buf =
  (int, Stdlib.Bigarray.int8_unsigned_elt, Stdlib.Bigarray.c_layout) Stdlib.Bigarray.Array1.t

let name = "hip"
let vendor_type = "__hip_fp8_e5m2"
let available = false
let unavailable () = failwith "fp8_soak: the hip arm needs the hipjit library"
let describe () : string = unavailable ()
let narrow_f32 ~base:_ ~count:_ (_ : bytes_buf) : unit = unavailable ()
let narrow_f64 ~base:_ ~count:_ ~lows:_ (_ : bytes_buf) : unit = unavailable ()
