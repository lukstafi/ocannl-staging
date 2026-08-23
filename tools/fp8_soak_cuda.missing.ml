(* The stand-in for the CUDA arm on a box without cudajit: the soak still builds and still runs its
   other arms, and asking for this one says so rather than failing to link (gh-ocannl-657). *)

type bytes_buf = (int, Stdlib.Bigarray.int8_unsigned_elt, Stdlib.Bigarray.c_layout) Stdlib.Bigarray.Array1.t

let name = "cuda"
let vendor_type = "__nv_fp8_e5m2"
let available = false
let unavailable () = failwith "fp8_soak: the cuda arm needs the cudajit library"
let describe () : string = unavailable ()
let narrow_f32 ~base:_ ~count:_ (_ : bytes_buf) : unit = unavailable ()
let narrow_f64 ~base:_ ~count:_ ~lows:_ (_ : bytes_buf) : unit = unavailable ()
