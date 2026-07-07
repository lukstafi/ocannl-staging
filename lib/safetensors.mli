(** {1 Reader for the safetensors tensor-serialization format.}

    The format (https://github.com/huggingface/safetensors): an unsigned little-endian 64-bit
    header length [n], followed by [n] bytes of UTF-8 JSON mapping tensor names to
    [{"dtype", "shape", "data_offsets"}] (plus an optional ["__metadata__"] object), followed by
    the byte buffer holding all tensor payloads in little-endian order. [data_offsets] are
    relative to the byte buffer, i.e. to file position [8 + n].

    This reader targets loading pretrained model checkpoints (e.g. HuggingFace GPT-2's
    [model.safetensors]); only reading is supported, and only little-endian hosts. *)

type tensor_info = {
  dtype : string;  (** As in the file, e.g. "F32", "F16", "BF16", "I64". *)
  shape : int list;  (** Row-major dimensions; [[]] for a scalar (1 element). *)
  offset : int;  (** Start of the payload relative to the byte-buffer start. *)
  nbytes : int;  (** Payload length in bytes. *)
}

type t

val read : string -> t
(** [read path] parses the header of the safetensors file at [path]. Tensor payloads are read
    lazily by the accessors below, so this is cheap even for multi-gigabyte files.

    Raises [Failure] on malformed headers and header/file-size inconsistencies. *)

val path : t -> string
val names : t -> string list
val info : t -> string -> tensor_info option

val metadata : t -> (string * string) list
(** The ["__metadata__"] string-to-string map, if present. *)

val to_float32 :
  t -> string -> (float, Bigarray.float32_elt, Bigarray.c_layout) Bigarray.Genarray.t
(** [to_float32 t name] reads the named tensor's payload from the file into a fresh float32
    Genarray of the tensor's shape (a scalar shape becomes a 1-element 1-D array).

    Raises [Failure] if the tensor is missing or its dtype is not "F32". *)

val to_ndarray : t -> string -> Ir.Ndarray.t
(** [to_ndarray t name] is {!to_float32} wrapped as an OCANNL single-precision ndarray, suitable
    for tensor [init_data] (e.g. [TDSL.wrap] / [TDSL.rebatch]). *)
