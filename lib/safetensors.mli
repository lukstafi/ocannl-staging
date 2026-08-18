(** {1 Reader for the safetensors tensor-serialization format.}

    The format (https://github.com/huggingface/safetensors): an unsigned little-endian 64-bit header
    length [n], followed by [n] bytes of UTF-8 JSON mapping tensor names to
    [{"dtype", "shape", "data_offsets"}] (plus an optional ["__metadata__"] object), followed by the
    byte buffer holding all tensor payloads in little-endian order. [data_offsets] are relative to
    the byte buffer, i.e. to file position [8 + n].

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
(** [read path] parses the header of the safetensors file at [path]. Tensor payloads are read lazily
    by the accessors below, so this is cheap even for multi-gigabyte files.

    The returned value {e owns} an open descriptor on the file, which the payload accessors address
    the file through: they never re-open [path], so a concurrent replacement of the file cannot pair
    this header's offsets with another file's bytes (gh-ocannl-587). The descriptor is closed by
    {!close}, or by the garbage collector once the value is unreachable; payloads already handed out
    stay valid either way (a mapping outlives its descriptor). The accessors share that one
    descriptor's file position, so a [t] must not be read from several threads at once.

    Raises [Failure] on malformed headers and header/file-size inconsistencies, including when the
    payload ranges do not tile the byte buffer exactly (overlapping or non-contiguous
    [data_offsets], or trailing uncovered bytes). *)

val close : t -> unit
(** [close t] releases [t]'s descriptor. Payload accessors raise [Failure] afterwards; ndarrays
    already returned by them remain valid, mapped ones included. Idempotent. *)

val path : t -> string
val names : t -> string list
val info : t -> string -> tensor_info option

val metadata : t -> (string * string) list
(** The ["__metadata__"] string-to-string map, if present. *)

val prec_of_dtype : string -> Ir.Ops.prec option
(** The OCANNL precision naming the same bit layout as a safetensors dtype, if there is one:
    F64/F32/F16/BF16/F8_E5M2, I64/I32, U64/U32/U16/U8, and BOOL (as [byte], i.e. the floats 0. and
    1.). [None] for the dtypes no OCANNL precision {e names}: I8 and I16 are signed where [byte] and
    [uint16] are not, and F8_E4M3 is a different 8-bit float from OCANNL's e5m2 [fp8]. *)

val to_ndarray : ?prec:Ir.Ops.prec -> t -> string -> Ir.Ndarray.t
(** [to_ndarray t name] is the named tensor's payload as an ndarray of the tensor's shape (a scalar
    shape becomes a 1-element 1-D array) and, by default, of the payload's own precision -- see
    {!prec_of_dtype}. Suitable for tensor [init_data] (e.g. [TDSL.wrap] / [TDSL.rebatch]).

    The payload is {e mapped}, not copied (gh-ocannl-587): the file's own bytes back the array,
    through a private, copy-on-write {!Unix.map_file} region read lazily by page, so writes to it
    never reach the file. Two cases fall back to decoding into a fresh host buffer: a big-endian
    host (the format is little-endian), and a payload whose file offset is not a multiple of its
    element size, which the format does not guarantee. Both produce the same values;
    {!ingestion_counts} is how to tell which ran.

    [?prec] converts the payload to that precision instead, through {!Ir.Ndarray.convert} -- a copy,
    unless it is already the payload's precision.

    Raises [Failure] if the tensor is missing, if its dtype has no matching OCANNL precision, or if
    [t] is closed. *)

val to_float32 : t -> string -> (float, Bigarray.float32_elt, Bigarray.c_layout) Bigarray.Genarray.t
(** [to_float32 t name] is {!to_ndarray} for an "F32" tensor, as a bare float32 Genarray. Same
    mapped-or-decoded ingestion, so the result is normally a copy-on-write view of the file rather
    than a fresh array.

    Raises [Failure] if the tensor is missing or its dtype is not "F32"; use [to_ndarray ~prec] to
    convert another dtype to floats. *)

val ingestion_counts : unit -> int * int
(** [(mapped, copied)] payload counts since the start of the process: how many payloads the
    accessors wrapped as file mappings, and how many they decoded into fresh host buffers. The two
    paths agree on values by construction, so this is the only way to observe which one ran. *)
