open Base
open Stdio

type tensor_info = { dtype : string; shape : int list; offset : int; nbytes : int }

type t = {
  path : string;  (** For error messages only: the payloads are addressed through [ic]. *)
  ic : Stdlib.in_channel;
      (** Owned, and kept open for the lifetime of [t]: the payloads are mapped from the descriptor
          the header was read through, not from a fresh open of [path] (gh-ocannl-587). A concurrent
          replacement of the file would otherwise pair one file's metadata with another's bytes. *)
  mutable closed : bool;
  buffer_start : int;  (** File position where the byte buffer (data section) starts. *)
  tensors : (string, tensor_info) Hashtbl.t;
  order : string list;  (** Names in header order. *)
  metadata : (string * string) list;
}

let path t = t.path
let names t = t.order
let info t name = Hashtbl.find t.tensors name
let metadata t = t.metadata

let close t =
  if not t.closed then begin
    t.closed <- true;
    Stdlib.close_in_noerr t.ic
  end

let checked_ic t what =
  if t.closed then failwith [%string "Safetensors.%{what} %{t.path}: the file has been closed"];
  t.ic

let dtype_size = function
  | "F64" | "I64" | "U64" -> 8
  | "F32" | "I32" | "U32" -> 4
  | "F16" | "BF16" | "I16" | "U16" -> 2
  | "I8" | "U8" | "BOOL" | "F8_E4M3" | "F8_E5M2" -> 1
  | dtype -> failwith [%string "Safetensors: unknown dtype %{dtype}"]

let fail_json path what json =
  failwith
    [%string "Safetensors.read %{path}: expected %{what}, got: %{Yojson.Safe.to_string json}"]

let read path =
  let ic = Stdlib.open_in_bin path in
  let parse () =
    let file_len = In_channel.length ic |> Int64.to_int_exn in
    if file_len < 8 then failwith [%string "Safetensors.read %{path}: file too short"];
    let len_bytes = Bytes.create 8 in
    In_channel.really_input_exn ic ~buf:len_bytes ~pos:0 ~len:8;
    let header_len = Stdlib.Bytes.get_int64_le len_bytes 0 |> Int64.to_int_exn in
    if header_len < 2 || header_len > file_len - 8 then
      failwith [%string "Safetensors.read %{path}: invalid header length %{header_len#Int}"];
    let header_bytes = Bytes.create header_len in
    In_channel.really_input_exn ic ~buf:header_bytes ~pos:0 ~len:header_len;
    let buffer_start = 8 + header_len in
    let buffer_len = file_len - buffer_start in
    let json = Yojson.Safe.from_string (Bytes.to_string header_bytes) in
    let entries =
      match json with `Assoc entries -> entries | _ -> fail_json path "a JSON object" json
    in
    let tensors = Hashtbl.create (module String) in
    let order = ref [] in
    let metadata = ref [] in
    List.iter entries ~f:(fun (name, entry) ->
        if String.equal name "__metadata__" then
          match entry with
          | `Assoc kvs ->
              metadata :=
                List.map kvs ~f:(function
                  | key, `String value -> (key, value)
                  | _, json -> fail_json path "a string metadata value" json)
          | _ -> fail_json path "a metadata object" entry
        else begin
          let member key =
            match entry with
            | `Assoc kvs -> List.Assoc.find kvs ~equal:String.equal key
            | _ -> fail_json path "a tensor-info object" entry
          in
          let dtype =
            match member "dtype" with
            | Some (`String s) -> s
            | _ -> fail_json path [%string "a dtype string for %{name}"] entry
          in
          let shape =
            match member "shape" with
            | Some (`List dims) ->
                List.map dims ~f:(function
                  | `Int d when d >= 0 -> d
                  | json -> fail_json path [%string "a dimension for %{name}"] json)
            | _ -> fail_json path [%string "a shape list for %{name}"] entry
          in
          let start, stop =
            match member "data_offsets" with
            | Some (`List [ `Int start; `Int stop ]) -> (start, stop)
            | _ -> fail_json path [%string "data_offsets for %{name}"] entry
          in
          let numel = List.fold shape ~init:1 ~f:( * ) in
          let nbytes = stop - start in
          if nbytes <> numel * dtype_size dtype then
            failwith
              [%string
                "Safetensors.read %{path}: tensor %{name} has %{nbytes#Int} bytes but shape \
                 requires %{(numel * dtype_size dtype)#Int}"];
          if start < 0 || stop > buffer_len then
            failwith [%string "Safetensors.read %{path}: tensor %{name} payload out of bounds"];
          (match Hashtbl.add tensors ~key:name ~data:{ dtype; shape; offset = start; nbytes } with
          | `Ok -> ()
          | `Duplicate ->
              failwith [%string "Safetensors.read %{path}: duplicate tensor name %{name}"]);
          order := name :: !order
        end);
    (* The safetensors invariant: the payload ranges tile the byte buffer exactly -- no overlaps,
       holes, or trailing bytes. Checking each range independently would accept corrupted or
       adversarial headers where distinct tensors alias the same bytes. *)
    let ranges =
      Hashtbl.fold tensors ~init:[] ~f:(fun ~key ~data acc ->
          (data.offset, data.offset + data.nbytes, key) :: acc)
      |> List.sort ~compare:(fun (s1, e1, _) (s2, e2, _) ->
          match Int.compare s1 s2 with 0 -> Int.compare e1 e2 | c -> c)
    in
    let final =
      List.fold ranges ~init:0 ~f:(fun cursor (start, stop, name) ->
          if start <> cursor then
            failwith
              [%string
                "Safetensors.read %{path}: tensor %{name} starts at byte %{start#Int} but the \
                 previous payload ends at %{cursor#Int} (overlapping or non-contiguous \
                 data_offsets)"];
          stop)
    in
    if final <> buffer_len then
      failwith
        [%string
          "Safetensors.read %{path}: payloads end at byte %{final#Int} but the byte buffer has \
           %{buffer_len#Int} bytes (trailing uncovered data)"];
    {
      path;
      ic;
      closed = false;
      buffer_start;
      tensors;
      order = List.rev !order;
      metadata = !metadata;
    }
  in
  match parse () with
  | t ->
      (* Dropping [t] without closing it must not leak the descriptor. *)
      Stdlib.Gc.finalise close t;
      t
  | exception exn ->
      Stdlib.close_in_noerr ic;
      raise exn

(** {2 Payload ingestion: mapped or copied (gh-ocannl-587)} *)

(* The safetensors dtypes that name the same bit layout as an OCANNL precision. The absent ones
   would each be a reinterpretation rather than a naming: I8/I16 are signed where [byte]/[uint16]
   are not, and F8_E4M3 is a different 8-bit float from OCANNL's [fp8], which is e5m2. BOOL is 0/1
   bytes, which [byte] reads as the floats 0. and 1. *)
let prec_of_dtype = function
  | "F64" -> Some Ir.Ops.double
  | "F32" -> Some Ir.Ops.single
  | "F16" -> Some Ir.Ops.half
  | "BF16" -> Some Ir.Ops.bfloat16
  | "F8_E5M2" -> Some Ir.Ops.fp8
  | "I64" -> Some Ir.Ops.int64
  | "U64" -> Some Ir.Ops.uint64
  | "I32" -> Some Ir.Ops.int32
  | "U32" -> Some Ir.Ops.uint32
  | "U16" -> Some Ir.Ops.uint16
  | "U8" | "BOOL" -> Some Ir.Ops.byte
  | _ -> None

(* Which ingestion path a payload took is otherwise invisible -- the two produce equal values by
   construction -- so it is counted, for tests and for diagnosing an unexpectedly slow load. *)
let mapped_count = Atomic.make 0
let copied_count = Atomic.make 0
let ingestion_counts () = (Atomic.get mapped_count, Atomic.get copied_count)

(* A payload can be mapped when the file's bytes ARE the host buffer's bytes. The format does that
   half: it fixes little-endian order and contiguous, unpadded payloads, and [read] has already
   checked that the ranges tile the byte buffer. The rest is {!Ir.Ndarray.mappable_file_region}'s --
   and the alignment it checks is the condition this format does not guarantee, the byte buffer
   starting at [8 + header_len] for a header length the producer chooses, with payload offsets then
   accumulating the preceding payloads' sizes. *)

let payload_info t what name =
  match info t name with
  | Some i -> i
  | None -> failwith [%string "Safetensors.%{what} %{t.path}: no tensor named %{name}"]

let to_ndarray ?prec t name =
  let { dtype; shape; offset; nbytes } = payload_info t "to_ndarray" name in
  let payload_prec =
    match prec_of_dtype dtype with
    | Some p -> p
    | None ->
        failwith
          [%string
            "Safetensors.to_ndarray %{t.path}: tensor %{name} has dtype %{dtype}, which has no \
             matching OCANNL precision"]
  in
  let ic = checked_ic t "to_ndarray" in
  let dims = match shape with [] -> [| 1 |] | dims -> Array.of_list dims in
  let numel = Array.fold dims ~init:1 ~f:( * ) in
  (* [read] checked this against [dtype_size]; restating it against the precision actually being
     ingested is what keeps a mapping from running past the payload if the two tables drift. *)
  if nbytes <> numel * Ir.Ops.prec_in_bytes payload_prec then
    failwith
      [%string
        "Safetensors.to_ndarray %{t.path}: tensor %{name} has %{nbytes#Int} bytes, but \
         %{numel#Int} elements of %{dtype} need %{(numel * Ir.Ops.prec_in_bytes payload_prec)#Int}"];
  let byte_offset = t.buffer_start + offset in
  let nd =
    if Ir.Ndarray.mappable_file_region ~prec:payload_prec ~byte_offset ~nbytes then begin
      Atomic.incr mapped_count;
      (* The descriptor the header was read through, not a fresh open of [t.path]: the offsets and
         extents being mapped describe the file this read started on. The mapping outlives the
         descriptor -- and the directory entry -- so nothing depends on the file staying put. *)
      Ir.Ndarray.map_file_array payload_prec ~dims ~byte_offset (Unix.descr_of_in_channel ic)
    end
    else begin
      Atomic.incr copied_count;
      let nd = Ir.Ndarray.create_array ~debug:name payload_prec ~dims ~padding:None in
      Stdlib.seek_in ic byte_offset;
      Ir.Ndarray.read_payload_from_channel nd ic nbytes;
      nd
    end
  in
  let result = match prec with None -> nd | Some prec -> Ir.Ndarray.convert prec nd in
  (* [t] is dead, as far as the compiler is concerned, once [ic] and [buffer_start] have been read
     out of it -- and its finaliser closes that very channel, which the allocation-heavy paths above
     are still using. So it has to be held past them. *)
  ignore (Stdlib.Sys.opaque_identity t : t);
  result

let to_float32 t name =
  let { dtype; _ } = payload_info t "to_float32" name in
  if not (String.equal dtype "F32") then
    failwith
      [%string
        "Safetensors.to_float32 %{t.path}: tensor %{name} has dtype %{dtype}, only F32 is supported"];
  match to_ndarray t name with
  | Ir.Ndarray.Single_nd arr -> arr
  | nd ->
      (* Unreachable: "F32" is [Ops.single], and neither ingestion path changes the precision. *)
      failwith
        [%string
          "Safetensors.to_float32 %{t.path}: tensor %{name} ingested as %{Ir.Ops.prec_string \
           (Ir.Ndarray.get_prec nd)}"]
