open Base
open Stdio

type tensor_info = { dtype : string; shape : int list; offset : int; nbytes : int }

type t = {
  path : string;
  buffer_start : int;  (** File position where the byte buffer (data section) starts. *)
  tensors : (string, tensor_info) Hashtbl.t;
  order : string list;  (** Names in header order. *)
  metadata : (string * string) list;
}

let path t = t.path
let names t = t.order
let info t name = Hashtbl.find t.tensors name
let metadata t = t.metadata

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
  In_channel.with_file path ~binary:true ~f:(fun ic ->
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
      (* The safetensors invariant: the payload ranges tile the byte buffer exactly -- no
         overlaps, holes, or trailing bytes. Checking each range independently would accept
         corrupted or adversarial headers where distinct tensors alias the same bytes. *)
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
      { path; buffer_start; tensors; order = List.rev !order; metadata = !metadata })

let to_float32 t name =
  let { dtype; shape; offset; nbytes } =
    match info t name with
    | Some i -> i
    | None -> failwith [%string "Safetensors.to_float32 %{t.path}: no tensor named %{name}"]
  in
  if not (String.equal dtype "F32") then
    failwith
      [%string
        "Safetensors.to_float32 %{t.path}: tensor %{name} has dtype %{dtype}, only F32 is \
         supported"];
  let payload = Bytes.create nbytes in
  In_channel.with_file t.path ~binary:true ~f:(fun ic ->
      In_channel.seek ic (Int64.of_int (t.buffer_start + offset));
      In_channel.really_input_exn ic ~buf:payload ~pos:0 ~len:nbytes);
  let dims = match shape with [] -> [| 1 |] | dims -> Array.of_list dims in
  let genarray = Bigarray.Genarray.create Bigarray.Float32 Bigarray.c_layout dims in
  let flat = Bigarray.reshape_1 genarray (nbytes / 4) in
  for i = 0 to (nbytes / 4) - 1 do
    (* Little-endian payload; decoding via int32 bits is endianness-correct on any host. *)
    flat.{i} <- Stdlib.Int32.float_of_bits (Stdlib.Bytes.get_int32_le payload (i * 4))
  done;
  genarray

let to_ndarray t name = Ir.Ndarray.as_array Ir.Ops.Single (to_float32 t name)
