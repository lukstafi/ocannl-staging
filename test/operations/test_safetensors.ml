(* Roundtrip test for the Safetensors reader: writes a small file in the safetensors format (8-byte
   LE header length, JSON header, little-endian payloads), reads it back, and pushes one tensor
   through a device roundtrip via TDSL.wrap. *)

open Base
open Ocannl
open Stdio
open Nn_blocks.DSL_modules

let le_bytes_of_floats values =
  let b = Bytes.create (4 * Array.length values) in
  Array.iteri values ~f:(fun i v ->
      Stdlib.Bytes.set_int32_le b (4 * i) (Stdlib.Int32.bits_of_float v));
  Bytes.to_string b

let () =
  let a_vals = [| 0.; 1.; 2.; 3.; 4.; 5. |] in
  let b_vals = [| 10.; 20.; 30.; 40. |] in
  let header =
    {|{"__metadata__":{"format":"pt"},"a":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]},"b":{"dtype":"F32","shape":[4],"data_offsets":[24,40]}}|}
  in
  let len_bytes = Bytes.create 8 in
  Stdlib.Bytes.set_int64_le len_bytes 0 (Int64.of_int (String.length header));
  let path = "roundtrip.safetensors" in
  Out_channel.write_all path
    ~data:
      (Bytes.to_string len_bytes ^ header ^ le_bytes_of_floats a_vals ^ le_bytes_of_floats b_vals);

  let st = Safetensors.read path in
  printf "names: %s\n" (String.concat ~sep:", " (Safetensors.names st));
  List.iter (Safetensors.metadata st) ~f:(fun (k, v) -> printf "metadata %s = %s\n" k v);
  (match Safetensors.info st "a" with
  | Some { dtype; shape; offset; nbytes } ->
      printf "a: dtype %s, shape [%s], offset %d, nbytes %d\n" dtype
        (String.concat ~sep:"; " (List.map shape ~f:Int.to_string))
        offset nbytes
  | None -> printf "a: missing!\n");
  let a = Safetensors.to_float32 st "a" in
  let a_ok =
    Array.for_alli a_vals ~f:(fun i v -> Float.equal (Bigarray.Genarray.get a [| i / 3; i % 3 |]) v)
  in
  printf "a roundtrips (2x3): %b\n" a_ok;

  (* Device roundtrip through an OCANNL tensor. *)
  let b = TDSL.wrap ~l:"b" ~b:[] ~o:[ 4 ] (Safetensors.to_ndarray st "b") () in
  let ctx = Train.forward_once (Context.auto ()) b in
  let got = Context.get_values ctx b.Tensor.value in
  printf "b via tensor: [%s]\n"
    (String.concat ~sep:"; " (Array.to_list (Array.map got ~f:(fun v -> Printf.sprintf "%.0f" v))));

  (* Error paths. *)
  match Safetensors.to_float32 st "missing" with
  | exception Failure msg -> printf "missing tensor: %s\n" msg
  | _ -> printf "missing tensor: unexpectedly succeeded\n"

(* The payload ranges must tile the byte buffer exactly (safetensors invariant): overlapping or
   non-contiguous data_offsets, and trailing uncovered bytes, are rejected. *)
let () =
  let write_file path header payload =
    let len_bytes = Bytes.create 8 in
    Stdlib.Bytes.set_int64_le len_bytes 0 (Int64.of_int (String.length header));
    Out_channel.write_all path ~data:(Bytes.to_string len_bytes ^ header ^ payload)
  in
  let payload = le_bytes_of_floats [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let overlapping =
    {|{"a":{"dtype":"F32","shape":[4],"data_offsets":[0,16]},"b":{"dtype":"F32","shape":[4],"data_offsets":[8,24]}}|}
  in
  write_file "overlap.safetensors" overlapping payload;
  (match Safetensors.read "overlap.safetensors" with
  | exception Failure msg -> printf "overlap rejected: %s\n" msg
  | _ -> printf "overlap rejected: unexpectedly accepted\n");
  let trailing = {|{"a":{"dtype":"F32","shape":[4],"data_offsets":[0,16]}}|} in
  write_file "trailing.safetensors" trailing payload;
  match Safetensors.read "trailing.safetensors" with
  | exception Failure msg -> printf "trailing rejected: %s\n" msg
  | _ -> printf "trailing rejected: unexpectedly accepted\n"
