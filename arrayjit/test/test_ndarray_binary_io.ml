open Base
module Nd = Ir.Ndarray
module Ops = Ir.Ops

(* One temp file PER test, not one shared path: each mapped read below leaves a live mapping of the
   file, and Windows refuses to truncate a file some section still maps (the section pins the file's
   size), so reusing the path would fail the next test's [open_out_bin] there with EINVAL
   (gh-ocannl-588). Deleting the file under a live mapping is fine on every platform -- modern
   Windows uses POSIX delete semantics, the same fact that lets a checkpoint save rename over a
   mapped file -- so each test removes its own file while its mapping is still alive. *)
let fresh_tmp_file () = Stdlib.Filename.temp_file "ndarray_binary_io_test" ".bin"

let test_round_trip_prec prec_name prec init_f =
  let tmp_file = fresh_tmp_file () in
  let dims = [| 3; 4 |] in
  let nd1 = Nd.create_array ~debug:"test" prec ~dims ~padding:None in
  (* Initialize with known values *)
  let idx = Array.create ~len:2 0 in
  for i = 0 to 2 do
    for j = 0 to 3 do
      idx.(0) <- i;
      idx.(1) <- j;
      init_f nd1 idx ((i * 4) + j)
    done
  done;
  (* Write payload to file *)
  let oc = Stdlib.open_out_bin tmp_file in
  let n_bytes = Nd.write_payload_to_channel nd1 oc in
  Stdlib.close_out oc;
  (* Read payload into fresh ndarray *)
  let nd2 = Nd.create_array ~debug:"test2" prec ~dims ~padding:None in
  let ic = Stdlib.open_in_bin tmp_file in
  Nd.read_payload_from_channel nd2 ic n_bytes;
  Stdlib.close_in ic;
  (* Compare using exact byte comparison *)
  Verdict.pass_fail (Printf.sprintf "%s round trip" prec_name) (Nd.payloads_equal nd1 nd2);
  (* The mapped read (gh-ocannl-467) reinterprets the payload bytes in place instead of decoding
     them element by element. That the two agree is the whole premise of mapped checkpoint loading:
     each precision's payload encoding has to be its in-memory representation. *)
  let fd = Unix.openfile tmp_file [ Unix.O_RDONLY ] 0 in
  let nd3 = Nd.map_file_array prec ~dims ~byte_offset:0 fd in
  Unix.close fd;
  Stdlib.Sys.remove tmp_file;
  Verdict.pass_fail (Printf.sprintf "%s mapped" prec_name) (Nd.payloads_equal nd1 nd3)

let test_padded () =
  let tmp_file = fresh_tmp_file () in
  let prec = Ops.single in
  let padding = Some ([| Ops.{ left = 1; right = 1 }; Ops.{ left = 0; right = 2 } |], 0.0) in
  (* Padded dims: 2+1+1=4 x 3+0+2=5, logical: 2x3 *)
  let dims = [| 4; 5 |] in
  let nd1 = Nd.create_array ~debug:"padded" prec ~dims ~padding in
  let padding_arr = [| Ops.{ left = 1; right = 1 }; Ops.{ left = 0; right = 2 } |] in
  (* Set logical values *)
  let idx = Array.create ~len:2 0 in
  for i = 0 to 1 do
    for j = 0 to 2 do
      idx.(0) <- i;
      idx.(1) <- j;
      Nd.set_from_float ~padding:padding_arr nd1 idx (Float.of_int ((i * 3) + j + 1))
    done
  done;
  (* Write payload with padding *)
  let oc = Stdlib.open_out_bin tmp_file in
  let n_bytes = Nd.write_payload_to_channel ~padding:padding_arr nd1 oc in
  Stdlib.close_out oc;
  (* Read into fresh padded ndarray *)
  let nd2 = Nd.create_array ~debug:"padded2" prec ~dims ~padding in
  let ic = Stdlib.open_in_bin tmp_file in
  Nd.read_payload_from_channel ~padding:padding_arr nd2 ic n_bytes;
  Stdlib.close_in ic;
  (* Compare logical payloads *)
  Stdlib.Sys.remove tmp_file;
  Verdict.pass_fail "padded round trip" (Nd.payloads_equal ~padding:padding_arr nd1 nd2)

(* A payload does not start at the beginning of the file, and its offset is not page aligned: the
   runtime maps from the enclosing page boundary and offsets the data pointer, so any offset works
   (which is why checkpoint alignment is about SIMD-friendly pointers, not about mappability). *)
let test_mapped_at_offset () =
  let prec = Ops.double in
  let dims = [| 5 |] in
  let nd1 = Nd.create_array ~debug:"offset" prec ~dims ~padding:None in
  for i = 0 to 4 do
    Nd.set_from_float nd1 [| i |] (Float.of_int i *. 1.5)
  done;
  let path = Stdlib.Filename.temp_file "ndarray_map_offset" ".bin" in
  let oc = Stdlib.open_out_bin path in
  let byte_offset = 7 in
  Stdlib.output_string oc (String.make byte_offset 'x');
  let _n : int = Nd.write_payload_to_channel nd1 oc in
  Stdlib.close_out oc;
  let fd = Unix.openfile path [ Unix.O_RDONLY ] 0 in
  let nd2 = Nd.map_file_array prec ~dims ~byte_offset fd in
  Unix.close fd;
  (* The mapping outlives the descriptor it was taken from, and the file it was taken from. *)
  Stdlib.Sys.remove path;
  Verdict.pass_fail "mapped at unaligned offset" (Nd.payloads_equal nd1 nd2)

let () =
  (* Test each precision type *)
  test_round_trip_prec "Byte" Ops.byte (fun nd idx i ->
      Nd.set_from_float nd idx (Float.of_int (i % 256)));
  test_round_trip_prec "Uint16" Ops.uint16 (fun nd idx i ->
      Nd.set_from_float nd idx (Float.of_int (i * 1000)));
  test_round_trip_prec "Int32" Ops.int32 (fun nd idx i ->
      Nd.set_from_float nd idx (Float.of_int ((i * 100000) - 500000)));
  test_round_trip_prec "Uint32" Ops.uint32 (fun nd idx i ->
      Nd.set_from_float nd idx (Float.of_int (i * 100000)));
  test_round_trip_prec "Int64" Ops.int64 (fun nd idx i ->
      Nd.set_from_float nd idx (Float.of_int ((i + 1) * 1_000_000_000_000)));
  test_round_trip_prec "Uint64" Ops.uint64 (fun nd idx i ->
      Nd.set_from_float nd idx (Float.of_int ((i + 1) * 1_000_000_000_000)));
  test_round_trip_prec "Half" Ops.half (fun nd idx i ->
      Nd.set_from_float nd idx (Float.of_int i *. 0.5));
  test_round_trip_prec "Bfloat16" Ops.bfloat16 (fun nd idx i ->
      Nd.set_from_float nd idx (Float.of_int i *. 0.25));
  test_round_trip_prec "Fp8" Ops.fp8 (fun nd idx i ->
      Nd.set_from_float nd idx (Float.of_int (i % 128) *. 0.1));
  test_round_trip_prec "Single" Ops.single (fun nd idx i ->
      Nd.set_from_float nd idx (Float.of_int i *. 3.14));
  test_round_trip_prec "Double" Ops.double (fun nd idx i ->
      Nd.set_from_float nd idx (Float.of_int i *. 2.71828));
  (* Note: Uint4x32 uses Complex.t carrier with raw byte access *)
  test_round_trip_prec "Uint4x32" Ops.uint4x32 (fun nd idx i ->
      Nd.set_from_float nd idx (Float.of_int ((i * 7) + 3)));
  (* Test padded tensor *)
  test_padded ();
  test_mapped_at_offset ()
