(* The uint4x32 FFI stubs take an [int array], a type that carries no length, and each one reads
   four lanes out of it. Before gh-ocannl-688 the C helper [ocaml_array_to_uint4x32] read Field(v,
   0..3) unconditionally, so a shorter array was an out-of-bounds read. It usually landed on
   adjacent heap words and passed unnoticed; it faulted when the array happened to be the topmost
   block in the minor heap, because then lanes 1..3 sit on the PROT_NONE guard page beyond young_end
   and the read is a SIGBUS rather than a wrong number.

   [Gc.minor ()] resets young_ptr to young_end, so the [Array.make 1 0] right after it is the first
   allocation of a fresh minor heap and therefore lands at the very top -- which makes the fault
   deterministic instead of depending on where the allocator happened to be. *)

open Ir

(* Each stub is exercised through a uniform wrapper so that a lane-count regression in any one of
   them is caught, not just in the two the crash reports happened to name. *)
let stubs : (string * (int array -> unit)) list =
  [
    ("threefry4x32", fun a -> ignore (Ops.threefry4x32 a a));
    ("threefry4x32_crypto", fun a -> ignore (Ops.threefry4x32_crypto a a));
    ("threefry4x32_light", fun a -> ignore (Ops.threefry4x32_light a a));
    ("uint4x32_to_single_uniform", fun a -> ignore (Ops.uint4x32_to_single_uniform a));
    ("uint4x32_to_double_uniform", fun a -> ignore (Ops.uint4x32_to_double_uniform a));
    ("uint4x32_to_int32_uniform", fun a -> ignore (Ops.uint4x32_to_int32_uniform a));
    ("uint4x32_to_int64_uniform", fun a -> ignore (Ops.uint4x32_to_int64_uniform a));
    ("uint4x32_to_uint32_uniform", fun a -> ignore (Ops.uint4x32_to_uint32_uniform a));
    ("uint4x32_to_uint64_uniform", fun a -> ignore (Ops.uint4x32_to_uint64_uniform a));
    ("uint4x32_to_byte_uniform", fun a -> ignore (Ops.uint4x32_to_byte_uniform a));
    ("uint4x32_to_uint16_uniform", fun a -> ignore (Ops.uint4x32_to_uint16_uniform a));
    ("uint4x32_to_bfloat16_uniform", fun a -> ignore (Ops.uint4x32_to_bfloat16_uniform a));
    ("uint4x32_to_half_uniform", fun a -> ignore (Ops.uint4x32_to_half_uniform a));
    ("uint4x32_to_fp8_uniform", fun a -> ignore (Ops.uint4x32_to_fp8_uniform a));
  ]

(* Every under-length arity below the four lanes a uint4x32 block has, plus the empty array. *)
let short_arities = [ 0; 1; 2; 3 ]

let () =
  let rejects_short =
    List.for_all
      (fun (_, call) ->
        List.for_all
          (fun n ->
            Gc.minor ();
            let a = Array.make n 0 in
            match call a with exception Invalid_argument _ -> true | () -> false)
          short_arities)
      stubs
  in
  Verdict.p "every uint4x32 stub rejects an under-length array" rejects_short;
  let accepts_four =
    List.for_all
      (fun (_, call) ->
        Gc.minor ();
        let a = [| 1; 2; 3; 4 |] in
        match call a with exception Invalid_argument _ -> false | () -> true)
      stubs
  in
  Verdict.p "every uint4x32 stub accepts a four-lane array" accepts_four;
  (* The linking-check block at the end of Ops runs at module-initialization time and calls every
     one of these stubs; reaching this line at all means it passed four-lane arrays. *)
  Verdict.p "Ops module initialization completed" true
