(* Regression test for gh-ocannl-344: under [large_models] the per-pool 4 GB cap is lifted, so a
   pooled byte offset can exceed UINT32_MAX. The Metal pooled slot table -- and the MSL type the
   generated shader declares for it -- must therefore be 64-bit ([ulong]); a 32-bit ([uint]) table
   would silently truncate large-model offsets, defeating the "large_models=true => 64-bit" AC for
   the Metal pool path.

   This compiles a pooled Metal kernel with [large_models = true] and inspects the emitted shader.
   The invariant pinned: the generated source declares [ulong* __pool_slots] and NOT [uint*
   __pool_slots]. If the slot table regressed to [uint] under [large_models], both claims below
   would fail. The harness condition that instantiates the AC is [large_models = true] set before
   compilation -- the same kernel under the default setting emits [uint], so the setting is what the
   claims actually exercise. *)

open! Base
open Ocannl
open Operation.DSL_modules

let make_const label v =
  let open Bigarray in
  let ga = Genarray.create Float32 c_layout [| 2 |] in
  Genarray.set ga [| 0 |] v;
  Genarray.set ga [| 1 |] (v +. 0.5);
  let nd = Ir.Ndarray.as_array Ir.Ops.Single ga in
  Tensor.term ~init_data:(Reshape nd) ~grad_spec:Tensor.Prohibit_grad ~label:[ label ]
    ~batch_dims:[] ~input_dims:[] ~output_dims:[ 2 ] ()

let read_metal_sources () =
  let dir = Utils.build_files_dir () in
  (try Stdlib.Sys.readdir dir |> Array.to_list with _ -> [])
  |> List.filter ~f:(String.is_suffix ~suffix:".metal")
  |> List.map ~f:(fun f -> Stdio.In_channel.read_all (Stdlib.Filename.concat dir f))

let () =
  Tensor.unsafe_reinitialize ();
  Utils.settings.large_models <- true;
  Utils.settings.output_debug_files_in_build_directory <- true;
  let ctx = Context.metal () in
  let sum = TDSL.O.(make_const "a" 1. + make_const "b" 2.) in
  let _ctx = Train.forward_once ctx sum in
  let srcs = read_metal_sources () in
  let has sub = List.exists srcs ~f:(String.is_substring ~substring:sub) in
  Verdict.p "large_models=true: generated slot table is ulong* __pool_slots"
    (has "ulong* __pool_slots");
  (* Phrased as the fact that holds, so that [true] is the passing reading on both lines: a designed
     negative and a blessed regression are the same line in a golden (gh-ocannl-624). *)
  if has "uint* __pool_slots" then
    Stdio.eprintf
      "large-model source unexpectedly contains a uint pool table (not part of the golden)\n";
  Verdict.p_none "large_models=true: generated slot table is not uint* __pool_slots" srcs
    ~f:(fun source -> String.is_substring source ~substring:"uint* __pool_slots")
