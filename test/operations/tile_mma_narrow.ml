(* Tile_mma register tiling for narrow (16-bit) operands (gh-ocannl-575, re-scoped from
   gh-ocannl-530, originally gh-ocannl-516/517).

   Three executed legs of the packed GEBP pipeline (the [schedule_pack_mma_matmul] composition:
   B~ packed at [k_o], A~ at [i_o], [Tensorize] at unit lane width), each against its serial twin:

   - bf16 storage computing in f32 (the default [narrow_compute_f32] policy): the packing [Stage]s
     mint their tiles at f32 ([tile_prec = Some single]) so the widening conversion rides the
     packing copy — the micro-kernel reads uniform f32 panels and only the accumulator crosses the
     memory boundary through the narrow bridge ("narrow storage bridged: d:bfloat16").
   - half storage with a HOISTED f32 B~ panel (gh-ocannl-470 composed with the widening pack): the
     host-side [pack_constant_tile] converts half host-init data into the f32 constant-pool panel.
   - pure fp16 (gh-ocannl-516 task 3): [cc_fp16_arithmetic=native] (forced via the environment —
     on a merely promoted target the compiler still implements per-op half rounding, so values are
     unchanged and only the throughput claim would be off) plus the [fp16_arithmetic] policy keep
     the compute precision at half: the register tiling runs at doubled lanes with f16 accumulators
     and identity bridges. Skipped (with a stderr note) where the toolchain has no [_Float16] at
     all — the forced-native kernel then fails to compile.

   Inputs are exactly representable at their storage precision with all partial sums exact (the
   [schedule_mma_matmul] discipline), so every parity check is BITWISE — in particular it cannot
   tell apart the fallback's per-k-step narrowing from the register tiling's whole-k compute-
   precision residency, both being exact on these inputs; the structural pins are what assert the
   rendering. On GPU backends the trailing [Tensorize] is dropped (a unit-lane [Tile_mma] must not
   reach hardware intrinsics): the widened packing is still exercised for value parity. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module Tn = Ir.Tnode
module LL = Ir.Low_level
module Sched = Ir.Schedule
module Asgns = Ir.Assignments
module Numerics = Ir.Numerics

(* Before any backend touch: the pure-f16 leg's probe override (read per call, env is live). *)
let () = Unix.putenv "OCANNL_CC_FP16_ARITHMETIC" "native"
let () = Utils.settings.output_debug_files_in_build_directory <- true
let p = Verdict.p

let nonzero name (a : float array) =
  if not (Array.exists a ~f:(fun x -> Float.(x <> 0.))) then
    failwith (name ^ ": the reference is all zeros — the parity checks against it are vacuous");
  a

let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")
let skipped = Verdict.skipped ~backend:backend_name
let on_cpu = String.is_substring backend_name ~substring:"cc"

let read_generated base_name =
  let ext = if String.is_substring backend_name ~substring:"metal" then ".metal" else ".c" in
  let ext = if String.is_substring backend_name ~substring:"cuda" then ".cu" else ext in
  let ext = if String.is_substring backend_name ~substring:"hip" then ".hip" else ext in
  let path = Utils.build_file (base_name ^ ext) in
  if Stdlib.Sys.file_exists path then Some (Stdio.In_channel.read_all path) else None

let named name (comp : Asgns.comp) : Asgns.comp =
  { comp with asgns = Asgns.Block_comment (name, comp.asgns) }

(* The single-child chain of loops from the top of each top-level nest. *)
let nest_paths (llc : LL.t) : Ir.Indexing.symbol list list =
  let strip stmts = List.filter stmts ~f:(function LL.Noop | LL.Comment _ -> false | _ -> true) in
  let rec path (llc : LL.t) : Ir.Indexing.symbol list =
    match llc with
    | LL.For_loop { index; body; _ } ->
        index :: (match strip (LL.flat_lines [ body ]) with [ single ] -> path single | _ -> [])
    | LL.If { body; _ } -> path body
    | _ -> []
  in
  List.filter_map (LL.flat_lines [ llc ]) ~f:(fun stmt ->
      match path stmt with [] -> None | p -> Some p)

let sink sym below = List.map below ~f:(fun inner -> Sched.Swap { outer = sym; inner })
let n = 64
let bm, bk = (16, 16)

(* Pure-f16 compute runs the register file at the machine's full vector width in 2-byte elements,
   so the typedef the kernel mints is per-machine (8 lanes on NEON, 16 on AVX2, 32 on AVX-512) and
   is computed rather than spelled — a hardcoded list of widths reads as a failure on the first
   machine wider than the ones it lists. GPU backends report 0 bytes and never reach this. *)
let half_vec_typ =
  let simd_vector_bytes =
    (Context.hardware_limits (Context.auto ())).Ir.Backend_intf.simd_vector_bytes
  in
  Printf.sprintf "ocannl_vec%dh"
    (Option.value ~default:0
       (Ir.Backend_intf.simd_lanes_for ~vector_bytes:simd_vector_bytes
          ~elt_bytes:(Ir.Ops.prec_in_bytes Ir.Ops.half) ~extent:n))

(* The composed packed pipeline, parameterized by the packed tiles' precision override. *)
let composed_schedule ~hoist_b ~tile_prec ~a ~b (opt : LL.optimized) : Sched.schedule =
  let paths = nest_paths opt.LL.llc in
  let i, j, k =
    match List.find_exn paths ~f:(fun p -> List.length p = 3) with
    | [ i; j; k ] -> (i, j, k)
    | _ -> assert false
  in
  let sp_i, i_o, i_i = Sched.split ~axis:i ~factor:bm ~outer:LL.Serial ~inner:LL.Serial in
  let sp_k, k_o, k_i = Sched.split ~axis:k ~factor:bk ~outer:LL.Serial ~inner:LL.Serial in
  let stage ~hoisted source tile_loops =
    Sched.Stage
      {
        source;
        tile_loops;
        shared = false;
        cooperative = None;
        hoisted;
        swizzle = None;
        pad_stride = None;
        pipeline_depth = 1;
        tile_prec;
      }
  in
  [ sp_i; sp_k ]
  @ sink j [ k_o ]
  @ sink i_i [ k_o ] @ sink i_o [ k_o ]
  @ [ stage ~hoisted:hoist_b b [ k_i; j ]; stage ~hoisted:false a [ i_i; k_i ] ]
  @ if on_cpu then [ fst (Sched.tensorize ~i:i_i ~j ~k:k_i ~simd_width:1) ] else []

let run ~name ?schedule (out : Tensor.t) =
  let comp = named name (Train.forward out) in
  let transform opt =
    match schedule with None -> opt | Some sched -> Sched.apply (sched opt) opt
  in
  let ctx = Context.auto () in
  let ctx, routine = Context.compile ~lowered_transform:transform ctx comp Ir.Indexing.Empty in
  let ctx = Context.run ctx routine in
  Context.get_values ctx out.Tensor.value

let () =
  (* === bf16 storage, f32 compute, both panels widened at pack time ===
     Inputs: a in {0, 1/4, 1/2}, b in {-1, -1/2, 0, 1/2, 1}; every product is a multiple of 1/8
     bounded by 1/2, and every partial sum over k = 64 is a multiple of 1/8 bounded by 32 —
     exactly representable in bf16's 8 significand bits, so parity is bitwise on either rendering
     and accumulator width. *)
  let mab =
    NTDSL.init ~l:"mab" ~prec:Ir.Ops.bfloat16 ~i:[ n ] ~o:[ n ]
      ~f:(fun idcs -> Float.of_int (((idcs.(0) * n) + idcs.(1)) % 3) *. 0.25)
      ()
  in
  let mbb =
    NTDSL.init ~l:"mbb" ~prec:Ir.Ops.bfloat16 ~i:[ n ] ~o:[ n ]
      ~f:(fun idcs -> (Float.of_int (((idcs.(0) * n) + idcs.(1)) % 5) -. 2.) *. 0.5)
      ()
  in
  let%op bc0 = mab * mbb in
  Tn.update_prec bc0.Tensor.value Ir.Ops.bfloat16;
  let want_b = nonzero "nrw_bf16_serial" (run ~name:"nrw_bf16_serial" bc0) in
  let%op bc1 = mab * mbb in
  Tn.update_prec bc1.Tensor.value Ir.Ops.bfloat16;
  let got_b =
    run ~name:"nrw_bf16_packed"
      ~schedule:
        (composed_schedule ~hoist_b:false ~tile_prec:(Some Ir.Ops.single) ~a:mab.Tensor.value
           ~b:mbb.Tensor.value)
      bc1
  in
  p "bf16 packed+tensorized matmul matches the serial twin bitwise"
    (Array.for_all2_exn got_b want_b ~f:Float.equal);
  (match read_generated "nrw_bf16_packed" with
  | None -> p "bf16 packing widens into f32 panels; only d crosses the narrow bridge" false
  | Some src ->
      let has s = String.is_substring src ~substring:s in
      let count_sub sub =
        String.substr_index_all src ~may_overlap:false ~pattern:sub |> List.length
      in
      p "bf16 packing widens into f32 panels; only d crosses the narrow bridge"
        (if on_cpu then
           (* Two f32 tile arrays: the widening (the C backends' [bfloat16_to_single]) rides the
              packing copy; the register tiling sees f32 panels, bridging only the accumulator. *)
           count_sub "float tile_" = 2
           && has "bfloat16_to_single"
           && has "Tile_mma register tiling"
           && has "narrow storage bridged: d:bfloat16."
           && has "OCANNL_VEC_WIDEN_BFLOAT16"
           && has "OCANNL_VEC_NARROW_BFLOAT16"
         else count_sub "float tile_" = 2 && not (has "tmma_")));

  (* === half storage, hoisted f32 B~ panel (host-side converting pack) + in-kernel f32 A~ === *)
  (* Half leaves are minted at their precision ([ndarray] settles a leaf as [Specified] single);
     the B operand is additionally declared constant so the hoisted pack applies to it. *)
  let mah =
    NTDSL.init ~l:"mah" ~prec:Ir.Ops.half ~i:[ n ] ~o:[ n ]
      ~f:(fun idcs -> Float.of_int (((idcs.(0) * n) + idcs.(1)) % 5) *. 0.125)
      ()
  in
  let mbh =
    NTDSL.init ~l:"mbh" ~prec:Ir.Ops.half ~i:[ n ] ~o:[ n ]
      ~f:(fun idcs -> (Float.of_int (((idcs.(0) * n) + idcs.(1)) % 7) -. 3.) *. 0.25)
      ()
  in
  Tn.set_host_constant mbh.Tensor.value;
  let%op hc0 = mah * mbh in
  Tn.update_prec hc0.Tensor.value Ir.Ops.half;
  let want_h = nonzero "nrw_half_serial" (run ~name:"nrw_half_serial" hc0) in
  let%op hc1 = mah * mbh in
  Tn.update_prec hc1.Tensor.value Ir.Ops.half;
  let got_h =
    run ~name:"nrw_half_hoisted"
      ~schedule:
        (composed_schedule ~hoist_b:true ~tile_prec:(Some Ir.Ops.single) ~a:mah.Tensor.value
           ~b:mbh.Tensor.value)
      hc1
  in
  p "half hoisted-packed matmul matches the serial twin bitwise"
    (Array.for_all2_exn got_h want_h ~f:Float.equal);
  (match read_generated "nrw_half_hoisted" with
  | None -> p "half hoisted pack converts on the host; only d crosses the narrow bridge" false
  | Some src ->
      let has s = String.is_substring src ~substring:s in
      let count_sub sub =
        String.substr_index_all src ~may_overlap:false ~pattern:sub |> List.length
      in
      p "half hoisted pack converts on the host; only d crosses the narrow bridge"
        ((* One in-kernel tile (A~); the B~ panel is a constant-pool buffer packed (and widened)
            on the host, so no in-kernel copy of it exists. *)
         count_sub "float tile_" = 1
        &&
        if on_cpu then
          has "Tile_mma register tiling"
          && has "narrow storage bridged: d:half."
          && has "OCANNL_VEC_WIDEN_HALF"
          && has "OCANNL_VEC_NARROW_HALF"
        else not (has "tmma_")));

  (* === pure fp16: native (forced) f16 arithmetic, f16 accumulators, doubled lanes === *)
  let saved_policy = Numerics.get () in
  Numerics.set_policy { saved_policy with fp16_arithmetic = true };
  (let%op fc0 = mah * mbh in
   Tn.update_prec fc0.Tensor.value Ir.Ops.half;
   let%op fc1 = mah * mbh in
   Tn.update_prec fc1.Tensor.value Ir.Ops.half;
   let parity_name = "pure-f16 packed+tensorized matmul matches the serial twin bitwise" in
   let structure_name = "pure-f16 register tiling: half lanes, identity bridges" in
   match
     let want_f = nonzero "nrw_f16_serial" (run ~name:"nrw_f16_serial" fc0) in
     let got_f =
       run ~name:"nrw_f16_packed"
         ~schedule:
           (composed_schedule ~hoist_b:false ~tile_prec:None ~a:mah.Tensor.value
              ~b:mbh.Tensor.value)
         fc1
     in
     (want_f, got_f)
   with
   | want_f, got_f ->
       p parity_name (Array.for_all2_exn got_f want_f ~f:Float.equal);
       (match read_generated "nrw_f16_packed" with
       | None -> p structure_name false
       | Some src ->
           let has s = String.is_substring src ~substring:s in
           let count_sub sub =
             String.substr_index_all src ~may_overlap:false ~pattern:sub |> List.length
           in
           p structure_name
             (if on_cpu then
                (* Half-typed tiles, an [ocannl_vec<2*f32 lanes>h] register file, no narrow
                   bridge — storage and compute coincide. *)
                count_sub "HALF_T tile_" = 2
                && has "Tile_mma register tiling"
                && (not (has "narrow storage bridged"))
                && has half_vec_typ
              else not (has "tmma_")))
   | exception e ->
       (* The one condition verified to warrant a skip: the generated kernel failed to COMPILE
          over the half type ([Backend_rejected]'s detail quotes the compiler output, which names
          [HALF_T]/[_Float16] when the type is the problem). In fact even a no-[_Float16]
          toolchain is expected to run this leg — [HALF_T] is then [uint16_t] and every half
          operation the tiling emits routes through the emulation-safe [OCANNL_HALF_FMA] /
          [HALF_TO_FLOAT] macros, the same ones the serial twin uses — so this arm should never
          fire; it exists so an exotic toolchain reads as a loud skip rather than a red suite.
          Every other failure (schedule construction, runtime, parity readback) propagates: a
          swallowed exception here would false-green exactly when the implementation breaks. *)
       let msg = Exn.to_string e in
       let mentions s = String.is_substring msg ~substring:s in
       if mentions "failed to compile" && (mentions "_Float16" || mentions "HALF_T") then (
         Stdio.eprintf "pure-f16 leg unavailable, toolchain rejected the half kernel (%s)\n%!" msg;
         skipped parity_name;
         skipped structure_name)
       else raise e);
  Numerics.set_policy saved_policy
