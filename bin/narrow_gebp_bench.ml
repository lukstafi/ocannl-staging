(* Narrow-operand GEBP benchmark (gh-ocannl-575): the CPU register-tiled [Tile_mma] over 16-bit
   storage, with the widening folded into the packing [Stage] ([tile_prec]). Times, on the C
   backends, the serial naive kernel against the all-Serial packed GEBP and the pool-parallel
   grid-outermost per-chunk-packing flavor (schedule_bench's packmma / pm_bpk shapes), at the
   requested storage precision — the packed panels are minted at the compute precision the
   emission resolves ([Numerics.cpu_compute_prec] over [Context.hardware_limits]), so an f32 run
   measures the plain GEBP, a bf16/f16 run measures f32-GEBP-over-narrow-storage, and an f16 run
   with --ocannl_fp16_arithmetic=true on a native-arithmetic target (NEON, AVX512-FP16) measures
   the pure-f16 GEBP — the honest first comparison the issue asks for. On a target that merely
   promotes ([cc_fp16_arithmetic] probes it), the policy is ignored and the run stays f32-compute.

   Usage: OCANNL_BACKEND=cc dune exec bin/narrow_gebp_bench.exe -- [f32|bf16|f16] [n] [repeats]
   (defaults f32, 512, 20; n a multiple of 64). Readbacks stay outside the timed region (the
   [Context.get_values] trap, docs/agent-notes.md). *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module LL = Ir.Low_level
module Sched = Ir.Schedule
module Asgns = Ir.Assignments
module Numerics = Ir.Numerics

let p = Stdio.printf

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

let named name (comp : Asgns.comp) : Asgns.comp =
  { comp with asgns = Asgns.Block_comment (name, comp.asgns) }

let sink sym below = List.map below ~f:(fun inner -> Sched.Swap { outer = sym; inner })

let () =
  let pos_args =
    Array.to_list (Sys.get_argv ())
    |> List.tl_exn
    |> List.filter ~f:(fun s -> not (String.is_prefix s ~prefix:"-"))
  in
  let prec_name = Option.value (List.nth pos_args 0) ~default:"f32" in
  let prec =
    match prec_name with
    | "f32" -> Ir.Ops.single
    | "bf16" -> Ir.Ops.bfloat16
    | "f16" -> Ir.Ops.half
    | s -> invalid_arg ("narrow_gebp_bench: precision f32|bf16|f16 expected, got " ^ s)
  in
  let arg i default =
    match List.nth pos_args i with Some s -> Int.of_string s | None -> default
  in
  let n = arg 1 512 in
  let repeats = arg 2 20 in
  assert (n % 64 = 0);
  let bm, bk = (64, 256) in
  let flops = 2.0 *. Float.of_int n *. Float.of_int n *. Float.of_int n in
  (* Exactly-representable inputs at every storage precision (the parity-test recipes); the spot
     check guards against an all-zeros or NaN run without a full reference. *)
  let ma =
    NTDSL.init ~l:"ma" ~prec ~i:[ n ] ~o:[ n ]
      ~f:(fun idcs -> Float.of_int (((idcs.(0) * n) + idcs.(1)) % 3) *. 0.25)
      ()
  in
  let mb =
    NTDSL.init ~l:"mb" ~prec ~i:[ n ] ~o:[ n ]
      ~f:(fun idcs -> (Float.of_int (((idcs.(0) * n) + idcs.(1)) % 5) -. 2.) *. 0.5)
      ()
  in
  let packed_schedule ~grid ~tile_prec ~mc (opt : LL.optimized) : Sched.schedule =
    let paths = nest_paths opt.LL.llc in
    let i, j, k =
      match List.find_exn paths ~f:(fun p -> List.length p = 3) with
      | [ i; j; k ] -> (i, j, k)
      | _ -> assert false
    in
    let outer_i = if grid then LL.Grid else LL.Serial in
    let sp_i, i_o, i_i = Sched.split ~axis:i ~factor:bm ~outer:outer_i ~inner:LL.Serial in
    let sp_k, k_o, k_i = Sched.split ~axis:k ~factor:bk ~outer:LL.Serial ~inner:LL.Serial in
    let stage source tile_loops =
      Sched.Stage
        {
          source;
          tile_loops;
          shared = false;
          cooperative = None;
          hoisted = false;
          swizzle = None;
          pad_stride = None;
          pipeline_depth = 1;
          tile_prec;
        }
    in
    let zops =
      if not grid then []
      else
        let ez, zsyms = Sched.expand_zero ~tn:mc in
        let zi = List.hd_exn zsyms in
        let sp_zi, _, _ = Sched.split ~axis:zi ~factor:bm ~outer:LL.Grid ~inner:LL.Serial in
        [ ez; sp_zi ]
    in
    let tz, _lane = Sched.tensorize ~i:i_i ~j ~k:k_i ~simd_width:1 in
    zops @ [ sp_i; sp_k ]
    @ sink j [ k_o ]
    @ sink i_i [ k_o ]
    @ (if grid then [] else sink i_o [ k_o ])
    @ [ stage mb.Tensor.value [ k_i; j ]; stage ma.Tensor.value [ i_i; k_i ] ]
    @ [ tz ]
  in
  let bench ~variant ~schedule () =
    let%op mc = ma * mb in
    Ir.Tnode.update_prec mc.Tensor.value prec;
    let comp = named ("ngb_" ^ variant) (Train.forward mc) in
    let transform opt =
      match schedule with None -> opt | Some s -> Sched.apply (s ~mc:mc.Tensor.value opt) opt
    in
    let ctx = Context.auto () in
    let ctx, routine = Context.compile ~lowered_transform:transform ctx comp Ir.Indexing.Empty in
    let ctx = Context.run ctx routine in
    let _ = Context.get_values ctx mc.Tensor.value in
    let start = Time_now.nanoseconds_since_unix_epoch () in
    let ctx =
      Stdlib.Array.fold_left
        (fun ctx () -> Context.run ctx routine)
        ctx (Stdlib.Array.make repeats ())
    in
    let values = Context.get_values ctx mc.Tensor.value in
    let stop = Time_now.nanoseconds_since_unix_epoch () in
    let secs = Float.of_int63 Int63.(stop - start) /. 1e9 /. Float.of_int repeats in
    p "%-12s %8.3f ms  %8.2f GFLOP/s  (spot check %.2f)\n" variant (secs *. 1e3)
      (flops /. secs /. 1e9)
      values.(n + 1);
    secs
  in
  let ctx0 = Context.auto () in
  let limits = Context.hardware_limits ctx0 in
  let cprec =
    Numerics.cpu_compute_prec ~native_fp16_arithmetic:limits.Ir.Backend_intf.native_fp16_arithmetic
      prec
  in
  let tile_prec = if Ir.Ops.equal_prec cprec prec then None else Some cprec in
  p "GEBP n=%d, %d repeats, storage %s, compute %s, packed panels %s\n" n repeats
    (Ir.Ops.prec_string prec) (Ir.Ops.prec_string cprec)
    (Option.value_map tile_prec ~default:"(storage)" ~f:Ir.Ops.prec_string);
  let t_naive = bench ~variant:"naive" ~schedule:None () in
  let t_pack = bench ~variant:"packmma" ~schedule:(Some (packed_schedule ~grid:false ~tile_prec)) () in
  let t_par =
    bench ~variant:"packmma_par" ~schedule:(Some (packed_schedule ~grid:true ~tile_prec)) ()
  in
  p "speedups vs naive: packmma %.1fx, packmma_par %.1fx\n" (t_naive /. t_pack) (t_naive /. t_par)
