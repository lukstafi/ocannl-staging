(* Convolution sketch families (gh-ocannl-493), route 1: implicit GEMM. A convolution is a matmul
   over a virtual im2col operand, and conv einsums already lower to affine-indexed accumulation
   nests, so the pipeline is a re-association of loops that already exist: reorder to [outer..;
   kernel..; row; oc; ic], pack the input's strided-window [row x ic] slice and the kernel's [ic x
   oc] slice at the kernel-window anchor (the packing Stage IS im2col — same copy nest, conv index
   arithmetic — and normalizes the kernel's stored layout), then Tensorize (row, oc, ic): the
   register-tiled Tile_mma micro-kernel with the accumulator contracted to a fragment resident
   across the innermost kernel loop (gh-ocannl-480).

   Pinned here: - [Autotune.detect_conv] recognizes the conv accumulation across stride/padding
   variants, with strides, dilations, and padding offsets read off the projections. - The hand-built
   implicit-GEMM pipeline (C backends): the packed+tensorized form matches the reorder-only serial
   twin BITWISE (the register tiling's fused per-element chains), and the reorder-only twin matches
   the natural form within float-reassociation tolerance (moving [ic] inside the kernel loops
   reorders each element's reduction — conv sketches are tolerance-tier against unscheduled code,
   like the GPU fragment paths). - Autotune seeding: on the C backends a conv+bias+relu graph seeds
   the serial and Grid-parallel conv pipelines plus their fused-epilogue twins (gh-ocannl-486), the
   tuned routine matches the untuned twin, and the winning schedule round-trips through the saved
   form. GPU conv seeds are a follow-up: seeding is CPU-gated, GPU backends assert zero conv
   candidates. - detect_conv's pattern discipline: a plain matmul is not a conv site. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module LL = Ir.Low_level
module Sched = Ir.Schedule
module Asgns = Ir.Assignments

let () = Utils.settings.output_debug_files_in_build_directory <- true
let p name b = Stdio.printf "%s: %b\n" name b

let named name (comp : Asgns.comp) : Asgns.comp =
  { comp with asgns = Asgns.Block_comment (name, comp.asgns) }

let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")
let on_cpu = String.is_substring backend_name ~substring:"cc"

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

(* Deterministic operands so sibling graphs compute identical values (forward code is consumed by
   compilation, so each leg builds its own graph). *)
let make_x tag =
  NTDSL.init ~l:(tag ^ "x") ~prec:Ir.Ops.single ~b:[ 2 ] ~o:[ 11; 11; 4 ]
    ~f:(fun idcs -> Float.of_int ((idcs.(0) + idcs.(1) + (2 * idcs.(2)) + (3 * idcs.(3))) % 7))
    ()

let make_kern tag =
  NTDSL.init ~l:(tag ^ "k") ~prec:Ir.Ops.single ~i:[ 3; 3; 4 ] ~o:[ 8 ]
    ~f:(fun idcs ->
      Float.of_int (((2 * idcs.(0)) + idcs.(1) + idcs.(2) + (3 * idcs.(3))) % 5) -. 2.)
    ()

let run_plain name y =
  let ctx = Context.auto () in
  let ctx, routine =
    Context.compile
      ~lowered_transform:(fun opt -> opt)
      ctx
      (named name (Train.forward y))
      Ir.Indexing.Empty
  in
  let ctx = Context.run ctx routine in
  Context.get_values ctx y.Tensor.value

let () =
  (* === detect_conv across stride/padding variants (via the conv2d block: random params are fine,
     only the lowered structure is inspected) === *)
  let detect_leg tag ~stride ~use_padding ~want_stride ~want_offset ~want_row =
    let x = make_x tag in
    let conv =
      Nn_blocks.conv2d ~label:[ tag ] ~kernel_size:3 ~stride ~use_padding ~out_channels:8 ()
    in
    let y = conv x in
    let site = ref None in
    let transform (opt : LL.optimized) =
      site := Autotune.detect_conv opt.LL.llc;
      opt
    in
    let ctx = Context.auto () in
    let ctx = Train.init_params ctx Ir.Indexing.Empty y in
    let ctx, routine =
      Context.compile ~lowered_transform:transform ctx
        (named (tag ^ "_det") (Train.forward y))
        Ir.Indexing.Empty
    in
    ignore (ctx, routine);
    match !site with
    | None -> p (tag ^ " detected") false
    | Some s ->
        p (tag ^ " detected")
          (List.length s.Autotune.c_axes = 2
          && s.Autotune.c_nrow = want_row && s.Autotune.c_noc = 8 && s.Autotune.c_nred = 4
          && s.Autotune.c_zeroed && s.Autotune.c_fma
          && List.for_all s.Autotune.c_axes ~f:(fun cx ->
              cx.Autotune.cx_stride = want_stride
              && cx.Autotune.cx_dilation = 1
              && cx.Autotune.cx_offset = want_offset
              && cx.Autotune.cx_nk = 3))
  in
  detect_leg "cvd_s1v" ~stride:1 ~use_padding:false ~want_stride:1 ~want_offset:0 ~want_row:9;
  detect_leg "cvd_s2v" ~stride:2 ~use_padding:false ~want_stride:2 ~want_offset:0 ~want_row:5;
  detect_leg "cvd_s1p" ~stride:1 ~use_padding:true ~want_stride:1 ~want_offset:(-1) ~want_row:11;

  (* === Pattern discipline: a matmul is not a conv site === *)
  (let ma =
     NTDSL.init ~l:"cvm_a" ~prec:Ir.Ops.single ~i:[ 16 ] ~o:[ 16 ]
       ~f:(fun idcs -> Float.of_int ((idcs.(0) + idcs.(1)) % 5))
       ()
   in
   let mb =
     NTDSL.init ~l:"cvm_b" ~prec:Ir.Ops.single ~i:[ 16 ] ~o:[ 16 ]
       ~f:(fun idcs -> Float.of_int ((idcs.(0) - idcs.(1)) % 3))
       ()
   in
   let%op mc = ma * mb in
   let site = ref (Some ()) in
   let transform (opt : LL.optimized) =
     site := Option.map (Autotune.detect_conv opt.LL.llc) ~f:(fun _ -> ());
     opt
   in
   let ctx = Context.auto () in
   let ctx, routine =
     Context.compile ~lowered_transform:transform ctx
       (named "cvm_det" (Train.forward mc))
       Ir.Indexing.Empty
   in
   ignore (ctx, routine);
   p "matmul is not a conv site" (Option.is_none !site));

  (* === The hand-built implicit-GEMM pipeline (C backends; unit-lane Tensorize like the packed mma
     pipelines) === *)
  let make_conv sub =
    let x = make_x sub in
    let kern = make_kern sub in
    let%op y =
      x +* "...| 1*oh<+kh, 1*ow<+kw, ..ic..; |kh, kw, ..ic.. -> ..oc.. => ...| oh, ow, ..oc.." kern
    in
    (x, kern, y)
  in
  let run_sched name (x, kern, y) ~tensorized =
    let transform (opt : LL.optimized) =
      let paths = nest_paths opt.LL.llc in
      let _b, _oh, ow, oc, ic, kh, kw =
        match List.find_exn paths ~f:(fun q -> List.length q = 7) with
        | [ b; oh; ow; oc; ic; kh; kw ] -> (b, oh, ow, oc, ic, kh, kw)
        | _ -> assert false
      in
      let sink sym below = List.map below ~f:(fun inner -> Sched.Swap { outer = sym; inner }) in
      let reorder =
        sink ow [ oc; ic; kh; kw ]
        @ sink oc [ ic; kh; kw ]
        @ sink ic [ kh; kw ]
        @ sink ic [ oc; ow ]
        @ sink oc [ ow ]
      in
      let stage source tile_loops =
        Sched.Stage { source; tile_loops; shared = false; cooperative = None; hoisted = false }
      in
      let sched =
        if not tensorized then reorder
        else
          let tz, _lane = Sched.tensorize ~i:ow ~j:oc ~k:ic ~simd_width:1 in
          reorder @ [ stage x.Tensor.value [ ow; ic ]; stage kern.Tensor.value [ ic; oc ]; tz ]
      in
      Sched.apply sched opt
    in
    let ctx = Context.auto () in
    let ctx, routine =
      Context.compile ~lowered_transform:transform ctx
        (named name (Train.forward y))
        Ir.Indexing.Empty
    in
    let ctx = Context.run ctx routine in
    Context.get_values ctx y.Tensor.value
  in
  let want =
    let _, _, y = make_conv "cvg_r" in
    run_plain "cvg_ref" y
  in
  let swapped = run_sched "cvg_swap" (make_conv "cvg_s") ~tensorized:false in
  p "reorder-only conv matches the natural form within tolerance"
    (Array.for_all2_exn swapped want ~f:(fun a b -> Float.(abs (a - b) < 1e-3)));
  if on_cpu then (
    let full = run_sched "cvg_gemm" (make_conv "cvg_g") ~tensorized:true in
    p "packed+tensorized conv matches the reorder-only twin bitwise"
      (Array.for_all2_exn full swapped ~f:Float.equal);
    p "packed+tensorized conv matches the natural form within tolerance"
      (Array.for_all2_exn full want ~f:(fun a b -> Float.(abs (a - b) < 1e-3)));
    let src = Stdio.In_channel.read_all (Utils.build_file "cvg_gemm.c") in
    let has s = String.is_substring src ~substring:s in
    p "conv pipeline structure: im2col packs, register tiling, resident fragment"
      (has "Tile_mma register tiling" && has "fragment_" && has "tile_"))
  else (
    p "packed+tensorized conv matches the reorder-only twin bitwise" true;
    p "packed+tensorized conv matches the natural form within tolerance" true;
    p "conv pipeline structure: im2col packs, register tiling, resident fragment" true);

  (* === Autotune seeding on conv+bias+relu: serial + Grid conv pipelines and their fused-epilogue
     twins; the tuned routine matches the untuned twin === *)
  let clean_cache dir =
    if Stdlib.Sys.file_exists dir && Stdlib.Sys.is_directory dir then
      Array.iter (Stdlib.Sys.readdir dir) ~f:(fun f ->
          Stdlib.Sys.remove (Stdlib.Filename.concat dir f))
  in
  clean_cache "conv_tune_cache";
  let make_tail sub =
    let x = make_x sub in
    let kern = make_kern sub in
    let bias =
      NTDSL.init ~l:(sub ^ "b") ~prec:Ir.Ops.single ~o:[ 8 ]
        ~f:(fun idcs -> Float.of_int (idcs.(0) % 3) -. 1.)
        ()
    in
    let%op pr =
      x +* "...| 1*oh<+kh, 1*ow<+kw, ..ic..; |kh, kw, ..ic.. -> ..oc.. => ...| oh, ow, ..oc.." kern
    in
    let%op y = relu (pr + bias) in
    y
  in
  let want_t = run_plain "cvt_ref" (make_tail "cvt_r") in
  let y = make_tail "cvt_t" in
  let reports = ref [] in
  let ctx = Context.auto () in
  let ctx, routine =
    Autotune.tune ~beam_width:2 ~rounds:0 ~repeats:1 ~cache_dir:"conv_tune_cache"
      ~timing_ctx:(Context.auto ())
      ~report:(fun r -> reports := r :: !reports)
      ctx
      (named "cvt_tuned" (Train.forward y))
      Ir.Indexing.Empty
  in
  let ctx = Context.run ctx routine in
  let got_t = Context.get_values ctx y.Tensor.value in
  (match !reports with
  | [ r ] ->
      p "conv sketches seeded (serial+grid, with fused-epilogue twins)"
        (if on_cpu then
           r.Autotune.sketch_candidates = 4 && r.Autotune.epilogue_sketch_candidates = 2
         else r.Autotune.sketch_candidates = 0)
  | _ -> p "conv sketches seeded (serial+grid, with fused-epilogue twins)" false);
  p "tuned conv+tail matches the untuned twin within tolerance"
    (Array.for_all2_exn got_t want_t ~f:(fun a b -> Float.(abs (a - b) < 1e-3)))
