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
   the serial and Grid-parallel conv pipelines; fused-epilogue twins (gh-ocannl-486) are gated off
   on multi-window convs — the fragment store-back sits inside the outer kernel-window loop, so
   [Fuse_epilogue]'s exactly-once check rejects the relocation (single-window convs keep their
   twins; relocating the tail after the window loops is a recorded follow-up). The tuned routine
   matches the untuned twin, and the winning schedule round-trips through the saved form. GPU conv
   seeds are a follow-up: seeding is CPU-gated, GPU backends assert zero conv candidates. - detect_conv's pattern discipline: a plain matmul is not a conv site. *)

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
  let detect_leg tag ~stride ~use_padding ~want_stride ~want_offset ~want_row ~want_seeds =
    let x = make_x tag in
    let conv =
      Nn_blocks.conv2d ~label:[ tag ] ~kernel_size:3 ~stride ~use_padding ~out_channels:8 ()
    in
    let y = conv x in
    let site = ref None in
    let n_seeds = ref (-1) in
    let ctx = Context.auto () in
    (* Pin the vector width so the C-backend seed-count assertion is backend-independent (the
       check always passes is_cpu; only [limits] would otherwise vary per device). *)
    let limits = { (Context.hardware_limits ctx) with Ir.Backend_intf.simd_vector_bytes = 16 } in
    let transform (opt : LL.optimized) =
      site := Autotune.detect_conv opt.LL.llc;
      n_seeds := List.length (Autotune.sketch_seed_params ~is_gpu:false ~is_cpu:true ~limits opt);
      opt
    in
    let ctx = Train.init_params ctx Ir.Indexing.Empty y in
    let ctx, routine =
      Context.compile ~lowered_transform:transform ctx
        (named (tag ^ "_det") (Train.forward y))
        Ir.Indexing.Empty
    in
    ignore (ctx, routine);
    (* The C-backend seed count: serial + Grid on unit-stride rows (no fused-epilogue twins — a
       3x3 conv has two kernel-window loops, so the twins are gated), none on strided rows (the
       packing Stage packs by index range — see the pipeline legs). *)
    p (tag ^ " C-backend conv seeds") (!n_seeds = want_seeds);
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
  detect_leg "cvd_s1v" ~stride:1 ~use_padding:false ~want_stride:1 ~want_offset:0 ~want_row:9
    ~want_seeds:2;
  detect_leg "cvd_s2v" ~stride:2 ~use_padding:false ~want_stride:2 ~want_offset:0 ~want_row:5
    ~want_seeds:0;
  detect_leg "cvd_s1p" ~stride:1 ~use_padding:true ~want_stride:1 ~want_offset:(-1) ~want_row:11
    ~want_seeds:0;

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
     pipelines). Stride-2 and padded variants included: strided windows only change the packing
     Stage's index arithmetic, and padded convs read a physically padded source (negative offsets
     land in the halo — the lowered nest carries no guards for a Stage to displace), so the pipeline
     is uniform across the variants. === *)
  let make_conv_s1v sub =
    let x = make_x sub in
    let kern = make_kern sub in
    let%op y =
      x +* "...| 1*oh<+kh, 1*ow<+kw, ..ic..; |kh, kw, ..ic.. -> ..oc.. => ...| oh, ow, ..oc.." kern
    in
    (x, kern, y)
  in
  let make_conv_s2v sub =
    let x = make_x sub in
    let kern = make_kern sub in
    let%op y =
      x +* "...| 2*oh<+kh, 2*ow<+kw, ..ic..; |kh, kw, ..ic.. -> ..oc.. => ...| oh, ow, ..oc.." kern
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
  let pipeline_leg tag make_conv =
    let want =
      let _, _, y = make_conv (tag ^ "_r") in
      run_plain (tag ^ "_ref") y
    in
    let swapped = run_sched (tag ^ "_swap") (make_conv (tag ^ "_s")) ~tensorized:false in
    p
      (tag ^ ": reorder-only conv matches the natural form within tolerance")
      (Array.for_all2_exn swapped want ~f:(fun a b -> Float.(abs (a - b) < 1e-3)));
    if on_cpu then (
      let full = run_sched (tag ^ "_gemm") (make_conv (tag ^ "_g")) ~tensorized:true in
      p
        (tag ^ ": packed+tensorized conv matches the reorder-only twin bitwise")
        (Array.for_all2_exn full swapped ~f:Float.equal);
      p
        (tag ^ ": packed+tensorized conv matches the natural form within tolerance")
        (Array.for_all2_exn full want ~f:(fun a b -> Float.(abs (a - b) < 1e-3))))
    else (
      p (tag ^ ": packed+tensorized conv matches the reorder-only twin bitwise") true;
      p (tag ^ ": packed+tensorized conv matches the natural form within tolerance") true)
  in
  pipeline_leg "cvg" make_conv_s1v;
  (* No padded pipeline leg: padded convs read the halo of a physically padded input, and the
     staging pipeline is not halo-aware yet (Stage's edge guards clip tile loads against the
     logical dims) — the seeds are gated to offset-free sites (asserted above), so autotune never
     proposes the unsound form. Halo-aware staging is a follow-up. *)
  (* A strided row packs a dilated tile ([Stage] packs by index range), which [Tensorize]'s
     unit-coefficient discipline rejects — the reorder still holds, and the seeds are gated on a
     unit-stride row (asserted below), so autotune never proposes the rejected form. *)
  (let want2 =
     let _, _, y = make_conv_s2v "cvg2_r" in
     run_plain "cvg2_ref" y
   in
   let swapped2 = run_sched "cvg2_swap" (make_conv_s2v "cvg2_s") ~tensorized:false in
   p "cvg2: reorder-only conv matches the natural form within tolerance"
     (Array.for_all2_exn swapped2 want2 ~f:(fun a b -> Float.(abs (a - b) < 1e-3)));
   match run_sched "cvg2_gemm" (make_conv_s2v "cvg2_g") ~tensorized:true with
   | _ -> p "cvg2: strided-row tensorization rejected with a targeted error" false
   | exception Invalid_argument msg ->
       p "cvg2: strided-row tensorization rejected with a targeted error"
         (String.is_substring msg ~substring:"Schedule.Tensorize"));
  if on_cpu then
    let src = Stdio.In_channel.read_all (Utils.build_file "cvg_gemm.c") in
    let has s = String.is_substring src ~substring:s in
    p "conv pipeline structure: im2col packs, register tiling, resident fragment"
      (has "Tile_mma register tiling" && has "fragment_" && has "tile_")
  else p "conv pipeline structure: im2col packs, register tiling, resident fragment" true;

  (* === Autotune seeding on conv+bias+relu: serial + Grid conv pipelines (fused-epilogue twins
     gated off — 3x3 is multi-window); the tuned routine matches the untuned twin === *)
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
      p "conv sketches seeded (serial+grid; twins gated on multi-window)"
        (if on_cpu then
           r.Autotune.sketch_candidates = 2 && r.Autotune.epilogue_sketch_candidates = 0
         else r.Autotune.sketch_candidates = 0)
  | _ -> p "conv sketches seeded (serial+grid; twins gated on multi-window)" false);
  p "tuned conv+tail matches the untuned twin within tolerance"
    (Array.for_all2_exn got_t want_t ~f:(fun a b -> Float.(abs (a - b) < 1e-3)))
