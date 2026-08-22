(* gh-ocannl-683: a contraction over several axes is a matmul site.

   Attention's out projection [{ w_o } * attn] contracts over the weight's two input axes (head,
   head_dim), so its lowering is [d[b,s,j] += w[j,h,e] * x[b,s,h,e]] — a reduction NEST of two
   loops. The matmul matcher took the single innermost loop as [k] and required every other loop
   to own an axis of [d], so the head loop refused the site: no matmul family was ever seeded
   there, and the kernel shipped as an untiled global-accumulator nest at 8 blocks (22% of the
   gpt2_mini step on gfx1151 at 9% of sgemm peak).

   Mechanism under test: the contraction nest is the maximal innermost suffix of loops absent from
   the accumulator ([classify_matmul]); its innermost loop is [m_k], the rest are [m_ko] — k-loops
   lowering has already split. Every pipeline treats them as k-block loops above the one its own
   k-split mints ([Sketch_families.k_blocks]): sunk below the output roles, the staged tiles
   reloaded at, the accumulator privatized over the outermost. Single-axis sites have an empty
   [m_ko] and keep byte-identical schedules, which the existing sketch suites pin.

   Two lowered shapes: the out projection itself (the [*] operator on a weight with two input
   axes, the issue's form), and a three-axis contraction whose materialized output feeds a
   bias+relu companion nest (companion coverage and epilogue twins on a multi-axis site).

   Executed assertions compare every candidate against a serial reference computed from the same
   discriminating inputs; the values vary with every index and keep all partial sums exactly
   representable in f32, so bitwise equality is required regardless of the accumulation order a
   tiling imposes. GPU backends execute the blocktile family (workgroup-shared staging); cc
   executes the CPU families (packed and register-tiled pipelines included), so every backend
   executes a multi-axis-contraction sketch. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module LL = Ir.Low_level
module Sched = Ir.Schedule
module Asgns = Ir.Assignments

let () = Utils.settings.output_debug_files_in_build_directory <- true
let p = Verdict.p
let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")
let skipped = Verdict.skipped ~backend:backend_name

let on_gpu =
  List.exists [ "metal"; "cuda"; "hip" ] ~f:(fun s -> String.is_substring backend_name ~substring:s)

module Generated = Test_utils.Generated

let () = Generated.init ~backend_name

let named name (comp : Asgns.comp) : Asgns.comp =
  { comp with asgns = Asgns.Block_comment (name, comp.asgns) }

(* Zeros compare equal to zeros: pin every reference nonzero so the parity claims have content. *)
let nonzero name (a : float array) =
  if not (Array.exists a ~f:(fun x -> Float.(x <> 0.))) then
    failwith (name ^ ": the reference is all zeros — the parity checks against it are vacuous");
  a

let compile_serial ~name tensor =
  let ctx, routine =
    Context.compile
      ~lowered_transform:(fun opt -> opt)
      (Context.auto ())
      (named name (Train.forward tensor))
      Ir.Indexing.Empty
  in
  nonzero name (Context.get_values (Context.run ctx routine) tensor.Tensor.value)

let capture fwd =
  let captured = ref None in
  let _ctx, _r =
    Context.compile
      ~lowered_transform:(fun opt ->
        captured := Some opt;
        opt)
      (Context.auto ()) fwd Ir.Indexing.Empty
  in
  Option.value_exn ~here:[%here] !captured

(* The unfused seeds of a family, via the public seeding API. Synthetic no-limits keep the
   enumeration machine-independent. *)
let unfused_seeds ~is_gpu ~is_cpu ~limits opt =
  Autotune.sketch_seed_params ~is_gpu ~is_cpu ~limits opt
  |> List.filter ~f:(fun q -> not q.Autotune.sk_epilogue)

(* A synthetic f32 mma capability makes the tensorized GPU branch seedable machine-independently
   (an f32 tile is not a hardware format on the wmma backends, so the real capability would refute
   it); one pipelined depth so the pipelined staged twins construct over a k-block nest too. *)
let mma_limits =
  {
    Ir.Backend_intf.no_hardware_limits with
    mma =
      Some
        {
          Ir.Backend_intf.mma_simd_width = 32;
          mma_tile = (8, 8, 8);
          mma_format_tiles =
            [
              ( (Ir.Backend_intf.Mma_f32, Ir.Backend_intf.Mma_f32, Ir.Backend_intf.Mma_f32),
                (8, 8, 8) );
            ];
          mma_staged_layouts = [];
          mma_pipeline_depths = [ 2 ];
        };
  }

(* Every seed's schedule applied as the pure IR transform it is — backend-independent. Returns
   the seeds whose schedule constructs and validates, and the ones that construct but fail
   [validate_parallel]; a construction failure is reported and counted against the claim. *)
let constructs_and_validates ~tag ~what seeds opt =
  let ok = ref true in
  let valid, invalid =
    List.partition_tf seeds ~f:(fun q ->
        match Sched.apply (Autotune.sketch_schedule ~p:q opt) opt with
        | o -> (
            match LL.validate_parallel o.LL.optimize_ctx.LL.placements o.LL.llc with
            | () -> true
            | exception _ -> false)
        | exception exn ->
            Stdio.eprintf "%s/%s: construct FAILED: %s\n" tag what (Exn.to_string exn);
            ok := false;
            false)
  in
  p (Printf.sprintf "%s: %s seeds are proposed" tag what) (not (List.is_empty seeds));
  p (Printf.sprintf "%s: every %s seed's schedule constructs" tag what) !ok;
  (valid, invalid)

let binds_hardware q opt =
  let o = Sched.apply (Autotune.sketch_schedule ~p:q opt) opt in
  let dims = LL.launch_dims o.LL.llc in
  let product = Array.fold ~init:1 ~f:( * ) in
  (product dims.LL.grid, product dims.LL.block)

(* Execute every seed against the serial reference, each under its own armed artifact. *)
let execute_seeds ~tag ~what ~fwd ~cand ~want seeds =
  let n_ran = ref 0 and n_match = ref 0 in
  List.iter seeds ~f:(fun q ->
      Generated.arm (tag ^ "_sched");
      match
        let ctx, routine =
          Context.compile
            ~lowered_transform:(fun o -> Sched.apply (Autotune.sketch_schedule ~p:q o) o)
            (Context.auto ()) fwd Ir.Indexing.Empty
        in
        Context.get_values (Context.run ctx routine) cand.Tensor.value
      with
      | got ->
          Int.incr n_ran;
          if Array.for_all2_exn got want ~f:Float.equal then Int.incr n_match
      | exception exn -> Stdio.eprintf "%s/%s: seed FAILED: %s\n" tag what (Exn.to_string exn));
  p
    (Printf.sprintf "%s: every %s seed compiles and runs" tag what)
    (!n_ran = List.length seeds && !n_ran > 0);
  p
    (Printf.sprintf "%s: every %s candidate matches the serial reference bitwise" tag what)
    (!n_ran = !n_match)

(* One leg. [ko_extents] are the expected extents of the outer contraction loops (nest order) and
   [nk] the innermost contraction extent. *)
let leg ~tag ~ko_extents ~nk ?(companion = false) ~build () =
  let want = compile_serial ~name:(tag ^ "_serial") (build ()) in
  let cand = build () in
  let fwd = named (tag ^ "_sched") (Train.forward cand) in
  let opt = capture fwd in
  (match Autotune.detect_matmul opt.LL.llc with
  | None ->
      p (tag ^ ": the multi-axis contraction is detected as a matmul site") false;
      p (tag ^ ": the outer contraction loops carry the expected extents") false;
      p (tag ^ ": m_k is the innermost contraction loop") false;
      p (tag ^ ": the accumulation is in fused form") false
  | Some site ->
      p (tag ^ ": the multi-axis contraction is detected as a matmul site") true;
      p
        (tag ^ ": the outer contraction loops carry the expected extents")
        (List.equal Int.equal (List.map site.Autotune.m_ko ~f:snd) ko_extents);
      p (tag ^ ": m_k is the innermost contraction loop") (site.Autotune.m_nk = nk);
      p (tag ^ ": the accumulation is in fused form") site.Autotune.m_fma);
  (* --- GPU families: structure everywhere. --- *)
  let gpu_seeds =
    unfused_seeds ~is_gpu:true ~is_cpu:false ~limits:Ir.Backend_intf.no_hardware_limits opt
  in
  let _, gpu_invalid = constructs_and_validates ~tag ~what:"GPU blocktile" gpu_seeds opt in
  p (tag ^ ": every GPU blocktile seed validates") (List.is_empty gpu_invalid);
  (* The geometry the untiled kernel never had: every seed tiles the output across a workgroup,
     and the batch-grid twins spread the batch across blocks (a 64x64 block tile of a 64x64 site is
     one block, legitimately). *)
  p
    (tag ^ ": every GPU blocktile seed binds a multi-thread workgroup")
    (List.for_all gpu_seeds ~f:(fun q -> snd (binds_hardware q opt) > 1));
  p
    (tag ^ ": every GPU batch-grid twin launches more than one block")
    (List.exists gpu_seeds ~f:(fun q -> q.Autotune.sk_batch_grid)
    && List.for_all gpu_seeds ~f:(fun q ->
        (not q.Autotune.sk_batch_grid) || fst (binds_hardware q opt) > 1));
  let mma_seeds =
    unfused_seeds ~is_gpu:true ~is_cpu:false ~limits:mma_limits opt
    |> List.filter ~f:(fun q -> q.Autotune.sk_mma)
  in
  let _, mma_invalid = constructs_and_validates ~tag ~what:"GPU tensorized" mma_seeds opt in
  p (tag ^ ": every GPU tensorized seed validates") (List.is_empty mma_invalid);
  p
    (tag ^ ": the tensorized seeds include unstaged, staged and pipelined-staged geometries")
    (List.exists mma_seeds ~f:(fun q -> q.Autotune.sk_bk = 0)
    && List.exists mma_seeds ~f:(fun q -> q.Autotune.sk_bk > 0 && q.Autotune.sk_depth = 1)
    && List.exists mma_seeds ~f:(fun q -> q.Autotune.sk_depth = 2));
  (* --- CPU families: structure everywhere, executed on cc. --- *)
  let cpu_limits =
    if on_gpu then Ir.Backend_intf.no_hardware_limits
    else Context.hardware_limits (Context.auto ())
  in
  let cpu_seeds = unfused_seeds ~is_gpu:false ~is_cpu:true ~limits:cpu_limits opt in
  let cpu_valid, cpu_invalid = constructs_and_validates ~tag ~what:"CPU" cpu_seeds opt in
  (* The CPU pipelines carry no companion coverage ([companion_geometry] is consulted by the GPU
     pipelines only), so on a site with a companion nest the pool-parallel CPU shapes — the ones
     binding a Grid dimension — decline at validation and are skipped, exactly as on a
     single-axis site; the all-serial shapes validate. *)
  p
    (tag ^ ": every CPU seed binding no hardware dimension validates")
    (List.for_all cpu_invalid ~f:(fun q -> fst (binds_hardware q opt) > 1));
  if not companion then p (tag ^ ": every CPU seed validates") (List.is_empty cpu_invalid)
  else if on_gpu then
    (* Built under no-limits here, the CPU family has no Grid shapes to decline. *)
    skipped (tag ^ ": the Grid-bound CPU shapes decline on the uncovered companion")
  else
    p
      (tag ^ ": the Grid-bound CPU shapes decline on the uncovered companion")
      (not (List.is_empty cpu_invalid));
  if on_gpu then begin
    execute_seeds ~tag ~what:"GPU blocktile" ~fwd ~cand ~want gpu_seeds;
    (* The seed that shipped untiled before: its kernel now carries a workgroup-shared tile. *)
    let shared =
      if String.is_substring backend_name ~substring:"metal" then "threadgroup " else "__shared__"
    in
    Generated.arm (tag ^ "_sched");
    let q = List.hd_exn gpu_seeds in
    let _ctx, _r =
      Context.compile
        ~lowered_transform:(fun o -> Sched.apply (Autotune.sketch_schedule ~p:q o) o)
        (Context.auto ()) fwd Ir.Indexing.Empty
    in
    Generated.assert_emits ~routine:(tag ^ "_sched") ~contains:shared
      (tag ^ ": the blocktiled kernel stages operands through workgroup-shared tiles");
    Stdio.eprintf "%s: %s executes the GPU families; the CPU families are structural here\n" tag
      backend_name;
    skipped (tag ^ ": the CPU seeds include the register-tiled packed pipelines");
    skipped (tag ^ ": every CPU seed compiles and runs");
    skipped (tag ^ ": every CPU candidate matches the serial reference bitwise")
  end
  else begin
    Stdio.eprintf "%s: %s cannot execute workgroup-shared staging — GPU execution legs skipped\n"
      tag backend_name;
    skipped (tag ^ ": every GPU blocktile seed compiles and runs");
    skipped (tag ^ ": every GPU blocktile candidate matches the serial reference bitwise");
    skipped (tag ^ ": the blocktiled kernel stages operands through workgroup-shared tiles");
    (* The whole-triple form is refuted on the weight's transposed storage ([j, ..., k]), as on
       any [j,k]-stored weight; the packed forms normalize the layout and are seeded. *)
    p
      (tag ^ ": the CPU seeds include the register-tiled packed pipelines")
      (List.exists cpu_seeds ~f:(fun q -> q.Autotune.sk_mma && q.Autotune.sk_bk > 0));
    execute_seeds ~tag ~what:"CPU" ~fwd ~cand ~want cpu_valid
  end

let () =
  (* --- The out projection: [{ w_o } * attn] with two input axes on the weight. --- *)
  (* Discriminating inputs: values vary with every index (the linear-index strides are coprime
     with the moduli) and every product is a small multiple of 1/8 with partial sums far below
     2^24, so f32 addition is exact in any order. *)
  let bb = 2 and ss = 64 and jj = 64 and hh = 4 and ee = 16 in
  let w () =
    NTDSL.init ~l:"cn_w" ~prec:Ir.Ops.single ~o:[ jj ] ~i:[ hh; ee ]
      ~f:(fun idcs ->
        (Float.of_int (((idcs.(0) * hh * ee) + (idcs.(1) * ee) + idcs.(2)) % 11) -. 5.) *. 0.5)
      ()
  in
  let att () =
    NTDSL.init ~l:"cn_att" ~prec:Ir.Ops.single ~b:[ bb; ss ] ~o:[ hh; ee ]
      ~f:(fun idcs ->
        Float.of_int
          (((idcs.(0) * ss * hh * ee) + (idcs.(1) * hh * ee) + (idcs.(2) * ee) + idcs.(3)) % 13)
        *. 0.25)
      ()
  in
  leg ~tag:"out_proj" ~ko_extents:[ hh ] ~nk:ee ~build:(fun () ->
      let wv = w () and av = att () in
      let%op out = wv * av in
      out)
    ();

  (* --- A three-axis contraction with a materialized output feeding a bias+relu companion. --- *)
  let bb2 = 2 and ss2 = 32 and jj2 = 64 and gg = 2 and hh2 = 2 and ee2 = 16 in
  let x () =
    NTDSL.init ~l:"cn_x" ~prec:Ir.Ops.single ~o:[ bb2; ss2; gg; hh2; ee2 ]
      ~f:(fun idcs ->
        Float.of_int
          (((idcs.(0) * ss2 * gg * hh2 * ee2)
           + (idcs.(1) * gg * hh2 * ee2)
           + (idcs.(2) * hh2 * ee2)
           + (idcs.(3) * ee2)
           + idcs.(4))
          % 13)
        *. 0.25)
      ()
  in
  let w3 () =
    NTDSL.init ~l:"cn_w3" ~prec:Ir.Ops.single ~o:[ jj2; gg; hh2; ee2 ]
      ~f:(fun idcs ->
        (Float.of_int
           (((idcs.(0) * gg * hh2 * ee2) + (idcs.(1) * hh2 * ee2) + (idcs.(2) * ee2) + idcs.(3))
           % 11)
        -. 5.)
        *. 0.5)
      ()
  in
  let bias () =
    NTDSL.init ~l:"cn_bias" ~prec:Ir.Ops.single ~o:[ jj2 ]
      ~f:(fun idcs -> (Float.of_int (idcs.(0) % 3) -. 1.) *. 0.5)
      ()
  in
  leg ~tag:"three_axis_companion" ~ko_extents:[ gg; hh2 ] ~nk:ee2 ~companion:true ~build:(fun () ->
      let xv = x () and wv = w3 () and bv = bias () in
      let%op z = xv +* "bsghe;jghe=>bsj" wv in
      Train.set_materialized z.Tensor.value;
      let%op y = relu (z + bv) in
      y)
    ()
