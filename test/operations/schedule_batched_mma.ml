(* Batched matmul sketch seeding and tensorization (gh-ocannl-528).

   Three lowered shapes drive the checks:

   - Leading batch: [out[b,s,j] += x[b,s,k] * w[k,j]] — the ffn/logits shape of a transformer
     forward. Detection assigns [i = s] (the deepest loop read by A and absent from B) and records
     [b] as an outer batch loop; the sketch pipelines leave it Serial.
   - Interior batch: [out[b,i,h,j] += att[b,i,h,k] * v[b,k,h,j]] — attention's scores-times-values,
     with the head axis BETWEEN the tile roles in both the output layout and the loop nest. The
     pipelines hoist [h] above [i] with Swaps, and [Tensorize] records leading-dimension strides
     larger than the minor dims ([Tile_mma.ldd]/[lda]/[ldb]) — a wrong stride reads/writes the
     wrong cells, so the bitwise parity against the serial twin is what pins the layout.
   - A variance-style self-product [v[b,s] += x[b,s,k] * x[b,s,k]] — the layer-norm shape whose
     reads mention every loop. It used to be mis-detected as a matmul site (its seeds were the only
     mma candidates gpt2_mini ever proposed, all failing at candidate compile); the role exclusions
     ([j] absent from A, [i] absent from B) must reject it, so it seeds nothing.

   The execution legs run the CPU tensorized sketch end-to-end through the public seeding API
   ([Autotune.sketch_seed_params] with synthetic limits, [Autotune.sketch_schedule]) on the C
   backends, where the register-tiled [Tile_mma] rendering promises bitwise parity for fused-f32
   accumulations. GPU backends keep the seeding checks (pure functions of the lowering) and skip
   the CPU-pipeline execution legs. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module Tn = Ir.Tnode
module LL = Ir.Low_level
module Sched = Ir.Schedule
module Asgns = Ir.Assignments

let () = Utils.settings.output_debug_files_in_build_directory <- true
let p name b = Stdio.printf "%s: %b\n" name b

(* Zeros compare equal to zeros. A fragment mapping that reads outside the staged block, a kernel
   that never ran, or a reference whose own setup silently collapsed all yield all-zeros, and a
   parity check between two zero arrays passes while covering nothing (gh-ocannl-481 item 3). Every
   reference array is pinned nonzero where it is produced, so the parity claims below have content.
   *)
let nonzero name (a : float array) =
  if not (Array.exists a ~f:(fun x -> Float.(x <> 0.))) then
    failwith (name ^ ": the reference is all zeros — the parity checks against it are vacuous");
  a
let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")

(* A leg this backend cannot run still prints its line, so the golden stays backend-uniform — but
   the skip is announced on stderr and marked in the source. A bare [p name true] is
   indistinguishable, both in the transcript and in review, from a verified run: that is how a
   Tensorize leg came to "cover" the gh-ocannl-528 interior-batch bug without ever checking it. *)
let skipped name =
  Stdio.eprintf "SKIPPED on %s (vacuous): %s\n%!" backend_name name;
  p name true

let on_gpu =
  List.exists [ "metal"; "cuda"; "hip" ] ~f:(fun s -> String.is_substring backend_name ~substring:s)

let read_generated base_name =
  let ext =
    if String.is_substring backend_name ~substring:"metal" then ".metal"
    else if String.is_substring backend_name ~substring:"hip" then ".hip"
    else if on_gpu then ".cu"
    else ".c"
  in
  let path = Utils.build_file (base_name ^ ext) in
  if Stdlib.Sys.file_exists path then Some (Stdio.In_channel.read_all path) else None

let named name (comp : Asgns.comp) : Asgns.comp =
  { comp with asgns = Asgns.Block_comment (name, comp.asgns) }

(* Synthetic limits keep the seeding checks machine-independent: a 32-byte vector file for the CPU
   register tiling, and an f32-capable mma capability for the GPU tensorized seeds. *)
let cpu_limits = { Ir.Backend_intf.no_hardware_limits with simd_vector_bytes = 32 }

let gpu_limits =
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
            mma_pipeline_depths = [];
        };
  }

let cpu_mma_seeds opt =
  Autotune.sketch_seed_params ~is_gpu:false ~is_cpu:true ~limits:cpu_limits opt
  |> List.filter ~f:(fun q -> q.Autotune.sk_mma)

let gpu_mma_seeds opt =
  Autotune.sketch_seed_params ~is_gpu:true ~is_cpu:false ~limits:gpu_limits opt
  |> List.filter ~f:(fun q -> q.Autotune.sk_mma)

let compile_serial ~name tensor =
  let ctx, routine =
    Context.compile
      ~lowered_transform:(fun opt -> opt)
      (Context.auto ())
      (named name (Train.forward tensor))
      Ir.Indexing.Empty
  in
  let ctx = Context.run ctx routine in
  nonzero name (Context.get_values ctx tensor.Tensor.value)

(* Compile [tensor] through [transform], capturing the lowering for the seeding assertions. *)
let with_lowering ~name tensor ~(transform : LL.optimized -> LL.optimized) =
  let captured = ref None in
  let ctx, routine =
    Context.compile
      ~lowered_transform:(fun opt ->
        captured := Some opt;
        transform opt)
      (Context.auto ())
      (named name (Train.forward tensor))
      Ir.Indexing.Empty
  in
  (Option.value_exn !captured, ctx, routine)

(* The whole-triple CPU tensorized sketch selected from the public seeding API, executed against
   the serial twin. On GPU backends only the lowering capture runs (identity transform; the
   routine is compiled but not executed). *)
let check_leg ~tag ~serial ~tensorized =
  let want = compile_serial ~name:(tag ^ "_serial") serial in
  if on_gpu then (
    let opt, _ctx, _routine = with_lowering ~name:(tag ^ "_mma") tensorized ~transform:Fn.id in
    p (tag ^ ": cpu mma seeds present") (not (List.is_empty (cpu_mma_seeds opt)));
    skipped (tag ^ " tensorized matches the serial twin bitwise");
    skipped (tag ^ " tensorized structure as expected");
    opt)
  else
    let seed = ref None in
    let transform opt =
      let q =
        List.find_exn (cpu_mma_seeds opt) ~f:(fun q -> q.Autotune.sk_bm = 0 && q.Autotune.sk_bk = 0)
      in
      seed := Some q;
      Sched.apply (Autotune.sketch_schedule ~p:q opt) opt
    in
    let opt, ctx, routine = with_lowering ~name:(tag ^ "_mma") tensorized ~transform in
    p (tag ^ ": cpu mma seeds present") (Option.is_some !seed);
    let ctx = Context.run ctx routine in
    let got = Context.get_values ctx tensorized.Tensor.value in
    p
      (tag ^ " tensorized matches the serial twin bitwise")
      (Array.for_all2_exn got want ~f:Float.equal);
    (match read_generated (tag ^ "_mma") with
    | None -> p (tag ^ " tensorized structure as expected") false
    | Some src ->
        p
          (tag ^ " tensorized structure as expected")
          (String.is_substring src ~substring:"Tile_mma register tiling"));
    opt

let () =
  let bt = 2 and ss = 16 and kk = 8 and jj = 32 in
  (* --- Leading batch: out[b,s,j] += x[b,s,k] * w[k,j] --- *)
  let xv =
    NTDSL.init ~l:"xv" ~prec:Ir.Ops.single ~o:[ bt; ss; kk ]
      ~f:(fun idcs ->
        Float.of_int (((idcs.(0) * ss * kk) + (idcs.(1) * kk) + idcs.(2)) % 13) *. 0.25)
      ()
  in
  let wv =
    NTDSL.init ~l:"wv" ~prec:Ir.Ops.single ~o:[ kk; jj ]
      ~f:(fun idcs -> (Float.of_int (((idcs.(0) * jj) + idcs.(1)) % 17) -. 8.) *. 0.5)
      ()
  in
  let%op lb0 = xv +* "bsk;kj=>bsj" wv in
  let%op lb1 = xv +* "bsk;kj=>bsj" wv in
  let opt_lb = check_leg ~tag:"batched_lb" ~serial:lb0 ~tensorized:lb1 in
  p "leading-batch: gpu mma seeds present" (not (List.is_empty (gpu_mma_seeds opt_lb)));

  (* --- Interior batch: out[b,i,h,j] += att[b,i,h,k] * v[b,k,h,j] --- *)
  let hh = 2 in
  let att =
    NTDSL.init ~l:"att" ~prec:Ir.Ops.single ~o:[ bt; ss; hh; kk ]
      ~f:(fun idcs ->
        Float.of_int
          (((idcs.(0) * ss * hh * kk) + (idcs.(1) * hh * kk) + (idcs.(2) * kk) + idcs.(3)) % 11)
        *. 0.125)
      ()
  in
  let vv =
    NTDSL.init ~l:"vv" ~prec:Ir.Ops.single ~o:[ bt; kk; hh; jj ]
      ~f:(fun idcs ->
        (Float.of_int
           (((idcs.(0) * kk * hh * jj) + (idcs.(1) * hh * jj) + (idcs.(2) * jj) + idcs.(3)) % 7)
        -. 3.)
        *. 0.5)
      ()
  in
  let%op ib0 = att +* "bihk;bkhj=>bihj" vv in
  let%op ib1 = att +* "bihk;bkhj=>bihj" vv in
  let opt_ib = check_leg ~tag:"batched_ib" ~serial:ib0 ~tensorized:ib1 in
  let gpu_seeds_ib = gpu_mma_seeds opt_ib in
  p "interior-batch: gpu mma seeds present" (not (List.is_empty gpu_seeds_ib));
  p "interior-batch: gpu staged sketch builds"
    (match
       List.find gpu_seeds_ib ~f:(fun q -> q.Autotune.sk_bk > 0)
       |> Option.map ~f:(fun q -> Autotune.sketch_schedule ~p:q opt_ib)
     with
    | Some (_ :: _) -> true
    | Some [] | None -> false
    | exception _ -> false);

  (* --- Variance-style self-product: must not be detected as a matmul site --- *)
  let x2 =
    NTDSL.init ~l:"x2" ~prec:Ir.Ops.single ~o:[ bt; ss; kk ]
      ~f:(fun idcs ->
        Float.of_int (((idcs.(0) * ss * kk) + (idcs.(1) * kk) + idcs.(2)) % 5) *. 0.5)
      ()
  in
  let%op var = x2 +* "bsk;bsk=>bs" x2 in
  let opt_var, _ctx, _routine = with_lowering ~name:"batched_var" var ~transform:Fn.id in
  p "variance-like site: no cpu mma seeds" (List.is_empty (cpu_mma_seeds opt_var));
  p "variance-like site: no gpu mma seeds" (List.is_empty (gpu_mma_seeds opt_var));

  (* --- The tensor-core legs, at bf16, against the backend's REAL advertised capability ---

     Everything above runs the batched sites through synthetic limits, which keeps the seeding
     checks machine-independent but leaves the question this file exists for unanswered on a GPU:
     the recorded leading-dimension strides ([Tile_mma.ldd]/[lda]/[ldb]) are consumed by the
     backends' own mma hooks, and a stride that is right for the register-tiled C rendering can
     still be wrong for a fragment load. On a real device an f32 site seeds nothing on the wmma
     backends (RDNA3.5 and CUDA have no f32 operand shape), so the reachable format is a narrow
     one; bf16 is the one both wmma backends and Metal have in the uniform combination.

     So: seed against [Context.hardware_limits], apply the real pipeline, execute, and require both
     that the values match the serial twin and that the emitted source carries the backend's
     intrinsic. The interior-batch leg is the load-bearing one — its [lda]/[ldb]/[ldd] are
     [hh]-times larger than the minor dims, which is exactly the case gh-ocannl-528 introduced and
     the case a fragment load addresses differently from a scalar loop.

     Tolerance rather than bitwise on HIP, for the reason [schedule_mma_matmul] documents at
     length: gfx1151's WMMA does not return the exactly-rounded dot product in any format
     combination, reproducibly so outside OCANNL. It is loose enough to admit that (worst measured
     5.86e-03 in the uniform-bf16 combination) and far tighter than a stride defect, which moves
     values by O(1). *)
  let real_limits = Context.hardware_limits (Context.auto ()) in
  let has_uniform_bf16_tile =
    match real_limits.Ir.Backend_intf.mma with
    | None -> false
    | Some cap ->
        List.exists cap.Ir.Backend_intf.mma_format_tiles ~f:(fun ((a, b, d), _) ->
            Ir.Backend_intf.equal_mma_input_format a Ir.Backend_intf.Mma_bf16
            && Ir.Backend_intf.equal_mma_input_format b Ir.Backend_intf.Mma_bf16
            && Ir.Backend_intf.equal_mma_input_format d Ir.Backend_intf.Mma_bf16)
  in
  let renders_intrinsic src =
    let has s = String.is_substring src ~substring:s in
    if String.is_substring backend_name ~substring:"metal" then has "simdgroup_bfloat8x8"
    else if String.is_substring backend_name ~substring:"hip" then has "rocwmma::mma_sync"
    else if String.is_substring backend_name ~substring:"cuda" then has "mma.sync.aligned.m16n8k16"
    else false
  in
  let close a b = Float.(abs (a - b) <= 0.05 * max 1. (abs b)) in
  (* At most this many candidates are executed per site: each one is a full backend compile, and on
     HIP the rocWMMA headers make that expensive. Reported on stderr so the bound is never silent. *)
  let max_candidates = 4 in
  let bf16_leg ~tag ~build =
    let ref_t = build () in
    let want =
      let ctx, routine =
        Context.compile
          ~lowered_transform:(fun opt -> opt)
          (Context.auto ())
          (named (tag ^ "_bf16_serial") (Train.forward ref_t))
          Ir.Indexing.Empty
      in
      nonzero (tag ^ "_bf16_serial") (Context.get_values (Context.run ctx routine) ref_t.Tensor.value)
    in
    let cand = build () in
    let fwd = named (tag ^ "_bf16_mma") (Train.forward cand) in
    let captured = ref None in
    let _ctx, _r =
      Context.compile
        ~lowered_transform:(fun opt ->
          captured := Some opt;
          opt)
        (Context.auto ()) fwd Ir.Indexing.Empty
    in
    let opt = Option.value_exn ~here:[%here] !captured in
    let seeds =
      Autotune.sketch_seed_params ~is_gpu:true ~is_cpu:false ~limits:real_limits opt
      |> List.filter ~f:(fun q -> q.Autotune.sk_mma)
    in
    if not has_uniform_bf16_tile then
      Stdio.eprintf
        "%s: %s advertises no uniform-bf16 mma tile — the tensor-core checks below are vacuous\n"
        tag backend_name
    else if List.length seeds > max_candidates then
      Stdio.eprintf "%s: executing %d of %d bf16 mma seeds (compile cost)\n" tag max_candidates
        (List.length seeds);
    let n_ran = ref 0 and n_close = ref 0 and n_intrinsic = ref 0 in
    List.iteri seeds ~f:(fun idx q ->
        if idx < max_candidates then
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
              if Array.for_all2_exn got want ~f:close then Int.incr n_close;
              if Option.value_map (read_generated (tag ^ "_bf16_mma")) ~default:false
                   ~f:renders_intrinsic
              then Int.incr n_intrinsic
          | exception _ -> ());
    p (tag ^ " bf16: the backend's advertised tile is seeded")
      ((not has_uniform_bf16_tile) || not (List.is_empty seeds));
    p (tag ^ " bf16: some candidate compiles and runs")
      ((not has_uniform_bf16_tile) || !n_ran >= 1);
    p (tag ^ " bf16: every running candidate matches the serial twin") (!n_ran = !n_close);
    p (tag ^ " bf16: some running candidate renders the tensor-core intrinsic")
      ((not has_uniform_bf16_tile) || !n_intrinsic >= 1)
  in
  let bf16_init ~l ~o ~f = NTDSL.init ~l ~prec:Ir.Ops.bfloat16 ~o ~f () in
  (* Products are multiples of 1/8 and every partial sum is bounded by 16, so the serial twin is
     exact and any deviation the tolerance admits is the tensor core's own. *)
  let ss2 = 32 and kk2 = 32 and jj2 = 32 in
  let xb =
    bf16_init ~l:"xb" ~o:[ bt; ss2; kk2 ] ~f:(fun idcs ->
        Float.of_int (((idcs.(0) * ss2 * kk2) + (idcs.(1) * kk2) + idcs.(2)) % 3) *. 0.25)
  in
  let wb =
    bf16_init ~l:"wb" ~o:[ kk2; jj2 ] ~f:(fun idcs ->
        (Float.of_int (((idcs.(0) * jj2) + idcs.(1)) % 5) -. 2.) *. 0.5)
  in
  bf16_leg ~tag:"batched_lb" ~build:(fun () ->
      let%op t = xb +* "bsk;kj=>bsj" wb in
      Tn.update_prec t.Tensor.value Ir.Ops.bfloat16;
      t);
  (* The interior-batch shape: [h] sits between the tile roles, so [lda]/[ldb]/[ldd] are all
     [hh]-times the minor dim (64 here, against minor dims of 32). *)
  let attb =
    bf16_init ~l:"attb" ~o:[ bt; ss2; hh; kk2 ] ~f:(fun idcs ->
        Float.of_int
          (((idcs.(0) * ss2 * hh * kk2) + (idcs.(1) * hh * kk2) + (idcs.(2) * kk2) + idcs.(3)) % 3)
        *. 0.25)
  in
  let vb =
    bf16_init ~l:"vb" ~o:[ bt; kk2; hh; jj2 ] ~f:(fun idcs ->
        (Float.of_int
           (((idcs.(0) * kk2 * hh * jj2) + (idcs.(1) * hh * jj2) + (idcs.(2) * jj2) + idcs.(3)) % 5)
        -. 2.)
        *. 0.5)
  in
  bf16_leg ~tag:"batched_ib" ~build:(fun () ->
      let%op t = attb +* "bihk;bkhj=>bihj" vb in
      Tn.update_prec t.Tensor.value Ir.Ops.bfloat16;
      t)
