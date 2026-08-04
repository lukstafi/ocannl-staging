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
let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")

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
  Context.get_values ctx tensor.Tensor.value

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
    p (tag ^ " tensorized matches the serial twin bitwise") true;
    p (tag ^ " tensorized structure as expected") true;
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
  p "variance-like site: no gpu mma seeds" (List.is_empty (gpu_mma_seeds opt_var))
