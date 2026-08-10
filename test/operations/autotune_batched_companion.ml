(* gh-ocannl-569: the companion-coverage rule (gh-ocannl-521) must not decline a BATCHED site.

   The workload is the shape of gpt2's FFN up-projection: a rank-3 GEMM (batch x tokens x features,
   the `8x128x1024` geometry of the gh-531 profile) whose materialized output feeds an elementwise
   companion nest (bias + relu) in the same routine. The site's chain is three loops, but
   {!Sched.aligned_chains} used to cap every nest's chain at two — so [companion_geometry]'s
   full-arity check could never match a rank-3 site, and every GPU sketch seed (scalar AND
   tensorized) declined with "the accumulation nest's aligned chain was trimmed below its
   8x128x1024 geometry". That single decline pinned five FFN-class kernels to ~1.3% of fp32 peak —
   70% of the gpt2_mini step on CUDA, 47% on HIP.

   Two layers of pinning:
   - Structural (backend-independent, runs on cc too): the scalar GPU seeds' schedules CONSTRUCT —
     [Autotune.sketch_schedule] does not raise the companion-coverage decline — and the constructed
     schedule spreads the minor output axis (j) across [Grid] blocks.
   - Executable (GPU backends): each unfused scalar GPU seed compiles and computes correctly. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module Sched = Ir.Schedule
module LL = Ir.Low_level
module Asgns = Ir.Assignments

let p name b = Stdio.printf "%s: %b\n" name b

let named name (comp : Asgns.comp) : Asgns.comp =
  { comp with asgns = Asgns.Block_comment (name, comp.asgns) }

let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")
let is_gpu = Sched.backend_is_gpu backend_name

let clean_cache dir =
  if Stdlib.Sys.file_exists dir && Stdlib.Sys.is_directory dir then
    Array.iter (Stdlib.Sys.readdir dir) ~f:(fun f ->
        Stdlib.Sys.remove (Stdlib.Filename.concat dir f))

let () =
  clean_cache "batched_companion_cache";
  let b = 4 and n = 32 and m = 64 and k = 16 in
  let xv = Array.init (b * n * k) ~f:(fun i -> Float.of_int (i % 7) *. 0.25) in
  let wv = Array.init (k * m) ~f:(fun i -> Float.of_int (i % 5) *. 0.125) in
  let bv = Array.init m ~f:(fun i -> Float.of_int (i % 3) -. 1.) in
  let expected =
    Array.init (b * n * m) ~f:(fun idx ->
        let bi = idx / (n * m) in
        let i = idx % (n * m) / m and j = idx % m in
        let acc = ref 0. in
        for kk = 0 to k - 1 do
          acc := !acc +. (xv.((bi * n * k) + (i * k) + kk) *. wv.((kk * m) + j))
        done;
        Float.max 0. (!acc +. bv.(j)))
  in
  let x = TDSL.ndarray xv ~label:[ "bc_x" ] ~batch_dims:[ b ] ~output_dims:[ n; k ] () in
  let w = TDSL.ndarray wv ~label:[ "bc_w" ] ~output_dims:[ k; m ] () in
  let bias = TDSL.ndarray bv ~label:[ "bc_b" ] ~output_dims:[ m ] () in
  let%op z = x +* "b|ik;kj=>b|ij" w in
  Train.set_materialized z.Tensor.value;
  let%op y = relu (z + bias) in
  let comp = named "bc_head" (Train.forward y) in

  (* Capture the lowered routine once; enumerate GPU seeds regardless of the running backend —
     schedule CONSTRUCTION (where the gh-521 companion-coverage decline fires) is
     backend-independent. *)
  let limits = Context.hardware_limits (Context.auto ()) in
  let captured = ref None in
  let _ctx, _r =
    Context.compile
      ~lowered_transform:(fun opt ->
        captured := Some opt;
        opt)
      (Context.auto ()) comp Ir.Indexing.Empty
  in
  let opt = Option.value_exn !captured in
  let gpu_seeds opt =
    List.filter
      (Autotune.sketch_seed_params ~is_gpu:true ~is_cpu:false ~limits opt)
      ~f:(fun p -> p.Autotune.sk_gpu && not p.Autotune.sk_epilogue)
  in
  let seeds = gpu_seeds opt in
  p "bc: scalar GPU seeds exist for the batched site"
    (List.exists seeds ~f:(fun p -> not p.Autotune.sk_mma));
  (* The production census declined the tensorized seeds in the same proportion as the scalar ones
     (10 and 10) — both flavors route through [companion_geometry]. Tensorized seeds only exist
     where the backend advertises an mma format for f32 (Metal locally; CUDA/HIP need the tf32
     arm), so this line is informational rather than golden-pinned. *)
  if not (List.exists seeds ~f:(fun p -> p.Autotune.sk_mma)) then
    Stdio.eprintf "bc: no tensorized seed for this site on %s — mma legs below are vacuous\n"
      backend_name;
  let constructed =
    List.map seeds ~f:(fun sp ->
        match Autotune.sketch_schedule ~p:sp opt with
        | sched -> Either.First sched
        | exception Ir.Schedule_outcome.Cause_at (_, Ir.Schedule_outcome.Unsupported { detail; _ })
          ->
            Either.Second detail
        | exception exn -> Either.Second (Exn.to_string exn))
  in
  let declines = List.filter_map constructed ~f:(function Either.Second e -> Some e | _ -> None) in
  p "bc: no seed declines on companion coverage" (List.is_empty declines);
  List.iter declines ~f:(fun e -> Stdio.eprintf "bc decline: %s\n" e);
  (* The point of the fix is reach: the minor output axis (j) must actually carry [Grid] blocks in
     the constructed schedule, not merely survive construction. A schedule constructed under the
     old rule could not exist at all, but guard against a future regression that constructs a
     j-serial form: the split must be a real spread (factor < extent) of an axis of j's extent —
     m = 64 is unique to the minor output axis here (b = 4, i = 32, k = 16), in the site's nest
     and its companions alike. *)
  let bounds = LL.loop_bounds opt.LL.llc in
  let spreads_j sched =
    List.exists sched ~f:(function
      | Sched.Split { axis; factor; outer = LL.Grid; _ } -> (
          factor < m
          &&
          match List.Assoc.find bounds axis ~equal:Ir.Indexing.equal_symbol with
          | Some (0, hi) -> hi + 1 = m
          | _ -> false)
      | _ -> false)
  in
  p "bc: constructed schedules spread the minor output axis across Grid blocks"
    (List.for_all constructed ~f:(function Either.First s -> spreads_j s | _ -> false));

  (* Executable parity, seed by seed, on backends that can run shared staging. Vacuous on cc —
     announced on stderr so it is not mistaken for coverage; the golden stays backend-independent. *)
  if not is_gpu then
    Stdio.eprintf "bc: %s is not a GPU backend — the executable parity check below is vacuous\n"
      backend_name;
  p "bc: every unfused GPU seed compiles and computes correctly"
    ((not is_gpu)
    || List.for_all (List.init (List.length seeds) ~f:Fn.id) ~f:(fun idx ->
           let transform opt =
             let sp = List.nth_exn (gpu_seeds opt) idx in
             Sched.apply (Autotune.sketch_schedule ~p:sp opt) opt
           in
           match
             let sctx, sroutine =
               Context.compile ~lowered_transform:transform (Context.auto ()) comp Ir.Indexing.Empty
             in
             let sctx = Context.run sctx sroutine in
             Context.get_values sctx y.Tensor.value
           with
           | got -> Array.for_all2_exn got expected ~f:(fun a b -> Float.(abs (a - b) < 1e-3))
           | exception exn ->
               Stdio.eprintf "bc: scalar GPU seed %d FAILED %s\n" idx (Exn.to_string exn);
               false));

  (* Tune integration: the search itself (fission, per-segment seeding, replay) must route through
     the widened coverage — and the winner must still compute the right values. On cc the GPU
     sketches are not seeded, so the decline assertion is vacuously true there; on GPU backends it
     is the production pin: gpt2's search logged this exact decline key 20 times per arm. *)
  let reports = ref [] in
  let ctx = Context.auto () in
  let ctx, routine =
    Autotune.tune ~beam_width:2 ~rounds:0 ~repeats:1 ~cache_dir:"batched_companion_cache"
      ~timing_ctx:(Context.auto ())
      ~report:(fun r -> reports := r :: !reports)
      ctx comp Ir.Indexing.Empty
  in
  let ctx = Context.run ctx routine in
  let got = Context.get_values ctx y.Tensor.value in
  p "bc: tuned batched head matches the reference"
    (Array.for_all2_exn got expected ~f:(fun a b -> Float.(abs (a - b) < 1e-3)));
  (* Exactly one report, then its census — a vacuous [for_all] over zero reports would claim the
     census was clean without having inspected one. *)
  p "bc: the tuning census records no companion-coverage decline"
    (match !reports with
    | [ r ] ->
        List.for_all r.Autotune.declines ~f:(fun d ->
            match d.Autotune.key with
            | Ir.Schedule_outcome.Unsupported_key k ->
                not (String.equal k "autotune_sketch_companion_coverage")
            | _ -> true)
    | _ -> false);

  (* The safety boundary the lifted arity must NOT cross (the lm_head shape): a companion that
     REDUCES over the site's minor output axis reads cells every j-block wrote, with no intra-kernel
     synchronization — spreading j is a race there until fission separates the reduction. Every
     seed must still raise the companion-coverage decline (here the analysis bails outright on the
     reduction target's whole-node [Zero_out]; a pre-zeroed variant would instead trim the
     component's common prefix below the site's arity — either way, no schedule is constructed). *)
  let x2 = TDSL.ndarray xv ~label:[ "bc_x2" ] ~batch_dims:[ b ] ~output_dims:[ n; k ] () in
  let w2 = TDSL.ndarray wv ~label:[ "bc_w2" ] ~output_dims:[ k; m ] () in
  let%op z2 = x2 +* "b|ik;kj=>b|ij" w2 in
  Train.set_materialized z2.Tensor.value;
  let%op r2 = z2 ++ "b|ij => b|i" in
  let comp2 = named "bc_rowsum" (Train.forward r2) in
  let captured2 = ref None in
  let _ctx, _r =
    Context.compile
      ~lowered_transform:(fun opt ->
        captured2 := Some opt;
        opt)
      (Context.auto ()) comp2 Ir.Indexing.Empty
  in
  let opt2 = Option.value_exn !captured2 in
  let seeds2 = gpu_seeds opt2 in
  p "bc: reduction-over-j companion still declines every GPU seed"
    ((not (List.is_empty seeds2))
    && List.for_all seeds2 ~f:(fun sp ->
           match Autotune.sketch_schedule ~p:sp opt2 with
           | _ -> false
           | exception
               Ir.Schedule_outcome.Cause_at (_, Ir.Schedule_outcome.Unsupported { feature; _ }) ->
               String.equal feature "autotune_sketch_companion_coverage"
           | exception _ -> false))
