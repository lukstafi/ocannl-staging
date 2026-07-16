(* Decline visibility for Tile_mma renderings and pool-parallel Grid privatization (gh-ocannl-474 /
   gh-ocannl-479).

   Declines are silent by design — the scalar fallback and the serial loop are correct — but
   indistinguishable from the intended rendering in timings, which already cost one wrong conclusion
   (docs/proposals/tensorize-mma.md: parity passed while tensor cores never ran). This test pins the
   three visibility mechanisms:

   - The [Ir.C_syntax.mma_census]: with [mma_census_enabled], each compiled [Tile_mma] statement
     records how it actually rendered (intrinsics / register-tiled / scalar fallback). Autotune
     reads it to annotate candidate timings; here we assert directly that a whole-triple [Tensorize]
     over a standard layout renders register-tiled on the C backends while a transposed-B layout
     (the gradient-GEMM shape) falls back to scalar.

   - The per-chunk privatization byte cap (config [cc_grid_private_bytes_cap], default
     256KB): a grid-outermost packed GEMM that re-packs the B~ panel per chunk (gh-ocannl-475's
     alternative shape) privatizes both tiles per chunk — under the default cap it fits at [n = 256]
     (80KB) and declines at [n = 1024] (272KB), rendering serial. Compile-only; the stderr
     diagnostics themselves are opt-in via [schedule_log_declines].

   - The autotune seeding pre-filter ([Autotune.sketch_seed_params], gh-ocannl-479): tensorized
     candidates that statically must render the scalar fallback are not seeded — transposed-B sites
     lose the whole-triple family (their packed flavors normalize the layout and stay), a
     non-hoistable transposed B additionally loses the hoisted-only Grid flavor (it would read B in
     place), and a column extent below one vector of lanes loses everything. Exactly one hoistable
     operand additionally seeds the mixed grid-outermost shape ([sk_pack_rest] with [sk_hoist],
     gh-ocannl-473); no hoistable operand seeds the grid-outermost per-chunk B~ re-packing shape
     ([sk_pack_rest] alone, gh-ocannl-475) when the tiles fit the privatization cap. Seed
     enumeration runs on the pre-schedule lowering with synthetic limits, so the printed counts
     are backend-independent. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module LL = Ir.Low_level
module Sched = Ir.Schedule
module Asgns = Ir.Assignments

let () = Utils.settings.output_debug_files_in_build_directory <- true
let p name b = Stdio.printf "%s: %b\n" name b
let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")
let on_cpu = String.is_substring backend_name ~substring:"cc"

let read_generated base_name =
  let ext = if String.is_substring backend_name ~substring:"metal" then ".metal" else ".c" in
  let ext = if String.is_substring backend_name ~substring:"cuda" then ".cu" else ext in
  let ext = if String.is_substring backend_name ~substring:"hip" then ".hip" else ext in
  let path = Utils.build_file (base_name ^ ext) in
  if Stdlib.Sys.file_exists path then Some (Stdio.In_channel.read_all path) else None

(* The zeroing nest is its own Grid loop and parallelizes independently of the accumulation loop:
   count constructs rather than testing presence. *)
let count_parallel_constructs src =
  List.length (String.substr_index_all src ~may_overlap:false ~pattern:"dispatch_apply")
  + List.length
      (String.substr_index_all src ~may_overlap:false ~pattern:"#pragma omp parallel for")

let named name (comp : Asgns.comp) : Asgns.comp =
  { comp with asgns = Asgns.Block_comment (name, comp.asgns) }

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

let accum_syms (opt : LL.optimized) =
  let paths = nest_paths opt.LL.llc in
  match List.find_exn paths ~f:(fun p -> List.length p = 3) with
  | [ i; j; k ] -> (i, j, k)
  | _ -> assert false

let sink sym below = List.map below ~f:(fun inner -> Sched.Swap { outer = sym; inner })

(* Compile [comp] under [transform] with the Tile_mma rendering census enabled; returns the
   renderings in emission order. *)
let compile_with_census ~transform comp =
  Ir.C_syntax.mma_census := [];
  Ir.C_syntax.mma_census_enabled := true;
  let ctx = Context.auto () in
  let (_ : Context.t * Context.routine) =
    Context.compile ~lowered_transform:transform ctx comp Ir.Indexing.Empty
  in
  Ir.C_syntax.mma_census_enabled := false;
  List.rev_map !Ir.C_syntax.mma_census ~f:snd

let n = 64

(* === Census: whole-triple Tensorize, standard vs. transposed-B layout (C backends). === *)
let () =
  let mav = Array.init (n * n) ~f:(fun x -> Float.of_int (x % 13) *. 0.25) in
  let mbv = Array.init (n * n) ~f:(fun x -> Float.of_int (x % 17) -. 8.) in
  let ma = TDSL.ndarray mav ~label:[ "tmd_a" ] ~input_dims:[ n ] ~output_dims:[ n ] () in
  let mb = TDSL.ndarray mbv ~label:[ "tmd_b" ] ~input_dims:[ n ] ~output_dims:[ n ] () in
  let tensorize_schedule ~out (opt : LL.optimized) : Sched.schedule =
    let i, j, k = accum_syms opt in
    let ez, zsyms = Sched.expand_zero ~tn:out in
    let zj = match zsyms with [ _; zj ] -> zj | _ -> assert false in
    let rz = Sched.Retype { axis = zj; ty = LL.Workgroup } in
    let tz, _lane = Sched.tensorize ~i ~j ~k ~simd_width:n in
    [ ez; rz; tz ]
  in
  let%op mc = ma * mb in
  let census =
    compile_with_census
      ~transform:(fun opt ->
        if on_cpu then Sched.apply (tensorize_schedule ~out:mc.Tensor.value opt) opt else opt)
      (named "tmd_regtile" (Train.forward mc))
  in
  p "census: standard layout renders register-tiled"
    ((not on_cpu)
    || List.equal Ir.C_syntax.equal_mma_rendering census [ Ir.C_syntax.Mma_register_tiled ]);
  let mta = TDSL.ndarray mav ~label:[ "tmd_ta" ] ~output_dims:[ n; n ] () in
  let mtb = TDSL.ndarray mbv ~label:[ "tmd_tb" ] ~output_dims:[ n; n ] () in
  let%op td = mta +* "ik;jk=>ij" mtb in
  let census_tb =
    compile_with_census
      ~transform:(fun opt ->
        if on_cpu then Sched.apply (tensorize_schedule ~out:td.Tensor.value opt) opt else opt)
      (named "tmd_scalar_tb" (Train.forward td))
  in
  p "census: transposed-B layout falls back to the lane-0 scalar rendering"
    ((not on_cpu)
    || List.equal Ir.C_syntax.equal_mma_rendering census_tb [ Ir.C_syntax.Mma_scalar_fallback ])

(* === The per-chunk privatization cap: grid-outermost packed GEMM re-packing B~ per chunk
   (gh-ocannl-475's shape). Both packing Stages land inside the Grid body, so both tiles must
   privatize per chunk: at [n = 256] the footprint (bk*n + bm*bk floats = 80KB) fits under the
   default 256KB cap and the loop pool-parallelizes; at [n = 1024] (272KB) the cap trips and the
   loop renders serial. Compile-only. === *)
let () =
  let bm, bk = (64, 64) in
  let bpack_schedule ~a ~b ~out (opt : LL.optimized) : Sched.schedule =
    let i, j, k = accum_syms opt in
    let ez, zsyms = Sched.expand_zero ~tn:out in
    let zi = match zsyms with [ zi; _ ] -> zi | _ -> assert false in
    let sp_zi, _, _ = Sched.split ~axis:zi ~factor:bm ~outer:LL.Grid ~inner:LL.Serial in
    let sp_i, _, i_i = Sched.split ~axis:i ~factor:bm ~outer:LL.Grid ~inner:LL.Serial in
    let sp_k, k_o, k_i = Sched.split ~axis:k ~factor:bk ~outer:LL.Serial ~inner:LL.Serial in
    let stage source tile_loops =
      Sched.Stage { source; tile_loops; shared = false; cooperative = None; hoisted = false }
    in
    let tz, _lane = Sched.tensorize ~i:i_i ~j ~k:k_i ~simd_width:1 in
    (* No [sink i_o [k_o]]: the Grid loop stays outermost, so the B~ pack at [k_o] lands inside the
       Grid body and each chunk re-packs its own panel. *)
    [ ez; sp_zi; sp_i; sp_k ] @ sink j [ k_o ] @ sink i_i [ k_o ]
    @ [ stage b [ k_i; j ]; stage a [ i_i; k_i ]; tz ]
  in
  let compile_bpack ?(run_parity = false) ~name ~dim () =
    let av = Array.init (dim * dim) ~f:(fun x -> Float.of_int (x % 13) *. 0.25) in
    let bv = Array.init (dim * dim) ~f:(fun x -> Float.of_int (x % 17) -. 8.) in
    let a = TDSL.ndarray av ~label:[ "tmd_bp_a" ] ~input_dims:[ dim ] ~output_dims:[ dim ] () in
    let b = TDSL.ndarray bv ~label:[ "tmd_bp_b" ] ~input_dims:[ dim ] ~output_dims:[ dim ] () in
    let%op c = a * b in
    let transform opt =
      if on_cpu then
        Sched.apply (bpack_schedule ~a:a.Tensor.value ~b:b.Tensor.value ~out:c.Tensor.value opt) opt
      else opt
    in
    let ctx = Context.auto () in
    let ctx, routine =
      Context.compile ~lowered_transform:transform ctx (named name (Train.forward c))
        Ir.Indexing.Empty
    in
    (* Executed parity, not just structure: per-chunk privatization rewrites where the packed
       values live, so compare against a serial twin of the same computation. *)
    if run_parity then (
      let ctx = Context.run ctx routine in
      let got = Context.get_values ctx c.Tensor.value in
      let a2 = TDSL.ndarray av ~label:[ "tmd_bp_a2" ] ~input_dims:[ dim ] ~output_dims:[ dim ] () in
      let b2 = TDSL.ndarray bv ~label:[ "tmd_bp_b2" ] ~input_dims:[ dim ] ~output_dims:[ dim ] () in
      let%op c2 = a2 * b2 in
      let sctx = Context.auto () in
      let sctx, sroutine =
        Context.compile
          ~lowered_transform:(fun opt -> opt)
          sctx
          (named (name ^ "_serial") (Train.forward c2))
          Ir.Indexing.Empty
      in
      let sctx = Context.run sctx sroutine in
      let want = Context.get_values sctx c2.Tensor.value in
      p "per-chunk B~ re-pack matches the serial twin bitwise"
        (Array.for_all2_exn got want ~f:Float.equal));
    read_generated name
  in
  (match compile_bpack ~run_parity:true ~name:"tmd_bpack_fits" ~dim:256 () with
  | None -> p "per-chunk B~ re-pack under the cap pool-parallelizes" false
  | Some src ->
      (* Two parallel loops: the expanded zeroing nest and the packed accumulation loop. *)
      p "per-chunk B~ re-pack under the cap pool-parallelizes"
        ((not on_cpu)
        || (count_parallel_constructs src = 2 && String.is_substring src ~substring:"tile_")));
  match compile_bpack ~name:"tmd_bpack_over" ~dim:1024 () with
  | None -> p "per-chunk B~ re-pack over the cap declines to serial" false
  | Some src ->
      (* Only the zeroing nest parallelizes; the accumulation loop's per-chunk tiles trip the cap
         and it renders serial. *)
      p "per-chunk B~ re-pack over the cap declines to serial"
        ((not on_cpu) || count_parallel_constructs src = 1)

(* === The seeding pre-filter (gh-ocannl-479) and the mixed grid-outermost seed (gh-ocannl-473).
   Synthetic limits pin the vector width (32 bytes = 8 f32 lanes), and seeds are enumerated on the
   pre-schedule lowering, so the counts hold on every backend. === *)
let () =
  let limits = { Ir.Backend_intf.no_hardware_limits with simd_vector_bytes = 32 } in
  let seeds_of ~name comp =
    let captured = ref None in
    let ctx = Context.auto () in
    let (_ : Context.t * Context.routine) =
      Context.compile
        ~lowered_transform:(fun opt ->
          captured := Some (Autotune.sketch_seed_params ~is_gpu:false ~is_cpu:true ~limits opt);
          opt)
        ctx (named name comp) Ir.Indexing.Empty
    in
    Option.value_exn !captured
  in
  let summarize label seeds =
    let count f = List.count seeds ~f in
    Stdio.printf "%s: total=%d whole=%d packed=%d hoist=%d grid=%d packrest=%d\n" label
      (List.length seeds)
      (count (fun p -> p.Autotune.sk_mma && p.Autotune.sk_bk = 0))
      (count (fun p -> p.Autotune.sk_mma && p.Autotune.sk_bk > 0))
      (count (fun p -> p.Autotune.sk_hoist))
      (count (fun p -> p.Autotune.sk_grid))
      (count (fun p -> p.Autotune.sk_pack_rest))
  in
  let mav = Array.init (n * n) ~f:(fun x -> Float.of_int (x % 13) *. 0.25) in
  let mbv = Array.init (n * n) ~f:(fun x -> Float.of_int (x % 17) -. 8.) in
  (* Both operands hoistable, standard layout: every family seeds, no mixed shape (nothing left to
     pack in-kernel under the hoisted grid-outermost flavor). *)
  let ma = TDSL.ndarray mav ~label:[ "tmd_s_a" ] ~input_dims:[ n ] ~output_dims:[ n ] () in
  let mb = TDSL.ndarray mbv ~label:[ "tmd_s_b" ] ~input_dims:[ n ] ~output_dims:[ n ] () in
  let%op sc = ma * mb in
  summarize "seeds: standard, both hoistable" (seeds_of ~name:"tmd_seeds_std" (Train.forward sc));
  (* Transposed B, both hoistable: the whole-triple family (which reads B in place) is filtered
     out; the packed flavors normalize the layout and stay, including the hoisted Grid flavor
     (hoisted B~ packs at link time, so nothing reads B in place). *)
  let mta = TDSL.ndarray mav ~label:[ "tmd_s_ta" ] ~output_dims:[ n; n ] () in
  let mtb = TDSL.ndarray mbv ~label:[ "tmd_s_tb" ] ~output_dims:[ n; n ] () in
  let%op st = mta +* "ik;jk=>ij" mtb in
  summarize "seeds: transposed B, hoistable" (seeds_of ~name:"tmd_seeds_tb" (Train.forward st));
  (* Exactly one hoistable operand (computed A x constant B), standard layout: the mixed
     grid-outermost shape ([sk_pack_rest]) joins the pool (gh-ocannl-473). *)
  let ma2 = TDSL.ndarray mav ~label:[ "tmd_s_a2" ] ~input_dims:[ n ] ~output_dims:[ n ] () in
  let mb2 = TDSL.ndarray mbv ~label:[ "tmd_s_b2" ] ~input_dims:[ n ] ~output_dims:[ n ] () in
  let%op xa = relu ma2 in
  let%op yc = xa * mb2 in
  summarize "seeds: computed A x hoistable B" (seeds_of ~name:"tmd_seeds_mixed" (Train.forward yc));
  (* Non-hoistable transposed B: the hoisted-only Grid flavor would read B in place and statically
     decline, so it is filtered; the mixed shape packs B in-kernel (normalizing it) and stays. *)
  let ma3 = TDSL.ndarray mav ~label:[ "tmd_s_a3" ] ~output_dims:[ n; n ] () in
  let mtb3 = TDSL.ndarray mbv ~label:[ "tmd_s_tb3" ] ~output_dims:[ n; n ] () in
  let%op xb = relu mtb3 in
  let%op yt = ma3 +* "ik;jk=>ij" xb in
  summarize "seeds: hoistable A x computed transposed B"
    (seeds_of ~name:"tmd_seeds_tb_mixed" (Train.forward yt));
  (* No hoistable operand (both trainable params): the grid-outermost per-chunk B~ re-packing
     shape ([sk_pack_rest] without [sk_hoist], gh-ocannl-475) is the only one-dispatch flavor, and
     is seeded whenever the packed tiles statically fit the per-chunk privatization cap. *)
  let wa = TDSL.param ~values:mav "tmd_wa" ~input_dims:[ n ] ~output_dims:[ n ] () in
  let wb = TDSL.param ~values:mbv "tmd_wb" ~input_dims:[ n ] ~output_dims:[ n ] () in
  let%op yw = wa * wb in
  summarize "seeds: no hoistable operand" (seeds_of ~name:"tmd_seeds_params" (Train.forward yw));
  (* Column extent below one vector of lanes (nj = 4 < 8): the register tiling statically declines,
     so no tensorized candidate is seeded at all (and no scalar tiling divides nj either). *)
  let mbn = Array.init (n * 4) ~f:(fun x -> Float.of_int (x % 17) -. 8.) in
  let ma4 = TDSL.ndarray mav ~label:[ "tmd_s_a4" ] ~input_dims:[ n ] ~output_dims:[ n ] () in
  let mb4 = TDSL.ndarray mbn ~label:[ "tmd_s_b4" ] ~input_dims:[ 4 ] ~output_dims:[ n ] () in
  let%op yn = ma4 * mb4 in
  summarize "seeds: narrow columns (nj < lanes)" (seeds_of ~name:"tmd_seeds_narrow" (Train.forward yn))
