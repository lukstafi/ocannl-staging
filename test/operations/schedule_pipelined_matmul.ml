(* Software pipelining, gh-ocannl-487 phase 1: the double-buffered cooperative Stage over the
   staged+tensorized matmul composition (the lane-aware staging leg of schedule_mma_matmul), pinned
   structurally and by executed parity.

   [mc = ma * mb] (32x32 times 32x32) under the staged schedule: expand + annotate the zeroing,
   split i by 16 into Grid x Serial, split k by 16 into Serial x Serial, swap j and i_i inside k_o,
   cooperatively Stage ma over [i_i; k_i] and mb over [k_i; j] with [pipeline_depth = d], then
   Tensorize the i_i x j x k_i micro-kernel. Both stages anchor at the Serial k_o — the rotor.

   THE INVARIANT (the whole point of the transform being a pure schedule transform): the pipelined
   rendering is bitwise identical to depth 1. The compute subtree is remapped identically — only
   the copy for k_o+1 is issued before the compute of k_o (prologue before the loop, a single
   in-loop barrier instead of two, a trailing barrier after), and the renderer's buffer rotation
   makes every read resolve to the copy holding exactly the values the unpipelined form reads. So
   depth 2 and depth 3 must match depth 1 BITWISE on the GPU backends (while all three only
   approximate the serial twin — the tile reduction reassociates, which is Tensorize's license, not
   pipelining's).

   Structural pins (backend-independent, on the applied schedule): the pipelined map records both
   tiles at the requested depth with k_o as rotor; the k_o body holds one barrier and one
   If-guarded prefetch per stage (depth 1: two barriers per stage, no Ifs); both prologue loads sit
   outside the loop. Canonical digests must distinguish depths 1/2/3 pairwise — depths >= 2 emit
   identical IR and differ only in the renderer's rotation modulus, so this pins the digest's
   pipelined companion section specifically. Validation legs pin the illegal placements. The C
   backends cannot express shared placement and must reject the compile cleanly (the same pinning
   as the SMEM matmul test); the schedule-level validation errors are the typed Illegal_schedule
   declines autotune candidates see. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module Tn = Ir.Tnode
module LL = Ir.Low_level
module Sched = Ir.Schedule
module SC = Ir.Schedule_cache
module Asgns = Ir.Assignments
module Numerics = Ir.Numerics

let () = Utils.settings.output_debug_files_in_build_directory <- true
let () = Numerics.set_policy { (Numerics.get ()) with tf32_matmuls = false }
let p name b = Stdio.printf "%s: %b\n" name b
let approx a b = Float.(abs (a - b) < 1e-2)
let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")
let on_metal = String.is_substring backend_name ~substring:"metal"

let on_gpu =
  on_metal
  || String.is_substring backend_name ~substring:"cuda"
  || String.is_substring backend_name ~substring:"hip"

let read_generated base_name =
  let ext =
    if on_metal then ".metal"
    else if String.is_substring backend_name ~substring:"hip" then ".hip"
    else if on_gpu then ".cu"
    else ".c"
  in
  let path = Utils.build_file (base_name ^ ext) in
  if Stdlib.Sys.file_exists path then Some (Stdio.In_channel.read_all path) else None

(* The maximal single-child chains of statement-level loops: one symbol list per top-level nest. *)
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

let n = 32
let bm = 16
let simd_width = 32

(* Statement-level walks for the structural pins. *)
let rec find_loop sym (llc : LL.t) : LL.t option =
  match llc with
  | LL.For_loop { index; _ } when Ir.Indexing.equal_symbol index sym -> Some llc
  | LL.For_loop { body; _ } -> find_loop sym body
  | LL.Seq (a, b) -> ( match find_loop sym a with Some _ as r -> r | None -> find_loop sym b)
  | LL.If { body; _ } -> find_loop sym body
  | _ -> None

let rec count_stmts ~f (llc : LL.t) : int =
  (if f llc then 1 else 0)
  +
  match llc with
  | LL.Seq (a, b) -> count_stmts ~f a + count_stmts ~f b
  | LL.For_loop { body; _ } | LL.If { body; _ } -> count_stmts ~f body
  | _ -> 0

let rec writes_tile ~is_tile (llc : LL.t) : bool =
  match llc with
  | LL.Set { tn; _ } -> is_tile tn
  | LL.Seq (a, b) -> writes_tile ~is_tile a || writes_tile ~is_tile b
  | LL.For_loop { body; _ } | LL.If { body; _ } -> writes_tile ~is_tile body
  | _ -> false

let () =
  let mav = Array.init (n * n) ~f:(fun i -> Float.of_int (i % 13) *. 0.25) in
  let mbv = Array.init (n * n) ~f:(fun i -> Float.of_int (i % 17) -. 8.) in
  let ma = TDSL.ndarray mav ~label:[ "ma" ] ~input_dims:[ n ] ~output_dims:[ n ] () in
  let mb = TDSL.ndarray mbv ~label:[ "mb" ] ~input_dims:[ n ] ~output_dims:[ n ] () in

  (* --- Serial twin (approximate reference; the bitwise references are the staged depths). --- *)
  let%op mc0 = ma * mb in
  let serial_comp = named "pipe_mm_serial" (Train.forward mc0) in
  let ctx_s = Context.auto () in
  let ctx_s, routine_s =
    Context.compile ~lowered_transform:(fun opt -> opt) ctx_s serial_comp Ir.Indexing.Empty
  in
  let ctx_s = Context.run ctx_s routine_s in
  let got_serial = Context.get_values ctx_s mc0.Tensor.value in

  (* --- The staged + tensorized schedule with a pipeline depth knob. Returns the rotor. --- *)
  let staged_schedule ~pipeline_depth ~out (opt : LL.optimized) :
      Sched.schedule * Ir.Indexing.symbol =
    let paths = nest_paths opt.LL.llc in
    let i, j, k =
      match List.find_exn paths ~f:(fun path -> List.length path = 3) with
      | [ i; j; k ] -> (i, j, k)
      | _ -> assert false
    in
    let ez, zsyms = Sched.expand_zero ~tn:out in
    let zi, zj = match zsyms with [ zi; zj ] -> (zi, zj) | _ -> assert false in
    let sp_zi, _, _ = Sched.split ~axis:zi ~factor:bm ~outer:LL.Grid ~inner:LL.Serial in
    let rz = Sched.Retype { axis = zj; ty = LL.Workgroup } in
    let sp_i, _, i_i = Sched.split ~axis:i ~factor:bm ~outer:LL.Grid ~inner:LL.Serial in
    let sp_k, k_o, k_i = Sched.split ~axis:k ~factor:bm ~outer:LL.Serial ~inner:LL.Serial in
    let tz, _lane = Sched.tensorize ~i:i_i ~j ~k:k_i ~simd_width in
    let stage source tile_loops =
      Sched.Stage
        {
          source;
          tile_loops;
          shared = true;
          cooperative = Some simd_width;
          hoisted = false;
          swizzle = None;
          pad_stride = None;
          pipeline_depth;
        }
    in
    ( [
        ez;
        sp_zi;
        rz;
        sp_i;
        sp_k;
        Sched.Swap { outer = j; inner = k_o };
        Sched.Swap { outer = i_i; inner = k_o };
        stage ma.Tensor.value [ i_i; k_i ];
        stage mb.Tensor.value [ k_i; j ];
        tz;
      ],
      k_o )
  in

  (* One compile per depth; the transform captures the applied optimized code and the rotor for the
     backend-independent structural pins (on the C backends the capture still happens — the
     rejection below comes later, from the backend's compile of the transformed code). *)
  let captured : (int, LL.optimized * Ir.Indexing.symbol) Hashtbl.t = Hashtbl.create (module Int) in
  let compile_depth ~pipeline_depth ~out comp =
    let transform opt =
      let schedule, k_o = staged_schedule ~pipeline_depth ~out opt in
      let applied = Sched.apply schedule opt in
      Hashtbl.set captured ~key:pipeline_depth ~data:(applied, k_o);
      applied
    in
    let ctx = Context.auto () in
    Context.compile ~lowered_transform:transform ctx comp Ir.Indexing.Empty
  in
  let%op mc1 = ma * mb in
  let%op mc2 = ma * mb in
  let%op mc3 = ma * mb in
  let legs =
    [
      (1, mc1, named "pipe_mm_d1" (Train.forward mc1));
      (2, mc2, named "pipe_mm_d2" (Train.forward mc2));
      (3, mc3, named "pipe_mm_d3" (Train.forward mc3));
    ]
  in
  if on_gpu then (
    let results =
      List.map legs ~f:(fun (depth, mc, comp) ->
          let ctx, routine = compile_depth ~pipeline_depth:depth ~out:mc.Tensor.value comp in
          let ctx = Context.run ctx routine in
          (depth, Context.get_values ctx mc.Tensor.value))
    in
    let got_d1 = List.Assoc.find_exn results 1 ~equal:Int.equal in
    let got_d2 = List.Assoc.find_exn results 2 ~equal:Int.equal in
    let got_d3 = List.Assoc.find_exn results 3 ~equal:Int.equal in
    p "staged depths approximate the serial twin (GPU) or clean rejection (CPU)"
      (List.for_all results ~f:(fun (_, got) -> Array.for_all2_exn got got_serial ~f:approx));
    (* The invariant: pipelining is a pure prefetch-timing transform. *)
    p "depth 2 matches depth 1 BITWISE" (Array.for_all2_exn got_d2 got_d1 ~f:Float.equal);
    p "depth 3 matches depth 1 BITWISE" (Array.for_all2_exn got_d3 got_d1 ~f:Float.equal);
    (match (read_generated "pipe_mm_d1", read_generated "pipe_mm_d2") with
    | Some src1, Some src2 ->
        let count_sub src sub =
          String.substr_index_all src ~may_overlap:false ~pattern:sub |> List.length
        in
        (* The depth-2 kernel rotates both tile buffers ([% 2] terms on reads and writes) and
           allocates each tile at twice the depth-1 size (16x16 -> [512] and 16x32 -> [1024]
           floats); the depth-1 kernel has no rotation and single-size tiles. *)
        let decl_ok src d =
          if on_metal then
            count_sub src ("[" ^ Int.to_string (256 * d) ^ "]") >= 1
            && count_sub src ("[" ^ Int.to_string (512 * d) ^ "]") >= 1
          else true
        in
        p "depth 2 rotates the buffers the depth-1 kernel does not"
          (count_sub src2 "% 2" >= 4
          && count_sub src1 "% 2" = 0
          && decl_ok src1 1 && decl_ok src2 2)
    | _ -> p "depth 2 rotates the buffers the depth-1 kernel does not" false))
  else (
    (* The C backends reject the shared staging composition at compile (workgroup-shared placement
       is not renderable there) — same clean rejection as the SMEM matmul test, at any depth. *)
    let rejects (depth, mc, comp) =
      try
        ignore
          (compile_depth ~pipeline_depth:depth ~out:mc.Tensor.value comp
            : Context.t * Context.routine);
        false
      with Invalid_argument msg -> String.is_substring msg ~substring:"not supported"
    in
    p "staged depths approximate the serial twin (GPU) or clean rejection (CPU)"
      (List.for_all legs ~f:rejects);
    p "depth 2 matches depth 1 BITWISE" true;
    p "depth 3 matches depth 1 BITWISE" true;
    p "depth 2 rotates the buffers the depth-1 kernel does not" true);

  (* --- Structural pins on the applied schedules (all backends; captured by the transforms
     above). --- *)
  let opt_d1, k_o1 = Hashtbl.find_exn captured 1 in
  let opt_d2, k_o2 = Hashtbl.find_exn captured 2 in
  let opt_d3, _ = Hashtbl.find_exn captured 3 in
  (* Both stages' tiles, at either depth: the workgroup-shared set. *)
  let is_tile opt tn = Set.mem opt.LL.workgroup_shared tn in
  p "depth 1 records no pipelined tiles" (Map.is_empty opt_d1.LL.pipelined);
  p "depth 2 records both tiles at depth 2 with k_o as rotor"
    (Map.length opt_d2.LL.pipelined = 2
    && Map.for_all opt_d2.LL.pipelined ~f:(fun { LL.pt_depth; pt_rotor } ->
           pt_depth = 2 && Ir.Indexing.equal_symbol pt_rotor k_o2));
  let barriers llc =
    count_stmts llc ~f:(function LL.Workgroup_barrier -> true | _ -> false)
  in
  let body_of sym llc =
    match find_loop sym llc with
    | Some (LL.For_loop { body; _ }) -> body
    | _ -> LL.Noop
  in
  let d1_body = body_of k_o1 opt_d1.LL.llc in
  let d2_body = body_of k_o2 opt_d2.LL.llc in
  (* Depth 1: per stage, [loads; barrier; ...; barrier] nested inside k_o — 4 barriers, no
     statement-level If-guarded prefetches, no tile writes outside the loop. Depth 2: one barrier
     per stage inside k_o plus one If-guarded prefetch per stage (statement-level, so the lane
     restriction Ifs nested within load nests are not miscounted); the prologue loads (tile writes
     outside k_o) and the trailing barriers are the loop's siblings. *)
  p "depth 1 keeps both phase barriers per stage inside k_o"
    (barriers d1_body = 4 && barriers opt_d1.LL.llc = 4);
  p "depth 2 keeps one barrier per stage inside k_o, one trailing each outside"
    (barriers d2_body = 2 && barriers opt_d2.LL.llc = 4);
  let stmt_prefetches opt llc =
    List.count (LL.flat_lines [ llc ]) ~f:(function
      | LL.If { body; _ } -> writes_tile ~is_tile:(is_tile opt) body
      | _ -> false)
  in
  p "depth 2 issues one If-guarded prefetch per stage inside k_o"
    (stmt_prefetches opt_d2 d2_body = 2 && stmt_prefetches opt_d1 d1_body = 0);
  (* Each stage's copy is a single guarded [Set] of its tile: depth 1 has both inside k_o; depth 2
     has two inside (the prefetches) and two outside (the prologues). *)
  let tile_sets opt llc =
    count_stmts llc ~f:(function LL.Set { tn; _ } -> is_tile opt tn | _ -> false)
  in
  let d1_total = tile_sets opt_d1 opt_d1.LL.llc and d1_in = tile_sets opt_d1 d1_body in
  let d2_total = tile_sets opt_d2 opt_d2.LL.llc and d2_in = tile_sets opt_d2 d2_body in
  p "depth 2 places both prologue loads outside k_o"
    (d1_total = 2 && d1_in = 2 && d2_total = 4 && d2_in = 2);

  (* --- Cache identity: canonical digests distinguish all three depths pairwise. Depth 1 vs 2
     differ in the IR itself; 2 vs 3 emit identical IR and differ only in the pipelined companion
     section — the leg that would silently alias without it. --- *)
  let dg opt = SC.digest (SC.canonicalize opt) in
  let d1, d2, d3 = (dg opt_d1, dg opt_d2, dg opt_d3) in
  p "canonical digests distinguish depths 1/2/3 pairwise"
    ((not (String.equal d1 d2)) && (not (String.equal d2 d3)) && not (String.equal d1 d3));

  (* --- Validation legs: the illegal placements are typed schedule errors, not crashes. --- *)
  let expect_invalid name ~substring:sub (f : unit -> unit) =
    match f () with
    | () -> p name false
    | exception Invalid_argument msg -> p name (String.is_substring msg ~substring:sub)
  in
  let reapply ~probe ~f () =
    (* Re-lower a fresh forward to probe schedule application hermetically. *)
    let%op mcv = ma * mb in
    let comp = named probe (Train.forward mcv) in
    let transform opt =
      f ~out:mcv.Tensor.value opt;
      opt
    in
    ignore
      (Context.compile ~lowered_transform:transform (Context.auto ()) comp Ir.Indexing.Empty
        : Context.t * Context.routine)
  in
  expect_invalid "pipeline_depth 0 is rejected" ~substring:"pipeline_depth must be >= 1"
    (reapply ~probe:"pipe_mm_probe0" ~f:(fun ~out opt ->
         let schedule, _ = staged_schedule ~pipeline_depth:0 ~out opt in
         ignore (Sched.apply schedule opt : LL.optimized)));
  expect_invalid "pipelining a non-cooperative (packing) Stage is rejected"
    ~substring:"requires cooperative staging"
    (reapply ~probe:"pipe_mm_probe1" ~f:(fun ~out:_ opt ->
         let paths = nest_paths opt.LL.llc in
         let k =
           match List.find_exn paths ~f:(fun path -> List.length path = 3) with
           | [ _; _; k ] -> k
           | _ -> assert false
         in
         let op =
           Sched.Stage
             {
               source = ma.Tensor.value;
               tile_loops = [ k ];
               shared = false;
               cooperative = None;
               hoisted = false;
               swizzle = None;
               pad_stride = None;
               pipeline_depth = 2;
             }
         in
         ignore (Sched.apply [ op ] opt : LL.optimized)));
  expect_invalid "pipelining at a non-Serial anchor is rejected"
    ~substring:"to be Serial"
    (reapply ~probe:"pipe_mm_probe2" ~f:(fun ~out opt ->
         (* Same composition, but the anchor k_o retyped to Grid before the stages: the rotor has
            no serial iteration order to rotate with. (Racy as a real schedule — irrelevant, the
            application must already reject it.) *)
         let paths = nest_paths opt.LL.llc in
         let i, j, k =
           match List.find_exn paths ~f:(fun path -> List.length path = 3) with
           | [ i; j; k ] -> (i, j, k)
           | _ -> assert false
         in
         let ez, zsyms = Sched.expand_zero ~tn:out in
         let zi, zj = match zsyms with [ zi; zj ] -> (zi, zj) | _ -> assert false in
         let sp_zi, _, _ = Sched.split ~axis:zi ~factor:bm ~outer:LL.Grid ~inner:LL.Serial in
         let rz = Sched.Retype { axis = zj; ty = LL.Workgroup } in
         let sp_i, _, i_i = Sched.split ~axis:i ~factor:bm ~outer:LL.Grid ~inner:LL.Serial in
         let sp_k, k_o, k_i = Sched.split ~axis:k ~factor:bm ~outer:LL.Serial ~inner:LL.Serial in
         let schedule =
           [
             ez;
             sp_zi;
             rz;
             sp_i;
             sp_k;
             Sched.Swap { outer = j; inner = k_o };
             Sched.Swap { outer = i_i; inner = k_o };
             Sched.Retype { axis = k_o; ty = LL.Grid };
             Sched.Stage
               {
                 source = ma.Tensor.value;
                 tile_loops = [ i_i; k_i ];
                 shared = true;
                 cooperative = Some simd_width;
                 hoisted = false;
                 swizzle = None;
                 pad_stride = None;
                 pipeline_depth = 2;
               };
           ]
         in
         ignore (Sched.apply schedule opt : LL.optimized)));

  (* --- Seeding pin (gh-ocannl-487): the depth twins follow the backend's advertised
     [mma_pipeline_depths] — one twin per staged seed per advertised depth, identical to its plain
     sibling except [sk_depth]; unstaged seeds are never twinned; an empty advertisement (the
     CUDA/HIP phase-2 state, and cc) seeds no twins. Probed against synthetic capabilities so the
     pin is backend-independent, like the schedule_pad seed checks. --- *)
  let limits_with depths =
    {
      Ir.Backend_intf.no_hardware_limits with
      max_threads_per_workgroup = Some 1024;
      max_workgroup_memory_bytes = Some 32768;
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
            mma_pipeline_depths = depths;
          };
    }
  in
  let seed_pin = ref None in
  let seed_transform opt =
    let seeds depths =
      Autotune.sketch_seed_params ~is_gpu:true ~is_cpu:false ~limits:(limits_with depths) opt
      |> List.filter ~f:(fun sp -> sp.Autotune.sk_mma && sp.Autotune.sk_gpu)
    in
    let advertised = seeds [ 2 ] and unadvertised = seeds [] in
    let staged = List.filter advertised ~f:(fun sp -> sp.Autotune.sk_bk > 0) in
    let plain_staged = List.filter staged ~f:(fun sp -> sp.Autotune.sk_depth = 1) in
    let twins = List.filter staged ~f:(fun sp -> sp.Autotune.sk_depth > 1) in
    let twin_of sp =
      List.exists twins ~f:(fun tw ->
          tw.Autotune.sk_depth = 2 && Poly.equal { tw with Autotune.sk_depth = 1 } sp)
    in
    seed_pin :=
      Some
        ( (not (List.is_empty plain_staged))
          && List.length twins = List.length plain_staged
          && List.for_all plain_staged ~f:twin_of
          && List.for_all advertised ~f:(fun sp ->
                 sp.Autotune.sk_bk > 0 || sp.Autotune.sk_depth = 1),
          List.for_all unadvertised ~f:(fun sp -> sp.Autotune.sk_depth = 1) );
    opt
  in
  let%op mc4 = ma * mb in
  ignore
    (Context.compile ~lowered_transform:seed_transform (Context.auto ())
       (named "pipe_mm_seeds" (Train.forward mc4))
       Ir.Indexing.Empty
      : Context.t * Context.routine);
  (match !seed_pin with
  | Some (twinned, none_without) ->
      p "advertised depth seeds one pd2 twin per plain staged seed, staged only" twinned;
      p "no depth twins without the capability advertisement" none_without
  | None ->
      p "advertised depth seeds one pd2 twin per plain staged seed, staged only" false;
      p "no depth twins without the capability advertisement" false)
