(* Software pipelining, gh-ocannl-487 phase 1: the double-buffered cooperative Stage over the
   staged+tensorized matmul composition (the lane-aware staging leg of schedule_mma_matmul), pinned
   structurally and by executed parity.

   [mc = ma * mb] (32x32 times 32x32) under the staged schedule: expand + annotate the zeroing,
   split i by 16 into Grid x Serial, split k by 16 into Serial x Serial, swap j and i_i inside k_o,
   cooperatively Stage ma over [i_i; k_i] and mb over [k_i; j] with [pipeline_depth = d], then
   Tensorize the i_i x j x k_i micro-kernel. Both stages anchor at the Serial k_o — the rotor.

   THE INVARIANT (the whole point of the transform being a pure schedule transform): the pipelined
   rendering is bitwise identical to depth 1. The compute subtree is remapped identically — only
   the copy for k_o+1 is issued before the compute of k_o (prologue before the loop, one shared
   in-loop barrier with both stages' prefetches grouped behind it, a trailing barrier after), and
   the renderer's buffer rotation makes every read resolve to the copy holding exactly the values
   the unpipelined form reads. So depth 2 must match depth 1 BITWISE on the GPU backends (while
   both only approximate the serial twin — the tile reduction reassociates, which is Tensorize's
   license, not pipelining's). Phase 1 implements depth 2 only; deeper lookahead is the phase-2
   async-copy arms', and depth 3 is pinned as a typed rejection.

   THE OTHER INVARIANT (gh-ocannl-567): the same-anchor load-phase grouping and the
   Tile_mma-bracket barrier elision, both of which PR #303 landed [opt.pipelined]-gated and gh-567
   widened to depth 1, change synchronization structure and never values. The reference is the
   un-elided leg below: the applied depth-1 schedule with the pre-gh-567 barrier structure rebuilt
   on top of it (a barrier after every load-phase statement and one closing the k-block — adding
   barriers is always conservative). Depth 1 must match it BITWISE.

   Structural pins (backend-independent, on the applied schedule): the pipelined map records both
   tiles at the requested depth with k_o as rotor; the k_o body holds ONE barrier for both
   same-anchor stages with both If-guarded prefetches grouped behind it; both prologue loads sit
   outside the loop. Depth 1 groups the same way and keeps exactly the one phase barrier the
   intrinsic does not provide (its bracket ends each k-block, but only some rendering forms open
   one per iteration — the fragment-scope form opens it once, outside the loop). Canonical digests
   distinguish the depths, and the pipelined companion section alone must separate byte-identical
   IR whose side-table depths differ (the phase-2 aliasing hazard, pinned by record surgery).
   Validation legs pin the illegal placements. The C backends cannot express shared placement and
   must reject the compile cleanly (the same pinning as the SMEM matmul test); the schedule-level
   validation errors are the typed Illegal_schedule declines autotune candidates see. *)

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

(* Zeros compare equal to zeros. A fragment mapping that reads outside the staged block, a kernel
   that never ran, or a reference whose own setup silently collapsed all yield all-zeros, and a
   parity check between two zero arrays passes while covering nothing (gh-ocannl-481 item 3). Every
   reference array is pinned nonzero where it is produced, so the parity claims below have content.
   *)
let nonzero name (a : float array) =
  if not (Array.exists a ~f:(fun x -> Float.(x <> 0.))) then
    failwith (name ^ ": the reference is all zeros — the parity checks against it are vacuous");
  a
let approx a b = Float.(abs (a - b) < 1e-2)
let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")

(* A leg this backend cannot run still prints its line, so the golden stays backend-uniform — but
   the skip is announced on stderr and marked in the source. A bare [p name true] is
   indistinguishable, both in the transcript and in review, from a verified run: that is how a
   Tensorize leg came to "cover" the gh-ocannl-528 interior-batch bug without ever checking it. *)
let skipped name =
  Stdio.eprintf "SKIPPED on %s (vacuous): %s\n%!" backend_name name;
  p name true
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
  let got_serial = nonzero "pd_serial" (Context.get_values ctx_s mc0.Tensor.value) in

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

  (* The gh-567 reference: rebuild the pre-gh-567 barrier structure on top of the applied depth-1
     schedule — drop the surviving barriers, put one after every load-phase statement, and close
     the k-block with one. Adding barriers can only remove races, so this leg is a sound reference
     for the invariant regardless of what the elision does. *)
  let reinsert_barriers ~is_tile ~k_o (llc : LL.t) : LL.t =
    let rec go (llc : LL.t) : LL.t =
      match llc with
      | LL.For_loop { index; from_; to_; body; axis } when Ir.Indexing.equal_symbol index k_o ->
          let stmts =
            LL.flat_lines [ body ]
            |> List.filter ~f:(function LL.Workgroup_barrier -> false | _ -> true)
            |> List.concat_map ~f:(fun s ->
                   if writes_tile ~is_tile s then [ s; LL.Workgroup_barrier ] else [ s ])
          in
          LL.For_loop
            { index; from_; to_; axis; body = LL.unflat_lines (stmts @ [ LL.Workgroup_barrier ]) }
      | LL.For_loop { index; from_; to_; body; axis } ->
          LL.For_loop { index; from_; to_; axis; body = go body }
      | LL.Seq (a, b) -> LL.Seq (go a, go b)
      | LL.If { cond; body } -> LL.If { cond; body = go body }
      | other -> other
    in
    go llc
  in
  (* One compile per leg; the transform captures the applied optimized code and the rotor for the
     backend-independent structural pins (on the C backends the capture still happens — the
     rejection below comes later, from the backend's compile of the transformed code). Key 0 is the
     un-elided depth-1 reference. *)
  let captured : (int, LL.optimized * Ir.Indexing.symbol) Hashtbl.t = Hashtbl.create (module Int) in
  let compile_depth ?post ~key ~pipeline_depth ~out comp =
    let transform opt =
      let schedule, k_o = staged_schedule ~pipeline_depth ~out opt in
      let applied = Sched.apply schedule opt in
      let applied =
        match post with
        | None -> applied
        | Some f ->
            let is_tile tn = Set.mem applied.LL.workgroup_shared tn in
            { applied with LL.llc = f ~is_tile ~k_o applied.LL.llc }
      in
      Hashtbl.set captured ~key ~data:(applied, k_o);
      applied
    in
    let ctx = Context.auto () in
    Context.compile ~lowered_transform:transform ctx comp Ir.Indexing.Empty
  in
  let%op mc0b = ma * mb in
  let%op mc1 = ma * mb in
  let%op mc2 = ma * mb in
  let legs =
    [
      (0, mc0b, named "pipe_mm_d1_barriers" (Train.forward mc0b));
      (1, mc1, named "pipe_mm_d1" (Train.forward mc1));
      (2, mc2, named "pipe_mm_d2" (Train.forward mc2));
    ]
  in
  let compile_leg (key, mc, comp) =
    let post = if key = 0 then Some reinsert_barriers else None in
    compile_depth ?post ~key ~pipeline_depth:(max key 1) ~out:mc.Tensor.value comp
  in
  if on_gpu then (
    let results =
      List.map legs ~f:(fun ((key, mc, _) as leg) ->
          let ctx, routine = compile_leg leg in
          let ctx = Context.run ctx routine in
          (key, Context.get_values ctx mc.Tensor.value))
    in
    let got_ref = List.Assoc.find_exn results 0 ~equal:Int.equal in
    let got_d1 = List.Assoc.find_exn results 1 ~equal:Int.equal in
    let got_d2 = List.Assoc.find_exn results 2 ~equal:Int.equal in
    p "staged depths approximate the serial twin (GPU) or clean rejection (CPU)"
      (List.for_all results ~f:(fun (_, got) -> Array.for_all2_exn got got_serial ~f:approx));
    (* The invariant: pipelining is a pure prefetch-timing transform. *)
    p "depth 2 matches depth 1 BITWISE" (Array.for_all2_exn got_d2 got_d1 ~f:Float.equal);
    (* The gh-567 invariant: grouping and eliding barriers is a pure synchronization transform. *)
    p "depth 1 matches the un-elided barrier structure BITWISE"
      (Array.for_all2_exn got_d1 got_ref ~f:Float.equal);
    let count_sub src sub =
      String.substr_index_all src ~may_overlap:false ~pattern:sub |> List.length
    in
    (match (read_generated "pipe_mm_d1", read_generated "pipe_mm_d2") with
    | Some src1, Some src2 ->
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
    | _ -> p "depth 2 rotates the buffers the depth-1 kernel does not" false);
    (* The count in the EMITTED kernel, where the intrinsic's own brackets are visible too: the
       reference and depth 1 differ by exactly the two barrier statements gh-567 dropped (the k-block
       has one loop body in the source, so this counts statements, not executions). *)
    match (read_generated "pipe_mm_d1", read_generated "pipe_mm_d1_barriers") with
    | Some src1, Some src_ref ->
        let barrier_kw = if on_metal then "threadgroup_barrier" else "__syncthreads" in
        p "the emitted depth-1 kernel sheds exactly the two elided barriers"
          (count_sub src_ref barrier_kw - count_sub src1 barrier_kw = 2)
    | _ -> p "the emitted depth-1 kernel sheds exactly the two elided barriers" false)
  else (
    (* The C backends reject the shared staging composition at compile (workgroup-shared placement
       is not renderable there) — same clean rejection as the SMEM matmul test, at any depth. *)
    let rejects leg =
      try
        ignore (compile_leg leg : Context.t * Context.routine);
        false
      with Invalid_argument msg -> String.is_substring msg ~substring:"not supported"
    in
    p "staged depths approximate the serial twin (GPU) or clean rejection (CPU)"
      (List.for_all legs ~f:rejects);
    skipped "depth 2 matches depth 1 BITWISE";
    skipped "depth 1 matches the un-elided barrier structure BITWISE";
    skipped "depth 2 rotates the buffers the depth-1 kernel does not";
    skipped "the emitted depth-1 kernel sheds exactly the two elided barriers");

  (* --- Structural pins on the applied schedules (all backends; captured by the transforms
     above). --- *)
  let opt_ref, k_o_ref = Hashtbl.find_exn captured 0 in
  let opt_d1, k_o1 = Hashtbl.find_exn captured 1 in
  let opt_d2, k_o2 = Hashtbl.find_exn captured 2 in
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
  let ref_body = body_of k_o_ref opt_ref.LL.llc in
  let d1_body = body_of k_o1 opt_d1.LL.llc in
  let d2_body = body_of k_o2 opt_d2.LL.llc in
  (* Depth 1 (gh-567): the two same-anchor stages share ONE barrier phase per k-block — both load
     nests back to back, then the barrier the compute's reads need — and the phase CLOSER is left
     to [Tile_mma]'s own trailing bracket ("Tile_mma is a barrier"). The k-block's opener stays
     explicit: only some intrinsic rendering forms bracket the block on both sides per iteration
     (the fragment-scope form opens once, outside the anchor loop). Without gh-567 that k-block
     carried FOUR barriers — the reference leg rebuilds three of them (the duplicated closer is
     semantically the same barrier). Depth 2 additionally elides the opener, against the previous
     iteration's bracket: both If-guarded prefetches (counted statement-level, so the lane
     restriction Ifs nested within load nests are not miscounted) sit behind no explicit barrier at
     all. Either way the one explicit barrier left in the routine outside the k-block is the
     trailing one after the contraction's store-back (its elision would need the region contract
     across the store — kept conservative). *)
  p "the un-elided reference carries a barrier per staged phase"
    (barriers ref_body = 3 && barriers opt_ref.LL.llc = 3);
  p "depth 1 groups both stages into one phase and leaves the closer to Tile_mma's bracket"
    (barriers d1_body = 1 && barriers opt_d1.LL.llc = 1);
  p "depth 2 leaves the per-iteration phase to Tile_mma's bracket, one trailing barrier"
    (barriers d2_body = 0 && barriers opt_d2.LL.llc = 1);
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

  (* --- Cache identity: canonical digests distinguish the depths. In phase 1 (depth <= 2) the
     depths already differ in the IR; the pipelined companion section is what keeps depths >= 2
     distinct once the phase-2 async-copy arms widen the legal range (their IR is identical, only
     the rotation modulus differs) — pinned here by digesting depth 2's optimized value against a
     copy whose side-table alone records depth 3 (byte-identical IR by construction), rather than
     discovered as an alias later. --- *)
  let dg opt = SC.digest (SC.canonicalize opt) in
  p "canonical digests distinguish depths 1 and 2"
    (not (String.equal (dg opt_d1) (dg opt_d2)));
  let opt_d2_as_d3 =
    {
      opt_d2 with
      LL.pipelined =
        Map.map opt_d2.LL.pipelined ~f:(fun pt -> { pt with LL.pt_depth = 3 });
    }
  in
  p "the pipelined companion section alone distinguishes same-IR depths"
    (not (String.equal (dg opt_d2) (dg opt_d2_as_d3)));

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
  expect_invalid "pipeline_depth 3 is rejected in phase 1"
    ~substring:"not implemented in phase 1"
    (reapply ~probe:"pipe_mm_probe3" ~f:(fun ~out opt ->
         let schedule, _ = staged_schedule ~pipeline_depth:3 ~out opt in
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
