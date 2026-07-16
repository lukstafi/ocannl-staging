(* Pool-backed Grid rendering on the CPU backends (docs/proposals/gh-ocannl-164.md): within-kernel
   parallelization over the thread grid to match GPU parallelism. Covered here:

   - The automatic CPU schedule ([Schedule.default_cpu], config [automatic_cpu_schedule]) annotates
   a large elementwise kernel, and the C backend renders the outermost Grid loop as chunked parallel
   loops on the native pool ([dispatch_apply] on macOS, OpenMP elsewhere) -- checked structurally on
   the generated source, and by value parity plus a bitwise determinism check. - A kernel below
   [cpu_schedule_min_parallel] stays entirely serial. - Per-chunk local privatization: a
   [Privatize]d matmul accumulator is a stack array written grid-invariantly (on GPU each thread has
   a private copy; one shared function-scope array would race across parallel chunks). Its accesses
   all sit inside the Grid body and each iteration's first access is a covering write (the init-load
   from the target), so the renderer privatizes it to per-chunk block-scope storage and parallelizes
   the loop ([C_syntax.parallel_grid_safe]'s privatization rule, gh-ocannl-469; a local failing that
   rule — e.g. carrying values across iterations — still keeps the loop serial).

   On GPU backends the same programs go through the GPU annotator / hardware binding; every printed
   boolean holds on every backend (structure checks dispatch on the configured backend). *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module LL = Ir.Low_level
module Sched = Ir.Schedule
module Asgns = Ir.Assignments

let () = Utils.settings.output_debug_files_in_build_directory <- true

(* Progress markers on stderr: the test's stdout is captured into [.exe.output], but stderr reaches
   dune's log, so a crash (e.g. in a native worker thread) is attributable in CI. *)
let phase name = Stdio.eprintf "cpu_parallel phase: %s\n%!" name
let p name b = Stdio.printf "%s: %b\n" name b
let approx a b = Float.(abs (a - b) < 1e-2)
let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")
let on_cpu = String.is_substring backend_name ~substring:"cc"

let read_generated base_name =
  let ext = if String.is_substring backend_name ~substring:"metal" then ".metal" else ".c" in
  let ext = if String.is_substring backend_name ~substring:"cuda" then ".cu" else ext in
  let ext = if String.is_substring backend_name ~substring:"hip" then ".hip" else ext in
  let path = Utils.build_file (base_name ^ ext) in
  if Stdlib.Sys.file_exists path then Some (Stdio.In_channel.read_all path) else None

let has_parallel_construct src =
  String.is_substring src ~substring:"dispatch_apply"
  || String.is_substring src ~substring:"#pragma omp parallel for"

let named name (comp : Asgns.comp) : Asgns.comp =
  { comp with asgns = Asgns.Block_comment (name, comp.asgns) }

(* The single-child chain of loops from the top of each top-level nest. *)
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

let () =
  (* --- Large elementwise kernel through the automatic schedule (no lowered_transform): 512 x 512 =
     262144 parallel iterations, above every threshold. --- *)
  let n = 512 in
  let av = Array.init (n * n) ~f:(fun i -> Float.of_int (i % 19) *. 0.5) in
  let bv = Array.init (n * n) ~f:(fun i -> Float.of_int (i % 23) -. 11.) in
  let expected = Array.init (n * n) ~f:(fun i -> av.(i) +. bv.(i)) in
  let a = TDSL.ndarray av ~label:[ "a" ] ~output_dims:[ n; n ] () in
  let b = TDSL.ndarray bv ~label:[ "b" ] ~output_dims:[ n; n ] () in
  let%op c = a + b in
  let comp = named "cpu_par_auto" (Train.forward c) in
  let ctx = Context.auto () in
  phase "auto: compile";
  let ctx, routine = Context.compile ctx comp Ir.Indexing.Empty in
  phase "auto: first run";
  let ctx = Context.run ctx routine in
  let got1 = Context.get_values ctx c.Tensor.value in
  p "automatic schedule values correct" (Array.for_all2_exn got1 expected ~f:approx);
  phase "auto: second run";
  let ctx = Context.run ctx routine in
  let got2 = Context.get_values ctx c.Tensor.value in
  p "parallel runs bitwise deterministic" (Array.equal Float.( = ) got1 got2);
  (match read_generated "cpu_par_auto" with
  | None -> p "large kernel rendered in parallel" false
  | Some src ->
      let ok =
        if on_cpu then has_parallel_construct src
        else (* GPU backends bind hardware indices instead. *)
          not (has_parallel_construct src)
      in
      p "large kernel rendered in parallel" ok);

  (* --- Small kernel: below [cpu_schedule_min_parallel], must stay serial. --- *)
  phase "small kernel";
  let m = 8 in
  let sv = Array.init (m * m) ~f:(fun i -> Float.of_int i) in
  let s1 = TDSL.ndarray sv ~label:[ "s1" ] ~output_dims:[ m; m ] () in
  let%op s2 = s1 *. s1 in
  let comp_small = named "cpu_par_small" (Train.forward s2) in
  let ctx_sm = Context.auto () in
  let ctx_sm, routine_sm = Context.compile ctx_sm comp_small Ir.Indexing.Empty in
  let ctx_sm = Context.run ctx_sm routine_sm in
  let got_small = Context.get_values ctx_sm s2.Tensor.value in
  p "small kernel values correct"
    (Array.for_all2_exn got_small (Array.map sv ~f:(fun x -> x *. x)) ~f:approx);
  (match read_generated "cpu_par_small" with
  | None -> p "small kernel stays serial" false
  | Some src -> p "small kernel stays serial" (not (has_parallel_construct src)));

  (* --- Per-chunk local privatization: privatized matmul accumulator under an explicit Grid retype.
     The accumulator tile is a stack array whose accesses do not mention the grid index — parallel
     chunks sharing one function-scope copy would race — but every iteration init-loads it before
     reading, so the renderer declares it per chunk inside the parallel construct and both nests
     parallelize (values must still match the twin). --- *)
  phase "matmul twin";
  let k = 64 in
  let mav = Array.init (k * k) ~f:(fun i -> Float.of_int (i % 13) *. 0.25) in
  let mbv = Array.init (k * k) ~f:(fun i -> Float.of_int (i % 17) -. 8.) in
  let ma = TDSL.ndarray mav ~label:[ "ma" ] ~input_dims:[ k ] ~output_dims:[ k ] () in
  let mb = TDSL.ndarray mbv ~label:[ "mb" ] ~input_dims:[ k ] ~output_dims:[ k ] () in
  let run_mm ~name ~transform mc =
    let comp = named name (Train.forward mc) in
    let ctx = Context.auto () in
    let ctx, routine = Context.compile ~lowered_transform:transform ctx comp Ir.Indexing.Empty in
    let ctx = Context.run ctx routine in
    Context.get_values ctx mc.Tensor.value
  in
  let%op mc0 = ma * mb in
  let mm_twin = run_mm ~name:"cpu_par_twin" ~transform:(fun opt -> opt) mc0 in
  phase "matmul privatized";
  let%op mc1 = ma * mb in
  let mm_priv =
    run_mm ~name:"cpu_par_privatized"
      ~transform:(fun opt ->
        let paths = nest_paths opt.LL.llc in
        let accum = List.find_exn paths ~f:(fun path -> List.length path = 3) in
        let i = List.hd_exn accum and red = List.last_exn accum in
        (* Whole-node [Zero_out] of a materialized node is rejected in multi-threaded kernels;
           expand it and give the zeroing nest the same one-Grid geometry as the accumulation. *)
        let zop, zsyms = Sched.expand_zero ~tn:mc1.Tensor.value in
        Sched.apply
          [
            zop;
            Sched.Retype { axis = List.hd_exn zsyms; ty = LL.Grid };
            Sched.Retype { axis = i; ty = LL.Grid };
            Sched.Privatize { target = mc1.Tensor.value; over = red };
          ]
          opt)
      mc1
  in
  p "privatized matmul values match the twin" (Array.for_all2_exn mm_priv mm_twin ~f:approx);
  match read_generated "cpu_par_privatized" with
  | None -> p "privatized accumulator gets per-chunk storage, both nests parallel" false
  | Some src ->
      (* The zeroing nest parallelizes (its write covers its grid index); the accumulation nest's
         grid loop parallelizes too, with the privatized accumulator declared per chunk inside the
         parallel construct (its init-load makes each iteration self-contained). So on CPU: two
         parallel constructs; on GPU: hardware bindings, none. *)
      let count =
        String.substr_index_all src ~may_overlap:false ~pattern:"Pool-backed Grid rendering"
        |> List.length
      in
      p "privatized accumulator gets per-chunk storage, both nests parallel"
        (if on_cpu then count = 2 else count = 0);

      (* --- Privatization write-dominance edge (Codex P2 on PR #159): a local whose covering write
         shares its loop with a read of the local, [for x { tmp[x] = ..; use tmp[0] }] — the write
         nest never completes before the reads execute, so per-chunk storage is not provably
         equivalent and the grid loop must stay serial. Hand-built body through the transform seam
         (no in-tree transform emits this shape); on GPU the Grid axis binds in hardware, where
         per-thread locals make it trivially legal. --- *)
      phase "interleaved hazard";
      let%op hz = ma + ma in
      let hazard_transform (opt : LL.optimized) : LL.optimized =
        let out_tn = hz.Tensor.value in
        let scratch =
          Ir.Tnode.create ~namespace:"cptest"
            (Ir.Tnode.Specified (Lazy.force out_tn.Ir.Tnode.prec))
            ~id:0 ~label:[ "hazard"; "scratch" ]
            ~unpadded_dims:(lazy [| k |])
            ~padding:(lazy None)
            ()
        in
        Ir.Tnode.Placements.update opt.LL.optimize_ctx.LL.placements scratch Ir.Tnode.Local 999;
        ignore (LL.get_node opt.LL.traced_store scratch : LL.traced_array);
        let i = Ir.Indexing.get_symbol () and x = Ir.Indexing.get_symbol () in
        let body =
          LL.Seq
            ( LL.Set
                {
                  tn = scratch;
                  idcs = [| Ir.Indexing.Iterator x |];
                  llsc =
                    LL.Get (ma.Tensor.value, [| Ir.Indexing.Iterator i; Ir.Indexing.Iterator x |]);
                  debug = "";
                },
              LL.Set
                {
                  tn = out_tn;
                  idcs = [| Ir.Indexing.Iterator i; Ir.Indexing.Iterator x |];
                  llsc = LL.Get (scratch, [| Ir.Indexing.Fixed_idx 0 |]);
                  debug = "";
                } )
        in
        let llc =
          LL.For_loop
            {
              index = i;
              from_ = 0;
              to_ = k - 1;
              axis = LL.Grid;
              trace_it = false;
              body =
                LL.For_loop
                  { index = x; from_ = 0; to_ = k - 1; axis = LL.Serial; trace_it = false; body };
            }
        in
        { opt with llc }
      in
      let got_hz = run_mm ~name:"cpu_par_hazard" ~transform:hazard_transform hz in
      (* Per (i, x): scratch[x] := ma[i, x] then out[i, x] := scratch[0], so every row of the output
         holds its ma row's first element (scratch[0] is rewritten at x = 0 before any read of it in
         the same grid iteration). *)
      let want_hz = Array.init (k * k) ~f:(fun idx -> mav.(idx / k * k)) in
      p "interleaved-write hazard values correct" (Array.for_all2_exn got_hz want_hz ~f:approx);
      (match read_generated "cpu_par_hazard" with
      | None -> p "interleaved covering write keeps the grid loop serial" false
      | Some src ->
          p "interleaved covering write keeps the grid loop serial"
            (not (has_parallel_construct src)));
      phase "done"
