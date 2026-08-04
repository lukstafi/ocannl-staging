(* 16-bit storage with f32 compute: does the halved memory traffic actually show up?
   (gh-ocannl-517 task 3.)

   Three streaming kernels over large 1-D arrays, each run at f32, bf16 and half storage, in the
   default rendering and with the innermost loop explicitly retyped [Vectorized]. Narrow storage
   moves half the bytes of the f32 leg; the compute is f32 either way, so a bandwidth-bound kernel
   should approach 2x and a compute-bound one should not move. That contrast is the point: it is
   what distinguishes a real traffic win from a measurement artifact.

   The reported GB/s counts the bytes the kernel's own precision moves, so the number is directly
   comparable to the machine's stream bandwidth; the "vs f32" column is the speedup that matters
   for the issue's claim.

   Measured on an Apple-Silicon M-series (NEON via -march=native, single stream, n = 2^22, 100
   repeats), "vs f32" column:

   {v
     kernel      rendering   f32 GB/s   bf16    half
     add         default       131.3    0.91x   1.97x
     mul_add     default       131.2    0.88x   1.56x
     polynomial  default        89.0    0.61x   0.76x
   v}

   Two findings, and the second is the surprise:

   - The traffic win is real where the kernel is actually bandwidth-bound. "add" at f32 runs at the
     machine's stream ceiling, and half storage nearly reaches the theoretical 2x. The compute-bound
     control stays below 1x, as it must -- there is no traffic to save there, only conversions to
     pay for. So the mechanism does what gh-ocannl-517 predicts.

   - It is *fp16*, not bf16, that collects the win on this target, the reverse of the issue's
     expectation. bf16's conversion is cheap in instruction count but not free: widening is a
     zero-extend plus a shift and narrowing is four vector ops (the round-to-nearest-even of
     [single_to_bfloat16]), which at 130 GB/s costs more than halving the bytes saves. fp16's
     conversion is a single NEON instruction each way. The route to making bf16 competitive is a
     hardware convert ([BFCVT] on ARMv8.6-A, AVX512-BF16 on x86) rather than the portable bit
     arithmetic -- with the caveat that hardware conversion would have to be shown to agree with
     [single_to_bfloat16] bitwise, NaN payloads included, or it breaks the parity the vectorized
     rendering owes its serial twin.

   Usage: dune exec bin/narrow_storage_bench.exe -- [n] [repeats] [threads] (defaults 4194304, 50,
   1).

   As in {!Cpu_vectorization_bench}, with no backend configured this pins the single-stream C
   backend rather than letting [Context.auto]'s metal/cuda-first order land a "CPU" benchmark on a
   GPU. An explicitly configured backend is respected -- but note the seam is CPU-only by design
   ({!Ir.Numerics.narrow_compute_f32}), so a GPU run measures native 16-bit arithmetic instead.

   Cross-check against the analytic cost model (gh-ocannl-491): its footprint extraction reads
   [prec_in_bytes] off each node, so the narrow legs' modelled bytes are exactly half, and the
   measured ratio below is the calibration of that prediction. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module LL = Ir.Low_level
module Sched = Ir.Schedule
module Tn = Ir.Tnode

let p = Stdio.printf

(* The innermost loop of the first top-level nest. *)
let rec innermost_loop (llc : LL.t) : Ir.Indexing.symbol option =
  let strip stmts = List.filter stmts ~f:(function LL.Noop | LL.Comment _ -> false | _ -> true) in
  match llc with
  | LL.Seq (a, b) -> ( match innermost_loop a with Some r -> Some r | None -> innermost_loop b)
  | LL.For_loop { index; body; _ } -> (
      match strip (LL.flat_lines [ body ]) with
      | [ single ] -> ( match innermost_loop single with Some r -> Some r | None -> Some index)
      | _ -> Some index)
  | LL.If { body; _ } -> innermost_loop body
  | _ -> None

let () =
  let argv = Sys.get_argv () in
  let pos_args =
    Array.filteri argv ~f:(fun i a -> i > 0 && not (String.is_prefix a ~prefix:"--"))
  in
  let n = if Array.length pos_args > 0 then Int.of_string pos_args.(0) else 1 lsl 22 in
  let repeats = if Array.length pos_args > 1 then Int.of_string pos_args.(1) else 50 in
  (* Threads: 1 (the default) measures one core's stream; 0 uses the whole pool, which is where a
     CPU actually reaches its memory ceiling and where a traffic win can show at all. *)
  let threads = if Array.length pos_args > 2 then Int.of_string pos_args.(2) else 1 in
  (* O(1) magnitudes so bf16's ~3 significant decimal digits still leave the result recognizable. *)
  let av = Array.init n ~f:(fun i -> 0.5 +. (Float.of_int (i % 1009) /. 1009.)) in
  let bv = Array.init n ~f:(fun i -> 0.5 +. (Float.of_int (i % 997) /. 997.)) in
  let make_ctx () =
    match Utils.get_global_arg ~arg_name:"backend" ~default:"" with
    | "" -> if threads > 0 then Context.cpu ~threads () else Context.cpu ()
    | _ -> Context.auto ()
  in
  let named name (comp : Ir.Assignments.comp) =
    { comp with Ir.Assignments.asgns = Ir.Assignments.Block_comment (name, comp.asgns) }
  in
  let serial opt = opt in
  let vectorize (opt : LL.optimized) =
    match innermost_loop opt.LL.llc with
    | None -> opt
    | Some j -> Sched.apply [ Sched.Retype { axis = j; ty = LL.Vectorized } ] opt
  in
  (* One (kernel, precision, rendering) measurement. [build] gets fresh leaves so each leg owns its
     own storage; [reads] is how many n-element arrays the kernel streams in (plus one written). *)
  let bench ~name ~build ~reads ~prec ~transform ~label =
    (* Minted at [prec] rather than re-tagged: [ndarray] settles a leaf's precision as [Specified],
       which [update_prec] then refuses to change. *)
    let leaf l vals =
      NTDSL.init ~l ~prec ~o:[ n ] ~f:(function [| i |] -> vals.(i) | _ -> assert false) ()
    in
    let a = leaf (label ^ "_a") av and b = leaf (label ^ "_b") bv in
    let t : Tensor.t = build a b in
    Tn.update_prec t.Tensor.value prec;
    let comp = named label (Train.forward t) in
    let ctx = make_ctx () in
    let ctx, routine = Context.compile ~lowered_transform:transform ctx comp Ir.Indexing.Empty in
    let ctx = Context.run ctx routine in
    let first = (Context.get_values ctx t.Tensor.value).(0) in
    (* [get_values] is deliberately outside the timed region: it converts the whole device buffer
       into an OCaml [float array] element by element, which costs O(n) *independently of the
       storage precision* and would swamp -- and exactly mask -- the traffic difference this
       benchmark exists to measure. The cc backend's scheduler is synchronous, so the fold below
       needs no separate await. *)
    let start = Time_now.nanoseconds_since_unix_epoch () in
    let ctx =
      Stdlib.Array.fold_left
        (fun ctx () -> Context.run ctx routine)
        ctx
        (Stdlib.Array.make repeats ())
    in
    let stop = Time_now.nanoseconds_since_unix_epoch () in
    let (_ : float array) = Context.get_values ctx t.Tensor.value in
    let secs = Float.of_int63 Int63.(stop - start) /. 1e9 /. Float.of_int repeats in
    let bytes = Float.of_int ((reads + 1) * n * Ir.Ops.prec_in_bytes prec) in
    ignore name;
    (secs, bytes /. secs /. 1e9, first)
  in
  let kernels =
    [
      (* Pure streaming: one add per 3 array touches -- the bandwidth-bound case the issue is
         about. *)
      ("add", (fun a b -> [%op a + b]), 2);
      (* Still streaming, twice the arithmetic per byte. *)
      ("mul_add", (fun a b -> [%op (a *. b) + a]), 2);
      (* Compute-bound control: the intermediates are virtual, so the byte traffic is unchanged
         from "add" while the FLOPs per byte are ~15x. Narrow storage should NOT speed this up
         much -- if it does, the "add" number is measuring something other than traffic. *)
      ( "polynomial",
        (fun a b ->
          let%op t = ((a *. b) + a) *. ((a *. b) + b) *. ((a + b) *. a) in
          let%op u = (t *. t) + t in
          let%op v = (u *. u) + u in
          [%op (v *. v) + v]),
        2 );
    ]
  in
  let precs =
    [ ("f32", Ir.Ops.single); ("bf16", Ir.Ops.bfloat16); ("half", Ir.Ops.half) ]
  in
  let renderings = [ ("default", serial); ("vectorized", vectorize) ] in
  p "narrow storage bench: n = %d, %d repeats, threads = %d, narrow_compute_f32 = %b\n" n repeats
    threads (Ir.Numerics.get ()).Ir.Numerics.narrow_compute_f32;
  p "%-11s %-11s %-6s %10s %10s %8s  %s\n" "kernel" "rendering" "prec" "us" "GB/s" "vs f32"
    "first value";
  List.iter kernels ~f:(fun (name, build, reads) ->
      List.iter renderings ~f:(fun (rname, transform) ->
          let baseline = ref None in
          List.iter precs ~f:(fun (pname, prec) ->
              let label = Printf.sprintf "nsb_%s_%s_%s" name rname pname in
              let secs, gbs, first = bench ~name ~build ~reads ~prec ~transform ~label in
              let ratio =
                match !baseline with
                | None ->
                    baseline := Some secs;
                    1.0
                | Some base -> base /. secs
              in
              p "%-11s %-11s %-6s %10.1f %10.2f %8.2fx  %g\n" name rname pname (secs *. 1e6) gbs
                ratio first)))
