(* CPU auto-vectorization benchmark (gh-ocannl-164, Phase 4): throughput of a float32 elementwise
   add and a dot-product reduction over large arrays on the configured (C) backend, with
   correctness spot-checked against OCaml-computed references.

   Usage: dune exec bin/cpu_vectorization_bench.exe -- [n] [repeats] (defaults 1048576 and 100).
   With no backend configured this pins the single-stream C backend itself: the root
   ocannl_config is a personal, gitignored file, so in fresh clones / CI / worktrees there is no
   config along the cwd path and [Context.auto]'s preference order would silently land a "CPU"
   benchmark on metal/cuda. A configured backend (personal ocannl_config, OCANNL_BACKEND,
   --ocannl_backend) is respected, e.g. to compare multicore_cc.

   To measure the vectorization delta, compare the default run against one with the compiler's
   vectorizers disabled at the same optimization level (cc_backend_simd_flags is appended verbatim
   to the compiler invocation, so it can carry arbitrary extra flags):

     dune exec bin/cpu_vectorization_bench.exe
     dune exec bin/cpu_vectorization_bench.exe -- \
       --ocannl_cc_backend_simd_flags="-fno-vectorize -fno-slp-vectorize"

   (the -fno-* spelling above is Clang's; for GCC use "-fno-tree-vectorize -fno-tree-slp-vectorize".
   The acceptance criterion is >= 2x on a float32 elementwise op or reduction at n >= 1024: on an
   Apple-Silicon M-series (NEON via -march=native) the compute-bound polynomial measures 2.0x —
   36.5 vs 18.2 GFLOP/s at the defaults; "add" is memory-bound and shows the bandwidth ceiling
   instead, and the strict-FP "dot" reduction cannot auto-vectorize without reassociation
   permission (enable cc_backend_fast_math=true to see it vectorize). *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules

let p = Stdio.printf

let () =
  let argv = Sys.get_argv () in
  let pos_args = Array.filteri argv ~f:(fun i a -> i > 0 && not (String.is_prefix a ~prefix:"--")) in
  let n = if Array.length pos_args > 0 then Int.of_string pos_args.(0) else 1 lsl 20 in
  let repeats = if Array.length pos_args > 1 then Int.of_string pos_args.(1) else 100 in
  (* Positive, O(1) values: a float32 dot product over ~10^6 elements accumulates rounding error;
     keeping all products positive avoids cancellation so a loose relative tolerance suffices. *)
  let av = Array.init n ~f:(fun i -> Float.of_int (i % 1009) /. 1009.) in
  let bv = Array.init n ~f:(fun i -> Float.of_int (i % 997) /. 997.) in
  let a = TDSL.ndarray av ~label:[ "a" ] ~output_dims:[ n ] () in
  let b = TDSL.ndarray bv ~label:[ "b" ] ~output_dims:[ n ] () in
  let named name (comp : Ir.Assignments.comp) =
    { comp with Ir.Assignments.asgns = Ir.Assignments.Block_comment (name, comp.asgns) }
  in
  (* A CPU benchmark must not silently land on a GPU backend: with no backend configured
     (config file / env / cmdline), pin the single-stream C backend instead of [Context.auto]'s
     metal/cuda-first preference order. An explicitly configured backend is respected. *)
  let make_ctx () =
    match Utils.get_global_arg ~arg_name:"backend" ~default:"" with
    | "" -> Context.cpu ~threads:1 ()
    | _ -> Context.auto ()
  in
  let bench ~variant ~(t : Tensor.t) ~check ~work_per_run ~unit_label =
    let comp = named variant (Train.forward t) in
    let ctx = make_ctx () in
    let ctx, routine = Context.compile ctx comp Ir.Indexing.Empty in
    (* Warmup: lazy initialization and host transfers. *)
    let ctx = Context.run ctx routine in
    let values = Context.get_values ctx t.Tensor.value in
    let ok = check values in
    let start = Time_now.nanoseconds_since_unix_epoch () in
    let ctx =
      Stdlib.Array.fold_left
        (fun ctx () -> Context.run ctx routine)
        ctx
        (Stdlib.Array.make repeats ())
    in
    let (_ : float array) = Context.get_values ctx t.Tensor.value in
    let stop = Time_now.nanoseconds_since_unix_epoch () in
    let secs = Float.of_int63 Int63.(stop - start) /. 1e9 /. Float.of_int repeats in
    p "%-12s %10.3f us  %8.2f %s  correct: %b\n" variant (secs *. 1e6)
      (work_per_run /. secs /. 1e9) unit_label ok
  in
  p "cpu vectorization bench: n = %d, %d repeats, backend from config/OCANNL_BACKEND\n" n repeats;
  let approx x y = Float.(abs (x - y) <= 1e-3 *. (1. +. abs y)) in
  (let%op c = a + b in
   bench ~variant:"add" ~t:c
     ~check:(fun got ->
       Array.length got = n
       && Array.for_alli got ~f:(fun i g -> approx g (av.(i) +. bv.(i))))
     (* 2 reads + 1 write of 4 bytes per element; memory-bound, reported for context. *)
     ~work_per_run:(Float.of_int (12 * n))
     ~unit_label:"GB/s   ");
  (* Compute-bound elementwise chain: the pointwise intermediates are virtual and inline into one
     loop (low-level CSE dedups the reuses), so vectorization — not memory bandwidth — sets the
     throughput. This is where the >= 2x acceptance criterion is measured. *)
  (let%op t = (((a *. b) + a) *. ((a *. b) + b) *. ((a + b) *. a)) + ((b *. b) *. (a + b)) in
   let%op u = (t *. t) + t in
   let%op v = (u *. u) + u in
   let%op poly = (v *. v) + v in
   let poly_ref i =
     let x = av.(i) and y = bv.(i) in
     let p = x *. y in
     let t = (((p +. x) *. (p +. y)) *. ((x +. y) *. x)) +. (y *. y *. (x +. y)) in
     let u = (t *. t) +. t in
     let v = (u *. u) +. u in
     (v *. v) +. v
   in
   bench ~variant:"polynomial" ~t:poly
     ~check:(fun got ->
       Array.length got = n && Array.for_alli got ~f:(fun i g -> approx g (poly_ref i)))
     (* 17 pointwise multiply/adds per element. *)
     ~work_per_run:(Float.of_int (17 * n))
     ~unit_label:"GFLOP/s");
  let%op d = a +* "i;i=>0" b in
  let dot_ref = Array.foldi av ~init:0. ~f:(fun i acc x -> acc +. (x *. bv.(i))) in
  bench ~variant:"dot" ~t:d
    ~check:(fun got ->
      (* The kernel accumulates in float32 (in whatever order vectorization picks); the reference
         fold is double. With all-positive products the drift stays well under 1%. *)
      Array.length got = 1 && Float.(abs (got.(0) - dot_ref) <= 1e-2 *. (1. +. abs dot_ref)))
    (* multiply + add per element. *)
    ~work_per_run:(Float.of_int (2 * n))
    ~unit_label:"GFLOP/s"
