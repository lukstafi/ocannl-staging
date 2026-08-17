(* Narrow-operand GEBP benchmark (gh-ocannl-575): the CPU register-tiled [Tile_mma] over 16-bit
   storage, with the widening folded into the packing [Stage] ([tile_prec]). Times, on the C
   backends, the serial naive kernel against the all-Serial packed GEBP and the pool-parallel
   grid-outermost per-chunk-packing flavor (schedule_bench's packmma / pm_bpk shapes), at the
   requested storage precision — the packed panels are minted at the compute precision the
   emission resolves ([Numerics.cpu_compute_prec] over [Context.hardware_limits]), so an f32 run
   measures the plain GEBP, a bf16/f16 run measures f32-GEBP-over-narrow-storage, and an f16 run
   with --ocannl_fp16_arithmetic=true on a native-arithmetic target (NEON, AVX512-FP16) measures
   the pure-f16 GEBP — the honest first comparison the issue asks for. On a target that merely
   promotes ([cc_fp16_arithmetic] probes it), the policy is ignored and the run stays f32-compute.

   Usage: OCANNL_BACKEND=cc dune exec bin/narrow_gebp_bench.exe -- [f32|bf16|f16] [n] [repeats]
   [bm] [bk] (defaults f32, 512, 20, 64, 256; [bm]/[bk] also as [--bm=]/[--bk=] flags). The packed
   variants schedule [Sched.split]s of factor [bm] over the i axis and [bk] over the k axis, so
   they need [n mod bm = 0] and [n mod bk = 0] (and an n big enough to leave a loop nest at all);
   an unschedulable n runs the unblocked naive variant alone, so an arbitrary extent still has
   something to compare against. Each line carries a position-weighted checksum of the whole
   output, which is what makes a mishandled edge region visible — the register-tiled micro-kernel
   peels row and column edges whenever its block shape does not cover the extent, and a single
   interior cell sees none of that. In a narrow-storage run the naive and packed legs accumulate
   at different widths, so the run says outright which checksums are comparable. Each packed line
   carries its
   [C_syntax.mma_census] rendering, because a [Tile_mma] whose preconditions fail (a column extent
   below the compute vector width is one of several ways in) renders the scalar fallback while
   still reporting as "packmma" — read the bracket, not the variant name, when deciding what a
   timing measured. Readbacks stay outside the timed region (the [Context.get_values] trap,
   docs/agent-notes.md). *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module LL = Ir.Low_level
module Sched = Ir.Schedule
module Asgns = Ir.Assignments
module Numerics = Ir.Numerics

let p = Stdio.printf

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

let sink sym below = List.map below ~f:(fun inner -> Sched.Swap { outer = sym; inner })

(* An aperiodic mix of the two indices, used to mint the operand values below. Aperiodic is the
   point: any value drawn from [index mod p] repeats under [k -> k + p], so if both operands share
   that period every packed K panel is identical and a staging bug that substitutes or repeats the
   wrong panel is invisible — and [bk] is a user argument, so no fixed period can be assumed to be
   coprime to it. Mixing has no shift symmetry at all, which removes the class rather than dodging
   it. Every intermediate is masked below 2^54, so nothing overflows a 63-bit int and the sequence
   is reproducible by an external oracle. *)
let mix ~salt a b =
  let x = (a * 73856093) lxor (b * 19349663) lxor salt in
  let x = (x lxor (x lsr 13)) land 0xFFFFFF in
  let x = x * 1274126177 land 0xFFFFFF in
  x lxor (x lsr 7)

let () =
  (* An option is [--]-prefixed (ours, and the [--ocannl_*] config flags) or a [-] followed by a
     non-digit. A bare [-64] is therefore a positional, not an option: dropping it would silently
     shift every later positional into the wrong slot and hide the invalid value from the
     validation below. *)
  let is_option s =
    String.is_prefix s ~prefix:"--"
    || (String.is_prefix s ~prefix:"-" && String.length s > 1 && not (Char.is_digit s.[1]))
  in
  let pos_args =
    Array.to_list (Sys.get_argv ()) |> List.tl_exn |> List.filter ~f:(fun s -> not (is_option s))
  in
  let prec_name = Option.value (List.nth pos_args 0) ~default:"f32" in
  let prec =
    match prec_name with
    | "f32" -> Ir.Ops.single
    | "bf16" -> Ir.Ops.bfloat16
    | "f16" -> Ir.Ops.half
    | s -> invalid_arg ("narrow_gebp_bench: precision f32|bf16|f16 expected, got " ^ s)
  in
  let arg i default =
    match List.nth pos_args i with Some s -> Int.of_string s | None -> default
  in
  let flag name =
    let prefix = "--" ^ name ^ "=" in
    Array.to_list (Sys.get_argv ())
    |> List.find_map ~f:(fun s -> Option.map (String.chop_prefix s ~prefix) ~f:Int.of_string)
  in
  let n = arg 1 512 in
  let repeats = arg 2 20 in
  let bm = Option.value (flag "bm") ~default:(arg 3 64) in
  let bk = Option.value (flag "bk") ~default:(arg 4 256) in
  (* Every integer argument is a positive extent or count; validate them as one domain rather than
     leaving each use site to discover its own bad input. *)
  List.iter
    [ ("n", n); ("repeats", repeats); ("bm", bm); ("bk", bk) ]
    ~f:(fun (name, v) ->
      if v < 1 then
        invalid_arg (Printf.sprintf "narrow_gebp_bench: %s must be positive, got %d" name v));
  (* What the packed variants require of n, in one place: an i/j/k nest to address (extent-1 loops
     are simplified away before the transform runs, leaving nothing to schedule), and extents
     divisible by the [Sched.split] factors. The naive variant has no blocking at all, so it runs
     for any n. *)
  let unschedulable =
    (if n < 2 then [ Printf.sprintf "n = %d leaves no i/j/k loop nest to schedule" n ] else [])
    @ List.filter_map [ ("bm", bm); ("bk", bk) ] ~f:(fun (name, f) ->
          if n % f = 0 then None
          else Some (Printf.sprintf "%s = %d does not divide n = %d (remainder %d)" name f n (n % f)))
  in
  let flops = 2.0 *. Float.of_int n *. Float.of_int n *. Float.of_int n in
  (* Operand values that vary with EVERY index at every n, drawn through [mix] so that no shift of
     any index is a symmetry, and exactly representable at every storage precision (the parity-test
     recipes). Two traps sit behind this, both of which cost a wrong version to find. The value must
     not be a modulus of the FLATTENED offset [(i * n + j) % p] — that loses its row dependence
     exactly when p divides n, which made an earlier ma constant along i at 3 | n and an earlier mb
     constant along k at 5 | n, with n = 960 collapsing both; a transform substituting the wrong row
     of a collapsed operand then computes the correct output, which no whole-output check can see.
     And reducing each index separately fixes that but leaves the period: with both operands drawn
     mod 5 in k, every packed K panel repeats under [k -> k + 5], so a run with [bk = 5] hides a
     panel-substitution bug just as thoroughly. [mix] has no shift symmetry at any lag, so neither
     survives — measured over n = 2..2000 for the collapse, over lags 1..256 for the shift, and as
     zero duplicate full B~ panels at every bk from 1 to 64.

     The value SETS are the original ones — ma in {0.25, 0.5, 0.75}, mb in {-1, -0.5, 0, 0.5, 1} —
     because the resulting 1/8 product granularity is load-bearing in a narrow-storage run (see the
     checksum's note below; a finer-grained attempt overflowed the bf16 mantissa and split the legs
     at n = 256). ma is strictly positive, so a dropped term cannot cancel into something plausible;
     mb is signed and near-mean-zero, which makes partial sums random-walk rather than grow with n
     — measured max |partial sum| 18 at n = 256 and 31 at n = 512, against the ~0.075n a
     periodic-in-k pairing produced. *)
  let ma =
    NTDSL.init ~l:"ma" ~prec ~i:[ n ] ~o:[ n ]
      ~f:(fun idcs -> Float.of_int (1 + (mix ~salt:0x5A17 idcs.(0) idcs.(1) % 3)) *. 0.25)
      ()
  in
  let mb =
    NTDSL.init ~l:"mb" ~prec ~i:[ n ] ~o:[ n ]
      ~f:(fun idcs -> Float.of_int ((mix ~salt:0x3C6E idcs.(0) idcs.(1) % 5) - 2) *. 0.5)
      ()
  in
  let packed_schedule ~grid ~tile_prec ~mc (opt : LL.optimized) : Sched.schedule =
    let paths = nest_paths opt.LL.llc in
    let i, j, k =
      (* The i/j/k nest is what the packed schedule addresses. Extent-1 loops are simplified away
         before the transform runs, so a degenerate n leaves no 3-deep nest to schedule — report
         that rather than raising [Not_found_s] out of a [find_exn]. *)
      match List.find paths ~f:(fun p -> List.length p = 3) with
      | Some [ i; j; k ] -> (i, j, k)
      | _ ->
          failwith
            (Printf.sprintf
               "narrow_gebp_bench: no 3-deep i/j/k loop nest to schedule at n = %d (deepest nest \
                found: %d loops) — the packed variants need a non-degenerate extent"
               n
               (List.fold paths ~init:0 ~f:(fun acc p -> Int.max acc (List.length p))))
    in
    let outer_i = if grid then LL.Grid else LL.Serial in
    let sp_i, i_o, i_i = Sched.split ~axis:i ~factor:bm ~outer:outer_i ~inner:LL.Serial in
    let sp_k, k_o, k_i = Sched.split ~axis:k ~factor:bk ~outer:LL.Serial ~inner:LL.Serial in
    let stage source tile_loops =
      Sched.Stage
        {
          source;
          tile_loops;
          shared = false;
          cooperative = None;
          hoisted = false;
          swizzle = None;
          pad_stride = None;
          pipeline_depth = 1;
          tile_prec;
        }
    in
    let zops =
      if not grid then []
      else
        let ez, zsyms = Sched.expand_zero ~tn:mc in
        let zi = List.hd_exn zsyms in
        let sp_zi, _, _ = Sched.split ~axis:zi ~factor:bm ~outer:LL.Grid ~inner:LL.Serial in
        [ ez; sp_zi ]
    in
    let tz, _lane = Sched.tensorize ~i:i_i ~j ~k:k_i ~simd_width:1 in
    zops @ [ sp_i; sp_k ]
    @ sink j [ k_o ]
    @ sink i_i [ k_o ]
    @ (if grid then [] else sink i_o [ k_o ])
    @ [ stage mb.Tensor.value [ k_i; j ]; stage ma.Tensor.value [ i_i; k_i ] ]
    @ [ tz ]
  in
  let bench ~variant ~schedule () =
    let%op mc = ma * mb in
    Ir.Tnode.update_prec mc.Tensor.value prec;
    let comp = named ("ngb_" ^ variant) (Train.forward mc) in
    let transform opt =
      match schedule with None -> opt | Some s -> Sched.apply (s ~mc:mc.Tensor.value opt) opt
    in
    let ctx = Context.auto () in
    (* The census records how codegen actually rendered each [Tile_mma] (gh-ocannl-479). A
       [Tile_mma] whose preconditions fail renders the scalar fallback and still reports as
       "packmma", so an unchecked run can present fallback timings as register-tiled ones — the
       column extent below the compute vector width is one way in, but so are a narrow
       [vector_bytes], a mixed operand precision, and [debug_log_from_routines]. Collecting the
       census only appends to a list, so it does not perturb what is compiled or timed. *)
    Ir.C_syntax.mma_census := [];
    Ir.C_syntax.mma_census_enabled := true;
    let ctx, routine =
      Exn.protect
        ~f:(fun () -> Context.compile ~lowered_transform:transform ctx comp Ir.Indexing.Empty)
        ~finally:(fun () -> Ir.C_syntax.mma_census_enabled := false)
    in
    let renderings = List.rev_map !Ir.C_syntax.mma_census ~f:snd in
    let ctx = Context.run ctx routine in
    let _ = Context.get_values ctx mc.Tensor.value in
    let start = Time_now.nanoseconds_since_unix_epoch () in
    let ctx =
      Stdlib.Array.fold_left
        (fun ctx () -> Context.run ctx routine)
        ctx (Stdlib.Array.make repeats ())
    in
    let stop = Time_now.nanoseconds_since_unix_epoch () in
    (* Readback OUTSIDE the timed region (the [Context.get_values] trap, docs/agent-notes.md):
       the cc scheduler is synchronous, so the run loop needs no readback to complete. *)
    let values = Context.get_values ctx mc.Tensor.value in
    (* Element [1][1] of the n*n result — an interior cell, away from the corners — except at
       n = 1, where the whole output is one element. One interior cell cannot see a remainder
       region, so the whole output is checksummed too: every correct variant prints the identical
       value, a tail-mishandling one does not.

       Which remainder, precisely — the obvious answer is the wrong one. A [Sched.split] whose
       factor does not divide its extent is NOT the case covered here: [unschedulable] rejects
       exactly those factors, so at a non-dividing [bm]/[bk] the packed variants never run and the
       naive line has nothing to be compared against. What IS covered is the register-tiled
       micro-kernel's own edge peel, which needs no split remainder: the [Tile_mma] covers full
       blocks only, and n is free of any width constraint because j is never split. At n = 77 with
       bm = 7, bk = 11 the emitted kernel carries two scalar peel loops — a column peel over
       j in [76, 77) and a row peel over i in [4, 7) — that the same kernel at n = 76 does not
       have, and all three variants agree on the checksum across it. So the tail this check guards
       is the micro-kernel peel and the packing [Stage] edges, not the block remainder.
       Position-weighted for a different reason than schedule_bench's: with the per-axis residues
       above there is no divisibility class on which a plain sum vanishes (unlike the flattened
       form, whose collapse also zeroed it). What an unweighted sum cannot see is a PERMUTATION —
       a tail written with the right values at the wrong offsets leaves the multiset intact, and
       the multiset is all a plain sum reads — and a misplaced tail is exactly what a non-dividing
       [bm]/[bk] risks. The weight is capped so that products of these exact-in-binary operands
       stay exact in the double accumulator.

       Cross-variant equality is therefore EXACT in an f32 run, at every extent: the products are
       multiples of 1/8 bounded by 0.75n, so the whole reduction is exact in the f32 accumulator
       and independent of the order the variants sum in. It is NOT unconditional in a narrow
       storage run, and the difference is a property of the bench rather than of this check. The
       naive variant accumulates in the storage precision while the packed ones accumulate through
       a wider register tile, so once a partial sum outgrows the storage mantissa (bf16 carries 8
       significant bits, i.e. multiples of 1/8 up to 32) the two round at different points and
       legitimately disagree. It predates this checksum — the pre-existing operands already split
       the legs at bf16 n = 320 — which is why the run announces the non-comparability outright
       when storage is narrower than compute, rather than leaving a reader to mistake it for a bad
       transform. The near-mean-zero mb keeps partial sums random-walking (max 31 at n = 512),
       which buys exactness to roughly n = 512 in bf16 rather than n = 320, but buys it only up to
       a size: an f32 run is the cross-variant oracle, and in a narrow run only the packed
       variants are comparable with each other. Both checks are outside the timed region. *)
    let checksum =
      Array.foldi values ~init:0.0 ~f:(fun i acc v -> acc +. (v *. Float.of_int (1 + (i % 251))))
    in
    let spot = Int.min (n + 1) (Array.length values - 1) in
    let secs = Float.of_int63 Int63.(stop - start) /. 1e9 /. Float.of_int repeats in
    p "%-12s %8.3f ms  %8.2f GFLOP/s  (spot check [%d] %.2f, chk %.10g)%s\n" variant (secs *. 1e3)
      (flops /. secs /. 1e9) spot values.(spot) checksum
      (match renderings with
      | [] -> ""
      | rs ->
          let counted =
            List.map (List.dedup_and_sort rs ~compare:Ir.C_syntax.compare_mma_rendering)
              ~f:(fun r ->
                let k = List.count rs ~f:(Ir.C_syntax.equal_mma_rendering r) in
                Printf.sprintf "%s x%d"
                  (Sexp.to_string (Ir.C_syntax.sexp_of_mma_rendering r))
                  k)
          in
          "  [" ^ String.concat ~sep:", " counted ^ "]");
    (secs, renderings)
  in
  let ctx0 = Context.auto () in
  let limits = Context.hardware_limits ctx0 in
  let cprec =
    Numerics.cpu_compute_prec ~native_fp16_arithmetic:limits.Ir.Backend_intf.native_fp16_arithmetic
      prec
  in
  let tile_prec = if Ir.Ops.equal_prec cprec prec then None else Some cprec in
  p "GEBP n=%d, %d repeats, blocking bm=%d bk=%d, storage %s, compute %s, packed panels %s\n" n
    repeats bm bk (Ir.Ops.prec_string prec) (Ir.Ops.prec_string cprec)
    (Option.value_map tile_prec ~default:"(storage)" ~f:Ir.Ops.prec_string);
  (* Say it at runtime rather than only in a comment: with storage narrower than compute the naive
     variant reduces in %s while the packed ones reduce through the wider register tile, so a
     checksum difference between them is expected arithmetic, not evidence of a bad transform. A
     reader comparing the column has to be told which comparisons are meaningful. *)
  if Option.is_some tile_prec then
    p
      "note: naive accumulates in %s (storage), packed variants in %s (register tile) — their\n\
      \      checksums are NOT comparable; the packed variants remain comparable with each other.\n\
      \      Use an f32 run as the cross-variant oracle.\n"
      (Ir.Ops.prec_string prec) (Ir.Ops.prec_string cprec);
  let t_naive, _ = bench ~variant:"naive" ~schedule:None () in
  match unschedulable with
  | _ :: _ ->
      p "skipping the packed variants — this n is not schedulable with this blocking:\n";
      List.iter unschedulable ~f:(fun reason -> p "  %s\n" reason);
      if n >= 2 then p "pass a compatible blocking, e.g. --bm=<f> --bk=<f> with f dividing %d.\n" n
  | [] ->
      let t_pack, r_pack =
        bench ~variant:"packmma" ~schedule:(Some (packed_schedule ~grid:false ~tile_prec)) ()
      in
      let t_par, r_par =
        bench ~variant:"packmma_par" ~schedule:(Some (packed_schedule ~grid:true ~tile_prec)) ()
      in
      p "speedups vs naive: packmma %.1fx, packmma_par %.1fx\n" (t_naive /. t_pack)
        (t_naive /. t_par);
      (* Count the fallback rather than "anything that is not [Mma_register_tiled]": on the C
         backends this bench targets the two are the same set, but the latter also indicts
         [Mma_intrinsics], so it would false-warn the moment an arm runs on a GPU. *)
      let declined =
        List.count (r_pack @ r_par)
          ~f:(Ir.C_syntax.equal_mma_rendering Ir.C_syntax.Mma_scalar_fallback)
      in
      if declined > 0 then
        p
          "WARNING: %d of %d Tile_mma statements rendered the scalar fallback (see the census\n\
           above) — these are NOT register-tiled timings. Re-run with\n\
           --ocannl_schedule_log_declines=true for the per-rule reason.\n"
          declined
          (List.length (r_pack @ r_par))
