(* Accumulator-width policy (gh-ocannl-639): a reduction accumulator over narrow-float storage
   resides at compute precision across the whole reduction nest and narrows once at the store —
   for EVERY rendering, the plain serial fallback included — so the effective accumulation width
   is set by the numerics policy ([Numerics.cpu_compute_prec]), never by which schedule happened
   to place the accumulator in a register. Before gh-ocannl-639 the unscheduled lowering
   round-tripped the accumulator through storage on every reduction step
   ([mc[..] = single_to_bfloat16(fmaf(..., bfloat16_to_single(mc[..])))]), so a bf16 result
   depended on whether a register-tiling schedule ran.

   Inputs are exact in bf16 while their PARTIAL SUMS are not: products are multiples of 15/128
   and their mean is nonzero (~0.23), so the running sum drifts past the range where bf16's 8
   significand bits can hold such multiples (a zero-mean b operand random-walks below 4 and every
   partial sum stays bf16-exact — the first draft of this test proved that the hard way), and
   per-k-step narrowing visibly diverges from whole-k f32 residency — the policy-off leg is the
   negative control proving the inputs discriminate. The whole reduction stays exact in f32
   (multiples of 1/128, magnitude far below 2^16), so the f64 host-side reference reproduces the
   kernel's fmaf chain exactly, and narrowing it once through the library's own bf16 conversion
   gives the normative result the widened kernel must match bitwise.

   The claims are CPU-emission claims ([comp_prec] is the identity on the GPU backends' native
   narrow arithmetic, so their serial legs keep storage-precision accumulation); on other
   backends they are skipped. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module Tn = Ir.Tnode
module LL = Ir.Low_level
module Sched = Ir.Schedule
module Asgns = Ir.Assignments
module Numerics = Ir.Numerics

let () = Utils.settings.output_debug_files_in_build_directory <- true
let p = Verdict.p
let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")
let skipped = Verdict.skipped ~backend:backend_name
let on_cpu = String.is_substring backend_name ~substring:"cc"

let read_generated base_name =
  let path = Utils.build_file (base_name ^ ".c") in
  if Stdlib.Sys.file_exists path then Some (Stdio.In_channel.read_all path) else None

let named name (comp : Asgns.comp) : Asgns.comp =
  { comp with asgns = Asgns.Block_comment (name, comp.asgns) }

(* The single-child chain of loops from the top of each top-level nest (tile_mma_narrow's
   helper): used to address the reduction axis for the unroll legs. *)
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

(* [Sched.Unroll] over the k axis of the matmul's i/j/k nest, in either representation. *)
let unroll_k ~materialize (opt : LL.optimized) : Sched.schedule =
  let k =
    match List.find_exn (nest_paths opt.LL.llc) ~f:(fun p -> List.length p = 3) with
    | [ _; _; k ] -> k
    | _ -> assert false
  in
  [ Sched.Unroll { axis = k; materialize } ]

(* The i/r/s nest of the two-axis reduction: pick which reduction axis to transform. *)
let two_axis_sched ~f (opt : LL.optimized) : Sched.schedule =
  match List.find_exn (nest_paths opt.LL.llc) ~f:(fun p -> List.length p = 3) with
  | [ _; r; s ] -> f ~r ~s
  | _ -> assert false

let run ~name ?schedule (out : Tensor.t) =
  let transform opt =
    match schedule with None -> opt | Some sched -> Sched.apply (sched opt) opt
  in
  let ctx = Context.auto () in
  let ctx, routine =
    Context.compile ~lowered_transform:transform ctx (named name (Train.forward out))
      Ir.Indexing.Empty
  in
  let ctx = Context.run ctx routine in
  Context.get_values ctx out.Tensor.value

let n = 64
let fa idcs = Float.of_int ((((idcs.(0) * n) + idcs.(1)) % 3) + 1) *. 0.375
let fb idcs = (Float.of_int (((idcs.(0) * n) + idcs.(1)) % 5) -. 1.5) *. 0.625

let claim_parity = "bf16 naive matmul equals the once-narrowed wide-accumulation reference"
let claim_shape = "the emitted serial k-loop narrows the accumulator once per cell, not per step"

let claim_off_value =
  "narrow_compute_f32=false recovers per-operator rounding: the result differs from the widened \
   default"

let claim_off_shape = "narrow_compute_f32=false brings back the per-k-step narrowing in the k-loop"

let claim_unroll_annot =
  "Unroll-annotated bf16 reduction keeps the wide accumulator (equals the serial result)"

let claim_unroll_mat =
  "materialized-unroll bf16 reduction keeps the wide accumulator (equals the serial result)"

let claim_2ax_ref = "two-axis bf16 reduction equals the once-narrowed wide-accumulation reference"

let claim_2ax_inner =
  "materialized-unrolled INNER reduction axis equals the serial result (the scope hoists through \
   the outer reduction loop)"

let claim_2ax_outer = "materialized-unrolled OUTER reduction axis equals the serial result"
let claim_2ax_annot = "Unroll-annotated both reduction axes equals the serial result"

let claim_adjacent =
  "adjacent accumulations into one cell keep their per-assignment narrowing (256 +1 +1 stays 256 \
   at bf16)"

let claim_guarded =
  "index-guarded bf16 reduction accumulates wide across the guard (256 +1x5 narrows once to 260)"

let claim_wgreduce =
  "a Workgroup_reduce loop serialized on cc keeps the Serial accumulator width (equals the serial \
   result)"

let all_claims =
  [
    claim_parity;
    claim_shape;
    claim_unroll_annot;
    claim_unroll_mat;
    claim_2ax_ref;
    claim_2ax_inner;
    claim_2ax_outer;
    claim_2ax_annot;
    claim_adjacent;
    claim_guarded;
    claim_wgreduce;
    claim_off_value;
    claim_off_shape;
  ]

let () =
  if not on_cpu then List.iter all_claims ~f:skipped
  else begin
    let ma = NTDSL.init ~l:"ma" ~prec:Ir.Ops.bfloat16 ~i:[ n ] ~o:[ n ] ~f:fa () in
    let mb = NTDSL.init ~l:"mb" ~prec:Ir.Ops.bfloat16 ~i:[ n ] ~o:[ n ] ~f:fb () in
    let%op mc = ma * mb in
    Tn.update_prec mc.Tensor.value Ir.Ops.bfloat16;
    let got = run ~name:"aw_bf16_naive" mc in
    (* The reference: the f64 dot products are exact reproductions of the kernel's f32 fmaf chain
       (every partial sum is a multiple of 1/64 well within f32's mantissa), narrowed exactly once
       per cell by minting a bf16 tensor from them and reading it back. *)
    let wide_sums =
      Array.init (n * n) ~f:(fun t ->
          let i = t / n and j = t % n in
          let acc = ref 0.0 in
          for k = 0 to n - 1 do
            acc := !acc +. (fa [| i; k |] *. fb [| k; j |])
          done;
          !acc)
    in
    let mref =
      NTDSL.init ~l:"mref" ~prec:Ir.Ops.bfloat16 ~i:[ n ] ~o:[ n ]
        ~f:(fun idcs -> wide_sums.((idcs.(0) * n) + idcs.(1)))
        ()
    in
    let want = run ~name:"aw_bf16_ref" mref in
    p claim_parity (Array.for_all2_exn got want ~f:Float.equal);
    (match read_generated "aw_bf16_naive" with
    | None -> Verdict.fail (claim_shape ^ " — generated source not found")
    | Some src ->
        let has s = String.is_substring src ~substring:s in
        p claim_shape ((not (has "single_to_bfloat16(fmaf(")) && has "fmaf("));
    (* Both unroll representations autotune proposes over small reduction loops (annotated:
       codegen repeats the body; materialized: the IR carries one copy per step and no loop)
       must keep the wide accumulator — the reduction is exact in f32 and narrows at the same
       single point, so both are bitwise equal to the serial leg; per-repetition narrowing
       would visibly diverge on these inputs (the same discrimination as the policy-off arm). *)
    let unroll_leg ~claim ~name ~materialize =
      let ma_u = NTDSL.init ~l:(name ^ "_a") ~prec:Ir.Ops.bfloat16 ~i:[ n ] ~o:[ n ] ~f:fa () in
      let mb_u = NTDSL.init ~l:(name ^ "_b") ~prec:Ir.Ops.bfloat16 ~i:[ n ] ~o:[ n ] ~f:fb () in
      let%op mc_u = ma_u * mb_u in
      Tn.update_prec mc_u.Tensor.value Ir.Ops.bfloat16;
      let got_u = run ~name ~schedule:(unroll_k ~materialize) mc_u in
      p claim (Array.for_all2_exn got_u got ~f:Float.equal)
    in
    unroll_leg ~claim:claim_unroll_annot ~name:"aw_bf16_unroll_annot" ~materialize:false;
    unroll_leg ~claim:claim_unroll_mat ~name:"aw_bf16_unroll_mat" ~materialize:true;
    (* === two-axis reduction out[i] = sum_{r,s} x[i,r,s]: unrolling EITHER reduction axis keeps
       the whole-nest accumulator. The inner-axis leg is the partial-materialization shape: the
       unroll mints a scope-form Set inside the still-serial outer reduction loop, and the
       codegen peel hoists that scope through it — without the hoist the accumulator would store
       and narrow once per outer iteration. Cell values are multiples of 1/64 in [0.3125, 0.5]
       (bf16-exact); the running sums (~14.6) are not, and the whole reduction is exact in f32. *)
    let ni, nr, ns = (4, 6, 6) in
    let fx idcs =
      Float.of_int ((((idcs.(0) * nr * ns) + (idcs.(1) * ns) + idcs.(2)) % 13) + 20) *. 0.015625
    in
    let run2 ~name ?schedule () =
      let x2 = NTDSL.init ~l:(name ^ "_x") ~prec:Ir.Ops.bfloat16 ~o:[ ni; nr; ns ] ~f:fx () in
      let%op out2 = x2 ++ "irs => i" in
      Tn.update_prec out2.Tensor.value Ir.Ops.bfloat16;
      run ~name ?schedule out2
    in
    let got2 = run2 ~name:"aw2_serial" () in
    let wide2 =
      Array.init ni ~f:(fun i ->
          let acc = ref 0.0 in
          for r = 0 to nr - 1 do
            for s = 0 to ns - 1 do
              acc := !acc +. fx [| i; r; s |]
            done
          done;
          !acc)
    in
    let mref2 =
      NTDSL.init ~l:"aw2_ref" ~prec:Ir.Ops.bfloat16 ~o:[ ni ] ~f:(fun idcs -> wide2.(idcs.(0))) ()
    in
    let want2 = run ~name:"aw2_refc" mref2 in
    p claim_2ax_ref (Array.for_all2_exn got2 want2 ~f:Float.equal);
    let leg2 ~claim ~name ~sched =
      let got_l = run2 ~name ~schedule:(two_axis_sched ~f:sched) () in
      p claim (Array.for_all2_exn got_l got2 ~f:Float.equal)
    in
    leg2 ~claim:claim_2ax_inner ~name:"aw2_unroll_inner" ~sched:(fun ~r:_ ~s ->
        [ Sched.Unroll { axis = s; materialize = true } ]);
    leg2 ~claim:claim_2ax_outer ~name:"aw2_unroll_outer" ~sched:(fun ~r ~s:_ ->
        [ Sched.Unroll { axis = r; materialize = true } ]);
    leg2 ~claim:claim_2ax_annot ~name:"aw2_unroll_annot" ~sched:(fun ~r ~s ->
        [
          Sched.Unroll { axis = r; materialize = false };
          Sched.Unroll { axis = s; materialize = false };
        ]);
    (* === adjacent accumulations: two SOURCE assignments into one cell are two stores, and each
       store narrows — they must NOT share an accumulator residency (that is the provenance
       boundary: only unrolled copies of one assignment may). 256 + 1 rounds to 256 at bf16
       twice over; a rewrite merging the pair would produce 258. *)
    let s_acc = NTDSL.init ~l:"aw_s" ~prec:Ir.Ops.bfloat16 ~o:[ 1 ] ~f:(fun _ -> 256.0) () in
    let ax1 = NTDSL.init ~l:"aw_ax1" ~prec:Ir.Ops.bfloat16 ~o:[ 1 ] ~f:(fun _ -> 1.0) () in
    let ax2 = NTDSL.init ~l:"aw_ax2" ~prec:Ir.Ops.bfloat16 ~o:[ 1 ] ~f:(fun _ -> 1.0) () in
    Train.set_materialized s_acc.Tensor.value;
    let adj_comp = Asgns.sequence [ [%cd s_acc =+ ax1]; [%cd s_acc =+ ax2] ] in
    let ctx = Context.auto () in
    let ctx, routine = Context.compile ctx (named "aw_adjacent" adj_comp) Ir.Indexing.Empty in
    let ctx = Context.run ctx routine in
    let adj = Context.get_values ctx s_acc.Tensor.value in
    p claim_adjacent (Float.equal adj.(0) 256.0);
    (* === index-guarded reduction: the gh-490 symbolic-extent guard shape [If (i < bound)] is
       transparent to the widening, so a runtime-bounded reduction keeps the compute-precision
       accumulator. Hand-built IR (the Assignments pipeline lowers clamped windows through
       interval-provable guards): 256 seeded, eight 1.0 contributions guarded to five — the wide
       accumulator reaches 261 and narrows once to 260 (round-to-even); per-step narrowing would
       absorb every +1 and leave 256. *)
    let bf16 = Ir.Ops.bfloat16 in
    let node = Ll_test.node_factory ~prec:bf16 ~first_id:9500 ~dims:[| 8 |] () in
    let gacc = node ~dims:[| 1 |] "aw_gacc" in
    let gxs = node "aw_gxs" in
    Ll_test.materialize gacc;
    Ll_test.materialize gxs;
    let gi = Ll_test.sym () in
    let iprec = Ir.Ops.index_prec () in
    let guard = LL.Binop (Ir.Ops.Cmplt, (Ll_test.embed gi, iprec), (LL.Constant 5.0, iprec)) in
    let upd =
      Ll_test.set gacc [| Ll_test.fixed 0 |]
        (LL.Binop
           ( Ir.Ops.Add,
             (Ll_test.get gacc [| Ll_test.fixed 0 |], bf16),
             (Ll_test.get gxs [| Ll_test.iter gi |], bf16) ))
    in
    let gllc = Ll_test.loop_n gi 8 (LL.If { cond = (guard, iprec); body = upd }) in
    let go = Ll_test.optimize ~materialized:[ gacc; gxs ] ~name:"aw_guarded" gllc in
    let gvals =
      Ll_test.execute ~name:"aw_guarded" go
        ~seed:[ (gacc, [| 256.0 |]); (gxs, Array.create ~len:8 1.0) ]
        ~read:[ gacc ]
    in
    p claim_guarded (Float.equal (List.hd_exn gvals).(0) 260.0);
    (* === Workgroup_reduce serialized on cc: retyping the reduction axis to a hardware kind the
       backend cannot bind must not change the accumulator width relative to the Serial spelling.
       16 terms push the running sums past 4, where bf16 can no longer represent the 1/64
       increments, so a per-step regression in the serialized fallback would diverge. *)
    let nw = 16 in
    let fw idcs = Float.of_int ((((idcs.(0) * nw) + idcs.(1)) % 13) + 20) *. 0.015625 in
    (* An explicit [=+] into a pre-zeroed materialized accumulator: an einsum-lowered sum's
       whole-node init nest would fail [validate_parallel] once the reduction axis is retyped to a
       hardware kind (whole-node zeroing is not distributed), and the init is not what this leg
       is about. *)
    let run_w ~name ?schedule () =
      let xw = NTDSL.init ~l:(name ^ "_x") ~prec:Ir.Ops.bfloat16 ~o:[ ni; nw ] ~f:fw () in
      let outw = NTDSL.init ~l:(name ^ "_out") ~prec:Ir.Ops.bfloat16 ~o:[ ni ] ~f:(fun _ -> 0.0) () in
      Train.set_materialized outw.Tensor.value;
      let comp = named name [%cd outw =+ id xw ~logic:"is => i"] in
      let transform opt =
        match schedule with None -> opt | Some sched -> Sched.apply (sched opt) opt
      in
      let ctx = Context.auto () in
      let ctx, routine = Context.compile ~lowered_transform:transform ctx comp Ir.Indexing.Empty in
      let ctx = Context.run ctx routine in
      Context.get_values ctx outw.Tensor.value
    in
    let got_w = run_w ~name:"aw_wgr_serial" () in
    let got_wg =
      run_w ~name:"aw_wgr_hw"
        ~schedule:(fun opt ->
          match List.find_exn (nest_paths opt.LL.llc) ~f:(fun p -> List.length p = 2) with
          | [ _; s ] -> [ Sched.Retype { axis = s; ty = LL.Workgroup_reduce } ]
          | _ -> assert false)
        ()
    in
    p claim_wgreduce (Array.for_all2_exn got_wg got_w ~f:Float.equal);
    (* Negative control: turning the policy off recovers the pre-gh-517/pre-gh-639 semantics —
       every operator, the accumulation update included, rounds to storage precision — which on
       these inputs must visibly differ from the widened default, proving the inputs discriminate
       the accumulator width. *)
    let saved_policy = Numerics.get () in
    Numerics.set_policy { saved_policy with narrow_compute_f32 = false };
    let ma2 = NTDSL.init ~l:"ma2" ~prec:Ir.Ops.bfloat16 ~i:[ n ] ~o:[ n ] ~f:fa () in
    let mb2 = NTDSL.init ~l:"mb2" ~prec:Ir.Ops.bfloat16 ~i:[ n ] ~o:[ n ] ~f:fb () in
    let%op mc2 = ma2 * mb2 in
    Tn.update_prec mc2.Tensor.value Ir.Ops.bfloat16;
    let got_off = run ~name:"aw_bf16_naive_off" mc2 in
    Numerics.set_policy saved_policy;
    p claim_off_value (not (Array.for_all2_exn got_off got ~f:Float.equal));
    match read_generated "aw_bf16_naive_off" with
    | None -> Verdict.fail (claim_off_shape ^ " — generated source not found")
    | Some src ->
        let has s = String.is_substring src ~substring:s in
        p claim_off_shape (has "single_to_bfloat16(fmaf(")
  end
