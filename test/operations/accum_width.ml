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

let claim_2ax_both_mat =
  "BOTH reduction axes materialized sequentially keep the whole-nest accumulator (equals the \
   serial result)"

let claim_2ax_pad_mat =
  "Pad-guarded materialized unroll equals the serial result (the guard peels into the scope)"

let claim_partition =
  "Partition segments share one accumulator scope (equals the unsplit serial result)"

let claim_partition_compose =
  "a partition segment stays addressable by later schedule ops (unrolling a segment equals the \
   serial result)"

let claim_mat_then_inner =
  "inner loops stay reachable after an outer materializing unroll (annotating them equals the \
   serial result)"

let claim_vec_nested =
  "a vectorized inner reduction axis folds into the whole-nest wide accumulator (equals the \
   serial result)"

let claim_simd_tail =
  "the SIMD reduction folds its non-divisible tail into the wide total (equals the serial result)"

let claim_scope_recurrence =
  "a non-reduction recurrence through a pre-existing scope keeps its per-iteration narrowing (256 \
   -0.5 -0.5 stays 256 at bf16)"

let claim_adjacent =
  "adjacent accumulations into one cell keep their per-assignment narrowing (256 +1 +1 stays 256 \
   at bf16)"

let claim_guarded =
  "index-guarded bf16 reduction accumulates wide across the guard (256 +1x5 narrows once to 260)"

let claim_wgreduce =
  "a Workgroup_reduce loop serialized on cc keeps the Serial accumulator width (equals the serial \
   result)"

(* In execution order — the GPU skip lines must match the cc run's golden line for line. *)
let all_claims =
  [
    claim_parity;
    claim_shape;
    claim_unroll_annot;
    claim_unroll_mat;
    claim_partition;
    claim_partition_compose;
    claim_2ax_ref;
    claim_2ax_inner;
    claim_2ax_outer;
    claim_2ax_annot;
    claim_2ax_both_mat;
    claim_2ax_pad_mat;
    claim_mat_then_inner;
    claim_vec_nested;
    claim_adjacent;
    claim_guarded;
    claim_scope_recurrence;
    claim_wgreduce;
    claim_simd_tail;
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
    let matmul_leg ~claim ~name ~sched =
      let ma_u = NTDSL.init ~l:(name ^ "_a") ~prec:Ir.Ops.bfloat16 ~i:[ n ] ~o:[ n ] ~f:fa () in
      let mb_u = NTDSL.init ~l:(name ^ "_b") ~prec:Ir.Ops.bfloat16 ~i:[ n ] ~o:[ n ] ~f:fb () in
      let%op mc_u = ma_u * mb_u in
      Tn.update_prec mc_u.Tensor.value Ir.Ops.bfloat16;
      let got_u = run ~name ~schedule:sched mc_u in
      p claim (Array.for_all2_exn got_u got ~f:Float.equal)
    in
    matmul_leg ~claim:claim_unroll_annot ~name:"aw_bf16_unroll_annot"
      ~sched:(unroll_k ~materialize:false);
    matmul_leg ~claim:claim_unroll_mat ~name:"aw_bf16_unroll_mat"
      ~sched:(unroll_k ~materialize:true);
    (* Partition is an index-set specialization of one reduction: its segment seams must not
       become narrowing points, so one accumulator scope spans all the segments (unlike the naive
       reading where each sibling segment loop would widen separately and narrow at every
       breakpoint — visibly on these drifting sums). *)
    matmul_leg ~claim:claim_partition ~name:"aw_bf16_partition" ~sched:(fun opt ->
        let k =
          match List.find_exn (nest_paths opt.LL.llc) ~f:(fun p -> List.length p = 3) with
          | [ _; _; k ] -> k
          | _ -> assert false
        in
        let pt, _segment_syms = Sched.partition ~axis:k ~breakpoints:[ 16; 40 ] in
        [ pt ]);
    (* The segment symbols Schedule.partition returns must stay usable AFTER the accumulation
       mint wrapped the segment loops in the scope: rewrite_loop descends into Local_scope
       bodies since gh-ocannl-639. *)
    matmul_leg ~claim:claim_partition_compose ~name:"aw_bf16_partition_compose" ~sched:(fun opt ->
        let k =
          match List.find_exn (nest_paths opt.LL.llc) ~f:(fun p -> List.length p = 3) with
          | [ _; _; k ] -> k
          | _ -> assert false
        in
        let pt, segment_syms = Sched.partition ~axis:k ~breakpoints:[ 16; 40 ] in
        [ pt; Sched.Unroll { axis = List.hd_exn segment_syms; materialize = false } ]);
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
    (* Sequential materialization of both axes: the second unroll must recognize the scope form
       the first one minted and reuse its accumulator — the outer loop is gone afterwards, so
       codegen's scope-base hoist could not recover a per-outer-iteration narrowing. *)
    leg2 ~claim:claim_2ax_both_mat ~name:"aw2_unroll_both_mat" ~sched:(fun ~r ~s ->
        [
          Sched.Unroll { axis = s; materialize = true };
          Sched.Unroll { axis = r; materialize = true };
        ]);
    (* Pad puts an [If (s < 6)] index guard on the leaf, then the materializing unroll must peel
       it into the scope: without that, the unroll would emit guarded per-copy Sets with no loop
       left, and every copy would narrow through storage. *)
    leg2 ~claim:claim_2ax_pad_mat ~name:"aw2_unroll_pad_mat" ~sched:(fun ~r:_ ~s ->
        [
          Sched.Pad { axis = s; to_multiple_of = 4 };
          Sched.Unroll { axis = s; materialize = true };
        ]);
    (* After an outer materializing unroll, the copied inner s loops live inside the scope and
       keep their symbol: a later op targeting s must reach all of them. *)
    leg2 ~claim:claim_mat_then_inner ~name:"aw2_mat_then_inner" ~sched:(fun ~r ~s ->
        [
          Sched.Unroll { axis = r; materialize = true };
          Sched.Unroll { axis = s; materialize = false };
        ]);
    (* === a VECTORIZED inner reduction axis === *)
    (* The nest peel rides through the Vectorized level, and the SIMD reduction rendering folds
       its chains into the scope LOCAL (no storage round-trip): the whole nest keeps one wide
       accumulator. Inner extent 32 clears the SIMD profitability gate; the reduction is exact in
       f32, so the vector reassociation is harmless and the comparison is bitwise. The structural
       conjunct asserts the vectorized rendering actually fired inside the scope. *)
    let nvr, nvs = (4, 32) in
    let fxv idcs =
      Float.of_int ((((idcs.(0) * nvr * nvs) + (idcs.(1) * nvs) + idcs.(2)) % 13) + 20) *. 0.015625
    in
    let run2v ~name ?schedule () =
      let xv = NTDSL.init ~l:(name ^ "_x") ~prec:Ir.Ops.bfloat16 ~o:[ ni; nvr; nvs ] ~f:fxv () in
      let%op outv = xv ++ "irs => i" in
      Tn.update_prec outv.Tensor.value Ir.Ops.bfloat16;
      run ~name ?schedule outv
    in
    let got_vn = run2v ~name:"aw_vecnest_serial" () in
    let got_vnv =
      run2v ~name:"aw_vecnest_vec"
        ~schedule:
          (two_axis_sched ~f:(fun ~r:_ ~s -> [ Sched.Retype { axis = s; ty = LL.Vectorized } ]))
        ()
    in
    let vecn_fired =
      match read_generated "aw_vecnest_vec" with
      | Some src -> String.is_substring src ~substring:"Vectorized reduction rendering"
      | None -> false
    in
    p claim_vec_nested (vecn_fired && Array.for_all2_exn got_vnv got_vn ~f:Float.equal);
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
    (* === a pre-existing scope carrying a NON-reduction recurrence must not be hoisted === *)
    (* The scope-base arm of the peel is licensed by the reduction reading alone: [local :=
       local - 0.5] narrows per enclosing iteration by the source's own semantics (256 - 0.5
       rounds back to 256 at bf16, twice over), and hoisting the enclosing loop into the scope
       would hold 255 instead. [valid_scope_updates] rejects the base. *)
    let gacc2 = node ~dims:[| 1 |] "aw_rec" in
    Ll_test.materialize gacc2;
    let ri = Ll_test.sym () in
    let rid = LL.get_scope gacc2 in
    let rec_scope_body =
      LL.Seq
        ( LL.Set_local (rid, LL.Get (gacc2, [| Ll_test.fixed 0 |])),
          LL.Set_local
            (rid, LL.Binop (Ir.Ops.Sub, (LL.Get_local rid, bf16), (LL.Constant 0.5, bf16))) )
    in
    let rec_llc =
      Ll_test.loop_n ri 2
        (LL.Set
           {
             tn = gacc2;
             idcs = [| Ll_test.fixed 0 |];
             llsc =
               LL.Local_scope
                 { id = rid; body = rec_scope_body; orig_indices = [| Ll_test.fixed 0 |] };
             debug = "";
           })
    in
    let ro = Ll_test.optimize ~materialized:[ gacc2 ] ~name:"aw_recurrence" rec_llc in
    let rvals =
      Ll_test.execute ~name:"aw_recurrence" ro ~seed:[ (gacc2, [| 256.0 |]) ] ~read:[ gacc2 ]
    in
    p claim_scope_recurrence (Float.equal (List.hd_exn rvals).(0) 256.0);
    (* === Workgroup_reduce serialized on cc: retyping the reduction axis to a hardware kind the
       backend cannot bind must not change the accumulator width relative to the Serial spelling.
       16 terms push the running sums past 4, where bf16 can no longer represent the 1/64
       increments, so a per-step regression in the serialized fallback would diverge. *)
    (* An explicit [=+] into a pre-zeroed materialized accumulator: an einsum-lowered sum's
       whole-node init nest would fail [validate_parallel] once the reduction axis is retyped to a
       hardware kind (whole-node zeroing is not distributed), and the init is not what these legs
       are about. *)
    let run_sum ~cols ~name ?schedule () =
      let fv idcs = Float.of_int ((((idcs.(0) * cols) + idcs.(1)) % 13) + 20) *. 0.015625 in
      let xw = NTDSL.init ~l:(name ^ "_x") ~prec:Ir.Ops.bfloat16 ~o:[ ni; cols ] ~f:fv () in
      let outw =
        NTDSL.init ~l:(name ^ "_out") ~prec:Ir.Ops.bfloat16 ~o:[ ni ] ~f:(fun _ -> 0.0) ()
      in
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
    let retype_reduction ty opt =
      match List.find_exn (nest_paths opt.LL.llc) ~f:(fun p -> List.length p = 2) with
      | [ _; s ] -> [ Sched.Retype { axis = s; ty } ]
      | _ -> assert false
    in
    let got_w = run_sum ~cols:16 ~name:"aw_wgr_serial" () in
    let got_wg =
      run_sum ~cols:16 ~name:"aw_wgr_hw"
        ~schedule:(retype_reduction LL.Workgroup_reduce)
        ()
    in
    p claim_wgreduce (Array.for_all2_exn got_wg got_w ~f:Float.equal);
    (* === the SIMD reduction's scalar remainder === *)
    (* 67 is no multiple of any chains*lanes step, so the vector partial must fold into the wide
       total together with the tail contributions and narrow once — the pre-fix rendering stored
       the partial to bf16 mid-way (at running sums ~25, where 1/64 increments are lost) and then
       narrowed per tail step. The structural conjunct keeps the leg honest: if the vectorized
       rendering declines, serial-equals-serial would be a false green. *)
    let got_v = run_sum ~cols:67 ~name:"aw_vec_serial" () in
    let got_vt =
      run_sum ~cols:67 ~name:"aw_vec_tail" ~schedule:(retype_reduction LL.Vectorized) ()
    in
    let vec_fired =
      match read_generated "aw_vec_tail" with
      | Some src -> String.is_substring src ~substring:"Vectorized reduction rendering"
      | None -> false
    in
    p claim_simd_tail (vec_fired && Array.for_all2_exn got_vt got_v ~f:Float.equal);
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
