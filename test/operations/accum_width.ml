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

let () =
  if not on_cpu then (
    skipped claim_parity;
    skipped claim_shape;
    skipped claim_unroll_annot;
    skipped claim_unroll_mat;
    skipped claim_off_value;
    skipped claim_off_shape)
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
