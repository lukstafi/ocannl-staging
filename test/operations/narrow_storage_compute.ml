(* 16-bit storage with f32 compute on the CPU backends (gh-ocannl-517): a narrow-float tensor node
   stays narrow in memory, but the arithmetic over it runs in f32 -- one widening per load, one
   narrowing per store, instead of a widen/op/narrow round-trip per operator.

   Three things are checked, in the order they can fail:

   1. Accuracy. An elementwise chain whose intermediates are virtual is run at bf16 storage against
      an all-f32 reference, once under the policy and once with [narrow_compute_f32 = false] (the
      per-operator rounding that predates this issue). Only the second rounds the intermediates, so
      its error is the larger one -- the assertion is the comparison, not a tolerance.

   2. Bitwise parity of the vectorized rendering against its serial twin, at bf16 and at half. The
      convert-on-load/store bridge and the scalar path must agree exactly: bf16's vector narrowing
      reimplements [single_to_bfloat16]'s round-to-nearest-even with vector arithmetic, and half's
      goes through [__builtin_convertvector] where the scalar path casts to [_Float16]. Anything
      approximate here would be a real defect, so the comparison is [=], not [approx].

   3. Structure of the generated source: a bf16 [Vectorized] loop must actually vectorize (it was
      gated to f32/f64 before this issue), and its arithmetic must be free of the per-operator
      [single_to_bfloat16] wrapping -- the narrowing appears once, on the way out.

   GPU backends are deliberately unaffected (they have native 16-bit types and arithmetic), so the
   structural checks are CPU-only; the parity checks hold everywhere. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module LL = Ir.Low_level
module Sched = Ir.Schedule
module Asgns = Ir.Assignments
module Tn = Ir.Tnode

let () = Utils.settings.output_debug_files_in_build_directory <- true
let p name b = Stdio.printf "%s: %b\n" name b
let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")
let on_cpu = String.is_substring backend_name ~substring:"cc"

let read_generated base_name =
  let ext =
    if String.is_substring backend_name ~substring:"metal" then ".metal"
    else if String.is_substring backend_name ~substring:"cuda" then ".cu"
    else if String.is_substring backend_name ~substring:"hip" then ".hip"
    else ".c"
  in
  let path = Utils.build_file (base_name ^ ext) in
  if Stdlib.Sys.file_exists path then Some (Stdio.In_channel.read_all path) else None

let named name (comp : Asgns.comp) : Asgns.comp =
  { comp with asgns = Asgns.Block_comment (name, comp.asgns) }

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

let n = 517
let av = Array.init n ~f:(fun i -> 0.125 +. (Float.of_int (i % 23) *. 0.3125))
let bv = Array.init n ~f:(fun i -> 0.5 +. (Float.of_int (i % 17) *. 0.1875))

(* An elementwise chain: every intermediate is virtual, so it lives in a register and is exactly
   what the storage/compute split is about. *)
let chain a b =
  let%op y = ((a *. b) + a) *. ((a + b) *. b) in
  y

let run ~name ~transform ~prec ~label () =
  (* The leaves are minted at [prec] rather than re-tagged: [ndarray] settles a leaf's precision as
     [Specified], which [update_prec] then refuses to change. *)
  let leaf l vals =
    NTDSL.init ~l ~prec ~o:[ n ] ~f:(function [| i |] -> vals.(i) | _ -> assert false) ()
  in
  let a = leaf (label ^ "a") av and b = leaf (label ^ "b") bv in
  let y = chain a b in
  Tn.update_prec y.Tensor.value prec;
  let comp = named name (Train.forward y) in
  let ctx = Context.auto () in
  let ctx, routine = Context.compile ~lowered_transform:transform ctx comp Ir.Indexing.Empty in
  let ctx = Context.run ctx routine in
  Context.get_values ctx y.Tensor.value

let serial _opt = _opt

let vectorize (opt : LL.optimized) =
  let j = Option.value_exn ~here:[%here] (innermost_loop opt.LL.llc) in
  Sched.apply [ Sched.Retype { axis = j; ty = LL.Vectorized } ] opt

let max_err x y =
  Array.foldi x ~init:0. ~f:(fun i acc v -> Float.max acc (Float.abs (v -. y.(i))))

let () =
  Tensor.unsafe_reinitialize ();
  let base = Ir.Numerics.get () in

  (* --- 1. Accuracy: bf16 storage, f32 compute vs. per-operator rounding. --- *)
  let reference = run ~name:"nsc_f32" ~transform:serial ~prec:Ir.Ops.single ~label:"f32_" () in
  Ir.Numerics.set_policy { base with narrow_compute_f32 = true };
  let wide = run ~name:"nsc_bf16_wide" ~transform:serial ~prec:Ir.Ops.bfloat16 ~label:"wide_" () in
  Ir.Numerics.set_policy { base with narrow_compute_f32 = false };
  let per_op =
    run ~name:"nsc_bf16_perop" ~transform:serial ~prec:Ir.Ops.bfloat16 ~label:"perop_" ()
  in
  Ir.Numerics.set_policy { base with narrow_compute_f32 = true };
  let err_wide = max_err reference wide and err_per_op = max_err reference per_op in
  p "f32 compute over bf16 storage beats per-operator rounding" Float.(err_wide < err_per_op);
  (* The wide leg's only rounding is the final store, so its relative error is bounded by bf16's
     half-ulp (2^-9); the per-operator leg compounds five of them. *)
  let rel arr = max_err reference arr /. Array.fold reference ~init:0. ~f:(fun m v ->
      Float.max m (Float.abs v)) in
  p "wide-compute relative error within one bf16 ulp" Float.(rel wide < 0.004);
  p "per-operator relative error exceeds it" Float.(rel per_op > 0.004);

  (* --- 2. Bitwise parity of the vectorized rendering against the serial twin. --- *)
  List.iter
    [ ("bf16", Ir.Ops.bfloat16); ("half", Ir.Ops.half) ]
    ~f:(fun (name, prec) ->
      let twin = run ~name:("nsc_twin_" ^ name) ~transform:serial ~prec ~label:("t" ^ name) () in
      let vec = run ~name:("nsc_vec_" ^ name) ~transform:vectorize ~prec ~label:("v" ^ name) () in
      p (name ^ " vectorized rendering is bitwise identical to the serial twin")
        (Array.for_all2_exn vec twin ~f:Float.equal));

  (* --- 3. Structure of the bf16 vectorized source. --- *)
  match read_generated "nsc_vec_bf16" with
  | None -> p "bf16 loop vectorizes with a converting load/store" (not on_cpu)
  | Some src ->
      let has s = String.is_substring src ~substring:s in
      if on_cpu then begin
        p "bf16 loop vectorizes with a converting load/store"
          (has "vector_size" && has "OCANNL_VEC_WIDEN_BFLOAT16" && has "OCANNL_VEC_NARROW_BFLOAT16");
        (* [narrow(op(...))] immediately re-widened is the signature of per-operator rounding; the
           seam makes it unspellable, in the vector body and in the serial remainder alike. *)
        p "no operator narrows only to be widened again"
          (not (has "bfloat16_to_single(single_to_bfloat16("));
        (* f32 arithmetic reached a bf16 kernel: the fused multiply-add is the f32 one. *)
        p "arithmetic is f32" (has "fmaf(" || has "__builtin_elementwise_fma")
      end
      else p "bf16 loop vectorizes with a converting load/store" true
