(* Explicit SIMD rendering of [Vectorized] loops (gh-ocannl-164 follow-up,
   docs/proposals/watch-ocannl-README-md-347818d3.md): executed parity of vector-rendered kernels
   against their serial twins, plus structural checks on the generated source.

   Covered here: - A plain [Retype ~ty:Vectorized] of the innermost loop of an elementwise mul-add
   kernel, with an extent that is not a lane multiple (517), so the serial remainder loop executes.
   - [Split] then [Retype] of the inner loop: the store/load index vectors carry an [Affine] last
   component (factor * outer + inner), exercising the coefficient-1 contiguity rule.

   On GPU backends (gh-ocannl-463) eligible loops render as 128-bit packed loads/stores
   ([reinterpret_cast] of the aligned pack structs, llm.c's Packed128) instead of vector extensions.
   The 517-column case is ineligible there — a 517-wide last dimension breaks the lane-alignment
   guarantee — and renders serially, while the split 512-column case is eligible (the [Affine]
   coefficient 8 and last dimension 512 are lane multiples). Every printed boolean holds on every
   backend. *)

open Base
open Ocannl
open Ocannl.Operation.DSL_modules
module LL = Ir.Low_level
module Sched = Ir.Schedule
module Asgns = Ir.Assignments

let () = Utils.settings.output_debug_files_in_build_directory <- true
let p name b = Stdio.printf "%s: %b\n" name b
let approx a b = Float.(abs (a - b) < 1e-4)
let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc")
let on_cpu = String.is_substring backend_name ~substring:"cc"

let read_generated base_name =
  let ext = if String.is_substring backend_name ~substring:"metal" then ".metal" else ".c" in
  let ext = if String.is_substring backend_name ~substring:"cuda" then ".cu" else ext in
  let ext = if String.is_substring backend_name ~substring:"hip" then ".hip" else ext in
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

let () =
  let rows = 6 and cols = 517 in
  let n = rows * cols in
  let av = Array.init n ~f:(fun i -> Float.of_int (i % 19) *. 0.5) in
  let bv = Array.init n ~f:(fun i -> Float.of_int (i % 23) -. 11.) in
  let a = TDSL.ndarray av ~label:[ "a" ] ~output_dims:[ rows; cols ] () in
  let b = TDSL.ndarray bv ~label:[ "b" ] ~output_dims:[ rows; cols ] () in
  let run ~name ~transform t =
    let comp = named name (Train.forward t) in
    let ctx = Context.auto () in
    let ctx, routine = Context.compile ~lowered_transform:transform ctx comp Ir.Indexing.Empty in
    let ctx = Context.run ctx routine in
    Context.get_values ctx t.Tensor.value
  in

  (* --- Retype the innermost loop (extent 517: 517 = lanes * k + remainder). --- *)
  let%op c0 = (a *. b) + a in
  let twin = run ~name:"simd_twin" ~transform:(fun opt -> opt) c0 in
  let%op c1 = (a *. b) + a in
  let got =
    run ~name:"simd_retype"
      ~transform:(fun opt ->
        let j = Option.value_exn ~here:[%here] (innermost_loop opt.LL.llc) in
        Sched.apply [ Sched.Retype { axis = j; ty = LL.Vectorized } ] opt)
      c1
  in
  p "vectorized mul-add values match the serial twin" (Array.for_all2_exn got twin ~f:approx);
  (match read_generated "simd_retype" with
  | None -> p "vectorized rendering with serial remainder" false
  | Some src ->
      let has s = String.is_substring src ~substring:s in
      let ok =
        if on_cpu then has "vector_size" && has "__builtin_memcpy" && not (has "#pragma")
        else
          (* 517 is not a lane multiple: rows other than the first start at unaligned offsets, so
             the packed rendering must decline and fall back to a plain serial loop. *)
          (not (has "vector_size")) && not (has "reinterpret_cast")
      in
      p "vectorized rendering with serial remainder" ok);

  (* --- Split by 8 then retype the inner loop: Affine last components (8*j_o + j_i). --- *)
  let cols2 = 512 in
  let n2 = rows * cols2 in
  let ev = Array.init n2 ~f:(fun i -> Float.of_int (i % 13) *. 0.25) in
  let fv = Array.init n2 ~f:(fun i -> Float.of_int (i % 11) +. 1.) in
  let e = TDSL.ndarray ev ~label:[ "e" ] ~output_dims:[ rows; cols2 ] () in
  let f = TDSL.ndarray fv ~label:[ "f" ] ~output_dims:[ rows; cols2 ] () in
  let%op d0 = e /. f in
  let twin2 = run ~name:"simd_split_twin" ~transform:(fun opt -> opt) d0 in
  let%op d1 = e /. f in
  let got2 =
    run ~name:"simd_split"
      ~transform:(fun opt ->
        let j = Option.value_exn ~here:[%here] (innermost_loop opt.LL.llc) in
        let op, _, inner = Sched.split ~axis:j ~factor:8 ~outer:LL.Serial ~inner:LL.Serial in
        Sched.apply [ op; Sched.Retype { axis = inner; ty = LL.Vectorized } ] opt)
      d1
  in
  p "split+vectorized values match the serial twin" (Array.for_all2_exn got2 twin2 ~f:approx);
  match read_generated "simd_split" with
  | None -> p "affine-indexed vector accesses rendered" false
  | Some src ->
      let has s = String.is_substring src ~substring:s in
      p "affine-indexed vector accesses rendered"
        (if on_cpu then has "vector_size" && not (has "#pragma")
         else
           (* Eligible on GPU: dims and the Affine coefficient are lane multiples, so the packed
              rendering emits reinterpret_cast float4_t loads/stores (gh-ocannl-463). *)
           has "reinterpret_cast" && has "float4_t" && not (has "vector_size"))
