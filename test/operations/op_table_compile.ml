(* gh-ocannl-1731: every (operator, precision) arm of the selected backend's operator tables renders
   and compiles. Two defects found by gh-ocannl-1684 shipped because nothing compiled the arms no
   value test happened to run: float [Mod] on cc rendered C's [%], which does not compile on floats,
   and half [Mod] on CUDA/HIP was an ambiguous [fmod] call.

   The sweep goes through the backend's real pipeline, the [?prelowered] seam of
   {!Context.compile_outcome}: one hand-built routine per storage precision applies every in-domain
   operator to that precision's operands and stores each result in a cell of its own, and the
   backend renders and compiles it. Nothing runs. The code reaching the backend is the hand-built
   [Low_level.t] AS WRITTEN (the optimizer only builds the record around it), so no simplification
   can drop an arm before it renders.

   - A table arm is an operator at a COMPUTE precision, and which compute precision a storage
   precision renders at is the numerics policy's choice ({!Ir.Numerics.t}: on cc, narrow storage
   computes in f32 unless [narrow_compute_f32 = false], or f16 natively under [Fp16_narrow]). So the
   sweep takes every policy in {!Ir.Numerics.all}, reads the backend's own resolution off
   {!Context.codegen_capabilities}, and compiles each distinct (storage, compute) pair once, under
   the first policy that reaches it. Arms at a compute precision no policy reaches are unreachable
   on this backend, and are listed on stderr rather than compiled. - An arm the backend REFUSES to
   render (an [Invalid_argument] or a typed refusal raised during code generation, such as [ToPowOf]
   at an integer precision) is a decline, not a defect: the refusal is a loud, attributed error.
   What must never happen is an arm that renders and then fails the backend's compiler -- that is
   the silent class, a kernel that only a user meets. Refusals are listed on stderr; the claims
   below bound where they may occur. - A routine that fails is split in halves until each failure is
   one arm, so a defect is named by its operator and precision, and a refused arm costs no compiler
   run (refusal happens before the compiler is invoked).

   The operator domain is read off {!Ops}, not listed: every derived enumeration of [unop], [binop],
   [ternop] and [vec_unop], minus three classes no lowering can emit, each excluded by a predicate
   over the operator rather than a list. The projections [Arg1]/[Arg2] have no spelling (the
   renderer emits the selected operand alone, {!Ops.binop_conditionality}). The Threefry operators
   are defined only on packed [uint4x32] state, so at a scalar precision they are out of domain, and
   at [uint4x32] they are the only arithmetic in domain ({!Ops.prec_family}: packed RNG state, not a
   scalar). The heterogeneous [uint4x32] consumers ({!Ops.is_homogeneous_prec_unop} and friends)
   take a [uint4x32] block and produce every scalar precision. Each homogeneous arithmetic arm is
   swept twice, over buffer reads and with an inlined [a + b] as its first operand ([Inlined]),
   since integer promotion types the two differently. Conversions are every ordered pair of distinct
   storage precisions the backend can store ({!Ir.Backend_intf.codegen_capabilities}'s
   [supports_f64]). *)

open Base
module LL = Ir.Low_level
module Ops = Ir.Ops
module Idx = Ir.Indexing
module Tn = Ir.Tnode
module Outcome = Ir.Schedule_outcome
open Verdict.Claims

type arm =
  | Unary of Ops.unop
  | Binary of Ops.binop
  | Ternary of Ops.ternop
  | Vector of Ops.vec_unop
  | Convert_from of Ops.prec
  | Inlined of arm
      (** A homogeneous arithmetic arm whose first operand is an inlined [a + b] rather than a
          buffer read: C-family integer promotion types that sum [int] where the read is [uchar], so
          an arm that compiles over a read can still be an ambiguous call over an expression. *)

let rec arm_name = function
  | Unary op -> "unop " ^ Sexp.to_string (Ops.sexp_of_unop op)
  | Binary op -> "binop " ^ Sexp.to_string (Ops.sexp_of_binop op)
  | Ternary op -> "ternop " ^ Sexp.to_string (Ops.sexp_of_ternop op)
  | Vector op -> "vec_unop " ^ Sexp.to_string (Ops.sexp_of_vec_unop op)
  | Convert_from q -> "convert from " ^ Ops.prec_string q
  | Inlined arm -> arm_name arm ^ " over an inlined operand"

type verdict =
  | Compiled
  | Refused of string  (** Refused while lowering or rendering: before any compiler ran. *)
  | Failed of string  (** Rendered, then the backend's compiler or loader rejected it. *)

let is_packed prec = match Ops.prec_family prec with Ops.Packed_integer -> true | _ -> false
let is_threefry = function Ops.Threefry4x32_crypto | Ops.Threefry4x32_light -> true | _ -> false

let is_projection op =
  match Ops.binop_conditionality op with Ops.Only_first | Ops.Only_second -> true | _ -> false

(* The in-domain arms at storage precision [store], over the storage precisions [precs] the backend
   can hold. *)
let arms_at ~precs store =
  let conversions =
    List.filter_map precs ~f:(fun q ->
        if Ops.equal_prec q store then None else Some (Convert_from q))
  in
  if is_packed store then
    List.filter_map Ops.all_of_binop ~f:(fun op ->
        if is_threefry op then Some (Binary op) else None)
    @ conversions
  else
    let arithmetic =
      List.map Ops.all_of_unop ~f:(fun op -> Unary op)
      @ List.filter_map Ops.all_of_binop ~f:(fun op ->
          if is_projection op || is_threefry op then None else Some (Binary op))
      @ List.map Ops.all_of_ternop ~f:(fun op -> Ternary op)
    in
    let homogeneous = function
      | Unary op -> Ops.is_homogeneous_prec_unop op
      | Binary op -> Ops.is_homogeneous_prec_binop op
      | Ternary op -> Ops.is_homogeneous_prec_ternop op
      | Vector _ | Convert_from _ | Inlined _ -> false
    in
    arithmetic
    @ List.filter_map arithmetic ~f:(fun arm ->
        if homogeneous arm then Some (Inlined arm) else None)
    @ List.map Ops.all_of_vec_unop ~f:(fun op -> Vector op)
    @ conversions

(* Node ids: this executable's range. *)
let next_id = ref 17_310_000

let node ~prec ?(dims = [| 1 |]) label =
  Int.incr next_id;
  let tn =
    Tn.create (Tn.Specified prec) ~id:!next_id ~label:[ label ]
      ~unpadded_dims:(lazy dims)
      ~padding:(lazy None)
      ()
  in
  Ll_test.materialize tn;
  tn

(* The operands and result cells of one storage precision's routine. Every operand is its own node,
   so no operator sees the same value twice and no read folds into another. *)
type operands = {
  store : Ops.prec;
  a : Tn.t;
  b : Tn.t;
  c : Tn.t;
  block : Tn.t;  (** A packed [uint4x32] block, for the heterogeneous consumers. *)
  lane : Tn.t;  (** An [int32] lane index. *)
  sources : (Ops.prec * Tn.t) list;  (** A source node per conversion. *)
  out : Tn.t;  (** One cell per scalar arm. *)
  vec_out : Tn.t;  (** The lanes of the vector arm. *)
}

let operands ~precs ~n_arms store =
  let tag = Ops.prec_string store in
  let mk label = node ~prec:store (tag ^ "_" ^ label) in
  {
    store;
    a = mk "a";
    b = mk "b";
    c = mk "c";
    block = node ~prec:Ops.uint4x32 (tag ^ "_block");
    lane = node ~prec:Ops.int32 (tag ^ "_lane");
    sources =
      List.filter_map precs ~f:(fun q ->
          if Ops.equal_prec q store then None
          else Some (q, node ~prec:q (tag ^ "_from_" ^ Ops.prec_string q)));
    out = node ~prec:store ~dims:[| n_arms |] (tag ^ "_out");
    vec_out =
      (if is_packed store then mk "vec_out_unused"
       else node ~prec:store ~dims:[| Ops.vec_unop_lanes store |] (tag ^ "_vec_out"));
  }

let at0 = [| Idx.Fixed_idx 0 |]
let read tn prec : LL.scalar_arg = (LL.Get (tn, at0), prec)

(* The statement applying [arm] (the [k]th of its routine). *)
let rec statement o k arm : LL.t =
  let s = o.store in
  let store llsc = Ll_test.set o.out [| Idx.Fixed_idx k |] llsc in
  let sum = (LL.Binop (Ops.Add, read o.a s, read o.b s), s) in
  match arm with
  | Inlined (Unary op) -> store (LL.Unop (op, sum))
  | Inlined (Binary op) -> store (LL.Binop (op, sum, read o.c s))
  | Inlined (Ternary op) -> store (LL.Ternop (op, sum, read o.b s, read o.c s))
  | Inlined arm -> statement o k arm
  | Unary op when Ops.is_homogeneous_prec_unop op -> store (LL.Unop (op, read o.a s))
  | Unary op -> store (LL.Unop (op, read o.block Ops.uint4x32))
  | Binary op when Ops.is_homogeneous_prec_binop op -> store (LL.Binop (op, read o.a s, read o.b s))
  | Binary op -> store (LL.Binop (op, read o.block Ops.uint4x32, read o.lane Ops.int32))
  | Ternary op -> store (LL.Ternop (op, read o.a s, read o.b s, read o.c s))
  | Vector vec_unop ->
      Ll_test.set_from_vec ~length:(Ops.vec_unop_lanes s) ~vec_unop o.vec_out at0
        (read o.block Ops.uint4x32)
  | Convert_from q ->
      let src = List.Assoc.find_exn o.sources ~equal:Ops.equal_prec q in
      store (LL.Get (src, at0))

let nodes o = [ o.a; o.b; o.c; o.block; o.lane; o.out; o.vec_out ] @ List.map o.sources ~f:snd
let base_ctx = Lazy.force Ll_test.base_ctx
let compiles = ref 0

(* A refusal is the renderer raising [Invalid_argument] or a [User_error] (or a typed [Unsupported]
   decline) -- the attributed errors a table or the code generator raises for a pair it does not
   render. The PHASE does not tell a refusal from a compiler rejection: cc renders inside its
   compile step, so both report [Backend_compile]. Anything else -- the backend's compiler or loader
   rejecting the rendered code ([Backend_rejected]), or an exception nothing classified -- is a
   failure, so an unrecognized outcome can only fail the claim, never pass it. *)
let is_refusal_constructor name =
  String.equal name "Invalid_argument" || String.is_suffix name ~suffix:"User_error"

let verdict_of_failure : Outcome.failure -> verdict =
  let phase_name phase = Sexp.to_string (Outcome.sexp_of_phase phase) in
  function
  | Outcome.Fatal { phase; exn; _ } ->
      let detail = phase_name phase ^ ": " ^ Exn.to_string exn in
      if is_refusal_constructor (Stdlib.Printexc.exn_slot_name exn) then Refused detail
      else Failed detail
  | Outcome.Classified { phase; cause; _ } -> (
      let detail = phase_name phase ^ ": " ^ Sexp.to_string (Outcome.sexp_of_cause cause) in
      match cause with
      | Outcome.Unsupported _ -> Refused detail
      | Outcome.Unclassified { exn_constructor; _ } when is_refusal_constructor exn_constructor ->
          Refused detail
      | _ -> Failed detail)

(* Compiles the [arms] (each with its routine-wide position) as one routine; [None] when it
   compiled. *)
let try_compile ~policy ~name o arms =
  let llc =
    List.fold_right arms ~init:LL.Noop ~f:(fun (k, arm) acc -> LL.Seq (statement o k arm, acc))
  in
  Int.incr compiles;
  let name = Printf.sprintf "%s_%d" name !compiles in
  Test_utils.with_policy
    (fun _ -> policy)
    (fun () ->
      match Ll_test.optimize ~materialized:(nodes o) ~name llc with
      | exception exn ->
          let detail = "lowering: " ^ Exn.to_string exn in
          Some
            (if is_refusal_constructor (Stdlib.Printexc.exn_slot_name exn) then Refused detail
             else Failed detail)
      | optimized -> (
          match
            Context.compile_outcome ~name ~provenance:Outcome.Candidate
              ~prelowered:{ optimized with LL.llc }
              ~lowered_transform:(fun x -> [ x ])
              base_ctx Ir.Assignments.empty_comp Idx.Empty
          with
          | Ok (_ctx, _routine) -> None
          | Error failure -> Some (verdict_of_failure failure)))

(* Routines that failed although every one of their arms compiled on its own: a failure of the
   combination (a binding limit, say), which no single arm explains. Splitting would otherwise drop
   it, and the sweep would report the arms it could not compile together as compiled. *)
let unexplained : (string * verdict) list ref = ref []

(* Splits a failing routine until each failure is one arm. *)
let rec probe ~policy ~name o arms =
  match try_compile ~policy ~name o arms with
  | None -> List.map arms ~f:(fun (_, arm) -> (arm, Compiled))
  | Some verdict -> (
      match arms with
      | [ (_, arm) ] -> [ (arm, verdict) ]
      | _ ->
          let left, right = List.split_n arms (List.length arms / 2) in
          let results = probe ~policy ~name o left @ probe ~policy ~name o right in
          if List.for_all results ~f:(fun (_, v) -> match v with Compiled -> true | _ -> false)
          then
            unexplained :=
              ( Printf.sprintf "%s over %d arms from %s" name (List.length arms)
                  (arm_name (snd (List.hd_exn arms))),
                verdict )
              :: !unexplained;
          results)

(* Whether the C reference refuses [arm] at these precisions too: the shared C spellings in {!Ops}
   raise for it, or it is [FMA] at a non-float precision, which the C-family emission seam
   ({!Ir.C_syntax}'s [pp_scalar], gh-ocannl-873) refuses for every backend before any table is
   consulted. *)
let rec c_reference_refuses ~store ~compute arm =
  let raises f = match f () with _ -> false | exception Invalid_argument _ -> true in
  match arm with
  | Unary op -> raises (fun () -> Ops.unop_c_syntax compute op)
  | Binary op -> raises (fun () -> Ops.binop_c_syntax compute op)
  | Ternary Ops.FMA -> not (Ops.is_float compute)
  | Ternary op -> raises (fun () -> Ops.ternop_c_syntax compute op)
  | Vector op -> raises (fun () -> Ops.vec_unop_c_syntax compute op)
  | Convert_from from -> raises (fun () -> Ops.c_convert_precision ~from ~to_:store)
  | Inlined arm -> c_reference_refuses ~store ~compute arm

type case = { store : Ops.prec; compute : Ops.prec; results : (arm * verdict) list }

let () =
  let caps = Context.codegen_capabilities base_ctx in
  let precs =
    List.filter Ops.storage_precs ~f:(fun p ->
        caps.supports_f64 || not (Ops.equal_prec p Ops.double))
  in
  (* Each distinct (storage, compute) pair, with the first policy reaching it. *)
  let pairs =
    List.concat_map Ir.Numerics.all ~f:(fun policy ->
        Test_utils.with_policy
          (fun _ -> policy)
          (fun () -> List.map precs ~f:(fun store -> (store, caps.compute_prec store, policy))))
    |> List.fold ~init:[] ~f:(fun seen ((s, c, _) as pair) ->
        if List.exists seen ~f:(fun (s', c', _) -> Ops.equal_prec s s' && Ops.equal_prec c c') then
          seen
        else pair :: seen)
    |> List.rev
    |> List.stable_sort ~compare:(fun (s1, _, _) (s2, _, _) -> Ops.compare_prec s1 s2)
  in
  let cases =
    List.map pairs ~f:(fun (store, compute, policy) ->
        let arms = arms_at ~precs store in
        let o = operands ~precs ~n_arms:(List.length arms) store in
        let name =
          Printf.sprintf "optbl_%s_at_%s" (Ops.prec_string store) (Ops.prec_string compute)
        in
        { store; compute; results = probe ~policy ~name o (List.mapi arms ~f:(fun k a -> (k, a))) })
  in
  let label { store; compute; _ } =
    Printf.sprintf "%s storage at %s compute" (Ops.prec_string store) (Ops.prec_string compute)
  in
  let all_results = List.concat_map cases ~f:(fun c -> List.map c.results ~f:(fun r -> (c, r))) in
  Stdio.eprintf "op_table_compile on %s (not part of the golden): %d compile attempts\n"
    (Context.backend_name base_ctx) !compiles;
  List.iter cases ~f:(fun case ->
      let count f = List.count case.results ~f:(fun (_, v) -> f v) in
      Stdio.eprintf "  %s: %d compiled, %d refused, %d failed\n" (label case)
        (count (function Compiled -> true | _ -> false))
        (count (function Refused _ -> true | _ -> false))
        (count (function Failed _ -> true | _ -> false)));
  let reached = List.map cases ~f:(fun c -> c.compute) in
  List.iter precs ~f:(fun p ->
      if not (List.mem reached p ~equal:Ops.equal_prec) then
        Stdio.eprintf "  no policy computes at %s: its arms are unreachable here\n"
          (Ops.prec_string p));
  List.iter all_results ~f:(fun (case, (arm, verdict)) ->
      match verdict with
      | Compiled -> ()
      | Refused why -> Stdio.eprintf "  refused: %s, %s -- %s\n" (label case) (arm_name arm) why
      | Failed why -> Stdio.eprintf "  FAILED: %s, %s -- %s\n" (label case) (arm_name arm) why);
  List.iter !unexplained ~f:(fun (what, verdict) ->
      Stdio.eprintf "  UNEXPLAINED: %s -- %s\n" what
        (match verdict with Compiled -> "" | Refused why | Failed why -> why));
  p_none ~min:200 "no in-domain arm renders and then fails the backend's compiler" all_results
    ~f:(fun (_, (_, verdict)) -> match verdict with Failed _ -> true | _ -> false);
  p_empty "every failing routine's failure is pinned to one arm" ~over:cases !unexplained;
  (* A backend may render MORE than the C reference (CUDA renders integer [Recip] as an integer
     division), never refuse more: a refusal outside the C reference's is an operator a user of this
     backend alone cannot use, or an arm the sweep built wrongly. At a float precision the reference
     refuses nothing but a conversion out of packed RNG state. *)
  p_all "every refused arm is one the C reference refuses too" all_results
    ~f:(fun (case, (arm, v)) ->
      match v with
      | Refused _ -> c_reference_refuses ~store:case.store ~compute:case.compute arm
      | Compiled | Failed _ -> true);
  p_all "every storage precision the backend holds compiled at least one arm" precs ~f:(fun p ->
      List.exists all_results ~f:(fun (case, (_, v)) ->
          Ops.equal_prec case.store p && match v with Compiled -> true | _ -> false))

(* The executed legs. Compiling an arm says nothing about its value, and the integer-precision table
   fixes here had values to get right. Each case is one the wrong fix gets wrong, with every operand
   and result small enough for a byte, so no out-of-range conversion is involved on any backend.

   [Min(a + b, c)] over [200 + 100, 250] is 250: C compares the integer-promoted sum, where
   narrowing the operands to [uchar] first, to resolve the overload, compares 44 with 250.

   [two * Sin(a)] over [a = 1] is 1: the sine stays a float until the store, where casting a math
   result back to the integer precision first gives 2 * 0.

   [two * Min(Sin(a), Sin(b))] over [a = b = 1] is 1: the comparison happens in the operands' common
   type, where casting them to an integer type to resolve the overload gives 2 * 0.

   [two * Min(Sin(b), Sqrt(Neg(a)))] over [a = b = 1] is 1: a floating comparison keeps [fmin]'s NaN
   handling, where a plain [<] selects the NaN, an invalid conversion at the byte store. (Both
   operands are float-valued: an integer read against a float intermediate is a mixed-type call
   CUDA's integer [min] does not resolve, the float-intermediate class the sweep leaves out -- see
   docs/agent-notes/backend-dialects-and-idents.md.)

   [two * Trunc(Sin(a))] over [a = 1] is 0: an operand at an integer precision can be float-valued,
   so integer [Trunc] is not the identity (that gives 2 * 0.84 = 1.68, stored 1). *)
let () =
  let byte = Ops.byte in
  let mk label = node ~prec:byte ("value_" ^ label) in
  let a = mk "a" and b = mk "b" and c = mk "c" and two = mk "two" in
  let cases =
    [
      ( "byte Min over an integer-promoted sum compares before narrowing",
        LL.Binop (Ops.Min, (LL.Binop (Ops.Add, read a byte, read b byte), byte), read c byte),
        [ (a, 200.); (b, 100.); (c, 250.) ],
        250. );
      ( "byte math over an integer operand keeps its float result until the store",
        LL.Binop (Ops.Mul, read two byte, (LL.Unop (Ops.Sin, read a byte), byte)),
        [ (a, 1.); (two, 2.) ],
        1. );
      ( "byte Min compares float-valued operands as floats",
        LL.Binop
          ( Ops.Mul,
            read two byte,
            ( LL.Binop
                ( Ops.Min,
                  (LL.Unop (Ops.Sin, read a byte), byte),
                  (LL.Unop (Ops.Sin, read b byte), byte) ),
              byte ) ),
        [ (a, 1.); (b, 1.); (two, 2.) ],
        1. );
      ( "byte Min of a float and a NaN intermediate is the float, as fmin's is",
        LL.Binop
          ( Ops.Mul,
            read two byte,
            ( LL.Binop
                ( Ops.Min,
                  (LL.Unop (Ops.Sin, read b byte), byte),
                  (LL.Unop (Ops.Sqrt, (LL.Unop (Ops.Neg, read a byte), byte)), byte) ),
              byte ) ),
        [ (a, 1.); (b, 1.); (two, 2.) ],
        1. );
      ( "byte Trunc truncates a float-valued operand",
        LL.Binop
          ( Ops.Mul,
            read two byte,
            (LL.Unop (Ops.Trunc, (LL.Unop (Ops.Sin, read a byte), byte)), byte) ),
        [ (a, 1.); (two, 2.) ],
        0. );
    ]
  in
  List.iteri cases ~f:(fun i (claim, llsc, seed, want) ->
      let out = mk (Printf.sprintf "out%d" i) in
      let llc = Ll_test.set out at0 llsc in
      let name = Printf.sprintf "optbl_value_%d" i in
      let inputs = List.map seed ~f:fst in
      let got =
        Ll_test.execute ~name
          { (Ll_test.optimize ~materialized:(out :: inputs) ~name llc) with LL.llc }
          ~seed:(List.map seed ~f:(fun (tn, v) -> (tn, [| v |])))
          ~read:[ out ]
      in
      p claim (Ll_test.same got [ [| want |] ]))
