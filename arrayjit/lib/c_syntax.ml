open Base
module Lazy = Utils.Lazy
open Backend_intf

let _get_local_debug_runtime = Utils.get_local_debug_runtime

[%%global_debug_log_level 0]

(* export OCANNL_LOG_LEVEL_C_SYNTAX=9 to enable debugging into the log_files/ directory. *)
[%%global_debug_log_level_from_env_var "OCANNL_LOG_LEVEL_C_SYNTAX"]

module Tn = Tnode

type t = PPrint.document

(* gh-ocannl-344: integer width of the Metal pooled slot table (pool_index, byte_offset per node).
   With [large_models] the per-pool 4 GB cap is lifted (see {!Backends.plan_pool_segments}), so a
   byte offset can exceed [UINT32_MAX] and the slot table -- together with the MSL type the shader
   declares -- must be 64-bit to avoid silent truncation; otherwise 32-bit suffices (offsets are
   capped under 4 GB). This is the single source of truth shared by the codegen (the [const ... *
   __pool_slots] MSL type) and the backend (the [Ctypes] element type), so the two cannot drift.
   [large_models] is a startup-fixed global, read identically when the source is generated and when
   the table is filled. *)
let pool_slot_is_64 () = Utils.settings.large_models
let pool_slot_msl_typ () = if pool_slot_is_64 () then "ulong" else "uint"

(* Opt-in rendering-decline diagnostics (config [schedule_log_declines]; gh-ocannl-474 /
   gh-ocannl-479): one stderr line per decline, naming the kernel, the construct, and the rule that
   failed — when a [Grid] loop fails pool-parallel eligibility ([parallel_grid_safe]) or a
   [Tile_mma] statement falls back from the intrinsic / register-tiled rendering. Complements
   [schedule_log_launches]: that one says what geometry a compile launched, this one says why a
   requested rendering was NOT used (declines are silent by design — the fallback is correct — but
   indistinguishable from the fast path in timings, which already cost one wrong conclusion:
   docs/proposals/tensorize-mma.md's unverified-T3 episode). *)
let log_declines = lazy (Utils.get_global_flag ~default:false ~arg_name:"schedule_log_declines")

(* Census of [Tile_mma] statement renderings, collected during codegen while [mma_census_enabled]
   (gh-ocannl-479): [Autotune] flips it around candidate compiles, because "the tensorized candidate
   lost" and "the tensorized candidate never ran tensorized" must be distinguishable in tuning logs.
   Entries are (kernel name, rendering), most recent first; tests assert on it directly. *)
type mma_rendering = Mma_intrinsics | Mma_register_tiled | Mma_scalar_fallback
[@@deriving sexp, compare, equal]

let mma_census_enabled = ref false
let mma_census : (string * mma_rendering) list ref = ref []

module type C_syntax_config = sig
  val procs : Low_level.optimized array
  (** The low-level prcedure to compile, and the arrays of the context it will be linked to if not
      shared and already known. *)

  type buffer_ptr

  val main_kernel_prefix : string
  val kernel_prep_line : string
  val buffer_prefix : string
  val buffer_suffix : pos:int -> string
  val arg_int_prefix : string
  val loop_index_type : string
  val extra_args : string list
  val typ_of_prec : Ops.prec -> string
  val vec_typ_of_prec : length:int -> Ops.prec -> string
  val ident_blacklist : string list

  val ptr_param_style : [ `Per_param | `Pooled of int ]
  (** How materialized in-context tensor nodes are passed to the kernel. [`Per_param] (the default,
      used by C and CUDA) emits one typed pointer parameter per node, whose host side binds
      [pool_base + offset] -- byte-identical to the pre-pooling codegen. [`Pooled n] (Metal) emits a
      fixed [n] byte-pointer pool parameters plus one (pool_index, byte_offset) slot table, and a
      kernel prologue that forms each node's typed pointer by casting (pools at slot.pool) + offset.
      This collapses O(num_nodes) buffer bindings to [n] + 1, which Metal needs to stay under its
      ~31 argument-buffer binding limit. *)

  val float_log_style : string
  (** Format specifier for printing floating point numbers in debug logs. *)

  val styled_log_arg : PPrint.document -> PPrint.document
  (** Function to convert potentially floating-point numeric values for logging. *)

  val ternop_syntax :
    Ops.prec ->
    Ops.ternop ->
    PPrint.document ->
    PPrint.document ->
    PPrint.document ->
    PPrint.document

  val binop_syntax : Ops.prec -> Ops.binop -> PPrint.document -> PPrint.document -> PPrint.document
  val unop_syntax : Ops.prec -> Ops.unop -> PPrint.document -> PPrint.document
  val vec_unop_syntax : Ops.prec -> Ops.vec_unop -> PPrint.document -> PPrint.document
  val convert_precision : from:Ops.prec -> to_:Ops.prec -> string * string

  val hardware_index : kind:[ `Grid | `Workgroup ] -> slot:int -> string option
  (** The hardware register expression an annotated loop's index binds to (e.g. ["blockIdx.x"],
      ["gid.y"]), or [None] when the backend cannot bind this axis in hardware — the loop then
      renders as a serial [for] (a legal implementation absent barriers; see
      docs/proposals/axis-types-for-loops.md §2/§5). Slots are positional: 0 = [.x], 1 = [.y], 2 =
      [.z]. *)

  val barrier_syntax : string option
  (** Workgroup barrier statement ([__syncthreads();] / [threadgroup_barrier(...);]); [None] makes
      [Workgroup_barrier] a compile-time error (serialization cannot implement a barrier). *)

  val parallel_grid_syntax : [ `None | `Dispatch | `Openmp ]
  (** Pool-backed [Grid] rendering (docs/proposals/gh-ocannl-164.md): how to render an eligible
      outermost [Grid] loop when [hardware_index] does not bind it. [`Dispatch] emits libdispatch's
      [dispatch_apply] over contiguous chunks (macOS; blocks extension), [`Openmp] a
      [#pragma omp parallel for] over the chunk loop; both runtimes own a single process-global
      thread pool, so no pool state lives in the compiled kernel. [`None] keeps the serial fallback.
      Eligibility is decided per loop by [compile_proc] (see [parallel_grid_safe]); [Workgroup]
      loops always stay serial inside a chunk. *)

  val parallel_grid_chunks : int
  (** Target chunk count for [parallel_grid_syntax] (e.g. a small multiple of the core count); the
      actual count is capped by the loop extent. Values [<= 1] disable parallel rendering. *)

  val shared_decl_prefix : string option
  (** Declaration prefix for workgroup-shared placements ([__shared__ ] / [threadgroup ]); [None]
      makes a non-empty [workgroup_shared] set a compile-time error. *)

  val volatile_scalar_rmw : bool
  (** Workaround for a Metal shader-compiler miscompilation (observed on macOS 15/Metal 3.1-3.2,
      reproduced standalone in [benchmarks/runners/ocannl/bench_metal_bug.ml]): a serial loop
      accumulating into a loop-invariant address of a kernel-parameter-derived pointer —
      [acc[k] = acc[k] + f(i)] with [k] free of [i] — can execute as if the load were hoisted above
      the loop and the store sunk below it {e without} carrying the accumulation, leaving only the
      last iteration's contribution (scalar losses collapsed to the last sample's CE; [w.grad]
      accumulated only the last batch element). The trigger involves pointers derived from
      dynamically-loaded offsets (the pooled-parameter slot table) but is otherwise capricious —
      plain-FMA and inlined-recompute statements alike miscompiled in some kernels and compiled fine
      in byte-alike others — so the rule keys on the pass's precondition: when [true], [Set]
      statements that read the written node at an index invariant across at least one enclosing
      serial [for] loop render both accesses through a [volatile]-qualified shadow pointer, pinning
      the per-iteration read-modify-write. This covers reduction accumulators (address invariant
      across the reduction loop); pointwise updates stay unqualified. *)

  val restrict_keyword : string option
  (** No-alias qualifier for kernel pointer parameters and, in the pooled style, for the derived
      per-node pointers ([restrict] / [__restrict__] / [__restrict]); [None] emits no qualifier.
      Sound because kernel parameters are buffer-owning roots addressing disjoint (sub-)ranges:
      alias views are rewritten to parent accesses at assignments lowering and never reach
      [compile_proc]'s parameter list (asserted there; gh-ocannl-164). The merge buffer stays
      unqualified — a streaming merge mode could point it at a live same-device buffer. *)

  val vectorize_pragma : string list
  (** Lines emitted verbatim before a [Vectorized]-typed loop's [for] statement (gh-ocannl-164),
      e.g. guarded [#pragma clang loop vectorize(enable)] / [#pragma GCC ivdep]. An empty list
      renders the loop as a plain serial [for] — the legal fallback, mirroring
      [hardware_index = None]. Used when explicit vector emission ([vector_bytes]) is disabled or
      the loop body is ineligible for it. *)

  val vector_bytes : int
  (** Vector register width in bytes for explicit SIMD rendering of [Vectorized] loops via GCC/Clang
      vector extensions (the [Vectorized] codegen follow-up of gh-ocannl-164 /
      docs/proposals/watch-ocannl-README-md-347818d3.md): eligible loop bodies emit vector-typed
      loads, arithmetic and stores in [lanes = vector_bytes / element size] chunks plus a serial
      remainder loop, instead of relying on the compiler's auto-vectorizer (which e.g. cannot
      reassociate strict-FP reductions — the [Vectorized] retype carries that permission, like
      [Swap]). A recognized accumulation body renders as independent accumulator chains with a
      horizontal reduce at loop exit (gh-ocannl-468; [`Vec_extensions] only). [0] disables explicit
      emission ([vectorize_pragma] fallback). *)

  val vector_style : [ `Vec_extensions | `Packed_struct ]
  (** How eligible [Vectorized] loops emit explicit vector code when [vector_bytes > 0].
      [`Vec_extensions] (CPU): GCC/Clang [vector_size] types, unaligned [__builtin_memcpy]
      loads/stores, vector-infix arithmetic. [`Packed_struct] (GPU, gh-ocannl-463; llm.c's
      [Packed128], llmc/cuda_utils.cuh): the backend's [vec_typ_of_prec] aggregate is loaded and
      stored through [reinterpret_cast] at guaranteed-aligned offsets — the 128-bit LDG/STS
      transactions that bandwidth-bound kernels need — while the arithmetic stays scalar in a
      per-lane loop over the pack's [.v] payload (on GPU the payoff is memory transactions, not SIMD
      ALUs; per-lane [fmaf]/[fma] also matches the serial path's rounding exactly). [`Packed_struct]
      eligibility additionally requires every vector-accessed node to be materialized (device
      buffers and pool offsets are [Ops.buffer_alignment]-aligned, stack and workgroup-shared arrays
      only element-aligned) and every access's non-loop offset contribution to be a lane multiple.
  *)

  val aligned_local_attr : string option
  (** Declaration suffix aligning stack-allocated local arrays for SIMD access, e.g.
      [__attribute__((aligned(32)))] (gh-ocannl-164). Applies to the plain stack-array branch only,
      never to workgroup-shared placements. *)

  val warp_size : int
  (** SIMD-group (warp) width for the warp-shuffle rendering of [Workgroup_reduce] accumulation
      loops (gh-ocannl-462; llm.c's [warpReduceSum]/[blockReduce] idiom). Backends setting this to a
      nonzero power of two must define [ocannl_shfl_xor(value, lane_mask)] overloads in their
      builtins for the supported accumulator precisions (single, and double where it exists), bind
      workgroup slot 0 in [hardware_index], and provide [barrier_syntax] plus [shared_decl_prefix]
      (needed by the two-phase multi-warp form). [0] disables the rendering: [Workgroup_reduce]
      loops render like [Workgroup] — hardware binding, or the serial fallback (which is the correct
      meaning of a recognized accumulation body on CPU backends). *)

  val mma_syntax :
    (d_prec:Ops.prec ->
    a_prec:Ops.prec ->
    b_prec:Ops.prec ->
    ta:bool ->
    tb:bool ->
    m:int ->
    n:int ->
    k:int ->
    d:PPrint.document * int * [ `Device | `Shared | `Thread | `Fragment of string ] ->
    a:PPrint.document * int * [ `Device | `Shared | `Thread | `Fragment of string ] ->
    b:PPrint.document * int * [ `Device | `Shared | `Thread | `Fragment of string ] ->
    PPrint.document option)
    option
  (** Cooperative tile-MMA emission for [Low_level.Tile_mma] (docs/proposals/tensorize-mma.md §4):
      given the per-operand precisions (the backend decides which combinations its units support —
      Metal [simdgroup_matrix] is uniform-precision only, CUDA wmma's flagship combination is mixed
      f16×f16→f32), the transposed-storage flags [ta]/[tb] (the operand's stored layout is the
      transpose of its role — load tiles with the hardware transpose flag and swapped offset
      arithmetic), the covered block extents [m]/[n]/[k], and per operand a pointer expression to
      the tile base (already offset), its leading-dimension stride in elements, and its address
      space, emit the intrinsic sequence (fragment declarations / loads / mma steps / stores)
      executed by every lane of the enclosing lane loop. Return [None] to decline a particular call
      (unsupported precision combination, extents not multiples of the intrinsic tile, thread-space
      operand) — the caller then renders the scalar [fallback] under an [if (lane == 0)] guard,
      which is also the path when the whole hook is [None] (cc, and any backend until wired). *)

  val mma_fragment_syntax :
    (d_prec:Ops.prec ->
    a_prec:Ops.prec ->
    b_prec:Ops.prec ->
    m:int ->
    n:int ->
    k:int ->
    fragment:string ->
    target:PPrint.document * int * [ `Device | `Shared | `Thread | `Fragment of string ] ->
    a:PPrint.document * int * [ `Device | `Shared | `Thread | `Fragment of string ] ->
    b:PPrint.document * int * [ `Device | `Shared | `Thread | `Fragment of string ] ->
    body:(unit -> PPrint.document) ->
    PPrint.document option)
    option
  (** Rendering of a marked cross-reduction accumulator lifetime. The callback receives the backing
      target and one representative [Tile_mma]'s operands/shape, so it can decline before forcing
      [body]. When accepted, forcing [body] renders its [Tile_mma] with [d] identified as
      [`Fragment fragment], allowing the backend to emit update-only MMA steps between one outer
      fragment load and store. *)

  val kernel_log_param : (string * string) option
  (** Kernel parameter for logging, if any. E.g., (Some ("int", "log_id")) or (Some ("const char*",
      "log_file_name")). *)

  val log_involves_file_management : bool
  (** Whether the logging setup involves opening/closing a FILE* (e.g., for fprintf). *)

  val pp_log_statement :
    log_param_c_expr_doc:PPrint.document option ->
    base_message_literal:string ->
    args_docs:PPrint.document list ->
    PPrint.document
  (** Generates a C log statement.
      - [log_param_c_expr_doc]: Document for the C expression of the log parameter (e.g.,
        [string "log_id"] or [string "log_file_name"]), if [kernel_log_param] is Some).
      - [base_message_literal]: The raw, unescaped, unquoted base printf-style format string (e.g.,
        "index %s = %d\n").
      - [args_docs]: Documents for the C expressions of the arguments to the format string. The
        implementation should handle quoting [base_message_literal], choosing the log function
        (printf, fprintf, os_log), and prepending any necessary prefixes (like a log_id or
        captured_log_prefix) to the format string and arguments. *)
end

module Pure_C_config (Input : sig
  type buffer_ptr

  val procs : Low_level.optimized array
  val full_printf_support : bool
end) =
struct
  let procs = Input.procs

  type nonrec buffer_ptr = Input.buffer_ptr

  let main_kernel_prefix = ""
  let kernel_prep_line = ""
  let buffer_prefix = ""
  let buffer_suffix = fun ~pos:_ -> ""

  (* Signed index arithmetic (docs/proposals/signed-index-precision.md); the width tracks
     [Ops.index_prec ()]. *)
  let arg_int_prefix = if Utils.settings.large_models then "const int64_t " else "const int32_t "
  let loop_index_type = if Utils.settings.large_models then "int64_t " else "int32_t "
  let extra_args = []
  let typ_of_prec = Ops.c_typ_of_prec
  let vec_typ_of_prec = Ops.c_vec_typ_of_prec
  let ptr_param_style = `Per_param

  (* Plain C backends bind no hardware axes: annotated loops fall back to serial [for] loops (sound
     absent barriers), and barriers / shared placements are compile-time errors. *)
  let hardware_index ~kind:_ ~slot:_ = None
  let barrier_syntax = None
  let parallel_grid_syntax = `None
  let parallel_grid_chunks = 1
  let vector_bytes = 0
  let vector_style = `Vec_extensions
  let shared_decl_prefix = None
  let restrict_keyword = Some "restrict"
  let volatile_scalar_rmw = false

  (* Clang defines both [__clang__] and [__GNUC__], so test [__clang__] first. *)
  let vectorize_pragma =
    [
      "#if defined(__clang__)";
      "#pragma clang loop vectorize(enable) interleave(enable)";
      "#elif defined(__GNUC__)";
      "#pragma GCC ivdep";
      "#endif";
    ]

  let aligned_local_attr = Some (Printf.sprintf "__attribute__((aligned(%d)))" Ops.buffer_alignment)

  (* No shuffle intrinsics on plain C backends: a [Workgroup_reduce] accumulation loop renders as
     the serial fallback, which is exactly its serial meaning. *)
  let warp_size = 0

  (* No tile-MMA units on plain C backends: [Tile_mma] renders its scalar fallback under the [lane
     == 0] guard. *)
  let mma_syntax = None
  let mma_fragment_syntax = None
  let float_log_style = if Input.full_printf_support then "%g" else "%de-3"

  let styled_log_arg doc =
    if Input.full_printf_support then doc
    else
      let open PPrint in
      string "(int)(" ^^ doc ^^ string " * 1000.0)"

  let ident_blacklist =
    (* Extract all maximal identifier-like substrings (starting with a letter or underscore,
       consisting of alphanumeric chars and underscores) from an op syntax prefix string. This
       correctly decomposes composite prefixes like "(fabsf(floorf(" into ["fabsf"; "floorf"] rather
       than the old remove_paren approach that produced the wrong concatenation "fabsffloorf". *)
    let extract_fn_names s =
      let n = String.length s in
      let result = ref [] in
      let i = ref 0 in
      while !i < n do
        if Char.is_alpha s.[!i] || Char.equal s.[!i] '_' then begin
          let j = ref !i in
          while !j < n && (Char.is_alphanum s.[!j] || Char.equal s.[!j] '_') do
            Int.incr j
          done;
          result := String.sub s ~pos:!i ~len:(!j - !i) :: !result;
          i := !j
        end
        else Int.incr i
      done;
      !result
    in
    let add_from_prefix functions prefix =
      List.iter (extract_fn_names prefix) ~f:(fun name -> functions := Set.add !functions name)
    in
    let functions = ref (Set.empty (module String)) in
    let precs = Ops.[ byte; int32; uint32; half; bfloat16; fp8; single; double ] in
    List.iter precs ~f:(fun prec ->
        List.iter
          Ops.[ Where; FMA; Mul3 ]
          ~f:(fun op ->
            let p, _, _, _ =
              try Ops.ternop_c_syntax prec op with Invalid_argument _ -> ("", "", "", "")
            in
            add_from_prefix functions p);
        List.iter
          Ops.
            [
              Add;
              Sub;
              Mul;
              Div;
              ToPowOf;
              Relu_gate;
              Satur01_gate;
              Max;
              Min;
              Mod;
              Cmplt;
              Cmpeq;
              Cmpne;
              Or;
              And;
              Threefry4x32_crypto;
              Threefry4x32_light;
            ]
          ~f:(fun op ->
            let p, _, _ =
              try Ops.binop_c_syntax prec op with Invalid_argument _ -> ("", "", "")
            in
            add_from_prefix functions p);
        List.iter
          Ops.
            [
              Identity;
              Relu;
              Satur01;
              Exp;
              Log;
              Exp2;
              Log2;
              Sin;
              Cos;
              Sqrt;
              Recip;
              Recip_sqrt;
              Neg;
              Tanh_approx;
              Not;
              Uint4x32_to_prec_uniform1;
            ]
          ~f:(fun op ->
            let p, _ = try Ops.unop_c_syntax prec op with Invalid_argument _ -> ("", "") in
            add_from_prefix functions p);
        List.iter
          Ops.[ Uint4x32_to_prec_uniform ]
          ~f:(fun op ->
            let p, _ = try Ops.vec_unop_c_syntax prec op with Invalid_argument _ -> ("", "") in
            add_from_prefix functions p));
    let c_keywords =
      [
        (* C89 keywords *)
        "auto";
        "break";
        "case";
        "char";
        "const";
        "continue";
        "default";
        "do";
        "double";
        "else";
        "enum";
        "extern";
        "float";
        "for";
        "goto";
        "if";
        "int";
        "long";
        "register";
        "return";
        "short";
        "signed";
        "sizeof";
        "static";
        "struct";
        "switch";
        "typedef";
        "union";
        "unsigned";
        "void";
        "volatile";
        "while";
        (* C99 additions *)
        "inline";
        "restrict";
        "_Bool";
        "_Complex";
        "_Imaginary";
        (* Scaffolding names emitted by generated code that must not clash with variable names *)
        "log_file";
        "log_file_name";
        "uint32_t";
        "uint64_t";
      ]
    in
    Set.to_list !functions @ c_keywords

  let ternop_syntax prec op v1 v2 v3 =
    let op_prefix, op_infix1, op_infix2, op_suffix = Ops.ternop_c_syntax prec op in
    let open PPrint in
    group
      (string op_prefix ^^ v1 ^^ string op_infix1
      ^^ ifflat (space ^^ v2) (nest 2 (break 1 ^^ v2))
      ^^ string op_infix2
      ^^ ifflat (space ^^ v3) (nest 2 (break 1 ^^ v3))
      ^^ string op_suffix)

  let binop_syntax prec op v1 v2 =
    let op_prefix, op_infix, op_suffix = Ops.binop_c_syntax prec op in
    let open PPrint in
    group
      (string op_prefix ^^ v1 ^^ string op_infix
      ^^ ifflat (space ^^ v2) (nest 2 (break 1 ^^ v2))
      ^^ string op_suffix)

  let unop_syntax prec op v =
    let op_prefix, op_suffix = Ops.unop_c_syntax prec op in
    let open PPrint in
    group (string op_prefix ^^ v ^^ string op_suffix)

  let vec_unop_syntax prec op v =
    let op_prefix, op_suffix = Ops.vec_unop_c_syntax prec op in
    let open PPrint in
    group (string op_prefix ^^ v ^^ string op_suffix)

  let convert_precision = Ops.c_convert_precision
  let kernel_log_param = Some ("const char*", "log_file_name")
  let log_involves_file_management = true

  let for_log_trace_tree =
    Utils.get_global_flag ~default:false ~arg_name:"debug_log_to_stream_files"

  let pp_log_statement ~log_param_c_expr_doc:_ ~base_message_literal ~args_docs =
    let open PPrint in
    let log_file_check =
      match kernel_log_param with
      | Some (_, lname) -> string ("if (" ^ lname ^ " && log_file) ")
      | None ->
          string "if (log_file) " (* Should not happen if log_involves_file_management is true *)
    in
    let base_message_literal =
      let with_ = if for_log_trace_tree then "$" else "\\n" in
      let res = String.substr_replace_all base_message_literal ~pattern:"\n" ~with_ in
      if for_log_trace_tree && String.is_suffix res ~suffix:"$" then
        String.drop_suffix res 1 ^ "\\n"
      else res
    in
    log_file_check
    ^^ group
         (string "fprintf(log_file, "
         ^^ dquotes (string base_message_literal)
         ^^ (if List.is_empty args_docs then empty
             else comma ^^ nest 4 (break 1 ^^ separate (comma ^^ break 1) args_docs))
         ^^ rparen ^^ semi)
end

module C_syntax (B : C_syntax_config) = struct
  let get_ident =
    Low_level.get_ident_within_code ~no_dots:true ~blacklist:B.ident_blacklist
    @@ Array.map B.procs ~f:(fun l -> l.llc)

  (* Set by [compile_proc]: the per-compilation-lineage placement resolution
     (docs/proposals/context-scoped-memory-modes.md). Codegen both consults and settles placements
     here -- never on the tnode. *)
  let current_placements : Tn.Placements.t option ref = ref None

  let placements () =
    Option.value_exn ~message:"C_syntax: placements consulted outside compile_proc"
      !current_placements

  let in_ctx tn = Tn.Placements.is_in_context_force (placements ()) tn 46

  let filter_and_prepend_builtins ~includes ~builtins ~proc_doc =
    let doc_buffer = Buffer.create 4096 in
    PPrint.ToBuffer.pretty 1.0 110 doc_buffer proc_doc;
    let doc_string = Buffer.contents doc_buffer in
    let result_buffer = Buffer.create 4096 in
    Buffer.add_string result_buffer includes;
    Buffer.add_string result_buffer "\n";

    (* Collect all needed keys, including dependencies *)
    let needed_keys = ref (Set.empty (module String)) in
    List.iter builtins ~f:(fun (key, _, _) ->
        if String.is_substring doc_string ~substring:key then
          needed_keys := Set.add !needed_keys key);

    (* Add dependencies recursively *)
    let processed_keys = ref (Set.empty (module String)) in
    let rec add_dependencies key =
      if not (Set.mem !processed_keys key) then (
        processed_keys := Set.add !processed_keys key;
        needed_keys := Set.add !needed_keys key;
        match List.find builtins ~f:(fun (k, _, _) -> String.equal k key) with
        | Some (_, _, deps) -> List.iter deps ~f:add_dependencies
        | None -> ())
    in
    Set.iter !needed_keys ~f:add_dependencies;

    (* Add the builtins in order *)
    List.iter builtins ~f:(fun (key, definition, _) ->
        if Set.mem !needed_keys key then (
          Buffer.add_string result_buffer definition;
          Buffer.add_string result_buffer "\n"));
    Buffer.add_string result_buffer doc_string;
    Buffer.contents result_buffer

  open Indexing
  open Doc_helpers

  let pp_array_offset (idcs, dims) =
    let open PPrint in
    if Array.is_empty idcs then string "0"
    else
      let doc = ref (pp_axis_index idcs.(0)) in
      for i = 1 to Array.length idcs - 1 do
        let idx_doc = pp_axis_index idcs.(i) in
        if PPrint.is_empty !doc then doc := idx_doc
        else if PPrint.is_empty idx_doc then
          doc := parens !doc ^^ string (" * " ^ Int.to_string dims.(i))
        else doc := parens !doc ^^ string (" * " ^ Int.to_string dims.(i) ^ " + ") ^^ idx_doc
      done;
      !doc

  (* gh-343: like [pp_array_offset] but at [dyn_axis] splices [dyn_idx_doc] (an integer expression
     derived from a runtime value) instead of the static [axis_index]. Used for [Get_dynamic]. *)
  let pp_array_offset_dyn (idcs, dims) ~dyn_axis ~dyn_idx_doc =
    let open PPrint in
    let axis_doc i = if i = dyn_axis then dyn_idx_doc else pp_axis_index idcs.(i) in
    if Array.is_empty idcs then string "0"
    else begin
      let doc = ref (axis_doc 0) in
      for i = 1 to Array.length idcs - 1 do
        let idx_doc = axis_doc i in
        if PPrint.is_empty !doc then doc := idx_doc
        else if PPrint.is_empty idx_doc then
          doc := parens !doc ^^ string (" * " ^ Int.to_string dims.(i))
        else doc := parens !doc ^^ string (" * " ^ Int.to_string dims.(i) ^ " + ") ^^ idx_doc
      done;
      !doc
    end

  let doc_to_string doc =
    let buf = Buffer.create 128 in
    PPrint.ToBuffer.compact buf doc;
    Buffer.contents buf

  let array_offset_to_string (idcs, dims) = doc_to_string @@ pp_array_offset (idcs, dims)

  let pp_local_defs (local_defs : (int * PPrint.document) list) =
    let open PPrint in
    List.dedup_and_sort local_defs ~compare:(fun (a, _) (b, _) -> Int.compare a b)
    |> List.map ~f:snd |> separate hardline

  let pp_scope_id Low_level.{ scope_id; tn } =
    let open PPrint in
    string ("v" ^ Int.to_string scope_id ^ "_" ^ get_ident tn)

  (* {3 Vector-extension accumulator grids (gh-ocannl-468 / gh-ocannl-469)}

     Emission helpers for grids of vector-register accumulators in the [`Vec_extensions] style. The
     SIMD reduction rendering of [Vectorized] accumulation loops (gh-ocannl-468; ggml's
     [ggml_vec_dot_f32] pattern) uses a 1×N grid — N independent accumulator chains folded at loop
     exit — and the register-tiled [Tile_mma] micro-kernel (gh-ocannl-469) will hold its C-tile as
     an RM×RN grid of vector registers across the fused k-loop. *)

  (* The vector-extension typedef shared by the explicit-SIMD renderings: [lanes] elements of [prec]
     (single or double). *)
  let vec_ext_typ ~prec ~lanes =
    let vtyp =
      Printf.sprintf "ocannl_vec%d%s" lanes (match prec with Ops.Double_prec _ -> "d" | _ -> "f")
    in
    ( vtyp,
      PPrint.string
        (Printf.sprintf "typedef %s %s __attribute__((vector_size(%d)));" (B.typ_of_prec prec) vtyp
           (lanes * Ops.prec_in_bytes prec)) )

  (* The names of a [rows]×[cols] grid of vector accumulator registers. *)
  let vec_acc_grid ~prefix ~rows ~cols : string array array =
    Array.init rows ~f:(fun r ->
        Array.init cols ~f:(fun c -> Printf.sprintf "%s_%d_%d__" prefix r c))

  (* [dst = op(dst, src)] on whole vector registers ([src] must also name a register): vector infix
     arithmetic for the ring operators, a fixed-trip per-lane loop for [Max]/[Min] (which have no
     vector infix; keeps the scalar path's [fmaxf]/[fminf] NaN semantics and SLP-vectorizes under
     -O2 like the per-lane FMA fallback). *)
  let vec_acc_combine ~prec ~lanes ~op ~dst ~src =
    let open PPrint in
    match op with
    | Ops.Add | Ops.Sub | Ops.Mul | Ops.Div ->
        let inf = match op with Ops.Add -> " + " | Sub -> " - " | Mul -> " * " | _ -> " / " in
        string (Printf.sprintf "%s = %s%s%s;" dst dst inf src)
    | Ops.Max | Ops.Min ->
        let lane v = string (v ^ "[ocannl_l__]") in
        string
          (Printf.sprintf
             "for (int ocannl_l__ = 0; ocannl_l__ < %d; ++ocannl_l__) %s[ocannl_l__] = " lanes dst)
        ^^ B.binop_syntax prec op (lane dst) (lane src)
        ^^ semi
    | _ -> invalid_arg "C_syntax.vec_acc_combine: not an accumulation operator"

  (* [dst = fma(a, b, dst)] elementwise-fused on vector registers (single rounding, matching the
     scalar path's [fmaf]/[fma]): clang's [__builtin_elementwise_fma] where available, otherwise the
     per-lane fused loop. *)
  let vec_acc_fma ~prec ~lanes ~dst ~a ~b =
    let open PPrint in
    let fma_fn = match prec with Ops.Double_prec _ -> "fma" | _ -> "fmaf" in
    string "#if OCANNL_HAS_ELEMENTWISE_FMA"
    ^^ hardline
    ^^ string (Printf.sprintf "%s = __builtin_elementwise_fma(%s, %s, %s);" dst a b dst)
    ^^ hardline ^^ string "#else" ^^ hardline
    ^^ string
         (Printf.sprintf
            "for (int ocannl_l__ = 0; ocannl_l__ < %d; ++ocannl_l__) %s[ocannl_l__] = \
             %s(%s[ocannl_l__], %s[ocannl_l__], %s[ocannl_l__]);"
            lanes dst fma_fn a b dst)
    ^^ hardline ^^ string "#endif"

  (* Fold every register of the grid into [grid.(0).(0)] (a statement list; empty for a 1×1
     grid). *)
  let vec_acc_grid_fold ~prec ~lanes ~op (grid : string array array) : PPrint.document list =
    let dst = grid.(0).(0) in
    Array.concat_map grid ~f:Fn.id |> Array.to_list
    |> List.filter ~f:(fun s -> not (String.equal s dst))
    |> List.map ~f:(fun src -> vec_acc_combine ~prec ~lanes ~op ~dst ~src)

  (* Horizontal reduction of one vector register into the fresh scalar [out] (a statement list): a
     linear lane chain through the scalar combine — per-element semantics identical to the serial
     loop's. *)
  let vec_acc_lane_fold ~prec ~lanes ~op ~vname ~out : PPrint.document list =
    let open PPrint in
    string (Printf.sprintf "%s %s = %s[0];" (B.typ_of_prec prec) out vname)
    :: List.init (lanes - 1) ~f:(fun l ->
        string (out ^ " = ")
        ^^ B.binop_syntax prec op (string out) (string (Printf.sprintf "%s[%d]" vname (l + 1)))
        ^^ semi)

  (* Set by [compile_proc] so that [pp_ll] can consult per-node tracing info (e.g. to elide a
     [Zero_out] loop that the declaration's [= {0}] already covers). *)
  let current_traced_store : Low_level.traced_store option ref = ref None

  (* Set by [compile_proc]: the kernel's hardware-annotated loops with their positional slots
     (docs/proposals/axis-types-for-loops.md §1/§5), consulted by [pp_ll]'s [For_loop] case to
     render hardware index bindings. *)
  let current_hardware_axes : Low_level.hardware_axis_info list ref = ref []

  (* Set by [compile_proc]: nodes placed in workgroup-shared memory. Their declarations carry
     [shared_decl_prefix] and cannot use [= {0}] (not allowed for [__shared__]/[threadgroup]), so
     their [Zero_out] is never elided. *)
  let current_workgroup_shared : Set.M(Tn).t ref = ref (Set.empty (module Tn))

  (* Marked local accumulator tiles and the one currently being rendered by a backend fragment
     scope. Outside such a scope they retain ordinary local-array semantics. *)
  let current_simdgroup_fragments : Set.M(Tn).t ref = ref (Set.empty (module Tn))
  let rendered_simdgroup_fragments : Set.M(Tn).t ref = ref (Set.empty (module Tn))

  (* Set by [compile_proc]: nodes stored XOR-swizzled ([Schedule.Stage ~swizzle], see
     {!Low_level.optimized.swizzled}). Scalar element accesses go through [pp_tn_offset] below;
     renderings that assume row-major storage (contiguous vector loads/stores, [Tile_mma]
     intrinsic / register-tiled / fragment paths) must decline these nodes. *)
  let current_swizzled : Set.M(Tn).t ref = ref (Set.empty (module Tn))
  let is_swizzled tn = Set.mem !current_swizzled tn

  (* The flat-offset rendering for an element access of [tn]'s buffer: row-major
     [pp_array_offset], except for swizzled nodes, where the minor-axis index is XORed with the low
     bits of the linearized row prefix — [P*C + col] becomes [P*C + (col ^ (P & (C-1)))], [C] the
     (power-of-two, checked by [Schedule.Stage]) minor dim. A bijection per row, so all-element
     traversals (zeroing) and matched read/write pairs are unaffected; same-column accesses from
     consecutive rows land in distinct shared-memory banks. The prefix expression is emitted twice;
     downstream C compilers CSE it. *)
  let pp_tn_offset tn (idcs, dims) =
    let open PPrint in
    let n = Array.length idcs in
    if (not (is_swizzled tn)) || n < 2 then pp_array_offset (idcs, dims)
    else
      let c = dims.(n - 1) in
      let prefix_doc =
        pp_array_offset (Array.sub idcs ~pos:0 ~len:(n - 1), Array.sub dims ~pos:0 ~len:(n - 1))
      in
      let col_doc = pp_axis_index idcs.(n - 1) in
      let col_doc = if PPrint.is_empty col_doc then string "0" else col_doc in
      if PPrint.is_empty prefix_doc then col_doc
      else
        parens prefix_doc
        ^^ string (" * " ^ Int.to_string c ^ " + ")
        ^^ parens
             (parens col_doc ^^ string " ^ "
             ^^ parens (parens prefix_doc ^^ string (" & " ^ Int.to_string (c - 1))))

  type active_mma_accumulator =
    | Active_fragment of Tn.t * string
    | Active_target of
        Tn.t * (PPrint.document * int * [ `Device | `Shared | `Thread | `Fragment of string ])

  let active_mma_accumulator : active_mma_accumulator option ref = ref None

  (* Set by [compile_proc]: outermost [Grid] loops eligible for pool-backed parallel rendering
     (docs/proposals/gh-ocannl-164.md), identified by index symbol. Empty unless
     [B.parallel_grid_syntax] renders in parallel. *)
  let current_parallel_grid : Set.M(Indexing.Symbol).t ref =
    ref (Set.empty (module Indexing.Symbol))

  (* Set by [compile_proc] alongside [current_parallel_grid]: per eligible [Grid] loop, the
     [Local]-placement tnodes privatized to per-chunk block-scope declarations inside that loop's
     chunk body (gh-ocannl-469: in-kernel packed operand tiles under a pool-parallel Grid). These
     are excluded from [compile_proc]'s function-scope [local_decls]. *)
  let current_grid_private : Tn.t list Map.M(Indexing.Symbol).t ref =
    ref (Map.empty (module Indexing.Symbol))

  (* Set by [compile_proc] alongside [current_parallel_grid]: [Local]-placement tnodes kept at
     function scope but accessed inside a pool-parallel Grid loop rendered as a blocks-extension
     [dispatch_apply] ([`Dispatch]). Blocks cannot refer to a declaration with an array type, but
     they capture pointers by value, so [local_decls] declares these behind a [const] pointer alias.
     Always empty for [`Openmp]/[`None]. *)
  let current_local_ptr_alias : Set.M(Tn).t ref = ref (Set.empty (module Tn))

  (* Set by [compile_proc] before any analysis or rendering: the kernel name, for the decline
     diagnostics ([log_declines]) and the [mma_census]. *)
  let current_kernel_name = ref ""

  let declinef fmt =
    Printf.ksprintf
      (fun s ->
        if Lazy.force log_declines then
          Stdlib.Printf.eprintf "declined: %s: %s\n%!" !current_kernel_name s)
      fmt

  (* Per-chunk private tiles live on the pool workers' stacks; libdispatch workers get 512KB by
     default, so cap their combined footprint per Grid loop (declining just keeps the loop serial).
     Config [cc_grid_private_bytes_cap] (gh-ocannl-474): raise it when the pool's worker
     stacks are known larger (e.g. under [OMP_STACKSIZE]) — for instance to let a grid-outermost
     packed GEMM privatize a whole B~ panel per chunk (gh-ocannl-475). *)
  let per_chunk_private_bytes_cap =
    lazy
      (match
         Int.of_string
            (String.strip
              (Utils.get_global_arg ~arg_name:"cc_grid_private_bytes_cap"
                 ~default:"262144"))
       with
      | n when n > 0 -> n
      | _ -> 256 * 1024
      | exception _ -> 256 * 1024)

  (* Shared traversal for the pool-parallel Grid analyses below: fires [access] for every
     tensor-node access event in program order (a [Set]'s right-hand side fires before its write).
     [Tile_mma] is traversed through its scalar [fallback]: every rendering of the statement
     (intrinsics, register tiling, lane-0 fallback) touches exactly the fallback's tensors over the
     fallback's index ranges, so the fallback IS the statement's access footprint. [on_stmt]
     observes the statement-level events the locals analysis keys on (opaque statements, scope
     declarations). [kind] tells the affine queries how to interpret the index vector: [`Whole] for
     accesses whose cells are not statically known ([Zero_out]'s every-cell write, data-dependent
     [Set_dynamic]/[Get_dynamic] slots, fired with empty indices), [`Vec] for vectorized writes
     whose last component is the base of a minor-axis run. *)
  let iter_local_accesses ~access ~on_stmt (root : Low_level.t) : unit =
    let rec go (llc : Low_level.t) =
      match llc with
      | Low_level.Noop | Comment _ -> ()
      | Staged_compilation _ | Workgroup_barrier -> on_stmt `Opaque
      | Tile_mma { fallback; _ } -> go fallback
      | Seq (a, b) ->
          go a;
          go b
      | For_loop { body; _ } -> go body
      | If { cond = c, _; body } ->
          go_sc c;
          go body
      | Zero_out tn -> access ~write:true ~kind:`Whole tn [||]
      | Set { tn; idcs; llsc; _ } ->
          go_sc llsc;
          access ~write:true ~kind:`Exact tn idcs
      | Set_dynamic { tn; dyn_value = v, _; llsc; _ } ->
          go_sc v;
          go_sc llsc;
          access ~write:true ~kind:`Whole tn [||]
      | Set_from_vec { tn; idcs; arg = a, _; _ } ->
          go_sc a;
          access ~write:true ~kind:`Vec tn idcs
      | Set_local (id, llsc) ->
          on_stmt (`Set_local id.Low_level.scope_id);
          go_sc llsc
      | Declare_local { id; _ } -> on_stmt (`Declare_local id.Low_level.scope_id)
    and go_sc (llsc : Low_level.scalar_t) =
      match llsc with
      | Local_scope { id; body; _ } ->
          on_stmt (`Declare_local id.Low_level.scope_id);
          go body
      | Get_local _ -> ()
      | Get (tn, idcs) -> access ~write:false ~kind:`Exact tn idcs
      | Get_dynamic { tn; dyn_value = v, _; _ } ->
          access ~write:false ~kind:`Whole tn [||];
          go_sc v
      | Get_merge_buffer _ -> ()
      | Ternop (_, (a, _), (b, _), (c, _)) ->
          go_sc a;
          go_sc b;
          go_sc c
      | Binop (_, (a, _), (b, _)) ->
          go_sc a;
          go_sc b
      | Unop (_, (a, _)) -> go_sc a
      | Constant _ | Constant_bits _ | Embed_index _ -> ()
    in
    go root

  (* Per-local access info under a candidate pool-parallel Grid loop's body. *)
  type grid_local_info = {
    gl_tn : Tn.t;
    mutable gl_written : bool;
    mutable gl_accs : (bool * [ `Exact | `Whole | `Vec ] * Indexing.axis_index array) list;
    mutable gl_count : int;
  }

  (* The privatization rule's write-dominance check: whether [tn]'s FIRST access (program order)
     under [body] is a standalone covering write — a [Zero_out], or an unguarded [Set] nest that
     rewrites the whole array and COMPLETES before any other access to [tn] executes. The completion
     requirement is structural (Codex P2 on PR #159): the covering [Set]'s per-axis coverage may
     only use loops that enclose NO other access to [tn] — descending from the body root, a loop
     stays usable while every access to [tn] in scope sits inside one child statement, and once
     siblings share the level (e.g. the pack nest followed by its consumer), the loops collected so
     far are discarded and coverage must come from the write's own nest. A write like [for x {
     tmp[x] = ..; use tmp[y] }] therefore declines: its only coverage loop [x] also interleaves the
     reads, which under per-chunk storage would observe missing prior iterations at chunk
     boundaries. Coverage per axis: a fresh usable-loop symbol of matching extent, or a mixed-radix
     affine combination of such symbols, each symbol used once across the index vector — the nest
     then enumerates every cell. Extra enclosing usable loops merely repeat the covering nest
     (required non-degenerate, or the write never executes). *)
  let first_access_standalone_covering (tn : Tn.t) (body : Low_level.t) : bool =
    let count_accesses (llc : Low_level.t) =
      let c = ref 0 in
      iter_local_accesses llc
        ~access:(fun ~write:_ ~kind:_ tn2 _ -> if tn2.Tn.uid = tn.Tn.uid then Int.incr c)
        ~on_stmt:(fun _ -> ());
      !c
    in
    let touches llc = count_accesses llc > 0 in
    let dims = Lazy.force tn.Tn.dims in
    (* Coverage by affine query ({!Affine.covers_box}): the nest enumerates every cell exactly
       once. Only the usable [loops] participate as the box environment — coverage through a loop
       that interleaves other accesses to [tn] must not count (see the completion requirement
       above). *)
    let covering ~loops (idcs : Indexing.axis_index array) =
      let range s =
        List.find_map loops ~f:(fun (s', from_, to_) ->
            if Indexing.equal_symbol s s' then Some (from_, to_) else None)
      in
      Affine.covers_box ~range ~dims idcs
    in
    let rec stmt (llc : Low_level.t) ~loops =
      match llc with
      | Low_level.Seq _ -> level (Low_level.flat_lines [ llc ]) ~loops
      | For_loop { index; from_; to_; body; _ } ->
          to_ >= from_ && level (Low_level.flat_lines [ body ]) ~loops:((index, from_, to_) :: loops)
      | Zero_out tn2 -> tn2.Tn.uid = tn.Tn.uid
      | Set { tn = tn2; idcs; _ } ->
          (* [count_accesses = 1]: the write itself, so the right-hand side does not read [tn] (a
             read-modify-write is not a covering first access). *)
          tn2.Tn.uid = tn.Tn.uid && count_accesses llc = 1 && covering ~loops idcs
      (* [If] = guarded (partial coverage); dynamic/vector writes and [Tile_mma] operand traffic are
         conservatively never covering. *)
      | Noop | Comment _ | Staged_compilation _ | Workgroup_barrier | If _ | Set_dynamic _
      | Set_from_vec _ | Set_local _ | Declare_local _ | Tile_mma _ ->
          false
    and level stmts ~loops =
      match List.filter stmts ~f:touches with
      | [] -> false
      | [ only ] -> stmt only ~loops
      | first :: _ :: _ ->
          (* Later siblings access [tn] too: they run after [first] completes, but the loops
             collected so far re-run them interleaved with [first] — coverage must come from
             [first]'s own nest. *)
          stmt first ~loops:[]
    in
    level (Low_level.flat_lines [ body ]) ~loops:[]

  (* Whether the body of an outermost [Grid] loop over [sym] tolerates its iterations being
     partitioned into chunks that execute on parallel CPU threads. Materialized accesses are already
     safe: [validate_parallel] requires every materialized write to cover [sym], so chunks write
     disjoint elements, and nests execute as separate parallel loops with a join in between
     (stronger than the GPU's single launch). The hazards are the stack arrays of [Local]-placement
     nodes: on GPU each thread gets a private copy, so a GPU-valid kernel may legally write them
     grid-invariantly (identical values per iteration) -- under one shared function-scope array that
     is a data race. A local written under the loop must satisfy one of:

     - Shared rule: every access to it (read or write) mentions [sym], and all accesses agree on
     every index component that mentions [sym] -- the same agreement rule as the default annotator's
     hazard analysis (mere mention is not enough, e.g. a stencil write [tmp[i]] + read [tmp[i-1]]
     both mention [sym] but reach across iterations). Distinct iterations then touch disjoint cells
     of one function-scope array. - Privatization rule: ALL of the node's accesses in the kernel sit
     inside this loop's body, and its first access per iteration is a standalone covering write -- a
     whole-array rewrite that completes before any other access to it executes (the write-dominance
     check of [first_access_standalone_covering]) -- so no value flows between iterations and each
     chunk can own a block-scope copy. This is what in-kernel packing [Stage] tiles satisfy
     (gh-ocannl-469): the pack nest fully rewrites the tile before the micro-kernel reads it. The
     combined per-chunk footprint is capped by [per_chunk_private_bytes_cap] (pool worker stacks).

     Under [`Dispatch] (the blocks extension), a block cannot refer to a declaration with an array
     type at all -- even read-only -- but it captures pointers by value, so every non-privatized
     local accessed under the loop is recorded for a pointer-alias declaration
     ([current_local_ptr_alias]). [Set_local] scope locals must have their declaration within the
     loop body (block scope = per-chunk storage). Opaque statements and barriers disqualify.

     Returns [Some (privatized, ptr_aliased)] when the loop can render in parallel. *)
  let parallel_grid_safe ~sym ~grid_range ~(global_counts : int Hashtbl.M(Int).t)
      (body : Low_level.t) : (Tn.t list * Tn.t list) option =
    let plc = placements () in
    let is_local tn =
      (not (Tn.Placements.is_virtual_force plc tn 431))
      && not (Tn.Placements.is_materialized_force plc tn 432)
    in
    let mentions_comp (idx : Indexing.axis_index) =
      match idx with
      | Indexing.Iterator s -> Indexing.equal_symbol s sym
      | Indexing.Affine { symbols; _ } ->
          List.exists symbols ~f:(fun (_, s) -> Indexing.equal_symbol s sym)
      | Indexing.Fixed_idx _ | Indexing.Sub_axis | Indexing.Concat _ -> false
    in
    let loop_ident = Indexing.symbol_ident sym in
    let locals : grid_local_info Hashtbl.M(Int).t = Hashtbl.create (module Int) in
    let declared_scopes = Hash_set.create (module Int) in
    let ok = ref true in
    let opaque = ref false in
    let escaped_scope = ref false in
    let access ~write ~kind tn idcs =
      if is_local tn then (
        let info =
          Hashtbl.find_or_add locals tn.Tn.uid ~default:(fun () ->
              { gl_tn = tn; gl_written = false; gl_accs = []; gl_count = 0 })
        in
        info.gl_count <- info.gl_count + 1;
        info.gl_written <- info.gl_written || write;
        info.gl_accs <- (write, kind, idcs) :: info.gl_accs)
    in
    iter_local_accesses body ~access ~on_stmt:(function
      | `Opaque ->
          ok := false;
          opaque := true
      | `Declare_local id -> Hash_set.add declared_scopes id
      | `Set_local id ->
          if not (Hash_set.mem declared_scopes id) then (
            ok := false;
            escaped_scope := true));
    if not !ok then (
      if !opaque then
        declinef "Grid loop %s stays serial: opaque statement or barrier under the loop body"
          loop_ident;
      if !escaped_scope then
        declinef
          "Grid loop %s stays serial: a scope local is set under the loop but declared outside it \
           (function-scope storage would race across chunks)"
          loop_ident;
      None)
    else
      let privatized = ref [] and ptr_aliased = ref [] in
      let private_bytes = ref 0 in
      (* The box environment for the affine conflict query: the Grid loop itself plus every loop
         bound under its body. Loops enclosing the Grid loop are shared across chunks (chunks of
         one dispatch execute under the same outer-iteration values, with a join before the next),
         which is exactly the query's treatment of unlisted symbols. *)
      let env = (sym, grid_range) :: Low_level.loop_bounds body in
      let range s = List.Assoc.find env s ~equal:Indexing.equal_symbol in
      let dup s = List.Assoc.mem env s ~equal:Indexing.equal_symbol in
      let feasible =
        Hashtbl.for_all locals ~f:(fun info ->
            (* The legacy procedural shared rule (every access mentions [sym] and all accesses
               agree on every mentioning component), kept for [legality_crosscheck]. *)
            let procedural_shared_ok () =
              (not info.gl_written)
              ||
              let maps = List.map info.gl_accs ~f:(fun (_, _, m) -> m) in
              List.for_all maps ~f:(fun idcs -> Array.exists idcs ~f:mentions_comp)
              &&
              let rank = List.fold maps ~init:0 ~f:(fun m a -> max m (Array.length a)) in
              let agree = ref true in
              for p = 0 to rank - 1 do
                let comps =
                  List.map maps ~f:(fun a ->
                      if p < Array.length a then a.(p) else Indexing.Fixed_idx 0)
                in
                if List.exists comps ~f:mentions_comp then
                  match comps with
                  | [] -> ()
                  | c0 :: rest ->
                      if not (List.for_all rest ~f:(Indexing.equal_axis_index c0)) then
                        agree := false
              done;
              !agree
            in
            (* Shared rule by affine query: chunks share one function-scope array, so every access
               pair involving a write must have its conflicts confined to a single chunk of [sym]
               ([Same_thread]) or be disjoint outright — which also admits patterns the agreement
               rule could only decline (constant-offset or strided-disjoint cells). *)
            let shared_ok =
              (not info.gl_written)
              ||
              let interp (kind, idcs) =
                match kind with
                | `Whole -> None
                | `Exact -> Some idcs
                | `Vec ->
                    if Array.is_empty idcs then None
                    else (
                      let m = Array.copy idcs in
                      m.(Array.length m - 1) <- Indexing.Sub_axis;
                      Some m)
              in
              let witness = ref "" in
              let q =
                List.for_all info.gl_accs ~f:(fun (wx, kx, mx) ->
                    List.for_all info.gl_accs ~f:(fun (wy, ky, my) ->
                        (not (wx || wy))
                        ||
                        match (interp (kx, mx), interp (ky, my)) with
                        | Some l, Some r -> (
                            match
                              Affine.pair_conflict ~range ~dup_left:dup ~dup_right:dup
                                ~pairs:[ (sym, sym) ] ~left:l ~right:r
                            with
                            | Affine.Disjoint | Affine.Same_thread -> true
                            | Affine.Cross_thread wit ->
                                witness := wit;
                                false)
                        | _ ->
                            witness := "statically unknown cells (whole-node or dynamic access)";
                            false))
              in
              Affine.crosscheck ~site:"cc pool-parallel shared rule"
                ~context:(Tn.debug_name info.gl_tn ^ " under Grid loop " ^ loop_ident)
                ~procedural_safe:procedural_shared_ok ~query_safe:q ~witness:!witness;
              q
            in
            if shared_ok then (
              (match B.parallel_grid_syntax with
              | `Dispatch -> ptr_aliased := info.gl_tn :: !ptr_aliased
              | `Openmp | `None -> ());
              true)
            else
              let all_inside =
                match Hashtbl.find global_counts info.gl_tn.Tn.uid with
                | Some total -> total = info.gl_count
                | None -> false
              in
              if not all_inside then (
                declinef
                  "Grid loop %s stays serial: local %s is written grid-variantly (fails the shared \
                   rule) and is also accessed outside the loop body (fails the privatization rule)"
                  loop_ident (Tn.debug_name info.gl_tn);
                false)
              else if not (first_access_standalone_covering info.gl_tn body) then (
                declinef
                  "Grid loop %s stays serial: local %s fails the shared rule, and its first access \
                   per iteration is not a standalone covering write (fails the privatization rule)"
                  loop_ident (Tn.debug_name info.gl_tn);
                false)
              else (
                private_bytes :=
                  !private_bytes
                  + (Tn.num_elems info.gl_tn * Ops.prec_in_bytes (Lazy.force info.gl_tn.Tn.prec));
                privatized := info.gl_tn :: !privatized;
                let fits = !private_bytes <= Lazy.force per_chunk_private_bytes_cap in
                if not fits then
                  declinef
                    "Grid loop %s stays serial: privatizing local %s brings the combined per-chunk \
                     tile footprint to %d bytes, over cc_grid_private_bytes_cap = %d"
                    loop_ident (Tn.debug_name info.gl_tn) !private_bytes
                    (Lazy.force per_chunk_private_bytes_cap);
                fits))
      in
      if feasible then Some (!privatized, !ptr_aliased) else None

  (* The outermost [Grid] loops safe to render in parallel, with each loop's privatized locals and
     (under [`Dispatch]) the pointer-aliased function-scope locals. Nested [Grid] loops render
     serially inside a chunk (still correct: write coverage holds per grid index). Runtime kernel
     logging writes to a shared FILE, so parallel rendering is skipped under
     [debug_log_from_routines]. *)
  let collect_parallel_grid (llc : Low_level.t) :
      Set.M(Indexing.Symbol).t * Tn.t list Map.M(Indexing.Symbol).t * Set.M(Tn).t =
    let empty =
      (Set.empty (module Indexing.Symbol), Map.empty (module Indexing.Symbol), Set.empty (module Tn))
    in
    if
      Poly.equal B.parallel_grid_syntax `None
      || B.parallel_grid_chunks <= 1 || Utils.debug_log_from_routines ()
    then empty
    else
      let plc = placements () in
      let is_local tn =
        (not (Tn.Placements.is_virtual_force plc tn 431))
        && not (Tn.Placements.is_materialized_force plc tn 432)
      in
      (* Whole-kernel access counts per local, so [parallel_grid_safe] can tell when a local's
         accesses all sit inside one loop's body (the privatization rule). Same traversal, so the
         counts are comparable by construction. *)
      let global_counts : int Hashtbl.M(Int).t = Hashtbl.create (module Int) in
      iter_local_accesses llc
        ~access:(fun ~write:_ ~kind:_ tn _ -> if is_local tn then Hashtbl.incr global_counts tn.Tn.uid)
        ~on_stmt:(fun _ -> ());
      let syms = ref (Set.empty (module Indexing.Symbol)) in
      let privs = ref (Map.empty (module Indexing.Symbol)) in
      let aliases = ref (Set.empty (module Tn)) in
      let rec go (llc : Low_level.t) =
        match llc with
        | Low_level.For_loop { axis = Grid; index; from_; to_; body; _ } -> (
            if from_ = 0 && to_ >= 1 then
              match parallel_grid_safe ~sym:index ~grid_range:(from_, to_) ~global_counts body with
              | Some (privatized, ptr_aliased) ->
                  syms := Set.add !syms index;
                  if not (List.is_empty privatized) then
                    privs := Map.set !privs ~key:index ~data:privatized;
                  aliases := List.fold ptr_aliased ~init:!aliases ~f:Set.add
              | None -> ())
        | For_loop { body; _ } -> go body
        | If { body; _ } -> go body
        | Seq (a, b) ->
            go a;
            go b
        | _ -> ()
      in
      go llc;
      (!syms, !privs, !aliases)

  (* Renders a [Local]-placement (routine-scope scratch) array declaration; shared by
     [compile_proc]'s function-scope [local_decls] and the per-chunk declarations of pool-parallel
     [Grid] loops. With [alias_ptr], the array gets a mangled name and a [const] pointer alias under
     the node's ident: [`Dispatch] blocks cannot refer to array declarations but capture pointers by
     value, and array indexing syntax is unchanged through the pointer. *)
  let local_array_decl ?(alias_ptr = false) ~zero_init (tn : Tn.t) : PPrint.document =
    let open PPrint in
    let typ = B.typ_of_prec @@ Lazy.force tn.Tn.prec in
    let ident = get_ident tn in
    let arr_name = if alias_ptr then ident ^ "_mem__" else ident in
    let align_doc =
      (* SIMD alignment for plain stack arrays (gh-ocannl-164). *)
      match B.aligned_local_attr with
      | Some attr -> string (" " ^ attr)
      | None -> empty
    in
    let init_doc = if zero_init then string " = {0}" else empty in
    let decl =
      string typ ^^ space ^^ string arr_name
      ^^ brackets (OCaml.int (Tn.num_elems tn))
      ^^ align_doc ^^ init_doc ^^ semi
    in
    if alias_ptr then
      decl ^^ hardline ^^ string (Printf.sprintf "%s * const %s = %s;" typ ident arr_name)
    else decl

  (* A [Zero_out] loop is redundant when the array's declaration already initializes it with [=
     {0}]. That happens for local (non-virtual, non-materialized) declarations whose traced node has
     [zero_initialized_by_code = true]; see [compile_proc]'s [local_decls]. Materialized (on-device)
     nodes do NOT get [= {0}] (allocation handles zeroing, and is skipped exactly when
     [zero_initialized_by_code] is true), so their [Zero_out] loop must be kept. *)
  let zero_out_loop_redundant tn =
    match !current_traced_store with
    | None -> false
    | Some traced_store -> (
        match Hashtbl.find traced_store tn with
        | Some node ->
            let plc = placements () in
            node.Low_level.zero_initialized_by_code
            && (not
                  (Tn.Placements.is_virtual_force plc tn 337
                  || Tn.Placements.is_materialized_force plc tn 338))
            && not (Set.mem !current_workgroup_shared tn)
        | None -> false)

  (* Tensor node ids whose [Zero_out] has already been encountered during the current [pp_ll]
     traversal. Only the *first-touch* [Zero_out tn] is made redundant by the declaration's [= {0}];
     any later [Zero_out tn] (e.g. in [Zero_out tn; Set tn; Zero_out tn], or any [Zero_out] reached
     inside a loop body) is a genuine re-zero and must still emit its loop. Cleared per
     [compile_proc]. *)
  let zero_out_seen : int Hash_set.t = Hash_set.create (module Int)

  (* Symbols of the serial [for] loops enclosing the current [pp_ll] rendering point (innermost
     first): maintained by [serial_loop] below, consulted by the [Set] case's [volatile_scalar_rmw]
     rule. *)
  let serial_loop_stack : Indexing.symbol list ref = ref []

  (* Recognize the exact scalar fallback region synthesized by
     [Schedule.contract_tensorized_accumulator]:

     lane { if lane==0 { fragment <- target } }; for k_o { ... Tile_mma(d=fragment) ... }; lane { if
     lane==0 { target <- fragment } }.

     The marker on [optimized] makes this structural, not a proof over arbitrary user IR. A backend
     hook may replace the region; declining leaves the ordinary local-array rendering untouched. *)
  let render_mma_fragment_scope ~render (c : Low_level.t) : PPrint.document option =
    let open Low_level in
    let open PPrint in
    let nonempty =
      List.filter (flat_lines [ c ]) ~f:(function Noop | Comment _ -> false | _ -> true)
    in
    let is_lane0_guard lane = function
      | Binop (Ops.Cmpeq, (Embed_index (Indexing.Iterator guard_lane), _), (Constant zero, _))
      | Binop (Ops.Cmpeq, (Constant zero, _), (Embed_index (Indexing.Iterator guard_lane), _)) ->
          Indexing.equal_symbol lane guard_lane && Float.equal zero 0.
      | _ -> false
    in
    let unwrap_transfer ~into_fragment = function
      | For_loop
          { index = lane; from_ = 0; to_; axis = Workgroup; body = If { cond = cond, _; body }; _ }
        when is_lane0_guard lane cond ->
          let rec descend syms = function
            | For_loop { index; axis = Serial; body; _ } -> descend (index :: syms) body
            | Set { tn = fragment; llsc = Get (target, target_idcs); _ } when into_fragment ->
                Some (lane, to_ + 1, fragment, target, target_idcs, syms)
            | Set { tn = target; llsc = Get (fragment, _); idcs = target_idcs; _ }
              when not into_fragment ->
                Some (lane, to_ + 1, fragment, target, target_idcs, syms)
            | _ -> None
          in
          descend [] body
      | _ -> None
    in
    let rec collect_fragment_tiles fragment acc = function
      | Tile_mma { d = d, _; _ } as tm when Tn.equal d fragment -> tm :: acc
      | Seq (a, b) -> collect_fragment_tiles fragment (collect_fragment_tiles fragment acc a) b
      | For_loop { body; _ } | If { body; _ } -> collect_fragment_tiles fragment acc body
      | _ -> acc
    in
    let zero_symbols syms idx =
      let bound s = List.mem syms s ~equal:Indexing.equal_symbol in
      match idx with
      | Indexing.Iterator s when bound s -> Indexing.Fixed_idx 0
      | Indexing.Affine { symbols; offset } -> (
          let symbols = List.filter symbols ~f:(fun (_, s) -> not (bound s)) in
          match (symbols, offset) with
          | [], k -> Indexing.Fixed_idx k
          | [ (1, s) ], 0 -> Indexing.Iterator s
          | _ -> Indexing.Affine { symbols; offset })
      | other -> other
    in
    let operand (tn, idcs) =
      let dims = Lazy.force tn.Tn.dims in
      let prec = Lazy.force tn.Tn.prec in
      let rank = Array.length dims in
      let ld = if rank >= 1 then dims.(rank - 1) else 1 in
      let space =
        if Set.mem !current_workgroup_shared tn then `Shared
        else if Tn.Placements.is_materialized_force (placements ()) tn 441 then `Device
        else `Thread
      in
      let ptr = parens (string (get_ident tn) ^^ string " + " ^^ pp_array_offset (idcs, dims)) in
      (prec, (ptr, ld, space))
    in
    match nonempty with
    | init :: reduction :: store :: rest -> (
        (* [rest] is any statements following the marked region in the same body — in particular
           the lane-0 epilogue statement fused by [Schedule.Fuse_epilogue] (gh-ocannl-486). They
           render after the region's rendering: the backend hooks end with the visibility barrier
           that the epilogue's re-reads of the just-stored target need. *)
        match
          (unwrap_transfer ~into_fragment:true init, unwrap_transfer ~into_fragment:false store)
        with
        | ( Some (lane1, width1, fragment, target, init_idcs, init_syms),
            Some (lane2, width2, fragment2, target2, store_idcs, _) )
          when Set.mem !current_simdgroup_fragments fragment
               && Tn.equal fragment fragment2 && Tn.equal target target2 && width1 = width2
               && Indexing.equal_symbol lane1 lane2
               && Array.equal Indexing.equal_axis_index init_idcs store_idcs -> (
            match collect_fragment_tiles fragment [] reduction with
            (* Swizzled operands decline the fragment hooks (row-major pointer+stride contract);
               falling through to [None] keeps the ordinary rendering, whose [Tile_mma]s decline
               to the swizzle-aware scalar fallback. *)
            | [ Tile_mma { a; b; ta; tb; m; n; k; _ } ]
              when not (List.exists [ target; fst a; fst b ] ~f:is_swizzled) -> (
                let target_base = Array.map init_idcs ~f:(zero_symbols init_syms) in
                let d_prec, target_op = operand (target, target_base) in
                let a_prec, a_op = operand a in
                let b_prec, b_op = operand b in
                let fragment_name = Printf.sprintf "__mma_fragment_%d" fragment.Tn.uid in
                let render_with active =
                  let old = !active_mma_accumulator in
                  active_mma_accumulator := Some active;
                  Exn.protect
                    ~f:(fun () -> render reduction)
                    ~finally:(fun () -> active_mma_accumulator := old)
                in
                let fragment_doc =
                  if Utils.debug_log_from_routines () then None
                  else
                    Option.bind B.mma_fragment_syntax ~f:(fun emit ->
                        emit ~d_prec ~a_prec ~b_prec ~m ~n ~k ~fragment:fragment_name
                          ~target:target_op ~a:a_op ~b:b_op ~body:(fun () ->
                            render_with (Active_fragment (fragment, fragment_name))))
                in
                let rendered =
                  match fragment_doc with
                  | Some _ as doc -> doc
                  | None when Utils.debug_log_from_routines () -> None
                  | None ->
                      Option.bind B.mma_syntax ~f:(fun emit ->
                          (* When the fragment hook declined this call (HIP until its
                             persistent-fragment mapping lands, CUDA's fp8 combination, unsupported
                             precisions), preserve the existing per-[k_o] intrinsic path by aliasing
                             the marked local back to the original target when that exact MMA call
                             is supported. Unsupported calls retain the explicit lane-0 local-array
                             fallback. *)
                          match
                            emit ~d_prec ~a_prec ~b_prec ~ta ~tb ~m ~n ~k ~d:target_op ~a:a_op
                              ~b:b_op
                          with
                          | Some _ -> Some (render_with (Active_target (fragment, target_op)))
                          | None -> None)
                in
                match rendered with
                | Some doc ->
                    rendered_simdgroup_fragments := Set.add !rendered_simdgroup_fragments fragment;
                    Some (separate hardline (doc :: List.map rest ~f:render))
                | None ->
                    (* Scalar/local fallback: lane 0 performs the synthesized transfers. Hardware
                       backends need the same outer visibility barriers as the ordinary Tile_mma
                       load/store path, so the init observes sibling zeroing and later statements
                       observe the final store. Serial C renderers need no barriers. *)
                    let body_doc =
                      separate hardline
                        (List.map (init :: reduction :: store :: rest) ~f:render)
                    in
                    Some
                      (match B.barrier_syntax with
                      | Some barrier ->
                          string barrier ^^ hardline ^^ body_doc ^^ hardline ^^ string barrier
                      | None -> body_doc))
            | _ -> None)
        | _ -> None)
    | _ -> None

  let try_mma_fragment_scope ~render c =
    if Set.is_empty !current_simdgroup_fragments then None else render_mma_fragment_scope ~render c

  let rec pp_ll ?(log_set_locals = true) ?(in_loop = false) (c : Low_level.t) : PPrint.document =
    let open PPrint in
    match c with
    | Low_level.Noop -> empty
    | Seq (c1, c2) -> (
        match
          try_mma_fragment_scope ~render:(fun body -> pp_ll ~log_set_locals ~in_loop body) c
        with
        | Some doc -> doc
        | None ->
            let d1 = pp_ll ~log_set_locals ~in_loop c1 in
            let d2 = pp_ll ~log_set_locals ~in_loop c2 in
            (* Avoid extra hardlines if one side is empty *)
            if PPrint.is_empty d1 then d2
            else if PPrint.is_empty d2 then d1
            else d1 ^^ hardline ^^ d2)
    | For_loop { index = i; from_; to_; body; trace_it = _; axis } -> (
        (* Rendering phase of docs/proposals/axis-types-for-loops.md (§5): [Serial] loops render as
           C [for] statements; [Grid]/[Workgroup]/[Workgroup_reduce] loops bind their index to the
           backend's hardware register (at the signed [loop_index_type] width, with an explicit cast
           from the unsigned register) when [B.hardware_index] provides one, and fall back to a
           serial loop otherwise (legal absent barriers); [Vectorized] loops render serially,
           prefixed with [B.vectorize_pragma] when non-empty; [Unrolled] loops emit the repeated
           body with the index bound as a per-block constant. *)
        let body_doc ?(body = body) () =
          let doc = ref (pp_ll ~log_set_locals ~in_loop:true body) in
          (if Utils.debug_log_from_routines () then
             let log_doc =
               let base_message = Printf.sprintf "index %s = %%d\n" (symbol_ident i) in
               let log_param_doc =
                 Option.map B.kernel_log_param ~f:(fun (_, name) -> string name)
               in
               B.pp_log_statement ~log_param_c_expr_doc:log_param_doc
                 ~base_message_literal:base_message
                 ~args_docs:[ pp_symbol i ]
             in
             doc := log_doc ^^ hardline ^^ !doc);
          !doc
        in
        let serial_loop () =
          (* gh-490 guard-fusion peephole: a body-wrapping symbolic-extent guard [if (i < s)] (with
             [s] a kernel parameter, not an enclosing loop index) hoists into the loop header as
             [i <= to_ && i < s]. The iteration variable is monotone, so once the guard fails it
             stays false: exiting the loop is equivalent to skipping the remaining iterations. *)
          let fused =
            match body with
            | If
                {
                  cond =
                    ( Binop
                        ( Ops.Cmplt,
                          (Embed_index (Indexing.Iterator i'), _),
                          (Embed_index (Indexing.Iterator s), _) ),
                      _ );
                  body = inner;
                }
              when Indexing.equal_symbol i' i
                   && not (List.mem !serial_loop_stack s ~equal:Indexing.equal_symbol) ->
                Some (s, inner)
            | _ -> None
          in
          let guard_doc =
            match fused with
            | None -> empty
            | Some (s, _) -> string " && " ^^ pp_symbol i ^^ string " < " ^^ pp_symbol s
          in
          let header =
            string ("for (" ^ B.loop_index_type)
            ^^ pp_symbol i ^^ string " = " ^^ PPrint.OCaml.int from_ ^^ semi ^^ space ^^ pp_symbol i
            ^^ string " <= " ^^ PPrint.OCaml.int to_ ^^ guard_doc ^^ semi ^^ space ^^ string "++"
            ^^ pp_symbol i ^^ string ")"
          in
          serial_loop_stack := i :: !serial_loop_stack;
          let body =
            Exn.protect
              ~f:(fun () -> body_doc ?body:(Option.map fused ~f:snd) ())
              ~finally:(fun () -> serial_loop_stack := List.tl_exn !serial_loop_stack)
          in
          group (header ^^ space ^^ lbrace ^^ nest 2 (hardline ^^ body) ^^ hardline ^^ rbrace)
        in
        let hardware_binding kind =
          let slot =
            match
              List.find !current_hardware_axes ~f:(fun a ->
                  Indexing.equal_symbol a.Low_level.ha_index i)
            with
            | Some a -> a.Low_level.ha_slot
            | None ->
                invalid_arg
                  ("C_syntax.pp_ll: hardware-annotated loop " ^ symbol_ident i
                 ^ " missing from the slot table (pp_ll called outside compile_proc?)")
          in
          match B.hardware_index ~kind ~slot with
          | None -> serial_loop ()
          | Some reg ->
              let cast = "(" ^ String.strip B.loop_index_type ^ ")" in
              let binding =
                string ("const " ^ B.loop_index_type)
                ^^ pp_symbol i
                ^^ string (" = " ^ cast ^ reg ^ ";")
              in
              group
                (lbrace
                ^^ nest 2 (hardline ^^ binding ^^ hardline ^^ body_doc ())
                ^^ hardline ^^ rbrace)
        in
        let parallel_grid_loop () =
          (* Pool-backed Grid rendering (gh-ocannl-164): contiguous chunks of the grid extent
             execute on the process-global native pool ([dispatch_apply] / OpenMP); [Workgroup]
             loops and nested [Grid] loops stay serial inside a chunk. Eligibility (including [from_
             = 0]) was established by [collect_parallel_grid]. *)
          let extent = to_ + 1 in
          let target = min B.parallel_grid_chunks extent in
          let grain = (extent + target - 1) / target in
          let nchunks = (extent + grain - 1) / grain in
          let it = B.loop_index_type in
          let ident = symbol_ident i in
          let chunk = ident ^ "_chunk" and lo = ident ^ "_lo" and hi = ident ^ "_hi" in
          let decls =
            string
              (Printf.sprintf "const %s%s = (%s)(%s * %d);" it lo (String.strip it) chunk grain)
            ^^ hardline
            ^^ string
                 (Printf.sprintf "const %s%s = %s + %d <= %d ? %s + %d : %d;" it hi lo grain extent
                    lo grain extent)
          in
          (* Locals privatized to this loop (see [parallel_grid_safe]): block-scope arrays inside
             the chunk body, one copy per chunk -- iterations rewrite them wholly before reading, so
             per-chunk storage matches the serial semantics. *)
          let decls =
            match Map.find !current_grid_private i with
            | None | Some [] -> decls
            | Some tns ->
                let zero_init tn =
                  match !current_traced_store with
                  | Some ts ->
                      Hashtbl.find ts tn
                      |> Option.value_map ~default:false ~f:(fun node ->
                          node.Low_level.zero_initialized_by_code)
                  | None -> false
                in
                List.fold tns ~init:decls ~f:(fun acc tn ->
                    acc ^^ hardline ^^ local_array_decl ~zero_init:(zero_init tn) tn)
          in
          let inner =
            string (Printf.sprintf "for (%s%s = %s; %s < %s; ++%s)" it ident lo ident hi ident)
            ^^ space ^^ lbrace
            ^^ nest 2 (hardline ^^ body_doc ())
            ^^ hardline ^^ rbrace
          in
          let comment =
            string
              (Printf.sprintf "/* Pool-backed Grid rendering: %d chunks of up to %d. */" nchunks
                 grain)
          in
          match B.parallel_grid_syntax with
          | `Dispatch ->
              comment ^^ hardline
              ^^ string
                   (Printf.sprintf "dispatch_apply((size_t)%d, DISPATCH_APPLY_AUTO, ^(size_t %s) {"
                      nchunks chunk)
              ^^ nest 2 (hardline ^^ decls ^^ hardline ^^ inner)
              ^^ hardline ^^ string "});"
          | `Openmp ->
              comment ^^ hardline
              ^^ string "#pragma omp parallel for schedule(static)"
              ^^ hardline
              ^^ string
                   (Printf.sprintf "for (%s%s = 0; %s < %d; ++%s)" it chunk chunk nchunks chunk)
              ^^ space ^^ lbrace
              ^^ nest 2 (hardline ^^ decls ^^ hardline ^^ inner)
              ^^ hardline ^^ rbrace
          | `None -> assert false
        in
        (* --- Shared analysis for the explicit-SIMD ([Vectorized]) and warp-shuffle
           ([Workgroup_reduce]) renderings below. --- *)
        let mentions_comp (idx : Indexing.axis_index) =
          match idx with
          | Indexing.Iterator s -> Indexing.equal_symbol s i
          | Indexing.Affine { symbols; _ } ->
              List.exists symbols ~f:(fun (_, s) -> Indexing.equal_symbol s i)
          | Indexing.Fixed_idx _ | Indexing.Sub_axis | Indexing.Concat _ -> false
        in
        let rec touches_tn tn (llsc : Low_level.scalar_t) =
          match llsc with
          | Low_level.Get (tn2, _) | Get_merge_buffer (tn2, _) -> Tn.equal tn tn2
          | Get_dynamic { tn = tn2; dyn_value = v, _; _ } -> Tn.equal tn tn2 || touches_tn tn v
          | Local_scope { body; _ } -> body_touches tn body
          | Get_local _ | Constant _ | Constant_bits _ | Embed_index _ -> false
          | Ternop (_, (a, _), (b, _), (c, _)) ->
              touches_tn tn a || touches_tn tn b || touches_tn tn c
          | Binop (_, (a, _), (b, _)) -> touches_tn tn a || touches_tn tn b
          | Unop (_, (a, _)) -> touches_tn tn a
        and body_touches tn (llc : Low_level.t) =
          match llc with
          | Low_level.Noop | Comment _ | Staged_compilation _ | Workgroup_barrier | Declare_local _
            ->
              false
          | Seq (a, b) -> body_touches tn a || body_touches tn b
          | For_loop { body; _ } -> body_touches tn body
          | If { cond = c, _; body } -> touches_tn tn c || body_touches tn body
          | Zero_out tn2 -> Tn.equal tn tn2
          | Set { tn = tn2; llsc; _ } -> Tn.equal tn tn2 || touches_tn tn llsc
          | Set_dynamic { tn = tn2; dyn_value = v, _; llsc; _ } ->
              Tn.equal tn tn2 || touches_tn tn v || touches_tn tn llsc
          | Set_from_vec { tn = tn2; arg = a, _; _ } -> Tn.equal tn tn2 || touches_tn tn a
          | Set_local (_, llsc) -> touches_tn tn llsc
          | Tile_mma { d = d_tn, _; a = a_tn, _; b = b_tn, _; _ } ->
              Tn.equal tn d_tn || Tn.equal tn a_tn || Tn.equal tn b_tn
        in
        let nonempty_stmts body =
          List.filter (Low_level.flat_lines [ body ]) ~f:(function
            | Low_level.Noop | Comment _ -> false
            | _ -> true)
        in
        (* A body that is a single accumulation statement [acc[idcs] = op(acc[idcs], contrib)] (or
           its FMA form [acc = FMA(a, b, acc)]) where [idcs] does not mention the loop index and
           [op] is an associative-commutative reduction — such a body IS the loop's serial meaning.
           Recognized by the warp-shuffle rendering of [Workgroup_reduce] loops (gh-ocannl-462) and
           by the SIMD reduction rendering of [Vectorized] loops (gh-ocannl-468). *)
        let recognize_accumulation stmts =
          match stmts with
          | [ Low_level.Set { tn; idcs; llsc; _ } ] when not (Array.exists idcs ~f:mentions_comp)
            -> (
              let is_acc s = Low_level.equal_scalar_t s (Low_level.Get (tn, idcs)) in
              let reduce_op = function
                | Ops.Add | Ops.Mul | Ops.Max | Ops.Min -> true
                | _ -> false
              in
              match llsc with
              | Binop (op, (a, _), (b, _)) when reduce_op op && is_acc a && not (touches_tn tn b) ->
                  Some (tn, idcs, op, b)
              | Binop (op, (a, _), (b, _)) when reduce_op op && is_acc b && not (touches_tn tn a) ->
                  Some (tn, idcs, op, a)
              | Ternop (Ops.FMA, (a, pa), (b, pb), (c, _))
                when is_acc c && (not (touches_tn tn a)) && not (touches_tn tn b) ->
                  Some (tn, idcs, Ops.Add, Low_level.Binop (Ops.Mul, (a, pa), (b, pb)))
              | _ -> None)
          | _ -> None
        in
        (* Eligibility bail-out of the explicit-SIMD renderings ([try_vectorize] /
           [try_vectorize_reduce]) back to the pragma/serial fallbacks. *)
        let exception Bail in
        let rec scalar_mentions (llsc : Low_level.scalar_t) =
          match llsc with
          | Low_level.Get (_, idcs) | Get_merge_buffer (_, idcs) ->
              Array.exists idcs ~f:mentions_comp
          | Get_dynamic { idcs; dyn_value = v, _; _ } ->
              Array.exists idcs ~f:mentions_comp || scalar_mentions v
          (* Scope-local bodies could bind or mention the index in statement position;
             conservatively ineligible. *)
          | Local_scope _ | Get_local _ -> raise Bail
          | Embed_index idx -> mentions_comp idx
          | Ternop (_, (a, _), (b, _), (c, _)) ->
              scalar_mentions a || scalar_mentions b || scalar_mentions c
          | Binop (_, (a, _), (b, _)) -> scalar_mentions a || scalar_mentions b
          | Unop (_, (a, _)) -> scalar_mentions a
          | Constant _ | Constant_bits _ -> false
        in
        let contiguous idcs =
          let n = Array.length idcs in
          n > 0
          && Array.for_alli idcs ~f:(fun p idx -> p = n - 1 || not (mentions_comp idx))
          &&
          match idcs.(n - 1) with
          | Indexing.Iterator s -> Indexing.equal_symbol s i
          | Indexing.Affine { symbols; _ } ->
              List.for_all symbols ~f:(fun (c, s) -> (not (Indexing.equal_symbol s i)) || c = 1)
              && List.count symbols ~f:(fun (_, s) -> Indexing.equal_symbol s i) = 1
          | _ -> false
        in
        let check_read ~written tn idcs =
          match Hashtbl.find written tn.Tn.uid with
          | Some w_idcs ->
              if not (Array.equal Indexing.equal_axis_index w_idcs idcs) then raise Bail
          | None -> ()
        in
        let rec no_written_reads ~written (llsc : Low_level.scalar_t) =
          match llsc with
          | Low_level.Get (tn, _) | Get_merge_buffer (tn, _) | Get_dynamic { tn; _ } ->
              if Hashtbl.mem written tn.Tn.uid then raise Bail
          | Local_scope _ | Get_local _ -> raise Bail
          | Embed_index _ | Constant _ | Constant_bits _ -> ()
          | Ternop (_, (a, _), (b, _), (c, _)) ->
              no_written_reads ~written a;
              no_written_reads ~written b;
              no_written_reads ~written c
          | Binop (_, (a, _), (b, _)) ->
              no_written_reads ~written a;
              no_written_reads ~written b
          | Unop (_, (a, _)) -> no_written_reads ~written a
        in
        let uniform_scalar ~written prec llsc =
          (* Uniform across lanes: a read of a stored node cannot equal its (index-mentioning) store
             vector, so reject those; then render as a plain scalar (vector-scalar arithmetic
             splats; in the packed style the scalar participates per lane). *)
          no_written_reads ~written llsc;
          let local_defs, sdoc = pp_scalar prec llsc in
          if not (List.is_empty local_defs) then raise Bail;
          parens sdoc
        in
        (* The [`Vec_extensions] expression renderer shared by the elementwise ([try_vectorize]) and
           reduction ([try_vectorize_reduce]) renderings. Emitted binding statements accumulate
           through [emit]; [written] maps written nodes to their store index vectors — every read of
           a written node must use that exact vector (vector semantics evaluates all lanes' loads
           before the store, so cross-lane flow would reorder against the serial loop). *)
        let vec_ext_machinery ~prec ~lanes ~vtyp ~written ~emit ~fresh =
          let vload tn idcs =
            (* A swizzled layout breaks row-major contiguity within a row. *)
            if is_swizzled tn then raise Bail;
            if not (contiguous idcs) then raise Bail;
            check_read ~written tn idcs;
            let name = fresh "vget" in
            let offset = pp_array_offset (idcs, Lazy.force tn.Tn.dims) in
            emit
              (string (vtyp ^ " " ^ name ^ ";")
              ^^ hardline
              ^^ string ("__builtin_memcpy(&" ^ name ^ ", &")
              ^^ string (get_ident tn)
              ^^ brackets offset
              ^^ string (", sizeof(" ^ name ^ "));"));
            string name
          in
          let rec vec_expr (llsc : Low_level.scalar_t) (p : Ops.prec) : PPrint.document =
            if not (scalar_mentions llsc) then uniform_scalar ~written prec llsc
            else if not (Ops.equal_prec p prec) then raise Bail
            else
              match llsc with
              | Low_level.Get (tn, idcs) ->
                  if not (Ops.equal_prec (Lazy.force tn.Tn.prec) prec) then raise Bail;
                  vload tn idcs
              | Binop (op, (a, pa), (b, pb)) ->
                  let inf =
                    match op with
                    | Ops.Add -> " + "
                    | Sub -> " - "
                    | Mul -> " * "
                    | Div -> " / "
                    | _ -> raise Bail
                  in
                  parens (vec_expr a pa ^^ string inf ^^ vec_expr b pb)
              | Ternop (Ops.FMA, (a, pa), (b, pb), (c, pc)) ->
                  (* Fused, matching the scalar path's [fmaf]/[fma] single rounding (the simplifier
                     synthesizes [FMA] from mul-add trees, so this is the hot case). A plain [a * b
                     + c] would be only maybe-contracted, so vector lanes could differ from the
                     serial remainder loop and twin. Operands bind to vector temps (lane-uniform
                     ones splat explicitly: vector = scalar init is invalid). *)
                  let bind llsc p =
                    let name = fresh "vfop" in
                    emit (string (vtyp ^ " " ^ name ^ " = ") ^^ vec_operand llsc p ^^ semi);
                    name
                  in
                  let na = bind a pa and nb = bind b pb and nc = bind c pc in
                  let nr = fresh "vfma" in
                  emit (string (vtyp ^ " " ^ nr ^ " = ") ^^ string nc ^^ semi);
                  emit (vec_acc_fma ~prec ~lanes ~dst:nr ~a:na ~b:nb);
                  string nr
              | Unop (Ops.Identity, (a, pa)) -> vec_expr a pa
              | Unop (Ops.Neg, (a, pa)) -> parens (string "-" ^^ vec_expr a pa)
              | _ -> raise Bail
          and vec_operand (llsc : Low_level.scalar_t) (p : Ops.prec) : PPrint.document =
            (* A vector-typed rendering even for lane-uniform values: initializers and builtin
               arguments need a vector, where the implicit vector-scalar splat of binary operators
               does not apply. *)
            if scalar_mentions llsc then vec_expr llsc p
            else string ("((" ^ vtyp ^ "){0} + ") ^^ uniform_scalar ~written prec llsc ^^ string ")"
          in
          (vec_expr, vec_operand)
        in
        (* Explicit SIMD rendering of a [Vectorized] loop via GCC/Clang vector extensions (portable
           across gcc/clang and AVX2/NEON; the [Vectorized]-codegen follow-up of gh-ocannl-164). The
           loop must start at 0 and its body must be a sequence of plain [Set] statements over one
           floating precision, with every access that mentions the loop index contiguous in it (the
           index appears only in the last component, with coefficient 1 — the flat offset then
           advances by exactly 1 per iteration). Index-free subexpressions render as scalars
           (vector-scalar arithmetic splats across lanes); vector subexpressions allow
           [Add]/[Sub]/[Mul]/[Div]/[Neg] and fused [FMA] (matching the scalar path's [fmaf]/[fma]
           rounding; see the note in [vec_expr]). At most one store per node, and every read of a
           stored node must use the store's exact index vector — vector semantics evaluates all
           lanes' loads before the store, so cross-lane flow would reorder against the serial loop.
           The main loop advances by [lanes]; a serial remainder loop reuses [body_doc]. Anything
           else falls back to [vectorize_pragma] / serial. *)
        let try_vectorize () : PPrint.document option =
          try
            if B.vector_bytes < 8 || from_ <> 0 || Utils.debug_log_from_routines () then raise Bail;
            let extent = to_ + 1 in
            let stmts = nonempty_stmts body in
            let sets =
              List.map stmts ~f:(function
                | Low_level.Set { tn; idcs; llsc; _ } -> (tn, idcs, llsc)
                | _ -> raise Bail)
            in
            if List.is_empty sets then raise Bail;
            let prec =
              let tn, _, _ = List.hd_exn sets in
              Lazy.force tn.Tn.prec
            in
            (match prec with Ops.Single_prec _ | Ops.Double_prec _ -> () | _ -> raise Bail);
            let lanes = B.vector_bytes / Ops.prec_in_bytes prec in
            if lanes < 2 || extent < lanes then raise Bail;
            let written = Hashtbl.create (module Int) in
            List.iter sets ~f:(fun (tn, idcs, _) ->
                if not (Ops.equal_prec (Lazy.force tn.Tn.prec) prec) then raise Bail;
                match Hashtbl.add written ~key:tn.Tn.uid ~data:idcs with
                | `Ok -> ()
                | `Duplicate -> raise Bail);
            let stmts_docs = ref [] in
            let emit d = stmts_docs := d :: !stmts_docs in
            let fresh =
              let ctr = ref 0 in
              fun pfx ->
                Int.incr ctr;
                Printf.sprintf "%s%d__" pfx !ctr
            in
            let prelude =
              match B.vector_style with
              | `Vec_extensions ->
                  let vtyp, typedef_doc = vec_ext_typ ~prec ~lanes in
                  let _vec_expr, vec_operand =
                    vec_ext_machinery ~prec ~lanes ~vtyp ~written ~emit ~fresh
                  in
                  List.iter sets ~f:(fun (tn, idcs, llsc) ->
                      if is_swizzled tn then raise Bail;
                      if not (contiguous idcs) then raise Bail;
                      let rhs = vec_operand llsc prec in
                      let vname = fresh "vset" in
                      emit (string (vtyp ^ " " ^ vname ^ " = ") ^^ rhs ^^ semi);
                      emit
                        (string "__builtin_memcpy(&"
                        ^^ string (get_ident tn)
                        ^^ brackets (pp_array_offset (idcs, Lazy.force tn.Tn.dims))
                        ^^ string (", &" ^ vname ^ ", sizeof(" ^ vname ^ "));")));
                  typedef_doc ^^ hardline
              | `Packed_struct ->
                  (* GPU 128-bit packed loads/stores (gh-ocannl-463; llm.c's Packed128): the
                     backend's aligned pack aggregate is loaded/stored via [reinterpret_cast] — one
                     128-bit memory transaction — while the arithmetic stays scalar in a per-lane
                     loop over the pack's [.v] payload (per-lane [fmaf]/[fma] keeps the serial
                     path's rounding). Sound only at provably lane-aligned offsets of
                     device-resident buffers, hence the extra eligibility checks. *)
                  let vtyp =
                    match B.vec_typ_of_prec ~length:lanes prec with
                    | s -> s
                    | exception _ -> raise Bail
                  in
                  (* The flat offset must stay a lane multiple whenever the loop index is one:
                     components before the last contribute stride multiples of [dims.(n - 1)], so
                     the last dimension must be a lane multiple (unless the access is 1-D), and the
                     last component's constant offset and non-index coefficients must be lane
                     multiples. Buffer bases and pool offsets are [Ops.buffer_alignment >= 16]
                     aligned, so lane-multiple element offsets are 16-byte-aligned addresses. *)
                  let lane_aligned tn idcs =
                    let dims = Lazy.force tn.Tn.dims in
                    let n = Array.length idcs in
                    n > 0
                    && (n = 1 || dims.(n - 1) % lanes = 0)
                    &&
                    match idcs.(n - 1) with
                    | Indexing.Iterator _ -> true
                    | Indexing.Affine { symbols; offset } ->
                        offset % lanes = 0
                        && List.for_all symbols ~f:(fun (c, s) ->
                            Indexing.equal_symbol s i || c % lanes = 0)
                    | Indexing.Fixed_idx _ | Indexing.Sub_axis | Indexing.Concat _ -> false
                  in
                  let eligible tn idcs =
                    contiguous idcs && lane_aligned tn idcs
                    && Tn.Placements.is_materialized_force (placements ()) tn 463
                    (* Stack and workgroup-shared arrays are only element-aligned. *)
                    && not (Set.mem !current_workgroup_shared tn)
                    && not (is_swizzled tn)
                  in
                  let vload tn idcs =
                    if not (eligible tn idcs) then raise Bail;
                    check_read ~written tn idcs;
                    let name = fresh "vget" in
                    emit
                      (string
                         (Printf.sprintf "const %s %s = *reinterpret_cast<%sconst %s*>(&" vtyp name
                            B.buffer_prefix vtyp)
                      ^^ string (get_ident tn)
                      ^^ brackets (pp_array_offset (idcs, Lazy.force tn.Tn.dims))
                      ^^ string ");");
                    name
                  in
                  let lane_var = "ocannl_l__" in
                  let rec lane_expr (llsc : Low_level.scalar_t) (p : Ops.prec) : PPrint.document =
                    if not (scalar_mentions llsc) then uniform_scalar ~written prec llsc
                    else if not (Ops.equal_prec p prec) then raise Bail
                    else
                      match llsc with
                      | Low_level.Get (tn, idcs) ->
                          if not (Ops.equal_prec (Lazy.force tn.Tn.prec) prec) then raise Bail;
                          string (vload tn idcs ^ ".v[" ^ lane_var ^ "]")
                      | Binop (((Ops.Add | Ops.Sub | Ops.Mul | Ops.Div) as op), (a, pa), (b, pb)) ->
                          B.binop_syntax prec op (lane_expr a pa) (lane_expr b pb)
                      | Ternop (Ops.FMA, (a, pa), (b, pb), (c, pc)) ->
                          B.ternop_syntax prec Ops.FMA (lane_expr a pa) (lane_expr b pb)
                            (lane_expr c pc)
                      | Unop (Ops.Identity, (a, pa)) -> lane_expr a pa
                      | Unop (Ops.Neg, (a, pa)) -> parens (string "-" ^^ lane_expr a pa)
                      | _ -> raise Bail
                  in
                  List.iter sets ~f:(fun (tn, idcs, llsc) ->
                      if not (eligible tn idcs) then raise Bail;
                      let rhs = lane_expr llsc prec in
                      let vname = fresh "vset" in
                      emit (string (vtyp ^ " " ^ vname ^ ";"));
                      emit
                        (string
                           (Printf.sprintf "for (int %s = 0; %s < %d; ++%s) { %s.v[%s] = " lane_var
                              lane_var lanes lane_var vname lane_var)
                        ^^ rhs ^^ string "; }");
                      emit
                        (string (Printf.sprintf "*reinterpret_cast<%s%s*>(&" B.buffer_prefix vtyp)
                        ^^ string (get_ident tn)
                        ^^ brackets (pp_array_offset (idcs, Lazy.force tn.Tn.dims))
                        ^^ string (") = " ^ vname ^ ";")));
                  empty
            in
            let ivar = symbol_ident i in
            let it = B.loop_index_type in
            let body_vec = separate hardline (List.rev !stmts_docs) in
            Some
              (string
                 (Printf.sprintf "{ /* Vectorized rendering: %d lanes of %s. */" lanes
                    (B.typ_of_prec prec))
              ^^ nest 2
                   (hardline ^^ prelude
                   ^^ string (Printf.sprintf "%s%s = 0;" it ivar)
                   ^^ hardline
                   ^^ string
                        (Printf.sprintf "for (; %s + %d <= %d; %s += %d) {" ivar lanes extent ivar
                           lanes)
                   ^^ nest 2 (hardline ^^ body_vec)
                   ^^ hardline ^^ string "}" ^^ hardline
                   ^^ string (Printf.sprintf "for (; %s <= %d; ++%s) {" ivar to_ ivar)
                   ^^ nest 2 (hardline ^^ body_doc ())
                   ^^ hardline ^^ string "}")
              ^^ hardline ^^ string "}")
          with Bail -> None
        in
        (* SIMD reduction rendering of a [Vectorized] accumulation loop (gh-ocannl-468; ggml's
           [ggml_vec_dot_f32] pattern, ggml/src/ggml-cpu/vec.h). A recognized accumulation body
           [acc[idcs] = op(acc[idcs], contrib(i))] renders as a 1×[chains] grid of independent
           vector accumulator registers — splitting the loop-carried dependency into [chains *
           lanes] independent chains is exactly the strict-FP reassociation the [Vectorized] retype
           licenses — updated in a fused main loop advancing by [chains * lanes], then folded
           register-wise, lane-wise, and finally into the accumulator cell; a serial tail loop
           reuses the scalar body. [chains] defaults to 4 (ggml's [GGML_F32_ARR]: enough independent
           chains to cover the FMA latency-throughput gap), halved until the first-block
           initialization fits the extent. Initializing the chains from the first [chains] blocks of
           contributions avoids needing an identity constant, so [Max]/[Min] reductions work
           unchanged. [`Vec_extensions] only: on GPU backends reductions parallelize via
           [Workgroup_reduce] warp shuffles instead, and a bailed-out accumulation falls back to a
           plain serial loop (never to [vectorize_pragma], which would assert iteration
           independence). *)
        let try_vectorize_reduce () : PPrint.document option =
          try
            (match B.vector_style with `Vec_extensions -> () | `Packed_struct -> raise Bail);
            if B.vector_bytes < 8 || from_ <> 0 || Utils.debug_log_from_routines () then raise Bail;
            let acc_tn, acc_idcs, op, contrib =
              match recognize_accumulation (nonempty_stmts body) with
              | Some r -> r
              | None -> raise Bail
            in
            let prec = Lazy.force acc_tn.Tn.prec in
            (match prec with Ops.Single_prec _ | Ops.Double_prec _ -> () | _ -> raise Bail);
            (* A loop-invariant contribution deserves strength reduction, not chains. *)
            if not (scalar_mentions contrib) then raise Bail;
            let lanes = B.vector_bytes / Ops.prec_in_bytes prec in
            let extent = to_ + 1 in
            if lanes < 2 || extent < lanes then raise Bail;
            let chains = if 4 * lanes <= extent then 4 else if 2 * lanes <= extent then 2 else 1 in
            let step = chains * lanes in
            let vtyp, typedef_doc = vec_ext_typ ~prec ~lanes in
            (* The accumulator is not vector-loaded ([contrib] cannot touch it, per the recognizer),
               so nothing is [written] from the vector expressions' viewpoint. *)
            let written = Hashtbl.create (module Int) in
            let stmts_docs = ref [] in
            let emit d = stmts_docs := d :: !stmts_docs in
            let take () =
              let docs = List.rev !stmts_docs in
              stmts_docs := [];
              docs
            in
            let fresh =
              let ctr = ref 0 in
              fun pfx ->
                Int.incr ctr;
                Printf.sprintf "%s%d__" pfx !ctr
            in
            let _vec_expr, vec_operand =
              vec_ext_machinery ~prec ~lanes ~vtyp ~written ~emit ~fresh
            in
            (* Chain [c] consumes the flat-offset window shifted by [c * lanes]: under the
               contiguity rule the shift is a constant added to each access's last (loop-index)
               component. *)
            let shift_idx ~by (idx : Indexing.axis_index) =
              match idx with
              | Indexing.Iterator s when Indexing.equal_symbol s i ->
                  Indexing.Affine { symbols = [ (1, s) ]; offset = by }
              | Indexing.Affine { symbols; offset }
                when List.exists symbols ~f:(fun (_, s) -> Indexing.equal_symbol s i) ->
                  Indexing.Affine { symbols; offset = offset + by }
              | _ -> idx
            in
            let rec shift ~by (llsc : Low_level.scalar_t) : Low_level.scalar_t =
              if by = 0 then llsc
              else
                match llsc with
                | Low_level.Get (tn, idcs) -> Low_level.Get (tn, Array.map idcs ~f:(shift_idx ~by))
                | Binop (op, (a, pa), (b, pb)) -> Binop (op, (shift ~by a, pa), (shift ~by b, pb))
                | Ternop (op, (a, pa), (b, pb), (c, pc)) ->
                    Ternop (op, (shift ~by a, pa), (shift ~by b, pb), (shift ~by c, pc))
                | Unop (op, (a, pa)) -> Unop (op, (shift ~by a, pa))
                (* Any other index-mentioning form bails inside [vec_ext_machinery]. *)
                | _ -> llsc
            in
            let ivar = symbol_ident i in
            let grid = vec_acc_grid ~prefix:("vred_" ^ ivar) ~rows:1 ~cols:chains in
            let acc_regs = grid.(0) in
            (* Chain initialization from the first [chains] blocks, read at [i = 0]. *)
            Array.iteri acc_regs ~f:(fun c name ->
                let rhs = vec_operand (shift ~by:(c * lanes) contrib) prec in
                emit (string (vtyp ^ " " ^ name ^ " = ") ^^ rhs ^^ semi));
            let init_docs = take () in
            (* The fused main-loop body: one independent update per chain. *)
            Array.iteri acc_regs ~f:(fun c name ->
                match (op, contrib) with
                | Ops.Add, Low_level.Binop (Ops.Mul, (a, pa), (b, pb)) ->
                    (* The dot-product case (also reached from the recognizer's FMA form):
                       fused-multiply-accumulate straight into the chain register. *)
                    let bind llsc p =
                      let nm = fresh "vfop" in
                      emit
                        (string (vtyp ^ " " ^ nm ^ " = ")
                        ^^ vec_operand (shift ~by:(c * lanes) llsc) p
                        ^^ semi);
                      nm
                    in
                    let na = bind a pa in
                    let nb = bind b pb in
                    emit (vec_acc_fma ~prec ~lanes ~dst:name ~a:na ~b:nb)
                | _ ->
                    let nm = fresh "vsrc" in
                    emit
                      (string (vtyp ^ " " ^ nm ^ " = ")
                      ^^ vec_operand (shift ~by:(c * lanes) contrib) prec
                      ^^ semi);
                    emit (vec_acc_combine ~prec ~lanes ~op ~dst:name ~src:nm));
            let update_docs = take () in
            let total = "vred_total_" ^ ivar ^ "__" in
            let acc_cell () =
              string (get_ident acc_tn)
              ^^ brackets (pp_array_offset (acc_idcs, Lazy.force acc_tn.Tn.dims))
            in
            let epilogue =
              vec_acc_grid_fold ~prec ~lanes ~op grid
              @ vec_acc_lane_fold ~prec ~lanes ~op ~vname:acc_regs.(0) ~out:total
              @ [
                  acc_cell () ^^ string " = "
                  ^^ B.binop_syntax prec op (acc_cell ()) (string total)
                  ^^ semi;
                ]
            in
            let it = B.loop_index_type in
            Some
              (string
                 (Printf.sprintf
                    "{ /* Vectorized reduction rendering: %d chain(s) of %d lanes of %s. */" chains
                    lanes (B.typ_of_prec prec))
              ^^ nest 2
                   (hardline ^^ typedef_doc ^^ hardline
                   ^^ string (Printf.sprintf "%s%s = 0;" it ivar)
                   ^^ hardline ^^ separate hardline init_docs ^^ hardline
                   ^^ string
                        (Printf.sprintf "for (%s = %d; %s + %d <= %d; %s += %d) {" ivar step ivar
                           step extent ivar step)
                   ^^ nest 2 (hardline ^^ separate hardline update_docs)
                   ^^ hardline ^^ string "}" ^^ hardline ^^ separate hardline epilogue ^^ hardline
                   ^^ string (Printf.sprintf "for (; %s <= %d; ++%s) {" ivar to_ ivar)
                   ^^ nest 2 (hardline ^^ body_doc ())
                   ^^ hardline ^^ string "}")
              ^^ hardline ^^ string "}")
          with Bail -> None
        in
        (* Warp-shuffle rendering of a [Workgroup_reduce] accumulation loop (gh-ocannl-462; llm.c's
           [warpReduceSum] / [blockReduce] idiom, llmc/cuda_utils.cuh). Recognizes a body that is a
           single accumulation statement [acc[idcs] = op(acc[idcs], contrib)] (or its FMA form [acc
           = FMA(a, b, acc)]) where [idcs] does not mention the loop index and [op] is an
           associative-commutative reduction — such a body IS the loop's serial meaning, so backends
           without shuffle support ([warp_size = 0]) render it with the ordinary fallbacks. With
           shuffle support the loop renders as: every thread computes its contribution, a log2(warp)
           [ocannl_shfl_xor] tree reduces within each warp, then (for multi-warp extents) lane 0 of
           each warp stages one value in a workgroup-shared slot, a barrier, and the first warp
           shuffle-reduces the per-warp partials — thread 0 finally folds the total into the
           accumulator (reassociation is the annotation's license, like [Vectorized]). This halves
           the shared-memory traffic and barrier count of the explicitly staged tree, which remains
           supported: unrecognized bodies keep the [Workgroup]-style hardware binding and their own
           staging and barriers.

           The multi-warp phase needs no identity constant: [num_warps] must be a power of two, and
           XOR with offsets [< num_warps] maps lanes [< num_warps] onto themselves, so the garbage
           held by lanes [>= num_warps] never mixes into the reduced prefix.

           A recognized accumulation that cannot be rendered (extent not covering whole warps,
           reduce axis not at workgroup slot 0, ...) raises: binding the index like a plain
           [Workgroup] axis would make every thread race the read-modify-write. *)
        let try_warp_reduce () : PPrint.document option =
          if B.warp_size <= 0 then None
          else
            let stmts = nonempty_stmts body in
            (* When this loop's extent is smaller than its slot's launch dimension,
               [guard_annotated_extents] has already wrapped the body in the synthetic launch guard
               [If (i < extent)]. Strip exactly that shape — it is vacuous with respect to the
               loop's own iteration space — so a guarded accumulation is still recognized, and then
               rejected by the extent-coverage check below, instead of silently racing under the
               hardware-binding fallback (PR #119 review). *)
            let stmts =
              match stmts with
              | [
               Low_level.If
                 {
                   cond =
                     Binop (Ops.Cmplt, (Embed_index (Indexing.Iterator s), _), (Constant c, _)), _;
                   body = guarded;
                 };
              ]
                when Indexing.equal_symbol s i && Float.equal c (Float.of_int (to_ - from_ + 1)) ->
                  nonempty_stmts guarded
              | _ -> stmts
            in
            match recognize_accumulation stmts with
            | None -> None
            | Some (tn, idcs, op, contrib) ->
                let fail msg =
                  invalid_arg
                    ("C_syntax.pp_ll: Workgroup_reduce loop " ^ symbol_ident i
                   ^ " is a recognized accumulation, but the warp-shuffle rendering requires " ^ msg
                   ^ " (a plain hardware binding would race the accumulator update)")
                in
                let warp = B.warp_size in
                assert (warp > 1 && Int.is_pow2 warp);
                let extent = to_ - from_ + 1 in
                let prec = Lazy.force tn.Tn.prec in
                (match prec with
                | Ops.Single_prec _ | Ops.Double_prec _ -> ()
                | _ -> fail "a single- or double-precision accumulator");
                if Utils.debug_log_from_routines () then
                  fail "debug_log_from_routines to be disabled";
                if extent % warp <> 0 then
                  fail
                    (Printf.sprintf "the extent (%d) to be a multiple of the warp size (%d)" extent
                       warp);
                let num_warps = extent / warp in
                let axes = !current_hardware_axes in
                (match
                   List.find axes ~f:(fun a -> Indexing.equal_symbol a.Low_level.ha_index i)
                 with
                | Some a when a.Low_level.ha_slot = 0 -> ()
                | Some _ ->
                    fail
                      "the reduce axis at workgroup slot 0 (warp lanes are consecutive .x threads)"
                | None ->
                    invalid_arg
                      ("C_syntax.pp_ll: hardware-annotated loop " ^ symbol_ident i
                     ^ " missing from the slot table (pp_ll called outside compile_proc?)"));
                let slot_max =
                  List.fold axes ~init:1 ~f:(fun m a ->
                      match a.Low_level.ha_kind with
                      | `Workgroup when a.Low_level.ha_slot = 0 -> max m a.Low_level.ha_extent
                      | _ -> m)
                in
                if extent <> slot_max then
                  fail
                    "the extent to cover the whole workgroup .x dimension (a smaller sibling \
                     extent would diverge the shuffles)";
                let reg =
                  match B.hardware_index ~kind:`Workgroup ~slot:0 with
                  | Some reg -> reg
                  | None -> fail "the backend to bind workgroup slot 0"
                in
                if num_warps > 1 then (
                  if not (Int.is_pow2 num_warps) then
                    fail "a power-of-two number of warps (for the identity-free second phase)";
                  if num_warps > warp then
                    fail "at most warp-size warps (one second-phase slot per warp)";
                  if Option.is_none B.barrier_syntax || Option.is_none B.shared_decl_prefix then
                    fail "barrier and workgroup-shared support";
                  if
                    List.exists axes ~f:(fun a ->
                        match a.Low_level.ha_kind with
                        | `Workgroup -> not (Indexing.equal_symbol a.Low_level.ha_index i)
                        | `Grid -> false)
                  then
                    fail
                      "the reduce axis to be the only workgroup axis (the per-warp staging slots \
                       are not replicated per sibling workgroup thread)");
                let ident = symbol_ident i in
                let ctyp = B.typ_of_prec prec in
                let cast = "(" ^ String.strip B.loop_index_type ^ ")" in
                let vname = "wred_v_" ^ ident ^ "__" in
                let combine a b = B.binop_syntax prec op a b in
                let rec halvings n = if n < 1 then [] else n :: halvings (n / 2) in
                let shuffle_stage off =
                  string (vname ^ " = ")
                  ^^ combine (string vname)
                       (string (Printf.sprintf "ocannl_shfl_xor(%s, %d)" vname off))
                  ^^ semi
                in
                let acc_doc =
                  string (get_ident tn) ^^ brackets (pp_array_offset (idcs, Lazy.force tn.Tn.dims))
                in
                let fold_total =
                  group
                    (string "if (" ^^ pp_symbol i ^^ string " == 0) " ^^ lbrace
                    ^^ nest 2
                         (hardline ^^ acc_doc ^^ string " = "
                         ^^ combine acc_doc (string vname)
                         ^^ semi)
                    ^^ hardline ^^ rbrace)
                in
                let tail =
                  if num_warps = 1 then fold_total
                  else
                    let pname = "wred_partials_" ^ ident ^ "__" in
                    let barrier = string (Option.value_exn B.barrier_syntax) in
                    string
                      (Printf.sprintf "%s%s %s[%d];"
                         (Option.value_exn B.shared_decl_prefix)
                         ctyp pname num_warps)
                    ^^ hardline
                    ^^ string
                         (Printf.sprintf "if ((%s & %d) == 0) { %s[%s >> %d] = %s; }" ident
                            (warp - 1) pname ident (Int.ceil_log2 warp) vname)
                    ^^ hardline ^^ barrier ^^ hardline
                    ^^ group
                         (string (Printf.sprintf "if (%s < %d) " ident warp)
                         ^^ lbrace
                         ^^ nest 2
                              (hardline
                              ^^ string
                                   (Printf.sprintf "if (%s < %d) { %s = %s[%s]; }" ident num_warps
                                      vname pname ident)
                              ^^ hardline
                              ^^ separate hardline
                                   (List.map (halvings (num_warps / 2)) ~f:shuffle_stage)
                              ^^ hardline ^^ fold_total)
                         ^^ hardline ^^ rbrace)
                    ^^ hardline ^^ barrier
                in
                let local_defs, contrib_doc = pp_scalar prec contrib in
                let local_defs = pp_local_defs local_defs in
                let binding =
                  string ("const " ^ B.loop_index_type)
                  ^^ pp_symbol i
                  ^^ string (" = " ^ cast ^ reg ^ ";")
                in
                Some
                  (string
                     (Printf.sprintf
                        "{ /* Workgroup_reduce warp-shuffle rendering: extent %d = %d simdgroup(s) \
                         of %d. */"
                        extent num_warps warp)
                  ^^ nest 2
                       (hardline ^^ binding ^^ hardline
                       ^^ (if PPrint.is_empty local_defs then empty else local_defs ^^ hardline)
                       ^^ string (ctyp ^ " " ^ vname ^ " = ")
                       ^^ contrib_doc ^^ semi ^^ hardline
                       ^^ separate hardline (List.map (halvings (warp / 2)) ~f:shuffle_stage)
                       ^^ hardline ^^ tail)
                  ^^ hardline ^^ rbrace)
        in
        match axis with
        | Low_level.Serial -> serial_loop ()
        | Grid when Set.mem !current_parallel_grid i -> parallel_grid_loop ()
        | Grid -> hardware_binding `Grid
        | Workgroup -> hardware_binding `Workgroup
        | Workgroup_reduce -> (
            match try_warp_reduce () with Some doc -> doc | None -> hardware_binding `Workgroup)
        | Vectorized -> (
            match try_vectorize_reduce () with
            | Some doc -> doc
            | None -> (
                match try_vectorize () with
                | Some doc -> doc
                | None -> (
                    if
                      (* gh-ocannl-164: a serial loop annotated with the backend's vectorization
                         pragmas; without them the plain serial loop is the legal fallback (same
                         discipline as unbound [Grid]/[Workgroup] axes). The pragmas assert
                         iteration independence, which a loop-carried accumulation does not satisfy
                         — an accumulating body that no explicit rendering accepted always falls
                         back to the plain serial loop (gh-ocannl-468: this is what lets the
                         autotune menu propose Retype-[Vectorized] over reductions). *)
                      Low_level.has_accumulation body
                    then serial_loop ()
                    else
                      match B.vectorize_pragma with
                      | [] -> serial_loop ()
                      | lines -> separate_map hardline string lines ^^ hardline ^^ serial_loop ())))
        | Unrolled ->
            separate hardline
            @@ List.init
                 (to_ - from_ + 1)
                 ~f:(fun k ->
                   let binding =
                     string ("const " ^ B.loop_index_type)
                     ^^ pp_symbol i
                     ^^ string (" = " ^ Int.to_string (from_ + k) ^ ";")
                   in
                   group
                     (lbrace
                     ^^ nest 2 (hardline ^^ binding ^^ hardline ^^ body_doc ())
                     ^^ hardline ^^ rbrace)))
    | Zero_out tn ->
        let first_touch = not (Hash_set.mem zero_out_seen tn.Tn.uid) in
        Hash_set.add zero_out_seen tn.Tn.uid;
        if first_touch && (not in_loop) && zero_out_loop_redundant tn then
          (* First-touch, executed once at function scope: the declaration's [= {0}] already covers
             it. A later [Zero_out tn], or one reached inside a loop, is a real re-zero and falls
             through to emit the zeroing loop below. *)
          empty
        else
          pp_ll ~log_set_locals ~in_loop
            (Low_level.loop_over_dims (Lazy.force tn.dims) ~body:(fun idcs ->
                 Set { tn; idcs; llsc = Constant 0.0; debug = get_ident tn ^ " := 0" }))
    | Set { tn; idcs; llsc; debug } ->
        let ident_doc = string (get_ident tn) in
        let dims = Lazy.force tn.dims in
        let prec = Lazy.force tn.prec in
        let local_defs, val_doc = pp_scalar prec llsc in
        let local_defs = pp_local_defs local_defs in
        let offset_doc = pp_tn_offset tn (idcs, dims) in
        (* See {!C_syntax_config.volatile_scalar_rmw}: pin the per-iteration read-modify-write of
           loop-invariant-address accumulators by shadowing the node's pointer with a
           volatile-qualified alias for the whole statement (the shadow also covers reads inside
           [local_defs]). The rule keys on the miscompiling pass's precondition — a
           read-modify-write whose address is invariant across at least one enclosing serial [for]
           loop (a scalar loss reduction's constant index, a gradient accumulated over an outer
           batch loop, a matmul/conv accumulator indexed only by loops outside its reduction) —
           because no finer syntactic discriminator survived the observed cases: plain-FMA and
           Local-scope-bearing statements both miscompiled in some kernels while byte-alike
           statements in others compiled fine. *)
        let mentions_sym s (idx : Indexing.axis_index) =
          match idx with
          | Indexing.Iterator s2 -> Indexing.equal_symbol s s2
          | Indexing.Affine { symbols; _ } ->
              List.exists symbols ~f:(fun (_, s2) -> Indexing.equal_symbol s s2)
          | Indexing.Fixed_idx _ | Indexing.Sub_axis | Indexing.Concat _ -> false
        in
        let rmw_volatile =
          B.volatile_scalar_rmw
          && List.exists !serial_loop_stack ~f:(fun s ->
              not (Array.exists idcs ~f:(mentions_sym s)))
          (* Only kernel-parameter-derived device pointers: routine-local scratch is declared as a
             plain local array (not address-castable, and compiler-visible anyway). *)
          && Tn.Placements.is_materialized_force (placements ()) tn 433
          &&
          let rec reads_tn (llsc : Low_level.scalar_t) =
            match llsc with
            | Low_level.Get (tn2, _) -> Tn.equal tn tn2
            | Get_dynamic { tn = tn2; dyn_value = v, _; _ } -> Tn.equal tn tn2 || reads_tn v
            | Local_scope { body; _ } -> body_reads_tn body
            | Get_local _ | Get_merge_buffer _ | Constant _ | Constant_bits _ | Embed_index _ ->
                false
            | Ternop (_, (a, _), (b, _), (c, _)) -> reads_tn a || reads_tn b || reads_tn c
            | Binop (_, (a, _), (b, _)) -> reads_tn a || reads_tn b
            | Unop (_, (a, _)) -> reads_tn a
          and body_reads_tn (body : Low_level.t) =
            match body with
            | Low_level.Seq (a, b) -> body_reads_tn a || body_reads_tn b
            | For_loop { body; _ } -> body_reads_tn body
            | If { cond = c, _; body } -> reads_tn c || body_reads_tn body
            | Set { llsc; _ } | Set_local (_, llsc) -> reads_tn llsc
            | Set_dynamic { dyn_value = v, _; llsc; _ } -> reads_tn v || reads_tn llsc
            | Set_from_vec { arg = a, _; _ } -> reads_tn a
            | Noop | Comment _ | Staged_compilation _ | Zero_out _ | Declare_local _
            | Workgroup_barrier | Tile_mma _ ->
                false
          in
          reads_tn llsc
        in
        let wrap_rmw_volatile stmt_doc =
          if not rmw_volatile then stmt_doc
          else
            let vol_ptr = string (B.buffer_prefix ^ "volatile " ^ B.typ_of_prec prec ^ "*") in
            (* The ident is for readability of the generated source; the uid suffix guarantees
               uniqueness — label-derived identifiers (get_ident) could legally collide with a
               prefixed name, and the inner scope must shadow nothing except the target node. *)
            let tmp = string ("__rmw_" ^ get_ident tn ^ "_" ^ Int.to_string tn.Tn.uid) in
            lbrace
            ^^ nest 2
                 (hardline ^^ vol_ptr ^^ space ^^ tmp ^^ string " = " ^^ ident_doc ^^ semi
                ^^ hardline ^^ lbrace
                 ^^ nest 2
                      (hardline ^^ vol_ptr ^^ space ^^ ident_doc ^^ string " = " ^^ tmp ^^ semi
                     ^^ hardline ^^ stmt_doc)
                 ^^ hardline ^^ rbrace)
            ^^ hardline ^^ rbrace
        in
        let assignment =
          group
            (ident_doc ^^ brackets offset_doc ^^ string " ="
            ^^ ifflat (space ^^ val_doc) (nest 4 (hardline ^^ val_doc))
            ^^ semi)
        in
        if Utils.debug_log_from_routines () then
          let num_typ = string (B.typ_of_prec prec) in
          let new_var = string "new_set_v" in
          let decl = num_typ ^^ space ^^ new_var ^^ string " = " ^^ val_doc ^^ semi in
          let debug_val_doc, debug_args_docs = debug_float prec llsc in
          let debug_val_str = doc_to_string debug_val_doc in
          let pp_args_docs =
            List.map debug_args_docs ~f:(function
              | `Accessor idx -> pp_array_offset idx
              | `Value v_doc -> B.styled_log_arg v_doc)
          in
          let log_args_for_printf =
            offset_doc
            :: B.styled_log_arg (ident_doc ^^ brackets offset_doc)
            :: B.styled_log_arg new_var :: pp_args_docs
          in
          let log_doc =
            let log_param_doc = Option.map B.kernel_log_param ~f:(fun (_, name) -> string name) in
            let comment_base_msg = "# " ^ debug ^ "\n" in
            let value_base_msg =
              Printf.sprintf "%s[%%u]{=%s} = %s = %s\n" (get_ident tn) B.float_log_style
                B.float_log_style debug_val_str
            in
            let comment_log =
              B.pp_log_statement ~log_param_c_expr_doc:log_param_doc
                ~base_message_literal:comment_base_msg ~args_docs:[]
            in
            let value_log =
              B.pp_log_statement ~log_param_c_expr_doc:log_param_doc
                ~base_message_literal:value_base_msg ~args_docs:log_args_for_printf
            in
            let flush_log =
              if B.log_involves_file_management then string "fflush(log_file);" else empty
            in
            comment_log ^^ hardline ^^ value_log ^^ hardline ^^ flush_log
          in
          let assignment' = ident_doc ^^ brackets offset_doc ^^ string " = " ^^ new_var ^^ semi in
          let block_content =
            if PPrint.is_empty local_defs then
              decl ^^ hardline ^^ log_doc ^^ hardline ^^ assignment'
            else local_defs ^^ hardline ^^ decl ^^ hardline ^^ log_doc ^^ hardline ^^ assignment'
          in
          wrap_rmw_volatile (lbrace ^^ nest 2 (hardline ^^ block_content) ^^ hardline ^^ rbrace)
        else if PPrint.is_empty local_defs then wrap_rmw_volatile assignment
        else
          let block_content = local_defs ^^ hardline ^^ assignment in
          wrap_rmw_volatile (lbrace ^^ nest 2 (hardline ^^ block_content) ^^ hardline ^^ rbrace)
    | Set_dynamic { tn; idcs; dyn_axis; dyn_value = iv, iprec; llsc; debug } ->
        (* gh-466: the scatter counterpart of the [Get_dynamic] gather — the write offset splices
           the runtime index (cast to [Ops.index_prec ()], mirroring the gather) at [dyn_axis]. The
           enclosing [If] guard (when interval analysis has not discharged it) guarantees the index
           is in range before this statement executes. *)
        if is_swizzled tn then
          invalid_arg
            ("C_syntax: Set_dynamic targets swizzled node " ^ Tn.debug_name tn
           ^ " (dynamic offsets are not swizzle-remapped)");
        let ident_doc = string (get_ident tn) in
        let dims = Lazy.force tn.dims in
        let prec = Lazy.force tn.prec in
        let dyn_defs, dyn_expr = pp_scalar iprec iv in
        let idx_typ = B.typ_of_prec (Ops.index_prec ()) in
        let dyn_idx_doc = string ("((" ^ idx_typ ^ ")(") ^^ dyn_expr ^^ string "))" in
        let offset_doc = pp_array_offset_dyn (idcs, dims) ~dyn_axis ~dyn_idx_doc in
        let val_defs, val_doc = pp_scalar prec llsc in
        let local_defs = pp_local_defs (dyn_defs @ val_defs) in
        let assignment =
          group
            (ident_doc ^^ brackets offset_doc ^^ string " ="
            ^^ ifflat (space ^^ val_doc) (nest 4 (hardline ^^ val_doc))
            ^^ semi)
        in
        if Utils.debug_log_from_routines () then
          let num_typ = string (B.typ_of_prec prec) in
          let new_var = string "new_set_v" in
          let decl = num_typ ^^ space ^^ new_var ^^ string " = " ^^ val_doc ^^ semi in
          let debug_val_doc, debug_args_docs = debug_float prec llsc in
          let debug_val_str = doc_to_string debug_val_doc in
          let pp_args_docs =
            List.map debug_args_docs ~f:(function
              | `Accessor idx -> pp_array_offset idx
              | `Value v_doc -> B.styled_log_arg v_doc)
          in
          let log_args_for_printf =
            offset_doc
            :: B.styled_log_arg (ident_doc ^^ brackets offset_doc)
            :: B.styled_log_arg new_var :: pp_args_docs
          in
          let log_doc =
            let log_param_doc = Option.map B.kernel_log_param ~f:(fun (_, name) -> string name) in
            let comment_base_msg = "# " ^ debug ^ "\n" in
            let value_base_msg =
              Printf.sprintf "%s[%%u]{=%s} = %s = %s\n" (get_ident tn) B.float_log_style
                B.float_log_style debug_val_str
            in
            let comment_log =
              B.pp_log_statement ~log_param_c_expr_doc:log_param_doc
                ~base_message_literal:comment_base_msg ~args_docs:[]
            in
            let value_log =
              B.pp_log_statement ~log_param_c_expr_doc:log_param_doc
                ~base_message_literal:value_base_msg ~args_docs:log_args_for_printf
            in
            let flush_log =
              if B.log_involves_file_management then string "fflush(log_file);" else empty
            in
            comment_log ^^ hardline ^^ value_log ^^ hardline ^^ flush_log
          in
          let assignment' = ident_doc ^^ brackets offset_doc ^^ string " = " ^^ new_var ^^ semi in
          let block_content =
            if PPrint.is_empty local_defs then
              decl ^^ hardline ^^ log_doc ^^ hardline ^^ assignment'
            else local_defs ^^ hardline ^^ decl ^^ hardline ^^ log_doc ^^ hardline ^^ assignment'
          in
          lbrace ^^ nest 2 (hardline ^^ block_content) ^^ hardline ^^ rbrace
        else if PPrint.is_empty local_defs then assignment
        else
          let block_content = local_defs ^^ hardline ^^ assignment in
          lbrace ^^ nest 2 (hardline ^^ block_content) ^^ hardline ^^ rbrace
    | Comment message ->
        if Utils.debug_log_from_routines () then
          let base_message = "COMMENT: " ^ message ^ "\n" in
          let log_param_doc = Option.map B.kernel_log_param ~f:(fun (_, name) -> string name) in
          B.pp_log_statement ~log_param_c_expr_doc:log_param_doc ~base_message_literal:base_message
            ~args_docs:[]
        else string "/* " ^^ string message ^^ string " */"
    | Staged_compilation callback -> callback ()
    | Set_from_vec { tn; idcs; length; vec_unop; arg = arg, arg_prec; debug } ->
        (* Multi-element consecutive-offset write: incompatible with a swizzled layout, and staged
           tiles are only ever written by the Stage-minted scalar load nest — fail loudly rather
           than miscompile if that invariant is ever broken. *)
        if is_swizzled tn then
          invalid_arg
            ("C_syntax: Set_from_vec targets swizzled node " ^ Tn.debug_name tn
           ^ " (row-major multi-element write into an XOR-swizzled layout)");
        let ident_doc = string (get_ident tn) in
        let dims = Lazy.force tn.dims in
        let prec = Lazy.force tn.prec in
        (* Determine argument precision based on operation homogeneity *)
        let arg_prec =
          if Ops.is_homogeneous_prec_vec_unop vec_unop then prec
            (* Homogeneous: argument uses result precision *)
          else arg_prec
        in
        let local_defs, arg_doc = pp_scalar arg_prec arg in
        let local_defs = pp_local_defs local_defs in
        (* Generate the function call *)
        let result_doc = B.vec_unop_syntax prec vec_unop arg_doc in
        (* The vector temporary always has the full 128-bit block width: a tail store ([length] <
           block lanes) stores only the leading lanes of the fully computed block. *)
        let block_lanes = Ops.vec_unop_lanes prec in
        (* Generate assignments for each output element *)
        let open PPrint in
        let vec_var = string "vec_result" in
        let vec_typ = string (B.vec_typ_of_prec ~length:block_lanes prec) in
        let vec_decl = vec_typ ^^ space ^^ vec_var ^^ string " = " ^^ result_doc ^^ semi in
        let assignments =
          let elem_assigns =
            List.init length ~f:(fun i ->
                let offset_doc =
                  match idcs.(Array.length idcs - 1) with
                  | Fixed_idx idx ->
                      (* For Fixed_idx, update the index and compute offset normally *)
                      let elem_idcs = Array.copy idcs in
                      elem_idcs.(Array.length elem_idcs - 1) <- Fixed_idx (idx + i);
                      pp_array_offset (elem_idcs, dims)
                  | _ ->
                      (* For non-Fixed_idx (Iterator, etc), add i to the computed offset *)
                      pp_array_offset (idcs, dims) ^^ string (" + " ^ Int.to_string i)
                in
                let value_doc =
                  if block_lanes = 1 then
                    (* A single-lane block has a scalar type, so no .v[] access *)
                    vec_var
                  else
                    (* When the block has multiple lanes, access the vector element *)
                    vec_var ^^ string (".v[" ^ Int.to_string i ^ "]")
                in
                ident_doc ^^ brackets offset_doc ^^ string " = " ^^ value_doc ^^ semi)
          in
          separate hardline elem_assigns
        in
        if Utils.debug_log_from_routines () then
          let open PPrint in
          let log_param_doc = Option.map B.kernel_log_param ~f:(fun (_, name) -> string name) in
          let comment_base_msg = "# " ^ debug ^ "\n" in
          let comment_log =
            B.pp_log_statement ~log_param_c_expr_doc:log_param_doc
              ~base_message_literal:comment_base_msg ~args_docs:[]
          in
          let value_logs =
            List.init length ~f:(fun i ->
                let elem_idcs = Array.copy idcs in
                (match elem_idcs.(Array.length elem_idcs - 1) with
                | Fixed_idx idx -> elem_idcs.(Array.length elem_idcs - 1) <- Fixed_idx (idx + i)
                | _ -> ());
                let offset_doc =
                  let base_offset = pp_array_offset (elem_idcs, dims) in
                  match elem_idcs.(Array.length elem_idcs - 1) with
                  | Fixed_idx _ -> base_offset
                  | _ -> base_offset ^^ string (" + " ^ Int.to_string i)
                in
                let value_base_msg =
                  Printf.sprintf "%s[%%u]{=%s} = vec_result.v[%d] = %s\n" (get_ident tn)
                    B.float_log_style i B.float_log_style
                in
                let log_args =
                  [
                    offset_doc;
                    B.styled_log_arg (ident_doc ^^ brackets offset_doc);
                    B.styled_log_arg (string ("vec_result.v[" ^ Int.to_string i ^ "]"));
                  ]
                in
                B.pp_log_statement ~log_param_c_expr_doc:log_param_doc
                  ~base_message_literal:value_base_msg ~args_docs:log_args)
          in
          let flush_log =
            if B.log_involves_file_management then string "fflush(log_file);" else empty
          in
          let log_docs =
            comment_log ^^ hardline ^^ separate hardline value_logs ^^ hardline ^^ flush_log
          in
          let block_content =
            if PPrint.is_empty local_defs then
              vec_decl ^^ hardline ^^ log_docs ^^ hardline ^^ assignments
            else
              local_defs ^^ hardline ^^ vec_decl ^^ hardline ^^ log_docs ^^ hardline ^^ assignments
          in
          lbrace ^^ nest 2 (hardline ^^ block_content) ^^ hardline ^^ rbrace
        else if PPrint.is_empty local_defs then vec_decl ^^ hardline ^^ assignments
        else
          let block_content = local_defs ^^ hardline ^^ vec_decl ^^ hardline ^^ assignments in
          lbrace ^^ nest 2 (hardline ^^ block_content) ^^ hardline ^^ rbrace
    | Set_local (({ tn = { prec; _ }; _ } as id), value) ->
        let prec = Lazy.force prec in
        let local_defs, value_doc = pp_scalar prec value in
        let local_defs = pp_local_defs local_defs in
        let assignment = pp_scope_id id ^^ string " = " ^^ value_doc ^^ semi in
        if Utils.debug_log_from_routines () && log_set_locals then
          let new_var = string "new_set_local_v" in
          let num_typ = string (B.typ_of_prec prec) in
          let decl = num_typ ^^ space ^^ new_var ^^ string " = " ^^ value_doc ^^ semi in
          let debug_val_doc, debug_args_docs = debug_float prec value in
          let debug_val_str = doc_to_string debug_val_doc in
          let pp_args_docs =
            List.map debug_args_docs ~f:(function
              | `Accessor idx -> pp_array_offset idx
              | `Value v_doc -> B.styled_log_arg v_doc)
          in
          let log_doc =
            let log_param_doc = Option.map B.kernel_log_param ~f:(fun (_, name) -> string name) in
            let scope_doc = pp_scope_id id in
            let comment_base_msg =
              "# local " ^ doc_to_string scope_doc ^ " := " ^ doc_to_string value_doc ^ "\n"
            in
            let value_base_msg =
              Printf.sprintf "%s{=%s} = %s = %s\n" (doc_to_string scope_doc) B.float_log_style
                B.float_log_style debug_val_str
            in
            let comment_log =
              B.pp_log_statement ~log_param_c_expr_doc:log_param_doc
                ~base_message_literal:comment_base_msg ~args_docs:[]
            in
            let value_log =
              B.pp_log_statement ~log_param_c_expr_doc:log_param_doc
                ~base_message_literal:value_base_msg
                ~args_docs:(B.styled_log_arg scope_doc :: B.styled_log_arg new_var :: pp_args_docs)
            in
            let flush_log =
              if B.log_involves_file_management then string "fflush(log_file);" else empty
            in
            comment_log ^^ hardline ^^ value_log ^^ hardline ^^ flush_log
          in
          let assignment' = pp_scope_id id ^^ string " = " ^^ new_var ^^ semi in
          let block_content =
            if PPrint.is_empty local_defs then
              decl ^^ hardline ^^ log_doc ^^ hardline ^^ assignment'
            else local_defs ^^ hardline ^^ decl ^^ hardline ^^ log_doc ^^ hardline ^^ assignment'
          in
          lbrace ^^ nest 2 (hardline ^^ block_content) ^^ hardline ^^ rbrace
        else if PPrint.is_empty local_defs then assignment
        else
          let block_content = local_defs ^^ hardline ^^ assignment in
          lbrace ^^ nest 2 (hardline ^^ block_content) ^^ hardline ^^ rbrace
    | Declare_local { id = { tn = { prec; _ }; _ } as id; needs_init } ->
        let scope_prec = Lazy.force prec in
        let num_typ = string (B.typ_of_prec scope_prec) in
        let init_zero =
          if needs_init then
            let prefix, postfix = B.convert_precision ~from:Ops.int32 ~to_:scope_prec in
            string " = " ^^ string prefix ^^ string "0" ^^ string postfix
          else empty
        in
        num_typ ^^ space ^^ pp_scope_id id ^^ init_zero ^^ semi
    | Workgroup_barrier -> (
        match B.barrier_syntax with
        | Some s -> string s
        | None ->
            invalid_arg
              "C_syntax.pp_ll: Workgroup_barrier not supported by this backend (serialization \
               cannot implement a barrier)")
    | Tile_mma { d; a; b; ta; tb; m; n; k; lane; fallback } -> (
        (* Cooperative tile-MMA (docs/proposals/tensorize-mma.md §4). Backends with an [mma_syntax]
           hook emit the intrinsic sequence on every lane; everywhere else (including per-call
           declines and logged runs, which must stay serial and deterministic) the scalar fallback
           runs once per simdgroup, guarded on lane 0 — the lane loop still supplies the launch's
           threads. *)
        let d_tn = fst d in
        let fragment_is_active =
          match !active_mma_accumulator with
          | Some (Active_fragment (fragment, _)) | Some (Active_target (fragment, _)) ->
              Tn.equal d_tn fragment
          | None -> false
        in
        if Set.mem !current_simdgroup_fragments d_tn && not fragment_is_active then
          declinef
            "marked simdgroup fragment %s rendered outside its recognized fragment scope; falling \
             back from the intended persistent intrinsic path"
            (Tn.debug_name d_tn);
        let operand (tn, idcs) =
          let dims = Lazy.force tn.Tn.dims in
          let prec = Lazy.force tn.Tn.prec in
          let rank = Array.length dims in
          let ld = if rank >= 1 then dims.(rank - 1) else 1 in
          let ptr_doc, ld, space =
            match !active_mma_accumulator with
            | Some (Active_fragment (fragment, name)) when Tn.equal tn fragment ->
                (string name, ld, `Fragment name)
            | Some (Active_target (fragment, (ptr, target_ld, space))) when Tn.equal tn fragment ->
                (ptr, target_ld, space)
            | _ ->
                let space =
                  if Set.mem !current_workgroup_shared tn then `Shared
                  else if Tn.Placements.is_materialized_force (placements ()) tn 440 then `Device
                  else `Thread
                in
                ( parens (string (get_ident tn) ^^ string " + " ^^ pp_array_offset (idcs, dims)),
                  ld,
                  space )
          in
          (prec, (ptr_doc, ld, space))
        in
        let lane0_guarded body_doc =
          let guarded =
            group
              (string "if (" ^^ pp_symbol lane ^^ string " == 0) " ^^ lbrace
              ^^ nest 2 (hardline ^^ body_doc)
              ^^ hardline ^^ rbrace)
          in
          (* On backends that bind the lane loop in hardware, sibling statements (e.g. the zeroing
             of [d]) execute lane-partitioned: bracket the single-lane fallback in barriers so lane
             0 observes the other lanes' writes, and they its. Uniform by the barrier-strength
             validation [Tile_mma] inherits. Serial renderers need no ordering. *)
          match B.barrier_syntax with
          | Some s -> string s ^^ hardline ^^ guarded ^^ hardline ^^ string s
          | None -> guarded
        in
        let record rendering =
          if !mma_census_enabled then
            mma_census := (!current_kernel_name, rendering) :: !mma_census
        in
        (* Shape facts for the decline diagnostics: enough to identify the statement and check every
           statically-checkable emission rule by eye. *)
        let describe () =
          let space_str = function
            | `Shared -> "shared"
            | `Device -> "device"
            | `Thread -> "thread"
            | `Fragment _ -> "fragment"
          in
          let op_str role (tn, idcs) =
            let prec, (_, ld, space) = operand (tn, idcs) in
            Printf.sprintf "%s=%s:%s[ld=%d,%s]" role (Tn.debug_name tn) (Ops.prec_string prec) ld
              (space_str space)
          in
          Printf.sprintf "%dx%dx%d ta=%b tb=%b %s %s %s" m n k ta tb (op_str "d" d) (op_str "a" a)
            (op_str "b" b)
        in
        let fallback_doc () =
          record Mma_scalar_fallback;
          lane0_guarded (pp_ll ~log_set_locals ~in_loop:true fallback)
        in
        (* Register-tiled CPU rendering (gh-ocannl-469; tinyBLAS/llamafile's [mnpack] and the S4
           micro-kernel shape): the C-tile lives in an RM×RN grid of vector-extension registers
           across the ENTIRE k-loop — per k step: RN B-row vector loads, RM A-element splats, and
           RM×RN fused-FMA updates — loaded from [d] at block entry (the statement's [+=] semantics)
           and stored back at block exit. RM = 4 rows; RN = 3 vector columns on AVX2-class
           16-register files ([vector_bytes = 32]), 6 on NEON/AVX-512-class 32-register files —
           RM×RN + RM + RN live registers, tinyBLAS's budget. Edge tiles are peeled into scalar
           loops, not masked. For each output element the k-chain runs in serial order with the same
           fused rounding, so the rendering is BITWISE equal to the scalar fallback; the plain-add
           (non-FMA) fallback form is declined — its [a * b + c] arithmetic is only
           maybe-contracted, so a vector twin could not promise that equality. Emitted under the
           same lane-0 guard as the fallback ([`Vec_extensions] backends render the lane loop
           serially; GPU backends never take this path). *)
        let try_register_tile () : PPrint.document option =
          (* [cond] names a rule violation: decline (with the per-rule diagnostic, gh-ocannl-479)
             when it holds. *)
          let no_test ~reason cond =
            if cond then (
              declinef "Tile_mma register tiling declined (%s): %s" reason (describe ());
              None)
            else Some ()
          in
          let ( let* ) o f = Option.bind o ~f in
          let* () =
            no_test ~reason:"packed-struct vector style"
              (match B.vector_style with `Vec_extensions -> false | `Packed_struct -> true)
          in
          let* () =
            no_test
              ~reason:(Printf.sprintf "vector_bytes = %d < 8" B.vector_bytes)
              (B.vector_bytes < 8)
          in
          let* () =
            no_test ~reason:"debug_log_from_routines (logged runs stay serial and deterministic)"
              (Utils.debug_log_from_routines ())
          in
          (* A transposed B ([tb]: [b[j*ldb+l]]) would turn the per-k row vector loads into
             stride-[ldb] gathers — decline it (the scalar fallback handles it; a packing [Stage]
             with [tile_loops] in micro-kernel order normalizes the layout instead). A transposed A
             ([ta]: [a[l*lda+i]]) costs nothing — the A feeds are scalar element loads (splats)
             either way — so only the index arithmetic swaps. *)
          let* () = no_test ~reason:"transposed B storage (tb)" tb in
          (* The C-tile pointers stream rows at the raw leading-dimension stride; a swizzled
             operand's elements are not where row-major arithmetic expects them. *)
          let* () =
            no_test ~reason:"swizzled operand layout"
              (List.exists [ fst d; fst a; fst b ] ~f:is_swizzled)
          in
          let d_tn = fst d in
          let prec = Lazy.force d_tn.Tn.prec in
          let* () =
            no_test
              ~reason:(Printf.sprintf "output precision %s is not f32/f64" (Ops.prec_string prec))
              (match prec with Ops.Single_prec _ | Ops.Double_prec _ -> false | _ -> true)
          in
          let* () =
            no_test ~reason:"mixed operand precisions"
              (not
                 (Ops.equal_prec (Lazy.force (fst a).Tn.prec) prec
                 && Ops.equal_prec (Lazy.force (fst b).Tn.prec) prec))
          in
          (* The fallback carries the arithmetic form; require the fused one (see above). *)
          let rec innermost_set (llc : Low_level.t) =
            match llc with
            | Low_level.For_loop { body; _ } | If { body; _ } -> innermost_set body
            | Seq (x, y) -> (
                match innermost_set x with Some _ as r -> r | None -> innermost_set y)
            | Set { tn; llsc; _ } -> Some (tn, llsc)
            | _ -> None
          in
          let* () =
            no_test ~reason:"accumulation is not in fused (FMA) form"
              (match innermost_set fallback with
              | Some (tn, Low_level.Ternop (Ops.FMA, _, _, (Low_level.Get (tn2, _), _))) ->
                  not (Tn.equal tn d_tn && Tn.equal tn2 d_tn)
              | _ -> true)
          in
          let lanes = B.vector_bytes / Ops.prec_in_bytes prec in
          let* () =
            no_test
              ~reason:(Printf.sprintf "n = %d below the vector width (lanes = %d)" n lanes)
              (lanes < 2 || n < lanes)
          in
          let rm = min 4 m in
          let rn = min (if B.vector_bytes = 32 then 3 else 6) (n / lanes) in
          let bw = rn * lanes in
          let m_full = m - (m % rm) in
          let n_full = n - (n % bw) in
          let vtyp, typedef_doc = vec_ext_typ ~prec ~lanes in
          let ctyp = B.typ_of_prec prec in
          let it = B.loop_index_type in
          let fma_fn = match prec with Ops.Double_prec _ -> "fma" | _ -> "fmaf" in
          let _, (d_ptr, ldd, _) = operand d in
          let _, (a_ptr, lda, _) = operand a in
          let _, (b_ptr, ldb, _) = operand b in
          (* The A element at (row expression, k expression), honoring [ta]'s storage order. *)
          let a_at ~row ~l =
            if ta then Printf.sprintf "tmma_a__[%s * %d + %s]" l lda row
            else Printf.sprintf "tmma_a__[%s * %d + %s]" row lda l
          in
          let stmts = separate hardline in
          (* Scalar peel of rows [i_lo, i_hi) × cols [j_lo, j_hi): same fmaf chain per element. *)
          let scalar_peel ~i_lo ~i_hi ~j_lo ~j_hi =
            if i_lo >= i_hi || j_lo >= j_hi then []
            else
              [
                string
                  (Printf.sprintf
                     "for (%stmma_i__ = %d; tmma_i__ < %d; ++tmma_i__) { for (%stmma_j__ = %d; \
                      tmma_j__ < %d; ++tmma_j__) {"
                     it i_lo i_hi it j_lo j_hi)
                ^^ nest 2
                     (hardline
                     ^^ string
                          (Printf.sprintf "%s tmma_acc__ = tmma_d__[tmma_i__ * %d + tmma_j__];" ctyp
                             ldd)
                     ^^ hardline
                     ^^ string
                          (Printf.sprintf
                             "for (%stmma_l__ = 0; tmma_l__ < %d; ++tmma_l__) tmma_acc__ = %s(%s, \
                              tmma_b__[tmma_l__ * %d + tmma_j__], tmma_acc__);"
                             it k fma_fn
                             (a_at ~row:"tmma_i__" ~l:"tmma_l__")
                             ldb)
                     ^^ hardline
                     ^^ string
                          (Printf.sprintf "tmma_d__[tmma_i__ * %d + tmma_j__] = tmma_acc__;" ldd))
                ^^ hardline ^^ string "} }";
              ]
          in
          let grid = vec_acc_grid ~prefix:"tmma_c" ~rows:rm ~cols:rn in
          let c_move ~load r c =
            let mem =
              Printf.sprintf "&tmma_d__[(tmma_i__ + %d) * %d + tmma_j__ + %d]" r ldd (c * lanes)
            in
            let reg = "&" ^ grid.(r).(c) in
            let src, dst = if load then (mem, reg) else (reg, mem) in
            string (Printf.sprintf "__builtin_memcpy(%s, %s, sizeof(%s));" dst src grid.(r).(c))
          in
          let full_blocks =
            if m_full = 0 || n_full = 0 then []
            else
              let k_body =
                List.init rn ~f:(fun c ->
                    string
                      (Printf.sprintf
                         "%s tmma_b_%d__; __builtin_memcpy(&tmma_b_%d__, &tmma_b__[tmma_l__ * %d + \
                          tmma_j__ + %d], sizeof(tmma_b_%d__));"
                         vtyp c c ldb (c * lanes) c))
                @ List.concat
                    (List.init rm ~f:(fun r ->
                         string
                           (Printf.sprintf "%s tmma_a_%d__ = ((%s){0} + %s);" vtyp r vtyp
                              (a_at ~row:(Printf.sprintf "(tmma_i__ + %d)" r) ~l:"tmma_l__"))
                         :: List.init rn ~f:(fun c ->
                             vec_acc_fma ~prec ~lanes
                               ~dst:grid.(r).(c)
                               ~a:(Printf.sprintf "tmma_a_%d__" r)
                               ~b:(Printf.sprintf "tmma_b_%d__" c))))
              in
              let per_cell f =
                List.concat (List.init rm ~f:(fun r -> List.init rn ~f:(fun c -> f r c)))
              in
              [
                string
                  (Printf.sprintf "for (%stmma_i__ = 0; tmma_i__ + %d <= %d; tmma_i__ += %d) {" it
                     rm m rm)
                ^^ nest 2
                     (hardline
                     ^^ string
                          (Printf.sprintf
                             "for (%stmma_j__ = 0; tmma_j__ + %d <= %d; tmma_j__ += %d) {" it bw n
                             bw)
                     ^^ nest 2
                          (hardline
                          ^^ stmts
                               (per_cell (fun r c ->
                                    string (Printf.sprintf "%s %s;" vtyp grid.(r).(c))
                                    ^^ space ^^ c_move ~load:true r c))
                          ^^ hardline
                          ^^ string
                               (Printf.sprintf "for (%stmma_l__ = 0; tmma_l__ < %d; ++tmma_l__) {"
                                  it k)
                          ^^ nest 2 (hardline ^^ stmts k_body)
                          ^^ hardline ^^ string "}" ^^ hardline
                          ^^ stmts (per_cell (fun r c -> c_move ~load:false r c)))
                     ^^ hardline ^^ string "}")
                ^^ hardline ^^ string "}";
              ]
          in
          let body =
            [
              typedef_doc;
              string (Printf.sprintf "%s *tmma_d__ = " ctyp) ^^ d_ptr ^^ semi;
              string (Printf.sprintf "const %s *tmma_a__ = " ctyp) ^^ a_ptr ^^ semi;
              string (Printf.sprintf "const %s *tmma_b__ = " ctyp) ^^ b_ptr ^^ semi;
            ]
            @ full_blocks
            @ scalar_peel ~i_lo:0 ~i_hi:m_full ~j_lo:n_full ~j_hi:n
            @ scalar_peel ~i_lo:m_full ~i_hi:m ~j_lo:0 ~j_hi:n
          in
          Some
            (string
               (Printf.sprintf
                  "{ /* Tile_mma register tiling: %dx%d C-tile of %d-lane %s held across the \
                   k-loop (full blocks %dx%d of %dx%d). */"
                  rm rn lanes ctyp m_full n_full m n)
            ^^ nest 2 (hardline ^^ stmts body)
            ^^ hardline ^^ string "}")
        in
        let fallback_or_tiled () =
          match try_register_tile () with
          | Some doc ->
              record Mma_register_tiled;
              lane0_guarded doc
          | None ->
              declinef "Tile_mma renders the lane-0 scalar fallback: %s" (describe ());
              fallback_doc ()
        in
        match B.mma_syntax with
        | None -> fallback_or_tiled ()
        | Some _ when Utils.debug_log_from_routines () ->
            declinef
              "Tile_mma intrinsics skipped (debug_log_from_routines: logged runs stay serial and \
               deterministic): %s"
              (describe ());
            fallback_doc ()
        | Some _ when List.exists [ fst d; fst a; fst b ] ~f:is_swizzled ->
            (* The intrinsic loads assume row-major pointer+stride operands; the scalar fallback
               reads elementwise through the swizzle-aware offsets and stays correct. *)
            declinef "Tile_mma intrinsics declined (swizzled operand layout): %s" (describe ());
            fallback_or_tiled ()
        | Some emit -> (
            let d_prec, d_op = operand d in
            let a_prec, a_op = operand a in
            let b_prec, b_op = operand b in
            match emit ~d_prec ~a_prec ~b_prec ~ta ~tb ~m ~n ~k ~d:d_op ~a:a_op ~b:b_op with
            | Some doc ->
                record Mma_intrinsics;
                doc
            | None ->
                declinef "Tile_mma intrinsics declined by the backend hook: %s" (describe ());
                fallback_or_tiled ()))
    | If { cond = c, cprec; body } ->
        (* Guarded statement (axis-types proposal §2): [body] executes iff [cond] is nonzero -- C's
           [if] tests exactly that. *)
        let local_defs, cond_doc = pp_scalar cprec c in
        let local_defs = pp_local_defs local_defs in
        let body_doc = pp_ll ~log_set_locals ~in_loop:true body in
        let if_doc =
          group
            (string "if (" ^^ cond_doc ^^ string ") " ^^ lbrace
            ^^ nest 2 (hardline ^^ body_doc)
            ^^ hardline ^^ rbrace)
        in
        if PPrint.is_empty local_defs then if_doc
        else lbrace ^^ nest 2 (hardline ^^ local_defs ^^ hardline ^^ if_doc) ^^ hardline ^^ rbrace

  and pp_scalar (prec : Ops.prec) (vcomp : Low_level.scalar_t) :
      (int * PPrint.document) list * PPrint.document =
    (* Returns (local definitions, value expression) *)
    let open PPrint in
    match vcomp with
    | Local_scope { id = { tn = { prec = scope_prec; _ }; scope_id } as id; body; orig_indices = _ }
      ->
        let scope_prec = Lazy.force scope_prec in
        let num_typ = string (B.typ_of_prec scope_prec) in
        let init_zero =
          if Low_level.reads_scope_before_set id body then
            let prefix, postfix = B.convert_precision ~from:Ops.int32 ~to_:scope_prec in
            string " = " ^^ string prefix ^^ string "0" ^^ string postfix
          else empty
        in
        let decl = num_typ ^^ space ^^ pp_scope_id id ^^ init_zero ^^ semi in
        (* A [Local_scope] body is a nested sub-computation; conservatively treat it like a loop
           body so a [Zero_out] reached through it is never mistaken for the function-scope
           first-touch that the declaration's [= {0}] covers. *)
        let body_doc = pp_ll ~log_set_locals:false ~in_loop:true body in
        let def_doc = decl ^^ hardline ^^ body_doc in
        let prefix, postfix = B.convert_precision ~from:scope_prec ~to_:prec in
        let expr = string prefix ^^ pp_scope_id id ^^ string postfix in
        ([ (scope_id, def_doc) ], expr)
    | Get_local id ->
        let scope_prec = Lazy.force id.tn.prec in
        let prefix, postfix = B.convert_precision ~from:scope_prec ~to_:prec in
        let expr = string prefix ^^ pp_scope_id id ^^ string postfix in
        ([], expr)
    | Get_merge_buffer (source, idcs) ->
        let tn = source in
        let dims = Lazy.force tn.dims in
        let from_prec = Lazy.force tn.prec in
        let prefix, postfix = B.convert_precision ~from:from_prec ~to_:prec in
        let offset_doc = pp_array_offset (idcs, dims) in
        let expr =
          string prefix ^^ string "merge_buffer" ^^ brackets offset_doc ^^ string postfix
        in
        ([], expr)
    | Get (tn, idcs) ->
        let ident_doc = string (get_ident tn) in
        let dims = Lazy.force tn.dims in
        let from_prec = Lazy.force tn.prec in
        let prefix, postfix = B.convert_precision ~from:from_prec ~to_:prec in
        let offset_doc = pp_tn_offset tn (idcs, dims) in
        let expr = string prefix ^^ ident_doc ^^ brackets offset_doc ^^ string postfix in
        ([], expr)
    | Get_dynamic { tn; idcs; dyn_axis; dyn_value = iv, iprec } ->
        (* gh-343: a guarded dynamic gather. The dynamic index is spliced into the row-major offset
           at [dyn_axis] as an integer; the enclosing [Where] guard (when interval analysis has not
           discharged it against proven bounds) guarantees it is in range before this load is
           evaluated (C ternary short-circuits). Cast to [Ops.index_prec ()] so the index tracks the
           same width as loop counters (signed int32 normally, int64 under large_models), preventing
           truncation for very large table/vocabulary axes. *)
        if is_swizzled tn then
          invalid_arg
            ("C_syntax: Get_dynamic reads swizzled node " ^ Tn.debug_name tn
           ^ " (dynamic offsets are not swizzle-remapped)");
        let ident_doc = string (get_ident tn) in
        let dims = Lazy.force tn.dims in
        let from_prec = Lazy.force tn.prec in
        let prefix, postfix = B.convert_precision ~from:from_prec ~to_:prec in
        let dyn_defs, dyn_expr = pp_scalar iprec iv in
        let idx_typ = B.typ_of_prec (Ops.index_prec ()) in
        let dyn_idx_doc = string ("((" ^ idx_typ ^ ")(") ^^ dyn_expr ^^ string "))" in
        let offset_doc = pp_array_offset_dyn (idcs, dims) ~dyn_axis ~dyn_idx_doc in
        let expr = string prefix ^^ ident_doc ^^ brackets offset_doc ^^ string postfix in
        (dyn_defs, expr)
    | Constant c ->
        let from_prec = Ops.double in
        let prefix, postfix = B.convert_precision ~from:from_prec ~to_:prec in
        let c_str =
          if Float.(c = infinity) then "INFINITY"
          else if Float.(c = neg_infinity) then "(-INFINITY)"
          else if Float.is_nan c then "NAN"
          else Printf.sprintf "%.16g" c
        in
        let expr =
          if String.is_empty prefix && Float.(c < 0.0) && not Float.(c = neg_infinity) then
            string "(" ^^ string c_str ^^ string ")" ^^ string postfix
          else string prefix ^^ string c_str ^^ string postfix
        in
        ([], expr)
    | Constant_bits i ->
        let from_prec = Ops.int64 in
        let prefix, postfix = B.convert_precision ~from:from_prec ~to_:prec in
        let expr = string prefix ^^ string (Printf.sprintf "%LdLL" i) ^^ string postfix in
        ([], expr)
    | Embed_index idx ->
        let from_prec = Ops.index_prec () in
        let prefix, postfix = B.convert_precision ~from:from_prec ~to_:prec in
        let idx_doc = pp_axis_index idx in
        let idx_doc = if PPrint.is_empty idx_doc then string "0" else idx_doc in
        let expr = string prefix ^^ idx_doc ^^ string postfix in
        ([], expr)
    | Binop (Arg1, (v1, _), _v2) -> pp_scalar prec v1
    | Binop (Arg2, _v1, (v2, _)) -> pp_scalar prec v2
    | Ternop (op, (v1, v1_prec), (v2, v2_prec), (v3, v3_prec)) ->
        let d1, e1, d2, e2, d3, e3 =
          if Ops.is_homogeneous_prec_ternop op then
            (* Homogeneous: all arguments use result precision *)
            let d1, e1 = pp_scalar prec v1 in
            let d2, e2 = pp_scalar prec v2 in
            let d3, e3 = pp_scalar prec v3 in
            (d1, e1, d2, e2, d3, e3)
          else
            (* Heterogeneous: arguments keep their natural precision *)
            match op with
            | Ops.Where ->
                (* For Where: condition keeps its precision, then/else use result precision *)
                (* Note: we evaluate condition without precision conversion, but then/else
                   need to match the result precision for the final assignment *)
                let d1, e1 = pp_scalar v1_prec v1 in
                (* condition: no conversion *)
                let d2, e2 = pp_scalar prec v2 in
                (* then: result precision *)
                let d3, e3 = pp_scalar prec v3 in
                (* else: result precision *)
                (d1, e1, d2, e2, d3, e3)
            | _ ->
                (* Other heterogeneous ternary ops would go here *)
                let d1, e1 = pp_scalar v1_prec v1 in
                let d2, e2 = pp_scalar v2_prec v2 in
                let d3, e3 = pp_scalar v3_prec v3 in
                (d1, e1, d2, e2, d3, e3)
        in
        let defs = List.concat [ d1; d2; d3 ] in
        let expr = group (B.ternop_syntax prec op e1 e2 e3) in
        (defs, expr)
    | Binop (op, (v1, v1_prec), (v2, v2_prec)) ->
        let d1, e1, d2, e2 =
          if Ops.is_homogeneous_prec_binop op then
            (* Homogeneous: both arguments use result precision *)
            let d1, e1 = pp_scalar prec v1 in
            let d2, e2 = pp_scalar prec v2 in
            (d1, e1, d2, e2)
          else
            (* Heterogeneous: arguments keep their natural precision *)
            (* Currently all binops are homogeneous, but this is here for future extension *)
            let d1, e1 = pp_scalar v1_prec v1 in
            let d2, e2 = pp_scalar v2_prec v2 in
            (d1, e1, d2, e2)
        in
        let defs = List.concat [ d1; d2 ] in
        let expr = group (B.binop_syntax prec op e1 e2) in
        (defs, expr)
    | Unop (op, (v, v_prec)) ->
        let arg_prec =
          if Ops.is_homogeneous_prec_unop op then prec
            (* Homogeneous: argument uses result precision *)
          else v_prec
        in
        let defs, expr_v = pp_scalar arg_prec v in
        let expr = group (B.unop_syntax prec op expr_v) in
        (defs, expr)

  and debug_float ?guard (prec : Ops.prec) (value : Low_level.scalar_t) :
      PPrint.document
      * [ `Accessor of Indexing.axis_index array * int array | `Value of PPrint.document ] list =
    (* Returns (value expression doc, list of arguments for printf).

       [guard] (task-9658aac9): when [Some cond_c] (a real C boolean expression), any array
       dereference produced here is only safe when [cond_c] holds, so its printf [`Value] argument
       is short-circuited as [(cond_c ? read : 0)]. This mirrors the runtime [Where] ternary's
       short-circuiting and closes the only debug-logging path that would otherwise dereference a
       conditionally-evaluated branch (e.g. the Stage B unit-solve then-branch producer read) out of
       bounds -- the same hazard gh-343 fixed for the sibling [Get_dynamic] gather. Only the
       dereferencing argument is gated; the displayed expression doc is unchanged. *)
    let open PPrint in
    let guarded_value access_doc =
      match guard with
      | None -> access_doc
      | Some g -> parens (g ^^ string " ? " ^^ access_doc ^^ string " : 0")
    in
    match value with
    | Local_scope { id; _ } ->
        (* Not printing the inlined definition: (1) code complexity; (2) don't overload the debug
           logs. *)
        debug_float prec @@ Get_local id
    | Get_local id ->
        let scope_prec = Lazy.force id.tn.prec in
        let prefix, postfix = B.convert_precision ~from:scope_prec ~to_:prec in
        let v_doc = string prefix ^^ pp_scope_id id ^^ string postfix in
        (v_doc ^^ braces (string ("=" ^ B.float_log_style)), [ `Value v_doc ])
    | Get_merge_buffer (source, idcs) ->
        let tn = source in
        let dims = Lazy.force tn.dims in
        let from_prec = Lazy.force tn.prec in
        let prefix, postfix = B.convert_precision ~from:from_prec ~to_:prec in
        let offset_doc = pp_array_offset (idcs, dims) in
        let access_doc =
          string prefix ^^ string "merge_buffer" ^^ brackets offset_doc ^^ string postfix
        in
        let expr_doc =
          string prefix ^^ string "merge_buffer"
          ^^ brackets (string "%u")
          ^^ string postfix
          ^^ braces (string ("=" ^ B.float_log_style))
        in
        (expr_doc, [ `Accessor (idcs, dims); `Value (guarded_value access_doc) ])
    | Get (tn, idcs) ->
        let ident_doc = string (get_ident tn) in
        let dims = Lazy.force tn.dims in
        let from_prec = Lazy.force tn.prec in
        let prefix, postfix = B.convert_precision ~from:from_prec ~to_:prec in
        let offset_doc = pp_tn_offset tn (idcs, dims) in
        let access_doc = string prefix ^^ ident_doc ^^ brackets offset_doc ^^ string postfix in
        let expr_doc =
          string prefix ^^ ident_doc
          ^^ brackets (string "%u")
          ^^ string postfix
          ^^ braces (string ("=" ^ B.float_log_style))
        in
        (expr_doc, [ `Accessor (idcs, dims); `Value (guarded_value access_doc) ])
    | Get_dynamic { tn; dyn_value = iv, iprec; _ } ->
        (* gh-343: do NOT dereference the table in debug logs. A [Where]'s [debug_float] collects
           all three branch values as printf arguments evaluated unconditionally, so returning the
           raw [table[((idx_typ)(idx))]] access here would read out of bounds for ids the
           surrounding guard is meant to exclude. Log the (always-safe) dynamic index value
           instead. *)
        let prefix, postfix = B.convert_precision ~from:iprec ~to_:prec in
        let _defs, idx_e = pp_scalar iprec iv in
        let idx_doc = string prefix ^^ idx_e ^^ string postfix in
        let label = string (get_ident tn ^ "@dyn_idx") in
        (label ^^ braces (string ("=" ^ B.float_log_style)), [ `Value idx_doc ])
    | Constant c ->
        let from_prec = Ops.double in
        let prefix, postfix = B.convert_precision ~from:from_prec ~to_:prec in
        let c_str =
          if Float.(c = infinity) then "INFINITY"
          else if Float.(c = neg_infinity) then "(-INFINITY)"
          else if Float.is_nan c then "NAN"
          else Printf.sprintf "%.16g" c
        in
        (string prefix ^^ string c_str ^^ string postfix, [])
    | Constant_bits i ->
        let from_prec = Ops.int64 in
        let prefix, postfix = B.convert_precision ~from:from_prec ~to_:prec in
        let expr = string prefix ^^ string (Printf.sprintf "%LdLL" i) ^^ string postfix in
        (expr, [])
    | Embed_index idx ->
        let idx_doc = pp_axis_index idx in
        ((if PPrint.is_empty idx_doc then string "0" else idx_doc), [])
    | Binop (Arg1, (v1, _), _v2) -> debug_float ?guard prec v1
    | Binop (Arg2, _v1, (v2, _)) -> debug_float ?guard prec v2
    | Ternop (op, (v1, v1_prec), (v2, v2_prec), (v3, v3_prec)) ->
        let v1_doc, idcs1, v2_doc, idcs2, v3_doc, idcs3 =
          if Ops.is_homogeneous_prec_ternop op then
            (* Homogeneous: all arguments use result precision *)
            let v1_doc, idcs1 = debug_float ?guard prec v1 in
            let v2_doc, idcs2 = debug_float ?guard prec v2 in
            let v3_doc, idcs3 = debug_float ?guard prec v3 in
            (v1_doc, idcs1, v2_doc, idcs2, v3_doc, idcs3)
          else
            (* Heterogeneous: handle based on operation *)
            match op with
            | Ops.Where ->
                (* task-9658aac9: short-circuit array-read leaves on the live [Where] condition so
                   debug value-logging never dereferences a conditionally-evaluated branch out of
                   bounds (the Stage B unit-solve then-branch producer read; gh-343's hazard class
                   for the sibling [Get_dynamic]). The displayed ternary is unchanged -- only the
                   dereferencing printf arguments are gated: then-reads under [cond], else-reads
                   under [!cond], each AND-composed with any enclosing [guard] so nested/triangular
                   range guards compose. [cond_c] is rendered twice -- via [pp_scalar] for the real
                   C guard expression here, and via [debug_float] for the annotated display
                   below. *)
                let _cond_defs, cond_c = pp_scalar v1_prec v1 in
                (* Conditions reaching a Where here are pure index/value comparisons (range guards,
                   in_range) with no Local_scope, so [_cond_defs] is empty. Even if a future
                   condition inlined a Local_scope, dropping the redundant re-definition is safe:
                   the local is already bound by the assignment computation that precedes the log
                   statement (the same invariant debug_float's Local_scope arm relies on). *)
                let compose outer c =
                  match outer with None -> c | Some g -> parens (g ^^ string " && " ^^ c)
                in
                let then_guard = compose guard cond_c in
                let else_guard = compose guard (parens (string "!" ^^ parens cond_c)) in
                let v1_doc, idcs1 = debug_float ?guard v1_prec v1 in
                (* condition: no precision conversion. It is evaluated whenever the enclosing branch
                   is reached, so its array reads (a nested [Where] condition may contain a [Get])
                   are gated by the incoming [guard] -- not by [cond_c], which the condition itself
                   computes. At the top level [guard = None], so this is a no-op for the common
                   pure-index-comparison case. *)
                let v2_doc, idcs2 = debug_float ~guard:then_guard prec v2 in
                (* then: result precision, gated by [cond] *)
                let v3_doc, idcs3 = debug_float ~guard:else_guard prec v3 in
                (* else: result precision, gated by [!cond] *)
                (v1_doc, idcs1, v2_doc, idcs2, v3_doc, idcs3)
            | _ ->
                let v1_doc, idcs1 = debug_float ?guard v1_prec v1 in
                let v2_doc, idcs2 = debug_float ?guard v2_prec v2 in
                let v3_doc, idcs3 = debug_float ?guard v3_prec v3 in
                (v1_doc, idcs1, v2_doc, idcs2, v3_doc, idcs3)
        in
        (B.ternop_syntax prec op v1_doc v2_doc v3_doc, idcs1 @ idcs2 @ idcs3)
    | Binop (op, (v1, v1_prec), (v2, v2_prec)) ->
        let v1_doc, idcs1, v2_doc, idcs2 =
          if Ops.is_homogeneous_prec_binop op then
            (* Homogeneous: both arguments use result precision *)
            let v1_doc, idcs1 = debug_float ?guard prec v1 in
            let v2_doc, idcs2 = debug_float ?guard prec v2 in
            (v1_doc, idcs1, v2_doc, idcs2)
          else
            (* Heterogeneous: arguments keep their natural precision *)
            let v1_doc, idcs1 = debug_float ?guard v1_prec v1 in
            let v2_doc, idcs2 = debug_float ?guard v2_prec v2 in
            (v1_doc, idcs1, v2_doc, idcs2)
        in
        (B.binop_syntax prec op v1_doc v2_doc, idcs1 @ idcs2)
    | Unop (op, (v, v_prec)) ->
        let arg_prec =
          if Ops.is_homogeneous_prec_unop op then prec
            (* Homogeneous: argument uses result precision *)
          else v_prec
        in
        let v_doc, idcs = debug_float ?guard arg_prec v in
        (B.unop_syntax prec op v_doc, idcs)

  let compile_main llc : PPrint.document = pp_ll llc

  let compile_proc ~name idx_params
      Low_level.
        {
          traced_store;
          llc;
          merge_node;
          optimize_ctx;
          workgroup_shared;
          simdgroup_fragments;
          swizzled;
        } : (string * kparam_source) list * PPrint.document * Low_level.launch_dims =
    let open PPrint in
    (if not (Set.is_empty workgroup_shared) then
       match B.shared_decl_prefix with
       | Some _ -> ()
       | None ->
           (* The local-declaration pass below would silently emit workgroup-shared nodes as
              per-thread stack arrays -- wrong sharing semantics. *)
           invalid_arg
             "C_syntax.compile_proc: workgroup-shared placement not supported by this backend");
    current_kernel_name := name;
    current_placements := Some optimize_ctx.Low_level.placements;
    Low_level.validate_parallel optimize_ctx.Low_level.placements llc;
    (* Launch-extent guards (construct-then-fold, axis-types proposal §2), only for kinds this
       backend binds in hardware -- the serial fallback iterates the true extent. *)
    let llc =
      Low_level.guard_annotated_extents
        ~should_guard:(fun kind -> Option.is_some (B.hardware_index ~kind ~slot:0))
        llc
    in
    let launch = Low_level.launch_dims llc in
    current_hardware_axes := Low_level.hardware_axes llc;
    (let parallel_grid, grid_private, local_ptr_alias = collect_parallel_grid llc in
     current_parallel_grid := parallel_grid;
     current_grid_private := grid_private;
     current_local_ptr_alias := local_ptr_alias);
    current_workgroup_shared := workgroup_shared;
    current_simdgroup_fragments := simdgroup_fragments;
    current_swizzled := swizzled;
    rendered_simdgroup_fragments := Set.empty (module Tn);
    current_traced_store := Some traced_store;
    Hash_set.clear zero_out_seen;
    (* The materialized in-context nodes, in deterministic [traced_store] order, with their
       per-param pointer declaration (used by the [`Per_param] style). *)
    let ptr_params : (string * Tn.t) list =
      List.rev
      @@ Hashtbl.fold traced_store ~init:[] ~f:(fun ~key:tn ~data:_ acc ->
          let backend_info, is_param =
            let plc = placements () in
            if Tn.Placements.is_virtual_force plc tn 334 then ("Virt", false)
            else if in_ctx tn then ("Ctx", true)
            else if Tn.Placements.is_materialized_force plc tn 335 then ("Global", true)
            else if Tn.Placements.known_not_materialized plc tn then ("Local", false)
            else assert false
          in
          let backend_info = Sexp.Atom backend_info in
          if not @@ Utils.sexp_mem ~elem:backend_info tn.backend_info then
            tn.backend_info <- Utils.sexp_append ~elem:backend_info tn.backend_info;
          if is_param then (
            (* Assignments lowering rewrites every alias-view access to a parent access, so alias
               tnodes never reach this parameter list — but hand-built [Low_level.t] (schedule
               layer, tests) could mint one, and with [restrict_keyword] an aliased parameter pair
               is a miscompile rather than a redundant pointer. Fail loudly (gh-ocannl-164). *)
            if Tn.is_alias tn then
              invalid_arg
                ("C_syntax.compile_proc: alias view " ^ Tn.debug_name tn
               ^ " as a kernel parameter: accesses must be rewritten to the buffer-owning parent \
                  (aliased parameters would falsify the restrict qualifier)");
            (* gh-ocannl-489: an aliasing candidate may be placed at bytes overlapping another
               parameter's by the link-time liveness planner, so its pointer must not promise
               [restrict] -- whether a candidate pair actually shares bytes is unknowable at
               codegen time. *)
            let restrict_ =
              match B.restrict_keyword with
              | Some kw when not (Hash_set.mem optimize_ctx.Low_level.alias_candidates tn) ->
                  kw ^ " "
              | _ -> ""
            in
            (B.typ_of_prec (Lazy.force tn.Tn.prec) ^ " *" ^ restrict_ ^ get_ident tn, tn) :: acc)
          else acc)
    in
    (* [`Per_param]: one typed pointer param per node (C/CUDA, byte-identical to before). [`Pooled
       n]: [n] byte-pointer pool params + one slot table (Metal binding fix). *)
    let kparams : (string * kparam_source) list =
      match B.ptr_param_style with
      | `Per_param -> List.map ptr_params ~f:(fun (decl, tn) -> (decl, Kparam_ptr tn))
      | `Pooled n_pools -> (
          match ptr_params with
          | [] -> []
          | _ ->
              let slabs =
                List.init n_pools ~f:(fun i ->
                    (Printf.sprintf "char* __pool%d" i, Kparam_pool_slab i))
              in
              let slots =
                ( Printf.sprintf "const %s* __pool_slots" (pool_slot_msl_typ ()),
                  Kparam_pool_slots (List.map ptr_params ~f:snd) )
              in
              slabs @ [ slots ])
    in
    let idx_params =
      List.map idx_params ~f:(fun s ->
          (B.arg_int_prefix ^ Indexing.symbol_ident s.Indexing.static_symbol, Static_idx s))
    in
    let log_file_param =
      if Utils.debug_log_from_routines () then
        match B.kernel_log_param with
        | Some (typ, name) -> [ (typ ^ " " ^ name, Log_file_name) ]
        | None -> []
      else []
    in
    let merge_param =
      Option.(
        to_list
        @@ map merge_node ~f:(fun tn ->
            ("const " ^ B.typ_of_prec (Lazy.force tn.prec) ^ " *merge_buffer", Merge_buffer)))
    in
    let all_params = log_file_param @ merge_param @ idx_params @ kparams in
    let sorted_params =
      List.sort all_params ~compare:(fun (p1_name, _) (p2_name, _) ->
          compare_string p1_name p2_name)
    in
    let args_docs =
      List.mapi sorted_params ~f:(fun pos (name, _) ->
          string (B.buffer_prefix ^ name ^ B.buffer_suffix ~pos))
      @ List.map B.extra_args ~f:string
    in
    let func_header =
      string B.main_kernel_prefix ^^ space ^^ string "void" ^^ space ^^ string name
      ^^ nest 4 (lparen ^^ hardline ^^ separate (comma ^^ hardline) args_docs ^^ rparen)
    in
    let body = ref empty in
    if not (String.is_empty B.kernel_prep_line) then
      body := !body ^^ string B.kernel_prep_line ^^ semi ^^ hardline;

    if Utils.debug_log_from_routines () && B.log_involves_file_management then
      let log_file_var_name =
        match B.kernel_log_param with
        | Some (_, name) -> name
        | None -> "log_file_name" (* Should ideally not be reached if management is true *)
      in
      body :=
        !body ^^ string "FILE* log_file = NULL;" ^^ hardline
        ^^ string ("if (" ^ log_file_var_name ^ ") ")
        ^^ lbrace
        ^^ nest 2 (hardline ^^ string ("log_file = fopen(" ^ log_file_var_name ^ ", \"w\");"))
        ^^ hardline ^^ rbrace ^^ hardline
    else body := !body ^^ hardline;

    (if Utils.debug_log_from_routines () then
       let debug_init_doc =
         string "/* Debug initial parameter state. */"
         ^^ hardline
         ^^ separate_map hardline
              (fun (p_name_and_type, source) ->
                let log_param_doc =
                  Option.map B.kernel_log_param ~f:(fun (_, name) -> string name)
                in
                match source with
                | Merge_buffer ->
                    let merge_tn = Option.value_exn ~here:[%here] merge_node in
                    let base_msg =
                      Printf.sprintf "%s &[%d] = %%p\n" p_name_and_type (Tnode.num_elems merge_tn)
                    in
                    B.pp_log_statement ~log_param_c_expr_doc:log_param_doc
                      ~base_message_literal:base_msg
                      ~args_docs:[ string @@ "(" ^ B.buffer_prefix ^ "void*)merge_buffer" ]
                | Log_file_name -> empty (* Already handled by fopen or if it's just an ID *)
                | Kparam_ptr tn ->
                    let base_msg =
                      Printf.sprintf "%s &[%d] = %%p\n" p_name_and_type (Tnode.num_elems tn)
                    in
                    let ident_doc = string (get_ident tn) in
                    B.pp_log_statement ~log_param_c_expr_doc:log_param_doc
                      ~base_message_literal:base_msg
                      ~args_docs:[ string ("(" ^ B.buffer_prefix ^ "void*)") ^^ ident_doc ]
                | Static_idx s ->
                    let base_msg = Printf.sprintf "%s = %%d\n" p_name_and_type in
                    let ident_doc = pp_symbol s.static_symbol in
                    B.pp_log_statement ~log_param_c_expr_doc:log_param_doc
                      ~base_message_literal:base_msg ~args_docs:[ ident_doc ]
                | Kparam_pool_slab _ | Kparam_pool_slots _ ->
                    (* Pooled (Metal): the per-node pointers are materialized as locals in the pool
                       prologue and logged at their first [Set]; the pool bases / slot table
                       themselves are not separately logged here. *)
                    empty)
              sorted_params
       in
       body := !body ^^ debug_init_doc ^^ hardline);

    (* Pooled (Metal) prologue: build the local pool-base array from the bound pool params, then
       form each materialized node's typed pointer from its slot. Indices match the
       [Kparam_pool_slots] order (= [ptr_params]). The rest of the body indexes [get_ident tn]
       exactly as in the per-param style. *)
    (match B.ptr_param_style with
    | `Per_param -> ()
    | `Pooled n_pools when not (List.is_empty ptr_params) ->
        let pools_decl =
          Printf.sprintf "%schar* __pools[%d] = { %s };" B.buffer_prefix n_pools
            (String.concat ~sep:", " (List.init n_pools ~f:(Printf.sprintf "__pool%d")))
        in
        let defs =
          (* The derived per-node pointers address disjoint slab sub-ranges, and the kernel body
             accesses nodes only through them (never through the pool bases), which is all the
             restrict qualifier asserts (gh-ocannl-164). The liveness planner (gh-ocannl-489)
             preserves the disjointness for pooled backends: they use segment-granularity liveness,
             under which any two nodes accessed by ONE kernel have overlapping live spans and are
             therefore never placed at overlapping offsets. *)
          let restrict_ = match B.restrict_keyword with Some kw -> kw ^ " " | None -> "" in
          List.mapi ptr_params ~f:(fun k (_decl, tn) ->
              let typ = B.typ_of_prec (Lazy.force tn.Tn.prec) in
              Printf.sprintf "%s%s* %s%s = (%s%s*)(__pools[__pool_slots[%d]] + __pool_slots[%d]);"
                B.buffer_prefix typ restrict_ (get_ident tn) B.buffer_prefix typ (2 * k)
                ((2 * k) + 1))
        in
        body :=
          !body
          ^^ string "/* Pool base pointers. */"
          ^^ hardline
          ^^ separate_map hardline string (pools_decl :: defs)
          ^^ hardline
    | `Pooled _ -> ());

    (* Render before declarations so accepted marked-fragment regions can suppress their otherwise
       dead scalar local arrays. Declining/debug paths leave the set empty and keep the arrays. *)
    let main_logic_doc = compile_main llc in
    let grid_privatized =
      Map.fold !current_grid_private
        ~init:(Set.empty (module Tn))
        ~f:(fun ~key:_ ~data acc -> List.fold data ~init:acc ~f:Set.add)
    in
    let local_decls =
      string "/* Local declarations and initialization. */"
      ^^ hardline
      ^^ separate_map empty
           (fun (tn, node) ->
             let plc = placements () in
             if
               (not
                  (Tn.Placements.is_virtual_force plc tn 333
                  || Tn.Placements.is_materialized_force plc tn 336))
               (* Privatized to a pool-parallel [Grid] loop: declared per chunk inside that loop's
                  body instead (see [parallel_grid_loop]). *)
               && (not (Set.mem grid_privatized tn))
               && not (Set.mem !rendered_simdgroup_fragments tn)
             then
               let is_shared = Set.mem workgroup_shared tn in
               if is_shared then
                 (* Workgroup-shared placement (axis-types proposal §3): one tile per workgroup
                    instead of one per thread. [= {0}] is not allowed for shared declarations, so
                    zero-initialization stays as explicit [Zero_out] code (never elided for shared
                    nodes; see [zero_out_loop_redundant]); shared placements also keep the backend's
                    default layout (no [aligned_local_attr]). *)
                 string (Option.value_exn ~here:[%here] B.shared_decl_prefix)
                 ^^ string (B.typ_of_prec @@ Lazy.force tn.prec)
                 ^^ space
                 ^^ string (get_ident tn)
                 ^^ brackets (OCaml.int (Tn.num_elems tn))
                 ^^ semi ^^ hardline
               else
                 local_array_decl
                   ~alias_ptr:(Set.mem !current_local_ptr_alias tn)
                   ~zero_init:node.Low_level.zero_initialized_by_code tn
                 ^^ hardline
             else empty)
           (Hashtbl.to_alist traced_store)
    in
    body := !body ^^ local_decls ^^ hardline;

    let main_logic = string "/* Main logic. */" ^^ hardline ^^ main_logic_doc in
    body := !body ^^ main_logic;

    if Utils.debug_log_from_routines () && B.log_involves_file_management then
      body :=
        !body ^^ hardline
        ^^ string "if (log_file) { fclose(log_file); log_file = NULL; }"
        ^^ hardline;

    let func_doc =
      func_header ^^ space ^^ lbrace ^^ nest 2 (hardline ^^ !body) ^^ hardline ^^ rbrace
    in
    (sorted_params, func_doc, launch)
end
