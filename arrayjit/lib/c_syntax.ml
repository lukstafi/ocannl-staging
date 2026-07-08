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
      docs/proposals/axis-types-for-loops.md §2/§5). Slots are positional: 0 = [.x], 1 = [.y],
      2 = [.z]. *)

  val barrier_syntax : string option
  (** Workgroup barrier statement ([__syncthreads();] / [threadgroup_barrier(...);]); [None] makes
      [Workgroup_barrier] a compile-time error (serialization cannot implement a barrier). *)

  val parallel_grid_syntax : [ `None | `Dispatch | `Openmp ]
  (** Pool-backed [Grid] rendering (docs/proposals/gh-ocannl-164.md): how to render an eligible
      outermost [Grid] loop when [hardware_index] does not bind it. [`Dispatch] emits libdispatch's
      [dispatch_apply] over contiguous chunks (macOS; blocks extension), [`Openmp] a
      [#pragma omp parallel for] over the chunk loop; both runtimes own a single process-global
      thread pool, so no pool state lives in the compiled kernel. [`None] keeps the serial
      fallback. Eligibility is decided per loop by [compile_proc] (see [parallel_grid_safe]);
      [Workgroup] loops always stay serial inside a chunk. *)

  val parallel_grid_chunks : int
  (** Target chunk count for [parallel_grid_syntax] (e.g. a small multiple of the core count); the
      actual count is capped by the loop extent. Values [<= 1] disable parallel rendering. *)

  val shared_decl_prefix : string option
  (** Declaration prefix for workgroup-shared placements ([__shared__ ] / [threadgroup ]); [None]
      makes a non-empty [workgroup_shared] set a compile-time error. *)

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
  (** Vector register width in bytes for explicit SIMD rendering of [Vectorized] loops via
      GCC/Clang vector extensions (the [Vectorized] codegen follow-up of gh-ocannl-164 /
      docs/proposals/watch-ocannl-README-md-347818d3.md): eligible loop bodies emit vector-typed
      loads, arithmetic and stores in [lanes = vector_bytes / element size] chunks plus a serial
      remainder loop, instead of relying on the compiler's auto-vectorizer (which e.g. cannot
      reassociate strict-FP reductions — the [Vectorized] retype carries that permission, like
      [Swap]). [0] disables explicit emission ([vectorize_pragma] fallback). *)

  val vector_style : [ `Vec_extensions | `Packed_struct ]
  (** How eligible [Vectorized] loops emit explicit vector code when [vector_bytes > 0].
      [`Vec_extensions] (CPU): GCC/Clang [vector_size] types, unaligned [__builtin_memcpy]
      loads/stores, vector-infix arithmetic. [`Packed_struct] (GPU, gh-ocannl-463; llm.c's
      [Packed128], llmc/cuda_utils.cuh): the backend's [vec_typ_of_prec] aggregate is loaded and
      stored through [reinterpret_cast] at guaranteed-aligned offsets — the 128-bit LDG/STS
      transactions that bandwidth-bound kernels need — while the arithmetic stays scalar in a
      per-lane loop over the pack's [.v] payload (on GPU the payoff is memory transactions, not
      SIMD ALUs; per-lane [fmaf]/[fma] also matches the serial path's rounding exactly).
      [`Packed_struct] eligibility additionally requires every vector-accessed node to be
      materialized (device buffers and pool offsets are [Ops.buffer_alignment]-aligned, stack and
      workgroup-shared arrays only element-aligned) and every access's non-loop offset
      contribution to be a lane multiple. *)

  val aligned_local_attr : string option
  (** Declaration suffix aligning stack-allocated local arrays for SIMD access, e.g.
      [__attribute__((aligned(32)))] (gh-ocannl-164). Applies to the plain stack-array branch only,
      never to workgroup-shared placements. *)

  val warp_size : int
  (** SIMD-group (warp) width for the warp-shuffle rendering of [Workgroup_reduce] accumulation
      loops (gh-ocannl-462; llm.c's [warpReduceSum]/[blockReduce] idiom). Backends setting this to
      a nonzero power of two must define [ocannl_shfl_xor(value, lane_mask)] overloads in their
      builtins for the supported accumulator precisions (single, and double where it exists), bind
      workgroup slot 0 in [hardware_index], and provide [barrier_syntax] plus [shared_decl_prefix]
      (needed by the two-phase multi-warp form). [0] disables the rendering: [Workgroup_reduce]
      loops render like [Workgroup] — hardware binding, or the serial fallback (which is the
      correct meaning of a recognized accumulation body on CPU backends). *)

  val mma_syntax :
    (d_prec:Ops.prec ->
    a_prec:Ops.prec ->
    b_prec:Ops.prec ->
    m:int ->
    n:int ->
    k:int ->
    d:PPrint.document * int * [ `Device | `Shared | `Thread ] ->
    a:PPrint.document * int * [ `Device | `Shared | `Thread ] ->
    b:PPrint.document * int * [ `Device | `Shared | `Thread ] ->
    PPrint.document option)
    option
  (** Cooperative tile-MMA emission for [Low_level.Tile_mma]
      (docs/proposals/tensorize-mma.md §4): given the per-operand precisions (the backend decides
      which combinations its units support — Metal [simdgroup_matrix] is uniform-precision only,
      CUDA wmma's flagship combination is mixed f16×f16→f32), the covered block extents
      [m]/[n]/[k], and per operand a pointer expression to the tile base (already offset), its
      leading-dimension stride in elements, and its address space, emit the intrinsic sequence
      (fragment declarations / loads / mma steps / stores) executed by every lane of the enclosing
      lane loop. Return [None] to decline a particular call (unsupported precision combination,
      extents not multiples of the intrinsic tile, thread-space operand) — the caller then renders
      the scalar [fallback] under an [if (lane == 0)] guard, which is also the path when the whole
      hook is [None] (cc, and any backend until wired). *)

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

  (* Plain C backends bind no hardware axes: annotated loops fall back to serial [for] loops
     (sound absent barriers), and barriers / shared placements are compile-time errors. *)
  let hardware_index ~kind:_ ~slot:_ = None
  let barrier_syntax = None
  let parallel_grid_syntax = `None
  let parallel_grid_chunks = 1
  let vector_bytes = 0
  let vector_style = `Vec_extensions
  let shared_decl_prefix = None
  let restrict_keyword = Some "restrict"

  (* Clang defines both [__clang__] and [__GNUC__], so test [__clang__] first. *)
  let vectorize_pragma =
    [
      "#if defined(__clang__)";
      "#pragma clang loop vectorize(enable) interleave(enable)";
      "#elif defined(__GNUC__)";
      "#pragma GCC ivdep";
      "#endif";
    ]

  let aligned_local_attr =
    Some (Printf.sprintf "__attribute__((aligned(%d)))" Ops.buffer_alignment)

  (* No shuffle intrinsics on plain C backends: a [Workgroup_reduce] accumulation loop renders as
     the serial fallback, which is exactly its serial meaning. *)
  let warp_size = 0

  (* No tile-MMA units on plain C backends: [Tile_mma] renders its scalar fallback under the
     [lane == 0] guard. *)
  let mma_syntax = None
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

  (* Set by [compile_proc]: outermost [Grid] loops eligible for pool-backed parallel rendering
     (docs/proposals/gh-ocannl-164.md), identified by index symbol. Empty unless
     [B.parallel_grid_syntax] renders in parallel. *)
  let current_parallel_grid : Set.M(Indexing.Symbol).t ref =
    ref (Set.empty (module Indexing.Symbol))

  (* Whether the body of an outermost [Grid] loop over [sym] tolerates its iterations being
     partitioned into chunks that execute on parallel CPU threads sharing the kernel's
     function-scope state. Materialized accesses are already safe: [validate_parallel] requires
     every materialized write to cover [sym], so chunks write disjoint elements, and nests execute
     as separate parallel loops with a join in between (stronger than the GPU's single launch).
     The hazards are the function-scope stack arrays declared by [local_decls]: on GPU each thread
     gets a private copy, so a GPU-valid kernel may legally write them grid-invariantly (identical
     values per iteration) -- under one shared array that is a data race. Hence, for a local
     written under the loop: every access to it (read or write) must mention [sym], and all
     accesses must agree on every index component that mentions [sym] -- the same agreement rule
     as the default annotator's hazard analysis; mere mention is not enough, e.g. a stencil write
     [tmp[i]] + read [tmp[i-1]] both mention [sym] but reach across iterations. [Set_local] scope
     locals must have their declaration within the loop body (block scope = per-chunk storage).
     Opaque statements and barriers disqualify. *)
  let parallel_grid_safe ~sym (body : Low_level.t) : bool =
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
    (* Per accessed function-scope local: (written under the loop, access index vectors). *)
    let locals : (bool * Indexing.axis_index array list) Hashtbl.M(Int).t =
      Hashtbl.create (module Int)
    in
    let declared_scopes = Hash_set.create (module Int) in
    let ok = ref true in
    let access tn idcs ~write =
      if is_local tn then
        Hashtbl.update locals tn.Tn.uid ~f:(fun st ->
            let written, accs = Option.value st ~default:(false, []) in
            (written || write, idcs :: accs))
    in
    let rec go (llc : Low_level.t) =
      match llc with
      | Low_level.Noop | Comment _ -> ()
      | Staged_compilation _ | Workgroup_barrier | Tile_mma _ -> ok := false
      | Seq (a, b) ->
          go a;
          go b
      | For_loop { body; _ } -> go body
      | If { cond = c, _; body } ->
          go_sc c;
          go body
      | Zero_out tn -> access tn [||] ~write:true
      | Set { tn; idcs; llsc; _ } ->
          access tn idcs ~write:true;
          go_sc llsc
      | Set_from_vec { tn; idcs; arg = a, _; _ } ->
          access tn idcs ~write:true;
          go_sc a
      | Set_local (id, llsc) ->
          if not (Hash_set.mem declared_scopes id.Low_level.scope_id) then ok := false;
          go_sc llsc
      | Declare_local { id; _ } -> Hash_set.add declared_scopes id.Low_level.scope_id
    and go_sc (llsc : Low_level.scalar_t) =
      match llsc with
      | Local_scope { id; body; _ } ->
          Hash_set.add declared_scopes id.Low_level.scope_id;
          go body
      | Get_local _ -> ()
      | Get (tn, idcs) -> access tn idcs ~write:false
      | Get_dynamic { tn; dyn_value = v, _; _ } ->
          (* The dynamic slot's effective index is data-dependent: conservatively a miss. *)
          access tn [||] ~write:false;
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
    go body;
    !ok
    && Hashtbl.for_all locals ~f:(fun (written, accs) ->
           (* The blocks extension ([`Dispatch]) cannot capture function-scope arrays at all:
              even a read-only reference fails to compile ("cannot refer to declaration with an
              array type inside block"). Fissioned segments hit this with serially-precomputed
              local scratch read under a Grid loop (e.g. softmax gradients in the backward
              segments of the training tests); OpenMP has no such restriction and keeps the
              finer written-under-the-loop analysis below. *)
           (match B.parallel_grid_syntax with `Dispatch -> false | `Openmp | `None -> true)
           && ((not written)
              || (* Distinct grid iterations must touch disjoint elements: every access mentions
                    [sym]... *)
              List.for_all accs ~f:(fun idcs -> Array.exists idcs ~f:mentions_comp)
           &&
           (* ...and all accesses agree on every component that mentions [sym], so within one
              iteration reads hit exactly the cells that iteration writes. *)
           let rank = List.fold accs ~init:0 ~f:(fun m a -> max m (Array.length a)) in
           let agree = ref true in
           for p = 0 to rank - 1 do
             let comps =
               List.map accs ~f:(fun a ->
                   if p < Array.length a then a.(p) else Indexing.Fixed_idx 0)
             in
             if List.exists comps ~f:mentions_comp then
               match comps with
               | [] -> ()
               | c0 :: rest ->
                   if not (List.for_all rest ~f:(Indexing.equal_axis_index c0)) then agree := false
           done;
           !agree))

  (* The outermost [Grid] loops safe to render in parallel. Nested [Grid] loops render serially
     inside a chunk (still correct: write coverage holds per grid index). Runtime kernel logging
     writes to a shared FILE, so parallel rendering is skipped under [debug_log_from_routines]. *)
  let collect_parallel_grid (llc : Low_level.t) : Set.M(Indexing.Symbol).t =
    if
      Poly.equal B.parallel_grid_syntax `None
      || B.parallel_grid_chunks <= 1
      || Utils.debug_log_from_routines ()
    then Set.empty (module Indexing.Symbol)
    else
      let acc = ref (Set.empty (module Indexing.Symbol)) in
      let rec go (llc : Low_level.t) =
        match llc with
        | Low_level.For_loop { axis = Grid; index; from_; to_; body; _ } ->
            if from_ = 0 && to_ >= 1 && parallel_grid_safe ~sym:index body then
              acc := Set.add !acc index
        | For_loop { body; _ } -> go body
        | If { body; _ } -> go body
        | Seq (a, b) ->
            go a;
            go b
        | _ -> ()
      in
      go llc;
      !acc

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

  let rec pp_ll ?(log_set_locals = true) ?(in_loop = false) (c : Low_level.t) : PPrint.document =
    let open PPrint in
    match c with
    | Low_level.Noop -> empty
    | Seq (c1, c2) ->
        let d1 = pp_ll ~log_set_locals ~in_loop c1 in
        let d2 = pp_ll ~log_set_locals ~in_loop c2 in
        (* Avoid extra hardlines if one side is empty *)
        if PPrint.is_empty d1 then d2 else if PPrint.is_empty d2 then d1 else d1 ^^ hardline ^^ d2
    | For_loop { index = i; from_; to_; body; trace_it = _; axis } -> (
        (* Rendering phase of docs/proposals/axis-types-for-loops.md (§5): [Serial] loops render as
           C [for] statements; [Grid]/[Workgroup]/[Workgroup_reduce] loops bind their index to the
           backend's hardware register (at the signed [loop_index_type] width, with an explicit
           cast from the unsigned register) when [B.hardware_index] provides one, and fall back to
           a serial loop otherwise (legal absent barriers); [Vectorized] loops render serially,
           prefixed with [B.vectorize_pragma] when non-empty; [Unrolled] loops emit the repeated
           body with the index bound as a per-block constant. *)
        let body_doc () =
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
          let header =
            string ("for (" ^ B.loop_index_type)
            ^^ pp_symbol i ^^ string " = " ^^ PPrint.OCaml.int from_ ^^ semi ^^ space ^^ pp_symbol i
            ^^ string " <= " ^^ PPrint.OCaml.int to_ ^^ semi ^^ space ^^ string "++" ^^ pp_symbol i
            ^^ string ")"
          in
          group (header ^^ space ^^ lbrace ^^ nest 2 (hardline ^^ body_doc ()) ^^ hardline ^^ rbrace)
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
             loops and nested [Grid] loops stay serial inside a chunk. Eligibility (including
             [from_ = 0]) was established by [collect_parallel_grid]. *)
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
        (* Explicit SIMD rendering of a [Vectorized] loop via GCC/Clang vector extensions
           (portable across gcc/clang and AVX2/NEON; the [Vectorized]-codegen follow-up of
           gh-ocannl-164). The loop must start at 0 and its body must be a sequence of plain [Set]
           statements over one floating precision, with every access that mentions the loop index
           contiguous in it (the index appears only in the last component, with coefficient 1 —
           the flat offset then advances by exactly 1 per iteration). Index-free subexpressions
           render as scalars (vector-scalar arithmetic splats across lanes); vector subexpressions
           allow [Add]/[Sub]/[Mul]/[Div]/[Neg] and fused [FMA] (matching the scalar path's
           [fmaf]/[fma] rounding; see the note in [vec_expr]). At most one store per node, and
           every read
           of a stored node must use the store's exact index vector — vector semantics evaluates
           all lanes' loads before the store, so cross-lane flow would reorder against the serial
           loop. The main loop advances by [lanes]; a serial remainder loop reuses [body_doc].
           Anything else falls back to [vectorize_pragma] / serial. *)
        let try_vectorize () : PPrint.document option =
          let exception Bail in
          let mentions_comp (idx : Indexing.axis_index) =
            match idx with
            | Indexing.Iterator s -> Indexing.equal_symbol s i
            | Indexing.Affine { symbols; _ } ->
                List.exists symbols ~f:(fun (_, s) -> Indexing.equal_symbol s i)
            | Indexing.Fixed_idx _ | Indexing.Sub_axis | Indexing.Concat _ -> false
          in
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
          try
            if B.vector_bytes < 8 || from_ <> 0 || Utils.debug_log_from_routines () then
              raise Bail;
            let extent = to_ + 1 in
            let stmts =
              List.filter (Low_level.flat_lines [ body ]) ~f:(function
                | Low_level.Noop | Comment _ -> false
                | _ -> true)
            in
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
            let contiguous idcs =
              let n = Array.length idcs in
              n > 0
              && Array.for_alli idcs ~f:(fun p idx -> p = n - 1 || not (mentions_comp idx))
              &&
              match idcs.(n - 1) with
              | Indexing.Iterator s -> Indexing.equal_symbol s i
              | Indexing.Affine { symbols; _ } ->
                  List.for_all symbols ~f:(fun (c, s) ->
                      (not (Indexing.equal_symbol s i)) || c = 1)
                  && List.count symbols ~f:(fun (_, s) -> Indexing.equal_symbol s i) = 1
              | _ -> false
            in
            let check_read tn idcs =
              match Hashtbl.find written tn.Tn.uid with
              | Some w_idcs ->
                  if not (Array.equal Indexing.equal_axis_index w_idcs idcs) then raise Bail
              | None -> ()
            in
            let rec no_written_reads (llsc : Low_level.scalar_t) =
              match llsc with
              | Low_level.Get (tn, _) | Get_merge_buffer (tn, _) | Get_dynamic { tn; _ } ->
                  if Hashtbl.mem written tn.Tn.uid then raise Bail
              | Local_scope _ | Get_local _ -> raise Bail
              | Embed_index _ | Constant _ | Constant_bits _ -> ()
              | Ternop (_, (a, _), (b, _), (c, _)) ->
                  no_written_reads a;
                  no_written_reads b;
                  no_written_reads c
              | Binop (_, (a, _), (b, _)) ->
                  no_written_reads a;
                  no_written_reads b
              | Unop (_, (a, _)) -> no_written_reads a
            in
            let stmts_docs = ref [] in
            let emit d = stmts_docs := d :: !stmts_docs in
            let fresh =
              let ctr = ref 0 in
              fun pfx ->
                Int.incr ctr;
                Printf.sprintf "%s%d__" pfx !ctr
            in
            let uniform_scalar llsc =
              (* Uniform across lanes: a read of a stored node cannot equal its (index-mentioning)
                 store vector, so reject those; then render as a plain scalar (vector-scalar
                 arithmetic splats; in the packed style the scalar participates per lane). *)
              no_written_reads llsc;
              let local_defs, sdoc = pp_scalar prec llsc in
              if not (List.is_empty local_defs) then raise Bail;
              parens sdoc
            in
            let prelude =
              match B.vector_style with
              | `Vec_extensions ->
                  let vtyp =
                    Printf.sprintf "ocannl_vec%d%s" lanes
                      (match prec with Ops.Double_prec _ -> "d" | _ -> "f")
                  in
                  let vload tn idcs =
                    if not (contiguous idcs) then raise Bail;
                    check_read tn idcs;
                    let name = fresh "vget" in
                    let offset = pp_array_offset (idcs, Lazy.force tn.Tn.dims) in
                    emit
                      (string (vtyp ^ " " ^ name ^ ";")
                      ^^ hardline
                      ^^ string ("__builtin_memcpy(&" ^ name ^ ", &")
                      ^^ string (get_ident tn) ^^ brackets offset
                      ^^ string (", sizeof(" ^ name ^ "));"));
                    string name
                  in
                  let rec vec_expr (llsc : Low_level.scalar_t) (p : Ops.prec) : PPrint.document =
                    if not (scalar_mentions llsc) then uniform_scalar llsc
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
                          (* Fused, matching the scalar path's [fmaf]/[fma] single rounding (the
                             simplifier synthesizes [FMA] from mul-add trees, so this is the hot
                             case): clang's [__builtin_elementwise_fma] where available, otherwise
                             a per-lane fused-fma loop (fixed trip count; SLP-vectorizes under
                             -mfma/NEON). A plain [a * b + c] would be only maybe-contracted, so
                             vector lanes could differ from the serial remainder loop and twin.
                             Operands bind to vector temps (lane-uniform ones splat explicitly:
                             vector = scalar init is invalid). *)
                          let bind llsc p =
                            let name = fresh "vfop" in
                            emit
                              (string (vtyp ^ " " ^ name ^ " = ") ^^ vec_operand llsc p ^^ semi);
                            name
                          in
                          let na = bind a pa and nb = bind b pb and nc = bind c pc in
                          let nr = fresh "vfma" in
                          let fma_fn =
                            match prec with Ops.Double_prec _ -> "fma" | _ -> "fmaf"
                          in
                          emit
                            (string (vtyp ^ " " ^ nr ^ ";")
                            ^^ hardline
                            ^^ string "#if OCANNL_HAS_ELEMENTWISE_FMA"
                            ^^ hardline
                            ^^ string
                                 (Printf.sprintf "%s = __builtin_elementwise_fma(%s, %s, %s);" nr
                                    na nb nc)
                            ^^ hardline ^^ string "#else" ^^ hardline
                            ^^ string
                                 (Printf.sprintf
                                    "for (int ocannl_l__ = 0; ocannl_l__ < %d; ++ocannl_l__) %s[ocannl_l__] = %s(%s[ocannl_l__], %s[ocannl_l__], %s[ocannl_l__]);"
                                    lanes nr fma_fn na nb nc)
                            ^^ hardline ^^ string "#endif");
                          string nr
                      | Unop (Ops.Identity, (a, pa)) -> vec_expr a pa
                      | Unop (Ops.Neg, (a, pa)) -> parens (string "-" ^^ vec_expr a pa)
                      | _ -> raise Bail
                  and vec_operand (llsc : Low_level.scalar_t) (p : Ops.prec) : PPrint.document =
                    (* A vector-typed rendering even for lane-uniform values: initializers and
                       builtin arguments need a vector, where the implicit vector-scalar splat of
                       binary operators does not apply. *)
                    if scalar_mentions llsc then vec_expr llsc p
                    else string ("((" ^ vtyp ^ "){0} + ") ^^ uniform_scalar llsc ^^ string ")"
                  in
                  List.iter sets ~f:(fun (tn, idcs, llsc) ->
                      if not (contiguous idcs) then raise Bail;
                      let rhs = vec_operand llsc prec in
                      let vname = fresh "vset" in
                      emit (string (vtyp ^ " " ^ vname ^ " = ") ^^ rhs ^^ semi);
                      emit
                        (string "__builtin_memcpy(&" ^^ string (get_ident tn)
                        ^^ brackets (pp_array_offset (idcs, Lazy.force tn.Tn.dims))
                        ^^ string (", &" ^ vname ^ ", sizeof(" ^ vname ^ "));")));
                  string
                    (Printf.sprintf "typedef %s %s __attribute__((vector_size(%d)));"
                       (B.typ_of_prec prec) vtyp
                       (lanes * Ops.prec_in_bytes prec))
                  ^^ hardline
              | `Packed_struct ->
                  (* GPU 128-bit packed loads/stores (gh-ocannl-463; llm.c's Packed128): the
                     backend's aligned pack aggregate is loaded/stored via [reinterpret_cast] —
                     one 128-bit memory transaction — while the arithmetic stays scalar in a
                     per-lane loop over the pack's [.v] payload (per-lane [fmaf]/[fma] keeps the
                     serial path's rounding). Sound only at provably lane-aligned offsets of
                     device-resident buffers, hence the extra eligibility checks. *)
                  let vtyp =
                    match B.vec_typ_of_prec ~length:lanes prec with
                    | s -> s
                    | exception _ -> raise Bail
                  in
                  (* The flat offset must stay a lane multiple whenever the loop index is one:
                     components before the last contribute stride multiples of [dims.(n - 1)], so
                     the last dimension must be a lane multiple (unless the access is 1-D), and
                     the last component's constant offset and non-index coefficients must be lane
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
                  in
                  let vload tn idcs =
                    if not (eligible tn idcs) then raise Bail;
                    check_read tn idcs;
                    let name = fresh "vget" in
                    emit
                      (string
                         (Printf.sprintf "const %s %s = *reinterpret_cast<%sconst %s*>(&" vtyp
                            name B.buffer_prefix vtyp)
                      ^^ string (get_ident tn)
                      ^^ brackets (pp_array_offset (idcs, Lazy.force tn.Tn.dims))
                      ^^ string ");");
                    name
                  in
                  let lane_var = "ocannl_l__" in
                  let rec lane_expr (llsc : Low_level.scalar_t) (p : Ops.prec) : PPrint.document =
                    if not (scalar_mentions llsc) then uniform_scalar llsc
                    else if not (Ops.equal_prec p prec) then raise Bail
                    else
                      match llsc with
                      | Low_level.Get (tn, idcs) ->
                          if not (Ops.equal_prec (Lazy.force tn.Tn.prec) prec) then raise Bail;
                          string (vload tn idcs ^ ".v[" ^ lane_var ^ "]")
                      | Binop (((Ops.Add | Ops.Sub | Ops.Mul | Ops.Div) as op), (a, pa), (b, pb))
                        ->
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
                           (Printf.sprintf "for (int %s = 0; %s < %d; ++%s) { %s.v[%s] = "
                              lane_var lane_var lanes lane_var vname lane_var)
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
        (* Warp-shuffle rendering of a [Workgroup_reduce] accumulation loop (gh-ocannl-462;
           llm.c's [warpReduceSum] / [blockReduce] idiom, llmc/cuda_utils.cuh). Recognizes a body
           that is a single accumulation statement [acc[idcs] = op(acc[idcs], contrib)] (or its
           FMA form [acc = FMA(a, b, acc)]) where [idcs] does not mention the loop index and [op]
           is an associative-commutative reduction — such a body IS the loop's serial meaning, so
           backends without shuffle support ([warp_size = 0]) render it with the ordinary
           fallbacks. With shuffle support the loop renders as: every thread computes its
           contribution, a log2(warp) [ocannl_shfl_xor] tree reduces within each warp, then (for
           multi-warp extents) lane 0 of each warp stages one value in a workgroup-shared slot, a
           barrier, and the first warp shuffle-reduces the per-warp partials — thread 0 finally
           folds the total into the accumulator (reassociation is the annotation's license, like
           [Vectorized]). This halves the shared-memory traffic and barrier count of the
           explicitly staged tree, which remains supported: unrecognized bodies keep the
           [Workgroup]-style hardware binding and their own staging and barriers.

           The multi-warp phase needs no identity constant: [num_warps] must be a power of two,
           and XOR with offsets [< num_warps] maps lanes [< num_warps] onto themselves, so the
           garbage held by lanes [>= num_warps] never mixes into the reduced prefix.

           A recognized accumulation that cannot be rendered (extent not covering whole warps,
           reduce axis not at workgroup slot 0, ...) raises: binding the index like a plain
           [Workgroup] axis would make every thread race the read-modify-write. *)
        let try_warp_reduce () : PPrint.document option =
          if B.warp_size <= 0 then None
          else
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
              | Get_dynamic { tn = tn2; dyn_value = v, _; _ } ->
                  Tn.equal tn tn2 || touches_tn tn v
              | Local_scope { body; _ } -> body_touches tn body
              | Get_local _ | Constant _ | Constant_bits _ | Embed_index _ -> false
              | Ternop (_, (a, _), (b, _), (c, _)) ->
                  touches_tn tn a || touches_tn tn b || touches_tn tn c
              | Binop (_, (a, _), (b, _)) -> touches_tn tn a || touches_tn tn b
              | Unop (_, (a, _)) -> touches_tn tn a
            and body_touches tn (llc : Low_level.t) =
              match llc with
              | Low_level.Noop | Comment _ | Staged_compilation _ | Workgroup_barrier
              | Declare_local _ ->
                  false
              | Seq (a, b) -> body_touches tn a || body_touches tn b
              | For_loop { body; _ } -> body_touches tn body
              | If { cond = c, _; body } -> touches_tn tn c || body_touches tn body
              | Zero_out tn2 -> Tn.equal tn tn2
              | Set { tn = tn2; llsc; _ } -> Tn.equal tn tn2 || touches_tn tn llsc
              | Set_from_vec { tn = tn2; arg = a, _; _ } -> Tn.equal tn tn2 || touches_tn tn a
              | Set_local (_, llsc) -> touches_tn tn llsc
              | Tile_mma { d = d_tn, _; a = a_tn, _; b = b_tn, _; _ } ->
                  Tn.equal tn d_tn || Tn.equal tn a_tn || Tn.equal tn b_tn
            in
            let stmts =
              List.filter (Low_level.flat_lines [ body ]) ~f:(function
                | Low_level.Noop | Comment _ -> false
                | _ -> true)
            in
            let recognized =
              match stmts with
              | [ Low_level.Set { tn; idcs; llsc; _ } ]
                when not (Array.exists idcs ~f:mentions_comp) -> (
                  let is_acc s = Low_level.equal_scalar_t s (Low_level.Get (tn, idcs)) in
                  let reduce_op = function
                    | Ops.Add | Ops.Mul | Ops.Max | Ops.Min -> true
                    | _ -> false
                  in
                  match llsc with
                  | Binop (op, (a, _), (b, _))
                    when reduce_op op && is_acc a && not (touches_tn tn b) ->
                      Some (tn, idcs, op, b)
                  | Binop (op, (a, _), (b, _))
                    when reduce_op op && is_acc b && not (touches_tn tn a) ->
                      Some (tn, idcs, op, a)
                  | Ternop (Ops.FMA, (a, pa), (b, pb), (c, _))
                    when is_acc c && (not (touches_tn tn a)) && not (touches_tn tn b) ->
                      Some (tn, idcs, Ops.Add, Low_level.Binop (Ops.Mul, (a, pa), (b, pb)))
                  | _ -> None)
              | _ -> None
            in
            match recognized with
            | None -> None
            | Some (tn, idcs, op, contrib) ->
                let fail msg =
                  invalid_arg
                    ("C_syntax.pp_ll: Workgroup_reduce loop " ^ symbol_ident i
                   ^ " is a recognized accumulation, but the warp-shuffle rendering requires "
                   ^ msg ^ " (a plain hardware binding would race the accumulator update)")
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
                    (Printf.sprintf "the extent (%d) to be a multiple of the warp size (%d)"
                       extent warp);
                let num_warps = extent / warp in
                let axes = !current_hardware_axes in
                (match
                   List.find axes ~f:(fun a -> Indexing.equal_symbol a.Low_level.ha_index i)
                 with
                | Some a when a.Low_level.ha_slot = 0 -> ()
                | Some _ ->
                    fail
                      "the reduce axis at workgroup slot 0 (warp lanes are consecutive .x \
                       threads)"
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
                      "the reduce axis to be the only workgroup axis (the per-warp staging \
                       slots are not replicated per sibling workgroup thread)");
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
                                   (Printf.sprintf "if (%s < %d) { %s = %s[%s]; }" ident
                                      num_warps vname pname ident)
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
                        "{ /* Workgroup_reduce warp-shuffle rendering: extent %d = %d \
                         simdgroup(s) of %d. */"
                        extent num_warps warp)
                  ^^ nest 2
                       (hardline ^^ binding ^^ hardline
                       ^^ (if PPrint.is_empty local_defs then empty
                           else local_defs ^^ hardline)
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
            match try_warp_reduce () with
            | Some doc -> doc
            | None -> hardware_binding `Workgroup)
        | Vectorized -> (
            match try_vectorize () with
            | Some doc -> doc
            | None -> (
                (* gh-ocannl-164: a serial loop annotated with the backend's vectorization
                   pragmas; without them the plain serial loop is the legal fallback (same
                   discipline as unbound [Grid]/[Workgroup] axes). *)
                match B.vectorize_pragma with
                | [] -> serial_loop ()
                | lines -> separate_map hardline string lines ^^ hardline ^^ serial_loop ()))
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
        let offset_doc = pp_array_offset (idcs, dims) in
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
        (* Generate assignments for each output element *)
        let open PPrint in
        let vec_var = string "vec_result" in
        let vec_typ = string (B.vec_typ_of_prec ~length prec) in
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
                  if length = 1 then
                    (* When length=1, vec_typ_of_prec returns a scalar type, so no .v[] access *)
                    vec_var
                  else
                    (* When length>1, access the vector element *)
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
    | Tile_mma { d; a; b; m; n; k; lane; fallback } -> (
        (* Cooperative tile-MMA (docs/proposals/tensorize-mma.md §4). Backends with an [mma_syntax]
           hook emit the intrinsic sequence on every lane; everywhere else (including per-call
           declines and logged runs, which must stay serial and deterministic) the scalar fallback
           runs once per simdgroup, guarded on lane 0 — the lane loop still supplies the launch's
           threads. *)
        let operand (tn, idcs) =
          let dims = Lazy.force tn.Tn.dims in
          let prec = Lazy.force tn.Tn.prec in
          let rank = Array.length dims in
          let ld = if rank >= 1 then dims.(rank - 1) else 1 in
          let space =
            if Set.mem !current_workgroup_shared tn then `Shared
            else if Tn.Placements.is_materialized_force (placements ()) tn 440 then `Device
            else `Thread
          in
          let ptr_doc =
            parens (string (get_ident tn) ^^ string " + " ^^ pp_array_offset (idcs, dims))
          in
          (prec, (ptr_doc, ld, space))
        in
        let fallback_doc () =
          let guarded =
            group
              (string "if (" ^^ pp_symbol lane ^^ string " == 0) " ^^ lbrace
              ^^ nest 2 (hardline ^^ pp_ll ~log_set_locals ~in_loop:true fallback)
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
        match B.mma_syntax with
        | None -> fallback_doc ()
        | Some _ when Utils.debug_log_from_routines () -> fallback_doc ()
        | Some emit -> (
            let d_prec, d_op = operand d in
            let a_prec, a_op = operand a in
            let b_prec, b_op = operand b in
            match emit ~d_prec ~a_prec ~b_prec ~m ~n ~k ~d:d_op ~a:a_op ~b:b_op with
            | Some doc -> doc
            | None -> fallback_doc ()))
    | If { cond = c, cprec; body } ->
        (* Guarded statement (axis-types proposal §2): [body] executes iff [cond] is nonzero --
           C's [if] tests exactly that. *)
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
        let offset_doc = pp_array_offset (idcs, dims) in
        let expr = string prefix ^^ ident_doc ^^ brackets offset_doc ^^ string postfix in
        ([], expr)
    | Get_dynamic { tn; idcs; dyn_axis; dyn_value = iv, iprec } ->
        (* gh-343: a guarded dynamic gather. The dynamic index is spliced into the row-major offset
           at [dyn_axis] as an integer; the enclosing [Where] guard (when interval analysis has not
           discharged it against proven bounds) guarantees it is in range before this load is
           evaluated (C ternary short-circuits). Cast to [Ops.index_prec ()] so the index tracks
           the same width as loop counters (signed int32 normally, int64 under large_models),
           preventing truncation for very large table/vocabulary axes. *)
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
        let offset_doc = pp_array_offset (idcs, dims) in
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
      Low_level.{ traced_store; llc; merge_node; optimize_ctx; workgroup_shared } :
      (string * kparam_source) list * PPrint.document * Low_level.launch_dims =
    let open PPrint in
    (if not (Set.is_empty workgroup_shared) then
       match B.shared_decl_prefix with
       | Some _ -> ()
       | None ->
           (* The local-declaration pass below would silently emit workgroup-shared nodes as
              per-thread stack arrays -- wrong sharing semantics. *)
           invalid_arg
             "C_syntax.compile_proc: workgroup-shared placement not supported by this backend");
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
    current_parallel_grid := collect_parallel_grid llc;
    current_workgroup_shared := workgroup_shared;
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
            let restrict_ =
              match B.restrict_keyword with Some kw -> kw ^ " " | None -> ""
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
             restrict qualifier asserts (gh-ocannl-164). *)
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

    let local_decls =
      string "/* Local declarations and initialization. */"
      ^^ hardline
      ^^ separate_map empty
           (fun (tn, node) ->
             let plc = placements () in
             if
               not
                 (Tn.Placements.is_virtual_force plc tn 333
                 || Tn.Placements.is_materialized_force plc tn 336)
             then
               let typ_doc = string (B.typ_of_prec @@ Lazy.force tn.prec) in
               let ident_doc = string (get_ident tn) in
               let num_elems = Tn.num_elems tn in
               let size_doc = OCaml.int num_elems in
               let is_shared = Set.mem workgroup_shared tn in
               let prefix_doc =
                 (* Workgroup-shared placement (axis-types proposal §3): one tile per workgroup
                    instead of one per thread. [= {0}] is not allowed for shared declarations, so
                    zero-initialization stays as explicit [Zero_out] code (never elided for shared
                    nodes; see [zero_out_loop_redundant]). *)
                 if is_shared then string (Option.value_exn ~here:[%here] B.shared_decl_prefix)
                 else empty
               in
               let init_doc =
                 if node.Low_level.zero_initialized_by_code && not is_shared then string " = {0}"
                 else empty
               in
               let align_doc =
                 (* SIMD alignment for plain stack arrays only (gh-ocannl-164); shared placements
                    keep the backend's default layout. *)
                 match B.aligned_local_attr with
                 | Some attr when not is_shared -> string (" " ^ attr)
                 | _ -> empty
               in
               prefix_doc ^^ typ_doc ^^ space ^^ ident_doc ^^ brackets size_doc ^^ align_doc
               ^^ init_doc ^^ semi ^^ hardline
             else empty)
           (Hashtbl.to_alist traced_store)
    in
    body := !body ^^ local_decls ^^ hardline;

    let main_logic = string "/* Main logic. */" ^^ hardline ^^ compile_main llc in
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
