open Base
module Lazy = Utils.Lazy
open Ir

let _get_local_debug_runtime = Utils.get_local_debug_runtime

[%%global_debug_log_level 0]

(* export OCANNL_LOG_LEVEL_CC_BACKEND=9 to enable debugging into the log_files/ directory. *)
[%%global_debug_log_level_from_env_var "OCANNL_LOG_LEVEL_CC_BACKEND"]

include Backend_impl.No_device_buffer_and_copying ()
open Backend_intf

let name = "cc"

let optimization_level () =
  Int.of_string @@ Utils.get_global_arg ~default:"3" ~arg_name:"cc_backend_optimization_level"

let arch_flags () = Utils.get_global_arg ~default:"-march=native" ~arg_name:"cc_backend_arch_flags"
let fast_math_enabled () = Utils.get_global_flag ~default:false ~arg_name:"cc_backend_fast_math"

let compiler_command =
  let default =
    (* TODO: there's a direct way to get the compiler command from the OCaml compiler. *)
    lazy
      (let ic = Unix.open_process_in "ocamlc -config" in
       let rec find_compiler () =
         match In_channel.input_line ic with
         | None -> "cc" (* Default fallback *)
         | Some line ->
             if String.is_prefix line ~prefix:"c_compiler: " then
               String.drop_prefix line 12 (* Length of "c_compiler: " *)
             else find_compiler ()
       in
       let compiler = find_compiler () in
       ignore (Unix.close_process_in ic);
       compiler)
  in
  fun () ->
    Utils.get_global_arg ~default:(Lazy.force default) ~arg_name:"cc_backend_compiler_command"

(* gh-ocannl-164: explicit SIMD flags appended to the compiler invocation. The "auto" default
   probes once per process, in two stages. Stage 1 test-compiles a translation unit that #errors
   unless the target selected by [arch_flags] alone already defines __AVX2__ and __FMA__ — so the
   explicit flags never escalate the ISA beyond the configured target (a non-AVX2 x86 CPU under
   -march=native, or any ARM target, gets no flags and the generated code runs wherever the target
   does). Stage 2 verifies the compiler accepts the candidate flags themselves, preferring the
   variant with -ftree-vectorize (explicit, though -O3 usually implies it). Any other config value
   is passed through verbatim (empty disables). *)
(* Probe whether the configured compiler accepts [flags] on the translation unit [data].
   [`Compile] is compile-only ([-c]); [`Link] builds a full executable ([data] must define
   [main]), catching runtimes that the compiler accepts flags for but cannot link -- e.g. clang
   accepting [-fopenmp] at compile time without libomp installed. Shared by the SIMD-flags and
   parallel-grid probes. *)
let probe_compiles ?(mode = `Compile) ~flags ~data () =
  let src = Stdlib.Filename.temp_file "ocannl_cc_probe_" ".c" in
  let out = Stdlib.Filename.temp_file "ocannl_cc_probe_" ".out" in
  let log = Stdlib.Filename.temp_file "ocannl_cc_probe_" ".log" in
  let ok =
    try
      Stdio.Out_channel.write_all src ~data;
      let compile_only = match mode with `Compile -> "-c " | `Link -> "" in
      let cmd =
        Printf.sprintf "%s %s %s%s -o %s > %s 2>&1" (compiler_command ()) flags compile_only src out
          log
      in
      Stdlib.Sys.command cmd = 0
    with _ -> false
  in
  List.iter [ src; out; log ] ~f:(fun f -> try Stdlib.Sys.remove f with _ -> ());
  ok

let simd_flags =
  let probed =
    lazy
      (let compile ~flags ~data = probe_compiles ~flags ~data () in
       let guard =
         "#if !defined(__AVX2__) || !defined(__FMA__)\n\
          #error \"target lacks AVX2/FMA\"\n\
          #endif\n\
          int ocannl_simd_probe;\n"
       in
       let trivial = "int ocannl_simd_probe;\n" in
       if not (compile ~flags:(String.strip (arch_flags ())) ~data:guard) then ""
       else if compile ~flags:"-mavx2 -mfma -ftree-vectorize" ~data:trivial then
         "-mavx2 -mfma -ftree-vectorize"
       else if compile ~flags:"-mavx2 -mfma" ~data:trivial then "-mavx2 -mfma"
       else "")
  in
  fun () ->
    match Utils.get_global_arg ~default:"auto" ~arg_name:"cc_backend_simd_flags" with
    | "auto" -> Lazy.force probed
    | flags -> flags

(* Pool-backed Grid rendering (docs/proposals/gh-ocannl-164.md): eligible outermost [Grid] loops
   render as chunked parallel loops on a process-global native pool -- libdispatch's
   [dispatch_apply] on macOS (blocks; no pool state in the compiled kernel), OpenMP elsewhere
   (libgomp/libomp is likewise process-global). The worker threads run pure C on raw kernel
   arguments; the OCaml runtime is never involved. "auto" probes what the configured compiler
   accepts, once per process. *)
let parallel_grid_syntax_setting =
  let probed =
    lazy
      ((* Probes link full executables: a compiler can accept the flags yet lack the runtime library
          at link time (clang without libomp), and the kernel command compiles and links in one
          step. *)
       let dispatch_src =
         "#include <dispatch/dispatch.h>\n\
          int main(void) {\n\
         \  dispatch_apply(1, DISPATCH_APPLY_AUTO, ^(size_t i) { (void)i; });\n\
         \  return 0;\n\
          }\n"
       in
       let omp_src =
         "int main(void) {\n\
         \  float a[4] = {0.0f, 0.0f, 0.0f, 0.0f};\n\
          #pragma omp parallel for\n\
         \  for (int i = 0; i < 4; ++i) a[i] += 1.0f;\n\
         \  return (int)a[0] - 1;\n\
          }\n"
       in
       if probe_compiles ~mode:`Link ~flags:"" ~data:dispatch_src () then `Dispatch
       else if probe_compiles ~mode:`Link ~flags:"-fopenmp" ~data:omp_src () then `Openmp
       else `None)
  in
  fun () ->
    match String.lowercase (Utils.get_global_arg ~default:"auto" ~arg_name:"cc_parallel_grid") with
    | "auto" -> Lazy.force probed
    | "dispatch" -> `Dispatch
    | "openmp" -> `Openmp
    | "none" | "false" -> `None
    | s -> invalid_arg ("cc_parallel_grid: expected auto | dispatch | openmp | none, got " ^ s)

(* Explicit SIMD width for [Vectorized] loops (gh-ocannl-164 follow-up): vector register bytes for
   the GCC/Clang vector-extension rendering in [C_syntax]. Auto (-1 or unset): 32 bytes when the
   SIMD probe found AVX2, else 16 (NEON width; clang/gcc lower 16-byte vectors natively on ARM). 0
   disables explicit emission (auto-vectorization pragmas remain). *)
let vector_bytes_setting () =
  match Int.of_string @@ Utils.get_global_arg ~default:"-1" ~arg_name:"cc_vector_bytes" with
  | n when n >= 0 -> n
  | _ -> if String.is_substring (simd_flags ()) ~substring:"avx2" then 32 else 16

let parallel_grid_chunks_setting () =
  match Int.of_string @@ Utils.get_global_arg ~default:"0" ~arg_name:"cc_parallel_chunks" with
  | n when n > 0 -> n
  (* Auto: a small multiple of the core count -- enough chunks that uneven per-chunk cost
     load-balances, few enough that per-chunk overhead stays negligible. *)
  | _ -> 4 * Stdlib.Domain.recommended_domain_count ()

module Tn = Tnode

type library = { lib : (Dl.library[@sexp.opaque]); libname : string } [@@deriving sexp_of]

type procedure = {
  bindings : Indexing.unit_bindings;
  name : string;
  result : library;
  kparams : (string * kparam_source) list;
}
[@@deriving sexp_of]

let get_global_run_id =
  let next_id = ref 0 in
  fun () ->
    Int.incr next_id;
    if !next_id < 0 then next_id := 0;
    !next_id

let%track7_sexp c_compile_and_load ~f_path =
  let base_name : string = Stdlib.Filename.chop_extension f_path in
  (* There can be only one library with a given name, the object gets cached. Moreover, [Dl.dlclose]
     is not required to unload the library, although ideally it should. *)
  let run_id = Int.to_string @@ get_global_run_id () in
  let libname =
    let file_stem = Stdlib.Filename.chop_extension @@ Stdlib.Filename.basename f_path in
    if Utils.get_global_flag ~default:false ~arg_name:"output_dlls_in_build_directory" then
      (* Use only the path from f_path for the linked library libname *)
      base_name ^ "_run_id_" ^ run_id ^ if Sys.win32 then ".dll" else ".so"
    else
      (* Use temp_file without the run_id component *)
      Stdlib.Filename.temp_file file_stem (if Sys.win32 then ".dll" else ".so")
  in
  (try Stdlib.Sys.remove libname with _ -> ());
  let kernel_link_flags =
    match Sys.os_type with
    | "Unix" ->
        if Stdlib.Sys.command "uname -s | grep -q Darwin" = 0 then
          "-bundle -undefined dynamic_lookup"
        else "-shared -fPIC"
    | "Win32" | "Cygwin" -> "-shared"
    | _ -> "-shared -fPIC"
  in
  let temp_log = Stdlib.Filename.temp_file "ocannl_cc_" ".log" in
  let compiler_flags =
    let optimization_flag = "-O" ^ Int.to_string (optimization_level ()) in
    let arch_flag = String.strip (arch_flags ()) in
    let simd_flag = String.strip (simd_flags ()) in
    let fast_math_flag = if fast_math_enabled () then Some "-ffast-math" else None in
    (* [-fopenmp] must also reach the link step (this command compiles and links); harmless for
       kernels without parallel Grid loops. *)
    let parallel_flag =
      match parallel_grid_syntax_setting () with
      | `Openmp -> Some "-fopenmp"
      | `Dispatch | `None -> None
    in
    [
      Some optimization_flag;
      Option.some_if (not (String.is_empty arch_flag)) arch_flag;
      Option.some_if (not (String.is_empty simd_flag)) simd_flag;
      fast_math_flag;
      parallel_flag;
    ]
    |> List.filter_opt |> String.concat ~sep:" "
  in
  let cmdline : string =
    Printf.sprintf "%s %s %s -o %s %s > %s 2>&1" (compiler_command ()) f_path compiler_flags libname
      kernel_link_flags temp_log
  in
  (* Debug: log the command if debugging is enabled *)
  [%log3 "command", cmdline];
  let rc : int = Stdlib.Sys.command cmdline in
  (if rc <> 0 then (
     let compiler_output =
       try Stdio.In_channel.read_all temp_log with _ -> "(unable to read compiler output)"
     in
     (try Stdlib.Sys.remove temp_log with _ -> ());
     let detail =
       Printf.sprintf
         "OCANNL cc backend: generated code failed to compile (exit code %d).\n\
          This is a bug in OCANNL. Please file an issue with the generated .c file at %s\n\
          Compilation command: %s\n\
          Compiler output:\n\
          %s"
         rc f_path cmdline compiler_output
     in
     raise
       (Schedule_outcome.Cause_at
          ( Schedule_outcome.Backend_compile,
            Schedule_outcome.Backend_rejected
              {
                backend = name;
                stage = "compiler";
                severity = Schedule_outcome.Compiler_bug;
                detail;
              } )))
   else try Stdlib.Sys.remove temp_log with _ -> ());
  (* Wait a moment for the file to be fully written on success *)
  let start_time = Unix.gettimeofday () in
  let timeout =
    Float.of_string
    @@ Utils.get_global_arg ~default:"10.0" ~arg_name:"cc_backend_post_compile_timeout"
  in
  while not (Stdlib.Sys.file_exists libname) do
    let elapsed = Unix.gettimeofday () -. start_time in
    if Float.(elapsed > timeout) then
      failwith
      @@ Printf.sprintf
           "Cc_backend.c_compile_and_load: compiled library %s not found after successful \
            compilation"
           libname;
    Unix.sleepf 0.001
  done;
  (* Expected to succeed on MacOS only. *)
  let verify_codesign =
    Utils.get_global_flag ~default:false ~arg_name:"cc_backend_verify_codesign"
  in
  (if verify_codesign then
     let null_device = if Sys.win32 then "nul" else "/dev/null" in
     let rc =
       Stdlib.Sys.command @@ Printf.sprintf "codesign -s - %s > %s 2>&1" libname null_device
     in
     if rc <> 0 then
       invalid_arg
       @@ Printf.sprintf
            "Cc_backend.c_compile_and_load: codesign failed with exit code %d for library %s" rc
            libname);
  (* Note: RTLD_DEEPBIND not available on MacOS. *)
  let result = { lib = Dl.dlopen ~filename:libname ~flags:[ RTLD_NOW ]; libname } in
  (match parallel_grid_syntax_setting () with
  | `Openmp ->
      (* Never dlclose kernels built with -fopenmp: unloading an object whose parallel regions
         executed is a documented GOMP restriction -- libgomp's pool threads can retain references
         into it, and the dlclose can drop libgomp itself (loaded only as this object's dependency)
         under its parked workers. Observed as a SIGSEGV shortly after a routine was collected
         (ubuntu CI, PR #97). Leak the mapping instead; kernels are small. *)
      ()
  | `Dispatch | `None ->
      let%track7_sexp finalize (lib : library) : unit = Dl.dlclose ~handle:lib.lib in
      Stdlib.Gc.finalise finalize result);
  result

module CC_syntax_config (Procs : sig
  val procs : Low_level.optimized array
end) =
struct
  include C_syntax.Pure_C_config (struct
    type nonrec buffer_ptr = buffer_ptr

    let procs = Procs.procs

    let full_printf_support =
      not @@ Utils.get_global_flag ~default:false ~arg_name:"prefer_backend_uniformity"
  end)

  let parallel_grid_syntax = parallel_grid_syntax_setting ()
  let parallel_grid_chunks = parallel_grid_chunks_setting ()
  let vector_bytes = vector_bytes_setting ()

  (* Override operation syntax to handle special precision types *)
  let ternop_syntax prec op v1 v2 v3 =
    match prec with
    | Ops.Bfloat16_prec _ ->
        (* For BFloat16, perform operations in float precision *)
        let open PPrint in
        let float_v1 = string "bfloat16_to_single(" ^^ v1 ^^ string ")" in
        let float_v2 = string "bfloat16_to_single(" ^^ v2 ^^ string ")" in
        let float_v3 = string "bfloat16_to_single(" ^^ v3 ^^ string ")" in
        let op_prefix, op_infix1, op_infix2, op_suffix = Ops.ternop_c_syntax Ops.single op in
        let float_result =
          group
            (string op_prefix ^^ float_v1 ^^ string op_infix1
            ^^ ifflat (space ^^ float_v2) (nest 2 (break 1 ^^ float_v2))
            ^^ string op_infix2
            ^^ ifflat (space ^^ float_v3) (nest 2 (break 1 ^^ float_v3))
            ^^ string op_suffix)
        in
        string "single_to_bfloat16(" ^^ float_result ^^ string ")"
    | Ops.Half_prec _ ->
        (* For Half, perform operations in float precision on non-native systems *)
        let open PPrint in
        let float_v1 = string "HALF_TO_FP(" ^^ v1 ^^ string ")" in
        let float_v2 = string "HALF_TO_FP(" ^^ v2 ^^ string ")" in
        let float_v3 = string "HALF_TO_FP(" ^^ v3 ^^ string ")" in
        let op_prefix, op_infix1, op_infix2, op_suffix = Ops.ternop_c_syntax Ops.single op in
        let float_result =
          group
            (string op_prefix ^^ float_v1 ^^ string op_infix1
            ^^ ifflat (space ^^ float_v2) (nest 2 (break 1 ^^ float_v2))
            ^^ string op_infix2
            ^^ ifflat (space ^^ float_v3) (nest 2 (break 1 ^^ float_v3))
            ^^ string op_suffix)
        in
        string "FP_TO_HALF(" ^^ float_result ^^ string ")"
    | Ops.Fp8_prec _ ->
        (* For FP8, perform operations in float precision *)
        let open PPrint in
        let float_v1 = string "fp8_to_single(" ^^ v1 ^^ string ")" in
        let float_v2 = string "fp8_to_single(" ^^ v2 ^^ string ")" in
        let float_v3 = string "fp8_to_single(" ^^ v3 ^^ string ")" in
        let op_prefix, op_infix1, op_infix2, op_suffix = Ops.ternop_c_syntax Ops.single op in
        let float_result =
          group
            (string op_prefix ^^ float_v1 ^^ string op_infix1
            ^^ ifflat (space ^^ float_v2) (nest 2 (break 1 ^^ float_v2))
            ^^ string op_infix2
            ^^ ifflat (space ^^ float_v3) (nest 2 (break 1 ^^ float_v3))
            ^^ string op_suffix)
        in
        string "single_to_fp8(" ^^ float_result ^^ string ")"
    | _ ->
        let op_prefix, op_infix1, op_infix2, op_suffix = Ops.ternop_c_syntax prec op in
        let open PPrint in
        group
          (string op_prefix ^^ v1 ^^ string op_infix1
          ^^ ifflat (space ^^ v2) (nest 2 (break 1 ^^ v2))
          ^^ string op_infix2
          ^^ ifflat (space ^^ v3) (nest 2 (break 1 ^^ v3))
          ^^ string op_suffix)

  let binop_syntax prec op v1 v2 =
    match op with
    | (Ops.Threefry4x32_crypto | Ops.Threefry4x32_light | Ops.Uint4x32_to_prec_uniform_lane) as op
      ->
        let call fn v1 v2 =
          let open PPrint in
          group (string (fn ^ "(") ^^ v1 ^^ string ", " ^^ v2 ^^ string ")")
        in
        C_syntax.rng_binop_syntax ~backend:"CC" ~call prec op v1 v2
    | _ -> (
        match prec with
        | Ops.Bfloat16_prec _ ->
            (* For BFloat16, perform all operations in float precision *)
            let open PPrint in
            let float_v1 = string "bfloat16_to_single(" ^^ v1 ^^ string ")" in
            let float_v2 = string "bfloat16_to_single(" ^^ v2 ^^ string ")" in
            let op_prefix, op_infix, op_suffix = Ops.binop_c_syntax Ops.single op in
            let float_result =
              group
                (string op_prefix ^^ float_v1 ^^ string op_infix
                ^^ ifflat (space ^^ float_v2) (nest 2 (break 1 ^^ float_v2))
                ^^ string op_suffix)
            in
            string "single_to_bfloat16(" ^^ float_result ^^ string ")"
        | Ops.Fp8_prec _ ->
            (* For FP8, perform all operations in float precision *)
            let open PPrint in
            let float_v1 = string "fp8_to_single(" ^^ v1 ^^ string ")" in
            let float_v2 = string "fp8_to_single(" ^^ v2 ^^ string ")" in
            let op_prefix, op_infix, op_suffix = Ops.binop_c_syntax Ops.single op in
            let float_result =
              group
                (string op_prefix ^^ float_v1 ^^ string op_infix
                ^^ ifflat (space ^^ float_v2) (nest 2 (break 1 ^^ float_v2))
                ^^ string op_suffix)
            in
            string "single_to_fp8(" ^^ float_result ^^ string ")"
        | Ops.Half_prec _ ->
            (* For Half, perform all operations in float precision on non-native systems *)
            let open PPrint in
            let float_v1 = string "HALF_TO_FP(" ^^ v1 ^^ string ")" in
            let float_v2 = string "HALF_TO_FP(" ^^ v2 ^^ string ")" in
            let op_prefix, op_infix, op_suffix = Ops.binop_c_syntax Ops.single op in
            let float_result =
              group
                (string op_prefix ^^ float_v1 ^^ string op_infix
                ^^ ifflat (space ^^ float_v2) (nest 2 (break 1 ^^ float_v2))
                ^^ string op_suffix)
            in
            string "FP_TO_HALF(" ^^ float_result ^^ string ")"
        | _ ->
            let op_prefix, op_infix, op_suffix = Ops.binop_c_syntax prec op in
            let open PPrint in
            group
              (string op_prefix ^^ v1 ^^ string op_infix
              ^^ ifflat (space ^^ v2) (nest 2 (break 1 ^^ v2))
              ^^ string op_suffix))

  let unop_syntax prec op v =
    match op with
    | Ops.Uint4x32_to_prec_uniform1 ->
        (* Heterogeneous op: the argument is uint4x32 whatever the result precision, so the
           per-precision float-bridging below must not wrap it (e.g. [fp8_to_single] on a uint4x32_t
           would not even compile); the builtin already returns the result precision's storage
           type. *)
        let op_prefix, op_suffix = Ops.unop_c_syntax prec op in
        let open PPrint in
        group (string op_prefix ^^ v ^^ string op_suffix)
    | _ -> (
        match prec with
        | Ops.Bfloat16_prec _ ->
            (* For BFloat16, perform operations in float precision *)
            let open PPrint in
            let float_v = string "bfloat16_to_single(" ^^ v ^^ string ")" in
            let op_prefix, op_suffix = Ops.unop_c_syntax Ops.single op in
            let float_result = group (string op_prefix ^^ float_v ^^ string op_suffix) in
            string "single_to_bfloat16(" ^^ float_result ^^ string ")"
        | Ops.Fp8_prec _ ->
            (* For FP8, perform operations in float precision *)
            let open PPrint in
            let float_v = string "fp8_to_single(" ^^ v ^^ string ")" in
            let op_prefix, op_suffix = Ops.unop_c_syntax Ops.single op in
            let float_result = group (string op_prefix ^^ float_v ^^ string op_suffix) in
            string "single_to_fp8(" ^^ float_result ^^ string ")"
        | Ops.Half_prec _ ->
            (* For Half, perform operations in float precision on non-native systems *)
            let open PPrint in
            let float_v = string "HALF_TO_FP(" ^^ v ^^ string ")" in
            let op_prefix, op_suffix = Ops.unop_c_syntax Ops.single op in
            let float_result = group (string op_prefix ^^ float_v ^^ string op_suffix) in
            string "FP_TO_HALF(" ^^ float_result ^^ string ")"
        | _ ->
            let op_prefix, op_suffix = Ops.unop_c_syntax prec op in
            let open PPrint in
            group (string op_prefix ^^ v ^^ string op_suffix))
end

(* Under [output_debug_files_in_build_directory] the generated source lives at the predictable
   shared path [build_files/<name>.c], which a concurrently running process compiling a same-named
   kernel can rewrite while our C compiler reads it — macOS CI hit this as an "extraneous closing
   brace" in a torn sum_serial.c (two tests, one base name). Compile from a private unique copy; the
   [build_files/] copy stays purely informational. Without debug files, [open_build_file] already
   returned a unique temp path — use it directly. *)
let compilation_copy ~name (build_file : Utils.build_file_channel) filtered_code =
  if Utils.settings.output_debug_files_in_build_directory then (
    let tmp = Stdlib.Filename.temp_file (name ^ "_") ".c" in
    Stdio.Out_channel.write_all tmp ~data:filtered_code;
    tmp)
  else build_file.f_path

let%diagn_sexp compile ~(name : string) bindings (lowered : Low_level.optimized) : procedure =
  let module Syntax = C_syntax.C_syntax (CC_syntax_config (struct
    let procs = [| lowered |]
  end))
  in
  let idx_params = Indexing.bound_symbols bindings in
  let build_file = Utils.open_build_file ~base_name:name ~extension:".c" in
  (* Launch dims are ignored: the C backends render annotated loops as serial [for] loops (legal
     absent barriers, which [pp_ll] rejects via [barrier_syntax = None]). *)
  let kparams, proc_doc, _launch = Syntax.compile_proc ~name idx_params lowered in
  let filtered_code =
    Syntax.filter_and_prepend_builtins ~includes:Builtins_cc.includes ~builtins:Builtins_cc.builtins
      ~proc_doc
  in
  (* Use ribbon = 1.0 for usual code formatting, width 110 *)
  Out_channel.output_string build_file.oc filtered_code;
  build_file.finalize ();

  let result_library =
    c_compile_and_load ~f_path:(compilation_copy ~name build_file filtered_code)
  in
  { result = result_library; kparams; bindings; name }

let%diagn_sexp compile_batch ~names bindings (lowereds : Low_level.optimized option array) :
    procedure option array =
  let module Syntax = C_syntax.C_syntax (CC_syntax_config (struct
    let procs = Array.filter_opt lowereds
  end))
  in
  (* FIXME: do we really want all of them, or only the used ones? *)
  let idx_params = Indexing.bound_symbols bindings in
  let base_name =
    String.(
      strip ~drop:(equal_char '_')
      @@ common_prefix (Array.to_list @@ Array.concat_map ~f:Option.to_array names))
  in
  let build_file = Utils.open_build_file ~base_name ~extension:".c" in
  let params_and_docs =
    Array.map2_exn names lowereds ~f:(fun name_opt lowered_opt ->
        Option.map2 name_opt lowered_opt ~f:(fun name lowered ->
            Syntax.compile_proc ~name idx_params lowered))
  in
  let all_proc_docs =
    List.filter_map (Array.to_list params_and_docs) ~f:(Option.map ~f:(fun (_, doc, _) -> doc))
  in
  let combined_proc_doc = PPrint.separate PPrint.hardline all_proc_docs in
  let filtered_code =
    Syntax.filter_and_prepend_builtins ~includes:Builtins_cc.includes ~builtins:Builtins_cc.builtins
      ~proc_doc:combined_proc_doc
  in
  Out_channel.output_string build_file.oc filtered_code;
  build_file.finalize ();
  let result_library =
    c_compile_and_load ~f_path:(compilation_copy ~name:base_name build_file filtered_code)
  in
  (* Note: for simplicity, we share ctx_arrays across all contexts. *)
  Array.mapi params_and_docs ~f:(fun i opt_params_and_doc ->
      Option.bind opt_params_and_doc ~f:(fun (kparams, _doc, _launch) ->
          Option.map names.(i) ~f:(fun name -> { result = result_library; kparams; bindings; name })))

let%track3_sexp link_compiled ?lowered_bindings ~merge_buffer ~resolve ~runner_label ctx_buffers
    (code : procedure) =
  let name : string = code.name in
  let log_file_name = Utils.diagn_log_file [%string "debug-%{runner_label}-%{code.name}.log"] in
  (* When [lowered_bindings] is given (batch linking, e.g. fissioned segments of one routine), the
     static-index refs are shared: looked up by the [Static_idx] param's symbol rather than freshly
     minted, so one bindings assoc drives every procedure of the batch. *)
  let idx_ref s = match lowered_bindings with Some lb -> Indexing.find_exn lb s | None -> ref 0 in
  let run_variadic =
    [%log_level
      0;
      let rec link :
          'a 'b 'idcs.
          'idcs Indexing.bindings ->
          kparam_source list ->
          ('a -> 'b) Ctypes.fn ->
          ('a -> 'b, 'idcs, 'p1, 'p2) Indexing.variadic =
       fun (type a b idcs) (binds : idcs Indexing.bindings) kparams (cs : (a -> b) Ctypes.fn) ->
        match (binds, kparams) with
        | Empty, [] -> Indexing.Result (Foreign.foreign ~from:code.result.lib name cs)
        | Bind _, [] -> invalid_arg "Cc_backend.link: too few static index params"
        | Bind (_, bs), Static_idx s :: ps -> Param_idx (idx_ref s, link bs ps Ctypes.(int @-> cs))
        | Empty, Static_idx _ :: _ -> invalid_arg "Cc_backend.link: too many static index params"
        | bs, Log_file_name :: ps ->
            Param_1 (ref (Some log_file_name), link bs ps Ctypes.(string @-> cs))
        | bs, Merge_buffer :: ps ->
            (* The device's merge buffer is a [buffer_loc] set (lazily) by a transfer routine;
               resolve it to the backend pointer at execution time. *)
            Param_2f (resolve, merge_buffer, link bs ps Ctypes.(ptr void @-> cs))
        | bs, Kparam_ptr tn :: ps ->
            let c_ptr =
              match Map.find ctx_buffers tn with
              | Some loc -> resolve loc
              | None ->
                  (* After gh-ocannl-333 there is no host array to fall back on: every in-context
                     node must be present in [ctx_buffers] (allocated by [alloc_if_needed]). The
                     [buffer_loc -> base] resolution is backend-private (the shared layer hands us
                     locations, never pointers). *)
                  raise
                  @@ Utils.User_error
                       [%string
                         "Cc_backend.link_compiled: node %{Tn.debug_name tn} missing from context: \
                          %{Tn.debug_memory_mode tn.Tn.memory_mode_intent}"]
            in
            Param_2 (ref (Some c_ptr), link bs ps Ctypes.(ptr void @-> cs))
        | _, (Kparam_pool_slab _ | Kparam_pool_slots _) :: _ ->
            (* The C backend uses per-tnode pointer params ([`Per_param] codegen); only the Metal
               backend emits the pooled slab / slot parameters. *)
            invalid_arg "Cc_backend.link: unexpected pooled kparam (C uses per-tnode pointers)"
      in
      (* Reverse the input order because [Indexing.apply] will reverse it again. Important:
         [code.bindings] are traversed in the wrong order but that's OK because [link] only uses
         them to check the number of indices. *)
      let kparams = List.rev_map code.kparams ~f:(fun (_, p) -> p) in
      link code.bindings kparams Ctypes.(void @-> returning void)]
  in
  let%diagn_sexp work () : unit =
    [%log_result name];
    (* Stdio.printf "launching %s\n" name; *)
    Indexing.apply run_variadic ();
    if Utils.debug_log_from_routines () then
      Utils.log_debug_routine_file ~log_file_name ~stream_name:runner_label
  in
  ( (match lowered_bindings with
    | Some lb -> lb
    | None -> Indexing.lowered_bindings code.bindings run_variadic),
    Task.Task
      {
        (* In particular, keep code alive so it doesn't get unloaded. *)
        context_lifetime = (ctx_buffers, code);
        description = "executes " ^ code.name ^ " on " ^ runner_label;
        work;
      } )
