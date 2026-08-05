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

let fast_math_enabled () = Utils.get_global_flag ~default:false ~arg_name:"cc_backend_fast_math"

(* Toolchain probing -- resolving the compiler command, [arch_flags], [simd_flags],
   [parallel_grid_syntax_setting], [fp16_arithmetic_support] -- costs the better part of a dozen
   subprocesses per process: one [ocamlc -config], plus compile and link probes for each of the
   rest. Each is memoized by a [lazy], so the memoization is process-local, while a [dune build]
   runs a couple hundred test executables. On Windows, where process creation is an order of
   magnitude costlier than on Unix, that probing dominates short tests -- measured on one Windows
   box, test/operations/hello_world_dim1x1 runs in 1.52s probing against 1.00s cached.

   So memoize across processes too, in files under [probe_cache_dir]. The key covers everything
   that can change an answer without changing a file's identity: the settings feeding the probes,
   plus a digest of PATH standing in for which toolchain is active (an opam switch change, a
   different compiler in front). A stale entry would silently select wrong ISA flags, so the key
   errs towards re-probing; [cc_backend_probe_cache=false] bypasses the files entirely.

   Every failure degrades to probing in-process rather than raising: an unwritable temp directory, a
   torn or truncated file, a rename losing a race. Dune runs these executables in parallel, so
   several processes can probe at once and race to publish -- harmless, since they compute the same
   answer, and one file per probe keeps a publication from ever being interleaved with another's. *)
let probe_cache_marker = "ocannl-cc-probe-v1\n"

(* The cached values are interpolated straight into [Sys.command], so where they are stored is a
   privilege boundary, not a detail: a predictable name under a world-writable POSIX /tmp would let
   any local account pre-create a file bearing our marker and thereby choose the compiler command
   OCANNL runs. Hence a per-user directory created 0700, refused unless it really is a directory we
   own with no group/other access -- a squatted path (or a symlink, which is why this is [lstat] and
   not [stat]) fails the test and probing simply stays in-process, as it was before this cache.
   Ownership and permissions are checked on POSIX only: Windows has no equivalent exposure, its
   temp directory being per-user already, and it reports both halves synthetically -- [getuid]
   answers 1 where [lstat] answers uid 0, and the mode always reads 0o777 -- so the test would
   reject OCANNL's own directory and quietly disable the cache on the platform that needs it most.
   That the path is a directory rather than a planted symlink is still checked everywhere. *)
let probe_cache_dir =
  lazy
    (let dir =
       Stdlib.Filename.concat
         (Stdlib.Filename.get_temp_dir_name ())
         (Printf.sprintf "ocannl_cc_probes_%d" (Unix.getuid ()))
     in
     try
       (try Unix.mkdir dir 0o700 with Unix.Unix_error (Unix.EEXIST, _, _) -> ());
       let st = Unix.lstat dir in
       let ours =
         match st.Unix.st_kind with
         | Unix.S_DIR ->
             Sys.win32 || (st.Unix.st_uid = Unix.getuid () && st.Unix.st_perm land 0o077 = 0)
         | _ -> false
       in
       Option.some_if ours dir
     with _ -> None)

let probe_cache_path =
  let key_digest =
    lazy
      (* Raw settings, not resolved values: [arch_flags] is itself one of the probes cached here, so
         keying on its result would both invert the dependency and spend the probes the key exists
         to avoid. Nothing is lost -- what it resolves to is a function of the compiler and the
         target, and both are already pinned by the entries below. *)
      (let key =
         String.concat ~sep:"\000"
           [
             Utils.get_global_arg ~default:"" ~arg_name:"cc_backend_compiler_command";
             Utils.get_global_arg ~default:"auto" ~arg_name:"cc_backend_arch_flags";
             Option.value (Stdlib.Sys.getenv_opt "PATH") ~default:"";
           ]
       in
       String.prefix (Stdlib.Digest.to_hex (Stdlib.Digest.string key)) 16)
  in
  fun ~name ->
    Option.map (Lazy.force probe_cache_dir) ~f:(fun dir ->
        Stdlib.Filename.concat dir
          (Printf.sprintf "ocannl_cc_probe_%s_%s.txt" name (Lazy.force key_digest)))

(* [compute] must be deterministic given the key above: its result is what every later process on
   this machine will use. [validate] rejects a cached value the caller cannot interpret, so that a
   foreign file carrying our marker re-probes instead of propagating a parse failure. *)
let cached_probe ?(validate = fun _ -> true) ~name ~compute () =
  match
    if Utils.get_global_flag ~default:true ~arg_name:"cc_backend_probe_cache" then
      probe_cache_path ~name
    else None
  with
  | None -> compute ()
  | Some path -> (
    let cached =
      try
        let data = Stdio.In_channel.read_all path in
        (* The marker separates "the probe answered with the empty string" -- which [simd_flags]
           legitimately does -- from a half-written or foreign file. *)
        if String.is_prefix data ~prefix:probe_cache_marker then
          let value = String.drop_prefix data (String.length probe_cache_marker) in
          Option.some_if (validate value) value
        else None
      with _ -> None
    in
    match cached with
    | Some value -> value
    | None ->
        let value = compute () in
        (try
           (* Publish by rename so a concurrent reader sees either the old file or the whole new
              one, never a partial write. Staging inside the same (private) directory keeps the
              rename on one volume, which it requires, and keeps the staging file unreadable to
              anyone else too. *)
           let tmp =
             Stdlib.Filename.temp_file ~temp_dir:(Stdlib.Filename.dirname path)
               "ocannl_cc_probe_" ".tmp"
           in
           Stdio.Out_channel.write_all tmp ~data:(probe_cache_marker ^ value);
           try Stdlib.Sys.rename tmp path
           with e ->
             (try Stdlib.Sys.remove tmp with _ -> ());
             raise e
         with _ -> ());
        value)

let compiler_command =
  let resolved =
    (* TODO: there's a direct way to get the compiler command from the OCaml compiler. *)
    lazy
      (cached_probe ~name:"compiler"
         ~compute:(fun () ->
           let ic = Unix.open_process_in "ocamlc -config" in
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
         ())
  in
  fun () ->
    (* Resolve lazily rather than passing [Lazy.force resolved] as [~default]: that spawns
       [ocamlc -config] even when the setting makes the answer irrelevant. An empty setting means
       unset -- which is how [ocannl_config.reference] spells it. *)
    match Utils.get_global_arg ~default:"" ~arg_name:"cc_backend_compiler_command" with
    | "" -> Lazy.force resolved
    | command -> command

(* gh-ocannl-164: explicit SIMD flags appended to the compiler invocation. The "auto" default
   probes in two stages (once per process, and once per machine via [cached_probe]). Stage 1
   test-compiles a translation unit that #errors
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

(* The flag that tells the compiler to target the host CPU. Its spelling is architecture-specific,
   and getting it wrong is worse than passing nothing: Apple clang accepts [-march=native] on arm64
   and *downgrades* the target with it -- 22 [__ARM_FEATURE_*] macros against 26 with no flag and 33
   with [-mcpu=native], losing [__ARM_FEATURE_FP16_VECTOR_ARITHMETIC] among them, so a machine with
   native 16-bit arithmetic looked like one without (gh-ocannl-516). On x86 the mistake runs the
   other way: [-mcpu=] is an alias for [-mtune=] there, which selects scheduling but not the ISA,
   so it would silently forgo AVX2.

   "auto" (the default) therefore asks the target which family it is in and then probes that
   family's spelling, falling back to no flag -- the compiler's own default target, which is the
   host on Apple and a portable baseline elsewhere. Any other config value is passed through
   verbatim, so [cc_backend_arch_flags=-march=native] restores the old behavior exactly. *)
let arch_flags =
  let probed =
    lazy
      (cached_probe ~name:"arch"
         ~compute:(fun () ->
           let is_arm =
             probe_compiles ~flags:""
               ~data:"#if !defined(__aarch64__) && !defined(__arm__) && !defined(_M_ARM64)\n\
                      #error \"not ARM\"\n\
                      #endif\n\
                      int ocannl_arch_probe;\n"
               ()
           in
           let trivial = "int ocannl_arch_probe;\n" in
           let candidate = if is_arm then "-mcpu=native" else "-march=native" in
           if probe_compiles ~flags:candidate ~data:trivial () then candidate else "")
         ())
  in
  fun () ->
    match Utils.get_global_arg ~default:"auto" ~arg_name:"cc_backend_arch_flags" with
    | "auto" -> Lazy.force probed
    (* The portable baseline, spelled as a word because a config source cannot carry an empty
       value (an empty setting means "unset"), and the `reproducible` profile has to be able to
       pin it. *)
    | "none" -> ""
    | flags -> flags

(* Floating-point contraction beyond what the codegen selects explicitly (it emits `fmaf` where it
   wants an FMA): whether the compiler is free to fuse a*b+c on its own is compiler- and
   target-discretionary, so a reproducible run pins it off. "auto" (the default) passes no flag,
   leaving the compiler's own default -- which is what every OCANNL release before gh-ocannl-559
   did. *)
let fp_contract_flag () =
  match String.lowercase (String.strip (Utils.get_global_arg ~default:"auto" ~arg_name:"cc_backend_fp_contract")) with
  | "auto" -> None
  | ("off" | "on" | "fast") as mode -> Some ("-ffp-contract=" ^ mode)
  | other -> invalid_arg ("cc_backend_fp_contract: expected auto | off | on | fast, got " ^ other)

let simd_flags =
  let probed =
    lazy
      (cached_probe ~name:"simd"
         ~compute:(fun () ->
           let compile ~flags ~data = probe_compiles ~flags ~data () in
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
         ())
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
   accepts, once per process (and once per machine, via [cached_probe]). *)
let parallel_grid_of_string = function
  | "dispatch" -> Some `Dispatch
  | "openmp" -> Some `Openmp
  | "none" | "false" -> Some `None
  | _ -> None

let parallel_grid_syntax_setting =
  let probed =
    lazy
      (let probe () =
         (* Probes link full executables: a compiler can accept the flags yet lack the runtime
            library at link time (clang without libomp), and the kernel command compiles and links
            in one step. *)
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
         if probe_compiles ~mode:`Link ~flags:"" ~data:dispatch_src () then "dispatch"
         else if probe_compiles ~mode:`Link ~flags:"-fopenmp" ~data:omp_src () then "openmp"
         else "none"
       in
       let cached =
         cached_probe ~name:"parallel_grid"
           ~validate:(fun s -> Option.is_some (parallel_grid_of_string s))
           ~compute:probe ()
       in
       (* [validate] already rejected anything unparseable, so the probe's own vocabulary is the
          only remaining source. *)
       Option.value_exn ~message:"Cc_backend: unexpected parallel-grid probe result"
         (parallel_grid_of_string cached))
  in
  fun () ->
    let setting =
      String.lowercase (Utils.get_global_arg ~default:"auto" ~arg_name:"cc_parallel_grid")
    in
    if String.equal setting "auto" then Lazy.force probed
    else
      match parallel_grid_of_string setting with
      | Some mode -> mode
      | None ->
          invalid_arg ("cc_parallel_grid: expected auto | dispatch | openmp | none, got " ^ setting)

(* gh-ocannl-516: whether the target has native fp16 arithmetic. Three states, and the middle one
   is the reason a boolean will not do:

   - [`None]: no [_Float16] at all. Half is emulated -- stored as uint16, computed through float.
   - [`Promoted]: the type exists and its arithmetic is correct, but the compiler implements it by
     promoting to float (x86-64 without AVX512-FP16). Better than bf16's explicit round-trips
     because the compiler owns the promotion and can keep values in registers, but there is no
     lane-count win, so the vector renderings and the cost model must not expect one.
   - [`Native]: genuine 16-bit vector arithmetic (ARMv8.2-FP16, AVX512-FP16) -- twice f32's lanes.

   All three are C-preprocessor facts, resolved when [cc] compiles a kernel, while the renderer
   decides what to emit in OCaml well before that. Probing the compiler once per process and
   carrying the answer in [hardware_limits] puts a target capability where target capabilities
   already live, and keeps a half kernel's body single-armed. *)
let fp16_arithmetic_of_string = function
  | "none" | "false" -> Some `None
  | "promoted" -> Some `Promoted
  | "native" | "true" -> Some `Native
  | _ -> None

let fp16_arithmetic_support =
  let probed =
    lazy
      (let cached =
         cached_probe ~name:"fp16"
           ~validate:(fun s -> Option.is_some (fp16_arithmetic_of_string s))
           ~compute:(fun () ->
             let flags = String.strip (arch_flags () ^ " " ^ simd_flags ()) in
             (* Vector arithmetic, not just the type: a target can define [__FLT16_MAX__] and still
                reject [_Float16] in a [vector_size] type. *)
             let typed =
               "#ifndef __FLT16_MAX__\n\
                #error \"no _Float16\"\n\
                #endif\n\
                typedef _Float16 ocannl_v8h __attribute__((vector_size(16)));\n\
                ocannl_v8h ocannl_fp16_probe(ocannl_v8h a, ocannl_v8h b, ocannl_v8h c) { return a \
                * b + c; }\n"
             in
             let native =
               "#if !defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && !defined(__AVX512FP16__)\n\
                #error \"fp16 arithmetic is promoted to float, not native\"\n\
                #endif\n\
                int ocannl_fp16_native_probe;\n"
             in
             if not (probe_compiles ~flags ~data:typed ()) then "none"
             else if probe_compiles ~flags ~data:native () then "native"
             else "promoted")
           ()
       in
       Option.value_exn ~message:"Cc_backend: unexpected fp16-arithmetic probe result"
         (fp16_arithmetic_of_string cached))
  in
  fun () ->
    let setting =
      String.lowercase (Utils.get_global_arg ~default:"auto" ~arg_name:"cc_fp16_arithmetic")
    in
    if String.equal setting "auto" then Lazy.force probed
    else
      match fp16_arithmetic_of_string setting with
      | Some support -> support
      | None ->
          invalid_arg
            ("cc_fp16_arithmetic: expected auto | none | promoted | native, got " ^ setting)

let has_native_fp16_arithmetic () =
  match fp16_arithmetic_support () with `Native -> true | `Promoted | `None -> false

(* Explicit SIMD width for [Vectorized] loops (gh-ocannl-164 follow-up): vector register bytes for
   the GCC/Clang vector-extension rendering in [C_syntax]. Auto (-1 or unset): 32 bytes when the
   SIMD probe found AVX2, else 16 (NEON width; clang/gcc lower 16-byte vectors natively on ARM). 0
   disables explicit emission (auto-vectorization pragmas remain). *)
let vector_bytes_setting () =
  match Int.of_string @@ Utils.get_global_arg ~default:"-1" ~arg_name:"cc_vector_bytes" with
  | n when n >= 0 -> n
  | _ -> if String.is_substring (simd_flags ()) ~substring:"avx2" then 32 else 16

(* Whether the kernel .so is a macOS bundle or an ELF/PE shared object. Distinguishing the two BSDs
   from Darwin needs [uname], and this used to shell out once per kernel compile -- on a machine
   compiling thousands of kernels per build, an entirely redundant `sh -c` plus `uname` plus `grep`
   each time. The answer cannot change within a process. *)
let kernel_link_flags =
  lazy
    (match Sys.os_type with
    | "Unix" ->
        if Stdlib.Sys.command "uname -s | grep -q Darwin" = 0 then
          "-bundle -undefined dynamic_lookup"
        else "-shared -fPIC"
    | "Win32" | "Cygwin" -> "-shared"
    | _ -> "-shared -fPIC")

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
  let kernel_link_flags = Lazy.force kernel_link_flags in
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
      fp_contract_flag ();
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

  let ident_blacklist = ident_blacklist @ C_syntax.builtin_idents Builtins_cc.builtins
  let parallel_grid_syntax = parallel_grid_syntax_setting ()
  let parallel_grid_chunks = parallel_grid_chunks_setting ()
  let vector_bytes = vector_bytes_setting ()

  (* gh-ocannl-517: a CPU has no 16-bit arithmetic -- the narrow arms of the operator renderings
     below are widen/op/narrow round-trips through f32, and [_Float16] arithmetic, where the type
     exists at all, is promoted to float by the compiler on targets without AVX512-FP16. So the
     arithmetic runs in f32 and the narrow format is a storage format: one widen per load, one
     narrow per store. Under [narrow_compute_f32 = false] the narrow arms below take over again and
     every operator rounds to the target's storage precision (the pre-gh-517 semantics). *)
  let compute_prec prec =
    match prec with
    (* gh-ocannl-516: fp16 is the one narrow format a CPU can execute natively, and where it does,
       computing in it doubles the lane count against f32. Opt-in, and only where the arithmetic is
       genuinely 16-bit -- on a target that merely promotes to float there is no throughput to win,
       so the policy is ignored rather than silently costing mantissa. *)
    | Ops.Half_prec _
      when (Numerics.get ()).fp16_arithmetic && has_native_fp16_arithmetic () ->
        prec
    | _ when Ops.is_narrow_float prec && (Numerics.get ()).narrow_compute_f32 -> Ops.single
    | _ -> prec

  (* The explicit vector renderings work at the compute precision, so admitting fp16 here is
     admitting native 16-bit vector arithmetic -- [vec_ext_typ] mints a [HALF_T] vector and the
     lane count doubles.

     The condition is the target's, not the policy's, and the two are not the same question.
     Arriving here at [Half_prec] does not mean [fp16_arithmetic] chose it: [narrow_compute_f32 =
     false] also leaves half alone, on any target, and there [HALF_T] is [uint16_t] -- a vector of
     those would do integer arithmetic on raw half bit patterns and quietly corrupt the loop. So
     ask the probe directly. Declining on a merely [`Promoted] target costs nothing that existed
     before gh-ocannl-516, when half never vectorized at all. *)
  let vector_prec_ok prec =
    match prec with
    | Ops.Single_prec _ | Ops.Double_prec _ -> true
    | Ops.Half_prec _ -> has_native_fp16_arithmetic ()
    | _ -> false

  (* Override operation syntax to handle special precision types *)
  let ternop_syntax prec op v1 v2 v3 =
    match (prec, op) with
    (* gh-ocannl-516: at fp16 compute precision the FMA goes through the shared macro, so the
       scalar path and the vector rendering's per-lane fallback are the same operation -- see
       [C_syntax.vec_acc_fma]. *)
    | Ops.Half_prec _, Ops.FMA ->
        let open PPrint in
        group
          (string "OCANNL_HALF_FMA(" ^^ v1 ^^ string ","
          ^^ ifflat (space ^^ v2) (nest 2 (break 1 ^^ v2))
          ^^ string ","
          ^^ ifflat (space ^^ v3) (nest 2 (break 1 ^^ v3))
          ^^ string ")")
    | _ -> (
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
        (* [FLOAT_TO_HALF], not [FP_TO_HALF]: the latter is the *identity* on a native [_Float16]
           target, which would leave the f32 result of a library call ([expf], [sqrtf], [fmaxf]) at
           f32 -- and C's usual arithmetic conversions then keep every enclosing operator at f32
           too, all the way to the store. The fp16-arithmetic policy promises fp16 intermediates,
           10-bit mantissa and a 65504 ceiling included, so the narrowing has to be a real cast
           (gh-ocannl-516 review). A no-op where the ring operators already compute in [_Float16],
           and unchanged on the emulated path, where both macros are [float_to_half_emulated].
           Same reasoning at the two [FLOAT_TO_HALF] sites below. *)
        string "FLOAT_TO_HALF(" ^^ float_result ^^ string ")"
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
          ^^ string op_suffix))

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
            string "FLOAT_TO_HALF(" ^^ float_result ^^ string ")"
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
            string "FLOAT_TO_HALF(" ^^ float_result ^^ string ")"
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
