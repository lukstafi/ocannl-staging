(* The [-march] compile matrix and the reduction-loop census (gh-ocannl-650, gh-ocannl-649).

   Emitted kernels are only ever compiled for the build host, so every [#elif] written for a foreign
   target -- {!C_syntax.vec_fma_builtin}'s AVX-512, AVX512-FP16 and NEON rows -- is preprocessed away
   locally and a syntax error, a wrong arity or a type mismatch inside one ships silently.
   gh-ocannl-621 caught exactly that (gcc's aarch64 fp16 vector builtin is typed in [__fp16], not the
   [_Float16] the emission carries) with a shell loop over [build_files/*.c] and seven [-march]
   flags, in a scratch directory that died with the session. This is that loop, as a check.

   The generated [.c] is self-contained -- it includes only libc headers -- so compiling it under an
   arbitrary [-march] needs no hardware and no runtime, only the toolchain. What that buys over and
   above "it compiles" is the {b census} ({!Test_utils.Asm_census}) of the innermost loop carrying
   each accumulator update: also a compile-time property, also needing no hardware, and the thing
   that separates "gcc accepted the arm" from "gcc kept it in registers as one vector operation".
   gh-ocannl-621 used it to find the gh-ocannl-614 spill and the scalarization class; gh-ocannl-649
   used it to find that [Max]/[Min] were worse off than either.

   {1 Why the widths come from child processes}

   [Cc_backend]'s vector width is bound at module initialization, so one process emits at one width.
   The driver therefore re-executes ITSELF once per width with [OCANNL_CC_VECTOR_BYTES] set, and each
   child writes the kernel it emitted into a directory the driver hands it. Three settings are forced
   on every child besides the width:

   - [OCANNL_BACKEND=cc], because this is a check about the cc backend's C emission and nothing
     else. The stanza therefore pins its backend rather than declaring [OCANNL_BACKEND], and the
     driver itself never creates a context.
   - [OCANNL_CC_FP16_ARITHMETIC=native] with [OCANNL_FP16_ARITHMETIC=true], which is what makes the
     fp16 rows REAL on a host that merely has [_Float16] with every operation promoted to float (x86
     without AVX512-FP16 -- see [Cc_backend]'s three-state probe). Forcing the probe's answer changes
     what is EMITTED, which is the whole subject here, and nothing in this check executes a kernel,
     so the host's inability to run 16-bit arithmetic natively costs nothing. Every (precision, lane
     count) row of {!C_syntax.vec_fma_builtin}'s table is reachable this way.

   {1 What the claims are, and what they are not}

   Census counts are host-toolchain facts -- they move with the gcc version and the [-march] -- so
   the table goes to stderr and NOT into the golden. What the golden holds are inequalities that a
   misclassified move instruction cannot flip:

   - every kernel compiles clean under every accepted target at [-O2] and at [-O3];
   - every accumulator loop is FOUND: a census that answers "no loop carried the anchor" is a
     failure, because scoring only the instructions the good outcome produces is what made a fully
     scalarized loop read as "no FMA loop found" for gh-ocannl-621 -- a pass, arrived at from the
     failure;
   - no accumulator loop calls [fmax]/[fmin]/[fma]. An opaque call cannot be vectorized at any
     optimization level or grid size, and until gh-ocannl-649 the [Max]/[Min] combine compiled to
     exactly that -- at every [-march] and both optimization levels, because gcc will not contract
     [fmax] into [maxsd] without [-ffinite-math-only], those instructions having the wrong NaN
     behaviour;
   - no accumulator loop is scalarized on a target whose ISA has the packed operations it needs.

   A target the toolchain does not accept is reported with {!Verdict.skipped}, never dropped: a
   silently missing column reads exactly like a passing one. The aarch64 columns need a cross gcc,
   which installs without root ([tools/install-cross-gcc.sh]); point [AARCH64_CROSS_GCC] at it, or
   leave it unset and the check looks for [aarch64-linux-gnu-gcc] on [PATH]. *)

open Base
open Ocannl.Operation.DSL_modules
module LL = Ir.Low_level
module Tn = Ir.Tnode
module Idx = Ir.Indexing
module Cc_backend = Context.Cc_backend
module Census = Test_utils.Asm_census
module Generated = Test_utils.Generated

(* {1 The emission fixture}

   Six [Vectorized] accumulation loops in one kernel: a [Max] reduction and an FMA dot product, at
   f32, f64 and fp16 storage. One kernel per width rather than six keeps the compile matrix at three
   sources, and the census tells the loops apart by anchoring on each one's own source array through
   the DWARF line numbers [-g] leaves in the assembly.

   The extent is a multiple of [chains * lanes] at every width the ladder can pick (4 chains x up to
   32 lanes = 128, and 4096 = 32 * 128), so the serial tail loop has zero trips and constant bounds
   and gcc deletes it. That is not cosmetic: a surviving tail is a SMALLER innermost loop mentioning
   the same array, and the census -- which picks the smallest span carrying the anchor -- would then
   report the tail's profile as the vector loop's. *)

let extent = 4096
let routine = "census_kernel"

type kernel_loop = {
  anchor : string;  (** the source array whose lines identify this loop *)
  op_class : Census.op_class;
  what : string;  (** how the census table names the row *)
}

(* Directory handling through the OCaml stdlib rather than [Sys.command "mkdir -p"] /
   [Sys.command "rm -rf"]: those are POSIX shell spellings, and this test has to work on the native
   Windows environment AGENTS.md supports, where neither exists. Same reasoning as the null device
   in {!Test_utils.Asm_census.accepts}. *)
let rec mkdir_p dir =
  if not (Stdlib.Sys.file_exists dir) then (
    let parent = Stdlib.Filename.dirname dir in
    if not (String.equal parent dir) then mkdir_p parent;
    try Stdlib.Sys.mkdir dir 0o755 with Sys_error _ -> ())

let rec rm_rf path =
  if Stdlib.Sys.file_exists path then
    if Stdlib.Sys.is_directory path then (
      Array.iter (Stdlib.Sys.readdir path) ~f:(fun e ->
          rm_rf (Stdlib.Filename.concat path e));
      try Stdlib.Sys.rmdir path with Sys_error _ -> ())
    else try Stdlib.Sys.remove path with Sys_error _ -> ()

let build (emit_dir : string) =
  Utils.settings.output_debug_files_in_build_directory <- true;
  let backend_name = String.lowercase (Utils.get_global_arg ~arg_name:"backend" ~default:"cc") in
  Generated.init ~backend_name;
  let nodes = ref [] in
  let mk =
    let next = ref 8100 in
    fun ~prec ~dims label ->
      Int.incr next;
      let tn =
        Tn.create (Tn.Specified prec) ~id:!next ~label:[ label ]
          ~unpadded_dims:(lazy dims)
          ~padding:(lazy None)
          ()
      in
      Ll_test.materialize tn;
      nodes := tn :: !nodes;
      tn
  in
  let loops = ref [] in
  let llc =
    [ ("f32", Ir.Ops.single); ("f64", Ir.Ops.double); ("f16", Ir.Ops.half) ]
    |> List.map ~f:(fun (tag, prec) ->
           let da = mk ~prec ~dims:[| extent |] ("dta_" ^ tag) in
           let db = mk ~prec ~dims:[| extent |] ("dtb_" ^ tag) in
           let dacc = mk ~prec ~dims:[| 1 |] ("dtc_" ^ tag) in
           let cell tn = LL.Get (tn, [| Idx.Fixed_idx 0 |]) in
           let at tn s = LL.Get (tn, [| Idx.Iterator s |]) in
           let vloop body =
             let index = Idx.get_symbol () in
             LL.For_loop
               { index; from_ = 0; to_ = extent - 1; axis = LL.Vectorized; body = body index }
           in
           (* [Max] AND [Min], each at each precision. Emitting only [Max] would leave
              [__builtin_aarch64_fminv4sf]/[fminv2df]/[fminv8hf] in no generated source at all, so
              the whole point of the matrix -- catching a missing builtin, a wrong signature or a
              broken guard in an arm no local hardware selects -- would not reach the [Min] half of
              {!C_syntax.vec_minmax_builtin}. The host runtime test cannot cover them either: it
              preprocesses those arms away. *)
           let minmax op name =
             let src = mk ~prec ~dims:[| extent |] (name ^ "s_" ^ tag) in
             let acc = mk ~prec ~dims:[| 1 |] (name ^ "a_" ^ tag) in
             loops :=
               { anchor = name ^ "s_" ^ tag; op_class = Census.Max_min; what = name ^ "/" ^ tag }
               :: !loops;
             vloop (fun i ->
                 LL.Set
                   {
                     tn = acc;
                     idcs = [| Idx.Fixed_idx 0 |];
                     llsc = LL.Binop (op, (cell acc, prec), (at src i, prec));
                     debug = "";
                   })
           in
           let max_loop = minmax Ir.Ops.Max "max" in
           let min_loop = minmax Ir.Ops.Min "min" in
           loops := { anchor = "dta_" ^ tag; op_class = Census.Fma; what = "dot/" ^ tag } :: !loops;
           let dot_loop =
             vloop (fun j ->
                 LL.Set
                   {
                     tn = dacc;
                     idcs = [| Idx.Fixed_idx 0 |];
                     llsc =
                       LL.Ternop (Ir.Ops.FMA, (at da j, prec), (at db j, prec), (cell dacc, prec));
                     debug = "";
                   })
           in
           LL.Seq (max_loop, LL.Seq (min_loop, dot_loop)))
    |> List.reduce_exn ~f:(fun a b -> LL.Seq (a, b))
  in
  (* [optimize_scoped] injects the nest AS WRITTEN past [LL.optimize], which is what keeps the
     [Vectorized] annotations and the exact loop shape -- the subject of the census -- from being
     rewritten on the way to the backend. *)
  let o = Ll_test.optimize_scoped ~materialized:!nodes ~name:routine ~raw:llc llc in
  ignore (Ll_test.link ~name:routine o : Context.t * Context.routine);
  Stdio.Out_channel.write_all
    (Stdlib.Filename.concat emit_dir (routine ^ ".c"))
    ~data:(Generated.read routine);
  Stdio.Out_channel.write_all
    (Stdlib.Filename.concat emit_dir (routine ^ ".loops"))
    ~data:
      (String.concat ~sep:"\n"
         (List.rev_map !loops ~f:(fun { anchor; op_class; what } ->
              Printf.sprintf "%s\t%s\t%s" anchor
                (match op_class with Census.Fma -> "fma" | Census.Max_min -> "maxmin")
                what)))

(* {1 The driver} *)

let widths = [ 16; 32; 64 ]

let toolchains () =
  let host = Cc_backend.compiler_command () in
  let cross =
    match Stdlib.Sys.getenv_opt "AARCH64_CROSS_GCC" with
    | Some c when not (String.is_empty (String.strip c)) -> String.strip c
    | _ -> "aarch64-linux-gnu-gcc"
  in
  let t label command march note = { Census.label; command; march; note } in
  [
    t "x86-64" host "x86-64" "SSE2 baseline: no FMA arm, no AVX";
    t "x86-64-v2" host "x86-64-v2" "SSE4.2: packed compares, still no FMA arm";
    t "x86-64-v3" host "x86-64-v3" "AVX2 + FMA: the shipped, hardware-measured x86 rows";
    t "x86-64-v4" host "x86-64-v4" "AVX-512F/VL/BW/DQ: the masked 512-bit FMA builtins";
    t "sapphirerapids" host "sapphirerapids"
      "AVX512-FP16: the only x86 target with native 16-bit arithmetic";
    t "aarch64/armv8-a" cross "armv8-a" "NEON f32/f64 builtins, no fp16 arithmetic";
    t "aarch64/armv8.2-a+fp16" cross "armv8.2-a+fp16"
      "ARMv8.2-FP16: the NEON fp16 vector rows, typed in __fp16";
  ]

(* The environment a child emits under: this process's, with every setting the child must not
   inherit replaced. Removing them first matters -- [Unix.create_process_env] passes the array
   verbatim, and a duplicate key resolves to whichever copy the child's getenv finds first. *)
let child_env ~width ~dir =
  let forced =
    [
      ("OCANNL_BACKEND", "cc");
      ("OCANNL_CC_VECTOR_BYTES", Int.to_string width);
      ("OCANNL_CC_FP16_ARITHMETIC", "native");
      ("OCANNL_FP16_ARITHMETIC", "true");
      ("CC_MARCH_CENSUS_EMIT", dir);
    ]
  in
  let overridden = List.map forced ~f:fst in
  let inherited =
    Unix.environment () |> Array.to_list
    |> List.filter ~f:(fun kv ->
           match String.lsplit2 kv ~on:'=' with
           | Some (k, _) -> not (List.mem overridden k ~equal:String.equal)
           | None -> true)
  in
  Array.of_list (inherited @ List.map forced ~f:(fun (k, v) -> k ^ "=" ^ v))

let spawn_emit ~exe ~width ~dir =
  mkdir_p dir;
  (* The child's stdout is captured, not inherited: it must not reach the golden, and a failed child
     is worth reporting in full. *)
  let log = Stdlib.Filename.concat dir "emit.log" in
  let out = Unix.openfile log [ Unix.O_WRONLY; Unix.O_CREAT; Unix.O_TRUNC ] 0o644 in
  let pid = Unix.create_process_env exe [| exe |] (child_env ~width ~dir) Unix.stdin out Unix.stderr in
  let _, status = Unix.waitpid [] pid in
  Unix.close out;
  match status with
  | Unix.WEXITED 0 -> Ok ()
  | _ -> Error (try Stdio.In_channel.read_all log with _ -> "(no output)")

type emitted = { width : int; src_path : string; source : string; loops : kernel_loop list }

type row = {
  toolchain : string;
  target : string;  (** the [-march] value, for the ISA-capability questions below *)
  width : int;
  opt : int;
  what : string;
  profile : Census.t option;  (** [None]: no loop carried the anchor *)
}

let emit_all ~exe ~root =
  List.filter_map widths ~f:(fun width ->
      let dir = Stdlib.Filename.concat root (Printf.sprintf "w%d" width) in
      match spawn_emit ~exe ~width ~dir with
      | Error out ->
          Verdict.fail (Printf.sprintf "width %d: the emit child failed:\n%s" width out);
          None
      | Ok () ->
          let src_path = Stdlib.Filename.concat dir (routine ^ ".c") in
          let loops_path = Stdlib.Filename.concat dir (routine ^ ".loops") in
          if not (Stdlib.Sys.file_exists src_path && Stdlib.Sys.file_exists loops_path) then (
            Verdict.fail
              (Printf.sprintf "width %d: the emit child produced no kernel in %s" width dir);
            None)
          else
            let loops =
              Stdio.In_channel.read_lines loops_path
              |> List.filter_map ~f:(fun l ->
                     match String.split l ~on:'\t' with
                     | [ anchor; cls; what ] ->
                         Some
                           {
                             anchor;
                             op_class =
                               (if String.equal cls "fma" then Census.Fma else Census.Max_min);
                             what;
                           }
                     | _ -> None)
            in
            Some { width; src_path; source = Stdio.In_channel.read_all src_path; loops })

(* {2 What each target's ISA can be held to}

   A claim about an accumulator loop is only meaningful where the ISA has the operation the loop
   wants, so each row is asked of its own target:

   - the [Max]/[Min] combine is a packed compare, a packed bitwise select and nothing else, which
     every x86 target from SSE2 up and every aarch64 target has;
   - the FMA update needs a fused multiply-add INSTRUCTION, which on x86 arrives with
     [x86-64-v3] and on aarch64 is baseline NEON ([fmla]). Below that the emission falls back to
     per-lane [fmaf] -- a libm call, and deliberately so: an FMA rounds once, so on a machine
     without the instruction the library call is what preserves the rounding the scalar path has,
     and substituting [a * b + c] would make the vectorized and serial paths disagree. That
     fallback is a cost of correctness, not a defect, so the libm claim excludes it and says so;
   - 16-bit arithmetic is the exception under both headings: it needs AVX512-FP16 or ARMv8.2-FP16,
     and elsewhere gcc widens every lane to float, which is scalar work by construction;
   - and the requested WIDTH has to fit the target's registers. A 32- or 64-byte GNU C vector on an
     SSE2-only target is emulated -- gcc splits it into 16-byte pieces and spills the rest -- which
     is why the same [Max] loop that is 49 instructions with 42 vector operations and no stack
     traffic at [cc_vector_bytes=16] is 677 instructions with 128 scalar ones and 238 stack
     references at 32. That is the configuration being wrong for the machine, not the emission being
     wrong: [cc_vector_bytes] is auto-probed from the host when unset, and the widths here are
     forced precisely so that every arm of the builtin table gets compiled. *)

let is_fma_row what = String.is_prefix what ~prefix:"dot/"

(* The widest vector register the target actually has. *)
let target_vector_bytes ~target =
  let has s = String.is_substring target ~substring:s in
  if has "sapphirerapids" || has "v4" then 64 else if has "v3" then 32 else (* SSE2/SSE4, NEON *) 16

let isa_has ~target ~what ~width =
  let has s = String.is_substring target ~substring:s in
  let fp16_native = has "sapphirerapids" || has "+fp16" in
  if width > target_vector_bytes ~target then false
  else if String.is_suffix what ~suffix:"/f16" && not fp16_native then false
  else if is_fma_row what then has "v3" || has "v4" || has "sapphirerapids" || has "armv8"
  else true

(* {2 Assembler dialects this host cannot produce}

   Two review findings in a row were about assembly shapes a Linux/gcc box never emits, each of
   which silenced the census WHOLESALE rather than skewing a number -- Apple's dot-less [LBB0_9]
   labels, which made every branch target unrecognizable, and clang's
   [.file 0 "dir" "name" md5 0x...], whose checksum reads as the path and leaves no file number
   matched. Both fail the same way: every row reports "no loop", on a platform CI actually builds.

   Neither is reachable from a fixture compiled here, so they are pinned against SYNTHETIC assembly
   instead. That is not a weaker check for this particular property -- what is being tested is the
   parser's handling of a documented directive shape, and the shape is what is written down. *)

let dialect_probes () =
  (* Line 3 carries the anchor; the [.loc] directives below point at it. *)
  let source = "int f(void) {\n  /* prologue */\n  ACC = MAXOF(ACC, SRC);\n  return 0;\n}\n" in
  let anchor = Census.anchor_lines ~source ~patterns:[ "MAXOF" ] in
  let elf =
    String.concat ~sep:"\n"
      [ "\t.file\t\"census_kernel.c\"";
        "\t.file 1 \"/build/census_kernel.c\"";
        "f:"; ".L2:"; "\t.loc 1 3 12"; "\taddps\t%xmm1, %xmm0"; "\tjne\t.L2"; "\tret" ]
  in
  (* Apple's local labels carry no leading dot, and clang's [.file] carries a trailing checksum. *)
  let darwin_clang =
    String.concat ~sep:"\n"
      [ "\t.file\t0 \"/build\" \"census_kernel.c\" md5 0x0123456789abcdef0123456789abcdef";
        "\t.file\t1 \"/build\" \"census_kernel.c\" md5 0x0123456789abcdef0123456789abcdef";
        "_f:"; "LBB0_1:"; "\t.loc\t1 3 12"; "\taddps\t%xmm1, %xmm0"; "\tjne\tLBB0_1"; "\tretq" ]
  in
  let found asm =
    Option.is_some
      (Census.census Census.Max_min ~asm ~source_basename:(routine ^ ".c") ~anchor)
  in
  Verdict.p_all "the census reads both assembler dialects, not only this host's"
    [ ("gnu/elf", elf); ("apple-clang", darwin_clang) ]
    ~f:(fun (_, asm) -> Census.loop_edges ~asm > 0 && found asm)

let () =
  match Stdlib.Sys.getenv_opt "CC_MARCH_CENSUS_EMIT" with
  | Some dir -> build dir
  | None ->
      let exe = Stdlib.Sys.executable_name in
      let root = Stdlib.Filename.concat (Stdlib.Sys.getcwd ()) "cc_march_census_kernels" in
      rm_rf root;
      mkdir_p root;
      dialect_probes ();
      let emitted = emit_all ~exe ~root in
      Verdict.p_all "every requested vector width emitted a kernel" widths ~f:(fun w ->
          List.exists emitted ~f:(fun e -> e.width = w));
      let all = toolchains () in
      let available = List.filter all ~f:Census.accepts in
      Stdio.eprintf "\n=== cc kernel census (not part of the golden) ===\n";
      Stdio.eprintf "host toolchain: %s\n" (Cc_backend.compiler_command ());
      let failed_compiles = ref [] in
      let edges = ref [] in
      let rows =
        List.concat_map available ~f:(fun t ->
            List.concat_map emitted ~f:(fun e ->
                List.concat_map [ 2; 3 ] ~f:(fun opt ->
                    let asm_path =
                      Stdlib.Filename.concat root
                        (Printf.sprintf "%s-w%d-O%d.s"
                           (String.map t.Census.label ~f:(fun c ->
                                if Char.is_alphanum c then c else '_'))
                           e.width opt)
                    in
                    match
                      Census.compile t ~opt_level:opt ~src_path:e.src_path ~asm_path
                    with
                    | Error out ->
                        failed_compiles :=
                          Printf.sprintf "%s -O%d w%d" t.Census.label opt e.width
                          :: !failed_compiles;
                        Verdict.fail
                          (Printf.sprintf
                             "%s -O%d, width %d: the emitted kernel does not compile:\n%s"
                             t.Census.label opt e.width out);
                        []
                    | Ok () ->
                        let asm = Stdio.In_channel.read_all asm_path in
                        (* Recorded before any anchor is looked for: an ISA whose branch spelling
                           {!Census.is_branch} does not know reports every anchor missing at once,
                           which is a defect in the census and not a finding about the kernel. *)
                        edges :=
                          (Printf.sprintf "%s w%d -O%d" t.Census.label e.width opt,
                           Census.loop_edges ~asm)
                          :: !edges;
                        List.map e.loops ~f:(fun { anchor; op_class; what } ->
                            let c =
                              Census.census op_class ~asm ~source_basename:(routine ^ ".c")
                                ~anchor:(Census.anchor_lines ~source:e.source ~patterns:[ anchor ])
                            in
                            (match c with
                            | Some c ->
                                Stdio.eprintf "  %-24s w%-3d -O%d %-9s %s\n" t.Census.label e.width
                                  opt what (Census.to_line c)
                            | None ->
                                Stdio.eprintf
                                  "  %-24s w%-3d -O%d %-9s NO LOOP CARRIED THE ANCHOR\n"
                                  t.Census.label e.width opt what);
                            {
                              toolchain = t.Census.label;
                              target = t.Census.march;
                              width = e.width;
                              opt;
                              what;
                              profile = c;
                            }))))
      in
      Stdio.eprintf "=== end census ===\n\n";
      let describe r = Printf.sprintf "%s w%d -O%d %s" r.toolchain r.width r.opt r.what in
      (* One line per COLUMN, always -- printed by {!Verdict.skipped} where the toolchain is absent,
         which prints exactly the line the verified case prints. Without this the golden would have
         a line count that depended on whether the box has a cross gcc: [skipped] emits a stdout
         line of its own, so reporting only the absent columns would make a machine WITH the cross
         compiler and a machine without disagree about the expected output. Here the count is fixed
         at one per toolchain and it is the run's stderr that says which were really evaluated. *)
      List.iter all ~f:(fun t ->
          let name =
            Printf.sprintf "%s: the -march column compiles clean and its loops are censused"
              t.Census.label
          in
          if List.mem available t ~equal:phys_equal then
            let mine = List.filter rows ~f:(fun r -> String.equal r.toolchain t.Census.label) in
            Verdict.p name
              ((not (List.is_empty mine)) && List.for_all mine ~f:(fun r -> Option.is_some r.profile))
          else Verdict.skipped ~backend:(t.Census.label ^ ", " ^ t.Census.note) name);
      Verdict.p_empty "every emitted kernel compiles clean at -O2 and -O3 under every accepted -march"
        ~over:rows !failed_compiles;
      Verdict.p_all "every compiled kernel has recognizable loop edges" !edges ~f:(fun (_, n) ->
          n > 0);
      Verdict.p_all "every accumulator loop is found by the census" rows ~f:(fun r ->
          Option.is_some r.profile);
      let counts r = Option.map r.profile ~f:(fun p -> p.Census.counts) in
      let libm r = match counts r with Some c -> c.Census.libm_calls > 0 | None -> false in
      let scalarized r =
        match counts r with
        | Some c -> c.Census.scalar_fp_ops > 0 || c.Census.vector_ops = 0
        | None -> false
      in
      (* Each claim's population is named once and used twice: for the claim, and for the report of
         what violated it. Only a claim's OWN population is reported -- the populations exclude
         configurations the target cannot serve, and printing those too buries the handful of real
         offenders in a page of expected ones. *)
      let report label population ~f =
        let bad = List.filter population ~f in
        if not (List.is_empty bad) then (
          Stdio.eprintf "  %d row(s) violating %S (not part of the golden):\n" (List.length bad)
            label;
          List.iter bad ~f:(fun r ->
              Stdio.eprintf "    %s -> %s\n" (describe r)
                (match r.profile with Some p -> Census.to_line p | None -> "no loop")))
      in
      let claim_none label population ~f =
        Verdict.p_none label population ~f;
        report label population ~f
      in
      let served = List.filter rows ~f:(fun r -> isa_has ~target:r.target ~what:r.what ~width:r.width) in
      (* The gh-ocannl-649 claim proper. Unconditional across the matrix, unlike the two below:
         [Max]/[Min] have a whole-vector spelling on every target here at every width -- an emulated
         wide vector still compiles to packed compares and selects, just more of them -- so a libm
         call in one of these loops is a defect wherever it appears. *)
      claim_none "no Max/Min accumulator loop calls libm"
        (List.filter rows ~f:(fun r -> not (is_fma_row r.what)))
        ~f:libm;
      claim_none "no FMA accumulator loop calls libm where the ISA has a fused multiply-add"
        (List.filter served ~f:(fun r -> is_fma_row r.what))
        ~f:libm;
      claim_none "no accumulator loop is scalarized where the ISA has the operation it wants" served
        ~f:scalarized
