(* The [-march] compile matrix and the kernel-loop census (gh-ocannl-650, gh-ocannl-649,
   gh-ocannl-752).

   Emitted kernels are only ever compiled for the build host, so every [#elif] written for a foreign
   target -- {!C_syntax.vec_fma_builtin}'s AVX-512, AVX512-FP16 and NEON rows -- is preprocessed
   away locally and a syntax error, a wrong arity or a type mismatch inside one ships silently.
   gh-ocannl-621 caught exactly that (gcc's aarch64 fp16 vector builtin is typed in [__fp16], not
   the [_Float16] the emission carries) with a shell loop over [build_files/*.c] and seven [-march]
   flags, in a scratch directory that died with the session. This is that loop, as a check.

   The generated [.c] is self-contained -- it includes only libc headers -- so compiling it under an
   arbitrary [-march] needs no hardware and no runtime, only the toolchain. What that buys over and
   above "it compiles" is the {b census} ({!Test_utils.Asm_census}) of the innermost loop carrying
   each accumulator update: also a compile-time property, also needing no hardware, and the thing
   that separates "gcc accepted the arm" from "gcc kept it in registers as one vector operation".
   gh-ocannl-621 used it to find the gh-ocannl-614 spill and the scalarization class; gh-ocannl-649
   used it to find that [Max]/[Min] were worse off than either.

   {1 Why the emission settings come from child processes}

   [Cc_backend]'s vector width binds at module initialization and the numerics policy is a global,
   so one process emits under one setting. The driver therefore re-executes ITSELF once per (vector
   width, fp16 policy) pair, and each child writes the kernel it emitted into a directory the driver
   hands it. Three settings are forced on every child besides those two:

   - [OCANNL_BACKEND=cc], because this is a check about the cc backend's C emission and nothing
   else. The stanza therefore pins its backend rather than declaring [OCANNL_BACKEND], and the
   driver itself never creates a context. - [OCANNL_CC_FP16_ARITHMETIC=native], which is what makes
   the native-fp16 rows REAL on a host that merely has [_Float16] with every operation promoted to
   float (x86 without AVX512-FP16 -- see [Cc_backend]'s three-state probe). Forcing the probe's
   answer changes what is EMITTED, which is the whole subject here, and nothing in this check
   executes a kernel, so the host's inability to run 16-bit arithmetic natively costs nothing. Every
   (precision, lane count) row of {!C_syntax.vec_fma_builtin}'s table is reachable this way. -
   [OCANNL_FP16_ARITHMETIC], the one setting that VARIES between children, because it is what
   decides whether fp16 computes in fp16 (the builtin rows) or in f32 (the bridge pair) -- see the
   fixture header below.

   {1 What the claims are, and what they are not}

   Census counts are host-toolchain facts -- they move with the gcc version and the [-march] -- so
   the table goes to stderr and NOT into the golden. What the golden holds are inequalities that a
   misclassified move instruction cannot flip:

   - every kernel compiles clean under every accepted target at [-O2] and at [-O3];
   - every loop is FOUND: a census that answers "no loop carried the anchor" is a failure, because
   scoring only the instructions the good outcome produces is what made a fully scalarized loop read
   as "no FMA loop found" for gh-ocannl-621 -- a pass, arrived at from the failure. For a
   [Tile_mma] row this claim does double duty: its anchor is a name only the register-tiled
   rendering emits, so a silently DECLINED tiling (the gh-ocannl-479 failure mode) reports "no
   loop";
   - no accumulator loop calls [fmax]/[fmin]/[fma]. An opaque call cannot be vectorized at any
   optimization level or grid size, and until gh-ocannl-649 the [Max]/[Min] combine compiled to
   exactly that -- at every [-march] and both optimization levels, because gcc will not contract
   [fmax] into [maxsd] without [-ffinite-math-only], those instructions having the wrong NaN
   behaviour;
   - no accumulator loop is rendered WHOLLY scalar on a target whose ISA has the packed operations
   it needs, and whose rendering set out to be whole-vector in the first place (the two exclusions
   -- fp8's per-lane bridge and the GEBP grid's A splats -- are argued at the claim);
   - no such loop carries scalar FP work AT ALL, which is the same claim sharpened, held over the
   rows whose storage bridge this TOOLCHAIN lowers whole-vector. The two are separate because the
   sharp reading is not purely a fact about the emission: gcc 13.4 lowers the fp16 widening bridge
   one lane at a time on [-march=sapphirerapids] where gcc 15.2 lowers it packed, from identical
   emission, and a golden holding the sharp claim over those rows would be pinning a compiler
   version. Which rows those are is asked of the compiler ({!packed_half_widen}), not assumed;
   - every register-tiled [Tile_mma] k-loop does more vector than scalar work where the ISA has an
   FMA.

   A target the toolchain does not accept is reported with {!Verdict.skipped}, never dropped: a
   silently missing column reads exactly like a passing one. So is a claim whose population a
   toolchain empties.

   {1 The census is a measurement of a COMPILER, and two of them build this repository}

   Every reading above is taken from assembly, so it is a joint fact about the emission and the
   compiler that rendered it, and CI runs two: gcc on ubuntu-latest and clang on macos-latest
   (arm64, where every named x86 [-march] is skipped and the [native] column is the whole matrix).
   Both of the census's inputs turned out to be compiler-dependent in ways a gcc-only box cannot
   see, and both made CI red at gh-ocannl-752 with this box green:

   - {b which line the loop's instructions are attributed to}, which is how a row is FOUND. clang
   attributes an inlined function's instructions to the CALLEE's lines, so an anchor on a line whose
   only content is a call to a same-kernel conversion carries no [.loc] inside the loop at all --
   see {!kernel_loop.anchors}, where the fix is to list a second pattern per tile that no inliner
   can move.
   - {b whether a whole-vector builtin is lowered whole-vector}, which is what the sharp
   scalarization claim reads -- see {!packed_half_widen}.

   The lesson for a new row: anchor it on a line the compiler cannot attribute elsewhere (a macro
   use or a builtin, not a call and not a bare load an optimizer can fold into its use), and before
   claiming a count is zero, ask whether the zero is the emission's or the compiler's. The aarch64 columns need a cross gcc,
   which installs without root -- [apt-get download gcc-<n>-aarch64-linux-gnu cpp-<n>-... binutils-
   aarch64-linux-gnu libc6-dev-arm64-cross linux-libc-dev-arm64-cross libgcc-<n>-dev-arm64-cross]
   then [dpkg-deb -x] each into one prefix. Point [AARCH64_CROSS_GCC] at the resulting
   [aarch64-linux-gnu-gcc-<n>], or leave it unset and the check looks for [aarch64-linux-gnu-gcc] on
   [PATH]. *)

open Base
open Ocannl.Operation.DSL_modules
module LL = Ir.Low_level
module Tn = Ir.Tnode
module Idx = Ir.Indexing
module Cc_backend = Context.Cc_backend
module Census = Test_utils.Asm_census
module Generated = Test_utils.Generated

(* {1 The emission fixture}

   [Vectorized] accumulation loops and register-tiled [Tile_mma] blocks in one kernel, the census
   telling them apart by anchoring on a substring each one alone carries, through the DWARF line
   numbers [-g] leaves in the assembly. One kernel per emission setting rather than one per loop
   keeps the compile matrix small.

   {2 The three groups of loops (gh-ocannl-650, then gh-ocannl-752)}

   - {b uniform-precision reductions}: a [Max], a [Min] and an FMA dot product, at f32, f64 and fp16
   storage, all with storage precision equal to compute precision. These are gh-ocannl-650's
   original six-then-nine, and they are what reaches {!C_syntax.vec_fma_builtin} and
   {!C_syntax.vec_minmax_builtin}: [vec_bridge] is the identity memcpy for all of them.
   - {b narrow-storage reductions}: the same three combines over bf16, fp8 and fp16 storage
   computing in f32 -- the [narrow_compute_f32] path most real kernels take (gh-ocannl-517). These
   are the ones that reach {!C_syntax.vec_bridge}'s widening and narrowing arms and, at the
   accumulator's store, [Ops.convert_precision]'s bf16/fp8/fp16 codecs. Every one of those is
   per-(storage, compute)-pair preprocessor text or a per-pair builtin, i.e. exactly the shape that
   is dead on the build host unless something forces it into a source.
   - {b a register-tiled [Tile_mma]}: the RMxRN GEBP grid ({!C_syntax.try_register_tile}), a
   different emission path from the 1xN accumulator fold with its own register-allocation behaviour
   -- and the one that carries the GFLOP/s. Emitted at f32 storage (the shipped, hardware-measured
   configuration) and at a narrow storage, so that the grid's [vec_bridge] loads and stores are
   compiled too.

   {2 Why two emission settings, and how a child knows which it is}

   fp16 is the one narrow format whose compute precision is a POLICY question rather than a fact
   about the format ({!Ir.Numerics.fp16_mode}), and a policy is per process. Under
   [fp16_arithmetic=true] on a native-16-bit target the fp16 rows compute in fp16 -- which is what
   makes {!C_syntax.vec_fma_builtin}'s AVX512-FP16 and ARMv8.2-FP16 rows real -- and under
   [fp16_arithmetic=false] they compute in f32, which is the (fp16, f32) bridge pair and the CPU
   default. Neither is redundant, so the driver emits both, and each child asks
   {!Ir.Numerics.cpu_compute_prec} -- the resolution [Cc_backend.compute_prec] itself defers to --
   which one it is, rather than being told a second time through the environment.

   {2 Extents}

   The reduction extent is a multiple of [chains * lanes] at every width the ladder can pick (4
   chains x up to 32 lanes = 128, and 4096 = 32 * 128), so the serial tail loop has zero trips and
   constant bounds and gcc deletes it. That is not cosmetic: a surviving tail is a SMALLER innermost
   loop mentioning the same array, and the census -- which picks the smallest span carrying the
   anchor -- would then report the tail's profile as the vector loop's.

   The [Tile_mma] block extents are chosen the same way, against the tiling's own geometry: [m] a
   multiple of [rm = 4], and [n = 192] divisible by every [rn * lanes] the cost model can pick at
   these widths, so neither scalar peel is emitted. The anchors are proof against a peel regardless
   -- each names a vector-register binding ([tmma_as_0__]) that only the full-block k-loop
   contains -- but a peel-free tile is also the shape the census is about. *)

let extent = 4096
let routine = "census_kernel"

(* The [Tile_mma] block: [m] rows, [n] columns, [k] deep. Small (one C-tile pass at every width) --
   the census is a static reading of the innermost k-loop, so a bigger block buys nothing and costs
   compile time in eight columns. *)
let mma_m = 8
let mma_n = 192
let mma_k = 32

(* The compute precision the cc backend resolves a storage precision to under this process's
   numerics policy. Not a second copy of that rule: [Cc_backend.compute_prec] is
   [Numerics.cpu_compute_prec] at the same probe, so a policy change moves both together. *)
let comp_prec p =
  Ir.Numerics.cpu_compute_prec
    ~native_fp16_arithmetic:(Cc_backend.has_native_fp16_arithmetic ())
    p

(** Which emission the loop belongs to. The claims below differ by it, and asking a field is what
    keeps a renamed row from silently changing which claims cover it. *)
type loop_kind =
  | Reduce  (** a [Vectorized] accumulation loop: {!C_syntax.try_vectorize_reduce}'s chain fold *)
  | Tile  (** the k-loop of a register-tiled [Tile_mma]: {!C_syntax.try_register_tile}'s GEBP grid *)

type kernel_loop = {
  anchors : string list;
      (** substrings of the generated source that only this loop's body carries: the source array
          for a reduction, and for a [Tile_mma] both its A-splat binding and its B-row load.

          A LIST rather than one substring because which of a statement's lines the compiler
          attributes the loop's instructions to is a compiler decision, and the two compilers CI
          builds on disagree. Where a line's only content is a call to a function defined in the
          same kernel -- the bf16 tile's [tmma_as_0__ = bfloat16_to_single(...)] -- gcc attributes
          the inlined body to the CALL SITE and clang attributes it to the CALLEE's lines, so on
          clang no [.loc] for the call site appears anywhere in the k-loop and the row reports "no
          loop carried the anchor". That is what made CI's macos-latest leg red across all three
          widths and both optimization levels (gh-ocannl-752), and clang folding a plain load into
          its use does the same to the f32 tile's [tmma_as_0__ = tmma_a__[...]] under some
          [-march]es. The B-row load survives both: it is a macro (or a [__builtin_memcpy]) whose
          instructions no inliner can move to another line. Listing both keeps the anchor working
          where either does, and every pattern of one loop must still name only that loop's lines --
          which is a claim of its own below. *)
  op_class : Census.op_class;
  kind : loop_kind;
  what : string;  (** how the census table names the row *)
  store : string;  (** the precision the loop's operands are STORED at *)
  comp : string;
      (** the precision this loop's ARITHMETIC resolved to, which is what an ISA-scoped claim has to
          ask about -- not the storage precision the row's name leads with. Recorded by the child
          that emitted the loop, from the same {!comp_prec} the emission used. *)
  widen : string;
      (** the [Ops.c_convert_precision] prefix that takes this loop's STORAGE precision to its
          COMPUTE precision, and {!field-narrow} the prefix that takes it back. Empty for a
          uniform-precision loop, which converts nothing. Taken from the compiler's own table by
          the child that emitted the loop rather than spelled here: the codec-coverage claim below
          is about whether the emission REACHED that table's entry, and a second copy of the entry
          would make the claim true of itself. *)
  narrow : string;
}

(* Directory handling through the OCaml stdlib rather than [Sys.command "mkdir -p"] / [Sys.command
   "rm -rf"]: those are POSIX shell spellings, and this test has to work on the native Windows
   environment AGENTS.md supports, where neither exists. Same reasoning as the null device in
   {!Test_utils.Asm_census.accepts}. *)
let rec mkdir_p dir =
  if not (Stdlib.Sys.file_exists dir) then (
    let parent = Stdlib.Filename.dirname dir in
    if not (String.equal parent dir) then mkdir_p parent;
    try Stdlib.Sys.mkdir dir 0o755 with Sys_error _ -> ())

let rec rm_rf path =
  if Stdlib.Sys.file_exists path then
    if Stdlib.Sys.is_directory path then (
      Array.iter (Stdlib.Sys.readdir path) ~f:(fun e -> rm_rf (Stdlib.Filename.concat path e));
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
  let add ~anchors ~op_class ~kind ~what ~store_prec =
    let compute_prec = comp_prec store_prec in
    loops :=
      {
        anchors;
        op_class;
        kind;
        what;
        store = Ir.Ops.prec_string store_prec;
        comp = Ir.Ops.prec_string compute_prec;
        widen = fst (Ir.Ops.c_convert_precision ~from:store_prec ~to_:compute_prec);
        narrow = fst (Ir.Ops.c_convert_precision ~from:compute_prec ~to_:store_prec);
      }
      :: !loops
  in
  (* One (storage precision, row-name tag) group of three reduction loops. [tag] names the row, and
     is also what makes each loop's arrays -- hence its anchor -- unique in the kernel. *)
  let reductions (tag, prec) =
    let da = mk ~prec ~dims:[| extent |] ("dta_" ^ tag) in
    let db = mk ~prec ~dims:[| extent |] ("dtb_" ^ tag) in
    let dacc = mk ~prec ~dims:[| 1 |] ("dtc_" ^ tag) in
    let cell tn = LL.Get (tn, [| Idx.Fixed_idx 0 |]) in
    let at tn s = LL.Get (tn, [| Idx.Iterator s |]) in
    let vloop body =
      let index = Idx.get_symbol () in
      LL.For_loop { index; from_ = 0; to_ = extent - 1; axis = LL.Vectorized; body = body index }
    in
    (* [Max] AND [Min], each at each precision. Emitting only [Max] would leave
       [__builtin_aarch64_fminv4sf]/[fminv2df]/[fminv8hf] in no generated source at all, so the
       whole point of the matrix -- catching a missing builtin, a wrong signature or a broken guard
       in an arm no local hardware selects -- would not reach the [Min] half of
       {!C_syntax.vec_minmax_builtin}. The host runtime test cannot cover them either: it
       preprocesses those arms away. *)
    let minmax op name =
      let src = mk ~prec ~dims:[| extent |] (name ^ "s_" ^ tag) in
      let acc = mk ~prec ~dims:[| 1 |] (name ^ "a_" ^ tag) in
      add ~anchors:[ name ^ "s_" ^ tag ] ~op_class:Census.Max_min ~kind:Reduce
        ~what:(name ^ "/" ^ tag) ~store_prec:prec;
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
    add ~anchors:[ "dta_" ^ tag ] ~op_class:Census.Fma ~kind:Reduce ~what:("dot/" ^ tag)
      ~store_prec:prec;
    let dot_loop =
      vloop (fun j ->
          LL.Set
            {
              tn = dacc;
              idcs = [| Idx.Fixed_idx 0 |];
              llsc = LL.Ternop (Ir.Ops.FMA, (at da j, prec), (at db j, prec), (cell dacc, prec));
              debug = "";
            })
    in
    LL.Seq (max_loop, LL.Seq (min_loop, dot_loop))
  in
  (* One [Tile_mma] block and the scalar micro-kernel it is equivalent to.
     {!Ll_test.optimize_scoped} needs both: [Tile_mma] is minted by schedule transforms AFTER the
     optimization pipeline and [Ir.Low_level.optimize] rejects it outright, so the [raw] twin -- the
     same nodes, reads and writes as a plain [i,j,k] nest -- is what builds the traced store and the
     placements, and the [Tile_mma] form then replaces the schedule wholesale.

     [anchor] is the tile's own A-splat binding, [tmma_as_0__ = <the A element>]. The A element
     carries the storage precision's scalar [convert_precision] spelling, which is what tells two
     tiles in one kernel apart; and the binding name is what keeps a scalar peel, which reads the
     same array through the same conversion into [tmma_acc__], from being the smaller loop the
     census would prefer. *)
  let tile_mma ~tag ~prec ~anchors =
    let d = mk ~prec ~dims:[| mma_m; mma_n |] ("mmad_" ^ tag) in
    let a = mk ~prec ~dims:[| mma_m; mma_k |] ("mmaa_" ^ tag) in
    let b = mk ~prec ~dims:[| mma_k; mma_n |] ("mmab_" ^ tag) in
    let i = Idx.get_symbol () and j = Idx.get_symbol () and k = Idx.get_symbol () in
    let lane = Idx.get_symbol () in
    let it s = Idx.Iterator s in
    let fallback =
      let serial index to_ body = LL.For_loop { index; from_ = 0; to_; axis = LL.Serial; body } in
      serial i (mma_m - 1)
      @@ serial j (mma_n - 1)
      @@ serial k (mma_k - 1)
      @@ LL.Set
           {
             tn = d;
             idcs = [| it i; it j |];
             llsc =
               LL.Ternop
                 ( Ir.Ops.FMA,
                   (LL.Get (a, [| it i; it k |]), prec),
                   (LL.Get (b, [| it k; it j |]), prec),
                   (LL.Get (d, [| it i; it j |]), prec) );
             debug = "";
           }
    in
    let origin = [| Idx.Fixed_idx 0; Idx.Fixed_idx 0 |] in
    add ~anchors ~op_class:Census.Fma ~kind:Tile ~what:("mma/" ^ tag) ~store_prec:prec;
    (* Lane extent 1: the cc backends render the [Workgroup] axis serially and guard the statement
       with [if (lane == 0)], so a wider lane loop would only wrap the tile in a loop that does
       nothing on all but one trip. *)
    let tile =
      LL.For_loop
        {
          index = lane;
          from_ = 0;
          to_ = 0;
          axis = LL.Workgroup;
          body =
            LL.Tile_mma
              {
                d = (d, origin);
                a = (a, origin);
                b = (b, origin);
                ta = false;
                tb = false;
                m = mma_m;
                n = mma_n;
                k = mma_k;
                ldd = mma_n;
                lda = mma_k;
                ldb = mma_n;
                lane;
                fallback;
              };
        }
    in
    (fallback, tile)
  in
  let seq = List.reduce_exn ~f:(fun a b -> LL.Seq (a, b)) in
  (* Which loops this child emits is decided by what its numerics policy resolves fp16 to -- see the
     fixture's header. The f32/f64 rows and the narrow bf16/fp8 ones do not depend on that policy,
     so they are emitted once, by the native-fp16 child, rather than compiled twice in eight
     columns. *)
  let native_fp16 = Ir.Ops.equal_prec (comp_prec Ir.Ops.half) Ir.Ops.half in
  let reduction_groups, tiles =
    if native_fp16 then
      ( [
          ("f32", Ir.Ops.single);
          ("f64", Ir.Ops.double);
          ("f16", Ir.Ops.half);
          ("bf16", Ir.Ops.bfloat16);
          ("fp8", Ir.Ops.fp8);
        ],
        [
          (* Identity bridges, f32 arithmetic: the shipped configuration. *)
          ( "f32",
            Ir.Ops.single,
            [ "float tmma_as_0__ = tmma_a__"; "__builtin_memcpy(&tmma_b_0__, &tmma_b__[" ] );
          (* The widening bridge inside the grid: the A splat converts per element, the B rows and
             the C tile cross through [vec_bridge]'s bf16 arms. *)
          ( "bf16",
            Ir.Ops.bfloat16,
            [ "tmma_as_0__ = bfloat16_to_single("; ", tmma_b_0__, &tmma_b__[" ] );
        ] )
    else
      ( [ ("f16w", Ir.Ops.half) ],
        [
          ( "f16w",
            Ir.Ops.half,
            [ "tmma_as_0__ = HALF_TO_FLOAT("; ", tmma_b_0__, &tmma_b__[" ] );
        ] )
  in
  let raw_tiles, scoped_tiles =
    List.map tiles ~f:(fun (tag, prec, anchors) -> tile_mma ~tag ~prec ~anchors) |> List.unzip
  in
  let reduce_llc = seq (List.map reduction_groups ~f:reductions) in
  let raw = seq (reduce_llc :: raw_tiles) in
  let scoped = seq (reduce_llc :: scoped_tiles) in
  (* [optimize_scoped] injects the nest AS WRITTEN past [LL.optimize], which is what keeps the
     [Vectorized] annotations, the [Tile_mma] statements and the exact loop shapes -- the subject of
     the census -- from being rewritten on the way to the backend. *)
  let o = Ll_test.optimize_scoped ~materialized:!nodes ~name:routine ~raw scoped in
  ignore (Ll_test.link ~name:routine o : Context.t * Context.routine);
  Stdio.Out_channel.write_all
    (Stdlib.Filename.concat emit_dir (routine ^ ".c"))
    ~data:(Generated.read routine);
  Stdio.Out_channel.write_all
    (Stdlib.Filename.concat emit_dir (routine ^ ".loops"))
    ~data:
      (String.concat ~sep:"\n"
         (List.rev_map !loops ~f:(fun { anchors; op_class; kind; what; store; comp; widen; narrow } ->
              Printf.sprintf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s"
                (String.concat ~sep:"|" anchors)
                (match op_class with Census.Fma -> "fma" | Census.Max_min -> "maxmin")
                (match kind with Reduce -> "reduce" | Tile -> "tile")
                what store comp widen narrow)))

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
    (* The host's own default target, with no [-march] at all: the one column every toolchain
       accepts, and therefore the one that keeps the matrix from being VACUOUS. Without it, a run on
       a host whose compiler accepts none of the named targets -- Apple clang on Apple Silicon
       rejects every x86 [-march] here, which is what CI's macos-latest is -- produces no rows, and
       the aggregate claims fail on emptiness having tested nothing. That failure is honest (the
       non-emptiness guards did their job) but useless: the fix is to give that host a column it can
       actually compile, not to let the claims pass empty. *)
    t "native" host "" "the host's own default target: the column that cannot be skipped";
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

(* The two numerics settings a child can emit under, named by what they resolve fp16 to. Only
   [fp16_arithmetic] differs: [native] keeps fp16 arithmetic 16-bit (so
   {!C_syntax.vec_fma_builtin}'s AVX512-FP16 and ARMv8.2-FP16 rows are real), [wide] computes fp16
   in f32 (the CPU default, and the (fp16, f32) pair of {!C_syntax.vec_bridge}). The child does not
   read this tag: it asks its own resolved compute precision which side it is on. *)
let fp16_settings = [ ("native", "true"); ("wide", "false") ]

(* The environment a child emits under: this process's, with every setting the child must not
   inherit replaced. Removing them first matters -- [Unix.create_process_env] passes the array
   verbatim, and a duplicate key resolves to whichever copy the child's getenv finds first. *)
let child_env ~width ~fp16 ~dir =
  let forced =
    [
      ("OCANNL_BACKEND", "cc");
      ("OCANNL_CC_VECTOR_BYTES", Int.to_string width);
      ("OCANNL_CC_FP16_ARITHMETIC", "native");
      ("OCANNL_FP16_ARITHMETIC", fp16);
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

let spawn_emit ~exe ~width ~fp16 ~dir =
  mkdir_p dir;
  (* The child's stdout is captured, not inherited: it must not reach the golden, and a failed child
     is worth reporting in full. *)
  let log = Stdlib.Filename.concat dir "emit.log" in
  let out = Unix.openfile log [ Unix.O_WRONLY; Unix.O_CREAT; Unix.O_TRUNC ] 0o644 in
  let pid =
    Unix.create_process_env exe [| exe |] (child_env ~width ~fp16 ~dir) Unix.stdin out Unix.stderr
  in
  let _, status = Unix.waitpid [] pid in
  Unix.close out;
  match status with
  | Unix.WEXITED 0 -> Ok ()
  | _ -> Error (try Stdio.In_channel.read_all log with _ -> "(no output)")

type caps = {
  vector_bytes : int;  (** the widest vector register the target has *)
  has_fma : bool;
  fp16_vector : bool;  (** 16-bit vector ARITHMETIC *)
  fp16_convert : bool;
      (** packed fp16 <-> f32 CONVERSION, which is a different ISA question from arithmetic and the
          one a narrow-fp16-storage loop asks: [OCANNL_VEC_WIDEN_HALF]'s
          [__builtin_convertvector] arm needs an instruction to convert with, and where there is
          none gcc widens lane by lane. x86 gets it with F16C (from [x86-64-v3]), aarch64 has it at
          the armv8-a baseline. *)
  named : bool;  (** a column whose [-march] this test chose, as opposed to the host's default *)
}

(* Read off the compiler's own predefined macros rather than pattern-matched from the label: what
   [-march=x86-64-v3] implies is gcc's fact to state, and the [native] column has no label to match
   in the first place. *)
let caps_of t =
  let d = Census.defines t in
  let has m = Set.mem d m in
  {
    vector_bytes =
      (if has "__AVX512F__" then 64 else if has "__AVX2__" || has "__AVX__" then 32 else 16);
    has_fma = has "__FMA__" || has "__ARM_FEATURE_FMA";
    fp16_vector = has "__AVX512FP16__" || has "__ARM_FEATURE_FP16_VECTOR_ARITHMETIC";
    fp16_convert = has "__F16C__" || has "__AVX512FP16__" || has "__ARM_FP16_FORMAT_IEEE";
    named = not (String.is_empty t.Census.march);
  }

(* Whether this target's ISA has the operations the loop wants, so that a claim about how well gcc
   rendered it means anything. Asked of the loop's own recorded facts, not of its row NAME: the
   arithmetic precision is the emission's decision (a narrow-storage loop computes in f32 and needs
   nothing 16-bit of the ISA), and whether a loop is an FMA one is the [op_class] the census is
   already told. Both used to be read back out of the [what] string, which meant a new row name
   silently changed which claims covered it. *)
let half = Ir.Ops.prec_string Ir.Ops.half

let isa_has ~caps ~(loop : kernel_loop) ~width =
  if width > caps.vector_bytes then false
    (* 16-bit ARITHMETIC, which only a loop that computes in fp16 needs. *)
  else if String.equal loop.comp half && not caps.fp16_vector then false
    (* 16-bit CONVERSION, which a loop that STORES fp16 and computes in f32 needs on both sides of
       its bridge -- and which gcc has no vector form for without F16C, widening each lane through
       a scalar [vcvtsh2ss] instead (visible as ~20 scalar FP ops in an otherwise packed loop at
       [x86-64] and [x86-64-v2]). *)
  else if String.equal loop.store half && (not (String.equal loop.comp half))
          && not caps.fp16_convert
  then false
  else match loop.op_class with Census.Fma -> caps.has_fma | Census.Max_min -> true

(* {2 What this TOOLCHAIN can express, as opposed to what the ISA HAS}

   [caps] above is read from the compiler's predefined macros and so answers what the TARGET has:
   [__AVX512FP16__] means the ISA converts fp16 to float a whole vector at a time. Whether the
   compiler USES that instruction for the bridge the emission writes is a second question, and the
   two gcc versions CI and this box run answer it differently: on [-march=sapphirerapids], where
   [_Float16] is a native arithmetic type, gcc 13.4 lowers [__builtin_convertvector] over a
   [_Float16] vector one lane at a time ([vcvtsh2ss] per lane, measured 4/8/16 of them at the three
   widths) while gcc 15.2 emits one [vcvtph2psx]. Both compile, both keep the ARITHMETIC packed, and
   the emission is the same whole-vector [__builtin_convertvector] either way -- so a claim that
   the loop carries no scalar FP at all is, on those rows, a claim about the compiler's version.
   That is what turned CI's ubuntu-latest leg red on 18 rows this box reported clean
   (gh-ocannl-752).

   So the toolchain is ASKED, the way it is asked which [-march]es it accepts: compile the arm
   {!C_syntax.vec_bridge} emits for (fp16 storage, f32 compute) -- a memcpy into a [_Float16]
   vector and a [__builtin_convertvector] out of it, which is [OCANNL_VEC_WIDEN_HALF]'s body -- and
   census the result. Any scalar FP instruction in a function whose whole content is one
   whole-vector conversion means this compiler lowers that conversion per lane. A version test would
   be a second copy of gcc's changelog; this is the property itself. *)

let half_widen_probe_source ~lanes =
  Printf.sprintf
    {|typedef _Float16 ocannl_probe_h __attribute__((vector_size(%d)));
typedef float ocannl_probe_f __attribute__((vector_size(%d)));
void ocannl_probe_widen(ocannl_probe_f *dst, const ocannl_probe_h *src) {
  ocannl_probe_h nh;
  __builtin_memcpy(&nh, src, sizeof(nh));
  *dst = __builtin_convertvector(nh, ocannl_probe_f);
}
|}
    (lanes * 2) (lanes * 4)

let half_widen_cache : (string * int, bool) Hashtbl.t = Hashtbl.Poly.create ()

(* [packed_half_widen t ~width] is whether [t] lowers the emitted fp16 widening bridge to
   whole-vector instructions at the lane count [width] implies. A toolchain that does not COMPILE
   the probe answers [false]: it cannot express the bridge whole-vector either, and the rows that
   would be excluded are reported. Memoized per (column, width): this is asked once per row and a
   run has hundreds. *)
let packed_half_widen t ~width =
  Hashtbl.find_or_add half_widen_cache (t.Census.label, width) ~default:(fun () ->
      let src = Stdlib.Filename.temp_file "ocannl_census_widen_" ".c" in
      let asm = Stdlib.Filename.temp_file "ocannl_census_widen_" ".s" in
      Stdio.Out_channel.write_all src ~data:(half_widen_probe_source ~lanes:(width / 4));
      let verdict =
        match Census.compile t ~opt_level:2 ~src_path:src ~asm_path:asm with
        | Error _ -> false
        | Ok () ->
            let p = Census.profile_all Census.Fma ~asm:(Stdio.In_channel.read_all asm) in
            p.Census.scalar_fp_ops = 0
      in
      (try Stdlib.Sys.remove src with _ -> ());
      (try Stdlib.Sys.remove asm with _ -> ());
      verdict)

type emitted = {
  width : int;
  fp16 : string;  (** the [fp16_arithmetic] setting this kernel was emitted under *)
  src_path : string;
  source : string;
  loops : kernel_loop list;
}

type row = {
  toolchain : string;
  caps : caps;  (** this column's ISA facts, read from its own compiler *)
  width : int;
  opt : int;
  loop : kernel_loop;
  bridge_packed : bool;
      (** whether this column's TOOLCHAIN lowers this loop's storage bridge whole-vector -- see
          {!packed_half_widen}. [true] for a loop that bridges nothing, and for a narrow bridge
          whose codec is integer shifts (bf16) rather than a conversion instruction. *)
  profile : Census.t option;  (** [None]: no loop carried the anchor *)
}

let is_fma_row r = match r.loop.op_class with Census.Fma -> true | Census.Max_min -> false

let emit_all ~exe ~root =
  List.cartesian_product widths fp16_settings
  |> List.filter_map ~f:(fun (width, (fp16_tag, fp16)) ->
         let describe = Printf.sprintf "width %d, fp16 %s" width fp16_tag in
         let dir = Stdlib.Filename.concat root (Printf.sprintf "w%d-%s" width fp16_tag) in
         match spawn_emit ~exe ~width ~fp16 ~dir with
         | Error out ->
             Verdict.fail (Printf.sprintf "%s: the emit child failed:\n%s" describe out);
             None
         | Ok () ->
             let src_path = Stdlib.Filename.concat dir (routine ^ ".c") in
             let loops_path = Stdlib.Filename.concat dir (routine ^ ".loops") in
             if not (Stdlib.Sys.file_exists src_path && Stdlib.Sys.file_exists loops_path) then (
               Verdict.fail
                 (Printf.sprintf "%s: the emit child produced no kernel in %s" describe dir);
               None)
             else
               let loops =
                 Stdio.In_channel.read_lines loops_path
                 |> List.filter_map ~f:(fun l ->
                     match String.split l ~on:'\t' with
                     | [ anchors; cls; kind; what; store; comp; widen; narrow ] ->
                         Some
                           {
                             anchors = String.split anchors ~on:'|';
                             op_class =
                               (if String.equal cls "fma" then Census.Fma else Census.Max_min);
                             kind = (if String.equal kind "tile" then Tile else Reduce);
                             what;
                             store;
                             comp;
                             widen;
                             narrow;
                           }
                     | _ -> None)
               in
               Some
                 {
                   width;
                   fp16 = fp16_tag;
                   src_path;
                   source = Stdio.In_channel.read_all src_path;
                   loops;
                 })

(* {2 What each target's ISA can be held to}

   A claim about an accumulator loop is only meaningful where the ISA has the operation the loop
   wants, so each row is asked of its own target:

   - the [Max]/[Min] combine is a packed compare, a packed bitwise select and nothing else, which
   every x86 target from SSE2 up and every aarch64 target has; - the FMA update needs a fused
   multiply-add INSTRUCTION, which on x86 arrives with [x86-64-v3] and on aarch64 is baseline NEON
   ([fmla]). Below that the emission falls back to per-lane [fmaf] -- a libm call, and deliberately
   so: an FMA rounds once, so on a machine without the instruction the library call is what
   preserves the rounding the scalar path has, and substituting [a * b + c] would make the
   vectorized and serial paths disagree. That fallback is a cost of correctness, not a defect, so
   the libm claim excludes it and says so; - 16-bit arithmetic is the exception under both headings:
   it needs AVX512-FP16 or ARMv8.2-FP16, and elsewhere gcc widens every lane to float, which is
   scalar work by construction; - and the requested WIDTH has to fit the target's registers. A 32-
   or 64-byte GNU C vector on an SSE2-only target is emulated -- gcc splits it into 16-byte pieces
   and spills the rest -- which is why the same [Max] loop that is 49 instructions with 42 vector
   operations and no stack traffic at [cc_vector_bytes=16] is 677 instructions with 128 scalar ones
   and 238 stack references at 32. That is the configuration being wrong for the machine, not the
   emission being wrong: [cc_vector_bytes] is auto-probed from the host when unset, and the widths
   here are forced precisely so that every arm of the builtin table gets compiled. *)

(* {2 Assembler dialects this host cannot produce}

   Two review findings in a row were about assembly shapes a Linux/gcc box never emits, each of
   which silenced the census WHOLESALE rather than skewing a number -- Apple's dot-less [LBB0_9]
   labels, which made every branch target unrecognizable, and clang's [.file 0 "dir" "name" md5
   0x...], whose checksum reads as the path and leaves no file number matched. Both fail the same
   way: every row reports "no loop", on a platform CI actually builds.

   Neither is reachable from a fixture compiled here, so they are pinned against SYNTHETIC assembly
   instead. That is not a weaker check for this particular property -- what is being tested is the
   parser's handling of a documented directive shape, and the shape is what is written down. *)

let dialect_probes () =
  (* Line 3 carries the anchor; the [.loc] directives below point at it. *)
  let source = "int f(void) {\n  /* prologue */\n  ACC = MAXOF(ACC, SRC);\n  return 0;\n}\n" in
  let anchor = Census.anchor_lines ~source ~patterns:[ "MAXOF" ] in
  let elf =
    String.concat ~sep:"\n"
      [
        "\t.file\t\"census_kernel.c\"";
        "\t.file 1 \"/build/census_kernel.c\"";
        "f:";
        ".L2:";
        "\t.loc 1 3 12";
        "\taddps\t%xmm1, %xmm0";
        "\tjne\t.L2";
        "\tret";
      ]
  in
  (* Every Mach-O detail that has silenced this census, in ONE fixture, because they were found one
     round at a time: dot-less [LBB] labels, a checksummed [.file], the [;] loop-header annotation
     Apple clang writes after a label, and the underscore Mach-O puts on every C symbol. The first
     version of this probe used a BARE [LBB0_1:] and so did not exercise the annotation -- which is
     how the [;] gap survived the round that added the probe. A fixture for a dialect should carry
     the dialect's noise, not just its identifiers. *)
  let darwin_clang =
    String.concat ~sep:"\n"
      [
        "\t.file\t0 \"/build\" \"census_kernel.c\" md5 0x0123456789abcdef0123456789abcdef";
        "\t.file\t1 \"/build\" \"census_kernel.c\" md5 0x0123456789abcdef0123456789abcdef";
        "_f:                                     ; @f";
        "LBB0_1:                                 ; =>This Inner Loop Header: Depth=1";
        "\t.loc\t1 3 12";
        "\tfmax.4s\tv0, v0, v1";
        "\tb.ne\tLBB0_1";
        "\tret";
      ]
  in
  (* The same loop with a libm call in it, which the census must SEE. Mach-O spells it [_fmaxf];
     matching [libm_names] without stripping that underscore reports zero calls and passes the
     central gh-ocannl-649 claim over a loop that is exactly the regression. *)
  let darwin_with_libm =
    String.concat ~sep:"\n"
      [
        "\t.file\t1 \"/build\" \"census_kernel.c\" md5 0x0123456789abcdef0123456789abcdef";
        "_f:";
        "LBB0_1:                                 ; =>This Inner Loop Header: Depth=1";
        "\t.loc\t1 3 12";
        "\tcallq\t_fmaxf";
        "\tjne\tLBB0_1";
        "\tretq";
      ]
  in
  let censused asm = Census.census Census.Max_min ~asm ~source_basename:(routine ^ ".c") ~anchor in
  Verdict.p_all "the census reads both assembler dialects, not only this host's"
    [ ("gnu/elf", elf); ("apple-clang", darwin_clang); ("apple-clang+libm", darwin_with_libm) ]
    ~f:(fun (_, asm) -> Census.loop_edges ~asm > 0 && Option.is_some (censused asm));
  Verdict.p "a libm call spelled the Mach-O way is still counted as one"
    (match censused darwin_with_libm with
    | Some c -> c.Census.counts.Census.libm_calls > 0
    | None -> false)

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
      (* The (fp16, f32) bridge pair and the native-fp16 builtin rows live in different children, so
         a child that silently emitted the other one's loop set would leave one of the two groups in
         no source at all -- and every claim below would still pass, over the surviving group.
 *)
      Verdict.p_all "every fp16 numerics setting emitted a kernel" fp16_settings
        ~f:(fun (tag, _) -> List.exists emitted ~f:(fun e -> String.equal e.fp16 tag));
      (* {2 That the narrow-storage rows are really narrow}

         Every claim below this point is about how well gcc rendered a loop, and a loop that stopped
         being a NARROW-storage one -- a numerics default flipped, a [vec_bridge] arm lost, a
         storage precision the emission silently computed at -- passes all of them. It is still a
         [Vectorized] fold over an array called [maxs_bf16]; it is just no longer the thing
         gh-ocannl-752 added it for, and nothing in a golden of inequalities would say so. So the
         COVERAGE is claimed too, from the emitted source rather than from the row names:

         - the whole-vector bridges, which are the discriminating half. [OCANNL_VEC_WIDEN_BFLOAT16]
         and its siblings are emitted by {!C_syntax.vec_bridge}'s narrow arms and by nothing else,
         and {!Cc_backend} prepends a builtin only where the kernel calls it (or something it calls
         does), so a kernel carrying the name is a kernel that bridged. The set comes from
         {!Context.Builtins_cc.builtins}: a bridge added for a fourth format fails this claim until
         the fixture emits a loop that reaches it.
         - the scalar codecs, which is where fp8 lives. {!C_syntax.vec_bridge} has no vector arm for
         it and converts per lane through [Ops.c_convert_precision] instead, so no [OCANNL_VEC_*]
         name attests to an fp8 bridge and the codec's own spelling is what does. Asked of every
         loop whose storage and compute precisions differ, in BOTH directions -- the load's widening
         and the accumulator store's narrowing are separate entries of that table, and a rendering
         that reached only one of them is a rendering this fixture was meant to catch. *)
      let vec_bridges =
        List.filter_map Context.Builtins_cc.builtins ~f:(fun (key, _, _) ->
            if
              String.is_prefix key ~prefix:"OCANNL_VEC_WIDEN_"
              || String.is_prefix key ~prefix:"OCANNL_VEC_NARROW_"
            then Some key
            else None)
      in
      Verdict.p_all "every whole-vector storage bridge the cc builtins define reaches a kernel"
        vec_bridges ~f:(fun key ->
            List.exists emitted ~f:(fun e -> String.is_substring e.source ~substring:key));
      let narrow_rows =
        List.concat_map emitted ~f:(fun e ->
            List.filter_map e.loops ~f:(fun l ->
                if String.equal l.store l.comp then None else Some (e, l)))
      in
      Verdict.p_all
        "every narrow-storage loop's kernel spells both directions of its conversion" narrow_rows
        ~f:(fun (e, l) ->
            (not (String.is_empty l.widen))
            && (not (String.is_empty l.narrow))
            && String.is_substring e.source ~substring:l.widen
            && String.is_substring e.source ~substring:l.narrow);
      (* {2 That each row's anchor names its own loop and no other}

         The census selects the SMALLEST-span loop carrying an anchor line, so two loops of one
         kernel whose anchors overlap do not report a conflict -- one of them silently answers for
         the other, at whichever profile is smaller. That is invisible in every claim below: the row
         is found, it compiles, it counts something. Only this says the anchors still partition the
         kernel's loops, which is what lets a claim be attributed to the loop it names. It matters
         most for the [Tile_mma] rows, whose anchors are now two patterns each rather than one, and
         for the narrow reduction groups, whose tags are substrings of one another's ([f16] of
         [f16w]) whenever a future setting emits both from the same child. *)
      Verdict.p_all "no two censused loops of a kernel share an anchor line"
        (List.concat_map emitted ~f:(fun e ->
             List.map e.loops ~f:(fun l ->
                 (e, l, Census.anchor_lines ~source:e.source ~patterns:l.anchors))))
        ~f:(fun (e, l, mine) ->
          (not (Set.is_empty mine))
          && List.for_all e.loops ~f:(fun other ->
                 String.equal other.what l.what
                 || Set.is_empty
                      (Set.inter mine
                         (Census.anchor_lines ~source:e.source ~patterns:other.anchors))));
      let all = toolchains () in
      let available = List.filter all ~f:Census.accepts in
      Stdio.eprintf "\n=== cc kernel census (not part of the golden) ===\n";
      Stdio.eprintf "host toolchain: %s\n" (Cc_backend.compiler_command ());
      let failed_compiles = ref [] in
      let edges = ref [] in
      let rows =
        List.concat_map available ~f:(fun t ->
            (* Once per toolchain, not once per row: [caps_of] launches a compiler process, and this
               sits above the map over every loop x kernel x optimization level, so computing it
               inside cost hundreds of redundant `cc -dM -E` invocations per run. *)
            let caps = caps_of t in
            List.concat_map emitted ~f:(fun e ->
                List.concat_map [ 2; 3 ] ~f:(fun opt ->
                    let kernel = Printf.sprintf "w%d-%s" e.width e.fp16 in
                    let asm_path =
                      Stdlib.Filename.concat root
                        (Printf.sprintf "%s-%s-O%d.s"
                           (String.map t.Census.label ~f:(fun c ->
                                if Char.is_alphanum c then c else '_'))
                           kernel opt)
                    in
                    match Census.compile t ~opt_level:opt ~src_path:e.src_path ~asm_path with
                    | Error out ->
                        failed_compiles :=
                          Printf.sprintf "%s -O%d %s" t.Census.label opt kernel :: !failed_compiles;
                        Verdict.fail
                          (Printf.sprintf "%s -O%d, %s: the emitted kernel does not compile:\n%s"
                             t.Census.label opt kernel out);
                        []
                    | Ok () ->
                        (* Parsed once per compiled file, then asked about each of its loops: the
                           line classification, the loop edges and the DWARF file numbers are
                           properties of the listing, and a listing runs to tens of thousands of
                           lines. *)
                        let parsed =
                          Census.parse
                            ~asm:(Stdio.In_channel.read_all asm_path)
                            ~source_basename:(routine ^ ".c")
                        in
                        (* Recorded before any anchor is looked for: an ISA whose branch spelling
                           {!Census.is_branch} does not know reports every anchor missing at once,
                           which is a defect in the census and not a finding about the kernel. *)
                        edges :=
                          ( Printf.sprintf "%s %s -O%d" t.Census.label kernel opt,
                            Census.edge_count parsed )
                          :: !edges;
                        List.map e.loops ~f:(fun loop ->
                            let c =
                              Census.census_in parsed loop.op_class
                                ~anchor:
                                  (Census.anchor_lines ~source:e.source ~patterns:loop.anchors)
                            in
                            (match c with
                            | Some c ->
                                Stdio.eprintf "  %-24s %-11s -O%d %-11s %s\n" t.Census.label kernel
                                  opt loop.what (Census.to_line c)
                            | None ->
                                Stdio.eprintf
                                  "  %-24s %-11s -O%d %-11s NO LOOP CARRIED THE ANCHOR\n"
                                  t.Census.label kernel opt loop.what);
                            {
                              toolchain = t.Census.label;
                              caps;
                              width = e.width;
                              opt;
                              loop;
                              bridge_packed =
                                (if
                                   String.equal loop.store half
                                   && not (String.equal loop.comp half)
                                 then packed_half_widen t ~width:e.width
                                 else true);
                              profile = c;
                            }))))
      in
      Stdio.eprintf "=== end census ===\n\n";
      let describe r = Printf.sprintf "%s w%d -O%d %s" r.toolchain r.width r.opt r.loop.what in
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
              ((not (List.is_empty mine))
              && List.for_all mine ~f:(fun r -> Option.is_some r.profile))
          else Verdict.skipped ~backend:(t.Census.label ^ ", " ^ t.Census.note) name);
      Verdict.p_empty
        "every emitted kernel compiles clean at -O2 and -O3 under every accepted -march" ~over:rows
        !failed_compiles;
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
      let claim_all label population ~f =
        Verdict.p_all label population ~f;
        report label population ~f:(fun r -> not (f r))
      in
      (* The scalarization claim is held only over columns whose [-march] this test named -- it is a
         claim about how gcc vectorizes a known target, and the [native] column is whatever compiler
         the run landed on (Apple clang on macOS CI), whose vectorizer this test does not model. The
         other two claims DO cover it: "compiles clean" and "no libm call in a Max/Min loop" hold by
         construction of the emission, whatever the host compiler does with it, which is what keeps
         a macOS run from verifying nothing. *)
      let served =
        List.filter rows ~f:(fun r -> isa_has ~caps:r.caps ~loop:r.loop ~width:r.width)
      in
      let served_named = List.filter served ~f:(fun r -> r.caps.named) in
      (* The gh-ocannl-649 claim proper. Unconditional across the matrix, unlike the two below:
         [Max]/[Min] have a whole-vector spelling on every target here at every width -- an emulated
         wide vector still compiles to packed compares and selects, just more of them -- so a libm
         call in one of these loops is a defect wherever it appears. *)
      claim_none "no Max/Min accumulator loop calls libm"
        (List.filter rows ~f:(fun r -> not (is_fma_row r)))
        ~f:libm;
      claim_none "no FMA accumulator loop calls libm where the ISA has a fused multiply-add"
        (List.filter served ~f:is_fma_row)
        ~f:libm;
      let scalarization_claim =
        "no accumulator loop is scalarized where a named target's ISA has the operation"
      in
      (* {b Which rows the scalarization claim can be held over.} "Scalarized" means [scalar_fp_ops
         > 0 || vector_ops = 0], and that reading is only a defect where the emission set out to
         render the whole loop as vector operations. Two groups of rows it did not, both by
         construction and both documented at their emission site, so a row of theirs is excluded
         here rather than left to fail:

         - {b the [Tile_mma] k-loop}. A GEBP grid's A feeds are scalar element loads by design --
         [rm] of them per k step, each splat across a vector -- so [scalar_fp_ops] is [rm] in a
         perfectly rendered tile (measured: exactly 4 at every x86 column with FMA). What pins
         these rows instead is the anchor itself: [tmma_as_0__] is a name only
         {!C_syntax.try_register_tile}'s full-block body emits, so a tiling that DECLINED and
         rendered the lane-0 scalar fallback carries it nowhere and the row reports "no loop
         carried the anchor" -- which "every accumulator loop is found by the census" above already
         treats as a failure. Declines being silent is exactly the gh-ocannl-479 failure mode, and
         this is the census's version of the check. The vector-majority claim below is what the
         rows say positively.
         - {b fp8 operands}. {!C_syntax.vec_bridge} has no vector codec for fp8 and says so: it
         converts one lane at a time through the scalar path's own [convert_precision], with the
         arithmetic around it staying vectorized. gcc keeps that fixed-trip loop AS a loop (a GNU
         vector's lane written through a variable index has to go through memory), so it is the
         innermost loop carrying an fp8 operand's source line and it is what these rows census: 8
         to 11 instructions, near zero vector operations, at every column. That is the emitted
         shape, not a regression -- what the fp8 rows pin is that the codec compiles under every
         [-march] and calls no libm, not how the arithmetic around it vectorized. *)
      let whole_vector_rendering r =
        match r.loop.kind with
        | Tile -> false
        | Reduce -> not (String.equal r.loop.store (Ir.Ops.prec_string Ir.Ops.fp8))
      in
      (* On a host whose compiler accepts none of the named [-march] targets -- Apple clang on Apple
         Silicon, which is CI's macos-latest -- only the [native] column survives, and this claim's
         population is empty. Reported as SKIPPED rather than left to fail on emptiness: the claim
         genuinely was not evaluated there, and [Verdict.skipped] is the channel that says so while
         keeping the golden one line per claim on every host. The two claims above still cover that
         run. *)
      let whole_vector_population = List.filter served_named ~f:whole_vector_rendering in
      (* {b The half of the reading that is not a compiler-version fact.} "Wholly scalar" --
         [vector_ops = 0] -- is the gh-ocannl-621 class: an arm gcc accepted and then rendered
         element by element, which is the outcome the census exists to catch and which no gcc since
         has disagreed about on any row here. It is held over EVERY row of the population, including
         the ones the strict claim below has to let go of, so that dropping a row from that claim
         never leaves it unclaimed. *)
      let wholly_scalar r = match counts r with Some c -> c.Census.vector_ops = 0 | None -> false in
      let wholly_scalar_claim =
        "no accumulator loop is rendered wholly scalar where a named target's ISA has the operation"
      in
      if List.is_empty whole_vector_population then
        Verdict.skipped ~backend:"no named -march target accepted by this host's toolchain"
          wholly_scalar_claim
      else claim_none wholly_scalar_claim whole_vector_population ~f:wholly_scalar;
      (* {b And the half that is.} [scalar_fp_ops = 0] additionally asks that the loop's storage
         BRIDGE compiled whole-vector, and for (fp16 storage, f32 compute) that is the compiler's
         choice rather than the target's -- gcc 13.4 lowers the emitted [__builtin_convertvector]
         per lane on [-march=sapphirerapids] where gcc 15.2 lowers it packed, over identical
         emission (see {!packed_half_widen}). Those rows are excluded HERE, on the toolchain's own
         answer to a probe of that one conversion, rather than the claim being lowered for every row
         to what the older compiler can do: on a toolchain that lowers the bridge packed -- which is
         every column of this box's gcc 15.2, and every non-fp16 column of gcc 13.4 -- the strict
         claim still covers them. The excluded rows are named on stderr; they keep the wholly-scalar
         claim above, and their bridge is still pinned by the codec-coverage claims, which are about
         the source and so are compiler-independent. *)
      let scalarization_population =
        List.filter whole_vector_population ~f:(fun r -> r.bridge_packed)
      in
      let bridge_excluded = List.filter whole_vector_population ~f:(fun r -> not r.bridge_packed) in
      if not (List.is_empty bridge_excluded) then (
        Stdio.eprintf
          "  %d row(s) outside %S: this toolchain lowers their fp16 widening bridge per lane (not \
           part of the golden):\n"
          (List.length bridge_excluded) scalarization_claim;
        List.iter bridge_excluded ~f:(fun r -> Stdio.eprintf "    %s\n" (describe r)));
      if List.is_empty scalarization_population then
        Verdict.skipped
          ~backend:
            "no named -march target whose toolchain lowers these loops' storage bridges \
             whole-vector"
          scalarization_claim
      else claim_none scalarization_claim scalarization_population ~f:scalarized;
      (* The [Tile_mma] rows' own claim, in the shape gh-ocannl-614 measured the GEBP kernel in: the
         k-loop's work is vector work. An inequality between the two counts rather than a threshold
         on either, so a misclassified move cannot flip it, and it is two-sided in the way that
         matters -- it fails both when the vector body collapses to per-lane arithmetic and when the
         bridges stop being whole-vector loads (a per-lane [vec_bridge] arm inside the k-loop would
         put [lanes] scalar conversions against the same [rm * rn] fused updates). Held where the
         ISA has a fused multiply-add, since without one the tile's updates are [fmaf] calls per
         lane and the row is about that instead (the "calls libm" claim above covers it there). *)
      let tile_rows =
        List.filter served ~f:(fun r -> match r.loop.kind with Tile -> true | Reduce -> false)
      in
      let tile_claim =
        "every register-tiled Tile_mma k-loop does more vector than scalar work where the ISA has \
         a fused multiply-add"
      in
      if List.is_empty tile_rows then
        Verdict.skipped ~backend:"no accepted target serves the Tile_mma rows" tile_claim
      else
        claim_all tile_claim tile_rows ~f:(fun r ->
            match counts r with
            | Some c -> c.Census.vector_ops > c.Census.scalar_fp_ops
            | None -> false)
