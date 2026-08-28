(* Standalone Metal check for the LOCALIZED spelling of the pooled-accumulation miscompile
   (gh-ocannl-731, follow-ups gh-ocannl-782), and the measurement of what the workaround costs.

   [bench_metal_bug.ml] beside this file reproduces the original manifestation: a device-memory
   read-modify-write [acc[0] = acc[0] + f(i)] over slot-table-derived pool pointers keeping only the
   last iteration. After the serial-reduction localizer (gh-ocannl-693) that statement shape is gone
   from most kernels — the accumulator lives in a scope local and the node is stored once — and the
   same compiler pass can corrupt THAT form instead. OCANNL works around it by declaring every
   reduction-shaped scope local [volatile] on Metal ([volatile_serial_accumulation] in
   arrayjit/lib/c_syntax.ml).

   This program has two halves, and no OCANNL dependency at all:

   - the REPRODUCER MATRIX: the emitted kernel of [test/operations/scalar_rmw_accumulation.ml]'s
     localized leg, verbatim, rendered from one template under the variations that isolate what the
     miscompiling pass keys on, each checked against the host oracle. That is the report an upstream
     bug filing needs, and it is what says which shapes a narrowed predicate would still have to
     cover.
   - the TAX: three localized-reduction kernel shapes (a streaming per-thread reduction, an
     accumulator-bound dependency chain, one long single-thread reduction — the scalar-loss shape),
     each rendered with and without the qualifier and timed on the GPU's own clock, interleaved,
     best of N repeats whose arm order rotates. That is what says whether the wide predicate is
     worth narrowing.

   Run it as [dune exec benchmarks/runners/ocannl/bench_metal_bug_local.exe]; it links [metal] and
   [ctypes] only, so it keeps working across OCANNL refactorings and can be handed to Apple. *)

module Me = Metal

(* The pooled binding preamble, verbatim as the Metal backend emits it (arrayjit/lib/metal_backend.ml
   — 16 slab pointers plus the dynamically-loaded [__pool_slots] offset table). It is kept
   byte-faithful because the emitted kernel is what the reproduction is OF; the matrix below shows
   it is not what the defect keys on — the [pools-literal] and [param-literal] rows drop every
   dynamic load and miscompile identically. *)
let pool_preamble =
  {|    device char* __pool0 [[buffer(0)]],
    device char* __pool1 [[buffer(1)]],
    device char* __pool10 [[buffer(2)]],
    device char* __pool11 [[buffer(3)]],
    device char* __pool12 [[buffer(4)]],
    device char* __pool13 [[buffer(5)]],
    device char* __pool14 [[buffer(6)]],
    device char* __pool15 [[buffer(7)]],
    device char* __pool2 [[buffer(8)]],
    device char* __pool3 [[buffer(9)]],
    device char* __pool4 [[buffer(10)]],
    device char* __pool5 [[buffer(11)]],
    device char* __pool6 [[buffer(12)]],
    device char* __pool7 [[buffer(13)]],
    device char* __pool8 [[buffer(14)]],
    device char* __pool9 [[buffer(15)]],
    device const uint* __pool_slots [[buffer(16)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) {

  /* Pool base pointers. */
  device char* __pools[16] = { __pool0, __pool1, __pool2, __pool3, __pool4, __pool5, __pool6, __pool7, __pool8, __pool9, __pool10, __pool11, __pool12, __pool13, __pool14, __pool15 };
|}

let kernel ~name ~body = Printf.sprintf "kernel void %s(\n%s%s}\n" name pool_preamble body

(* Pool layout, in bytes within the single slab every buffer index is bound to. Separate regions so
   no two kernels alias, and the reproducer's inputs survive the timing runs. *)
let stream_threads = 16384
let stream_k = 1024
let chain_threads = 65536
let chain_k = 4096
let scalar_n = 1 lsl 20
let out_off = 0
let out_floats = stream_threads
let produced_off = 1 lsl 20
let total_off = 2 lsl 20
let src_off = 4 lsl 20
let src_floats = stream_threads * stream_k
let pool_bytes = src_off + (4 * src_floats) + 4096
let seq_len = 4
let width = 16

(* Slot-table entries 4..7, read by the [`Slots_read] contribution below. Small integers, distinct,
   none zero. *)
let slot_values = [| 3; 5; 7; 11 |]

(* The emitted body of [total_fwd__seg1] (scalar_rmw_accumulation's localized leg, Metal, f32): the
   accumulator opens into a scope local, the nest updates the local, the node is stored once.

   The knobs turn that one kernel into a matrix isolating what the miscompiling pass keys on, which
   is what an upstream report needs and what decides whether a narrower or cheaper workaround than
   [volatile] exists:

   - [qualifier]: the shipped workaround, on the accumulator's declaration.
   - [restrict]: whether the pooled pointers carry [__restrict] (the Metal backend emits it).
   - [pointers]: how a per-node pointer is formed. [`Slots] is what the backend emits — a slab base
     chosen by a value LOADED from the slot table, offset by another. [`Slots_in_locals] performs
     the same two loads into named locals first. [`Pools_literal] indexes the pool array with a
     literal and offsets by a literal, so nothing is loaded. [`Param_literal] skips the pool array
     as well, offsetting the kernel parameter directly.
   - [prezero]: where the device store that precedes the nest lands — the accumulated cell (what
     codegen emits, a [Zero_out]), a neighbouring cell, a cell of the READ node, or nowhere.
   - [opening]: whether the accumulator opens by reading its cell (what the localizer emits) or from
     a literal.
   - [fence]: a device memory barrier between that store and the nest.
   - [src_volatile]: the qualifier on the READ pointer rather than on the accumulator — a candidate
     workaround that would leave the accumulator register-resident.
   - [contribution]: what the loop body reads. [`Pooled_read] reads the pooled input node;
     [`Slots_read] reads only the (device, dynamically indexed) slot table, so the accumulation
     depends on device memory without dereferencing any slot-DERIVED pointer. *)
let repro_body ?(qualifier = "") ?(restrict = " __restrict") ?(pointers = `Slots)
    ?(src_volatile = false) ?(contribution = `Pooled_read) ?(opening = `Read_cell)
    ?(prezero = `Accumulated_cell) ?(fence = false) () =
  let ptr ~qual ~name ~slot_base ~slot_off ~byte_off =
    let typ = qual ^ "float" in
    match pointers with
    | `Slots ->
        Printf.sprintf "  device %s*%s %s = (device %s*)(__pools[__pool_slots[%d]] + __pool_slots[%d]);\n"
          typ restrict name typ slot_base slot_off
    | `Slots_in_locals ->
        Printf.sprintf
          "  const uint %s_pool = __pool_slots[%d];\n\
          \  const uint %s_offset = __pool_slots[%d];\n\
          \  device %s*%s %s = (device %s*)(__pools[%s_pool] + %s_offset);\n"
          name slot_base name slot_off typ restrict name typ name name
    | `Pools_literal ->
        Printf.sprintf "  device %s*%s %s = (device %s*)(__pools[0] + %d);\n" typ restrict name typ
          byte_off
    | `Param_literal ->
        Printf.sprintf "  device %s*%s %s = (device %s*)(__pool0 + %d);\n" typ restrict name typ
          byte_off
  in
  let decls =
    ptr
      ~qual:(if src_volatile then "volatile " else "")
      ~name:"produced" ~slot_base:0 ~slot_off:1 ~byte_off:produced_off
    ^ ptr ~qual:"" ~name:"total" ~slot_base:2 ~slot_off:3 ~byte_off:total_off
  in
  let term =
    match contribution with
    | `Pooled_read -> "((float)((16*i88+i89)) + produced[((0) * 4 + i88) * 16 + i89])"
    | `Slots_read -> "((float)((16*i88+i89)) + (float)(__pool_slots[4 + ((16*i88+i89) & 3)]))"
    | `Index_only -> "((float)((16*i88+i89)))"
  in
  Printf.sprintf
    "%s  /* Local declarations and initialization. */\n\n\
    \  /* Main logic. */\n\
     %s  {\n\
    \    %sfloat v33_total;\n\
    \    v33_total = %s;\n\
    \    for (int32_t i88 = 0; i88 <= %d; ++i88) {\n\
    \      for (int32_t i89 = 0; i89 <= %d; ++i89) {\n\
    \        v33_total = (v33_total + %s);\n\
    \      }\n\
    \    }\n\
    \    total[0] = v33_total;\n\
    \  }\n\
    \  /* end */\n"
    decls
    ((match prezero with
     | `Accumulated_cell -> "  total[0] = (float)(0.0);\n"
     | `Other_cell -> "  total[1] = (float)(0.0);\n"
     | `Read_node_cell -> Printf.sprintf "  produced[%d] = (float)(0.0);\n" (seq_len * width)
     | `None -> "")
    ^ if fence then "  threadgroup_barrier(mem_flags::mem_device);\n" else "")
    qualifier
    (match opening with `Read_cell -> "total[0]" | `Literal_zero -> "(float)(0.0)")
    (seq_len - 1) (width - 1) term

(* The reproducer's input: [produced = source * 1.25 - 0.5] over [source i = 1 + 0.125 i], the
   values scalar_rmw_accumulation feeds its localized leg. Every cell distinct, none zero, so a
   dropped or replayed iteration cannot hide. *)
let produced i = ((1.0 +. (float_of_int i *. 0.125)) *. 1.25) -. 0.5

let oracle ?(scale = 1.0) contribution =
  let acc = ref 0.0 in
  for n = 0 to (seq_len * width) - 1 do
    let term =
      match contribution with
      | `Pooled_read -> scale *. produced n
      | `Slots_read -> float_of_int slot_values.(n land 3)
      | `Index_only -> 0.0
    in
    acc := !acc +. float_of_int n +. term
  done;
  !acc

type variant = { vname : string; vbody : string; vexpected : float; vnote : string }

let variants =
  [
    {
      vname = "emitted-plain";
      vbody = repro_body ();
      vexpected = oracle `Pooled_read;
      vnote = "what codegen would emit without the workaround";
    };
    {
      vname = "emitted-volatile";
      vbody = repro_body ~qualifier:"volatile " ();
      vexpected = oracle `Pooled_read;
      vnote = "the shipped workaround: volatile accumulator";
    };
    {
      vname = "no-restrict";
      vbody = repro_body ~restrict:"" ();
      vexpected = oracle `Pooled_read;
      vnote = "plain accumulator, pooled pointers without __restrict";
    };
    {
      vname = "slots-in-locals";
      vbody = repro_body ~pointers:`Slots_in_locals ();
      vexpected = oracle `Pooled_read;
      vnote = "plain, same two slot loads named in locals first";
    };
    {
      vname = "pools-literal";
      vbody = repro_body ~pointers:`Pools_literal ();
      vexpected = oracle `Pooled_read;
      vnote = "plain, pool array indexed by a literal, literal offset";
    };
    {
      vname = "param-literal";
      vbody = repro_body ~pointers:`Param_literal ();
      vexpected = oracle `Pooled_read;
      vnote = "plain, straight off the kernel parameter, literal offset";
    };
    {
      vname = "volatile-source";
      vbody = repro_body ~src_volatile:true ();
      vexpected = oracle `Pooled_read;
      vnote = "plain accumulator, volatile READ pointer";
    };
    {
      vname = "slots-read-only";
      vbody = repro_body ~contribution:`Slots_read ();
      vexpected = oracle `Slots_read;
      vnote = "plain, contribution reads the slot table, no pooled pointer";
    };
    {
      vname = "slots-read-volatile";
      vbody = repro_body ~qualifier:"volatile " ~contribution:`Slots_read ();
      vexpected = oracle `Slots_read;
      vnote = "the same, with the volatile accumulator";
    };
    {
      vname = "init-literal";
      vbody = repro_body ~opening:`Literal_zero ();
      vexpected = oracle `Pooled_read;
      vnote = "plain, accumulator opened from a literal instead of the cell";
    };
    {
      vname = "no-prezero";
      vbody = repro_body ~prezero:`None ();
      vexpected = oracle `Pooled_read;
      vnote = "plain, cell zeroed by the host instead of by the kernel";
    };
    {
      vname = "prezero-fenced";
      vbody = repro_body ~fence:true ();
      vexpected = oracle `Pooled_read;
      vnote = "plain, a device memory barrier after the kernel's zeroing store";
    };
    {
      vname = "prezero-other-cell";
      vbody = repro_body ~prezero:`Other_cell ();
      vexpected = oracle `Pooled_read;
      vnote = "plain, the kernel's zeroing store goes to a neighbouring cell";
    };
    {
      vname = "prezero-read-node";
      vbody = repro_body ~prezero:`Read_node_cell ();
      vexpected = oracle `Pooled_read;
      vnote = "plain, the zeroing store goes to an unread cell of the READ node";
    };
    {
      vname = "index-only";
      vbody = repro_body ~contribution:`Index_only ();
      vexpected = oracle `Index_only;
      vnote = "plain, contribution reads nothing at all";
    };
  ]

(* Timing shapes. Each is a localized serial reduction over pooled pointers — the exact statement
   shape the predicate qualifies — differing in what bounds it. *)

(* Each timing body carries the miscompiling shape's full context — the zeroing store to the
   accumulated cell, then the serial nest, then the single closing store — so the arms are timing
   the kernel the predicate actually decides about. [acc] qualifies the accumulator (the shipped
   workaround), [src] qualifies the READ pointer (the candidate the matrix found sufficient). *)
let timing_decls src =
  Printf.sprintf
    "  device %sfloat* __restrict src = (device %sfloat*)(__pools[__pool_slots[0]] + \
     __pool_slots[1]);\n\
    \  device float* __restrict out = (device float*)(__pools[__pool_slots[2]] + \
     __pool_slots[3]);\n"
    src src

let stream_body ~acc ~src =
  Printf.sprintf
    {|%s  const int32_t row = (int32_t)(gid.x * 256 + lid.x);
  out[row] = (float)(0.0);
  {
    %sfloat acc;
    acc = out[row];
    for (int32_t k = 0; k <= %d; ++k) {
      acc = (acc + src[row * %d + k]);
    }
    out[row] = acc;
  }
|}
    (timing_decls src) acc (stream_k - 1) stream_k

(* Accumulator-bound: sixteen cached floats per thread, a long dependency chain on the accumulator.
   This is where a per-iteration store/load of the accumulator has to show up. *)
let chain_body ~acc ~src =
  Printf.sprintf
    {|%s  const int32_t row = (int32_t)(gid.x * 256 + lid.x);
  out[row] = (float)(0.0);
  {
    %sfloat acc;
    acc = out[row];
    for (int32_t k = 0; k <= %d; ++k) {
      acc = fma(src[row * 16 + (k & 15)], (float)(k & 7), acc);
    }
    out[row] = acc;
  }
|}
    (timing_decls src) acc (chain_k - 1)

(* The scalar-loss shape: one thread, one long serial reduction, the accumulator's latency fully
   exposed. This is the kernel [scalar_rmw_accumulation] and the RoPE gradient loss actually run. *)
let scalar_body ~acc ~src =
  Printf.sprintf
    {|%s  out[0] = (float)(0.0);
  {
    %sfloat acc;
    acc = out[0];
    for (int32_t k = 0; k <= %d; ++k) {
      acc = (acc + src[k]);
    }
    out[0] = acc;
  }
|}
    (timing_decls src) acc (scalar_n - 1)

(* The three renderings compared: none, the shipped one, and the matrix's cheaper candidate. *)
let arms =
  [
    ("plain", ("", ""));
    ("volatile-acc", ("volatile ", ""));
    ("volatile-src", ("", "volatile "));
  ]

(* Every kernel is compiled as its own single-kernel library. One library holding the whole matrix
   would leave every verdict open to the objection that a neighbouring kernel steered the
   optimizer; separate libraries make each row a self-contained translation unit, which is also the
   form an upstream report wants. *)
let translation_unit ~name ~body =
  "#include <metal_stdlib>\nusing namespace metal;\n\n" ^ kernel ~name ~body

let timing_kernels =
  List.concat_map
    (fun (arm, (acc, src)) ->
      List.map
        (fun (base, body) -> (base ^ "_" ^ arm, body ~acc ~src))
        [ ("stream", stream_body); ("chain", chain_body); ("scalar", scalar_body) ])
    arms

let all_units =
  List.map (fun v -> ("repro_" ^ v.vname, v.vbody)) variants
  @ List.map (fun (name, body) -> (name, body)) timing_kernels

(* The cleanest run of an arm, which is what the project's own segment timings report: a GPU kernel
   has a floor and only ever loses time to interference, so the minimum is the estimate that noise
   cannot inflate. Interleaving and rotation are what keep the two arms' floors comparable. *)
let best xs = List.fold_left min infinity xs

(* MSL forbids [-] and [.] in identifiers; the variant names carry dashes for readability. *)
let mangle = String.map (fun c -> if c = '-' then '_' else c)

let () =
  if Array.exists (String.equal "--dump-source") Sys.argv then
    List.iter
      (fun (name, body) -> print_string (mangle (translation_unit ~name ~body) ^ "\n"))
      all_units;
  let device = Me.Device.create_system_default () in
  let queue = Me.CommandQueue.on_device device in
  let options = Me.CompileOptions.init () in
  let states = Hashtbl.create 16 in
  List.iter
    (fun (name, body) ->
      let name = mangle name in
      let source = mangle (translation_unit ~name ~body) in
      let library = Me.Library.on_device device ~source options in
      let func = Me.Library.new_function_with_name library name in
      Hashtbl.replace states name (fst (Me.ComputePipelineState.on_device_with_function device func)))
    all_units;
  let pso name = Hashtbl.find states (mangle name) in
  let ropts =
    Me.ResourceOptions.(
      storage_mode_shared + cpu_cache_mode_write_combined + hazard_tracking_mode_untracked)
  in
  let pool = Me.Buffer.on_device device ~length:pool_bytes ropts in
  let slots_repro = Me.Buffer.on_device device ~length:64 ropts in
  let slots_bench = Me.Buffer.on_device device ~length:64 ropts in
  let open Ctypes in
  let fp = coerce (ptr void) (ptr float) (Me.Buffer.contents pool) in
  let at byte_off = fp +@ (byte_off / 4) in
  for i = 0 to (seq_len * width) - 1 do
    at produced_off +@ i <-@ produced i
  done;
  at total_off <-@ 0.0;
  (* Timing input: bounded values, so the streamed and chained sums stay finite and comparable. *)
  for i = 0 to src_floats - 1 do
    at src_off +@ i <-@ (0.5 +. float_of_int (i land 1023)) /. 1024.0
  done;
  for i = 0 to out_floats - 1 do
    at out_off +@ i <-@ 0.0
  done;
  let set_slots buf values =
    let sp = coerce (ptr void) (ptr uint32_t) (Me.Buffer.contents buf) in
    List.iteri (fun i v -> sp +@ i <-@ Unsigned.UInt32.of_int v) values
  in
  (* (pool index, byte offset) pairs, in the order the kernel's preamble consumes them, then the
     four scalars the [`Slots_read] contribution sums. *)
  set_slots slots_repro
    ([ 0; produced_off; 0; total_off ] @ Array.to_list slot_values);
  set_slots slots_bench [ 0; src_off; 0; out_off ];
  let dispatch ~name ~slots ~groups ~threads =
    let pso = pso name in
    let cb = Me.CommandBuffer.on_queue queue in
    let enc = Me.ComputeCommandEncoder.on_buffer cb in
    Me.ComputeCommandEncoder.set_compute_pipeline_state enc pso;
    for i = 0 to 15 do
      Me.ComputeCommandEncoder.set_buffer enc ~index:i pool
    done;
    Me.ComputeCommandEncoder.set_buffer enc ~index:16 slots;
    Me.ComputeCommandEncoder.dispatch_threadgroups enc
      ~threadgroups_per_grid:{ width = groups; height = 1; depth = 1 }
      ~threads_per_threadgroup:{ width = threads; height = 1; depth = 1 };
    Me.ComputeCommandEncoder.end_encoding enc;
    Me.CommandBuffer.commit cb;
    Me.CommandBuffer.wait_until_completed cb;
    Me.CommandBuffer.get_gpu_end_time cb -. Me.CommandBuffer.get_gpu_start_time cb
  in

  (* ---- Half 1: the reproducer matrix. ---- *)
  Printf.printf "== localized pooled accumulation, %d x %d, f32 ==\n%!" seq_len width;
  let run v =
    at total_off <-@ 0.0;
    ignore (dispatch ~name:("repro_" ^ v.vname) ~slots:slots_repro ~groups:1 ~threads:1 : float);
    let got = !@(at total_off) in
    let ok = abs_float (got -. v.vexpected) < 1e-3 in
    Printf.printf "  %-20s %12.4f  expected %10.4f  %s   %s\n%!" v.vname got v.vexpected
      (if ok then "ok        " else "MISCOMPILED")
      v.vnote;
    ok
  in
  let outcomes = List.map (fun v -> (v.vname, run v)) variants in
  let ok name = match List.assoc_opt name outcomes with Some b -> b | None -> true in
  Printf.printf "  verdict: %s\n%!"
    (if ok "emitted-plain" then
       "this toolchain computes the unqualified emitted form correctly (defect not observed here)"
     else "the unqualified emitted form MISCOMPILES here; see the matrix for what changes that");

  (* How the error behaves as the input moves: an additive constant would mean the accumulator
     started from something other than the cell it read, while an error proportional to the input
     would mean the contributions themselves are being read or combined wrongly. *)
  Printf.printf "\n  error vs. input scale (emitted-plain):\n%!";
  List.iter
    (fun scale ->
      for i = 0 to (seq_len * width) - 1 do
        at produced_off +@ i <-@ (scale *. produced i)
      done;
      at total_off <-@ 0.0;
      ignore (dispatch ~name:"repro_emitted-plain" ~slots:slots_repro ~groups:1 ~threads:1 : float);
      let got = !@(at total_off) in
      let want = oracle ~scale `Pooled_read in
      Printf.printf "    input x %-6.2f got %12.4f  expected %12.4f  error %+12.4f\n%!" scale got
        want (got -. want))
    [ 0.0; 1.0; 2.0; 10.0 ];
  for i = 0 to (seq_len * width) - 1 do
    at produced_off +@ i <-@ produced i
  done;

  (* ---- Half 2: the tax. ---- *)
  (* A multiple of the arm count, so the rotation gives each arm each position equally often. *)
  let repeats = 30 in
  Printf.printf "\n== volatile tax, GPU clock, best of %d rotated interleaved repeats ==\n%!"
    repeats;
  Printf.printf "  %-8s %12s %12s %8s %12s %8s   %s\n%!" "shape" "plain(ms)" "vol-acc(ms)" "ratio"
    "vol-src(ms)" "ratio" "note";
  let shapes =
    [
      ( "stream",
        stream_threads / 256,
        256,
        Printf.sprintf "%d threads x %d-element serial reduction (memory bound)" stream_threads
          stream_k );
      ( "chain",
        chain_threads / 256,
        256,
        Printf.sprintf "%d threads x %d accumulator-bound FMAs" chain_threads chain_k );
      ("scalar", 1, 1, Printf.sprintf "1 thread x %d-element serial reduction (loss shape)" scalar_n);
    ]
  in
  (* Left-rotation by [n], used to move each arm through each position within a repeat. *)
  let rotate n lst =
    let len = List.length lst in
    let n = ((n mod len) + len) mod len in
    List.filteri (fun i _ -> i >= n) lst @ List.filteri (fun i _ -> i < n) lst
  in
  List.iter
    (fun (shape, groups, threads, note) ->
      (* Warm every pipeline up before any of them is timed, then interleave, so a thermal or clock
         ramp is shared by the arms instead of landing on whichever ran first. *)
      List.iter
        (fun (arm, _) ->
          ignore (dispatch ~name:(shape ^ "_" ^ arm) ~slots:slots_bench ~groups ~threads : float))
        arms;
      let samples = Hashtbl.create 4 in
      List.iter (fun (arm, _) -> Hashtbl.replace samples arm []) arms;
      (* Interleaving alone would leave each arm pinned to a position within the cycle, so a clock
         or thermal drift ACROSS a cycle would be read as a property of whichever qualifier sits
         last (Codex P2, round 1). Rotating the order each repeat gives every arm each position an
         equal number of times over a multiple of three repeats, so position averages out of the
         the reported floors instead of loading onto one treatment. *)
      for r = 1 to repeats do
        List.iter
          (fun (arm, _) ->
            let t = dispatch ~name:(shape ^ "_" ^ arm) ~slots:slots_bench ~groups ~threads in
            Hashtbl.replace samples arm (t :: Hashtbl.find samples arm))
          (rotate r arms)
      done;
      let med arm = best (Hashtbl.find samples arm) in
      let plain = med "plain" and vacc = med "volatile-acc" and vsrc = med "volatile-src" in
      let ratio x = if plain > 0.0 then x /. plain else 0.0 in
      Printf.printf "  %-8s %12.4f %12.4f %7.2fx %12.4f %7.2fx   %s\n%!" shape (plain *. 1000.0)
        (vacc *. 1000.0) (ratio vacc) (vsrc *. 1000.0) (ratio vsrc) note)
    shapes;
  Printf.printf
    "\n\
     (GPU times come from the command buffer's own clock; thermal state still moves them, so \
     compare ratios across runs rather than absolute milliseconds. The memory-bound leg's ratio \
     rides on traffic the qualifier does not change and lands either side of 1.0: read it as \"no \
     measurable tax\", not as a speedup.)\n\
     %!"
