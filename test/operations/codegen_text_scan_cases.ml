(** What the codegen-text scan calls a member, on input built to break it rather than on whatever
    the repository happens to hold today (gh-ocannl-712).

    The live census in [codegen_text_inventory] exercises only the shapes some golden or some test
    currently spells, and both halves of the classifier are wrong in ways that are silent. A marker
    that stops matching shrinks the inventory rather than failing it, so a golden quietly leaves the
    list a codegen author reads. A marker that matches too much adds a file nobody has to re-run,
    which is the cheaper error but trains the reader to skim. And on the source side an unrecognised
    idiom loses the fragment, not the file, so the inventory keeps looking complete.

    So each rule is pinned here on input of its own, and -- for every marker whose needle a claim
    label could plausibly contain -- beside the nearest thing that must NOT be a member: a
    {!Verdict} line quoting the kernel's vocabulary, a memory-mode table naming [On_device], prose
    about how a constant is spelled. Those are the negative controls of the RULE. *)

(* This file's own fixtures spell emitted-kernel syntax, and this file is deliberately NOT a member
   of either population: the fixtures are inputs to the classifier, not assertions about a kernel
   this repository emits, so a codegen change owes them nothing. What decides that is the rule
   itself rather than an exemption -- the source rule asks whether a file READS generated source,
   and nothing here does. *)

open Base
open Stdio
module Scan = Test_utils.Codegen_text_scan

let fail fmt = Printf.ksprintf Verdict.fail fmt

(** How a classified golden reads, for comparison: the families, then where the evidence came
    from. [none] is a file the scan does not call a member. *)
let render_golden = function
  | None -> "none"
  | Some (g : Scan.golden) ->
      Printf.sprintf "[%s]%s%s"
        (String.concat ~sep:" " g.Scan.families)
        (match g.Scan.by_extension with Some ext -> " extension " ^ ext | None -> "")
        (match g.Scan.tags with [] -> "" | tags -> " markers " ^ String.concat ~sep:" " tags)

let c_kernel =
  {|
 void zero_out_codegen(
    float *restrict out) {

  /* Local declarations and initialization. */
  float acc_a[2] __attribute__((aligned(32))) = {0};

  /* Main logic. */
  for (int32_t i2 = 0; i2 <= 1; ++i2) {
    acc_a[i2] = (float)(0.0);
  }
}
|}

let golden_cases =
  [
    (* The marker families, each on the text it was written for. *)
    ( "a C kernel is a member by its markers, under any name",
      "some_test.expected",
      c_kernel,
      "[c] markers c-align-attr c-decl-banner c-for c-logic-banner c-prec-cast c-restrict" );
    ( "a CUDA kernel names the launch vocabulary",
      "gpu_test.expected",
      "__global__ void k(float *restrict o) { o[threadIdx.x] = (float)(1.0); }",
      "[c cuda] markers c-prec-cast c-restrict cuda-kernel" );
    ( "a Metal kernel names its address spaces",
      "msl_test.expected",
      "kernel void k(device float *o [[buffer(0)]]) { threadgroup float frag[8]; }",
      "[metal] markers metal-kernel" );
    ( "a low-level IR dump is a member by its loop headers and assignments",
      "dump_test.expected",
      "c_fwd (): /* c fwd */\n  for i10 = 0 to 2 {\n    a[i10] := 5*i10;\n  }\n  /* end */",
      "[ll] markers ll-assign ll-end ll-loop" );
    ( "a routine log carries both the IR line and the C rendering",
      "run-cc-0-0.log.expected",
      "COMMENT: init params for g\n# a[0] := -4.0;\na[0]{=MAYBE UNINITIALIZED} = (float)(-4.0)",
      "[c ll routine-log] markers c-prec-cast ll-assign routine-log" );
    (* Declared by extension: a member whatever it holds, because the name says which artifact it
       snapshots. A machine without the toolchain records a notice here and the file is still that
       backend's snapshot, to be re-recorded when the hardware next runs. *)
    ( "a snapshot is a member by its extension whatever it holds",
      "top_down_prec.cu.expected",
      "CUDA is not available on this machine.\n",
      "[cuda] extension .cu.expected" );
    ( "a declaring extension names the substrate; markers stay as evidence",
      "top_down_prec.metal.expected",
      "kernel void k() { }",
      "[metal] extension .metal.expected markers metal-kernel" );
    ( "HIP spells CUDA's launch vocabulary, and is still HIP",
      "zero_out_local_decl.hip.expected",
      "__global__ void k(float *restrict o) { o[threadIdx.x] = (float)(1.0); }",
      "[hip] extension .hip.expected markers c-prec-cast c-restrict cuda-kernel" );
    ( "a .hip snapshot is its own family, not CUDA's",
      "x.hip.expected",
      "nothing here\n",
      "[hip] extension .hip.expected" );
    ( "an .ll snapshot is declared even when the dump is empty",
      "x.ll.expected",
      "",
      "[ll] extension .ll.expected" );
    (* The negative controls: the nearest legitimate goldens the classifier must leave alone. Every
       one of these is a real shape from this repository's test output. *)
    ( "a verdict quoting the kernel's vocabulary is prose, not Metal",
      "schedule_pad.expected",
      "padded GPU intrinsics fire against the threadgroup fragment: true\n\
       padded 33x65x70 runs the register-tiled micro-kernel: true\n",
      "none" );
    ( "a claim about how a constant is spelled is not a constant",
      "prose.expected",
      "every emitted float constant carries a radix point, as in (float)(0.0): true\n",
      "none" );
    ( "a PASS column is a verdict however it reads",
      "columns.expected",
      "the kernel declares threadgroup float fragments: PASS\n",
      "none" );
    ( "a memory-mode table is not device code",
      "test_metal_storage_mode.expected",
      "On_device              -> Shared\nOn_host                -> Managed\n",
      "none" );
    ( "a numeric table of shapes is not a loop nest",
      "shapes.expected",
      "batch 4 x input 3 -> output 2\ntotal elements: 24\n",
      "none" );
    ( "a failure line is a verdict too",
      "failed.expected",
      "FAIL: the kernel contains (float)(0.0) where it should not\n",
      "none" );
  ]

let render_site = function
  | None -> "none"
  | Some (s : Scan.site) ->
      Printf.sprintf "%s%s%s"
        (String.concat ~sep:" " s.Scan.pins)
        (if s.Scan.partial then " +partial" else "")
        (if s.Scan.direct then " +direct" else "")

let source_cases =
  [
    ( "assert_emits pins its contains argument",
      {ocaml|let () = Generated.assert_emits ~routine:"r" ~contains:"__shared__" "shared"|ocaml},
      {|"__shared__"|} );
    ( "assert_omits pins the fragment that must be absent",
      {ocaml|let () = Test_utils.Generated.assert_omits ~routine:r ~contains:"volatile" "no rmw"|ocaml},
      {|"volatile"|} );
    ( "the has idiom: a predicate closing over the source",
      {ocaml|let () =
  let src = Generated.read "pad_packed" in
  let has s = String.is_substring src ~substring:s in
  p "tiled" (has "Tile_mma register tiling" && not (has "tmma_"))|ocaml},
      {|"Tile_mma register tiling" "tmma_"|} );
    ( "a predicate taking the source as a parameter pins at a tainted call site",
      {ocaml|let src_has src s = String.is_substring src ~substring:s
let () =
  let vec = Generated.read "nsc_vec_bf16" in
  p "widened" (src_has vec "OCANNL_VEC_WIDEN_BFLOAT16")|ocaml},
      {|"OCANNL_VEC_WIDEN_BFLOAT16"|} );
    ( "the same predicate over text that is not generated source pins nothing",
      {ocaml|let src_has src s = String.is_substring src ~substring:s
let () =
  let _ = Generated.read "r" in
  p "backend" (src_has backend_name "metal")|ocaml},
      "" );
    ( "a substring test straight against the read is a pin",
      {ocaml|let () = p "cast" (String.is_substring (Generated.read "r") ~substring:"(float)(0.0)")|ocaml},
      {|"(float)(0.0)"|} );
    ( "a sprintf format is a pinned fragment with a hole in it",
      {ocaml|let () =
  let src = Generated.read "pad_packed" in
  let has s = String.is_substring src ~substring:s in
  p "bound" (has (Printf.sprintf "< (int)(%d.0))) {" m_ext))|ocaml},
      {|sprintf "< (int)(%d.0))) {"|} );
    ( "text the scan cannot name marks the itemisation partial, without losing the file",
      {ocaml|let () =
  let src = Generated.read "r" in
  let has s = String.is_substring src ~substring:s in
  p "shared" (has shared_keyword)|ocaml},
      "+partial" );
    ( "taint reaches through a helper that returns the source",
      {ocaml|let read_on_cpu routine = if on_cpu then Generated.read routine else ""
let () =
  let src = read_on_cpu "nsc_half_fma" in
  let has t = String.is_substring src ~substring:t in
  p "fma" (has "OCANNL_HALF_FMA")|ocaml},
      {|"OCANNL_HALF_FMA"|} );
    ( "a routine name is not a pinned fragment",
      {ocaml|let () = Generated.assert_emits ~routine:"aw_bf16_naive" ~contains:"fmaf(" "fma"|ocaml},
      {|"fmaf("|} );
    ( "a test opening build_files/ itself is a member, and is told apart",
      {ocaml|let sources =
  Stdlib.Sys.readdir (Utils.build_files_dir ())
  |> Array.to_list
  |> List.map ~f:(fun f -> Stdio.In_channel.read_all f)
let () =
  let has substring = List.exists sources ~f:(String.is_substring ~substring) in
  p "guard" (has ": (float)(0.0))")|ocaml},
      {|": (float)(0.0))" +direct|} );
    ( "counting occurrences with ~pattern pins the fragment counted",
      {ocaml|let () =
  let count_sub src sub = String.substr_index_all src ~may_overlap:false ~pattern:sub |> List.length in
  let src2 = Generated.read "pipe_mm_d2" in
  p "rotated" (count_sub src2 "% 2" >= 4)|ocaml},
      {|"% 2"|} );
    ( "a concatenation with a literal part is a fragment with a hole in it",
      {ocaml|let () =
  let src = Generated.read "flit_f32" in
  let has s = String.is_substring src ~substring:s in
  p "spelled" (has ("(float)(" ^ spelling ^ ")"))|ocaml},
      {|"(float)(" ^ ... ^ ")"|} );
    ( "the haystack may reach the test through a local binding",
      {ocaml|let has sub s =
  let body = match String.substr_index s ~pattern:"Main logic" with
    | Some i -> String.subo s ~pos:i
    | None -> s
  in
  String.is_substring body ~substring:sub
let () =
  let src = Generated.read "uvl_fwd" in
  p "lane" (has "_uniform_lane(" src)|ocaml},
      {|"Main logic" "_uniform_lane("|} );
    ( "a predicate over the backend's NAME pins nothing, however it is spelled",
      {ocaml|let () =
  let _ = Generated.read "r" in
  let on s = String.is_substring backend_name ~substring:s in
  p "gpu" (on "cuda" || on "metal")|ocaml},
      "" );
    ( "a source reached through a tuple pattern is still a source",
      {ocaml|let run () = (values, Generated.read "uvl_fwd")
let () =
  let _vals, src = run () in
  p "vec" (String.is_substring src ~substring:"_uniform_vec(")|ocaml},
      {|"_uniform_vec("|} );
    ( "a test binding its own build_file is not reading the artifact directory",
      {ocaml|let build_file path ~extra_pad entries = write path entries ~extra_pad
let () =
  let _ = build_file "aligned.safetensors" ~extra_pad:0 entries in
  p "aligned" true|ocaml},
      "none" );
    ( "the qualified Utils.build_file is a read of the artifact directory",
      {ocaml|let () = p "wrote" (Stdlib.Sys.file_exists (Utils.build_file "k.c"))|ocaml},
      "+direct" );
    ( "a test that reads no generated source is not a member",
      {ocaml|let () = p "shapes agree" (String.is_substring rendered ~substring:"3x5")|ocaml},
      "none" );
    ( "naming the reader in a comment is not reading it",
      {ocaml|(* Generated.read would answer this, but the check is on values. *)
let () = p "values" (Array.for_all2_exn got want ~f:Float.equal)|ocaml},
      "none" );
  ]

let () =
  List.iter golden_cases ~f:(fun (name, path, contents, expected) ->
      let found = render_golden (Scan.classify_golden ~path ~contents) in
      if String.equal found expected then printf "ok: golden -- %s\n" name
      else fail "golden -- %s: expected [%s], found [%s]" name expected found);
  List.iter source_cases ~f:(fun (name, source, expected) ->
      let found =
        try render_site (Scan.classify_source ~path:"case.ml" ~contents:source)
        with _ ->
          fail "source -- %s: the snippet does not parse" name;
          "<unparsed>"
      in
      if String.equal (String.strip found) (String.strip expected) then
        printf "ok: source -- %s\n" name
      else fail "source -- %s: expected [%s], found [%s]" name expected found)
