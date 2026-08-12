(* The nvrtc [--gpu-architecture] policy (gh-ocannl-481 item 3, D4).

   [Cuda_backend.gpu_arch_options] decides one thing with two very different answers:

   - a FLOOR: the lowest arch whose PTX contains the instructions a kernel uses. The driver
     forward-JIT-compiles floor-targeted PTX on every later GPU, so one compile covers everything
     above it — which is why a triggered floor is deliberately NOT raised to the device arch, only
     up to compute_75 (CUDA 13 dropped offline compilation below that).
   - a FAMILY target: a [compute_120a]-style architecture-specific arch, loadable only by the device
     family it names. Blackwell's block-scaled [kind::mxf8f6f4] forms exist only under such a
     target, so it is the one marker that gives up forward-JIT portability — and therefore the one
     gated on the attached devices' own family, so family PTX is never produced for a device that
     could not load it.

   No arm emits [(mma-mxfp8)] yet (block scaling is blocked on OCANNL having microscaling storage
   at all), so this pins the MECHANISM: that the marker selects a family target on a family device,
   that it does not on any other device, and that its presence changes nothing for every existing
   floor marker. The function is pure, so no GPU is needed — this file is selected on cudajit being
   installed, not on hardware being present. *)

open Base

let check label ~device_cc src ~expected =
  let got = Cuda_backend.gpu_arch_options ~device_cc src in
  let render l = if List.is_empty l then "(none)" else String.concat ~sep:" " l in
  if not (List.equal String.equal got expected) then
    failwith
      (Printf.sprintf "%s: expected %s, got %s" label (render expected) (render got))

let () =
  (* No markers, no arch option at all: the kernel compiles at nvrtc's default target. *)
  check "plain kernel" ~device_cc:120 "__global__ void k() {}" ~expected:[];
  (* Floors, and their forward-JIT contract: the option names the FLOOR, never the device. *)
  check "fp8 mma.sync floor" ~device_cc:120 "/* tile_mma 16x8x32 (mma-fp8) e5m2 */"
    ~expected:[ "--gpu-architecture=compute_89" ];
  check "bf16 mma.sync floor" ~device_cc:120 "/* tile_mma 16x8x16 (mma-bf16) */"
    ~expected:[ "--gpu-architecture=compute_80" ];
  check "wmma floor" ~device_cc:120 "nvcuda::wmma::mma_sync(a, b, c, d);"
    ~expected:[ "--gpu-architecture=compute_75" ];
  check "wmma bf16 floor" ~device_cc:120 "nvcuda::wmma::mma_sync(x); /* (wmma-bf16) */"
    ~expected:[ "--gpu-architecture=compute_80" ];
  (* The [cp.async] staging builtins (gh-ocannl-487 phase 2): the builtin's own name is the
     marker, so any emitted call (or the prepended definition) triggers the sm_80 floor. *)
  check "cp.async floor" ~device_cc:120 "ocannl_cp_async4(&tile[0], &src[0]);"
    ~expected:[ "--gpu-architecture=compute_80" ];
  (* Batched sources take the max of the triggered floors, not the first match. *)
  check "batched floors take the max" ~device_cc:120
    "nvcuda::wmma::mma_sync(x); /* (mma-fp8) */" ~expected:[ "--gpu-architecture=compute_89" ];
  check "cp.async batched with fp8 takes the max" ~device_cc:120
    "ocannl_cp_async_wait_all(); /* (mma-fp8) */" ~expected:[ "--gpu-architecture=compute_89" ];
  (* A device below the raise point keeps the literal floor (it must be paired with an nvrtc that
     still accepts it). *)
  check "half arithmetic on an old device" ~device_cc:70 "__hfma(a, b, c)"
    ~expected:[ "--gpu-architecture=compute_70" ];
  check "half arithmetic on a new device" ~device_cc:120 "__hfma(a, b, c)"
    ~expected:[ "--gpu-architecture=compute_75" ];
  (* The family marker: architecture-specific target on a family device... *)
  check "mxfp8 marker on sm_120" ~device_cc:120 "/* (mma-mxfp8) */"
    ~expected:[ "--gpu-architecture=compute_120a" ];
  check "mxfp8 marker on sm_121" ~device_cc:121 "/* (mma-mxfp8) */"
    ~expected:[ "--gpu-architecture=compute_121a" ];
  (* ...and nothing of the sort anywhere else: an sm_89 device gets the ordinary floor policy, so
     no unloadable PTX is ever produced for it. *)
  check "mxfp8 marker off-family falls back to the floor" ~device_cc:89
    "/* (mma-mxfp8) */ /* (mma-fp8) */" ~expected:[ "--gpu-architecture=compute_89" ];
  check "mxfp8 marker off-family with no floor" ~device_cc:89 "/* (mma-mxfp8) */" ~expected:[];
  (* The family target supersedes floors on a family device: family PTX is not forward-JIT PTX, and
     mixing the two targets in one compile is not expressible. *)
  check "mxfp8 marker supersedes a floor on-family" ~device_cc:120
    "/* (mma-mxfp8) */ /* (mma-fp8) */" ~expected:[ "--gpu-architecture=compute_120a" ];
  Stdio.print_endline "cuda gpu_arch_options: ok"
