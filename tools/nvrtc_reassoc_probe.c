/* Does nvrtc's fast math reassociate reductions, and can anything stop it?
 * gh-ocannl-784; the recorded verdict lives in arrayjit/lib/compiler_options.ml
 * and docs/agent-notes/backend-precision-and-simd.md.
 *
 * Standalone on purpose: it links nvrtc and the CUDA driver directly, so it
 * answers a question about the TOOLKIT rather than about OCANNL, and it stays
 * runnable on a box where the OCaml switch is not set up.  Build and run:
 *
 *   gcc -O0 -o /tmp/nvrtc_reassoc_probe tools/nvrtc_reassoc_probe.c \
 *     -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -L/usr/lib/wsl/lib \
 *     -lnvrtc -lcuda
 *   LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/wsl/lib /tmp/nvrtc_reassoc_probe
 *
 * (Drop the WSL library path off WSL.)  Re-run it after a CUDA toolkit upgrade,
 * BEFORE suspecting anything else about a `reduction_forms` red on CUDA.
 *
 * How it decides.  A 128-element float sum whose strictly-sequential IEEE value
 * is exactly 1.0f -- each 3e-8f addend falls below half an ulp of 1.0f, so every
 * partial sum rounds back to 1.0f -- but whose value under ANY grouping that adds
 * the small addends to each other first exceeds 1.0f.  The data lives in device
 * memory, so nothing is constant-folded; the kernel is compiled by nvrtc, loaded
 * through the driver API, and EXECUTED.  A result that is not bit-exactly 1.0f
 * proves the compiler reassociated.
 *
 * Three kernels, because gh-ocannl-735's hiprtc reassociation appeared in some
 * spellings and not others: a counted loop, a runtime-bound loop, and 128
 * repeated statements.  Plus `cancel`, an (a+b)-a probe that a compiler allowed
 * to reassociate folds to b.
 *
 * The detector is not blind, and that is checkable without a GPU: the same
 * reduction built by host gcc at -O3 -ffast-math DOES reassociate (returning
 * 1.00000286 here), and -fno-associative-math restores the sequential value.
 *
 * The tail of the run asks a different question -- not what the options DO, but
 * which of them nvrtc will even accept.  Every clang-shaped reassociation guard
 * is rejected as an unrecognized option; `--fassociative-math` is not, but it is
 * a bare opt-in flag with no negative spelling and no entry in NVIDIA's nvrtc
 * option reference.  That asymmetry is why `Compiler_options.nvrtc`'s test claims
 * MEMBERSHIP (the vector never contains the opt-in) where HIP's claims ordering.
 */

#include <cuda.h>
#include <nvrtc.h>
#include <stdio.h>
#include <string.h>

#define N 128

static const char *kernel_src =
    "extern \"C\" __global__ void unrolled(const float *in, float *out) {\n"
    "  float acc = 0.f;\n"
    "#pragma unroll\n"
    "  for (int i = 0; i < 128; ++i) acc += in[i];\n"
    "  out[0] = acc;\n"
    "}\n"
    "extern \"C\" __global__ void runtime_n(const float *in, int n, float *out) {\n"
    "  float acc = 0.f;\n"
    "  for (int i = 0; i < n; ++i) acc += in[i];\n"
    "  out[0] = acc;\n"
    "}\n"
    "#define S4(i) acc += in[i]; acc += in[i+1]; acc += in[i+2]; acc += in[i+3];\n"
    "#define S16(i) S4(i) S4(i+4) S4(i+8) S4(i+12)\n"
    "extern \"C\" __global__ void repeated(const float *in, float *out) {\n"
    "  float acc = 0.f;\n"
    "  S16(0) S16(16) S16(32) S16(48) S16(64) S16(80) S16(96) S16(112)\n"
    "  out[0] = acc;\n"
    "}\n"
    /* Classic associativity probe: (a+b)-a is b only if the compiler may reassociate. */
    "extern \"C\" __global__ void cancel(const float *in, float *out) {\n"
    "  float a = in[0], b = in[1];\n"
    "  out[0] = (a + b) - a;\n"
    "}\n"
    /* Distribution probe: a*b + a*c == a*(b+c) only under reassociation. */
    "extern \"C\" __global__ void distrib(const float *in, float *out) {\n"
    "  float a = in[0], b = in[1], c = in[2];\n"
    "  out[0] = a * b + a * c;\n"
    "}\n";

#define CHECK(x)                                                                                   \
  do {                                                                                             \
    CUresult r_ = (x);                                                                             \
    if (r_ != CUDA_SUCCESS) {                                                                      \
      const char *m_ = 0;                                                                          \
      cuGetErrorString(r_, &m_);                                                                   \
      fprintf(stderr, "CUDA error at %d: %s\n", __LINE__, m_ ? m_ : "?");                          \
      return 2;                                                                                    \
    }                                                                                              \
  } while (0)

static CUdeviceptr d_in, d_out, d_cancel;

/* Returns 0 on a successful compile, 1 if nvrtc rejected the options. */
static int compile(const char **opts, int nopts, char **ptx_out, char **log_out) {
  nvrtcProgram prog;
  nvrtcResult rc = nvrtcCreateProgram(&prog, kernel_src, "probe.cu", 0, 0, 0);
  if (rc != NVRTC_SUCCESS) {
    fprintf(stderr, "nvrtcCreateProgram: %s\n", nvrtcGetErrorString(rc));
    return 3;
  }
  rc = nvrtcCompileProgram(prog, nopts, opts);
  size_t logsz = 0;
  nvrtcGetProgramLogSize(prog, &logsz);
  char *log = (char *)malloc(logsz + 1);
  nvrtcGetProgramLog(prog, log);
  log[logsz] = 0;
  *log_out = log;
  if (rc != NVRTC_SUCCESS) {
    nvrtcDestroyProgram(&prog);
    *ptx_out = 0;
    return 1;
  }
  size_t ptxsz = 0;
  nvrtcGetPTXSize(prog, &ptxsz);
  char *ptx = (char *)malloc(ptxsz + 1);
  nvrtcGetPTX(prog, ptx);
  ptx[ptxsz] = 0;
  nvrtcDestroyProgram(&prog);
  *ptx_out = ptx;
  return 0;
}

static int count_substr(const char *hay, const char *needle) {
  int n = 0;
  size_t l = strlen(needle);
  for (const char *p = hay; (p = strstr(p, needle)); p += l) n++;
  return n;
}

static int run_case(const char *label, const char **opts, int nopts) {
  char *ptx = 0, *log = 0;
  int st = compile(opts, nopts, &ptx, &log);
  printf("== %s\n   options:", label);
  for (int i = 0; i < nopts; i++) printf(" %s", opts[i]);
  printf("\n");
  if (st == 1) {
    printf("   REJECTED by nvrtc: %s\n", log[0] ? log : "(empty log)");
    free(log);
    return 0;
  }
  if (st) {
    free(log);
    return st;
  }
  printf("   accepted; PTX add.f32=%d add.ftz.f32=%d add.rn.f32=%d fma.rn.f32=%d\n",
         count_substr(ptx, "add.f32"), count_substr(ptx, "add.ftz.f32"),
         count_substr(ptx, "add.rn.f32"), count_substr(ptx, "fma.rn.f32"));

  CUmodule mod;
  CHECK(cuModuleLoadData(&mod, ptx));
  const char *names[3] = { "unrolled", "runtime_n", "repeated" };
  {
    /* cancel: strict IEEE gives 0; reassociated gives b. */
    CUfunction f;
    CHECK(cuModuleGetFunction(&f, mod, "cancel"));
    float zero = 0.f;
    CHECK(cuMemcpyHtoD(d_out, &zero, sizeof(float)));
    void *a2[2] = { &d_cancel, &d_out };
    CHECK(cuLaunchKernel(f, 1, 1, 1, 1, 1, 1, 0, 0, a2, 0));
    CHECK(cuCtxSynchronize());
    float got = -1.f;
    CHECK(cuMemcpyDtoH(&got, d_out, sizeof(float)));
    printf("   %-10s -> %.9g %s\n", "cancel", (double)got,
           got == 0.f ? "strict ((a+b)-a == 0)" : "REASSOCIATED ((a+b)-a == b)");
  }
  for (int k = 0; k < 3; k++) {
    CUfunction f;
    CHECK(cuModuleGetFunction(&f, mod, names[k]));
    float zero = 0.f;
    CHECK(cuMemcpyHtoD(d_out, &zero, sizeof(float)));
    int n = N;
    void *args_plain[2] = { &d_in, &d_out };
    void *args_n[3] = { &d_in, &n, &d_out };
    CHECK(cuLaunchKernel(f, 1, 1, 1, 1, 1, 1, 0, 0, k == 1 ? args_n : args_plain, 0));
    CHECK(cuCtxSynchronize());
    float got = -1.f;
    CHECK(cuMemcpyDtoH(&got, d_out, sizeof(float)));
    unsigned int bits;
    memcpy(&bits, &got, sizeof(bits));
    printf("   %-10s -> %.9g (0x%08x) %s\n", names[k], (double)got, bits,
           got == 1.0f ? "sequential-order (no reassociation)" : "REASSOCIATED");
  }
  CHECK(cuModuleUnload(mod));
  free(ptx);
  free(log);
  return 0;
}

int main(void) {
  int major = 0, minor = 0;
  nvrtcVersion(&major, &minor);
  printf("nvrtc version: %d.%d\n", major, minor);

  CHECK(cuInit(0));
  CUdevice dev;
  CHECK(cuDeviceGet(&dev, 0));
  char devname[128];
  CHECK(cuDeviceGetName(devname, sizeof(devname), dev));
  int ccmaj = 0, ccmin = 0;
  CHECK(cuDeviceGetAttribute(&ccmaj, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev));
  CHECK(cuDeviceGetAttribute(&ccmin, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev));
  printf("device: %s (sm_%d%d)\n", devname, ccmaj, ccmin);
  int drv = 0;
  cuDriverGetVersion(&drv);
  printf("driver version: %d\n\n", drv);
  CUcontext ctx;
  CHECK(cuDevicePrimaryCtxRetain(&ctx, dev));
  CHECK(cuCtxSetCurrent(ctx));

  float host[N];
  host[0] = 1.0f;
  for (int i = 1; i < N; i++) host[i] = 3e-8f;
  float seq = 0.f;
  for (int i = 0; i < N; i++) seq += host[i];
  printf("host sequential float sum = %.9g (expected exactly 1)\n", (double)seq);
  double exact = 0.0;
  for (int i = 0; i < N; i++) exact += (double)host[i];
  printf("exact (double) sum        = %.9g\n\n", exact);

  CHECK(cuMemAlloc(&d_in, sizeof(host)));
  CHECK(cuMemAlloc(&d_out, sizeof(float)));
  CHECK(cuMemcpyHtoD(d_in, host, sizeof(host)));
  float cancel_in[3] = { 1e10f, 1.0f, 2.0f };
  CHECK(cuMemAlloc(&d_cancel, sizeof(cancel_in)));
  CHECK(cuMemcpyHtoD(d_cancel, cancel_in, sizeof(cancel_in)));

  char arch[64];
  snprintf(arch, sizeof(arch), "--gpu-architecture=compute_%d%d", ccmaj, ccmin);

  {
    const char *o[] = { arch };
    if (run_case("baseline (arch only)", o, 1)) return 2;
  }
  {
    const char *o[] = { arch, "--use_fast_math" };
    if (run_case("OCANNL today: --use_fast_math", o, 2)) return 2;
  }
  {
    const char *o[] = { arch, "--use_fast_math", "--extra-device-vectorization" };
    if (run_case("--use_fast_math + --extra-device-vectorization", o, 3)) return 2;
  }
  {
    const char *o[] = { arch, "--use_fast_math", "--dopt=on", "--ptxas-options=-O3" };
    if (run_case("--use_fast_math + max opt", o, 4)) return 2;
  }
  {
    const char *o[] = { arch, "--use_fast_math", "--fmad=false" };
    if (run_case("--use_fast_math + --fmad=false", o, 3)) return 2;
  }

  {
    const char *o[] = { arch, "--fassociative-math" };
    if (run_case("opt-in: --fassociative-math alone", o, 2)) return 2;
  }
  {
    const char *o[] = { arch, "--use_fast_math", "--fassociative-math" };
    if (run_case("opt-in: --use_fast_math + --fassociative-math", o, 3)) return 2;
  }
  {
    const char *o[] = { arch, "--fassociative-math", "--extra-device-vectorization" };
    if (run_case("opt-in: --fassociative-math + vectorization", o, 3)) return 2;
  }
  /* Not "does it help" but "is it even a word": which of the guards a clang user
     would reach for does nvrtc accept? */
  static const char *guards[] = {
    "-fno-associative-math", "--fno-associative-math", "-fno-unsafe-math-optimizations",
    "-ffp-contract=off",     "--fp-contract=off",      "-fno-fast-math",
    "--no-fast-math",        "--prec-reassoc=false",   "--fassociative-math=false",
    "-fno-reassociate",      "--allow-reassociation=false",
  };
  for (unsigned i = 0; i < sizeof(guards) / sizeof(*guards); i++) {
    const char *o[] = { arch, "--use_fast_math", guards[i] };
    char label[128];
    snprintf(label, sizeof(label), "guard candidate %s", guards[i]);
    if (run_case(label, o, 3)) return 2;
  }
  return 0;
}
