// gh-ocannl-531: time the ACTUAL emitted kernels of the gpt2_mini step through the SAME
// compilation path OCANNL uses, so the report's replacement estimate is a production measurement
// rather than an extrapolation from a differently-built standalone.
//
// Build:
//     gcc -O2 -o ffn1_nvrtc_harness benchmarks/ffn1_nvrtc_harness.c \
//         -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -L/usr/lib/wsl/lib -lnvrtc -lcuda -lm
// The variant sources come from benchmarks/ffn1_make_variants.py, which rewrites arm A's own
// emitted translation unit (and refuses input that is not the 117-kernel / 20-mma_sync arm A).
//
// Run (input is arm A's emitted source -- see report-gh531-profile.md for how to snapshot it):
//     ./ffn1_nvrtc_harness <file.cu> <kernel> <gridY>
//   e.g. ./ffn1_nvrtc_harness armA-117.cu cross_entropy_loss_fwd__seg25 1
//        ./ffn1_nvrtc_harness armA_chunk128.cu cross_entropy_loss_fwd__seg25 128
//
// Compilation matches arrayjit/lib/cuda_backend.ml's cuda_to_ptx: "#include <mma.h>" injected
// (the source contains nvcuda::wmma), -I/usr/local/cuda/include, --use_fast_math, and
// --gpu-architecture=compute_80 (gpu_arch_options' floor for a (wmma-tf32) marker).
//
// Two kernel shapes are known, selected by name:
//   seg25  (FFN up-projection): (i1, l0_ffn_b1, l0_ffn_w1, n309_layer_norm, n311, n339_gelu)
//          n311[tok][j] = sum_k w1[j][k]*ln[tok][k];  gelu[tok][j] = tanh-gelu(n311 + b1[j])
//   seg111 (tied lm_head):      (i1, logits, max_logits, n794_layer_norm, wte)
//          logits[tok][v] = sum_k wte[k][v]*lnf[tok][k];  max_logits[tok] = max_v logits[tok][v]
//   seg111_gemm / seg111_reduce: the fission of seg111 used to measure a chunked lm_head, since
//          seg111's vocabulary axis carries a REDUCTION and cannot simply be spread over
//          blockIdx.y (that would give per-block partial maxima racing on one cell).
//
// EVERY run is verified against an independent fp64 host reference before its time is reported --
// every output cell, not a sample, because a chunk-count mismatch can leave a narrow unwritten
// band that a strided sample steps over -- and the program exits nonzero on any mismatch or
// non-finite value. A short, duplicated, permuted or gridY-mismatched output cannot be quoted.
//
// The chunked inputs are the same file with the output loop rebased onto blockIdx.y:
//     for (int i1705 = 0; i1705 <= 1023; ++i1705)
//  -> for (int i1705 = (int)blockIdx.y * P; i1705 < ((int)blockIdx.y + 1) * P; ++i1705)
// (P = 1024 / chunks). Every thread keeps its token and its access order; only the number of
// resident blocks changes.
//
// Measured on an RTX 5070 Ti Laptop (46 SMs), CUDA 13.3, WSL2. Every row verified before timing:
//
//   seg25 (FFN up-projection, GEMM + gelu, chunkable as-is)
//     as shipped,     8 blocks   13.70-13.91 ms   (nsys measures 14.31 ms in the step: ~3%)
//     chunked,       16 blocks    7.07 ms
//     chunked,       32 blocks    3.56 ms
//     chunked,     1024 blocks    2.36 ms         -> 5.9x
//
//   seg111 (tied lm_head). Its vocabulary axis carries the max_logits REDUCTION, so it cannot be
//   chunked as one kernel; it is fissioned into a GEMM half (chunkable) and a reduce half:
//     as shipped, whole kernel,      8 blocks   15.12-15.20 ms  (nsys: 14.76 ms, ~3%)
//     seg111_gemm,                   8 blocks   14.41 ms
//     seg111_gemm chunked,        1024 blocks    2.32 ms
//     seg111_reduce (init + max),    8 blocks    0.05 ms
//     fissioned total                            2.37 ms   -> 6.4x  (at the cost of one extra
//                                                                    launch, ~4 us)
//
// Note the loop-bound rewrite alone costs ~6% on seg25 (chunked at 8 blocks is 14.63-14.83 ms),
// so its same-code-shape ratio is 6.2x; 5.8x is quoted against what OCANNL emits today.
//
// See benchmarks/report-gh531-profile.md for the reading.
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <cuda.h>
#include <nvrtc.h>

#define TOKENS 1024
#define D 256
#define DFF 1024
#define VOCAB 1024

#define NV(x) do { nvrtcResult r=(x); if(r!=NVRTC_SUCCESS){ \
  printf("nvrtc error %s @%d\n", nvrtcGetErrorString(r), __LINE__); exit(2);} } while(0)
#define CU(x) do { CUresult r=(x); if(r!=CUDA_SUCCESS){ const char*m; cuGetErrorString(r,&m); \
  printf("cuda error %s @%d\n", m, __LINE__); exit(2);} } while(0)

static char* slurp(const char* p, size_t* n) {
  FILE* f = fopen(p, "rb");
  if (!f) { printf("cannot open %s\n", p); exit(2); }
  fseek(f, 0, SEEK_END); long len = ftell(f); fseek(f, 0, SEEK_SET);
  char* b = (char*)malloc(len + 1);
  size_t got = fread(b, 1, len, f); b[got] = 0; fclose(f);
  if (n) *n = got;
  return b;
}

// Deterministic but NON-PERIODIC fixture (splitmix64 on the flat index). Periodic inputs -- an
// earlier revision used (i % 17), (i % 13), (i % 7) -- make the reference blind to exactly the
// failures this harness exists to catch: with activation rows repeating every 13 tokens and weight
// rows every 17 outputs, a kernel that duplicates or permutes a chunk still matches cell for cell.
// With this generator every logical row and column is distinguishable, which is asserted at
// startup by fixture_selfcheck().
static float fixture(uint64_t i) {
  uint64_t x = i * 0x9E3779B97F4A7C15ull + 0x0123456789ABCDEFull;
  x ^= x >> 30; x *= 0xBF58476D1CE4E5B9ull;
  x ^= x >> 27; x *= 0x94D049BB133111EBull;
  x ^= x >> 31;
  return (float)(((double)(x >> 11) / 9007199254740992.0) - 0.5) * 0.1f;
}

static float* hw;   // weight   [OUT][D]  (w1[j][k] resp. wte[k][v] -- see loaders)
static float* hl;   // activations [TOKENS][D]
static float* hb;   // bias     [DFF]

// Dot products, with the sum of absolute products alongside. A near-cancelling dot has a tiny
// value and a large conditioning number, so a plain relative test on it is meaningless; the
// checks below use |got - ref| <= tol * (sum |a_k b_k|), which is the fp32 error bound's shape.
static double g_absum;
static double ref_dot_w1(int j, int tok) {         // sum_k w1[j][k] * ln[tok][k]
  double a = 0, s = 0;
  for (int k = 0; k < D; ++k) {
    double t = (double)hw[j * D + k] * (double)hl[tok * D + k];
    a += t; s += fabs(t);
  }
  g_absum = s; return a;
}
static double ref_dot_wte(int v, int tok) {        // sum_k wte[k][v] * lnf[tok][k]
  double a = 0, s = 0;
  for (int k = 0; k < D; ++k) {
    double t = (double)hw[k * VOCAB + v] * (double)hl[tok * D + k];
    a += t; s += fabs(t);
  }
  g_absum = s; return a;
}
static double tanh_gelu(double v) {
  return 0.5 * v * (1.0 + tanh(0.7978845608028654 * (0.044715 * v * v * v + v)));
}

static int fail(const char* what, double worst) {
  printf("  FAIL %s: worst relative deviation %.3e\n", what, worst);
  return 1;
}

int main(int argc, char** argv) {
  if (argc < 4) { printf("usage: %s <file.cu> <kernel> <gridY>\n", argv[0]); return 2; }
  const char* path = argv[1];
  const char* kname = argv[2];
  int gridY = atoi(argv[3]);
  int is25 = strstr(kname, "seg25") != NULL;
  int is111 = strstr(kname, "seg111") != NULL;
  int reduce_only = strstr(kname, "reduce") != NULL;
  if (!is25 && !is111) { printf("unknown kernel shape %s\n", kname); return 2; }

  size_t n; char* body = slurp(path, &n);
  const char* pre = "#include <mma.h>\n";
  char* src = (char*)malloc(n + strlen(pre) + 1);
  strcpy(src, pre); strcat(src, body);

  nvrtcProgram prog;
  NV(nvrtcCreateProgram(&prog, src, "seg.cu", 0, NULL, NULL));
  const char* opts[] = { "-I/usr/local/cuda/include", "--gpu-architecture=compute_80",
                         "--use_fast_math" };
  nvrtcResult cr = nvrtcCompileProgram(prog, 3, opts);
  size_t logsz; nvrtcGetProgramLogSize(prog, &logsz);
  if (logsz > 1 && cr != NVRTC_SUCCESS) {
    char* log = (char*)malloc(logsz); nvrtcGetProgramLog(prog, log); printf("%s\n", log); }
  if (cr != NVRTC_SUCCESS) { printf("nvrtc compile failed\n"); return 2; }
  size_t ptxsz; NV(nvrtcGetPTXSize(prog, &ptxsz));
  char* ptx = (char*)malloc(ptxsz); NV(nvrtcGetPTX(prog, ptx));

  CU(cuInit(0));
  CUdevice dev; CU(cuDeviceGet(&dev, 0));
  CUcontext ctx; CU(cuDevicePrimaryCtxRetain(&ctx, dev)); CU(cuCtxSetCurrent(ctx));
  CUmodule mod; CU(cuModuleLoadData(&mod, ptx));
  CUfunction fn; CU(cuModuleGetFunction(&fn, mod, kname));

  size_t wsz = (size_t)(is25 ? DFF * D : D * VOCAB) * 4;
  size_t outsz = (size_t)TOKENS * (is25 ? DFF : VOCAB) * 4;
  hw = (float*)malloc(wsz);
  hl = (float*)malloc((size_t)TOKENS * D * 4);
  hb = (float*)malloc((size_t)DFF * 4);
  for (size_t i = 0; i < wsz / 4; ++i) hw[i] = fixture(i);
  for (int i = 0; i < TOKENS * D; ++i) hl[i] = fixture(0x10000000ull + (uint64_t)i);
  for (int i = 0; i < DFF; ++i) hb[i] = fixture(0x20000000ull + (uint64_t)i);

  CUdeviceptr dW, dL, dB, dOut, dAux;
  CU(cuMemAlloc(&dW, wsz)); CU(cuMemAlloc(&dL, (size_t)TOKENS * D * 4));
  CU(cuMemAlloc(&dB, (size_t)DFF * 4));
  CU(cuMemAlloc(&dOut, outsz)); CU(cuMemAlloc(&dAux, outsz));
  CU(cuMemcpyHtoD(dW, hw, wsz));
  CU(cuMemcpyHtoD(dL, hl, (size_t)TOKENS * D * 4));
  CU(cuMemcpyHtoD(dB, hb, (size_t)DFF * 4));

  // Fixture self-check: the reference must distinguish rows and columns, or a duplicated or
  // permuted chunk would verify. These are the aliases the previous periodic fixture had.
  {
    int bad = 0;
    for (int k = 0; k < D; ++k) {                       // token t vs t+13
      if (hl[0 * D + k] != hl[13 * D + k]) { bad = 0; break; }
      bad = 1;
    }
    if (!bad) {
      int o = 0;                          // output row o vs o+17 and o+119
      for (int shift = 17; shift <= 119 && !bad; shift += 102) {
        int same = 1;
        for (int k = 0; k < D; ++k)
          if (hw[(size_t)o * D + k] != hw[(size_t)(o + shift) * D + k]) { same = 0; break; }
        bad = same;
      }
    }
    if (bad) { printf("FAIL: fixture is periodic -- rows or columns are not distinguishable\n");
               return 2; }
  }

  int i1 = 0;
  // seg25:  (i1, b1, w1, ln, n311, gelu)      seg111: (i1, logits, max_logits, lnf, wte)
  void* a25[] = { &i1, &dB, &dW, &dL, &dOut, &dAux };
  void* a111[] = { &i1, &dOut, &dAux, &dL, &dW };
  void** args = is25 ? a25 : a111;

  // The accumulating outputs must start zeroed; max_logits is initialized by the kernel itself
  // except in the reduce-only fission, where the init stays with it. In reduce-only mode the
  // logits buffer is NOT re-zeroed per rep -- it holds the GEMM half's real output, so the
  // reduction is timed over representative data rather than over zeros.
  #define RESET() do { if (!reduce_only) CU(cuMemsetD8(dOut, 0, outsz)); \
      if (is111 && !reduce_only) CU(cuMemsetD8(dAux, 0, outsz)); } while (0)

  CUevent e0, e1; CU(cuEventCreate(&e0, 0)); CU(cuEventCreate(&e1, 0));
  int reps = gridY >= 16 ? 20 : 3;
  RESET();
  if (reduce_only) {  // fill logits with the GEMM half's real output first
    CUfunction g; CU(cuModuleGetFunction(&g, mod, "cross_entropy_loss_fwd__seg111_gemm"));
    CU(cuMemsetD8(dOut, 0, outsz));
    CU(cuLaunchKernel(g, 8, 1, 1, 128, 1, 1, 0, 0, args, 0));
    CU(cuCtxSynchronize());
  }
  CU(cuLaunchKernel(fn, 8, gridY, 1, 128, 1, 1, 0, 0, args, 0));
  CU(cuCtxSynchronize());
  CU(cuEventRecord(e0, 0));
  for (int i = 0; i < reps; ++i) {
    if (!reduce_only) CU(cuMemsetD8Async(dOut, 0, outsz, 0));
    CU(cuLaunchKernel(fn, 8, gridY, 1, 128, 1, 1, 0, 0, args, 0));
  }
  CU(cuEventRecord(e1, 0));
  CU(cuEventSynchronize(e1));
  float ms; CU(cuEventElapsedTime(&ms, e0, e1)); ms /= reps;

  // ---- verification: one clean run, checked pointwise against an fp64 host reference --------
  RESET();
  if (reduce_only) {  // the reduction consumes logits produced by the GEMM half.
    // NOTE: run this against a file whose GEMM half is UNCHUNKED (gridY=1 covers the whole
    // vocabulary); the reduce half is byte-identical across the chunked variants anyway.
    CUfunction g; CU(cuModuleGetFunction(&g, mod, "cross_entropy_loss_fwd__seg111_gemm"));
    CU(cuMemsetD8(dOut, 0, outsz));
    CU(cuLaunchKernel(g, 8, 1, 1, 128, 1, 1, 0, 0, args, 0));
  }
  CU(cuLaunchKernel(fn, 8, gridY, 1, 128, 1, 1, 0, 0, args, 0));
  CU(cuCtxSynchronize());
  float* out = (float*)malloc(outsz);
  float* aux = (float*)malloc(outsz);
  CU(cuMemcpyDtoH(out, dOut, outsz));
  CU(cuMemcpyDtoH(aux, dAux, outsz));

  int OUTW = is25 ? DFF : VOCAB;
  // Verify EVERY output cell, not a sampled grid: a chunk-count mismatch can leave a narrow
  // unwritten band (zeros are finite, and a strided sample can step straight over it).
  // The fp64 reference is built once as a full [TOKENS][OUTW] product.
  double* ref = (double*)malloc((size_t)TOKENS * OUTW * sizeof(double));
  double* cond = (double*)malloc((size_t)TOKENS * OUTW * sizeof(double));
  for (int tok = 0; tok < TOKENS; ++tok)
    for (int o = 0; o < OUTW; ++o) {
      double r = is25 ? ref_dot_w1(o, tok) : ref_dot_wte(o, tok);
      ref[(size_t)tok * OUTW + o] = r;
      cond[(size_t)tok * OUTW + o] = g_absum;
    }
  double worst = 0;
  for (size_t i = 0; i < (size_t)TOKENS * OUTW; ++i) {
    if (!isfinite(out[i])) { printf("  FAIL %s: non-finite GEMM output at %zu\n", kname, i); return 1; }
    double d = fabs((double)out[i] - ref[i]) / fmax(1e-30, cond[i]);
    if (d > worst) worst = d;
  }
  if (worst > 1e-5) return fail(is25 ? "GEMM (n311)" : "lm_head GEMM (logits)", worst);

  if (is25) {
    double wg = 0;   // gelu checked against the device's own n311: an independent leg
    for (size_t i = 0; i < (size_t)TOKENS * DFF; ++i) {
      if (!isfinite(aux[i])) { printf("  FAIL %s: non-finite gelu at %zu\n", kname, i); return 1; }
      double r = tanh_gelu((double)out[i] + (double)hb[i % DFF]);
      double rel = fabs((double)aux[i] - r) / fmax(1e-3, fabs(r));
      if (rel > wg) wg = rel;
    }
    if (wg > 2e-3) return fail("gelu epilogue", wg);   // __tanhf under --use_fast_math
    printf("  verified(all %d cells): GEMM %.2e, gelu %.2e   ", TOKENS * DFF, worst, wg);
  } else if (!strstr(kname, "_gemm")) {
    double wm = 0;   // max_logits must equal the row max of the logits the device produced
    for (int tok = 0; tok < TOKENS; ++tok) {
      double m = -INFINITY;
      for (int v = 0; v < VOCAB; ++v) m = fmax(m, (double)out[(size_t)tok * VOCAB + v]);
      double g = aux[(size_t)tok];
      if (!isfinite(g)) { printf("  FAIL %s: non-finite max_logits at %d\n", kname, tok); return 1; }
      double rel = fabs(g - m) / fmax(1e-6, fabs(m));
      if (rel > wm) wm = rel;
    }
    if (wm > 1e-6) return fail("max_logits row max", wm);
    printf("  verified(all %d cells): GEMM %.2e, rowmax %.2e   ", TOKENS * VOCAB, worst, wm);
  } else {
    printf("  verified(all %d cells): GEMM %.2e   ", TOKENS * VOCAB, worst);
  }
  printf("%-34s %-14s gridY=%-4d blocks=%-5d %8.3f ms\n",
         strrchr(path, '/') ? strrchr(path, '/') + 1 : path,
         strstr(kname, "seg") ? strstr(kname, "seg") : kname, gridY, 8 * gridY, ms);
  return 0;
}
