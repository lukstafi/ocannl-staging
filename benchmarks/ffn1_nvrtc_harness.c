// gh-ocannl-531: time the ACTUAL emitted FFN up-projection kernel through the SAME compilation
// path OCANNL uses, so the report's replacement estimate is a production measurement rather than
// an extrapolation from a differently-built standalone.
//
// Build:
//     gcc -O2 -o ffn1_nvrtc_harness benchmarks/ffn1_nvrtc_harness.c \
//         -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -L/usr/lib/wsl/lib -lnvrtc -lcuda
// Run (input is arm A's emitted source -- see report-gh531-profile.md for how to snapshot it):
//     ./ffn1_nvrtc_harness armA-117.cu 1        # as shipped: grid=(8,1), 8 blocks
//     ./ffn1_nvrtc_harness armA_chunk128.cu 128 # j range split across blockIdx.y
//
// Compilation matches arrayjit/lib/cuda_backend.ml's cuda_to_ptx: "#include <mma.h>" injected
// (the source contains nvcuda::wmma), -I/usr/local/cuda/include, --use_fast_math, and
// --gpu-architecture=compute_80 (gpu_arch_options' floor for a (wmma-tf32) marker).
//
// The chunked inputs are the same file with seg25's two output loops rebased onto blockIdx.y:
//     for (int i1705 = 0; i1705 <= 1023; ++i1705)
//  -> for (int i1705 = (int)blockIdx.y * P; i1705 < ((int)blockIdx.y + 1) * P; ++i1705)
// (P = 1024 / chunks; same substitution for the gelu loop i1736). Every thread keeps its token
// and its access order; only the number of resident blocks changes.
//
// Measured on an RTX 5070 Ti Laptop (46 SMs), CUDA 13.3, WSL2 -- checksums identical throughout:
//     as shipped, 8 blocks        13.86-13.97 ms   (nsys measures 14.31 ms in the step: 2.8%)
//     chunked,   16 blocks         7.24 ms
//     chunked,   32 blocks         3.50 ms
//     chunked,  128 blocks         2.47 ms
//     chunked, 1024 blocks         2.37-2.40 ms    -> 5.8x
// Note the loop-bound rewrite alone costs ~6% (chunk1 at 8 blocks is 14.63-14.83 ms), so the
// same-code-shape ratio is 6.2x; 5.8x is quoted against what OCANNL emits today.
//
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include <nvrtc.h>

#define TOKENS 1024
#define D 256
#define DFF 1024

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

int main(int argc, char** argv) {
  if (argc < 3) { printf("usage: %s <file.cu> <gridY>\n", argv[0]); return 2; }
  int gridY = atoi(argv[2]);
  size_t n; char* body = slurp(argv[1], &n);
  const char* pre = "#include <mma.h>\n";
  char* src = (char*)malloc(n + strlen(pre) + 1);
  strcpy(src, pre); strcat(src, body);

  nvrtcProgram prog;
  NV(nvrtcCreateProgram(&prog, src, "seg.cu", 0, NULL, NULL));
  const char* opts[] = { "-I/usr/local/cuda/include",
                         "--gpu-architecture=compute_80",
                         "--use_fast_math" };
  nvrtcResult cr = nvrtcCompileProgram(prog, 3, opts);
  size_t logsz; nvrtcGetProgramLogSize(prog, &logsz);
  if (logsz > 1) { char* log = (char*)malloc(logsz); nvrtcGetProgramLog(prog, log);
                   if (cr != NVRTC_SUCCESS) printf("%s\n", log); free(log); }
  if (cr != NVRTC_SUCCESS) { printf("nvrtc compile failed\n"); return 2; }
  size_t ptxsz; NV(nvrtcGetPTXSize(prog, &ptxsz));
  char* ptx = (char*)malloc(ptxsz); NV(nvrtcGetPTX(prog, ptx));

  CU(cuInit(0));
  CUdevice dev; CU(cuDeviceGet(&dev, 0));
  CUcontext ctx; CU(cuDevicePrimaryCtxRetain(&ctx, dev)); CU(cuCtxSetCurrent(ctx));
  CUmodule mod; CU(cuModuleLoadData(&mod, ptx));
  CUfunction fn; CU(cuModuleGetFunction(&fn, mod, "cross_entropy_loss_fwd__seg25"));

  CUdeviceptr b1, w1, ln, n311, gelu;
  CU(cuMemAlloc(&b1, (size_t)DFF * 4));
  CU(cuMemAlloc(&w1, (size_t)DFF * D * 4));
  CU(cuMemAlloc(&ln, (size_t)TOKENS * D * 4));
  CU(cuMemAlloc(&n311, (size_t)TOKENS * DFF * 4));
  CU(cuMemAlloc(&gelu, (size_t)TOKENS * DFF * 4));
  float* h = (float*)malloc((size_t)DFF * D * 4);
  for (int i = 0; i < DFF * D; ++i) h[i] = (float)((i % 17) - 8) * 0.01f;
  CU(cuMemcpyHtoD(w1, h, (size_t)DFF * D * 4));
  for (int i = 0; i < TOKENS * D; ++i) h[i] = (float)((i % 13) - 6) * 0.01f;
  CU(cuMemcpyHtoD(ln, h, (size_t)TOKENS * D * 4));
  for (int i = 0; i < DFF; ++i) h[i] = 0.001f * (float)(i % 7);
  CU(cuMemcpyHtoD(b1, h, (size_t)DFF * 4));

  int i1 = 0;
  void* args[] = { &i1, &b1, &w1, &ln, &n311, &gelu };

  CUevent e0, e1; CU(cuEventCreate(&e0, 0)); CU(cuEventCreate(&e1, 0));
  int reps = gridY >= 16 ? 20 : 3;
  // warmup
  CU(cuMemsetD8(n311, 0, (size_t)TOKENS * DFF * 4));
  CU(cuLaunchKernel(fn, 8, gridY, 1, 128, 1, 1, 0, 0, args, 0));
  CU(cuCtxSynchronize());
  CU(cuEventRecord(e0, 0));
  for (int i = 0; i < reps; ++i) {
    CU(cuMemsetD8Async(n311, 0, (size_t)TOKENS * DFF * 4, 0));
    CU(cuLaunchKernel(fn, 8, gridY, 1, 128, 1, 1, 0, 0, args, 0));
  }
  CU(cuEventRecord(e1, 0));
  CU(cuEventSynchronize(e1));
  float ms; CU(cuEventElapsedTime(&ms, e0, e1)); ms /= reps;

  // checksum so a miscompiled/short kernel cannot pass unnoticed
  float* out = (float*)malloc((size_t)TOKENS * DFF * 4);
  CU(cuMemcpyDtoH(out, gelu, (size_t)TOKENS * DFF * 4));
  double sum = 0; for (size_t i = 0; i < (size_t)TOKENS * DFF; ++i) sum += out[i];
  printf("%-42s gridY=%-4d blocks=%-5d %8.3f ms   checksum %.6e\n",
         argv[1], gridY, 8 * gridY, ms, sum);
  return 0;
}
