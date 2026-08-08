// Measured local roofline for gfx1151 (Radeon 8060S iGPU, shared LPDDR5X), WSL2/ROCm 7.14.
// Three legs, deliberately mirroring the method in benchmarks/report-gh531-profile.md Part 3:
//   (a) fp32 GEMM throughput  -- rocBLAS sgemm 4096^3, the achievable-compute leg
//   (b) fp32 FMA issue peak   -- a dependency-free register FMA kernel, the instruction-issue
//                                upper bound (a GEMM can never beat it)
//   (c) device-to-device copy -- 256 MiB, the achievable-bandwidth leg
// On an APU the "device memory" is the same LPDDR5X the CPU uses, so (c) is a shared-controller
// number and is only meaningful with the CPU quiet.
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>

#define HC(x) do { hipError_t _hc = (x); if (_hc != hipSuccess) { \
  fprintf(stderr, "HIP error %s at %d: %s\n", #x, __LINE__, hipGetErrorString(_hc)); exit(1);} } while(0)
#define RC(x) do { rocblas_status s = (x); if (s != rocblas_status_success) { \
  fprintf(stderr, "rocBLAS error %s at %d: %d\n", #x, __LINE__, (int)s); exit(1);} } while(0)

// (b) 64 independent FMA chains per thread: enough ILP to saturate the FP pipes, no memory
// traffic in the loop, and the result is consumed so nothing is dead-code eliminated.
__global__ void fma_peak(float *out, int iters) {
  float a[64];
#pragma unroll
  for (int i = 0; i < 64; ++i) a[i] = (float)(threadIdx.x + i) * 1e-6f;
  float b = 1.0000001f, c = 1e-7f;
  for (int t = 0; t < iters; ++t) {
#pragma unroll
    for (int i = 0; i < 64; ++i) a[i] = fmaf(a[i], b, c);
  }
  float s = 0.f;
#pragma unroll
  for (int i = 0; i < 64; ++i) s += a[i];
  if (s == 12345.678f) out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}

__global__ void copy_kernel(const float4 *__restrict__ src, float4 *__restrict__ dst, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = (size_t)gridDim.x * blockDim.x;
  for (; i < n; i += stride) dst[i] = src[i];
}

static double time_ms(void (*fn)(void *), void *arg, int reps) {
  std::vector<double> ts;
  for (int r = 0; r < reps; ++r) {
    HC(hipDeviceSynchronize());
    hipEvent_t b, e; HC(hipEventCreate(&b)); HC(hipEventCreate(&e));
    HC(hipEventRecord(b));
    fn(arg);
    HC(hipEventRecord(e));
    HC(hipEventSynchronize(e));
    float ms; HC(hipEventElapsedTime(&ms, b, e));
    ts.push_back(ms);
    HC(hipEventDestroy(b)); HC(hipEventDestroy(e));
  }
  std::sort(ts.begin(), ts.end());
  return ts[ts.size() / 2];
}

struct GemmArg { rocblas_handle h; int n; float *A, *B, *C; float alpha, beta; int inner; };
static void run_gemm(void *p) {
  GemmArg *g = (GemmArg *)p;
  for (int k = 0; k < g->inner; ++k)
    RC(rocblas_sgemm(g->h, rocblas_operation_none, rocblas_operation_none, g->n, g->n, g->n,
                     &g->alpha, g->A, g->n, g->B, g->n, &g->beta, g->C, g->n));
}
struct FmaArg { int blocks, threads, iters; float *out; };
static void run_fma(void *p) {
  FmaArg *f = (FmaArg *)p;
  hipLaunchKernelGGL(fma_peak, dim3(f->blocks), dim3(f->threads), 0, 0, f->out, f->iters);
}
struct CopyArg { const float4 *s; float4 *d; size_t n; int blocks, threads, inner; };
static void run_copy(void *p) {
  CopyArg *c = (CopyArg *)p;
  for (int k = 0; k < c->inner; ++k)
    hipLaunchKernelGGL(copy_kernel, dim3(c->blocks), dim3(c->threads), 0, 0, c->s, c->d, c->n);
}

int main() {
  hipDeviceProp_t prop; HC(hipGetDeviceProperties(&prop, 0));
  printf("device: %s  CUs=%d  clock=%d MHz  gcn=%s\n",
         prop.name, prop.multiProcessorCount, prop.clockRate / 1000, prop.gcnArchName);

  // ---- (a) rocBLAS sgemm 4096^3
  {
    const int n = 4096;
    rocblas_handle h; RC(rocblas_create_handle(&h));
    float *A, *B, *C;
    HC(hipMalloc(&A, (size_t)n * n * sizeof(float)));
    HC(hipMalloc(&B, (size_t)n * n * sizeof(float)));
    HC(hipMalloc(&C, (size_t)n * n * sizeof(float)));
    std::vector<float> host((size_t)n * n);
    for (size_t i = 0; i < host.size(); ++i) host[i] = (float)((i % 97) - 48) * 0.01f;
    HC(hipMemcpy(A, host.data(), host.size() * sizeof(float), hipMemcpyHostToDevice));
    HC(hipMemcpy(B, host.data(), host.size() * sizeof(float), hipMemcpyHostToDevice));
    HC(hipMemset(C, 0, host.size() * sizeof(float)));
    GemmArg g{h, n, A, B, C, 1.0f, 0.0f, 5};
    run_gemm(&g); HC(hipDeviceSynchronize());          // warm up / autotune
    double ms = time_ms(run_gemm, &g, 5) / g.inner;
    double gflops = 2.0 * n * n * n / (ms * 1e6);
    printf("sgemm 4096^3 fp32 : %8.3f ms/call  ->  %8.1f GFLOP/s  (%.2f TFLOP/s)\n",
           ms, gflops, gflops / 1000.0);
    HC(hipFree(A)); HC(hipFree(B)); HC(hipFree(C)); RC(rocblas_destroy_handle(h));
  }

  // ---- (b) FMA issue peak
  {
    const int threads = 256, blocks = prop.multiProcessorCount * 8, iters = 4096;
    float *out; HC(hipMalloc(&out, (size_t)blocks * threads * sizeof(float)));
    FmaArg f{blocks, threads, iters, out};
    run_fma(&f); HC(hipDeviceSynchronize());
    double ms = time_ms(run_fma, &f, 7);
    double flops = 2.0 * 64.0 * iters * (double)blocks * threads;
    printf("fma issue peak    : %8.3f ms         ->  %8.1f GFLOP/s  (%.2f TFLOP/s)  [%d blk x %d thr]\n",
           ms, flops / (ms * 1e6), flops / (ms * 1e9), blocks, threads);
    HC(hipFree(out));
  }

  // ---- (c) device-to-device copy, 256 MiB each way
  {
    const size_t bytes = 256ull << 20;
    const size_t n4 = bytes / sizeof(float4);
    float4 *s, *d;
    HC(hipMalloc(&s, bytes)); HC(hipMalloc(&d, bytes));
    HC(hipMemset(s, 1, bytes));
    CopyArg c{s, d, n4, prop.multiProcessorCount * 16, 256, 5};
    run_copy(&c); HC(hipDeviceSynchronize());
    double ms = time_ms(run_copy, &c, 5) / c.inner;
    double gbs = 2.0 * bytes / (ms * 1e6);   // read + write
    printf("d2d copy 256 MiB  : %8.3f ms         ->  %8.1f GB/s  (read+write)\n", ms, gbs);
    HC(hipFree(s)); HC(hipFree(d));
  }
  return 0;
}
