// gh-ocannl-531: same-geometry control for the gpt2_mini step's dominant kernel.
//
// Standalone, no OCANNL. Build and run:
//     nvcc -O3 -arch=sm_120 -o ffn1_geometry_probe benchmarks/ffn1_geometry_probe.cu
//     ./ffn1_geometry_probe
//
// Why it exists: 4 of the 5 kernels that make up 70% of the tuned gpt2_mini step are the FFN
// up-projection, emitted as two serial loops with a global-memory accumulator at
// grid=(8,1,1) x block=(128,1,1). The report needed to know which property is actually binding
// -- the read-modify-write, or the 1024-thread launch -- and `ncu`'s traffic counters are
// unavailable under WSL without admin (ERR_NVGPUCTRPERM). So vary one thing at a time instead.
//
// Measured on an RTX 5070 Ti Laptop (46 SMs, sm_120), CUDA 13.3, WSL2:
//     A  global RMW,   grid=8    block=128   12.69 ms    42.3 GFLOP/s
//     B  register acc, grid=8    block=128   12.63 ms    42.5 GFLOP/s
//     C  register acc, grid=4096 block=256    2.28 ms   235.7 GFLOP/s
//     D  global RMW,   grid=4096 block=256    2.26 ms   237.8 GFLOP/s
// A reproduces the in-step kernel (14.31 ms there, the difference being its gelu epilogue).
// Removing the RMW at the shipped geometry changes nothing; widening the grid is worth 5.6x
// with the RMW still in place. The binding resource is occupancy, not memory traffic.
//
// A: byte-for-byte the loop nest OCANNL's shipping tuned artifact emits for
//    cross_entropy_loss_fwd__seg25 (global-memory RMW accumulator), same launch geometry.
// B: identical loops, identical reads, register accumulator (one store per output element).
// C: B, but with the grid widened to fill the GPU (occupancy control).
// D: A, but with the grid widened to fill the GPU (does occupancy alone fix the RMW form?).
//
// All four compute the same thing: out[tok][j] = sum_k w1[j][k] * ln[tok][k].
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define TOKENS 1024   // 8 batch * 128 seq
#define D 256
#define DFF 1024

#define CHECK(x) do { cudaError_t e=(x); if(e){printf("CUDA error %s @%d\n",cudaGetErrorString(e),__LINE__);exit(1);} } while(0)

// --- A: exactly the emitted form: grid=(8), block=(128), global RMW ---------
__global__ void A_global_rmw(const float* __restrict__ w1, const float* __restrict__ ln,
                             float* __restrict__ out) {
  const int b = (int)blockIdx.x;
  const int t = (int)threadIdx.x;
  for (int j = 0; j <= DFF - 1; ++j)
    for (int k = 0; k <= D - 1; ++k)
      out[((b)*128 + t) * DFF + j] =
          fmaf(w1[(j)*D + k], ln[((b)*128 + t) * D + k], out[((b)*128 + t) * DFF + j]);
}

// --- B: same geometry, same reads, register accumulator ---------------------
__global__ void B_reg_acc(const float* __restrict__ w1, const float* __restrict__ ln,
                          float* __restrict__ out) {
  const int b = (int)blockIdx.x;
  const int t = (int)threadIdx.x;
  for (int j = 0; j <= DFF - 1; ++j) {
    float acc = 0.f;
    for (int k = 0; k <= D - 1; ++k)
      acc = fmaf(w1[(j)*D + k], ln[((b)*128 + t) * D + k], acc);
    out[((b)*128 + t) * DFF + j] = acc;
  }
}

// --- C: register accumulator, one thread per (token, j) — fills the GPU -----
__global__ void C_reg_acc_wide(const float* __restrict__ w1, const float* __restrict__ ln,
                               float* __restrict__ out) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;   // over TOKENS*DFF
  if (idx >= TOKENS * DFF) return;
  int tok = idx / DFF, j = idx % DFF;
  float acc = 0.f;
  for (int k = 0; k < D; ++k) acc = fmaf(w1[j * D + k], ln[tok * D + k], acc);
  out[tok * DFF + j] = acc;
}

// --- D: global RMW, one thread per (token, j) — occupancy without registers -
__global__ void D_global_rmw_wide(const float* __restrict__ w1, const float* __restrict__ ln,
                                  float* __restrict__ out) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= TOKENS * DFF) return;
  int tok = idx / DFF, j = idx % DFF;
  for (int k = 0; k < D; ++k)
    out[tok * DFF + j] = fmaf(w1[j * D + k], ln[tok * D + k], out[tok * DFF + j]);
}

float bench(void (*launch)(const float*, const float*, float*), const char* name,
            const float* w1, const float* ln, float* out, int reps) {
  cudaMemset(out, 0, (size_t)TOKENS * DFF * sizeof(float));
  launch(w1, ln, out);  // warmup
  CHECK(cudaDeviceSynchronize());
  cudaEvent_t ev0, ev1; cudaEventCreate(&ev0); cudaEventCreate(&ev1);
  cudaMemset(out, 0, (size_t)TOKENS * DFF * sizeof(float));
  cudaEventRecord(ev0);
  for (int i = 0; i < reps; ++i) launch(w1, ln, out);
  cudaEventRecord(ev1);
  CHECK(cudaEventSynchronize(ev1));
  float ms; cudaEventElapsedTime(&ms, ev0, ev1);
  return ms / reps;
}

static void la(const float* w, const float* l, float* o) { A_global_rmw<<<8, 128>>>(w, l, o); }
static void lb(const float* w, const float* l, float* o) { B_reg_acc<<<8, 128>>>(w, l, o); }
static void lc(const float* w, const float* l, float* o) {
  C_reg_acc_wide<<<(TOKENS * DFF + 255) / 256, 256>>>(w, l, o);
}
static void ld(const float* w, const float* l, float* o) {
  D_global_rmw_wide<<<(TOKENS * DFF + 255) / 256, 256>>>(w, l, o);
}

int main() {
  float *w1, *ln, *out, *ref;
  // Plain device memory: managed memory without prefetch faults per page and makes the
  // wide-grid variants look absurdly slow (measured once; do not use it here).
  float* hw = (float*)malloc((size_t)DFF * D * sizeof(float));
  float* hl = (float*)malloc((size_t)TOKENS * D * sizeof(float));
  for (int i = 0; i < DFF * D; ++i) hw[i] = (float)((i % 17) - 8) * 0.01f;
  for (int i = 0; i < TOKENS * D; ++i) hl[i] = (float)((i % 13) - 6) * 0.01f;
  CHECK(cudaMalloc(&w1, (size_t)DFF * D * sizeof(float)));
  CHECK(cudaMalloc(&ln, (size_t)TOKENS * D * sizeof(float)));
  CHECK(cudaMalloc(&out, (size_t)TOKENS * DFF * sizeof(float)));
  CHECK(cudaMemcpy(w1, hw, (size_t)DFF * D * sizeof(float), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(ln, hl, (size_t)TOKENS * D * sizeof(float), cudaMemcpyHostToDevice));
  ref = (float*)malloc((size_t)TOKENS * DFF * sizeof(float));

  cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
  printf("device: %s, %d SMs, cc %d.%d\n", p.name, p.multiProcessorCount, p.major, p.minor);
  double flops = 2.0 * TOKENS * D * DFF;
  printf("GEMM: [%d tokens x %d] x [%d x %d], %.1f MFLOP\n\n", TOKENS, D, D, DFF, flops / 1e6);

  struct { const char* n; void (*f)(const float*, const float*, float*); int threads; } ks[] = {
      {"A  global RMW,  grid=8   block=128  (as shipped)", la, 1024},
      {"B  register acc, grid=8   block=128 (same geometry)", lb, 1024},
      {"C  register acc, grid=4096 block=256 (fills GPU)", lc, 1048576},
      {"D  global RMW,  grid=4096 block=256 (fills GPU)", ld, 1048576},
  };
  printf("%-52s %10s %12s %9s\n", "variant", "ms", "GFLOP/s", "threads");
  float base = 0;
  for (int i = 0; i < 4; ++i) {
    int reps = (i == 0 || i == 1) ? 3 : 20;
    float ms = bench(ks[i].f, ks[i].n, w1, ln, out, reps);
    if (i == 0) base = ms;
    printf("%-52s %10.3f %12.1f %9d   %.2fx\n", ks[i].n, ms, flops / (ms / 1e3) / 1e9,
           ks[i].threads, base / ms);
  }
  // Correctness, separately: the RMW variants accumulate, so one zeroed invocation each.
  float* got = (float*)malloc((size_t)TOKENS * DFF * sizeof(float));
  double worst = 0;
  for (int i = 0; i < 4; ++i) {
    CHECK(cudaMemset(out, 0, (size_t)TOKENS * DFF * sizeof(float)));
    ks[i].f(w1, ln, out);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(i == 0 ? ref : got, out, (size_t)TOKENS * DFF * sizeof(float),
                     cudaMemcpyDeviceToHost));
    if (i) { double m = 0;
      for (size_t x = 0; x < (size_t)TOKENS * DFF; ++x) m = fmax(m, fabs(got[x] - ref[x]));
      printf("max |%c - A| = %.3e\n", 'A' + i, m); worst = fmax(worst, m); }
  }
  printf("worst deviation from A: %.3e (all variants compute the same GEMM)\n", worst);
  return 0;
}
