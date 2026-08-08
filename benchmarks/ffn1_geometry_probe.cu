// gh-ocannl-531: what binds the gpt2_mini step's dominant kernel?
//
// Standalone, no OCANNL. Build and run:
//     nvcc -O3 -arch=sm_120 -o ffn1_geometry_probe benchmarks/ffn1_geometry_probe.cu
//     ./ffn1_geometry_probe          # exits nonzero if ANY timed variant disagrees
//
// Context. Four of the five kernels that make up 70% of the tuned gpt2_mini step are the FFN
// up-projection: out[tok][j] = sum_k w1[j][k] * ln[tok][k], with tok = 8*128 = 1024 tokens,
// j < 1024, k < 256. OCANNL emits it as two serial loops launched at
// grid=(8,1,1) x block=(128,1,1) -- 1024 threads on a 46-SM GPU -- and it takes 14.31 ms
// (12.7 ms without its gelu epilogue) for 536.9 MFLOP, i.e. 0.22% of this card's fp32 peak.
//
// Two things this probe had to get right, both learned the hard way:
//
//  1. The emitted C source accumulates into GLOBAL MEMORY (out[...] = fma(..., out[...])), but
//     that is not what runs. OCANNL's own PTX for cross_entropy_loss_fwd__seg25 keeps the
//     accumulator in a register across the whole k loop (one ld.global per output element, an
//     unrolled fma chain, one st.global), and nvcc does the same to the source form below.
//     So "global read-modify-write per iteration" describes the source, not the machine code,
//     and a register-accumulator variant is NOT a control -- the compiler makes the two
//     identical (verifiable: `cuobjdump -sass` shows one STG per kernel either way).
//
//  2. Widening the grid by giving one thread per output element also changes the thread ->
//     address mapping (adjacent threads move from 4 KiB apart to adjacent), so it confounds
//     occupancy with coalescing. The E variants below avoid that: they keep A's exact
//     thread -> token mapping and per-thread inner loop, and only split the j range across
//     blockIdx.y. Every thread touches the same addresses in the same order as in A; the only
//     thing that changes is how many blocks are resident.
//
// Measured on an RTX 5070 Ti Laptop (46 SMs, sm_120), CUDA 13.3, WSL2:
//
//     variant                         blocks   threads        ms   GFLOP/s   vs A
//     A  grid=(8,1)   block=128            8      1024    12.730      42.2   1.00x  (as shipped)
//     E  grid=(8,1)   block=128            8      1024    12.596      42.6   1.01x
//     E  grid=(8,2)   block=128           16      2048     6.285      85.4   2.03x
//     E  grid=(8,4)   block=128           32      4096     3.182     168.7   4.00x
//     E  grid=(8,16)  block=128          128     16384     2.393     224.3   5.32x
//     E  grid=(8,128) block=128         1024    131072     2.324     231.0   5.48x
//     C  one thread per output          4096   1048576     2.282     235.2   5.58x
//
// Time falls linearly with resident blocks and saturates around 5.48x, with every thread
// touching the same addresses in the same order throughout -- the signature of a
// parallelism-starved kernel, not a bandwidth-saturated one. Changing the mapping as well (C)
// adds a further 1%, so coalescing is not the story either. A reproduces the in-step kernel
// (14.31 ms there, the difference being its gelu epilogue).
//
// See benchmarks/report-gh531-profile.md for the reading.
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <cuda_runtime.h>

#define TOKENS 1024   // 8 batch * 128 seq
#define D 256
#define DFF 1024

#define CHECK(x) do { cudaError_t err_=(x); if(err_){ \
  printf("CUDA error %s @%d\n",cudaGetErrorString(err_),__LINE__); exit(2);} } while(0)

// --- A: the emitted form, verbatim, at the emitted launch geometry ----------
__global__ void A_shipped(const float* __restrict__ w1, const float* __restrict__ ln,
                          float* __restrict__ out) {
  const int b = (int)blockIdx.x;
  const int t = (int)threadIdx.x;
  for (int j = 0; j <= DFF - 1; ++j)
    for (int k = 0; k <= D - 1; ++k)
      out[((b)*128 + t) * DFF + j] =
          fmaf(w1[(j)*D + k], ln[((b)*128 + t) * D + k], out[((b)*128 + t) * DFF + j]);
}

// --- E: identical mapping and inner loop; only the block count changes ------
// grid = (8, chunks, 1), block = (128,1,1). Thread (blockIdx.x, threadIdx.x) owns the same
// token as in A and walks the same addresses; blockIdx.y selects which slice of j it does.
__global__ void E_occupancy(const float* __restrict__ w1, const float* __restrict__ ln,
                            float* __restrict__ out, int per_chunk) {
  const int b = (int)blockIdx.x;
  const int t = (int)threadIdx.x;
  const int j0 = (int)blockIdx.y * per_chunk;
  const int j1 = j0 + per_chunk;
  for (int j = j0; j < j1; ++j)
    for (int k = 0; k <= D - 1; ++k)
      out[((b)*128 + t) * DFF + j] =
          fmaf(w1[(j)*D + k], ln[((b)*128 + t) * D + k], out[((b)*128 + t) * DFF + j]);
}

// --- C: one thread per output element (mapping CHANGES -- reference only) ---
__global__ void C_thread_per_output(const float* __restrict__ w1, const float* __restrict__ ln,
                                    float* __restrict__ out) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= TOKENS * DFF) return;
  int tok = idx / DFF, j = idx % DFF;
  float acc = 0.f;
  for (int k = 0; k < D; ++k) acc = fmaf(w1[j * D + k], ln[tok * D + k], acc);
  out[tok * DFF + j] = acc;
}

// Deterministic but NON-PERIODIC inputs (splitmix64 on the flat index). Periodic fixtures make
// the reference blind to mapping errors: with activation rows repeating every 13 tokens, a kernel
// reading token t+13 instead of t matches cell for cell. Kept identical to ffn1_nvrtc_harness.c.
static float fixture(uint64_t i) {
  uint64_t x = i * 0x9E3779B97F4A7C15ull + 0x0123456789ABCDEFull;
  x ^= x >> 30; x *= 0xBF58476D1CE4E5B9ull;
  x ^= x >> 27; x *= 0x94D049BB133111EBull;
  x ^= x >> 31;
  return (float)(((double)(x >> 11) / 9007199254740992.0) - 0.5) * 0.1f;
}

static float *g_w1, *g_ln, *g_out;

static void run_A() { A_shipped<<<8, 128>>>(g_w1, g_ln, g_out); }
static void run_C() {
  C_thread_per_output<<<(TOKENS * DFF + 255) / 256, 256>>>(g_w1, g_ln, g_out);
}
static int g_chunks = 1;
static void run_E() {
  dim3 grid(8, g_chunks, 1);
  E_occupancy<<<grid, 128>>>(g_w1, g_ln, g_out, DFF / g_chunks);
}

// Timed: the accumulating forms must start from zero each rep, so the memset is inside the
// loop for all variants alike (it is ~0.1% of the fastest variant's time).
static float bench(void (*launch)(), int reps) {
  CHECK(cudaMemset(g_out, 0, (size_t)TOKENS * DFF * sizeof(float)));
  launch();
  CHECK(cudaDeviceSynchronize());
  cudaEvent_t ev0, ev1;
  CHECK(cudaEventCreate(&ev0)); CHECK(cudaEventCreate(&ev1));
  CHECK(cudaEventRecord(ev0));
  for (int i = 0; i < reps; ++i) {
    CHECK(cudaMemsetAsync(g_out, 0, (size_t)TOKENS * DFF * sizeof(float)));
    launch();
  }
  CHECK(cudaEventRecord(ev1));
  CHECK(cudaEventSynchronize(ev1));
  float ms; CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
  CHECK(cudaEventDestroy(ev0)); CHECK(cudaEventDestroy(ev1));
  return ms / reps;
}

// Independent fp64 reference over EVERY output cell (a strided sample can step over a narrow
// unwritten band, e.g. from a chunk-count mismatch). The reference is built once by the caller.
static double* g_ref;   // [TOKENS][DFF] exact value
static double* g_cond;  // [TOKENS][DFF] sum |a_k b_k| -- a near-cancelling dot has no meaningful
                        // relative error, so the bound is |got - ref| <= tol * cond.
static void build_reference(const float* hw, const float* hl) {
  g_ref = (double*)malloc((size_t)TOKENS * DFF * sizeof(double));
  g_cond = (double*)malloc((size_t)TOKENS * DFF * sizeof(double));
  for (int tok = 0; tok < TOKENS; ++tok)
    for (int j = 0; j < DFF; ++j) {
      double a = 0, c = 0;
      for (int k = 0; k < D; ++k) {
        double t = (double)hw[j * D + k] * (double)hl[tok * D + k];
        a += t; c += fabs(t);
      }
      g_ref[(size_t)tok * DFF + j] = a;
      g_cond[(size_t)tok * DFF + j] = c;
    }
}

static int verify(void (*launch)(), const char* name) {
  CHECK(cudaMemset(g_out, 0, (size_t)TOKENS * DFF * sizeof(float)));
  launch();
  CHECK(cudaDeviceSynchronize());
  CHECK(cudaGetLastError());
  float* got = (float*)malloc((size_t)TOKENS * DFF * sizeof(float));
  CHECK(cudaMemcpy(got, g_out, (size_t)TOKENS * DFF * sizeof(float), cudaMemcpyDeviceToHost));
  double worst = 0.0;
  for (size_t i = 0; i < (size_t)TOKENS * DFF; ++i) {
    if (!isfinite(got[i])) {
      printf("  FAIL %s: non-finite at %zu\n", name, i); free(got); return 1; }
    double d = fabs((double)got[i] - g_ref[i]) / fmax(1e-30, g_cond[i]);
    if (d > worst) worst = d;
  }
  free(got);
  if (worst > 1e-5) { printf("  FAIL %s: worst deviation %.3e\n", name, worst); return 1; }
  printf("  ok   %-34s all %d cells, worst %.2e\n", name, TOKENS * DFF, worst);
  return 0;
}

int main() {
  float* hw = (float*)malloc((size_t)DFF * D * sizeof(float));
  float* hl = (float*)malloc((size_t)TOKENS * D * sizeof(float));
  for (int i = 0; i < DFF * D; ++i) hw[i] = fixture((uint64_t)i);
  for (int i = 0; i < TOKENS * D; ++i) hl[i] = fixture(0x10000000ull + (uint64_t)i);
  // Plain device memory: managed memory without prefetch faults per page and makes the
  // many-block variants look absurdly slow. Do not "simplify" this to cudaMallocManaged.
  CHECK(cudaMalloc(&g_w1, (size_t)DFF * D * sizeof(float)));
  CHECK(cudaMalloc(&g_ln, (size_t)TOKENS * D * sizeof(float)));
  CHECK(cudaMalloc(&g_out, (size_t)TOKENS * DFF * sizeof(float)));
  CHECK(cudaMemcpy(g_w1, hw, (size_t)DFF * D * sizeof(float), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(g_ln, hl, (size_t)TOKENS * D * sizeof(float), cudaMemcpyHostToDevice));

  cudaDeviceProp p; CHECK(cudaGetDeviceProperties(&p, 0));
  double flops = 2.0 * TOKENS * D * DFF;
  printf("device: %s, %d SMs, cc %d.%d\n", p.name, p.multiProcessorCount, p.major, p.minor);
  printf("GEMM: out[%d][%d] = sum_k w1[%d][k] * ln[tok][k], k < %d -- %.1f MFLOP\n\n",
         TOKENS, DFF, DFF, D, flops / 1e6);

  printf("occupancy sweep: A's thread->address mapping held FIXED, only the block count varies\n");
  printf("%-34s %8s %9s %10s %9s\n", "variant", "blocks", "threads", "ms", "GFLOP/s");
  float a_ms = bench(run_A, 3);
  printf("%-34s %8d %9d %10.3f %9.1f   (as shipped)\n", "A  grid=(8,1)   block=128", 8, 1024,
         a_ms, flops / (a_ms / 1e3) / 1e9);
  int chunks[] = {1, 2, 4, 8, 16, 32, 64, 128};
  float best_e = 1e30f; int best_c = 1;
  for (int i = 0; i < 8; ++i) {
    g_chunks = chunks[i];
    int reps = chunks[i] >= 16 ? 20 : 3;
    float ms = bench(run_E, reps);
    if (ms < best_e) { best_e = ms; best_c = chunks[i]; }
    char lbl[64];
    snprintf(lbl, sizeof lbl, "E  grid=(8,%d) block=128", chunks[i]);
    printf("%-34s %8d %9d %10.3f %9.1f   %.2fx vs A\n", lbl, 8 * chunks[i], 1024 * chunks[i], ms,
           flops / (ms / 1e3) / 1e9, a_ms / ms);
  }
  float c_ms = bench(run_C, 20);
  printf("\n%-34s %8d %9d %10.3f %9.1f   %.2fx vs A  (mapping differs; reference only)\n",
         "C  one thread per output", (TOKENS * DFF + 255) / 256, TOKENS * DFF, c_ms,
         flops / (c_ms / 1e3) / 1e9, a_ms / c_ms);
  printf("\nbest same-mapping variant: E with %d chunks, %.3f ms (%.2fx over A)\n", best_c, best_e,
         a_ms / best_e);

  // Correctness for EVERY variant that was timed above -- including each intermediate chunk
  // count, since an invalid intermediate row would otherwise still support the scaling curve.
  printf("\ncorrectness (independent fp64 reference, every cell, every timed variant):\n");
  build_reference(hw, hl);
  int bad = 0;
  bad |= verify(run_A, "A  as shipped");
  for (int i = 0; i < 8; ++i) {
    char lbl[64];
    g_chunks = chunks[i];
    snprintf(lbl, sizeof lbl, "E  grid=(8,%d)", chunks[i]);
    bad |= verify(run_E, lbl);
  }
  bad |= verify(run_C, "C  one thread per output");
  if (bad) { printf("\nFAILED: at least one timed variant does not compute the GEMM\n"); return 1; }
  printf("\nevery timed variant agrees with the fp64 reference\n");
  return 0;
}
