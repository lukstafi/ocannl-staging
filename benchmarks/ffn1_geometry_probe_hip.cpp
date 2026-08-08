// gh-ocannl-569: the HIP analog of benchmarks/ffn1_geometry_probe.cu.
//
// The question, transplanted from gh-ocannl-531: the gpt2_mini FFN up-projection ships as the
// naive scalar form -- serial output loop, serial reduction, launched at grid=(8,1) x block=(128,1)
// -- so 1024 threads occupy 8 of the device's 20 workgroup processors. Is what binds it the memory
// traffic, or the parallelism? Spreading the OUTPUT axis across blocks changes the second and not
// the first.
//
// The control is what makes the answer mean anything, and it is the same one the CUDA probe used:
// each thread keeps its inner k loop and its token mapping, and only the j range is split across
// blockIdx.y in CONTIGUOUS chunks. Every thread therefore walks the same addresses in the same
// order as in the shipped kernel; the only thing that varies across rows is how many blocks are
// resident. A kernel bound by the bytes it moves would not speed up under that change.
//
// Inputs are deliberately non-periodic (an irrational-stride recurrence), so a variant that
// duplicated or permuted a chunk of the output could not accidentally match the reference. Every
// variant is verified against the shipped kernel's output BITWISE before its time is reported --
// legitimate here, and not a violation of the gfx1151 WMMA caveat, because no variant uses WMMA:
// each thread performs exactly the same fmaf chain in the same order, so equality is exact.
//
//   hipcc --offload-arch=gfx1151 -O3 -o probe ffn1_geometry_probe_hip.cpp && ./probe

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <cmath>

#define HC(x) do { hipError_t _hc = (x); if (_hc != hipSuccess) { \
  fprintf(stderr, "HIP %s @%d: %s\n", #x, __LINE__, hipGetErrorString(_hc)); exit(1);} } while(0)

// gpt2_mini geometry: batch 8 x seq 128 = 1024 rows, d_model 256, d_ff 1024.
static const int ROWS = 1024, DM = 256, DFF = 1024;

// The shipped kernel, transcribed from the emitted arm-A source (loop bounds and index
// expressions unchanged; only the tensor names are shortened).
__global__ void ffn1_shipped(const float *__restrict__ w1, const float *__restrict__ b1,
                             const float *__restrict__ ln, float *__restrict__ acc,
                             float *__restrict__ out) {
  const int bx = (int)blockIdx.x;
  const int tx = (int)threadIdx.x;
  for (int j = 0; j <= DFF - 1; ++j)
    for (int k = 0; k <= DM - 1; ++k)
      acc[(bx * 128 + tx) * DFF + j] =
          fmaf(w1[j * DM + k], ln[(bx * 128 + tx) * DM + k], acc[(bx * 128 + tx) * DFF + j]);
  for (int j = 0; j <= DFF - 1; ++j) {
    float v = acc[(bx * 128 + tx) * DFF + j] + b1[j];
    out[(bx * 128 + tx) * DFF + j] =
        0.5f * v * (1.0f + tanhf(0.7978845608028654f * fmaf(0.044715f, v * v * v, v)));
  }
}

// The same kernel with the j range split across blockIdx.y in contiguous chunks. Each thread walks
// a contiguous sub-range of j in increasing order, exactly as it did before -- same addresses,
// same order, same arithmetic -- and the only difference is how many blocks are resident.
__global__ void ffn1_chunked(const float *__restrict__ w1, const float *__restrict__ b1,
                             const float *__restrict__ ln, float *__restrict__ acc,
                             float *__restrict__ out, int chunk) {
  const int bx = (int)blockIdx.x;
  const int tx = (int)threadIdx.x;
  const int j0 = (int)blockIdx.y * chunk;
  const int j1 = min(j0 + chunk, DFF);
  for (int j = j0; j < j1; ++j)
    for (int k = 0; k <= DM - 1; ++k)
      acc[(bx * 128 + tx) * DFF + j] =
          fmaf(w1[j * DM + k], ln[(bx * 128 + tx) * DM + k], acc[(bx * 128 + tx) * DFF + j]);
  for (int j = j0; j < j1; ++j) {
    float v = acc[(bx * 128 + tx) * DFF + j] + b1[j];
    out[(bx * 128 + tx) * DFF + j] =
        0.5f * v * (1.0f + tanhf(0.7978845608028654f * fmaf(0.044715f, v * v * v, v)));
  }
}

static float *dw1, *db1, *dln, *dacc, *dout;

static double run_and_time(int blocks_y, int reps, std::vector<float> &host_out) {
  const int chunk = (blocks_y == 1) ? DFF : (DFF + blocks_y - 1) / blocks_y;
  auto launch = [&]() {
    HC(hipMemset(dacc, 0, (size_t)ROWS * DFF * sizeof(float)));  // the step's zero-init
    if (blocks_y == 1)
      hipLaunchKernelGGL(ffn1_shipped, dim3(8, 1, 1), dim3(128, 1, 1), 0, 0, dw1, db1, dln, dacc, dout);
    else
      hipLaunchKernelGGL(ffn1_chunked, dim3(8, blocks_y, 1), dim3(128, 1, 1), 0, 0, dw1, db1, dln,
                         dacc, dout, chunk);
  };
  launch();
  HC(hipDeviceSynchronize());
  host_out.resize((size_t)ROWS * DFF);
  HC(hipMemcpy(host_out.data(), dout, host_out.size() * sizeof(float), hipMemcpyDeviceToHost));

  // Time the kernel alone: the memset is a separate dispatch and is excluded by recording the
  // start event after it.
  std::vector<double> ms;
  for (int r = 0; r < reps; ++r) {
    HC(hipMemset(dacc, 0, (size_t)ROWS * DFF * sizeof(float)));
    HC(hipDeviceSynchronize());
    hipEvent_t b, e; HC(hipEventCreate(&b)); HC(hipEventCreate(&e));
    HC(hipEventRecord(b));
    if (blocks_y == 1)
      hipLaunchKernelGGL(ffn1_shipped, dim3(8, 1, 1), dim3(128, 1, 1), 0, 0, dw1, db1, dln, dacc, dout);
    else
      hipLaunchKernelGGL(ffn1_chunked, dim3(8, blocks_y, 1), dim3(128, 1, 1), 0, 0, dw1, db1, dln,
                         dacc, dout, chunk);
    HC(hipEventRecord(e));
    HC(hipEventSynchronize(e));
    float t; HC(hipEventElapsedTime(&t, b, e)); ms.push_back(t);
    HC(hipEventDestroy(b)); HC(hipEventDestroy(e));
  }
  std::sort(ms.begin(), ms.end());
  return ms[ms.size() / 2];
}

int main() {
  hipDeviceProp_t prop; HC(hipGetDeviceProperties(&prop, 0));
  printf("device: %s (%s), %d workgroup processors, %d MHz\n",
         prop.name, prop.gcnArchName, prop.multiProcessorCount, prop.clockRate / 1000);
  printf("FFN up-projection %dx%d reducing over %d, %.1f MFLOP\n\n",
         ROWS, DFF, DM, 2.0 * ROWS * DFF * DM / 1e6);

  std::vector<float> hw1((size_t)DFF * DM), hb1(DFF), hln((size_t)ROWS * DM);
  // Non-periodic by construction: an irrational-increment sawtooth has no repeating block, so a
  // chunk computed from the wrong j range cannot coincide with the right one.
  double s = 0.31830988618379067;
  for (auto &v : hw1) { s += 0.6180339887498949; s -= (int)s; v = (float)(s - 0.5); }
  for (auto &v : hln) { s += 0.4142135623730951; s -= (int)s; v = (float)(s - 0.5); }
  for (auto &v : hb1) { s += 0.7320508075688772; s -= (int)s; v = (float)(s - 0.5); }

  HC(hipMalloc(&dw1, hw1.size() * sizeof(float)));
  HC(hipMalloc(&db1, hb1.size() * sizeof(float)));
  HC(hipMalloc(&dln, hln.size() * sizeof(float)));
  HC(hipMalloc(&dacc, (size_t)ROWS * DFF * sizeof(float)));
  HC(hipMalloc(&dout, (size_t)ROWS * DFF * sizeof(float)));
  HC(hipMemcpy(dw1, hw1.data(), hw1.size() * sizeof(float), hipMemcpyHostToDevice));
  HC(hipMemcpy(db1, hb1.data(), hb1.size() * sizeof(float), hipMemcpyHostToDevice));
  HC(hipMemcpy(dln, hln.data(), hln.size() * sizeof(float), hipMemcpyHostToDevice));

  std::vector<float> ref, got;
  const double base = run_and_time(1, 10, ref);
  printf("| variant | blocks | ms | vs shipped | verified |\n|---|---:|---:|---:|---|\n");
  printf("| **as shipped**, grid=(8,1) | 8 | **%.2f** | 1.00x | reference |\n", base);

  for (int by : {2, 4, 16, 128}) {
    const double t = run_and_time(by, 10, got);
    size_t bad = 0;
    for (size_t i = 0; i < ref.size(); ++i)
      if (ref[i] != got[i]) ++bad;            // bitwise: identical fmaf chain per thread
    printf("| j chunked, grid=(8,%d) | %d | %.2f | %.2fx | %s |\n", by, 8 * by, t, base / t,
           bad == 0 ? "bitwise identical" : "MISMATCH");
    if (bad) { printf("  %zu of %zu cells differ -- variant rejected\n", bad, ref.size()); return 1; }
  }
  return 0;
}
