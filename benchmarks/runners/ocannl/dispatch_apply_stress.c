/* Standalone discriminator for gh-ocannl-870: is the SIGTRAP seen under the cc backend's
   pool-backed Grid rendering a buffer or stream lifetime defect of OCANNL's, or does
   libdispatch trap on its own at that fork/join rate?

   This file contains no OCANNL code -- no lowering, no context, no buffers, no ctypes. It only
   calls dispatch_apply in a loop, which is exactly what a cc kernel's parallel Grid loop does
   (`C_syntax`'s `parallel_grid_loop`, rendered under `cc_parallel_grid=dispatch`, the probed
   default on Apple platforms).

   Build and run (macOS only; the header does not exist elsewhere):

     clang -O2 -o /tmp/dispatch_apply_stress \
         benchmarks/runners/ocannl/dispatch_apply_stress.c
     for i in $(seq 1 20); do
       /tmp/dispatch_apply_stress 5000000 6 block auto >/dev/null 2>&1; printf " %d" $?
     done; echo

   An exit status of 133 (128 + SIGTRAP) is the reproduction. The crash report under
   ~/Library/Logs/DiagnosticReports reads

     BUG IN CLIENT OF LIBMALLOC: memory corruption of free block
     _xzm_xzone_malloc_freelist_outlined <- _dispatch_calloc_typed
       <- _dispatch_apply_with_attr_f <- dispatch_apply

   which is the same signature, frame for frame above the call site, as the one a generated
   kernel produces. Measured 2026-09-05 on an M4 Max, macOS 26.6.2 build 25G83: 10 of 40 runs
   trapped at 5,000,000 applies of extent 6 (two batches of 20, 4 and 6). The trap is
   probabilistic and heap-layout dependent; a single clean run proves nothing, so always run a
   batch.

   The arms exist so that a later reader can re-establish, rather than assume, which shapes are
   affected after an OS update: `block` builds a stack block per call (what OCANNL emits) and
   `f` passes an outlined function plus a context pointer; `auto` passes DISPATCH_APPLY_AUTO
   (what OCANNL emits) and `global` an explicit global concurrent queue. As of the measurement
   above no arm was reliably clean -- do not read a quiet arm as a fix without a batch behind
   it. */

#include <dispatch/dispatch.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void body_f(void *ctx, size_t chunk) {
  float *p = (float *)ctx;
  p[chunk] += 1.0f;
}

int main(int argc, char **argv) {
  long calls = argc > 1 ? atol(argv[1]) : 5000000;
  size_t extent = argc > 2 ? (size_t)atol(argv[2]) : 6;
  const char *form = argc > 3 ? argv[3] : "block";
  const char *queue = argc > 4 ? argv[4] : "auto";
  /* Heap, not stack: the block captures a pointer to it exactly as a generated kernel captures
     its context buffer parameter. */
  float *p = (float *)calloc(extent, sizeof(float));
  dispatch_queue_t q = strcmp(queue, "auto") == 0
                           ? DISPATCH_APPLY_AUTO
                           : dispatch_get_global_queue(QOS_CLASS_DEFAULT, 0);
  if (p == NULL) return 2;
  if (strcmp(form, "block") == 0)
    for (long r = 0; r < calls; ++r)
      dispatch_apply(extent, q, ^(size_t chunk) { p[chunk] += 1.0f; });
  else
    for (long r = 0; r < calls; ++r) dispatch_apply_f(extent, q, p, body_f);
  /* Printed so a run that completed is distinguishable from one that never started, and so the
     compiler cannot discard the loop. */
  printf("completed %ld dispatch_apply calls, extent %zu, form %s, queue %s (p[0] = %g)\n", calls,
         extent, form, queue, (double)p[0]);
  free(p);
  return 0;
}
