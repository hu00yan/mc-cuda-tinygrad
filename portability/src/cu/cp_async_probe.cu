// Direct probe of the code path that silently degrades.
//
// cutlass/arch/memory_sm80.h defines
//     #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
//       #define CUDA_CP_ASYNC_ACTIVATED 1
//     #else
//       #define CUDA_CP_ASYNC_ACTIVATED 0
//     #endif
// and cp_async_zfill's body is
//     #if CUDA_CP_ASYNC_ACTIVATED
//         asm volatile("... @p cp.async.ca.shared.global [%1], [%2], %3;");
//     #else
//         *static_cast<AccessType*>(smem_ptr) = *static_cast<AccessType const*>(global_ptr);
//     #endif
//
// That is a device-side gate on __CUDA_ARCH__, so compiling the SAME source with
// different -arch silently changes the instruction. This file calls that exact
// function, so the SASS reading is CUTLASS's behaviour, not a paraphrase of it.

#include <cstdint>
#include "cutlass/arch/memory_sm80.h"

using C = cutlass::arch::cp_async_zfill<16>;

extern "C" __global__ void k_cp_async_f32(float const* __restrict__ g, float* s) {
  int lane = threadIdx.x & 31;
  int tid = threadIdx.x;
  __shared__ float buf[32 * 8];
  float* smem = &buf[lane * 8 + tid / 32];
  C(smem, g + tid * 8, true);
  cutlass::arch::cp_async_fence();
  cutlass::arch::cp_async_wait<0>();
  if (smem[0] == 1234.5f) s[tid] = smem[1];   // keep the copy alive
}

int main() { return 0; }
