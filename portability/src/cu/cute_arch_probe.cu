// Direct CuTe probe: ask for a Hopper warpgroup MMA and an H100 TMA copy from a
// file compiled for an OLDER arch, with the CUTLASS host-side arch macros left
// undefined (which is exactly what a CUTLASS build for sm_80 produces).
//
// cute/arch/config.hpp gates CUTE_ARCH_MMA_SM90A_ENABLED on
// CUTLASS_ARCH_MMA_SM90A_ENABLED -- a *host-side* macro, not __CUDA_ARCH__.
// So the question is what the wgmma call site becomes when the gate is closed.
// cute/config.hpp:160 says:
//     #define CUTE_INVALID_CONTROL_PATH(x) assert(0 && x); printf(x)
// i.e. a runtime assert, NOT a compile error.  This file checks whether that is
// really what lands in the cubin, by looking for the diagnostic string in SASS.

#include <cstdio>
#include "cute/config.hpp"
#include "cute/arch/mma_sm90_gmma.hpp"
#include "cute/arch/copy_sm90_tma.hpp"

#if defined(CL_ARG_WGMMA)
extern "C" __global__ void k_wgmma_probe(uint64_t da, uint64_t db, uint32_t* out) {
  uint32_t d0 = 0, d1 = 0;
  cute::SM90::GMMA::MMA_64x8x16_F16F16F16_SS<
      cute::SM90::GMMA::Major::K, cute::SM90::GMMA::Major::K>::fma(
      da, db, d0, d1, cute::SM90::GMMA::ScaleOut::One);
  out[threadIdx.x] = d0 + d1;
}
#endif

#if defined(CL_ARG_TMA)
extern "C" __global__ void k_tma_probe(void const* g, uint64_t* mbar, void* s) {
  cute::SM90_TMA_LOAD_1D::copy(g, mbar, uint64_t(0), s, 0);
}
#endif

#if defined(CL_ARG_TMA_BULK)
extern "C" __global__ void k_tma_bulk_probe(void const* g, uint64_t* mbar, void* s) {
  cute::SM90_BULK_COPY_G2S::copy(g, mbar, s, 256);
}
#endif

int main() { return 0; }
