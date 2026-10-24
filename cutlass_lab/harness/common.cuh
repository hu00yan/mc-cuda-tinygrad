// Shared scaffolding for the CUTLASS lab harnesses.
//
// Every harness is arch-parameterised at compile time via -DLAB_SM=NNN so a single
// source file can be compiled once per rung of the ladder. The SM number selects
// both the CUTLASS arch tag and the tile/warp/stage configuration that CUTLASS's
// own heuristics record as the default for that arch (see RESULTS.md for the
// provenance of every number in here).
//
// PROVENANCE OF THE TILE NUMBERS
//   sm_75 : include/cutlass/gemm/device/default_gemm_configuration.h
//           DefaultGemmConfiguration<OpClassTensorOp, arch::Sm75, ...>
//           -> ThreadblockShape<128,256,32> WarpShape<64,64,32> InstrShape<16,8,8> Stages=2
//   sm_80 : same file, arch::Sm80 -> <128,256,64> <64,64,64> <16,8,16> Stages=3
//   sm_89 : same file, DefaultGemmConfigurationSm89F8 -> <128,256,64> <64,64,64> <16,8,32> Stages=3
//           (FP8 only. For f16/bf16 CUTLASS 4.8.0 supplies NO Sm89 specialisation
//            and DefaultGemm<..., arch::Sm89, ...> forwards to the Sm80 Mma; see
//            include/cutlass/gemm/kernel/default_gemm.h:271. That is itself a finding.)
//   simt fp32 (all archs): DefaultGemmConfiguration<OpClassSimt, ArchTag, f32,...>
//           -> <128,128,8> <32,64,8> <1,1,1> Stages=2, alignment 1

#pragma once

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "cutlass/cutlass.h"
#include "cutlass/util/device_memory.h"

namespace cutlass_lab {

// ---------------------------------------------------------------------------
// Problem shape. Fixed and not a multiple of any tile, on purpose: a
// power-of-two-multiple shape hides edge/predication bugs in the epilogue and
// would make a numerically wrong kernel look correct.
// ---------------------------------------------------------------------------
inline constexpr int kM = 1027;
inline constexpr int kN = 1031;
inline constexpr int kK = 2003;  // odd, and > every tile K we use
inline constexpr int kL = 1;

struct GemmProblem {
  int m = kM, n = kN, k = kK, l = kL;
};

// Deterministic input, identical on host and device, so the numpy reference and
// the device kernel see bit-identical inputs. Uses a plain integer LCG mapped
// into [-1, 1) and exactly representable in float32.
inline void fill_deterministic(std::vector<float>& v, unsigned seed = 12345u) {
  unsigned state = seed;
  for (size_t i = 0; i < v.size(); ++i) {
    state = state * 1664525u + 1013904223u;
    // 24 bits -> exactly representable mantissa, so no rounding surprise.
    float u = static_cast<float>(state >> 8) / static_cast<float>(1u << 24);
    v[i] = u * 2.0f - 1.0f;
  }
}

}  // namespace cutlass_lab
