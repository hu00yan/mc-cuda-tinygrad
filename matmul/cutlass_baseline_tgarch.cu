/*
 * CUTLASS hand-written baseline for sm_75 / sm_80 / sm_89, via CUTLASS 2.11
 * (../cutlass2x).  The Hopper and Blackwell rungs use CUTLASS 3.x/4.x instead
 * (see cutlass_baseline_sm90a.cu, cutlass_baseline_sm100a.cu) because the 2.x
 * code has no Hopper or Blackwell path at all.
 *
 * PROVENANCE -- the important part
 * --------------------------------
 * Not my kernels. Every inner loop is CUTLASS 2.x
 * (`cutlass::gemm::device::Gemm` -> `DefaultGemm` -> `cutlass::gemm::kernel::Gemm`).
 * I chose only the tile configuration, and every configuration below is copied
 * verbatim from a file CUTLASS itself ships, so the instantiation is known to be
 * one CUTLASS supports:
 *
 *   (A) fp32 SIMT, FFMA pipe
 *       threadblock 128x128x8, warp 32x64x8, instruction 1x1x1, 3 stages,
 *       scalar epilogue.
 *       SOURCE: cutlass test/unit/gemm/device/simt_sgemm_tn_sm80.cu
 *       (this tile is also the centre of CUTLASS's own SGEMM profiler sweep
 *        for SM70..SM80)
 *
 *   (B) fp16 tensor-op, HMMA pipe
 *       sm_75: threadblock 128x128x32, warp 64x64x32, instruction 16x8x8,
 *              2 stages.
 *              SOURCE: cutlass examples/12_gemm_bias_relu/gemm_bias_relu.cu
 *              (Sm75 + cutlass::half_t). Note instruction 16x8x8, i.e. the
 *              m16n8k8 shape, which is the largest half-precision mma.sync
 *              shape Turing supports.
 *       sm_80/sm_89: threadblock 128x128x32, warp 64x64x32, instruction 16x8x16,
 *              4 stages.
 *              SOURCE: cutlass examples/47_ampere_gemm_universal_streamk
 *              (ArchTag Sm80, NumStages 4). The 4-stage multistage mainloop is
 *              what lowers the global->shared copies to cp.async / LDGSTS.
 *              Ada (sm_89) uses the identical collective; only -arch differs.
 *
 * WHY 2.11 AND NOT 4.8: CUTLASS 4.8's default_gemm.h has no 2.x OpClassSimt
 * specialization for Sm75 at all, and its Sm80 Ampere path does not compile
 * under clang-20 (we have no nvcc -- see Dockerfile.toolchain). The 2.x kernels
 * are still the shipping implementation for Turing and Ampere-class tensor-op
 * GEMM, so they are the fair "human state of the art" for these three rungs.
 *
 * TEMPLATE CONSTRAINTS (from CUTLASS's own default_gemm.h; violating one yields
 * a misleading "implicit instantiation of undefined template"):
 *   * OpClassSimt requires instruction GemmShape<1,1,1> and epilogue kCount==1.
 *   * OpClassTensorOp + Sm75 hard-codes Stages == 2.
 *   * Trailing params (AlignmentA/B, SplitKSerial, Operator, Gather*, ...) are
 *     defaulted by device::Gemm; leave them defaulted.
 *
 * LIMITATION: these are published configurations, NOT autotuned on this machine
 * (no GPU available). They are a legitimate "what does a careful human write"
 * baseline for an instruction-class comparison. They are NOT a peak-TFLOP/s
 * claim, and the stage counts in particular are CUTLASS's documented defaults
 * rather than the output of CUTLASS's autotuner on this exact problem size.
 *
 * Device-only compilation; main() exists only to mark the kernels as used.
 */

#include <cstdio>
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/kernel/default_gemm.h"
#include "cutlass/numeric_types.h"

#ifndef TG_ARCH
#define TG_ARCH 75
#endif

// Select which of the two baselines to instantiate, so each lands in its own
// cubin and its instruction histogram is not polluted by the other.
//   SELECT=0 -> fp32 SIMT (FFMA pipe) only
//   SELECT=1 -> fp16 tensor-op (HMMA pipe) only
#ifndef SELECT
#define SELECT 0
#endif

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;
using ElementFP32 = float;
using ElementFP16 = cutlass::half_t;
using ElementAcc = float;

// CUTLASS 2.x's DefaultGemm only has a *generic*-ArchTag OpClassSimt
// specialization (Stages fixed at 2) and an Sm80-specific one (variable
// Stages). There is no Sm75 OpClassSimt specialization at all, so for sm_75 we
// instantiate the generic one with its required 2 stages.
#if TG_ARCH == 75
using SimtArch   = cutlass::arch::Sm75;
using TensorArch = cutlass::arch::Sm75;
constexpr int kStagesSIMT = 2;
#elif TG_ARCH == 80 || TG_ARCH == 89
using SimtArch   = cutlass::arch::Sm80;
using TensorArch = cutlass::arch::Sm80;
constexpr int kStagesSIMT = 3;
#endif

// --- (A) fp32 SIMT: the FFMA-pipe baseline ---------------------------------
// config copied from cutlass test/unit/gemm/device/simt_sgemm_tn_sm80.cu
using EpilogueSIMT = cutlass::epilogue::thread::LinearCombination<
  ElementFP32, 1, ElementAcc, ElementAcc>;   // kCount must be 1 for SIMT

using GemmSIMT = cutlass::gemm::device::Gemm<
  ElementFP32, LayoutA,
  ElementFP32, LayoutB,
  ElementFP32, LayoutC,
  ElementAcc,
  cutlass::arch::OpClassSimt, SimtArch,
  cutlass::gemm::GemmShape<128, 128, 8>,
  cutlass::gemm::GemmShape<32, 64, 8>,
  cutlass::gemm::GemmShape<1, 1, 1>,
  EpilogueSIMT,
  cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
  kStagesSIMT>;

// --- (B) fp16 tensor-op: the HMMA-pipe baseline ---------------------------
#if TG_ARCH == 75
using GemmFP16 = cutlass::gemm::device::Gemm<
  ElementFP16, LayoutA,
  ElementFP16, LayoutB,
  ElementFP32, LayoutC,
  ElementAcc,
  cutlass::arch::OpClassTensorOp, cutlass::arch::Sm75,
  cutlass::gemm::GemmShape<128, 128, 32>,    // examples/12_gemm_bias_relu
  cutlass::gemm::GemmShape<64, 64, 32>,
  cutlass::gemm::GemmShape<16, 8, 8>,        // Turing: m16n8k8
  EpilogueSIMT,
  cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
  2>;
#else
using GemmFP16 = cutlass::gemm::device::Gemm<
  ElementFP16, LayoutA,
  ElementFP16, LayoutB,
  ElementFP32, LayoutC,
  ElementAcc,
  cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
  cutlass::gemm::GemmShape<128, 128, 32>,    // examples/47 (ArchTag Sm80)
  cutlass::gemm::GemmShape<64, 64, 32>,
  cutlass::gemm::GemmShape<16, 8, 16>,       // Ampere: m16n8k16
  EpilogueSIMT,
  cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
  4>;                                        // multistage -> cp.async / LDGSTS
#endif

// CUTLASS 2.x Gemm::Arguments is
//   (GemmCoord, TensorRefA, TensorRefB, TensorRefC, TensorRefD,
//    EpilogueOutputOp::Params, split_k_slices, gather/scatter...)
// We never dereference the refs -- nothing is linked or run -- but we must build
// one so that clang's "used __global__" analysis emits the kernel.
template <class Gemm>
void mark_used(int m, int n, int k) {
  using TA = typename Gemm::TensorRefA;
  using TB = typename Gemm::TensorRefB;
  using TC = typename Gemm::TensorRefC;
  using TD = typename Gemm::TensorRefD;
  // CUTLASS 2.x TensorRef ctor is (Element* ptr, Layout const& layout).
  typename TA::Layout la(typename TA::Layout::LongIndex(1));
  typename TB::Layout lb(typename TB::Layout::LongIndex(1));
  typename TC::Layout lc(typename TC::Layout::LongIndex(1));
  typename TD::Layout ld(typename TD::Layout::LongIndex(1));
  TA a(nullptr, la);
  TB b(nullptr, lb);
  TC c(nullptr, lc);
  TD d(nullptr, ld);
  typename Gemm::Arguments args(cutlass::gemm::GemmCoord(m, n, k), a, b, c, d);
  Gemm op;
  // operator()(args, workspace, stream) -> initialize() + run(), and run() is
  // what references GemmKernel's __global__. Constructing Arguments alone is not
  // enough for clang to emit it. Under --cuda-device-only no host code is
  // generated, so the null workspace/stream are never dereferenced -- we only
  // need the call to be *analysed*.
  (void)op(args, nullptr, cudaStream_t(nullptr));
}

int main() {
#if SELECT == 0
  mark_used<GemmSIMT>(4096, 4096, 4096);
#else
  mark_used<GemmFP16>(4096, 4096, 4096);
#endif
  return 0;
}
