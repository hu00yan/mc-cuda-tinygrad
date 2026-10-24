// Compile-time reporter for the resources CUTLASS's chosen configuration needs.
//
// This exists because the GEMM harnesses can be compiled but not RUN here, so
// their runtime `sizeof(SharedStorage)` printout is unreachable. Everything
// below is a compile-time constant, so it can be reported from a host binary
// that needs no GPU at all. That makes the per-arch resource table in RESULTS.md
// measured rather than recalled.
//
// It is a pure reporter: it instantiates the same device::Gemm as sgemm.cu (with
// the same "let CUTLASS choose everything" argument list) and prints the numbers.
//
// Build:  nvcc -DLAB_SM=NNN -DLAB_MODE=0|1 print_cfg.cu     (no -arch needed)
// Run:    ./print_cfg
#include <cstdio>

#include "cutlass/gemm/device/gemm.h"
// Deliberately does NOT include cutlass/util/device_memory.h or the 3.x builder
// headers: this is a pure compile-time reporter, so keeping it dependent on
// nothing but cutlass/gemm/device/gemm.h means it builds with a single -I and no
// -arch, and it cannot break when an unrelated header changes.

using namespace cutlass;

#ifndef LAB_SM
#define LAB_SM 750
#endif
#ifndef LAB_MODE
#define LAB_MODE 0
#endif

#if LAB_SM == 750
using ArchTag = arch::Sm75;
#elif LAB_SM == 800
using ArchTag = arch::Sm80;
#elif LAB_SM == 890
using ArchTag = arch::Sm89;
#elif LAB_SM == 900
using ArchTag = arch::Sm90;
#elif LAB_SM == 1000
using ArchTag = arch::Sm100;
#endif

#if LAB_MODE == 0
using Cfg = gemm::device::DefaultGemmConfiguration<
    arch::OpClassSimt, ArchTag, float, float, float, float>;
static const char* kOpClass = "OpClassSimt";
using LabGemm = gemm::device::Gemm<
    float, layout::RowMajor, float, layout::ColumnMajor,
    float, layout::RowMajor, float,
    arch::OpClassSimt, ArchTag>;
#else
using Cfg = gemm::device::DefaultGemmConfiguration<
    arch::OpClassTensorOp, ArchTag, float, float, float, float>;
static const char* kOpClass = "OpClassTensorOp";
using LabGemm = gemm::device::Gemm<
    float, layout::RowMajor, float, layout::ColumnMajor,
    float, layout::RowMajor, float,
    arch::OpClassTensorOp, ArchTag>;
#endif

int main() {
  using Kernel = typename LabGemm::GemmKernel;
  using Mma = typename Kernel::Mma;

  std::printf("arch                sm_%d\n", LAB_SM);
  std::printf("operator class      %s\n", kOpClass);
  std::printf("threadblock tile    <%d, %d, %d>\n",
              Cfg::ThreadblockShape::kM, Cfg::ThreadblockShape::kN, Cfg::ThreadblockShape::kK);
  std::printf("warp tile           <%d, %d, %d>\n",
              Cfg::WarpShape::kM, Cfg::WarpShape::kN, Cfg::WarpShape::kK);
  std::printf("mma instruction     <%d, %d, %d>\n",
              Cfg::InstructionShape::kM, Cfg::InstructionShape::kN, Cfg::InstructionShape::kK);
  std::printf("pipeline stages     %d\n", int(Cfg::kStages));
  std::printf("alignment A/B       %d / %d\n", int(Cfg::kAlignmentA), int(Cfg::kAlignmentB));
  std::printf("warps per block     %d\n", int(Kernel::Mma::WarpCount::kCount));
  std::printf("threads per block   %d\n", int(Kernel::kThreadCount));
  std::printf("warps M x N         %d x %d\n",
              Cfg::ThreadblockShape::kM / Cfg::WarpShape::kM,
              Cfg::ThreadblockShape::kN / Cfg::WarpShape::kN);
  std::printf("mma per warp k-step %d\n", Cfg::WarpShape::kK / Cfg::InstructionShape::kK);
  std::printf("kernel smem bytes   %d\n", int(sizeof(typename Kernel::SharedStorage)));
  std::printf("  of which epilogue  %d\n", int(sizeof(typename Kernel::Epilogue::SharedStorage)));
  std::printf("mainloop smem bytes %d\n",
              int(sizeof(typename Kernel::SharedStorage)) -
                  int(sizeof(typename Kernel::Epilogue::SharedStorage)));

  // CUTLASS only records a shared-memory capacity constant for the newest archs:
  //   sm100 -> arch.h:45  (232448),  sm107 -> arch.h:46, sm120 -> arch.h:47
  //   sm90  -> gemm/collective/builders/sm90_common.inl:57 (232448)
  // There is no capacity constant for Sm75/Sm80/Sm89 in CUTLASS, so the value
  // for those rungs is left out rather than guessed from the CUDA programming
  // guide.
  int capacity = -1;
  const char* cap_src = "not recorded in CUTLASS";
#if LAB_SM == 900
  // 232448, transcribed from gemm/collective/builders/sm90_common.inl:57
  // (`constexpr int sm90_smem_capacity_bytes = 232448;`). Hard-coded rather than
  // referenced, because pulling in the sm90 3.x builder headers does not compile
  // in a host-only 2.x translation unit.
  capacity = 232448;
  cap_src = "sm90_common.inl:57";
#elif LAB_SM == 1000
  capacity = arch::sm100_smem_capacity_bytes;
  cap_src = "arch.h:45";
#endif
  if (capacity > 0) {
    std::printf("arch smem capacity  %d   (%s)\n", capacity, cap_src);
    std::printf("smem used / capacity %d / %d  (%.1f%%)\n",
                int(sizeof(typename Kernel::SharedStorage)), capacity,
                100.0 * int(sizeof(typename Kernel::SharedStorage)) / capacity);
  } else {
    std::printf("arch smem capacity  %s\n", cap_src);
  }
  return 0;
}
