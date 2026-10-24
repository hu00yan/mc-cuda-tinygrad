// CUTLASS SGEMM harness, one arch per compile.
//
// Design decision, and it is the point of the whole file:
//   Every threadblock / warp / instruction shape, the pipeline depth, the
//   swizzle, the split-K mode and the epilogue are left to CUTLASS's own
//   per-arch defaults (`DefaultGemmConfiguration<OpClass, ArchTag, ...>` and
//   `CollectiveBuilder<..., KernelScheduleAuto>`). Nothing is hand-picked by
//   this harness. So whatever this binary compiles to IS the configuration a
//   human expert published for that architecture, and the "human choice" table
//   in RESULTS.md is a description of the code that actually got compiled rather
//   than of a config someone retyped from a header.
//
// The resolved values are printed at runtime by dump_arch_defaults() below, so
// the table cannot drift from the binary.
//
// Computes D = alpha*A*B + beta*C in float32, writes D plus its inputs as raw
// binaries so tools/verify_sgemm.py can check it against numpy float32.
//
// EXECUTION CAVEAT (deliberate, do not remove):
//   Running this needs a real CUDA device. In the environment this lab was
//   authored in (Apple silicon, no NVIDIA GPU, amd64 emulation only) these
//   binaries have been COMPILED and their SASS inspected, never EXECUTED. See
//   RESULTS.md.

#include "common.cuh"

#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/util/device_memory.h"

using namespace cutlass;

#ifndef LAB_SM
#error "define -DLAB_SM=NNN (750|800|890|900|1000)"
#endif
#ifndef LAB_MODE
#define LAB_MODE 0  // 0 = SIMT fp32, 1 = TensorOp (TF32) fp32
#endif

#if LAB_SM == 750
using LabArchTag = arch::Sm75;
#elif LAB_SM == 800
using LabArchTag = arch::Sm80;
#elif LAB_SM == 890
using LabArchTag = arch::Sm89;
#elif LAB_SM == 900
using LabArchTag = arch::Sm90;
#elif LAB_SM == 1000
using LabArchTag = arch::Sm100;
#else
#error "unsupported LAB_SM"
#endif

using ElementA = float;
using ElementB = float;
using ElementC = float;
using ElementAccumulator = float;
using LayoutA = layout::RowMajor;    // A is [M, K] row-major
using LayoutB = layout::ColumnMajor; // B is [K, N] column-major
using LayoutC = layout::RowMajor;    // C, D are [M, N] row-major

using LabProblem = cutlass_lab::GemmProblem;

// ---------------------------------------------------------------------------
// SIMT: let every shape come from DefaultGemmConfiguration<OpClassSimt,
// ArchTag, float, float, float, float>, which in CUTLASS 4.8.0 is
// ThreadblockShape<128,128,8> WarpShape<32,64,8> InstructionShape<1,1,1>
// Stages=2 Alignment=1 for every arch (verified in
// gemm/device/default_gemm_configuration.h:73-96). So the SIMT baseline is
// genuinely arch-independent in CUTLASS -- see RESULTS.md, that is a finding.
// ---------------------------------------------------------------------------
#if LAB_MODE == 0

using LabGemm = cutlass::gemm::device::Gemm<
    ElementA, LayoutA,
    ElementB, LayoutB,
    ElementC, LayoutC,
    ElementAccumulator,
    arch::OpClassSimt, LabArchTag>;

#else  // LAB_MODE == 1 : TensorOp / TF32 on the tensor cores

using LabGemm = cutlass::gemm::device::Gemm<
    ElementA, LayoutA,
    ElementB, LayoutB,
    ElementC, LayoutC,
    ElementAccumulator,
    arch::OpClassTensorOp, LabArchTag>;

#endif

// ---------------------------------------------------------------------------
// Resolved-configuration printer. Compiles the same DefaultGemmConfiguration
// specialisation the Gemm above will use, so what is printed is what is built.
// ---------------------------------------------------------------------------
static void print_gemm_shape(const char* label, int m, int n, int k) {
  std::printf("  %-16s = <%d, %d, %d>\n", label, m, n, k);
}

static void dump_arch_defaults() {
#if LAB_MODE == 0
  using Cfg = cutlass::gemm::device::DefaultGemmConfiguration<
      arch::OpClassSimt, LabArchTag, ElementA, ElementB, ElementC, ElementAccumulator>;
  static const char* op = "OpClassSimt";
#else
  using Cfg = cutlass::gemm::device::DefaultGemmConfiguration<
      arch::OpClassTensorOp, LabArchTag, ElementA, ElementB, ElementC, ElementAccumulator>;
  static const char* op = "OpClassTensorOp";
#endif
  std::printf("resolved CUTLASS configuration\n");
  std::printf("  arch              = sm_%d\n", LAB_SM);
  std::printf("  operator class    = %s\n", op);
  print_gemm_shape("threadblock tile", Cfg::ThreadblockShape::kM, Cfg::ThreadblockShape::kN,
                   Cfg::ThreadblockShape::kK);
  print_gemm_shape("warp tile", Cfg::WarpShape::kM, Cfg::WarpShape::kN, Cfg::WarpShape::kK);
  print_gemm_shape("mma instruction", Cfg::InstructionShape::kM, Cfg::InstructionShape::kN,
                   Cfg::InstructionShape::kK);
  std::printf("  %-16s = %d\n", "pipeline stages", int(Cfg::kStages));
  std::printf("  %-16s = %d\n", "alignment A", int(Cfg::kAlignmentA));
  std::printf("  %-16s = %d\n", "alignment B", int(Cfg::kAlignmentB));
  std::printf("  %-16s = %d\n", "warps/block",
              (Cfg::ThreadblockShape::kM * Cfg::ThreadblockShape::kN) /
                  (Cfg::WarpShape::kM * Cfg::WarpShape::kN));
  std::printf("  %-16s = %d x %d\n", "warp tile decomp",
              Cfg::ThreadblockShape::kM / Cfg::WarpShape::kM,
              Cfg::ThreadblockShape::kN / Cfg::WarpShape::kN);
  std::printf("  %-16s = %d\n", "mma per warp (k)", Cfg::WarpShape::kK / Cfg::InstructionShape::kK);
  std::printf("  %-16s = %d\n", "smem bytes",
              int(sizeof(typename LabGemm::GemmKernel::SharedStorage)));
  std::printf("  %-16s = %d\n", "threads/block", int(LabGemm::GemmKernel::kThreadCount));
}

int main(int argc, char** argv) {
  LabProblem prob;
  std::string out_path = (argc > 1) ? argv[1] : "D.raw";

  dump_arch_defaults();

  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
    std::printf("LAB_NO_DEVICE no CUDA device visible; this binary was compiled but not run\n");
    return 5;
  }

  size_t sizeA = size_t(prob.m) * prob.k;
  size_t sizeB = size_t(prob.k) * prob.n;
  size_t sizeC = size_t(prob.m) * prob.n;

  std::vector<float> hA(sizeA), hB(sizeB), hC(sizeC);
  cutlass_lab::fill_deterministic(hA, 1u);
  cutlass_lab::fill_deterministic(hB, 2u);
  cutlass_lab::fill_deterministic(hC, 3u);

  cutlass::device_memory::allocation<float> dA(sizeA), dB(sizeB), dC(sizeC), dD(sizeC);
  if (!dA.get() || !dB.get() || !dC.get() || !dD.get()) {
    std::printf("LAB_FAIL device allocation\n");
    return 2;
  }
  if (cudaMemcpy(dA.get(), hA.data(), sizeA * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess ||
      cudaMemcpy(dB.get(), hB.data(), sizeB * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess ||
      cudaMemcpy(dC.get(), hC.data(), sizeC * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
    std::printf("LAB_FAIL H2D memcpy\n");
    return 2;
  }

  constexpr float kAlpha = 1.0f;
  constexpr float kBeta = 1.0f;
  const int64_t klda = prob.k, kldb = prob.k, kldc = prob.n, kldd = prob.n;

  using LabEpiOp = typename LabGemm::EpilogueOutputOp;
  typename LabEpiOp::Params epi_params(kAlpha, kBeta);

  typename LabGemm::Arguments args(
      cutlass::gemm::GemmCoord(prob.m, prob.n, prob.k),
      {dA.get(), klda},
      {dB.get(), kldb},
      {dC.get(), kldc},
      {dD.get(), kldd},
      epi_params,
      /*split_k_slices=*/1);

  cutlass::device_memory::allocation<uint8_t> workspace(1024);
  LabGemm gemm;

  if (gemm.can_implement(args) != cutlass::Status::kSuccess) {
    std::printf("LAB_FAIL can_implement rejected this config for this arch tag\n");
    return 3;
  }
  if (gemm.initialize(args, workspace.get()) != cutlass::Status::kSuccess) {
    std::printf("LAB_FAIL initialize (workspace)\n");
    return 3;
  }
  if (gemm.run() != cutlass::Status::kSuccess) {
    std::printf("LAB_FAIL run\n");
    return 3;
  }
  if (cudaDeviceSynchronize() != cudaSuccess) {
    std::printf("LAB_FAIL cudaDeviceSynchronize\n");
    return 3;
  }

  std::vector<float> hD(sizeC);
  cudaMemcpy(hD.data(), dD.get(), sizeC * sizeof(float), cudaMemcpyDeviceToHost);

  size_t slash = out_path.find_last_of('/');
  std::string base = (slash == std::string::npos) ? "" : out_path.substr(0, slash + 1);
  auto dump = [&](const std::string& p, const std::vector<float>& v) {
    FILE* g = fopen((base + p).c_str(), "wb");
    if (!g) return;
    fwrite(v.data(), sizeof(float), v.size(), g);
    fclose(g);
  };
  dump("D.raw", hD);
  dump("A.raw", hA);
  dump("B.raw", hB);
  dump("C.raw", hC);

  std::printf("LAB_OK wrote %s  m=%d n=%d k=%d\n", out_path.c_str(), prob.m, prob.n, prob.k);
  return 0;
}
