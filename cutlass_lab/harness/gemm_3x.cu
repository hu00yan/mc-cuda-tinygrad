// CUTLASS 3.x collective path: the wgmma (sm_90a) and tcgen05 (sm_100a) kernels.
//
// WHY THIS IS A SEPARATE FILE
//   The 2.x device::Gemm path used by sgemm.cu has no wgmma and no tcgen05: those
//   instructions are reached only through the 3.x CollectiveBuilder, which is
//   what CUTLASS 3.x/4.x actually ships for Hopper and Blackwell. If the study
//   wants "the best Hopper kernel" and "the best Blackwell kernel" as the human
//   reference points, this is the file that has to compile.
//
//   A FINDING WORTH FLAGGING TO THE STUDY
//   Unlike the 2.x path, the 3.x CollectiveBuilder has NO default threadblock
//   tile. Its template is
//       CollectiveBuilder<ArchTag, OpClass, ElementA, LayoutA, AlignA,
//                          ElementB, LayoutB, AlignB, ElementAccumulator,
//                          TileShape_MNK, ClusterShape_MNK,
//                          StageCountType, BuilderScheduleTag>
//   with no defaults on TileShape_MNK / ClusterShape_MNK / StageCountType. Only
//   the *schedule* is auto-selectable (KernelScheduleAuto picks TMA-vs-cp.async
//   and cooperative-vs-pingpong; see sm90_gmma_builder.inl:998-1029). So on
//   Hopper and Blackwell the human's choice is a named (tile, cluster, stages)
//   triple, and CUTLASS offers no single "the default" to point at. The values
//   below are therefore CHOSEN, and labelled as such, not quoted.
//
// CONFIGURATION PROVENANCE
//   TileShape 128x256x64 with a 1x1x1 cluster and auto-staged pipeline. The
//   128x256x64 threadblock tile is the one CUTLASS's own recorded default for
//   Ampere f16 (gemm/device/default_gemm_configuration.h, arch::Sm80 ->
//   ThreadblockShape<128,256,64>), reused here so that the tile does not move
//   between rungs of the ladder and confound the arch comparison. The stage
//   count is left to StageCountAutoCarveout so CUTLASS derives it from the
//   hardware shared-memory capacity. This is a choice made by this file, and
//   the study should treat it as one number among many that a tuner sweeps.
//
// EXECUTION CAVEAT
//   Needs a real CUDA device. Built but not run in the environment this was
//   authored in. See RESULTS.md.

#include "common.cuh"

#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.h"
#include "cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized.hpp"
#include "cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/util/device_memory.h"

using namespace cutlass;

#ifndef LAB_SM
#error "define -DLAB_SM=900 or 1000"
#endif

#if LAB_SM == 900
using LabArchTag = arch::Sm90;
#elif LAB_SM == 1000
using LabArchTag = arch::Sm100;
#else
#error "this file targets sm_90a and sm_100a only"
#endif

using ElementA = half_t;
using ElementB = half_t;
using ElementC = half_t;
using ElementAcc = float;
using LayoutA = layout::RowMajor;
using LayoutB = layout::ColumnMajor;
using LayoutC = layout::RowMajor;

// THE TILE IS A COMPILE-TIME PARAMETER, AND WHICH TILES CUTLASS ACCEPTS IS
// ARCH-DEPENDENT. That boundary is itself a result, so it is swept rather than
// guessed (see build_3x.sh and RESULTS.md).
//   sm_90a accepts 128x256x64 f16 with StageCountAutoCarveout; CUTLASS derives
//   4 stages from the 232,448 B smem capacity.
//   sm_100a REJECTS 128x256x64 AND 128x128x64 with
//       sm100_gemm_tma_warpspecialized.hpp:410: static assertion failed with
//       "SMEM usage exceeded capacity."
//   Blackwell's warp-specialised kernel carries more state in the same 232,448 B
//   (tensor-memory allocation, a separate epilogue-load pipeline, an MMA-transform
//   stage), so a tile that fits on Hopper does not transfer.
#ifndef LAB_TILE_M
#define LAB_TILE_M 128
#endif
#ifndef LAB_TILE_N
#define LAB_TILE_N 256
#endif
#ifndef LAB_TILE_K
#define LAB_TILE_K 64
#endif

using TileShape = cute::Shape<cute::Int<LAB_TILE_M>, cute::Int<LAB_TILE_N>, cute::Int<LAB_TILE_K>>;
using ClusterShape = cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>;
using EpilogueTileType = cutlass::epilogue::collective::EpilogueTileAuto;

// 4.x takes the tile scheduler as a bare tag and resolves it against the
// mainloop's DispatchPolicy inside kernel::GemmUniversal. StreamK is what
// CUTLASS itself selects for these archs in its 3.x examples.
using TileSchedulerTag = cutlass::gemm::StreamKScheduler;

// NOTE / CONSTRAINT DISCOVERED IN CUTLASS 4.8.0: the 3.x GEMM kernel takes the
// problem shape as a COMPILE-TIME type, not a runtime value. In
// kernel::GemmUniversal (sm90_gemm_tma_warpspecialized.hpp:77) `ProblemShape` is
// just the template argument, and for a GEMM
// conv::detail::get_problem_shape_MNKL_helper just returns it unchanged
// (conv/detail.hpp:49-51) rather than translating a conv problem shape. So the
// M/N/K must be baked into the type. That is itself worth recording: it means
// CUTLASS 4.8's warp-specialised GEMM path does not by itself support a
// runtime-sized problem without another wrapper.
using ProblemShape = cute::Shape<
    cute::Int<cutlass_lab::kM>, cute::Int<cutlass_lab::kN>, cute::Int<cutlass_lab::kK>>;

// 128-bit alignment in elements for 16-bit types. Spelled as a literal because
// `sizeof_bits` is ambiguous once both cutlass and cute are in scope.
static constexpr int kAlignA = 8;
static constexpr int kAlignB = 8;
static_assert(sizeof(ElementA) == 2 && sizeof(ElementB) == 2, "this file is f16-specific");

using MainloopSmemStageCount =
#if defined(LAB_STAGES)
    cutlass::gemm::collective::StageCount<LAB_STAGES>;
#else
    cutlass::gemm::collective::StageCountAutoCarveout<
        sizeof(typename cutlass::PipelineTmaAsync<1>::SharedStorage)>;
#endif

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    LabArchTag, arch::OpClassTensorOp,
    ElementA, LayoutA, kAlignA,
    ElementB, LayoutB, kAlignB,
    ElementAcc,
    TileShape, ClusterShape,
    MainloopSmemStageCount,
#if defined(LAB_MMA_TRANSFORM)
    // Blackwell's "MMA transform" schedule: a dedicated warpgroup converts A to
    // tensor memory, the UMMA warpgroup runs tcgen05, and the epilogue drains
    // tmem. This is the schedule that matches the 5-role WarpCategory split
    // documented in sm100_gemm_tma_warpspecialized.hpp:137-146.
    cutlass::gemm::KernelTmaWarpSpecialized1SmSm100
#else
    cutlass::gemm::collective::KernelScheduleAuto   // <-- CUTLASS picks the schedule
#endif
    >::CollectiveOp;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    LabArchTag, arch::OpClassTensorOp,
    TileShape, ClusterShape,
    EpilogueTileType,
    ElementAcc, ElementAcc,
    ElementC, LayoutC, kAlignA,
    ElementC, LayoutC, kAlignA,
    cutlass::epilogue::collective::EpilogueScheduleAuto
    >::CollectiveOp;

using LabGemmKernel = cutlass::gemm::kernel::GemmUniversal<
    ProblemShape,
    CollectiveMainloop,
    CollectiveEpilogue,
    TileSchedulerTag>;

using LabGemm = cutlass::gemm::device::GemmUniversalAdapter<LabGemmKernel>;

// The epilogue Arguments struct has a member named `thread`, but its type
// differs by arch, and some of those types are aggregates rather than classes
// with a (alpha, beta) constructor. Two small helpers cover all the variants
// CUTLASS 4.8 uses:
//   * epilogue::thread::LinearCombination::Params   -> has a 2-arg ctor
//   * epilogue::fusion::FusionCallbacks::Arguments  -> aggregate, brace-init
template <class Epi, class = void>
struct LabEpiThreadArgsImpl {
  using type = typename Epi::ThreadEpilogueOp::Params;
  static type make(float a, float b) { return type(a, b); }
};
template <class Epi>
struct LabEpiThreadArgsImpl<Epi, cute::void_t<typename Epi::FusionCallbacks>> {
  using type = typename Epi::FusionCallbacks::Arguments;
  static type make(float a, float b) {
    typename Epi::FusionCallbacks::Arguments args{};
    args.alpha = a;
    args.beta = b;
    return args;
  }
};
template <class Epi>
using LabEpiThreadArgs = LabEpiThreadArgsImpl<Epi>;

int main(int argc, char** argv) {
  cutlass_lab::GemmProblem prob;
  std::string out_path = (argc > 1) ? argv[1] : "D.raw";

  using Kernel = typename LabGemm::GemmKernel;
  using Params = typename Kernel::Params;

  using TiledMma = typename CollectiveMainloop::TiledMma;
  // The adapter re-exposes the 3.x kernel's resolved shape as 2.x-shaped types,
  // which is exactly the "what did CUTLASS pick" data the study needs.
  std::printf("resolved CUTLASS 3.x configuration\n");
  std::printf("  arch              = sm_%d\n", LAB_SM);
  std::printf("  threadblock tile  = <%d, %d, %d>  (CHOSEN by this file)\n",
              LabGemm::ThreadblockShape::kM, LabGemm::ThreadblockShape::kN,
              LabGemm::ThreadblockShape::kK);
  std::printf("  warp tile (approx) = <%d, %d, %d>  (adapter estimate)\n",
              LabGemm::WarpShape::kM, LabGemm::WarpShape::kN, LabGemm::WarpShape::kK);
  std::printf("  mma atom          = <%d, %d, %d>  (CUTLASS-selected)\n",
              LabGemm::InstructionShape::kM, LabGemm::InstructionShape::kN,
              LabGemm::InstructionShape::kK);
  std::printf("  cluster           = <%d, %d, %d>  (CHOSEN by this file)\n",
              LabGemm::ClusterShape::kM, LabGemm::ClusterShape::kN, LabGemm::ClusterShape::kK);
  std::printf("  mainloop stages   = %d  (CUTLASS-derived from smem capacity)\n",
              int(CollectiveMainloop::DispatchPolicy::Stages));
  std::printf("  threads/block     = %d\n", int(LabGemm::kThreadCount));
  std::printf("  smem bytes        = %d\n", int(LabGemmKernel::SharedStorageSize));
  std::printf("  tile scheduler    = StreamK (CUTLASS default for this arch)\n");
  std::printf("  align A/B/C/D     = %d / %d / %d / %d\n", int(LabGemm::kAlignmentA),
              int(LabGemm::kAlignmentB), int(LabGemm::kAlignmentC), int(LabGemm::kAlignmentD));

  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
    std::printf("LAB_NO_DEVICE no CUDA device visible; compiled but not run\n");
    return 5;
  }

  const size_t sizeA = size_t(prob.m) * prob.k;
  const size_t sizeB = size_t(prob.k) * prob.n;
  const size_t sizeC = size_t(prob.m) * prob.n;

  std::vector<float> fA(sizeA), fB(sizeB), fC(sizeC);
  cutlass_lab::fill_deterministic(fA, 1u);
  cutlass_lab::fill_deterministic(fB, 2u);
  cutlass_lab::fill_deterministic(fC, 3u);
  // f16 inputs: round the deterministic values to half so the device and the
  // numpy reference see bit-identical inputs.
  std::vector<ElementA> hA(sizeA), hB(sizeB);
  std::vector<ElementC> hC(sizeC);
  for (size_t i = 0; i < sizeA; ++i) hA[i] = static_cast<ElementA>(fA[i]);
  for (size_t i = 0; i < sizeB; ++i) hB[i] = static_cast<ElementB>(fB[i]);
  for (size_t i = 0; i < sizeC; ++i) hC[i] = static_cast<ElementC>(fC[i]);

  cutlass::device_memory::allocation<ElementA> dA(sizeA), dB(sizeB);
  cutlass::device_memory::allocation<ElementC> dC(sizeC), dD(sizeC);
  if (!dA.get() || !dB.get() || !dC.get() || !dD.get()) {
    std::printf("LAB_FAIL device allocation\n");
    return 2;
  }
  if (cudaMemcpy(dA.get(), hA.data(), sizeA * sizeof(ElementA), cudaMemcpyHostToDevice) != cudaSuccess ||
      cudaMemcpy(dB.get(), hB.data(), sizeB * sizeof(ElementB), cudaMemcpyHostToDevice) != cudaSuccess ||
      cudaMemcpy(dC.get(), hC.data(), sizeC * sizeof(ElementC), cudaMemcpyHostToDevice) != cudaSuccess) {
    std::printf("LAB_FAIL H2D memcpy\n");
    return 2;
  }

  constexpr float kAlpha = 1.0f, kBeta = 1.0f;
  const int64_t klda = prob.k, kldb = prob.k, kldc = prob.n;

  // 3.x kernels take the mainloop and epilogue arguments separately. Strides are
  // cute tuples ordered to match the (M, N, K, L) problem the collective builds
  // internally, with L == 1 here.
  // These Arguments are aggregates (no user ctor), so they need brace-init;
  // parenthesised init of an aggregate is C++20 only.
  typename CollectiveMainloop::Arguments mainloop_args{
      /*ptr_A=*/dA.get(),
      /*dA=*/cute::make_tuple(int64_t(prob.k), cute::Int<1>{}, int64_t(prob.m) * prob.k),
      /*ptr_B=*/dB.get(),
      /*dB=*/cute::make_tuple(int64_t(prob.k), cute::Int<1>{}, int64_t(prob.n) * prob.k)};

  typename LabEpiThreadArgs<CollectiveEpilogue>::type thread_args =
      LabEpiThreadArgs<CollectiveEpilogue>::make(kAlpha, kBeta);
  typename CollectiveEpilogue::Arguments epilogue_args{
      /*thread=*/thread_args,
      /*ptr_C=*/dC.get(),
      /*dC=*/cute::make_tuple(int64_t(prob.n), cute::Int<1>{}, int64_t(prob.m) * prob.n),
      /*ptr_D=*/dD.get(),
      /*dD=*/cute::make_tuple(int64_t(prob.n), cute::Int<1>{}, int64_t(prob.m) * prob.n)};

  // The problem shape is compile-time (see the ProblemShape alias above), so the
  // runtime GemmCoord is NOT passed; the kernel reads M/N/K out of the type.
  // The cooperative/pingpong kernels give Arguments no user constructor, so all
  // six fields must be brace-initialised positionally.
  typename LabGemm::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm,
      ProblemShape{},
      mainloop_args,
      epilogue_args,
      cutlass::KernelHardwareInfo{},
      typename LabGemmKernel::TileSchedulerArguments{}};

  cutlass::device_memory::allocation<uint8_t> workspace(size_t(LabGemm::get_workspace_size(args)));
  if (!workspace.get()) {
    std::printf("LAB_FAIL workspace allocation (%zu bytes)\n", size_t(LabGemm::get_workspace_size(args)));
    return 2;
  }

  LabGemm gemm;
  if (gemm.can_implement(args) != cutlass::Status::kSuccess) {
    std::printf("LAB_FAIL can_implement rejected this config\n");
    return 3;
  }
  if (gemm.initialize(args, workspace.get()) != cutlass::Status::kSuccess) {
    std::printf("LAB_FAIL initialize\n");
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
  cudaMemcpy(hD.data(), dD.get(), sizeC * sizeof(ElementC), cudaMemcpyDeviceToHost);

  size_t slash = out_path.find_last_of('/');
  std::string base = (slash == std::string::npos) ? "" : out_path.substr(0, slash + 1);
  auto dump = [&](const std::string& p, const void* v, size_t n) {
    FILE* g = fopen((base + p).c_str(), "wb");
    if (!g) return;
    fwrite(v, 1, n, g);
    fclose(g);
  };
  dump("D.raw", hD.data(), sizeC * sizeof(float));
  dump("A.raw", hA.data(), sizeA * sizeof(ElementA));
  dump("B.raw", hB.data(), sizeB * sizeof(ElementB));
  dump("C.raw", hC.data(), sizeC * sizeof(ElementC));

  std::printf("LAB_OK wrote %s  m=%d n=%d k=%d\n", out_path.c_str(), prob.m, prob.n, prob.k);
  return 0;
}
