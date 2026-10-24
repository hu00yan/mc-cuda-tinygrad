/*
 * CUTLASS hand-written baseline for sm_90a (Hopper).
 *
 * PROVENANCE
 * ----------
 * Every line of the mainloop, the collective schedule, the TMA copy atoms and
 * the mma atoms is CUTLASS 4.8.0. I did not write a kernel; I reproduced the
 * *type construction* from CUTLASS's own shipped example
 *
 *     cutlass/examples/49_hopper_gemm_with_collective_builder/49_collective_builder.cu
 *
 * verbatim (same layouts, same element types, same alignments, same
 * Shape<_128,_128,_64> threadblock tile, same Shape<_2,_1,_1> cluster, same
 * KernelScheduleAuto / EpilogueTileAuto / StageCountAutoCarveout). I took that
 * example's collective construction rather than writing my own so that the
 * "hand-written" side of this study is literally NVIDIA's recommended Hopper
 * GEMM.
 *
 * The only things I changed relative to the example are mechanical:
 *   - dropped its host-side test harness (HostTensor, curand fill, reference
 *     comparison) so we can compile device-only with no CUDA runtime;
 *   - a `main()` that constructs Arguments and calls the adapter's operator()
 *     purely so clang marks the __global__ as used. Nothing is linked or run.
 *
 * Why sm_90a and not sm_90: wgmma and the Hopper TMA path are arch-conditional
 * and only assemble for the `a` target. ptxas 12.9 supports sm_90a.
 *
 * KernelScheduleAuto is important to the experiment: it makes CUTLASS choose
 * its own mainloop schedule (here: TMA + warp-specialised wgmma) and its own
 * stage count. That is the human state of the art, not a configuration I
 * tuned.
 *
 * LIMITATION: CUTLASS's published defaults, not autotuned on this machine (no
 * GPU). Valid for an instruction-class comparison; not a TFLOP/s claim.
 */

#include <cstdio>
#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"

using namespace cute;

// --- copied from examples/49_hopper_gemm_with_collective_builder ---------
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::ColumnMajor;
using LayoutD = cutlass::layout::ColumnMajor;

using ElementA = cutlass::half_t;
using ElementB = cutlass::half_t;
using ElementC = cutlass::half_t;
using ElementD = cutlass::half_t;
using ElementAccumulator = float;
using ElementCompute = float;

static constexpr int AlignmentA = 16 / sizeof(ElementA);   // 8: 16B -> enables TMA
static constexpr int AlignmentB = 16 / sizeof(ElementB);
static constexpr int AlignmentC = 16 / sizeof(ElementC);
static constexpr int AlignmentD = 16 / sizeof(ElementD);

using EpilogueScheduleType = cutlass::epilogue::collective::EpilogueScheduleAuto;
using MainloopScheduleType = cutlass::gemm::collective::KernelScheduleAuto;
using StageCountType       = cutlass::gemm::collective::StageCountAuto;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    Shape<_128, _128, _64>, Shape<_1, _1, _1>,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementCompute,
    ElementC, LayoutC, AlignmentC,
    ElementD, LayoutD, AlignmentD,
    EpilogueScheduleType
  >::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    ElementA, LayoutA, AlignmentA,
    ElementB, LayoutB, AlignmentB,
    ElementAccumulator,
    Shape<_128, _128, _64>, Shape<_2, _1, _1>,
    cute::conditional_t<cute::is_same_v<StageCountType,
        cutlass::gemm::collective::StageCountAuto>,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
        StageCountType>,
    MainloopScheduleType
  >::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

// CUTLASS 3.x GemmUniversal::Arguments is an aggregate of defaulted members
// (mode, problem_shape, mainloop, epilogue, hw_info, scheduler), so we
// default-construct it and set the two fields that matter. We never run it --
// initialize() + run() are called purely so clang marks the __global__ kernel
// as used and emits it. Pointers stay null; nothing is linked.
template <class T>
void mark_used(int m, int n, int k) {
  using Kernel = typename T::GemmKernel;
  typename T::Arguments args{};
  args.mode = cutlass::gemm::GemmUniversalMode::kGemm;
  args.problem_shape = typename Kernel::ProblemShape{m, n, k, 1};
  args.hw_info.sm_count = 1;
  T op;
  (void)op.initialize(args, nullptr);
  (void)op.run();
}

int main() {
  mark_used<Gemm>(4096, 4096, 4096);
  return 0;
}
