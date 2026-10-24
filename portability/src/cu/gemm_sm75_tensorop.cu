// Same operator, but with the ArchTag a Turing/Volta user would actually choose.
// This is the honest "state of the art for sm_75" control: HMMA m16n8k8, and
// a 2-stage (non-async) mainloop, because cp.async does not exist on sm_75.

#include <cstdint>
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/util/device_memory.h"

using ElementA = cutlass::half_t;
using ElementB = cutlass::half_t;
using ElementC = cutlass::half_t;
using ElementAccumulator = float;

using GemmSm75 = cutlass::gemm::device::Gemm<
    ElementA, cutlass::layout::RowMajor,
    ElementB, cutlass::layout::ColumnMajor,
    ElementC, cutlass::layout::RowMajor,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm75,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 8>,     // m16n8k8 = the Turing instruction
    cutlass::epilogue::thread::LinearCombination<ElementC, 8,
        ElementAccumulator, ElementAccumulator,
        cutlass::epilogue::thread::ScaleType::NoBetaScaling>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    2                                       // no cp.async -> pipelined, not multistage
>;

int main() {
  int M = 1024, N = 1024, K = 1024;
  size_t sa = size_t(M) * K, sb = size_t(K) * N, sc = size_t(M) * N;
  cutlass::device_memory::allocation<ElementA> a(sa * 2);
  cutlass::device_memory::allocation<ElementB> b(sb * 2);
  cutlass::device_memory::allocation<ElementC> c(sc * 2);

  typename GemmSm75::Arguments args(
      cutlass::gemm::GemmCoord{M, N, K},
      cutlass::TensorRef<ElementA const, cutlass::layout::RowMajor>(
          (ElementA const*)a.get(), M, K, K),
      cutlass::TensorRef<ElementB const, cutlass::layout::ColumnMajor>(
          (ElementB const*)b.get(), K, N, N),
      cutlass::TensorRef<ElementC const, cutlass::layout::RowMajor>(
          (ElementC const*)c.get(), M, N, N),
      cutlass::TensorRef<ElementC, cutlass::layout::RowMajor>(
          (ElementC*)c.get(), M, N, N),
      {ElementAccumulator(1.f), ElementAccumulator(0.f)});

  if (op.can_implement(args) != cutlass::Status::kSuccess) {
    printf("can_implement failed\n");
    return 1;
  }
  GemmSm75 op;
  return op(args) == cutlass::Status::kSuccess ? 0 : 1;
}
