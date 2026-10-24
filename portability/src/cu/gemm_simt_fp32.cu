// fp32 SIMT GEMM: the arch-agnostic fallback. Compiles for every arch, and on
// an old card it is the only thing that will. Included so the matrix has a
// "degrades to what?" answer for the tensor-op rows.

#include <cstdint>
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/util/device_memory.h"

using E = float;
using GemmS = cutlass::gemm::device::Gemm<
    E, cutlass::layout::RowMajor,
    E, cutlass::layout::ColumnMajor,
    E, cutlass::layout::RowMajor,
    E,
    cutlass::arch::OpClassSimt,
    cutlass::arch::Sm50,                     // oldest tag; works everywhere
    cutlass::gemm::GemmShape<128, 128, 8>,
    cutlass::gemm::GemmShape<32, 64, 8>,
    cutlass::gemm::GemmShape<1, 1, 1>,
    cutlass::epilogue::thread::LinearCombination<E, 1,
        E, E, cutlass::epilogue::thread::ScaleType::NoBetaScaling>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;

int main() {
  int M = 1024, N = 1024, K = 1024;
  size_t sa = size_t(M) * K, sb = size_t(K) * N, sc = size_t(M) * N;
  cutlass::device_memory::allocation<ElementA> a(sa * 4);
  cutlass::device_memory::allocation<ElementB> b(sb * 4);
  cutlass::device_memory::allocation<ElementC> c(sc * 4);

  typename GemmS::Arguments args(
      cutlass::gemm::GemmCoord{M, N, K},
      cutlass::TensorRef<ElementA const, cutlass::layout::RowMajor>(
          (ElementA const*)a.get(), M, K, K),
      cutlass::TensorRef<ElementB const, cutlass::layout::ColumnMajor>(
          (ElementB const*)b.get(), K, N, N),
      cutlass::TensorRef<ElementC const, cutlass::layout::RowMajor>(
          (ElementC const*)c.get(), M, N, N),
      cutlass::TensorRef<ElementC, cutlass::layout::RowMajor>(
          (ElementC*)c.get(), M, N, N),
      {E(1.f), E(0.f)});

  if (op.can_implement(args) != cutlass::Status::kSuccess) {
    printf("can_implement failed\n");
    return 1;
  }
  GemmS op;
  return op(args) == cutlass::Status::kSuccess ? 0 : 1;
}
