#!/usr/bin/env python3
"""Rewrite the main() of the CUTLASS 2.x probes with explicit TensorRef construction.

CUTLASS 4.x's `Gemm::Arguments` takes
    (GemmCoord, TensorRef<const ElementA,LayoutA>, TensorRef<const ElementB,LayoutB>,
     TensorRef<const ElementC,LayoutC>, TensorRef<ElementC,LayoutC>,
     EpilogueOutputOp::Params, split_k_slices, ...)
and braced-init-lists do not deduce to it, so the refs are built explicitly here.
The tile shapes / stage count / ArchTag in the type alias above main() are NOT
touched -- those are the experimental subject.
"""
import pathlib
import re

CU = pathlib.Path(__file__).resolve().parent / "cu"

TPL = '''
int main() {
  int M = 1024, N = 1024, K = 1024;
  size_t sa = size_t(M) * K, sb = size_t(K) * N, sc = size_t(M) * N;
  cutlass::device_memory::allocation<ElementA> a(sa * EA_BYTES);
  cutlass::device_memory::allocation<ElementB> b(sb * EB_BYTES);
  cutlass::device_memory::allocation<ElementC> c(sc * EC_BYTES);

  typename GemmTAG::Arguments args(
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
    printf("can_implement failed\\n");
    return 1;
  }
  GemmTAG op;
  return op(args) == cutlass::Status::kSuccess ? 0 : 1;
}
'''

CFG = {
    "gemm_sm80_tensorop.cu": dict(tag="GemmSm80", ea=2, eb=2, ec=2,
                                  acc="ElementAccumulator"),
    "gemm_sm75_tensorop.cu": dict(tag="GemmSm75", ea=2, eb=2, ec=2,
                                  acc="ElementAccumulator"),
    "gemm_simt_fp32.cu":     dict(tag="GemmS", ea=4, eb=4, ec=4, acc="E"),
}

for fname, cfg in CFG.items():
    p = CU / fname
    t = p.read_text()
    head = t[:t.index("int main(")]
    body = (TPL.replace("GemmTAG", cfg["tag"])
                .replace("EA_BYTES", str(cfg["ea"]))
                .replace("EB_BYTES", str(cfg["eb"]))
                .replace("EC_BYTES", str(cfg["ec"]))
                .replace("ElementAccumulator", cfg["acc"]))
    p.write_text(head + body.lstrip())
    print("rewrote", fname)
