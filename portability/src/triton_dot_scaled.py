"""`tl.dot_scaled` (the fp8/fp4 block-scaled path) across capabilities.

First attempt used a/w scale shapes equal to the operand shapes, which is wrong:
the scale tensors are per-32-element *blocks*, i.e. [M, K//32] and [K//32, N].
This script uses the correct shapes so that a rejection is attributable to the
arch rather than to my kernel.
"""
import pathlib
import re

import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
from triton.compiler import compile as tcompile

OUT = pathlib.Path("/w/out/triton")
BM, BN, BK = 64, 64, 64          # BK=64 so one 32-elem scale block == 2 k-iterations
KS = BK // 32                    # scale extent along K


@triton.jit
def k_dot_scaled_f8(A, B, C, SA, SB, M, N, K,
                    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pm, pn = tl.program_id(0), tl.program_id(1)
    om, on = pm * BM + tl.arange(0, BM), pn * BN + tl.arange(0, BN)
    ok = tl.arange(0, BK)
    oks = tl.arange(0, BK // 32)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        a = tl.load(A + om[:, None] * K + (k + ok)[None, :])
        b = tl.load(B + (k + ok)[:, None] * N + on[None, :])
        sa = tl.load(SA + om[:, None] * (K // 32) + (k // 32 + oks)[None, :])
        sb = tl.load(SB + (k // 32 + oks)[:, None] * N + on[None, :])
        acc = tl.dot_scaled(a, sa, "e4m3", b, sb, "e4m3", acc)
    tl.store(C + om[:, None] * N + on[None, :], acc.to(tl.bfloat16))


sig = {"A": "*fp8e4nv", "B": "*fp8e4nv", "C": "*bf16", "SA": "*fp8e4nv",
       "SB": "*fp8e4nv", "M": "i32", "N": "i32", "K": "i32",
       "BM": "constexpr", "BN": "constexpr", "BK": "constexpr"}
cex = {"BM": BM, "BN": BN, "BK": BK}

print(f"### triton {triton.__version__}  tl.dot_scaled ladder")
for cap in [80, 86, 89, 90, 100, 120]:
    try:
        k = tcompile(ASTSource(fn=k_dot_scaled_f8, signature=sig, constexprs=cex),
                     target=GPUTarget("cuda", cap, 32), options={"num_warps": 4})
        ptx = k.asm["ptx"]
        mix = {n: len(re.findall(p, ptx, re.I)) for n, p in
               (("wgmma", r"\bwgmma\b"), ("tcgen05", r"\btcgen05\b"),
                ("mma.sync", r"\bmma\.sync\b"), ("mma.scaled", r"\bmma\.sync.*scale"),
                ("FMA", r"\bfma\.rn\.f32\b")).items() if re.search(p, ptx, re.I)}
        (OUT / f"dot_scaled__sm{cap}.ptx").write_text(ptx)
        print(f"  sm_{cap:<4} OK      {mix}")
    except Exception as e:
        first = re.sub(r"\s+", " ", str(e).splitlines()[0])[:120]
        print(f"  sm_{cap:<4} REJECT  {first}")
