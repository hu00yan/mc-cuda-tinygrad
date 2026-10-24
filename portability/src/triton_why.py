"""Capture Triton's verbatim diagnostics for a rejected (arch, kernel) cell."""
import re
import sys

import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
from triton.compiler import compile as tcompile


@triton.jit
def k_dot_i8(A, B, C, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pm = tl.program_id(0)
    pn = tl.program_id(1)
    om = pm * BM + tl.arange(0, BM)
    on = pn * BN + tl.arange(0, BN)
    ok = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.int32)
    for k in range(0, K, BK):
        a = tl.load(A + om[:, None] * K + (k + ok)[None, :])
        b = tl.load(B + (k + ok)[:, None] * N + on[None, :])
        acc = tl.dot(a, b, acc, out_dtype=tl.int32)
    tl.store(C + om[:, None] * N + on[None, :], acc)


sig = {"A": "*i8", "B": "*i8", "C": "*i32", "M": "i32", "N": "i32", "K": "i32",
       "BM": "constexpr", "BN": "constexpr", "BK": "constexpr"}
cex = {"BM": 64, "BN": 64, "BK": 64}

cap = int(sys.argv[1])
try:
    tcompile(ASTSource(fn=k_dot_i8, signature=sig, constexprs=cex),
             target=GPUTarget("cuda", cap, 32), options={"num_warps": 4})
    print(f"sm_{cap}: OK")
except Exception as e:
    txt = str(e)
    print(f"sm_{cap}: {type(e).__name__}")
    for line in txt.splitlines():
        if re.search(r"error|assert|require|capab|sm_|80|75|only|support", line, re.I):
            print("   ", re.sub(r"\s+", " ", line).strip()[:220])
