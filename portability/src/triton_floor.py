"""Find Triton's actual minimum accepted compute capability, and its fp8 gate text.

Three questions, three measurements:
  1. Does the Python/MLIR layer refuse any capability at all?  Sweep 35..90.
  2. What is the verbatim fp8 diagnostic (it names the supported set)?
  3. Which bundled ptxas does Triton use, and what can IT target?
     (Triton's ceiling is its own ptxas, not the system nvcc.)
"""
import pathlib
import re
import subprocess
import sys

import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
from triton.compiler import compile as tcompile

OUT = pathlib.Path("/w/out/triton")


@triton.jit
def k_copy(X, Y, n, BLOCK: tl.constexpr):
    o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = o < n
    tl.store(Y + o, tl.load(X + o, mask=m), mask=m)


@triton.jit
def k_dot_f16(A, B, C, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pm, pn = tl.program_id(0), tl.program_id(1)
    om, on = pm * BM + tl.arange(0, BM), pn * BN + tl.arange(0, BN)
    ok = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        a = tl.load(A + om[:, None] * K + (k + ok)[None, :])
        b = tl.load(B + (k + ok)[:, None] * N + on[None, :])
        acc = tl.dot(a, b, acc)
    tl.store(C + om[:, None] * N + on[None, :], acc.to(tl.float16))


@triton.jit
def k_dot_f8(A, B, C, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pm, pn = tl.program_id(0), tl.program_id(1)
    om, on = pm * BM + tl.arange(0, BM), pn * BN + tl.arange(0, BN)
    ok = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        a = tl.load(A + om[:, None] * K + (k + ok)[None, :])
        b = tl.load(B + (k + ok)[:, None] * N + on[None, :])
        acc = tl.dot(a, b, acc)
    tl.store(C + om[:, None] * N + on[None, :], acc.to(tl.bfloat16))


SIGS = {
    "copy": (k_copy, {"X": "*fp16", "Y": "*fp16", "n": "i32", "BLOCK": "constexpr"},
             {"BLOCK": 64}),
    "dot_f16": (k_dot_f16, {"A": "*fp16", "B": "*fp16", "C": "*fp16", "M": "i32",
                            "N": "i32", "K": "i32", "BM": "constexpr",
                            "BN": "constexpr", "BK": "constexpr"},
                {"BM": 64, "BN": 64, "BK": 32}),
    "dot_f8": (k_dot_f8, {"A": "*fp8e4nv", "B": "*fp8e4nv", "C": "*bf16", "M": "i32",
                          "N": "i32", "K": "i32", "BM": "constexpr",
                          "BN": "constexpr", "BK": "constexpr"},
               {"BM": 64, "BN": 64, "BK": 32}),
}

print(f"### triton {triton.__version__}")
print("\n### 1. capability sweep (does anything get refused outright?)")
for name, (fn, sig, cex) in SIGS.items():
    line = f"{name:<10}"
    for cap in [35, 50, 60, 70, 75, 80, 89, 90, 100, 120]:
        try:
            k = tcompile(ASTSource(fn=fn, signature=sig, constexprs=cex),
                         target=GPUTarget("cuda", cap, 32), options={"num_warps": 4})
            n_fma = len(re.findall(r"\bfma\.rn\.f32\b", k.asm["ptx"]))
            n_mma = len(re.findall(r"\bmma\.sync\b", k.asm["ptx"]))
            n_wgm = len(re.findall(r"\bwgmma\b", k.asm["ptx"]))
            n_tcg = len(re.findall(r"\btcgen05\b", k.asm["ptx"]))
            tag = ("wgmma" if n_wgm else "tcgen05" if n_tcg else
                   "mma.sync" if n_mma else f"FMAx{n_fma}" if n_fma else "-")
            line += f"  {cap}:{tag}"
        except Exception as e:
            first = re.sub(r"\s+", " ", str(e).splitlines()[0])[:34]
            line += f"  {cap}:REJECT"
    print(line)

print("\n### 2. verbatim fp8 diagnostic at sm_80")
try:
    fn, sig, cex = SIGS["dot_f8"]
    tcompile(ASTSource(fn=fn, signature=sig, constexprs=cex),
             target=GPUTarget("cuda", 80, 32), options={"num_warps": 4})
    print("   (compiled -- no diagnostic)")
except Exception as e:
    for ln in str(e).splitlines():
        if "fp8" in ln or "supported" in ln:
            print("   ", re.sub(r"\s+", " ", ln).strip()[:400])

print("\n### 3. Triton's bundled ptxas ceiling")
import triton.backends.nvidia as bn
ptxas = pathlib.Path(bn.__file__).parent / "bin" / "ptxas"
r = subprocess.run([str(ptxas), "--help"], capture_output=True, text=True)
arches = re.search(r"Allowed values for this option:\s*(.+?)\n\n", r.stdout, re.S)
if arches:
    got = sorted(set(re.findall(r"'(compute_\d+[a-z]?)'", arches.group(1))),
                 key=lambda s: int(re.sub(r"\D", "", s)))
    print(f"   bundled ptxas targets: {' '.join(got)}")
    print(f"   max compute_ arch: {got[-1]}")
r2 = subprocess.run([str(ptxas), "--version"], capture_output=True, text=True)
print("  ", [l for l in r2.stdout.splitlines() if "release" in l])
