"""
AOT-compile Triton kernels for every compute capability, no GPU present.

This is the Triton half of the portability matrix.  It uses `triton.compile`
with an explicit GPUTarget, so the only thing that varies is the target arch.

Kernel ladder (deliberately ordered by how modern the mechanism is):
  copy         plain ld/st -- the floor
  dot_f16      tl.dot fp16xfp16->fp32, the workhorse
  dot_f16_f16acc  tl.dot with fp16 accumulate
  dot_i8       tl.dot int8
  dot_f8e4nv   tl.dot fp8 (e4m3) x fp8 -> fp32
  dot_f8_scaled tl.dot_scaled, the fp8/fp4 path
  tma_load     tl.make_tensor_descriptor + descriptor load  (sm_90+)
  tma_store    descriptor store
  warp_spec    tl.async_task / warp-specialisation hints

Run inside a container that has the triton wheel installed.
"""
import json
import os
import pathlib
import re
import sys
import traceback

import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource, compile as triton_compile

OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "out/triton")
OUT.mkdir(parents=True, exist_ok=True)

CAPS = [70, 75, 80, 86, 87, 89, 90, 100, 101, 120]


# ------------------------------------------------------------------ kernels
@triton.jit
def k_copy(X, Y, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    o = pid * BLOCK + tl.arange(0, BLOCK)
    m = o < n
    tl.store(Y + o, tl.load(X + o, mask=m), mask=m)


@triton.jit
def k_dot_f16(A, B, C, M, N, K,
              BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pm = tl.program_id(0)
    pn = tl.program_id(1)
    om = pm * BM + tl.arange(0, BM)
    on = pn * BN + tl.arange(0, BN)
    ok = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        a = tl.load(A + om[:, None] * K + (k + ok)[None, :])
        b = tl.load(B + (k + ok)[:, None] * N + on[None, :])
        acc = tl.dot(a, b, acc)
    tl.store(C + om[:, None] * N + on[None, :], acc.to(tl.float16))


@triton.jit
def k_dot_f16_f16acc(A, B, C, M, N, K,
                     BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pm = tl.program_id(0)
    pn = tl.program_id(1)
    om = pm * BM + tl.arange(0, BM)
    on = pn * BN + tl.arange(0, BN)
    ok = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.float16)
    for k in range(0, K, BK):
        a = tl.load(A + om[:, None] * K + (k + ok)[None, :])
        b = tl.load(B + (k + ok)[:, None] * N + on[None, :])
        acc = tl.dot(a, b, acc, out_dtype=tl.float16)
    tl.store(C + om[:, None] * N + on[None, :], acc)


@triton.jit
def k_dot_i8(A, B, C, M, N, K,
             BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
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


@triton.jit
def k_dot_f8e4nv(A, B, C, M, N, K,
                 BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pm = tl.program_id(0)
    pn = tl.program_id(1)
    om = pm * BM + tl.arange(0, BM)
    on = pn * BN + tl.arange(0, BN)
    ok = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        a = tl.load(A + om[:, None] * K + (k + ok)[None, :])
        b = tl.load(B + (k + ok)[:, None] * N + on[None, :])
        acc = tl.dot(a, b, acc)
    tl.store(C + om[:, None] * N + on[None, :], acc.to(tl.bfloat16))


@triton.jit
def k_dot_f8_scaled(A, B, C, SA, SB, M, N, K,
                    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pm = tl.program_id(0)
    pn = tl.program_id(1)
    om = pm * BM + tl.arange(0, BM)
    on = pn * BN + tl.arange(0, BN)
    ok = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        a = tl.load(A + om[:, None] * K + (k + ok)[None, :])
        b = tl.load(B + (k + ok)[:, None] * N + on[None, :])
        sa = tl.load(SA + om[:, None] * K + (k + ok)[None, :])
        sb = tl.load(SB + (k + ok)[:, None] * N + on[None, :])
        acc = tl.dot_scaled(a, sa, "e4m3", b, sb, "e4m3", acc)
    tl.store(C + om[:, None] * N + on[None, :], acc.to(tl.bfloat16))


@triton.jit
def k_tma_load(X, OUT, M, N, BM: tl.constexpr, BN: tl.constexpr):
    # Device-side TMA: Triton 3.3+ builds the tensormap in the kernel.
    desc = tl.make_tensor_descriptor(X, shape=[M, N], strides=[N, 1],
                                     block_shape=[BM, BN])
    a = desc.load([0, 0])
    tl.store(OUT + tl.arange(0, BM)[:, None] * BN + tl.arange(0, BN)[None, :], a)


@triton.jit
def k_tma_store(X, OUT, M, N, BM: tl.constexpr, BN: tl.constexpr):
    desc = tl.make_tensor_descriptor(OUT, shape=[M, N], strides=[N, 1],
                                     block_shape=[BM, BN])
    x = tl.load(X + tl.arange(0, BM)[:, None] * BN + tl.arange(0, BN)[None, :])
    desc.store([0, 0], x)


# ------------------------------------------------------------------ signatures
f16 = "*fp16"
i8 = "*i8"
f8 = "*fp8e4nv"
i32 = "*i32"
bf16 = "*bf16"

KERNELS = {
    "copy":        (k_copy,       {"X": f16, "Y": f16, "n": "i32", "BLOCK": "constexpr"},
                    {"BLOCK": 64}),
    "dot_f16":     (k_dot_f16,    {"A": f16, "B": f16, "C": f16, "M": "i32", "N": "i32",
                                  "K": "i32", "BM": "constexpr", "BN": "constexpr",
                                  "BK": "constexpr"},
                    {"BM": 64, "BN": 64, "BK": 32}),
    "dot_f16_f16acc": (k_dot_f16_f16acc,
                    {"A": f16, "B": f16, "C": f16, "M": "i32", "N": "i32", "K": "i32",
                     "BM": "constexpr", "BN": "constexpr", "BK": "constexpr"},
                    {"BM": 64, "BN": 64, "BK": 32}),
    "dot_i8":      (k_dot_i8,     {"A": i8, "B": i8, "C": i32, "M": "i32", "N": "i32",
                                  "K": "i32", "BM": "constexpr", "BN": "constexpr",
                                  "BK": "constexpr"},
                    {"BM": 64, "BN": 64, "BK": 64}),
    "dot_f8e4nv":  (k_dot_f8e4nv, {"A": f8, "B": f8, "C": bf16, "M": "i32", "N": "i32",
                                  "K": "i32", "BM": "constexpr", "BN": "constexpr",
                                  "BK": "constexpr"},
                    {"BM": 64, "BN": 64, "BK": 32}),
    "dot_f8_scaled": (k_dot_f8_scaled,
                    {"A": f8, "B": f8, "C": bf16, "SA": f8, "SB": f8,
                     "M": "i32", "N": "i32", "K": "i32",
                     "BM": "constexpr", "BN": "constexpr", "BK": "constexpr"},
                    {"BM": 64, "BN": 64, "BK": 32}),
    "tma_load":    (k_tma_load,   {"X": f16, "OUT": f16, "M": "i32", "N": "i32",
                                  "BM": "constexpr", "BN": "constexpr"},
                    {"BM": 64, "BN": 64}),
    "tma_store":   (k_tma_store,  {"X": f16, "OUT": f16, "M": "i32", "N": "i32",
                                  "BM": "constexpr", "BN": "constexpr"},
                    {"BM": 64, "BN": 64}),
}

# SASS/PTX instruction mix, per class.
MIX = {
    "wgmma":     r"\bwgmma\b",
    "tcgen05":   r"\btcgen05\b",
    "mma.sync":  r"\bmma\.sync\b",
    "FMA.f32":   r"\bfma\.rn\.f32\b",
    "cp.async":  r"\bcp\.async\b",
    "tensormap": r"\btensormap\b",
    "mbarrier":  r"\bmbarrier\b",
    "ldmatrix":  r"\bldmatrix\b",
    "stmatrix":  r"\bstmatrix\b",
    "cluster":   r"\b(cluster\.|mapa\.)",
    "elect":     r"\belect\b",
    "bar.sync":  r"\bbar\.sync\b",
}


def classify(ptx: str):
    return {k: len(re.findall(p, ptx, re.I)) for k, p in MIX.items()
            if re.search(p, ptx, re.I)}


def main():
    results = []
    print(f"triton {triton.__version__}", flush=True)
    for name, (fn, sig, cexprs) in KERNELS.items():
        for cap in CAPS:
            rec = {"kernel": name, "cap": cap, "status": "", "error": "",
                   "ptx_insts": {}, "ptx_shasum": ""}
            try:
                src = ASTSource(fn=fn, signature=sig, constexprs=cexprs)
                target = GPUTarget("cuda", cap, 32)
                k = triton_compile(src, target=target, options={"num_warps": 4})
                ptx = k.asm["ptx"]
                rec["status"] = "OK"
                rec["ptx_insts"] = classify(ptx)
                rec["ptx_shasum"] = __import__("hashlib").sha1(ptx.encode()).hexdigest()[:12]
                rec["regs"] = getattr(k, "n_regs", None)
                rec["spills"] = getattr(k, "n_spills", None)
                rec["shared"] = getattr(k, "metadata", None) and \
                    getattr(k.metadata, "shared", None)
                (OUT / f"{name}__sm{cap}.ptx").write_text(ptx)
            except Exception as e:
                rec["status"] = "REJECT"
                msg = re.sub(r"\s+", " ", f"{type(e).__name__}: {e}")
                rec["error"] = msg[:400]
            results.append(rec)
            mix = ", ".join(f"{a}x{b}" for a, b in rec["ptx_insts"].items()) or "-"
            print(f"{name:<16} sm_{cap:<4} {rec['status']:<7} {mix}"
                  f"{'  ' + rec['error'][:110] if rec['error'] else ''}", flush=True)

    (OUT / "triton_ladder.json").write_text(json.dumps(
        {"triton_version": triton.__version__, "cells": results}, indent=1))
    ok = sum(r["status"] == "OK" for r in results)
    print(f"\ncells: {len(results)}  compiled: {ok}  rejected: {len(results)-ok}")
    print(f"wrote {OUT/'triton_ladder.json'}")


if __name__ == "__main__":
    main()
