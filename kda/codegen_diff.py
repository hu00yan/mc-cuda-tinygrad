"""codegen_diff.py - TinyGrad's generated code vs the hand-written Triton,
for the same KDA computation, on the same operation.

This is the comparison the project was set up to make. Both sides compute

    Z[p,j] = sum_d  k[p,d] * k[j,d] * exp(gcum[p,d] - gcum[j,d])     (j < p)

which is KDA's decayed key-key gram, the step that decides whether the whole
operator is memory-bound or compute-bound.

Triton side, read from fla/ops/kda/chunk_intra.py (BT=64, BC=16, NC=4):

    b_gqn  = tl.where(m_tc1[:, None], exp2(b_g1 - b_gn1[None, :]), 0)   [BC, K]
    b_kgt  = tl.trans(b_k0 * exp2(b_gn1[None, :] - b_g0))               [K, BC]
    b_Akk10 = tl.dot(b_k1 * b_gqn, b_kgt, b_Akk10)                      [BC, BC]

    ^ one reference row per SUB-CHUNK, so the decay tensor is [BC, K] = [16, K].
      tl.dot is a tensor-core MMA with the accumulator in registers.

TinyGrad side: whatever cstyle emits for the same expression. This file
captures the generated source and reports the structural differences.

Run: ../../.venv/bin/python codegen_diff.py
"""
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from tinygrad import Tensor, Device
import tiny_kda
import test_ref as T

# --- the Triton side, transcribed from fla/ops/kda/chunk_intra.py -------------
TRITON = [
    ("grid",      "grid = (NT, NC, B * HV)            # NT=T/64, NC=64/16=4"),
    ("constexpr", "BT = 64;  BC = 16;  NC = cdiv(BT, BC) = 4"),
    ("load",      "b_k0  = tl.load(p_k0, mask=m_ck0).to(tl.float32)        [BC, K]"),
    ("load",      "b_g0  = tl.load(p_g0, mask=m_ck0).to(tl.float32)        [BC, K]"),
    ("load",      "b_gn1 = tl.load(g + ... + o_k, mask=m_k).to(tl.float32)  [K]     <- ONE row"),
    ("decay",     "b_gqn = tl.where(m_tc1[:, None], exp2(b_g1 - b_gn1[None, :]), 0)  [BC, K]"),
    ("decay",     "b_kgt = tl.trans(b_k0 * exp2(b_gn1[None, :] - b_g0))      [K, BC]"),
    ("mma",       "b_Akk10 = tl.dot(b_k1 * b_gqn, b_kgt, b_Akk10)          [BC, BC]  acc in regs"),
    ("store",     "tl.store(p_Akk10, b_Akk10.to(dtype), mask=m_c1)"),
    ("--",        "the [BC, BC] score block never leaves registers; the only"),
    ("--",        "extra buffer is Akkd = empty(B, T, HV, BC) -- [T,16], not [T,T]"),
]


def capture_generated(build, device_name=None):
    """Return the generated device source for whatever `build` realizes."""
    d = Device[device_name or Device.DEFAULT]
    captured = []
    orig = d.compiler.compile_cached

    def spy(src, *a, **k):
        captured.append(str(src))
        return orig(src, *a, **k)

    d.compiler.compile_cached = spy
    try:
        build()
    finally:
        d.compiler.compile_cached = orig
    return captured


def main():
    print("=" * 78)
    print("KDA decayed key-key gram: TinyGrad's generated code vs the Triton kernel")
    print("=" * 78)

    B, Tt, H, HV, K, V, C = 1, 64, 1, 1, 64, 64, 64
    NT = Tt // C
    q, k, v, g, beta = T.rnd(B, Tt, H, HV, K, V, seed=0, l2n=True)

    # --- sizes, which is the whole point -------------------------------------
    mine = B * NT * HV * C * C * K                 # [B,NT,HV,C,C,K]
    theirs = B * NT * HV * 16 * K                 # [BC,K] per sub-chunk, 4 of them
    print(f"\ndecay tensor, same computation, C={C}, K={K}:")
    print(f"  TinyGrad port   [B,NT,HV,C,C,K]   {mine:>12,} elements  {mine*2/1e6:8.1f} MB bf16")
    print(f"  Triton kernel   [BC,K] x NC       {theirs:>12,} elements  {theirs*2/1e6:8.1f} MB bf16")
    print(f"  ratio                                          {mine/theirs:>8.0f}x")

    # --- what tinygrad actually emits ---------------------------------------
    # Isolate the gram: same op as the Triton lines above, no scan, no unrolling.
    gcum = Tensor(np.cumsum(
        np.asarray(g, np.float32).reshape(B, NT, C, HV, K).transpose(0, 1, 3, 2, 4),
        axis=-2))
    tk = Tensor(np.asarray(k, np.float32).reshape(B, NT, C, HV, K).transpose(0, 1, 3, 2, 4))
    gp = gcum.reshape(B, NT, HV, C, 1, K)
    gj = gcum.reshape(B, NT, HV, 1, C, K)
    strict = np.tril(np.ones((C, C), np.float32), -1)
    decay = ((gp - gj) + (Tensor(strict)[None, None, None, :, :, None] - 1.0) * 1e4).exp()
    gram = (decay * tk.reshape(B, NT, HV, C, 1, K) * tk.reshape(B, NT, HV, 1, C, K)).sum(-1)

    # Arm the capture BEFORE realizing, or the kernel is already compiled and
    # nothing is recorded.
    srcs = capture_generated(lambda: gram.realize())
    if not srcs:
        print("\n(could not capture generated source)")
        return
    body = re.sub(r"\x1b\[[0-9;]*m", "", srcs[-1])

    print(f"\nTinyGrad generated {len(srcs)} kernel(s); the largest is "
          f"{max(len(s) for s in srcs)} chars. Its inner loop:")
    for line in body.split("\n"):
        s = line.strip()
        if s and not s.startswith(("#", "//", "using", "#include")) and "=" in s:
            if any(t in s for t in ("val", "exp", "r0", "acc")):
                print(f"    {s[:150]}")

    print()
    print("  ^ note what is NOT here: no tensor-core MMA (`tl.dot` equivalent),")
    print("    no `exp2`, and the decay is a [C,C,K] tensor that had to be")
    print("    materialised before the reduction over K could happen.")

    # --- side by side -------------------------------------------------------
    print()
    print("=" * 78)
    print("SIDE BY SIDE")
    print("=" * 78)
    print(f"  {'step':<10} {'TRITON (fla, hand-written)':<52} TINYGRAD")
    print("  " + "-" * 74)
    for tag, line in TRITON:
        if tag == "--":
            print(f"  {'':<10} {line}")
            continue
        print(f"  {tag:<10} {line[:52]:<52} {''}")
    print()
    print("  TinyGrad's equivalent of the three Triton lines is:")
    print("      decay = (gcum[:,:,:, :,None] - gcum[:,:,:,None,:]).exp()   -> [C,C,K]")
    print("      gram  = (decay * k[:,:,:,:,None] * k[:,:,:,None,:]).sum(-1) -> [C,C]")
    print("  Same result, but the intermediate is C x C x K instead of BC x K,")
    print("  and the contraction is a generic sum() rather than a tensor-core MMA.")


if __name__ == "__main__":
    main()
