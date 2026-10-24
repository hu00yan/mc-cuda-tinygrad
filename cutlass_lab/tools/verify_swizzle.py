#!/usr/bin/env python3
"""Verify, on the CPU, the shared-memory swizzles CUTLASS uses per architecture.

This is the part of the lab's verification that runs without a GPU, so it is
deliberately narrow and stays inside what can actually be proven.

WHAT IS CHECKED
    BIJECTIVITY. Each swizzle must map distinct logical coordinates to distinct
    byte offsets inside its tile. A non-bijective swizzle silently computes the
    wrong answer, so this is a real correctness check on arithmetic a human wrote
    by hand, not a restatement of what the code does.

WHAT IS DELIBERATELY NOT CHECKED
    Bank-conflict degree. The number of conflicts a given access pattern suffers
    depends on CUTLASS's thread maps (PitchLinearWarpStripedThreadMap /
    PitchLinearStripminedThreadMap) as much as on the swizzle, and modelling
    those faithfully is a separate, error-prone job. An earlier draft of this
    file measured conflicts under a guessed access pattern and produced a number
    that made the swizzle look worse than no swizzle at all -- a clear sign the
    pattern, not the swizzle, was wrong. Rather than publish a number that
    cannot be defended, the measurement was removed. Anyone extending this file
    should port the thread maps from
    include/cutlass/transform/thread_map.h and
    include/cutlass/gemm/threadblock/default_mma_core_sm80.h first.

PROVENANCE
    Every formula below is transcribed from the pinned tag (v4.8.0) with the
    source path and line recorded, so each can be re-checked against the clone.
"""
from __future__ import annotations

import json
import sys

# ---------------------------------------------------------------------------
# sm_75 -- cutlass/layout/tensor_op_multiplicand_sm75.h:407, operator() of the
# congruous operand layout used for InstructionShape<16,8,8> f16. kEPA is 4 for
# the 128-bit access size.
#     offset = (c ^ (2 * s)) * kEPA + s * stride + tc*32 + ts*stride*4 + (contiguous % kEPA)
# with tc = contiguous/32, ts = strided/4, c = (contiguous%32)/kEPA, s = strided%4.
# One XOR term, against the strided index scaled by 2.
# ---------------------------------------------------------------------------
def sm75_offset(contiguous: int, strided: int, stride: int, kEPA: int = 4) -> int:
    tc, ts = contiguous // 32, strided // 4
    c, s = (contiguous % 32) // kEPA, strided % 4
    return (c ^ (2 * s)) * kEPA + s * stride + tc * 32 + ts * stride * 4 + contiguous % kEPA


# ---------------------------------------------------------------------------
# sm_80 / sm_89 -- cutlass/layout/tensor_op_multiplicand_sm80.h:107, operator()
# of TensorOpMultiplicandCongruous64b. Units are 16-bit elements.
#     tc = contiguous/16, ts = strided/4, c = contiguous%16, s = strided%4
#     bank = ((((c & 1) * 4 + (c & 6) / 2)) ^ (s & 1)) * 2 + (c / 8)
#     row  = (c & 6) / 2
#     bank ^= ((s & 2) * 2)
#     offset = tc*16 + bank + (ts*4 + row) * stride
# Two XOR terms, and the base is no longer 1: the "congruous" layouts fold the
# 4-row group into the bank index directly instead of XOR-ing a single column.
# ---------------------------------------------------------------------------
def sm80_offset(contiguous: int, strided: int, stride: int) -> int:
    tc, ts = contiguous // 16, strided // 4
    c, s = contiguous % 16, strided % 4
    bank = ((((c & 1) * 4 + (c & 6) // 2)) ^ (s & 1)) * 2 + (c // 8)
    row = (c & 6) // 2
    bank ^= (s & 2) * 2
    return tc * 16 + bank + (ts * 4 + row) * stride


# ---------------------------------------------------------------------------
# sm_90 and sm_100 -- cute::Swizzle<B, M, S> from include/cute/swizzle.hpp:73:
#     bit_msk = (1 << B) - 1
#     yyy_msk = bit_msk << (M + max(0, S))
#     apply(off) = off ^ ((off & yyy_msk) >> S)
#
# Instantiated in:
#   include/cute/atom/mma_traits_sm90_gmma.hpp:76-84
#       Layout_MN_SW32_Atom_Bits   = Swizzle<1,4,3>  over  256 x 8 bits
#       Layout_MN_SW64_Atom_Bits   = Swizzle<2,4,3>  over  512 x 8 bits
#       Layout_MN_SW128_Atom_Bits  = Swizzle<3,4,3>  over 1024 x 8 bits
#   include/cute/atom/mma_traits_sm100.hpp:73
#       Layout_MN_SW128_32B_Atom_Bits = Swizzle<2,5,2> over 1024 x 4 bits  (new on Blackwell)
#
# Note the parameterisation changed at Hopper: base 4 (16 rows kept fixed)
# instead of 2, and the three widths are three separate atoms the builder
# selects between rather than one formula. Blackwell adds a fourth variant with
# a 4-row base for the 256-bit smem atom its UMMA can consume.
# ---------------------------------------------------------------------------
def cute_swizzle(offset: int, B: int, M: int, S: int) -> int:
    bit_msk = (1 << B) - 1
    yyy_msk = bit_msk << (M + max(0, S))
    return offset ^ ((offset & yyy_msk) >> S)


def check_bijection_2d(fn, n_contig: int, n_strided: int, stride) -> dict:
    seen: dict[int, tuple[int, int]] = {}
    collisions = 0
    for m in range(n_strided):
        for k in range(n_contig):
            off = fn(k, m, stride)
            if off in seen:
                collisions += 1
            else:
                seen[off] = (k, m)
    return {
        "contiguous_extent": n_contig,
        "strided_extent": n_strided,
        "elements": n_contig * n_strided,
        "distinct_offsets": len(seen),
        "collisions": collisions,
        "is_bijection": collisions == 0,
    }


def check_bijection_1d(sw, n: int) -> dict:
    seen: set[int] = set()
    collisions = 0
    for off in range(n):
        s = sw(off)
        if s in seen:
            collisions += 1
        else:
            seen.add(s)
    return {
        "range": n,
        "distinct_images": len(seen),
        "collisions": collisions,
        "is_bijection": collisions == 0 and len(seen) == n,
        "in_range": all(0 <= s < n for s in seen),
    }


def main() -> int:
    out: dict = {}

    out["sm75_congruous"] = {
        "arch": "sm_75",
        "source": "include/cutlass/layout/tensor_op_multiplicand_sm75.h:407",
        "shape": "one XOR term: (c ^ (2*s))",
        "check": check_bijection_2d(sm75_offset, 32 * 4, 4 * 4, 32 * 4),
    }
    out["sm80_congruous64b"] = {
        "arch": "sm_80 / sm_89",
        "source": "include/cutlass/layout/tensor_op_multiplicand_sm80.h:107",
        "shape": "two XOR terms, base-4 row fold into the bank index",
        "check": check_bijection_2d(sm80_offset, 16 * 4, 4 * 4, 16 * 4),
    }

    for label, (B, M, S, nbits) in {
        "sm90_sw32": (1, 4, 3, 256),
        "sm90_sw64": (2, 4, 3, 512),
        "sm90_sw128": (3, 4, 3, 1024),
        "sm100_sw128_32b": (2, 5, 2, 1024),
    }.items():
        out[label] = {
            "arch": "sm_90" if label.startswith("sm90") else "sm_100",
            "source": (
                "include/cute/atom/mma_traits_sm90_gmma.hpp:76-84"
                if label.startswith("sm90")
                else "include/cute/atom/mma_traits_sm100.hpp:73"
            ),
            "shape": f"cute::Swizzle<{B},{M},{S}> over {nbits} bits per row",
            "check": check_bijection_1d(lambda o, B=B, M=M, S=S: cute_swizzle(o, B, M, S), nbits),
        }

    print(json.dumps(out, indent=2))
    print()
    all_ok = True
    for name, spec in out.items():
        c = spec["check"]
        ok = c["is_bijection"] and c.get("in_range", True)
        all_ok = all_ok and ok
        extra = ""
        if "collisions" in c:
            extra = f"collisions={c['collisions']}"
        else:
            extra = f"distinct_images={c['distinct_images']}/{c['range']} in_range={c['in_range']}"
        print(f"{'PASS' if ok else 'FAIL'}  {name:<20} {spec['shape']:<52} {extra}")
    print()
    print("NOTE: bank-conflict degree is NOT reported here; see the module docstring.")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
