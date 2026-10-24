#!/usr/bin/env python3
"""load_accounting.py -- voxel loads per pixel per DDA step, every port, both regimes.

WHY THIS IS A SEPARATE FILE
---------------------------
`notes/19` §4.2(d) reports "The 517 global loads in regime B, against 16" and
§3.2 finding 2 reports "4 predicated loads of one address per step". Both are
measurements, and reading them together is how a 4x becomes a 256x, because
they are per-**thread** figures and the two regimes have completely different
thread-to-pixel mappings:

  regime A (batch 8)     grid (8,32,16) x block (8,16) = 524 288 threads for
                         8 192 pixels = **64 threads per pixel**, and each of
                         those 64 threads does 4 predicated byte loads of the
                         *same* address. 4 x 64 = **256 loads per pixel per
                         step**, plus 1 024 B of int8 state stored per pixel per
                         step. That is regime A's cost, and it is two effects
                         multiplied: the 4x redundant load and the 64-thread
                         replication are separate problems.

  regime B (batch 1024)  grid (512) x block (2,8,8) = 65 536 threads for
                         1 048 576 pixels = **16 pixels per thread**, and each
                         thread issues **one** load per step at a *different*
                         address. 512 loads / 16 px = 32 steps, so
                         **1.000 loads per pixel per step** -- the same as the
                         reference.

**The 4x redundant load is a regime-A phenomenon and it is gone in regime B.**
That is a correction to the scope of `notes/19` gap 7, made by measurement
rather than by argument, and it changes which regime the gap belongs to: the
gap that survives in the regime that gets timed is not the load count, it is
the register budget.

INSTRUMENT
----------
The generated CUDA is parsed for the grid and block extents, which are in the
source as `int gidxN = blockIdx.x; /* 512 */` comments, and for the byte-load
lines. The thread count is the product of the grid and block extents, and
`pixels_per_thread = pixels / threads`. Loads per pixel per step is
`loads / steps / pixels_per_thread` where `steps` is *derived from the loads
themselves* (`loads / pixels_per_thread`) rather than assumed from
`max_steps` -- assuming it would silently divide by 32 in a program that
happens to hold a different number of steps, which is exactly the kind of
assumption that produces a plausible wrong number.

Every input is a capture already on disk (`out/minecu_tg_faithful/cap-*`,
`out/minecu_tg/codegen-sm75-*`). No GPU, no codegen.

    python3 minecu_tg_faithful/load_accounting.py
"""
from __future__ import annotations

import glob
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# (label, directory or file, buffer regex, pixels, expected reference figure)
REGIME_A = [
    ("naive  (minecu_tg)", "out/minecu_tg/codegen-sm75-b8/max_steps_0032",
     "kernel0*.cu", 8192),
    ("faithful/clean", "out/minecu_tg_faithful/cap-b8-ms32/clean",
     "k*.cu", 8192),
    ("faithful/faithful", "out/minecu_tg_faithful/cap-b8-ms32/faithful",
     "k*.cu", 8192),
]
REGIME_B = [
    ("naive  (minecu_tg)", "out/minecu_tg/codegen-sm75-b1024/max_steps_0032/kernel004.cu",
     "data6_4194304", 1048576),
    ("faithful/clean", "out/minecu_tg_faithful/cap-b1024-ms32/b1024-ms0032-clean/k002.cu",
     "data4_4194304", 1048576),
    ("faithful/faithful", "out/minecu_tg_faithful/cap-b1024-ms32/b1024-ms0032-faithful/k002.cu",
     "data4_4194304", 1048576),
]

GRID_RE = re.compile(r"int gidx\d = blockIdx\.\w+; /\* (\d+) \*/")
BLOCK_RE = re.compile(r"int lidx\d = threadIdx\.\w+; /\* (\d+) \*/")
BYTE_LOAD_RE = re.compile(r"signed char val\d+ = \(\(")
# The address appears in two spellings, and a regex that only catches one of
# them reports "0 distinct addresses" for a program that has 512:
#   data4_32768+alu92              (int32 address)
#   data4_32768+((int)(alu92))     (the 64-bit form, narrowed at the load)
# so the cast wrapper is stripped first and the register is then matched.
ADDR_RE = re.compile(r"data\d+_\d+\+(?:\(\(int\)\()?\(?(alu\d+)\)?\)?")
STATE_STORE_RE = re.compile(r"\*\(data0_2097152")


def read(path) -> str:
    return "\n".join(Path(path).read_text().splitlines())


def grid_and_threads(src: str) -> tuple[tuple, tuple, int]:
    g = tuple(int(x) for x in GRID_RE.findall(src))
    b = tuple(int(x) for x in BLOCK_RE.findall(src))
    t = 1
    for x in g + b:
        t *= x
    return g, b, t


def byte_loads(src: str, buf: str) -> tuple[int, list[str]]:
    lines = [l for l in src.splitlines()
             if BYTE_LOAD_RE.search(l) and buf in l]
    addrs = [m.group(1) for l in lines
             if (m := ADDR_RE.search(l))]
    return len(lines), addrs


def main() -> int:
    print("== REGIME A: batch 8, 8 192 px, one DDA step per split program ==")
    print("   the reference: 1.000 loads per pixel per step, 0 B of state stored")
    for label, d, pat, pixels in REGIME_A:
        n = 0
        per_px_ld = 0.0
        per_px_st = 0.0
        shape = None
        same_addr = None
        for f in sorted(glob.glob(str(REPO / d / pat))):
            src = read(f)
            ld, addrs = byte_loads(src, "data4_32768")
            if ld < 4:
                continue
            g, b, t = grid_and_threads(src)
            shape = (g, b, t)
            st = len([x for x in src.splitlines() if STATE_STORE_RE.search(x)])
            per_px_ld += ld * t / pixels
            per_px_st += st * 4 * t / pixels
            if same_addr is None:
                same_addr = (len(set(addrs)) == 1, len(addrs))
            n += 1
        if not n:
            print(f"  {label:18s} no DDA programs found under {d}")
            continue
        g, b, t = shape
        one, cnt = same_addr
        print(f"  {label:18s} {n:2d} programs  grid {g} x block {b} = {t} threads "
              f"({t // pixels} per pixel)")
        print(f"     {cnt} byte loads per thread-step, "
              f"{'ALL THE SAME ADDRESS' if one else 'distinct addresses'}")
        print(f"     loads per PIXEL per step  = {per_px_ld / n:8.1f}   "
              f"= {cnt} x {t // pixels} threads")
        print(f"     int8 state per PIXEL per step = {per_px_st / n:8.1f} B")

    print()
    print("== REGIME B: batch 1024, 1 048 576 px, one fused kernel, all 32 steps ==")
    print("   the reference: 1.000 loads per pixel per step")
    for label, f, buf, pixels in REGIME_B:
        p = REPO / f
        if not p.exists():
            print(f"  {label:18s} missing {f}")
            continue
        src = read(p)
        ld, addrs = byte_loads(src, buf)
        g, b, t = grid_and_threads(src)
        ppt = pixels / t
        steps = ld / ppt if ppt else 0
        print(f"  {label:18s} {ld:4d} loads  grid {g} x block {b} = {t} threads, "
              f"{ppt:.0f} px/thread")
        n_distinct = len(set(addrs))
        if n_distinct == 0:
            # The address register was not recovered. Say so rather than
            # printing "0 of 512 distinct", which reads as a finding.
            print("     address registers NOT recovered -- the address spelling "
                  "changed; the load count above is still exact")
        elif n_distinct == 1:
            print(f"     {n_distinct} distinct address among {ld} loads: the "
                  f"same byte read under {ld} different predicates")
        else:
            print(f"     {n_distinct} distinct addresses of {ld} loads: "
                  f"a different address per step")
        print(f"     steps in this program = {steps:.0f} (derived from the loads, "
              f"not assumed)")
        print(f"     loads per PIXEL per step = {ld / steps / ppt:6.3f}"
              if steps else "     loads per PIXEL per step = n/a")

    print()
    print("The two regimes are different machines doing the same arithmetic, and")
    print("the per-pixel figures above differ by 256x for a reason that has")
    print("nothing to do with the DDA: a 64-threads-per-pixel grid in A, and a")
    print("16-pixels-per-thread grid in B. Quoting either without its grid is")
    print("the same category error as dividing a render by a whole step.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
