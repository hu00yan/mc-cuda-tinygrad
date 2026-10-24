"""dbg_oc.py - find where the TinyGrad camera diverges from the C reference.

Prints the id map for one frame side by side as ASCII (block id glyph), plus
the first few disagreeing pixels with every intermediate DDA value, so a
divergence shows up as a structural shift rather than a count.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tinygrad import Tensor
from tiny_oc import (OC_NPIX, OC_W, OC_H, RG_STRIDE, render, sin_table,
                     SinTable, load_regions, ray_dirs_batched, f32)
from run_oc import read_ref, CANON, EYE

GLYPH = {0: ".", 1: "#", 2: ":", 3: "o", 4: '"', 16: "c", 17: "l", 18: "w",
         58: "t"}


def show(a, title):
    print(f"--- {title} ---")
    for y in range(OC_H):
        print("".join(GLYPH.get(int(v), "?") for v in a[y * OC_W:(y + 1) * OC_W]))


def main():
    seed, pi = 0, 0
    yaw, pitch = CANON[pi]
    cells_np = np.fromfile(f"cells_{seed}.bin", dtype=np.uint16).astype(np.int32)
    print(f"cells: n={cells_np.size} expect={RG_STRIDE}  max={cells_np.max()}  "
          f"nonzero={int((cells_np != 0).sum())}  "
          f"ids={sorted(set((cells_np >> 4).tolist()))[:14]}")

    ref = read_ref("ref.bin")
    r_id, r_dep, r_edge = ref[seed * len(CANON) + pi]

    sin = SinTable(sin_table())
    cells = Tensor(cells_np)
    ids, depth, edge = render(cells, [EYE], [yaw], [pitch], sin)
    g_id, g_dep, g_edge = ids.numpy()[0], depth.numpy()[0], edge.numpy()[0]

    show(r_id.astype(int), f"REFERENCE (C, seed={seed}, pose {pi}, yaw={yaw} pitch={pitch})")
    show(g_id.astype(int), "TINYGRAD")
    bad = np.nonzero(g_id != r_id)[0]
    print(f"\nmismatched pixels: {len(bad)}/{OC_NPIX}")
    print(f"{'pix':>5} {'xy':>7} {'ref':>5} {'tiny':>5} {'refdep':>7} {'tinydep':>8}")
    for p in bad[:14]:
        print(f"{p:5d} {p%OC_W:3d},{p//OC_W:<3d} {int(r_id[p]):5d} {int(g_id[p]):5d} "
              f"{int(r_dep[p]):7d} {int(g_dep[p]):8d}")

    # where does the sky/non-sky split disagree?
    print(f"\nref sky   = {int((r_id==0).sum())}")
    print(f"tiny sky  = {int((g_id==0).sum())}")
    print(f"ref solid but tiny sky: {int(((r_id!=0)&(g_id==0)).sum())}")
    print(f"tiny solid but ref sky: {int(((g_id!=0)&(r_id==0)).sum())}")
    print(f"both solid, different id: {int(((g_id!=0)&(r_id!=0)&(g_id!=r_id)).sum())}")


if __name__ == "__main__":
    main()
