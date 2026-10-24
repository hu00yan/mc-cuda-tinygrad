"""trace_oc.py - step-by-step trace of ONE ray through the TinyGrad camera.

Same DDA, but prints vx,vy,vz, tmx,tmy,tmz, t, the gathered cell and the take
mask after every unrolled step, for a single pixel. This is how you find out
whether a divergence is in the ray setup, the gather, or the march.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tinygrad import Tensor, dtypes
from tiny_oc import (OC_NPIX, OC_W, OC_H, OC_FAR, OC_STEPS, RG, RG_STRIDE,
                     sin_table, SinTable, load_regions, ray_dirs_batched, f32)
from run_oc import read_ref, CANON, EYE

F, I = dtypes.float32, dtypes.int32
W = lambda c, a, b: Tensor.where(c, a, b)


def main():
    seed, pi, pix = 0, 0, 18 * OC_W + 32      # center pixel
    yaw, pitch = CANON[pi]
    cells = load_regions([f"cells_{seed}.bin"])
    cells_np = cells.numpy()
    sin = SinTable(sin_table())

    dx, dy, dz = ray_dirs_batched([yaw], [pitch], sin)
    dx, dy, dz = dx.numpy()[0][pix], dy.numpy()[0][pix], dz.numpy()[0][pix]
    ex, ey, ez = EYE
    print(f"ray dir = ({dx:.7f}, {dy:.7f}, {dz:.7f})  |d|={np.sqrt(dx*dx+dy*dy+dz*dz):.7f}")

    # scalar mirror of the DDA, in float32, using the SAME ops
    f = np.float32
    vx, vy, vz = f(np.floor(f(ex))), f(np.floor(f(ey))), f(np.floor(f(ez)))
    dx, dy, dz = f(dx), f(dy), f(dz)
    sx = f(1) if dx > 0 else f(-1)
    sy = f(1) if dy > 0 else f(-1)
    sz = f(1) if dz > 0 else f(-1)
    BIG = f(1e30)
    tdx = f(abs(f(1) / dx)) if dx != 0 else BIG
    tdy = f(abs(f(1) / dy)) if dy != 0 else BIG
    tdz = f(abs(f(1) / dz)) if dz != 0 else BIG
    tmx = f(vx + 1 - ex) * tdx if dx > 0 else f(ex - vx) * tdx if dx != 0 else BIG
    tmy = f(vy + 1 - ey) * tdy if dy > 0 else f(ey - vy) * tdy if dy != 0 else BIG
    tmz = f(vz + 1 - ez) * tdz if dz > 0 else f(ez - vz) * tdz if dz != 0 else BIG
    t = f(0)

    ref = read_ref("ref.bin")
    r_id, r_dep, _ = ref[seed * len(CANON) + pi]
    print(f"reference: id={int(r_id[pix])} depth={int(r_dep[pix])}")

    print(f"\n{'k':>3} {'cell':>7} {'id':>4} {'vx,vy,vz':>20} {'t':>8} "
          f"{'tmx':>10} {'tmy':>10} {'tmz':>10}")
    for k in range(OC_STEPS):
        if not (t < OC_FAR):
            print(f"{k:3d}  --- t >= OC_FAR, stop ---")
            break
        ix, iy, iz = int(vx) - RG["x0"], int(vy) - RG["y0"], int(vz) - RG["z0"]
        inb = 0 <= ix < RG["nx"] and 0 <= iy < RG["ny"] and 0 <= iz < RG["nz"]
        cell = cells_np[(ix * RG["ny"] + iy) * RG["nz"] + iz] if inb else 0
        bid = cell >> 4
        if bid:
            print(f"{k:3d} {cell:7d} {bid:4d} {f'{vx},{vy},{vz}':>20} {t:8.3f} "
                  f"{tmx:10.3f} {tmy:10.3f} {tmz:10.3f}   <== HIT, "
                  f"depth={(int)(t*f(4))}")
            return
        if tmx < tmy and tmx < tmz:
            t = tmx; vx = f(vx + sx); tmx = f(tmx + tdx)
        elif tmy < tmz:
            t = tmy; vy = f(vy + sy); tmy = f(tmy + tdy)
        else:
            t = tmz; vz = f(vz + sz); tmz = f(tmz + tdz)
    print("no hit within OC_FAR -> sky")


if __name__ == "__main__":
    main()
