"""trace2_oc.py - the TENSOR DDA traced step by step for one pixel, beside a
scalar numpy mirror of the same DDA, so the first divergence is one table row.

Per-step values are collected as tensors and realized ONCE at the end: calling
.numpy() inside the loop realizes the whole graph every step, which is
O(steps^2) and takes minutes.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tinygrad import Tensor, dtypes
from tiny_oc import OC_FAR, OC_STEPS, RG, sin_table, SinTable, ray_dirs_batched, f32
from run_oc import read_ref, CANON, EYE

F, I = dtypes.float32, dtypes.int32
W = lambda c, a, b: Tensor.where(c, a, b)
X = lambda a: Tensor(np.array([[a]], dtype=np.float32))
NX, NY, NZ = RG["nx"], RG["ny"], RG["nz"]


def scalar_first_hit(cells_np, ex, ey, ez, dx, dy, dz):
    """The C DDA, in float32, returning (step, vx,vy,vz, t, cell, id)."""
    f = np.float32
    vx, vy, vz = f(np.floor(f(ex))), f(np.floor(f(ey))), f(np.floor(f(ez)))
    dx, dy, dz = f(dx), f(dy), f(dz)
    sx = f(1) if dx > 0 else f(-1)
    sy = f(1) if dy > 0 else f(-1)
    sz = f(1) if dz > 0 else f(-1)
    tdx, tdy, tdz = f(abs(f(1) / dx)), f(abs(f(1) / dy)), f(abs(f(1) / dz))
    tmx = f(vx + 1 - ex) * tdx if dx > 0 else f(ex - vx) * tdx
    tmy = f(vy + 1 - ey) * tdy if dy > 0 else f(ey - vy) * tdy
    tmz = f(vz + 1 - ez) * tdz if dz > 0 else f(ez - vz) * tdz
    t = f(0)
    for k in range(OC_STEPS):
        if not (t < OC_FAR):
            return k, (vx, vy, vz), float(t), 0, 0
        ix, iy, iz = int(vx) - RG["x0"], int(vy) - RG["y0"], int(vz) - RG["z0"]
        inb = 0 <= ix < NX and 0 <= iy < NY and 0 <= iz < NZ
        cell = int(cells_np[(ix * NY + iy) * NZ + iz]) if inb else 0
        if cell >> 4:
            return k, (vx, vy, vz), float(t), cell, cell >> 4
        if tmx < tmy and tmx < tmz:
            t = tmx; vx = f(vx + sx); tmx = f(tmx + tdx)
        elif tmy < tmz:
            t = tmy; vy = f(vy + sy); tmy = f(tmy + tdy)
        else:
            t = tmz; vz = f(vz + sz); tmz = f(tmz + tdz)
    return OC_STEPS, (vx, vy, vz), float(t), 0, 0


def main():
    seed, pi = 0, 0
    pix = int(sys.argv[1]) if len(sys.argv) > 1 else 1168
    yaw, pitch = CANON[pi]
    cells_np = np.fromfile(f"cells_{seed}.bin", dtype=np.uint16).astype(np.int32)
    cells = Tensor(cells_np)
    sin = SinTable(sin_table())

    dx, dy, dz = ray_dirs_batched([yaw], [pitch], sin)
    dx, dy, dz = [x.numpy()[0][pix] for x in (dx, dy, dz)]
    ref = read_ref("ref.bin")
    r_id, r_dep, _ = ref[seed * len(CANON) + pi]
    print(f"pixel {pix} ({pix%64},{pix//64})  ray=({dx:.6f},{dy:.6f},{dz:.6f})")
    print(f"C reference : id={int(r_id[pix])} depth={int(r_dep[pix])}")
    sk, svox, stt, scell, sbid = scalar_first_hit(cells_np, EYE[0], EYE[1], EYE[2], dx, dy, dz)
    print(f"numpy mirror: step={sk} id={sbid} t={stt:.3f} depth={int(np.float32(stt)*np.float32(4))}"
          f"  v={tuple(round(float(v),3) for v in svox)}\n")

    ex, ey, ez = EYE
    vx, vy, vz = X(np.floor(ex)), X(np.floor(ey)), X(np.floor(ez))
    Dx, Dy, Dz = X(dx), X(dy), X(dz)
    BIG = f32(1e30)
    tdx = W(Dx != 0, (f32(1.0) / Dx).abs(), BIG)
    tdy = W(Dy != 0, (f32(1.0) / Dy).abs(), BIG)
    tdz = W(Dz != 0, (f32(1.0) / Dz).abs(), BIG)
    tmx = W(Dx != 0, W(Dx > 0, vx + 1 - X(ex), X(ex) - vx) * tdx, BIG)
    tmy = W(Dy != 0, W(Dy > 0, vy + 1 - X(ey), X(ey) - vy) * tdy, BIG)
    tmz = W(Dz != 0, W(Dz > 0, vz + 1 - X(ez), X(ez) - vz) * tdz, BIG)
    sx = W(Dx > 0, 1.0, -1.0); sy = W(Dy > 0, 1.0, -1.0); sz = W(Dz > 0, 1.0, -1.0)
    t = X(0.0)
    alive = Tensor.ones(1, 1, dtype=dtypes.bool)
    hit_id, hit_t = X(0.0), X(0.0)
    zI = Tensor.zeros(1, 1, dtype=I)

    cols = []
    for k in range(OC_STEPS):
        in_range = t < OC_FAR
        go = in_range & alive
        ix = (vx - f32(RG["x0"])).cast(I); iy = (vy - f32(RG["y0"])).cast(I); iz = (vz - f32(RG["z0"])).cast(I)
        inb = (ix >= 0) & (iy >= 0) & (iz >= 0) & (ix < NX) & (iy < NY) & (iz < NZ)
        lin = (ix * NY + iy) * NZ + iz
        cell = cells[W(inb, lin, zI).reshape(-1)].reshape(1, 1)
        bid = cell >> 4
        take = in_range & alive & (bid != 0)
        hit_id = W(take, bid, hit_id); hit_t = W(take, t, hit_t)
        alive = alive & ~take
        cols.append(Tensor.stack([vx.reshape(()), vy.reshape(()), vz.reshape(()), t.reshape(()),
                                 cell.reshape(()).cast(F), bid.reshape(()).cast(F),
                                 inb.reshape(()).cast(F), take.reshape(()).cast(F)]))
        c_x = tmx < tmy; c_y = tmy < tmz
        mv_x = go & c_x & c_y; mv_y = go & ~c_x & c_y; mv_z = go & ~c_x & ~c_y
        t = W(mv_x, tmx, W(mv_y, tmy, W(mv_z, tmz, t)))
        vx = W(mv_x, vx + sx, vx); vy = W(mv_y, vy + sy, vy); vz = W(mv_z, vz + sz, vz)
        tmx = W(mv_x, tmx + tdx, tmx); tmy = W(mv_y, tmy + tdy, tmy); tmz = W(mv_z, tmz + tdz, tmz)
        if k > sk + 3:
            break

    A = Tensor.stack(cols).numpy()
    print(f"{'k':>3} | {'vx,vy,vz (tensor)':>22} {'t':>8} {'inb':>4} {'cell':>6} {'id':>3} {'take':>5}")
    for k in range(len(A)):
        v = (int(A[k, 0]), int(A[k, 1]), int(A[k, 2]))
        print(f"{k:3d} | {str(v):>22} {A[k,3]:8.3f} {int(A[k,6]):4d} {int(A[k,4]):6d} "
              f"{int(A[k,5]):3d} {int(A[k,7]):5d}" + ("   <== tensor hit" if A[k, 7] else ""))
    print(f"\ntensor: hit_id={float(hit_id.numpy()[0,0]):.0f} t={float(hit_t.numpy()[0,0]):.3f}")


if __name__ == "__main__":
    main()
