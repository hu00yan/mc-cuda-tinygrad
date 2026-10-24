"""np_oc.py - an independent NumPy implementation of blaze/core/obs_camera.h.

Purpose: a third opinion. ref_oc.c is the C truth, tiny_oc.py is the port under
test, and this is a hand-checkable middle that shares no code with either. When
all three disagree, the bug is in the transcription everyone copied; when only
tiny_oc disagrees, it is the port.

Slow and written to look like the C line for line. Float32 throughout, matching
obs_camera.h's "the whole ray path runs in float32" note.
"""
import numpy as np

from tiny_oc import (OC_W, OC_H, OC_NPIX, OC_FAR, OC_TANY, OC_TANX, OC_PI,
                     OC_EDGE_W, RG, sin_table)

f = np.float32


def build_sin_table():
    """(float)sin((double)i*PI*2/65536), same operand order as
    mc_sin_table_init so the bytes match the C reference."""
    i = np.arange(65536, dtype=np.float64)
    return np.sin(((i * np.pi) * 2.0) / 65536.0).astype(np.float32)


def mc_f2i(x):
    """Java float->int narrowing, truncating toward zero."""
    fl = np.floor(x)
    return (fl + ((x < 0) & (x != fl)).astype(np.float32)).astype(np.int32)


def mc_sin(tab, v):
    return tab[mc_f2i(v * f(10430.378)) & 65535]


def mc_cos(tab, v):
    return tab[mc_f2i(v * f(10430.378) + f(16384.0)) & 65535]


def render(cells, ex, ey, ez, yaw_deg, pitch_deg, tab=None):
    """cells: 1-D uint16 packed states. Returns (ids, depth, edge) int arrays."""
    if tab is None:
        tab = build_sin_table()
    NXR, NYR, NZR = RG["nx"], RG["ny"], RG["nz"]

    pix = np.arange(OC_NPIX, dtype=np.int32)
    px, py = pix % OC_W, pix // OC_W

    fex, fey, fez = f(ex), f(ey), f(ez)
    yaw = f(yaw_deg) * f(OC_PI / 180.0)
    pitch = f(pitch_deg) * f(OC_PI / 180.0)
    lx = f(-1) * mc_sin(tab, yaw) * mc_cos(tab, pitch)
    ly = f(-1) * mc_sin(tab, pitch)
    lz = mc_cos(tab, yaw) * mc_cos(tab, pitch)
    hn = f(np.sqrt(lx * lx + lz * lz + f(1e-12)))
    rx, rz = f(-lz / hn), f(lx / hn)
    ux = f(-rz * ly)
    uy = f(rz * lx - rx * lz)
    uz = f(rx * ly)
    ny = f(1.0) - f(f(2.0) * (py.astype(np.float32) + f(0.5)) / f(OC_H))
    nx = f(f(2.0) * (px.astype(np.float32) + f(0.5)) / f(OC_W) - f(1.0))

    ddx = f(lx + nx * OC_TANX * rx + ny * OC_TANY * ux)
    ddy = f(ly + ny * OC_TANY * uy)
    ddz = f(lz + nx * OC_TANX * rz + ny * OC_TANY * uz)
    n = f(np.sqrt(ddx * ddx + ddy * ddy + ddz * ddz))
    dx, dy, dz = f(ddx / n), f(ddy / n), f(ddz / n)

    vx = f(np.floor(fex)) * np.ones(OC_NPIX, np.float32)
    vy = f(np.floor(fey)) * np.ones(OC_NPIX, np.float32)
    vz = f(np.floor(fez)) * np.ones(OC_NPIX, np.float32)
    sx = np.where(dx > 0, f(1), f(-1)).astype(np.float32)
    sy = np.where(dy > 0, f(1), f(-1)).astype(np.float32)
    sz = np.where(dz > 0, f(1), f(-1)).astype(np.float32)
    BIG = f(1e30)
    tdx = np.where(dx != 0, f(np.abs(f(1.0) / dx)), BIG).astype(np.float32)
    tdy = np.where(dy != 0, f(np.abs(f(1.0) / dy)), BIG).astype(np.float32)
    tdz = np.where(dz != 0, f(np.abs(f(1.0) / dz)), BIG).astype(np.float32)
    tmx = np.where(dx != 0, np.where(dx > 0, vx + 1 - fex, fex - vx) * tdx, BIG).astype(np.float32)
    tmy = np.where(dy != 0, np.where(dy > 0, vy + 1 - fey, fey - vy) * tdy, BIG).astype(np.float32)
    tmz = np.where(dz != 0, np.where(dz > 0, vz + 1 - fez, fez - vz) * tdz, BIG).astype(np.float32)

    t = np.zeros(OC_NPIX, np.float32)
    axis = np.full(OC_NPIX, -1, np.int32)
    out = np.zeros(OC_NPIX, np.int32)
    hit_t = np.full(OC_NPIX, f(OC_FAR), np.float32)
    hit_ax = np.full(OC_NPIX, -1, np.int32)
    live = np.ones(OC_NPIX, bool)

    for _ in range(200):
        act = live & (t < OC_FAR)
        if not act.any():
            break
        ix = vx.astype(np.int32) - RG["x0"]
        iy = vy.astype(np.int32) - RG["y0"]
        iz = vz.astype(np.int32) - RG["z0"]
        inb = (ix >= 0) & (iy >= 0) & (iz >= 0) & (ix < NXR) & (iy < NYR) & (iz < NZR)
        lin = (ix * NYR + iy) * NZR + iz
        g = np.where(inb, lin, 0)
        bid = (cells[g] >> 4).astype(np.int32)
        hit = act & (bid != 0)
        out = np.where(hit, bid, out)
        hit_t = np.where(hit, t, hit_t)
        hit_ax = np.where(hit, axis, hit_ax)
        live = live & ~hit
        act = live & (t < OC_FAR)

        cond_x = (tmx < tmy) & (tmx < tmz)
        cond_y = (tmy < tmz)
        mx = act & cond_x
        my = act & ~cond_x & cond_y
        mz = act & ~cond_x & ~cond_y
        t = np.where(mx, tmx, np.where(my, tmy, np.where(mz, tmz, t))).astype(np.float32)
        vx = np.where(mx, vx + sx, vx).astype(np.float32)
        vy = np.where(my, vy + sy, vy).astype(np.float32)
        vz = np.where(mz, vz + sz, vz).astype(np.float32)
        tmx = np.where(mx, tmx + tdx, tmx).astype(np.float32)
        tmy = np.where(my, tmy + tdy, tmy).astype(np.float32)
        tmz = np.where(mz, tmz + tdz, tmz).astype(np.float32)
        axis = np.where(mx, 0, np.where(my, 1, np.where(mz, 2, axis))).astype(np.int32)

    hit_t = np.where(live, f(OC_FAR), hit_t)
    hit_ax = np.where(live, -1, hit_ax)

    d = (hit_t * f(4.0))
    depth = np.where(out != 0, np.where(d > 255, 255, d), 255).astype(np.int32)
    edge = np.zeros(OC_NPIX, np.int32)
    for a, h in enumerate((fex + hit_t * dx, fey + hit_t * dy, fez + hit_t * dz)):
        fr = (h - np.floor(h)).astype(np.float32)
        near = (fr < OC_EDGE_W) | (fr > f(1.0) - OC_EDGE_W)
        tang = hit_ax != a
        edge = np.where((out != 0) & tang & near, 1, edge).astype(np.int32)
    edge = np.where(hit_ax < 0, 0, edge).astype(np.int32)
    return out, depth, edge
