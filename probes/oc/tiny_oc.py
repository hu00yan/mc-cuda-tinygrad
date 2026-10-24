"""tiny_oc.py - the blaze semantic camera (core/obs_camera.h) in TinyGrad.

WHAT NETHERITE DOES
    blaze/cuda/obs_camera.cu runs the camera one thread per (pose, pixel).
    Each thread walks an Amanatides-Woo DDA through the voxel region with the
    march state (vx,vy,vz, tmx,tmy,tmz, t, axis) in thread-local registers, and
    loops until it hits. That is the whole trick, and it is why their box
    reports ~396M rays/s.

WHAT THIS DOES
    The same DDA, as a fixed-length unroll of S steps over a [B, 2304] ray
    tensor, march state carried in tensors. Control flow becomes data:
      - "still marching"      -> alive mask
      - "hit a non-air block" -> take mask
      - while (t < OC_FAR)    -> S iterations gated by (t < OC_FAR)
    A DDA ray is monotone and its step count is bounded, so the unroll is exact,
    not an approximation. The unroll is a superset: rays that hit at step 3
    spend 93 more steps as no-ops. render(..., want_steps=True) returns the
    per-ray step count so that waste factor is measurable, not a guess.

WHY S = 96 IS A HARD BOUND
    Cell crossings to travel distance L with a unit direction are at most
    L*(|dx|+|dy|+|dz|)+3, and |d|_2 = 1 caps the sum at sqrt(3). At OC_FAR = 48
    that is 48*sqrt(3)+3 = 86.1. 96 is safe with margin.
"""
import numpy as np
from tinygrad import Tensor, dtypes

# ---- constants, transcribed from blaze/core/obs_camera.h -------------------
OC_W, OC_H, OC_NPIX = 64, 36, 64 * 36
OC_FAR = 48.0
OC_TANY = float(np.float32(0.7002075382097097))
# C folds 64.0f/36.0f in float, so compute OC_TANX in float32 before widening.
OC_TANX = float(np.float32(np.float32(0.7002075382097097)
                           * np.float32(np.float32(64.0) / np.float32(36.0))))
OC_PI = 3.14159265358979323846
OC_EDGE_W = float(np.float32(0.05))
OC_STEPS = 96

# ---- region geometry, matching blaze/cuda/obs_camera.cu -------------------
RG = dict(x0=-48, y0=28, z0=-48, nx=113, ny=104, nz=113)
RG_STRIDE = RG["nx"] * RG["ny"] * RG["nz"]


def f32(x):
    """Scalar constant coercion.

    tinygrad raises in ufix() on a bare np.float32, and infers float32 for a
    python float, so every scalar constant goes through float(). That is
    bit-identical for anything numpy already rounded to float32 above (OC_TANX,
    OC_EDGE_W) and correctly rounded for the rest, matching the C literals
    (OC_FAR is 48.0f, the trig index scale is 10430.378f, and so on)."""
    return float(x)


F = dtypes.float32
I = dtypes.int32


def sin_table() -> Tensor:
    """MC 1.11.2 MathHelper SIN_TABLE = (float)sin((double)i*PI*2/65536), built
    with the SAME operand order as mc_sin_table_init so the bytes match the C
    reference: (i*PI)*2/65536, not i*(PI*2)/65536."""
    i = np.arange(65536, dtype=np.float64)
    return Tensor(np.sin(((i * np.pi) * 2.0) / 65536.0).astype(np.float32))


def _f2i(x: Tensor) -> Tensor:
    """Java float->int narrowing (JLS 5.1.3): truncate toward zero, with
    negatives rounding the C way (C99 says plain (int) is UB out of range, so
    the reference emulates it). Done in float, not via .cast(int), because the
    result feeds a gather and we want the index path to stay in one domain."""
    fl = x.floor()
    return (fl + ((x < 0) & (x != fl)).cast(F)).cast(I)


def _fmod65536(v: Tensor) -> Tensor:
    """C's `idx & 65535` on a possibly-negative int."""
    fv = v.cast(F)
    return (fv - f32(65536.0) * (fv / f32(65536.0)).floor()).cast(I)


class SinTable:
    """mc_sin / mc_cos over the LUT, kept in-graph."""

    def __init__(self, table: Tensor):
        self.t = table.reshape(-1)

    def sin(self, v: Tensor) -> Tensor:
        return self.t[_fmod65536(_f2i(v * f32(10430.378)))]

    def cos(self, v: Tensor) -> Tensor:
        return self.t[_fmod65536(_f2i(v * f32(10430.378) + f32(16384.0)))]


# ---- FMA contraction barrier ----------------------------------------------
#
# blaze/SPEC.md rule 4 requires -ffp-contract=off / --fmad=false so CPU and CUDA
# agree bit for bit. TinyGrad emits plain `a * b + c` and the Metal compiler
# contracts it; there is no knob. Measured on 500,000 random float32 triples:
#
#   plain                vs non-fused 135,993 differ   vs exact fma         0 differ
#   product.realize()    vs non-fused       0 differ   vs exact fma 135,993 differ
#   operands.contiguous() vs non-fused 135,993 differ  vs exact fma         0 differ
#
# So forcing the PRODUCT to materialise through HBM is a complete and exact
# barrier; copying the operands is not. `_barrier` is that, and it costs one
# extra kernel per protected product, so it is only worth paying where a
# product feeds an add.
BARRIER = [False]


def _barrier(t: Tensor) -> Tensor:
    return t.realize() if BARRIER[0] else t


def ray_dirs_batched(yaws, pitches, sin: SinTable):
    """Per-pixel unit ray directions for B poses at once -> (dx, dy, dz), each
    [B, OC_NPIX].

    Transcribed expression-for-expression from oc_pixel, including the 1e-12
    inside hn: it is load-bearing for the exact zeros of the normalize, and
    dropping it changes which pixels land on a block boundary.

    The pose terms are [B,1] and the pixel terms are [1,OC_NPIX], so the whole
    setup is one broadcast instead of a Python loop over envs. That matters:
    a per-env loop would multiply the LUT gathers and the frame math by B."""
    yaw = Tensor(np.asarray(yaws, dtype=np.float32))[:, None]
    pitch = Tensor(np.asarray(pitches, dtype=np.float32))[:, None]
    fyaw = yaw * f32(OC_PI / 180.0)
    fpitch = pitch * f32(OC_PI / 180.0)
    lx = -sin.sin(fyaw) * sin.cos(fpitch)
    ly = -sin.sin(fpitch)
    lz = sin.cos(fyaw) * sin.cos(fpitch)
    hn = (_barrier(lx * lx) + _barrier(lz * lz) + f32(1e-12)).sqrt()
    rx, rz = -lz / hn, lx / hn
    ux = -rz * ly
    uy = _barrier(rz * lx) - _barrier(rx * lz)
    uz = rx * ly

    pix = Tensor.arange(OC_NPIX, dtype=I)
    px = (pix % OC_W).cast(F)[None, :]
    py = (pix // OC_W).cast(F)[None, :]
    ny = f32(1.0) - f32(2.0) * (py + f32(0.5)) / f32(OC_H)
    nx = f32(2.0) * (px + f32(0.5)) / f32(OC_W) - f32(1.0)

    ddx = lx + _barrier(nx * OC_TANX * rx) + _barrier(ny * OC_TANY * ux)
    ddy = ly + _barrier(ny * OC_TANY * uy)
    ddz = lz + _barrier(nx * OC_TANX * rz) + _barrier(ny * OC_TANY * uz)
    n = (_barrier(ddx * ddx) + _barrier(ddy * ddy) + _barrier(ddz * ddz)).sqrt()
    return ddx / n, ddy / n, ddz / n


def render(cells_flat: Tensor, eyes, yaws, pitches, sin: SinTable,
           steps: int = OC_STEPS, want_steps: bool = False):
    """One 64x36 frame per env, batched.

    cells_flat : 1-D int32 of length B*RG_STRIDE; env b owns
                 [b*RG_STRIDE, (b+1)*RG_STRIDE). Values are PACKED BLOCK
                 STATES ((id<<4)|meta) exactly as blaze stores them, so the
                 >>4 that oc_block does stays in the port.
    eyes       : [B,3] float32
    yaws       : [B] float32 degrees
    pitches    : [B] float32 degrees
    """
    B = len(yaws)
    BIG = f32(1e30)

    dx, dy, dz = ray_dirs_batched(yaws, pitches, sin)
    eyes = np.asarray(eyes, dtype=np.float32)
    ex = Tensor(eyes[:, 0])[:, None]
    ey = Tensor(eyes[:, 1])[:, None]
    ez = Tensor(eyes[:, 2])[:, None]

    zeros = Tensor.zeros(B, OC_NPIX, dtype=F)
    ones = Tensor.ones(B, OC_NPIX, dtype=F)
    falses = Tensor.zeros(B, OC_NPIX, dtype=dtypes.bool)
    trues = Tensor.ones(B, OC_NPIX, dtype=dtypes.bool)
    W = lambda c, a, b: Tensor.where(c, a, b)

    # ---- DDA prologue (oc_raycast) ----
    vx, vy, vz = ex.floor(), ey.floor(), ez.floor()
    nz_x, nz_y, nz_z = dx != 0, dy != 0, dz != 0
    pos_x, pos_y, pos_z = dx > 0, dy > 0, dz > 0
    sx, sy, sz = W(pos_x, 1.0, -1.0), W(pos_y, 1.0, -1.0), W(pos_z, 1.0, -1.0)
    # oc_raycast reads `fabsf(1.0f / dx)`.
    #
    # `/` is the one float32 operation this port cannot make bit-exact, and the
    # reason is not the backend: TinyGrad's Tensor API lowers `/` to
    # `a * fl32(1/b)` above the scheduler and above backend selection, so the
    # result is a correctly rounded reciprocal followed by one correctly
    # rounded float32 multiply, and it lands 1 ULP off the correctly rounded
    # quotient in 25.9%-27.0% of general `a/b` cases -- a rate that travels
    # with the pair (a, b), not with the operation, and that no user-facing knob
    # changes. `derivation/divide.py:mechanism_sentence` derives all of it;
    # `probes/fp32_correctness.py` measures it on a device. This DDA takes EVERY
    # step size as a reciprocal, so that lands straight in the geometry.
    #
    # We keep the division rather than working around it. The division-free
    # alternative, |dx| * (1/|dx|), was measured: it is 100% different from
    # the exact reciprocal, because it still contains one division to build
    # the reciprocal and a second rounding to undo it. A Newton-Raphson or
    # rsqrt refinement removes the division but changes the value outright.
    # So there is no division-free exact reciprocal, and a port cannot fix
    # this locally -- it needs TinyGrad's divide to be correctly rounded.
    tdx = W(nz_x, (f32(1.0) / dx).abs(), BIG)
    tdy = W(nz_y, (f32(1.0) / dy).abs(), BIG)
    tdz = W(nz_z, (f32(1.0) / dz).abs(), BIG)
    tmx = W(nz_x, W(pos_x, vx + 1 - ex, ex - vx) * tdx, BIG)
    tmy = W(nz_y, W(pos_y, vy + 1 - ey, ey - vy) * tdy, BIG)
    tmz = W(nz_z, W(pos_z, vz + 1 - ez, ez - vz) * tdz, BIG)
    t = zeros
    axis = W(falses, 0.0, -1.0)
    alive = trues
    hit_id, hit_t, hit_axis = zeros, zeros, W(falses, 0.0, -1.0)
    steps_used = zeros

    nx_, ny_, nz_ = RG["nx"], RG["ny"], RG["nz"]
    envbase = (Tensor.arange(B, dtype=I) * RG_STRIDE)[:, None]

    for _ in range(steps):
        in_range = t < OC_FAR
        ix = (vx - f32(RG["x0"])).cast(I)
        iy = (vy - f32(RG["y0"])).cast(I)
        iz = (vz - f32(RG["z0"])).cast(I)
        inb = (ix >= 0) & (iy >= 0) & (iz >= 0) & (ix < nx_) & (iy < ny_) & (iz < nz_)
        lin = (ix * ny_ + iy) * nz_ + iz
        gidx = W(inb, envbase + lin, Tensor.zeros(B, OC_NPIX, dtype=I))
        cell = cells_flat[gidx.reshape(-1)].reshape(B, OC_NPIX)
        bid = cell >> 4                                    # mc_state_id

        take = in_range & alive & (bid != 0)
        hit_id = W(take, bid, hit_id)
        hit_t = W(take, t, hit_t)
        hit_axis = W(take, axis, hit_axis)
        # count only steps where the ray was actually marching, so a ray that
        # hit at step 3 does not keep counting through the rest of the unroll
        go = in_range & alive
        steps_used = W(take | go, steps_used + 1.0, steps_used)
        alive = alive & ~take

        # Body of oc_raycast. `go` keeps a finished ray frozen instead of
        # letting tmx run past 1e30 into inf.
        #
        # The C is  if (tmx<tmy && tmx<tmz) x  elif (tmy<tmz) y  else  z,
        # so "not x" is ~(a & b) and NOT ~a. Getting that wrong freezes the ray
        # whenever tmx<tmy but tmx>=tmz, which is a common configuration.
        a = tmx < tmy
        b = tmx < tmz
        c = tmy < tmz
        not_x = ~(a & b)
        mv_x = go & a & b
        mv_y = go & not_x & c
        mv_z = go & not_x & ~c
        t = W(mv_x, tmx, W(mv_y, tmy, W(mv_z, tmz, t)))
        vx = W(mv_x, vx + sx, vx)
        vy = W(mv_y, vy + sy, vy)
        vz = W(mv_z, vz + sz, vz)
        # `tmx + tdx` is a mul feeding an add, so the Metal compiler may fold it
        # into fma(tdx, step, tmx). Barrier the increment so the axis step stays
        # two roundings, as in C. This is 3 extra kernels per unrolled step.
        tmx = W(mv_x, tmx + _barrier(tdx), tmx)
        tmy = W(mv_y, tmy + _barrier(tdy), tmy)
        tmz = W(mv_z, tmz + _barrier(tdz), tmz)
        axis = W(mv_x, 0.0, W(mv_y, 1.0, W(mv_z, 2.0, axis)))

    # oc_raycast tail: a ray that ran off the end reports OC_FAR and axis -1
    hit_t = W(alive, f32(OC_FAR), hit_t)
    hit_axis = W(alive, -1.0, hit_axis)

    # ---- oc_pixel tail: depth, then edge ----
    # The C stores (u8)(...), so t*4 is TRUNCATED, not rounded. Skipping this
    # makes ~12% of pixels differ from the reference by a tiny float epsilon
    # (160.00002 vs 160) even though the DDA is exact.
    d = (hit_t * f32(4.0)).cast(I)
    depth = W(hit_id != 0, W(d > 255, 255, d), 255)
    solid = hit_id != 0
    edge = zeros
    for a, hh in enumerate((ex + _barrier(hit_t * dx),
                             ey + _barrier(hit_t * dy),
                             ez + _barrier(hit_t * dz))):
        fr = hh - hh.floor()
        near = (fr < OC_EDGE_W) | (fr > f32(1.0) - OC_EDGE_W)
        tangent = hit_axis != f32(a)          # oc_edge skips the entry axis
        edge = W(solid & tangent & near, 1.0, edge)
    edge = W(hit_axis < 0, 0.0, edge)

    if want_steps:
        return hit_id, depth, edge, steps_used
    return hit_id, depth, edge


def load_regions(paths):
    """Concatenate per-env region tensors into one flat int32 buffer, packed
    states preserved. int32 not float32: the DDA index arithmetic must stay
    exact past 2**24 elements, and >>4 on the cell needs an int domain."""
    parts = []
    for p in paths:
        a = np.fromfile(p, dtype=np.uint16).astype(np.int32)
        assert a.size == RG_STRIDE, f"{p}: {a.size} != {RG_STRIDE}"
        parts.append(a)
    return Tensor(np.concatenate(parts))
