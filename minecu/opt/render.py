"""The voxel DDA render, written the way competent code writes it.

WHAT CHANGED RELATIVE TO `minecu/tinygrad/render.py`, AND WHY EACH ONE IS
ALGORITHMIC
----------------------------------------------------------------------
The earlier port was told "write it plainly, do not write anything
hardware-adjacent" and was executed as *write it naively*. Per DDA step it did
three float->int casts, three clips, and rebuilt a 4-index gather out of three
float coordinates -- on every one of 32 steps. That is not plain, it is a worse
algorithm than the one the reference implements, and it is what produced the
47.2x instruction ratio and the 26.2x time ratio `notes/19` reports.

Four rewrites, in the order they are applied. `ablate.py` prices each one
separately, so none of them has to be taken on trust.

1. **THE CURSOR IS AN INTEGER.** `minecu/tinygrad/render.py:169-171` computes
   `voxel_x = eye_x.floor()` -- a *float* -- and then, on every one of the 32
   steps, casts it back to int in order to index with. The cursor is the index
   of a voxel, so it is an integer; the reference says so at `kernels.cu:638`
   (`int voxel_x = (int)floorf(eye_x)`) and increments it as an int. Hoisting
   the cast out of the loop is the textbook fix for a `float` that is only
   ever integral.

2. **THE GATHER IS ONE INDEX, NOT FOUR.** `voxels[b, y, x, z]` is four
   indices where the flat address is one. The reference keeps a flat voxel
   address in the pointer it hands to `__ldg` (`kernels.cu:669`), and a gather
   on a flattened view with a single index is the same byte for one index
   instead of four.

3. **THE FLAT ADDRESS IS MAINTAINED, NOT REBUILT, AND THE BOUNDS TEST IS
   COMPUTED ONCE.** These are one change, because they are the same change.

   Amanatides-Woo's entire point is that the traversal state is *incremental*.
   `minecu/tinygrad/render.py:241-247` throws that away: it recomputes
   `b*wxyz + y*wxz + x*wz + z` from three coordinates every step, re-tests six
   bounds predicates on values that each change by a known constant, and clips
   three indices to keep the gather in range.

   Instead, per ray and before the loop:
     * one int32 flat address, advanced by one add of a **precomputed stride**
       -- `+wz`, `+wxz`, `+1`, each already carrying its sign;
     * per axis, `n_a` = **how many steps axis `a` has left before its cursor
       leaves the world** (`size-1-cursor` heading up, `cursor` heading down),
       decremented on whichever axis was taken.

   The per-step bounds test is then `n_x >= 0 && n_y >= 0 && n_z >= 0` -- three
   integer compares on values that are already live -- and the per-axis
   coordinates are *gone from the loop entirely*, which is what makes
   maintaining the address a win rather than a fourth piece of state. A cursor
   that is known to be out of range does not need a clamped index, it needs a
   safe one, and a single `where` supplies it.

   The reference instead re-derives the bounds from the cursor every step and
   folds six signed compares into three `ISETP.GE.U32.OR` by casting to
   unsigned (`kernels.cu:666-668`). `notes/19` gap 6 measured that fold as
   free on sm_75 -- both forms are three instructions -- so this rewrite is not
   "beating the reference's trick". It removes a test the reference also has
   to perform: the reference is a scalar kernel with a real branch and it can
   afford a per-step test; a masked array update has no branch to hang it on,
   so its per-step test has to be cheap. The asymmetry is in the programming
   model, not in the hardware.

4. **THE LIVE STATE IS THE SMALLEST THE ALGORITHM ALLOWS.** Per step the
   reference carries three `t_max` floats, three constant `t_delta` floats and
   three integer coordinates; the masked formulation additionally needs a
   hit-so-far flag. Two reductions, both hoisting:

   * `found` is redundant. `hit < 0` already means "this ray has not hit yet",
     because `hit` starts at `AIR == -1` and is only ever assigned a
     non-negative block id. Dropping the separate flag removes one tensor and
     one `|` from every step.
   * `t_delta` and the step signs are loop-*invariant*, and rewrite 3 folds the
     signs into the address strides, so they leave the loop entirely.

WHAT IS DELIBERATELY *NOT* DONE
-------------------------------
No `custom_kernel`, no tile size, no `TC_SELECT`, no `ALLOW_TF32`, no
`OPT_LEVEL`, no `Context(...)`, no hand-unrolling of the DDA loop, no
hand-staging, no hand-fusion, no device- or architecture-specific branch, no
per-thread scratch. The step loop is a Python `for` over `max_steps` with one
body, exactly as before; what TinyGrad does with that is still the
measurement.

The masked-loop formulation is also unchanged, and unchanged for the same
reason as before: the reference's inner loop is data-dependent per ray
(`goto done` on a hit) and a tensor program has no `goto`. So every ray still
runs the full `max_steps` iterations and "done" is a predicate. That costs
`max_steps` steps per ray against the reference's measured mean, and it is a
property of the formulation rather than of this rewrite. Removing it would
require `Tensor.While` over a convergence mask, which is a different
experiment and is not smuggled in here.

THE COLOUR LOOKUP
------------------
Unchanged from `minecu/tinygrad/render.py` and for the same reason: the reference
reaches the sky through a *branch* (`if (hit_block >= 0) ... else ...`), and a
tensor program has no `if` on a tensor, so AIR maps to row 13 of a 14-row table
and both cases become one gather. It is a change of expression, not of
schedule, and it is disclosed rather than hidden.

MEMORY SAFETY
-------------
A tensor gather has no bounds check, while the reference's bounds test gates a
pointer it never forms. A one-dimensional gather on a flat address makes that
sharper than in the 4-index form, not weaker: `inside.where(addr, 0)` supplies
an in-range address, and `inside` -- which is computed anyway -- decides
whether the byte read counts. Every index handed to the gather is in
`[0, B*WX*WY*WZ)`.
"""

from tinygrad import Tensor, dtypes

from minecu.tinygrad.blocks import AIR, colour_lut

# The reference's `1e30f` guard for a zero ray component (kernels.cu:646-648).
_BIG = 1e30


def camera_basis(cameras: Tensor, fov_degrees: float, width: int, height: int) -> Tensor:
    """`precompute_camera_basis_kernel` (kernels.cu:511): [B, 5] -> [B, 14].

    IMPORTED from `minecu.tinygrad.render` rather than copied, so the two ports
    differ in the DDA and in nothing else. A diff showing a different camera
    basis would be measuring the wrong thing.
    """
    from minecu.tinygrad.render import camera_basis as _cb
    return _cb(cameras, fov_degrees, width, height)


def _setup(voxels: Tensor, basis: Tensor, width: int, height: int):
    """Everything before the step loop, computed once.

    Byte-for-byte the same graph in all five rungs of the ladder, so any
    difference between V0 and V4 is inside the loop. The ladder therefore
    prices the *step body*, which is the only thing this note is about.
    """
    B = voxels.shape[0]
    wy, wx, wz = voxels.shape[1], voxels.shape[2], voxels.shape[3]
    f32, i32 = dtypes.float32, dtypes.int32

    # The eye and the basis, lifted to [B, 1, 1, 1] so they broadcast over the
    # pixel grid the way one scalar does per thread in the reference. The slice
    # is `i:i+1` and not `i`: `basis[:, i]` drops to rank 1, and `[None, None]`
    # on a rank-1 tensor is (1, 1, B) -- the batch in the wrong axis, which is
    # invisible at B=1 and wrong at every other B.
    def comp(i):
        return basis[:, i:i + 1, None, None]

    eye_x, eye_y, eye_z = comp(0), comp(1), comp(2)
    fwd_x, fwd_y, fwd_z = comp(3), comp(4), comp(5)
    right_x, right_z = comp(6), comp(7)
    up_x, up_y, up_z = comp(8), comp(9), comp(10)
    scale_u, scale_v = comp(11), comp(12)

    # The pixel grid. px along the last axis, py along the third, so the output
    # is already [B, 1, H, W] and needs no transpose later.
    px = Tensor(list(range(width)), dtype=f32).reshape(1, 1, 1, width)
    py = Tensor(list(range(height)), dtype=f32).reshape(1, 1, height, 1)

    # kernels.cu:619-620
    u = ((px + 0.5) * (2.0 / float(width)) - 1.0) * scale_u
    v = (1.0 - (py + 0.5) * (2.0 / float(height))) * scale_v

    # kernels.cu:622-624. right_y is 0, so ray_y has no u term -- the same
    # fact the reference hard-codes in its basis layout.
    ray_x = fwd_x + u * right_x + v * up_x
    ray_y = fwd_y + v * up_y
    ray_z = fwd_z + u * right_z + v * up_z

    # kernels.cu:626-629, rsqrtf.
    ray_len = (ray_x * ray_x + ray_y * ray_y + ray_z * ray_z).rsqrt()
    ray_x, ray_y, ray_z = ray_x * ray_len, ray_y * ray_len, ray_z * ray_len

    # kernels.cu:642-644, the step sign per axis.
    pos_x, pos_y, pos_z = ray_x >= 0, ray_y >= 0, ray_z >= 0
    # As a float, for the V0 body, which steps the float cursor by `+step_x`
    # exactly as `minecu/tinygrad/render.py:259` does. The int32 form is built in
    # `_ladder_init`, where rewrite 1 hoists the cast.
    signf = (pos_x.where(1.0, -1.0), pos_y.where(1.0, -1.0), pos_z.where(1.0, -1.0))

    # kernels.cu:646-652, the reciprocal guard and t_delta.
    inv_x = (ray_x != 0).where(ray_x.reciprocal(), _BIG)
    inv_y = (ray_y != 0).where(ray_y.reciprocal(), _BIG)
    inv_z = (ray_z != 0).where(ray_z.reciprocal(), _BIG)
    td_x, td_y, td_z = inv_x.abs(), inv_y.abs(), inv_z.abs()

    # kernels.cu:638-640, the starting cursor. FLOAT here on purpose: `floorf`
    # is a float operation in the reference too, and the ladder casts it once
    # (rewrite 1) rather than 32 times.
    vx, vy, vz = eye_x.floor(), eye_y.floor(), eye_z.floor()

    # kernels.cu:654-659, t_max. Uses the float cursor, so the arithmetic is
    # the reference's arithmetic.
    t_max_x = pos_x.where((vx + 1.0 - eye_x) * inv_x, (vx - eye_x) * inv_x)
    t_max_y = pos_y.where((vy + 1.0 - eye_y) * inv_y, (vy - eye_y) * inv_y)
    t_max_z = pos_z.where((vz + 1.0 - eye_z) * inv_z, (vz - eye_z) * inv_z)

    return dict(B=B, wy=wy, wx=wx, wz=wz, f32=f32, i32=i32, voxels=voxels,
                pos=(pos_x, pos_y, pos_z), signf=signf,
                t_delta=(td_x, td_y, td_z),
                t_max=(t_max_x, t_max_y, t_max_z), v=(vx, vy, vz),
                width=width, height=height)


# ---------------------------------------------------------------------------
# the five step bodies
# ---------------------------------------------------------------------------
def _step_naive(S, hit, found, steps, k, count_steps):
    """V0 -- `minecu/tinygrad/render.py`'s body, structurally verbatim.

    Three float->int casts, three clips, a 4-index gather rebuilt from three
    float coordinates, six bounds predicates on floats, a separate `found`
    flag. This is the port the 47.2x instruction ratio was measured on.
    """
    i32, B, wy, wx, wz = S["i32"], S["B"], S["wy"], S["wx"], S["wz"]
    vx, vy, vz = S["v"]
    t_max_x, t_max_y, t_max_z = S["t_max"]
    td_x, td_y, td_z = S["t_delta"]

    if count_steps:
        steps = (~found).where(k + 1, steps)

    b_index = Tensor.arange(B, dtype=i32).reshape(B, 1, 1, 1)
    inside = ((vx >= 0) & (vx < float(wx)) &
              (vy >= 0) & (vy < float(wy)) &
              (vz >= 0) & (vz < float(wz)))
    ix = vx.cast(i32).clip(0, wx - 1)
    iy = vy.cast(i32).clip(0, wy - 1)
    iz = vz.cast(i32).clip(0, wz - 1)
    block = S["voxels"][b_index, iy, ix, iz].cast(i32)

    solid = (block >= 0) & inside & ~found
    hit = solid.where(block, hit)
    found = found | solid

    take_x = (t_max_x < t_max_y) & (t_max_x < t_max_z)
    take_y = (t_max_y < t_max_z) & ~take_x
    t_max_x = take_x.where(t_max_x + td_x, t_max_x)
    t_max_y = take_y.where(t_max_y + td_y, t_max_y)
    t_max_z = (~take_x & ~take_y).where(t_max_z + td_z, t_max_z)
    S["t_max"] = (t_max_x, t_max_y, t_max_z)
    sx, sy, sz = S["signf"]
    S["v"] = (take_x.where(vx + sx, vx), take_y.where(vy + sy, vy),
              (~take_x & ~take_y).where(vz + sz, vz))
    return hit, found, steps


def _step_int_cursor(S, hit, found, steps, k, count_steps):
    """V1 -- V0 plus rewrite 1: the cursor is an integer, cast once.

    Same 4-index gather, same clips, same float bounds test. The only change
    is that `voxel_x` is int32 rather than float32, so the three per-step casts
    and the three float comparisons go away.
    """
    i32, B, wy, wx, wz = S["i32"], S["B"], S["wy"], S["wx"], S["wz"]
    t_max_x, t_max_y, t_max_z = S["t_max"]
    td_x, td_y, td_z = S["t_delta"]
    vx, vy, vz = S["v"]                      # already int32, cast once outside

    if count_steps:
        steps = (~found).where(k + 1, steps)

    b_index = Tensor.arange(B, dtype=i32).reshape(B, 1, 1, 1)
    inside = ((vx >= 0) & (vx < wx) & (vy >= 0) & (vy < wy) & (vz >= 0) & (vz < wz))
    ix, iy, iz = vx.clip(0, wx - 1), vy.clip(0, wy - 1), vz.clip(0, wz - 1)
    block = S["voxels"][b_index, iy, ix, iz].cast(i32)

    solid = (block >= 0) & inside & ~found
    hit = solid.where(block, hit)
    found = found | solid

    take_x = (t_max_x < t_max_y) & (t_max_x < t_max_z)
    take_y = (t_max_y < t_max_z) & ~take_x
    take_z = ~(take_x | take_y)
    t_max_x = take_x.where(t_max_x + td_x, t_max_x)
    t_max_y = take_y.where(t_max_y + td_y, t_max_y)
    t_max_z = take_z.where(t_max_z + td_z, t_max_z)
    S["t_max"] = (t_max_x, t_max_y, t_max_z)
    sx, sy, sz = S["sign"]
    S["v"] = (take_x.where(vx + sx, vx), take_y.where(vy + sy, vy),
              take_z.where(vz + sz, vz))
    return hit, found, steps


def _step_flat_gather(S, hit, found, steps, k, count_steps):
    """V2 -- V1 plus rewrite 2: one flat index instead of a 4-index gather.

    The address is still rebuilt from the three integer coordinates every
    step, so this rung prices *only* the gather's shape. The per-axis clips
    are still there, because a per-axis index still has to be clamped; what
    disappears is the fourth index and the per-consumer predicated loads the
    4-index form forces on the generator.
    """
    i32, wy, wx, wz = S["i32"], S["wy"], S["wx"], S["wz"]
    t_max_x, t_max_y, t_max_z = S["t_max"]
    td_x, td_y, td_z = S["t_delta"]
    vx, vy, vz = S["v"]

    if count_steps:
        steps = (~found).where(k + 1, steps)

    inside = ((vx >= 0) & (vx < wx) & (vy >= 0) & (vy < wy) & (vz >= 0) & (vz < wz))
    wxz = wx * wz
    b_off = S["b_off"]
    addr = b_off + (vy * wxz + vx * wz + vz)
    safe = inside.where(addr, 0)
    block = S["flat"][safe].cast(i32)

    solid = (block >= 0) & inside & ~found
    hit = solid.where(block, hit)
    found = found | solid

    take_x = (t_max_x < t_max_y) & (t_max_x < t_max_z)
    take_y = (t_max_y < t_max_z) & ~take_x
    take_z = ~(take_x | take_y)
    t_max_x = take_x.where(t_max_x + td_x, t_max_x)
    t_max_y = take_y.where(t_max_y + td_y, t_max_y)
    t_max_z = take_z.where(t_max_z + td_z, t_max_z)
    S["t_max"] = (t_max_x, t_max_y, t_max_z)
    sx, sy, sz = S["sign"]
    S["v"] = (take_x.where(vx + sx, vx), take_y.where(vy + sy, vy),
              take_z.where(vz + sz, vz))
    return hit, found, steps


def _step_incremental(S, hit, found, steps, k, count_steps):
    """V3 -- V2 plus rewrite 3: the address is maintained; bounds computed once.

    The per-axis coordinates are no longer live. What is live is one int32
    flat address and three int32 steps-to-exit counters, and a step is one
    address add, one counter decrement, and three integer compares.
    """
    i32 = S["i32"]
    t_max_x, t_max_y, t_max_z = S["t_max"]
    td_x, td_y, td_z = S["t_delta"]
    addr = S["addr"]
    n_x, n_y, n_z = S["n"]

    if count_steps:
        steps = (~found).where(k + 1, steps)

    inside = (n_x >= 0) & (n_y >= 0) & (n_z >= 0)
    block = S["flat"][inside.where(addr, 0)].cast(i32)

    solid = (block >= 0) & inside & ~found
    hit = solid.where(block, hit)
    found = found | solid

    take_x = (t_max_x < t_max_y) & (t_max_x < t_max_z)
    take_y = (t_max_y < t_max_z) & ~take_x
    take_z = ~(take_x | take_y)
    t_max_x = take_x.where(t_max_x + td_x, t_max_x)
    t_max_y = take_y.where(t_max_y + td_y, t_max_y)
    t_max_z = take_z.where(t_max_z + td_z, t_max_z)
    S["t_max"] = (t_max_x, t_max_y, t_max_z)
    st_x, st_y, st_z = S["stride"]
    S["addr"] = addr + take_x.where(st_x, take_y.where(st_y, st_z))
    S["n"] = (take_x.where(n_x - 1, n_x), take_y.where(n_y - 1, n_y),
              take_z.where(n_z - 1, n_z))
    return hit, found, steps


def _step_lean(S, hit, found, steps, k, count_steps):
    """V4 -- V3 plus rewrite 4: the smallest live state the algorithm allows.

    `found` is gone: `hit < 0` already means "this ray has not hit yet",
    because `hit` starts at `AIR == -1` and is only ever assigned a
    non-negative block id. One fewer tensor and one fewer `|` per step.

    This is the submitted version -- `render()` is `render_variant(...,
    "lean")`.
    """
    i32 = S["i32"]
    t_max_x, t_max_y, t_max_z = S["t_max"]
    td_x, td_y, td_z = S["t_delta"]
    addr = S["addr"]
    n_x, n_y, n_z = S["n"]
    live = hit < 0                              # == ~found, derived from hit

    if count_steps:
        steps = live.where(k + 1, steps)

    inside = (n_x >= 0) & (n_y >= 0) & (n_z >= 0)
    block = S["flat"][inside.where(addr, 0)].cast(i32)

    solid = (block >= 0) & inside & live
    hit = solid.where(block, hit)

    take_x = (t_max_x < t_max_y) & (t_max_x < t_max_z)
    take_y = (t_max_y < t_max_z) & ~take_x
    take_z = ~(take_x | take_y)
    t_max_x = take_x.where(t_max_x + td_x, t_max_x)
    t_max_y = take_y.where(t_max_y + td_y, t_max_y)
    t_max_z = take_z.where(t_max_z + td_z, t_max_z)
    S["t_max"] = (t_max_x, t_max_y, t_max_z)
    st_x, st_y, st_z = S["stride"]
    S["addr"] = addr + take_x.where(st_x, take_y.where(st_y, st_z))
    S["n"] = (take_x.where(n_x - 1, n_x), take_y.where(n_y - 1, n_y),
              take_z.where(n_z - 1, n_z))
    return hit, None, steps


#: The ablation ladder. Each rung is the rung above plus exactly one rewrite,
#: so `ablate.py` can price them separately.
#:
#:   naive        minecu/tinygrad/render.py's body, structurally verbatim
#:   int_cursor   + rewrite 1: the cursor is an integer, cast once
#:   flat_gather  + rewrite 2: one flat index into the gather, not four
#:   incremental  + rewrite 3: the address is maintained with precomputed
#:                strides, and the bounds test is computed once per ray
#:   lean         + rewrite 4: no separate `found` flag
VARIANTS = {
    "naive": _step_naive,
    "int_cursor": _step_int_cursor,
    "flat_gather": _step_flat_gather,
    "incremental": _step_incremental,
    "lean": _step_lean,
}

#: The submitted version.
BEST = "lean"


def _ladder_init(S, variant):
    """The prologue state each rung needs, set up once, outside the loop."""
    B, wy, wx, wz, i32 = S["B"], S["wy"], S["wx"], S["wz"], S["i32"]
    if variant != "naive":
        # Rewrite 1: the cursor becomes an integer, once.
        S["v"] = tuple(v.cast(i32) for v in S["v"])
        S["sign"] = tuple(p.where(1, -1).cast(i32) for p in S["pos"])
    if variant in ("flat_gather", "incremental", "lean"):
        wxz, wxyz = wx * wz, wx * wy * wz
        S["flat"] = S["voxels"].reshape(B * wxyz)
        S["b_off"] = (Tensor.arange(B, dtype=i32) * wxyz).reshape(B, 1, 1, 1)
    if variant in ("incremental", "lean"):
        vx, vy, vz = S["v"]
        wxz, wxyz = wx * wz, wx * wy * wz
        # Rewrite 3: one flat address, and the three signed strides that move
        # it. The sign is already folded in, so a step is a single add.
        S["addr"] = S["b_off"] + (vy * wxz + vx * wz + vz)
        sx, sy, sz = S["sign"]
        S["stride"] = (sx * wz, sy * wxz, sz)
        # Rewrite 3: the per-axis step budget. A cursor that starts outside the
        # world is given a negative budget, so it reads as out of range on the
        # first step rather than as a large positive one. The camera never does
        # that here; writing it so costs three comparisons outside the loop.
        S["n"] = tuple(
            ((c >= 0) & (c < s)).where(p.where(s - 1 - c.clip(0, s - 1),
                                              c.clip(0, s - 1)), -1)
            for p, c, s in zip(S["pos"], S["v"], (wx, wy, wz)))


def render_variant(voxels: Tensor, basis: Tensor, width: int, height: int,
                   max_steps: int, variant: str = "lean",
                   count_steps: bool = False):
    """The uint8 observation [B, H, W, 3], at one rung of the ladder.

    All five rungs share the same prologue graph, so any difference between
    them is inside the step loop. They differ in the *state* they carry, not in
    the ray they trace; `verify_opt.py` checks that all five produce
    bit-identical observations and step counts.
    """
    if variant not in VARIANTS:
        raise SystemExit(f"unknown variant {variant!r}; have {sorted(VARIANTS)}")
    S = _setup(voxels, basis, width, height)
    _ladder_init(S, variant)
    body = VARIANTS[variant]
    B, height = S["B"], S["height"]
    i32 = S["i32"]

    hit = Tensor.full((B, 1, height, S["width"]), AIR, dtype=i32)
    found = Tensor.full((B, 1, 1, 1), False, dtype=dtypes.bool)
    steps = Tensor.zeros((B, 1, height, S["width"]), dtype=i32)

    for k in range(max_steps):
        hit, found, steps = body(S, hit, found, steps, k, count_steps)

    # The store, kernels.cu:690-699, as a table lookup. AIR is row 13; the
    # reference reaches the sky by a branch, which a tensor program has not
    # got, so both cases are the same gather.
    lut = colour_lut()
    colour_row = (hit >= 0).where(hit, 13)
    obs = lut[colour_row].reshape(B, height, S["width"], 3)
    return (obs, steps) if count_steps else obs.contiguous()


def render(voxels: Tensor, basis: Tensor, width: int, height: int,
           max_steps: int, count_steps: bool = False):
    """The submitted version: `render_variant(..., "lean")`.

    Same signature and same semantics as `minecu/tinygrad/render.py:render`, so it
    drops into every harness in this repo, and it produces the same
    observation -- `verify_opt.py` asserts it, bit for bit.
    """
    return render_variant(voxels, basis, width, height, max_steps, BEST,
                          count_steps)
