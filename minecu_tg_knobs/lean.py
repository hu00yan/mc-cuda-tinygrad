"""lean.py -- arm B's port: the same DDA, written the way Amanatides-Woo says.

WHY THIS FILE EXISTS AT ALL
--------------------------
The brief for this arm says: *do not sweep knobs over the naive port*. And it is
right, and the reason is ledger **C-14**. `minecu_tg/render.py:241-247` recomputes
the flat voxel address from three float coordinates on all 32 steps -- three
`cast(i32)`, three `clip`, and a four-index gather -- where the algorithm the
reference implements precomputes `tMax`/`tDelta` and then *advances*. Sweeping
knobs on top of that measures a port defect and a knob at the same time, and
attributing the result to the compiler would be the exact error C-14 corrects.

So this file is a port, not a kernel. It is ordinary TinyGrad, line for line the
same kind of array code `minecu_tg/render.py` is, and it obeys every constraint
the study imposes:

  * no `Tensor.custom_kernel`, no inline PTX, no `Context(...)` steering
  * no codegen knob read here -- arm B's knobs are applied *around* the call,
    by `minecu_tg_knobs/registry.py`
  * no hand-unrolling, no hand-fusion, no hand-staging, no tile size, no
    `max_steps` special case

The only thing that differs from the naive port is that the port is *right*: it
is the same masked-array formulation, carrying the index arithmetic the
algorithm actually specifies. Three changes, each a port fix and not a schedule
choice, and each separately switchable so its effect is measured rather than
asserted:

  1. **The cursor is integral.** The reference does `(int)floorf(eye_x)` once,
     in its prologue, and the cursor is an `int` for the rest of the kernel
     (`kernels.cu:638-640`, `:669`, `:673-677`). The naive port keeps `voxel_x`
     as a *float* for all 32 steps and pays 3 `cast(i32)` + 3 `clip` per step
     just to index the gather with it -- work that exists only because of the
     encoding the port chose. `cursor_int`.

  2. **The address is advanced, not recomputed.** The reference's gather address
     is `b*wxyz + vy*wxz + vx*wz + vz` (`kernels.cu:669`), so an x step adds
     `wz`, a y step adds `wxz`, and a z step adds `1` (`kernels.cu:673-677`) --
     all constants of the world shape. The whole flat address is therefore one
     accumulator that takes one of three constant increments per step, selected
     by the same three-way `t_max` comparison that chooses the axis. This is
     the C-14 term, removed. `flat_advance`.

  3. **The bounds test is the reference's.** The reference writes
     `(unsigned)voxel_x < (unsigned)wx` and `&&`s three of them
     (`kernels.cu:666-668`) -- one unsigned compare per axis, which ptxas emits
     as a single `ISETP`. The naive port writes the six-compare signed form and
     declines the fold on purpose (`render.py:224-228`). Here it is one cast to
     uint32 and one compare per axis. `unsigned_bounds`.

A structural fact about the state that is worth stating because it bounds what
can be recovered: **all rays in an environment share the eye but not the
traversal.** `vx`/`vy`/`vz`, `addr`, `t_max` and `t_delta` all start at
`[B, 1, 1, 1]` -- one element per environment -- and become `[B, 1, H, W]` on
the *first* step, because the axis test is a function of the per-ray `t_max`.
So the cursor state is per ray, exactly as the reference's per-thread cursor is,
and `hit`/`found`/`steps` are per pixel. This is why there is no way to hoist
the whole traversal out of a per-pixel loop, and it is checked by
`verify.py` rather than asserted here.

A trap found while writing this, recorded because the wrong version of it is
easy to write and looks like an optimisation:

    dv = take_x.where(sx, take_y.where(sy, sz));  vx += dv; vy += dv; vz += dv

saves two selects and **moves y and z whenever x moved**, so the traversal
takes a different path -- it never terminates early and the step count pins at
`max_steps`. The reference advances exactly one axis per step
(`kernels.cu:672-678`), so each variable needs its own gate. `verify.py`'s step
count is what caught it: the image was still plausible.

WHAT IS DELIBERATELY *NOT* FIXED
--------------------------------
These are properties of the array formulation itself, not of the writing, and
"fixing" any of them would be hand-staging rather than expressing the same
computation:

  * **No per-ray early exit.** A tensor program has no `goto`; the honest array
    form is a `found` mask, and the honest cost is that every ray runs all
    `max_steps` bodies. The reference's `goto done` becomes 33 `BSSY`/`BSYNC`
    pairs it does not pay here -- and does correspondingly less total work.
    Notes/19 §4.2c prices both sides of that trade; neither side is a defect.

  * **No carry-until-converged loop.** "Stop marching this ray" is expressed as
    data -- accumulate `found` and stop *accepting* results -- which is the
    vectorised-raymarcher idiom. A carry form would need low-level loop
    constructs.

  * **The sky is a table row, not a branch.** Same as the naive port and the
    same reasoning (`blocks.py:42`); a change of *shape*, not of schedule.

WHAT ARM B MAY NOT CLAIM
------------------------
This port is not "tuned for T4". It is a port that does not carry a known
defect. Its absolute ratio against the reference is therefore **not comparable
to arm A's**, and no number in `notes/27-minecu-knobs-arm.md` divides a
lean-port figure by an arm-A figure. What the knob sweep needs is the
comparison *within* this file, and that one is exact.
"""

from __future__ import annotations

from tinygrad import Tensor, dtypes

from minecu_tg.blocks import colour_lut, AIR

_BIG = 1e30      # kernels.cu:646-648, the zero-ray-component guard


def camera_basis(cameras: Tensor, fov_degrees: float, width: int, height: int) -> Tensor:
    """`minecu_tg/render.py:65`, reproduced so this module has no dependency on
    the port under comparison.

    The camera basis is a separate kernel that runs once per batch and is *not*
    part of the timed render graph: it is an input, exactly as it is for the
    reference, whose `render_kernel_uint8_prebasis` takes the 14-slot basis that
    `precompute_camera_basis_kernel` wrote. It is not one of the things the knob
    sweep is about, and `census.py` records which of the two ports emits it.
    """
    f32 = dtypes.float32
    cam_x, cam_y, cam_z = cameras[:, 0], cameras[:, 1], cameras[:, 2]
    yaw, pitch = cameras[:, 3], cameras[:, 4]
    zero = yaw * 0

    sin_yaw, cos_yaw = yaw.sin(), yaw.cos()
    sin_pitch, cos_pitch = pitch.sin(), pitch.cos()

    half_fov = Tensor([fov_degrees * 0.01745329 * 0.5], dtype=f32).tan()[0] + zero
    aspect = float(width) / float(height)

    eye = cam_x.stack(cam_y + 1.6, cam_z, dim=1).cast(f32)
    forward = (sin_yaw * cos_pitch).stack(sin_pitch, cos_yaw * cos_pitch, dim=1)
    right = cos_yaw.stack(-sin_yaw, dim=1)                    # right_y == 0
    up = (-sin_yaw * sin_pitch).stack(cos_pitch, -cos_yaw * sin_pitch, dim=1)
    scales = (half_fov * aspect).stack(half_fov, dim=1)
    return eye.cat(forward, right, up, scales, dim=1).contiguous()


def render(voxels: Tensor, basis: Tensor, width: int, height: int,
           max_steps: int, count_steps: bool = False,
           cursor_int: bool = True, flat_advance: bool = True,
           unsigned_bounds: bool = True):
    """The uint8 observation, [B, H, W, 3].

    voxels: [B, WY, WX, WZ] int8.  basis: [B, 14] float32.  Returns `obs`, or
    `(obs, steps)` when `count_steps` -- the same loop with a counter, which is
    instrumentation and is never on the timed path.

    The three keyword flags switch off the three port fixes above, so that each
    one's cost is measured rather than asserted. `flat_advance` needs an
    integral cursor (it advances an int address by a constant), so setting it
    with `cursor_int=False` is refused rather than silently doing something
    else.
    """
    if flat_advance and not cursor_int:
        raise ValueError("flat_advance advances an int32 address by a constant "
                         "stride, so it requires cursor_int=True")

    B = voxels.shape[0]
    wy, wx, wz = voxels.shape[1], voxels.shape[2], voxels.shape[3]
    f32, i32 = dtypes.float32, dtypes.int32
    wxyz = wy * wx * wz

    def comp(i):
        # The `i:i+1` slice, not `i`: `basis[:, i]` drops to rank 1 and
        # `[None, None]` on a rank 1 gives (1, 1, B), which is invisible at
        # B=1 and wrong at every other B. See notes/19 §1.3.
        return basis[:, i:i + 1, None, None]

    eye_x, eye_y, eye_z = comp(0), comp(1), comp(2)
    fwd_x, fwd_y, fwd_z = comp(3), comp(4), comp(5)
    right_x, right_z = comp(6), comp(7)
    up_x, up_y, up_z = comp(8), comp(9), comp(10)
    scale_u, scale_v = comp(11), comp(12)

    # ---- the pixel grid, and the ray. kernels.cu:618-629.
    px = Tensor(list(range(width)), dtype=f32).reshape(1, 1, 1, width)
    py = Tensor(list(range(height)), dtype=f32).reshape(1, 1, height, 1)

    u = ((px + 0.5) * (2.0 / float(width)) - 1.0) * scale_u
    v = (1.0 - (py + 0.5) * (2.0 / float(height))) * scale_v

    ray_x = fwd_x + u * right_x + v * up_x
    ray_y = fwd_y + v * up_y
    ray_z = fwd_z + u * right_z + v * up_z

    ray_len = (ray_x * ray_x + ray_y * ray_y + ray_z * ray_z).rsqrt()
    ray_x, ray_y, ray_z = ray_x * ray_len, ray_y * ray_len, ray_z * ray_len

    # ---- traversal setup, kernels.cu:638-659. All of it is prologue: the
    # reference computes `tDelta` and `tMax` once and never recomputes them.
    if cursor_int:
        vx = eye_x.floor().cast(i32)      # kernels.cu:638-640
        vy = eye_y.floor().cast(i32)
        vz = eye_z.floor().cast(i32)
    else:
        vx, vy, vz = eye_x.floor(), eye_y.floor(), eye_z.floor()

    sx = (ray_x >= 0).where(1, -1)         # kernels.cu:642-644
    sy = (ray_y >= 0).where(1, -1)
    sz = (ray_z >= 0).where(1, -1)

    inv_x = (ray_x != 0).where(ray_x.reciprocal(), _BIG)   # :646-648
    inv_y = (ray_y != 0).where(ray_y.reciprocal(), _BIG)
    inv_z = (ray_z != 0).where(ray_z.reciprocal(), _BIG)

    t_delta_x, t_delta_y, t_delta_z = inv_x.abs(), inv_y.abs(), inv_z.abs()  # :650-652

    t_max_x = (ray_x >= 0).where((vx + 1.0 - eye_x) * inv_x, (vx - eye_x) * inv_x)  # :654
    t_max_y = (ray_y >= 0).where((vy + 1.0 - eye_y) * inv_y, (vy - eye_y) * inv_y)  # :656
    t_max_z = (ray_z >= 0).where((vz + 1.0 - eye_z) * inv_z, (vz - eye_z) * inv_z)  # :658

    if flat_advance:
        # The flat address, ONCE, then advanced. `kernels.cu:669` indexes
        # `b*wxyz + vy*wxz + vx*wz + vz`, so each axis step adds a CONSTANT of
        # the world shape. `vox_flat` is the
        # [B, WY, WX, WZ] grid viewed flat, which makes the per-step gather a
        # single-index load -- one address, one load, rather than the naive
        # port's four-index gather that lowers to four predicated byte loads
        # of the same address.
        b_index = Tensor.arange(B, dtype=i32).reshape(B, 1, 1, 1)
        addr = b_index * wxyz + (vy * (wx * wz) + vx * wz + vz)
        # strides of `b*wxyz + vy*wxz + vx*wz + vz`: x -> wz, y -> wx*wz,
        # z -> 1, each times the sign.
        inc_x = (sx * wz).cast(i32)
        inc_y = (sy * (wx * wz)).cast(i32)
        inc_z = sz.cast(i32)
        # One 1-D buffer, not [B, wxyz]: the batch offset is already inside
        # `addr` (the reference's `b*wxyz`, kernels.cu:669), so a 1-D index
        # tensor indexes it elementwise and `block` comes out with exactly
        # `addr`'s shape -- [B,1,1,1] on the first step, [B,1,H,W] after.
        # Indexing a [B, wxyz] tensor would append a trailing axis instead,
        # which is a different program.
        vox_flat = voxels.reshape(B * wxyz)
    else:
        b_index = Tensor.arange(B, dtype=i32).reshape(B, 1, 1, 1)

    hit = Tensor.full((B, 1, height, width), AIR, dtype=i32)
    found = Tensor.full((B, 1, 1, 1), False, dtype=dtypes.bool)
    steps = Tensor.zeros((B, 1, height, width), dtype=i32)

    # ---- the DDA, kernels.cu:664-679
    for step in range(max_steps):
        if count_steps:
            # the body runs once per ray per step until it hits, so the count is
            # the 1-based index of the step that hit, or max_steps
            steps = (~found).where(step + 1, steps)

        if flat_advance:
            # `addr` starts [B,1,1,1] (the eye is shared by every ray in an env)
            # and becomes [B,1,H,W] on the first step, because `take_*` is a
            # function of the per-ray `t_max`. So this is a PER-RAY gather, as
            # the reference's per-thread `__ldg` is: one address per ray, one
            # load. Do NOT force it back to [B,1,1,1] -- that would be wrong,
            # because the rays diverge at the first axis step.
            block = vox_flat[addr.clip(0, B * wxyz - 1)]
        else:
            ix = vx.cast(i32).clip(0, wx - 1)
            iy = vy.cast(i32).clip(0, wy - 1)
            iz = vz.cast(i32).clip(0, wz - 1)
            block = voxels[b_index, iy, ix, iz]

        if unsigned_bounds:
            # kernels.cu:666-668, one unsigned compare per axis
            inside = ((vx.cast(dtypes.uint32) < wx) & (vy.cast(dtypes.uint32) < wy) &
                      (vz.cast(dtypes.uint32) < wz))
        else:
            inside = ((vx >= 0) & (vx < wx) &
                      (vy >= 0) & (vy < wy) &
                      (vz >= 0) & (vz < wz))

        # `inside` gates the value; `found` is separate because a ray that
        # leaves the world has not hit anything, and conflating the two corrupts
        # the step count for exactly the rays that dominate this workload.
        solid = (block >= 0) & inside & ~found
        hit = solid.where(block.cast(i32), hit)
        found = found | solid

        # the step, kernels.cu:672-678, as three selects. Exactly one of the
        # three axes moves, so the "else" branch of the nested `where` is the z
        # increment and the third axis costs no separate test.
        take_x = (t_max_x < t_max_y) & (t_max_x < t_max_z)
        take_y = ~take_x & (t_max_y < t_max_z)

        # EXACTLY ONE axis moves, and each variable is gated by its own gate.
        # A single shared `dv = where(take_x, sx, where(take_y, sy, sz))` added
        # to all three cursors looks like a saving and is not the same
        # algorithm: it moves y and z whenever x moved. The z gate is the
        # complement of the other two, which is why the reference's three-way
        # `if/else if/else` (kernels.cu:672-678) needs no third test.
        if flat_advance:
            # One selected CONSTANT per step, then one add. `take_x` picks the
            # x stride, `take_y` the y stride, and the `else` of the inner
            # `where` is the z stride -- so the address advances by exactly the
            # amount the one axis that moved contributes.
            addr = addr + take_x.where(inc_x, take_y.where(inc_y, inc_z))

        vx = vx + take_x.where(sx, 0)
        vy = vy + take_y.where(sy, 0)
        vz = vz + (~take_x & ~take_y).where(sz, 0)

        # Each t_max advances by ITS OWN delta, gated by ITS OWN axis. The
        # reference advances exactly one (kernels.cu:673/675/677).
        t_max_x = t_max_x + take_x.where(t_delta_x, 0)
        t_max_y = t_max_y + take_y.where(t_delta_y, 0)
        t_max_z = t_max_z + (~take_x & ~take_y).where(t_delta_z, 0)

    # ---- the store, kernels.cu:690-699, as one table lookup
    lut = colour_lut()                                       # [14, 3] uint8
    colour_row = (hit >= 0).where(hit, 13)                   # [B, 1, H, W] int32
    obs = lut[colour_row].reshape(B, height, width, 3)       # [B, H, W, 3] uint8
    return (obs, steps) if count_steps else obs.contiguous()
