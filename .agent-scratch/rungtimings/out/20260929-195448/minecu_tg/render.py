"""The render, written as plain tensor code.

This is the file the whole study rests on, so the constraint that produced it
is worth writing down at the top: **this is the "plain C++" side of the
comparison.** It is what a person writes when they are handed
`notes/17-minecu-spec.md` and told to implement
`render_kernel_uint8_prebasis` with array operations and a loop. It is not
what a person writes after reading the reference's SASS.

Therefore, and specifically, this file does NOT:

  * call `Tensor.custom_kernel` or any other escape hatch that emits CUDA
  * set a tile size, `TC_SELECT`, `ALLOW_TF32`, a block size, or any other
    knob the code generator reads
  * unroll the DDA loop by hand, fuse stages by hand, or stage data through
    global memory to shape the schedule
  * pass a `Context(...)` to steer the scheduler
  * special-case `max_steps` (e.g. peel a 16-wide body) to imitate the
    reference's `#pragma unroll 4`

The DDA loop is a Python `for` over `max_steps` with one body, exactly as one
would write a vectorised raymarcher. What TinyGrad does with that unrolled op
chain -- fuses it into one kernel, splits it into `max_steps` kernels, or
anything in between -- is the measurement, not a choice made here.

THE ONE PLACE THE SHAPE IS CHOSEN, AND WHY IT IS NOT A SCHEDULE CHOICE
---------------------------------------------------------------------
The reference's inner loop is data-dependent per ray: `if (block >= 0) { ...
goto done; }`. A tensor program has no `goto`, and the honest way to express
"this ray is finished" in array code is to keep marching and stop *accepting
results* -- carry an `active` mask, and make every update a `where(active & c,
new, old)`. That is the standard vectorised-raymarcher idiom and it is what
this file does. The alternative, a `while`/carry-until-converged form, is
expressible in TinyGrad only through low-level loop constructs that are
hand-staging, so it is not used.

The consequence is honest and is *the finding*, not a defect of the writing:
every ray runs the full `max_steps` iterations whether it hits on step 1 or
step 32, so the work is `max_steps` per pixel rather than the measured mean of
~26. The hand-written kernel's `#pragma unroll 4` (16 static steps, 38 SASS
instructions apart) exists to keep 16 *load chains* in flight, which is a
memory-level-parallelism argument about one ray's dependent chain; a masked
tensor loop has no dependent chain to overlap because all rays are already in
flight together. These are different machines doing the same arithmetic, and
the notes file measures both and does not pretend they are the same thing.

NO EARLY EXIT PER RAY, AND WHY THAT IS THE POINT
------------------------------------------------
`for i in range(max_steps)` -- the loop bound is the reference's `max_steps`,
its per-ray early exit becomes an `active` mask, and the whole body is one
unrolled chain. If TinyGrad fuses across iterations we get one kernel with
32 inlined steps; if it does not, we get one kernel per step with the ray
position materialised in between. Both are outcomes. The code does not choose
between them.
"""

from tinygrad import Tensor, dtypes

from .blocks import colour_lut, AIR

# The reference's `1e30f` guard for a zero ray component (kernels.cu:646-648).
_BIG = 1e30


def camera_basis(cameras: Tensor, fov_degrees: float, width: int, height: int) -> Tensor:
    """precompute_camera_basis_kernel (kernels.cu:511): [B, 5] -> [B, 14].

    cameras is (x, y, z, yaw, pitch). The output layout is the reference's:

        [0..2]  eye, with y raised by the eye height 1.6
        [3..5]  forward = (sin_yaw*cos_pitch, sin_pitch, cos_yaw*cos_pitch)
        [6..7]  right.x, right.z only -- right.y is identically 0
        [8..10] up
        [11]    scale_u = half_fov * aspect
        [12]    scale_v = half_fov

    Transcribed statement for statement, including the dropped right.y (a 3
    vector in 2 slots, which a compiler would have to be told) and the
    `fov * 0.01745329f` degree-to-radian constant. `--use_fast_math` turns
    sinf/cosf/tanf into approximate intrinsics in the reference; here they are
    TinyGrad's sin/cos/tan, which the CUDA backend lowers to the same MUFU
    sequence up to a last-ulp difference. The DDA comparison is a float
    compare, so a last-ulp difference can move a ray one voxel at an exact tie;
    that is why correctness is pinned to a CPU reference and to the DDA step
    count rather than to a bit-exact image against the GPU.
    """
    f32 = dtypes.float32
    cam_x, cam_y, cam_z = cameras[:, 0], cameras[:, 1], cameras[:, 2]
    yaw, pitch = cameras[:, 3], cameras[:, 4]
    zero = yaw * 0                       # a [B] zero, for broadcasting a scalar

    sin_yaw, cos_yaw = yaw.sin(), yaw.cos()
    sin_pitch, cos_pitch = pitch.sin(), pitch.cos()

    # tanf in float32 on the device, as the reference does -- not math.tan in
    # double, which is a different last-ulp behaviour from the one being
    # compared against.
    half_fov = Tensor([fov_degrees * 0.01745329 * 0.5], dtype=f32).tan()[0] + zero
    aspect = float(width) / float(height)

    eye = cam_x.stack(cam_y + 1.6, cam_z, dim=1).cast(f32)
    forward = (sin_yaw * cos_pitch).stack(sin_pitch, cos_yaw * cos_pitch, dim=1)
    right = cos_yaw.stack(-sin_yaw, dim=1)                    # right_y == 0
    up = (-sin_yaw * sin_pitch).stack(cos_pitch, -cos_yaw * sin_pitch, dim=1)
    scales = (half_fov * aspect).stack(half_fov, dim=1)
    return eye.cat(forward, right, up, scales, dim=1).contiguous()


def render(voxels: Tensor, basis: Tensor, width: int, height: int,
           max_steps: int, count_steps: bool = False):
    """The uint8 observation, [B, H, W, 3].

    voxels: [B, WY, WX, WZ] int8. basis: [B, 14] float32, from `camera_basis`.

    The body is the reference's `render_kernel_uint8_prebasis`
    (kernels.cu:568) written over arrays, in the order the reference does it:
    build the ray, normalise it, set up the grid traversal, loop, then look the
    block up in the colour table.

    `count_steps` returns `(obs, steps)` where `steps` is [B, 1, H, W] int32, the
    number of DDA bodies a ray executed. It is **instrumentation**, not part of
    the computation: it is the same loop with a counter, it cannot change `obs`
    (`verify.py` asserts the two are bit-identical), and every timing in this
    study is taken with it off. It exists because the step count is a discrete,
    tie-sensitive quantity: two implementations that take the same traversal
    decisions have identical step counts, and two that do not, do not. The
    image is a lossy readout of that -- many step counts collapse to the same
    colour -- so the step count is the sharper of the two checks.
    """
    B = voxels.shape[0]
    wy, wx, wz = voxels.shape[1], voxels.shape[2], voxels.shape[3]
    f32, i32 = dtypes.float32, dtypes.int32

    # ---- the eye and the basis, lifted to [B, 1, 1, 1] so they broadcast
    # over the pixel grid the way one scalar does per thread in the reference.
    # The slice is `i:i+1` and not `i`: `basis[:, i]` drops to rank 1, and
    # `[None, None]` on a rank-1 tensor gives (1, 1, B) -- the batch lands in
    # the wrong axis, which is invisible at B=1 and wrong at every other B.
    def comp(i):
        return basis[:, i:i + 1, None, None]

    eye_x, eye_y, eye_z = comp(0), comp(1), comp(2)
    fwd_x, fwd_y, fwd_z = comp(3), comp(4), comp(5)
    right_x, right_z = comp(6), comp(7)
    up_x, up_y, up_z = comp(8), comp(9), comp(10)
    scale_u, scale_v = comp(11), comp(12)

    # ---- the pixel grid. px runs along the last axis, py along the third, so
    # the output is already [B, 1, H, W] here and needs no transpose later.
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

    # kernels.cu:626-629, rsqrtf. tinygrad's `rsqrt` lowers to the same
    # MUFU.RSQ on sm_75.
    ray_len = (ray_x * ray_x + ray_y * ray_y + ray_z * ray_z).rsqrt()
    ray_x, ray_y, ray_z = ray_x * ray_len, ray_y * ray_len, ray_z * ray_len

    # ---- grid traversal setup, kernels.cu:638-659
    voxel_x = eye_x.floor()
    voxel_y = eye_y.floor()
    voxel_z = eye_z.floor()

    step_x = (ray_x >= 0).where(1.0, -1.0)
    step_y = (ray_y >= 0).where(1.0, -1.0)
    step_z = (ray_z >= 0).where(1.0, -1.0)

    # (ray != 0) ? 1/ray : 1e30. The `where` evaluates both branches and picks
    # one, which is the same trade the reference's ternary makes; the discarded
    # branch is an infinity, never a NaN, so nothing escapes.
    inv_x = (ray_x != 0).where(ray_x.reciprocal(), _BIG)
    inv_y = (ray_y != 0).where(ray_y.reciprocal(), _BIG)
    inv_z = (ray_z != 0).where(ray_z.reciprocal(), _BIG)

    t_delta_x, t_delta_y, t_delta_z = inv_x.abs(), inv_y.abs(), inv_z.abs()

    t_max_x = (ray_x >= 0).where((voxel_x + 1.0 - eye_x) * inv_x, (voxel_x - eye_x) * inv_x)
    t_max_y = (ray_y >= 0).where((voxel_y + 1.0 - eye_y) * inv_y, (voxel_y - eye_y) * inv_y)
    t_max_z = (ray_z >= 0).where((voxel_z + 1.0 - eye_z) * inv_z, (voxel_z - eye_z) * inv_z)

    # ---- the DDA, kernels.cu:664-679
    #
    # `hit` starts at AIR and `found` starts false. Each iteration reads the
    # voxel under the cursor, records the hit for a ray that is still looking
    # and still inside the world, then steps. A ray that has hit stops changing
    # `hit`; it keeps stepping, harmlessly, because every update is masked.
    #
    # Three things about this loop are deliberate and none of them is a
    # scheduling decision:
    #
    #  * `found` and `inside` are separate masks. The reference's `goto done`
    #    fires on a hit; its bounds test gates only the *load*. A ray that
    #    leaves the world has not hit anything, and conflating the two would
    #    corrupt the step count for exactly the rays that dominate this
    #    workload -- most of them run off the end of the world, not into a
    #    block.
    #
    #  * `inside` is re-evaluated every iteration, from the live cursor. The
    #    reference's `(unsigned)voxel_x < (unsigned)wx` is inside its loop for
    #    the same reason: the cursor moves.
    #
    #  * the gather index is clamped. That is memory safety, because a tensor
    #    gather has no bounds check and the reference's bounds test gates a
    #    pointer it simply never forms. `inside` is what decides whether the
    #    value read counts.
    #
    #  * the gather is written as a 4-index `voxels[b, y, x, z]`, which is the
    #    obvious way to say it and which tinygrad lowers to four predicated
    #    1-byte loads of the *same* address -- one per `bool` consumer of the
    #    step, because the compiler cannot prove the four predicates are
    #    related. The reference's `__ldg` is one `LDG.E.S8.CONSTANT.SYS`. That
    #    is 4x the read instructions on the hot path and it is a cost of the
    #    masked formulation, not a codegen failure. See notes/19 gap 7.
    #
    # The bounds predicate is written the obvious signed way. The reference
    # folds each axis into one `(unsigned)v < (unsigned)w`, which ptxas turns
    # into a single `ISETP.GE.U32.OR`; that fold is a source-level encoding
    # trick, not a different predicate, and it is left out here on purpose --
    # see notes/19, which prices it rather than quietly taking it.
    b_index = Tensor.arange(B, dtype=i32).reshape(B, 1, 1, 1)

    hit = Tensor.full((B, 1, height, width), AIR, dtype=i32)
    found = Tensor.full((B, 1, 1, 1), False, dtype=dtypes.bool)
    steps = Tensor.zeros((B, 1, height, width), dtype=i32)

    for step in range(max_steps):
        if count_steps:
            # the body runs once per ray per step until it hits, so the count is
            # the 1-based index of the step that hit, or max_steps
            steps = (~found).where(step + 1, steps)

        inside = ((voxel_x >= 0) & (voxel_x < float(wx)) &
                  (voxel_y >= 0) & (voxel_y < float(wy)) &
                  (voxel_z >= 0) & (voxel_z < float(wz)))
        ix = voxel_x.cast(i32).clip(0, wx - 1)
        iy = voxel_y.cast(i32).clip(0, wy - 1)
        iz = voxel_z.cast(i32).clip(0, wz - 1)
        block = voxels[b_index, iy, ix, iz].cast(i32)

        solid = (block >= 0) & inside & ~found
        hit = solid.where(block, hit)
        found = found | solid

        # the step itself, kernels.cu:672-678, as three selects
        step_taken_x = (t_max_x < t_max_y) & (t_max_x < t_max_z)
        step_taken_y = (t_max_y < t_max_z) & ~step_taken_x
        t_max_x = step_taken_x.where(t_max_x + t_delta_x, t_max_x)
        t_max_y = step_taken_y.where(t_max_y + t_delta_y, t_max_y)
        t_max_z = (~step_taken_x & ~step_taken_y).where(t_max_z + t_delta_z, t_max_z)
        voxel_x = step_taken_x.where(voxel_x + step_x, voxel_x)
        voxel_y = step_taken_y.where(voxel_y + step_y, voxel_y)
        voxel_z = (~step_taken_x & ~step_taken_y).where(voxel_z + step_z, voxel_z)

    # ---- the store, kernels.cu:690-699, as a table lookup
    #
    # `colour_lut` has 14 rows: 0..12 are the blocks, row 13 is the sky. A ray
    # that hit holds a block id in [0, 12]; a ray that did not still holds AIR
    # and is mapped to row 13. The reference reaches the sky by a *branch*
    # (`if (hit_block >= 0) ... else ...`); here both are the same gather.
    # That is a change of expression, not of schedule -- no knob, no fusion
    # hint, no tile size.
    lut = colour_lut()                                       # [14, 3] uint8
    colour_row = (hit >= 0).where(hit, 13)                   # [B, 1, H, W] int32
    obs = lut[colour_row].reshape(B, height, width, 3)       # [B, H, W, 3] uint8
    return (obs, steps) if count_steps else obs.contiguous()
