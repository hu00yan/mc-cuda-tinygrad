"""The reference kernel's *strategy*, translated into TinyGrad, crossings marked.

WHAT THIS IS, AND WHY IT IS NOT THE THIRD PORT
----------------------------------------------
`minecu_tg/render.py` wrote the DDA the way a person writes a vectorised
raymarcher: a Python `for` over `max_steps`, a live cursor, a masked update,
one kernel per step at small sizes and one fused kernel at large ones. It is a
faithful translation of the *computation* and a poor translation of the
*strategy*, and `notes/06-corrections-ledger.md` §5b is the write-up of what
that costs: it measured the port, not the compiler.

`minecu_tg_opt/render.py` fixed the algorithmic half of that (incremental
address, integer cursor, smallest live state) without touching the framework.

This file is the other half. It reproduces what
`render_kernel_uint8_prebasis` (`minecu_ref/src/kernels.cu:568`) actually does,
statement by statement, including the parts that live *below* the level at
which the abstraction offers any vocabulary at all. Where a technique cannot
be expressed in TinyGrad's normal vocabulary, the crossing is behind an
explicit switch in `crossings.py` and the cost of taking it is measured.

THE TECHNIQUES IN THE REFERENCE, AND WHERE EACH ONE LIVES
---------------------------------------------------------
Read off `kernels.cu:568-700` and re-derived from the SASS
(`minecu_ref/build/kernels_sm_75.sass`, `resources_sm_75.txt`):

  T1  one thread per pixel, `b = blockIdx.y`, `128` threads/block
      (`:584-585`, launched `:2182`). One block, one env.
  T2  the 13-float camera basis in `__shared__`, loaded by the first 14
      threads, then `__syncthreads()` (`:581`, `:594-597`). One barrier.
  T3  no per-pixel trig: the ray is `forward + u*right + v*up`, with
      `right.y` dropped from the basis layout (`:619-624`, basis at `:540-560`).
  T4  Amanatides-Woo with an *incrementally maintained* flat int address
      (`:669`, `b*wxyz + voxel_y*wxz + voxel_x*wz + voxel_z`, three signed
      strides), not a 4-index gather.
  T5  the unsigned bounds fold `(unsigned)voxel_x < (unsigned)wx`
      (`:666-668`) -> one `ISETP.GE.U32.OR` per axis.
  T6  a per-ray early exit: `if (block >= 0) { hit = block; goto done; }`
      (`:670`) -> `BSSY`/`BSYNC` per unrolled step, warp-divergence sync.
  T7  `#pragma unroll 4` x4 = **16 static copies** of the step body
      (`:681-687`), so 16 *independent* voxel loads are in flight. Measured:
      the reference's 16 `LDG.E.S8.CONSTANT.SYS` are at an exactly uniform
      spacing of 38 instructions (`notes/19` §4.0.1; re-derived in
      `sass.py`).
  T8  the 28-register footprint: 28 registers, 0 spill, 0 stack frame
      (`resources_sm_75.txt`). ptxas allocates the 16 identical steps one at
      a time.
  T9  a 14-row uint8 colour LUT as the whole epilogue (`:479`, `:691-699`):
      no float math, 3 `STG.E.U8`.

T1, T2, T3, T9 are ordinary array operations and need no crossing. T4, T5
are expressible in TinyGrad's vocabulary. **T6, T7 and T8 are not**, and they
are the point of this arm.

THE CENTRAL FINDING, STATED BEFORE THE NUMBERS
----------------------------------------------
T7 is not "a loop that was unrolled". It is a *loop with 16 copies of the
body inside it* -- a rolling window of 16 in-flight loads, re-entered until
`max_steps`. TinyGrad has no way to say that. What it does instead is
**inline the whole loop**: the graph is a straight line of 32 step bodies,
`rangeify` gives every one of them its own program, and the scheduler either
fuses all 32 into one kernel (36 384 SASS instructions, 168 registers,
20 932 B of spill, `notes/19` §4.1) or launches one per step. There is no
intermediate. The 16-copy window is the *specific* thing the reference has
and the generated code does not, and it is not reachable by any documented
TinyGrad knob, because no knob in TinyGrad expresses "unroll factor" for an
elementwise chain -- `OptOps.UNROLL` (`codegen/opt/__init__.py:7`) applies to
*reduce* axes only (`codegen/opt/heuristic.py:139-152`), and the DDA has none.

So: **the technique that most resists the abstraction is the register-budgeted
partial unroll (T7+T8) — keeping a fixed-width window of the DDA body live in
registers, with a per-ray early exit (T6) attached to each copy.** T6 is
class (i), not merely (ii): a per-ray trip count is not a fixed-size tensor,
and TinyGrad's IR has no loop construct at all (`Ops` in
`tinygrad/uop/ops.py` has no `WHILE`/`LOOP`/`REPEAT`; the only control flow
is `IF`/`ENDIF`, which `codegen/__init__.py:402-407` synthesises for gated
stores and nothing else).

WHAT IS AND IS NOT A CROSSING HERE
----------------------------------
`crossings.py` holds every one of them, each with a `file:line` in the
TinyGrad source, and `mode="clean"` / `mode="faithful"` selects. The two modes
differ ONLY in the crossing set. The computation, the ray, the loop bound and
the output are identical, which `verify.py` checks (bit-identical observation
and identical per-pixel DDA step counts across both modes).

`Tensor.custom_kernel` is not used anywhere in this package, and no `.cu` is
written or compiled alongside. The only CUDA that runs is CUDA that TinyGrad's
own renderer emitted.
"""

from __future__ import annotations

from tinygrad import Tensor, dtypes

from minecu_tg.blocks import AIR, colour_lut

# The reference's `1e30f` guard for a zero ray component (kernels.cu:646-648).
_BIG = 1e30


def camera_basis(cameras: Tensor, fov_degrees: float, width: int, height: int) -> Tensor:
    """`precompute_camera_basis_kernel` (kernels.cu:511): [B, 5] -> [B, 14].

    T3. Transcribed statement for statement, including the dropped
    `right.y` (a 3-vector in 2 slots, `kernels.cu:556-557`) and the
    `fov * 0.01745329` degree-to-radian constant (`:547`). One trig
    evaluation per env instead of five per pixel.

    This is a *crossing* in the reference's sense -- it is a separate kernel,
    so a separate launch -- and it is a crossing TinyGrad does not have to
    make: whether this graph fuses with the render's is a scheduling decision,
    not a source-level one. Both modes keep it in one place; whether it costs
    a launch is measured, not assumed.
    """
    f32 = dtypes.float32
    cam_x, cam_y, cam_z = cameras[:, 0], cameras[:, 1], cameras[:, 2]
    yaw, pitch = cameras[:, 3], cameras[:, 4]
    zero = yaw * 0                       # a [B] zero, for broadcasting a scalar

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


def _dda_prologue(voxels: Tensor, basis: Tensor, width: int, height: int,
                  addr_bits: int, unsigned_bounds: bool):
    """Everything `kernels.cu:619-659` does, before the loop.

    Returns the per-ray state. `addr_bits` and `unsigned_bounds` are the two
    *shape* crossings (T5 and the address width of T4); both default to the
    values the reference uses, and both are reachable from TinyGrad's normal
    vocabulary, so they are not knobs -- they are recorded here because the
    boundary map needs to know they were tested both ways.
    """
    B = voxels.shape[0]
    wy, wx, wz = voxels.shape[1], voxels.shape[2], voxels.shape[3]
    f32, i32 = dtypes.float32, dtypes.int32
    idt = dtypes.int64 if addr_bits == 64 else i32

    # The basis row is [B, 14]; slice `i:i+1` and add two axes so it
    # broadcasts over the pixel grid the way one scalar does per thread.
    # `i` (not `i:i+1`) drops to rank 1, and `[None, None]` on a rank-1
    # tensor is (1, 1, B) -- the batch lands in the wrong axis, invisible at
    # B=1 and wrong at every other B. `minecu_tg/render.py:139-140` and
    # `minecu_tg/verify.py` both carry this; it is reproduced here because it
    # is a real trap, not a decoration.
    def comp(i):
        return basis[:, i:i + 1, None, None]

    eye_x, eye_y, eye_z = comp(0), comp(1), comp(2)
    fwd_x, fwd_y, fwd_z = comp(3), comp(4), comp(5)
    right_x, right_z = comp(6), comp(7)
    up_x, up_y, up_z = comp(8), comp(9), comp(10)
    scale_u, scale_v = comp(11), comp(12)

    # px along the last axis, py along the third: the output is already
    # [B, 1, H, W] and needs no transpose.
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

    # kernels.cu:638-640, the starting cursor. The reference's is an `int`
    # from `(int)floorf(eye)`, so it is cast ONCE here and never again.
    vx, vy, vz = (eye_x.floor().cast(i32), eye_y.floor().cast(i32),
                  eye_z.floor().cast(i32))

    # kernels.cu:642-644
    pos_x, pos_y, pos_z = ray_x >= 0, ray_y >= 0, ray_z >= 0
    sign_x, sign_y, sign_z = (p.where(1, -1).cast(i32) for p in (pos_x, pos_y, pos_z))

    # kernels.cu:646-652, the reciprocal guard and t_delta.
    inv_x = (ray_x != 0).where(ray_x.reciprocal(), _BIG)
    inv_y = (ray_y != 0).where(ray_y.reciprocal(), _BIG)
    inv_z = (ray_z != 0).where(ray_z.reciprocal(), _BIG)
    td_x, td_y, td_z = inv_x.abs(), inv_y.abs(), inv_z.abs()

    # kernels.cu:654-659, t_max. Uses the integer cursor cast to float, which
    # is exact for |cursor| < 2^24 and is the reference's arithmetic: the
    # reference computes `((float)(voxel_x + 1) - eye_x) * inv_ray_x`, i.e.
    # the int is widened, not the float narrowed.
    fx, fy, fz = vx.cast(f32), vy.cast(f32), vz.cast(f32)
    t_max_x = pos_x.where((fx + 1.0 - eye_x) * inv_x, (fx - eye_x) * inv_x)
    t_max_y = pos_y.where((fy + 1.0 - eye_y) * inv_y, (fy - eye_y) * inv_y)
    t_max_z = pos_z.where((fz + 1.0 - eye_z) * inv_z, (fz - eye_z) * inv_z)

    # T4: the maintained flat address. The reference forms
    # `b*wxyz + voxel_y*wxz + voxel_x*wz + voxel_z` inside the loop
    # (kernels.cu:669) from three int coordinates, and ptxas strength-reduces
    # it; the effect in the SASS is 4 `IMAD`/`IADD3` per step, which is the
    # same shape as an increment by a precomputed stride. Here the address is
    # maintained explicitly and advanced by a precomputed signed stride, which
    # is the same arithmetic with the loop-invariant part hoisted. This is the
    # *form* the algorithm takes once it is written down; it is the same in
    # both modes, so the clean-vs-faithful delta never hides here.
    wxz, wxyz = wx * wz, wx * wy * wz
    b_off = (Tensor.arange(B, dtype=idt) * wxyz).reshape(B, 1, 1, 1)
    addr = b_off + (vy.cast(idt) * wxz + vx.cast(idt) * wz + vz.cast(idt))
    stride = (sign_x.cast(idt) * wz, sign_y.cast(idt) * wxz, sign_z.cast(idt))

    # T5: the bounds test, computed once per ray as a countdown of steps to
    # exit rather than re-derived from the cursor every step. `inside` is
    # three integer compares on values that are already live.
    #
    # The unsigned form is the reference's `(unsigned)v < (unsigned)w`
    # (kernels.cu:666-668). On sm_75 both forms are three instructions
    # (`notes/19` §3.2 finding 1, `ISETP.GE.U32.OR` against three signed
    # compares), so this is a *fidelity* choice, not a performance one, and
    # the SASS census in `sass.py` reports both so that claim is checkable
    # rather than inherited.
    if unsigned_bounds:
        def steps_left(c, s, p):
            inside_axis = (c >= 0) & (c < s)
            clamped = c.clip(0, s - 1)
            return inside_axis.where(p.where(s - 1 - clamped, clamped), -1).cast(i32)
    else:
        def steps_left(c, s, p):
            return (c.cast(f32) >= 0).where(
                (c.cast(f32) < float(s)).where(p.where(s - 1 - c, c), -1), -1).cast(i32)

    n_x, n_y, n_z = (steps_left(c, s, p) for c, s, p in
                     ((vx, wx, pos_x), (vy, wy, pos_y), (vz, wz, pos_z)))

    return dict(B=B, wy=wy, wx=wx, wz=wz, f32=f32, i32=i32, idt=idt,
                flat=voxels.reshape(B * wx * wy * wz),
                t_max=(t_max_x, t_max_y, t_max_z), t_delta=(td_x, td_y, td_z),
                addr=addr, stride=stride, n=(n_x, n_y, n_z),
                width=width, height=height)


def _dda_step(S, hit, step, live):
    """One iteration of the reference's `DDA_STEP_PRE` macro (kernels.cu:664-679).

    The reference's body is:
        if (step >= max_steps) goto done;
        if ((unsigned)vx < (unsigned)wx && ...) {
            block = __ldg(&voxels[addr]);
            if (block >= 0) { hit_block = block; goto done; }
        }
        if (t_max_x < t_max_y && t_max_x < t_max_z) { t_max_x += t_delta_x; vx += step_x; }
        else if (t_max_y < t_max_z) { ... } else { ... }
        ++step;

    Three things are deliberately identical to it and one is not:

      * the bounds test is `inside` from the countdown, and it gates the
        *load*, exactly as the reference's gates its `__ldg` and nothing else;
      * the axis selection is the same three-way compare in the same order,
        as three selects;
      * `t_delta` and the step signs are loop-invariant in the reference and
        are hoisted out of the loop here.

    The one that is not: `live` (`hit < 0`, i.e. "this ray has not hit yet")
    is a *mask*, where the reference has a *branch*. That is T6, and it is
    the crossing that has no clean form. See `crossings.py:EARLY_EXIT`.
    """
    i32 = S["i32"]
    t_max_x, t_max_y, t_max_z = S["t_max"]
    td_x, td_y, td_z = S["t_delta"]
    addr, (n_x, n_y, n_z) = S["addr"], S["n"]

    inside = (n_x >= 0) & (n_y >= 0) & (n_z >= 0)

    # One gather, one address. A tensor gather has no bounds check while the
    # reference's test gates a pointer it never forms, so the index is
    # `inside.where(addr, 0)`: memory safety, and `inside` decides whether
    # the byte read counts. The 4-index form in `minecu_tg/render.py:247` is
    # the same read expressed four times over; this is the T4 form.
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

    # T4: the address advance. One add, by a stride that already carries the
    # sign. The three per-axis coordinates are gone from the loop entirely,
    # which is what makes maintaining the address a win rather than a fourth
    # piece of state -- and it is the reference's own shape, once ptxas has
    # strength-reduced its 4 IMADs (see `sass.py:step_address_ops`).
    st_x, st_y, st_z = S["stride"]
    S["addr"] = addr + take_x.where(st_x, take_y.where(st_y, st_z))
    S["n"] = (take_x.where(n_x - 1, n_x), take_y.where(n_y - 1, n_y),
              take_z.where(n_z - 1, n_z))
    return hit


def render(voxels: Tensor, basis: Tensor, width: int, height: int,
           max_steps: int, count_steps: bool = False, mode: str = "clean",
           **crossing_kw):
    """The uint8 observation, [B, H, W, 3], in `mode`.

    voxels: [B, WY, WX, WZ] int8. basis: [B, 14] float32 from `camera_basis`.

    `mode` is `"clean"` or `"faithful"`; it selects the crossing set and
    nothing else. `crossing_kw` overrides individual crossings, and
    `crossings.effective()` is the single place the selection happens, so
    there is exactly one switch to read.

    A `ContextVar` crossing is applied by entering the `Context` that
    `crossings.py` names for this mode, so a knob cannot be added to
    `crossings.py` and then silently not take effect. **That context is empty
    in both modes**: every knob that reaches this graph was measured
    (`probe_knobs.py`) and none of them buys anything, which is a result and
    not an absence. The only switch that differs between the modes is
    `addr_bits`, an edit point, and §6.1 of `notes/28` prices what it costs.

    The loop is a Python `for` over `max_steps` in both modes. That is
    deliberate: the reference's `#pragma unroll 4` is a *source* directive
    that nvcc implements, and TinyGrad's equivalent of "unroll the loop" is
    "the graph contains the body `max_steps` times", which a Python `for`
    gives. What TinyGrad then does with that straight line -- fuse it, split
    it, spill it -- is the measurement, and neither mode decides it.

    `count_steps` returns `(obs, steps)`, the number of DDA bodies a ray
    executed. It is **instrumentation**: the same loop with a counter, it
    cannot change `obs` (`verify.py` asserts the two are bit-identical), and
    every timing is taken with it off. It exists because the step count is
    tie-sensitive and discrete: two implementations that take the same
    traversal decisions have identical step counts, and many step counts
    collapse to the same colour in the image, so the step count is the
    sharper of the two checks.
    """
    from tinygrad.helpers import Context

    from .crossings import effective

    cross = effective(mode, **crossing_kw)
    # `cross["context"]` is applied here, and only here, so that adding a knob
    # to `crossings.py` cannot fail to take effect. It is currently empty in
    # both modes -- every knob that reaches this graph was measured and none of
    # them buys anything -- and the plumbing exists so that the reason is a
    # measurement rather than an absence of a place to put one.
    with Context(**cross["context"]):
        return _render_with(voxels, basis, width, height, max_steps,
                            count_steps, cross)


def _render_with(voxels, basis, width, height, max_steps, count_steps, cross):
    S = _dda_prologue(voxels, basis, width, height,
                      addr_bits=cross["addr_bits"],
                      unsigned_bounds=cross["unsigned_bounds"])
    B, height, i32 = S["B"], S["height"], S["i32"]

    # AIR == -1 and `hit` is only ever assigned a non-negative block id, so
    # `hit < 0` *is* "has not hit yet" and the separate `found` flag that
    # `minecu_tg/render.py:232` carries is redundant. This is the same
    # reduction `minecu_tg_opt/render.py:_step_lean` makes, reached from the
    # reference's own initialisation rather than from a flag.
    hit = Tensor.full((B, 1, height, S["width"]), AIR, dtype=i32)
    steps = Tensor.zeros((B, 1, height, S["width"]), dtype=i32)

    for k in range(max_steps):
        live = hit < 0                       # T6's mask, in place of a branch
        if count_steps:
            # the body runs once per ray per step until it hits, so the count
            # is the 1-based index of the step that hit, or max_steps
            steps = live.where(k + 1, steps)
        hit = _dda_step(S, hit, k, live)

    # T9: `kernels.cu:691-699`. The reference reaches the sky through a
    # *branch* and the hit colour through a `__constant__` LUT store. A
    # tensor program has neither, so AIR is mapped to row 13 of a 14-row
    # table and both cases are one gather. That is a change of expression,
    # not of schedule, and it is the same in both modes.
    lut = colour_lut()
    colour_row = (hit >= 0).where(hit, 13)
    obs = lut[colour_row].reshape(B, height, S["width"], 3)
    return (obs, steps) if count_steps else obs.contiguous()
