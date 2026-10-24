"""A float64 CPU reference of the same algorithm.

WHY A SEPARATE REFERENCE, INSTEAD OF COMPARING TO THE GPU. The two renderers
are not bit-comparable and it is not worth pretending otherwise:

  * the reference runs `--use_fast_math`, so `sinf`/`cosf`/`tanf`/`rsqrtf` and
    the divisions are approximate, and `__sincosf` is the fast intrinsic;
  * TinyGrad's CUDA backend lowers `sin`/`cos`/`tan`/`rsqrt` to its own
    sequence, and its `/` is a plain divide;
  * a DDA step is chosen by `t_max_x < t_max_y`, so a one-ulp difference in a
    `t_max` can move a ray to a different voxel, and then the image differs by
    a whole block colour.

So the contract this module enforces is: **the tensor render agrees with an
independent float64 transcription of the same algorithm.** The float64
transcription is written from `notes/17-minecu-spec.md` and from the reference's
source, and it is checked against the reference's own CPU model
(`minecu_ref/bench/dda_model.py`, an independent third transcription) so that a
bug in this file cannot silently become the definition of correct.

The two output-level claims `verify.py` makes are therefore:

  1. the TinyGrad image equals this float64 image, exactly, on uint8 colours;
  2. the mean DDA steps per pixel agree to 3 decimal places with the
     reference's own measured 26.416 at the on-surface configuration.

Claim 2 is the one that actually pins the *workload*, because it is a
discrete, tie-sensitive quantity: if the two implementations made different
traversal decisions, the step count would move.
"""

import math

import numpy as np

from .blocks import BLOCK_COLORS_U8, SKY_COLOR_U8

AIR = -1
# The 14-row table: 0..12 are the blocks, row 13 is the sky. Same table the
# tensor version gathers from, so the comparison isolates the DDA and not the
# palette.
LUT = np.array(BLOCK_COLORS_U8 + [SKY_COLOR_U8], dtype=np.uint8)


def camera_basis(cameras: np.ndarray, fov_degrees: float, width: int, height: int) -> np.ndarray:
    """`render.camera_basis`, in float64, for one camera.

    Returns the same 14 numbers the kernel stores, in the same order, so the
    comparison is not sensitive to the packing.
    """
    cam_x, cam_y, cam_z, yaw, pitch = (float(np.reshape(cameras, -1)[i]) for i in range(5))
    sin_yaw, cos_yaw = math.sin(yaw), math.cos(yaw)
    sin_pitch, cos_pitch = math.sin(pitch), math.cos(pitch)
    half_fov = math.tan(fov_degrees * 0.01745329 * 0.5)
    aspect = width / height
    return np.array([
        cam_x, cam_y + 1.6, cam_z,
        sin_yaw * cos_pitch, sin_pitch, cos_yaw * cos_pitch,
        cos_yaw, -sin_yaw,
        -sin_yaw * sin_pitch, cos_pitch, -cos_yaw * sin_pitch,
        half_fov * aspect, half_fov,
    ], dtype=np.float64)


def render_one(voxels: np.ndarray, basis: np.ndarray, width: int, height: int,
               max_steps: int):
    """One env. voxels is [WY, WX, WZ] int8; basis is the 14-vector.

    Returns (image [H, W, 3] uint8, steps [H, W] int32), the step count being
    the number of times the DDA body ran, which is the quantity the reference's
    device-side probe reports.
    """
    wy, wx, wz = voxels.shape
    out = np.empty((height, width, 3), dtype=np.uint8)
    steps = np.zeros((height, width), dtype=np.int32)

    eye_x, eye_y, eye_z = basis[0], basis[1], basis[2]
    fwd_x, fwd_y, fwd_z = basis[3], basis[4], basis[5]
    right_x, right_z = basis[6], basis[7]
    up_x, up_y, up_z = basis[8], basis[9], basis[10]
    scale_u, scale_v = basis[11], basis[12]

    for py in range(height):
        for px in range(width):
            # kernels.cu:619-629
            u = (2.0 * (px + 0.5) / width - 1.0) * scale_u
            v = (1.0 - 2.0 * (py + 0.5) / height) * scale_v
            ray_x = fwd_x + u * right_x + v * up_x
            ray_y = fwd_y + v * up_y
            ray_z = fwd_z + u * right_z + v * up_z
            rl = 1.0 / math.sqrt(ray_x * ray_x + ray_y * ray_y + ray_z * ray_z)
            ray_x, ray_y, ray_z = ray_x * rl, ray_y * rl, ray_z * rl

            # kernels.cu:638-659
            vx, vy, vz = int(math.floor(eye_x)), int(math.floor(eye_y)), int(math.floor(eye_z))
            sx = 1 if ray_x >= 0 else -1
            sy = 1 if ray_y >= 0 else -1
            sz = 1 if ray_z >= 0 else -1
            ix = 1.0 / ray_x if ray_x != 0 else 1e30
            iy = 1.0 / ray_y if ray_y != 0 else 1e30
            iz = 1.0 / ray_z if ray_z != 0 else 1e30
            dtx, dty, dtz = abs(ix), abs(iy), abs(iz)
            tmx = (vx + 1 - eye_x) * ix if ray_x >= 0 else (vx - eye_x) * ix
            tmy = (vy + 1 - eye_y) * iy if ray_y >= 0 else (vy - eye_y) * iy
            tmz = (vz + 1 - eye_z) * iz if ray_z >= 0 else (vz - eye_z) * iz

            # kernels.cu:664-687. `n` counts body executions, which is what the
            # reference's probe reports as "DDA steps/pixel".
            hit = AIR
            n = 0
            for _ in range(max_steps):
                n += 1
                if 0 <= vx < wx and 0 <= vy < wy and 0 <= vz < wz:
                    blk = int(voxels[vy, vx, vz])
                    if blk >= 0:
                        hit = blk
                        break
                if tmx < tmy and tmx < tmz:
                    tmx += dtx; vx += sx
                elif tmy < tmz:
                    tmy += dty; vy += sy
                else:
                    tmz += dtz; vz += sz
            steps[py, px] = n
            out[py, px] = LUT[13 if hit < 0 else hit]
    return out, steps
