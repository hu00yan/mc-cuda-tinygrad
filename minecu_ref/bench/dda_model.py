#!/usr/bin/env python3
"""dda_model.py -- a CPU reference of mine.cu's render kernels, used to

  (a) pin the exact DDA semantics that notes/17-minecu-spec.md specifies, and
  (b) count, on the exact world the project generates, how many DDA steps a
      pixel actually takes, which is what decides whether the 49.6M env-steps/s
      headline is arithmetically reachable on an RTX 3090.

THIS IS NOT A BENCHMARK. It runs on the CPU in numpy and measures nothing about
any GPU. Its only output is a step count and a pixel checksum, both of which are
reproducible offline. Any timing in this study must come from the CUDA binary
in minecu_ref/bench/, never from here.

The kernel transcribed is render_kernel_uint8_prebasis (kernels.cu:568), which
is the one the uint8 step paths use, and the DDA body is byte-for-byte the same
macro as render_kernel (kernels.cu:225-255) apart from the colour LUT and the
absence of fog. Both are transcribed in the same function with a flag.

Usage:
    python3 dda_model.py --batch 64 --res 32 --world 16 --max-steps 32
    python3 dda_model.py --out stepcount.json
"""

import argparse
import hashlib
import json
import math

import numpy as np

AIR = -1
GRASS, DIRT, STONE, OAKLOG, LEAVES, SAND, WATER, GLASS, BRICK, COBBLE, PLANKS, SNOW, BEDROCK = range(13)

# kernels.cu:479 BLOCK_COLORS_U8, kernels.cu:495 SKY_COLOR_U8
BLOCK_COLORS_U8 = np.array(
    [[76, 166, 51], [140, 89, 51], [127, 127, 127], [140, 102, 64], [51, 140, 38],
     [229, 217, 153], [51, 102, 204], [217, 229, 242], [178, 89, 76], [102, 102, 102],
     [191, 153, 102], [242, 242, 250], [38, 38, 38]], dtype=np.uint8)
SKY_COLOR_U8 = np.array([135, 206, 235], dtype=np.uint8)

# kernels.cu:27 BLOCK_COLORS, kernels.cu:44 SKY_COLOR, the float32 source tables
BLOCK_COLORS_F32 = np.array(
    [[0.30, 0.65, 0.20], [0.55, 0.35, 0.20], [0.50, 0.50, 0.50], [0.55, 0.40, 0.25],
     [0.20, 0.55, 0.15], [0.90, 0.85, 0.60], [0.20, 0.40, 0.80], [0.85, 0.90, 0.95],
     [0.70, 0.35, 0.30], [0.40, 0.40, 0.40], [0.75, 0.60, 0.40], [0.95, 0.95, 0.98],
     [0.15, 0.15, 0.15]], dtype=np.float32)
SKY_COLOR_F32 = np.array([0.53, 0.81, 0.92], dtype=np.float32)


def build_world(world: int, ground: int, with_tree: bool = True, with_house: bool = True) -> np.ndarray:
    """The exact world minecu's _generate_world produces, [Y, X, Z] int8.

    generate_world_kernel (kernels.cu:1793): BEDROCK at y==0, DIRT for
    0 < y < ground-1, GRASS at y == ground-1, AIR above. Every env gets the same
    world; there is no seed use.
    """
    vox = np.full((world, world, world), AIR, dtype=np.int8)
    vox[0, :, :] = BEDROCK
    vox[1:ground - 1, :, :] = DIRT
    vox[ground - 1, :, :] = GRASS
    if with_tree:
        # place_tree_kernel (kernels.cu:1831): 3 log at y = ground+dy, leaf at ground+3
        tx, tz = world // 2, world // 2 + 2
        for dy in range(3):
            y = ground + dy
            if y < world:
                vox[y, tx, tz] = OAKLOG
        if ground + 3 < world:
            vox[ground + 3, tx, tz] = LEAVES
    if with_house:
        # place_house_kernel (kernels.cu:1868): 5 wide x 4 deep x 4 tall at
        # (world//2 + 6, world//2 - 2), so entirely OUTSIDE a 16-wide world
        # (x = 14 with W=5 needs x < 16 for dx<2, only dx=0,1 land). Kept here
        # for fidelity; the SET_VOXEL bounds check drops what does not fit.
        hx, hz = world // 2 + 6, world // 2 - 2
        base = ground
        W_, D_, H_ = 5, 4, 4

        def put(dy, dx, dz, blk):
            vx, vy, vz = hx + dx, base + dy, hz + dz
            if 0 <= vx < world and 0 <= vy < world and 0 <= vz < world:
                vox[vy, vx, vz] = blk

        for dx in range(W_):
            for dz in range(D_):
                put(0, dx, dz, COBBLE)
        for dy in range(1, 4):
            for dx in range(W_):
                if dx == 2 and dy <= 2:
                    continue
                put(dy, dx, 0, PLANKS)
                put(dy, dx, D_ - 1, PLANKS)
            for dz in range(1, D_ - 1):
                put(dy, 0, dz, PLANKS)
                put(dy, W_ - 1, dz, PLANKS)
        for dy in range(1, 4):
            put(dy, 0, 0, OAKLOG)
            put(dy, W_ - 1, 0, OAKLOG)
            put(dy, 0, D_ - 1, OAKLOG)
            put(dy, W_ - 1, D_ - 1, OAKLOG)
        for dx in range(W_):
            for dz in range(D_):
                put(H_, dx, dz, BRICK)
    return vox


def camera_basis(yaw: float, pitch: float, fov: float, width: int, height: int):
    """precompute_camera_basis_kernel (kernels.cu:511), float32 throughout."""
    f = np.float32
    yaw, pitch = f(yaw), f(pitch)
    sy, cy = np.sin(yaw, dtype=f), np.cos(yaw, dtype=f)
    sp, cp = np.sin(pitch, dtype=f), np.cos(pitch, dtype=f)
    half_fov = f(np.tan(f(fov * f(0.01745329)) * f(0.5)))
    aspect = f(width) / f(height)
    return dict(
        forward=(f(sy * cp), f(sp), f(cy * cp)),
        right=(f(cy), f(0.0), f(-sy)),
        up=(f(-sy * sp), f(cp), f(-cy * sp)),
        scale_u=f(half_fov * aspect),
        scale_v=half_fov,
    )


def render(voxels: np.ndarray, eye, basis, width: int, height: int, max_steps: int,
           pitch: float = -0.2, yaw: float = 0.0, variant: str = "prebasis",
           view_distance: float = 32.0, ground_height: int = 8):
    """One render kernel, transcribed statement by statement.

    variant:
      prebasis  render_kernel_uint8_prebasis (kernels.cu:568). No AABB, no
                view_distance test, uint8 out, basis from smem. 776 SASS
                instructions, 38.4 per DDA step.
      fp32      render_kernel (kernels.cu:87). Adds the AABB entry/exit test,
                the `t >= view_distance` test inside the DDA, the face shading
                and the fog mix, and writes float32. 1224 SASS instructions,
                58.4 per DDA step.

    Returns (image, step_count). The image dtype is uint8 for `prebasis` and
    float32 for `fp32`. The step count is the number of times the DDA body ran,
    which is what both the 16x SASS unroll and the branch nest scale with.
    """
    f = np.float32
    # fp32 and fast differ in the OUTPUT dtype as well as the body:
    #   render_kernel                 -> float32*   (kernels.cu:90)
    #   render_kernel_uint8_prebasis  -> uint8_t*   (kernels.cu:571)
    #   render_kernel_uint8_fast      -> uint8_t*   (kernels.cu:1272)
    out_dtype = np.float32 if variant == "fp32" else np.uint8
    out = np.empty((height, width, 3), dtype=out_dtype)
    steps = np.zeros((height, width), dtype=np.int32)
    ex, ey, ez = (f(v) for v in eye)
    fx, fy, fz = basis["forward"]
    rx, ry, rz = basis["right"]
    ux, uy, uz = basis["up"]
    su, sv = basis["scale_u"], basis["scale_v"]
    W = voxels.shape[0]

    for py in range(height):
        for px in range(width):
            # kernels.cu:619-624, fp32
            u = f(f(f(2.0) * f(px + 0.5) / f(width) - f(1.0)) * su)
            v = f(f(f(1.0) - f(f(2.0) * f(py + 0.5) / f(height))) * sv)
            ray_x = f(fx + f(u * rx) + f(v * ux))
            ray_y = f(fy + f(v * uy))
            ray_z = f(fz + f(u * rz) + f(v * uz))
            # rsqrtf, then multiply (kernels.cu:626-629)
            rl = f(np.float32(1.0) / f(np.sqrt(f(f(ray_x * ray_x) + f(ray_y * ray_y) + f(ray_z * ray_z)))))
            ray_x, ray_y, ray_z = f(ray_x * rl), f(ray_y * rl), f(ray_z * rl)

            # ray_intersects_aabb, kernels.cu:51. Only the fp32 and fast
            # kernels call it; prebasis and minimal skip it entirely.
            if variant in ("fp32", "fast"):
                ivx = f(1.0) / ray_x if ray_x != 0 else f(1e30)
                ivy = f(1.0) / ray_y if ray_y != 0 else f(1e30)
                ivz = f(1.0) / ray_z if ray_z != 0 else f(1e30)
                hi = f(W)
                ts = [(f(0.0) - ex) * ivx, (hi - ex) * ivx,
                      (f(0.0) - ey) * ivy, (hi - ey) * ivy,
                      (f(0.0) - ez) * ivz, (hi - ez) * ivz]
                tmin = max(min(ts[0], ts[1]), min(ts[2], ts[3]), min(ts[4], ts[5]))
                tmax_ = min(max(ts[0], ts[1]), max(ts[2], ts[3]), max(ts[4], ts[5]))
                hits = not (tmax_ < 0 or tmin > tmax_)
                if not hits or (variant == "fp32" and tmin > f(view_distance)):
                    out[py, px] = (SKY_COLOR_F32 if variant == "fp32" else SKY_COLOR_U8)
                    steps[py, px] = 0   # the DDA never runs
                    continue

            vx = int(np.floor(ex)); vy = int(np.floor(ey)); vz = int(np.floor(ez))
            sx = 1 if ray_x >= 0 else -1
            sy_ = 1 if ray_y >= 0 else -1
            sz = 1 if ray_z >= 0 else -1
            ix = f(1.0) / ray_x if ray_x != 0 else f(1e30)
            iy = f(1.0) / ray_y if ray_y != 0 else f(1e30)
            iz = f(1.0) / ray_z if ray_z != 0 else f(1e30)
            dx_, dy_, dz_ = f(abs(ix)), f(abs(iy)), f(abs(iz))
            tmx = f(f(vx + 1) - ex) * ix if ray_x >= 0 else f(f(vx) - ex) * ix
            tmy = f(f(vy + 1) - ey) * iy if ray_y >= 0 else f(f(vy) - ey) * iy
            tmz = f(f(vz + 1) - ez) * iz if ray_z >= 0 else f(f(vz) - ez) * iz

            hit = AIR
            n = 0
            t = f(0.0)
            face = -1
            for step in range(max_steps):
                # fp32 path only: kernels.cu:226 `t >= view_distance`
                if variant == "fp32" and t >= f(view_distance):
                    break
                n += 1
                if 0 <= vx < W and 0 <= vy < W and 0 <= vz < W:
                    blk = int(voxels[vy, vx, vz])
                    if blk >= 0:
                        hit = blk
                        break
                # fp32 path only: kernels.cu:1386, ground early-out
                if variant == "fast" and vy < ground_height:
                    hit = GRASS if vy == ground_height - 1 else DIRT
                    break
                if tmx < tmy and tmx < tmz:
                    t = tmx; tmx = f(tmx + dx_); vx += sx; face = 0
                elif tmy < tmz:
                    t = tmy; tmy = f(tmy + dy_); vy += sy_; face = 1
                else:
                    t = tmz; tmz = f(tmz + dz_); vz += sz; face = 2
            steps[py, px] = n
            if variant == "fp32":
                # fog is computed from t, which the fp32 path tracks; the
                # prebasis and fast paths have no t at all.
                if hit >= 0:
                    col = BLOCK_COLORS_F32[hit]
                    shade = 1.0
                    if face == 0:
                        shade = 0.8
                    elif face == 2:
                        shade = 0.9
                    elif face == 1 and sy_ < 0:
                        shade = 0.6
                    fog = min(t / f(view_distance), f(1.0))
                    fog = f(fog * fog)
                    out[py, px] = [
                        f(f(col[0] * f(shade) * f(f(1.0) - fog)) + f(f(SKY_COLOR_F32[0] * fog))),
                        f(f(col[1] * f(shade) * f(f(1.0) - fog)) + f(f(SKY_COLOR_F32[1] * fog))),
                        f(f(col[2] * f(shade) * f(f(1.0) - fog)) + f(f(SKY_COLOR_F32[2] * fog))),
                    ]
                else:
                    out[py, px] = SKY_COLOR_F32
            else:
                out[py, px] = SKY_COLOR_U8 if hit < 0 else BLOCK_COLORS_U8[hit]
    return out, steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", type=int, default=16)
    ap.add_argument("--res", type=int, default=32)
    ap.add_argument("--max-steps", type=int, default=32)
    ap.add_argument("--ground", type=int, default=8)
    ap.add_argument("--fov", type=float, default=70.0)
    ap.add_argument("--pitch", type=float, default=0.0)
    ap.add_argument("--yaw", type=float, default=0.0)
    ap.add_argument("--variant", default="prebasis",
                    choices=("prebasis", "fp32", "fast"),
                    help="which render kernel to model; prebasis=uint8_prebasis, "
                         "fp32=render_kernel, fast=render_kernel_uint8_fast")
    ap.add_argument("--view-distance", type=float, default=32.0)
    ap.add_argument("--batch", type=int, default=32768)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    vox = build_world(a.world, a.ground)
    solid = int((vox >= 0).sum())
    spawn = (a.world / 2.0, float(a.ground + 2), a.world / 2.0)
    eye = (spawn[0], spawn[1] + 1.6, spawn[2])  # kernels.cu:541 basis[1] = cam_y + 1.6
    basis = camera_basis(a.yaw, a.pitch, a.fov, a.res, a.res)

    img, steps = render(vox, eye, basis, a.res, a.res, a.max_steps, a.pitch, a.yaw,
                        variant=a.variant, view_distance=a.view_distance, ground_height=a.ground)

    B = a.batch
    pixels = B * a.res * a.res
    mean_steps = float(steps.mean())
    # Only render_kernel writes the float32 SKY_COLOR; the uint8 kernels all
    # write SKY_COLOR_U8, including render_kernel_uint8_fast, which converts
    # the same float mix with (uint8_t)(x*255).
    sky_ref = SKY_COLOR_F32 if a.variant == "fp32" else SKY_COLOR_U8
    sky_tol = 1e-3 if a.variant == "fp32" else 0.5
    # Cast before subtracting: on a uint8 image `img - sky_ref` wraps around,
    # so 0 - 200 becomes 56 and every non-sky pixel looks close to sky. The
    # .astype(np.int32) is load-bearing, not decoration.
    sky_mask = np.abs(img.astype(np.float64) - sky_ref.astype(np.float64)).max(axis=2) < sky_tol
    solid_frac = float(sky_mask.mean())
    out_dtype = "float32" if a.variant == "fp32" else "uint8"

    # SASS geometry, MEASURED off build/kernels_sm_75.sass (nvcc 12.9, -O3
    # --use_fast_math, -arch=sm_75) on this host. The per-step figure is not
    # estimated: it is the load-to-load spacing inside the unrolled DDA body,
    # which is a run of identical values.
    #   prebasis  776 instrs, 16 DDA loads, spacing exactly 38, so 38/step.
    #   fp32     1224 instrs, 16 DDA loads, spacing 59,59,59,62 repeating,
    #            so (953+... )/16 = 59.6/step.
    # "fast" is the same body as fp32 plus a ground test, so it takes fp32's
    # per-step cost; its prologue/epilogue are the fp32 ones too. That is an
    # approximation and is labelled as one where it is used.
    SASS = {
        "prebasis": (776, 152, 38.0, 54),
        "fp32":     (1224, 243, 59.6, 87),
        "fast":     (1296, 260, 59.6, 87),
    }
    total, PRO, STEP_SASS, EPI = SASS[a.variant]
    instr_per_pixel = PRO + STEP_SASS * mean_steps + EPI

    print("== mine.cu render, CPU reference (no GPU, no timing) ==")
    print(f"  variant {a.variant}  -> {out_dtype} obs, {total} SASS instructions "
          f"({PRO} prologue + {STEP_SASS:.1f}/DDA step + {EPI} epilogue)")
    print(f"  world {a.world}^3 = {a.world**3} voxels, {solid} solid "
          f"({100*solid/a.world**3:.1f}%), {a.world**3 - solid} air")
    print(f"  eye   ({eye[0]:.2f}, {eye[1]:.2f}, {eye[2]:.2f})   "
          f"yaw={a.yaw} pitch={a.pitch} fov={a.fov} max_steps={a.max_steps} "
          f"view={a.view_distance}")
    print(f"  image {a.res}x{a.res}, {100*solid_frac:.1f}% sky pixels, "
          f"{len(np.unique(img.reshape(-1, 3), axis=0))} distinct colours")
    print(f"  DDA steps: mean {mean_steps:.3f}  min {steps.min()}  max {steps.max()}  "
          f"p50 {np.percentile(steps, 50):.0f}  p90 {np.percentile(steps, 90):.0f}")
    hit_cap = int((steps >= a.max_steps).sum())
    print(f"  pixels that ran the full {a.max_steps}-step budget: "
          f"{hit_cap}/{a.res*a.res} ({100*hit_cap/(a.res*a.res):.1f}%)")

    print()
    print("== instruction budget, per step, extrapolated to batch=%d ==" % B)
    print(f"  SASS per pixel      {PRO:.0f} + {STEP_SASS:.1f}*{mean_steps:.2f} + {EPI} "
          f"= {instr_per_pixel:.0f}")
    lanes = pixels / 32.0  # warps
    warp_instr = lanes * instr_per_pixel
    # GA102: 82 SM x 4 schedulers x 1 warp-instr/cycle, 1.70 GHz
    # T4:     40 SM x 4 schedulers x 1 warp-instr/cycle, 1.41 GHz
    for name, sms, ghz, bw in (("RTX 3090 (sm_86, datasheet, NOT measured)", 82, 1.70, 936.2e9),
                               ("Tesla T4  (sm_75, datasheet, NOT measured)", 40, 1.41, 320.0e9)):
        issue = sms * 4 * ghz * 1e9          # warp-instructions per second
        t_issue = warp_instr / issue
        print(f"  {name}")
        print(f"    issue ceiling    {issue/1e9:8.1f} G warp-instr/s")
        print(f"    time, issue bound{warp_instr:12.3e} / {issue/1e9:.0f}G = {t_issue*1e6:8.1f} us/step"
              f"  -> {B/t_issue/1e6:6.1f} M env-steps/s")
        for lbl, bytes_moved in (("obs uint8 write only", pixels * 3),
                                 ("obs uint8 + voxel compulsory", pixels * 3 + B * a.world**3),
                                 ("obs fp32 write only", pixels * 12),
                                 ("obs fp32 + voxel compulsory", pixels * 12 + B * a.world**3)):
            t_bw = bytes_moved / bw
            print(f"    time, bw bound   {lbl:32s} {t_bw*1e6:8.1f} us/step"
                  f"  -> {B/t_bw/1e6:6.1f} M env-steps/s")

    print()
    print("== the 49.6M claim, checked against the instruction-issue bound ==")
    claim_us = B / 49.63e6 * 1e6
    print(f"  49.63 M env-steps/s at batch {B} is {claim_us:.1f} us/step")
    issue3090 = 82 * 4 * 1.70e9
    budget = claim_us * 1e-6 * issue3090 / lanes
    print(f"  that buys {budget:.1f} SASS instructions per pixel on an RTX 3090")
    print(f"  this kernel needs {PRO:.0f} + {EPI} = {PRO + EPI} before the first DDA step,")
    print(f"  and {STEP_SASS:.1f} per DDA step")
    slack = (budget - PRO - EPI) / STEP_SASS
    print(f"  so the claim allows a mean of {slack:.2f} DDA steps per pixel")
    print(f"  the reference trace measures {mean_steps:.2f}")
    verdict = "ARITHMETICALLY REACHABLE" if slack >= mean_steps else "NOT REACHABLE"
    print(f"  VERDICT: {verdict}  (needs mean <= {slack:.2f}, measured {mean_steps:.2f})")
    print("  This is a bound, not a measurement. It uses the datasheet issue rate")
    print("  (82 SM x 4 schedulers x 1 warp-instr/cycle at 1.70 GHz), which no run")
    print("  on this host can confirm, and it assumes the whole SM issues DDA SASS")
    print("  every cycle with no stalls. A real kernel is slower, so the bound is")
    print("  generous to the claim and the verdict still bites. Only a measured")
    print("  RTX 3090 number settles it either way.")

    if a.out:
        img_hash = hashlib.sha256(img.tobytes()).hexdigest()
        rec = dict(
            world=a.world, res=a.res, max_steps=a.max_steps, ground=a.ground,
            yaw=a.yaw, pitch=a.pitch, fov=a.fov,
            solid_voxels=solid, air_voxels=int(a.world**3 - solid),
            sky_pixel_fraction=solid_frac,
            dda_steps_mean=mean_steps, dda_steps_min=int(steps.min()),
            dda_steps_max=int(steps.max()),
            pixels_at_step_cap=hit_cap,
            image_sha256=img_hash,
            step_histogram={str(int(k)): int(v) for k, v in zip(*np.unique(steps, return_counts=True))},
            sass_per_pixel=instr_per_pixel,
            claim_budget_instr_per_pixel=budget,
            claim_budget_mean_steps=slack,
        )
        with open(a.out, "w") as fh:
            json.dump(rec, fh, indent=2, sort_keys=True)
        print(f"\nwrote {a.out}  image sha256 {img_hash[:16]}...")


if __name__ == "__main__":
    main()
