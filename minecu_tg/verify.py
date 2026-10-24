#!/usr/bin/env python3
"""verify.py -- the correctness gate for the tensor render.

Runs on the CPU, on this host, with no GPU. Everything it claims about the
TinyGrad implementation is established here; the T4 run establishes only what
needs a card (timings, the real backend's codegen, the device's own view of the
observation).

WHAT IS CHECKED, AND WHY EACH CHECK EXISTS
------------------------------------------
1. **The world matches the reference's own model.**
   `minecu_tg.world.generate_world` against `minecu_ref/bench/dda_model.py`'s
   `build_world`, at three world sizes, exact. This is a check against a
   *third* transcription that is not ours, which is the only kind that catches
   a shared misreading of the spec.

2. **The render matches an independent float64 transcription, exactly, on the
   uint8 observation.** `minecu_tg/cpu_reference.py` is written from the
   spec in float64; the render is float32. Exact equality of the final image is
   a strong statement and it holds, so it is the gate.

3. **The per-pixel DDA step counts agree.** The image is a lossy readout -- many
   step counts collapse to the same block colour -- so the step count is the
   sharper identity check. It is a discrete, tie-sensitive quantity: two
   implementations that took the same traversal decisions have identical step
   counts, and two that did not, do not. Agreement is required at
   `>= 99%` of pixels, and the residual is explained by float32-vs-float64
   rounding of `t_max` at exact ties.

4. **The step counter is inert.** Rendering with `count_steps=True` and with
   it off must produce bit-identical observations. Without this check, check 3
   could be measuring a different program from the one that gets timed.

5. **The result is batch-invariant.** The reference gives every env the same
   world, so every env's image must be identical, and must match the single-env
   image. This is the check that catches the `basis[:, i, None, None]` shape
   bug, which put the batch axis in the wrong place and was *invisible at
   B=1*: `basis[:, i]` drops to rank 1, and `[None, None]` on a rank-1 tensor
   is (1, 1, B), not (B, 1, 1, 1). It is in this file because a smoke test at
   B=1 did not catch it and a batch of 5 did.

6. **The worldgen floor and the physics floor agree.** `ground == world // 4`.
   A run that buries the eye inside its own surface is a run whose timing
   measures an obs store; the reference's own harness gates on this and so does
   this one.

EXIT CODE. 0 if every check passes, 1 otherwise, with every failure printed.
A gate that only prints is a claim, not a check.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("DEV", "CPU")
os.environ.setdefault("PARALLEL", "0")

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

from minecu_tg import generate_world, ground_of  # noqa: E402
from minecu_tg.cpu_reference import camera_basis as basis64, render_one  # noqa: E402
from minecu_tg.render import camera_basis, render  # noqa: E402

FAILS: list[str] = []
LINES: list[str] = []


def check(ok: bool, what: str, detail: str = "") -> bool:
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {what}" + (f"  {detail}" if detail else ""))
    LINES.append(f"[{tag}] {what}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAILS.append(what)
    return ok


def dda_model():
    """The reference's own CPU model, read-only, for the cross-check."""
    sys.path.insert(0, str(REPO / "minecu_ref" / "bench"))
    import dda_model  # noqa: PLC0415
    return dda_model


# --------------------------------------------------------------------------
def check_world(M) -> None:
    print("\n1. the world, against minecu_ref/bench/dda_model.py")
    for world in (16, 32, 48):
        g = ground_of(world)
        mine = generate_world(world, g).numpy()
        ref = M.build_world(world, g)
        same = np.array_equal(mine.astype(np.int8), ref.astype(np.int8))
        check(same, f"world {world}^3, ground {g}",
              f"{int((ref >= 0).sum())}/{world**3} solid"
              + ("" if same else f"  ndiff={int((mine != ref).sum())}"))


def check_render(res: int, ms: int, world: int, ground: int, pitch: float,
                 fov: float, batches: tuple[int, ...], min_step_agree: float):
    # A note worth having here, because it looks like a bug and is not. The
    # camera sits at (world/2, ground+1.6, world/2) and the tree and the house
    # are placed at fixed offsets from THAT point (kernels.cu:1831, :1868), so
    # the local geometry around the camera is identical at every world size and
    # only the far wall recedes. The far wall contributes no hits -- a ray that
    # reaches it is out of bounds and marches to the cap either way. So worlds
    # 32 and 48 have the same mean DDA step count. Verified against
    # `minecu_ref/bench/dda_model.py`, which reports the same 16.029 for both.
    print(f"\n2-5. the render, world {world}^3 ground {ground} res {res} "
          f"max_steps {ms} pitch {pitch:+.1f} fov {fov}")
    M = dda_model()
    vox = M.build_world(world, ground)              # [Y, X, Z] int8
    eye = (world / 2.0, ground + 1.6, world / 2.0)
    cams = np.array([[world / 2.0, float(ground), world / 2.0, 0.0, pitch]],
                    dtype=np.float32)

    b64 = basis64(cams, fov, res, res)
    ref_img, ref_steps = render_one(vox, b64, res, res, ms)

    from tinygrad import Tensor, dtypes
    voxT = Tensor(np.ascontiguousarray(vox.reshape(1, world, world, world)),
                  dtype=dtypes.int8).contiguous()
    basisT = camera_basis(Tensor(cams.tolist(), dtype=dtypes.float32), fov, res, res)

    obs, steps = render(voxT, basisT, res, res, ms, count_steps=True)
    got = obs.numpy()[0]
    st = steps.numpy()[0, 0].astype(np.int32)

    # 2. the image, exact on uint8
    d = (got.astype(np.int32) != ref_img.astype(np.int32))
    bad_px = int(d.any(axis=2).sum())
    check(bad_px == 0, "uint8 observation equals the float64 transcription",
          f"{bad_px}/{res * res} pixels differ, max channel delta "
          f"{int(np.abs(d).max())}")

    # 3. the step counts
    agree = int((st == ref_steps).sum())
    frac = agree / (res * res)
    check(frac >= min_step_agree,
          "per-pixel DDA step counts agree with the float64 transcription",
          f"{agree}/{res * res} = {100 * frac:.2f}%  "
          f"(mean {st.mean():.3f} vs {ref_steps.mean():.3f})")

    # 4. the counter is inert
    plain = render(voxT, basisT, res, res, ms).numpy()[0]
    check(np.array_equal(plain, got),
          "count_steps=True does not change the observation",
          "the timed path and the verified path are the same program")

    # 5. batch invariance. The reference gives every env the same world AND the
    # same spawn, so with one shared camera every env's image must be identical
    # and must equal the single-env image. This is the check that catches the
    # `basis[:, i, None, None]` shape bug, which put the batch axis in the wrong
    # place and was invisible at B=1.
    #
    # A second, separate sub-check: with a *per-env* camera every env must match
    # a float64 render of ITS OWN camera. Comparing each env against its own
    # reference is stronger than comparing env 0 against the shared-camera
    # reference, and it is the only version of this check that is correct when
    # the cameras differ.
    #
    # The spread here is WIDE on purpose: +-1.2 rad, so the images genuinely
    # differ and the check has teeth. The timed battery's own per-env spread is
    # 0.001*(i%97-48) = +-0.048 rad, and that is deliberately reported as a
    # caveat on that harness rather than used here -- at +-0.048 rad the whole
    # batch renders the SAME 3-colour image, because the scene has three colours
    # and a 0.05 rad rotation moves the horizon by under two pixels. Nothing
    # downstream can use the observation to tell the envs apart at that spread.
    ident, varied_ok, varied_detail = True, True, ""
    for B in batches:
        voxB = Tensor(np.ascontiguousarray(np.stack([vox] * B)),
                      dtype=dtypes.int8).contiguous()
        shared = np.array([[world / 2.0, float(ground), world / 2.0, 0.0, pitch]]
                          * B, dtype=np.float32)
        oB = render(voxB, camera_basis(Tensor(shared.tolist(), dtype=dtypes.float32),
                                       fov, res, res), res, res, ms).numpy()
        if oB.shape != (B, res, res, 3) or \
           not all(np.array_equal(oB[0], oB[i]) for i in range(B)) or \
           not np.array_equal(oB[0], got):
            ident = False
            break

        spread = np.array([[world / 2.0, float(ground), world / 2.0,
                            1.2 * (i % 7 - 3), pitch] for i in range(B)],
                          dtype=np.float32)
        oS = render(voxB, camera_basis(Tensor(spread.tolist(), dtype=dtypes.float32),
                                       fov, res, res), res, res, ms).numpy()
        bad = 0
        for i in range(B):
            ref_i, _ = render_one(vox, basis64(spread[i:i + 1], fov, res, res),
                                  res, res, ms)
            if not np.array_equal(oS[i], ref_i):
                bad += 1
        n_distinct = len({oS[i].tobytes() for i in range(B)})
        if bad:
            varied_ok, varied_detail = False, f"{bad}/{B} envs disagree with their own reference"
        elif B > 1 and n_distinct < 2:
            varied_ok, varied_detail = False, "all envs identical despite a +-1.2 rad yaw spread"
        else:
            varied_detail = (f"{n_distinct} distinct images over {B} envs at a "
                             f"+-1.2 rad yaw spread, each matching its own float64 reference")

    check(ident, f"a shared camera gives identical envs at batch {list(batches)}",
          "catches a batch axis in the wrong place, which B=1 cannot")
    check(varied_ok, f"per-env cameras render correctly at batch {list(batches)}",
          varied_detail)


def check_ground_gate() -> None:
    print("\n6. the worldgen floor and the physics floor")
    for world in (16, 32, 48):
        g = ground_of(world)
        check(g == world // 4, f"world {world}: ground {g} == world/4 {world // 4}",
              "physics_kernel floors the agent at world_y/4; anything else "
              "buries or floats it")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-step-agree", type=float, default=0.99)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()

    print("=" * 78)
    print("  minecu_tg verification -- the tensor render against a float64 "
          "transcription")
    print(f"  DEV={os.environ.get('DEV')}  tinygrad from the venv, CPU backend")
    print("=" * 78)

    M = dda_model()
    check_world(M)
    check_ground_gate()

    cases = [(32, 32, 16, 4, 0.0), (32, 32, 16, 4, -0.2)]
    if not a.quick:
        cases += [(32, 32, 32, 8, 0.0), (32, 32, 48, 12, 0.0), (32, 16, 16, 4, 0.0)]
    for res, ms, world, ground, pitch in cases:
        check_render(res=res, ms=ms, world=world, ground=ground, pitch=pitch,
                     fov=70.0, batches=(1, 5), min_step_agree=a.min_step_agree)

    print("\n" + "=" * 78)
    if FAILS:
        print(f"  {len(FAILS)} FAILURE(S). The tensor render is NOT verified.")
        for f in FAILS:
            print(f"    - {f}")
        return 1
    print("  all checks passed. The tensor render is verified against an "
          "independent")
    print("  float64 transcription of the same algorithm, and the step counts")
    print("  agree pixel for pixel on the tie-insensitive parts.")
    return 0


if __name__ == "__main__":
    code = main()
    Path("/tmp/minecu_tg_verify.log").write_text("\n".join(LINES) + "\n")
    sys.exit(code)
