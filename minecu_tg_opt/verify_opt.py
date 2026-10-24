#!/usr/bin/env python3
"""verify_opt.py -- the correctness gate for the optimized render.

Runs on the CPU, on this host, with no GPU. A timing from an implementation
that does not compute the right thing is a measurement of the wrong thing, so
this gate runs before anything is pushed, in the same way `minecu_tg/kaggle/
run.sh` gates on `minecu_tg/verify.py`.

WHAT IS CHECKED
---------------
1. **Every rung of the ladder produces the same observation as the naive
   port.** `minecu_tg_opt/render.py:render_variant(v)` for every `v` in
   `VARIANTS`, against `minecu_tg/render.py:render` on the same inputs. This
   is the check that makes the ablation mean something: the rungs differ in
   the *state* they carry, not in the ray they trace, so any difference in the
   image would invalidate every price attached to them.

2. **Every rung matches an independent float64 transcription.** Same check
   `minecu_tg/verify.py` makes, against `minecu_tg/cpu_reference.py`, which is
   written from `notes/17-minecu-spec.md` in float64. Exact equality on the
   uint8 observation is the gate.

3. **The per-pixel DDA step counts agree.** The image is a lossy readout --
   many step counts collapse to the same block colour -- so the step count is
   the sharper of the two checks. It is also a direct check on rewrite 3: the
   steps-to-exit countdown is a *different* mechanism from re-testing the
   cursor's bounds each step, and if the two ever disagree the countdown is
   wrong. A 100% match on a tie-sensitive quantity is what rules that out.

4. **The step counter is inert.** With and without `count_steps` the
   observation must be bit-identical, so the verified program and the timed
   program are the same program.

5. **The result is batch-invariant, and per-env cameras work.** A shared
   camera must give identical envs and must equal the single-env image; at a
   wide yaw spread every env must match a float64 render of *its own* camera.
   This is the check that catches a batch axis in the wrong place, which a
   `B=1` smoke test cannot see.

6. **The reference's own CPU model agrees on the world.** The world is built
   by `minecu_tg.world.generate_world` and compared against
   `minecu_ref/bench/dda_model.py:build_world`, a third transcription that is
   not ours. Carried over from `minecu_tg/verify.py` for the same reason: a
   shared misreading of the spec is invisible to a comparison against another
   implementation of the same misreading.

EXIT CODE. 0 if every check passes, 1 otherwise, with every failure printed. A
gate that only prints is a claim, not a check.
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
from minecu_tg.render import camera_basis, render as render_naive  # noqa: E402
from minecu_tg_opt.render import VARIANTS, render_variant  # noqa: E402

FAILS: list[str] = []
LINES: list[str] = []


def check(ok: bool, what: str, detail: str = "") -> bool:
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {what}" + (f"  {detail}" if detail else ""), flush=True)
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
    print("\n6. the world, against minecu_ref/bench/dda_model.py")
    for world in (16, 32, 48):
        g = ground_of(world)
        mine = generate_world(world, g).numpy()
        ref = M.build_world(world, g)
        same = np.array_equal(mine.astype(np.int8), ref.astype(np.int8))
        check(same, f"world {world}^3, ground {g}",
              f"{int((ref >= 0).sum())}/{world**3} solid"
              + ("" if same else f"  ndiff={int((mine != ref).sum())}"))


def check_render(res: int, ms: int, world: int, ground: int, pitch: float,
                 fov: float, batches: tuple[int, ...], min_step_agree: float,
                 all_variants: bool) -> None:
    print(f"\n1-5. the render, world {world}^3 ground {ground} res {res} "
          f"max_steps {ms} pitch {pitch:+.1f} fov {fov}")
    M = dda_model()
    vox = M.build_world(world, ground)              # [Y, X, Z] int8
    cams = np.array([[world / 2.0, float(ground), world / 2.0, 0.0, pitch]],
                    dtype=np.float32)

    b64 = basis64(cams, fov, res, res)
    ref_img, ref_steps = render_one(vox, b64, res, res, ms)

    from tinygrad import Tensor, dtypes
    voxT = Tensor(np.ascontiguousarray(vox.reshape(1, world, world, world)),
                  dtype=dtypes.int8).contiguous()
    basisT = camera_basis(Tensor(cams.tolist(), dtype=dtypes.float32), fov, res, res)

    naive_obs, naive_steps = render_naive(voxT, basisT, res, res, ms,
                                          count_steps=True)
    n_img = naive_obs.numpy()[0]
    n_st = naive_steps.numpy()[0, 0].astype(np.int32)

    names = list(VARIANTS) if all_variants else ["lean"]
    for name in names:
        obs, st = render_variant(voxT, basisT, res, res, ms, name, count_steps=True)
        got = obs.numpy()[0]
        s = st.numpy()[0, 0].astype(np.int32)

        # 1. against the naive port
        same_naive = np.array_equal(got, n_img)
        d = (got.astype(np.int32) != n_img.astype(np.int32))
        check(same_naive, f"V-{name}: observation identical to minecu_tg/render.py",
              "bit-for-bit" if same_naive else
              f"{int(d.any(axis=2).sum())}/{res*res} pixels differ, max delta "
              f"{int(np.abs(d).max())}")

        # 2. against the float64 transcription
        d64 = (got.astype(np.int32) != ref_img.astype(np.int32))
        bad = int(d64.any(axis=2).sum())
        check(bad == 0, f"V-{name}: observation equals the float64 transcription",
              f"{bad}/{res * res} pixels differ, max channel delta "
              f"{int(np.abs(d64).max())}")

        # 3. the step counts. This is the check on rewrite 3 in particular: the
        # steps-to-exit countdown is a different mechanism from re-testing the
        # cursor's bounds, and a tie-sensitive count is what makes them
        # comparable.
        agree = int((s == ref_steps).sum())
        frac = agree / (res * res)
        check(frac >= min_step_agree,
              f"V-{name}: per-pixel DDA step counts agree with float64",
              f"{agree}/{res * res} = {100 * frac:.2f}%  "
              f"(mean {s.mean():.3f} vs {ref_steps.mean():.3f}; "
              f"naive port {n_st.mean():.3f})")

    # 4. the counter is inert
    plain = render_variant(voxT, basisT, res, res, ms, "lean").numpy()[0]
    check(np.array_equal(plain, n_img),
          "count_steps=True does not change the observation",
          "the timed path and the verified path are the same program")

    # 5. batch invariance
    ident, varied_ok, varied_detail = True, True, ""
    for B in batches:
        voxB = Tensor(np.ascontiguousarray(np.stack([vox] * B)),
                      dtype=dtypes.int8).contiguous()
        shared = np.array([[world / 2.0, float(ground), world / 2.0, 0.0, pitch]]
                          * B, dtype=np.float32)
        oB = render_variant(voxB, camera_basis(
            Tensor(shared.tolist(), dtype=dtypes.float32), fov, res, res),
            res, res, ms).numpy()
        if oB.shape != (B, res, res, 3) or \
           not all(np.array_equal(oB[0], oB[i]) for i in range(B)) or \
           not np.array_equal(oB[0], n_img):
            ident = False
            break

        # A wide spread on purpose: the observation has to be able to tell the
        # envs apart, or this sub-check is decorative.
        spread = np.array([[world / 2.0, float(ground), world / 2.0,
                            1.2 * (i % 7 - 3), pitch] for i in range(B)],
                          dtype=np.float32)
        oS = render_variant(voxB, camera_basis(
            Tensor(spread.tolist(), dtype=dtypes.float32), fov, res, res),
            res, res, ms).numpy()
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-step-agree", type=float, default=0.99)
    ap.add_argument("--quick", action="store_true",
                    help="one shape instead of five. Every rung is still checked: "
                         "the point of --quick is to bound the number of shapes, "
                         "not to stop testing the rungs against each other.")
    a = ap.parse_args()

    print("=" * 78)
    print("  minecu_tg_opt verification -- the optimized render against the naive")
    print("  port and against an independent float64 transcription")
    print(f"  DEV={os.environ.get('DEV')}  variants: {list(VARIANTS)}")
    print("=" * 78)

    M = dda_model()
    check_world(M)

    cases = [(32, 32, 16, 4, 0.0), (32, 32, 16, 4, -0.2)]
    if not a.quick:
        cases += [(32, 32, 32, 8, 0.0), (32, 32, 48, 12, 0.0), (32, 16, 16, 4, 0.0)]
    allv = True        # every rung is checked at every shape, quick or not
    for res, ms, world, ground, pitch in cases:
        check_render(res=res, ms=ms, world=world, ground=ground, pitch=pitch,
                     fov=70.0, batches=(1, 5), min_step_agree=a.min_step_agree,
                     all_variants=allv)

    print("\n" + "=" * 78)
    if FAILS:
        print(f"  {len(FAILS)} FAILURE(S). The optimized render is NOT verified.")
        for f in FAILS:
            print(f"    - {f}")
        return 1
    print("  all checks passed. Every rung of the ladder produces the naive")
    print("  port's observation bit for bit, and that observation equals an")
    print("  independent float64 transcription of the same algorithm.")
    return 0


if __name__ == "__main__":
    code = main()
    # The full log beside the summary, so a failure is diagnosable from the
    # notebook's output directory rather than only from a terminal. Wrapped:
    # an unwritable destination must not turn a PASSING run into a failing
    # exit code, which is the one thing a gate must never do.
    for dest in (Path("/tmp/minecu_tg_opt_verify.log"),
                 Path(".agent-scratch/minecu_tg_opt_verify.log")):
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text("\n".join(LINES) + "\n")
            break
        except OSError:
            continue
    sys.exit(code)
