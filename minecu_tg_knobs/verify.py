#!/usr/bin/env python3
"""verify.py -- is the lean port the same computation as the naive one?

    DEV=CPU python -m minecu_tg_knobs.verify

WHY THIS GATE IS BEFORE ANY MEASUREMENT
---------------------------------------
A knob sweep over a port that computes the wrong thing measures the wrong
thing, faster. And the specific trap here is that the three port fixes in
`lean.py` are *semantic* changes -- an integral cursor, an advanced address, a
folded bounds test -- so each one is a chance to write a different algorithm
that still looks right. `minecu_tg_opt/verify_opt.py`'s log is the standing
reminder: its "V-lean" checks FAIL on step counts, and an arm built on that
would have reported a fast kernel that traverses differently.

The contract enforced here is deliberately the *discrete* one:

  * the uint8 observation equals an independent float64 transcription, and
  * the per-pixel DDA step count equals it too.

The step count is the sharper of the two. The image is lossy -- many step
counts collapse to the same block colour -- so an image match alone would pass
a traversal that stops one step early on every ray.

WHAT IS COMPARED AGAINST WHAT
-----------------------------
    float64   `minecu_tg/cpu_reference.py`, a transcription written from
              `notes/17-minecu-spec.md` and the reference's source. It is
              imported, never modified, and it is *not* this lane's file: a
              bug shared between the port and its checker is the failure mode
              a same-repo checker cannot see. `minecu_tg/world.py` is checked
              against `minecu_ref/bench/dda_model.py`, a third transcription
              nobody in this lane wrote, which is what stops that failure.

  naive      `minecu_tg/render.py` -- arm A's port, imported unchanged. The
              lean port must agree with it as well as with float64, because
              "lean" must not quietly mean "different".

A PASS here is a correctness gate and **not** performance evidence. Nothing in
this file says anything about speed.
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("DEV", "CPU")
os.environ.setdefault("PARALLEL", "0")

from pathlib import Path  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

from minecu_tg.blocks import colour_lut  # noqa: E402
from minecu_tg import cpu_reference  # noqa: E402
from minecu_tg.render import render as render_naive  # noqa: E402
from minecu_tg.world import generate_world  # noqa: E402
from minecu_tg_knobs.lean import render as render_lean  # noqa: E402

# The three port fixes, as a named matrix. Each is a semantic change, so each
# variant is verified separately: a variant that disagrees is a bug in that
# variant, and finding it is the point of running all of them.
VARIANTS = {
    "lean":            dict(cursor_int=True,  flat_advance=True,  unsigned_bounds=True),
    "-unsigned":       dict(cursor_int=True,  flat_advance=True,  unsigned_bounds=False),
    "-flat":           dict(cursor_int=True,  flat_advance=False, unsigned_bounds=True),
    "-cursor":         dict(cursor_int=False, flat_advance=False, unsigned_bounds=True),
    "naive-equivalent": dict(cursor_int=False, flat_advance=False, unsigned_bounds=False),
}

# (world, ground, pitch, max_steps) -- the same cases minecu_tg/verify.py uses,
# so the two ports are held to the same bar rather than to a private one.
CASES = list(dict.fromkeys([
    (16, 4, 0.0, 32),
    (16, 4, -0.2, 32),
    (32, 8, 0.0, 32),
    (48, 12, 0.0, 32),
    (16, 4, 0.0, 16),
]))

FAILED: list[str] = []


def check(ok: bool, label: str, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}"
          + (f"  {detail}" if detail else ""))
    if not ok:
        FAILED.append(label)


def section(title: str) -> None:
    print()
    print(title)


def main() -> int:
    print("=" * 78)
    print("  minecu_tg_knobs -- the lean port, against a float64 transcription")
    print(f"  DEV={os.environ.get('DEV', 'CPU')}  tinygrad from the venv")
    print("=" * 78)

    lut = colour_lut().numpy()

    # ---- 1. the world, against a transcription this lane did not write
    section("1. the world, against minecu_ref/bench/dda_model.py")
    import minecu_ref.bench.dda_model as dda_model
    for w, g, _, _ in CASES:
        world = generate_world(w, g).numpy()
        model = np.asarray(dda_model.build_world(w, g))
        solid = int((world >= 0).sum())
        same = np.array_equal(world, model)
        check(same, f"world {w}^3, ground {g}",
              f"{solid}/{world.size} solid, identical to dda_model: {same}")

    # ---- 2-5. the render, every variant, every case
    for w, g, pitch, max_steps in CASES:
        world = generate_world(w, g).numpy()
        wy, wx, wz = world.shape
        res = 32
        fov = 70.0

        # the same camera minecu_tg/verify.py uses: eye in the world, a yaw
        # that makes ray_x zero for one column, and a pitch that does not.
        cam = np.array([[w / 2.0 + 0.5, wy / 4 + 0.5, w / 2.0 + 0.5,
                         0.37 + pitch, 0.11]], dtype=np.float64)
        # `camera_basis` flattens: it returns the 14 numbers for the one camera
        # it is given, not a [B, 14] batch.
        basis64 = cpu_reference.camera_basis(cam, fov, res, res)
        img64, st64 = cpu_reference.render_one(world, basis64, res, res, max_steps)

        # a zero ray component is a real case, not a corner: at yaw 0, fwd_x is
        # 0 and a whole pixel column has ray_x == 0, which is the `1e30` guard.
        section(f"2-5. render, world {w}^3 ground {g} res {res} max_steps {max_steps} "
                f"pitch {pitch:+.1f} fov {fov}")
        print(f"  float64 reference: mean steps {st64.mean():.3f}, "
              f"distinct colours {len(np.unique(img64.reshape(-1, 3), axis=0))}")

        from tinygrad import Tensor, dtypes
        vox_t = Tensor(world.astype(np.int8)).unsqueeze(0)
        basis_t = Tensor(basis64.astype(np.float32)[None, :])
        # Arm A's port, same inputs. Both sides are [B, ...] and float64 is a
        # single env, so the comparison is against element 0 -- which is the
        # whole batch when B==1, and is itself the B=1 coverage that
        # notes/19 §1.3 says a B=1 smoke test cannot give. `minecu_tg/verify.py`
        # owns the B>1 per-env-camera case; this gate is the float64 contract
        # and the arm-A cross-check, and both are stated as what they are.
        # The two outputs have DIFFERENT ranks, and getting this wrong is
        # silent: `obs` is [B,H,W,3] and `steps` is [B,1,H,W] (render.py:222,
        # :233), so the same index that drops the batch axis on one leaves a
        # leading H on the other. Indexing both `[0,0]` gives `(32,3)` against
        # a `(32,32,3)` reference, which broadcasts, compares garbage, and
        # reports ~500 of 1024 pixels differing while the step counts agree at
        # 1024/1024. Found by exactly that disagreement.
        obs_a, st_a = render_naive(vox_t, basis_t, res, res, max_steps, count_steps=True)
        oa, sa = obs_a.numpy()[0], st_a.numpy()[0, 0]
        if oa.shape != img64.shape or sa.shape != st64.shape:
            check(False, "naive port: shapes",
                  f"obs {oa.shape} vs {img64.shape}, steps {sa.shape} vs "
                  f"{st64.shape}")
        else:
            check(np.array_equal(oa, img64), "naive port equals the float64 image",
                  f"{int((oa != img64).any(axis=-1).sum())}/{res*res} pixels differ")
            check(np.array_equal(sa, st64), "naive port step counts equal float64",
                  f"{int((sa == st64).sum())}/{res*res} agree, mean {sa.mean():.3f}")

        for name, kw in VARIANTS.items():
            try:
                obs, st = render_lean(vox_t, basis_t, res, res, max_steps,
                                      count_steps=True, **kw)
            except Exception as exc:  # noqa: BLE001
                check(False, f"variant {name}", f"raised {type(exc).__name__}: {exc}")
                continue
            o, s = obs.numpy()[0], st.numpy()[0, 0]
            # shape guard FIRST. A broadcasting comparison is the failure mode
            # this gate has to be immune to: `(32,3)` against `(32,32,3)` does
            # not raise, it reports a confident wrong number.
            if o.shape != img64.shape or s.shape != st64.shape:
                check(False, f"variant {name}: shapes",
                      f"obs {o.shape} vs {img64.shape}, steps {s.shape} vs "
                      f"{st64.shape}. A broadcasting comparison here would "
                      f"report a confident wrong answer, so it is refused.")
                continue
            dpx = int((o != img64).any(axis=-1).sum())
            agree = int((s == st64).sum())
            dmax = int(np.abs(o.astype(np.int32) - img64.astype(np.int32)).max())
            check(dpx == 0 and dmax == 0 and agree == res * res,
                  f"variant {name}: image and step counts vs float64",
                  f"{dpx}/{res*res} pixels differ, {agree}/{res*res} steps agree, "
                  f"max channel delta {dmax}")

    # ---- 6. the step counter is inert
    section("6. instrumentation is inert")
    for name, kw in VARIANTS.items():
        vox_t = Tensor(world.astype(np.int8)).unsqueeze(0)
        basis_t = Tensor(basis64.astype(np.float32))[None, :]
        withc = render_lean(vox_t, basis_t, res, res, max_steps, **kw).numpy()[0]
        without = render_lean(vox_t, basis_t, res, res, max_steps,
                               count_steps=True, **kw)[0].numpy()[0]
        check(np.array_equal(withc, without),
              f"variant {name}: count_steps=True does not change the observation",
              "the timed path and the verified path are the same program")

    print()
    print("=" * 78)
    if FAILED:
        print(f"  {len(FAILED)} FAILED:")
        for f in FAILED:
            print(f"    - {f}")
        print("  A knob sweep on a port that fails here measures a different")
        print("  algorithm. Stopping.")
        return 1
    print("  all checks passed. This is a CORRECTNESS gate and not performance")
    print("  evidence: nothing above has been timed and nothing above is a SASS")
    print("  count. The measurements are census.py (codegen) and battery.py (T4).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
