#!/usr/bin/env python3
"""verify.py -- the correctness gate. This host, or a card with --dev.

Runs before anything is pushed and before any timing is quoted. A timing from
an implementation that does not compute the right thing is a measurement of
the wrong thing.

WHAT IS CHECKED
---------------
1. **Both modes, and both edit points, produce the same observation as the
   naive port.** `minecu_tg_faithful.render` in `clean` and in `faithful`,
   against `minecu_tg/render.py` on the same inputs. Bit-identical, not
   close.

2. **Both modes equal an independent float64 transcription.**
   `minecu_tg/cpu_reference.py`, written from `notes/17-minecu-spec.md` in
   float64. Exact equality on the uint8 observation.

3. **The per-pixel DDA step counts agree with float64.** The image is a lossy
   readout -- many step counts collapse to the same block colour -- so the step
   count is the sharper of the two checks, and it is tie-sensitive in exactly
   the way that would expose a difference in the traversal.

4. **The two modes agree with EACH OTHER on the step count**, which is the
   check the whole arm rests on: the crossings must not change the
   computation. `addr_bits=64` in particular is a fidelity switch on the
   *index width*; if it moved a traversal decision the comparison would be
   measuring two different computations.

5. **The step counter is inert.** With and without `count_steps` the
   observation is bit-identical, so the verified program and the timed program
   are the same program.

6. **Batch invariance and per-env cameras.** A shared camera gives identical
   envs; at a wide yaw spread every env matches a float64 render of *its own*
   camera. This is the check that catches a batch axis in the wrong place,
   which a `B=1` smoke test cannot see.

7. **The world, against the reference's own model.** `minecu_tg.world`
   compared against `minecu_ref/bench/dda_model.py:build_world` -- a third
   transcription, not ours, so it can catch a shared misreading of the spec.

EXIT CODE. 0 if every check passes, 1 otherwise, with every failure printed.
A gate that only prints is a claim, not a check.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# ---- the device, resolved BEFORE anything imports tinygrad ----------------
#
# This has to be here, above the first `import tinygrad`, and not in `main()`.
# `tinygrad.helpers` reads `DEV` into a `ContextVar` at *import* time
# (`ContextVar.__init__`: `getenv(key, default_value)`, helpers.py:182), so a
# `Context(DEV=...)` or an argparse default set after the import is already
# too late: the device has been chosen. `verify.py --dev CUDA` printed
# `DEV=CUDA` in its banner and then ran on the CPU backend anyway, and died
# with `FileNotFoundError: 'clang'` on the card -- a correct-looking log line
# over the wrong device, which is the worst kind of wrong.
#
# `--dev` is pre-scanned from argv here, and the environment still wins if the
# caller sets DEV directly. The T4 image has no `clang` and tinygrad's CPU
# backend shells out to one (`ClangRenderer`, renderer/cstyle.py:255), so
# `DEV=CPU` cannot run on a card at all. Verifying on the card's own device is
# not a workaround: it checks the render on the device every timing will be
# taken on, against the same float64 reference.
if "--dev" in sys.argv:
    os.environ["DEV"] = sys.argv[sys.argv.index("--dev") + 1]
os.environ.setdefault("DEV", "CPU")
os.environ.setdefault("PARALLEL", "0")

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

from minecu_tg import generate_world, ground_of  # noqa: E402
from minecu_tg.cpu_reference import camera_basis as basis64, render_one  # noqa: E402
from minecu_tg.render import camera_basis, render as render_naive  # noqa: E402

from minecu_tg_faithful import crossings  # noqa: E402
from minecu_tg_faithful.render import render as render_faithful  # noqa: E402

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


# ---------------------------------------------------------------------------
# the configurations, so the two edit points are each tested both ways
# ---------------------------------------------------------------------------
def variants() -> list[tuple[str, str, dict]]:
    """(label, mode, overrides). Both modes x both addr widths.

    `unsigned_bounds` is also swept. On sm_75 the signed and unsigned folds are
    both 3 instructions, so the sweep is a *claim to check* rather than a
    performance choice -- and if the step counts agree across all four, the
    claim is checked rather than inherited from `notes/19`.
    """
    out = []
    for mode in crossings.MODES:
        for bits in (32, 64):
            for ub in (True, False):
                out.append((f"{mode}/addr{bits}/uint{int(ub)}", mode,
                            {"addr_bits": bits, "unsigned_bounds": ub}))
    # mode-major and complete, which is what `quick_variants` slices by axis and
    # what the coverage assert in `check_render` relies on. Asserted here, where
    # the ordering is built, rather than where it is consumed.
    #
    # `[m for m in MODES for _ in range(4)]`, not `list(MODES) * 4`: the latter
    # repeats the whole list and alternates the modes. Writing the wrong one
    # here is exactly the class of slip this assert exists to stop, so it is
    # worth the two lines to be explicit about which is which.
    assert [m for _, m, _ in out] == [m for m in crossings.MODES for _ in range(4)], \
        "variants() must be mode-major and complete"
    return out


def quick_variants() -> list[tuple[str, str, dict]]:
    """Both modes x both bounds folds, at one address width -- what `--quick` checks.

    NOT `variants()[:2]`. `variants()` is ordered mode-major, so its first two
    entries are `clean/addr32/uint1` and `clean/addr32/uint0`: slicing two of
    eight returns the same mode twice, and the quick gate -- the one `run.sh`
    runs before it will push anything, and the one the card runs -- never
    rendered in `faithful` mode at all. The `--quick` help text said "the two
    modes only" and the code did not do it.

    Selecting by axis instead of by slice is what a subset of a sweep has to be
    keyed on. This keeps the folds (both were covered before, and the fold is an
    edit point) and adds the modes (neither was), so the quick gate is a strict
    superset of what it was: same number of renders, and every mode in it. The
    width axis is the one left out, and the full gate covers both widths.
    """
    out = [(label, mode, kw) for label, mode, kw in variants()
           if kw["addr_bits"] == 32]
    return out


def check_world(M) -> None:
    print("\n7. the world, against minecu_ref/bench/dda_model.py")
    for world in (16, 32, 48):
        g = ground_of(world)
        mine = generate_world(world, g).numpy()
        ref = M.build_world(world, g)
        same = np.array_equal(mine.astype(np.int8), ref.astype(np.int8))
        check(same, f"world {world}^3, ground {g}",
              f"{int((ref >= 0).sum())}/{world ** 3} solid"
              + ("" if same else f"  ndiff={int((mine != ref).sum())}"))


def check_render(res: int, ms: int, world: int, ground: int, pitch: float,
                 fov: float, batches: tuple[int, ...], min_step_agree: float,
                 all_variants: bool) -> None:
    print(f"\n1-6. the render, world {world}^3 ground {ground} res {res} "
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

    vs = variants() if all_variants else quick_variants()
    assert {m for _, m, _ in vs} == set(crossings.MODES), \
        "the gate must render in every mode; a mode missing here is an untested mode"
    for label, mode, kw in vs:
        obs, st = render_faithful(voxT, basisT, res, res, ms,
                                  count_steps=True, mode=mode, **kw)
        got = obs.numpy()[0]
        s = st.numpy()[0, 0].astype(np.int32)

        # 1. against the naive port, bit for bit
        same_naive = np.array_equal(got, n_img)
        d = (got.astype(np.int32) != n_img.astype(np.int32))
        check(same_naive, f"{label}: observation identical to minecu_tg/render.py",
              "bit-for-bit" if same_naive else
              f"{int(d.any(axis=2).sum())}/{res * res} pixels differ, max delta "
              f"{int(np.abs(d).max())}")

        # 2. against the float64 transcription
        d64 = (got.astype(np.int32) != ref_img.astype(np.int32))
        bad = int(d64.any(axis=2).sum())
        check(bad == 0, f"{label}: observation equals the float64 transcription",
              f"{bad}/{res * res} pixels differ, max channel delta "
              f"{int(np.abs(d64).max())}")

        # 3. step counts against float64
        agree = int((s == ref_steps).sum())
        frac = agree / (res * res)
        check(frac >= min_step_agree,
              f"{label}: per-pixel DDA step counts agree with float64",
              f"{agree}/{res * res} = {100 * frac:.2f}%  "
              f"(mean {s.mean():.3f} vs {ref_steps.mean():.3f}; "
              f"naive port {n_st.mean():.3f})")

        # 4. against the naive port's step count, which is the same
        # computation ported a second time
        check(np.array_equal(s, n_st),
              f"{label}: step counts identical to minecu_tg/render.py",
              f"mean {s.mean():.3f} vs {n_st.mean():.3f}; a difference here "
              f"means the crossings changed the traversal")

    # 5. the counter is inert
    plain = render_faithful(voxT, basisT, res, res, ms, mode="faithful").numpy()[0]
    check(np.array_equal(plain, n_img),
          "count_steps=True does not change the observation",
          "the timed path and the verified path are the same program")

    # 6. batch invariance and per-env cameras
    ident, varied_ok, varied_detail = True, True, ""
    for B in batches:
        voxB = Tensor(np.ascontiguousarray(np.stack([vox] * B)),
                      dtype=dtypes.int8).contiguous()
        shared = np.array([[world / 2.0, float(ground), world / 2.0, 0.0, pitch]]
                          * B, dtype=np.float32)
        oB = render_faithful(voxB, camera_basis(
            Tensor(shared.tolist(), dtype=dtypes.float32), fov, res, res),
            res, res, ms, mode="faithful").numpy()
        if oB.shape != (B, res, res, 3) or \
           not all(np.array_equal(oB[0], oB[i]) for i in range(B)) or \
           not np.array_equal(oB[0], n_img):
            ident = False
            break

        spread = np.array([[world / 2.0, float(ground), world / 2.0,
                            1.2 * (i % 7 - 3), pitch] for i in range(B)],
                          dtype=np.float32)
        oS = render_faithful(voxB, camera_basis(
            Tensor(spread.tolist(), dtype=dtypes.float32), fov, res, res),
            res, res, ms, mode="faithful").numpy()
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
                    help="two shapes, both modes x both bounds folds at "
                         "addr_bits=32 (41 checks). The full gate adds the "
                         "other address width and three more world sizes "
                         "(114 checks). Both cover every mode; see "
                         "quick_variants()")
    ap.add_argument("--dev", default=None,
                    help="override DEV. Handled ABOVE the tinygrad import, "
                         "because DEV is read into a ContextVar at import time "
                         "(helpers.py:182) and setting it later changes nothing")
    a = ap.parse_args()

    print("=" * 78)
    print("  minecu_tg_faithful verification -- the faithful port, both modes,")
    print("  against the naive port and an independent float64 transcription")
    print(f"  DEV={os.environ.get('DEV')}  modes={list(crossings.MODES)}  "
          f"edit points={sorted(crossings.EDIT_POINTS)}")
    # The banner is not evidence. `Device.DEFAULT` is what actually ran, and
    # the two can disagree if anything reordered the imports -- which is
    # exactly the bug this gate exists to catch, so it is printed from the
    # framework rather than from the environment.
    from tinygrad import Device  # noqa: PLC0415
    print(f"  Device.DEFAULT={Device.DEFAULT}  (from tinygrad, not os.environ)")
    print("=" * 78)

    M = dda_model()
    check_world(M)

    cases = [(32, 32, 16, 4, 0.0), (32, 32, 16, 4, -0.2)]
    if not a.quick:
        cases += [(32, 32, 32, 8, 0.0), (32, 32, 48, 12, 0.0), (32, 16, 16, 4, 0.0)]
    for i, (res, ms, world, ground, pitch) in enumerate(cases):
        # The full 2 modes x 2 address widths x 2 bounds folds is 8 renders per
        # shape, and each is ~40 s on the CPU backend at 32x32, so the sweep
        # runs on the FIRST shape only. The other shapes check the two modes,
        # which is what catches a shape-dependent regression. Spending 20
        # minutes to re-confirm that a signed and an unsigned compare agree
        # on a different world size is not a better use of the gate.
        check_render(res=res, ms=ms, world=world, ground=ground, pitch=pitch,
                     fov=70.0, batches=(1, 5), min_step_agree=a.min_step_agree,
                     all_variants=(i == 0 and not a.quick))

    print("\n" + "=" * 78)
    if FAILS:
        print(f"  {len(FAILS)} FAILURE(S). The faithful port is NOT verified.")
        for f in FAILS:
            print(f"    - {f}")
        return 1
    # The closing claim names exactly what ran. `verify.py --quick` does not
    # cover both address widths; saying so beats a sentence that is true of the
    # full gate and false of this one.
    covered = sorted({f"{m}/addr{kw['addr_bits']}"
                      for _, m, kw in (variants() if not a.quick else quick_variants())})
    print(f"  all {len(LINES)} checks passed, over {len(covered)} configurations:")
    print(f"    {' '.join(covered)}")
    print("  Every one produces the naive port's observation bit for bit, that")
    print("  observation equals an independent float64 transcription, and the")
    print("  per-pixel DDA step counts are identical across all of them -- so")
    print("  the crossings did not change the computation.")
    if a.quick:
        print("  (`--quick`: both modes and both bounds folds at addr_bits=32. The")
        print("  other address width, and the other world sizes, are the full gate.)")
    return 0


if __name__ == "__main__":
    code = main()
    # written next to the run, not into the repo: on a card the cwd is
    # /kaggle/working and there is no .agent-scratch, and a missing directory
    # here would turn a PASSING gate into a traceback at the last line.
    dest = Path(os.environ.get("VERIFY_LOG",
                               ".agent-scratch/minecu_tg_faithful_verify.log"))
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text("\n".join(LINES) + "\n")
    except OSError as e:
        print(f"(could not write the log to {dest}: {e})")
    sys.exit(code)
