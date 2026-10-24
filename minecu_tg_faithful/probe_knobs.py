#!/usr/bin/env python3
"""probe_knobs.py -- which ContextVars actually reach this render, and which do not.

WHY THIS FILE IS IN THE PACKAGE
-------------------------------
`crossings.py` asserts things about a dozen TinyGrad knobs. Most of those
assertions are "this knob cannot matter here", and a claim of the form "knob
X does not affect this kernel" is exactly the kind of claim that goes stale
when the framework moves. So they are measured, on this host, by diffing the
*generated CUDA* with and without the knob set.

INSTRUMENT AND ITS LIMITS
-------------------------
The instrument is the generated source: compile the graph for sm_75 through
TinyGrad's own renderer and compare the sources byte for byte. That is the
right instrument for "does this knob reach this graph at all" and the wrong
one for anything downstream of ptxas -- two different sources can compile to
the same SASS, and one source can compile to two different SASS depending on
ptxas's mood. So a DIFF here means "the knob changed the program", not "the
knob changes performance", and a SAME means the knob is not reachable from
this graph, which is a strong statement.

There is no GPU here and no nvrtc: `matmul/tg_backend.py` supplies a
capture-only `CUDA:0` whose compiler records the source and returns a
sentinel. The schedule, the fusion, `hand_coded_optimizations` and the
renderer are all stock.

COST, AND WHY `--steps` MATTERS
-------------------------------
This is slow -- minutes per configuration, and it grows with `max_steps`
because the DDA is inlined once per step. `--steps 4` is the default for
exactly that reason: knob *reachability* does not depend on `max_steps`, and a
4-step graph exercises the same code path as a 32-step one. The `max_steps=32`
census in `census.py` is where the shape that gets timed is captured.

    python3 minecu_tg_faithful/probe_knobs.py --steps 4 --batch 8
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PARALLEL", "0")
os.environ.setdefault("DEV", "CUDA:0")
os.environ.setdefault("ALLOW_DEVICE_USAGE", "1")

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "matmul"))
sys.path.insert(0, str(REPO / "minecu_ref" / "bench"))

import numpy as np  # noqa: E402

import dda_model  # noqa: E402
from tg_backend import generate  # noqa: E402

from tinygrad.helpers import Context  # noqa: E402

from minecu_tg_faithful import crossings  # noqa: E402
from minecu_tg_faithful.render import camera_basis, render  # noqa: E402

# Every knob the brief names, plus the ones `crossings.py` actually reasons
# about. `USE_TC` is named `TC` on the command line -- helpers.py:238 reads
# `getenv("TC", 1)` -- so the ContextVar key is `TC` and saying otherwise is
# how a knob ends up "not tested" when it was tested under the wrong name.
CANDIDATES: list[tuple[str, int]] = [
    ("NOOPT", 1),
    ("TC", 0),
    ("ALLOW_TF32", 1),
    ("NOLOCALS", 1),
    ("TC_OPT", 2),
    ("TC_SELECT", 0),
    ("TRANSCENDENTAL", 2),
    ("DISABLE_FAST_IDIV", 0),
    ("IMAGE", 1),
    ("FLOAT16", 1),
    ("MAX_KERNEL_BUFFERS", 2),
    ("MAX_KERNEL_BUFFERS", 4),
    ("MAX_KERNEL_BUFFERS", 8),
    ("MAX_KERNEL_BUFFERS", 64),
]


def digest(ks) -> str:
    """One hash over the whole program's source, in order.

    Order matters: a knob that permutes two kernels changes the schedule and
    must show as DIFF, so the digest is over the ordered concatenation and not
    over a set of per-kernel hashes.
    """
    h = hashlib.sha256()
    for k in ks:
        h.update(k.name.encode())
        h.update(k.src.encode())
    return h.hexdigest()[:16]


def build_inputs(world: int, batch: int, ground: int | None = None):
    g = world // 4 if ground is None else ground
    vox = np.ascontiguousarray(np.stack([dda_model.build_world(world, g)] * batch))
    cams = np.array([[world / 2.0, float(g), world / 2.0, 0.001 * (i % 13 - 6), 0.0]
                     for i in range(batch)], dtype=np.float32)
    return vox, cams


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch", default="sm_75")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--world", type=int, default=16)
    ap.add_argument("--res", type=int, default=32)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--modes", default="clean,faithful")
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args()

    vox, cams = build_inputs(a.world, a.batch)
    from tinygrad import Tensor, dtypes

    def sig(mode: str, ctx: dict) -> tuple[str, list]:
        def build():
            v = Tensor(vox, dtype=dtypes.int8)
            b = camera_basis(Tensor(cams, dtype=dtypes.float32), 70.0, a.res, a.res)
            o = render(v, b, a.res, a.res, a.steps, mode=mode)
            return o.schedule_linear(o)
        if ctx:
            with Context(**ctx):
                ks = generate(a.arch, build)
        else:
            ks = generate(a.arch, build)
        return digest(ks), ks

    report = {"arch": a.arch, "batch": a.batch, "steps": a.steps,
              "res": a.res, "rows": []}
    for mode in [m for m in a.modes.split(",") if m]:
        t = time.time()
        base, ks0 = sig(mode, {})
        print(f"== {mode}: {len(ks0)} kernels, {sum(len(k.src) for k in ks0)} B of "
              f"source, digest {base}  ({time.time() - t:.0f}s)", flush=True)
        for name, val in CANDIDATES:
            t = time.time()
            try:
                d, ks = sig(mode, {name: val})
            except Exception as e:  # a knob that raises is a result too
                report["rows"].append({"mode": mode, "knob": name, "value": val,
                                       "result": f"ERR {type(e).__name__}: {str(e)[:80]}"})
                print(f"   {name}={val:<4} ERR {type(e).__name__}: {str(e)[:70]}", flush=True)
                continue
            verdict = "SAME" if d == base else "DIFF"
            row = {"mode": mode, "knob": name, "value": val, "result": verdict,
                   "kernels": len(ks),
                   "source_bytes": sum(len(k.src) for k in ks),
                   "delta_bytes": sum(len(k.src) for k in ks) - sum(len(k.src) for k in ks0),
                   "seconds": time.time() - t}
            report["rows"].append(row)
            print(f"   {name}={val:<4} {verdict}  kernels={len(ks):3d} "
                  f"source={row['source_bytes']:9d}B ({row['delta_bytes']:+d})  "
                  f"({row['seconds']:.0f}s)", flush=True)

    # the claims in crossings.py, re-checked against what was just measured
    print("\n== crossings.py claims vs what was measured ==")
    by_knob: dict[str, str] = {}
    for r in report["rows"]:
        if r["result"] in ("SAME", "DIFF"):
            by_knob.setdefault(r["knob"], r["result"])
    # `USE_TC` is the ContextVar's *name*; the key it is set by is `TC`
    # (`helpers.py:238` reads `getenv("TC", 1)`). A lookup on the display name
    # would report "not tested" for a knob that was tested, which is exactly
    # the kind of report that gets copied into a note as a gap.
    ALIAS = {"USE_TC": "TC"}
    for k, v in crossings.EXPOSED.items():
        got = by_knob.get(ALIAS.get(k, k), "not tested")
        claim = v["effect"].split(".")[0]
        flag = "" if got == claim else "   <-- does not match the claim"
        print(f"   {k:22s} measured={got:10s} claims={claim}{flag}")

    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(report, indent=2))
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
