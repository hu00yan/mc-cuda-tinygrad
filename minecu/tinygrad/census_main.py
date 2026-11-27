#!/usr/bin/env python3
"""census_main.py -- the census driver.

Runs one render per `max_steps` and reports the kernel count, the distinct
program count, and (on a backend with a `compile_cached`) the generated source.

    DEV=CPU  python3 minecu/tinygrad/census_main.py --outdir out/tg-codegen-cpu
    DEV=NULL:CUDA:75 python3 minecu/tinygrad/census_main.py --outdir out/tg-codegen-sm75

`NULL:CUDA:75` runs the CUDA *renderer* with no hardware behind it, which is how
the generated C is obtained on a host with no NVIDIA card. It is a different
device string from `CUDA`, so a number produced that way is labelled as such
everywhere it is quoted; the T4 run produces the same number from the real
backend.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# PARALLEL=0 must be set before tinygrad builds any device: it is what makes
# get_worker_pool() return None so the compile_cached hook in codegen.py sees
# every kernel instead of only the len(todo)==1 rounds. See codegen.py's
# docstring and notes/13-schedule-vs-decay.md section 4.
os.environ.setdefault("PARALLEL", "0")
os.environ.setdefault("DEBUG", "0")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))


def build_render(B: int, world: int, res: int, fov: float, ground: int | None):
    import numpy as np
    from tinygrad import Tensor, dtypes

    from minecu.tinygrad import generate_world
    from minecu.tinygrad.render import camera_basis, render

    g = world // 4 if ground is None else ground
    # numpy in, not a nested Python list. `Tensor(list)` takes a different,
    # much slower path that issues hundreds of small copies and would be
    # counted as part of the render; that is an input-construction artefact,
    # not a property of the DDA. See the census note in the README.
    vox_np = np.ascontiguousarray(np.stack([generate_world(world, g).numpy()] * B))
    # vary the camera per env so nothing downstream can dedupe the batch away
    cams = np.array([[world / 2.0, float(g), world / 2.0, 0.001 * (i % 13 - 6), 0.0]
                     for i in range(B)], dtype=np.float32)
    return vox_np, cams, render, camera_basis, dtypes, np


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--world", type=int, default=16)
    ap.add_argument("--res", type=int, default=32)
    ap.add_argument("--fov", type=float, default=70.0)
    ap.add_argument("--ground", type=int, default=None)
    ap.add_argument("--steps", default="1,2,3,4,6,8,12,16,24,32,48,64")
    ap.add_argument("--outdir", type=Path, default=None)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    from tinygrad import Tensor, Device

    from minecu.tinygrad import codegen

    vox_np, cams, render, camera_basis, dtypes, np = build_render(
        a.batch, a.world, a.res, a.fov, a.ground)

    vox = Tensor(vox_np, dtype=dtypes.int8)
    cams_t = Tensor(cams, dtype=dtypes.float32)
    basis = camera_basis(cams_t, a.fov, a.res, a.res).contiguous()
    basis.numpy()          # pre-arm, so the basis kernel is not in the count

    steps = [int(x) for x in a.steps.split(",")]
    c = codegen.census(lambda ms: render(vox, basis, a.res, a.res, ms),
                       steps, a.outdir)
    c["config"] = {"batch": a.batch, "world": a.world, "res": a.res,
                   "fov": a.fov, "ground": a.ground}
    c["device_note"] = ("NULL device with the CUDA renderer attached; the same "
                        "code generator, no hardware" if Device.DEFAULT == "NULL"
                        else Device.DEFAULT)

    if a.outdir:
        a.outdir.mkdir(parents=True, exist_ok=True)
        (a.outdir / "census.json").write_text(json.dumps(c, indent=2))
    if a.json:
        print(json.dumps(c, indent=2))
    else:
        print(codegen.summarise(c))
    return 0


if __name__ == "__main__":
    sys.exit(main())
