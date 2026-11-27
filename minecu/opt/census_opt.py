#!/usr/bin/env python3
"""census_opt.py -- kernel launches against max_steps and against batch.

The fusion decision is a property of the *scheduler*, so it is visible with no
GPU: `DEV=NULL:CUDA:sm_75` attaches the CUDA renderer to tinygrad's no-op
device and the schedule, the fusion and the launch list are the real ones.
That is what `capture_opt.py` uses, and this module sweeps the two axes the
earlier work found the decision to depend on.

  * `max_steps` -- does the launch count track the loop trip count? A count of
    `max_steps + k` means the loop was unrolled and NOT fused: one kernel per
    step, the loop-carried state in global memory between them. A count
    independent of `max_steps` means the scheduler fused the whole unrolled
    body into one kernel.
  * `batch` -- does that decision depend on the tensor size? It does, and where
    it flips is the crossover, so the batch axis is swept as finely as the time
    budget allows rather than assumed.

Both axes are swept for the naive port and for the optimized one, because the
fusion decision is about the graph the port hands over and the graph is exactly
what this lane changed.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("PARALLEL", "0")
os.environ.setdefault("DEBUG", "0")
os.environ.setdefault("NULL_ALLOW_COPYOUT", "1")

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batches", default="64,256,512,768,1024,1536,2048,4096")
    ap.add_argument("--steps", default="4,16,32")
    ap.add_argument("--world", type=int, default=16)
    ap.add_argument("--res", type=int, default=32)
    ap.add_argument("--variants", default="naive,lean")
    ap.add_argument("--arch", default="sm_75")
    ap.add_argument("--device", default="NULL",
                    help="NULL (CUDA renderer, no hardware) or CPU. The fusion "
                         "decision is the scheduler's, so NULL is the right "
                         "default and needs no card; the card run repeats it on "
                         "the real backend.")
    ap.add_argument("--out", type=Path, default=REPO / "out" / "minecu.opt" / "census.json")
    a = ap.parse_args()

    dev_arch = re.sub(r"(sm_\d+)[a-z]$", r"\1", a.arch)
    os.environ["DEV"] = f"{a.device}:CUDA:{dev_arch}" if a.device == "NULL" else a.device

    from tinygrad.renderer import cstyle
    _orig = cstyle.CUDARenderer.__init__

    def patched(self, target, use_nvcc=False):
        return _orig(self, target, use_nvcc=True)

    cstyle.CUDARenderer.__init__ = patched
    from tinygrad.device import Compiler
    Compiler.compile_cached = lambda self, src, *a2, **k: b""

    import numpy as np
    from tinygrad import Tensor, Device, dtypes

    from minecu.tinygrad import codegen, generate_world
    from minecu.tinygrad.render import camera_basis
    from minecu.opt.render import render_variant

    variants = [v.strip() for v in a.variants.split(",") if v.strip()]
    batches = [int(x) for x in a.batches.split(",")]
    steps = [int(x) for x in a.steps.split(",")]
    g = a.world // 4
    one = generate_world(a.world, g).numpy()

    print("=" * 78)
    print(f"  kernel census, DEV={os.environ['DEV']}, world {a.world}, "
          f"res {a.res}, max_steps {steps}")
    print("=" * 78)
    rows = []
    for b in batches:
        vox = Tensor(np.ascontiguousarray(np.stack([one] * b)), dtype=dtypes.int8)
        cams = np.array([[a.world / 2.0, float(g), a.world / 2.0,
                          0.001 * (i % 97 - 48), 0.0] for i in range(b)],
                        dtype=np.float32)
        basis = camera_basis(Tensor(cams, dtype=dtypes.float32), 70.0, a.res, a.res)
        basis.realize()
        for name in variants:
            c = codegen.census(
                lambda ms, n=name: render_variant(vox, basis, a.res, a.res, ms, n),
                steps)
            for r in c["rows"]:
                rows.append({"batch": b, "pixels": b * a.res * a.res,
                             "variant": name, "max_steps": r["max_steps"],
                             "launches": r["kernel_launches"],
                             "distinct": r["distinct_programs_launched"],
                             "names": r["n_distinct_names"],
                             "uops": r["uops_total_graph"]})
        line = "  " + f"batch {b:>5} ({b * a.res * a.res:>9,} px)  "
        for name in variants:
            sel = [r for r in rows if r["batch"] == b and r["variant"] == name]
            line += f"{name}: " + " ".join(f"ms{r['max_steps']}={r['launches']}" for r in sel) + "   "
        print(line, flush=True)
        del vox, basis

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"device": os.environ["DEV"], "arch": a.arch,
                                 "world": a.world, "res": a.res, "rows": rows},
                                indent=2))
    print(f"\n  wrote {a.out}")
    print(f"  device {Device.DEFAULT}, {len(rows)} measurements")

    # the crossover, stated rather than left for the reader
    print()
    print("  where the launch count stops depending on max_steps:")
    for name in variants:
        for b in batches:
            sel = [r for r in rows if r["batch"] == b and r["variant"] == name]
            lc = [r["launches"] for r in sorted(sel, key=lambda x: x["max_steps"])]
            if len(set(lc)) == 1:
                print(f"    {name:<8} batch {b:>5}: {lc} -- constant, so the "
                      f"unrolled body is fused")
    print("  (a launch count that rises with max_steps is one kernel per DDA "
          "step, state materialised in global memory between them)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
