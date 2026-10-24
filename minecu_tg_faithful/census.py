#!/usr/bin/env python3
"""census.py -- launch and program counts, and the generated CUDA, offline.

NO GPU REQUIRED. The CUDA *renderer* is a pure function of the UOp graph, so
it runs with the compiler stubbed. `matmul/tg_backend.py` (read-only import)
supplies a capture-only `CUDA:0` whose compiler records the source and returns
a sentinel: `Compiled.__init__`, `NVCCRenderer` and the whole
`Schedule`/`to_program` pipeline are stock tinygrad, unmodified. The
`capture_cuda` harness in `minecu_tg/` does the same thing through the NULL
device; this one is the same idea with a shorter path from numpy to a kernel,
and it captures *both modes* so the clean-vs-faithful delta can be read off
the source rather than asserted.

What it reports, per mode and per shape:

    launches        GlobalCounters.kernel_count for one `realize()`
    distinct        distinct compiled *programs* (object id), not names
    per-kernel      name, source bytes, and which source file it went to

The distinct-vs-name distinction is carried over from `minecu_tg/codegen.py`
for the same reason it was introduced there: all 16 DDA programs at
`max_steps=32` are named `E_8_32_4_4_8_16_4` because the name comes from the
kernel's *bounds*, so a census that counted names would report "31 launches
of one kernel", which is false.

`PARALLEL=0` is mandatory, not an optimisation. `lower_and_compile` compiles
in a multiprocessing worker pool whenever more than one kernel is outstanding
(`tinygrad/engine/realize.py:273`), and the capture hook only sees in-process
compiles without it.

    python3 minecu_tg_faithful/census.py --batch 8 --steps 32 --outdir out/x
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PARALLEL", "0")
os.environ.setdefault("DEBUG", "0")
# DEV=CUDA:0 is the capture-only device installed by tg_backend.install(); it
# cannot allocate, so nothing here may realize or read back a device tensor.
os.environ.setdefault("DEV", "CUDA:0")
os.environ.setdefault("ALLOW_DEVICE_USAGE", "1")

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "matmul"))
sys.path.insert(0, str(REPO / "minecu_ref" / "bench"))

import numpy as np  # noqa: E402

import dda_model  # noqa: E402  the reference's own world builder
from tg_backend import generate  # noqa: E402

from minecu_tg_faithful import crossings  # noqa: E402
from minecu_tg_faithful.render import camera_basis, render  # noqa: E402


def inputs(world: int, batch: int, ground: int | None = None):
    """The world and the cameras, from the reference's own model.

    `dda_model.build_world` rather than `minecu_tg.world`, so the capture runs
    on voxels whose provenance is the reference's file and not this repo's
    transcription of it. `verify.py` checks the two agree exactly, so this
    choice costs nothing and buys an independent check on the census.
    """
    g = world // 4 if ground is None else ground
    vox = np.ascontiguousarray(np.stack([dda_model.build_world(world, g)] * batch))
    # a small yaw spread, so the basis kernel is not a constant-folded no-op
    cams = np.array([[world / 2.0, float(g), world / 2.0, 0.001 * (i % 13 - 6), 0.0]
                     for i in range(batch)], dtype=np.float32)
    return vox, cams


def build_fn(vox_np, cams, res, ms, mode, **kw):
    from tinygrad import Tensor, dtypes

    def build():
        vox = Tensor(vox_np, dtype=dtypes.int8)
        basis = camera_basis(Tensor(cams, dtype=dtypes.float32), 70.0, res, res)
        o = render(vox, basis, res, res, ms, mode=mode, **kw)
        # schedule_linear, NOT realize(): `Tensor.realize` filters out already-
        # realised uops (tinygrad/tensor.py:422) and is a silent no-op the
        # second time, which would report "0 kernels" and look like a result.
        return o.schedule_linear(o)
    return build


def census_one(arch: str, vox_np, cams, res, ms, mode, **kw) -> dict:
    ks = generate(arch, build_fn(vox_np, cams, res, ms, mode, **kw))
    counts = collections.Counter(k.name for k in ks)
    # The DDA programs are selected BY SIGNATURE, not by size. They are the
    # ones that read an int8 buffer and hold at least two `signed char val`
    # temporaries, which is what one voxel read per step looks like in the
    # generated C. Selecting by size would pick the same kernel in both modes
    # by accident rather than by construction, and "which kernel is this" is
    # the kind of thing that has to be pinned down rather than inferred.
    dda = [k for k in ks
           if re.search(r"signed char\* data\d+_\d+", k.src)
           and k.src.count("signed char val") >= 2]
    hot = max(dda or ks, key=lambda k: len(k.src))
    return {
        "mode": mode,
        "crossings": dict(crossings.effective(mode, **kw)),
        "kernels": len(ks),
        "dda_programs": len(dda),
        "source_bytes_total": sum(len(k.src) for k in ks),
        "distinct_names": len(counts),
        "per_kernel": [{"name": k.name, "src_bytes": len(k.src),
                        "dda": k in dda} for k in ks],
        "hot_name": hot.name if hot else None,
        "hot_src": hot.src if hot else "",
        "all_src": [k.src for k in ks],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch", default="sm_75")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--world", type=int, default=16)
    ap.add_argument("--res", type=int, default=32)
    ap.add_argument("--steps", default="32")
    ap.add_argument("--modes", default="clean,faithful")
    ap.add_argument("--also", default="",
                    help="extra addr_bits values to census, e.g. 32,64")
    ap.add_argument("--outdir", type=Path, required=True)
    a = ap.parse_args()

    vox, cams = inputs(a.world, a.batch)
    a.outdir.mkdir(parents=True, exist_ok=True)
    report = {"arch": a.arch, "batch": a.batch, "world": a.world,
              "res": a.res, "device": "capture-only CUDA renderer, no hardware",
              "rows": []}

    jobs: list[tuple[str, dict]] = [(m, {}) for m in a.modes.split(",") if m]
    for bits in [int(x) for x in a.also.split(",") if x]:
        for m in a.modes.split(","):
            if m:
                jobs.append((m, {"addr_bits": bits}))

    for ms in [int(x) for x in a.steps.split(",")]:
        for mode, kw in jobs:
            label = f"{mode}{'' if not kw else ''.join(f'/{k}{v}' for k, v in kw.items())}"
            t = time.time()
            row = census_one(a.arch, vox, cams, a.res, ms, mode, **kw)
            row.update(max_steps=ms, label=label, seconds=time.time() - t)
            d = a.outdir / f"b{a.batch}-ms{ms:04d}-{label}"
            d.mkdir(parents=True, exist_ok=True)
            # every program, not just the hot one: the whole-render ratio is
            # a sum over the graph, and a 13-instruction LUT kernel is as much
            # a part of this render as a 36 384-instruction fused one
            for i, src in enumerate(row.pop("all_src")):
                (d / f"k{i:03d}.cu").write_text(src)
            (d / "hot.cu").write_text(row["hot_src"])
            row["src_dir"] = str(d)
            report["rows"].append(row)
            print(f"  ms={ms:3d} {label:18s} kernels={row['kernels']:4d} "
                  f"(dda {row['dda_programs']:3d}) "
                  f"src={row['source_bytes_total']:9d}B "
                  f"names={row['distinct_names']:4d} "
                  f"hot={row['hot_name']} "
                  f"({row['seconds']:.0f}s)", flush=True)

    (a.outdir / "census.json").write_text(json.dumps(report, indent=2))
    print(f"\nwrote {a.outdir / 'census.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
