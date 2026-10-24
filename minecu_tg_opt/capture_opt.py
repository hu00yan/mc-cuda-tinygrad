#!/usr/bin/env python3
"""capture_opt.py -- the generated CUDA for the optimized render, no GPU needed.

WHY THE NULL DEVICE. The CUDA *renderer* is a pure function of the UOp graph;
it does not need a device. `DEV=NULL:CUDA:sm_75` attaches the CUDA renderer to
tinygrad's no-op device, so the schedule, the fusion, the scheduling and the
emitted C are the real ones and nothing is executed. `use_nvcc` is forced on
because the default NVRTC path dlopen()s libnvrtc, which does not exist here.
This is a DIFFERENT device string from `CUDA`, so anything measured through it
is labelled as such; `battery_opt.py` measures the same numbers on the real
backend and the notebook quotes the T4 numbers where they exist.

The inputs are real bytes handed to a device that cannot execute, and the
capture is done in a child process because the NULL device cannot copy results
out (`ops_null.py:28`).

    python3 minecu_tg_opt/capture_opt.py --outdir out/minecu_tg_opt/codegen \\
        --variants naive,int_cursor,flat_gather,incremental,lean --steps 4,8,16,32,64

Every captured program is written out as `max_steps_NNNN/variant-K.cu`, so the
SASS stage compiles the exact source the scheduler produced and the instruction
counts are re-derivable from what is on disk.
"""
from __future__ import annotations

import argparse
import json
import os
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
    ap.add_argument("--arch", default="sm_75")
    ap.add_argument("--batch", type=int, default=1024,
                    help="the batch at which the graph is captured. 1024 is the "
                         "regime where the scheduler fuses; 8 is the regime where "
                         "it does not. Both are measured.")
    ap.add_argument("--world", type=int, default=16)
    ap.add_argument("--res", type=int, default=32)
    ap.add_argument("--steps", default="4,8,16,32,64")
    ap.add_argument("--variants", default="naive,int_cursor,flat_gather,incremental,lean")
    ap.add_argument("--outdir", type=Path, required=True)
    a = ap.parse_args()

    # tinygrad's tensor-core table does int(arch[3:]) and cannot parse an
    # architecture-specific suffix, so hand it the base arch and let nvcc do the
    # real compile with the full name. Same table for both sides.
    import re as _re
    dev_arch = _re.sub(r"(sm_\d+)[a-z]$", r"\1", a.arch)
    os.environ["DEV"] = f"NULL:CUDA:{dev_arch}"

    from tinygrad.renderer import cstyle
    _orig = cstyle.CUDARenderer.__init__

    def patched(self, target, use_nvcc=False):
        return _orig(self, target, use_nvcc=True)

    cstyle.CUDARenderer.__init__ = patched

    # This host has no nvcc and the NULL program never executes the binary, so
    # the compile is replaced by a capture. The SOURCE -- which is what this
    # script exists for -- is finished before compile_cached is called.
    from tinygrad.device import Compiler
    ALL_SRC: list[str] = []
    _orig_cc = Compiler.compile_cached

    def _capture_only(self, src, *a, **k):
        ALL_SRC.append(str(src))
        return b""

    Compiler.compile_cached = _capture_only

    import numpy as np
    from tinygrad import Tensor, Device, GlobalCounters, dtypes

    from minecu_tg import codegen
    from minecu_tg_opt.render import camera_basis, render_variant

    # The world, built on the CPU in a child process, because the NULL device
    # cannot copy results out and the inputs have to be real bytes.
    cache = a.outdir / "inputs.npz"
    a.outdir.mkdir(parents=True, exist_ok=True)
    if not cache.exists():
        child = (
            "import sys; sys.path.insert(0, %r)\n"
            "import os; os.environ['DEV']='CPU'\n"
            "import numpy as np\n"
            "from minecu_tg import generate_world\n"
            "w, g, B = %d, %d, %d\n"
            "vox = np.ascontiguousarray(np.stack([generate_world(w, g).numpy()]*B))\n"
            "cams = np.array([[w/2.0, float(g), w/2.0, 0.001*(i%%97-48), 0.0]\n"
            " for i in range(B)], dtype=np.float32)\n"
            "np.savez(%r, vox=vox, cams=cams)\n"
        ) % (str(REPO), a.world, a.world // 4, a.batch, str(cache))
        subprocess.run([sys.executable, "-c", child], check=True)
    z = np.load(cache)
    vox = Tensor(z["vox"], dtype=dtypes.int8)
    basis = camera_basis(Tensor(z["cams"], dtype=dtypes.float32), 70.0, a.res, a.res)
    basis.realize()          # realize, never read back: NULL cannot copy out

    variants = [v.strip() for v in a.variants.split(",") if v.strip()]
    steps = [int(x) for x in a.steps.split(",")]

    index = []
    for name in variants:
        # Each rung gets its own staging tree, because `codegen.census` writes
        # `max_steps_NNNN/kernelNNN.cu` and the rungs would otherwise overwrite
        # each other. The files are moved to
        # `<outdir>/max_steps_NNNN/<rung>/kernelNNN.cu` afterwards, which is the
        # layout `compile_sass.sh` walks, so the rung a SASS file came from is
        # in its own path.
        stage = a.outdir / f"_stage_{name}"
        c = codegen.census(
            lambda ms, n=name: render_variant(vox, basis, a.res, a.res, ms, n),
            steps, stage, keep_sources=True)
        c["variant"] = name
        c["uops"] = [r["uops_total_graph"] for r in c["rows"]]
        index.append(c)
        for r in c["rows"]:
            print(f"  {name:12s} max_steps={r['max_steps']:3d}  "
                  f"launches={r['kernel_launches']:3d}  "
                  f"distinct={r['distinct_programs_launched']:3d}  "
                  f"uops={r['uops_total_graph']:>12,}", flush=True)
            d = stage / f"max_steps_{r['max_steps']:04d}"
            if d.is_dir():
                tgt = a.outdir / f"max_steps_{r['max_steps']:04d}" / name
                tgt.mkdir(parents=True, exist_ok=True)
                for f in sorted(d.glob("kernel*.cu")):
                    f.replace(tgt / f.name)
        # the sources have been moved out; drop the staging tree
        for p in sorted(stage.rglob("max_steps_*"), reverse=True):
            if p.is_dir() and not any(p.iterdir()):
                p.rmdir()
        if stage.is_dir() and not any(stage.iterdir()):
            stage.rmdir()
    (a.outdir / "census.json").write_text(json.dumps(index, indent=2))
    (a.outdir / "arch.txt").write_text(a.arch + "\n")
    print(f"\ncaptured {len(ALL_SRC)} kernel sources -> {a.outdir}/")
    print(f"device {Device.DEFAULT}, arch {a.arch}, batch {a.batch}, "
          f"world {a.world}, res {a.res}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
