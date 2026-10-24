#!/usr/bin/env python3
"""capture_cuda.py -- the generated CUDA C for this render, on a host with no GPU.

WHY THE NULL DEVICE. The CUDA *renderer* is a pure function of the UOp graph;
it does not need a device. `DEV=NULL:CUDA:sm_75` attaches the CUDA renderer to
tinygrad's no-op device, so the schedule, the fusion, the scheduling and the
emitted C are the real ones and nothing is executed. `use_nvcc` is forced on
because the default NVRTC path dlopen()s libnvrtc, which does not exist here.

This is a DIFFERENT device string from `CUDA`, so anything measured through it
is labelled as such. The T4 run measures the same numbers from the real backend
-- that is what `minecu_tg/kaggle/` does -- and the notes quote the T4 numbers
where they exist and these only for the source text.

    python3 minecu_tg/capture_cuda.py --outdir out/tg-codegen-sm75 --steps 1,4,32
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("PARALLEL", "0")
os.environ.setdefault("DEBUG", "0")
# the NULL allocator refuses copyout; this run never reads results back, and the
# flag only keeps a stray .numpy() from aborting the census
os.environ.setdefault("NULL_ALLOW_COPYOUT", "1")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch", default="sm_75")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--world", type=int, default=16)
    ap.add_argument("--res", type=int, default=32)
    ap.add_argument("--steps", default="1,4,8,32")
    ap.add_argument("--outdir", type=Path, required=True)
    a = ap.parse_args()

    # tinygrad's tensor-core table does int(arch[3:]) and cannot parse an
    # architecture-specific suffix, so hand it the base arch and let nvcc do the
    # real compile with the full name. Same table for both.
    dev_arch = re.sub(r"(sm_\d+)[a-z]$", r"\1", a.arch)
    os.environ["DEV"] = f"NULL:CUDA:{dev_arch}"

    from tinygrad.renderer import cstyle
    _orig = cstyle.CUDARenderer.__init__

    def patched(self, target, use_nvcc=False):
        return _orig(self, target, use_nvcc=True)

    cstyle.CUDARenderer.__init__ = patched

    # This host has no nvcc, and the NULL program never executes the binary, so
    # the compile is replaced by a capture. The SOURCE -- which is what this
    # script exists for -- is finished before compile_cached is called, so the
    # generated C is the real one. Any cubin for this capture has to come from
    # the T4 run or the offline archlab route; there is none here, and the notes
    # do not quote SASS from this path.
    #
    # Installed here, before any tinygrad import that can build a device,
    # because the renderer is constructed on first use and compiles then.
    from tinygrad.device import Compiler
    ALL_SRC: list[str] = []
    _orig_cc = Compiler.compile_cached

    def _capture_only(self, src, *a, **k):
        ALL_SRC.append(str(src))
        return b""

    Compiler.compile_cached = _capture_only

    import numpy as np
    from tinygrad import Tensor, dtypes, Device, GlobalCounters

    from minecu_tg import codegen
    from minecu_tg.render import camera_basis, render

    # The world is built by `minecu_tg.generate_world` on the CPU in a child
    # process, because the NULL device cannot copy results out (ops_null.py:28
    # raises "no copyout on NULL"), and the inputs have to be real bytes before
    # they are handed to a device that cannot execute. A subprocess rather than
    # an in-process DEV switch, because tinygrad resolves Device.DEFAULT once
    # and caches it.
    import subprocess
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
            "cams = np.array([[w/2.0, float(g), w/2.0, 0.001*(i%%13-6), 0.0]"
            " for i in range(B)], dtype=np.float32)\n"
            "np.savez(%r, vox=vox, cams=cams)\n"
        ) % (str(HERE.parent), a.world, a.world // 4, a.batch, str(cache))
        subprocess.run([sys.executable, "-c", child], check=True)
    z = np.load(cache)
    vox_np, cams = z["vox"], z["cams"]
    vox = Tensor(vox_np, dtype=dtypes.int8)
    basis = camera_basis(Tensor(cams, dtype=dtypes.float32), 70.0, a.res, a.res).contiguous()
    basis.realize()        # realize, never read back: NULL cannot copy out

    steps = [int(x) for x in a.steps.split(",")]
    c = codegen.census(lambda ms: render(vox, basis, a.res, a.res, ms),
                       steps, a.outdir, capture_only=True)
    c["arch"] = a.arch
    c["device_note"] = "NULL device, CUDA renderer attached, no hardware"

    import json
    (a.outdir / "census.json").write_text(json.dumps(c, indent=2))
    (a.outdir / "all_sources.txt").write_text(
        "\n\n/* ===== next kernel ===== */\n\n".join(ALL_SRC))
    print(f"captured {len(ALL_SRC)} kernel sources in total "
          f"({sum(len(s) for s in ALL_SRC)} bytes) -> {a.outdir}/all_sources.txt")
    print()
    print(codegen.summarise(c))
    print()
    for r in c["rows"]:
        counts = r["launch_counts_by_program"]
        top = max(counts.items(), key=lambda kv: kv[1])
        print(f"  ms={r['max_steps']:3d}  {r['kernel_launches']:4d} launches, "
              f"{r['distinct_programs_launched']:3d} distinct programs, "
              f"most-repeated: {top[1]:3d} x {top[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
