#!/usr/bin/env python3
"""Capture the generated CUDA for flat_gather with NOLOCALS off and on.

Runs on the NULL device, so the real scheduler, the real fusion decision and the
real C renderer all execute and nothing runs on a device. What comes out is the
string the CUDA backend compiles on the card.
"""
import os, sys
from pathlib import Path
os.environ.setdefault("DEV", "NULL:CUDA:sm_75")
os.environ.setdefault("NULL_ALLOW_COPYOUT", "1")
os.environ.setdefault("PARALLEL", "0")
REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
import numpy as np
from tinygrad import Tensor, dtypes
from minecu_tg import generate_world
from minecu_tg.render import camera_basis
from minecu_tg.codegen import install_capture, uninstall
from minecu_tg_opt.render import render_variant
from minecu_tg_opt.lm38_falsify import Nolocals, source_facts

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else REPO / ".agent-scratch" / "lm38b" / "nolocals_src")
OUT.mkdir(parents=True, exist_ok=True)
world, ground, res, steps, batch = 16, 4, 32, 32, 1024
one = generate_world(world, ground).numpy()
vox = Tensor(np.ascontiguousarray(np.stack([one] * batch)), dtype=dtypes.int8)
cams = np.array([[world/2.0, float(ground), world/2.0, 0.001*(i%97-48), 0.0] for i in range(batch)], dtype=np.float32)
basis = camera_basis(Tensor(cams, dtype=dtypes.float32), 70.0, res, res).contiguous()
basis.realize()

for tag, on in (("stock", False), ("nolocals", True)):
    sink, _ = install_capture()
    try:
        sink.clear()
        with Nolocals(on):
            g = render_variant(vox, basis, res, res, steps, "flat_gather")
            g.realize()
        srcs = list(sink)
    finally:
        uninstall(sink)
    big = max(srcs, key=len)
    f = source_facts(big)
    (OUT / f"flat_gather_{tag}.cu").write_text(big)
    print(f"{tag:<10} sources={len(srcs)} big={len(big):,}B stmts={f['source_statements']:,} "
          f"lb={f['launch_bounds']} sha={f['source_sha256'][:16]}", flush=True)
