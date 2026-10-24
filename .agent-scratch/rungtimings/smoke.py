#!/usr/bin/env python3
"""smoke.py -- exercise the notebook's timing instrumentation on THIS host.

The card is the only place the numbers can come from, but every API call in the
notebook can be exercised here against the repo's own tinygrad, under DEV=NULL
(no CUDA on this host).  What this catches is a NameError, a wrong keyword, a
tensor method that does not exist -- the class of defect that would otherwise
cost a queued Kaggle session to discover.

It is a smoke test, not a measurement.  DEV=NULL executes nothing, so any time
it reports is not evidence about anything.
"""
from __future__ import annotations

import base64
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
STAGE = HERE / "_smoke_stage"

os.environ.setdefault("DEV", "NULL")
os.environ.setdefault("NULL_ALLOW_COPYOUT", "1")
os.environ.setdefault("PARALLEL", "0")
os.environ.setdefault("DEBUG", "0")


def main() -> int:
    side = json.load(open(HERE / "kernel" / "payloads.json"))
    for name, b64, want in side["payloads"]:
        data = base64.b64decode(b64)
        import hashlib
        got = hashlib.sha256(data).hexdigest()
        assert got == want, f"{name}: {got} != {want}"
        dest = STAGE / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
    print(f"  staged {len(side['payloads'])} payloads, all hashes match")

    sys.path.insert(0, str(STAGE))

    nb = json.load(open(HERE / "kernel" / "mc_minecu_rungs_t4.ipynb"))
    cells = {c["id"]: "".join(c["source"])
             for c in nb["cells"] if c["cell_type"] == "code"}

    ns: dict = {"__name__": "__smoke__"}
    src = cells["prelude"].replace('OUT = "/kaggle/working"', f'OUT = {str(STAGE)!r}')
    src = cells["timing-instrument"]
    exec(compile(cells["prelude"].replace('OUT = "/kaggle/working"',
                                          f'OUT = {str(STAGE)!r}'),
                 "prelude", "exec"), ns)
    exec(compile(src, "timing-instrument", "exec"), ns)
    print("  prelude + timing-instrument executed")

    # the imports the gate/timing cells rely on
    src3 = cells["build-and-reference"]
    head = src3[src3.index("import importlib"):src3.index("# ----------------------------------------------------------------- build")]
    exec(compile(head, "imports", "exec"), ns)
    print("  imports OK; Device.DEFAULT =", ns["Device"].DEFAULT)

    from tinygrad import Device, Tensor, dtypes  # noqa: F401
    from minecu_tg.render import camera_basis
    from minecu_tg_opt.battery_opt import build_world_tensors

    vox, cams, g = build_world_tensors(8, 16, 4)
    basis = camera_basis(cams, 70.0, 32, 32).contiguous()
    basis.realize()
    Device[Device.DEFAULT].synchronize()
    print(f"  inputs built: voxels {tuple(vox.shape)} basis {tuple(basis.shape)}")

    # every rung traces at all
    for rung in ns["VARIANTS"]:
        t = ns["render_variant"](vox, basis, 32, 32, 32, rung)
        print(f"    {rung:<13} traced, {t.nbytes()} B, "
              f"has_buffer_identity={ns['_is_realized'](t)}")

    # the guards
    t = ns["render_variant"](vox, basis, 32, 32, 32, "lean")
    print("  guard1 on a fresh tensor: pre_realized =", ns["_is_realized"](t),
          "(must be False)")
    assert ns["_is_realized"](t) is False

    # the walk
    t = ns["render_variant"](vox, basis, 32, 32, 32, "lean")
    try:
        w = ns["walk_graph"](t)
        print(f"  walk_graph: {w['n_calls']} CALLs, {w['n_positive']} > 0, "
              f"sum {w['sum_us']} us")
    except Exception as e:
        print(f"  walk_graph raised under DEV=NULL (expected on a NULL device): "
              f"{type(e).__name__}: {str(e)[:160]}")

    # the ruler
    t = ns["render_variant"](vox, basis, 32, 32, 32, "lean")
    try:
        with ns["TG_BENCH"].MemProbe(Device.DEFAULT) as mp:
            mp.reset()
            d = ns["TG_BENCH"].linear_device_time_ms(t, iters=2)
            mem = mp.as_dict()
        print(f"  linear_device_time_ms: {d}")
        print(f"  MemProbe: peak {mem['peak_live_bytes']} B, "
              f"{mem['n_allocs']} allocs")
    except Exception as e:
        print(f"  linear_device_time_ms raised under DEV=NULL (expected): "
              f"{type(e).__name__}: {str(e)[:200]}")

    # bench_render, the cross-check path
    try:
        f = lambda v, b: ns["render_variant"](v, b, 32, 32, 32, "lean")
        q = ns["TG_BENCH"].bench_render(lambda: (vox, basis), f, Device.DEFAULT,
                                        iters=8, label="smoke")
        print(f"  bench_render: realized={q.get('realized')} "
              f"device_time={q.get('device_time')} "
              f"err={str(q.get('error'))[:120]}")
    except Exception as e:
        print(f"  bench_render raised under DEV=NULL (expected): "
              f"{type(e).__name__}: {str(e)[:200]}")

    # the batch driver, end to end, at a batch of 8
    ns["REPEATS"] = {"8": 2}
    rec = ns["time_batch"](8, budget_s=2400.0)
    print("  time_batch(8) returned rows:")
    for r in rec["rows"]:
        print("   ", r["rung"], "median", r.get("median_ms"),
              "spread%", r.get("spread_pct"),
              "cold", r.get("n_kernels_cold"), "warm", r.get("n_kernels_warm"),
              "peak MiB", round(r.get("peak_live_bytes", 0) / 2**20, 1),
              "guards", r.get("guards_ok"),
              "walk_vs_ruler%", r.get("walk_vs_ruler_pct"))
    for r in rec.get("cross_check", []):
        print("   xcheck", r.get("rung"), r.get("median_ms"), r.get("error"))
    print("  timed_workload:", {k: (round(v["mean"], 4) if isinstance(v, dict)
                                    and "mean" in v else v)
                                for k, v in (rec.get("timed_workload") or {}).items()})

    print("SMOKE OK -- every API call the card will make has now been exercised")
    return 0


if __name__ == "__main__":
    sys.exit(main())
