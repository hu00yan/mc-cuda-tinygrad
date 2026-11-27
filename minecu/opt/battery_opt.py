#!/usr/bin/env python3
"""battery_opt.py -- the whole measurement, in one process, on a real card.

WHAT IS MEASURED, IN ORDER
--------------------------
  1. environment          -- device string, the card as the driver sees it
  2. identity             -- the optimized render's observation against the naive
                             port's, on this card, at this batch, byte for byte
  3. timing               -- the render graph's device time, per variant, per
                             batch, with the DDA workload of the same run next
                             to it
  4. memory               -- the allocator's own peak, per variant, per batch
  5. observation digest   -- sha256, so two runs can be compared without
                             shipping the image

WHY THE INSTRUMENTS ARE IMPORTED RATHER THAN REWRITTEN
------------------------------------------------------
`minecu.tinygrad.bench` and `minecu.tinygrad.codegen` are imported, not copied. The
earlier work timed the naive port with those exact instruments, and a timing
taken with a different one is not comparable to it however correct it is. The
optimized port changes the *program*; it must not change the *ruler*.

THE TIMING BOUNDARY
-------------------
`bench.linear_device_time_ms` sums the per-kernel device times over the whole
compiled LINEAR, which is the same quantity as the reference's
`cudaEvent`-bracketed single launch: both are device time for producing this
step's observation, and nothing host-side is divided out. Host tracing is
reported separately and never mixed in, because a tracing framework's Python
cost is real and is not the thing a kernel comparison is about.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("PARALLEL", "0")
os.environ.setdefault("DEBUG", "0")

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))

REPORT: dict = {"started": time.time(), "argv": sys.argv}


def log(*a):
    print(*a, flush=True)


def section(name: str) -> None:
    log("")
    log("=" * 78)
    log(f"  {name}")
    log("=" * 78)


def build_world_tensors(batch: int, world: int, ground: int | None):
    """The reference's world, as tensors on this device.

    The reference gives every env the same world, so `minecu.tinygrad.world
    .generate_world` builds one [Y, X, Z] and it is repeated. The camera varies
    per env on a +-0.048 rad yaw spread, so nothing downstream can prove the
    batch collapsible to one env. At that spread the whole batch renders the
    same 3-colour image -- the observation cannot tell the envs apart, which is
    why `verify_opt.py` uses a +-1.2 rad spread for the identity check and this
    uses the small one for the timing.
    """
    import numpy as np
    from tinygrad import Tensor, dtypes

    from minecu.tinygrad import generate_world

    g = world // 4 if ground is None else ground
    one = generate_world(world, g).numpy()
    vox = Tensor(np.ascontiguousarray(np.stack([one] * batch)), dtype=dtypes.int8)
    cams = np.array([[world / 2.0, float(g), world / 2.0,
                      0.001 * (i % 97 - 48), 0.0] for i in range(batch)],
                    dtype=np.float32)
    return vox, Tensor(cams, dtype=dtypes.float32), g


def check_configuration(world: int, res: int, ground: int) -> list[str]:
    """Refuse to measure a degenerate configuration.

    The reference's harness gates on a DDA step count and on the fraction of
    pixels hitting their first voxel, and a failure means *no timings are
    printed at all*. A render whose rays all die on their first voxel test is
    an obs store, and a comparison against one is a comparison of two stubs.
    """
    fails = []
    if ground >= world:
        fails.append(f"ground_height {ground} >= world {world}: the volume is "
                     f"solid, every ray terminates on its first voxel test")
    if ground != world // 4:
        fails.append(f"ground_height {ground} != world/4 ({world // 4}): the "
                     f"visible surface and physics_kernel's floor disagree")
    return fails


def check_workload(dda: dict, max_steps: int) -> list[str]:
    floor = max(2.0, 0.10 * max_steps)
    fails = []
    if dda["mean"] < floor:
        fails.append(f"mean DDA {dda['mean']:.3f} is below the floor {floor:.2f}")
    if dda["first_voxel_frac"] > 0.20:
        fails.append(f"{100 * dda['first_voxel_frac']:.1f}% of pixels hit on "
                     f"their first voxel test (limit 20%)")
    return fails


def environment() -> dict:
    section("1. environment")
    import numpy as np
    import tinygrad
    from tinygrad import Device

    env = {"Device.DEFAULT": Device.DEFAULT, "numpy": np.__version__,
           "tinygrad_path": str(Path(tinygrad.__file__).resolve().parent),
           "PARALLEL": os.environ.get("PARALLEL")}
    try:
        import subprocess
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                              "--format=csv,noheader"], capture_output=True, text=True)
        env["nvidia_smi"] = out.stdout.strip()[:300]
    except Exception as e:
        env["nvidia_smi"] = f"unavailable: {e}"
    for k, v in env.items():
        log(f"  {k:22s} {v}")
    return env


def run_identity(vox, basis, res, ms, variants, batch_cap: int = 64) -> dict:
    """Both ports against each other, on a small batch, on THIS card.

    The authoritative identity check is in the notebook, and it is the stronger
    one: it compares both ports against the hand-written side's OWN dumped
    bytes. This one is a cheap in-process sanity check that costs one render
    per port at a batch small enough not to matter, so a variant that has gone
    wrong is caught here rather than three renders into the gate cell.
    """
    section("2. identity on this card: optimized vs naive port, byte for byte")
    import numpy as np
    from minecu.tinygrad.render import render as render_naive
    from minecu.opt.render import render_variant

    n = min(vox.shape[0], batch_cap)
    sub, subb = vox[:n], basis[:n]
    n_img = render_naive(sub, subb, res, res, ms).numpy()
    out = {"batch": n, "pixels": n * res * res, "rows": []}
    log(f"  batch {n}, {n * res * res:,} pixels, {res}x{res}, max_steps {ms}")
    h_n = hashlib.sha256(np.ascontiguousarray(n_img)).hexdigest()
    for name in variants:
        img = render_variant(sub, subb, res, res, ms, name).numpy()
        bad = int((img != n_img).any(axis=3).sum())
        h_o = hashlib.sha256(np.ascontiguousarray(img)).hexdigest()
        r = {"variant": name, "pixels_differing": bad,
             "identical": bool(np.array_equal(img, n_img)),
             "sha256": h_o, "sha256_naive": h_n}
        out["rows"].append(r)
        log(f"  {name:<13} pixels differing {bad:>9} / {n * res * res:>9}   "
            f"sha256 {h_o[:24]}  {'IDENTICAL' if r['identical'] else 'DIFFERS'}")
    out["all_identical"] = all(r["identical"] for r in out["rows"])
    if not out["all_identical"]:
        log("  *** A PORT DISAGREES WITH THE NAIVE PORT. Every number attached "
            "to it is meaningless. ***")
    del sub, subb
    return out


def run_timing(batches, world, ground, res, ms, iters, variants,
               workload_batch: int | None = None, budget_s: float = 1200.0
               ) -> list[dict]:
    section(f"3. timing the render graph, max_steps={ms}")
    import numpy as np
    from tinygrad import Device
    from minecu.tinygrad import bench                      # the earlier work's instrument
    from minecu.opt.render import render_variant

    dev = Device.DEFAULT
    rows = []
    cell_start = time.perf_counter()
    log(f"  a {budget_s:.0f}s budget over {len(batches)} batches x "
        f"{len(variants)} variants; a shape that has not started when it is gone "
        f"is reported as skipped, not as slow")
    # ONE graph build per (batch, variant). The previous shape of this loop
    # built four: one for the trace-only number, one for trace+realize, one
    # inside `bench_render`, and one more for the DDA workload. At batch 2048
    # and max_steps 32 a single build is tens of seconds of tracing plus an
    # NVRTC compile of a megabyte of CUDA, so four per cell was the whole
    # 1600 s budget and produced nothing. What is measured now is the device
    # time of the graph, the allocator's peak, and the DDA workload at ONE
    # batch -- the three numbers the note needs -- with one build each.
    for b in batches:
        gc.collect()
        try:
            vox, cams, g = build_world_tensors(b, world, ground)
            from minecu.tinygrad.render import camera_basis
            basis = camera_basis(cams, 70.0, res, res).contiguous()
            basis.realize()
        except Exception as e:
            rows.append({"batch": b, "realized": False,
                         "error": f"input build: {type(e).__name__}: {e}"[:300]})
            log(f"  batch {b:6d}  input build FAILED: {str(e)[:150]}")
            continue

        for name in variants:
            def f(vox_, basis_, n=name):
                return render_variant(vox_, basis_, res, res, ms, n)
            t0 = time.perf_counter()
            # A budget gate, not a timeout. tinygrad's cost here is dominated by
            # NVRTC compiling a kernel that grows with max_steps, and that
            # compile is not interruptible, so this bounds WHAT IS ATTEMPTED: a
            # shape already inside its compile finishes and is reported, and one
            # that has not started when the budget is gone is skipped and
            # reported as skipped. Without it the largest batch consumes the
            # whole cell and every smaller batch -- the ones the fusion
            # crossover sits between -- goes home unmeasured.
            r = {"batch": b, "variant": name, "realized": False, "error": None,
                 "label": f"{name}_b{b}"}
            if t0 - cell_start > budget_s:
                r["error"] = (f"skipped: the {budget_s:.0f}s budget was already "
                              f"spent on earlier shapes")
                log(f"  batch {b:6d} {name:<13} SKIPPED: {r['error']}")
                rows.append(r)
                continue
            try:
                r.update(bench.bench_render(lambda: (vox, basis), f, dev,
                                             iters=iters, label=r["label"]))
            except Exception as e:              # a failed shape is data
                r["error"] = f"{type(e).__name__}: {e}"[:300]
            r["batch"] = b
            r["variant"] = name
            r["build_seconds"] = time.perf_counter() - t0
            if not r.get("realized"):
                log(f"  batch {b:6d} {name:<13} FAILED: {r['error'][:150]}")
                rows.append(r)
                continue
            d = r.get("device_time") or {}
            d_ms = d.get("median_ms")
            pix = b * res * res
            log(f"  batch {b:6d} {name:<13} device "
                f"{d_ms if d_ms is None else round(d_ms, 3)}"
                f" ms over {d.get('n_kernels', '?')} kernels  "
                f"peak {r['mem']['peak_live_bytes'] / 2**20:9.1f} MiB  "
                f"({r['build_seconds']:.0f}s to build+time)"
                + (f"  {pix / (d_ms / 1e3) / 1e6:8.2f} Mpixel/s" if d_ms else ""))
            if d.get("error"):
                log(f"    device timing unavailable: {d['error']}")
            rows.append(r)

            # The DDA workload, at ONE batch. It is a property of the world and
            # the cameras, not of the generated code, and the cameras differ
            # only by a +-0.048 rad yaw spread, so measuring it at one batch and
            # saying so is honest where measuring it at four and timing out is
            # not. The gate still runs, and a failure is reported as a failure.
            if workload_batch is not None and b == workload_batch:
                out = render_variant(vox, basis, res, res, ms, name,
                                     count_steps=True)
                steps = out[1].numpy().reshape(-1, res, res)
                dd = {"mean": float(steps.mean()), "min": int(steps.min()),
                      "max": int(steps.max()),
                      "p50": float(np.percentile(steps, 50)),
                      "at_cap_frac": float((steps >= ms).mean()),
                      "first_voxel_frac": float((steps == 1).mean())}
                r["dda"] = dd
                fails = check_workload(dd, ms)
                r["gate"] = {"pass": not fails, "failures": fails}
                r["obs_sha256"] = hashlib.sha256(
                    np.ascontiguousarray(out[0].numpy())).hexdigest()
                log(f"    DDA mean {dd['mean']:.3f}  at cap "
                    f"{100 * dd['at_cap_frac']:.1f}%  first voxel "
                    f"{100 * dd['first_voxel_frac']:.1f}%"
                    f"  gate {'PASS' if not fails else 'FAIL -- ' + '; '.join(fails)}")
                del out, steps
        del vox, basis
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--world", type=int, default=16)
    ap.add_argument("--res", type=int, default=32)
    ap.add_argument("--ground", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=32)
    ap.add_argument("--variants", default="naive,lean")
    ap.add_argument("--time-batches", default="256,512,1024,2048,4096")
    ap.add_argument("--workload-batch", type=int, default=None,
                    help="the one batch at which the DDA workload is measured. "
                         "It is a property of the world and the cameras, not of "
                         "the generated code, so one batch carries it; measuring "
                         "it at every batch cost a build per batch and ran the "
                         "cell out of time.")
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--budget", type=float, default=1200.0,
                    help="seconds of wall clock for the timing stage. Bounds "
                         "what is ATTEMPTED: a shape already inside its NVRTC "
                         "compile finishes, and one not yet started is reported "
                         "as skipped. A shape is not abandoned mid-compile, "
                         "because that compile cannot be interrupted.")
    a = ap.parse_args()

    ground = a.world // 4 if a.ground is None else a.ground
    REPORT["config"] = vars(a) | {"out": str(a.out), "ground": ground}
    a.out.mkdir(parents=True, exist_ok=True)
    variants = [v.strip() for v in a.variants.split(",") if v.strip()]
    batches = [int(x) for x in a.time_batches.split(",") if x.strip()]
    wb = a.workload_batch
    if wb is None and batches:
        wb = batches[-1]

    section("0. configuration gate")
    cfg = check_configuration(a.world, a.res, ground)
    REPORT["config_gate"] = {"pass": not cfg, "failures": cfg}
    log(f"  world={a.world} res={a.res} ground={ground} max_steps={a.max_steps} "
        f"variants={variants} batches={batches}")
    if cfg:
        for f in cfg:
            log(f"  FAIL: {f}")
        log("  refusing to measure. No timings will be reported.")
        (a.out / "report.json").write_text(json.dumps(REPORT, indent=2, default=str))
        return 2
    log("  PASS")

    try:
        REPORT["env"] = environment()
    except Exception:
        REPORT["env_error"] = traceback.format_exc()[-2000:]

    try:
        vox, cams, g = build_world_tensors(min(64, batches[0] if batches else 64),
                                           a.world, ground)
        from minecu.tinygrad.render import camera_basis
        basis = camera_basis(cams, 70.0, a.res, a.res).contiguous()
        basis.realize()
        REPORT["identity"] = run_identity(vox, basis, a.res, a.max_steps, variants,
                                          batch_cap=64)
        del vox, basis
    except Exception:
        REPORT["identity_error"] = traceback.format_exc()[-2000:]
        log(REPORT["identity_error"])

    try:
        REPORT["timing"] = run_timing(batches, a.world, ground, a.res, a.max_steps,
                                      a.iters, variants, workload_batch=wb,
                                      budget_s=a.budget)
    except Exception:
        REPORT["timing_error"] = traceback.format_exc()[-2000:]
        log(REPORT["timing_error"])

    REPORT["finished"] = time.time()
    (a.out / "report.json").write_text(json.dumps(REPORT, indent=2, default=str))
    section(f"wrote {a.out / 'report.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
