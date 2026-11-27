#!/usr/bin/env python3
"""battery.py -- the whole TinyGrad-side measurement, in one process.

Run on a real device. Emits JSON and a human-readable table, and never asserts:
every shape is reported as either measured or failed, and a failure is data.

WHAT IS MEASURED, IN ORDER
--------------------------
  1. environment          -- device string, driver-visible name if available
  2. census               -- kernel launches vs max_steps, distinct programs
  3. memory               -- allocator peak vs (batch, max_steps), including
                             the shapes that fail to allocate
  4. timing               -- the render graph, event-free wall clock around a
                             synchronized realize
  5. DDA workload         -- the mean steps/pixel the TinyGrad graph actually
                             executed, so the timing is attached to a workload
                             rather than to a kernel name
  6. observation digest   -- sha256 of the uint8 observation, so two runs can
                             be compared without shipping 96 MB

Item 5 is the one that makes the rest meaningful: the buried-agent episode in
`bench_cuda/buried_agent_fix.md` was four weeks of numbers meaning nothing
because nobody recorded what work the timed kernel did. Every timing here is
printed next to the DDA step count of the same run.
"""
from __future__ import annotations

import argparse
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
sys.path.insert(0, str(HERE.parent))

REPORT: dict = {"started": time.time(), "argv": sys.argv}


def log(*a):
    print(*a, flush=True)


def section(name: str) -> None:
    log("")
    log("=" * 78)
    log(f"  {name}")
    log("=" * 78)


# --------------------------------------------------------------------------
# 1. environment
# --------------------------------------------------------------------------
def environment() -> dict:
    section("1. environment")
    import numpy as np
    import tinygrad
    from tinygrad import Device

    env = {
        "Device.DEFAULT": Device.DEFAULT,
        "numpy": np.__version__,
        "tinygrad_path": str(Path(tinygrad.__file__).resolve().parent),
        "PARALLEL": os.environ.get("PARALLEL"),
        "DEBUG": os.environ.get("DEBUG"),
    }
    try:                                    # the card, as the driver sees it
        import subprocess
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                              "--format=csv,noheader"], capture_output=True, text=True)
        env["nvidia_smi"] = out.stdout.strip()[:300]
    except Exception as e:
        env["nvidia_smi"] = f"unavailable: {e}"
    for k, v in env.items():
        log(f"  {k:22s} {v}")
    return env


# --------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------
def build_world_tensors(batch: int, world: int, ground: int | None):
    """The reference's world, as tensors on the current device.

    The world is built by `minecu.tinygrad.world.generate_world`, which is verified
    against the reference's own CPU model (`minecu/dda_model.py`) at
    three world sizes. Identical for every env, as in the reference, which
    accepts a `seed` and never reads it.

    The third argument is the GROUND HEIGHT, not the resolution. Calling it
    `res` and passing a resolution of 32 into a 16-voxel world is not a typo
    that shows up as an error: it silently builds a world of solid dirt, the
    eye ends up inside it, every ray terminates on its first voxel test, and
    the timing then reports the cost of an obs store. That is the failure mode
    `bench_cuda/buried_agent_fix.md` exists to document, reached from the other
    direction. `check_configuration` is the gate that catches it.
    """
    import numpy as np
    from tinygrad import Tensor, dtypes

    from minecu.tinygrad import generate_world

    g = world // 4 if ground is None else ground
    one = generate_world(world, g).numpy()               # [Y, X, Z] int8
    vox = Tensor(np.ascontiguousarray(np.stack([one] * batch)), dtype=dtypes.int8)
    # The reference's timed state has per-env yaw and position because its agent
    # walks. Varying yaw here is the same kind of spread, and it stops any
    # downstream pass from proving the batch collapsible to one env.
    cams = np.array([[world / 2.0, float(g), world / 2.0,
                      0.001 * (i % 97 - 48), 0.0] for i in range(batch)],
                    dtype=np.float32)
    return vox, Tensor(cams, dtype=dtypes.float32), g


def check_configuration(world: int, res: int, ground: int) -> list[str]:
    """Refuse to run a degenerate configuration. Returns the failures.

    The reference's harness gates on a DDA step count and on the fraction of
    pixels hitting their first voxel, and a failure means *no timings are
    printed at all*, not a number with a caveat attached
    (`bench_cuda/bench.cu`, `dda_floor`). The same gate is here for the
    same reason: a render whose rays all die on their first voxel test is an
    obs store, and any comparison against one is a comparison of two stubs.
    """
    fails = []
    if ground >= world:
        fails.append(f"ground_height {ground} >= world {world}: the volume is solid, "
                     "every ray terminates on its first voxel test")
    if ground != world // 4:
        fails.append(f"ground_height {ground} != world/4 ({world // 4}): the visible "
                     "surface and physics_kernel's analytic floor disagree")
    return fails


def check_workload(dda: dict, max_steps: int) -> list[str]:
    """The runtime half of the gate, on the DDA statistics of the timed run.

    Floor is `max(2.0, 0.10 * max_steps)`, the reference's rule. It is a floor,
    not a target: the purpose is to catch a workload that collapsed, not to
    certify a particular one.
    """
    fails = []
    floor = max(2.0, 0.10 * max_steps)
    if dda["mean"] < floor:
        fails.append(f"mean DDA {dda['mean']:.3f} is below the floor {floor:.2f}")
    if dda["first_voxel_frac"] > 0.20:
        fails.append(f"{100 * dda['first_voxel_frac']:.1f}% of pixels hit on their "
                     "first voxel test (limit 20%)")
    return fails


# --------------------------------------------------------------------------
# 2. census
# --------------------------------------------------------------------------
def run_census(batch: int, world: int, ground: int, res: int,
               steps: list[int]) -> dict:
    section("2. kernel census: launches and distinct programs vs max_steps")
    from tinygrad import Device
    from minecu.tinygrad import codegen

    vox, cams, g = build_world_tensors(batch, world, ground)
    from minecu.tinygrad.render import camera_basis, render
    basis = camera_basis(cams, 70.0, res, res).contiguous()
    basis.realize()
    c = codegen.census(lambda ms: render(vox, basis, res, res, ms), steps)
    c["config"] = {"batch": batch, "world": world, "res": res, "ground": g}
    log(codegen.summarise(c))
    log("")
    log("  the same law, isolated: launches - 8 against max_steps")
    for r in c["rows"]:
        log(f"    ms={r['max_steps']:3d}  launches={r['kernel_launches']:3d}  "
            f"launches-8={r['kernel_launches'] - 8:3d}  "
            f"distinct={r['distinct_programs_launched']:3d}  "
            f"names={r['n_distinct_names']}")
    return c


# --------------------------------------------------------------------------
# 3. memory
# --------------------------------------------------------------------------
def run_memory(batch_list: list[int], world: int, ground: int, res: int,
               max_steps: int) -> list[dict]:
    section(f"3. allocator peak vs batch, at max_steps={max_steps}")
    import gc

    from tinygrad import Device
    from minecu.tinygrad import bench

    dev = Device.DEFAULT
    rows = []
    # ASCENDING on purpose, and with a liveness probe after every failure. At
    # max_steps=32 the state buffer is 256 B per pixel, so the headline shape
    # asks for 8 GiB per buffer and cannot be allocated. The batches below it
    # are the ones the comparison needs, so they are measured FIRST; and if a
    # failed allocation poisons the context, the probe detects that here rather
    # than leaving the timing stage to report nothing and be mistaken for
    # "TinyGrad is slow".
    for b in sorted(batch_list):
        gc.collect()
        try:
            vox, cams, g = build_world_tensors(b, world, ground)
            from minecu.tinygrad.render import camera_basis, render
            basis = camera_basis(cams, 70.0, res, res).contiguous()
            basis.realize()
        except Exception as e:
            rows.append({"batch": b, "realized": False,
                         "error": f"input build: {type(e).__name__}: {e}"[:300]})
            log(f"  batch {b:6d}  input build FAILED: {str(e)[:150]}")
            continue
        r = bench.bench_render(lambda: (vox, basis),
                               lambda vox, basis: render(vox, basis, res, res, max_steps),
                               dev, iters=3, warmup=1, label=f"mem_b{b}")
        r["batch"] = b
        rows.append(r)
        if r.get("realized"):
            log(f"  batch {b:6d}  peak live {r['mem']['peak_live_bytes']/2**20:10.1f} MiB  "
                f"total alloc {r['mem']['total_allocated_bytes']/2**20:10.1f} MiB  "
                f"allocs {r['mem']['n_allocs']}")
        else:
            log(f"  batch {b:6d}  NOT ALLOCATABLE: {r['error'][:200]}")
            live = probe_alive(dev, world, ground, res)
            log(f"  batch {b:6d}  context liveness probe after the failure: "
                f"{'OK' if live else 'DEAD -- the remaining batches will not be measured'}")
            if not live:
                rows.append({"batch": None, "realized": False,
                             "error": "context did not survive the failed allocation; "
                                      "remaining batches abandoned rather than "
                                      "reported as slow"})
                break
    return rows


def probe_alive(dev, world: int, ground: int, res: int, batch: int = 64) -> bool:
    """Can the device still run a small render? Used after a failed allocation."""
    import gc
    from minecu.tinygrad.render import camera_basis, render
    try:
        gc.collect()
        vox, cams, _ = build_world_tensors(batch, world, ground)
        basis = camera_basis(cams, 70.0, res, res).contiguous()
        basis.realize()
        out = render(vox, basis, res, res, 4)
        out.realize()
        n = int(out.shape[0])
        return n == batch
    except Exception:
        return False


def basis_ok(render, res, max_steps):
    def f(vox, basis):
        return render(vox, basis, res, res, max_steps)
    return f


# --------------------------------------------------------------------------
# 4/5/6. timing + workload + digest
# --------------------------------------------------------------------------
def run_timing(batch_list: list[int], world: int, ground: int, res: int,
               max_steps: int, iters: int, count_steps: bool) -> list[dict]:
    section(f"4. timing the render graph, max_steps={max_steps}")
    import gc

    from tinygrad import Device
    from minecu.tinygrad import bench

    dev = Device.DEFAULT
    rows = []
    for b in sorted(batch_list):
        gc.collect()
        try:
            vox, cams, g = build_world_tensors(b, world, ground)
            from minecu.tinygrad.render import camera_basis, render
            basis = camera_basis(cams, 70.0, res, res).contiguous()
            basis.realize()
        except Exception as e:
            rows.append({"batch": b, "realized": False,
                         "error": f"input build: {type(e).__name__}: {e}"[:300]})
            log(f"  batch {b:6d}  input build FAILED: {str(e)[:150]}")
            continue

        def f(vox_, basis_, ms=max_steps, cs=count_steps):
            return render(vox_, basis_, res, res, ms, count_steps=cs)

        # Two host-side numbers, both measured:
        #   t_trace  -- build the graph and throw it away. Pure Python tracing.
        #   t_end2end -- build the graph AND realize it, with one sync. What a
        #                caller pays, tracing included.
        # and one device number, measured with tinygrad's own per-kernel event
        # path (`bench.linear_device_time_ms`). The device number is the one
        # comparable to the reference's cudaEvent-bracketed kernel time; the
        # end-to-end one is what a caller actually waits for.
        t_trace = bench.time_call(lambda: render(vox, basis, res, res, max_steps),
                                  dev, iters=max(3, iters // 2), warmup=2)

        def f_full():
            out = render(vox, basis, res, res, max_steps)
            out.realize()

        t_end2end = bench.time_call(f_full, dev, iters=iters, warmup=3)

        r = bench.bench_render(lambda: (vox, basis), f, dev, iters=iters,
                               label=f"t_b{b}")
        r["batch"] = b
        r["ground"] = g
        if not r.get("realized"):
            log(f"  batch {b:6d}  FAILED: {r['error'][:150]}")
            rows.append(r)
            continue

        dev_t = r.get("device_time") or {}
        d_ms = dev_t.get("median_ms")
        r["t_trace_only"] = t_trace
        r["t_end_to_end"] = t_end2end
        pix = b * res * res
        line = (f"  batch {b:6d}  device {d_ms if d_ms is None else round(d_ms, 3)}"
                f" ms over {dev_t.get('n_kernels', '?')} kernels  "
                f"per-kernel {round(dev_t.get('median_per_kernel_us', 0), 1)} us")
        log(line)
        if d_ms:
            log(f"             {pix/(d_ms/1e3)/1e6:8.2f} Mpixel/s  "
                f"{b/(d_ms/1e3)/1e6:8.4f} M env-steps/s (device only)")
        log(f"             host: trace {t_trace['median_ms']:.3f} ms  "
            f"end-to-end {t_end2end['median_ms']:.3f} ms  "
            f"peak {r['mem']['peak_live_bytes']/2**20:9.1f} MiB  "
            f"allocs {r['mem']['n_allocs']}")
        if dev_t.get("error"):
            log(f"             device timing unavailable: {dev_t['error']}")

        # the workload this timing actually did
        if count_steps:
            out = render(vox, basis, res, res, max_steps, count_steps=True)
            steps = out[1].numpy().reshape(-1, res, res)
            r["dda"] = {
                "mean": float(steps.mean()),
                "min": int(steps.min()),
                "max": int(steps.max()),
                "p50": float(np.percentile(steps, 50)),
                "at_cap_frac": float((steps >= max_steps).mean()),
                "first_voxel_frac": float((steps == 1).mean()),
            }
            d = r["dda"]
            fails = check_workload(d, max_steps)
            r["gate"] = {"pass": not fails, "failures": fails}
            log(f"             DDA: mean {d['mean']:.3f}  min {d['min']}  "
                f"max {d['max']}  p50 {d['p50']:.0f}  "
                f"at cap {100*d['at_cap_frac']:.1f}%  "
                f"first voxel {100*d['first_voxel_frac']:.1f}%")
            log(f"             gate: {'PASS' if not fails else 'FAIL -- ' + '; '.join(fails)}")
            obs = out[0].numpy()
            h = hashlib.sha256(np.ascontiguousarray(obs)).hexdigest()
            r["obs_sha256"] = h
            log(f"             obs sha256 {h[:32]}...")
        rows.append(r)
    return rows


import numpy as np   # noqa: E402  (after sys.path, for the digest path above)


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--world", type=int, default=16)
    ap.add_argument("--res", type=int, default=32)
    ap.add_argument("--ground", type=int, default=None,
                    help="ground height; default world//4, the only value at which "
                         "the visible surface and physics_kernel's floor agree")
    ap.add_argument("--max-steps", type=int, default=32)
    ap.add_argument("--census-batch", type=int, default=64)
    ap.add_argument("--census-steps", default="1,2,4,8,16,32,64")
    ap.add_argument("--mem-batches", default="256,1024,4096,16384,32768")
    ap.add_argument("--time-batches", default="256,1024,4096,8192,16384")
    ap.add_argument("--iters", type=int, default=20)
    a = ap.parse_args()

    ground = a.world // 4 if a.ground is None else a.ground
    REPORT["config"] = (vars(a) | {"out": str(a.out), "ground": ground})
    a.out.mkdir(parents=True, exist_ok=True)
    mem_batches = [int(x) for x in a.mem_batches.split(",") if x.strip()]
    time_batches = [int(x) for x in a.time_batches.split(",") if x.strip()]

    section("0. configuration gate")
    cfg_fails = check_configuration(a.world, a.res, ground)
    REPORT["config_gate"] = {"pass": not cfg_fails, "failures": cfg_fails,
                             "world": a.world, "res": a.res, "ground": ground}
    log(f"  world={a.world} res={a.res} ground={ground} max_steps={a.max_steps}")
    if cfg_fails:
        for f in cfg_fails:
            log(f"  FAIL: {f}")
        log("  refusing to measure. No timings will be reported.")
        REPORT["finished"] = time.time()
        (a.out / "report.json").write_text(json.dumps(REPORT, indent=2, default=str))
        return 2
    log("  PASS")

    try:
        REPORT["env"] = environment()
        REPORT["census"] = run_census(
            a.census_batch, a.world, ground, a.res,
            [int(x) for x in a.census_steps.split(",")])
    except Exception:
        REPORT["census_error"] = traceback.format_exc()[-2000:]
        log(REPORT["census_error"])

    for key, fn, extra in (
        ("memory", run_memory, (mem_batches, a.world, ground, a.res, a.max_steps)),
        ("timing", run_timing, (time_batches, a.world, ground, a.res, a.max_steps,
                                a.iters, True)),
    ):
        try:
            REPORT[key] = fn(*extra)
        except Exception:
            REPORT[f"{key}_error"] = traceback.format_exc()[-2000:]
            log(REPORT[f"{key}_error"])

    REPORT["finished"] = time.time()
    (a.out / "report.json").write_text(json.dumps(REPORT, indent=2, default=str))
    section("wrote " + str(a.out / "report.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
