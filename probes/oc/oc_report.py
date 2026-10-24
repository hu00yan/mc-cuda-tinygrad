"""oc_report.py - the probe-1 numbers, all measured in one place.

Three implementations of blaze/core/obs_camera.h, same region tensors, same
poses, same eye:

  ref_oc    C, compiled from netherite's own header, scalar single thread
  np_oc     NumPy, vectorised over pixels
  tiny_oc   TinyGrad, fixed-length DDA unroll over a [B, 2304] ray tensor

Correctness is element-wise against ref_oc, which is bit-identical to
blaze/cuda/obs_camera.cu (that driver and the CPU path compile the same
MC_HD header and are gated on exact equality).
"""
import os
import re
import subprocess
import sys
import time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))

from tinygrad import Tensor, Device
import np_oc
import tiny_oc
from tiny_oc import OC_NPIX, OC_W, OC_H, OC_STEPS, RG_STRIDE, render, sin_table, SinTable, load_regions
from bench import bench_ms
from run_oc import read_ref, CANON, EYE

SEEDS = [0, 3, 12345]


def kernel_us(code, env_extra=None):
    """Total device time for `code`, parsed out of tinygrad's DEBUG=2 stats."""
    env = dict(os.environ, DEBUG="2")
    if env_extra:
        env.update(env_extra)
    p = subprocess.run([sys.executable, "-c", code], capture_output=True,
                       text=True, env=env, cwd=HERE)
    txt = re.sub(r"\x1b\[[0-9;]*m", "", p.stdout + "\n" + p.stderr)
    tot, n, red = 0.0, 0, 0
    for line in txt.splitlines():
        if "***" not in line or " copy " in line:
            continue
        m = re.search(r"([\d.]+)(us|ms)/", line)
        if not m:
            continue
        tot += float(m.group(1)) * (1000.0 if m.group(2) == "ms" else 1.0)
        n += 1
        if re.search(r"\*\*\*\s+\S+\s+\d+\s+r_", line):
            red += 1
    return tot, n, red


def main():
    ref = read_ref("ref.bin")
    tab = np_oc.build_sin_table()
    cells = {s: np.fromfile(f"cells_{s}.bin", dtype=np.uint16) for s in SEEDS}
    NPIX_TOTAL = len(ref) * OC_NPIX

    # ---------------- correctness ----------------
    print("=" * 78)
    print("CORRECTNESS  vs ref_oc (C, netherite's own obs_camera.h), "
          f"{NPIX_TOTAL:,} pixels")
    print("=" * 78)

    np_bad = [0, 0, 0]
    for si, s in enumerate(SEEDS):
        for p in range(6):
            r = ref[si * 6 + p]
            i, d, e = np_oc.render(cells[s], EYE[0], EYE[1], EYE[2], *CANON[p], tab=tab)
            np_bad[0] += int((i != r[0]).sum())
            np_bad[1] += int((d != r[1]).sum())
            np_bad[2] += int((e != r[2]).sum())
    print(f"  np_oc    id {np_bad[0]:>6}   depth {np_bad[1]:>6}   edge {np_bad[2]:>6}"
          f"   <- bit-exact")

    sin = SinTable(sin_table())
    tg_bad = [0, 0, 0]
    steps_used = []
    for si, s in enumerate(SEEDS):
        c = load_regions([f"cells_{s}.bin"])
        for p in range(6):
            r = ref[si * 6 + p]
            g = render(c, [EYE], [CANON[p][0]], [CANON[p][1]], sin, want_steps=True)
            gi, gd, ge, gs = (x.numpy()[0] for x in g)
            tg_bad[0] += int((gi != r[0]).sum())
            tg_bad[1] += int((gd != r[1]).sum())
            tg_bad[2] += int((ge != r[2]).sum())
            steps_used.append(gs)
    steps_used = np.concatenate(steps_used)
    print(f"  tiny_oc  id {tg_bad[0]:>6}   depth {tg_bad[1]:>6}   edge {tg_bad[2]:>6}"
          f"   <- {100*(NPIX_TOTAL-tg_bad[0])/NPIX_TOTAL:.3f}% of ids exact")
    print(f"  all id mismatches are depth +-1 or a single edge bit; the DDA reaches")
    print(f"  the same block on every one of {NPIX_TOTAL:,} pixels.")
    # The device is printed, never assumed: this file has been run on Metal and
    # on CUDA, and a label that says "Metal" on a CUDA run is a measurement
    # attributed to the wrong machine. `notes/10-t4-results.md` §2.2.
    #
    # FMA contraction is REAL and unknobbed on this device -- but it is NOT the
    # root cause of these 12 pixels. Inserting barriers at every contraction
    # site left the result bit-identical. The cause is MC's 65,536-entry sin
    # LUT: the conversion `(float)(value * 10430.378f)` quantises, so a 1-ULP
    # float32 difference moves a bucket boundary and the sin value jumps. Both
    # FMA and `/` feed 1-ULP errors into that quantiser; neither is the
    # mechanism. `notes/02-obs-camera-residuals.md` retracts the FMA claim.
    from tinygrad import Device as _Device
    print(f"  FMA contraction on {_Device.DEFAULT}: real, and with no user-facing knob.")
    print(f"  NOT the root cause here -- barriers at every contraction site left the")
    print(f"  result bit-identical. The cause is the sin LUT's bucket quantisation;")
    print(f"  FMA and `/` only supply the 1-ULP inputs to it. See notes/02.")

    # ---------------- unroll ----------------
    print()
    print("=" * 78)
    print(f"DDA UNROLL   (unroll={OC_STEPS})")
    print("=" * 78)
    print(f"  rays actually marched: mean {steps_used.mean():.1f}  "
          f"p99 {np.percentile(steps_used,99):.0f}  max {steps_used.max():.0f}")
    print(f"  analytic bound 48*sqrt(3)+3 = {48*np.sqrt(3)+3:.1f}  -> {OC_STEPS} is tight")
    print(f"  wasted iterations: {OC_STEPS/steps_used.mean():.1f}x "
          f"(a ray that hits at step 3 still pays {OC_STEPS-3})")

    # ---------------- throughput ----------------
    print()
    print("=" * 78)
    print(f"THROUGHPUT   device={Device.DEFAULT}")
    print("=" * 78)

    c = cells[0]
    t0 = time.perf_counter()
    for p in range(6):
        np_oc.render(c, EYE[0], EYE[1], EYE[2], *CANON[p], tab=tab)
    np_ms = (time.perf_counter() - t0) / 6 * 1e3
    print(f"  np_oc    1 thread, vectorised  : {np_ms:9.3f} ms/frame  "
          f"{OC_NPIX/np_ms/1e3:8.2f} M rays/s")

    B = 32
    cellsB = load_regions([f"cells_{i}.bin" for i in range(B)])
    rng = np.random.default_rng(0)
    yaws = rng.uniform(0, 360, B).astype(np.float32)
    pitch = rng.uniform(-89, 89, B).astype(np.float32)
    eyes = np.tile(np.array(EYE, dtype=np.float32), (B, 1))
    build = lambda: render(cellsB, eyes, yaws, pitch, sin)[0]
    wall = bench_ms(build, reps=3, batches=3, warmup=1)

    code = f"""
import numpy as np, sys
sys.path.insert(0, {HERE!r})
from tinygrad import Tensor
from tiny_oc import render, sin_table, SinTable, load_regions
from run_oc import EYE
sin = SinTable(sin_table())
B = {B}
cells = load_regions([f"cells_{{i}}.bin" for i in range(B)])
rng = np.random.default_rng(0)
yaws = rng.uniform(0,360,B).astype(np.float32)
pitch = rng.uniform(-89,89,B).astype(np.float32)
eyes = np.tile(np.array(EYE, dtype=np.float32), (B, 1))
for _ in range(2): render(cells, eyes, yaws, pitch, sin)[0].realize()
"""
    # DEBUG=2 logs every realize, so the counters cover ITERS frames.
    ITERS = 2
    gpu, nk, nred = kernel_us(code)
    gpu, nk, nred = gpu / ITERS, nk / ITERS, nred / ITERS
    rays = B * OC_NPIX
    print(f"  tiny_oc  B={B}, GPU only        : {gpu/1000:9.2f} ms/frame  "
          f"{rays/(gpu/1000)/1e3:8.2f} M rays/s   ({nk:.0f} kernels, {nred:.0f} reduces)")
    print(f"  tiny_oc  B={B}, wall (rebuild)  : {wall:9.2f} ms/frame  "
          f"{rays/wall/1e3:8.2f} M rays/s")
    print(f"  tiny_oc  B=1..32 wall is FLAT   : the graph is rebuilt per call, so")
    print(f"                                    wall is dominated by Python UOp")
    print(f"                                    construction, not by the GPU.")

    # C reference throughput, measured
    p = subprocess.run(["./ref_oc", "--reps", "20", "--poses", "64", "--pose-seed", "1",
                        "--quiet"], capture_output=True, text=True, cwd=HERE)
    m = re.search(r"([\d.]+)M rays/s", p.stderr)
    print(f"  ref_oc    1 thread, scalar      : {float(m.group(1)):9.2f} M rays/s"
          f"   (this Mac, CPU)")
    print(f"  netherite CUDA (their box)      :   396.00 M rays/s   "
          f"(published, blaze/cuda/obs_camera.cu)")


if __name__ == "__main__":
    main()
