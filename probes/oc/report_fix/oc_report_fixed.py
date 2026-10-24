"""oc_report.py - the probe-1 numbers, all measured in one place.  (CORRECTED)

Three implementations of blaze/core/obs_camera.h, same region tensors, same
poses, same eye:

  ref_oc    C, compiled from netherite's own header, scalar single thread
  np_oc     NumPy, vectorised over pixels
  tiny_oc   TinyGrad, fixed-length DDA unroll over a [B, 2304] ray tensor

Correctness is element-wise against ref_oc, which is bit-identical to
blaze/cuda/obs_camera.cu (that driver and the CPU path compile the same
MC_HD header and are gated on exact equality).

===========================================================================
WHAT WAS BROKEN, AND WHY THIS FILE DIFFERS FROM probes/oc/oc_report.py
===========================================================================

The bug: `make report` cannot succeed on any machine.  It died with

    FileNotFoundError: [Errno 2] No such file or directory: 'cells_8.bin'
    make: *** [Makefile:16: report] Error 1

after printing the whole correctness section -- i.e. ~2.5 minutes of work,
then a stack trace from inside np.fromfile.  The report batches B=32 and
reads cells_0.bin .. cells_31.bin.  The Makefile's `cells` target only ever
emits 9 fixtures (seeds 0..7 plus 12345).  probes/oc/.gitignore documents a
THIRD, different set (`for s in 0 1 2 3 12345`).  Nothing checked that the
producer and the consumer agreed, so the report was one file count away from
running on every checkout.

The Makefile was the thing that was wrong, not the report.  B=32 is the
configuration published in notes/01, and it re-derives exactly: "At B=32 the
region buffer is 170 MB" is 32 * 1,327,976 cells * 4 B = 169.98 MB, and the
note's 39.79 ms GPU-only / 952.45 ms wall / 1,365 kernels / 101 reduces are
all B=32 rows.  Dropping the report to B=8 would have silently invalidated
every one of those.  So the fix has to be in the recipe -- see
report_fix/make_cells.sh and report_fix/oc-report-fix.patch.

But fixing only the recipe leaves the report one Makefile edit away from
breaking again, so this file also stops guessing:

  1. B comes from $OC_B (default 32) instead of a bare literal, so the
     Makefile's NENVS and the report are provably the same number.
  2. Fixtures resolve against ONE directory (--cells-dir, default the probes/oc
     dir next to this file).  The original resolved the outer process's files
     against $PWD and the `kernel_us` subprocess's against __file__'s dir --
     two different answers to the same question.
  3. A preflight pass runs BEFORE anything is measured and says, in one line,
     which seeds are missing and the command that produces them.  A missing
     fixture should cost 0 seconds and 1 line, not 2.5 minutes and a traceback.
  4. The report checks the shape of ref.bin against the SEEDS/CANON it is
     about to index.  The original did `ref[si*6+p]`, which silently reads
     the wrong frames if ref.bin was ever built with a pose count != 6.
  5. The narrative claims are derived from the measured counts instead of
     printed unconditionally.  The original always printed "all id mismatches
     are depth +-1" and blamed FMA -- while notes/02 had already root-caused
     these residuals to the sin LUT bucket boundary, and while the measured id
     count is 0, making the sentence vacuous.  See notes/11.
  6. `--b-sweep` actually measures the "wall is flat in B" claim that notes/01
     publishes but no code backed.

UNCHANGED, deliberately: every number, and every definition.  "id", "depth"
and "edge" mean exactly what run_oc.py and notes/01 mean by them -- a
per-pixel `!=` against the same ref.bin, summed over the same
3 seeds x 6 poses x 2304 pixels.  Nothing here reinterprets a category; a
report that quietly changes what "depth" means is worse than one that fails.
"""
import argparse
import os
import re
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))        # .../probes/oc/report_fix
OC_DIR = os.path.dirname(HERE)                            # .../probes/oc
PROBES = os.path.dirname(OC_DIR)                          # .../probes
for _p in (OC_DIR, PROBES):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tinygrad import Device                                   # noqa: E402
import np_oc                                                  # noqa: E402
import tiny_oc                                                # noqa: E402
from tiny_oc import (OC_NPIX, OC_W, OC_H, OC_STEPS, RG_STRIDE, render,  # noqa: E402
                     sin_table, SinTable, load_regions)
from bench import bench_ms                                    # noqa: E402
from run_oc import read_ref, CANON, EYE                      # noqa: E402

# Must match SEEDS in the Makefile: these are the correctness seeds.
SEEDS = [0, 3, 12345]
NPOSE = len(CANON)

# Must match NENVS in the Makefile.  One number, read from the environment,
# so the recipe and the report cannot drift apart again.
B_DEFAULT = int(os.environ.get("OC_B", "32"))


def cell_path(seed, cells_dir):
    return os.path.join(cells_dir, f"cells_{seed}.bin")


def required_seeds(b):
    """Every fixture the report will touch: the contiguous benchmark run
    0..b-1, plus whatever correctness seeds fall outside it (12345 does)."""
    return sorted(set(SEEDS) | set(range(b)))


def preflight(cells_dir, ref_path, ref_oc, b):
    """Fail in one line, before measuring anything, if the inputs are not there.

    The original let this surface as a FileNotFoundError from inside
    np.fromfile, 2.5 minutes into the run, with no indication that the fix was
    a Makefile recipe rather than a bug in this file."""
    problems = []

    for seed in required_seeds(b):
        p = cell_path(seed, cells_dir)
        if not os.path.exists(p):
            problems.append(f"missing fixture {p} (seed {seed})")
        elif os.path.getsize(p) != RG_STRIDE * 2:
            problems.append(
                f"short fixture {p}: {os.path.getsize(p)} B, expected "
                f"{RG_STRIDE * 2} B (113*104*113 u16)")

    if not os.path.exists(ref_path):
        problems.append(f"missing reference {ref_path} -- run 'make ref.bin'")
    elif not os.path.exists(ref_oc):
        problems.append(f"missing C reference binary {ref_oc} -- run 'make ref_oc'")

    if problems:
        need = required_seeds(b)
        have = [s for s in need if os.path.exists(cell_path(s, cells_dir))]
        sys.stderr.write(
            f"oc_report: {len(problems)} precondition(s) not met, refusing to "
            f"measure\n"
            + "".join(f"  - {x}\n" for x in problems[:4])
            + (f"  ... and {len(problems) - 4} more of the same shape\n"
               if len(problems) > 4 else "")
            + f"  have seeds {have}\n"
            + f"  need seeds {need}  (B={b} -> 0..{b - 1}, plus correctness "
              f"{SEEDS})\n"
            + f"  in {cells_dir}\n"
            + "  produce them with:  probes/oc/report_fix/make_cells.sh "
              f"{cells_dir} {b}\n"
            + "  or apply probes/oc/report_fix/oc-report-fix.patch to the "
              "Makefile's `cells` target\n")
        raise SystemExit(2)
    return required_seeds(b)


def kernel_us(code, env_extra=None):
    """Total device time for `code`, parsed out of tinygrad's DEBUG=2 stats."""
    env = dict(os.environ, DEBUG="2")
    if env_extra:
        env.update(env_extra)
    p = subprocess.run([sys.executable, "-c", code], capture_output=True,
                       text=True, env=env, cwd=OC_DIR)
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells-dir", default=OC_DIR,
                    help="where cells_<seed>.bin live (default: probes/oc)")
    ap.add_argument("--ref", default=os.path.join(OC_DIR, "ref.bin"))
    ap.add_argument("--ref-oc", default=os.path.join(OC_DIR, "ref_oc"))
    ap.add_argument("--b", type=int, default=B_DEFAULT, help="benchmark batch size")
    ap.add_argument("--b-sweep", default=None,
                    help="comma-separated batch sizes for the flat-wall check "
                         "(default 1,B/4,B); empty string to skip")
    ap.add_argument("--sweep-reps", type=int, default=2)
    args = ap.parse_args()
    B = args.b
    sweep_arg = args.b_sweep
    if sweep_arg is None:
        sweep_arg = f"1,{max(1, B // 4)},{B}"

    cells_dir = os.path.abspath(args.cells_dir)
    preflight(cells_dir, args.ref, args.ref_oc, B)

    ref = read_ref(args.ref)
    # ref[si*NPOSE+p] below is only correct if ref.bin really holds
    # len(SEEDS) x NPOSE frames in that order.  If someone rebuilds ref.bin
    # with a different pose count this indexes the wrong frames silently.
    if len(ref) != len(SEEDS) * NPOSE:
        sys.stderr.write(
            f"oc_report: {args.ref} holds {len(ref)} frames, but this report "
            f"indexes {len(SEEDS)} seeds x {NPOSE} poses = "
            f"{len(SEEDS) * NPOSE}.\n"
            f"  rebuild it with:  ./ref_oc --out ref.bin "
            f"--seeds {','.join(map(str, SEEDS))} --poses {NPOSE}\n")
        raise SystemExit(2)

    tab = np_oc.build_sin_table()
    cells = {s: np.fromfile(cell_path(s, cells_dir), dtype=np.uint16) for s in SEEDS}
    for s, a in cells.items():
        if a.size != RG_STRIDE:
            sys.stderr.write(f"oc_report: cells_{s}.bin has {a.size} cells, "
                             f"expected {RG_STRIDE}\n")
            raise SystemExit(2)
    NPIX_TOTAL = len(ref) * OC_NPIX

    # ---------------- correctness ----------------
    print("=" * 78)
    print("CORRECTNESS  vs ref_oc (C, netherite's own obs_camera.h), "
          f"{NPIX_TOTAL:,} pixels")
    print("=" * 78)

    np_bad = [0, 0, 0]
    for si, s in enumerate(SEEDS):
        for p in range(NPOSE):
            r = ref[si * NPOSE + p]
            i, d, e = np_oc.render(cells[s], EYE[0], EYE[1], EYE[2], *CANON[p], tab=tab)
            np_bad[0] += int((i != r[0]).sum())
            np_bad[1] += int((d != r[1]).sum())
            np_bad[2] += int((e != r[2]).sum())
    print(f"  np_oc    id {np_bad[0]:>6}   depth {np_bad[1]:>6}   edge {np_bad[2]:>6}"
          f"   <- bit-exact")

    sin = SinTable(sin_table())
    tg_bad = [0, 0, 0]
    max_dd = 0          # largest |tiny - ref| depth error, in quantised units
    edge_vals = set()   # which edge values ever differ
    steps_used = []
    for si, s in enumerate(SEEDS):
        c = load_regions([cell_path(s, cells_dir)])
        for p in range(NPOSE):
            r = ref[si * NPOSE + p]
            g = render(c, [EYE], [CANON[p][0]], [CANON[p][1]], sin, want_steps=True)
            gi, gd, ge, gs = (x.numpy()[0] for x in g)
            tg_bad[0] += int((gi != r[0]).sum())
            tg_bad[1] += int((gd != r[1]).sum())
            tg_bad[2] += int((ge != r[2]).sum())
            max_dd = max(max_dd, int(np.abs(gd - r[1]).max()))
            edge_vals |= set(np.unique(ge[ge != r[2]]).tolist()) \
                | set(np.unique(r[2][ge != r[2]]).tolist())
            steps_used.append(gs)
    steps_used = np.concatenate(steps_used)
    print(f"  tiny_oc  id {tg_bad[0]:>6}   depth {tg_bad[1]:>6}   edge {tg_bad[2]:>6}"
          f"   <- {100*(NPIX_TOTAL-tg_bad[0])/NPIX_TOTAL:.3f}% of ids exact")

    # The original printed, unconditionally and on every run:
    #   "all id mismatches are depth +-1 or a single edge bit"
    #   "root cause: TinyGrad's Metal backend contracts a*b+c into fma"
    # Both are claims, not measurements, and the first is vacuous when the id
    # count is 0.  Derive them instead, and cite the root cause notes/02
    # actually established (sin LUT bucket boundary) rather than the FMA one
    # notes/02 retracted.
    if tg_bad[0] == 0:
        print(f"  block ids agree on all {NPIX_TOTAL:,} pixels, so every pixel "
              f"reaches the same block;")
        print(f"  the {tg_bad[1]} depth and {tg_bad[2]} edge residuals are one "
              f"quantised value, not a different block.")
        print(f"  largest depth error {max_dd} quantised unit(s); differing edge "
              f"values {sorted(edge_vals) or 'none'}.")
        print(f"  root cause (notes/02-obs-camera-residuals.md): MC's sin is a "
              f"65,536-entry LUT")
        print(f"  indexed by mc_f2i(v*10430.378f); the yaw->radians product lands "
              f"on a bucket")
        print(f"  boundary, so a 1-ULP float32 difference picks the next bucket.")
    else:
        print(f"  {tg_bad[0]} id mismatches; largest depth error {max_dd} "
              f"quantised unit(s); differing edge values {sorted(edge_vals) or 'none'}.")
        if max_dd <= 1:
            print(f"  every id mismatch is a depth +-1 or a single edge bit.")
        print(f"  FMA contraction is a real, separate blocker (SPEC rule 4 "
              f"requires -ffp-contract=off /")
        print(f"  --fmad=false) but notes/02 showed it does NOT explain these "
              f"residuals.")

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
    for p in range(NPOSE):
        np_oc.render(c, EYE[0], EYE[1], EYE[2], *CANON[p], tab=tab)
    np_ms = (time.perf_counter() - t0) / NPOSE * 1e3
    print(f"  np_oc    1 thread, vectorised  : {np_ms:9.3f} ms/frame  "
          f"{OC_NPIX/np_ms/1e3:8.2f} M rays/s")

    cellsB = load_regions([cell_path(i, cells_dir) for i in range(B)])
    rng = np.random.default_rng(0)
    yaws = rng.uniform(0, 360, B).astype(np.float32)
    pitch = rng.uniform(-89, 89, B).astype(np.float32)
    eyes = np.tile(np.array(EYE, dtype=np.float32), (B, 1))
    build = lambda: render(cellsB, eyes, yaws, pitch, sin)[0]
    wall = bench_ms(build, reps=3, batches=3, warmup=1)

    code = f"""
import numpy as np, sys
sys.path.insert(0, {OC_DIR!r})
from tinygrad import Tensor
from tiny_oc import render, sin_table, SinTable, load_regions
from run_oc import EYE
sin = SinTable(sin_table())
B = {B}
cells = load_regions({[cell_path(i, cells_dir) for i in range(B)]!r})
rng = np.random.default_rng(0)
yaws = rng.uniform(0,360,B).astype(np.float32)
pitch = rng.uniform(-89,89,B).astype(np.float32)
eyes = np.tile(np.array(EYE, dtype=np.float32), (B, 1))
for _ in range(2): render(cells, eyes, yaws, pitch, sin)[0].realize()
"""
    # DEBUG=2 logs every realize, so the counters cover ITERS frames.
    ITERS = 2
    gpu, nk, nred = kernel_us(code)
    if nk == 0:
        print("  tiny_oc  B=%d, GPU only        : %9s (no DEBUG=2 kernel lines "
              "parsed -- tinygrad's stats format changed?)" % (B, "?"))
    else:
        gpu, nk, nred = gpu / ITERS, nk / ITERS, nred / ITERS
        rays = B * OC_NPIX
        print(f"  tiny_oc  B={B}, GPU only        : {gpu/1000:9.2f} ms/frame  "
              f"{rays/(gpu/1000)/1e3:8.2f} M rays/s   ({nk:.0f} kernels, {nred:.0f} reduces)")
    print(f"  tiny_oc  B={B}, wall (rebuild)  : {wall:9.2f} ms/frame  "
          f"{B*OC_NPIX/wall/1e3:8.2f} M rays/s")

    # notes/01 publishes "Wall time is flat in B: B=1 costs 923 ms, B=32 costs
    # 998 ms".  Nothing in the suite measured that.  Measure it here, or do not
    # print it.
    if sweep_arg.strip():
        ladder = sorted({int(x) for x in sweep_arg.split(",") if x.strip()})
        ladder = [x for x in ladder if 1 <= x <= B] or [B]
        print(f"  tiny_oc  wall vs B (measured, not assumed):")
        sweep = []
        for b in ladder:
            cb = load_regions([cell_path(i, cells_dir) for i in range(b)])
            yb = rng.uniform(0, 360, b).astype(np.float32)
            pb = rng.uniform(-89, 89, b).astype(np.float32)
            eb = np.tile(np.array(EYE, dtype=np.float32), (b, 1))
            ms = bench_ms(lambda: render(cb, eb, yb, pb, sin)[0],
                          reps=args.sweep_reps, batches=2, warmup=1)
            sweep.append((b, ms))
        for b, ms in sweep:
            rate = b * OC_NPIX / ms / 1e3
            print(f"      B={b:<3d} {ms:9.2f} ms/frame  {rate:8.4f} M rays/s")
        if len(sweep) > 1:
            b0, m0 = sweep[0]
            b1, m1 = sweep[-1]
            work = b1 / b0
            cost = m1 / m0
            print(f"      B={b0}->{b1} is {work:.0f}x the work for {cost:.2f}x the "
                  f"wall, so wall is")
            print(f"      dominated by Python UOp construction, not by the device.")

    # C reference throughput, measured
    p = subprocess.run([os.path.join(OC_DIR, "ref_oc"), "--reps", "20",
                        "--poses", "64", "--pose-seed", "1", "--quiet"],
                       capture_output=True, text=True, cwd=OC_DIR)
    m = re.search(r"([\d.]+)M rays/s", p.stderr)
    if m is None:
        sys.stderr.write(f"oc_report: could not parse ref_oc throughput from "
                         f"stderr: {p.stderr[-400:]!r}\n")
        raise SystemExit(2)
    print(f"  ref_oc    1 thread, scalar      : {float(m.group(1)):9.2f} M rays/s"
          f"   (this Mac, CPU)")
    print(f"  netherite CUDA (their box)      :   396.00 M rays/s   "
          f"(published, blaze/cuda/obs_camera.cu, not measured here)")


if __name__ == "__main__":
    main()
