#!/usr/bin/env python3
"""run_all.py -- run every experiment, in order, and report honestly.

    python3 experiment_b/run_all.py            # the full suite
    python3 experiment_b/run_all.py --fast     # small grids, ~4 min
    python3 experiment_b/run_all.py --only b2 b4
    python3 experiment_b/run_all.py --allow-gpu  (deliberately; see common)

Exit codes
  0  every selected experiment passed its own checks
  1  an experiment raised, printed a drift, or produced an empty result
  2  a GPU is visible and --allow-gpu was not given

The suite is device-free by construction. `common.require_no_gpu` enforces it,
because a green run on a machine with a card would no longer be evidence that
the questions are device-free -- tinygrad's `Device.DEFAULT` could resolve to a
real CUDA device somewhere and change what "DEV=CUDA" means.

`--fast` is the grid to use while iterating. The full suite spends most of its
time in b3's C=96/C=128 points and b2's three-backend census, both of which are
already measured and pinned; re-running them is for a clean verification, not
for information.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)

from experiment_b.common import (  # noqa: E402
    ArtifactDrift, device_status, head, pin_report, require_no_gpu, rule)

SCRIPTS = [
    ("b1", "byte accounting, from the emitted CUDA", ["b1_bytes.py"]),
    ("b2", "device-free schedule census on the CUDA backend",
     ["b2_schedule.py", "--arch-census"]),
    ("b3", "the kernel-count law in (C, NT)",
     ["b3_kernel_count.py", "--grid", "8,12,16,24,32,48,64",
      "--nt", "1,2,3,4,8,16"]),
    ("b4", "the instruction-class matrix, both sides",
     ["b4_codegen.py"]),
    ("b5", "why (a) is invalid here and (b) is not", ["b5_arch_gate.py"]),
    ("b6", "the one measurement that would give (b) a cost",
     ["b6_next_measurement.py"]),
]

FAST = {
    "b3": ["b3_kernel_count.py", "--grid", "8,12,16,24,32", "--nt", "1,2,4,8"],
    "b2": ["b2_schedule.py", "--quick"],
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--allow-gpu", action="store_true")
    a = ap.parse_args()

    head("EXPERIMENT B  --  the full suite")
    st = device_status()
    print(f"  GPU visible: {st['gpu_present']}   "
          f"nvcc: {st['nvcc']}   ptxas: {st['ptxas']}   ncu: {st['ncu']}")
    print(f"  mode: {'fast' if a.fast else 'full'}")
    print()

    if not a.allow_gpu:
        try:
            require_no_gpu()
        except RuntimeError as e:
            print(str(e))
            return 2
    else:
        print("  --allow-gpu given: the device-free property is NOT being asserted")
        print("  by this run, and any verdict below should be re-checked.\n")

    # artifact pins first: a drift here invalidates everything downstream
    head("ARTIFACT PINS")
    bad = 0
    for rel, want, status in pin_report():
        flag = "ok " if status == "OK" else "!! "
        print(f"  {flag}{status:<10} {rel}")
        if status != "OK":
            bad += 1
    print()
    if bad:
        print(f"  {bad} pinned artifact(s) MISSING or DRIFTED. Refusing to run:")
        print("  every number below would be stale. Re-derive against the new")
        print("  artifacts, then re-pin with:  python3 -m experiment_b.repin <path>")
        return 1
    print(f"  all {len(pin_report())} pins OK. Continuing.")

    todo = [s for s in SCRIPTS if a.only is None or s[0] in a.only]
    results = []
    for key, blurb, args in todo:
        argv = list(FAST[key]) if a.fast and key in FAST else list(args)
        cmd = [sys.executable, os.path.join(HERE, argv[0])] + argv[1:]
        head(f"RUNNING {key}: {blurb}")
        print("  " + " ".join(["python3", f"experiment_b/{argv[0]}"] + argv[1:]))
        t0 = time.perf_counter()
        p = subprocess.run(cmd, cwd=REPO)
        dt = time.perf_counter() - t0
        results.append((key, p.returncode, dt))
        print()
        print(f"  [{key}] rc={p.returncode}  {dt:.0f}s")

    head("SUMMARY")
    print(f"  {'exp':<5}{'rc':>5}{'seconds':>10}  what it settles")
    rule("-", 92)
    for key, blurb, _ in SCRIPTS:
        for k, rc, dt in results:
            if k == key:
                print(f"  {k:<5}{rc:>5}{dt:>9.0f}s  {blurb}")
    rc = max((r for _, r, _ in results), default=0)
    print()
    if rc == 0:
        print("  every selected experiment passed. The (b) findings are in")
        print("  notes/15-experiment-b.md, and every number in that note is")
        print("  printed by one of the runs above.")
    else:
        print(f"  at least one experiment exited nonzero (max rc {rc}). Whatever")
        print("  it printed is not a result until that is fixed.")
    return rc


if __name__ == "__main__":
    sys.exit(main())
