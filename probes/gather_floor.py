"""gather_floor.py - does a TinyGrad gather have an O(source) floor?

Motivating question. blaze/core/obs_camera.h needs 96 voxels per ray per frame
out of a 1,328,176-cell region tensor. netherite's CUDA kernel does one indexed
load per DDA step, one thread per ray, and reads exactly the cells it visits.

If TinyGrad's gather instead touches the whole source no matter how few elements
you want, then a 96-step DDA over a 1.33M-cell region has a hard traffic floor
of 96 x 5.3 MB = 509 MB per env per frame, to deliver 96 useful voxels. That
would be the real reason the port is 4 orders of magnitude off, and it would be
a property of the lowering rather than of the kernel.

Method: gather a FIXED, tiny number of elements (64) and grow the source. If the
cost tracks the source size, the floor is real. Compare against gathering many
elements from a small source, which is the case a compiler is designed for.
"""
import os
import re
import subprocess
import sys

SNIP = """
import numpy as np
from tinygrad import Tensor
S = {src}; O = {out}
Tensor(np.random.randint(0, 4096, size=S).astype(np.int32))[
    Tensor(np.random.randint(0, S, size=O).astype(np.int32))].realize()
"""


def run(src, out):
    env = dict(os.environ, DEBUG="2")
    p = subprocess.run([sys.executable, "-c", SNIP.format(src=src, out=out)],
                       capture_output=True, text=True, env=env)
    txt = re.sub(r"\x1b\[[0-9;]*m", "", p.stdout + "\n" + p.stderr)
    tot, red = 0.0, 0
    for line in txt.splitlines():
        if "***" not in line or " copy " in line:
            continue
        m = re.search(r"([\d.]+)(us|ms)/", line)
        if m:
            tot += float(m.group(1)) * (1000.0 if m.group(2) == "ms" else 1.0)
        if re.search(r"\*\*\*\s+\S+\s+\d+\s+r_", line):
            red += 1
    return tot, red


def main():
    print("Gathering 64 elements out of a source of growing size.")
    print("If cost tracks source size, the gather has an O(source) floor.\n")
    print(f"{'source':>12} {'src MB':>8} {'us':>9} {'GB/s over src':>15} {'reduces':>8}")
    for src in (65536, 262144, 1048576, 1328176, 4194304, 16777216):
        t, red = run(src, 64)
        print(f"{src:>12,} {src*4/1e6:8.1f} {t:9.1f} {src*4/t/1e3:15.1f} {red:8d}")

    print("\nFor contrast, many elements from a small source (the case a")
    print("compiler is designed for):\n")
    print(f"{'source':>12} {'out':>10} {'us':>9} {'reduces':>8}")
    for src, out in ((16384, 4096), (16384, 65536), (16384, 1048576)):
        t, red = run(src, out)
        print(f"{src:>12,} {out:>10,} {t:9.1f} {red:8d}")


if __name__ == "__main__":
    main()
