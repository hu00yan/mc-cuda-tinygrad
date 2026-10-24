"""gather_scale.py - is a TinyGrad indexed load O(output) or O(source)?

This is the question that decides whether the DDA camera is portable at all.
blaze/core/obs_camera.h gathers one voxel per ray per step out of a
113x104x113 = 1,328,176-cell region tensor. If the gather costs O(source), every
DDA step materialises an intermediate proportional to the whole world volume and
the port is dead on arrival. If it costs O(output), it is a normal indexed load
and the cost is just the unroll.

tinygrad's Tensor.gather is written as
    (index.unsqueeze(-1)._one_hot_along_dim(...).where(x, 0)).sum(-1)
which is O(source) by construction, and mixin/op.py has a special linear-index
fast path for *consecutive* multi-dim tensor indices. This measures which one a
1-D fancy index actually takes, on the GPU, by parsing DEBUG=2 kernel times.
"""
import os
import re
import subprocess
import sys

SRC_SIZES = [1024, 16384, 262144, 1328176, 5312704]
OUT = 32 * 2304

SNIP = """
import numpy as np
from tinygrad import Tensor
SRC = {src}; OUT = {out}
Tensor(np.random.randint(0, 4096, size=SRC).astype(np.int32))[
    Tensor(np.random.randint(0, SRC, size=OUT).astype(np.int32))].realize()
"""

TIME_RE = re.compile(r"([\d.]+)(us|ms)/")
KERNEL_RE = re.compile(r"\*\*\*\s+\S+\s+\d+\s+(\S+)\s+arg\s+(\d+)\s+mem")
COPY_RE = re.compile(r"\*\*\*\s+\w+\s+\d+\s+copy")


def run(src, out=OUT):
    env = dict(os.environ, DEBUG="2")
    p = subprocess.run([sys.executable, "-c", SNIP.format(src=src, out=out)],
                       capture_output=True, text=True, env=env)
    # tinygrad's DEBUG=2 per-kernel stats go to stdout; the device banner and
    # any traceback go to stderr. Stats have landed in both across versions.
    txt = re.sub(r"\x1b\[[0-9;]*m", "", p.stdout + "\n" + p.stderr)
    total, parts = 0.0, []
    for line in txt.splitlines():
        if "***" not in line:
            continue
        m, k = TIME_RE.search(line), KERNEL_RE.search(line)
        if not m:
            continue
        v = float(m.group(1)) * (1000.0 if m.group(2) == "ms" else 1.0)
        is_copy = bool(COPY_RE.search(line))
        total += v
        parts.append(("copy " if is_copy else "") + (k.group(1) if k else "?"))
    return total, parts


def main():
    print(f"indexed load, output fixed at {OUT:,} elements, source varied\n")
    print(f"{'source':>10} {'src MB (i32)':>12} {'GPU us':>10} {'us/1k src elems':>17}  kernels")
    base = None
    for s in SRC_SIZES:
        t, parts = run(s)
        dev = [p for p in parts if not p.startswith("copy")]
        if base is None:
            base = t
        print(f"{s:>10,} {s*4/1e6:12.1f} {t:10.1f} {t/(s/1000):17.4f}  "
              f"{len(dev)} device: {', '.join(dev[:3])}")
    print("\nIf the last column is flat, the gather is O(output) (a real indexed")
    print("load). If it grows with source, it is O(source) (one-hot + sum).")


if __name__ == "__main__":
    main()
