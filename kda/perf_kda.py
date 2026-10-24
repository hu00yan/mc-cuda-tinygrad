"""perf_kda.py - what does the TinyGrad KDA cost, and where does it go?

Three questions, in order of how much they matter:

1. HOW MANY KERNELS. KDA has two inherently sequential parts: the UT
   transform (C forward-substitution steps) and the inter-chunk scan (NT
   steps). A static-shape graph has to unroll both, and netherite's camera
   already showed that an unrolled sequential chain costs about one kernel per
   step. At C=64 and NT=64 that predicts ~130 kernels before any real work.

2. DOES FUSION SURVIVE THE UNROLL. The camera's elementwise chain fused into
   one kernel, so fusion itself is not the question. The question is whether
   the C^2 and C^2 x K intermediates -- the big ones -- get materialised.

3. WHERE THE TIME GOES. Parse DEBUG=2 by kernel shape and attribute time to
   the algorithm step. `E_` is elementwise, `r_` is a reduce (so a matmul or a
   gather), `c_` would be a custom kernel.

This is a cost model, not a competitive benchmark: there is no NVIDIA GPU here
and no Triton, so the hand-optimized reference is the published analysis in
kda/NOTES.md, not a measured number. What we CAN measure is how many
intermediates the port creates and what that implies.
"""
import os
import re
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "probes"))

from tinygrad import Device
import ref
import tiny_kda
import test_ref as T
from bench import bench_ms


def kernel_stats(code, iters=2):
    """(total GPU us, kernel count, per-shape breakdown) from DEBUG=2."""
    env = dict(os.environ, DEBUG="2")
    p = subprocess.run([sys.executable, "-c", code], capture_output=True,
                       text=True, env=env)
    txt = re.sub(r"\x1b\[[0-9;]*m", "", p.stdout + "\n" + p.stderr)
    total, n = 0.0, 0
    by = {}
    for line in txt.splitlines():
        if "***" not in line or " copy " in line:
            continue
        m = re.search(r"([\d.]+)(us|ms)/", line)
        k = re.search(r"\*\*\*\s+\S+\s+\d+\s+(\S+)", line)
        if not m:
            continue
        # The first number is THIS kernel's time, the second is cumulative
        # device time. Using the first is right; just be sure to divide both
        # totals and the per-shape sums by the iteration count consistently.
        v = float(m.group(1)) * (1000.0 if m.group(2) == "ms" else 1.0)
        total += v
        n += 1
        name = (k.group(1) if k else "?").split("_")[0]
        e = by.setdefault(name, [0, 0.0])
        e[0] += 1
        e[1] += v
    for e in by.values():
        e[0] /= iters
        e[1] /= iters
    return total / iters, n / iters, by


def _timeit(fn, reps=2, batches=2, warmup=1):
    """Plain wall clock. `bench_ms` is for TinyGrad graphs (it calls
    .realize(), which a numpy function does not have), and the float64 numpy
    reference is not a graph at all -- and far too slow to run 20x."""
    for _ in range(warmup):
        fn()
    best = float("inf")
    for _ in range(batches):
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        best = min(best, (time.perf_counter() - t0) / reps)
    return best * 1e3


def main():
    B, Tt, H, HV, K, V = 2, 512, 4, 4, 64, 64
    C = 64
    q, k, v, g, beta = T.rnd(B, Tt, H, HV, K, V, seed=0, l2n=True)  # see note below
    # q and k MUST be L2-normalised, which is what the real KDA layer does.
    # With raw Gaussians, k^T k ~ K = 64, the delta rule matrix
    # (I - beta k k^T) has eigenvalues near 1-64*beta, and the state diverges:
    # |S| reaches 1e140 in float64 and overflows to inf in float32. The first
    # version of this benchmark used l2n=False and was measuring a graph that
    # produced NaN. The graph SHAPE, and so the kernel count, is unaffected.
    NT = Tt // C
    print("=" * 78)
    print(f"KDA cost model   device={Device.DEFAULT}")
    print(f"  B={B} T={Tt} H={H} HV={HV} K={K} V={V}  chunk C={C}  NT={NT}")
    print("=" * 78)

    # ---- intermediate sizes the port must materialise ----
    print("\nintermediates the port builds (bf16 = 2 bytes, as the kernels use):")
    items = [
        ("g_cum   [B,NT,HV,C,K]", B * NT * HV * C * K),
        ("Dq_all  [B,NT,HV,C,C,K]", B * NT * HV * C * C * K),
        ("Dl      [B,NT,HV,C,C,K]", B * NT * HV * C * C * K),
        ("Z/Btri  [B,NT,HV,C,C]", B * NT * HV * C * C),
        ("M       [B,NT,HV,C,C]", B * NT * HV * C * C),
        ("Aqk     [B,NT,HV,C,C]", B * NT * HV * C * C),
        ("w       [B,NT,HV,C,K]", B * NT * HV * C * K),
        ("u       [B,NT,HV,C,V]", B * NT * HV * C * V),
        ("v_new   [B,NT,HV,C,V]", B * NT * HV * C * V),
        ("o       [B,T,HV,V]", B * Tt * HV * V),
        ("S (state, live) [B,HV,K,V]", B * HV * K * V),
    ]
    print(f"  {'tensor':<28} {'elements':>14} {'bf16 MB':>10}")
    for name, n in items:
        print(f"  {name:<28} {n:>14,} {n*2/1e6:10.1f}")

    # ---- sequential depth: the structural cost ----
    print(f"\nsequential depth (unrolled, no TinyGrad loop primitive):")
    print(f"  UT transform   C  = {C} steps   (forward substitution)")
    print(f"  inter-chunk    NT = {NT} steps  (carrying [B,HV,K,V] = {B*HV*K*V:,} floats)")
    print(f"  total                    {C + NT} steps, each becoming graph ops")
    print(f"  the camera probe found 1 extra kernel per sequential step, so the")
    print(f"  prediction is ~{C + NT} kernels before the real matmuls.")

    # ---- measured ----
    code = f"""
import sys; sys.path.insert(0, {os.path.dirname(os.path.abspath(__file__))!r})
import numpy as np
from tiny_kda import kda_chunk
import test_ref as T
q,k,v,g,beta = T.rnd({B},{Tt},{H},{HV},{K},{V},seed=0,l2n=True)
for _ in range({2}):
    kda_chunk(q,k,v,g,beta,chunk_size={C})
"""
    gpu, nk, by = kernel_stats(code)
    # The device is whatever ran. It is printed, never assumed: this file has
    # been run on Metal and on CUDA, and a label that says "Metal" in the
    # second case is a measurement attributed to the wrong machine.
    print(f"\nmeasured ({Device.DEFAULT}, B={B} T={Tt} K={K} C={C}):")
    print(f"  kernels per call : {nk:.0f}")
    print(f"  GPU time         : {gpu:,.0f} us  ({gpu/1000:.1f} ms)")
    for name, (cnt, us) in sorted(by.items(), key=lambda x: -x[1][1]):
        print(f"    {name:<4} {cnt:5.0f} kernels  {us:10,.0f} us  {us/gpu*100:5.1f}%")
    # ---- wall time, and the python tax ----
    print(f"\nwall time:")
    wall_ref = _timeit(lambda: ref.kda_chunk(q, k, v, g, beta, chunk_size=C))
    print(f"  numpy float64 reference : {wall_ref:9.1f} ms")
    # tiny_kda.kda_chunk ends in .numpy(), which realizes, so a plain wall
    # clock is the honest number here -- it captures graph build + launch +
    # execute, which is exactly the split we want to report.
    wall_tg = _timeit(lambda: tiny_kda.kda_chunk(q, k, v, g, beta, chunk_size=C))
    print(f"  tinygrad float32        : {wall_tg:9.1f} ms")
    print(f"  ratio                   : {wall_tg/wall_ref:9.2f}x")
    print(f"  of which GPU             : {gpu/1000:9.1f} ms "
          f"({gpu/1000/wall_tg*100:.0f}%)")
    print(f"  of which python/build   : {wall_tg - gpu/1000:9.1f} ms "
          f"({(wall_tg-gpu/1000)/wall_tg*100:.0f}%)")


if __name__ == "__main__":
    main()
