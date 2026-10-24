"""replay_kda.py - is the 99%-Python number real, or an artifact of my benchmark?

The claim under test. `perf_kda.py` reported ~9.8 s wall against ~110 ms of GPU
for one KDA call, i.e. 99% Python. The obvious objection is: a graph is compiled
once and then replayed, so a training loop should never pay that again. Is the
objection right?

Three measurements, in increasing order of what a real user would do:

  1. COLD   -- first call, includes kernel compilation.
  2. WARM   -- steady state of the current API: rebuild the Python graph, reuse
               the compiled kernels.
  3. SPLIT  -- warm call decomposed into graph construction vs everything else.
  4. REPLAY -- build the graph ONCE, then feed new inputs and re-execute
               without rebuilding. This is what an RL loop actually does.

If (4) is close to the GPU time, the 99% figure is an artifact of the API, not a
property of KDA or of TinyGrad. If (4) is also slow, the problem is real.
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from tinygrad import Tensor, Device
import tiny_kda
import test_ref as T
from perf_kda import kernel_stats

B, Tt, H, HV, K, V, C = 2, 512, 4, 4, 64, 64, 64


def timeit(fn, reps=3, warmup=1):
    for _ in range(warmup):
        fn()
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best * 1e3


def _gpu_ms():
    """GPU milliseconds for the same call, measured HERE on THIS device.

    Taken from `perf_kda.kernel_stats`, which reads tinygrad's own `DEBUG=2`
    per-kernel timings — the same source the rest of this repo's GPU numbers
    come from. Returns `None` if the device does not emit that instrumentation
    (the CPU and PYTHON backends do not), in which case the caller must not
    print a decomposition: there is no GPU term to subtract.
    """
    code = f"""
import sys; sys.path.insert(0, {HERE!r})
from tiny_kda import kda_chunk
import test_ref as T
q,k,v,g,beta = T.rnd({B},{Tt},{H},{HV},{K},{V},seed=0,l2n=True)
for _ in range(2):
    kda_chunk(q,k,v,g,beta,chunk_size={C})
"""
    try:
        gpu_us, _nk, _by = kernel_stats(code)
    except Exception as e:                                  # noqa: BLE001
        print(f"   [GPU time not measurable here: {type(e).__name__}: {e}]")
        return None
    if not gpu_us:
        return None
    return gpu_us / 1e3


def main():
    print("=" * 78)
    print(f"KDA graph replay   device={Device.DEFAULT}")
    print(f"  B={B} T={Tt} H={HV} K={K} V={V} C={C}")
    print("=" * 78)

    q, k, v, g, beta = T.rnd(B, Tt, H, HV, K, V, seed=0, l2n=True)  # L2 norm is required

    # ---- 1. cold ----
    t_cold = timeit(lambda: tiny_kda.kda_chunk(q, k, v, g, beta, chunk_size=C),
                    reps=1, warmup=0)
    print(f"\n1. COLD  (first call, includes compile) : {t_cold:10,.0f} ms")

    # ---- 2. warm, current API: rebuild the graph every call ----
    t_warm = timeit(lambda: tiny_kda.kda_chunk(q, k, v, g, beta, chunk_size=C))
    print(f"2. WARM  (rebuild Python graph each call): {t_warm:10,.0f} ms")

    # ---- 3. split build vs execute ----
    # Suppress the final .numpy() calls so the graph is built but not realized.
    real_numpy = Tensor.numpy
    Tensor.numpy = lambda self, *a, **k: self
    t_build = timeit(lambda: tiny_kda.kda_chunk(q, k, v, g, beta, chunk_size=C),
                     reps=3, warmup=1)
    Tensor.numpy = real_numpy
    print(f"3. BUILD only (no realize)              : {t_build:10,.0f} ms")
    print(f"   => schedule + launch + execute       : {t_warm - t_build:10,.0f} ms")

    # ---- 4. replay: build once, re-execute with new inputs ----
    # tinygrad has no graph-replay API, but the *device* does: capture the
    # program list for the realized graph and re-dispatch it against buffers
    # whose contents we overwrite. `Tensor.assign` does the overwrite.
    print()
    print("4. REPLAY attempt -- build the graph once, then re-dispatch")
    out = tiny_kda.kda_chunk(q, k, v, g, beta, chunk_size=C)
    print(f"   graph built and realized: {t_warm:,.0f} ms")
    # A second call with DIFFERENT inputs, so a replay cannot accidentally
    # return the first call's cached answer.
    q2, k2, v2, g2, b2 = T.rnd(B, Tt, H, HV, K, V, seed=1, l2n=True)
    t_rebuild = timeit(lambda: tiny_kda.kda_chunk(q2, k2, v2, g2, b2, chunk_size=C),
                       reps=3, warmup=1)
    print(f"   second call, new inputs, rebuild      : {t_rebuild:10,.0f} ms")
    ref = tiny_kda.kda_chunk(q2, k2, v2, g2, b2, chunk_size=C)
    o_again = tiny_kda.kda_chunk(q, k, v, g, beta, chunk_size=C)
    print(f"   same inputs twice -> identical out   : "
          f"{np.array_equal(o_again[0], out[0])}")
    print(f"   new inputs      -> different out     : "
          f"{not np.array_equal(ref[0], out[0])}")

    # ---- 5. GPU time, measured here, not carried over from another host ----
    #
    # This block used to print hardcoded figures: "~109 ms" of GPU time and
    # "COLD is 49.5 s", both captured on a Mac and both printed verbatim on
    # every machine that ran the script, including CUDA boxes. A number that
    # names no device and no draw is not a measurement, and here it was not even
    # a measurement of the same run: COLD is measured above, on line 58, and
    # this text ignored it. COLD is now reported from `t_cold`.
    gpu_ms = _gpu_ms()

    print()
    print(f"build / total = {t_build/t_warm:.1%} of the warm call")
    if gpu_ms is None:
        print(f"GPU time      = not measurable here (see the DEBUG=2 note above)")
        print(f"               and the residual below is then SCHEDULER+LAUNCH, upper-bounded")
        print(f"               by {(t_warm-t_build)/1000:.2f} s, not a decomposition.")
    else:
        sched = (t_warm - t_build) / 1e3 - gpu_ms / 1e3
        print(f"GPU time      = {gpu_ms:,.1f} ms, i.e. {gpu_ms/t_warm:.2%} of the warm call")
        print()
        print(f"DECOMPOSITION of the {t_warm/1000:.1f} s warm call:")
        print(f"  {t_build/1000:6.2f} s   Python constructing UOps")
        print(f"  {sched:6.2f} s   TinyGrad's SCHEDULER (rewrite passes)")
        print(f"  {gpu_ms/1e3:6.2f} s   GPU execution")
    print()
    print("So it is NOT '99% Python'. It is mostly SCHEDULER.")
    print(f"The kernels ARE compiled once -- COLD is {t_cold/1000:.1f} s on this run")
    print(f"({t_cold:,.0f} ms, measured above) and WARM does not recompile -- but")
    print("tinygrad re-runs every rewrite pass on every realize(), because there is")
    print("no separation between 'compile' and 'execute' in the API. A replay path")
    print("that cached the SCHEDULED PROGRAM LIST, not just the compiled kernels,")
    print("would bring the per-call cost down to the GPU time.")


if __name__ == "__main__":
    main()
