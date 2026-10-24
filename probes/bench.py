"""bench.py - one reliable timing helper for the probes.

Two traps this avoids:

1. tinygrad's `Tensor.realize()` on an ALREADY-realized graph is a no-op, so
   timing a pre-built `expr` measures nothing (it reads as ~3ns). Every timed
   unit here is a *callable* that rebuilds the graph, so each iteration really
   re-plans and re-launches.
2. The first call includes kernel compilation. We warm up, then take best-of-K
   batches to reject scheduler noise.
"""
import time


def bench_ms(build, reps=20, batches=5, warmup=3):
  """Best per-call wall time in ms. `build()` must return an unrealized Tensor."""
  for _ in range(warmup):
    build().realize()
  best = float("inf")
  for _ in range(batches):
    t0 = time.perf_counter()
    for _ in range(reps):
      build().realize()
    best = min(best, (time.perf_counter() - t0) / reps)
  return best * 1e3


def bench_us(build, **kw):
  return bench_ms(build, **kw) * 1e3
