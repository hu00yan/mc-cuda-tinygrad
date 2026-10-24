"""fp32_div_row.py - what does the "/" row of probes/fp32_correctness.py
actually compute?

Reading the loop:

    for name, a in regimes.items():
      for op, fn in (("+",...), ("-",...), ("*",...), ("/",...)):
        b = regimes[name] if op in "*/" else a

`regimes[name]` IS `a` -- it is the same dict entry. So for op="/" the probe
computes a/a, not a/b. It never divides by an independent divisor.

Why that matters: `a*(1/a)` is much closer to 1 than `a*(1/b)` is to the true
quotient, so `a/a` is a corner-pinned test and understates the error rate of
general division. Two probes, same backend, same session, disagreeing:

  out/kaggle/20260928-014248-gpu/logs/fp32_correctness.log
      /  322,233 (15.365%)   for "mixed +-2"      <- a/a
  out/kaggle/20260928-014248-gpu/logs/divide_precision.log
      CUDA : 54,270 wrong (27.13%)                 <- a/b, one divisor law

and the 15.365% here is bit-for-bit the same number divide_precision's step-1
a/a row reports for CUDA (15.37%), which is the check that identifies it. Note
that 27.13% is ONE divisor law's worth at ONE n -- it is not a rate for `/`,
and the rate is a function of the pair (a, b). The citable band, its divisor
law, and its sampling interval are in `derivation/divide.py`.

This script re-derives both rows on whatever device is available. Every number
it prints is measured here; the two log figures above are quoted from the logs
and are labelled as such.
"""
import numpy as np
from tinygrad import Tensor, Device

N = 1 << 21
rng = np.random.default_rng(0)


def regime(nm):
    if nm == "near 1.0 (LUT regime)":
        return rng.random(N).astype(np.float32) * np.float32(0.02) + np.float32(0.99)
    if nm == "mixed +-2":
        return rng.random(N).astype(np.float32) * np.float32(4) - np.float32(2)
    return rng.random(N).astype(np.float32) * np.float32(2000) - np.float32(1000)


def correctly_rounded_div(a, b):
    return (a.astype(np.float64) / b.astype(np.float64)).astype(np.float32)


def device_div(a, b):
    return (Tensor(a) / Tensor(b)).numpy()


# the exact call sequence fp32_correctness.py makes, so the rng stream matches
REGIMES = {nm: regime(nm) for nm in
           ("near 1.0 (LUT regime)", "mixed +-2", "wide +-1e3")}

print(f"device = {Device.DEFAULT}")
print()
print(f"{'regime':<22} {'b := a  (a/a)':>18} {'b ~ U[0.01, 2.01]':>19} "
      f"{'understated by':>15}")
for name, a in REGIMES.items():
    ref_aa = correctly_rounded_div(a, a)                 # what the probe measured
    got_aa = device_div(a, a)
    bad_aa = int((got_aa != ref_aa).sum()) / N

    b = rng.random(N).astype(np.float32) * np.float32(2) + np.float32(0.01)
    ref_ab = correctly_rounded_div(a, b)
    got_ab = device_div(a, b)
    bad_ab = int((got_ab != ref_ab).sum()) / N
    print(f"{name:<22} {bad_aa:17.3%} {bad_ab:18.3%} "
          f"{bad_ab/max(bad_aa,1e-9):14.2f}x")

print()
print("  The left column is a DEGENERATE test, not a rate for `/`: its exact")
print("  quotient is 1.0, which is representable, so the second rounding starts")
print("  pinned at a corner of an asymmetric rounding cell and can only fail on")
print("  the lower quarter of the reciprocal's error range. The right column is")
print("  a rate for ONE divisor law, and the rate is a function of the pair")
print("  (a, b) -- so it is published as a band with the law attached, never as a")
print("  scalar. See derivation/divide.py:published_rate_table and")
print("  derivation/divide.py:sampling_interval.")
print()
print("sanity: is np.float32(a/a) exactly 1.0 everywhere? numpy says",
      bool((REGIMES['mixed +-2'] / REGIMES['mixed +-2'] ==
            np.float32(1.0)).all()), "(numpy reference column, 0.000%)")
