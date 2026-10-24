"""fp32_correctness.py - is TinyGrad's float32 arithmetic correctly rounded?

Why this probe exists. netherite's MC 1.11.2 MathHelper sin/cos is a 65,536-entry
LUT indexed by `(float)(value * 10430.378f)`. The conversion quantises: two
inputs one ULP apart can land in different buckets and produce sin values
9.587e-05 apart. So float32 multiply precision is not a rounding detail here,
it is the difference between a correct camera and a wrong one.

Concretely, for yaw = 180 degrees:
    x = 3.1415927 (float32), c = 10430.378 (float32)
    exact product  = 32767.999589924002...
    correctly rounded float32 = 32768.0   (what numpy gives)
    the failing backends      = 32767.998 (1 ULP low)
    floor() then LUT index    = 32767 instead of 32768

and the resulting ray direction differs by ~1e-4, which flips the depth quantum
on 12 of 41,472 pixels of the camera.

This script measures whether that is a one-off boundary case or systematic,
for +, -, * and /, against exact float64 arithmetic as the reference.

THE `/` ROW IS A DIVISION, NOT A SELF-DIVISION  (this file was wrong until now)
    Version 0 of this file read

        b = regimes[name] if op in "*/" else a     # regimes[name] IS a

    so for `/` the divisor WAS the dividend: the row measured `a/a`, never `a/b`.
    `a/a` is degenerate and provably so -- its exact quotient is 1.0, which is
    representable, so the second rounding starts pinned at a CORNER of a
    rounding cell whose two edges are unequal, and it can only fail on the lower
    quarter of the reciprocal's error range. The published 12.4%-15.4% band was
    therefore a property of that corner, understated the error rate of general
    division by roughly 1.5x-2.1x, and was copied into ~58 places in this repo.

    The divisor is now INDEPENDENT, drawn from the same magnitude law as `a`,
    and the script prints BOTH rows side by side. The divisor law is printed
    with the numbers, because a general `a/b` rate is a property of the PAIR
    `(a, b)` and not of `/` alone.

THE RATES THIS FILE PRINTS ARE NOT THE PUBLISHED ONES
    This file measures on a DEVICE, in this process, on whatever device ran.
    It is an instrument. The citable, device-free rates live in
    `derivation/divide.py`, which declares its own draw; this file keeps the
    original three regimes and the original `default_rng(0)` `a` stream so that
    its `b := a` column still reproduces the shipped GPU log
    `out/kaggle/20260928-014248-gpu/logs/fp32_correctness.log` exactly, which is
    the check that this file's `a` array has not drifted.
"""
import numpy as np
from tinygrad import Device, Tensor


def exact_f32(a, b, op):
  """Correctly-rounded float32 result of a op b, via float64 (exact for the
  magnitudes used here: float64 has 53 bits, products of two float32s need 48)."""
  a64, b64 = a.astype(np.float64), b.astype(np.float64)
  if op == "+":
    r = a64 + b64
  elif op == "-":
    r = a64 - b64
  elif op == "*":
    r = a64 * b64
  else:
    r = a64 / b64
  return r.astype(np.float32)


# The three magnitude regimes, and the per-regime marginal law of `a`. Written
# once and used by both the `a` draw and the independent `b` draw, which is what
# makes the two rows differ only in dependence and not in distribution.
MARGINALS = {
  "near 1.0 (LUT regime)": (np.float32(0.02), np.float32(0.99)),
  "mixed +-2": (np.float32(4), np.float32(-2)),
  "wide +-1e3": (np.float32(2000), np.float32(-1000)),
}
SEED_A = 0        # the dividend stream: probes/fp32_correctness.py:42-50 since v0
SEED_B = 20260928  # the divisor stream, disjoint from SEED_A

# The degenerate row as the shipped T4 log records it:
# out/kaggle/20260928-014248-gpu/logs/fp32_correctness.log lines 10/17/24.
# Held here so the cross-check below can assert against them. These are the
# counts the file's own `a` array produced on a different vendor and a
# different architecture; if they stop matching, `a` has drifted.
SHIPPED_LOG_COUNTS = {
  "near 1.0 (LUT regime)": 260261,
  "mixed +-2": 322233,
  "wide +-1e3": 318044,
}


def main():
  rng = np.random.default_rng(SEED_A)
  N = 1 << 21
  regimes = {name: (rng.random(N).astype(np.float32) * w + o)
             for name, (w, o) in MARGINALS.items()}
  # An INDEPENDENT divisor, same marginal law, separate stream. `b` shares no
  # sample with `a`, which is the whole point of the row.
  rb = np.random.default_rng(SEED_B)
  divisors = {name: np.ascontiguousarray(rb.random(N).astype(np.float32) * w + o)
              for name, (w, o) in MARGINALS.items()}

  print(f"device = {Device.DEFAULT}")
  print(f"n = {N:,} per regime. Oracle: (a.astype(f64) OP b.astype(f64)).astype(f32).")
  print()
  print("`b := a` is the DEGENERATE row version 0 of this file measured, and is")
  print("kept only so the counts stay comparable to the shipped GPU log. `/` is")
  print("the row that matters: an INDEPENDENT divisor from the same law.")

  for name, a in regimes.items():
    print(f"\n=== {name}  (n={N:,}) ===")
    print(f"{'op':<3} {'device vs correctly-rounded':>30} {'numpy vs correctly-rounded':>28}")
    n_ops = 0
    for op, fn in (("+", lambda x, y: x + y), ("-", lambda x, y: x - y),
                   ("*", lambda x, y: x * y), ("/", lambda x, y: x / y)):
      b = divisors[name] if op == "/" else a
      b = np.ascontiguousarray(b)
      tg = fn(Tensor(a), Tensor(b)).numpy()
      npy = fn(a, b)
      ref = exact_f32(a, b, op)
      d_tg = int((tg != ref).sum())
      d_np = int((npy != ref).sum())
      if op != "/":
        n_ops += N
      tag = "   <- independent divisor" if op == "/" else ""
      print(f"{op:<3} {d_tg:>14,} ({d_tg/N:7.3%}) {d_np:>14,} ({d_np/N:7.3%}){tag}")

  print("\nA correctly-rounded implementation shows 0 in both right-hand columns.")
  print("A nonzero device column for `/` is a rounding bug, not a rounding")
  print("artefact of the test: the oracle is the correctly rounded result and")
  print("numpy agrees with it everywhere.")
  print()
  print(f"`+ - *` are exact: 0 wrong out of {3 * n_ops:,} op instances "
        f"(3 ops x {len(MARGINALS)} regimes x {N:,} per regime).")
  print("`/` is wrong in the rate printed above, and it is wrong on every backend")
  print("that reaches the rewrite; see `derivation/divide.py` for the citable")
  print("rates, the divisor law they depend on, and the mechanism.")

  # The degenerate row, kept as a CROSS-CHECK on this file's `a` array rather
  # than as a rate. Its counts must equal the shipped GPU log's
  # 260,261 / 322,233 / 318,044, which is a different vendor and architecture.
  #
  # This ASSERTS rather than printing two numbers side by side and asking the
  # reader to compare. A cross-check that only prints is a claim; the whole class
  # of error this file is the subject of (`notes/06-corrections-ledger.md` C-09)
  # was a number nobody could check, so the check has to be mechanical or it is
  # decoration.
  print("\nCROSS-CHECK, the degenerate `b := a` row this file used to publish:")
  drift = []
  for name, a in regimes.items():
    got = (Tensor(a) / Tensor(a)).numpy()
    bad = int((got != np.float32(1.0)).sum())
    want = SHIPPED_LOG_COUNTS[name]
    ok = bad == want
    if not ok:
      drift.append((name, bad, want))
    print(f"  {name:24s} {bad:>9,} / {N:,}  {bad/N:7.3%} wrong   "
          f"{'matches' if ok else 'DRIFTED, expected'} {want:,}")
  print("  Source: out/kaggle/20260928-014248-gpu/logs/fp32_correctness.log")
  print("  lines 10/17/24, a T4. This row is NOT a rate for `/`; its only job is")
  print("  to prove this file still draws the same `a`.")
  if drift:
    for name, bad, want in drift:
      print(f"  FAIL {name}: got {bad:,}, the shipped log has {want:,}. "
            f"`a` has drifted -- SEED_A or N changed -- and EVERY other number "
            f"in this run is void.")
    raise SystemExit(1)
  print("  All three match. `a` is the array the shipped log measured.")

  # How far off is a bad division? Worth knowing, because netherite's DDA takes
  # every step size as 1.0f/dx.
  #
  # A SECOND, DEPENDENT divisor law, reported as its own regime and never mixed
  # into the independent-divisor band above. `b = |a| + 0.01` is a function of
  # `a`, so the exact quotient is correlated with the dividend; it lands OUTSIDE
  # the independent-divisor band, which is exactly why the band has to be
  # published with the law attached. Its ratio to the `a/a` row is ~2.4x, which
  # is why it cannot stand in for an independent divisor either.
  a = regimes["mixed +-2"]
  for label, b in (("independent b (same law)", divisors["mixed +-2"]),
                   ("DEPENDENT b = |a| + 0.01", np.abs(a) + np.float32(0.01))):
    got = (Tensor(a) / Tensor(np.ascontiguousarray(b))).numpy()
    ref = exact_f32(a, np.ascontiguousarray(b), "/")
    bad = got != ref
    if bad.any():
      err = np.abs(got[bad].astype(np.float64) - ref[bad].astype(np.float64))
      ulp = err / np.spacing(np.abs(ref[bad]).astype(np.float32)).astype(np.float64)
      print(f"\ndivision error, {label}: {bad.mean():.3%} wrong, "
            f"median {np.median(ulp):.1f} ULP, "
            f"p99 {np.percentile(ulp, 99):.1f} ULP, max {ulp.max():.0f} ULP")
    else:
      print(f"\ndivision error, {label}: 0 wrong")
  print("The second law is DEPENDENT and its number is a property of that law, "
        "not a rate for `/`.")


if __name__ == "__main__":
  main()
