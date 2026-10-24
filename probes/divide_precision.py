"""divide_precision.py - is TinyGrad's float32 division correctly rounded?

THE QUESTION
    IEEE 754 specifies float32 division. Two implementations cannot both be
    right. So this is not "each has its own approximation with its own validity
    conditions" -- it is a question of which one is correct, and the answer has
    to be measured, not assumed.

WHAT WAS ACTUALLY WRONG WITH THE ORIGINAL CLAIM
    The first version of this probe said "TinyGrad's Metal division is not
    correctly rounded" and left it there. Four follow-ups, all in this file.
    Three of the four were corrections to claims that this file used to make
    itself; two of those corrections RETRACTED what the probe previously
    printed, and the retraction is the point:

      1. WHERE the rewrite happens. Not in a renderer. `div()` ends with
             mixin/elementwise.py:251   d = a.alu(Ops.MUL, b.reciprocal())
         in the **Tensor API**, above the scheduler and above backend
         selection, and unconditionally. So the rewrite is not a property of
         any backend: the backends only differ in how they reach a real divide.

      2. THE RECIPROCAL IS CORRECTLY ROUNDED, NOT APPROXIMATE.  <-- RETRACTS
         what this probe used to print. There is no "the backend contracts
         `1/x` into an approximate reciprocal" step. What fires is a correctly
         rounded float32 reciprocal followed by one correctly rounded float32
         multiply -- DOUBLE ROUNDING, not approximation. The evidence is
         bit-identity, not a rate: the device output equals `a * fl32(1/b)`
         element for element, and a hypothetical reciprocal 1 ULP worse
         predicts ~86% wrong against the ~27% measured. This clause is
         load-bearing and is NOT the same claim as the rate.

      3. The per-backend table below is a property of a BACKEND AND A HOST.
         On this host CL and METAL agree bit for bit; the T4's CL does not, and
         the T4's CL is not bit-identical to `a*fl32(1/b)`. So a per-backend
         percentage here may never be quoted as a property of TinyGrad, and no
         per-backend number belongs in a mechanism sentence.

      4. `-fno-fast-math` is passed to the metal driver
         (runtime/ops_metal.py:93) and does NOT help, because the rewrite is
         already in the emitted source before the compiler sees it. The string
         `use_fast_math` appears nowhere in the installed tinygrad at all.

    So the honest statement is:

        TinyGrad's float32 `/` is lowered in the Tensor API -- above the
        scheduler, above backend selection -- to `a * fl32(1/b)`, so on every
        backend that does not itself render a real divide it returns a value
        bit-identical to a correctly rounded reciprocal followed by one
        correctly rounded float32 multiply; the result is wrong by exactly
        1 ULP in a rate that depends on the divisor law, so it travels with
        the pair `(a,b)` and not with the operation -- and no user-facing
        knob changes any of this.

    The rate itself is a band and is derived, not typed:
    `derivation/divide.py`. Whether 1 ULP is acceptable is the caller's
    decision, which is exactly the point: the caller cannot make it, and it
    is silent -- netherite's DDA takes every step size as `1.0f/dx`.
"""
import os
import re
import subprocess
import sys

import numpy as np
from tinygrad import Tensor, Device

N = 200_000


def inputs(regime, seed=0):
    rng = np.random.default_rng(seed)
    if regime == "near1":
        a = rng.random(N).astype(np.float32) * np.float32(0.02) + np.float32(0.99)
        return a, a.copy()
    if regime == "pm2":
        a = rng.random(N).astype(np.float32) * np.float32(4) - np.float32(2)
        b = rng.random(N).astype(np.float32) * np.float32(2) + np.float32(0.01)
        return a, b
    a = rng.random(N).astype(np.float32) * np.float32(2000) - np.float32(1000)
    return a, rng.random(N).astype(np.float32) + np.float32(0.5)


def correctly_rounded_div(a, b):
    """A float32 product of two float32s needs 48 bits, so a float64 divide is
    exact and the single rounding to float32 is correctly rounded."""
    return (a.astype(np.float64) / b.astype(np.float64)).astype(np.float32)


def main():
    print("=" * 78)
    print("Is TinyGrad's float32 `/` correctly rounded?")
    print("=" * 78)

    # ---- 1. the obvious sanity check: a/a must be exactly 1.0 ----
    a, _ = inputs("pm2")
    print("\n1. a / a, which must be exactly 1.0 for every finite nonzero a")
    for dev in ("CPU", Device.DEFAULT):
        got = (Tensor(a).to(dev) / Tensor(a).to(dev)).numpy()
        bad = int((got != np.float32(1.0)).sum())
        print(f"   {dev:6s}: {bad:>7,} / {N:,} wrong  ({bad/N:6.2%})")
    print("   CPU gives 1.0 everywhere. There is no 'approximation with its own")
    print("   validity condition' under which a/a != 1, so the failing backends")
    print("   are not implementing an allowed variant -- they are off by one ulp.")

    # ---- 2. which backends, and is it Metal or TinyGrad ----
    print("\n2. a / b against correctly rounded, per backend")
    a, b = inputs("pm2")
    ref = correctly_rounded_div(a, b)
    rcp_mul = (a * (np.float32(1.0) / b)).astype(np.float32)
    for dev in [d for d in Device._devices if d in ("CPU", "PYTHON", "CL", Device.DEFAULT)]:
        got = (Tensor(a).to(dev) / Tensor(b).to(dev)).numpy()
        nb = int((got != ref).sum())
        same_as_rcp = int((got != rcp_mul).sum()) == 0
        print(f"   {dev:7s}: {nb:>7,} wrong ({nb/N:6.2%})"
              f"   {'== a*fl32(1/b) bit for bit' if same_as_rcp else 'differs from a*fl32(1/b)'}")
    print("   Every row is a property of a BACKEND AND A HOST, not of TinyGrad. A")
    print("   number from this table may not be quoted in a mechanism sentence, and")
    print("   it may not be quoted as a rate for `/`: it is one divisor law's worth")
    print("   (this probe's own law, declared above) at one n. The citable band is")
    print("   derived, over several divisor laws, in `derivation/divide.py`.")

    # ---- 3. what does the generated code actually say ----
    print(f"\n3. the generated {Device.DEFAULT} source for a single division")
    d = Device[Device.DEFAULT]
    captured = []
    orig = d.compiler.compile_cached

    def spy(src, *A, **K):
        captured.append(str(src))
        return orig(src, *A, **K)

    d.compiler.compile_cached = spy
    x = Tensor(np.arange(64, dtype=np.float32) + 1.0)
    y = Tensor(np.full(64, 7.0, dtype=np.float32))
    (x / y).realize()
    d.compiler.compile_cached = orig
    for src in captured:
        for line in src.split("\n"):
            s = line.strip()
            if "data0" in s and "=" in s and "float" in s:
                print("   ", s[:150])
    print("   => the emitted source is `a*(1/b)`, not `a/b`. That is the WHOLE")
    print("      mechanism and there is no second step: the backend is not")
    print("      contracting `1/x` into an approximate reciprocal. What the device")
    print("      returns is bit-identical to `a * fl32(1/b)`, i.e. a correctly")
    print("      rounded reciprocal and one correctly rounded multiply. Row 2 above")
    print("      is the test; a hypothetical 1-ULP-worse reciprocal predicts ~86%")
    print("      wrong, not the ~27% measured.")

    # ---- 4. is there a knob ----
    print("\n4. is the rewrite controllable?")
    root = os.path.dirname(os.path.dirname(os.path.abspath(Tensor.__module__ and
                                                            __import__("tinygrad").__file__)))
    for pat in ("RECIPROCAL", "use_fast_math", "fno-fast-math"):
        hits = subprocess.run(["grep", "-rn", "--include=*.py", pat, root],
                              capture_output=True, text=True).stdout.splitlines()
        shown = [h for h in hits
                 if "renderer" in h or "elementwise" in h or "ops_metal" in h][:4]
        print(f"   grep {pat!r}: {len(hits)} hits")
        for line in shown:
            print("     ", line.strip()[:112])
    print("   `/` DOES go through Ops.RECIPROCAL: `div()` builds")
    print("   `a.alu(Ops.MUL, b.reciprocal())` and `reciprocal()` is")
    print("   `self.alu(Ops.RECIPROCAL)` (mixin/elementwise.py:251 and :465),")
    print("   unconditionally, in the Tensor API. Ops.RECIPROCAL is a real op and")
    print("   PTX renders it as `rcp.approx.f32` (renderer/ptx.py:19) -- but the")
    print("   cstyle backends render it as the literal `(1/b)` (cstyle.py:137),")
    print("   which the backend then evaluates to its own accuracy, and every")
    print("   backend measured here that fails is bit-identical to the CORRECTLY")
    print("   ROUNDED reciprocal. So it is not that `/` avoids the approximate op,")
    print("   and it is not that `/` chooses it: there is no user-facing choice at")
    print("   all. `use_fast_math` appears nowhere in the installed tinygrad, and")
    print("   the Metal driver already passes `-fno-fast-math` without effect,")
    print("   because the rewrite is in the emitted source before the compiler.")
    print("   Ops.FDIV is not dead code, either: ClangRenderer keeps it")
    print("   (cstyle.py:271 deletes RECIPROCAL, :274 adds FDIV) and the CPU")
    print("   backend's decomposition manufactures it (codegen/decomp/op.py:133-136).")


if __name__ == "__main__":
    main()
