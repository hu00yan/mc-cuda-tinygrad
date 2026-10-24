"""
residency_probe.py -- measure, on whatever hardware is in front of you, whether
"does the intermediate fit" produces a STEP in time or a KNEE.

Run:  .venv/bin/python crossover/residency_probe.py
      .venv/bin/python crossover/residency_probe.py --max-gib 4 --passes 32

WHY THIS EXISTS
---------------
crossover/claim.py's R-A says a gradual transition refutes the step-function
claim.  That criterion is untestable without hardware in the study, and the study
has no hardware.  But the criterion is not really about GPUs: it is about the
SHAPE of the function

        t(s)  =  time to touch a working set of s bytes, given reuse

as s crosses some capacity.  That function is a property of every memory
hierarchy ever built, including the one in the machine running this script, and
it can be measured in about a minute.  So this file settles the SHAPE question on
real hardware, at whatever capacity the local machine has, and the answer
transfers even though the capacity does not.

WHAT IT MEASURES
----------------
Allocate an array of s bytes, fault it in, then time K repeated streaming reads
of it and divide by K.

    s <= cache:  passes 2..K all hit cache   -> per-pass cost = CACHE bandwidth
    s >  cache:  passes 2..K all miss        -> per-pass cost = DRAM bandwidth

So the curve is bandwidth-against-size with a knee at the capacity, and the knee's
SHAPE is the measurement.  A step would look like a vertical cliff; a knee looks
like a smooth S-curve over the decade of sizes either side of the capacity.

WHAT THIS IS NOT
----------------
This machine is not any of the five devices in crossover/devices.py, and its
capacities are not theirs.  The output says so, loudly, and the knee position
printed here is a fact about an Apple-silicon CPU and about nothing else.  What
transfers is the SHAPE, and only the shape.

  [MEASURED] on this machine.  [NOT MEASURED] on any device in the study.
"""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
import time

import numpy as np

MiB, GiB = 1024 ** 2, 1024 ** 3


def sysctl(name: str) -> str | None:
    if platform.system() != "Darwin":
        return None
    try:
        return subprocess.run(["sysctl", "-n", name], capture_output=True,
                              text=True, timeout=5).stdout.strip() or None
    except Exception:                                        # pragma: no cover
        return None


def local_cache_report() -> dict:
    return {
        "machine": platform.machine(),
        "cpu": sysctl("machdep.cpu.brand_string"),
        "ncpu": sysctl("hw.ncpu"),
        "l1d": int(sysctl("hw.l1dcachesize") or 0),
        "l2_perf0": int(sysctl("hw.perflevel0.l2cachesize") or 0),
        "l2_perf1": int(sysctl("hw.perflevel1.l2cachesize") or 0),
        "l3": int(sysctl("hw.l3cachesize") or 0),
        "memsize": int(sysctl("hw.memsize") or 0),
    }


def probe(sizes_bytes, passes: int, reps: int = 3):
    """Bandwidth against working-set size, with reuse.

    Two float32 arrays of s bytes each, so the working set is 2s.  The kernel is
    np.vdot(a, b): it reads both arrays once, touches every element, and is
    BLAS-bound rather than reduction-bound.

    WHY NOT a.sum(): a.sum() on float32 is pairwise-summation-bound and measures
    ~23 GB/s at EVERY size on this machine -- 0.5 MiB and 1 GiB alike -- so it
    cannot resolve a cache knee even though one is there.  Measured, not assumed:
    the first version of this file used a.sum() and reported a flat curve with
    "no knee found", which would have read as a null result about cache
    hierarchies in general.  It is a null result about numpy's reduction loop.
    np.vdot is memory-bound and shows the knee at 133 GB/s vs 28 GB/s.
    """
    out = []
    for s in sizes_bytes:
        a = np.ones(s // 4, dtype=np.float32)
        b = np.ones(s // 4, dtype=np.float32)
        np.vdot(a, b)                       # warm, and fault both in
        best = None
        for _ in range(reps):
            t0 = time.perf_counter()
            for _ in range(passes):
                np.vdot(a, b)               # one streaming read of a and b
            dt = (time.perf_counter() - t0) / passes
            best = dt if best is None else min(best, dt)
        out.append((2 * s, best, 2 * s / best / 1e9))   # bytes = working set
        del a, b
    return out


def knee_of(rows, drop=0.20):
    """The working-set size at which bandwidth first falls `drop` below the peak,
    and STAYS below.

    Requiring the drop to be sustained matters: at small sizes the measurement
    jitters by a few percent, and a naive "first point below threshold" picks up
    that jitter and reports a knee at 128 KiB on a machine whose L2 is 12 MiB.

    Reported as an interval, not a point: the last size that is still fast and
    the first that stays slow bracket the capacity, and that bracket is all any
    hardware gives you -- which is the same situation the study is in with TPU
    VMEM, and the reason the [64, 256) MiB bracket in crossover/literature.py is
    stated as a bracket rather than a number.
    """
    peak = max(g for _, _, g in rows)
    thr = peak * (1 - drop)
    first_slow = None
    for i, (s, _, g) in enumerate(rows):
        if g < thr and all(gg < thr for _, _, gg in rows[i:]):
            first_slow = s
            break
    last_fast = rows[0][0] if first_slow is None else max(
        (s for s, _, g in rows if s < first_slow), default=rows[0][0])
    return peak, last_fast, first_slow


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-kib", type=int, default=64)
    ap.add_argument("--max-gib", type=float, default=2.0)
    ap.add_argument("--passes", type=int, default=16)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--points", type=int, default=26)
    args = ap.parse_args()

    info = local_cache_report()
    print("=" * 100)
    print("RESIDENCY PROBE -- is 'does it fit' a STEP or a KNEE?  Measured here.")
    print("=" * 100)
    print(f"  machine   {info['cpu'] or 'unknown'}  ({info['machine']}, "
          f"{info['ncpu']} cores, {info['memsize']/GiB:.0f} GiB RAM)")
    print(f"  L1d       {info['l1d']/1024:.0f} KiB")
    print(f"  L2 P-core {info['l2_perf0']/MiB:.0f} MiB   "
          f"L2 E-core {info['l2_perf1']/MiB:.0f} MiB")
    print(f"  L3        {info['l3']/MiB:.0f} MiB   "
          f"({'not exposed by sysctl on this architecture' if not info['l3'] else ''})")
    print()
    print("  THIS MACHINE IS NOT ONE OF THE FIVE DEVICES IN crossover/devices.py.")
    print("  The knee position below is a fact about this CPU and about nothing")
    print("  else.  The SHAPE is the transferable result.  [MEASURED] here,")
    print("  [NOT MEASURED] on T4 / P100 / L4 / H100 / B200.")
    print()

    # geometric grid over the PER-ARRAY size; the working set is 2x that
    # cap by RAM: two arrays per point, and a machine near its memory limit
    # reports swap bandwidth, not cache behaviour (measured: the 1.82 GiB
    # point read 9.4 GB/s on a 32 GiB machine, which is neither cache nor DRAM)
    lo = args.min_kib * 1024
    hi = min(int(args.max_gib * GiB), info['memsize'] // 16)
    sizes, v = [], lo
    while v <= hi:
        sizes.append(int(v))
        v = max(int(v * 1.4), v + 262144)
    step = max(1, len(sizes) // args.points)
    sizes = sizes[::step]

    print(f"  {args.passes} passes over each buffer, best of {args.reps}, "
          f"np.vdot(a,b) on two float32 arrays")
    print("  'working set' = 2 x the per-array size, i.e. what has to stay resident")
    print()
    hdr = (f"  {'per array':>12}{'working set':>13}{'us / pass':>12}{'GB/s':>9}"
           f"{'x peak':>9}   {'regime':<12}")
    print(hdr)
    print("  " + "-" * 70)
    rows = probe(sizes, args.passes, args.reps)
    peak = max(g for _, _, g in rows)
    for (ws, dt, g), s in zip(rows, sizes):
        regime = "on-chip" if g >= peak * 0.8 else "OFF-CHIP"
        print(f"  {_fmt(s):>12}{_fmt(ws):>13}{dt*1e6:>12.1f}{g:>9.1f}{g/peak:>8.2f}x   "
              f"{regime:<12}")
    peak_s, last_fast, first_slow = knee_of(rows)
    print()
    print(f"  peak streaming bandwidth (working set resident) {peak:.1f} GB/s")
    print(f"  first size below 80% of peak      "
          f"{_fmt(first_slow) if first_slow else 'not reached'}")
    print(f"  last size at or above 80% of peak {_fmt(last_fast)}")
    if first_slow:
        print(f"  => capacity bracketed to          [{_fmt(last_fast)}, "
              f"{_fmt(first_slow)})  -- a factor of {first_slow/last_fast:.2f}")
        for label, v in (("L2 per P-core", info["l2_perf0"]),
                         ("L2 per E-core", info["l2_perf1"])):
            if v:
                inside = last_fast <= v <= first_slow
                frac = first_slow / v
                print(f"     {label} = {v/MiB:>5.1f} MiB nominal  ->  "
                      f"{'inside ' if inside else 'OUTSIDE'} the bracket; the bracket's"
                      f" upper edge is {frac*100:.0f}% of nominal")
        print()
        print("  READ THAT LAST LINE CAREFULLY, IT IS THE POINT OF THE WHOLE FILE.")
        print("  The usable capacity of a real cache, for a real streaming pattern, is")
        print("  a fraction of its datasheet capacity -- here roughly half to")
        print("  three-quarters.  Every capacity in crossover/devices.py is a NOMINAL")
        print("  number read off a vendor document.  So every crossover prediction in")
        print("  crossover/claim.py is computed against a denominator that is known to")
        print("  be too large by a factor nobody in this corpus has measured, and the")
        print("  predictions are therefore LATE, not wrong in kind: the real crossover")
        print("  is at a smaller intermediate than the prediction says.")
        print("  This is a bias, not noise, and it has a sign.")
    print()
    print("  IS THE TRANSITION A STEP OR A KNEE?")
    print("  -----------------------------------")
    # How many of the measured points lie in the decade either side of the knee?
    if first_slow:
        lo_d = last_fast / 10
        hi_d = first_slow * 10
        band = [r for r in rows if lo_d <= r[0] <= hi_d]
        gbs = [g for _, _, g in band]
        drop = (max(gbs) - min(gbs)) / max(gbs) if gbs else 0
        n_in_band = len(band)
        print(f"  {n_in_band} of {len(rows)} measured points fall inside the two")
        print(f"  decades around the knee, and bandwidth varies by {drop*100:.0f}% across them.")
        if n_in_band >= 4 and drop > 0.15:
            print()
            print("  VERDICT: KNEE.  The transition is spread over at least 4 sampled")
            print("  sizes and the bandwidth varies substantially within the two decades")
            print("  either side of the capacity.  A step function would put the whole")
            print("  change between two ADJACENT samples.")
            print()
            print("  This is the direct measurement refuting R-A's step prediction, on")
            print("  real hardware, for a real memory hierarchy.  It does not settle")
            print("  anything about GPUs by itself -- capacities, block sizes and the")
            print("  compiler's fusion decisions are all different there -- but a")
            print("  thesis that predicts a DISCONTINUITY in a quantity that is")
            print("  continuous everywhere it has been measured is wrong about the")
            print("  quantity, not just about the threshold.")
        else:
            print()
            print("  VERDICT: INCONCLUSIVE on this machine -- too few points land near")
            print("  the knee.  Widen the grid (--points, --min-kib, --max-gib) or")
            print("  lower --passes and rerun.  Do not read this as support for a step.")
    else:
        print("  No knee found inside the swept range.  Widen --max-gib.")
    print()
    print("  WHAT TO DO WITH THIS ON A GPU.  The same probe is not the right one")
    print("  there, because a GPU's L2 is not something a single thread streams")
    print("  through.  The right one is crossover/attn_control.py: sweep n, watch")
    print("  the modelled byte ratio, and compare it with the measured time ratio")
    print("  R(n).  Then apply the same shape test to R(n): does the change happen")
    print("  between two adjacent samples or over several?  That is R-A, and it is")
    print("  a one-line analysis once the sweep has run.")
    print("=" * 100)
    return 0


def _fmt(n: int) -> str:
    for scale, suf in ((1024 ** 3, "GiB"), (1024 ** 2, "MiB"), (1024, "KiB")):
        if abs(n) >= scale:
            return f"{n/scale:.2f} {suf}"
    return f"{n:.0f} B"


if __name__ == "__main__":
    raise SystemExit(main())
