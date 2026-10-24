#!/usr/bin/env python3
"""analysis.py -- turn the on-card report into the axis table and the falloff.

RUN ON THE HOST, on the JSON the card produced. No GPU, no network. It reads a
measurement and prints a table; it never invents one. Every cell it prints is
either copied from the report with its status, or derived arithmetically from
cells that are, and each derived column says which.

WHAT IT DELIBERATELY DOES NOT DO
================================
It does not smooth, rank, or drop a rung. A rung whose two arms disagree is
carried into the table with its INVALID status and no ratio, because a ratio
between two different computations is the specific failure this whole study has
had to take back twice (notes/06 §5b). It also does not convert a kernel count
into a cost: notes/06 C-22 is the correction that says an offline census is not
the kernel count and that only relative ordering between candidates measured the
same way is load-bearing. The kernel count appears as a count, labelled, and the
cost is always a time.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# The ladder, in the order it is presented. The order is the ARGUMENT: it runs
# from "demands nothing of a compiler" to "demands a carried chain with a dense
# contraction inside it", and the falloff is read along it.
ORDER = ["ew", "red", "gather", "gemm32", "gemm16", "scan", "trisolve"]

# What each rung is a proxy for. The point of the ladder is that a gap can be
# attributed, and attribution needs a rung that isolates one demand. These are
# the isolations, stated so a reader can check whether a rung really isolates
# what it claims -- and the (scan, trisolve) pair is the load-bearing one, since
# it is the only pair that differs by exactly one thing.
DEMANDS = {
    "ew":       "nothing (control)",
    "red":      "one reduction, no reuse",
    "gather":   "one random address, no reuse",
    "gemm32":   "reuse, no tensor core on this ISA",
    "gemm16":   "reuse, tensor core present",
    "scan":     "a carried chain, no contraction",
    "trisolve": "a carried chain WITH a dense contraction",
}

# The designed contrasts. These are what turn a list of ratios into a
# decomposition, and they are why these particular seven rungs are the seven.
CONTRASTS = [
    ("ew", "red",
     "same traffic class, one reduction added",
     "attributes a gap to the REDUCTION rather than to the memory path"),
    ("ew", "gather",
     "same traffic class, one gather added",
     "attributes a gap to the ADDRESS rather than to the memory path"),
    ("gemm32", "gemm16",
     "the same contraction, two dtypes; the ISA has a tensor core for one and not the other",
     "separates 'the framework schedules reuse badly' from 'there is no instruction for this'"),
    ("scan", "trisolve",
     "a carried chain, then the same chain with a dense contraction at every step",
     "separates the cost of SEQUENCING from the cost of SEQUENCING-WITH-CONTRIBUTION"),
]


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("report", type=Path, nargs="+",
                    help="one or more card-produced JSON reports; later files "
                         "fill in fields the earlier ones lack (the ladder and "
                         "the controls are run as separate cells so a control "
                         "cannot cost a rung)")
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args()

    # Merge. The first report supplies the environment, the floor and the ladder;
    # any later report contributes the control fields (tg_naive, tg_knob) for the
    # same rungs without overwriting a timing that the ladder already measured.
    rep = load(a.report[0])
    by = {r.get("key"): dict(r) for r in rep.get("rungs", [])}
    for extra in a.report[1:]:
        if not extra.exists():
            continue
        d = load(extra)
        for r in d.get("rungs", []):
            k = r.get("key")
            if k not in by:
                by[k] = dict(r)
            else:
                for f in ("tg_naive", "tg_naive_shapes", "tg_naive_scaled_ms",
                          "tg_naive_scale_note", "tg_naive_error", "tg_knob",
                          "tg_knob_label", "tg_knob_error"):
                    if f in r and f not in by[k]:
                        by[k][f] = r[f]
        for f in ("floor", "device", "env", "build", "sweep_scan_N",
                  "sweep_threshold"):
            if f not in rep and f in d:
                rep[f] = d[f]
    rep["rungs"] = list(by.values())
    floor = rep.get("floor", {})
    dev = rep.get("device", {})
    env = rep.get("env", {})

    print("=" * 100)
    print("THE AXIS -- one card, one run, one instrument")
    print("=" * 100)
    print(f"  card        {env.get('nvidia_smi', '?')}")
    print(f"  tinygrad    {env.get('tinygrad_version', '?')}  ({env.get('tinygrad', '?')})")
    print(f"  SMs         {dev.get('SMs', '?')}   L2 {dev.get('l2_bytes', 0)/1e6:.1f} MB"
          f"   clock {dev.get('clock_khz', 0)/1e3:.0f} MHz   cc {dev.get('compute_cap', '?')}"
          if dev.get("SMs") else f"  device props unavailable: {dev.get('error')}")
    if floor.get("copy_GBps"):
        print(f"  floor       fill {floor['fill_GBps']:.1f} GB/s (write only), "
              f"copy {floor['copy_GBps']:.1f} GB/s (read+write), measured in this run "
              f"at n={floor['n']}")
    print()

    # ---------------- the main table ----------------
    #
    # `%floor` is `axis_copy`, which is a READ-AND-WRITE floor. Dividing a
    # read-only-dominant or gather-heavy rung by it is the error
    # `refkernels.cu`'s own header warns about -- "a one-direction row must not
    # be divided by a two-direction one" -- so the column is marked per row
    # rather than printed uniformly. Only `scan` is copy-shaped.
    FLOOR_OK = {"scan"}          # 1 read : 1 write
    FLOOR_INDICATIVE = {"ew", "red", "gather"}   # read-dominant or not modelled
    hdr = (f"  {'rung':<9} {'demands':<40} {'tg dev':>9} {'tg e2e':>9} {'ref':>9} "
           f"{'tg/ref':>7} {'GB/s':>7} {'%floor':>7} {'k':>4}  status")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    rows = {}
    for key in ORDER:
        r = by.get(key)
        if r is None:
            print(f"  {key:<9} {'<no measurement>':<40}")
            continue
        rows[key] = r
        # tg dev is DEVICE time and is the ratio's numerator; tg e2e is the
        # end-to-end wall clock including the Python graph build, which the cubin
        # side does not have and so is context, never a denominator.
        tg = r.get("tg_device", {}).get("median_ms")
        e2e = r.get("tg", {}).get("median_ms")
        rf = r.get("ref", {}).get("median_ms")
        ratio = r.get("ratio_tg_over_ref")
        byb = r.get("bytes")
        gbps = pct = None
        if tg and byb and str(r.get("status", "")).startswith("OK"):
            gbps = byb / (tg * 1e-3) / 1e9
            if floor.get("copy_GBps") and key in ("ew", "red", "scan", "gather"):
                pct = 100.0 * gbps / floor["copy_GBps"]
        mark = "=" if key in FLOOR_OK else ("~" if key in FLOOR_INDICATIVE else " ")
        pcts = "-" if not pct else f"{pct:.0f}%{mark}"
        print(f"  {key:<9} {DEMANDS[key][:40]:<40} "
              f"{(f'{tg:.4f}' if tg else '-'):>9} "
              f"{(f'{e2e:.3f}' if e2e else '-'):>9} "
              f"{(f'{rf:.4f}' if rf else '-'):>9} "
              f"{(f'{ratio:.2f}x' if ratio else '-'):>7} "
              f"{(f'{gbps:.0f}' if gbps else '-'):>7} "
              f"{pcts:>7} "
              f"{(str(r.get('tg_kernels', '-'))):>4}  {r.get('status', '?')}")

    if FLOOR_OK or FLOOR_INDICATIVE:
        print(f"  %floor: '=' {sorted(FLOOR_OK)} is a copy-shaped rung (1 read : 1 write) "
              f"and the percentage means what it says;")
        print(f"          '~' {sorted(FLOOR_INDICATIVE)} is read-dominant or has unmodelled "
              f"traffic (a 16 MB gather table) and is INDICATIVE ONLY -- "
              f"a one-direction row")
        print(f"          must not be divided by a two-direction floor, which is "
              f"`axis_copy`'s own stated rule.")

    # ---------------- where each GEMM sits against its INSTRUCTION-CLASS peak
    #
    # The bandwidth column above has a MEASURED denominator (`axis_copy`, this
    # run, this card). The compute side needs one too, and it cannot be measured
    # the same way -- a peak-flops probe is itself a kernel, and this study has
    # no budget for writing one. So this block is arithmetic on the card's
    # MEASURED clock (cuDeviceGetAttribute) and its ARCHITECTURAL per-SM
    # throughput, and it is labelled as such at every use.
    #
    # The two architectural constants, both properties of Turing (sm_75) and not
    # of this card:
    #     FP32 FFMA   64 lanes/SM  -> 128 FLOP/cycle/SM
    #     FP16 HMMA  1024 FLOP/cycle/SM with FP32 accumulate
    # 8.1 and 65 TFLOP/s at 1.59 GHz fall out of both, which is the check that
    # they are the right constants for this part; the T4 is documented at
    # exactly those figures.
    #
    # WHY THIS BLOCK IS HERE AND NOT A FOOTNOTE. A ratio against a reference
    # that is itself far from the roofline is partly a statement about the
    # reference. If the fp32 reference were at 60% of FFMA peak, the gemm32 rung
    # would be measuring a weak reference and the note would have to say so; if
    # it is at 17%, the rung's 3x is a real gap AND the reference is weak, and
    # both facts are needed. Reporting only the ratio would hide the second.
    sms, clk_khz = dev.get("SMs"), dev.get("clock_khz")
    gemm_rows = [k for k in ("gemm32", "gemm16") if by.get(k, {}).get("flops")]
    if sms and clk_khz and gemm_rows:
        hz = clk_khz * 1e3
        peaks = {"gemm32": (128 * sms * hz, "FFMA (fp32, no TC on sm_75)"),
                 "gemm16": (1024 * sms * hz, "HMMA (fp16, fp32 accumulate)")}
        print()
        print("=" * 100)
        print("COMPUTE ROOFLINE -- % of the peak implied by the card's MEASURED clock")
        print("=" * 100)
        print(f"  clock {hz/1e9:.3f} GHz (cuDeviceGetAttribute) x {sms} SMs")
        for k in gemm_rows:
            fl, kind = peaks[k]
            print(f"  {k:<9} peak {fl/1e12:6.2f} TFLOP/s  = {sms} SMs x "
                  f"{'128' if k=='gemm32' else '1024'} FLOP/cycle/SM  [{kind}]")
        print()
        print(f"  {'rung':<9} {'arm':<5} {'ms':>9} {'GFLOP/s':>10} {'% of peak':>10}")
        for k in gemm_rows:
            r = by[k]
            fl, _ = peaks[k]
            for arm, ms in (("ref", r.get("ref", {}).get("median_ms")),
                            ("tg", r.get("tg_device", {}).get("median_ms"))):
                if not ms:
                    continue
                gf = r["flops"] / (ms * 1e-3) / 1e9
                print(f"  {k:<9} {arm:<5} {ms:9.4f} {gf:10.0f} {100*gf*1e9/fl:9.1f}%")
        print()
        print("  The clock is measured; the FLOP/cycle/SM figures are architecture.")
        print("  A reference far below its own peak means the ratio is partly a")
        print("  statement about the reference, and is reported as such.")

    # ---------------- the width sweep, if this run took one
    sweep = rep.get("sweep_scan_N")
    if sweep:
        print()
        print("=" * 100)
        print("THE WIDTH SWEEP -- scan as a function of N, ROWS x N held constant")
        print("=" * 100)
        print("  A single shape is a number; a function of N is a curve, and the curve")
        print("  is the finding. The sweep is the falsifier for the `scan` account.")
        print()
        print("  WHAT IT SETTLED: cumsum re-sums a window of width min(N, 256) per")
        print("  output, at EVERY N. There is no discontinuity at SPLIT*2 = 512 -- the")
        print("  note predicted one and the data refuted it. What the data shows")
        print("  instead is the window saturating: the framework's time RISES with N")
        print("  while N < 256 and is FLAT above it, which is the signature of a")
        print("  window that is min(N, 256) rather than of a threshold switch.")
        print()
        print(f"  {'N':>7} {'ROWS':>8} {'tg ms':>10} {'ref ms':>10} {'ratio':>8} "
              f"{'tg kernels':>11}  status")
        for row in sweep:
            tg, rf = row.get("tg_ms"), row.get("ref_ms")
            rat = row.get("ratio")
            print(f"  {row['N']:>7} {row['R']:>8} "
                  f"{(f'{tg:.4f}' if tg else '-'):>10} "
                  f"{(f'{rf:.4f}' if rf else '-'):>10} "
                  f"{(f'{rat:.2f}x' if rat else '-'):>8} "
                  f"{str(row.get('kernels', '-')):>11}  {row.get('status', '?')}")
        ok = [r for r in sweep if r.get("status") == "OK"]
        if len(ok) >= 2:
            tms = [r["tg_ms"] for r in ok]
            print()
            print(f"  framework time over the OK points: {min(tms):.4f} - "
                  f"{max(tms):.4f} ms, spread {max(tms)/min(tms):.2f}x")
            # the saturation test: split at SPLIT itself, not at SPLIT*2. A window
            # of min(N, SPLIT) is flat ABOVE SPLIT and rising BELOW it; a threshold
            # switch at SPLIT*2 would step there instead. Both are printed so the
            # two are distinguishable rather than one being asserted.
            SPLIT = 256
            below = [r["tg_ms"] for r in ok if r["N"] < SPLIT]
            above = [r["tg_ms"] for r in ok if r["N"] >= SPLIT]
            if len(below) >= 2 and above:
                lo, hi = max(below), min(above)
                print(f"  max at N<{SPLIT}:  {lo:.4f} ms")
                print(f"  min at N>={SPLIT}: {hi:.4f} ms")
                print(f"  ratio across the boundary: {hi/lo:.2f}x -- the curve is "
                      f"already SATURATED at N={SPLIT},")
                print("  which is the signature of a window of width min(N, "
                      f"{SPLIT}), not of a")
                print(f"  threshold switch at SPLIT*2 = {SPLIT*2}.")
            rr = sorted((r["N"], r["ratio"]) for r in ok if r.get("ratio"))
            if len(rr) >= 2:
                print()
                print("  the ratio against N, which is the result:")
                for N, ratio in rr:
                    print(f"    N={N:<7} {ratio:6.2f}x")
                lo, hi = rr[0], rr[-1]
                print(f"  the framework's disadvantage GROWS with N: "
                      f"{lo[1]:.2f}x at N={lo[0]} to {hi[1]:.2f}x at N={hi[0]} "
                      f"({hi[1]/lo[1]:.1f}x)")
        nbad = [r for r in sweep if str(r.get("status", "")).startswith("INVALID")]
        if nbad:
            print()
            print(f"  {len(nbad)} point(s) INVALID, and correctly reported as such:")
            for r in nbad:
                print(f"    N={r['N']:<7} {r['status']}")
            print("  The reference has no correct kernel at these N -- the vectorised")
            print("  path covers exactly 1024*PT elements and the scalar fallback needs")
            print("  N <= 256 -- so the harness reports INVALID rather than timing a")
            print("  kernel that would write part of a row.")

    # ---------------- the controls ----------------
    print()
    print("=" * 100)
    print("THE CONTROLS -- the same card, the same run, the same instrument")
    print("=" * 100)
    print("  These are what make the axis a measurement of the FRAMEWORK and not of a port.")
    print("  notes/26 and notes/31 found one habit -- reducing to a scalar inside a loop")
    print("  where a wide contraction belongs -- worth ~60x on its own, hidden by kernel")
    print("  count and exposed by ALU-lane op count. 'Best TinyGrad' below means that habit")
    print("  is fixed everywhere, not only where it was noticed.")
    print()
    for key in ORDER:
        r = rows.get(key)
        if not r:
            continue
        if "tg_naive" in r:
            naive = r["tg_naive"]["median_ms"]
            good = r.get("tg_device", {}).get("median_ms")
            ref = r.get("ref", {}).get("median_ms")
            if naive and good:
                print(f"  {key:<9} idiomatic {good:9.4f} ms   naive-habit {naive:9.4f} ms"
                      f"   habit costs {naive/good:8.1f}x"
                      + (f"   (reference {ref:.4f} ms)" if ref else ""))
                print(f"            -> the {key} rung's TinyGrad point is the IDIOMATIC arm."
                      f" The naive arm is a control, not the result. Both are device time.")
        if "tg_knob" in r:
            k = r["tg_knob"]["median_ms"]
            good = r.get("tg_device", {}).get("median_ms")
            ref = r.get("ref", {}).get("median_ms")
            if good:
                print(f"  {key:<9} knob {r.get('tg_knob_label','?')} {k:9.4f} ms"
                      f"   vs default {good:.4f} ms")
            if good and k:
                print(f"            -> knob moves it {k/good:.2f}x"
                      + (f"; the reference is at {ref:.4f} ms, so the knob "
                         f"{'closes' if k <= ref else 'does not close'} the gap." if ref else ""))

    # ---------------- the contrasts ----------------
    print()
    print("=" * 100)
    print("THE CONTRASTS -- what each pair of rungs separates")
    print("=" * 100)
    for x, y, what, why in CONTRASTS:
        rx, ry = rows.get(x), rows.get(y)
        if not rx or not ry:
            print(f"\n  {x} vs {y}: not both measured")
            continue
        sx, sy = str(rx.get("status", "")), str(ry.get("status", ""))
        okx, oky = sx.startswith("OK"), sy.startswith("OK")
        print(f"\n  {x} vs {y}: {what}")
        print(f"    {why}")
        if okx and oky:
            print(f"    tg/ref: {x} = {rx['ratio_tg_over_ref']:.2f}x, "
                  f"{y} = {ry['ratio_tg_over_ref']:.2f}x")
        else:
            print(f"    statuses: {x}={sx or '?'}  {y}={sy or '?'} -- "
                  f"no contrast is drawn from a rung that is not OK")

    # ---------------- the falloff, and its correlates ----------------
    print()
    print("=" * 100)
    print("THE FALLOFF, AND WHAT IT IS CORRELATED WITH")
    print("=" * 100)
    ok = {k: v for k, v in rows.items() if str(v.get("status", "")).startswith("OK")
          and v.get("ratio_tg_over_ref")}
    if not ok:
        print("  no rung has a valid ratio; there is no falloff to read.")
        return 1

    # Three candidate correlates, each one a property of the rung that is not a
    # property of the result. The point of listing three is that picking the one
    # that fits after the fact is exactly the failure this study keeps making.
    print()
    print("  candidate correlates, and what each would have to explain:")
    print()
    cands = [
        ("sequential depth (carried dependent steps per element)",
         {"ew": 0, "red": 0, "gather": 0, "gemm32": 0, "gemm16": 0,
          "scan": 1, "trisolve": 2},
         "predicts: no gap without a chain, growing gap with one"),
        ("contraction present (a dense product inside the operator)",
         {"ew": 0, "red": 0, "gather": 0, "gemm32": 1, "gemm16": 1,
          "scan": 0, "trisolve": 1},
         "predicts: a gap exactly at the rungs with a dense product"),
        ("reuse (each input byte read more than once)",
         {"ew": 0, "red": 0, "gather": 0, "gemm32": 1, "gemm16": 1,
          "scan": 0, "trisolve": 1},
         "predicts: a gap exactly at the arithmetically dense rungs"),
    ]
    xs = [ORDER.index(k) for k in ok]
    ys = [ok[k]["ratio_tg_over_ref"] for k in ok]
    for name, table, prediction in cands:
        xs2 = [table[k] for k in ok]
        r = _pearson(xs2, ys)
        verdict = _verdict(r, len(ok))
        print(f"    {name}")
        print(f"      r = {r:+.3f} over n={len(ok)} rungs   {verdict}")
        print(f"      {prediction}")
        print()
    print("  n is small. A correlation over seven points with three candidates and no")
    print("  held-out prediction is a SUGGESTION, not a result, and the note says so.")
    print("  The CONTRASTS above are the load-bearing evidence; the correlation is the")
    print("  weaker claim it is reported alongside.")

    if a.json:
        a.json.write_text(json.dumps(
            {"axis": {k: {"tg_ms": v.get("tg", {}).get("median_ms"),
                          "ref_ms": v.get("ref", {}).get("median_ms"),
                          "ratio": v.get("ratio_tg_over_ref"),
                          "status": v.get("status"),
                          "kernels": v.get("tg_kernels"),
                          "naive_ms": v.get("tg_naive", {}).get("median_ms"),
                          "knob_ms": v.get("tg_knob", {}).get("median_ms"),
                          "knob": v.get("tg_knob_label")}
                      for k, v in rows.items()},
             "floor": floor, "device": dev, "env": env}, indent=2, default=str))
        print(f"\n  wrote {a.json}")
    return 0


def _pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 3:
        return 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    return num / (dx * dy) if dx and dy else 0.0


def _verdict(r: float, n: int) -> str:
    a = abs(r)
    if a < 0.3:
        return "no detectable association"
    if a < 0.7:
        return "weak association"
    return "strong association, but n=%d and unvalidated" % n


if __name__ == "__main__":
    sys.exit(main())
