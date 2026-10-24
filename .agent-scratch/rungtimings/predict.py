#!/usr/bin/env python3
"""predict.py -- which currency orders the five rungs' wall clock?

Four candidate explanations for the same five numbers, scored the same way:
Pearson r against the measured median device time, at each batch. The point is
not to find one that wins by a hair; it is to find out which ones CANNOT order
the five rungs at all, because a currency that puts two rungs in the wrong order
is not a weaker explanation, it is not one.

    python3 predict.py <run-dir> <opcount.json>
"""
from __future__ import annotations

import json
import math
import sys

LADDER = ["naive", "int_cursor", "flat_gather", "incremental", "lean"]


def pearson(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    syy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return sxy / (sxx * syy) if sxx and syy else float("nan")


def spearman(xs, ys):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    return pearson(rank(xs), rank(ys))


def main():
    T = json.load(open(f"{sys.argv[1]}/rung_timings.json"))
    oc = json.load(open(sys.argv[2]))
    g, sa = oc["graph"], oc["sass"]
    ref_sass = oc["reference"]["instructions_per_step"]

    cand = {
        "graph ops / pixel / step": {r: g[r]["per_pixel_per_step"] for r in LADDER},
        "generated stmts / pixel / step": {
            r: oc["statement_slope"][r]["per_pixel_per_step"] for r in LADDER},
        "SASS / pixel / step": {
            r: sa[r]["slope_hot_per_pixel_per_step"] for r in LADDER},
        "registers": {r: float(sa[r]["32"]["registers"]) for r in LADDER},
        "local spill bytes @ ms=32": {r: float(sa[r]["32"]["local_bytes"]) for r in LADDER},
    }

    for b in sorted(T, key=int):
        rows = {r["rung"]: r for r in T[b]["rows"] if not r.get("error")}
        have = [r for r in LADDER if r in rows]
        if len(have) < 2:
            continue
        t = [rows[r]["median_ms"] for r in have]
        launches = [rows[r]["n_kernels_cold"][0] for r in have]
        print("=" * 78)
        print(f"batch {b}   time (ms): " +
              "  ".join(f"{r}={rows[r]['median_ms']:.3f}" for r in have))
        print(f"  launches: " + "  ".join(f"{r}={l}" for r, l in zip(have, launches)))
        rows_out = []
        for name, d in cand.items():
            xs = [d[r] for r in have]
            rows_out.append((spearman(xs, t), pearson(xs, t), name, xs))
        rows_out.append((spearman(launches, t), pearson(launches, t),
                         "launch count (measured)", launches))
        for sp, pe, name, xs in sorted(rows_out, key=lambda z: -abs(z[0])):
            print(f"    {name:<34} spearman {sp:+.4f}   pearson {pe:+.4f}"
                  f"   [{', '.join(f'{x:g}' for x in xs)}]")
        print("    NOTE: every candidate above is a cost that should INCREASE time,")
        print("    so the correlation is expected NEGATIVE. The ranking is what")
        print("    matters, and rank is scale-free.")

    print()
    print("=" * 78)
    print("THE STEP-BY-STEP PREDICTION: what each rewrite was worth")
    print("=" * 78)
    b = "1024"
    rows = {r["rung"]: r for r in T[b]["rows"] if not r.get("error")}
    print(f"  at batch {b}")
    print(f"  {'rewrite':<34}{'measured':>10}{'from sass':>11}{'excess':>9}"
          f"{'from spill':>12}{'excess':>9}")
    names = {"int_cursor": "1  cursor cast once",
             "flat_gather": "2  one flat index",
             "incremental": "3  maintained address",
             "lean": "4  no found flag"}
    ex_s = ex_l = 1.0
    for i, r in enumerate(LADDER):
        if i == 0:
            continue
        up = LADDER[i - 1]
        m = rows[up]["median_ms"] / rows[r]["median_ms"]
        s = sa[up]["slope_hot_per_pixel_per_step"] / sa[r]["slope_hot_per_pixel_per_step"]
        lb = float(sa[up]["32"]["local_bytes"]) / max(1.0, float(sa[r]["32"]["local_bytes"]))
        ex_s *= m / s
        ex_l *= m / lb
        print(f"  {names[r]:<34}{m:>9.3f}x{s:>10.3f}x{m/s:>8.3f}x"
              f"{lb:>11.3f}x{m/lb:>8.3f}x")
    print(f"  cumulative excess over the whole ladder: sass {ex_s:.3f}x, "
          f"spill {ex_l:.3f}x")
    print(f"  the same, from the naive end: measured {rows['naive']['median_ms']/rows['lean']['median_ms']:.3f}x, "
          f"sass explains {sa['naive']['slope_hot_per_pixel_per_step']/sa['lean']['slope_hot_per_pixel_per_step']:.3f}x, "
          f"spill explains {float(sa['naive']["32"]['local_bytes'])/float(sa['lean']["32"]['local_bytes']):.3f}x")
    print()
    print(f"  hand-written reference: {ref_sass} SASS per pixel per DDA step")
    for r in LADDER:
        print(f"    {r:<14}{sa[r]['slope_hot_per_pixel_per_step']:6.1f} = "
              f"{sa[r]['slope_hot_per_pixel_per_step']/ref_sass:.3f}x the reference")


if __name__ == "__main__":
    main()
