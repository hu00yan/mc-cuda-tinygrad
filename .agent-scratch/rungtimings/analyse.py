#!/usr/bin/env python3
"""analyse.py -- the deliverable tables, computed from the run's own JSON.

Nothing here is transcribed. The wall clock, the launch counts, the allocator
peaks and the reference denominator all come out of the run directory; the graph
operation counts and the SASS counts come out of the offline instrument this
script is pointed at. The two are joined only at the end, because the join is
the finding: which rung's time moved, and which rung's instruction count moved.

    python3 analyse.py <run-dir> [--opcount <opcount.json>] [--batch 1024]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics

LADDER = ["naive", "int_cursor", "flat_gather", "incremental", "lean"]


def find(dest: str, name: str):
    for p in [os.path.join(dest, name)] + glob.glob(os.path.join(dest, "*", name)):
        if os.path.exists(p):
            return json.load(open(p))
    return None


def mib(v):
    return v / 2**20 if isinstance(v, (int, float)) else v


def one(v, fmt="{:.3f}"):
    if isinstance(v, (int, float)):
        return fmt.format(v)
    if isinstance(v, list):
        return "/".join(fmt.format(x) for x in v)
    return str(v)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--opcount", default=None)
    ap.add_argument("--census", default=None,
                    help="capture_opt.py's census.json, for the schedule's own "
                         "footprint estimate (a MODELLED number, not a measured "
                         "one -- see the caching-allocator caveat in the note)")
    ap.add_argument("--batch", type=int, default=1024,
                    help="the batch at which the ladder is walked rung by rung")
    a = ap.parse_args()

    T = find(a.run, "rung_timings.json") or {}
    ref = find(a.run, "ref_timing.json") or {}
    eq = find(a.run, "obs_equivalence.json") or {}
    oc = json.load(open(a.opcount)) if a.opcount and os.path.exists(a.opcount) else {}
    census = json.load(open(a.census)) if a.census and os.path.exists(a.census) else []

    # per-rung footprint straight out of the census, at max_steps 32
    est = {}
    for entry in census:
        v = entry.get("variant") or entry.get("rung")
        if not v:
            continue
        for row in entry.get("rows", []):
            if row.get("max_steps") == 32:
                est[v] = {"bytes_estimate": row.get("bytes_estimate"),
                          "kernel_launches": row.get("kernel_launches")}

    batches = sorted(T, key=int)
    rows = {}
    for b in batches:
        rows[b] = {r["rung"]: r for r in T[b].get("rows", []) if not r.get("error")}

    print("=" * 78)
    print("1. THE GATE, on the card, before anything was timed")
    print("=" * 78)
    if eq:
        for r in eq.get("rows", []):
            print(f"  {r['rung']:<13} {r['pixels_differing']:>9} / {eq['pixels']:>9}"
                  f" pixels differ   {'BYTE-IDENTICAL' if r['identical'] else 'DIFFERS'}")
        d = eq.get("dda", {})
        if d:
            print("  mean DDA steps/pixel: " + "  ".join(
                f"{k}={v['mean']:.6f}" for k, v in d.items()))
            sp = eq.get("dda_spread", {})
            print(f"  spread across the five rungs: {sp.get('abs_steps'):.6g} steps"
                  f"  ({sp.get('spread_pct'):.4f}%)")
    else:
        print("  (no obs_equivalence.json)")

    print()
    print("=" * 78)
    print("2. WALL CLOCK, ALL FIVE RUNGS, AND THE RATIO TO THE REFERENCE")
    print("=" * 78)
    print("  ref = cudaEvent median render, MICROseconds, measured in THIS run.")
    print("  tg  = sum of per-CALL device times for the whole generated graph, ms.")
    print("  ratio = tg_ms*1000/ref_us, i.e. how many times slower. 1.0 is parity.")
    print()
    head = "  batch  pixels     ref us |"
    for r in LADDER:
        head += f" {r[:11]:>16} |"
    print(head)
    for b in batches:
        ru = (ref.get(b) or {}).get("render_us", [None])[0]
        line = f"  {b:>5} {T[b].get('pixels', 0):>9,} {round(ru,1) if ru else 0:>8} |"
        for r in LADDER:
            e = rows[b].get(r)
            if e and ru:
                ms, ra = e["median_ms"], e["median_ms"] * 1000.0 / ru
                line += f"  {ms:8.3f} {ra:5.1f}x |"
            else:
                line += f" {'-':>16} |"
        print(line)

    print()
    print("=" * 78)
    print("3. LAUNCH COUNT, ALLOCATOR PEAK, REPEAT SPREAD, GUARDS")
    print("=" * 78)
    print("  THE ALLOCATOR-PEAK CAVEAT, and it is not a small one. MemProbe sees")
    print("  exactly one thing: calls to Allocator.alloc. tinygrad's CUDA backend")
    print("  CACHES device buffers, so only the FIRST graph built at a given shape")
    print("  goes through cudaMalloc. Every later graph at the same shape is served")
    print("  from buffers the process already holds, so n_allocs reads 0 and the")
    print("  probe's live-bytes high-water mark reads 0 -- an artefact of the")
    print("  measurement ORDER, not a statement that the rung allocates nothing.")
    print("  Five rungs in one process therefore yield ONE clean peak per shape, on")
    print("  whichever rung happened to be measured first.")
    print()
    for b in batches:
        print(f"  batch {b}   ({T[b].get('pixels',0):,} pixels)")
        print(f"    {'rung':<13}{'ms':>9}{'spread':>9}{'CALLs':>7}{'walk':>7}"
              f"{'walk sum ms':>13}{'peak MiB':>10}{'allocs':>8}{'guards':>8}"
              f"{'e2e s':>8}")
        for r in LADDER:
            e = rows[b].get(r)
            if not e:
                print(f"    {r:<13}{'NOT MEASURED':>9}")
                continue
            w = e.get("walk") or {}
            wr = e.get("walk_vs_ruler_pct")
            line = (f"    {r:<13}{e['median_ms']:>9.3f}{e['spread_pct']:>8.2f}%"
                    f"{one(e.get('n_kernels_cold'),'{:g}'):>7}"
                    f"{one(w.get('n_calls'),'{:g}'):>7}"
                    f"{one(w.get('sum_ms'),'{:.3f}'):>13}"
                    f"{one(mib(e['peak_live_bytes']),'{:.1f}'):>10}"
                    f"{one(e.get('n_allocs_cold'),'{:g}'):>8}"
                    f"{'OK' if e['guards_ok'] else 'FAILED':>8}")
            if wr is not None:
                line += f"   walk vs ruler {wr:+.3f}%"
            print(line)
        xs = {x["rung"]: x for x in T[b].get("cross_check", []) if x.get("median_ms")}
        if xs:
            same = []
            for r in LADDER:
                if r in xs:
                    d = (xs[r]["median_ms"] - rows[b][r]["median_ms"]) / rows[b][r]["median_ms"] * 100
                    same.append(f"{r}:{d:+.1f}%")
            print(f"    cross-check via bench_render vs the fresh-graph loop: "
                  + "  ".join(same))
        wl = T[b].get("timed_workload")
        if isinstance(wl, dict) and wl:
            print("    workload: " + "  ".join(
                f"{r}={wl[r]['mean']:.4f}" for r in LADDER
                if isinstance(wl.get(r), dict) and "mean" in wl[r]))

    if not oc:
        return

    print()
    print("=" * 78)
    print(f"4. THE LADDER AT BATCH {a.batch}: TIME NEXT TO INSTRUCTIONS")
    print("=" * 78)
    print("  ops  = graph operations per pixel per DDA step  (opcount.json graph)")
    print("  sass = SASS instructions per pixel per DDA step (opcount.json sass)")
    print("  Both are the DIFFERENCE method: the same program compiled at five")
    print("  max_steps, slope against trip count. Not a static count read off one")
    print("  kernel. re-derived offline, not quoted from a note.")
    print()
    b = str(a.batch)
    g = (oc.get("graph") or {})
    s = (oc.get("sass") or {})
    print(f"  {'rung':<14}{'ms':>9}{'x vs above':>11}{'CALLs cold/warm':>17}"
          f"{'ops':>8}{'d ops':>8}{'sass':>8}{'d sass':>8}{'sass x':>8}"
          f"{'spill B':>10}")
    prev_ms = prev_ops = prev_sass = None
    for r in LADDER:
        e = rows.get(b, {}).get(r)
        ms = e["median_ms"] if e else None
        ops = (g.get(r) or {}).get("per_pixel_per_step")
        sass = (s.get(r) or {}).get("slope_hot_per_pixel_per_step")
        spill = (s.get(r) or {}).get("32", {}).get("local_bytes")
        spd = "" if (prev_ms is None or not ms) else f"{prev_ms/ms:.3f}"
        dops = "" if prev_ops is None else f"{ops - prev_ops:+.1f}"
        dsass = "" if prev_sass is None else f"{sass - prev_sass:+.1f}"
        sx = "" if (prev_sass is None or not sass) else f"{prev_sass/sass:.3f}"
        nk = "-" if not e else (f"{e.get('n_kernels_cold')}/{e.get('n_kernels_warm')}"
                                if e.get("n_kernels_cold") != e.get("n_kernels_warm")
                                else str(e.get('n_kernels_cold')))
        print(f"  {r:<14}{one(ms):>9}{spd:>11}{nk:>17}"
              f"{one(ops,'{:.2f}'):>8}{dops:>8}{one(sass,'{:.1f}'):>8}{dsass:>8}{sx:>8}"
              f"{one(spill,'{:g}'):>10}")
        prev_ms, prev_ops, prev_sass = (ms if ms else prev_ms), ops, sass
    print("  spill B is ptxas's own `local memory frame` for the fused kernel at")
    print("  max_steps 32 -- bytes each thread spills to local memory, which on a")
    print("  T4 is DRAM traffic, not a register.")
    if est:
        e0 = min(est, key=lambda k: est[k].get("bytes_estimate") or 0)
        e1 = max(est, key=lambda k: est[k].get("bytes_estimate") or 0)
        lo, hi = est[e0]["bytes_estimate"], est[e1]["bytes_estimate"]
        print(f"  NOT shown as a memory column: the census's own bytes_estimate at")
        print(f"  max_steps 32 ranges only over {lo/2**20:.1f}-{hi/2**20:.1f} MiB across")
        print(f"  the five rungs ({100*(hi-lo)/hi:.1f}% apart), because it is dominated by the")
        print(f"  world array every rung reads. It does not separate them.")

    refi = ((oc.get("reference") or {}).get("instructions_per_step"))
    print(f"  hand-written reference: {refi} SASS per pixel per DDA step")
    for r in LADDER:
        sass = (s.get(r) or {}).get("slope_hot_per_pixel_per_step")
        if sass and refi:
            print(f"    {r:<14}{sass:8.1f}  = {sass/refi:.3f}x the reference")

    print()
    print("=" * 78)
    print("5. WHERE THE WALL CLOCK BREAKS AWAY FROM THE INSTRUCTION COUNT")
    print("=" * 78)
    base_ms = rows.get(b, {}).get("naive", {}).get("median_ms")
    base_s = (s.get("naive") or {}).get("slope_hot_per_pixel_per_step")
    base_o = (g.get("naive") or {}).get("per_pixel_per_step")
    if not (base_ms and base_s):
        return
    print(f"  Walked down from `naive` at batch {b}. 'time left' is the fraction")
    print("  of the naive port's wall clock still remaining; 'sass left' and")
    print("  'ops left' are the same fraction of the instruction and operation")
    print("  counts still remaining. Where the time column falls faster than the")
    print("  other two, that rung is where the clock left the instruction count.")
    print()
    print(f"  {'at rung':<14}{'ms':>9}{'time left':>11}{'ops left':>10}"
          f"{'sass left':>11}{'launches':>10}{'peak MiB':>10}")
    for r in LADDER:
        e = rows.get(b, {}).get(r)
        if not e:
            continue
        ms = e["median_ms"]
        ops = (g.get(r) or {}).get("per_pixel_per_step")
        sass = (s.get(r) or {}).get("slope_hot_per_pixel_per_step")
        print(f"  {r:<14}{ms:>9.3f}{100*ms/base_ms:>10.1f}%"
              f"{one(ops,'{:.1f}'):>10}"
              f"{one(100*sass/base_s,'{:.1f}'):>10}"
              f"{one(e.get('n_kernels_cold'),'{:g}'):>10}"
              f"{one(mib(e['peak_live_bytes']),'{:.1f}'):>10}")

    print()
    print("  Each rewrite's own price, both currencies:")
    print(f"  {'rewrite':<34}{'d ms':>9}{'x':>8}{'d ops':>8}{'d sass':>9}"
          f"{'CALLs':>12}{'peak MiB':>16}")
    names = {
        "int_cursor": "1 int cursor",
        "flat_gather": "2 one flat index",
        "incremental": "3 maintained address",
        "lean": "4 no found flag",
    }
    for i, r in enumerate(LADDER):
        if i == 0:
            continue
        up = LADDER[i - 1]
        eu, ee = rows.get(b, {}).get(up), rows.get(b, {}).get(r)
        if not (eu and ee):
            continue
        ou = (g.get(up) or {}).get("per_pixel_per_step")
        on = (g.get(r) or {}).get("per_pixel_per_step")
        su = (s.get(up) or {}).get("slope_hot_per_pixel_per_step")
        sn = (s.get(r) or {}).get("slope_hot_per_pixel_per_step")
        pk = f"{mib(eu['peak_live_bytes']):.1f} -> {mib(ee['peak_live_bytes']):.1f}"
        print(f"  {names[r]:<34}{ee['median_ms']-eu['median_ms']:>+9.3f}"
              f"{eu['median_ms']/ee['median_ms']:>8.3f}"
              f"{on-ou:>+8.1f}{sn-su:>+9.1f}"
              f"{one(eu.get('n_kernels_cold'),'{:g}')+'->'+one(ee.get('n_kernels_cold'),'{:g}'):>12}"
              f"{pk:>16}")

    print()
    print("=" * 78)
    print("6. THE BATCH-SIZE TEST -- does time move the way work moves?")
    print("=" * 78)
    bs = sorted(rows, key=lambda x: int(x))
    for r in LADDER:
        have = [(b, rows[b][r]) for b in bs if r in rows[b]]
        if len(have) < 2:
            continue
        (b1, e1), (b2, e2) = have[0], have[-1]
        px1, px2 = T[b1].get("pixels"), T[b2].get("pixels")
        if px1 and px2:
            print(f"  {r:<13} pixels {px1:>9,} -> {px2:>9,} "
                  f"({px2/px1:.1f}x)   time {e1['median_ms']:.3f} -> {e2['median_ms']:.3f} ms "
                  f"({e2['median_ms']/e1['median_ms']:.3f}x)   "
                  f"CALLs {e1.get('n_kernels_cold')} -> {e2.get('n_kernels_cold')}")


if __name__ == "__main__":
    main()
