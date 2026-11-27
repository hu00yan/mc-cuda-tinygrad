#!/usr/bin/env python3
"""analyse_run.py -- the render-versus-render table, from a completed run's
own artefacts.

WHY THIS EXISTS RATHER THAN A RE-RUN. The first completed T4 run
(`out/minecu/opt/20260929-151936`) printed its ratio column as
`ref_microseconds / tg_milliseconds`. That is a mixed-unit quotient: it reports
6.43 where the true slowdown is 155x, and it passes through 1.0 exactly when
the two sides are equal, which is the one value a reader cannot mistake for a
mistake. The measurements themselves are unaffected -- they are two device-time
numbers from two instruments in the same run -- so the table is recomputed here
from the run's own `ref_timing.json` and `tgopt/report.json` rather than by
spending another GPU session re-measuring a number that was already measured.

`make_notebook.py` carries the unit fix for future runs; this module is the
correction for the run that already happened, and it reads only that run's files.

    python3 minecu/opt/analyse_run.py out/minecu/opt/20260929-151936
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def ratio_us(ref_us, tg_ms):
    """How many times slower `tg_ms` milliseconds is than `ref_us` microseconds.

    One unit, explicitly. The naive form of this expression is
    `ref_us / tg_ms`, which is wrong by 1000x AND inverted, so it reports a
    slowdown below 1.0 for a kernel that is two orders of magnitude slower.
    """
    if not (ref_us and tg_ms):
        return None
    return (float(tg_ms) * 1000.0) / float(ref_us)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path, help="the downloaded run directory")
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args()

    ref = json.loads((a.run / "ref_timing.json").read_text())
    rep = json.loads((a.run / "tgopt" / "report.json").read_text())
    equiv = json.loads((a.run / "obs_equivalence.json").read_text())

    rows: dict[int, dict] = {}
    for v in rep.get("timing") or []:
        b, name = v.get("batch"), v.get("variant")
        if not b or not name:
            continue
        rows.setdefault(b, {})[name] = v

    # `ref_timing.json` is keyed by JSON, so its batch keys are STRINGS, while
    # `report.json`'s are ints. Merging the two key spaces and sorting them
    # together is a TypeError; normalising to int first is not.
    ref = {int(k): v for k, v in ref.items()}
    rows = {int(k): v for k, v in rows.items()}

    out = {"run": str(a.run), "batches": {}}
    print("=" * 78)
    print("  render versus render, equal batch, same card, same run, same world")
    print("  max_steps 32, uint8 out.  ref in us (cudaEvent), tg in ms (device)")
    print("=" * 78)
    print(f"  {'batch':>7} {'ref render us':>14} {'ref step us':>13} "
          f"{'naive ms':>10} {'lean ms':>10} {'naive/ref':>10} {'lean/ref':>9} "
          f"{'kernels n/l':>12}")
    for b in sorted(set(list(ref) + list(rows))):
        r = ref.get(b) or {}
        rr = (r.get("render_us") or [None])[0]
        rs = (r.get("step_us") or [None])[0] if r.get("step_us") else None
        nv, lnv = rows.get(b, {}).get("naive", {}), rows.get(b, {}).get("lean", {})
        dm_n = (nv.get("device_time") or {}).get("median_ms")
        dm_l = (lnv.get("device_time") or {}).get("median_ms")
        kn = (nv.get("device_time") or {}).get("n_kernels")
        kl = (lnv.get("device_time") or {}).get("n_kernels")
        q = ratio_us(rr, dm_n)
        p = ratio_us(rr, dm_l)
        rec = {"ref_render_us": rr, "ref_step_us": rs,
               "naive_ms": dm_n, "lean_ms": dm_l,
               "naive_over_ref": q, "lean_over_ref": p,
               "kernels_naive": kn, "kernels_lean": kl,
               "peak_mib_naive": ((nv.get("mem") or {}).get("peak_live_bytes") or 0) / 2**20,
               "peak_mib_lean": ((lnv.get("mem") or {}).get("peak_live_bytes") or 0) / 2**20,
               "allocs_naive": (nv.get("mem") or {}).get("n_allocs"),
               "allocs_lean": (lnv.get("mem") or {}).get("n_allocs")}
        if q and p:
            rec["time_removed_fraction"] = 1.0 - dm_n / dm_l
            rec["speedup_lean_over_naive"] = dm_n / dm_l
        out["batches"][str(b)] = rec
        print(f"  {b:>7} {round(rr,1) if rr else '-':>14} "
              f"{round(rs,1) if rs else '-':>13} "
              f"{(round(dm_n,3) if dm_n else '-'):>10} "
              f"{(round(dm_l,3) if dm_l else '-'):>10} "
              f"{(f'{q:10.2f}' if q else f'{chr(45):>10}')} "
              f"{(f'{p:9.2f}' if p else f'{chr(45):>9}')} "
              f"{f'{kn}/{kl}':>12}")

    print()
    print("  the optimisation, per batch:")
    for b, r in out["batches"].items():
        if "speedup_lean_over_naive" not in r:
            continue
        s = r["speedup_lean_over_naive"]
        verdict = ("faster" if s > 1.0 else "SLOWER")
        print(f"    batch {b:>5}: {s:6.3f}x the naive port's time ({verdict}); "
              f"the gap against the reference goes "
              f"{r['naive_over_ref']:.1f}x -> {r['lean_over_ref']:.1f}x")
        print(f"              peak memory {r['peak_mib_naive']:.1f} -> "
              f"{r['peak_mib_lean']:.1f} MiB, allocations "
              f"{r['allocs_naive']} -> {r['allocs_lean']}")

    print()
    print("  identity on the state the hand-written side itself produced:")
    for r in equiv.get("rows", []):
        print(f"    {r['variant']:<10} {r['pixels_differing']:>8} / "
              f"{equiv['pixels']:>8} pixels differ  "
              f"{'BYTE-IDENTICAL' if r['identical'] else 'DIFFERS'}")
    out["identity"] = equiv

    print()
    print("  the DDA workload of the timed runs:")
    for v in (rep.get("timing") or []):
        if v.get("dda"):
            d = v["dda"]
            print(f"    batch {v['batch']:>5} {v['variant']:<8} mean {d['mean']:6.3f}"
                  f"  at the 32 cap {100*d['at_cap_frac']:5.1f}%"
                  f"  first voxel {100*d['first_voxel_frac']:4.1f}%"
                  f"  gate {'PASS' if v.get('gate', {}).get('pass') else 'FAIL'}")
    out["report_config"] = rep.get("config")
    out["report_env"] = rep.get("env")

    if a.json:
        a.json.write_text(json.dumps(out, indent=2, default=str))
        print(f"\n  wrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
