#!/usr/bin/env python3
"""Render out/ladder.tsv as a compact arch-ladder: which arches accept each witness."""
import csv
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
rows = list(csv.DictReader((ROOT / "out" / "ladder.tsv").open()))
ARCHS = []
for r in rows:
    if r["arch"] not in ARCHS:
        ARCHS.append(r["arch"])


def key(a):
    return int(a.split("_")[1].rstrip("af"))


ARCHS.sort(key=key)
grid, reasons = {}, {}
for r in rows:
    w = r["witness"].split("@")[0]
    grid.setdefault(r["witness"], {})[r["arch"]] = r["status"]
    if r["status"] == "REJECT" and r["detail"]:
        reasons.setdefault(r["detail"], []).append(f"{r['witness'].split('@')[0]}@{r['arch']}")

hdr = "witness".ljust(34) + "".join(a.replace("sm_", "").rjust(6) for a in ARCHS)
print(hdr)
print("-" * len(hdr))
for w in sorted(grid, key=lambda x: (min([key(a) for a in grid[x]
                                          if grid[x][a] == "OK"] or [999]), x)):
    row = grid[w]
    cells = ""
    for a in ARCHS:
        s = row.get(a, "?")
        cells += ("   .  " if s == "REJECT" else "   Y  ").rjust(6)
    print(w.ljust(34) + cells)

print("\n\n=== rejection reasons (deduplicated) ===")
seen = {}
for d, who in reasons.items():
    seen.setdefault(d, []).extend(who)
for d in sorted(seen, key=lambda k: -len(seen[k])):
    print(f"\n[{len(seen[d])} cells] {d}")
    print("   e.g. " + ", ".join(sorted(set(seen[d]))[:6]))
