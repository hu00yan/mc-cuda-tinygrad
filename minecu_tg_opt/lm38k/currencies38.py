#!/usr/bin/env python3
"""currencies38.py -- which quantity does `notes/35` section 5 actually name,
and are the two candidates the same number?  (OFFLINE, no GPU)

`probe_maxntid.py` found the real PTX's resource report reads

    1312 bytes stack frame, 1772 bytes spill stores, 1296 bytes spill loads

and 1772 + 1296 = 3 068, which is EXACTLY the figure `notes/35` section 5 and
`notes/36` section 5 quote for `flat_gather`. So the number those sections call
"ptxas's local-memory frame" is not the stack frame. It is the sum of the spill
store and spill load byte counts.

That is worth separating, because the two are physically different currencies:

  * **stack frame** -- a *capacity*, bytes of local memory the thread reserves.
    It is allocated whether or not it is touched, so it is a weak proxy for cost.
  * **spill stores + spill loads** -- a *count of the traffic* ptxas scheduled.
    This is the quantity with a mechanism: extra loads and stores inside the
    fused region.

This script reads BOTH for all five rungs from the PTX `notes/36`'s run
captured, so the note can say which one the ladder's correlation is a
correlation of -- and whether the two rank the rungs the same way.

It also checks the arithmetic claim directly: does any rung's published figure
equal spill_stores + spill_loads, and does it differ from the stack frame?

These are OFFLINE numbers from CUDA 12.9 on captured PTX. They are `PREDICTED`
here and re-measured on the card in the timing session; nothing here is quoted
as a measurement.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
PTXAS = "/usr/local/lib/python3.12/site-packages/nvidia/cuda_nvcc/bin/ptxas"
FRAME_RE = re.compile(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads")
REG_RE = re.compile(r"Used (\d+) registers")
RUNGS = ["naive", "int_cursor", "flat_gather", "incremental", "lean"]
#: What notes/36 section 5 published, batch 1024, max_steps 32. Used ONLY to
#: check the arithmetic identity; every number below is re-derived here.
PUBLISHED = {"naive": 20932, "int_cursor": 13216, "flat_gather": 3068,
             "incremental": 2984, "lean": 3572}
say = lambda *x: print(*x, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=REPO / ".agent-scratch" / "lm38b")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    d = REPO / ".agent-scratch" / "rungtimings" / "sass" / "sass-b1024"
    say("=" * 92)
    say("  currencies38 -- stack frame vs spill stores+loads, all five rungs (OFFLINE, 12.9)")
    say("=" * 92)
    say(f"  {'rung':<14} {'stack frame':>12} {'spill st':>9} {'spill ld':>9} "
        f"{'st+ld':>8} {'regs':>5}  {'published':>10} {'st+ld==pub':>11}")
    rows = []
    for r in RUNGS:
        ptx = d / f"max_steps_0032__{r}__kernel000.ptx"
        if not ptx.exists():
            say(f"  {r:<14} no PTX")
            continue
        with tempfile.TemporaryDirectory(dir=str(a.out)) as td:
            c = Path(td) / "k.cubin"
            p = subprocess.run([PTXAS, "-arch=sm_75", "-v", "-O3", "-o", str(c), str(ptx)],
                               capture_output=True, text=True, timeout=1800)
            txt = (p.stderr or "") + (p.stdout or "")
            fr, rg = FRAME_RE.search(txt), REG_RE.search(txt)
            if not fr:
                say(f"  {r:<14} ptxas rc={p.returncode}: {txt[-200:]}")
                continue
            sf, ss, sl = int(fr.group(1)), int(fr.group(2)), int(fr.group(3))
            rows.append({"rung": r, "stack_frame": sf, "spill_stores": ss,
                         "spill_loads": sl, "spill_total": ss + sl,
                         "registers": int(rg.group(1)) if rg else None,
                         "published": PUBLISHED.get(r),
                         "ptx_sha256": hashlib.sha256(ptx.read_bytes()).hexdigest()})
            say(f"  {r:<14} {sf:>12,} {ss:>9,} {sl:>9,} {ss+sl:>8,} "
                f"{(int(rg.group(1)) if rg else 0):>5}  {PUBLISHED.get(r, 0):>10,} "
                f"{str(ss + sl == PUBLISHED.get(r)):>11}")
    if len(rows) == 5:
        by_frame = [r["rung"] for r in sorted(rows, key=lambda r: r["stack_frame"])]
        by_spill = [r["rung"] for r in sorted(rows, key=lambda r: r["spill_total"])]
        by_pub = [r["rung"] for r in sorted(rows, key=lambda r: r["published"])]
        say("")
        say(f"  ranked by STACK FRAME : {by_frame}")
        say(f"  ranked by SPILL st+ld  : {by_spill}")
        say(f"  ranked by PUBLISHED    : {by_pub}")
        say(f"  the two rank identically: {by_frame == by_spill}")
        say(f"  published == spill st+ld for all five: "
            f"{all(r['spill_total'] == r['published'] for r in rows)}")
        say(f"  published == stack frame for all five: "
            f"{all(r['stack_frame'] == r['published'] for r in rows)}")
    (a.out / "currencies38.json").write_text(json.dumps(
        {"ptxas": "12.9 offline", "rows": rows}, indent=1, default=str))
    say("")
    say(f"wrote {a.out / 'currencies38.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
