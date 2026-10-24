#!/usr/bin/env python3
"""probe_maxntid.py -- is `.maxntid` what makes -maxrregcount inert on the REAL
workload PTX?  (OFFLINE, no GPU)

`design38.py` swept nineteen `-maxrregcount` values over the PTX that
`notes/36`'s own run captured from the framework's fused kernel and got ONE
cubin every time: 255 registers, a 1 312-byte frame, 481 960 bytes of code.
`why38.py` then showed that on a synthetic body the same flag moves the
allocation freely when `.maxntid` is absent.

Those two facts together make a specific, testable claim about the real
workload: **the register-budget lever is inert on this workload because the
framework's PTX carries `.maxntid 128, 1, 1`, and ptxas resolves the register
ceiling from the block size instead of from the command line.**

This script tests exactly that claim on the real artifact, by transforming one
captured PTX four ways and sweeping the budget over each:

    as captured                    (byte-identical to what the card compiles)
    .maxntid deleted               (same code, no block-size constraint)
    .maxntid -> 1024               (block size raised)
    .maxntid -> 32                 (block size lowered)

If deleting `.maxntid` restores a live lever, the claim is confirmed and the
on-card experiment has a lever to use. If it does not, the lever is unavailable
on this workload for some other reason and `notes/38` has to say so instead of
carrying a notebook built on it.

The source string tinygrad emits is NOT modified by this: this is a diagnostic
on a captured PTX file, and it establishes whether an experiment is possible
before one is written. Every number is CUDA 12.9, offline.
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
BUDGETS = [None, 240, 200, 168, 128, 96, 64, 48, 32, 24]

say = lambda *x: print(*x, flush=True)


def variants(text: str) -> dict[str, str]:
    def sub(pat, rep):
        return re.sub(pat, rep, text, count=1, flags=re.M)
    return {
        "as_captured": text,
        "maxntid_deleted": sub(r"^\.maxntid .*$", ""),
        "maxntid_1024": sub(r"^\.maxntid .*$", ".maxntid 1024, 1, 1"),
        "maxntid_32": sub(r"^\.maxntid .*$", ".maxntid 32, 1, 1"),
    }


def one(ptx_text: str, mrr, td: Path) -> dict:
    p, c = Path(td) / f"k_{mrr}.ptx", Path(td) / f"k_{mrr}.cubin"
    p.write_text(ptx_text)
    cmd = [PTXAS, "-arch=sm_75", "-v", "-O3"]
    if mrr is not None:
        cmd.append(f"-maxrregcount={mrr}")
    cmd += ["-o", str(c), str(p)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    txt = (r.stderr or "") + (r.stdout or "")
    fr, rg = FRAME_RE.search(txt), REG_RE.search(txt)
    secs = re.search(r"Compile time = ([0-9.]+) ms", txt)
    return {
        "budget": mrr, "rc": r.returncode,
        "stack_frame": int(fr.group(1)) if fr else None,
        "spill_stores": int(fr.group(2)) if fr else None,
        "spill_loads": int(fr.group(3)) if fr else None,
        "registers": int(rg.group(1)) if rg else None,
        "cubin_sha": hashlib.sha256(c.read_bytes()).hexdigest()[:16] if c.exists() else None,
        "cubin_bytes": c.stat().st_size if c.exists() else None,
        "ptxas_ms": float(secs.group(1)) if secs else None,
        "err": txt[-200:] if r.returncode else "",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=REPO / ".agent-scratch" / "lm38b")
    ap.add_argument("--rung", default="flat_gather")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    ptx = (REPO / ".agent-scratch" / "rungtimings" / "sass" / "sass-b1024" /
           f"max_steps_0032__{a.rung}__kernel000.ptx")
    if not ptx.exists():
        say(f"no PTX at {ptx}")
        return 2
    text = ptx.read_text()
    say("=" * 78)
    say(f"  probe_maxntid -- is .maxntid the reason the lever is inert?  (OFFLINE)")
    say("=" * 78)
    say(f"  {ptx.name}  {len(text):,} B  sha256 {hashlib.sha256(text.encode()).hexdigest()[:16]}")
    m = re.search(r"^\.maxntid .*$", text, re.M)
    say(f"  .maxntid: {m.group(0) if m else '(absent)'}")
    say("")
    rep = {"rung": a.rung, "ptx_sha256": hashlib.sha256(text.encode()).hexdigest(),
           "variants": {}}
    for name, vtext in variants(text).items():
        say(f"--- {name} ---")
        say(f"    {'budget':>7} {'frame B':>9} {'spill st':>9} {'spill ld':>9} "
            f"{'regs':>5} {'cubin B':>9} {'ptxas s':>8}  cubin sha")
        rows = []
        with tempfile.TemporaryDirectory(dir=str(a.out)) as td:
            for mrr in BUDGETS:
                r = one(vtext, mrr, Path(td))
                rows.append(r)
                say(f"    {str(mrr):>7} {str(r['stack_frame']):>9} "
                    f"{str(r['spill_stores']):>9} {str(r['spill_loads']):>9} "
                    f"{str(r['registers']):>5} {str(r['cubin_bytes']):>9} "
                    f"{(r['ptxas_ms'] or 0)/1000:8.1f}  {r['cubin_sha']}")
                if r["err"]:
                    say(f"        !! {r['err'][-160:]}")
        rep["variants"][name] = rows
        ok = [r for r in rows if r["rc"] == 0 and r["stack_frame"] is not None]
        say(f"    -> {len(ok)}/{len(rows)} ok; "
            f"{len({r['stack_frame'] for r in ok})} distinct frames, "
            f"{len({r['cubin_sha'] for r in ok})} distinct cubins, "
            f"registers {sorted({r['registers'] for r in ok})}")
        if ok:
            f0 = ok[0]["stack_frame"]
            big = [r["budget"] for r in ok if r["stack_frame"] and f0 and r["stack_frame"] >= 4 * f0]
            say(f"    -> budgets with >=4x the first frame ({f0} B): {big}")
        say("")
    (a.out / "probe_maxntid.json").write_text(json.dumps(rep, indent=1, default=str))
    say(f"wrote {a.out / 'probe_maxntid.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
