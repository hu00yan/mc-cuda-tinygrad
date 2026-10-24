#!/usr/bin/env python3
"""design38.py -- OFFLINE. Find the register budgets that make the frame move,
before a GPU session is spent on them.

THE POINT
---------
The experiment needs "same graph size, a much bigger frame". Holding the graph
fixed is free (the generated CUDA is a byte-identical string, sha256-checked);
making the frame *move a lot* is the part that has to be designed, because a
few percent of frame cannot falsify anything. So this sweeps `-maxrregcount`
over the PTX that `notes/36`'s run already captured, **on this host, with no
GPU**, and asks:

  * which budgets produce a frame an order of magnitude above the stock one,
  * whether the frame is monotone in the budget (a monotone lever gives a
    clean dose-response curve; a non-monotone one is a finding),
  * and what the SASS-instruction-free numbers say about how many of those
    budgets are *distinct* rather than aliased to the same allocation.

`ptxas` comes from the project's own `archlab/ptxas:12.9` image via docker. The
version differs from the card's 12.8, so **these numbers are a design
instrument, not a measurement** -- every number that reaches the note is
re-measured on the card in the same session that times it. What is carried
across is the *set of budgets worth putting in the notebook*, which is a
property of the lever and not of the compiler version.

    docker run --rm -v "$PWD:/w" -w /w archlab/ptxas:12.9 \\
        python3 /w/minecu_tg_opt/lm38k/design38.py --out /w/.agent-scratch/lm38b
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))

#: The stock point is `None` -- ptxas with no cap, which is what the framework's
#: own path does. The rest are candidate interventions.
BUDGETS = [None, 255, 240, 224, 208, 192, 176, 160, 144, 128, 112, 96, 80,
           64, 56, 48, 40, 32, 24]

PTXAS = "/usr/local/lib/python3.12/site-packages/nvidia/cuda_nvcc/bin/ptxas"

FRAME_RE = re.compile(
    r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads")
REG_RE = re.compile(r"Used (\d+) registers")


def run_ptxas(ptx: Path, mrr: int | None, outdir: Path) -> dict:
    cub = outdir / f"k_{mrr}.cubin"
    cmd = [PTXAS, "-arch=sm_75", "-v", "-O3"]
    if mrr is not None:
        cmd.append(f"-maxrregcount={mrr}")
    cmd += ["-o", str(cub), str(ptx)]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
    txt = (p.stderr or "") + (p.stdout or "")
    fr, rg = FRAME_RE.search(txt), REG_RE.search(txt)
    return {
        "budget": mrr,
        "rc": p.returncode,
        "stack_frame": int(fr.group(1)) if fr else None,
        "spill_stores": int(fr.group(2)) if fr else None,
        "spill_loads": int(fr.group(3)) if fr else None,
        "spill_bytes": (int(fr.group(2)) + int(fr.group(3))) if fr else None,
        "registers": int(rg.group(1)) if rg else None,
        "cubin_bytes": cub.stat().st_size if cub.exists() else None,
        "cmd": " ".join(cmd),
        "err_tail": txt[-300:] if p.returncode else "",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=REPO / ".agent-scratch" / "lm38b")
    ap.add_argument("--sass-dir",
                    default=str(REPO / ".agent-scratch" / "rungtimings" / "sass"
                                / "sass-b1024"))
    ap.add_argument("--rungs", default="flat_gather,naive,lean")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    sass = Path(a.sass_dir)

    say = lambda *x: print(*x, flush=True)
    say("=" * 78)
    say("  design38 -- which register budgets move the frame a lot?  (OFFLINE)")
    say("=" * 78)
    p = subprocess.run([PTXAS, "--version"], capture_output=True, text=True)
    say(f"  ptxas: {((p.stdout or p.stderr).strip().splitlines() or ['?'])[0]}")
    say(f"  NOTE: this is a DESIGN INSTRUMENT. The card runs 12.8; every number")
    say(f"  that reaches notes/38 is re-measured there, in the timing session.")
    say("")

    rep = {"ptxas_version": (p.stdout or p.stderr).strip(), "rungs": {}}
    for rung in a.rungs.split(","):
        stem = f"max_steps_0032__{rung}__kernel000"
        ptx = sass / f"{stem}.ptx"
        if not ptx.exists():
            say(f"  {rung}: no PTX at {ptx.name} -- skipped")
            continue
        say(f"--- {rung}  ({ptx.stat().st_size:,} B of PTX, "
            f"sha256 {hashlib.sha256(ptx.read_bytes()).hexdigest()[:16]}) ---")
        say(f"    {'budget':>7} {'frame B':>9} {'spill st':>9} {'spill ld':>9} "
            f"{'regs':>5} {'cubin B':>9}  {'frame/stock':>11}")
        rows = []
        stock = None
        with tempfile.TemporaryDirectory(dir=str(a.out)) as td:
            for mrr in BUDGETS:
                r = run_ptxas(ptx, mrr, Path(td))
                rows.append(r)
                if mrr is None:
                    stock = r["stack_frame"]
                ratio = (f"{r['stack_frame']/stock:.2f}x"
                         if stock and r["stack_frame"] is not None else "-")
                say(f"    {str(mrr):>7} {str(r['stack_frame']):>9} "
                    f"{str(r['spill_stores']):>9} {str(r['spill_loads']):>9} "
                    f"{str(r['registers']):>5} {str(r['cubin_bytes']):>9}  {ratio:>11}"
                    + ("   *** ptxas FAILED ***" if r["rc"] else ""))
        rep["rungs"][rung] = {"ptx_sha256": hashlib.sha256(ptx.read_bytes()).hexdigest(),
                              "ptx_bytes": ptx.stat().st_size, "rows": rows}
        ok = [r for r in rows if r["rc"] == 0 and r["stack_frame"] is not None]
        frames = [r["stack_frame"] for r in ok]
        uniq = sorted(set(frames))
        say(f"    {len(ok)}/{len(rows)} budgets compiled; "
            f"{len(uniq)} DISTINCT frame values: {uniq}")
        if stock and ok:
            big = [r["budget"] for r in ok if r["stack_frame"] >= 4 * stock]
            say(f"    budgets giving >= 4x the stock frame ({stock} B): {big}")
        say("")

    (a.out / "design38.json").write_text(json.dumps(rep, indent=1, default=str))
    say(f"wrote {a.out / 'design38.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
