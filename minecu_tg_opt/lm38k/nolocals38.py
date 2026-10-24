#!/usr/bin/env python3
"""nolocals38.py -- what does the ONE valid same-graph-size point actually say?

WHY THIS EXISTS
---------------
`out/lm38/20260930-233053` produced one configuration in which the framework's
own register knob was turned and everything else was held fixed:

    flat_gather, NOLOCALS=1, batch 1024
      9 CALLs, byte-identical output (gate_bad = 0 over 1 048 576 pixels)
      57.2876 ms

against the same source at stock:

      9 CALLs, 3.0151 ms

Same launch count, same observation, same shape, same session, same card. That
is the discriminating case `notes/35` section 5 does not have: graph size is
equal by construction, and the register allocation is not.

This script establishes the other half -- **how far the allocation actually
moved** -- by compiling the same generated CUDA twice, once as the framework
emits it and once with `__launch_bounds__` changed, and reading ptxas's own
report on both. That is the part of the case that decides whether it is a
*large* frame difference or a small one, and "a few percent cannot falsify
anything" applies to it exactly as it would to any other cell.

It also reports the two currencies separately, because §5 names one and means
the other (see `currencies38.py`).

OFFLINE. ptxas is x86-64 Linux and runs in the project's own image; the
generated CUDA is captured on the NULL device, which runs the real scheduler,
the real fusion decision and the real C renderer with nothing executed.

    bash  minecu_tg_opt/lm38k/nolocals38.sh          # capture + compile
    .venv/bin/python minecu_tg_opt/lm38k/nolocals38.py   # read the report
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
REPORT = REPO / ".agent-scratch" / "lm38b" / "nolocals38.txt"

FRAME = re.compile(r"(\d+) bytes stack frame, (\d+) bytes spill stores, "
                   r"(\d+) bytes spill loads")
USED = re.compile(r"Used (\d+) registers")


def parse(path: Path) -> dict:
    if not path.exists():
        return {}
    out, cur = {}, None
    for line in path.read_text().splitlines():
        if line.startswith("### "):
            cur = line[4:].split(" ###")[0].strip()
            out[cur] = {}
            continue
        if cur is None:
            continue
        m = FRAME.search(line)
        if m:
            sf, ss, sl = (int(m.group(i)) for i in (1, 2, 3))
            out[cur].update(stack_frame=sf, spill_stores=ss, spill_loads=sl,
                            spill_total=ss + sl)
        m = USED.search(line)
        if m:
            out[cur]["registers"] = int(m.group(1))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", type=Path, default=REPORT)
    a = ap.parse_args()
    say = lambda *x: print(*x, flush=True)
    d = parse(a.report)
    if not d:
        say(f"no report at {a.report}; run nolocals38.sh first")
        return 2

    say("=" * 92)
    say("  nolocals38 -- the allocation behind the one valid same-graph-size point")
    say("=" * 92)
    say(f"  source of the numbers: {a.report}")
    say(f"  ptxas: {'see the header written by nolocals38.sh'}")
    say("")
    say(f"  {'configuration':<28} {'stack frame':>12} {'spill st':>9} {'spill ld':>9} "
        f"{'st+ld':>9} {'regs':>5} {'launch_bounds':>15}")
    for k, v in d.items():
        say(f"  {k:<28} {v.get('stack_frame', 0):>12,} "
            f"{v.get('spill_stores', 0):>9,} {v.get('spill_loads', 0):>9,} "
            f"{v.get('spill_total', 0):>9,} {v.get('registers', 0):>5} "
            f"{str(v.get('launch_bounds', '?')):>15}")

    base = d.get("flat_gather_stock") or d.get("stock")
    var = d.get("flat_gather_nolocals") or d.get("nolocals")
    if base and var:
        say("")
        say("  the movement, which is what makes this a case or not a case:")
        for field, label in (("stack_frame", "stack frame"),
                             ("spill_stores", "spill stores"),
                             ("spill_loads", "spill loads"),
                             ("spill_total", "spill stores + loads"),
                             ("registers", "registers")):
            b, v = base.get(field), var.get(field)
            if b and v:
                say(f"    {label:<22} {b:>8,} -> {v:>8,}   {v/b:>7.3f}x")
        say("")
        say("  `__launch_bounds__(1)` gives ptxas a ceiling of 65 536 registers per")
        say("  thread, which is far above the hardware maximum of 255, so it cannot")
        say("  lower the allocation. If the frame is unchanged, then NOLOCALS is")
        say("  NOT a same-graph-size/different-frame case, and the 20x it costs on")
        say("  the card is something else -- which is a finding, and not the one the")
        say("  note was looking for.")
    print()
    say(f"wrote {a.report.with_suffix('.json')}")
    a.report.with_suffix(".json").write_text(json.dumps(d, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
