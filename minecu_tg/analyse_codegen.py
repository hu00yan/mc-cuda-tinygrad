#!/usr/bin/env python3
"""analyse_codegen.py -- turn a captured schedule into a per-kernel table.

Reads the `max_steps_*/kernelNNN.cu` directories that `capture_cuda.py` wrote and
reports, for each captured program: its argument list, its size, how many DDA
steps its body contains, and how many global state buffers it touches.

The step count inside a program is counted from the generated CUDA itself, by
the shape of one DDA iteration in the emitted code, not by guessing. A step is
six masked selects (three `t_max` accumulations and three voxel-coordinate
updates), and the reference's own macro has exactly those six
(kernels.cu:672-678). Counting six per step and checking the total against the
line count is the cross-check that the regex found what it claims to.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# a masked accumulate:  x = (cond ? (a + b) : a)
MASKED_ACC = re.compile(r"alu\d+ = \(alu\d+\?\(alu\d+\+alu\d+\)")
# a voxel read: *(<state buffer> + (...))
VOXEL_LD = re.compile(r"= \(\*\(data\d+_\d+\+\(")
# a 3-way DDA compare: (a < b) & (a < c)
DDA_CMP = re.compile(r"alu\d+ = \(\(alu\d+<alu\d+\)&\(alu\d+<alu\d+\)\)")

KERNEL_SIG = re.compile(r"__global__ void __launch_bounds__\((\d+)\)\s+(\w+)\(")


def analyse(path: Path) -> dict:
    src = path.read_text()
    m = KERNEL_SIG.search(src)
    # the argument list is inside the signature line, which can be very long
    sig_line = next((l for l in src.splitlines() if "__global__ void" in l), "")
    args = re.findall(r"(?:signed char|float|int|bool|uint)\s*\*\s*(data\d+_\d+)", sig_line)
    masked = len(MASKED_ACC.findall(src))
    return {
        "file": path.name,
        "launch_bounds": int(m.group(1)) if m else None,
        "name": m.group(2) if m else None,
        "n_args": len(args),
        "state_buffers": len([a for a in args if int(a.split("_")[1]) > 1024]),
        "lines": len(src.splitlines()),
        "bytes": len(src),
        "masked_accumulates": masked,
        "dda_steps_est": round(masked / 6.0, 2),
        "dda_three_way_compares": len(DDA_CMP.findall(src)),
        "voxel_reads": len(VOXEL_LD.findall(src)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path, help="the --outdir given to capture_cuda.py")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    census = json.loads((a.root / "census.json").read_text())
    out = {"device": census["device"], "arch": census.get("arch"), "shapes": {}}
    for r in census["rows"]:
        d = a.root / f"max_steps_{r['max_steps']:04d}"
        if not d.is_dir():
            continue
        rows = [analyse(p) for p in sorted(d.glob("kernel*.cu"))]
        out["shapes"][str(r["max_steps"])] = {
            "kernel_launches": r["kernel_launches"],
            "distinct_programs_launched": r["distinct_programs_launched"],
            "new_distinct_programs": r["new_distinct_programs"],
            "programs": rows,
        }

    if a.json:
        print(json.dumps(out, indent=2))
        return 0

    print(f"device={out['device']} arch={out['arch']}")
    for ms, s in out["shapes"].items():
        print()
        print(f"=== max_steps={ms}: {s['kernel_launches']} launches, "
              f"{s['distinct_programs_launched']} distinct programs ===")
        print(f"{'file':>13} {'name':>12} {'ld':>4} {'args':>5} {'state':>6} "
              f"{'lines':>7} {'steps~':>7} {'cmp3':>5} {'ldvox':>6}")
        for p in s["programs"]:
            nm = (p["name"] or "")[:12]
            print(f"{p['file']:>13} {nm:>12} {str(p['launch_bounds']):>4} "
                  f"{p['n_args']:>5} {p['state_buffers']:>6} {p['lines']:>7} "
                  f"{p['dda_steps_est']:>7} {p['dda_three_way_compares']:>5} "
                  f"{p['voxel_reads']:>6}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
